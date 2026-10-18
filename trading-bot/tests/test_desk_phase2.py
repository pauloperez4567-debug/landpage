"""Tests end-to-end de la fase 2 en la mesa: backtest por comando, playbook, operativa automática por gráfica,
alarmas y avisos, Telegram con sesión falsa, ``status.json`` ampliado y resistencia del bucle a los trabajos de
backtest. Sin red: ``SimMarketData``/``FakeMarketData``, ``PaperBroker`` y ``SimClock``.

Los backtests usan pocas combinaciones y tramos para que la suite siga siendo rápida; las cifras que salen son
las que salen (sobre paseos aleatorios, lo normal es que nada apruebe fuera de muestra).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from desk.clock import SimClock
from desk.commands import CommandParser
from desk.config import load_settings
from desk.desk import TradingDesk
from desk.exchange.paper import PaperBroker
from desk.exchange.sim import DEFAULT_ANCHOR_TS, SimMarketData
from desk.jobs import BacktestOutcome, BacktestWorker
from desk.models import Command, CommandAction
from desk.optimizer import ReportStore
from desk.strategy import create_strategy
from desk.tui import load_snapshot, normalize_status

from synthetic import trend_bars

try:
    from matplotlib.figure import Figure as _Figure
    HAS_MPL = _Figure is not None
except ImportError:  # pragma: no cover - depende del entorno
    HAS_MPL = False

T0 = DEFAULT_ANCHOR_TS
TOKEN = "123456:TEST-token-secreto"
CHAT = 555


class Out(list):
    def __call__(self, text: str) -> None:
        self.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self)


def fast_settings(tmp_path: Path, **kw):
    """Configuración de test: backtests pequeños (pocas combinaciones y tramos) y sin estrategias por defecto."""
    base = dict(state_dir=str(tmp_path / "state"), llm_enabled=False, bt_max_combos=4, bt_folds=2,
                auto_default_strategies=False, symbols=["SOL"])
    base.update(kw)
    return load_settings(env={}, dotenv_path=None, **base)


def sim_desk(settings, clock: Optional[SimClock] = None, **kw):
    clock = clock or SimClock(T0)
    market = SimMarketData(clock)
    out = kw.pop("out", None) or Out()
    broker = PaperBroker(market, settings, clock, settings.state_path / "paper_broker.json")
    desk = TradingDesk(settings, market, broker, clock, out=out, **kw)
    return desk, market, clock, out


def alert_records(settings) -> list[dict]:
    p = settings.state_path / "alerts.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def outbox(settings) -> str:
    p = settings.state_path / "outbox.log"
    return p.read_text(encoding="utf-8") if p.exists() else ""


# --------------------------------------------------------------------------- backtest -> informe -> activa -> auto
def test_backtest_command_report_adopt_and_automatic_trade_with_sim_market(tmp_path):
    settings = fast_settings(tmp_path)
    desk, market, clock, out = sim_desk(settings)
    reply = desk.submit("backtest SOL 1h 40d")
    assert reply.startswith("Backtest SOL 1h (40 días, 960 velas, 5 estrategias) en marcha")
    assert "FUERA de muestra" in reply and "no garantiza" in reply
    assert desk.reports.latest("SOL", "1h") is None            # con reloj simulado corre en la etapa del ciclo
    assert desk.run_cycle() is True

    rep = desk.reports.latest("SOL", "1h")
    assert rep is not None and rep.bars == 960 and len(rep.evaluations) == 5
    reports = settings.state_path / "reports"
    assert (reports / f"SOL_1h_{rep.id}.json").exists() and (reports / f"SOL_1h_{rep.id}.html").exists()
    assert (reports / f"SOL_1h_{rep.id}.png").exists() is HAS_MPL
    # Aviso al terminar: resumen (dentro vs fuera de muestra), rutas y advertencia de honestidad.
    assert "FUERA de muestra (lo que cuenta)" in out.text and f"HTML {reports}" in out.text
    assert "un backtest no garantiza resultados futuros" in out.text
    assert f"Informe {rep.id}." in outbox(settings)
    [done] = [r for r in alert_records(settings) if r["kind"] == "backtest"]
    assert done["coin"] == "SOL" and done["level"] == "info"

    # Sin ganadora aprobada no se activa nada (salvo "forzar", con advertencia).
    evaluable = [e for e in rep.evaluations if not e.error]
    if rep.winner is None:
        reply = desk.submit("activa SOL 1h")
        assert "no tiene ganadora" in reply and "No se activa" in reply and "forzar" in reply
        assert desk.playbook.entries() == []
        pick = max(evaluable, key=lambda e: e.oos_metrics.get("n_trades", 0)).strategy
        reply = desk.submit(f"activa SOL 1h {pick} forzar")
        assert reply.startswith("ATENCIÓN — FORZADO") and "NO aprobó fuera de muestra" in reply
    else:                                                       # pragma: no cover - depende del camino simulado
        pick = rep.winner
        reply = desk.submit("activa SOL 1h")
        assert "FORZADO" not in reply
    assert f"Señales 'auto:{pick}@1h'" in reply
    [entry] = desk.playbook.entries()
    assert (entry.coin, entry.interval, entry.strategy, entry.report_id) == ("SOL", "1h", pick, rep.id)
    assert "SOL 1h" in desk.submit("playbook") and rep.id in desk.submit("informes")

    # La estrategia adoptada opera sola, con su temporalidad y sus parámetros, protegida con SL/TP.
    opened = None
    for _ in range(24 * 20):
        clock.advance(3600)
        desk.run_cycle()
        if "SOL" in desk.engine.brackets:
            opened = desk.engine.brackets["SOL"]
            break
    assert opened is not None, "la estrategia adoptada no abrió ninguna operación en 20 días simulados"
    assert opened.origin == f"auto:{pick}@1h"
    assert sorted(o.tag for o in desk.broker.open_orders("SOL")) == ["sl", "tp"]
    assert f"AUTO [{pick}] SOL" in out.text and "gráfica SOL 1h (playbook)" in out.text
    desk.write_status_snapshot()
    status = json.loads((settings.state_path / "status.json").read_text(encoding="utf-8"))
    assert status["signals"][-1]["strategy"] == f"{pick}@1h" and status["signals"][-1]["source"] == "playbook"
    assert status["positions"][0]["origin"] == f"auto:{pick}@1h"


def trend_market(tmp_path, history=2400):
    """``FakeMarketData`` con velas de 1h en tendencias alternas (ventaja real para un seguidor de tendencia):
    las ``history`` primeras ya cerradas a las ``T0 + 30 min``."""
    from conftest import FakeMarketData
    bars = trend_bars(seed=1)
    bars.index = pd.to_datetime([T0 + 3600 * (i - history) for i in range(len(bars))], unit="s", utc=True)
    clock = SimClock(T0 + 1800)
    market = FakeMarketData(clock)
    market.set_candles("BTC", bars.iloc[:history + 1], "1h")
    market.set_price("BTC", float(bars["close"].iloc[history - 1]))
    return bars, market, clock


def test_approved_winner_is_adopted_without_force_and_trades_its_chart(tmp_path):
    settings = fast_settings(tmp_path, bt_max_combos=12, bt_folds=4, symbols=["BTC"])
    bars, market, clock = trend_market(tmp_path)
    out = Out()
    desk = TradingDesk(settings, market, PaperBroker(market, settings, clock, settings.state_path / "pb.json"), clock,
                       out=out, strategies=[])
    assert "en marcha" in desk.submit("backtest BTC 1h 100d donchian")
    desk.run_cycle()
    rep = desk.reports.latest("BTC", "1h")
    assert rep.winner == "donchian_breakout" and rep.evaluations[0].passed
    assert "Ganadora: donchian_breakout" in out.text and "activa BTC 1h" in out.text

    reply = desk.submit("activa BTC 1h")
    assert reply.startswith("BTC 1h en automático con donchian_breakout") and "FORZADO" not in reply
    assert "Fuera de muestra (informe" in reply and "Modo paper" in reply
    entry = desk.playbook.get("BTC", "1h")
    assert entry.params == rep.evaluations[0].params

    # Primera vela posterior a la historia en la que la estrategia adoptada (con SUS parámetros) da señal.
    strat = create_strategy(entry.strategy, settings, interval="1h", **entry.params)
    sig = strat.signals(bars)
    j = next(j for j in range(2400, len(bars) - 1) if sig["side"].iloc[j])
    market.set_candles("BTC", bars.iloc[:j + 2], "1h")         # j cerrada, j + 1 en curso
    clock.set(bars.index[j].timestamp() + 3600 + 5)
    market.set_price("BTC", float(bars["close"].iloc[j]))
    desk.run_cycle()
    br = desk.engine.brackets["BTC"]
    assert br.origin == "auto:donchian_breakout@1h" and br.side.value == sig["side"].iloc[j]
    assert "gráfica BTC 1h (playbook)" in out.text

    assert "Fuera del automático: BTC 1h (donchian_breakout)" in desk.submit("desactiva BTC 1h")
    assert desk.playbook.entries() == [] and "no está en el playbook" in desk.submit("desactiva BTC 1h")


def test_adopt_without_report_and_unknown_strategy(tmp_path):
    settings = fast_settings(tmp_path)
    desk, *_ = sim_desk(settings)
    assert "No hay ningún backtest de SOL 1h" in desk.submit("activa SOL 1h")
    assert desk.submit("backtest DOGE 1h").startswith("No conozco la moneda DOGE")
    assert "sólo hay" in desk.submit("backtest SOL 1d 30d")       # 30 velas diarias: no da para walk-forward
    assert "No hay informes guardados" in desk.submit("informes")


def test_backtest_history_is_capped_at_5000_candles_and_says_so(tmp_path):
    settings = fast_settings(tmp_path)
    seen = []

    def runner(job, *a, **kw):
        seen.append(len(job.bars))
        return BacktestOutcome(job=job, error="sin calcular (test)")

    clock = SimClock(T0)
    worker = BacktestWorker(settings, ReportStore(settings.state_path), clock, threaded=False, runner=runner)
    desk, *_ = sim_desk(settings, clock, backtests=worker)
    reply = desk.submit("backtest SOL 15m 120d")              # 11520 velas pedidas
    assert "4999 velas" in reply and "historia recortada a 4999 velas" in reply and "como mucho 5000" in reply
    desk.run_cycle()
    assert seen == [4999]


def test_replaced_playbook_strategy_that_degrades_is_reported(tmp_path):
    settings = fast_settings(tmp_path)
    desk, market, clock, out = sim_desk(settings)
    desk.submit("backtest SOL 1h 40d")
    desk.run_cycle()
    rep = desk.reports.latest("SOL", "1h")
    rejected = next(e for e in rep.evaluations if not e.passed and not e.error)
    desk.submit(f"activa SOL 1h {rejected.strategy} forzar")
    clock.advance(3600)
    desk.submit("backtest SOL 1h 40d")                         # re-evaluación: sigue sin aprobar
    desk.run_cycle()
    assert "PLAYBOOK DEGRADADO: SOL 1h opera" in out.text and "desactiva SOL 1h" in out.text
    assert any(r["kind"] == "playbook" and r["level"] == "warn" for r in alert_records(settings))


# --------------------------------------------------------------------------- alarmas y avisos
def test_price_and_rsi_alarms_fire_and_are_notified(settings, market, clock):
    settings.auto_enabled = False
    out = Out()
    desk = TradingDesk(settings, market, PaperBroker(market, settings, clock, settings.state_path / "pb.json"), clock,
                       out=out, strategies=[])
    # RSI de velas de 15m: oscila (RSI ~50) y la vela que cierra tras crear la alarma cae un 8 %.
    closes = [100.0]
    for i in range(119):
        closes.append(closes[-1] * (1.005 if i % 2 else 0.995))
    closes.append(closes[-1] * 0.92)                           # vela n-2: la caída
    closes.append(closes[-1])                                  # vela n-1: en curso
    t1 = clock.now() + 600                                     # apertura de la vela en curso tras avanzar
    idx = pd.to_datetime([t1 - 900 * (len(closes) - 1 - i) for i in range(len(closes))], unit="s", utc=True)
    df = pd.DataFrame({"open": [closes[0]] + closes[:-1], "high": [c * 1.001 for c in closes],
                       "low": [c * 0.999 for c in closes], "close": closes, "volume": 1000.0}, index=idx)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "close"]].min(axis=1) * 0.999
    market.set_candles("SOL", df.iloc[:-1], "15m")             # ahora: la vela de la caída está en curso

    assert desk.submit("alerta SOL > 151").startswith("Alarma creada: #1 SOL precio > 151 · una vez")
    reply = desk.submit("alerta SOL rsi < 30 15m")
    assert reply.startswith("Alarma creada: #2 SOL RSI(14) 15m < 30") and "velas cerradas de 15m" in reply
    assert "#1 SOL precio > 151" in desk.submit("alertas")
    desk.run_cycle()
    assert "ALARMA" not in out.text                            # nada cruzó todavía

    clock.set(t1 + 10)                                         # cierra la vela de la caída
    market.set_candles("SOL", df, "15m")
    market.set_price("SOL", 152.0)                             # y el precio cruza 151
    desk.run_cycle()
    assert "ALARMA #1 SOL: precio 152 supera 151" in out.text
    assert "ALARMA #2 SOL: RSI(14) 15m" in out.text and "cae por debajo de 30" in out.text
    fired = {r["rule_id"]: r for r in alert_records(settings) if r["rule_id"]}
    assert set(fired) == {1, 2} and fired[1]["level"] == "warn" and fired[2]["kind"] == "rsi_below"
    assert desk.alerts.rules() == []                           # alarmas de una vez: eliminadas
    assert "No hay alarmas activas" in desk.submit("alertas")

    # Silencio: la alarma queda registrada como silenciada y no sale por los canales.
    assert desk.submit("silencio 30m").startswith("Avisos no críticos silenciados 30 min")
    desk.submit("alerta SOL < 151")
    n_out = len(out)
    market.set_price("SOL", 150.0)
    clock.advance(10)
    desk.run_cycle()
    assert not any("ALARMA #3" in x for x in out[n_out:])
    assert alert_records(settings)[-1]["muted"] is True and "ALARMA #3" in alert_records(settings)[-1]["text"]
    assert desk.submit("sonido").startswith("Avisos reactivados (1 silenciado(s)")
    desk.submit("alerta SOL > 200")
    assert desk.submit("borra alerta 9") == "No existe la alarma #9. 'alertas' para ver las activas."
    assert desk.submit("borra alertas") == "1 alarma(s) borradas."


def test_daily_loss_warning_at_80_percent_of_the_limit(settings, market, clock):
    settings.auto_enabled = False
    out = Out()
    desk = TradingDesk(settings, market, PaperBroker(market, settings, clock, settings.state_path / "pb.json"), clock,
                       out=out, strategies=[])
    desk.run_cycle()
    desk.submit("long SOL a mercado")
    pos = desk.broker.account().positions["SOL"]
    # Hueco a través del SL: la pérdida realizada deja el día en -2.5 % (límite 3 %, aviso al 80 % = 2.4 %).
    target = pos.entry_px - 0.025 * settings.paper_start_equity / pos.size
    clock.advance(10)
    market.set_price("SOL", target)
    desk.run_cycle()
    assert desk.risk.active_halt() is None
    warn = [x for x in out if "Pérdida del día al" in x]
    assert len(warn) == 1 and "del límite" in warn[0] and warn[0].startswith("\a[AVISO] ")
    [rec] = [r for r in alert_records(settings) if r["kind"] == "risk" and "Pérdida del día" in r["text"]]
    assert rec["level"] == "warn"
    clock.advance(10)
    desk.run_cycle()                                           # una vez por cruce: no se repite cada ciclo
    assert sum("Pérdida del día al" in x for x in out) == 1


def test_halt_is_critical_and_bypasses_mute(settings, market, clock):
    settings.auto_enabled = False
    out = Out()
    desk = TradingDesk(settings, market, PaperBroker(market, settings, clock, settings.state_path / "pb.json"), clock,
                       out=out, strategies=[])
    desk.run_cycle()
    desk.submit("silencio")
    desk.submit("long SOL a mercado 4u")
    pos = desk.broker.account().positions["SOL"]
    clock.advance(10)
    market.set_price("SOL", pos.entry_px - 0.04 * settings.paper_start_equity / pos.size)
    desk.run_cycle()
    assert desk.risk.active_halt() == "daily_loss"
    halt = [x for x in out if "HALT por pérdida diaria" in x]
    assert halt and halt[0].startswith("\a[CRÍTICO] ")
    assert any(r["kind"] == "halt" and r["level"] == "critical" and not r.get("muted")
               for r in alert_records(settings))


def test_repeated_cycle_failures_are_notified_once_per_threshold(settings, market, clock, monkeypatch):
    from desk.exchange.base import BrokerError
    out = Out()
    desk = TradingDesk(settings, market, PaperBroker(market, settings, clock, settings.state_path / "pb.json"), clock,
                       out=out, strategies=[])
    monkeypatch.setattr(desk.broker, "account", lambda: (_ for _ in ()).throw(BrokerError("timeout simulado")))
    for _ in range(4):
        desk.run_cycle()
    warns = [x for x in out if "La mesa lleva" in x]
    assert len(warns) == 1 and "La mesa lleva 3 ciclos seguidos con fallos" in warns[0]
    assert "timeout simulado" in warns[0] and warns[0].startswith("\a[AVISO] ")
    monkeypatch.undo()
    desk.run_cycle()
    assert "La mesa se recuperó tras 4 ciclos con fallos" in out.text


# --------------------------------------------------------------------------- Telegram
class FakeHttp:
    """Sesión HTTP falsa para el bot (``getUpdates``/``sendMessage``) y el canal de avisos (``sendPhoto``)."""

    def __init__(self, updates=None):
        self.updates = list(updates or [])
        self.calls: list[tuple[str, dict]] = []
        self.lock = threading.Lock()

    def post(self, url, json=None, data=None, files=None, timeout=None, **kw):
        method = url.rsplit("/", 1)[-1]
        assert url.startswith(f"https://api.telegram.org/bot{TOKEN}/")
        with self.lock:
            self.calls.append((method, {"json": json, "data": data, "files": files}))
        body = {"ok": True, "result": {"message_id": 1}}
        if method == "getUpdates":
            confirm = json.get("limit") == 1                   # el bot confirma el offset antes de ejecutar
            batch = self.updates.pop(0) if self.updates and not confirm else []
            body = {"ok": True, "result": batch}
        return _Resp(body)

    def sent(self, method="sendMessage"):
        with self.lock:
            return [c for m, c in self.calls if m == method]


class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body
        self.text = "ok"
        self.headers = {}

    def json(self):
        return self._body


def tg_update(uid, text, chat=CHAT):
    return {"update_id": uid, "message": {"message_id": uid * 10, "date": T0, "chat": {"id": chat, "type": "private"},
                                          "from": {"id": chat, "username": "operador"}, "text": text}}


def test_telegram_command_is_executed_and_answered_and_backtest_png_is_sent(tmp_path):
    settings = fast_settings(tmp_path, telegram_bot_token=TOKEN, telegram_chat_ids=[str(CHAT)], telegram_in_sim=True)
    http = FakeHttp([[tg_update(1, "/estado"), tg_update(2, "long SOL", chat=999)],
                     [tg_update(3, "backtest SOL 1h 40d")]])
    desk, market, clock, out = sim_desk(settings, http_session=http)
    assert desk.telegram is not None and "telegram (>= info)" in desk.notifier.describe()
    assert desk.telegram.poll_once() == 2
    [answer] = http.sent()
    assert answer["json"]["chat_id"] == str(CHAT) and answer["json"]["text"].startswith("[SIM fast] Estado de la mesa")
    assert desk.broker.account().positions == {}               # el chat no autorizado se ignora
    assert desk.telegram.ignored == 1 and '"source": "telegram"' in (
        settings.state_path / "events.jsonl").read_text(encoding="utf-8")

    assert desk.telegram.poll_once() == 1
    assert "en marcha" in http.sent()[-1]["json"]["text"]
    desk.run_cycle()                                           # termina el backtest -> aviso con la imagen
    assert desk.notifier.flush(5.0)
    if HAS_MPL:
        [photo] = http.sent("sendPhoto")
        assert photo["data"]["chat_id"] == str(CHAT) and photo["files"]["photo"][1].startswith(b"\x89PNG")
    texts = [c["json"]["text"] for c in http.sent()]
    assert any("FUERA de muestra (lo que cuenta)" in t for t in texts)
    assert TOKEN not in out.text


def test_telegram_submit_is_thread_safe_with_the_running_loop(tmp_path):
    """El bot llama a ``submit`` desde su hilo mientras el bucle cicla: el lock de la mesa los serializa."""
    settings = fast_settings(tmp_path, telegram_bot_token=TOKEN, telegram_chat_ids=[str(CHAT)], telegram_in_sim=True)
    http = FakeHttp([[tg_update(i, "estado")] for i in range(1, 21)])
    desk, market, clock, out = sim_desk(settings, http_session=http)
    errors: list[BaseException] = []

    def poll():
        try:
            for _ in range(20):
                desk.telegram.poll_once()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
    t = threading.Thread(target=poll)
    t.start()
    for _ in range(30):
        clock.advance(10)
        assert desk.run_cycle() is True
    t.join(10)
    assert not errors and len(http.sent()) == 20
    assert all(c["json"]["text"].startswith("[SIM fast] Estado de la mesa") for c in http.sent())


# --------------------------------------------------------------------------- status.json ampliado
def test_status_json_is_extended_for_the_dashboard(tmp_path):
    settings = fast_settings(tmp_path)
    desk, market, clock, out = sim_desk(settings)
    desk.submit("backtest SOL 1h 40d")
    desk.run_cycle()
    rep = desk.reports.latest("SOL", "1h")
    desk.submit(f"activa SOL 1h {rep.evaluations[0].strategy} forzar")
    desk.submit("alerta SOL > 1000")
    desk.submit("long SOL a mercado")
    desk.submit("silencio 10m")
    clock.advance(10)
    desk.run_cycle()                                           # tras los comandos se reescribe status.json
    raw = json.loads((settings.state_path / "status.json").read_text(encoding="utf-8"))
    for key in ("mode", "data", "equity", "available", "day_pnl", "risk", "positions", "orders", "pending",
                "playbook", "alerts", "signals", "backtests", "notifier", "auto", "halt", "symbols"):
        assert key in raw, key
    assert raw["risk"]["daily_loss_limit_pct"] == settings.daily_loss_limit_pct
    assert raw["alerts"]["rules"] == 1 and raw["alerts"]["muted"] is True
    st = normalize_status(raw)
    assert st["positions"][0]["coin"] == "SOL" and st["positions"][0]["origin"] == "comando"
    assert st["playbook"][0]["strategy"] == rep.evaluations[0].strategy
    assert st["daily_used_pct"] is not None and st["dd_used_pct"] is not None
    snap = load_snapshot(settings.state_path, clock)
    assert snap.report is not None and snap.report["id"] == rep.id
    assert snap.alert_stats["muted"] is True and len(snap.playbook) == 1


# --------------------------------------------------------------------------- el bucle sobrevive a los backtests
def test_loop_survives_a_backtest_job_that_raises(tmp_path):
    settings = fast_settings(tmp_path)

    def boom(job, settings, store, clock, **kw):
        raise RuntimeError("fallo de prueba en el optimizador")

    clock = SimClock(T0)
    worker = BacktestWorker(settings, ReportStore(settings.state_path), clock, threaded=False, runner=boom)
    desk, market, _, out = sim_desk(settings, clock, backtests=worker)
    assert "en marcha" in desk.submit("backtest SOL 1h 40d")
    assert desk.run_cycle() is True
    assert "Backtest SOL 1h (40 días) falló: RuntimeError: fallo de prueba en el optimizador" in out.text
    assert desk.submit("long SOL ya").startswith("SOL LONG")    # la mesa sigue
    clock.advance(10)
    assert desk.run_cycle() is True and worker.failed == 1


def test_threaded_backtest_does_not_block_the_loop_and_reports_when_done(tmp_path):
    settings = fast_settings(tmp_path)
    release = threading.Event()
    started = threading.Event()

    def slow(job, settings, store, clock, **kw):
        started.set()
        release.wait(10)
        raise ValueError("terminó con error controlado")

    clock = SimClock(T0)
    worker = BacktestWorker(settings, ReportStore(settings.state_path), clock, threaded=True, runner=slow)
    desk, market, _, out = sim_desk(settings, clock, backtests=worker)
    reply = desk.submit("backtest SOL 1h 40d")
    assert "en marcha" in reply and "La mesa sigue operando" in reply
    assert started.wait(5)
    assert "Ya hay un backtest de SOL 1h" in desk.submit("backtest SOL 1h 40d")
    for _ in range(5):                                         # el hilo trabaja; el bucle no espera
        clock.advance(10)
        assert desk.run_cycle() is True
    assert "falló" not in out.text and "En marcha: SOL 1h" in desk.submit("informes")
    release.set()
    deadline = time.monotonic() + 5
    while not worker.has_results() and time.monotonic() < deadline:
        time.sleep(0.01)
    desk.run_cycle()
    assert "Backtest SOL 1h (40 días) falló: ValueError: terminó con error controlado" in out.text
    desk.shutdown()
    assert not (worker._thread and worker._thread.is_alive())


def test_a_broken_notice_or_worker_never_stops_the_loop(tmp_path, monkeypatch):
    settings = fast_settings(tmp_path)
    desk, market, clock, out = sim_desk(settings)
    desk.submit("backtest SOL 1h 40d")
    monkeypatch.setattr(desk, "_on_backtest_done", lambda o: (_ for _ in ()).throw(KeyError("raro")))
    assert desk.run_cycle() is True                            # el fallo al avisar queda en el log
    monkeypatch.setattr(desk.backtests, "poll", lambda: (_ for _ in ()).throw(RuntimeError("cola rota")))
    clock.advance(10)
    assert desk.run_cycle() is False and desk.stage_errors["backtests"] == 1
    monkeypatch.undo()
    clock.advance(10)
    assert desk.run_cycle() is True


def test_shutdown_cancels_pending_backtests(tmp_path):
    settings = fast_settings(tmp_path)
    clock = SimClock(T0)
    calls = []

    def runner(job, *a, **kw):
        calls.append(job.id)
        return BacktestOutcome(job=job, error="no debería correr")

    worker = BacktestWorker(settings, ReportStore(settings.state_path), clock, threaded=False, runner=runner)
    desk, market, _, out = sim_desk(settings, clock, backtests=worker)
    desk.submit("backtest SOL 1h 40d")
    desk.shutdown()
    assert calls == [] and "1 backtest(s) en marcha o en cola cancelados" in out.text and "Mesa detenida" in out[-1]
    assert "se está deteniendo" in desk.submit("backtest SOL 1h 40d")


# --------------------------------------------------------------------------- IA: qué pide confirmación
class FakeLLM:
    def __init__(self, cmd):
        self.cmd = cmd

    def interpret(self, text):
        from dataclasses import replace
        return replace(self.cmd, raw=text, source="llm")


def test_llm_backtest_runs_without_confirmation_but_adopt_asks(tmp_path):
    settings = fast_settings(tmp_path, llm_enabled=True)
    llm = FakeLLM(Command(CommandAction.BACKTEST, coin="SOL", interval="1h", days=40.0))
    desk, *_ = sim_desk(settings, parser=CommandParser(settings, llm=llm))
    assert "en marcha" in desk.submit("quiero ver cómo le habría ido a solana")
    llm.cmd = Command(CommandAction.ADOPT, coin="SOL", interval="1h")
    assert "¿Lo ejecuto?" in desk.submit("me gustaría que lo de solana fuese solo")


# --------------------------------------------------------------------------- CLI
def test_cli_backtest_offline_with_sim_and_csv(tmp_path, capsys, monkeypatch):
    from desk.__main__ import main
    for key, value in (("DESK_BT_MAX_COMBOS", "3"), ("DESK_BT_FOLDS", "2")):
        monkeypatch.setenv(key, value)
    state = tmp_path / "st"
    env = str(tmp_path / "no.env")
    assert main(["backtest", "--data", "sim", "--coin", "sol", "--interval", "1h", "--days", "30", "--state-dir",
                 str(state), "--env-file", env]) == 0
    printed = capsys.readouterr().out
    assert "Backtest SOL 1h: 720 velas cerradas" in printed and "FUERA de muestra (lo que cuenta)" in printed
    assert "Informe HTML:" in printed and "un backtest no garantiza" in printed
    [report] = (state / "reports").glob("SOL_1h_*.json")
    assert report.with_suffix(".html").exists() and report.with_suffix(".png").exists() is HAS_MPL

    bars = trend_bars(seed=2, n=900)
    csv = tmp_path / "btc_1h.csv"
    bars.assign(timestamp=[int(t.timestamp()) for t in bars.index]).to_csv(csv, index=False)
    assert main(["backtest", "--csv", str(csv), "--coin", "BTC", "--interval", "1h", "--days", "30",
                 "--strategy", "donchian", "--no-render", "--state-dir", str(state), "--env-file", env]) == 0
    printed = capsys.readouterr().out
    assert "Backtest BTC 1h: 720 velas cerradas" in printed and "donchian_breakout" in printed
    assert not list((state / "reports").glob("BTC_1h_*.html"))
    assert main(["backtest", "--data", "sim", "--coin", "SOL", "--interval", "7h", "--state-dir", str(state),
                 "--env-file", env]) == 2


# --------------------------------------------------------------------------- revisión: validez y seguridad de avisos
import logging  # noqa: E402

from conftest import FakeMarketData  # noqa: E402
from desk.desk import desk_identity  # noqa: E402
from desk.exchange.base import BrokerError  # noqa: E402
from desk.models import OrderKind  # noqa: E402
from desk.notify import Channel, Notifier  # noqa: E402
from desk.optimizer import OptimizationReport, StrategyEvaluation  # noqa: E402


class Rec(Channel):
    """Canal que guarda lo que recibe (como Telegram o el webhook, sin red)."""

    def __init__(self, name, min_level):
        super().__init__(min_level)
        self.name = name
        self.got: list[tuple[str, str]] = []

    def send(self, n):
        self.got.append((n.level, n.text))


def recorded_desk(settings, clock=None, market=None, **kw):
    clock = clock or SimClock(T0)
    tg, wh = Rec("telegram", "info"), Rec("webhook", "warn")
    notifier = Notifier([tg, wh], clock=clock, threaded=False)
    if market is None:
        desk, market, clock, out = sim_desk(settings, clock, notifier=notifier, **kw)
    else:
        broker = PaperBroker(market, settings, clock, settings.state_path / "pb.json")
        desk = TradingDesk(settings, market, broker, clock, out=Out(), notifier=notifier, strategies=[], **kw)
    return desk, market, clock, tg, wh


def fake_report(rid, coin, interval, created_ts, verdicts):
    base = load_settings(env={}, dotenv_path=None, llm_enabled=False)
    evs = [StrategyEvaluation(strategy=name, params=create_strategy(name, base, interval=interval).params(),
                              is_metrics={}, oos_metrics={"total_return_pct": 3.0,
                              "profit_factor": 1.5, "max_drawdown_pct": 2.0, "n_trades": 30}, oos_trades=[], folds=[],
                              score=1.0, passed=ok, reasons=[] if ok else ["profit factor fuera de muestra 0.90 < 1.15"])
           for name, ok in verdicts.items()]
    win = next((e.strategy for e in evs if e.passed), None)
    return OptimizationReport(id=rid, coin=coin, interval=interval, days=40, bars=960, created_ts=created_ts,
                              evaluations=evs, winner=win)


def test_adopt_requires_force_if_a_recent_report_of_the_chart_rejected_the_strategy(tmp_path):
    """Repetir backtests de la gráfica hasta que uno apruebe sube la tasa de falsos positivos: 'activa' exige
    'forzar' si en los últimos 30 días otro informe de la misma gráfica rechazó esa estrategia."""
    settings = fast_settings(tmp_path)
    desk, *_ = sim_desk(settings)
    desk.reports.save(fake_report("aaaa0001", "SOL", "1h", T0 - 5 * 86_400, {"ema_trend": False}))
    desk.reports.save(fake_report("aaaa0002", "SOL", "1h", T0 - 3_600, {"ema_trend": True}))
    reply = desk.submit("activa SOL 1h")
    assert reply.startswith("No se activó SOL 1h") and "aaaa0001" in reply and "forzar" in reply
    assert desk.playbook.entries() == []
    reply = desk.submit("activa SOL 1h ema_trend forzar")
    assert "ATENCIÓN — FORZADO" in reply and "aaaa0001" in reply and desk.playbook.get("SOL", "1h") is not None
    # Un rechazo de hace más de 30 días ya no cuenta.
    desk.reports.save(fake_report("bbbb0001", "BTC", "1h", T0 - 45 * 86_400, {"ema_trend": False}))
    desk.reports.save(fake_report("bbbb0002", "BTC", "1h", T0 - 3_600, {"ema_trend": True}))
    reply = desk.submit("activa BTC 1h")
    assert reply.startswith("BTC 1h en automático con ema_trend") and "FORZADO" not in reply


def test_backtest_rejecting_a_default_strategy_warns_and_desactiva_turns_it_off(tmp_path):
    settings = fast_settings(tmp_path, auto_default_strategies=True, symbols=["SOL"])
    clock = SimClock(T0)
    seen = []

    def runner(job, *a, **kw):
        seen.append(job.start_equity)
        return BacktestOutcome(job=job, report=fake_report("cccc0001", "SOL", "1h", clock.now(),
                                                           {"squeeze_breakout": False}))

    worker = BacktestWorker(settings, ReportStore(settings.state_path), clock, threaded=False, runner=runner)
    desk, market, clock, out = sim_desk(settings, clock, backtests=worker)
    assert ("SOL", "squeeze_breakout@1h") in [(s.coin, s.label) for s in desk.auto_slots()]
    reply = desk.submit("backtest SOL 1h 40d squeeze_breakout")
    assert "Capital supuesto: 10,000 USD (equity de la cuenta)" in reply
    desk.run_cycle()
    assert seen == [10_000.0]                                  # el capital de la cuenta llega al trabajo
    assert "ESTRATEGIA POR DEFECTO RECHAZADA: SOL 1h opera squeeze_breakout" in out.text
    assert any(r["kind"] == "playbook" and r["level"] == "warn" for r in alert_records(settings))
    reply = desk.submit("desactiva SOL 1h")
    assert "SOL squeeze_breakout@1h (por defecto)" in reply
    assert [s.label for s in desk.auto_slots()] == ["liquidity_sweep@15m"]
    again, *_ = sim_desk(settings, clock)                      # persistente (desk.json)
    assert [s.label for s in again.auto_slots()] == ["liquidity_sweep@15m"]
    assert "Desactivadas con 'desactiva': SOL squeeze_breakout@1h" in again.submit("playbook")


def test_playbook_coin_does_not_also_run_unvalidated_default_strategies(tmp_path):
    settings = fast_settings(tmp_path, auto_default_strategies=True, symbols=["SOL", "BTC"])
    desk, *_ = sim_desk(settings)
    desk.reports.save(fake_report("dddd0001", "SOL", "4h", T0 - 60, {"ema_trend": True}))
    assert desk.submit("activa SOL 4h").startswith("SOL 4h en automático")
    slots = [(s.coin, s.label, s.source) for s in desk.auto_slots()]
    assert [x for x in slots if x[0] == "SOL"] == [("SOL", "ema_trend@4h", "playbook")]
    assert {x[2] for x in slots if x[0] == "BTC"} == {"defecto"}


def test_backtest_download_runs_in_the_worker_thread_not_under_the_desk_lock(tmp_path):
    settings = fast_settings(tmp_path)
    gate, called = threading.Event(), threading.Event()

    class SlowCandles(SimMarketData):
        def candles(self, coin, interval, limit):
            if limit > 500:
                called.set()
                gate.wait(10)
            return super().candles(coin, interval, limit)

    clock = SimClock(T0)
    market = SlowCandles(clock)
    worker = BacktestWorker(settings, ReportStore(settings.state_path), clock, threaded=True)
    out = Out()
    desk = TradingDesk(settings, market, PaperBroker(market, settings, clock, settings.state_path / "pb.json"), clock,
                       out=out, strategies=[], backtests=worker)
    try:
        t = time.perf_counter()
        reply = desk.submit("backtest SOL 1h 40d")
        assert time.perf_counter() - t < 2.0 and "~960 velas" in reply
        assert called.wait(5)                                  # el hilo del trabajo está descargando...
        t = time.perf_counter()
        assert desk.run_cycle() is True                        # ... y el bucle no espera
        assert time.perf_counter() - t < 2.0
        gate.set()
        deadline = time.monotonic() + 60
        while "Backtest SOL 1h" not in out.text and "FUERA de muestra" not in out.text and time.monotonic() < deadline:
            time.sleep(0.05)
            desk.run_cycle()
        assert desk.reports.latest("SOL", "1h") is not None
    finally:
        gate.set()
        desk.backtests.stop(5)


def test_simulation_does_not_use_the_real_telegram_bot_or_webhook(tmp_path):
    class AnyHttp:
        def __init__(self):
            self.urls = []

        def post(self, url, **kw):
            self.urls.append(url)
            return _Resp({"ok": True, "result": []})

    http = AnyHttp()
    settings = fast_settings(tmp_path, data_source="sim", telegram_bot_token=TOKEN, telegram_chat_ids=[str(CHAT)],
                             webhook_url="https://discord.com/api/webhooks/1/secreto")
    desk, *_ = sim_desk(settings, http_session=http)
    assert desk.telegram is None and desk.notifier.describe() == "consola (>= info)"
    assert any("DESACTIVADOS" in n for n in desk.notices) and "DESACTIVADOS" in desk.banner()
    desk.notify("HALT de prueba", logging.CRITICAL)
    desk.notifier.flush(2)
    assert http.urls == []
    # Pedido a propósito: sí, y cada aviso dice de qué mesa viene.
    opt_in = fast_settings(tmp_path / "b", data_source="sim", telegram_bot_token=TOKEN,
                           telegram_chat_ids=[str(CHAT)], telegram_in_sim=True)
    http2 = FakeHttp()
    desk2, *_ = sim_desk(opt_in, http_session=http2)
    assert desk2.telegram is not None and desk2.telegram.label == "SIM fast"
    desk2.notify("aviso de prueba", logging.WARNING)
    assert desk2.notifier.flush(5)
    from desk.notify import LEVEL_LABELS
    assert [c["json"]["text"] for c in http2.sent()] == [f"[SIM fast] {LEVEL_LABELS['warn']}aviso de prueba"]


def test_desk_identity_labels():
    from dataclasses import replace
    s = load_settings(env={}, dotenv_path=None, llm_enabled=False)
    assert desk_identity(s) == "PAPER" and desk_identity(s, sim_clock=True) == "SIM fast"
    assert desk_identity(replace(s, data_source="sim")) == "SIM"
    assert desk_identity(replace(s, mode="live")) == "LIVE testnet"
    assert desk_identity(replace(s, mode="live", hl_mainnet=True)) == "LIVE mainnet"


def test_repeated_telegram_409_is_a_critical_notice(tmp_path):
    class StubBot:
        running, handled, ignored, errors, last_error, conflicts = True, 0, 0, 3, "getUpdates: HTTP 409 Conflict", 3
        label = ""

        def start(self):
            return True

        def stop(self, timeout=0):
            return None

    settings = fast_settings(tmp_path)
    desk, market, clock, tg, wh = recorded_desk(settings, telegram=StubBot())
    for _ in range(5):
        clock.advance(10)
        desk.run_cycle()
    crit = [t for lv, t in wh.got if lv == "critical" and "OTRO PROCESO" in t]
    assert len(crit) == 1                                     # anti-spam: uno por alert_cooldown_seconds


def test_unprotected_position_and_safety_close_are_critical_even_when_muted(tmp_path):
    settings = fast_settings(tmp_path)
    desk, market, clock, tg, wh = recorded_desk(settings)
    orig = desk.broker.place_order

    def place(coin, is_buy, size, kind, **kw):
        if kind is OrderKind.STOP:
            raise BrokerError("timeout colocando el stop (simulado)")
        return orig(coin, is_buy, size, kind, **kw)

    desk.broker.place_order = place
    desk.run_cycle()
    assert "silenciados" in desk.submit("silencio")
    desk.submit("long SOL a mercado")
    for _ in range(12):
        clock.advance(10)
        desk.run_cycle()
    for ch in (tg, wh):
        unprotected = [lv for lv, t in ch.got if "NO colocado" in t]
        assert unprotected == ["critical"]                     # suena, llega al webhook y sólo una vez (anti-spam)
        assert [lv for lv, t in ch.got if "por seguridad" in t] == ["critical"]
    assert "SOL" not in desk.broker.account().positions


def test_liquidation_is_critical(tmp_path):
    settings = fast_settings(tmp_path, leverage=10, daily_loss_limit_pct=50, max_drawdown_pct=90)
    clock = SimClock(T0)
    market = FakeMarketData(clock)
    desk, market, clock, tg, wh = recorded_desk(settings, clock, market)
    orig = desk.broker.place_order
    desk.broker.place_order = lambda c, ib, sz, k, **kw: ((_ for _ in ()).throw(BrokerError("timeout"))
                                                          if k is OrderKind.STOP else orig(c, ib, sz, k, **kw))
    desk.run_cycle()
    desk.submit("silencio")
    desk.submit("long SOL a mercado")
    pos = desk.broker.account().positions["SOL"]
    clock.advance(10)
    market.set_price("SOL", pos.liq_px * 0.99)
    desk.run_cycle()
    assert [lv for lv, t in wh.got if "LIQUIDACIÓN" in t] == ["critical"]


def test_kill_with_an_entry_that_cannot_be_cancelled_does_not_flood_the_channels(tmp_path):
    settings = fast_settings(tmp_path, loop_seconds=5)
    desk, market, clock, tg, wh = recorded_desk(settings)
    desk.run_cycle()
    mid = market.mid("SOL")
    assert "límite" in desk.submit(f"long SOL a {mid * 0.95:.2f}")
    desk.broker.cancel = lambda coin, oid: (_ for _ in ()).throw(BrokerError("cancel: 502 (simulado)"))
    desk.submit("kill")
    tg.got.clear()
    for _ in range(60):
        clock.advance(5)
        desk.run_cycle()
    crit = [t for lv, t in tg.got if lv == "critical"]
    assert len(crit) == 1 and "SIN PODER CANCELAR" in crit[0] and "paper-" in crit[0]
    assert "0 posición" not in crit[0]


def test_persistently_rejected_take_profit_is_reported_once_and_then_disabled(tmp_path):
    from desk.engine import MAX_TP_FAILURES
    settings = fast_settings(tmp_path, loop_seconds=5)
    desk, market, clock, tg, wh = recorded_desk(settings)
    orig = desk.broker.place_order

    def place(coin, is_buy, size, kind, **kw):
        if kind is OrderKind.TAKE_PROFIT:
            kw["trigger_px"] = -1.0                            # el exchange lo rechaza siempre
        return orig(coin, is_buy, size, kind, **kw)

    desk.broker.place_order = place
    desk.run_cycle()
    desk.submit("long SOL a mercado")
    for _ in range(120):
        clock.advance(5)
        desk.run_cycle()
    rejected = [(lv, t) for lv, t in tg.got if "TP" in t and "RECHAZADO" in t]
    assert 1 <= len(rejected) <= 2 and all(lv == "warn" for lv, _ in rejected)
    assert any("TP desactivado" in t for _, t in rejected)
    br = desk.engine.brackets["SOL"]
    assert br.tp is None and br.sl is not None and "SOL" in desk.broker.account().positions
    assert sum(1 for line in (settings.state_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
               if '"protect_rejected"' in line and '"tp"' in line) == MAX_TP_FAILURES


def test_deferred_download_errors_come_back_in_the_outcome(tmp_path):
    """Con hilo, la descarga la hace el trabajo: un fallo de datos (API caída, pocas velas) llega en el aviso del
    resultado con un texto para el usuario y el hilo sigue vivo."""
    from desk.exchange.base import MarketDataError
    from desk.jobs import BacktestJob
    settings = fast_settings(tmp_path)
    clock = SimClock(T0)
    worker = BacktestWorker(settings, ReportStore(settings.state_path), clock, threaded=False)

    def down():
        raise MarketDataError("candleSnapshot: HTTP 502")

    def short():
        return trend_bars(1, n=40), []

    for i, fetch in enumerate((down, short)):
        worker.submit(BacktestJob(id=f"j{i}", coin="SOL", interval="1h", days=40, strategy=None, bars=None,
                                  fetch=fetch))
    outs = worker.poll()
    assert [o.ok for o in outs] == [False, False]
    assert "no se pudieron descargar las velas de SOL 1h" in outs[0].error and "502" in outs[0].error
    assert "sólo hay 40 velas cerradas" in outs[1].error

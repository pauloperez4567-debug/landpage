"""Tests end-to-end de la mesa: FakeMarketData + PaperBroker + SimClock, sin red.

Cubren el flujo completo de comandos (zona → ejecución → SL/TP → OCO → PnL), cierres, kill, halts de
riesgo con flatten, reinicios con estado persistido, el inbox con offset, la confirmación de comandos
interpretados por IA, la operativa automática y la resistencia del bucle a fallos.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import signal
from dataclasses import replace
from typing import Optional

import pytest

from desk.commands import CommandParser
from desk.config import load_settings
from desk.desk import MAX_BACKOFF_S, TradingDesk
from desk.exchange.base import BrokerError
from desk.exchange.paper import PaperBroker
from desk.models import Command, CommandAction, EntryMode, OrderKind, OrderStatus, Side, Signal
from desk.strategy import Strategy

LONG, SHORT = Side.LONG, Side.SHORT
DAY = 86_400.0


# --------------------------------------------------------------------------- utilidades
class Out(list):
    """Salida de la mesa capturada (lo que vería el operador en pantalla)."""

    def __call__(self, text: str) -> None:
        self.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self)


class StubStrategy(Strategy):
    """Estrategia de prueba: señal fija por moneda (``plan``) sobre la última vela."""

    name = "stub"
    interval = "15m"
    lookback = 10
    min_bars = 1

    def __init__(self, settings, clock=None, plan: Optional[dict] = None, error: Optional[Exception] = None):
        super().__init__(settings, clock)
        self.plan = plan or {}
        self.error = error
        self.calls: list[str] = []

    def evaluate(self, coin, candles, mid, zones=None, now=None):
        self.calls.append(coin)
        if self.error is not None:
            raise self.error
        side = self.plan.get(coin)
        if side is None:
            return None
        sl = mid * (0.98 if side is LONG else 1.02)
        tp = mid * (1.04 if side is LONG else 0.96)
        return Signal(coin, side, EntryMode.MARKET, self.name, "señal de prueba", entry_px=mid, sl_price=sl,
                      tp_price=tp)

    def _evaluate_closed(self, coin, bars, mid, zones):
        return None


class FakeLLM:
    def __init__(self, cmd: Optional[Command] = None):
        self.cmd = cmd
        self.calls: list[str] = []

    def interpret(self, text: str) -> Optional[Command]:
        self.calls.append(text)
        return None if self.cmd is None else replace(self.cmd, raw=text, source="llm")


def paper_for(market, settings, clock) -> PaperBroker:
    return PaperBroker(market, settings, clock, settings.state_path / "paper_broker.json")


def make_desk(settings, market, clock, *, broker=None, parser=None, strategies=None, auto=False, out=None):
    settings.auto_enabled = auto
    broker = broker or paper_for(market, settings, clock)
    return TradingDesk(settings, market, broker, clock, parser=parser,
                       strategies=[] if strategies is None else strategies, out=out if out is not None else Out())


def hype_book(market):
    """Muro de bids en 85.25: zona LONG 85.241-85.336, entrada 85.288 con el mid en 86.285."""
    market.set_book("HYPE", [(86.28, 10.0), (85.25, 50_000.0)], [(86.29, 10.0), (87.5, 100.0)])


def move(market, clock, coin, px, secs=10.0):
    clock.advance(secs)
    market.set_price(coin, px)


def trades(settings):
    path = settings.state_path / "trades.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def outbox(settings) -> str:
    p = settings.state_path / "outbox.log"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def tags(broker, coin=None):
    return sorted(o.tag for o in broker.open_orders(coin))


# --------------------------------------------------------------------------- flujo principal
def test_long_hype_zone_fill_brackets_take_profit_oco_pnl(settings, market, clock):
    hype_book(market)
    out = Out()
    desk = make_desk(settings, market, clock, out=out)
    reply = desk.submit("long HYPE")
    first = reply.splitlines()[0]
    assert first.startswith("HYPE LONG 11.72 @ 85.288 (zona liq 85.241-85.336, $1,000 notional, 2x aislado) "
                            "SL 84.009 TP 87.846"), reply
    assert "Orden límite en espera" in reply and "apalancamiento recortado de 5x a 2x" in reply
    entry = desk.broker.open_orders("HYPE")
    assert len(entry) == 1 and entry[0].kind is OrderKind.LIMIT and entry[0].price == 85.288

    desk.run_cycle()                                       # el precio no ha llegado: sigue en espera
    assert desk.broker.account().positions == {}

    move(market, clock, "HYPE", 85.2)                      # baja a la zona: se ejecuta la entrada
    assert desk.run_cycle()
    pos = desk.broker.account().positions["HYPE"]
    assert pos.size == pytest.approx(11.72) and pos.entry_px == 85.288
    assert tags(desk.broker, "HYPE") == ["sl", "tp"]
    sl_id = desk.engine.brackets["HYPE"].sl_order_id
    assert "entrada ejecutada" in out.text and "SL 84.009 colocado" in out.text

    move(market, clock, "HYPE", 88.0)                      # sube y toca el TP
    assert desk.run_cycle()
    assert desk.broker.account().positions == {} and desk.broker.open_orders() == []
    assert desk.broker.get_order(sl_id).status is OrderStatus.CANCELED          # OCO
    rows = trades(settings)
    assert len(rows) == 1 and rows[0]["exit_reason"] == "tp" and float(rows[0]["pnl"]) > 0
    assert desk.risk.consecutive_losses == 0
    assert desk.session["trades"] == 1 and desk.session["wins"] == 1
    assert "cerrada por take-profit" in out.text and "SL cancelado (OCO)" in out.text
    box = outbox(settings)
    assert "[c-000001] HYPE LONG 11.72 @ 85.288" in box and "[desk]" in box


def test_stop_loss_hit_records_loss(settings, market, clock):
    hype_book(market)
    desk = make_desk(settings, market, clock)
    desk.submit("long HYPE")
    move(market, clock, "HYPE", 85.2)
    desk.run_cycle()
    tp_id = desk.engine.brackets["HYPE"].tp_order_id
    move(market, clock, "HYPE", 83.9)
    desk.run_cycle()
    rows = trades(settings)
    assert len(rows) == 1 and rows[0]["exit_reason"] == "sl" and float(rows[0]["pnl"]) < 0
    assert desk.broker.get_order(tp_id).status is OrderStatus.CANCELED
    assert desk.risk.consecutive_losses == 1


def test_market_command_is_protected_immediately(settings, market, clock):
    desk = make_desk(settings, market, clock)
    reply = desk.submit("long SOL 2u a mercado sl 1% tp 3%")
    assert reply.startswith("SOL LONG ") and "a mercado" in reply and "Ejecutada a mercado" in reply
    assert tags(desk.broker, "SOL") == ["sl", "tp"]          # sin esperar al siguiente ciclo
    br = desk.engine.brackets["SOL"]
    assert br.origin == "command" and br.sl == pytest.approx(150.02 * 0.99, abs=0.01)


def test_close_all_closes_positions_and_cancels_pending(settings, market, clock):
    hype_book(market)
    desk = make_desk(settings, market, clock)
    desk.submit("long SOL a mercado")
    desk.submit("short BTC ya")
    desk.submit("long HYPE")                                 # entrada pendiente en zona
    assert len(desk.broker.account().positions) == 2
    reply = desk.submit("cierra todo")
    assert "Cerrando todo: 2 posición(es)" in reply and "HYPE" in reply
    assert desk.broker.account().positions == {} and desk.broker.open_orders() == []
    assert desk.engine.brackets == {}
    assert sorted(r["exit_reason"] for r in trades(settings)) == ["close_all", "close_all"]


def test_close_single_coin(settings, market, clock):
    desk = make_desk(settings, market, clock)
    desk.submit("long SOL a mercado")
    reply = desk.submit("cierra SOL")
    assert "cierre a mercado enviado" in reply and "cerrada por cierre manual" in reply
    assert desk.submit("cierra SOL").startswith("No hay posición")


def test_kill_flattens_halts_and_needs_reset(settings, market, clock):
    hype_book(market)
    desk = make_desk(settings, market, clock)
    desk.submit("long SOL a mercado")
    desk.submit("long HYPE")
    reply = desk.submit("kill")
    assert "PÁNICO" in reply and "reset halt" in reply
    assert desk.broker.account().positions == {} and desk.broker.open_orders() == []
    assert desk.risk.halted and desk.risk.halt_reason == "kill"
    blocked = desk.submit("long ETH ya")
    assert "rechazada por riesgo" in blocked and "reset halt" in blocked
    desk.run_cycle()
    assert "sigue el halt" in desk.submit("reanuda")
    assert "levantado" in desk.submit("reset halt").lower()
    assert desk.submit("long ETH ya").startswith("ETH LONG")


# --------------------------------------------------------------------------- halts de riesgo
def test_daily_loss_halt_blocks_entries_and_lifts_next_day(settings, market, clock):
    out = Out()
    desk = make_desk(settings, market, clock, out=out)
    desk.submit("long SOL a mercado")
    desk.run_cycle()
    desk.broker.cash -= 350.0                             # -3.5% en el día (drawdown 3.5% < 5%)
    clock.advance(10)
    desk.run_cycle()
    assert desk.risk.halted and desk.risk.halt_reason == "daily_loss"
    assert "SOL" in desk.broker.account().positions       # sin flatten por defecto
    assert "HALT por pérdida diaria" in out.text
    assert "rechazada por riesgo" in desk.submit("long ETH ya")
    assert "HALT daily_loss" in desk.status_report()
    clock.set((clock.now() // DAY + 1) * DAY + 30)        # siguiente día UTC
    desk.run_cycle()
    assert not desk.risk.halted
    assert "se levantó" in out.text
    assert desk.submit("long ETH ya").startswith("ETH LONG")


def test_daily_loss_halt_with_flatten_closes_everything(settings, market, clock):
    settings.flatten_on_daily_limit = True
    hype_book(market)
    desk = make_desk(settings, market, clock)
    desk.submit("long SOL a mercado")
    desk.submit("long HYPE")                               # entrada pendiente: se cancela con el halt
    desk.run_cycle()
    desk.broker.cash -= 350.0
    clock.advance(10)
    desk.run_cycle()
    assert desk.risk.halt_reason == "daily_loss"
    assert desk.broker.account().positions == {} and desk.broker.open_orders() == []
    assert [r["exit_reason"] for r in trades(settings)] == ["flatten"]


def test_max_drawdown_halt_flattens_and_only_reset_lifts(settings, market, clock):
    out = Out()
    hype_book(market)
    desk = make_desk(settings, market, clock, out=out)
    desk.submit("long SOL a mercado")
    desk.submit("short BTC a mercado")
    desk.submit("long HYPE")
    desk.run_cycle()
    desk.broker.cash -= 600.0                             # -6% desde el máximo
    clock.advance(10)
    desk.run_cycle()
    assert desk.risk.halt_reason == "max_drawdown"
    assert desk.broker.account().positions == {} and desk.broker.open_orders() == []
    assert sorted(r["exit_reason"] for r in trades(settings)) == ["flatten", "flatten"]
    assert "HALT por drawdown máximo" in out.text
    clock.set((clock.now() // DAY + 1) * DAY + 30)        # el día nuevo no lo levanta
    desk.run_cycle()
    assert desk.risk.halt_reason == "max_drawdown"
    assert "sigue el halt" in desk.submit("reanuda")
    desk.submit("reset halt")
    desk.run_cycle()
    assert not desk.risk.halted                           # HWM reiniciado: no vuelve a saltar


def test_loss_streak_cooldown_blocks_auto_but_not_commands(settings, market, clock):
    settings.symbols = ["SOL"]
    stub = StubStrategy(settings, clock, plan={"SOL": LONG})
    desk = make_desk(settings, market, clock, strategies=[stub], auto=True)
    desk.risk.cooldown_until = clock.now() + 3600
    desk.run_cycle()
    assert desk.broker.account().positions == {} and stub.calls == []
    assert desk.submit("long SOL ya").startswith("SOL LONG")
    assert "cooldown" in desk.status_line()


# --------------------------------------------------------------------------- reinicios y persistencia
def test_restart_keeps_brackets_and_oco(settings, market, clock):
    desk1 = make_desk(settings, market, clock)
    desk1.submit("long SOL a mercado")
    desk1.run_cycle()
    sl_id = desk1.engine.brackets["SOL"].sl_order_id
    tp_px = desk1.engine.brackets["SOL"].tp
    out = Out()
    desk2 = make_desk(settings, market, clock, out=out)   # broker paper recargado del disco
    assert desk2.engine.brackets["SOL"].sl_order_id == sl_id
    desk2.run_cycle()
    assert "adoptada" not in out.text and desk2.engine.brackets["SOL"].sl_order_id == sl_id
    move(market, clock, "SOL", tp_px + 1)
    desk2.run_cycle()
    assert trades(settings)[-1]["exit_reason"] == "tp"
    assert desk2.broker.open_orders() == []


def test_restart_without_engine_state_adopts_position_with_stop(settings, market, clock):
    desk1 = make_desk(settings, market, clock)
    desk1.submit("long SOL a mercado")
    desk1.broker.cancel_all("SOL")                        # se perdieron SL/TP...
    (settings.state_path / "engine.json").unlink()        # ...y el estado del motor
    out = Out()
    desk2 = make_desk(settings, market, clock, out=out)
    assert desk2.engine.brackets == {}
    desk2.run_cycle()
    br = desk2.engine.brackets["SOL"]
    pos = desk2.broker.account().positions["SOL"]
    assert br.origin == "adopted" and br.sl == pytest.approx(pos.entry_px * 0.985, abs=0.01)
    stops = [o for o in desk2.broker.open_orders("SOL") if o.kind is OrderKind.STOP]
    assert len(stops) == 1 and stops[0].trigger_px == br.sl and stops[0].reduce_only
    assert "adoptada" in out.text


def test_params_and_auto_flag_survive_restart(settings, market, clock):
    desk1 = make_desk(settings, market, clock, auto=True)
    assert "risk_per_trade_pct = 0.5" in desk1.submit("riesgo 0.5")
    assert "se redondeó" in desk1.submit("apalancamiento 2.5")
    assert "No se cambió" in desk1.submit("riesgo 50")
    desk1.submit("auto off")
    s2 = load_settings(env={}, dotenv_path=None, state_dir=settings.state_dir, llm_enabled=False)
    desk2 = TradingDesk(s2, market, paper_for(market, s2, clock), clock, strategies=[])
    assert s2.risk_per_trade_pct == 0.5 and s2.leverage == 2
    assert not desk2.auto_enabled
    assert any("risk_per_trade_pct" in p for p in desk2.restored_params)


# --------------------------------------------------------------------------- inbox / outbox
def test_inbox_offset_persists_and_partial_lines_wait(settings, market, clock):
    inbox = settings.state_path / "inbox.txt"
    settings.state_path.mkdir(parents=True, exist_ok=True)
    out = Out()
    desk1 = make_desk(settings, market, clock, auto=True, out=out)
    inbox.write_text("long SOL a mercado\n# comentario\n[mi-id] estado\nauto o", encoding="utf-8")
    desk1.run_cycle()
    assert "SOL" in desk1.broker.account().positions
    box = outbox(settings)
    assert "[c-000001] SOL LONG" in box and "[mi-id] Estado de la mesa" in box
    assert "» [c-000001] long SOL a mercado" in out.text
    assert desk1.auto_enabled                              # "auto o" aún sin salto de línea

    desk2 = make_desk(settings, market, clock, auto=True)  # reinicio: no se repite nada
    n_before = box.count("[c-")
    desk2.run_cycle()
    assert outbox(settings).count("[c-") == n_before
    with open(inbox, "a", encoding="utf-8") as f:
        f.write("ff\n")                                    # completa "auto off"
    desk2.run_cycle()
    assert not desk2.auto_enabled
    assert "[c-000002] Operativa automática DESACTIVADA" in outbox(settings)   # el contador sobrevive


def test_truncated_inbox_is_read_from_start(settings, market, clock):
    settings.state_path.mkdir(parents=True, exist_ok=True)
    inbox = settings.state_path / "inbox.txt"
    desk = make_desk(settings, market, clock)
    inbox.write_text("estado\nestado\n", encoding="utf-8")
    desk.run_cycle()
    inbox.write_text("pausa\n", encoding="utf-8")         # alguien vació el fichero
    desk.run_cycle()
    assert desk.risk.paused


def test_enqueue_is_processed_next_cycle(settings, market, clock):
    desk = make_desk(settings, market, clock)
    cid = desk.enqueue("long SOL ya")
    assert desk.broker.account().positions == {}
    desk.run_cycle()
    assert "SOL" in desk.broker.account().positions and f"[{cid}] SOL LONG" in outbox(settings)


# --------------------------------------------------------------------------- comandos interpretados por IA
def test_llm_open_requires_confirmation(settings, market, clock):
    settings.llm_enabled = True
    llm = FakeLLM(Command(CommandAction.OPEN, coin="SOL", side=LONG, entry_mode=EntryMode.MARKET, units=1.0))
    desk = make_desk(settings, market, clock, parser=CommandParser(settings, llm=llm))
    reply = desk.submit("quiero apostar por solana")
    assert llm.calls == ["quiero apostar por solana"]
    assert "Interpreté con IA" in reply and "abrir LONG SOL" in reply and "'si'" in reply
    assert desk.broker.account().positions == {}
    reply = desk.submit("si")
    assert reply.startswith("Confirmado") and "SOL LONG" in reply
    assert "SOL" in desk.broker.account().positions
    assert desk.submit("si").startswith("No hay ningún comando pendiente")


def test_llm_command_rejected_or_expired(settings, market, clock):
    settings.llm_enabled = True
    llm = FakeLLM(Command(CommandAction.KILL))
    desk = make_desk(settings, market, clock, parser=CommandParser(settings, llm=llm))
    # ("apágalo todo ya mismo" ya lo entienden las reglas como KILL: aquí hace falta texto que sólo entienda la IA)
    desk.submit("esto se hunde, sácame de todo")
    assert desk.submit("no").startswith("Descartado")
    assert not desk.risk.halted
    desk.submit("esto se hunde, sácame de todo")
    clock.advance(301)
    assert desk.submit("si").startswith("No hay ningún comando pendiente")
    assert not desk.risk.halted
    llm.cmd = Command(CommandAction.STATUS)                # lectura: sin confirmación
    assert desk.submit("¿cómo vamos hoy?").startswith("Estado de la mesa")


def test_unknown_and_help(settings, market, clock):
    desk = make_desk(settings, market, clock)
    assert desk.submit("blablá").startswith("No entendí")
    assert "Comandos" in desk.submit("ayuda")


def test_set_stop_by_percent_and_validation(settings, market, clock):
    desk = make_desk(settings, market, clock)
    desk.submit("long SOL a mercado")
    entry = desk.broker.account().positions["SOL"].entry_px
    reply = desk.submit("sl SOL 2%")
    assert "SL movido a" in reply
    assert desk.engine.brackets["SOL"].sl == pytest.approx(entry * 0.98, abs=0.01)
    assert "No se movió el TP" in desk.submit("tp SOL 100")
    assert "No hay posición" in desk.submit("sl ETH 2%")


# --------------------------------------------------------------------------- operativa automática
def test_auto_opens_on_signal_and_leaves_command_positions_alone(settings, market, clock):
    settings.symbols = ["SOL", "HYPE", "ETH"]
    out = Out()
    stub = StubStrategy(settings, clock, plan={"SOL": LONG, "HYPE": SHORT})
    desk = make_desk(settings, market, clock, strategies=[stub], auto=True, out=out)
    desk.submit("long HYPE a mercado")                     # posición abierta por comando
    desk.run_cycle()
    acct = desk.broker.account()
    assert acct.positions["SOL"].size > 0
    assert desk.engine.brackets["SOL"].origin == "auto:stub"
    assert acct.positions["HYPE"].size > 0 and desk.engine.brackets["HYPE"].origin == "command"
    assert "HYPE" not in stub.calls                        # ni siquiera se evalúa
    assert "AUTO [stub] SOL LONG" in out.text and "Motivo: señal de prueba" in out.text
    assert tags(desk.broker, "SOL") == ["sl", "tp"]
    # Una sola evaluación por vela cerrada: ETH (sin señal) no se re-evalúa hasta la vela siguiente.
    assert stub.calls.count("ETH") == 1
    clock.advance(60)
    desk.run_cycle()
    assert stub.calls.count("ETH") == 1
    clock.advance(15 * 60)
    desk.run_cycle()
    assert stub.calls.count("ETH") == 2 and "ETH" not in desk.broker.account().positions


def test_auto_respects_pause_and_auto_off(settings, market, clock):
    settings.symbols = ["SOL"]
    stub = StubStrategy(settings, clock, plan={"SOL": LONG})
    desk = make_desk(settings, market, clock, strategies=[stub], auto=True)
    desk.submit("pausa")
    desk.run_cycle()
    assert desk.broker.account().positions == {} and stub.calls == []
    desk.submit("reanuda")
    desk.submit("auto off")
    clock.advance(60)
    desk.run_cycle()
    assert desk.broker.account().positions == {} and stub.calls == []
    assert "ACTIVADA" in desk.submit("auto on")
    clock.advance(60)
    desk.run_cycle()
    assert "SOL" in desk.broker.account().positions


def test_auto_respects_max_open_positions(settings, market, clock):
    settings.symbols = ["BTC", "ETH", "SOL"]
    settings.max_open_positions = 1
    stub = StubStrategy(settings, clock, plan={"BTC": LONG, "ETH": LONG, "SOL": LONG})
    desk = make_desk(settings, market, clock, strategies=[stub], auto=True)
    desk.run_cycle()
    assert len(desk.broker.account().positions) == 1


# --------------------------------------------------------------------------- robustez del bucle
def test_data_errors_do_not_kill_the_loop(settings, market, clock):
    settings.symbols = ["SOL"]
    stub = StubStrategy(settings, clock, plan={"SOL": LONG})
    desk = make_desk(settings, market, clock, strategies=[stub], auto=True)
    market.fail_next = 1_000
    reply = desk.submit("long SOL ya")
    assert "fallo de conexión o de datos" in reply
    for _ in range(5):
        desk.run_cycle()                                   # no lanza
        clock.advance(60)
    assert desk.broker.account().positions == {}
    market.fail_next = 0
    clock.advance(15 * 60)
    desk.run_cycle()
    assert "SOL" in desk.broker.account().positions        # se recupera solo


def test_stage_failures_back_off_exponentially_and_recover(settings, market, clock, monkeypatch):
    desk = make_desk(settings, market, clock)
    original = desk.broker.account

    def down():
        raise BrokerError("timeout de red simulado")

    monkeypatch.setattr(desk.broker, "account", down)
    delays = []
    for _ in range(7):
        assert desk.run_cycle() is False
        delays.append(desk.next_delay())
    assert delays[:4] == [20.0, 40.0, 80.0, 160.0] and delays[-1] == MAX_BACKOFF_S
    assert "cuenta" in desk.stage_errors
    monkeypatch.setattr(desk.broker, "account", original)
    assert desk.run_cycle() is True
    assert desk.error_streak == 0 and desk.next_delay() == settings.loop_seconds


def test_buggy_strategy_does_not_kill_the_loop(settings, market, clock):
    settings.symbols = ["SOL"]
    stub = StubStrategy(settings, clock, error=RuntimeError("bug en la estrategia"))
    desk = make_desk(settings, market, clock, strategies=[stub], auto=True)
    assert desk.run_cycle() is False
    assert desk.stage_errors.get("estrategias") == 1
    assert desk.submit("long SOL ya").startswith("SOL LONG")   # los comandos siguen


def test_run_forever_with_failures_uses_backoff_and_finishes(settings, market, clock, monkeypatch):
    desk = make_desk(settings, market, clock)
    monkeypatch.setattr(desk.broker, "account", lambda: (_ for _ in ()).throw(BrokerError("caído")))
    t0 = clock.now()
    n = desk.run_forever(4, install_signals=False)
    assert n == 4
    assert clock.now() - t0 == pytest.approx(20 + 40 + 80)     # esperas entre los 4 ciclos


def test_sigterm_saves_state_and_exits_cleanly(settings, market, clock):
    out = Out()
    desk = make_desk(settings, market, clock, out=out)
    previous = signal.getsignal(signal.SIGTERM)

    def before(n):
        if n == 1:
            desk.submit("long SOL a mercado")
            os.kill(os.getpid(), signal.SIGTERM)

    n = desk.run_forever(100, before_cycle=before)
    assert n == 2
    assert signal.getsignal(signal.SIGTERM) == previous        # manejador restaurado
    assert "Mesa detenida" in out[-1]
    for name in ("desk.json", "risk.json", "engine.json", "status.json", "paper_broker.json"):
        assert (settings.state_path / name).exists(), name
    saved = json.loads((settings.state_path / "engine.json").read_text(encoding="utf-8"))
    assert "SOL" in saved["brackets"]


def test_status_line_and_snapshot(settings, market, clock):
    desk = make_desk(settings, market, clock)
    desk.submit("long SOL a mercado")
    desk.run_cycle()
    line = desk.status_line()
    for part in ("equity $", "PnL día", "posiciones 1: SOL L", "auto OFF", "halt no"):
        assert part in line, line
    snap = json.loads((settings.state_path / "status.json").read_text(encoding="utf-8"))
    assert "Estado de la mesa" in snap["report"] and snap["line"]
    assert (settings.state_path / "equity.csv").exists() and (settings.state_path / "heartbeat").exists()


# --------------------------------------------------------------------------- CLI
@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)
            h.close()
    for h in handlers:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(level)


def test_cli_fast_sim_run_send_and_status(tmp_path, capsys, restore_logging):
    from desk.__main__ import main
    state = tmp_path / "st"
    env = str(tmp_path / "no.env")
    rc = main(["run", "--data", "sim", "--fast", "--cycles", "40", "--state-dir", str(state), "--env-file", env,
               "--no-input", "--cmd", "0:long HYPE a mercado", "--cmd", "3:estado"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "HYPE LONG" in printed and "Estado de la mesa" in printed and "Sesión: 40 ciclos" in printed
    profile = json.loads((state / "profile.json").read_text(encoding="utf-8"))
    assert profile["clock"] == "sim" and profile["clock_ts"] > 1_767_225_600

    assert main(["status", "--state-dir", str(state)]) == 0
    assert "Estado de la mesa" in capsys.readouterr().out

    assert main(["send", "--no-wait", "--state-dir", str(state), "cierra", "todo"]) == 0
    line = (state / "inbox.txt").read_text(encoding="utf-8").strip()
    assert line.startswith("[cli-") and line.endswith("] cierra todo")

    rc = main(["run", "--data", "sim", "--fast", "--cycles", "2", "--state-dir", str(state), "--env-file", env,
               "--no-input"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "Cerrando todo" in printed
    assert json.loads((state / "profile.json").read_text(encoding="utf-8"))["clock_ts"] > profile["clock_ts"]

    with pytest.raises(SystemExit, match="otra configuración"):
        main(["run", "--data", "sim", "--state-dir", str(state), "--env-file", env, "--no-input", "--cycles", "1"])

    assert main(["run", "--data", "sim", "--fast", "--fresh", "--cycles", "1", "--state-dir", str(state),
                 "--env-file", env, "--no-input", "--no-auto"]) == 0
    assert not (state / "trades.csv").exists() and not (state / "inbox.txt").exists()
    assert json.loads((state / "profile.json").read_text(encoding="utf-8"))["clock_ts"] == 1_767_225_600


def test_cli_rejects_fast_with_live_data(tmp_path, restore_logging, capsys):
    from desk.__main__ import main
    assert main(["run", "--fast", "--data", "hyperliquid", "--state-dir", str(tmp_path),
                 "--env-file", str(tmp_path / "no.env")]) == 2
    assert main(["run", "--fresh", "--data", "sim", "--state-dir", str(tmp_path),
                 "--env-file", str(tmp_path / "no.env")]) == 2
    keep = tmp_path / "importante"
    keep.mkdir()
    (keep / "notas.txt").write_text("no borrar", encoding="utf-8")
    assert main(["run", "--fresh", "--fast", "--data", "sim", "--state-dir", str(keep), "--cycles", "1",
                 "--no-input", "--env-file", str(tmp_path / "no.env")]) == 2
    assert (keep / "notas.txt").exists()                     # --fresh no borra lo que no es una simulación


def test_env_example_documents_every_setting_with_defaults():
    from pathlib import Path

    from dotenv import dotenv_values

    from desk.config import _ENV_MAP, Settings
    path = Path(__file__).resolve().parents[1] / ".env.example"
    env = {k: v for k, v in dotenv_values(path).items() if v is not None}
    assert [k for k in _ENV_MAP if k not in env] == []
    loaded = load_settings(env=env, dotenv_path=None)
    defaults = Settings()
    assert {k: getattr(loaded, k) for k in vars(defaults)} == vars(defaults)

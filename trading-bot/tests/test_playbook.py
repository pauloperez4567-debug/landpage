"""Playbook: gráficas en automático (una por moneda + temporalidad), persistentes y a prueba de ficheros raros.

Sólo se adopta lo aprobado fuera de muestra (salvo que se fuerce) y una entrada rota nunca tumba la mesa.
"""

from __future__ import annotations

import json

import pytest

from desk.clock import SimClock
from desk.optimizer import optimize
from desk.playbook import PLAYBOOK_FILE, Playbook, PlaybookEntry, entry_from_report, validate_entry
from desk.strategy import DonchianBreakoutStrategy, EmaTrendStrategy
from desk.tui import normalize_playbook
from synthetic import CoinFlip, HourPattern, pattern_bars

T0 = 1_790_000_000.0
OOS = {"total_return_pct": 8.4, "profit_factor": 1.62, "max_drawdown_pct": 6.3, "n_trades": 41}


def entry(coin="BTC", interval="1h", strategy="ema_trend", params=None, units=None, **kw):
    return PlaybookEntry(coin=coin, interval=interval, strategy=strategy,
                         params={"fast": 12, "slow": 100} if params is None else params, units=units,
                         adopted_ts=kw.pop("adopted_ts", T0), report_id=kw.pop("report_id", "abc123"),
                         oos_metrics=kw.pop("oos_metrics", dict(OOS)), **kw)


@pytest.fixture
def pb(settings, clock):
    return Playbook(settings.state_dir, clock)


def test_adopt_persists_atomically_and_reloads(pb, settings, clock):
    assert pb.adopt(entry(), settings) is None
    assert pb.adopt(entry("ETH", "4h", "donchian_breakout", {"channel": 30}, units=2), settings) is None
    raw = json.loads((pb.path).read_text(encoding="utf-8"))
    assert raw["version"] == 1 and [e["coin"] for e in raw["entries"]] == ["BTC", "ETH"]
    assert not list(pb.path.parent.glob("*.tmp"))
    again = Playbook(settings.state_dir, clock)
    assert [e.to_dict() for e in again.entries()] == [e.to_dict() for e in pb.entries()]
    eth = again.get("eth", "4h")
    assert eth.units == 2.0 and eth.params == {"channel": 30} and eth.oos_metrics == OOS
    # el panel (``python -m desk dashboard``) lo lee tal cual
    norm = normalize_playbook(raw)
    assert [(e["coin"], e["interval"], e["strategy"]) for e in norm] == [("BTC", "1h", "ema_trend"),
                                                                       ("ETH", "4h", "donchian_breakout")]
    assert norm[0]["oos"]["pf"] == 1.62 and norm[1]["units"] == 2.0


def test_one_entry_per_chart_adopting_again_replaces(pb, settings):
    pb.adopt(entry(params={"fast": 9, "slow": 50}), settings)
    old = pb.adopt(entry(strategy="donchian_breakout", params={"channel": 20}), settings)
    assert old.strategy == "ema_trend" and old.params == {"fast": 9, "slow": 50}
    pb.adopt(entry(interval="15m", strategy="rsi_reversion", params={}), settings)
    pb.adopt(entry(coin="btc", interval="1h", strategy="squeeze_breakout", params={}), settings)   # misma gráfica
    assert sorted((e.coin.upper(), e.interval, e.strategy) for e in pb.entries()) == [
        ("BTC", "15m", "rsi_reversion"), ("BTC", "1h", "squeeze_breakout")]


@pytest.mark.parametrize("bad, words", [
    (dict(strategy="no_existe"), "desconocida"),
    (dict(interval="7h"), "intervalo"),
    (dict(params={"fast": 50, "slow": 20}), "fast < slow"),
    (dict(params={"inventado": 1}), "desconocido"),
    (dict(units=0), "unidades"),
    (dict(units=float("nan")), "unidades"),
    (dict(coin="  "), "moneda"),
])
def test_invalid_entries_are_refused_and_nothing_changes(pb, settings, bad, words):
    pb.adopt(entry(), settings)
    with pytest.raises(ValueError, match=words):
        pb.adopt(entry(**bad), settings)
    assert [e.strategy for e in pb.entries()] == ["ema_trend"]


def test_validate_entry_normalizes(settings):
    e = validate_entry(entry(coin=" xyz:GOLD ", strategy="EMA-Trend", units="1.5"), settings)
    assert e.coin == "xyz:GOLD" and e.strategy == "ema_trend" and e.units == 1.5


def test_remove_by_coin_or_chart(pb, settings):
    for coin, interval in (("BTC", "1h"), ("BTC", "4h"), ("ETH", "1h")):
        pb.adopt(entry(coin, interval), settings)
    assert pb.remove("btc", "4h") == 1
    assert pb.remove("BTC", "4h") == 0
    assert pb.remove("ETH") == 1 and pb.remove("SOL") == 0
    assert [(e.coin, e.interval) for e in pb.entries()] == [("BTC", "1h")]
    assert len(Playbook(settings.state_dir)) == 1


def test_strategies_use_each_entry_interval_and_params_and_are_reused(pb, settings, clock):
    pb.adopt(entry("BTC", "4h", "ema_trend", {"fast": 12, "slow": 100, "tp_r": 3.0}), settings)
    pb.adopt(entry("ETH", "1h", "donchian_breakout", {"channel": 30}), settings)
    pairs = pb.strategies(settings, clock)
    assert [(e.coin, type(s)) for e, s in pairs] == [("BTC", EmaTrendStrategy), ("ETH", DonchianBreakoutStrategy)]
    ema = pairs[0][1]
    assert ema.interval == "4h" and ema.fast == 12 and ema.slow == 100 and ema.tp_r == 3.0 and ema.clock is clock
    assert pairs[1][1].interval == "1h" and pairs[1][1].channel == 30
    # misma entrada -> misma instancia (conserva "una señal por vela"); entrada cambiada -> nueva
    again = pb.strategies(settings, clock)
    assert again[0][1] is ema and again[1][1] is pairs[1][1]
    pb.adopt(entry("BTC", "4h", "ema_trend", {"fast": 20, "slow": 100}), settings)
    third = dict((e.coin, s) for e, s in pb.strategies(settings, clock))
    assert third["BTC"] is not ema and third["BTC"].fast == 20 and third["ETH"] is pairs[1][1]
    # pausada -> fuera de la operativa, pero se recuerda
    assert pb.set_enabled("ETH", "1h", False) and not pb.set_enabled("SOL", "1h", True)
    assert [e.coin for e, _ in pb.strategies(settings, clock)] == ["BTC"]
    assert pb.get("ETH", "1h").enabled is False and "[PAUSADA]" in pb.summary_text()


def test_broken_file_entries_never_crash(settings, clock):
    path = settings.state_path / PLAYBOOK_FILE
    settings.state_path.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "entries": [
        {"coin": "BTC", "interval": "1h", "strategy": "ema_trend", "params": {"fast": 12, "slow": 100}},
        {"coin": "ETH", "interval": "1h", "strategy": "estrategia_borrada", "params": {}},
        {"coin": "SOL", "interval": "1h", "strategy": "ema_trend", "params": {"fast": 90, "slow": 10}},
        {"interval": "1h", "strategy": "ema_trend"}, "basura", 3,
    ]}), encoding="utf-8")
    pb = Playbook(settings.state_dir, clock)
    assert [e.coin for e in pb.entries()] == ["BTC", "ETH", "SOL"]          # sin moneda: fuera
    assert [e.coin for e, _ in pb.strategies(settings, clock)] == ["BTC"]   # las inválidas se saltan
    path.write_text("{ roto", encoding="utf-8")
    pb.reload()
    assert pb.entries() == [] and pb.strategies(settings, clock) == []
    assert list(settings.state_path.glob(f"{PLAYBOOK_FILE}.corrupt-*"))     # el ilegible se aparta, no se pierde
    path.write_text(json.dumps({"BTC|1h": {"coin": "BTC", "interval": "1h", "strategy": "ema_trend"}}),
                    encoding="utf-8")
    pb.reload()
    assert [e.coin for e in pb.entries()] == ["BTC"]


def test_failed_save_leaves_state_unchanged(pb, settings, monkeypatch):
    pb.adopt(entry(), settings)

    def boom(*_a, **_k):
        raise OSError("disco lleno")

    monkeypatch.setattr(pb._journal, "save_json", boom)
    with pytest.raises(OSError):
        pb.adopt(entry("ETH"), settings)
    with pytest.raises(OSError):
        pb.remove("BTC")
    assert [e.coin for e in pb.entries()] == ["BTC"]


def test_entries_are_copies(pb, settings):
    pb.adopt(entry(), settings)
    got = pb.entries()[0]
    got.params["fast"] = 999
    got.enabled = False
    assert pb.get("BTC", "1h").params["fast"] == 12 and pb.get("BTC", "1h").enabled


# --- desde un informe -----------------------------------------------------------------------

@pytest.fixture
def report(settings):
    return optimize("HYPE", "1h", pattern_bars(n=1200), settings, strategies=[HourPattern, CoinFlip],
                    clock=SimClock(T0))


def test_entry_from_report_takes_the_winner(report, pb, settings):
    clock = SimClock(T0 + 3600)
    e = entry_from_report(report, units=2, clock=clock)
    assert (e.coin, e.interval, e.strategy, e.params) == ("HYPE", "1h", "hour_pattern", {"hour": 7})
    assert e.report_id == report.id and e.adopted_ts == T0 + 3600 and e.units == 2 and e.enabled
    assert e.oos_metrics == report.get("hour_pattern").oos_metrics
    assert entry_from_report(report, "Hour-Pattern", clock=clock).strategy == "hour_pattern"
    # una estrategia de prueba no está en STRATEGIES: el playbook la rechaza al validar
    with pytest.raises(ValueError, match="desconocida"):
        pb.adopt(e, settings)


def test_entry_from_report_refuses_what_did_not_pass(report):
    with pytest.raises(ValueError, match="no aprobó fuera de muestra"):
        entry_from_report(report, "coin_flip")
    forced = entry_from_report(report, "coin_flip", allow_rejected=True, clock=SimClock(T0))
    assert forced.strategy == "coin_flip"
    with pytest.raises(ValueError, match="no se evaluó"):
        entry_from_report(report, "ema_trend")
    report.winner = None
    with pytest.raises(ValueError, match="no tiene ganadora"):
        entry_from_report(report)


def test_entry_from_report_never_adopts_a_failed_evaluation(settings):
    rep = optimize("BTC", "4h", pattern_bars(n=60, hour=3), settings, strategies=["ema_trend"], clock=SimClock(T0))
    with pytest.raises(ValueError, match="falló"):
        entry_from_report(rep, "ema_trend", allow_rejected=True)


def test_adopt_a_real_strategy_from_a_report_end_to_end(settings, clock):
    """Informe -> entrada -> playbook -> estrategia instanciada con los parámetros propuestos."""
    from conftest import make_candles

    rep = optimize("SOL", "1h", make_candles(n=900, interval="1h", vol_pct=0.6, seed=4), settings,
                   strategies=["donchian_breakout"], clock=SimClock(T0))
    pb = Playbook(settings.state_dir, clock)
    pb.adopt(entry_from_report(rep, "donchian_breakout", allow_rejected=True, clock=clock), settings)
    (e, strat), = pb.strategies(settings, clock)
    assert strat.params() == rep.get("donchian_breakout").params and strat.interval == "1h"


def test_summary_text(pb, settings):
    assert "Playbook vacío" in pb.summary_text()
    pb.adopt(entry(oos_metrics={**OOS, "profit_factor": 999.0}), settings)
    pb.adopt(entry("ETH", "4h", "donchian_breakout", {"channel": 30}, units=2, oos_metrics={}), settings)
    text = pb.summary_text()
    assert "2 gráfica(s) en automático" in text
    assert "BTC 1h · ema_trend (fast=12, slow=100)" in text and "+8.4 %" in text and "PF ∞" in text
    assert "ETH 4h · donchian_breakout (channel=30) · 2 u" in text and "orientativas" in text

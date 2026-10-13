"""Alarmas: cada tipo dispara y no dispara, cruces, repeat/cooldown, silencio, persistencia y anti-spam."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from conftest import FakeMarketData, make_candles
from desk.alerts import ZONE_MAX_AGE_S, AlertManager, AlertRule, describe_rule
from desk.indicators import rsi
from desk.models import Side, Zone

STEP = 3600   # velas de 1h


class Recorder:
    """Notifier falso: guarda cada envío."""

    def __init__(self):
        self.sent: list[dict] = []

    def send(self, text, level="info", kind="", coin=None, image=None):
        self.sent.append({"text": text, "level": level, "kind": kind, "coin": coin, "image": image})

    @property
    def texts(self):
        return [s["text"] for s in self.sent]


class CountingMarket(FakeMarketData):
    def __init__(self, clock):
        super().__init__(clock)
        self.candle_calls: list[tuple] = []
        self.mid_calls = 0

    def candles(self, coin, interval, limit):
        self.candle_calls.append((coin, interval, limit))
        return super().candles(coin, interval, limit)

    def mid(self, coin):
        self.mid_calls += 1
        return super().mid(coin)


class StubScanner:
    def __init__(self):
        self.zones_by_coin: dict[str, dict] = {}
        self.fail = False

    def zones(self, coin):
        if self.fail:
            raise RuntimeError("scanner roto")
        return self.zones_by_coin.get(coin, {Side.LONG: None, Side.SHORT: None})


@pytest.fixture
def notifier():
    return Recorder()


@pytest.fixture
def market(clock):
    return CountingMarket(clock)


@pytest.fixture
def mgr(settings, clock, notifier):
    return AlertManager(settings, clock, settings.state_dir, notifier)


def set_bars(market, clock, closes, volumes=None, coin="BTC", interval="1h", in_progress_close=None):
    """Velas de 1h cuya última vela cerrada termina justo ahora (+ opcionalmente la vela en curso)."""
    df = make_candles(closes=closes, interval=interval, end_ts=clock.now() - STEP)
    if volumes is not None:
        df["volume"] = np.asarray(volumes, dtype=float)
    if in_progress_close is not None:
        ts = pd.Timestamp(clock.now(), unit="s", tz="UTC")
        live = pd.DataFrame({"open": [closes[-1]], "high": [max(closes[-1], in_progress_close)],
                             "low": [min(closes[-1], in_progress_close)], "close": [in_progress_close],
                             "volume": [1e9]}, index=[ts])
        df = pd.concat([df, live])
    market.set_candles(coin, df, interval)


def next_bar(clock):
    clock.advance(STEP)


def read_log(settings):
    path = settings.state_path / "alerts.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


# =========================================================================== precio
def test_price_above_fires_when_price_crosses_up(mgr, market, notifier):
    market.set_price("BTC", 83_000)
    rule = mgr.add("btc", {"kind": "price_above", "level": 84_000}, ref_px=market.mid("BTC"))
    assert rule.coin == "BTC" and rule.id == 1
    assert mgr.evaluate(market) == []
    market.set_price("BTC", 84_100)
    fired = mgr.evaluate(market)
    assert len(fired) == 1 and "ALARMA #1 BTC" in fired[0] and "supera 84000" in fired[0]
    assert notifier.sent[0]["level"] == "warn" and notifier.sent[0]["kind"] == "price_above"
    assert notifier.sent[0]["coin"] == "BTC"
    assert mgr.rules() == []                        # una vez: se elimina
    assert mgr.evaluate(market) == []


def test_price_above_created_above_waits_for_a_new_cross(mgr, market):
    market.set_price("BTC", 85_000)
    mgr.add("BTC", {"kind": "price_above", "level": 84_000}, ref_px=85_000)
    assert mgr.evaluate(market) == []               # ya estaba por encima: no dispara al crear
    market.set_price("BTC", 85_500)
    assert mgr.evaluate(market) == []
    market.set_price("BTC", 83_900)                 # baja: se rearma
    assert mgr.evaluate(market) == []
    market.set_price("BTC", 84_001)
    assert len(mgr.evaluate(market)) == 1


def test_price_above_without_ref_uses_first_evaluation_as_reference(mgr, market):
    market.set_price("BTC", 85_000)
    mgr.add("BTC", {"kind": "price_above", "level": 84_000})
    assert mgr.evaluate(market) == []
    market.set_price("BTC", 86_000)
    assert mgr.evaluate(market) == []


def test_price_above_immediate_fires_if_already_true(mgr, market):
    market.set_price("BTC", 85_000)
    rule = mgr.add("BTC", {"kind": "price_above", "level": 84_000, "immediate": True}, ref_px=85_000)
    assert "ya cumplida" in describe_rule(rule)
    assert len(mgr.evaluate(market)) == 1


def test_price_level_touch_then_above_counts_as_cross(mgr, market):
    market.set_price("BTC", 84_000)                  # exactamente en el nivel: todavía no está "por encima"
    mgr.add("BTC", {"kind": "price_above", "level": 84_000})
    assert mgr.evaluate(market) == []
    market.set_price("BTC", 84_000.5)
    assert len(mgr.evaluate(market)) == 1


def test_price_below_fires_on_cross_down_only(mgr, market):
    market.set_price("ETH", 3_200)
    mgr.add("ETH", {"kind": "price_below", "level": 3_000}, ref_px=3_200)
    market.set_price("ETH", 3_100)
    assert mgr.evaluate(market) == []
    market.set_price("ETH", 2_990)
    fired = mgr.evaluate(market)
    assert len(fired) == 1 and "cae por debajo de 3000" in fired[0]


def test_price_cross_fires_both_ways_with_repeat(mgr, market):
    market.set_price("SOL", 150)
    rule = mgr.add("SOL", {"kind": "price_cross", "level": 155, "repeat": True, "cooldown_s": 0}, ref_px=150)
    market.set_price("SOL", 154)
    assert mgr.evaluate(market) == []
    market.set_price("SOL", 156)
    up = mgr.evaluate(market)
    market.set_price("SOL", 155)                     # en el nivel: no es un cruce
    assert mgr.evaluate(market) == []
    market.set_price("SOL", 153)
    down = mgr.evaluate(market)
    assert "al alza" in up[0] and "a la baja" in down[0]
    assert mgr.get(rule.id).fired_count == 2


def test_repeat_respects_cooldown(mgr, market, clock):
    market.set_price("BTC", 83_000)
    rule = mgr.add("BTC", {"kind": "price_above", "level": 84_000, "repeat": True, "cooldown_s": 600},
                   ref_px=83_000)
    market.set_price("BTC", 84_500)
    assert len(mgr.evaluate(market)) == 1
    clock.advance(60)
    market.set_price("BTC", 83_500)
    mgr.evaluate(market)
    market.set_price("BTC", 84_500)
    assert mgr.evaluate(market) == []               # cruce dentro de la pausa: descartado
    clock.advance(600)
    market.set_price("BTC", 83_500)
    mgr.evaluate(market)
    market.set_price("BTC", 84_500)
    assert len(mgr.evaluate(market)) == 1
    assert mgr.get(rule.id).fired_count == 2
    assert mgr.get(rule.id).last_fired_ts == clock.now()


def test_default_cooldown_comes_from_settings(settings, clock, notifier):
    settings.alert_cooldown_seconds = 42
    m = AlertManager(settings, clock, settings.state_dir, notifier)
    assert m.add("BTC", {"kind": "price_above", "level": 1}).cooldown_s == 42


# =========================================================================== RSI
def _rsi_drop_series():
    base = [100 + 0.3 * math.sin(i / 2.0) for i in range(120)]
    return base, base + [base[-1] - 6.0]


def test_rsi_below_fires_on_cross_of_a_closed_bar(mgr, market, clock):
    base, dropped = _rsi_drop_series()
    r = rsi(pd.Series(dropped), 14)
    assert r.iloc[-2] > 30 > r.iloc[-1]            # el escenario cruza de verdad
    set_bars(market, clock, base)
    mgr.add("BTC", {"kind": "rsi_below", "level": 30, "interval": "1h"})
    assert mgr.evaluate(market) == []
    next_bar(clock)
    set_bars(market, clock, dropped)
    fired = mgr.evaluate(market)
    assert len(fired) == 1 and "RSI(14) 1h" in fired[0] and "cae por debajo de 30" in fired[0]


def test_rsi_ignores_the_candle_in_progress(mgr, market, clock):
    base, _ = _rsi_drop_series()
    set_bars(market, clock, base)
    mgr.add("BTC", {"kind": "rsi_below", "level": 30})
    mgr.evaluate(market)
    next_bar(clock)
    # la vela cerrada es normal; la vela en curso se desploma: no cuenta hasta que cierre
    set_bars(market, clock, base + [base[-1]], in_progress_close=base[-1] - 20)
    assert mgr.evaluate(market) == []


def test_rsi_already_below_does_not_fire_without_immediate(mgr, market, clock):
    _, dropped = _rsi_drop_series()
    set_bars(market, clock, dropped)
    mgr.add("BTC", {"kind": "rsi_below", "level": 30})
    assert mgr.evaluate(market) == []
    next_bar(clock)
    set_bars(market, clock, dropped + [dropped[-1] - 0.5])
    assert mgr.evaluate(market) == []               # sigue por debajo: no hay cruce nuevo


def test_rsi_immediate_fires_if_condition_holds(mgr, market, clock):
    _, dropped = _rsi_drop_series()
    set_bars(market, clock, dropped)
    mgr.add("BTC", {"kind": "rsi_below", "level": 30, "immediate": True})
    assert len(mgr.evaluate(market)) == 1


def test_rsi_above_fires_on_cross_up(mgr, market, clock):
    base = [100 + 0.3 * math.sin(i / 2.0) for i in range(120)]
    set_bars(market, clock, base)
    mgr.add("BTC", {"kind": "rsi_above", "level": 70, "period": 14})
    assert mgr.evaluate(market) == []
    next_bar(clock)
    set_bars(market, clock, base + [base[-1] + 6.0])
    fired = mgr.evaluate(market)
    assert len(fired) == 1 and "sube por encima de 70" in fired[0]


def test_bar_closed_before_creation_does_not_count(mgr, market, clock):
    _, dropped = _rsi_drop_series()
    # la vela que cruza cerró justo al crear la alarma: es anterior, no dispara
    set_bars(market, clock, dropped)
    mgr.add("BTC", {"kind": "rsi_below", "level": 30})
    clock.advance(10)
    assert mgr.evaluate(market) == []


def test_bar_closing_after_creation_counts_even_if_first_evaluation_is_late(mgr, market, clock):
    base, dropped = _rsi_drop_series()
    set_bars(market, clock, base)
    mgr.add("BTC", {"kind": "rsi_below", "level": 30})
    next_bar(clock)                                 # primera evaluación ya con la vela nueva cerrada
    set_bars(market, clock, dropped)
    assert len(mgr.evaluate(market)) == 1


# =========================================================================== cruce de medias
def test_ma_cross_up_down_any(mgr, market, clock):
    falling = [10.0] * 10 + [9, 8, 7, 6, 5]
    set_bars(market, clock, falling)
    up = mgr.add("BTC", {"kind": "ma_cross", "fast": 2, "slow": 4, "direction": "up", "interval": "1h"})
    down = mgr.add("BTC", {"kind": "ma_cross", "fast": 2, "slow": 4, "direction": "down"})
    anyway = mgr.add("BTC", {"kind": "ma_cross", "fast": 2, "slow": 4, "ma": "ema"})
    assert mgr.evaluate(market) == []
    next_bar(clock)
    set_bars(market, clock, falling + [20])         # SMA2 12.5 > SMA4 9.5: cruce al alza
    fired = mgr.evaluate(market)
    ids = sorted(int(t.split("#")[1].split()[0]) for t in fired)
    assert ids == [up.id, anyway.id]
    assert any("SMA(2) cruza al alza la SMA(4) en 1h" in t for t in fired)
    assert mgr.get(down.id) is not None


def test_ma_cross_down(mgr, market, clock):
    rising = [10.0] * 10 + [11, 12, 13, 14, 15]
    set_bars(market, clock, rising)
    mgr.add("BTC", {"kind": "ma_cross", "fast": 2, "slow": 4, "direction": "down"})
    mgr.evaluate(market)
    next_bar(clock)
    set_bars(market, clock, rising + [1])
    fired = mgr.evaluate(market)
    assert len(fired) == 1 and "a la baja" in fired[0]


def test_ma_cross_no_fire_without_cross(mgr, market, clock):
    rising = [10.0 + i for i in range(20)]
    set_bars(market, clock, rising)
    mgr.add("BTC", {"kind": "ma_cross", "fast": 2, "slow": 4})
    mgr.evaluate(market)
    next_bar(clock)
    set_bars(market, clock, rising + [31])
    assert mgr.evaluate(market) == []


# =========================================================================== zona de liquidez
def _zone(side, low, high, coin="BTC"):
    return Zone(coin=coin, side=side, low=low, high=high, usd=2_500_000, source="liq")


def test_zone_fires_when_price_enters_previous_zone(mgr, market):
    sc = StubScanner()
    market.set_price("BTC", 100.0)
    sc.zones_by_coin["BTC"] = {Side.LONG: _zone(Side.LONG, 95, 97), Side.SHORT: _zone(Side.SHORT, 104, 106)}
    mgr.add("BTC", {"kind": "zone", "side": "long"})
    assert mgr.evaluate(market, sc) == []           # primera vez: sólo mira dónde están las zonas
    market.set_price("BTC", 98.0)
    assert mgr.evaluate(market, sc) == []
    market.set_price("BTC", 96.5)
    fired = mgr.evaluate(market, sc)
    assert len(fired) == 1 and "zona de liquidez LONG 95-97" in fired[0] and "$2,500,000.00" in fired[0]


def test_zone_side_filter_and_short_side(mgr, market):
    sc = StubScanner()
    market.set_price("BTC", 100.0)
    sc.zones_by_coin["BTC"] = {Side.LONG: _zone(Side.LONG, 95, 97), Side.SHORT: _zone(Side.SHORT, 104, 106)}
    only_long = mgr.add("BTC", {"kind": "zone", "side": Side.LONG})
    both = mgr.add("BTC", {"kind": "zone"})
    mgr.evaluate(market, sc)
    market.set_price("BTC", 104.5)
    fired = mgr.evaluate(market, sc)
    assert len(fired) == 1 and f"#{both.id}" in fired[0] and "SHORT" in fired[0]
    assert mgr.get(only_long.id) is not None


def test_zone_stale_snapshot_is_not_used(mgr, market, clock):
    sc = StubScanner()
    market.set_price("BTC", 100.0)
    sc.zones_by_coin["BTC"] = {Side.LONG: _zone(Side.LONG, 95, 97), Side.SHORT: None}
    mgr.add("BTC", {"kind": "zone", "side": "long"})
    mgr.evaluate(market, sc)
    clock.advance(ZONE_MAX_AGE_S + 1)
    market.set_price("BTC", 96.0)
    assert mgr.evaluate(market, sc) == []


def test_zone_uses_internal_scanner_when_none_given(mgr, market):
    market.set_price("BTC", 100.0)
    market.set_book("BTC", bids=[(99.5, 0.01), (98.0, 500.0)], asks=[(100.5, 0.01)])   # muro de compra en 98
    mgr.add("BTC", {"kind": "zone", "side": "long"})
    assert mgr.evaluate(market) == []
    market.set_price("BTC", 98.5)                   # se acerca pero no llega a la zona
    assert mgr.evaluate(market) == []
    market.set_price("BTC", 98.0)
    fired = mgr.evaluate(market)
    assert len(fired) == 1 and "zona de liquidez LONG" in fired[0] and "(book," in fired[0]


# =========================================================================== volumen y movimiento
def test_volume_spike_fires_and_not(mgr, market, clock):
    closes = [100.0] * 40
    set_bars(market, clock, closes, volumes=[1_000.0] * 40)
    mgr.add("BTC", {"kind": "volume_spike", "mult": 3, "lookback": 20})
    assert mgr.evaluate(market) == []
    next_bar(clock)
    set_bars(market, clock, closes + [100.0], volumes=[1_000.0] * 40 + [2_000.0])
    assert mgr.evaluate(market) == []               # 2x < 3x
    next_bar(clock)
    set_bars(market, clock, closes + [100.0, 100.0], volumes=[1_000.0] * 40 + [2_000.0, 4_000.0])
    fired = mgr.evaluate(market)
    assert len(fired) == 1 and "volumen 1h 4,000" in fired[0] and "x la media de 20 velas" in fired[0]


def test_volume_spike_before_creation_needs_immediate(mgr, market, clock):
    set_bars(market, clock, [100.0] * 30, volumes=[1_000.0] * 29 + [9_000.0])
    mgr.add("BTC", {"kind": "volume_spike"})
    assert mgr.evaluate(market) == []
    mgr.add("BTC", {"kind": "volume_spike", "immediate": True})
    assert len(mgr.evaluate(market)) == 1


def test_move_pct_fires_with_direction(mgr, market, clock):
    closes = [100.0] * 10
    set_bars(market, clock, closes)
    any_way = mgr.add("BTC", {"kind": "move_pct", "pct": 2, "interval": "1h"})
    down = mgr.add("BTC", {"kind": "move_pct", "pct": -2})          # negativo = caída
    big = mgr.add("BTC", {"kind": "move_pct", "pct": 5})
    assert mgr.get(down.id).params["direction"] == "down"
    mgr.evaluate(market)
    next_bar(clock)
    set_bars(market, clock, closes + [103.0])
    fired = mgr.evaluate(market)
    assert len(fired) == 1 and f"#{any_way.id}" in fired[0] and "+3.00%" in fired[0]
    assert mgr.get(down.id) is not None and mgr.get(big.id) is not None
    next_bar(clock)
    set_bars(market, clock, closes + [103.0, 99.0])
    fired = mgr.evaluate(market)
    assert len(fired) == 1 and f"#{down.id}" in fired[0] and "-3.88%" in fired[0]


# =========================================================================== caché y frecuencia
def test_candles_fetched_once_per_cycle_and_only_when_a_bar_can_close(mgr, market, clock):
    base, _ = _rsi_drop_series()
    set_bars(market, clock, base)
    mgr.add("BTC", {"kind": "rsi_below", "level": 30})
    mgr.add("BTC", {"kind": "rsi_above", "level": 70})
    mgr.add("BTC", {"kind": "ma_cross", "fast": 5, "slow": 20})
    mgr.add("ETH", {"kind": "price_above", "level": 5_000})
    mgr.add("ETH", {"kind": "price_below", "level": 1_000})
    mgr.evaluate(market)
    assert len(market.candle_calls) == 1              # tres alarmas de BTC 1h: una sola petición
    assert market.candle_calls[0][2] >= 140           # pide lo que necesita la más exigente (RSI)
    assert market.mid_calls == 1                      # dos alarmas de ETH: un solo precio
    for _ in range(5):                                # misma vela: no se piden velas otra vez
        clock.advance(60)
        mgr.evaluate(market)
    assert len(market.candle_calls) == 1
    next_bar(clock)
    set_bars(market, clock, base + [base[-1]])
    mgr.evaluate(market)
    assert len(market.candle_calls) == 2


def test_each_bar_is_evaluated_once(mgr, market, clock):
    base, dropped = _rsi_drop_series()
    set_bars(market, clock, base)
    mgr.add("BTC", {"kind": "rsi_below", "level": 30, "repeat": True, "cooldown_s": 0})
    mgr.evaluate(market)
    next_bar(clock)
    set_bars(market, clock, dropped)
    assert len(mgr.evaluate(market)) == 1
    clock.advance(30)
    assert mgr.evaluate(market) == []


# =========================================================================== silencio
def test_mute_silences_non_critical_but_logs_them(mgr, market, notifier, clock, settings):
    msg = mgr.mute(30)
    assert "30 min" in msg and mgr.is_muted()
    market.set_price("BTC", 83_000)
    mgr.add("BTC", {"kind": "price_above", "level": 84_000}, ref_px=83_000)
    market.set_price("BTC", 84_500)
    fired = mgr.evaluate(market)
    assert len(fired) == 1 and notifier.sent == []
    assert mgr.system("halt", "HALT por drawdown", level="critical") is True     # críticos: siempre
    assert mgr.system("info", "nota") is False
    assert [s["text"] for s in notifier.sent] == ["HALT por drawdown"]
    log = read_log(settings)
    assert [r.get("muted", False) for r in log] == [True, False, True]
    assert mgr.stats()["silenced"] == 2
    clock.advance(31 * 60)
    mgr.evaluate(market)                              # el silencio caduca solo y se avisa del recuento
    assert not mgr.is_muted()
    assert "Fin del silencio de avisos: 2 aviso(s)" in notifier.texts[-1]


def test_mute_until_unmute_and_unmute_message(mgr):
    assert "hasta 'unmute'" in mgr.mute(None)
    assert mgr.is_muted() and "silenciados hasta 'unmute'" in mgr.list_text()
    mgr.system("x", "uno")
    assert "1 silenciado" in mgr.unmute()
    assert not mgr.is_muted()
    assert mgr.unmute() == "Los avisos no estaban silenciados."
    with pytest.raises(ValueError):
        mgr.mute(-5)
    mgr.mute(10)
    assert "reactivados" in mgr.mute(0)


def test_critical_rule_alerts_are_never_muted(mgr, market, notifier):
    mgr.mute(None)
    market.set_price("BTC", 83_000)
    mgr.add("BTC", {"kind": "price_below", "level": 82_000}, ref_px=83_000)
    mgr.add("BTC", {"kind": "price_below", "level": 81_000, "severity": "info"}, ref_px=83_000)
    crit = mgr.add("BTC", {"kind": "price_below", "level": 80_000, "severity": "critical", "note": "stop mental"},
                   ref_px=83_000)
    assert crit.severity == "critical" and crit.params == {"level": 80_000.0}
    market.set_price("BTC", 79_000)
    assert len(mgr.evaluate(market)) == 3
    assert len(notifier.sent) == 1 and notifier.sent[0]["level"] == "critical"
    assert "stop mental" in notifier.sent[0]["text"]


# =========================================================================== avisos de la mesa
def test_system_anti_spam_by_key(mgr, notifier, clock, settings):
    settings.alert_cooldown_seconds = 300
    assert mgr.system("halt", "HALT diario", level="warn", key="halt") is True
    clock.advance(10)
    assert mgr.system("halt", "HALT diario", level="warn", key="halt") is False
    assert mgr.system("halt", "HALT diario", level="warn", key="halt") is False
    assert mgr.system("fill", "otra clave", key="fill:BTC", coin="BTC") is True
    clock.advance(300)
    assert mgr.system("halt", "HALT diario", level="warn", key="halt") is True
    assert notifier.texts == ["HALT diario", "otra clave", "HALT diario (+2 aviso(s) iguales omitidos)"]
    assert notifier.sent[1]["coin"] == "BTC" and notifier.sent[1]["kind"] == "fill"


def test_system_without_key_is_not_throttled(mgr, notifier):
    for _ in range(3):
        assert mgr.system("status", "línea de estado") is True
    assert len(notifier.sent) == 3


def test_system_critical_with_new_text_passes_cooldown(mgr, notifier):
    assert mgr.system("flatten", "SIN CERRAR: BTC", level="critical", key="flatten")
    assert not mgr.system("flatten", "SIN CERRAR: BTC", level="critical", key="flatten")
    assert mgr.system("flatten", "SIN CERRAR: BTC, ETH", level="critical", key="flatten")
    assert [s["level"] for s in notifier.sent] == ["critical", "critical"]


def test_system_logs_to_jsonl_with_contract_fields(mgr, settings, clock):
    mgr.system("halt", "HALT", level="error", coin="ETH")
    rec = read_log(settings)[0]
    assert {"ts", "level", "kind", "coin", "text", "rule_id"} <= set(rec)
    assert (rec["ts"], rec["level"], rec["kind"], rec["coin"], rec["text"], rec["rule_id"]) == \
        (clock.now(), "critical", "halt", "ETH", "HALT", None)


def test_system_key_memory_is_bounded(mgr, clock):
    from desk import alerts as mod
    for i in range(mod.SYSTEM_KEYS_KEPT + 50):
        mgr.system("k", f"t{i}", key=f"k{i}")
    assert len(mgr._sys_seen) <= mod.SYSTEM_KEYS_KEPT


# =========================================================================== persistencia
def test_rules_mute_and_cross_state_survive_restart(settings, clock, market, notifier):
    m1 = AlertManager(settings, clock, settings.state_dir, notifier)
    market.set_price("BTC", 83_000)
    a = m1.add("BTC", {"kind": "price_above", "level": 84_000, "note": "romper máximo"}, ref_px=83_000)
    b = m1.add("ETH", {"kind": "rsi_below", "level": 25, "interval": "4h", "repeat": True})
    m1.add("SOL", {"kind": "zone", "side": "short"})
    assert m1.remove(3) is True and m1.remove(3) is False
    m1.mute(None)

    m2 = AlertManager(settings, clock, settings.state_dir, Recorder())
    rules = m2.rules()
    assert [(r.id, r.coin, r.kind) for r in rules] == [(a.id, "BTC", "price_above"), (b.id, "ETH", "rsi_below")]
    assert rules[0].note == "romper máximo" and rules[0].state == {"s": -1}
    assert rules[1].interval == "4h" and rules[1].repeat is True and rules[1].params["period"] == 14
    assert m2.is_muted()
    assert m2.add("BTC", {"kind": "price_below", "level": 1}).id == 4        # los ids no se reutilizan
    m2.unmute()
    market.set_price("BTC", 84_100)                                          # la referencia sobrevivió
    fired = m2.evaluate(market)
    assert any("ALARMA #1 BTC" in t and "romper máximo" in t for t in fired)
    m3 = AlertManager(settings, clock, settings.state_dir, Recorder())
    assert [r.id for r in m3.rules()] == [b.id, 4]                           # la de una vez ya no está


def test_corrupt_state_starts_empty_and_is_set_aside(settings, clock, notifier):
    path = settings.state_path / "alerts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{no es json", encoding="utf-8")
    m = AlertManager(settings, clock, settings.state_dir, notifier)
    assert m.rules() == []
    assert any(p.name.startswith("alerts.json.corrupt-") for p in path.parent.iterdir())
    assert m.add("BTC", {"kind": "price_above", "level": 1}).id == 1


def test_bad_saved_rules_are_skipped(settings, clock):
    path = settings.state_path / "alerts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    good = AlertRule(id=7, coin="BTC", kind="price_above", params={"level": 10.0}).to_dict()
    path.write_text(json.dumps({"version": 1, "next_id": 2, "rules": [
        {"id": 3, "coin": "ETH", "kind": "teletransporte", "params": {}},
        {"id": 4, "coin": "ETH", "kind": "rsi_below", "params": {"level": 500}},
        {"id": 5, "coin": "ETH", "kind": "rsi_below", "params": {"level": 30}, "interval": "7x"},
        "basura", good]}), encoding="utf-8")
    m = AlertManager(settings, clock, settings.state_dir, Recorder())
    assert [r.id for r in m.rules()] == [7]
    assert m.add("BTC", {"kind": "price_below", "level": 1}).id == 8


def test_save_failure_does_not_raise(mgr, monkeypatch):
    def broken(*a, **k):
        raise OSError("disco lleno")
    monkeypatch.setattr(mgr.journal, "save_json", broken)
    rule = mgr.add("BTC", {"kind": "price_above", "level": 1})
    assert mgr.get(rule.id) is not None


def test_fired_alert_goes_to_jsonl_with_rule_id(mgr, market, settings):
    market.set_price("BTC", 83_000)
    rule = mgr.add("BTC", {"kind": "price_above", "level": 84_000}, ref_px=83_000)
    market.set_price("BTC", 85_000)
    mgr.evaluate(market)
    rec = read_log(settings)[-1]
    assert rec["rule_id"] == rule.id and rec["kind"] == "price_above" and rec["coin"] == "BTC"
    assert rec["level"] == "warn" and "ALARMA #1" in rec["text"]


# =========================================================================== validación y listado
@pytest.mark.parametrize("spec, match", [
    ({"kind": "teletransporte"}, "desconocido"),
    ({"kind": "price_above"}, "falta 'level'"),
    ({"kind": "price_above", "level": -3}, "level"),
    ({"kind": "price_above", "level": "abc"}, "level"),
    ({"kind": "rsi_below", "level": 150}, "RSI"),
    ({"kind": "rsi_below", "level": 30, "period": 1}, "period"),
    ({"kind": "ma_cross", "fast": 50, "slow": 20}, "menor"),
    ({"kind": "ma_cross", "ma": "wma"}, "sma o ema"),
    ({"kind": "ma_cross", "direction": "lateral"}, "dirección"),
    ({"kind": "zone", "side": "arriba"}, "lado"),
    ({"kind": "volume_spike", "mult": 0}, "mult"),
    ({"kind": "move_pct", "pct": 0}, "pct"),
    ({"kind": "rsi_below", "level": 30, "interval": "7x"}, "intervalo"),
    ({"kind": "price_above", "level": 1, "cooldown_s": -1}, "cooldown"),
    ("no es un dict", "inválida"),
])
def test_add_rejects_invalid_specs(mgr, spec, match):
    with pytest.raises(ValueError, match=match):
        mgr.add("BTC", spec)
    assert mgr.rules() == []


def test_add_requires_coin_and_respects_max_rules(settings, clock):
    m = AlertManager(settings, clock, settings.state_dir, None, max_rules=2)
    with pytest.raises(ValueError, match="moneda"):
        m.add("", {"kind": "price_above", "level": 1})
    m.add("BTC", {"kind": "price_above", "level": 1})
    m.add("BTC", {"kind": "price_above", "level": 2})
    with pytest.raises(ValueError, match="demasiadas"):
        m.add("BTC", {"kind": "price_above", "level": 3})
    assert m.clear() == 2 and m.rules() == []


def test_list_text_and_describe(mgr, clock):
    assert "No hay alarmas" in mgr.list_text()
    mgr.add("BTC", {"kind": "price_above", "level": 100_000})
    mgr.add("ETH", {"kind": "rsi_below", "level": 30, "interval": "4h", "repeat": True, "cooldown_s": 900})
    mgr.add("SOL", {"kind": "ma_cross", "fast": 9, "slow": 21, "ma": "ema", "direction": "up"})
    mgr.add("HYPE", {"kind": "zone"})
    mgr.add("BTC", {"kind": "volume_spike", "mult": 2.5})
    mgr.add("BTC", {"kind": "move_pct", "pct": 3, "interval": "15m"})
    text = mgr.list_text()
    assert "Alarmas activas (6):" in text
    for piece in ("#1 BTC precio > 100000 · una vez", "#2 ETH RSI(14) 4h < 30 · repetir (pausa 900 s)",
                  "cruce al alza EMA(9)/EMA(21) 1h", "#4 HYPE entra en zona de liquidez LONG o SHORT",
                  "volumen 1h >= 2.5x la media de 20 velas", "movimiento ±3% en una vela de 15m"):
        assert piece in text
    assert mgr.stats() == {"rules": 6, "fired_total": 0, "muted": False, "muted_until": None, "silenced": 0}


# =========================================================================== datos malos
class BadMarket(FakeMarketData):
    def __init__(self, clock, mode):
        super().__init__(clock)
        self.mode = mode

    def mid(self, coin):
        if self.mode == "raise":
            raise RuntimeError("API caída")
        if self.mode == "nan":
            return float("nan")
        if self.mode == "str":
            return "abc"
        return -5.0

    def candles(self, coin, interval, limit):
        if self.mode == "raise":
            raise RuntimeError("API caída")
        if self.mode == "nan":
            df = make_candles(n=200, interval=interval, end_ts=self.clock.now())
            df[:] = np.nan
            return df
        if self.mode == "str":
            return "no soy un DataFrame"
        return pd.DataFrame({"precio": [1, 2, 3]})


@pytest.mark.parametrize("mode", ["raise", "nan", "str", "weird"])
def test_evaluate_never_raises_with_bad_data(settings, clock, mode, caplog):
    m = AlertManager(settings, clock, settings.state_dir, Recorder())
    for spec in ({"kind": "price_above", "level": 1}, {"kind": "price_cross", "level": 1},
                 {"kind": "rsi_below", "level": 30}, {"kind": "ma_cross"}, {"kind": "zone"},
                 {"kind": "volume_spike"}, {"kind": "move_pct", "pct": 1}):
        m.add("BTC", spec)
    market = BadMarket(clock, mode)
    for _ in range(3):
        assert m.evaluate(market) == []
        clock.advance(STEP)
    assert len(m.rules()) == 7


def test_evaluate_survives_broken_scanner_notifier_and_odd_frames(settings, clock, market):
    class Exploding:
        def send(self, *a, **k):
            raise RuntimeError("notifier roto")

    m = AlertManager(settings, clock, settings.state_dir, Exploding())
    sc = StubScanner()
    sc.fail = True
    m.add("BTC", {"kind": "zone"})
    assert m.evaluate(market, sc) == []
    # velas desordenadas, duplicadas y con un hueco de NaN: se ordenan y se usan sin lanzar
    df = make_candles(n=60, interval="1h", end_ts=clock.now() - STEP)
    df = pd.concat([df.iloc[30:], df.iloc[:31]])
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    market.set_candles("BTC", df, "1h")
    m.add("BTC", {"kind": "move_pct", "pct": 0.01, "immediate": True})
    market.set_price("BTC", 83_000)
    m.add("BTC", {"kind": "price_above", "level": 84_000}, ref_px=83_000)
    market.set_price("BTC", 85_000)
    fired = m.evaluate(market, sc)                   # el notifier roto no impide marcar ni devolver
    assert any("supera" in t for t in fired)


def test_evaluate_with_no_rules_does_not_touch_market(mgr, market):
    assert mgr.evaluate(market) == []
    assert market.mid_calls == 0 and market.candle_calls == []

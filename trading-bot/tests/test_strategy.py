"""Estrategias autónomas con velas construidas a mano (``make_candles(closes=...)``).

``make_candles`` pone la apertura de la última vela en ``END``; con ``now = END +
duración`` todas las velas están cerradas. Cada caso negativo rompe una sola
condición del patrón que sí dispara.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_candles
from desk.clock import SimClock
from desk.indicators import atr
from desk.models import EntryMode, Side, Zone
from desk.strategy import (LiquiditySweepStrategy, SqueezeBreakoutStrategy, Strategy, build_strategies,
                           closed_candles, interval_seconds)

END = 1_767_225_600.0
M15, H1 = 900, 3600

# --- patrones de barrido (15m) --------------------------------------------------
# LONG: caída 105 -> 100, la vela anterior barre hasta 98.0 (low 97.90) y la última recupera a 98.6.
DECLINE = list(np.linspace(105.0, 100.0, 40))
LONG_SWEEP = DECLINE + [98.0, 98.6]
LONG_ZONE = Zone("HYPE", Side.LONG, 97.85, 97.96, 1_000_000.0, "liq", 0.7)
# SHORT (espejo): subida 95 -> 100, la vela anterior barre hasta 102.0 (high 102.10) y la última cae a 101.4.
RISE = list(np.linspace(95.0, 100.0, 40))
SHORT_SWEEP = RISE + [102.0, 101.4]
SHORT_ZONE = Zone("HYPE", Side.SHORT, 102.04, 102.15, 1_000_000.0, "liq", 0.7)

# --- patrones de squeeze (1h) ---------------------------------------------------
# Tendencia lenta (0.02 %/vela) con velas pequeñas = Bollinger dentro de Keltner; luego vela de +-2 %.
RAMP_UP = list(100.0 * 1.0002 ** np.arange(80))
RAMP_DOWN = list(100.0 * 0.9998 ** np.arange(80))
SQ_LONG = RAMP_UP + [RAMP_UP[-1] * 1.02]
SQ_SHORT = RAMP_DOWN + [RAMP_DOWN[-1] * 0.98]
# Rango lateral equilibrado (+DM ~ -DM): hay squeeze y ruptura, pero ADX ~12.
CYCLE = [100.0, 100.1, 100.0, 99.9] * 20


def candles(closes, interval="15m"):
    return make_candles(closes=closes, interval=interval, end_ts=END)


def sweep(closes, zone, mid, now=END + M15, settings=None, **kw):
    strat = LiquiditySweepStrategy(settings, **kw)
    return strat.evaluate("HYPE", candles(closes), mid, {zone.side: zone} if zone else {}, now=now)


def assert_bracket(sig, side, tp_r):
    if side is Side.LONG:
        assert sig.sl_price < sig.entry_px < sig.tp_price
    else:
        assert sig.tp_price < sig.entry_px < sig.sl_price
    assert abs(sig.tp_price - sig.entry_px) == pytest.approx(tp_r * abs(sig.entry_px - sig.sl_price))


# --- utilidades -----------------------------------------------------------------

def test_interval_seconds():
    assert interval_seconds("15m") == 900 and interval_seconds("1h") == 3600
    with pytest.raises(ValueError):
        interval_seconds("7m")


def test_closed_candles_drops_only_the_candle_in_progress():
    df = candles([1.0, 2.0, 3.0])
    assert len(closed_candles(df, "15m", now=END + M15)) == 3        # la última ya cerró
    assert len(closed_candles(df, "15m", now=END + M15 - 1)) == 2    # la última sigue en curso
    assert len(closed_candles(df, "15m", now=None)) == 2             # sin hora: se asume en curso
    assert closed_candles(df, "15m", now=None).index[-1] == df.index[-2]
    assert len(closed_candles(df.iloc[:0], "15m", now=END)) == 0


# --- LiquiditySweepStrategy -----------------------------------------------------

def test_sweep_long_fires_with_stop_below_the_zone(settings):
    sig = sweep(LONG_SWEEP, LONG_ZONE, mid=98.6, settings=settings)
    assert sig is not None
    assert sig.coin == "HYPE" and sig.side is Side.LONG and sig.entry_mode is EntryMode.MARKET
    assert sig.strategy == "liquidity_sweep" and "barrido" in sig.reason
    assert sig.entry_px == 98.6
    df = candles(LONG_SWEEP)
    atr_now = atr(df["high"], df["low"], df["close"], 14).iloc[-1]
    assert sig.sl_price == pytest.approx(LONG_ZONE.low - 0.5 * atr_now)
    assert sig.sl_price < LONG_ZONE.low
    assert_bracket(sig, Side.LONG, settings.default_tp_r)


def test_sweep_short_fires_with_stop_above_the_zone(settings):
    sig = sweep(SHORT_SWEEP, SHORT_ZONE, mid=101.4, settings=settings)
    assert sig is not None and sig.side is Side.SHORT and sig.entry_mode is EntryMode.MARKET
    df = candles(SHORT_SWEEP)
    atr_now = atr(df["high"], df["low"], df["close"], 14).iloc[-1]
    assert sig.sl_price == pytest.approx(SHORT_ZONE.high + 0.5 * atr_now)
    assert sig.sl_price > SHORT_ZONE.high
    assert_bracket(sig, Side.SHORT, settings.default_tp_r)


@pytest.mark.parametrize("closes,zone,mid", [
    # sin zona de liquidez
    (LONG_SWEEP, None, 98.6),
    # LONG: la vela anterior no llega a la zona (low 97.90 > 97.11)
    (LONG_SWEEP, Zone("HYPE", Side.LONG, 97.0, 97.11, 1e6, "liq"), 98.6),
    # LONG: perfora pero no recupera (cierre 98.2 < zone.low 98.25): ruptura, no barrido
    (DECLINE + [98.0, 98.2], Zone("HYPE", Side.LONG, 98.25, 98.36, 1e6, "liq"), 98.2),
    # LONG: RSI bajando (la última cierra por debajo de la anterior)
    (DECLINE + [100.5, 98.0, 97.97], LONG_ZONE, 97.97),
    # LONG: RSI alto (contexto alcista, > 40) aunque haya barrido y recuperación
    (list(np.linspace(90.0, 100.0, 40)) + [99.0, 99.6], Zone("HYPE", Side.LONG, 98.85, 98.95, 1e6, "liq"), 99.6),
    # LONG: el precio ya está por debajo del stop -> SL en el lado equivocado
    (LONG_SWEEP, LONG_ZONE, 97.5),
    # SHORT: la vela anterior no llega a la zona
    (SHORT_SWEEP, Zone("HYPE", Side.SHORT, 103.0, 103.11, 1e6, "liq"), 101.4),
    # SHORT: RSI bajo (contexto bajista, < 60)
    (list(np.linspace(110.0, 100.0, 40)) + [101.0, 100.4], Zone("HYPE", Side.SHORT, 101.05, 101.15, 1e6, "liq"),
     100.4),
    # SHORT: el precio ya está por encima del stop
    (SHORT_SWEEP, SHORT_ZONE, 103.0),
    # zona de otra moneda
    (LONG_SWEEP, Zone("BTC", Side.LONG, 97.85, 97.96, 1e6, "liq"), 98.6),
    # pocas velas para calentar indicadores
    (LONG_SWEEP[-20:], LONG_ZONE, 98.6),
])
def test_sweep_does_not_fire(settings, closes, zone, mid):
    assert sweep(closes, zone, mid, settings=settings) is None


def test_sweep_zone_dict_keyed_by_the_wrong_side_is_ignored(settings):
    strat = LiquiditySweepStrategy(settings)
    assert strat.evaluate("HYPE", candles(LONG_SWEEP), 98.6, {Side.SHORT: LONG_ZONE}, now=END + M15) is None


def test_sweep_ignores_the_candle_in_progress(settings):
    # el barrido + recuperación sólo existe si la última vela (aún abierta) cuenta
    assert sweep(LONG_SWEEP, LONG_ZONE, 98.6, now=END + M15 / 2, settings=settings) is None
    assert sweep(LONG_SWEEP, LONG_ZONE, 98.6, now=None, settings=settings) is None
    # con reloj la estrategia sabe que ya cerró
    assert sweep(LONG_SWEEP, LONG_ZONE, 98.6, now=None, settings=settings, clock=SimClock(END + M15)) is not None
    # sin hora, una vela extra al final se toma como la vela en curso y el patrón sí está cerrado
    assert sweep(LONG_SWEEP + [98.7], LONG_ZONE, 98.7, now=None, settings=settings) is not None


def test_sweep_skips_frozen_candles(settings):
    # 1 vela de retraso (falta la vela en curso): vale. Más de 1 vela + 120 s: datos congelados.
    assert sweep(LONG_SWEEP, LONG_ZONE, 98.6, now=END + 2 * M15, settings=settings) is not None
    assert sweep(LONG_SWEEP, LONG_ZONE, 98.6, now=END + 3 * M15, settings=settings) is None


def test_one_signal_per_closed_candle(settings):
    strat = LiquiditySweepStrategy(settings)
    df, zones = candles(LONG_SWEEP), {Side.LONG: LONG_ZONE}
    assert strat.evaluate("HYPE", df, 98.6, zones, now=END + M15) is not None
    assert strat.evaluate("HYPE", df, 98.6, zones, now=END + M15 + 60) is None   # misma vela, minuto siguiente
    other = Zone("SOL", Side.LONG, 97.85, 97.96, 1e6, "liq")
    assert strat.evaluate("SOL", df, 98.6, {Side.LONG: other}, now=END + M15) is not None   # por moneda


def test_malformed_candles_do_not_raise(settings):
    strat = LiquiditySweepStrategy(settings)
    df = candles(LONG_SWEEP)
    assert strat.evaluate("HYPE", df.iloc[:0], 98.6, {Side.LONG: LONG_ZONE}) is None
    assert strat.evaluate("HYPE", df.drop(columns=["low"]), 98.6, {Side.LONG: LONG_ZONE}) is None
    assert strat.evaluate("HYPE", None, 98.6, {Side.LONG: LONG_ZONE}) is None
    broken = df.copy()
    broken.iloc[-1, broken.columns.get_loc("close")] = np.nan
    assert strat.evaluate("HYPE", broken, 98.6, {Side.LONG: LONG_ZONE}, now=END + M15) is None


# --- SqueezeBreakoutStrategy ----------------------------------------------------

def squeeze_signal(closes, settings, mid="last", now=END + H1):
    df = candles(closes, "1h")
    ref = float(df["close"].iloc[-1]) if mid == "last" else mid
    return SqueezeBreakoutStrategy(settings).evaluate("BTC", df, ref, {}, now=now), df


@pytest.mark.parametrize("closes,side", [(SQ_LONG, Side.LONG), (SQ_SHORT, Side.SHORT)])
def test_squeeze_breakout_fires_in_breakout_direction(settings, closes, side):
    sig, df = squeeze_signal(closes, settings)
    assert sig is not None and sig.side is side and sig.entry_mode is EntryMode.MARKET
    assert sig.strategy == "squeeze_breakout" and "squeeze" in sig.reason
    assert sig.entry_px == pytest.approx(closes[-1])
    atr_now = atr(df["high"], df["low"], df["close"], 14).iloc[-1]
    assert abs(sig.entry_px - sig.sl_price) == pytest.approx(1.5 * atr_now)
    assert_bracket(sig, side, settings.default_tp_r)


@pytest.mark.parametrize("closes", [
    CYCLE + [102.0],                                     # squeeze liberado y cierre sobre BB, pero ADX ~12
    CYCLE + [98.0],                                      # ídem a la baja
    RAMP_UP + [RAMP_UP[-1] * 1.0002],                    # sigue en squeeze: no hay ruptura
    list(100.0 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 80)))) + [95.0],  # sin squeeze previo
    SQ_LONG[-40:],                                       # pocas velas
])
def test_squeeze_breakout_does_not_fire(settings, closes):
    sig, _ = squeeze_signal(closes, settings)
    assert sig is None


def test_squeeze_breakout_waits_for_the_breakout_candle_to_close(settings):
    assert squeeze_signal(SQ_LONG, settings, now=END + H1 / 2)[0] is None
    assert squeeze_signal(SQ_LONG, settings, now=None)[0] is None
    assert squeeze_signal(SQ_LONG, settings, now=END + H1)[0] is not None


def test_squeeze_uses_last_close_without_mid_and_live_tp_r(settings):
    settings.default_tp_r = 3.0                       # cambio en caliente ("tp 3")
    sig, _ = squeeze_signal(SQ_SHORT, settings, mid=None)
    assert sig.entry_px == pytest.approx(SQ_SHORT[-1])
    assert_bracket(sig, Side.SHORT, 3.0)


def test_squeeze_with_fake_market_and_clock(market, clock, settings):
    df = make_candles(closes=SQ_LONG, interval="1h", end_ts=clock.now() - H1)   # la ruptura cerró justo ahora
    market.set_candles("BTC", df, "1h")
    market.set_price("BTC", SQ_LONG[-1])
    strat = next(s for s in build_strategies(settings, clock) if s.name == "squeeze_breakout")
    bars = market.candles("BTC", strat.interval, strat.lookback)
    sig = strat.evaluate("BTC", bars, market.mid("BTC"), {Side.LONG: None, Side.SHORT: None})
    assert sig is not None and sig.side is Side.LONG
    assert_bracket(sig, Side.LONG, settings.default_tp_r)


# --- build_strategies -----------------------------------------------------------

def test_build_strategies_defaults(settings):
    strategies = build_strategies(settings)
    assert [s.name for s in strategies] == ["liquidity_sweep", "squeeze_breakout"]
    assert [s.interval for s in strategies] == ["15m", "1h"]
    assert all(isinstance(s, Strategy) and s.lookback > s.min_bars and s.clock is None for s in strategies)


def test_build_strategies_normalises_names_and_passes_clock(settings, clock):
    settings.strategies = ["Squeeze-Breakout", "squeeze_breakout", " liquidity_sweep ", ""]
    strategies = build_strategies(settings, clock)
    assert [s.name for s in strategies] == ["squeeze_breakout", "liquidity_sweep"]
    assert all(s.clock is clock and s.settings is settings for s in strategies)
    settings.strategies = []
    assert build_strategies(settings) == []


def test_build_strategies_rejects_unknown_names(settings):
    settings.strategies = ["liquidity_sweep", "martingala"]
    with pytest.raises(ValueError, match="martingala"):
        build_strategies(settings)

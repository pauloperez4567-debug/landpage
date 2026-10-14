"""Estrategias autónomas con velas construidas a mano (``make_candles(closes=...)``).

``make_candles`` pone la apertura de la última vela en ``END``; con ``now = END +
duración`` todas las velas están cerradas. Cada caso negativo rompe una sola
condición del patrón que sí dispara.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from desk.clock import SimClock
from desk.indicators import adx, atr, bollinger, ema
from desk.liquidity import historical_zones, row_zones, zones_frame
from desk.models import EntryMode, Side, Zone
from desk.strategy import (SIGNAL_COLUMNS, STRATEGIES, DonchianBreakoutStrategy, EmaTrendStrategy,
                           LiquiditySweepStrategy, RsiReversionStrategy, SqueezeBreakoutStrategy, Strategy,
                           build_strategies, closed_candles, create_strategy, interval_seconds)

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


# === Fase 2: una sola lógica (signals) para vivo y backtest ======================

ALL = sorted(STRATEGIES)


def random_bars(strat, n=1200, seed=5):
    """Paseo aleatorio en la temporalidad de la estrategia: con esta semilla todas disparan en ambos lados."""
    return make_candles(n=n, seed=seed, vol_pct=0.5, interval=strat.interval, end_ts=END)


def zones_for(strat, bars, settings):
    return historical_zones(bars, settings) if strat.uses_zones else None


def test_registry_has_the_new_strategies(settings):
    assert {"liquidity_sweep", "squeeze_breakout", "ema_trend", "rsi_reversion", "donchian_breakout"} <= set(ALL)
    assert isinstance(create_strategy("EMA-Trend", settings), EmaTrendStrategy)
    with pytest.raises(ValueError, match="martingala"):
        create_strategy("martingala", settings)


@pytest.mark.parametrize("name", ALL)
def test_signals_frame_shape(settings, name):
    strat = create_strategy(name, settings)
    bars = random_bars(strat)
    sig = strat.signals(bars, zones_for(strat, bars, settings))
    assert list(sig.columns) == SIGNAL_COLUMNS and sig.index.equals(bars.index)
    fired = sig[sig["side"] != ""]
    assert set(fired["side"]) == {"long", "short"}          # el paseo de prueba ejercita ambos lados
    assert set(fired["entry_mode"]) == {"market"}
    assert (sig.loc[sig["side"] == "", "entry_mode"] == "").all() and sig.loc[sig["side"] == "", "sl"].isna().all()
    longs, shorts = fired[fired["side"] == "long"], fired[fired["side"] == "short"]
    assert ((longs["sl"] < longs["entry_px"]) & (longs["entry_px"] < longs["tp"])).all()
    assert ((shorts["tp"] < shorts["entry_px"]) & (shorts["entry_px"] < shorts["sl"])).all()
    assert (fired["reason"].str.len() > 10).all()
    assert (sig["side"].iloc[:strat.min_bars - 1] == "").all()      # sin calentar no hay señal
    # datos inservibles: nunca lanza
    assert (strat.signals(bars.iloc[:0])["side"] == "").all()
    assert (strat.signals(bars.drop(columns=["low"]))["side"] == "").all()
    assert len(strat.signals(None)) == 0


@pytest.mark.parametrize("name", ALL)
def test_live_evaluate_equals_the_backtest_row(settings, name):
    """Paridad exacta: en cada vela, ``evaluate`` en vivo == la fila de ``signals`` sobre toda la historia."""
    strat = create_strategy(name, settings)
    bars = random_bars(strat)
    zones = zones_for(strat, bars, settings)
    full = strat.signals(bars, zones)
    step = interval_seconds(strat.interval)
    fired = np.flatnonzero(full["side"].to_numpy() != "")
    quiet = np.flatnonzero(full["side"].to_numpy() == "")
    rows = list(fired[:12]) + list(quiet[strat.min_bars::97][:8]) + [strat.min_bars - 2]
    assert len(fired) >= 4
    for i in rows:
        live = create_strategy(name, settings)                    # instancia nueva: sin memoria de la vela
        zd = row_zones(zones, i, "BTC") if zones is not None else {}
        sig = live.evaluate("BTC", bars.iloc[:i + 1], None, zd, now=bars.index[i].timestamp() + step)
        row = full.iloc[i]
        if row["side"] == "":
            assert sig is None, i
            continue
        assert sig is not None, i
        assert (sig.side.value, sig.entry_mode.value, sig.strategy) == (row["side"], row["entry_mode"], name)
        assert (sig.entry_px, sig.sl_price, sig.tp_price) == (row["entry_px"], row["sl"], row["tp"])
        assert sig.reason == row["reason"] and sig.confidence == row["confidence"]


@pytest.mark.parametrize("name", ALL)
def test_live_signal_is_the_last_row_of_signals_with_the_mid(settings, name):
    """Con mid distinto del cierre, la señal viva es la última fila de ``signals(..., ref_px=mid)``."""
    strat = create_strategy(name, settings)
    bars = random_bars(strat)
    zones = zones_for(strat, bars, settings)
    full = strat.signals(bars, zones)
    i = int(np.flatnonzero(full["side"].to_numpy() != "")[-1])
    window, mid = bars.iloc[:i + 1], float(bars["close"].iloc[i]) * 1.0003
    zd = row_zones(zones, i, "BTC") if zones is not None else {}
    zdf = zones_frame(window.index, zd, "BTC") if strat.uses_zones else None
    expected = strat.signals(window, zdf, ref_px=mid).iloc[-1]
    sig = create_strategy(name, settings).evaluate("BTC", window, mid, zd,
                                                   now=window.index[-1].timestamp() + interval_seconds(strat.interval))
    if expected["side"] == "":
        assert sig is None
    else:
        assert sig.side.value == expected["side"] and sig.entry_px == mid
        assert (sig.sl_price, sig.tp_price) == (expected["sl"], expected["tp"])


def _scramble_future(bars, k, seed):
    """Copia con las velas posteriores a ``k`` sustituidas por otro paseo (precio y volumen)."""
    rng = np.random.default_rng(seed)
    out = bars.copy()
    m = len(out) - k - 1
    factor = np.exp(np.cumsum(rng.normal(0.002, 0.01, m)))
    for col in ("open", "high", "low", "close"):
        out.iloc[k + 1:, out.columns.get_loc(col)] = out[col].iloc[k + 1:].to_numpy() * factor
    out.iloc[k + 1:, out.columns.get_loc("volume")] = rng.uniform(10, 10_000, m)
    return out


@pytest.mark.parametrize("name", ALL)
def test_signals_never_look_ahead(settings, name):
    """Cambiar (o quitar) velas futuras no cambia ninguna señal pasada; las zonas se recalculan también."""
    strat = create_strategy(name, settings)
    bars = random_bars(strat)
    base = strat.signals(bars, zones_for(strat, bars, settings))
    for n_cut, k in enumerate((len(bars) // 3, len(bars) // 2, 2 * len(bars) // 3)):
        altered = _scramble_future(bars, k, seed=n_cut)
        alt = strat.signals(altered, zones_for(strat, altered, settings))
        pd.testing.assert_frame_equal(base.iloc[:k + 1], alt.iloc[:k + 1])
        assert not alt.iloc[k + 1:].equals(base.iloc[k + 1:])      # el cambio sí afecta al futuro
        cut = bars.iloc[:k + 1]
        pd.testing.assert_frame_equal(base.iloc[:k + 1], strat.signals(cut, zones_for(strat, cut, settings)))
    # Cortes en muchas velas: las que disparan, la anterior a cada una (una fuga suele adelantar la señal
    # una vela) y una muestra regular. La última fila de la historia cortada es la fila de la completa.
    fired = np.flatnonzero(base["side"].to_numpy() != "")[:25]
    rows = sorted({*fired.tolist(), *(fired - 1).tolist(), *range(strat.min_bars, len(bars), 41)})
    for k in rows:
        cut = bars.iloc[:k + 1]
        last = strat.signals(cut, zones_for(strat, cut, settings)).iloc[-1]
        assert last.equals(base.iloc[k]), k


# --- parámetros -------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_param_grid_is_valid_and_params_roundtrip(settings, name):
    cls = STRATEGIES[name]
    assert cls.PARAM_GRID and set(cls.PARAM_GRID) <= set(cls.PARAMS)
    keys = list(cls.PARAM_GRID)
    for combo in itertools.product(*(cls.PARAM_GRID[k] for k in keys)):   # todas las combinaciones construyen
        strat = cls(settings, **dict(zip(keys, combo)))
        assert strat.lookback > strat.min_bars >= cls.min_bars
        assert all(strat.params()[k] == v for k, v in zip(keys, combo))
    strat = cls(settings)
    params = strat.params()
    assert list(params) == list(cls.PARAMS) and json.loads(json.dumps(params)) == params
    assert cls(settings, **params).params() == params


@pytest.mark.parametrize("name,kwargs", [
    ("ema_trend", {"martingala": 2}),             # nombre desconocido
    ("ema_trend", {"fast": 50, "slow": 20}),      # fast >= slow
    ("ema_trend", {"fast": 9.5}),                 # periodo no entero
    ("ema_trend", {"fast": True}),
    ("rsi_reversion", {"rsi_low": 80.0}),         # rsi_low >= rsi_high
    ("donchian_breakout", {"channel": 1}),
    ("squeeze_breakout", {"adx_min": float("nan")}),
    ("liquidity_sweep", {"tp_r": 0}),
    ("liquidity_sweep", {"sl_atr_mult": "mucho"}),
])
def test_invalid_params_raise(settings, name, kwargs):
    with pytest.raises(ValueError):
        create_strategy(name, settings, **kwargs)


def test_interval_and_params_from_the_constructor(settings, clock):
    strat = EmaTrendStrategy(settings, clock, interval="4h", fast=12, slow=200, tp_r=3)
    assert strat.interval == "4h" and strat.clock is clock
    assert strat.params()["fast"] == 12 and isinstance(strat.params()["fast"], int)
    assert strat.params()["tp_r"] == 3.0 and isinstance(strat.params()["tp_r"], float)
    assert strat.min_bars > 200 and strat.lookback >= strat.min_bars + 4 * 200   # calienta la EMA lenta
    assert EmaTrendStrategy.interval == "1h"                                      # la clase no cambia
    with pytest.raises(ValueError):
        EmaTrendStrategy(settings, interval="7m")


def test_default_warmup_of_existing_strategies_is_unchanged(settings):
    assert (LiquiditySweepStrategy(settings).lookback, LiquiditySweepStrategy(settings).min_bars) == (120, 30)
    assert (SqueezeBreakoutStrategy(settings).lookback, SqueezeBreakoutStrategy(settings).min_bars) == (150, 50)


def test_tp_r_param_overrides_the_live_default(settings):
    sig = SqueezeBreakoutStrategy(settings, tp_r=1.5).evaluate("BTC", candles(SQ_LONG, "1h"), None, {},
                                                               now=END + H1)
    assert_bracket(sig, Side.LONG, 1.5)
    settings.default_tp_r = 3.0                                   # no le afecta: tiene su propio tp_r
    sig = SqueezeBreakoutStrategy(settings, tp_r=1.5).evaluate("BTC", candles(SQ_LONG, "1h"), None, {},
                                                               now=END + H1)
    assert_bracket(sig, Side.LONG, 1.5)


def test_parameters_change_the_signals(settings):
    bars = make_candles(n=1200, seed=5, vol_pct=0.5, interval="4h", end_ts=END)
    a = DonchianBreakoutStrategy(settings, channel=10).signals(bars)
    b = DonchianBreakoutStrategy(settings, channel=55).signals(bars)
    assert (a["side"] != "").sum() > (b["side"] != "").sum() > 0


# --- estrategias nuevas con patrones construidos a mano -----------------------------

EMA_UP = list(np.linspace(110, 100, 120)) + list(np.linspace(100, 112, 60))     # bajada y giro alcista
EMA_DOWN = list(np.linspace(100, 110, 120)) + list(np.linspace(110, 98, 60))


@pytest.mark.parametrize("closes,side", [(EMA_UP, Side.LONG), (EMA_DOWN, Side.SHORT)])
def test_ema_trend_fires_on_the_cross_bar(settings, closes, side):
    df = candles(closes, "1h")
    sig = EmaTrendStrategy(settings).signals(df)
    rows = np.flatnonzero(sig["side"].to_numpy() != "")
    fast, slow = ema(df["close"], 20), ema(df["close"], 50)
    diff = (fast - slow).to_numpy()
    crosses = np.flatnonzero((np.sign(diff[1:]) != np.sign(diff[:-1])) & np.isfinite(diff[:-1])) + 1
    assert list(rows) == list(crosses) and len(rows) == 1
    i = int(rows[0])
    assert adx(df["high"], df["low"], df["close"], 14).iloc[i] > 20
    # evaluate en esa vela: misma señal, SL a 2 ATR y TP a default_tp_r R
    s = EmaTrendStrategy(settings).evaluate("ETH", df.iloc[:i + 1], None, {}, now=df.index[i].timestamp() + H1)
    assert s is not None and s.side is side and "cruce" in s.reason
    a = atr(df["high"], df["low"], df["close"], 14).iloc[i]
    assert abs(s.entry_px - s.sl_price) == pytest.approx(2.0 * a)
    assert_bracket(s, side, settings.default_tp_r)
    # con un filtro de ADX imposible no hay señal
    assert (EmaTrendStrategy(settings, adx_min=99.0).signals(df)["side"] == "").all()


WAVE = [100 + 0.3 * np.sin(i / 2) for i in range(60)]
RSI_DROP = WAVE + [99.5, 98.8, 98.0, 97.2]
RSI_POP = WAVE + [100.5, 101.2, 102.0, 102.8]


@pytest.mark.parametrize("closes,side", [(RSI_DROP, Side.LONG), (RSI_POP, Side.SHORT)])
def test_rsi_reversion_targets_the_mean(settings, closes, side):
    df = candles(closes)
    s = RsiReversionStrategy(settings).evaluate("SOL", df, None, {}, now=END + M15)
    assert s is not None and s.side is side and s.entry_mode is EntryMode.MARKET and "RSI" in s.reason
    up, mid, lo = bollinger(df["close"], 20, 2.0)
    assert s.tp_price == pytest.approx(mid.iloc[-1])                     # objetivo: la media
    assert abs(s.entry_px - s.sl_price) == pytest.approx(1.5 * atr(df["high"], df["low"], df["close"], 14).iloc[-1])
    if side is Side.LONG:
        assert df["low"].iloc[-1] <= lo.iloc[-1] and s.sl_price < s.entry_px < s.tp_price
    else:
        assert df["high"].iloc[-1] >= up.iloc[-1] and s.tp_price < s.entry_px < s.sl_price


def test_rsi_reversion_needs_room_to_the_mean_and_an_extreme(settings):
    df = candles(RSI_DROP)
    assert RsiReversionStrategy(settings, min_rr=10.0).evaluate("SOL", df, None, {}, now=END + M15) is None
    assert RsiReversionStrategy(settings, rsi_low=5.0).evaluate("SOL", df, None, {}, now=END + M15) is None
    assert RsiReversionStrategy(settings).evaluate("SOL", candles(WAVE), None, {}, now=END + M15) is None


RANGE = [100 + (0.8 if i % 2 else -0.8) for i in range(40)]


@pytest.mark.parametrize("closes,side", [(RANGE + [102.0, 102.5], Side.LONG), (RANGE + [98.0, 97.5], Side.SHORT)])
def test_donchian_breakout_fires_once_on_the_first_close_outside(settings, closes, side):
    df = candles(closes, "4h")
    sig = DonchianBreakoutStrategy(settings).signals(df)
    assert list(np.flatnonzero(sig["side"].to_numpy() != "")) == [40]   # la ruptura, no la vela siguiente
    row = sig.iloc[40]
    assert row["side"] == side.value and "Donchian 20" in row["reason"]
    a = atr(df["high"], df["low"], df["close"], 14).iloc[40]
    assert abs(row["entry_px"] - row["sl"]) == pytest.approx(2.0 * a)
    assert abs(row["tp"] - row["entry_px"]) == pytest.approx(settings.default_tp_r * 2.0 * a)


def test_liquidity_sweep_without_zones_never_fires(settings):
    strat = LiquiditySweepStrategy(settings)
    assert (strat.signals(candles(LONG_SWEEP)).side == "").all()
    assert strat.evaluate("HYPE", candles(LONG_SWEEP), 98.6, {}, now=END + M15) is None

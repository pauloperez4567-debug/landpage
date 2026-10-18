"""Motor de backtest: reglas de ejecución con señales a mano, dimensionamiento como el riesgo en vivo,
métricas sobre operaciones construidas a mano y rendimiento.

Las señales se fijan con ``Scripted`` (una estrategia cuyo ``signals`` devuelve un plan), así cada
ejecución, comisión y PnL se calcula a mano. Ajustes de los casos a mano: deslizamiento 10 bps,
taker 5 bps, maker 2 bps, 1 unidad = 500 USD a 5x (notional 2500), riesgo 10 % (no recorta).
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from desk.backtest import (EXIT_REASONS, METRIC_KEYS, PF_CAP, BacktestResult, BTTrade, compute_metrics,
                           position_size, run_backtest)
from desk.clock import SimClock
from desk.liquidity import historical_zones
from desk.models import AccountState, AssetMeta, EntryMode, Side, TradeIntent
from desk.risk import RiskManager
from desk.strategy import STRATEGIES, Strategy, create_strategy, empty_signals

META = AssetMeta("TEST", sz_decimals=2, max_leverage=10)
H1 = 3600


class Scripted(Strategy):
    """Señales fijas: ``plan = {vela: (lado, modo, entry_px, sl, tp)}``."""

    name = "scripted"
    interval = "1h"
    lookback = 10
    min_bars = 1

    def __init__(self, settings, plan):
        super().__init__(settings)
        self.plan = plan

    def signals(self, bars, zones=None, *, ref_px=None):
        out = empty_signals(bars.index)
        for i, (side, mode, px, sl, tp) in self.plan.items():
            for col, val in (("side", side), ("entry_mode", mode), ("entry_px", px), ("sl", sl), ("tp", tp),
                             ("reason", "plan"), ("confidence", 0.5)):
                out.iloc[i, out.columns.get_loc(col)] = val
        return out


def bars_of(rows):
    """Velas de 1h a mano: filas (open, high, low, close)."""
    idx = pd.date_range("2026-03-01", periods=len(rows), freq="1h", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["volume"] = 1_000.0
    return df


def ts(bars, i):
    return bars.index[i].timestamp()


@pytest.fixture
def bt_settings(settings):
    settings.paper_slippage_bps = 10.0
    settings.paper_taker_fee_bps = 5.0
    settings.paper_maker_fee_bps = 2.0
    settings.risk_per_trade_pct = 10.0
    settings.leverage = 5
    settings.unit_margin_usd = 500.0
    settings.auto_units = 1.0
    settings.paper_start_equity = 10_000.0
    return settings


def run(settings, rows, plan, **kw):
    bars = bars_of(rows)
    kw.setdefault("meta", META)
    return run_backtest(Scripted(settings, plan), bars, settings, coin="TEST", **kw), bars


FLAT = (100.0, 101.0, 99.0, 100.0)
LONG_MKT = ("long", "market", 100.0, 95.0, 110.0)
SHORT_MKT = ("short", "market", 100.0, 105.0, 90.0)


# --- ejecución a mercado, SL/TP, comisiones exactas ----------------------------------

def test_market_long_to_take_profit_exact_fills_fees_and_equity(bt_settings):
    rows = [FLAT, FLAT, (100.0, 103.0, 99.5, 102.0), (102.0, 111.0, 101.0, 108.0), (108.0, 109.0, 107.0, 108.0)]
    res, bars = run(bt_settings, rows, {1: LONG_MKT})
    assert len(res.trades) == 1
    t = res.trades[0]
    entry, exit_ = 100.0 * 1.001, 110.0 * 0.999          # open + 10 bps / TP − 10 bps
    fee_in, fee_out = 25 * entry * 0.0005, 25 * exit_ * 0.0005
    pnl = (exit_ - entry) * 25 - fee_in - fee_out
    assert (t.side, t.exit_reason, t.size, t.bars_held) == ("long", "tp", 25.0, 2)
    assert (t.entry_ts, t.exit_ts) == (ts(bars, 2), ts(bars, 3))
    assert t.entry_px == pytest.approx(entry) and t.exit_px == pytest.approx(exit_)
    assert t.fees == pytest.approx(fee_in + fee_out, abs=1e-12)
    assert t.pnl == pytest.approx(pnl, abs=1e-9)
    assert t.pnl_pct == pytest.approx(pnl / 10_000 * 100)
    assert t.r_multiple == pytest.approx(pnl / (25 * (entry - 95.0)))
    eq = res.equity.to_numpy()
    assert eq[:2].tolist() == [10_000.0, 10_000.0]
    assert eq[2] == pytest.approx(10_000 - fee_in + (102.0 - entry) * 25)    # marcado a mercado, sin comisión de salida
    assert eq[3] == pytest.approx(10_000 + pnl) and eq[4] == eq[3]
    assert res.metrics["fees_usd"] == pytest.approx(fee_in + fee_out)
    assert res.metrics["n_trades"] == 1 and res.metrics["n_signals"] == 1 and res.metrics["n_skipped"] == 0


def test_market_short_mirror(bt_settings):
    rows = [FLAT, FLAT, (100.0, 101.0, 98.0, 99.0), (99.0, 99.5, 89.0, 92.0)]
    res, _ = run(bt_settings, rows, {1: SHORT_MKT})
    t = res.trades[0]
    entry, exit_ = 100.0 * 0.999, 90.0 * 1.001
    assert (t.side, t.exit_reason) == ("short", "tp")
    assert t.pnl == pytest.approx((entry - exit_) * 25 - 25 * (entry + exit_) * 0.0005)


def test_candle_touching_sl_and_tp_takes_the_stop_first(bt_settings):
    rows = [FLAT, FLAT, (100.0, 103.0, 99.5, 102.0), (102.0, 111.0, 94.0, 100.0), FLAT]
    t = run(bt_settings, rows, {1: LONG_MKT})[0].trades[0]
    assert t.exit_reason == "sl" and t.exit_px == pytest.approx(95.0 * 0.999)


def test_gap_through_stop_fills_at_the_worse_open(bt_settings):
    rows = [FLAT, FLAT, (100.0, 103.0, 99.5, 102.0), (93.0, 94.0, 92.0, 93.5), FLAT]
    t = run(bt_settings, rows, {1: LONG_MKT})[0].trades[0]
    assert t.exit_reason == "sl" and t.exit_px == pytest.approx(93.0 * 0.999)    # no al SL (95): al open
    assert t.r_multiple < -1.3


def test_gap_through_take_profit_gets_no_bonus(bt_settings):
    rows = [FLAT, FLAT, (100.0, 103.0, 99.5, 102.0), (112.0, 113.0, 111.0, 112.0), FLAT]
    t = run(bt_settings, rows, {1: LONG_MKT})[0].trades[0]
    assert t.exit_reason == "tp" and t.exit_px == pytest.approx(110.0 * 0.999)   # al TP, no al open 112


def test_stop_on_the_entry_candle(bt_settings):
    rows = [FLAT, FLAT, (100.0, 101.0, 94.0, 96.0), FLAT]
    t = run(bt_settings, rows, {1: LONG_MKT})[0].trades[0]
    assert (t.exit_reason, t.bars_held) == ("sl", 1)
    rows = [FLAT, FLAT, (100.0, 111.0, 99.0, 108.0), FLAT]       # a mercado en el open: el TP de la vela vale
    assert run(bt_settings, rows, {1: LONG_MKT})[0].trades[0].exit_reason == "tp"


@pytest.mark.parametrize("open_px", [94.0, 111.0])
def test_signal_dropped_when_the_next_open_is_beyond_sl_or_tp(bt_settings, open_px):
    rows = [FLAT, FLAT, (open_px, open_px + 1, open_px - 1, open_px), FLAT]
    res, _ = run(bt_settings, rows, {1: LONG_MKT})
    assert res.trades == [] and res.metrics["n_skipped"] == 1
    assert res.equity.iloc[-1] == 10_000.0


def test_open_position_is_closed_at_the_end(bt_settings):
    rows = [FLAT, FLAT, (100.0, 103.0, 99.5, 102.0), (102.0, 104.0, 101.0, 103.0)]
    res, bars = run(bt_settings, rows, {1: LONG_MKT})
    t = res.trades[0]
    assert t.exit_reason == "end" and t.exit_px == pytest.approx(103.0 * 0.999) and t.exit_ts == ts(bars, 3)
    assert res.equity.iloc[-1] == pytest.approx(10_000 + t.pnl)


def test_signal_on_the_last_candle_is_not_traded(bt_settings):
    res, _ = run(bt_settings, [FLAT, FLAT, FLAT], {2: LONG_MKT})
    assert res.trades == [] and res.metrics["n_signals"] == 0


# --- entradas límite / zona --------------------------------------------------------

LONG_ZONE = ("long", "zone", 98.0, 95.0, 104.0)


def test_zone_entry_fills_on_touch_as_maker_and_ignores_tp_on_the_fill_candle(bt_settings):
    rows = [FLAT, FLAT,
            (100.0, 105.0, 99.0, 101.0),    # 2: no toca 98 (el TP aquí no cuenta: aún no hay posición)
            (101.0, 104.5, 97.5, 103.0),    # 3: toca 98 -> ejecuta; toca 104 pero en la vela de entrada no cuenta
            (103.0, 105.0, 102.0, 104.0)]   # 4: TP
    res, bars = run(bt_settings, rows, {1: LONG_ZONE})
    t = res.trades[0]
    size = 25.51                                          # floor(2500 / 98, 2 decimales)
    assert (t.size, t.entry_px, t.entry_ts, t.exit_ts, t.exit_reason) == (size, 98.0, ts(bars, 3), ts(bars, 4), "tp")
    fee_in, fee_out = size * 98.0 * 0.0002, size * 104.0 * 0.999 * 0.0005    # maker / taker
    assert t.fees == pytest.approx(fee_in + fee_out, abs=1e-12)
    assert t.pnl == pytest.approx((104.0 * 0.999 - 98.0) * size - fee_in - fee_out, abs=1e-9)


def test_zone_entry_stop_on_the_fill_candle(bt_settings):
    rows = [FLAT, FLAT, (100.0, 101.0, 99.0, 100.0), (101.0, 102.0, 94.0, 96.0), FLAT]
    t = run(bt_settings, rows, {1: LONG_ZONE})[0].trades[0]
    assert (t.entry_px, t.exit_reason, t.bars_held) == (98.0, "sl", 1)


def test_zone_entry_expires_after_the_ttl(bt_settings):
    bt_settings.entry_ttl_minutes = 120                   # colocada al cierre de la vela 1: valen las velas 2 y 3
    touch = (100.0, 101.0, 97.0, 99.0)
    res, _ = run(bt_settings, [FLAT, FLAT, FLAT, FLAT, touch, FLAT], {1: LONG_ZONE})
    assert res.trades == [] and res.metrics["n_skipped"] == 1
    res, _ = run(bt_settings, [FLAT, FLAT, FLAT, touch, FLAT, FLAT], {1: LONG_ZONE})
    assert len(res.trades) == 1
    bt_settings.entry_ttl_minutes = 30                    # menos de una vela: ninguna vela entra entera
    res, _ = run(bt_settings, [FLAT, FLAT, touch, FLAT], {1: LONG_ZONE})
    assert res.trades == []


# --- una posición a la vez, giro opcional -------------------------------------------

def test_signals_while_in_a_position_are_ignored_and_flip_is_optional(bt_settings):
    rows = [FLAT, FLAT, (100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 98.0), FLAT]
    plan = {1: LONG_MKT, 2: LONG_MKT, 3: SHORT_MKT}
    res, _ = run(bt_settings, rows, plan)
    assert [(t.side, t.exit_reason) for t in res.trades] == [("long", "end")]
    res, bars = run(bt_settings, rows, plan, allow_flip=True)
    assert [(t.side, t.exit_reason) for t in res.trades] == [("long", "flip"), ("short", "end")]
    flip = res.trades[0]
    assert flip.exit_ts == ts(bars, 4) and flip.exit_px == pytest.approx(100.0 * 0.999) and flip.bars_held == 2
    assert res.trades[1].entry_ts == ts(bars, 4)


# --- dimensionamiento como el riesgo en vivo ----------------------------------------

@pytest.mark.parametrize("changes,sl", [
    ({}, 95.0),                                          # unidad completa (riesgo 10 %)
    ({"risk_per_trade_pct": 1.0}, 95.0),                 # recortado por riesgo en el SL (con comisiones)
    ({"risk_per_trade_pct": 1.0}, 99.3),                 # SL cerca: el recorte por riesgo no actúa, sí la unidad
    ({"leverage": 20}, 85.0),                            # SL lejos: apalancamiento bajado por la liquidación
    ({"max_notional_per_coin_usd": 800.0}, 95.0),        # máximo por moneda
    ({"auto_units": 9.0, "unit_margin_usd": 3_000.0}, 99.0),   # unidades máximas y máximo por moneda
    ({"auto_units": 9.0, "unit_margin_usd": 3_000.0, "max_notional_per_coin_usd": 1e6}, 99.0),  # exposición y margen
])
def test_position_size_equals_the_live_risk_manager(bt_settings, changes, sl):
    for k, v in changes.items():
        setattr(bt_settings, k, v)
    meta = AssetMeta("TEST", sz_decimals=2, max_leverage=20)
    res, _ = run(bt_settings, [FLAT, FLAT, (100.0, 100.5, 99.5, 100.0), FLAT], {1: ("long", "market", 100.0, sl, 120.0)},
                 meta=meta)
    risk = RiskManager(bt_settings, SimClock())
    intent = TradeIntent(coin="TEST", side=Side.LONG, units=bt_settings.auto_units, entry_mode=EntryMode.MARKET,
                         origin="auto:scripted", sl_price=sl, tp_price=120.0)
    decision = risk.evaluate(intent, AccountState(equity=10_000.0, available=10_000.0), meta, 100.0, 100.0, 0.0)
    assert decision.approved, decision.reason
    assert res.trades[0].size == decision.size
    assert position_size(bt_settings, meta, 10_000.0, 100.0, sl) == (decision.size, decision.leverage)


def test_position_below_the_minimum_notional_is_skipped(bt_settings):
    bt_settings.unit_margin_usd = 1.0                      # 5 USD de notional < 10 USD mínimo
    res, _ = run(bt_settings, [FLAT, FLAT, FLAT, FLAT], {1: LONG_MKT})
    assert res.trades == [] and res.metrics["n_skipped"] == 1
    risk = RiskManager(bt_settings, SimClock())
    intent = TradeIntent(coin="TEST", side=Side.LONG, units=1.0, entry_mode=EntryMode.MARKET, origin="auto:x",
                         sl_price=95.0, tp_price=110.0)
    assert not risk.evaluate(intent, AccountState(equity=10_000.0, available=10_000.0), META, 100.0, 100.0, 0.0).approved
    assert position_size(bt_settings, META, 10_000.0, 100.0, 95.0) == (0.0, 0)


def test_sizing_follows_the_equity(bt_settings):
    bt_settings.risk_per_trade_pct = 1.0
    rows = [FLAT, FLAT, (100.0, 101.0, 94.0, 96.0), FLAT, (100.0, 100.5, 99.5, 100.0), FLAT]
    res, _ = run(bt_settings, rows, {1: LONG_MKT, 3: LONG_MKT})
    first, second = res.trades
    assert first.exit_reason == "sl" and second.size < first.size       # menos equity -> menos tamaño


# --- métricas sobre operaciones construidas a mano ------------------------------------

def trade(pnl, r, pct, held, fees=1.0):
    return BTTrade("X", "long", 0.0, 100.0, 1.0, 101.0, 1.0, pnl, pct, r, fees, "tp" if pnl > 0 else "sl", held)


def test_metrics_on_hand_built_trades_and_curve():
    trades = [trade(1_000.0, 1.0, 10.0, 1), trade(-1_100.0, -1.0, -10.0, 1), trade(990.0, 0.9, 10.0, 1)]
    idx = pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC")
    equity = pd.Series([11_000.0, 9_900.0, 10_890.0], index=idx)
    m = compute_metrics(trades, equity, 10_000.0, "1d", n_bars=3)
    assert set(METRIC_KEYS) <= set(m)
    assert m["n_trades"] == 3 and isinstance(m["n_trades"], int)
    assert m["total_return_pct"] == pytest.approx(8.9)
    assert m["cagr_pct"] == pytest.approx((1.089 ** (365.25 / 3) - 1) * 100)
    rets = np.array([0.1, -0.1, 0.1])
    per_year = math.sqrt(365.25)
    assert m["sharpe"] == pytest.approx(rets.mean() / rets.std(ddof=1) * per_year)
    assert m["sortino"] == pytest.approx(rets.mean() / math.sqrt(0.01 / 3) * per_year)
    assert m["max_drawdown_pct"] == pytest.approx(10.0)
    assert m["calmar"] == pytest.approx(m["cagr_pct"] / 10.0)
    assert m["win_rate_pct"] == pytest.approx(200 / 3)
    assert m["profit_factor"] == pytest.approx(1_990 / 1_100)
    assert m["expectancy_r"] == pytest.approx(0.3)
    assert m["avg_trade_pct"] == pytest.approx(10 / 3)
    assert (m["best_trade_pct"], m["worst_trade_pct"]) == (10.0, -10.0)
    assert m["avg_bars_held"] == 1.0 and m["exposure_pct"] == pytest.approx(100.0)
    assert m["fees_usd"] == pytest.approx(3.0)


def test_metrics_edge_cases_are_finite():
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    flat = pd.Series([10_000.0] * 4, index=idx)
    m = compute_metrics([], flat, 10_000.0, "1h")
    assert all(m[k] == 0 for k in METRIC_KEYS)
    wins_only = [trade(50.0, 0.5, 0.5, 2), trade(30.0, 0.3, 0.3, 2)]
    m = compute_metrics(wins_only, pd.Series([10_000.0, 10_050.0, 10_050.0, 10_080.0], index=idx), 10_000.0, "1h")
    assert m["profit_factor"] == PF_CAP and m["win_rate_pct"] == 100.0 and m["max_drawdown_pct"] == 0.0
    assert m["sortino"] == 0.0 and m["calmar"] == 0.0 and m["exposure_pct"] == 100.0
    ruin = compute_metrics([trade(-10_000.0, -1.0, -100.0, 1)], pd.Series([0.0], index=idx[:1]), 10_000.0, "1h")
    assert ruin["cagr_pct"] == -100.0 and ruin["max_drawdown_pct"] == 100.0
    assert all(math.isfinite(v) for v in ruin.values())


# --- estrategias reales: invariantes, paridad de tramos, serialización ---------------------

@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_real_strategies_accounting_invariants(settings, name):
    strat = create_strategy(name, settings)
    bars = make_candles(n=2500, seed=5, vol_pct=0.5, interval=strat.interval)
    meta = AssetMeta("BTC", sz_decimals=3, max_leverage=20)
    res = run_backtest(strat, bars, settings, coin="BTC", meta=meta)
    assert res.trades, "la estrategia debería operar en este paseo"
    assert (res.strategy, res.interval, res.params, res.bars) == (name, strat.interval, strat.params(), len(bars))
    assert res.equity.index.equals(bars.index)
    assert res.equity.iloc[-1] == pytest.approx(settings.paper_start_equity + sum(t.pnl for t in res.trades))
    assert res.metrics["fees_usd"] == pytest.approx(sum(t.fees for t in res.trades))
    assert {t.exit_reason for t in res.trades} <= set(EXIT_REASONS) - {"flip"}
    for prev, nxt in zip(res.trades, res.trades[1:]):
        assert nxt.entry_ts >= prev.exit_ts                    # una posición a la vez
    for t in res.trades:
        assert t.size > 0 and t.bars_held >= 1 and t.fees > 0
        assert t.size == round(t.size, 3)
    assert all(math.isfinite(v) for v in res.metrics.values())


def test_backtest_never_looks_ahead(settings):
    strat = create_strategy("donchian_breakout", settings)
    bars = make_candles(n=2000, seed=5, vol_pct=0.5, interval=strat.interval)
    k = 1200
    changed = bars.copy()
    changed.iloc[k + 1:, :4] = changed.iloc[k + 1:, :4].to_numpy() * 1.05
    a = run_backtest(strat, bars, settings, coin="BTC")
    b = run_backtest(strat, changed, settings, coin="BTC")
    cutoff = bars.index[k].timestamp()
    done_a = [t for t in a.trades if t.exit_ts <= cutoff]
    assert done_a and done_a == [t for t in b.trades if t.exit_ts <= cutoff]
    assert a.equity.iloc[:k].equals(b.equity.iloc[:k])


def test_trade_from_and_precomputed_signals(settings):
    strat = create_strategy("ema_trend", settings)
    bars = make_candles(n=3000, seed=5, vol_pct=0.5, interval=strat.interval)
    sig = strat.signals(bars)
    k = 2000
    res = run_backtest(strat, bars, settings, coin="BTC", trade_from=k, signals=sig)
    assert res.bars == 1000 and res.equity.index[0] == bars.index[k] and res.start_ts == bars.index[k].timestamp()
    assert res.equity.iloc[0] == settings.paper_start_equity          # la primera entrada posible es en k + 1
    assert all(t.entry_ts > bars.index[k].timestamp() for t in res.trades)
    same = run_backtest(strat, bars, settings, coin="BTC", trade_from=bars.index[k])
    assert same.trades == res.trades and same.equity.equals(res.equity)
    # señales calculadas sobre una historia más larga con el mismo inicio: valen para un tramo anterior
    head = run_backtest(strat, bars.iloc[:k], settings, coin="BTC", signals=sig)
    assert head.trades == run_backtest(strat, bars.iloc[:k], settings, coin="BTC").trades
    with pytest.raises(ValueError):
        run_backtest(strat, bars, settings, coin="BTC", signals=sig.iloc[1:])


def test_liquidity_sweep_backtest_uses_historical_zones(settings):
    strat = create_strategy("liquidity_sweep", settings)
    bars = make_candles(n=2500, seed=5, vol_pct=0.5, interval=strat.interval)
    res = run_backtest(strat, bars, settings, coin="BTC")
    assert res.trades
    given = run_backtest(strat, bars, settings, coin="BTC", zones=historical_zones(bars, settings))
    assert given.trades == res.trades


def test_result_to_dict_is_json_and_sampled(settings):
    strat = create_strategy("rsi_reversion", settings)
    bars = make_candles(n=3000, seed=5, vol_pct=0.5, interval=strat.interval)
    res = run_backtest(strat, bars, settings, coin="SOL")
    d = json.loads(json.dumps(res.to_dict(), allow_nan=False))
    assert len(d["equity"]) <= 500
    assert d["equity"][0] == [bars.index[0].timestamp(), res.equity.iloc[0]]
    assert d["equity"][-1] == [bars.index[-1].timestamp(), res.equity.iloc[-1]]
    assert d["metrics"] == res.metrics and len(d["trades"]) == len(res.trades)
    back = BacktestResult.from_dict(d)
    assert back.trades == res.trades and back.metrics == res.metrics and back.params == res.params
    assert back.equity.iloc[-1] == res.equity.iloc[-1] and len(back.equity) == len(d["equity"])


def test_bad_inputs(settings):
    strat = create_strategy("ema_trend", settings)
    bars = make_candles(n=300, interval="1h")
    with pytest.raises(ValueError):
        run_backtest(strat, bars.drop(columns=["close"]), settings, coin="BTC")
    with pytest.raises(ValueError):
        run_backtest(strat, bars.reset_index(drop=True), settings, coin="BTC")
    empty = run_backtest(strat, bars.iloc[:0], settings, coin="BTC")
    assert empty.trades == [] and empty.bars == 0 and empty.metrics["total_return_pct"] == 0.0
    nan_tail = bars.copy()
    nan_tail.iloc[-3:, :4] = np.nan                          # velas rotas al final: no rompe la contabilidad
    res = run_backtest(strat, nan_tail, settings, coin="BTC")
    assert np.isfinite(res.equity.to_numpy()).all()


# --- rendimiento ------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_5000_bars_backtest_under_100ms(settings, name):
    """Contrato: 5000 velas con un juego de parámetros en < 100 ms (señales + zonas + simulación)."""
    strat = create_strategy(name, settings)
    bars = make_candles(n=5000, seed=3, vol_pct=0.4, interval=strat.interval)
    meta = AssetMeta("BTC", sz_decimals=5, max_leverage=40)
    run_backtest(strat, bars, settings, coin="BTC", meta=meta)          # calentamiento (imports, cachés)
    best = math.inf
    for _ in range(3):
        t0 = time.perf_counter()
        run_backtest(strat, bars, settings, coin="BTC", meta=meta)
        best = min(best, time.perf_counter() - t0)
    assert best < 0.100, f"{name}: {best * 1000:.1f} ms"


# --- frenos del riesgo en vivo (halt por drawdown, pérdida diaria, cooldown) leídos de settings ---------------

LOSS = (100.0, 100.5, 94.0, 95.0)


def test_drawdown_halt_closes_and_stops_trading_like_the_live_risk(bt_settings):
    """Con ``max_drawdown_pct`` = 2 % la segunda pérdida cruza el límite a mitad de vela: se cierra ahí ("halt",
    antes que el SL) y la señal siguiente no se opera. Sin frenos (entrenamiento) se opera todo."""
    rows = [FLAT, LOSS, FLAT, LOSS, FLAT, (100.0, 101.0, 99.0, 100.0), FLAT]
    plan = {0: LONG_MKT, 2: LONG_MKT, 4: LONG_MKT}
    bt_settings.max_drawdown_pct, bt_settings.daily_loss_limit_pct, bt_settings.max_consecutive_losses = 2.0, 50.0, 0
    free, _ = run(bt_settings, rows, plan)
    assert [t.exit_reason for t in free.trades] == ["sl", "sl", "end"] and "halted" not in free.metrics
    res, bars = run(bt_settings, rows, plan, brakes=True)
    assert [t.exit_reason for t in res.trades] == ["sl", "halt"]
    halt = res.trades[1]
    assert 95.0 < halt.exit_px < 100.0                         # el freno salta antes que el SL (95)
    assert res.metrics["halted"] == 1.0 and res.metrics["halt_ts"] == ts(bars, 3) and res.metrics["n_blocked"] == 1
    assert res.metrics["max_drawdown_pct"] == pytest.approx(2.0, abs=0.1)
    # El mismo backtest con los límites por defecto (DD 2.7 % < 5 %) no se detiene: los frenos salen de settings.
    bt_settings.max_drawdown_pct = 5.0
    res5, _ = run(bt_settings, rows, plan, brakes=True)
    assert [t.exit_reason for t in res5.trades] == ["sl", "sl", "end"] and res5.metrics["halted"] == 0.0


def test_daily_loss_limit_blocks_entries_until_the_next_utc_day(bt_settings):
    rows = [FLAT, LOSS] + [FLAT] * 26 + [(100.0, 101.0, 99.0, 100.0)]
    plan = {0: LONG_MKT, 2: LONG_MKT, 25: LONG_MKT}
    bt_settings.max_drawdown_pct, bt_settings.daily_loss_limit_pct, bt_settings.max_consecutive_losses = 50.0, 1.0, 0
    res, bars = run(bt_settings, rows, plan, brakes=True)
    entries = [pd.Timestamp(t.entry_ts, unit="s", tz="UTC") for t in res.trades]
    assert entries == [bars.index[1], bars.index[26]]          # la señal del mismo día se pierde; la del siguiente no
    assert res.metrics["daily_halts"] == 1 and res.metrics["n_blocked"] == 1 and res.metrics["halted"] == 0.0


def test_losing_streak_cooldown_blocks_automatic_entries(bt_settings):
    rows = [FLAT, LOSS, FLAT, LOSS, FLAT, FLAT, FLAT, FLAT, FLAT, (100.0, 101.0, 99.0, 100.0), FLAT]
    plan = {0: LONG_MKT, 2: LONG_MKT, 4: LONG_MKT, 8: LONG_MKT}
    bt_settings.max_drawdown_pct, bt_settings.daily_loss_limit_pct = 50.0, 50.0
    bt_settings.max_consecutive_losses, bt_settings.loss_cooldown_minutes = 2, 180
    res, bars = run(bt_settings, rows, plan, brakes=True)
    assert [t.entry_ts for t in res.trades] == [ts(bars, 1), ts(bars, 3), ts(bars, 9)]
    assert res.metrics["cooldowns"] == 1 and res.metrics["n_blocked"] == 1


def test_brakes_state_carries_over_between_chained_runs(bt_settings):
    """Los tramos fuera de muestra encadenados comparten el estado de los frenos: tras un halt no se opera más."""
    from desk.backtest import LiveBrakes
    rows = [FLAT, LOSS, FLAT, LOSS, FLAT, (100.0, 101.0, 99.0, 100.0), FLAT, FLAT, FLAT]
    bt_settings.max_drawdown_pct, bt_settings.daily_loss_limit_pct, bt_settings.max_consecutive_losses = 2.0, 50.0, 0
    brakes = LiveBrakes.from_settings(bt_settings)
    plan = {0: LONG_MKT, 2: LONG_MKT, 4: LONG_MKT, 6: LONG_MKT}
    first, bars = run(bt_settings, rows, plan, brakes=brakes, trade_from=0)
    assert brakes.halted and first.trades[-1].exit_reason == "halt"
    later, _ = run(bt_settings, rows, plan, brakes=brakes, trade_from=5)
    assert later.trades == [] and later.metrics["halted"] == 1.0

"""RiskManager: dimensionamiento con recortes, validaciones de entrada, halts, pausa, cooldown y persistencia."""

from __future__ import annotations

import json
import math

import pytest

from desk.models import AccountState, AssetMeta, EntryMode, Order, OrderKind, Position, Side, TradeIntent, Zone
from desk.risk import (EVENT_COOLDOWN_END, EVENT_COOLDOWN_START, EVENT_FLATTEN, EVENT_HALT_DAILY,
                       EVENT_HALT_DRAWDOWN, EVENT_HALT_LIFTED, EVENT_NEW_DAY, HALT_DAILY, HALT_DRAWDOWN,
                       RiskManager, estimated_liq_px)

TAKER = 4.5 / 10_000
DAY = 86_400.0

META = {
    "BTC": AssetMeta("BTC", sz_decimals=5, max_leverage=40),
    "ETH": AssetMeta("ETH", sz_decimals=4, max_leverage=25),
    "SOL": AssetMeta("SOL", sz_decimals=2, max_leverage=20),
    "HYPE": AssetMeta("HYPE", sz_decimals=2, max_leverage=2),
}
LONG, SHORT = Side.LONG, Side.SHORT
LIMIT, MARKET, ZONE = EntryMode.LIMIT, EntryMode.MARKET, EntryMode.ZONE


@pytest.fixture
def rm(settings, clock):
    return RiskManager(settings, clock)


def acct(equity=10_000.0, available=None, positions=None, orders=None):
    return AccountState(equity=equity, available=equity if available is None else available,
                        positions=positions or {}, open_orders=orders or [])


def position(coin, size, px, lev=5):
    return Position(coin=coin, size=size, entry_px=px, leverage=lev, mark_px=px,
                    margin_used=abs(size) * px / lev)


def entry_order(coin, size, px, *, reduce_only=False):
    return Order(id=f"o-{coin}", coin=coin, is_buy=True, size=size, kind=OrderKind.LIMIT, price=px,
                 reduce_only=reduce_only)


def intent(coin="SOL", side=LONG, units=1.0, mode=LIMIT, origin="command", **kw):
    if mode is LIMIT and "limit_px" not in kw:
        kw["limit_px"] = 100.0
    return TradeIntent(coin=coin, side=side, units=units, entry_mode=mode, origin=origin, **kw)


def ev(rm, it, account=None, *, bid=99.99, ask=100.01, age=1.0, meta=None):
    """SOL por defecto con bid/ask 99.99/100.01 (spread 2 bps)."""
    return rm.evaluate(it, account if account is not None else acct(), meta or META[it.coin], bid, ask, age)


def loss_at_sl(d):
    return d.size * (abs(d.entry_px - d.sl_price) + TAKER * (d.entry_px + d.sl_price))


def has(warnings, text):
    return any(text in w for w in warnings)


# --------------------------------------------------------------------------- ejemplo de referencia

def test_hype_reference_example_limit(rm, settings):
    assert settings.leverage == 5 and settings.unit_margin_usd == 500
    d = ev(rm, intent("HYPE", limit_px=85.249), bid=86.28, ask=86.29)
    assert d.approved, d.reason
    assert d.leverage == 2                          # min(5, max_leverage 2)
    assert d.entry_px == 85.249
    assert d.size == 11.73                          # floor(1000 / 85.249, 2 decimales)
    assert d.notional == pytest.approx(11.73 * 85.249) and d.notional == pytest.approx(1000, abs=0.1)
    assert d.sl_price == 83.97                      # SL por defecto 1.5 %
    assert d.tp_price == pytest.approx(85.249 + 2 * (85.249 - 83.97))
    assert has(d.warnings, "apalancamiento recortado de 5x a 2x")
    assert not has(d.warnings, "recortado por")     # ningún recorte de tamaño


def test_hype_reference_example_market_uses_ask(rm):
    d = ev(rm, intent("HYPE", mode=MARKET), bid=85.248, ask=85.249)
    assert d.approved and (d.leverage, d.size, d.entry_px) == (2, 11.73, 85.249)


def test_short_market_uses_bid_and_mirrored_levels(rm):
    d = ev(rm, intent("SOL", side=SHORT, mode=MARKET, sl_pct=1.0))
    assert d.approved and d.entry_px == 99.99
    assert d.sl_price == pytest.approx(100.99) and d.tp_price == pytest.approx(99.99 - 2 * 1.0)


# --------------------------------------------------------------------------- recortes de tamaño

def test_risk_cut_keeps_loss_at_sl_within_budget_including_fees(rm):
    d = ev(rm, intent(sl_pct=5.0))                  # 1u SOL 5x = 2500 USD; 5 % de SL = 125 USD > 100
    assert d.approved and d.leverage == 5
    assert loss_at_sl(d) <= 100.0 + 1e-9
    assert loss_at_sl(d) > 99.0                     # aprovecha el presupuesto (sólo pierde el redondeo)
    assert d.notional < 2_500
    assert has(d.warnings, "recortado por riesgo")


def test_risk_budget_scales_with_equity_and_setting(rm, settings):
    settings.risk_per_trade_pct = 0.5
    d = ev(rm, intent(sl_pct=5.0), acct(20_000))
    assert d.approved and loss_at_sl(d) <= 100.0 + 1e-9 and loss_at_sl(d) > 99.0


def test_no_risk_cut_when_loss_within_budget(rm):
    d = ev(rm, intent(sl_pct=2.0))                  # 2500 · 2 % = 50 USD + comisiones
    assert d.approved and d.size == 25.0 and not has(d.warnings, "riesgo:")
    assert loss_at_sl(d) == pytest.approx(25 * (2 + TAKER * 198))


def test_per_coin_cap(rm, settings):
    settings.max_notional_per_coin_usd = 1_500
    d = ev(rm, intent())
    assert d.approved and d.size == 15.0 and d.notional == pytest.approx(1_500)
    assert has(d.warnings, "máximo por moneda")


def test_gross_exposure_cap_counts_positions_and_pending_entries(rm):
    positions = {"BTC": position("BTC", 0.4, 100_000.0)}          # 40 000 USD
    orders = [entry_order("ETH", 3.0, 3_000.0)]                    # 9 000 USD pendientes
    d = ev(rm, intent(), acct(positions=positions, orders=orders))  # quedan 1 000 de 50 000
    assert d.approved and d.size == 10.0 and d.notional == pytest.approx(1_000)
    assert has(d.warnings, "exposición bruta")


def test_gross_exposure_exhausted_rejects(rm):
    positions = {"BTC": position("BTC", 0.5, 100_000.0)}          # 50 000 = máximo
    d = ev(rm, intent(), acct(positions=positions))
    assert not d.approved and "exposición bruta máxima" in d.reason


def test_margin_cap_uses_available_minus_entry_fee(rm):
    d = ev(rm, intent(), acct(available=200.0))
    cap = 200.0 / (1 / 5 + TAKER)
    assert d.approved and d.notional <= cap and d.size == math.floor(cap / 100 * 100) / 100
    assert d.notional / d.leverage + d.notional * TAKER <= 200.0
    assert has(d.warnings, "margen disponible")


def test_no_available_margin_rejects(rm):
    d = ev(rm, intent(), acct(available=0.0))
    assert not d.approved and "margen" in d.reason


def test_cuts_apply_in_order_and_each_is_explained(rm, settings):
    settings.max_notional_per_coin_usd = 2_000
    d = ev(rm, intent(sl_pct=5.0), acct(available=300.0))
    # riesgo: 2500 -> ~1965; moneda: no recorta (1965 < 2000); margen: -> ~1496
    assert d.approved
    assert has(d.warnings, "recortado por riesgo") and has(d.warnings, "margen disponible")
    assert not has(d.warnings, "máximo por moneda")


def test_units_clipped_to_max_per_trade(rm, settings):
    settings.max_notional_per_coin_usd = 1e9
    settings.max_gross_notional_usd = 1e9
    settings.risk_per_trade_pct = 10.0
    d = ev(rm, intent(units=10, sl_pct=1.0), acct(1e6))
    assert d.approved and has(d.warnings, "unidades recortadas de 10 a 4")
    assert d.notional == pytest.approx(4 * 500 * 5)


def test_below_min_notional_rejected(rm, settings):
    settings.unit_margin_usd = 1.0                  # 1u · 5x = 5 USD < 10 USD
    d = ev(rm, intent())
    assert not d.approved and "mínimo" in d.reason


def test_invalid_units_rejected(rm):
    for units in (0, -1, float("nan")):
        assert not ev(rm, intent(units=units)).approved


# --------------------------------------------------------------------------- cartera

def test_max_open_positions(rm):
    positions = {c: position(c, 0.01, 1_000.0) for c in ("BTC", "ETH", "HYPE")}
    d = ev(rm, intent(), acct(positions=positions))
    assert not d.approved and "máximo de posiciones" in d.reason


def test_pending_entries_count_as_positions_but_reduce_only_do_not(rm):
    positions = {c: position(c, 0.01, 1_000.0) for c in ("BTC", "ETH")}
    d = ev(rm, intent(), acct(positions=positions, orders=[entry_order("HYPE", 5, 80.0)]))
    assert not d.approved and "máximo de posiciones" in d.reason
    d = ev(rm, intent(), acct(positions=positions, orders=[entry_order("BTC", 0.01, 900.0, reduce_only=True)]))
    assert d.approved


def test_existing_position_in_coin_rejected(rm):
    d = ev(rm, intent(), acct(positions={"SOL": position("SOL", -3.0, 100.0)}))
    assert not d.approved and "ya hay una posición" in d.reason


def test_pending_entry_in_coin_rejected(rm):
    d = ev(rm, intent(), acct(orders=[entry_order("SOL", 3.0, 95.0)]))
    assert not d.approved and "entrada pendiente" in d.reason


# --------------------------------------------------------------------------- mercado

def test_wide_spread_rejected(rm):
    d = ev(rm, intent(), bid=99.8, ask=100.2)       # 40 bps > 25
    assert not d.approved and "spread" in d.reason


@pytest.mark.parametrize("bid,ask", [(0.0, 100.0), (100.2, 100.1), (float("nan"), 100.0)])
def test_invalid_quotes_rejected(rm, bid, ask):
    d = ev(rm, intent(), bid=bid, ask=ask)
    assert not d.approved and "precio inválido" in d.reason


def test_stale_data_rejected(rm, settings):
    d = ev(rm, intent(), age=settings.max_data_age_seconds + 1)
    assert not d.approved and "viejos" in d.reason
    assert not ev(rm, intent(), age=float("nan")).approved


def test_unknown_data_age_approved_with_warning(rm):
    d = ev(rm, intent(), age=None)
    assert d.approved and has(d.warnings, "desconocida")


def test_non_positive_equity_rejected(rm):
    d = ev(rm, intent(), acct(equity=0.0, available=0.0))
    assert not d.approved and "equity" in d.reason


# --------------------------------------------------------------------------- entrada, SL y TP

def test_limit_without_price_rejected(rm):
    d = ev(rm, TradeIntent("SOL", LONG, 1, LIMIT, "command"))
    assert not d.approved and "LIMIT" in d.reason


def test_limit_crossing_book_priced_at_touch(rm):
    d = ev(rm, intent(limit_px=101.0))
    assert d.approved and d.entry_px == 100.01 and has(d.warnings, "cruza el ask")
    d = ev(rm, intent(side=SHORT, limit_px=99.0))
    assert d.approved and d.entry_px == 99.99 and has(d.warnings, "cruza el bid")


def test_zone_entry_uses_zone_mid(rm):
    zone = Zone("SOL", LONG, low=97.0, high=97.2, usd=1e6, source="mixed", distance_pct=2.9)
    d = ev(rm, intent(mode=ZONE, zone=zone))
    assert d.approved and d.entry_px == pytest.approx(97.1)
    assert d.sl_price == pytest.approx(97.1 * 0.985, abs=0.01)


def test_zone_side_mismatch_or_missing_rejected(rm):
    zone = Zone("SOL", SHORT, low=103.0, high=103.2, usd=1e6, source="book")
    assert not ev(rm, intent(mode=ZONE, zone=zone)).approved
    d = ev(rm, TradeIntent("SOL", LONG, 1, ZONE, "auto:liquidity_sweep"))
    assert not d.approved and "zona" in d.reason
    d = ev(rm, TradeIntent("SOL", LONG, 1, ZONE, "command", limit_px=97.5))   # sin zona pero con precio
    assert d.approved and d.entry_px == 97.5


def test_sl_and_tp_pct_are_relative_to_entry(rm):
    d = ev(rm, intent(limit_px=100.0, sl_pct=2.0, tp_pct=4.0))
    assert (d.sl_price, d.tp_price) == (pytest.approx(98.0), pytest.approx(104.0))
    d = ev(rm, intent(side=SHORT, limit_px=102.0, sl_pct=2.0, tp_pct=4.0))
    assert d.approved and d.entry_px == 102.0
    assert (d.sl_price, d.tp_price) == (pytest.approx(104.04), pytest.approx(97.92))


def test_explicit_prices_win_over_pct_with_warning(rm):
    d = ev(rm, intent(sl_price=97.0, sl_pct=1.0, tp_price=110.0, tp_pct=1.0))
    assert d.approved and d.sl_price == 97.0 and d.tp_price == 110.0
    assert has(d.warnings, "SL en precio y en %") and has(d.warnings, "TP en precio y en %")


def test_default_tp_is_r_multiple(rm, settings):
    d = ev(rm, intent(sl_price=98.0))
    assert d.tp_price == pytest.approx(104.0)       # 2R
    settings.default_tp_r = 3.0
    assert ev(rm, intent(sl_price=98.0)).tp_price == pytest.approx(106.0)


@pytest.mark.parametrize("side,kw", [
    (LONG, {"sl_price": 101.0}), (LONG, {"sl_price": 100.0}), (SHORT, {"sl_price": 99.0}),
])
def test_sl_on_wrong_side_rejected(rm, side, kw):
    d = ev(rm, intent(side=side, limit_px=100.0, **kw))
    assert not d.approved and "SL del lado equivocado" in d.reason


def test_market_sl_inside_spread_rejected(rm):
    # Entrada a 100.01 (ask) y SL 100.0: por debajo de la entrada pero por encima del bid → saltaría al instante.
    d = ev(rm, intent(mode=MARKET, sl_price=100.0))
    assert not d.approved and "SL del lado equivocado" in d.reason


@pytest.mark.parametrize("side,kw", [(LONG, {"tp_price": 99.0}), (SHORT, {"tp_price": 101.0})])
def test_tp_on_wrong_side_rejected(rm, side, kw):
    d = ev(rm, intent(side=side, limit_px=100.0, **kw))
    assert not d.approved and "TP del lado equivocado" in d.reason


def test_invalid_pct_rejected(rm):
    assert not ev(rm, intent(sl_pct=0.0)).approved
    assert not ev(rm, intent(tp_pct=-1.0)).approved


def test_no_stop_when_not_required(rm, settings):
    settings.require_stop = False
    d = ev(rm, intent())
    assert d.approved and d.sl_price is None and d.tp_price is None
    assert d.size == 25.0 and has(d.warnings, "sin stop-loss")


def test_leverage_lowered_when_liquidation_would_hit_before_sl(rm):
    d = ev(rm, intent(sl_pct=20.0))
    # SOL: mmr = 1/(2·20) = 2.5 %. A 5x la liquidación (~82.05) queda antes que el SL 80; a 4x (~76.9) no.
    assert d.approved and d.leverage == 4
    assert has(d.warnings, "apalancamiento bajado de 5x a 4x")
    liq = estimated_liq_px(d.entry_px, d.leverage, True, 0.025)
    assert d.entry_px - liq >= (d.entry_px - d.sl_price) * 1.1


def test_sl_beyond_liquidation_even_at_1x_rejected(rm):
    d = ev(rm, intent(side=SHORT, limit_px=100.0, sl_price=300.0))
    assert not d.approved and "liquidación" in d.reason


def test_meta_for_other_coin_rejected(rm):
    d = rm.evaluate(intent("SOL"), acct(), META["BTC"], 99.99, 100.01, 1.0)
    assert not d.approved


# --------------------------------------------------------------------------- halt por pérdida diaria

def test_daily_loss_halt_lifts_on_next_utc_day(rm, clock):
    assert rm.update(acct(10_000)) == []
    assert rm.update(acct(9_750)) == []             # -2.5 %
    events = rm.update(acct(9_690))                 # -3.1 % ≥ 3 %
    assert events == [EVENT_HALT_DAILY]
    assert rm.active_halt() == HALT_DAILY
    for origin in ("command", "auto:squeeze_breakout"):
        d = ev(rm, intent(origin=origin), acct(9_690))
        assert not d.approved and "daily_loss" in d.reason
    assert rm.update(acct(9_800)) == []             # mismo día: sigue detenida
    assert rm.active_halt() == HALT_DAILY

    clock.set(1_767_225_600.0 + DAY + 1)            # 2026-01-02 00:00:01 UTC
    assert rm.active_halt() is None                 # ya no bloquea aunque falte el update()
    assert ev(rm, intent(), acct(9_800)).approved
    events = rm.update(acct(9_800))
    assert events == [EVENT_NEW_DAY, EVENT_HALT_LIFTED]
    assert not rm.halted and rm.day == "2026-01-02" and rm.day_start_equity == 9_800


def test_daily_loss_does_not_flatten_by_default(rm, settings):
    assert settings.flatten_on_daily_limit is False
    rm.update(acct(10_000))
    events = rm.update(acct(9_600, positions={"SOL": position("SOL", 5, 100.0)}))
    assert events == [EVENT_HALT_DAILY]


def test_daily_loss_flattens_when_configured(rm, settings):
    settings.flatten_on_daily_limit = True
    rm.update(acct(10_000))
    held = acct(9_600, positions={"SOL": position("SOL", 5, 100.0)})
    assert rm.update(held) == [EVENT_HALT_DAILY, EVENT_FLATTEN]


def test_new_day_resets_daily_reference(rm, clock):
    rm.update(acct(10_000))
    clock.advance(DAY)
    assert rm.update(acct(9_600)) == [EVENT_NEW_DAY]     # el día empieza en 9600: no hay pérdida diaria
    assert rm.day_start_equity == 9_600 and not rm.halted


# --------------------------------------------------------------------------- halt por drawdown

def test_max_drawdown_halt_flattens_until_flat_and_needs_reset(rm, clock):
    held = {"SOL": position("SOL", 5, 100.0)}
    rm.update(acct(10_000))
    rm.update(acct(10_600))                         # nuevo máximo
    assert rm.high_water == 10_600
    events = rm.update(acct(10_050, positions=held))     # -5.19 % desde el máximo (el día va +0.5 %)
    assert events == [EVENT_HALT_DRAWDOWN, EVENT_FLATTEN]
    assert rm.active_halt() == HALT_DRAWDOWN
    assert rm.update(acct(10_040, positions=held)) == [EVENT_FLATTEN]       # el cierre no terminó: insiste
    assert rm.update(acct(10_040, orders=[entry_order("ETH", 1, 3_000.0)])) == [EVENT_FLATTEN]
    assert rm.update(acct(10_040)) == []            # plana: nada que cerrar

    clock.advance(DAY)
    assert rm.update(acct(10_040)) == [EVENT_NEW_DAY]    # no se levanta con el día
    assert rm.active_halt() == HALT_DRAWDOWN
    assert not ev(rm, intent(), acct(10_040)).approved
    assert "reset halt" in rm.resume()               # resume no levanta un halt
    assert rm.active_halt() == HALT_DRAWDOWN

    msg = rm.reset_halt()
    assert "max_drawdown" in msg and not rm.halted
    assert rm.high_water == 10_040                  # el drawdown se mide desde aquí
    assert rm.update(acct(10_040)) == []
    assert ev(rm, intent(), acct(10_040)).approved


def test_drawdown_upgrades_daily_halt(rm, clock):
    rm.update(acct(10_000))
    assert rm.update(acct(9_690)) == [EVENT_HALT_DAILY]
    assert rm.update(acct(9_400)) == [EVENT_HALT_DRAWDOWN]   # sin exposición: no hace falta flatten
    assert rm.halt_reason == HALT_DRAWDOWN
    clock.advance(DAY)
    assert rm.update(acct(9_400)) == [EVENT_NEW_DAY]
    assert rm.active_halt() == HALT_DRAWDOWN


def test_reset_daily_halt_rebases_day_reference(rm):
    rm.update(acct(10_000))
    rm.update(acct(9_600))
    assert rm.reset_halt().startswith("halt 'daily_loss' levantado")
    assert rm.update(acct(9_600)) == [] and not rm.halted
    assert rm.reset_halt() == "no había ningún halt activo"


def test_manual_kill_halt(rm, clock):
    rm.update(acct(10_000))
    assert rm.halt("kill", "pánico del usuario", flatten=True) == ["halt:kill"]
    held = acct(10_000, positions={"SOL": position("SOL", 5, 100.0)})
    assert rm.update(held) == [EVENT_FLATTEN]
    clock.advance(DAY)
    rm.update(acct(10_000))
    assert rm.active_halt() == "kill"
    assert "kill" in ev(rm, intent()).reason
    rm.reset_halt()
    assert ev(rm, intent()).approved


def test_nan_equity_is_ignored(rm):
    rm.update(acct(10_000))
    assert rm.update(acct(float("nan"))) == []
    assert rm.last_equity == 10_000


# --------------------------------------------------------------------------- pausa y cooldown

def test_pause_blocks_every_origin_until_resume(rm):
    rm.pause()
    for origin in ("command", "auto:liquidity_sweep"):
        d = ev(rm, intent(origin=origin))
        assert not d.approved and "pausa" in d.reason
    assert rm.resume() == "mesa reanudada"
    assert ev(rm, intent()).approved


def test_loss_streak_cooldown_blocks_only_auto_entries(rm, clock, settings):
    assert settings.max_consecutive_losses == 4 and settings.loss_cooldown_minutes == 120
    for _ in range(3):
        assert rm.record_trade_result(-10.0) == []
    assert rm.record_trade_result(-10.0) == [EVENT_COOLDOWN_START]
    assert rm.in_cooldown and not rm.allows_auto()

    d = ev(rm, intent(origin="auto:liquidity_sweep"))
    assert not d.approved and "cooldown" in d.reason
    assert ev(rm, intent(origin="command")).approved     # el usuario sigue mandando

    clock.advance(119 * 60)
    assert rm.in_cooldown
    clock.advance(2 * 60)
    assert not rm.in_cooldown and ev(rm, intent(origin="auto:liquidity_sweep")).approved
    assert EVENT_COOLDOWN_END in rm.update(acct())
    assert rm.allows_auto()


def test_win_resets_streak_and_breakeven_is_neutral(rm):
    for pnl in (-1, -1, -1, 5, -1, 0.0, -1):
        assert rm.record_trade_result(pnl) == []
    assert rm.consecutive_losses == 2 and not rm.in_cooldown


def test_reset_halt_also_clears_cooldown(rm):
    for _ in range(4):
        rm.record_trade_result(-1)
    assert rm.in_cooldown
    assert "cooldown reiniciado" in rm.reset_halt()
    assert not rm.in_cooldown


# --------------------------------------------------------------------------- persistencia

def test_state_roundtrip_through_json(rm, settings, clock):
    rm.update(acct(10_000))
    rm.update(acct(10_600))
    rm.update(acct(10_000))                         # -5.66 % → halt por drawdown
    for _ in range(4):
        rm.record_trade_result(-1)
    rm.record_trade_result(-1)
    rm.pause()
    data = json.loads(json.dumps(rm.to_dict()))
    restored = RiskManager.from_dict(data, settings, clock)
    assert restored.to_dict() == rm.to_dict()
    assert restored.active_halt() == HALT_DRAWDOWN and restored.high_water == 10_600
    assert restored.in_cooldown and restored.paused and restored.consecutive_losses == 1
    assert not ev(restored, intent()).approved


def test_daily_halt_restored_after_restart_lifts_next_day(rm, settings, clock):
    rm.update(acct(10_000))
    rm.update(acct(9_600))                          # -4 %: diario sí, drawdown (5 %) no
    restored = RiskManager.from_dict(rm.to_dict(), settings, clock)
    assert restored.active_halt() == HALT_DAILY
    clock.advance(DAY)
    assert restored.update(acct(9_600)) == [EVENT_NEW_DAY, EVENT_HALT_LIFTED]


def test_from_dict_tolerates_garbage(settings, clock):
    assert RiskManager.from_dict(None, settings, clock).to_dict() == RiskManager(settings, clock).to_dict()
    assert not RiskManager.from_dict(["no", "dict"], settings, clock).halted
    rm = RiskManager.from_dict({"halted": True, "halt_reason": "max_drawdown", "high_water": "nan?",
                                "consecutive_losses": "x", "cooldown_until": None}, settings, clock)
    assert rm.active_halt() == HALT_DRAWDOWN        # el halt nunca se pierde
    assert rm.high_water is None and rm.consecutive_losses == 0 and rm.cooldown_until == 0.0


def test_describe_mentions_state(rm):
    rm.update(acct(10_000))
    assert rm.describe().startswith("riesgo:")
    rm.halt("kill")
    rm.pause()
    text = rm.describe()
    assert "HALT kill" in text and "PAUSA" in text


# --------------------------------------------------------------------------- regresiones (revisión)
def test_limit_far_across_the_book_is_rejected_not_turned_into_market(rm):
    # "short BTC a 83.825" (miles en español leído como decimal): antes se convertía en un short a mercado.
    d = rm.evaluate(intent("BTC", SHORT, limit_px=83.825), acct(), META["BTC"], 83_770.0, 83_780.0, 1.0)
    assert not d.approved and "mal escrito" in d.reason
    d = ev(rm, intent(limit_px=102.0))                 # cruza el ask un ~2 % (> LIMIT_MAX_CROSS_PCT)
    assert not d.approved and "por encima del ask" in d.reason
    d = ev(rm, intent(limit_px=100.5))                 # cruce pequeño: taker al ask, con aviso
    assert d.approved and d.entry_px == 100.01


def test_limit_resting_too_far_from_mid_is_rejected(rm):
    d = ev(rm, intent(limit_px=70.0))                  # 30 % por debajo del mid
    assert not d.approved and "del precio actual" in d.reason
    assert ev(rm, intent(limit_px=85.0, sl_pct=1.0)).approved      # 15 %: dentro del collar


def test_evaluate_checks_loss_limits_with_the_fresh_account(rm, settings):
    # El equity cruzó el límite diario entre ciclos: update() aún no lo vio, pero evaluate() sí.
    settings.daily_loss_limit_pct = 0.5
    rm.update(acct(10_000))
    d = ev(rm, intent(), acct(9_920))
    assert not d.approved and "pérdida del día" in d.reason and not rm.halted
    settings.daily_loss_limit_pct = 3.0
    settings.max_drawdown_pct = 1.0
    d = ev(rm, intent(), acct(9_850))
    assert not d.approved and "drawdown" in d.reason


def test_cash_flow_moves_high_water_and_day_reference(rm):
    rm.update(acct(10_000))
    rm.apply_cash_flow(-1_000.0)                       # retirada / transferencia perp -> spot
    assert rm.high_water == 9_000 and rm.day_start_equity == 9_000
    assert rm.update(acct(9_000)) == [] and not rm.halted
    rm.apply_cash_flow(500.0)                          # depósito: tampoco es ganancia de la operativa
    assert rm.high_water == 9_500 and rm.day_loss_pct(9_500) == pytest.approx(0.0)

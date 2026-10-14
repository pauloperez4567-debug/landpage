"""PaperBroker: semántica completa de órdenes, posiciones, margen, liquidación y persistencia."""

from __future__ import annotations

import json

import pytest

from desk.exchange.base import BrokerError
from desk.exchange.paper import PaperBroker
from desk.models import OrderKind, OrderStatus

TAKER = 4.5 / 10_000
MAKER = 1.5 / 10_000
MMR = 0.5 / 100

MARKET, LIMIT, STOP, TP = OrderKind.MARKET, OrderKind.LIMIT, OrderKind.STOP, OrderKind.TAKE_PROFIT


@pytest.fixture
def cfg(settings):
    settings.paper_slippage_bps = 0.0    # precios exactos; el deslizamiento tiene sus propios tests
    return settings


@pytest.fixture
def broker(market, cfg, clock):
    market.set_bid_ask("SOL", 99.9, 100.1)    # mid 100
    return PaperBroker(market, cfg, clock)


def buy(b, size, coin="SOL", **kw):
    return b.place_order(coin, True, size, kw.pop("kind", MARKET), **kw)


def sell(b, size, coin="SOL", **kw):
    return b.place_order(coin, False, size, kw.pop("kind", MARKET), **kw)


# --------------------------------------------------------------------------- cuenta inicial

def test_initial_account(broker, cfg):
    acct = broker.account()
    assert acct.equity == cfg.paper_start_equity == 10_000.0
    assert acct.available == 10_000.0
    assert acct.positions == {} and acct.open_orders == []
    assert broker.poll_fills() == []


# --------------------------------------------------------------------------- MARKET

def test_market_buy_fills_at_ask_with_taker_fee(broker):
    o = buy(broker, 10, tag="entry")
    assert o.status is OrderStatus.FILLED and o.filled_size == 10 and o.avg_fill_px == 100.1
    [f] = broker.poll_fills()
    assert (f.order_id, f.is_buy, f.size, f.price, f.tag) == (o.id, True, 10, 100.1, "entry")
    assert f.fee == pytest.approx(10 * 100.1 * TAKER)
    assert f.realized_pnl == 0.0 and not f.liquidation
    acct = broker.account()
    pos = acct.positions["SOL"]
    assert pos.size == 10 and pos.entry_px == 100.1 and pos.leverage == 5
    assert pos.margin_used == pytest.approx(1001.0 / 5)
    assert pos.mark_px == pytest.approx(100.0)
    assert pos.unrealized_pnl == pytest.approx(-1.0)
    assert acct.equity == pytest.approx(10_000 - f.fee - 1.0)
    assert acct.available == pytest.approx(acct.equity - 200.2)


def test_market_sell_opens_short_at_bid(broker):
    o = sell(broker, 3)
    assert o.status is OrderStatus.FILLED and o.avg_fill_px == 99.9
    pos = broker.account().positions["SOL"]
    assert pos.size == -3 and pos.entry_px == 99.9
    assert pos.unrealized_pnl == pytest.approx(-0.3)


def test_market_slippage_applied(market, settings, clock):
    settings.paper_slippage_bps = 2.0
    market.set_bid_ask("SOL", 99.9, 100.1)
    b = PaperBroker(market, settings, clock)
    assert buy(b, 1).avg_fill_px == pytest.approx(100.1 * 1.0002)
    assert sell(b, 2).avg_fill_px == pytest.approx(99.9 * 0.9998)


def test_market_worst_price_protection(broker):
    r = buy(broker, 1, price=100.05)
    assert r.status is OrderStatus.REJECTED and "peor" in r.reason
    assert broker.account().positions == {}
    assert buy(broker, 1, price=100.2).status is OrderStatus.FILLED
    r = sell(broker, 1, price=99.95)
    assert r.status is OrderStatus.REJECTED
    assert sell(broker, 1, price=99.9).status is OrderStatus.FILLED


# --------------------------------------------------------------------------- LIMIT

def test_limit_buy_rests_then_fills_as_maker_at_limit(broker, market):
    o = buy(broker, 10, kind=LIMIT, price=99.5, tag="entry")
    assert o.status is OrderStatus.OPEN
    assert [x.id for x in broker.open_orders("SOL")] == [o.id]
    broker.step()
    assert broker.poll_fills() == []
    market.set_bid_ask("SOL", 99.0, 99.2)       # el ask atraviesa el límite: ejecuta AL LÍMITE
    broker.step()
    [f] = broker.poll_fills()
    assert f.price == 99.5 and f.size == 10 and f.tag == "entry"
    assert f.fee == pytest.approx(10 * 99.5 * MAKER)
    assert broker.open_orders() == []
    assert broker.get_order(o.id).status is OrderStatus.FILLED
    assert broker.account().positions["SOL"].entry_px == 99.5


def test_limit_sell_rests_until_bid_reaches_price(broker, market):
    o = sell(broker, 5, kind=LIMIT, price=101.0)
    assert o.status is OrderStatus.OPEN
    market.set_bid_ask("SOL", 100.9, 101.1)
    broker.step()
    assert broker.poll_fills() == []
    market.set_bid_ask("SOL", 101.0, 101.2)
    broker.step()
    [f] = broker.poll_fills()
    assert not f.is_buy and f.price == 101.0 and f.fee == pytest.approx(5 * 101.0 * MAKER)
    assert broker.account().positions["SOL"].size == -5


def test_marketable_limit_is_taker_capped_at_limit(market, settings, clock):
    market.set_bid_ask("SOL", 99.9, 100.1)
    b = PaperBroker(market, settings, clock)             # deslizamiento 2 bps
    o = b.place_order("SOL", True, 2, LIMIT, price=100.5)
    assert o.status is OrderStatus.FILLED and o.avg_fill_px == pytest.approx(100.1 * 1.0002)
    [f] = b.poll_fills()
    assert f.fee == pytest.approx(2 * f.price * TAKER)
    settings.paper_slippage_bps = 50.0                   # 100.1 * 1.005 > 100.15 -> tope en el límite
    o = b.place_order("SOL", True, 2, LIMIT, price=100.15)
    assert o.avg_fill_px == 100.15
    o = b.place_order("SOL", False, 1, LIMIT, price=99.85)
    assert o.avg_fill_px == 99.85


def test_alo_rejected_if_marketable_else_maker(broker, market):
    r = buy(broker, 1, kind=LIMIT, price=100.1, tif="Alo")
    assert r.status is OrderStatus.REJECTED and "Alo" in r.reason
    o = buy(broker, 1, kind=LIMIT, price=99.0, tif="Alo")
    assert o.status is OrderStatus.OPEN
    market.set_bid_ask("SOL", 98.8, 99.0)
    broker.step()
    [f] = broker.poll_fills()
    assert f.price == 99.0 and f.fee == pytest.approx(99.0 * MAKER)


def test_ioc_fills_now_or_is_rejected(broker):
    r = buy(broker, 1, kind=LIMIT, price=100.0, tif="Ioc")
    assert r.status is OrderStatus.REJECTED and "Ioc" in r.reason
    assert broker.open_orders() == []
    o = buy(broker, 1, kind=LIMIT, price=100.1, tif="Ioc")
    assert o.status is OrderStatus.FILLED and o.avg_fill_px == 100.1


def test_limit_price_rounded_to_valid_tick(broker):
    o = buy(broker, 1, kind=LIMIT, price=95.123456)
    assert o.price == 95.123                      # SOL: sz_decimals 2 -> 5 cifras significativas
    o = buy(broker, 0.001, coin="BTC", kind=LIMIT, price=80_123.4)
    assert o.price == 80_123.0                    # BTC por encima de 10k: precio entero


# --------------------------------------------------------------------------- STOP / TAKE_PROFIT

def _long_with_brackets(b, size=10, sl=95.0, tp=110.0):
    buy(b, size, tag="entry")
    s = sell(b, size, kind=STOP, trigger_px=sl, reduce_only=True, tag="sl")
    t = sell(b, size, kind=TP, trigger_px=tp, reduce_only=True, tag="tp")
    b.poll_fills()
    return s, t


def test_long_stop_triggers_on_mid_and_cancels_tp(broker, market):
    s, t = _long_with_brackets(broker)
    assert s.status is OrderStatus.OPEN and t.status is OrderStatus.OPEN
    market.set_bid_ask("SOL", 95.05, 95.15)        # mid 95.1 > 95: no dispara
    broker.step()
    assert broker.poll_fills() == []
    market.set_bid_ask("SOL", 94.9, 95.1)          # mid 95.0 <= 95: dispara, ejecuta a mercado (bid)
    broker.step()
    [f] = broker.poll_fills()
    assert f.order_id == s.id and f.tag == "sl" and not f.is_buy and f.price == 94.9
    assert f.realized_pnl == pytest.approx(10 * (94.9 - 100.1))
    assert f.fee == pytest.approx(10 * 94.9 * TAKER)
    assert broker.account().positions == {}
    assert broker.open_orders() == []
    tp = broker.get_order(t.id)
    assert tp.status is OrderStatus.CANCELED and "plana" in tp.reason


def test_long_take_profit_triggers_when_mid_above(broker, market):
    s, t = _long_with_brackets(broker)
    market.set_bid_ask("SOL", 109.8, 110.0)        # mid 109.9
    broker.step()
    assert broker.poll_fills() == []
    market.set_bid_ask("SOL", 110.0, 110.2)        # mid 110.1 >= 110
    broker.step()
    [f] = broker.poll_fills()
    assert f.order_id == t.id and f.price == 110.0
    assert f.realized_pnl == pytest.approx(10 * (110.0 - 100.1))
    assert broker.get_order(s.id).status is OrderStatus.CANCELED


def test_short_stop_and_take_profit(broker, market):
    sell(broker, 10)                                # short @ 99.9
    s = buy(broker, 10, kind=STOP, trigger_px=105.0, reduce_only=True, tag="sl")
    t = buy(broker, 10, kind=TP, trigger_px=90.0, reduce_only=True, tag="tp")
    broker.poll_fills()
    market.set_bid_ask("SOL", 104.8, 105.0)         # mid 104.9 < 105
    broker.step()
    assert broker.poll_fills() == []
    market.set_bid_ask("SOL", 104.9, 105.1)         # mid 105 >= 105: stop del short
    broker.step()
    [f] = broker.poll_fills()
    assert f.order_id == s.id and f.is_buy and f.price == 105.1
    assert f.realized_pnl == pytest.approx(10 * (99.9 - 105.1))
    assert broker.get_order(t.id).status is OrderStatus.CANCELED

    market.set_bid_ask("SOL", 99.9, 100.1)
    sell(broker, 10)                                # nuevo short @ 99.9
    t2 = buy(broker, 10, kind=TP, trigger_px=90.0, reduce_only=True)
    market.set_bid_ask("SOL", 90.05, 90.15)         # mid 90.1 > 90
    broker.step()
    assert broker.get_order(t2.id).status is OrderStatus.OPEN
    market.set_bid_ask("SOL", 89.9, 90.1)           # mid 90 <= 90: TP del short
    broker.step()
    fills = broker.poll_fills()
    assert fills[-1].order_id == t2.id and fills[-1].price == 90.1
    assert fills[-1].realized_pnl == pytest.approx(10 * (99.9 - 90.1))
    assert broker.account().positions == {}


def test_trigger_already_crossed_fires_on_next_step(broker):
    buy(broker, 2)
    s = sell(broker, 2, kind=STOP, trigger_px=100.5, reduce_only=True)   # mid 100 ya está por debajo
    assert s.status is OrderStatus.OPEN
    broker.step()
    assert broker.get_order(s.id).status is OrderStatus.FILLED


def test_trigger_validation(broker):
    buy(broker, 2)
    r = sell(broker, 2, kind=STOP, trigger_px=95.0)
    assert r.status is OrderStatus.REJECTED and "reduce_only" in r.reason
    r = sell(broker, 2, kind=TP, reduce_only=True)
    assert r.status is OrderStatus.REJECTED and "trigger_px" in r.reason
    assert sell(broker, 2, kind=STOP, trigger_px=94.98765, reduce_only=True).trigger_px == 94.988


# --------------------------------------------------------------------------- reduce-only

def test_reduce_only_rejections(broker):
    r = sell(broker, 1, reduce_only=True)
    assert r.status is OrderStatus.REJECTED and "reduce-only" in r.reason
    buy(broker, 2)
    r = buy(broker, 1, reduce_only=True)            # mismo sentido que la posición
    assert r.status is OrderStatus.REJECTED
    r = buy(broker, 1, kind=STOP, trigger_px=90.0, reduce_only=True)
    assert r.status is OrderStatus.REJECTED


def test_reduce_only_trimmed_to_position(broker):
    buy(broker, 3)
    o = sell(broker, 10, reduce_only=True, tag="close")
    assert o.status is OrderStatus.FILLED and o.size == 3 and "recortada" in o.reason
    assert broker.account().positions == {}
    s = broker.poll_fills()[-1]
    assert s.size == 3 and s.realized_pnl == pytest.approx(3 * (99.9 - 100.1))


def test_partial_reduce_keeps_entry_and_brackets(broker, market):
    s, t = _long_with_brackets(broker, size=10)
    market.set_bid_ask("SOL", 104.9, 105.1)
    o = sell(broker, 4, reduce_only=True)
    [f] = broker.poll_fills()
    assert f.realized_pnl == pytest.approx(4 * (104.9 - 100.1))
    pos = broker.account().positions["SOL"]
    assert pos.size == 6 and pos.entry_px == 100.1
    assert pos.margin_used == pytest.approx(6 * 100.1 / 5)
    assert {x.id for x in broker.open_orders()} == {s.id, t.id}
    # Al dispararse, el stop (10) se recorta a lo que queda (6).
    market.set_bid_ask("SOL", 94.0, 94.2)
    broker.step()
    [f] = broker.poll_fills()
    assert f.order_id == s.id and f.size == 6
    assert o.status is OrderStatus.FILLED


def test_resting_reduce_only_limit_trimmed_at_fill(broker, market):
    buy(broker, 10)
    lim = sell(broker, 10, kind=LIMIT, price=105.0, reduce_only=True)
    assert lim.status is OrderStatus.OPEN
    sell(broker, 6, reduce_only=True)
    broker.poll_fills()
    market.set_bid_ask("SOL", 105.0, 105.2)
    broker.step()
    [f] = broker.poll_fills()
    assert f.order_id == lim.id and f.size == 4 and f.price == 105.0
    assert f.fee == pytest.approx(4 * 105.0 * MAKER)
    assert broker.account().positions == {}


def test_reduce_only_exempt_from_min_notional(broker):
    buy(broker, 1)                                   # 100 USD
    o = sell(broker, 0.05, reduce_only=True)         # 5 USD < 10 USD mínimo
    assert o.status is OrderStatus.FILLED
    r = sell(broker, 0.05)                           # sin reduce-only sí se valida
    assert r.status is OrderStatus.REJECTED and "notional" in r.reason


# --------------------------------------------------------------------------- posición

def test_average_entry_on_increase(broker, market):
    buy(broker, 1)
    market.set_bid_ask("SOL", 109.9, 110.1)
    buy(broker, 3)
    pos = broker.account().positions["SOL"]
    assert pos.size == 4
    assert pos.entry_px == pytest.approx((100.1 + 3 * 110.1) / 4)
    assert pos.margin_used == pytest.approx(4 * pos.entry_px / 5)


def test_flip_long_to_short(broker, market):
    buy(broker, 2)
    s = sell(broker, 2, kind=STOP, trigger_px=90.0, reduce_only=True)
    market.set_bid_ask("SOL", 104.9, 105.1)
    o = sell(broker, 5)
    assert o.status is OrderStatus.FILLED
    fills = broker.poll_fills()
    assert fills[-1].size == 5
    assert fills[-1].realized_pnl == pytest.approx(2 * (104.9 - 100.1))
    pos = broker.account().positions["SOL"]
    assert pos.size == -3 and pos.entry_px == 104.9
    assert pos.margin_used == pytest.approx(3 * 104.9 / 5)
    stop = broker.get_order(s.id)
    assert stop.status is OrderStatus.CANCELED and "sentido" in stop.reason


def test_flip_releases_margin_of_closed_side(broker):
    buy(broker, 490)                                  # margen 9809.8 de 10000
    assert buy(broker, 10).status is OrderStatus.REJECTED   # no queda margen para aumentar
    o = sell(broker, 980)                             # gira a short 490: usa el margen liberado
    assert o.status is OrderStatus.FILLED, o.reason
    assert broker.account().positions["SOL"].size == -490


def test_cash_identity_after_round_trips(broker, market):
    buy(broker, 5)
    market.set_bid_ask("SOL", 101.9, 102.1)
    sell(broker, 8)
    market.set_bid_ask("SOL", 98.9, 99.1)
    buy(broker, 3, reduce_only=True)
    fills = broker.poll_fills()
    st = broker.stats()
    assert st["total_fees"] == pytest.approx(sum(f.fee for f in fills))
    assert st["total_realized"] == pytest.approx(sum(f.realized_pnl for f in fills))
    assert st["cash"] == pytest.approx(10_000 + st["total_realized"] - st["total_fees"])
    assert st["total_realized"] == pytest.approx(5 * (101.9 - 100.1) + 3 * (101.9 - 99.1))
    assert broker.account().positions == {}
    assert broker.account().equity == pytest.approx(st["cash"])


# --------------------------------------------------------------------------- liquidación

def test_liquidation_price_formula(broker):
    buy(broker, 10)
    pos = broker.account().positions["SOL"]
    assert pos.liq_px == pytest.approx((100.1 - 100.1 / 5) / (1 - MMR))
    sell(broker, 20)
    pos = broker.account().positions["SOL"]
    assert pos.size == -10
    assert pos.liq_px == pytest.approx((99.9 + 99.9 / 5) / (1 + MMR))


def test_long_liquidated_when_mark_crosses(broker, market):
    s, t = _long_with_brackets(broker, size=10, sl=50.0, tp=150.0)
    liq = (100.1 - 20.02) / (1 - MMR)                  # ≈ 80.48
    market.set_bid_ask("SOL", 80.9, 81.1)
    broker.step()
    assert broker.poll_fills() == []
    market.set_bid_ask("SOL", 80.3, 80.5)              # mid 80.4 <= liq
    assert 80.4 <= liq
    broker.step()
    [f] = broker.poll_fills()
    assert f.liquidation and f.tag == "liquidation" and not f.is_buy
    assert f.price == pytest.approx(80.4)
    assert f.realized_pnl == pytest.approx(10 * (80.4 - 100.1))
    assert f.fee == pytest.approx(10 * 80.4 * TAKER)
    assert broker.account().positions == {}
    assert broker.get_order(s.id).status is OrderStatus.CANCELED
    assert broker.get_order(t.id).status is OrderStatus.CANCELED


def test_liquidation_gap_loss_capped_at_margin(broker, market):
    buy(broker, 10)
    broker.poll_fills()
    market.set_bid_ask("SOL", 69.9, 70.1)              # salto muy por debajo de la quiebra (80.08)
    broker.step()
    [f] = broker.poll_fills()
    assert f.liquidation and f.price == pytest.approx(100.1 - 20.02)
    assert f.realized_pnl == pytest.approx(-200.2)     # pierde exactamente el margen aislado


def test_short_liquidation(broker, market):
    sell(broker, 10)
    broker.poll_fills()
    market.set_bid_ask("SOL", 119.4, 119.6)             # mid 119.5 >= liq ≈ 119.28
    broker.step()
    [f] = broker.poll_fills()
    assert f.liquidation and f.is_buy and f.price == pytest.approx(119.5)
    assert f.realized_pnl == pytest.approx(10 * (99.9 - 119.5))


def test_stop_beats_liquidation_in_same_step(broker, market):
    s, _ = _long_with_brackets(broker, size=10, sl=90.0, tp=150.0)
    market.set_bid_ask("SOL", 79.9, 80.1)              # cruza stop y liquidación a la vez
    broker.step()
    [f] = broker.poll_fills()
    assert f.order_id == s.id and not f.liquidation


# --------------------------------------------------------------------------- validaciones

def test_size_validations(broker):
    r = buy(broker, 0.004)                             # SOL: mínimo 0.01
    assert r.status is OrderStatus.REJECTED and "mínimo" in r.reason
    r = buy(broker, -1)
    assert r.status is OrderStatus.REJECTED
    r = buy(broker, 0.05)                              # 5 USD < 10 USD
    assert r.status is OrderStatus.REJECTED and "notional" in r.reason
    o = buy(broker, 0.123456789, coin="BTC")           # se redondea hacia abajo a 5 decimales
    assert o.status is OrderStatus.FILLED and o.size == 0.12345


def test_margin_validation(broker):
    r = buy(broker, 600)                               # 60060 / 5 = 12012 > 10000
    assert r.status is OrderStatus.REJECTED and "margen insuficiente" in r.reason
    assert buy(broker, 400).status is OrderStatus.FILLED


def test_resting_orders_reserve_margin(broker):
    o = buy(broker, 20, kind=LIMIT, price=95.0)
    assert broker.account().available == pytest.approx(10_000 - 20 * 95.0 / 5)
    r = buy(broker, 490)                               # 9809.8 > 10000 - 380
    assert r.status is OrderStatus.REJECTED
    assert broker.cancel("SOL", o.id)
    assert broker.account().available == pytest.approx(10_000)
    assert buy(broker, 490).status is OrderStatus.FILLED


def test_leverage_validations(broker, market):
    with pytest.raises(ValueError):
        broker.set_leverage("HYPE", 3)                 # HYPE: máximo 2x
    with pytest.raises(ValueError):
        broker.set_leverage("SOL", 2.5)
    with pytest.raises(ValueError):
        broker.set_leverage("SOL", 0)
    with pytest.raises(ValueError):
        broker.set_leverage("NOPE", 2)
    broker.set_leverage("HYPE", 2)
    o = buy(broker, 1, coin="HYPE")
    assert o.status is OrderStatus.FILLED
    assert broker.account().positions["HYPE"].leverage == 2
    broker.leverage["HYPE"] = 5                        # p. ej. estado viejo con otro máximo
    r = buy(broker, 1, coin="HYPE")
    assert r.status is OrderStatus.REJECTED and "apalancamiento" in r.reason


def test_default_leverage_clipped_to_asset_max(broker):
    buy(broker, 1, coin="HYPE")                        # settings.leverage=5, HYPE máx 2
    assert broker.account().positions["HYPE"].leverage == 2


def test_set_leverage_remargins_open_position(broker):
    buy(broker, 10)
    broker.set_leverage("SOL", 10)
    pos = broker.account().positions["SOL"]
    assert pos.leverage == 10 and pos.margin_used == pytest.approx(100.1)
    broker.set_leverage("SOL", 2)
    assert broker.account().positions["SOL"].margin_used == pytest.approx(500.5)
    assert buy(broker, 180).status is OrderStatus.FILLED   # deja muy poco margen libre
    with pytest.raises(ValueError):
        broker.set_leverage("SOL", 1)


def test_order_parameter_validations(broker):
    assert buy(broker, 1, coin="NOPE").status is OrderStatus.REJECTED
    r = buy(broker, 1, kind=LIMIT)
    assert r.status is OrderStatus.REJECTED and "price" in r.reason
    r = buy(broker, 1, kind=LIMIT, price=99.0, tif="Fok")
    assert r.status is OrderStatus.REJECTED and "tif" in r.reason
    assert buy(broker, 1, price=-1.0).status is OrderStatus.REJECTED
    with pytest.raises(ValueError):
        broker.place_order("SOL", True, 1, "teleport")


# --------------------------------------------------------------------------- cancelación y fills

def test_cancel_and_cancel_all(broker):
    a = buy(broker, 1, kind=LIMIT, price=95.0)
    b = buy(broker, 1, kind=LIMIT, price=94.0)
    c = buy(broker, 0.001, coin="BTC", kind=LIMIT, price=80_000.0)
    assert not broker.cancel("BTC", a.id)              # moneda equivocada
    assert broker.cancel("SOL", a.id)
    assert not broker.cancel("SOL", a.id)              # ya cancelada
    assert broker.get_order(a.id).status is OrderStatus.CANCELED
    assert broker.cancel_all("SOL") == 1
    assert [o.id for o in broker.open_orders()] == [c.id]
    assert broker.cancel_all() == 1
    assert broker.open_orders() == [] and b.id != c.id


def test_poll_fills_delivers_each_fill_once(broker):
    buy(broker, 1)
    sell(broker, 1)
    first = broker.poll_fills()
    assert len(first) == 2
    assert broker.poll_fills() == []
    buy(broker, 1)
    assert len(broker.poll_fills()) == 1


def test_returned_orders_are_copies(broker):
    o = buy(broker, 1, kind=LIMIT, price=95.0)
    o.price = 1.0
    listed = broker.open_orders()[0]
    listed.size = 999
    assert broker.open_orders()[0].price == 95.0 and broker.open_orders()[0].size == 1


def test_order_ids_unique_and_sequential(broker):
    ids = [buy(broker, 1).id, buy(broker, 0.0001).id, sell(broker, 1).id]   # el rechazado también tiene id
    assert len(set(ids)) == 3 and ids == sorted(ids)


# --------------------------------------------------------------------------- datos de mercado caídos

def test_market_data_failure_paths(broker, market):
    buy(broker, 10)
    market.fail_next = 1
    with pytest.raises(BrokerError):
        buy(broker, 1)                                 # sin precio no se puede ejecutar
    market.set_bid_ask("SOL", 104.9, 105.1)
    broker.account()                                   # mark conocido: 105
    market.fail_next = 1
    broker.step()                                      # no lanza: la moneda queda marcada
    assert "SOL" in broker.stale_coins
    market.fail_next = 1
    acct = broker.account()                            # usa el último mark conocido
    assert acct.positions["SOL"].mark_px == pytest.approx(105.0)
    broker.step()
    assert broker.stale_coins == {}


def test_metadata_outage_is_a_transport_error(broker, market, monkeypatch):
    from desk.exchange.base import MarketDataError

    def down(coin):
        raise MarketDataError("meta caída")

    monkeypatch.setattr(market, "meta", down)
    with pytest.raises(BrokerError, match="metadatos"):
        buy(broker, 1)


# --------------------------------------------------------------------------- persistencia

def test_state_roundtrip(market, cfg, clock, tmp_path):
    path = tmp_path / "state" / "paper.json"
    market.set_bid_ask("SOL", 99.9, 100.1)
    b1 = PaperBroker(market, cfg, clock, path, autosave=False)
    b1.set_leverage("SOL", 4)
    buy(b1, 10, tag="entry")
    sl = sell(b1, 10, kind=STOP, trigger_px=95.0, reduce_only=True, tag="sl")
    lim = buy(b1, 1, coin="ETH", kind=LIMIT, price=3_000.0)
    b1.save_state()
    assert path.exists() and not (path.parent / "paper.json.tmp").exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and len(data["pending_fills"]) == 1

    clock.advance(3600)
    b2 = PaperBroker(market, cfg, clock, path)
    assert b2.load_error == ""
    assert b2.cash == pytest.approx(b1.cash)
    pos = b2.account().positions["SOL"]
    assert (pos.size, pos.entry_px, pos.leverage) == (10, 100.1, 4)
    assert pos.margin_used == pytest.approx(1001 / 4)
    assert {o.id for o in b2.open_orders()} == {sl.id, lim.id}
    assert b2.leverage == {"SOL": 4}
    [f] = b2.poll_fills()                              # el fill no entregado sobrevive al reinicio
    assert f.tag == "entry" and f.price == 100.1
    new = buy(b2, 1)
    assert new.id not in {sl.id, lim.id} and new.id > lim.id
    # El estado recargado sigue funcionando: el stop se dispara.
    market.set_bid_ask("SOL", 94.0, 94.2)
    b2.step()
    assert b2.get_order(sl.id).status is OrderStatus.FILLED


def test_autosave_prevents_redelivery_after_restart(market, cfg, clock, tmp_path):
    path = tmp_path / "paper.json"
    market.set_bid_ask("SOL", 99.9, 100.1)
    b1 = PaperBroker(market, cfg, clock, path)
    buy(b1, 2)
    assert len(b1.poll_fills()) == 1
    b2 = PaperBroker(market, cfg, clock, path)         # "reinicio" sin save_state explícito
    assert b2.poll_fills() == []
    assert b2.account().positions["SOL"].size == 2


def test_corrupt_state_is_set_aside(market, cfg, clock, tmp_path):
    path = tmp_path / "paper.json"
    path.write_text("{ esto no es json", encoding="utf-8")
    b = PaperBroker(market, cfg, clock, path)
    assert "ilegible" in b.load_error
    assert b.cash == cfg.paper_start_equity and b.positions == {}
    assert list(tmp_path.glob("paper.json.corrupt-*"))


def test_no_state_file_means_no_persistence(broker, tmp_path):
    buy(broker, 1)
    broker.save_state()                                # no-op sin fichero
    assert list(tmp_path.iterdir()) == []


def test_from_settings_uses_state_dir(market, cfg, clock):
    b = PaperBroker.from_settings(market, cfg, clock)
    market.set_bid_ask("SOL", 99.9, 100.1)
    buy(b, 1)
    assert (cfg.state_path / "paper_broker.json").exists()

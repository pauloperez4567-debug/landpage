"""Tests del motor de ejecución: entradas, brackets SL/TP, OCO, caducidad, reconciliación y persistencia.

Se usan dos brokers:
- ``PaperBroker`` real sobre ``FakeMarketData`` (ejecuciones completas, como en paper).
- ``RecordingBroker`` (abajo): registra llamadas y permite fills parciales y rechazos a voluntad.
"""

from __future__ import annotations

import csv
from dataclasses import replace
import pytest

from desk.engine import CLOSING, OPEN, PENDING, PLACE_GRACE_S, ExecutionEngine, fmt_px, fmt_usd
from desk.exchange.base import Broker, BrokerError, MarketDataError
from desk.exchange.paper import PaperBroker
from desk.journal import Journal
from desk.models import (AccountState, EntryMode, Fill, Order, OrderKind, OrderStatus, Position, RiskDecision, Side,
                         TradeIntent)

LONG, SHORT = Side.LONG, Side.SHORT


class RecordingBroker(Broker):
    """Broker de juguete: guarda órdenes, permite fills parciales (``fill``) y rechazos por tipo."""

    def __init__(self):
        self.orders: dict[str, Order] = {}
        self.calls: list[tuple] = []
        self.positions: dict[str, Position] = {}
        self.equity = 10_000.0
        self.reject_kinds: set[OrderKind] = set()
        self.pending_fills: list[Fill] = []
        self.leverage: dict[str, int] = {}
        self._n = 0

    def account(self) -> AccountState:
        return AccountState(self.equity, self.equity, {c: replace(p) for c, p in self.positions.items()},
                            self.open_orders(), 0.0)

    def set_leverage(self, coin, leverage, cross=False):
        self.leverage[coin] = leverage

    def place_order(self, coin, is_buy, size, kind, *, price=None, trigger_px=None, tif="Gtc", reduce_only=False,
                    tag=""):
        self._n += 1
        o = Order(id=f"o{self._n}", coin=coin, is_buy=is_buy, size=size, kind=OrderKind(kind), price=price,
                  trigger_px=trigger_px, tif=tif, reduce_only=reduce_only, tag=tag)
        self.calls.append(("place", o.id, o.kind, tag, size, trigger_px if trigger_px else price))
        if o.kind in self.reject_kinds:
            o.status, o.reason = OrderStatus.REJECTED, "rechazo simulado"
            return replace(o)
        self.orders[o.id] = o
        return replace(o)

    def cancel(self, coin, order_id):
        self.calls.append(("cancel", order_id))
        o = self.orders.get(order_id)
        if o is not None and o.is_open:
            o.status = OrderStatus.CANCELED
            return True
        return False

    def open_orders(self, coin=None):
        return [replace(o) for o in self.orders.values() if o.is_open and (coin is None or o.coin == coin)]

    def poll_fills(self):
        out, self.pending_fills = self.pending_fills, []
        return out

    # --- control del test --------------------------------------------------
    def fill(self, oid: str, size: float, px: float, *, realized: float = 0.0, fee: float = 0.0,
             ts: float = 0.0) -> Fill:
        o = self.orders[oid]
        o.filled_size += size
        if o.filled_size >= o.size - 1e-12:
            o.status = OrderStatus.FILLED
        signed = size if o.is_buy else -size
        pos = self.positions.get(o.coin)
        new = (pos.size if pos else 0.0) + signed
        if abs(new) < 1e-12:
            self.positions.pop(o.coin, None)
        elif pos is None:
            self.positions[o.coin] = Position(o.coin, new, px, leverage=2, mark_px=px)
        else:
            pos.size = new
        return Fill(order_id=oid, coin=o.coin, is_buy=o.is_buy, size=size, price=px, fee=fee, ts=ts, tag=o.tag,
                    realized_pnl=realized)

    def placed(self, kind: OrderKind) -> list[tuple]:
        return [c for c in self.calls if c[0] == "place" and c[2] is kind]


# --------------------------------------------------------------------------- utilidades
def hype_intent(mode=EntryMode.LIMIT, side=LONG, origin="command"):
    return TradeIntent(coin="HYPE", side=side, units=1.0, entry_mode=mode, origin=origin,
                       limit_px=85.288 if mode is not EntryMode.MARKET else None, reason="test")


def hype_decision(entry=85.288, size=11.72, sl=84.009, tp=87.846):
    return RiskDecision(approved=True, reason="ok", leverage=2, size=size, notional=size * entry, entry_px=entry,
                        sl_price=sl, tp_price=tp)


@pytest.fixture
def journal(settings, clock):
    return Journal(settings.state_path, clock)


@pytest.fixture
def paper(market, settings, clock):
    return PaperBroker(market, settings, clock, settings.state_path / "paper_broker.json")


@pytest.fixture
def engine(paper, market, settings, clock, journal):
    return ExecutionEngine(paper, market, settings, clock, journal)


@pytest.fixture
def rec():
    return RecordingBroker()


@pytest.fixture
def rengine(rec, market, settings, clock, journal):
    return ExecutionEngine(rec, market, settings, clock, journal)


def by_tag(orders, tag):
    return [o for o in orders if o.tag == tag]


def trades(settings):
    path = settings.state_path / "trades.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- formato
def test_formatters():
    assert fmt_px(85.249) == "85.249" and fmt_px(83766.0) == "83766" and fmt_px(None) == "-"
    assert fmt_usd(1234.5) == "+$1,234.50" and fmt_usd(-5) == "-$5.00" and fmt_usd(1000, sign=False) == "$1,000.00"


# --------------------------------------------------------------------------- entradas
def test_open_limit_places_gtc_entry_and_pending_bracket(engine, paper, settings, clock):
    order = engine.open(hype_intent(), hype_decision(), cmd_id="c-1")
    assert order.status is OrderStatus.OPEN and order.kind is OrderKind.LIMIT
    assert order.price == 85.288 and order.tif == "Gtc" and order.tag == "entry" and not order.reduce_only
    assert paper.leverage["HYPE"] == 2
    br = engine.brackets["HYPE"]
    assert br.status == PENDING and br.entry_order_id == order.id and br.cmd_id == "c-1"
    assert (br.sl, br.tp) == (84.009, 87.846)
    assert br.expires_ts == pytest.approx(clock.now() + settings.entry_ttl_minutes * 60)
    assert not by_tag(paper.open_orders(), "sl")                   # SL/TP sólo al ejecutarse


def test_open_market_uses_worst_price_and_brackets_on_fill(engine, paper, market, settings):
    d = hype_decision(entry=86.29, size=11.58, sl=85.0, tp=88.87)
    order = engine.open(hype_intent(EntryMode.MARKET), d)
    assert order.status is OrderStatus.FILLED                       # paper ejecuta al instante
    worst = 86.29 * (1 + settings.market_slippage_bps / 10_000)
    assert order.avg_fill_px <= worst
    closed = engine.on_fills(paper.poll_fills())
    assert closed == []
    br = engine.brackets["HYPE"]
    assert br.status == OPEN and br.position == pytest.approx(11.58)
    sl, tp = by_tag(paper.open_orders(), "sl"), by_tag(paper.open_orders(), "tp")
    assert len(sl) == 1 and sl[0].kind is OrderKind.STOP and sl[0].trigger_px == 85.0 and sl[0].reduce_only
    assert len(tp) == 1 and tp[0].kind is OrderKind.TAKE_PROFIT and tp[0].trigger_px == 88.87 and tp[0].reduce_only
    assert sl[0].size == tp[0].size == 11.58 and not sl[0].is_buy
    msgs = " ".join(engine.pop_messages())
    assert "entrada ejecutada" in msgs and "SL 85 colocado" in msgs and "TP 88.87 colocado" in msgs


def test_market_entry_rejected_when_price_jumped_beyond_slippage(engine, paper, market):
    # Decisión tomada a 80 pero el ask ya está en 86.29: la MARKET con peor precio 80.08 se rechaza.
    order = engine.open(hype_intent(EntryMode.MARKET), hype_decision(entry=80.0, size=12.0, sl=79.0, tp=82.0))
    assert order.status is OrderStatus.REJECTED and "peor que el límite" in order.reason
    assert "HYPE" not in engine.brackets


def test_open_rejected_when_already_managed(engine):
    engine.open(hype_intent(), hype_decision())
    second = engine.open(hype_intent(), hype_decision())
    assert second.status is OrderStatus.REJECTED and "ya hay una operación gestionada" in second.reason


def test_open_requires_approved_decision(engine):
    with pytest.raises(ValueError):
        engine.open(hype_intent(), RiskDecision(approved=False, reason="no"))


def test_set_leverage_failure_is_a_local_reject(rengine, rec, monkeypatch):
    def boom(*a, **k):
        raise ValueError("margen insuficiente")
    monkeypatch.setattr(rec, "set_leverage", boom)
    order = rengine.open(hype_intent(), hype_decision())
    assert order.status is OrderStatus.REJECTED and "apalancamiento" in order.reason
    assert not rec.placed(OrderKind.LIMIT)


# --------------------------------------------------------------------------- brackets + OCO (paper)
def _fill_zone_entry(engine, paper, market):
    engine.open(hype_intent(), hype_decision())
    market.set_price("HYPE", 85.2)            # ask 85.204 <= 85.288: la compra límite se ejecuta (maker)
    paper.step()
    engine.on_fills(paper.poll_fills())
    engine.pop_messages()


def test_take_profit_triggers_oco_and_records_trade(engine, paper, market, settings):
    _fill_zone_entry(engine, paper, market)
    sl_id = engine.brackets["HYPE"].sl_order_id
    market.set_price("HYPE", 88.0)
    paper.step()
    closed = engine.on_fills(paper.poll_fills())
    assert len(closed) == 1
    t = closed[0]
    assert t.exit_reason == "tp" and t.coin == "HYPE" and t.side is LONG and t.size == pytest.approx(11.72)
    assert t.pnl > 0 and t.pnl == pytest.approx(t.gross_pnl - t.fees)
    assert "HYPE" not in engine.brackets
    assert paper.get_order(sl_id).status is OrderStatus.CANCELED   # OCO
    assert paper.open_orders() == []
    rows = trades(settings)
    assert len(rows) == 1 and rows[0]["exit_reason"] == "tp" and float(rows[0]["pnl"]) == pytest.approx(t.pnl)
    assert "SL cancelado (OCO)" in " ".join(engine.pop_messages())


def test_stop_loss_triggers_oco_and_negative_pnl(engine, paper, market):
    _fill_zone_entry(engine, paper, market)
    tp_id = engine.brackets["HYPE"].tp_order_id
    market.set_price("HYPE", 83.9)
    paper.step()
    closed = engine.on_fills(paper.poll_fills())
    assert len(closed) == 1 and closed[0].exit_reason == "sl" and closed[0].pnl < 0
    assert paper.get_order(tp_id).status is OrderStatus.CANCELED
    assert "TP cancelado (OCO)" in " ".join(engine.pop_messages())


def test_short_bracket_levels_are_mirrored(engine, paper, market):
    d = RiskDecision(approved=True, reason="ok", leverage=2, size=11.0, notional=950.0, entry_px=86.28,
                     sl_price=87.6, tp_price=83.6)
    engine.open(hype_intent(EntryMode.MARKET, side=SHORT), d)
    engine.on_fills(paper.poll_fills())
    sl = by_tag(paper.open_orders(), "sl")[0]
    tp = by_tag(paper.open_orders(), "tp")[0]
    assert sl.is_buy and tp.is_buy and sl.trigger_px == 87.6 and tp.trigger_px == 83.6
    market.set_price("HYPE", 88.0)
    paper.step()
    closed = engine.on_fills(paper.poll_fills())
    assert closed[0].exit_reason == "sl" and closed[0].pnl < 0


# --------------------------------------------------------------------------- fills parciales
def test_partial_entry_fills_resize_brackets_place_then_cancel(rengine, rec):
    order = rengine.open(hype_intent(), hype_decision())
    f1 = rec.fill(order.id, 5.0, 85.288, fee=0.06)
    rengine.on_fills([f1])
    br = rengine.brackets["HYPE"]
    assert br.status == OPEN and br.position == 5.0
    first_sl, first_tp = br.sl_order_id, br.tp_order_id
    assert rec.orders[first_sl].size == 5.0 and rec.orders[first_tp].size == 5.0
    assert "parcialmente" in " ".join(rengine.pop_messages())

    f2 = rec.fill(order.id, 6.72, 85.288, fee=0.08)
    rengine.on_fills([f2])
    assert br.position == pytest.approx(11.72)
    assert br.sl_order_id != first_sl and rec.orders[br.sl_order_id].size == pytest.approx(11.72)
    assert rec.orders[first_sl].status is OrderStatus.CANCELED
    assert rec.orders[first_tp].status is OrderStatus.CANCELED
    # La orden nueva se coloca ANTES de cancelar la vieja: la posición nunca queda sin stop.
    i_new = next(i for i, c in enumerate(rec.calls) if c[0] == "place" and c[1] == br.sl_order_id)
    i_cancel = rec.calls.index(("cancel", first_sl))
    assert i_new < i_cancel


def test_partial_exit_does_not_close_bracket(rengine, rec):
    order = rengine.open(hype_intent(), hype_decision())
    rengine.on_fills([rec.fill(order.id, 11.72, 85.288)])
    sl_id = rengine.brackets["HYPE"].sl_order_id
    rengine.on_fills([rec.fill(sl_id, 4.0, 84.0, realized=-5.0)])
    assert rengine.brackets["HYPE"].position == pytest.approx(7.72)
    closed = rengine.on_fills([rec.fill(sl_id, 7.72, 84.0, realized=-9.0, fee=0.3)])
    assert len(closed) == 1 and closed[0].exit_reason == "sl"
    assert closed[0].gross_pnl == pytest.approx(-14.0) and closed[0].pnl == pytest.approx(-14.3)


def test_oco_cancels_sibling_explicitly(rengine, rec):
    # A diferencia de paper, este broker NO barre las reduce-only al quedar plano: el OCO es del motor.
    order = rengine.open(hype_intent(), hype_decision())
    rengine.on_fills([rec.fill(order.id, 11.72, 85.288)])
    br = rengine.brackets["HYPE"]
    sl_id, tp_id = br.sl_order_id, br.tp_order_id
    closed = rengine.on_fills([rec.fill(tp_id, 11.72, 87.846, realized=29.98, fee=0.46)])
    assert closed[0].exit_reason == "tp" and closed[0].pnl == pytest.approx(29.52)
    assert rec.orders[sl_id].status is OrderStatus.CANCELED and ("cancel", tp_id) not in rec.calls
    assert rec.open_orders() == []


def test_stop_out_with_partial_entry_cancels_rest_of_entry(rengine, rec):
    order = rengine.open(hype_intent(), hype_decision())
    rengine.on_fills([rec.fill(order.id, 3.0, 85.288)])
    sl_id = rengine.brackets["HYPE"].sl_order_id
    rengine.on_fills([rec.fill(sl_id, 3.0, 84.0)])
    assert "HYPE" not in rengine.brackets
    assert rec.orders[order.id].status is OrderStatus.CANCELED       # resto de la entrada


# --------------------------------------------------------------------------- reconciliación
def test_entry_expires_after_ttl(engine, paper, clock, settings):
    order = engine.open(hype_intent(), hype_decision())
    engine.reconcile(paper.account())
    assert "HYPE" in engine.brackets
    clock.advance(settings.entry_ttl_minutes * 60 + 1)
    engine.reconcile(paper.account())
    assert "HYPE" not in engine.brackets
    assert paper.get_order(order.id).status is OrderStatus.CANCELED
    assert "caducada" in " ".join(engine.pop_messages())


def test_entry_missing_from_exchange_is_dropped_after_two_checks(rengine, rec):
    order = rengine.open(hype_intent(), hype_decision())
    rec.orders[order.id].status = OrderStatus.CANCELED      # cancelada fuera de la mesa
    rengine.reconcile(rec.account())
    assert "HYPE" in rengine.brackets                       # una vez puede ser latencia
    rengine.reconcile(rec.account())
    assert "HYPE" not in rengine.brackets


def test_position_without_seen_fill_is_protected(rengine, rec):
    order = rengine.open(hype_intent(), hype_decision())
    rec.fill(order.id, 11.72, 85.288)                        # el fill no se entrega (latencia)
    rengine.reconcile(rec.account())
    br = rengine.brackets["HYPE"]
    assert br.status == OPEN and br.sl_order_id and br.tp_order_id


def test_missing_stop_is_replaced(engine, paper, market, clock):
    _fill_zone_entry(engine, paper, market)
    old = engine.brackets["HYPE"].sl_order_id
    paper.cancel("HYPE", old)                                # alguien lo borró en el exchange
    engine.reconcile(paper.account())
    assert engine.brackets["HYPE"].sl_order_id == old        # recién colocada: periodo de gracia
    clock.advance(PLACE_GRACE_S + 1)
    engine.reconcile(paper.account())
    new = engine.brackets["HYPE"].sl_order_id
    assert new != old and paper.get_order(new).is_open and paper.get_order(new).trigger_px == 84.009
    assert "recolocado" in " ".join(engine.pop_messages())


def test_orphan_reduce_only_orders_are_cancelled(rengine, rec):
    o = rec.place_order("SOL", False, 1.0, OrderKind.STOP, trigger_px=140.0, reduce_only=True, tag="sl")
    rengine.reconcile(rec.account())
    assert rec.orders[o.id].status is OrderStatus.CANCELED
    assert "huérfana" in " ".join(rengine.pop_messages())


def test_position_adopted_with_default_stop(engine, paper, market, settings):
    paper.place_order("SOL", True, 10.0, OrderKind.MARKET)    # posición abierta fuera de la mesa
    paper.poll_fills()
    engine.reconcile(paper.account())
    br = engine.brackets["SOL"]
    pos = paper.account().positions["SOL"]
    assert br.origin == "adopted" and br.status == OPEN and br.position == 10.0
    expected = round(pos.entry_px * (1 - settings.default_sl_pct / 100), 3)
    assert br.sl == pytest.approx(expected, abs=0.01) and br.tp is None
    sl = by_tag(paper.open_orders("SOL"), "sl")
    assert len(sl) == 1 and sl[0].trigger_px == br.sl and sl[0].size == 10.0
    assert "adoptada" in " ".join(engine.pop_messages())


def test_adoption_reuses_existing_stop(engine, paper):
    paper.place_order("SOL", True, 10.0, OrderKind.MARKET)
    existing = paper.place_order("SOL", False, 10.0, OrderKind.STOP, trigger_px=140.0, reduce_only=True, tag="sl")
    engine.reconcile(paper.account())
    br = engine.brackets["SOL"]
    assert br.sl == 140.0 and br.sl_order_id == existing.id
    assert len(by_tag(paper.open_orders("SOL"), "sl")) == 1      # no duplica


def test_flat_position_closes_bracket_as_external(rengine, rec):
    order = rengine.open(hype_intent(), hype_decision())
    rengine.on_fills([rec.fill(order.id, 11.72, 85.288)])
    rec.positions.clear()                                        # cerrada a mano en el exchange
    assert rengine.reconcile(rec.account()) == []
    closed = rengine.reconcile(rec.account())
    assert len(closed) == 1 and closed[0].exit_reason == "external"


def test_stop_rejected_repeatedly_closes_position(rengine, rec):
    rec.reject_kinds.add(OrderKind.STOP)
    order = rengine.open(hype_intent(), hype_decision())
    rengine.on_fills([rec.fill(order.id, 11.72, 85.288)])
    for _ in range(3):
        rengine.reconcile(rec.account())
    closes = [c for c in rec.calls if c[0] == "place" and c[3] == "no_sl"]
    assert len(closes) == 1 and rengine.brackets["HYPE"].status == CLOSING
    assert "por seguridad" in " ".join(rengine.pop_messages())


# --------------------------------------------------------------------------- cierres, cancelaciones, niveles
def test_close_pending_entry_cancels_it(engine, paper):
    order = engine.open(hype_intent(), hype_decision())
    assert engine.close("HYPE") is None
    assert paper.get_order(order.id).status is OrderStatus.CANCELED and "HYPE" not in engine.brackets


def test_close_position_market_reduce_only(engine, paper, market):
    _fill_zone_entry(engine, paper, market)
    order = engine.close("HYPE")
    assert order.kind is OrderKind.MARKET and order.reduce_only and order.tag == "close"
    closed = engine.on_fills(paper.poll_fills())
    assert closed[0].exit_reason == "close" and paper.account().positions == {}
    assert paper.open_orders() == []


def test_close_all_and_cancel(engine, paper, market):
    _fill_zone_entry(engine, paper, market)
    paper.place_order("SOL", True, 10.0, OrderKind.LIMIT, price=140.0, tag="entry")   # entrada en reposo
    orders = engine.close_all()
    assert len(orders) == 1
    engine.on_fills(paper.poll_fills())
    assert paper.account().positions == {} and paper.open_orders() == []


def test_cancel_only_touches_entries(engine, paper, market):
    _fill_zone_entry(engine, paper, market)
    engine.open(TradeIntent("SOL", LONG, 1, EntryMode.LIMIT, "command", limit_px=140.0),
                RiskDecision(True, "ok", 5, 10.0, 1400.0, 140.0, 138.0, 144.0))
    assert engine.cancel(None) == 1
    assert "SOL" not in engine.brackets
    assert len(by_tag(paper.open_orders("HYPE"), "sl")) == 1     # el SL de la posición sigue


def test_set_stop_moves_order(engine, paper, market):
    _fill_zone_entry(engine, paper, market)
    old = engine.brackets["HYPE"].sl_order_id
    msg = engine.set_stop("HYPE", 84.5)
    assert "84.5" in msg and "antes 84.009" in msg
    br = engine.brackets["HYPE"]
    assert br.sl == 84.5 and br.sl_order_id != old
    assert paper.get_order(old).status is OrderStatus.CANCELED
    assert paper.get_order(br.sl_order_id).trigger_px == 84.5


def test_set_take_profit_and_validation(engine, paper, market):
    _fill_zone_entry(engine, paper, market)
    assert "TP movido a 90" in engine.set_take_profit("HYPE", 90)
    with pytest.raises(ValueError, match="por debajo del precio actual"):
        engine.set_stop("HYPE", 86.0)                   # SL de un long por encima del precio
    with pytest.raises(ValueError, match="por encima del precio actual"):
        engine.set_take_profit("HYPE", 80.0)
    with pytest.raises(ValueError, match="no hay posición"):
        engine.set_stop("SOL", 100.0)


def test_set_stop_on_pending_entry(engine):
    engine.open(hype_intent(), hype_decision())
    assert "entrada pendiente" in engine.set_stop("HYPE", 84.2)
    assert engine.brackets["HYPE"].sl == 84.2
    with pytest.raises(ValueError):
        engine.set_stop("HYPE", 86.0)                   # por encima de la entrada 85.288


def test_unmatched_fill_is_reported(rengine):
    f = Fill(order_id="x", coin="ETH", is_buy=True, size=0.1, price=3200.0, fee=0.1, ts=0.0, tag="")
    assert rengine.on_fills([f]) == []
    assert "fuera de una operación gestionada" in " ".join(rengine.pop_messages())


# --------------------------------------------------------------------------- persistencia
def test_state_persists_and_oco_works_after_restart(engine, paper, market, settings, clock, journal):
    _fill_zone_entry(engine, paper, market)
    sl_id = engine.brackets["HYPE"].sl_order_id
    paper2 = PaperBroker(market, settings, clock, settings.state_path / "paper_broker.json")
    engine2 = ExecutionEngine(paper2, market, settings, clock, Journal(settings.state_path, clock))
    br = engine2.brackets["HYPE"]
    assert br.status == OPEN and br.sl_order_id == sl_id and br.side is LONG and br.origin == "command"
    engine2.reconcile(paper2.account())
    assert engine2.brackets["HYPE"].sl_order_id == sl_id              # nada que recolocar
    market.set_price("HYPE", 88.0)
    paper2.step()
    closed = engine2.on_fills(paper2.poll_fills())
    assert closed[0].exit_reason == "tp" and paper2.get_order(sl_id).status is OrderStatus.CANCELED


def test_corrupt_engine_state_is_ignored(settings, clock, market, paper):
    j = Journal(settings.state_path, clock)
    (settings.state_path / "engine.json").write_text("{no es json", encoding="utf-8")
    e = ExecutionEngine(paper, market, settings, clock, j)
    assert e.brackets == {}
    assert list(settings.state_path.glob("engine.json.corrupt-*"))


# --------------------------------------------------------------------------- regresiones (revisión)
class FlakyBroker(RecordingBroker):
    """RecordingBroker con fallos de transporte a voluntad: ``fail`` = {(moneda, tipo)} lanza ``BrokerError``;
    con ``after`` la orden llega al exchange antes del fallo (timeout tras aceptarla)."""

    def __init__(self):
        super().__init__()
        self.fail: set[tuple[str, OrderKind]] = set()
        self.after = False
        self.cancel_ok = True

    def place_order(self, coin, is_buy, size, kind, **kw):
        if (coin, OrderKind(kind)) in self.fail:
            if self.after:
                super().place_order(coin, is_buy, size, kind, **kw)
            raise BrokerError(f"order {coin}: ReadTimeout")
        return super().place_order(coin, is_buy, size, kind, **kw)

    def cancel(self, coin, order_id):
        if not self.cancel_ok:
            self.calls.append(("cancel-fail", order_id))
            return False
        return super().cancel(coin, order_id)


@pytest.fixture
def flaky():
    return FlakyBroker()


@pytest.fixture
def fengine(flaky, market, settings, clock, journal):
    return ExecutionEngine(flaky, market, settings, clock, journal)


def _sol_long(engine, paper, size=31.91, lev=5, sl=147.02, tp=156.02):
    d = RiskDecision(True, "ok", lev, size, size * 150.02, 150.02, sl, tp)
    engine.open(TradeIntent("SOL", LONG, 1, EntryMode.MARKET, "command"), d)
    engine.on_fills(paper.poll_fills())
    engine.pop_messages()


def test_set_stop_beyond_liquidation_or_risk_budget_is_rejected(engine, paper):
    # 'stop SOL 1' (un % sin el símbolo): antes se aceptaba y la posición se liquidaba antes que el stop.
    _sol_long(engine, paper)
    sl_id = engine.brackets["SOL"].sl_order_id
    with pytest.raises(ValueError, match="más allá de la liquidación"):
        engine.set_stop("SOL", 1.0)
    with pytest.raises(ValueError, match="máximo por operación"):
        engine.set_stop("SOL", 125.0)                  # antes de la liquidación, pero ~8x el riesgo aprobado
    br = engine.brackets["SOL"]
    assert br.sl == 147.02 and br.sl_order_id == sl_id and paper.get_order(sl_id).is_open
    assert "SL movido a 148" in engine.set_stop("SOL", 148.0)       # acercarlo siempre se puede


def test_set_stop_on_pending_entry_cannot_widen_beyond_budget_or_liquidation(engine):
    engine.open(hype_intent(), hype_decision())        # 11.72 HYPE @ 85.288, 2x, riesgo aprobado ~$15
    with pytest.raises(ValueError, match="liquidación"):
        engine.set_stop("HYPE", 10.0)
    with pytest.raises(ValueError, match="máximo por operación"):
        engine.set_stop("HYPE", 60.0)
    assert engine.brackets["HYPE"].sl == 84.009


def test_adopted_position_default_stop_never_beyond_liquidation(rengine, rec):
    rec.positions["SOL"] = Position("SOL", 10.0, 150.0, leverage=50, mark_px=150.0, liq_px=148.0)
    rengine.reconcile(rec.account())
    br = rengine.brackets["SOL"]
    assert br.origin == "adopted" and 148.0 < br.sl < 150.0        # el 1.5 % por defecto (147.75) no vale
    assert br.sl >= 150.0 - (150.0 - 148.0) / 1.10 - 0.01


def test_failed_cancel_on_expiry_keeps_the_planned_bracket(fengine, flaky, clock, settings):
    order = fengine.open(hype_intent(), hype_decision(entry=85.288, size=100.0, sl=85.032, tp=85.80))
    flaky.cancel_ok = False
    clock.advance(settings.entry_ttl_minutes * 60 + 1)
    fengine.reconcile(flaky.account())
    br = fengine.brackets["HYPE"]
    assert br.status == PENDING and br.canceling
    assert "no se confirmó" in " ".join(fengine.pop_messages())
    fengine.on_fills([flaky.fill(order.id, 100.0, 85.288)])        # se ejecuta antes de poder cancelarla
    br = fengine.brackets["HYPE"]
    assert br.status == OPEN and br.origin == "command" and (br.sl, br.tp) == (85.032, 85.80)
    assert [c[5] for c in flaky.placed(OrderKind.STOP)] == [85.032]
    assert [c[5] for c in flaky.placed(OrderKind.TAKE_PROFIT)] == [85.80]


def test_failed_cancel_is_retried_and_bracket_dropped_once_confirmed(fengine, flaky):
    order = fengine.open(hype_intent(), hype_decision())
    flaky.cancel_ok = False
    assert fengine.cancel("HYPE") == 0
    assert fengine.brackets["HYPE"].canceling                       # no se anuncia "cancelada"
    flaky.cancel_ok = True
    fengine.reconcile(flaky.account())
    assert "HYPE" not in fengine.brackets and flaky.orders[order.id].status is OrderStatus.CANCELED


def test_close_is_retried_while_position_remains(rengine, rec, clock, market):
    from desk.engine import CLOSE_RETRY_S, MAX_CLOSE_ATTEMPTS
    rec.positions["SOL"] = Position("SOL", 100.0, 155.0, leverage=5, mark_px=150.0, liq_px=125.0)
    rengine.reconcile(rec.account())
    first = rengine.close("SOL")
    # Cierre con tolerancia amplia (5 % del mid), no los 10 bps de las entradas.
    assert first.reduce_only and first.price == pytest.approx(market.mid("SOL") * 0.95, rel=1e-4)
    rengine.on_fills([rec.fill(first.id, 40.0, 149.9)])             # IOC parcial: quedan 60
    assert rengine.brackets["SOL"].status == CLOSING
    clock.advance(CLOSE_RETRY_S)
    rengine.reconcile(rec.account())
    closes = [c for c in rec.placed(OrderKind.MARKET)]
    assert len(closes) == 2 and closes[-1][4] == 60.0
    assert closes[-1][5] == pytest.approx(market.mid("SOL") * 0.90, rel=1e-4)   # más tolerancia al reintentar
    rec.reject_kinds = {OrderKind.MARKET}                           # y si el exchange lo rechaza, se sigue intentando
    for _ in range(MAX_CLOSE_ATTEMPTS + 2):
        clock.advance(CLOSE_RETRY_S)
        rengine.reconcile(rec.account())
    assert len(rec.placed(OrderKind.MARKET)) == MAX_CLOSE_ATTEMPTS
    assert "no se completó" in " ".join(rengine.pop_messages())


def test_late_entry_fill_after_reconcile_is_not_counted_twice(rengine, rec):
    order = rengine.open(hype_intent(), hype_decision())
    late = rec.fill(order.id, 11.72, 85.288)                        # la cuenta lo ve antes que el fill
    rengine.reconcile(rec.account())
    assert rengine.brackets["HYPE"].position == 11.72
    rengine.on_fills([late])
    br = rengine.brackets["HYPE"]
    assert (br.position, br.filled, br.sl_size) == (11.72, 11.72, 11.72)
    assert len(rec.placed(OrderKind.STOP)) == 1
    closed = rengine.on_fills([rec.fill(br.sl_order_id, 11.72, 84.0, realized=-15.1)])
    assert closed[0].size == 11.72 and closed[0].exit_reason == "sl"


def test_close_all_keeps_going_when_one_coin_fails(engine, paper, market, monkeypatch):
    for coin, size in (("BTC", 0.03), ("ETH", 0.8), ("SOL", 16.0)):
        paper.place_order(coin, True, size, OrderKind.MARKET)
    paper.poll_fills()
    real = market.bid_ask

    def no_btc(coin):
        if coin == "BTC":
            raise MarketDataError("BTC: libro vacío por un lado")
        return real(coin)
    monkeypatch.setattr(market, "bid_ask", no_btc)
    orders = engine.close_all("kill")
    assert sorted(o.coin for o in orders) == ["ETH", "SOL"]
    assert list(engine.close_failures) == ["BTC"]
    engine.on_fills(paper.poll_fills())
    assert list(paper.account().positions) == ["BTC"]


def test_entry_timeout_keeps_approved_bracket_and_protects_the_fill(fengine, flaky):
    flaky.fail, flaky.after = {("HYPE", OrderKind.MARKET)}, True
    with pytest.raises(BrokerError):
        fengine.open(hype_intent(EntryMode.MARKET), hype_decision(entry=86.29, size=11.58, sl=85.9, tp=87.07))
    assert fengine.is_unconfirmed("HYPE")
    oid = next(iter(flaky.orders))
    flaky.fill(oid, 11.58, 86.29)                                   # la orden sí se ejecutó en el exchange
    flaky.fail = set()
    fengine.reconcile(flaky.account())
    br = fengine.brackets["HYPE"]
    assert br.origin == "command" and br.status == OPEN and (br.sl, br.tp) == (85.9, 87.07)
    assert [c[5] for c in flaky.placed(OrderKind.STOP)] == [85.9]  # no el SL por defecto de una adopción


def test_unconfirmed_limit_entry_is_found_or_dropped(fengine, flaky, clock, settings):
    flaky.fail, flaky.after = {("HYPE", OrderKind.LIMIT)}, True
    with pytest.raises(BrokerError):
        fengine.open(hype_intent(), hype_decision())
    flaky.fail = set()
    fengine.reconcile(flaky.account())
    br = fengine.brackets["HYPE"]
    assert br.entry_order_id == "o1" and "está en el exchange" in " ".join(fengine.pop_messages())
    clock.advance(settings.entry_ttl_minutes * 60 + 1)              # y caduca con su TTL
    fengine.reconcile(flaky.account())
    assert "HYPE" not in fengine.brackets and flaky.orders["o1"].status is OrderStatus.CANCELED
    flaky.fail, flaky.after = {("SOL", OrderKind.LIMIT)}, False    # nunca llegó: se descarta
    with pytest.raises(BrokerError):
        fengine.open(TradeIntent("SOL", LONG, 1, EntryMode.LIMIT, "command", limit_px=140.0),
                     RiskDecision(True, "ok", 5, 10.0, 1400.0, 140.0, 138.0, 144.0))
    fengine.reconcile(flaky.account())
    fengine.reconcile(flaky.account())
    assert "SOL" not in fengine.brackets and "no aparece en el exchange" in " ".join(fengine.pop_messages())


def test_stop_failure_on_one_coin_does_not_lose_other_coins_trades(fengine, flaky):
    btc = fengine.open(TradeIntent("BTC", LONG, 1, EntryMode.LIMIT, "command", limit_px=83_000.0),
                       RiskDecision(True, "ok", 5, 0.03, 2490.0, 83_000.0, 82_000.0, 85_000.0))
    fengine.on_fills([flaky.fill(btc.id, 0.03, 83_000.0)])
    sol = fengine.open(TradeIntent("SOL", LONG, 1, EntryMode.LIMIT, "command", limit_px=149.0),
                       RiskDecision(True, "ok", 5, 16.0, 2384.0, 149.0, 147.0, 153.0))
    flaky.fail = {("SOL", OrderKind.STOP)}
    batch = [flaky.fill(fengine.brackets["BTC"].sl_order_id, 0.03, 82_000.0, realized=-30.0),
             flaky.fill(sol.id, 16.0, 149.0)]
    closed = fengine.on_fills(batch)
    assert [t.coin for t in closed] == ["BTC"]                     # la pérdida de BTC llega al riesgo
    br = fengine.brackets["SOL"]
    assert br.status == OPEN and br.sl_errors == 1 and br.tp_order_id
    from desk.engine import MAX_SL_ERRORS
    for _ in range(MAX_SL_ERRORS):                                  # SL imposible por conexión: cierre de seguridad
        fengine.reconcile(flaky.account())
    assert fengine.brackets["SOL"].status == CLOSING and fengine.brackets["SOL"].exit_reason == "no_sl"


def test_reconcile_failure_on_one_coin_does_not_starve_the_others(fengine, flaky, clock):
    from desk.engine import CLOSE_RETRY_S
    flaky.positions["BTC"] = Position("BTC", 0.03, 83_000.0, leverage=5, mark_px=83_000.0)
    flaky.positions["SOL"] = Position("SOL", 16.0, 150.0, leverage=5, mark_px=150.0)
    flaky.fail = {("BTC", OrderKind.MARKET), ("BTC", OrderKind.STOP)}
    fengine.reconcile(flaky.account())                              # adopta las dos; el SL de BTC falla
    with pytest.raises(BrokerError):
        fengine.close("BTC")                                        # el cierre de BTC también falla (red)
    flaky.orders.clear()                                            # SL/TP de SOL desaparecen del exchange
    clock.advance(CLOSE_RETRY_S)
    fengine.reconcile(flaky.account())                              # BTC vuelve a fallar...
    assert {o.coin for o in flaky.open_orders() if o.kind is OrderKind.STOP} == {"SOL"}   # ...SOL se protege


def test_redelivered_fills_after_restart_are_ignored(engine, paper, market, settings, clock, journal):
    d = hype_decision(entry=86.29, size=11.58, sl=85.0, tp=88.87)
    engine.open(hype_intent(EntryMode.MARKET), d)
    fills = paper.poll_fills()
    engine.on_fills(fills)
    br = engine.brackets["HYPE"]
    before = (br.filled, br.position, br.fees, br.sl_order_id)
    engine2 = ExecutionEngine(paper, market, settings, clock, Journal(settings.state_path, clock))   # reinicio
    assert engine2.on_fills(list(fills)) == []                      # cursor inclusivo: el último fill otra vez
    br2 = engine2.brackets["HYPE"]
    assert (br2.filled, br2.position, br2.fees, br2.sl_order_id) == before
    rows = list(csv.DictReader(open(settings.state_path / "fills.csv", encoding="utf-8")))
    assert len(rows) == 1
    twin = replace(fills[0], order_id="otra")                        # mismo instante, otra orden: sí cuenta
    engine2.on_fills([twin])
    assert engine2.brackets["HYPE"].filled == pytest.approx(2 * 11.58)


def test_old_exit_fill_is_not_imputed_to_a_new_trade(rengine, rec, clock):
    # Re-entrega tras reiniciar: el cierre de la operación ANTERIOR no puede caer en el bracket nuevo (falseaba su
    # PnL y el que va al riesgo).
    old_ts = clock.now()
    clock.advance(60)
    order = rengine.open(hype_intent(EntryMode.MARKET), hype_decision(entry=86.29, size=11.58, sl=85.0, tp=88.87))
    rengine.on_fills([rec.fill(order.id, 11.58, 86.29, ts=clock.now())])
    old_exit = Fill(order_id="viejo", coin="HYPE", is_buy=False, size=11.58, price=84.0, fee=0.4, ts=old_ts,
                    tag="sl", realized_pnl=-23.4)
    assert rengine.on_fills([old_exit]) == []
    br = rengine.brackets["HYPE"]
    assert br.position == 11.58 and br.realized == 0.0 and br.exit_size == 0.0
    assert "fuera de una operación gestionada" in " ".join(rengine.pop_messages())


def test_identical_partial_fills_in_separate_batches_without_time_are_applied(rengine, rec):
    order = rengine.open(hype_intent(), hype_decision())
    rengine.on_fills([rec.fill(order.id, 3.0, 85.288)])
    rengine.on_fills([rec.fill(order.id, 3.0, 85.288)])             # otra contrapartida igual (sin hora)
    assert rengine.brackets["HYPE"].filled == 6.0

"""Broker simulado (paper trading) con margen aislado, comisiones y persistencia.

Modelo (deliberadamente simple y predecible, ver ARCHITECTURE.md):

- ``cash`` = equity inicial + PnL realizado − comisiones. ``equity`` = cash + PnL no realizado
  (marcado al mid). ``available`` = equity − margen de posiciones − margen reservado por
  órdenes límite en reposo que aumentan exposición.
- Margen aislado: ``|size| · entry_px / leverage`` por posición. Liquidación cuando
  margen + PnL ≤ margen de mantenimiento (``paper_maintenance_margin_pct`` del notional al mark).
  La liquidación ejecuta al mark, pero nunca peor que el precio de quiebra: en aislado la
  pérdida queda limitada al margen aunque el precio salte por encima de la liquidación.
- Liquidez infinita al mejor precio: las órdenes se ejecutan completas (sin fills parciales).
  MARKET y disparos SL/TP pagan taker + deslizamiento; los límites en reposo pagan maker al
  precio límite.
- Órdenes reduce-only exentas del notional mínimo (siempre se puede cerrar un resto pequeño).

Todas las horas salen del ``Clock`` inyectado. Las órdenes devueltas son copias: el estado
real se consulta con ``open_orders()`` / ``account()`` / ``poll_fills()`` como en live.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Union

from ..clock import Clock
from ..config import Settings
from ..models import AccountState, AssetMeta, Fill, Order, OrderKind, OrderStatus, Position
from .base import Broker, BrokerError, MarketData, MarketDataError, round_price, round_size

log = logging.getLogger(__name__)

STATE_VERSION = 1
VALID_TIFS = ("Gtc", "Alo", "Ioc")
TRIGGER_KINDS = (OrderKind.STOP, OrderKind.TAKE_PROFIT)
DEFAULT_STATE_NAME = "paper_broker.json"


class _Reject(Exception):
    """Rechazo de negocio: la orden vuelve con ``status=REJECTED`` y este motivo."""


class PaperBroker(Broker):
    def __init__(self, market: MarketData, settings: Settings, clock: Clock,
                 state_file: Optional[Union[str, Path]] = None, *, autosave: bool = True,
                 history: int = 500):
        self.market = market
        self.settings = settings
        self.clock = clock
        self.state_file = Path(state_file) if state_file else None
        # Con fichero de estado, cada cambio (orden, fill, entrega de fills) se guarda al momento:
        # un reinicio nunca re-entrega fills ya consumidos ni pierde órdenes.
        self.autosave = autosave and self.state_file is not None
        self._lock = threading.RLock()
        self._history = history
        self.load_error = ""
        self.stale_coins: dict[str, str] = {}   # monedas sin precio en el último step()
        self._reset()
        if self.state_file is not None and self.state_file.exists():
            self._load()

    @classmethod
    def from_settings(cls, market: MarketData, settings: Settings, clock: Clock, **kwargs: Any) -> "PaperBroker":
        return cls(market, settings, clock, settings.state_path / DEFAULT_STATE_NAME, **kwargs)

    def _reset(self) -> None:
        self.start_equity = float(self.settings.paper_start_equity)
        self.cash = self.start_equity
        self.positions: dict[str, Position] = {}
        self.leverage: dict[str, int] = {}
        self._orders: dict[str, Order] = {}                  # abiertas, en orden de llegada
        self._closed: deque[Order] = deque(maxlen=self._history)
        self._pending: list[Fill] = []                       # fills aún no entregados
        self._marks: dict[str, float] = {}
        self._next_id = 1
        self.total_fees = 0.0
        self.total_realized = 0.0
        self.fill_count = 0
        self._dirty = False

    # ------------------------------------------------------------------ parámetros
    @property
    def taker_rate(self) -> float:
        return self.settings.paper_taker_fee_bps / 10_000.0

    @property
    def maker_rate(self) -> float:
        return self.settings.paper_maker_fee_bps / 10_000.0

    @property
    def slippage(self) -> float:
        return self.settings.paper_slippage_bps / 10_000.0

    @property
    def mmr(self) -> float:
        return self.settings.paper_maintenance_margin_pct / 100.0

    # ------------------------------------------------------------------ utilidades
    def _new_id(self) -> str:
        oid = f"paper-{self._next_id:06d}"
        self._next_id += 1
        return oid

    def _meta(self, coin: str) -> Optional[AssetMeta]:
        """Metadatos o None si la moneda no existe. Un fallo de datos es de transporte: ``BrokerError``."""
        try:
            return self.market.meta(coin)
        except KeyError:
            return None
        except MarketDataError as e:
            raise BrokerError(f"sin metadatos de {coin}: {e}") from e

    def _lev(self, coin: str, meta: Optional[AssetMeta] = None) -> int:
        if coin in self.leverage:
            return self.leverage[coin]
        default = max(int(self.settings.leverage), 1)
        if meta is None:
            try:
                meta = self._meta(coin)
            except BrokerError:
                return default
        return max(1, min(default, meta.max_leverage)) if meta else default

    def _quote(self, coin: str) -> tuple[float, float]:
        try:
            bid, ask = self.market.bid_ask(coin)
        except MarketDataError as e:
            raise BrokerError(f"sin precio de {coin}: {e}") from e
        self._marks[coin] = (bid + ask) / 2.0
        return bid, ask

    def _mark(self, coin: str) -> float:
        """Mid actual; si los datos fallan, el último conocido (o la entrada)."""
        try:
            bid, ask = self.market.bid_ask(coin)
        except MarketDataError:
            if coin in self._marks:
                return self._marks[coin]
            pos = self.positions.get(coin)
            return pos.entry_px if pos else math.nan
        self._marks[coin] = (bid + ask) / 2.0
        return self._marks[coin]

    def _taker_px(self, is_buy: bool, bid: float, ask: float) -> float:
        return ask * (1.0 + self.slippage) if is_buy else bid * (1.0 - self.slippage)

    def _increase_part(self, coin: str, is_buy: bool, size: float) -> float:
        """Parte de una orden que abre o aumenta exposición (lo que cierra no necesita margen)."""
        pos = self.positions.get(coin)
        if pos is None or (pos.size > 0) == is_buy:
            return size
        return max(0.0, size - abs(pos.size))

    def _reserved(self) -> float:
        total = 0.0
        for o in self._orders.values():
            if o.kind is OrderKind.LIMIT and not o.reduce_only and o.price:
                total += self._increase_part(o.coin, o.is_buy, o.remaining) * o.price / self._lev(o.coin)
        return total

    def _available(self) -> float:
        upnl = sum(p.size * (self._mark(c) - p.entry_px) for c, p in self.positions.items())
        margin = sum(p.margin_used for p in self.positions.values())
        return self.cash + upnl - margin - self._reserved()

    def _liq_px(self, pos: Position) -> Optional[float]:
        size = abs(pos.size)
        if size <= 0:
            return None
        per_unit = pos.margin_used / size
        if pos.size > 0:
            px = (pos.entry_px - per_unit) / (1.0 - self.mmr)
            return px if px > 0 else None
        return (pos.entry_px + per_unit) / (1.0 + self.mmr)

    def _bankruptcy_px(self, pos: Position) -> float:
        per_unit = pos.margin_used / abs(pos.size)
        return pos.entry_px - per_unit if pos.size > 0 else pos.entry_px + per_unit

    def _autosave(self) -> None:
        if self.autosave and self._dirty:
            try:
                self.save_state()
            except BrokerError as e:      # no romper el flujo de órdenes por un fallo de disco
                log.error("%s", e)

    # ------------------------------------------------------------------ Broker
    def account(self) -> AccountState:
        with self._lock:
            positions: dict[str, Position] = {}
            upnl_total = margin = 0.0
            for coin, pos in self.positions.items():
                mark = self._mark(coin)
                upnl = pos.size * (mark - pos.entry_px)
                positions[coin] = replace(pos, mark_px=mark, unrealized_pnl=upnl, liq_px=self._liq_px(pos))
                upnl_total += upnl
                margin += pos.margin_used
            equity = self.cash + upnl_total
            available = max(0.0, equity - margin - self._reserved())
            return AccountState(equity=equity, available=available, positions=positions,
                                open_orders=self.open_orders(), ts=self.clock.now())

    def set_leverage(self, coin: str, leverage: int, cross: bool = False) -> None:
        """Apalancamiento aislado para la moneda (``cross`` se ignora: paper siempre es aislado).

        Lanza ``ValueError`` si la moneda no existe, el valor no es entero en [1, max_leverage] o,
        con posición abierta, no hay margen libre para bajar el apalancamiento.
        """
        with self._lock:
            meta = self._meta(coin)
            if meta is None:
                raise ValueError(f"moneda desconocida: {coin}")
            if isinstance(leverage, bool) or int(leverage) != leverage:
                raise ValueError(f"apalancamiento inválido: {leverage!r} (debe ser entero)")
            lev = int(leverage)
            if not 1 <= lev <= meta.max_leverage:
                raise ValueError(f"apalancamiento {lev}x fuera de rango para {coin} (1..{meta.max_leverage}x)")
            pos = self.positions.get(coin)
            if pos is not None and pos.leverage != lev:
                new_margin = abs(pos.size) * pos.entry_px / lev
                extra = new_margin - pos.margin_used
                if extra > self._available() + 1e-9:
                    raise ValueError(f"margen insuficiente para bajar {coin} a {lev}x (faltan {extra:.2f} USD)")
                pos.leverage, pos.margin_used = lev, new_margin
                self._dirty = True
            if self.leverage.get(coin) != lev:
                self.leverage[coin] = lev
                self._dirty = True
            self._autosave()

    def place_order(self, coin: str, is_buy: bool, size: float, kind: OrderKind, *,
                    price: Optional[float] = None, trigger_px: Optional[float] = None,
                    tif: str = "Gtc", reduce_only: bool = False, tag: str = "") -> Order:
        kind = OrderKind(kind)
        with self._lock:
            now = self.clock.now()
            order = Order(id=self._new_id(), coin=coin, is_buy=bool(is_buy), size=float(size), kind=kind,
                          price=None if price is None else float(price),
                          trigger_px=None if trigger_px is None else float(trigger_px),
                          tif=tif, reduce_only=bool(reduce_only), created_ts=now, updated_ts=now, tag=tag)
            self._dirty = True
            try:
                self._submit(order)
            except _Reject as r:
                order.status, order.reason, order.updated_ts = OrderStatus.REJECTED, str(r), now
                self._closed.append(order)
                log.info("orden rechazada %s %s %s: %s", coin, "compra" if is_buy else "venta", kind.value, r)
            self._autosave()
            return replace(order)

    def _submit(self, order: Order) -> None:
        coin = order.coin
        meta = self._meta(coin)
        if meta is None:
            raise _Reject(f"moneda desconocida: {coin}")
        if not math.isfinite(order.size) or order.size <= 0:
            raise _Reject(f"tamaño inválido: {order.size}")
        if order.kind is OrderKind.LIMIT:
            if order.tif not in VALID_TIFS:
                raise _Reject(f"tif inválido {order.tif!r} (usa Gtc, Alo o Ioc)")
            if order.price is None or not order.price > 0:
                raise _Reject("LIMIT requiere price > 0")
            order.price = round_price(order.price, meta.sz_decimals)
        elif order.kind in TRIGGER_KINDS:
            if order.trigger_px is None or not order.trigger_px > 0:
                raise _Reject(f"{order.kind.value} requiere trigger_px > 0")
            if not order.reduce_only:
                raise _Reject("STOP / TAKE_PROFIT deben ser reduce_only=True")
            order.trigger_px = round_price(order.trigger_px, meta.sz_decimals)
        elif order.price is not None and not order.price > 0:
            raise _Reject("MARKET: price (peor precio aceptable) debe ser > 0")

        min_size = 10.0 ** -meta.sz_decimals
        size = round_size(order.size, meta.sz_decimals)
        if size < min_size - 1e-12:
            raise _Reject(f"tamaño {order.size:g} menor que el mínimo {min_size:g} {coin}")
        order.size = size

        pos = self.positions.get(coin)
        if order.reduce_only:
            if pos is None or (pos.size > 0) == order.is_buy:
                raise _Reject(f"reduce-only: no hay posición contraria en {coin} que reducir")
            if order.size > abs(pos.size):
                order.size = abs(pos.size)
                order.reason = f"recortada a {order.size:g} (tamaño de la posición)"

        if order.kind in TRIGGER_KINDS:          # se evalúan en step() con el mid
            self._orders[order.id] = order
            return

        bid, ask = self._quote(coin)
        taker_px = self._taker_px(order.is_buy, bid, ask)
        if order.kind is OrderKind.MARKET:
            px, immediate = taker_px, True
            if order.price is not None and (px > order.price if order.is_buy else px < order.price):
                raise _Reject(f"MARKET: precio ejecutable {px:g} peor que el límite {order.price:g}")
        else:
            marketable = order.price >= ask if order.is_buy else order.price <= bid
            if marketable:
                if order.tif == "Alo":
                    raise _Reject("Alo: la orden cruzaría el libro (sólo maker)")
                px = min(taker_px, order.price) if order.is_buy else max(taker_px, order.price)
                immediate = True
            else:
                if order.tif == "Ioc":
                    raise _Reject("Ioc: sin contrapartida al precio límite")
                px, immediate = order.price, False

        if not order.reduce_only:
            self._check_risk(order, meta, px, immediate)
        if immediate:
            self._execute(order, px, maker=False)
        else:
            self._orders[order.id] = order

    def _check_risk(self, order: Order, meta: AssetMeta, px: float, immediate: bool) -> None:
        coin = order.coin
        lev = self._lev(coin, meta)
        if lev > meta.max_leverage:
            raise _Reject(f"apalancamiento {lev}x supera el máximo de {coin} ({meta.max_leverage}x)")
        notional = order.size * px
        if notional < meta.min_notional_usd - 1e-9:
            raise _Reject(f"notional {notional:.2f} USD menor que el mínimo {meta.min_notional_usd:g} USD")
        incr = self._increase_part(coin, order.is_buy, order.size)
        if incr <= 0:
            return
        need = incr * px / lev
        avail = self._available()
        pos = self.positions.get(coin)
        if immediate and pos is not None and (pos.size > 0) != order.is_buy:
            # Un giro inmediato libera primero el margen de la posición que cierra.
            avail += pos.margin_used * min(order.size, abs(pos.size)) / abs(pos.size)
        if need > avail + 1e-9:
            raise _Reject(f"margen insuficiente: requiere {need:.2f} USD, disponible {avail:.2f} USD")

    # ------------------------------------------------------------------ ejecución
    def _execute(self, order: Order, px: float, *, maker: bool, liquidation: bool = False) -> Fill:
        size = order.remaining
        fill = self._trade(order.coin, order.is_buy, size, px, maker=maker, order_id=order.id,
                           tag=order.tag, liquidation=liquidation)
        now = self.clock.now()
        prev = order.filled_size
        order.avg_fill_px = px if prev <= 0 else ((order.avg_fill_px or px) * prev + px * size) / (prev + size)
        order.filled_size = prev + size
        order.status, order.updated_ts = OrderStatus.FILLED, now
        self._orders.pop(order.id, None)
        self._closed.append(order)
        self._sweep_reduce_only(order.coin)
        self._dirty = True
        return fill

    def _trade(self, coin: str, is_buy: bool, size: float, px: float, *, maker: bool, order_id: str,
               tag: str, liquidation: bool) -> Fill:
        try:
            meta = self._meta(coin)
        except BrokerError:          # sólo hace falta para redondear tamaños
            meta = None
        dec = meta.sz_decimals if meta else 8
        now = self.clock.now()
        lev = self._lev(coin, meta)
        q = size if is_buy else -size
        pos = self.positions.get(coin)
        realized = 0.0
        if pos is None:                                          # abre
            pos = self.positions[coin] = Position(coin=coin, size=round(q, dec), entry_px=px, leverage=lev,
                                                  opened_ts=now)
        elif (pos.size > 0) == (q > 0):                          # aumenta: precio medio
            new_size = round(pos.size + q, dec)
            pos.entry_px = (abs(pos.size) * pos.entry_px + size * px) / abs(new_size)
            pos.size, pos.leverage = new_size, lev
        else:                                                    # reduce, cierra o gira
            held = abs(pos.size)
            sign = 1.0 if pos.size > 0 else -1.0
            closing = min(size, held)
            realized = closing * (px - pos.entry_px) * sign
            left, extra = round(held - closing, dec), round(size - closing, dec)
            if left > 0:
                pos.size = sign * left
            elif extra > 0:
                pos = self.positions[coin] = Position(coin=coin, size=math.copysign(extra, q), entry_px=px,
                                                      leverage=lev, opened_ts=now)
            else:
                del self.positions[coin]
                pos = None
        if pos is not None:
            pos.margin_used = abs(pos.size) * pos.entry_px / pos.leverage
        fee = size * px * (self.maker_rate if maker else self.taker_rate)
        self.cash += realized - fee
        self.total_fees += fee
        self.total_realized += realized
        self.fill_count += 1
        fill = Fill(order_id=order_id, coin=coin, is_buy=is_buy, size=size, price=px, fee=fee, ts=now,
                    tag=tag, realized_pnl=realized, liquidation=liquidation)
        self._pending.append(fill)
        return fill

    def _cancel_order(self, order: Order, reason: str) -> None:
        order.status, order.reason, order.updated_ts = OrderStatus.CANCELED, reason, self.clock.now()
        self._orders.pop(order.id, None)
        self._closed.append(order)
        self._dirty = True

    def _sweep_reduce_only(self, coin: str) -> None:
        """Cancela las reduce-only que ya no pueden reducir nada (posición plana o girada)."""
        pos = self.positions.get(coin)
        for o in [o for o in self._orders.values() if o.coin == coin and o.reduce_only]:
            if pos is None:
                self._cancel_order(o, "reduce-only cancelada: la posición quedó plana")
            elif (pos.size > 0) == o.is_buy:
                self._cancel_order(o, "reduce-only cancelada: la posición cambió de sentido")

    def _fit_reduce_only(self, order: Order) -> bool:
        """Recorta una reduce-only al tamaño actual de la posición; False si ya no aplica."""
        pos = self.positions.get(order.coin)
        if pos is None or (pos.size > 0) == order.is_buy:
            self._cancel_order(order, "reduce-only cancelada: no hay posición que reducir")
            return False
        if order.remaining > abs(pos.size):
            order.size = order.filled_size + abs(pos.size)
        return True

    @staticmethod
    def _triggered(order: Order, mid: float) -> bool:
        if order.kind is OrderKind.STOP:          # stop de un short = compra; de un long = venta
            return mid >= order.trigger_px if order.is_buy else mid <= order.trigger_px
        return mid <= order.trigger_px if order.is_buy else mid >= order.trigger_px

    def step(self) -> None:
        """Procesa disparos SL/TP, límites en reposo y liquidaciones contra el precio actual.

        Una moneda sin datos se salta en este paso (queda en ``stale_coins``); nunca lanza por ello.
        """
        with self._lock:
            coins = {o.coin for o in self._orders.values()} | set(self.positions)
            for coin in sorted(coins):
                try:
                    bid, ask = self.market.bid_ask(coin)
                except MarketDataError as e:
                    self.stale_coins[coin] = str(e)
                    log.warning("paper step: sin precio de %s (%s)", coin, e)
                    continue
                self.stale_coins.pop(coin, None)
                mid = (bid + ask) / 2.0
                self._marks[coin] = mid
                self._process_coin(coin, bid, ask, mid)
            self._autosave()

    def _process_coin(self, coin: str, bid: float, ask: float, mid: float) -> None:
        # 1) SL/TP: se disparan con el mid y ejecutan a mercado.
        for o in [o for o in self._orders.values() if o.coin == coin and o.kind in TRIGGER_KINDS]:
            if o.id in self._orders and self._triggered(o, mid) and self._fit_reduce_only(o):
                self._execute(o, self._taker_px(o.is_buy, bid, ask), maker=False)
        # 2) Límites en reposo: ejecutan a su precio, como maker.
        for o in [o for o in self._orders.values() if o.coin == coin and o.kind is OrderKind.LIMIT]:
            if o.id not in self._orders:
                continue
            if (ask <= o.price) if o.is_buy else (bid >= o.price):
                if o.reduce_only and not self._fit_reduce_only(o):
                    continue
                self._execute(o, o.price, maker=True)
        # 3) Liquidación por mark.
        pos = self.positions.get(coin)
        if pos is not None:
            liq = self._liq_px(pos)
            if liq is not None and (mid <= liq if pos.size > 0 else mid >= liq):
                self._liquidate(pos, mid, liq)

    def _liquidate(self, pos: Position, mark: float, liq: float) -> None:
        bankrupt = self._bankruptcy_px(pos)
        px = max(mark, bankrupt) if pos.size > 0 else min(mark, bankrupt)
        now = self.clock.now()
        order = Order(id=self._new_id(), coin=pos.coin, is_buy=pos.size < 0, size=abs(pos.size),
                      kind=OrderKind.MARKET, reduce_only=True, created_ts=now, updated_ts=now,
                      tag="liquidation", reason=f"liquidación: mark {mark:g} cruzó {liq:g}")
        log.warning("LIQUIDACIÓN paper %s %+g @ %g (liq %g)", pos.coin, pos.size, px, liq)
        self._execute(order, px, maker=False, liquidation=True)

    def cancel(self, coin: str, order_id: str) -> bool:
        with self._lock:
            order = self._orders.get(order_id)
            if order is None or order.coin != coin:
                return False
            self._cancel_order(order, "cancelada")
            self._autosave()
            return True

    def open_orders(self, coin: Optional[str] = None) -> list[Order]:
        with self._lock:
            return [replace(o) for o in self._orders.values() if coin is None or o.coin == coin]

    def poll_fills(self) -> list[Fill]:
        with self._lock:
            out, self._pending = self._pending, []
            if out:
                self._dirty = True
                self._autosave()
            return out

    # ------------------------------------------------------------------ consultas extra
    def get_order(self, order_id: str) -> Optional[Order]:
        """Orden abierta o reciente (últimas ``history``) por id, como copia."""
        with self._lock:
            if order_id in self._orders:
                return replace(self._orders[order_id])
            for o in reversed(self._closed):
                if o.id == order_id:
                    return replace(o)
            return None

    def stats(self) -> dict[str, float]:
        with self._lock:
            return {"start_equity": self.start_equity, "cash": self.cash, "total_fees": self.total_fees,
                    "total_realized": self.total_realized, "fill_count": self.fill_count}

    # ------------------------------------------------------------------ persistencia
    def _snapshot(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "saved_ts": self.clock.now(),
            "start_equity": self.start_equity,
            "cash": self.cash,
            "total_fees": self.total_fees,
            "total_realized": self.total_realized,
            "fill_count": self.fill_count,
            "next_id": self._next_id,
            "leverage": dict(self.leverage),
            "positions": {c: p.to_dict() for c, p in self.positions.items()},
            "open_orders": [o.to_dict() for o in self._orders.values()],
            "closed_orders": [o.to_dict() for o in self._closed],
            "pending_fills": [f.to_dict() for f in self._pending],
            "marks": dict(self._marks),
        }

    def save_state(self) -> None:
        if self.state_file is None:
            return
        with self._lock:
            path = self.state_file
            tmp = path.with_name(path.name + ".tmp")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._snapshot(), f, ensure_ascii=False, indent=1)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            except OSError as e:
                raise BrokerError(f"no se pudo guardar el estado paper en {path}: {e}") from e
            self._dirty = False

    def _load(self) -> None:
        path = self.state_file
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") != STATE_VERSION:
                raise ValueError(f"versión de estado no soportada: {data.get('version')!r}")
            positions = {c: Position.from_dict(p) for c, p in data["positions"].items()}
            orders = [Order.from_dict(o) for o in data["open_orders"]]
            closed = [Order.from_dict(o) for o in data.get("closed_orders", [])]
            pending = [Fill(**f) for f in data.get("pending_fills", [])]
            leverage = {c: int(v) for c, v in data.get("leverage", {}).items()}
            marks = {c: float(v) for c, v in data.get("marks", {}).items()}
            scalars = {k: float(data[k]) for k in ("start_equity", "cash", "total_fees", "total_realized")}
            next_id, fill_count = int(data["next_id"]), int(data.get("fill_count", 0))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            # Estado ilegible: se aparta (no se pisa) y se empieza de cero, avisando.
            backup = path.with_name(f"{path.name}.corrupt-{int(self.clock.now())}")
            try:
                os.replace(path, backup)
            except OSError:
                backup = path
            self.load_error = f"estado paper ilegible ({e}); copia en {backup.name}; se empieza de cero"
            log.error("%s", self.load_error)
            return
        self.positions, self.leverage, self._marks, self._pending = positions, leverage, marks, pending
        self._orders = {o.id: o for o in orders}
        self._closed = deque(closed, maxlen=self._history)
        self.start_equity, self.cash = scalars["start_equity"], scalars["cash"]
        self.total_fees, self.total_realized = scalars["total_fees"], scalars["total_realized"]
        self._next_id, self.fill_count = next_id, fill_count
        log.info("estado paper recargado: %d posiciones, %d órdenes abiertas, cash %.2f",
                 len(positions), len(orders), self.cash)

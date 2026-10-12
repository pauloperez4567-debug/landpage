"""Broker real en Hyperliquid con el SDK oficial (``hyperliquid-python-sdk``).

- Import perezoso: el SDK y ``eth_account`` sólo se cargan al construir este broker, así el resto de
  la mesa (paper, sim, tests) funciona sin ellos.
- Sólo se construye en modo live con ``settings.validate()`` superado (frase de confirmación, clave y
  datos de Hyperliquid). **Testnet por defecto**; mainnet sólo con ``hl_mainnet=True``.
- La clave privada nunca se guarda en texto ni se escribe en logs ni en mensajes de error: sólo vive
  dentro del objeto ``LocalAccount`` que firma.
- ``HL_ACCOUNT_ADDRESS`` permite operar con una *API wallet* (agente) que firma en nombre de la cuenta
  principal; sin ella se usa la dirección derivada de la clave.

Mapeo de órdenes (precios con ``round_price``, tamaños con ``round_size``):

- LIMIT → ``{"limit": {"tif": "Gtc"|"Alo"|"Ioc"}}`` al precio dado.
- MARKET → IOC límite agresiva a mid ± ``market_slippage_bps`` (o al ``price`` dado = peor aceptable).
- STOP / TAKE_PROFIT → ``{"trigger": {"triggerPx", "isMarket": True, "tpsl": "sl"|"tp"}}``, siempre
  ``reduce_only=True``; el precio límite que exige el exchange se pone a ``TRIGGER_SLIPPAGE`` del disparo
  (tolerancia de deslizamiento del stop a mercado).

Respuestas: ``status != "ok"`` o ``statuses[i].error`` → orden ``REJECTED`` con el motivo; ``resting`` →
``OPEN`` con el ``oid`` como id; ``filled`` → ``FILLED`` (o parcial). Los fallos de transporte (red,
HTTP 4xx/5xx del SDK, rate limit) → ``BrokerError``. Ojo: si el transporte falla *después* de que la
orden llegue al exchange, la orden puede existir; ``ExecutionEngine.reconcile`` la verá en
``open_orders()``.

``poll_fills()`` lee ``user_fills_by_time`` desde la última marca vista (inclusive) y deduplica por
``tid``: cada fill se entrega una sola vez aunque aparezca en varias consultas.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from types import SimpleNamespace
from typing import Any, Optional

import requests

from ..clock import Clock
from ..config import Settings
from ..models import AccountState, AssetMeta, Fill, Order, OrderKind, OrderStatus, Position
from .base import Broker, BrokerError, MarketData, MarketDataError, round_price, round_size

log = logging.getLogger(__name__)

# Mismos valores que ``hyperliquid.utils.constants`` (se usan los del SDK cuando está cargado).
MAINNET_API_URL = "https://api.hyperliquid.xyz"
TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"

VALID_TIFS = ("Gtc", "Alo", "Ioc")
TRIGGER_KINDS = (OrderKind.STOP, OrderKind.TAKE_PROFIT)
TRIGGER_SLIPPAGE = 0.05      # límite de un SL/TP a mercado: 5 % más allá del disparo (como el SDK)
FILLS_PAGE_LIMIT = 2000      # Hyperliquid devuelve como mucho 2000 fills por consulta
MAX_FILL_PAGES = 10
META_TTL = 3600.0
MAX_TAGS = 5000
_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


class _Reject(Exception):
    """Rechazo de negocio: la orden vuelve con ``status=REJECTED`` y este motivo."""


def _load_sdk() -> SimpleNamespace:
    """Importa el SDK de Hyperliquid y eth_account sólo cuando hacen falta."""
    try:
        from eth_account import Account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
    except ImportError as e:
        raise RuntimeError("El modo live necesita el SDK de Hyperliquid: "
                           "pip install hyperliquid-python-sdk eth-account") from e
    return SimpleNamespace(Account=Account, Exchange=Exchange, Info=Info, constants=constants)


def _wallet_from_key(private_key: str) -> Any:
    """``LocalAccount`` a partir de la clave. El error nunca incluye la clave (ni la encadena)."""
    try:
        from eth_account import Account
    except ImportError as e:
        raise RuntimeError("El modo live necesita eth-account: pip install eth-account") from e
    try:
        return Account.from_key(private_key)
    except Exception:  # noqa: BLE001 - cualquier fallo de formato; el mensaje original podría citar la clave
        raise ValueError("HL_PRIVATE_KEY no es una clave privada válida (64 caracteres hex, con o sin 0x)") from None


def _dex_of(coin: str) -> str:
    """Dex HIP-3 de una moneda ("xyz:GOLD" → "xyz"); "" para el dex principal."""
    return coin.split(":", 1)[0] if ":" in coin else ""


def _f(x: Any, default: Optional[float] = None) -> Optional[float]:
    """float tolerante: None / "" → ``default``."""
    if x is None or x == "":
        return default
    return float(x)


def _err_text(e: BaseException) -> str:
    # ClientError / ServerError del SDK guardan el mensaje en atributos, no en str(e).
    for attr in ("error_message", "message"):
        msg = getattr(e, attr, None)
        if msg:
            code = getattr(e, "status_code", None)
            return f"HTTP {code}: {msg}" if code else str(msg)
    return str(e) or type(e).__name__


class LiveHyperliquidBroker(Broker):
    def __init__(self, market: Optional[MarketData], settings: Settings, clock: Clock, *,
                 exchange: Any = None, info: Any = None, address: Optional[str] = None,
                 timeout: float = 10.0, fills_since_ts: Optional[float] = None):
        """``exchange`` / ``info`` inyectables (tests); si faltan se crean con el SDK.

        ``market`` (opcional) aporta metadatos y precio medio con su propio cache; sin él se piden
        a ``info.meta`` / ``info.all_mids``. ``fills_since_ts``: desde cuándo entregar fills
        (por defecto, desde ahora: lo anterior ya lo conoce la mesa o lo adopta ``reconcile``).
        """
        if not settings.is_live:
            raise ValueError("LiveHyperliquidBroker sólo se usa en modo live (DESK_MODE=live)")
        settings.validate()          # frase DESK_LIVE_CONFIRM, HL_PRIVATE_KEY y DESK_DATA=hyperliquid
        self.market = market
        self.settings = settings
        self.clock = clock
        self.mainnet = bool(settings.hl_mainnet)
        self.testnet = not self.mainnet
        self.base_url = MAINNET_API_URL if self.mainnet else TESTNET_API_URL
        self.dexes = [""] + sorted({_dex_of(c) for c in settings.symbols if ":" in c})
        self._lock = threading.RLock()
        self._metas: dict[str, tuple[float, dict[str, AssetMeta]]] = {}
        self._tags: dict[str, str] = {}               # oid -> tag ("entry", "sl", ...) de órdenes propias
        self._opened: dict[str, float] = {}           # moneda -> primera vez que se vio la posición
        self._seen: dict[str, int] = {}               # tid -> time (ms) de fills ya entregados
        self._local_seq = 0
        self.leverage: dict[str, int] = {}
        start = self.clock.now() if fills_since_ts is None else float(fills_since_ts)
        self._fills_cursor_ms = int(start * 1000)

        wallet = None
        if exchange is None or not (address or settings.hl_account_address):
            wallet = _wallet_from_key(settings.hl_private_key)
        addr = address or settings.hl_account_address or wallet.address
        if not _ADDRESS_RE.match(str(addr)):
            raise ValueError(f"HL_ACCOUNT_ADDRESS inválida: {addr!r} (se espera 0x + 40 hex)")
        self.address = str(addr)

        if exchange is None or (info is None and getattr(exchange, "info", None) is None):
            sdk = _load_sdk()
            self.base_url = sdk.constants.MAINNET_API_URL if self.mainnet else sdk.constants.TESTNET_API_URL
            try:
                if exchange is None:
                    agent = self.address if self.address.lower() != wallet.address.lower() else None
                    exchange = sdk.Exchange(wallet, self.base_url, account_address=agent,
                                            perp_dexs=self.dexes if len(self.dexes) > 1 else None,
                                            timeout=timeout)
                if info is None:
                    info = getattr(exchange, "info", None) or sdk.Info(self.base_url, skip_ws=True, timeout=timeout)
            except Exception as e:  # noqa: BLE001 - el SDK hace peticiones de metadatos al construirse
                raise BrokerError(f"no se pudo conectar con Hyperliquid ({self.base_url}): "
                                  f"{type(e).__name__}: {_err_text(e)}") from e
        self.exchange = exchange
        self.info = info if info is not None else exchange.info
        if self.mainnet:
            log.warning("broker LIVE en MAINNET (dinero real), cuenta %s", self.address)
        else:
            log.info("broker live en TESTNET, cuenta %s", self.address)

    @classmethod
    def from_settings(cls, market: Optional[MarketData], settings: Settings, clock: Clock,
                      **kwargs: Any) -> "LiveHyperliquidBroker":
        return cls(market, settings, clock, **kwargs)

    def __repr__(self) -> str:      # nunca la clave
        return f"LiveHyperliquidBroker({'mainnet' if self.mainnet else 'testnet'}, {self.address})"

    @property
    def fills_cursor_ts(self) -> float:
        """Marca (s) desde la que se piden fills; la mesa puede guardarla para el próximo arranque."""
        return self._fills_cursor_ms / 1000.0

    # ------------------------------------------------------------------ llamadas al SDK
    @staticmethod
    def _call(what: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Lectura (Info): cualquier fallo es de transporte → ``BrokerError``."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - red, HTTP, JSON: todo recuperable
            raise BrokerError(f"{what}: {type(e).__name__}: {_err_text(e)}") from e

    @staticmethod
    def _act(what: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Acción firmada (Exchange). Red / HTTP → ``BrokerError``; errores de validación locales
        del SDK (moneda que no conoce, precio con demasiados decimales) → ``_Reject``."""
        try:
            return fn(*args, **kwargs)
        except requests.RequestException as e:
            raise BrokerError(f"{what}: {type(e).__name__}: {e}") from e
        except (KeyError, ValueError, TypeError) as e:
            raise _Reject(f"{what}: el SDK no aceptó la petición ({type(e).__name__}: {e})") from e
        except Exception as e:  # noqa: BLE001 - ClientError / ServerError del SDK y similares
            raise BrokerError(f"{what}: {type(e).__name__}: {_err_text(e)}") from e

    @staticmethod
    def _statuses(resp: Any) -> list:
        """``statuses`` de una respuesta ok; ``_Reject`` con el motivo si no es ok."""
        if not isinstance(resp, dict):
            raise _Reject(f"respuesta inesperada de Hyperliquid: {str(resp)[:200]}")
        if resp.get("status") != "ok":
            raise _Reject(f"Hyperliquid: {str(resp.get('response', resp))[:300]}")
        data = resp.get("response")
        data = data.get("data") if isinstance(data, dict) else None
        statuses = data.get("statuses") if isinstance(data, dict) else None
        if not isinstance(statuses, list) or not statuses:
            raise _Reject(f"respuesta sin 'statuses': {str(resp)[:200]}")
        return statuses

    # ------------------------------------------------------------------ metadatos y precios
    def _meta(self, coin: str) -> Optional[AssetMeta]:
        if self.market is not None:
            try:
                return self.market.meta(coin)
            except KeyError:
                return None
            except MarketDataError as e:
                raise BrokerError(f"sin metadatos de {coin}: {e}") from e
        dex = _dex_of(coin)
        now = self.clock.now()
        cached = self._metas.get(dex)
        stale = cached is None or now - cached[0] >= META_TTL
        if stale or (coin not in cached[1] and now - cached[0] >= 60.0):
            data = self._call("meta", self.info.meta, dex=dex) if dex else self._call("meta", self.info.meta)
            universe = data.get("universe") if isinstance(data, dict) else None
            if not isinstance(universe, list):
                raise BrokerError("meta: falta 'universe' en la respuesta")
            metas: dict[str, AssetMeta] = {}
            for a in universe:
                try:
                    if a.get("isDelisted"):
                        continue
                    name = str(a["name"])
                    if dex and ":" not in name:
                        name = f"{dex}:{name}"
                    metas[name] = AssetMeta(name, sz_decimals=int(a["szDecimals"]),
                                            max_leverage=int(a.get("maxLeverage", 1)))
                except (KeyError, TypeError, ValueError):
                    continue
            self._metas[dex] = (now, metas)
        return self._metas[dex][1].get(coin)

    def _mid(self, coin: str) -> float:
        if self.market is not None:
            try:
                return float(self.market.mid(coin))
            except (MarketDataError, KeyError) as e:
                raise BrokerError(f"sin precio de {coin}: {e}") from e
        dex = _dex_of(coin)
        mids = self._call("all_mids", self.info.all_mids, dex) if dex else self._call("all_mids", self.info.all_mids)
        try:
            px = float(mids[coin])
        except (KeyError, TypeError, ValueError) as e:
            raise BrokerError(f"sin precio medio de {coin} en all_mids") from e
        if not px > 0:
            raise BrokerError(f"precio medio inválido de {coin}: {px}")
        return px

    # ------------------------------------------------------------------ órdenes
    def _local_id(self) -> str:
        self._local_seq += 1
        return f"hl-local-{self._local_seq:06d}"

    def _remember_tag(self, oid: str, tag: str) -> None:
        if not oid or not tag:
            return
        self._tags[oid] = tag
        while len(self._tags) > MAX_TAGS:
            self._tags.pop(next(iter(self._tags)))

    def place_order(self, coin: str, is_buy: bool, size: float, kind: OrderKind, *,
                    price: Optional[float] = None, trigger_px: Optional[float] = None,
                    tif: str = "Gtc", reduce_only: bool = False, tag: str = "") -> Order:
        kind = OrderKind(kind)
        with self._lock:
            now = self.clock.now()
            order = Order(id="", coin=coin, is_buy=bool(is_buy), size=float(size), kind=kind,
                          price=None if price is None else float(price),
                          trigger_px=None if trigger_px is None else float(trigger_px),
                          tif=tif, reduce_only=bool(reduce_only), created_ts=now, updated_ts=now, tag=tag)
            try:
                limit_px, order_type = self._build(order)
                resp = self._act(f"order {coin}", self.exchange.order, coin, order.is_buy, order.size,
                                 limit_px, order_type, reduce_only=order.reduce_only)
                self._apply_response(order, resp)
            except _Reject as r:
                order.status, order.reason = OrderStatus.REJECTED, str(r)
                order.id = order.id or self._local_id()
                log.info("orden rechazada %s %s %s: %s", coin, "compra" if is_buy else "venta", kind.value, r)
            order.updated_ts = self.clock.now()
            self._remember_tag(order.id, tag)
            return order

    def _build(self, order: Order) -> tuple[float, dict]:
        """Valida y redondea la orden; devuelve (precio límite, order_type del SDK)."""
        meta = self._meta(order.coin)
        if meta is None:
            raise _Reject(f"moneda desconocida: {order.coin}")
        szd = meta.sz_decimals
        if not math.isfinite(order.size) or order.size <= 0:
            raise _Reject(f"tamaño inválido: {order.size}")
        size = round_size(order.size, szd)
        if size <= 0:
            raise _Reject(f"tamaño {order.size:g} menor que el mínimo {10.0 ** -szd:g} {order.coin}")
        order.size = size

        if order.kind is OrderKind.LIMIT:
            if order.tif not in VALID_TIFS:
                raise _Reject(f"tif inválido {order.tif!r} (usa Gtc, Alo o Ioc)")
            if order.price is None or not order.price > 0:
                raise _Reject("LIMIT requiere price > 0")
            order.price = round_price(order.price, szd)
            return order.price, {"limit": {"tif": order.tif}}

        if order.kind is OrderKind.MARKET:
            if order.price is not None:
                if not order.price > 0:
                    raise _Reject("MARKET: price (peor precio aceptable) debe ser > 0")
                px = order.price
            else:
                slip = max(self.settings.market_slippage_bps, 0.0) / 10_000.0
                mid = self._mid(order.coin)
                px = mid * (1.0 + slip) if order.is_buy else mid * (1.0 - slip)
            order.price, order.tif = round_price(px, szd), "Ioc"
            return order.price, {"limit": {"tif": "Ioc"}}

        if order.trigger_px is None or not order.trigger_px > 0:
            raise _Reject(f"{order.kind.value} requiere trigger_px > 0")
        if not order.reduce_only:
            raise _Reject("STOP / TAKE_PROFIT deben ser reduce_only=True")
        tpx = round_price(order.trigger_px, szd)
        order.trigger_px, order.price = tpx, None
        limit = round_price(tpx * (1.0 + TRIGGER_SLIPPAGE) if order.is_buy else tpx * (1.0 - TRIGGER_SLIPPAGE), szd)
        tpsl = "sl" if order.kind is OrderKind.STOP else "tp"
        return limit, {"trigger": {"triggerPx": tpx, "isMarket": True, "tpsl": tpsl}}

    def _apply_response(self, order: Order, resp: Any) -> None:
        st = self._statuses(resp)[0]
        if isinstance(st, str):
            # Órdenes agrupadas (TP/SL hijas): "waitingForFill" / "waitingForTrigger", sin oid propio.
            order.id, order.status, order.reason = self._local_id(), OrderStatus.OPEN, st
            return
        if not isinstance(st, dict):
            raise _Reject(f"estado de orden desconocido: {st!r}")
        if "error" in st:
            raise _Reject(str(st["error"]))
        try:
            if "resting" in st:
                order.id, order.status = str(st["resting"]["oid"]), OrderStatus.OPEN
                return
            if "filled" in st:
                f = st["filled"]
                order.id = str(f["oid"])
                order.filled_size = float(f["totalSz"])
                order.avg_fill_px = float(f["avgPx"])
        except (KeyError, TypeError, ValueError) as e:
            raise _Reject(f"respuesta de orden malformada ({e}): {str(st)[:200]}") from e
        if "filled" not in st:
            raise _Reject(f"estado de orden desconocido: {str(st)[:200]}")
        if order.filled_size >= order.size - 1e-12:
            order.status = OrderStatus.FILLED
        elif order.kind is OrderKind.LIMIT and order.tif in ("Gtc", "Alo"):
            order.status = OrderStatus.OPEN       # el resto sigue en el libro
            order.reason = f"ejecutada parcialmente {order.filled_size:g} de {order.size:g}; el resto reposa"
        else:
            order.status = OrderStatus.FILLED     # IOC: el resto lo cancela el exchange
            order.reason = f"IOC ejecutada parcialmente {order.filled_size:g} de {order.size:g}; resto cancelado"

    def cancel(self, coin: str, order_id: str) -> bool:
        try:
            oid = int(str(order_id))
        except ValueError:
            return False                       # id local (rechazada / sin oid): nada que cancelar
        with self._lock:
            try:
                st = self._statuses(self._act(f"cancel {coin}", self.exchange.cancel, coin, oid))[0]
            except _Reject as r:
                log.info("cancelación de %s %s rechazada: %s", coin, oid, r)
                return False
            if st == "success":
                return True
            log.info("cancelación de %s %s no aplicada: %s", coin, oid, st)
            return False

    def cancel_all(self, coin: Optional[str] = None) -> int:
        with self._lock:
            orders = [o for o in self.open_orders(coin) if o.id.isdigit()]
            bulk = getattr(self.exchange, "bulk_cancel", None)
            if not orders or bulk is None:
                return super().cancel_all(coin) if orders else 0
            try:
                statuses = self._statuses(self._act("bulk_cancel", bulk,
                                                    [{"coin": o.coin, "oid": int(o.id)} for o in orders]))
            except _Reject as r:
                log.info("cancelación masiva rechazada (%s); se cancela una a una", r)
                return super().cancel_all(coin)
            return sum(1 for s in statuses if s == "success")

    def set_leverage(self, coin: str, leverage: int, cross: bool = False) -> None:
        """Fija el apalancamiento (aislado por defecto). ``ValueError`` si el exchange lo rechaza."""
        with self._lock:
            meta = self._meta(coin)
            if meta is None:
                raise ValueError(f"moneda desconocida: {coin}")
            if isinstance(leverage, bool) or int(leverage) != leverage:
                raise ValueError(f"apalancamiento inválido: {leverage!r} (debe ser entero)")
            lev = int(leverage)
            if not 1 <= lev <= meta.max_leverage:
                raise ValueError(f"apalancamiento {lev}x fuera de rango para {coin} (1..{meta.max_leverage}x)")
            try:
                resp = self._act(f"update_leverage {coin}", self.exchange.update_leverage, lev, coin,
                                 is_cross=bool(cross))
            except _Reject as r:
                raise ValueError(str(r)) from None
            if not isinstance(resp, dict) or resp.get("status") != "ok":
                detail = resp.get("response", resp) if isinstance(resp, dict) else resp
                raise ValueError(f"Hyperliquid rechazó {lev}x en {coin}: {str(detail)[:300]}")
            self.leverage[coin] = lev

    # ------------------------------------------------------------------ cuenta
    def account(self) -> AccountState:
        with self._lock:
            equity = available = 0.0
            positions: dict[str, Position] = {}
            for dex in self.dexes:
                st = (self._call("user_state", self.info.user_state, self.address, dex) if dex
                      else self._call("user_state", self.info.user_state, self.address))
                eq, av, pos = self._parse_user_state(st, dex)
                equity += eq
                available += av
                positions.update(pos)
            now = self.clock.now()
            for coin in list(self._opened):
                if coin not in positions:
                    del self._opened[coin]
            for coin, p in positions.items():
                p.opened_ts = self._opened.setdefault(coin, now)
            return AccountState(equity=equity, available=max(available, 0.0), positions=positions,
                                open_orders=self.open_orders(), ts=now)

    @staticmethod
    def _parse_user_state(st: Any, dex: str) -> tuple[float, float, dict[str, Position]]:
        where = f"user_state{'/' + dex if dex else ''}"
        if not isinstance(st, dict):
            raise BrokerError(f"{where}: respuesta inesperada")
        try:
            equity = float(st["marginSummary"]["accountValue"])
            available = _f(st.get("withdrawable"), 0.0)
            positions: dict[str, Position] = {}
            for ap in st.get("assetPositions") or []:
                p = ap["position"]
                szi = float(p["szi"])
                if szi == 0:
                    continue
                coin = str(p["coin"])
                lev = p.get("leverage") or {}
                value = _f(p.get("positionValue"), 0.0)
                positions[coin] = Position(
                    coin=coin, size=szi, entry_px=_f(p.get("entryPx"), 0.0),
                    leverage=int(float(lev.get("value", 1))) if isinstance(lev, dict) else 1,
                    unrealized_pnl=_f(p.get("unrealizedPnl"), 0.0),
                    mark_px=value / abs(szi) if value > 0 else None,
                    liq_px=_f(p.get("liquidationPx")),
                    margin_used=_f(p.get("marginUsed"), 0.0))
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise BrokerError(f"{where}: respuesta malformada ({type(e).__name__}: {e})") from e
        return equity, available, positions

    def open_orders(self, coin: Optional[str] = None) -> list[Order]:
        with self._lock:
            dexes = [_dex_of(coin)] if coin is not None else self.dexes
            out: list[Order] = []
            for dex in dexes:
                rows = (self._call("frontend_open_orders", self.info.frontend_open_orders, self.address, dex) if dex
                        else self._call("frontend_open_orders", self.info.frontend_open_orders, self.address))
                if not isinstance(rows, list):
                    raise BrokerError("frontend_open_orders: se esperaba una lista")
                for r in rows:
                    try:
                        o = self._parse_order(r)
                    except (KeyError, TypeError, ValueError, AttributeError) as e:
                        log.warning("orden abierta ilegible (%s): %s", e, str(r)[:200])
                        continue
                    if coin is None or o.coin == coin:
                        out.append(o)
            return out

    def _parse_order(self, r: dict) -> Order:
        oid = str(r["oid"])
        remaining = _f(r.get("sz"), 0.0)
        orig = _f(r.get("origSz"), remaining)
        otype = str(r.get("orderType") or "Limit").lower()
        is_trigger = bool(r.get("isTrigger"))
        if "take profit" in otype:
            kind = OrderKind.TAKE_PROFIT
        elif is_trigger or "stop" in otype:
            kind = OrderKind.STOP
        else:
            kind = OrderKind.LIMIT
        trigger = _f(r.get("triggerPx")) if kind in TRIGGER_KINDS else None
        ts = _f(r.get("timestamp"), 0.0) / 1000.0
        tag = self._tags.get(oid) or {OrderKind.STOP: "sl", OrderKind.TAKE_PROFIT: "tp"}.get(kind, "")
        return Order(id=oid, coin=str(r["coin"]), is_buy=r.get("side") == "B", size=orig, kind=kind,
                     price=None if kind in TRIGGER_KINDS else _f(r.get("limitPx")),
                     trigger_px=trigger if trigger else None, tif=str(r.get("tif") or "Gtc"),
                     reduce_only=bool(r.get("reduceOnly")), status=OrderStatus.OPEN,
                     filled_size=max(orig - remaining, 0.0), created_ts=ts, updated_ts=ts, tag=tag,
                     reason="TP/SL de la posición completa" if r.get("isPositionTpsl") else "")

    def poll_fills(self) -> list[Fill]:
        with self._lock:
            out: list[Fill] = []
            start = self._fills_cursor_ms
            for _ in range(MAX_FILL_PAGES):
                rows = self._call("user_fills_by_time", self.info.user_fills_by_time, self.address, start)
                if not isinstance(rows, list):
                    raise BrokerError("user_fills_by_time: se esperaba una lista")
                rows = sorted((r for r in rows if isinstance(r, dict)), key=lambda r: int(r.get("time") or 0))
                newest = start
                for r in rows:
                    t = int(r.get("time") or 0)
                    newest = max(newest, t)
                    key = str(r["tid"]) if r.get("tid") is not None else \
                        f"{r.get('hash')}:{r.get('oid')}:{t}:{r.get('sz')}:{r.get('px')}"
                    if key in self._seen:
                        continue
                    self._seen[key] = t
                    try:
                        out.append(self._parse_fill(r))
                    except (KeyError, TypeError, ValueError) as e:
                        log.warning("fill ilegible (%s): %s", e, str(r)[:200])
                more = len(rows) >= FILLS_PAGE_LIMIT and newest > start
                start = newest
                if not more:
                    break
            self._fills_cursor_ms = start
            # La consulta incluye el milisegundo del cursor: sólo hace falta recordar esos tids.
            self._seen = {k: t for k, t in self._seen.items() if t >= start}
            return out

    def _parse_fill(self, r: dict) -> Fill:
        oid = str(r["oid"])
        fee = _f(r.get("fee"), 0.0) + _f(r.get("builderFee"), 0.0)
        return Fill(order_id=oid, coin=str(r["coin"]), is_buy=r.get("side") == "B", size=float(r["sz"]),
                    price=float(r["px"]), fee=fee, ts=int(r["time"]) / 1000.0, tag=self._tags.get(oid, ""),
                    realized_pnl=_f(r.get("closedPnl"), 0.0), liquidation=bool(r.get("liquidation")))

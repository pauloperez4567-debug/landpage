"""Tipos compartidos por todos los módulos de la mesa.

Convenciones (todas las piezas las respetan):
- Precios y tamaños son ``float``. El tamaño de una posición es *con signo*:
  positivo = long, negativo = short. El tamaño de una orden es siempre positivo
  y la dirección la da ``is_buy``.
- Las marcas de tiempo son segundos UNIX (``float``) en UTC, obtenidas del
  ``Clock`` inyectado (nunca de ``time.time()`` directamente) para que la
  simulación y los tests sean deterministas.
- Dimensionamiento estilo "unidad": ``notional = units * unit_margin_usd * leverage``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def is_buy(self) -> bool:
        return self is Side.LONG

    @property
    def opposite(self) -> "Side":
        return Side.SHORT if self is Side.LONG else Side.LONG


class OrderKind(str, Enum):
    LIMIT = "limit"          # orden límite (GTC / ALO / IOC según tif)
    MARKET = "market"        # IOC agresiva contra el libro
    STOP = "stop"            # stop-loss: se dispara cuando el precio cruza trigger_px
    TAKE_PROFIT = "take_profit"  # take-profit: se dispara cuando el precio cruza trigger_px


class OrderStatus(str, Enum):
    OPEN = "open"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class EntryMode(str, Enum):
    ZONE = "zone"        # límite en el centro de la zona de liquidez más fuerte
    MARKET = "market"    # entrada inmediata (IOC)
    LIMIT = "limit"      # límite a un precio dado por el usuario


@dataclass
class AssetMeta:
    coin: str
    sz_decimals: int
    max_leverage: int
    min_notional_usd: float = 10.0


@dataclass
class Order:
    id: str
    coin: str
    is_buy: bool
    size: float
    kind: OrderKind
    price: Optional[float] = None        # límite (LIMIT/MARKET) o None
    trigger_px: Optional[float] = None   # sólo STOP / TAKE_PROFIT
    tif: str = "Gtc"                     # "Gtc" | "Ioc" | "Alo"
    reduce_only: bool = False
    status: OrderStatus = OrderStatus.OPEN
    filled_size: float = 0.0
    avg_fill_px: Optional[float] = None
    created_ts: float = 0.0
    updated_ts: float = 0.0
    tag: str = ""                        # "entry" | "sl" | "tp" | "close" | ...
    reason: str = ""                     # texto libre (rechazos, etc.)

    @property
    def remaining(self) -> float:
        return max(self.size - self.filled_size, 0.0)

    @property
    def is_open(self) -> bool:
        return self.status is OrderStatus.OPEN

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Order":
        d = dict(d)
        d["kind"] = OrderKind(d["kind"])
        d["status"] = OrderStatus(d["status"])
        return cls(**d)


@dataclass
class Fill:
    order_id: str
    coin: str
    is_buy: bool
    size: float
    price: float
    fee: float
    ts: float
    tag: str = ""
    realized_pnl: float = 0.0   # PnL realizado por esta ejecución (sin comisiones)
    liquidation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Position:
    coin: str
    size: float                 # con signo: + long, - short
    entry_px: float
    leverage: int = 1
    unrealized_pnl: float = 0.0
    mark_px: Optional[float] = None
    liq_px: Optional[float] = None
    margin_used: float = 0.0
    opened_ts: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def side(self) -> Side:
        return Side.LONG if self.size > 0 else Side.SHORT

    @property
    def notional(self) -> float:
        px = self.mark_px if self.mark_px is not None else self.entry_px
        return abs(self.size) * px

    def pnl_pct(self) -> float:
        """PnL no realizado sobre el margen (equivale al ROE de Hyperliquid), en %."""
        if self.margin_used <= 0:
            return 0.0
        return self.unrealized_pnl / self.margin_used * 100.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Position":
        return cls(**d)


@dataclass
class AccountState:
    equity: float                       # efectivo + PnL no realizado
    available: float                    # equity - margen usado
    positions: dict[str, Position] = field(default_factory=dict)
    open_orders: list[Order] = field(default_factory=list)
    ts: float = 0.0

    @property
    def gross_notional(self) -> float:
        return sum(p.notional for p in self.positions.values())


@dataclass
class Zone:
    """Zona de liquidez donde la mesa quiere entrar.

    ``side`` es el lado de la *entrada* que la zona sirve: una zona LONG está por
    debajo del precio (liquidez compradora / liquidaciones de longs), una zona
    SHORT por encima.
    """

    coin: str
    side: Side
    low: float
    high: float
    usd: float                  # dólares de liquidez acumulados en la zona
    source: str                 # "book" | "liq" | "mixed"
    distance_pct: float = 0.0   # distancia del centro de la zona al precio actual, en %

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0

    def contains(self, px: float) -> bool:
        return self.low <= px <= self.high


class CommandAction(str, Enum):
    OPEN = "open"            # abrir posición (side, coin, units, entry_mode, ...)
    CLOSE = "close"          # cerrar posición de una moneda
    CLOSE_ALL = "close_all"  # cerrar todas las posiciones
    CANCEL = "cancel"        # cancelar órdenes abiertas (de una moneda o todas)
    SET_SL = "set_sl"        # mover / poner stop-loss a una posición
    SET_TP = "set_tp"        # mover / poner take-profit a una posición
    PAUSE = "pause"          # dejar de abrir operaciones nuevas (gestiona las abiertas)
    RESUME = "resume"        # volver a operar (no levanta un halt de riesgo)
    AUTO_ON = "auto_on"      # activar estrategias autónomas
    AUTO_OFF = "auto_off"    # desactivar estrategias autónomas
    STATUS = "status"        # informe de cuenta, posiciones y órdenes
    SET_PARAM = "set_param"  # cambiar un parámetro (param, value)
    KILL = "kill"            # pánico: cerrar todo, cancelar todo y detener
    RESET_HALT = "reset_halt"  # levantar manualmente un halt de riesgo
    CONFIRM = "confirm"      # confirmar un comando pendiente (interpretado por IA)
    REJECT = "reject"        # descartar un comando pendiente
    HELP = "help"


@dataclass
class Command:
    action: CommandAction
    raw: str = ""
    source: str = "regex"                # "regex" | "llm"
    coin: Optional[str] = None
    side: Optional[Side] = None
    units: Optional[float] = None
    entry_mode: Optional[EntryMode] = None
    price: Optional[float] = None        # precio límite (LIMIT) o precio de SL/TP en SET_SL/SET_TP
    sl_price: Optional[float] = None
    sl_pct: Optional[float] = None       # distancia del SL en % desde la entrada
    tp_price: Optional[float] = None
    tp_pct: Optional[float] = None       # distancia del TP en % desde la entrada
    param: Optional[str] = None          # SET_PARAM: nombre canónico del parámetro
    value: Optional[float] = None        # SET_PARAM: valor
    id: str = ""                         # asignado por la mesa al recibirlo

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        d["side"] = self.side.value if self.side else None
        d["entry_mode"] = self.entry_mode.value if self.entry_mode else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Command":
        d = dict(d)
        d["action"] = CommandAction(d["action"])
        d["side"] = Side(d["side"]) if d.get("side") else None
        d["entry_mode"] = EntryMode(d["entry_mode"]) if d.get("entry_mode") else None
        return cls(**d)


@dataclass
class Signal:
    """Propuesta de una estrategia autónoma (todavía sin pasar por riesgo)."""

    coin: str
    side: Side
    entry_mode: EntryMode
    strategy: str
    reason: str
    entry_px: Optional[float] = None     # precio límite propuesto (ZONE/LIMIT) o referencia (MARKET)
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    confidence: float = 0.5              # 0..1, informativo


@dataclass
class TradeIntent:
    """Intención de abrir una posición, venga de un comando o de una estrategia."""

    coin: str
    side: Side
    units: float
    entry_mode: EntryMode
    origin: str                          # "command" | "auto:<estrategia>"
    reason: str = ""
    limit_px: Optional[float] = None
    sl_price: Optional[float] = None
    sl_pct: Optional[float] = None
    tp_price: Optional[float] = None
    tp_pct: Optional[float] = None
    zone: Optional[Zone] = None


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    leverage: int = 0
    size: float = 0.0                    # tamaño en monedas, ya redondeado a sz_decimals
    notional: float = 0.0
    entry_px: float = 0.0                # precio usado para dimensionar
    sl_price: Optional[float] = None     # SL final (siempre presente si approved y require_stop)
    tp_price: Optional[float] = None
    warnings: list[str] = field(default_factory=list)

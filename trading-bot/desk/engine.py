"""Ejecución: entradas, brackets SL/TP (OCO), caducidad, reconciliación y cierres.

Cada operación gestionada es un ``Bracket`` (uno por moneda) con este ciclo de vida::

    open()        on_fills() (entrada)           on_fills() (SL/TP/cierre) o reconcile()
  ─────────► PENDING ───────────────► OPEN ──────────────────────────────────► cerrada (ClosedTrade)
               │  (caduca / se cancela)  │  close() ─► CLOSING ─► cerrada
               └──► descartada           └─ SL/TP reduce-only por el tamaño de la posición

- **Entrada**: ZONE/LIMIT → LIMIT Gtc a ``decision.entry_px``; MARKET → MARKET con peor precio aceptable
  ``entry_px ± market_slippage_bps`` (protege de saltos entre la evaluación y el envío). El bracket se
  registra (y se guarda) *antes* de enviar: si el envío falla por transporte (timeout con la orden ya en el
  exchange) la operación no se pierde; ``reconcile`` encuentra la orden o la posición y la protege con el
  SL/TP aprobados, o la descarta si no aparece.
- **Brackets**: al ejecutarse la entrada (total o parcial) se colocan STOP y TAKE_PROFIT reduce-only por
  el tamaño abierto; si la posición crece (más fills parciales) se coloca la nueva orden y *después* se
  cancela la vieja (nunca queda la posición sin stop). Si salta el SL se retira el TP y viceversa (OCO).
- **reconcile(account)**: caducidad de entradas (``entry_ttl_minutes``), entradas que desaparecen del
  exchange, posiciones cerradas fuera de la mesa, SL/TP que faltan (se recolocan), cierres que no se
  completaron (se reintentan), órdenes reduce-only huérfanas (se cancelan) y posiciones sin bracket (se
  adoptan con SL por defecto: p. ej. tras un reinicio sin estado o una orden manual en el exchange).
  Cada moneda se reconcilia por separado: un fallo de una no deja sin protección a las demás.
- Una entrada pendiente sólo se descarta cuando su cancelación se confirma (o la orden ya no existe): si la
  cancelación falla y la entrada se ejecuta, la posición conserva el SL/TP planificados.
- Si el SL de una posición se rechaza ``MAX_SL_FAILURES`` veces seguidas (o falla por conexión
  ``MAX_SL_ERRORS`` veces), la posición se cierra: la mesa nunca mantiene una posición sin stop.
- Mover un SL (``set_stop``) nunca lo deja más allá de la liquidación ni ensancha la pérdida por encima de
  ``risk_per_trade_pct`` del equity.
- Cierres a mercado: IOC con un límite ``CLOSE_SLIPPAGE`` peor que la referencia (mid o mark de la posición;
  la tolerancia crece en cada reintento). Lo importante es salir: no se usa el margen estrecho de las entradas.
- ``on_fills`` es idempotente ante la re-entrega del último instante ya aplicado (el cursor de fills de live es
  inclusivo y un reinicio puede volver a entregarlo): las claves de esos fills se guardan con el estado.

El estado de los brackets se guarda con ``Journal.save_json("engine")`` y se recarga al construir.
Los mensajes legibles para el operador se acumulan en ``messages`` (``pop_messages()``).
Todas las horas salen del ``Clock`` inyectado.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

from .clock import Clock
from .config import Settings
from .exchange.base import Broker, BrokerError, MarketData, MarketDataError, round_price, round_size
from .journal import Journal
from .models import (AccountState, AssetMeta, EntryMode, Fill, Order, OrderKind, OrderStatus, Position,
                     RiskDecision, Side, TradeIntent)
from .risk import LIQ_BUFFER, estimated_liq_px, maintenance_rate

log = logging.getLogger(__name__)

STATE_NAME = "engine"
STATE_VERSION = 1

PENDING = "pending"     # entrada enviada, sin ejecutar
OPEN = "open"           # posición abierta (total o parcialmente) con SL/TP
CLOSING = "closing"     # cierre a mercado enviado

MAX_SL_FAILURES = 3     # SL rechazado N veces seguidas -> cierre de seguridad
MAX_SL_ERRORS = 10      # SL sin poder colocarse por fallos de conexión N veces seguidas -> cierre de seguridad
MAX_TP_FAILURES = 5     # TP rechazado N veces seguidas -> se deja de reintentar (la posición sigue con su SL)

# Niveles de los mensajes del motor (los de los avisos: ``notify``/``alerts``).
INFO, WARN, CRITICAL = "info", "warn", "critical"


class EngineMessage(str):
    """Mensaje del motor para el operador: un ``str`` normal con su nivel de aviso (``level``: "info" | "warn" |
    "critical") y una clave anti-spam opcional (``key``, por moneda y tipo). Los críticos (posición sin SL, cierre
    de seguridad, liquidación, ejecución fuera de una operación, cierre que no sale) no se silencian y suenan."""

    level: str
    key: Optional[str]

    def __new__(cls, text: str, level: str = INFO, key: Optional[str] = None) -> "EngineMessage":
        obj = super().__new__(cls, text)
        obj.level = level
        obj.key = key
        return obj
PLACE_GRACE_S = 5.0     # una orden recién colocada puede tardar en verse en open_orders (live)
MISSING_CHECKS = 2      # reconciliaciones seguidas antes de dar por perdida una entrada / posición
CLOSE_SLIPPAGE = 0.05   # cierre a mercado: límite 5 % peor que la referencia (como market_close del SDK)
CLOSE_MAX_SLIPPAGE = 0.15
CLOSE_RETRY_S = 10.0    # espera antes de reintentar un cierre que dejó posición abierta
MAX_CLOSE_ATTEMPTS = 6  # cierre inicial + 5 reintentos automáticos; después sólo se avisa
FILL_CLOCK_SKEW_S = 5.0  # un fill anterior a la creación del bracket (menos este margen) no es suyo

# Motivo de salida -> texto para el operador.
EXIT_LABELS = {
    "sl": "stop-loss",
    "tp": "take-profit",
    "close": "cierre manual",
    "close_all": "cierre de todo",
    "flatten": "cierre por límite de riesgo",
    "kill": "pánico (kill)",
    "no_sl": "cierre de seguridad (sin SL)",
    "liquidation": "LIQUIDACIÓN",
    "external": "cierre fuera de la mesa",
}


# --------------------------------------------------------------------------- formato
def fmt_px(px: Optional[float]) -> str:
    """Precio legible: hasta 6 cifras significativas (85.249, 83766, 0.0012345)."""
    if px is None or not isinstance(px, (int, float)) or not math.isfinite(px):
        return "-"
    return f"{px:.6g}"


def fmt_usd(x: float, sign: bool = True) -> str:
    """Importe en USD: ``+$1,234.56`` / ``-$5.00`` (o ``$1,234.56`` sin signo)."""
    if x is None or not math.isfinite(x):
        return "-"
    s = f"${abs(x):,.2f}"
    if not sign:
        return s if x >= 0 else "-" + s
    return ("+" if x >= 0 else "-") + s


def fmt_size(x: float) -> str:
    return f"{x:.8g}"


# --------------------------------------------------------------------------- modelos
@dataclass
class Bracket:
    coin: str
    side: Side
    origin: str                          # "command" | "auto:<estrategia>@<temporalidad>" | "adopted"
    status: str = PENDING
    entry_mode: str = ""
    entry_order_id: str = ""
    entry_px: float = 0.0                # precio planificado de entrada
    planned_size: float = 0.0
    leverage: int = 1
    sl: Optional[float] = None
    tp: Optional[float] = None
    expires_ts: Optional[float] = None   # caducidad de la entrada (sólo límites)
    created_ts: float = 0.0
    opened_ts: float = 0.0
    position: float = 0.0                # tamaño abierto (sin signo) según fills / cuenta
    filled: float = 0.0                  # entrada ejecutada acumulada
    entry_cost: float = 0.0              # suma tamaño·precio de la entrada (precio medio)
    exit_size: float = 0.0
    exit_value: float = 0.0
    realized: float = 0.0                # PnL realizado bruto (sin comisiones)
    fees: float = 0.0
    sl_order_id: str = ""
    sl_size: float = 0.0                 # tamaño que cubre la orden SL actual
    sl_placed_ts: float = 0.0
    tp_order_id: str = ""
    tp_size: float = 0.0
    tp_placed_ts: float = 0.0
    close_order_ids: list[str] = field(default_factory=list)
    exit_reason: str = ""
    sl_failures: int = 0
    tp_failures: int = 0
    sl_errors: int = 0                   # intentos seguidos de colocar el SL fallidos por conexión
    flat_checks: int = 0                 # reconciliaciones seguidas sin posición en la cuenta
    missing_checks: int = 0              # reconciliaciones seguidas sin la orden de entrada
    canceling: bool = False              # se pidió cancelar la entrada pero no se confirmó
    assumed: float = 0.0                 # entrada dada por ejecutada según la cuenta, sin su fill todavía
    close_attempts: int = 0              # cierres a mercado enviados (incluidos reintentos)
    close_ts: float = 0.0                # último intento de cierre
    close_warned: bool = False           # ya se avisó de que el cierre no se completa
    reason: str = ""                     # por qué se abrió (comando / señal)
    zone: Optional[dict] = None
    cmd_id: str = ""

    @property
    def avg_entry(self) -> float:
        return self.entry_cost / self.filled if self.filled > 0 else self.entry_px

    @property
    def avg_exit(self) -> Optional[float]:
        return self.exit_value / self.exit_size if self.exit_size > 0 else None

    @property
    def is_auto(self) -> bool:
        return self.origin.startswith("auto")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Bracket":
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in d.items() if k in known}
        data["side"] = Side(data["side"])
        data["close_order_ids"] = [str(x) for x in data.get("close_order_ids") or []]
        return cls(**data)


@dataclass
class ClosedTrade:
    """Operación cerrada: lo que la mesa pasa a ``RiskManager.record_trade_result(pnl)``."""

    coin: str
    side: Side
    origin: str
    size: float
    entry_px: float
    exit_px: Optional[float]
    pnl: float                           # neto: realizado - comisiones (entrada + salida)
    gross_pnl: float
    fees: float
    exit_reason: str
    opened_ts: float
    closed_ts: float
    reason: str = ""
    cmd_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value
        d["duration_min"] = round(max(self.closed_ts - self.opened_ts, 0.0) / 60.0, 1) if self.opened_ts else None
        return d


# --------------------------------------------------------------------------- motor
class ExecutionEngine:
    def __init__(self, broker: Broker, market: MarketData, settings: Settings, clock: Clock,
                 journal: Optional[Journal] = None, *, load: bool = True):
        self.broker = broker
        self.market = market
        self.settings = settings
        self.clock = clock
        self.journal = journal
        self.brackets: dict[str, Bracket] = {}
        self.messages: list[str] = []
        self._protect_failed: set[str] = set()            # "sl"/"tp" que falló en el último _protect
        self._retry_cancel: list[tuple[str, str]] = []     # (moneda, id) de cancelaciones fallidas
        self._metas: dict[str, AssetMeta] = {}
        self._reject_seq = 0
        self._saved: Optional[dict] = None
        self.load_error = ""
        self.close_failures: dict[str, str] = {}          # monedas que el último close_all() no pudo cerrar
        # Fills aplicados en el último instante (ts) visto: {clave: veces}. Una re-entrega se descarta.
        self._last_fill_ts = 0.0
        self._last_fill_keys: dict[str, int] = {}
        if journal is not None and load:
            self.load_state()

    # ------------------------------------------------------------------ utilidades
    def pop_messages(self) -> list[str]:
        out, self.messages = self.messages, []
        return out

    def _say(self, text: str, level: str = INFO, key: Optional[str] = None) -> None:
        self.messages.append(EngineMessage(text, level, key))   # la mesa los muestra y los avisa con su nivel
        log.debug("%s", text)

    def _say_protect(self, coin: str, done: list[str], suffix: str = "") -> None:
        """Lo que hizo ``_protect``: crítico si el SL no quedó colocado (posición sin stop), aviso si fue el TP;
        con clave por moneda y tipo (un fallo que se repite cada ciclo no inunda los canales)."""
        if not done:
            return
        if "sl" in self._protect_failed:
            level, key = CRITICAL, f"protect:{coin}:sl"
        elif "tp_off" in self._protect_failed:          # aviso final: no lo tapa el anti-spam de los rechazos
            level, key = WARN, f"protect:{coin}:tp_off"
        elif "tp" in self._protect_failed:
            level, key = WARN, f"protect:{coin}:tp"
        else:
            level, key = INFO, None
        self._say(f"{coin}: " + ", ".join(done) + suffix, level, key)

    def _event(self, event_kind: str, /, **data: Any) -> None:
        if self.journal is not None:
            self.journal.event(event_kind, **data)

    def _meta(self, coin: str) -> AssetMeta:
        meta = self._metas.get(coin)
        if meta is None:
            meta = self._metas[coin] = self.market.meta(coin)
        return meta

    def _lot(self, coin: str) -> float:
        try:
            return 10.0 ** -self._meta(coin).sz_decimals
        except (KeyError, MarketDataError):
            return 1e-8

    def _round_px(self, coin: str, px: float) -> float:
        try:
            return round_price(px, self._meta(coin).sz_decimals)
        except (KeyError, MarketDataError):
            return px

    def _mid(self, coin: str) -> Optional[float]:
        try:
            return self.market.mid(coin)
        except (MarketDataError, KeyError):
            return None

    def _liq_px(self, coin: str, side: Side, entry: float, leverage: int, known: Optional[float] = None
                ) -> Optional[float]:
        """Liquidación de la posición: la del exchange si se conoce; si no, la estimada como en riesgo."""
        if known is not None and known > 0:
            return float(known)
        if not entry or entry <= 0:
            return None
        lev = max(int(leverage or 1), 1)
        try:
            max_lev = self._meta(coin).max_leverage
        except (KeyError, MarketDataError):
            max_lev = lev
        return estimated_liq_px(entry, lev, side is Side.LONG, maintenance_rate(self.settings, max_lev))

    @staticmethod
    def _liq_limit(side: Side, entry: float, liq: Optional[float]) -> Optional[float]:
        """SL más lejano que aún deja la liquidación ``LIQ_BUFFER`` veces más lejos que el stop."""
        if liq is None or not entry:
            return None
        if side is Side.LONG:
            return entry - (entry - liq) / LIQ_BUFFER
        return entry + (liq - entry) / LIQ_BUFFER

    def _check_stop(self, coin: str, side: Side, entry: float, size: float, leverage: int, sl: float,
                    old_sl: Optional[float], equity: Optional[float], liq: Optional[float] = None) -> None:
        """``ValueError`` si el SL ``sl`` queda más allá de la liquidación o, siendo más ancho que el actual,
        pondría en riesgo más de ``risk_per_trade_pct`` del equity (mismas reglas que al abrir)."""
        long = side is Side.LONG
        liq = self._liq_px(coin, side, entry, leverage, liq)
        limit = self._liq_limit(side, entry, liq)
        if limit is not None and (sl < limit if long else sl > limit):
            raise ValueError(f"el SL {fmt_px(sl)} queda más allá de la liquidación (~{fmt_px(liq)} a {leverage}x): la "
                             f"posición se liquidaría antes de que salte el stop. Como mucho hasta "
                             f"{fmt_px(self._round_px(coin, limit))}")
        if size <= 0 or equity is None or not math.isfinite(equity) or equity <= 0:
            return
        taker = max(self.settings.paper_taker_fee_bps, 0.0) / 10_000.0

        def loss(px: float) -> float:
            move = (entry - px) if long else (px - entry)
            return size * (max(move, 0.0) + taker * (entry + px))

        max_loss = equity * self.settings.risk_per_trade_pct / 100.0
        new = loss(sl)
        old = loss(old_sl) if old_sl else math.inf
        if new > max_loss + 1e-9 and new > old + 1e-9:
            per = max_loss / size
            edge = (entry * (1 + taker) - per) / (1 - taker) if long else (per + entry * (1 - taker)) / (1 + taker)
            raise ValueError(f"con el SL en {fmt_px(sl)} se perderían {fmt_usd(new, sign=False)} "
                             f"({new / equity * 100:.2f}% del equity), más que el máximo por operación "
                             f"({self.settings.risk_per_trade_pct:g}% = {fmt_usd(max_loss, sign=False)}). Como mucho "
                             f"hasta {fmt_px(self._round_px(coin, edge))} con este tamaño, o reduce la posición")

    def _local_reject(self, coin: str, is_buy: bool, size: float, kind: OrderKind, reason: str,
                      price: Optional[float] = None) -> Order:
        self._reject_seq += 1
        now = self.clock.now()
        return Order(id=f"engine-reject-{self._reject_seq}", coin=coin, is_buy=is_buy, size=size, kind=kind,
                     price=price, status=OrderStatus.REJECTED, created_ts=now, updated_ts=now, tag="entry",
                     reason=reason)

    def _cancel_quiet(self, coin: str, oid: str) -> bool:
        """Cancela sin propagar fallos de transporte (se reintenta en ``reconcile``)."""
        if not oid:
            return False
        try:
            return self.broker.cancel(coin, oid)
        except BrokerError as e:
            log.warning("no se pudo cancelar %s %s (%s); se reintentará", coin, oid, e)
            if (coin, oid) not in self._retry_cancel:
                self._retry_cancel.append((coin, oid))
            return False

    @staticmethod
    def _positions(account: AccountState) -> dict[str, Position]:
        return {c: p for c, p in account.positions.items() if p.size != 0}

    def active(self, coin: str) -> Optional[Bracket]:
        return self.brackets.get(coin)

    # ------------------------------------------------------------------ entrada
    def open(self, intent: TradeIntent, decision: RiskDecision, *, cmd_id: str = "") -> Order:
        """Fija el apalancamiento, registra el bracket pendiente y envía la entrada.

        Devuelve la orden de entrada (``REJECTED`` con motivo si no se pudo enviar). Lanza ``BrokerError``
        sólo en fallos de transporte; en ese caso el bracket se conserva *sin confirmar* (``entry_order_id``
        vacío): la orden pudo llegar al exchange y ``reconcile`` la encontrará (o la descartará)."""
        if not decision.approved:
            raise ValueError(f"decisión de riesgo no aprobada: {decision.reason}")
        coin, side = intent.coin, Side(intent.side)
        mode = EntryMode(intent.entry_mode)
        kind = OrderKind.MARKET if mode is EntryMode.MARKET else OrderKind.LIMIT
        if coin in self.brackets:
            return self._local_reject(coin, side.is_buy, decision.size, kind,
                                      f"ya hay una operación gestionada en {coin} ({self.brackets[coin].status})")
        try:
            self.broker.set_leverage(coin, int(decision.leverage))
        except ValueError as e:
            return self._local_reject(coin, side.is_buy, decision.size, kind, f"apalancamiento no aplicado: {e}")
        meta = self._meta(coin)
        now = self.clock.now()
        ttl = max(self.settings.entry_ttl_minutes, 0.0) * 60.0
        zone = intent.zone
        br = Bracket(
            coin=coin, side=side, origin=intent.origin, status=PENDING, entry_mode=mode.value,
            entry_order_id="", entry_px=decision.entry_px, planned_size=decision.size,
            leverage=int(decision.leverage), sl=decision.sl_price, tp=decision.tp_price,
            expires_ts=(now + ttl) if kind is OrderKind.LIMIT and ttl > 0 else None, created_ts=now,
            reason=intent.reason, cmd_id=cmd_id,
            zone=None if zone is None else {"low": zone.low, "high": zone.high, "usd": zone.usd,
                                            "source": zone.source},
        )
        # Registrado y guardado ANTES de enviar: un timeout (o un reinicio) tras llegar la orden al exchange
        # no deja una posición sin el SL/TP aprobados.
        self.brackets[coin] = br
        self.save_state()
        try:
            if kind is OrderKind.MARKET:
                slip = max(self.settings.market_slippage_bps, 0.0) / 10_000.0
                worst = decision.entry_px * (1.0 + slip) if side.is_buy else decision.entry_px * (1.0 - slip)
                order = self.broker.place_order(coin, side.is_buy, decision.size, OrderKind.MARKET,
                                                price=round_price(worst, meta.sz_decimals), tag="entry")
            else:
                order = self.broker.place_order(coin, side.is_buy, decision.size, OrderKind.LIMIT,
                                                price=decision.entry_px, tif="Gtc", tag="entry")
        except BrokerError as e:
            self._event("entry_uncertain", coin=coin, side=side.value, origin=intent.origin, error=str(e),
                        sl=br.sl, tp=br.tp, cmd_id=cmd_id)
            log.warning("%s: envío de la entrada sin confirmar (%s); reconcile buscará la orden", coin, e)
            raise                                  # el bracket (ya guardado) queda sin confirmar
        if order.status is OrderStatus.REJECTED:
            del self.brackets[coin]
            self.save_state()
            self._event("entry_rejected", coin=coin, side=side.value, origin=intent.origin, reason=order.reason,
                        cmd_id=cmd_id)
            return order
        br.entry_order_id = order.id
        self._event("entry_placed", coin=coin, side=side.value, origin=intent.origin, order=order.to_dict(),
                    sl=br.sl, tp=br.tp, leverage=br.leverage, expires_ts=br.expires_ts, cmd_id=cmd_id)
        self.save_state()
        return order

    def is_unconfirmed(self, coin: str) -> bool:
        """True si la entrada de ``coin`` se envió pero el exchange no la confirmó (fallo de transporte)."""
        br = self.brackets.get(coin)
        return br is not None and br.status == PENDING and not br.entry_order_id

    # ------------------------------------------------------------------ ejecuciones
    def on_fills(self, fills: list[Fill]) -> list[ClosedTrade]:
        """Aplica ejecuciones: coloca/ajusta SL y TP tras la entrada; OCO y cierre del bracket tras la salida.

        Devuelve las operaciones cerradas (su ``pnl`` va a ``RiskManager.record_trade_result``)."""
        closed: list[ClosedTrade] = []
        touched: dict[str, dict[str, float]] = {}
        fills = self._drop_redelivered(fills)
        for f in fills:
            if self.journal is not None:
                self.journal.fill(f)
            br = self.brackets.get(f.coin)
            if br is not None and f.ts > 0 and br.created_ts > 0 and f.ts < br.created_ts - FILL_CLOCK_SKEW_S:
                br = None                            # fill de antes de esta operación (p. ej. re-entregado)
            if br is None:
                what = "compra" if f.is_buy else "venta"
                liq = " — LIQUIDACIÓN" if f.liquidation else ""
                self._say(f"{f.coin}: ejecución fuera de una operación gestionada: {what} {fmt_size(f.size)} "
                          f"@ {fmt_px(f.price)} ({f.tag or 'sin etiqueta'}){liq}", CRITICAL)
                self._event("fill_unmatched", fill=f.to_dict())
                continue
            info = touched.setdefault(f.coin, {"entry": 0.0, "entry_cost": 0.0, "exit": 0.0, "assumed": 0.0})
            if f.is_buy == br.side.is_buy:          # abre / aumenta
                if not br.entry_order_id and f.order_id:
                    br.entry_order_id = f.order_id    # entrada sin confirmar: el fill dice qué orden era
                # Parte ya contada por reconcile (posición vista en la cuenta antes que su fill): no se suma dos veces.
                known = min(br.assumed, f.size)
                br.assumed -= known
                new = f.size - known
                br.filled += new
                br.entry_cost += new * f.price
                br.position += new
                info["entry"] += new
                info["entry_cost"] += new * f.price
                info["assumed"] += known
                if br.status == PENDING:
                    br.status = OPEN
                    br.opened_ts = f.ts
                    if br.canceling:
                        br.canceling = False
                        self._say(f"{f.coin}: la entrada se ejecutó antes de confirmarse su cancelación; se gestiona "
                                  "con su SL/TP", WARN)
            else:                                    # reduce / cierra
                br.position = max(br.position - f.size, 0.0)
                br.exit_size += f.size
                br.exit_value += f.size * f.price
                br.exit_reason = self._exit_role(br, f)
                info["exit"] += f.size
            br.realized += f.realized_pnl
            br.fees += f.fee

        for coin, info in touched.items():
            br = self.brackets.get(coin)
            if br is None:
                continue
            try:
                lot = self._lot(coin)
                if info["entry"] > 0:
                    px = info["entry_cost"] / info["entry"]
                    full = br.filled >= br.planned_size - lot / 2
                    what = "ejecutada" if full else f"ejecutada parcialmente ({fmt_size(br.filled)} de {fmt_size(br.planned_size)})"
                    self._say(f"{coin} {br.side.value.upper()}: entrada {what}: {fmt_size(info['entry'])} @ {fmt_px(px)}")
                if br.status in (OPEN, CLOSING) and br.position < lot / 2 and info["exit"] > 0:
                    closed.append(self._finish(br))
                elif br.position >= lot / 2 and br.status in (OPEN, CLOSING):
                    done = self._protect(br)
                    self._say_protect(coin, done, f" (cubren {fmt_size(br.position)} {coin})")
            except (BrokerError, MarketDataError) as e:   # una moneda nunca deja sin procesar a las demás
                log.warning("%s: fallo procesando ejecuciones (%s); reconcile lo reintentará", coin, e)
                self._event("fills_error", coin=coin, error=str(e))
        if fills:
            self.save_state()
        return closed

    @staticmethod
    def _fill_key(f: Fill) -> str:
        return f"{f.order_id}|{f.coin}|{'B' if f.is_buy else 'S'}|{f.size:.10g}|{f.price:.10g}|{f.ts:.3f}"

    def _drop_redelivered(self, fills: list[Fill]) -> list[Fill]:
        """Quita los fills ya aplicados en el último instante visto (re-entrega tras un reinicio con el cursor
        inclusivo de live) y actualiza ese registro. Un fill idéntico dentro del mismo lote sí se aplica (dos
        contrapartidas del mismo tamaño y precio en la misma orden). Sin hora (``ts`` 0) no se deduplica."""
        prior = dict(self._last_fill_keys)
        out: list[Fill] = []
        for f in fills:
            key = self._fill_key(f)
            if 0 < f.ts <= self._last_fill_ts and prior.get(key, 0) > 0:
                prior[key] -= 1
                log.warning("%s: fill ya aplicado re-entregado (%s); se ignora", f.coin, key)
                self._event("fill_duplicate", fill=f.to_dict())
                continue
            out.append(f)
        for f in out:
            if f.ts <= 0:
                continue
            if f.ts > self._last_fill_ts:
                self._last_fill_ts, self._last_fill_keys = f.ts, {}
            if f.ts == self._last_fill_ts:
                key = self._fill_key(f)
                self._last_fill_keys[key] = self._last_fill_keys.get(key, 0) + 1
        return out

    def _exit_role(self, br: Bracket, f: Fill) -> str:
        oid = f.order_id
        if f.liquidation or f.tag == "liquidation":
            return "liquidation"
        if oid and oid in br.close_order_ids:
            return br.exit_reason or "close"
        if (oid and oid == br.sl_order_id) or f.tag == "sl":
            return "sl"
        if (oid and oid == br.tp_order_id) or f.tag == "tp":
            return "tp"
        if f.tag in EXIT_LABELS:
            return f.tag
        return br.exit_reason or "external"

    def _finish(self, br: Bracket) -> ClosedTrade:
        """Cierra el bracket: OCO (retira la otra orden), cancela restos de entrada y registra la operación."""
        coin = br.coin
        reason = br.exit_reason or "external"
        notes = []
        if br.sl_order_id and reason != "sl":
            self._cancel_quiet(coin, br.sl_order_id)
            if reason == "tp":
                notes.append("SL cancelado (OCO)")
        if br.tp_order_id and reason != "tp":
            self._cancel_quiet(coin, br.tp_order_id)
            if reason == "sl":
                notes.append("TP cancelado (OCO)")
        if br.entry_order_id and br.filled < br.planned_size - self._lot(coin) / 2:
            if self._cancel_quiet(coin, br.entry_order_id):
                notes.append("resto de la entrada cancelado")
        now = self.clock.now()
        pnl = br.realized - br.fees
        trade = ClosedTrade(coin=coin, side=br.side, origin=br.origin, size=br.filled, entry_px=br.avg_entry,
                            exit_px=br.avg_exit, pnl=pnl, gross_pnl=br.realized, fees=br.fees, exit_reason=reason,
                            opened_ts=br.opened_ts, closed_ts=now, reason=br.reason, cmd_id=br.cmd_id)
        if self.journal is not None:
            self.journal.trade(trade.to_dict())
        self._event("trade_closed", **trade.to_dict())
        del self.brackets[coin]
        label = EXIT_LABELS.get(reason, reason)
        tail = f". {'; '.join(notes)}" if notes else ""
        self._say(f"{coin} {br.side.value.upper()} cerrada por {label}: {fmt_size(br.filled)} @ {fmt_px(br.avg_entry)} "
                  f"→ {fmt_px(br.avg_exit)}; PnL neto {fmt_usd(pnl)} (bruto {fmt_usd(br.realized)}, "
                  f"comisiones {fmt_usd(br.fees, sign=False)}){tail}",
                  CRITICAL if reason in ("liquidation", "no_sl") else INFO)
        return trade

    # ------------------------------------------------------------------ protección (SL/TP)
    def _protect(self, br: Bracket, open_ids: Optional[set[str]] = None) -> list[str]:
        """Garantiza STOP y TAKE_PROFIT reduce-only por el tamaño abierto. Devuelve lo que hizo.

        ``open_ids`` (ids abiertos según el exchange) permite detectar órdenes que faltan; sin él se
        confía en los ids conocidos. Coloca la orden nueva antes de cancelar la vieja. Nunca lanza por
        fallos de transporte: los anota (``sl_errors``) y la próxima reconciliación lo reintenta."""
        coin = br.coin
        try:
            meta = self._meta(coin)
        except (KeyError, MarketDataError) as e:
            log.warning("%s: sin metadatos para proteger la posición (%s)", coin, e)
            return []
        size = round_size(br.position, meta.sz_decimals)
        if size <= 0:
            return []
        lot = 10.0 ** -meta.sz_decimals
        now = self.clock.now()
        done: list[str] = []
        self._protect_failed = set()
        for attr, kind, px in (("sl", OrderKind.STOP, br.sl), ("tp", OrderKind.TAKE_PROFIT, br.tp)):
            if px is None:
                continue
            label = attr.upper()
            oid: str = getattr(br, f"{attr}_order_id")
            covered: float = getattr(br, f"{attr}_size")
            present = bool(oid)
            if oid and open_ids is not None and oid not in open_ids:
                if now - getattr(br, f"{attr}_placed_ts") < PLACE_GRACE_S:
                    continue                       # recién colocada: puede no verse todavía
                present = False
            if present and covered >= size - lot / 2:
                continue
            try:
                order = self.broker.place_order(coin, not br.side.is_buy, size, kind, trigger_px=px,
                                                reduce_only=True, tag=attr)
            except BrokerError as e:              # se reintenta en la próxima reconciliación
                if attr == "sl":
                    br.sl_errors += 1
                self._protect_failed.add(attr)
                done.append(f"{label} {fmt_px(px)} NO colocado (fallo de conexión: {e}; se reintenta)")
                self._event("protect_error", coin=coin, level=attr, px=px, error=str(e))
                continue
            if attr == "sl":
                br.sl_errors = 0
            if order.status is OrderStatus.REJECTED:
                fails = getattr(br, f"{attr}_failures") + 1
                setattr(br, f"{attr}_failures", fails)
                self._protect_failed.add(attr)
                self._event("protect_rejected", coin=coin, level=attr, px=px, reason=order.reason)
                if attr == "tp" and fails >= MAX_TP_FAILURES:
                    # Un TP que el exchange rechaza una y otra vez no se reintenta para siempre: la posición sigue
                    # protegida por su SL (el SL rechazado sí acaba en cierre de seguridad).
                    br.tp, br.tp_failures = None, 0
                    self._protect_failed.add("tp_off")
                    done.append(f"TP {fmt_px(px)} RECHAZADO ({order.reason}) {fails} veces seguidas: TP desactivado "
                                f"(la posición sigue con su SL; 'tp {coin} <precio>' para ponerlo otra vez)")
                    self._event("tp_disabled", coin=coin, px=px, failures=fails)
                    continue
                done.append(f"{label} {fmt_px(px)} RECHAZADO ({order.reason})")
                continue
            setattr(br, f"{attr}_failures", 0)
            setattr(br, f"{attr}_order_id", order.id)
            setattr(br, f"{attr}_size", size)
            setattr(br, f"{attr}_placed_ts", now)
            if not oid:
                done.append(f"{label} {fmt_px(order.trigger_px or px)} colocado")
            elif present:
                self._cancel_quiet(coin, oid)
                done.append(f"{label} {fmt_px(order.trigger_px or px)} actualizado")
            else:
                done.append(f"{label} {fmt_px(order.trigger_px or px)} recolocado (no estaba en el exchange)")
            self._event("protect_placed", coin=coin, level=attr, px=px, size=size, order_id=order.id,
                        replaced=oid or None)
        return done

    # ------------------------------------------------------------------ reconciliación
    def reconcile(self, account: AccountState) -> list[ClosedTrade]:
        """Compara los brackets con la cuenta real y corrige diferencias (ver docstring del módulo).

        Cada moneda se procesa aparte: un fallo de transporte en una (p. ej. al cerrar) se registra y la
        reconciliación sigue con las demás."""
        now = self.clock.now()
        closed: list[ClosedTrade] = []
        positions = self._positions(account)
        open_orders = [o for o in account.open_orders if o.is_open]
        open_ids = {o.id for o in open_orders}

        # 0) Cancelaciones que fallaron por transporte (las entradas en cancelación las reintenta el paso 1).
        if self._retry_cancel:
            pending, self._retry_cancel = self._retry_cancel, []
            handled = {b.entry_order_id for b in self.brackets.values() if b.status == PENDING and b.canceling}
            for coin, oid in pending:
                if oid in open_ids and oid not in handled and self._cancel_quiet(coin, oid):
                    open_ids.discard(oid)

        # 1) Brackets conocidos.
        for coin, br in list(self.brackets.items()):
            try:
                trade = self._reconcile_bracket(br, positions.get(coin), open_orders, open_ids, account, now)
            except (BrokerError, MarketDataError) as e:
                log.warning("%s: reconciliación incompleta (%s); se reintenta en el próximo ciclo", coin, e)
                self._event("reconcile_error", coin=coin, error=str(e))
                continue
            if trade is not None:
                closed.append(trade)

        # 2) Posiciones sin bracket: se adoptan con SL por defecto.
        for coin, pos in positions.items():
            if coin not in self.brackets:
                try:
                    self._adopt(pos, open_orders, protect=True)
                except (BrokerError, MarketDataError) as e:
                    log.warning("%s: no se pudo adoptar la posición (%s); se reintenta", coin, e)

        # 3) Órdenes reduce-only huérfanas (sin posición que reducir). También con una entrada pendiente en
        #    la moneda: un SL viejo reduciría la posición nueva en cuanto se abriera.
        for o in open_orders:
            if o.reduce_only and o.coin not in positions:
                if self._cancel_quiet(o.coin, o.id):
                    tag = f" {o.tag}" if o.tag else ""
                    self._say(f"{o.coin}: orden reduce-only huérfana cancelada ({o.kind.value}{tag} "
                              f"{fmt_px(o.trigger_px or o.price)})")
        self.save_state()
        return closed

    def _reconcile_bracket(self, br: Bracket, pos: Optional[Position], open_orders: list[Order], open_ids: set[str],
                           account: AccountState, now: float) -> Optional[ClosedTrade]:
        coin = br.coin
        same_side = pos is not None and (pos.size > 0) == br.side.is_buy
        if br.status == PENDING:
            if not br.entry_order_id:
                self._claim_entry(br, open_orders)
            entry_open = bool(br.entry_order_id) and br.entry_order_id in open_ids
            if same_side:
                # La posición existe pero el fill no llegó (latencia o fill perdido): se da por abierta. Lo que se
                # cuenta aquí queda como ``assumed`` para no sumarlo otra vez cuando llegue el fill.
                br.status, br.missing_checks, br.canceling = OPEN, 0, False
                size = abs(pos.size)
                br.assumed = max(size - br.filled, 0.0)
                br.position = size
                br.filled = max(br.filled, size)
                if br.entry_cost <= 0:
                    br.entry_cost = size * pos.entry_px
                br.opened_ts = br.opened_ts or pos.opened_ts or now
                self._say(f"{coin}: la entrada aparece ejecutada en la cuenta ({fmt_size(size)} "
                          f"@ {fmt_px(pos.entry_px)}); se protege", WARN)
            elif entry_open and (br.canceling or (br.expires_ts is not None and now >= br.expires_ts)):
                expired = not br.canceling
                if self._cancel_quiet(coin, br.entry_order_id):
                    del self.brackets[coin]
                    self._event("entry_expired" if expired else "entry_canceled", coin=coin,
                                order_id=br.entry_order_id)
                    what = (f"caducada sin ejecutarse ({self.settings.entry_ttl_minutes:g} min); cancelada" if expired
                            else "cancelada (confirmado)")
                    self._say(f"{coin} {br.side.value.upper()}: entrada límite @ {fmt_px(br.entry_px)} {what}")
                elif not br.canceling:
                    # Sin confirmar (red, o se acaba de ejecutar): se conserva con su SL/TP por si se ejecuta.
                    br.canceling = True
                    self._say(f"{coin} {br.side.value.upper()}: entrada límite @ {fmt_px(br.entry_px)} caducada, pero "
                              "su cancelación no se confirmó; se reintenta (si se ejecuta, se protege con su SL/TP)", WARN)
                return None
            elif not entry_open:
                br.missing_checks += 1
                if br.missing_checks >= MISSING_CHECKS:
                    del self.brackets[coin]
                    self._event("entry_missing", coin=coin, order_id=br.entry_order_id, canceling=br.canceling,
                                unconfirmed=not br.entry_order_id)
                    if br.canceling:
                        why = "entrada cancelada (ya no está en el exchange)"
                    elif not br.entry_order_id:
                        why = "la entrada sin confirmar no aparece en el exchange (no llegó o se rechazó)"
                    else:
                        why = "la orden de entrada ya no está en el exchange (cancelada o rechazada fuera de la mesa)"
                    self._say(f"{coin} {br.side.value.upper()}: {why}; operación descartada")
                return None
            else:
                br.missing_checks = 0
                return None
        if not same_side:
            br.flat_checks += 1
            if br.flat_checks >= MISSING_CHECKS:
                if not br.exit_reason:
                    br.exit_reason = "external"
                br.position = 0.0
                return self._finish(br)
            return None
        br.flat_checks = 0
        size = abs(pos.size)
        if abs(size - br.position) >= self._lot(coin) / 2:
            log.info("%s: tamaño gestionado %g -> %g (según la cuenta)", coin, br.position, size)
            br.position = size
        if br.entry_order_id in open_ids and br.expires_ts is not None and now >= br.expires_ts:
            if self._cancel_quiet(coin, br.entry_order_id):
                self._say(f"{coin}: resto de la entrada caducado y cancelado; se gestiona lo ejecutado "
                          f"({fmt_size(size)})")
        done = self._protect(br, open_ids)
        self._say_protect(coin, done)
        if br.status == CLOSING:
            self._retry_close(br, account, now)
        elif br.sl is not None and (br.sl_failures >= MAX_SL_FAILURES or br.sl_errors >= MAX_SL_ERRORS):
            why = (f"se rechazó {br.sl_failures} veces seguidas" if br.sl_failures >= MAX_SL_FAILURES
                   else f"no se pudo colocar {br.sl_errors} veces seguidas (conexión)")
            self._say(f"{coin}: el SL {why}; cierro la posición por seguridad", CRITICAL, f"no_sl:{coin}")
            self.close(coin, reason="no_sl", account=account)
        return None

    def _claim_entry(self, br: Bracket, open_orders: list[Order]) -> None:
        """Entrada sin confirmar (timeout al enviarla): busca en el exchange la orden que se envió."""
        lot = self._lot(br.coin)
        for o in open_orders:
            if (o.coin == br.coin and not o.reduce_only and o.is_buy == br.side.is_buy
                    and abs(o.size - br.planned_size) < lot / 2
                    and all(b.entry_order_id != o.id for b in self.brackets.values())):
                br.entry_order_id, br.missing_checks = o.id, 0
                self._event("entry_found", coin=br.coin, order_id=o.id)
                self._say(f"{br.coin} {br.side.value.upper()}: la entrada sin confirmar está en el exchange "
                          f"(orden {o.id}); se gestiona con su SL/TP y caducidad")
                return

    def _retry_close(self, br: Bracket, account: AccountState, now: float) -> None:
        """Cierre pedido que dejó posición (IOC parcial o rechazada): se reintenta con más tolerancia."""
        if now - br.close_ts < CLOSE_RETRY_S:
            return
        label = EXIT_LABELS.get(br.exit_reason, br.exit_reason or "cierre")
        if br.close_attempts < MAX_CLOSE_ATTEMPTS:
            self._say(f"{br.coin}: la posición sigue abierta tras el cierre ({label}); reintento "
                      f"{br.close_attempts}/{MAX_CLOSE_ATTEMPTS - 1}", WARN)
            self.close(br.coin, br.exit_reason or "close", account=account)
        elif not br.close_warned:
            br.close_warned = True
            self._say(f"{br.coin}: ATENCIÓN: el cierre ({label}) no se completó tras {br.close_attempts} intentos; la "
                      f"posición sigue abierta con su SL/TP. Revisa el exchange o repite 'cierra {br.coin}'.", CRITICAL)
            self._event("close_gave_up", coin=br.coin, attempts=br.close_attempts, reason=br.exit_reason)

    def _default_sl(self, coin: str, side: Side, entry: float, mark: Optional[float],
                    liq: Optional[float] = None) -> float:
        """SL por defecto (``default_sl_pct``) que nunca queda más allá de la liquidación ``liq``."""
        pct = max(self.settings.default_sl_pct, 0.01) / 100.0
        long = side is Side.LONG
        sl = entry * (1.0 - pct) if long else entry * (1.0 + pct)
        limit = self._liq_limit(side, entry, liq)
        if limit is not None and (sl < limit if long else sl > limit):
            sl = limit                                                 # la liquidación llegaría antes
        if mark is not None and mark > 0 and ((long and sl >= mark) or (not long and sl <= mark)):
            sl = mark * (1.0 - pct) if long else mark * (1.0 + pct)     # ya cruzado: desde el precio actual
            if liq is not None and liq > 0 and (sl <= liq if long else sl >= liq):
                sl = (mark + liq) / 2.0                                 # pegado a la liquidación: a medio camino
        return self._round_px(coin, sl)

    def _adopt(self, pos: Position, open_orders: list[Order], *, protect: bool) -> Bracket:
        """Crea un bracket para una posición que la mesa no gestionaba (reinicio sin estado, orden manual).

        Reutiliza SL/TP reduce-only que ya existan en el exchange; si no hay SL pone uno por defecto."""
        coin, side = pos.coin, pos.side
        size = abs(pos.size)
        now = self.clock.now()
        exit_buy = not side.is_buy
        mine = [o for o in open_orders if o.coin == coin and o.reduce_only and o.is_buy == exit_buy and o.is_open]
        sl_o = next((o for o in mine if o.kind is OrderKind.STOP and o.trigger_px), None)
        tp_o = next((o for o in mine if o.kind is OrderKind.TAKE_PROFIT and o.trigger_px), None)
        mark = pos.mark_px if pos.mark_px else self._mid(coin)
        br = Bracket(coin=coin, side=side, origin="adopted", status=OPEN, entry_mode="", entry_px=pos.entry_px,
                     planned_size=size, leverage=int(pos.leverage or 1), created_ts=now,
                     opened_ts=pos.opened_ts or now, position=size, filled=size, entry_cost=size * pos.entry_px,
                     reason="posición adoptada")
        if sl_o is not None:
            br.sl, br.sl_order_id, br.sl_size, br.sl_placed_ts = sl_o.trigger_px, sl_o.id, sl_o.remaining, now
        else:
            liq = self._liq_px(coin, side, pos.entry_px, br.leverage, pos.liq_px)
            br.sl = self._default_sl(coin, side, pos.entry_px, mark, liq)
        if tp_o is not None:
            br.tp, br.tp_order_id, br.tp_size, br.tp_placed_ts = tp_o.trigger_px, tp_o.id, tp_o.remaining, now
        self.brackets[coin] = br
        self._event("position_adopted", coin=coin, side=side.value, size=size, entry_px=pos.entry_px,
                    sl=br.sl, tp=br.tp, reused_sl=sl_o.id if sl_o else None)
        where = "existente" if sl_o is not None else f"por defecto ({self.settings.default_sl_pct:g}%)"
        self._say(f"{coin} {side.value.upper()} {fmt_size(size)} @ {fmt_px(pos.entry_px)}: posición sin bracket "
                  f"adoptada (SL {where} {fmt_px(br.sl)}" + (f", TP {fmt_px(br.tp)}" if br.tp else "") + ")", WARN)
        if protect:
            done = self._protect(br, {o.id for o in open_orders})
            self._say_protect(coin, done)
        return br

    # ------------------------------------------------------------------ cierres y cancelaciones
    def _cancel_entries(self, coin: Optional[str]) -> set[str]:
        """Cancela las órdenes de entrada (no reduce-only) abiertas. Devuelve los ids cuya cancelación se confirmó."""
        done: set[str] = set()
        for o in self.broker.open_orders(coin):
            if not o.reduce_only and o.is_open and self._cancel_quiet(o.coin, o.id):
                done.add(o.id)
        return done

    def _drop_pending(self, coin: str, canceled: set[str], reason: str) -> Optional[bool]:
        """Tras pedir cancelar la entrada pendiente de ``coin``: True si se descartó (cancelación confirmada),
        False si queda *en cancelación* (conserva su SL/TP por si se ejecuta; ``reconcile`` lo resuelve) y
        None si no había entrada pendiente sin ejecutar."""
        br = self.brackets.get(coin)
        if br is None or br.status != PENDING or br.filled > 0:
            return None
        if br.entry_order_id and br.entry_order_id in canceled:
            del self.brackets[coin]
            self._event("entry_canceled", coin=coin, order_id=br.entry_order_id, reason=reason)
            return True
        if not br.canceling:
            br.canceling = True
            self._event("entry_cancel_unconfirmed", coin=coin, order_id=br.entry_order_id, reason=reason)
            self._say(f"{coin}: la cancelación de la entrada pendiente no se confirmó; se reintenta (si se ejecuta, "
                      "se protege con su SL/TP)", WARN, f"cancel:{coin}")
        return False

    def cancel(self, coin: Optional[str] = None) -> int:
        """Cancela órdenes de *entrada* (no reduce-only) de ``coin`` o de todas. Los SL/TP de posiciones
        abiertas se mantienen (para salir: ``close``). Devuelve cuántas se cancelaron (confirmadas)."""
        canceled = self._cancel_entries(coin)
        changed = bool(canceled)
        for c in list(self.brackets):
            if coin is None or c == coin:
                changed = self._drop_pending(c, canceled, "cancel") is not None or changed
        if changed:
            self.save_state()
        return len(canceled)

    def _close_limit(self, coin: str, pos: Position, attempt: int) -> Optional[float]:
        """Peor precio aceptable de un cierre a mercado: ``CLOSE_SLIPPAGE`` (creciente por intento) más allá de
        la referencia (mid; sin datos de mercado, el mark de la posición). None si no hay referencia."""
        ref = self._mid(coin) or pos.mark_px
        if not ref or ref <= 0:
            return None
        slip = min(CLOSE_SLIPPAGE * max(attempt, 1), CLOSE_MAX_SLIPPAGE)
        return self._round_px(coin, ref * (1.0 - slip) if pos.is_long else ref * (1.0 + slip))

    def close(self, coin: str, reason: str = "close", *, account: Optional[AccountState] = None) -> Optional[Order]:
        """Cierra a mercado la posición de ``coin`` (reduce-only) y cancela su entrada pendiente.

        Devuelve la orden de cierre (``REJECTED`` si el broker la rechazó) o None si no había posición. La
        intención de cerrar queda en el bracket (``CLOSING``): si el cierre se rechaza o deja posición (IOC
        parcial), ``reconcile`` lo reintenta. Lanza ``BrokerError`` en fallos de transporte."""
        try:
            canceled = self._cancel_entries(coin)
        except BrokerError as e:                  # lo urgente es cerrar la posición
            log.warning("%s: no se pudieron cancelar las entradas antes de cerrar (%s)", coin, e)
            canceled = set()
        acct = account if account is not None else self.broker.account()
        pos = self._positions(acct).get(coin)
        br = self.brackets.get(coin)
        if pos is None:
            dropped = self._drop_pending(coin, canceled, reason)
            if dropped:
                self._say(f"{coin}: entrada pendiente cancelada")
                self.save_state()
            elif dropped is False:
                self.save_state()
            elif canceled:
                self._say(f"{coin}: {len(canceled)} orden(es) de entrada cancelada(s)")
            return None
        if br is None or br.side is not pos.side:
            br = self._adopt(pos, [o for o in acct.open_orders if o.is_open], protect=False)
        if br.status != CLOSING:
            br.close_attempts, br.close_warned = 0, False
        br.status, br.exit_reason = CLOSING, reason
        br.close_attempts += 1
        br.close_ts = self.clock.now()
        try:
            order = self.broker.place_order(coin, not pos.is_long, abs(pos.size), OrderKind.MARKET,
                                            price=self._close_limit(coin, pos, br.close_attempts),
                                            reduce_only=True, tag=reason)
        except BrokerError:
            self.save_state()                     # la intención de cerrar queda guardada: se reintenta
            raise
        if order.status is OrderStatus.REJECTED:
            self._say(f"{coin}: el cierre fue rechazado por el broker: {order.reason}; se reintentará", CRITICAL,
                      f"close:{coin}")
            self._event("close_rejected", coin=coin, reason=order.reason, attempt=br.close_attempts)
            self.save_state()
            return order
        br.close_order_ids.append(order.id)
        self._event("close_sent", coin=coin, reason=reason, order=order.to_dict(), attempt=br.close_attempts)
        self._say(f"{coin}: cierre a mercado enviado ({'venta' if pos.is_long else 'compra'} "
                  f"{fmt_size(abs(pos.size))}, {EXIT_LABELS.get(reason, reason)})")
        self.save_state()
        return order

    def close_all(self, reason: str = "close_all") -> list[Order]:
        """Cancela todas las entradas y cierra todas las posiciones a mercado.

        Cada moneda por separado: una que falla (sin precio, rechazo, red) no impide cerrar las demás. Las que
        no se pudieron cerrar quedan en ``close_failures`` ({moneda: motivo}) y en los mensajes. Devuelve las
        órdenes de cierre aceptadas (no cuenta las rechazadas). Lanza ``BrokerError`` sólo si no puede leer
        la cuenta."""
        self.close_failures = {}
        try:
            canceled = self._cancel_entries(None)
        except BrokerError as e:
            log.warning("close_all: no se pudieron cancelar las entradas (%s); se cierran las posiciones", e)
            self._say(f"No se pudieron cancelar las entradas pendientes ({e}); se reintentará", WARN, "cancel_entries")
            canceled = set()
        for c in list(self.brackets):
            self._drop_pending(c, canceled, reason)
        acct = self.broker.account()
        out = []
        for coin in sorted(self._positions(acct)):
            try:
                order = self.close(coin, reason, account=acct)
            except (BrokerError, MarketDataError, KeyError) as e:
                self.close_failures[coin] = str(e)
                self._say(f"{coin}: no se pudo enviar el cierre ({e}); se reintentará", CRITICAL, f"close:{coin}")
                self._event("close_failed", coin=coin, reason=reason, error=str(e))
                continue
            if order is None:
                continue
            if order.status is OrderStatus.REJECTED:
                self.close_failures[coin] = order.reason or "rechazado"
                continue
            out.append(order)
        self.save_state()
        return out

    def set_stop(self, coin: str, px: float) -> str:
        """Mueve (o pone) el SL de la posición o de la entrada pendiente de ``coin``. ``ValueError`` si no procede
        (lado equivocado, más allá de la liquidación o pérdida mayor que ``risk_per_trade_pct``)."""
        return self._set_level(coin, "sl", px)

    def set_take_profit(self, coin: str, px: float) -> str:
        """Mueve (o pone) el TP de la posición o de la entrada pendiente de ``coin``. ``ValueError`` si no procede."""
        return self._set_level(coin, "tp", px)

    def reference_px(self, coin: str) -> Optional[tuple[Side, float]]:
        """(lado, precio de entrada) de la posición o la entrada pendiente de ``coin`` (para SL/TP en %)."""
        br = self.brackets.get(coin)
        acct = self.broker.account()
        pos = self._positions(acct).get(coin)
        if pos is not None:
            return pos.side, pos.entry_px
        if br is not None:
            return br.side, br.entry_px
        return None

    def _set_level(self, coin: str, attr: str, px: float) -> str:
        label = "SL" if attr == "sl" else "TP"
        if px is None or not isinstance(px, (int, float)) or not math.isfinite(px) or px <= 0:
            raise ValueError(f"{label} inválido: {px!r}")
        px = self._round_px(coin, float(px))
        br = self.brackets.get(coin)
        acct = self.broker.account()
        pos = self._positions(acct).get(coin)
        if pos is None:
            if br is not None and br.status == PENDING:
                long = br.side is Side.LONG
                ref = br.entry_px
                wrong = (px >= ref if long else px <= ref) if attr == "sl" else (px <= ref if long else px >= ref)
                if wrong:
                    where = ("por debajo" if long else "por encima") if attr == "sl" else ("por encima" if long else "por debajo")
                    raise ValueError(f"el {label} de un {br.side.value.upper()} debe quedar {where} de la entrada "
                                     f"{fmt_px(ref)}")
                if attr == "sl":                   # el tamaño se aprobó con el SL original: no se ensancha el riesgo
                    self._check_stop(coin, br.side, ref, br.planned_size, br.leverage, px, br.sl, acct.equity)
                old = getattr(br, attr)
                setattr(br, attr, px)
                self.save_state()
                return (f"{coin}: {label} de la entrada pendiente movido a {fmt_px(px)} (antes {fmt_px(old)}); "
                        "se colocará al ejecutarse la entrada")
            raise ValueError(f"no hay posición abierta ni entrada pendiente en {coin}")
        long = pos.size > 0
        mark = pos.mark_px if pos.mark_px else self._mid(coin)
        if mark is not None:
            if attr == "sl" and (px >= mark if long else px <= mark):
                where = "por debajo" if long else "por encima"
                raise ValueError(f"el SL de un {pos.side.value.upper()} debe quedar {where} del precio actual "
                                 f"({fmt_px(mark)}); para salir ya usa 'cierra {coin}'")
            if attr == "tp" and (px <= mark if long else px >= mark):
                where = "por encima" if long else "por debajo"
                raise ValueError(f"el TP de un {pos.side.value.upper()} debe quedar {where} del precio actual "
                                 f"({fmt_px(mark)}); para salir ya usa 'cierra {coin}'")
        open_orders = [o for o in acct.open_orders if o.is_open]
        if br is None or br.side is not pos.side:
            br = self._adopt(pos, open_orders, protect=False)
        if attr == "sl":
            lev = int(pos.leverage or br.leverage or 1)
            self._check_stop(coin, pos.side, pos.entry_px, abs(pos.size), lev, px, br.sl, acct.equity, pos.liq_px)
        old = getattr(br, attr)
        setattr(br, attr, px)
        setattr(br, f"{attr}_size", 0.0)           # fuerza colocar la orden nueva
        if attr == "tp":
            br.tp_failures = 0                     # un TP nuevo empieza de cero (tras desactivarse por rechazos)
        br.position = abs(pos.size)
        done = self._protect(br, {o.id for o in open_orders})
        self.save_state()
        failed = [d for d in done if "RECHAZADO" in d or "NO colocado" in d]
        if failed:
            return f"{coin}: no se pudo mover el {label}: {'; '.join(failed)}"
        return f"{coin}: {label} movido a {fmt_px(px)} (antes {fmt_px(old)})"

    # ------------------------------------------------------------------ persistencia
    def snapshot(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "brackets": {c: b.to_dict() for c, b in sorted(self.brackets.items())},
            "retry_cancel": [list(x) for x in self._retry_cancel],
            "last_fills": {"ts": self._last_fill_ts, "keys": dict(sorted(self._last_fill_keys.items()))},
        }

    def save_state(self, force: bool = False) -> None:
        if self.journal is None:
            return
        data = self.snapshot()
        if not force and data == self._saved:
            return
        self.journal.save_json(STATE_NAME, data)
        self._saved = data

    def load_state(self) -> None:
        data = self.journal.load_json(STATE_NAME, None) if self.journal is not None else None
        if data is None:
            return
        if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
            self.load_error = f"estado del motor ilegible o de otra versión ({type(data).__name__}); se ignora"
            log.error("%s", self.load_error)
            return
        for coin, raw in (data.get("brackets") or {}).items():
            try:
                self.brackets[coin] = Bracket.from_dict(raw)
            except (KeyError, TypeError, ValueError) as e:
                log.error("bracket %s ilegible (%s); se ignora (reconcile adoptará la posición)", coin, e)
        self._retry_cancel = [(str(c), str(o)) for c, o in data.get("retry_cancel") or [] if c and o]
        last = data.get("last_fills") if isinstance(data.get("last_fills"), dict) else {}
        try:
            self._last_fill_ts = float(last.get("ts") or 0.0)
            self._last_fill_keys = {str(k): int(v) for k, v in (last.get("keys") or {}).items() if int(v) > 0}
        except (TypeError, ValueError, AttributeError):
            self._last_fill_ts, self._last_fill_keys = 0.0, {}
        self._saved = self.snapshot()
        if self.brackets:
            log.info("motor: %d bracket(s) recargados: %s", len(self.brackets), ", ".join(sorted(self.brackets)))

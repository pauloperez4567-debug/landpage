"""Ejecución: entradas, brackets SL/TP (OCO), caducidad, reconciliación y cierres.

Cada operación gestionada es un ``Bracket`` (uno por moneda) con este ciclo de vida::

    open()        on_fills() (entrada)           on_fills() (SL/TP/cierre) o reconcile()
  ─────────► PENDING ───────────────► OPEN ──────────────────────────────────► cerrada (ClosedTrade)
               │  (caduca / se cancela)  │  close() ─► CLOSING ─► cerrada
               └──► descartada           └─ SL/TP reduce-only por el tamaño de la posición

- **Entrada**: ZONE/LIMIT → LIMIT Gtc a ``decision.entry_px``; MARKET → MARKET con peor precio aceptable
  ``entry_px ± market_slippage_bps`` (protege de saltos entre la evaluación y el envío).
- **Brackets**: al ejecutarse la entrada (total o parcial) se colocan STOP y TAKE_PROFIT reduce-only por
  el tamaño abierto; si la posición crece (más fills parciales) se coloca la nueva orden y *después* se
  cancela la vieja (nunca queda la posición sin stop). Si salta el SL se retira el TP y viceversa (OCO).
- **reconcile(account)**: caducidad de entradas (``entry_ttl_minutes``), entradas que desaparecen del
  exchange, posiciones cerradas fuera de la mesa, SL/TP que faltan (se recolocan), órdenes reduce-only
  huérfanas (se cancelan) y posiciones sin bracket (se adoptan con SL por defecto: p. ej. tras un reinicio
  sin estado o una orden manual en el exchange).
- Si el SL de una posición se rechaza ``MAX_SL_FAILURES`` veces seguidas, la posición se cierra: la mesa
  nunca mantiene una posición sin stop.

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

log = logging.getLogger(__name__)

STATE_NAME = "engine"
STATE_VERSION = 1

PENDING = "pending"     # entrada enviada, sin ejecutar
OPEN = "open"           # posición abierta (total o parcialmente) con SL/TP
CLOSING = "closing"     # cierre a mercado enviado

MAX_SL_FAILURES = 3     # SL rechazado N veces seguidas -> cierre de seguridad
PLACE_GRACE_S = 5.0     # una orden recién colocada puede tardar en verse en open_orders (live)
MISSING_CHECKS = 2      # reconciliaciones seguidas antes de dar por perdida una entrada / posición

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
    origin: str                          # "command" | "auto:<estrategia>" | "adopted"
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
    flat_checks: int = 0                 # reconciliaciones seguidas sin posición en la cuenta
    missing_checks: int = 0              # reconciliaciones seguidas sin la orden de entrada
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
        self._retry_cancel: list[tuple[str, str]] = []     # (moneda, id) de cancelaciones fallidas
        self._metas: dict[str, AssetMeta] = {}
        self._reject_seq = 0
        self._saved: Optional[dict] = None
        self.load_error = ""
        if journal is not None and load:
            self.load_state()

    # ------------------------------------------------------------------ utilidades
    def pop_messages(self) -> list[str]:
        out, self.messages = self.messages, []
        return out

    def _say(self, text: str) -> None:
        self.messages.append(text)       # la mesa los muestra y los registra (logger desk.notice)
        log.debug("%s", text)

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
        """Fija el apalancamiento, envía la entrada y registra el bracket pendiente.

        Devuelve la orden de entrada (``REJECTED`` con motivo si no se pudo enviar). Lanza ``BrokerError``
        sólo en fallos de transporte."""
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
        if kind is OrderKind.MARKET:
            slip = max(self.settings.market_slippage_bps, 0.0) / 10_000.0
            worst = decision.entry_px * (1.0 + slip) if side.is_buy else decision.entry_px * (1.0 - slip)
            order = self.broker.place_order(coin, side.is_buy, decision.size, OrderKind.MARKET,
                                            price=round_price(worst, meta.sz_decimals), tag="entry")
        else:
            order = self.broker.place_order(coin, side.is_buy, decision.size, OrderKind.LIMIT,
                                            price=decision.entry_px, tif="Gtc", tag="entry")
        if order.status is OrderStatus.REJECTED:
            self._event("entry_rejected", coin=coin, side=side.value, origin=intent.origin, reason=order.reason,
                        cmd_id=cmd_id)
            return order
        now = self.clock.now()
        ttl = max(self.settings.entry_ttl_minutes, 0.0) * 60.0
        zone = intent.zone
        br = Bracket(
            coin=coin, side=side, origin=intent.origin, status=PENDING, entry_mode=mode.value,
            entry_order_id=order.id, entry_px=decision.entry_px, planned_size=decision.size,
            leverage=int(decision.leverage), sl=decision.sl_price, tp=decision.tp_price,
            expires_ts=(now + ttl) if kind is OrderKind.LIMIT and ttl > 0 else None, created_ts=now,
            reason=intent.reason, cmd_id=cmd_id,
            zone=None if zone is None else {"low": zone.low, "high": zone.high, "usd": zone.usd,
                                            "source": zone.source},
        )
        self.brackets[coin] = br
        self._event("entry_placed", coin=coin, side=side.value, origin=intent.origin, order=order.to_dict(),
                    sl=br.sl, tp=br.tp, leverage=br.leverage, expires_ts=br.expires_ts, cmd_id=cmd_id)
        self.save_state()
        return order

    # ------------------------------------------------------------------ ejecuciones
    def on_fills(self, fills: list[Fill]) -> list[ClosedTrade]:
        """Aplica ejecuciones: coloca/ajusta SL y TP tras la entrada; OCO y cierre del bracket tras la salida.

        Devuelve las operaciones cerradas (su ``pnl`` va a ``RiskManager.record_trade_result``)."""
        closed: list[ClosedTrade] = []
        touched: dict[str, dict[str, float]] = {}
        for f in fills:
            if self.journal is not None:
                self.journal.fill(f)
            br = self.brackets.get(f.coin)
            if br is None:
                what = "compra" if f.is_buy else "venta"
                liq = " — LIQUIDACIÓN" if f.liquidation else ""
                self._say(f"{f.coin}: ejecución fuera de una operación gestionada: {what} {fmt_size(f.size)} "
                          f"@ {fmt_px(f.price)} ({f.tag or 'sin etiqueta'}){liq}")
                self._event("fill_unmatched", fill=f.to_dict())
                continue
            info = touched.setdefault(f.coin, {"entry": 0.0, "entry_cost": 0.0, "exit": 0.0})
            if f.is_buy == br.side.is_buy:          # abre / aumenta
                br.filled += f.size
                br.entry_cost += f.size * f.price
                br.position += f.size
                info["entry"] += f.size
                info["entry_cost"] += f.size * f.price
                if br.status == PENDING:
                    br.status = OPEN
                    br.opened_ts = f.ts
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
                if done:
                    self._say(f"{coin}: " + ", ".join(done) + f" (cubren {fmt_size(br.position)} {coin})")
        if fills:
            self.save_state()
        return closed

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
                  f"comisiones {fmt_usd(br.fees, sign=False)}){tail}")
        return trade

    # ------------------------------------------------------------------ protección (SL/TP)
    def _protect(self, br: Bracket, open_ids: Optional[set[str]] = None) -> list[str]:
        """Garantiza STOP y TAKE_PROFIT reduce-only por el tamaño abierto. Devuelve lo que hizo.

        ``open_ids`` (ids abiertos según el exchange) permite detectar órdenes que faltan; sin él se
        confía en los ids conocidos. Coloca la orden nueva antes de cancelar la vieja."""
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
            order = self.broker.place_order(coin, not br.side.is_buy, size, kind, trigger_px=px,
                                            reduce_only=True, tag=attr)
            if order.status is OrderStatus.REJECTED:
                setattr(br, f"{attr}_failures", getattr(br, f"{attr}_failures") + 1)
                done.append(f"{label} {fmt_px(px)} RECHAZADO ({order.reason})")
                self._event("protect_rejected", coin=coin, level=attr, px=px, reason=order.reason)
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
        """Compara los brackets con la cuenta real y corrige diferencias (ver docstring del módulo)."""
        now = self.clock.now()
        closed: list[ClosedTrade] = []
        positions = self._positions(account)
        open_orders = [o for o in account.open_orders if o.is_open]
        open_ids = {o.id for o in open_orders}

        # 0) Cancelaciones que fallaron por transporte.
        if self._retry_cancel:
            pending, self._retry_cancel = self._retry_cancel, []
            for coin, oid in pending:
                if oid in open_ids:
                    self._cancel_quiet(coin, oid)

        # 1) Brackets conocidos.
        for coin, br in list(self.brackets.items()):
            pos = positions.get(coin)
            same_side = pos is not None and (pos.size > 0) == br.side.is_buy
            if br.status == PENDING:
                entry_open = br.entry_order_id in open_ids
                if same_side:
                    # La posición existe pero el fill no llegó (latencia o fill perdido): se da por abierta.
                    br.status, br.missing_checks = OPEN, 0
                    br.position = abs(pos.size)
                    br.filled = max(br.filled, abs(pos.size))
                    if br.entry_cost <= 0:
                        br.entry_cost = abs(pos.size) * pos.entry_px
                    br.opened_ts = br.opened_ts or pos.opened_ts or now
                    self._say(f"{coin}: la entrada aparece ejecutada en la cuenta ({fmt_size(abs(pos.size))} "
                              f"@ {fmt_px(pos.entry_px)}); se protege")
                elif entry_open and br.expires_ts is not None and now >= br.expires_ts:
                    self._cancel_quiet(coin, br.entry_order_id)
                    del self.brackets[coin]
                    self._event("entry_expired", coin=coin, order_id=br.entry_order_id)
                    self._say(f"{coin} {br.side.value.upper()}: entrada límite @ {fmt_px(br.entry_px)} caducada sin "
                              f"ejecutarse ({self.settings.entry_ttl_minutes:g} min); cancelada")
                    continue
                elif not entry_open:
                    br.missing_checks += 1
                    if br.missing_checks >= MISSING_CHECKS:
                        del self.brackets[coin]
                        self._event("entry_missing", coin=coin, order_id=br.entry_order_id)
                        self._say(f"{coin} {br.side.value.upper()}: la orden de entrada ya no está en el exchange "
                                  "(cancelada o rechazada fuera de la mesa); operación descartada")
                    continue
                else:
                    br.missing_checks = 0
                    continue
            if not same_side:
                br.flat_checks += 1
                if br.flat_checks >= MISSING_CHECKS:
                    if not br.exit_reason:
                        br.exit_reason = "external"
                    br.position = 0.0
                    closed.append(self._finish(br))
                continue
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
            if done:
                self._say(f"{coin}: " + ", ".join(done))
            if br.sl is not None and br.sl_failures >= MAX_SL_FAILURES and br.status != CLOSING:
                self._say(f"{coin}: el SL se rechazó {br.sl_failures} veces seguidas; cierro la posición por seguridad")
                self.close(coin, reason="no_sl", account=account)

        # 2) Posiciones sin bracket: se adoptan con SL por defecto.
        for coin, pos in positions.items():
            if coin not in self.brackets:
                self._adopt(pos, open_orders, protect=True)

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

    def _default_sl(self, coin: str, side: Side, entry: float, mark: Optional[float]) -> float:
        pct = max(self.settings.default_sl_pct, 0.01) / 100.0
        long = side is Side.LONG
        sl = entry * (1.0 - pct) if long else entry * (1.0 + pct)
        if mark is not None and mark > 0 and ((long and sl >= mark) or (not long and sl <= mark)):
            sl = mark * (1.0 - pct) if long else mark * (1.0 + pct)     # ya cruzado: desde el precio actual
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
            br.sl = self._default_sl(coin, side, pos.entry_px, mark)
        if tp_o is not None:
            br.tp, br.tp_order_id, br.tp_size, br.tp_placed_ts = tp_o.trigger_px, tp_o.id, tp_o.remaining, now
        self.brackets[coin] = br
        self._event("position_adopted", coin=coin, side=side.value, size=size, entry_px=pos.entry_px,
                    sl=br.sl, tp=br.tp, reused_sl=sl_o.id if sl_o else None)
        where = "existente" if sl_o is not None else f"por defecto ({self.settings.default_sl_pct:g}%)"
        self._say(f"{coin} {side.value.upper()} {fmt_size(size)} @ {fmt_px(pos.entry_px)}: posición sin bracket "
                  f"adoptada (SL {where} {fmt_px(br.sl)}" + (f", TP {fmt_px(br.tp)}" if br.tp else "") + ")")
        if protect:
            done = self._protect(br, {o.id for o in open_orders})
            if done:
                self._say(f"{coin}: " + ", ".join(done))
        return br

    # ------------------------------------------------------------------ cierres y cancelaciones
    def _cancel_entries(self, coin: Optional[str]) -> int:
        n = 0
        for o in self.broker.open_orders(coin):
            if not o.reduce_only and o.is_open and self._cancel_quiet(o.coin, o.id):
                n += 1
        return n

    def cancel(self, coin: Optional[str] = None) -> int:
        """Cancela órdenes de *entrada* (no reduce-only) de ``coin`` o de todas. Los SL/TP de posiciones
        abiertas se mantienen (para salir: ``close``). Devuelve cuántas se cancelaron."""
        n = self._cancel_entries(coin)
        for c, br in list(self.brackets.items()):
            if coin is not None and c != coin:
                continue
            if br.status == PENDING and br.filled <= 0:
                del self.brackets[c]
                self._event("entry_canceled", coin=c, order_id=br.entry_order_id)
        if n:
            self.save_state()
        return n

    def close(self, coin: str, reason: str = "close", *, account: Optional[AccountState] = None) -> Optional[Order]:
        """Cierra a mercado la posición de ``coin`` (reduce-only) y cancela su entrada pendiente.

        Devuelve la orden de cierre o None si no había posición."""
        canceled = self._cancel_entries(coin)
        acct = account if account is not None else self.broker.account()
        pos = self._positions(acct).get(coin)
        br = self.brackets.get(coin)
        if pos is None:
            if br is not None and br.status == PENDING:
                del self.brackets[coin]
                self._event("entry_canceled", coin=coin, order_id=br.entry_order_id, reason=reason)
                self._say(f"{coin}: entrada pendiente cancelada")
                self.save_state()
            elif canceled:
                self._say(f"{coin}: {canceled} orden(es) de entrada cancelada(s)")
            return None
        if br is None or br.side is not pos.side:
            br = self._adopt(pos, [o for o in acct.open_orders if o.is_open], protect=False)
        order = self.broker.place_order(coin, not pos.is_long, abs(pos.size), OrderKind.MARKET,
                                        reduce_only=True, tag=reason)
        if order.status is OrderStatus.REJECTED:
            self._say(f"{coin}: el cierre fue rechazado por el broker: {order.reason}")
            self._event("close_rejected", coin=coin, reason=order.reason)
            return order
        br.status = CLOSING
        br.exit_reason = reason
        br.close_order_ids.append(order.id)
        self._event("close_sent", coin=coin, reason=reason, order=order.to_dict())
        self._say(f"{coin}: cierre a mercado enviado ({'venta' if pos.is_long else 'compra'} "
                  f"{fmt_size(abs(pos.size))}, {EXIT_LABELS.get(reason, reason)})")
        self.save_state()
        return order

    def close_all(self, reason: str = "close_all") -> list[Order]:
        """Cancela todas las entradas y cierra todas las posiciones a mercado."""
        self._cancel_entries(None)
        for c, br in list(self.brackets.items()):
            if br.status == PENDING and br.filled <= 0:
                del self.brackets[c]
        acct = self.broker.account()
        out = []
        for coin in sorted(self._positions(acct)):
            order = self.close(coin, reason, account=acct)
            if order is not None:
                out.append(order)
        self.save_state()
        return out

    def set_stop(self, coin: str, px: float) -> str:
        """Mueve (o pone) el SL de la posición o de la entrada pendiente de ``coin``. ``ValueError`` si no procede."""
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
        old = getattr(br, attr)
        setattr(br, attr, px)
        setattr(br, f"{attr}_size", 0.0)           # fuerza colocar la orden nueva
        br.position = abs(pos.size)
        done = self._protect(br, {o.id for o in open_orders})
        self.save_state()
        failed = [d for d in done if "RECHAZADO" in d]
        if failed:
            return f"{coin}: no se pudo mover el {label}: {'; '.join(failed)}"
        return f"{coin}: {label} movido a {fmt_px(px)} (antes {fmt_px(old)})"

    # ------------------------------------------------------------------ persistencia
    def snapshot(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "brackets": {c: b.to_dict() for c, b in sorted(self.brackets.items())},
            "retry_cancel": [list(x) for x in self._retry_cancel],
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
        self._saved = self.snapshot()
        if self.brackets:
            log.info("motor: %d bracket(s) recargados: %s", len(self.brackets), ", ".join(sorted(self.brackets)))

"""La mesa: bucle 24/7 que une comandos, estrategias, riesgo y ejecución.

Ciclo (cada ``loop_seconds``), cada etapa en su propio try/except::

    comandos (cola en memoria + state/inbox.txt) → broker.step() → poll_fills → engine.on_fills
    → account → risk.update (halts / flatten) → engine.reconcile
    → cada strategy_seconds, si auto y riesgo lo permite: estrategias por símbolo → riesgo → engine
    → guardar estado → línea de estado cada status_every_seconds

- Una etapa que falla se registra y el ciclo sigue con las demás. Si hay fallos en ciclos consecutivos,
  la espera entre ciclos crece exponencialmente (``loop_seconds · 2^n``, máximo 300 s): la mesa nunca muere
  por un fallo de red o de datos y tampoco martillea una API caída.
- Comandos: ``submit(text)`` los ejecuta al momento y devuelve la respuesta legible (consola);
  ``enqueue(text)`` y ``state/inbox.txt`` se procesan al principio del ciclo (una línea nueva en el inbox
  corta la espera entre ciclos, también con backoff). El inbox se lee con un offset persistido que se guarda
  *antes* de ejecutar (como mucho una vez: tras un reinicio nunca se repite un "long HYPE"), junto a una huella
  de los últimos bytes leídos y los últimos ids ``[id]`` ejecutados. Si el fichero encoge o se reescribe (la
  huella no cuadra) no se relee desde el principio: sólo se ejecutan líneas con un ``[id]`` no visto. Sin
  ``desk.json`` válido (borrado o ilegible) el historial del inbox se ignora (salvo en un estado nuevo, sin
  ``outbox.log``). Una línea ``[id] texto`` usa ese id; si no, la mesa asigna uno (``c-000001``). Las
  respuestas van a ``outbox.log`` con el id del comando.
- Comandos interpretados por IA (``source="llm"``) que cambian algo quedan pendientes de ``si``/``no`` si
  ``llm_confirm_trades`` (5 min para confirmar).
- La operativa automática sólo abre en símbolos de ``settings.symbols`` sin posición, sin entrada pendiente
  y sin operación gestionada: nunca toca posiciones abiertas por comando (ni las cierra: cada operación sale
  por su propio SL/TP o por orden del usuario). La vela evaluada por (moneda, estrategia) se guarda en
  ``desk.json``: un reinicio dentro de la misma vela no repite la señal.
- Riesgo: el estado (halt, pausa, cooldown) se guarda en ``risk.json`` en cuanto cambia (un ``kill`` sobrevive
  a un crash inmediato). Los movimientos de saldo que no explican las ejecuciones ni el PnL no realizado
  (depósitos, retiradas, transferencias perp↔spot) se descuentan del máximo de equity y de la referencia
  diaria: no disparan un halt por drawdown.
- Salud: cada ciclo escribe ``heartbeat`` (hora) y ``health.json`` (racha de fallos, etapas que fallan y la
  última vez que se leyó bien la cuenta); ``python -m desk health`` falla si la mesa no ve el exchange.
- SIGINT/SIGTERM → termina el ciclo en curso, guarda el estado y sale; las órdenes SL/TP quedan en el exchange
  (en paper, en el estado guardado del broker). Un segundo Ctrl+C fuerza la salida.

Todas las horas salen del ``Clock`` inyectado.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import signal
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .clock import Clock
from .commands import HELP_TEXT, PARAM_DEFAULTS, CommandParser, describe
from .config import TUNABLE, Settings
from .engine import PENDING, ClosedTrade, ExecutionEngine, fmt_px, fmt_size, fmt_usd
from .exchange.base import Broker, BrokerError, MarketData, MarketDataError, round_price
from .exchange.intervals import interval_seconds
from .journal import Journal
from .liquidity import LiquidityScanner
from .models import (AccountState, Command, CommandAction, EntryMode, OrderStatus, RiskDecision, Side, Signal,
                     TradeIntent)
from .risk import (EVENT_COOLDOWN_END, EVENT_COOLDOWN_START, EVENT_FLATTEN, EVENT_HALT_DAILY, EVENT_HALT_DRAWDOWN,
                   EVENT_HALT_LIFTED, EVENT_NEW_DAY, LIMIT_MAX_DISTANCE_PCT, RiskManager)
from .strategy import Strategy, build_strategies

log = logging.getLogger(__name__)
# Avisos para el operador (ya salen por pantalla vía ``out``): logger aparte para poder filtrarlo en consola.
notice_log = logging.getLogger("desk.notice")

STATE_NAME = "desk"
RISK_STATE_NAME = "risk"
STATUS_NAME = "status"
HEARTBEAT_FILE = "heartbeat"
HEALTH_FILE = "health.json"
STATE_VERSION = 1

MAX_BACKOFF_S = 300.0
CONFIRM_TTL_S = 300.0
EQUITY_EVERY_S = 60.0
INBOX_ID_RE = re.compile(r"^\[([A-Za-z0-9_.:\-]{1,64})\]\s*(.*)$")
INBOX_MARK_RE = re.compile(r"\[[A-Za-z0-9_.:\-]{1,64}\]")
INBOX_FP_BYTES = 64          # huella del inbox: hash de los últimos bytes antes del offset
INBOX_IDS_KEPT = 500         # ids [id] ya ejecutados que se recuerdan (dedupe)
HEALTH_STAGES = ("cuenta", "fills")   # sin ellas la mesa no ve el exchange
# Movimiento de saldo no explicado por ejecuciones ni PnL no realizado: por debajo de esto cuenta como PnL
# (funding, redondeos); por encima, y con las posiciones cuadradas, es un depósito / retirada / transferencia.
FLOW_MIN_PCT = 0.5
FLOW_MIN_USD = 1.0

# Acciones de la IA que se ejecutan sin confirmar (no abren ni cierran nada, o sólo frenan).
_LLM_SAFE = frozenset({CommandAction.STATUS, CommandAction.HELP, CommandAction.PAUSE, CommandAction.AUTO_OFF})


def _hhmm(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class TradingDesk:
    def __init__(self, settings: Settings, market: MarketData, broker: Broker, clock: Clock,
                 parser: Any = None, journal: Optional[Journal] = None, *,
                 strategies: Optional[list[Strategy]] = None, scanner: Optional[LiquidityScanner] = None,
                 risk: Optional[RiskManager] = None, engine: Optional[ExecutionEngine] = None,
                 out: Optional[Callable[[str], None]] = None):
        self.settings = settings
        self.market = market
        self.broker = broker
        self.clock = clock
        self.journal = journal if journal is not None else Journal(settings.state_path, clock)
        self.parser = parser if parser is not None else CommandParser(settings)
        self.out = out
        self.risk = risk if risk is not None else RiskManager.from_dict(
            self.journal.load_json(RISK_STATE_NAME, None), settings, clock)
        self.engine = engine if engine is not None else ExecutionEngine(broker, market, settings, clock, self.journal)
        self.scanner = scanner if scanner is not None else LiquidityScanner(market, settings)
        self.strategies = strategies if strategies is not None else build_strategies(settings, clock)

        self._lock = threading.RLock()
        self._queue: deque[tuple[str, str, str]] = deque()      # (id, texto, origen)
        self._stop_requested = False
        self.cycle = 0
        self.error_streak = 0                 # ciclos seguidos con alguna etapa fallida
        self.stage_errors: dict[str, int] = {}
        self.last_errors: dict[str, str] = {}
        self._last_cycle_s = 0.0
        self._account: Optional[AccountState] = None
        self._account_dirty = False
        self._last_strategy_ts = -math.inf
        self._last_status_ts = -math.inf
        self._last_equity_ts = -math.inf
        self._last_broker_save_ts = -math.inf
        self._bar_seen: dict[tuple[str, str], float] = {}
        self._coin_errors: dict[str, str] = {}
        self.pending: Optional[tuple[Command, float]] = None   # comando IA pendiente de confirmar
        self._saved: dict[str, Any] = {}
        self.session = {"trades": 0, "wins": 0, "pnl": 0.0, "fees": 0.0}
        self.started_ts = clock.now()
        self.stage_ok_ts: dict[str, float] = {}    # última vez que cada etapa terminó bien (salud)
        self._close_failures_seen: dict[str, str] = {}
        # Detección de depósitos / retiradas: cuenta de referencia y lo que las ejecuciones explican desde ella.
        self._flow_base: Optional[tuple[float, float, dict[str, float], float]] = None
        self._flow_pnl = 0.0
        self._flow_sizes: dict[str, float] = {}

        state = self.journal.load_json(STATE_NAME, None)
        self.notices: list[str] = []              # avisos de arranque (se muestran con el banner)
        if not isinstance(state, dict):
            state = {}
            # Sin estado válido de la mesa (desk.json borrado o ilegible) el historial del inbox ya se ejecutó en
            # su día: se empieza al final. Sólo un estado nuevo (sin outbox.log) lee el inbox desde el principio.
            end = self._inbox_end() if self.journal.outbox_path.exists() else 0
            if end:
                self.notices.append(f"Sin estado de la mesa (desk.json ausente o ilegible): se ignoran los {end} bytes "
                                    "ya existentes de inbox.txt (no se re-ejecuta su historial).")
                log.warning("%s", self.notices[-1])
            state = {"inbox_offset": end}
        self.inbox_offset = int(state.get("inbox_offset") or 0)
        self.inbox_fp: Optional[str] = state.get("inbox_fp") if isinstance(state.get("inbox_fp"), str) else None
        if self.inbox_offset and self.inbox_fp is None and "inbox_fp" not in state:
            self.inbox_fp = self._inbox_fingerprint(self.inbox_offset)
        # Claves de líneas ya ejecutadas: "id:<id>" o, sin id, "txt:<hash del texto>" (dedupe del inbox).
        self.inbox_seen: deque[str] = deque((str(x) for x in state.get("inbox_seen") or []), maxlen=INBOX_IDS_KEPT)
        for key, bar in (state.get("bar_seen") or {}).items():
            coin, _, strat = str(key).partition("|")
            try:
                self._bar_seen[(coin, strat)] = float(bar)
            except (TypeError, ValueError):
                continue
        self.cmd_seq = int(state.get("cmd_seq") or 0)
        # DESK_AUTO=false / --no-auto siempre arrancan en OFF; si no, se respeta el último "auto on/off".
        self.auto_enabled = bool(settings.auto_enabled) and bool(state.get("auto_enabled", True))
        self.param_overrides: dict[str, float] = {}
        self._param_base = {k: getattr(settings, k) for k in TUNABLE}   # valores de arranque ('reset params')
        self.day_stats: dict[str, Any] = dict(state.get("day_stats") or {})
        restored = []
        for key, value in (state.get("param_overrides") or {}).items():
            try:
                k = settings.set_param(key, float(value))
            except (ValueError, TypeError) as e:
                log.warning("parámetro guardado %s=%r ignorado: %s", key, value, e)
                continue
            self.param_overrides[k] = getattr(settings, k)
            restored.append(f"{k}={getattr(settings, k)}")
        if restored:
            log.warning("parámetros cambiados por comando restaurados del estado: %s", ", ".join(restored))
        self.restored_params = restored

    # ================================================================== utilidades
    def _emit(self, text: str) -> None:
        if self.out is not None:
            try:
                self.out(text)
            except Exception:  # noqa: BLE001 - la salida por pantalla nunca tumba la mesa
                log.exception("error escribiendo en la salida")

    def notify(self, text: str, level: int = logging.INFO) -> None:
        """Aviso de la mesa: pantalla, ``outbox.log`` (id ``desk``) y log."""
        notice_log.log(level, "%s", text)
        self.journal.outbox("desk", text)
        self._emit(text)

    def _flush_engine(self, target: Optional[list[str]] = None) -> None:
        for m in self.engine.pop_messages():
            if target is not None:
                target.append(m)
            else:
                self.notify(m)

    def _next_id(self) -> str:
        self.cmd_seq += 1
        return f"c-{self.cmd_seq:06d}"

    def _data_age(self, coin: str) -> Optional[float]:
        try:
            ts = self.market.last_update_ts(coin)
        except Exception:  # noqa: BLE001 - dato informativo
            return None
        return None if ts is None else max(0.0, self.clock.now() - float(ts))

    # ================================================================== comandos
    def enqueue(self, text: str, cmd_id: Optional[str] = None, source: str = "cola") -> str:
        """Encola un comando para el próximo ciclo (seguro desde otros hilos). Devuelve su id."""
        with self._lock:
            cid = cmd_id or self._next_id()
        self._queue.append((cid, text, source))
        return cid

    def submit(self, text: str, source: str = "consola", cmd_id: Optional[str] = None) -> str:
        """Ejecuta un comando ya y devuelve la respuesta legible (también va a ``outbox.log``)."""
        with self._lock:
            cid = cmd_id or self._next_id()
            return self._process(cid, text, source)

    def _process(self, cmd_id: str, text: str, source: str) -> str:
        text = (text or "").strip()
        self.journal.event("command", id=cmd_id, text=text, source=source)
        try:
            reply = self._handle_text(cmd_id, text)
        except (MarketDataError, BrokerError) as e:
            log.warning("comando %s (%r): %s", cmd_id, text, e)
            reply = f"No se pudo completar '{text}': fallo de conexión o de datos ({e}). Inténtalo de nuevo."
        except Exception as e:  # noqa: BLE001 - un comando nunca tumba la mesa
            log.exception("error ejecutando el comando %s (%r)", cmd_id, text)
            reply = f"Error inesperado ejecutando '{text}': {type(e).__name__}: {e}"
        self.journal.outbox(cmd_id, reply)
        self.journal.event("reply", id=cmd_id, text=reply)
        try:
            self._save_desk_state()
        except OSError as e:
            log.error("no se pudo guardar el estado de la mesa: %s", e)
        self._save_risk_quiet()                  # pausa / reanuda / reset halt / kill: a disco ya
        return reply

    def _parse(self, text: str) -> Optional[Command]:
        p = self.parser
        if hasattr(p, "parse"):
            return p.parse(text)
        return p(text)

    def _handle_text(self, cmd_id: str, text: str) -> str:
        if not text:
            return "Comando vacío. Escribe 'ayuda' para ver ejemplos."
        cmd = self._parse(text)
        if cmd is None:
            return f"No entendí '{text}'. Escribe 'ayuda' para ver ejemplos."
        cmd.id = cmd_id
        if not cmd.raw:
            cmd.raw = text
        now = self.clock.now()
        if cmd.action is CommandAction.CONFIRM:
            if self.pending is None or now > self.pending[1]:
                self.pending = None
                return "No hay ningún comando pendiente de confirmar."
            pending, _ = self.pending
            self.pending = None
            self.journal.event("confirmed", id=cmd_id, command=pending.to_dict())
            return f"Confirmado ({describe(pending)}).\n" + self._execute(pending)
        if cmd.action is CommandAction.REJECT:
            if self.pending is None or now > self.pending[1]:
                self.pending = None
                return "No hay ningún comando pendiente que descartar."
            pending, _ = self.pending
            self.pending = None
            return f"Descartado: {describe(pending)}."
        if cmd.source == "llm" and self.settings.llm_confirm_trades and cmd.action not in _LLM_SAFE:
            replaced = f" (sustituye a '{describe(self.pending[0])}')" if self.pending else ""
            self.pending = (cmd, now + CONFIRM_TTL_S)
            self.journal.event("confirm_requested", id=cmd_id, command=cmd.to_dict())
            return (f"Interpreté con IA: {describe(cmd)}{replaced}. ¿Lo ejecuto? Responde 'si' para confirmar o "
                    f"'no' para descartar (caduca en {CONFIRM_TTL_S / 60:.0f} min).")
        return self._execute(cmd)

    def _execute(self, cmd: Command) -> str:
        a = cmd.action
        handler = {
            CommandAction.OPEN: self._cmd_open,
            CommandAction.CLOSE: self._cmd_close,
            CommandAction.CLOSE_ALL: self._cmd_close_all,
            CommandAction.CANCEL: self._cmd_cancel,
            CommandAction.SET_SL: self._cmd_set_level,
            CommandAction.SET_TP: self._cmd_set_level,
            CommandAction.PAUSE: lambda c: self._cap(self.risk.pause()),
            CommandAction.RESUME: lambda c: self._cap(self.risk.resume()),
            CommandAction.AUTO_ON: self._cmd_auto,
            CommandAction.AUTO_OFF: self._cmd_auto,
            CommandAction.STATUS: lambda c: self.status_report(),
            CommandAction.SET_PARAM: self._cmd_param,
            CommandAction.KILL: self._cmd_kill,
            CommandAction.RESET_HALT: self._cmd_reset_halt,
            CommandAction.HELP: lambda c: HELP_TEXT,
        }.get(a)
        if handler is None:
            return f"Acción no soportada: {a.value}"
        return handler(cmd)

    @staticmethod
    def _cap(text: str) -> str:
        return text[:1].upper() + text[1:] if text else text

    # ------------------------------------------------------------------ abrir
    def _cmd_open(self, cmd: Command) -> str:
        coin, side = cmd.coin, cmd.side
        if not coin or side is None:
            return "Falta la moneda o el lado (long/short)."
        try:
            self.market.meta(coin)
        except KeyError:
            return f"No conozco la moneda {coin} en este mercado."
        mode = cmd.entry_mode or EntryMode(self.settings.default_entry_mode)
        for what, px in (("el precio límite", cmd.price if mode is EntryMode.LIMIT else None),
                         ("el SL", cmd.sl_price), ("el TP", cmd.tp_price)):
            hint = self._thousands_hint(coin, px)
            if hint:
                return f"{coin} {side.value.upper()}: no se envió nada: {what} {hint}"
        units = float(cmd.units) if cmd.units else 1.0
        notes: list[str] = []
        zone = None
        if mode is EntryMode.ZONE:
            self.market.bid_ask(coin)          # sin precio no hay zona: mejor decir el motivo real
            zone = self.scanner.zone(coin, side)
            if zone is None:
                s = self.settings
                if s.zone_fallback == "market":
                    mode = EntryMode.MARKET
                    notes.append("sin zona de liquidez clara: entrada a mercado (DESK_ZONE_FALLBACK=market)")
                else:
                    return (f"{coin} {side.value.upper()}: no encuentro una zona de liquidez clara entre "
                            f"{s.zone_min_distance_pct:g}% y {s.zone_max_distance_pct:g}% del precio. Prueba "
                            f"'{side.value} {coin} a mercado' o con precio límite ('{side.value} {coin} a <precio>').")
        intent = TradeIntent(
            coin=coin, side=side, units=units, entry_mode=mode, origin="command",
            reason=f"comando {cmd.id}: {cmd.raw}", limit_px=cmd.price if mode is EntryMode.LIMIT else None,
            sl_price=cmd.sl_price, sl_pct=cmd.sl_pct, tp_price=cmd.tp_price, tp_pct=cmd.tp_pct, zone=zone)
        _, text = self._submit_intent(intent, cmd.id, notes)
        return text

    def _submit_intent(self, intent: TradeIntent, cmd_id: str, notes: Optional[list[str]] = None
                       ) -> tuple[bool, str]:
        """Riesgo → motor. Devuelve (abierta, texto para el operador)."""
        coin = intent.coin
        meta = self.market.meta(coin)
        bid, ask = self.market.bid_ask(coin)
        account = self.broker.account()
        decision = self.risk.evaluate(intent, account, meta, bid, ask, self._data_age(coin))
        self.journal.event("risk_decision", id=cmd_id, coin=coin, side=intent.side.value, origin=intent.origin,
                           approved=decision.approved, reason=decision.reason, warnings=decision.warnings,
                           size=decision.size, entry_px=decision.entry_px, sl=decision.sl_price,
                           tp=decision.tp_price, leverage=decision.leverage)
        if not decision.approved:
            return False, f"{coin} {intent.side.value.upper()} rechazada por riesgo: {decision.reason}"
        extra: list[str] = []
        try:
            order = self.engine.open(intent, decision, cmd_id=cmd_id)
        except BrokerError as e:
            if not self.engine.is_unconfirmed(coin):
                raise
            # La orden pudo llegar al exchange (timeout tras enviarla): el bracket queda registrado con el SL/TP
            # aprobados y reconcile la encuentra o la descarta. Repetir el comando duplicaría la exposición.
            self._account_dirty = True
            self._sync_fills_quiet(extra)
            lines = [self._describe_entry(intent, decision, None, account.equity),
                     f"ESTADO INCIERTO: el envío falló ({e}) pero la orden pudo llegar al exchange. NO la repitas: la "
                     "mesa la reconcilia en los próximos ciclos (si se ejecutó, la protege con el SL/TP de arriba; si "
                     "no aparece, la descarta y avisa)."]
            return True, "\n".join(lines + extra)
        if order.status is OrderStatus.REJECTED:
            self._flush_engine()
            return False, f"{coin} {intent.side.value.upper()}: el broker rechazó la entrada: {order.reason}"
        self._sync_fills_quiet(extra)            # entrada a mercado: SL/TP colocados ya, no en el próximo ciclo
        lines = [self._describe_entry(intent, decision, order, account.equity)]
        if order.status is OrderStatus.FILLED:
            lines.append(f"Ejecutada a mercado @ {fmt_px(order.avg_fill_px)}.")
        else:
            ttl = self.settings.entry_ttl_minutes
            lines.append(f"Orden límite en espera (caduca en {ttl:g} min); SL y TP se colocan al ejecutarse.")
        warns = list(notes or []) + list(decision.warnings)
        if warns:
            lines.append("Avisos: " + "; ".join(warns))
        lines.extend(extra)
        return True, "\n".join(lines)

    def _describe_entry(self, intent: TradeIntent, d: RiskDecision, order: Any, equity: float) -> str:
        z = intent.zone
        if intent.entry_mode is EntryMode.ZONE and z is not None:
            try:
                szd = self.market.meta(intent.coin).sz_decimals
                lo, hi = round_price(z.low, szd), round_price(z.high, szd)
            except (KeyError, MarketDataError, ValueError):
                lo, hi = z.low, z.high
            how = f"zona liq {fmt_px(lo)}-{fmt_px(hi)}"
        elif intent.entry_mode is EntryMode.LIMIT:
            how = "límite"
        else:
            how = "a mercado"
        text = (f"{intent.coin} {intent.side.value.upper()} {fmt_size(d.size)} @ {fmt_px(d.entry_px)} "
                f"({how}, ${d.notional:,.0f} notional, {d.leverage}x aislado)")
        if d.sl_price is not None:
            text += f" SL {fmt_px(d.sl_price)}"
        if d.tp_price is not None:
            text += f" TP {fmt_px(d.tp_price)}"
        if d.sl_price is not None and equity and equity > 0:
            loss = d.size * (abs(d.entry_px - d.sl_price) + self.risk.taker_rate * (d.entry_px + d.sl_price))
            text += f" | riesgo en el SL ≈ ${loss:,.2f} ({loss / equity * 100:.2f}% del equity)"
        return text

    # ------------------------------------------------------------------ cerrar / cancelar / niveles
    def _cmd_close(self, cmd: Command) -> str:
        coin = cmd.coin
        if not coin:
            return "¿Qué moneda cierro? Ejemplo: 'cierra HYPE' o 'cierra todo'."
        try:
            self.market.meta(coin)
        except KeyError:
            if coin not in self.engine.brackets:
                return f"No conozco la moneda {coin} en este mercado. Para cerrar todo: 'cierra todo'."
        except MarketDataError as e:            # sin metadatos se intenta cerrar igual
            log.warning("cierre de %s sin metadatos (%s)", coin, e)
        msgs: list[str] = []
        order = self.engine.close(coin, "close")
        self._flush_engine(msgs)
        if order is None:
            return "\n".join(msgs) if msgs else f"No hay posición ni entrada pendiente en {coin}."
        self._sync_fills_quiet(msgs)
        return "\n".join(msgs)

    def _cmd_close_all(self, cmd: Command) -> str:
        msgs: list[str] = []
        pending = [c for c, b in self.engine.brackets.items() if b.status == PENDING]
        orders = self.engine.close_all("close_all")
        failed = dict(self.engine.close_failures)
        self._flush_engine(msgs)
        self._sync_fills_quiet(msgs)
        canceled = [c for c in pending if c not in self.engine.brackets]
        if not orders and not pending and not failed:
            return "No había posiciones abiertas ni entradas pendientes."
        head = f"Cerrando todo: {len(orders)} posición(es) a mercado"
        if canceled:
            head += f", entradas pendientes canceladas ({', '.join(sorted(canceled))})"
        lines = [head + "."]
        if failed:
            lines.append("SIN CERRAR: " + "; ".join(f"{c} ({why})" for c, why in sorted(failed.items()))
                         + ". Se reintenta en la reconciliación; repite 'cierra todo' si hace falta.")
        return "\n".join(lines + msgs)

    def _cmd_cancel(self, cmd: Command) -> str:
        n = self.engine.cancel(cmd.coin)
        what = f"de {cmd.coin}" if cmd.coin else "de todas las monedas"
        msgs: list[str] = []
        self._flush_engine(msgs)
        if n == 0:
            return "\n".join([f"No había órdenes de entrada abiertas {what}. Los SL/TP de las posiciones no se "
                              "tocan."] + msgs)
        return "\n".join([f"{n} orden(es) de entrada canceladas {what}. Los SL/TP de las posiciones abiertas se "
                          "mantienen (para salir usa 'cierra <moneda>')."] + msgs)

    def _cmd_set_level(self, cmd: Command) -> str:
        coin = cmd.coin
        is_sl = cmd.action is CommandAction.SET_SL
        label = "SL" if is_sl else "TP"
        pct = cmd.sl_pct if is_sl else cmd.tp_pct
        px = cmd.price
        hint = self._thousands_hint(coin, px)
        if hint:
            return f"No se movió el {label} de {coin}: el precio {hint}"
        if px is None:
            if pct is None:
                return f"Indica el precio o el % del {label}. Ejemplo: '{'stop' if is_sl else 'tp'} {coin} 84.5'."
            ref = self.engine.reference_px(coin)
            if ref is None:
                return f"No hay posición abierta ni entrada pendiente en {coin}."
            side, entry = ref
            down = (side is Side.LONG) == is_sl          # SL de un long / TP de un short: por debajo
            px = entry * (1.0 - pct / 100.0) if down else entry * (1.0 + pct / 100.0)
        try:
            text = (self.engine.set_stop if is_sl else self.engine.set_take_profit)(coin, px)
        except ValueError as e:
            return f"No se movió el {label} de {coin}: {e}"
        msgs: list[str] = []
        self._flush_engine(msgs)
        return "\n".join([text] + msgs)

    # ------------------------------------------------------------------ control
    def _cmd_auto(self, cmd: Command) -> str:
        on = cmd.action is CommandAction.AUTO_ON
        self.auto_enabled = on
        self._save_desk_state()
        if not on:
            return ("Operativa automática DESACTIVADA. Las posiciones abiertas siguen con sus SL/TP; los comandos "
                    "siguen funcionando.")
        names = ", ".join(s.name for s in self.strategies) or "ninguna estrategia configurada"
        text = f"Operativa automática ACTIVADA ({names}) sobre {', '.join(self.settings.symbols)}."
        block = self.risk.block_reason("auto")
        if block:
            text += f" Ahora mismo no abrirá nada: {block}."
        return text

    def _cmd_param(self, cmd: Command) -> str:
        if cmd.param == PARAM_DEFAULTS:
            if not self.param_overrides:
                return "No había parámetros cambiados por comando: ya se usan los de arranque."
            restored = []
            for key in sorted(self.param_overrides):
                setattr(self.settings, key, self._param_base[key])
                restored.append(f"{key}={self._param_base[key]}")
            self.param_overrides = {}
            self._save_desk_state()
            self.journal.event("params_reset", params=restored)
            return f"Parámetros de arranque restaurados: {', '.join(restored)}."
        if cmd.param is None or cmd.value is None:
            return "Falta el parámetro o el valor. Ejemplo: 'apalancamiento 3', 'riesgo 0.5', 'unidad 250'."
        try:
            key = self.settings.set_param(cmd.param, float(cmd.value))
        except ValueError as e:
            ranges = ", ".join(f"{k} [{lo:g}-{hi:g}]" for k, (lo, hi) in TUNABLE.items())
            return f"No se cambió: {e}. Ajustables: {ranges}"
        value = getattr(self.settings, key)
        self.param_overrides[key] = value
        self._save_desk_state()
        self.journal.event("param_changed", param=key, value=value)
        note = ""
        if isinstance(value, int) and value != cmd.value:
            note = f" (se redondeó {cmd.value:g} a {value}: el parámetro es entero)"
        return f"{key} = {value}{note}. Se aplica a las próximas operaciones y se recuerda tras reiniciar."

    def _cmd_kill(self, cmd: Command) -> str:
        self.risk.halt("kill", "pánico manual", flatten=True)
        self._save_risk_quiet()                  # antes de tocar el exchange: el halt sobrevive a un crash
        msgs: list[str] = []
        problems: list[str] = []
        canceled = 0
        try:
            canceled = self.engine.cancel(None)
        except (MarketDataError, BrokerError) as e:
            problems.append(f"no se pudieron cancelar las entradas ({e})")
        orders = []
        try:
            orders = self.engine.close_all("kill")
        except (MarketDataError, BrokerError) as e:
            problems.append(f"no se pudo leer la cuenta para cerrar ({e})")
        failed = dict(self.engine.close_failures)
        self._flush_engine(msgs)
        self._sync_fills_quiet(msgs)
        self._save_desk_state()
        lines = [f"PÁNICO: {len(orders)} posición(es) cerrándose a mercado, {canceled} entrada(s) canceladas. "
                 "Mesa DETENIDA: no se abre nada (ni comandos ni automático) hasta 'reset halt'."]
        if failed:
            problems.append("sin cerrar todavía: " + "; ".join(f"{c} ({why})" for c, why in sorted(failed.items())))
        if problems:
            lines.append("ATENCIÓN: " + "; ".join(problems) + ". El halt queda activo y el cierre se reintenta en "
                         "cada ciclo mientras quede exposición.")
        return "\n".join(lines + msgs)

    def _cmd_reset_halt(self, cmd: Command) -> str:
        # Equity fresco antes de reiniciar las referencias (máximo y día) del riesgo.
        try:
            for ev in self.risk.update(self.broker.account()):
                log.info("riesgo (antes de reset halt): %s", ev)
        except (MarketDataError, BrokerError) as e:
            log.warning("reset halt sin equity actualizado: %s", e)
        return self._cap(self.risk.reset_halt())

    # ================================================================== ciclo
    def _stage(self, name: str, fn: Callable[[], Any], errors: list[str]) -> Any:
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001 - ninguna etapa tumba la mesa
            errors.append(name)
            n = self.stage_errors[name] = self.stage_errors.get(name, 0) + 1
            msg = f"{type(e).__name__}: {e}"
            if n == 1 or msg != self.last_errors.get(name):
                log.error("etapa '%s' falló (%d seguidas): %s", name, n, msg,
                          exc_info=not isinstance(e, (MarketDataError, BrokerError)))
            self.last_errors[name] = msg
            return None
        self.stage_ok_ts[name] = self.clock.now()
        if self.stage_errors.get(name):
            log.warning("etapa '%s' recuperada tras %d fallo(s)", name, self.stage_errors[name])
            self.stage_errors[name] = 0
            self.last_errors.pop(name, None)
        return result

    def run_cycle(self) -> bool:
        """Un ciclo completo. Devuelve True si ninguna etapa falló."""
        with self._lock:
            t0 = self.clock.now()
            self.cycle += 1
            errors: list[str] = []
            self._account = None
            self._stage("comandos", self._stage_commands, errors)
            self._stage("broker", self.broker.step, errors)
            self._stage("fills", self._sync_fills, errors)
            self._account = self._stage("cuenta", self.broker.account, errors)
            if self._account is not None:
                self._stage("riesgo", self._stage_risk, errors)
                self._stage("reconciliación", self._stage_reconcile, errors)
            self._stage("estrategias", self._stage_strategies, errors)
            self._stage("guardar", self.save_state, errors)
            self._stage("estado", self._stage_status, errors)
            self.error_streak = self.error_streak + 1 if errors else 0
            if errors and self.error_streak in (1, 3, 10) or (errors and self.error_streak % 50 == 0):
                log.warning("ciclo %d con fallos en: %s (racha %d; próxima espera %.0f s)", self.cycle,
                            ", ".join(errors), self.error_streak, self.next_delay())
            self._last_cycle_s = max(self.clock.now() - t0, 0.0)
            self._write_health(errors)
            return not errors

    def health(self) -> dict[str, Any]:
        """Salud de la mesa (lo que se escribe en ``health.json``): racha de ciclos con fallos, etapas que fallan con
        su último error y la última vez que cada etapa crítica (``HEALTH_STAGES``) terminó bien."""
        failing = sorted(k for k, v in self.stage_errors.items() if v)
        return {"ts": self.clock.now(), "time": self.clock.iso(), "cycle": self.cycle, "started_ts": self.started_ts,
                "error_streak": self.error_streak, "failing": failing,
                "errors": {k: self.last_errors[k] for k in failing if k in self.last_errors},
                "last_ok": {k: self.stage_ok_ts.get(k) for k in HEALTH_STAGES},
                "next_delay_s": round(self.next_delay(), 1)}

    def _write_health(self, errors: list[str]) -> None:
        """``health.json`` en cada ciclo; ``heartbeat`` sólo si el ciclo leyó la cuenta y los fills: un latido viejo
        significa "la mesa no ve el exchange" (o está parada), no sólo "el proceso sigue vivo"."""
        try:
            self.journal.touch(HEALTH_FILE, json.dumps(self.health(), ensure_ascii=False) + "\n")
        except (TypeError, ValueError) as e:      # touch ya registra los errores de disco
            log.error("no se pudo escribir %s: %s", HEALTH_FILE, e)
        if not any(stage in errors for stage in HEALTH_STAGES):
            self.journal.touch(HEARTBEAT_FILE, f"{self.clock.now()}\n")

    def next_delay(self) -> float:
        """Espera hasta el próximo ciclo: ``loop_seconds`` o backoff exponencial si hay fallos seguidos."""
        base = max(float(self.settings.loop_seconds), 0.1)
        if self.error_streak <= 0:
            return max(base - self._last_cycle_s, 0.0)
        return min(base * (2 ** min(self.error_streak, 20)), MAX_BACKOFF_S)

    # ------------------------------------------------------------------ etapas
    def _stage_commands(self) -> None:
        # El inbox se lee antes de vaciar la cola: si falla (disco, permisos) los comandos de la cola no se pierden.
        inbox_error: Optional[Exception] = None
        try:
            inbox = self._read_inbox()
        except Exception as e:  # noqa: BLE001 - se re-lanza tras procesar la cola
            inbox, inbox_error = [], e
        items: list[tuple[str, str, str]] = []
        while self._queue:
            items.append(self._queue.popleft())
        items.extend(inbox)
        for cid, text, source in items:
            reply = self._process(cid, text, source)
            self._emit(f"» [{cid}] {text}\n{reply}")
        if items:
            self._account_dirty = True
        if inbox_error is not None:
            raise inbox_error

    # ------------------------------------------------------------------ inbox
    def _inbox_stat(self) -> Optional[tuple[int, int]]:
        try:
            st = self.journal.inbox_path.stat()
        except OSError:
            return None
        return st.st_size, st.st_mtime_ns

    def _inbox_end(self) -> int:
        """Posición tras la última línea completa de inbox.txt (0 si no existe)."""
        try:
            data = self.journal.inbox_path.read_bytes()
        except OSError:
            return 0
        return data.rfind(b"\n") + 1

    def _inbox_fingerprint(self, offset: int) -> Optional[str]:
        """Huella de los ``INBOX_FP_BYTES`` bytes anteriores a ``offset`` (detecta un fichero reescrito)."""
        if offset <= 0:
            return None
        try:
            with open(self.journal.inbox_path, "rb") as f:
                start = max(offset - INBOX_FP_BYTES, 0)
                f.seek(start)
                data = f.read(offset - start)
        except OSError:
            return None
        if len(data) != offset - start:
            return None
        return hashlib.sha1(data).hexdigest()[:16]

    def _read_inbox(self) -> list[tuple[str, str, str]]:
        path = self.journal.inbox_path
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return []
        rewritten = size < self.inbox_offset
        if not rewritten and self.inbox_offset and self.inbox_fp is not None:
            rewritten = self._inbox_fingerprint(self.inbox_offset) != self.inbox_fp
        if not rewritten and size == self.inbox_offset:
            return []
        start = 0 if rewritten else self.inbox_offset
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(size - start)
        end = data.rfind(b"\n")
        if end < 0 and not rewritten:
            return []                            # línea a medio escribir: se espera al salto de línea
        chunk = data[:end + 1]
        previous = (self.inbox_offset, self.inbox_fp, list(self.inbox_seen))
        self.inbox_offset = start + len(chunk)
        self.inbox_fp = self._inbox_fingerprint(self.inbox_offset)
        out: list[tuple[str, str, str]] = []
        skipped = 0
        seen = set(self.inbox_seen)
        for raw in chunk.decode("utf-8", "replace").splitlines():
            for cid, text in self._split_inbox_line(raw):
                key = f"id:{cid}" if cid is not None else "txt:" + hashlib.sha1(text.encode()).hexdigest()[:16]
                # Un [id] ya ejecutado nunca se repite. Sin id, sólo en un fichero reescrito se descarta un texto ya
                # ejecutado (en uso normal repetir "estado" es legítimo).
                if key in seen and (cid is not None or rewritten):
                    skipped += 1
                    continue
                seen.add(key)
                self.inbox_seen.append(key)
                out.append((cid if cid is not None else self._next_id(), text, "inbox"))
        if rewritten:
            msg = (f"inbox.txt cambió por debajo de la mesa (encogió o se reescribió): no se relee su historial; "
                   f"se ejecutan {len(out)} comando(s) nuevos y se ignoran {skipped} línea(s) ya ejecutadas. No borres "
                   "ni edites líneas del inbox: para anular algo envía otro comando (p. ej. 'cancela HYPE').")
            log.warning("%s", msg)
            self.notify(msg, logging.WARNING)
        elif skipped:
            log.warning("inbox: %d línea(s) con un [id] ya ejecutado se ignoran", skipped)
        # Offset (y huella / ids) guardados ANTES de ejecutar: un comando del inbox se ejecuta como mucho una vez.
        try:
            self._save_desk_state()
        except OSError:
            self.inbox_offset, self.inbox_fp, keys = previous   # sin offset guardado no se ejecuta: se reintenta
            self.inbox_seen = deque(keys, maxlen=INBOX_IDS_KEPT)
            raise
        return out

    def _split_inbox_line(self, raw: str) -> list[tuple[Optional[str], str]]:
        """Línea del inbox → [(id o None, texto)]. Ignora vacías y comentarios. Si un ``[id]`` aparece a mitad
        de línea (``send`` escribió detrás de una línea sin salto final) se separan los dos comandos."""
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("#"):
            return []
        marks = [m.start() for m in INBOX_MARK_RE.finditer(line) if m.start() > 0 and INBOX_ID_RE.match(line[m.start():])]
        parts = [line]
        if marks:
            cuts = [0] + marks + [len(line)]
            parts = [line[a:b].strip() for a, b in zip(cuts, cuts[1:]) if line[a:b].strip()]
            log.warning("inbox: línea con %d comandos pegados (falta un salto de línea): %r", len(parts), line[:200])
        out: list[tuple[Optional[str], str]] = []
        for part in parts:
            m = INBOX_ID_RE.match(part)
            out.append((m.group(1), m.group(2).strip()) if m else (None, part))
        return out

    def _sync_fills(self, target: Optional[list[str]] = None) -> list[ClosedTrade]:
        fills = self.broker.poll_fills()
        if not fills:
            self._flush_engine(target)
            return []
        for f in fills:                          # lo que explican las ejecuciones (detección de depósitos/retiradas)
            self._flow_pnl += f.realized_pnl - f.fee
            self._flow_sizes[f.coin] = self._flow_sizes.get(f.coin, 0.0) + (f.size if f.is_buy else -f.size)
        closed = self.engine.on_fills(fills)
        self._flush_engine(target)
        self._record_closed(closed, target)
        if getattr(self.broker, "fills_cursor_ts", None) is not None:
            # Live: el cursor de fills a disco junto al motor (un crash no re-entrega lo ya aplicado).
            try:
                self._save_desk_state()
            except OSError as e:
                log.error("no se pudo guardar el cursor de fills: %s", e)
        return closed

    def _sync_fills_quiet(self, target: list[str]) -> None:
        """Fills tras una orden ya aceptada: un fallo al leerlos no convierte la respuesta en 'inténtalo de nuevo'
        (la orden ya está en el exchange); se avisa y el próximo ciclo lo completa."""
        try:
            self._sync_fills(target)
        except (MarketDataError, BrokerError) as e:
            log.warning("fills tras la orden no disponibles: %s", e)
            target.append(f"Orden aceptada, pero sus ejecuciones aún no se pudieron leer ({e}): el SL/TP y el "
                          "estado se completan en el próximo ciclo. No repitas el comando.")

    def _thousands_hint(self, coin: Optional[str], px: Optional[float]) -> Optional[str]:
        """Texto de aviso si ``px`` parece un precio con punto de miles ("83.825" con BTC a ~83,800): leído como
        decimal está a >99 % del mercado y ×1000 cae cerca del precio. None si no hay ambigüedad."""
        if not coin or px is None or not isinstance(px, (int, float)) or not math.isfinite(px) or px <= 0:
            return None
        try:
            mid = self.market.mid(coin)
        except (MarketDataError, KeyError):
            return None
        if not mid or mid <= 0 or px * 100.0 >= mid:
            return None
        k = px * 1000.0
        if abs(k / mid - 1.0) * 100.0 > LIMIT_MAX_DISTANCE_PCT:
            return None
        return (f"{fmt_px(px)} parece un precio con punto de miles ({coin} cotiza a {fmt_px(mid)}): ¿querías decir "
                f"{k:.10g}? Escríbelo sin separador de miles (p. ej. {k:.10g}) o con decimales explícitos.")

    def _detect_cash_flow(self, account: AccountState) -> None:
        """Depósitos / retiradas / transferencias: la parte del cambio de equity que no explican las ejecuciones
        (realizado − comisiones) ni el PnL no realizado, con las posiciones cuadradas, se descuenta del máximo de
        equity y de la referencia diaria del riesgo (no es una pérdida ni una ganancia de la operativa).

        Conservador: si las posiciones no cuadran con los fills recibidos (un fill aún no ha llegado) no se
        clasifica nada; por debajo de ``FLOW_MIN_PCT`` cuenta como PnL (funding, redondeos); en live se descuenta
        además el funding cobrado/pagado si el broker lo sabe (``funding_since``). Una *salida* de saldo sólo se
        clasifica si no hubo ejecuciones entre las dos lecturas: con fills (p. ej. una liquidación que se lleva
        más margen que su ``closedPnl``) cuenta como pérdida; ante la duda, que salte el halt."""
        eq = float(account.equity) if account.equity is not None else math.nan
        live = {c: p for c, p in account.positions.items() if p.size != 0}
        upnl = sum(float(p.unrealized_pnl or 0.0) for p in live.values())
        sizes = {c: float(p.size) for c, p in live.items()}
        base, self._flow_base = self._flow_base, (eq, upnl, sizes, self.clock.now())
        pnl, moved = self._flow_pnl, self._flow_sizes
        self._flow_pnl, self._flow_sizes = 0.0, {}
        if base is None or not math.isfinite(eq) or not math.isfinite(base[0]):
            return
        b_eq, b_upnl, b_sizes, b_ts = base
        for c in set(b_sizes) | set(sizes) | set(moved):
            expected = b_sizes.get(c, 0.0) + moved.get(c, 0.0)
            if abs(expected - sizes.get(c, 0.0)) > 1e-6 * max(1.0, abs(expected)):
                return                           # hay ejecuciones que aún no han llegado: no se clasifica
        unexplained = (eq - b_eq) - (upnl - b_upnl) - pnl
        threshold = max(FLOW_MIN_USD, abs(b_eq) * FLOW_MIN_PCT / 100.0)
        if abs(unexplained) < threshold:
            return
        if unexplained < 0 and moved:
            log.warning("caída de equity no explicada por los fills (%+.2f USD) en un intervalo con ejecuciones: "
                        "cuenta como pérdida", unexplained)
            return
        funding_since = getattr(self.broker, "funding_since", None)
        if callable(funding_since):
            try:
                unexplained -= float(funding_since(b_ts))
            except (BrokerError, MarketDataError, TypeError, ValueError) as e:
                log.warning("funding no disponible (%s): el movimiento de saldo cuenta como PnL", e)
                return
            if abs(unexplained) < threshold:
                return
        self.risk.apply_cash_flow(unexplained)
        self.journal.event("cash_flow", amount=unexplained, equity=eq, previous_equity=b_eq)
        what = "depósito / transferencia entrante" if unexplained > 0 else "retirada / transferencia saliente"
        self.notify(f"Movimiento de saldo ajeno a la operativa: {fmt_usd(unexplained)} ({what}). No cuenta como PnL: "
                    f"máximo de equity y referencia del día ajustados (máximo {fmt_usd(self.risk.high_water or 0, sign=False)}).",
                    logging.WARNING)

    def _record_closed(self, trades: list[ClosedTrade], target: Optional[list[str]] = None) -> None:
        today = self.clock.utc_day()
        if self.day_stats.get("day") != today:
            self.day_stats = {"day": today, "trades": 0, "wins": 0, "pnl": 0.0}
        for t in trades:
            self.session["trades"] += 1
            self.session["pnl"] += t.pnl
            self.session["fees"] += t.fees
            self.day_stats["trades"] += 1
            self.day_stats["pnl"] += t.pnl
            if t.pnl > 0:
                self.session["wins"] += 1
                self.day_stats["wins"] += 1
            for ev in self.risk.record_trade_result(t.pnl):
                if ev == EVENT_COOLDOWN_START:
                    msg = (f"Cooldown: {self.risk.cooldown_detail}; sin entradas automáticas hasta "
                           f"{_hhmm(self.risk.cooldown_until)} (los comandos siguen permitidos)")
                    if target is not None:
                        target.append(msg)
                    else:
                        self.notify(msg, logging.WARNING)

    def _stage_risk(self) -> None:
        account = self._account
        self._detect_cash_flow(account)
        events = self.risk.update(account)
        if events:
            self._save_risk_quiet()
        halted_now = False
        for ev in events:
            if ev == EVENT_NEW_DAY:
                self.notify(f"Nuevo día UTC {self.risk.day}: referencia de PnL diario = "
                            f"{fmt_usd(self.risk.day_start_equity, sign=False)}")
            elif ev == EVENT_HALT_DAILY:
                halted_now = True
                tail = "; cerrando todo" if self.risk.flatten_required else "; las posiciones abiertas siguen con su SL/TP"
                self.notify(f"HALT por pérdida diaria: {self.risk.halt_detail}. Sin entradas nuevas hasta el próximo "
                            f"día UTC (o 'reset halt'){tail}.", logging.WARNING)
            elif ev == EVENT_HALT_DRAWDOWN:
                halted_now = True
                self.notify(f"HALT por drawdown máximo: {self.risk.halt_detail}. Cerrando todo; sólo se reanuda con "
                            "'reset halt'.", logging.WARNING)
            elif ev == EVENT_HALT_LIFTED:
                self.notify("Día nuevo: el halt por pérdida diaria se levantó; la mesa vuelve a operar.")
            elif ev == EVENT_COOLDOWN_END:
                self.notify("Fin del cooldown: la operativa automática puede volver a abrir.")
        acted = False
        if halted_now:
            n = self.engine.cancel(None)
            acted = acted or n > 0
            if n:
                self.notify(f"{n} entrada(s) pendientes canceladas por el halt.")
        if EVENT_FLATTEN in events:
            reason = "kill" if self.risk.halt_reason == "kill" else "flatten"
            had_pending = any(b.status == PENDING for b in self.engine.brackets.values())
            orders = self.engine.close_all(reason)
            failed = dict(self.engine.close_failures)
            if orders or had_pending:
                acted = True
                self.notify(f"Riesgo ({self.risk.halt_reason}): cerrando {len(orders)} posición(es) a mercado.",
                            logging.WARNING)
            if failed and failed != self._close_failures_seen:
                self.notify("Riesgo: SIN CERRAR todavía: " + "; ".join(f"{c} ({why})" for c, why in sorted(failed.items()))
                            + ". Se reintenta en cada ciclo.", logging.WARNING)
            self._close_failures_seen = failed
            self._flush_engine()
            self._sync_fills()
        if acted:
            self._account = self.broker.account()

    def _stage_reconcile(self) -> None:
        closed = self.engine.reconcile(self._account)
        self._flush_engine()
        self._record_closed(closed)

    def _stage_strategies(self) -> None:
        now = self.clock.now()
        if now - self._last_strategy_ts < self.settings.strategy_seconds:
            return
        self._last_strategy_ts = now
        if not self.auto_enabled or not self.strategies:
            return
        block = self.risk.block_reason("auto")
        if block:
            log.debug("auto en espera: %s", block)
            return
        account = self.broker.account()
        busy = {c for c, p in account.positions.items() if p.size != 0}
        busy |= {o.coin for o in account.open_orders if o.is_open and not o.reduce_only}
        busy |= set(self.engine.brackets)
        if len(busy) >= self.settings.max_open_positions:
            return
        for coin in self.settings.symbols:
            if coin in busy:
                continue
            try:
                opened = self._auto_coin(coin, now)
            except (MarketDataError, BrokerError, KeyError, ValueError) as e:
                msg = f"{type(e).__name__}: {e}"
                if self._coin_errors.get(coin) != msg:
                    log.warning("auto %s: %s", coin, msg)
                self._coin_errors[coin] = msg
                continue
            self._coin_errors.pop(coin, None)
            if opened:
                busy.add(coin)
                if len(busy) >= self.settings.max_open_positions:
                    break

    def _auto_coin(self, coin: str, now: float) -> bool:
        """Evalúa las estrategias de ``coin`` (una vez por vela cerrada). True si abrió una operación."""
        due = []
        for strat in self.strategies:
            sec = interval_seconds(strat.interval)
            bar = math.floor(now / sec) * sec            # apertura de la vela en curso
            if self._bar_seen.get((coin, strat.name)) != bar:
                due.append((strat, bar))
        if not due:
            return False
        mid = self.market.mid(coin)
        zones = self.scanner.zones(coin)
        candles_cache: dict[str, Any] = {}
        for strat, bar in due:
            key = f"{strat.interval}:{strat.lookback}"
            if key not in candles_cache:
                candles_cache[key] = self.market.candles(coin, strat.interval, strat.lookback)
            signal = strat.evaluate(coin, candles_cache[key], mid, zones)
            self._bar_seen[(coin, strat.name)] = bar
            if signal is None:
                continue
            if self._open_signal(signal, zones):
                return True
        return False

    def _open_signal(self, sig: Signal, zones: dict) -> bool:
        mode = sig.entry_mode
        intent = TradeIntent(
            coin=sig.coin, side=sig.side, units=self.settings.auto_units, entry_mode=mode,
            origin=f"auto:{sig.strategy}", reason=sig.reason,
            limit_px=sig.entry_px if mode is not EntryMode.MARKET else None,
            sl_price=sig.sl_price, tp_price=sig.tp_price,
            zone=zones.get(sig.side) if mode is EntryMode.ZONE else None)
        self.journal.event("signal", coin=sig.coin, side=sig.side.value, strategy=sig.strategy, reason=sig.reason,
                           entry_px=sig.entry_px, sl=sig.sl_price, tp=sig.tp_price, confidence=sig.confidence)
        ok, text = self._submit_intent(intent, f"auto-{sig.strategy}")
        if ok:
            self._account_dirty = True
            self.notify(f"AUTO [{sig.strategy}] {text}\nMotivo: {sig.reason}")
        else:
            log.info("AUTO [%s] señal %s %s descartada: %s", sig.strategy, sig.coin, sig.side.value, text)
        return ok

    def _stage_status(self) -> None:
        now = self.clock.now()
        if self._account_dirty:                  # el automático abrió algo después de leer la cuenta
            self._account_dirty = False
            try:
                self._account = self.broker.account()
            except (MarketDataError, BrokerError) as e:
                log.warning("estado: cuenta no releída (%s); se usa la lectura del ciclo", e)
        acct = self._account
        if acct is not None and now - self._last_equity_ts >= EQUITY_EVERY_S:
            self._last_equity_ts = now
            self.journal.equity(now, acct.equity, acct.available,
                                sum(1 for p in acct.positions.values() if p.size != 0))
        if now - self._last_status_ts >= self.settings.status_every_seconds:
            self._last_status_ts = now
            if acct is None:
                # Sin cuenta no hay informe, pero status.json no se queda congelado con el último informe sano.
                err = self.last_errors.get("cuenta", "sin lectura de la cuenta")
                line = (f"SIN DATOS DE LA CUENTA ({err}) | ciclos seguidos con fallos {self.error_streak + 1} | "
                        f"halt {self.risk.active_halt() or 'no'}")
                self.notify(line, logging.WARNING)
                self.journal.save_json(STATUS_NAME, {"ts": now, "time": self.clock.iso(), "line": line,
                                                     "report": line, "account_ok": False})
                return
            line = self.status_line(acct)
            self.notify(line)
            self.write_status_snapshot(acct, line)

    # ================================================================== estado
    def status_line(self, account: Optional[AccountState] = None) -> str:
        """``equity | PnL día | posiciones | auto ON/OFF | halt``."""
        acct = account if account is not None else self.broker.account()
        eq = acct.equity
        start = self.risk.day_start_equity
        day_pnl = eq - start if start else 0.0
        day_pct = day_pnl / start * 100.0 if start else 0.0
        pos = [p for p in acct.positions.values() if p.size != 0]
        pos_txt = ", ".join(f"{p.coin} {'L' if p.size > 0 else 'S'} {fmt_size(abs(p.size))} {fmt_usd(p.unrealized_pnl)}"
                            for p in sorted(pos, key=lambda p: p.coin))
        auto = "ON" if self.auto_enabled else "OFF"
        if self.risk.paused:
            auto += " (pausa)"
        elif self.auto_enabled and self.risk.in_cooldown:
            auto += " (cooldown)"
        halt = self.risk.active_halt() or "no"
        return (f"equity {fmt_usd(eq, sign=False)} | PnL día {fmt_usd(day_pnl)} "
                f"({day_pct:+.2f}%) | posiciones {len(pos)}{': ' + pos_txt if pos else ''} | auto {auto} | halt {halt}")

    def status_report(self, account: Optional[AccountState] = None) -> str:
        acct = account if account is not None else self.broker.account()
        s = self.settings
        eq = acct.equity
        start = self.risk.day_start_equity
        day_pnl = eq - start if start else 0.0
        day_pct = day_pnl / start * 100.0 if start else 0.0
        lines = [f"Estado de la mesa — {self.clock.iso()} UTC (modo {s.mode}, datos {s.data_source})",
                 f"Equity {fmt_usd(eq, sign=False)} | disponible {fmt_usd(acct.available, sign=False)} | PnL día "
                 f"{fmt_usd(day_pnl)} ({day_pct:+.2f}%) | drawdown {self.risk.drawdown_pct(eq):.2f}% "
                 f"(límites: día {s.daily_loss_limit_pct:g}%, máx. {s.max_drawdown_pct:g}%)"]
        names = ", ".join(st.name for st in self.strategies) or "sin estrategias"
        lines.append(f"Automático {'ON' if self.auto_enabled else 'OFF'} ({names}) sobre {', '.join(s.symbols)} | "
                     f"{self.risk.describe()}")
        lines.append(f"Dimensionamiento: 1 unidad = ${s.unit_margin_usd:,.0f} de margen x {s.leverage}x | riesgo "
                     f"máx. {s.risk_per_trade_pct:g}% por operación | SL por defecto {s.default_sl_pct:g}% | "
                     f"TP {s.default_tp_r:g}R")
        pos = [p for p in acct.positions.values() if p.size != 0]
        lines.append(f"Posiciones ({len(pos)}):" if pos else "Posiciones: ninguna")
        for p in sorted(pos, key=lambda p: p.coin):
            br = self.engine.brackets.get(p.coin)
            origin = "sin gestionar"
            if br is not None:
                origin = {"command": "comando", "adopted": "adoptada"}.get(br.origin, br.origin)
            lines.append(f"  {p.coin} {p.side.value.upper()} {fmt_size(abs(p.size))} @ {fmt_px(p.entry_px)} → "
                         f"{fmt_px(p.mark_px)} | PnL {fmt_usd(p.unrealized_pnl)} ({p.pnl_pct():+.1f}% s/margen) | "
                         f"{p.leverage}x | SL {fmt_px(br.sl if br else None)} TP {fmt_px(br.tp if br else None)} | "
                         f"{origin}")
        pend = [b for b in self.engine.brackets.values() if b.status == PENDING]
        if pend:
            lines.append(f"Entradas pendientes ({len(pend)}):")
            for b in sorted(pend, key=lambda b: b.coin):
                exp = f", caduca {_hhmm(b.expires_ts)}" if b.expires_ts else ""
                origin = {"command": "comando"}.get(b.origin, b.origin)
                lines.append(f"  {b.coin} {b.side.value.upper()} {fmt_size(b.planned_size)} @ {fmt_px(b.entry_px)} "
                             f"(SL {fmt_px(b.sl)} TP {fmt_px(b.tp)}{exp}) | {origin}")
        orders = [o for o in acct.open_orders if o.is_open]
        if orders:
            lines.append(f"Órdenes abiertas ({len(orders)}):")
            for o in sorted(orders, key=lambda o: (o.coin, o.kind.value)):
                px = o.trigger_px if o.trigger_px else o.price
                lines.append(f"  {o.coin} {'compra' if o.is_buy else 'venta'} {o.kind.value} {fmt_size(o.remaining)} "
                             f"@ {fmt_px(px)}{' reduce-only' if o.reduce_only else ''}{' [' + o.tag + ']' if o.tag else ''}")
        today = self.clock.utc_day()
        ds = self.day_stats if self.day_stats.get("day") == today else {"trades": 0, "wins": 0, "pnl": 0.0}
        lines.append(f"Hoy: {ds['trades']} operación(es) cerrada(s), {ds['wins']} ganadora(s), PnL neto "
                     f"{fmt_usd(ds['pnl'])}")
        if self.pending is not None and self.clock.now() <= self.pending[1]:
            lines.append(f"Pendiente de confirmar: {describe(self.pending[0])} (responde 'si' o 'no')")
        if self.param_overrides:
            lines.append("Parámetros cambiados por comando: "
                         + ", ".join(f"{k}={v}" for k, v in sorted(self.param_overrides.items())))
        if self.stage_errors and any(self.stage_errors.values()):
            errs = ", ".join(f"{k} ({v})" for k, v in self.stage_errors.items() if v)
            lines.append(f"Etapas con fallos recientes: {errs}")
        return "\n".join(lines)

    def write_status_snapshot(self, account: Optional[AccountState] = None, line: Optional[str] = None) -> None:
        """Guarda ``status.json`` (lo lee ``python -m desk status``)."""
        try:
            report = self.status_report(account)
            line = line or self.status_line(account)
        except (MarketDataError, BrokerError) as e:
            report, line = f"(estado no disponible: {e})", line or ""
        self.journal.save_json(STATUS_NAME, {"ts": self.clock.now(), "time": self.clock.iso(), "line": line,
                                             "report": report})

    def summary(self) -> str:
        """Resumen de la sesión (para el final de una simulación)."""
        s = self.session
        acct = self._account
        try:
            acct = self.broker.account()
        except (MarketDataError, BrokerError):
            pass
        eq = f"{fmt_usd(acct.equity, sign=False)}" if acct is not None else "-"
        win = f"{s['wins'] / s['trades'] * 100:.0f}%" if s["trades"] else "-"
        return (f"Sesión: {self.cycle} ciclos | operaciones cerradas {s['trades']} (aciertos {win}) | PnL neto "
                f"{fmt_usd(s['pnl'])} | comisiones {fmt_usd(s['fees'], sign=False)} | equity {eq}")

    # ------------------------------------------------------------------ persistencia
    def _desk_state(self) -> dict[str, Any]:
        data = {
            "version": STATE_VERSION,
            "inbox_offset": self.inbox_offset,
            "inbox_fp": self.inbox_fp,
            "inbox_seen": list(self.inbox_seen),
            "cmd_seq": self.cmd_seq,
            "auto_enabled": self.auto_enabled,
            "param_overrides": dict(self.param_overrides),
            "day_stats": dict(self.day_stats),
            "bar_seen": {f"{c}|{st}": bar for (c, st), bar in sorted(self._bar_seen.items())},
        }
        cursor = getattr(self.broker, "fills_cursor_ts", None)
        if isinstance(cursor, (int, float)):
            data["fills_cursor_ts"] = cursor
            seen = getattr(self.broker, "fills_seen_keys", None)
            if isinstance(seen, list):
                data["fills_seen"] = seen          # fills ya entregados en el ms del cursor (no se re-entregan)
        return data

    def _save_if_changed(self, name: str, data: dict[str, Any], compare: Optional[dict[str, Any]] = None) -> None:
        key = compare if compare is not None else data
        if self._saved.get(name) == key:
            return
        self.journal.save_json(name, data)
        self._saved[name] = key

    def _save_desk_state(self) -> None:
        self._save_if_changed(STATE_NAME, self._desk_state())

    def _save_risk_state(self) -> None:
        risk = self.risk.to_dict()
        # last_equity cambia en cada ciclo; sólo provoca escritura junto a otro cambio.
        self._save_if_changed(RISK_STATE_NAME, risk, {k: v for k, v in risk.items() if k != "last_equity"})

    def _save_risk_quiet(self) -> None:
        """Halt / pausa / cooldown a disco al momento (sobreviven a un crash antes del próximo ciclo)."""
        try:
            self._save_risk_state()
        except OSError as e:
            log.error("no se pudo guardar el estado de riesgo: %s", e)

    def save_state(self, force: bool = False) -> None:
        """Guarda mesa, riesgo, motor y broker (sólo lo que cambió).

        El broker paper ya guarda solo en cada cambio (``autosave``); aquí sólo se le pide al detener la mesa
        (``force``) o, si no tiene autosave, como mucho una vez por minuto."""
        self._save_desk_state()
        self._save_risk_state()
        self.engine.save_state()
        now = self.clock.now()
        if force or (not getattr(self.broker, "autosave", False) and now - self._last_broker_save_ts >= 60.0):
            self._last_broker_save_ts = now
            self.broker.save_state()

    # ================================================================== bucle 24/7
    def request_stop(self) -> None:
        self._stop_requested = True

    def _on_signal(self, signum: int, frame: Any) -> None:
        if self._stop_requested:
            raise KeyboardInterrupt
        self._stop_requested = True
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        log.warning("señal %s recibida: la mesa se detiene al terminar el ciclo (otra vez para forzar)", name)

    def _sleep(self, seconds: float) -> None:
        """Espera hasta el próximo ciclo; la corta un comando encolado o una línea nueva en inbox.txt (un
        ``kill`` enviado con ``python -m desk send`` no espera al backoff, que puede llegar a 300 s)."""
        remaining = seconds
        inbox = self._inbox_stat()
        while remaining > 0 and not self._stop_requested:
            step = min(remaining, 1.0)
            self.clock.sleep(step)
            remaining -= step
            if self._queue or self._inbox_stat() != inbox:
                break                            # hay comandos: ciclo ya

    def banner(self) -> str:
        s = self.settings
        names = ", ".join(st.name for st in self.strategies) or "sin estrategias"
        try:
            eq = fmt_usd(self.broker.account().equity, sign=False)
        except (MarketDataError, BrokerError) as e:
            eq = f"no disponible ({e})"
        lines = [f"Mesa en marcha: modo {s.mode.upper()}, datos {s.data_source}, símbolos {', '.join(s.symbols)}",
                 f"Automático {'ON' if self.auto_enabled else 'OFF'} ({names}); ciclo {s.loop_seconds:g} s; "
                 f"equity {eq}; estado en {self.journal.dir}",
                 f"Riesgo: {s.risk_per_trade_pct:g}%/operación, pérdida diaria {s.daily_loss_limit_pct:g}%, drawdown "
                 f"máx. {s.max_drawdown_pct:g}%, máx. {s.max_open_positions} posiciones"]
        if self.restored_params:
            lines.append("Parámetros restaurados del estado: " + ", ".join(self.restored_params))
        lines.extend(self.notices)
        if self.risk.active_halt():
            lines.append(f"ATENCIÓN: {self.risk.block_reason()}")
        return "\n".join(lines)

    def run_forever(self, max_cycles: Optional[int] = None, *, install_signals: bool = True,
                    before_cycle: Optional[Callable[[int], None]] = None) -> int:
        """Bucle 24/7. ``max_cycles`` limita el número de ciclos (simulaciones/tests). Devuelve los ciclos hechos."""
        old_handlers = {}
        if install_signals and threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    old_handlers[sig] = signal.signal(sig, self._on_signal)
                except (ValueError, OSError):
                    pass
        self.journal.event("start", settings=self.settings.public_dict())
        self.notify(self.banner())
        n = 0
        try:
            while not self._stop_requested:
                if before_cycle is not None:
                    before_cycle(n)
                self.run_cycle()
                n += 1
                if max_cycles is not None and n >= max_cycles:
                    break
                self._sleep(self.next_delay())
        except KeyboardInterrupt:
            log.warning("interrupción: se guarda el estado y se sale")
        finally:
            self.shutdown()
            for sig, h in old_handlers.items():
                try:
                    signal.signal(sig, h)
                except (ValueError, OSError):
                    pass
        return n

    def shutdown(self) -> None:
        with self._lock:
            for what, fn in (("estado", lambda: self.save_state(force=True)), ("snapshot", self.write_status_snapshot)):
                try:
                    fn()
                except Exception as e:  # noqa: BLE001
                    log.error("al detener: no se pudo guardar %s: %s", what, e)
            self.journal.event("stop", cycles=self.cycle, session=self.session)
            where = "en el exchange" if self.settings.is_live else "en el estado guardado del broker paper"
            self.notify(f"Mesa detenida tras {self.cycle} ciclos. Las órdenes SL/TP siguen {where}.")

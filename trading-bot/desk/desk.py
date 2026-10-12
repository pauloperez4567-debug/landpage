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
  ``enqueue(text)`` y ``state/inbox.txt`` se procesan al principio del ciclo. El inbox se lee con un offset
  persistido que se guarda *antes* de ejecutar (como mucho una vez: tras un reinicio nunca se repite un
  "long HYPE"). Una línea ``[id] texto`` usa ese id; si no, la mesa asigna uno (``c-000001``). Las respuestas
  van a ``outbox.log`` con el id del comando.
- Comandos interpretados por IA (``source="llm"``) que cambian algo quedan pendientes de ``si``/``no`` si
  ``llm_confirm_trades`` (5 min para confirmar).
- La operativa automática sólo abre en símbolos de ``settings.symbols`` sin posición, sin entrada pendiente
  y sin operación gestionada: nunca toca posiciones abiertas por comando (ni las cierra: cada operación sale
  por su propio SL/TP o por orden del usuario).
- SIGINT/SIGTERM → termina el ciclo en curso, guarda el estado y sale; las órdenes SL/TP quedan en el exchange
  (en paper, en el estado guardado del broker). Un segundo Ctrl+C fuerza la salida.

Todas las horas salen del ``Clock`` inyectado.
"""

from __future__ import annotations

import logging
import math
import re
import signal
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .clock import Clock
from .commands import HELP_TEXT, CommandParser, describe
from .config import TUNABLE, Settings
from .engine import PENDING, ClosedTrade, ExecutionEngine, fmt_px, fmt_size, fmt_usd
from .exchange.base import Broker, BrokerError, MarketData, MarketDataError, round_price
from .exchange.intervals import interval_seconds
from .journal import Journal
from .liquidity import LiquidityScanner
from .models import (AccountState, Command, CommandAction, EntryMode, OrderStatus, RiskDecision, Side, Signal,
                     TradeIntent)
from .risk import (EVENT_COOLDOWN_END, EVENT_COOLDOWN_START, EVENT_FLATTEN, EVENT_HALT_DAILY, EVENT_HALT_DRAWDOWN,
                   EVENT_HALT_LIFTED, EVENT_NEW_DAY, RiskManager)
from .strategy import Strategy, build_strategies

log = logging.getLogger(__name__)
# Avisos para el operador (ya salen por pantalla vía ``out``): logger aparte para poder filtrarlo en consola.
notice_log = logging.getLogger("desk.notice")

STATE_NAME = "desk"
RISK_STATE_NAME = "risk"
STATUS_NAME = "status"
HEARTBEAT_FILE = "heartbeat"
STATE_VERSION = 1

MAX_BACKOFF_S = 300.0
CONFIRM_TTL_S = 300.0
EQUITY_EVERY_S = 60.0
INBOX_ID_RE = re.compile(r"^\[([A-Za-z0-9_.:\-]{1,64})\]\s*(.*)$")

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
        self._bar_seen: dict[tuple[str, str], float] = {}
        self._coin_errors: dict[str, str] = {}
        self.pending: Optional[tuple[Command, float]] = None   # comando IA pendiente de confirmar
        self._saved: dict[str, Any] = {}
        self.session = {"trades": 0, "wins": 0, "pnl": 0.0, "fees": 0.0}

        state = self.journal.load_json(STATE_NAME, {}) or {}
        if not isinstance(state, dict):
            state = {}
        self.inbox_offset = int(state.get("inbox_offset") or 0)
        self.cmd_seq = int(state.get("cmd_seq") or 0)
        # DESK_AUTO=false / --no-auto siempre arrancan en OFF; si no, se respeta el último "auto on/off".
        self.auto_enabled = bool(settings.auto_enabled) and bool(state.get("auto_enabled", True))
        self.param_overrides: dict[str, float] = {}
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
        order = self.engine.open(intent, decision, cmd_id=cmd_id)
        if order.status is OrderStatus.REJECTED:
            self._flush_engine()
            return False, f"{coin} {intent.side.value.upper()}: el broker rechazó la entrada: {order.reason}"
        extra: list[str] = []
        self._sync_fills(extra)                  # entrada a mercado: SL/TP colocados ya, no en el próximo ciclo
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
        msgs: list[str] = []
        order = self.engine.close(coin, "close")
        self._flush_engine(msgs)
        if order is None:
            return "\n".join(msgs) if msgs else f"No hay posición ni entrada pendiente en {coin}."
        self._sync_fills(msgs)
        return "\n".join(msgs)

    def _cmd_close_all(self, cmd: Command) -> str:
        msgs: list[str] = []
        pending = [c for c, b in self.engine.brackets.items() if b.status == PENDING]
        orders = self.engine.close_all("close_all")
        self._flush_engine(msgs)
        self._sync_fills(msgs)
        if not orders and not pending:
            return "No había posiciones abiertas ni entradas pendientes."
        head = f"Cerrando todo: {len(orders)} posición(es) a mercado"
        if pending:
            head += f", entradas pendientes canceladas ({', '.join(sorted(pending))})"
        return "\n".join([head + "."] + msgs)

    def _cmd_cancel(self, cmd: Command) -> str:
        n = self.engine.cancel(cmd.coin)
        what = f"de {cmd.coin}" if cmd.coin else "de todas las monedas"
        msgs: list[str] = []
        self._flush_engine(msgs)
        if n == 0:
            return f"No había órdenes de entrada abiertas {what}. Los SL/TP de las posiciones no se tocan."
        return "\n".join([f"{n} orden(es) de entrada canceladas {what}. Los SL/TP de las posiciones abiertas se "
                          "mantienen (para salir usa 'cierra <moneda>')."] + msgs)

    def _cmd_set_level(self, cmd: Command) -> str:
        coin = cmd.coin
        is_sl = cmd.action is CommandAction.SET_SL
        label = "SL" if is_sl else "TP"
        pct = cmd.sl_pct if is_sl else cmd.tp_pct
        px = cmd.price
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
        msgs: list[str] = []
        canceled = self.engine.cancel(None)
        orders = self.engine.close_all("kill")
        self._flush_engine(msgs)
        self._sync_fills(msgs)
        self._save_desk_state()
        head = (f"PÁNICO: {len(orders)} posición(es) cerrándose a mercado, {canceled} entrada(s) canceladas. "
                "Mesa DETENIDA: no se abre nada (ni comandos ni automático) hasta 'reset halt'.")
        return "\n".join([head] + msgs)

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
            return not errors

    def next_delay(self) -> float:
        """Espera hasta el próximo ciclo: ``loop_seconds`` o backoff exponencial si hay fallos seguidos."""
        base = max(float(self.settings.loop_seconds), 0.1)
        if self.error_streak <= 0:
            return max(base - self._last_cycle_s, 0.0)
        return min(base * (2 ** min(self.error_streak, 20)), MAX_BACKOFF_S)

    # ------------------------------------------------------------------ etapas
    def _stage_commands(self) -> None:
        items: list[tuple[str, str, str]] = []
        while self._queue:
            items.append(self._queue.popleft())
        items.extend(self._read_inbox())
        for cid, text, source in items:
            reply = self._process(cid, text, source)
            self._emit(f"» [{cid}] {text}\n{reply}")

    def _read_inbox(self) -> list[tuple[str, str, str]]:
        path = self.journal.inbox_path
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return []
        if size < self.inbox_offset:
            log.warning("inbox.txt es más corto que el offset guardado (%d < %d): se relee desde el principio",
                        size, self.inbox_offset)
            self.inbox_offset = 0
        if size == self.inbox_offset:
            return []
        with open(path, "rb") as f:
            f.seek(self.inbox_offset)
            data = f.read(size - self.inbox_offset)
        end = data.rfind(b"\n")
        if end < 0:
            return []                            # línea a medio escribir: se espera al salto de línea
        chunk = data[:end + 1]
        previous = self.inbox_offset
        self.inbox_offset += len(chunk)
        # Offset guardado ANTES de ejecutar: un comando del inbox se ejecuta como mucho una vez.
        out: list[tuple[str, str, str]] = []
        for raw in chunk.decode("utf-8", "replace").splitlines():
            line = raw.strip().lstrip("﻿")
            if not line or line.startswith("#"):
                continue
            m = INBOX_ID_RE.match(line)
            cid, text = (m.group(1), m.group(2).strip()) if m else (self._next_id(), line)
            out.append((cid, text, "inbox"))
        try:
            self._save_desk_state()
        except OSError:
            self.inbox_offset = previous          # sin offset guardado no se ejecuta: se reintenta luego
            raise
        return out

    def _sync_fills(self, target: Optional[list[str]] = None) -> list[ClosedTrade]:
        fills = self.broker.poll_fills()
        if not fills:
            self._flush_engine(target)
            return []
        closed = self.engine.on_fills(fills)
        self._flush_engine(target)
        self._record_closed(closed, target)
        return closed

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
        events = self.risk.update(account)
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
            if orders or had_pending:
                acted = True
                self.notify(f"Riesgo ({self.risk.halt_reason}): cerrando {len(orders)} posición(es) a mercado.",
                            logging.WARNING)
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
        self.journal.touch(HEARTBEAT_FILE, f"{now}\n")
        if self._account_dirty:                  # el automático abrió algo después de leer la cuenta
            self._account_dirty = False
            self._account = self.broker.account()
        acct = self._account
        if acct is not None and now - self._last_equity_ts >= EQUITY_EVERY_S:
            self._last_equity_ts = now
            self.journal.equity(now, acct.equity, acct.available,
                                sum(1 for p in acct.positions.values() if p.size != 0))
        if now - self._last_status_ts >= self.settings.status_every_seconds:
            self._last_status_ts = now
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
            "cmd_seq": self.cmd_seq,
            "auto_enabled": self.auto_enabled,
            "param_overrides": dict(self.param_overrides),
            "day_stats": dict(self.day_stats),
        }
        cursor = getattr(self.broker, "fills_cursor_ts", None)
        if isinstance(cursor, (int, float)):
            data["fills_cursor_ts"] = cursor
        return data

    def _save_if_changed(self, name: str, data: dict[str, Any], compare: Optional[dict[str, Any]] = None) -> None:
        key = compare if compare is not None else data
        if self._saved.get(name) == key:
            return
        self.journal.save_json(name, data)
        self._saved[name] = key

    def _save_desk_state(self) -> None:
        self._save_if_changed(STATE_NAME, self._desk_state())

    def save_state(self) -> None:
        """Guarda mesa, riesgo, motor y broker (sólo lo que cambió)."""
        self._save_desk_state()
        risk = self.risk.to_dict()
        # last_equity cambia en cada ciclo; sólo provoca escritura junto a otro cambio.
        self._save_if_changed(RISK_STATE_NAME, risk, {k: v for k, v in risk.items() if k != "last_equity"})
        self.engine.save_state()
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
        remaining = seconds
        while remaining > 0 and not self._stop_requested:
            step = min(remaining, 1.0)
            self.clock.sleep(step)
            remaining -= step
            if self._queue:
                break                            # hay comandos encolados: ciclo ya

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
            for what, fn in (("estado", self.save_state), ("snapshot", self.write_status_snapshot)):
                try:
                    fn()
                except Exception as e:  # noqa: BLE001
                    log.error("al detener: no se pudo guardar %s: %s", what, e)
            self.journal.event("stop", cycles=self.cycle, session=self.session)
            where = "en el exchange" if self.settings.is_live else "en el estado guardado del broker paper"
            self.notify(f"Mesa detenida tras {self.cycle} ciclos. Las órdenes SL/TP siguen {where}.")

"""La mesa: bucle 24/7 que une comandos, estrategias, riesgo y ejecución.

Ciclo (cada ``loop_seconds``), cada etapa en su propio try/except::

    comandos (cola en memoria + state/inbox.txt) → broker.step() → poll_fills → engine.on_fills
    → account → risk.update (halts / flatten) → engine.reconcile
    → cada strategy_seconds, si auto y riesgo lo permite: playbook + estrategias por defecto → riesgo → engine
    → alarmas (AlertManager.evaluate) → backtests terminados (avisos) → guardar estado
    → status.json (ampliado, para el panel) y línea de estado cada status_every_seconds

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

Fase 2 (backtest por gráfica, playbook, alarmas, Telegram):

- ``backtest COIN tf [días] [estrategia]``: la mesa descarga las velas al momento (un error de datos se contesta ya)
  y encola el trabajo en ``jobs.BacktestWorker``: un hilo propio, uno a la vez, cancelable al apagar. El bucle sigue
  vigilando posiciones mientras tanto; al terminar, un aviso con ``summary_text`` y las rutas del informe
  (JSON/HTML/PNG; el PNG va a Telegram con ``sendPhoto``). Con reloj simulado (``--fast``, tests) el trabajo se
  ejecuta en la etapa "backtests" del ciclo siguiente, sin hilos: determinista.
- ``activa COIN tf [estrategia] [forzar]``: el último informe de la gráfica → ``Playbook``. Sólo lo aprobado fuera de
  muestra se activa; "forzar" lo permite con una advertencia explícita. La operativa automática evalúa cada entrada
  del playbook con su temporalidad, parámetros y unidades, más las estrategias por defecto sobre ``symbols`` si
  ``auto_default_strategies``; origen de las señales ``auto:<estrategia>@<tf>``; mismas protecciones de riesgo.
- Avisos: ``notify`` escribe ``outbox.log`` y los pasa a ``AlertManager.system`` (``alerts.jsonl`` + ``Notifier``:
  pantalla, Telegram, webhook; anti-spam por clave y silencio con ``mute``). Halts = crítico; consumo de la pérdida
  diaria (``daily_loss_warn_frac``) y del drawdown (80 %), fallos repetidos y datos viejos = aviso.
- ``TelegramBot`` (si hay token): cada mensaje autorizado va a ``submit`` (mismo lock que el ciclo: seguro entre
  hilos) y la respuesta vuelve al chat.

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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .alerts import CANDLE_KINDS, PRICE_KINDS, AlertManager, describe_rule
from .clock import Clock, SimClock
from .commands import HELP_TEXT, PARAM_DEFAULTS, CommandParser, describe
from .config import TUNABLE, Settings
from .engine import PENDING, ClosedTrade, ExecutionEngine, fmt_px, fmt_size, fmt_usd
from .exchange.base import Broker, BrokerError, MarketData, MarketDataError, round_price
from .exchange.intervals import interval_seconds
from .jobs import BacktestJob, BacktestOutcome, BacktestWorker, fetch_bars
from .journal import Journal
from .liquidity import LiquidityScanner
from .models import (AccountState, Command, CommandAction, EntryMode, OrderStatus, RiskDecision, Side, Signal,
                     TradeIntent)
from .notify import ConsoleChannel, Notifier, build_notifier
from .optimizer import MIN_SEGMENT_BARS, OptimizationReport, ReportStore, format_params
from .playbook import Playbook, entry_from_report
from .risk import (EVENT_COOLDOWN_END, EVENT_COOLDOWN_START, EVENT_FLATTEN, EVENT_HALT_DAILY, EVENT_HALT_DRAWDOWN,
                   EVENT_HALT_LIFTED, EVENT_NEW_DAY, LIMIT_MAX_DISTANCE_PCT, RiskManager)
from .strategy import STRATEGIES, Strategy, build_strategies

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

# --- fase 2 ---
SNAPSHOT_EVERY_S = 30.0      # status.json ampliado (lo lee el panel): como mucho cada 30 s de reloj, o tras un cambio
SIGNALS_KEPT = 20            # últimas señales automáticas en status.json
DD_WARN_FRAC = 0.8           # aviso al consumir el 80 % del drawdown máximo
WARN_REARM_FRAC = 0.1        # histéresis: los avisos de consumo se rearman al bajar 10 puntos del umbral
STALE_CHECK_S = 60.0         # cada cuánto se comprueba la frescura de los datos de las posiciones abiertas
ERROR_STREAK_WARN = 3        # ciclos seguidos con fallos que disparan un aviso (y luego 10, 50, 100...)
BT_MIN_BARS = 2 * MIN_SEGMENT_BARS   # un tramo de entrenamiento y uno de validación como mínimo
DEFAULT_BT_INTERVAL = "1h"   # temporalidad de "backtest COIN" sin temporalidad (ni estrategia)
REPORT_STALE_DAYS = 30.0     # al activar un informe más antiguo se sugiere repetirlo

# Acciones de la IA que se ejecutan sin confirmar (no abren ni cierran nada, o sólo frenan). Un backtest sólo
# lee velas y escribe un informe; silenciar avisos no toca dinero.
_LLM_SAFE = frozenset({CommandAction.STATUS, CommandAction.HELP, CommandAction.PAUSE, CommandAction.AUTO_OFF,
                       CommandAction.BACKTEST, CommandAction.PLAYBOOK, CommandAction.REPORTS,
                       CommandAction.ALERT_LIST, CommandAction.MUTE})


def _hhmm(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _alert_level(level: int) -> str:
    """Nivel de ``logging`` → nivel de aviso ("info" | "warn" | "critical")."""
    if level >= logging.ERROR:
        return "critical"
    if level >= logging.WARNING:
        return "warn"
    return "info"


def _strategy_key(name: Optional[str]) -> str:
    return str(name or "").strip().lower().replace("-", "_").replace(" ", "_")


def _fmt_metrics(m: dict[str, Any]) -> str:
    """Métricas fuera de muestra en una línea: retorno, PF, DD, operaciones."""
    def num(key: str) -> float:
        try:
            v = float(m.get(key))
        except (TypeError, ValueError):
            return 0.0
        return v if math.isfinite(v) else 0.0
    pf = num("profit_factor")
    pf_txt = "∞" if pf >= 999 else f"{pf:.2f}"
    return (f"{num('total_return_pct'):+.1f} %, PF {pf_txt}, DD {num('max_drawdown_pct'):.1f} %, "
            f"{int(num('n_trades'))} ops")


@dataclass
class AutoSlot:
    """Una estrategia que la operativa automática evalúa sobre una moneda (del playbook o por defecto)."""

    coin: str
    strategy: Strategy
    units: float
    source: str                          # "playbook" | "defecto"

    @property
    def label(self) -> str:
        """``nombre@temporalidad``: clave de "una evaluación por vela" y sufijo del origen ``auto:``."""
        return f"{self.strategy.name}@{self.strategy.interval}"

    @property
    def origin(self) -> str:
        return f"auto:{self.label}"


class TradingDesk:
    def __init__(self, settings: Settings, market: MarketData, broker: Broker, clock: Clock,
                 parser: Any = None, journal: Optional[Journal] = None, *,
                 strategies: Optional[list[Strategy]] = None, scanner: Optional[LiquidityScanner] = None,
                 risk: Optional[RiskManager] = None, engine: Optional[ExecutionEngine] = None,
                 out: Optional[Callable[[str], None]] = None, notifier: Optional[Notifier] = None,
                 alerts: Optional[AlertManager] = None, playbook: Optional[Playbook] = None,
                 reports: Optional[ReportStore] = None, backtests: Optional[BacktestWorker] = None,
                 telegram: Any = None, http_session: Any = None):
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
        # Estrategias por defecto (sobre ``symbols``, si ``auto_default_strategies``); el playbook va aparte.
        self.strategies = strategies if strategies is not None else build_strategies(settings, clock)
        # --- fase 2: avisos, alarmas, playbook, informes, backtests y Telegram ---
        self.sim_clock = isinstance(clock, SimClock)
        # La pantalla (``out``) es un canal más del Notifier: un aviso silenciado o repetido no se pinta.
        self.notifier = notifier if notifier is not None else build_notifier(
            settings, out=self._emit if out is not None else None, console=out is not None, clock=clock,
            session=http_session)
        self._screen_via_notifier = any(isinstance(ch, ConsoleChannel) for ch in self.notifier.channels)
        self.alerts = alerts if alerts is not None else AlertManager(settings, clock, self.journal.dir, self.notifier)
        self.playbook = playbook if playbook is not None else Playbook(self.journal.dir, clock)
        self.reports = reports if reports is not None else ReportStore(self.journal.dir)
        self.backtests = backtests if backtests is not None else BacktestWorker(
            settings, self.reports, clock, threaded=not self.sim_clock)
        self.telegram = telegram if telegram is not None else self._build_telegram(http_session)
        self.signals_log: deque[dict[str, Any]] = deque(maxlen=SIGNALS_KEPT)
        self._cmd_source = ""
        self._snapshot_dirty = False
        self._last_snapshot_ts = -math.inf
        self._last_stale_check_ts = -math.inf
        self._warned_daily: Optional[str] = None     # día UTC en que ya se avisó del consumo de la pérdida diaria
        self._warned_dd = False

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

    def _build_telegram(self, session: Any) -> Any:
        """``TelegramBot`` si hay ``TELEGRAM_BOT_TOKEN`` (se arranca en ``run_forever``). Sin chats autorizados
        arranca igual: ignora todo y registra el id de quien escribe (así se descubre el chat id)."""
        token = str(self.settings.telegram_bot_token or "").strip()
        if not token:
            return None
        from .telegram_bot import MAX_AGE_S, TelegramBot
        chats = [str(c).strip() for c in self.settings.telegram_chat_ids or [] if str(c).strip()]
        # Con reloj simulado la "edad" de un mensaje no tiene sentido: no se descartan por antiguos.
        return TelegramBot(token, chats, self.submit, self.clock, session=session,
                           max_age_s=None if self.sim_clock else MAX_AGE_S)

    # ================================================================== utilidades
    def _emit(self, text: str) -> None:
        if self.out is not None:
            try:
                self.out(text)
            except Exception:  # noqa: BLE001 - la salida por pantalla nunca tumba la mesa
                log.exception("error escribiendo en la salida")

    def notify(self, text: str, level: int = logging.INFO, *, kind: str = "desk", key: Optional[str] = None,
               coin: Optional[str] = None, image: Optional[bytes] = None, push: bool = True) -> None:
        """Aviso de la mesa: log, ``outbox.log`` (id ``desk``) y, con ``push``, ``AlertManager.system`` (``alerts.jsonl``
        y los canales del Notifier: pantalla, Telegram, webhook, según su nivel mínimo). Con ``key`` un aviso igual no
        se repite antes de ``alert_cooldown_seconds`` (ni en el outbox); ``mute`` silencia los no críticos en los
        canales (el outbox y ``alerts.jsonl`` los conservan). Sin ``push``: sólo log, outbox y pantalla."""
        lvl = _alert_level(level)
        delivered = True
        if push:
            delivered = self.alerts.system(kind, text, lvl, key=key, coin=coin, image=image)
            muted = lvl != "critical" and self.alerts.is_muted()
            if not delivered and key is not None and not muted:
                log.debug("aviso repetido omitido (%s): %s", key, text[:120])
                return
        notice_log.log(level, "%s", text)
        self.journal.outbox("desk", text)
        if not push or not self._screen_via_notifier:
            self._emit(text)

    def _flush_engine(self, target: Optional[list[str]] = None) -> None:
        for m in self.engine.pop_messages():
            if target is not None:
                target.append(m)
            else:
                warn = m.split(":", 1)[-1].lstrip().startswith("ATENCIÓN") or "por seguridad" in m
                self.notify(m, logging.WARNING if warn else logging.INFO,
                            kind="trade" if " cerrada por " in m else "exec")

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
        self._cmd_source = source
        self._snapshot_dirty = True              # el panel ve el efecto del comando sin esperar 30 s
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
            CommandAction.BACKTEST: self._cmd_backtest,
            CommandAction.ADOPT: self._cmd_adopt,
            CommandAction.DEACTIVATE: self._cmd_deactivate,
            CommandAction.PLAYBOOK: self._cmd_playbook,
            CommandAction.REPORTS: self._cmd_reports,
            CommandAction.ALERT_ADD: self._cmd_alert_add,
            CommandAction.ALERT_LIST: lambda c: self.alerts.list_text(),
            CommandAction.ALERT_DEL: self._cmd_alert_del,
            CommandAction.MUTE: self._cmd_mute,
            CommandAction.UNMUTE: lambda c: self.alerts.unmute(),
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
        parts = []
        if self.settings.auto_default_strategies and self.strategies:
            parts.append(f"{', '.join(s.name for s in self.strategies)} sobre {', '.join(self.settings.symbols)}")
        n_pb = sum(1 for e in self.playbook.entries() if e.enabled)
        if n_pb:
            parts.append(f"{n_pb} gráfica(s) del playbook")
        text = f"Operativa automática ACTIVADA ({'; '.join(parts) or 'ninguna estrategia ni gráfica en el playbook'})."
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

    # ------------------------------------------------------------------ backtest por gráfica y playbook
    def _cmd_backtest(self, cmd: Command) -> str:
        coin = cmd.coin
        if not coin:
            return "¿De qué gráfica? Ejemplos: 'backtest HYPE 1h', 'backtest BTC 15m 90d', 'prueba SOL 1h ema_trend'."
        strategy: Optional[str] = None
        if cmd.strategy:
            strategy = _strategy_key(cmd.strategy)
            if strategy not in STRATEGIES:
                return f"Estrategia desconocida: {cmd.strategy}. Disponibles: {', '.join(sorted(STRATEGIES))}."
        interval = cmd.interval or (STRATEGIES[strategy].interval if strategy else DEFAULT_BT_INTERVAL)
        try:
            interval_seconds(interval)
        except ValueError as e:
            return f"No se lanzó el backtest: {e}"
        days = float(cmd.days) if cmd.days else float(self.settings.bt_default_days)
        if not math.isfinite(days) or days <= 0:
            return "Los días de historia deben ser un número > 0 (p. ej. 'backtest HYPE 1h 90d')."
        try:
            meta = self.market.meta(coin)
        except KeyError:
            return f"No conozco la moneda {coin} en este mercado."
        chart = f"{coin} {interval}"
        if self.backtests.busy_with(coin, interval, strategy):
            return f"Ya hay un backtest de {chart} en marcha o en cola: espera al aviso de que terminó."
        try:
            bars, notes = fetch_bars(self.market, coin, interval, days, self.clock)
        except (KeyError, ValueError) as e:      # temporalidad que esta fuente de datos no sirve, etc.
            return f"No se lanzó el backtest de {chart}: sin velas ({e})."
        n = 0 if bars is None else len(bars)
        if n < BT_MIN_BARS:
            return (f"{chart}: sólo hay {n} velas cerradas y el walk-forward necesita al menos {BT_MIN_BARS} (tramos "
                    "de entrenamiento y de validación). Prueba con más días o una temporalidad menor.")
        job = BacktestJob(id=cmd.id or self._next_id(), coin=coin, interval=interval, days=days, strategy=strategy,
                          bars=bars, meta=meta, requested_ts=self.clock.now(), cmd_id=cmd.id, notes=notes,
                          source=self._cmd_source, settings=replace(self.settings))
        try:
            ahead = self.backtests.submit(job)
        except ValueError as e:
            return f"No se lanzó el backtest de {chart}: {e}."
        self.journal.event("backtest_requested", id=job.id, coin=coin, interval=interval, days=days,
                           strategy=strategy, bars=n, source=self._cmd_source)
        what = strategy or f"{len(STRATEGIES)} estrategias"
        queue = f" (en cola: {ahead} por delante)" if ahead else ""
        lines = [f"Backtest {chart} ({days:g} días, {n} velas, {what}) en marcha{queue}..."]
        if notes:
            lines.append("Notas: " + "; ".join(notes))
        if self.backtests.threaded:
            lines.append("La mesa sigue operando mientras tanto; te aviso al terminar con el resumen y el informe "
                         "(HTML y PNG en reports/).")
        else:
            lines.append("Reloj simulado: se ejecuta en este mismo ciclo (determinista) y luego llega el aviso.")
        lines.append("Decide con las métricas FUERA de muestra: un backtest no garantiza resultados futuros.")
        return "\n".join(lines)

    def _on_backtest_done(self, out: BacktestOutcome) -> None:
        """Aviso con el resultado de un backtest (o su fallo) y comprobación del playbook de esa gráfica."""
        job = out.job
        self._snapshot_dirty = True
        rep = out.report
        self.journal.event("backtest_done", id=job.id, coin=job.coin, interval=job.interval, ok=out.ok,
                           error=out.error or None, report=rep.id if rep else None, winner=rep.winner if rep else None,
                           elapsed_s=round(out.elapsed_s, 2), path=str(out.report_path) if out.report_path else None)
        if not out.ok:
            self.notify(f"Backtest {job.label()} falló: {out.error or 'sin informe'}. La mesa sigue operando; puedes "
                        "repetirlo.", logging.WARNING, kind="backtest", coin=job.coin)
            return
        lines = [rep.summary_text()]
        files = [("HTML", out.html_path), ("PNG", out.png_path), ("JSON", out.report_path)]
        lines.append("Informe: " + " | ".join(f"{what} {path}" for what, path in files if path))
        if rep.winner:
            lines.append(f"Antes de activarla revisa el informe (fuera de muestra, operaciones, drawdown). Para operarla "
                         f"en automático: 'activa {rep.coin} {rep.interval}' (empieza en paper y obsérvala).")
        lines.append(f"({out.elapsed_s:.0f} s de cálculo.)")
        self.notify("\n".join(lines), kind="backtest", coin=job.coin, image=out.png)
        self._check_playbook_after(rep)

    def _check_playbook_after(self, rep: OptimizationReport) -> None:
        """Si la gráfica está en el playbook y su estrategia ya no aprueba en el informe nuevo: aviso (no se desactiva
        sola: decide el operador)."""
        entry = self.playbook.get(rep.coin, rep.interval)
        if entry is None:
            return
        ev = rep.get(entry.strategy)
        if ev is None:
            return
        chart = f"{rep.coin} {rep.interval}"
        if ev.error or not ev.passed:
            why = "; ".join(ev.reasons) or ev.error or "no aprobó"
            self.notify(f"PLAYBOOK DEGRADADO: {chart} opera {entry.strategy} en automático, pero en el backtest nuevo "
                        f"(informe {rep.id}) NO aprueba fuera de muestra: {why}. Sigue activa: revisa el informe y usa "
                        f"'desactiva {chart}' para pararla.", logging.WARNING, kind="playbook", coin=rep.coin)
        elif rep.winner and rep.winner != entry.strategy:
            self.notify(f"{chart}: el backtest nuevo (informe {rep.id}) da como ganadora {rep.winner}; el playbook sigue "
                        f"con {entry.strategy}, que también aprueba. Para cambiar: 'activa {chart}'.",
                        kind="playbook", coin=rep.coin)

    def _cmd_adopt(self, cmd: Command) -> str:
        coin, interval = cmd.coin, cmd.interval
        if not coin or not interval:
            return "Indica la gráfica completa: 'activa HYPE 1h' (moneda y temporalidad)."
        chart = f"{coin} {interval}"
        report = self.reports.latest(coin, interval)
        if report is None:
            running = " (hay uno en marcha: espera al aviso)" if self.backtests.busy_chart(coin, interval) else ""
            return f"No hay ningún backtest de {chart}{running}. Primero: 'backtest {chart}'."
        strategy = _strategy_key(cmd.strategy) if cmd.strategy else None
        picked = False
        if strategy is None and report.winner is None and cmd.force:
            best = report.best()
            if best is None:
                return f"El backtest de {chart} (informe {report.id}) no tiene ninguna estrategia evaluable."
            strategy, picked = best.strategy, True
        try:
            entry = entry_from_report(report, strategy, clock=self.clock, allow_rejected=cmd.force)
        except ValueError as e:
            text = str(e)
            ev = report.get(strategy) if strategy else None
            if not cmd.force and (ev is None or not ev.error) and ("no aprobó" in text or "no tiene ganadora" in text):
                text += (f" Si aun así quieres operarla (NO recomendado: lo que no aprueba fuera de muestra es "
                         f"probablemente ruido): 'activa {chart} [estrategia] forzar'.")
            return text
        try:
            old = self.playbook.adopt(entry, self.settings)
        except ValueError as e:
            return f"No se activó {chart}: {e}."
        except OSError as e:
            return f"No se activó {chart}: no se pudo guardar el playbook ({e})."
        ev = report.get(entry.strategy)
        passed = bool(ev is not None and ev.passed)
        self._snapshot_dirty = True
        self.journal.event("playbook_adopt", coin=entry.coin, interval=entry.interval, strategy=entry.strategy,
                           params=entry.params, report=report.id, forced=not passed)
        units = f"{entry.units:g} u" if entry.units is not None else f"{self.settings.auto_units:g} u (DESK_AUTO_UNITS)"
        lines = []
        if not passed:
            why = "; ".join(ev.reasons) if ev is not None and ev.reasons else "sin aprobar"
            pick = " (no había ganadora: se tomó la mejor del ranking)" if picked else ""
            lines.append(f"ATENCIÓN — FORZADO{pick}: {entry.strategy} NO aprobó fuera de muestra ({why}). Lo más "
                         "probable es que no tenga ventaja real: usa pocas unidades, vigílala y "
                         f"'desactiva {chart}' para pararla.")
        lines.append(f"{chart} en automático con {entry.strategy} ({format_params(entry.params)}), {units} por "
                     f"operación. Señales 'auto:{entry.strategy}@{interval}' con las mismas protecciones de riesgo "
                     "(SL obligatorio, límites de pérdida diaria y drawdown).")
        if entry.oos_metrics:
            age = (self.clock.now() - report.created_ts) / 86_400.0 if report.created_ts else 0.0
            lines.append(f"Fuera de muestra (informe {report.id}): {_fmt_metrics(entry.oos_metrics)}. Orientativo, "
                         "no una promesa.")
            if age > REPORT_STALE_DAYS:
                lines.append(f"El informe tiene {age:.0f} días: considera repetir 'backtest {chart}'.")
        if old is not None:
            lines.append(f"Sustituye a {old.strategy} ({format_params(old.params)}).")
        if not self.auto_enabled:
            lines.append("El automático está OFF: no operará hasta 'auto on'.")
        else:
            block = self.risk.block_reason("auto")
            if block:
                lines.append(f"Ahora mismo no abrirá nada: {block}.")
        if self.settings.is_live:
            lines.append("MODO LIVE: operará con dinero real.")
        else:
            lines.append("Modo paper: obsérvala unos días (panel, 'playbook', 'estado') antes de pasarla a testnet.")
        return "\n".join(lines)

    def _cmd_deactivate(self, cmd: Command) -> str:
        coin, interval = cmd.coin, cmd.interval
        if not coin:
            return "¿Qué gráfica? Ejemplos: 'desactiva HYPE 1h' o 'desactiva HYPE' (todas sus temporalidades)."
        chart = f"{coin} {interval}" if interval else coin
        gone = [e for e in self.playbook.entries()
                if e.coin.upper() == coin.upper() and (interval is None or e.interval == interval)]
        try:
            n = self.playbook.remove(coin, interval)
        except OSError as e:
            return f"No se pudo guardar el playbook ({e}): {chart} sigue activa."
        if not n:
            return f"{chart} no está en el playbook. 'playbook' para ver las gráficas activas."
        self._snapshot_dirty = True
        self.journal.event("playbook_remove", coin=coin, interval=interval, removed=n)
        text = (f"Fuera del automático: {', '.join(f'{e.coin} {e.interval} ({e.strategy})' for e in gone)}. Las "
                f"posiciones abiertas siguen con su SL/TP (para salir: 'cierra {coin}').")
        if self.settings.auto_default_strategies and self.strategies and coin.upper() in self.settings.symbols:
            text += (f" Ojo: {coin} sigue en DESK_SYMBOLS con las estrategias por defecto "
                     "(DESK_AUTO_DEFAULT_STRATEGIES=false para no usarlas).")
        return text

    def _cmd_playbook(self, cmd: Command) -> str:
        lines = [self.playbook.summary_text()]
        if self.settings.auto_default_strategies and self.strategies:
            names = ", ".join(f"{s.name}@{s.interval}" for s in self.strategies)
            lines.append(f"Estrategias por defecto: {names} sobre {', '.join(self.settings.symbols)} "
                         "(DESK_AUTO_DEFAULT_STRATEGIES=false para operar sólo el playbook).")
        else:
            lines.append("Sin estrategias por defecto: sólo opera el playbook.")
        state = "ON" if self.auto_enabled else "OFF ('auto on' para que opere)"
        block = self.risk.block_reason("auto") if self.auto_enabled else None
        lines.append(f"Automático {state}" + (f"; ahora mismo en espera: {block}" if block else "") + ".")
        return "\n".join(lines)

    def _cmd_reports(self, cmd: Command) -> str:
        lines = []
        bt = self.backtests.describe()
        if bt["running"]:
            pct = f" ({bt['progress_pct']:.0f} %)" if bt.get("progress_pct") is not None else ""
            lines.append(f"En marcha: {bt['running']}{pct}" + (f"; en cola: {', '.join(bt['queued'])}"
                                                                if bt["queued"] else ""))
        reps = self.reports.list(10)
        if not reps:
            lines.append("No hay informes guardados. Prueba 'backtest HYPE 1h'.")
            return "\n".join(lines)
        lines.append(f"Últimos informes ({len(reps)}), métricas FUERA de muestra:")
        for r in reps:
            best = r.best()
            head = f"  · {r.id} {r.coin} {r.interval} {r.days:g}d · {_hhmm(r.created_ts)} · "
            head += f"ganadora {r.winner}" if r.winner else "sin ganadora"
            if best is not None and best.oos_metrics:
                who = "" if r.winner else f" (mejor: {best.strategy})"
                head += f"{who}: {_fmt_metrics(best.oos_metrics)}"
            lines.append(head)
        lines.append(f"Ficheros en {self.reports.dir}: <MONEDA>_<tf>_<id>.json / .html / .png. Un backtest no garantiza "
                     "resultados futuros.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ alarmas
    def _cmd_alert_add(self, cmd: Command) -> str:
        coin, spec = cmd.coin, dict(cmd.alert or {})
        if not coin or not spec:
            return "Ejemplos: 'alerta BTC > 100000', 'alerta HYPE rsi < 30 1h', 'alerta SOL zona long'."
        try:
            self.market.meta(coin)
        except KeyError:
            return f"No conozco la moneda {coin} en este mercado."
        kind = str(spec.get("kind") or "")
        if kind in PRICE_KINDS:
            hint = self._thousands_hint(coin, spec.get("level"))
            if hint:
                return f"No se creó la alarma: el nivel {hint}"
        ref: Optional[float] = None
        try:
            ref = self.market.mid(coin)
        except (MarketDataError, KeyError) as e:      # sin precio la referencia del cruce es la primera evaluación
            log.info("alarma %s sin precio de referencia: %s", coin, e)
        try:
            rule = self.alerts.add(coin, spec, ref_px=ref)
        except ValueError as e:
            return f"No se creó la alarma: {e}."
        self.journal.event("alert_added", id=rule.id, coin=rule.coin, kind=rule.kind, params=rule.params,
                           interval=rule.interval, repeat=rule.repeat)
        text = f"Alarma creada: {describe_rule(rule)}."
        if kind in PRICE_KINDS and ref and not rule.params.get("immediate"):
            text += (f" Avisa al CRUZAR el nivel (ahora {fmt_px(ref)}): si ya está al otro lado, espera a que vuelva y "
                     "cruce.")
        elif kind in CANDLE_KINDS:
            text += f" Se evalúa con velas cerradas de {rule.interval}, una vez por vela."
        if self.alerts.is_muted() and rule.severity != "critical":
            text += " (Ahora mismo los avisos están silenciados: 'sonido' para reactivarlos.)"
        return text

    def _cmd_alert_del(self, cmd: Command) -> str:
        if cmd.target_id is None:
            n = self.alerts.clear()
            return f"{n} alarma(s) borradas." if n else "No había alarmas."
        if self.alerts.remove(cmd.target_id):
            return f"Alarma #{cmd.target_id} borrada."
        return f"No existe la alarma #{cmd.target_id}. 'alertas' para ver las activas."

    def _cmd_mute(self, cmd: Command) -> str:
        try:
            return self.alerts.mute(cmd.value)
        except ValueError as e:
            return f"No se silenció: {e}."

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
            self._stage("alarmas", self._stage_alerts, errors)
            self._stage("backtests", self._stage_backtests, errors)
            self._stage("guardar", self.save_state, errors)
            self._stage("estado", self._stage_status, errors)
            previous_streak = self.error_streak
            self.error_streak = self.error_streak + 1 if errors else 0
            if errors and self.error_streak in (1, 3, 10) or (errors and self.error_streak % 50 == 0):
                log.warning("ciclo %d con fallos en: %s (racha %d; próxima espera %.0f s)", self.cycle,
                            ", ".join(errors), self.error_streak, self.next_delay())
            self._notify_error_streak(errors, previous_streak)
            self._last_cycle_s = max(self.clock.now() - t0, 0.0)
            self._write_health(errors)
            return not errors

    def _notify_error_streak(self, errors: list[str], previous: int) -> None:
        """Aviso (warn) cuando los fallos se repiten ciclo tras ciclo (3, 10, 50, 100...) y al recuperarse."""
        try:
            n = self.error_streak
            if errors and (n in (ERROR_STREAK_WARN, 10) or (n >= 50 and n % 50 == 0)):
                last = "; ".join(f"{k}: {self.last_errors[k]}" for k in errors if k in self.last_errors)
                self.notify(f"La mesa lleva {n} ciclos seguidos con fallos ({', '.join(errors)}){': ' + last if last else ''}. "
                            f"Sigue reintentando con espera creciente (próximo ciclo en {self.next_delay():.0f} s).",
                            logging.WARNING, kind="errors", key=f"errors:{n}")
            elif not errors and previous >= ERROR_STREAK_WARN:
                self.notify(f"La mesa se recuperó tras {previous} ciclos con fallos.", kind="errors")
        except Exception:  # noqa: BLE001 - avisar nunca tumba el ciclo
            log.exception("no se pudo avisar de la racha de fallos")

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
        if trades:
            self._snapshot_dirty = True
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
                        self.notify(msg, logging.WARNING, kind="risk")

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
                            f"{fmt_usd(self.risk.day_start_equity, sign=False)}", kind="risk")
            elif ev == EVENT_HALT_DAILY:
                halted_now = True
                tail = "; cerrando todo" if self.risk.flatten_required else "; las posiciones abiertas siguen con su SL/TP"
                self.notify(f"HALT por pérdida diaria: {self.risk.halt_detail}. Sin entradas nuevas hasta el próximo "
                            f"día UTC (o 'reset halt'){tail}.", logging.CRITICAL, kind="halt")
            elif ev == EVENT_HALT_DRAWDOWN:
                halted_now = True
                self.notify(f"HALT por drawdown máximo: {self.risk.halt_detail}. Cerrando todo; sólo se reanuda con "
                            "'reset halt'.", logging.CRITICAL, kind="halt")
            elif ev == EVENT_HALT_LIFTED:
                self.notify("Día nuevo: el halt por pérdida diaria se levantó; la mesa vuelve a operar.", kind="halt")
            elif ev == EVENT_COOLDOWN_END:
                self.notify("Fin del cooldown: la operativa automática puede volver a abrir.", kind="risk")
        acted = False
        if halted_now:
            n = self.engine.cancel(None)
            acted = acted or n > 0
            if n:
                self.notify(f"{n} entrada(s) pendientes canceladas por el halt.", kind="halt")
        if EVENT_FLATTEN in events:
            reason = "kill" if self.risk.halt_reason == "kill" else "flatten"
            had_pending = any(b.status == PENDING for b in self.engine.brackets.values())
            orders = self.engine.close_all(reason)
            failed = dict(self.engine.close_failures)
            if orders or had_pending:
                acted = True
                self.notify(f"Riesgo ({self.risk.halt_reason}): cerrando {len(orders)} posición(es) a mercado.",
                            logging.CRITICAL, kind="halt")
            if failed and failed != self._close_failures_seen:
                self.notify("Riesgo: SIN CERRAR todavía: " + "; ".join(f"{c} ({why})" for c, why in sorted(failed.items()))
                            + ". Se reintenta en cada ciclo.", logging.CRITICAL, kind="halt")
            self._close_failures_seen = failed
            self._flush_engine()
            self._sync_fills()
        if acted:
            self._account = self.broker.account()
        self._check_risk_usage(self._account)

    def _check_risk_usage(self, account: AccountState) -> None:
        """Avisos (warn) al consumir ``daily_loss_warn_frac`` del límite de pérdida diaria y el 80 % del drawdown
        máximo, antes de que salte el halt. Una vez por cruce (se rearman al bajar 10 puntos del umbral)."""
        if self.risk.active_halt() or account is None:
            return
        s = self.settings
        eq = account.equity
        frac = float(s.daily_loss_warn_frac or 0.0)
        if s.daily_loss_limit_pct > 0 and 0.0 < frac < 1.0:
            loss = max(self.risk.day_loss_pct(eq), 0.0)
            used = loss / s.daily_loss_limit_pct
            day = self.risk.day
            if used >= frac and self._warned_daily != day:
                self._warned_daily = day
                self.notify(f"Pérdida del día al {used * 100:.0f} % del límite: -{loss:.2f} % de {s.daily_loss_limit_pct:g} % "
                            f"(equity {fmt_usd(eq, sign=False)}). Al llegar al límite la mesa se detiene (halt) hasta el "
                            "próximo día UTC.", logging.WARNING, kind="risk", key=f"daily_warn:{day}")
            elif self._warned_daily == day and used < frac - WARN_REARM_FRAC:
                self._warned_daily = None
        if s.max_drawdown_pct > 0:
            dd = self.risk.drawdown_pct(eq)
            used = dd / s.max_drawdown_pct
            if used >= DD_WARN_FRAC and not self._warned_dd:
                self._warned_dd = True
                self.notify(f"Drawdown al {used * 100:.0f} % del máximo: -{dd:.2f} % desde el máximo de equity "
                            f"{fmt_usd(self.risk.high_water or 0.0, sign=False)} (límite {s.max_drawdown_pct:g} %: cierra "
                            "todo y detiene la mesa).", logging.WARNING, kind="risk", key="dd_warn")
            elif self._warned_dd and used < DD_WARN_FRAC - WARN_REARM_FRAC:
                self._warned_dd = False

    def _stage_reconcile(self) -> None:
        closed = self.engine.reconcile(self._account)
        self._flush_engine()
        self._record_closed(closed)

    def auto_slots(self) -> list[AutoSlot]:
        """Lo que evalúa la operativa automática: las entradas activas del playbook (cada una con su temporalidad,
        parámetros y unidades) y, si ``auto_default_strategies``, las estrategias por defecto sobre ``symbols`` (salvo
        la misma estrategia y temporalidad que el playbook ya cubre en esa moneda). El playbook va primero."""
        slots: list[AutoSlot] = []
        taken: set[tuple[str, str]] = set()
        for entry, strat in self.playbook.strategies(self.settings, self.clock):
            units = entry.units if entry.units is not None else self.settings.auto_units
            slot = AutoSlot(entry.coin, strat, float(units), "playbook")
            slots.append(slot)
            taken.add((entry.coin.upper(), slot.label))
        if self.settings.auto_default_strategies:
            for coin in self.settings.symbols:
                for strat in self.strategies:
                    slot = AutoSlot(coin, strat, float(self.settings.auto_units), "defecto")
                    if (coin.upper(), slot.label) not in taken:
                        slots.append(slot)
        return slots

    def _stage_strategies(self) -> None:
        now = self.clock.now()
        if now - self._last_strategy_ts < self.settings.strategy_seconds:
            return
        self._last_strategy_ts = now
        if not self.auto_enabled:
            return
        slots = self.auto_slots()
        if not slots:
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
        by_coin: dict[str, list[AutoSlot]] = {}
        for slot in slots:
            by_coin.setdefault(slot.coin, []).append(slot)
        failures: list[str] = []
        for coin, coin_slots in by_coin.items():
            if coin in busy:
                continue
            try:
                opened = self._auto_coin(coin, coin_slots, now, failures)
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
        if failures:
            # Las demás estrategias y monedas ya se evaluaron; la etapa queda marcada como fallida (una vez por vela).
            raise RuntimeError("estrategias con error: " + "; ".join(failures))

    def _auto_coin(self, coin: str, slots: list[AutoSlot], now: float, failures: list[str]) -> bool:
        """Evalúa las estrategias de ``coin`` (una vez por vela cerrada cada una). True si abrió una operación.
        Una estrategia que lanza se apunta en ``failures`` y no frena a las demás."""
        due = []
        for slot in slots:
            sec = interval_seconds(slot.strategy.interval)
            bar = math.floor(now / sec) * sec            # apertura de la vela en curso
            if self._bar_seen.get((coin, slot.label)) != bar:
                due.append((slot, bar))
        if not due:
            return False
        mid = self.market.mid(coin)
        zones = self.scanner.zones(coin)
        candles_cache: dict[str, Any] = {}
        for slot, bar in due:
            strat = slot.strategy
            key = f"{strat.interval}:{strat.lookback}"
            if key not in candles_cache:
                candles_cache[key] = self.market.candles(coin, strat.interval, strat.lookback)
            self._bar_seen[(coin, slot.label)] = bar     # también si falla: se reintenta en la vela siguiente
            try:
                signal = strat.evaluate(coin, candles_cache[key], mid, zones)
            except (MarketDataError, BrokerError):
                raise
            except Exception as e:  # noqa: BLE001 - una estrategia rota no frena a las demás
                log.error("auto %s %s: la estrategia falló: %s", coin, slot.label, e, exc_info=True)
                failures.append(f"{coin} {slot.label}: {type(e).__name__}: {e}")
                continue
            if signal is None:
                continue
            if self._open_signal(signal, zones, slot):
                return True
        return False

    def _open_signal(self, sig: Signal, zones: dict, slot: Optional[AutoSlot] = None) -> bool:
        mode = sig.entry_mode
        label = slot.label if slot is not None else sig.strategy
        origin = slot.origin if slot is not None else f"auto:{sig.strategy}"
        units = slot.units if slot is not None else self.settings.auto_units
        intent = TradeIntent(
            coin=sig.coin, side=sig.side, units=units, entry_mode=mode,
            origin=origin, reason=sig.reason,
            limit_px=sig.entry_px if mode is not EntryMode.MARKET else None,
            sl_price=sig.sl_price, tp_price=sig.tp_price,
            zone=zones.get(sig.side) if mode is EntryMode.ZONE else None)
        interval = slot.strategy.interval if slot is not None else ""
        source = slot.source if slot is not None else ""
        self.journal.event("signal", coin=sig.coin, side=sig.side.value, strategy=sig.strategy, interval=interval,
                           origin=origin, source=source, reason=sig.reason, entry_px=sig.entry_px, sl=sig.sl_price,
                           tp=sig.tp_price, confidence=sig.confidence)
        ok, text = self._submit_intent(intent, f"auto-{label}")
        now = self.clock.now()
        self.signals_log.append({"ts": now, "time": self.clock.iso(), "coin": sig.coin, "side": sig.side.value,
                                 "strategy": label, "source": source, "reason": sig.reason, "opened": ok,
                                 "result": text.splitlines()[0][:200] if text else ""})
        self._snapshot_dirty = True
        if ok:
            self._account_dirty = True
            chart = f" · gráfica {sig.coin} {interval} ({source})" if slot is not None else ""
            self.notify(f"AUTO [{sig.strategy}] {text}\nMotivo: {sig.reason}{chart}", kind="signal", coin=sig.coin)
        else:
            log.info("AUTO [%s] señal %s %s descartada: %s", label, sig.coin, sig.side.value, text)
        return ok

    def _stage_alerts(self) -> None:
        """Alarmas del usuario (``AlertManager.evaluate`` nunca lanza) y frescura de los datos de las posiciones."""
        self.alerts.evaluate(self.market, self.scanner)
        self._check_stale_data()

    def _check_stale_data(self) -> None:
        """Aviso (warn, con anti-spam por moneda) si el precio de una moneda con posición abierta lleva más de
        ``max_data_age_seconds`` sin actualizarse según el exchange: los SL/TP siguen en el exchange, pero la mesa
        está mirando datos viejos. Cada ``STALE_CHECK_S``; sólo monedas con posición (las demás no se consultan)."""
        now = self.clock.now()
        if now - self._last_stale_check_ts < STALE_CHECK_S or self._account is None:
            return
        self._last_stale_check_ts = now
        limit = float(self.settings.max_data_age_seconds)
        for coin in sorted(c for c, p in self._account.positions.items() if p.size != 0):
            try:
                self.market.mid(coin)                 # refresca el libro: la edad es la del exchange
            except (MarketDataError, KeyError) as e:
                log.debug("frescura %s: sin precio (%s)", coin, e)
                continue
            age = self._data_age(coin)
            if age is not None and age > limit:
                self.notify(f"Datos viejos de {coin}: el último precio del exchange es de hace {age:.0f} s (límite "
                            f"{limit:g} s). La posición sigue protegida por su SL/TP en el exchange; las entradas "
                            "nuevas se rechazan mientras tanto.", logging.WARNING, kind="data", key=f"stale:{coin}",
                            coin=coin)

    def _stage_backtests(self) -> None:
        """Backtests terminados → aviso. Con reloj simulado, aquí se ejecutan los pendientes (síncrono)."""
        for out in self.backtests.poll():
            try:
                self._on_backtest_done(out)
            except Exception:  # noqa: BLE001 - un informe raro no tumba el ciclo
                log.exception("no se pudo avisar del backtest %s", out.job.label())

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
                self.notify(line, logging.WARNING, push=False)
                data = {"ts": now, "time": self.clock.iso(), "line": line, "report": line, "account_ok": False}
                data.update(self._status_extras())
                self.journal.save_json(STATUS_NAME, data)
                return
            line = self.status_line(acct)
            self.notify(line, push=False)        # periódica: pantalla y outbox, no Telegram
            self.write_status_snapshot(acct, line)
            return
        # El panel lee status.json: se refresca tras un cambio (comando, señal, cierre) o cada SNAPSHOT_EVERY_S.
        if acct is not None and (self._snapshot_dirty or now - self._last_snapshot_ts >= SNAPSHOT_EVERY_S):
            self.write_status_snapshot(acct)

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
        lines.append(f"Automático {'ON' if self.auto_enabled else 'OFF'} ({self._default_names()}) sobre "
                     f"{', '.join(s.symbols)} | {self.risk.describe()}")
        lines.append(self._playbook_line())
        lines.append(f"Dimensionamiento: 1 unidad = ${s.unit_margin_usd:,.0f} de margen x {s.leverage}x | riesgo "
                     f"máx. {s.risk_per_trade_pct:g}% por operación | SL por defecto {s.default_sl_pct:g}% | "
                     f"TP {s.default_tp_r:g}R")
        pos = [p for p in acct.positions.values() if p.size != 0]
        lines.append(f"Posiciones ({len(pos)}):" if pos else "Posiciones: ninguna")
        for p in sorted(pos, key=lambda p: p.coin):
            br = self.engine.brackets.get(p.coin)
            origin = self._origin_label(br.origin if br is not None else None)
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
        lines.append(self._alerts_line())
        bt = self.backtests.describe()
        if bt["running"] or bt["queued"]:
            pct = f" ({bt['progress_pct']:.0f} %)" if bt.get("progress_pct") is not None else ""
            lines.append(f"Backtests: en marcha {bt['running'] or '-'}{pct}"
                         + (f"; en cola {', '.join(bt['queued'])}" if bt["queued"] else ""))
        if self.stage_errors and any(self.stage_errors.values()):
            errs = ", ".join(f"{k} ({v})" for k, v in self.stage_errors.items() if v)
            lines.append(f"Etapas con fallos recientes: {errs}")
        return "\n".join(lines)

    def _default_names(self) -> str:
        if not self.strategies:
            return "sin estrategias por defecto"
        if not self.settings.auto_default_strategies:
            return "estrategias por defecto desactivadas"
        return ", ".join(st.name for st in self.strategies)

    def _playbook_line(self) -> str:
        entries = self.playbook.entries()
        if not entries:
            return "Playbook: vacío ('backtest COIN tf' y 'activa COIN tf' para añadir gráficas)"
        items = [f"{e.coin} {e.interval} {e.strategy}{'' if e.enabled else ' (pausada)'}" for e in entries]
        return f"Playbook ({len(entries)}): " + ", ".join(items)

    def _alerts_line(self) -> str:
        st = self.alerts.stats()
        text = f"Alarmas: {st['rules']} activa(s), {st['fired_total']} disparada(s) desde el arranque"
        if st["muted"]:
            until = f" hasta {_hhmm(st['muted_until'])}" if st.get("muted_until") else " hasta 'sonido'"
            text += f" | avisos SILENCIADOS{until}"
        text += f" | canales: {self.notifier.describe()}"
        if self.telegram is not None:
            text += f" | Telegram {'escuchando' if self.telegram.running else 'parado'}"
        return text

    def _origin_label(self, origin: Optional[str]) -> str:
        if origin is None:
            return "sin gestionar"
        return {"command": "comando", "adopted": "adoptada"}.get(origin, origin)

    def _status_extras(self) -> dict[str, Any]:
        """Campos de ``status.json`` que no dependen de la cuenta: playbook, alarmas, señales, backtests, avisos."""
        out: dict[str, Any] = {}
        for key, fn in (("playbook", lambda: [e.to_dict() for e in self.playbook.entries()]),
                        ("alerts", self.alerts.stats),
                        ("signals", lambda: list(self.signals_log)),
                        ("backtests", self.backtests.describe),
                        ("notifier", lambda: {"channels": self.notifier.describe(), "sent": dict(self.notifier.sent),
                                              "failed": dict(self.notifier.failed), "dropped": self.notifier.dropped,
                                              "last_error": self.notifier.last_error}),
                        ("telegram", lambda: None if self.telegram is None else {
                            "running": self.telegram.running, "handled": self.telegram.handled,
                            "ignored": self.telegram.ignored, "errors": self.telegram.errors,
                            "last_error": self.telegram.last_error})):
            try:
                out[key] = fn()
            except Exception as e:  # noqa: BLE001 - un campo raro no deja status.json sin escribir
                log.warning("status.json: campo %s no disponible: %s", key, e)
        return out

    def _status_fields(self, acct: AccountState) -> dict[str, Any]:
        """Campos estructurados de ``status.json`` (los nombres que reconoce ``tui.normalize_status``)."""
        s = self.settings
        eq = float(acct.equity)
        start = self.risk.day_start_equity
        day_pnl = eq - start if start else 0.0
        loss = max(self.risk.day_loss_pct(eq), 0.0)
        dd = self.risk.drawdown_pct(eq)
        positions = []
        for p in sorted((p for p in acct.positions.values() if p.size != 0), key=lambda p: p.coin):
            br = self.engine.brackets.get(p.coin)
            positions.append({"coin": p.coin, "side": p.side.value, "size": abs(p.size), "entry_px": p.entry_px,
                              "mark_px": p.mark_px, "unrealized_pnl": p.unrealized_pnl, "pnl_pct": p.pnl_pct(),
                              "leverage": p.leverage, "liq_px": p.liq_px, "sl": br.sl if br else None,
                              "tp": br.tp if br else None, "origin": self._origin_label(br.origin if br else None)})
        orders = [{"coin": o.coin, "is_buy": o.is_buy, "kind": o.kind.value, "size": o.remaining,
                   "trigger_px": o.trigger_px, "price": o.price, "reduce_only": o.reduce_only, "tag": o.tag}
                  for o in sorted((o for o in acct.open_orders if o.is_open), key=lambda o: (o.coin, o.kind.value))]
        pending = [{"coin": b.coin, "side": b.side.value, "planned_size": b.planned_size, "entry_px": b.entry_px,
                    "sl": b.sl, "tp": b.tp, "expires_ts": b.expires_ts or None, "origin": self._origin_label(b.origin)}
                   for b in sorted((b for b in self.engine.brackets.values() if b.status == PENDING),
                                   key=lambda b: b.coin)]
        halt = self.risk.active_halt() or ""
        data: dict[str, Any] = {
            "account_ok": True, "mode": s.mode, "data": s.data_source,
            "network": ("mainnet" if s.hl_mainnet else "testnet") if s.is_live else "",
            "equity": eq, "available": acct.available, "day_pnl": day_pnl,
            "day_pnl_pct": day_pnl / start * 100.0 if start else 0.0, "drawdown_pct": dd,
            "auto": self.auto_enabled, "paused": self.risk.paused,
            "cooldown": bool(self.auto_enabled and self.risk.in_cooldown), "halt": halt, "symbols": list(s.symbols),
            "risk": {"daily_loss_pct": loss, "daily_loss_limit_pct": s.daily_loss_limit_pct,
                     "daily_loss_used_pct": loss / s.daily_loss_limit_pct * 100.0 if s.daily_loss_limit_pct > 0 else None,
                     "drawdown_pct": dd, "max_drawdown_pct": s.max_drawdown_pct,
                     "drawdown_used_pct": dd / s.max_drawdown_pct * 100.0 if s.max_drawdown_pct > 0 else None,
                     "daily_loss_warn_frac": s.daily_loss_warn_frac},
            "positions": positions, "orders": orders, "pending": pending,
            "strategies": [{"name": st.name, "interval": st.interval} for st in self.strategies]
            if s.auto_default_strategies else [],
        }
        return data

    def write_status_snapshot(self, account: Optional[AccountState] = None, line: Optional[str] = None) -> None:
        """Guarda ``status.json`` (lo leen ``python -m desk status`` y el panel): ``line``/``report`` en texto y los
        campos estructurados de la fase 2 (posiciones, órdenes, consumo de límites, playbook, alarmas, señales...)."""
        fields: dict[str, Any] = {}
        try:
            acct = account if account is not None else self.broker.account()
            report = self.status_report(acct)
            line = line or self.status_line(acct)
            try:
                fields = self._status_fields(acct)
            except Exception as e:  # noqa: BLE001 - el texto se guarda aunque falle la parte estructurada
                log.warning("status.json sin campos estructurados: %s", e, exc_info=True)
        except (MarketDataError, BrokerError) as e:
            report, line = f"(estado no disponible: {e})", line or ""
        data = {"ts": self.clock.now(), "time": self.clock.iso(), "line": line, "report": report}
        data.update(fields)
        data.update(self._status_extras())
        self.journal.save_json(STATUS_NAME, data)
        self._last_snapshot_ts = self.clock.now()
        self._snapshot_dirty = False

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
        try:
            eq = fmt_usd(self.broker.account().equity, sign=False)
        except (MarketDataError, BrokerError) as e:
            eq = f"no disponible ({e})"
        lines = [f"Mesa en marcha: modo {s.mode.upper()}, datos {s.data_source}, símbolos {', '.join(s.symbols)}",
                 f"Automático {'ON' if self.auto_enabled else 'OFF'} ({self._default_names()}); ciclo "
                 f"{s.loop_seconds:g} s; equity {eq}; estado en {self.journal.dir}",
                 self._playbook_line(),
                 f"Riesgo: {s.risk_per_trade_pct:g}%/operación, pérdida diaria {s.daily_loss_limit_pct:g}%, drawdown "
                 f"máx. {s.max_drawdown_pct:g}%, máx. {s.max_open_positions} posiciones",
                 self._alerts_line()]
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
        if self.telegram is not None:
            try:
                started = self.telegram.start()
            except Exception as e:  # noqa: BLE001 - sin Telegram la mesa sigue
                log.error("Telegram: no se pudo arrancar el bot: %s", type(e).__name__)
                started = False
            if not started:
                self.notices.append("Telegram: el bot no arrancó (revisa TELEGRAM_BOT_TOKEN); los avisos siguen por "
                                    "los demás canales.")
        self.notify(self.banner(), kind="start")
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
        # Fuera del lock: el hilo de Telegram puede estar esperándolo dentro de ``submit``.
        if self.telegram is not None:
            try:
                self.telegram.stop(timeout=3.0)
            except Exception as e:  # noqa: BLE001
                log.error("al detener: Telegram: %s", type(e).__name__)
        try:
            canceled = self.backtests.stop()
        except Exception as e:  # noqa: BLE001
            log.error("al detener: backtests: %s", e)
            canceled = 0
        with self._lock:
            if canceled:
                self.notify(f"{canceled} backtest(s) en marcha o en cola cancelados al detener la mesa.", kind="backtest")
            for what, fn in (("estado", lambda: self.save_state(force=True)), ("snapshot", self.write_status_snapshot)):
                try:
                    fn()
                except Exception as e:  # noqa: BLE001
                    log.error("al detener: no se pudo guardar %s: %s", what, e)
            self.journal.event("stop", cycles=self.cycle, session=self.session)
            where = "en el exchange" if self.settings.is_live else "en el estado guardado del broker paper"
            self.notify(f"Mesa detenida tras {self.cycle} ciclos. Las órdenes SL/TP siguen {where}.", kind="stop")
        try:
            self.notifier.close(timeout=5.0)     # entrega lo que quede en cola (Telegram/webhook) antes de salir
        except Exception as e:  # noqa: BLE001
            log.error("al detener: avisos: %s", type(e).__name__)

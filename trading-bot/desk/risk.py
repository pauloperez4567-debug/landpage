"""Gestor de riesgo de la mesa: límites de pérdida, halts, pausa, cooldown y dimensionamiento.

Todo lo que *abre* exposición pasa por ``RiskManager.evaluate()``, venga de un comando o de una
estrategia. Las órdenes que reducen o cierran (cierres, SL/TP, flatten, kill) nunca pasan por aquí:
siempre están permitidas, incluso con la mesa detenida.

Estados que bloquean entradas nuevas:

- **halt "daily_loss"**: la pérdida del día UTC (equity frente al equity al inicio del día) alcanza
  ``daily_loss_limit_pct``. Se levanta solo al empezar el siguiente día UTC (o con RESET_HALT). Con
  ``flatten_on_daily_limit`` además pide cerrar todo.
- **halt "max_drawdown"**: la caída desde el máximo de equity (high-water mark) alcanza
  ``max_drawdown_pct``. Pide cerrar todo y sólo se levanta con RESET_HALT.
- **halt manual** (``halt()``, p. ej. "kill"): sólo se levanta con RESET_HALT.
- **pausa** (PAUSE / RESUME): RESUME no levanta un halt.
- **cooldown**: tras ``max_consecutive_losses`` pérdidas seguidas, ``loss_cooldown_minutes`` sin entradas
  *automáticas* (origen "auto:*"). Los comandos del usuario siguen permitidos.

``update()`` devuelve eventos (cadenas, ver ``EVENT_*``). Mientras un halt pida cierre, cada ``update()``
repite "flatten" si queda exposición (posiciones u órdenes de entrada): si un cierre falla, el siguiente
ciclo lo reintenta; una entrada que se ejecute tarde también se cierra. ``evaluate()`` comprueba además los
límites de pérdida diaria y drawdown con la cuenta que recibe (puede ir por delante del último ``update()``).
``apply_cash_flow()`` descuenta depósitos / retiradas del máximo y de la referencia diaria (la mesa los detecta).

Collar de precio: un límite que cruza el libro más de ``LIMIT_MAX_CROSS_PCT`` o reposa a más de
``LIMIT_MAX_DISTANCE_PCT`` del mid se rechaza (casi siempre es un precio mal escrito).

Dimensionamiento en ``evaluate()`` (ver ARCHITECTURE.md):
``notional = units · unit_margin_usd · leverage`` con ``leverage = min(settings.leverage, meta.max_leverage)``,
recortado en este orden por: riesgo en el SL (pérdida + comisiones taker estimadas de entrada y salida
≤ ``risk_per_trade_pct`` del equity), máximo por moneda, exposición bruta restante y margen disponible.
Cada recorte se explica en ``RiskDecision.warnings``. El tamaño final se redondea hacia abajo con
``round_size`` y se rechaza si queda por debajo del notional mínimo del exchange.

Comisión taker estimada = ``paper_taker_fee_bps`` (4.5 bps por defecto, la tarifa base de Hyperliquid),
también en live. Todas las horas salen del ``Clock`` inyectado.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

from .clock import Clock
from .config import Settings
from .exchange.base import round_price, round_size
from .models import AccountState, AssetMeta, EntryMode, Order, RiskDecision, Side, TradeIntent

log = logging.getLogger(__name__)

STATE_VERSION = 1

HALT_DAILY = "daily_loss"
HALT_DRAWDOWN = "max_drawdown"

EVENT_NEW_DAY = "new_day"                              # cambio de día UTC
EVENT_HALT_DAILY = f"halt:{HALT_DAILY}"                # se alcanzó el límite de pérdida diaria
EVENT_HALT_DRAWDOWN = f"halt:{HALT_DRAWDOWN}"          # se alcanzó el drawdown máximo
EVENT_HALT_LIFTED = f"halt_lifted:{HALT_DAILY}"        # día nuevo: el halt diario se levantó solo
EVENT_FLATTEN = "flatten"                              # la mesa debe cerrar todo y cancelar entradas
EVENT_COOLDOWN_START = "cooldown_start"                # racha de pérdidas: pausa de entradas automáticas
EVENT_COOLDOWN_END = "cooldown_end"

# La liquidación estimada debe quedar al menos un 10 % más lejos que el SL (margen para el
# deslizamiento del stop a mercado).
LIQ_BUFFER = 1.10

# Collar de precio de las entradas límite: un límite que cruza el libro más de esto, o que reposa más
# lejos del mid que esto, casi siempre es un precio mal escrito ("83.825" por 83,825): se rechaza en vez
# de convertirlo en una entrada a mercado o dejarlo en el libro.
LIMIT_MAX_CROSS_PCT = 1.0
LIMIT_MAX_DISTANCE_PCT = 20.0


def is_auto(origin: str) -> bool:
    """True si la intención viene de una estrategia autónoma (origen "auto" o "auto:<nombre>")."""
    return str(origin or "").strip().lower().startswith("auto")


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _hhmm(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def pending_entries(account: AccountState) -> list[Order]:
    """Órdenes abiertas que abrirían o aumentarían exposición (todo lo que no es reduce-only)."""
    return [o for o in account.open_orders if o.is_open and not o.reduce_only]


def estimated_liq_px(entry: float, leverage: int, is_long: bool, mmr: float) -> float:
    """Precio de liquidación aproximado en margen aislado (mismo modelo que ``PaperBroker``)."""
    if is_long:
        return max(entry * (1.0 - 1.0 / leverage) / (1.0 - mmr), 0.0)
    return entry * (1.0 + 1.0 / leverage) / (1.0 + mmr)


def maintenance_rate(settings: Settings, max_leverage: int) -> float:
    """Margen de mantenimiento estimado (fracción del notional) como en Hyperliquid: la mitad del margen
    inicial al apalancamiento máximo de la moneda, nunca menos que el del modelo paper."""
    return max(settings.paper_maintenance_margin_pct / 100.0, 1.0 / (2.0 * max(int(max_leverage), 1)))


class RiskManager:
    def __init__(self, settings: Settings, clock: Clock):
        self.settings = settings
        self.clock = clock
        self.day: str = ""                          # día UTC del último update()
        self.day_start_equity: Optional[float] = None
        self.high_water: Optional[float] = None     # máximo de equity visto (high-water mark)
        self.last_equity: Optional[float] = None
        self.consecutive_losses: int = 0
        self.cooldown_until: float = 0.0
        self.cooldown_detail: str = ""
        self.halted: bool = False
        self.halt_reason: str = ""                  # "daily_loss" | "max_drawdown" | manual ("kill", ...)
        self.halt_detail: str = ""
        self.halt_ts: float = 0.0
        self.flatten_required: bool = False         # el halt actual pide cerrar todo
        self.paused: bool = False

    # ------------------------------------------------------------------ parámetros
    @property
    def taker_rate(self) -> float:
        return max(self.settings.paper_taker_fee_bps, 0.0) / 10_000.0

    # ------------------------------------------------------------------ consultas
    def active_halt(self) -> Optional[str]:
        """Motivo del halt vigente o None. Un halt diario de un día anterior ya no cuenta
        (se levanta formalmente en el próximo ``update()``)."""
        if not self.halted:
            return None
        if self.halt_reason == HALT_DAILY and self.day and self.clock.utc_day() != self.day:
            return None
        return self.halt_reason

    @property
    def in_cooldown(self) -> bool:
        return self.clock.now() < self.cooldown_until

    def day_loss_pct(self, equity: Optional[float] = None) -> float:
        """Pérdida del día en % del equity inicial (negativo = ganancia)."""
        eq = self.last_equity if equity is None else equity
        start = self.day_start_equity
        if eq is None or not start or start <= 0:
            return 0.0
        return (start - eq) / start * 100.0

    def drawdown_pct(self, equity: Optional[float] = None) -> float:
        """Caída desde el máximo de equity, en %."""
        eq = self.last_equity if equity is None else equity
        hwm = self.high_water
        if eq is None or not hwm or hwm <= 0:
            return 0.0
        return max(hwm - eq, 0.0) / hwm * 100.0

    def block_reason(self, origin: str = "command") -> Optional[str]:
        """Texto con el motivo por el que una entrada de ``origin`` no se permite ahora, o None."""
        halt = self.active_halt()
        if halt:
            how = "se levanta solo el próximo día UTC o con 'reset halt'" if halt == HALT_DAILY \
                else "sólo se levanta con 'reset halt'"
            detail = f": {self.halt_detail}" if self.halt_detail else ""
            return f"mesa detenida por riesgo ({halt}{detail}); {how}"
        if self.paused:
            return "mesa en pausa: no se abren operaciones nuevas (usa 'reanuda')"
        if is_auto(origin) and self.in_cooldown:
            detail = f" ({self.cooldown_detail})" if self.cooldown_detail else ""
            return f"cooldown de entradas automáticas hasta {_hhmm(self.cooldown_until)}{detail}"
        return None

    def allows_auto(self) -> bool:
        """True si la operativa automática puede proponer entradas ahora."""
        return self.block_reason("auto") is None

    # ------------------------------------------------------------------ estado
    def update(self, account: AccountState) -> list[str]:
        """Actualiza día, máximo de equity, halts y cooldown con la cuenta actual. Devuelve eventos."""
        s = self.settings
        events: list[str] = []
        if not _finite(account.equity):
            log.warning("riesgo: equity no numérico (%r); se ignora esta lectura", account.equity)
            return events
        equity = float(account.equity)
        now = self.clock.now()
        today = self.clock.utc_day()

        if self.day != today:
            if self.day:
                events.append(EVENT_NEW_DAY)
            self.day = today
            self.day_start_equity = equity
            if self.halted and self.halt_reason == HALT_DAILY:
                self._lift()
                events.append(EVENT_HALT_LIFTED)
                log.info("riesgo: día nuevo (%s), halt por pérdida diaria levantado", today)
        if self.day_start_equity is None:
            self.day_start_equity = equity
        if self.high_water is None or equity > self.high_water:
            self.high_water = equity
        self.last_equity = equity

        if self.cooldown_until and now >= self.cooldown_until:
            self.cooldown_until, self.cooldown_detail = 0.0, ""
            events.append(EVENT_COOLDOWN_END)

        dd = self.drawdown_pct(equity)
        loss = self.day_loss_pct(equity)
        if self.halt_reason != HALT_DRAWDOWN and s.max_drawdown_pct > 0 and dd >= s.max_drawdown_pct:
            detail = (f"drawdown {dd:.2f}% desde el máximo {self.high_water:.2f} "
                      f"(límite {s.max_drawdown_pct:g}%)")
            self._set_halt(HALT_DRAWDOWN, detail, flatten=True)
            events.append(EVENT_HALT_DRAWDOWN)
        elif not self.halted and s.daily_loss_limit_pct > 0 and loss >= s.daily_loss_limit_pct:
            detail = (f"pérdida del día {loss:.2f}% desde {self.day_start_equity:.2f} "
                      f"(límite {s.daily_loss_limit_pct:g}%)")
            self._set_halt(HALT_DAILY, detail, flatten=bool(s.flatten_on_daily_limit))
            events.append(EVENT_HALT_DAILY)

        if self.halted and self.flatten_required and self._has_exposure(account):
            events.append(EVENT_FLATTEN)
        return events

    def record_trade_result(self, pnl: float) -> list[str]:
        """Registra el PnL de una operación cerrada. Una racha de pérdidas activa el cooldown."""
        s = self.settings
        events: list[str] = []
        if not _finite(pnl):
            return events
        pnl = float(pnl)
        if pnl < 0:
            self.consecutive_losses += 1
            n = int(s.max_consecutive_losses)
            if n > 0 and self.consecutive_losses >= n:
                self.cooldown_until = self.clock.now() + max(s.loss_cooldown_minutes, 0.0) * 60.0
                self.cooldown_detail = f"{self.consecutive_losses} pérdidas seguidas"
                self.consecutive_losses = 0
                events.append(EVENT_COOLDOWN_START)
                log.warning("riesgo: %s; sin entradas automáticas hasta %s",
                            self.cooldown_detail, _hhmm(self.cooldown_until))
        elif pnl > 0:
            self.consecutive_losses = 0
        return events

    def apply_cash_flow(self, amount: float) -> None:
        """Depósito (+) o retirada / transferencia (−) ajena a la operativa: se traslada al máximo de equity y
        a la referencia del día para que no cuente como ganancia ni como pérdida (ni dispare un halt)."""
        if not _finite(amount) or float(amount) == 0.0:
            return
        amount = float(amount)
        if self.high_water is not None:
            self.high_water = max(self.high_water + amount, 0.0)
        if self.day_start_equity is not None:
            self.day_start_equity = max(self.day_start_equity + amount, 0.0)
        log.warning("riesgo: movimiento de saldo ajeno a la operativa %+.2f USD; máximo y referencia diaria ajustados",
                    amount)

    def halt(self, reason: str, detail: str = "", *, flatten: bool = False) -> list[str]:
        """Halt manual (p. ej. "kill"). Sólo se levanta con ``reset_halt()``. Devuelve eventos."""
        self._set_halt(reason or "manual", detail, flatten=flatten)
        return [f"halt:{self.halt_reason}"]

    def reset_halt(self) -> str:
        """Levanta cualquier halt (y el cooldown) a petición del usuario.

        El máximo de equity y el equity de inicio del día se reinician al último equity conocido:
        sin esto el mismo drawdown / pérdida volvería a detener la mesa en el siguiente ciclo.
        """
        parts = []
        if self.halted:
            parts.append(f"halt '{self.halt_reason}' levantado")
            self._lift()
            if self.last_equity is not None:
                self.high_water = self.last_equity
                self.day_start_equity = self.last_equity
                parts.append(f"máximo de equity y referencia diaria reiniciados a {self.last_equity:.2f}")
        if self.cooldown_until or self.consecutive_losses:
            self.cooldown_until, self.cooldown_detail, self.consecutive_losses = 0.0, "", 0
            parts.append("cooldown reiniciado")
        if not parts:
            return "no había ningún halt activo"
        log.warning("riesgo: reset manual (%s)", "; ".join(parts))
        return "; ".join(parts)

    def pause(self) -> str:
        self.paused = True
        return "mesa en pausa: no se abrirán operaciones nuevas (las abiertas se siguen gestionando)"

    def resume(self) -> str:
        self.paused = False
        halt = self.active_halt()
        if halt:
            return f"pausa levantada, pero sigue el halt de riesgo '{halt}' (usa 'reset halt')"
        return "mesa reanudada"

    def _set_halt(self, reason: str, detail: str, *, flatten: bool) -> None:
        self.halted, self.halt_reason, self.halt_detail = True, reason, detail
        self.halt_ts = self.clock.now()
        self.flatten_required = bool(flatten)
        log.warning("riesgo: HALT %s%s%s", reason, f" ({detail})" if detail else "",
                    " -> cerrar todo" if flatten else "")

    def _lift(self) -> None:
        self.halted, self.halt_reason, self.halt_detail = False, "", ""
        self.halt_ts, self.flatten_required = 0.0, False

    @staticmethod
    def _has_exposure(account: AccountState) -> bool:
        return any(p.size != 0 for p in account.positions.values()) or bool(pending_entries(account))

    def describe(self) -> str:
        """Resumen de una línea para la línea de estado de la mesa."""
        parts = []
        halt = self.active_halt()
        if halt:
            parts.append(f"HALT {halt}" + (f" ({self.halt_detail})" if self.halt_detail else ""))
        if self.paused:
            parts.append("PAUSA")
        if self.in_cooldown:
            parts.append(f"cooldown auto hasta {_hhmm(self.cooldown_until)}")
        if self.last_equity is not None:
            parts.append(f"día {self.day} {-self.day_loss_pct():+.2f}%")
            parts.append(f"dd {self.drawdown_pct():.2f}% (máx. {self.high_water:.2f})")
        parts.append(f"pérdidas seguidas {self.consecutive_losses}/{self.settings.max_consecutive_losses}")
        return "riesgo: " + (" | ".join(parts) if parts else "OK")

    # ------------------------------------------------------------------ persistencia
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "day": self.day,
            "day_start_equity": self.day_start_equity,
            "high_water": self.high_water,
            "last_equity": self.last_equity,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until,
            "cooldown_detail": self.cooldown_detail,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "halt_detail": self.halt_detail,
            "halt_ts": self.halt_ts,
            "flatten_required": self.flatten_required,
            "paused": self.paused,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]], settings: Settings, clock: Clock) -> "RiskManager":
        """Reconstruye el estado guardado. Campos ausentes o ilegibles toman el valor por defecto
        (avisando); un halt guardado nunca se pierde por un campo secundario malformado."""
        rm = cls(settings, clock)
        if not isinstance(data, dict):
            if data is not None:
                log.error("riesgo: estado guardado ilegible (%r); se empieza de cero", type(data).__name__)
            return rm

        def get(key: str, conv: Any, default: Any) -> Any:
            if key not in data or data[key] is None:
                return default
            try:
                v = conv(data[key])
            except (TypeError, ValueError):
                log.error("riesgo: campo %s ilegible (%r); uso %r", key, data[key], default)
                return default
            if isinstance(v, float) and not math.isfinite(v):
                return default
            return v

        rm.day = get("day", str, "")
        rm.day_start_equity = get("day_start_equity", float, None)
        rm.high_water = get("high_water", float, None)
        rm.last_equity = get("last_equity", float, None)
        rm.consecutive_losses = max(get("consecutive_losses", int, 0), 0)
        rm.cooldown_until = get("cooldown_until", float, 0.0)
        rm.cooldown_detail = get("cooldown_detail", str, "")
        rm.halted = get("halted", _as_bool, False)
        rm.halt_reason = get("halt_reason", str, "")
        rm.halt_detail = get("halt_detail", str, "")
        rm.halt_ts = get("halt_ts", float, 0.0)
        rm.flatten_required = get("flatten_required", _as_bool, False)
        rm.paused = get("paused", _as_bool, False)
        if rm.halted and not rm.halt_reason:
            rm.halt_reason = "manual"
        if not rm.halted:
            rm.halt_reason, rm.halt_detail, rm.flatten_required = "", "", False
        return rm

    # ------------------------------------------------------------------ evaluación
    def evaluate(self, intent: TradeIntent, account: AccountState, meta: AssetMeta, bid: float, ask: float,
                 data_age_s: Optional[float]) -> RiskDecision:
        """Aprueba (con tamaño, SL y TP finales) o rechaza una intención de abrir posición."""
        s = self.settings
        warnings: list[str] = []
        coin = intent.coin
        try:
            side = Side(intent.side)
            mode = EntryMode(intent.entry_mode)
        except ValueError as e:
            return self._reject(intent, f"intención inválida: {e}", warnings)
        is_long = side is Side.LONG
        szd = int(meta.sz_decimals)

        # --- bloqueos de estado y de cartera ---------------------------------------
        blocked = self.block_reason(intent.origin)
        if blocked:
            return self._reject(intent, blocked, warnings)
        # La cuenta que llega puede ir por delante del último update() (comandos entre ciclos, o antes de la
        # etapa de riesgo): los límites de pérdida se comprueban también con ella.
        if _finite(account.equity):
            eq = float(account.equity)
            dd = self.drawdown_pct(eq)
            if s.max_drawdown_pct > 0 and dd >= s.max_drawdown_pct:
                return self._reject(intent, f"drawdown {dd:.2f}% desde el máximo {self.high_water:.2f} (límite "
                                            f"{s.max_drawdown_pct:g}%): la mesa se detendrá en este ciclo", warnings)
            loss = self.day_loss_pct(eq)
            if (s.daily_loss_limit_pct > 0 and self.day == self.clock.utc_day()
                    and loss >= s.daily_loss_limit_pct):
                return self._reject(intent, f"pérdida del día {loss:.2f}% (límite {s.daily_loss_limit_pct:g}%): sin "
                                            "entradas nuevas hasta el próximo día UTC", warnings)
        if meta.coin != coin:
            return self._reject(intent, f"metadatos de {meta.coin} para una orden de {coin}", warnings)
        units = float(intent.units) if _finite(intent.units) else 0.0
        if units <= 0:
            return self._reject(intent, f"unidades inválidas: {intent.units!r}", warnings)
        if s.max_units_per_trade > 0 and units > s.max_units_per_trade:
            warnings.append(f"unidades recortadas de {units:g} a {s.max_units_per_trade:g} (máximo por operación)")
            units = float(s.max_units_per_trade)

        pos = account.positions.get(coin)
        if pos is not None and pos.size != 0:
            return self._reject(intent, f"ya hay una posición abierta en {coin} ({pos.size:+g}); "
                                        "ciérrala o ajústala antes de abrir otra", warnings)
        entries = pending_entries(account)
        if any(o.coin == coin for o in entries):
            return self._reject(intent, f"ya hay una orden de entrada pendiente en {coin} "
                                        "(cancélala antes de abrir otra)", warnings)
        busy = {c for c, p in account.positions.items() if p.size != 0} | {o.coin for o in entries}
        if len(busy) >= s.max_open_positions:
            return self._reject(intent, f"máximo de posiciones alcanzado ({len(busy)}/{s.max_open_positions}, "
                                        "contando entradas pendientes)", warnings)

        # --- calidad del mercado ---------------------------------------------------
        if not (_finite(bid) and _finite(ask)) or bid <= 0 or ask <= 0 or bid > ask:
            return self._reject(intent, f"precio inválido para {coin}: bid={bid} ask={ask}", warnings)
        bid, ask = float(bid), float(ask)
        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / mid * 10_000.0
        if spread_bps > s.max_spread_bps:
            return self._reject(intent, f"spread de {coin} demasiado amplio: {spread_bps:.1f} bps "
                                        f"(máximo {s.max_spread_bps:g} bps)", warnings)
        if data_age_s is None:
            warnings.append("antigüedad de los datos desconocida")
        elif not _finite(data_age_s) or data_age_s > s.max_data_age_seconds:
            return self._reject(intent, f"datos de {coin} viejos: {data_age_s} s "
                                        f"(máximo {s.max_data_age_seconds:g} s)", warnings)
        equity = float(account.equity) if _finite(account.equity) else 0.0
        if equity <= 0:
            return self._reject(intent, f"equity no positivo ({account.equity}); no se abre nada", warnings)

        # --- precio de entrada -----------------------------------------------------
        if mode is EntryMode.MARKET:
            entry = ask if is_long else bid
        else:
            if mode is EntryMode.ZONE and intent.zone is not None:
                if intent.zone.side != side:
                    return self._reject(intent, f"la zona es para {intent.zone.side.value.upper()} y la orden "
                                                f"es {side.value.upper()}", warnings)
                entry = intent.zone.mid
            elif intent.limit_px is not None and _finite(intent.limit_px) and intent.limit_px > 0:
                entry = float(intent.limit_px)
            elif mode is EntryMode.ZONE:
                return self._reject(intent, f"entrada en zona sin zona de liquidez para {coin}", warnings)
            else:
                return self._reject(intent, "entrada LIMIT sin precio límite válido", warnings)
            if is_long and entry >= ask:
                cross = (entry - ask) / ask * 100.0
                if cross > LIMIT_MAX_CROSS_PCT:
                    return self._reject(intent, f"el límite {entry:g} está un {cross:.1f}% por encima del ask {ask:g} "
                                                "(¿precio mal escrito?); para entrar ya usa 'a mercado'", warnings)
                warnings.append(f"el límite {entry:g} cruza el ask {ask:g}: se ejecutará al instante (taker) a {ask:g}")
                entry = ask
            elif not is_long and entry <= bid:
                cross = (bid - entry) / bid * 100.0
                if cross > LIMIT_MAX_CROSS_PCT:
                    return self._reject(intent, f"el límite {entry:g} está un {cross:.1f}% por debajo del bid {bid:g} "
                                                "(¿precio mal escrito?); para entrar ya usa 'a mercado'", warnings)
                warnings.append(f"el límite {entry:g} cruza el bid {bid:g}: se ejecutará al instante (taker) a {bid:g}")
                entry = bid
            elif abs(entry - mid) / mid * 100.0 > LIMIT_MAX_DISTANCE_PCT:
                dist = abs(entry - mid) / mid * 100.0
                return self._reject(intent, f"el límite {entry:g} está a un {dist:.1f}% del precio actual ({mid:g}; "
                                            f"máximo {LIMIT_MAX_DISTANCE_PCT:g}%): ¿precio mal escrito?", warnings)
        entry = round_price(entry, szd)

        # --- stop-loss ----------------------------------------------------------------
        # Para MARKET el SL también debe quedar fuera del precio de salida actual (si no, saltaría al momento).
        sl_ref = entry if mode is not EntryMode.MARKET else (min(entry, bid) if is_long else max(entry, ask))
        sl: Optional[float] = None
        if intent.sl_price is not None:
            if not _finite(intent.sl_price) or intent.sl_price <= 0:
                return self._reject(intent, f"SL inválido: {intent.sl_price!r}", warnings, entry_px=entry)
            sl = float(intent.sl_price)
            if intent.sl_pct is not None:
                warnings.append("se indicó SL en precio y en %: se usa el precio")
        elif intent.sl_pct is not None:
            if not _finite(intent.sl_pct) or intent.sl_pct <= 0:
                return self._reject(intent, f"SL en % inválido: {intent.sl_pct!r}", warnings, entry_px=entry)
            sl = entry * (1.0 - intent.sl_pct / 100.0) if is_long else entry * (1.0 + intent.sl_pct / 100.0)
        elif s.require_stop:
            sl = entry * (1.0 - s.default_sl_pct / 100.0) if is_long else entry * (1.0 + s.default_sl_pct / 100.0)
            warnings.append(f"SL por defecto a {s.default_sl_pct:g}% de la entrada")
        if sl is not None:
            if sl <= 0:
                return self._reject(intent, f"SL fuera de rango ({sl:g})", warnings, entry_px=entry)
            sl = round_price(sl, szd)
            if is_long and sl >= sl_ref:
                return self._reject(intent, f"SL del lado equivocado: en un LONG debe quedar por debajo de "
                                            f"{sl_ref:g} (SL {sl:g})", warnings, entry_px=entry)
            if not is_long and sl <= sl_ref:
                return self._reject(intent, f"SL del lado equivocado: en un SHORT debe quedar por encima de "
                                            f"{sl_ref:g} (SL {sl:g})", warnings, entry_px=entry)
        else:
            warnings.append("sin stop-loss: la pérdida de esta operación no está acotada")

        # --- take-profit --------------------------------------------------------------
        tp: Optional[float] = None
        if intent.tp_price is not None:
            if not _finite(intent.tp_price) or intent.tp_price <= 0:
                return self._reject(intent, f"TP inválido: {intent.tp_price!r}", warnings, entry_px=entry)
            tp = float(intent.tp_price)
            if intent.tp_pct is not None:
                warnings.append("se indicó TP en precio y en %: se usa el precio")
        elif intent.tp_pct is not None:
            if not _finite(intent.tp_pct) or intent.tp_pct <= 0:
                return self._reject(intent, f"TP en % inválido: {intent.tp_pct!r}", warnings, entry_px=entry)
            tp = entry * (1.0 + intent.tp_pct / 100.0) if is_long else entry * (1.0 - intent.tp_pct / 100.0)
        elif sl is not None and s.default_tp_r > 0:
            dist = abs(entry - sl) * s.default_tp_r
            tp = entry + dist if is_long else entry - dist
            if tp <= 0:
                tp = None
                warnings.append("TP por defecto fuera de rango: sin TP")
        if tp is not None:
            if tp <= 0:
                return self._reject(intent, f"TP fuera de rango ({tp:g})", warnings, entry_px=entry, sl_price=sl)
            tp = round_price(tp, szd)
            if is_long and tp <= entry:
                return self._reject(intent, f"TP del lado equivocado: en un LONG debe quedar por encima de la "
                                            f"entrada {entry:g} (TP {tp:g})", warnings, entry_px=entry, sl_price=sl)
            if not is_long and tp >= entry:
                return self._reject(intent, f"TP del lado equivocado: en un SHORT debe quedar por debajo de la "
                                            f"entrada {entry:g} (TP {tp:g})", warnings, entry_px=entry, sl_price=sl)

        # --- apalancamiento -------------------------------------------------------------
        max_lev = max(int(meta.max_leverage), 1)
        lev = max(1, min(int(s.leverage), max_lev))
        if lev < s.leverage:
            warnings.append(f"apalancamiento recortado de {s.leverage}x a {lev}x (máximo de {coin})")
        if sl is not None:
            mmr = maintenance_rate(s, max_lev)
            sl_dist = abs(entry - sl)
            safe = lev
            while safe >= 1 and abs(entry - estimated_liq_px(entry, safe, is_long, mmr)) < sl_dist * LIQ_BUFFER:
                safe -= 1
            if safe < 1:
                return self._reject(intent, f"el SL {sl:g} está más allá de la liquidación incluso a 1x",
                                    warnings, entry_px=entry, sl_price=sl, tp_price=tp)
            if safe < lev:
                liq = estimated_liq_px(entry, lev, is_long, mmr)
                warnings.append(f"apalancamiento bajado de {lev}x a {safe}x: a {lev}x la liquidación (~{liq:g}) "
                                f"llegaría antes que el SL {sl:g}")
                lev = safe

        # --- dimensionamiento -------------------------------------------------------------
        notional = units * s.unit_margin_usd * lev
        taker = self.taker_rate
        loss_per_coin: Optional[float] = None
        if sl is not None:
            loss_per_coin = abs(entry - sl) + taker * (entry + sl)
            max_loss = equity * s.risk_per_trade_pct / 100.0
            cap = max_loss / loss_per_coin * entry
            if notional > cap:
                loss = notional / entry * loss_per_coin
                warnings.append(f"recortado por riesgo: con SL en {sl:g} perdería {loss:.2f} USD (con comisiones) "
                                f"> {max_loss:.2f} USD ({s.risk_per_trade_pct:g}% del equity); "
                                f"notional {notional:.2f} -> {cap:.2f} USD")
                notional = cap
        if notional > s.max_notional_per_coin_usd:
            warnings.append(f"recortado por máximo por moneda: notional {notional:.2f} -> "
                            f"{s.max_notional_per_coin_usd:.2f} USD")
            notional = float(s.max_notional_per_coin_usd)
        pending_usd = sum(o.remaining * (o.price or o.trigger_px or 0.0) for o in entries)
        gross_room = s.max_gross_notional_usd - account.gross_notional - pending_usd
        if gross_room < meta.min_notional_usd:
            return self._reject(intent, f"exposición bruta máxima alcanzada ({account.gross_notional + pending_usd:.2f} "
                                        f"de {s.max_gross_notional_usd:.2f} USD, incluidas entradas pendientes)",
                                warnings, leverage=lev, entry_px=entry, sl_price=sl, tp_price=tp)
        if notional > gross_room:
            warnings.append(f"recortado por exposición bruta: quedan {gross_room:.2f} USD de "
                            f"{s.max_gross_notional_usd:.2f}; notional {notional:.2f} -> {gross_room:.2f} USD")
            notional = gross_room
        available = float(account.available) if _finite(account.available) else 0.0
        if available <= 0:
            return self._reject(intent, f"sin margen disponible ({account.available})", warnings,
                                leverage=lev, entry_px=entry, sl_price=sl, tp_price=tp)
        # El margen (notional/lev) más la comisión de entrada deben caber en el disponible.
        margin_cap = available / (1.0 / lev + taker)
        if notional > margin_cap:
            warnings.append(f"recortado por margen disponible ({available:.2f} USD a {lev}x): "
                            f"notional {notional:.2f} -> {margin_cap:.2f} USD")
            notional = margin_cap

        size = round_size(notional / entry, szd)
        final_notional = size * entry
        if size <= 0 or final_notional < meta.min_notional_usd - 1e-9:
            return self._reject(intent, f"tamaño resultante {size:g} {coin} ({final_notional:.2f} USD) por debajo del "
                                        f"mínimo de {meta.min_notional_usd:g} USD", warnings,
                                leverage=lev, entry_px=entry, sl_price=sl, tp_price=tp)

        reason = (f"aprobada: {side.value.upper()} {coin} {size:g} @ {entry:g} ({lev}x, {final_notional:.2f} USD, "
                  f"margen {final_notional / lev:.2f} USD)")
        if sl is not None and loss_per_coin is not None:
            loss = size * loss_per_coin
            reason += f"; SL {sl:g} (riesgo {loss:.2f} USD = {loss / equity * 100.0:.2f}% del equity)"
        if tp is not None:
            reason += f"; TP {tp:g}"
        log.info("riesgo %s %s [%s]: %s", coin, side.value, intent.origin, reason)
        return RiskDecision(approved=True, reason=reason, leverage=lev, size=size, notional=final_notional,
                            entry_px=entry, sl_price=sl, tp_price=tp, warnings=warnings)

    @staticmethod
    def _reject(intent: TradeIntent, reason: str, warnings: list[str], **kw: Any) -> RiskDecision:
        log.info("riesgo %s %s [%s] rechazada: %s", intent.coin, getattr(intent.side, "value", intent.side),
                 intent.origin, reason)
        return RiskDecision(approved=False, reason=reason, warnings=warnings, **kw)


def _as_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "si", "sí", "on"}
    return bool(v)

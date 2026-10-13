"""Alarmas de mercado (las crea el usuario) y avisos de la mesa (halts, ejecuciones...) con anti-spam.

Tipos de alarma (``Command.alert = {"kind": ..., <parámetros>, "interval": ..., "repeat": bool}``):

==================  ==============================================  =====================================
tipo                parámetros (por defecto)                        dato que se vigila
==================  ==============================================  =====================================
``price_above``     ``level``                                       precio actual (mid), cada ciclo
``price_below``     ``level``                                       precio actual (mid), cada ciclo
``price_cross``     ``level``                                       precio actual (mid), cada ciclo
``rsi_below``       ``level``, ``period`` (14)                      RSI de velas cerradas de ``interval``
``rsi_above``       ``level``, ``period`` (14)                      RSI de velas cerradas de ``interval``
``ma_cross``        ``fast`` (20), ``slow`` (50), ``ma`` ("sma"),   medias de velas cerradas
                    ``direction`` ("any": "up" | "down" | "any")
``zone``            ``side`` (None = las dos)                       zona de liquidez del ``LiquidityScanner``
``volume_spike``    ``mult`` (3), ``lookback`` (20)                 volumen de la última vela cerrada
``move_pct``        ``pct``, ``direction`` ("any")                  cierre contra cierre en una vela
==================  ==============================================  =====================================

Opcionales en cualquier tipo: ``interval`` (velas; por defecto "1h"), ``repeat`` (False), ``cooldown_s``
(``alert_cooldown_seconds``), ``note``, ``severity`` (nivel del aviso: "warn"; "critical" no se silencia) e
``immediate`` (False).

Semántica (lo que el usuario puede esperar):

- **Cruces, no estados.** ``price_above``, ``price_below``, ``price_cross``, ``rsi_*`` y ``ma_cross`` disparan
  cuando el valor *pasa* al otro lado del nivel respecto al valor previo observado: ``price_above 100`` creada
  con el precio en 105 no dispara; dispara cuando el precio baja a <= 100 y vuelve a superar 100. Para que
  dispare en cuanto se cumpla la condición (también si ya se cumple al crearla) se pide ``immediate: True``
  ("avísame si está por encima"). ``price_cross`` vigila los dos sentidos e ignora ``immediate``.
- La referencia inicial de las alarmas de precio es el ``ref_px`` que pase quien la crea (el mid del momento:
  un cruce entre crear la alarma y la primera evaluación no se pierde) o, sin él, el primer precio evaluado.
  Las de velas toman como referencia la última vela cerrada *antes* de crear la alarma: sólo cuentan las
  velas que cierran después.
- **Velas cerradas.** Los indicadores se calculan con velas cerradas de ``interval`` (la vela en curso se
  descarta) y cada alarma se evalúa una vez por vela; el mercado se consulta una vez por ciclo y moneda
  (caché por ciclo) y sólo cuando alguna alarma tiene una vela nueva pendiente.
- ``zone``: el precio entra en la zona de liquidez vista en el ciclo anterior (LONG: precio <= borde
  superior de la zona; SHORT: precio >= borde inferior). Las zonas se recalculan cada ciclo con el precio.
- ``volume_spike``: volumen de la vela >= ``mult`` veces la media de las ``lookback`` anteriores.
- ``move_pct``: |cierre / cierre anterior - 1| >= ``pct`` % (la ventana es una vela de ``interval``).
- ``repeat=False`` (por defecto): dispara una vez y se elimina. ``repeat=True``: puede volver a disparar,
  nunca antes de ``cooldown_s`` desde el último aviso (un cruce dentro de la pausa se descarta).
- ``mute(minutos)`` silencia los avisos no críticos (se siguen registrando en ``alerts.jsonl``); los críticos
  nunca se silencian. Al terminar el silencio se avisa de cuántos se silenciaron.
- ``system(kind, text, level, key)``: avisos de la mesa; con ``key``, el mismo aviso no se repite antes de
  ``alert_cooldown_seconds`` (un crítico con texto distinto sí pasa: trae información nueva).

Persistencia: reglas en ``state/alerts.json`` (escritura atómica); cada aviso entregado o silenciado en
``state/alerts.jsonl`` (``{"ts","time","level","kind","coin","text","rule_id"}`` y ``"muted": true`` si estaba
silenciado), que leen el panel y la consola. ``evaluate`` nunca lanza: datos malos o una API caída sólo
dejan la alarma sin evaluar ese ciclo (y se registra en el log, sin repetir el mismo error cada ciclo).
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np
import pandas as pd

from .clock import Clock
from .config import Settings
from .engine import fmt_px, fmt_usd
from .exchange.intervals import interval_seconds
from .indicators import ema, rsi, sma
from .journal import Journal
from .liquidity import LiquidityScanner
from .models import Side
from .notify import normalize_level

log = logging.getLogger(__name__)

STATE_NAME = "alerts"                 # state/alerts.json
LOG_FILE = "alerts.jsonl"
LOG_MAX_BYTES = 5_000_000             # al superarlo, alerts.jsonl -> alerts.jsonl.1 (se guarda una copia)
STATE_VERSION = 1
MAX_RULES = 200
MAX_BARS = 1000                       # velas pedidas como mucho para una alarma
MAX_MUTE_MINUTES = 30 * 24 * 60
DEFAULT_INTERVAL = "1h"
ZONE_MAX_AGE_S = 900.0                # una zona vista hace más de esto no sirve de referencia
WARN_EVERY_S = 600.0                  # el mismo error de datos se registra como mucho cada 10 min
SYSTEM_KEYS_KEPT = 1000

_REQUIRED = object()

KINDS: dict[str, dict[str, Any]] = {
    "price_above": {"level": _REQUIRED},
    "price_below": {"level": _REQUIRED},
    "price_cross": {"level": _REQUIRED},
    "rsi_below": {"level": _REQUIRED, "period": 14},
    "rsi_above": {"level": _REQUIRED, "period": 14},
    "ma_cross": {"fast": 20, "slow": 50, "ma": "sma", "direction": "any"},
    "zone": {"side": None},
    "volume_spike": {"mult": 3.0, "lookback": 20},
    "move_pct": {"pct": _REQUIRED, "direction": "any"},
}
PRICE_KINDS = frozenset({"price_above", "price_below", "price_cross"})
CANDLE_CROSS_KINDS = frozenset({"rsi_below", "rsi_above", "ma_cross"})
CANDLE_EVENT_KINDS = frozenset({"volume_spike", "move_pct"})
CANDLE_KINDS = CANDLE_CROSS_KINDS | CANDLE_EVENT_KINDS

_DIRECTIONS = {"up": "up", "alza": "up", "al alza": "up", "sube": "up",
               "down": "down", "baja": "down", "a la baja": "down", "cae": "down",
               "any": "any", "both": "any", "cualquiera": "any", "ambos": "any"}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _hhmm(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%d/%m %H:%M UTC")


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _as_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "si", "sí", "on"}
    return bool(v)


def _sign(x: Optional[float]) -> Optional[int]:
    if x is None or not math.isfinite(x):
        return None
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _crossed(direction: str, prev: Optional[int], cur: int) -> bool:
    """Cruce respecto al valor previo. ``up``: de <= 0 a > 0; ``down``: de >= 0 a < 0; ``any``: cambio de signo."""
    if prev is None:
        return False
    if direction == "up":
        return prev <= 0 < cur
    if direction == "down":
        return prev >= 0 > cur
    return prev != 0 and cur != 0 and prev != cur


def _store_sign(state: dict, direction: str, cur: Optional[int]) -> bool:
    """Guarda el lado actual (``any`` sólo recuerda lados no nulos). True si cambió."""
    if cur is None or (direction == "any" and cur == 0) or state.get("s") == cur:
        return False
    state["s"] = cur
    return True


def _armed(direction: str) -> Optional[int]:
    """Referencia de ``immediate``: el lado contrario, para que la primera evaluación dispare si ya se cumple."""
    return {"up": -1, "down": 1}.get(direction)


def _ts(t: Any) -> float:
    return float(pd.Timestamp(t).timestamp())


@dataclass
class AlertRule:
    id: int
    coin: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    interval: str = ""
    repeat: bool = False
    cooldown_s: float = 300.0
    created_ts: float = 0.0
    last_fired_ts: Optional[float] = None
    fired_count: int = 0
    note: str = ""
    severity: str = "warn"               # nivel de los avisos que genera ("info" | "warn" | "critical")
    state: dict[str, Any] = field(default_factory=dict)   # referencia de cruces ("s") y última vela ("bar")

    @property
    def direction(self) -> str:
        if self.kind in ("price_above", "rsi_above"):
            return "up"
        if self.kind in ("price_below", "rsi_below"):
            return "down"
        return str(self.params.get("direction") or "any")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AlertRule":
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in dict(d).items() if k in known}
        rule = cls(**data)
        rule.id = int(rule.id)
        rule.coin = str(rule.coin).upper()
        rule.params = dict(rule.params or {})
        rule.state = dict(rule.state or {})
        rule.cooldown_s = float(rule.cooldown_s)
        rule.created_ts = float(rule.created_ts or 0.0)
        rule.fired_count = int(rule.fired_count or 0)
        rule.severity = normalize_level(rule.severity, default="warn")
        if rule.kind not in KINDS:
            raise ValueError(f"tipo de alarma desconocido: {rule.kind!r}")
        return rule


def describe_rule(rule: AlertRule) -> str:
    """Una línea legible: ``#3 BTC precio > 100000 · una vez``."""
    p, k = rule.params, rule.kind
    if k == "price_above":
        cond = f"precio > {fmt_px(p.get('level'))}"
    elif k == "price_below":
        cond = f"precio < {fmt_px(p.get('level'))}"
    elif k == "price_cross":
        cond = f"precio cruza {fmt_px(p.get('level'))}"
    elif k in ("rsi_below", "rsi_above"):
        cond = f"RSI({p.get('period')}) {rule.interval} {'<' if k == 'rsi_below' else '>'} {p.get('level'):g}"
    elif k == "ma_cross":
        ma = str(p.get("ma", "sma")).upper()
        way = {"up": " al alza", "down": " a la baja"}.get(rule.direction, "")
        cond = f"cruce{way} {ma}({p.get('fast')})/{ma}({p.get('slow')}) {rule.interval}"
    elif k == "zone":
        side = p.get("side")
        cond = f"entra en zona de liquidez {side.upper() if side else 'LONG o SHORT'}"
    elif k == "volume_spike":
        cond = f"volumen {rule.interval} >= {p.get('mult'):g}x la media de {p.get('lookback')} velas"
    elif k == "move_pct":
        sign = {"up": "+", "down": "-"}.get(rule.direction, "±")
        cond = f"movimiento {sign}{p.get('pct'):g}% en una vela de {rule.interval}"
    else:
        cond = k
    if p.get("immediate") and k != "price_cross":
        cond += " (ya cumplida también avisa)"
    mode = f"repetir (pausa {rule.cooldown_s:g} s)" if rule.repeat else "una vez"
    out = f"#{rule.id} {rule.coin} {cond} · {mode}"
    if rule.fired_count:
        out += f" · disparada {rule.fired_count}x"
        if rule.last_fired_ts:
            out += f", última {_hhmm(rule.last_fired_ts)}"
    if rule.note:
        out += f" · {rule.note}"
    return out


def _closed_bars(df: Any, step: int, now: float) -> Optional[pd.DataFrame]:
    """Velas cerradas con columnas numéricas, ordenadas y sin duplicados; None si no hay datos utilizables."""
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or "close" not in df.columns:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")].sort_index()
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    bars = df[cols].apply(pd.to_numeric, errors="coerce")
    if _ts(bars.index[-1]) + step > now:      # vela en curso
        bars = bars.iloc[:-1]
    return bars if len(bars) else None


class _Cycle:
    """Caché de una evaluación: cada dato de mercado se pide como mucho una vez por ciclo."""

    def __init__(self, market: Any, scanner: Callable[[], Any], now: float, limits: dict[tuple[str, str], int],
                 warn: Callable[[str, str], None]):
        self.market = market
        self._scanner = scanner
        self.now = now
        self.limits = limits
        self._warn = warn
        self._mid: dict[str, Optional[float]] = {}
        self._bars: dict[tuple[str, str], Optional[pd.DataFrame]] = {}
        self._zones: dict[str, Optional[dict]] = {}
        self._memo: dict[tuple, Any] = {}

    def mid(self, coin: str) -> Optional[float]:
        if coin not in self._mid:
            px: Optional[float] = None
            try:
                px = _num(self.market.mid(coin))
                if px is not None and px <= 0:
                    px = None
            except Exception as e:  # noqa: BLE001 - un dato malo nunca tumba las alarmas
                self._warn(f"mid:{coin}", f"alarmas: sin precio de {coin} ({type(e).__name__}: {e})")
            self._mid[coin] = px
        return self._mid[coin]

    def bars(self, coin: str, interval: str) -> Optional[pd.DataFrame]:
        key = (coin, interval)
        if key not in self._bars:
            bars = None
            try:
                limit = self.limits.get(key, 10)
                raw = self.market.candles(coin, interval, limit)
                bars = _closed_bars(raw, interval_seconds(interval), self.now)
                if bars is None:
                    self._warn(f"bars:{coin}:{interval}", f"alarmas: velas {coin} {interval} vacías o ilegibles")
            except Exception as e:  # noqa: BLE001
                self._warn(f"bars:{coin}:{interval}",
                           f"alarmas: sin velas {coin} {interval} ({type(e).__name__}: {e})")
            self._bars[key] = bars
        return self._bars[key]

    def zones(self, coin: str) -> Optional[dict]:
        if coin not in self._zones:
            zs = None
            try:
                scanner = self._scanner()
                zs = scanner.zones(coin) if scanner is not None else None
                if not isinstance(zs, Mapping):
                    zs = None
            except Exception as e:  # noqa: BLE001
                self._warn(f"zones:{coin}", f"alarmas: sin zonas de liquidez de {coin} ({type(e).__name__}: {e})")
            self._zones[coin] = zs
        return self._zones[coin]

    def memo(self, key: tuple, fn: Callable[[], Any]) -> Any:
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]


class AlertManager:
    """Alarmas del usuario + avisos de la mesa. Seguro entre hilos (la consola o Telegram pueden crear alarmas
    mientras el bucle las evalúa). Ningún método público lanza salvo ``add``/``mute`` ante una especificación
    inválida (``ValueError`` con el motivo, para responder al usuario)."""

    def __init__(self, settings: Settings, clock: Clock, state_dir: Union[str, Path], notifier: Any = None, *,
                 max_rules: int = MAX_RULES):
        self.settings = settings
        self.clock = clock
        self.notifier = notifier
        self.max_rules = int(max_rules)
        self.dir = Path(state_dir)
        self.journal = Journal(self.dir, clock)
        self.log_path = self.dir / LOG_FILE
        self._lock = threading.RLock()
        self._rules: dict[int, AlertRule] = {}
        self._next_id = 1
        self.muted_until: Optional[float] = None      # None = no silenciado; inf = hasta ``unmute``
        self.silenced = 0                              # avisos silenciados en el silencio actual
        self.fired_total = 0                           # alarmas disparadas desde el arranque
        self._dirty = False
        self._zone_seen: dict[int, tuple[float, dict[str, Optional[tuple]]]] = {}
        self._sys_seen: dict[str, list] = {}           # clave -> [ts, texto, omitidos]
        self._warned: dict[str, float] = {}
        self._scanner: Optional[LiquidityScanner] = None
        self._load()

    # ================================================================== reglas
    def add(self, coin: str, spec: Mapping[str, Any], *, ref_px: Optional[float] = None) -> AlertRule:
        """Crea una alarma. ``ref_px`` (opcional): precio actual, referencia de los cruces de precio.

        Lanza ``ValueError`` (mensaje para el usuario) si la especificación no es válida."""
        rule = self._build(coin, spec)
        with self._lock:
            if len(self._rules) >= self.max_rules:
                raise ValueError(f"demasiadas alarmas ({len(self._rules)}); borra alguna antes de crear otra")
            rule.id = self._next_id
            self._next_id += 1
            rule.created_ts = float(self.clock.now())
            if rule.kind in PRICE_KINDS:
                armed = _armed(rule.direction) if rule.params.get("immediate") else None
                if armed is not None:
                    rule.state["s"] = armed
                else:
                    ref = _num(ref_px)
                    if ref is not None and ref > 0:
                        _store_sign(rule.state, rule.direction, _sign(ref - rule.params["level"]))
            self._rules[rule.id] = rule
            self._save()
        log.info("alarma creada: %s", describe_rule(rule))
        return rule

    def remove(self, rule_id: Any) -> bool:
        with self._lock:
            try:
                rid = int(rule_id)
            except (TypeError, ValueError):
                return False
            rule = self._rules.pop(rid, None)
            if rule is None:
                return False
            self._zone_seen.pop(rid, None)
            self._save()
        log.info("alarma #%d borrada", rid)
        return True

    def clear(self) -> int:
        with self._lock:
            n = len(self._rules)
            self._rules.clear()
            self._zone_seen.clear()
            self._save()
        return n

    def rules(self) -> list[AlertRule]:
        with self._lock:
            return [self._rules[k] for k in sorted(self._rules)]

    def get(self, rule_id: int) -> Optional[AlertRule]:
        with self._lock:
            return self._rules.get(int(rule_id))

    def list_text(self) -> str:
        """Respuesta a "alarmas": una línea por alarma y el estado del silencio."""
        with self._lock:
            rules = self.rules()
            lines = [f"Alarmas activas ({len(rules)}):"] + [describe_rule(r) for r in rules] if rules else \
                ["No hay alarmas activas. Ejemplo: alerta BTC > 100000"]
            if self.is_muted():
                until = "hasta 'unmute'" if self.muted_until == math.inf else f"hasta {_hhmm(self.muted_until)}"
                lines.append(f"Avisos no críticos silenciados {until} ({self.silenced} silenciados).")
            return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        """Contadores para ``status.json``."""
        with self._lock:
            until = self.muted_until
            return {"rules": len(self._rules), "fired_total": self.fired_total, "muted": self.is_muted(),
                    "muted_until": None if until is None or until == math.inf else until,
                    "silenced": self.silenced}

    def _build(self, coin: Any, spec: Mapping[str, Any]) -> AlertRule:
        if not isinstance(spec, Mapping):
            raise ValueError("especificación de alarma inválida")
        kind = str(spec.get("kind") or "").strip().lower()
        if kind not in KINDS:
            raise ValueError(f"tipo de alarma desconocido: {kind or '(vacío)'!r}. Tipos: {', '.join(KINDS)}")
        coin = str(coin or spec.get("coin") or "").strip().upper()
        if not coin:
            raise ValueError("falta la moneda de la alarma")
        params: dict[str, Any] = {}
        for name, default in KINDS[kind].items():
            v = spec.get(name)
            if v is None:
                v = default
            if v is _REQUIRED:
                raise ValueError(f"falta '{name}' para una alarma {kind}")
            params[name] = v
        params = _validate_params(kind, params)
        if _as_bool(spec.get("immediate", False)):
            params["immediate"] = True

        interval = ""
        if kind in CANDLE_KINDS:
            interval = str(spec.get("interval") or DEFAULT_INTERVAL).strip()
            interval_seconds(interval)   # ValueError con la lista de intervalos válidos
        cooldown = spec.get("cooldown_s", spec.get("cooldown"))
        cooldown = self.settings.alert_cooldown_seconds if cooldown is None else _num(cooldown)
        if cooldown is None or cooldown < 0:
            raise ValueError("cooldown de la alarma inválido (segundos >= 0)")
        return AlertRule(id=0, coin=coin, kind=kind, params=params, interval=interval,
                         repeat=_as_bool(spec.get("repeat", False)), cooldown_s=float(cooldown),
                         note=str(spec.get("note") or "").strip()[:200],
                         severity=normalize_level(spec.get("severity"), default="warn"))

    # ================================================================== evaluación
    def evaluate(self, market: Any, scanner: Optional[LiquidityScanner] = None) -> list[str]:
        """Evalúa todas las alarmas (una vez por ciclo). Devuelve los textos de las que dispararon (también las
        silenciadas por ``mute``, que no se envían pero quedan en ``alerts.jsonl``). Nunca lanza."""
        fired: list[str] = []
        try:
            with self._lock:
                now = float(self.clock.now())
                self._check_mute_expiry(now)
                rules = self.rules()
                if not rules:
                    return fired
                due = [r for r in rules if self._is_due(r, now)]
                limits: dict[tuple[str, str], int] = {}
                for r in due:
                    if r.kind in CANDLE_KINDS:
                        key = (r.coin, r.interval)
                        limits[key] = max(limits.get(key, 0), _bars_needed(r))
                ctx = _Cycle(market, lambda: self._get_scanner(market, scanner), now, limits, self._warn)
                for rule in due:
                    try:
                        detail = self._eval_rule(rule, ctx, now)
                    except Exception as e:  # noqa: BLE001 - una alarma rota no frena a las demás
                        self._warn(f"rule:{rule.id}", f"alarmas: error evaluando #{rule.id} ({type(e).__name__}: {e})")
                        continue
                    if detail:
                        text = self._fire(rule, detail, now)
                        if text:
                            fired.append(text)
                if self._dirty:
                    self._save()
        except Exception:  # noqa: BLE001 - nunca tumba el bucle
            log.exception("alarmas: error inesperado al evaluar")
        return fired

    def _get_scanner(self, market: Any, scanner: Optional[LiquidityScanner]) -> Optional[LiquidityScanner]:
        if scanner is not None:
            return scanner
        if self._scanner is None or self._scanner.market is not market:
            self._scanner = LiquidityScanner(market, self.settings)
        return self._scanner

    def _is_due(self, rule: AlertRule, now: float) -> bool:
        """Las alarmas de velas sólo se evalúan si puede haber cerrado una vela nueva (ahorra peticiones)."""
        if rule.kind not in CANDLE_KINDS:
            return True
        bar = _num(rule.state.get("bar"))
        if bar is None:
            return True
        return now >= bar + 2 * interval_seconds(rule.interval)

    def _eval_rule(self, rule: AlertRule, ctx: _Cycle, now: float) -> Optional[str]:
        if rule.kind in PRICE_KINDS:
            return self._eval_price(rule, ctx)
        if rule.kind == "zone":
            return self._eval_zone(rule, ctx, now)
        return self._eval_candles(rule, ctx)

    def _eval_price(self, rule: AlertRule, ctx: _Cycle) -> Optional[str]:
        px = ctx.mid(rule.coin)
        if px is None:
            return None
        level = float(rule.params["level"])
        cur = _sign(px - level)
        direction = rule.direction
        prev = rule.state.get("s")
        if prev is None:   # primera observación: sólo fija la referencia
            self._dirty |= _store_sign(rule.state, direction, cur)
            return None
        hit = _crossed(direction, prev, cur)
        self._dirty |= _store_sign(rule.state, direction, cur)
        if not hit:
            return None
        if rule.kind == "price_above":
            return f"precio {fmt_px(px)} supera {fmt_px(level)}"
        if rule.kind == "price_below":
            return f"precio {fmt_px(px)} cae por debajo de {fmt_px(level)}"
        return f"precio {fmt_px(px)} cruza {fmt_px(level)} {'al alza' if cur > 0 else 'a la baja'}"

    def _eval_zone(self, rule: AlertRule, ctx: _Cycle, now: float) -> Optional[str]:
        px = ctx.mid(rule.coin)
        zones = ctx.zones(rule.coin)
        want = rule.params.get("side")
        sides = [want] if want else ["long", "short"]
        detail = None
        seen = self._zone_seen.get(rule.id)
        if px is not None and seen is not None and now - seen[0] <= ZONE_MAX_AGE_S:
            for side in sides:
                z = seen[1].get(side)
                if z is None:
                    continue
                low, high, usd, source = z
                if (px <= high) if side == "long" else (px >= low):
                    detail = (f"precio {fmt_px(px)} entra en la zona de liquidez {side.upper()} "
                              f"{fmt_px(low)}-{fmt_px(high)} ({source}, {fmt_usd(usd, sign=False)})")
                    break
        if zones is not None:
            snap: dict[str, Optional[tuple]] = {}
            for s in (Side.LONG, Side.SHORT):
                z = zones.get(s)
                ok = z is not None and all(_num(v) is not None for v in (z.low, z.high, z.usd))
                snap[s.value] = (float(z.low), float(z.high), float(z.usd), str(z.source)) if ok else None
            self._zone_seen[rule.id] = (now, snap)
        return detail

    def _eval_candles(self, rule: AlertRule, ctx: _Cycle) -> Optional[str]:
        step = interval_seconds(rule.interval)
        bars = ctx.bars(rule.coin, rule.interval)
        if bars is None or len(bars) < 2:
            return None
        crossing = rule.kind in CANDLE_CROSS_KINDS
        if crossing:
            value, describe = self._cross_series(rule, ctx, bars)
        else:
            value, describe = None, None
        last = len(bars) - 1
        last_ts = _ts(bars.index[last])
        st = rule.state
        immediate = bool(rule.params.get("immediate"))

        if _num(st.get("bar")) is None:
            # Referencia: la última vela cerrada antes de crear la alarma (las anteriores no cuentan).
            base = last
            for i in range(last, -1, -1):
                if _ts(bars.index[i]) + step <= rule.created_ts:
                    base = i
                    break
            if immediate:
                base = min(base, last - 1)
            st["bar"] = _ts(bars.index[base])
            if crossing:
                ref = _armed(rule.direction) if immediate else _sign(value(base))
                if ref is not None:
                    st["s"] = ref
            self._dirty = True

        if last_ts <= float(st["bar"]):
            return None
        st["bar"] = last_ts
        self._dirty = True
        if crossing:
            cur = _sign(value(last))
            if cur is None:
                return None
            hit = _crossed(rule.direction, st.get("s"), cur)
            _store_sign(st, rule.direction, cur)
            return describe(last, cur) if hit else None
        return self._candle_event(rule, bars, last)

    def _cross_series(self, rule: AlertRule, ctx: _Cycle, bars: pd.DataFrame
                      ) -> tuple[Callable[[int], Optional[float]], Callable[[int, int], str]]:
        """(valor con signo respecto al nivel en la vela i, descripción del cruce) de rsi_* y ma_cross."""
        p = rule.params
        close = bars["close"]
        base_key = (rule.coin, rule.interval)
        if rule.kind in ("rsi_below", "rsi_above"):
            period, level = int(p["period"]), float(p["level"])
            series = ctx.memo(base_key + ("rsi", period), lambda: rsi(close, period).to_numpy(dtype=float))

            def value(i: int) -> Optional[float]:
                v = _num(series[i])
                return None if v is None else v - level

            def describe(i: int, cur: int) -> str:
                verb = "sube por encima" if cur > 0 else "cae por debajo"
                return f"RSI({period}) {rule.interval} {series[i]:.1f} {verb} de {level:g} (cierre {fmt_px(close.iloc[i])})"

            return value, describe

        kind = str(p.get("ma", "sma"))
        fn = ema if kind == "ema" else sma
        fast_n, slow_n = int(p["fast"]), int(p["slow"])
        fast = ctx.memo(base_key + (kind, fast_n), lambda: fn(close, fast_n).to_numpy(dtype=float))
        slow = ctx.memo(base_key + (kind, slow_n), lambda: fn(close, slow_n).to_numpy(dtype=float))

        def value(i: int) -> Optional[float]:
            f, s = _num(fast[i]), _num(slow[i])
            return None if f is None or s is None else f - s

        def describe(i: int, cur: int) -> str:
            ma = kind.upper()
            way = "al alza" if cur > 0 else "a la baja"
            return (f"{ma}({fast_n}) cruza {way} la {ma}({slow_n}) en {rule.interval} "
                    f"({fmt_px(fast[i])} vs {fmt_px(slow[i])})")

        return value, describe

    @staticmethod
    def _candle_event(rule: AlertRule, bars: pd.DataFrame, i: int) -> Optional[str]:
        p = rule.params
        if rule.kind == "volume_spike":
            if "volume" not in bars.columns:
                return None
            lookback = int(p["lookback"])
            if i < lookback:
                return None
            vol = bars["volume"].to_numpy(dtype=float)
            ref = vol[i - lookback:i]
            ref = ref[np.isfinite(ref)]
            cur = _num(vol[i])
            if cur is None or len(ref) < max(2, lookback // 2):
                return None
            mean = float(ref.mean())
            if mean <= 0 or cur < float(p["mult"]) * mean:
                return None
            return (f"volumen {rule.interval} {cur:,.0f} = {cur / mean:.1f}x la media de {lookback} velas "
                    f"(umbral {float(p['mult']):g}x)")
        # move_pct
        close = bars["close"].to_numpy(dtype=float)
        c0, c1 = _num(close[i - 1]), _num(close[i])
        if c0 is None or c1 is None or c0 <= 0:
            return None
        chg = (c1 / c0 - 1.0) * 100.0
        pct = float(p["pct"])
        direction = rule.direction
        if abs(chg) < pct or (direction == "up" and chg <= 0) or (direction == "down" and chg >= 0):
            return None
        return f"movimiento de {chg:+.2f}% en la última vela de {rule.interval} (cierre {fmt_px(c1)}, umbral {pct:g}%)"

    def _fire(self, rule: AlertRule, detail: str, now: float) -> Optional[str]:
        if rule.repeat and rule.last_fired_ts is not None and now - rule.last_fired_ts < rule.cooldown_s:
            log.debug("alarma #%d en pausa (%s)", rule.id, detail)
            return None
        rule.last_fired_ts = now
        rule.fired_count += 1
        self.fired_total += 1
        self._dirty = True
        text = f"ALARMA #{rule.id} {rule.coin}: {detail}"
        if rule.note:
            text += f" · {rule.note}"
        if not rule.repeat:
            self._rules.pop(rule.id, None)
            self._zone_seen.pop(rule.id, None)
            text += " (alarma de una vez: eliminada)"
        self._deliver(text, rule.severity, rule.kind, rule.coin, rule.id)
        return text

    # ================================================================== avisos de la mesa
    def system(self, kind: str, text: str, level: str = "info", key: Optional[str] = None,
               coin: Optional[str] = None, image: Optional[bytes] = None) -> bool:
        """Aviso de la mesa. Con ``key``, uno igual no se repite antes de ``alert_cooldown_seconds`` (al pasar se
        dice cuántos se omitieron). True si se envió a los canales (False: omitido por anti-spam o silenciado)."""
        try:
            with self._lock:
                now = float(self.clock.now())
                self._check_mute_expiry(now)
                level = normalize_level(level)
                text = str(text)
                if key is not None:
                    k = str(key)
                    seen = self._sys_seen.get(k)
                    cooldown = float(self.settings.alert_cooldown_seconds)
                    if seen is not None and now - seen[0] < cooldown and (level != "critical" or seen[1] == text):
                        seen[2] += 1
                        return False
                    omitted = seen[2] if seen is not None else 0
                    self._sys_seen[k] = [now, text, 0]
                    self._prune_system_keys(now, cooldown)
                    if omitted:
                        text += f" (+{omitted} aviso(s) iguales omitidos)"
                return self._deliver(text, level, str(kind or ""), coin, None, image)
        except Exception:  # noqa: BLE001
            log.exception("alarmas: error enviando un aviso de la mesa")
            return False

    def _prune_system_keys(self, now: float, cooldown: float) -> None:
        if len(self._sys_seen) <= SYSTEM_KEYS_KEPT:
            return
        for k in [k for k, v in self._sys_seen.items() if now - v[0] >= cooldown]:
            del self._sys_seen[k]
        while len(self._sys_seen) > SYSTEM_KEYS_KEPT:      # todas recientes: se olvidan las más antiguas
            del self._sys_seen[min(self._sys_seen, key=lambda key: self._sys_seen[key][0])]

    # ================================================================== silencio
    def is_muted(self, now: Optional[float] = None) -> bool:
        with self._lock:
            if self.muted_until is None:
                return False
            return (float(self.clock.now()) if now is None else now) < self.muted_until

    def mute(self, minutes: Optional[float] = None) -> str:
        """Silencia los avisos no críticos ``minutes`` minutos (None = hasta ``unmute``). Devuelve la respuesta."""
        with self._lock:
            now = float(self.clock.now())
            if minutes is None:
                until = math.inf
            else:
                m = _num(minutes)
                if m is None or m < 0:
                    raise ValueError("minutos de silencio inválidos")
                if m == 0:
                    return self.unmute()
                until = now + min(m, MAX_MUTE_MINUTES) * 60.0
            if self.muted_until is None:
                self.silenced = 0
            self.muted_until = until
            self._save()
        if until == math.inf:
            span = "hasta 'unmute'"
        else:
            span = f"{(until - now) / 60:g} min (hasta {_hhmm(until)})"
        return (f"Avisos no críticos silenciados {span}. Los críticos siguen llegando y todo queda en "
                f"{LOG_FILE}.")

    def unmute(self) -> str:
        with self._lock:
            was, n = self.muted_until is not None, self.silenced
            self.muted_until = None
            self.silenced = 0
            self._save()
        if not was:
            return "Los avisos no estaban silenciados."
        return f"Avisos reactivados ({n} silenciado(s) mientras tanto, en {LOG_FILE})."

    def _check_mute_expiry(self, now: float) -> None:
        if self.muted_until is None or self.muted_until == math.inf or now < self.muted_until:
            return
        n = self.silenced
        self.muted_until = None
        self.silenced = 0
        self._save()
        extra = f": {n} aviso(s) silenciados, en {LOG_FILE}" if n else ""
        self._deliver(f"Fin del silencio de avisos{extra}.", "info", "mute", None, None)

    # ================================================================== entrega y diario
    def _deliver(self, text: str, level: str, kind: str, coin: Optional[str], rule_id: Optional[int],
                 image: Optional[bytes] = None) -> bool:
        muted = level != "critical" and self.is_muted()
        now = float(self.clock.now())
        rec: dict[str, Any] = {"ts": now, "time": _iso(now), "level": level, "kind": kind, "coin": coin,
                               "text": text, "rule_id": rule_id}
        if muted:
            rec["muted"] = True
        self._append_log(rec)
        if muted:
            self.silenced += 1
            return False
        if self.notifier is None:
            return True
        try:
            self.notifier.send(text, level=level, kind=kind, coin=coin, image=image)
        except Exception as e:  # noqa: BLE001 - el notifier ya no debería lanzar; por si acaso
            log.warning("alarmas: el notificador falló (%s)", type(e).__name__)
            return False
        return True

    def _append_log(self, rec: dict[str, Any]) -> None:
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        try:
            with self._lock:
                try:
                    if self.log_path.stat().st_size > LOG_MAX_BYTES:
                        os.replace(self.log_path, self.log_path.with_name(LOG_FILE + ".1"))
                except FileNotFoundError:
                    pass
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except OSError as e:
            log.error("alarmas: no se pudo escribir %s: %s", LOG_FILE, e)

    def _warn(self, key: str, msg: str) -> None:
        """Registra un problema de datos como mucho cada ``WARN_EVERY_S`` por clave (sin llenar el log)."""
        now = float(self.clock.now())
        last = self._warned.get(key)
        if last is not None and now - last < WARN_EVERY_S:
            log.debug("%s", msg)
            return
        self._warned[key] = now
        log.warning("%s", msg)

    # ================================================================== persistencia
    def _save(self) -> None:
        mute = None
        if self.muted_until is not None:
            mute = {"until": None if self.muted_until == math.inf else self.muted_until, "silenced": self.silenced}
        data = {"version": STATE_VERSION, "next_id": self._next_id, "mute": mute,
                "rules": [r.to_dict() for r in self.rules()]}
        try:
            self.journal.save_json(STATE_NAME, data)
            self._dirty = False
        except (OSError, TypeError, ValueError) as e:
            log.error("alarmas: no se pudo guardar %s.json: %s", STATE_NAME, e)

    def _load(self) -> None:
        raw = self.journal.load_json(STATE_NAME, None)
        if raw is None:
            return
        if not isinstance(raw, dict):
            log.error("alarmas: %s.json con formato inesperado; se empieza sin alarmas", STATE_NAME)
            return
        for d in raw.get("rules") or []:
            try:
                rule = AlertRule.from_dict(d)
                rule.params = _validate_params(rule.kind, rule.params)
                if rule.kind in CANDLE_KINDS:
                    interval_seconds(rule.interval)
            except Exception as e:  # noqa: BLE001 - una regla ilegible no impide cargar las demás
                log.warning("alarmas: regla guardada ilegible ignorada (%s): %r", e, d)
                continue
            self._rules[rule.id] = rule
        try:
            next_id = int(raw.get("next_id") or 1)
        except (TypeError, ValueError):
            next_id = 1
        self._next_id = max([next_id, 1] + [rid + 1 for rid in self._rules])
        mute = raw.get("mute")
        if isinstance(mute, dict):
            until = _num(mute.get("until"))
            self.muted_until = math.inf if mute.get("until") is None else until
            try:
                self.silenced = int(mute.get("silenced") or 0)
            except (TypeError, ValueError):
                self.silenced = 0


def _validate_params(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """Normaliza y valida los parámetros de ``kind``. ``ValueError`` con el motivo en español."""
    p = dict(params)

    def positive(name: str) -> float:
        v = _num(p.get(name))
        if v is None or v <= 0:
            raise ValueError(f"'{name}' debe ser un número > 0 (recibido {p.get(name)!r})")
        return v

    def integer(name: str, lo: int, hi: int) -> int:
        v = _num(p.get(name))
        if v is None or v != int(v) or not lo <= v <= hi:
            raise ValueError(f"'{name}' debe ser un entero entre {lo} y {hi} (recibido {p.get(name)!r})")
        return int(v)

    def direction() -> str:
        d = _DIRECTIONS.get(str(p.get("direction") or "any").strip().lower())
        if d is None:
            raise ValueError(f"dirección inválida {p.get('direction')!r}: usa up, down o any")
        return d

    if kind in PRICE_KINDS:
        p["level"] = positive("level")
    elif kind in ("rsi_below", "rsi_above"):
        level = _num(p.get("level"))
        if level is None or not 0 < level < 100:
            raise ValueError(f"nivel de RSI inválido {p.get('level')!r} (entre 0 y 100)")
        p["level"] = level
        p["period"] = integer("period", 2, 100)
    elif kind == "ma_cross":
        p["fast"] = integer("fast", 1, 400)
        p["slow"] = integer("slow", 2, 500)
        if p["fast"] >= p["slow"]:
            raise ValueError(f"la media rápida ({p['fast']}) debe ser menor que la lenta ({p['slow']})")
        ma = str(p.get("ma") or "sma").strip().lower()
        if ma not in ("sma", "ema"):
            raise ValueError(f"media {p.get('ma')!r} inválida: usa sma o ema")
        p["ma"] = ma
        p["direction"] = direction()
    elif kind == "zone":
        side = p.get("side")
        side = side.value if isinstance(side, Side) else (str(side).strip().lower() if side else None)
        if side not in (None, "long", "short"):
            raise ValueError(f"lado de zona inválido {p.get('side')!r}: usa long o short")
        p["side"] = side
    elif kind == "volume_spike":
        p["mult"] = positive("mult")
        p["lookback"] = integer("lookback", 2, 500)
    elif kind == "move_pct":
        pct = _num(p.get("pct"))
        if pct is None or pct == 0:
            raise ValueError(f"'pct' debe ser un número distinto de 0 (recibido {p.get('pct')!r})")
        p["direction"] = direction()
        if pct < 0 and p["direction"] == "any":   # "move_pct -3" = caída de al menos un 3 %
            p["direction"] = "down"
        p["pct"] = abs(pct)
    return p


def _bars_needed(rule: AlertRule) -> int:
    """Velas a pedir para calcular la alarma (con margen para la vela en curso y la referencia)."""
    p = rule.params
    if rule.kind in ("rsi_below", "rsi_above"):
        n = max(10 * int(p["period"]), 100)          # Wilder necesita historia para converger
    elif rule.kind == "ma_cross":
        n = int(p["slow"]) * (4 if p.get("ma") == "ema" else 1) + 5
    elif rule.kind == "volume_spike":
        n = int(p["lookback"]) + 3
    else:
        n = 5
    return min(n + 3, MAX_BARS)


__all__ = ["AlertRule", "AlertManager", "KINDS", "describe_rule"]

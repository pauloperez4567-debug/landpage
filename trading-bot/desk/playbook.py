"""Playbook: las gráficas (moneda + temporalidad) que la mesa opera en automático, cada una con su estrategia
y sus parámetros, adoptados normalmente desde un backtest walk-forward (``optimizer``).

- Una entrada por gráfica (``coin`` + ``interval``): adoptar otra estrategia en la misma gráfica la sustituye.
- Persistencia en ``<state_dir>/playbook.json`` = ``{"version": 1, "entries": [...]}`` con escritura atómica
  (``Journal.save_json``: tmp + fsync + rename). Un fichero ilegible se aparta (``*.corrupt-<ts>``) y se
  empieza vacío; una entrada inválida se ignora con un aviso en el log (nunca tumba la mesa).
- ``strategies(settings, clock)`` instancia las entradas activas con su temporalidad y parámetros; una
  entrada que ya no es válida (estrategia renombrada, parámetro que no existe) se salta y se avisa. Las
  instancias se reutilizan mientras la entrada no cambie (conservan su "una señal por vela").
- ``entry_from_report`` sólo adopta estrategias **aprobadas fuera de muestra** (salvo que se fuerce): que
  algo saliera bien en el pasado no garantiza nada, pero lo que ni siquiera pasó la validación fuera de
  muestra no debería operar solo.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Union

from .clock import Clock, RealClock
from .config import Settings
from .exchange.intervals import interval_seconds
from .journal import Journal
from .strategy import STRATEGIES, Strategy, create_strategy

log = logging.getLogger(__name__)

PLAYBOOK_FILE = "playbook.json"
PLAYBOOK_VERSION = 1


def _key(name: Any) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def _finite(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


@dataclass
class PlaybookEntry:
    """Una gráfica en automático. ``units`` None = ``settings.auto_units``; ``oos_metrics`` = métricas fuera de
    muestra del informe del que salió (``report_id``), sólo informativas."""

    coin: str
    interval: str
    strategy: str
    params: dict[str, Any]
    units: Optional[float]
    adopted_ts: float
    report_id: str
    oos_metrics: dict[str, Any]
    enabled: bool = True

    @property
    def key(self) -> tuple[str, str]:
        """Gráfica (moneda sin distinguir mayúsculas + temporalidad): una entrada por clave."""
        return self.coin.upper(), self.interval

    def to_dict(self) -> dict[str, Any]:
        return {"coin": self.coin, "interval": self.interval, "strategy": self.strategy, "params": dict(self.params),
                "units": self.units, "adopted_ts": self.adopted_ts, "report_id": self.report_id,
                "oos_metrics": dict(self.oos_metrics), "enabled": self.enabled}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlaybookEntry":
        """Desde JSON. ``ValueError`` si falta lo esencial (moneda, temporalidad, estrategia)."""
        if not isinstance(d, dict):
            raise ValueError("entrada del playbook que no es un objeto")
        coin, interval, strategy = (str(d.get(k) or "").strip() for k in ("coin", "interval", "strategy"))
        if not coin or not interval or not strategy:
            raise ValueError("entrada del playbook sin moneda, temporalidad o estrategia")
        enabled = d.get("enabled", True)
        return cls(coin=coin, interval=interval, strategy=_key(strategy),
                   params=dict(d["params"]) if isinstance(d.get("params"), dict) else {},
                   units=_finite(d.get("units")), adopted_ts=_finite(d.get("adopted_ts")) or 0.0,
                   report_id=str(d.get("report_id") or ""),
                   oos_metrics=dict(d["oos_metrics"]) if isinstance(d.get("oos_metrics"), dict) else {},
                   enabled=enabled if isinstance(enabled, bool) else str(enabled).lower() not in ("0", "false", "no"))

    def copy(self) -> "PlaybookEntry":
        return replace(self, params=dict(self.params), oos_metrics=dict(self.oos_metrics))

    def describe(self) -> str:
        """Una línea legible: gráfica, estrategia, parámetros, unidades y métricas fuera de muestra."""
        from .optimizer import format_params
        m = self.oos_metrics
        oos = ""
        if m:
            pf = _finite(m.get("profit_factor"))
            parts = [f"{_finite(m.get('total_return_pct')) or 0.0:+.1f} %",
                     "PF ∞" if pf is not None and pf >= 999 else f"PF {pf or 0.0:.2f}",
                     f"DD {_finite(m.get('max_drawdown_pct')) or 0.0:.1f} %", f"{int(_finite(m.get('n_trades')) or 0)} ops"]
            oos = " · fuera de muestra " + ", ".join(parts)
        units = "unidades por defecto" if self.units is None else f"{self.units:g} u"
        state = "" if self.enabled else " [PAUSADA]"
        rep = f" · informe {self.report_id}" if self.report_id else ""
        return (f"{self.coin} {self.interval} · {self.strategy} ({format_params(self.params)}) · {units}"
                f"{oos}{rep}{state}")


def validate_entry(entry: PlaybookEntry, settings: Optional[Settings] = None) -> PlaybookEntry:
    """Comprueba la entrada (moneda, temporalidad, estrategia, parámetros, unidades) y la devuelve normalizada.
    ``ValueError`` con el motivo en español si no vale."""
    e = entry.copy()
    e.coin = str(e.coin or "").strip()
    e.interval = str(e.interval or "").strip()
    e.strategy = _key(e.strategy)
    if not e.coin:
        raise ValueError("falta la moneda")
    interval_seconds(e.interval)                       # ValueError si la temporalidad no existe
    if e.strategy not in STRATEGIES:
        raise ValueError(f"estrategia desconocida: {entry.strategy!r} (disponibles: {', '.join(sorted(STRATEGIES))})")
    if e.units is not None:
        u = _finite(e.units)
        if u is None or u <= 0:
            raise ValueError(f"unidades inválidas: {entry.units!r} (deben ser > 0)")
        e.units = u
    try:
        json.dumps(e.params)
    except (TypeError, ValueError):
        raise ValueError("parámetros no serializables") from None
    create_strategy(e.strategy, settings or Settings(), interval=e.interval, **e.params)   # parámetros válidos
    return e


class Playbook:
    """Gráficas en automático, persistidas en ``<state_dir>/playbook.json``. Ver el docstring del módulo."""

    def __init__(self, state_dir: Union[str, Path], clock: Optional[Clock] = None):
        self.clock: Clock = clock if clock is not None else RealClock()
        self._journal = Journal(state_dir, self.clock)
        self.path = Path(state_dir) / PLAYBOOK_FILE
        self._lock = threading.RLock()
        self._entries: list[PlaybookEntry] = []
        self._cache: dict[tuple, Strategy] = {}
        self.reload()

    # ------------------------------------------------------------------ persistencia
    def reload(self) -> None:
        """Relee ``playbook.json`` (lo que haya cambiado fuera). Ilegible -> vacío (el fichero se aparta)."""
        raw = self._journal.load_json(PLAYBOOK_FILE, None)
        items: Any = raw
        if isinstance(raw, dict):              # {"entries": [...]} (lo que se escribe) o {clave: entrada}
            items = raw["entries"] if "entries" in raw else [v for v in raw.values() if isinstance(v, dict)]
        if isinstance(items, dict):
            items = list(items.values())
        entries: list[PlaybookEntry] = []
        for d in items if isinstance(items, list) else []:
            try:
                e = PlaybookEntry.from_dict(d)
            except ValueError as err:
                log.warning("playbook: entrada ignorada (%s): %r", err, d)
                continue
            entries = [x for x in entries if x.key != e.key] + [e]       # una por gráfica: manda la última
        with self._lock:
            self._entries = entries

    def _save(self, entries: list[PlaybookEntry]) -> None:
        """Guarda y, sólo si se pudo, pasa a ser el estado en memoria (``OSError`` se propaga)."""
        self._journal.save_json(PLAYBOOK_FILE, {"version": PLAYBOOK_VERSION, "entries": [e.to_dict() for e in entries]})
        self._entries = entries

    # ------------------------------------------------------------------ operaciones
    def adopt(self, entry: PlaybookEntry, settings: Optional[Settings] = None) -> Optional[PlaybookEntry]:
        """Añade (o sustituye) la gráfica de ``entry``. Devuelve la entrada sustituida o None.
        ``ValueError`` si la entrada no es válida; ``OSError`` si no se pudo guardar (nada cambia)."""
        e = validate_entry(entry, settings)
        with self._lock:
            old = next((x for x in self._entries if x.key == e.key), None)
            self._save([x for x in self._entries if x.key != e.key] + [e])
        return old.copy() if old is not None else None

    def remove(self, coin: str, interval: Optional[str] = None) -> int:
        """Quita la gráfica (o todas las temporalidades de ``coin`` si ``interval`` es None). Devuelve cuántas."""
        c = str(coin or "").strip().upper()
        with self._lock:
            keep = [x for x in self._entries
                    if not (x.coin.upper() == c and (interval is None or x.interval == interval))]
            n = len(self._entries) - len(keep)
            if n:
                self._save(keep)
        return n

    def set_enabled(self, coin: str, interval: str, enabled: bool) -> bool:
        """Pausa o reactiva una gráfica sin olvidar su configuración. False si no está en el playbook."""
        c = str(coin or "").strip().upper()
        with self._lock:
            entries = [x.copy() for x in self._entries]
            hit = next((x for x in entries if x.coin.upper() == c and x.interval == interval), None)
            if hit is None:
                return False
            if hit.enabled != bool(enabled):
                hit.enabled = bool(enabled)
                self._save(entries)
        return True

    def entries(self) -> list[PlaybookEntry]:
        with self._lock:
            return [e.copy() for e in self._entries]

    def get(self, coin: str, interval: str) -> Optional[PlaybookEntry]:
        c = str(coin or "").strip().upper()
        with self._lock:
            hit = next((x for x in self._entries if x.coin.upper() == c and x.interval == interval), None)
        return hit.copy() if hit is not None else None

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------ operativa automática
    def strategies(self, settings: Settings, clock: Optional[Clock] = None) -> list[tuple[PlaybookEntry, Strategy]]:
        """(entrada, estrategia) de cada gráfica activa, con su temporalidad y parámetros. Nunca lanza: una
        entrada que no se puede instanciar se salta con un aviso en el log."""
        out: list[tuple[PlaybookEntry, Strategy]] = []
        cache: dict[tuple, Strategy] = {}
        with self._lock:
            entries = [e.copy() for e in self._entries if e.enabled]
            for e in entries:
                sig = (e.coin, e.interval, e.strategy, json.dumps(e.params, sort_keys=True, default=str),
                       id(settings), id(clock))
                strat = self._cache.get(sig)
                if strat is None:
                    try:
                        strat = create_strategy(e.strategy, settings, clock, interval=e.interval, **e.params)
                    except Exception as err:  # noqa: BLE001 - una entrada rota no frena las demás
                        log.warning("playbook: %s %s %s no se puede usar: %s", e.coin, e.interval, e.strategy, err)
                        continue
                cache[sig] = strat
                out.append((e, strat))
            self._cache = cache
        return out

    def summary_text(self) -> str:
        """Listado para el comando ``playbook``."""
        entries = self.entries()
        if not entries:
            return ("Playbook vacío: ninguna gráfica en automático por backtest. Prueba \"backtest HYPE 1h\" y, "
                    "si hay ganadora aprobada, \"activa HYPE 1h\".")
        active = sum(1 for e in entries if e.enabled)
        lines = [f"Playbook: {active} gráfica(s) en automático" + (f" ({len(entries) - active} pausadas)"
                                                                     if active < len(entries) else "")]
        lines += [f"  · {e.describe()}" for e in entries]
        lines.append("Las métricas son fuera de muestra del backtest: orientativas, no una promesa.")
        return "\n".join(lines)


def entry_from_report(report: Any, strategy: Optional[str] = None, *, units: Optional[float] = None,
                      clock: Optional[Clock] = None, allow_rejected: bool = False) -> PlaybookEntry:
    """Entrada del playbook desde un ``OptimizationReport``: la ganadora o ``strategy``.

    ``ValueError`` (con el motivo, en español, para el usuario) si no hay ganadora, la estrategia no se
    evaluó, su evaluación falló o no aprobó fuera de muestra (esto último sólo se acepta con
    ``allow_rejected=True``, y nunca si la evaluación falló).
    """
    chart = f"{report.coin} {report.interval}"
    if strategy is None:
        ev = report.winner_evaluation
        if ev is None:
            raise ValueError(f"El backtest de {chart} (informe {report.id}) no tiene ganadora: ninguna estrategia "
                             "aprobó fuera de muestra. No se activa nada.")
    else:
        ev = report.get(strategy)
        if ev is None:
            raise ValueError(f"{_key(strategy)} no se evaluó en el backtest de {chart} (informe {report.id}).")
    if ev.error or not ev.params:
        raise ValueError(f"La evaluación de {ev.strategy} en {chart} falló ({'; '.join(ev.reasons) or ev.error}): "
                         "no se puede activar.")
    if not ev.passed and not allow_rejected:
        raise ValueError(f"{ev.strategy} no aprobó fuera de muestra en {chart}: {'; '.join(ev.reasons)}. "
                         "No se activa.")
    clk = clock if clock is not None else RealClock()
    return PlaybookEntry(coin=report.coin, interval=report.interval, strategy=ev.strategy, params=dict(ev.params),
                         units=units, adopted_ts=float(clk.now()), report_id=report.id,
                         oos_metrics=dict(ev.oos_metrics), enabled=True)


__all__ = ["PLAYBOOK_FILE", "PlaybookEntry", "Playbook", "entry_from_report", "validate_entry"]

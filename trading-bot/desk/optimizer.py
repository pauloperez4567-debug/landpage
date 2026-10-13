"""Optimización walk-forward por gráfica (moneda + temporalidad): elegir parámetros sin mirar el futuro.

Un backtest NO garantiza el rendimiento futuro. Elegir los parámetros que mejor fueron en una historia
y medirlos en esa misma historia (dentro de muestra) siempre sale mejor de lo que saldrá después: con
suficientes combinaciones hasta el ruido parece una estrategia. Por eso aquí lo único que decide es lo
medido **fuera de muestra**, en velas que el optimizador no vio al elegir:

- **Walk-forward anclado** de ``bt_folds`` tramos: las velas se parten en ``bt_folds + 1`` bloques
  iguales; el tramo ``k`` entrena con los bloques ``0..k-1`` (desde el principio: anclado) y se valida
  en el bloque ``k``. Así cada vela fuera de muestra se mide con parámetros elegidos sólo con su pasado.
- En cada entrenamiento se prueban hasta ``bt_max_combos`` combinaciones del ``PARAM_GRID`` de la
  estrategia (muestreo determinista con ``seed``; la configuración por defecto siempre entra) y gana la
  de mejor ``bt_objective`` *con un mínimo de operaciones* (una combinación con 2 operaciones ganadoras
  no es información, es suerte).
- La curva fuera de muestra encadena los tramos de validación (cada uno empieza con el capital con el que
  acabó el anterior). Sus métricas son las que se enseñan y las que aprueban o rechazan:
  PF >= ``bt_min_profit_factor``, retorno > 0, operaciones >= ``bt_min_trades``, drawdown <=
  ``bt_max_drawdown_pct`` y **ventaja significativa** (abajo). ``reasons`` explica cada rechazo. Ranking por
  ``bt_objective`` fuera de muestra.
- Significación: con 20-30 operaciones una estrategia sin ventaja saca PF >= 1.15 casi la mitad de las
  veces, y comparando 5 estrategias casi siempre "gana" alguna. Medido con las 5 estrategias sobre paseos
  aleatorios de 3000 velas (ruido puro): sólo con los cuatro filtros, 17 de 24 gráficas tenían "ganadora";
  con la significación, 0 de 40 (y las tendencias reales siguen aprobando). Por eso además se exige que el R medio
  por operación fuera de muestra sea mayor que cero con t de Student >= el cuantil ``1 - alfa/m``
  (``alfa`` = ``SIGNIFICANCE_ALPHA``, ``m`` = estrategias comparadas en el informe: corrección de
  Bonferroni). Con pocas operaciones sólo pasa una ventaja grande; es lo honesto.
- Parámetros propuestos para adoptar = los mejores del **último** entrenamiento (su validación es el
  último tramo). ``is_metrics`` son los de esos parámetros en su propio entrenamiento: optimistas por
  construcción, sólo para comparar.

Sin fuga de datos: las señales son causales (la fila ``i`` sólo usa velas ``<= i``), así que se calculan
una vez por combinación sobre toda la historia y cada entrenamiento simula sólo sus velas. Como red de
seguridad, en cada validación las señales de la combinación elegida se recalculan sólo con las velas
hasta el final del tramo y se comparan con las usadas al entrenar: si difieren, la estrategia mira el
futuro y su evaluación se invalida.

Rendimiento (por evitar recalcular): zonas históricas una vez por gráfica, señales una vez por combinación
y simulaciones sobre arrays. 5 estrategias x 60 combinaciones x 4 tramos sobre 3000 velas: ~7.5 s en un
proceso (contenedor de 4 núcleos), sin necesidad de paralelizar.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd

from .backtest import BacktestResult, BTTrade, compute_metrics, run_backtest
from .clock import Clock, RealClock
from .config import Settings
from .exchange.base import CANDLE_COLUMNS
from .exchange.intervals import interval_seconds
from .journal import Journal
from .liquidity import historical_zones
from .models import AssetMeta
from .strategy import STRATEGIES, Strategy

log = logging.getLogger(__name__)

# ``bt_objective`` -> métrica que se maximiza (en el entrenamiento) y que ordena (fuera de muestra).
OBJECTIVES: dict[str, str] = {"sharpe": "sharpe", "calmar": "calmar", "profit_factor": "profit_factor",
                              "return": "total_return_pct"}
OBJECTIVE_LABELS = {"sharpe": "Sharpe", "calmar": "Calmar", "profit_factor": "profit factor",
                    "total_return_pct": "retorno"}
DEFAULT_OBJECTIVE = "sharpe"
MIN_SEGMENT_BARS = 50       # velas mínimas de cada bloque (con menos se reducen los tramos)
MIN_TRAIN_TRADES = 3        # operaciones mínimas para que una combinación cuente al entrenar
TRAIN_TRADES_FRAC = 0.5     # ... y además la mitad de las que exigiría ``bt_min_trades`` a ese ritmo
REPORT_VERSION = 1
SIGNIFICANCE_ALPHA = 0.05   # falsos positivos admitidos por informe (una cola), repartidos entre las estrategias
T_CAP = 99.0                # t sin dispersión (todas las operaciones iguales): tope finito para JSON

ProgressFn = Callable[[int, int, str], Any]
StopFn = Callable[[], bool]

DISCLAIMER = ("Aviso: un backtest no garantiza resultados futuros. Las cifras dentro de muestra son optimistas "
              "(los parámetros se eligieron con esos mismos datos); sólo cuentan las de fuera de muestra, y aun "
              "así el mercado puede cambiar. Comisiones y deslizamiento incluidos; funding y liquidaciones no.")


class _Stopped(Exception):
    """``should_stop`` pidió parar: se devuelve lo terminado hasta ahora."""


class _Leak(Exception):
    """La estrategia no es causal: sus señales pasadas cambian con velas posteriores."""


# --- resultados -----------------------------------------------------------------------

def _f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _metrics(d: Any) -> dict[str, float]:
    """Métricas desde JSON (tolerante): números finitos; ``n_trades`` entero."""
    out = {k: _f(v) for k, v in (d.items() if isinstance(d, dict) else ())
           if isinstance(v, (int, float, str)) and not isinstance(v, bool)}
    if "n_trades" in out:
        out["n_trades"] = int(out["n_trades"])
    return out


@dataclass
class FoldResult:
    """Un tramo del walk-forward. Horas en segundos UNIX: inicio = apertura de la primera vela, fin = cierre
    de la última. ``params`` son los elegidos con el entrenamiento; ``is_metrics`` los suyos en ese
    entrenamiento y ``oos_metrics`` los del bloque siguiente (que no vieron)."""

    fold: int                         # 1..folds
    train_start_ts: float
    train_end_ts: float
    test_start_ts: float
    test_end_ts: float
    train_bars: int
    test_bars: int
    params: dict[str, Any]
    is_metrics: dict[str, float]
    oos_metrics: dict[str, float]
    combos: int = 0                   # combinaciones probadas
    min_train_trades: int = 0         # operaciones mínimas exigidas al elegir

    def to_dict(self) -> dict[str, Any]:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FoldResult":
        return cls(fold=int(_f(d.get("fold"))), train_start_ts=_f(d.get("train_start_ts")),
                   train_end_ts=_f(d.get("train_end_ts")), test_start_ts=_f(d.get("test_start_ts")),
                   test_end_ts=_f(d.get("test_end_ts")), train_bars=int(_f(d.get("train_bars"))),
                   test_bars=int(_f(d.get("test_bars"))),
                   params=dict(d.get("params") or {}) if isinstance(d.get("params"), dict) else {},
                   is_metrics=_metrics(d.get("is_metrics")), oos_metrics=_metrics(d.get("oos_metrics")),
                   combos=int(_f(d.get("combos"))), min_train_trades=int(_f(d.get("min_train_trades"))))


@dataclass
class StrategyEvaluation:
    """Evaluación walk-forward de una estrategia en una gráfica.

    - ``params``: propuestos para adoptar (los mejores del último entrenamiento; todos los ``PARAMS``).
    - ``is_metrics``: esos parámetros en su entrenamiento (dentro de muestra: optimista).
    - ``oos_metrics`` / ``oos_trades``: la curva fuera de muestra encadenada de todos los tramos.
    - ``score``: ``bt_objective`` fuera de muestra (None si la evaluación falló).
    - ``passed`` / ``reasons``: aprobada sólo si pasa todos los filtros fuera de muestra.
    - ``oos_result``: la curva fuera de muestra como ``BacktestResult`` (sólo en memoria: ``to_dict`` no la
      incluye; ``ReportStore.save(report, result)`` la guarda aparte).
    """

    strategy: str
    params: dict[str, Any]
    is_metrics: dict[str, float]
    oos_metrics: dict[str, float]
    oos_trades: list[BTTrade]
    folds: list[FoldResult]
    score: Optional[float]
    passed: bool
    reasons: list[str]
    combos: int = 0                   # combinaciones probadas en cada entrenamiento
    error: str = ""                   # fallo de la evaluación (no de la estrategia en el mercado)
    oos_result: Optional[BacktestResult] = field(default=None, repr=False, compare=False)

    @property
    def folds_positive(self) -> int:
        """Tramos fuera de muestra que acabaron en positivo."""
        return sum(1 for f in self.folds if f.oos_metrics.get("total_return_pct", 0.0) > 0)

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "params": dict(self.params), "is_metrics": dict(self.is_metrics),
                "oos_metrics": dict(self.oos_metrics), "oos_trades": [t.to_dict() for t in self.oos_trades],
                "folds": [f.to_dict() for f in self.folds], "score": self.score, "passed": self.passed,
                "reasons": list(self.reasons), "combos": self.combos, "error": self.error}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyEvaluation":
        trades, folds = [], []
        for t in d.get("oos_trades") if isinstance(d.get("oos_trades"), list) else []:
            try:
                trades.append(BTTrade.from_dict(t))
            except (KeyError, TypeError):
                continue
        for f in d.get("folds") if isinstance(d.get("folds"), list) else []:
            if isinstance(f, dict):
                folds.append(FoldResult.from_dict(f))
        score = d.get("score")
        return cls(strategy=str(d.get("strategy") or ""),
                   params=dict(d.get("params") or {}) if isinstance(d.get("params"), dict) else {},
                   is_metrics=_metrics(d.get("is_metrics")), oos_metrics=_metrics(d.get("oos_metrics")),
                   oos_trades=trades, folds=folds, score=None if score is None else _f(score),
                   passed=bool(d.get("passed")), reasons=[str(r) for r in d.get("reasons") or [] if r],
                   combos=int(_f(d.get("combos"))), error=str(d.get("error") or ""))


@dataclass
class OptimizationReport:
    """Informe de una optimización. ``winner`` = nombre de la mejor estrategia aprobada (o None).

    ``evaluations`` van ordenadas: aprobadas primero y, dentro, por ``score`` (fuera de muestra).
    Extras: ``objective``, ``folds`` (tramos usados), ``seed``, ``criteria`` (umbrales de aprobación),
    ``start_ts``/``end_ts`` (velas), ``oos_start_ts`` (inicio de lo fuera de muestra) y ``complete``
    (False si se paró antes de terminar: sólo están las estrategias evaluadas del todo).
    """

    id: str
    coin: str
    interval: str
    days: float
    bars: int
    created_ts: float
    evaluations: list[StrategyEvaluation]
    winner: Optional[str]
    notes: list[str] = field(default_factory=list)
    objective: str = "sharpe"
    folds: int = 0
    seed: int = 7
    criteria: dict[str, Any] = field(default_factory=dict)
    start_ts: float = 0.0
    end_ts: float = 0.0
    oos_start_ts: float = 0.0
    complete: bool = True

    # ------------------------------------------------------------------ consultas
    def get(self, strategy: str) -> Optional[StrategyEvaluation]:
        key = str(strategy or "").strip().lower().replace("-", "_")
        return next((e for e in self.evaluations if e.strategy == key), None)

    @property
    def winner_evaluation(self) -> Optional[StrategyEvaluation]:
        return self.get(self.winner) if self.winner else None

    def best(self) -> Optional[StrategyEvaluation]:
        """La ganadora o, si no hay, la primera del ranking que se pudo evaluar (para enseñar sus gráficos)."""
        return self.winner_evaluation or next((e for e in self.evaluations if not e.error), None)

    def result(self, strategy: Optional[str] = None) -> Optional[BacktestResult]:
        """Curva fuera de muestra (en memoria) de ``strategy`` o de ``best()``. None en un informe leído de JSON."""
        ev = self.get(strategy) if strategy else self.best()
        return ev.oos_result if ev is not None else None

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> dict[str, Any]:
        return {"version": REPORT_VERSION, "id": self.id, "coin": self.coin, "interval": self.interval,
                "days": self.days, "bars": self.bars, "created_ts": self.created_ts,
                "evaluations": [e.to_dict() for e in self.evaluations], "winner": self.winner,
                "notes": list(self.notes), "objective": self.objective, "folds": self.folds, "seed": self.seed,
                "criteria": dict(self.criteria), "start_ts": self.start_ts, "end_ts": self.end_ts,
                "oos_start_ts": self.oos_start_ts, "complete": self.complete}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizationReport":
        """Acepta ``to_dict()`` o ``{"report": ..., "result": ...}`` (lo que guarda ``ReportStore``)."""
        if isinstance(d.get("report"), dict):
            d = d["report"]
        evals = [StrategyEvaluation.from_dict(e) for e in d.get("evaluations") or [] if isinstance(e, dict)]
        winner = d.get("winner")
        if isinstance(winner, dict):
            winner = winner.get("strategy")
        folds = d.get("folds")
        return cls(id=str(d.get("id") or ""), coin=str(d.get("coin") or ""), interval=str(d.get("interval") or ""),
                   days=_f(d.get("days")), bars=int(_f(d.get("bars"))), created_ts=_f(d.get("created_ts")),
                   evaluations=evals, winner=str(winner) if winner else None,
                   notes=[str(n) for n in d.get("notes") or [] if n] if isinstance(d.get("notes"), list) else [],
                   objective=str(d.get("objective") or DEFAULT_OBJECTIVE),
                   folds=len(folds) if isinstance(folds, list) else int(_f(folds)),
                   seed=int(_f(d.get("seed"), 7)), criteria=dict(d.get("criteria") or {})
                   if isinstance(d.get("criteria"), dict) else {}, start_ts=_f(d.get("start_ts")),
                   end_ts=_f(d.get("end_ts")), oos_start_ts=_f(d.get("oos_start_ts")),
                   complete=bool(d.get("complete", True)))

    # ------------------------------------------------------------------ texto
    def summary_text(self, max_rows: int = 10) -> str:
        """Resumen en español para la consola y Telegram: tabla dentro vs fuera de muestra, ganadora,
        motivos de rechazo y la advertencia de que el pasado no garantiza el futuro."""
        obj_key = OBJECTIVES.get(self.objective, "sharpe")
        crit = self.criteria
        lines = [f"Backtest {self.coin} {self.interval} · {self.days:.0f} días ({self.bars} velas)"
                 + (f" · {_day(self.start_ts)} → {_day(self.end_ts)}" if self.start_ts and self.end_ts else "")]
        if self.folds:
            lines.append(f"Walk-forward anclado de {self.folds} tramos: parámetros elegidos con el pasado (hasta "
                         f"{crit.get('max_combos', '?')} combinaciones) y medidos en el tramo siguiente, "
                         f"que no vieron. Objetivo: {OBJECTIVE_LABELS.get(obj_key, obj_key)}.")
            if self.oos_start_ts:
                lines.append(f"Fuera de muestra: {_day(self.oos_start_ts)} → {_day(self.end_ts)}.")
        if not self.complete:
            lines.append("INCOMPLETO: la optimización se detuvo antes de terminar.")
        if self.evaluations:
            lines += ["", f"{'':18}{'DENTRO':>12} | {'FUERA de muestra (lo que cuenta)':^43}",
                      f"{'Estrategia':<18}{'Ret %':>6}{'PF':>6} | {'Ret %':>6}{'PF':>6}{'DD %':>6}{'Ops':>5}"
                      f"{'t':>6}{_short(obj_key):>8}"]
            for e in self.evaluations[:max_rows]:
                name = e.strategy[:17]
                if e.error:
                    lines.append(f"{name:<18}{'—':>6}{'—':>6} | no evaluada")
                    continue
                i, o = e.is_metrics, e.oos_metrics
                lines.append(f"{name:<18}{_pct(i.get('total_return_pct')):>6}{_pf(i.get('profit_factor')):>6} | "
                             f"{_pct(o.get('total_return_pct')):>6}{_pf(o.get('profit_factor')):>6}"
                             f"{_num(o.get('max_drawdown_pct'), 1):>6}{int(o.get('n_trades', 0)):>5}"
                             f"{_tval(o.get('t_stat')):>6}{_num(e.score, 2):>8} {'✔' if e.passed else '✘'}")
            if len(self.evaluations) > max_rows:
                lines.append(f"... y {len(self.evaluations) - max_rows} más")
        lines.append("")
        win = self.winner_evaluation
        if win is not None:
            lines.append(f"Ganadora: {win.strategy} (aprobada fuera de muestra; tramos en positivo "
                         f"{win.folds_positive}/{len(win.folds)})")
            lines.append(f"  Parámetros propuestos: {format_params(win.params)}")
            lines.append(f"  Para operarla en automático: activa {self.coin} {self.interval}")
        elif self.evaluations:
            lines.append("Sin ganadora: ninguna estrategia pasó los filtros fuera de muestra "
                         f"(PF >= {_f(crit.get('min_profit_factor')):.2f}, retorno > 0, "
                         f">= {int(_f(crit.get('min_trades')))} operaciones, DD <= "
                         f"{_f(crit.get('max_drawdown_pct')):.0f} %, ventaja significativa). "
                         "Mejor no operar esta gráfica en automático.")
        else:
            lines.append("Sin evaluaciones.")
        rejected = [e for e in self.evaluations[:max_rows] if not e.passed and e.reasons]
        if rejected:
            lines.append("Rechazadas:")
            lines += [f"  · {e.strategy}: {'; '.join(e.reasons)}" for e in rejected]
        if self.notes:
            lines.append("Notas:")
            lines += [f"  · {n}" for n in self.notes]
        lines += ["", DISCLAIMER]
        if self.id:
            lines.append(f"Informe {self.id}.")
        return "\n".join(lines)


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else "?"


def _pct(v: Any) -> str:
    return f"{_f(v):+.1f}" if v is not None else "—"


def _pf(v: Any) -> str:
    if v is None:
        return "—"
    x = _f(v)
    return "∞" if x >= 999 else f"{x:.2f}"


def _num(v: Any, nd: int) -> str:
    return f"{_f(v):.{nd}f}" if v is not None else "—"


def _tval(v: Any) -> str:
    """t de la ventaja: ±∞ sin dispersión (todas las operaciones con el mismo R)."""
    if v is None:
        return "—"
    x = _f(v)
    return ("∞" if x > 0 else "-∞") if abs(x) >= T_CAP else f"{x:.1f}"


def _short(metric: str) -> str:
    return {"total_return_pct": "Ret %", "profit_factor": "PF", "sharpe": "Sharpe", "calmar": "Calmar"}.get(
        metric, metric[:7])


def format_params(params: dict[str, Any]) -> str:
    """``fast=12, slow=100, tp_r=2`` (floats sin ceros de sobra)."""
    out = []
    for k, v in (params or {}).items():
        out.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
    return ", ".join(out) or "(por defecto)"


# --- piezas del walk-forward ----------------------------------------------------------

def walk_forward_splits(n_bars: int, folds: int, min_segment: int = MIN_SEGMENT_BARS
                        ) -> list[tuple[int, int, int]]:
    """Tramos anclados ``[(train_end, test_start, test_end)]`` sobre ``n_bars`` velas (el entrenamiento
    siempre empieza en 0 y ``train_end == test_start``). ``folds + 1`` bloques iguales (el último se
    queda el resto); si no caben bloques de ``min_segment`` velas se usan menos tramos ([] si ni uno)."""
    folds = min(int(folds), n_bars // max(int(min_segment), 1) - 1)
    if folds < 1:
        return []
    block = n_bars // (folds + 1)
    bounds = [block * k for k in range(1, folds + 1)] + [n_bars]
    return [(bounds[k], bounds[k], bounds[k + 1]) for k in range(folds)]


def _strategy_key(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def _default_combo(cls: type[Strategy], settings: Settings) -> dict[str, Any]:
    """Valores por defecto de los parámetros del grid (``tp_r=None`` -> ``default_tp_r`` actual: así los
    parámetros propuestos no dependen de un ajuste que puede cambiar en caliente)."""
    out = {}
    for k in cls.PARAM_GRID:
        v = getattr(cls, k, None)
        if v is None and k == "tp_r":
            v = float(settings.default_tp_r)
        out[k] = v
    return out


def param_combos(cls: type[Strategy], settings: Settings, max_combos: int, seed: int = 7) -> list[dict[str, Any]]:
    """Combinaciones del ``PARAM_GRID`` a probar: todas si caben en ``max_combos``; si no, una muestra
    determinista (misma semilla -> mismas combinaciones, en cualquier máquina). La configuración por
    defecto va siempre la primera (a igualdad de puntuación, gana). Las inválidas se descartan."""
    grid = {k: list(v) for k, v in cls.PARAM_GRID.items() if isinstance(v, (list, tuple)) and len(v)}
    keys = list(grid)
    valid = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, values))
        if _valid_combo(cls, settings, combo):
            valid.append(combo)
    default = {k: v for k, v in _default_combo(cls, settings).items() if k in grid}
    head = [default] if _valid_combo(cls, settings, default) else []
    pool = [c for c in valid if c != default]
    room = max(int(max_combos), 1) - len(head)
    if len(pool) > room:
        rng = random.Random(f"{int(seed)}:{cls.name}")          # semilla str: estable entre procesos
        pick = sorted(rng.sample(range(len(pool)), max(room, 0)))
        pool = [pool[i] for i in pick]
    return (head + pool)[:max(int(max_combos), 1)]


def _valid_combo(cls: type[Strategy], settings: Settings, combo: dict[str, Any]) -> bool:
    try:
        cls(settings, **combo)
    except ValueError:
        return False
    return True


def _resolve(strategies: Optional[Iterable[Union[str, type[Strategy]]]]) -> list[type[Strategy]]:
    """Clases a evaluar (nombres normalizados o clases), sin duplicados. Nombre desconocido -> ``ValueError``."""
    if strategies is None:
        return list(STRATEGIES.values())
    if isinstance(strategies, str):
        strategies = [strategies]
    out: list[type[Strategy]] = []
    for item in strategies:
        if isinstance(item, type) and issubclass(item, Strategy):
            cls = item
        else:
            cls = STRATEGIES.get(_strategy_key(item))
            if cls is None:
                raise ValueError(f"Estrategia desconocida: {item!r}. Disponibles: {', '.join(sorted(STRATEGIES))}")
        if cls not in out:
            out.append(cls)
    return out


def _clean_bars(bars: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    if bars is None or any(c not in bars.columns for c in CANDLE_COLUMNS):
        raise ValueError(f"velas sin columnas {CANDLE_COLUMNS}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("las velas deben tener un DatetimeIndex (apertura de cada vela)")
    if not bars.index.is_monotonic_increasing:
        bars = bars.sort_index()
        notes.append("velas desordenadas: se ordenaron por fecha")
    if bars.index.has_duplicates:
        dup = int(bars.index.duplicated(keep="last").sum())
        bars = bars[~bars.index.duplicated(keep="last")]
        notes.append(f"{dup} velas repetidas descartadas")
    return bars


def _fingerprint(sig: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Lo que decide una operación: lado (1/-1/0) y niveles (entrada, SL, TP)."""
    side = sig["side"].to_numpy()
    code = np.where(side == "long", 1, np.where(side == "short", -1, 0)).astype(np.int8)
    levels = sig[["entry_px", "sl", "tp"]].to_numpy(dtype=float)
    return code, levels


def _same_prefix(fp: tuple[np.ndarray, np.ndarray], sig: pd.DataFrame, upto: int) -> bool:
    code, levels = _fingerprint(sig.iloc[:upto])
    return bool(np.array_equal(code, fp[0][:upto])
                and np.allclose(levels, fp[1][:upto], rtol=1e-9, atol=1e-12, equal_nan=True))


def _ts(index: pd.DatetimeIndex, pos: int) -> float:
    return float(index[pos].timestamp())


@dataclass
class _Context:
    coin: str
    interval: str
    bars: pd.DataFrame
    settings: Settings
    meta: Optional[AssetMeta]
    zones: Optional[pd.DataFrame]
    splits: list[tuple[int, int, int]]
    objective: str                     # clave de métrica
    seed: int
    clock: Optional[Clock]
    step: int
    compared: int                      # estrategias comparadas (corrección de la significación)
    tick: Callable[[int, str], None]
    stop: Callable[[], bool]


def _min_train_trades(settings: Settings, train_bars: int, oos_bars: int) -> int:
    """Operaciones mínimas al entrenar: la mitad de las que, a ritmo de ``bt_min_trades`` en todo lo fuera de
    muestra, tocarían a ``train_bars`` velas (y nunca menos de ``MIN_TRAIN_TRADES``)."""
    need = TRAIN_TRADES_FRAC * max(settings.bt_min_trades, 0) * train_bars / max(oos_bars, 1)
    return max(MIN_TRAIN_TRADES, int(math.ceil(need)))


def _evaluate(cls: type[Strategy], combos: list[dict[str, Any]], ctx: _Context) -> StrategyEvaluation:
    s, bars, splits = ctx.settings, ctx.bars, ctx.splits
    n_folds = len(splits)
    strategies = [cls(s, ctx.clock, interval=ctx.interval, **c) for c in combos]
    zones = ctx.zones if cls.uses_zones else None
    # 1) entrenamiento: señales una vez por combinación, una simulación por tramo (sólo sus velas)
    scores = np.zeros((len(strategies), n_folds))
    trades = np.zeros((len(strategies), n_folds), dtype=np.int64)
    train_metrics: list[list[dict[str, float]]] = []
    prints = []
    for ci, strat in enumerate(strategies):
        if ctx.stop():
            raise _Stopped
        sig = strat.signals(bars, zones)
        prints.append(_fingerprint(sig))
        row = []
        for k, (train_end, _, _) in enumerate(splits):
            res = run_backtest(strat, bars.iloc[:train_end], s, coin=ctx.coin, meta=ctx.meta, signals=sig)
            row.append(res.metrics)
            scores[ci, k] = res.metrics.get(ctx.objective, 0.0)
            trades[ci, k] = res.metrics.get("n_trades", 0)
        train_metrics.append(row)
        ctx.tick(n_folds, f"{cls.name}: combinación {ci + 1}/{len(strategies)}")
    # 2) elección por tramo y validación en el bloque siguiente (capital encadenado)
    oos_bars = sum(e - t for _, t, e in splits)
    equity0 = float(s.paper_start_equity)
    eq = equity0
    folds: list[FoldResult] = []
    all_trades: list[BTTrade] = []
    curves: list[pd.Series] = []
    n_signals = n_skipped = 0
    for k, (train_end, test_start, test_end) in enumerate(splits):
        if ctx.stop():
            raise _Stopped
        need = _min_train_trades(s, train_end, oos_bars)
        best = max(range(len(strategies)),
                   key=lambda ci: (trades[ci, k] >= need, scores[ci, k], trades[ci, k], -ci))
        strat = strategies[best]
        sub = bars.iloc[:test_end]
        sig = strat.signals(sub, zones.iloc[:test_end] if zones is not None else None)
        if not _same_prefix(prints[best], sig, test_end):
            raise _Leak(f"las señales de {cls.name} cambian al añadir velas posteriores (mira el futuro): "
                        "evaluación no válida")
        res = run_backtest(strat, sub, s, coin=ctx.coin, meta=ctx.meta, start_equity=eq, signals=sig,
                           trade_from=test_start)
        if len(res.equity):
            eq = float(res.equity.iloc[-1])
        all_trades += res.trades
        curves.append(res.equity)
        n_signals += int(res.metrics.get("n_signals", 0))
        n_skipped += int(res.metrics.get("n_skipped", 0))
        idx = bars.index
        folds.append(FoldResult(
            fold=k + 1, train_start_ts=_ts(idx, 0), train_end_ts=_ts(idx, train_end - 1) + ctx.step,
            test_start_ts=_ts(idx, test_start), test_end_ts=_ts(idx, test_end - 1) + ctx.step,
            train_bars=train_end, test_bars=test_end - test_start, params=strat.params(),
            is_metrics=dict(train_metrics[best][k]), oos_metrics=dict(res.metrics), combos=len(strategies),
            min_train_trades=need))
        ctx.tick(1, f"{cls.name}: tramo {k + 1}/{n_folds} validado")
    equity = pd.concat(curves) if curves else pd.Series(dtype=float)
    end_ts = folds[-1].test_end_ts
    oos = compute_metrics(all_trades, equity, equity0, ctx.interval, n_bars=oos_bars, end_ts=end_ts)
    oos["n_signals"], oos["n_skipped"] = n_signals, n_skipped
    oos["t_stat"] = edge_t_stat(all_trades)
    oos["t_min"] = required_t_stat(len(all_trades), ctx.compared)
    last = folds[-1]
    score = float(oos.get(ctx.objective, 0.0))
    reasons = approval_reasons(oos, s)
    result = BacktestResult(coin=ctx.coin, interval=ctx.interval, strategy=cls.name, params=dict(last.params),
                            start_ts=folds[0].test_start_ts, end_ts=end_ts, bars=oos_bars, trades=all_trades,
                            equity=equity, metrics=dict(oos))
    return StrategyEvaluation(strategy=cls.name, params=dict(last.params), is_metrics=dict(last.is_metrics),
                              oos_metrics=oos, oos_trades=all_trades, folds=folds, score=score,
                              passed=not reasons, reasons=reasons, combos=len(strategies), oos_result=result)


def edge_t_stat(trades: list[BTTrade]) -> float:
    """t de Student del R medio por operación (¿la ventaja es distinguible de cero?). 0 con menos de 2."""
    r = np.array([t.r_multiple for t in trades], dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    if not sd > 1e-12 * max(abs(mean), 1.0):
        return math.copysign(T_CAP, mean) if mean else 0.0
    return float(min(max(mean / (sd / math.sqrt(len(r))), -T_CAP), T_CAP))


def required_t_stat(n_trades: int, compared: int = 1, alpha: float = SIGNIFICANCE_ALPHA) -> float:
    """t mínima para aprobar: cuantil ``1 - alpha/compared`` de una t de Student con ``n_trades - 1`` grados de
    libertad (Bonferroni por las estrategias comparadas; expansión de Cornish-Fisher, sin scipy)."""
    p = 1.0 - min(max(float(alpha), 1e-9), 0.5) / max(int(compared), 1)
    z = NormalDist().inv_cdf(p)
    df = float(max(int(n_trades) - 1, 1))
    return (z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))


def approval_reasons(oos: dict[str, float], settings: Settings) -> list[str]:
    """Motivos (en español) por los que unas métricas fuera de muestra NO aprueban ([] = aprobada).
    La significación sólo se exige si ``oos`` trae ``t_stat`` y ``t_min`` (los pone ``optimize``)."""
    reasons = []
    n = int(oos.get("n_trades", 0))
    pf = float(oos.get("profit_factor", 0.0))
    ret = float(oos.get("total_return_pct", 0.0))
    dd = float(oos.get("max_drawdown_pct", 0.0))
    if n < settings.bt_min_trades:
        reasons.append(f"pocas operaciones fuera de muestra: {n} < {settings.bt_min_trades}")
    if pf < settings.bt_min_profit_factor:
        reasons.append(f"profit factor fuera de muestra {pf:.2f} < {settings.bt_min_profit_factor:.2f}")
    if not ret > 0:
        reasons.append(f"retorno fuera de muestra {ret:+.2f} % (debe ser > 0)")
    if dd > settings.bt_max_drawdown_pct:
        reasons.append(f"drawdown fuera de muestra {dd:.1f} % > {settings.bt_max_drawdown_pct:.1f} %")
    if "t_stat" in oos and "t_min" in oos and n >= 2 and float(oos["t_stat"]) < float(oos["t_min"]):
        t = float(oos["t_stat"])
        shown = ("-∞" if t < 0 else "∞") if abs(t) >= T_CAP else f"{t:.2f}"
        if t > 0:
            reasons.append(f"ventaja no significativa: t = {shown} < {float(oos['t_min']):.2f} "
                           f"(con {n} operaciones puede ser suerte)")
        else:
            reasons.append(f"R medio por operación <= 0 fuera de muestra (t = {shown})")
    return reasons


def _failed(name: str, reason: str, error: str = "") -> StrategyEvaluation:
    return StrategyEvaluation(strategy=name, params={}, is_metrics={}, oos_metrics={}, oos_trades=[], folds=[],
                              score=None, passed=False, reasons=[reason], error=error or reason)


def _rank_key(e: StrategyEvaluation) -> tuple:
    return (not e.passed, e.score is None, -(e.score if e.score is not None else 0.0), e.strategy)


# --- optimize -------------------------------------------------------------------------

def optimize(coin: str, interval: str, bars: pd.DataFrame, settings: Settings, *,
             strategies: Optional[Iterable[Union[str, type[Strategy]]]] = None, meta: Optional[AssetMeta] = None,
             seed: int = 7, progress: Optional[ProgressFn] = None, should_stop: Optional[StopFn] = None,
             clock: Optional[Clock] = None, zones: Optional[pd.DataFrame] = None) -> OptimizationReport:
    """Walk-forward de cada estrategia sobre ``bars`` (velas **cerradas** de ``coin`` en ``interval``).

    - ``strategies``: nombres (o clases) a evaluar; None = todas las de ``STRATEGIES``. Cada una se
      evalúa en la temporalidad de la gráfica. Nombre desconocido -> ``ValueError``.
    - ``seed``: semilla del muestreo de combinaciones (misma semilla y datos -> mismo informe).
    - ``progress(hechas, total, texto)``: avance (unidades = simulaciones); sus errores se ignoran.
    - ``should_stop()``: se consulta entre combinaciones; True -> se devuelve lo terminado
      (``complete=False``).
    - ``clock``: hora del informe (``created_ts``); sin él, la real. ``zones``: zonas por vela para
      las estrategias de liquidez (sin ellas, ``liquidity.historical_zones`` una vez).

    Una estrategia que falla (error, historia corta, fuga de datos) sale rechazada con el motivo; nunca
    tumba la optimización. ``ValueError`` sólo por entradas inválidas (columnas, índice, nombres).
    """
    notes: list[str] = []
    classes = _resolve(strategies)
    step = interval_seconds(interval)
    bars = _clean_bars(bars, notes)
    clk = clock if clock is not None else RealClock()
    created = float(clk.now())
    n = len(bars)
    obj_name = str(settings.bt_objective or "").strip().lower()
    if obj_name not in OBJECTIVES:
        notes.append(f"objetivo {settings.bt_objective!r} desconocido: se usa {DEFAULT_OBJECTIVE}")
        obj_name = DEFAULT_OBJECTIVE
    objective = OBJECTIVES[obj_name]
    want = int(settings.bt_folds)
    if want < 1:
        notes.append(f"bt_folds={settings.bt_folds} no vale: se usa 1 tramo")
        want = 1
    splits = walk_forward_splits(n, want)
    if splits and len(splits) < want:
        notes.append(f"historia corta ({n} velas): {len(splits)} tramos en vez de {want}")
    criteria = {"objective": obj_name, "folds": settings.bt_folds, "max_combos": settings.bt_max_combos,
                "min_trades": settings.bt_min_trades, "min_profit_factor": settings.bt_min_profit_factor,
                "max_drawdown_pct": settings.bt_max_drawdown_pct, "alpha": SIGNIFICANCE_ALPHA,
                "compared": len(classes)}
    start_ts = _ts(bars.index, 0) if n else 0.0
    end_ts = _ts(bars.index, n - 1) + step if n else 0.0
    report = OptimizationReport(
        id=_report_id(coin, interval, created, bars, seed, classes), coin=coin, interval=interval,
        days=round((end_ts - start_ts) / 86_400.0, 2), bars=n, created_ts=created, evaluations=[], winner=None,
        notes=notes, objective=obj_name, folds=len(splits), seed=int(seed), criteria=criteria,
        start_ts=start_ts, end_ts=end_ts, oos_start_ts=_ts(bars.index, splits[0][1]) if splits else 0.0)
    if not splits:
        need = 2 * MIN_SEGMENT_BARS
        reason = f"historia insuficiente: {n} velas (hacen falta al menos {need} para un tramo walk-forward)"
        report.evaluations = [_failed(c.name, reason) for c in classes]
        return report

    combos = {c: param_combos(c, settings, settings.bt_max_combos, seed) for c in classes}
    total = sum(len(combos[c]) * len(splits) + len(splits) for c in classes)
    done = [0, -1, False]                                # [simulaciones hechas, último % avisado, callback roto]

    def tick(units: int, text: str) -> None:
        done[0] += units
        pct = int(done[0] * 100 / max(total, 1))
        if progress is None or done[2] or (pct < done[1] + 5 and done[0] < total):
            return
        done[1] = pct
        try:
            progress(done[0], total, text)
        except Exception:  # noqa: BLE001 - el aviso de progreso nunca para la optimización
            done[2] = True                                   # se registra una vez y no se vuelve a llamar
            log.exception("optimize: fallo en el callback de progreso (no se usará más en esta optimización)")

    def stop() -> bool:
        if should_stop is None:
            return False
        try:
            return bool(should_stop())
        except Exception:  # noqa: BLE001
            log.exception("optimize: fallo en should_stop (se sigue)")
            return False

    if zones is None and any(c.uses_zones for c in classes):
        try:
            zones = historical_zones(bars, settings)
        except Exception as e:  # noqa: BLE001 - sin zonas, las estrategias de liquidez no operan (y se rechazan)
            log.exception("optimize %s %s: fallo calculando zonas históricas", coin, interval)
            notes.append(f"no se pudieron calcular las zonas históricas ({e}): las estrategias de liquidez no "
                         "tienen señales")
    elif zones is not None and not zones.index.equals(bars.index):
        zones = zones.reindex(bars.index)
    ctx = _Context(coin=coin, interval=interval, bars=bars, settings=settings, meta=meta, zones=zones,
                   splits=splits, objective=objective, seed=int(seed), clock=clock, step=step,
                   compared=len(classes), tick=tick, stop=stop)
    evaluations = []
    for cls in classes:
        try:
            if not combos[cls]:
                evaluations.append(_failed(cls.name, "sin combinaciones de parámetros válidas"))
                continue
            evaluations.append(_evaluate(cls, combos[cls], ctx))
        except _Stopped:
            report.complete = False
            notes.append(f"detenida por petición antes de terminar ({len(evaluations)} de {len(classes)} "
                         "estrategias evaluadas)")
            break
        except _Leak as e:
            log.error("optimize %s %s: %s", coin, interval, e)
            evaluations.append(_failed(cls.name, str(e)))
        except Exception as e:  # noqa: BLE001 - una estrategia rota no tumba el resto
            log.exception("optimize %s %s: fallo evaluando %s", coin, interval, cls.name)
            evaluations.append(_failed(cls.name, f"error al evaluar: {e}", repr(e)))
    evaluations.sort(key=_rank_key)
    report.evaluations = evaluations
    win = next((e for e in evaluations if e.passed), None)
    report.winner = win.strategy if win is not None else None
    if any(c.uses_zones for c in classes):
        names = ", ".join(c.name for c in classes if c.uses_zones)
        notes.append(f"{names}: en backtest las zonas de liquidez son aproximadas (pivotes, sin libro ni "
                     "liquidaciones históricas); su resultado es sólo orientativo")
    if report.complete and progress is not None and done[0] < total:
        tick(total - done[0], "optimización terminada")
    return report


def _report_id(coin: str, interval: str, created: float, bars: pd.DataFrame, seed: int,
               classes: list[type[Strategy]]) -> str:
    n = len(bars)
    span = f"{bars.index[0].value}-{bars.index[-1].value}" if n else "0"
    raw = f"{coin}|{interval}|{created!r}|{span}|{n}|{seed}|{','.join(c.name for c in classes)}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=4).hexdigest()


# --- informes guardados ---------------------------------------------------------------

REPORTS_DIR = "reports"
_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe(part: str) -> str:
    return _SAFE.sub("-", str(part)).strip("-") or "x"


class ReportStore:
    """Informes en ``<state_dir>/reports/<COIN>_<interval>_<id>.json`` = ``{"report": ..., "result": ...}``
    (``result`` = ``BacktestResult.to_dict()`` de la curva que se enseña, o None). Escritura atómica; un
    fichero ilegible se ignora (y se avisa en el log), nunca tumba a quien lee. Guarda como mucho ``keep``
    informes: al guardar se borran los más antiguos."""

    def __init__(self, state_dir: Union[str, Path], *, keep: int = 200):
        self.dir = Path(state_dir) / REPORTS_DIR
        self.keep = max(int(keep), 1)

    def path_for(self, report: OptimizationReport) -> Path:
        return self.dir / f"{_safe(report.coin)}_{_safe(report.interval)}_{_safe(report.id)}.json"

    def save(self, report: OptimizationReport, result: Optional[BacktestResult] = None) -> Path:
        """Guarda el informe (y la curva ``result``; por defecto la de ``report.result()`` si está en memoria)."""
        if result is None:
            result = report.result()
        path = self.path_for(report)
        journal = Journal(self.dir)
        journal.save_json(path.name, {"report": report.to_dict(),
                                      "result": result.to_dict() if result is not None else None})
        self._prune(keep_path=path)
        return path

    # ------------------------------------------------------------------ lectura
    def _files(self, pattern: str = "*.json") -> list[Path]:
        """Ficheros de informe, el más reciente primero (por fecha de modificación)."""
        try:
            files = [p for p in self.dir.glob(pattern) if p.is_file()]
        except OSError:
            return []
        dated = []
        for p in files:
            try:
                dated.append((p.stat().st_mtime, p.name, p))
            except OSError:
                continue
        return [p for _, _, p in sorted(dated, reverse=True)]

    @staticmethod
    def _read(path: Path) -> Optional[dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("informe ilegible %s: %s", path.name, e)
            return None
        return data if isinstance(data, dict) else None

    def _load(self, path: Path) -> Optional[OptimizationReport]:
        data = self._read(path)
        if data is None:
            return None
        try:
            return OptimizationReport.from_dict(data)
        except Exception as e:  # noqa: BLE001 - contenido raro: se ignora ese informe
            log.warning("informe inválido %s: %s", path.name, e)
            return None

    def list(self, limit: int = 10) -> list[OptimizationReport]:
        """Los ``limit`` informes más recientes (por ``created_ts``)."""
        out = []
        for p in self._files():
            rep = self._load(p)
            if rep is not None:
                out.append(rep)
            if len(out) >= max(int(limit), 0) * 2 + 5:     # leer de más por si las fechas no cuadran con el mtime
                break
        out.sort(key=lambda r: r.created_ts, reverse=True)
        return out[:max(int(limit), 0)]

    def latest(self, coin: str, interval: str) -> Optional[OptimizationReport]:
        """El informe más reciente de la gráfica (moneda + temporalidad) o None."""
        best = None
        for p in self._files(f"{_safe(coin)}_{_safe(interval)}_*.json"):
            rep = self._load(p)
            if rep is None or rep.coin != coin or rep.interval != interval:
                continue
            if best is None or rep.created_ts > best.created_ts:
                best = rep
        return best

    def get(self, report_id: str) -> Optional[OptimizationReport]:
        p = self._find(report_id)
        return self._load(p) if p is not None else None

    def get_result(self, report_id: str) -> Optional[BacktestResult]:
        """La curva guardada con el informe (para volver a pintarlo) o None."""
        p = self._find(report_id)
        data = self._read(p) if p is not None else None
        res = data.get("result") if data else None
        if not isinstance(res, dict):
            return None
        try:
            return BacktestResult.from_dict(res)
        except Exception as e:  # noqa: BLE001
            log.warning("curva inválida en %s: %s", p.name, e)
            return None

    def _find(self, report_id: str) -> Optional[Path]:
        rid = _safe(str(report_id or "").strip())
        if not rid or rid == "x":
            return None
        files = self._files(f"*_{rid}.json")
        return files[0] if files else None

    def _prune(self, keep_path: Path) -> None:
        for p in self._files()[self.keep:]:
            if p == keep_path:
                continue
            try:
                os.remove(p)
            except OSError as e:
                log.warning("no se pudo borrar el informe antiguo %s: %s", p.name, e)


__all__ = ["OBJECTIVES", "DISCLAIMER", "FoldResult", "StrategyEvaluation", "OptimizationReport", "ReportStore",
           "optimize", "param_combos", "walk_forward_splits", "approval_reasons", "edge_t_stat", "required_t_stat",
           "format_params"]

"""Informes de backtest para personas: HTML autocontenido, PNG y mini-gráficas de texto.

- ``render_html(report, result, bars)``: una página HTML sin dependencias (SVG en línea, CSS y un poco de
  JavaScript opcional para la lectura al pasar el ratón): precio con las entradas/salidas fuera de muestra
  y los tramos walk-forward sombreados, curva de capital, drawdown, tabla de estrategias (dentro vs fuera
  de muestra), tramos y operaciones. Se lee en claro y en oscuro (``prefers-color-scheme``) y se adapta al
  móvil. Todo el texto que viene de datos se escapa.
- ``render_png(report, result, bars)``: lo mismo en una imagen (matplotlib con el backend Agg, sin
  ventanas ni estado global) para Telegram; None si matplotlib no está o algo falla (nunca lanza).
- ``sparkline(values, width=40)``: bloques unicode para la consola, el panel y Telegram.

``result`` es normalmente la curva **fuera de muestra** encadenada (``OptimizationReport.result()`` o
``ReportStore.get_result(id)``): es la única que se enseña como resultado. Cualquiera de los tres
argumentos puede faltar (None): se pinta lo que se pueda.
"""

from __future__ import annotations

import html
import io
import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .backtest import BacktestResult, BTTrade
from .exchange.intervals import interval_seconds

log = logging.getLogger(__name__)

SPARK = "▁▂▃▄▅▆▇█"
MAX_PRICE_POINTS = 900          # el precio se resume (máximo/mínimo/cierre por cubo) por encima de esto
MAX_CURVE_POINTS = 600
MAX_TRADE_ROWS = 60
_MONTHS = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")

DISCLAIMER = ("Un backtest no garantiza resultados futuros. Las cifras dentro de muestra son optimistas (los "
              "parámetros se eligieron con esos datos); sólo cuentan las de fuera de muestra, y aun así el mercado "
              "puede cambiar. Incluye comisiones y deslizamiento; no incluye funding ni liquidaciones.")

# Paleta (instancia de referencia validada): una serie (azul), estados fijos (bien/mal) y tinta neutra.
LIGHT = {"surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
         "grid": "#e1e0d9", "axis": "#c3c2b7", "series": "#2a78d6", "good": "#0ca30c", "bad": "#d03b3b",
         "good_text": "#006300", "warn": "#fab219"}
DARK = {"surface": "#1a1a19", "page": "#0d0d0d", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "series": "#3987e5", "good": "#0ca30c", "bad": "#d03b3b",
        "good_text": "#0ca30c", "warn": "#fab219"}


# --- texto ----------------------------------------------------------------------------

def sparkline(values: Iterable[Any], width: int = 40) -> str:
    """Mini-gráfica con bloques unicode (``▁▂▃▄▅▆▇█``) de toda la serie resumida a ``width`` puntos como
    mucho (muestreo uniforme que conserva el primero y el último). Valores no numéricos o no finitos se
    ignoran; con menos de 2 valores, cadena vacía; una serie plana, bloques medios."""
    vals = []
    for v in values if values is not None else ():
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and not isinstance(v, bool):
            vals.append(x)
    width = max(int(width), 2)
    if len(vals) < 2:
        return ""
    if len(vals) > width:
        pos = np.linspace(0, len(vals) - 1, width).round().astype(int)
        vals = [vals[i] for i in pos]
    lo, hi = min(vals), max(vals)
    if hi - lo <= 0:
        return SPARK[len(SPARK) // 2] * len(vals)
    top = len(SPARK) - 1
    return "".join(SPARK[int(round((v - lo) / (hi - lo) * top))] for v in vals)


def _esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def _fin(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _day(ts: float, *, hours: bool = False) -> str:
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    out = f"{d.day} {_MONTHS[d.month - 1]} {d.year}"
    return out + (f" {d:%H:%M}" if hours else "")


def _fmt_px(x: Optional[float]) -> str:
    if x is None:
        return "—"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 1:
        return f"{x:,.2f}"
    return f"{x:.6g}"


def _compact(x: float) -> str:
    """Etiqueta de eje: 84.2k, 1.5M, 0.0123."""
    ax = abs(x)
    if ax >= 1e6:
        return f"{x / 1e6:.3g}M"
    if ax >= 1e4:
        return f"{x / 1e3:.3g}k"
    if ax >= 100:
        return f"{x:.0f}"
    return f"{x:.4g}"


def _pct(v: Any, nd: int = 1, signed: bool = True) -> str:
    x = _fin(v)
    return "—" if x is None else (f"{x:+.{nd}f} %" if signed else f"{x:.{nd}f} %")


def _pf(v: Any) -> str:
    x = _fin(v)
    return "—" if x is None else ("∞" if x >= 999 else f"{x:.2f}")


def _n(v: Any, nd: int = 2) -> str:
    x = _fin(v)
    return "—" if x is None else f"{x:.{nd}f}"


def _t(v: Any) -> str:
    """t de la ventaja: ±∞ cuando no hay dispersión (todas las operaciones iguales)."""
    x = _fin(v)
    if x is None:
        return "—"
    return ("∞" if x > 0 else "−∞") if abs(x) >= 99 else f"{x:.1f}"


# --- datos ----------------------------------------------------------------------------

def _index_seconds(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return index.as_unit("ns").asi8.astype(np.float64) / 1e9
    return np.asarray(index, dtype=float)


def _price_series(bars: Optional[pd.DataFrame]) -> Optional[dict[str, np.ndarray]]:
    """ts (apertura), cierre, máximo y mínimo; resumido a ``MAX_PRICE_POINTS`` cubos (máx/mín/último cierre)."""
    if bars is None or len(bars) == 0 or not {"high", "low", "close"} <= set(bars.columns):
        return None
    ts = _index_seconds(bars.index)
    c, h, lo = (bars[k].to_numpy(dtype=float) for k in ("close", "high", "low"))
    ok = np.isfinite(ts) & np.isfinite(c)
    if ok.sum() < 2:
        return None
    ts, c, h, lo = ts[ok], c[ok], np.where(np.isfinite(h[ok]), h[ok], c[ok]), np.where(np.isfinite(lo[ok]), lo[ok], c[ok])
    n = len(ts)
    if n > MAX_PRICE_POINTS:
        edges = np.linspace(0, n, MAX_PRICE_POINTS + 1).astype(int)
        starts = np.unique(edges[:-1])
        ends = np.r_[starts[1:], n] - 1
        h, lo = np.maximum.reduceat(h, starts), np.minimum.reduceat(lo, starts)
        ts, c = ts[ends], c[ends]
    return {"ts": ts, "close": c, "high": h, "low": lo}


def _curve(series: Optional[pd.Series], max_points: int = MAX_CURVE_POINTS) -> tuple[np.ndarray, np.ndarray]:
    if series is None or len(series) == 0:
        return np.array([]), np.array([])
    ts = _index_seconds(series.index)
    v = series.to_numpy(dtype=float)
    ok = np.isfinite(ts) & np.isfinite(v)
    ts, v = ts[ok], v[ok]
    if len(v) > max_points:
        pos = np.unique(np.linspace(0, len(v) - 1, max_points).round().astype(int))
        # el punto más bajo cuenta (el drawdown máximo no puede desaparecer al resumir)
        pos = np.unique(np.r_[pos, int(np.argmin(v)), int(np.argmax(v))])
        ts, v = ts[pos], v[pos]
    return ts, v


def _start_equity(result: Optional[BacktestResult], eq: np.ndarray) -> Optional[float]:
    """Capital inicial de la curva (el retorno total es relativo a él): la base de capital y drawdown."""
    tr = _fin(result.metrics.get("total_return_pct")) if result is not None and len(eq) else None
    return float(eq[-1]) / (1 + tr / 100.0) if tr is not None and tr > -100 else None


def _drawdown(values: np.ndarray, start: Optional[float]) -> np.ndarray:
    if len(values) == 0:
        return values
    curve = np.r_[float(start), values] if start is not None and start > 0 else values
    peak = np.maximum.accumulate(curve)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(peak > 0, (curve - peak) / peak * 100.0, 0.0)
    return dd[1:] if len(curve) > len(values) else dd


def _evaluation(report: Any, result: Optional[BacktestResult]) -> Any:
    if report is None:
        return None
    if result is not None and hasattr(report, "get"):
        ev = report.get(result.strategy)
        if ev is not None:
            return ev
    return report.best() if hasattr(report, "best") else None


def _oos_windows(report: Any, result: Optional[BacktestResult]) -> list[tuple[float, float, int]]:
    """(inicio, fin, tramo) de los bloques fuera de muestra de la estrategia mostrada."""
    ev = _evaluation(report, result)
    if ev is not None and getattr(ev, "folds", None):
        return [(f.test_start_ts, f.test_end_ts, f.fold) for f in ev.folds if f.test_end_ts > f.test_start_ts]
    if report is not None and getattr(report, "oos_start_ts", 0) and getattr(report, "end_ts", 0):
        return [(report.oos_start_ts, report.end_ts, 0)]
    if result is not None and result.end_ts > result.start_ts:
        return [(result.start_ts, result.end_ts, 0)]
    return []


# --- SVG ------------------------------------------------------------------------------

W = 720.0
ML, MR, MT, MB = 62.0, 14.0, 14.0, 28.0


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    if hi <= lo:
        hi, lo = lo + (abs(lo) * 0.01 or 1.0), lo - (abs(lo) * 0.01 or 1.0)
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step) * step
    out, v = [], first
    while v <= hi + step * 1e-9 and len(out) < 20:
        out.append(round(v, 12))
        v += step
    return out


_TIME_STEPS = (3600, 3 * 3600, 6 * 3600, 12 * 3600, 86400, 2 * 86400, 7 * 86400, 14 * 86400, 30 * 86400,
               61 * 86400, 91 * 86400, 182 * 86400, 365 * 86400)


def _time_ticks(t0: float, t1: float, n: int = 6) -> tuple[list[float], int]:
    """Marcas del eje de tiempo alineadas a horas/días UTC y el paso elegido (s)."""
    span = max(t1 - t0, 1.0)
    year = _TIME_STEPS[-1]
    step = next((s for s in _TIME_STEPS if span / s <= n), year * math.ceil(span / (n * year)))
    first = math.ceil(t0 / step) * step
    return [float(t) for t in np.arange(first, t1 + 1e-6, step)][:n + 1], step


def _tick_label(ts: float, step: int, prev: Optional[float]) -> str:
    """``8 mar`` (con el año en la primera marca o al cambiar de año); con pasos de horas, ``8 mar 14:00``."""
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    out = f"{d.day} {_MONTHS[d.month - 1]}"
    if step < 86400:
        return out + f" {d:%H:%M}"
    if prev is None or datetime.fromtimestamp(prev, tz=timezone.utc).year != d.year:
        out += f" {d.year}"
    return out


class _Plot:
    """Ejes y escalas de un gráfico SVG (coordenadas de usuario: ancho ``W``)."""

    def __init__(self, h: float, x0: float, x1: float, y0: float, y1: float, pad: float = 0.06):
        self.h = h
        if x1 <= x0:
            x1 = x0 + 1.0
        if not y1 > y0:
            y0, y1 = y0 - (abs(y0) * 0.01 or 1.0), y1 + (abs(y1) * 0.01 or 1.0)
        span = y1 - y0
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0 - span * pad, y1 + span * pad
        self.left, self.right, self.top, self.bottom = ML, W - MR, MT, h - MB

    def x(self, t: float) -> float:
        return self.left + (t - self.x0) / (self.x1 - self.x0) * (self.right - self.left)

    def y(self, v: float) -> float:
        return self.bottom - (v - self.y0) / (self.y1 - self.y0) * (self.bottom - self.top)

    def path(self, ts: np.ndarray, vs: np.ndarray) -> str:
        parts, pen = [], "M"
        for t, v in zip(ts.tolist(), vs.tolist()):
            if not (math.isfinite(t) and math.isfinite(v)):
                pen = "M"
                continue
            parts.append(f"{pen}{self.x(t):.1f},{self.y(v):.1f}")
            pen = "L"
        return "".join(parts)

    def area(self, ts: np.ndarray, top: np.ndarray, bottom: np.ndarray) -> str:
        """Polígono entre ``top`` y ``bottom`` (los puntos no finitos se omiten)."""
        ok = np.isfinite(ts) & np.isfinite(top) & np.isfinite(bottom)
        ts, top, bottom = ts[ok], top[ok], bottom[ok]
        if len(ts) < 2:
            return ""
        pts = [f"{self.x(t):.1f},{self.y(v):.1f}" for t, v in zip(ts.tolist(), top.tolist())]
        pts += [f"{self.x(t):.1f},{self.y(v):.1f}" for t, v in zip(ts[::-1].tolist(), bottom[::-1].tolist())]
        return "M" + "L".join(pts) + "Z"

    def axes(self, yfmt=_compact, y_ticks: Optional[list[float]] = None) -> str:
        out = []
        for v in (y_ticks if y_ticks is not None else _nice_ticks(self.y0, self.y1)):
            if not (self.y0 <= v <= self.y1):
                continue
            y = self.y(v)
            out.append(f'<line class="grid" x1="{self.left:.1f}" x2="{self.right:.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
                       f'<text class="tick" x="{self.left - 6:.1f}" y="{y + 4:.1f}" text-anchor="end">'
                       f"{_esc(yfmt(v))}</text>")
        ticks, step = _time_ticks(self.x0, self.x1)
        prev = None
        for i, t in enumerate(ticks):
            if not (self.x0 <= t <= self.x1):
                continue
            x = self.x(t)
            label = _tick_label(t, step, prev)
            prev = t
            cls = "tick alt" if i % 2 else "tick"
            out.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{self.bottom:.1f}" y2="{self.bottom + 4:.1f}"/>'
                       f'<text class="{cls}" x="{x:.1f}" y="{self.bottom + 18:.1f}" text-anchor="middle">'
                       f"{_esc(label)}</text>")
        out.append(f'<line class="axis" x1="{self.left:.1f}" x2="{self.right:.1f}" y1="{self.bottom:.1f}" '
                   f'y2="{self.bottom:.1f}"/>')
        return "".join(out)


def _hover_attr(plot: _Plot, ts: np.ndarray, labels: Sequence[str]) -> str:
    """Puntos para la lectura al pasar el ratón: [[x, texto], ...] (JSON escapado en un atributo)."""
    pts = [[round(plot.x(t), 1), lab] for t, lab in zip(ts.tolist(), labels)]
    return _esc(json.dumps(pts, ensure_ascii=False))


def _svg(h: float, body: str, label: str, hover: str = "") -> str:
    data = f' data-hover="{hover}" tabindex="0"' if hover else ""
    return (f'<svg class="chart" viewBox="0 0 {W:.0f} {h:.0f}" role="img" aria-label="{_esc(label)}"'
            f'{data} preserveAspectRatio="xMidYMid meet">{body}'
            f'<line class="xhair" x1="0" x2="0" y1="{MT:.0f}" y2="{h - MB:.0f}" visibility="hidden"/></svg>')


def _oos_shading(plot: _Plot, windows: list[tuple[float, float, int]]) -> str:
    out = []
    for t0, t1, fold in windows:
        a, b = plot.x(max(t0, plot.x0)), plot.x(min(t1, plot.x1))
        if b <= a:
            continue
        out.append(f'<rect class="oos" x="{a:.1f}" y="{plot.top:.1f}" width="{b - a:.1f}" '
                   f'height="{plot.bottom - plot.top:.1f}"/>')
        out.append(f'<line class="fold" x1="{a:.1f}" x2="{a:.1f}" y1="{plot.top:.1f}" y2="{plot.bottom:.1f}"/>')
        if fold:
            out.append(f'<text class="tick fold-label" x="{a + 4:.1f}" y="{plot.top + 12:.1f}">T{fold}</text>')
    return "".join(out)


def _trade_marks(plot: _Plot, trades: list[BTTrade]) -> str:
    out = []
    for t in trades:
        win = t.pnl > 0
        xe, ye = plot.x(t.entry_ts), plot.y(t.entry_px)
        xx, yx = plot.x(t.exit_ts), plot.y(t.exit_px)
        tri = (f"M{xe:.1f},{ye - 6:.1f}L{xe + 5.5:.1f},{ye + 4:.1f}L{xe - 5.5:.1f},{ye + 4:.1f}Z" if t.side == "long"
               else f"M{xe:.1f},{ye + 6:.1f}L{xe + 5.5:.1f},{ye - 4:.1f}L{xe - 5.5:.1f},{ye - 4:.1f}Z")
        tip = (f"{'Long' if t.side == 'long' else 'Short'} {_day(t.entry_ts, hours=True)} a {_fmt_px(t.entry_px)} → "
               f"{_day(t.exit_ts, hours=True)} a {_fmt_px(t.exit_px)} ({t.exit_reason}): "
               f"{t.pnl:+,.2f} USD, {t.r_multiple:+.2f} R")
        exit_mark = (f'<circle class="exit win" cx="{xx:.1f}" cy="{yx:.1f}" r="4.5"/>' if win else
                     f'<path class="exit loss" d="M{xx:.1f},{yx - 5.5:.1f}L{xx + 5.5:.1f},{yx:.1f}L{xx:.1f},'
                     f'{yx + 5.5:.1f}L{xx - 5.5:.1f},{yx:.1f}Z"/>')
        out.append(f'<g class="trade"><title>{_esc(tip)}</title>'
                   f'<line class="hold {"win" if win else "loss"}" x1="{xe:.1f}" y1="{ye:.1f}" x2="{xx:.1f}" '
                   f'y2="{yx:.1f}"/><path class="entry" d="{tri}"/>{exit_mark}'
                   f'<circle class="hit" cx="{xx:.1f}" cy="{yx:.1f}" r="11"/></g>')
    return "".join(out)


def _price_svg(px: dict[str, np.ndarray], trades: list[BTTrade], windows: list[tuple[float, float, int]],
               hours: bool) -> str:
    h = 320.0
    ts = px["ts"]
    lo = float(np.nanmin(px["low"]))
    hi = float(np.nanmax(px["high"]))
    tr_px = [p for t in trades for p in (t.entry_px, t.exit_px) if math.isfinite(p)]
    if tr_px:
        lo, hi = min(lo, min(tr_px)), max(hi, max(tr_px))
    x1 = float(ts[-1])
    if trades:
        x1 = max(x1, max(t.exit_ts for t in trades))
    plot = _Plot(h, float(ts[0]), x1, lo, hi)
    body = (plot.axes() + _oos_shading(plot, windows)
            + f'<path class="band" d="{plot.area(ts, px["high"], px["low"])}"/>'
            + f'<path class="price" d="{plot.path(ts, px["close"])}"/>' + _trade_marks(plot, trades))
    labels = [f"{_day(t, hours=hours)} · cierre {_fmt_px(v)}" for t, v in zip(ts.tolist(), px["close"].tolist())]
    return _svg(h, body, "Precio con entradas y salidas fuera de muestra", _hover_attr(plot, ts, labels))


def _equity_svg(ts: np.ndarray, eq: np.ndarray, start: Optional[float], windows, hours: bool) -> str:
    h = 220.0
    lo, hi = float(np.min(eq)), float(np.max(eq))
    if start is not None:
        lo, hi = min(lo, start), max(hi, start)
    plot = _Plot(h, float(ts[0]), float(ts[-1]), lo, hi)
    base = start if start is not None else float(eq[0])
    body = plot.axes() + _oos_shading(plot, [w for w in windows if w[2]])
    yb = plot.y(base)
    body += f'<line class="base" x1="{plot.left:.1f}" x2="{plot.right:.1f}" y1="{yb:.1f}" y2="{yb:.1f}"/>'
    body += f'<path class="eq-area" d="{plot.area(ts, eq, np.full(len(eq), base))}"/>'
    body += f'<path class="equity" d="{plot.path(ts, eq)}"/>'
    xe, ye = plot.x(float(ts[-1])), plot.y(float(eq[-1]))
    body += f'<circle class="end" cx="{xe:.1f}" cy="{ye:.1f}" r="4"/>'
    labels = [f"{_day(t, hours=hours)} · capital {v:,.2f} USD" for t, v in zip(ts.tolist(), eq.tolist())]
    return _svg(h, body, "Curva de capital fuera de muestra", _hover_attr(plot, ts, labels))


def _drawdown_svg(ts: np.ndarray, dd: np.ndarray, hours: bool) -> str:
    h = 160.0
    lo = float(min(np.min(dd), -0.1))
    plot = _Plot(h, float(ts[0]), float(ts[-1]), lo, 0.0, pad=0.04)
    body = plot.axes(yfmt=lambda v: f"{v:.3g} %")
    y0 = plot.y(0.0)
    body += f'<line class="base" x1="{plot.left:.1f}" x2="{plot.right:.1f}" y1="{y0:.1f}" y2="{y0:.1f}"/>'
    body += f'<path class="dd-area" d="{plot.area(ts, np.zeros(len(dd)), dd)}"/>'
    body += f'<path class="dd" d="{plot.path(ts, dd)}"/>'
    i = int(np.argmin(dd))
    if dd[i] < 0:
        xi, yi = plot.x(float(ts[i])), plot.y(float(dd[i]))
        anchor = "end" if xi > W * 0.75 else "start"
        dx = -8 if anchor == "end" else 8
        body += (f'<circle class="dd-min" cx="{xi:.1f}" cy="{yi:.1f}" r="4"/><text class="label" x="{xi + dx:.1f}" '
                 f'y="{min(yi + 4, plot.bottom - 4):.1f}" text-anchor="{anchor}">máx. {dd[i]:.2f} %</text>')
    labels = [f"{_day(t, hours=hours)} · drawdown {v:.2f} %" for t, v in zip(ts.tolist(), dd.tolist())]
    return _svg(h, body, "Drawdown fuera de muestra", _hover_attr(plot, ts, labels))


# --- HTML -----------------------------------------------------------------------------

def _tokens(p: dict[str, str]) -> str:
    return ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in p.items())


_CSS = """
:root{color-scheme:light;%(light)s}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;%(dark)s}}
:root[data-theme="dark"]{color-scheme:dark;%(dark)s}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.5rem;margin:0 0 4px;font-weight:650}
h2{font-size:1.05rem;margin:28px 0 8px;font-weight:600}
.sub,.note,figcaption{color:var(--ink2)}
.sub{margin:0}
.card{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:12px 12px 8px}
.warn{margin:16px 0;padding:10px 12px;border-left:4px solid var(--warn);background:var(--surface);border-radius:6px}
.warn b{margin-right:4px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:10px;margin:16px 0 4px}
.tile{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:10px 12px}
.tile .k{color:var(--ink2);font-size:.82rem}
.tile .v{font-size:1.35rem;font-weight:600}
.status{font-weight:600}
.status.ok{color:var(--good-text)}
.status.no{color:var(--ink2)}
figure{margin:0;position:relative}
svg.chart{width:100%%;height:auto;display:block;touch-action:pan-y}
svg.chart:focus{outline:2px solid var(--series);outline-offset:2px;border-radius:6px}
svg text{fill:var(--muted);font-size:12px;font-family:inherit}
svg text.label{fill:var(--ink2)}
.grid{stroke:var(--grid);stroke-width:1}
.axis,.base{stroke:var(--axis);stroke-width:1}
.oos{fill:var(--muted);opacity:.09}
.fold{stroke:var(--axis);stroke-width:1}
.band{fill:var(--ink2);opacity:.10}
.price{fill:none;stroke:var(--ink2);stroke-width:1.5;stroke-linejoin:round;stroke-linecap:round}
.equity{fill:none;stroke:var(--series);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.eq-area{fill:var(--series);opacity:.10}
.end{fill:var(--series);stroke:var(--surface);stroke-width:2}
.dd{fill:none;stroke:var(--bad);stroke-width:2;stroke-linejoin:round}
.dd-area{fill:var(--bad);opacity:.12}
.dd-min{fill:var(--bad);stroke:var(--surface);stroke-width:2}
.entry{fill:var(--series);stroke:var(--surface);stroke-width:1.5}
.exit.win{fill:var(--good);stroke:var(--surface);stroke-width:1.5}
.exit.loss{fill:var(--bad);stroke:var(--surface);stroke-width:1.5}
.hold{stroke-width:1;opacity:.55}
.hold.win{stroke:var(--good)}
.hold.loss{stroke:var(--bad)}
.hit{fill:transparent}
.trade:hover .entry,.trade:hover .exit{stroke:var(--ink)}
.xhair{stroke:var(--ink2);stroke-width:1;pointer-events:none}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:6px 2px 2px;color:var(--ink2);font-size:.85rem}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend svg{width:14px;height:14px}
.tip{position:absolute;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--grid);
 border-radius:6px;padding:4px 8px;font-size:.82rem;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%%;font-size:.88rem;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--ink2);font-weight:600}
th:first-child,td:first-child{text-align:left}
tr.has-why td{border-bottom:none}
tr.why td{text-align:left;white-space:normal;color:var(--ink2);font-size:.82rem;padding-top:0}
tr.win td:first-child{font-weight:600}
.grp{text-align:center;border-bottom:none}
.notes li{color:var(--ink2)}
footer{margin-top:28px;color:var(--ink2);font-size:.85rem}
@media (max-width:600px){svg text{font-size:22px}svg text.alt{display:none}.wrap{padding:16px 16px 40px}
 h1{font-size:1.25rem}.tile .v{font-size:1.15rem}}
@media print{.tip{display:none}}
"""

_JS = """
(function(){
  document.querySelectorAll('svg.chart[data-hover]').forEach(function(svg){
    var pts; try { pts = JSON.parse(svg.getAttribute('data-hover')); } catch (e) { return; }
    if (!pts.length) return;
    var fig = svg.parentNode, tip = document.createElement('div'), line = svg.querySelector('.xhair'), idx = -1;
    tip.className = 'tip'; fig.appendChild(tip);
    function show(i){
      idx = Math.max(0, Math.min(pts.length - 1, i));
      var x = pts[idx][0];
      line.setAttribute('x1', x); line.setAttribute('x2', x); line.setAttribute('visibility', 'visible');
      tip.textContent = pts[idx][1]; tip.style.display = 'block';
      var r = svg.getBoundingClientRect(), fx = x / svg.viewBox.baseVal.width * r.width;
      var left = fx + 12; if (left + tip.offsetWidth > r.width) left = fx - tip.offsetWidth - 12;
      tip.style.left = Math.max(0, left) + 'px'; tip.style.top = '8px';
    }
    function hide(){ line.setAttribute('visibility', 'hidden'); tip.style.display = 'none'; }
    function nearest(x){ var lo = 0, hi = pts.length - 1;
      while (hi - lo > 1) { var m = (lo + hi) >> 1; if (pts[m][0] < x) lo = m; else hi = m; }
      return Math.abs(pts[lo][0] - x) <= Math.abs(pts[hi][0] - x) ? lo : hi; }
    svg.addEventListener('pointermove', function(ev){
      var m = svg.getScreenCTM(); if (!m) return;
      var p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
      show(nearest(p.matrixTransform(m.inverse()).x));
    });
    svg.addEventListener('pointerleave', hide); svg.addEventListener('blur', hide);
    svg.addEventListener('focus', function(){ show(idx < 0 ? pts.length - 1 : idx); });
    svg.addEventListener('keydown', function(ev){
      var step = ev.shiftKey ? 10 : 1;
      if (ev.key === 'ArrowLeft') { show(idx - step); ev.preventDefault(); }
      else if (ev.key === 'ArrowRight') { show(idx + step); ev.preventDefault(); }
      else if (ev.key === 'Escape') hide();
    });
  });
})();
"""


def _legend(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(f"<span>{icon}{_esc(text)}</span>" for icon, text in items) + "</div>"


_ICON_ENTRY = '<svg viewBox="0 0 14 14" aria-hidden="true"><path class="entry" d="M7,1L13,12L1,12Z"/></svg>'
_ICON_WIN = '<svg viewBox="0 0 14 14" aria-hidden="true"><circle class="exit win" cx="7" cy="7" r="5"/></svg>'
_ICON_LOSS = '<svg viewBox="0 0 14 14" aria-hidden="true"><path class="exit loss" d="M7,1L13,7L7,13L1,7Z"/></svg>'
_ICON_OOS = ('<svg viewBox="0 0 14 14" aria-hidden="true"><rect x="1" y="1" width="12" height="12" rx="2" '
             'style="fill:var(--muted);opacity:.3"/></svg>')
_ICON_PRICE = ('<svg viewBox="0 0 14 14" aria-hidden="true"><line x1="1" y1="7" x2="13" y2="7" '
               'style="stroke:var(--ink2);stroke-width:2"/></svg>')


def _tile(label: str, value: str) -> str:
    return f'<div class="tile"><div class="k">{_esc(label)}</div><div class="v">{_esc(value)}</div></div>'


def _strategies_table(report: Any) -> str:
    rows = []
    for e in report.evaluations:
        i, o = e.is_metrics, e.oos_metrics
        win = report.winner == e.strategy
        if e.error:
            status = '<span class="status no">✘ no evaluada</span>'
        elif e.passed:
            status = '<span class="status ok">✔ aprobada</span>' + (" (ganadora)" if win else "")
        else:
            status = '<span class="status no">✘ rechazada</span>'
        cls = " ".join(c for c in ("win" if win else "", "has-why" if e.reasons else "") if c)
        rows.append(
            f'<tr class="{cls}"><td>{_esc(e.strategy)}</td>'
            f"<td>{_pct(i.get('total_return_pct'))}</td><td>{_pf(i.get('profit_factor'))}</td>"
            f"<td>{_pct(o.get('total_return_pct'))}</td><td>{_pf(o.get('profit_factor'))}</td>"
            f"<td>{_pct(o.get('max_drawdown_pct'), signed=False)}</td><td>{int(_fin(o.get('n_trades')) or 0)}</td>"
            f"<td>{_t(o.get('t_stat'))}</td><td>{_n(o.get('sharpe'))}</td><td>{status}</td></tr>")
        if e.reasons:
            rows.append(f'<tr class="why"><td colspan="10">{_esc("; ".join(e.reasons))}</td></tr>')
    head = ('<tr><th></th><th class="grp" colspan="2">Dentro de muestra</th><th class="grp" colspan="6">'
            'Fuera de muestra (lo que cuenta)</th><th></th></tr>'
            '<tr><th>Estrategia</th><th>Retorno</th><th>PF</th><th>Retorno</th><th>PF</th><th>DD máx.</th>'
            '<th>Ops</th><th>t</th><th>Sharpe</th><th>Estado</th></tr>')
    return f'<div class="card scroll"><table>{head}{"".join(rows)}</table></div>'


def _folds_table(ev: Any) -> str:
    from .optimizer import format_params
    rows = []
    for f in ev.folds:
        i, o = f.is_metrics, f.oos_metrics
        rows.append(f'<tr class="has-why"><td>T{f.fold}</td><td>{_esc(_day(f.train_start_ts))} – '
                    f"{_esc(_day(f.train_end_ts))}</td><td>{_esc(_day(f.test_start_ts))} – {_esc(_day(f.test_end_ts))}"
                    f"</td><td>{_pct(i.get('total_return_pct'))}</td><td>{_pct(o.get('total_return_pct'))}</td>"
                    f"<td>{_pf(o.get('profit_factor'))}</td><td>{int(_fin(o.get('n_trades')) or 0)}</td></tr>"
                    f'<tr class="why"><td colspan="7">Parámetros elegidos con el entrenamiento: '
                    f"{_esc(format_params(f.params))}</td></tr>")
    head = ("<tr><th>Tramo</th><th>Entrenamiento</th><th>Validación</th><th>Ret. dentro</th><th>Ret. fuera</th>"
            "<th>PF fuera</th><th>Ops fuera</th></tr>")
    return f'<div class="card scroll"><table>{head}{"".join(rows)}</table></div>'


def _trades_table(trades: list[BTTrade], hours: bool = True) -> str:
    shown = trades[-MAX_TRADE_ROWS:]
    rows = []
    for t in shown:
        rows.append(f"<tr><td>{_esc(_day(t.entry_ts, hours=hours))}</td><td>{'Long' if t.side == 'long' else 'Short'}</td>"
                    f"<td>{_esc(_fmt_px(t.entry_px))}</td><td>{_esc(_day(t.exit_ts, hours=hours))}</td>"
                    f"<td>{_esc(_fmt_px(t.exit_px))}</td><td>{_esc(t.exit_reason)}</td><td>{t.pnl:+,.2f}</td>"
                    f"<td>{t.r_multiple:+.2f}</td><td>{t.bars_held}</td></tr>")
    head = ("<tr><th>Entrada</th><th>Lado</th><th>Precio</th><th>Salida</th><th>Precio</th><th>Motivo</th>"
            "<th>PnL USD</th><th>R</th><th>Velas</th></tr>")
    more = (f'<p class="note">Últimas {len(shown)} de {len(trades)} operaciones.</p>' if len(trades) > len(shown)
            else "")
    return f'<div class="card scroll"><table>{head}{"".join(rows)}</table></div>{more}'


def render_html(report: Any, result: Optional[BacktestResult], bars: Optional[pd.DataFrame]) -> str:
    """HTML autocontenido del informe (ver el docstring del módulo). Nunca lanza por datos raros: una sección
    que no se puede pintar sale como una nota."""
    coin = getattr(report, "coin", None) or getattr(result, "coin", "") or ""
    interval = getattr(report, "interval", None) or getattr(result, "interval", "") or ""
    try:
        hours = interval_seconds(interval) < 86400 if interval else True
    except ValueError:
        hours = True
    ev = _evaluation(report, result)
    if result is None and ev is not None:
        result = getattr(ev, "oos_result", None)
    trades = list(result.trades) if result is not None else []
    windows = _oos_windows(report, result)
    title = f"Backtest {coin} {interval}".strip()
    parts = [f"<header><h1>{_esc(title)}</h1>"]
    sub = []
    if report is not None:
        sub.append(f"{report.days:.0f} días · {report.bars} velas")
        if report.start_ts and report.end_ts:
            sub.append(f"{_day(report.start_ts)} → {_day(report.end_ts)}")
        if report.folds:
            sub.append(f"walk-forward anclado de {report.folds} tramos")
        if report.id:
            sub.append(f"informe {report.id}")
        if report.created_ts:
            sub.append(f"creado {_day(report.created_ts, hours=True)} UTC")
    if sub:
        parts.append(f'<p class="sub">{_esc(" · ".join(sub))}</p>')
    parts.append("</header>")
    parts.append(f'<div class="warn" role="note"><b>⚠ Aviso.</b>{_esc(DISCLAIMER)}</div>')

    # tarjetas de la estrategia mostrada (fuera de muestra)
    if result is not None:
        m = result.metrics
        status = ""
        if ev is not None:
            status = "✔ aprobada" if ev.passed else "✘ rechazada"
        who = result.strategy + (" (ganadora)" if report is not None and report.winner == result.strategy else "")
        parts.append(f"<h2>{_esc(who)}: fuera de muestra</h2>")
        if status:
            cls = "ok" if ev.passed else "no"
            parts.append(f'<p class="sub"><span class="status {cls}">{_esc(status)}</span>'
                         + (f" · {_esc('; '.join(ev.reasons))}" if ev.reasons else "") + "</p>")
        parts.append('<div class="tiles">' + "".join([
            _tile("Retorno", _pct(m.get("total_return_pct"))), _tile("Profit factor", _pf(m.get("profit_factor"))),
            _tile("Drawdown máx.", _pct(m.get("max_drawdown_pct"), signed=False)),
            _tile("Operaciones", str(int(_fin(m.get("n_trades")) or 0))),
            _tile("Acierto", _pct(m.get("win_rate_pct"), 0, signed=False)), _tile("Sharpe", _n(m.get("sharpe"))),
        ]) + "</div>")

    # gráficos
    try:
        px = _price_series(bars)
        if px is not None:
            parts.append("<h2>Precio y operaciones</h2>")
            parts.append(f'<figure class="card">{_price_svg(px, trades, windows, hours)}'
                         + _legend([(_ICON_PRICE, "cierre"), (_ICON_ENTRY, "entrada (▲ long, ▼ short)"),
                                    (_ICON_WIN, "salida con ganancia"), (_ICON_LOSS, "salida con pérdida"),
                                    (_ICON_OOS, "fuera de muestra (T1…: tramos)")]) + "</figure>")
        ts, eq = _curve(result.equity if result is not None else None)
        if len(eq) >= 2:
            start = _start_equity(result, eq)
            parts.append("<h2>Capital fuera de muestra</h2>")
            parts.append(f'<figure class="card">{_equity_svg(ts, eq, start, windows, hours)}'
                         '<figcaption class="note">Tramos de validación encadenados: cada uno con los parámetros '
                         "elegidos sólo con su pasado.</figcaption></figure>")
            parts.append("<h2>Drawdown</h2>")
            parts.append(f'<figure class="card">{_drawdown_svg(ts, _drawdown(eq, start), hours)}</figure>')
    except Exception as e:  # noqa: BLE001 - un gráfico roto no impide el resto del informe
        log.exception("render_html: fallo pintando los gráficos")
        parts.append(f'<p class="note">No se pudieron pintar los gráficos: {_esc(e)}</p>')

    if report is not None and report.evaluations:
        parts.append("<h2>Estrategias: dentro vs fuera de muestra</h2>")
        parts.append(_strategies_table(report))
        crit = report.criteria or {}
        parts.append(f'<p class="note">Aprobada sólo si fuera de muestra: PF ≥ {_n(crit.get("min_profit_factor"))}, '
                     f'retorno &gt; 0, ≥ {int(_fin(crit.get("min_trades")) or 0)} operaciones, drawdown ≤ '
                     f'{_n(crit.get("max_drawdown_pct"), 0)} % y ventaja significativa (t del R medio corregida por '
                     f'{int(_fin(crit.get("compared")) or len(report.evaluations))} estrategias comparadas).</p>')
    if ev is not None and getattr(ev, "folds", None):
        parts.append(f"<h2>Tramos walk-forward: {_esc(ev.strategy)}</h2>")
        parts.append(_folds_table(ev))
    if trades:
        parts.append("<h2>Operaciones fuera de muestra</h2>")
        parts.append(_trades_table(trades, hours))
    notes = list(getattr(report, "notes", []) or [])
    if notes:
        parts.append('<h2>Notas</h2><ul class="notes">' + "".join(f"<li>{_esc(n)}</li>" for n in notes) + "</ul>")
    parts.append(f"<footer>{_esc(DISCLAIMER)}</footer>")
    css = _CSS % {"light": _tokens(LIGHT), "dark": _tokens(DARK)}
    return ("<!doctype html>\n<html lang=\"es\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{_esc(title or 'Backtest')}</title><style>{css}</style></head>"
            f"<body><main class=\"wrap\">{''.join(parts)}</main><script>{_JS}</script></body></html>")


# --- PNG ------------------------------------------------------------------------------

def _mpl_date_label(d: datetime) -> str:
    """Fecha en español para los ejes de matplotlib: ``8 mar`` o ``8 mar 14:00`` (``1 ene 2027`` al cambiar de año)."""
    out = f"{d.day} {_MONTHS[d.month - 1]}"
    if d.hour or d.minute:
        return out + f" {d:%H:%M}"
    return out + (f" {d.year}" if d.month == 1 and d.day == 1 else "")


def render_png(report: Any, result: Optional[BacktestResult], bars: Optional[pd.DataFrame], *,
               dpi: int = 110) -> Optional[bytes]:
    """PNG del informe (precio + operaciones, capital y drawdown fuera de muestra). None si matplotlib no está
    instalado o algo falla: el texto del informe basta y nunca debe tumbar a quien lo pide."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        import matplotlib.dates as mdates
    except Exception:  # noqa: BLE001 - matplotlib es opcional
        return None
    try:
        ev = _evaluation(report, result)
        if result is None and ev is not None:
            result = getattr(ev, "oos_result", None)
        trades = list(result.trades) if result is not None else []
        windows = _oos_windows(report, result)
        p = LIGHT
        fig = Figure(figsize=(10, 7.2), dpi=dpi, facecolor=p["surface"])
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 1.2], hspace=0.12, left=0.08, right=0.98, top=0.9,
                              bottom=0.1)
        axes = [fig.add_subplot(gs[i]) for i in range(3)]
        for ax in axes:
            ax.set_facecolor(p["surface"])
            ax.grid(True, color=p["grid"], linewidth=0.8)
            ax.tick_params(colors=p["muted"], labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(p["axis"])

        def dt(ts: Any) -> Any:
            return mdates.date2num(pd.to_datetime(np.asarray(ts, dtype=float), unit="s", utc=True).tz_localize(None))

        px = _price_series(bars)
        ax = axes[0]
        if px is not None:
            x = dt(px["ts"])
            ax.fill_between(x, px["low"], px["high"], color=p["ink2"], alpha=0.10, linewidth=0)
            ax.plot(x, px["close"], color=p["ink2"], linewidth=1.1)
        for t0, t1, _ in windows:
            for a in axes:
                a.axvspan(dt([t0])[0], dt([t1])[0], color=p["muted"], alpha=0.08, linewidth=0)
                a.axvline(dt([t0])[0], color=p["axis"], linewidth=0.8)
        if trades:
            for side, marker in (("long", "^"), ("short", "v")):
                sel = [t for t in trades if t.side == side]
                if sel:
                    ax.scatter(dt([t.entry_ts for t in sel]), [t.entry_px for t in sel], marker=marker, s=34,
                               color=p["series"], edgecolors=p["surface"], linewidths=0.8, zorder=3,
                               label=f"entrada {side}")
            for win, marker, color, label in ((True, "o", p["good"], "salida con ganancia"),
                                              (False, "D", p["bad"], "salida con pérdida")):
                sel = [t for t in trades if (t.pnl > 0) is win]
                if sel:
                    ax.scatter(dt([t.exit_ts for t in sel]), [t.exit_px for t in sel], marker=marker, s=26,
                               color=color, edgecolors=p["surface"], linewidths=0.8, zorder=3, label=label)
            ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=4, labelcolor=p["ink2"])
        ax.set_ylabel("precio", color=p["ink2"], fontsize=8)
        ts, eq = _curve(result.equity if result is not None else None)
        if len(eq) >= 2:
            x = dt(ts)
            axes[1].plot(x, eq, color=p["series"], linewidth=1.6)
            axes[1].fill_between(x, eq, float(eq[0]), color=p["series"], alpha=0.10, linewidth=0)
            axes[1].set_ylabel("capital USD", color=p["ink2"], fontsize=8)
            dd = _drawdown(eq, _start_equity(result, eq))
            axes[2].fill_between(x, dd, 0, color=p["bad"], alpha=0.15, linewidth=0)
            axes[2].plot(x, dd, color=p["bad"], linewidth=1.2)
            axes[2].set_ylabel("drawdown %", color=p["ink2"], fontsize=8)
        x_all = [a.get_xlim() for a in axes if a.has_data()]
        if x_all:
            lo, hi = min(v[0] for v in x_all), max(v[1] for v in x_all)
            for a in axes:
                a.set_xlim(lo, hi)
        from matplotlib.ticker import FuncFormatter
        loc = mdates.AutoDateLocator(maxticks=8)
        for a in axes:
            a.xaxis.set_major_locator(loc)
            a.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _mpl_date_label(mdates.num2date(v))))
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _compact(v)))
        axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _compact(v)))
        for a in axes[:-1]:
            a.tick_params(labelbottom=False)
        coin = getattr(report, "coin", None) or getattr(result, "coin", "")
        interval = getattr(report, "interval", None) or getattr(result, "interval", "")
        head = f"Backtest {coin} {interval}"
        if result is not None:
            m = result.metrics
            head += (f" · {result.strategy} fuera de muestra: {_pct(m.get('total_return_pct'))}, PF "
                     f"{_pf(m.get('profit_factor'))}, DD {_pct(m.get('max_drawdown_pct'), signed=False)}, "
                     f"{int(_fin(m.get('n_trades')) or 0)} ops")
            if ev is not None:
                head += " · " + ("APROBADA" if ev.passed else "rechazada")
        fig.suptitle(head, x=0.08, ha="left", fontsize=10, color=p["ink"])
        fig.text(0.08, 0.02, "Sombreado: fuera de muestra (walk-forward). Un backtest no garantiza resultados "
                 "futuros.", fontsize=7.5, color=p["ink2"])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=p["surface"])
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        log.exception("render_png: no se pudo generar la imagen")
        return None


__all__ = ["render_html", "render_png", "sparkline", "DISCLAIMER"]

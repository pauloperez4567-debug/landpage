"""Panel en vivo de la mesa (``python -m desk dashboard``): sólo lectura sobre los ficheros de estado.

No habla con la mesa ni con el exchange: lee lo que la mesa deja en ``state_dir`` y lo pinta con ``rich`` cada
segundo. Se puede abrir y cerrar cuando se quiera (también varios a la vez) sin afectar a la mesa.

Qué lee (todo opcional: lo que falta sale como "sin datos", nunca como error):

- ``status.json``: ``ts``, ``time``, ``line``, ``report`` y ``account_ok`` (lo que la mesa escribe hoy) y, si están,
  los campos ampliados de la fase 2 (``equity``, ``day_pnl``, ``positions``, ``orders``, ``risk``...; ver
  ``normalize_status``). Sin campos ampliados, cabecera, posiciones y órdenes salen del texto de ``line``/``report``.
- ``risk.json`` (halt, pausa, cooldown: se guarda en cuanto cambia) y ``health.json`` (ciclo y fallos).
- ``alerts.jsonl`` (últimos avisos) y ``alerts.json`` (reglas y silencio, si ``status.json`` no trae contadores).
- ``playbook.json`` (gráficas en automático) y ``reports/*.json`` (último informe de backtest).
- ``events.jsonl`` (señales y operaciones cerradas), ``outbox.log`` (avisos ``desk``) y ``equity.csv``.

Tolerancia: un JSON ilegible o a medio escribir se sustituye por la última lectura buena (y se avisa en la
cabecera); de los diarios sólo se leen las últimas líneas completas (una a medio escribir se ignora); los campos
ausentes o con tipos raros se muestran como "-". Cada panel se construye por separado: si uno falla, ese panel
muestra el error y los demás siguen. Las métricas de backtest que se muestran son siempre **fuera de muestra**
(walk-forward) y van con su advertencia: un backtest no garantiza resultados futuros.

``build_layout(snapshot)`` y las funciones ``build_*`` son puras (snapshot -> renderable de rich): se prueban
renderizando a texto con ``Console(record=True)``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .clock import Clock, RealClock
from .engine import fmt_px, fmt_size, fmt_usd
from .journal import EQUITY_COLUMNS, EQUITY_FILE, EVENTS_FILE, OUTBOX_FILE, OUTBOX_RE
from .notify import normalize_level

log = logging.getLogger(__name__)

STATUS_FILE = "status.json"
RISK_FILE = "risk.json"
HEALTH_FILE = "health.json"
ALERTS_LOG = "alerts.jsonl"           # avisos disparados (los escribe desk/alerts.py)
ALERTS_STATE = "alerts.json"          # reglas y silencio
PLAYBOOK_FILE = "playbook.json"
REPORTS_DIR = "reports"               # reports/<COIN>_<interval>_<id>.json

REFRESH_S = 1.0
TAIL_BYTES = 256 * 1024               # de cada diario se leen como mucho los últimos 256 KB
REPORTS_TRIED = 5                     # informes más recientes que se intentan leer (uno roto no deja el panel vacío)
STALE_S = 600.0                       # sin ciclos (health.json) en 10 min: la mesa parece detenida
MAX_ROWS = 12
MAX_ALERTS = 8
MAX_SIGNALS = 6
MAX_TRADES = 6
MAX_NOTICES = 5
MAX_EVALS = 5
NOTICE_CHARS = 160                    # un aviso largo de la mesa se corta (el completo está en outbox.log)
SPARK_POINTS = 32
TWO_COLUMNS_MIN_WIDTH = 120
HALT_DAILY = "daily_loss"             # halt diario: caduca solo al cambiar el día UTC (ver risk.active_halt)

LEVEL_STYLES = {"info": "cyan", "warn": "bold yellow", "critical": "bold white on red"}
LEVEL_LABELS = {"info": "INFO", "warn": "AVISO", "critical": "CRÍTICO"}
_SPARK = "▁▂▃▄▅▆▇█"


# --------------------------------------------------------------------------- utilidades
def _num(v: Any) -> Optional[float]:
    """Número finito o None (cadenas numéricas valen; booleanos, NaN y basura no)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _first(d: Any, *keys: str) -> Any:
    """Primer valor presente (no None ni "") de ``keys`` en ``d`` (alias de un mismo campo)."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            return v
    return None


def _pick(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _str(v: Any) -> str:
    return "" if v is None else str(v)


def _bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"on", "true", "1", "si", "sí", "yes"}:
            return True
        if s in {"off", "false", "0", "no", ""}:
            return False
    return None


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _when(ts: Any = None, time_text: Any = None) -> str:
    """``dd/mm HH:MM`` desde una marca de tiempo o desde ``YYYY-MM-DD HH:MM:SS``; "-" si no hay."""
    x = _num(ts)
    if x is not None:
        try:
            return datetime.fromtimestamp(x, tz=timezone.utc).strftime("%d/%m %H:%M")
        except (OverflowError, OSError, ValueError):
            pass
    s = _str(time_text)
    m = re.match(r"^\d{4}-(\d{2})-(\d{2})[ T](\d{2}:\d{2})", s)
    return f"{m.group(2)}/{m.group(1)} {m.group(3)}" if m else "-"


def fmt_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    s = max(0.0, float(seconds))
    if s < 90:
        return f"{s:.0f} s"
    if s < 90 * 60:
        return f"{s / 60:.0f} min"
    if s < 48 * 3600:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} d"


def sparkline(values: Iterable[float], width: int = SPARK_POINTS) -> str:
    """Mini-gráfica con bloques unicode de los últimos ``width`` valores (vacía con menos de 2)."""
    vals = [v for v in (_num(x) for x in values) if v is not None][-width:]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo <= 0:
        return _SPARK[len(_SPARK) // 2] * len(vals)
    top = len(_SPARK) - 1
    return "".join(_SPARK[int(round((v - lo) / (hi - lo) * top))] for v in vals)


def _equity_spark(values: list[float]) -> str:
    """Mini-gráfica de la equity: la de ``report_render`` si está (la misma que en Telegram); si no, la local."""
    try:
        from .report_render import sparkline as shared
        out = shared(values, width=SPARK_POINTS)
        if isinstance(out, str):
            return out
    except Exception:  # noqa: BLE001 - módulo ausente o datos raros: la local basta
        pass
    return sparkline(values)


def pnl_style(x: Optional[float]) -> str:
    if x is None or x == 0:
        return "dim"
    return "green" if x > 0 else "red"


def pnl_text(usd: Optional[float], pct: Optional[float] = None, *, bold: bool = False) -> Text:
    """``+$12.30 (+0.12%)`` en verde / rojo (gris si es cero o no se sabe)."""
    ref = usd if usd is not None else pct
    style = pnl_style(ref) + (" bold" if bold and ref else "")
    body = fmt_usd(usd) if usd is not None else "-"
    if pct is not None:
        body += f" ({pct:+.2f}%)"
    return Text(body, style=style)


def usage_style(used_pct: Optional[float]) -> str:
    if used_pct is None:
        return "dim"
    if used_pct >= 80:
        return "bold red"
    if used_pct >= 50:
        return "yellow"
    return "green"


def usage_bar(used_pct: Optional[float], width: int = 20) -> Text:
    """Barra de consumo de un límite de riesgo: ``███████░░░░ 35%`` (verde < 50 %, amarillo < 80 %, rojo)."""
    if used_pct is None:
        return Text("░" * width + "   s/d", style="dim")
    frac = min(max(used_pct, 0.0), 100.0) / 100.0
    full = int(round(frac * width))
    style = usage_style(used_pct)
    t = Text("█" * full, style=style)
    t.append("░" * (width - full), style="dim")
    t.append(f" {used_pct:3.0f}%", style=style)
    return t


def level_badge(level: Any) -> Text:
    lv = normalize_level(level)
    return Text(f" {LEVEL_LABELS[lv]} ", style=LEVEL_STYLES[lv])


def side_text(side: Any) -> Text:
    s = _str(side).lower()
    if s in ("long", "buy", "compra"):
        return Text("LONG" if s == "long" else "compra", style="green")
    if s in ("short", "sell", "venta"):
        return Text("SHORT" if s == "short" else "venta", style="red")
    return Text(s or "-", style="dim")


def _origin(origin: Any) -> str:
    o = _str(origin)
    if o.startswith("auto:"):
        return o[5:] or "auto"                     # el nombre de la estrategia ya dice que es automática
    return {"command": "comando", "adopted": "adoptada", "": "-"}.get(o, o)


def _params_text(params: Any) -> str:
    if not isinstance(params, dict) or not params:
        return ""
    return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())


# --------------------------------------------------------------------------- lectura de ficheros
def read_json(path: Path) -> tuple[Any, Optional[str]]:
    """(contenido, error). Un fichero ausente devuelve (None, None): no es un error."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, None
    except OSError as e:
        return None, f"no se pudo leer ({e.strerror or e})"
    try:
        return json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, ValueError):
        return None, "ilegible o a medio escribir"


def tail_lines(path: Path, max_bytes: int = TAIL_BYTES) -> list[str]:
    """Últimas líneas completas del fichero (como mucho ``max_bytes``). La última, si no acaba en salto de línea
    (a medio escribir), se ignora; la primera del trozo leído también (puede estar cortada)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_bytes)
            f.seek(start)
            data = f.read(size - start)
    except OSError:
        return []
    end = data.rfind(b"\n")
    if end < 0:
        return []
    data = data[:end]
    if start > 0:
        nl = data.find(b"\n")
        if nl < 0:
            return []
        data = data[nl + 1:]
    return data.decode("utf-8", "replace").splitlines()


def parse_jsonl(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Objetos JSON de un diario; las líneas rotas se saltan."""
    out: list[dict[str, Any]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


# --------------------------------------------------------------------------- normalización
_MONEY = r"([\d,]+(?:\.\d+)?)"
_RX_EQUITY = re.compile(r"equity (-?)\$" + _MONEY)
_RX_DAY = re.compile(r"PnL día ([+-])\$" + _MONEY + r" \(([+-]?\d+(?:\.\d+)?)%\)")
_RX_AUTO = re.compile(r"\bauto (ON|OFF)(?: \(([^)]*)\))?")
_RX_HALT = re.compile(r"\bhalt (\S+)\s*$")
_RX_MODE = re.compile(r"\(modo (\w+), datos ([^)]+)\)")
_RX_AVAIL = re.compile(r"disponible (-?)\$" + _MONEY)
_RX_LIMITS = re.compile(r"drawdown (-?\d+(?:\.\d+)?)% \(límites: día (\d+(?:\.\d+)?)%, máx\. (\d+(?:\.\d+)?)%\)")
_RX_SYMBOLS = re.compile(r"^Automático \w+ \(.*?\) sobre (.+?) \|", re.M)
_RX_POS = re.compile(r"^\s+(\S+) (LONG|SHORT) (\S+) @ (\S+) → (\S+) \| PnL ([+-])\$" + _MONEY
                     + r" \(([+-]?\d+(?:\.\d+)?)% s/margen\) \| (\d+)x \| SL (\S+) TP (\S+) \| (.*)$")
_RX_PEND = re.compile(r"^\s+(\S+) (LONG|SHORT) (\S+) @ (\S+) \(SL (\S+) TP (\S+)(?:, caduca ([^)]*))?\) \| (.*)$")
_RX_ORDER = re.compile(r"^\s+(\S+) (compra|venta) (\S+) (\S+) @ (\S+)( reduce-only)?(?: \[(.*)\])?$")


def _money(sign: str, digits: str) -> Optional[float]:
    x = _num(digits.replace(",", ""))
    return None if x is None else (-x if sign == "-" else x)


def _parse_status_text(line: str, report: str) -> dict[str, Any]:
    """Lo que la mesa escribe hoy como texto (``status_line`` / ``status_report``) en campos. Mejor esfuerzo:
    lo que no casa se queda sin rellenar."""
    out: dict[str, Any] = {}
    text = f"{line}\n{report}"
    if m := _RX_EQUITY.search(text):
        out["equity"] = _money(m.group(1), m.group(2))
    if m := _RX_DAY.search(text):
        out["day_pnl"] = _money(m.group(1), m.group(2))
        out["day_pnl_pct"] = _num(m.group(3))
    if m := _RX_AUTO.search(line):
        out["auto"] = m.group(1) == "ON"
        note = (m.group(2) or "").lower()
        out["paused"] = "pausa" in note
        out["cooldown"] = "cooldown" in note
    if m := _RX_HALT.search(line):
        out["halt"] = "" if m.group(1) == "no" else m.group(1)
    if m := _RX_MODE.search(report):
        out["mode"], out["data"] = m.group(1), m.group(2)
    if m := _RX_AVAIL.search(report):
        out["available"] = _money(m.group(1), m.group(2))
    if m := _RX_LIMITS.search(report):
        out["drawdown_pct"] = _num(m.group(1))
        out["daily_loss_limit_pct"] = _num(m.group(2))
        out["max_drawdown_pct"] = _num(m.group(3))
    if m := _RX_SYMBOLS.search(report):
        out["symbols"] = [s.strip() for s in m.group(1).split(",") if s.strip()]
    positions, pending, orders = [], [], []
    section = ""
    for raw in report.splitlines():
        if not raw.startswith(" "):
            section = ("pos" if raw.startswith("Posiciones (") else "pend" if raw.startswith("Entradas pendientes")
                       else "ord" if raw.startswith("Órdenes abiertas") else "")
            continue
        if section == "pos" and (m := _RX_POS.match(raw)):
            positions.append({"coin": m.group(1), "side": m.group(2).lower(), "size": m.group(3),
                              "entry_px": m.group(4), "mark_px": m.group(5),
                              "unrealized_pnl": _money(m.group(6), m.group(7)), "pnl_pct": m.group(8),
                              "leverage": m.group(9), "sl": m.group(10), "tp": m.group(11), "origin": m.group(12)})
        elif section == "pend" and (m := _RX_PEND.match(raw)):
            pending.append({"coin": m.group(1), "side": m.group(2).lower(), "size": m.group(3),
                            "entry_px": m.group(4), "sl": m.group(5), "tp": m.group(6),
                            "expires": m.group(7) or "", "origin": m.group(8)})
        elif section == "ord" and (m := _RX_ORDER.match(raw)):
            orders.append({"coin": m.group(1), "side": "buy" if m.group(2) == "compra" else "sell",
                           "kind": m.group(3), "size": m.group(4), "price": m.group(5),
                           "reduce_only": bool(m.group(6)), "tag": m.group(7) or ""})
    out["positions"], out["pending"], out["orders"] = positions, pending, orders
    return out


def _norm_position(d: Any, key: Any = None) -> Optional[dict[str, Any]]:
    if not isinstance(d, dict):
        return None
    coin = _str(_first(d, "coin") or key).strip()
    if not coin:
        return None
    size = _num(_first(d, "size", "szi"))
    side = _str(d.get("side")).lower()
    if side not in ("long", "short"):
        side = "long" if (size or 0) > 0 else "short" if (size or 0) < 0 else ""
    return {"coin": coin, "side": side, "size": None if size is None else abs(size),
            "entry_px": _num(_first(d, "entry_px", "entry")), "mark_px": _num(_first(d, "mark_px", "mark")),
            "pnl": _num(_first(d, "unrealized_pnl", "pnl", "upnl")), "pnl_pct": _num(_first(d, "pnl_pct", "roe_pct")),
            "leverage": _num(_first(d, "leverage", "lev")), "liq_px": _num(_first(d, "liq_px")),
            "sl": _num(_first(d, "sl", "sl_px")), "tp": _num(_first(d, "tp", "tp_px")),
            "origin": _str(_first(d, "origin"))}


def _norm_order(d: Any, key: Any = None) -> Optional[dict[str, Any]]:
    if not isinstance(d, dict) or not _first(d, "coin"):
        return None
    is_buy = d.get("is_buy")
    if not isinstance(is_buy, bool):
        is_buy = {"buy": True, "long": True, "compra": True, "sell": False, "short": False,
                  "venta": False}.get(_str(d.get("side")).lower())
    return {"coin": _str(d["coin"]), "side": "" if is_buy is None else ("buy" if is_buy else "sell"),
            "kind": _str(_first(d, "kind", "type")).lower(), "size": _num(_first(d, "remaining", "size")),
            "px": _num(_first(d, "trigger_px", "price", "px")), "reduce_only": bool(_bool(d.get("reduce_only"))),
            "tag": _str(d.get("tag"))}


def _norm_pending(d: Any, key: Any = None) -> Optional[dict[str, Any]]:
    if not isinstance(d, dict) or not _first(d, "coin"):
        return None
    exp = _num(_first(d, "expires_ts"))
    return {"coin": _str(d["coin"]), "side": _str(d.get("side")).lower(),
            "size": _num(_first(d, "planned_size", "size")), "entry_px": _num(_first(d, "entry_px", "price")),
            "sl": _num(d.get("sl")), "tp": _num(d.get("tp")),
            "expires": _when(exp) if exp else _str(d.get("expires")), "origin": _str(d.get("origin"))}


def _norm_list(raw: Any, fn: Callable[..., Optional[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Lista normalizada desde una lista o un dict ``{clave: item}`` (la clave sirve de moneda si falta)."""
    if isinstance(raw, dict):
        items = [(v, k) for k, v in raw.items()]
    elif isinstance(raw, list):
        items = [(v, None) for v in raw]
    else:
        return []
    out = []
    for v, k in items:
        try:
            n = fn(v, k)
        except Exception:  # noqa: BLE001 - un elemento raro no estropea los demás
            n = None
        if n is not None:
            out.append(n)
    return out


def _norm_metrics(d: Any) -> dict[str, Optional[float]]:
    d = _dict(d)
    return {"ret": _num(_first(d, "total_return_pct", "return_pct")), "pf": _num(d.get("profit_factor")),
            "dd": _num(d.get("max_drawdown_pct")), "trades": _num(_first(d, "n_trades", "trades")),
            "win": _num(d.get("win_rate_pct")), "sharpe": _num(d.get("sharpe"))}


def _norm_playbook_entry(d: Any, key: Any = None) -> Optional[dict[str, Any]]:
    if not isinstance(d, dict) or not _first(d, "coin"):
        return None
    enabled = _bool(d.get("enabled"))
    return {"coin": _str(d["coin"]), "interval": _str(d.get("interval")), "strategy": _str(d.get("strategy")),
            "params": d.get("params") if isinstance(d.get("params"), dict) else {},
            "units": _num(d.get("units")), "enabled": True if enabled is None else enabled,
            "oos": _norm_metrics(d.get("oos_metrics")), "adopted_ts": _num(d.get("adopted_ts")),
            "report_id": _str(d.get("report_id"))}


def normalize_playbook(raw: Any) -> list[dict[str, Any]]:
    """Entradas del playbook desde ``playbook.json`` (lista, ``{"entries": [...]}`` o ``{clave: entrada}``)."""
    items: Any = raw
    if isinstance(raw, dict):
        if "entries" in raw:
            items = raw["entries"]
        elif "playbook" in raw:
            items = raw["playbook"]
        else:
            items = [v for v in raw.values() if isinstance(v, dict)]
    if isinstance(items, dict):
        items = list(items.values())
    return _norm_list(items, _norm_playbook_entry) if isinstance(items, list) else []


def _norm_signal(d: Any, key: Any = None) -> Optional[dict[str, Any]]:
    if not isinstance(d, dict) or not _first(d, "coin"):
        return None
    return {"ts": _num(d.get("ts")), "time": _str(d.get("time")), "coin": _str(d["coin"]),
            "side": _str(d.get("side")).lower(), "strategy": _str(d.get("strategy")), "reason": _str(d.get("reason"))}


def normalize_status(raw: Any) -> dict[str, Any]:
    """``status.json`` en un dict canónico. Acepta el formato de hoy (``line``/``report`` en texto) y los campos
    ampliados de la fase 2, con sus alias; los campos estructurados mandan sobre lo que se saca del texto.

    Campos ampliados reconocidos: ``mode``, ``data``, ``network``, ``equity``, ``available``, ``day_pnl``,
    ``day_pnl_pct``, ``drawdown_pct``, ``auto``, ``paused``, ``cooldown``, ``halt``, ``symbols``,
    ``risk`` {``daily_loss_pct``, ``daily_loss_limit_pct``, ``daily_loss_used_pct``, ``drawdown_pct``,
    ``max_drawdown_pct``, ``drawdown_used_pct``}, ``positions``, ``orders``, ``pending``, ``playbook``,
    ``alerts`` (``AlertManager.stats()``) y ``signals``."""
    st: dict[str, Any] = {
        "present": False, "ts": None, "time": "", "line": "", "report": "", "account_ok": True,
        "mode": "", "data": "", "network": "", "equity": None, "available": None, "day_pnl": None,
        "day_pnl_pct": None, "drawdown_pct": None, "daily_loss_limit_pct": None, "max_drawdown_pct": None,
        "daily_used_pct": None, "dd_used_pct": None, "auto": None, "paused": None, "cooldown": None, "halt": None,
        "symbols": [], "positions": [], "orders": [], "pending": [], "playbook": None, "alerts": None,
        "signals": None,
    }
    if not isinstance(raw, dict):
        return st
    st["present"] = True
    st["ts"], st["time"] = _num(raw.get("ts")), _str(raw.get("time"))
    st["line"], st["report"] = _str(raw.get("line")), _str(raw.get("report"))
    st["account_ok"] = raw.get("account_ok") is not False
    txt = _parse_status_text(st["line"], st["report"])
    risk = _dict(_first(raw, "risk", "risk_usage", "limits"))
    st["mode"] = _str(_first(raw, "mode")) or txt.get("mode", "")
    st["data"] = _str(_first(raw, "data", "data_source")) or txt.get("data", "")
    st["network"] = _str(_first(raw, "network"))
    st["equity"] = _pick(_num(raw.get("equity")), txt.get("equity"))
    st["available"] = _pick(_num(raw.get("available")), txt.get("available"))
    st["day_pnl"] = _pick(_num(_first(raw, "day_pnl", "pnl_day")), _num(risk.get("day_pnl")), txt.get("day_pnl"))
    st["day_pnl_pct"] = _pick(_num(raw.get("day_pnl_pct")), _num(risk.get("day_pnl_pct")), txt.get("day_pnl_pct"))
    loss = _num(_first(risk, "daily_loss_pct", "day_loss_pct"))
    if st["day_pnl_pct"] is None and loss is not None:
        st["day_pnl_pct"] = -loss
    st["drawdown_pct"] = _pick(_num(risk.get("drawdown_pct")), _num(raw.get("drawdown_pct")), txt.get("drawdown_pct"))
    st["daily_loss_limit_pct"] = _pick(_num(_first(risk, "daily_loss_limit_pct", "daily_limit_pct")),
                                       _num(raw.get("daily_loss_limit_pct")), txt.get("daily_loss_limit_pct"))
    st["max_drawdown_pct"] = _pick(_num(_first(risk, "max_drawdown_pct", "drawdown_limit_pct")),
                                   _num(raw.get("max_drawdown_pct")), txt.get("max_drawdown_pct"))
    st["daily_used_pct"] = _num(_first(risk, "daily_loss_used_pct", "daily_used_pct"))
    st["dd_used_pct"] = _num(_first(risk, "drawdown_used_pct", "dd_used_pct"))
    complete_usage(st)
    st["auto"] = _pick(_bool(_first(raw, "auto", "auto_enabled")), txt.get("auto"))
    st["paused"] = _pick(_bool(raw.get("paused")), txt.get("paused"))
    st["cooldown"] = _pick(_bool(raw.get("cooldown")), txt.get("cooldown"))
    halt = raw.get("halt")
    if halt is None and "halted" in raw:
        halt = _str(raw.get("halt_reason")) or "halt" if _bool(raw.get("halted")) else ""
    if halt is False or (isinstance(halt, str) and halt.strip().lower() in ("", "no", "none")):
        halt = ""
    st["halt"] = _str(halt) if halt is not None else txt.get("halt")
    sym = raw.get("symbols")
    st["symbols"] = [str(s) for s in sym if s] if isinstance(sym, list) else txt.get("symbols", [])
    pos = raw.get("positions")
    st["positions"] = _norm_list(pos, _norm_position) if isinstance(pos, (list, dict)) else \
        _norm_list(txt["positions"], _norm_position)
    orders = _first(raw, "orders", "open_orders")
    st["orders"] = _norm_list(orders, _norm_order) if isinstance(orders, list) else _norm_list(txt["orders"], _norm_order)
    pend = _first(raw, "pending", "pending_entries")
    st["pending"] = _norm_list(pend, _norm_pending) if isinstance(pend, list) else \
        _norm_list(txt["pending"], _norm_pending)
    if "playbook" in raw:
        st["playbook"] = normalize_playbook(raw.get("playbook"))
    if isinstance(raw.get("alerts"), dict):
        st["alerts"] = raw["alerts"]
    sig = _first(raw, "signals", "last_signals", "recent_signals")
    if isinstance(sig, list):
        st["signals"] = _norm_list(sig, _norm_signal)
    return st


def complete_usage(st: dict[str, Any]) -> None:
    """Consumo (% del límite) de la pérdida diaria y del drawdown si no viene calculado y se conocen los datos."""
    day, lim = st.get("day_pnl_pct"), st.get("daily_loss_limit_pct")
    if st.get("daily_used_pct") is None and day is not None and lim:
        st["daily_used_pct"] = max(0.0, -day) / lim * 100.0
    dd, mx = st.get("drawdown_pct"), st.get("max_drawdown_pct")
    if st.get("dd_used_pct") is None and dd is not None and mx:
        st["dd_used_pct"] = max(0.0, dd) / mx * 100.0


def normalize_report(data: Any, path: Optional[Path] = None, mtime: Optional[float] = None) -> Optional[dict[str, Any]]:
    """Informe de backtest (``OptimizationReport.to_dict()``, solo o dentro de ``{"report": ..., "result": ...}``)."""
    if not isinstance(data, dict):
        return None
    rep = data["report"] if isinstance(data.get("report"), dict) else data
    evals = []
    for e in rep.get("evaluations") or []:
        if isinstance(e, dict) and e.get("strategy"):
            evals.append({"strategy": _str(e["strategy"]), "params": e.get("params") if isinstance(e.get("params"), dict)
                          else {}, "oos": _norm_metrics(e.get("oos_metrics")), "passed": bool(_bool(e.get("passed"))),
                          "score": _num(e.get("score")), "reasons": [str(r) for r in e.get("reasons") or []
                                                                    if r] if isinstance(e.get("reasons"), list) else []})
    winner = rep.get("winner")
    win_name = _str(winner.get("strategy")) if isinstance(winner, dict) else _str(winner)
    win = next((e for e in evals if e["strategy"] == win_name), None)
    if win is None and isinstance(winner, dict):
        win = {"strategy": win_name, "params": _dict(winner.get("params")), "oos": _norm_metrics(winner.get("oos_metrics")),
               "passed": True, "score": _num(winner.get("score")), "reasons": []}
    folds = _num(rep.get("folds"))
    first = next((e for e in rep.get("evaluations") or [] if isinstance(e, dict)), None)
    if folds is None and first is not None:
        f0 = first.get("folds")
        folds = float(len(f0)) if isinstance(f0, list) else _num(f0)
    if not (rep.get("coin") or evals):
        return None
    return {"id": _str(rep.get("id")), "coin": _str(rep.get("coin")), "interval": _str(rep.get("interval")),
            "days": _num(rep.get("days")), "bars": _num(rep.get("bars")),
            "created_ts": _pick(_num(rep.get("created_ts")), mtime), "evaluations": evals, "winner": win,
            "folds": folds, "notes": rep.get("notes"), "path": str(path) if path else ""}


# --------------------------------------------------------------------------- foto del estado
@dataclass
class Snapshot:
    """Todo lo que pinta el panel, ya leído y normalizado (``build_layout`` no toca disco)."""

    state_dir: Path
    now: float                                    # hora (del reloj inyectado) de la lectura
    status: dict[str, Any] = field(default_factory=lambda: normalize_status(None))
    status_age_s: Optional[float] = None          # antigüedad de status.json (mtime)
    health: dict[str, Any] = field(default_factory=dict)
    health_age_s: Optional[float] = None
    risk: dict[str, Any] = field(default_factory=dict)
    alert_stats: dict[str, Any] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    playbook: list[dict[str, Any]] = field(default_factory=list)
    report: Optional[dict[str, Any]] = None
    reports_count: int = 0
    signals: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)
    notices: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


class SnapshotReader:
    """Lee ``state_dir`` y devuelve un ``Snapshot``. Recuerda la última lectura buena de cada JSON (un fichero a
    medio escribir no deja el panel en blanco) y no vuelve a leer lo que no ha cambiado (informes, diarios)."""

    def __init__(self, state_dir: Union[str, Path], clock: Optional[Clock] = None, *, tail_bytes: int = TAIL_BYTES):
        self.dir = Path(state_dir)
        self.clock: Clock = clock if clock is not None else RealClock()
        self.tail_bytes = tail_bytes
        self._good: dict[str, Any] = {}
        self._tails: dict[str, tuple[tuple[int, int], list[str]]] = {}
        self._reports: dict[str, tuple[tuple[int, int], Optional[dict[str, Any]]]] = {}

    # ------------------------------------------------------------------ ayudas
    @staticmethod
    def _sig(path: Path) -> Optional[tuple[int, int]]:
        try:
            s = path.stat()
        except OSError:
            return None
        return (s.st_size, s.st_mtime_ns)

    def _age(self, path: Path, now: float) -> Optional[float]:
        try:
            return max(0.0, now - path.stat().st_mtime)
        except OSError:
            return None

    def _json(self, name: str, problems: list[str]) -> Any:
        data, err = read_json(self.dir / name)
        if err:
            if name in self._good:
                problems.append(f"{name} {err}: se muestra la lectura anterior")
                return self._good[name]
            problems.append(f"{name} {err}")
            return None
        if data is None:
            self._good.pop(name, None)
        else:
            self._good[name] = data
        return data

    def _tail(self, name: str) -> list[str]:
        path = self.dir / name
        sig = self._sig(path)
        if sig is None:
            self._tails.pop(name, None)
            return []
        cached = self._tails.get(name)
        if cached and cached[0] == sig:
            return cached[1]
        lines = tail_lines(path, self.tail_bytes)
        self._tails[name] = (sig, lines)
        return lines

    # ------------------------------------------------------------------ lectura
    def read(self) -> Snapshot:
        snap = Snapshot(self.dir, float(self.clock.now()))
        for what, fn in (("estado", self._read_status), ("riesgo", self._read_risk), ("alarmas", self._read_alerts),
                         ("playbook", self._read_playbook), ("informes", self._read_report),
                         ("eventos", self._read_events), ("avisos", self._read_notices),
                         ("equity", self._read_equity)):
            try:
                fn(snap)
            except Exception as e:  # noqa: BLE001 - leer un fichero raro nunca tumba el panel
                log.debug("panel: fallo leyendo %s", what, exc_info=True)
                snap.problems.append(f"{what}: {type(e).__name__}: {e}")
        return snap

    def _read_status(self, snap: Snapshot) -> None:
        snap.status = normalize_status(self._json(STATUS_FILE, snap.problems))
        snap.status_age_s = self._age(self.dir / STATUS_FILE, snap.now)
        snap.health = _dict(self._json(HEALTH_FILE, snap.problems))
        snap.health_age_s = self._age(self.dir / HEALTH_FILE, snap.now)

    def _read_risk(self, snap: Snapshot) -> None:
        """``risk.json`` se guarda en cuanto cambia: manda sobre ``status.json`` (que puede tener minutos) en halt,
        pausa y cooldown. Si ``status.json`` no trae los números de riesgo se calculan con él."""
        risk = _dict(self._json(RISK_FILE, snap.problems))
        snap.risk = risk
        if not risk:
            return
        st = snap.status
        desk_now = _pick(_num(snap.health.get("ts")), st.get("ts"), snap.now)
        halt = _str(risk.get("halt_reason")) or "halt" if _bool(risk.get("halted")) else ""
        if halt == HALT_DAILY and risk.get("day"):
            try:
                today = datetime.fromtimestamp(desk_now, tz=timezone.utc).strftime("%Y-%m-%d")
            except (OverflowError, OSError, ValueError):
                today = risk.get("day")
            if today != risk.get("day"):
                halt = ""                      # halt diario de otro día: ya no cuenta
        st["halt"] = halt
        if _bool(risk.get("paused")) is not None:
            st["paused"] = bool(risk.get("paused"))
        until = _num(risk.get("cooldown_until"))
        if until is not None:
            st["cooldown"] = until > desk_now
        eq = st.get("equity")
        start, hwm = _num(risk.get("day_start_equity")), _num(risk.get("high_water"))
        if eq is not None and start and start > 0 and st.get("day_pnl_pct") is None:
            st["day_pnl"] = eq - start
            st["day_pnl_pct"] = (eq - start) / start * 100.0
        if eq is not None and hwm and hwm > 0 and st.get("drawdown_pct") is None:
            st["drawdown_pct"] = max(hwm - eq, 0.0) / hwm * 100.0
        complete_usage(st)

    def _read_alerts(self, snap: Snapshot) -> None:
        recs = parse_jsonl(self._tail(ALERTS_LOG))[-MAX_ALERTS:]
        snap.alerts = [{"ts": _num(r.get("ts")), "time": _str(r.get("time")), "level": normalize_level(r.get("level")),
                        "kind": _str(r.get("kind")), "coin": _str(r.get("coin")), "text": _str(r.get("text")),
                        "muted": bool(r.get("muted"))} for r in recs]
        stats = snap.status.get("alerts")
        if isinstance(stats, dict):
            snap.alert_stats = dict(stats)
            return
        state = _dict(self._json(ALERTS_STATE, snap.problems))
        if state:
            rules = state.get("rules") if isinstance(state.get("rules"), list) else []
            mute = state.get("mute")
            until = _num(mute.get("until")) if isinstance(mute, dict) else None
            muted = isinstance(mute, dict) and (mute.get("until") is None or (until or 0) > snap.now)
            snap.alert_stats = {"rules": len(rules), "muted": muted, "muted_until": until,
                                "silenced": _num(mute.get("silenced")) if isinstance(mute, dict) else None}

    def _read_playbook(self, snap: Snapshot) -> None:
        raw = self._json(PLAYBOOK_FILE, snap.problems)
        if raw is not None:
            snap.playbook = normalize_playbook(raw)
        elif snap.status.get("playbook") is not None:
            snap.playbook = list(snap.status["playbook"])

    def _read_report(self, snap: Snapshot) -> None:
        d = self.dir / REPORTS_DIR
        try:
            files = [p for p in d.glob("*.json") if p.is_file()]
        except OSError:
            files = []
        snap.reports_count = len(files)
        dated: list[tuple[float, Path]] = []
        for p in files:
            try:
                dated.append((p.stat().st_mtime, p))
            except OSError:
                continue
        for mtime, p in sorted(dated, key=lambda x: x[0], reverse=True)[:REPORTS_TRIED]:
            key, sig = str(p), self._sig(p)
            cached = self._reports.get(key)
            if cached and cached[0] == sig:
                rep = cached[1]
            else:
                data, err = read_json(p)
                rep = normalize_report(data, p, mtime) if not err else None
                if sig is not None:
                    self._reports[key] = (sig, rep)
            if rep is not None:
                snap.report = rep
                return

    def _read_events(self, snap: Snapshot) -> None:
        events = parse_jsonl(self._tail(EVENTS_FILE))
        sigs = snap.status.get("signals")
        if isinstance(sigs, list) and sigs:
            snap.signals = sigs[-MAX_SIGNALS:]
        else:
            snap.signals = [s for s in (_norm_signal(e) for e in events if e.get("kind") == "signal")
                            if s is not None][-MAX_SIGNALS:]
        trades = []
        for e in events:
            if e.get("kind") != "trade_closed" or not e.get("coin"):
                continue
            trades.append({"ts": _num(_first(e, "closed_ts", "ts")), "coin": _str(e["coin"]),
                           "side": _str(e.get("side")).lower(), "pnl": _num(e.get("pnl")),
                           "exit_reason": _str(e.get("exit_reason")), "origin": _str(e.get("origin"))})
        snap.trades = trades[-MAX_TRADES:]

    def _read_notices(self, snap: Snapshot) -> None:
        snap.notices = desk_notices(self._tail(OUTBOX_FILE))[-MAX_NOTICES:]

    def _read_equity(self, snap: Snapshot) -> None:
        col = EQUITY_COLUMNS.index("equity")
        vals = []
        for ln in self._tail(EQUITY_FILE)[-SPARK_POINTS:]:
            parts = ln.split(",")
            if len(parts) > col and (x := _num(parts[col])) is not None:
                vals.append(x)
        snap.equity_curve = vals


def desk_notices(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Avisos de la mesa (id ``desk``) del outbox. Los renglones seguidos con la misma hora son un solo aviso."""
    out: list[dict[str, Any]] = []
    prev_stamp: Optional[str] = None
    for raw in lines:
        m = OUTBOX_RE.match(raw)
        if not m or m.group(2) != "desk":
            prev_stamp = None
            continue
        stamp, text = m.group(1), m.group(3)
        if out and prev_stamp == stamp:
            out[-1]["text"] += "\n" + text
        else:
            out.append({"time": stamp, "text": text})
        prev_stamp = stamp
    return out


def load_snapshot(state_dir: Union[str, Path], clock: Optional[Clock] = None) -> Snapshot:
    """Una lectura suelta (sin memoria de lecturas anteriores)."""
    return SnapshotReader(state_dir, clock).read()


# --------------------------------------------------------------------------- paneles
def _table(*columns: tuple[str, str], flex: Optional[str] = None) -> Table:
    """Tabla compacta: columnas de una línea con su ancho natural; ``flex`` (la de texto largo) se queda con el
    sitio que sobra y se corta con "…" si no cabe."""
    t = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False, pad_edge=False, collapse_padding=True,
              header_style="bold")
    for name, justify in columns:
        t.add_column(name, justify=justify, no_wrap=True, overflow="ellipsis", ratio=1 if name == flex else None,
                     min_width=4 if name == flex else None)
    return t


def _empty(text: str) -> Text:
    return Text(text, style="dim")


def _mode_label(st: dict[str, Any]) -> Text:
    mode = (st.get("mode") or "").lower()
    if mode == "live":
        net = (st.get("network") or "").lower()
        if net == "mainnet":
            return Text(" LIVE MAINNET (DINERO REAL) ", style="bold white on red")
        return Text(f" LIVE {net.upper()} ".replace("  ", " "), style="bold black on yellow")
    if mode:
        return Text(f" {mode.upper()} ", style="bold black on cyan")
    return Text(" MODO ? ", style="dim")


def build_header(snap: Snapshot) -> RenderableType:
    st = snap.status
    lines: list[RenderableType] = []
    if not st["present"]:
        lines.append(Text(f"Sin status.json en {snap.state_dir}: ¿ha arrancado la mesa con este estado?",
                          style="yellow"))
    elif not st["account_ok"]:
        lines.append(Text(st["line"] or "SIN DATOS DE LA CUENTA", style="bold red"))

    eq = st.get("equity")
    t = Text()
    t.append("Equity ", style="bold")
    t.append(fmt_usd(eq, sign=False) if eq is not None else "-", style="bold")
    spark = _equity_spark(snap.equity_curve)
    if spark:
        trend = pnl_style(snap.equity_curve[-1] - snap.equity_curve[0])
        t.append(f"  {spark}", style=trend)
    if st.get("available") is not None:
        t.append(f"   disponible {fmt_usd(st['available'], sign=False)}", style="dim")
    t.append("   PnL día ", style="bold")
    t.append_text(pnl_text(st.get("day_pnl"), st.get("day_pnl_pct"), bold=True))
    lines.append(t)

    bars = Table.grid(padding=(0, 2))
    bars.add_column()
    bars.add_column()
    day_pct, lim = st.get("day_pnl_pct"), st.get("daily_loss_limit_pct")
    dd, mx = st.get("drawdown_pct"), st.get("max_drawdown_pct")
    b1 = Text("Pérdida diaria ", style="bold") + usage_bar(st.get("daily_used_pct"), 16)
    if day_pct is not None and lim:
        b1.append(f"  {max(0.0, -day_pct):.2f}% de {lim:g}%", style="dim")
    b2 = Text("Drawdown ", style="bold") + usage_bar(st.get("dd_used_pct"), 16)
    if dd is not None and mx:
        b2.append(f"  {dd:.2f}% de {mx:g}%", style="dim")
    bars.add_row(b1, b2)
    lines.append(bars)

    s = Text()
    auto = st.get("auto")
    s.append("Automático ", style="bold")
    s.append("ON" if auto else "OFF" if auto is False else "?", style="green" if auto else "yellow")
    if st.get("paused"):
        s.append(" · EN PAUSA", style="bold yellow")
    if st.get("cooldown"):
        s.append(" · cooldown tras pérdidas", style="yellow")
    s.append("   ")
    halt = st.get("halt")
    if halt:
        detail = _str(snap.risk.get("halt_detail"))
        s.append(f" HALT: {halt}{' - ' + detail if detail else ''} ", style="bold white on red")
    else:
        s.append("Halt ", style="bold")
        s.append("no" if halt == "" else "?", style="green" if halt == "" else "dim")
    a = snap.alert_stats
    if a:
        s.append("   Alarmas ", style="bold")
        s.append(f"{int(_num(a.get('rules')) or 0)} activas")
        if _num(a.get("fired_total")) is not None:
            s.append(f", {int(_num(a.get('fired_total')) or 0)} disparadas")
        if a.get("muted"):
            until = _num(a.get("muted_until"))
            s.append(f" · SILENCIADAS{' hasta ' + _when(until) if until else ''}", style="yellow")
    lines.append(s)

    h = Text()
    if snap.health:
        age = snap.health_age_s
        cycle = _first(snap.health, "cycle")
        h.append(f"Ciclo {cycle if cycle is not None else '-'} (hace {fmt_age(age)})",
                 style="bold red" if age is not None and age > STALE_S else "dim")
        if age is not None and age > STALE_S:
            h.append(" - la mesa parece detenida", style="bold red")
        streak = int(_num(snap.health.get("error_streak")) or 0)
        failing = [str(x) for x in snap.health.get("failing") or []] if isinstance(snap.health.get("failing"),
                                                                                    list) else []
        if streak or failing:
            h.append(f"   ciclos con fallos {streak}{': ' + ', '.join(failing) if failing else ''}", style="yellow")
    else:
        h.append("Sin health.json (¿la mesa no ha completado ningún ciclo?)", style="dim")
    lines.append(h)
    for p in snap.problems:
        lines.append(Text(f"Aviso del panel: {p}", style="yellow"))

    border = "red" if (halt or not st["account_ok"]) else "yellow" if (st.get("paused") or not st["present"]) \
        else "green"
    title = Text.assemble(("Mesa de trading ", "bold"), _mode_label(st),
                          (f" datos {st['data']} " if st.get("data") else "", "dim"))
    sub = f"hora de la mesa {st['time']} UTC · status.json hace {fmt_age(snap.status_age_s)}" if st["present"] \
        else "sin estado"
    return Panel(Group(*lines), title=title, title_align="left", subtitle=Text(sub, style="dim"),
                 subtitle_align="right", border_style=border)


class _ByWidth:
    """Elige qué pintar según el ancho disponible (no se sabe hasta pintar). Un fallo se pinta como error."""

    def __init__(self, build: Callable[[int], RenderableType]):
        self.build = build

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        try:
            yield self.build(options.max_width)
        except Exception as e:  # noqa: BLE001 - el panel nunca se cae por un dato raro
            log.debug("panel: fallo pintando", exc_info=True)
            yield Text(f"No se pudo pintar ({type(e).__name__}: {e})", style="red")


POSITIONS_COMPACT_WIDTH = 90          # por debajo, sin Lev / % / Origen: nunca se recortan precios ni importes


def _positions_table(pos: list[dict[str, Any]], compact: bool) -> Table:
    cols = [("Moneda", "left"), ("Lado", "left"), ("Tamaño", "right"), ("Entrada", "right"), ("Marca", "right"),
            ("PnL", "right"), ("%", "right"), ("Lev", "right"), ("SL", "right"), ("TP", "right"), ("Origen", "left")]
    hidden = {"%", "Lev", "Origen"} if compact else set()
    t = _table(*[c for c in cols if c[0] not in hidden], flex=None if compact else "Origen")
    for p in pos[:MAX_ROWS]:
        pct, lev = p.get("pnl_pct"), p.get("leverage")
        cells = {"Moneda": Text(p["coin"], style="bold"), "Lado": side_text(p.get("side")),
                 "Tamaño": Text(fmt_size(p["size"]) if p.get("size") is not None else "-"),
                 "Entrada": Text(fmt_px(p.get("entry_px"))), "Marca": Text(fmt_px(p.get("mark_px"))),
                 "PnL": pnl_text(p.get("pnl")),
                 "%": Text(f"{pct:+.1f}%" if pct is not None else "-", style=pnl_style(pct)),
                 "Lev": Text(f"{lev:g}x" if lev is not None else "-"), "SL": Text(fmt_px(p.get("sl")), style="red"),
                 "TP": Text(fmt_px(p.get("tp")), style="green"), "Origen": Text(_origin(p.get("origin")), style="dim")}
        t.add_row(*[cells[name] for name, _ in cols if name not in hidden])
    return t


def build_positions(snap: Snapshot) -> RenderableType:
    pos = snap.status.get("positions") or []
    if not pos:
        return Panel(_empty("Sin posiciones abiertas."), title="Posiciones (0)", border_style="blue")
    total = sum(p.get("pnl") or 0.0 for p in pos)
    more = len(pos) - MAX_ROWS
    sub = Text("PnL no realizado ") + pnl_text(total, bold=True)
    if more > 0:
        sub.append(f" · y {more} más", style="dim")
    body = _ByWidth(lambda w: _positions_table(pos, compact=w < POSITIONS_COMPACT_WIDTH))
    return Panel(body, title=f"Posiciones ({len(pos)})", border_style="blue", subtitle=sub, subtitle_align="right")


_KIND_LABELS = {"stop": "stop (SL)", "take_profit": "take profit", "limit": "límite", "market": "mercado"}


def build_orders(snap: Snapshot) -> RenderableType:
    st = snap.status
    pend, orders = st.get("pending") or [], st.get("orders") or []
    n = len(pend) + len(orders)
    if not n:
        return Panel(_empty("Sin órdenes abiertas ni entradas pendientes."), title="Órdenes (0)", border_style="blue")
    t = _table(("Moneda", "left"), ("Tipo", "left"), ("Lado", "left"), ("Tamaño", "right"), ("Precio", "right"),
               ("Notas", "left"), flex="Notas")
    rows = 0
    for b in pend:
        if rows >= MAX_ROWS:
            break
        notes = f"SL {fmt_px(b.get('sl'))} TP {fmt_px(b.get('tp'))}"
        if b.get("expires"):
            notes += f" · caduca {b['expires']}"
        t.add_row(Text(b["coin"], style="bold"), Text("entrada pendiente", style="cyan"), side_text(b.get("side")),
                  Text(fmt_size(b["size"]) if b.get("size") is not None else "-"), Text(fmt_px(b.get("entry_px"))),
                  Text(notes, style="dim"))
        rows += 1
    for o in orders:
        if rows >= MAX_ROWS:
            break
        notes = " ".join(x for x in ("reduce-only" if o.get("reduce_only") else "",
                                     f"[{o['tag']}]" if o.get("tag") else "") if x)
        kind = o.get("kind") or ""
        t.add_row(Text(o["coin"], style="bold"), Text(_KIND_LABELS.get(kind, kind or "-")), side_text(o.get("side")),
                  Text(fmt_size(o["size"]) if o.get("size") is not None else "-"), Text(fmt_px(o.get("px"))),
                  Text(notes, style="dim"))
        rows += 1
    title = f"Órdenes ({len(orders)})" + (f" y entradas pendientes ({len(pend)})" if pend else "")
    return Panel(t, title=title, border_style="blue")


def _metric(x: Optional[float], fmt: str, good: Optional[Callable[[float], bool]] = None) -> Text:
    if x is None:
        return Text("-", style="dim")
    style = "" if good is None else ("green" if good(x) else "red")
    return Text(format(x, fmt), style=style)


OOS_NOTE = "Métricas fuera de muestra (walk-forward). Un backtest no garantiza resultados futuros."


def build_playbook(snap: Snapshot) -> RenderableType:
    pb = snap.playbook
    if not pb:
        return Panel(_empty("Playbook vacío: el automático usa sólo las estrategias por defecto (si están activadas). "
                            "Tras un backtest, 'adopta' añade la gráfica."), title="Playbook (0)", border_style="magenta")
    t = _table(("Moneda", "left"), ("TF", "left"), ("Estrategia", "left"), ("Uds", "right"), ("Ret. OOS", "right"),
               ("PF OOS", "right"), ("DD OOS", "right"), ("Estado", "left"), flex="Estrategia")
    for e in pb[:MAX_ROWS]:
        oos = e.get("oos") or {}
        strat = Text(e.get("strategy") or "-")
        params = _params_text(e.get("params"))
        if params:
            strat.append(f" ({params})", style="dim")
        units = e.get("units")
        t.add_row(Text(e["coin"], style="bold"), Text(e.get("interval") or "-"), strat,
                  Text(f"{units:g}" if units is not None else "-"),
                  _metric(oos.get("ret"), "+.1f", lambda x: x > 0), _metric(oos.get("pf"), ".2f", lambda x: x >= 1),
                  _metric(oos.get("dd"), ".1f"),
                  Text("activa", style="green") if e.get("enabled", True) else Text("desactivada", style="dim"))
    return Panel(Group(t, Text(OOS_NOTE, style="dim italic")), title=f"Playbook ({len(pb)})", border_style="magenta")


def build_backtest(snap: Snapshot) -> RenderableType:
    rep = snap.report
    if not rep:
        return Panel(_empty("Sin informes de backtest todavía (comando: backtest BTC 1h)."), title="Último backtest",
                     border_style="magenta")
    head = Text()
    head.append(f"{rep['coin'] or '?'} {rep['interval'] or ''}", style="bold")
    if rep.get("days"):
        head.append(f" · {rep['days']:g} días")
    if rep.get("bars"):
        head.append(f" · {rep['bars']:g} velas")
    head.append(f" · {_when(rep.get('created_ts'))} UTC", style="dim")
    if rep.get("id"):
        head.append(f" · id {rep['id']}", style="dim")
    win = rep.get("winner")
    w = Text()
    if win:
        w.append("Ganadora: ", style="bold")
        w.append(win["strategy"], style="bold green")
        params = _params_text(win.get("params"))
        if params:
            w.append(f" ({params})", style="dim")
    else:
        w.append("Ninguna estrategia aprobada fuera de muestra: no hay nada que adoptar.", style="yellow")
    parts: list[RenderableType] = [head, w]
    evals = sorted(rep.get("evaluations") or [],
                   key=lambda e: (not e["passed"], -(e["score"] if e["score"] is not None else -math.inf)))
    if evals:
        t = _table(("Estrategia", "left"), ("Ret. OOS", "right"), ("PF", "right"), ("DD", "right"), ("Ops", "right"),
                   ("Veredicto", "left"), flex="Veredicto")
        for e in evals[:MAX_EVALS]:
            oos = e["oos"]
            verdict = Text("aprobada", style="green") if e["passed"] else \
                Text("rechazada" + (f": {e['reasons'][0]}" if e["reasons"] else ""), style="dim")
            t.add_row(Text(e["strategy"]), _metric(oos.get("ret"), "+.1f", lambda x: x > 0),
                      _metric(oos.get("pf"), ".2f", lambda x: x >= 1), _metric(oos.get("dd"), ".1f"),
                      _metric(oos.get("trades"), ".0f"), verdict)
        parts.append(t)
    folds = rep.get("folds")
    note = OOS_NOTE if not folds else OOS_NOTE.replace("(walk-forward)", f"(walk-forward, {folds:g} tramos)")
    parts.append(Text(note, style="dim italic"))
    title = "Último backtest" + (f" ({snap.reports_count} informes)" if snap.reports_count > 1 else "")
    return Panel(Group(*parts), title=title, border_style="magenta")


def build_alerts(snap: Snapshot) -> RenderableType:
    recs = snap.alerts
    title = Text("Alarmas")
    a = snap.alert_stats
    if a and a.get("muted"):
        title.append(" · SILENCIADAS", style="bold yellow")
    if not recs:
        return Panel(_empty("Sin avisos todavía."), title=title, border_style="yellow")
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column(no_wrap=True)
    t.add_column(no_wrap=True)
    t.add_column(ratio=1, overflow="fold")
    for r in reversed(recs):                       # el más reciente arriba
        text = Text(r["text"] or r["kind"] or "-", style="dim" if r["muted"] else "")
        if r["muted"]:
            text.append(" (silenciada)", style="dim italic")
        t.add_row(Text(_when(r["ts"], r["time"]), style="dim"), level_badge(r["level"]), text)
    return Panel(t, title=title, border_style="yellow")


def build_signals(snap: Snapshot) -> RenderableType:
    sigs = snap.signals
    if not sigs:
        return Panel(_empty("Sin señales recientes de las estrategias."), title="Últimas señales", border_style="cyan")
    t = _table(("Hora", "left"), ("Moneda", "left"), ("Lado", "left"), ("Estrategia", "left"), ("Motivo", "left"),
               flex="Motivo")
    for s in reversed(sigs):
        t.add_row(Text(_when(s.get("ts"), s.get("time")), style="dim"), Text(s["coin"], style="bold"),
                  side_text(s.get("side")), Text(s.get("strategy") or "-"), Text(s.get("reason") or "", style="dim"))
    return Panel(t, title="Últimas señales", border_style="cyan")


_EXIT_LABELS = {"sl": "stop-loss", "tp": "take-profit", "close": "cierre", "close_all": "cierra todo",
                "flatten": "halt", "kill": "kill", "no_sl": "sin SL", "liquidation": "LIQUIDACIÓN",
                "external": "externa"}


def build_trades(snap: Snapshot) -> RenderableType:
    tr = snap.trades
    if not tr:
        return Panel(_empty("Sin operaciones cerradas recientes."), title="Operaciones cerradas", border_style="cyan")
    t = _table(("Hora", "left"), ("Moneda", "left"), ("Lado", "left"), ("PnL neto", "right"), ("Salida", "left"),
               flex="Salida")
    for x in reversed(tr):
        t.add_row(Text(_when(x.get("ts")), style="dim"), Text(x["coin"], style="bold"), side_text(x.get("side")),
                  pnl_text(x.get("pnl")), Text(_EXIT_LABELS.get(x.get("exit_reason") or "", x.get("exit_reason") or "-"),
                                               style="dim"))
    total = sum(x.get("pnl") or 0.0 for x in tr)
    return Panel(t, title="Operaciones cerradas", border_style="cyan",
                 subtitle=Text(f"últimas {len(tr)}: ") + pnl_text(total), subtitle_align="right")


# La mesa marca lo grave en mayúsculas ("HALT por pérdida diaria", "Mesa DETENIDA", "ESTADO INCIERTO"...): se
# distingue así de la línea de estado periódica, que siempre lleva "halt no" en minúsculas.
_CRITICAL_MARKS = ("HALT", "DETENIDA", "KILL", "SIN DATOS DE LA CUENTA", "INCIERTO", "NO colocado", "LIQUIDACIÓN",
                   "Riesgo (")
_WARN_MARKS = ("atención", "rechaz", "no se pudo", "no se envió", "fallo", "error", "cooldown", "aviso")
TEXT_STYLES = {"info": "", "warn": "yellow", "critical": "bold red"}


def notice_level(text: str) -> str:
    """Nivel de un aviso de la mesa (``outbox.log`` no lo guarda): "critical", "warn" o "info" por su texto."""
    if any(m in text for m in _CRITICAL_MARKS):
        return "critical"
    low = text.lower()
    return "warn" if any(w in low for w in _WARN_MARKS) else "info"


def _notice_style(text: str) -> str:
    return TEXT_STYLES[notice_level(text)]


def build_notices(snap: Snapshot) -> RenderableType:
    ns = snap.notices
    if not ns:
        return Panel(_empty("Sin avisos de la mesa en outbox.log."), title="Avisos de la mesa", border_style="white")
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column(no_wrap=True)
    t.add_column(ratio=1, overflow="fold")
    for n in reversed(ns):
        text = n["text"] if len(n["text"]) <= NOTICE_CHARS else n["text"][:NOTICE_CHARS - 1] + "…"
        t.add_row(Text(_when(None, n["time"]), style="dim"), Text(text, style=_notice_style(n["text"])))
    return Panel(t, title="Avisos de la mesa", border_style="white")


def _safe(title: str, builder: Callable[[Snapshot], RenderableType], snap: Snapshot) -> RenderableType:
    """Un panel que falla se pinta como error; los demás siguen."""
    try:
        return builder(snap)
    except Exception as e:  # noqa: BLE001 - datos inesperados nunca tumban el panel
        log.debug("panel: fallo construyendo %s", title, exc_info=True)
        return Panel(Text(f"No se pudo pintar ({type(e).__name__}: {e})", style="red"), title=title, border_style="red")


class _Columns:
    """Dos columnas si caben (terminal ancho); si no, una debajo de otra."""

    def __init__(self, left: list[RenderableType], right: list[RenderableType], min_width: int = TWO_COLUMNS_MIN_WIDTH):
        self.left, self.right, self.min_width = left, right, min_width

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if options.max_width >= self.min_width:
            grid = Table.grid(expand=True, padding=(0, 1))
            grid.add_column(ratio=3)
            grid.add_column(ratio=2)
            grid.add_row(Group(*self.left), Group(*self.right))
            yield grid
        else:
            yield Group(*self.left, *self.right)


def build_layout(snap: Snapshot) -> RenderableType:
    """Vista completa del panel para una foto del estado."""
    left = [_safe("Posiciones", build_positions, snap), _safe("Órdenes", build_orders, snap),
            _safe("Playbook", build_playbook, snap), _safe("Último backtest", build_backtest, snap)]
    right = [_safe("Alarmas", build_alerts, snap), _safe("Últimas señales", build_signals, snap),
             _safe("Operaciones cerradas", build_trades, snap), _safe("Avisos de la mesa", build_notices, snap)]
    foot = Text(f"Sólo lectura de {snap.state_dir} · se actualiza cada {REFRESH_S:g} s · Ctrl+C para salir",
                style="dim")
    return Group(_safe("Mesa de trading", build_header, snap), _Columns(left, right), foot)


# --------------------------------------------------------------------------- bucle
def _frame(reader: SnapshotReader) -> RenderableType:
    try:
        return build_layout(reader.read())
    except Exception as e:  # noqa: BLE001 - el panel nunca se cae
        log.exception("panel: error construyendo la vista")
        return Panel(Text(f"Error leyendo el estado: {type(e).__name__}: {e}", style="red"), title="Panel",
                     border_style="red")


def run_dashboard(state_dir: Union[str, Path], *, clock: Optional[Clock] = None, console: Optional[Console] = None,
                  refresh_s: float = REFRESH_S, once: bool = False, max_frames: Optional[int] = None,
                  screen: bool = True) -> int:
    """Panel en vivo hasta Ctrl+C (o ``max_frames`` refrescos). Con ``once`` o sin terminal pinta una vez y sale."""
    clock = clock if clock is not None else RealClock()
    console = console if console is not None else Console()
    reader = SnapshotReader(state_dir, clock)
    if once or not console.is_terminal:
        console.print(_frame(reader))
        return 0
    from rich.live import Live
    frames = 0
    try:
        with Live(_frame(reader), console=console, screen=screen, auto_refresh=False,
                  redirect_stdout=False, redirect_stderr=False) as live:
            while max_frames is None or frames < max_frames:
                clock.sleep(refresh_s)
                frames += 1
                try:
                    live.update(_frame(reader), refresh=True)
                except Exception as e:  # noqa: BLE001 - un fallo de pintado no cierra el panel
                    log.exception("panel: error pintando")
                    live.update(Panel(Text(f"Error pintando: {type(e).__name__}: {e}", style="red")), refresh=True)
    except KeyboardInterrupt:
        pass
    return 0

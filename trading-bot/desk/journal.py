"""Diario de la mesa: todo lo que pasa queda en ``state_dir`` en formatos fáciles de leer.

Ficheros (todos en ``state_dir``):

- ``events.jsonl``: un evento JSON por línea (comandos, decisiones de riesgo, órdenes, halts...).
- ``fills.csv``: cada ejecución del broker.
- ``trades.csv``: cada operación cerrada (entrada → salida) con su PnL neto.
- ``equity.csv``: curva de equity.
- ``outbox.log``: respuestas a comandos, una línea por renglón con el id del comando
  (``2026-01-01 00:00:00 [c-000001] texto``); los avisos de la mesa llevan el id ``desk``.
- ``inbox.txt``: comandos de entrada (lo escribe ``python -m desk send`` o cualquier editor);
  lo lee la mesa, no este módulo.
- ``<nombre>.json``: estados guardados con ``save_json`` (escritura atómica: tmp + fsync + rename).

Un fallo de disco al *añadir* a un diario se registra en el log y no se propaga: el diario nunca
tumba la mesa. ``save_json`` sí lanza ``OSError`` (la mesa lo captura en su etapa de guardado).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import re
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from .clock import Clock, RealClock
from .models import Fill

log = logging.getLogger(__name__)

EVENTS_FILE = "events.jsonl"
FILLS_FILE = "fills.csv"
TRADES_FILE = "trades.csv"
EQUITY_FILE = "equity.csv"
OUTBOX_FILE = "outbox.log"
INBOX_FILE = "inbox.txt"

FILL_COLUMNS = ["time", "ts", "coin", "side", "size", "price", "fee", "realized_pnl", "tag", "order_id",
                "liquidation"]
TRADE_COLUMNS = ["closed_time", "coin", "side", "origin", "size", "entry_px", "exit_px", "pnl", "gross_pnl",
                 "fees", "exit_reason", "opened_ts", "closed_ts", "duration_min", "reason"]
EQUITY_COLUMNS = ["time", "ts", "equity", "available", "positions"]

# Línea del outbox: "<fecha> <hora> [<id>] <texto>"
OUTBOX_RE = re.compile(r"^(\S+ \S+) \[([^\]]+)\] ?(.*)$")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _jsonable(o: Any) -> Any:
    """``default`` de ``json.dumps``: enums, dataclasses, rutas y NaN/inf (como None)."""
    if isinstance(o, Enum):
        return o.value
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _clean(v: Any) -> Any:
    """NaN/inf no son JSON válido: se guardan como None."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


class Journal:
    def __init__(self, state_dir: Union[str, Path], clock: Optional[Clock] = None):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        # Sin reloj se usa el real; la mesa inyecta el suyo (SimClock en tests y en --fast).
        self.clock: Clock = clock if clock is not None else RealClock()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ rutas
    def path(self, name: str) -> Path:
        return self.dir / name

    @property
    def inbox_path(self) -> Path:
        return self.dir / INBOX_FILE

    @property
    def outbox_path(self) -> Path:
        return self.dir / OUTBOX_FILE

    def _json_path(self, name: str) -> Path:
        return self.dir / (name if name.endswith(".json") else f"{name}.json")

    # ------------------------------------------------------------------ escritura
    def _append(self, name: str, text: str) -> None:
        try:
            with self._lock, open(self.dir / name, "a", encoding="utf-8", newline="") as f:
                f.write(text)
        except OSError as e:
            log.error("diario: no se pudo escribir en %s: %s", name, e)

    def _append_csv(self, name: str, columns: list[str], row: dict[str, Any]) -> None:
        path = self.dir / name
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        with self._lock:
            try:
                new = not path.exists() or path.stat().st_size == 0
            except OSError:
                new = True
            if new:
                w.writeheader()
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
            self._append(name, buf.getvalue())

    def event(self, kind: str, /, **data: Any) -> dict[str, Any]:
        """Añade un evento a ``events.jsonl`` y lo devuelve (``kind`` manda sobre un campo homónimo)."""
        now = self.clock.now()
        rec = {"ts": now, "time": _iso(now), **data, "kind": kind}
        try:
            line = json.dumps(_clean(rec), default=_jsonable, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            line = json.dumps({"ts": now, "time": _iso(now), "kind": kind, "error": f"evento no serializable: {e}"})
        self._append(EVENTS_FILE, line + "\n")
        return rec

    def fill(self, fill: Fill) -> None:
        self._append_csv(FILLS_FILE, FILL_COLUMNS, {
            "time": _iso(fill.ts), "ts": fill.ts, "coin": fill.coin, "side": "buy" if fill.is_buy else "sell",
            "size": fill.size, "price": fill.price, "fee": round(fill.fee, 8),
            "realized_pnl": round(fill.realized_pnl, 8), "tag": fill.tag, "order_id": fill.order_id,
            "liquidation": int(bool(fill.liquidation)),
        })

    def trade(self, record: dict[str, Any]) -> None:
        """Operación cerrada (ver ``TRADE_COLUMNS``) en ``trades.csv``."""
        row = dict(record)
        row.setdefault("closed_time", _iso(float(row.get("closed_ts") or self.clock.now())))
        for k in ("pnl", "gross_pnl", "fees"):
            if isinstance(row.get(k), float):
                row[k] = round(row[k], 6)
        for k in ("entry_px", "exit_px", "size"):
            if isinstance(row.get(k), float):
                row[k] = float(f"{row[k]:.10g}")
        self._append_csv(TRADES_FILE, TRADE_COLUMNS, row)

    def equity(self, ts: float, equity: float, available: Optional[float] = None,
               positions: Optional[int] = None) -> None:
        self._append_csv(EQUITY_FILE, EQUITY_COLUMNS, {
            "time": _iso(ts), "ts": ts, "equity": round(float(equity), 6),
            "available": None if available is None else round(float(available), 6), "positions": positions,
        })

    def outbox(self, cmd_id: str, text: str) -> None:
        """Respuesta a un comando: cada renglón lleva la hora y el id (``grep <id> outbox.log``)."""
        stamp = _iso(self.clock.now())
        lines = str(text).splitlines() or [""]
        self._append(OUTBOX_FILE, "".join(f"{stamp} [{cmd_id}] {ln}\n" for ln in lines))

    def touch(self, name: str, text: str = "") -> None:
        """Escribe un fichero pequeño sin fsync (latido para healthchecks)."""
        try:
            (self.dir / name).write_text(text, encoding="utf-8")
        except OSError as e:
            log.error("diario: no se pudo escribir %s: %s", name, e)

    # ------------------------------------------------------------------ estado JSON
    def save_json(self, name: str, obj: Any) -> None:
        """Escritura atómica: un corte de luz deja el fichero anterior o el nuevo, nunca uno a medias."""
        path = self._json_path(name)
        tmp = path.with_name(path.name + ".tmp")
        data = json.dumps(_clean(obj), default=_jsonable, ensure_ascii=False, indent=1)
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

    def load_json(self, name: str, default: Any = None) -> Any:
        """Contenido guardado o ``default``. Un fichero ilegible se aparta (``*.corrupt-<ts>``) y se avisa."""
        path = self._json_path(name)
        with self._lock:
            if not path.exists():
                return default
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                backup = path.with_name(f"{path.name}.corrupt-{int(self.clock.now())}")
                try:
                    os.replace(path, backup)
                except OSError:
                    backup = path
                log.error("diario: %s ilegible (%s); copia en %s, se usa el valor por defecto",
                          path.name, e, backup.name)
                return default


def read_outbox(path: Union[str, Path], cmd_id: str, offset: int = 0) -> tuple[list[str], int]:
    """Renglones de ``outbox.log`` con ``cmd_id`` a partir de ``offset`` (bytes). Devuelve (renglones, nuevo offset).

    Sólo consume líneas completas (terminadas en salto de línea)."""
    p = Path(path)
    try:
        with open(p, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    out = []
    for raw in data[:end + 1].decode("utf-8", "replace").splitlines():
        m = OUTBOX_RE.match(raw)
        if m and m.group(2) == cmd_id:
            out.append(m.group(3))
    return out, offset + end + 1

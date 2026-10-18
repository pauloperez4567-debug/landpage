"""Backtests por gráfica (moneda + temporalidad) fuera del bucle 24/7.

Un backtest walk-forward de las 5 estrategias tarda segundos (más con 5000 velas): si corriera dentro del ciclo,
la mesa dejaría de vigilar posiciones, alarmas y comandos mientras tanto. Por eso:

- ``fetch_bars`` pide las velas al ``MarketData`` y se queda con las **cerradas**. Velas = días · velas por día,
  con tope ``MAX_BARS`` (5000, lo que da Hyperliquid como mucho): si se recorta se dice. Con hilo, la descarga
  la hace el propio trabajo (``BacktestJob.fetch``): con una API lenta puede tardar decenas de segundos y no debe
  bloquear el bucle de la mesa (que espera el mismo lock que los comandos de Telegram). Quien recibe el comando
  sólo valida lo barato (moneda, temporalidad) y un fallo de datos llega en el aviso del resultado.
- ``BacktestWorker`` ejecuta los trabajos de uno en uno, en orden de llegada, en un hilo propio (cola con tope
  ``MAX_QUEUE``). Los resultados se recogen con ``poll()`` desde el bucle de la mesa (que es quien avisa): el
  hilo nunca toca la mesa. ``stop()`` (al apagar) cancela lo pendiente y pide parar al que corre
  (``optimize`` lo consulta entre combinaciones).
- ``threaded=False`` (reloj simulado, ``--fast`` y tests): nada de hilos; ``poll()`` ejecuta lo pendiente en el
  momento en que se llama. Determinista: el mismo comando en la misma vela da el mismo informe.
- ``run_job``: ``optimize`` -> ``ReportStore.save`` -> HTML (``render_html``) y PNG (``render_png``, si hay
  matplotlib) junto al JSON del informe (``reports/<COIN>_<tf>_<id>.html|png``). Un fallo al pintar no pierde
  el informe (queda una nota); cualquier otra excepción se devuelve como ``error`` del resultado: el hilo nunca
  muere y la mesa sigue.

Honestidad: el informe enseña dentro y fuera de muestra, pero sólo lo de fuera de muestra decide (ver
``optimizer``); un backtest no garantiza el rendimiento futuro.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from .clock import Clock
from .config import Settings
from .exchange.base import MarketDataError
from .exchange.intervals import interval_seconds
from .models import AssetMeta
from .optimizer import MIN_SEGMENT_BARS, OptimizationReport, ReportStore, optimize
from .strategy import closed_candles

log = logging.getLogger(__name__)

MAX_BARS = 5000          # velas que da Hyperliquid como mucho en una petición (incluida la vela en curso)
MIN_BARS = 2 * MIN_SEGMENT_BARS   # un tramo de entrenamiento y uno de validación como mínimo
MAX_QUEUE = 4            # trabajos en espera (además del que corre)
STOP_JOIN_S = 5.0


def bars_for(days: float, interval: str) -> tuple[int, bool]:
    """(velas cerradas a pedir para ``days`` días de ``interval``, recortado a ``MAX_BARS``)."""
    per_day = 86_400.0 / interval_seconds(interval)
    want = max(int(math.ceil(float(days) * per_day)), 1)
    cap = MAX_BARS - 1                     # la petición incluye la vela en curso, que se descarta
    return min(want, cap), want > cap


def fetch_bars(market: Any, coin: str, interval: str, days: float, clock: Clock) -> tuple[pd.DataFrame, list[str]]:
    """Velas **cerradas** de la gráfica para ``days`` días y notas para el usuario (recorte, menos historia).

    Lanza lo que lance ``market.candles`` (``KeyError`` moneda desconocida, ``MarketDataError``)."""
    n, trimmed = bars_for(days, interval)
    candles = market.candles(coin, interval, n + 1)
    bars = closed_candles(candles, interval, clock.now())
    bars = bars.tail(n) if bars is not None else bars
    notes: list[str] = []
    per_day = 86_400.0 / interval_seconds(interval)
    if trimmed:
        notes.append(f"historia recortada a {n} velas (~{n / per_day:.0f} días de {interval}): Hyperliquid da "
                     f"como mucho {MAX_BARS} velas por gráfica")
    got = 0 if bars is None else len(bars)
    if got and got < n * 0.95 and not trimmed:
        notes.append(f"sólo hay {got} velas cerradas (~{got / per_day:.0f} días) de las {n} pedidas")
    return bars, notes


@dataclass
class BacktestJob:
    """Un backtest pedido: la gráfica, sus velas cerradas (ya descargadas, o ``fetch`` para descargarlas en el hilo
    del trabajo) y quién lo pidió. ``start_equity`` = capital de la cuenta al pedirlo (None = ``paper_start_equity``)."""

    id: str
    coin: str
    interval: str
    days: float                            # días pedidos
    strategy: Optional[str]                # None = todas
    bars: Optional[pd.DataFrame]
    meta: Optional[AssetMeta] = None
    requested_ts: float = 0.0
    cmd_id: str = ""
    notes: list[str] = field(default_factory=list)
    source: str = ""                       # quién lo pidió ("consola", "telegram", "inbox"...)
    settings: Optional[Settings] = None    # copia de la configuración al pedirlo (None = la del trabajador)
    start_equity: Optional[float] = None   # equity de la cuenta al pedirlo
    fetch: Optional[Callable[[], tuple[pd.DataFrame, list[str]]]] = field(default=None, repr=False)

    @property
    def chart(self) -> tuple[str, str]:
        return self.coin.upper(), self.interval

    def label(self) -> str:
        what = f", {self.strategy}" if self.strategy else ""
        return f"{self.coin} {self.interval} ({self.days:g} días{what})"


class JobDataError(Exception):
    """Sin velas suficientes para el backtest: el texto va tal cual al usuario."""


@dataclass
class BacktestOutcome:
    """Resultado de un trabajo. ``error`` != "" si falló; ``canceled`` si se paró al apagar la mesa."""

    job: BacktestJob
    report: Optional[OptimizationReport] = None
    report_path: Optional[Path] = None
    html_path: Optional[Path] = None
    png_path: Optional[Path] = None
    png: Optional[bytes] = None
    error: str = ""
    canceled: bool = False
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.report is not None and not self.error and not self.canceled


def run_job(job: BacktestJob, settings: Settings, store: ReportStore, clock: Clock, *,
            should_stop: Optional[Callable[[], bool]] = None,
            progress: Optional[Callable[[int, int, str], Any]] = None, render: bool = True) -> BacktestOutcome:
    """Descarga las velas si hace falta (``job.fetch``), optimiza la gráfica, guarda el informe y pinta HTML/PNG.
    Lanza si la descarga, ``optimize`` o el guardado fallan (``BacktestWorker`` lo convierte en
    ``BacktestOutcome.error``); ``JobDataError`` con un texto para el usuario si no hay velas suficientes."""
    t0 = time.monotonic()
    settings = job.settings if job.settings is not None else settings
    if job.bars is None and job.fetch is not None:
        try:
            job.bars, notes = job.fetch()
        except (KeyError, ValueError) as e:        # temporalidad que esta fuente de datos no sirve, etc.
            raise JobDataError(f"sin velas de {job.coin} {job.interval} ({e})") from e
        except MarketDataError as e:               # API caída o lenta: se puede repetir
            raise JobDataError(f"no se pudieron descargar las velas de {job.coin} {job.interval} ({e}); "
                               "repítelo en un rato") from e
        job.notes = list(notes) + [n for n in job.notes if n not in notes]
    n = 0 if job.bars is None else len(job.bars)
    if n < MIN_BARS:
        raise JobDataError(f"{job.coin} {job.interval}: sólo hay {n} velas cerradas y el walk-forward necesita al "
                           f"menos {MIN_BARS} (tramos de entrenamiento y de validación). Prueba con más días o una "
                           "temporalidad menor.")
    report = optimize(job.coin, job.interval, job.bars, settings,
                      strategies=[job.strategy] if job.strategy else None, meta=job.meta,
                      progress=progress, should_stop=should_stop, clock=clock, start_equity=job.start_equity)
    for note in job.notes:
        if note not in report.notes:
            report.notes.insert(0, note)
    out = BacktestOutcome(job=job, report=report)
    if should_stop is not None and not report.complete and should_stop():
        out.canceled = True                # se pidió parar (la mesa se apaga): no se guarda un informe a medias
        out.elapsed_s = time.monotonic() - t0
        return out
    result = report.result()
    out.report_path = store.save(report, result)
    if render:
        _render(out, result)
    out.elapsed_s = time.monotonic() - t0
    return out


def _render(out: BacktestOutcome, result: Any) -> None:
    """HTML y PNG junto al JSON del informe. Un fallo aquí sólo deja una nota en el log (el informe ya está)."""
    from .report_render import render_html, render_png
    base = out.report_path
    try:
        html = render_html(out.report, result, out.job.bars)
        path = base.with_suffix(".html")
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(html, encoding="utf-8")
        tmp.replace(path)
        out.html_path = path
    except Exception as e:  # noqa: BLE001 - el informe JSON ya está guardado
        log.warning("backtest %s: no se pudo escribir el HTML: %s", out.job.label(), e)
    try:
        png = render_png(out.report, result, out.job.bars)
        if png:
            path = base.with_suffix(".png")
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(png)
            tmp.replace(path)
            out.png_path, out.png = path, png
    except Exception as e:  # noqa: BLE001
        log.warning("backtest %s: no se pudo escribir el PNG: %s", out.job.label(), e)
    _prune_renders(base.parent)


def _prune_renders(folder: Path) -> None:
    """Borra HTML/PNG cuyo informe JSON ya no existe (``ReportStore`` poda los JSON antiguos)."""
    try:
        for p in list(folder.glob("*.html")) + list(folder.glob("*.png")):
            if not p.with_suffix(".json").exists():
                p.unlink()
    except OSError as e:
        log.debug("no se pudieron podar informes pintados: %s", e)


class BacktestWorker:
    """Cola de backtests de uno en uno (ver el docstring del módulo). Seguro entre hilos."""

    def __init__(self, settings: Settings, store: ReportStore, clock: Clock, *, threaded: bool = True,
                 max_queue: int = MAX_QUEUE, runner: Optional[Callable[..., BacktestOutcome]] = None):
        self.settings = settings
        self.store = store
        self.clock = clock
        self.threaded = bool(threaded)
        self.max_queue = max(int(max_queue), 0)
        self.runner = runner or run_job
        self._cond = threading.Condition()
        self._queue: deque[BacktestJob] = deque()
        self._done: deque[BacktestOutcome] = deque()
        self._running: Optional[BacktestJob] = None
        self._progress: tuple[int, int] = (0, 0)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.finished = 0
        self.failed = 0

    # ------------------------------------------------------------------ consultas
    def busy_with(self, coin: str, interval: str, strategy: Optional[str] = None) -> bool:
        """True si ya hay un trabajo (en curso o en cola) de esa gráfica y estrategia."""
        key = (str(coin).upper(), interval, strategy)
        with self._cond:
            jobs = list(self._queue) + ([self._running] if self._running else [])
            return any((j.coin.upper(), j.interval, j.strategy) == key for j in jobs)

    def busy_chart(self, coin: str, interval: str) -> bool:
        """True si hay algún trabajo (en curso o en cola) de esa gráfica, sea cual sea la estrategia."""
        key = (str(coin).upper(), interval)
        with self._cond:
            jobs = list(self._queue) + ([self._running] if self._running else [])
            return any((j.coin.upper(), j.interval) == key for j in jobs)

    def pending(self) -> int:
        with self._cond:
            return len(self._queue) + (1 if self._running else 0)

    def has_results(self) -> bool:
        with self._cond:
            return bool(self._done)

    def describe(self) -> dict[str, Any]:
        """Estado para ``status.json`` y el informe de estado."""
        with self._cond:
            run = self._running
            done, total = self._progress
            return {"running": run.label() if run else None,
                    "progress_pct": round(done * 100.0 / total, 1) if run and total else None,
                    "queued": [j.label() for j in self._queue], "finished": self.finished, "failed": self.failed,
                    "threaded": self.threaded}

    # ------------------------------------------------------------------ cola
    def submit(self, job: BacktestJob) -> int:
        """Encola ``job``. Devuelve cuántos trabajos tiene delante (0 = empieza ya). ``ValueError`` si la cola está
        llena o la mesa se está apagando."""
        with self._cond:
            if self._stop.is_set():
                raise ValueError("la mesa se está deteniendo: no se aceptan backtests")
            if len(self._queue) >= self.max_queue:
                raise ValueError(f"ya hay {len(self._queue)} backtests en cola; espera a que terminen")
            ahead = len(self._queue) + (1 if self._running else 0)
            self._queue.append(job)
            self._cond.notify_all()
        if self.threaded:
            self._ensure_thread()
        return ahead

    def poll(self) -> list[BacktestOutcome]:
        """Resultados terminados desde la última llamada. Sin hilo, ejecuta aquí lo pendiente (síncrono)."""
        if not self.threaded:
            while True:
                with self._cond:
                    if self._stop.is_set() or not self._queue:
                        break
                    job = self._queue.popleft()
                    self._running = job
                self._run(job)
        with self._cond:
            out = list(self._done)
            self._done.clear()
        return out

    def stop(self, timeout: float = STOP_JOIN_S) -> int:
        """Cancela lo pendiente y pide parar al trabajo en curso. Devuelve cuántos trabajos se cancelaron."""
        with self._cond:
            self._stop.set()
            n = len(self._queue) + (1 if self._running else 0)
            self._queue.clear()
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(timeout, 0.0))
        return n

    # ------------------------------------------------------------------ ejecución
    def _ensure_thread(self) -> None:
        with self._cond:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, name="desk-backtest", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop.is_set():
                    self._cond.wait(timeout=1.0)
                if self._stop.is_set():
                    return
                job = self._queue.popleft()
                self._running = job
            self._run(job)

    def _on_progress(self, done: int, total: int, text: str) -> None:
        with self._cond:
            self._progress = (int(done), int(total))

    def _run(self, job: BacktestJob) -> None:
        t0 = time.monotonic()
        with self._cond:
            self._progress = (0, 0)
        try:
            out = self.runner(job, self.settings, self.store, self.clock, should_stop=self._stop.is_set,
                              progress=self._on_progress)
        except JobDataError as e:
            log.warning("backtest %s sin datos: %s", job.label(), e)
            out = BacktestOutcome(job=job, error=str(e), elapsed_s=time.monotonic() - t0)
        except Exception as e:  # noqa: BLE001 - un backtest roto nunca tumba el hilo ni la mesa
            log.exception("backtest %s falló", job.label())
            out = BacktestOutcome(job=job, error=f"{type(e).__name__}: {e}", elapsed_s=time.monotonic() - t0)
        with self._cond:
            self._running = None
            if out.canceled or self._stop.is_set():
                log.info("backtest %s cancelado al detener la mesa", job.label())
                return
            if out.error:
                self.failed += 1
            else:
                self.finished += 1
            self._done.append(out)


__all__ = ["MAX_BARS", "MIN_BARS", "BacktestJob", "BacktestOutcome", "BacktestWorker", "JobDataError", "bars_for",
           "fetch_bars", "run_job"]

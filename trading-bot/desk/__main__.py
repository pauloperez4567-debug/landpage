"""Línea de comandos de la mesa.

    python -m desk run [--mode paper|live] [--data hyperliquid|sim|replay:x.csv] [--symbols BTC,ETH]
                       [--no-auto] [--fast] [--cycles N] [--loop-seconds S] [--cmd "N:texto"] [--state-dir DIR]
    python -m desk send "long HYPE"        (escribe en state/inbox.txt y espera la respuesta en outbox.log)
    python -m desk status                  (último estado guardado por la mesa y avisos de salud)
    python -m desk health                  (0 = sana; 1 = parada o sin ver el exchange: para healthchecks)
    python -m desk dashboard [--once]      (panel en vivo de sólo lectura sobre los ficheros de estado)
    python -m desk console                 (consola interactiva conectada a la mesa en marcha por inbox/outbox)
    python -m desk backtest --coin SOL --interval 1h --days 120 [--strategy X] [--data sim | --csv velas.csv]
                                           (backtest walk-forward suelto: resumen + informe JSON/HTML/PNG)

``--fast`` usa un reloj simulado (``SimClock``) con datos ``sim`` o ``replay``: cada ciclo avanza
``loop_seconds`` de tiempo simulado sin esperar, así se simulan días en segundos. Su estado va a
``<state_dir>/fast`` (no se mezcla con el paper real) y continúa donde lo dejó la simulación anterior
(la hora simulada más reciente que guardó la mesa, también tras un corte brusco) salvo ``--fresh``.

``profile.json`` guarda con qué configuración se creó el estado (modo, datos, reloj y, en live, red y cuenta):
arrancar con otra sobre el mismo estado se rechaza.

Mientras corre, la consola acepta comandos (``long HYPE``, ``estado``, ``ayuda``...); ``salir`` detiene la mesa.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from .clock import Clock, RealClock, SimClock
from .config import LIVE_CONFIRM_PHRASE, Settings, load_settings
from .journal import Journal
# append_inbox_line se reexporta: la usan los tests y quien escribía en el inbox importándola de aquí.
from .remote import DEFAULT_TIMEOUT_S, append_inbox_line, no_reply_message, send_command, wait_reply  # noqa: F401

log = logging.getLogger("desk.cli")

PROFILE_FILE = "profile"     # profile.json: configuración con la que se creó el estado
HEALTH_FILE = "health.json"  # salud de cada ciclo (la escribe la mesa)
HEARTBEAT_FILE = "heartbeat"  # último ciclo que leyó la cuenta y los fills
HEALTH_MAX_AGE_S = 600.0     # sin ciclos en 10 min: parada (el backoff máximo entre ciclos es de 5)
STAGE_MAX_STALE_S = 600.0    # sin leer la cuenta / los fills en 10 min: no ve el exchange
# Claves del perfil que existen desde la primera versión; las demás (red, cuenta) sólo se comprueban si el
# estado las tiene guardadas (un perfil antiguo se acepta y se completa).
BASE_PROFILE_KEYS = ("mode", "data", "clock")


# --------------------------------------------------------------------------- utilidades
class ClockFormatter(logging.Formatter):
    """Formato de log con la hora del reloj de la mesa (simulada en --fast)."""

    def __init__(self, fmt: str, clock: Clock):
        super().__init__(fmt)
        self.clock = clock

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        return self.clock.iso()


class _NoNotices(logging.Filter):
    """La consola ya muestra los avisos de la mesa (``out``): no repetirlos como log."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name != "desk.notice"


def setup_logging(state_dir: Path, clock: Clock, verbose: bool) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    state_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(state_dir / "desk.log", maxBytes=5_000_000, backupCount=5,
                                              encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(ClockFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s", clock))
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO if verbose else logging.WARNING)
    ch.addFilter(_NoNotices())
    ch.setFormatter(ClockFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s", clock))
    root.addHandler(ch)
    for noisy in ("urllib3", "requests", "httpx", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def make_out(clock: Clock, stream: Any = None) -> Callable[[str], None]:
    lock = threading.Lock()

    def out(text: str) -> None:
        s = stream or sys.stdout
        stamp = clock.iso()
        lines = str(text).splitlines() or [""]
        pad = " " * len(stamp)
        with lock:
            s.write(f"{stamp} {lines[0]}\n" + "".join(f"{pad} {ln}\n" for ln in lines[1:]))
            s.flush()
    return out


def _load_env(env_file: Optional[str]) -> None:
    if env_file and Path(env_file).exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file, override=False)
        except ImportError:
            pass


def _base_state_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "state_dir", None):
        return Path(args.state_dir)
    _load_env(getattr(args, "env_file", ".env"))
    base = Path(os.environ.get("DESK_STATE_DIR") or "state")
    return base / "fast" if getattr(args, "fast", False) else base


def _data_kind(data_source: str) -> str:
    return data_source.split(":", 1)[0].strip().lower()


# --------------------------------------------------------------------------- construcción
def build_market(settings: Settings, clock: Clock) -> Any:
    kind = _data_kind(settings.data_source)
    if kind == "sim":
        from .exchange.sim import SimMarketData
        market = SimMarketData(clock)
        unknown = [c for c in settings.symbols if c not in market.coins()]
        if unknown:
            log.warning("datos sim: monedas sin simulación ignoradas: %s", ", ".join(unknown))
            settings.symbols = [c for c in settings.symbols if c in market.coins()] or market.coins()
        return market
    if kind == "replay":
        path = settings.data_source.split(":", 1)[1] if ":" in settings.data_source else ""
        if not path:
            raise SystemExit("DESK_DATA=replay necesita una ruta: replay:datos/btc_15m.csv")
        from .exchange.sim import ReplayMarketData
        market = ReplayMarketData(path, clock)
        settings.symbols = [c for c in settings.symbols if c in market.coins()] or market.coins()
        return market
    if kind == "hyperliquid":
        from .exchange.hyperliquid_data import HyperliquidMarketData
        return HyperliquidMarketData.from_settings(settings, clock)
    raise SystemExit(f"DESK_DATA desconocido: {settings.data_source!r} (usa hyperliquid, sim o replay:<csv>)")


def build_broker(settings: Settings, market: Any, clock: Clock, journal: Journal) -> Any:
    if settings.is_live:
        from .exchange.hyperliquid_live import LiveHyperliquidBroker
        saved = journal.load_json("desk", {}) or {}
        since = saved.get("fills_cursor_ts")
        seen = saved.get("fills_seen")
        # El cursor es inclusivo: los fills ya entregados en ese milisegundo no se vuelven a entregar.
        return LiveHyperliquidBroker.from_settings(market, settings, clock,
                                                   fills_since_ts=float(since) if since else None,
                                                   fills_seen=[str(k) for k in seen] if isinstance(seen, list) else None)
    from .exchange.paper import PaperBroker
    broker = PaperBroker.from_settings(market, settings, clock)
    if broker.load_error:
        log.error("%s", broker.load_error)
    return broker


def live_account(settings: Settings) -> Optional[str]:
    """Cuenta que opera la mesa en live (``HL_ACCOUNT_ADDRESS`` o la de la clave), en minúsculas; None si no se sabe."""
    if settings.hl_account_address:
        return settings.hl_account_address.strip().lower()
    try:
        from .exchange.hyperliquid_live import _wallet_from_key
        return str(_wallet_from_key(settings.hl_private_key).address).lower()
    except (RuntimeError, ValueError):      # sin eth-account o clave inválida: lo dirá el broker al arrancar
        return None


def build_profile(settings: Settings, fast: bool) -> dict[str, Any]:
    """Configuración a la que pertenece el estado. En live incluye la red y la cuenta: el máximo de equity y la
    referencia diaria de testnet no valen para mainnet (dispararían un halt por drawdown y un cierre de todo)."""
    profile: dict[str, Any] = {"mode": settings.mode, "data": settings.data_source, "clock": "sim" if fast else "real"}
    if settings.is_live:
        profile["network"] = "mainnet" if settings.hl_mainnet else "testnet"
        account = live_account(settings)
        if account:
            profile["account"] = account
    return profile


def check_profile(journal: Journal, profile: dict[str, Any]) -> None:
    """El estado guardado (posiciones paper, brackets, riesgo) sólo vale para la misma configuración."""
    saved = journal.load_json(PROFILE_FILE, None)
    if isinstance(saved, dict):
        keys = [k for k in profile if k in BASE_PROFILE_KEYS or k in saved]
        old = {k: saved.get(k) for k in keys}
        new = {k: profile[k] for k in keys}
        if old != new:
            raise SystemExit(
                f"El estado en {journal.dir} es de otra configuración ({old}) y ahora se pide {new}.\n"
                "Mezclarlos daría posiciones y límites de riesgo sin sentido (en live, el máximo de equity de otra red "
                "o cuenta haría saltar el halt por drawdown y cerrar todo). Usa otro --state-dir (p. ej. uno por red "
                "y cuenta) o borra esa carpeta.")
    data = dict(saved) if isinstance(saved, dict) else {}
    data.update(profile)
    journal.save_json(PROFILE_FILE, data)


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    """JSON de un fichero que escribe la mesa (sin apartarlo si está mal, a diferencia de ``Journal.load_json``)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def resume_clock_ts(state_dir: Path) -> Optional[float]:
    """Hora simulada desde la que sigue una simulación ``--fast``: la más reciente que guardó la mesa (``clock_ts``
    al salir, ``health.json`` y ``heartbeat`` de cada ciclo, ``status.json``). Así un corte brusco (SIGKILL, OOM)
    no hace volver el reloj atrás respecto al estado guardado."""
    found: list[float] = []
    prof = _read_json(state_dir / f"{PROFILE_FILE}.json") or {}
    for d, key in ((prof, "clock_ts"), (_read_json(state_dir / HEALTH_FILE) or {}, "ts"),
                   (_read_json(state_dir / "status.json") or {}, "ts")):
        try:
            found.append(float(d[key]))
        except (KeyError, TypeError, ValueError):
            pass
    try:
        found.append(float((state_dir / HEARTBEAT_FILE).read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        pass
    return max(found) if found else None


# --------------------------------------------------------------------------- run
def cmd_run(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {}
    if args.mode:
        overrides["mode"] = args.mode
    if args.data:
        overrides["data_source"] = args.data
    if args.symbols:
        overrides["symbols"] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.no_auto:
        overrides["auto_enabled"] = False
    if args.loop_seconds:
        overrides["loop_seconds"] = float(args.loop_seconds)
    _load_env(args.env_file)
    if args.fast and not os.environ.get("DESK_STATUS_EVERY_SECONDS"):
        overrides["status_every_seconds"] = 3600.0    # simulando días, una línea de estado por hora simulada
    try:
        settings = load_settings(dotenv_path=args.env_file, **overrides)
    except ValueError as e:
        print(f"Configuración inválida: {e}", file=sys.stderr)
        return 2
    kind = _data_kind(settings.data_source)
    if args.fast and kind not in ("sim", "replay"):
        print("--fast sólo tiene sentido con datos simulados: añade --data sim o --data replay:<csv>", file=sys.stderr)
        return 2
    if args.fresh and not args.fast:
        print("--fresh sólo se permite con --fast (borra el estado de la simulación)", file=sys.stderr)
        return 2
    state_dir = Path(args.state_dir) if args.state_dir else (
        settings.state_path / "fast" if args.fast else settings.state_path)
    if args.fresh and state_dir.exists():
        prof = Journal(state_dir, RealClock()).load_json(PROFILE_FILE, None)
        if any(state_dir.iterdir()) and not (isinstance(prof, dict) and prof.get("clock") == "sim"):
            print(f"--fresh sólo borra estados de simulaciones --fast y {state_dir} no lo es; bórralo a mano si "
                  "de verdad quieres empezar de cero.", file=sys.stderr)
            return 2
        shutil.rmtree(state_dir)
    settings.state_dir = str(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    clock: Clock
    if args.fast:
        from .exchange.sim import DEFAULT_ANCHOR_TS
        clock = SimClock(resume_clock_ts(state_dir) or DEFAULT_ANCHOR_TS)
    else:
        clock = RealClock()
    setup_logging(state_dir, clock, args.verbose)
    journal = Journal(state_dir, clock)
    check_profile(journal, build_profile(settings, args.fast))
    out = make_out(clock)

    if settings.is_live:
        net = "MAINNET (DINERO REAL)" if settings.hl_mainnet else "TESTNET"
        out(f"*** MODO LIVE en Hyperliquid {net} ***  (frase {LIVE_CONFIRM_PHRASE} aceptada)")

    from .desk import TradingDesk
    try:
        market = build_market(settings, clock)
        broker = build_broker(settings, market, clock, journal)
        desk = TradingDesk(settings, market, broker, clock, journal=journal, out=out)
    except (ValueError, RuntimeError) as e:
        print(f"No se pudo arrancar la mesa: {e}", file=sys.stderr)
        return 2

    scheduled: dict[int, list[str]] = {}
    for spec in args.cmd or []:
        head, sep, rest = spec.partition(":")
        if sep and head.strip().isdigit():
            scheduled.setdefault(int(head), []).append(rest.strip())
        else:
            scheduled.setdefault(0, []).append(spec.strip())

    def before_cycle(n: int) -> None:
        for text in scheduled.pop(n, []):
            desk.enqueue(text, source="--cmd")

    if not args.no_input and sys.stdin is not None and sys.stdin.isatty():
        def console() -> None:
            for line in sys.stdin:
                text = line.strip()
                if not text:
                    continue
                if text.lower() in {"salir", "exit", "quit"}:
                    out("Deteniendo la mesa...")
                    desk.request_stop()
                    return
                out(f"» {text}\n{desk.submit(text, source='consola')}")
        threading.Thread(target=console, name="consola", daemon=True).start()

    try:
        desk.run_forever(args.cycles, before_cycle=before_cycle)
    finally:
        if args.fast:                        # la próxima simulación continúa desde aquí
            data = journal.load_json(PROFILE_FILE, {}) or {}
            data["clock_ts"] = clock.now()
            journal.save_json(PROFILE_FILE, data)
    out(desk.summary())
    return 0


# --------------------------------------------------------------------------- send / status
def cmd_send(args: argparse.Namespace) -> int:
    """Envía un comando por el inbox y espera la respuesta (la lógica, compartida con la consola, en ``remote``)."""
    state_dir = _base_state_dir(args)
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        sent = send_command(state_dir, " ".join(args.text), prefix="cli")
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    if args.no_wait:
        print(f"Enviado [{sent.cmd_id}] a {sent.inbox}")
        return 0
    lines = wait_reply(sent, args.timeout, RealClock())
    if lines:
        print("\n".join(lines))
        return 0
    print(no_reply_message(sent, args.timeout, state_dir), file=sys.stderr)
    return 1


def health_check(state_dir: Path, now: float, max_age: float = HEALTH_MAX_AGE_S,
                 max_stale: float = STAGE_MAX_STALE_S) -> tuple[list[str], list[str]]:
    """(problemas, detalles) de la mesa según ``health.json`` / ``heartbeat``. Sin problemas = sana.

    Parada: no escribe ``health.json`` desde hace ``max_age`` s (hora real). Sin exchange: no lee la cuenta o
    los fills desde hace ``max_stale`` s (hora de la mesa, también simulada)."""
    problems: list[str] = []
    info: list[str] = []
    hp, hb = state_dir / HEALTH_FILE, state_dir / HEARTBEAT_FILE
    health = _read_json(hp)
    if health is None:
        if not hb.exists():
            return [f"no hay latido ni salud en {state_dir} (¿ha arrancado la mesa con este estado?)"], info
        age = now - hb.stat().st_mtime        # mesa de una versión sin health.json: sólo el latido
        if age > max_age:
            problems.append(f"último latido hace {age / 60:.0f} min: parece detenida")
        else:
            info.append(f"último latido hace {age:.0f} s")
        return problems, info
    age = now - hp.stat().st_mtime
    if age > max_age:
        problems.append(f"no completa ciclos desde hace {age / 60:.0f} min: parece detenida")
    try:
        ts = float(health.get("ts") or 0.0)
        started = float(health.get("started_ts") or ts)
    except (TypeError, ValueError):
        ts = started = 0.0
    last_ok = health.get("last_ok") if isinstance(health.get("last_ok"), dict) else {}
    errors = health.get("errors") if isinstance(health.get("errors"), dict) else {}
    for stage, what in (("cuenta", "la cuenta"), ("fills", "las ejecuciones (fills)")):
        try:
            ref = float(last_ok.get(stage) or started)
        except (TypeError, ValueError):
            ref = started
        since = ts - ref
        if since > max_stale:
            err = f" (último error: {errors[stage]})" if errors.get(stage) else ""
            never = "desde que arrancó, hace" if not last_ok.get(stage) else "desde hace"
            problems.append(f"no lee {what} {never} {since / 60:.0f} min{err}")
    failing = [str(x) for x in health.get("failing") or []]
    info.append(f"ciclo {health.get('cycle')} a las {health.get('time')} UTC (hora de la mesa), hace {age:.0f} s; "
                f"ciclos seguidos con fallos: {health.get('error_streak', 0)}"
                + (f"; etapas fallando: {', '.join(failing)}" if failing else ""))
    return problems, info


def cmd_status(args: argparse.Namespace) -> int:
    state_dir = _base_state_dir(args)
    journal = Journal(state_dir, RealClock())
    snap = journal.load_json("status", None)
    if not isinstance(snap, dict):
        print(f"No hay estado guardado en {state_dir} (¿la mesa ha corrido alguna vez con este directorio?)")
        return 1
    print(snap.get("report") or snap.get("line") or "(vacío)")
    print(f"\n(estado guardado a las {snap.get('time')} UTC, hora de la mesa)")
    problems, info = health_check(state_dir, RealClock().now())
    for line in info:
        print(f"Salud: {line}.")
    for line in problems:
        print(f"ATENCIÓN: la mesa {line}.")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Para healthchecks (Docker, systemd, cron): 0 si la mesa completa ciclos y ve el exchange, 1 si no."""
    state_dir = _base_state_dir(args)
    problems, info = health_check(state_dir, RealClock().now(), args.max_age, args.max_stale)
    if problems:
        print("MAL: la mesa " + "; ".join(problems) + ("" if not info else f" [{'; '.join(info)}]"))
        return 1
    print("OK: " + "; ".join(info))
    return 0


# --------------------------------------------------------------------------- backtest offline
def _backtest_market(settings: Settings, state_dir: Path) -> tuple[Any, Clock]:
    """Mercado y reloj para un backtest suelto. ``sim``: la historia termina en la hora simulada de ``state_dir`` (la
    de una mesa ``--fast`` que use ese estado) o en el ancla de la simulación. ``replay``: el CSV tal cual (reloj al
    final de su última vela). ``hyperliquid``: datos reales, hora real."""
    kind = _data_kind(settings.data_source)
    if kind == "sim":
        from .exchange.sim import DEFAULT_ANCHOR_TS, SimMarketData
        clock = SimClock(resume_clock_ts(state_dir) or DEFAULT_ANCHOR_TS)
        return SimMarketData(clock), clock
    if kind == "replay":
        path = settings.data_source.split(":", 1)[1] if ":" in settings.data_source else ""
        if not path:
            raise SystemExit("Falta la ruta del CSV: --csv datos/btc_1h.csv (o --data replay:<ruta>)")
        from .exchange.base import MarketDataError
        from .exchange.sim import ReplayMarketData
        clock = SimClock(0.0)
        try:
            market = ReplayMarketData(path, clock, offset=0.0)
        except MarketDataError as e:
            raise SystemExit(str(e)) from None
        clock.set(max(float(s.ts[-1]) + s.interval_s for s in market.series.values()))
        return market, clock
    if kind == "hyperliquid":
        from .exchange.hyperliquid_data import HyperliquidMarketData
        clock = RealClock()
        return HyperliquidMarketData.from_settings(settings, clock), clock
    raise SystemExit(f"Datos desconocidos: {settings.data_source!r} (usa hyperliquid, sim o --csv <fichero>)")


def cmd_backtest(args: argparse.Namespace) -> int:
    """Backtest walk-forward de una gráfica sin arrancar la mesa: imprime el resumen y guarda JSON/HTML/PNG en
    ``<estado>/reports`` (donde ``activa COIN tf`` lo encuentra si la mesa usa el mismo estado)."""
    from .commands import normalize_coin, strategy_key
    from .exchange.base import MarketDataError
    from .exchange.intervals import interval_seconds
    from .jobs import BacktestJob, fetch_bars, run_job
    from .optimizer import MIN_SEGMENT_BARS, ReportStore
    from .strategy import STRATEGIES

    overrides: dict[str, Any] = {"auto_enabled": False}
    if args.data:
        overrides["data_source"] = args.data
    if args.csv:
        overrides["data_source"] = f"replay:{args.csv}"
    _load_env(args.env_file)
    try:
        settings = load_settings(dotenv_path=args.env_file, **overrides)
    except ValueError as e:
        print(f"Configuración inválida: {e}", file=sys.stderr)
        return 2
    coin = normalize_coin(args.coin or "")
    if not coin:
        print(f"Moneda inválida: {args.coin!r}", file=sys.stderr)
        return 2
    try:
        interval_seconds(args.interval)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    strategy = None
    if args.strategy:
        strategy = strategy_key(args.strategy)          # "donchian" o "EMA-trend" también valen
        if strategy not in STRATEGIES:
            print(f"Estrategia desconocida: {args.strategy}. Disponibles: {', '.join(sorted(STRATEGIES))}",
                  file=sys.stderr)
            return 2
    days = float(args.days) if args.days else float(settings.bt_default_days)
    if days <= 0:
        print("--days debe ser > 0", file=sys.stderr)
        return 2
    kind = _data_kind(settings.data_source)
    if args.state_dir:
        state_dir = Path(args.state_dir)
    else:
        # Informes de datos simulados junto a la mesa simulada (state/fast), nunca junto a la mesa real.
        base = settings.state_path
        state_dir = base / "fast" if kind in ("sim", "replay") or args.fast else base
    state_dir.mkdir(parents=True, exist_ok=True)
    market, clock = _backtest_market(settings, state_dir)
    try:
        meta = market.meta(coin)
    except KeyError:
        print(f"No conozco la moneda {coin} en los datos {settings.data_source}.", file=sys.stderr)
        return 1
    try:
        bars, notes = fetch_bars(market, coin, args.interval, days, clock)
    except (MarketDataError, KeyError, ValueError) as e:
        print(f"No se pudieron obtener las velas de {coin} {args.interval}: {e}", file=sys.stderr)
        return 1
    need = 2 * MIN_SEGMENT_BARS
    if bars is None or len(bars) < need:
        print(f"{coin} {args.interval}: sólo hay {0 if bars is None else len(bars)} velas cerradas; el walk-forward "
              f"necesita al menos {need}.", file=sys.stderr)
        return 1
    what = strategy or f"{len(STRATEGIES)} estrategias"
    print(f"Backtest {coin} {args.interval}: {len(bars)} velas cerradas (~{days:g} días), {what}, datos "
          f"{settings.data_source}. Calculando...", flush=True)
    for note in notes:
        print(f"  Nota: {note}")
    tty = sys.stderr.isatty()
    last = [-1]

    def progress(done: int, total: int, text: str) -> None:
        pct = int(done * 100 / total) if total else 100
        if tty and pct != last[0]:
            last[0] = pct
            sys.stderr.write(f"\r  {pct:3d} % {text[:60]:<60}")
            sys.stderr.flush()

    job = BacktestJob(id="cli", coin=coin, interval=args.interval, days=days, strategy=strategy, bars=bars, meta=meta,
                      requested_ts=clock.now(), notes=notes, source="cli")
    try:
        out = run_job(job, settings, ReportStore(state_dir), clock, progress=progress, render=not args.no_render)
    except (ValueError, OSError) as e:
        print(f"\nEl backtest falló: {e}", file=sys.stderr)
        return 1
    finally:
        if tty:
            sys.stderr.write("\r" + " " * 70 + "\r")
    print()
    print(out.report.summary_text())
    print()
    for what, path in (("Informe HTML", out.html_path), ("Gráfico PNG", out.png_path), ("Datos JSON", out.report_path)):
        if path:
            print(f"{what}: {path}")
    if not args.no_render and out.png_path is None:
        print("(Sin PNG: instala matplotlib para generarlo.)")
    if out.report.winner:
        print(f"\nPara operarla en automático con una mesa que use este estado ({state_dir}): 'activa {coin} "
              f"{args.interval}'. Empieza en paper y obsérvala antes de pasar a testnet.")
    print(f"({out.elapsed_s:.1f} s)")
    return 0


# --------------------------------------------------------------------------- dashboard / console
def cmd_dashboard(args: argparse.Namespace) -> int:
    """Panel en vivo de sólo lectura (rich) sobre los ficheros de estado."""
    try:
        from .tui import run_dashboard
    except ImportError as e:
        print(f"El panel necesita 'rich' (pip install rich): {e}", file=sys.stderr)
        return 2
    state_dir = _base_state_dir(args)
    if not state_dir.exists():
        print(f"No existe {state_dir} (¿la mesa ha corrido alguna vez con este estado?).", file=sys.stderr)
        return 1
    return run_dashboard(state_dir, refresh_s=max(0.2, float(args.refresh)), once=args.once)


def cmd_console(args: argparse.Namespace) -> int:
    """Consola interactiva conectada por inbox/outbox a una mesa en marcha."""
    try:
        from .console import run_console
    except ImportError as e:
        print(f"La consola necesita 'rich' (pip install rich): {e}", file=sys.stderr)
        return 2
    state_dir = _base_state_dir(args)
    state_dir.mkdir(parents=True, exist_ok=True)
    problems, _ = health_check(state_dir, RealClock().now())
    return run_console(state_dir, timeout=args.timeout, warnings=[f"salud de la mesa: {p}" for p in problems])


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m desk",
                                description="Mesa de trading automatizada 24/7 (paper por defecto).")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="arranca la mesa (bucle 24/7)")
    r.add_argument("--mode", choices=["paper", "live"], help="paper (simulado, por defecto) o live (real)")
    r.add_argument("--data", help="hyperliquid | sim | replay:<ruta.csv>")
    r.add_argument("--symbols", help="monedas para la operativa automática, p. ej. BTC,ETH,SOL,HYPE")
    r.add_argument("--no-auto", action="store_true", help="arranca sin operativa automática (sólo comandos)")
    r.add_argument("--fast", action="store_true", help="reloj simulado (sólo con --data sim/replay): días en segundos")
    r.add_argument("--fresh", action="store_true", help="con --fast: empieza la simulación desde cero")
    r.add_argument("--cycles", type=int, help="sale tras N ciclos")
    r.add_argument("--loop-seconds", type=float, help="segundos entre ciclos (DESK_LOOP_SECONDS)")
    r.add_argument("--cmd", action="append", metavar="[N:]TEXTO",
                   help="comando a enviar antes del ciclo N (0 por defecto); repetible")
    r.add_argument("--state-dir", help="carpeta de estado (por defecto DESK_STATE_DIR o state/; state/fast con --fast)")
    r.add_argument("--no-input", action="store_true", help="no leer comandos de la consola")
    r.add_argument("--env-file", default=".env", help="fichero de configuración (por defecto .env)")
    r.add_argument("-v", "--verbose", action="store_true", help="log detallado en consola")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("send", help="envía un comando a la mesa en marcha y espera la respuesta")
    s.add_argument("text", nargs="+", help='p. ej. "long HYPE 2u zona"')
    s.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help=f"segundos de espera ({DEFAULT_TIMEOUT_S:g})")
    s.add_argument("--no-wait", action="store_true", help="no esperar la respuesta")
    s.add_argument("--state-dir")
    s.add_argument("--fast", action="store_true", help="usar el estado de la simulación --fast")
    s.add_argument("--env-file", default=".env")
    s.set_defaults(func=cmd_send)

    st = sub.add_parser("status", help="muestra el último estado guardado")
    st.add_argument("--state-dir")
    st.add_argument("--fast", action="store_true", help="estado de la simulación --fast")
    st.add_argument("--env-file", default=".env")
    st.set_defaults(func=cmd_status)

    h = sub.add_parser("health", help="salud de la mesa para healthchecks (código de salida 0 = sana)")
    h.add_argument("--state-dir")
    h.add_argument("--fast", action="store_true", help="salud de la simulación --fast")
    h.add_argument("--env-file", default=".env")
    h.add_argument("--max-age", type=float, default=HEALTH_MAX_AGE_S,
                   help=f"segundos sin ciclos para darla por parada ({HEALTH_MAX_AGE_S:g})")
    h.add_argument("--max-stale", type=float, default=STAGE_MAX_STALE_S,
                   help=f"segundos sin leer la cuenta o los fills para darla por ciega ({STAGE_MAX_STALE_S:g})")
    h.set_defaults(func=cmd_health)

    b = sub.add_parser("backtest", help="backtest walk-forward de una gráfica (sin arrancar la mesa)")
    b.add_argument("--coin", required=True, help="moneda, p. ej. SOL")
    b.add_argument("--interval", default="1h", help="temporalidad (1m 5m 15m 1h 4h 1d...; 1h por defecto)")
    b.add_argument("--days", type=float, help="días de historia (DESK_BT_DEFAULT_DAYS; máx. 5000 velas)")
    b.add_argument("--strategy", help="una sola estrategia (por defecto, todas)")
    b.add_argument("--data", help="hyperliquid | sim | replay:<ruta.csv>")
    b.add_argument("--csv", help="velas de un CSV (timestamp,open,high,low,close,volume) en vez de --data")
    b.add_argument("--state-dir", help="dónde guardar reports/ (por defecto state/, o state/fast con sim/replay)")
    b.add_argument("--fast", action="store_true", help="guardar en el estado de la simulación --fast")
    b.add_argument("--no-render", action="store_true", help="sólo JSON (sin HTML ni PNG)")
    b.add_argument("--env-file", default=".env")
    b.set_defaults(func=cmd_backtest)

    d = sub.add_parser("dashboard", help="panel en vivo (sólo lectura) del estado de la mesa")
    d.add_argument("--state-dir")
    d.add_argument("--fast", action="store_true", help="panel de la simulación --fast")
    d.add_argument("--env-file", default=".env")
    d.add_argument("--refresh", type=float, default=1.0, help="segundos entre refrescos (1)")
    d.add_argument("--once", action="store_true", help="pinta una vez y sale (también sin terminal)")
    d.set_defaults(func=cmd_dashboard)

    c = sub.add_parser("console", help="consola interactiva conectada a la mesa en marcha (inbox/outbox)")
    c.add_argument("--state-dir")
    c.add_argument("--fast", action="store_true", help="hablar con la simulación --fast")
    c.add_argument("--env-file", default=".env")
    c.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   help=f"segundos de espera de cada respuesta ({DEFAULT_TIMEOUT_S:g})")
    c.set_defaults(func=cmd_console)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())

"""Línea de comandos de la mesa.

    python -m desk run [--mode paper|live] [--data hyperliquid|sim|replay:x.csv] [--symbols BTC,ETH]
                       [--no-auto] [--fast] [--cycles N] [--loop-seconds S] [--cmd "N:texto"] [--state-dir DIR]
    python -m desk send "long HYPE"        (escribe en state/inbox.txt y espera la respuesta en outbox.log)
    python -m desk status                  (último estado guardado por la mesa)

``--fast`` usa un reloj simulado (``SimClock``) con datos ``sim`` o ``replay``: cada ciclo avanza
``loop_seconds`` de tiempo simulado sin esperar, así se simulan días en segundos. Su estado va a
``<state_dir>/fast`` (no se mezcla con el paper real) y continúa donde lo dejó la simulación anterior
salvo ``--fresh``.

Mientras corre, la consola acepta comandos (``long HYPE``, ``estado``, ``ayuda``...); ``salir`` detiene la mesa.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import shutil
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .clock import Clock, RealClock, SimClock
from .config import LIVE_CONFIRM_PHRASE, Settings, load_settings
from .journal import INBOX_FILE, OUTBOX_FILE, Journal, read_outbox

log = logging.getLogger("desk.cli")

PROFILE_FILE = "profile"     # profile.json: configuración con la que se creó el estado


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
        return LiveHyperliquidBroker.from_settings(market, settings, clock,
                                                   fills_since_ts=float(since) if since else None)
    from .exchange.paper import PaperBroker
    broker = PaperBroker.from_settings(market, settings, clock)
    if broker.load_error:
        log.error("%s", broker.load_error)
    return broker


def check_profile(journal: Journal, profile: dict[str, Any]) -> None:
    """El estado guardado (posiciones paper, brackets, riesgo) sólo vale para la misma configuración."""
    saved = journal.load_json(PROFILE_FILE, None)
    if isinstance(saved, dict):
        old = {k: saved.get(k) for k in profile}
        if old != profile:
            raise SystemExit(
                f"El estado en {journal.dir} es de otra configuración ({old}) y ahora se pide {profile}.\n"
                "Mezclarlos daría posiciones y límites de riesgo sin sentido. Usa otro --state-dir o borra esa carpeta.")
    data = dict(saved) if isinstance(saved, dict) else {}
    data.update(profile)
    journal.save_json(PROFILE_FILE, data)


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
        prof = Journal(state_dir, RealClock()).load_json(PROFILE_FILE, {}) or {}
        start = float(prof.get("clock_ts") or DEFAULT_ANCHOR_TS)
        clock = SimClock(start)
    else:
        clock = RealClock()
    setup_logging(state_dir, clock, args.verbose)
    journal = Journal(state_dir, clock)
    profile = {"mode": settings.mode, "data": settings.data_source, "clock": "sim" if args.fast else "real"}
    check_profile(journal, profile)
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
    state_dir = _base_state_dir(args)
    state_dir.mkdir(parents=True, exist_ok=True)
    text = " ".join(args.text).strip()
    if not text or "\n" in text:
        print("Comando vacío o con saltos de línea.", file=sys.stderr)
        return 2
    cmd_id = f"cli-{uuid.uuid4().hex[:8]}"
    outbox = state_dir / OUTBOX_FILE
    try:
        offset = outbox.stat().st_size
    except FileNotFoundError:
        offset = 0
    with open(state_dir / INBOX_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{cmd_id}] {text}\n")
    if args.no_wait:
        print(f"Enviado [{cmd_id}] a {state_dir / INBOX_FILE}")
        return 0
    clock = RealClock()
    deadline = clock.now() + args.timeout
    lines: list[str] = []
    got_at: Optional[float] = None
    while clock.now() < deadline:
        new, offset = read_outbox(outbox, cmd_id, offset)
        if new:
            lines.extend(new)
            got_at = got_at or clock.now()
        elif got_at is not None and clock.now() - got_at > 0.5:
            break
        clock.sleep(0.25)
    if lines:
        print("\n".join(lines))
        return 0
    print(f"La mesa no respondió en {args.timeout:g} s (¿está corriendo con este estado: {state_dir}?).\n"
          f"El comando [{cmd_id}] queda en {INBOX_FILE} y se ejecutará cuando la mesa lo lea; para anularlo, "
          "borra esa línea antes.", file=sys.stderr)
    return 1


def cmd_status(args: argparse.Namespace) -> int:
    state_dir = _base_state_dir(args)
    journal = Journal(state_dir, RealClock())
    snap = journal.load_json("status", None)
    if not isinstance(snap, dict):
        print(f"No hay estado guardado en {state_dir} (¿la mesa ha corrido alguna vez con este directorio?)")
        return 1
    print(snap.get("report") or snap.get("line") or "(vacío)")
    print(f"\n(estado guardado a las {snap.get('time')} UTC, hora de la mesa)")
    hb = state_dir / "heartbeat"
    if hb.exists():
        age = RealClock().now() - hb.stat().st_mtime
        if age > 600:
            print(f"ATENCIÓN: último latido de la mesa hace {age / 60:.0f} min: parece detenida.")
        else:
            print(f"Último latido hace {age:.0f} s.")
    return 0


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
    s.add_argument("--timeout", type=float, default=30.0, help="segundos de espera (30)")
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

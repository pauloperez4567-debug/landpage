"""Consola interactiva conectada a una mesa en marcha (``python -m desk console``).

Habla con la mesa como ``python -m desk send`` (``desk.remote``): cada comando va a ``inbox.txt`` con un id propio
(``con-xxxxxxxx``) y la respuesta se espera con ese id en ``outbox.log``. Así funciona con una mesa arrancada en
otro terminal, en tmux o en Docker (con el estado montado), y con varias consolas a la vez.

- Historial de readline en ``<state_dir>/.console_history`` (flechas arriba/abajo, Ctrl+R).
- Tab autocompleta verbos (español e inglés), monedas (las de ``status.json`` y del playbook) y temporalidades.
- Respuestas en color: errores y rechazos en rojo, lo que queda pendiente en amarillo, lo ejecutado en verde y los
  importes con signo (``+$12.30`` / ``-$5.00``) en verde / rojo.
- Un hilo sigue ``alerts.jsonl`` (alarmas; las silenciadas con ``mute`` no se imprimen) y los avisos ``desk`` de
  ``outbox.log`` y los imprime al llegar. Sólo los nuevos: abrir la consola no repite el historial. Un mismo texto
  que llega por los dos ficheros a la vez sale una sola vez.
- ``salir`` (o ``exit``, ``quit``, Ctrl+D) cierra **sólo esta consola**: la mesa sigue en marcha. (En la consola de
  ``python -m desk run``, en cambio, ``salir`` detiene la mesa.)
- Si la mesa no responde a tiempo, el comando sigue en el inbox y se ejecutará una sola vez: no hay que repetirlo.
  Su respuesta se imprime aquí cuando llegue. Ctrl+C mientras se espera deja de esperar (no anula el comando).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

from rich.console import Console, RenderableType
from rich.text import Text

from .clock import Clock, RealClock
from .commands import normalize
from .exchange.intervals import INTERVAL_SECONDS
from .journal import INBOX_FILE, OUTBOX_FILE, read_outbox
from .notify import normalize_level
from .remote import DEFAULT_TIMEOUT_S, SentCommand, no_reply_message, send_command, wait_reply
from .tui import (ALERTS_LOG, LEVEL_LABELS, LEVEL_STYLES, PLAYBOOK_FILE, STATUS_FILE, TEXT_STYLES, desk_notices,
                  normalize_playbook, normalize_status, notice_level, parse_jsonl, read_json)

log = logging.getLogger(__name__)

HISTORY_FILE = ".console_history"
HISTORY_LENGTH = 2000
PROMPT = "mesa> "
CMD_PREFIX = "con"                  # ids de la consola: [con-xxxxxxxx] (los de send son [cli-...])
FOLLOW_POLL_S = 0.5
DEDUPE_S = 120.0                    # el mismo texto por alerts.jsonl y outbox.log en este margen sale una vez
LATE_MAX_S = 24 * 3600.0            # respuestas tardías que se siguen esperando como mucho
LATE_KEPT = 50
MAX_READ = 4 * 1024 * 1024          # lo nuevo de un diario que se lee de una vez (si hay más, sólo lo último)
EXIT_WORDS = frozenset({"salir", "exit", "quit"})
INTERVALS: tuple[str, ...] = tuple(INTERVAL_SECONDS)

# --- vocabulario del autocompletado (los verbos que entiende commands.parse, en español e inglés) -------------
OPEN_VERBS = ("long", "short", "compra", "vende", "buy", "sell")
VERBS: tuple[str, ...] = OPEN_VERBS + (
    "cierra", "close", "flat", "cancela", "cancel", "stop", "sl", "tp",
    "estado", "status", "posiciones", "positions", "pausa", "pause", "reanuda", "resume", "auto", "kill", "panico",
    "reset", "si", "no", "confirmar", "confirm", "ayuda", "help",
    "apalancamiento", "leverage", "riesgo", "risk", "unidad", "unit",
    "backtest", "adopta", "adopt", "desactiva", "deactivate", "playbook", "informes", "reports",
    "alerta", "alert", "alarmas", "alerts", "borra", "delete", "silencia", "mute", "unmute",
    "salir", "exit",
)
# Verbos que piden moneda; de ellos, los que después piden temporalidad (gráfica = moneda + temporalidad).
COIN_VERBS = frozenset(OPEN_VERBS + ("cierra", "close", "cancela", "cancel", "stop", "sl", "tp", "backtest", "adopta",
                                     "adopt", "desactiva", "deactivate", "alerta", "alert"))
INTERVAL_VERBS = frozenset({"backtest", "adopta", "adopt", "desactiva", "deactivate", "alerta", "alert"})
OPEN_MODIFIERS = ("zona", "mercado", "ya", "limit", "sl", "tp", "zone", "market")
FOLLOW_WORDS: dict[str, tuple[str, ...]] = {
    "auto": ("on", "off"), "reset": ("halt", "params"), "cierra": ("todo",), "close": ("all",),
    "cancela": ("todo",), "cancel": ("all",), "borra": ("alarma", "alarmas"), "delete": ("alert", "alerts"),
}


def completions(prior: list[str], text: str, coins: Iterable[str], intervals: Iterable[str] = INTERVALS) -> list[str]:
    """Candidatos para completar ``text`` cuando ya se escribió ``prior`` (palabras anteriores de la línea).

    Primera palabra: verbos. Después, según el verbo: ``on``/``off``, ``todo``/``all``, monedas, temporalidades
    (tras la moneda en backtest/adopta/alerta) o modos de entrada (tras la moneda al abrir). Sin distinguir
    mayúsculas ni acentos."""
    coins = list(dict.fromkeys(str(c) for c in coins if c))
    intervals = list(intervals)
    prefix = normalize(text)
    if not prior:
        pool: list[str] = list(VERBS)
    else:
        verb = normalize(prior[0])
        rest = [normalize(p) for p in prior[1:]]
        has_coin = any(r in {normalize(c) for c in coins} for r in rest)
        pool = []
        if not rest:
            pool += FOLLOW_WORDS.get(verb, ())
        if verb in COIN_VERBS:
            if not has_coin:
                pool += coins
            elif verb in INTERVAL_VERBS:
                if not any(r in intervals for r in rest):
                    pool += intervals
            elif verb in OPEN_VERBS:
                pool += OPEN_MODIFIERS
        elif verb not in VERBS:                     # texto libre: lo más útil son monedas y temporalidades
            pool = coins + intervals
    return list(dict.fromkeys(w for w in pool if normalize(w).startswith(prefix)))


# --- colores de las respuestas -------------------------------------------------------------------------------
_BAD = ("no entendí", "no se pudo", "no se envió", "no se movió", "no se cambió", "rechaz", "error", "incierto",
        "no conozco", "no colocado", "atención", "fallo", "acción no soportada", "liquidaci")
_WARN = ("¿", "pendiente", "falta", "no hay", "descartado", "pausa", "cooldown", "aviso", "caduca")
_GOOD = ("ejecutad", "colocad", "cerrad", "cancelad", "enviad", "confirmado", "restaurad", "se aplica", "activad",
         "adoptad", "creada", "reactivad")


def reply_style(line: str) -> str:
    low = line.lower()
    if any(w in low for w in _BAD):
        return "red"
    if any(w in low for w in _WARN):
        return "yellow"
    if any(w in low for w in _GOOD):
        return "green"
    return ""


def color_amounts(t: Text) -> Text:
    """Importes con signo en verde / rojo (``+$1,234.50`` / ``-$5.00``)."""
    t.highlight_regex(r"\+\$[\d,]+(?:\.\d+)?", "green")
    t.highlight_regex(r"-\$[\d,]+(?:\.\d+)?", "red")
    return t


def format_reply(lines: Iterable[str]) -> Text:
    t = Text()
    for i, ln in enumerate(lines):
        if i:
            t.append("\n")
        t.append(ln, style=reply_style(ln))
    return color_amounts(t)


# --- seguimiento de alarmas y avisos ---------------------------------------------------------------------------
@dataclass
class Notice:
    source: str                     # "alarma" | "mesa" | "respuesta" (tardía, a un comando de esta consola)
    level: str
    text: str
    time: str = ""
    coin: str = ""
    cmd_id: str = ""


def format_notice(n: Notice) -> Text:
    stamp = n.time[11:19] if len(n.time) >= 19 else n.time
    t = Text()
    if stamp:
        t.append(f"{stamp} ", style="dim")
    if n.source == "respuesta":
        t.append(f"Respuesta tardía [{n.cmd_id}]:\n", style="bold")
        t.append_text(format_reply(n.text.splitlines()))
        return t
    lv = normalize_level(n.level)
    if n.source == "alarma":
        t.append(" ALARMA ", style="bold black on yellow" if lv != "critical" else LEVEL_STYLES["critical"])
    else:
        t.append(" MESA ", style="bold black on cyan" if lv == "info" else LEVEL_STYLES[lv])
    if lv != "info":
        t.append(f" {LEVEL_LABELS[lv]}", style=LEVEL_STYLES[lv])
    t.append(" ")
    t.append(n.text, style=TEXT_STYLES[lv])
    return color_amounts(t)


class Follower:
    """Lo nuevo de ``alerts.jsonl`` y de los avisos ``desk`` de ``outbox.log`` desde que se creó (o desde el
    principio con ``from_start``), más las respuestas tardías de los comandos que se dejaron de esperar.

    Sólo consume líneas completas; si un fichero se rota, se trunca o se recrea, vuelve a empezar por el
    principio del nuevo. ``poll()`` nunca lanza por un fichero ausente o ilegible."""

    def __init__(self, state_dir: Union[str, Path], clock: Optional[Clock] = None, *, from_start: bool = False):
        self.dir = Path(state_dir)
        self.clock: Clock = clock if clock is not None else RealClock()
        self._pos: dict[str, tuple[Optional[int], int]] = {}
        for name in (ALERTS_LOG, OUTBOX_FILE):
            self._pos[name] = (None, 0) if from_start else self._end(name)
        self._late: list[tuple[SentCommand, float]] = []
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def _end(self, name: str) -> tuple[Optional[int], int]:
        try:
            st = (self.dir / name).stat()
        except OSError:
            return (None, 0)
        return (st.st_ino, st.st_size)

    def watch(self, sent: SentCommand) -> None:
        """Imprimir la respuesta de ``sent`` cuando llegue (se dejó de esperar)."""
        with self._lock:
            self._late.append((sent, self.clock.now()))
            del self._late[:-LATE_KEPT]

    @property
    def watching(self) -> list[str]:
        return [s.cmd_id for s, _ in self._late]

    def _new_lines(self, name: str) -> list[str]:
        path = self.dir / name
        try:
            st = path.stat()
        except OSError:
            self._pos[name] = (None, 0)          # aún no existe (o se está rotando): se leerá entero al aparecer
            return []
        ino, off = self._pos[name]
        if (ino is not None and st.st_ino != ino) or st.st_size < off:
            off = 0                              # rotado, truncado o recreado
        skip_first = False
        if st.st_size - off > MAX_READ:
            off, skip_first = st.st_size - MAX_READ, True
        if st.st_size == off:
            self._pos[name] = (st.st_ino, off)
            return []
        try:
            with open(path, "rb") as f:
                f.seek(off)
                data = f.read(st.st_size - off)
        except OSError:
            return []
        end = data.rfind(b"\n")
        if end < 0:
            self._pos[name] = (st.st_ino, off)  # línea a medio escribir: se espera al salto
            return []
        self._pos[name] = (st.st_ino, off + end + 1)
        lines = data[:end + 1].decode("utf-8", "replace").splitlines()
        return lines[1:] if skip_first else lines

    def poll(self) -> list[Notice]:
        with self._lock:
            now = self.clock.now()
            out: list[Notice] = []
            for rec in parse_jsonl(self._new_lines(ALERTS_LOG)):
                if rec.get("muted"):
                    continue                     # silenciada con 'mute': se respeta
                text = str(rec.get("text") or rec.get("kind") or "")
                out.append(Notice("alarma", normalize_level(rec.get("level")), text, str(rec.get("time") or ""),
                                  coin=str(rec.get("coin") or "")))
            for n in desk_notices(self._new_lines(OUTBOX_FILE)):
                out.append(Notice("mesa", notice_level(n["text"]), n["text"], n["time"]))
            out = self._dedupe(out, now)
            keep: list[tuple[SentCommand, float]] = []
            for sent, since in self._late:
                lines, sent.offset = read_outbox(sent.outbox, sent.cmd_id, sent.offset)
                if lines:
                    out.append(Notice("respuesta", "info", "\n".join(lines), cmd_id=sent.cmd_id))
                elif now - since < LATE_MAX_S:
                    keep.append((sent, since))
            self._late = keep
            return out

    def _dedupe(self, notices: list[Notice], now: float) -> list[Notice]:
        out = []
        for n in notices:
            key = " ".join(n.text.split())
            last = self._seen.get(key)
            if last is not None and now - last < DEDUPE_S:
                continue
            self._seen[key] = now
            out.append(n)
        if len(self._seen) > 500:
            self._seen = {k: t for k, t in self._seen.items() if now - t < DEDUPE_S}
        return out


# --- historial -------------------------------------------------------------------------------------------------
class History:
    """Historial de readline en un fichero del estado (sobrevive entre sesiones de la consola)."""

    def __init__(self, path: Union[str, Path], length: int = HISTORY_LENGTH):
        self.path = Path(path)
        self.length = length

    def load(self, rl: Any) -> None:
        rl.set_history_length(self.length)
        try:
            rl.read_history_file(str(self.path))
        except OSError:                          # primera vez o ilegible: se empieza vacío
            pass

    def save(self, rl: Any) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rl.set_history_length(self.length)
            rl.write_history_file(str(self.path))
        except OSError as e:
            log.warning("consola: no se pudo guardar el historial en %s: %s", self.path, e)


# --- consola ---------------------------------------------------------------------------------------------------
class DeskConsole:
    def __init__(self, state_dir: Union[str, Path], *, clock: Optional[Clock] = None,
                 console: Optional[Console] = None, timeout: float = DEFAULT_TIMEOUT_S,
                 input_fn: Optional[Callable[[str], str]] = None, use_readline: bool = True, follow: bool = True,
                 poll_s: float = FOLLOW_POLL_S, warnings: Optional[list[str]] = None):
        self.dir = Path(state_dir)
        self.clock: Clock = clock if clock is not None else RealClock()
        self.console = console if console is not None else Console(highlight=False, markup=False, emoji=False,
                                                                   soft_wrap=True)
        self.timeout = float(timeout)
        self.input_fn = input_fn if input_fn is not None else input
        self.use_readline = use_readline
        self.follow = follow
        self.poll_s = poll_s
        self.warnings = list(warnings or [])
        self.follower = Follower(self.dir, self.clock)
        self.history = History(self.dir / HISTORY_FILE)
        self._lock = threading.RLock()
        self._at_prompt = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._rl: Any = None
        self._rl_saved: Optional[tuple[Any, str]] = None
        self._matches: list[str] = []
        self._coins: tuple[Any, list[str]] = (None, [])

    # ------------------------------------------------------------------ salida
    def _print(self, renderable: RenderableType, *, from_thread: bool = False) -> None:
        """Imprime sin mezclarse con otro hilo. Un aviso que llega mientras se escribe en el prompt se pinta encima
        y se vuelve a dibujar el prompt con lo que había escrito."""
        with self._lock:
            redraw = from_thread and self._at_prompt and self.console.is_terminal
            try:
                if redraw:
                    self.console.file.write("\r\x1b[2K")
                self.console.print(renderable)
                if redraw:
                    buf = self._rl.get_line_buffer() if self._rl is not None else ""
                    self.console.file.write(PROMPT + buf)
                    self.console.file.flush()
            except (OSError, ValueError) as e:   # terminal cerrado: no tumbar el hilo
                log.debug("consola: no se pudo escribir: %s", e)

    # ------------------------------------------------------------------ autocompletado
    def coins(self) -> list[str]:
        """Monedas para autocompletar: símbolos, posiciones y órdenes de ``status.json`` y gráficas del playbook.
        Se releen sólo si los ficheros cambian."""
        paths = (self.dir / STATUS_FILE, self.dir / PLAYBOOK_FILE)
        sig = []
        for p in paths:
            try:
                s = p.stat()
                sig.append((s.st_size, s.st_mtime_ns))
            except OSError:
                sig.append(None)
        if self._coins[0] == sig:
            return self._coins[1]
        st = normalize_status(read_json(paths[0])[0])
        found = list(st["symbols"]) + [x["coin"] for x in st["positions"] + st["orders"] + st["pending"]]
        found += [e["coin"] for e in normalize_playbook(read_json(paths[1])[0]) + (st["playbook"] or [])]
        coins = list(dict.fromkeys(c for c in found if c))
        self._coins = (sig, coins)
        return coins

    def complete(self, prior: list[str], text: str) -> list[str]:
        try:
            return completions(prior, text, self.coins())
        except Exception:  # noqa: BLE001 - autocompletar nunca rompe la consola
            log.debug("consola: fallo autocompletando", exc_info=True)
            return []

    def _rl_complete(self, text: str, state: int) -> Optional[str]:
        if state == 0:
            prior: list[str] = []
            try:
                buf = self._rl.get_line_buffer()
                prior = buf[:self._rl.get_begidx()].split()
            except Exception:  # noqa: BLE001
                pass
            self._matches = self.complete(prior, text)
        return self._matches[state] if state < len(self._matches) else None

    def _setup_readline(self) -> None:
        if not self.use_readline:
            return
        try:
            import readline
        except ImportError:                      # Windows sin pyreadline: sin historial ni Tab
            return
        self._rl = readline
        self._rl_saved = (readline.get_completer(), readline.get_completer_delims())
        readline.set_completer(self._rl_complete)
        readline.set_completer_delims(" \t\n")   # "xyz:GOLD" es una sola palabra
        if "libedit" in (readline.__doc__ or ""):
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
        self.history.load(readline)

    def _teardown_readline(self) -> None:
        if self._rl is None:
            return
        self.history.save(self._rl)
        if self._rl_saved is not None:
            self._rl.set_completer(self._rl_saved[0])
            self._rl.set_completer_delims(self._rl_saved[1])
        self._rl = None

    # ------------------------------------------------------------------ comandos
    def send(self, text: str) -> list[str]:
        """Envía ``text`` a la mesa, espera e imprime la respuesta. Devuelve sus renglones ([] si no llegó)."""
        try:
            sent = send_command(self.dir, text, prefix=CMD_PREFIX)
        except ValueError as e:
            self._print(Text(str(e), style="red"))
            return []
        except OSError as e:
            self._print(Text(f"No se pudo escribir en {self.dir / INBOX_FILE}: {e}", style="red"))
            return []
        interrupted = False
        try:
            lines = wait_reply(sent, self.timeout, self.clock)
        except KeyboardInterrupt:
            lines, interrupted = list(sent.lines), True
        if lines:
            self._print(format_reply(lines))
            return lines
        if interrupted:
            msg = (f"Sin esperar más: [{sent.cmd_id}] sigue en {INBOX_FILE} y se ejecutará una sola vez cuando la mesa "
                   "lo lea.")
        else:
            msg = no_reply_message(sent, self.timeout, self.dir)
        self._print(Text(msg + " Su respuesta saldrá aquí cuando llegue.", style="yellow"))
        self.follower.watch(sent)
        return []

    def handle(self, line: str) -> bool:
        """Procesa una línea. False = salir de la consola."""
        text = (line or "").strip()
        if not text:
            return True
        if normalize(text) in EXIT_WORDS:
            self._print(Text("Consola cerrada. La mesa sigue en marcha.", style="dim"))
            return False
        self.send(text)
        return True

    # ------------------------------------------------------------------ avisos en tiempo real
    def pump(self) -> int:
        """Una pasada del seguimiento de alarmas y avisos (la hace el hilo cada ``poll_s``)."""
        try:
            notices = self.follower.poll()
        except Exception:  # noqa: BLE001 - el hilo de avisos nunca muere
            log.exception("consola: fallo siguiendo alarmas y avisos")
            return 0
        for n in notices:
            self._print(format_notice(n), from_thread=True)
        return len(notices)

    def _follow_loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            self.pump()

    def start_follow(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._follow_loop, name="consola-avisos", daemon=True)
        self._thread.start()

    def stop_follow(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------ bucle
    def banner(self) -> Text:
        t = Text()
        t.append("Consola de la mesa", style="bold")
        t.append(f" · estado en {self.dir}\n", style="dim")
        t.append("Comandos en español o inglés ('ayuda' para ver ejemplos). Tab autocompleta; flechas: historial.\n")
        t.append("'salir' cierra esta consola; la mesa sigue en marcha.", style="dim")
        st = normalize_status(read_json(self.dir / STATUS_FILE)[0])
        if (st["mode"] or "").lower() == "live":
            t.append("\nMODO LIVE: los comandos operan con dinero real.", style="bold white on red")
        if st["line"]:
            t.append("\nÚltimo estado: ", style="bold")
            t.append_text(color_amounts(Text(st["line"])))
        elif not st["present"]:
            t.append(f"\nNo hay status.json en {self.dir}: ¿está la mesa en marcha con este estado?", style="yellow")
        for w in self.warnings:
            t.append(f"\nATENCIÓN: {w}", style="bold yellow")
        return t

    def run(self) -> int:
        self._setup_readline()
        self._print(self.banner())
        if self.follow:
            self.start_follow()
        try:
            while True:
                self._at_prompt = True
                try:
                    line = self.input_fn(PROMPT)
                except EOFError:
                    self._print(Text("Consola cerrada. La mesa sigue en marcha.", style="dim"))
                    break
                except KeyboardInterrupt:
                    self._print(Text("\n(Ctrl+C) Para cerrar la consola escribe 'salir' o pulsa Ctrl+D.", style="dim"))
                    continue
                except OSError:                  # sin entrada estándar
                    break
                finally:
                    self._at_prompt = False
                if not self.handle(line):
                    break
        finally:
            self.stop_follow()
            self._teardown_readline()
        return 0


def run_console(state_dir: Union[str, Path], *, timeout: float = DEFAULT_TIMEOUT_S,
                warnings: Optional[list[str]] = None, clock: Optional[Clock] = None,
                console: Optional[Console] = None) -> int:
    return DeskConsole(state_dir, timeout=timeout, warnings=warnings, clock=clock, console=console).run()

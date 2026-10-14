"""Cliente de una mesa en marcha: envía comandos por ``inbox.txt`` y espera la respuesta en ``outbox.log``.

Lo comparten ``python -m desk send`` y la consola interactiva (``python -m desk console``). Protocolo:

- Cada comando es una línea ``[<id>] texto`` añadida a ``inbox.txt`` con bloqueo y con su salto final (si la
  última línea del fichero no lo tenía, se antepone uno: el comando nunca se pega a una línea escrita a mano).
- La mesa ejecuta cada ``[id]`` una sola vez y responde en ``outbox.log`` con renglones
  ``<fecha hora> [<id>] texto`` (ver ``journal.read_outbox``).
- El outbox se lee desde el tamaño que tenía *antes* de enviar: una respuesta rápida nunca se pierde.
- Si la mesa no responde a tiempo el comando sigue en el inbox y se ejecutará (una vez) cuando la lea: no hay
  que borrarlo ni repetirlo (``no_reply_message`` lo explica al usuario).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from .clock import Clock, RealClock
from .journal import INBOX_FILE, OUTBOX_FILE, read_outbox

DEFAULT_TIMEOUT_S = 30.0
SETTLE_S = 0.5      # tras el primer renglón de la respuesta, margen para que lleguen los siguientes
POLL_S = 0.25


@dataclass
class SentCommand:
    """Comando ya escrito en el inbox. ``offset`` avanza a medida que se lee su respuesta del outbox."""

    cmd_id: str
    text: str
    inbox: Path
    outbox: Path
    offset: int = 0
    lines: list[str] = field(default_factory=list)


def new_command_id(prefix: str = "cli") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def validate_command_text(text: str) -> str:
    """Texto listo para el inbox (una sola línea, sin espacios sobrantes). ``ValueError`` si no vale."""
    text = (text or "").strip()
    if not text or "\n" in text or "\r" in text:
        raise ValueError("Comando vacío o con saltos de línea.")
    return text


def append_inbox_line(path: Union[str, Path], line: str) -> None:
    """Añade una línea al inbox con su salto final. Si el fichero no acaba en salto de línea (alguien escribió a
    mano sin él) antepone uno: si no, el comando se pegaría a esa línea. Con bloqueo para escritores concurrentes."""
    data = (line.rstrip("\n") + "\n").encode("utf-8")
    with open(path, "a+b") as f:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):     # Windows o sistema de ficheros sin bloqueos
            pass
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                data = b"\n" + data
        f.write(data)
        f.flush()


def send_command(state_dir: Union[str, Path], text: str, *, prefix: str = "cli",
                 cmd_id: Optional[str] = None) -> SentCommand:
    """Escribe ``[id] texto`` en ``<state_dir>/inbox.txt`` y devuelve lo necesario para esperar la respuesta.

    ``ValueError`` si el texto está vacío o tiene saltos de línea; ``OSError`` si no se puede escribir."""
    text = validate_command_text(text)
    d = Path(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    cid = cmd_id or new_command_id(prefix)
    outbox = d / OUTBOX_FILE
    try:
        offset = outbox.stat().st_size
    except FileNotFoundError:
        offset = 0
    append_inbox_line(d / INBOX_FILE, f"[{cid}] {text}")
    return SentCommand(cmd_id=cid, text=text, inbox=d / INBOX_FILE, outbox=outbox, offset=offset)


def wait_reply(sent: SentCommand, timeout: float = DEFAULT_TIMEOUT_S, clock: Optional[Clock] = None, *,
               settle_s: float = SETTLE_S, poll_s: float = POLL_S,
               should_stop: Optional[Callable[[], bool]] = None) -> list[str]:
    """Renglones de la respuesta a ``sent`` (lista vacía si no llega en ``timeout`` s o si ``should_stop()``).

    En cuanto llega el primer renglón se espera ``settle_s`` a que dejen de llegar más (respuestas largas)."""
    clock = clock or RealClock()
    deadline = clock.now() + timeout
    got_at: Optional[float] = None
    start = len(sent.lines)
    while clock.now() < deadline:
        new, sent.offset = read_outbox(sent.outbox, sent.cmd_id, sent.offset)
        if new:
            sent.lines.extend(new)
            got_at = got_at or clock.now()
        elif got_at is not None and clock.now() - got_at > settle_s:
            break
        if should_stop is not None and should_stop():
            break
        clock.sleep(poll_s)
    return sent.lines[start:]


def no_reply_message(sent: SentCommand, timeout: float, state_dir: Union[str, Path]) -> str:
    """Qué pasa con un comando sin respuesta: sigue en el inbox y se ejecutará una sola vez."""
    return (f"La mesa no respondió en {timeout:g} s (¿está corriendo con este estado: {state_dir}?).\n"
            f"El comando [{sent.cmd_id}] queda en {INBOX_FILE} y se ejecutará (una sola vez) cuando la mesa lo lea. No "
            "borres ni edites líneas del inbox: si ya no lo quieres, envía después el comando contrario (p. ej. "
            "'cancela HYPE' o 'cierra HYPE').")

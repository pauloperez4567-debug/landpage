"""Canales de aviso: consola, fichero, Telegram y webhook (Discord / Slack).

``Notifier(channels).send(text, level, kind, coin, image)`` reparte cada aviso entre los canales cuyo
nivel mínimo lo admite (niveles: "info" < "warn" < "critical").

- Los canales locales (consola, fichero) se escriben al momento: son rápidos y no dependen de la red.
- Los de red (Telegram, webhook) van a una cola que vacía un hilo daemon: la mesa nunca se bloquea
  esperando a un servidor HTTP. Si la cola se llena se descarta el aviso más antiguo (y se avisa en el log).
- Un fallo de un canal se registra y se traga: avisar nunca tumba el bucle 24/7.
- Nunca se registran el token de Telegram ni la URL completa del webhook (llevan el secreto en la ruta):
  los mensajes de error pasan por ``redact`` y los canales sólo se identifican por el host.

Telegram:

- ``sendMessage`` en texto plano (sin ``parse_mode``: un ``_`` o un ``*`` de un aviso no rompe el envío);
  los mensajes de más de 4000 caracteres se trocean (por líneas si se puede).
- Con ``image`` (PNG en bytes) se usa ``sendPhoto`` (multipart); si el texto no cabe en el pie de foto
  (1024 caracteres) la foto va sin pie y el texto detrás, troceado.
- Los avisos "info" llegan sin sonido (``disable_notification``); "warn" y "critical", con sonido.
- Un 429 (límite de envíos) se reintenta una vez respetando ``retry_after`` (con tope).

Webhook: ``POST`` JSON con ``content`` (Discord) y ``text`` (Slack), troceado a 1900 caracteres porque
Discord rechaza mensajes de más de 2000. Las imágenes no se envían por webhook.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union
from urllib.parse import quote, urlsplit

import requests

from .clock import Clock, RealClock

log = logging.getLogger(__name__)

LEVELS: tuple[str, ...] = ("info", "warn", "critical")
_LEVEL_RANK = {name: i for i, name in enumerate(LEVELS)}
_LEVEL_ALIASES = {
    "warning": "warn", "aviso": "warn", "error": "critical", "crit": "critical",
    "critico": "critical", "crítico": "critical", "debug": "info",
}
LEVEL_LABELS = {"info": "", "warn": "[AVISO] ", "critical": "[CRÍTICO] "}

TELEGRAM_API = "https://api.telegram.org"
TELEGRAM_CHUNK = 4000        # Telegram admite 4096 caracteres por mensaje: margen para el prefijo
TELEGRAM_CAPTION = 1024      # pie de foto máximo
WEBHOOK_CHUNK = 1900         # Discord rechaza "content" de más de 2000 caracteres
MAX_RETRY_AFTER_S = 10.0     # tope de espera ante un 429 (el hilo de envío no se queda colgado)
QUEUE_SIZE = 500


def normalize_level(level: Any, default: str = "info") -> str:
    """Nivel canónico ("info" | "warn" | "critical"). Admite alias ("warning", "error"...).

    Un valor desconocido devuelve ``default`` (un nivel mal escrito en la configuración no debe
    dejar a un canal mudo ni tumbar la mesa)."""
    key = str(level or "").strip().lower()
    key = _LEVEL_ALIASES.get(key, key)
    return key if key in _LEVEL_RANK else default


def level_rank(level: Any) -> int:
    return _LEVEL_RANK[normalize_level(level)]


def redact(text: Any, secrets: Iterable[str]) -> str:
    """``text`` con cada secreto (y su forma codificada en URL) sustituido por ``***``."""
    out = str(text)
    for sec in secrets:
        if not sec or len(sec) < 4:
            continue
        for form in {sec, quote(sec, safe=""), quote(sec, safe="/")}:
            out = out.replace(form, "***")
    return out


def safe_url(url: str) -> str:
    """Sólo esquema y host (``https://discord.com/…``): lo único de una URL de webhook que se puede mostrar."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<url inválida>"
    if not parts.netloc:
        return "<url inválida>"
    return f"{parts.scheme or 'https'}://{parts.hostname or parts.netloc}/…"


def split_message(text: str, limit: int = TELEGRAM_CHUNK) -> list[str]:
    """Trocea ``text`` en fragmentos de como mucho ``limit`` caracteres, cortando por líneas siempre que se
    puede (una línea más larga que el límite se corta a pelo). Nunca devuelve fragmentos vacíos."""
    text = str(text)
    limit = max(int(limit), 1)
    if len(text) <= limit:
        return [text] if text.strip() else []
    chunks: list[str] = []
    cur: Optional[str] = None
    for line in text.split("\n"):
        while len(line) > limit:
            if cur is not None:
                chunks.append(cur)
                cur = None
            chunks.append(line[:limit])
            line = line[limit:]
        if cur is None:
            cur = line
        elif len(cur) + 1 + len(line) <= limit:
            cur += "\n" + line
        else:
            chunks.append(cur)
            cur = line
    if cur is not None:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


@dataclass
class Notification:
    text: str
    level: str = "info"
    kind: str = ""
    coin: Optional[str] = None
    image: Optional[bytes] = None
    ts: float = 0.0

    def line(self) -> str:
        """Texto con la etiqueta del nivel (``[AVISO] ...``)."""
        return LEVEL_LABELS.get(self.level, "") + self.text


class ChannelError(RuntimeError):
    """Fallo al entregar un aviso. El mensaje ya viene sin secretos."""


# =========================================================================== canales
class Channel:
    """Interfaz de un canal. ``blocking=True`` = usa la red: el ``Notifier`` lo envía desde su hilo."""

    name: str = "canal"
    blocking: bool = False

    def __init__(self, min_level: str = "info"):
        self.min_level = normalize_level(min_level)

    def accepts(self, level: str) -> bool:
        return level_rank(level) >= level_rank(self.min_level)

    def send(self, n: Notification) -> None:  # puede lanzar: el Notifier lo captura
        raise NotImplementedError

    def secrets(self) -> list[str]:
        """Cadenas que nunca deben aparecer en el log (token, URL...)."""
        return []

    def close(self) -> None:
        return None


class ConsoleChannel(Channel):
    """Avisos por pantalla. ``out``: None (``sys.stdout``), un flujo con ``write`` o una función ``out(texto)``.

    Con un flujo y ``rich`` instalado los niveles salen en color (rich desactiva el color solo si el flujo no
    es un terminal). Con ``bell`` los avisos "warn" y "critical" emiten un pitido (``\\a``); a una función se le
    pasa el ``\\a`` delante del texto.
    """

    name = "consola"

    _STYLES = {"info": "", "warn": "bold yellow", "critical": "bold white on red"}

    def __init__(self, out: Any = None, bell: bool = True, min_level: str = "info", *,
                 color: Optional[bool] = None):
        super().__init__(min_level)
        self.bell = bool(bell)
        self._lock = threading.Lock()
        self._func: Optional[Callable[[str], Any]] = None
        self._stream: Any = None
        if out is not None and not hasattr(out, "write") and callable(out):
            self._func = out
        else:
            self._stream = out
        self._console: Any = None
        if self._func is None and color is not False:
            try:
                from rich.console import Console
                # file=None: rich resuelve sys.stdout en cada escritura (sigue a redirecciones posteriores)
                self._console = Console(file=self._stream, highlight=False, soft_wrap=True,
                                        markup=False, emoji=False)
            except Exception:  # noqa: BLE001 - rich es opcional
                self._console = None

    def _ring(self, level: str) -> bool:
        return self.bell and level in ("warn", "critical")

    def send(self, n: Notification) -> None:
        line = n.line()
        with self._lock:
            if self._func is not None:
                self._func(("\a" if self._ring(n.level) else "") + line)
                return
            stream = self._stream or sys.stdout
            if self._ring(n.level):
                stream.write("\a")
            if self._console is not None:
                from rich.text import Text
                # Text (y no markup): un "[id]" del aviso no se interpreta como etiqueta de estilo.
                self._console.print(Text(line, style=self._STYLES.get(n.level, "")))
            else:
                stream.write(line + "\n")
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass


class FileChannel(Channel):
    """Un aviso JSON por línea: ``{"ts","time","level","kind","coin","text","image"}``."""

    name = "fichero"

    def __init__(self, path: Union[str, Path], min_level: str = "info"):
        super().__init__(min_level)
        self.path = Path(path)
        self._lock = threading.Lock()

    def send(self, n: Notification) -> None:
        rec = {"ts": n.ts, "time": datetime.fromtimestamp(n.ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
               "level": n.level, "kind": n.kind, "coin": n.coin, "text": n.text, "image": bool(n.image)}
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)


class _HttpChannel(Channel):
    """Base de los canales de red: sesión inyectable, timeout y un reintento ante 429."""

    blocking = True

    def __init__(self, min_level: str, session: Optional[requests.Session], timeout: float,
                 sleep: Optional[Callable[[float], None]]):
        super().__init__(min_level)
        self.session = session if session is not None else requests.Session()
        self.timeout = float(timeout)
        self._sleep = sleep if sleep is not None else time.sleep

    def _post(self, url: str, what: str, **kwargs: Any) -> requests.Response:
        """POST con un reintento ante 429. Lanza ``ChannelError`` (sin secretos) si no hay respuesta 2xx."""
        for attempt in range(2):
            try:
                resp = self.session.post(url, timeout=self.timeout, **kwargs)
            except requests.RequestException as e:
                # El mensaje de requests incluye la URL (con el token o el secreto del webhook): se limpia y
                # se corta la cadena de excepciones para que ningún traceback la arrastre al log.
                raise ChannelError(f"{what}: {type(e).__name__}: {redact(e, self.secrets())}") from None
            status = int(getattr(resp, "status_code", 0) or 0)
            if status == 429 and attempt == 0:
                self._sleep(min(_retry_after(resp), MAX_RETRY_AFTER_S))
                continue
            if 200 <= status < 300:
                return resp
            raise ChannelError(f"{what}: HTTP {status} {redact(_describe_error(resp), self.secrets())}".rstrip())
        raise ChannelError(f"{what}: HTTP 429 (límite de envíos)")


class TelegramChannel(_HttpChannel):
    """Avisos a uno o varios chats de Telegram (``sendMessage`` / ``sendPhoto``)."""

    name = "telegram"

    def __init__(self, token: str, chat_ids: Iterable[Any], min_level: str = "info",
                 session: Optional[requests.Session] = None, *, timeout: float = 10.0,
                 api_base: str = TELEGRAM_API, sleep: Optional[Callable[[float], None]] = None):
        super().__init__(min_level, session, timeout, sleep)
        self._token = str(token or "").strip()
        self.chat_ids = [str(c).strip() for c in (chat_ids or []) if str(c).strip()]
        self.api_base = api_base.rstrip("/")

    def __repr__(self) -> str:  # nunca el token
        return f"TelegramChannel(chats={len(self.chat_ids)}, min_level={self.min_level!r})"

    def secrets(self) -> list[str]:
        return [self._token]

    def _url(self, method: str) -> str:
        return f"{self.api_base}/bot{self._token}/{method}"

    def _call(self, method: str, **kwargs: Any) -> None:
        resp = self._post(self._url(method), f"Telegram {method}", **kwargs)
        try:
            body = resp.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and body.get("ok") is False:
            raise ChannelError(f"Telegram {method}: {redact(body.get('description') or 'ok=false', self.secrets())}")

    def send(self, n: Notification) -> None:
        if not self._token or not self.chat_ids:
            raise ChannelError("Telegram sin token o sin chats configurados")
        text = n.line()
        silent = n.level == "info"
        errors = []
        for chat in self.chat_ids:
            rest = text
            if n.image:
                caption = text if len(text) <= TELEGRAM_CAPTION else ""
                data = {"chat_id": chat, "disable_notification": "true" if silent else "false"}
                if caption:
                    data["caption"] = caption
                try:
                    self._call("sendPhoto", data=data, files={"photo": ("aviso.png", n.image, "image/png")})
                    rest = "" if caption else text
                except ChannelError as e:   # sin imagen, pero el texto llega igual
                    errors.append(f"chat {chat}: foto: {e}")
            try:
                for chunk in split_message(rest, TELEGRAM_CHUNK):
                    self._call("sendMessage", json={"chat_id": chat, "text": chunk,
                                                    "disable_web_page_preview": True,
                                                    "disable_notification": silent})
            except ChannelError as e:
                errors.append(f"chat {chat}: {e}")
        if errors:
            raise ChannelError("; ".join(errors))


class WebhookChannel(_HttpChannel):
    """Avisos a un webhook entrante de Discord o Slack: JSON ``{"content": ..., "text": ...}``."""

    name = "webhook"

    def __init__(self, url: str, min_level: str = "warn", session: Optional[requests.Session] = None, *,
                 timeout: float = 10.0, sleep: Optional[Callable[[float], None]] = None):
        super().__init__(min_level, session, timeout, sleep)
        self._url = str(url or "").strip()

    def __repr__(self) -> str:  # nunca la URL completa
        return f"WebhookChannel({safe_url(self._url)}, min_level={self.min_level!r})"

    def secrets(self) -> list[str]:
        out = [self._url]
        try:
            parts = urlsplit(self._url)
            out += [parts.path, parts.path.lstrip("/"), parts.query]
        except ValueError:
            pass
        return [s for s in out if s and s != "/"]

    def send(self, n: Notification) -> None:
        if not self._url:
            raise ChannelError("webhook sin URL configurada")
        what = f"webhook {safe_url(self._url)}"
        for chunk in split_message(n.line(), WEBHOOK_CHUNK):
            self._post(self._url, what, json={"content": chunk, "text": chunk})


def _retry_after(resp: Any) -> float:
    """Segundos de espera de un 429: ``parameters.retry_after`` (Telegram), ``retry_after`` (Discord) o la
    cabecera ``Retry-After``. 1 s si no viene."""
    try:
        body = resp.json()
        if isinstance(body, dict):
            v = (body.get("parameters") or {}).get("retry_after", body.get("retry_after"))
            if v is not None and math.isfinite(float(v)):
                return max(float(v), 0.0)
    except (ValueError, TypeError, AttributeError):
        pass
    try:
        return max(float((getattr(resp, "headers", None) or {}).get("Retry-After")), 0.0)
    except (TypeError, ValueError):
        return 1.0


def _describe_error(resp: Any) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict):
            return str(body.get("description") or body.get("message") or "")[:200]
    except (ValueError, TypeError, AttributeError):
        pass
    try:
        return str(resp.text or "")[:200]
    except Exception:  # noqa: BLE001
        return ""


# =========================================================================== notifier
_STOP = object()


class Notifier:
    """Reparte avisos entre canales. ``send`` nunca lanza ni espera a la red.

    - ``threaded=False`` envía también los canales de red en línea (útil en scripts de una sola vez).
    - ``flush(timeout)`` espera a que la cola se vacíe; ``close()`` la vacía y para el hilo.
    - ``sent``, ``failed`` y ``dropped`` cuentan entregas por canal; ``last_error`` guarda el último error
      (ya sin secretos) para el informe de estado.
    """

    def __init__(self, channels: Optional[Iterable[Channel]] = None, *, clock: Optional[Clock] = None,
                 threaded: bool = True, queue_size: int = QUEUE_SIZE):
        self.channels: list[Channel] = list(channels or [])
        self.clock: Clock = clock if clock is not None else RealClock()
        self.threaded = threaded
        self._q: queue.Queue = queue.Queue(maxsize=max(int(queue_size), 1))
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.sent: dict[str, int] = {}
        self.failed: dict[str, int] = {}
        self.dropped = 0
        self.last_error = ""

    def add(self, channel: Channel) -> None:
        self.channels.append(channel)

    # ------------------------------------------------------------------ envío
    def send(self, text: str, level: str = "info", kind: str = "", coin: Optional[str] = None,
             image: Optional[bytes] = None) -> None:
        try:
            n = Notification(text=str(text), level=normalize_level(level), kind=str(kind or ""),
                             coin=coin, image=image if isinstance(image, (bytes, bytearray)) else None,
                             ts=float(self.clock.now()))
            for ch in list(self.channels):
                try:
                    if not ch.accepts(n.level):
                        continue
                    if ch.blocking and self.threaded:
                        self._enqueue(ch, n)
                    else:
                        self._deliver(ch, n)
                except Exception as e:  # noqa: BLE001 - un canal roto no frena a los demás
                    self._record_error(ch, e)
        except Exception as e:  # noqa: BLE001 - avisar nunca tumba la mesa
            log.error("avisos: error inesperado al enviar: %s", type(e).__name__)

    def _deliver(self, ch: Channel, n: Notification) -> None:
        try:
            ch.send(n)
        except Exception as e:  # noqa: BLE001
            self._record_error(ch, e)
        else:
            self.sent[ch.name] = self.sent.get(ch.name, 0) + 1

    def _record_error(self, ch: Channel, e: BaseException) -> None:
        try:
            secrets = ch.secrets()
        except Exception:  # noqa: BLE001
            secrets = []
        msg = redact(e, secrets) if isinstance(e, ChannelError) else f"{type(e).__name__}: {redact(e, secrets)}"
        self.failed[ch.name] = self.failed.get(ch.name, 0) + 1
        self.last_error = f"{ch.name}: {msg}"
        # Sin traceback: la cadena de excepciones de requests podría arrastrar la URL con el secreto.
        log.warning("avisos: fallo del canal %s: %s", ch.name, msg)

    def _enqueue(self, ch: Channel, n: Notification) -> None:
        self._ensure_thread()
        while True:
            try:
                self._q.put_nowait((ch, n))
                return
            except queue.Full:
                try:  # se descarta el más antiguo: el aviso nuevo es el más relevante
                    self._q.get_nowait()
                    self._q.task_done()
                except queue.Empty:
                    continue
                self.dropped += 1
                if self.dropped == 1 or self.dropped % 100 == 0:
                    log.warning("avisos: cola llena, %d aviso(s) descartados (¿red caída?)", self.dropped)

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._worker, name="desk-notify", daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is _STOP:
                    return
                ch, n = item
                self._deliver(ch, n)
            except Exception as e:  # noqa: BLE001 - el hilo de envío no muere nunca
                log.error("avisos: error en el hilo de envío: %s", type(e).__name__)
            finally:
                self._q.task_done()

    # ------------------------------------------------------------------ control
    def pending(self) -> int:
        return self._q.unfinished_tasks

    def flush(self, timeout: float = 5.0) -> bool:
        """Espera (como mucho ``timeout`` s reales) a que se entregue lo encolado. True si la cola quedó vacía."""
        deadline = time.monotonic() + max(timeout, 0.0)
        cond = self._q.all_tasks_done
        with cond:
            while self._q.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                cond.wait(remaining)
        return True

    def close(self, timeout: float = 5.0) -> None:
        """Vacía la cola (hasta ``timeout``) y para el hilo. Un ``send`` posterior lo vuelve a arrancar."""
        self.flush(timeout)
        with self._lock:
            thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            try:
                self._q.put(_STOP, timeout=1.0)
            except queue.Full:
                pass
            thread.join(timeout=max(timeout, 0.1))
        for ch in self.channels:
            try:
                ch.close()
            except Exception:  # noqa: BLE001
                pass

    def describe(self) -> str:
        """Canales activos con su nivel mínimo, sin secretos (para el arranque y el estado)."""
        if not self.channels:
            return "sin canales de aviso"
        return ", ".join(f"{ch.name} (>= {ch.min_level})" for ch in self.channels)


def build_notifier(settings: Any, out: Any = None, *, console: bool = True, clock: Optional[Clock] = None,
                   session: Optional[requests.Session] = None, file_path: Optional[Union[str, Path]] = None,
                   threaded: bool = True) -> Notifier:
    """``Notifier`` según la configuración: consola (si ``console``), fichero (si ``file_path``), Telegram si hay
    ``telegram_bot_token`` y ``telegram_chat_ids``, webhook si hay ``webhook_url``."""
    channels: list[Channel] = []
    if console:
        channels.append(ConsoleChannel(out, bell=bool(getattr(settings, "alert_bell", True))))
    if file_path is not None:
        channels.append(FileChannel(file_path))
    token = str(getattr(settings, "telegram_bot_token", "") or "").strip()
    chats = list(getattr(settings, "telegram_chat_ids", None) or [])
    if token and chats:
        channels.append(TelegramChannel(token, chats, _setting_level(settings, "alert_min_level_telegram", "info"),
                                        session=session))
    elif token or chats:
        log.warning("avisos: Telegram necesita TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID; canal desactivado")
    url = str(getattr(settings, "webhook_url", "") or "").strip()
    if url:
        channels.append(WebhookChannel(url, _setting_level(settings, "alert_min_level_webhook", "warn"),
                                       session=session))
    return Notifier(channels, clock=clock, threaded=threaded)


def _setting_level(settings: Any, attr: str, default: str) -> str:
    raw = getattr(settings, attr, default)
    level = normalize_level(raw, default="")
    if not level:
        log.warning("avisos: nivel %r inválido en %s; se usa %r", raw, attr, default)
        return default
    return level


__all__ = [
    "LEVELS", "Notification", "Channel", "ChannelError", "ConsoleChannel", "FileChannel", "TelegramChannel",
    "WebhookChannel", "Notifier", "build_notifier", "normalize_level", "level_rank", "redact", "safe_url",
    "split_message",
]

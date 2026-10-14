"""Bot de Telegram: el operador da órdenes a la mesa desde el móvil, con los mismos textos que en la consola.

- Long polling de ``getUpdates`` (``timeout`` 25 s) en un hilo daemon; ``start()`` / ``stop()``.
- Sólo se atienden mensajes de los chats de ``chat_ids``. El resto se ignora **sin responder** (no se confirma
  ni que el bot existe) y queda en el log con el id del chat, para poder autorizarlo si era legítimo.
- ``/start`` y ``/help`` -> ayuda (presentación + la ayuda de la mesa). Un ``/comando`` se traduce a texto:
  ``/estado`` -> "estado", ``/cierra_todo`` -> "cierra todo", ``/long HYPE`` -> "long HYPE" (el ``@NombreBot``
  de los grupos se quita). Cada texto va a ``submit(text, source="telegram")`` y la respuesta vuelve al chat
  (troceada si pasa de 4000 caracteres), como respuesta al mensaje original.
- Como mucho una vez: el ``offset`` de updates vive en memoria y se *confirma* a Telegram (``getUpdates`` con el
  offset nuevo) **antes** de ejecutar el lote. Si la mesa se cae a mitad de un "long HYPE", al volver Telegram
  no lo reenvía: nunca se ejecuta dos veces.
- Mensajes antiguos: uno con más de ``max_age_s`` (5 min por defecto) se descarta y se avisa al chat (p. ej.
  escrito mientras la mesa estaba caída o sin red: ejecutar tarde un "long" a otro precio es peor que pedir
  que se reenvíe). La edad se mide con el ``Clock`` inyectado: con un ``SimClock`` desfasado de la hora real
  conviene ``max_age_s=None``.
- Errores (red, 5xx, 409 por otro proceso con el mismo bot, 429) -> espera exponencial 1, 2, 4... hasta 60 s
  (respetando ``retry_after``); token inválido (401/404) -> 5 min. El bucle nunca muere.
- Nunca registra el token: los errores pasan por ``redact`` (las excepciones de ``requests`` llevan la URL).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Iterable, Optional

import requests

from .clock import Clock, RealClock
from .notify import TELEGRAM_API, TELEGRAM_CHUNK, redact, split_message

log = logging.getLogger(__name__)

POLL_TIMEOUT_S = 25            # long polling: Telegram retiene la petición hasta 25 s si no hay mensajes
MAX_BACKOFF_S = 60.0
AUTH_BACKOFF_S = 300.0         # token inválido o revocado: no martillear la API
MAX_AGE_S = 300.0              # mensajes más antiguos no se ejecutan
MIN_EMPTY_POLL_S = 1.0         # una respuesta vacía inmediata (proxy raro) no debe girar en vacío
HELP_WORDS = frozenset({"start", "help"})

BOT_HELP = (
    "Mesa de trading por Telegram.\n"
    "Escribe los comandos igual que en la consola, por ejemplo:\n"
    "  estado | long HYPE 2u zona sl 1.5% tp 4% | cierra HYPE | cierra todo | pausa | reanuda | kill\n"
    "También con barra: /estado, /pausa, /reanuda, /cierra_todo (el _ es un espacio).\n"
    "Las órdenes interpretadas por IA piden confirmación: responde 'si' o 'no'.\n"
    "Los mensajes de hace más de unos minutos no se ejecutan (reenvíalos)."
)


def translate_command(text: str) -> str:
    """``/comando@Bot args`` -> ``comando args`` (``_`` del comando -> espacio). Un texto sin barra no cambia."""
    text = str(text or "").strip()
    if not text.startswith("/"):
        return text
    parts = text[1:].split(None, 1)
    if not parts:
        return ""
    head = parts[0].split("@", 1)[0].replace("_", " ").strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    return f"{head} {rest}".strip()


class _ApiError(RuntimeError):
    def __init__(self, message: str, status: int = 0, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class TelegramBot:
    def __init__(self, token: str, chat_ids: Iterable[Any], submit: Callable[..., Any],
                 clock: Optional[Clock] = None, session: Optional[requests.Session] = None, *,
                 poll_timeout: int = POLL_TIMEOUT_S, max_age_s: Optional[float] = MAX_AGE_S,
                 api_base: str = TELEGRAM_API):
        self._token = str(token or "").strip()
        self.chat_ids: set[str] = {str(c).strip() for c in (chat_ids or []) if str(c).strip()}
        self.submit = submit
        self.clock: Clock = clock if clock is not None else RealClock()
        self.session = session if session is not None else requests.Session()
        self.poll_timeout = max(int(poll_timeout), 0)
        self.max_age_s = max_age_s
        self.api_base = api_base.rstrip("/")
        self.offset: Optional[int] = None     # siguiente update_id a pedir (en memoria)
        self.errors = 0                       # errores seguidos de getUpdates
        self.backoff_s = 0.0                  # espera antes del siguiente getUpdates
        self.last_error = ""
        self.last_ok_ts: Optional[float] = None
        self.handled = 0                      # mensajes atendidos
        self.ignored = 0                      # mensajes de chats no autorizados
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __repr__(self) -> str:  # nunca el token
        return f"TelegramBot(chats={len(self.chat_ids)}, running={self.running})"

    # ------------------------------------------------------------------ ciclo de vida
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Arranca el hilo de escucha. False si falta el token (el bot queda desactivado, la mesa sigue)."""
        if not self._token:
            log.error("Telegram: falta TELEGRAM_BOT_TOKEN; bot desactivado")
            return False
        if not self.chat_ids:
            log.warning("Telegram: sin TELEGRAM_CHAT_ID autorizados; todos los mensajes se ignorarán")
        if self.running:
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="desk-telegram", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        """Pide parar y espera al hilo (como mucho ``timeout`` s). Lo que llegue después no se ejecuta."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(timeout, 0.0))

    def _run(self) -> None:
        log.info("Telegram: bot escuchando (%d chat(s) autorizados)", len(self.chat_ids))
        while not self._stop.is_set():
            started = time.monotonic()
            n = -1
            try:
                n = self.poll_once()
            except Exception as e:  # noqa: BLE001 - poll_once no debería lanzar; el hilo nunca muere
                self._on_error(f"{type(e).__name__}: {redact(e, [self._token])}")
            wait = self.backoff_s
            if n == 0 and not wait and time.monotonic() - started < MIN_EMPTY_POLL_S:
                wait = MIN_EMPTY_POLL_S
            if wait > 0:
                self._stop.wait(wait)
        log.info("Telegram: bot detenido")

    # ------------------------------------------------------------------ API
    def _api(self, method: str, payload: dict[str, Any], http_timeout: float = 10.0) -> Any:
        """Llama a la Bot API y devuelve ``result``. Lanza ``_ApiError`` (sin el token) si algo falla."""
        url = f"{self.api_base}/bot{self._token}/{method}"
        try:
            resp = self.session.post(url, json=payload, timeout=http_timeout)
        except requests.RequestException as e:
            raise _ApiError(f"{method}: {type(e).__name__}: {redact(e, [self._token])}") from None
        status = int(getattr(resp, "status_code", 0) or 0)
        try:
            body = resp.json()
        except (ValueError, AttributeError):
            body = None
        if not isinstance(body, dict):
            raise _ApiError(f"{method}: HTTP {status} respuesta no JSON", status)
        if 200 <= status < 300 and body.get("ok") is True:
            return body.get("result")
        params = body.get("parameters") if isinstance(body.get("parameters"), dict) else {}
        retry_after = params.get("retry_after")
        desc = redact(str(body.get("description") or "")[:200], [self._token])
        raise _ApiError(f"{method}: HTTP {status} {desc}".rstrip(), status,
                        float(retry_after) if isinstance(retry_after, (int, float)) else None)

    def _on_error(self, msg: str, status: int = 0, retry_after: Optional[float] = None) -> None:
        self.errors += 1
        self.last_error = msg
        delay = min(2.0 ** (self.errors - 1), MAX_BACKOFF_S)
        if retry_after is not None:
            delay = max(delay, min(float(retry_after), MAX_BACKOFF_S * 5))
        hint = ""
        if status in (401, 404):
            delay = AUTH_BACKOFF_S
            hint = " (¿token inválido o revocado?)"
        elif status == 409:
            hint = " (¿otro proceso usa el mismo bot o hay un webhook configurado?)"
        self.backoff_s = delay
        if self.errors <= 3 or self.errors % 20 == 0:
            log.warning("Telegram: error %d seguido: %s%s; reintento en %.0f s", self.errors, msg, hint, delay)
        else:
            log.debug("Telegram: error %d seguido: %s", self.errors, msg)

    # ------------------------------------------------------------------ polling
    def poll_once(self, timeout: Optional[int] = None) -> int:
        """Un ``getUpdates`` y el tratamiento de lo recibido. Devuelve cuántos updates se trataron; nunca lanza
        (un error deja ``backoff_s`` > 0 para el bucle)."""
        wait = self.poll_timeout if timeout is None else max(int(timeout), 0)
        payload: dict[str, Any] = {"timeout": wait, "allowed_updates": ["message"]}
        if self.offset is not None:
            payload["offset"] = self.offset
        try:
            result = self._api("getUpdates", payload, http_timeout=wait + 10.0)
        except _ApiError as e:
            self._on_error(str(e), e.status, e.retry_after)
            return 0
        self.errors = 0
        self.backoff_s = 0.0
        self.last_ok_ts = self.clock.now()
        if self._stop.is_set():          # parada pedida durante la espera: no se ejecuta (ni confirma) nada
            return 0
        batch = []
        for upd in result if isinstance(result, list) else []:
            uid = upd.get("update_id") if isinstance(upd, dict) else None
            if not isinstance(uid, int) or isinstance(uid, bool):
                continue
            if self.offset is not None and uid < self.offset:     # ya tratado: nunca se repite
                continue
            batch.append(upd)
        if not batch:
            return 0
        batch.sort(key=lambda u: u["update_id"])
        self.offset = batch[-1]["update_id"] + 1
        self._confirm()
        for upd in batch:
            try:
                self._handle(upd)
            except Exception as e:  # noqa: BLE001 - un mensaje raro no para el bot
                log.error("Telegram: error tratando el update %s: %s", upd.get("update_id"),
                          redact(f"{type(e).__name__}: {e}", [self._token]))
        return len(batch)

    def _confirm(self) -> None:
        """Confirma a Telegram los updates < ``offset`` antes de ejecutarlos (como mucho una vez). Lo que devuelva
        esta llamada no se trata: llegará en el siguiente ``getUpdates``."""
        try:
            self._api("getUpdates", {"offset": self.offset, "timeout": 0, "limit": 1,
                                     "allowed_updates": ["message"]}, http_timeout=10.0)
        except _ApiError as e:
            log.debug("Telegram: no se pudo confirmar el offset %s: %s", self.offset, e)

    def _handle(self, upd: dict[str, Any]) -> None:
        msg = upd.get("message")
        if not isinstance(msg, dict):          # ediciones, botones, canales...: no se ejecutan
            return
        chat = msg.get("chat") if isinstance(msg.get("chat"), dict) else {}
        chat_id = str(chat.get("id", "")).strip()
        if not chat_id or chat_id not in self.chat_ids:
            self.ignored += 1
            sender = msg.get("from") if isinstance(msg.get("from"), dict) else {}
            who = sender.get("username") or sender.get("first_name") or "?"
            log.warning("Telegram: mensaje de un chat no autorizado ignorado (chat %s, %s): %r", chat_id or "?",
                        str(who)[:40], str(msg.get("text") or "")[:60])
            return
        msg_id = msg.get("message_id")
        text = msg.get("text")
        if not isinstance(text, str) or not text.strip():
            self.reply(chat_id, "Sólo entiendo mensajes de texto. Escribe /help para ver ejemplos.", msg_id)
            return
        date = msg.get("date")
        if self.max_age_s is not None and isinstance(date, (int, float)):
            age = float(self.clock.now()) - float(date)
            if age > self.max_age_s:
                log.warning("Telegram: mensaje de hace %.0f s ignorado por antiguo: %r", age, text[:80])
                self.reply(chat_id, f"Mensaje de hace {age / 60:.0f} min NO ejecutado por antiguo: "
                                    f"«{text.strip()[:80]}». Reenvíalo si sigue vigente.", msg_id)
                return
        command = translate_command(text)
        self.handled += 1
        if command.lower() in HELP_WORDS:
            self.reply(chat_id, self._help(), msg_id)
            return
        if not command:
            return
        log.info("Telegram: comando de %s: %s", chat_id, command[:200])
        try:
            answer = self.submit(command, source="telegram")
        except Exception as e:  # noqa: BLE001 - un fallo de la mesa se cuenta al usuario
            log.exception("Telegram: error ejecutando %r", command[:80])
            answer = f"Error procesando el comando ({type(e).__name__}). Revisa el log de la mesa."
        self.reply(chat_id, str(answer) if answer not in (None, "") else "(sin respuesta)", msg_id)

    def _help(self) -> str:
        try:
            desk_help = self.submit("ayuda", source="telegram")
        except Exception as e:  # noqa: BLE001
            log.warning("Telegram: no se pudo obtener la ayuda de la mesa (%s)", type(e).__name__)
            desk_help = ""
        return BOT_HELP + (f"\n\n{desk_help}" if desk_help else "")

    def reply(self, chat_id: Any, text: str, reply_to: Optional[int] = None) -> bool:
        """Envía ``text`` al chat (troceado). True si todo se envió; los errores se registran y se tragan."""
        ok = True
        for i, chunk in enumerate(split_message(str(text), TELEGRAM_CHUNK)):
            payload: dict[str, Any] = {"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True}
            if i == 0 and isinstance(reply_to, int):
                payload["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}
            try:
                self._api("sendMessage", payload)
            except _ApiError as e:
                ok = False
                log.warning("Telegram: no se pudo responder al chat %s: %s", chat_id, e)
                break
        return ok


__all__ = ["TelegramBot", "translate_command", "BOT_HELP"]

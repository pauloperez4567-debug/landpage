"""Bot de Telegram con una sesión HTTP falsa: chats autorizados, comandos, offset, antigüedad y errores."""

from __future__ import annotations

import logging
import threading

import pytest
import requests

from desk.clock import SimClock
from desk.telegram_bot import (AUTH_BACKOFF_S, BOT_HELP, MAX_BACKOFF_S, TelegramBot, command_target,
                               translate_command)

TOKEN = "987654321:AAF-token_super_secreto-QWE"
OWNER = 555
NOW = 1_800_000_000.0


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = {"ok": True, "result": []} if body is None else body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeTelegram:
    """Sesión falsa: ``getUpdates`` sirve lotes de ``batches`` (lista de updates, FakeResponse o excepción);
    ``sendMessage`` se registra. ``log`` guarda el orden de todo lo que pasa (también las llamadas a submit)."""

    def __init__(self, batches=None):
        self.batches = list(batches or [])
        self.calls: list[tuple[str, dict]] = []
        self.log: list[str] = []
        self.send_error = None

    def post(self, url, json=None, timeout=None, **kw):
        method = url.rsplit("/", 1)[-1]
        assert url.startswith(f"https://api.telegram.org/bot{TOKEN}/")
        self.calls.append((method, json))
        if method == "getUpdates":
            self.log.append(f"getUpdates offset={json.get('offset')} timeout={json.get('timeout')}")
            if json.get("timeout") == 0:               # confirmación del offset
                return FakeResponse()
            if not self.batches:
                return FakeResponse()
            item = self.batches.pop(0)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, FakeResponse):
                return item
            return FakeResponse(200, {"ok": True, "result": item})
        if method == "getMe":
            return FakeResponse(200, {"ok": True, "result": {"id": 1, "is_bot": True, "username": "MesaBot"}})
        if method == "sendMessage":
            self.log.append(f"send {json['chat_id']}")
            if self.send_error is not None:
                raise self.send_error
            return FakeResponse(200, {"ok": True, "result": {"message_id": 1}})
        raise AssertionError(method)

    def sent(self):
        return [p for m, p in self.calls if m == "sendMessage"]

    def polls(self):
        return [p for m, p in self.calls if m == "getUpdates" and p.get("timeout") != 0]


def upd(uid, text="estado", chat=OWNER, date=NOW - 5, mid=None, **extra):
    msg = {"message_id": mid or uid * 10, "date": date, "chat": {"id": chat, "type": "private"},
           "from": {"id": chat, "username": "alguien"}}
    if text is not None:
        msg["text"] = text
    msg.update(extra)
    return {"update_id": uid, "message": msg}


class Desk:
    def __init__(self, log=None, fail=False):
        self.calls: list[tuple[str, str]] = []
        self.log = log
        self.fail = fail

    def submit(self, text, source="consola"):
        self.calls.append((text, source))
        if self.log is not None:
            self.log.append(f"submit {text}")
        if self.fail:
            raise RuntimeError("la mesa explotó")
        if text == "ayuda":
            return "AYUDA DE LA MESA"
        return f"OK: {text}"


def make_bot(batches=None, chats=(str(OWNER),), fail=False, **kw):
    session = FakeTelegram(batches)
    desk = Desk(session.log, fail=fail)
    bot = TelegramBot(TOKEN, chats, desk.submit, clock=SimClock(NOW), session=session, **kw)
    return bot, session, desk


@pytest.fixture
def debug_logs(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog


# --------------------------------------------------------------------------- autorización
def test_unauthorized_chat_is_ignored_silently(debug_logs):
    bot, session, desk = make_bot([[upd(1, "long HYPE", chat=999)]])
    assert bot.poll_once() == 1
    assert desk.calls == [] and session.sent() == []
    assert bot.ignored == 1 and bot.handled == 0
    assert "no autorizado" in debug_logs.text and "999" in debug_logs.text


def test_authorized_text_goes_to_submit_and_reply_threads_the_message():
    bot, session, desk = make_bot([[upd(1, "estado", mid=77)]])
    bot.poll_once()
    assert desk.calls == [("estado", "telegram")]
    (reply,) = session.sent()
    assert reply["chat_id"] == str(OWNER) and reply["text"] == "OK: estado"
    assert reply["reply_parameters"] == {"message_id": 77, "allow_sending_without_reply": True}


def test_chat_ids_accept_ints_and_strings():
    group = upd(1, chat=-100123)
    group["message"]["from"] = {"id": OWNER}
    bot, _, desk = make_bot([[group, upd(2, chat=OWNER)]], chats=[-100123, f" {OWNER} "], user_ids=[OWNER])
    bot.poll_once()
    assert len(desk.calls) == 2


def group_msg(uid, text, user, chat=-1001234567890, **extra):
    u = upd(uid, text, chat=chat, **extra)
    u["message"]["chat"]["type"] = "supergroup"
    u["message"]["from"] = {"id": user, "username": f"u{user}"}
    return u


def test_group_without_user_ids_accepts_orders_from_nobody(debug_logs):
    bot, session, desk = make_bot([[group_msg(1, "kill", 42), group_msg(2, "estado", OWNER)]],
                                  chats=["-1001234567890"])
    assert bot.group_without_users()
    bot.poll_once()
    assert desk.calls == [] and session.sent() == [] and bot.ignored == 2
    assert "TELEGRAM_USER_IDS" in debug_logs.text


def test_group_with_user_ids_only_listed_people_give_orders():
    anon = group_msg(4, "reset halt", 1087968824)                       # admin anónimo (GroupAnonymousBot)
    channel = group_msg(5, "long BTC 10u a mercado", OWNER, sender_chat={"id": -100777, "type": "channel"})
    bot, _, desk = make_bot([[group_msg(1, "kill", 666), group_msg(2, "estado", OWNER),
                              group_msg(3, "long BTC 10u a mercado", 777), anon, channel]],
                            chats=["-1001234567890"], user_ids=[str(OWNER)])
    bot.poll_once()
    assert desk.calls == [("estado", "telegram")] and bot.ignored == 4


def test_user_ids_also_apply_to_private_chats():
    bot, _, desk = make_bot([[upd(1, "estado")]], user_ids=["12345"])
    bot.poll_once()
    assert desk.calls == [] and bot.ignored == 1


def test_commands_addressed_to_another_bot_are_ignored():
    assert command_target("/cierra_todo@OtroBot") == "OtroBot" and command_target("/estado") is None
    assert command_target("long HYPE") is None
    bot, _, desk = make_bot([[upd(1, "/cierra_todo@OtroBot"), upd(2, "/estado@mesabot"), upd(3, "/pausa")]])
    bot.poll_once()
    assert [t for t, _ in desk.calls] == ["estado", "pausa"]


def test_without_own_name_addressed_commands_are_not_executed():
    class NoMe(FakeTelegram):
        def post(self, url, json=None, timeout=None, **kw):
            if url.endswith("/getMe"):
                raise requests.ConnectionError("sin red")
            return super().post(url, json=json, timeout=timeout, **kw)

    session = NoMe([[upd(1, "/kill@MesaBot"), upd(2, "kill")]])
    desk = Desk()
    bot = TelegramBot(TOKEN, [OWNER], desk.submit, clock=SimClock(NOW), session=session)
    bot.poll_once()
    assert desk.calls == [("kill", "telegram")]


def test_label_prefixes_replies():
    bot, session, _ = make_bot([[upd(1, "estado")]], label="SIM fast")
    bot.poll_once()
    assert session.sent()[0]["text"] == "[SIM fast] OK: estado"


# --------------------------------------------------------------------------- comandos
@pytest.mark.parametrize("raw, text", [
    ("/estado", "estado"),
    ("/cierra_todo", "cierra todo"),
    ("/long@MesaBot HYPE 2u zona", "long HYPE 2u zona"),
    ("  long HYPE  ", "long HYPE"),
    ("/", ""),
    ("/kill   ya", "kill ya"),
])
def test_translate_command(raw, text):
    assert translate_command(raw) == text


def test_slash_commands_are_translated_before_submit():
    bot, _, desk = make_bot([[upd(1, "/estado"), upd(2, "/cierra_todo@MesaBot"), upd(3, "/sl_HYPE 84.5")]])
    bot.poll_once()
    assert [t for t, _ in desk.calls] == ["estado", "cierra todo", "sl HYPE 84.5"]


@pytest.mark.parametrize("cmd", ["/start", "/help", "/help@MesaBot"])
def test_start_and_help_reply_with_help(cmd):
    bot, session, desk = make_bot([[upd(1, cmd)]])
    bot.poll_once()
    (reply,) = session.sent()
    assert reply["text"].startswith(BOT_HELP) and "AYUDA DE LA MESA" in reply["text"]
    assert desk.calls == [("ayuda", "telegram")]


def test_help_still_works_if_desk_fails():
    bot, session, _ = make_bot([[upd(1, "/start")]], fail=True)
    bot.poll_once()
    assert session.sent()[0]["text"] == BOT_HELP


def test_long_answers_are_chunked():
    bot, session, _ = make_bot([[upd(1, "estado")]])
    bot.submit = lambda text, source: "\n".join(f"posición {i} " + "x" * 80 for i in range(120))
    bot.poll_once()
    sent = session.sent()
    assert len(sent) == 3 and all(len(p["text"]) <= 4000 for p in sent)
    assert "reply_parameters" in sent[0] and "reply_parameters" not in sent[1]


def test_empty_answer_and_non_text_and_edits():
    bot, session, desk = make_bot([[upd(1, "estado"), upd(2, text=None, photo=[{"file_id": "x"}]),
                                    {"update_id": 3, "edited_message": upd(3, "long HYPE")["message"]}]])
    bot.submit = lambda text, source: ""
    bot.poll_once()
    texts = [p["text"] for p in session.sent()]
    assert texts == ["(sin respuesta)", "Sólo entiendo mensajes de texto. Escribe /help para ver ejemplos."]
    assert bot.offset == 4                           # la edición se consume pero no se ejecuta


def test_submit_exception_is_reported_and_bot_continues(debug_logs):
    bot, session, _ = make_bot([[upd(1, "long HYPE"), upd(2, "estado")]], fail=True)
    assert bot.poll_once() == 2
    texts = [p["text"] for p in session.sent()]
    assert len(texts) == 2 and all("Error procesando el comando (RuntimeError)" in t for t in texts)


def test_old_messages_are_not_executed():
    bot, session, desk = make_bot([[upd(1, "long HYPE", date=NOW - 3600), upd(2, "estado", date=NOW - 30)]])
    bot.poll_once()
    assert desk.calls == [("estado", "telegram")]
    first = session.sent()[0]["text"]
    assert "NO ejecutado por antiguo" in first and "60 min" in first and "long HYPE" in first


def test_max_age_none_disables_age_check():
    bot, _, desk = make_bot([[upd(1, "estado", date=NOW - 10 * 86400)]], max_age_s=None)
    bot.poll_once()
    assert desk.calls == [("estado", "telegram")]


# --------------------------------------------------------------------------- offset
def test_offset_advances_and_is_confirmed_before_executing():
    bot, session, desk = make_bot([[upd(5, "uno"), upd(6, "dos")], [upd(6, "dos"), upd(7, "tres")]])
    assert bot.poll_once() == 2
    assert bot.offset == 7
    # la confirmación (getUpdates con el offset nuevo y timeout 0) va antes de ejecutar nada
    assert session.log[:3] == ["getUpdates offset=None timeout=25", "getUpdates offset=7 timeout=0", "submit uno"]
    assert bot.poll_once() == 1                     # el 6 reenviado no se repite
    assert [t for t, _ in desk.calls] == ["uno", "dos", "tres"]
    assert session.polls()[1]["offset"] == 7
    assert bot.offset == 8
    bot.poll_once()
    assert session.polls()[2]["offset"] == 8


def test_out_of_order_and_malformed_updates():
    bot, _, desk = make_bot([[upd(9, "b"), "basura", {"update_id": "x"}, upd(8, "a"), {"sin": "id"}]])
    assert bot.poll_once() == 2
    assert [t for t, _ in desk.calls] == ["a", "b"] and bot.offset == 10


def test_confirm_failure_defers_execution_until_telegram_confirms(debug_logs):
    """Sin confirmar el offset no se ejecuta: tras un reinicio Telegram reenviaría el update (dos ejecuciones)."""
    class FailingConfirm(FakeTelegram):
        fail = True

        def post(self, url, json=None, timeout=None, **kw):
            if url.endswith("getUpdates") and json.get("timeout") == 0 and self.fail:
                raise requests.ConnectionError(f"url: /bot{TOKEN}/getUpdates")
            return super().post(url, json=json, timeout=timeout, **kw)

    session = FailingConfirm([[upd(1, "long HYPE a mercado")], requests.ConnectionError("red caída"), []])
    desk = Desk()
    bot = TelegramBot(TOKEN, [OWNER], desk.submit, clock=SimClock(NOW), session=session)
    assert bot.poll_once() == 0 and desk.calls == [] and bot.offset == 2 and bot.backoff_s >= 1
    assert bot.poll_once() == 0 and desk.calls == []           # el getUpdates siguiente falla: sigue pendiente
    session.fail = False
    assert bot.poll_once() == 1                                # getUpdates(offset=2) bien = confirmado: se ejecuta
    assert desk.calls == [("long HYPE a mercado", "telegram")]
    assert bot.poll_once() == 0 and len(desk.calls) == 1        # una sola vez
    assert TOKEN not in debug_logs.text


def test_restart_after_failed_confirmation_executes_once():
    """Proceso 1: la confirmación falla y se cae antes del siguiente getUpdates. Proceso 2: Telegram lo reenvía y
    se ejecuta una vez (en total, una)."""
    class Server(FakeTelegram):
        def __init__(self):
            super().__init__()
            self.queue = [upd(1, "long HYPE a mercado")]
            self.fail_confirm = True

        def post(self, url, json=None, timeout=None, **kw):
            method = url.rsplit("/", 1)[-1]
            if method == "getUpdates":
                off = json.get("offset")
                if json.get("timeout") == 0 and self.fail_confirm:
                    raise requests.ConnectionError("corte")
                if off is not None:
                    self.queue = [u for u in self.queue if u["update_id"] >= off]
                return FakeResponse(200, {"ok": True, "result": [] if json.get("limit") == 1 else list(self.queue)})
            return super().post(url, json=json, timeout=timeout, **kw)

    srv = Server()
    calls = []
    TelegramBot(TOKEN, [OWNER], lambda t, source: calls.append(t) or "ok", clock=SimClock(NOW), session=srv).poll_once()
    assert calls == []                                          # proceso 1 cae sin ejecutar
    srv.fail_confirm = False
    TelegramBot(TOKEN, [OWNER], lambda t, source: calls.append(t) or "ok", clock=SimClock(NOW), session=srv).poll_once()
    assert calls == ["long HYPE a mercado"]


def test_conflict_409_is_counted_and_reset():
    busy = FakeResponse(409, {"ok": False, "description": "Conflict: terminated by other getUpdates request"})
    bot, _, _ = make_bot([busy, busy, busy, []])
    for _ in range(3):
        bot.poll_once()
    assert bot.conflicts == 3
    bot.poll_once()
    assert bot.conflicts == 0


# --------------------------------------------------------------------------- errores y backoff
def test_network_errors_back_off_exponentially_without_leaking_token(debug_logs):
    err = requests.ConnectionError(f"HTTPSConnectionPool(host='api.telegram.org'): url: /bot{TOKEN}/getUpdates")
    bot, _, desk = make_bot([err] * 8 + [[upd(1, "estado")]])
    delays = []
    for _ in range(8):
        assert bot.poll_once() == 0
        delays.append(bot.backoff_s)
    assert delays == [1, 2, 4, 8, 16, 32, 60, 60]
    assert bot.errors == 8 and "ConnectionError" in bot.last_error
    assert bot.poll_once() == 1                     # al volver la red: sin espera y se procesa
    assert bot.backoff_s == 0 and bot.errors == 0 and bot.last_ok_ts == NOW
    assert desk.calls == [("estado", "telegram")]
    assert TOKEN not in debug_logs.text and TOKEN not in bot.last_error and TOKEN not in repr(bot)


def test_http_errors_rate_limit_and_auth(debug_logs):
    bot, _, _ = make_bot([FakeResponse(429, {"ok": False, "description": "Too Many Requests",
                                             "parameters": {"retry_after": 30}}),
                          FakeResponse(401, {"ok": False, "description": "Unauthorized"}),
                          FakeResponse(409, {"ok": False, "description": "Conflict: terminated by other getUpdates"}),
                          FakeResponse(502, ValueError("html")),
                          FakeResponse(200, {"ok": True, "result": "no es lista"})])
    bot.poll_once()
    assert bot.backoff_s == 30
    bot.poll_once()
    assert bot.backoff_s == AUTH_BACKOFF_S and "token inválido" in debug_logs.text
    bot.poll_once()
    assert "otro proceso" in debug_logs.text and bot.backoff_s <= MAX_BACKOFF_S
    bot.poll_once()
    assert "no JSON" in bot.last_error
    assert bot.poll_once() == 0 and bot.backoff_s == 0       # respuesta rara pero ok: nada que hacer


def test_reply_failure_is_swallowed(debug_logs):
    bot, session, desk = make_bot([[upd(1, "estado"), upd(2, "pausa")]])
    session.send_error = requests.Timeout(f"read timeout url: /bot{TOKEN}/sendMessage")
    assert bot.poll_once() == 2
    assert [t for t, _ in desk.calls] == ["estado", "pausa"]
    assert bot.reply(OWNER, "hola") is False
    assert "no se pudo responder" in debug_logs.text and TOKEN not in debug_logs.text


# --------------------------------------------------------------------------- hilo
def test_start_and_stop_cleanly():
    bot, session, desk = make_bot([[upd(1, "estado")]])
    assert bot.start() is True and bot.running
    assert bot.start() is True                     # idempotente
    for _ in range(200):
        if desk.calls:
            break
        threading.Event().wait(0.01)
    bot.stop(timeout=5)
    assert not bot.running
    assert desk.calls == [("estado", "telegram")]
    assert bot._thread.name == "desk-telegram" and bot._thread.daemon


def test_nothing_is_processed_after_stop():
    bot, session, desk = make_bot([[upd(1, "long HYPE")]])
    bot._stop.set()
    assert bot.poll_once() == 0
    assert desk.calls == [] and bot.offset is None  # ni se ejecuta ni se confirma: llegará al reanudar


def test_start_without_token_is_refused(debug_logs):
    bot = TelegramBot("", [OWNER], lambda t, source: "")
    assert bot.start() is False and not bot.running
    assert "TELEGRAM_BOT_TOKEN" in debug_logs.text


def test_start_without_chats_warns(debug_logs):
    bot, _, _ = make_bot([], chats=[])
    assert bot.start() is True
    bot.stop(timeout=5)
    assert "sin TELEGRAM_CHAT_ID" in debug_logs.text

"""Canales de aviso: niveles, cola en hilo, troceo, fallos de red tragados y sin fugas de secretos."""

from __future__ import annotations

import io
import json
import logging
import threading

import pytest
import requests

from desk.clock import SimClock
from desk.config import load_settings
from desk.notify import (TELEGRAM_CHUNK, WEBHOOK_CHUNK, Channel, ChannelError, ConsoleChannel, FileChannel,
                         Notification, Notifier, TelegramChannel, WebhookChannel, build_notifier, level_rank,
                         normalize_level, redact, safe_url, split_message)

TOKEN = "123456789:AAH-secreto_del_bot-XYZ"
HOOK = "https://discord.com/api/webhooks/987654/s3cr3t-webhook-token"


# --------------------------------------------------------------------------- dobles
class FakeResponse:
    def __init__(self, status: int = 200, body=None, headers=None, text: str = ""):
        self.status_code = status
        self._body = {"ok": True, "result": {}} if body is None else body
        self.headers = headers or {}
        self.text = text or json.dumps(self._body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    """Registra cada POST. ``responses``: cola de FakeResponse o excepciones (vacía = 200 ok)."""

    def __init__(self, responses=None):
        self.calls: list[dict] = []
        self.responses = list(responses or [])

    def post(self, url, timeout=None, **kwargs):
        self.calls.append({"url": url, "timeout": timeout, **kwargs})
        if self.responses:
            r = self.responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        return FakeResponse()

    def methods(self):
        return [c["url"].rsplit("/", 1)[-1] for c in self.calls]


class Recorder(Channel):
    name = "rec"

    def __init__(self, min_level="info", blocking=False):
        super().__init__(min_level)
        self.blocking = blocking
        self.got: list[Notification] = []

    def send(self, n):
        self.got.append(n)


class Boom(Channel):
    name = "boom"

    def send(self, n):
        raise RuntimeError("canal roto")


@pytest.fixture
def debug_logs(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog


# --------------------------------------------------------------------------- niveles
def test_normalize_level_aliases_and_unknown():
    assert normalize_level("WARNING") == "warn"
    assert normalize_level("error") == "critical"
    assert normalize_level("crítico") == "critical"
    assert normalize_level("nada") == "info"
    assert normalize_level(None) == "info"
    assert normalize_level("raro", default="warn") == "warn"
    assert level_rank("info") < level_rank("warn") < level_rank("critical")


def test_min_level_filters_each_channel():
    all_ch, warn_ch, crit_ch = Recorder("info"), Recorder("warn"), Recorder("critical")
    n = Notifier([all_ch, warn_ch, crit_ch], threaded=False)
    for lvl in ("info", "warn", "critical"):
        n.send(f"aviso {lvl}", level=lvl)
    assert [x.level for x in all_ch.got] == ["info", "warn", "critical"]
    assert [x.level for x in warn_ch.got] == ["warn", "critical"]
    assert [x.level for x in crit_ch.got] == ["critical"]


def test_notification_carries_metadata_and_clock_time():
    clock = SimClock(1_800_000_000.0)
    rec = Recorder()
    Notifier([rec], clock=clock).send("hola", level="warning", kind="halt", coin="BTC", image=b"png")
    n = rec.got[0]
    assert (n.text, n.level, n.kind, n.coin, n.image, n.ts) == ("hola", "warn", "halt", "BTC", b"png", 1_800_000_000.0)


def test_send_never_raises_with_bad_inputs_or_broken_channels(debug_logs):
    rec = Recorder()
    n = Notifier([Boom(), rec], threaded=False)
    n.send(None, level=object(), kind=None, coin=None, image="no-bytes")   # type: ignore[arg-type]
    n.send("sigue", level="critical")
    assert [x.text for x in rec.got] == ["None", "sigue"]     # un canal roto no frena a los demás
    assert rec.got[0].image is None
    assert n.failed["boom"] == 2
    assert "canal roto" in n.last_error


# --------------------------------------------------------------------------- cola e hilo
class Gate(Channel):
    """Canal 'de red' que se bloquea hasta que el test lo libera."""

    name = "gate"
    blocking = True

    def __init__(self):
        super().__init__("info")
        self.started = threading.Event()
        self.release = threading.Event()
        self.got: list[str] = []
        self.threads: set[str] = set()

    def send(self, n):
        self.threads.add(threading.current_thread().name)
        self.started.set()
        assert self.release.wait(5)
        self.got.append(n.text)


def test_network_channels_do_not_block_the_caller():
    gate, local = Gate(), Recorder()
    n = Notifier([gate, local])
    n.send("uno")
    assert gate.started.wait(2)                 # el hilo lo está enviando y sigue bloqueado...
    n.send("dos")                               # ...pero send() vuelve al momento
    assert [x.text for x in local.got] == ["uno", "dos"]   # los canales locales, en línea
    assert n.pending() >= 1
    gate.release.set()
    assert n.flush(5)
    assert gate.got == ["uno", "dos"]
    assert gate.threads == {"desk-notify"}
    n.close()


def test_full_queue_drops_oldest(debug_logs):
    gate = Gate()
    n = Notifier([gate], queue_size=2)
    n.send("1")
    assert gate.started.wait(2)                 # "1" en vuelo; la cola admite 2 más
    for t in ("2", "3", "4", "5"):
        n.send(t)
    assert n.dropped == 2
    assert "cola llena" in debug_logs.text
    gate.release.set()
    assert n.flush(5)
    assert gate.got == ["1", "4", "5"]
    n.close()


def test_worker_survives_channel_errors_and_close_stops_thread():
    class Flaky(Recorder):
        def send(self, n):
            if n.text == "falla":
                raise ChannelError("red caída")
            super().send(n)

    ch = Flaky(blocking=True)
    n = Notifier([ch])
    for t in ("a", "falla", "b"):
        n.send(t)
    assert n.flush(5)
    assert [x.text for x in ch.got] == ["a", "b"]
    assert n.failed["rec"] == 1 and n.sent["rec"] == 2
    thread = n._thread
    n.close()
    assert thread is not None and not thread.is_alive()
    n.send("después")                           # tras close, un envío vuelve a arrancar el hilo
    assert n.flush(5)
    assert ch.got[-1].text == "después"
    n.close()


# --------------------------------------------------------------------------- troceo
def test_split_message_prefers_line_boundaries_and_respects_limit():
    lines = [f"línea {i:03d} " + "x" * 50 for i in range(200)]
    text = "\n".join(lines)
    chunks = split_message(text, 1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)
    assert "\n".join(chunks) == text            # cortes sólo en saltos de línea: nada se pierde
    assert all(c.startswith("línea") for c in chunks)


def test_split_message_hard_splits_long_lines_and_skips_empty():
    text = "a" * 9000
    chunks = split_message(text, TELEGRAM_CHUNK)
    assert [len(c) for c in chunks] == [4000, 4000, 1000]
    assert "".join(chunks) == text
    assert split_message("") == [] and split_message("   \n ") == []
    assert split_message("corto") == ["corto"]


# --------------------------------------------------------------------------- consola y fichero
def test_console_labels_bell_and_no_markup_interpretation():
    buf = io.StringIO()
    ch = ConsoleChannel(buf, bell=True)
    ch.send(Notification("todo bien [c-000001] [bold]x[/bold]", "info"))
    ch.send(Notification("ojo", "warn"))
    ch.send(Notification("HALT", "critical"))
    out = buf.getvalue()
    assert "todo bien [c-000001] [bold]x[/bold]" in out          # rich no se come los corchetes
    assert out.count("\a") == 2                                  # pitido sólo en warn y critical
    assert "[AVISO] ojo" in out and "[CRÍTICO] HALT" in out
    assert "\a" not in out.split("\n")[0]


def test_console_without_bell_and_with_callable_out():
    buf = io.StringIO()
    ConsoleChannel(buf, bell=False).send(Notification("ojo", "critical"))
    assert "\a" not in buf.getvalue()

    got: list[str] = []
    ch = ConsoleChannel(got.append, bell=True)
    ch.send(Notification("info", "info"))
    ch.send(Notification("aviso", "warn"))
    assert got == ["info", "\a[AVISO] aviso"]


def test_console_plain_fallback_without_rich():
    buf = io.StringIO()
    ConsoleChannel(buf, bell=False, color=False).send(Notification("texto", "warn"))
    assert buf.getvalue() == "[AVISO] texto\n"


def test_file_channel_writes_jsonl(tmp_path):
    path = tmp_path / "sub" / "avisos.jsonl"
    n = Notifier([FileChannel(path)], clock=SimClock(1_767_225_600.0))
    n.send("uno", level="warn", kind="halt", coin="BTC")
    n.send("dos", image=b"\x89PNG")
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert recs[0] == {"ts": 1_767_225_600.0, "time": "2026-01-01 00:00:00", "level": "warn", "kind": "halt",
                       "coin": "BTC", "text": "uno", "image": False}
    assert recs[1]["image"] is True and recs[1]["level"] == "info"


# --------------------------------------------------------------------------- Telegram
def test_telegram_send_message_to_every_chat():
    s = FakeSession()
    ch = TelegramChannel(TOKEN, ["111", " 222 ", ""], session=s)
    ch.send(Notification("hola", "info"))
    ch.send(Notification("ojo", "warn"))
    assert s.methods() == ["sendMessage"] * 4
    assert s.calls[0]["url"] == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    first, second, third = (c["json"] for c in s.calls[:3])
    assert first == {"chat_id": "111", "text": "hola", "disable_web_page_preview": True, "disable_notification": True}
    assert second["chat_id"] == "222"
    assert third["text"] == "[AVISO] ojo" and third["disable_notification"] is False   # warn suena
    assert s.calls[0]["timeout"] == 10.0


def test_telegram_chunks_long_messages():
    s = FakeSession()
    text = "\n".join(f"fila {i} " + "y" * 90 for i in range(100))       # ~10 000 caracteres
    TelegramChannel(TOKEN, ["1"], session=s).send(Notification(text, "info"))
    sent = [c["json"]["text"] for c in s.calls]
    assert len(sent) == 3 and all(len(t) <= TELEGRAM_CHUNK for t in sent)
    assert "\n".join(sent) == text


def test_telegram_send_photo_multipart_with_caption():
    s = FakeSession()
    TelegramChannel(TOKEN, ["1"], session=s).send(Notification("informe", "warn", image=b"\x89PNGdata"))
    assert s.methods() == ["sendPhoto"]
    call = s.calls[0]
    assert call["data"]["caption"] == "[AVISO] informe" and call["data"]["chat_id"] == "1"
    name, blob, mime = call["files"]["photo"]
    assert blob == b"\x89PNGdata" and mime == "image/png"


def test_telegram_photo_with_long_text_sends_text_after():
    s = FakeSession()
    TelegramChannel(TOKEN, ["1"], session=s).send(Notification("z" * 1500, "info", image=b"img"))
    assert s.methods() == ["sendPhoto", "sendMessage"]
    assert "caption" not in s.calls[0]["data"]
    assert s.calls[1]["json"]["text"] == "z" * 1500


def test_telegram_photo_failure_still_delivers_text():
    s = FakeSession([FakeResponse(400, {"ok": False, "description": "Bad Request: PHOTO_INVALID_DIMENSIONS"})])
    ch = TelegramChannel(TOKEN, ["1"], session=s)
    with pytest.raises(ChannelError, match="foto: .*PHOTO_INVALID"):
        ch.send(Notification("informe", "warn", image=b"img"))
    assert s.methods() == ["sendPhoto", "sendMessage"]
    assert s.calls[1]["json"]["text"] == "[AVISO] informe"


def test_telegram_retries_once_on_429():
    waits: list[float] = []
    s = FakeSession([FakeResponse(429, {"ok": False, "parameters": {"retry_after": 3}}), FakeResponse()])
    ch = TelegramChannel(TOKEN, ["1"], session=s, sleep=waits.append)
    ch.send(Notification("hola"))
    assert waits == [3.0] and len(s.calls) == 2

    s2 = FakeSession([FakeResponse(429, {"ok": False, "parameters": {"retry_after": 999}})] * 2)
    waits.clear()
    with pytest.raises(ChannelError, match="429"):
        TelegramChannel(TOKEN, ["1"], session=s2, sleep=waits.append).send(Notification("hola"))
    assert waits == [10.0]          # con tope: el hilo de envío no se queda colgado


def test_telegram_network_errors_are_swallowed_without_leaking_token(debug_logs):
    err = requests.ConnectionError(
        f"HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries exceeded with url: "
        f"/bot{TOKEN}/sendMessage (Caused by NameResolutionError)")
    s = FakeSession([err, FakeResponse(400, {"ok": False, "description": "Bad Request: chat not found"})])
    ch = TelegramChannel(TOKEN, ["1", "2"], session=s)
    n = Notifier([ch], threaded=False)
    n.send("hola")                                      # no lanza
    assert n.failed["telegram"] == 1
    assert "chat not found" in n.last_error and "ConnectionError" in n.last_error
    assert TOKEN not in debug_logs.text and TOKEN not in n.last_error
    assert "AAH-secreto" not in debug_logs.text
    assert TOKEN not in repr(ch) and TOKEN not in n.describe()


def test_telegram_ok_false_is_an_error_and_missing_config_fails_softly(debug_logs):
    s = FakeSession([FakeResponse(200, {"ok": False, "description": "Forbidden: bot was blocked by the user"})])
    with pytest.raises(ChannelError, match="blocked"):
        TelegramChannel(TOKEN, ["1"], session=s).send(Notification("x"))
    n = Notifier([TelegramChannel("", ["1"], session=FakeSession())], threaded=False)
    n.send("x")
    assert "sin token" in n.last_error


def test_telegram_through_notifier_thread_is_delivered():
    s = FakeSession()
    n = Notifier([TelegramChannel(TOKEN, ["1"], session=s)])
    n.send("en cola")
    assert n.flush(5)
    assert s.calls[0]["json"]["text"] == "en cola"
    n.close()


# --------------------------------------------------------------------------- webhook
def test_webhook_payload_has_content_and_text_and_chunks():
    s = FakeSession([FakeResponse(204, {})] * 5)
    ch = WebhookChannel(HOOK, session=s)
    assert ch.min_level == "warn"
    ch.send(Notification("halt", "critical"))
    assert s.calls[0]["url"] == HOOK
    assert s.calls[0]["json"] == {"content": "[CRÍTICO] halt", "text": "[CRÍTICO] halt"}
    s.calls.clear()
    ch.send(Notification("w" * 5000, "warn"))
    assert len(s.calls) == 3 and all(len(c["json"]["content"]) <= WEBHOOK_CHUNK for c in s.calls)


def test_webhook_errors_never_log_full_url(debug_logs):
    err = requests.ConnectionError("Max retries exceeded with url: /api/webhooks/987654/s3cr3t-webhook-token")
    s = FakeSession([err, FakeResponse(404, {"message": "Unknown Webhook"})])
    ch = WebhookChannel(HOOK, min_level="info", session=s)
    n = Notifier([ch], threaded=False)
    n.send("a")
    n.send("b")
    assert n.failed["webhook"] == 2
    assert "Unknown Webhook" in n.last_error and "discord.com" in n.last_error
    for leak in (HOOK, "s3cr3t-webhook-token", "/api/webhooks/987654"):
        assert leak not in debug_logs.text
        assert leak not in repr(ch)
    assert safe_url(HOOK) == "https://discord.com/…"
    assert safe_url("no es url") == "<url inválida>"


def test_redact_handles_url_encoded_secrets():
    assert redact(f"x {TOKEN} y", [TOKEN]) == "x *** y"
    assert TOKEN not in redact("a 123456789%3AAAH-secreto_del_bot-XYZ b", [TOKEN])
    assert redact("abc", ["", "ab"]) == "abc"            # secretos vacíos o diminutos no destrozan el texto


# --------------------------------------------------------------------------- configuración
def test_build_notifier_from_settings(tmp_path, debug_logs):
    s = load_settings(env={}, dotenv_path=None, state_dir=str(tmp_path), llm_enabled=False,
                      telegram_bot_token=TOKEN, telegram_chat_ids=["42"], webhook_url=HOOK,
                      alert_min_level_telegram="warning", alert_min_level_webhook="nivel-raro", alert_bell=False)
    n = build_notifier(s, out=io.StringIO(), session=FakeSession(), file_path=tmp_path / "n.jsonl")
    kinds = {type(c).__name__: c for c in n.channels}
    assert set(kinds) == {"ConsoleChannel", "FileChannel", "TelegramChannel", "WebhookChannel"}
    assert kinds["TelegramChannel"].min_level == "warn"
    assert kinds["WebhookChannel"].min_level == "warn"           # nivel inválido -> por defecto, con aviso
    assert kinds["ConsoleChannel"].bell is False
    assert "inválido" in debug_logs.text
    desc = n.describe()
    assert "telegram" in desc and TOKEN not in desc and HOOK not in desc


def test_build_notifier_without_secrets_is_console_only(tmp_path, debug_logs):
    s = load_settings(env={}, dotenv_path=None, state_dir=str(tmp_path), llm_enabled=False, telegram_bot_token=TOKEN)
    n = build_notifier(s, out=io.StringIO())
    assert [type(c).__name__ for c in n.channels] == ["ConsoleChannel"]
    assert "TELEGRAM_CHAT_ID" in debug_logs.text
    assert Notifier().describe() == "sin canales de aviso"

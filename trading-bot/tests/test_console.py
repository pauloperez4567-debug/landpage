"""Consola interactiva (``desk/console.py``): autocompletado, envío por inbox/outbox con espera de la respuesta,
seguimiento de alarmas y avisos de la mesa, historial y bucle. Sin red ni terminal: el inbox/outbox van a
``tmp_path`` y la "mesa" es un reloj simulado que responde al dormir (o una ``TradingDesk`` real con datos falsos).
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from typing import Callable, Optional

import pytest
from rich.console import Console

from desk import console as con
from desk.__main__ import main
from desk.clock import SimClock
from desk.console import (CMD_PREFIX, HISTORY_FILE, VERBS, DeskConsole, Follower, History, Notice, completions,
                          format_notice, format_reply, reply_style)
from desk.journal import OUTBOX_RE, Journal

T0 = 1_767_225_600.0          # 2026-01-01 00:00:00 UTC
COINS = ["BTC", "ETH", "HYPE", "xyz:GOLD"]


# --------------------------------------------------------------------------- utilidades
def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def alert(path: Path, text: str, level: str = "warn", ts: float = T0, **extra) -> None:
    rec = {"ts": ts, "time": "2026-01-01 00:00:00", "level": level, "kind": "price_above", "coin": "BTC",
           "text": text, "rule_id": 1, **extra}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def outbox_line(path: Path, cmd_id: str, text: str, stamp: str = "2026-01-01 00:00:00") -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(f"{stamp} [{cmd_id}] {ln}\n" for ln in text.splitlines()))


def inbox_commands(state: Path) -> list[tuple[str, str]]:
    """(id, texto) de cada línea ``[id] texto`` del inbox (las escritas a mano sin id se saltan)."""
    p = state / "inbox.txt"
    if not p.exists():
        return []
    found = (re.match(r"^\[([^\]]+)\] (.*)$", ln) for ln in p.read_text(encoding="utf-8").splitlines())
    return [m.groups() for m in found if m]


def recorder(width: int = 120) -> Console:
    return Console(record=True, width=width, file=io.StringIO(), color_system="truecolor", force_terminal=False,
                   highlight=False, markup=False, emoji=False)


class ReplyingClock(SimClock):
    """Reloj simulado que hace de mesa: al dormir, ``responder(id, texto)`` contesta a lo nuevo del inbox."""

    def __init__(self, state: Path, responder: Optional[Callable[[str, str], Optional[str]]], *, delay: int = 1):
        super().__init__(T0)
        self.state, self.responder, self.delay = state, responder, delay
        self.sleeps = 0
        self.seen: set[str] = set()

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        self.sleeps += 1
        if self.responder is None or self.sleeps < self.delay:
            return
        for cid, text in inbox_commands(self.state):
            if cid in self.seen:
                continue
            self.seen.add(cid)
            reply = self.responder(cid, text)
            if reply is not None:
                outbox_line(self.state / "outbox.log", cid, reply)


def make_console(state: Path, clock=None, *, timeout: float = 5.0, inputs=None, **kw) -> DeskConsole:
    it = iter(inputs or [])

    def fake_input(prompt: str) -> str:
        try:
            item = next(it)
        except StopIteration:
            raise EOFError from None
        if isinstance(item, BaseException):
            raise item
        return item
    return DeskConsole(state, clock=clock or SimClock(T0), console=recorder(), timeout=timeout,
                       input_fn=fake_input, use_readline=False, follow=False, **kw)


def printed(c: DeskConsole) -> str:
    return c.console.export_text(clear=False)


def style_of(c: DeskConsole, needle: str):
    for seg in c.console._record_buffer:
        if needle in seg.text:
            return seg.style
    raise AssertionError(f"{needle!r} no se imprimió")


# --------------------------------------------------------------------------- autocompletado
def test_first_word_completes_spanish_and_english_verbs():
    assert set(completions([], "", COINS)) == set(VERBS)
    assert completions([], "ci", COINS) == ["cierra"]
    assert completions([], "cl", COINS) == ["close"]
    assert set(completions([], "pa", COINS)) == {"pausa", "pause", "panico"}
    assert completions([], "PÁN", COINS) == ["panico"]                         # sin mayúsculas ni acentos
    assert "estado" in completions([], "es", COINS) and "status" in completions([], "st", COINS)
    assert {"backtest", "adopta", "alerta", "mute", "silencia"} <= set(VERBS)


def test_coins_after_verbs_and_modifiers_after_the_coin():
    assert completions(["long"], "", COINS) == COINS
    assert completions(["long"], "h", COINS) == ["HYPE"]
    assert completions(["COMPRA"], "x", COINS) == ["xyz:GOLD"]                 # HIP-3 completo
    assert completions(["vende"], "e", COINS) == ["ETH"]
    after = completions(["long", "HYPE"], "", COINS)
    assert "zona" in after and "mercado" in after and "HYPE" not in after
    assert completions(["long", "hype", "2u"], "z", COINS) == ["zona", "zone"]
    assert completions(["cierra"], "", COINS) == ["todo"] + COINS
    assert completions(["close"], "a", COINS) == ["all"]
    assert completions(["sl"], "b", COINS) == ["BTC"]


def test_intervals_after_the_coin_of_chart_commands():
    assert completions(["backtest"], "b", COINS) == ["BTC"]
    assert completions(["backtest", "BTC"], "1", COINS) == ["1m", "15m", "1h", "12h", "1d", "1w"]
    assert "4h" in completions(["adopta", "eth"], "", COINS)
    assert completions(["backtest", "BTC", "4h"], "", COINS) == []              # ya tiene temporalidad
    assert completions(["alerta", "HYPE"], "15", COINS) == ["15m"]


def test_follow_words_and_free_text():
    assert completions(["auto"], "", COINS) == ["on", "off"]
    assert completions(["reset"], "h", COINS) == ["halt"]
    assert completions(["estado"], "", COINS) == []
    assert completions(["que", "tal"], "b", COINS) == ["BTC"]                  # texto libre: monedas
    assert completions(["que"], "4", COINS) == ["4h"]                          # ... y temporalidades


def test_console_coins_come_from_status_and_playbook_and_are_cached(tmp_path):
    c = make_console(tmp_path)
    assert c.coins() == []
    write_json(tmp_path / "status.json", {"line": "", "report": "Automático ON (x) sobre BTC, ETH | riesgo",
                                          "positions": [{"coin": "SOL", "size": 1}],
                                          "pending": [{"coin": "xyz:GOLD", "side": "long"}]})
    write_json(tmp_path / "playbook.json", {"entries": [{"coin": "HYPE", "interval": "1h", "strategy": "ema_trend"}]})
    assert c.coins() == ["BTC", "ETH", "SOL", "xyz:GOLD", "HYPE"]
    assert c.complete(["long"], "s") == ["SOL"]
    (tmp_path / "status.json").write_text('{"symbols": ["DOGE"', encoding="utf-8")   # a medio escribir
    assert c.coins() == ["HYPE"] and c.complete(["long"], "") == ["HYPE"]


def test_readline_completer_uses_the_words_before_the_cursor(tmp_path):
    write_json(tmp_path / "status.json", {"symbols": ["BTC", "BCH", "ETH"]})
    c = make_console(tmp_path)

    class FakeReadline:
        buffer = "backtest b"

        def get_line_buffer(self):
            return self.buffer

        def get_begidx(self):
            return len(self.buffer) - 1
    c._rl = FakeReadline()
    assert [c._rl_complete("b", i) for i in range(3)] == ["BTC", "BCH", None]
    FakeReadline.buffer = "backtest BTC "
    FakeReadline.get_begidx = lambda self: len(self.buffer)
    assert c._rl_complete("", 0) == "1m" and "4h" in c._matches


def test_completion_never_raises(tmp_path, monkeypatch):
    c = make_console(tmp_path)
    monkeypatch.setattr(con, "completions", lambda *a, **k: 1 / 0)
    assert c.complete([], "lo") == []


# --------------------------------------------------------------------------- envío y respuesta
def test_send_writes_to_inbox_and_prints_the_reply_in_color(tmp_path):
    clock = ReplyingClock(tmp_path, lambda cid, text: f"Rechazada: {text} (spread alto)\nPnL día -$5.00 | +$7.25")
    c = make_console(tmp_path, clock)
    lines = c.send("long HYPE 2u zona")
    [(cid, text)] = inbox_commands(tmp_path)
    assert cid.startswith(f"{CMD_PREFIX}-") and text == "long HYPE 2u zona"
    assert lines == ["Rechazada: long HYPE 2u zona (spread alto)", "PnL día -$5.00 | +$7.25"]
    out = printed(c)
    assert "Rechazada: long HYPE 2u zona" in out
    assert style_of(c, "Rechazada").color.name == "red"
    assert style_of(c, "+$7.25").color.name == "green" and style_of(c, "-$5.00").color.name == "red"


def test_multiline_reply_is_collected_until_it_settles(tmp_path):
    def slow(cid, text):                    # la mesa escribe la respuesta en dos tandas
        return "Estado de la mesa"
    clock = ReplyingClock(tmp_path, slow)
    c = make_console(tmp_path, clock)
    orig = clock.sleep

    def sleep(seconds):
        orig(seconds)
        if clock.sleeps == 2:
            [(cid, _)] = inbox_commands(tmp_path)
            outbox_line(tmp_path / "outbox.log", cid, "Posiciones (0)")
    clock.sleep = sleep
    assert c.send("estado") == ["Estado de la mesa", "Posiciones (0)"]


def test_reply_of_another_command_or_old_outbox_lines_are_not_taken(tmp_path):
    outbox_line(tmp_path / "outbox.log", "con-viejo", "respuesta vieja")
    clock = ReplyingClock(tmp_path, lambda cid, text: "nueva")
    c = make_console(tmp_path, clock)
    orig = clock.sleep

    def sleep(seconds):
        outbox_line(tmp_path / "outbox.log", "cli-otro", "de otra consola")
        orig(seconds)
    clock.sleep = sleep
    assert c.send("estado") == ["nueva"]
    assert "de otra consola" not in printed(c) and "respuesta vieja" not in printed(c)


def test_no_reply_keeps_the_command_and_prints_it_when_it_arrives(tmp_path):
    clock = ReplyingClock(tmp_path, None)
    c = make_console(tmp_path, clock, timeout=3)
    assert c.send("cierra HYPE") == []
    out = printed(c)
    assert "no respondió en 3 s" in out and "se ejecutará (una sola vez)" in out and "saldrá aquí" in out
    [(cid, _)] = inbox_commands(tmp_path)
    assert c.follower.watching == [cid]
    outbox_line(tmp_path / "outbox.log", cid, "HYPE: cierre enviado")
    assert c.pump() == 1
    assert f"Respuesta tardía [{cid}]" in printed(c) and "HYPE: cierre enviado" in printed(c)
    assert c.follower.watching == [] and c.pump() == 0


def test_ctrl_c_while_waiting_stops_waiting_but_not_the_command(tmp_path):
    class Interrupting(SimClock):
        def sleep(self, seconds):
            raise KeyboardInterrupt
    c = make_console(tmp_path, Interrupting(T0))
    assert c.send("long BTC") == []
    assert "Sin esperar más" in printed(c) and len(inbox_commands(tmp_path)) == 1
    assert len(c.follower.watching) == 1


def test_bad_text_is_refused_locally_and_never_glued_to_a_partial_line(tmp_path):
    c = make_console(tmp_path, ReplyingClock(tmp_path, lambda cid, text: "ok"))
    assert c.send("long\nBTC") == [] and "saltos de línea" in printed(c)
    assert not (tmp_path / "inbox.txt").exists()
    (tmp_path / "inbox.txt").write_text("estado", encoding="utf-8")         # alguien escribió sin salto final
    c.send("pausa")
    lines = (tmp_path / "inbox.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "estado" and re.match(rf"^\[{CMD_PREFIX}-[0-9a-f]{{8}}\] pausa$", lines[1])


def test_unwritable_inbox_is_reported(tmp_path):
    (tmp_path / "inbox.txt").mkdir()                                          # no se puede abrir como fichero
    c = make_console(tmp_path)
    assert c.send("estado") == [] and "No se pudo escribir" in printed(c)


def test_reply_styles():
    assert reply_style("Orden rechazada: spread") == "red"
    assert reply_style("ESTADO INCIERTO: no repitas") == "red"
    assert reply_style("¿Confirmas? (si/no)") == "yellow"
    assert reply_style("HYPE: SL colocado en 84.5") == "green"
    assert reply_style("Estado de la mesa") == ""
    t = format_reply(["equity $1 | PnL día +$3.00", "pérdida -$2.50"])
    assert t.plain == "equity $1 | PnL día +$3.00\npérdida -$2.50"


# --------------------------------------------------------------------------- consola: bucle y salir
def test_salir_closes_only_the_console_and_sends_nothing(tmp_path):
    clock = ReplyingClock(tmp_path, lambda cid, text: f"hecho: {text}")
    c = make_console(tmp_path, clock, inputs=["", "estado", "  SALIR  ", "pausa"])
    assert c.run() == 0
    assert [t for _, t in inbox_commands(tmp_path)] == ["estado"]             # "pausa" nunca se envió
    out = printed(c)
    assert "hecho: estado" in out and "Consola cerrada. La mesa sigue en marcha." in out


def test_eof_ctrl_c_and_unexpected_errors_never_break_the_loop(tmp_path, monkeypatch):
    clock = ReplyingClock(tmp_path, lambda cid, text: "ok")
    c = make_console(tmp_path, clock, inputs=[KeyboardInterrupt(), "boom", "estado"])
    orig = c.send

    def send(text):
        if text == "boom":
            raise RuntimeError("fallo raro")
        return orig(text)
    monkeypatch.setattr(c, "send", send)
    assert c.run() == 0                                                     # termina con EOF (Ctrl+D)
    out = printed(c)
    assert "(Ctrl+C) Para cerrar la consola" in out and "Error inesperado en la consola (RuntimeError" in out
    assert [t for _, t in inbox_commands(tmp_path)] == ["estado"] and out.rstrip().endswith("La mesa sigue en marcha.")


def test_banner_shows_last_status_live_mode_and_warnings(tmp_path):
    c = make_console(tmp_path, warnings=["la mesa no lee la cuenta desde hace 12 min"])
    text = c.banner().plain
    assert "No hay status.json" in text and "ATENCIÓN: la mesa no lee la cuenta" in text
    write_json(tmp_path / "status.json", {"line": "equity $1,000.00 | PnL día +$5.00 (+0.50%) | auto ON | halt no",
                                          "report": "Estado de la mesa — x (modo live, datos hyperliquid)"})
    text = c.banner().plain
    assert "MODO LIVE" in text and "Último estado: equity $1,000.00" in text


# --------------------------------------------------------------------------- historial
class FakeHistoryReadline:
    def __init__(self):
        self.items: list[str] = []
        self.length = None

    def set_history_length(self, n):
        self.length = n

    def read_history_file(self, path):
        with open(path, encoding="utf-8") as f:
            self.items = f.read().splitlines()

    def write_history_file(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.items) + "\n")


def test_history_lives_in_the_state_dir_and_survives_sessions(tmp_path):
    h = History(tmp_path / HISTORY_FILE, length=50)
    rl = FakeHistoryReadline()
    h.load(rl)                                                              # primera vez: sin fichero
    assert rl.items == [] and rl.length == 50
    rl.items = ["long HYPE", "estado"]
    h.save(rl)
    rl2 = FakeHistoryReadline()
    History(tmp_path / HISTORY_FILE).load(rl2)
    assert rl2.items == ["long HYPE", "estado"]
    (tmp_path / "ro").mkdir()
    History(tmp_path / "ro").save(rl2)                                      # ruta imposible: se avisa, no lanza


def test_real_readline_history_and_completer_are_restored(tmp_path):
    readline = pytest.importorskip("readline")
    before = readline.get_completer()
    c = DeskConsole(tmp_path, clock=SimClock(T0), console=recorder(), input_fn=lambda p: "salir", follow=False)
    assert c.run() == 0
    assert (tmp_path / HISTORY_FILE).exists()
    assert readline.get_completer() is before


# --------------------------------------------------------------------------- seguimiento de alarmas y avisos
def test_follower_only_reports_new_complete_lines(tmp_path):
    alerts, box = tmp_path / "alerts.jsonl", tmp_path / "outbox.log"
    alert(alerts, "vieja: no se repite al abrir")
    outbox_line(box, "desk", "aviso viejo")
    f = Follower(tmp_path, SimClock(T0))
    assert f.poll() == []
    alert(alerts, "BTC cruzó 85000 al alza", level="info")
    with open(alerts, "a", encoding="utf-8") as fh:
        fh.write('{"ts": 1, "level": "critical", "text": "a medio')          # sin salto: espera
    outbox_line(box, "desk", "AUTO [ema_trend] BTC LONG: entrada ejecutada\nMotivo: cruce EMA")
    outbox_line(box, "c-000001", "respuesta a otro: no es un aviso")
    got = f.poll()
    assert [(n.source, n.level, n.text) for n in got] == [
        ("alarma", "info", "BTC cruzó 85000 al alza"),
        ("mesa", "info", "AUTO [ema_trend] BTC LONG: entrada ejecutada\nMotivo: cruce EMA")]
    with open(alerts, "a", encoding="utf-8") as fh:
        fh.write(' escribir"}\n')
    [n] = f.poll()
    assert n.level == "critical" and n.text == "a medio escribir" and n.time.startswith("1970-01-01")


def test_follower_skips_muted_alerts_and_dedupes_the_same_text_from_both_files(tmp_path):
    clock = SimClock(T0)
    f = Follower(tmp_path, clock)
    alert(tmp_path / "alerts.jsonl", "silenciada", muted=True)
    alert(tmp_path / "alerts.jsonl", "HALT por pérdida diaria", level="critical")
    outbox_line(tmp_path / "outbox.log", "desk", "HALT por pérdida diaria")
    got = f.poll()
    assert [(n.source, n.text) for n in got] == [("alarma", "HALT por pérdida diaria")]
    clock.advance(600)                                                      # más tarde, el mismo texto sí sale
    outbox_line(tmp_path / "outbox.log", "desk", "HALT por pérdida diaria")
    assert [n.source for n in f.poll()] == ["mesa"]
    alert(tmp_path / "alerts.jsonl", "BTC cruzó 85000 al alza")             # una alarma que se repite por el
    alert(tmp_path / "alerts.jsonl", "BTC cruzó 85000 al alza")             # mismo fichero sale cada vez
    assert [n.text for n in f.poll()] == ["BTC cruzó 85000 al alza"] * 2


def test_periodic_status_line_is_printed_dimmed(tmp_path):
    f = Follower(tmp_path, SimClock(T0))
    outbox_line(tmp_path / "outbox.log", "desk", "equity $9,998.04 | PnL día -$1.96 (-0.02%) | posiciones 1: SOL L 16.66 "
                "-$0.83 | auto OFF | halt no")
    [n] = f.poll()
    assert n.source == "estado"
    c = recorder()
    c.print(format_notice(n))
    assert all(seg.style.dim for seg in c._record_buffer if "equity" in seg.text)


def test_follower_restarts_on_rotation_truncation_and_late_creation(tmp_path):
    alerts = tmp_path / "alerts.jsonl"
    f = Follower(tmp_path, SimClock(T0))                                    # aún no existe
    alert(alerts, "primera")
    assert [n.text for n in f.poll()] == ["primera"]
    alerts.rename(tmp_path / "alerts.jsonl.1")                              # rotación (alerts.py la hace así)
    alert(alerts, "tras rotar")
    assert [n.text for n in f.poll()] == ["tras rotar"]
    alerts.write_text("", encoding="utf-8")                                 # truncado y vuelto a crecer (más
    alert(alerts, "tras truncar y crecer")                                  # largo que lo leído) entre dos lecturas
    assert [n.text for n in f.poll()] == ["tras truncar y crecer"]
    alerts.write_text("", encoding="utf-8")
    assert f.poll() == []
    alert(alerts, "tras truncar")
    assert [n.text for n in f.poll()] == ["tras truncar"]


def test_follower_takes_a_line_half_written_when_the_console_opened_as_new(tmp_path):
    alerts = tmp_path / "alerts.jsonl"
    alert(alerts, "vieja")
    with open(alerts, "a", encoding="utf-8") as fh:
        fh.write('{"ts": 1, "level": "warn", "text": "llega')
    f = Follower(tmp_path, SimClock(T0))
    with open(alerts, "a", encoding="utf-8") as fh:
        fh.write(' ahora"}\n')
    assert [n.text for n in f.poll()] == ["llega ahora"]


def test_follower_survives_garbage(tmp_path):
    (tmp_path / "alerts.jsonl").write_bytes(b"")
    f = Follower(tmp_path, SimClock(T0))
    (tmp_path / "alerts.jsonl").write_bytes(b'\xff\xfe\nnope\n[1]\n{"level": 7, "text": null}\n{"text": "ok"}\n')
    (tmp_path / "outbox.log").write_bytes(b"\x00 basura\n")
    assert [(n.level, n.text) for n in f.poll()] == [("info", "ok")]


def test_notice_format_and_colors():
    c = recorder()
    c.print(format_notice(Notice("alarma", "critical", "HALT por drawdown", "2026-01-01 10:20:30")))
    c.print(format_notice(Notice("mesa", "warn", "ATENCIÓN: no se pudo cerrar BTC", "2026-01-01 10:21:00")))
    c.print(format_notice(Notice("alarma", "info", "ETH +$12.00", "")))
    text = c.export_text(clear=False)
    assert "10:20:30  ALARMA  CRÍTICO HALT por drawdown" in text
    assert "10:21:00  MESA  AVISO ATENCIÓN: no se pudo cerrar BTC" in text
    styles = [(seg.text, seg.style) for seg in c._record_buffer if seg.text.strip()]
    badges = [(text, st.bgcolor.name) for text, st in styles if text in (" ALARMA ", " MESA ")]
    assert badges == [(" ALARMA ", "red"), (" MESA ", "yellow"), (" ALARMA ", "cyan")]     # color = nivel
    assert dict(styles)["HALT por drawdown"].color.name == "red"
    assert dict(styles)["+$12.00"].color.name == "green"


def test_follow_thread_prints_alarms_in_real_time(tmp_path):
    c = DeskConsole(tmp_path, clock=SimClock(T0), console=recorder(), use_readline=False, poll_s=0.02)
    c.start_follow()
    try:
        alert(tmp_path / "alerts.jsonl", "HYPE entró en la zona LONG", level="warn")
        outbox_line(tmp_path / "outbox.log", "desk", "AUTO [donchian_breakout] ETH SHORT: entrada ejecutada")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and "ETH SHORT" not in printed(c):
            time.sleep(0.01)
    finally:
        c.stop_follow()
    out = printed(c)
    assert "HYPE entró en la zona LONG" in out and "AUTO [donchian_breakout] ETH SHORT" in out
    assert c._thread is None


def test_follow_thread_never_dies_on_errors(tmp_path, monkeypatch):
    c = make_console(tmp_path)
    monkeypatch.setattr(c.follower, "poll", lambda: 1 / 0)
    assert c.pump() == 0


def test_notice_while_typing_redraws_the_prompt(tmp_path):
    out = io.StringIO()
    term = Console(file=out, force_terminal=True, width=100, color_system=None)
    c = DeskConsole(tmp_path, clock=SimClock(T0), console=term, use_readline=False, follow=False)

    class Buf:
        def get_line_buffer(self):
            return "long HY"
    c._rl, c._at_prompt = Buf(), True
    alert(tmp_path / "alerts.jsonl", "BTC cruzó 85000")
    c.follower = Follower(tmp_path, SimClock(T0), from_start=True)
    assert c.pump() == 1
    assert out.getvalue().startswith("\r\x1b[2K") and out.getvalue().endswith("mesa> long HY")


# --------------------------------------------------------------------------- con una mesa real (sin red)
def test_console_talks_to_a_real_desk_through_inbox_and_outbox(settings, market, clock):
    from desk.desk import TradingDesk
    from desk.exchange.paper import PaperBroker
    settings.auto_enabled = False
    state = settings.state_path
    state.mkdir(parents=True, exist_ok=True)
    desk = TradingDesk(settings, market, PaperBroker(market, settings, clock, state / "paper_broker.json"), clock,
                       journal=Journal(state, clock), strategies=[], out=lambda text: None)
    desk.run_cycle()

    class DeskClock(SimClock):
        def sleep(self, seconds):                                          # mientras la consola espera, la mesa cicla
            super().sleep(seconds)
            clock.advance(seconds)
            desk.run_cycle()
    c = make_console(state, DeskClock(T0), timeout=10, inputs=["long SOL a mercado", "estado", "salir"])
    assert c.run() == 0
    out = printed(c)
    assert "SOL LONG" in out and "Estado de la mesa" in out
    assert "SOL" in desk.broker.account().positions
    ids = [cid for cid, _ in inbox_commands(state)]
    box = (state / "outbox.log").read_text(encoding="utf-8")
    assert len(ids) == 2 and all(f"[{cid}]" in box for cid in ids)
    assert all(OUTBOX_RE.match(ln) for ln in box.splitlines())


# --------------------------------------------------------------------------- CLI
def test_cli_console_starts_and_exits_on_eof(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(EOFError()))
    assert main(["console", "--state-dir", str(tmp_path), "--timeout", "1"]) == 0
    out = capsys.readouterr().out
    assert "Consola de la mesa" in out and "La mesa sigue en marcha" in out
    assert "ATENCIÓN: salud de la mesa: no hay latido" in out              # sin mesa en marcha: se avisa


def test_send_cli_still_uses_the_shared_client(tmp_path, capsys):
    """``python -m desk send`` y la consola comparten ``desk.remote`` (sin lógica duplicada)."""
    import desk.__main__ as cli
    import desk.remote as remote
    assert cli.send_command is remote.send_command and con.send_command is remote.send_command
    assert cli.wait_reply is remote.wait_reply and con.wait_reply is remote.wait_reply
    assert main(["send", "estado", "--no-wait", "--state-dir", str(tmp_path)]) == 0
    assert "Enviado [cli-" in capsys.readouterr().out and inbox_commands(tmp_path)[0][1] == "estado"

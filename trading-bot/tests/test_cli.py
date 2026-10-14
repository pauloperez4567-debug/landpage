"""CLI (``python -m desk``): perfil del estado, ``send``, reanudación del reloj ``--fast`` y salud.

Sin red: el modo live sólo se usa para construir el perfil (la cuenta se deriva de una clave de prueba) y para
comprobar qué recibe el broker live (constructor sustituido).
"""

from __future__ import annotations

import json
import logging
import os

import pytest

from desk import __main__ as cli
from desk.__main__ import (append_inbox_line, build_profile, check_profile, health_check, main, resume_clock_ts)
from desk.clock import SimClock
from desk.config import LIVE_CONFIRM_PHRASE, load_settings
from desk.journal import Journal

KEY = "0x" + "4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"
T0 = 1_767_225_600.0


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)
            h.close()
    for h in handlers:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(level)


def live(tmp_path, **kw):
    return load_settings(env={}, dotenv_path=None, state_dir=str(tmp_path / "state"), llm_enabled=False,
                         mode="live", live_confirm=LIVE_CONFIRM_PHRASE, hl_private_key=KEY, **kw)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# --------------------------------------------------------------------------- perfil del estado
def test_live_profile_records_network_and_account(tmp_path):
    p = build_profile(live(tmp_path), fast=False)
    assert p["mode"] == "live" and p["network"] == "testnet" and p["account"].startswith("0x")
    assert build_profile(live(tmp_path, hl_mainnet=True), fast=False)["network"] == "mainnet"
    other = build_profile(live(tmp_path, hl_account_address="0x" + "AB" * 20), fast=False)
    assert other["account"] == "0x" + "ab" * 20
    paper = load_settings(env={}, dotenv_path=None, state_dir=str(tmp_path), data_source="sim")
    assert build_profile(paper, fast=True) == {"mode": "paper", "data": "sim", "clock": "sim"}


def test_testnet_state_is_refused_on_mainnet_or_another_account(tmp_path):
    # Antes el máximo de equity de testnet se aplicaba a la cuenta real: halt por drawdown y cierre de todo.
    j = Journal(tmp_path / "state", SimClock())
    check_profile(j, build_profile(live(tmp_path), fast=False))
    with pytest.raises(SystemExit, match="otra configuración"):
        check_profile(j, build_profile(live(tmp_path, hl_mainnet=True), fast=False))
    with pytest.raises(SystemExit, match="otra configuración"):
        check_profile(j, build_profile(live(tmp_path, hl_account_address="0x" + "cd" * 20), fast=False))
    check_profile(j, build_profile(live(tmp_path), fast=False))          # la misma: se acepta


def test_old_profile_without_network_is_accepted_and_completed(tmp_path):
    j = Journal(tmp_path / "state", SimClock())
    j.save_json("profile", {"mode": "live", "data": "hyperliquid", "clock": "real"})
    check_profile(j, build_profile(live(tmp_path), fast=False))
    saved = j.load_json("profile")
    assert saved["network"] == "testnet" and saved["account"].startswith("0x")
    with pytest.raises(SystemExit):                                       # y a partir de ahí se comprueba
        check_profile(j, build_profile(live(tmp_path, hl_mainnet=True), fast=False))


def test_build_broker_passes_the_seen_fills_of_the_cursor(tmp_path, monkeypatch):
    from desk.exchange import hyperliquid_live as hl
    got = {}
    monkeypatch.setattr(hl.LiveHyperliquidBroker, "from_settings",
                        classmethod(lambda cls, market, settings, clock, **kw: got.update(kw) or "broker"))
    s = live(tmp_path)
    j = Journal(s.state_path, SimClock())
    j.save_json("desk", {"fills_cursor_ts": T0 + 1.5, "fills_seen": ["1001", "1002"]})
    assert cli.build_broker(s, None, SimClock(), j) == "broker"
    assert got == {"fills_since_ts": T0 + 1.5, "fills_seen": ["1001", "1002"]}


# --------------------------------------------------------------------------- send
def test_send_adds_missing_newline_before_its_line(tmp_path):
    inbox = tmp_path / "inbox.txt"
    inbox.write_bytes(b"pausa")                                           # escrito a mano sin salto final
    append_inbox_line(inbox, "[cli-1] cierra todo")
    append_inbox_line(inbox, "[cli-2] estado")
    assert inbox.read_text(encoding="utf-8") == "pausa\n[cli-1] cierra todo\n[cli-2] estado\n"
    empty = tmp_path / "nuevo.txt"
    append_inbox_line(empty, "[cli-3] estado")
    assert empty.read_text(encoding="utf-8") == "[cli-3] estado\n"


def test_send_cli_never_glues_to_a_partial_line(tmp_path, capsys):
    state = tmp_path / "st"
    state.mkdir()
    (state / "inbox.txt").write_text("pausa", encoding="utf-8")
    assert main(["send", "--no-wait", "--state-dir", str(state), "cierra", "todo"]) == 0
    lines = (state / "inbox.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "pausa" and lines[1].startswith("[cli-") and lines[1].endswith("] cierra todo")


def test_send_timeout_message_does_not_suggest_deleting_lines(tmp_path, capsys):
    state = tmp_path / "st"
    assert main(["send", "--timeout", "0.3", "--state-dir", str(state), "estado"]) == 1
    err = capsys.readouterr().err
    assert "No borres" in err and "borra esa línea" not in err


# --------------------------------------------------------------------------- reloj --fast
def test_resume_clock_uses_the_latest_time_saved_by_the_desk(tmp_path):
    state = tmp_path / "fast"
    assert resume_clock_ts(state) is None
    write_json(state / "profile.json", {"mode": "paper", "data": "sim", "clock": "sim", "clock_ts": T0 + 60})
    assert resume_clock_ts(state) == T0 + 60
    # Tras un SIGKILL el finally no guardó clock_ts, pero cada ciclo dejó su hora en health.json / heartbeat.
    write_json(state / "health.json", {"ts": T0 + 8 * 3600})
    (state / "heartbeat").write_text(f"{T0 + 8 * 3600 - 10}\n", encoding="utf-8")
    assert resume_clock_ts(state) == T0 + 8 * 3600


def test_fast_run_after_a_crash_does_not_go_back_in_time(tmp_path, capsys, restore_logging):
    state = tmp_path / "st"
    env = str(tmp_path / "no.env")
    assert main(["run", "--data", "sim", "--fast", "--cycles", "5", "--state-dir", str(state), "--env-file", env,
                 "--no-input", "--no-auto"]) == 0
    prof = json.loads((state / "profile.json").read_text(encoding="utf-8"))
    later = prof["clock_ts"] + 3 * 3600
    prof["clock_ts"] = T0                                                 # como si el finally no hubiera corrido
    write_json(state / "profile.json", prof)
    write_json(state / "health.json", {"ts": later})
    capsys.readouterr()
    assert main(["run", "--data", "sim", "--fast", "--cycles", "1", "--state-dir", str(state), "--env-file", env,
                 "--no-input", "--no-auto"]) == 0
    assert json.loads((state / "profile.json").read_text(encoding="utf-8"))["clock_ts"] >= later


# --------------------------------------------------------------------------- salud
def health(state, *, ts=T0 + 3600, started=T0, cuenta=T0 + 3600, fills=T0 + 3600, failing=(), errors=None):
    write_json(state / "health.json", {"ts": ts, "time": "x", "cycle": 9, "started_ts": started,
                                       "error_streak": len(failing), "failing": list(failing),
                                       "errors": errors or {}, "last_ok": {"cuenta": cuenta, "fills": fills}})


def test_health_ok_when_cycles_see_the_exchange(tmp_path, capsys):
    health(tmp_path)
    problems, info = health_check(tmp_path, os.path.getmtime(tmp_path / "health.json") + 5)
    assert problems == [] and "ciclo 9" in info[0]
    assert main(["health", "--state-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.startswith("OK")


def test_health_fails_when_account_is_unreadable_for_too_long(tmp_path, capsys):
    # Antes el latido y el HEALTHCHECK decían "sano" con todas las etapas de exchange fallando durante horas.
    health(tmp_path, ts=T0 + 16 * 3600, cuenta=T0, fills=T0, failing=("cuenta", "fills"),
           errors={"cuenta": "BrokerError: user_state: ConnectionError"})
    problems, _ = health_check(tmp_path, os.path.getmtime(tmp_path / "health.json"))
    assert len(problems) == 2 and "no lee la cuenta desde hace 960 min" in problems[0]
    assert "ConnectionError" in problems[0]
    assert main(["health", "--state-dir", str(tmp_path)]) == 1
    assert capsys.readouterr().out.startswith("MAL: la mesa no lee la cuenta")


def test_health_never_ok_since_start_and_stopped_desk(tmp_path):
    health(tmp_path, ts=T0 + 900, started=T0, cuenta=None, fills=T0 + 900)
    problems, _ = health_check(tmp_path, os.path.getmtime(tmp_path / "health.json"))
    assert problems == ["no lee la cuenta desde que arrancó, hace 15 min"]
    health(tmp_path)
    old = os.path.getmtime(tmp_path / "health.json") - 3600
    os.utime(tmp_path / "health.json", (old, old))
    problems, _ = health_check(tmp_path, old + 3600)
    assert problems and "parece detenida" in problems[0]
    assert health_check(tmp_path / "nada", T0)[0][0].startswith("no hay latido")


def test_status_warns_about_health(tmp_path, capsys):
    write_json(tmp_path / "status.json", {"ts": T0, "time": "2026-01-01 00:00:00", "line": "x", "report": "Equity"})
    health(tmp_path, ts=T0 + 7200, cuenta=T0, failing=("cuenta",), errors={"cuenta": "BrokerError: 429"})
    assert main(["status", "--state-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ATENCIÓN: la mesa no lee la cuenta desde hace 120 min" in out and "etapas fallando: cuenta" in out

"""Tests del diario: eventos, CSV, outbox con id, inbox/outbox para la CLI y JSON atómico."""

from __future__ import annotations

import csv
import json

from desk.journal import Journal, read_outbox
from desk.models import Fill


def test_event_fill_trade_equity_and_outbox(tmp_path, clock):
    j = Journal(tmp_path / "st", clock)
    rec = j.event("command", id="c-1", text="long HYPE", kind="ignorado", value=float("nan"))
    assert rec["kind"] == "command"
    line = json.loads((tmp_path / "st" / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["kind"] == "command" and line["value"] is None and line["time"] == "2026-01-01 00:00:00"

    j.fill(Fill("o1", "HYPE", True, 11.72, 85.288, 0.15, clock.now(), tag="entry"))
    j.fill(Fill("o2", "HYPE", False, 11.72, 87.9, 0.46, clock.now(), tag="tp", realized_pnl=30.6))
    rows = list(csv.DictReader(open(tmp_path / "st" / "fills.csv", encoding="utf-8")))
    assert [r["side"] for r in rows] == ["buy", "sell"] and rows[1]["tag"] == "tp"

    j.trade({"coin": "HYPE", "side": "long", "pnl": 29.987654321, "exit_px": 87.90000000001, "closed_ts": clock.now()})
    t = list(csv.DictReader(open(tmp_path / "st" / "trades.csv", encoding="utf-8")))[0]
    assert t["pnl"] == "29.987654" and t["exit_px"] == "87.9" and t["closed_time"] == "2026-01-01 00:00:00"

    j.equity(clock.now(), 10_000.0, 9_500.0, 1)
    assert (tmp_path / "st" / "equity.csv").read_text(encoding="utf-8").startswith("time,ts,equity")

    j.outbox("c-7", "línea 1\nlínea 2")
    j.outbox("desk", "aviso")
    lines, offset = read_outbox(j.outbox_path, "c-7")
    assert lines == ["línea 1", "línea 2"] and offset == j.outbox_path.stat().st_size
    assert read_outbox(j.outbox_path, "c-7", offset) == ([], offset)


def test_read_outbox_ignores_partial_lines(tmp_path):
    p = tmp_path / "outbox.log"
    p.write_text("2026-01-01 00:00:00 [a] hola\n2026-01-01 00:00:00 [a] a medi", encoding="utf-8")
    lines, offset = read_outbox(p, "a")
    assert lines == ["hola"] and offset == len("2026-01-01 00:00:00 [a] hola\n".encode())


def test_save_and_load_json_atomic_and_corrupt(tmp_path, clock):
    j = Journal(tmp_path, clock)
    assert j.load_json("desk", {"x": 1}) == {"x": 1}
    j.save_json("desk", {"a": 1.5, "b": [1, 2]})
    assert j.load_json("desk.json") == {"a": 1.5, "b": [1, 2]}
    assert not list(tmp_path.glob("*.tmp"))
    (tmp_path / "risk.json").write_text("{roto", encoding="utf-8")
    assert j.load_json("risk", "def") == "def"
    assert list(tmp_path.glob("risk.json.corrupt-*")) and not (tmp_path / "risk.json").exists()

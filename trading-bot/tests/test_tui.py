"""Panel en vivo (``desk/tui.py``): lectura tolerante del estado y paneles renderizados a texto con rich."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from rich.console import Console
from rich.panel import Panel

from desk import tui
from desk.__main__ import main
from desk.clock import SimClock
from desk.tui import (Snapshot, SnapshotReader, build_alerts, build_header, build_layout, build_positions, desk_notices,
                      load_snapshot, normalize_playbook, normalize_report, normalize_status, notice_level,
                      run_dashboard, sparkline, tail_lines, usage_bar)

T0 = 1_767_225_600.0          # 2026-01-01 00:00:00 UTC


# --------------------------------------------------------------------------- utilidades
def render(renderable, width: int = 160) -> str:
    console = Console(record=True, width=width, file=io.StringIO(), color_system=None)
    console.print(renderable)
    return console.export_text()


def style_of(renderable, needle: str, width: int = 160):
    """Estilo con el que se pinta el primer trozo que contiene ``needle``."""
    console = Console(width=width, file=io.StringIO(), force_terminal=True, color_system="truecolor")
    for seg in console.render(renderable):
        if needle in seg.text:
            return seg.style
    raise AssertionError(f"{needle!r} no aparece")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def jsonl(path: Path, records, tail: str = "") -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records) + tail, encoding="utf-8")


# Lo que la mesa escribe HOY en status.json (sólo texto: ver TradingDesk.write_status_snapshot).
TODAY_LINE = ("equity $9,921.52 | PnL día -$18.49 (-0.19%) | posiciones 2: BTC L 0.01772 -$4.99, ETH S 0.475 +$15.38 "
              "| auto ON | halt no")
TODAY_REPORT = "\n".join([
    "Estado de la mesa — 2026-01-02 00:59:30 UTC (modo paper, datos sim)",
    "Equity $9,921.52 | disponible $8,921.61 | PnL día -$18.49 (-0.19%) | drawdown 0.49% (límites: día 3%, máx. 5%)",
    "Automático ON (liquidity_sweep, squeeze_breakout) sobre BTC, ETH, SOL, HYPE | riesgo: día 2026-01-02 -0.19%",
    "Dimensionamiento: 1 unidad = $500 de margen x 3x | riesgo máx. 1% por operación | SL por defecto 1.5% | TP 2R",
    "Posiciones (2):",
    "  BTC LONG 0.01772 @ 84659.9 → 84378.5 | PnL -$4.99 (-1.0% s/margen) | 3x | SL 84090 TP 85736 | "
    "auto:squeeze_breakout",
    "  ETH SHORT 0.475 @ 3156.97 → 3124.6 | PnL +$15.38 (+3.1% s/margen) | 3x | SL 3212.9 TP 3047.9 | comando",
    "Entradas pendientes (1):",
    "  SOL LONG 3.2 @ 148.5 (SL 146.3 TP 152.9, caduca 2026-01-02 02:00 UTC) | comando",
    "Órdenes abiertas (3):",
    "  BTC venta stop 0.01772 @ 84090 reduce-only [sl]",
    "  BTC venta take_profit 0.01772 @ 85736 reduce-only [tp]",
    "  SOL compra limit 3.2 @ 148.5 [entry]",
    "Hoy: 1 operación(es) cerrada(s), 0 ganadora(s), PnL neto -$11.07",
])
TODAY_STATUS = {"ts": T0 + 86400 + 3570, "time": "2026-01-02 00:59:30", "line": TODAY_LINE, "report": TODAY_REPORT}

IS_RETURN = 77.77             # dentro de muestra: NUNCA debe aparecer en el panel


def full_status() -> dict:
    """``status.json`` con los campos ampliados de la fase 2."""
    return {
        "ts": T0 + 3600, "time": "2026-01-01 01:00:00", "line": "equity $10,127.00 | ...", "report": "",
        "mode": "paper", "data": "sim", "equity": 10127.0, "available": 9000.0, "day_pnl": -123.0,
        "day_pnl_pct": -1.2, "auto": True, "paused": False, "cooldown": False, "halt": None,
        "symbols": ["BTC", "ETH", "HYPE", "xyz:GOLD"],
        "risk": {"daily_loss_pct": 1.2, "daily_loss_limit_pct": 3.0, "daily_loss_used_pct": 55.0,
                 "drawdown_pct": 4.1, "max_drawdown_pct": 5.0, "drawdown_used_pct": 82.0},
        "positions": [
            {"coin": "BTC", "side": "long", "size": 0.05, "entry_px": 84000, "mark_px": 85000, "unrealized_pnl": 50.0,
             "pnl_pct": 3.6, "leverage": 3, "sl": 83000, "tp": 87000, "origin": "auto:ema_trend"},
            {"coin": "HYPE", "size": -12, "entry_px": 83.2, "mark_px": 84.1, "unrealized_pnl": -10.8, "pnl_pct": -2.1,
             "leverage": 2, "sl": 85.0, "tp": 80.0, "origin": "command"},
        ],
        "orders": [{"coin": "BTC", "is_buy": False, "kind": "stop", "size": 0.05, "trigger_px": 83000,
                    "reduce_only": True, "tag": "sl"}],
        "pending": [{"coin": "ETH", "side": "long", "planned_size": 0.5, "entry_px": 3100, "sl": 3050, "tp": 3200,
                     "expires_ts": T0 + 7200, "origin": "command"}],
        "alerts": {"rules": 3, "fired_total": 7, "muted": False, "muted_until": None, "silenced": 0},
        "signals": [{"ts": T0 + 1800, "coin": "BTC", "side": "long", "strategy": "ema_trend",
                     "reason": "cruce EMA 12/26 al alza"}],
    }


def write_full_state(d: Path) -> None:
    write_json(d / "status.json", full_status())
    write_json(d / "health.json", {"ts": T0 + 3600, "cycle": 360, "error_streak": 0, "failing": [],
                                   "last_ok": {"cuenta": T0 + 3600, "fills": T0 + 3600}})
    write_json(d / "risk.json", {"version": 1, "day": "2026-01-01", "day_start_equity": 10250.0,
                                 "high_water": 10560.0, "halted": False, "halt_reason": "", "paused": False,
                                 "cooldown_until": 0.0})
    jsonl(d / "alerts.jsonl", [
        {"ts": T0 + 100, "time": "2026-01-01 00:01:40", "level": "info", "kind": "price_above", "coin": "BTC",
         "text": "BTC cruzó 85000 al alza", "rule_id": 1},
        {"ts": T0 + 200, "time": "2026-01-01 00:03:20", "level": "warn", "kind": "rsi_below", "coin": "ETH",
         "text": "ETH RSI(14) 1h bajo 30 (28.4)", "rule_id": 2},
        {"ts": T0 + 300, "time": "2026-01-01 00:05:00", "level": "critical", "kind": "halt", "coin": None,
         "text": "HALT por pérdida diaria", "rule_id": None},
        {"ts": T0 + 400, "time": "2026-01-01 00:06:40", "level": "info", "kind": "volume_spike", "coin": "SOL",
         "text": "SOL volumen x3.2", "rule_id": 3, "muted": True},
    ], tail='{"ts": 1767226100, "level": "info", "text": "a medio escr')
    write_json(d / "playbook.json", {"version": 1, "entries": [
        {"coin": "BTC", "interval": "1h", "strategy": "ema_trend", "params": {"fast": 12, "slow": 26}, "units": 1.0,
         "adopted_ts": T0, "report_id": "abc123", "enabled": True,
         "oos_metrics": {"total_return_pct": 8.4, "profit_factor": 1.62, "max_drawdown_pct": 6.3, "n_trades": 41}},
    ]})
    write_json(d / "reports" / "BTC_1h_abc123.json", {"report": {
        "id": "abc123", "coin": "BTC", "interval": "1h", "days": 90, "bars": 2160, "created_ts": T0 - 3600,
        "winner": "ema_trend", "notes": [],
        "evaluations": [
            {"strategy": "ema_trend", "params": {"fast": 12, "slow": 26},
             "is_metrics": {"total_return_pct": IS_RETURN, "profit_factor": 4.4},
             "oos_metrics": {"total_return_pct": 8.4, "profit_factor": 1.62, "max_drawdown_pct": 6.3, "n_trades": 41},
             "oos_trades": 41, "folds": 4, "score": 1.1, "passed": True, "reasons": []},
            {"strategy": "rsi_reversion", "params": {"period": 14},
             "is_metrics": {"total_return_pct": IS_RETURN, "profit_factor": 4.4},
             "oos_metrics": {"total_return_pct": -3.2, "profit_factor": 0.81, "max_drawdown_pct": 12.0, "n_trades": 25},
             "folds": 4, "score": -0.4, "passed": False, "reasons": ["PF fuera de muestra 0.81 < 1.2"]},
        ]}, "result": {"equity": [[T0, 10000.0], [T0 + 3600, 10100.0]]}})
    jsonl(d / "events.jsonl", [
        {"ts": T0 + 10, "time": "2026-01-01 00:00:10", "kind": "start", "settings": {"mode": "paper"}},
        {"ts": T0 + 900, "time": "2026-01-01 00:15:00", "kind": "trade_closed", "coin": "SOL", "side": "short",
         "pnl": -11.07, "exit_reason": "sl", "closed_ts": T0 + 900, "origin": "auto:rsi_reversion"},
        {"ts": T0 + 1000, "time": "2026-01-01 00:16:40", "kind": "trade_closed", "coin": "ETH", "side": "long",
         "pnl": 22.42, "exit_reason": "tp", "closed_ts": T0 + 1000, "origin": "command"},
    ])
    (d / "outbox.log").write_text(
        "2026-01-01 00:30:00 [desk] AUTO [ema_trend] BTC LONG: entrada ejecutada\n"
        "2026-01-01 00:30:00 [desk] Motivo: cruce EMA\n"
        "2026-01-01 00:31:00 [c-000001] respuesta privada a un comando\n"
        "2026-01-01 00:40:00 [desk] HALT por pérdida diaria: -3.1%\n", encoding="utf-8")
    (d / "equity.csv").write_text("time,ts,equity,available,positions\n" + "".join(
        f"2026-01-01 00:{i:02d}:00,{T0 + 60 * i},{10000 + 10 * i},9000,1\n" for i in range(20)), encoding="utf-8")


# --------------------------------------------------------------------------- lectura de ficheros
def test_tail_lines_skips_partial_first_and_last_lines(tmp_path):
    p = tmp_path / "x.log"
    p.write_bytes(b"uno\ndos\ntres\ncua")
    assert tail_lines(p) == ["uno", "dos", "tres"]
    assert tail_lines(p, max_bytes=9) == ["tres"]          # el trozo empieza a mitad de "dos": se descarta
    assert tail_lines(tmp_path / "no.log") == []
    p.write_bytes(b"sin salto")
    assert tail_lines(p) == []


def test_desk_notices_groups_lines_of_the_same_notice_and_skips_other_ids():
    lines = ["2026-01-01 00:00:00 [desk] línea 1", "2026-01-01 00:00:00 [desk] línea 2",
             "2026-01-01 00:00:00 [c-1] respuesta", "2026-01-01 00:00:00 [desk] otro aviso", "basura"]
    assert desk_notices(lines) == [{"time": "2026-01-01 00:00:00", "text": "línea 1\nlínea 2"},
                                   {"time": "2026-01-01 00:00:00", "text": "otro aviso"}]


def test_normalize_status_reads_todays_text_format():
    st = normalize_status(TODAY_STATUS)
    assert st["present"] and st["mode"] == "paper" and st["data"] == "sim"
    assert st["equity"] == pytest.approx(9921.52) and st["available"] == pytest.approx(8921.61)
    assert st["day_pnl"] == pytest.approx(-18.49) and st["day_pnl_pct"] == pytest.approx(-0.19)
    assert st["daily_loss_limit_pct"] == 3 and st["max_drawdown_pct"] == 5 and st["drawdown_pct"] == pytest.approx(0.49)
    assert st["daily_used_pct"] == pytest.approx(0.19 / 3 * 100) and st["dd_used_pct"] == pytest.approx(9.8)
    assert st["auto"] is True and st["paused"] is False and st["halt"] == ""
    assert st["symbols"] == ["BTC", "ETH", "SOL", "HYPE"]
    btc, eth = st["positions"]
    assert (btc["coin"], btc["side"], btc["size"], btc["pnl"], btc["sl"]) == ("BTC", "long", 0.01772, -4.99, 84090)
    assert (eth["side"], eth["pnl"], eth["pnl_pct"], eth["origin"]) == ("short", 15.38, 3.1, "comando")
    assert [(o["coin"], o["kind"], o["side"], o["reduce_only"], o["tag"]) for o in st["orders"]] == [
        ("BTC", "stop", "sell", True, "sl"), ("BTC", "take_profit", "sell", True, "tp"),
        ("SOL", "limit", "buy", False, "entry")]
    assert st["pending"][0]["coin"] == "SOL" and st["pending"][0]["expires"] == "2026-01-02 02:00 UTC"


def test_normalize_status_prefers_structured_fields_and_accepts_aliases():
    raw = full_status()
    raw["positions"] = {"BTC": {"szi": "-0.5", "entryPx": 1, "entry": "84000", "upnl": "12.5"}}   # dict por moneda
    raw.pop("orders")
    raw["open_orders"] = [{"coin": "BTC", "side": "buy", "type": "limit", "price": "83000"}]
    raw["halted"], raw["halt_reason"] = True, "kill"
    raw.pop("halt")
    st = normalize_status(raw)
    assert st["equity"] == 10127.0 and st["daily_used_pct"] == 55.0 and st["dd_used_pct"] == 82.0
    assert st["positions"] == [{"coin": "BTC", "side": "short", "size": 0.5, "entry_px": 84000.0, "mark_px": None,
                                "pnl": 12.5, "pnl_pct": None, "leverage": None, "liq_px": None, "sl": None,
                                "tp": None, "origin": ""}]
    assert st["orders"][0]["side"] == "buy" and st["orders"][0]["px"] == 83000.0
    assert st["halt"] == "kill"
    assert st["signals"][0]["reason"] == "cruce EMA 12/26 al alza"


def test_normalize_playbook_and_report_accept_several_shapes():
    entry = {"coin": "ETH", "interval": "4h", "strategy": "donchian_breakout", "params": {"n": 20}}
    assert [e["coin"] for e in normalize_playbook([entry, 1, "x", {"interval": "1h"}])] == ["ETH"]
    assert normalize_playbook({"entries": {"ETH|4h": entry}})[0]["enabled"] is True
    assert normalize_playbook({"ETH|4h": entry, "version": 1})[0]["strategy"] == "donchian_breakout"
    assert normalize_playbook("basura") == []
    rep = normalize_report({"id": "z", "coin": "SOL", "interval": "15m", "winner": None,
                            "evaluations": [{"strategy": "ema_trend", "passed": False, "reasons": ["pocas operaciones"],
                                             "oos_metrics": {"n_trades": 3}, "folds": [{}, {}, {}]}]})
    assert rep["winner"] is None and rep["folds"] == 3 and rep["evaluations"][0]["oos"]["trades"] == 3
    win = normalize_report({"coin": "SOL", "winner": {"strategy": "x", "params": {"a": 1},
                                                      "oos_metrics": {"profit_factor": 1.5}}})
    assert win["winner"]["strategy"] == "x" and win["winner"]["oos"]["pf"] == 1.5
    assert normalize_report([1, 2]) is None and normalize_report({"foo": 1}) is None


# --------------------------------------------------------------------------- snapshot completo
def test_full_snapshot_renders_every_panel(tmp_path):
    write_full_state(tmp_path)
    snap = SnapshotReader(tmp_path, SimClock(T0 + 3600)).read()
    assert snap.problems == []
    text = render(build_layout(snap))
    # cabecera: modo, equity, PnL del día, consumo de límites, auto/halt, alarmas y salud
    for s in ("Mesa de trading", "PAPER", "datos sim", "$10,127.00", "-$123.00 (-1.20%)", "disponible $9,000.00",
              "Pérdida diaria", " 55%", "1.20% de 3%", "Drawdown", " 82%", "4.10% de 5%", "Automático ON",
              "Halt no", "Alarmas 3 activas, 7 disparadas", "Ciclo 360"):
        assert s in text, s
    # posiciones y órdenes
    for s in ("Posiciones (2)", "84000", "85000", "+$50.00", "+3.6%", "HYPE", "SHORT", "-$10.80", "ema_trend",
              "comando", "PnL no realizado +$39.20", "stop (SL)", "reduce-only [sl]", "entrada pendiente",
              "caduca 01/01 02:00"):
        assert s in text, s
    # playbook y último backtest: sólo métricas fuera de muestra, con su advertencia
    for s in ("Playbook (1)", "ema_trend (fast=12, slow=26)", "+8.4", "1.62", "Último backtest", "BTC 1h",
              "90 días", "id abc123", "Ganadora: ema_trend", "rsi_reversion", "rechazada: PF fuera de muestra 0.81",
              "walk-forward, 4 tramos", "Un backtest no garantiza resultados futuros"):
        assert s in text, s
    assert "77.7" not in text and "77.8" not in text and "4.40" not in text      # nada dentro de muestra
    # alarmas (la línea a medio escribir no sale), señales, operaciones y avisos de la mesa
    for s in ("BTC cruzó 85000 al alza", " AVISO ", " CRÍTICO ", "SOL volumen x3.2 (silenciada)",
              "cruce EMA 12/26 al alza", "+$22.42", "-$11.07", "take-profit", "stop-loss",
              "AUTO [ema_trend] BTC LONG: entrada ejecutada", "Motivo: cruce EMA", "HALT por pérdida diaria: -3.1%"):
        assert s in text, s
    assert "a medio escr" not in text and "respuesta privada" not in text
    assert text.index("BTC cruzó 85000") > text.index("ETH RSI(14)")             # lo más reciente arriba


def test_pnl_risk_bars_and_alarm_levels_have_their_colors(tmp_path):
    write_full_state(tmp_path)
    snap = load_snapshot(tmp_path, SimClock(T0 + 3600))
    pos = build_positions(snap)
    assert style_of(pos, "+$50.00").color.name == "green"
    assert style_of(pos, "-$10.80").color.name == "red"
    head = build_header(snap)
    assert style_of(head, "-$123.00").color.name == "red"
    assert style_of(head, " 55%").color.name == "yellow"
    hot = style_of(head, " 82%")
    assert hot.color.name == "red" and hot.bold
    alerts = build_alerts(snap)
    assert style_of(alerts, " CRÍTICO ").bgcolor.name == "red"
    assert style_of(alerts, " AVISO ").color.name == "yellow"
    assert usage_bar(10).spans[0].style == "green" and usage_bar(None).plain.endswith("s/d")


def test_narrow_terminal_stacks_panels_and_never_truncates_prices(tmp_path):
    write_full_state(tmp_path)
    text = render(build_layout(load_snapshot(tmp_path, SimClock(T0))), width=80)
    assert text.index("Último backtest") < text.index("Alarmas")                  # una columna
    assert "84000" in text and "85000" in text and "-$10.80" in text and "Origen" not in text
    assert max(len(ln) for ln in text.splitlines()) <= 80


def test_risk_json_overrides_halt_pause_and_cooldown(tmp_path):
    write_full_state(tmp_path)
    write_json(tmp_path / "risk.json", {"day": "2026-01-01", "day_start_equity": 10250.0, "high_water": 10560.0,
                                        "halted": True, "halt_reason": "daily_loss", "halt_detail": "-3.10% hoy",
                                        "paused": True, "cooldown_until": T0 + 9999})
    snap = load_snapshot(tmp_path, SimClock(T0))
    assert snap.status["halt"] == "daily_loss" and snap.status["paused"] and snap.status["cooldown"]
    head = build_header(snap)
    text = render(head)
    assert "HALT: daily_loss - -3.10% hoy" in text and "EN PAUSA" in text and "cooldown" in text
    assert style_of(head, "HALT: daily_loss").bgcolor.name == "red"
    # Un halt diario de otro día (la mesa aún no ha hecho su update) ya no cuenta.
    write_json(tmp_path / "health.json", {"ts": T0 + 86400 + 60, "cycle": 1})
    assert load_snapshot(tmp_path, SimClock(T0)).status["halt"] == ""


# --------------------------------------------------------------------------- snapshot parcial (formato de hoy)
def test_todays_status_without_other_files_renders_from_the_text(tmp_path):
    write_json(tmp_path / "status.json", TODAY_STATUS)
    snap = load_snapshot(tmp_path, SimClock(T0))
    text = render(build_layout(snap))
    for s in ("$9,921.52", "-$18.49 (-0.19%)", "  6%", "0.19% de 3%", " 10%", "0.49% de 5%", "Automático ON",
              "Halt no", "84659.9", "84378.5", "-$4.99", "+$15.38", "squeeze_breakout", "Órdenes (3) y entradas "
              "pendientes (1)", "take profit", "reduce-only [tp]", "límite", "[entry]", "Sin health.json",
              "Playbook vacío", "Sin informes de backtest", "Sin avisos todavía", "Sin señales recientes",
              "Sin operaciones cerradas", "Sin avisos de la mesa"):
        assert s in text, s
    assert snap.problems == []


def test_status_without_account_is_shown_in_red(tmp_path):
    line = "SIN DATOS DE LA CUENTA (timeout) | ciclos seguidos con fallos 4 | halt no"
    write_json(tmp_path / "status.json", {"ts": T0, "time": "2026-01-01 00:00:00", "line": line, "report": line,
                                          "account_ok": False})
    head = build_header(load_snapshot(tmp_path, SimClock(T0)))
    assert "SIN DATOS DE LA CUENTA (timeout)" in render(head)
    assert style_of(head, "SIN DATOS DE LA CUENTA").color.name == "red"


def test_empty_state_dir_renders_placeholders(tmp_path):
    snap = load_snapshot(tmp_path / "nada", SimClock(T0))
    text = render(build_layout(snap))
    assert "Sin status.json" in text and "Posiciones (0)" in text and "MODO ?" in text
    assert snap.problems == []


# --------------------------------------------------------------------------- snapshots corruptos
def test_half_written_status_keeps_the_previous_reading(tmp_path):
    write_full_state(tmp_path)
    reader = SnapshotReader(tmp_path, SimClock(T0))
    assert reader.read().status["equity"] == 10127.0
    (tmp_path / "status.json").write_text('{"ts": 1, "equity": 99', encoding="utf-8")
    snap = reader.read()
    assert snap.status["equity"] == 10127.0
    assert any("status.json ilegible o a medio escribir: se muestra la lectura anterior" in p for p in snap.problems)
    assert "Aviso del panel: status.json ilegible" in render(build_layout(snap))
    # Sin lectura anterior: se avisa y se pinta vacío.
    fresh = load_snapshot(tmp_path, SimClock(T0))
    assert fresh.status["present"] is False and any("status.json" in p for p in fresh.problems)
    assert "Sin status.json" in render(build_layout(fresh))


def test_garbage_everywhere_never_breaks_the_panel(tmp_path):
    write_json(tmp_path / "status.json", {
        "ts": "ayer", "time": 5, "line": None, "report": ["x"], "equity": "abc", "day_pnl": float("inf"),
        "risk": [1, 2], "positions": [{"coin": "BTC", "size": "abc", "entry_px": None, "unrealized_pnl": "NaN",
                                       "leverage": {"x": 1}}, "junk", 5, {"size": 1}],
        "orders": {"a": 1}, "pending": "x", "alerts": "muchas", "signals": [1, None, {"coin": 5}], "halt": 7,
        "playbook": {"entries": "x"}, "symbols": "BTC"})
    write_json(tmp_path / "health.json", ["no", "es", "un", "dict"])
    write_json(tmp_path / "risk.json", {"halted": "quizá", "cooldown_until": "luego", "day_start_equity": "x"})
    (tmp_path / "alerts.jsonl").write_bytes(b'\xff\xfe basura\n[1, 2]\n42\n{"level": 7, "text": null}\n')
    write_json(tmp_path / "alerts.json", {"rules": "x", "mute": {"until": "nunca"}})
    write_json(tmp_path / "playbook.json", [1, "x", {"coin": "BTC", "params": "x", "units": "dos", "enabled": "no"}])
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "roto.json").write_text("{", encoding="utf-8")
    write_json(tmp_path / "reports" / "raro.json", {"coin": "ETH", "evaluations": [{"strategy": "x", "score": "?",
                                                                                     "oos_metrics": [1]}]})
    (tmp_path / "events.jsonl").write_text('{"kind": "trade_closed", "coin": "BTC", "pnl": "abc"}\nnope\n'
                                           '{"kind": "signal", "coin": "ETH", "side": 3}\n', encoding="utf-8")
    (tmp_path / "outbox.log").write_bytes(b"\x00\x01 basura\n")
    (tmp_path / "equity.csv").write_text("a,b,c\n1,2,x\n", encoding="utf-8")
    snap = load_snapshot(tmp_path, SimClock(T0))
    text = render(build_layout(snap))
    assert "Posiciones (1)" in text and "BTC" in text and "Playbook (1)" in text and "desactivada" in text
    assert "Último backtest" in text and "ETH" in text
    assert "No se pudo pintar" not in text


def test_a_panel_that_fails_is_drawn_as_an_error_and_the_rest_survive(tmp_path, monkeypatch):
    write_full_state(tmp_path)
    snap = load_snapshot(tmp_path, SimClock(T0))

    def boom(_snap):
        raise RuntimeError("dato imposible")
    monkeypatch.setattr(tui, "build_signals", boom)
    text = render(build_layout(snap))
    assert "No se pudo pintar (RuntimeError: dato imposible)" in text
    assert "Posiciones (2)" in text and "BTC cruzó 85000" in text


def test_newest_readable_report_wins_over_a_broken_newer_one(tmp_path):
    write_full_state(tmp_path)
    broken = tmp_path / "reports" / "ETH_4h_zzz.json"
    broken.write_text('{"report": {"coin": "ETH"', encoding="utf-8")
    old = tmp_path / "reports" / "BTC_1h_abc123.json"
    os.utime(old, (T0, T0))
    snap = load_snapshot(tmp_path, SimClock(T0))
    assert snap.report["id"] == "abc123" and snap.reports_count == 2


def test_sparkline_and_notice_levels():
    assert sparkline([1, 2, 3]) == "▁▅█" and sparkline([5]) == "" and sparkline([2, 2]) == "▅▅"
    assert notice_level("HALT por drawdown máximo: -5%") == "critical"
    assert notice_level("equity $1 | PnL día +$0.00 (+0.00%) | posiciones 0 | auto ON | halt no") == "info"
    assert notice_level("ATENCIÓN: no se pudo cerrar BTC") == "warn"


# --------------------------------------------------------------------------- bucle y CLI
def test_run_dashboard_once_prints_a_frame(tmp_path):
    write_full_state(tmp_path)
    console = Console(record=True, width=140, file=io.StringIO(), color_system=None)
    assert run_dashboard(tmp_path, console=console, once=True) == 0
    assert "Mesa de trading" in console.export_text()


def test_run_dashboard_live_loop_refreshes_with_the_injected_clock(tmp_path):
    write_full_state(tmp_path)
    out = io.StringIO()
    console = Console(file=out, force_terminal=True, width=140, color_system=None)
    clock = SimClock(T0)
    assert run_dashboard(tmp_path, clock=clock, console=console, max_frames=3, screen=False) == 0
    assert clock.now() == T0 + 3.0 and "Mesa de trading" in out.getvalue()

    class Interrupt(SimClock):
        def sleep(self, seconds):
            raise KeyboardInterrupt
    assert run_dashboard(tmp_path, clock=Interrupt(T0), console=console, screen=False) == 0   # Ctrl+C: sale limpio


def test_cli_dashboard_once(tmp_path, capsys):
    write_json(tmp_path / "status.json", TODAY_STATUS)
    assert main(["dashboard", "--once", "--state-dir", str(tmp_path)]) == 0
    assert "$9,921.52" in capsys.readouterr().out
    assert main(["dashboard", "--once", "--state-dir", str(tmp_path / "no-existe")]) == 1


def test_snapshot_defaults_are_independent():
    a, b = Snapshot(Path("x"), 0.0), Snapshot(Path("y"), 0.0)
    a.status["positions"].append({"coin": "BTC"})
    assert b.status["positions"] == []
    assert isinstance(build_header(a), Panel)

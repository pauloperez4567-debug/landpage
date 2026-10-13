"""Parser de comandos: reglas ES/EN (deterministas) + fallback IA con un cliente falso.

Ningún test toca la red: el cliente de Anthropic se sustituye por un objeto con
``beta.messages.create`` que devuelve respuestas fabricadas o lanza errores.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields
from types import SimpleNamespace

import pytest

import desk.commands as commands
from desk.commands import (
    HELP_TEXT, LLM_ACTIONS, SCHEMA, SYSTEM_PROMPT, CommandParser, LLMCommandInterpreter,
    command_from_llm, describe, has_anthropic_credentials, normalize_coin, param_key, parse,
    parse_number, parse_rules, tokenize,
)
from desk.config import PARAM_ALIASES, TUNABLE, load_settings
from desk.models import Command, CommandAction as A, EntryMode as M, Side

L, S = Side.LONG, Side.SHORT
OPEN_FIELDS = ("coin", "side", "units", "entry_mode", "price", "sl_price", "sl_pct", "tp_price", "tp_pct")


def _assert_fields(cmd: Command, expected: dict) -> None:
    for k, v in expected.items():
        got = getattr(cmd, k)
        if isinstance(v, float):
            assert got == pytest.approx(v), (k, got, v)
        else:
            assert got == v, (k, got, v)


def _open(text, settings, **expected) -> Command:
    cmd = parse_rules(text, settings)
    assert cmd is not None, text
    assert cmd.action is A.OPEN and cmd.source == "regex" and cmd.raw == text.strip()
    full = {f: None for f in OPEN_FIELDS}
    full["entry_mode"] = M.ZONE                      # settings.default_entry_mode por defecto
    full.update(expected)
    _assert_fields(cmd, full)
    return cmd


# --- abrir posiciones ----------------------------------------------------------------

OPEN_CASES = [
    # lado + moneda (ejemplos del contrato)
    ("long HYPE", dict(coin="HYPE", side=L)),
    ("short btc", dict(coin="BTC", side=S)),
    ("compra SOL", dict(coin="SOL", side=L)),
    ("vende ETH", dict(coin="ETH", side=S)),
    ("buy eth", dict(coin="ETH", side=L)),
    ("sell SOL", dict(coin="SOL", side=S)),
    ("go long HYPE", dict(coin="HYPE", side=L)),
    ("abre un long en HYPE", dict(coin="HYPE", side=L)),
    ("pon un short en BTC", dict(coin="BTC", side=S)),
    ("HYPE long", dict(coin="HYPE", side=L)),
    ("long $HYPE", dict(coin="HYPE", side=L)),
    ("long HYPE-PERP", dict(coin="HYPE", side=L)),
    ("short BTC/USDC", dict(coin="BTC", side=S)),
    ("long HYPE por favor", dict(coin="HYPE", side=L)),
    ("long HYPE please!", dict(coin="HYPE", side=L)),
    # unidades
    ("long HYPE 2 unidades", dict(coin="HYPE", side=L, units=2.0)),
    ("long HYPE 2u", dict(coin="HYPE", side=L, units=2.0)),
    ("long HYPE 2 units", dict(coin="HYPE", side=L, units=2.0)),
    ("long HYPE 1 unidad", dict(coin="HYPE", side=L, units=1.0)),
    ("long HYPE 0,5u", dict(coin="HYPE", side=L, units=0.5)),
    ("long HYPE 1.5 uds", dict(coin="HYPE", side=L, units=1.5)),
    ("compra 2u de SOL", dict(coin="SOL", side=L, units=2.0)),
    ("long HYPE con 3 unidades", dict(coin="HYPE", side=L, units=3.0)),
    # modo zona
    ("long HYPE en zona", dict(coin="HYPE", side=L, entry_mode=M.ZONE)),
    ("long HYPE en zona de liquidez", dict(coin="HYPE", side=L, entry_mode=M.ZONE)),
    ("long HYPE zona", dict(coin="HYPE", side=L, entry_mode=M.ZONE)),
    ("short ETH in liquidity zone", dict(coin="ETH", side=S, entry_mode=M.ZONE)),
    # modo mercado
    ("long HYPE a mercado", dict(coin="HYPE", side=L, entry_mode=M.MARKET)),
    ("long HYPE market", dict(coin="HYPE", side=L, entry_mode=M.MARKET)),
    ("long HYPE ya", dict(coin="HYPE", side=L, entry_mode=M.MARKET)),
    ("short BTC al mercado", dict(coin="BTC", side=S, entry_mode=M.MARKET)),
    ("short BTC now", dict(coin="BTC", side=S, entry_mode=M.MARKET)),
    ("compra SOL a mercado", dict(coin="SOL", side=L, entry_mode=M.MARKET)),
    # límite
    ("long HYPE a 85.2", dict(coin="HYPE", side=L, entry_mode=M.LIMIT, price=85.2)),
    ("long HYPE @85.2", dict(coin="HYPE", side=L, entry_mode=M.LIMIT, price=85.2)),
    ("long HYPE @ 85.2", dict(coin="HYPE", side=L, entry_mode=M.LIMIT, price=85.2)),
    ("long HYPE limit 85.2", dict(coin="HYPE", side=L, entry_mode=M.LIMIT, price=85.2)),
    ("long HYPE a 85,2", dict(coin="HYPE", side=L, entry_mode=M.LIMIT, price=85.2)),
    ("long HYPE límite 85,2", dict(coin="HYPE", side=L, entry_mode=M.LIMIT, price=85.2)),
    ("long HYPE at 85.2", dict(coin="HYPE", side=L, entry_mode=M.LIMIT, price=85.2)),
    ("long HYPE en 85.2", dict(coin="HYPE", side=L, entry_mode=M.LIMIT, price=85.2)),
    ("vende ETH a precio 3300", dict(coin="ETH", side=S, entry_mode=M.LIMIT, price=3300.0)),
    ("short BTC limit @ 83,770.5", dict(coin="BTC", side=S, entry_mode=M.LIMIT, price=83770.5)),
    ("short BTC @ 83770", dict(coin="BTC", side=S, entry_mode=M.LIMIT, price=83770.0)),
    ("long SOL @ 149.985", dict(coin="SOL", side=L, entry_mode=M.LIMIT, price=149.985)),
    ("long HYPE a 2u", dict(coin="HYPE", side=L, units=2.0)),     # "a" + unidades no es límite
    # SL / TP dentro de la orden: % = distancia, sin % = precio
    ("long HYPE sl 1.5%", dict(coin="HYPE", side=L, sl_pct=1.5)),
    ("long HYPE stop 84.5", dict(coin="HYPE", side=L, sl_price=84.5)),
    ("long HYPE tp 4%", dict(coin="HYPE", side=L, tp_pct=4.0)),
    ("long HYPE tp 90", dict(coin="HYPE", side=L, tp_price=90.0)),
    ("long HYPE sl 1,5 %", dict(coin="HYPE", side=L, sl_pct=1.5)),
    ("long HYPE sl -1.5%", dict(coin="HYPE", side=L, sl_pct=1.5)),
    ("long HYPE tp +4%", dict(coin="HYPE", side=L, tp_pct=4.0)),
    ("long HYPE stop loss 84.5 take profit 90", dict(coin="HYPE", side=L, sl_price=84.5, tp_price=90.0)),
    ("long HYPE stop-loss 2 por ciento", dict(coin="HYPE", side=L, sl_pct=2.0)),
    ("long HYPE sl: 84.5 tp= 90", dict(coin="HYPE", side=L, sl_price=84.5, tp_price=90.0)),
    ("long HYPE con stop en 84.5", dict(coin="HYPE", side=L, sl_price=84.5)),
    ("short ETH objetivo 3000 sl 3350", dict(coin="ETH", side=S, tp_price=3000.0, sl_price=3350.0)),
    ("long HYPE sl 1.5 pct tp 4 percent", dict(coin="HYPE", side=L, sl_pct=1.5, tp_pct=4.0)),
    # combinaciones
    ("long HYPE 2u zona sl 1.5% tp 4%",
     dict(coin="HYPE", side=L, units=2.0, entry_mode=M.ZONE, sl_pct=1.5, tp_pct=4.0)),
    ("Long hype a 85,2 SL 84 TP 90",
     dict(coin="HYPE", side=L, entry_mode=M.LIMIT, price=85.2, sl_price=84.0, tp_price=90.0)),
    ("vende ETH @ 3300 sl 3350 tp 3100 1u",
     dict(coin="ETH", side=S, units=1.0, entry_mode=M.LIMIT, price=3300.0, sl_price=3350.0, tp_price=3100.0)),
    ("short btc 2 units market sl 1% tp 3%",
     dict(coin="BTC", side=S, units=2.0, entry_mode=M.MARKET, sl_pct=1.0, tp_pct=3.0)),
    ("long HYPE, 2u, sl 1.5%, tp 4%.", dict(coin="HYPE", side=L, units=2.0, sl_pct=1.5, tp_pct=4.0)),
    ("compra SOL 3u ya stop 140 objetivo 165",
     dict(coin="SOL", side=L, units=3.0, entry_mode=M.MARKET, sl_price=140.0, tp_price=165.0)),
    # monedas HIP-3: dex en minúsculas, ticker en mayúsculas
    ("long xyz:GOLD", dict(coin="xyz:GOLD", side=L)),
    ("long XYZ:gold 1u", dict(coin="xyz:GOLD", side=L, units=1.0)),
    ("short cash:tsla @ 250", dict(coin="cash:TSLA", side=S, entry_mode=M.LIMIT, price=250.0)),
    ("short km:us500 a mercado", dict(coin="km:US500", side=S, entry_mode=M.MARKET)),
]


@pytest.mark.parametrize("text,expected", OPEN_CASES, ids=[c[0] for c in OPEN_CASES])
def test_open_commands(settings, text, expected):
    _open(text, settings, **expected)


EQUIVALENT_OPEN = [
    "long HYPE 2u zona sl 1.5% tp 4%",
    "LONG hype 2U ZONA SL 1,5% TP 4%",
    "Long HYPE, 2 unidades, en zona de liquidez, sl 1.5 %, tp 4 por ciento",
    "compra HYPE 2 units zone stop loss 1.5 pct take profit 4 percent",
    "cómprame HYPE 2 unidades en la zona, stop 1,5% y objetivo 4%",
    "  lóng   HYPÉ   2u   zona  sl 1.5%  tp 4%  ",
    "abre long en HYPE con 2 unidades en zona sl 1.5% tp 4% porfa",
]


@pytest.mark.parametrize("text", EQUIVALENT_OPEN)
def test_case_accents_commas_and_wording_do_not_matter(settings, text):
    _open(text, settings, coin="HYPE", side=L, units=2.0, entry_mode=M.ZONE, sl_pct=1.5, tp_pct=4.0)


def test_default_entry_mode_comes_from_settings(tmp_path):
    s = load_settings(env={}, dotenv_path=None, state_dir=str(tmp_path), llm_enabled=False,
                      default_entry_mode="market")
    assert parse_rules("long HYPE", s).entry_mode is M.MARKET
    assert parse_rules("long HYPE zona", s).entry_mode is M.ZONE       # explícito gana
    assert parse_rules("long HYPE @ 85", s).entry_mode is M.LIMIT
    assert parse_rules("long HYPE", None).entry_mode is M.ZONE         # sin settings: zona


# --- otras acciones ------------------------------------------------------------------

ACTION_CASES = [
    # cerrar
    ("cierra HYPE", A.CLOSE, dict(coin="HYPE")),
    ("close HYPE", A.CLOSE, dict(coin="HYPE")),
    ("CIERRA hype ya", A.CLOSE, dict(coin="HYPE")),
    ("cierra el long de HYPE", A.CLOSE, dict(coin="HYPE")),
    ("close my BTC short", A.CLOSE, dict(coin="BTC")),
    ("sal de SOL", A.CLOSE, dict(coin="SOL")),
    ("exit eth", A.CLOSE, dict(coin="ETH")),
    ("flat HYPE", A.CLOSE, dict(coin="HYPE")),
    ("cierra la posición de xyz:gold", A.CLOSE, dict(coin="xyz:GOLD")),
    ("cierra todo", A.CLOSE_ALL, {}),
    ("close all", A.CLOSE_ALL, {}),
    ("flat", A.CLOSE_ALL, {}),
    ("Flatten", A.CLOSE_ALL, {}),
    ("cierra todas las posiciones", A.CLOSE_ALL, {}),
    ("close all positions", A.CLOSE_ALL, {}),
    ("cierra posiciones", A.CLOSE_ALL, {}),
    ("go flat", A.CLOSE_ALL, {}),
    # cancelar órdenes
    ("cancela HYPE", A.CANCEL, dict(coin="HYPE")),
    ("cancel HYPE", A.CANCEL, dict(coin="HYPE")),
    ("cancela las órdenes de HYPE", A.CANCEL, dict(coin="HYPE")),
    ("cancel orders BTC", A.CANCEL, dict(coin="BTC")),
    ("cancel all", A.CANCEL, dict(coin=None)),
    ("cancela todo", A.CANCEL, dict(coin=None)),
    ("cancela todas las órdenes", A.CANCEL, dict(coin=None)),
    ("cancel orders", A.CANCEL, dict(coin=None)),
    # SL / TP sobre una posición abierta
    ("stop HYPE 84.5", A.SET_SL, dict(coin="HYPE", price=84.5, sl_pct=None)),
    ("sl HYPE 84.5", A.SET_SL, dict(coin="HYPE", price=84.5, sl_pct=None)),
    ("sl HYPE 84,5", A.SET_SL, dict(coin="HYPE", price=84.5)),
    ("SL hype a 84.5", A.SET_SL, dict(coin="HYPE", price=84.5)),
    ("stop loss HYPE 84.5", A.SET_SL, dict(coin="HYPE", price=84.5)),
    ("mueve el stop de HYPE a 84,5", A.SET_SL, dict(coin="HYPE", price=84.5)),
    ("move stop BTC to 82000", A.SET_SL, dict(coin="BTC", price=82000.0)),
    ("move stop BTC to 82,000.5", A.SET_SL, dict(coin="BTC", price=82000.5)),
    ("pon el sl de mi long de HYPE en 84.5", A.SET_SL, dict(coin="HYPE", price=84.5)),
    ("sl HYPE 2%", A.SET_SL, dict(coin="HYPE", price=None, sl_pct=2.0)),
    ("stop 84.5 HYPE", A.SET_SL, dict(coin="HYPE", price=84.5)),
    ("tp HYPE 90", A.SET_TP, dict(coin="HYPE", price=90.0, tp_pct=None)),
    ("take profit HYPE 90", A.SET_TP, dict(coin="HYPE", price=90.0)),
    ("objetivo HYPE 90", A.SET_TP, dict(coin="HYPE", price=90.0)),
    ("sube el tp de HYPE a 92,5", A.SET_TP, dict(coin="HYPE", price=92.5)),
    ("tp HYPE 4%", A.SET_TP, dict(coin="HYPE", price=None, tp_pct=4.0)),
    # control de la mesa
    ("pausa", A.PAUSE, {}),
    ("pause", A.PAUSE, {}),
    ("Pausa el bot", A.PAUSE, {}),
    ("deja de operar", A.PAUSE, {}),
    ("stop trading", A.PAUSE, {}),
    ("reanuda", A.RESUME, {}),
    ("resume", A.RESUME, {}),
    ("reanudá", A.RESUME, {}),
    ("continue trading", A.RESUME, {}),
    ("vuelve a operar", A.RESUME, {}),
    ("auto on", A.AUTO_ON, {}),
    ("auto off", A.AUTO_OFF, {}),
    ("automatico on", A.AUTO_ON, {}),
    ("automático off", A.AUTO_OFF, {}),
    ("AUTO: ON", A.AUTO_ON, {}),
    ("modo automático activado", A.AUTO_ON, {}),
    ("activa el modo automático", A.AUTO_ON, {}),
    ("apaga el auto", A.AUTO_OFF, {}),
    ("turn off autopilot", A.AUTO_OFF, {}),
    ("estado", A.STATUS, {}),
    ("status", A.STATUS, {}),
    ("posiciones", A.STATUS, {}),
    ("Estado de la cuenta", A.STATUS, {}),
    ("mis posiciones", A.STATUS, {}),
    ("show positions", A.STATUS, {}),
    ("¿cómo vamos?", A.STATUS, {}),
    ("kill", A.KILL, {}),
    ("KILL!", A.KILL, {}),
    ("panico", A.KILL, {}),
    ("¡PÁNICO!", A.KILL, {}),
    ("panic", A.KILL, {}),
    ("apaga todo", A.KILL, {}),
    ("apágalo todo", A.KILL, {}),
    ("kill switch", A.KILL, {}),
    ("reset halt", A.RESET_HALT, {}),
    ("resetea el halt", A.RESET_HALT, {}),
    ("levanta el bloqueo", A.RESET_HALT, {}),
    ("unhalt", A.RESET_HALT, {}),
    # confirmaciones
    ("si", A.CONFIRM, {}),
    ("sí", A.CONFIRM, {}),
    ("SÍ", A.CONFIRM, {}),
    ("confirmar", A.CONFIRM, {}),
    ("confirmo", A.CONFIRM, {}),
    ("yes", A.CONFIRM, {}),
    ("confirm", A.CONFIRM, {}),
    ("sí, confirmo", A.CONFIRM, {}),
    ("no", A.REJECT, {}),
    ("NO", A.REJECT, {}),
    ("cancelar", A.REJECT, {}),
    ("cancel", A.REJECT, {}),
    ("reject", A.REJECT, {}),
    ("rechazar", A.REJECT, {}),
    # ayuda
    ("ayuda", A.HELP, {}),
    ("help", A.HELP, {}),
    ("?", A.HELP, {}),
    ("Comandos", A.HELP, {}),
]


@pytest.mark.parametrize("text,action,expected", ACTION_CASES, ids=[c[0] for c in ACTION_CASES])
def test_action_commands(settings, text, action, expected):
    cmd = parse_rules(text, settings)
    assert cmd is not None, text
    assert cmd.action is action and cmd.source == "regex" and cmd.raw == text.strip()
    _assert_fields(cmd, expected)
    if action not in (A.OPEN,):
        assert cmd.side is None and cmd.units is None


# --- parámetros -------------------------------------------------------------------------

PARAM_CASES = [
    ("apalancamiento 3", "leverage", 3.0),
    ("leverage 3", "leverage", 3.0),
    ("Apalancamiento 3x", "leverage", 3.0),
    ("leverage x3", "leverage", 3.0),
    ("lev 10", "leverage", 10.0),
    ("pon el apalancamiento a 3", "leverage", 3.0),
    ("set leverage to 5", "leverage", 5.0),
    ("leverage=4", "leverage", 4.0),
    ("riesgo 0.5", "risk_per_trade_pct", 0.5),
    ("riesgo 0,5", "risk_per_trade_pct", 0.5),
    ("riesgo 0,5%", "risk_per_trade_pct", 0.5),
    ("risk 1 %", "risk_per_trade_pct", 1.0),
    ("unidad 250", "unit_margin_usd", 250.0),
    ("unidad $250", "unit_margin_usd", 250.0),
    ("unidad 250 usd", "unit_margin_usd", 250.0),
    ("unit 1,000", "unit_margin_usd", None),            # ambiguo (¿mil o 1,000?) -> None
    ("unidad 1.500,50", "unit_margin_usd", 1500.5),
    ("posiciones 2", "max_open_positions", 2.0),
    ("max positions 4", "max_open_positions", 4.0),
    ("pérdida diaria 3%", "daily_loss_limit_pct", 3.0),
    ("daily loss 2.5", "daily_loss_limit_pct", 2.5),
    ("drawdown 8%", "max_drawdown_pct", 8.0),
    ("ttl 60", "entry_ttl_minutes", 60.0),
    ("ttl 90 min", "entry_ttl_minutes", 90.0),
    ("rr 2.5", "default_tp_r", 2.5),
    ("tp 3", "default_tp_r", 3.0),                       # sin moneda: TP por defecto en R
    ("tp 3r", "default_tp_r", 3.0),
    ("sl 2%", "default_sl_pct", 2.0),                    # sin moneda: SL por defecto en %
    ("sl 1.5", "default_sl_pct", 1.5),
    ("stop loss 2%", "default_sl_pct", 2.0),
    ("set risk_per_trade_pct 0.75", "risk_per_trade_pct", 0.75),
    ("default_tp_r 2", "default_tp_r", 2.0),
]


@pytest.mark.parametrize("text,param,value", PARAM_CASES, ids=[c[0] for c in PARAM_CASES])
def test_set_param(settings, text, param, value):
    cmd = parse_rules(text, settings)
    if value is None:
        assert cmd is None
        return
    assert cmd is not None and cmd.action is A.SET_PARAM
    assert cmd.param == param and cmd.value == pytest.approx(value) and cmd.coin is None


@pytest.mark.parametrize("alias,canonical", sorted(PARAM_ALIASES.items()))
def test_every_param_alias_parses_to_canonical(settings, alias, canonical):
    cmd = parse_rules(f"{alias} 2", settings)        # 2 está dentro del rango de todos
    assert cmd is not None and cmd.action is A.SET_PARAM
    assert cmd.param == canonical and cmd.value == 2.0


@pytest.mark.parametrize("name", sorted(TUNABLE))
def test_canonical_param_names_parse(settings, name):
    cmd = parse_rules(f"{name} 2", settings)
    assert cmd is not None and cmd.param == name


def test_parsed_set_param_is_accepted_by_settings(settings):
    for text in ("apalancamiento 3", "riesgo 0,5", "unidad 250", "posiciones 2", "sl 2%", "tp 3"):
        cmd = parse_rules(text, settings)
        settings.set_param(cmd.param, cmd.value)
    assert settings.leverage == 3 and isinstance(settings.leverage, int)
    assert settings.risk_per_trade_pct == 0.5 and settings.unit_margin_usd == 250.0
    assert settings.max_open_positions == 2 and settings.default_sl_pct == 2.0 and settings.default_tp_r == 3.0


def test_param_out_of_range_reaches_settings_validation(settings):
    # Nombre explícito: el parser no adivina límites, los valida Settings con un mensaje claro.
    cmd = parse_rules("apalancamiento 100", settings)
    assert cmd.param == "leverage" and cmd.value == 100.0
    with pytest.raises(ValueError, match="fuera de rango"):
        settings.set_param(cmd.param, cmd.value)


def test_param_key_helper():
    assert param_key("Apalancamiento") == "leverage"
    assert param_key("pérdida diaria") == "daily_loss_limit_pct"
    assert param_key("max_open_positions") == "max_open_positions"
    assert param_key("color") is None


# --- SL/TP: sobre posición vs dentro de una orden de apertura ---------------------------

@pytest.mark.parametrize("text,action,field,value", [
    ("stop HYPE 84.5", A.SET_SL, "price", 84.5),
    ("long HYPE stop 84.5", A.OPEN, "sl_price", 84.5),
    ("sl HYPE 1.5%", A.SET_SL, "sl_pct", 1.5),
    ("long HYPE sl 1.5%", A.OPEN, "sl_pct", 1.5),
    ("tp HYPE 90", A.SET_TP, "price", 90.0),
    ("long HYPE tp 90", A.OPEN, "tp_price", 90.0),
    ("tp HYPE 4%", A.SET_TP, "tp_pct", 4.0),
    ("long HYPE tp 4%", A.OPEN, "tp_pct", 4.0),
    ("sl 1.5%", A.SET_PARAM, "value", 1.5),
    ("tp 3", A.SET_PARAM, "value", 3.0),
])
def test_sl_tp_on_position_vs_inside_open(settings, text, action, field, value):
    cmd = parse_rules(text, settings)
    assert cmd.action is action and getattr(cmd, field) == pytest.approx(value)
    if action is A.SET_SL:
        assert cmd.sl_price is None and cmd.side is None
    if action is A.OPEN:
        assert cmd.price is None                     # el precio de SL/TP no es precio de entrada


# --- textos ambiguos o no entendidos -> None --------------------------------------------

AMBIGUOUS = [
    "", "   ", "hola", "qué tal el mercado", "long", "short", "compra", "long 2u", "cierra", "close",
    "exit", "cancela la", "stop", "sl HYPE", "tp HYPE", "apalancamiento", "leverage abc", "auto",
    "long HYPE 2",                       # número suelto: ¿unidades? ¿precio?
    "long HYPE y BTC",                   # dos monedas
    "long short HYPE",                   # dos lados
    "long HYPE zona ya",                 # modos contradictorios
    "long HYPE zona @ 85",
    "long HYPE market limit 85",
    "long HYPE @",                       # límite sin precio
    "long HYPE limit",
    "long HYPE sl",                      # SL sin valor
    "long HYPE sl 1% sl 2%",             # SL duplicado
    "long HYPE 2u 3u",                   # unidades duplicadas
    "long HYPE 0u",
    "long HYPE -2u",
    "long HYPE tp 0",
    "long HYPE sl 150%",
    "long HYPE @ 85 @ 86",
    "long HYPE a 83,770",                # ¿83770 o 83,77? ambiguo
    "long HYPE @ 1.000",                 # ¿mil o uno?
    "move stop BTC to 82,000",
    "long HYPE 2u pero no mucho",        # palabras desconocidas
    "long HYPE sl 1.5% tp 3r",           # R múltiplos no caben en una orden
    "cierra HYPE y BTC",
    "cancela HYPE BTC",
    "sl HYPE 84 tp 90",                  # dos acciones a la vez
    "sl HYPE 84 85",
    "tp 4%",                             # el TP por defecto va en R, no en %
    "stop 84.5",                         # ¿precio sin moneda? fuera de rango para sl por defecto
    "tp 90",
    "riesgo 2x",                         # unidad que no corresponde al parámetro
    "apalancamiento 3%",
    "vende todo",                        # ¿cerrar todo? ¿short? -> no adivinar
    "stop all",
    "reset",
    "ok",                                # no confirma: demasiado fácil escribirlo sin querer
    "dale",
    "estado HYPE ya",
]


@pytest.mark.parametrize("text", AMBIGUOUS)
def test_ambiguous_or_unknown_text_returns_none(settings, text):
    assert parse_rules(text, settings) is None
    assert parse(text, settings) is None


# --- piezas sueltas ------------------------------------------------------------------------

@pytest.mark.parametrize("tok,expected", [
    ("85.2", 85.2), ("85,2", 85.2), ("0,5", 0.5), (".5", 0.5), (",5", 0.5), ("2", 2.0),
    ("83770", 83770.0), ("149.985", 149.985), ("0.125", 0.125), ("0,125", 0.125),
    ("1,000,000", 1_000_000.0), ("1.000.000", 1_000_000.0), ("83,770.5", 83770.5),
    ("83.770,5", 83770.5), ("1.5", 1.5), ("10.25", 10.25), ("85.2500", 85.25),
    # ambiguos o mal formados
    ("83,770", None), ("1,500", None), ("1.000", None), ("3.200", None), ("1.5.2", None),
    ("1,5,2", None), ("1.000,5,5", None), ("abc", None), ("", None), ("-2", None), ("1e3", None),
    ("12,34.5", None),
])
def test_parse_number(tok, expected):
    got = parse_number(tok)
    assert got == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize("tok,expected", [
    ("hype", "HYPE"), ("BTC", "BTC"), ("$sol", "SOL"), ("Btc-Perp", "BTC"), ("eth/usdc", "ETH"),
    ("xyz:gold", "xyz:GOLD"), ("XYZ:GOLD", "xyz:GOLD"), ("km:us500", "km:US500"), ("1000pepe", "1000PEPE"),
    ("123", None), ("", None), ("hy pe", None), ("!!", None), ("xyz:", None),
])
def test_normalize_coin(tok, expected):
    assert normalize_coin(tok) == expected


def test_tokenize_splits_suffixes_and_keeps_hip3():
    assert tokenize("Long xyz:GOLD 2u @85,2 SL:1.5% $250 x3") == \
        ["long", "xyz:gold", "2", "u", "@", "85,2", "sl", "1.5", "%", "250", "x", "3"]
    assert tokenize("¿Cómo vamos? por favor") == ["como", "vamos"]
    assert tokenize("stop-loss 2 por ciento") == ["stoploss", "2", "por", "ciento"]


def test_describe_is_readable(settings):
    assert describe(parse_rules("long HYPE 2u zona sl 1.5% tp 4%", settings)) == \
        "abrir LONG HYPE 2u en zona de liquidez SL 1.5% TP 4%"
    assert describe(parse_rules("vende ETH @ 3300 sl 3350", settings)) == "abrir SHORT ETH límite 3300 SL 3350"
    assert describe(parse_rules("stop HYPE 84.5", settings)) == "mover stop-loss de HYPE a 84.5"
    assert describe(parse_rules("tp HYPE 4%", settings)) == "mover take-profit de HYPE a 4% desde la entrada"
    assert describe(parse_rules("cancel all", settings)) == "cancelar todas las órdenes"
    assert describe(parse_rules("apalancamiento 3", settings)) == "cambiar leverage a 3"
    assert "PÁNICO" in describe(parse_rules("kill", settings))


def test_help_text_examples_parse(settings):
    for text in ("long HYPE 2u zona sl 1.5% tp 4%", "cierra HYPE", "cancel all", "stop HYPE 84.5",
                 "sl HYPE 1%", "apalancamiento 3", "auto off", "reset halt"):
        assert text.split()[0] in HELP_TEXT.lower() or text in HELP_TEXT
        assert parse_rules(text, settings) is not None


# --- regresiones (revisión) --------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "cierra todas las posiciones abiertas", "close all open positions", "close all trades",
    "cierra todas las operaciones", "cierra todo ahora mismo", "cierra todo ya",
])
def test_close_everything_with_filler_words_is_close_all(settings, text):
    # Antes una palabra de relleno ("abiertas", "trades", "mismo") se tomaba por moneda: no se cerraba nada.
    cmd = parse_rules(text, settings)
    assert cmd is not None and cmd.action is A.CLOSE_ALL and cmd.coin is None, text


@pytest.mark.parametrize("text", ["cierra todo BTC", "close all SOL", "cancela todo HYPE"])
def test_all_plus_a_coin_is_ambiguous(settings, text):
    assert parse_rules(text, settings) is None


def test_cancel_all_open_orders_with_fillers(settings):
    cmd = parse_rules("cancela todas las ordenes abiertas", settings)
    assert cmd is not None and cmd.action is A.CANCEL and cmd.coin is None


@pytest.mark.parametrize("text", ["panico ya", "pánico ahora mismo", "kill now", "kill todo ya", "panic right now",
                                  "apaga todo ya mismo"])
def test_kill_accepts_urgency_words(settings, text):
    # "pánico ya" no puede caer en la IA y esperar un "si" con las posiciones abiertas.
    cmd = parse_rules(text, settings)
    assert cmd is not None and cmd.action is A.KILL, text


@pytest.mark.parametrize("text", ["reset params", "restablece parámetros", "restaura los parametros por defecto",
                                  "parametros por defecto"])
def test_reset_params(settings, text):
    cmd = parse_rules(text, settings)
    assert cmd is not None and cmd.action is A.SET_PARAM and cmd.param == commands.PARAM_DEFAULTS
    assert cmd.value is None and describe(cmd) == "volver a los parámetros de arranque"


# --- fallback IA: cliente falso ----------------------------------------------------------

class FakeMessages:
    def __init__(self, result):
        self.result = result
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, result):
        self.messages = FakeMessages(result)
        self.beta = SimpleNamespace(messages=self.messages)


def llm_json(**kw) -> dict:
    data = {k: None for k in SCHEMA["required"]}
    data.update(kw)
    return data


def fake_response(payload, stop_reason="end_turn", blocks_before=()):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(stop_reason=stop_reason, stop_details=None,
                           content=[*blocks_before, SimpleNamespace(type="text", text=text)])


@pytest.fixture
def llm_settings(tmp_path):
    return load_settings(env={}, dotenv_path=None, state_dir=str(tmp_path / "state"), llm_enabled=True)


def make_llm(llm_settings, result):
    client = FakeClient(result)
    return LLMCommandInterpreter(llm_settings, client=client), client


def test_llm_success_builds_command_and_sends_expected_request(llm_settings):
    text = "méteme un largo chiquito en hype cerca del soporte, con stop a 84"
    llm, client = make_llm(llm_settings, fake_response(
        llm_json(action="open", coin="hype", side="long", units=1, entry_mode="zone", sl_price=84)))
    cmd = llm.interpret(text)
    assert cmd is not None and cmd.source == "llm" and cmd.raw == text
    assert (cmd.action, cmd.coin, cmd.side, cmd.units, cmd.entry_mode, cmd.sl_price) == \
        (A.OPEN, "HYPE", L, 1.0, M.ZONE, 84.0)
    assert llm.last_error == ""
    (kw,) = client.messages.calls
    assert kw["model"] == llm_settings.llm_model == "claude-opus-5-5"
    assert kw["max_tokens"] == 1024
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["fallbacks"] == "default"
    assert kw["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert kw["system"] == SYSTEM_PROMPT
    assert kw["messages"] == [{"role": "user", "content": text}]
    assert "temperature" not in kw and "thinking" not in kw


def test_llm_schema_mirrors_command():
    props = SCHEMA["properties"]
    assert SCHEMA["type"] == "object" and SCHEMA["additionalProperties"] is False
    assert set(SCHEMA["required"]) == set(props)
    command_fields = {f.name for f in fields(Command)}
    assert set(props) <= command_fields
    assert set(props) == {"action", "coin", "side", "units", "entry_mode", "price", "sl_price", "sl_pct",
                          "tp_price", "tp_pct", "param", "value",
                          "interval", "strategy", "days", "alert"}         # fase 2
    for name, schema in props.items():
        assert {"type": "null"} in schema["anyOf"], name          # todos nullable
    # La API limita los parámetros con unión (anyOf) de un esquema estricto: no crecer sin mirar el límite.
    assert sum("anyOf" in s for s in props.values()) <= 16
    assert props["interval"]["anyOf"][0]["enum"] == list(commands.INTERVALS)
    action_enum = props["action"]["anyOf"][0]["enum"]
    assert set(action_enum) == set(LLM_ACTIONS) and all(A(a) for a in action_enum)
    assert "confirm" not in action_enum and "reset_halt" not in action_enum
    assert set(props["param"]["anyOf"][0]["enum"]) == set(TUNABLE)
    assert json.loads(json.dumps(SCHEMA)) == SCHEMA


def test_llm_refusal_returns_none(llm_settings):
    resp = fake_response(llm_json(action="kill"), stop_reason="refusal")
    resp.stop_details = SimpleNamespace(category="cyber")
    llm, _ = make_llm(llm_settings, resp)
    assert llm.interpret("haz algo raro") is None
    assert "declinó" in llm.last_error


@pytest.mark.parametrize("resp", [
    fake_response("esto no es json"),
    fake_response('{"action": "open", "coin": '),
    fake_response(llm_json(action="open", coin="HYPE", side="long"), stop_reason="max_tokens"),
    SimpleNamespace(stop_reason="end_turn", stop_details=None, content=[]),
    SimpleNamespace(stop_reason="end_turn", stop_details=None, content=None),
    None,
])
def test_llm_unusable_response_returns_none(llm_settings, resp):
    llm, _ = make_llm(llm_settings, resp)
    assert llm.interpret("abre algo en hype") is None
    assert llm.last_error


def test_llm_skips_fallback_blocks_before_text(llm_settings):
    resp = fake_response(llm_json(action="close", coin="eth"),
                         blocks_before=[SimpleNamespace(type="fallback"), SimpleNamespace(type="thinking")])
    llm, _ = make_llm(llm_settings, resp)
    cmd = llm.interpret("sácame de eth")
    assert cmd.action is A.CLOSE and cmd.coin == "ETH" and cmd.source == "llm"


@pytest.mark.parametrize("payload", [
    llm_json(action=None),
    llm_json(action="fly"),
    llm_json(action="confirm"),                       # la IA no confirma
    llm_json(action="reset_halt"),                    # ni levanta halts
    llm_json(action="open", coin="HYPE"),             # sin lado
    llm_json(action="open", coin="HYPE", side="up"),
    llm_json(action="open", side="long"),             # sin moneda
    llm_json(action="open", coin="!!", side="long"),
    llm_json(action="open", coin="HYPE", side="long", units=-1),
    llm_json(action="open", coin="HYPE", side="long", units=0),
    llm_json(action="open", coin="HYPE", side="long", units=True),
    llm_json(action="open", coin="HYPE", side="long", units="2"),
    llm_json(action="open", coin="HYPE", side="long", units=float("inf")),
    llm_json(action="open", coin="HYPE", side="long", entry_mode="limit"),          # límite sin precio
    llm_json(action="open", coin="HYPE", side="long", entry_mode="zone", price=85),  # contradictorio
    llm_json(action="open", coin="HYPE", side="long", entry_mode="instant"),
    llm_json(action="open", coin="HYPE", side="long", sl_pct=150),
    llm_json(action="open", coin="HYPE", side="long", sl_pct=1, sl_price=84),
    llm_json(action="set_sl", price=84),              # sin moneda
    llm_json(action="set_sl", coin="HYPE"),           # sin nivel
    llm_json(action="set_sl", coin="HYPE", price=84, sl_pct=2),
    llm_json(action="set_tp", coin="HYPE", price=-90),
    llm_json(action="close"),
    llm_json(action="set_param", param="color", value=3),
    llm_json(action="set_param", param="leverage"),
    llm_json(action="set_param", param="leverage", value=-3),
    [1, 2, 3],
    "open",
])
def test_llm_invalid_json_content_is_rejected(llm_settings, payload):
    llm, _ = make_llm(llm_settings, fake_response(payload))
    assert llm.interpret("algo raro para la mesa") is None


@pytest.mark.parametrize("payload,expected", [
    (llm_json(action="open", coin="HYPE", side="short", price=90), dict(entry_mode=M.LIMIT, price=90.0)),
    (llm_json(action="open", coin="HYPE", side="short", entry_mode="limit", price=90, tp_pct=3),
     dict(entry_mode=M.LIMIT, price=90.0, tp_pct=3.0)),
    (llm_json(action="open", coin="BTC", side="long"), dict(entry_mode=M.ZONE, units=None)),
    (llm_json(action="open", coin="XYZ:gold", side="long", entry_mode="market"),
     dict(coin="xyz:GOLD", entry_mode=M.MARKET)),
    (llm_json(action="set_sl", coin="hype", sl_price=84.5), dict(action=A.SET_SL, coin="HYPE", price=84.5)),
    (llm_json(action="set_sl", coin="hype", sl_pct=2), dict(action=A.SET_SL, price=None, sl_pct=2.0)),
    (llm_json(action="set_tp", coin="hype", price=95), dict(action=A.SET_TP, price=95.0, tp_pct=None)),
    (llm_json(action="set_tp", coin="hype", tp_pct=5), dict(action=A.SET_TP, price=None, tp_pct=5.0)),
    (llm_json(action="close", coin="btc-perp"), dict(action=A.CLOSE, coin="BTC")),
    (llm_json(action="cancel"), dict(action=A.CANCEL, coin=None)),
    (llm_json(action="cancel", coin="sol"), dict(action=A.CANCEL, coin="SOL")),
    (llm_json(action="set_param", param="leverage", value=3), dict(action=A.SET_PARAM, param="leverage", value=3.0)),
    (llm_json(action="kill", coin="HYPE", side="long"), dict(action=A.KILL, coin=None, side=None)),
    (llm_json(action="status"), dict(action=A.STATUS)),
])
def test_llm_valid_json_content(llm_settings, payload, expected):
    llm, _ = make_llm(llm_settings, fake_response(payload))
    cmd = llm.interpret("lo que sea")
    assert cmd is not None and cmd.source == "llm"
    _assert_fields(cmd, expected)


def test_command_from_llm_direct():
    cmd = command_from_llm(llm_json(action="pause"), raw="frena un rato")
    assert cmd.action is A.PAUSE and cmd.raw == "frena un rato" and cmd.source == "llm"
    assert command_from_llm(None, raw="x") is None
    data = json.loads('{"action": "open", "coin": "HYPE", "side": "long", "units": NaN}')
    assert math.isnan(data["units"]) and command_from_llm(data, "x") is None


def _sdk_errors():
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return {
        "rate_limit": anthropic.RateLimitError("demasiadas", response=httpx2.Response(429, request=req), body=None),
        "status": anthropic.APIStatusError("boom", response=httpx2.Response(500, request=req), body=None),
        "bad_request": anthropic.BadRequestError("mal", response=httpx2.Response(400, request=req), body=None),
        "connection": anthropic.APIConnectionError(request=req),
        "timeout": anthropic.APITimeoutError(request=req),
    }


@pytest.mark.parametrize("kind,needle", [
    ("rate_limit", "429"), ("status", "500"), ("bad_request", "400"), ("connection", "conexión"),
    ("timeout", "conexión"),
])
def test_llm_sdk_errors_return_none(llm_settings, kind, needle):
    llm, client = make_llm(llm_settings, _sdk_errors()[kind])
    assert llm.interpret("abre algo en hype") is None
    assert needle in llm.last_error and len(client.messages.calls) == 1


def test_llm_unexpected_exception_returns_none(llm_settings):
    llm, _ = make_llm(llm_settings, RuntimeError("se rompió"))
    assert llm.interpret("abre algo en hype") is None
    assert "se rompió" in llm.last_error


def test_llm_does_not_send_long_or_empty_text(llm_settings):
    llm, client = make_llm(llm_settings, fake_response(llm_json(action="status")))
    assert llm.interpret("x" * 501) is None and "largo" in llm.last_error
    assert llm.interpret("   ") is None
    assert client.messages.calls == []


def test_llm_disabled_never_calls_client(settings):
    assert settings.llm_enabled is False
    llm, client = make_llm(settings, fake_response(llm_json(action="status")))
    assert not llm.available() and llm.interpret("cómo va la cosa") is None
    assert client.messages.calls == []


def test_llm_availability_needs_sdk_and_credentials(llm_settings, monkeypatch):
    monkeypatch.setattr(commands, "_sdk_installed", lambda: True)
    assert not LLMCommandInterpreter(llm_settings, env={}).available()
    assert LLMCommandInterpreter(llm_settings, env={"ANTHROPIC_API_KEY": "k"}).available()
    monkeypatch.setattr(commands, "_sdk_installed", lambda: False)
    assert not LLMCommandInterpreter(llm_settings, env={"ANTHROPIC_API_KEY": "k"}).available()
    assert LLMCommandInterpreter(llm_settings, client=FakeClient(None)).available()   # inyectado
    # sin cliente ni credenciales: no intenta crear el cliente
    assert LLMCommandInterpreter(llm_settings, env={}).interpret("algo") is None


def test_has_anthropic_credentials(tmp_path):
    assert not has_anthropic_credentials({})
    assert has_anthropic_credentials({"ANTHROPIC_API_KEY": "sk-ant-x"})
    assert has_anthropic_credentials({"ANTHROPIC_AUTH_TOKEN": "tok"})
    assert has_anthropic_credentials({"ANTHROPIC_PROFILE": "dev"})
    assert not has_anthropic_credentials({"ANTHROPIC_API_KEY": ""})
    assert not has_anthropic_credentials({"ANTHROPIC_FEDERATION_RULE_ID": "r"})
    assert has_anthropic_credentials({"ANTHROPIC_FEDERATION_RULE_ID": "r", "ANTHROPIC_ORGANIZATION_ID": "o"})
    assert not has_anthropic_credentials({"HOME": str(tmp_path)})
    configs = tmp_path / ".config" / "anthropic" / "configs"
    configs.mkdir(parents=True)
    (configs / "default.json").write_text("{}")
    assert has_anthropic_credentials({"HOME": str(tmp_path)})


# --- parse() / CommandParser con fallback --------------------------------------------------

class FakeInterpreter:
    def __init__(self, result=None, exc: BaseException | None = None):
        self.result, self.exc, self.calls = result, exc, []

    def interpret(self, text):
        self.calls.append(text)
        if self.exc:
            raise self.exc
        return self.result


def test_parse_uses_rules_first_and_llm_only_as_fallback(llm_settings):
    llm_cmd = Command(A.STATUS, raw="¿qué onda la cuenta?", source="llm")
    llm = FakeInterpreter(llm_cmd)
    cmd = parse("long HYPE 2u", llm_settings, llm)
    assert cmd.action is A.OPEN and cmd.source == "regex" and llm.calls == []
    assert parse("¿qué onda la cuenta?", llm_settings, llm) is llm_cmd
    assert llm.calls == ["¿qué onda la cuenta?"]
    assert parse("", llm_settings, llm) is None and len(llm.calls) == 1


def test_parse_respects_llm_enabled_flag(settings):
    llm = FakeInterpreter(Command(A.STATUS, source="llm"))
    assert parse("¿qué onda la cuenta?", settings, llm) is None
    assert llm.calls == []


def test_parse_never_raises_when_llm_fails(llm_settings):
    llm = FakeInterpreter(exc=RuntimeError("se cayó"))
    assert parse("¿qué onda la cuenta?", llm_settings, llm) is None
    assert parse("estado", llm_settings, llm).action is A.STATUS


def test_command_parser_wiring(settings, llm_settings):
    p = CommandParser(settings)
    assert isinstance(p.llm, LLMCommandInterpreter)
    assert p.parse("cierra todo").action is A.CLOSE_ALL
    assert p("hola") is None                          # llm desactivado en el fixture: no llama a nadie
    assert CommandParser(settings, use_llm=False).llm is None
    fake = FakeInterpreter(Command(A.PAUSE, source="llm"))
    p2 = CommandParser(llm_settings, llm=fake)
    assert p2.parse("frena un poco").action is A.PAUSE and fake.calls == ["frena un poco"]
    # con un cliente falso de extremo a extremo
    llm, client = make_llm(llm_settings, fake_response(llm_json(action="close_all")))
    assert CommandParser(llm_settings, llm=llm).parse("sácame de todo ya mismo").action is A.CLOSE_ALL
    assert len(client.messages.calls) == 1


# ======================================================================================
# Fase 2: backtest por gráfica, playbook, alarmas y silencio
# ======================================================================================

CHART_FIELDS = ("coin", "interval", "days", "strategy")


def _phase2(text, settings, action, **expected) -> Command:
    cmd = parse_rules(text, settings)
    assert cmd is not None, text
    assert cmd.action is action and cmd.source == "regex" and cmd.raw == text.strip(), (text, cmd)
    _assert_fields(cmd, expected)
    assert cmd.side is None and cmd.units is None and cmd.price is None    # nada de órdenes
    return cmd


BACKTEST_CASES = [
    # ejemplos del contrato
    ("backtest HYPE 1h", dict(coin="HYPE", interval="1h")),
    ("backtest BTC 15m 90d", dict(coin="BTC", interval="15m", days=90.0)),
    ("backtest SOL 4h 180 dias", dict(coin="SOL", interval="4h", days=180.0)),
    ("backtest SOL 4h 180 días", dict(coin="SOL", interval="4h", days=180.0)),
    ("optimiza ETH 1h", dict(coin="ETH", interval="1h")),
    ("prueba HYPE 1h ema_trend", dict(coin="HYPE", interval="1h", strategy="ema_trend")),
    ("backtest HYPE", dict(coin="HYPE")),                                  # temporalidad por defecto de la mesa
    # duraciones de la historia
    ("backtest HYPE 1h 3 meses", dict(coin="HYPE", interval="1h", days=90.0)),
    ("backtest HYPE 1h 1 año", dict(coin="HYPE", interval="1h", days=365.0)),
    ("backtest HYPE 4h 2 años", dict(coin="HYPE", interval="4h", days=730.0)),
    ("backtest HYPE 1h un año", dict(coin="HYPE", interval="1h", days=365.0)),
    ("backtest HYPE 1h 1y", dict(coin="HYPE", interval="1h", days=365.0)),
    ("backtest HYPE 1h 6 months", dict(coin="HYPE", interval="1h", days=180.0)),
    ("backtest HYPE 1h 2 semanas", dict(coin="HYPE", interval="1h", days=14.0)),
    ("backtest HYPE 1h 1,5 meses", dict(coin="HYPE", interval="1h", days=45.0)),
    ("backtest HYPE 1h 90 d", dict(coin="HYPE", interval="1h", days=90.0)),
    ("Backtest hype 1H 90D", dict(coin="HYPE", interval="1h", days=90.0)),
    ("backtest HYPE 90d", dict(coin="HYPE", days=90.0)),                    # sólo historia
    # "1d"/"3d": temporalidad salvo que ya haya otra
    ("backtest btc 1d", dict(coin="BTC", interval="1d")),
    ("backtest btc 3d", dict(coin="BTC", interval="3d")),
    ("backtest btc 1d 365d", dict(coin="BTC", interval="1d", days=365.0)),
    ("backtest btc 365d 1d", dict(coin="BTC", interval="1d", days=365.0)),
    ("backtest btc 1h 3d", dict(coin="BTC", interval="1h", days=3.0)),
    ("backtest btc 90 dias 1d", dict(coin="BTC", interval="1d", days=90.0)),
    ("backtest btc 1w", dict(coin="BTC", interval="1w")),
    ("backtest btc 1 week", dict(coin="BTC", days=7.0)),
    # temporalidades con palabras
    ("backtest de BTC en 15 minutos", dict(coin="BTC", interval="15m")),
    ("backtest ETH 4 horas", dict(coin="ETH", interval="4h")),
    ("optimize SOL 15min last 30 days", dict(coin="SOL", interval="15m", days=30.0)),
    # verbos y relleno
    ("haz un backtest de HYPE en 1h con los ultimos 90 dias", dict(coin="HYPE", interval="1h", days=90.0)),
    ("bt HYPE 1h", dict(coin="HYPE", interval="1h")),
    ("optimizar BTC 4h", dict(coin="BTC", interval="4h")),
    ("test ETH 1h", dict(coin="ETH", interval="1h")),
    ("backtest HYPE 1h todas las estrategias", dict(coin="HYPE", interval="1h", strategy=None)),
    ("backtest 1h HYPE", dict(coin="HYPE", interval="1h")),                 # el orden no importa
    # monedas
    ("backtest $HYPE 1h", dict(coin="HYPE", interval="1h")),
    ("backtest HYPE-PERP 1h", dict(coin="HYPE", interval="1h")),
    ("backtest xyz:GOLD 1h", dict(coin="xyz:GOLD", interval="1h")),
    ("backtest W 1h", dict(coin="W", interval="1h")),                       # "W" suelto es moneda, no semana
    ("backtest ME 1h", dict(coin="ME", interval="1h")),
    ("backtest ID 4h", dict(coin="ID", interval="4h")),
    # estrategias: nombre canónico, con espacios/guiones o una palabra que sólo es suya
    ("prueba la estrategia rsi_reversion en ETH 15m", dict(coin="ETH", interval="15m", strategy="rsi_reversion")),
    ("prueba HYPE 1h ema trend", dict(coin="HYPE", interval="1h", strategy="ema_trend")),
    ("prueba HYPE 1h EMA-Trend", dict(coin="HYPE", interval="1h", strategy="ema_trend")),
    ("prueba HYPE 1h donchian", dict(coin="HYPE", interval="1h", strategy="donchian_breakout")),
    ("prueba HYPE 1h squeeze", dict(coin="HYPE", interval="1h", strategy="squeeze_breakout")),
    ("prueba HYPE 15m liquidity_sweep", dict(coin="HYPE", interval="15m", strategy="liquidity_sweep")),
    ("backtest HYPE 4h donchian_breakout 1 año",
     dict(coin="HYPE", interval="4h", days=365.0, strategy="donchian_breakout")),
    ("backtest ema_trend HYPE 90d 1h", dict(coin="HYPE", interval="1h", days=90.0, strategy="ema_trend")),
]


@pytest.mark.parametrize("text,expected", BACKTEST_CASES, ids=[c[0] for c in BACKTEST_CASES])
def test_backtest_commands(settings, text, expected):
    full = {f: None for f in CHART_FIELDS}
    full.update(expected)
    _phase2(text, settings, A.BACKTEST, **full)


@pytest.mark.parametrize("interval", list(commands.INTERVALS))
def test_every_exchange_interval_is_a_valid_chart(settings, interval):
    cmd = parse_rules(f"backtest BTC {interval}", settings)
    assert cmd is not None and cmd.interval == interval and cmd.days is None
    assert parse_rules(f"activa BTC {interval}", settings).interval == interval


def test_strategy_names_cover_registry_and_known_names():
    from desk.strategy import STRATEGIES
    names = commands.strategy_names()
    assert set(STRATEGIES) <= set(names) and set(commands.KNOWN_STRATEGIES) <= set(names)
    assert {"ema_trend", "rsi_reversion", "donchian_breakout"} <= set(names)


@pytest.mark.parametrize("name", commands.strategy_names())
def test_every_strategy_name_parses(settings, name):
    cmd = parse_rules(f"prueba HYPE 1h {name}", settings)
    assert cmd is not None and cmd.action is A.BACKTEST and cmd.strategy == name
    assert parse_rules(f"activa HYPE 1h {name}", settings).strategy == name
    assert commands.strategy_key(name.upper().replace("_", " ")) == name


def test_strategy_registered_later_is_accepted(settings, monkeypatch):
    import desk.strategy as strategy_mod
    monkeypatch.setitem(strategy_mod.STRATEGIES, "vwap_bounce", strategy_mod.STRATEGIES["ema_trend"])
    assert parse_rules("prueba HYPE 1h vwap_bounce", settings).strategy == "vwap_bounce"
    assert commands.strategy_key("macd") is None and commands.strategy_key("breakout") is None   # ambigua


@pytest.mark.parametrize("text,expected", [
    ("1h", "1h"), ("1H", "1h"), ("15m", "15m"), ("15 min", "15m"), ("4 horas", "4h"), ("1 dia", "1d"),
    ("1 día", "1d"), ("1w", "1w"), ("una hora", "1h"), ("7h", None), ("60m", None), ("90d", None), ("1,5h", None),
    ("hype", None), ("", None), ("1h 4h", None),
])
def test_parse_interval(text, expected):
    assert commands.parse_interval(text) == expected


ADOPT_CASES = [
    ("activa HYPE 1h", dict(coin="HYPE", interval="1h")),
    ("usa la mejor HYPE 1h", dict(coin="HYPE", interval="1h")),
    ("activa HYPE 1h ema_trend", dict(coin="HYPE", interval="1h", strategy="ema_trend")),
    ("adopt HYPE 1h", dict(coin="HYPE", interval="1h")),
    ("adopta BTC 4h", dict(coin="BTC", interval="4h")),
    ("usa la mejor estrategia en SOL 15m", dict(coin="SOL", interval="15m")),
    ("activa la ganadora de ETH 1h en automático", dict(coin="ETH", interval="1h")),
    ("usa ema_trend en HYPE 1h", dict(coin="HYPE", interval="1h", strategy="ema_trend")),
    ("activa HYPE 1 hora", dict(coin="HYPE", interval="1h")),
    ("activa HYPE 1d", dict(coin="HYPE", interval="1d")),
    ("enable BTC 4h rsi_reversion", dict(coin="BTC", interval="4h", strategy="rsi_reversion")),
    ("aplica la mejor a xyz:GOLD 1h", dict(coin="xyz:GOLD", interval="1h")),
]


@pytest.mark.parametrize("text,expected", ADOPT_CASES, ids=[c[0] for c in ADOPT_CASES])
def test_adopt_commands(settings, text, expected):
    full = {f: None for f in CHART_FIELDS}
    full.update(expected)
    _phase2(text, settings, A.ADOPT, **full)


DEACTIVATE_CASES = [
    ("desactiva HYPE 1h", dict(coin="HYPE", interval="1h")),
    ("desactiva HYPE", dict(coin="HYPE")),                                # todas sus temporalidades
    ("deactivate BTC 4h", dict(coin="BTC", interval="4h")),
    ("disable ETH 15m", dict(coin="ETH", interval="15m")),
    ("desactiva la gráfica de HYPE 1h", dict(coin="HYPE", interval="1h")),
    ("desactiva HYPE 1d", dict(coin="HYPE", interval="1d")),
    ("desactiva HYPE en todas las temporalidades", dict(coin="HYPE")),
    ("quita HYPE 1h del playbook", dict(coin="HYPE", interval="1h")),
    ("saca SOL del playbook", dict(coin="SOL")),
    ("remove ETH 4h from playbook", dict(coin="ETH", interval="4h")),
]


@pytest.mark.parametrize("text,expected", DEACTIVATE_CASES, ids=[c[0] for c in DEACTIVATE_CASES])
def test_deactivate_commands(settings, text, expected):
    full = {f: None for f in CHART_FIELDS}
    full.update(expected)
    _phase2(text, settings, A.DEACTIVATE, **full)


SIMPLE_PHASE2 = [
    ("playbook", A.PLAYBOOK), ("ver playbook", A.PLAYBOOK), ("mi playbook", A.PLAYBOOK),
    ("gráficas activas", A.PLAYBOOK), ("graficas en automatico", A.PLAYBOOK), ("estrategias activas", A.PLAYBOOK),
    ("show playbook", A.PLAYBOOK), ("active strategies", A.PLAYBOOK), ("muestra las gráficas activas", A.PLAYBOOK),
    ("informes", A.REPORTS), ("reports", A.REPORTS), ("últimos backtests", A.REPORTS), ("ultimos informes", A.REPORTS),
    ("mis informes", A.REPORTS), ("backtests", A.REPORTS), ("latest reports", A.REPORTS), ("ver informes", A.REPORTS),
    ("informes de backtest", A.REPORTS),
    ("alertas", A.ALERT_LIST), ("alarmas", A.ALERT_LIST), ("alerts", A.ALERT_LIST), ("mis alertas", A.ALERT_LIST),
    ("ver alarmas", A.ALERT_LIST), ("alertas activas", A.ALERT_LIST), ("¿qué alertas hay?", A.ALERT_LIST),
    ("list alerts", A.ALERT_LIST), ("Alarmas", A.ALERT_LIST), ("avisos", A.ALERT_LIST),
    ("sonido", A.UNMUTE), ("unmute", A.UNMUTE), ("activa avisos", A.UNMUTE), ("activa las alertas", A.UNMUTE),
    ("reactiva los avisos", A.UNMUTE), ("enciende el sonido", A.UNMUTE), ("enable alerts", A.UNMUTE),
    ("turn on notifications", A.UNMUTE), ("quita el silencio", A.UNMUTE), ("desilencia", A.UNMUTE),
    ("sonido on", A.UNMUTE), ("activa las alarmas de nuevo", A.UNMUTE),
]


@pytest.mark.parametrize("text,action", SIMPLE_PHASE2, ids=[c[0] for c in SIMPLE_PHASE2])
def test_phase2_simple_commands(settings, text, action):
    cmd = _phase2(text, settings, action)
    assert cmd.coin is None and cmd.alert is None and cmd.target_id is None and cmd.value is None


def _spec(kind, interval=None, repeat=False, **params) -> dict:
    return {"kind": kind, **params, "interval": interval, "repeat": repeat}


ALERT_CASES = [
    # ejemplos del contrato
    ("alerta HYPE > 90", "HYPE", _spec("price_above", level=90.0)),
    ("alerta BTC < 80000", "BTC", _spec("price_below", level=80000.0)),
    ("alerta SOL cruza 150", "SOL", _spec("price_cross", level=150.0)),
    ("alerta HYPE rsi < 30 1h", "HYPE", _spec("rsi_below", "1h", level=30.0)),
    ("alerta ETH rsi > 70 4h", "ETH", _spec("rsi_above", "4h", level=70.0)),
    ("alerta BTC cruce ema 20 50 1h", "BTC", _spec("ma_cross", "1h", fast=20, slow=50, ma="ema")),
    ("alerta BTC cruce sma 50 200 1d alcista", "BTC",
     _spec("ma_cross", "1d", fast=50, slow=200, ma="sma", direction="up")),
    ("alerta BTC cruce sma 50 200 1d bajista", "BTC",
     _spec("ma_cross", "1d", fast=50, slow=200, ma="sma", direction="down")),
    ("alerta SOL zona", "SOL", _spec("zone")),
    ("alerta SOL zona long", "SOL", _spec("zone", side="long")),
    ("alerta BTC volumen 3x 15m", "BTC", _spec("volume_spike", "15m", mult=3.0)),
    ("alerta HYPE mueve 3% 1h", "HYPE", _spec("move_pct", "1h", pct=3.0)),
    # repetir
    ("alerta HYPE > 90 siempre", "HYPE", _spec("price_above", repeat=True, level=90.0)),
    ("alerta HYPE > 90 cada vez", "HYPE", _spec("price_above", repeat=True, level=90.0)),
    ("alerta HYPE > 90 repeat", "HYPE", _spec("price_above", repeat=True, level=90.0)),
    ("alert BTC above 100000 always", "BTC", _spec("price_above", repeat=True, level=100000.0)),
    ("alerta HYPE rsi < 30 1h siempre", "HYPE", _spec("rsi_below", "1h", repeat=True, level=30.0)),
    ("alerta HYPE > 90 una vez", "HYPE", _spec("price_above", level=90.0)),
    # sinónimos y formas de decirlo
    ("alarma HYPE > 90", "HYPE", _spec("price_above", level=90.0)),
    ("ALARMA hype > 90 SIEMPRE", "HYPE", _spec("price_above", repeat=True, level=90.0)),
    ("alerta HYPE>90", "HYPE", _spec("price_above", level=90.0)),
    ("alerta HYPE >= 90", "HYPE", _spec("price_above", level=90.0)),
    ("alerta HYPE <= 80", "HYPE", _spec("price_below", level=80.0)),
    ("alerta HYPE por encima de 90", "HYPE", _spec("price_above", level=90.0)),
    ("alerta HYPE por debajo de 80,5", "HYPE", _spec("price_below", level=80.5)),
    ("alerta HYPE supera 95", "HYPE", _spec("price_above", level=95.0)),
    ("alerta HYPE baja de 80", "HYPE", _spec("price_below", level=80.0)),
    ("alerta HYPE más de 90", "HYPE", _spec("price_above", level=90.0)),
    ("alerta HYPE 90", "HYPE", _spec("price_cross", level=90.0)),           # nivel sin sentido: cualquiera
    ("alerta HYPE toca 90", "HYPE", _spec("price_cross", level=90.0)),
    ("alerta SOL cruza 150 al alza", "SOL", _spec("price_above", level=150.0)),
    ("alerta SOL cruza 150 a la baja", "SOL", _spec("price_below", level=150.0)),
    ("avísame si BTC < 80000", "BTC", _spec("price_below", level=80000.0)),
    ("avisame cuando ETH supere 4000", "ETH", _spec("price_above", level=4000.0)),
    ("pon una alerta en HYPE > 90", "HYPE", _spec("price_above", level=90.0)),
    ("nueva alarma SOL < 120", "SOL", _spec("price_below", level=120.0)),
    ("alert me when ETH < 3000", "ETH", _spec("price_below", level=3000.0)),
    # números a la española / inglesa (como el resto del parser)
    ("alerta BTC < 80.000,5", "BTC", _spec("price_below", level=80000.5)),
    ("alerta BTC < 1.000.000", "BTC", _spec("price_below", level=1_000_000.0)),
    ("alerta BTC > 100,000.5", "BTC", _spec("price_above", level=100000.5)),
    ("alerta $BTC > 100000$", "BTC", _spec("price_above", level=100000.0)),
    # RSI con periodo
    ("alerta HYPE rsi(14) < 30 1h", "HYPE", _spec("rsi_below", "1h", level=30.0, period=14)),
    ("alerta HYPE rsi 7 > 80 15m", "HYPE", _spec("rsi_above", "15m", level=80.0, period=7)),
    ("alerta HYPE rsi14 < 25", "HYPE", _spec("rsi_below", level=25.0, period=14)),
    ("alerta ETH rsi < 30", "ETH", _spec("rsi_below", level=30.0)),          # temporalidad: la de la mesa de alarmas
    ("alerta ETH rsi por encima de 70 en velas de 4h", "ETH", _spec("rsi_above", "4h", level=70.0)),
    ("alerta ETH rsi < 30 en 1 hora", "ETH", _spec("rsi_below", "1h", level=30.0)),
    # cruces de medias
    ("alerta BTC cruce ema 50 20 4h", "BTC", _spec("ma_cross", "4h", fast=20, slow=50, ma="ema")),
    ("alerta BTC ema20 cruza ema50 1h", "BTC", _spec("ma_cross", "1h", fast=20, slow=50, ma="ema")),
    ("alerta BTC golden cross 1d", "BTC", _spec("ma_cross", "1d", fast=50, slow=200, ma="sma", direction="up")),
    ("alerta BTC cruce de la muerte 1d", "BTC",
     _spec("ma_cross", "1d", fast=50, slow=200, ma="sma", direction="down")),
    ("alerta BTC cruce de medias 1h", "BTC", _spec("ma_cross", "1h")),
    ("alerta BTC cruce ema 9 21 15m alcista siempre", "BTC",
     _spec("ma_cross", "15m", repeat=True, fast=9, slow=21, ma="ema", direction="up")),
    # zona
    ("alerta SOL zona short", "SOL", _spec("zone", side="short")),
    ("alerta SOL zona de liquidez long", "SOL", _spec("zone", side="long")),
    ("alerta SOL zona de compra", "SOL", _spec("zone", side="long")),
    ("alerta SOL zona de venta", "SOL", _spec("zone", side="short")),
    ("alerta SOL toca zona short", "SOL", _spec("zone", side="short")),
    # volumen
    ("alerta BTC volumen 3 veces 15m", "BTC", _spec("volume_spike", "15m", mult=3.0)),
    ("alerta BTC volumen x2.5 1h", "BTC", _spec("volume_spike", "1h", mult=2.5)),
    ("alerta BTC volumen 1h", "BTC", _spec("volume_spike", "1h")),
    ("alerta BTC pico de volumen > 4x 5m", "BTC", _spec("volume_spike", "5m", mult=4.0)),
    # movimiento
    ("alerta HYPE sube 5% 4h", "HYPE", _spec("move_pct", "4h", pct=5.0, direction="up")),
    ("alerta HYPE cae 3% 1h", "HYPE", _spec("move_pct", "1h", pct=3.0, direction="down")),
    ("alerta HYPE 3% 1h", "HYPE", _spec("move_pct", "1h", pct=3.0)),
    ("alerta HYPE se mueve 3 por ciento en 1 hora", "HYPE", _spec("move_pct", "1h", pct=3.0)),
    ("alerta HYPE mueve 2,5% 15m", "HYPE", _spec("move_pct", "15m", pct=2.5)),
    # monedas raras
    ("alert HYPE move 5% 1h", "HYPE", _spec("move_pct", "1h", pct=5.0)),
    ("alerta W < 0.1", "W", _spec("price_below", level=0.1)),               # "W" suelto es moneda, no semana
    ("alerta ME > 1.5", "ME", _spec("price_above", level=1.5)),             # ME es moneda...
    ("alert me when ME > 1.5", "ME", _spec("price_above", level=1.5)),
    ("alerta ID < 0.2", "ID", _spec("price_below", level=0.2)),
    ("alerta xyz:GOLD > 3000", "xyz:GOLD", _spec("price_above", level=3000.0)),
]


@pytest.mark.parametrize("text,coin,spec", ALERT_CASES, ids=[c[0] for c in ALERT_CASES])
def test_alert_add_commands(settings, text, coin, spec):
    cmd = _phase2(text, settings, A.ALERT_ADD, coin=coin)
    assert cmd.alert == spec, (text, cmd.alert)
    assert cmd.interval is None and cmd.target_id is None                  # la temporalidad va en el spec


@pytest.mark.parametrize("text,coin,spec", ALERT_CASES, ids=[c[0] for c in ALERT_CASES])
def test_parsed_alerts_are_accepted_by_alert_manager(tmp_path, settings, clock, text, coin, spec):
    from desk.alerts import AlertManager
    mgr = AlertManager(settings, clock, tmp_path / "alerts")
    cmd = parse_rules(text, settings)
    rule = mgr.add(cmd.coin, cmd.alert, ref_px=100.0)                       # contrato: add(coin, Command.alert)
    assert rule.kind == spec["kind"] and rule.repeat is spec["repeat"]
    if spec["kind"] in commands.CANDLE_ALERT_KINDS:
        assert rule.interval == (spec["interval"] or "1h")                  # sin temporalidad: la de las alarmas


def test_alert_kinds_mirror_alert_manager():
    from desk.alerts import CANDLE_KINDS, KINDS
    assert set(commands.ALERT_KINDS) == set(KINDS)
    assert commands.CANDLE_ALERT_KINDS == CANDLE_KINDS


def test_alert_ranges_are_left_to_alert_manager(tmp_path, settings, clock):
    # Como con los parámetros: el parser no adivina límites; AlertManager responde con un motivo claro.
    from desk.alerts import AlertManager
    mgr = AlertManager(settings, clock, tmp_path / "alerts")
    for text, needle in (("alerta HYPE rsi > 150 1h", "RSI"), ("alerta BTC cruce ema 20 20 1h", "rápida")):
        cmd = parse_rules(text, settings)
        assert cmd is not None and cmd.action is A.ALERT_ADD, text
        with pytest.raises(ValueError, match=needle):
            mgr.add(cmd.coin, cmd.alert)


ALERT_DEL_CASES = [
    ("borra alerta 3", 3), ("elimina alarma 3", 3), ("borra la alerta #3", 3), ("borra alerta #12", 12),
    ("delete alert 12", 12), ("remove alert 2", 2), ("cancela la alarma 5", 5), ("borrar alerta numero 4", 4),
    ("quita la alerta 7", 7),
    ("borra alertas", None), ("borra todas las alertas", None), ("elimina las alarmas", None),
    ("delete all alerts", None), ("clear alerts", None), ("borra todas las alarmas activas", None),
    ("cancela todas las alertas", None),
]


@pytest.mark.parametrize("text,target", ALERT_DEL_CASES, ids=[c[0] for c in ALERT_DEL_CASES])
def test_alert_del_commands(settings, text, target):
    _phase2(text, settings, A.ALERT_DEL, target_id=target, coin=None)


MUTE_CASES = [
    ("silencio 30m", 30.0), ("silencia 2h", 120.0), ("mute 15", 15.0), ("silencio", None), ("mute", None),
    ("silencio 1 hora", 60.0), ("silencia los avisos 45 min", 45.0), ("silencio por 30 minutos", 30.0),
    ("mute alerts for 2 hours", 120.0), ("silencia 1d", 1440.0), ("silencio 1.5h", 90.0), ("silencio media hora", 30.0),
    ("mute 1 día", 1440.0), ("silencia todo 2h", 120.0), ("silencia las alarmas", None),
    ("desactiva avisos", None), ("desactiva las alertas 1h", 60.0), ("apaga las notificaciones", None),
    ("sin sonido", None), ("sonido off", None), ("turn off alerts", None), ("disable notifications 30 min", 30.0),
]


@pytest.mark.parametrize("text,minutes", MUTE_CASES, ids=[c[0] for c in MUTE_CASES])
def test_mute_commands(settings, text, minutes):
    _phase2(text, settings, A.MUTE, value=minutes)


PHASE2_INVALID = [
    # backtest / gráficas
    "backtest", "backtest 1h", "backtest HYPE 7h", "backtest HYPE 60m", "backtest HYPE 1h 4h", "backtest HYPE 90",
    "backtest HYPE BTC 1h", "backtest HYPE 1h macd", "backtest HYPE 1h 0d", "backtest HYPE 1h 90d 30d",
    "backtest HYPE 1h breakout",                      # palabra de dos estrategias: ambigua
    "backtest HYPE 1h ema_trend rsi_reversion", "backtest HYPE 1h 3%", "backtest HYPE 1h 0,5 dias",
    "backtest HYPE 1h 1d 3d", "backtest HYPE long 1h", "backtest HYPE 1h ya", "prueba",
    "activa HYPE", "adopt HYPE",                      # dinero en automático: hace falta la gráfica completa
    "activa HYPE 1h 90d", "activa 1h", "activa HYPE BTC 1h", "usa la mejor", "activa HYPE 1h macd",
    "desactiva", "desactiva todo", "desactiva HYPE 1h ema_trend", "desactiva HYPE 90d", "desactiva el bot",
    "quita del playbook",
    # alarmas
    "alerta", "alerta HYPE", "alerta > 90", "alerta HYPE > 90 < 80", "alerta HYPE > 90 95", "alerta HYPE > 90 1h",
    "alerta HYPE rsi 30", "alerta HYPE rsi cruza 30", "alerta HYPE rsi < 30 1h 4h", "alerta HYPE rsi < 30 7h",
    "alerta HYPE rsi < 30 50 70", "alerta HYPE zona 3", "alerta HYPE zona 1h", "alerta HYPE zona long short",
    "alerta HYPE long > 90", "alerta HYPE cruce ema 20 1h", "alerta HYPE cruce ema sma 20 50",
    "alerta HYPE cruce ema 20 50 > 3", "alerta HYPE volumen 3 15m", "alerta HYPE volumen 3x 2x",
    "alerta HYPE volumen baja 3x", "alerta HYPE mueve 3", "alerta HYPE mueve 3% 7h", "alerta HYPE mueve 3% 2%",
    "alerta HYPE mueve 3% cruza", "alerta HYPE BTC > 90", "alerta HYPE > 0", "alerta HYPE > -5",
    "alerta BTC < 80.000",                            # ¿80 mil u 80? ambiguo, como en las órdenes
    "alerta BTC < 83,770", "alerta HYPE > 90 siempre una vez", "alerta HYPE rsi < 30 zona", "alerta HYPE 3x",
    "alerta HYPE x3 > 90", "alerta HYPE > 90 ya", "alerta HYPE > 90 por favor ahora", "alerta HYPE precio",
    "alertas HYPE", "alerta HYPE rsi > 70 30%",
    "alerta MOVE > 0.5",                              # "move" es palabra reservada (como en "long MOVE")
    # borrar alarmas
    "borra alerta", "borra alerta 0", "borra alerta 3.5", "borra todas las alertas 3", "borra alertas de HYPE",
    "borra alerta 3 4", "delete alert abc",
    # silencio
    "silencio 0", "silencio 30%", "silencio HYPE", "silencio 3 meses", "silencio 1h 30m", "mute 2x",
    "desactiva alertas 30",                           # sin unidad tras "desactiva": ¿minutos? ¿la alarma 30?
    "desactiva la alerta 3", "silencia la alerta 3", "sonido alto", "activa avisos HYPE", "unmute 30",
]


@pytest.mark.parametrize("text", PHASE2_INVALID)
def test_phase2_invalid_or_ambiguous_returns_none(settings, text):
    assert parse_rules(text, settings) is None, (text, parse_rules(text, settings))
    assert parse(text, settings) is None


@pytest.mark.parametrize("text,action,coin", [
    # "activa ..." / "desactiva ...": avisos, modo automático y gráficas no se pisan
    ("activa avisos", A.UNMUTE, None),
    ("activa HYPE 1h", A.ADOPT, "HYPE"),
    ("activa el modo automático", A.AUTO_ON, None),
    ("activa auto", A.AUTO_ON, None),
    ("activa el auto", A.AUTO_ON, None),
    ("auto on", A.AUTO_ON, None),
    ("desactiva auto", A.AUTO_OFF, None),
    ("desactiva el modo automatico", A.AUTO_OFF, None),
    ("desactiva avisos", A.MUTE, None),
    ("desactiva HYPE", A.DEACTIVATE, "HYPE"),
    ("enable auto", A.AUTO_ON, None),
    ("enable alerts", A.UNMUTE, None),
    ("reactiva", A.RESUME, None),
    ("reactiva avisos", A.UNMUTE, None),
    ("apaga todo", A.KILL, None),
    ("apaga avisos", A.MUTE, None),
    ("apaga el auto", A.AUTO_OFF, None),
    # cerrar / cancelar siguen siendo de posiciones y órdenes
    ("cierra HYPE", A.CLOSE, "HYPE"),
    ("cierra todo", A.CLOSE_ALL, None),
    ("cancela HYPE", A.CANCEL, "HYPE"),
    ("cancel all", A.CANCEL, None),
    ("cancela", A.REJECT, None),
    ("cancela alerta 3", A.ALERT_DEL, None),
    # informe (singular) = estado; informes = backtests
    ("informe", A.STATUS, None),
    ("report", A.STATUS, None),
    ("informes", A.REPORTS, None),
    ("borra parámetros", A.SET_PARAM, None),
    ("borra alertas", A.ALERT_DEL, None),
    ("quita el halt", A.RESET_HALT, None),
    ("quita el silencio", A.UNMUTE, None),
    ("set leverage to 5", A.SET_PARAM, None),
    ("pon el apalancamiento a 3", A.SET_PARAM, None),
    ("pon una alerta en BTC > 100000", A.ALERT_ADD, "BTC"),
    ("pon un short en BTC", A.OPEN, "BTC"),
])
def test_phase2_does_not_steal_existing_commands(settings, text, action, coin):
    cmd = parse_rules(text, settings)
    assert cmd is not None and cmd.action is action, (text, cmd)
    assert cmd.coin == coin


@pytest.mark.parametrize("text", ["cierra alertas", "cierra playbook", "cancela backtest", "long alerta",
                                  "close alerts", "cierra avisos", "backtest durante 1h", "activa nueva 1h",
                                  "desactiva cuando", "backtest hazme", "alerta activas 3", "desactiva longs"])
def test_phase2_nouns_are_never_coins(settings, text):
    # Antes "cierra alertas" era CLOSE de la moneda ALERTAS.
    assert parse_rules(text, settings) is None


def test_phase2_command_roundtrips_through_dict(settings):
    cmd = parse_rules("alerta BTC cruce sma 50 200 1d alcista siempre", settings)
    back = Command.from_dict(json.loads(json.dumps(cmd.to_dict())))
    assert back == cmd and back.alert["direction"] == "up"
    cmd = parse_rules("backtest HYPE 1h 90d ema_trend", settings)
    assert Command.from_dict(cmd.to_dict()) == cmd


@pytest.mark.parametrize("text,expected", [
    ("backtest BTC 15m 90d", "backtest + optimización walk-forward de BTC 15m (90 días, todas las estrategias)"),
    ("prueba HYPE 1h ema_trend",
     "backtest + optimización walk-forward de HYPE 1h (historia por defecto, estrategia ema_trend)"),
    ("backtest HYPE", "backtest + optimización walk-forward de HYPE (temporalidad por defecto) "
                      "(historia por defecto, todas las estrategias)"),
    ("activa HYPE 1h", "operar en automático HYPE 1h con la ganadora de su último backtest"),
    ("activa HYPE 1h ema_trend", "operar en automático HYPE 1h con la estrategia ema_trend"),
    ("desactiva HYPE 1h", "quitar HYPE 1h del playbook"),
    ("desactiva HYPE", "quitar HYPE del playbook (todas sus temporalidades)"),
    ("playbook", "ver el playbook (gráficas en automático)"),
    ("informes", "ver los últimos informes de backtest"),
    ("alerta HYPE > 90", "crear alarma: HYPE > 90, una vez"),
    ("alerta BTC < 80000 siempre", "crear alarma: BTC < 80000, siempre"),
    ("alerta SOL cruza 150", "crear alarma: SOL cruza 150, una vez"),
    ("alerta HYPE rsi(14) < 30 1h", "crear alarma: HYPE RSI(14) < 30 en 1h, una vez"),
    ("alerta BTC cruce sma 50 200 1d alcista", "crear alarma: BTC cruce al alza SMA 50/200 en 1d, una vez"),
    ("alerta BTC cruce de medias 4h", "crear alarma: BTC cruce medias en 4h, una vez"),
    ("alerta SOL zona long", "crear alarma: SOL entra en zona de liquidez LONG, una vez"),
    ("alerta SOL zona", "crear alarma: SOL entra en zona de liquidez LONG o SHORT, una vez"),
    ("alerta BTC volumen 3x 15m", "crear alarma: BTC pico de volumen 3x en 15m, una vez"),
    ("alerta HYPE cae 3% 1h", "crear alarma: HYPE se mueve 3% a la baja en 1h, una vez"),
    ("alertas", "ver las alarmas"),
    ("borra alerta 3", "borrar la alarma #3"),
    ("borra alertas", "borrar todas las alarmas"),
    ("silencio 30m", "silenciar los avisos no críticos 30 min"),
    ("silencia 2h", "silenciar los avisos no críticos 2 h"),
    ("silencio 1d", "silenciar los avisos no críticos 1 d"),
    ("silencio 90", "silenciar los avisos no críticos 90 min"),
    ("silencio", "silenciar los avisos no críticos hasta 'sonido'"),
    ("sonido", "reactivar los avisos"),
])
def test_describe_phase2(settings, text, expected):
    assert describe(parse_rules(text, settings)) == expected


def test_help_text_lists_phase2_examples(settings):
    examples = ("backtest HYPE 1h", "backtest BTC 15m 90d", "optimiza ETH 1h 3 meses", "prueba HYPE 1h ema_trend",
                "informes", "ultimos backtests", "activa HYPE 1h", "usa la mejor HYPE 1h", "activa HYPE 1h ema_trend",
                "desactiva HYPE 1h", "desactiva HYPE", "playbook", "alerta HYPE > 90", "alerta BTC < 80000",
                "alerta SOL cruza 150", "alerta HYPE rsi < 30 1h", "alerta BTC cruce ema 20 50 1h alcista",
                "alerta SOL zona long", "alerta BTC volumen 3x 15m", "alerta HYPE mueve 3% 1h", "alertas",
                "borra alerta 3", "borra alertas", "silencio 30m", "silencio", "sonido")
    for text in examples:
        assert text in HELP_TEXT, text
        assert parse_rules(text, settings) is not None, text
    assert "FUERA de muestra" in HELP_TEXT                  # un backtest no garantiza nada
    for interval in commands.INTERVALS:
        assert interval in HELP_TEXT


# --- fase 2 por IA ----------------------------------------------------------------------------

@pytest.mark.parametrize("payload,expected", [
    (llm_json(action="backtest", coin="hype", interval="1h", days=90, strategy="EMA trend"),
     dict(action=A.BACKTEST, coin="HYPE", interval="1h", days=90.0, strategy="ema_trend")),
    (llm_json(action="backtest", coin="BTC"), dict(action=A.BACKTEST, coin="BTC", interval=None, days=None)),
    (llm_json(action="backtest", coin="SOL", strategy="donchian"), dict(strategy="donchian_breakout")),
    (llm_json(action="adopt", coin="hype", interval="4h"), dict(action=A.ADOPT, coin="HYPE", interval="4h")),
    (llm_json(action="adopt", coin="hype", interval="4h", strategy="rsi_reversion", days=30),
     dict(action=A.ADOPT, strategy="rsi_reversion", days=None)),            # días: no aplica, se ignora
    (llm_json(action="deactivate", coin="sol"), dict(action=A.DEACTIVATE, coin="SOL", interval=None)),
    (llm_json(action="deactivate", coin="sol", interval="15m", strategy="ema_trend"),
     dict(action=A.DEACTIVATE, interval="15m", strategy=None)),
    (llm_json(action="playbook"), dict(action=A.PLAYBOOK)),
    (llm_json(action="reports"), dict(action=A.REPORTS)),
    (llm_json(action="alert_list"), dict(action=A.ALERT_LIST)),
    (llm_json(action="unmute"), dict(action=A.UNMUTE)),
    (llm_json(action="alert_add", coin="hype", alert="rsi < 30", interval="1h"),
     dict(action=A.ALERT_ADD, coin="HYPE", alert=_spec("rsi_below", "1h", level=30.0))),
    (llm_json(action="alert_add", coin="BTC", alert="< 80000 siempre", interval="1h"),
     dict(alert=_spec("price_below", repeat=True, level=80000.0))),        # precio: la temporalidad no aplica
    (llm_json(action="alert_add", coin="HYPE", alert="HYPE > 90"), dict(alert=_spec("price_above", level=90.0))),
    (llm_json(action="alert_add", coin="SOL", alert="alerta SOL zona long"),
     dict(coin="SOL", alert=_spec("zone", side="long"))),
    (llm_json(action="alert_add", coin="BTC", alert="cruce sma 50 200 alcista", interval="1d"),
     dict(alert=_spec("ma_cross", "1d", fast=50, slow=200, ma="sma", direction="up"))),
    (llm_json(action="alert_add", coin="BTC", alert="volumen 3x 15m", interval="15m"),
     dict(alert=_spec("volume_spike", "15m", mult=3.0))),
    (llm_json(action="alert_del", value=3), dict(action=A.ALERT_DEL, target_id=3, value=None)),
    (llm_json(action="alert_del"), dict(action=A.ALERT_DEL, target_id=None)),
    (llm_json(action="mute", value=30), dict(action=A.MUTE, value=30.0)),
    (llm_json(action="mute"), dict(action=A.MUTE, value=None)),
])
def test_llm_phase2_valid(llm_settings, payload, expected):
    llm, _ = make_llm(llm_settings, fake_response(payload))
    cmd = llm.interpret("lo que sea")
    assert cmd is not None and cmd.source == "llm", payload
    _assert_fields(cmd, expected)


@pytest.mark.parametrize("payload", [
    llm_json(action="backtest"),                                         # sin moneda
    llm_json(action="backtest", coin="HYPE", interval="7h"),
    llm_json(action="backtest", coin="HYPE", interval=5),
    llm_json(action="backtest", coin="HYPE", strategy="macd"),           # estrategia desconocida
    llm_json(action="backtest", coin="HYPE", strategy=3),
    llm_json(action="backtest", coin="HYPE", days=0.5),
    llm_json(action="backtest", coin="HYPE", days=-30),
    llm_json(action="backtest", coin="HYPE", days="90"),
    llm_json(action="adopt", coin="HYPE"),                               # sin temporalidad
    llm_json(action="adopt", coin="HYPE", interval="1h", strategy="la buena"),
    llm_json(action="deactivate"),
    llm_json(action="alert_add", coin="HYPE"),                           # sin condición
    llm_json(action="alert_add", coin="HYPE", alert="lo que sea"),
    llm_json(action="alert_add", coin="HYPE", alert="BTC > 90"),         # otra moneda
    llm_json(action="alert_add", alert="HYPE > 90"),                     # sin moneda en el campo
    llm_json(action="alert_add", coin="HYPE", alert=123),
    llm_json(action="alert_add", coin="HYPE", alert="rsi < 30 4h", interval="1h"),   # temporalidades distintas
    llm_json(action="alert_add", coin="HYPE", alert="> 90.000"),         # ambiguo
    llm_json(action="alert_del", value=2.5),
    llm_json(action="alert_del", value=-1),
    llm_json(action="mute", value=-5),
    llm_json(action="mute", value=0),
])
def test_llm_phase2_invalid(llm_settings, payload):
    llm, _ = make_llm(llm_settings, fake_response(payload))
    assert llm.interpret("algo raro para la mesa") is None, payload


def test_llm_prompt_documents_phase2_actions():
    for action in ("backtest", "adopt", "deactivate", "playbook", "reports", "alert_add", "alert_list", "alert_del",
                   "mute", "unmute"):
        assert f"- {action}" in SYSTEM_PROMPT or f" {action}:" in SYSTEM_PROMPT, action
        assert action in LLM_ACTIONS
    for name in commands.KNOWN_STRATEGIES + commands.INTERVALS:
        assert name in SYSTEM_PROMPT


def test_parser_falls_back_to_llm_for_unparsed_phase2_text(llm_settings):
    llm_cmd = Command(A.ALERT_ADD, raw="x", source="llm", coin="HYPE", alert=_spec("price_above", level=90.0))
    fake = FakeInterpreter(llm_cmd)
    assert parse("alerta HYPE > 90", llm_settings, fake).source == "regex" and fake.calls == []
    assert parse("avísame cuando hype pase de los noventa", llm_settings, fake) is llm_cmd

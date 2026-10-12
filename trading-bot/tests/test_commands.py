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
                          "tp_price", "tp_pct", "param", "value"}
    for name, schema in props.items():
        assert {"type": "null"} in schema["anyOf"], name          # todos nullable
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

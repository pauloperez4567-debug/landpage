"""Comandos de la mesa: texto libre en español o inglés -> ``Command``.

Dos capas:

1. **Reglas** (``parse_rules``): normalización (sin mayúsculas ni acentos que
   importen) + vocabulario fijo ES/EN. Es determinista y *estricta*: si sobra o
   falta algo (dos monedas, un número suelto, modos de entrada contradictorios,
   una palabra desconocida) devuelve ``None`` en vez de adivinar. Ejemplos::

       long HYPE 2u zona sl 1.5% tp 4%     compra SOL a mercado     vende ETH @ 3300
       cierra HYPE / close all / flat      cancela HYPE / cancel all
       stop HYPE 84.5 / tp HYPE 90         apalancamiento 3 / riesgo 0,5 / unidad 250
       pausa / reanuda / auto on|off / estado / kill / reset halt / si / no / ayuda
       backtest HYPE 1h 90d / optimiza ETH 1h / prueba HYPE 1h ema_trend / informes
       activa HYPE 1h / usa la mejor HYPE 1h / desactiva HYPE 1h / playbook
       alerta HYPE > 90 / alerta HYPE rsi < 30 1h / alerta BTC cruce ema 20 50 1h alcista
       alerta SOL zona long / alerta BTC volumen 3x 15m / alerta HYPE mueve 3% 1h siempre
       alertas / borra alerta 3 / borra alertas / silencio 30m / sonido

2. **IA** (``LLMCommandInterpreter``): sólo si las reglas no entienden el texto,
   ``settings.llm_enabled`` y hay SDK ``anthropic`` + credenciales. Devuelve un
   ``Command`` validado con ``source="llm"`` (la mesa pide confirmación antes de
   abrir) o ``None``; nunca lanza.

Convenciones del resultado:

- ``coin`` en mayúsculas; activos HIP-3 como ``dex:TICKER`` (``xyz:GOLD``; el
  prefijo del dex queda en minúsculas, como lo nombra Hyperliquid).
- OPEN: ``entry_mode`` siempre presente (LIMIT si trae precio; si no dice nada,
  el de ``settings.default_entry_mode``). ``units`` None = la mesa decide.
  SL/TP con ``%`` -> ``sl_pct``/``tp_pct`` (distancia desde la entrada); sin
  ``%`` -> ``sl_price``/``tp_price`` (precio absoluto).
- SET_SL / SET_TP (``stop HYPE 84.5``): sobre la posición abierta de ``coin``;
  ``price`` si es un precio, ``sl_pct``/``tp_pct`` si viene en %. Sin moneda
  (``sl 2%``, ``tp 3``) cambian el parámetro por defecto (SET_PARAM).
- SET_PARAM: ``param`` es el nombre canónico (``PARAM_ALIASES``/``TUNABLE``) y
  ``value`` un float; el rango lo valida ``Settings.set_param``. ``param=PARAM_DEFAULTS`` (sin ``value``,
  ``reset params`` / ``restablece parámetros``) vuelve a los valores de arranque.
- Números con punto o coma decimal (``85.2`` / ``85,2``); los ambiguos entre
  miles y decimales (``83,770``, ``1.000``) se rechazan (ver ``parse_number``).
- Fase 2 (gráfica = moneda + temporalidad de ``exchange.intervals``):
  BACKTEST lleva ``coin``, ``interval`` (None = el de la mesa), ``days`` (``90d``, ``90 días``,
  ``3 meses`` = 90, ``1 año`` = 365; None = por defecto) y ``strategy`` (None = todas). ADOPT exige
  ``coin`` + ``interval`` (``strategy`` None = la ganadora del último backtest). DEACTIVATE: ``coin`` e
  ``interval`` opcional (None = todas sus temporalidades). ALERT_ADD: ``coin`` + ``alert`` =
  ``{"kind", <parámetros dados>, "interval", "repeat"}`` (los parámetros que el usuario no da se omiten:
  los pone ``AlertManager``, que también valida rangos con un mensaje claro). ALERT_DEL: ``target_id``
  (None = todas). MUTE: ``value`` = minutos (None = hasta UNMUTE).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol

from .config import PARAM_ALIASES, TUNABLE, Settings
from .exchange.intervals import INTERVAL_SECONDS
from .models import Command, CommandAction, EntryMode, Side

log = logging.getLogger(__name__)

Level = tuple[float, bool]  # (valor, es_porcentaje)

# --- vocabulario (ya normalizado: minúsculas y sin acentos) -------------------

_LONG_WORDS = frozenset({"long", "largo", "larga", "buy", "compra", "comprar", "compro", "compre",
                         "comprame"})
_SHORT_WORDS = frozenset({"short", "corto", "sell", "vende", "vender", "vendo", "vendeme", "shortea"})
_SIDE_WORDS = _LONG_WORDS | _SHORT_WORDS
_UNIT_WORDS = frozenset({"u", "ud", "uds", "unidad", "unidades", "unit", "units"})
_ZONE_WORDS = frozenset({"zona", "zone", "zonas", "liquidez", "liquidity", "liq"})
_MARKET_WORDS = frozenset({"mercado", "market", "mkt", "ya", "now", "ahora", "inmediato",
                           "inmediatamente", "immediately", "instant"})
_LIMIT_WORDS = frozenset({"@", "limit", "limite", "lim", "lmt", "precio", "price"})
_LIMIT_PREPS = frozenset({"a", "at", "en", "al"})      # "a 85.2": límite sólo si sigue un número
_LIMIT_CONNECTORS = frozenset({"a", "at", "en", "al", "@", "de", "precio", "price", "="})
_SL_WORDS = frozenset({"sl", "stop", "stoploss"})       # + "stop loss"
_TP_WORDS = frozenset({"tp", "takeprofit", "objetivo", "target", "tgt"})  # + "take profit"
_PCT_WORDS = frozenset({"%", "pct", "porciento", "percent"})  # + "por ciento"
_ALL_WORDS = frozenset({"todo", "toda", "todos", "todas", "all", "everything"})
_ORDER_WORDS = frozenset({"orden", "ordenes", "order", "orders", "pendiente", "pendientes", "pending",
                          "entrada", "entradas", "entry", "entries"})
_ARTICLES = frozenset({"el", "la", "los", "las", "un", "una", "mi", "mis", "my", "the", "a", "an"})
_POSITION_WORDS = frozenset({"posicion", "position", "pos", "operacion", "trade"})
_URGENT_WORDS = frozenset({"ya", "now", "ahora", "inmediatamente", "immediately"})

_OPEN_FILLERS = _ARTICLES | _POSITION_WORDS | frozenset({
    "abre", "abrir", "abri", "open", "entra", "entrar", "enter", "go", "ve", "ir", "mete", "meter",
    "haz", "hacer", "quiero", "ponte", "pon", "dame", "en", "on", "in", "at", "al", "de", "del", "con",
    "with", "y", "and", "para", "for", "of", "orden", "order", "entrada", "entry"})
_CLOSE_VERBS = frozenset({"cierra", "cierre", "cerrar", "cierro", "cierrame", "close", "exit", "sal",
                          "salir", "salte", "liquida", "liquidar", "flat", "flatten"})
# Relleno habitual en "cierra todas las posiciones abiertas ahora mismo" / "close all open trades".
_OPEN_STATE_WORDS = frozenset({"abierta", "abiertas", "abierto", "abiertos", "open", "operaciones", "trades",
                               "mismo", "right", "inmediato"})
_CLOSE_FILLERS = _ARTICLES | _POSITION_WORDS | _SIDE_WORDS | _URGENT_WORDS | _OPEN_STATE_WORDS | frozenset({
    "de", "del", "en", "on", "of", "from", "out"})
_CANCEL_VERBS = frozenset({"cancela", "cancelar", "cancel", "cancelo", "cancelalas", "anula", "anular"})
_CANCEL_FILLERS = _ARTICLES | _SIDE_WORDS | _URGENT_WORDS | _OPEN_STATE_WORDS | frozenset({
    "de", "del", "en", "on", "for", "of", "limite", "limit"})
_SET_VERBS = frozenset({"mueve", "mover", "muevo", "move", "pon", "poner", "ponle", "set", "cambia",
                        "cambiar", "change", "ajusta", "ajustar", "adjust", "sube", "subir", "raise",
                        "baja", "bajar", "lower", "actualiza", "actualizar", "update", "coloca",
                        "colocar", "put", "configura", "configurar", "config", "fija", "fijar",
                        "establece", "establecer"})
_LEVEL_FILLERS = _ARTICLES | _POSITION_WORDS | _SIDE_WORDS | frozenset({
    "a", "al", "at", "to", "en", "on", "in", "de", "del", "of", "for", "para", "=", "@", "nivel", "level"})
_PARAM_CONNECTORS = frozenset({"a", "al", "=", "to", "en", "at", "de", "is", "es", "in"})
_PARAM_NAME_NOISE = frozenset({"de", "del", "la", "el", "the", "of", "por", "per"})

# Unidades que se aceptan detrás (o delante: "x3") del valor de cada parámetro.
_PCT_PARAMS = frozenset(k for k in TUNABLE if k.endswith("_pct"))
_PARAM_UNITS: dict[str, frozenset[str]] = {
    "leverage": frozenset({"x"}),
    "unit_margin_usd": frozenset({"usd", "usdc", "dolares", "dollars"}),
    "default_tp_r": frozenset({"r"}),
    "entry_ttl_minutes": frozenset({"m", "min", "mins", "minutos", "minutes"}),
    **{k: _PCT_WORDS for k in _PCT_PARAMS},
}
_ANY_PARAM_UNIT = frozenset().union(*_PARAM_UNITS.values())

# Alias extra (multipalabra o habituales) que no están en ``config.PARAM_ALIASES``.
_EXTRA_PARAM_ALIASES: dict[str, str] = {
    "stop_loss": "default_sl_pct", "take_profit": "default_tp_r",
    "max_posiciones": "max_open_positions", "maximo_posiciones": "max_open_positions",
    "posiciones_maximas": "max_open_positions",
    "riesgo_operacion": "risk_per_trade_pct", "risk_per_trade": "risk_per_trade_pct",
    "margen": "unit_margin_usd", "margin": "unit_margin_usd", "tamano_unidad": "unit_margin_usd",
    "unit_size": "unit_margin_usd", "max_drawdown": "max_drawdown_pct",
}

_POLITENESS = frozenset({"please", "pls", "plz", "porfa", "porfavor", "porfis"})

# Sustantivos de la fase 2: nunca son monedas ("cierra alertas" no es cerrar la moneda ALERTAS).
_ALERT_NOUNS = frozenset({"alerta", "alertas", "alarma", "alarmas", "alert", "alerts", "alarm", "alarms"})
_PHASE2_NOUNS = _ALERT_NOUNS | frozenset({"avisos", "playbook", "backtest", "backtests", "informes", "reports"})

_RESERVED = frozenset().union(
    _SIDE_WORDS, _UNIT_WORDS, _ZONE_WORDS, _MARKET_WORDS, _LIMIT_WORDS, _LIMIT_PREPS, _SL_WORDS,
    _TP_WORDS, _PCT_WORDS, _ALL_WORDS, _ORDER_WORDS, _ARTICLES, _POSITION_WORDS, _OPEN_FILLERS,
    _CLOSE_VERBS, _CLOSE_FILLERS, _CANCEL_VERBS, _CANCEL_FILLERS, _SET_VERBS, _LEVEL_FILLERS,
    _PARAM_CONNECTORS, _ANY_PARAM_UNIT, _POLITENESS, _PHASE2_NOUNS,
    {"loss", "take", "profit", "por", "ciento", "x", "r", "posiciones", "positions"})

# --- frases fijas (todo el texto normalizado debe coincidir) -------------------

_AUTO = r"(?:auto|automatico|autonomo|autopilot|autopiloto|automatic|autotrading|auto trading)"
_ON = r"(?:on|si|yes|activado|activo|activar|encendido|encender|enable|enabled|true|1)"
_OFF = r"(?:off|no|desactivado|desactivar|apagado|apagar|disable|disabled|false|0)"

# SET_PARAM especial: olvidar los parámetros cambiados por comando (vuelven los de la configuración).
PARAM_DEFAULTS = "defaults"
_RESET_PARAMS_RX = re.compile(
    r"(?:reset|resetea|resetear|restablece|restablecer|restaura|restaurar|borra|borrar|olvida|clear|restore)"
    r"(?: (?:los|mis|the|my))? (?:parametros|params|parameters|ajustes)"
    r"(?: (?:por defecto|de arranque|originales|to defaults|default|defaults))?"
    r"|(?:parametros|params|parameters) (?:por defecto|de arranque|default|defaults)")

_SHOW = r"(?:(?:ver|muestra|muestrame|mostrar|ensena|ensename|show|dame|lista|listar|list|see|que|cuales) )?"
_THE = r"(?:(?:el|la|las|los|mi|mis|my|the) )?"

_PHRASES: list[tuple[CommandAction, re.Pattern[str]]] = [(a, re.compile(p)) for a, p in [
    (CommandAction.HELP, r"ayuda|help|comandos|commands|menu|que puedes hacer|what can you do"),
    # Confirmación explícita: "ok"/"dale" NO confirman (demasiado fáciles de escribir sin querer).
    (CommandAction.CONFIRM,
     r"(?:si|yes)(?: (?:confirmo|confirmar|confirma|confirm|adelante|hazlo|go ahead|do it))?"
     r"|confirmar|confirmo|confirma|confirmado|confirm|confirmed|adelante|ejecuta|ejecutalo"
     r"|go ahead|do it"),
    # "cancelar" a secas descarta lo pendiente; "cancela HYPE"/"cancel all" cancelan órdenes.
    (CommandAction.REJECT,
     r"no(?: gracias| thanks| cancela| cancelar)?|nope|cancelar|cancela|cancel|cancelalo|rechazar"
     r"|rechazo|rechaza|rechazado|reject|rejected|descartar|descarta|descartalo|discard|olvidalo"
     r"|forget it"),
    (CommandAction.KILL,
     r"(?:kill(?: (?:all|everything|switch|it|todo))?|killswitch|panico|panic|boton (?:de )?panico"
     r"|panic button|emergencia|emergency|apaga(?:lo)? todo|apagar todo|shutdown|shut (?:it )?down)"
     r"(?: (?:ya|ya mismo|now|right now|ahora|ahora mismo|inmediatamente|immediately))?"),
    (CommandAction.RESET_HALT,
     r"(?:reset|resetea|resetear|reinicia|reiniciar|levanta|levantar|quita|quitar|clear|lift|unlock"
     r"|desbloquea|desbloquear|libera|liberar) (?:el |la |the )?(?:halt|bloqueo|freno)|unhalt"),
    (CommandAction.PAUSE,
     r"(?:pausa|pausar|pausate|pausalo|pause)(?: (?:el bot|la mesa|el trading|trading|todo"
     r"|las entradas|entradas|the bot|bot|entries))?|(?:para|deja|dejar) de operar"
     r"|(?:para|deten|detener|stop) (?:el|the) bot|stop trading|no operes"),
    (CommandAction.RESUME,
     r"(?:reanuda|reanudar|reanudalo|resume|continua|continuar|continue|sigue|seguir|unpause"
     r"|despausa|despausar|reactiva|reactivar)(?: (?:el bot|la mesa|el trading|trading|the bot|bot"
     r"|operando))?|vuelve a operar|start trading"),
    (CommandAction.AUTO_ON,
     rf"(?:modo )?{_AUTO}(?: mode| modo)? {_ON}"
     rf"|(?:activa|activar|enciende|encender|prende|prender|enable|turn on|start) (?:el |the )?"
     rf"(?:modo )?{_AUTO}(?: mode)?"),
    (CommandAction.AUTO_OFF,
     rf"(?:modo )?{_AUTO}(?: mode| modo)? {_OFF}"
     rf"|(?:desactiva|desactivar|apaga|apagar|disable|turn off|stop) (?:el |the )?"
     rf"(?:modo )?{_AUTO}(?: mode)?"),
    (CommandAction.STATUS,
     r"(?:(?:ver|muestra|muestrame|mostrar|show|dame|give me|check) )?(?:(?:el|la|las|los|mi|mis|my"
     r"|the|me) )?(?:estado|status|stats|posiciones|positions|pos|resumen|summary|balance|saldo"
     r"|cuenta|account|pnl|ordenes|orders|reporte|report|informe)(?: (?:de la cuenta|de cuenta"
     r"|del bot|de la mesa|abiertas|open))?|como vamos|como va(?: todo)?"),
    (CommandAction.CLOSE_ALL, r"flat|flatten|go flat|ponte flat|todo flat|aplana(?:r)?(?: todo)?"),
    # Fase 2 (sólo lectura). "informe"/"report" en singular siguen siendo STATUS (arriba).
    (CommandAction.PLAYBOOK,
     rf"{_SHOW}{_THE}(?:playbook|(?:graficas|graficos|estrategias|charts|strategies) (?:activas|activos"
     rf"|en (?:auto|automatico)|active|live)|(?:active|live) (?:charts|strategies))"),
    (CommandAction.REPORTS,
     rf"{_SHOW}{_THE}(?:(?:ultimos|ultimas|last|latest|recent) )?(?:informes|reports|backtests"
     rf"|informes de backtests?|backtest reports|resultados de (?:los )?backtests?)(?: (?:recientes|recent))?"),
    (CommandAction.ALERT_LIST,
     rf"{_SHOW}{_THE}(?:alertas|alarmas|alerts|alarms|avisos)(?: (?:activas|activos|active|pendientes|pending"
     rf"|configuradas|puestas))?(?: (?:hay|tengo))?"),
]]


# --- normalización ------------------------------------------------------------

def normalize(text: str) -> str:
    """Minúsculas, sin acentos (``á``->``a``, ``ñ``->``n``) y espacios simples."""
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s.lower()).strip()


_NUM_RE = re.compile(r"^(?:\d+(?:[.,]\d+)*|[.,]\d+)$")
_NUM_SUFFIX_RE = re.compile(
    r"(?<![\w.,])(\d+(?:[.,]\d+)*)(u|uds?|unidad(?:es)?|units?|x|r|usdc?|pct|mins?|minutos|minutes|m)\b")
_HIP3_RE = re.compile(r"([a-z][a-z0-9]{0,9}):([a-z0-9]{1,20})")
_COIN_RE = re.compile(r"[a-z0-9]{1,20}")


def tokenize(text: str) -> list[str]:
    """Texto -> tokens normalizados.

    Separa ``@``, ``%``, ``=``, ``<`` y ``>``; parte sufijos pegados al número (``2u``, ``3x``,
    ``250usd``); quita ``$`` (``$HYPE``, ``250$``), comas y puntos que no son
    decimales y las fórmulas de cortesía. ``:`` sólo se conserva en tickers HIP-3.
    """
    s = normalize(text)
    s = re.sub(r"[-/](?:perp|usdc|usdt|usd)\b", "", s)          # HYPE-PERP, BTC/USDC
    s = re.sub(r"(?<=[a-z])-(?=[a-z])", "", s)                   # stop-loss -> stoploss
    s = re.sub(r"(?<!\S)[+-](?=\d+(?:[.,]\d+)?\s*%)", "", s)     # "sl -1.5%" == "sl 1.5%"
    s = s.replace("$", " ")
    s = re.sub(r"(?<!\d),|,(?!\d)", " ", s)                      # comas que no son decimales
    s = re.sub(r"\.(?!\d)", " ", s)                              # puntos finales
    s = re.sub(r"[@%=<>≤≥]", lambda m: f" {m.group()} ", s)
    s = re.sub(r"[;!?¿¡()\[\]{}\"'`´*+]", " ", s)
    s = _NUM_SUFFIX_RE.sub(r"\1 \2", s)
    s = re.sub(r"(?<!\w)x(?=\d)", "x ", s)                      # x3 -> x 3
    toks: list[str] = []
    for tok in s.split():
        if ":" in tok:
            m = _HIP3_RE.fullmatch(tok)
            if m and m.group(1) not in _RESERVED:
                toks.append(tok)
            else:                                                # "sl:1.5" -> "sl", "1.5"
                toks.extend(p for p in tok.split(":") if p)
        elif tok not in _POLITENESS:
            toks.append(tok)
    # "por favor" (sin confundir con "por ciento")
    out: list[str] = []
    for i, tok in enumerate(toks):
        if tok == "por" and i + 1 < len(toks) and toks[i + 1] == "favor":
            continue
        if tok == "favor" and i > 0 and toks[i - 1] == "por":
            continue
        out.append(tok)
    return out


def parse_number(tok: str) -> Optional[float]:
    """Número con punto o coma decimal y separador de miles opcional; None si no lo es.

    - ``85.2``, ``85,2``, ``0,5``, ``.5`` -> decimales.
    - ``83,770.5`` / ``83.770,5`` -> el último separador es el decimal.
    - ``1,000,000`` / ``1.000.000`` -> miles (grupos de 3 obligatorios).
    - Un único separador seguido de exactamente 3 dígitos con parte entera != 0 es
      ambiguo: con coma (``83,770``: miles en inglés, decimales en español) o con
      punto y cero final (``1.000``, ``3.200``: miles en español) -> None.
      ``149.985`` se acepta como decimal.
    """
    tok = tok.strip()
    if not _NUM_RE.match(tok):
        return None
    has_dot, has_comma = "." in tok, "," in tok
    if has_dot and has_comma:
        dec = "." if tok.rfind(".") > tok.rfind(",") else ","
        thou = "," if dec == "." else "."
        if tok.count(dec) > 1:
            return None
        int_part, frac = tok.rsplit(dec, 1)
        if not _valid_groups(int_part, thou):
            return None
        return float(int_part.replace(thou, "") + "." + frac)
    if not (has_dot or has_comma):
        return float(tok)
    sep = "." if has_dot else ","
    if tok.count(sep) > 1:
        return float(tok.replace(sep, "")) if _valid_groups(tok, sep) else None
    int_part, frac = tok.split(sep)
    if len(frac) == 3 and int_part.strip("0") and (sep == "," or frac.endswith("0")):
        return None                                   # ¿miles o decimales? mejor no adivinar
    return float((int_part or "0") + "." + frac)


def _valid_groups(s: str, sep: str) -> bool:
    parts = s.split(sep)
    return 1 <= len(parts[0]) <= 3 and all(len(p) == 3 and p.isdigit() for p in parts[1:])


def _is_number(tok: str) -> bool:
    return bool(_NUM_RE.match(tok))


def normalize_coin(tok: str) -> Optional[str]:
    """Ticker en mayúsculas (``hype`` -> ``HYPE``); HIP-3 ``xyz:gold`` -> ``xyz:GOLD``.

    Quita sufijos de par (``-PERP``, ``/USDC``). None si no parece un ticker (vacío,
    sólo dígitos, caracteres raros).
    """
    t = re.sub(r"[-/](?:perp|usdc|usdt|usd)$", "", normalize(tok).lstrip("$"))
    m = _HIP3_RE.fullmatch(t)
    if m:
        return f"{m.group(1)}:{m.group(2).upper()}"
    if _COIN_RE.fullmatch(t) and re.search(r"[a-z]", t):
        return t.upper()
    return None


def _coin_token(tok: str) -> Optional[str]:
    return None if tok in _RESERVED else normalize_coin(tok)


def _skip(toks: list[str], i: int, words: Iterable[str]) -> int:
    words = words if isinstance(words, (set, frozenset)) else frozenset(words)
    while i < len(toks) and toks[i] in words:
        i += 1
    return i


def _pct_at(toks: list[str], j: int) -> int:
    """Cuántos tokens ocupa un marcador de porcentaje en ``j`` (0 si no hay)."""
    if j < len(toks) and toks[j] in _PCT_WORDS:
        return 1
    if j + 1 < len(toks) and toks[j] == "por" and toks[j + 1] == "ciento":
        return 2
    return 0


def _keyword_len(toks: list[str], i: int, words: frozenset[str], bigram: tuple[str, str]) -> int:
    """Longitud de la palabra clave SL/TP en ``i`` (``sl``=1, ``stop loss``=2; 0 si no hay)."""
    if i + 1 < len(toks) and (toks[i], toks[i + 1]) == bigram:
        return 2
    return 1 if i < len(toks) and toks[i] in words else 0


def _sl_len(toks: list[str], i: int) -> int:
    return _keyword_len(toks, i, _SL_WORDS, ("stop", "loss"))


def _tp_len(toks: list[str], i: int) -> int:
    return _keyword_len(toks, i, _TP_WORDS, ("take", "profit"))


def _level(toks: list[str], i: int) -> Optional[tuple[Level, int]]:
    """Lee ``[conectores] NÚMERO [%]`` desde ``i`` -> ((valor, es_pct), índice siguiente)."""
    j = _skip(toks, i, _LIMIT_CONNECTORS)
    v = parse_number(toks[j]) if j < len(toks) else None
    if v is None or v <= 0:
        return None
    j += 1
    k = _pct_at(toks, j)
    if k and v >= 100:
        return None                                   # un SL/TP del 100 % no tiene sentido
    return (v, bool(k)), j + k


# --- gramáticas -------------------------------------------------------------------

def _default_entry_mode(settings: Optional[Settings]) -> EntryMode:
    try:
        return EntryMode(getattr(settings, "default_entry_mode", "zone"))
    except ValueError:
        return EntryMode.ZONE


def _parse_open(toks: list[str], raw: str, settings: Optional[Settings]) -> Optional[Command]:
    side: Optional[Side] = None
    coin: Optional[str] = None
    units: Optional[float] = None
    price: Optional[float] = None
    mode: Optional[EntryMode] = None
    sl: Optional[Level] = None
    tp: Optional[Level] = None
    n, i = len(toks), 0
    while i < n:
        t = toks[i]
        if t in _SIDE_WORDS:
            s = Side.LONG if t in _LONG_WORDS else Side.SHORT
            if side is not None and side is not s:
                return None
            side, i = s, i + 1
            continue
        if _is_number(t):                             # sólo "N u/unidades/units"; un número suelto es ambiguo
            v = parse_number(t)
            if v is None or v <= 0 or units is not None or i + 1 >= n or toks[i + 1] not in _UNIT_WORDS:
                return None
            units, i = v, i + 2
            continue
        if t in _LIMIT_WORDS or t in _LIMIT_PREPS:
            j = _skip(toks, i + 1, _LIMIT_CONNECTORS)
            num_follows = j < n and _is_number(toks[j])
            unit_after = j + 1 < n and (toks[j + 1] in _UNIT_WORDS or _pct_at(toks, j + 1))
            if t in _LIMIT_WORDS or (num_follows and not unit_after):
                v = parse_number(toks[j]) if num_follows else None
                if v is None or v <= 0 or unit_after or (price is not None and price != v):
                    return None
                price, i = v, j + 1
                continue
            i += 1                                    # "a mercado", "en zona": preposición sin más
            continue
        if t in _ZONE_WORDS or t in _MARKET_WORDS:
            m = EntryMode.ZONE if t in _ZONE_WORDS else EntryMode.MARKET
            if mode is not None and mode is not m:
                return None
            mode, i = m, i + 1
            continue
        k_sl, k_tp = _sl_len(toks, i), _tp_len(toks, i)
        if k_sl or k_tp:
            r = _level(toks, i + (k_sl or k_tp))
            if r is None or (sl if k_sl else tp) is not None:
                return None
            if k_sl:
                sl = r[0]
            else:
                tp = r[0]
            i = r[1]
            continue
        if t in _OPEN_FILLERS:
            i += 1
            continue
        c = _coin_token(t)
        if c is None or (coin is not None and coin != c):
            return None
        coin, i = c, i + 1
    if side is None or coin is None:
        return None
    if price is not None:
        if mode not in (None, EntryMode.LIMIT):
            return None                               # "zona @ 85" / "a mercado limit 85": contradictorio
        mode = EntryMode.LIMIT
    elif mode is None:
        mode = _default_entry_mode(settings)
    return Command(
        CommandAction.OPEN, raw=raw, coin=coin, side=side, units=units, entry_mode=mode, price=price,
        sl_price=sl[0] if sl and not sl[1] else None, sl_pct=sl[0] if sl and sl[1] else None,
        tp_price=tp[0] if tp and not tp[1] else None, tp_pct=tp[0] if tp and tp[1] else None,
    )


def _parse_close(toks: list[str], raw: str) -> Optional[Command]:
    coin: Optional[str] = None
    every = False
    for t in toks[1:]:
        if t in _ALL_WORDS or t in {"posiciones", "positions"}:
            every = True
        elif t in _CLOSE_FILLERS:
            continue
        else:
            c = _coin_token(t)
            if c is None or (coin is not None and coin != c):
                return None
            coin = c
    if coin is not None:
        # "cierra todo X": ¿todo o sólo X? (o una palabra desconocida tomada por moneda): mejor no adivinar.
        return None if every else Command(CommandAction.CLOSE, raw=raw, coin=coin)
    return Command(CommandAction.CLOSE_ALL, raw=raw) if every else None


def _parse_cancel(toks: list[str], raw: str) -> Optional[Command]:
    coin: Optional[str] = None
    target = every = False                            # "órdenes" / "todo": todas las monedas
    for t in toks[1:]:
        if t in _ALL_WORDS:
            target = every = True
        elif t in _ORDER_WORDS:
            target = True
        elif t in _CANCEL_FILLERS:
            continue
        else:
            c = _coin_token(t)
            if c is None or (coin is not None and coin != c):
                return None
            coin = c
    if coin is not None and every:
        return None                                   # "cancela todo X": ambiguo
    if coin is not None or target:
        return Command(CommandAction.CANCEL, raw=raw, coin=coin)
    return None


def _parse_level_cmd(toks: list[str], raw: str) -> Optional[Command]:
    """``sl|stop|tp ...``: con moneda -> SET_SL/SET_TP; sin moneda -> SET_PARAM."""
    k_sl = _sl_len(toks, 0)
    is_sl = bool(k_sl)
    i, n = (k_sl or _tp_len(toks, 0)), len(toks)
    coin: Optional[str] = None
    value: Optional[float] = None
    pct = r_unit = False
    while i < n:
        t = toks[i]
        if _is_number(t):
            v = parse_number(t)
            if v is None or v <= 0 or value is not None:
                return None
            value, i = v, i + 1
            k = _pct_at(toks, i)
            pct, i = bool(k), i + k
            continue
        if t == "r" and value is not None and not is_sl:   # "tp 3r"
            r_unit, i = True, i + 1
            continue
        if t in _LEVEL_FILLERS:
            i += 1
            continue
        c = _coin_token(t)
        if c is None or (coin is not None and coin != c):
            return None
        coin, i = c, i + 1
    if value is None or (pct and value >= 100):
        return None
    if coin is not None:
        if r_unit:
            return None
        if is_sl:
            return Command(CommandAction.SET_SL, raw=raw, coin=coin,
                           price=None if pct else value, sl_pct=value if pct else None)
        return Command(CommandAction.SET_TP, raw=raw, coin=coin,
                       price=None if pct else value, tp_pct=value if pct else None)
    # Sin moneda: parámetro por defecto. El TP por defecto va en R, no en %. Un valor
    # fuera de rango ("stop 84.5") seguramente era un precio sin moneda: ambiguo -> None.
    key = PARAM_ALIASES["sl" if is_sl else "tp"]
    lo, hi = TUNABLE[key]
    if (pct and key not in _PCT_PARAMS) or not lo <= value <= hi:
        return None
    return Command(CommandAction.SET_PARAM, raw=raw, param=key, value=value)


def param_key(name: str) -> Optional[str]:
    """Nombre canónico de un parámetro ajustable (alias ES/EN o nombre del campo)."""
    key = normalize(name).replace(" ", "_").replace("-", "_")
    key = PARAM_ALIASES.get(key) or _EXTRA_PARAM_ALIASES.get(key) or key
    return key if key in TUNABLE else None


def _parse_param(toks: list[str], raw: str) -> Optional[Command]:
    n, i = len(toks), 0
    name: list[str] = []
    while i < n and not _is_number(toks[i]) and toks[i] not in _PARAM_CONNECTORS \
            and toks[i] not in _ANY_PARAM_UNIT:
        name.append(toks[i])
        i += 1
    if not name:
        return None
    key = param_key("_".join(name)) or param_key("_".join(t for t in name if t not in _PARAM_NAME_NOISE))
    if key is None:
        return None
    value: Optional[float] = None
    units: list[str] = []
    while i < n:
        t = toks[i]
        if _is_number(t):
            v = parse_number(t)
            if v is None or value is not None:
                return None
            value, i = v, i + 1
        elif _pct_at(toks, i):
            units.append("%")
            i += _pct_at(toks, i)
        elif t in _ANY_PARAM_UNIT:
            units.append(t)
            i += 1
        elif t in _PARAM_CONNECTORS:
            i += 1
        else:
            return None
    if value is None or any(u not in _PARAM_UNITS.get(key, frozenset()) for u in units):
        return None
    return Command(CommandAction.SET_PARAM, raw=raw, param=key, value=value)


# --- fase 2: backtest por gráfica, playbook, alarmas y silencio -----------------------

INTERVALS: tuple[str, ...] = tuple(INTERVAL_SECONDS)
# Estrategias que la mesa conoce aunque ``strategy.STRATEGIES`` no las registre todavía (se unen al registro).
KNOWN_STRATEGIES: tuple[str, ...] = ("liquidity_sweep", "squeeze_breakout", "ema_trend", "rsi_reversion",
                                     "donchian_breakout")
# Tipos de alarma de ``alerts.KINDS`` (copiados para no cargar pandas al parsear; un test vigila que coincidan).
ALERT_KINDS: tuple[str, ...] = ("price_above", "price_below", "price_cross", "rsi_below", "rsi_above", "ma_cross",
                                "zone", "volume_spike", "move_pct")
CANDLE_ALERT_KINDS = frozenset({"rsi_below", "rsi_above", "ma_cross", "volume_spike", "move_pct"})

# Unidad detrás de un número -> clase. Las letras (``15m``, ``1h``, ``1d``, ``1w``) son las de las temporalidades;
# las palabras completas ("90 días", "2 semanas", "3 meses") son duraciones.
_UNIT_CLASS: dict[str, str] = {
    **dict.fromkeys(("m", "min", "mins", "minuto", "minutos", "minute", "minutes"), "m"),
    **dict.fromkeys(("h", "hr", "hrs", "hora", "horas", "hour", "hours"), "h"),
    "d": "d", **dict.fromkeys(("dia", "dias", "day", "days"), "day"),
    "w": "w", **dict.fromkeys(("semana", "semanas", "week", "weeks"), "week"),
    **dict.fromkeys(("mes", "meses", "month", "months"), "month"),
    **dict.fromkeys(("ano", "anos", "year", "years"), "year"),
    **dict.fromkeys(("x", "veces", "times"), "x"),
}
_INTERVAL_LETTER = {"m": "m", "h": "h", "d": "d", "day": "d", "w": "w", "week": "w"}
_DAYS_PER = {"day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0}     # "3 meses" = 90 días, "1 año" = 365
_MINUTES_PER = {"": 1.0, "m": 1.0, "h": 60.0, "d": 1440.0, "day": 1440.0, "w": 10080.0, "week": 10080.0}
_ONE_WORDS = {"un": 1.0, "una": 1.0, "an": 1.0, "one": 1.0, "media": 0.5, "half": 0.5}   # "una hora", "media hora"
# Sufijos pegados que ``tokenize`` no separa: "1h", "90d", "1w", "1y", "2semanas"; e indicadores "rsi14", "ema20".
_COMPACT_RE = re.compile(r"(\d+(?:[.,]\d+)?)(h|hr|hrs|d|w|y|horas?|hours?|dias?|days?|semanas?|weeks?|mes|meses"
                         r"|months?|anos?|years?|veces|times)")
_IND_NUM_RE = re.compile(r"(rsi|ema|sma)(\d+)")

# Vocabulario de gráficas (backtest / activar / desactivar).
_BACKTEST_LEAD = frozenset({"haz", "hazme", "hacer", "run", "lanza", "lanzar", "corre", "correr", "un", "una", "a",
                            "el", "la"})
_BACKTEST_VERBS = frozenset({"backtest", "backtestea", "backtestear", "bt", "optimiza", "optimizar", "optimize",
                             "optimise", "prueba", "probar", "testea", "testear", "test"})
_ADOPT_VERBS = frozenset({"activa", "activar", "adopta", "adoptar", "adopt", "usa", "usar", "use", "aplica",
                          "aplicar", "apply", "enable"})
_DEACTIVATE_VERBS = frozenset({"desactiva", "desactivar", "deactivate", "disable"})
_CHART_FILLERS = frozenset({
    "de", "del", "en", "on", "in", "para", "for", "con", "with", "el", "la", "los", "las", "lo", "the", "a", "al",
    "of", "y", "and", "grafica", "grafico", "chart", "temporalidad", "timeframe", "tf", "velas", "vela", "candles",
    "estrategia", "strategy", "usando", "using"})
_BACKTEST_FILLERS = _CHART_FILLERS | frozenset({
    "ultimos", "ultimas", "last", "past", "historia", "history", "historico", "datos", "data", "todas", "todos",
    "all", "estrategias", "strategies"})
_ADOPT_FILLERS = _CHART_FILLERS | frozenset({
    "mejor", "best", "ganadora", "ganador", "winner", "top", "auto", "automatico", "automatica", "autonomo",
    "automatic", "playbook"})
_DEACTIVATE_FILLERS = _CHART_FILLERS | frozenset({
    "auto", "automatico", "automatica", "playbook", "from", "todas", "temporalidades", "timeframes"})

# Vocabulario de alarmas.
_ALERT_PLURALS = frozenset({"alertas", "alarmas", "alerts", "alarms"})
_ALERT_HEADS = _ALERT_NOUNS | frozenset({"avisame", "avisa", "avisas", "avisarme", "notificame", "notify"})
_ALERT_LEAD = frozenset({"pon", "ponme", "crea", "crear", "creame", "create", "nueva", "nuevo", "new", "add",
                         "anade", "agrega", "agregar", "set", "un", "una", "an", "la", "el", "me"})
_ALERT_FILLERS = frozenset({
    "si", "cuando", "when", "if", "el", "la", "los", "las", "de", "del", "en", "a", "al", "the", "of", "on", "in",
    "at", "me", "precio", "price", "que", "than", "por", "nivel", "level", "velas", "vela", "candle", "candles",
    "timeframe", "tf", "grafica", "chart", "para", "for", "un", "una", "y", "and", "esta", "is", "se", "va", "goes",
    "it", "to", "hasta", "periodo", "periodos", "period", "periods", "movil", "moviles", "moving", "average",
    "averages", "spike", "pico", "entra", "entre", "enters", "enter", "solo", "only"})
_ABOVE_WORDS = frozenset({">", "≥", "encima", "above", "over", "supera", "superar", "supere", "sobre", "mayor", "mas",
                          "arriba", "sube", "suba", "subir", "sobrepasa", "rises", "exceeds", "alza", "alcista",
                          "bullish", "up"})
_BELOW_WORDS = frozenset({"<", "≤", "debajo", "below", "under", "bajo", "menor", "menos", "abajo", "baja", "baje",
                          "bajar", "cae", "caiga", "caer", "pierde", "pierda", "perfora", "falls", "drops", "bajista",
                          "bearish", "down"})
_CROSS_WORDS = frozenset({"cruza", "cruce", "cruzan", "cruzar", "cross", "crosses", "crossover", "toca", "toque",
                          "tocar", "touch", "touches", "llega", "llegue", "reaches", "hits", "="})
_GOLDEN_WORDS = frozenset({"golden", "dorado"})          # golden cross: SMA 50/200 al alza
_DEATH_WORDS = frozenset({"death", "muerte"})            # death cross: SMA 50/200 a la baja
_MA_TYPES: dict[str, Optional[str]] = {"ema": "ema", "mme": "ema", "exponencial": "ema", "exponential": "ema",
                                       "sma": "sma", "mms": "sma", "simple": "sma", "media": None, "medias": None,
                                       "ma": None, "mm": None}
_VOLUME_WORDS = frozenset({"volumen", "volume", "vol"})
_MOVE_WORDS = frozenset({"mueve", "movimiento", "move", "moves", "varia", "variacion", "oscila", "cambia", "cambio"})
_ZONE_SIDES: dict[str, str] = {**dict.fromkeys(_LONG_WORDS | {"longs", "compras"}, "long"),
                               **dict.fromkeys(_SHORT_WORDS | {"shorts", "venta", "ventas"}, "short")}
_REPEAT_WORDS = frozenset({"siempre", "repeat", "repetir", "repite", "repetida", "repetidamente", "always",
                           "recurrente", "recurring", "repeating", "repeatedly"})
_REPEAT_BIGRAMS = frozenset({("cada", "vez"), ("every", "time")})
_ALERT_DEL_FILLERS = frozenset({"la", "el", "las", "los", "mi", "mis", "my", "the", "de", "numero", "num", "nro",
                                "id", "activas", "activos", "active"})

# Silencio de avisos.
_NOTICE_NOUNS = _ALERT_PLURALS | frozenset({"avisos", "notificaciones", "notifications", "sonido", "sound"})
_MUTE_VERBS = frozenset({"silencio", "silencia", "silenciar", "silencialo", "silencialas", "silencialos", "mute",
                         "mutea", "mutear", "calla", "callate"})
_UNMUTE_VERBS = frozenset({"unmute", "desmutea", "desmutear", "desilencia", "desilenciar"})
_OFF_VERBS = frozenset({"desactiva", "desactivar", "apaga", "apagar", "disable"})
_ON_VERBS = frozenset({"activa", "activar", "reactiva", "reactivar", "enciende", "encender", "prende", "prender",
                       "enable"})
_DELETE_VERBS = frozenset({"borra", "borrar", "borrala", "borralas", "elimina", "eliminar", "eliminala", "quita",
                           "quitar", "saca", "sacar", "delete", "del", "remove", "rm", "clear", "cancela", "cancelar",
                           "cancel", "anula", "anular"})
_MUTE_FILLERS = _NOTICE_NOUNS | frozenset({
    "los", "las", "el", "la", "the", "por", "durante", "for", "during", "de", "en", "todo", "todos", "todas", "all",
    "me", "them", "everything"})
_UNMUTE_FILLERS = _NOTICE_NOUNS | frozenset({
    "los", "las", "el", "la", "the", "de", "nuevo", "again", "otra", "vez", "on", "si", "todo", "todos", "todas",
    "all", "mis", "my", "silencio", "mute"})

# Palabras de la fase 2 que nunca se toman por moneda (además de ``_RESERVED``). No entran las palabras cortas de
# relleno ("me", "id", "si"...) ni las unidades de una letra, que sólo cuentan detrás de un número: hay monedas
# que se llaman así (ME, ID, W).
_PHASE2_WORDS = frozenset().union(
    (u for u in _UNIT_CLASS if len(u) > 1), _BACKTEST_VERBS, _BACKTEST_LEAD - {"un", "una", "a", "el", "la"},
    _ADOPT_VERBS, _DEACTIVATE_VERBS, _BACKTEST_FILLERS, _ADOPT_FILLERS, _DEACTIVATE_FILLERS, _ALERT_HEADS,
    _ABOVE_WORDS, _BELOW_WORDS, _CROSS_WORDS, _GOLDEN_WORDS, _DEATH_WORDS, _MA_TYPES, _VOLUME_WORDS, _MOVE_WORDS,
    _REPEAT_WORDS, _NOTICE_NOUNS, _MUTE_VERBS, _UNMUTE_VERBS, _ON_VERBS, _OFF_VERBS, _DELETE_VERBS,
    {"rsi", "bot", "mesa", "trading", "aviso", "notificacion", "silencio", "vez", "cada", "every", "once", "crea",
     "crear", "creame", "create", "nueva", "nuevo", "new", "anade", "agrega", "agregar", "ponme", "cuando", "nivel",
     "periodo", "periodos", "period", "periods", "pico", "spike", "entra", "entre", "enters", "hasta", "durante",
     "during", "again", "numero", "nro", "activas", "activos", "active", "longs", "shorts", "compras", "venta",
     "ventas"})
# Relleno que también es una moneda (ME): sólo es la moneda de la alarma si no aparece otra ("alerta ME > 1"
# frente a "alert me when ETH < 3000").
_SOFT_FILLERS = frozenset({"me"})


def strategy_names() -> tuple[str, ...]:
    """Estrategias válidas: las del registro ``strategy.STRATEGIES`` (si carga) más ``KNOWN_STRATEGIES``."""
    names = set(KNOWN_STRATEGIES)
    try:
        from .strategy import STRATEGIES
        names.update(str(k) for k in STRATEGIES)
    except Exception as e:  # el parser no depende de que el módulo de estrategias cargue
        log.debug("sin registro de estrategias: %s", e)
    return tuple(sorted(names))


def _strategy_match(words: list[str]) -> Optional[str]:
    names = strategy_names()
    joined = "".join(words).replace("_", "").replace("-", "")
    for name in names:
        if name.replace("_", "") == joined:
            return name
    if len(words) == 1 and len(joined) >= 3:                 # "donchian", "rsi": palabra de una sola estrategia
        hits = [name for name in names if joined in name.split("_")]
        if len(hits) == 1:
            return hits[0]
    return None


def strategy_key(name: str) -> Optional[str]:
    """Nombre canónico de una estrategia (``EMA-Trend``, ``ema trend``, ``ematrend`` -> ``ema_trend``) o None.

    También vale una palabra que sólo aparece en el nombre de una estrategia (``donchian``, ``rsi``)."""
    words = normalize(name).replace("-", " ").replace("_", " ").split()
    return _strategy_match(words) if words else None


def _strategy_at(toks: list[str], i: int) -> tuple[Optional[str], int]:
    """Estrategia que empieza en ``i`` (1 a 3 tokens: ``ema_trend``, ``ema trend``) -> (nombre, tokens)."""
    for k in (3, 2, 1):
        if i + k <= len(toks):
            name = _strategy_match(toks[i:i + k])
            if name:
                return name, k
    return None, 0


def _strategy_words() -> frozenset[str]:
    return frozenset(part for name in strategy_names() for part in name.split("_"))


def _split_compact(toks: list[str]) -> list[str]:
    """Separa sufijos pegados (``1h`` -> ``1 h``, ``1y`` -> ``1 anos``, ``rsi14`` -> ``rsi 14``) y quita ``#``."""
    out: list[str] = []
    for tok in toks:
        tok = tok.lstrip("#")
        if not tok:
            continue
        m = _COMPACT_RE.fullmatch(tok) or _IND_NUM_RE.fullmatch(tok)
        if m:
            num, unit = m.groups()
            out += [num, "anos" if unit == "y" else unit]
        else:
            out.append(tok)
    return out


def _quantity(toks: list[str], i: int) -> Optional[tuple[Optional[float], str, int]]:
    """``NÚMERO [unidad]`` (o ``una hora`` / ``media hora``) en ``i`` -> (valor, clase, índice siguiente).

    Clase: ``""`` número suelto, ``"%"``, ``"x"`` o la de ``_UNIT_CLASS``. None si en ``i`` no empieza una
    cantidad; valor None = número mal escrito o ambiguo (``1.000``): el comando entero se rechaza."""
    n, t = len(toks), toks[i]
    if t in _ONE_WORDS and i + 1 < n and len(toks[i + 1]) > 1 and toks[i + 1] in _UNIT_CLASS \
            and _UNIT_CLASS[toks[i + 1]] != "x":
        return _ONE_WORDS[t], _UNIT_CLASS[toks[i + 1]], i + 2
    if not _is_number(t):
        return None
    v = parse_number(t)
    k = _pct_at(toks, i + 1)
    if k:
        return v, "%", i + 1 + k
    if i + 1 < n and toks[i + 1] in _UNIT_CLASS:
        return v, _UNIT_CLASS[toks[i + 1]], i + 2
    return v, "", i + 1


def _interval_name(value: Optional[float], cls: str) -> Optional[str]:
    letter = _INTERVAL_LETTER.get(cls)
    if letter is None or value is None or value <= 0 or value != int(value):
        return None
    name = f"{int(value)}{letter}"
    return name if name in INTERVAL_SECONDS else None


def parse_interval(text: str) -> Optional[str]:
    """Temporalidad de vela en forma libre -> nombre de Hyperliquid o None.

    ``"1H"`` -> ``"1h"``, ``"15 min"`` -> ``"15m"``, ``"4 horas"`` -> ``"4h"``, ``"1 día"`` -> ``"1d"``;
    ``"7h"`` (no existe) -> None."""
    toks = _split_compact(tokenize(text))
    q = _quantity(toks, 0) if toks else None
    if q is None or q[2] != len(toks):
        return None
    return _interval_name(q[0], q[1])


def _phase2_coin(tok: str) -> Optional[str]:
    if tok in _PHASE2_WORDS or tok in _strategy_words():
        return None
    return _coin_token(tok)


def _whole(v: float) -> Any:
    return int(v) if v == int(v) else v


def _parse_chart(action: CommandAction, toks: list[str], raw: str) -> Optional[Command]:
    """``[verbo] MONEDA [temporalidad] [días] [estrategia]`` en cualquier orden (BACKTEST/ADOPT/DEACTIVATE).

    ``90d``/``90 días``/``3 meses``/``1 año`` son días de historia; ``15m``/``1h``/``4 horas`` temporalidades.
    ``1d``/``3d`` son las dos cosas: temporalidad salvo que ya haya otra (``backtest BTC 1h 3d`` = 3 días)."""
    fillers = {CommandAction.BACKTEST: _BACKTEST_FILLERS, CommandAction.ADOPT: _ADOPT_FILLERS}.get(
        action, _DEACTIVATE_FILLERS)
    coin: Optional[str] = None
    strategy: Optional[str] = None
    intervals: list[str] = []
    days: list[float] = []
    either: list[tuple[str, float]] = []                     # "1d": (temporalidad, días)
    i, n = 0, len(toks)
    while i < n:
        q = _quantity(toks, i)
        if q is not None:
            v, cls, i = q
            if v is None or v <= 0:
                return None
            if cls in ("m", "h", "w", "d"):
                name = _interval_name(v, cls)
                if cls == "d" and name is not None:
                    either.append((name, v))
                elif cls == "d":
                    days.append(v)                           # "90d": no es temporalidad -> días
                elif name is None:
                    return None                              # "7h", "60m": temporalidad inexistente
                else:
                    intervals.append(name)
            elif cls in _DAYS_PER:
                days.append(v * _DAYS_PER[cls])
            else:
                return None                                  # número suelto, %, "3x": ambiguo
            continue
        name, k = _strategy_at(toks, i)
        if name:
            if strategy not in (None, name):
                return None
            strategy, i = name, i + k
            continue
        if toks[i] in fillers:
            i += 1
            continue
        c = _phase2_coin(toks[i])
        if c is None or coin not in (None, c):
            return None
        coin, i = c, i + 1
    if coin is None or len(intervals) > 1 or len(days) > 1 or len(either) > 1:
        return None
    if either:
        name, v = either[0]
        if not intervals:
            intervals.append(name)
        elif not days:
            days.append(v)
        else:
            return None
    interval = intervals[0] if intervals else None
    hist = days[0] if days else None
    if action is CommandAction.BACKTEST:
        if hist is not None and hist < 1:
            return None
        return Command(action, raw=raw, coin=coin, interval=interval, days=hist, strategy=strategy)
    if hist is not None:
        return None                                          # activar/desactivar no llevan historia
    if action is CommandAction.ADOPT:
        # Poner dinero en automático exige la gráfica completa: "activa HYPE" sin temporalidad es ambiguo.
        return Command(action, raw=raw, coin=coin, interval=interval, strategy=strategy) if interval else None
    if strategy is not None:
        return None                                          # el playbook tiene una entrada por gráfica
    return Command(action, raw=raw, coin=coin, interval=interval)


def _alert_spec(toks: list[str], coin: Optional[str] = None,
                interval: Optional[str] = None) -> Optional[tuple[str, dict[str, Any]]]:
    """Condición de alarma (tokens sin el verbo) -> (moneda, especificación ``Command.alert``) o None.

    ``coin``/``interval`` (fallback IA): moneda e intervalo ya conocidos; si el texto trae otros, None. El
    ``interval`` externo sólo se usa en alarmas de velas (en las de precio/zona se ignora)."""
    toks = _split_compact(toks)
    n = len(toks)
    found: Optional[str] = None
    soft: Optional[str] = None
    nums: list[tuple[float, int]] = []                       # (valor, posición) de los números sueltos
    pcts: list[float] = []
    mults: list[float] = []
    intervals: list[str] = []
    marks: set[str] = set()                                  # "up" | "down" | "cross"
    first_cmp: Optional[int] = None
    kinds: set[str] = set()                                  # "rsi" | "ma" | "zone" | "volume" | "move"
    ma_types: set[str] = set()
    preset = False                                           # golden / death cross
    side: Optional[str] = None
    repeats: set[bool] = set()
    i = 0
    while i < n:
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < n else ""
        if t in _REPEAT_WORDS or (t, nxt) in _REPEAT_BIGRAMS:
            repeats.add(True)
            i += 2 if (t, nxt) in _REPEAT_BIGRAMS else 1
            continue
        if t == "once" or (t, nxt) == ("una", "vez"):
            repeats.add(False)
            i += 1 if t == "once" else 2
            continue
        if t == "x" and _is_number(nxt):                     # "volumen x3"
            v = parse_number(nxt)
            if v is None or v <= 0:
                return None
            mults.append(v)
            i += 2
            continue
        q = _quantity(toks, i)
        if q is not None:
            v, cls, j = q
            if v is None or v <= 0:
                return None
            if cls == "%":
                pcts.append(v)
            elif cls == "x":
                mults.append(v)
            elif cls == "":
                nums.append((v, i))
            else:
                name = _interval_name(v, cls)
                if name is None:
                    return None
                intervals.append(name)
            i = j
            continue
        if t in _ABOVE_WORDS or t in _BELOW_WORDS or t in _CROSS_WORDS:
            marks.add("up" if t in _ABOVE_WORDS else "down" if t in _BELOW_WORDS else "cross")
            first_cmp = i if first_cmp is None else first_cmp
            i += 2 if t in ("<", ">") and nxt == "=" else 1  # ">=" llega como ">" "="
            continue
        if t in _GOLDEN_WORDS or t in _DEATH_WORDS:
            kinds.add("ma")
            preset = True
            marks.add("up" if t in _GOLDEN_WORDS else "down")
        elif t == "rsi":
            kinds.add("rsi")
        elif t in _MA_TYPES:
            kinds.add("ma")
            if _MA_TYPES[t]:
                ma_types.add(_MA_TYPES[t])
        elif t in _VOLUME_WORDS:
            kinds.add("volume")
        elif t in _MOVE_WORDS:
            kinds.add("move")
        elif t in _ZONE_WORDS:
            kinds.add("zone")
        elif t in _ZONE_SIDES:
            if side not in (None, _ZONE_SIDES[t]):
                return None
            side = _ZONE_SIDES[t]
        elif t in _SOFT_FILLERS:
            soft = soft or _coin_token(t)
        elif t not in _ALERT_FILLERS:
            c = _phase2_coin(t)
            if c is None or found not in (None, c):
                return None
            found = c
        i += 1
    found = found or soft

    if found is not None and coin is not None and normalize_coin(coin) != found:
        return None
    coin = normalize_coin(coin) if coin is not None else found
    if coin is None or len(kinds) > 1 or len(intervals) > 1 or len(repeats) > 1 or {"up", "down"} <= marks:
        return None
    tag = next(iter(kinds)) if kinds else ("move" if pcts else "price")
    if (pcts and tag != "move") or (mults and tag != "volume") or (side is not None and tag != "zone"):
        return None
    direction = "up" if "up" in marks else "down" if "down" in marks else None
    text_iv = intervals[0] if intervals else None
    if interval is not None and text_iv is not None and interval != text_iv:
        return None
    spec: dict[str, Any]
    if tag == "price":
        if text_iv is not None or len(nums) != 1:            # el precio se vigila cada ciclo, no por velas
            return None
        kind = {"up": "price_above", "down": "price_below"}.get(direction or "", "price_cross")
        spec = {"kind": kind, "level": nums[0][0]}
    elif tag == "rsi":
        if direction is None:                                # el RSI se cruza hacia arriba o hacia abajo
            return None
        if len(nums) == 1:
            spec = {"level": nums[0][0]}
        elif len(nums) == 2 and first_cmp is not None and nums[0][1] < first_cmp < nums[1][1]:
            spec = {"level": nums[1][0], "period": _whole(nums[0][0])}      # "rsi 14 < 30"
        else:
            return None
        spec = {"kind": "rsi_above" if direction == "up" else "rsi_below", **spec}
    elif tag == "ma":
        if len(ma_types) > 1 or len(nums) not in (0, 2):
            return None
        spec = {"kind": "ma_cross"}
        if nums:
            fast, slow = sorted(v for v, _ in nums)          # "ema 50 20" = la misma pareja
            spec["fast"], spec["slow"] = _whole(fast), _whole(slow)
        elif preset:
            spec["fast"], spec["slow"] = 50, 200
        if ma_types or preset:
            spec["ma"] = next(iter(ma_types)) if ma_types else "sma"
        if direction:
            spec["direction"] = direction
    elif tag == "zone":
        if nums or text_iv is not None or direction:
            return None
        spec = {"kind": "zone"}
        if side:
            spec["side"] = side
    elif tag == "volume":
        if nums or len(mults) > 1 or direction == "down" or "cross" in marks:
            return None
        spec = {"kind": "volume_spike"}
        if mults:
            spec["mult"] = mults[0]
    else:
        if nums or len(pcts) != 1 or "cross" in marks:
            return None
        spec = {"kind": "move_pct", "pct": pcts[0]}
        if direction:
            spec["direction"] = direction
    spec["interval"] = (text_iv or interval) if spec["kind"] in CANDLE_ALERT_KINDS else None
    spec["repeat"] = True in repeats
    return coin, spec


def _parse_alert_add(toks: list[str], raw: str) -> Optional[Command]:
    r = _alert_spec(toks)
    if r is None:
        return None
    return Command(CommandAction.ALERT_ADD, raw=raw, coin=r[0], alert=r[1])


def _parse_alert_del(toks: list[str], raw: str) -> Optional[Command]:
    """``borra alerta 3`` -> ``target_id=3``; ``borra alertas`` / ``borra todas las alarmas`` -> todas."""
    target: Optional[int] = None
    plural = every = False
    for t in toks:
        if t in _ALERT_PLURALS:
            plural = True
        elif t in _ALERT_NOUNS:
            continue
        elif t in _ALL_WORDS:
            every = True
        elif _is_number(t):
            v = parse_number(t)
            if v is None or v < 1 or v != int(v) or target is not None:
                return None
            target = int(v)
        elif t not in _ALERT_DEL_FILLERS:
            return None
    if target is not None:
        return None if every else Command(CommandAction.ALERT_DEL, raw=raw, target_id=target)
    # "borra alerta" (singular, sin número): ¿cuál? -> None
    return Command(CommandAction.ALERT_DEL, raw=raw) if plural or every else None


def _parse_mute(toks: list[str], raw: str, *, bare_minutes: bool = True) -> Optional[Command]:
    """``silencio [duración]`` -> MUTE con ``value`` en minutos (None = hasta reactivar).

    ``bare_minutes``: un número sin unidad son minutos (``mute 15``); tras "desactiva alertas" se exige unidad."""
    minutes: Optional[float] = None
    i, n = 0, len(toks)
    while i < n:
        q = _quantity(toks, i)
        if q is not None:
            v, cls, i = q
            per = _MINUTES_PER.get(cls)
            if v is None or v <= 0 or per is None or minutes is not None or (cls == "" and not bare_minutes):
                return None
            minutes = v * per
            continue
        if toks[i] not in _MUTE_FILLERS:
            return None
        i += 1
    return Command(CommandAction.MUTE, raw=raw, value=minutes)


def _parse_unmute(toks: list[str], raw: str) -> Optional[Command]:
    return Command(CommandAction.UNMUTE, raw=raw) if all(t in _UNMUTE_FILLERS for t in toks) else None


def _parse_phase2(toks: list[str], raw: str) -> tuple[bool, Optional[Command]]:
    """Gramáticas de la fase 2 (backtest, playbook, alarmas, silencio).

    -> ``(False, None)`` si el texto no es de la fase 2 (siguen las demás gramáticas) o ``(True, cmd)``: un
    comando de la fase 2 mal formado es None, nunca se reinterpreta como otra cosa."""
    toks = _split_compact(toks)
    if not toks:
        return False, None
    t0, rest = toks[0], toks[1:]
    notice = any(t in _NOTICE_NOUNS for t in rest)
    if t0 in _MUTE_VERBS:
        return True, _parse_mute(rest, raw)
    if t0 in _UNMUTE_VERBS:
        return True, _parse_unmute(rest, raw)
    if t0 in ("sonido", "sound"):
        if rest in (["off"], ["no"], ["desactivado"]):
            return True, Command(CommandAction.MUTE, raw=raw)
        return True, _parse_unmute(rest, raw)
    if toks == ["sin", "sonido"]:
        return True, Command(CommandAction.MUTE, raw=raw)
    # "desactiva/apaga los avisos [1h]", "activa avisos": antes que desactivar/activar una gráfica.
    if toks[:2] == ["turn", "off"] and any(t in _NOTICE_NOUNS for t in toks[2:]):
        return True, _parse_mute(toks[2:], raw, bare_minutes=False)
    if toks[:2] == ["turn", "on"] and any(t in _NOTICE_NOUNS for t in toks[2:]):
        return True, _parse_unmute(toks[2:], raw)
    if t0 in _OFF_VERBS and notice:
        return True, _parse_mute(rest, raw, bare_minutes=False)
    if t0 in _ON_VERBS and notice:
        return True, _parse_unmute(rest, raw)
    if t0 in _DELETE_VERBS:
        if "silencio" in rest or "mute" in rest:             # "quita el silencio"
            return True, _parse_unmute(rest, raw)
        if any(t in _ALERT_NOUNS for t in rest):
            return True, _parse_alert_del(rest, raw)
        if "playbook" in rest:                               # "quita HYPE 1h del playbook"
            return True, _parse_chart(CommandAction.DEACTIVATE, rest, raw)
        return False, None                                   # "cancela HYPE", "cancel all": órdenes
    k = _skip(toks, 0, _ALERT_LEAD)
    if k < len(toks) and toks[k] in _ALERT_HEADS:
        return True, _parse_alert_add(toks[k + 1:], raw)
    k = _skip(toks, 0, _BACKTEST_LEAD)
    if k < len(toks) and toks[k] in _BACKTEST_VERBS:
        return True, _parse_chart(CommandAction.BACKTEST, toks[k + 1:], raw)
    if t0 in _DEACTIVATE_VERBS:
        return True, _parse_chart(CommandAction.DEACTIVATE, rest, raw)
    if t0 in _ADOPT_VERBS:
        return True, _parse_chart(CommandAction.ADOPT, rest, raw)
    return False, None


def parse_rules(text: str, settings: Optional[Settings] = None) -> Optional[Command]:
    """Interpreta ``text`` sólo con reglas (sin IA). None si no lo entiende sin ambigüedad."""
    raw = (text or "").strip()
    if not raw:
        return None
    if raw in {"?", "¿?"}:
        return Command(CommandAction.HELP, raw=raw)
    toks = tokenize(raw)
    if not toks:
        return None
    joined = " ".join(toks)
    if _RESET_PARAMS_RX.fullmatch(joined):
        return Command(CommandAction.SET_PARAM, raw=raw, param=PARAM_DEFAULTS)
    for action, rx in _PHRASES:
        if rx.fullmatch(joined):
            return Command(action, raw=raw)
    handled, cmd = _parse_phase2(toks, raw)              # antes que cancelar: "cancela la alerta 3"
    if handled:
        return cmd
    if toks[0] in _CLOSE_VERBS:
        return _parse_close(toks, raw)
    if toks[0] in _CANCEL_VERBS:
        return _parse_cancel(toks, raw)
    k = _skip(toks, 0, _SET_VERBS | _ARTICLES)        # "mueve el stop ...", "pon el apalancamiento ..."
    if k < len(toks) and (_sl_len(toks, k) or _tp_len(toks, k)):
        return _parse_level_cmd(toks[k:], raw)
    if any(t in _SIDE_WORDS for t in toks):
        return _parse_open(toks, raw, settings)
    return _parse_param(toks[k:], raw)


# --- interfaz pública ---------------------------------------------------------------

class LLMInterpreter(Protocol):
    """Lo mínimo que necesita ``parse`` de un intérprete IA (inyectable en tests)."""

    def interpret(self, text: str) -> Optional[Command]: ...


def parse(text: str, settings: Optional[Settings], llm: Optional[LLMInterpreter] = None) -> Optional[Command]:
    """Reglas primero; si no entienden, ``llm`` (si se pasa y ``settings.llm_enabled``).

    Nunca lanza: un fallo del intérprete IA se registra y devuelve None.
    """
    cmd = parse_rules(text, settings)
    if cmd is not None or llm is None or not getattr(settings, "llm_enabled", False):
        return cmd
    if not (text or "").strip():
        return None
    try:
        return llm.interpret(text)
    except Exception as e:  # el intérprete no debe tumbar la mesa
        log.warning("intérprete IA falló: %s", e)
        return None


class CommandParser:
    """Parser con fallback IA. ``llm=None`` crea un ``LLMCommandInterpreter`` (que sólo
    actúa si ``settings.llm_enabled`` y hay SDK + credenciales); ``use_llm=False`` lo omite."""

    def __init__(self, settings: Settings, llm: Optional[LLMInterpreter] = None, use_llm: bool = True):
        self.settings = settings
        self.llm: Optional[LLMInterpreter] = llm if llm is not None else (
            LLMCommandInterpreter(settings) if use_llm else None)

    def parse(self, text: str) -> Optional[Command]:
        return parse(text, self.settings, self.llm)

    __call__ = parse


# --- intérprete IA (Claude) ---------------------------------------------------------

# La IA no puede confirmar comandos pendientes ni levantar un halt de riesgo: eso
# exige la palabra exacta del operador.
LLM_ACTIONS: tuple[str, ...] = tuple(
    a.value for a in CommandAction if a not in (CommandAction.CONFIRM, CommandAction.RESET_HALT))

_NUMBER_FIELDS = ("units", "price", "sl_price", "sl_pct", "tp_price", "tp_pct", "value", "days")


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


# Cada campo es "tipo o null" (una unión). La API limita las uniones de un esquema estricto, así que la fase 2
# añade sólo cuatro campos: ``alert`` es la condición como texto con la sintaxis de las reglas (la valida el
# mismo parser determinista) y ``value`` hace de minutos en MUTE y de número de alarma en ALERT_DEL.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": _nullable({"type": "string", "enum": list(LLM_ACTIONS)}),
        "coin": _nullable({"type": "string"}),
        "side": _nullable({"type": "string", "enum": [s.value for s in Side]}),
        "units": _nullable({"type": "number"}),
        "entry_mode": _nullable({"type": "string", "enum": [m.value for m in EntryMode]}),
        "price": _nullable({"type": "number"}),
        "sl_price": _nullable({"type": "number"}),
        "sl_pct": _nullable({"type": "number"}),
        "tp_price": _nullable({"type": "number"}),
        "tp_pct": _nullable({"type": "number"}),
        "param": _nullable({"type": "string", "enum": sorted(TUNABLE)}),
        "value": _nullable({"type": "number"}),
        "interval": _nullable({"type": "string", "enum": list(INTERVALS)}),
        "strategy": _nullable({"type": "string"}),
        "days": _nullable({"type": "number"}),
        "alert": _nullable({"type": "string"}),
    },
    "required": ["action", "coin", "side", "units", "entry_mode", "price", "sl_price", "sl_pct",
                 "tp_price", "tp_pct", "param", "value", "interval", "strategy", "days", "alert"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = f"""\
Eres el intérprete de comandos de una mesa de trading de futuros perpetuos (Hyperliquid).
Recibes una instrucción del operador, en español o en inglés, y la conviertes en UN comando
JSON con el esquema dado. Rellena con null todo campo que no aplique.

Acciones:
- open: abrir posición. side ("long"/"short") y coin obligatorios. units sólo si el operador
  da un número de unidades. entry_mode: "zone" (en la zona de liquidez), "market" (ya, a
  mercado) o "limit" (con price obligatorio); null si no lo dice.
  Stop-loss / take-profit: si se dan en porcentaje van en sl_pct / tp_pct (1.5 para 1,5 %);
  si se dan como precio, en sl_price / tp_price.
- close: cerrar la posición de coin. close_all: cerrar todas.
- cancel: cancelar órdenes abiertas de coin (coin null = todas).
- set_sl / set_tp: mover el stop-loss / take-profit de la posición abierta en coin: price si
  es un precio, sl_pct / tp_pct si es un porcentaje.
- pause / resume: dejar de abrir operaciones / volver a operar. auto_on / auto_off: activar o
  desactivar las estrategias automáticas.
- status: estado de la cuenta. help: ayuda. reject: descartar el comando pendiente.
- kill: pánico (cerrar todo, cancelar todo y detener).
- set_param: cambiar un parámetro; param es uno de: {", ".join(sorted(TUNABLE))}; value es el número.
- backtest: backtest + optimización walk-forward de una gráfica. coin obligatorio; interval = temporalidad
  de las velas (null = la de la mesa); days = días de historia (3 meses = 90, 1 año = 365; null = por
  defecto); strategy = una de: {", ".join(KNOWN_STRATEGIES)} (null = todas).
- adopt: operar en automático la gráfica coin + interval (los dos obligatorios) con la estrategia ganadora
  de su último backtest, o con strategy si el operador la nombra.
- deactivate: quitar coin del playbook (interval = sólo esa temporalidad; null = todas).
- playbook: ver las gráficas en automático. reports: ver los últimos informes de backtest.
- alert_add: crear una alarma sobre coin. alert = la condición con esta sintaxis, sin la moneda:
  "> 90", "< 80000", "cruza 150", "rsi < 30", "rsi > 70", "cruce ema 20 50", "cruce sma 50 200 alcista",
  "zona", "zona long", "volumen 3x", "mueve 3%", "sube 3%", "cae 3%"; números sin separador de miles y con
  punto decimal; añade " siempre" si debe avisar cada vez (si no, avisa una vez y se borra). interval =
  temporalidad de las velas en las alarmas de rsi, cruce, volumen y movimiento (null en precio y zona).
- alert_list: ver las alarmas. alert_del: borrar la alarma número value (value null = todas).
- mute: silenciar los avisos no críticos value minutos (null = hasta reactivarlos). unmute: reactivarlos.
- Temporalidades válidas: {", ".join(INTERVALS)}.

Reglas:
- coin: ticker en mayúsculas (BTC, ETH, SOL, HYPE). Activos HIP-3 como dex:TICKER (xyz:GOLD).
- Nunca inventes precios, cantidades ni monedas que el operador no haya dicho.
- Si el texto no es una instrucción para la mesa, es ambiguo, pide varias acciones a la vez,
  sólo confirma algo pendiente o pide levantar un bloqueo de riesgo: action null.
"""


def _sdk_installed() -> bool:
    return importlib.util.find_spec("anthropic") is not None


def has_anthropic_credentials(env: Optional[Mapping[str, str]] = None) -> bool:
    """¿Hay credenciales que el SDK pueda resolver? (variables de entorno o perfil en disco).

    Cubre ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``, perfiles (``ANTHROPIC_PROFILE``,
    ``ANTHROPIC_CONFIG_DIR`` o ``~/.config/anthropic/configs/*.json`` tras ``ant auth login``)
    y federación de identidad (``ANTHROPIC_FEDERATION_RULE_ID`` + ``ANTHROPIC_ORGANIZATION_ID``).
    """
    env = os.environ if env is None else env
    if any(env.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE",
                                "ANTHROPIC_CONFIG_DIR")):
        return True
    if env.get("ANTHROPIC_FEDERATION_RULE_ID") and env.get("ANTHROPIC_ORGANIZATION_ID"):
        return True
    home = env.get("HOME")
    if not home:
        return False
    try:
        return any((Path(home) / ".config" / "anthropic" / "configs").glob("*.json"))
    except OSError:
        return False


def _sdk_error_types() -> tuple[type[BaseException], type[BaseException], type[BaseException]]:
    """(RateLimitError, APIStatusError, APIConnectionError) del SDK, o marcadores si no está."""
    try:
        import anthropic
    except ImportError:
        class _NoSDK(Exception):
            pass
        return _NoSDK, _NoSDK, _NoSDK
    return anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError


def _num_field(data: Mapping[str, Any], key: str) -> tuple[bool, Optional[float]]:
    """(válido, valor): None se acepta; si hay número debe ser finito y > 0 (bool no vale)."""
    v = data.get(key)
    if v is None:
        return True, None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False, None
    v = float(v)
    return (math.isfinite(v) and v > 0), v


def command_from_llm(data: Any, raw: str, settings: Optional[Settings] = None) -> Optional[Command]:
    """Valida el JSON devuelto por la IA y construye el ``Command`` (``source="llm"``).

    Rechaza (None): acción desconocida/no permitida o null, lado inválido, números no
    positivos, monedas que no parecen tickers, campos obligatorios ausentes o
    combinaciones contradictorias (p. ej. ``limit`` sin precio).
    """
    if not isinstance(data, Mapping):
        return None
    action_s = data.get("action")
    if action_s not in LLM_ACTIONS:
        return None
    action = CommandAction(action_s)
    nums: dict[str, Optional[float]] = {}
    for key in _NUMBER_FIELDS:
        ok, v = _num_field(data, key)
        if not ok:
            return None
        nums[key] = v
    for key in ("sl_pct", "tp_pct"):
        if nums[key] is not None and nums[key] >= 100:
            return None
    coin: Optional[str] = None
    if data.get("coin") is not None:
        if not isinstance(data["coin"], str):
            return None
        coin = normalize_coin(data["coin"].strip())
        if coin is None:
            return None
    cmd = Command(action, raw=raw, source="llm")

    if action is CommandAction.OPEN:
        try:
            side = Side(data.get("side"))
            mode = EntryMode(data["entry_mode"]) if data.get("entry_mode") is not None else None
        except (ValueError, TypeError):
            return None
        if coin is None:
            return None
        price = nums["price"]
        if price is not None and mode in (None, EntryMode.LIMIT):
            mode = EntryMode.LIMIT
        elif price is not None or mode is EntryMode.LIMIT:
            return None                               # limit sin precio, o precio con zona/mercado
        if (nums["sl_price"] and nums["sl_pct"]) or (nums["tp_price"] and nums["tp_pct"]):
            return None
        cmd.coin, cmd.side, cmd.units, cmd.price = coin, side, nums["units"], price
        cmd.entry_mode = mode or _default_entry_mode(settings)
        cmd.sl_price, cmd.sl_pct = nums["sl_price"], nums["sl_pct"]
        cmd.tp_price, cmd.tp_pct = nums["tp_price"], nums["tp_pct"]
    elif action in (CommandAction.SET_SL, CommandAction.SET_TP):
        pct = nums["sl_pct"] if action is CommandAction.SET_SL else nums["tp_pct"]
        price = nums["price"]
        if price is None and action is CommandAction.SET_SL:
            price = nums["sl_price"]
        elif price is None:
            price = nums["tp_price"]
        if coin is None or (price is None) == (pct is None):
            return None                               # hace falta exactamente uno: precio o %
        cmd.coin, cmd.price = coin, price
        if action is CommandAction.SET_SL:
            cmd.sl_pct = pct
        else:
            cmd.tp_pct = pct
    elif action is CommandAction.CLOSE:
        if coin is None:
            return None
        cmd.coin = coin
    elif action is CommandAction.CANCEL:
        cmd.coin = coin
    elif action is CommandAction.SET_PARAM:
        key = param_key(data["param"]) if isinstance(data.get("param"), str) else None
        if key is None or nums["value"] is None:
            return None
        cmd.param, cmd.value = key, nums["value"]
    elif action in (CommandAction.BACKTEST, CommandAction.ADOPT, CommandAction.DEACTIVATE,
                    CommandAction.ALERT_ADD):
        interval = data.get("interval")
        if coin is None or (interval is not None and interval not in INTERVAL_SECONDS):
            return None
        if action is CommandAction.ALERT_ADD:
            text = data.get("alert")
            toks = tokenize(text) if isinstance(text, str) else []
            toks = toks[_skip(toks, 0, _ALERT_LEAD | _ALERT_HEADS):]      # "alerta HYPE > 90" vale igual
            spec = _alert_spec(toks, coin=coin, interval=interval) if toks else None
            if spec is None:
                return None                           # la condición pasa por el mismo parser que las reglas
            cmd.coin, cmd.alert = spec
            return cmd
        strategy: Optional[str] = None
        if action is not CommandAction.DEACTIVATE and data.get("strategy") is not None:
            strategy = strategy_key(data["strategy"]) if isinstance(data["strategy"], str) else None
            if strategy is None:
                return None                           # estrategia desconocida: no se cambia por otra
        cmd.coin, cmd.interval, cmd.strategy = coin, interval, strategy
        if action is CommandAction.BACKTEST:
            if nums["days"] is not None and nums["days"] < 1:
                return None
            cmd.days = nums["days"]
        elif action is CommandAction.ADOPT and interval is None:
            return None
    elif action is CommandAction.ALERT_DEL:
        target = nums["value"]
        if target is not None and target != int(target):
            return None
        cmd.target_id = int(target) if target is not None else None
    elif action is CommandAction.MUTE:
        cmd.value = nums["value"]
    return cmd


class LLMCommandInterpreter:
    """Fallback IA con Claude: texto libre -> ``Command`` (``source="llm"``) o None.

    - Sólo actúa si ``settings.llm_enabled`` y hay cliente (inyectado) o SDK
      ``anthropic`` + credenciales (``available()``).
    - Salida estructurada (``output_config.format`` json_schema = ``SCHEMA``), esfuerzo
      bajo y ``fallbacks="default"`` (si el modelo declina por política, la API reintenta
      con el modelo de respaldo recomendado). ``stop_reason == "refusal"`` -> None.
    - Nunca lanza: errores de la API, de red o JSON inválido -> None (``last_error``).
    - ``client`` inyectable (cualquier objeto con ``beta.messages.create``) para tests.
    """

    max_tokens = 1024
    max_chars = 500                                  # un comando no es un ensayo: no se envía más

    def __init__(self, settings: Settings, client: Any = None, env: Optional[Mapping[str, str]] = None,
                 timeout: float = 30.0, max_retries: int = 2):
        self.settings = settings
        self._client = client
        self._env = env
        self.timeout = timeout
        self.max_retries = max_retries
        self.last_error = ""

    def available(self) -> bool:
        if not self.settings.llm_enabled:
            return False
        if self._client is not None:
            return True
        return _sdk_installed() and has_anthropic_credentials(self._env)

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic
            # Credenciales del entorno (API key, token o perfil). Timeout corto: la mesa
            # no puede quedarse 10 min colgada esperando a la IA.
            self._client = anthropic.Anthropic(timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def _fail(self, msg: str) -> None:
        self.last_error = msg
        log.warning("IA: %s", msg)

    def interpret(self, text: str) -> Optional[Command]:
        raw = (text or "").strip()
        self.last_error = ""
        if not raw or not self.available():
            return None
        if len(raw) > self.max_chars:
            self._fail(f"texto demasiado largo para un comando ({len(raw)} caracteres)")
            return None
        rate_err, status_err, conn_err = _sdk_error_types()
        try:
            resp = self._get_client().beta.messages.create(
                model=self.settings.llm_model,
                max_tokens=self.max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": raw}],
            )
        except rate_err as e:
            self._fail(f"límite de peticiones de la API (429): {e}")
            return None
        except status_err as e:
            self._fail(f"la API respondió con error {getattr(e, 'status_code', '?')}: {e}")
            return None
        except conn_err as e:
            self._fail(f"sin conexión con la API: {e}")
            return None
        except Exception as e:  # credenciales rotas, SDK incompatible, ...
            self._fail(f"error inesperado llamando a la API: {e!r}")
            return None
        return self._command_from_response(resp, raw)

    def _command_from_response(self, resp: Any, raw: str) -> Optional[Command]:
        try:
            stop = getattr(resp, "stop_reason", None)
            if stop == "refusal":
                details = getattr(resp, "stop_details", None)
                self._fail(f"el modelo declinó interpretar el comando ({getattr(details, 'category', None)})")
                return None
            if stop == "max_tokens":
                self._fail("respuesta truncada (max_tokens)")
                return None
            text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
            if text is None:
                self._fail("respuesta sin texto")
                return None
            data = json.loads(text)
        except Exception as e:
            self._fail(f"respuesta ilegible: {e!r}")
            return None
        cmd = command_from_llm(data, raw, self.settings)
        if cmd is None and isinstance(data, Mapping) and data.get("action") is not None:
            self._fail(f"comando de la IA descartado por inválido: {data}")
        return cmd


# --- textos para el operador -------------------------------------------------------

HELP_TEXT = """\
Comandos (español o inglés, sin importar mayúsculas ni acentos):
  long HYPE | short btc | compra SOL | vende ETH        abrir (entrada por defecto)
    ... 2u | 2 unidades | 2 units                        tamaño en unidades
    ... zona | en zona de liquidez | a mercado | ya       modo de entrada
    ... a 85.2 | @85.2 | limit 85,2                       orden límite
    ... sl 1.5% | stop 84.5 | tp 4% | tp 90               stop-loss / take-profit (% o precio)
    ej.: long HYPE 2u zona sl 1.5% tp 4%
  cierra HYPE | close HYPE | cierra todo | close all | flat
  cancela HYPE | cancel all
  stop HYPE 84.5 | sl HYPE 1% | tp HYPE 90              mover SL/TP de una posición
  apalancamiento 3 | riesgo 0.5 | unidad 250 | sl 2% | tp 3   parámetros
  reset params | restablece parametros                 volver a los parámetros de arranque
  pausa | reanuda | auto on | auto off | estado | kill | reset halt
  si / confirmar  -  no / cancelar  (comandos pendientes)   ayuda
Gráficas (moneda + temporalidad: 1m 3m 5m 15m 30m 1h 2h 4h 8h 12h 1d 3d 1w):
  backtest HYPE 1h | backtest BTC 15m 90d | optimiza ETH 1h 3 meses   backtest + walk-forward (todas)
  prueba HYPE 1h ema_trend                                          una sola estrategia
  informes | ultimos backtests                                      informes guardados
  activa HYPE 1h | usa la mejor HYPE 1h | activa HYPE 1h ema_trend   operar esa gráfica en automático
  desactiva HYPE 1h | desactiva HYPE | playbook                     quitar / ver gráficas activas
    (un backtest no garantiza resultados: decide con las métricas FUERA de muestra)
Alarmas (alerta o alarma):
  alerta HYPE > 90 | alerta BTC < 80000 | alerta SOL cruza 150      precio
  alerta HYPE rsi < 30 1h | alerta BTC cruce ema 20 50 1h alcista   indicadores (velas cerradas)
  alerta SOL zona long | alerta BTC volumen 3x 15m | alerta HYPE mueve 3% 1h
    ... siempre | cada vez                                          repetir (si no, avisa una vez)
  alertas | borra alerta 3 | borra alertas                          ver / borrar
  silencio 30m | silencio | sonido                                  silenciar avisos no críticos"""


def _fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:g}"


def _fmt_minutes(minutes: float) -> str:
    if minutes >= 1440 and minutes % 1440 == 0:
        return f"{minutes / 1440:g} d"
    if minutes >= 60 and minutes % 60 == 0:
        return f"{minutes / 60:g} h"
    return f"{minutes:g} min"


def describe_alert(coin: Optional[str], spec: Mapping[str, Any]) -> str:
    """Resumen de una especificación de alarma (``Command.alert``): ``HYPE RSI < 30 en 1h, una vez``."""
    kind, c = spec.get("kind"), coin or "?"
    iv = f" en {spec['interval']}" if spec.get("interval") else ""
    way = {"up": " al alza", "down": " a la baja"}.get(spec.get("direction") or "", "")
    if kind in ("price_above", "price_below", "price_cross"):
        op = {"price_above": ">", "price_below": "<"}.get(kind, "cruza")
        what = f"{c} {op} {_fmt(spec.get('level'))}"
    elif kind in ("rsi_above", "rsi_below"):
        period = f"({_fmt(spec['period'])})" if spec.get("period") is not None else ""
        what = f"{c} RSI{period} {'>' if kind == 'rsi_above' else '<'} {_fmt(spec.get('level'))}{iv}"
    elif kind == "ma_cross":
        ma = str(spec["ma"]).upper() if spec.get("ma") else "medias"
        pair = f" {_fmt(spec['fast'])}/{_fmt(spec['slow'])}" if spec.get("fast") is not None else ""
        what = f"{c} cruce{way} {ma}{pair}{iv}"
    elif kind == "zone":
        side = spec.get("side")
        what = f"{c} entra en zona de liquidez {str(side).upper() if side else 'LONG o SHORT'}"
    elif kind == "volume_spike":
        mult = f" {_fmt(spec['mult'])}x" if spec.get("mult") is not None else ""
        what = f"{c} pico de volumen{mult}{iv}"
    elif kind == "move_pct":
        what = f"{c} se mueve {_fmt(spec.get('pct'))}%{way}{iv}"
    else:
        what = f"{c} {kind}"
    return what + (", siempre" if spec.get("repeat") else ", una vez")


def describe(cmd: Command) -> str:
    """Resumen legible de un comando (para confirmaciones y respuestas)."""
    a = cmd.action
    if a is CommandAction.OPEN:
        parts = [f"{cmd.side.value.upper() if cmd.side else '?'} {cmd.coin}"]
        if cmd.units is not None:
            parts.append(f"{_fmt(cmd.units)}u")
        if cmd.entry_mode is EntryMode.LIMIT:
            parts.append(f"límite {_fmt(cmd.price)}")
        elif cmd.entry_mode is EntryMode.MARKET:
            parts.append("a mercado")
        elif cmd.entry_mode is EntryMode.ZONE:
            parts.append("en zona de liquidez")
        if cmd.sl_pct is not None or cmd.sl_price is not None:
            parts.append(f"SL {_fmt(cmd.sl_pct)}%" if cmd.sl_pct is not None else f"SL {_fmt(cmd.sl_price)}")
        if cmd.tp_pct is not None or cmd.tp_price is not None:
            parts.append(f"TP {_fmt(cmd.tp_pct)}%" if cmd.tp_pct is not None else f"TP {_fmt(cmd.tp_price)}")
        return "abrir " + " ".join(parts)
    if a in (CommandAction.SET_SL, CommandAction.SET_TP):
        pct = cmd.sl_pct if a is CommandAction.SET_SL else cmd.tp_pct
        what = "stop-loss" if a is CommandAction.SET_SL else "take-profit"
        level = f"{_fmt(pct)}% desde la entrada" if pct is not None else _fmt(cmd.price)
        return f"mover {what} de {cmd.coin} a {level}"
    if a is CommandAction.SET_PARAM:
        if cmd.param == PARAM_DEFAULTS:
            return "volver a los parámetros de arranque"
        return f"cambiar {cmd.param} a {_fmt(cmd.value)}"
    if a is CommandAction.CLOSE:
        return f"cerrar {cmd.coin}"
    if a is CommandAction.CANCEL:
        return f"cancelar órdenes de {cmd.coin}" if cmd.coin else "cancelar todas las órdenes"
    if a is CommandAction.BACKTEST:
        extra = [f"{_fmt(cmd.days)} días" if cmd.days else "historia por defecto",
                 f"estrategia {cmd.strategy}" if cmd.strategy else "todas las estrategias"]
        return (f"backtest + optimización walk-forward de {cmd.coin} {cmd.interval or '(temporalidad por defecto)'}"
                f" ({', '.join(extra)})")
    if a is CommandAction.ADOPT:
        who = f"la estrategia {cmd.strategy}" if cmd.strategy else "la ganadora de su último backtest"
        return f"operar en automático {cmd.coin} {cmd.interval} con {who}"
    if a is CommandAction.DEACTIVATE:
        if cmd.interval:
            return f"quitar {cmd.coin} {cmd.interval} del playbook"
        return f"quitar {cmd.coin} del playbook (todas sus temporalidades)"
    if a is CommandAction.ALERT_ADD:
        return "crear alarma: " + describe_alert(cmd.coin, cmd.alert or {})
    if a is CommandAction.ALERT_DEL:
        return f"borrar la alarma #{cmd.target_id}" if cmd.target_id is not None else "borrar todas las alarmas"
    if a is CommandAction.MUTE:
        if cmd.value:
            return f"silenciar los avisos no críticos {_fmt_minutes(cmd.value)}"
        return "silenciar los avisos no críticos hasta 'sonido'"
    return {
        CommandAction.CLOSE_ALL: "cerrar todas las posiciones",
        CommandAction.PAUSE: "pausar nuevas entradas",
        CommandAction.RESUME: "reanudar",
        CommandAction.AUTO_ON: "activar operativa automática",
        CommandAction.AUTO_OFF: "desactivar operativa automática",
        CommandAction.STATUS: "estado",
        CommandAction.KILL: "PÁNICO: cerrar todo, cancelar todo y detener",
        CommandAction.RESET_HALT: "levantar el halt de riesgo",
        CommandAction.CONFIRM: "confirmar",
        CommandAction.REJECT: "descartar",
        CommandAction.HELP: "ayuda",
        CommandAction.PLAYBOOK: "ver el playbook (gráficas en automático)",
        CommandAction.REPORTS: "ver los últimos informes de backtest",
        CommandAction.ALERT_LIST: "ver las alarmas",
        CommandAction.UNMUTE: "reactivar los avisos",
    }.get(a, a.value)

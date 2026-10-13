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
  ``value`` un float; el rango lo valida ``Settings.set_param``.
- Números con punto o coma decimal (``85.2`` / ``85,2``); los ambiguos entre
  miles y decimales (``83,770``, ``1.000``) se rechazan (ver ``parse_number``).
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

_RESERVED = frozenset().union(
    _SIDE_WORDS, _UNIT_WORDS, _ZONE_WORDS, _MARKET_WORDS, _LIMIT_WORDS, _LIMIT_PREPS, _SL_WORDS,
    _TP_WORDS, _PCT_WORDS, _ALL_WORDS, _ORDER_WORDS, _ARTICLES, _POSITION_WORDS, _OPEN_FILLERS,
    _CLOSE_VERBS, _CLOSE_FILLERS, _CANCEL_VERBS, _CANCEL_FILLERS, _SET_VERBS, _LEVEL_FILLERS,
    _PARAM_CONNECTORS, _ANY_PARAM_UNIT, _POLITENESS,
    {"loss", "take", "profit", "por", "ciento", "x", "r", "posiciones", "positions"})

# --- frases fijas (todo el texto normalizado debe coincidir) -------------------

_AUTO = r"(?:auto|automatico|autonomo|autopilot|autopiloto|automatic|autotrading|auto trading)"
_ON = r"(?:on|si|yes|activado|activo|activar|encendido|encender|enable|enabled|true|1)"
_OFF = r"(?:off|no|desactivado|desactivar|apagado|apagar|disable|disabled|false|0)"

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

    Separa ``@``, ``%`` y ``=``; parte sufijos pegados al número (``2u``, ``3x``,
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
    s = re.sub(r"[@%=]", lambda m: f" {m.group()} ", s)
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
    for action, rx in _PHRASES:
        if rx.fullmatch(joined):
            return Command(action, raw=raw)
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

_NUMBER_FIELDS = ("units", "price", "sl_price", "sl_pct", "tp_price", "tp_pct", "value")


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


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
    },
    "required": ["action", "coin", "side", "units", "entry_mode", "price", "sl_price", "sl_pct",
                 "tp_price", "tp_pct", "param", "value"],
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
  pausa | reanuda | auto on | auto off | estado | kill | reset halt
  si / confirmar  -  no / cancelar  (comandos pendientes)   ayuda"""


def _fmt(x: Optional[float]) -> str:
    return "" if x is None else f"{x:g}"


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
        return f"cambiar {cmd.param} a {_fmt(cmd.value)}"
    if a is CommandAction.CLOSE:
        return f"cerrar {cmd.coin}"
    if a is CommandAction.CANCEL:
        return f"cancelar órdenes de {cmd.coin}" if cmd.coin else "cancelar todas las órdenes"
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
    }.get(a, a.value)

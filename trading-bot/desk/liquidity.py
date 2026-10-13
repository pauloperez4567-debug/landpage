"""Zonas de fuerte liquidez: dónde quiere entrar la mesa.

Una "zona de fuerte liquidez" es un tramo de precio donde se acumulan:

- precios de liquidación de posiciones grandes (fuente principal, peso
  ``zone_liq_weight``): al llegar ahí el mercado fuerza ventas (longs) o compras
  (shorts) en cascada y suele girarse;
- muros del libro de órdenes (peso ``zone_book_weight``).

Para una entrada LONG se busca la zona por *debajo* del precio (liquidaciones de
longs + bids); para una SHORT, por *encima* (liquidaciones de shorts + asks). La
entrada se coloca en el centro de la zona (``Zone.mid``).

Los tramos son una rejilla anclada en el precio actual, de ancho
``zone_width_pct`` % del mid: el tramo ``j`` cubre distancias
``[j·ancho, (j+1)·ancho)`` desde el mid, sólo del lado que toca. Así ninguna
zona contiene el precio actual (una compra límite en la zona nunca queda por
encima del mercado).

Para el backtest no existen libro ni liquidaciones históricas: ``historical_zones``
aproxima las zonas del pasado con mínimos/máximos de pivote confirmados (ver su
docstring y sus límites). Las estrategias reciben las zonas como un DataFrame por
vela (``ZONE_COLUMNS``); en vivo sólo la última fila tiene datos (``zones_frame``).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .config import Settings
from .exchange.base import Book, LiqLevels, MarketData, MarketDataError
from .models import Side, Zone

log = logging.getLogger(__name__)

Point = tuple[float, float]  # (precio, usd)


def _num(v: Any) -> Optional[float]:
    """Convierte a float finito; None si no se puede (strings de la API, None, NaN...)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def book_levels(book: Optional[Book]) -> list[Point]:
    """Niveles del libro como ``(px, usd)``, bids y asks juntos.

    Acepta niveles ``(px, sz, ...)`` o ``{"px", "sz"}`` (también en strings, como
    los devuelve Hyperliquid). Ignora niveles inválidos o con precio/tamaño <= 0.
    No hace falta separar bids de asks: ``find_zone`` filtra por lado del precio.
    """
    if not book:
        return []
    out: list[Point] = []
    for key in ("bids", "asks"):
        for lvl in book.get(key) or ():
            if isinstance(lvl, dict):
                px, sz = _num(lvl.get("px")), _num(lvl.get("sz"))
            elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                px, sz = _num(lvl[0]), _num(lvl[1])
            else:
                continue
            if px is None or sz is None or px <= 0 or sz <= 0:
                continue
            out.append((px, px * sz))
    return out


def liq_levels(levels: Optional[LiqLevels], side: Optional[Side] = None) -> list[Point]:
    """Niveles de liquidación como ``(px, usd)``.

    Formato esperado (``MarketData.liquidation_levels``):
    ``{"liq_px": float, "usd": float, "side": "long"|"short"}``. Por tolerancia
    también acepta ``liq_price``/``value`` (formato crudo de Moon Dev) y tuplas
    ``(px, usd)``.

    Si se pasa ``side`` sólo se quedan las liquidaciones de posiciones de ese lado
    (las de longs alimentan zonas LONG, las de shorts zonas SHORT); los niveles sin
    lado conocido se conservan.
    """
    out: list[Point] = []
    for lvl in levels or ():
        if isinstance(lvl, dict):
            px = _num(lvl.get("liq_px", lvl.get("liq_price")))
            usd = _num(lvl.get("usd", lvl.get("value")))
            lvl_side = str(lvl.get("side") or "").strip().lower()
            if side is not None and lvl_side in ("long", "short") and lvl_side != side.value:
                continue
        elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
            px, usd = _num(lvl[0]), _num(lvl[1])
        else:
            continue
        if px is None or usd is None or px <= 0 or usd <= 0:
            continue
        out.append((px, abs(usd)))
    return out


def find_zone(coin: str, side: Side, mid: float, book_pts: Iterable[Point], liq_pts: Iterable[Point],
              settings: Settings) -> Optional[Zone]:
    """Tramo de mayor puntuación del lado correcto del precio, o None.

    - LONG: sólo puntos por debajo de ``mid``; SHORT: sólo por encima.
    - Se descartan los puntos (y los tramos, medidos en su centro, que es donde se
      entraría) a menos de ``zone_min_distance_pct`` o a más de
      ``zone_max_distance_pct`` del mid.
    - Puntuación = ``zone_book_weight``·usd_libro + ``zone_liq_weight``·usd_liq.
      Una fuente con peso <= 0 se ignora por completo.
    - Empate (misma puntuación): gana el tramo más cercano al precio (más
      probable que se ejecute).
    - ``Zone.usd`` = dólares sin ponderar; ``source`` = "book" | "liq" | "mixed".
    """
    mid_f = _num(mid)
    if mid_f is None or mid_f <= 0:
        return None
    width = mid_f * settings.zone_width_pct / 100.0
    if width <= 0:
        return None
    min_d, max_d = settings.zone_min_distance_pct, settings.zone_max_distance_pct
    below = side is Side.LONG

    # tramo j -> [usd del libro, usd de liquidaciones]
    buckets: dict[int, list[float]] = {}
    sources = ((0, book_pts, settings.zone_book_weight), (1, liq_pts, settings.zone_liq_weight))
    for slot, pts, weight in sources:
        if weight <= 0:
            continue
        for px, usd in pts or ():
            if not (math.isfinite(px) and math.isfinite(usd)) or usd <= 0:
                continue
            if px >= mid_f if below else px <= mid_f:   # lado equivocado del precio
                continue
            dist = abs(px - mid_f)
            if not (min_d <= dist / mid_f * 100.0 <= max_d):
                continue
            j = int(math.floor(dist / width))
            buckets.setdefault(j, [0.0, 0.0])[slot] += usd

    best: Optional[tuple[float, int]] = None   # (puntuación, j)
    for j in sorted(buckets):                  # de más cercano a más lejano: el empate lo gana el primero
        book_usd, liq_usd = buckets[j]
        score = settings.zone_book_weight * book_usd + settings.zone_liq_weight * liq_usd
        if score <= 0 or not (min_d <= (j + 0.5) * settings.zone_width_pct <= max_d):
            continue
        if best is None or (score > best[0] and not math.isclose(score, best[0], rel_tol=1e-9)):
            best = (score, j)
    if best is None:
        return None

    j = best[1]
    book_usd, liq_usd = buckets[j]
    if below:
        low, high = mid_f - (j + 1) * width, mid_f - j * width
    else:
        low, high = mid_f + j * width, mid_f + (j + 1) * width
    source = "mixed" if book_usd > 0 and liq_usd > 0 else ("liq" if liq_usd > 0 else "book")
    center = (low + high) / 2.0
    return Zone(coin=coin, side=side, low=low, high=high, usd=book_usd + liq_usd, source=source,
                distance_pct=abs(center - mid_f) / mid_f * 100.0)


class LiquidityScanner:
    """Envoltorio sobre ``MarketData``: libro agregado + liquidaciones -> ``Zone``.

    Usa ``l2_book(coin, n_sig_figs=3)`` (niveles agrupados para ver más
    profundidad que el top del libro) y ``liquidation_levels(coin)``.

    Nunca lanza ``MarketDataError``: si fallan el precio o el libro devuelve None;
    si sólo falla la fuente de liquidaciones (opcional, de terceros) sigue con el
    libro y lo registra en el log.
    """

    def __init__(self, market: MarketData, settings: Settings, n_sig_figs: int = 3):
        self.market = market
        self.settings = settings
        self.n_sig_figs = n_sig_figs

    def zone(self, coin: str, side: Side) -> Optional[Zone]:
        snap = self._snapshot(coin)
        if snap is None:
            return None
        mid, book_pts, liqs = snap
        return find_zone(coin, side, mid, book_pts, liq_levels(liqs, side), self.settings)

    def zones(self, coin: str) -> dict[Side, Optional[Zone]]:
        """Zona LONG y SHORT con una sola lectura del mercado (lo que piden las estrategias)."""
        snap = self._snapshot(coin)
        if snap is None:
            return {Side.LONG: None, Side.SHORT: None}
        mid, book_pts, liqs = snap
        return {s: find_zone(coin, s, mid, book_pts, liq_levels(liqs, s), self.settings)
                for s in (Side.LONG, Side.SHORT)}

    def _snapshot(self, coin: str) -> Optional[tuple[float, list[Point], LiqLevels]]:
        try:
            mid = self.market.mid(coin)
            book = self.market.l2_book(coin, n_sig_figs=self.n_sig_figs)
        except MarketDataError as e:
            log.warning("liquidez %s: sin libro/precio (%s)", coin, e)
            return None
        try:
            liqs = self.market.liquidation_levels(coin) or []
        except MarketDataError as e:
            log.warning("liquidez %s: sin niveles de liquidación, sólo libro (%s)", coin, e)
            liqs = []
        return mid, book_levels(book), liqs


# --- zonas por vela (estrategias y backtest) ------------------------------------

# Columnas del DataFrame de zonas por vela (NaN = sin zona). ``long_source``/``short_source``
# son opcionales ("book" | "liq" | "mixed"; sólo las rellena ``zones_frame`` en vivo).
ZONE_COLUMNS = ["long_low", "long_high", "long_usd", "short_low", "short_high", "short_usd"]
SOURCE_COLUMNS = ["long_source", "short_source"]
HIST_SOURCE = "pivots"      # fuente de las zonas sin columna de fuente (las de ``historical_zones``)

PIVOT_BARS = 3              # un pivote necesita PIVOT_BARS velas a cada lado (se confirma PIVOT_BARS velas después)
PIVOT_MAX_AGE_BARS = 300    # un pivote más antiguo que esto deja de contar como liquidez


def empty_zones(index: pd.Index) -> pd.DataFrame:
    """DataFrame de zonas sin ninguna zona (todo NaN) con el índice dado."""
    return pd.DataFrame(np.nan, index=index, columns=ZONE_COLUMNS, dtype=float)


def zones_frame(index: pd.Index, zones: Optional[Mapping[Side, Optional[Zone]]],
                coin: Optional[str] = None) -> pd.DataFrame:
    """Zonas en vivo (``LiquidityScanner.zones``) como DataFrame por vela: sólo la última fila tiene datos.

    Es lo que ve la estrategia en vivo: el libro y las liquidaciones de *ahora*, asignados a la última
    vela cerrada. Se ignoran las zonas de otra moneda (si se pasa ``coin``) y las guardadas bajo el lado
    equivocado del diccionario.
    """
    out = empty_zones(index)
    out["long_source"] = ""
    out["short_source"] = ""
    if len(index) == 0 or not zones:
        return out
    last = len(index) - 1
    for side in (Side.LONG, Side.SHORT):
        zone = zones.get(side)
        if zone is None or zone.side is not side or (coin is not None and zone.coin != coin):
            continue
        p = side.value
        for col, val in ((f"{p}_low", zone.low), (f"{p}_high", zone.high), (f"{p}_usd", zone.usd)):
            out.iloc[last, out.columns.get_loc(col)] = float(val)
        out.iloc[last, out.columns.get_loc(f"{p}_source")] = str(zone.source)
    return out


def row_zones(zones: Optional[pd.DataFrame], pos: int, coin: str) -> dict[Side, Optional[Zone]]:
    """Zonas de la fila ``pos`` de un DataFrame de zonas como objetos ``Zone`` (lo inverso de ``zones_frame``).

    ``source`` es la columna ``<lado>_source`` si existe, o ``HIST_SOURCE`` (zonas históricas).
    ``distance_pct`` no se puede saber sin el precio: queda en 0.
    """
    out: dict[Side, Optional[Zone]] = {Side.LONG: None, Side.SHORT: None}
    if zones is None or len(zones) == 0:
        return out
    row = zones.iloc[pos]
    for side in (Side.LONG, Side.SHORT):
        p = side.value
        low, high, usd = (_num(row.get(f"{p}_{k}")) for k in ("low", "high", "usd"))
        if low is None or high is None or usd is None:
            continue
        source = row.get(f"{p}_source")
        source = source if isinstance(source, str) and source else HIST_SOURCE
        out[side] = Zone(coin=coin, side=side, low=low, high=high, usd=usd, source=source)
    return out


def historical_zones(bars: pd.DataFrame, settings: Settings, *, pivot_bars: int = PIVOT_BARS,
                     max_age: int = PIVOT_MAX_AGE_BARS) -> pd.DataFrame:
    """Zonas de liquidez *aproximadas* para cada vela del pasado (backtest), sin mirar el futuro.

    No hay histórico del libro ni de las liquidaciones, así que se usa un sustituto clásico: la
    liquidez (stops y liquidaciones) se acumula bajo los mínimos y sobre los máximos de giro.

    - Pivote mínimo en ``p``: ``low[p]`` menor que los ``pivot_bars`` mínimos anteriores y no mayor
      que los ``pivot_bars`` siguientes (máximo: simétrico). Sólo se **confirma** en la vela
      ``p + pivot_bars`` y sólo se usa desde ahí: la fila ``i`` nunca depende de velas posteriores a ``i``.
    - Un pivote deja de contar cuando una vela *cierra* más allá de él (nivel roto: la liquidez ya
      se consumió) o cuando tiene más de ``max_age`` velas. Una mecha que lo perfora sin cerrar más
      allá no lo borra: es justo el barrido que buscan las estrategias.
    - Peso de cada pivote = volumen de su vela · precio (USD aproximados; sin volumen válido pesa 1).
    - Igual que ``find_zone``: rejilla de tramos de ``zone_width_pct`` anclada en el cierre de la vela,
      mínimos por debajo -> zona LONG, máximos por encima -> zona SHORT, filtros de distancia
      ``zone_min_distance_pct``..``zone_max_distance_pct`` (al punto y al centro del tramo), gana el
      tramo de más USD y, a igualdad, el más cercano.

    Limitación (importante al leer un backtest): esto NO es lo que ve la estrategia en vivo (libro
    real + liquidaciones de terceros). Es una aproximación razonable y sin look-ahead, pero el
    resultado de las estrategias de liquidez en backtest es sólo orientativo.

    Devuelve un DataFrame con el índice de ``bars`` y ``ZONE_COLUMNS`` (NaN = sin zona).
    """
    out = empty_zones(bars.index)
    n = len(bars)
    k = max(int(pivot_bars), 1)
    width_pct = float(settings.zone_width_pct)
    min_d, max_d = float(settings.zone_min_distance_pct), float(settings.zone_max_distance_pct)
    if n < 2 * k + 2 or width_pct <= 0 or max_d <= 0 or max_d < min_d:
        return out
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    vol = bars["volume"].to_numpy(dtype=float) if "volume" in bars.columns else np.full(n, np.nan)

    for side, series in ((Side.LONG, low), (Side.SHORT, high)):
        piv, px = _pivots(series, k, lows=side is Side.LONG)
        if len(piv) == 0:
            continue
        usd = vol[piv] * px
        usd = np.where(np.isfinite(usd) & (usd > 0), usd, 1.0)
        start = piv + k                                              # vela de confirmación
        end = np.minimum(np.minimum(piv + max_age + 1, n),
                         _first_break(close, start, px, max_age, below=side is Side.LONG))
        cols = [out.columns.get_loc(f"{side.value}_{c}") for c in ("low", "high", "usd")]
        _fill_zone_rows(out, cols, close, start, end, px, usd, width_pct, min_d, max_d,
                        below=side is Side.LONG)
    return out


def _pivots(values: np.ndarray, k: int, *, lows: bool) -> tuple[np.ndarray, np.ndarray]:
    """Índices y precios de los pivotes (mínimos si ``lows``) con ``k`` velas a cada lado."""
    win = np.lib.stride_tricks.sliding_window_view(values, 2 * k + 1)
    center = win[:, k]
    with np.errstate(invalid="ignore"):
        if lows:
            ok = (center < win[:, :k].min(axis=1)) & (center <= win[:, k + 1:].min(axis=1))
        else:
            ok = (center > win[:, :k].max(axis=1)) & (center >= win[:, k + 1:].max(axis=1))
    pos = np.flatnonzero(ok & np.isfinite(center) & (center > 0))
    return pos + k, center[pos]


def _first_break(close: np.ndarray, start: np.ndarray, px: np.ndarray, max_age: int, *,
                 below: bool) -> np.ndarray:
    """Primera vela >= ``start`` que cierra más allá del pivote (n si ninguna en ``max_age`` velas)."""
    n = len(close)
    out = np.full(len(start), n, dtype=np.int64)
    for i, (s, p) in enumerate(zip(start.tolist(), px.tolist())):
        seg = close[s:min(s + max_age + 1, n)]
        hits = np.flatnonzero(seg < p if below else seg > p)
        if len(hits):
            out[i] = s + int(hits[0])
    return out


def _fill_zone_rows(out: pd.DataFrame, cols: list[int], close: np.ndarray, start: np.ndarray, end: np.ndarray,
                    px: np.ndarray, usd: np.ndarray, width_pct: float, min_d: float, max_d: float, *,
                    below: bool) -> None:
    """Para cada vela, tramo de más USD entre los pivotes activos (vectorizado sobre pares vela-pivote)."""
    lengths = np.clip(end - start, 0, None)
    total = int(lengths.sum())
    if total == 0:
        return
    offsets = np.repeat(np.cumsum(lengths) - lengths, lengths)
    rows = np.repeat(start, lengths) + (np.arange(total) - offsets)
    p = np.repeat(px, lengths)
    w = np.repeat(usd, lengths)
    mid = close[rows]
    dist = (mid - p) if below else (p - mid)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist_pct = dist / mid * 100.0
        width = mid * width_pct / 100.0
        j = np.floor(dist / width)
        ok = (dist > 0) & (dist_pct >= min_d) & (dist_pct <= max_d) & np.isfinite(j)
        ok &= ((j + 0.5) * width_pct >= min_d) & ((j + 0.5) * width_pct <= max_d)
    if not ok.any():
        return
    rows, j, w = rows[ok], j[ok].astype(np.int64), w[ok]
    n_buckets = int(j.max()) + 1
    keys, inv = np.unique(rows * n_buckets + j, return_inverse=True)   # ordenadas: por vela y, dentro, por tramo
    score = np.bincount(inv, weights=w)
    key_row, key_j = keys // n_buckets, keys % n_buckets
    first = np.flatnonzero(np.r_[True, key_row[1:] != key_row[:-1]])    # inicio de cada vela
    best = np.maximum.reduceat(score, first)
    sizes = np.diff(np.r_[first, len(keys)])
    is_best = score == np.repeat(best, sizes)
    # El primer tramo con la puntuación máxima de cada vela es el más cercano al precio.
    _, pick = np.unique(key_row[is_best], return_index=True)
    pick = np.flatnonzero(is_best)[pick]
    r, jj = key_row[pick], key_j[pick].astype(float)
    wd = close[r] * width_pct / 100.0
    if below:
        lo, hi = close[r] - (jj + 1) * wd, close[r] - jj * wd
    else:
        lo, hi = close[r] + jj * wd, close[r] + (jj + 1) * wd
    values = out.to_numpy(copy=True)
    values[r, cols[0]], values[r, cols[1]], values[r, cols[2]] = lo, hi, score[pick]
    out.iloc[:, :] = values

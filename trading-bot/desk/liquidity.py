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
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Optional

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

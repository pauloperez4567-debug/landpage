"""Zonas de liquidez: funciones puras + ``LiquidityScanner`` contra ``FakeMarketData``.

Con los valores por defecto (mid 100): tramos de 0.11, distancia permitida
[0.05 %, 3 %], peso libro 1, peso liquidaciones 2.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from desk.exchange.base import MarketDataError
from desk.liquidity import (HIST_SOURCE, ZONE_COLUMNS, LiquidityScanner, book_levels, find_zone, historical_zones,
                            liq_levels, row_zones, zones_frame)
from desk.models import Side, Zone

MID = 100.0


def zone_of(settings, side, book=(), liq=(), mid=MID, coin="HYPE"):
    return find_zone(coin, side, mid, list(book), list(liq), settings)


# --- conversión de datos crudos --------------------------------------------------

def test_book_levels_converts_to_usd_and_skips_garbage():
    book = {
        "bids": [(99.0, 10.0), ("98.5", "2"), {"px": "98", "sz": "1.5", "n": 3},
                 (0.0, 5.0), (97.0, -1.0), ("x", 1.0), (96.0,)],
        "asks": [(101.0, 3.0), (float("nan"), 1.0)],
        "ts": 0.0,
    }
    assert book_levels(book) == pytest.approx([(99.0, 990.0), (98.5, 197.0), (98.0, 147.0), (101.0, 303.0)])
    assert book_levels(None) == [] and book_levels({}) == [] and book_levels({"bids": None}) == []


def test_liq_levels_accepts_formats_and_filters_by_position_side():
    raw = [
        {"liq_px": 95.0, "usd": 1_000_000.0, "side": "long"},
        {"liq_price": "105", "value": "250000", "side": "short"},   # formato crudo de Moon Dev
        {"liq_px": 90.0, "usd": 5_000.0},                           # sin lado: se conserva siempre
        (93.0, 7_000.0),
        {"liq_px": None, "usd": 1.0}, {"liq_px": 94.0, "usd": 0.0}, "basura",
    ]
    assert liq_levels(raw) == [(95.0, 1e6), (105.0, 250_000.0), (90.0, 5_000.0), (93.0, 7_000.0)]
    assert liq_levels(raw, Side.LONG) == [(95.0, 1e6), (90.0, 5_000.0), (93.0, 7_000.0)]
    assert liq_levels(raw, Side.SHORT) == [(105.0, 250_000.0), (90.0, 5_000.0), (93.0, 7_000.0)]
    assert liq_levels(None) == []


# --- lado correcto y geometría ---------------------------------------------------

def test_long_zone_is_an_aligned_bucket_below_price(settings):
    z = zone_of(settings, Side.LONG, book=[(99.5, 99_500.0), (100.5, 500_000.0)])
    assert z is not None and z.side is Side.LONG and z.coin == "HYPE"
    # 99.5 está a 0.5 % -> tramo j=4 de ancho 0.11: [100 - 5·0.11, 100 - 4·0.11]
    assert z.low == pytest.approx(99.45) and z.high == pytest.approx(99.56)
    assert z.high - z.low == pytest.approx(MID * settings.zone_width_pct / 100.0)
    assert z.contains(99.5) and z.high < MID
    assert z.mid == pytest.approx(99.505)
    assert z.distance_pct == pytest.approx(0.495)
    assert z.usd == pytest.approx(99_500.0) and z.source == "book"


def test_short_zone_is_above_price(settings):
    z = zone_of(settings, Side.SHORT, book=[(99.5, 900_000.0), (100.5, 100_000.0)])
    assert z is not None and z.side is Side.SHORT
    assert z.low == pytest.approx(100.44) and z.high == pytest.approx(100.55)
    assert z.contains(100.5) and z.low > MID
    assert z.usd == pytest.approx(100_000.0)


def test_liquidity_on_the_wrong_side_is_ignored(settings):
    assert zone_of(settings, Side.SHORT, book=[(99.0, 1e6)], liq=[(98.0, 1e6)]) is None
    assert zone_of(settings, Side.LONG, book=[(101.0, 1e6)], liq=[(102.0, 1e6)]) is None
    assert zone_of(settings, Side.LONG, book=[(MID, 1e9)]) is None   # justo en el precio: no es zona


# --- filtros de distancia --------------------------------------------------------

def test_too_close_and_too_far_are_discarded(settings):
    assert zone_of(settings, Side.LONG, book=[(99.97, 1e6)]) is None       # 0.03 % < 0.05 %
    assert zone_of(settings, Side.LONG, liq=[(96.5, 1e6)]) is None         # 3.5 % > 3 %
    assert zone_of(settings, Side.SHORT, liq=[(103.5, 1e6)]) is None
    z = zone_of(settings, Side.LONG, liq=[(97.1, 1e6)])                    # 2.9 %: dentro
    assert z is not None and z.contains(97.1)


def test_zone_center_must_also_respect_max_distance(settings):
    # 2.99 % cae en el tramo [2.97 %, 3.08 %): su centro (la entrada) estaría a 3.025 % > 3 %.
    assert zone_of(settings, Side.LONG, liq=[(97.01, 1e6)]) is None


def test_min_distance_skips_near_walls(settings):
    settings.zone_min_distance_pct = 1.0
    z = zone_of(settings, Side.LONG, book=[(99.5, 1e6), (98.5, 1e5)])
    assert z is not None and z.contains(98.5)


# --- pesos, agregación y fuentes -------------------------------------------------

def test_liquidations_outweigh_a_bigger_book_wall_by_weight(settings):
    book, liq = [(99.0, 990_000.0)], [(98.0, 600_000.0)]
    z = zone_of(settings, Side.LONG, book=book, liq=liq)       # 2·600k = 1.2M > 990k
    assert z.source == "liq" and z.contains(98.0) and z.usd == pytest.approx(600_000.0)

    settings.zone_liq_weight = 1.0                              # 600k < 990k
    z = zone_of(settings, Side.LONG, book=book, liq=liq)
    assert z.source == "book" and z.contains(99.0)


def test_zero_weight_ignores_that_source(settings):
    book, liq = [(99.0, 5_000_000.0)], [(98.0, 1_000.0)]
    settings.zone_book_weight = 0.0
    z = zone_of(settings, Side.LONG, book=book, liq=liq)
    assert z.source == "liq" and z.contains(98.0)
    settings.zone_liq_weight = 0.0
    assert zone_of(settings, Side.LONG, book=book, liq=liq) is None


def test_levels_in_the_same_bucket_add_up(settings):
    small = [(99.50, 40_000.0), (99.52, 40_000.0), (99.54, 40_000.0)]   # mismo tramo j=4
    z = zone_of(settings, Side.LONG, book=small + [(98.0, 100_000.0)])
    assert z.contains(99.52) and z.usd == pytest.approx(120_000.0)


@pytest.mark.parametrize("side,book,liq,expected_px,expected_source", [
    (Side.LONG, [(99.5, 1e6)], [(99.52, 2e5)], 99.5, "mixed"),
    (Side.LONG, [(99.5, 1e6)], [], 99.5, "book"),
    (Side.LONG, [], [(99.52, 2e5)], 99.52, "liq"),
    (Side.SHORT, [(100.5, 1e6)], [(100.52, 2e5)], 100.5, "mixed"),
])
def test_source_reflects_where_the_liquidity_comes_from(settings, side, book, liq, expected_px,
                                                        expected_source):
    z = zone_of(settings, side, book=book, liq=liq)
    assert z.source == expected_source and z.contains(expected_px)
    assert z.usd == pytest.approx(sum(u for _, u in book) + sum(u for _, u in liq))


# --- empates ---------------------------------------------------------------------

@pytest.mark.parametrize("side,near,far", [(Side.LONG, 99.5, 98.0), (Side.SHORT, 100.5, 102.0)])
def test_tie_goes_to_the_zone_closest_to_price(settings, side, near, far):
    for liq in ([(near, 1e5), (far, 1e5)], [(far, 1e5), (near, 1e5)]):   # el orden no importa
        z = zone_of(settings, side, liq=liq)
        assert z.contains(near)


def test_weighted_tie_across_sources_goes_to_closest(settings):
    # libro 200k (peso 1) empata con liquidaciones 100k (peso 2)
    z = zone_of(settings, Side.LONG, book=[(98.0, 200_000.0)], liq=[(99.0, 100_000.0)])
    assert z.source == "liq" and z.contains(99.0)
    z = zone_of(settings, Side.LONG, book=[(99.0, 200_000.0)], liq=[(98.0, 100_000.0)])
    assert z.source == "book" and z.contains(99.0)


# --- sin datos -------------------------------------------------------------------

def test_no_data_returns_none(settings):
    assert zone_of(settings, Side.LONG) is None
    assert zone_of(settings, Side.SHORT, book=[], liq=[]) is None
    assert zone_of(settings, Side.LONG, book=[(99.0, 1e6)], mid=0.0) is None
    assert zone_of(settings, Side.LONG, book=[(99.0, 1e6)], mid=float("nan")) is None
    assert zone_of(settings, Side.LONG, book=[(99.0, 0.0), (float("nan"), 1e6), (98.0, math.inf)]) is None


# --- LiquidityScanner -------------------------------------------------------------

def _setup_market(market):
    market.set_price("HYPE", 100.0)
    market.set_book("HYPE", bids=[(99.98, 50_000.0), (99.5, 1_000.0)], asks=[(100.02, 50_000.0), (100.5, 1_000.0)])
    market.set_liqs("HYPE", [
        {"liq_px": 98.0, "usd": 2_000_000.0, "side": "long"},
        {"liq_px": 102.0, "usd": 50_000.0, "side": "short"},
        {"liq_px": 101.0, "usd": 9_000_000.0, "side": "long"},   # long "liquidable" por encima: dato viejo
    ])


def _spy_book(market, monkeypatch):
    calls = []
    original = market.l2_book

    def spy(coin, n_sig_figs=None):
        calls.append((coin, n_sig_figs))
        return original(coin, n_sig_figs)

    monkeypatch.setattr(market, "l2_book", spy)
    return calls


def test_scanner_combines_aggregated_book_and_liquidations(market, settings, monkeypatch):
    _setup_market(market)
    calls = _spy_book(market, monkeypatch)
    scanner = LiquidityScanner(market, settings)

    long_z = scanner.zone("HYPE", Side.LONG)
    assert calls == [("HYPE", 3)]
    assert long_z.side is Side.LONG and long_z.source == "liq" and long_z.contains(98.0)

    # SHORT: muro de asks 100.5k (peso 1) > liquidaciones de shorts 50k·2 = 100k; la de longs a 101 no cuenta
    short_z = scanner.zone("HYPE", Side.SHORT)
    assert short_z.source == "book" and short_z.contains(100.5)


def test_scanner_zones_reads_market_once_for_both_sides(market, settings, monkeypatch):
    _setup_market(market)
    calls = _spy_book(market, monkeypatch)
    zones = LiquidityScanner(market, settings).zones("HYPE")
    assert len(calls) == 1
    assert zones[Side.LONG].contains(98.0) and zones[Side.SHORT].contains(100.5)


def test_scanner_without_depth_returns_none(market, settings):
    # el libro por defecto de FakeMarketData sólo tiene el top (a 0.5 bps del mid): demasiado cerca
    scanner = LiquidityScanner(market, settings)
    assert scanner.zone("SOL", Side.LONG) is None
    assert scanner.zones("SOL") == {Side.LONG: None, Side.SHORT: None}


def test_scanner_market_data_error_returns_none(market, settings, monkeypatch):
    _setup_market(market)
    scanner = LiquidityScanner(market, settings)
    market.fail_next = 1                       # falla el precio
    assert scanner.zone("HYPE", Side.LONG) is None
    market.fail_next = 1
    assert scanner.zones("HYPE") == {Side.LONG: None, Side.SHORT: None}
    assert scanner.zone("HYPE", Side.LONG) is not None   # se recupera en la siguiente llamada

    def broken_book(coin, n_sig_figs=None):
        raise MarketDataError("libro caído")

    monkeypatch.setattr(market, "l2_book", broken_book)
    assert scanner.zone("HYPE", Side.LONG) is None


def test_scanner_degrades_to_book_when_liquidation_feed_fails(market, settings, monkeypatch):
    _setup_market(market)

    def broken_liqs(coin):
        raise MarketDataError("api de liquidaciones caída")

    monkeypatch.setattr(market, "liquidation_levels", broken_liqs)
    z = LiquidityScanner(market, settings).zone("HYPE", Side.LONG)
    assert z is not None and z.source == "book" and z.contains(99.5)


# --- zonas históricas (backtest) ----------------------------------------------------

def ohlc(rows, start="2026-01-01", freq="15min"):
    """Velas a mano: filas (open, high, low, close[, volume])."""
    rows = [tuple(r) + (1_000.0,) * (5 - len(r)) for r in rows]
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx, dtype=float)


def v_bottom(after):
    """Bajada de 102.5 a 100 (mínimo de pivote 99.9 en la vela 5), subida y luego ``after`` (cierres)."""
    closes = [102.5, 102.0, 101.5, 101.0, 100.5, 100.0, 100.5, 101.0, 101.5] + list(after)
    rows, prev = [], closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + 0.1, min(prev, c) - 0.1, c))
        prev = c
    return rows


def reference_zone(bars, i, side, settings, k=3, max_age=300):
    """Implementación directa (lenta) de la definición, sobre ``find_zone``: pivotes confirmados en ``i``."""
    lo, hi, cl, vol = (bars[c].to_numpy() for c in ("low", "high", "close", "volume"))
    pts = []
    for p in range(k, i - k + 1):
        if i - p > max_age:
            continue
        if side is Side.LONG:
            ok = lo[p] < lo[p - k:p].min() and lo[p] <= lo[p + 1:p + k + 1].min()
            px, broken = lo[p], (cl[p + k:i + 1] < lo[p]).any()
        else:
            ok = hi[p] > hi[p - k:p].max() and hi[p] >= hi[p + 1:p + k + 1].max()
            px, broken = hi[p], (cl[p + k:i + 1] > hi[p]).any()
        if ok and not broken:
            pts.append((px, vol[p] * px))
    return find_zone("X", side, cl[i], [], pts, settings)


def test_historical_zones_match_the_direct_definition(settings):
    bars = make_candles(n=900, seed=11, vol_pct=0.4)
    bars["volume"] = np.random.default_rng(3).uniform(100, 5_000, len(bars))     # el volumen pesa
    z = historical_zones(bars, settings)
    assert list(z.columns) == ZONE_COLUMNS and z.index.equals(bars.index)
    checked = 0
    for i in range(10, len(bars), 7):
        for side in (Side.LONG, Side.SHORT):
            ref = reference_zone(bars, i, side, settings)
            p = side.value
            got = z.iloc[i][[f"{p}_low", f"{p}_high", f"{p}_usd"]].to_numpy(dtype=float)
            if ref is None:
                assert np.isnan(got).all(), (i, side)
            else:
                checked += 1
                assert got == pytest.approx([ref.low, ref.high, ref.usd], rel=1e-12), (i, side)
                assert ref.high <= bars["close"].iloc[i] if side is Side.LONG else ref.low >= bars["close"].iloc[i]
    assert checked > 150


def test_pivot_counts_only_once_confirmed(settings):
    bars = ohlc(v_bottom([102.0, 102.0]))
    z = historical_zones(bars, settings)
    assert z["long_low"].iloc[:8].isna().all()                  # el mínimo de la vela 5 aún no está confirmado
    assert z["long_low"].iloc[8] <= 99.9 <= z["long_high"].iloc[8]
    assert z["long_usd"].iloc[8] == pytest.approx(1_000.0 * 99.9)


def test_wick_through_keeps_the_pivot_close_through_removes_it(settings):
    rows = v_bottom([101.0])
    rows.append((101.0, 101.1, 99.5, 100.6))                  # 10: mecha bajo 99.9 pero cierra encima
    rows.append((100.6, 101.2, 100.5, 101.0))                 # 11
    rows.append((101.0, 101.0, 99.0, 99.6))                   # 12: cierra bajo 99.9 -> nivel roto
    rows.append((99.6, 100.4, 99.5, 100.3))                   # 13
    z = historical_zones(ohlc(rows), settings)
    has = lambda i: z["long_low"].iloc[i] <= 99.9 <= z["long_high"].iloc[i]    # noqa: E731
    assert has(10) and has(11)
    assert not has(12) and not has(13)


def test_old_pivots_expire(settings):
    bars = ohlc(v_bottom([102.0] * 10))
    assert historical_zones(bars, settings)["long_low"].iloc[8:].notna().all()
    z = historical_zones(bars, settings, max_age=6)
    assert z["long_low"].iloc[8:12].notna().all() and z["long_low"].iloc[12:].isna().all()   # 5 + 6 = 11


def test_historical_zones_short_side_mirror(settings):
    rows = [(2 * 101.0 - o, 2 * 101.0 - lo, 2 * 101.0 - hi, 2 * 101.0 - c) for o, hi, lo, c in v_bottom([100.0])]
    z = historical_zones(ohlc(rows), settings)
    assert z["short_low"].iloc[:8].isna().all()
    assert z["short_low"].iloc[8] <= 102.1 <= z["short_high"].iloc[8]
    assert z["short_low"].iloc[8] > ohlc(rows)["close"].iloc[8]


def test_historical_zones_never_look_ahead(settings):
    bars = make_candles(n=1500, seed=4, vol_pct=0.5)
    full = historical_zones(bars, settings)
    for k in (300, 777, 1200):
        changed = bars.copy()
        changed.iloc[k + 1:, :4] = changed.iloc[k + 1:, :4].to_numpy() * 0.93
        pd.testing.assert_frame_equal(full.iloc[:k + 1], historical_zones(changed, settings).iloc[:k + 1])
        pd.testing.assert_frame_equal(full.iloc[:k + 1], historical_zones(bars.iloc[:k + 1], settings))


def test_historical_zones_degenerate_inputs(settings):
    assert historical_zones(make_candles(n=5), settings).isna().all().all()
    assert len(historical_zones(make_candles(n=200).iloc[:0], settings)) == 0
    settings.zone_width_pct = 0.0
    assert historical_zones(make_candles(n=200), settings).isna().all().all()


def test_zones_frame_puts_live_zones_on_the_last_row_only():
    idx = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    long_z = Zone("HYPE", Side.LONG, 97.0, 97.1, 5e5, "liq")
    short_z = Zone("HYPE", Side.SHORT, 103.0, 103.1, 2e5, "book")
    zf = zones_frame(idx, {Side.LONG: long_z, Side.SHORT: short_z}, "HYPE")
    assert zf[ZONE_COLUMNS].iloc[:3].isna().all().all()
    assert zf.iloc[-1][["long_low", "long_high", "long_usd"]].tolist() == [97.0, 97.1, 5e5]
    assert zf.iloc[-1]["short_source"] == "book"
    back = row_zones(zf, -1, "HYPE")
    assert back[Side.LONG] == long_z and back[Side.SHORT] == short_z
    assert row_zones(zf, 0, "HYPE") == {Side.LONG: None, Side.SHORT: None}
    # otra moneda o lado equivocado del diccionario: se ignoran
    other = zones_frame(idx, {Side.LONG: Zone("BTC", Side.LONG, 1, 2, 3, "liq"), Side.SHORT: long_z}, "HYPE")
    assert other[ZONE_COLUMNS].isna().all().all()
    # sin columna de fuente (zonas históricas) la fuente es HIST_SOURCE
    assert row_zones(zf[ZONE_COLUMNS], -1, "HYPE")[Side.LONG].source == HIST_SOURCE

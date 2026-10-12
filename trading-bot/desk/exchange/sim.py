"""Mercados offline: simulación determinista y reproducción de velas desde un CSV.

``SimMarketData`` (``DESK_DATA=sim``)
    Paseo aleatorio geométrico por moneda. El precio en un instante ``t`` es una función
    pura de ``(semilla, moneda, t)``: la misma semilla da siempre el mismo mercado, da igual
    en qué orden o cuántas veces se consulte. El tiempo se divide en pasos de
    ``step_seconds`` agrupados en bloques de una hora; cada bloque genera sus rendimientos
    con un RNG sembrado por ``(semilla, moneda, bloque)``, con regímenes de volatilidad
    (para que haya compresiones y rupturas) y algún salto ocasional. Entre dos pasos el
    precio se interpola, así que avanza de forma continua con el ``Clock``.

``ReplayMarketData`` (``DESK_DATA=replay:<csv>``)
    Reproduce velas reales ``timestamp,open,high,low,close,volume`` desplazadas al tiempo
    del ``Clock``: el precio actual se interpola dentro de la vela en curso
    (apertura → mínimo → máximo → cierre en velas alcistas; apertura → máximo → mínimo →
    cierre en bajistas).

Ambos generan un libro sintético (con "muros" de liquidez que duran un rato) y niveles de
liquidación sintéticos, para que las zonas de liquidez funcionen sin red.
"""

from __future__ import annotations

import logging
import math
import re
import zlib
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from ..clock import Clock
from ..models import AssetMeta
from .base import CANDLE_COLUMNS, Book, LiqLevels, MarketData, MarketDataError
from .hyperliquid_data import empty_candles, utc_index
from .intervals import interval_seconds

log = logging.getLogger(__name__)

BLOCK_SECONDS = 3600          # un bloque de rendimientos = una hora
# Instante en que cada moneda vale su ``start_px`` (2026-01-01 00:00 UTC, el inicio de ``SimClock``).
# Es fijo a propósito: el mercado simulado es la misma función del tiempo tras cada reinicio, así
# las posiciones paper persistidas siguen teniendo sentido con un reloj real.
DEFAULT_ANCHOR_TS = 1_767_225_600.0
BOOK_MAX_LEVELS = 20          # Hyperliquid devuelve como mucho 20 niveles por lado

# Flujos independientes del RNG (para que el libro no altere el camino de precios, etc.).
_PATH, _REGIME, _BOOK, _WALLS, _LIQS = 1, 2, 3, 4, 5


@dataclass(frozen=True)
class SimAsset:
    """Perfil de una moneda simulada."""

    meta: AssetMeta
    start_px: float
    daily_vol_pct: float = 3.0     # volatilidad diaria del paseo, en %
    spread_bps: float = 1.0
    depth_usd: float = 50_000.0    # liquidez media por nivel fino del libro


# Mismos metadatos que ``tests/conftest.py`` (FakeMarketData).
DEFAULT_ASSETS: dict[str, SimAsset] = {
    "BTC": SimAsset(AssetMeta("BTC", sz_decimals=5, max_leverage=40), 83_775.0, 2.5, 1.0, 400_000.0),
    "ETH": SimAsset(AssetMeta("ETH", sz_decimals=4, max_leverage=25), 3_200.0, 3.5, 1.5, 250_000.0),
    "SOL": SimAsset(AssetMeta("SOL", sz_decimals=2, max_leverage=20), 150.0, 4.5, 1.5, 100_000.0),
    "HYPE": SimAsset(AssetMeta("HYPE", sz_decimals=2, max_leverage=2), 86.285, 6.0, 1.2, 60_000.0),
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _coin_key(coin: str) -> int:
    return zlib.crc32(coin.encode("utf-8"))   # estable entre ejecuciones (hash() no lo es)


def _rng(*keys: int) -> np.random.Generator:
    return np.random.default_rng([int(k) % (1 << 63) for k in keys])


def price_decimals(px: float, sz_decimals: int) -> int:
    """Decimales de precio válidos en Hyperliquid (5 cifras significativas, máx. 6 - sz_decimals)."""
    if px >= 100_000:
        return 0
    max_dec = max(6 - sz_decimals, 0)
    sig_dec = max(5 - int(math.floor(math.log10(px))) - 1, 0)
    return min(max_dec, sig_dec)


def _snap(px: float, dec: int, up: bool) -> float:
    q = 10.0 ** dec
    v = math.ceil(px * q - 1e-9) if up else math.floor(px * q + 1e-9)
    return round(v / q, dec)


def _quote(fair: float, spread_bps: float, sz_decimals: int) -> tuple[float, float]:
    """bid/ask en ticks válidos alrededor del precio justo, con al menos un tick de spread."""
    dec = price_decimals(fair, sz_decimals)
    half = fair * spread_bps / 20_000.0
    bid = _snap(fair - half, dec, up=False)
    ask = _snap(fair + half, dec, up=True)
    if ask <= bid:
        ask = round(bid + 10.0 ** -dec, dec)
    return bid, ask


def _aggregate(levels: list[tuple[float, float]], ref_px: float, n_sig_figs: int,
               is_bid: bool) -> list[tuple[float, float]]:
    """Agrupa niveles como ``nSigFigs`` de Hyperliquid: bids hacia abajo, asks hacia arriba."""
    exp10 = int(math.floor(math.log10(ref_px))) - n_sig_figs + 1
    bucket = 10.0 ** exp10
    dec = max(0, -exp10)
    acc: dict[int, float] = {}
    for px, sz in levels:
        k = math.floor(px / bucket + 1e-9) if is_bid else math.ceil(px / bucket - 1e-9)
        acc[k] = acc.get(k, 0.0) + sz
    keys = sorted(acc, reverse=is_bid)
    return [(round(k * bucket, dec), acc[k]) for k in keys if k > 0]


class _DepthModel:
    """Libro y liquidaciones sintéticos, deterministas por (semilla, moneda, tiempo).

    ``anchor_px(coin, t)`` da el precio en un instante pasado: los muros y los clusters de
    liquidación se colocan respecto al precio al inicio de su época, así se quedan quietos
    mientras el precio se mueve (y desaparecen cuando el precio los atraviesa).
    """

    def __init__(self, seed: int, anchor_px: Callable[[str, float], float], *,
                 levels: int = 250, level_bps: float = 2.0, refresh_s: float = 30.0,
                 wall_period_s: float = 900.0, liq_period_s: float = 900.0):
        self.seed = seed
        self.anchor_px = anchor_px
        self.levels = levels
        self.level_bps = level_bps
        self.refresh_s = refresh_s
        self.wall_period_s = wall_period_s
        self.liq_period_s = liq_period_s

    def book(self, coin: str, meta: AssetMeta, depth_usd: float, bid: float, ask: float, t: float,
             n_sig_figs: Optional[int]) -> Book:
        fair = (bid + ask) / 2.0
        dec = price_decimals(fair, meta.sz_decimals)
        tick = 10.0 ** -dec
        spacing = max(tick, fair * self.level_bps / 10_000.0)
        j = np.arange(self.levels)
        bid_px = np.round(np.floor((bid - j * spacing) / tick + 1e-9) * tick, dec)
        ask_px = np.round(np.ceil((ask + j * spacing) / tick - 1e-9) * tick, dec)
        ck = _coin_key(coin)
        rng = _rng(self.seed, ck, _BOOK, int(t // self.refresh_s))
        depth_curve = 1.0 + j / self.levels          # más liquidez cuanto más lejos del mid
        bid_usd = depth_usd * rng.lognormal(0.0, 0.5, self.levels) * depth_curve
        ask_usd = depth_usd * rng.lognormal(0.0, 0.5, self.levels) * depth_curve

        # Muros: 3 por lado entre 0.15% y 2.8% del precio de anclaje, 30x-90x la liquidez media de
        # un nivel (siguen destacando al agrupar con nSigFigs, donde un tramo suma decenas de niveles).
        epoch = int(t // self.wall_period_s)
        anchor = self.anchor_px(coin, epoch * self.wall_period_s)
        wr = _rng(self.seed, ck, _WALLS, epoch)
        for usd, px_arr, sign in ((bid_usd, bid_px, -1.0), (ask_usd, ask_px, 1.0)):
            dist = wr.uniform(0.0015, 0.028, 3)
            mult = wr.uniform(30.0, 90.0, 3)
            for d, m in zip(dist, mult):
                wall_px = anchor * (1.0 + sign * d)
                idx = int(round((px_arr[0] - wall_px) * -sign / spacing))
                if 0 <= idx < self.levels:      # fuera de rango = el precio ya lo atravesó
                    usd[idx] += depth_usd * m

        bids = [(float(p), float(u / p)) for p, u in zip(bid_px, bid_usd) if p > 0]
        asks = [(float(p), float(u / p)) for p, u in zip(ask_px, ask_usd)]
        bids, asks = _dedupe(bids), _dedupe(asks)
        if n_sig_figs is not None:
            nsf = min(max(int(n_sig_figs), 2), 5)
            bids = _aggregate(bids, fair, nsf, is_bid=True)
            asks = _aggregate(asks, fair, nsf, is_bid=False)
        return {"bids": bids[:BOOK_MAX_LEVELS], "asks": asks[:BOOK_MAX_LEVELS], "ts": t}

    def liquidations(self, coin: str, depth_usd: float, fair: float, t: float) -> LiqLevels:
        epoch = int(t // self.liq_period_s)
        anchor = self.anchor_px(coin, epoch * self.liq_period_s)
        rng = _rng(self.seed, _coin_key(coin), _LIQS, epoch)
        levs = np.array([3, 5, 10, 15, 20, 25, 40, 50], dtype=float)
        probs = np.array([0.08, 0.14, 0.24, 0.14, 0.18, 0.1, 0.07, 0.05])
        out: LiqLevels = []
        for side, sign in (("long", -1.0), ("short", 1.0)):
            n = 40
            entry = anchor * np.exp(rng.normal(0.0, 0.012, n))
            lev = rng.choice(levs, n, p=probs)
            usd = depth_usd * rng.lognormal(1.0, 0.9, n)
            liq = entry * (1.0 + sign * 0.9 / lev)
            for px, u in zip(liq, usd):
                # Sólo las que siguen vivas: liquidación de longs por debajo, de shorts por encima.
                alive = px < fair * 0.9995 if side == "long" else px > fair * 1.0005
                if alive and abs(px / fair - 1.0) <= 0.25:
                    out.append({"liq_px": float(px), "usd": float(u), "side": side})
        out.sort(key=lambda d: d["liq_px"])
        return out


def _dedupe(levels: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for px, sz in levels:
        if out and out[-1][0] == px:
            out[-1] = (px, out[-1][1] + sz)
        else:
            out.append((px, sz))
    return out


# ---------------------------------------------------------------------------
# SimMarketData
# ---------------------------------------------------------------------------

class SimMarketData(MarketData):
    def __init__(self, clock: Clock, *, seed: int = 7, assets: Optional[dict[str, SimAsset]] = None,
                 step_seconds: int = 5, anchor_ts: Optional[float] = None, cache_blocks: int = 512):
        if step_seconds <= 0 or BLOCK_SECONDS % step_seconds:
            raise ValueError(f"step_seconds debe dividir {BLOCK_SECONDS}: {step_seconds}")
        self.clock = clock
        self.seed = int(seed)
        self.assets: dict[str, SimAsset] = dict(DEFAULT_ASSETS if assets is None else assets)
        self.step = int(step_seconds)
        self.n = BLOCK_SECONDS // self.step
        # El precio al inicio del bloque que contiene ``anchor_ts`` es ``start_px``.
        self.anchor_block = int((DEFAULT_ANCHOR_TS if anchor_ts is None else anchor_ts) // BLOCK_SECONDS)
        self.cache_blocks = cache_blocks
        self._arrays: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._summary: dict[tuple[str, int], tuple[float, float, float, float]] = {}
        self._levels: dict[str, dict[int, float]] = {}
        self._known: dict[str, tuple[int, int]] = {}
        self._regimes: dict[tuple[str, int], float] = {}
        self.depth = _DepthModel(self.seed, self.price_at)

    def add_asset(self, asset: SimAsset) -> None:
        self.assets[asset.meta.coin] = asset

    # --- camino de precios ----------------------------------------------------
    def _sigma(self, coin: str) -> float:
        return self.assets[coin].daily_vol_pct / 100.0 * math.sqrt(self.step / 86_400.0)

    def _block(self, coin: str, b: int) -> np.ndarray:
        """Log-precios del bloque relativos a su inicio: ``n + 1`` puntos (el último = inicio del siguiente)."""
        key = (coin, b)
        x = self._arrays.get(key)
        if x is not None:
            self._arrays.move_to_end(key)
            return x
        ck = _coin_key(coin)
        regime_z = self._regimes.get((coin, b // 8))                       # régimen de volatilidad de 8 h
        if regime_z is None:
            regime_z = self._regimes[(coin, b // 8)] = float(_rng(self.seed, ck, _REGIME, b // 8).standard_normal())
        rng = _rng(self.seed, ck, _PATH, b)
        hour_z = rng.standard_normal()
        # Normalizado para que la varianza media coincida con la volatilidad nominal.
        vol = self._sigma(coin) * math.exp(0.45 * regime_z + 0.2 * hour_z - (0.45 ** 2 + 0.2 ** 2))
        r = rng.standard_normal(self.n) * vol
        jumps = rng.random(self.n) < 0.0015
        if jumps.any():
            r[jumps] += rng.standard_normal(int(jumps.sum())) * vol * 6.0
        x = np.empty(self.n + 1)
        x[0] = 0.0
        np.cumsum(r, out=x[1:])
        self._arrays[key] = x
        if len(self._arrays) > self.cache_blocks:
            self._arrays.popitem(last=False)
        self._summary[key] = (float(x[-1]), float(x.max()), float(x.min()), float(np.abs(r).sum()))
        return x

    def _block_summary(self, coin: str, b: int) -> tuple[float, float, float, float]:
        s = self._summary.get((coin, b))
        if s is None:
            self._block(coin, b)
            s = self._summary[(coin, b)]
        return s

    def _level(self, coin: str, b: int) -> float:
        """Log-precio al inicio del bloque ``b`` (se encadena desde el bloque de anclaje)."""
        levels = self._levels.get(coin)
        if levels is None:
            levels = self._levels[coin] = {self.anchor_block: math.log(self.assets[coin].start_px)}
            self._known[coin] = (self.anchor_block, self.anchor_block)
        if b in levels:
            return levels[b]
        lo, hi = self._known[coin]
        while hi < b:
            levels[hi + 1] = levels[hi] + self._block_summary(coin, hi)[0]
            hi += 1
        while lo > b:
            levels[lo - 1] = levels[lo] - self._block_summary(coin, lo - 1)[0]
            lo -= 1
        self._known[coin] = (lo, hi)
        return levels[b]

    def _log_price(self, coin: str, t: float) -> float:
        kf = t / self.step
        k = math.floor(kf)
        frac = kf - k
        b, i = divmod(k, self.n)
        x = self._block(coin, b)
        return self._level(coin, b) + x[i] + (x[i + 1] - x[i]) * frac

    def price_at(self, coin: str, t: float) -> float:
        """Precio justo (sin spread) de ``coin`` en el instante ``t``."""
        if coin not in self.assets:
            raise KeyError(coin)
        return math.exp(self._log_price(coin, t))

    def fair_price(self, coin: str) -> float:
        return self.price_at(coin, self.clock.now())

    def _ohlcv(self, coin: str, t0: float, t1: float) -> tuple[float, float, float, float, float]:
        lo_open, lo_close = self._log_price(coin, t0), self._log_price(coin, t1)
        hi, lo = max(lo_open, lo_close), min(lo_open, lo_close)
        activity = 0.0          # suma de |rendimientos| / precio base del bloque (aditiva entre intervalos)
        b0, b1 = int(t0 // BLOCK_SECONDS), int(t1 // BLOCK_SECONDS)
        if b1 > b0 and t1 == b1 * BLOCK_SECONDS:
            b1 -= 1                                   # el borde ya es el último punto del bloque anterior
        for b in range(b0, b1 + 1):
            start = b * BLOCK_SECONDS
            i0 = max(0, math.ceil((t0 - start) / self.step - 1e-9))
            i1 = min(self.n, math.floor((t1 - start) / self.step + 1e-9))
            if i0 > i1:
                continue
            base = self._level(coin, b)
            if i0 == 0 and i1 == self.n:              # bloque entero: basta el resumen
                _, mx, mn, ab = self._block_summary(coin, b)
                hi, lo = max(hi, base + mx), min(lo, base + mn)
            else:
                seg = self._block(coin, b)[i0:i1 + 1]
                hi, lo = max(hi, base + float(seg.max())), min(lo, base + float(seg.min()))
                ab = float(np.abs(np.diff(seg)).sum())
            activity += ab / math.exp(base)
        # Volumen sintético (en monedas) proporcional a la actividad del camino.
        volume = activity / self._sigma(coin) * self.assets[coin].depth_usd * 0.05
        return math.exp(lo_open), math.exp(hi), math.exp(lo), math.exp(lo_close), volume

    # --- MarketData -----------------------------------------------------------
    def meta(self, coin: str) -> AssetMeta:
        return replace(self.assets[coin].meta)

    def coins(self) -> list[str]:
        return list(self.assets)

    def bid_ask(self, coin: str) -> tuple[float, float]:
        asset = self.assets[coin]
        return _quote(self.fair_price(coin), asset.spread_bps, asset.meta.sz_decimals)

    def l2_book(self, coin: str, n_sig_figs: Optional[int] = None) -> Book:
        asset = self.assets[coin]
        bid, ask = self.bid_ask(coin)
        return self.depth.book(coin, asset.meta, asset.depth_usd, bid, ask, self.clock.now(), n_sig_figs)

    def candles(self, coin: str, interval: str, limit: int) -> pd.DataFrame:
        if coin not in self.assets:
            raise KeyError(coin)
        sec = interval_seconds(interval)
        if sec % self.step or (sec % BLOCK_SECONDS and BLOCK_SECONDS % sec):
            raise ValueError(f"intervalo {interval} incompatible con pasos de {self.step} s")
        limit = max(int(limit), 1)
        now = self.clock.now()
        current_open = math.floor(now / sec) * sec
        opens = [current_open - sec * (limit - 1 - j) for j in range(limit)]
        rows = [self._ohlcv(coin, o, min(o + sec, now)) for o in opens]
        df = pd.DataFrame(rows, columns=CANDLE_COLUMNS,
                          index=utc_index(np.array(opens, dtype="int64"), "s"))
        return df.astype(float)

    def liquidation_levels(self, coin: str) -> LiqLevels:
        asset = self.assets[coin]
        return self.depth.liquidations(coin, asset.depth_usd, self.fair_price(coin), self.clock.now())

    def last_update_ts(self, coin: str) -> Optional[float]:
        return self.clock.now()


# ---------------------------------------------------------------------------
# ReplayMarketData
# ---------------------------------------------------------------------------

@dataclass
class _Series:
    ts: np.ndarray          # apertura de cada vela, segundos UNIX (tiempo del CSV)
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray           # noqa: E741 - nombre de columna OHLC
    c: np.ndarray
    v: np.ndarray
    interval_s: int


_TS_ALIASES = ("timestamp", "time", "datetime", "date", "t", "open_time")


def _parse_timestamps(raw: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(raw):
        vals = raw.astype(float).to_numpy()
        if len(vals) and np.nanmedian(vals) > 1e11:   # milisegundos
            vals = vals / 1000.0
        return vals
    dt = pd.to_datetime(raw, utc=True)
    return (dt - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()


def _guess_meta(coin: str, px: float) -> AssetMeta:
    if coin in DEFAULT_ASSETS:
        return replace(DEFAULT_ASSETS[coin].meta)
    szd = max(0, min(5, int(math.floor(math.log10(px))) + 1)) if px > 0 else 0
    return AssetMeta(coin, sz_decimals=szd, max_leverage=10)


class ReplayMarketData(MarketData):
    """Reproduce un CSV de velas. Sin columna ``coin``/``symbol`` el CSV es de una sola moneda
    (``coin`` o, si no se da, el primer token del nombre del fichero: ``btc_15m.csv`` → BTC).

    ``offset`` = tiempo del CSV − tiempo del reloj. Por defecto se calcula para que la
    reproducción empiece tras ``warmup_candles`` velas (historia para los indicadores) y las
    velas queden alineadas a su intervalo en el tiempo del reloj. ``offset=0`` reproduce en
    tiempo absoluto (útil con un ``SimClock`` puesto en la fecha del CSV).
    """

    def __init__(self, csv_path: Union[str, Path], clock: Clock, *, coin: Optional[str] = None,
                 meta: Optional[AssetMeta] = None, warmup_candles: Optional[int] = None,
                 offset: Optional[float] = None, spread_bps: float = 2.0,
                 depth_usd: Optional[float] = None, seed: int = 7):
        self.path = Path(csv_path)
        self.clock = clock
        self.spread_bps = spread_bps
        try:
            df = pd.read_csv(self.path)
        except (OSError, ValueError) as e:
            raise MarketDataError(f"no se pudo leer el CSV de replay {self.path}: {e}") from e
        cols = {str(c).strip().lower(): c for c in df.columns}
        ts_col = next((cols[a] for a in _TS_ALIASES if a in cols), None)
        missing = [c for c in CANDLE_COLUMNS if c not in cols]
        if ts_col is None or missing:
            raise MarketDataError(
                f"CSV {self.path.name}: se esperan columnas timestamp,open,high,low,close,volume "
                f"(faltan: {', '.join(missing + ([] if ts_col is not None else ['timestamp']))})")
        coin_col = cols.get("coin") or cols.get("symbol")
        frame = pd.DataFrame({"ts": _parse_timestamps(df[ts_col]),
                              **{c: pd.to_numeric(df[cols[c]], errors="coerce") for c in CANDLE_COLUMNS}})
        if coin_col is not None:
            frame["coin"] = df[coin_col].astype(str).str.strip().str.upper()
        else:
            frame["coin"] = (coin or self._coin_from_name()).upper()
        frame = frame.dropna()
        if frame.empty:
            raise MarketDataError(f"CSV {self.path.name}: sin velas válidas")

        self.series: dict[str, _Series] = {}
        self.metas: dict[str, AssetMeta] = {}
        self.depths: dict[str, float] = {}
        for name, g in frame.groupby("coin", sort=False):
            g = g.sort_values("ts").drop_duplicates("ts", keep="last")
            ts = g["ts"].to_numpy(dtype=float)
            diffs = np.diff(ts)
            interval = int(round(float(np.median(diffs)))) if len(diffs) else 60
            if interval <= 0:
                raise MarketDataError(f"CSV {self.path.name}: timestamps de {name} no crecientes")
            self.series[name] = _Series(ts, *(g[c].to_numpy(dtype=float) for c in CANDLE_COLUMNS),
                                        interval_s=interval)
            first_px = float(g["close"].iloc[0])
            self.metas[name] = replace(meta, coin=name) if meta is not None else _guess_meta(name, first_px)
            default_depth = DEFAULT_ASSETS[name].depth_usd if name in DEFAULT_ASSETS else 50_000.0
            self.depths[name] = depth_usd if depth_usd is not None else default_depth

        ref = next(iter(self.series.values()))
        if offset is None:
            warm = min(200, len(ref.ts) // 2) if warmup_candles is None else warmup_candles
            warm = min(max(int(warm), 0), len(ref.ts) - 1)
            now0 = clock.now()
            offset = float(ref.ts[warm]) - math.floor(now0 / ref.interval_s) * ref.interval_s
        self.offset = float(offset)
        self.depth = _DepthModel(int(seed), self.price_at)
        self._warned_end = False

    def _coin_from_name(self) -> str:
        tokens = [t for t in re.split(r"[^A-Za-z0-9]+", self.path.stem) if t]
        return tokens[0] if tokens else "BTC"

    # --- reproducción -----------------------------------------------------------
    def _series(self, coin: str) -> _Series:
        return self.series[coin]       # KeyError si la moneda no está en el CSV

    @staticmethod
    def _vertices(o: float, h: float, lo: float, c: float) -> tuple[float, float, float, float]:
        return (o, lo, h, c) if c >= o else (o, h, lo, c)

    @classmethod
    def _path(cls, o: float, h: float, lo: float, c: float, f: float) -> tuple[float, float, float]:
        """(precio, máximo visto, mínimo visto) a una fracción ``f`` de la vela."""
        v = cls._vertices(o, h, lo, c)
        f = min(max(f, 0.0), 1.0)
        seg = min(int(f * 3), 2)
        px = v[seg] + (v[seg + 1] - v[seg]) * (f * 3 - seg)
        seen = v[:seg + 1] + (px,)
        return px, max(seen), min(seen)

    def _locate(self, s: _Series, t: float) -> tuple[int, float]:
        """Índice de la vela en curso y fracción transcurrida (``idx=-1`` antes del inicio)."""
        csv_t = t + self.offset
        idx = int(np.searchsorted(s.ts, csv_t, side="right")) - 1
        if idx < 0:
            return -1, 0.0
        return idx, (csv_t - s.ts[idx]) / s.interval_s

    def price_at(self, coin: str, t: float) -> float:
        s = self._series(coin)
        idx, f = self._locate(s, t)
        if idx < 0:
            return float(s.o[0])
        if idx == len(s.ts) - 1 and f >= 1.0:
            return float(s.c[-1])
        return self._path(s.o[idx], s.h[idx], s.l[idx], s.c[idx], f)[0]

    @property
    def finished(self) -> bool:
        """True cuando el reloj ha pasado la última vela de todas las monedas."""
        now = self.clock.now()
        return all(now + self.offset >= s.ts[-1] + s.interval_s for s in self.series.values())

    # --- MarketData -------------------------------------------------------------
    def meta(self, coin: str) -> AssetMeta:
        return replace(self.metas[coin])

    def coins(self) -> list[str]:
        return list(self.series)

    def bid_ask(self, coin: str) -> tuple[float, float]:
        if self.finished and not self._warned_end:
            self._warned_end = True
            log.warning("replay %s terminado: el precio queda congelado en el último cierre", self.path.name)
        return _quote(self.price_at(coin, self.clock.now()), self.spread_bps, self.metas[coin].sz_decimals)

    def l2_book(self, coin: str, n_sig_figs: Optional[int] = None) -> Book:
        bid, ask = self.bid_ask(coin)
        return self.depth.book(coin, self.metas[coin], self.depths[coin], bid, ask, self.clock.now(), n_sig_figs)

    def candles(self, coin: str, interval: str, limit: int) -> pd.DataFrame:
        s = self._series(coin)
        sec = interval_seconds(interval)
        if sec < s.interval_s or sec % s.interval_s:
            raise MarketDataError(
                f"replay {coin}: el CSV tiene velas de {s.interval_s} s; no se puede servir {interval}")
        limit = max(int(limit), 1)
        idx, f = self._locate(s, self.clock.now())
        if idx < 0:
            return empty_candles()
        ratio = sec // s.interval_s
        lo = max(0, idx + 1 - (limit + 1) * ratio)
        sl = slice(lo, idx + 1)
        o, h, low, c, v = (a[sl].copy() for a in (s.o, s.h, s.l, s.c, s.v))
        if f < 1.0:                                   # vela en curso: sólo lo "visto" hasta ahora
            px, hi_seen, lo_seen = self._path(o[-1], h[-1], low[-1], c[-1], f)
            h[-1], low[-1], c[-1], v[-1] = hi_seen, lo_seen, px, v[-1] * f
        opens = np.round(s.ts[sl] - self.offset).astype("int64")
        df = pd.DataFrame({"open": o, "high": h, "low": low, "close": c, "volume": v},
                          index=utc_index(opens, "s"))
        if ratio > 1:
            df = (df.resample(f"{sec}s", origin="epoch", label="left", closed="left")
                    .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                    .dropna(subset=["open"]))
        return df[CANDLE_COLUMNS].astype(float).tail(limit)

    def liquidation_levels(self, coin: str) -> LiqLevels:
        now = self.clock.now()
        return self.depth.liquidations(coin, self.depths[coin], self.price_at(coin, now), now)

    def last_update_ts(self, coin: str) -> Optional[float]:
        # Al acabarse el CSV los datos envejecen: el riesgo dejará de abrir operaciones.
        s = self._series(coin)
        return min(self.clock.now(), float(s.ts[-1] + s.interval_s - self.offset))

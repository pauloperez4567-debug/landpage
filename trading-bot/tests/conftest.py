"""Fixtures compartidas: un mercado falso, determinista y controlable desde el test."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desk.clock import SimClock  # noqa: E402
from desk.config import load_settings  # noqa: E402
from desk.exchange.base import MarketData  # noqa: E402
from desk.models import AssetMeta  # noqa: E402

INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


class FakeMarketData(MarketData):
    """Mercado en memoria. El test fija precios, libro, velas y liquidaciones a mano.

    - ``set_price(coin, mid, spread_bps)`` mueve bid/ask.
    - ``set_book(coin, bids, asks)`` fija el libro (listas de (px, sz)).
    - ``set_candles(coin, df)`` fija velas para cualquier intervalo (o por intervalo).
    - ``set_liqs(coin, levels)`` fija niveles de liquidación.
    """

    def __init__(self, clock: Optional[SimClock] = None):
        self.clock = clock or SimClock()
        self._meta: dict[str, AssetMeta] = {
            "BTC": AssetMeta("BTC", sz_decimals=5, max_leverage=40),
            "ETH": AssetMeta("ETH", sz_decimals=4, max_leverage=25),
            "SOL": AssetMeta("SOL", sz_decimals=2, max_leverage=20),
            "HYPE": AssetMeta("HYPE", sz_decimals=2, max_leverage=2),
        }
        self._px: dict[str, tuple[float, float]] = {
            "BTC": (83_770.0, 83_780.0),
            "ETH": (3_199.5, 3_200.5),
            "SOL": (149.98, 150.02),
            "HYPE": (86.28, 86.29),
        }
        self._books: dict[str, dict] = {}
        self._candles: dict[tuple[str, str], pd.DataFrame] = {}
        self._liqs: dict[str, list] = {}
        self._ts: dict[str, float] = {}
        self.fail_next: int = 0   # simula errores de red en las próximas N llamadas

    # --- controles del test -------------------------------------------------
    def set_price(self, coin: str, mid: float, spread_bps: float = 1.0) -> None:
        half = mid * spread_bps / 20_000.0
        self._px[coin] = (mid - half, mid + half)
        self._ts[coin] = self.clock.now()

    def set_bid_ask(self, coin: str, bid: float, ask: float) -> None:
        self._px[coin] = (bid, ask)
        self._ts[coin] = self.clock.now()

    def set_book(self, coin: str, bids: list, asks: list) -> None:
        self._books[coin] = {"bids": list(bids), "asks": list(asks), "ts": self.clock.now()}

    def set_candles(self, coin: str, df: pd.DataFrame, interval: Optional[str] = None) -> None:
        self._candles[(coin, interval or "*")] = df

    def set_liqs(self, coin: str, levels: list) -> None:
        self._liqs[coin] = list(levels)

    def add_meta(self, meta: AssetMeta, bid: float, ask: float) -> None:
        self._meta[meta.coin] = meta
        self._px[meta.coin] = (bid, ask)

    def _maybe_fail(self) -> None:
        if self.fail_next > 0:
            self.fail_next -= 1
            from desk.exchange.base import MarketDataError
            raise MarketDataError("fallo de red simulado")

    # --- MarketData ---------------------------------------------------------
    def meta(self, coin: str) -> AssetMeta:
        return self._meta[coin]

    def coins(self) -> list[str]:
        return list(self._meta)

    def bid_ask(self, coin: str) -> tuple[float, float]:
        self._maybe_fail()
        return self._px[coin]

    def l2_book(self, coin: str, n_sig_figs: Optional[int] = None) -> dict:
        self._maybe_fail()
        if coin in self._books:
            return self._books[coin]
        bid, ask = self._px[coin]
        return {"bids": [(bid, 1.0)], "asks": [(ask, 1.0)], "ts": self.clock.now()}

    def candles(self, coin: str, interval: str, limit: int) -> pd.DataFrame:
        self._maybe_fail()
        df = self._candles.get((coin, interval)) or self._candles.get((coin, "*"))
        if df is None:
            df = make_candles(mid=self.mid(coin), n=max(limit, 50), interval=interval,
                              end_ts=self.clock.now())
        return df.tail(limit)

    def liquidation_levels(self, coin: str) -> list:
        return list(self._liqs.get(coin, []))

    def last_update_ts(self, coin: str) -> Optional[float]:
        return self._ts.get(coin, self.clock.now())


def make_candles(mid: float = 100.0, n: int = 200, interval: str = "15m", end_ts: float = 1_767_225_600.0,
                 seed: int = 7, vol_pct: float = 0.3, drift_pct: float = 0.0,
                 closes: Optional[list] = None) -> pd.DataFrame:
    """Velas sintéticas deterministas que terminan cerca de ``mid``.

    Si se pasa ``closes`` se usan esos cierres exactos (open = cierre anterior,
    high/low = +-0.1% alrededor del cuerpo).
    """
    step = INTERVAL_SECONDS.get(interval, 900)
    if closes is not None:
        c = np.asarray(closes, dtype=float)
        n = len(c)
    else:
        rng = np.random.default_rng(seed)
        rets = rng.normal(drift_pct / 100.0, vol_pct / 100.0, n)
        c = mid * np.exp(np.cumsum(rets) - np.sum(rets))
    o = np.concatenate([[c[0]], c[:-1]])
    hi = np.maximum(o, c) * 1.001
    lo = np.minimum(o, c) * 0.999
    vol = np.full(n, 1_000.0)
    idx = pd.to_datetime([end_ts - step * (n - 1 - i) for i in range(n)], unit="s", utc=True)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol}, index=idx)


@pytest.fixture
def clock() -> SimClock:
    return SimClock()


@pytest.fixture
def market(clock: SimClock) -> FakeMarketData:
    return FakeMarketData(clock)


@pytest.fixture
def settings(tmp_path):
    return load_settings(env={}, dotenv_path=None, state_dir=str(tmp_path / "state"), llm_enabled=False)

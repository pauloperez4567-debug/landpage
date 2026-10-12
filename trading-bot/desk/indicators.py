"""Indicadores técnicos en pandas puro (sin TA-Lib).

Todas las funciones reciben ``pd.Series`` alineadas por índice y devuelven series
del mismo índice; los primeros valores (calentamiento) quedan en ``NaN``.
Las medias "Wilder" se siembran con la media simple de los primeros ``n`` valores,
igual que TA-Lib, para que los resultados coincidan con las plataformas habituales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(series: pd.Series, n: int) -> pd.Series:
    """Suavizado de Wilder (RMA) con semilla = media simple de los primeros n valores válidos."""
    s = series.astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.dropna()
    if len(valid) < n:
        return out
    start = s.index.get_loc(valid.index[n - 1])
    values = s.to_numpy()
    prev = float(valid.iloc[:n].mean())
    out.iloc[start] = prev
    for i in range(start + 1, len(values)):
        v = values[i]
        if np.isnan(v):
            out.iloc[i] = prev
            continue
        prev = prev + (v - prev) / n
        out.iloc[i] = prev
    return out


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), n)
    loss = wilder((-delta).clip(lower=0), n)
    total = gain + loss
    out = 100.0 * gain / total
    # Ventana totalmente plana: 0/0. Se define como 50 (neutral) en vez de NaN.
    return out.where(total != 0, 50.0).where(total.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Devuelve (superior, media, inferior). Desviación poblacional, como TA-Lib."""
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def keltner(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20,
            mult: float = 1.5) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Canales de Keltner sobre SMA + múltiplo del ATR. Devuelve (superior, media, inferior)."""
    mid = sma(close, n)
    a = atr(high, low, close, n)
    return mid + mult * a, mid, mid - mult * a


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average Directional Index (Wilder)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    tr = true_range(high, low, close)
    tr.iloc[0] = np.nan
    atr_n = wilder(tr, n)
    plus_di = 100.0 * wilder(plus_dm, n) / atr_n
    minus_di = 100.0 * wilder(minus_dm, n) / atr_n
    di_sum = plus_di + minus_di
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum).where(di_sum != 0, 0.0)
    return wilder(dx, n)


def squeeze(high: pd.Series, low: pd.Series, close: pd.Series, bb_n: int = 20, bb_k: float = 2.0,
            kc_n: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """True cuando las bandas de Bollinger están dentro de los canales de Keltner (volatilidad comprimida)."""
    bb_up, _, bb_lo = bollinger(close, bb_n, bb_k)
    kc_up, _, kc_lo = keltner(high, low, close, kc_n, kc_mult)
    sq = (bb_up < kc_up) & (bb_lo > kc_lo)
    return sq.where(bb_up.notna() & kc_up.notna(), False).astype(bool)

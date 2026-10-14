"""Datos y estrategias sintéticas para los tests del optimizador, el playbook y los informes.

- ``trend_bars``: tendencias alternas (subida/bajada cada ``leg`` velas) con ruido: un seguidor de
  tendencia tiene ventaja real.
- ``pattern_bars``: paseo aleatorio sin deriva salvo un patrón claro: la vela que abre a la hora
  ``hour + 1`` sube un 1.5 % sin apenas mecha inferior.
- ``HourPattern``: compra al cierre de la vela de la hora ``hour`` (el parámetro a optimizar).
- ``CoinFlip``: entradas al azar con semilla = parámetro: con 400 semillas siempre hay alguna que
  "funciona" dentro de muestra (sobreajuste de libro).
- ``Leaky``: mira la vela siguiente (fuga de datos deliberada).
- ``Broken``: falla al calcular señales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from desk.models import Side
from desk.strategy import Strategy, _Proposal


def _frame(o, h, l, c, start="2026-03-01", freq="1h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(c), freq=freq, tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1_000.0}, index=idx)


def trend_bars(seed: int = 1, n: int = 3000, leg: int = 400, mu: float = 0.0012, vol: float = 0.006) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.where((np.arange(n) // leg) % 2 == 0, mu, -mu)
    c = 80_000.0 * np.exp(np.cumsum(rng.normal(drift, vol)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    return _frame(o, h, l, c)


def pattern_bars(hour: int = 7, n: int = 1500, seed: int = 3, jump: float = 0.015, vol: float = 0.003) -> pd.DataFrame:
    """La vela de la hora ``hour + 1`` sube ``jump`` (mecha inferior de 0.05 %); el resto, ruido con una deriva
    que compensa los saltos (el precio no tiene tendencia: sólo el patrón da ventaja)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-03-01", periods=n, freq="1h", tz="UTC")
    special = idx.hour.to_numpy() == (hour + 1) % 24
    rets = rng.normal(-jump / 23.0, vol, n)
    rets[special] = jump
    c = 100.0 * np.exp(np.cumsum(rets))
    o = np.r_[100.0, c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.001, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.001, n)))
    h[special] = c[special] * 1.001
    l[special] = o[special] * 0.9995
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1_000.0}, index=idx)


class HourPattern(Strategy):
    """LONG al cierre de la vela de la hora ``hour``; SL -1 %, TP +1 % (1R)."""

    name = "hour_pattern"
    interval = "1h"
    lookback = 10
    min_bars = 2
    hour = 0
    PARAMS = ("hour",)
    PARAM_GRID = {"hour": list(range(24))}

    def _proposals(self, bars, zones, ref):
        mask = bars.index.hour.to_numpy() == self.hour
        return [_Proposal(Side.LONG, mask, ref * 0.99, ref * 1.01, lambda i: f"hora {self.hour}")]


class CoinFlip(Strategy):
    """Entradas al azar: en cada vela, con probabilidad 0.15, LONG o SHORT a cara o cruz (SL/TP a 1 %).

    El azar de la fila ``i`` sólo depende de ``i`` y de la semilla (el mismo prefijo de números aleatorios):
    es causal, sólo que no tiene nada que ver con el mercado. (Sortear ``u`` y luego ``d`` con dos llamadas
    no lo sería: ``d[i]`` dependería del número de velas, y el optimizador lo detecta como fuga.)"""

    name = "coin_flip"
    interval = "1h"
    lookback = 10
    min_bars = 2
    seed = 0
    PARAMS = ("seed",)
    PARAM_GRID = {"seed": list(range(400))}

    def _proposals(self, bars, zones, ref):
        draws = np.random.default_rng(self.seed).random((len(bars), 2))   # fila i = sorteos 2i y 2i+1 (causal)
        u, d = draws[:, 0], draws[:, 1]
        fire = u < 0.15
        return [_Proposal(Side.LONG, fire & (d < 0.5), ref * 0.99, ref * 1.01, lambda i: "cara"),
                _Proposal(Side.SHORT, fire & (d >= 0.5), ref * 1.01, ref * 0.99, lambda i: "cruz")]


class Leaky(Strategy):
    """Fuga deliberada: LONG si la vela SIGUIENTE cierra más arriba, SHORT si no (mira el futuro)."""

    name = "leaky"
    interval = "1h"
    lookback = 10
    min_bars = 2
    k = 1
    PARAMS = ("k",)
    PARAM_GRID = {"k": [1, 2]}

    def _proposals(self, bars, zones, ref):
        c = bars["close"].to_numpy(dtype=float)
        nxt = np.r_[c[self.k:], np.full(self.k, np.nan)]
        with np.errstate(invalid="ignore"):
            up, down = nxt > c, nxt < c
        return [_Proposal(Side.LONG, up, ref * 0.99, ref * 1.01, lambda i: "futuro"),
                _Proposal(Side.SHORT, down, ref * 1.01, ref * 0.99, lambda i: "futuro")]


class Broken(Strategy):
    name = "broken"
    interval = "1h"
    lookback = 10
    min_bars = 2
    x = 1
    PARAMS = ("x",)
    PARAM_GRID = {"x": [1, 2]}

    def _proposals(self, bars, zones, ref):
        raise RuntimeError("fallo de prueba")

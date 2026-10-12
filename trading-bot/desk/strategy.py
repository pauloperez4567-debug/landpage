"""Estrategias autónomas: leen velas (+ zonas de liquidez) y proponen ``Signal``.

Reglas comunes (las aplica ``Strategy.evaluate`` antes de llamar a la estrategia):

- **Sólo velas cerradas.** ``MarketData.candles`` puede incluir la vela en curso
  (Hyperliquid siempre la incluye), así que la mesa pasa las velas tal cual y aquí
  se decide qué es cerrado:
  * si se conoce la hora (``now`` en ``evaluate`` o el ``Clock`` del constructor),
    la última vela se descarta cuando ``apertura + duración > now``;
  * si no se conoce, se asume que la última está en curso y se descarta siempre
    (en el peor caso la señal llega una vela tarde, nunca antes de tiempo).
- Con hora conocida, si la última vela cerrada es de hace más de una vela (+
  ``candle_max_lag_s``) los datos se consideran congelados y no hay señal.
- Como mucho una señal por moneda y vela cerrada: la mesa evalúa cada
  ``strategy_seconds`` y no debe repetir la misma entrada durante toda la vela.
- SL y TP siempre del lado correcto (LONG: SL < entrada < TP; SHORT al revés).
  TP = ``default_tp_r`` veces el riesgo, leído en cada evaluación para respetar
  cambios en caliente (``tp 3``).
- Entrada a mercado: ``Signal.entry_px`` es la referencia (el mid actual, o el
  último cierre si no hay mid válido).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Mapping, Optional

import pandas as pd

from .clock import Clock
from .config import Settings
from .exchange.base import CANDLE_COLUMNS
from .exchange.intervals import interval_seconds
from .indicators import adx, atr, bollinger, rsi, squeeze
from .models import EntryMode, Side, Signal, Zone

log = logging.getLogger(__name__)

Zones = Mapping[Side, Optional[Zone]]


def closed_candles(candles: pd.DataFrame, interval: str, now: Optional[float] = None) -> pd.DataFrame:
    """Quita la vela en curso (ver docstring del módulo). Devuelve una vista, no copia."""
    if candles is None or len(candles) == 0:
        return candles
    if now is None or not isinstance(candles.index, pd.DatetimeIndex):
        return candles.iloc[:-1]
    last_open = candles.index[-1].timestamp()
    if last_open + interval_seconds(interval) > now:
        return candles.iloc[:-1]
    return candles


def _bracket(side: Side, entry: float, sl: float, tp_r: float) -> Optional[tuple[float, float]]:
    """(SL, TP) con TP a ``tp_r`` R; None si el SL no queda del lado correcto de la entrada."""
    risk = entry - sl if side is Side.LONG else sl - entry
    if not (math.isfinite(risk) and risk > 0 and tp_r > 0):
        return None
    tp = entry + tp_r * risk if side is Side.LONG else entry - tp_r * risk
    if tp <= 0:
        return None
    return sl, tp


def _finite(*xs: float) -> bool:
    return all(x is not None and math.isfinite(x) for x in xs)


class Strategy(ABC):
    """Base de las estrategias. Las subclases implementan ``_evaluate_closed``.

    ``lookback`` es cuántas velas debe pedir la mesa (incluida la posible vela en
    curso); ``min_bars`` cuántas velas *cerradas* hacen falta para calentar los
    indicadores.
    """

    name: str = "base"
    interval: str = "15m"
    lookback: int = 100
    min_bars: int = 30
    candle_max_lag_s: float = 120.0

    def __init__(self, settings: Settings, clock: Optional[Clock] = None):
        self.settings = settings
        self.clock = clock
        self._last_signal_bar: dict[str, pd.Timestamp] = {}

    def evaluate(self, coin: str, candles: pd.DataFrame, mid: Optional[float],
                 zones: Optional[Zones] = None, now: Optional[float] = None) -> Optional[Signal]:
        """Señal sobre la última vela cerrada o None. Nunca lanza por datos raros."""
        if candles is None or len(candles) == 0 or any(c not in candles.columns for c in CANDLE_COLUMNS):
            return None
        if now is None and self.clock is not None:
            now = self.clock.now()
        bars = closed_candles(candles, self.interval, now)
        if len(bars) < self.min_bars:
            return None
        if not bars[["open", "high", "low", "close"]].iloc[-2:].notna().all().all():
            return None
        if now is not None and isinstance(bars.index, pd.DatetimeIndex):
            step = interval_seconds(self.interval)
            closed_at = bars.index[-1].timestamp() + step
            if now - closed_at > step + self.candle_max_lag_s:
                log.warning("%s %s: velas congeladas (última cerrada hace %.0f s)", self.name, coin,
                            now - closed_at)
                return None
        bar_ts = bars.index[-1]
        if self._last_signal_bar.get(coin) == bar_ts:
            return None
        signal = self._evaluate_closed(coin, bars, mid, zones or {})
        if signal is not None:
            self._last_signal_bar[coin] = bar_ts
        return signal

    @abstractmethod
    def _evaluate_closed(self, coin: str, bars: pd.DataFrame, mid: Optional[float],
                         zones: Zones) -> Optional[Signal]:
        """Lógica de la estrategia: ``bars`` son sólo velas cerradas (al menos ``min_bars``)."""

    @staticmethod
    def _reference_px(mid: Optional[float], bars: pd.DataFrame) -> float:
        if mid is not None and math.isfinite(mid) and mid > 0:
            return float(mid)
        return float(bars["close"].iloc[-1])


class LiquiditySweepStrategy(Strategy):
    """Reversión tras barrer una zona de liquidez (velas de 15m).

    LONG: la vela anterior perfora la zona LONG (``low <= zone.high``), la última
    cierra por encima de ``zone.low`` (recupera) y el RSI(14) está por debajo de
    40 y subiendo. SL bajo ``zone.low - 0.5·ATR(14)``. SHORT es el espejo: la
    vela anterior entra en la zona SHORT (``high >= zone.low``), la última cierra
    por debajo de ``zone.high`` y el RSI está por encima de 60 y bajando.
    """

    name = "liquidity_sweep"
    interval = "15m"
    lookback = 120
    min_bars = 30
    rsi_period = 14
    atr_period = 14
    rsi_long_max = 40.0
    rsi_short_min = 60.0
    sl_atr_mult = 0.5

    def _evaluate_closed(self, coin, bars, mid, zones):
        r = rsi(bars["close"], self.rsi_period)
        a = atr(bars["high"], bars["low"], bars["close"], self.atr_period)
        r_now, r_prev, atr_now = float(r.iloc[-1]), float(r.iloc[-2]), float(a.iloc[-1])
        if not _finite(r_now, r_prev, atr_now) or atr_now <= 0:
            return None
        prev, last = bars.iloc[-2], bars.iloc[-1]
        entry = self._reference_px(mid, bars)

        for side in (Side.LONG, Side.SHORT):
            zone = zones.get(side)
            if zone is None or zone.side is not side or zone.coin != coin:
                continue
            if side is Side.LONG:
                swept = prev["low"] <= zone.high and last["close"] > zone.low
                momentum = r_now < self.rsi_long_max and r_now > r_prev
                sl = zone.low - self.sl_atr_mult * atr_now
            else:
                swept = prev["high"] >= zone.low and last["close"] < zone.high
                momentum = r_now > self.rsi_short_min and r_now < r_prev
                sl = zone.high + self.sl_atr_mult * atr_now
            if not (swept and momentum):
                continue
            bracket = _bracket(side, entry, sl, self.settings.default_tp_r)
            if bracket is None:
                continue
            word = "bajo" if side is Side.LONG else "sobre"
            return Signal(
                coin=coin, side=side, entry_mode=EntryMode.MARKET, strategy=self.name,
                reason=(f"barrido de liquidez {word} el precio en {zone.low:.6g}-{zone.high:.6g} "
                        f"({zone.source}, ${zone.usd:,.0f}) y recuperación; RSI {r_prev:.1f}->{r_now:.1f}"),
                entry_px=entry, sl_price=bracket[0], tp_price=bracket[1],
                confidence=0.7 if zone.source in ("liq", "mixed") else 0.55,
            )
        return None


class SqueezeBreakoutStrategy(Strategy):
    """Ruptura al final de una compresión de volatilidad (velas de 1h).

    Squeeze = Bollinger(20, 2) dentro de Keltner(20, 1.5·ATR). Señal cuando la
    vela anterior estaba en squeeze y la última ya no, ADX(14) > 25 y la última
    cierra fuera de Bollinger: por encima -> LONG, por debajo -> SHORT.
    SL a 1.5·ATR(14) de la entrada.
    """

    name = "squeeze_breakout"
    interval = "1h"
    lookback = 150
    min_bars = 50
    bb_n = 20
    bb_k = 2.0
    kc_n = 20
    kc_mult = 1.5
    adx_period = 14
    adx_min = 25.0
    atr_period = 14
    sl_atr_mult = 1.5

    def _evaluate_closed(self, coin, bars, mid, zones):
        h, l, c = bars["high"], bars["low"], bars["close"]
        sq = squeeze(h, l, c, self.bb_n, self.bb_k, self.kc_n, self.kc_mult)
        if not (bool(sq.iloc[-2]) and not bool(sq.iloc[-1])):
            return None
        adx_now = float(adx(h, l, c, self.adx_period).iloc[-1])
        if not _finite(adx_now) or adx_now <= self.adx_min:
            return None
        bb_up, _, bb_lo = bollinger(c, self.bb_n, self.bb_k)
        close, up, lo = float(c.iloc[-1]), float(bb_up.iloc[-1]), float(bb_lo.iloc[-1])
        if close > up:
            side = Side.LONG
        elif close < lo:
            side = Side.SHORT
        else:
            return None
        atr_now = float(atr(h, l, c, self.atr_period).iloc[-1])
        if not _finite(atr_now) or atr_now <= 0:
            return None
        entry = self._reference_px(mid, bars)
        sl = entry - self.sl_atr_mult * atr_now if side is Side.LONG else entry + self.sl_atr_mult * atr_now
        bracket = _bracket(side, entry, sl, self.settings.default_tp_r)
        if bracket is None:
            return None
        band = f"> BB sup {up:.6g}" if side is Side.LONG else f"< BB inf {lo:.6g}"
        return Signal(
            coin=coin, side=side, entry_mode=EntryMode.MARKET, strategy=self.name,
            reason=f"fin de squeeze {self.interval} y ruptura: cierre {close:.6g} {band}, ADX {adx_now:.1f}",
            entry_px=entry, sl_price=bracket[0], tp_price=bracket[1],
            confidence=min(0.9, 0.5 + (adx_now - self.adx_min) / 100.0),
        )


STRATEGIES: dict[str, type[Strategy]] = {
    LiquiditySweepStrategy.name: LiquiditySweepStrategy,
    SqueezeBreakoutStrategy.name: SqueezeBreakoutStrategy,
}


def build_strategies(settings: Settings, clock: Optional[Clock] = None) -> list[Strategy]:
    """Instancia las estrategias de ``settings.strategies`` (sin duplicados, en orden).

    Con ``clock`` las estrategias saben exactamente si la última vela está en curso.
    Un nombre desconocido es un error de configuración: lanza ``ValueError`` al
    arrancar en vez de operar en silencio sin la estrategia pedida.
    """
    out: list[Strategy] = []
    seen: set[str] = set()
    for raw in settings.strategies:
        key = raw.strip().lower().replace("-", "_")
        if not key or key in seen:
            continue
        cls = STRATEGIES.get(key)
        if cls is None:
            raise ValueError(f"Estrategia desconocida: {raw!r}. Disponibles: {', '.join(sorted(STRATEGIES))}")
        seen.add(key)
        out.append(cls(settings, clock))
    return out

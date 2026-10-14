"""Estrategias autónomas: leen velas (+ zonas de liquidez) y proponen ``Signal``.

Una sola lógica para vivo y backtest:

- ``Strategy.signals(bars, zones)`` es **la** definición de cada estrategia: calcula, vectorizado
  sobre todas las velas, la señal de cada vela cerrada (columnas ``SIGNAL_COLUMNS``). La fila ``i``
  sólo usa datos de las filas ``<= i`` (sin look-ahead): los indicadores son causales y las zonas
  por vela (``liquidity.historical_zones`` en backtest) también.
- En vivo, ``evaluate`` -> ``_evaluate_closed`` = **última fila** de ``signals`` sobre las velas
  cerradas, con la última fila de zonas tomada del ``LiquidityScanner`` (``liquidity.zones_frame``) y
  el mid como precio de referencia de esa fila (``ref_px``). Con ``mid`` = último cierre (o sin mid)
  la señal en vivo es exactamente la fila del backtest.
- Matiz honesto: las medias recursivas (EMA, Wilder: RSI/ATR/ADX) dependen de dónde empiezan. En vivo
  se calculan sobre ``lookback`` velas y en backtest sobre toda la historia; ``lookback`` deja varias
  veces el periodo de calentamiento para que la diferencia sea despreciable, pero no es cero.

Reglas comunes (las aplica ``Strategy.evaluate`` antes de calcular la señal):

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
  TP = ``tp_r`` veces el riesgo; con ``tp_r=None`` (por defecto) se usa ``default_tp_r``,
  leído en cada evaluación para respetar cambios en caliente (``tp 3``). ``rsi_reversion``
  apunta a la media de Bollinger en vez de a un múltiplo de R.
- Entrada a mercado: ``Signal.entry_px`` es la referencia (el mid actual, o el
  último cierre si no hay mid válido); el SL/TP salen de esa referencia.

Parámetros: cada estrategia los declara como atributos de clase (``PARAMS``) y ``PARAM_GRID``
dice qué valores probar al optimizar. ``Strategy(settings, clock=None, *, interval=None, **params)``
fija temporalidad y parámetros (nombre desconocido o valor inválido -> ``ValueError``);
``params()`` devuelve los valores actuales. ``min_bars``/``lookback`` crecen con los periodos.
"""

from __future__ import annotations

import logging
import math
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np
import pandas as pd

from .clock import Clock
from .config import Settings
from .exchange.base import CANDLE_COLUMNS
from .exchange.intervals import interval_seconds
from .indicators import adx, atr, bollinger, ema, rsi, squeeze
from .liquidity import HIST_SOURCE, zones_frame
from .models import EntryMode, Side, Signal, Zone

log = logging.getLogger(__name__)

Zones = Mapping[Side, Optional[Zone]]

# Columnas de ``Strategy.signals``. ``side``: "long" | "short" | "" (sin señal); ``entry_mode``:
# "market" | "zone" | "limit" ("" sin señal); ``confidence`` (0..1) es informativa.
SIGNAL_COLUMNS = ["side", "entry_mode", "entry_px", "sl", "tp", "reason", "confidence"]

_OHLC = ["open", "high", "low", "close"]


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


def empty_signals(index: pd.Index) -> pd.DataFrame:
    """DataFrame de señales sin ninguna señal, con el índice dado."""
    n = len(index)
    return pd.DataFrame({
        "side": np.full(n, "", dtype=object), "entry_mode": np.full(n, "", dtype=object),
        "entry_px": np.full(n, np.nan), "sl": np.full(n, np.nan), "tp": np.full(n, np.nan),
        "reason": np.full(n, "", dtype=object), "confidence": np.full(n, np.nan),
    }, index=index)


# --- utilidades vectorizadas --------------------------------------------------------

def _shift(a: np.ndarray) -> np.ndarray:
    """``a`` desplazado una vela (la fila i ve el valor de i-1; la primera queda NaN)."""
    out = np.empty(len(a), dtype=float)
    if len(a):
        out[0] = np.nan
        out[1:] = a[:-1]
    return out


def _r_bracket(side: Side, entry: np.ndarray, sl: np.ndarray, tp_r: float) -> tuple[np.ndarray, np.ndarray]:
    """(válida, TP) con TP a ``tp_r`` R. Inválida si el SL no queda del lado correcto o el TP <= 0."""
    with np.errstate(invalid="ignore"):
        risk = entry - sl if side is Side.LONG else sl - entry
        tp = entry + tp_r * risk if side is Side.LONG else entry - tp_r * risk
        ok = np.isfinite(risk) & (risk > 0) & np.isfinite(tp) & (tp > 0) & (tp_r > 0)
    return ok, tp


def _target_bracket(side: Side, entry: np.ndarray, sl: np.ndarray, tp: np.ndarray,
                    min_rr: float) -> np.ndarray:
    """Válida si SL < entrada < TP (LONG; espejo en SHORT) y beneficio/riesgo >= ``min_rr``."""
    with np.errstate(invalid="ignore", divide="ignore"):
        risk = entry - sl if side is Side.LONG else sl - entry
        reward = tp - entry if side is Side.LONG else entry - tp
        ok = np.isfinite(risk) & np.isfinite(reward) & (risk > 0) & (reward > 0) & (tp > 0)
        ok &= reward / risk >= min_rr
    return ok


def _zone_arrays(zones: Optional[pd.DataFrame], index: pd.Index, side: Side
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(low, high, usd, source) de las zonas de ``side`` alineadas con ``index`` (NaN / "" sin zona)."""
    n = len(index)
    nan = np.full(n, np.nan)
    if zones is None or len(zones) == 0:
        return nan, nan, nan, np.full(n, "", dtype=object)
    if not zones.index.equals(index):
        zones = zones.reindex(index)
    p = side.value

    def col(name: str) -> np.ndarray:
        return zones[name].to_numpy(dtype=float) if name in zones.columns else nan

    src = zones[f"{p}_source"] if f"{p}_source" in zones.columns else None
    src = (src.where(src.notna(), "").astype(str).to_numpy(dtype=object) if src is not None
           else np.full(n, "", dtype=object))
    return col(f"{p}_low"), col(f"{p}_high"), col(f"{p}_usd"), src


@dataclass
class _Proposal:
    """Candidatas de un lado: máscara, niveles y cómo explicar cada fila (sólo se formatean las que salen)."""

    side: Side
    mask: np.ndarray
    sl: np.ndarray
    tp: np.ndarray
    reason: Callable[[int], str]
    confidence: Union[np.ndarray, float] = 0.55
    entry: Optional[np.ndarray] = None          # None = la referencia (cierre / mid)
    mode: EntryMode = EntryMode.MARKET


def _coerce_param(strategy: str, key: str, value: Any, default: Any) -> Any:
    """Convierte ``value`` al tipo del valor por defecto del parámetro (``ValueError`` si no encaja)."""
    try:
        if value is None:
            if default is None:
                return None
            raise ValueError("no puede ser None")
        if isinstance(value, bool) and not isinstance(default, bool):
            raise ValueError("no es un número")
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, int):
            f = float(value)
            if not f.is_integer():
                raise ValueError("debe ser entero")
            return int(f)
        if isinstance(default, float) or default is None:
            f = float(value)
            if not math.isfinite(f):
                raise ValueError("no es finito")
            return f
        return type(default)(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{strategy}: valor inválido para {key}: {value!r} ({e})") from None


class Strategy(ABC):
    """Base de las estrategias. Las subclases implementan ``_proposals`` (la lógica vectorizada).

    ``lookback`` es cuántas velas debe pedir la mesa (incluida la posible vela en
    curso); ``min_bars`` cuántas velas *cerradas* hacen falta para calentar los
    indicadores. Ambos se recalculan en el constructor según los parámetros.
    """

    name: str = "base"
    interval: str = "15m"
    lookback: int = 100
    min_bars: int = 30
    candle_max_lag_s: float = 120.0
    uses_zones: bool = False                    # True: necesita zonas de liquidez (en backtest, históricas)
    PARAMS: tuple[str, ...] = ()                # parámetros ajustables (atributos de clase con su valor por defecto)
    PARAM_GRID: dict[str, list] = {}            # valores a probar al optimizar (subconjunto de PARAMS)

    def __init__(self, settings: Settings, clock: Optional[Clock] = None, *, interval: Optional[str] = None,
                 **params: Any):
        self.settings = settings
        self.clock = clock
        if interval is not None:
            interval_seconds(interval)          # ValueError si no es un intervalo conocido
            self.interval = interval
        unknown = sorted(set(params) - set(self.PARAMS))
        if unknown:
            known = ", ".join(self.PARAMS) or "ninguno"
            raise ValueError(f"{self.name}: parámetro desconocido {', '.join(unknown)} (admite: {known})")
        for key, value in params.items():
            setattr(self, key, _coerce_param(self.name, key, value, getattr(type(self), key)))
        self._check_params()
        self.min_bars, self.lookback = self._warmup()
        self._last_signal_bar: dict[str, pd.Timestamp] = {}

    # ------------------------------------------------------------------ parámetros
    def params(self) -> dict[str, Any]:
        """Valores actuales de los parámetros ajustables."""
        return {k: getattr(self, k) for k in self.PARAMS}

    def _check_params(self) -> None:
        """Validación semántica de los parámetros (las subclases añaden la suya). ``ValueError`` si no valen."""
        tp_r = getattr(self, "tp_r", None)
        if tp_r is not None and tp_r <= 0:
            raise ValueError(f"{self.name}: tp_r debe ser > 0 (o None para usar default_tp_r)")

    def _require(self, ok: bool, msg: str) -> None:
        if not ok:
            raise ValueError(f"{self.name}: {msg}")

    def _warmup(self) -> tuple[int, int]:
        """(min_bars, lookback) para los parámetros actuales. Por defecto, los de la clase."""
        cls = type(self)
        return cls.min_bars, cls.lookback

    def _warmup_for(self, need: int, recursive: int) -> tuple[int, int]:
        """``need`` velas mínimas; ``lookback`` deja 4 periodos recursivos más para que converjan."""
        cls = type(self)
        min_bars = max(cls.min_bars, int(need))
        return min_bars, max(cls.lookback, min_bars + 4 * int(recursive))

    def _tp_r(self) -> float:
        tp_r = getattr(self, "tp_r", None)
        return float(self.settings.default_tp_r if tp_r is None else tp_r)

    # ------------------------------------------------------------------ vivo
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
        if not bars[_OHLC].iloc[-2:].notna().all().all():
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

    def _evaluate_closed(self, coin: str, bars: pd.DataFrame, mid: Optional[float],
                         zones: Zones) -> Optional[Signal]:
        """Última fila de ``signals`` (``bars`` son sólo velas cerradas, al menos ``min_bars``).

        Las zonas del escáner van a la última fila (``zones_frame``) y el mid es la referencia de
        esa fila: misma lógica, mismos números que el backtest en esa vela.
        """
        zdf = zones_frame(bars.index, zones, coin) if self.uses_zones else None
        ref = float(mid) if mid is not None and math.isfinite(mid) and mid > 0 else None
        return self.signal_at(self.signals(bars, zdf, ref_px=ref), -1, coin)

    def signal_at(self, signals: pd.DataFrame, pos: int, coin: str) -> Optional[Signal]:
        """Fila ``pos`` de ``signals`` como ``Signal`` (None si esa vela no tiene señal)."""
        if signals is None or len(signals) == 0:
            return None
        row = signals.iloc[pos]
        if row["side"] not in ("long", "short"):
            return None
        conf = float(row["confidence"])
        return Signal(coin=coin, side=Side(row["side"]), entry_mode=EntryMode(row["entry_mode"]),
                      strategy=self.name, reason=str(row["reason"]), entry_px=float(row["entry_px"]),
                      sl_price=float(row["sl"]), tp_price=float(row["tp"]),
                      confidence=conf if math.isfinite(conf) else 0.5)

    # ------------------------------------------------------------------ vectorizado
    def signals(self, bars: pd.DataFrame, zones: Optional[pd.DataFrame] = None, *,
                ref_px: Optional[float] = None) -> pd.DataFrame:
        """Señal de cada vela (cerrada) de ``bars``: DataFrame con su índice y ``SIGNAL_COLUMNS``.

        - La fila ``i`` sólo depende de ``bars``/``zones`` hasta ``i`` (sin look-ahead).
        - ``zones``: DataFrame por vela con ``liquidity.ZONE_COLUMNS`` (NaN = sin zona); sólo lo usan
          las estrategias con ``uses_zones``. Sin zonas, esas estrategias no dan señales.
        - ``ref_px``: precio de referencia de la **última** fila (en vivo, el mid); el resto usa el
          cierre de su vela. Las entradas a mercado se anclan en esa referencia.
        - Filas sin señal: ``side == ""``. Las primeras ``min_bars - 1`` filas y las que tienen
          OHLC inválido en la vela o en la anterior nunca dan señal (igual que ``evaluate``).
        """
        if bars is None:
            return empty_signals(pd.Index([]))
        out = empty_signals(bars.index)
        n = len(bars)
        if n == 0 or any(c not in bars.columns for c in CANDLE_COLUMNS):
            return out
        ref = bars["close"].to_numpy(dtype=float, copy=True)
        if ref_px is not None and math.isfinite(ref_px) and ref_px > 0:
            ref[-1] = float(ref_px)
        ohlc_ok = np.isfinite(bars[_OHLC].to_numpy(dtype=float)).all(axis=1)
        free = (np.arange(n) >= self.min_bars - 1) & ohlc_ok
        free[1:] &= ohlc_ok[:-1]
        free[0] = False
        cols = {c: out[c].to_numpy(copy=True) for c in SIGNAL_COLUMNS}
        for prop in self._proposals(bars, zones, ref):
            take = np.flatnonzero(free & prop.mask)
            if len(take) == 0:
                continue
            free[take] = False                  # prioridad: la primera propuesta que encaja gana la vela
            entry = ref if prop.entry is None else prop.entry
            conf = np.broadcast_to(np.asarray(prop.confidence, dtype=float), (n,))
            cols["side"][take] = prop.side.value
            cols["entry_mode"][take] = prop.mode.value
            cols["entry_px"][take] = entry[take]
            cols["sl"][take] = prop.sl[take]
            cols["tp"][take] = prop.tp[take]
            cols["confidence"][take] = conf[take]
            for i in take.tolist():
                cols["reason"][i] = prop.reason(i)
        return pd.DataFrame(cols, index=bars.index)

    def _proposals(self, bars: pd.DataFrame, zones: Optional[pd.DataFrame], ref: np.ndarray) -> list[_Proposal]:
        """Candidatas por lado, en orden de prioridad. ``ref`` = cierre de cada vela (o el mid en la última)."""
        raise NotImplementedError(f"{type(self).__name__} no implementa signals()")


class LiquiditySweepStrategy(Strategy):
    """Reversión tras barrer una zona de liquidez (velas de 15m).

    LONG: la vela anterior perfora la zona LONG (``low <= zone.high``), la última
    cierra por encima de ``zone.low`` (recupera) y el RSI(14) está por debajo de
    40 y subiendo. SL bajo ``zone.low - 0.5·ATR(14)``. SHORT es el espejo: la
    vela anterior entra en la zona SHORT (``high >= zone.low``), la última cierra
    por debajo de ``zone.high`` y el RSI está por encima de 60 y bajando.

    En vivo la zona es la del escáner (libro + liquidaciones); en backtest, la aproximación de
    ``liquidity.historical_zones`` (pivotes): sus resultados históricos son sólo orientativos.
    """

    name = "liquidity_sweep"
    interval = "15m"
    lookback = 120
    min_bars = 30
    uses_zones = True
    rsi_period = 14
    atr_period = 14
    rsi_long_max = 40.0
    rsi_short_min = 60.0
    sl_atr_mult = 0.5
    tp_r: Optional[float] = None
    PARAMS = ("rsi_period", "atr_period", "rsi_long_max", "rsi_short_min", "sl_atr_mult", "tp_r")
    PARAM_GRID = {
        "rsi_period": [9, 14],
        "rsi_long_max": [30.0, 35.0, 40.0, 45.0],
        "rsi_short_min": [55.0, 60.0, 65.0, 70.0],
        "sl_atr_mult": [0.25, 0.5, 1.0],
        "tp_r": [1.5, 2.0, 3.0],
    }

    def _check_params(self) -> None:
        super()._check_params()
        self._require(self.rsi_period >= 2 and self.atr_period >= 1, "periodos demasiado cortos")
        self._require(0 < self.rsi_long_max < 100 and 0 < self.rsi_short_min < 100, "umbrales de RSI fuera de 0-100")
        self._require(self.sl_atr_mult >= 0, "sl_atr_mult no puede ser negativo")

    def _warmup(self) -> tuple[int, int]:
        p = max(self.rsi_period, self.atr_period)
        return self._warmup_for(2 * p + 2, p)

    def _proposals(self, bars, zones, ref):
        if zones is None:
            return []
        c_s = bars["close"]
        h, l, c = (bars[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
        r = rsi(c_s, self.rsi_period).to_numpy()
        a = atr(bars["high"], bars["low"], c_s, self.atr_period).to_numpy()
        r_prev = _shift(r)
        prev_low, prev_high = _shift(l), _shift(h)
        tp_r = self._tp_r()
        with np.errstate(invalid="ignore"):
            base = np.isfinite(r) & np.isfinite(r_prev) & np.isfinite(a) & (a > 0)
        out = []
        for side in (Side.LONG, Side.SHORT):
            z_low, z_high, z_usd, z_src = _zone_arrays(zones, bars.index, side)
            with np.errstate(invalid="ignore"):
                if side is Side.LONG:
                    swept = (prev_low <= z_high) & (c > z_low)
                    momentum = (r < self.rsi_long_max) & (r > r_prev)
                    sl = z_low - self.sl_atr_mult * a
                else:
                    swept = (prev_high >= z_low) & (c < z_high)
                    momentum = (r > self.rsi_short_min) & (r < r_prev)
                    sl = z_high + self.sl_atr_mult * a
            ok, tp = _r_bracket(side, ref, sl, tp_r)
            word = "bajo" if side is Side.LONG else "sobre"

            def reason(i: int, word=word, z_low=z_low, z_high=z_high, z_usd=z_usd, z_src=z_src) -> str:
                src = z_src[i] or HIST_SOURCE      # sin fuente = zonas históricas (``historical_zones``)
                return (f"barrido de liquidez {word} el precio en {z_low[i]:.6g}-{z_high[i]:.6g} "
                        f"({src}, ${z_usd[i]:,.0f}) y recuperación; RSI {r_prev[i]:.1f}->{r[i]:.1f}")

            conf = np.where(np.isin(z_src, ["liq", "mixed"]), 0.7, 0.55)
            out.append(_Proposal(side, base & swept & momentum & ok, sl, tp, reason, conf))
        return out


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
    tp_r: Optional[float] = None
    PARAMS = ("bb_n", "bb_k", "kc_n", "kc_mult", "adx_period", "adx_min", "atr_period", "sl_atr_mult", "tp_r")
    PARAM_GRID = {
        "bb_k": [1.8, 2.0, 2.2],
        "kc_mult": [1.25, 1.5, 2.0],
        "adx_min": [15.0, 20.0, 25.0, 30.0],
        "sl_atr_mult": [1.0, 1.5, 2.0],
        "tp_r": [1.5, 2.0, 3.0],
    }

    def _check_params(self) -> None:
        super()._check_params()
        self._require(self.bb_n >= 2 and self.kc_n >= 2 and self.adx_period >= 2 and self.atr_period >= 1,
                      "periodos demasiado cortos")
        self._require(self.bb_k > 0 and self.kc_mult > 0 and self.sl_atr_mult > 0,
                      "bb_k, kc_mult y sl_atr_mult deben ser > 0")

    def _warmup(self) -> tuple[int, int]:
        need = max(self.bb_n, self.kc_n, 2 * self.adx_period, self.atr_period) + 2
        return self._warmup_for(need, max(self.kc_n, self.adx_period, self.atr_period))

    def _proposals(self, bars, zones, ref):
        h_s, l_s, c_s = bars["high"], bars["low"], bars["close"]
        c = c_s.to_numpy(dtype=float)
        sq = squeeze(h_s, l_s, c_s, self.bb_n, self.bb_k, self.kc_n, self.kc_mult).to_numpy(dtype=bool)
        sq_prev = np.r_[False, sq[:-1]]
        adx_v = adx(h_s, l_s, c_s, self.adx_period).to_numpy()
        bb_up, _, bb_lo = (s.to_numpy() for s in bollinger(c_s, self.bb_n, self.bb_k))
        a = atr(h_s, l_s, c_s, self.atr_period).to_numpy()
        tp_r = self._tp_r()
        with np.errstate(invalid="ignore"):
            base = sq_prev & ~sq & (adx_v > self.adx_min) & np.isfinite(a) & (a > 0)
            conf = np.minimum(0.9, 0.5 + (adx_v - self.adx_min) / 100.0)
        out = []
        for side in (Side.LONG, Side.SHORT):
            with np.errstate(invalid="ignore"):
                broke = c > bb_up if side is Side.LONG else c < bb_lo
                sl = ref - self.sl_atr_mult * a if side is Side.LONG else ref + self.sl_atr_mult * a
            ok, tp = _r_bracket(side, ref, sl, tp_r)

            def reason(i: int, side=side) -> str:
                band = f"> BB sup {bb_up[i]:.6g}" if side is Side.LONG else f"< BB inf {bb_lo[i]:.6g}"
                return f"fin de squeeze {self.interval} y ruptura: cierre {c[i]:.6g} {band}, ADX {adx_v[i]:.1f}"

            out.append(_Proposal(side, base & broke & ok, sl, tp, reason, conf))
        return out


class EmaTrendStrategy(Strategy):
    """Seguimiento de tendencia: cruce de EMA rápida/lenta con filtro de ADX (velas de 1h).

    LONG cuando la EMA rápida cruza por encima de la lenta en la vela (antes estaba en o por
    debajo) y ADX > ``adx_min`` (hay tendencia, no rango). SHORT es el espejo. SL a
    ``sl_atr_mult``·ATR de la entrada, TP a ``tp_r`` R.
    """

    name = "ema_trend"
    interval = "1h"
    lookback = 200
    min_bars = 60
    fast = 20
    slow = 50
    adx_period = 14
    adx_min = 20.0
    atr_period = 14
    sl_atr_mult = 2.0
    tp_r: Optional[float] = None
    PARAMS = ("fast", "slow", "adx_period", "adx_min", "atr_period", "sl_atr_mult", "tp_r")
    PARAM_GRID = {
        "fast": [9, 12, 20, 30],
        "slow": [50, 100, 200],
        "adx_min": [15.0, 20.0, 25.0, 30.0],
        "sl_atr_mult": [1.5, 2.0, 3.0],
        "tp_r": [1.5, 2.0, 3.0],
    }

    def _check_params(self) -> None:
        super()._check_params()
        self._require(2 <= self.fast < self.slow, "hace falta 2 <= fast < slow")
        self._require(self.adx_period >= 2 and self.atr_period >= 1, "periodos demasiado cortos")
        self._require(self.sl_atr_mult > 0, "sl_atr_mult debe ser > 0")

    def _warmup(self) -> tuple[int, int]:
        need = max(self.slow, 2 * self.adx_period, self.atr_period) + 2
        return self._warmup_for(need, max(self.slow, self.adx_period, self.atr_period))

    def _proposals(self, bars, zones, ref):
        h_s, l_s, c_s = bars["high"], bars["low"], bars["close"]
        f = ema(c_s, self.fast).to_numpy()
        s = ema(c_s, self.slow).to_numpy()
        f_prev, s_prev = _shift(f), _shift(s)
        adx_v = adx(h_s, l_s, c_s, self.adx_period).to_numpy()
        a = atr(h_s, l_s, c_s, self.atr_period).to_numpy()
        tp_r = self._tp_r()
        with np.errstate(invalid="ignore"):
            base = (adx_v > self.adx_min) & np.isfinite(a) & (a > 0)
            conf = np.minimum(0.85, 0.5 + (adx_v - self.adx_min) / 100.0)
        out = []
        for side in (Side.LONG, Side.SHORT):
            with np.errstate(invalid="ignore"):
                if side is Side.LONG:
                    cross = (f > s) & (f_prev <= s_prev)
                    sl = ref - self.sl_atr_mult * a
                else:
                    cross = (f < s) & (f_prev >= s_prev)
                    sl = ref + self.sl_atr_mult * a
            ok, tp = _r_bracket(side, ref, sl, tp_r)

            def reason(i: int, side=side) -> str:
                word, op = ("alcista", ">") if side is Side.LONG else ("bajista", "<")
                return (f"cruce {word} EMA{self.fast}/EMA{self.slow} ({f[i]:.6g} {op} {s[i]:.6g}) "
                        f"en {self.interval}, ADX {adx_v[i]:.1f}")

            out.append(_Proposal(side, base & cross & ok, sl, tp, reason, conf))
        return out


class RsiReversionStrategy(Strategy):
    """Reversión a la media: RSI extremo + toque de Bollinger, objetivo la media (velas de 15m).

    LONG cuando RSI < ``rsi_low`` y el mínimo de la vela toca la banda inferior; TP en la media
    de Bollinger (SMA), SL a ``sl_atr_mult``·ATR bajo la entrada. Se descarta si la media no
    deja al menos ``min_rr`` R de recorrido (no merece la pena con comisiones). SHORT espejo.
    """

    name = "rsi_reversion"
    interval = "15m"
    lookback = 120
    min_bars = 40
    rsi_period = 14
    rsi_low = 30.0
    rsi_high = 70.0
    bb_n = 20
    bb_k = 2.0
    atr_period = 14
    sl_atr_mult = 1.5
    min_rr = 0.8
    PARAMS = ("rsi_period", "rsi_low", "rsi_high", "bb_n", "bb_k", "atr_period", "sl_atr_mult", "min_rr")
    PARAM_GRID = {
        "rsi_period": [7, 14],
        "rsi_low": [20.0, 25.0, 30.0],
        "rsi_high": [70.0, 75.0, 80.0],
        "bb_k": [2.0, 2.5],
        "sl_atr_mult": [1.0, 1.5, 2.0],
    }

    def _check_params(self) -> None:
        super()._check_params()
        self._require(self.rsi_period >= 2 and self.bb_n >= 2 and self.atr_period >= 1, "periodos demasiado cortos")
        self._require(0 < self.rsi_low < self.rsi_high < 100, "hace falta 0 < rsi_low < rsi_high < 100")
        self._require(self.bb_k > 0 and self.sl_atr_mult > 0 and self.min_rr >= 0,
                      "bb_k y sl_atr_mult deben ser > 0 y min_rr >= 0")

    def _warmup(self) -> tuple[int, int]:
        p = max(self.rsi_period, self.atr_period)
        return self._warmup_for(max(self.bb_n, 2 * p) + 2, p)

    def _proposals(self, bars, zones, ref):
        h_s, l_s, c_s = bars["high"], bars["low"], bars["close"]
        h, l = h_s.to_numpy(dtype=float), l_s.to_numpy(dtype=float)
        r = rsi(c_s, self.rsi_period).to_numpy()
        up, mid, lo = (s.to_numpy() for s in bollinger(c_s, self.bb_n, self.bb_k))
        a = atr(h_s, l_s, c_s, self.atr_period).to_numpy()
        with np.errstate(invalid="ignore"):
            base = np.isfinite(a) & (a > 0) & np.isfinite(mid)
        out = []
        for side in (Side.LONG, Side.SHORT):
            with np.errstate(invalid="ignore"):
                if side is Side.LONG:
                    setup = (r < self.rsi_low) & (l <= lo)
                    sl = ref - self.sl_atr_mult * a
                else:
                    setup = (r > self.rsi_high) & (h >= up)
                    sl = ref + self.sl_atr_mult * a
            ok = _target_bracket(side, ref, sl, mid, self.min_rr)

            def reason(i: int, side=side) -> str:
                if side is Side.LONG:
                    return (f"RSI {r[i]:.1f} < {self.rsi_low:g} y mínimo {l[i]:.6g} en la Bollinger inferior "
                            f"{lo[i]:.6g}; objetivo la media {mid[i]:.6g}")
                return (f"RSI {r[i]:.1f} > {self.rsi_high:g} y máximo {h[i]:.6g} en la Bollinger superior "
                        f"{up[i]:.6g}; objetivo la media {mid[i]:.6g}")

            out.append(_Proposal(side, base & setup & ok, sl, mid, reason, 0.55))
        return out


class DonchianBreakoutStrategy(Strategy):
    """Ruptura de canal de Donchian (velas de 4h).

    LONG en la primera vela que cierra por encima del máximo de las ``channel`` velas
    anteriores (la vela previa aún cerraba dentro del canal); SHORT espejo con el mínimo.
    SL a ``sl_atr_mult``·ATR de la entrada, TP a ``tp_r`` R.
    """

    name = "donchian_breakout"
    interval = "4h"
    lookback = 120
    min_bars = 30
    channel = 20
    atr_period = 14
    sl_atr_mult = 2.0
    tp_r: Optional[float] = None
    PARAMS = ("channel", "atr_period", "sl_atr_mult", "tp_r")
    PARAM_GRID = {
        "channel": [10, 20, 30, 55],
        "sl_atr_mult": [1.0, 1.5, 2.0, 3.0],
        "tp_r": [1.5, 2.0, 3.0],
    }

    def _check_params(self) -> None:
        super()._check_params()
        self._require(self.channel >= 2 and self.atr_period >= 1, "periodos demasiado cortos")
        self._require(self.sl_atr_mult > 0, "sl_atr_mult debe ser > 0")

    def _warmup(self) -> tuple[int, int]:
        return self._warmup_for(max(self.channel + 2, 2 * self.atr_period + 2), self.atr_period)

    def _proposals(self, bars, zones, ref):
        h_s, l_s, c_s = bars["high"], bars["low"], bars["close"]
        c = c_s.to_numpy(dtype=float)
        upper = h_s.rolling(self.channel, min_periods=self.channel).max().shift(1).to_numpy()
        lower = l_s.rolling(self.channel, min_periods=self.channel).min().shift(1).to_numpy()
        c_prev, up_prev, lo_prev = _shift(c), _shift(upper), _shift(lower)
        a = atr(h_s, l_s, c_s, self.atr_period).to_numpy()
        tp_r = self._tp_r()
        with np.errstate(invalid="ignore"):
            base = np.isfinite(a) & (a > 0)
        out = []
        for side in (Side.LONG, Side.SHORT):
            with np.errstate(invalid="ignore"):
                if side is Side.LONG:
                    brk = (c > upper) & (c_prev <= up_prev)
                    sl = ref - self.sl_atr_mult * a
                else:
                    brk = (c < lower) & (c_prev >= lo_prev)
                    sl = ref + self.sl_atr_mult * a
            ok, tp = _r_bracket(side, ref, sl, tp_r)

            def reason(i: int, side=side) -> str:
                if side is Side.LONG:
                    return f"ruptura del canal Donchian {self.channel}: cierre {c[i]:.6g} > máximo {upper[i]:.6g}"
                return f"ruptura del canal Donchian {self.channel}: cierre {c[i]:.6g} < mínimo {lower[i]:.6g}"

            out.append(_Proposal(side, base & brk & ok, sl, tp, reason, 0.55))
        return out


STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (LiquiditySweepStrategy, SqueezeBreakoutStrategy, EmaTrendStrategy,
                              RsiReversionStrategy, DonchianBreakoutStrategy)
}


def _key(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def create_strategy(name: str, settings: Settings, clock: Optional[Clock] = None, *,
                    interval: Optional[str] = None, **params: Any) -> Strategy:
    """Instancia una estrategia por nombre (normalizado) con temporalidad y parámetros opcionales.

    ``ValueError`` si el nombre, el intervalo o algún parámetro no son válidos.
    """
    cls = STRATEGIES.get(_key(name))
    if cls is None:
        raise ValueError(f"Estrategia desconocida: {name!r}. Disponibles: {', '.join(sorted(STRATEGIES))}")
    return cls(settings, clock, interval=interval, **params)


def build_strategies(settings: Settings, clock: Optional[Clock] = None) -> list[Strategy]:
    """Instancia las estrategias de ``settings.strategies`` (sin duplicados, en orden).

    Con ``clock`` las estrategias saben exactamente si la última vela está en curso.
    Un nombre desconocido es un error de configuración: lanza ``ValueError`` al
    arrancar en vez de operar en silencio sin la estrategia pedida.
    """
    out: list[Strategy] = []
    seen: set[str] = set()
    for raw in settings.strategies:
        key = _key(raw)
        if not key or key in seen:
            continue
        cls = STRATEGIES.get(key)
        if cls is None:
            raise ValueError(f"Estrategia desconocida: {raw!r}. Disponibles: {', '.join(sorted(STRATEGIES))}")
        seen.add(key)
        out.append(cls(settings, clock))
    return out

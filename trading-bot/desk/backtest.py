"""Backtest por gráfica: una estrategia sobre velas históricas con reglas de ejecución conservadoras.

Un backtest NO garantiza el rendimiento futuro. Sobre la misma historia en la que se eligieron los
parámetros (dentro de muestra) siempre sale mejor de lo que saldrá después: lo que cuenta es el
resultado fuera de muestra (``optimizer``). Estas reglas intentan no regalar nada:

- **Señales**: las de ``Strategy.signals`` (la misma lógica que en vivo), calculadas al cierre de la
  vela ``i``. Sólo se evalúan con la cuenta plana y sin entrada pendiente (en vivo la mesa no evalúa
  una moneda con posición, entrada o bracket). Una posición a la vez.
- **Entrada MARKET**: al open de ``i+1`` ± ``paper_slippage_bps`` en contra, comisión taker. Si a ese
  precio el SL o el TP ya quedan del lado equivocado, la señal se descarta (el riesgo en vivo la
  rechazaría igual).
- **Entrada ZONE/LIMIT**: límite en ``entry_px`` colocado al cierre de ``i``; sólo cuentan las velas que
  terminan antes de ``entry_ttl_minutes`` (una vela que cruza la caducidad no cuenta: podría tocarse
  tarde). Se ejecuta si una vela lo toca (LONG: ``low <= px``), al precio límite, comisión maker. En la
  vela de la ejecución sólo cuenta el SL: no se sabe si el máximo llegó antes que la ejecución.
- **SL/TP**: intravela con ``high``/``low``. Si una vela toca ambos -> **SL primero**. Un hueco al
  abrir a través del SL ejecuta al open (peor que el SL); a través del TP, al TP (sin regalo). Las
  salidas son disparos a mercado: comisión taker y deslizamiento en contra.
- **Fin**: lo que siga abierto se cierra al cierre de la última vela (``exit_reason="end"``).
- **Dimensionamiento** igual que ``RiskManager.evaluate`` en vivo: ``auto_units`` (recortado por
  ``max_units_per_trade``) · ``unit_margin_usd`` · apalancamiento min(settings, meta) (bajado si la
  liquidación llegaría antes que el SL), recortado por ``risk_per_trade_pct`` en el SL (con comisiones),
  ``max_notional_per_coin_usd``, exposición bruta y margen; tamaño redondeado hacia abajo y descartado
  bajo el notional mínimo. Sin ``meta`` no se redondean precios ni (casi) tamaños.
- No se modelan funding, liquidaciones, el spread (sólo el deslizamiento), halts ni cooldowns del riesgo.

Rendimiento: 5000 velas con un juego de parámetros en < 100 ms (señales vectorizadas + un bucle
simple sobre arrays). Para el walk-forward: ``signals=`` reutiliza señales ya calculadas y
``trade_from=`` usa las velas anteriores sólo para calentar indicadores.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .config import Settings
from .exchange.base import CANDLE_COLUMNS, round_price, round_size
from .exchange.intervals import interval_seconds
from .liquidity import historical_zones
from .models import AssetMeta
from .risk import LIQ_BUFFER, estimated_liq_px, maintenance_rate
from .strategy import Strategy

log = logging.getLogger(__name__)

EXIT_REASONS = ("sl", "tp", "end", "flip")
METRIC_KEYS = ("total_return_pct", "cagr_pct", "sharpe", "sortino", "max_drawdown_pct", "calmar", "win_rate_pct",
               "profit_factor", "expectancy_r", "n_trades", "avg_trade_pct", "avg_bars_held", "exposure_pct",
               "best_trade_pct", "worst_trade_pct", "fees_usd")
PF_CAP = 999.0          # profit factor sin pérdidas (sería infinito): tope para que sea JSON y comparable
EQUITY_POINTS = 500     # puntos de la curva de capital en ``to_dict``
YEAR_S = 365.25 * 86_400.0


@dataclass
class BTTrade:
    """Una operación del backtest. Horas = apertura (etiqueta) de la vela en la que ocurre, en segundos UNIX."""

    coin: str
    side: str                 # "long" | "short"
    entry_ts: float
    entry_px: float           # precio de ejecución (con deslizamiento si fue a mercado)
    exit_ts: float
    exit_px: float            # precio de ejecución de la salida (con deslizamiento)
    size: float               # monedas
    pnl: float                # USD netos (bruto − comisiones de entrada y salida)
    pnl_pct: float            # pnl / equity al entrar · 100 (impacto en la cuenta)
    r_multiple: float         # pnl / riesgo inicial (size · |entrada − SL|)
    fees: float               # USD
    exit_reason: str          # "sl" | "tp" | "end" | "flip"
    bars_held: int            # velas con la posición abierta (la de entrada y la de salida cuentan)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BTTrade":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


@dataclass
class BacktestResult:
    coin: str
    interval: str
    strategy: str
    params: dict[str, Any]
    start_ts: float           # apertura de la primera vela operada
    end_ts: float             # cierre de la última vela
    bars: int                 # velas operadas (sin las de calentamiento)
    trades: list[BTTrade]
    equity: pd.Series         # capital marcado a mercado al cierre de cada vela (índice = apertura de la vela)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, max_points: int = EQUITY_POINTS) -> dict[str, Any]:
        """Serializable a JSON. La curva de capital se muestrea a ``max_points`` puntos como mucho
        (siempre incluye el primero y el último) como ``[[ts, equity], ...]``."""
        eq = self.equity
        n = len(eq)
        if n > max_points > 1:
            pos = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
        else:
            pos = np.arange(n)
        ts = _index_seconds(eq.index)
        values = eq.to_numpy(dtype=float)
        return {
            "coin": self.coin, "interval": self.interval, "strategy": self.strategy, "params": dict(self.params),
            "start_ts": self.start_ts, "end_ts": self.end_ts, "bars": self.bars,
            "trades": [t.to_dict() for t in self.trades],
            "equity": [[float(ts[i]), float(values[i])] for i in pos.tolist()],
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BacktestResult":
        pts = d.get("equity") or []
        idx = pd.to_datetime([p[0] for p in pts], unit="s", utc=True)
        equity = pd.Series([float(p[1]) for p in pts], index=idx, dtype=float)
        return cls(coin=d["coin"], interval=d["interval"], strategy=d["strategy"], params=dict(d.get("params") or {}),
                   start_ts=float(d["start_ts"]), end_ts=float(d["end_ts"]), bars=int(d["bars"]),
                   trades=[BTTrade.from_dict(t) for t in d.get("trades") or []], equity=equity,
                   metrics=dict(d.get("metrics") or {}))


def _index_seconds(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return index.as_unit("ns").asi8.astype(np.float64) / 1e9
    return np.asarray(index, dtype=float)


# --- dimensionamiento (mismo cálculo que RiskManager.evaluate) ----------------------

def position_size(settings: Settings, meta: Optional[AssetMeta], equity: float, entry: float, sl: float
                  ) -> tuple[float, int]:
    """(tamaño, apalancamiento) que aprobaría el riesgo en vivo para una entrada automática; (0, 0) si ninguno.

    Replica ``RiskManager.evaluate`` con la cuenta plana (sin otras posiciones ni entradas pendientes):
    ``entry`` es el precio de referencia (ask/bid o el límite), ya redondeado si hay ``meta``.
    """
    s = settings
    if not (math.isfinite(equity) and equity > 0 and math.isfinite(entry) and entry > 0 and math.isfinite(sl)):
        return 0.0, 0
    units = float(s.auto_units)
    if s.max_units_per_trade > 0:
        units = min(units, float(s.max_units_per_trade))
    if units <= 0:
        return 0.0, 0
    is_long = sl < entry
    max_lev = max(int(meta.max_leverage), 1) if meta is not None else max(int(s.leverage), 1)
    lev = max(1, min(int(s.leverage), max_lev))
    mmr = maintenance_rate(s, max_lev)
    sl_dist = abs(entry - sl)
    while lev >= 1 and abs(entry - estimated_liq_px(entry, lev, is_long, mmr)) < sl_dist * LIQ_BUFFER:
        lev -= 1
    if lev < 1:
        return 0.0, 0
    notional = units * s.unit_margin_usd * lev
    taker = max(s.paper_taker_fee_bps, 0.0) / 10_000.0
    loss_per_coin = sl_dist + taker * (entry + sl)
    if loss_per_coin > 0:
        notional = min(notional, equity * s.risk_per_trade_pct / 100.0 / loss_per_coin * entry)
    notional = min(notional, float(s.max_notional_per_coin_usd))
    min_notional = meta.min_notional_usd if meta is not None else 10.0
    if s.max_gross_notional_usd < min_notional:
        return 0.0, 0
    notional = min(notional, float(s.max_gross_notional_usd))
    notional = min(notional, equity / (1.0 / lev + taker))
    size = round_size(notional / entry, meta.sz_decimals if meta is not None else 8)
    if size <= 0 or size * entry < min_notional - 1e-9:
        return 0.0, 0
    return size, lev


# --- métricas -----------------------------------------------------------------------

def compute_metrics(trades: list[BTTrade], equity: pd.Series, start_equity: float, interval: str, *,
                    n_bars: Optional[int] = None, end_ts: Optional[float] = None) -> dict[str, float]:
    """Métricas de un backtest (``METRIC_KEYS``), todas finitas (aptas para JSON).

    - Retornos por vela de la curva de capital (empezando en ``start_equity``); Sharpe y Sortino
      anualizados con las velas de un año 24/7 (cripto), sin tipo libre de riesgo.
    - ``max_drawdown_pct`` positivo (caída máxima desde un máximo, en %); ``calmar`` = CAGR / DD.
    - ``profit_factor`` = ganancias / pérdidas brutas de las operaciones (``PF_CAP`` si no hay pérdidas).
    - ``expectancy_r`` = R medio; ``exposure_pct`` = velas con posición / velas.
    """
    step = interval_seconds(interval)
    eq = equity.to_numpy(dtype=float) if equity is not None else np.array([])
    n = int(n_bars if n_bars is not None else len(eq))
    out: dict[str, float] = {k: 0.0 for k in METRIC_KEYS}
    out["n_trades"] = len(trades)
    final = float(eq[-1]) if len(eq) else float(start_equity)
    if start_equity > 0:
        out["total_return_pct"] = (final / start_equity - 1.0) * 100.0
    if len(eq):
        start_ts = float(_index_seconds(equity.index)[0])
        end = float(end_ts) if end_ts is not None else float(_index_seconds(equity.index)[-1]) + step
        years = (end - start_ts) / YEAR_S
        if years > 0 and start_equity > 0:
            ratio = final / start_equity
            out["cagr_pct"] = (ratio ** (1.0 / years) - 1.0) * 100.0 if ratio > 0 else -100.0
        curve = np.r_[float(start_equity), eq]
        with np.errstate(invalid="ignore", divide="ignore"):
            rets = np.diff(curve) / curve[:-1]
        rets = rets[np.isfinite(rets)]
        per_year = math.sqrt(YEAR_S / step)
        if len(rets) > 1:
            mean, sd = float(rets.mean()), float(rets.std(ddof=1))
            if sd > 0:
                out["sharpe"] = mean / sd * per_year
            downside = math.sqrt(float(np.mean(np.minimum(rets, 0.0) ** 2)))
            if downside > 0:
                out["sortino"] = mean / downside * per_year
        peak = np.maximum.accumulate(curve)
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = np.where(peak > 0, (peak - curve) / peak, 0.0)
        out["max_drawdown_pct"] = float(np.nanmax(dd)) * 100.0
        if out["max_drawdown_pct"] > 0:
            out["calmar"] = out["cagr_pct"] / out["max_drawdown_pct"]
    if trades:
        pnl = np.array([t.pnl for t in trades], dtype=float)
        wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        out["win_rate_pct"] = float((pnl > 0).mean() * 100.0)
        out["profit_factor"] = float(min(wins / losses, PF_CAP)) if losses > 0 else (PF_CAP if wins > 0 else 0.0)
        out["expectancy_r"] = float(np.mean([t.r_multiple for t in trades]))
        pct = np.array([t.pnl_pct for t in trades], dtype=float)
        out["avg_trade_pct"] = float(pct.mean())
        out["best_trade_pct"], out["worst_trade_pct"] = float(pct.max()), float(pct.min())
        held = sum(t.bars_held for t in trades)
        out["avg_bars_held"] = held / len(trades)
        out["exposure_pct"] = min(held / n * 100.0, 100.0) if n > 0 else 0.0
        out["fees_usd"] = float(sum(t.fees for t in trades))
    clean = {k: float(v) if math.isfinite(v) else 0.0 for k, v in out.items()}
    clean["n_trades"] = len(trades)
    return clean


# --- motor --------------------------------------------------------------------------

def run_backtest(strategy: Strategy, bars: pd.DataFrame, settings: Settings, *, coin: str,
                 meta: Optional[AssetMeta] = None, start_equity: Optional[float] = None,
                 zones: Optional[pd.DataFrame] = None, signals: Optional[pd.DataFrame] = None,
                 trade_from: Union[int, pd.Timestamp, None] = None, allow_flip: bool = False) -> BacktestResult:
    """Backtest de ``strategy`` sobre ``bars`` (velas **cerradas**, índice = apertura UTC). Ver el docstring del módulo.

    - ``zones``: zonas por vela para las estrategias de liquidez; sin ellas se usan
      ``liquidity.historical_zones`` (aproximación sin look-ahead).
    - ``signals``: señales ya calculadas con ``strategy.signals`` sobre estas velas (o sobre una
      historia más larga con el mismo inicio: son causales). Evita recalcularlas en cada tramo.
    - ``trade_from``: posición (o marca de tiempo) de la primera vela operada; las anteriores sólo
      calientan indicadores. La curva de capital y las métricas empiezan ahí.
    - ``allow_flip``: con True una señal contraria con posición abierta la cierra al open siguiente
      (``"flip"``) y abre la nueva. Por defecto False, como en vivo (con posición no se evalúa).
    """
    if bars is None or any(c not in bars.columns for c in CANDLE_COLUMNS):
        raise ValueError(f"velas sin columnas {CANDLE_COLUMNS}")
    if len(bars) and not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("las velas deben tener un DatetimeIndex (apertura de cada vela)")
    n = len(bars)
    step = interval_seconds(strategy.interval)
    equity0 = float(settings.paper_start_equity if start_equity is None else start_equity)
    start = _start_pos(bars.index, trade_from)

    if signals is None:
        if strategy.uses_zones and zones is None:
            zones = historical_zones(bars, settings)
        signals = strategy.signals(bars, zones)
    signals = signals.iloc[:n]
    if len(signals) != n or (n and not signals.index.equals(bars.index)):
        raise ValueError("las señales no corresponden a estas velas")

    sim = _Simulator(settings, meta, coin, equity0, step, allow_flip)
    trades, eq = sim.run(bars, signals, start)
    index = bars.index[start:]
    equity = pd.Series(eq, index=index, dtype=float)
    ts = _index_seconds(bars.index)
    start_ts = float(ts[start]) if start < n else (float(ts[-1]) + step if n else 0.0)
    end_ts = float(ts[-1]) + step if n else 0.0
    metrics = compute_metrics(trades, equity, equity0, strategy.interval, n_bars=n - start, end_ts=end_ts)
    metrics["n_signals"] = sim.n_signals
    metrics["n_skipped"] = sim.n_skipped
    return BacktestResult(coin=coin, interval=strategy.interval, strategy=strategy.name, params=strategy.params(),
                          start_ts=start_ts, end_ts=end_ts, bars=n - start, trades=trades, equity=equity,
                          metrics=metrics)


def _start_pos(index: pd.Index, trade_from: Union[int, pd.Timestamp, None]) -> int:
    n = len(index)
    if trade_from is None:
        return 0
    if isinstance(trade_from, (int, np.integer)) and not isinstance(trade_from, bool):
        return int(min(max(int(trade_from), 0), n))
    return int(index.searchsorted(pd.Timestamp(trade_from)))


class _Simulator:
    """Bucle vela a vela sobre arrays de numpy (estado mínimo: una posición y una orden pendiente)."""

    def __init__(self, settings: Settings, meta: Optional[AssetMeta], coin: str, equity: float, step: int,
                 allow_flip: bool):
        self.s = settings
        self.meta = meta
        self.coin = coin
        self.cash = equity
        self.step = step
        self.allow_flip = allow_flip
        self.slip = max(settings.paper_slippage_bps, 0.0) / 10_000.0
        self.taker = max(settings.paper_taker_fee_bps, 0.0) / 10_000.0
        self.maker = max(settings.paper_maker_fee_bps, 0.0) / 10_000.0
        self.ttl = max(settings.entry_ttl_minutes, 0.0) * 60.0
        self.trades: list[BTTrade] = []
        self.n_signals = 0
        self.n_skipped = 0
        # [dir, size, entry_fill, sl, tp, entry_i, entry_fee, equity_at_entry, entry_ts]
        self.pos: Optional[list] = None
        self.pend: Optional[tuple] = None  # (dir, is_limit, px, sl, tp, expiry_ts)
        self.flip_pending = False

    def _px(self, p: float) -> float:
        return round_price(p, self.meta.sz_decimals) if self.meta is not None and p > 0 else p

    def run(self, bars: pd.DataFrame, sig: pd.DataFrame, start: int) -> tuple[list[BTTrade], np.ndarray]:
        n = len(bars)
        o, h, l, c = (bars[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        ts = _index_seconds(bars.index)
        side = sig["side"].to_numpy()
        dirs = np.where(side == "long", 1, np.where(side == "short", -1, 0)).tolist()
        limit = np.isin(sig["entry_mode"].to_numpy(), ["zone", "limit"]).tolist()
        epx, sl, tp = (sig[k].to_numpy(dtype=float).tolist() for k in ("entry_px", "sl", "tp"))
        o_l, h_l, l_l, c_l, ts_l = o.tolist(), h.tolist(), l.tolist(), c.tolist(), ts.tolist()
        eq = np.empty(n - start, dtype=float)
        mark = math.nan
        for j in range(start, n):
            oj, hj, lj, cj = o_l[j], h_l[j], l_l[j], c_l[j]
            entered = 0                                   # 1 = a mercado en el open, 2 = límite en la vela
            # 1) órdenes que se activan en esta vela
            if self.pend is not None:
                d, is_limit, px, p_sl, p_tp, expiry = self.pend
                if self.flip_pending and self.pos is not None:
                    self._exit(j, oj, "flip", ts_l[j], held=j - self.pos[5])
                self.flip_pending = False
                if not is_limit:
                    self.pend = None
                    if self._enter(j, d, oj * (1.0 + d * self.slip), oj, p_sl, p_tp, self.taker, ts_l[j]):
                        entered = 1
                elif ts_l[j] + self.step > expiry:
                    self.pend = None                      # caducada sin ejecutar
                    self.n_skipped += 1
                elif (lj <= px) if d > 0 else (hj >= px):
                    self.pend = None
                    if self._enter(j, d, px, px, p_sl, p_tp, self.maker, ts_l[j]):
                        entered = 2
            # 2) salidas de la posición abierta
            pos = self.pos
            if pos is not None:
                d, p_sl, p_tp = pos[0], pos[3], pos[4]
                if entered == 0 and ((oj <= p_sl) if d > 0 else (oj >= p_sl)):
                    self._exit(j, oj, "sl", ts_l[j])     # hueco a través del SL: al open (peor)
                elif entered == 0 and ((oj >= p_tp) if d > 0 else (oj <= p_tp)):
                    self._exit(j, p_tp, "tp", ts_l[j])   # hueco a través del TP: al TP, sin regalo
                elif (lj <= p_sl) if d > 0 else (hj >= p_sl):
                    self._exit(j, p_sl, "sl", ts_l[j])   # SL primero aunque la vela toque también el TP
                elif entered != 2 and ((hj >= p_tp) if d > 0 else (lj <= p_tp)):
                    self._exit(j, p_tp, "tp", ts_l[j])
            # 3) capital marcado a mercado al cierre (último cierre válido); en la última vela se cierra todo
            if cj == cj:
                mark = cj
            if j == n - 1 and self.pos is not None:
                self._exit(j, mark, "end", ts_l[j])
            pos = self.pos
            eq[j - start] = self.cash + (pos[0] * (mark - pos[2]) * pos[1] if pos is not None else 0.0)
            # 4) señal al cierre de la vela -> orden para la siguiente
            d = dirs[j]
            if d == 0 or j == n - 1 or self.pend is not None or self.cash <= 0:
                continue
            if self.pos is not None and not (self.allow_flip and self.pos[0] != d):
                continue
            self.n_signals += 1
            self.flip_pending = self.pos is not None
            expiry = ts_l[j] + self.step + self.ttl       # colocada al cierre de la vela j
            self.pend = (d, limit[j], epx[j], sl[j], tp[j], expiry)
        return self.trades, eq

    def _enter(self, j: int, d: int, fill: float, ref: float, p_sl: float, p_tp: float, fee_rate: float,
               ts: float) -> bool:
        """Abre si el riesgo en vivo lo aprobaría (niveles del lado correcto, tamaño >= mínimo)."""
        if not (math.isfinite(fill) and fill > 0 and math.isfinite(p_sl) and p_sl > 0
                and math.isfinite(p_tp) and p_tp > 0):
            self.n_skipped += 1
            return False
        p_sl, p_tp, ref = self._px(p_sl), self._px(p_tp), self._px(ref)
        if not ((p_sl < min(fill, ref) and p_tp > max(fill, ref)) if d > 0
                else (p_sl > max(fill, ref) and p_tp < min(fill, ref))):
            self.n_skipped += 1                           # SL/TP ya del lado equivocado al ejecutar
            return False
        size, _ = position_size(self.s, self.meta, self.cash, ref, p_sl)
        if size <= 0:
            self.n_skipped += 1
            return False
        fee = size * fill * fee_rate
        self.pos = [d, size, fill, p_sl, p_tp, j, fee, self.cash, ts]
        self.cash -= fee
        return True

    def _exit(self, j: int, px: float, reason: str, ts: float, held: Optional[int] = None) -> None:
        d, size, entry, p_sl, _, entry_i, entry_fee, eq_entry, entry_ts = self.pos
        fill = px * (1.0 - d * self.slip)
        fee = size * fill * self.taker
        gross = d * (fill - entry) * size
        self.cash += gross - fee
        pnl = gross - entry_fee - fee
        risk = size * abs(entry - p_sl)
        self.trades.append(BTTrade(
            coin=self.coin, side="long" if d > 0 else "short", entry_ts=entry_ts, entry_px=entry, exit_ts=ts,
            exit_px=fill, size=size, pnl=pnl, pnl_pct=pnl / eq_entry * 100.0 if eq_entry > 0 else 0.0,
            r_multiple=pnl / risk if risk > 0 else 0.0, fees=entry_fee + fee, exit_reason=reason,
            bars_held=held if held is not None else j - entry_i + 1))
        self.pos = None

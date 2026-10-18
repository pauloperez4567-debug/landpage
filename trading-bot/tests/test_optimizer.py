"""Optimizador walk-forward: sin fuga de datos, determinista, honesto con el ruido y rápido.

Lo importante (un backtest no garantiza nada; sólo cuenta lo de fuera de muestra):

- Cada tramo elige parámetros sólo con su entrenamiento: cambiar las velas posteriores no cambia lo
  elegido; una estrategia que mira el futuro se detecta y se invalida.
- Una estrategia que sobreajusta ruido (400 semillas de entradas al azar) brilla dentro de muestra y se
  rechaza fuera de muestra; en un paseo aleatorio con las estrategias reales no hay ganadora (y la
  significación es lo que lo impide: los cuatro filtros del contrato solos dejaban pasar suerte).
- Con un patrón claro (o tendencias reales) sí hay ganadora aprobada, con los parámetros correctos.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_candles
from desk.backtest import BTTrade
from desk.clock import SimClock
from desk.config import load_settings
from desk.optimizer import (DISCLAIMER, OptimizationReport, ReportStore, StrategyEvaluation, approval_reasons,
                            edge_t_stat, optimize, param_combos, required_t_stat, walk_forward_splits)
from desk.strategy import STRATEGIES, EmaTrendStrategy
from desk.tui import normalize_report
from synthetic import Broken, CoinFlip, HourPattern, Leaky, pattern_bars, trend_bars

ROOT = Path(__file__).resolve().parents[1]
T0 = 1_790_000_000.0


def new_settings(tmp: Path, **kw):
    return load_settings(env={}, dotenv_path=None, llm_enabled=False, state_dir=str(tmp / "state"), **kw)


@pytest.fixture(scope="module")
def noise(tmp_path_factory):
    """Las 5 estrategias reales x 60 combinaciones x 4 tramos sobre 3000 velas de puro ruido (una sola vez)."""
    s = new_settings(tmp_path_factory.mktemp("noise"))
    bars = make_candles(n=3000, interval="1h", vol_pct=0.6, seed=7)
    t = time.perf_counter()
    rep = optimize("BTC", "1h", bars, s, clock=SimClock(T0))
    return rep, bars, s, time.perf_counter() - t


# --- rendimiento e integración con las estrategias reales -------------------------------

def test_five_strategies_60_combos_4_folds_3000_bars_under_60s(noise):
    rep, bars, s, elapsed = noise
    assert elapsed < 60.0, f"optimización demasiado lenta: {elapsed:.1f} s"
    assert [e.strategy for e in sorted(rep.evaluations, key=lambda e: e.strategy)] == sorted(STRATEGIES)
    assert rep.folds == 4 and rep.bars == 3000
    for e in rep.evaluations:
        assert not e.error, e.reasons
        assert len(e.folds) == 4 and e.combos == min(60, len(param_combos(STRATEGIES[e.strategy], s, 10**6)))
        assert e.oos_result is not None and e.oos_result.bars == sum(f.test_bars for f in e.folds)
        assert e.oos_metrics["n_trades"] == len(e.oos_trades)


def test_random_walk_has_no_approved_winner_and_significance_is_what_stops_luck(noise):
    """En esta gráfica de ruido liquidity_sweep saca PF 1.81 con 27 operaciones fuera de muestra: pasaba los
    cuatro filtros del contrato por pura suerte. El filtro de significación (t del R medio, corregida por las
    5 estrategias comparadas) es lo que lo rechaza."""
    rep, *_ = noise
    assert rep.winner is None and rep.winner_evaluation is None
    assert all(not e.passed for e in rep.evaluations)
    # (liquidity_sweep además sale "no validable": sus zonas de backtest no son las que opera en vivo)
    own = {e.strategy: [r for r in e.reasons if not r.startswith("no validable")] for e in rep.evaluations}
    lucky = [e for e in rep.evaluations
             if len(own[e.strategy]) == 1 and own[e.strategy][0].startswith("ventaja no significativa")]
    assert lucky, "debería haber una estrategia que sólo cae por significación"
    o = lucky[0].oos_metrics
    assert o["profit_factor"] >= 1.15 and o["total_return_pct"] > 0 and o["n_trades"] >= 20
    assert o["t_stat"] < o["t_min"]
    assert "Sin ganadora" in rep.summary_text()


def test_overfitting_random_entries_shine_in_sample_and_are_rejected_out_of_sample(tmp_path):
    # Sin los frenos del riesgo en vivo (que cortarían la curva): aquí se mide el sobreajuste, con muchas operaciones.
    s = new_settings(tmp_path, max_drawdown_pct=90, daily_loss_limit_pct=50, max_consecutive_losses=0)
    for seed in (1, 2, 3):
        bars = make_candles(n=2400, interval="1h", vol_pct=0.5, seed=seed)
        rep = optimize("ETH", "1h", bars, s, strategies=[CoinFlip], clock=SimClock(T0))
        ev = rep.evaluations[0]
        assert rep.winner is None and not ev.passed and ev.reasons
        assert ev.oos_metrics["n_trades"] >= 100               # muchas operaciones: el azar no se disfraza
        # cada tramo eligió la "mejor" semilla de 60 con su entrenamiento: dentro de muestra parece una ventaja
        assert all(f.is_metrics["sharpe"] > 0 for f in ev.folds)
        assert ev.is_metrics["profit_factor"] > ev.oos_metrics["profit_factor"]
        assert len({f.params["seed"] for f in ev.folds}) > 1  # cada tramo "encuentra" una semilla distinta


def test_clear_pattern_produces_an_approved_winner_with_the_right_parameter(tmp_path):
    s = new_settings(tmp_path)
    bars = pattern_bars(hour=7)
    rep = optimize("HYPE", "1h", bars, s, strategies=[HourPattern], clock=SimClock(T0))
    ev = rep.evaluations[0]
    assert rep.winner == "hour_pattern" and ev.passed and ev.reasons == []
    assert ev.params == {"hour": 7} and all(f.params == {"hour": 7} for f in ev.folds)
    assert ev.combos == 24                                      # el grid cabe entero
    o = ev.oos_metrics
    assert o["n_trades"] >= 40 and o["total_return_pct"] > 0 and o["profit_factor"] > 5 and o["t_stat"] > o["t_min"]
    assert ev.folds_positive == len(ev.folds)


@pytest.mark.parametrize("seed", [1, 2])
def test_trend_following_on_real_trends_is_approved(tmp_path, seed):
    s = new_settings(tmp_path)
    rep = optimize("BTC", "1h", trend_bars(seed), s, strategies=["donchian_breakout"], clock=SimClock(T0))
    ev = rep.evaluations[0]
    assert rep.winner == "donchian_breakout" and ev.passed
    assert ev.oos_metrics["t_stat"] > ev.oos_metrics["t_min"]


# --- sin fuga de datos ------------------------------------------------------------------

def _alter_after(bars: pd.DataFrame, pos: int, seed: int) -> pd.DataFrame:
    """Otra historia a partir de ``pos`` (mismo número de velas): lo anterior queda intacto."""
    other = make_candles(mid=float(bars["close"].iloc[pos - 1]) * 1.3, n=len(bars) - pos, interval="1h",
                         vol_pct=1.2, seed=seed)
    out = bars.copy()
    out.iloc[pos:, :] = other.to_numpy()
    return out


@pytest.mark.parametrize("names", [["ema_trend", "donchian_breakout"], ["liquidity_sweep"]])
def test_each_fold_only_sees_its_training_window(tmp_path, names):
    s = new_settings(tmp_path, bt_max_combos=12)
    bars = make_candles(n=1500, interval="1h", vol_pct=0.6, seed=11)
    base = optimize("SOL", "1h", bars, s, strategies=names, clock=SimClock(T0))
    splits = walk_forward_splits(len(bars), s.bt_folds)
    for k in (1, 2):                                            # cambiar el futuro del entrenamiento del tramo k
        train_end = splits[k - 1][0]
        alt = optimize("SOL", "1h", _alter_after(bars, train_end, seed=100 + k), s, strategies=names,
                       clock=SimClock(T0))
        for name in names:
            a, b = base.get(name), alt.get(name)
            for j in range(k):                                  # tramos 1..k: mismos parámetros y mismo entrenamiento
                assert a.folds[j].params == b.folds[j].params
                assert a.folds[j].is_metrics == b.folds[j].is_metrics
            # y el dato cambiado sí cuenta para lo que viene después (no es un test vacío)
            assert a.folds[k - 1].oos_metrics != b.folds[k - 1].oos_metrics or a.folds[k:] != b.folds[k:]


def test_a_strategy_that_looks_ahead_is_detected_and_invalidated(tmp_path):
    s = new_settings(tmp_path)
    bars = make_candles(n=1200, interval="1h", vol_pct=0.6, seed=5)
    rep = optimize("BTC", "1h", bars, s, strategies=[Leaky, HourPattern], clock=SimClock(T0))
    ev = rep.get("leaky")
    assert not ev.passed and ev.error and "mira el futuro" in ev.reasons[0]
    assert rep.winner != "leaky"
    assert not rep.get("hour_pattern").error                   # la otra se evalúa con normalidad


# --- determinismo y muestreo --------------------------------------------------------------

def test_same_seed_same_report_and_seed_changes_the_sample(tmp_path):
    s = new_settings(tmp_path, bt_max_combos=15)
    bars = make_candles(n=1200, interval="1h", vol_pct=0.6, seed=3)
    kw = dict(strategies=["ema_trend", "rsi_reversion"], clock=SimClock(T0))
    a = optimize("BTC", "1h", bars, s, seed=7, **kw)
    b = optimize("BTC", "1h", bars, s, seed=7, **kw)
    c = optimize("BTC", "1h", bars, s, seed=8, **kw)
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)
    assert a.id == b.id and a.id != c.id
    assert param_combos(EmaTrendStrategy, s, 15, seed=7) != param_combos(EmaTrendStrategy, s, 15, seed=8)


def test_param_combos_default_first_valid_bounded_and_full_grid_when_it_fits(settings):
    for cls in STRATEGIES.values():
        combos = param_combos(cls, settings, 60, seed=7)
        grid_size = math.prod(len(v) for v in cls.PARAM_GRID.values())
        assert len(combos) == min(60, grid_size)
        assert len({json.dumps(c, sort_keys=True) for c in combos}) == len(combos)
        default = {k: (settings.default_tp_r if k == "tp_r" and getattr(cls, k) is None else getattr(cls, k))
                   for k in cls.PARAM_GRID}
        assert combos[0] == default                             # la configuración por defecto siempre se prueba
        for c in combos:
            cls(settings, **c)                                  # todas válidas
    every = param_combos(STRATEGIES["donchian_breakout"], settings, 10**6)
    assert len(every) == 48
    assert param_combos(EmaTrendStrategy, settings, 0) == param_combos(EmaTrendStrategy, settings, 1)


def test_sampling_is_stable_across_processes(settings):
    """Mismo resultado aunque cambie PYTHONHASHSEED (nada de ``hash()`` de cadenas)."""
    code = ("import json,sys; sys.path.insert(0, %r); from desk.config import Settings; "
            "from desk.optimizer import param_combos; from desk.strategy import STRATEGIES; "
            "print(json.dumps(param_combos(STRATEGIES['ema_trend'], Settings(), 20, seed=7)))" % str(ROOT))
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    from desk.config import Settings
    assert json.loads(out.stdout) == param_combos(STRATEGIES["ema_trend"], Settings(), 20, seed=7)


def test_walk_forward_splits_are_anchored_contiguous_and_shrink_with_short_history():
    assert walk_forward_splits(3000, 4) == [(600, 600, 1200), (1200, 1200, 1800), (1800, 1800, 2400),
                                            (2400, 2400, 3000)]
    sp = walk_forward_splits(1003, 4)
    assert sp[0][0] == 200 and sp[-1][2] == 1003                # el último bloque se queda el resto
    assert all(a[2] == b[1] for a, b in zip(sp, sp[1:]))        # contiguos
    assert len(walk_forward_splits(160, 4)) == 2                # 160 velas: sólo caben 3 bloques de 50
    assert walk_forward_splits(99, 4) == [] and walk_forward_splits(5000, 0) == []


# --- aprobación y significación -----------------------------------------------------------

def test_approval_reasons_explain_every_rejection(settings):
    good = {"n_trades": 40, "profit_factor": 1.6, "total_return_pct": 5.0, "max_drawdown_pct": 4.0,
            "t_stat": 3.0, "t_min": 2.4}
    assert approval_reasons(good, settings) == []
    # El límite de drawdown en vivo (max_drawdown_pct = 5 %: cierra todo y halt) manda sobre bt_max_drawdown_pct (25 %)
    live = approval_reasons(dict(good, max_drawdown_pct=7.7), settings)
    assert len(live) == 1 and "7.7 % >= 5 %" in live[0] and "max_drawdown_pct" in live[0]
    halted = approval_reasons(dict(good, halted=1.0, halt_ts=1_767_225_600.0), settings)
    assert len(halted) == 1 and "se habría detenido el 2026-01-01" in halted[0]
    bad = {"n_trades": 12, "profit_factor": 0.9, "total_return_pct": -1.0, "max_drawdown_pct": 31.0,
           "t_stat": 1.0, "t_min": 2.4}
    reasons = approval_reasons(bad, settings)
    assert len(reasons) == 5
    text = " ".join(reasons)
    for word in ("pocas operaciones", "profit factor", "retorno", "drawdown", "no significativa"):
        assert word in text
    assert approval_reasons(dict(good, t_stat=-0.5), settings) == ["R medio por operación <= 0 fuera de muestra "
                                                                   "(t = -0.50)"]
    assert approval_reasons({k: v for k, v in good.items() if not k.startswith("t_")}, settings) == []


def test_t_statistic_and_threshold():
    def tr(r):
        return BTTrade("X", "long", 0, 1, 1, 1, 1, r, r, r, 0, "tp", 1)

    assert edge_t_stat([]) == 0.0 and edge_t_stat([tr(1.0)]) == 0.0
    rs = [1.0, -1.0, 2.0, -1.0, 1.5]
    expected = np.mean(rs) / (np.std(rs, ddof=1) / math.sqrt(len(rs)))
    assert edge_t_stat([tr(r) for r in rs]) == pytest.approx(expected)
    assert edge_t_stat([tr(0.9)] * 5) == 99.0 and edge_t_stat([tr(-1.0)] * 5) == -99.0
    # cuantiles de Student conocidos (una cola) y Bonferroni
    assert required_t_stat(20, 1, alpha=0.05) == pytest.approx(1.729, abs=0.005)
    assert required_t_stat(10, 1, alpha=0.01) == pytest.approx(2.821, abs=0.03)
    assert required_t_stat(10_000, 1) == pytest.approx(1.645, abs=0.002)
    assert required_t_stat(30, 5) > required_t_stat(30, 1) > required_t_stat(300, 1)


# --- robustez: nunca tumba la optimización ------------------------------------------------

def test_a_broken_strategy_is_reported_and_the_rest_continue(tmp_path):
    s = new_settings(tmp_path)
    rep = optimize("BTC", "1h", pattern_bars(), s, strategies=[Broken, HourPattern], clock=SimClock(T0))
    assert rep.winner == "hour_pattern"
    bad = rep.get("broken")
    assert not bad.passed and bad.score is None and "error al evaluar" in bad.reasons[0]
    assert rep.evaluations[-1].strategy == "broken"             # los fallidos al final del ranking
    assert "no evaluada" in OptimizationReport.from_dict(rep.to_dict()).summary_text() or "—" in rep.summary_text()


def test_short_history_and_bad_inputs(tmp_path):
    s = new_settings(tmp_path)
    rep = optimize("BTC", "4h", make_candles(n=80, interval="4h"), s, strategies=["donchian_breakout"],
                   clock=SimClock(T0))
    assert rep.winner is None and rep.folds == 0
    assert "historia insuficiente" in rep.evaluations[0].reasons[0]
    assert "Sin ganadora" in rep.summary_text()
    with pytest.raises(ValueError):
        optimize("BTC", "1h", make_candles(n=500, interval="1h"), s, strategies=["no_existe"])
    with pytest.raises(ValueError):
        optimize("BTC", "1h", make_candles(n=500, interval="1h").reset_index(drop=True), s)
    with pytest.raises(ValueError):
        optimize("BTC", "1h", make_candles(n=500, interval="1h")[["open", "close"]], s)
    with pytest.raises(ValueError):
        optimize("BTC", "7h", make_candles(n=500, interval="1h"), s)


def test_unsorted_and_duplicated_bars_are_cleaned_with_a_note(tmp_path):
    s = new_settings(tmp_path)
    bars = pattern_bars(n=800)
    messy = pd.concat([bars.iloc[400:], bars.iloc[:400], bars.iloc[10:12]])
    rep = optimize("BTC", "1h", messy, s, strategies=[HourPattern], clock=SimClock(T0))
    clean = optimize("BTC", "1h", bars, s, strategies=[HourPattern], clock=SimClock(T0))
    assert rep.bars == 800 and any("desordenadas" in n for n in rep.notes) and any("repetidas" in n for n in rep.notes)
    assert rep.evaluations[0].oos_metrics == clean.evaluations[0].oos_metrics


def test_progress_and_should_stop(tmp_path):
    s = new_settings(tmp_path, bt_max_combos=8)
    bars = pattern_bars(n=800)
    calls = []
    rep = optimize("BTC", "1h", bars, s, strategies=[HourPattern, "ema_trend"], clock=SimClock(T0),
                   progress=lambda d, t, txt: calls.append((d, t, txt)))
    assert calls and calls[-1][0] == calls[-1][1]               # termina en el 100 %
    assert [c[0] for c in calls] == sorted(c[0] for c in calls)
    assert all(isinstance(c[2], str) and c[2] for c in calls)

    def boom(*_):
        raise RuntimeError("el aviso falla")

    assert optimize("BTC", "1h", bars, s, strategies=[HourPattern], clock=SimClock(T0), progress=boom).winner

    asked = []

    def stop_after_first_strategy():
        asked.append(1)
        return len(asked) > 8 + 4 + 2                           # cada estrategia pregunta 8 + 4 veces; para en la 2.ª

    rep = optimize("BTC", "1h", bars, s, strategies=[HourPattern, "ema_trend"], clock=SimClock(T0),
                   should_stop=stop_after_first_strategy)
    assert not rep.complete and [e.strategy for e in rep.evaluations] == ["hour_pattern"]
    assert any("detenida" in n for n in rep.notes) and "INCOMPLETO" in rep.summary_text()
    rep = optimize("BTC", "1h", bars, s, strategies=[HourPattern], clock=SimClock(T0), should_stop=boom)
    assert rep.complete                                         # un should_stop roto no para nada


def test_unknown_objective_falls_back_to_sharpe_and_ranking_follows_the_objective(tmp_path):
    s = new_settings(tmp_path, bt_objective="lo_que_sea", bt_max_combos=6)
    rep = optimize("BTC", "1h", pattern_bars(n=900), s, strategies=[HourPattern, "ema_trend"], clock=SimClock(T0))
    assert rep.objective == "sharpe" and any("desconocido" in n for n in rep.notes)
    for e in rep.evaluations:
        if not e.error:
            assert e.score == pytest.approx(e.oos_metrics["sharpe"])
    s.bt_objective = "return"
    rep = optimize("BTC", "1h", pattern_bars(n=900), s, strategies=[HourPattern], clock=SimClock(T0))
    assert rep.evaluations[0].score == pytest.approx(rep.evaluations[0].oos_metrics["total_return_pct"])


# --- informe: JSON, texto, panel ----------------------------------------------------------

def test_report_json_round_trip_and_dashboard_compatibility(noise, tmp_path):
    rep, *_ = noise
    s = new_settings(tmp_path)
    good = optimize("HYPE", "1h", pattern_bars(), s, strategies=[HourPattern, "ema_trend"], clock=SimClock(T0))
    for r in (rep, good):
        d = r.to_dict()
        text = json.dumps(d, allow_nan=False)                   # JSON estricto: nada de NaN/inf
        back = OptimizationReport.from_dict(json.loads(text))
        assert back.to_dict() == d
        assert OptimizationReport.from_dict({"report": d, "result": None}).id == r.id
        for a, b in zip(r.evaluations, back.evaluations):
            assert b.oos_result is None and a.oos_trades == b.oos_trades and a.folds == b.folds
        n = normalize_report(d)                                 # lo que pinta ``python -m desk dashboard``
        assert n["id"] == r.id and n["folds"] == r.folds and len(n["evaluations"]) == len(r.evaluations)
    n = normalize_report(good.to_dict())
    assert n["winner"]["strategy"] == "hour_pattern" and n["winner"]["passed"]
    assert normalize_report(rep.to_dict())["winner"] is None


def test_from_dict_tolerates_junk():
    rep = OptimizationReport.from_dict({"coin": "BTC", "winner": {"strategy": "x"}, "folds": [1, 2],
                                        "evaluations": [{"strategy": "x", "oos_trades": 41, "folds": 4,
                                                         "score": "?", "oos_metrics": {"n_trades": "3", "pf": None}},
                                                        "basura"]})
    assert rep.winner == "x" and rep.folds == 2 and rep.evaluations[0].oos_trades == []
    assert rep.evaluations[0].oos_metrics["n_trades"] == 3 and rep.evaluations[0].score == 0.0
    assert StrategyEvaluation.from_dict({}).strategy == ""


def test_summary_text_spanish_in_vs_out_of_sample_with_disclaimer(tmp_path):
    s = new_settings(tmp_path)
    rep = optimize("HYPE", "1h", pattern_bars(), s, strategies=[HourPattern, CoinFlip], clock=SimClock(T0))
    text = rep.summary_text()
    assert "DENTRO" in text and "FUERA de muestra" in text and "Walk-forward anclado de 4 tramos" in text
    assert "Ganadora: hour_pattern" in text and "hour=7" in text and "activa HYPE 1h" in text
    assert "Rechazadas:" in text and "coin_flip:" in text
    assert DISCLAIMER in text and "no garantiza" in text
    table = [ln for ln in text.splitlines() if ln.startswith(("hour_pattern", "coin_flip"))]
    assert len(table) == 2 and all(len(ln) <= 80 for ln in table)
    assert table[0].startswith("hour_pattern") and table[0].endswith("✔") and table[1].endswith("✘")


# --- ReportStore -----------------------------------------------------------------------------

def test_report_store_save_latest_list_get_and_result(tmp_path):
    s = new_settings(tmp_path)
    store = ReportStore(s.state_dir)
    bars = pattern_bars(n=800)
    reps = []
    for i, (coin, interval) in enumerate([("HYPE", "1h"), ("BTC", "1h"), ("HYPE", "1h"), ("HYPE", "4h")]):
        rep = optimize(coin, interval, bars, s, strategies=[HourPattern], clock=SimClock(T0 + i * 60))
        path = store.save(rep)
        assert path.name == f"{coin}_{interval}_{rep.id}.json" and path.parent == Path(s.state_dir) / "reports"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {"report", "result"} and data["result"]["strategy"] == "hour_pattern"
        reps.append(rep)
    assert store.latest("HYPE", "1h").id == reps[2].id
    assert store.latest("BTC", "1h").id == reps[1].id and store.latest("ETH", "1h") is None
    assert [r.id for r in store.list(3)] == [reps[3].id, reps[2].id, reps[1].id]
    assert store.get(reps[0].id).created_ts == T0 and store.get("nope") is None and store.get("") is None
    res = store.get_result(reps[2].id)
    mem = reps[2].result()
    assert res.strategy == "hour_pattern" and [t.to_dict() for t in res.trades] == [t.to_dict() for t in mem.trades]
    assert res.equity.iloc[-1] == pytest.approx(mem.equity.iloc[-1])


def test_report_store_survives_junk_prunes_and_sanitizes_names(tmp_path):
    s = new_settings(tmp_path)
    store = ReportStore(s.state_dir, keep=3)
    (store.dir).mkdir(parents=True)
    (store.dir / "HYPE_1h_roto.json").write_text("{", encoding="utf-8")
    (store.dir / "HYPE_1h_lista.json").write_text("[1, 2]", encoding="utf-8")
    assert store.latest("HYPE", "1h") is None and store.list() == [] and store.get("roto") is None
    bars = pattern_bars(n=800)
    saved = []
    for i in range(4):
        rep = optimize("xyz:GOLD", "1h", bars, s, strategies=[HourPattern], clock=SimClock(T0 + i))
        saved.append(store.save(rep, result=None if i else rep.result()))
        os.utime(saved[-1], (T0 + 100 + i, T0 + 100 + i))       # orden de modificación determinista
    assert all(":" not in p.name for p in saved) and saved[0].name.startswith("xyz-GOLD_1h_")
    assert len(list(store.dir.glob("*.json"))) == 3             # se quedan los 3 más recientes
    assert not saved[0].exists() and store.latest("xyz:GOLD", "1h").created_ts == T0 + 3


def test_zone_failure_or_bad_folds_setting_never_crash(tmp_path, monkeypatch):
    import desk.optimizer as opt

    s = new_settings(tmp_path, bt_max_combos=4)
    bars = pattern_bars(n=900)

    def boom(*_a, **_k):
        raise RuntimeError("zonas rotas")

    monkeypatch.setattr(opt, "historical_zones", boom)
    rep = optimize("BTC", "1h", bars, s, strategies=["liquidity_sweep", HourPattern], clock=SimClock(T0))
    assert rep.winner == "hour_pattern" and any("zonas históricas" in n for n in rep.notes)
    sweep = rep.get("liquidity_sweep")
    assert not sweep.passed and sweep.oos_metrics["n_trades"] == 0
    s.bt_folds = 0
    rep = optimize("BTC", "1h", bars, s, strategies=[HourPattern], clock=SimClock(T0))
    assert rep.folds == 1 and any("bt_folds" in n for n in rep.notes)


# --- lo que se adopta, frenos en vivo, capital, Bonferroni y zonas (revisión de validez) -----------------------

def regime_bars(n: int = 1500, flip_at: int = 1200, hour: int = 7, jump: float = 0.015, vol: float = 0.003,
                seed: int = 3) -> pd.DataFrame:
    """Como ``pattern_bars`` hasta ``flip_at``; desde ahí el patrón se invierte (la vela de la hora ``hour + 1``
    cae): cambio de régimen justo en el último tramo de validación."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-03-01", periods=n, freq="1h", tz="UTC")
    special = idx.hour.to_numpy() == (hour + 1) % 24
    sign = np.where(np.arange(n) < flip_at, 1.0, -1.0)
    rets = rng.normal(-jump / 23.0 * sign, vol, n)
    rets[special] = jump * sign[special]
    c = 100.0 * np.exp(np.cumsum(rets))
    o = np.r_[100.0, c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.001, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.001, n)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1_000.0}, index=idx)


def test_adopted_parameters_must_win_in_their_own_out_of_sample_fold(tmp_path):
    """La curva encadenada (3 tramos buenos y 1 malo) pasa todos los filtros, pero lo que se adoptaría -los
    parámetros del último entrenamiento- pierde en su único tramo fuera de muestra: no se aprueba."""
    s = new_settings(tmp_path)
    rep = optimize("ETH", "1h", regime_bars(), s, strategies=[HourPattern], clock=SimClock(T0))
    e = rep.get("hour_pattern")
    assert approval_reasons(e.oos_metrics, s) == []            # el procedimiento, encadenado, sí "aprobaría"
    assert not e.passed and rep.winner is None
    assert len(e.reasons) == 1 and "parámetros que se adoptarían" in e.reasons[0] and "T4" in e.reasons[0]
    ad = e.adopted_oos
    assert ad["fold"] == 4 and ad["total_return_pct"] < 0 and ad["n_trades"] >= 10
    assert e.params == e.folds[-1].params == {"hour": 7}
    back = StrategyEvaluation.from_dict(json.loads(json.dumps(e.to_dict())))
    assert back.adopted_metrics == pytest.approx(e.adopted_metrics)
    from desk.playbook import PlaybookEntry, entry_from_report
    entry = entry_from_report(rep, "hour_pattern", allow_rejected=True, clock=SimClock(T0))
    assert entry.adopted_metrics["fold"] == 4 and entry.oos_metrics == e.oos_metrics
    text = PlaybookEntry.from_dict(entry.to_dict()).describe()
    assert "procedimiento" in text and "con estos parámetros (tramo T4) -" in text


def test_adopted_reasons_unit(settings):
    from desk.optimizer import adopted_reasons
    assert adopted_reasons({"fold": 4, "total_return_pct": 2.0, "profit_factor": 1.5, "n_trades": 9}, settings) == []
    [r] = adopted_reasons({"fold": 4, "total_return_pct": -0.04, "profit_factor": 0.99, "n_trades": 11}, settings)
    assert "T4" in r and "-0.04 %" in r
    [h] = adopted_reasons({"fold": 2, "total_return_pct": 5.0, "profit_factor": 2.0, "halted": 1.0}, settings)
    assert "halt por drawdown" in h
    assert adopted_reasons({}, settings) == [] and adopted_reasons(None, settings) == []


def test_out_of_sample_curve_is_cut_by_the_live_drawdown_halt(tmp_path):
    """Con max_drawdown_pct = 1 % la curva fuera de muestra se corta en el halt (como la mesa en vivo, que cierra
    todo y no vuelve a operar sin 'reset halt') y no se aprueba, aunque el DD esté muy por debajo de
    bt_max_drawdown_pct."""
    loose = new_settings(tmp_path, max_drawdown_pct=50.0, daily_loss_limit_pct=50.0, max_consecutive_losses=0)
    tight = new_settings(tmp_path, max_drawdown_pct=1.0, daily_loss_limit_pct=50.0, max_consecutive_losses=0)
    bars = trend_bars(2, vol=0.014)
    a = optimize("BTC", "1h", bars, loose, strategies=["donchian_breakout"], clock=SimClock(T0)).get("donchian_breakout")
    b = optimize("BTC", "1h", bars, tight, strategies=["donchian_breakout"], clock=SimClock(T0)).get("donchian_breakout")
    assert a.oos_metrics["halted"] == 0.0 and a.oos_metrics["n_trades"] > b.oos_metrics["n_trades"]
    assert b.oos_metrics["halted"] == 1.0 and not b.passed
    assert any("se habría detenido" in r for r in b.reasons)
    halt_ts = b.oos_metrics["halt_ts"]
    assert all(t.entry_ts <= halt_ts for t in b.oos_trades)
    assert b.oos_metrics["max_drawdown_pct"] < tight.bt_max_drawdown_pct


def test_account_equity_is_the_starting_capital(tmp_path):
    s = new_settings(tmp_path)
    bars = pattern_bars(n=1500)
    base = optimize("ETH", "1h", bars, s, strategies=[HourPattern], clock=SimClock(T0))
    small = optimize("ETH", "1h", bars, s, strategies=[HourPattern], clock=SimClock(T0), start_equity=2_000.0)
    assert base.start_equity == s.paper_start_equity and base.criteria["equity_source"] == "paper_start_equity"
    assert small.start_equity == 2_000.0 and small.criteria["equity_source"] == "account"
    res = small.result()
    assert res.equity.iloc[0] == pytest.approx(2_000.0, rel=0.05)
    assert "Capital supuesto: 2,000 USD (equity de la cuenta)" in small.summary_text()
    # Tamaño en USD fijos: con menos capital el mismo PnL pesa más en %.
    assert small.get("hour_pattern").oos_metrics["total_return_pct"] > base.get("hour_pattern").oos_metrics[
        "total_return_pct"]
    bad = optimize("ETH", "1h", bars, s, strategies=[HourPattern], clock=SimClock(T0), start_equity=-5)
    assert bad.start_equity == s.paper_start_equity and any("no válido" in n for n in bad.notes)
    assert OptimizationReport.from_dict(small.to_dict()).start_equity == 2_000.0


def test_significance_correction_counts_every_strategy_even_if_one_is_requested(tmp_path):
    """Pedir una estrategia sola no baja el listón (m = todas las que se pueden probar en la gráfica)."""
    s = new_settings(tmp_path, bt_max_combos=8)
    bars = make_candles(n=1500, interval="1h", vol_pct=0.6, seed=11)
    one = optimize("BTC", "1h", bars, s, strategies=["squeeze_breakout"], clock=SimClock(T0))
    assert one.criteria["compared"] == len(STRATEGIES)
    e = one.get("squeeze_breakout")
    assert e.oos_metrics["t_min"] == pytest.approx(required_t_stat(int(e.oos_metrics["n_trades"]), len(STRATEGIES)))


def test_liquidity_strategies_are_not_validable_with_approximate_zones(tmp_path):
    from desk.liquidity import historical_zones
    s = new_settings(tmp_path, bt_max_combos=2, bt_folds=2)
    bars = make_candles(n=900, interval="15m", vol_pct=0.4, seed=5)
    rep = optimize("SOL", "15m", bars, s, strategies=["liquidity_sweep"], clock=SimClock(T0))
    e = rep.get("liquidity_sweep")
    assert not e.passed and any(r.startswith("no validable") for r in e.reasons)
    assert any("no aprueba (no validable)" in n for n in rep.notes)
    # Con zonas grabadas de verdad (aquí, unas dadas) ya no es "no validable" por eso.
    given = optimize("SOL", "15m", bars, s, strategies=["liquidity_sweep"], clock=SimClock(T0),
                     zones=historical_zones(bars, s)).get("liquidity_sweep")
    assert not any(r.startswith("no validable") for r in given.reasons)

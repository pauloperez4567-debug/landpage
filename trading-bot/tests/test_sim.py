"""Mercados offline: SimMarketData (paseo determinista) y ReplayMarketData (CSV)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from desk.clock import SimClock
from desk.exchange.base import CANDLE_COLUMNS, MarketDataError, round_price
from desk.exchange.paper import PaperBroker
from desk.exchange.sim import DEFAULT_ASSETS, ReplayMarketData, SimMarketData
from desk.models import OrderKind, OrderStatus

from conftest import FakeMarketData, make_candles

T0 = 1_767_225_600.0


def epoch_s(index: pd.DatetimeIndex) -> np.ndarray:
    """Segundos UNIX del índice, sin depender de su unidad interna (s/ms/ns)."""
    return np.array([ts.timestamp() for ts in index])


def check_candle_format(df: pd.DataFrame, sec: int, now: float) -> None:
    assert list(df.columns) == CANDLE_COLUMNS
    assert all(df[c].dtype == float for c in CANDLE_COLUMNS)
    assert isinstance(df.index, pd.DatetimeIndex) and str(df.index.tz) == "UTC"
    assert df.index.unit == "ns"                                         # unidad homogénea entre fuentes
    assert df.index.is_monotonic_increasing and df.index.is_unique
    steps = np.diff(epoch_s(df.index))
    assert (steps == sec).all()
    assert df.index[-1].timestamp() == math.floor(now / sec) * sec      # la última es la vela en curso
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df["low"] > 0).all() and (df["volume"] >= 0).all()


# =========================================================================== SimMarketData

def test_metas_match_conftest():
    fake = FakeMarketData()
    sim = SimMarketData(SimClock())
    assert sim.coins() == ["BTC", "ETH", "SOL", "HYPE"]
    for coin in sim.coins():
        assert sim.meta(coin) == fake.meta(coin)
    with pytest.raises(KeyError):
        sim.meta("DOGE")
    with pytest.raises(KeyError):
        sim.bid_ask("DOGE")
    with pytest.raises(KeyError):
        sim.candles("DOGE", "1h", 5)


def test_meta_returns_copies():
    sim = SimMarketData(SimClock())
    sim.meta("BTC").max_leverage = 1
    assert sim.meta("BTC").max_leverage == 40
    assert DEFAULT_ASSETS["BTC"].meta.max_leverage == 40


def test_start_prices_and_valid_quotes():
    clock = SimClock()
    sim = SimMarketData(clock)
    for coin, asset in DEFAULT_ASSETS.items():
        assert sim.fair_price(coin) == pytest.approx(asset.start_px)
        bid, ask = sim.bid_ask(coin)
        assert 0 < bid < ask
        szd = asset.meta.sz_decimals
        assert round_price(bid, szd) == bid and round_price(ask, szd) == ask      # ticks válidos
        assert (ask - bid) / sim.mid(coin) * 1e4 < asset.spread_bps + 2.0
    assert sim.last_update_ts("BTC") == clock.now()


def test_deterministic_for_same_seed_and_independent_of_query_order():
    c1, c2 = SimClock(), SimClock()
    a, b = SimMarketData(c1, seed=11), SimMarketData(c2, seed=11)
    # ``a`` se consulta a menudo; ``b`` sólo al final: mismo mercado.
    for _ in range(50):
        c1.advance(37)
        a.bid_ask("ETH")
        a.candles("ETH", "15m", 5)
    c2.advance(50 * 37)
    assert a.bid_ask("ETH") == b.bid_ask("ETH")
    pd.testing.assert_frame_equal(a.candles("ETH", "15m", 100), b.candles("ETH", "15m", 100))
    assert a.l2_book("ETH", n_sig_figs=4) == b.l2_book("ETH", n_sig_figs=4)
    assert a.liquidation_levels("ETH") == b.liquidation_levels("ETH")


def test_same_market_after_restart():
    # Un reinicio (nueva instancia construida más tarde) ve el mismo mercado: las posiciones
    # paper persistidas siguen marcándose contra precios coherentes.
    c1 = SimClock()
    first = SimMarketData(c1)
    c1.advance(3 * 86_400 + 77)
    c2 = SimClock(c1.now())
    restarted = SimMarketData(c2)
    assert restarted.bid_ask("SOL") == first.bid_ask("SOL")
    custom = SimMarketData(SimClock(), anchor_ts=T0 + 3600)
    assert custom.price_at("BTC", T0 + 3600) == pytest.approx(DEFAULT_ASSETS["BTC"].start_px)


def test_different_seeds_give_different_markets():
    c = SimClock()
    a, b = SimMarketData(c, seed=1), SimMarketData(c, seed=2)
    c.advance(3 * 3600)
    assert a.fair_price("BTC") != b.fair_price("BTC")


def test_price_advances_with_clock_and_is_continuous():
    clock = SimClock()
    sim = SimMarketData(clock)
    prices = []
    for _ in range(200):
        prices.append(sim.fair_price("BTC"))
        clock.advance(1)                                   # pasos de 1 s: interpolación entre puntos de 5 s
    assert len(set(prices)) > 150
    rets = np.abs(np.diff(np.log(prices)))
    assert rets.max() < 0.01                               # sin saltos absurdos en 1 s
    clock.advance(86_400)
    assert sim.fair_price("BTC") != prices[0]


def test_realised_volatility_close_to_profile():
    clock = SimClock()
    sim = SimMarketData(clock, seed=3)
    df = sim.candles("SOL", "5m", 2016)                    # 7 días de historia
    rets = np.diff(np.log(df["close"].to_numpy()[:-1]))
    daily = rets.std() * math.sqrt(288) * 100
    assert 0.5 * DEFAULT_ASSETS["SOL"].daily_vol_pct < daily < 2.0 * DEFAULT_ASSETS["SOL"].daily_vol_pct


@pytest.mark.parametrize("interval,sec", [("1m", 60), ("15m", 900), ("1h", 3600), ("4h", 14400)])
def test_candle_format(interval, sec):
    clock = SimClock()
    clock.advance(5 * 3600 + 1234)
    sim = SimMarketData(clock)
    df = sim.candles("HYPE", interval, 120)
    assert len(df) == 120
    check_candle_format(df, sec, clock.now())
    # Velas contiguas: el cierre de una es la apertura de la siguiente (mismo instante del camino).
    assert np.allclose(df["open"].to_numpy()[1:], df["close"].to_numpy()[:-1], rtol=0, atol=1e-9)
    # La vela en curso cierra en el precio actual.
    assert df["close"].iloc[-1] == pytest.approx(sim.fair_price("HYPE"))


def test_closed_candles_never_change():
    clock = SimClock()
    sim = SimMarketData(clock)
    clock.advance(10 * 3600 + 450)
    before = sim.candles("BTC", "15m", 40)
    clock.advance(3 * 900)
    after = sim.candles("BTC", "15m", 40)
    common = before.index[:-1].intersection(after.index)
    assert len(common) == 36
    pd.testing.assert_frame_equal(before.loc[common], after.loc[common])


def test_hourly_candles_consistent_with_15m():
    clock = SimClock()
    sim = SimMarketData(clock)
    clock.advance(30 * 3600)
    h = sim.candles("ETH", "1h", 10).iloc[:-1]
    q = sim.candles("ETH", "15m", 40)
    for ts, row in h.iterrows():
        part = q.loc[ts:ts + pd.Timedelta(minutes=45)]
        assert len(part) == 4
        assert row["open"] == pytest.approx(part["open"].iloc[0])
        assert row["close"] == pytest.approx(part["close"].iloc[-1])
        assert row["high"] == pytest.approx(part["high"].max())
        assert row["low"] == pytest.approx(part["low"].min())
        assert row["volume"] == pytest.approx(part["volume"].sum())


def test_history_before_anchor_and_long_lookback():
    clock = SimClock()
    sim = SimMarketData(clock)
    df = sim.candles("BTC", "1h", 600)                     # 25 días antes del ancla
    check_candle_format(df, 3600, clock.now())
    assert df["close"].iloc[-1] == pytest.approx(DEFAULT_ASSETS["BTC"].start_px)
    assert df["low"].min() > 0


def test_invalid_configuration():
    with pytest.raises(ValueError):
        SimMarketData(SimClock(), step_seconds=7)
    with pytest.raises(ValueError):
        SimMarketData(SimClock()).candles("BTC", "2m", 10)


def test_l2_book_shape_and_walls():
    clock = SimClock()
    sim = SimMarketData(clock)
    clock.advance(1800)
    bid, ask = sim.bid_ask("BTC")
    book = sim.l2_book("BTC")
    assert book["ts"] == clock.now()
    assert book["bids"][0][0] == bid and book["asks"][0][0] == ask
    assert len(book["bids"]) == 20 and len(book["asks"]) == 20       # como Hyperliquid
    bpx = [p for p, _ in book["bids"]]
    apx = [p for p, _ in book["asks"]]
    assert bpx == sorted(bpx, reverse=True) and apx == sorted(apx)
    assert all(sz > 0 for _, sz in book["bids"] + book["asks"])
    # Agrupado (más profundidad): precios múltiplos de 100 en BTC con 3 cifras y muros visibles.
    agg = sim.l2_book("BTC", n_sig_figs=3)
    assert all(p % 100 == 0 for p, _ in agg["bids"] + agg["asks"])
    assert agg["bids"][0][0] <= bid and agg["asks"][0][0] >= ask
    usd = np.array([p * s for p, s in agg["bids"] + agg["asks"]])
    assert usd.max() >= 3 * np.median(usd)
    assert sim.l2_book("BTC", n_sig_figs=3) == agg                    # determinista


def test_liquidation_levels_shape():
    clock = SimClock()
    sim = SimMarketData(clock)
    clock.advance(4000)
    mid = sim.fair_price("SOL")
    levels = sim.liquidation_levels("SOL")
    assert levels and all(set(d) == {"liq_px", "usd", "side"} for d in levels)
    longs = [d for d in levels if d["side"] == "long"]
    shorts = [d for d in levels if d["side"] == "short"]
    assert longs and shorts
    assert all(d["liq_px"] < mid for d in longs) and all(d["liq_px"] > mid for d in shorts)
    assert all(d["usd"] > 0 for d in levels)


def test_sim_drives_paper_broker(settings):
    clock = SimClock()
    sim = SimMarketData(clock, seed=5)
    broker = PaperBroker(sim, settings, clock)
    broker.set_leverage("BTC", 3)
    o = broker.place_order("BTC", True, 0.01, OrderKind.MARKET, tag="entry")
    assert o.status is OrderStatus.FILLED
    stop = broker.place_order("BTC", False, 0.01, OrderKind.STOP, trigger_px=o.avg_fill_px * 0.995,
                              reduce_only=True, tag="sl")
    equities = []
    for _ in range(24 * 60):                               # un día en pasos de 1 min
        clock.advance(60)
        broker.step()
        equities.append(broker.account().equity)
        if broker.get_order(stop.id).status is OrderStatus.FILLED:
            break
    assert len(set(round(e, 6) for e in equities)) > 1     # el mark se mueve con el reloj
    fills = broker.poll_fills()
    assert fills[0].tag == "entry"


# =========================================================================== ReplayMarketData

def write_csv(path, n=300, interval="15m", ms=True, coin_col=None, seed=7):
    df = make_candles(mid=100.0, n=n, interval=interval, end_ts=T0 - 86_400 * 30, seed=seed)
    secs = epoch_s(df.index).astype("int64")
    out = df.reset_index(drop=True)
    out.insert(0, "timestamp", secs * 1000 if ms else pd.to_datetime(secs, unit="s", utc=True).astype(str))
    if coin_col:
        out.insert(1, "coin", coin_col)
    out.to_csv(path, index=False)
    return df


def test_replay_starts_after_warmup_and_interpolates(tmp_path):
    src = write_csv(tmp_path / "btc_15m.csv")
    clock = SimClock()
    rep = ReplayMarketData(tmp_path / "btc_15m.csv", clock)
    assert rep.coins() == ["BTC"] and rep.meta("BTC") == DEFAULT_ASSETS["BTC"].meta
    row = src.iloc[150]                                    # warmup = min(200, 300 // 2)
    assert rep.price_at("BTC", clock.now()) == pytest.approx(row["open"])
    bullish = row["close"] >= row["open"]
    first, second = (row["low"], row["high"]) if bullish else (row["high"], row["low"])
    clock.advance(150)                                     # 1/6 de la vela: a mitad de camino del primer extremo
    assert rep.price_at("BTC", clock.now()) == pytest.approx((row["open"] + first) / 2)
    clock.advance(150)                                     # 1/3: el primer extremo
    assert rep.price_at("BTC", clock.now()) == pytest.approx(first)
    clock.advance(300)                                     # 2/3: el segundo extremo
    assert rep.price_at("BTC", clock.now()) == pytest.approx(second)
    clock.advance(300)                                     # vela siguiente: su apertura
    assert rep.price_at("BTC", clock.now()) == pytest.approx(src.iloc[151]["open"])
    bid, ask = rep.bid_ask("BTC")
    assert bid <= rep.price_at("BTC", clock.now()) <= ask


def test_replay_candles_shifted_to_clock_time(tmp_path):
    src = write_csv(tmp_path / "sol.csv")
    clock = SimClock()
    rep = ReplayMarketData(tmp_path / "sol.csv", clock, coin="SOL", warmup_candles=100)
    clock.advance(5 * 900 + 300)                           # vela 105 a 1/3
    df = rep.candles("SOL", "15m", 50)
    check_candle_format(df, 900, clock.now())
    assert len(df) == 50
    closed = df.iloc[:-1]
    expected = src.iloc[56:105]
    assert np.allclose(closed[CANDLE_COLUMNS].to_numpy(), expected[CANDLE_COLUMNS].to_numpy())
    cur, row = df.iloc[-1], src.iloc[105]
    assert cur["open"] == pytest.approx(row["open"])
    assert cur["close"] == pytest.approx(rep.price_at("SOL", clock.now()))
    assert cur["volume"] == pytest.approx(row["volume"] / 3)
    assert rep.last_update_ts("SOL") == clock.now()


def test_replay_resamples_to_larger_intervals(tmp_path):
    src = write_csv(tmp_path / "eth.csv", n=400)
    clock = SimClock()
    rep = ReplayMarketData(tmp_path / "eth.csv", clock, coin="ETH", warmup_candles=200)
    df = rep.candles("ETH", "1h", 10)
    check_candle_format(df, 3600, clock.now())
    # Comparar una hora cerrada con la agregación manual de las 4 velas de 15m del CSV.
    shifted = src.copy()
    shifted.index = shifted.index - pd.Timedelta(seconds=rep.offset)
    ts = df.index[-2]
    part = shifted.loc[ts:ts + pd.Timedelta(minutes=45)]
    assert len(part) == 4
    row = df.loc[ts]
    assert row["open"] == pytest.approx(part["open"].iloc[0])
    assert row["high"] == pytest.approx(part["high"].max())
    assert row["low"] == pytest.approx(part["low"].min())
    assert row["close"] == pytest.approx(part["close"].iloc[-1])
    assert row["volume"] == pytest.approx(part["volume"].sum())
    with pytest.raises(MarketDataError):
        rep.candles("ETH", "5m", 10)                        # más fino que el CSV: imposible


def test_replay_end_of_data(tmp_path):
    src = write_csv(tmp_path / "hype.csv", n=60)
    clock = SimClock()
    rep = ReplayMarketData(tmp_path / "hype.csv", clock, coin="HYPE")
    assert not rep.finished
    clock.advance(86_400)
    assert rep.finished
    assert rep.price_at("HYPE", clock.now()) == pytest.approx(src["close"].iloc[-1])
    assert rep.last_update_ts("HYPE") < clock.now() - 3600       # datos viejos: el riesgo no entra
    assert len(rep.candles("HYPE", "15m", 1000)) == 60
    bid, ask = rep.bid_ask("HYPE")
    assert bid < ask


def test_replay_iso_timestamps_multi_coin_and_absolute_time(tmp_path):
    a = make_candles(mid=100.0, n=20, interval="1h", end_ts=T0, seed=1)
    b = make_candles(mid=2_000.0, n=20, interval="1h", end_ts=T0, seed=2)
    rows = []
    for coin, df in (("SOL", a), ("ETH", b)):
        for ts, r in df.iterrows():
            rows.append({"time": ts.isoformat(), "symbol": coin.lower(), **r.to_dict()})
    pd.DataFrame(rows).to_csv(tmp_path / "multi.csv", index=False)
    clock = SimClock(T0 - 5 * 3600)                         # reloj en la fecha del CSV
    rep = ReplayMarketData(tmp_path / "multi.csv", clock, offset=0)
    assert sorted(rep.coins()) == ["ETH", "SOL"]
    assert rep.price_at("SOL", clock.now()) == pytest.approx(a.iloc[14]["open"])
    assert rep.price_at("ETH", clock.now()) == pytest.approx(b.iloc[14]["open"])
    df = rep.candles("ETH", "1h", 5)
    assert df.index[-1] == pd.Timestamp(T0 - 5 * 3600, unit="s", tz="UTC")


def test_replay_book_and_liquidations(tmp_path):
    write_csv(tmp_path / "btc.csv")
    clock = SimClock()
    rep = ReplayMarketData(tmp_path / "btc.csv", clock)
    bid, ask = rep.bid_ask("BTC")
    book = rep.l2_book("BTC", n_sig_figs=4)
    assert book["bids"][0][0] <= bid and book["asks"][0][0] >= ask
    mid = rep.price_at("BTC", clock.now())
    for d in rep.liquidation_levels("BTC"):
        assert (d["liq_px"] < mid) if d["side"] == "long" else (d["liq_px"] > mid)


def test_replay_bad_csv(tmp_path):
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(MarketDataError, match="columnas"):
        ReplayMarketData(tmp_path / "bad.csv", SimClock())
    with pytest.raises(MarketDataError):
        ReplayMarketData(tmp_path / "missing.csv", SimClock())

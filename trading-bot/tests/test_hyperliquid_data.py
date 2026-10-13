"""HyperliquidMarketData con una sesión HTTP falsa (los tests nunca tocan la red)."""

from __future__ import annotations

import copy
from typing import Any, Callable, Optional

import pandas as pd
import pytest
import requests

from desk.clock import SimClock
from desk.config import Settings
from desk.exchange.base import CANDLE_COLUMNS, MarketDataError
from desk.exchange.hyperliquid_data import (
    MAINNET_INFO_URL,
    MOONDEV_POSITIONS_URL,
    TESTNET_INFO_URL,
    HttpStatusError,
    HyperliquidMarketData,
)

T0 = 1_767_225_600.0      # 2026-01-01 00:00:00 UTC (SimClock por defecto)

# --- respuestas con la forma real de la API ------------------------------------------------

META = {
    "universe": [
        {"szDecimals": 5, "name": "BTC", "maxLeverage": 40, "marginTableId": 56},
        {"szDecimals": 4, "name": "ETH", "maxLeverage": 25, "marginTableId": 55},
        {"szDecimals": 2, "name": "SOL", "maxLeverage": 20, "marginTableId": 54},
        {"szDecimals": 0, "name": "MATIC", "maxLeverage": 20, "marginTableId": 20, "isDelisted": True},
        {"szDecimals": 2, "name": "HYPE", "maxLeverage": 10, "marginTableId": 52},
    ],
    "marginTables": [[56, {"description": "", "marginTiers": [{"lowerBound": "0.0", "maxLeverage": 40}]}]],
}

META_XYZ = {"universe": [{"szDecimals": 4, "name": "xyz:GOLD", "maxLeverage": 20, "onlyIsolated": True}]}

BOOK_BTC = {
    "coin": "BTC",
    "time": 1_767_225_600_123,
    "levels": [
        [{"px": "83770.0", "sz": "1.53201", "n": 7}, {"px": "83769.0", "sz": "0.25", "n": 2},
         {"px": "83765.0", "sz": "12.0", "n": 31}],
        [{"px": "83771.0", "sz": "0.81234", "n": 3}, {"px": "83772.0", "sz": "2.1", "n": 5}],
    ],
}


def candle_rows(n: int = 4, interval_ms: int = 900_000, end_open_ms: int = int(T0 * 1000)) -> list[dict]:
    rows = []
    for i in range(n):
        t = end_open_ms - (n - 1 - i) * interval_ms
        base = 83_000.0 + 10 * i
        rows.append({"t": t, "T": t + interval_ms - 1, "s": "BTC", "i": "15m",
                     "o": f"{base:.1f}", "c": f"{base + 5:.1f}", "h": f"{base + 20:.1f}",
                     "l": f"{base - 15:.1f}", "v": f"{12.5 + i:.5f}", "n": 300 + i})
    return rows


MOONDEV = {
    "symbols": {
        "BTC": {
            "longs": [
                {"address": "0xabc", "coin": "BTC", "liq_price": 80123.5, "value": 2_500_000.0,
                 "entry_price": 84_000.0, "distance_pct": 4.35, "leverage": 20},
                {"address": "0xdef", "coin": "BTC", "liq_price": None, "value": 10.0},
                {"address": "0x123", "coin": "BTC", "liq_price": 79000.0, "value": 0},
            ],
            "shorts": [{"address": "0x456", "coin": "BTC", "liq_price": "88000.0", "value": "1200000"}],
        },
        "xyz:GOLD": {"longs": [], "shorts": []},
    },
    "total_symbols": 2,
}


# --- sesión falsa ----------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None, headers: Optional[dict] = None,
                 bad_json: bool = False):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self._bad_json = bad_json
        self.text = "<html>error</html>" if bad_json else str(payload)[:100]

    def json(self) -> Any:
        if self._bad_json:
            raise ValueError("Expecting value")
        return copy.deepcopy(self._payload)


class FakeSession:
    """Responde con ``router(method, url, body, headers)``; si el router devuelve una lista,
    se consumen sus elementos en orden (respuestas o excepciones)."""

    def __init__(self, router: Callable[..., Any]):
        self.router = router
        self.calls: list[dict] = []
        self._queues: dict[int, list] = {}

    def _dispatch(self, method: str, url: str, body: Any, headers: Any, timeout: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, "body": copy.deepcopy(body),
                           "headers": headers, "timeout": timeout})
        out = self.router(method, url, body, headers)
        if isinstance(out, list):
            q = self._queues.setdefault(id(out), out)
            out = q.pop(0) if len(q) > 1 else q[0]
        if isinstance(out, Exception):
            raise out
        return out

    def post(self, url, json=None, headers=None, timeout=None):
        return self._dispatch("POST", url, json, headers, timeout)

    def get(self, url, headers=None, timeout=None, params=None):
        return self._dispatch("GET", url, None, headers, timeout)

    def info_calls(self, kind: str) -> list[dict]:
        return [c for c in self.calls if c["body"] and c["body"].get("type") == kind]


def default_router(method, url, body, headers):
    if method == "GET":
        return FakeResponse(200, MOONDEV)
    kind = body["type"]
    if kind == "meta":
        return FakeResponse(200, META_XYZ if body.get("dex") == "xyz" else META)
    if kind == "l2Book":
        return FakeResponse(200, BOOK_BTC if body["coin"] == "BTC" else None)
    if kind == "candleSnapshot":
        return FakeResponse(200, candle_rows())
    return FakeResponse(400, {"error": "unknown"})


@pytest.fixture
def clock():
    return SimClock(T0)


def make(clock, router=default_router, **kw) -> tuple[HyperliquidMarketData, FakeSession]:
    session = FakeSession(router)
    kw.setdefault("moondev_api_key", "")
    return HyperliquidMarketData(clock, session=session, **kw), session


# --- meta ------------------------------------------------------------------------------------

def test_meta_parsing_and_request_shape(clock):
    md, s = make(clock)
    m = md.meta("BTC")
    assert (m.coin, m.sz_decimals, m.max_leverage, m.min_notional_usd) == ("BTC", 5, 40, 10.0)
    assert md.meta("HYPE").max_leverage == 10
    assert md.coins() == ["BTC", "ETH", "SOL", "HYPE"]           # MATIC está delisted
    call = s.calls[0]
    assert call["method"] == "POST" and call["url"] == MAINNET_INFO_URL
    assert call["body"] == {"type": "meta"} and call["timeout"] == 10.0


def test_meta_is_cached_and_refreshed_after_ttl(clock):
    md, s = make(clock, meta_ttl=3600)
    for _ in range(5):
        md.meta("BTC")
        md.coins()
    assert len(s.info_calls("meta")) == 1
    clock.advance(3601)
    md.meta("ETH")
    assert len(s.info_calls("meta")) == 2


def test_unknown_or_delisted_coin_raises_keyerror_with_bounded_refresh(clock):
    md, s = make(clock)
    with pytest.raises(KeyError):
        md.meta("MATIC")
    with pytest.raises(KeyError):
        md.meta("DOGE")
    assert len(s.info_calls("meta")) == 1                       # recién cargado: no refresca
    clock.advance(61)
    with pytest.raises(KeyError):
        md.meta("DOGE")
    assert len(s.info_calls("meta")) == 2                       # moneda nueva: un refresco por minuto


def test_hip3_meta_uses_dex(clock):
    md, s = make(clock)
    gold = md.meta("xyz:GOLD")
    assert gold.sz_decimals == 4 and gold.max_leverage == 20
    assert s.info_calls("meta")[0]["body"] == {"type": "meta", "dex": "xyz"}
    assert "xyz:GOLD" not in md.coins()


def test_stale_meta_used_when_refresh_fails(clock):
    state = {"fail": False}

    def router(method, url, body, headers):
        if state["fail"]:
            return FakeResponse(503)
        return default_router(method, url, body, headers)

    md, s = make(clock, router, retries=0)
    md.meta("BTC")
    state["fail"] = True
    clock.advance(4000)
    assert md.meta("BTC").sz_decimals == 5


def test_testnet_and_from_settings(clock):
    md, s = make(clock, testnet=True)
    md.coins()
    assert s.calls[0]["url"] == TESTNET_INFO_URL
    paper = HyperliquidMarketData.from_settings(Settings(mode="paper"), clock, session=FakeSession(default_router))
    assert paper.info_url == MAINNET_INFO_URL                  # paper: precios reales de mainnet
    live = HyperliquidMarketData.from_settings(Settings(mode="live"), clock, session=FakeSession(default_router))
    assert live.info_url == TESTNET_INFO_URL                   # live: testnet salvo hl_mainnet
    live_main = HyperliquidMarketData.from_settings(Settings(mode="live", hl_mainnet=True, moondev_api_key="k"),
                                                    clock, session=FakeSession(default_router))
    assert live_main.info_url == MAINNET_INFO_URL and live_main.moondev_api_key == "k"


# --- libro -----------------------------------------------------------------------------------

def test_l2_book_and_bid_ask(clock):
    md, s = make(clock)
    book = md.l2_book("BTC")
    assert book["bids"][0] == (83770.0, 1.53201) and book["asks"][0] == (83771.0, 0.81234)
    assert len(book["bids"]) == 3 and len(book["asks"]) == 2
    assert book["ts"] == T0 and book["exchange_ts"] == pytest.approx(1_767_225_600.123)
    assert all(isinstance(px, float) and isinstance(sz, float) for px, sz in book["bids"])
    assert md.bid_ask("BTC") == (83770.0, 83771.0)
    assert md.mid("BTC") == 83770.5
    assert md.last_update_ts("BTC") == T0
    assert s.info_calls("l2Book")[0]["body"] == {"type": "l2Book", "coin": "BTC"}


def test_l2_book_n_sig_figs_and_short_cache(clock):
    md, s = make(clock, book_ttl=1.0)
    md.l2_book("BTC", n_sig_figs=3)
    md.l2_book("BTC", n_sig_figs=3)                           # cache del mismo ciclo
    md.l2_book("BTC", n_sig_figs=9)                           # se acota a 5
    calls = s.info_calls("l2Book")
    assert len(calls) == 2
    assert calls[0]["body"] == {"type": "l2Book", "coin": "BTC", "nSigFigs": 3}
    assert calls[1]["body"]["nSigFigs"] == 5
    clock.advance(1.5)
    md.l2_book("BTC", n_sig_figs=3)
    assert len(s.info_calls("l2Book")) == 3
    # El libro falso lleva siempre time = T0 + 0.123: la edad del dato es la del exchange, no la de la petición.
    assert md.last_update_ts("BTC") == pytest.approx(T0 + 0.123, abs=1e-6)
    assert md.last_update_ts("ETH") is None


def test_book_errors(clock):
    def router(method, url, body, headers):
        if body["type"] == "l2Book" and body["coin"] == "THIN":
            return FakeResponse(200, {"coin": "THIN", "time": 1, "levels": [[{"px": "1.0", "sz": "2", "n": 1}], []]})
        if body["type"] == "l2Book" and body["coin"] == "BAD":
            return FakeResponse(200, {"coin": "BAD", "levels": [[{"px": "x", "sz": "2"}], []]})
        return default_router(method, url, body, headers)

    md, _ = make(clock, router)
    with pytest.raises(MarketDataError, match="vacío"):
        md.bid_ask("THIN")
    with pytest.raises(MarketDataError, match="malformado"):
        md.l2_book("BAD")
    with pytest.raises(MarketDataError, match="levels"):
        md.l2_book("NOPE")                                     # Hyperliquid responde null


# --- velas -----------------------------------------------------------------------------------

def test_candles_request_and_dataframe(clock):
    clock.advance(400)                                         # a mitad de una vela de 15m
    md, s = make(clock)
    df = md.candles("BTC", "15m", 3)
    body = s.info_calls("candleSnapshot")[0]["body"]
    assert body["req"]["coin"] == "BTC" and body["req"]["interval"] == "15m"
    assert body["req"]["endTime"] == int((T0 + 400) * 1000)
    assert body["req"]["startTime"] == int((T0 - 3 * 900) * 1000)
    assert list(df.columns) == CANDLE_COLUMNS
    assert all(df[c].dtype == float for c in CANDLE_COLUMNS)
    assert str(df.index.tz) == "UTC" and df.index.is_monotonic_increasing
    assert df.index.unit == "ns"
    assert len(df) == 3                                        # 4 recibidas, tail(3)
    assert df.index[-1] == pd.Timestamp(T0, unit="s", tz="UTC")
    last = df.iloc[-1]
    assert (last.open, last.high, last.low, last.close, last.volume) == (83030.0, 83050.0, 83015.0, 83035.0, 15.5)
    assert md.last_update_ts("BTC") == T0 + 400


def test_candles_dedupe_sort_and_empty(clock):
    rows = candle_rows(3)
    shuffled = [rows[2], rows[0], rows[1], dict(rows[2], c="99999.0")]

    def router(method, url, body, headers):
        if body["type"] == "candleSnapshot":
            return FakeResponse(200, [] if body["req"]["coin"] == "EMPTY" else shuffled)
        return default_router(method, url, body, headers)

    md, _ = make(clock, router)
    df = md.candles("BTC", "15m", 10)
    assert len(df) == 3 and df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 99999.0                     # duplicado: gana el último
    empty = md.candles("EMPTY", "1h", 10)
    assert empty.empty and list(empty.columns) == CANDLE_COLUMNS and str(empty.index.tz) == "UTC"


def test_candles_validation(clock):
    def router(method, url, body, headers):
        if body["type"] == "candleSnapshot":
            return FakeResponse(200, [{"t": 1, "o": "1"}])
        return default_router(method, url, body, headers)

    md, s = make(clock, router)
    with pytest.raises(ValueError):
        md.candles("BTC", "7m", 10)
    with pytest.raises(MarketDataError, match="malformada"):
        md.candles("BTC", "1h", 10)
    body = s.info_calls("candleSnapshot")[0]["body"]
    assert body["req"]["startTime"] == int((T0 - 10 * 3600) * 1000)


def test_candles_limit_capped_at_5000(clock):
    md, s = make(clock)
    md.candles("BTC", "1m", 100_000)
    req = s.info_calls("candleSnapshot")[0]["body"]["req"]
    assert req["endTime"] - req["startTime"] == 5000 * 60_000


# --- transporte: reintentos y errores ----------------------------------------------------------

def test_retries_with_backoff_on_429_and_5xx(clock):
    seq = [FakeResponse(429), FakeResponse(502), FakeResponse(200, META)]
    md, s = make(clock, lambda *a: seq, backoff_base=0.5)
    assert md.meta("BTC").sz_decimals == 5
    assert len(s.calls) == 3
    assert clock.now() == T0 + 0.5 + 1.0                       # esperas con el Clock inyectado


def test_retry_after_header_is_honoured(clock):
    seq = [FakeResponse(429, headers={"Retry-After": "3"}), FakeResponse(200, META)]
    md, _ = make(clock, lambda *a: seq)
    md.coins()
    assert clock.now() == T0 + 3.0


def test_connection_errors_are_retried(clock):
    seq = [requests.ConnectionError("reset"), requests.Timeout("lento"), FakeResponse(200, META)]
    md, s = make(clock, lambda *a: seq)
    assert "BTC" in md.coins()
    assert len(s.calls) == 3


def test_retries_exhausted_raise_market_data_error(clock):
    md, s = make(clock, lambda *a: FakeResponse(500), retries=3)
    with pytest.raises(MarketDataError, match="4 intentos"):
        md.bid_ask("BTC")
    assert len(s.calls) == 4
    assert clock.now() == T0 + 0.5 + 1.0 + 2.0


def test_client_errors_fail_fast(clock):
    md, s = make(clock, lambda *a: FakeResponse(422, {"error": "bad"}))
    with pytest.raises(HttpStatusError) as ei:
        md.l2_book("BTC")
    assert ei.value.status == 422 and isinstance(ei.value, MarketDataError)
    assert len(s.calls) == 1 and clock.now() == T0


def test_invalid_json_raises(clock):
    md, _ = make(clock, lambda *a: FakeResponse(200, bad_json=True))
    with pytest.raises(MarketDataError, match="JSON"):
        md.coins()


# --- liquidaciones (Moon Dev) ------------------------------------------------------------------

def test_liquidations_without_key_make_no_request(clock, monkeypatch):
    monkeypatch.delenv("MOONDEV_API_KEY", raising=False)
    md, s = make(clock, moondev_api_key=None)
    assert md.liquidation_levels("BTC") == []
    assert s.calls == []


def test_liquidations_parsed_from_moondev(clock):
    md, s = make(clock, moondev_api_key="secret")
    levels = md.liquidation_levels("BTC")
    assert levels == [
        {"liq_px": 80123.5, "usd": 2_500_000.0, "side": "long"},
        {"liq_px": 88000.0, "usd": 1_200_000.0, "side": "short"},
    ]
    call = s.calls[0]
    assert call["method"] == "GET" and call["url"] == MOONDEV_POSITIONS_URL
    assert call["headers"] == {"X-API-Key": "secret"} and call["timeout"] == 10.0
    assert md.liquidation_levels("xyz:GOLD") == []
    assert md.liquidation_levels("DOGE") == []
    assert len(s.calls) == 1                                    # un único JSON para todas las monedas


def test_liquidations_env_key_and_30s_cache(clock, monkeypatch):
    monkeypatch.setenv("MOONDEV_API_KEY", " env-key ")
    md, s = make(clock, moondev_api_key=None)
    md.liquidation_levels("BTC")
    clock.advance(20)
    md.liquidation_levels("BTC")
    assert len(s.calls) == 1
    clock.advance(11)
    md.liquidation_levels("BTC")
    assert len(s.calls) == 2
    assert s.calls[0]["headers"] == {"X-API-Key": "env-key"}


def test_liquidations_degrade_gracefully(clock):
    state = {"status": 200}

    def router(method, url, body, headers):
        if method == "GET":
            return FakeResponse(state["status"], MOONDEV)
        return default_router(method, url, body, headers)

    md, s = make(clock, router, moondev_api_key="k")
    assert len(md.liquidation_levels("BTC")) == 2
    state["status"] = 503
    clock.advance(31)
    assert len(md.liquidation_levels("BTC")) == 2               # cache viejo (< 10 min) en vez de fallar
    assert "503" in md.liq_error
    n = len(s.calls)
    clock.advance(10)
    md.liquidation_levels("BTC")
    assert len(s.calls) == n                                    # en pausa tras el fallo
    clock.advance(700)
    assert md.liquidation_levels("BTC") == []                   # cache demasiado viejo: sin datos
    state["status"] = 200
    clock.advance(31)
    assert len(md.liquidation_levels("BTC")) == 2 and md.liq_error == ""


def test_liquidations_invalid_key_pauses_longer(clock):
    md, s = make(clock, lambda m, u, b, h: FakeResponse(401, {"error": "invalid_api_key"}), moondev_api_key="bad")
    assert md.liquidation_levels("BTC") == []
    assert len(s.calls) == 1                                    # 401 no se reintenta
    clock.advance(120)
    assert md.liquidation_levels("BTC") == []
    assert len(s.calls) == 1                                    # pausa de 10 min con clave inválida
    clock.advance(500)
    md.liquidation_levels("BTC")
    assert len(s.calls) == 2


# --- regresiones (revisión) -------------------------------------------------------------------

def test_frozen_book_reports_its_exchange_age(clock):
    # Un nodo atrasado sirve un libro congelado de hace 1 h: la edad que ve riesgo es la del dato, no la de la
    # petición (antes salía ~0 s y 'long BTC a mercado' se aprobaba con max_data_age_seconds=120).
    frozen = dict(BOOK_BTC, time=int((T0 - 3600) * 1000))

    def router(method, url, body, headers):
        if body and body.get("type") == "l2Book":
            return FakeResponse(200, frozen)
        return default_router(method, url, body, headers)
    md, _ = make(clock, router)
    md.bid_ask("BTC")
    assert clock.now() - md.last_update_ts("BTC") == pytest.approx(3600.0)


def test_book_time_ahead_of_local_clock_is_capped_at_request_time(clock):
    ahead = dict(BOOK_BTC, time=int((T0 + 30) * 1000))      # reloj del exchange por delante del local

    def router(method, url, body, headers):
        if body and body.get("type") == "l2Book":
            return FakeResponse(200, ahead)
        return default_router(method, url, body, headers)
    md, _ = make(clock, router)
    md.bid_ask("BTC")
    assert md.last_update_ts("BTC") == pytest.approx(T0, abs=1e-6)

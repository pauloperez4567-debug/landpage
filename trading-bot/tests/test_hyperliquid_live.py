"""LiveHyperliquidBroker con Exchange/Info falsos: nunca toca la red ni firma nada real."""

from __future__ import annotations

import logging

import pytest
import requests

from desk.config import LIVE_CONFIRM_PHRASE, load_settings
from desk.exchange import hyperliquid_live as hl
from desk.exchange.base import BrokerError
from desk.exchange.hyperliquid_live import LiveHyperliquidBroker
from desk.models import OrderKind, OrderStatus

# Clave de prueba (no es una cuenta real) y una dirección de cuenta principal distinta.
KEY = "0x" + "4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"
ADDR = "0x" + "ab" * 20

MARKET, LIMIT, STOP, TP = OrderKind.MARKET, OrderKind.LIMIT, OrderKind.STOP, OrderKind.TAKE_PROFIT


def ok(*statuses, kind="order"):
    return {"status": "ok", "response": {"type": kind, "data": {"statuses": list(statuses)}}}


class FakeExchange:
    """Registra las llamadas con la firma del SDK y devuelve respuestas encoladas."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.responses: list = []
        self.raise_next: Exception | None = None
        self.next_oid = 1000

    def _reply(self, default):
        if self.raise_next is not None:
            e, self.raise_next = self.raise_next, None
            raise e
        return self.responses.pop(0) if self.responses else default

    def order(self, name, is_buy, sz, limit_px, order_type, reduce_only=False, cloid=None, builder=None):
        self.calls.append(("order", name, is_buy, sz, limit_px, order_type, reduce_only))
        self.next_oid += 1
        return self._reply(ok({"resting": {"oid": self.next_oid}}))

    def cancel(self, name, oid):
        self.calls.append(("cancel", name, oid))
        return self._reply(ok("success", kind="cancel"))

    def bulk_cancel(self, cancel_requests):
        self.calls.append(("bulk_cancel", cancel_requests))
        return self._reply(ok(*["success"] * len(cancel_requests), kind="cancel"))

    def update_leverage(self, leverage, name, is_cross=True):
        self.calls.append(("update_leverage", leverage, name, is_cross))
        return self._reply({"status": "ok", "response": {"type": "default"}})


class FakeInfo:
    def __init__(self):
        self.calls: list[tuple] = []
        self.state: dict = {"marginSummary": {"accountValue": "10000.0"}, "withdrawable": "10000.0",
                            "assetPositions": []}
        self.orders: list[dict] = []
        self.fills: list[dict] = []
        self.page = 2000
        self.fail: Exception | None = None

    def _check(self):
        if self.fail is not None:
            raise self.fail

    def user_state(self, address, dex=""):
        self.calls.append(("user_state", address, dex))
        self._check()
        return self.state

    def frontend_open_orders(self, address, dex=""):
        self.calls.append(("frontend_open_orders", address, dex))
        self._check()
        return list(self.orders)

    def user_fills_by_time(self, address, start_time, end_time=None, aggregate_by_time=False):
        self.calls.append(("user_fills_by_time", address, start_time))
        self._check()
        rows = sorted((f for f in self.fills if f["time"] >= start_time), key=lambda f: f["time"])
        return rows[:self.page]

    def meta(self, dex=""):
        self.calls.append(("meta", dex))
        return {"universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 40},
                             {"name": "SOL", "szDecimals": 2, "maxLeverage": 20},
                             {"name": "OLD", "szDecimals": 0, "maxLeverage": 3, "isDelisted": True}]}

    def all_mids(self, dex=""):
        self.calls.append(("all_mids", dex))
        return {"BTC": "100000.0", "SOL": "150.0"}


@pytest.fixture
def live_settings(tmp_path):
    return load_settings(env={}, dotenv_path=None, state_dir=str(tmp_path / "state"), llm_enabled=False,
                         mode="live", live_confirm=LIVE_CONFIRM_PHRASE, hl_private_key=KEY,
                         hl_account_address=ADDR)


@pytest.fixture
def fx():
    return FakeExchange()


@pytest.fixture
def fi():
    return FakeInfo()


@pytest.fixture
def broker(market, live_settings, clock, fx, fi):
    market.set_price("SOL", 150.0)                  # mid exacto 150
    return LiveHyperliquidBroker(market, live_settings, clock, exchange=fx, info=fi)


def fill_row(tid, oid, t, *, coin="SOL", side="B", sz="1.0", px="150.0", fee="0.0675", pnl="0.0", **extra):
    return {"coin": coin, "px": px, "sz": sz, "side": side, "time": t, "oid": oid, "tid": tid, "fee": fee,
            "closedPnl": pnl, "dir": "Open Long", "crossed": True, "hash": f"0x{tid:x}", **extra}


# --------------------------------------------------------------------------- construcción

def test_paper_settings_cannot_build_live_broker(market, settings, clock, fx, fi):
    with pytest.raises(ValueError, match="modo live"):
        LiveHyperliquidBroker(market, settings, clock, exchange=fx, info=fi)


def test_live_without_confirmation_is_refused(market, live_settings, clock, fx, fi):
    live_settings.live_confirm = ""
    with pytest.raises(ValueError, match="DESK_LIVE_CONFIRM"):
        LiveHyperliquidBroker(market, live_settings, clock, exchange=fx, info=fi)
    live_settings.live_confirm = "yes please"
    with pytest.raises(ValueError, match="Modo live bloqueado"):
        LiveHyperliquidBroker(market, live_settings, clock, exchange=fx, info=fi)
    assert fx.calls == [] and fi.calls == []


def test_live_without_key_or_with_wrong_data_is_refused(market, live_settings, clock, fx, fi):
    live_settings.hl_private_key = ""
    with pytest.raises(ValueError, match="HL_PRIVATE_KEY"):
        LiveHyperliquidBroker(market, live_settings, clock, exchange=fx, info=fi)
    live_settings.hl_private_key = KEY
    live_settings.data_source = "sim"
    with pytest.raises(ValueError, match="DESK_DATA"):
        LiveHyperliquidBroker(market, live_settings, clock, exchange=fx, info=fi)


def test_testnet_by_default_mainnet_only_when_asked(market, live_settings, clock, fx, fi):
    b = LiveHyperliquidBroker(market, live_settings, clock, exchange=fx, info=fi)
    assert b.testnet and b.base_url == hl.TESTNET_API_URL
    live_settings.hl_mainnet = True
    b = LiveHyperliquidBroker(market, live_settings, clock, exchange=fx, info=fi)
    assert not b.testnet and b.base_url == hl.MAINNET_API_URL


def test_url_constants_match_sdk():
    from hyperliquid.utils import constants
    assert hl.MAINNET_API_URL == constants.MAINNET_API_URL
    assert hl.TESTNET_API_URL == constants.TESTNET_API_URL


def test_invalid_account_address_refused(market, live_settings, clock, fx, fi):
    live_settings.hl_account_address = "0x1234"
    with pytest.raises(ValueError, match="HL_ACCOUNT_ADDRESS"):
        LiveHyperliquidBroker(market, live_settings, clock, exchange=fx, info=fi)


class _RecordingExchange(FakeExchange):
    """Sustituye a ``hyperliquid.exchange.Exchange`` para probar la construcción con el SDK."""

    last: "_RecordingExchange | None" = None

    def __init__(self, wallet, base_url=None, meta=None, vault_address=None, account_address=None,
                 spot_meta=None, perp_dexs=None, timeout=None):
        super().__init__()
        self.wallet, self.base_url, self.account_address = wallet, base_url, account_address
        self.perp_dexs, self.timeout = perp_dexs, timeout
        self.info = FakeInfo()
        _RecordingExchange.last = self


def test_builds_with_sdk_lazily_and_never_leaks_key(monkeypatch, caplog, market, live_settings, clock):
    import hyperliquid.exchange
    from eth_account import Account
    monkeypatch.setattr(hyperliquid.exchange, "Exchange", _RecordingExchange)
    caplog.set_level(logging.DEBUG)
    wallet_addr = Account.from_key(KEY).address

    b = LiveHyperliquidBroker(market, live_settings, clock)
    ex = _RecordingExchange.last
    assert ex.wallet.address == wallet_addr
    assert ex.base_url == "https://api.hyperliquid-testnet.xyz"
    assert ex.account_address == ADDR               # la API wallet firma para la cuenta principal
    assert ex.perp_dexs is None and ex.timeout == 10.0
    assert b.address == ADDR and b.info is ex.info  # reutiliza el Info del Exchange
    secret = KEY[2:].lower()
    assert secret not in repr(b).lower() and secret not in str(vars(b)).lower()
    assert all(secret not in r.getMessage().lower() for r in caplog.records)

    live_settings.hl_account_address = ""
    b = LiveHyperliquidBroker(market, live_settings, clock)
    assert b.address == wallet_addr and _RecordingExchange.last.account_address is None


def test_hip3_symbols_load_their_dexes(monkeypatch, market, live_settings, clock):
    import hyperliquid.exchange
    monkeypatch.setattr(hyperliquid.exchange, "Exchange", _RecordingExchange)
    live_settings.symbols = ["BTC", "xyz:GOLD"]
    b = LiveHyperliquidBroker(market, live_settings, clock)
    assert b.dexes == ["", "xyz"] and _RecordingExchange.last.perp_dexs == ["", "xyz"]


def test_invalid_private_key_error_hides_key(market, live_settings, clock):
    bad = "0x" + "zz" * 32
    live_settings.hl_private_key = bad
    live_settings.hl_account_address = ""
    with pytest.raises(ValueError) as exc:
        LiveHyperliquidBroker(market, live_settings, clock)
    assert "zz" not in str(exc.value) and exc.value.__cause__ is None and exc.value.__suppress_context__


def test_sdk_connection_failure_is_broker_error(monkeypatch, market, live_settings, clock):
    import hyperliquid.exchange

    def boom(*a, **k):
        raise requests.ConnectionError("sin red")
    monkeypatch.setattr(hyperliquid.exchange, "Exchange", boom)
    with pytest.raises(BrokerError, match="no se pudo conectar"):
        LiveHyperliquidBroker(market, live_settings, clock)


# --------------------------------------------------------------------------- órdenes

def test_limit_order_mapping(broker, fx):
    o = broker.place_order("HYPE", True, 11.734, LIMIT, price=85.2491, tif="Alo", tag="entry")
    assert fx.calls == [("order", "HYPE", True, 11.73, 85.249, {"limit": {"tif": "Alo"}}, False)]
    assert o.status is OrderStatus.OPEN and o.id == "1001"
    assert (o.size, o.price, o.tif, o.tag, o.kind) == (11.73, 85.249, "Alo", "entry", LIMIT)


def test_market_order_is_aggressive_ioc_around_mid(broker, fx):
    fx.responses.append(ok({"filled": {"totalSz": "2.0", "avgPx": "150.03", "oid": 77}}))
    o = broker.place_order("SOL", True, 2, MARKET, tag="entry")
    assert fx.calls[-1] == ("order", "SOL", True, 2.0, 150.15, {"limit": {"tif": "Ioc"}}, False)
    assert o.status is OrderStatus.FILLED and o.id == "77"
    assert (o.filled_size, o.avg_fill_px, o.price, o.tif) == (2.0, 150.03, 150.15, "Ioc")

    # Un cierre (reduce-only sin precio) no usa los 10 bps de las entradas sino CLOSE_SLIPPAGE (5 %): con
    # spread > 20 bps un IOC a mid - 10 bps no casaría nunca.
    broker.place_order("SOL", False, 2, MARKET, reduce_only=True, tag="close")
    assert fx.calls[-1] == ("order", "SOL", False, 2.0, 142.5, {"limit": {"tif": "Ioc"}}, True)


def test_market_order_with_worst_price(broker, fx):
    broker.place_order("SOL", True, 1, MARKET, price=151.237)
    assert fx.calls[-1][4] == 151.24 and fx.calls[-1][5] == {"limit": {"tif": "Ioc"}}


def test_without_market_uses_info_meta_and_mids(live_settings, clock, fx, fi):
    b = LiveHyperliquidBroker(None, live_settings, clock, exchange=fx, info=fi)
    b.place_order("SOL", False, 1.239, MARKET)
    assert fx.calls[-1] == ("order", "SOL", False, 1.23, 149.85, {"limit": {"tif": "Ioc"}}, False)
    assert ("meta", "") in fi.calls and ("all_mids", "") in fi.calls
    o = b.place_order("OLD", True, 1, LIMIT, price=1.0)        # delisted: no está en meta
    assert o.status is OrderStatus.REJECTED and "desconocida" in o.reason


def test_stop_loss_mapping(broker, fx):
    o = broker.place_order("SOL", False, 1.5, STOP, trigger_px=145.123, reduce_only=True, tag="sl")
    assert fx.calls[-1] == ("order", "SOL", False, 1.5, 137.86,
                            {"trigger": {"triggerPx": 145.12, "isMarket": True, "tpsl": "sl"}}, True)
    assert o.status is OrderStatus.OPEN and o.kind is STOP
    assert (o.trigger_px, o.price, o.reduce_only, o.tag) == (145.12, None, True, "sl")


def test_take_profit_mapping_for_short(broker, fx):
    o = broker.place_order("SOL", True, 2, TP, trigger_px=140.0, reduce_only=True, tag="tp")
    assert fx.calls[-1] == ("order", "SOL", True, 2.0, 147.0,
                            {"trigger": {"triggerPx": 140.0, "isMarket": True, "tpsl": "tp"}}, True)
    assert o.kind is TP and o.status is OrderStatus.OPEN


@pytest.mark.parametrize("kwargs,reason", [
    ({"kind": STOP, "trigger_px": 140.0}, "reduce_only"),
    ({"kind": TP, "reduce_only": True}, "trigger_px"),
    ({"kind": LIMIT}, "price"),
    ({"kind": LIMIT, "price": 150.0, "tif": "Fok"}, "tif"),
    ({"kind": LIMIT, "price": 150.0, "size": 0.004}, "mínimo"),
    ({"kind": LIMIT, "price": 150.0, "coin": "DOGE"}, "desconocida"),
    ({"kind": MARKET, "price": -1.0}, "price"),
])
def test_local_rejections_never_reach_exchange(broker, fx, kwargs, reason):
    kw = dict(kwargs)
    coin, size, kind = kw.pop("coin", "SOL"), kw.pop("size", 1.0), kw.pop("kind")
    o = broker.place_order(coin, False, size, kind, **kw)
    assert o.status is OrderStatus.REJECTED and reason in o.reason
    assert o.id.startswith("hl-local-") and fx.calls == []


def test_exchange_rejections_map_to_rejected(broker, fx):
    fx.responses.append(ok({"error": "Order must have minimum value of $10. asset=5"}))
    o = broker.place_order("SOL", True, 0.05, LIMIT, price=150.0)
    assert o.status is OrderStatus.REJECTED and "minimum value" in o.reason

    fx.responses.append({"status": "err", "response": "User or API Wallet does not exist."})
    o = broker.place_order("SOL", True, 1, LIMIT, price=150.0)
    assert o.status is OrderStatus.REJECTED and "does not exist" in o.reason

    fx.responses.append({"status": "ok", "response": {"type": "order", "data": {"statuses": []}}})
    assert broker.place_order("SOL", True, 1, LIMIT, price=150.0).status is OrderStatus.REJECTED

    fx.raise_next = KeyError("SOL")                 # el SDK no conoce la moneda
    o = broker.place_order("SOL", True, 1, LIMIT, price=150.0)
    assert o.status is OrderStatus.REJECTED and "SDK" in o.reason


def test_transport_errors_raise_broker_error(broker, fx):
    from hyperliquid.utils.error import ClientError, ServerError
    for exc in (requests.ConnectionError("reset"), requests.Timeout("lento"),
                ServerError(502, "bad gateway"), ClientError(429, None, "rate limited", None)):
        fx.raise_next = exc
        with pytest.raises(BrokerError):
            broker.place_order("SOL", True, 1, LIMIT, price=150.0)
    fx.raise_next = ServerError(502, "bad gateway")
    with pytest.raises(BrokerError, match="502"):
        broker.place_order("SOL", True, 1, LIMIT, price=150.0)


def test_partial_fills(broker, fx):
    fx.responses.append(ok({"filled": {"totalSz": "1.5", "avgPx": "150.1", "oid": 5}}))
    o = broker.place_order("SOL", True, 2, MARKET)
    assert o.status is OrderStatus.FILLED and o.filled_size == 1.5 and "parcialmente" in o.reason
    fx.responses.append(ok({"filled": {"totalSz": "0.5", "avgPx": "150.0", "oid": 6}}))
    o = broker.place_order("SOL", True, 2, LIMIT, price=150.0)
    assert o.status is OrderStatus.OPEN and o.filled_size == 0.5 and o.remaining == 1.5


def test_grouped_waiting_status_is_open(broker, fx):
    fx.responses.append(ok("waitingForTrigger"))
    o = broker.place_order("SOL", False, 1, STOP, trigger_px=140.0, reduce_only=True)
    assert o.status is OrderStatus.OPEN and o.reason == "waitingForTrigger"


# --------------------------------------------------------------------------- cancelación y apalancamiento

def test_cancel(broker, fx):
    assert broker.cancel("SOL", "1001") is True
    assert fx.calls[-1] == ("cancel", "SOL", 1001)
    fx.responses.append(ok({"error": "Order was never placed, already canceled, or filled."}, kind="cancel"))
    assert broker.cancel("SOL", "1002") is False
    n = len(fx.calls)
    assert broker.cancel("SOL", "hl-local-000001") is False and len(fx.calls) == n
    fx.raise_next = requests.ConnectionError("x")
    with pytest.raises(BrokerError):
        broker.cancel("SOL", "1003")


def test_cancel_all_uses_bulk_cancel(broker, fx, fi):
    fi.orders = [
        {"coin": "SOL", "oid": 11, "side": "B", "sz": "1", "origSz": "1", "limitPx": "140", "orderType": "Limit",
         "isTrigger": False, "reduceOnly": False, "tif": "Gtc", "timestamp": 1},
        {"coin": "BTC", "oid": 12, "side": "A", "sz": "0.1", "origSz": "0.1", "limitPx": "90000",
         "orderType": "Stop Market", "isTrigger": True, "triggerPx": "95000", "reduceOnly": True, "timestamp": 2},
    ]
    assert broker.cancel_all() == 2
    assert fx.calls[-1] == ("bulk_cancel", [{"coin": "SOL", "oid": 11}, {"coin": "BTC", "oid": 12}])
    assert broker.cancel_all("SOL") == 1
    assert fx.calls[-1] == ("bulk_cancel", [{"coin": "SOL", "oid": 11}])
    fi.orders = []
    n = len(fx.calls)
    assert broker.cancel_all() == 0 and len(fx.calls) == n


def test_set_leverage(broker, fx):
    broker.set_leverage("SOL", 3)
    assert fx.calls[-1] == ("update_leverage", 3, "SOL", False)          # aislado por defecto
    broker.set_leverage("BTC", 10, cross=True)
    assert fx.calls[-1] == ("update_leverage", 10, "BTC", True)
    assert broker.leverage == {"SOL": 3, "BTC": 10}
    n = len(fx.calls)
    for coin, lev in (("HYPE", 3), ("SOL", 2.5), ("SOL", 0), ("DOGE", 2)):
        with pytest.raises(ValueError):
            broker.set_leverage(coin, lev)
    assert len(fx.calls) == n
    fx.responses.append({"status": "err", "response": "Cannot decrease leverage with open position"})
    with pytest.raises(ValueError, match="decrease"):
        broker.set_leverage("SOL", 2)
    fx.raise_next = requests.ConnectionError("x")
    with pytest.raises(BrokerError):
        broker.set_leverage("SOL", 2)


# --------------------------------------------------------------------------- cuenta y órdenes abiertas

USER_STATE = {
    "marginSummary": {"accountValue": "10234.5", "totalMarginUsed": "520.0", "totalNtlPos": "2600.0",
                      "totalRawUsd": "7634.5"},
    "withdrawable": "9714.5",
    "assetPositions": [
        {"type": "oneWay", "position": {
            "coin": "SOL", "szi": "10.0", "entryPx": "148.0", "positionValue": "1500.0",
            "unrealizedPnl": "20.0", "returnOnEquity": "0.067", "liquidationPx": "120.5",
            "leverage": {"type": "isolated", "value": 5, "rawUsd": "-1180.0"}, "marginUsed": "300.0"}},
        {"type": "oneWay", "position": {
            "coin": "BTC", "szi": "-0.011", "entryPx": "101000.0", "positionValue": "1100.0",
            "unrealizedPnl": "11.0", "liquidationPx": None, "leverage": {"type": "cross", "value": 5},
            "marginUsed": "220.0"}},
        {"type": "oneWay", "position": {"coin": "ETH", "szi": "0.0", "entryPx": None, "positionValue": "0.0",
                                        "unrealizedPnl": "0.0", "leverage": {"type": "cross", "value": 3},
                                        "marginUsed": "0.0"}},
    ],
}


def test_account_mapping(broker, fi, clock):
    fi.state = USER_STATE
    fi.orders = [{"coin": "SOL", "oid": 21, "side": "A", "sz": "10.0", "origSz": "10.0", "limitPx": "114.0",
                  "orderType": "Stop Market", "isTrigger": True, "triggerPx": "120.0", "reduceOnly": True,
                  "tif": None, "timestamp": 1_767_225_000_000}]
    a = broker.account()
    assert a.equity == 10_234.5 and a.available == 9_714.5 and a.ts == clock.now()
    assert set(a.positions) == {"SOL", "BTC"}
    sol, btc = a.positions["SOL"], a.positions["BTC"]
    assert (sol.size, sol.entry_px, sol.leverage, sol.unrealized_pnl) == (10.0, 148.0, 5, 20.0)
    assert sol.mark_px == pytest.approx(150.0) and sol.liq_px == 120.5 and sol.margin_used == 300.0
    assert btc.size == -0.011 and not btc.is_long and btc.liq_px is None
    assert btc.mark_px == pytest.approx(100_000.0)
    assert [o.id for o in a.open_orders] == ["21"] and a.open_orders[0].kind is STOP
    assert fi.calls[0] == ("user_state", ADDR, "")

    opened = sol.opened_ts
    clock.advance(60)
    assert broker.account().positions["SOL"].opened_ts == opened           # estable entre lecturas
    fi.state = {"marginSummary": {"accountValue": "10000"}, "withdrawable": "10000", "assetPositions": []}
    assert broker.account().positions == {}
    fi.state = USER_STATE
    clock.advance(60)
    assert broker.account().positions["SOL"].opened_ts == clock.now()      # posición nueva


def test_account_errors_are_broker_errors(broker, fi):
    fi.state = {"assetPositions": []}               # falta marginSummary
    with pytest.raises(BrokerError, match="malformada"):
        broker.account()
    fi.fail = requests.ConnectionError("caída")
    with pytest.raises(BrokerError, match="user_state"):
        broker.account()


def test_open_orders_mapping_and_tags(broker, fx, fi):
    entry = broker.place_order("SOL", True, 1, LIMIT, price=140.0, tag="entry")
    fi.orders = [
        {"coin": "SOL", "oid": int(entry.id), "side": "B", "sz": "0.4", "origSz": "1.0", "limitPx": "140.0",
         "orderType": "Limit", "isTrigger": False, "reduceOnly": False, "tif": "Gtc", "timestamp": 1_000},
        {"coin": "BTC", "oid": 31, "side": "A", "sz": "0.01", "origSz": "0.01", "limitPx": "90250.0",
         "orderType": "Stop Market", "isTrigger": True, "triggerPx": "95000.0", "reduceOnly": True,
         "tif": None, "timestamp": 2_000},
        {"coin": "BTC", "oid": 32, "side": "A", "sz": "0.01", "origSz": "0.01", "limitPx": "104500.0",
         "orderType": "Take Profit Market", "isTrigger": True, "triggerPx": "110000.0", "reduceOnly": True,
         "tif": None, "timestamp": 3_000, "isPositionTpsl": True},
        {"bad": "row"},
    ]
    orders = broker.open_orders()
    assert [o.id for o in orders] == [entry.id, "31", "32"]
    lim, sl, tp = orders
    assert (lim.kind, lim.is_buy, lim.size, lim.filled_size, lim.remaining) == (LIMIT, True, 1.0, 0.6, 0.4)
    assert (lim.price, lim.trigger_px, lim.tif, lim.tag, lim.created_ts) == (140.0, None, "Gtc", "entry", 1.0)
    assert (sl.kind, sl.is_buy, sl.trigger_px, sl.price, sl.reduce_only, sl.tag) == (STOP, False, 95_000.0, None, True, "sl")
    assert (tp.kind, tp.trigger_px, tp.tag) == (TP, 110_000.0, "tp") and "posición completa" in tp.reason
    assert all(o.status is OrderStatus.OPEN for o in orders)
    assert [o.id for o in broker.open_orders("BTC")] == ["31", "32"]


# --------------------------------------------------------------------------- fills

def test_poll_fills_maps_dedupes_and_advances_cursor(broker, fx, fi, clock):
    t0 = int(clock.now() * 1000)
    entry = broker.place_order("SOL", True, 2, LIMIT, price=150.0, tag="entry")
    oid = int(entry.id)
    fi.fills = [
        fill_row(1, 999, t0 - 5_000),                                      # anterior al arranque
        fill_row(2, oid, t0 + 1_000, sz="1.2", px="150.0", fee="0.081", builderFee="0.01"),
        fill_row(3, oid, t0 + 2_000, sz="0.8", px="149.9", fee="0.054"),
    ]
    fills = broker.poll_fills()
    assert [(f.order_id, f.size, f.price) for f in fills] == [(entry.id, 1.2, 150.0), (entry.id, 0.8, 149.9)]
    f = fills[0]
    assert f.is_buy and f.tag == "entry" and f.fee == pytest.approx(0.091) and f.ts == (t0 + 1_000) / 1000
    assert fi.calls[-1] == ("user_fills_by_time", ADDR, t0)
    assert broker.poll_fills() == []                                       # la misma ventana no repite
    assert fi.calls[-1] == ("user_fills_by_time", ADDR, t0 + 2_000)
    assert broker.fills_cursor_ts == (t0 + 2_000) / 1000

    # Un fill nuevo en el mismo milisegundo que el último visto, y otro posterior: cierre con PnL.
    fi.fills.append(fill_row(4, 555, t0 + 2_000, side="A", sz="0.5", px="151.0", pnl="0.55"))
    fi.fills.append(fill_row(5, 556, t0 + 9_000, side="A", sz="1.5", px="140.0", pnl="-15.0",
                             liquidation={"liquidatedUser": ADDR, "markPx": "140.0", "method": "market"}))
    fills = broker.poll_fills()
    assert [f.order_id for f in fills] == ["555", "556"]
    assert not fills[0].is_buy and fills[0].realized_pnl == 0.55 and not fills[0].liquidation
    assert fills[1].realized_pnl == -15.0 and fills[1].liquidation
    assert broker.poll_fills() == []


def test_poll_fills_paginates(broker, fi, clock, monkeypatch):
    monkeypatch.setattr(hl, "FILLS_PAGE_LIMIT", 2)
    fi.page = 2
    t0 = int(clock.now() * 1000)
    fi.fills = [fill_row(i, 100 + i, t0 + i * 10) for i in range(1, 6)]
    fills = broker.poll_fills()
    assert [f.order_id for f in fills] == ["101", "102", "103", "104", "105"]
    assert broker.poll_fills() == []


def test_poll_fills_since_custom_ts_and_errors(market, live_settings, clock, fx, fi):
    t0 = int(clock.now() * 1000)
    fi.fills = [fill_row(1, 7, t0 - 60_000)]
    b = LiveHyperliquidBroker(market, live_settings, clock, exchange=fx, info=fi,
                              fills_since_ts=clock.now() - 3600)
    assert [f.order_id for f in b.poll_fills()] == ["7"]
    fi.fail = requests.Timeout("lento")
    with pytest.raises(BrokerError, match="user_fills_by_time"):
        b.poll_fills()


def test_step_and_save_state_are_noops(broker, fx, fi):
    broker.step()
    broker.save_state()
    assert fx.calls == [] and fi.calls == []


# --------------------------------------------------------------------------- SDK real (firma), sin red

def test_real_sdk_exchange_signs_our_orders_without_network(market, live_settings, clock):
    """El ``Exchange`` real del SDK con metadatos precargados no hace peticiones al construirse;
    se intercepta ``post`` para comprobar el payload firmado que saldría hacia ``/exchange``."""
    from eth_account import Account
    from hyperliquid.exchange import Exchange

    meta = {"universe": [{"name": "BTC", "szDecimals": 5}, {"name": "SOL", "szDecimals": 2},
                         {"name": "HYPE", "szDecimals": 2}]}
    ex = Exchange(Account.from_key(KEY), hl.TESTNET_API_URL, meta=meta,
                  spot_meta={"universe": [], "tokens": []}, account_address=ADDR)
    sent = []

    def fake_post(path, payload=None):
        sent.append((path, payload))
        return ok({"resting": {"oid": 4242}})
    ex.post = fake_post
    market.set_price("SOL", 150.0)
    b = LiveHyperliquidBroker(market, live_settings, clock, exchange=ex, info=FakeInfo())

    o = b.place_order("SOL", True, 1.234, LIMIT, price=140.123, tag="entry")
    assert o.status is OrderStatus.OPEN and o.id == "4242"
    path, payload = sent[-1]
    assert path == "/exchange" and {"r", "s", "v"} <= set(payload["signature"])
    assert payload["action"] == {"type": "order", "grouping": "na", "orders": [
        {"a": 1, "b": True, "p": "140.12", "s": "1.23", "r": False, "t": {"limit": {"tif": "Gtc"}}}]}

    b.place_order("SOL", False, 1.23, STOP, trigger_px=131.37, reduce_only=True, tag="sl")
    wire = sent[-1][1]["action"]["orders"][0]
    assert wire["r"] is True and wire["p"] == "124.8"
    assert wire["t"] == {"trigger": {"isMarket": True, "triggerPx": "131.37", "tpsl": "sl"}}

    b.place_order("HYPE", True, 11.73, MARKET)
    wire = sent[-1][1]["action"]["orders"][0]
    assert wire["a"] == 2 and wire["t"] == {"limit": {"tif": "Ioc"}} and wire["s"] == "11.73"

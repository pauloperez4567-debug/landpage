"""Datos públicos de Hyperliquid (REST ``POST /info``) + liquidaciones de Moon Dev (opcional).

Sólo lectura, sin claves de Hyperliquid. Pensado para correr 24/7:

- Una ``requests.Session`` reutilizada (inyectable para tests), timeout de 10 s.
- Reintentos con backoff exponencial en 429 / 5xx / errores de conexión; el resto de
  códigos y las respuestas malformadas se convierten en ``MarketDataError`` al momento.
- ``last_update_ts`` (edad de los datos para riesgo) es la hora del libro *según el exchange* (``time`` del
  l2Book): un nodo atrasado que sirve un libro congelado no pasa por dato fresco.
- ``meta`` se cachea (cambia muy poco); el libro tiene un cache muy corto (por defecto 1 s)
  para que varias consultas del mismo ciclo no gasten límite de peticiones.
- ``liquidation_levels`` usa ``/api/positions/all.json`` de Moon Dev si hay ``MOONDEV_API_KEY``.
  Es una fuente de enriquecimiento: si falla se degrada (cache viejo o ``[]``) en lugar de
  lanzar, para que una caída de Moon Dev no anule las zonas que salen del libro.

Las esperas del backoff usan ``clock.sleep`` (con ``SimClock`` los tests no esperan de verdad).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Optional

import pandas as pd
import requests

from ..clock import Clock, RealClock
from ..models import AssetMeta
from .base import CANDLE_COLUMNS, Book, LiqLevels, MarketData, MarketDataError
from .intervals import interval_seconds

log = logging.getLogger(__name__)

MAINNET_INFO_URL = "https://api.hyperliquid.xyz/info"
TESTNET_INFO_URL = "https://api.hyperliquid-testnet.xyz/info"
MOONDEV_POSITIONS_URL = "https://api.moondev.com/api/positions/all.json"

MAX_CANDLES = 5000          # Hyperliquid no devuelve más de 5000 velas por petición
MIN_ORDER_USD = 10.0        # valor mínimo de orden en Hyperliquid


class HttpStatusError(MarketDataError):
    """Respuesta HTTP no recuperable (4xx distinto de 429). Guarda el código."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def utc_index(values: Any, unit: str) -> pd.DatetimeIndex:
    """Índice UTC desde marcas epoch. Siempre en nanosegundos: pandas >= 3 conserva la unidad de
    entrada (s / ms) y mezclar unidades entre fuentes de velas rompe comparaciones y ``asi8``."""
    return pd.DatetimeIndex(pd.to_datetime(values, unit=unit, utc=True)).as_unit("ns")


def empty_candles() -> pd.DataFrame:
    """DataFrame de velas vacío con el formato del contrato."""
    return pd.DataFrame({c: pd.Series(dtype=float) for c in CANDLE_COLUMNS},
                        index=pd.DatetimeIndex([], tz="UTC").as_unit("ns"))


class HyperliquidMarketData(MarketData):
    def __init__(self, clock: Optional[Clock] = None, *, testnet: bool = False,
                 session: Optional[requests.Session] = None,
                 moondev_api_key: Optional[str] = None,
                 timeout: float = 10.0, retries: int = 3,
                 backoff_base: float = 0.5, backoff_max: float = 8.0,
                 meta_ttl: float = 3600.0, book_ttl: float = 1.0, liq_ttl: float = 30.0,
                 info_url: Optional[str] = None, moondev_url: str = MOONDEV_POSITIONS_URL):
        self.clock = clock or RealClock()
        self.testnet = testnet
        self.info_url = info_url or (TESTNET_INFO_URL if testnet else MAINNET_INFO_URL)
        self.moondev_url = moondev_url
        if moondev_api_key is None:
            moondev_api_key = os.environ.get("MOONDEV_API_KEY", "")
        self.moondev_api_key = moondev_api_key.strip()
        self.session = session if session is not None else requests.Session()
        self.timeout = timeout
        self.retries = max(int(retries), 0)
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.meta_ttl = meta_ttl
        self.book_ttl = book_ttl
        self.liq_ttl = liq_ttl
        # caches
        self._metas: dict[str, tuple[float, dict[str, AssetMeta]]] = {}   # dex -> (ts, {coin: meta})
        self._books: dict[tuple[str, Optional[int]], Book] = {}
        self._last_ts: dict[str, float] = {}
        self._book_ts: dict[str, float] = {}       # hora (del exchange) del último libro de cada moneda
        self._liq_data: Optional[dict] = None
        self._liq_ts: float = -math.inf
        self._liq_retry_at: float = -math.inf
        self.liq_error: str = ""     # último error de Moon Dev (para el informe de estado)

    @classmethod
    def from_settings(cls, settings: Any, clock: Clock, session: Optional[requests.Session] = None,
                      **kwargs: Any) -> "HyperliquidMarketData":
        """Construye el adaptador según la configuración.

        En ``paper`` se leen precios de mainnet (los reales). En ``live`` se usa testnet salvo
        ``hl_mainnet=True``, para que los precios coincidan con el exchange en el que se opera.
        """
        testnet = bool(settings.is_live and not settings.hl_mainnet)
        return cls(clock, testnet=testnet, session=session,
                   moondev_api_key=settings.moondev_api_key or "", **kwargs)

    # ------------------------------------------------------------------ HTTP
    def _retry_delay(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return min(max(retry_after, 0.0), self.backoff_max * 4)
        return min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)

    def _request(self, method: str, url: str, *, body: Optional[dict] = None,
                 headers: Optional[dict] = None, what: str = "", retries: Optional[int] = None) -> Any:
        retries = self.retries if retries is None else retries
        last_err = ""
        retry_after: Optional[float] = None
        for attempt in range(retries + 1):
            if attempt:
                self.clock.sleep(self._retry_delay(attempt, retry_after))
            retry_after = None
            try:
                if method == "POST":
                    resp = self.session.post(url, json=body, headers=headers, timeout=self.timeout)
                else:
                    resp = self.session.get(url, headers=headers, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
                continue
            status = int(resp.status_code)
            if status == 429 or status >= 500:
                last_err = f"HTTP {status}"
                retry_after = _parse_retry_after(resp)
                continue
            if not 200 <= status < 300:
                raise HttpStatusError(f"{what}: HTTP {status} {_short_text(resp)}", status)
            try:
                return resp.json()
            except ValueError as e:
                raise MarketDataError(f"{what}: la respuesta no es JSON válido") from e
        raise MarketDataError(f"{what}: sin respuesta tras {retries + 1} intentos ({last_err})")

    def _info(self, body: dict, what: str) -> Any:
        return self._request("POST", self.info_url, body=body, what=what)

    # ------------------------------------------------------------------ meta
    def _load_metas(self, dex: str) -> dict[str, AssetMeta]:
        body: dict[str, Any] = {"type": "meta"}
        if dex:
            body["dex"] = dex
        data = self._info(body, f"meta{'/' + dex if dex else ''}")
        universe = data.get("universe") if isinstance(data, dict) else None
        if not isinstance(universe, list):
            raise MarketDataError("meta: falta 'universe' en la respuesta")
        out: dict[str, AssetMeta] = {}
        for asset in universe:
            try:
                name = str(asset["name"])
                szd = int(asset["szDecimals"])
                max_lev = int(asset.get("maxLeverage", 1))
            except (KeyError, TypeError, ValueError):
                continue
            if asset.get("isDelisted"):
                continue
            if dex and ":" not in name:          # activos HIP-3: "dex:NOMBRE"
                name = f"{dex}:{name}"
            out[name] = AssetMeta(name, sz_decimals=szd, max_leverage=max_lev, min_notional_usd=MIN_ORDER_USD)
        return out

    def _metas_for(self, dex: str, force: bool = False) -> dict[str, AssetMeta]:
        now = self.clock.now()
        cached = self._metas.get(dex)
        if cached and not force and now - cached[0] < self.meta_ttl:
            return cached[1]
        try:
            metas = self._load_metas(dex)
        except MarketDataError as e:
            if cached:      # mejor metadatos de hace un rato que ninguno
                log.warning("meta de Hyperliquid no disponible, uso la cacheada: %s", e)
                return cached[1]
            raise
        self._metas[dex] = (now, metas)
        return metas

    def meta(self, coin: str) -> AssetMeta:
        dex = coin.split(":", 1)[0] if ":" in coin else ""
        metas = self._metas_for(dex)
        if coin not in metas:
            # Moneda recién listada: refresca como mucho una vez por minuto.
            cached_ts = self._metas.get(dex, (0.0, {}))[0]
            if self.clock.now() - cached_ts >= 60.0:
                metas = self._metas_for(dex, force=True)
        if coin not in metas:
            raise KeyError(coin)
        return metas[coin]

    def coins(self) -> list[str]:
        """Perps del dex principal (sin los delisted)."""
        return list(self._metas_for(""))

    # ------------------------------------------------------------------ libro
    def l2_book(self, coin: str, n_sig_figs: Optional[int] = None) -> Book:
        nsf = None if n_sig_figs is None else min(max(int(n_sig_figs), 2), 5)
        key = (coin, nsf)
        now = self.clock.now()
        cached = self._books.get(key)
        if cached is not None and now - cached["ts"] < self.book_ttl:
            return cached
        body: dict[str, Any] = {"type": "l2Book", "coin": coin}
        if nsf is not None:
            body["nSigFigs"] = nsf
        data = self._info(body, f"l2Book {coin}")
        if not isinstance(data, dict) or not isinstance(data.get("levels"), list) or len(data["levels"]) != 2:
            raise MarketDataError(f"l2Book {coin}: respuesta sin 'levels' (¿moneda inexistente?)")
        try:
            bids = [(float(lv["px"]), float(lv["sz"])) for lv in data["levels"][0]]
            asks = [(float(lv["px"]), float(lv["sz"])) for lv in data["levels"][1]]
        except (KeyError, TypeError, ValueError) as e:
            raise MarketDataError(f"l2Book {coin}: nivel malformado ({e})") from e
        book: Book = {"bids": bids, "asks": asks, "ts": now}
        if isinstance(data.get("time"), (int, float)):
            book["exchange_ts"] = data["time"] / 1000.0
        self._books[key] = book
        self._last_ts[coin] = now
        # Edad del precio = la del dato según el exchange (un nodo atrasado puede servir un libro congelado),
        # nunca posterior a la petición (si el reloj del exchange va por delante del local).
        self._book_ts[coin] = min(now, book.get("exchange_ts", now))
        return book

    def bid_ask(self, coin: str) -> tuple[float, float]:
        book = self.l2_book(coin)
        if not book["bids"] or not book["asks"]:
            raise MarketDataError(f"{coin}: libro vacío por un lado (bids={len(book['bids'])}, asks={len(book['asks'])})")
        bid, ask = book["bids"][0][0], book["asks"][0][0]
        if bid <= 0 or ask <= 0 or bid >= ask:
            raise MarketDataError(f"{coin}: precios incoherentes bid={bid} ask={ask}")
        return bid, ask

    # ------------------------------------------------------------------ velas
    def candles(self, coin: str, interval: str, limit: int) -> pd.DataFrame:
        sec = interval_seconds(interval)
        limit = min(max(int(limit), 1), MAX_CANDLES)
        now = self.clock.now()
        current_open = math.floor(now / sec) * sec
        # Una vela de margen: 1w/1d de Hyperliquid pueden no alinear con múltiplos del epoch.
        start = current_open - limit * sec
        body = {"type": "candleSnapshot",
                "req": {"coin": coin, "interval": interval,
                        "startTime": int(start * 1000), "endTime": int(now * 1000)}}
        data = self._info(body, f"candleSnapshot {coin} {interval}")
        if not isinstance(data, list):
            raise MarketDataError(f"candleSnapshot {coin}: se esperaba una lista")
        if not data:
            return empty_candles()
        try:
            ts = [int(c["t"]) for c in data]
            cols = {name: [float(c[k]) for c in data]
                    for name, k in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"), ("volume", "v"))}
        except (KeyError, TypeError, ValueError) as e:
            raise MarketDataError(f"candleSnapshot {coin}: vela malformada ({e})") from e
        df = pd.DataFrame(cols, index=utc_index(ts, "ms"))[CANDLE_COLUMNS]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        self._last_ts[coin] = now
        return df.tail(limit)

    # ------------------------------------------------------------------ liquidaciones
    def _moondev_positions(self) -> Optional[dict]:
        now = self.clock.now()
        if self._liq_data is not None and now - self._liq_ts < self.liq_ttl:
            return self._liq_data
        if now < self._liq_retry_at:          # en pausa tras un fallo reciente
            return self._stale_liqs(now)
        try:
            data = self._request("GET", self.moondev_url, headers={"X-API-Key": self.moondev_api_key},
                                 what="Moon Dev positions", retries=1)
            if not isinstance(data, dict) or not isinstance(data.get("symbols"), dict):
                raise MarketDataError("Moon Dev positions: falta 'symbols'")
        except MarketDataError as e:
            # 401/403 = clave inválida: no insistir (la API castiga los reintentos con clave muerta).
            pause = 600.0 if isinstance(e, HttpStatusError) and e.status in (401, 403) else self.liq_ttl
            self._liq_retry_at = now + pause
            self.liq_error = str(e)
            log.warning("liquidaciones de Moon Dev no disponibles (%s); reintento en %.0f s", e, pause)
            return self._stale_liqs(now)
        self._liq_data, self._liq_ts, self.liq_error = data, now, ""
        return data

    def _stale_liqs(self, now: float) -> Optional[dict]:
        # Un mapa de liquidaciones de hasta 10 min sigue siendo útil; más viejo, no.
        if self._liq_data is not None and now - self._liq_ts < 600.0:
            return self._liq_data
        return None

    def liquidation_levels(self, coin: str) -> LiqLevels:
        if not self.moondev_api_key:
            return []
        data = self._moondev_positions()
        if not data:
            return []
        entry = data["symbols"].get(coin) or {}
        out: LiqLevels = []
        for key, side in (("longs", "long"), ("shorts", "short")):
            for p in entry.get(key) or []:
                try:
                    px = float(p.get("liq_price"))
                    usd = float(p.get("value"))
                except (AttributeError, TypeError, ValueError):
                    continue
                if px > 0 and usd > 0 and math.isfinite(px) and math.isfinite(usd):
                    out.append({"liq_px": px, "usd": usd, "side": side})
        return out

    def last_update_ts(self, coin: str) -> Optional[float]:
        """Hora del último precio: la del libro según el exchange (campo ``time`` del l2Book); sin libro, la de
        la última petición de datos de la moneda."""
        ts = self._book_ts.get(coin)
        return ts if ts is not None else self._last_ts.get(coin)


def _parse_retry_after(resp: Any) -> Optional[float]:
    headers = getattr(resp, "headers", None) or {}
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _short_text(resp: Any) -> str:
    try:
        return str(resp.text)[:200]
    except Exception:  # noqa: BLE001 - sólo para el mensaje de error
        return ""

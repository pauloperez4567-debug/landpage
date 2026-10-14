"""Interfaces de mercado. Toda la mesa habla con estas dos abstracciones:

- ``MarketData``: datos públicos (metadatos, precios, libro, velas, liquidaciones).
- ``Broker``: cuenta y órdenes (simulado en ``paper.py``, real en ``hyperliquid_live.py``).

Las implementaciones deben lanzar ``MarketDataError`` / ``BrokerError`` ante fallos
recuperables (red, rate limit, respuesta inválida): la mesa los captura, reintenta
con backoff y nunca se cae por ellos.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from ..models import AccountState, AssetMeta, Fill, Order, OrderKind

# Velas: DataFrame con índice DatetimeIndex UTC (apertura de la vela) y columnas
# exactamente ["open", "high", "low", "close", "volume"] en float, orden ascendente.
CANDLE_COLUMNS = ["open", "high", "low", "close", "volume"]

# Libro: {"bids": [(px, sz), ...] de mejor a peor, "asks": [(px, sz), ...] de mejor a peor, "ts": float}
Book = dict

# Liquidaciones conocidas (posiciones grandes con su precio de liquidación):
# lista de dicts {"liq_px": float, "usd": float, "side": "long"|"short"}
LiqLevels = list


class MarketDataError(RuntimeError):
    """Fallo recuperable al obtener datos de mercado."""


class BrokerError(RuntimeError):
    """Fallo recuperable al consultar la cuenta o enviar órdenes."""


class MarketData(ABC):
    @abstractmethod
    def meta(self, coin: str) -> AssetMeta:
        """Metadatos de la moneda. Lanza ``KeyError`` si no existe."""

    @abstractmethod
    def coins(self) -> list[str]:
        """Monedas operables."""

    @abstractmethod
    def bid_ask(self, coin: str) -> tuple[float, float]:
        """(mejor bid, mejor ask)."""

    def mid(self, coin: str) -> float:
        bid, ask = self.bid_ask(coin)
        return (bid + ask) / 2.0

    @abstractmethod
    def l2_book(self, coin: str, n_sig_figs: Optional[int] = None) -> Book:
        """Libro agregado. ``n_sig_figs`` (2..5) agrupa niveles como Hyperliquid para ver más profundidad."""

    @abstractmethod
    def candles(self, coin: str, interval: str, limit: int) -> pd.DataFrame:
        """Últimas ``limit`` velas cerradas o en curso. ``interval``: 1m, 5m, 15m, 1h, 4h, 1d."""

    def liquidation_levels(self, coin: str) -> LiqLevels:
        """Precios de liquidación de posiciones grandes. Por defecto no hay fuente: lista vacía."""
        return []

    def last_update_ts(self, coin: str) -> Optional[float]:
        """Marca de tiempo del último dato recibido (para detectar datos viejos). None = desconocido."""
        return None


class Broker(ABC):
    @abstractmethod
    def account(self) -> AccountState:
        """Estado actual: equity, disponible, posiciones (con mark/PnL) y órdenes abiertas."""

    @abstractmethod
    def set_leverage(self, coin: str, leverage: int, cross: bool = False) -> None:
        """Fija el apalancamiento (aislado por defecto) para las próximas órdenes de la moneda."""

    @abstractmethod
    def place_order(self, coin: str, is_buy: bool, size: float, kind: OrderKind, *,
                    price: Optional[float] = None, trigger_px: Optional[float] = None,
                    tif: str = "Gtc", reduce_only: bool = False, tag: str = "") -> Order:
        """Envía una orden. Nunca lanza por rechazos de negocio: devuelve la orden con
        ``status=REJECTED`` y ``reason``. Sólo lanza ``BrokerError`` en fallos de transporte.

        - LIMIT: ``price`` obligatorio; ``tif`` "Gtc" (reposa), "Alo" (sólo maker), "Ioc".
        - MARKET: IOC agresiva; ``price`` opcional = peor precio aceptable.
        - STOP / TAKE_PROFIT: ``trigger_px`` obligatorio, se ejecutan a mercado al dispararse;
          deben ser ``reduce_only=True``.
        """

    @abstractmethod
    def cancel(self, coin: str, order_id: str) -> bool:
        """Cancela una orden. True si estaba abierta y se canceló."""

    def cancel_all(self, coin: Optional[str] = None) -> int:
        n = 0
        for o in self.open_orders(coin):
            if self.cancel(o.coin, o.id):
                n += 1
        return n

    @abstractmethod
    def open_orders(self, coin: Optional[str] = None) -> list[Order]:
        """Órdenes abiertas (incluidas las de disparo SL/TP)."""

    @abstractmethod
    def poll_fills(self) -> list[Fill]:
        """Ejecuciones nuevas desde la última llamada (cada fill se entrega una sola vez)."""

    def step(self) -> None:
        """Avanza la simulación (paper): procesa órdenes contra el mercado actual. Live: no-op."""
        return None

    def save_state(self) -> None:
        """Persiste el estado propio si lo tiene (paper). Live: no-op."""
        return None


# ---------------------------------------------------------------------------
# Redondeos de Hyperliquid (perps): tamaño con ``sz_decimals`` decimales; precio con
# como máximo 5 cifras significativas y como máximo ``6 - sz_decimals`` decimales.
# Los precios enteros siempre son válidos.
# ---------------------------------------------------------------------------

def round_size(size: float, sz_decimals: int) -> float:
    """Redondea hacia abajo (nunca agranda una orden por redondeo)."""
    q = 10 ** sz_decimals
    return math.floor(abs(size) * q + 1e-9) / q


def round_price(px: float, sz_decimals: int, max_decimals_base: int = 6) -> float:
    if px <= 0:
        raise ValueError(f"precio inválido: {px}")
    if px >= 100_000:
        return float(round(px))
    max_dec = max(max_decimals_base - sz_decimals, 0)
    sig_dec = max(5 - int(math.floor(math.log10(px))) - 1, 0)
    return round(px, min(max_dec, sig_dec))

"""Intervalos de vela admitidos (los mismos nombres que usa Hyperliquid)."""

from __future__ import annotations

INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "8h": 28800, "12h": 43200,
    "1d": 86400, "3d": 259200, "1w": 604800,
}


def interval_seconds(interval: str) -> int:
    """Duración en segundos de ``interval``. Lanza ``ValueError`` si no es un intervalo conocido."""
    try:
        return INTERVAL_SECONDS[interval]
    except KeyError:
        raise ValueError(
            f"intervalo de vela desconocido: {interval!r} (válidos: {', '.join(INTERVAL_SECONDS)})"
        ) from None

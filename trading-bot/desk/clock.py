"""Reloj inyectable: real para producción, simulado para tests y simulaciones aceleradas."""

from __future__ import annotations

import time
from datetime import datetime, timezone


class Clock:
    """Interfaz mínima. Toda la mesa obtiene la hora de aquí."""

    def now(self) -> float:  # segundos UNIX UTC
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:
        raise NotImplementedError

    def utc_day(self) -> str:
        """Día UTC actual como 'YYYY-MM-DD' (para el límite de pérdida diaria)."""
        return datetime.fromtimestamp(self.now(), tz=timezone.utc).strftime("%Y-%m-%d")

    def iso(self) -> str:
        return datetime.fromtimestamp(self.now(), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class RealClock(Clock):
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class SimClock(Clock):
    """Reloj manual: ``sleep`` avanza el tiempo al instante, sin esperar."""

    def __init__(self, start: float = 1_767_225_600.0):  # 2026-01-01 00:00:00 UTC
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._t += seconds

    def advance(self, seconds: float) -> None:
        self.sleep(seconds)

    def set(self, ts: float) -> None:
        self._t = float(ts)

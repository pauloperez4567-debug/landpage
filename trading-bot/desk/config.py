"""Configuración de la mesa. Se lee de variables de entorno (``DESK_*``) y ``.env``.

Todos los parámetros tienen un valor por defecto prudente: el bot arranca en modo
``paper`` (simulado) y el modo real exige confirmaciones explícitas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional

LIVE_CONFIRM_PHRASE = "I_UNDERSTAND_REAL_MONEY"

# Alias aceptados en el comando "set" (español e inglés) -> nombre del campo.
PARAM_ALIASES: dict[str, str] = {
    "leverage": "leverage", "apalancamiento": "leverage", "lev": "leverage",
    "unit": "unit_margin_usd", "unidad": "unit_margin_usd", "unit_margin": "unit_margin_usd",
    "risk": "risk_per_trade_pct", "riesgo": "risk_per_trade_pct",
    "sl": "default_sl_pct", "stop": "default_sl_pct",
    "tp": "default_tp_r", "rr": "default_tp_r",
    "max_positions": "max_open_positions", "posiciones": "max_open_positions",
    "daily_loss": "daily_loss_limit_pct", "perdida_diaria": "daily_loss_limit_pct",
    "drawdown": "max_drawdown_pct",
    "ttl": "entry_ttl_minutes",
}

# Parámetros que se pueden cambiar en caliente con un comando, con sus límites.
TUNABLE: dict[str, tuple[float, float]] = {
    "leverage": (1, 50),
    "unit_margin_usd": (1, 1_000_000),
    "risk_per_trade_pct": (0.05, 10),
    "default_sl_pct": (0.1, 20),
    "default_tp_r": (0.5, 10),
    "max_open_positions": (1, 20),
    "daily_loss_limit_pct": (0.5, 50),
    "max_drawdown_pct": (1, 90),
    "entry_ttl_minutes": (1, 7 * 24 * 60),
}


def _parse_bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "si", "sí", "on"}


@dataclass
class Settings:
    # --- modo y mercado --------------------------------------------------------
    mode: str = "paper"                      # "paper" | "live"
    data_source: str = "hyperliquid"         # "hyperliquid" | "sim" | "replay:<ruta.csv>"
    symbols: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL", "HYPE"])
    state_dir: str = "state"

    # --- dimensionamiento ------------------------------------------------------
    leverage: int = 5                        # apalancamiento deseado (se recorta al máximo del exchange)
    unit_margin_usd: float = 500.0           # 1 unidad = este margen en USD
    max_units_per_trade: float = 4.0

    # --- riesgo ----------------------------------------------------------------
    risk_per_trade_pct: float = 1.0          # pérdida máxima en el SL, % del equity
    require_stop: bool = True                # nunca abrir sin stop-loss
    default_sl_pct: float = 1.5              # SL por defecto si el comando no trae uno
    default_tp_r: float = 2.0                # TP por defecto = R múltiplos del riesgo
    max_open_positions: int = 3
    max_notional_per_coin_usd: float = 25_000.0
    max_gross_notional_usd: float = 50_000.0
    daily_loss_limit_pct: float = 3.0        # halt de nuevas entradas hasta el siguiente día UTC
    max_drawdown_pct: float = 5.0            # desde el máximo de equity: cierra todo y detiene
    flatten_on_daily_limit: bool = False
    max_consecutive_losses: int = 4
    loss_cooldown_minutes: float = 120.0
    max_spread_bps: float = 25.0             # no entrar si el spread es mayor
    max_data_age_seconds: float = 120.0      # no entrar con datos viejos

    # --- ejecución -------------------------------------------------------------
    default_entry_mode: str = "zone"         # "zone" | "market"
    zone_fallback: str = "skip"              # si no hay zona: "skip" | "market"
    entry_ttl_minutes: float = 240.0         # cancelar entradas límite no ejecutadas
    market_slippage_bps: float = 10.0        # tolerancia del precio límite de las órdenes IOC

    # --- liquidez --------------------------------------------------------------
    zone_width_pct: float = 0.11
    zone_max_distance_pct: float = 3.0
    zone_min_distance_pct: float = 0.05
    zone_book_weight: float = 1.0
    zone_liq_weight: float = 2.0             # las liquidaciones pesan más que el libro

    # --- operativa autónoma ----------------------------------------------------
    auto_enabled: bool = True
    strategies: list[str] = field(default_factory=lambda: ["liquidity_sweep", "squeeze_breakout"])
    auto_units: float = 1.0
    loop_seconds: float = 10.0
    strategy_seconds: float = 60.0
    status_every_seconds: float = 300.0

    # --- paper -----------------------------------------------------------------
    paper_start_equity: float = 10_000.0
    paper_taker_fee_bps: float = 4.5
    paper_maker_fee_bps: float = 1.5
    paper_slippage_bps: float = 2.0
    paper_maintenance_margin_pct: float = 0.5

    # --- IA (interpretar comandos en lenguaje libre) ----------------------------
    llm_enabled: bool = True
    llm_model: str = "claude-opus-5-5"
    llm_confirm_trades: bool = True          # pedir "confirmar" a las órdenes interpretadas por IA

    # --- live (Hyperliquid) ----------------------------------------------------
    hl_mainnet: bool = False                 # por defecto testnet incluso en modo live
    live_confirm: str = ""
    hl_private_key: str = field(default="", repr=False)
    hl_account_address: str = ""
    moondev_api_key: str = field(default="", repr=False)

    # ------------------------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return Path(self.state_dir)

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    def validate(self) -> None:
        if self.mode not in {"paper", "live"}:
            raise ValueError(f"DESK_MODE inválido: {self.mode!r} (usa 'paper' o 'live')")
        if self.default_entry_mode not in {"zone", "market"}:
            raise ValueError("DESK_DEFAULT_ENTRY_MODE debe ser 'zone' o 'market'")
        if self.zone_fallback not in {"skip", "market"}:
            raise ValueError("DESK_ZONE_FALLBACK debe ser 'skip' o 'market'")
        if self.leverage < 1:
            raise ValueError("DESK_LEVERAGE debe ser >= 1")
        if not self.symbols:
            raise ValueError("DESK_SYMBOLS no puede estar vacío")
        if self.is_live:
            if self.live_confirm != LIVE_CONFIRM_PHRASE:
                raise ValueError(
                    f"Modo live bloqueado: define DESK_LIVE_CONFIRM={LIVE_CONFIRM_PHRASE} "
                    "si de verdad quieres operar con dinero real."
                )
            if not self.hl_private_key:
                raise ValueError("Modo live requiere HL_PRIVATE_KEY")
            if self.data_source != "hyperliquid":
                raise ValueError("Modo live sólo admite DESK_DATA=hyperliquid")

    def set_param(self, name: str, value: float) -> str:
        """Cambia un parámetro ajustable en caliente. Devuelve el nombre canónico."""
        key = PARAM_ALIASES.get(name.strip().lower(), name.strip().lower())
        if key not in TUNABLE:
            raise ValueError(f"Parámetro no ajustable: {name!r}. Ajustables: {', '.join(sorted(TUNABLE))}")
        lo, hi = TUNABLE[key]
        if not (lo <= value <= hi):
            raise ValueError(f"{key} fuera de rango [{lo}, {hi}]: {value}")
        current = getattr(self, key)
        setattr(self, key, type(current)(value))
        return key

    def public_dict(self) -> dict[str, Any]:
        """Configuración sin secretos (para logs / estado)."""
        out = {}
        for f in fields(self):
            if f.name in {"hl_private_key", "moondev_api_key", "live_confirm"}:
                continue
            out[f.name] = getattr(self, f.name)
        return out


_ENV_MAP: dict[str, str] = {
    "DESK_MODE": "mode",
    "DESK_DATA": "data_source",
    "DESK_SYMBOLS": "symbols",
    "DESK_STATE_DIR": "state_dir",
    "DESK_LEVERAGE": "leverage",
    "DESK_UNIT_MARGIN_USD": "unit_margin_usd",
    "DESK_MAX_UNITS_PER_TRADE": "max_units_per_trade",
    "DESK_RISK_PER_TRADE_PCT": "risk_per_trade_pct",
    "DESK_REQUIRE_STOP": "require_stop",
    "DESK_DEFAULT_SL_PCT": "default_sl_pct",
    "DESK_DEFAULT_TP_R": "default_tp_r",
    "DESK_MAX_OPEN_POSITIONS": "max_open_positions",
    "DESK_MAX_NOTIONAL_PER_COIN_USD": "max_notional_per_coin_usd",
    "DESK_MAX_GROSS_NOTIONAL_USD": "max_gross_notional_usd",
    "DESK_DAILY_LOSS_LIMIT_PCT": "daily_loss_limit_pct",
    "DESK_MAX_DRAWDOWN_PCT": "max_drawdown_pct",
    "DESK_FLATTEN_ON_DAILY_LIMIT": "flatten_on_daily_limit",
    "DESK_MAX_CONSECUTIVE_LOSSES": "max_consecutive_losses",
    "DESK_LOSS_COOLDOWN_MINUTES": "loss_cooldown_minutes",
    "DESK_MAX_SPREAD_BPS": "max_spread_bps",
    "DESK_MAX_DATA_AGE_SECONDS": "max_data_age_seconds",
    "DESK_DEFAULT_ENTRY_MODE": "default_entry_mode",
    "DESK_ZONE_FALLBACK": "zone_fallback",
    "DESK_ENTRY_TTL_MINUTES": "entry_ttl_minutes",
    "DESK_MARKET_SLIPPAGE_BPS": "market_slippage_bps",
    "DESK_ZONE_WIDTH_PCT": "zone_width_pct",
    "DESK_ZONE_MAX_DISTANCE_PCT": "zone_max_distance_pct",
    "DESK_ZONE_MIN_DISTANCE_PCT": "zone_min_distance_pct",
    "DESK_ZONE_BOOK_WEIGHT": "zone_book_weight",
    "DESK_ZONE_LIQ_WEIGHT": "zone_liq_weight",
    "DESK_AUTO": "auto_enabled",
    "DESK_STRATEGIES": "strategies",
    "DESK_AUTO_UNITS": "auto_units",
    "DESK_LOOP_SECONDS": "loop_seconds",
    "DESK_STRATEGY_SECONDS": "strategy_seconds",
    "DESK_STATUS_EVERY_SECONDS": "status_every_seconds",
    "DESK_PAPER_START_EQUITY": "paper_start_equity",
    "DESK_PAPER_TAKER_FEE_BPS": "paper_taker_fee_bps",
    "DESK_PAPER_MAKER_FEE_BPS": "paper_maker_fee_bps",
    "DESK_PAPER_SLIPPAGE_BPS": "paper_slippage_bps",
    "DESK_PAPER_MAINTENANCE_MARGIN_PCT": "paper_maintenance_margin_pct",
    "DESK_LLM_ENABLED": "llm_enabled",
    "DESK_LLM_MODEL": "llm_model",
    "DESK_LLM_CONFIRM_TRADES": "llm_confirm_trades",
    "DESK_HL_MAINNET": "hl_mainnet",
    "DESK_LIVE_CONFIRM": "live_confirm",
    "HL_PRIVATE_KEY": "hl_private_key",
    "HL_ACCOUNT_ADDRESS": "hl_account_address",
    "MOONDEV_API_KEY": "moondev_api_key",
}


def _coerce(name: str, raw: str, default: Any) -> Any:
    if isinstance(default, bool):
        return _parse_bool(raw)
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        return [s.strip().upper() if name == "symbols" else s.strip() for s in raw.split(",") if s.strip()]
    return raw.strip()


def load_settings(env: Optional[Mapping[str, str]] = None, dotenv_path: Optional[str] = ".env",
                  **overrides: Any) -> Settings:
    """Construye ``Settings`` desde el entorno (y ``.env`` si existe) más ``overrides``."""
    if env is None:
        if dotenv_path and Path(dotenv_path).exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path, override=False)
            except ImportError:  # python-dotenv es opcional
                pass
        env = os.environ
    s = Settings()
    for env_key, attr in _ENV_MAP.items():
        if env_key in env and env[env_key] != "":
            setattr(s, attr, _coerce(attr, env[env_key], getattr(s, attr)))
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(s, k):
            raise AttributeError(f"Parámetro desconocido: {k}")
        setattr(s, k, v)
    s.symbols = [c.upper() for c in s.symbols]
    s.validate()
    return s

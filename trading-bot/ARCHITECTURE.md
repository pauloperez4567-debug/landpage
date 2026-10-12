# Arquitectura de la mesa (`desk`)

Contrato entre módulos. Cualquier cambio de interfaz se refleja aquí.

```
                 comandos (stdin / inbox / CLI "send")
                              │
                       commands.parse()  ── regex ES/EN ──► Command
                              │              └─ fallback IA (Claude) + confirmación
                              ▼
 estrategias ──Signal──► TradingDesk (desk.py) ──TradeIntent──► RiskManager.evaluate()
 (strategy.py)            bucle 24/7                                   │ RiskDecision
      ▲                      │                                         ▼
 liquidity.py ◄── MarketData │                              ExecutionEngine (engine.py)
 indicators.py               │                                         │ órdenes + brackets SL/TP
                             └──────────── journal.py ◄────────────────┤
                                       (estado, CSV, JSONL)             ▼
                                                              Broker: PaperBroker | LiveHyperliquidBroker
```

## Módulos base (ya escritos, no cambiar interfaces sin actualizar este archivo)

- `desk/models.py`: `Side`, `OrderKind`, `OrderStatus`, `EntryMode`, `AssetMeta`, `Order`, `Fill`,
  `Position`, `AccountState`, `Zone`, `CommandAction`, `Command`, `Signal`, `TradeIntent`, `RiskDecision`.
- `desk/config.py`: `Settings`, `load_settings(env=None, dotenv_path=".env", **overrides)`,
  `PARAM_ALIASES`, `TUNABLE`, `LIVE_CONFIRM_PHRASE`.
- `desk/clock.py`: `Clock`, `RealClock`, `SimClock` (``sleep`` avanza el tiempo al instante).
- `desk/indicators.py`: `sma, ema, wilder, rsi, true_range, atr, bollinger, keltner, adx, squeeze`.
- `desk/exchange/base.py`: `MarketData`, `Broker`, `MarketDataError`, `BrokerError`,
  `round_size`, `round_price`, `CANDLE_COLUMNS`.
- `tests/conftest.py`: `FakeMarketData`, `make_candles`, fixtures `clock`, `market`, `settings`.

## Módulos a implementar

### `desk/exchange/hyperliquid_data.py` — `HyperliquidMarketData(MarketData)`
- REST público `POST https://api.hyperliquid.xyz/info` (testnet: `https://api.hyperliquid-testnet.xyz/info`).
- `meta()`/`coins()` desde `{"type":"meta"}` (`universe[i].name/szDecimals/maxLeverage`), cacheado.
- `bid_ask()`/`l2_book()` desde `{"type":"l2Book","coin":c,"nSigFigs":n}` (`levels[0]`=bids, `levels[1]`=asks,
  cada nivel `{"px","sz","n"}` en strings).
- `candles()` desde `{"type":"candleSnapshot","req":{"coin","interval","startTime","endTime"}}`
  (campos `t,o,h,l,c,v`), devuelve DataFrame con `CANDLE_COLUMNS`.
- `liquidation_levels()` opcional: si hay `MOONDEV_API_KEY`, `GET https://api.moondev.com/api/positions/all.json`
  con cabecera `X-API-Key`; `data["symbols"][coin]["longs"|"shorts"]`, cada posición con `liq_price` y `value`.
  Sin clave → `[]`. Cache corto (30 s).
- `requests.Session`, timeout 10 s, reintentos con backoff en 429/5xx, errores → `MarketDataError`.
- Inyectable `session` para tests (los tests NO usan red: mockear la sesión).

### `desk/exchange/sim.py` — mercados offline
- `SimMarketData(MarketData)`: paseo aleatorio determinista (semilla) por moneda, avanza con el `Clock`
  inyectado; libro sintético con "muros" de liquidez y niveles de liquidación sintéticos; velas
  generadas a partir del camino de precios. Sirve para la demo sin red: `DESK_DATA=sim`.
- `ReplayMarketData(MarketData)`: reproduce un CSV de velas (`timestamp,open,high,low,close,volume`)
  avanzando con el `Clock`; precio actual = interpolación dentro de la vela en curso. `DESK_DATA=replay:<csv>`.

### `desk/exchange/paper.py` — `PaperBroker(Broker)`
- Constructor: `PaperBroker(market: MarketData, settings: Settings, clock: Clock, state_file: Path|None)`.
- Equity inicial `settings.paper_start_equity`. Comisiones taker/maker en bps, deslizamiento en bps.
- MARKET: ejecuta al instante contra ask (compra) / bid (venta) + deslizamiento; si se pasa `price` y el
  ejecutable es peor, se rechaza.
- LIMIT: si es ejecutable al colocar (compra ≥ ask / venta ≤ bid) → taker inmediato (salvo `Alo`, que se
  rechaza); si no, reposa. En `step()`: compra se ejecuta si ask ≤ precio límite (a precio límite, maker);
  venta si bid ≥ precio límite.
- STOP/TAKE_PROFIT (`reduce_only=True` obligatorio): se disparan con el mid. Stop de un long (orden de venta)
  cuando mid ≤ trigger; TP de un long cuando mid ≥ trigger; simétrico para shorts. Ejecutan como MARKET.
- `reduce_only`: recorta al tamaño de la posición; si no hay posición en sentido opuesto → rechazo.
  Si la posición queda plana, las órdenes reduce-only restantes de esa moneda se cancelan.
- Posición: precio medio en aumentos, PnL realizado en reducciones, giro (flip) permitido.
  Margen aislado = notional / leverage. Precio de liquidación aproximado con margen de mantenimiento;
  en `step()` si el mark cruza la liquidación → cierre forzado (`Fill.liquidation=True`).
- Validaciones: tamaño ≥ 10^-sz_decimals, notional ≥ `min_notional_usd`, margen disponible suficiente,
  apalancamiento ≤ `max_leverage`.
- `poll_fills()` entrega cada fill una vez. `save_state()` escribe JSON atómico (tmp + rename) y el
  constructor lo recarga si existe (sobrevive a reinicios).

### `desk/exchange/hyperliquid_live.py` — `LiveHyperliquidBroker(Broker)`
- Usa `hyperliquid-python-sdk` (`hyperliquid.exchange.Exchange`, `hyperliquid.info.Info`,
  `hyperliquid.utils.constants.MAINNET_API_URL/TESTNET_API_URL`) y `eth_account.Account.from_key`.
  Import perezoso (el resto del bot funciona sin el SDK instalado).
- Sólo construible si `settings.validate()` pasó en modo live; testnet salvo `hl_mainnet=True`.
- LIMIT → `{"limit":{"tif":...}}`; MARKET → IOC límite agresivo (mid ± `market_slippage_bps`);
  STOP/TP → `{"trigger":{"triggerPx":px,"isMarket":True,"tpsl":"sl"|"tp"}}` con `reduce_only=True`.
  Precios con `round_price`, tamaños con `round_size`.
- `account()` desde `info.user_state(address)`; `open_orders()` desde `info.frontend_open_orders`;
  `poll_fills()` desde `info.user_fills_by_time` desde el último ts visto (dedupe por `tid`).
- Errores de transporte → `BrokerError`; respuestas `status != "ok"` o `statuses[i].error` → orden REJECTED.

### `desk/liquidity.py`
- Funciones puras + un envoltorio:
  - `book_levels(book) -> list[(px, usd)]`
  - `liq_levels(levels) -> list[(px, usd)]`
  - `find_zone(coin, side, mid, book_pts, liq_pts, settings) -> Zone|None`: agrupa en tramos de
    `zone_width_pct` del precio; sólo del lado correcto (LONG: por debajo del mid; SHORT: por encima);
    descarta lo que esté a menos de `zone_min_distance_pct` o más de `zone_max_distance_pct`;
    puntuación = `zone_book_weight`·usd_libro + `zone_liq_weight`·usd_liq; devuelve el tramo de mayor
    puntuación con `source` "book"/"liq"/"mixed" y `distance_pct`.
  - `LiquidityScanner(market, settings).zone(coin, side) -> Zone|None` (usa `l2_book(coin, n_sig_figs=3)`
    + `liquidation_levels(coin)`; captura `MarketDataError` → None).

### `desk/strategy.py`
- `Strategy` (ABC): `name`, `interval`, `lookback`, `evaluate(coin, candles, mid, zones: dict[Side, Zone|None]) -> Signal|None`.
- `LiquiditySweepStrategy` (15m): reversión tras barrer una zona de liquidez: la vela anterior perfora la
  zona LONG (low ≤ zone.high) y la última cierra por encima de `zone.low`, RSI(14) < 40 y subiendo → LONG
  con `EntryMode.MARKET`, SL bajo `zone.low - 0.5·ATR`, TP a `default_tp_r` R. Simétrico para SHORT.
- `SqueezeBreakoutStrategy` (1h): fin de squeeze (BB dentro de Keltner) en la vela anterior y no en la
  actual, ADX(14) > 25, cierre fuera de Bollinger → entrada MARKET en dirección de la ruptura, SL a
  1.5·ATR, TP a `default_tp_r` R.
- `build_strategies(settings) -> list[Strategy]` según `settings.strategies`.
- Sólo señales sobre velas cerradas (ignorar la vela en curso).

### `desk/commands.py`
- `parse(text, settings, llm=None) -> Command | None` y `CommandParser` con fallback IA.
- Regex español + inglés, insensible a mayúsculas/acentos. Ejemplos que DEBEN funcionar:
  `long HYPE`, `short btc`, `compra SOL`, `vende ETH`, `long HYPE 2 unidades` / `2u` / `2 units`,
  `long HYPE en zona` / `en zona de liquidez` / `zona` (ZONE), `long HYPE a mercado` / `market` / `ya` (MARKET),
  `long HYPE a 85.2` / `@85.2` / `limit 85.2` (LIMIT), `sl 1.5%` / `stop 84.5`, `tp 4%` / `tp 90`,
  combinables (`long HYPE 2u zona sl 1.5% tp 4%`), `cierra HYPE` / `close HYPE`, `cierra todo` / `close all` / `flat`,
  `cancela HYPE` / `cancel all`, `stop HYPE 84.5` / `sl HYPE 84.5` (SET_SL sobre posición), `tp HYPE 90` (SET_TP),
  `pausa`/`pause`, `reanuda`/`resume`, `auto on|off` / `automatico on|off`, `estado`/`status`/`posiciones`,
  `kill` / `panico` / `apaga todo`, `reset halt`, `apalancamiento 3` / `leverage 3` / `riesgo 0.5` / `unidad 250`
  (SET_PARAM con nombre canónico de `PARAM_ALIASES`), `si`/`confirmar`/`yes`/`confirm`, `no`/`cancelar`/`reject`,
  `ayuda`/`help`.
- Monedas: mayúsculas; acepta prefijos HIP-3 tipo `xyz:GOLD`.
- Fallback IA (`LLMCommandInterpreter`): sólo si la regex no entiende, `settings.llm_enabled` y el SDK
  `anthropic` + credenciales están disponibles. Modelo `settings.llm_model` (por defecto `claude-opus-5-5`),
  esfuerzo bajo, salida estructurada JSON (`output_config.format` json_schema) que mapea a `Command`,
  `fallbacks="default"` (beta `server-side-fallback-2026-07-01`). Comprobar `stop_reason` (`refusal` → None).
  Resultado con `source="llm"`. Nunca lanzar: cualquier error → None. Interfaz inyectable para tests.

### `desk/risk.py` — `RiskManager`
- `RiskManager(settings, clock)`; estado persistible (`to_dict`/`from_dict`): día UTC, equity al inicio del
  día, máximo de equity (high-water mark), pérdidas consecutivas, cooldown hasta, `halted` + motivo.
- `update(account) -> list[str]` (eventos): rollover diario; si pérdida del día ≥ `daily_loss_limit_pct`
  → halt "daily_loss" (se levanta solo al día siguiente); si drawdown desde HWM ≥ `max_drawdown_pct`
  → halt "max_drawdown" + evento "flatten" (sólo se levanta con RESET_HALT).
- `record_trade_result(pnl)`: racha de pérdidas → cooldown de `loss_cooldown_minutes` para entradas automáticas.
- `evaluate(intent, account, meta, bid, ask, data_age_s) -> RiskDecision`:
  apalancamiento = min(settings.leverage, meta.max_leverage); precio de entrada (zona/límite/mercado);
  SL por defecto `default_sl_pct` si no viene (si `require_stop`); valida lado correcto del SL/TP;
  notional = units·unit_margin_usd·leverage, recortado para que la pérdida en el SL ≤ `risk_per_trade_pct`
  del equity, ≤ `max_notional_per_coin_usd`, ≤ exposición bruta restante, ≤ margen disponible;
  rechaza si halted/pausado, `max_open_positions` alcanzado, ya hay posición en la moneda, spread >
  `max_spread_bps`, datos viejos, notional < mínimo; TP por defecto a `default_tp_r` R.
  Tamaño redondeado con `round_size`. Las órdenes de reducción/cierre nunca pasan por aquí (siempre permitidas).

### `desk/engine.py` — `ExecutionEngine`
- `ExecutionEngine(broker, market, settings, clock, journal)`; estado persistible de brackets.
- `open(intent, decision) -> Order`: fija apalancamiento, coloca entrada (ZONE/LIMIT → LIMIT Gtc en
  `decision.entry_px`; MARKET → MARKET), registra bracket pendiente {coin, sl, tp, origin, expires_ts}.
- `on_fills(fills)`: cuando la entrada se ejecuta (total o parcial) coloca/ajusta STOP y TAKE_PROFIT
  reduce-only por el tamaño de la posición; cuando se ejecuta SL o TP cancela el otro (OCO) y devuelve el
  PnL realizado para `RiskManager.record_trade_result`.
- `reconcile(account)`: caduca entradas no ejecutadas (`entry_ttl_minutes`), cancela órdenes reduce-only
  huérfanas si la posición está plana, re-coloca SL/TP que falten en posiciones abiertas, adopta
  posiciones sin bracket (p. ej. tras reinicio) poniéndoles SL por defecto.
- `close(coin)`, `close_all()`, `set_stop(coin, px)`, `set_take_profit(coin, px)`, `cancel(coin|None)`.

### `desk/journal.py` — `Journal`
- `Journal(state_dir)`: `event(kind, **data)` → `events.jsonl`; `fill(fill)` → `fills.csv`;
  `equity(ts, equity)` → `equity.csv`; `save_json(name, obj)`/`load_json(name, default)` atómicos;
  `outbox(cmd_id, text)` → `outbox.log` (respuestas a comandos).

### `desk/desk.py` — `TradingDesk`
- `TradingDesk(settings, market, broker, clock, parser=None, journal=None)`; `run_forever()` y `run_cycle()`.
- Ciclo (cada `loop_seconds`): leer comandos (cola en memoria + `state/inbox.txt` con offset persistido) →
  `broker.step()` → `poll_fills` → `engine.on_fills` → `account` → `risk.update` (halt/flatten) →
  `engine.reconcile` → cada `strategy_seconds`, si auto y no pausado/halted/cooldown: estrategias por
  símbolo → intents → riesgo → engine → guardar estado → línea de estado cada `status_every_seconds`.
- Cada etapa en su propio try/except; fallos consecutivos → backoff exponencial (máx. 300 s); nunca muere.
- `submit(text) -> str` (respuesta legible) para comandos interactivos; comandos de IA que abren
  operaciones quedan pendientes de `confirmar` si `llm_confirm_trades`.
- Las posiciones abiertas por comando (origen "command") no las toca la operativa automática.
- SIGINT/SIGTERM → guardar estado y salir limpio (las órdenes SL/TP quedan en el exchange).

### `desk/__main__.py` — CLI
- `python -m desk run [--mode paper|live] [--data hyperliquid|sim|replay:x.csv] [--symbols ...] [--no-auto] [--fast]`
- `python -m desk send "long HYPE"` (añade al inbox y espera la respuesta en `outbox.log`)
- `python -m desk status` (lee el estado guardado).

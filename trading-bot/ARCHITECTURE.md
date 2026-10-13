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
- `last_update_ts(coin)` = hora del último libro **según el exchange** (`time` del l2Book, nunca posterior a la
  petición); sin libro, la de la última petición. Un nodo que sirve un libro congelado no pasa por dato fresco.
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
- LIMIT → `{"limit":{"tif":...}}`; MARKET → IOC límite agresivo (mid ± `market_slippage_bps`; un MARKET
  `reduce_only` sin `price` usa `CLOSE_SLIPPAGE` = 5 %: con 10 bps un cierre no casa en cuanto el spread pasa de ~20 bps);
  STOP/TP → `{"trigger":{"triggerPx":px,"isMarket":True,"tpsl":"sl"|"tp"}}` con `reduce_only=True`.
  Precios con `round_price`, tamaños con `round_size`.
- `account()` desde `info.user_state(address)`; `open_orders()` desde `info.frontend_open_orders`;
  `poll_fills()` desde `info.user_fills_by_time` desde el último ts visto (inclusive; dedupe por `tid`).
  Constructor con `fills_since_ts` y `fills_seen` (tids ya entregados en ese ms); `fills_cursor_ts` y
  `fills_seen_keys` los expone para que la mesa los guarde: un reinicio no re-entrega el último fill.
- Errores de transporte → `BrokerError`; respuestas `status != "ok"` o `statuses[i].error` → orden REJECTED.

### `desk/liquidity.py`
- Funciones puras + un envoltorio:
  - `book_levels(book) -> list[(px, usd)]`
  - `liq_levels(levels, side=None) -> list[(px, usd)]` (con `side`, sólo liquidaciones de posiciones de
    ese lado: longs → zonas LONG, shorts → zonas SHORT; los niveles sin lado se conservan)
  - `find_zone(coin, side, mid, book_pts, liq_pts, settings) -> Zone|None`: agrupa en tramos de
    `zone_width_pct` del precio; sólo del lado correcto (LONG: por debajo del mid; SHORT: por encima);
    descarta lo que esté a menos de `zone_min_distance_pct` o más de `zone_max_distance_pct`;
    puntuación = `zone_book_weight`·usd_libro + `zone_liq_weight`·usd_liq; devuelve el tramo de mayor
    puntuación con `source` "book"/"liq"/"mixed" y `distance_pct`.
    Tramos anclados en el mid: el tramo j cubre distancias [j·ancho, (j+1)·ancho) del lado que toca (ninguna
    zona contiene el precio). El filtro de distancia se aplica a cada punto y al centro del tramo (la entrada).
    Empate de puntuación → gana el tramo más cercano. Fuente con peso ≤ 0 → ignorada.
  - `LiquidityScanner(market, settings, n_sig_figs=3).zone(coin, side) -> Zone|None` y
    `.zones(coin) -> dict[Side, Zone|None]` (ambos lados con una sola lectura; es lo que piden las
    estrategias). Usa `l2_book(coin, n_sig_figs=3)` + `liquidation_levels(coin)`; `MarketDataError` en
    precio/libro → None; si sólo fallan las liquidaciones, sigue con el libro.

### `desk/strategy.py`
- `Strategy` (ABC): `name`, `interval`, `lookback`, `evaluate(coin, candles, mid, zones: dict[Side, Zone|None]) -> Signal|None`.
- `LiquiditySweepStrategy` (15m): reversión tras barrer una zona de liquidez: la vela anterior perfora la
  zona LONG (low ≤ zone.high) y la última cierra por encima de `zone.low`, RSI(14) < 40 y subiendo → LONG
  con `EntryMode.MARKET`, SL bajo `zone.low - 0.5·ATR`, TP a `default_tp_r` R. Simétrico para SHORT.
- `SqueezeBreakoutStrategy` (1h): fin de squeeze (BB dentro de Keltner) en la vela anterior y no en la
  actual, ADX(14) > 25, cierre fuera de Bollinger → entrada MARKET en dirección de la ruptura, SL a
  1.5·ATR, TP a `default_tp_r` R.
- `build_strategies(settings, clock=None) -> list[Strategy]` según `settings.strategies` (sin duplicados;
  nombre desconocido → `ValueError`). Registro en `STRATEGIES`. Constructor: `Strategy(settings, clock=None)`.
- Sólo señales sobre velas cerradas: la mesa pasa las velas tal cual las da `MarketData.candles` y
  `evaluate(..., now=None)` decide con `closed_candles(candles, interval, now)`: con hora conocida (`now` o el
  `clock` del constructor) descarta la última si `apertura + duración > now`; sin hora asume que la última
  está en curso (Hyperliquid siempre la incluye) y la descarta. **Recomendado: `build_strategies(settings,
  clock)`.** Con hora conocida, si la última cerrada tiene más de una vela + 120 s de retraso → sin señal.
- Como mucho una señal por moneda y vela cerrada (la mesa evalúa cada `strategy_seconds`).
- `lookback` = velas a pedir (incluida la posible vela en curso); `min_bars` = velas cerradas mínimas.

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
  `ayuda`/`help`. KILL admite sufijos de urgencia (`pánico ya`, `kill now`). `cierra todas las posiciones
  abiertas` / `close all trades` = CLOSE_ALL (las palabras de relleno nunca se toman por moneda); `cierra todo BTC`
  / `cancela todo HYPE` son ambiguos → None. `reset params` / `restablece parámetros` → `SET_PARAM` con
  `param=PARAM_DEFAULTS` ("defaults") y sin `value`: la mesa vuelve a los parámetros de arranque.
- Monedas: mayúsculas; acepta prefijos HIP-3 tipo `xyz:GOLD`.
- `SET_SL`/`SET_TP` llevan `price` (precio absoluto) **o** `sl_pct`/`tp_pct` (`sl HYPE 2%`: distancia desde la
  entrada; la mesa lo convierte a precio). Sin moneda (`sl 2%`, `tp 3`) son `SET_PARAM` de los valores por defecto.
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
- `evaluate` comprueba también pérdida diaria y drawdown con el `account.equity` recibido (puede ir por delante
  del último `update()`: comandos entre ciclos). Collar de las entradas límite/zona: si cruzan el libro más de
  `LIMIT_MAX_CROSS_PCT` (1 %) o reposan a más de `LIMIT_MAX_DISTANCE_PCT` (20 %) del mid → rechazo ("¿precio mal
  escrito?"), nunca una conversión silenciosa a mercado.
- `apply_cash_flow(amount)`: depósito (+) / retirada o transferencia (−) ajena a la operativa; se traslada al
  máximo de equity y a la referencia del día (no es ganancia ni pérdida). La detecta la mesa.
- Utilidades: `estimated_liq_px(entry, lev, is_long, mmr)`, `maintenance_rate(settings, max_lev)`, `LIQ_BUFFER`.

### `desk/engine.py` — `ExecutionEngine`
- `ExecutionEngine(broker, market, settings, clock, journal=None, *, load=True)`; estado persistible de brackets
  (`journal.save_json("engine")`, se recarga al construir; un fichero ilegible se aparta y se ignora).
- Un `Bracket` por moneda: `PENDING` (entrada enviada) → `OPEN` (posición con SL/TP) → `CLOSING` (cierre enviado)
  → `ClosedTrade` (coin, side, origin, size, entry_px, exit_px, `pnl` neto = realizado − comisiones, gross_pnl,
  fees, exit_reason `sl|tp|close|close_all|flatten|kill|no_sl|liquidation|external`, opened_ts, closed_ts).
- `open(intent, decision, *, cmd_id="") -> Order`: fija apalancamiento (un `ValueError` del broker → orden
  `REJECTED` local), registra y **guarda** el bracket pendiente {coin, sl, tp, origin, expires_ts (sólo límites,
  `entry_ttl_minutes`)} y *después* coloca la entrada (ZONE/LIMIT → LIMIT Gtc en `decision.entry_px`; MARKET →
  MARKET con peor precio `entry_px ± market_slippage_bps`). Rechaza si ya hay un bracket en la moneda. Un
  `BrokerError` al enviar se propaga pero el bracket queda *sin confirmar* (`is_unconfirmed(coin)`): `reconcile`
  busca la orden en el exchange (misma moneda, lado y tamaño) o la posición y la protege con el SL/TP aprobados,
  o la descarta tras 2 comprobaciones.
- `on_fills(fills) -> list[ClosedTrade]`: identifica cada fill por id de orden (entrada/SL/TP/cierre) y, si no,
  por etiqueta y dirección. Entrada (total o parcial) → coloca/ajusta STOP y TAKE_PROFIT reduce-only por el tamaño
  abierto (la orden nueva antes de cancelar la vieja). Posición plana tras SL/TP/cierre → cancela la otra (OCO) y
  el resto de entrada, escribe `trades.csv` y devuelve la operación (la mesa pasa `pnl` a
  `RiskManager.record_trade_result`). Fills sin bracket → aviso y diario. Idempotente ante la re-entrega del
  último instante ya aplicado (claves persistidas en `engine.json`); un fill anterior a la creación del bracket
  (− 5 s) no se le imputa. Lo que `reconcile` ya dio por ejecutado sin fill (`assumed`) no se suma dos veces.
  Cada moneda se procesa aparte: un `BrokerError` al proteger una no pierde las operaciones cerradas de otras.
- `reconcile(account) -> list[ClosedTrade]`: caduca entradas no ejecutadas; descarta entradas que desaparecen del
  exchange (2 comprobaciones); da por abierta una entrada cuya posición aparece sin fill; cierra brackets cuya
  posición desapareció (2 comprobaciones, `external`); re-coloca SL/TP que falten (con 5 s de gracia tras colocar);
  cierra la posición si el SL se rechaza 3 veces (o no se puede colocar por conexión 10 veces); adopta posiciones sin
  bracket (origen `adopted`, reutiliza STOP/TP reduce-only existentes o pone SL `default_sl_pct`, nunca más allá de
  la liquidación); cancela reduce-only huérfanas sin posición; reintenta (cada 10 s, hasta 6 envíos) los cierres
  `CLOSING` que dejaron posición. Una entrada que caduca o se cancela sólo se descarta si la cancelación se
  confirma (si no, queda `canceling` con su SL/TP). Cada moneda por separado (un fallo no frena a las demás).
- `close(coin, reason="close", *, account=None) -> Order|None` (IOC reduce-only con límite `CLOSE_SLIPPAGE` peor
  que el mid, o que el mark de la posición sin datos de mercado; crece con cada reintento), `close_all(reason=
  "close_all") -> list[Order]` (moneda a moneda: las que fallan van a `close_failures` {moneda: motivo} y no frenan
  a las demás; no cuenta las órdenes rechazadas), `set_stop(coin, px) -> str`, `set_take_profit(coin, px) -> str`
  (`ValueError` con el motivo si el nivel queda del lado equivocado, no hay posición/entrada, el SL queda más allá
  de la liquidación −con `LIQ_BUFFER`− o, al ensancharlo, la pérdida superaría `risk_per_trade_pct` del equity;
  sobre una entrada pendiente sólo cambia el nivel), `cancel(coin|None) -> int` (sólo órdenes de entrada
  confirmadas: los SL/TP de posiciones abiertas se mantienen), `reference_px(coin)`, `is_unconfirmed(coin)`.
- Mensajes legibles para el operador en `messages` / `pop_messages()`. Formato: `fmt_px`, `fmt_usd`, `fmt_size`.

### `desk/journal.py` — `Journal`
- `Journal(state_dir, clock=None)`: `event(kind, /, **data)` → `events.jsonl`; `fill(fill)` → `fills.csv`;
  `trade(record)` → `trades.csv`; `equity(ts, equity, available=None, positions=None)` → `equity.csv`;
  `save_json(name, obj)`/`load_json(name, default)` atómicos (tmp + fsync + rename; ilegible → `*.corrupt-<ts>`);
  `outbox(cmd_id, text)` → `outbox.log`, un renglón por línea: `<fecha hora> [<id>] <texto>` (avisos: id `desk`);
  `touch(name, text)` (sin fsync, atómico); `inbox_path`/`outbox_path`. `read_outbox(path, cmd_id, offset) ->
  (líneas, offset)`.
- Añadir a un diario nunca lanza (error de disco → log); `save_json` sí lanza `OSError`.

### `desk/desk.py` — `TradingDesk`
- `TradingDesk(settings, market, broker, clock, parser=None, journal=None, *, strategies=None, scanner=None,
  risk=None, engine=None, out=None)`; `run_forever(max_cycles=None, *, install_signals=True, before_cycle=None)
  -> int` y `run_cycle() -> bool` (False si alguna etapa falló).
- Ciclo (cada `loop_seconds`): leer comandos (cola en memoria + `state/inbox.txt` con offset persistido) →
  `broker.step()` → `poll_fills` → `engine.on_fills` → `account` → `risk.update` (halt/flatten) →
  `engine.reconcile` → cada `strategy_seconds`, si auto y no pausado/halted/cooldown: estrategias por
  símbolo → intents → riesgo → engine → guardar estado → línea de estado cada `status_every_seconds`.
- Cada etapa en su propio try/except; fallos en ciclos consecutivos → espera `loop_seconds·2^n` (máx. 300 s);
  nunca muere. Tras abrir (comando o señal) se sincronizan los fills al momento: una entrada a mercado queda
  protegida con SL/TP sin esperar al siguiente ciclo.
- `submit(text, source="consola", cmd_id=None) -> str` (respuesta legible, también a `outbox.log` con el id),
  `enqueue(text, cmd_id=None) -> id`. Inbox: offset guardado antes de ejecutar (como mucho una vez) junto a una
  huella de los últimos bytes leídos y los últimos `[id]` ejecutados; si el fichero encoge o se reescribe no se
  relee desde 0: sólo se ejecutan `[id]` no vistos (y se avisa). Sin `desk.json` válido el historial del inbox se
  ignora (salvo en un estado nuevo, sin `outbox.log`). Líneas `[id] texto` usan ese id; un `[id]` a mitad de línea
  separa dos comandos pegados; líneas a medio escribir esperan al salto de línea; `#` = comentario. Si la lectura
  del inbox falla, la cola en memoria se procesa igual. Una línea nueva en el inbox corta la espera entre ciclos.
- Un `BrokerError` tras enviar una entrada no es "inténtalo de nuevo": la respuesta dice ESTADO INCIERTO (no
  repetir) y la reconciliación la resuelve. Un fallo al colocar el SL tras la entrada se informa ("NO colocado")
  y se reintenta.
- Comandos de IA (`source="llm"`) que no sean de sólo lectura/freno (estado, ayuda, pausa, auto off) quedan
  pendientes de `si`/`no` 5 min si `llm_confirm_trades`.
- `SET_SL`/`SET_TP` con `%` se convierten a precio desde la entrada de la posición (o de la entrada pendiente).
- `SET_PARAM` y `auto on/off` se guardan en `desk.json` y se restauran al arrancar (`DESK_AUTO=false`/`--no-auto`
  siempre arrancan en OFF). `SET_PARAM` con `param=PARAM_DEFAULTS` vuelve a los valores de arranque.
- El estado de riesgo (`risk.json`) se guarda en cuanto cambia (comandos y eventos), no sólo en "guardar".
- Movimientos de saldo: entre dos lecturas de la cuenta, el cambio de equity que no explican las ejecuciones
  (realizado − comisiones) ni el PnL no realizado, con las posiciones cuadradas y por encima de 0,5 % (y del funding
  si el broker lo conoce: `funding_since(ts)`), es un depósito/retirada → `risk.apply_cash_flow` + aviso. Una
  salida sólo se clasifica en un intervalo sin ejecuciones (con fills cuenta como pérdida: ante la duda, el halt).
- KILL = `risk.halt("kill", flatten=True)` + **guardar `risk.json`** + cancelar entradas + `close_all("kill")`; la
  respuesta lista las monedas sin cerrar. Halts → cancelar entradas pendientes; evento "flatten" →
  `close_all("flatten")` (se repite cada ciclo mientras quede exposición).
- La operativa automática evalúa cada estrategia una vez por vela (por moneda) y sólo en símbolos sin posición,
  sin entrada pendiente y sin bracket: las posiciones abiertas por comando (origen "command") no las toca. La vela
  evaluada por (moneda, estrategia) se guarda en `desk.json` (`bar_seen`): un reinicio no repite la señal.
- Salud: al final de cada ciclo `health.json` (`health()`: ts, ciclo, racha de fallos, etapas que fallan con su
  último error, última vez que `cuenta` y `fills` terminaron bien) y `heartbeat` sólo si el ciclo leyó la cuenta y
  los fills. Sin cuenta, `status.json` se reescribe con una línea "SIN DATOS DE LA CUENTA".
- `status_line()` (`equity | PnL día | posiciones | auto ON/OFF | halt`), `status_report()`, `summary()`,
  `request_stop()`, `next_delay()`, `health()`. Estado: `desk.json`, `risk.json`, `engine.json`, `status.json`,
  `health.json`, `heartbeat`.
- SIGINT/SIGTERM → guardar estado y salir limpio (las órdenes SL/TP quedan en el exchange); segunda señal fuerza.

### `desk/__main__.py` — CLI
- `python -m desk run [--mode paper|live] [--data hyperliquid|sim|replay:x.csv] [--symbols ...] [--no-auto]
  [--fast] [--fresh] [--cycles N] [--loop-seconds S] [--cmd "N:texto"] [--state-dir DIR] [--no-input] [-v]`
  - `--fast`: `SimClock` (sólo datos sim/replay); estado en `<state_dir>/fast`, continúa la simulación anterior
    desde la hora más reciente guardada (`clock_ts` de `profile.json`, `health.json`, `heartbeat`, `status.json`:
    también tras un SIGKILL) salvo `--fresh`.
  - `profile.json` guarda modo/datos/reloj del estado y, en live, la red (`network`: testnet/mainnet) y la cuenta
    (`account`): arrancar con otra configuración sobre el mismo estado se rechaza (no mezclar posiciones paper de
    sim con datos reales, paper con live, ni testnet con mainnet u otra cuenta). Un perfil antiguo sin red/cuenta
    se acepta y se completa. `build_profile(settings, fast)`, `check_profile(journal, profile)`.
  - Consola interactiva si stdin es un terminal (`salir` detiene la mesa).
- `python -m desk send "long HYPE" [--timeout 30] [--no-wait] [--fast|--state-dir]` (añade `[cli-xxxx] texto` al
  inbox con `append_inbox_line`: con bloqueo y anteponiendo un salto si la última línea no lo tenía; espera la
  respuesta con ese id en `outbox.log`).
- `python -m desk status [--fast|--state-dir]` (lee `status.json` y los avisos de `health_check`).
- `python -m desk health [--fast|--state-dir] [--max-age 600] [--max-stale 600]`: 0 si la mesa completa ciclos
  (`health.json` reciente) y lee la cuenta y los fills; 1 si no (HEALTHCHECK del Dockerfile).
  `build_broker` pasa al broker live `fills_cursor_ts` y `fills_seen` guardados en `desk.json`.

---

# Fase 2: backtest por gráfica, playbook, alarmas e interactividad

Campos nuevos ya añadidos en `desk/models.py` (`CommandAction.BACKTEST/ADOPT/DEACTIVATE/PLAYBOOK/REPORTS/
ALERT_ADD/ALERT_LIST/ALERT_DEL/MUTE/UNMUTE`; `Command.interval/strategy/days/alert/target_id`) y en
`desk/config.py` (`bt_*`, `auto_default_strategies`, `alert_*`, `daily_loss_warn_frac`, `telegram_*`, `webhook_url`).

## `desk/strategy.py` (refactor: una sola lógica para vivo y backtest)
- Cada estrategia declara `PARAM_GRID: dict[str, list]` (valores a probar) y sus parámetros como atributos.
  `Strategy(settings, clock=None, *, interval=None, **params)` permite fijar temporalidad y parámetros
  (nombre desconocido -> `ValueError`). `params() -> dict` devuelve los valores actuales.
- `signals(bars, zones=None) -> pd.DataFrame` **vectorizado**, mismo índice que `bars`, columnas
  `side` ("long"/"short"/""), `entry_mode` ("market"/"zone"/"limit"), `entry_px`, `sl`, `tp`, `reason`.
  La fila i sólo usa datos de filas <= i (sin look-ahead). `zones` es un DataFrame por vela con
  `long_low, long_high, long_usd, short_low, short_high, short_usd` (NaN = sin zona).
- En vivo, `_evaluate_closed` = última fila de `signals(bars, zones_df)` donde la última fila de zonas
  viene del `LiquidityScanner` (paridad exacta vivo/backtest).
- `liquidity.historical_zones(bars, settings) -> DataFrame`: zonas aproximadas para el pasado (no hay libro
  ni liquidaciones históricas): clusters de máximos/mínimos de pivote confirmados (sin look-ahead) con
  volumen como peso. Documentar la limitación.
- Estrategias nuevas registradas en `STRATEGIES`: `ema_trend` (cruce EMA rápida/lenta + filtro ADX),
  `rsi_reversion` (RSI extremo + toque de Bollinger, objetivo media), `donchian_breakout` (ruptura de canal
  N velas + stop ATR). Todas con SL/TP coherentes y `PARAM_GRID`.

## `desk/backtest.py`
- `BTTrade(coin, side, entry_ts, entry_px, exit_ts, exit_px, size, pnl, pnl_pct, r_multiple, fees, exit_reason, bars_held)`;
  `exit_reason` in {"sl","tp","end","flip"}.
- `BacktestResult(coin, interval, strategy, params, start_ts, end_ts, bars, trades, equity: pd.Series, metrics: dict)`
  con `to_dict()` (equity muestreada a <= 500 puntos).
- `metrics`: `total_return_pct, cagr_pct, sharpe, sortino, max_drawdown_pct, calmar, win_rate_pct, profit_factor,
  expectancy_r, n_trades, avg_trade_pct, avg_bars_held, exposure_pct, best_trade_pct, worst_trade_pct, fees_usd`.
- `run_backtest(strategy, bars, settings, *, coin, meta=None, start_equity=None, zones=None) -> BacktestResult`:
  señal al cierre de la vela i -> MARKET al open de i+1 ± `paper_slippage_bps` (comisión taker);
  ZONE/LIMIT: límite en `entry_px` válido `entry_ttl_minutes`, se ejecuta si la vela lo toca (comisión maker);
  una posición a la vez; SL/TP intravela, si una vela toca ambos -> **SL primero**; hueco a través del SL ->
  ejecución al open (peor). Dimensionamiento igual que el riesgo en vivo (`auto_units`, `unit_margin_usd`,
  apalancamiento min(settings, meta), recorte por `risk_per_trade_pct` en el SL y `max_notional_per_coin_usd`).
  5000 velas con un juego de parámetros en < 100 ms.

## `desk/optimizer.py`
- `StrategyEvaluation(strategy, params, is_metrics, oos_metrics, oos_trades, folds, score, passed, reasons)`.
- `OptimizationReport(id, coin, interval, days, bars, created_ts, evaluations, winner, notes)` con
  `to_dict()/from_dict()` y `summary_text()` (tabla en español: dentro vs fuera de muestra).
- `optimize(coin, interval, bars, settings, *, strategies=None, meta=None, seed=7, progress=None, should_stop=None)`:
  walk-forward anclado de `bt_folds` tramos; en cada tramo se elige la mejor combinación (<= `bt_max_combos`,
  muestreo determinista) en el tramo de entrenamiento y se mide en el siguiente (fuera de muestra). Los
  parámetros propuestos para adoptar = mejores del último entrenamiento. Aprobada sólo si fuera de muestra:
  PF >= `bt_min_profit_factor`, retorno > 0, operaciones >= `bt_min_trades`, DD <= `bt_max_drawdown_pct`;
  `reasons` explica cada rechazo. Ranking por `bt_objective` fuera de muestra. `winner` = mejor aprobada o None.
- `ReportStore(state_dir)`: `save(report, result=None) -> path` en `state/reports/<COIN>_<interval>_<id>.json`,
  `latest(coin, interval)`, `list(limit)`, `get(id)`.

## `desk/playbook.py`
- `PlaybookEntry(coin, interval, strategy, params, units, adopted_ts, report_id, oos_metrics, enabled=True)`.
- `Playbook(state_dir)` (`state/playbook.json`, escritura atómica): `adopt(entry)` (una entrada por moneda+temporalidad),
  `remove(coin, interval=None) -> int`, `entries()`, `strategies(settings, clock) -> list[(PlaybookEntry, Strategy)]`.
- La operativa automática usa las entradas del playbook (más las estrategias por defecto sobre `symbols` si
  `auto_default_strategies`). Cada entrada opera con su temporalidad y sus parámetros.

## `desk/report_render.py`
- `render_html(report, result, bars) -> str`: HTML autocontenido (SVG en línea, sin dependencias): precio con
  entradas/salidas, curva de capital, drawdown y tabla de estrategias (dentro vs fuera de muestra). Claro y oscuro.
- `render_png(report, result, bars) -> bytes | None` (matplotlib si está instalado, backend Agg).
- `sparkline(values, width=40) -> str` (bloques unicode) para consola y Telegram.

## `desk/alerts.py`
- `AlertRule(id, coin, kind, params, interval, repeat, cooldown_s, created_ts, last_fired_ts, fired_count, note)`.
- Tipos (`Command.alert["kind"]`): `price_above`, `price_below`, `price_cross` (`level`); `rsi_below`, `rsi_above`
  (`level`, `period`=14); `ma_cross` (`fast`, `slow`, `ma`="sma"|"ema", `direction`="up"|"down"|"any");
  `zone` (`side`="long"|"short"|None: el precio entra en la zona de liquidez); `volume_spike` (`mult`, `lookback`=20);
  `move_pct` (`pct`, ventana = `interval`). `Command.alert` = `{"kind":..., <params>, "interval":..., "repeat": bool}`.
- `AlertManager(settings, clock, state_dir, notifier)`: `add(coin, spec) -> AlertRule`, `remove(id) -> bool`, `clear() -> int`,
  `rules()`, `evaluate(market, scanner=None) -> list[str]` (cada ciclo; nunca lanza; indicadores sobre velas cerradas
  con caché por ciclo), `system(kind, text, level="info", key=None, coin=None)` (avisos de la mesa con anti-spam por
  `key` y `alert_cooldown_seconds`), `mute(minutes|None)`, `unmute()` (los críticos nunca se silencian).
- Persistencia: reglas en `state/alerts.json`; cada aviso disparado en `state/alerts.jsonl`
  (`{"ts","level","kind","coin","text","rule_id"}`), que leen el panel y la consola.

## `desk/notify.py`
- `Notifier(channels)`: `send(text, level="info", kind="", coin=None, image=None)`; envío de red en un hilo con cola
  (la mesa nunca se bloquea por HTTP); errores de red se registran y se tragan.
- Canales: `ConsoleChannel(out, bell)` (colores con rich si está disponible), `FileChannel(path)` (jsonl),
  `TelegramChannel(token, chat_ids, min_level, session=None)` (`sendMessage`, `sendPhoto` si hay imagen),
  `WebhookChannel(url, min_level, session=None)` (JSON con `content` y `text`: vale para Discord y Slack).
- Niveles: "info" < "warn" < "critical".

## `desk/telegram_bot.py`
- `TelegramBot(token, chat_ids, submit, clock=None, session=None)`: long polling de `getUpdates` (timeout 25 s) en un
  hilo daemon; sólo acepta mensajes de `chat_ids` (el resto se ignora sin responder y se registra);
  `/start` y `/help` -> ayuda; un `/comando` se traduce a texto (`/estado` -> "estado"); cada texto se pasa a
  `submit(text, source="telegram") -> str` y la respuesta vuelve al chat. `start()`, `stop()`. Nunca registra el token.

## `desk/tui.py` y `desk/console.py`
- `python -m desk dashboard`: panel `rich.live` de sólo lectura sobre los archivos de estado (`status.json`,
  `alerts.jsonl`, `playbook.json`, índice de informes, `events.jsonl`): cabecera (modo, equity, PnL del día, barras
  de consumo de pérdida diaria y drawdown, auto/halt), posiciones con PnL en color, órdenes, playbook, últimas
  alarmas y señales, último backtest. Refresco 1 s; tolera archivos ausentes o a medio escribir.
- `python -m desk console`: consola interactiva conectada a una mesa en marcha (inbox/outbox): historial readline,
  autocompletado de verbos/monedas/temporalidades, respuestas en color y alarmas en tiempo real (hilo que sigue
  `alerts.jsonl` y los avisos `desk` de `outbox.log`).
- `status.json` (lo escribe la mesa) se amplía con: posiciones, órdenes, playbook, contadores de alarmas, últimas
  señales y consumo de límites de riesgo en %.

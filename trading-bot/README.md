# Mesa de trading automatizada (`desk`)

Bot de trading para perpetuos de **Hyperliquid** organizado como la mesa de una firma: recibe órdenes en
lenguaje natural (español o inglés), las pasa siempre por un gestor de riesgo con reglas de cuenta de fondeo,
ejecuta con stop-loss y take-profit obligatorios, y además **opera solo 24/7** con estrategias propias, le des
comandos o no.

- **Comandos**: `long HYPE`, `short btc 2u a mercado sl 1% tp 3%`, `cierra todo`, `kill`, `estado`...
  Desde la consola, desde otro terminal (`python -m desk send "..."`) o escribiendo en `state/inbox.txt`.
- **Entradas en zonas de liquidez**: por defecto un `long` no compra a mercado: busca la zona con más
  liquidez (muros del libro + niveles de liquidación) por debajo del precio y deja una orden límite ahí.
- **Riesgo tipo firma de fondeo**: 1 % de riesgo por operación, pérdida diaria máx. 3 %, drawdown máx. 5 %,
  máximo de posiciones, cooldown tras rachas de pérdidas. Todo ajustable.
- **Ejecución profesional**: SL y TP reduce-only en cuanto se ejecuta la entrada (OCO), caducidad de
  órdenes límite, reconciliación con el exchange en cada ciclo, adopción de posiciones tras un reinicio.
- **24/7**: bucle que no muere (cada etapa aislada, backoff exponencial ante fallos de red), estado en disco,
  apagado limpio con SIGTERM, listo para systemd o Docker.
- **Modos**: `paper` (dinero simulado con precios reales, por defecto), `sim`/`replay` (todo offline, días
  simulados en segundos) y `live` (dinero real; testnet por defecto).

---

## ⚠️ Aviso de riesgo

Esto es software experimental. Operar con apalancamiento puede hacerte perder **todo** el capital, y más
rápido de lo que crees. Ninguna estrategia de este repositorio tiene rentabilidad demostrada; los resultados
en `paper` o `sim` no garantizan nada en real (deslizamiento, liquidez, caídas de la API, errores del propio
bot). Úsalo primero en `paper` y en **testnet**, durante semanas, y no arriesgues dinero que no puedas perder.
Nada de esto es asesoramiento financiero.

---

## Instalación

Requiere Python 3.11+ (probado con 3.13).

```bash
cd trading-bot
python3 -m venv venv
source venv/bin/activate            # en Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # opcional: todos los valores por defecto son prudentes
```

`hyperliquid-python-sdk`, `eth-account` y `anthropic` sólo hacen falta para el modo live y para la IA;
el resto de la mesa funciona sin ellos.

---

## Arranque rápido

### 1. Simulación offline (sin red, días en segundos)

```bash
python -m desk run --data sim --fast --cycles 3000 --cmd "0:long HYPE" --cmd "200:estado"
```

- `--data sim`: mercado simulado determinista (BTC, ETH, SOL, HYPE) con libro, muros de liquidez y
  liquidaciones sintéticas.
- `--fast`: reloj simulado; cada ciclo avanza `DESK_LOOP_SECONDS` (10 s) sin esperar. 3000 ciclos ≈ 8 h
  simuladas en unos segundos. Con `--loop-seconds 60 --cycles 2880` simulas 2 días.
- `--cmd "N:texto"`: envía un comando antes del ciclo N (repetible).
- El estado va a `state/fast/` y la siguiente ejecución **continúa** donde lo dejó; `--fresh` empieza de cero.
- También puedes reproducir velas reales: `--data replay:datos/btc_15m.csv` (columnas
  `timestamp,open,high,low,close,volume`).

Al terminar imprime un resumen (`Sesión: 3000 ciclos | operaciones cerradas ... | PnL neto ...`) y deja
todos los diarios en `state/fast/` (ver [Archivos y logs](#archivos-y-logs)).

### 2. Paper con precios reales de Hyperliquid (24/7)

```bash
python -m desk run                  # = --mode paper --data hyperliquid
```

Usa los datos públicos de Hyperliquid (mainnet) y una cuenta simulada de 10.000 USD
(`DESK_PAPER_START_EQUITY`) con comisiones y deslizamiento realistas. Escribe comandos directamente en la
consola; `salir` (o Ctrl+C) la detiene guardando el estado. Opciones útiles:

```bash
python -m desk run --symbols BTC,ETH,SOL --no-auto      # sólo comandos, sin operativa automática
python -m desk run -v                                   # log detallado en consola
```

Niveles de liquidación reales (opcional): pon tu `MOONDEV_API_KEY` en `.env`; sin ella las zonas se
calculan sólo con el libro de órdenes.

### 3. Hablarle desde otro terminal

```bash
python -m desk send "long HYPE 2u zona sl 1.5% tp 4%"   # espera la respuesta (30 s)
python -m desk send --no-wait "cierra todo"
python -m desk status                                   # último estado guardado
echo "estado" >> state/inbox.txt                         # también vale (una línea por comando)
```

(Añade `--fast` a `send`/`status` para hablar con una simulación `--fast`.)

---

## Comandos

Español o inglés, sin importar mayúsculas ni acentos; decimales con punto o coma. Si las reglas no entienden
el texto y hay `ANTHROPIC_API_KEY`, la mesa lo interpreta con IA (Claude) y **pide confirmación** (`si`/`no`)
antes de ejecutar cualquier cosa que abra, cierre o cambie algo.

### Abrir posiciones

| Comando | Qué hace |
|---|---|
| `long HYPE` · `compra SOL` · `short btc` · `vende ETH` | Abre con la entrada por defecto (zona de liquidez), 1 unidad |
| `long HYPE 2u` · `2 unidades` · `2 units` | Tamaño en unidades (1 u = `DESK_UNIT_MARGIN_USD` de margen × apalancamiento) |
| `long HYPE zona` · `en zona` · `en zona de liquidez` | Límite en el centro de la zona de liquidez más fuerte |
| `long HYPE a mercado` · `market` · `ya` | Entrada inmediata (IOC con precio máximo protegido) |
| `long HYPE a 85.2` · `@85.2` · `limit 85,2` | Orden límite a ese precio |
| `... sl 1.5%` · `... stop 84.5` | Stop-loss en % desde la entrada o a un precio |
| `... tp 4%` · `... tp 90` | Take-profit en % o a un precio (por defecto: `DESK_DEFAULT_TP_R` R) |
| `long HYPE 2u zona sl 1.5% tp 4%` | Todo combinado |
| `long xyz:GOLD a mercado` | Activos HIP-3 con prefijo de dex |

Respuesta típica:

```
HYPE LONG 11.72 @ 85.288 (zona liq 85.241-85.336, $1,000 notional, 2x aislado) SL 84.009 TP 87.846 | riesgo en el SL ≈ $15.88 (0.16% del equity)
Orden límite en espera (caduca en 240 min); SL y TP se colocan al ejecutarse.
Avisos: SL por defecto a 1.5% de la entrada; apalancamiento recortado de 5x a 2x (máximo de HYPE)
```

y cuando el precio llega: `HYPE LONG: entrada ejecutada: 11.72 @ 85.288` → `HYPE: SL 84.009 colocado, TP
87.846 colocado` → `HYPE LONG cerrada por take-profit: ... PnL neto +$30.91 ... SL cancelado (OCO)`.

### Gestionar

| Comando | Qué hace |
|---|---|
| `cierra HYPE` · `close HYPE` | Cierra la posición a mercado (y cancela su entrada pendiente) |
| `cierra todo` · `close all` · `flat` | Cierra todas las posiciones y cancela todas las entradas |
| `cancela HYPE` · `cancel all` · `cancela todo` | Cancela órdenes de **entrada** (los SL/TP de posiciones abiertas se mantienen) |
| `stop HYPE 84.5` · `sl HYPE 84.5` · `mueve el stop de HYPE a 84` | Mueve el SL de la posición (o de la entrada pendiente) |
| `sl HYPE 1%` · `tp HYPE 3%` · `tp HYPE 90` | SL/TP en % desde la entrada o a precio |

### Control de la mesa

| Comando | Qué hace |
|---|---|
| `estado` · `status` · `posiciones` | Equity, PnL del día, drawdown, posiciones con SL/TP, órdenes, entradas pendientes |
| `pausa` · `pause` | No abre nada nuevo (las posiciones abiertas se siguen gestionando) |
| `reanuda` · `resume` | Quita la pausa (no levanta un halt de riesgo) |
| `auto on` · `auto off` · `automatico on` | Activa/desactiva la operativa autónoma |
| `kill` · `panico` · `apaga todo` | **Pánico**: cierra todo, cancela todo y detiene la mesa hasta `reset halt` |
| `reset halt` | Levanta un halt de riesgo (diario, drawdown o kill) y reinicia las referencias |
| `si` · `confirmar` · `yes` — `no` · `cancelar` · `reject` | Confirma / descarta el comando pendiente interpretado por IA (5 min) |
| `ayuda` · `help` · `?` | Chuleta de comandos |

### Parámetros en caliente

Se aplican a las siguientes operaciones y **se recuerdan tras reiniciar** (aparecen en `estado`).

| Comando | Parámetro | Rango |
|---|---|---|
| `apalancamiento 3` · `leverage 3` | `leverage` | 1–50 (se recorta al máximo de cada moneda) |
| `unidad 250` | `unit_margin_usd` | 1–1.000.000 |
| `riesgo 0.5` | `risk_per_trade_pct` | 0,05–10 |
| `sl 2%` | `default_sl_pct` | 0,1–20 |
| `tp 3` | `default_tp_r` (en R) | 0,5–10 |
| `posiciones 4` | `max_open_positions` | 1–20 |
| `perdida diaria 2` | `daily_loss_limit_pct` | 0,5–50 |
| `drawdown 4` | `max_drawdown_pct` | 1–90 |
| `ttl 60` | `entry_ttl_minutes` | 1–10.080 |

---

## Cómo opera sola 24/7

Con `DESK_AUTO=true` (por defecto) la mesa evalúa sus estrategias sobre `DESK_SYMBOLS` cada
`DESK_STRATEGY_SECONDS`, **una vez por vela cerrada** (nunca con la vela en curso):

- **`liquidity_sweep` (15 m)** — reversión tras un barrido de liquidez: la vela anterior perfora la zona de
  liquidez (bids/liquidaciones de longs bajo el precio) y la última cierra de vuelta por encima, con RSI(14)
  < 40 y girando al alza → LONG a mercado, SL bajo la zona − 0,5·ATR, TP a `DESK_DEFAULT_TP_R` R.
  Simétrico para SHORT (RSI > 60 y bajando).
- **`squeeze_breakout` (1 h)** — fin de una compresión de volatilidad (Bollinger dentro de Keltner) con
  ADX(14) > 25 y cierre fuera de Bollinger → entrada en la dirección de la ruptura, SL a 1,5·ATR.

Reglas de la operativa automática:

- Cada señal pasa por el **mismo gestor de riesgo** que los comandos, con `DESK_AUTO_UNITS` unidades.
- Sólo abre en monedas **sin** posición, sin entrada pendiente y sin operación gestionada: nunca toca ni
  cierra posiciones abiertas por comando. Cada operación automática sale por su propio SL/TP.
- Se para con `pausa`, `auto off`, cualquier halt de riesgo o el cooldown tras `DESK_MAX_CONSECUTIVE_LOSSES`
  pérdidas seguidas (los comandos siguen funcionando durante el cooldown).

---

## Gestión de riesgo

Todo lo que **abre** exposición pasa por el `RiskManager`; cerrar, mover stops y el pánico siempre están
permitidos. Reglas por defecto (estilo cuenta de fondeo):

| Regla | Por defecto | Qué pasa |
|---|---|---|
| Riesgo por operación | 1 % del equity | El tamaño se recorta para que la pérdida en el SL (con comisiones) no lo supere |
| Stop obligatorio | sí | Sin SL explícito se pone a `DESK_DEFAULT_SL_PCT` (1,5 %) |
| **Pérdida diaria** | **3 %** | Halt de entradas nuevas hasta el siguiente día UTC (con `DESK_FLATTEN_ON_DAILY_LIMIT=true`, cierra todo) |
| **Drawdown máximo** | **5 %** desde el máximo de equity | Cierra todo, cancela todo y detiene hasta `reset halt` |
| Posiciones simultáneas | 3 | Las entradas pendientes cuentan |
| Exposición | 25.000 USD por moneda, 50.000 USD total | Recorte del tamaño |
| Racha de pérdidas | 4 seguidas | 120 min sin entradas automáticas |
| Calidad de mercado | spread ≤ 25 bps, datos ≤ 120 s | Si no, no entra |
| Apalancamiento | `DESK_LEVERAGE` (5x), aislado | Se baja si la liquidación quedaría antes que el SL |

Dimensionamiento: `notional = unidades × DESK_UNIT_MARGIN_USD × apalancamiento`, recortado por las reglas
anteriores y por el margen disponible. Ejemplo: `long HYPE` con 500 USD/unidad y HYPE limitado a 2x → 1.000 USD
de notional, SL al 1,5 % → riesgo ≈ 16 USD (0,16 %).

## Ejecución

- **Entrada**: zona/límite → orden límite Gtc (caduca a los `DESK_ENTRY_TTL_MINUTES`); mercado → IOC con
  precio máximo (`DESK_MARKET_SLIPPAGE_BPS`).
- **Brackets**: al ejecutarse la entrada se colocan STOP y TAKE-PROFIT *reduce-only* por el tamaño abierto
  (ajustados si la entrada se llena por partes: primero la orden nueva, luego se cancela la vieja).
- **OCO**: si salta el SL se cancela el TP y viceversa; el PnL neto (con comisiones) se registra en
  `trades.csv` y alimenta el control de rachas.
- **Reconciliación cada ciclo**: re-coloca SL/TP que falten, cancela órdenes reduce-only huérfanas, detecta
  posiciones cerradas fuera de la mesa y **adopta** posiciones sin bracket (p. ej. tras perder el estado)
  poniéndoles el SL por defecto. Si el exchange rechaza el SL 3 veces seguidas, la posición se cierra.

---

## Pasar a live (dinero real): checklist

1. ☐ Semanas en `paper` con la configuración que vas a usar; revisa `trades.csv` y `equity.csv`.
2. ☐ Crea una **API wallet** (agente) en Hyperliquid para el bot: firma órdenes pero no puede retirar fondos.
   Pon su clave en `HL_PRIVATE_KEY` y la dirección de tu cuenta principal en `HL_ACCOUNT_ADDRESS`.
3. ☐ **Testnet primero** (es el valor por defecto en live): consigue fondos de testnet y ejecuta
   ```bash
   DESK_MODE=live DESK_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY python -m desk run --no-auto
   ```
   Prueba `long BTC a mercado`, `estado`, `stop BTC ...`, `cierra todo`, `kill`, un reinicio con posición
   abierta (debe conservar su SL) y comprueba todo en la web de Hyperliquid.
4. ☐ Ajusta el riesgo a la cuenta real (`unidad`, `riesgo`, `perdida diaria`, `drawdown`); empieza pequeño.
5. ☐ Activa la operativa automática en testnet unos días (`auto on`).
6. ☐ Sólo entonces mainnet: `DESK_HL_MAINNET=true` (la consola lo anuncia: `MODO LIVE ... MAINNET (DINERO REAL)`).
7. ☐ Vigila: `python -m desk status`, `state/outbox.log`, alertas si el healthcheck falla. Ten a mano `kill`.

El modo live exige `DESK_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY`, `HL_PRIVATE_KEY` y datos de Hyperliquid; la
clave nunca se escribe en logs ni en el estado. Al detener la mesa, los SL/TP quedan puestos en el exchange.

---

## Correrlo 24/7

**systemd** (servidor Linux; se reinicia solo si cae o si reinicia la máquina):

```bash
sudo cp deploy/desk.service /etc/systemd/system/      # ajusta rutas y usuario dentro
sudo systemctl daemon-reload && sudo systemctl enable --now desk
journalctl -u desk -f                                   # salida en vivo
```

**Docker**:

```bash
docker build -f deploy/Dockerfile -t desk .
docker run -d --name desk --restart=always --env-file .env -v "$PWD/state:/app/state" desk
docker exec -it desk python -m desk send "estado"
docker logs -f desk
```

**tmux** (rápido, en tu máquina o un VPS):

```bash
tmux new -s desk 'source venv/bin/activate && python -m desk run'
# Ctrl+B D para salir sin pararla; tmux attach -t desk para volver y escribir comandos
```

En los tres casos `SIGTERM`/Ctrl+C termina el ciclo en curso, guarda el estado y sale; al arrancar de nuevo
recupera posiciones, brackets, límites de riesgo del día y comandos pendientes del inbox (cada línea del inbox
se ejecuta como mucho una vez).

---

## Archivos y logs

```
trading-bot/
├── desk/
│   ├── __main__.py        CLI: run / send / status
│   ├── desk.py            TradingDesk: bucle 24/7, comandos, operativa automática, estado
│   ├── engine.py          ExecutionEngine: entradas, brackets SL/TP, OCO, reconciliación
│   ├── risk.py            RiskManager: tamaño, halts diario/drawdown, cooldown, pausa
│   ├── strategy.py        Estrategias autónomas (liquidity_sweep, squeeze_breakout)
│   ├── liquidity.py       Zonas de liquidez (libro + liquidaciones)
│   ├── commands.py        Comandos ES/EN por reglas + fallback IA
│   ├── journal.py         Diarios y estado en disco
│   ├── indicators.py      RSI, ATR, Bollinger, Keltner, ADX...
│   ├── models.py config.py clock.py
│   └── exchange/          hyperliquid_data (datos), paper (broker simulado), sim (offline), hyperliquid_live
├── tests/                 Tests (sin red): python -m pytest -q
├── deploy/                Dockerfile y desk.service (systemd)
├── .env.example           Toda la configuración comentada
└── ARCHITECTURE.md        Contrato entre módulos
```

En `state/` (o `DESK_STATE_DIR`; `state/fast/` con `--fast`):

| Fichero | Contenido |
|---|---|
| `outbox.log` | Respuestas a comandos con su id (`[c-000001]`, `[cli-ab12cd34]`) y avisos de la mesa (`[desk]`) |
| `inbox.txt` | Comandos de entrada (una línea cada uno; `[mi-id] texto` para elegir el id) |
| `trades.csv` | Operaciones cerradas: entrada, salida, motivo (sl/tp/close/flatten/...), PnL neto y comisiones |
| `fills.csv` | Cada ejecución |
| `equity.csv` | Curva de equity (cada minuto) |
| `events.jsonl` | Todo lo que pasa: comandos, decisiones de riesgo, órdenes, halts, señales |
| `desk.log` | Log técnico (rota a los 5 MB) |
| `status.json` | Último estado (lo muestra `python -m desk status`) |
| `heartbeat` | Latido de cada ciclo (healthcheck) |
| `paper_broker.json`, `engine.json`, `risk.json`, `desk.json`, `profile.json` | Estado para sobrevivir a reinicios |

Para empezar de cero en paper: detén la mesa y borra `state/`.

---

## Tests

```bash
python -m pytest -q
```

Todos corren sin red (mercados falsos/simulados y SDKs simulados), incluidos los flujos completos: comando →
orden en zona → ejecución → SL/TP → TP con OCO → PnL; stop-loss; cierra todo; kill; halts diario y de
drawdown con cierre; reinicio con estado persistido; inbox con offset; confirmación de comandos de IA;
operativa automática con pausa/auto off; y fallos de datos que no tumban el bucle.

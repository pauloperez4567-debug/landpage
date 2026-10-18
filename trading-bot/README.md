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
- **Backtest por gráfica**: `backtest HYPE 1h` prueba 5 estrategias con walk-forward (parámetros elegidos sólo
  con el pasado, medidos fuera de muestra) sin parar la mesa, y deja un informe HTML/PNG. `activa HYPE 1h` pone
  la ganadora en automático (sólo si aprobó fuera de muestra).
- **Alarmas y avisos**: `alerta BTC > 100000`, `alerta HYPE rsi < 30 1h`... y avisos de la mesa (ejecuciones,
  cierres con PnL, halts, consumo de límites) por consola, Telegram (que además acepta comandos) y
  Discord/Slack. Panel en vivo (`dashboard`) y consola interactiva (`console`).

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
el resto de la mesa funciona sin ellos. `rich` es para el panel y la consola en color; `matplotlib`, para el PNG
de los informes de backtest (sin él sólo se genera el HTML).

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
- El estado va a `state/fast/` y la siguiente ejecución **continúa** donde lo dejó (la hora simulada más
  reciente que guardó la mesa, también tras un corte brusco); `--fresh` empieza de cero.
- También puedes reproducir velas reales: `--data replay:datos/btc_15m.csv` (columnas
  `timestamp,open,high,low,close,volume`).
- Una simulación (datos `sim`/`replay` o `--fast`) **no usa Telegram ni el webhook** aunque estén en `.env`: con el
  mismo bot competiría con la mesa real por tus órdenes (un `kill` podría ejecutarlo la simulación) y mandaría
  avisos simulados a tu chat. `DESK_TELEGRAM_IN_SIM=true` los activa a propósito; entonces cada aviso y respuesta
  lleva delante la identidad de la mesa (`[SIM fast]`, `[PAPER]`, `[LIVE mainnet]`...).

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
python -m desk status                                   # último estado guardado y avisos de salud
python -m desk health                                   # 0 = sana; 1 = parada o sin ver el exchange
echo "estado" >> state/inbox.txt                         # también vale (una línea por comando)
```

(Añade `--fast` a `send`/`status`/`health` para hablar con una simulación `--fast`.)

Cada línea del inbox se ejecuta **una sola vez** (la mesa guarda hasta dónde leyó, una huella del fichero y
los `[id]` ya ejecutados). **No borres ni edites líneas** del inbox: si el fichero encoge o se reescribe, la
mesa avisa y sólo ejecuta las líneas con un `[id]` que no haya visto. Para anular un comando que ya se
ejecutó, envía el contrario (`cancela HYPE`, `cierra HYPE`).

### 4. Panel en vivo y consola interactiva

```bash
python -m desk dashboard            # panel (sólo lectura) que se refresca cada segundo; Ctrl+C para salir
python -m desk dashboard --once     # pinta una vez y sale (también sin terminal, p. ej. por ssh o en un log)
python -m desk console              # consola conectada a la mesa en marcha (por inbox/outbox)
```

- **Panel**: equity con mini-gráfica, PnL del día en verde/rojo, barras de consumo de la pérdida diaria y del
  drawdown (verde < 50 %, amarillo < 80 %, rojo), automático/pausa/halt, posiciones, órdenes, playbook, último
  backtest (**sólo métricas fuera de muestra**, con su advertencia), alarmas por nivel, señales, operaciones
  cerradas y avisos de la mesa. Sólo lee `state/`: se abre y se cierra sin afectar a la mesa, y un fichero
  ausente o a medio escribir no lo rompe.
- **Consola**: los mismos comandos que `send`, con historial (`state/.console_history`, flechas y Ctrl+R),
  Tab para autocompletar verbos, monedas y temporalidades, respuestas en color y las alarmas y avisos de la
  mesa en cuanto llegan. `salir` (o Ctrl+D) cierra **sólo la consola**: la mesa sigue en marcha.

(Ambos aceptan `--fast` y `--state-dir` como `send`.)

### 5. Backtest de una gráfica sin arrancar la mesa

```bash
python -m desk backtest --data sim --coin SOL --interval 1h --days 120      # offline, datos simulados
python -m desk backtest --coin HYPE --interval 15m --days 60               # velas reales de Hyperliquid
python -m desk backtest --csv datos/btc_1h.csv --coin BTC --interval 1h --strategy donchian
```

Imprime el resumen (dentro vs **fuera de muestra**) y guarda `reports/<MONEDA>_<tf>_<id>.json|html|png` en el
estado (`state/`, o `state/fast/` con datos `sim`/replay, para no mezclar informes de datos simulados con la mesa
real). Una mesa que use ese estado lo encuentra con `activa <MONEDA> <tf>`. Ver
[Backtest por gráfica, playbook y alarmas](#backtest-por-gráfica-playbook-y-alarmas).

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
| `cierra todo` · `close all` · `flat` · `cierra todas las posiciones abiertas` | Cierra todas las posiciones y cancela todas las entradas (`cierra todo BTC` es ambiguo y no hace nada) |
| `cancela HYPE` · `cancel all` · `cancela todo` | Cancela órdenes de **entrada** (los SL/TP de posiciones abiertas se mantienen) |
| `stop HYPE 84.5` · `sl HYPE 84.5` · `mueve el stop de HYPE a 84` | Mueve el SL de la posición (o de la entrada pendiente). Se rechaza si queda más allá de la liquidación o si, al ensancharlo, la pérdida superaría `DESK_RISK_PER_TRADE_PCT` |
| `sl HYPE 1%` · `tp HYPE 3%` · `tp HYPE 90` | SL/TP en % desde la entrada o a precio |

### Control de la mesa

| Comando | Qué hace |
|---|---|
| `estado` · `status` · `posiciones` | Equity, PnL del día, drawdown, posiciones con SL/TP, órdenes, entradas pendientes |
| `pausa` · `pause` | No abre nada nuevo (las posiciones abiertas se siguen gestionando) |
| `reanuda` · `resume` | Quita la pausa (no levanta un halt de riesgo) |
| `auto on` · `auto off` · `automatico on` | Activa/desactiva la operativa autónoma |
| `kill` · `panico` · `apaga todo` · `pánico ya` · `kill now` | **Pánico**: cierra todo, cancela todo y detiene la mesa hasta `reset halt`. El halt se guarda en disco antes de tocar el exchange; si una moneda no se puede cerrar (sin precio, red) se cierran las demás y se reintenta cada ciclo |
| `reset halt` | Levanta un halt de riesgo (diario, drawdown o kill) y reinicia las referencias |
| `si` · `confirmar` · `yes` — `no` · `cancelar` · `reject` | Confirma / descarta el comando pendiente interpretado por IA (5 min) |
| `ayuda` · `help` · `?` | Chuleta de comandos |

### Gráficas, informes y alarmas (resumen)

| Comando | Qué hace |
|---|---|
| `backtest HYPE 1h` · `backtest BTC 15m 90d` · `optimiza ETH 1h 3 meses` · `prueba SOL 1h ema_trend` | Backtest walk-forward de la gráfica (en segundo plano) |
| `informes` · `ultimos backtests` | Informes guardados (métricas fuera de muestra) |
| `activa HYPE 1h` · `activa HYPE 1h ema_trend` · `... forzar` | Opera esa gráfica en automático con la ganadora (o esa estrategia) |
| `desactiva HYPE 1h` · `desactiva HYPE` · `playbook` | Quita una gráfica / todas las de una moneda · lista las activas |
| `alerta HYPE > 90` · `alerta BTC rsi < 30 1h` · `alerta SOL zona long` | Alarmas de precio, indicadores y zonas |
| `alertas` · `borra alerta 3` · `borra alertas` · `silencio 30m` · `sonido` | Ver / borrar alarmas, silenciar avisos no críticos |

Detalles en [Backtest por gráfica, playbook y alarmas](#backtest-por-gráfica-playbook-y-alarmas).

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

`reset params` · `restablece parámetros` vuelve a los valores de arranque (`.env`) y olvida los cambiados por
comando (no hace falta borrar `desk.json`).

**Precios con punto de miles**: `83.825` se lee como decimal. Si la moneda cotiza cerca de 83.825 (BTC), la mesa
no envía nada y pide escribirlo sin separador (`83825`); además, un límite que cruza el libro más de un 1 % o
reposa a más de un 20 % del precio se rechaza por "¿precio mal escrito?" en vez de ejecutarse a mercado.

---

## Backtest por gráfica, playbook y alarmas

### Flujo recomendado

1. **Backtest** de la gráfica que te interesa: `backtest HYPE 1h` (o `python -m desk backtest ...`).
2. **Revisa el informe** (`reports/HYPE_1h_<id>.html` en el navegador): sólo las métricas **fuera de muestra**,
   cuántas operaciones son, el drawdown y si los tramos van todos en la misma dirección.
3. **Actívala en paper**: `activa HYPE 1h`. Sólo se activa lo aprobado fuera de muestra.
4. **Obsérvala** unos días o semanas (`playbook`, `estado`, panel): ¿se parece lo que hace a lo del informe?
5. **Testnet** (ver [Pasar a live](#pasar-a-live-dinero-real-checklist)) y sólo después, con poco, mainnet.

Un backtest no garantiza resultados futuros: describe cómo le habría ido a unas reglas en un pasado concreto,
con supuestos de ejecución. Lo de dentro de muestra está inflado por construcción; lo de fuera de muestra es
mejor indicio, pero el mercado cambia. Que una gráfica no tenga ganadora es un resultado normal y útil.

### Backtest por gráfica

`backtest COIN [tf] [días] [estrategia]` (temporalidad por defecto `1h`; días, `DESK_BT_DEFAULT_DAYS` = 120):

- La mesa comprueba al momento lo barato (moneda, temporalidad) y lanza el trabajo en un hilo propio, uno a la vez
  y en cola: la **descarga de las velas cerradas** (con la API lenta puede tardar decenas de segundos) y el cálculo
  van en ese hilo, así que **el bucle sigue vigilando posiciones, alarmas y comandos** (también un `kill` por
  Telegram) mientras tanto. Responde `Backtest HYPE 1h (120 días, ~2880 velas, 5 estrategias) en marcha...` y, al
  terminar, avisa con el resumen, la ruta del informe HTML/PNG y, si hay ganadora aprobada, `activa HYPE 1h`; un
  fallo de datos llega en ese aviso. Por Telegram llega además el PNG. Con `--fast` (reloj simulado) se ejecuta
  dentro del ciclo: determinista.
- **Capital supuesto = el equity de la cuenta** (sin cuenta, `DESK_PAPER_START_EQUITY`): el tamaño va en USD fijos
  (unidades), así que el retorno y el drawdown en % sólo significan algo con tu capital real. El informe lo dice.
- **Con los frenos del riesgo en vivo**: fuera de muestra el backtest aplica los mismos límites que la mesa, leídos
  de tu configuración (`DESK_MAX_DRAWDOWN_PCT`: cierra y no opera más; `DESK_DAILY_LOSS_LIMIT_PCT`: sin entradas
  hasta el día siguiente; cooldown tras `DESK_MAX_CONSECUTIVE_LOSSES`), con el equity mirado en el peor extremo de
  cada vela. Cambia tu perfil de riesgo y el backtest lo modela igual.
- Historia: días × velas por día, con tope de 5000 velas (lo que da Hyperliquid); si se recorta, se avisa.
- Al apagar la mesa, los backtests en marcha o en cola se cancelan (no se guarda un informe a medias).

Cómo leer el resumen:

```
                        DENTRO |      FUERA de muestra (lo que cuenta)
Estrategia         Ret %    PF |  Ret %    PF  DD %  Ops     t  Sharpe
rsi_reversion      +16.4  2.41 |   +8.4  1.63   3.1   41   1.6    2.91 ✘
donchian_breakout   +8.2  1.35 |   -3.7  0.87  11.2   71  -0.4   -0.82 ✘
...
Rechazadas:
  · rsi_reversion: ventaja no significativa: t = 1.64 < 2.42 (con 41 operaciones puede ser suerte)
```

- **DENTRO de muestra**: los parámetros se eligieron con esos mismos datos. Siempre sale mejor; no lo uses para
  decidir.
- **FUERA de muestra**: walk-forward anclado de `DESK_BT_FOLDS` tramos. En cada tramo se eligen los parámetros
  con el pasado y se miden en el bloque siguiente, que no vieron; la curva fuera de muestra encadena esos
  bloques. Comisiones y deslizamiento incluidos; funding y liquidaciones no.
- **✔ aprobada** sólo si fuera de muestra: PF ≥ `DESK_BT_MIN_PROFIT_FACTOR` (1,15), retorno > 0, ≥
  `DESK_BT_MIN_TRADES` (20) operaciones, drawdown ≤ `DESK_BT_MAX_DRAWDOWN_PCT` (25 %) **y por debajo de
  `DESK_MAX_DRAWDOWN_PCT`** (sin halt: en vivo la mesa se habría detenido), ventaja estadísticamente significativa
  (`t`: R medio por operación, corregido por todas las estrategias que se pueden probar en la gráfica, aunque pidas
  una sola) y **los parámetros que se adoptarían ganan en su propio tramo**. La curva encadenada mide el
  procedimiento (reoptimizar en cada tramo); lo que se activa son los parámetros del último entrenamiento, cuya
  única prueba fuera de muestra es el último tramo: retorno > 0 y PF ≥ el mínimo también ahí.
- `liquidity_sweep` sale **no validable**: en el backtest sus zonas son aproximadas (pivotes; no hay libro ni
  liquidaciones históricas) y en vivo vienen del libro y las liquidaciones, así que lo medido no es lo que opera.
  Sólo se activa con `forzar`.
  El ejemplo de arriba (datos simulados, un paseo aleatorio) no tiene ganadora aunque `rsi_reversion` ganara un
  8 %: con 41 operaciones eso puede ser suerte. Es exactamente lo que debe pasar sobre ruido.
- `informes` lista los últimos (con su id y la mejor estrategia fuera de muestra).

### Activar gráficas (playbook)

- `activa HYPE 1h` toma el **último informe** de esa gráfica y pone la ganadora en `state/playbook.json` con sus
  parámetros. `activa HYPE 1h ema_trend` elige otra estrategia del informe (si aprobó).
- Si no aprobó fuera de muestra, **no se activa**. `activa HYPE 1h [estrategia] forzar` la activa igualmente con
  una advertencia explícita (NO recomendado: lo más probable es que no tenga ventaja real).
- Tampoco sin `forzar` si **otro informe de los últimos 30 días** de la misma gráfica rechazó esa estrategia:
  repetir backtests (otros días, otra estrategia) hasta que uno apruebe convierte el ruido en "ganadoras".
- La respuesta (y `playbook`) enseña las dos cifras fuera de muestra: la del procedimiento (curva encadenada) y la
  de los parámetros adoptados en su tramo.
- Cada entrada del playbook opera **su** gráfica (temporalidad y parámetros), con `DESK_AUTO_UNITS` unidades,
  pasando por el mismo gestor de riesgo que todo lo demás. Las señales llevan el origen
  `auto:<estrategia>@<temporalidad>` (en `estado`, el panel y `trades.csv`).
- Además del playbook, si `DESK_AUTO_DEFAULT_STRATEGIES=true` (por defecto) siguen las estrategias de
  `DESK_STRATEGIES` sobre `DESK_SYMBOLS` **sin validar**, sólo en monedas sin gráficas en el playbook (una
  estrategia adoptada no compite por la moneda con otras: su secuencia de operaciones se parece a la de su
  backtest). Pon `false` para operar sólo lo aprobado (en live el banner lo recuerda).
- Si un backtest rechaza una estrategia por defecto que opera esa gráfica, la mesa avisa (`ESTRATEGIA POR DEFECTO
  RECHAZADA`) y `desactiva SOL 1h` la apaga en esa gráfica (persistente, en `desk.json`).
- `desactiva HYPE 1h` (o `desactiva HYPE`) la quita; las posiciones abiertas siguen con su SL/TP. `playbook`
  lista lo activo. Si un backtest nuevo de una gráfica activa ya no la aprueba, la mesa avisa
  (`PLAYBOOK DEGRADADO`) pero no la desactiva sola: decides tú.

### Alarmas

| Ejemplo | Dispara cuando |
|---|---|
| `alerta HYPE > 90` · `alerta BTC < 80000` | El precio **cruza** el nivel (si ya está al otro lado al crearla, espera a que vuelva y cruce) |
| `alerta SOL cruza 150` | Cruza el nivel en cualquier sentido |
| `alerta HYPE rsi < 30 1h` · `alerta BTC rsi > 70 15m` | El RSI de velas **cerradas** de esa temporalidad cruza el nivel |
| `alerta BTC cruce ema 20 50 1h alcista` · `alerta BTC golden cross 1d` | Cruce de medias |
| `alerta SOL zona long` | El precio llega a la zona de liquidez (a su centro, la entrada; no basta con el borde pegado al precio) |
| `alerta BTC volumen 3x 15m` | Volumen de la vela ≥ 3 veces la media |
| `alerta HYPE mueve 3% 1h` · `alerta HYPE sube 5% 4h` | Movimiento de una vela |
| `... siempre` · `... cada vez` | Repite (con pausa de `DESK_ALERT_COOLDOWN_SECONDS`); sin esto avisa una vez y se borra |

`alertas` las lista; `borra alerta 3` / `borra alertas`; `silencio 30m` (o `silencio` hasta `sonido`) calla los
avisos no críticos (quedan registrados en `state/alerts.jsonl`; los críticos siempre llegan). Las alarmas se
guardan en `state/alerts.json` y sobreviven a reinicios.

### Avisos de la mesa

Además de las alarmas, la mesa avisa sola (pantalla, `alerts.jsonl`, `outbox.log` y Telegram/webhook según el
nivel mínimo de cada canal):

| Nivel | Avisos |
|---|---|
| info | Mesa arrancada/detenida, ejecuciones y SL/TP colocados, cierres con su PnL, señales automáticas abiertas, backtest terminado |
| warn (aviso) | Consumo de la pérdida diaria ≥ `DESK_DAILY_LOSS_WARN_FRAC` (80 %) del límite, drawdown al 80 % del máximo, fallos repetidos (3, 10, 50... ciclos seguidos), datos viejos de una moneda con posición, cooldown por racha de pérdidas, backtest fallido, playbook degradado o estrategia por defecto rechazada, TP rechazado (y desactivado tras 5 rechazos seguidos: la posición sigue con su SL), posición sin bracket adoptada |
| critical | Halts (pérdida diaria, drawdown máximo) y cierres por riesgo (se avisa cuando cambia lo que pasa: cierres enviados, entradas que no se pueden cancelar), **posición sin SL** (`SL ... NO colocado/RECHAZADO`), cierre de seguridad por SL imposible, **liquidación**, ejecución fuera de una operación gestionada, cierre rechazado o que no se pudo enviar, otro proceso leyendo el bot de Telegram (409); nunca se silencian, suenan y llegan al webhook |

Un aviso repetido no se reenvía antes de `DESK_ALERT_COOLDOWN_SECONDS` (un SL que no se puede colocar avisa una vez
y luego cada 5 min, no en cada ciclo). Con la red lenta y la cola de envío llena se descartan primero los info,
luego los avisos; nunca un crítico por otro de menos nivel. Las ejecuciones que provoca un comando van en la
respuesta a ese comando. La línea de estado periódica sólo sale por pantalla y `outbox.log`.

### Telegram paso a paso

1. En Telegram, habla con **@BotFather**: `/newbot`, elige nombre y usuario. Te da un token tipo
   `123456789:AAF...`. Ponlo en `.env`: `TELEGRAM_BOT_TOKEN=123456789:AAF...`
2. Escribe cualquier cosa a tu bot (búscalo por su usuario y pulsa *Iniciar*).
3. Obtén tu **chat id**: abre en el navegador `https://api.telegram.org/bot<TOKEN>/getUpdates` y busca
   `"chat":{"id":123456789`. (Otra forma: arranca la mesa sólo con el token; ignora los mensajes de chats no
   autorizados y escribe en `state/desk.log` "mensaje de un chat no autorizado ignorado (chat 123456789, ...)").
   Para un grupo, añade el bot al grupo; su id es negativo (`-100...`).
4. `TELEGRAM_CHAT_ID=123456789` en `.env` (varios separados por comas) y reinicia la mesa. El banner dice
   `canales: consola (>= info), telegram (>= info) | Telegram escuchando`.
   **En un grupo** pon además `TELEGRAM_USER_IDS` con el id de usuario de cada persona que puede operar (el
   `"from":{"id":...}` de `getUpdates`): sin él, en el grupo nadie da órdenes (si no, cualquier miembro podría
   mandar un `kill` o un `long`). Con `TELEGRAM_USER_IDS` también se exige en chats privados.
5. Escríbele igual que en la consola: `estado`, `long HYPE 2u zona`, `backtest HYPE 1h`, `alerta BTC > 100000`,
   `/estado`, `/cierra_todo` (el `_` es un espacio). `/help` da la ayuda.

Seguridad: **sólo** los chats de `TELEGRAM_CHAT_ID` (y, si lo pones, sólo las personas de `TELEGRAM_USER_IDS`)
pueden dar órdenes; el resto se ignora sin responder y queda en el log. Nunca se aceptan mensajes en nombre de un
canal, de un admin anónimo ni de otros bots, ni un `/comando@OtroBot` dirigido a otro bot del grupo. El token nunca
se escribe en logs. Cada mensaje se ejecuta como mucho una vez: si Telegram no confirma que lo entregó, no se
ejecuta hasta que lo confirma (si la mesa se reinicia antes, lo vuelve a recibir y lo ejecuta una sola vez), y uno
de hace más de 5 minutos no se ejecuta (se avisa para reenviarlo). Cada respuesta y aviso lleva delante la mesa que
lo manda (`[PAPER]`, `[LIVE mainnet]`...). Si otro proceso lee el mismo bot (otra mesa, una simulación con
`DESK_TELEGRAM_IN_SIM=true`), Telegram responde 409 y la mesa avisa en crítico. Las órdenes
interpretadas por IA piden `si`/`no`. `DESK_ALERT_MIN_LEVEL_TELEGRAM=warn` deja en el móvil sólo avisos y
críticos (los resultados de backtest son "info").

### Webhook de Discord o Slack

- **Discord**: Ajustes del canal → Integraciones → Webhooks → Nuevo webhook → Copiar URL.
- **Slack**: crea una app con *Incoming Webhooks* y copia la URL del canal.

`DESK_WEBHOOK_URL=https://discord.com/api/webhooks/...` en `.env`. Recibe los avisos de nivel
`DESK_ALERT_MIN_LEVEL_WEBHOOK` (`warn` por defecto) o más; el webhook sólo avisa, no acepta comandos. La URL lleva
el secreto: no se escribe en logs (sólo su dominio).

### Panel y consola

`python -m desk dashboard` y `python -m desk console` (ver [Arranque rápido](#4-panel-en-vivo-y-consola-interactiva))
leen lo que escribe la mesa: `status.json` (se refresca cada 30 s y tras cada comando, señal o cierre, con
posiciones, órdenes, consumo de límites, playbook, alarmas y últimas señales), `alerts.jsonl`, `playbook.json` y
`reports/`.

---

## Cómo opera sola 24/7

Con `DESK_AUTO=true` (por defecto) la mesa evalúa las gráficas del **playbook** (ver arriba) y sus estrategias
por defecto sobre `DESK_SYMBOLS` **en el primer ciclo tras el cierre de cada vela** (y además cada
`DESK_STRATEGY_SECONDS`), **una vez por vela cerrada** (nunca con la vela en curso). Como en el backtest, que entra
en el open de la vela siguiente: una señal que llega con la moneda ocupada (o con la mesa en pausa, halt, cooldown,
auto OFF o en `DESK_MAX_OPEN_POSITIONS`) se pierde, y una vela que cerró hace más de ~2 minutos (reinicio, fallos de
datos) no se opera tarde:

- **`liquidity_sweep` (15 m)** — reversión tras un barrido de liquidez: la vela anterior perfora la zona de
  liquidez (bids/liquidaciones de longs bajo el precio) y la última cierra de vuelta por encima, con RSI(14)
  < 40 y girando al alza → LONG a mercado, SL bajo la zona − 0,5·ATR, TP a `DESK_DEFAULT_TP_R` R.
  Simétrico para SHORT (RSI > 60 y bajando).
- **`squeeze_breakout` (1 h)** — fin de una compresión de volatilidad (Bollinger dentro de Keltner) con
  ADX(14) > 25 y cierre fuera de Bollinger → entrada en la dirección de la ruptura, SL a 1,5·ATR.

Disponibles pero no activas por defecto (añádelas a `DESK_STRATEGIES` si las quieres):

- **`ema_trend` (1 h)** — cruce de EMA rápida/lenta (20/50) con ADX > 20; SL a 2·ATR, TP a R.
- **`rsi_reversion` (15 m)** — RSI extremo (< 30 / > 70) con toque de la Bollinger; objetivo la media, SL a 1,5·ATR.
- **`donchian_breakout` (4 h)** — primer cierre fuera del canal de 20 velas; SL a 2·ATR, TP a R.

Cada estrategia define su lógica una sola vez (`Strategy.signals`, vectorizada): el backtest y la operativa
en vivo usan exactamente el mismo código, y la señal en vivo es la última fila de ese cálculo. Que una
estrategia exista aquí no significa que gane dinero: un backtest no garantiza nada y sólo cuenta el
resultado fuera de muestra.

### Detalles del walk-forward (para desarrolladores)

`desk/optimizer.py` prueba las estrategias sobre una gráfica (moneda + temporalidad) sin hacerse trampas:

- **Walk-forward anclado** (`DESK_BT_FOLDS` tramos): los parámetros de cada tramo se eligen sólo con su pasado
  (hasta `DESK_BT_MAX_COMBOS` combinaciones, muestreo determinista) y se miden en el bloque siguiente, que no
  vieron. Lo que se enseña y lo que decide es esa curva **fuera de muestra**; lo de dentro de muestra sale al
  lado sólo para ver cuánto se infla.
- **Aprobada** sólo si fuera de muestra: PF ≥ `DESK_BT_MIN_PROFIT_FACTOR`, retorno > 0, ≥ `DESK_BT_MIN_TRADES`
  operaciones, drawdown ≤ `DESK_BT_MAX_DRAWDOWN_PCT` y sin el halt de `DESK_MAX_DRAWDOWN_PCT` (la curva se simula con
  los frenos del riesgo en vivo), parámetros adoptados en positivo en su propio tramo **y ventaja estadísticamente
  significativa** (t del R medio por operación, corregida por todas las estrategias que se pueden probar en la
  gráfica). Ese filtro no es decorativo: sobre gráficas de ruido puro (paseos aleatorios), sólo con los cuatro
  primeros 17 de 24 tenían "ganadora". Con todos, 3 de 120 (2,5 %; por diseño, hasta un 5 %): un informe aprobado
  todavía puede ser suerte, y repetir backtests de la misma gráfica hasta que uno apruebe sube esa cifra (por eso
  `activa` pide `forzar` si un informe reciente la rechazó). Lo normal es que muchas gráficas no tengan ganadora: es
  la respuesta honesta.
- Una estrategia que mira el futuro se detecta (sus señales pasadas cambian al añadir velas) y se invalida.
- Informes en `state/reports/` (`ReportStore`), con resumen en texto, HTML autocontenido (precio con entradas y
  salidas, capital, drawdown, claro/oscuro) y PNG (`desk/report_render.py`). Las gráficas adoptadas quedan en
  `state/playbook.json` (`desk/playbook.py`: una estrategia con sus parámetros por moneda + temporalidad; sólo se
  adopta lo aprobado fuera de muestra).

```python
from desk.optimizer import optimize, ReportStore
report = optimize("HYPE", "1h", velas_cerradas, settings, meta=meta)   # ~7.5 s: 5 estrategias x 60 x 4 tramos, 3000 velas
print(report.summary_text())
ReportStore(settings.state_dir).save(report)
```

Reglas de la operativa automática:

- Cada señal pasa por el **mismo gestor de riesgo** que los comandos, con `DESK_AUTO_UNITS` unidades (o las de
  su entrada del playbook). Con varias gráficas en la misma moneda, el playbook va primero.
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
| Calidad de mercado | spread ≤ 25 bps, datos ≤ 120 s | Si no, no entra. La edad es la del libro según el exchange (un nodo con un libro congelado no pasa por dato fresco) |
| Apalancamiento | `DESK_LEVERAGE` (5x), aislado | Se baja si la liquidación quedaría antes que el SL |

Los límites diario y de drawdown se comprueban también al evaluar cada entrada con la cuenta del momento (un
comando que llega entre ciclos no abre si la cuenta ya está fuera de límites). Los **depósitos, retiradas y
transferencias** perp↔spot no cuentan como PnL: la mesa detecta el cambio de equity que no explican las
ejecuciones ni el PnL no realizado, lo avisa y lo descuenta del máximo de equity y de la referencia del día
(retirar un 10 % no dispara el halt por drawdown ni cierra las posiciones).

Dimensionamiento: `notional = unidades × DESK_UNIT_MARGIN_USD × apalancamiento`, recortado por las reglas
anteriores y por el margen disponible. Ejemplo: `long HYPE` con 500 USD/unidad y HYPE limitado a 2x → 1.000 USD
de notional, SL al 1,5 % → riesgo ≈ 16 USD (0,16 %).

## Ejecución

- **Entrada**: zona/límite → orden límite Gtc (caduca a los `DESK_ENTRY_TTL_MINUTES`); mercado → IOC con
  precio máximo (`DESK_MARKET_SLIPPAGE_BPS`). La operación (con su SL/TP aprobados) se registra *antes* de
  enviar: si el envío da timeout pero la orden llegó al exchange, la mesa responde "ESTADO INCIERTO: no la
  repitas" y la reconciliación la encuentra y la protege con esos niveles (o la descarta si no aparece).
- **Cancelar / caducar**: una entrada sólo se olvida cuando su cancelación se confirma; si falla y la orden se
  ejecuta, la posición conserva su SL/TP.
- **Cierres** (`cierra`, `cierra todo`, kill, límites de riesgo): IOC con un 5 % de tolerancia (creciente en
  cada reintento), no los 10 bps de las entradas. Si el cierre se ejecuta a medias o se rechaza, se reintenta
  en la reconciliación (hasta 6 intentos; después avisa).
- **Brackets**: al ejecutarse la entrada se colocan STOP y TAKE-PROFIT *reduce-only* por el tamaño abierto
  (ajustados si la entrada se llena por partes: primero la orden nueva, luego se cancela la vieja).
- **OCO**: si salta el SL se cancela el TP y viceversa; el PnL neto (con comisiones) se registra en
  `trades.csv` y alimenta el control de rachas.
- **Reconciliación cada ciclo**: re-coloca SL/TP que falten, cancela órdenes reduce-only huérfanas, detecta
  posiciones cerradas fuera de la mesa y **adopta** posiciones sin bracket (p. ej. tras perder el estado)
  poniéndoles el SL por defecto (nunca más allá de la liquidación). Si el exchange rechaza el SL 3 veces seguidas
  (o no se puede colocar por fallos de conexión 10 veces), la posición se cierra. Cada moneda se procesa por
  separado: un fallo en una no deja sin SL a las demás.

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
6. ☐ Sólo entonces mainnet: `DESK_HL_MAINNET=true` (la consola lo anuncia: `MODO LIVE ... MAINNET (DINERO REAL)`),
   **con otro `DESK_STATE_DIR`**: `profile.json` guarda la red y la cuenta del estado y la mesa se niega a arrancar
   en mainnet (u otra cuenta) sobre el estado de testnet (su máximo de equity dispararía el halt por drawdown).
7. ☐ Vigila: `python -m desk status`, `state/outbox.log`, alertas si `python -m desk health` falla. Ten a mano `kill`.

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
se ejecuta como mucho una vez). Un `kill`, `pausa` o halt se guarda en el momento: sobrevive a un corte brusco
(SIGKILL, OOM). Tampoco se repite la señal automática de una vela ya evaluada.

**Salud**: cada ciclo escribe `health.json` (racha de fallos, etapas que fallan, última lectura buena de la
cuenta y los fills) y `heartbeat` sólo cuando el ciclo leyó la cuenta y los fills. `python -m desk health`
devuelve 1 si la mesa no completa ciclos en 10 min o no ve el exchange desde hace 10 min (es el HEALTHCHECK del
Dockerfile; úsalo en cron/alertas con systemd). `python -m desk status` muestra los mismos avisos.

---

## Archivos y logs

```
trading-bot/
├── desk/
│   ├── __main__.py        CLI: run / send / status / health / dashboard / console / backtest
│   ├── desk.py            TradingDesk: bucle 24/7, comandos, operativa automática (playbook), avisos, estado
│   ├── jobs.py            Backtests en segundo plano (un hilo, cola, cancelables al apagar)
│   ├── engine.py          ExecutionEngine: entradas, brackets SL/TP, OCO, reconciliación
│   ├── risk.py            RiskManager: tamaño, halts diario/drawdown, cooldown, pausa
│   ├── strategy.py        Estrategias (liquidity_sweep, squeeze_breakout, ema_trend, rsi_reversion, donchian_breakout)
│   ├── backtest.py        Backtest por gráfica: ejecución conservadora, métricas
│   ├── optimizer.py       Walk-forward: parámetros sin mirar el futuro, aprobación fuera de muestra, informes
│   ├── playbook.py        Gráficas adoptadas en automático (estrategia + parámetros por moneda y temporalidad)
│   ├── report_render.py   Informes HTML (SVG en línea) y PNG, mini-gráficas de texto
│   ├── alerts.py          Alarmas del usuario y avisos de la mesa (anti-spam, silencio)
│   ├── notify.py          Canales de aviso: consola, fichero, Telegram, webhook (Discord/Slack)
│   ├── telegram_bot.py    Bot de Telegram: comandos desde el móvil
│   ├── tui.py console.py remote.py   Panel en vivo, consola interactiva y cliente inbox/outbox
│   ├── liquidity.py       Zonas de liquidez (libro + liquidaciones; históricas por pivotes para backtest)
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
| `status.json` | Último estado (lo muestra `python -m desk status`) y sus campos para el panel (posiciones, órdenes, consumo de límites, playbook, alarmas, señales); sin cuenta, una línea "SIN DATOS DE LA CUENTA" |
| `health.json` | Salud de cada ciclo: racha de fallos, etapas que fallan, última lectura buena de cuenta y fills |
| `heartbeat` | Hora del último ciclo que leyó la cuenta y los fills (healthcheck) |
| `paper_broker.json`, `engine.json`, `risk.json`, `desk.json`, `profile.json` | Estado para sobrevivir a reinicios |
| `reports/<COIN>_<tf>_<id>.json` | Informes de backtest walk-forward (dentro vs fuera de muestra) y su curva fuera de muestra |
| `reports/<COIN>_<tf>_<id>.html` / `.png` | El mismo informe para abrir en el navegador / enviar por Telegram |
| `alerts.json` | Alarmas activas y estado del silencio |
| `alerts.jsonl` | Cada alarma o aviso enviado (o silenciado), con nivel y moneda (lo leen el panel y la consola) |
| `playbook.json` | Gráficas adoptadas en automático: estrategia, parámetros, unidades y métricas fuera de muestra |

Para empezar de cero en paper: detén la mesa y borra `state/`.

---

## Tests

```bash
python -m pytest -q
```

Todos corren sin red (mercados falsos/simulados y SDKs simulados), incluidos los flujos completos: comando →
orden en zona → ejecución → SL/TP → TP con OCO → PnL; stop-loss; cierra todo; kill; halts diario y de
drawdown con cierre; reinicio con estado persistido; inbox con offset; confirmación de comandos de IA;
operativa automática con pausa/auto off; fallos de datos que no tumban el bucle; y regresiones de casos
límite (SL más allá de la liquidación, cierres que no casan, timeouts con la orden ya en el exchange,
fills re-entregados, inbox reescrito, retiradas, salud, perfil testnet/mainnet). La fase 2 se prueba de punta a
punta en `tests/test_desk_phase2.py`: backtest por comando → informe → `activa` → operación automática de la
estrategia adoptada; alarmas de precio y RSI; aviso al 80 % de la pérdida diaria; halts críticos; Telegram con
una sesión HTTP falsa; `status.json` ampliado; y el bucle sigue vivo aunque un backtest falle o tarde.

# Version corta


# Datos

- **Limit Order Book (LOB)** : Estructura de datos clásica y estandarizada en temas de "Microestructura de mercados financieros", contiene Ordenes de compra y de venta organizadas en un tensor de 2x25x2.

- **Public Trades (PT)** : El registro de operaciones (compras efectuadas, ventas efectuadas) entre 2 participantes de un mercado (el registro cuando ya se concretó una operación, y es irreversible por definición). 

- **Exchange** : Empresa y espacio electrónico (Servidor) donde se ejecutan las operaciones de compra-venta. Cada Exchange tiene un solo "Matching Engine" estríctamente local y por lo tanto está georeferenciado y expuesto a efectos de latencia de datos. Binance está en Asia, Coinbase está en Americas, Kraken está en Europa. Como referencia, aunque no considerado en este proyecto, Bitso es análogo a estos 3 y está localizado en México. 

- **Recurrence Plot** : La representación en 2D, de acuerdo a la disciplina de Sistemas Dinámicos, de un conjunto de observaciones en el tiempo, en este caso se utilizan varios LOBs, separados en grupos de acuerdo a un criterio de "time-aware sampling" que utiliza PTs. De tal manera que 1 RP es una sola imagen en 2d que funciona como representación de la dinámica del mercado (LOBs y PTs) durante un period de tiempo definido.

## Stats de Datasets

### Orderbooks

exchange    market       files      total rows   avg rows/file           elapsed (s)
----------------------------------------------------------------------------------
binance     BTC-USDC       142        23326867       164273.71  665918.2 (7d 16:58:38)
coinbase    BTC-USDC       142        34176878       240682.24  668811.8 (7d 17:46:52)
kraken      BTC-USDC       142         4977216        35050.82  665916.8 (7d 16:58:37)

Per-market roll-up (avg elapsed = mean across the 3 exchanges):
market       files      total rows   avg rows/file       avg elapsed (s)
----------------------------------------------------------------------
BTC-USDC       426        62480961       146668.92  666882.2 (7d 17:14:42)

### Trades

exchange    market       files      total rows   avg rows/file           elapsed (s)
----------------------------------------------------------------------------------
binance     BTC-USDC       142         2091622        14729.73  665913.3 (7d 16:58:33)
coinbase    BTC-USDC       142         1352539         9524.92  668812.1 (7d 17:46:52)
kraken      BTC-USDC       142           20245          142.57  665906.4 (7d 16:58:26)

Per-market roll-up (avg elapsed = mean across the 3 exchanges):
market       files      total rows   avg rows/file       avg elapsed (s)
----------------------------------------------------------------------
BTC-USDC       426         3464406         8132.41  666877.3 (7d 17:14:37)

# Pre-procesamiento de datos

De la carpeta de `datasets/raw` se utilizan LOBs y PTs para generar RPs, que se almacenan
en la carpeta de `datasets/rp`. Esto lo realiza el código llamado [takens-rs](https://github.com/convergentist/takens-rs).

# Tecnicas de muestreo

1. Ventanas fijas de 60 segundos (No importando cuantos OBs hay).
2. Ventanas fijas de 64 Orderbooks (no importando cuando tiempo abarca).
3. Ventanas de eventos (Cada que se active el disparador de actividad de mercado).


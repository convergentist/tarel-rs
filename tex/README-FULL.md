# Article

## Intro

### Datasets

En la carpeta `/datasets` existen 3 sub-carpets, cada una representa un diferente `exchange` (o equivalente
a bolsa de valores). Los utilizados en este proyecto fueron: Binance (Localizada en Asia), Coinbase (Localizada en Americas), y Kraken (Lozalida en Europa), cada una elegida para representar, georeferenciadamente, las diferentes
regiones de mercados financieros del mundo. 

Las estructuras de datos utilizadas fueron: "Limit Order Books (LOB)", una estructura que es 
prevalente en practicamente todos los mercados financieros organizados (Tanto tradicionales como de
cryptomonedas), y sobre todo en aquellos mercados que operan bajo una lógica llamada 
"Order-Driven Market". Desde la perspectiva de datos, un LOB es un tensor de dimensiones SxLxPV, donde S=2 
ya que representa el sentido o intencion de los participantes a traves de las "Orders" que colocan para 
ya sea comprar (en el lado de los Bids), o, vender (en el lado de los Asks). L = la cantidad de niveles que 
existen en el LOB, puede ser desde 1 hasta, tipiciamente, 100. Y, en cada nivel L, se tienen dos campos, Price (P) y Volume (V). Las dimensiones de todos los datasets utilizados en este proyecto fueron: 

- Bids / Asks -> S = 2
- L -> Profundidad simetrica = 25
- PV -> Precio y Volume para cada nivel.

La dimension de todos y cada uno de los LOB recolectados es de 2x25x2. Y la frecuencia de muestreo fue elegida de 250 millisegundos. 



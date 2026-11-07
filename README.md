# Ayano

Motor de ajedrez UCI escrito desde cero en C++20, con las técnicas de los motores modernos:
bitboards, búsqueda alfa-beta con podas agresivas y evaluación con una red neuronal NNUE propia,
entrenada con datos públicos de Leela Chess Zero.

Como Stockfish, es un programa de consola sin tablero: se conecta a una interfaz gráfica
mediante el protocolo UCI.

**Lo escribió una IA:** Claude, de Anthropic, programó el motor, entrenó sus redes y midió su
fuerza. Los detalles están en [Cómo se hizo](#cómo-se-hizo).

**Es una versión preliminar:** toda su fuerza se ha medido en un solo equipo. Será la 1.0 cuando
tenga una medida externa y se haya probado en otros equipos. La versión 0.8.0 se publicó primero
como 1.0, y sus archivos conservan ese nombre.

*English summary: [see below](#english-summary).*

## Resultados

Medido en un AMD Ryzen 5 5600 (6 núcleos, 12 hilos). Las cifras de Elo llevan su margen de
error del 95 %.

### Fuerza absoluta

La regla es la lista [CCRL Blitz](https://computerchess.org.uk/ccrl/404/) del 31 de octubre de 2026.
Ayano 0.10.0 ronda los **3659, entre 3627 y 3689**, y la 0.9.1, los **3502, entre 3473 y 3535**: la
mejora es de unos **+157 Elo**.

Se jugaron 100 partidas contra cada rival a 120 s + 1 s, con un hilo y 64 MB de hash por motor:

| Versión | Rival y su rating | Resultado | Diferencia | Rating que implica |
|---|---|---|---|---|
| 0.10.0 | Clover 9.1, 3726 | +13 =56 −31 | −63 ± 45 | 3663 |
| 0.10.0 | Motor 0.9.0, 3694 | +14 =61 −25 | −38 ± 43 | 3656 |
| 0.9.1 | Carp 3.0.1, 3525 | +22 =45 −33 | −38 ± 51 | 3487 |
| 0.9.1 | Frozenight 6.0.0, 3361 | +52 =42 −6 | +173 ± 53 | 3534 |
| 0.9.1 | Mantissa 3.7.2, 3311 | +57 =33 −10 | +177 ± 60 | 3488 |

Son medidas propias, no un rating de CCRL: esa lista usa otras máquinas y otro grupo de rivales. La
0.10.0 solo jugó contra dos, ambos más fuertes que ella, y casi todas las partidas de la 0.9.1 se
jugaron con el equipo ocupado en entrenar. Stockfish 19 con 8 hilos está en 3803 en esa lista.

Las cifras que se publicaron antes, como los 3275 de la 0.9.0, usan la escala `UCI_Elo` de Stockfish
limitado y no se pueden comparar con estas: la misma red y la misma búsqueda dan 3502 aquí. El
detalle está en [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#fuerza-en-la-escala-ccrl-blitz).

### Fuerza relativa

Estas cifras comparan versiones del propio motor, así que no tienen por qué repetirse contra
otros rivales.

| Partidas con 5 s por bando y 0,05 s de incremento | Partidas | Resultado |
|---|---|---|
| Ayano 0.10.0 contra 0.9.1 | 200 | **+189 ± 42 Elo** |
| Ayano 0.9.1 contra 0.9.0 | 200 | −12 ± 28 Elo, sin diferencia apreciable |
| Ayano 0.9.0 contra 0.8.0 | 1000 | **+113 ± 17 Elo** |
| Red gen3 contra la evaluación clásica, con sus tablas anteriores | 160 | **+364 ± 71 Elo** |
| Red gen3 contra la red gen2 | 1000 | **+39 ± 16 Elo** |
| 4 hilos contra 1 hilo | 100 | **+151 ± 53 Elo** |

Las tres últimas filas son anteriores a gen4.

### Otras medidas

| Medida | Resultado |
|---|---|
| Velocidad con la red | 2,0 millones de nodos por segundo y hilo, un 16 % menos que la 0.9.1 |
| Profundidad completada desde la posición inicial | 26 en 5 segundos |
| Bratko-Kopec, 2 s y 2 hilos | 20 de 24, en dos pasadas |
| Perft | Exacto en toda la suite, unos 800 millones de nodos |

La red nueva es mayor y por eso va más lenta. Las cifras de fuerza ya incluyen ese coste.

### Prueba en Cute Chess

La 0.9.1 nace de probar la 0.9.0 en Cute Chess 1.5.1, el primer gestor externo que usó. No jugó
ninguna jugada ilegal ni se colgó, pero aparecieron fallos que las pruebas anteriores no veían:

| Prueba | 0.9.0 | 0.9.1 | 0.10.0 |
|---|---|---|---|
| Reloj agotado a 5 s por bando, sin incremento | 10 de 48 partidas | 0 de 48 | 0 de 48 |
| Reloj agotado a 2 s por bando, sin incremento | 9 de 32 | 0 de 32 | 0 de 32 |
| Avisos de jugadas ilegales en las líneas de análisis | 202 en 338 partidas | 0 en 726 | 0 en 370 |

Con las opciones por defecto, la 0.9.1 y la 0.10.0 tampoco agotaron el reloj con incremento ni con
control de jugadas. La 0.9.1 sí lo agotó 1 vez en 48 con `MultiPV` 3 a 3 s + 0,03. La 0.10.0 no lo
agotó en 38, pero ese caso extremo no se ha corregido. El detalle está en
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#prueba-en-cute-chess).

## Descargar

Los ejecutables para Windows y Linux están en la
[última versión](https://github.com/CloseNoJona/ayano/releases/latest). Llevan la red integrada y
no necesitan nada más. Elige según tu CPU:

| Archivo | Para qué CPU |
|---|---|
| `bmi2` | La mayoría: Intel desde 2013 y AMD Ryzen 5000 o posterior |
| `avx2` | AMD Ryzen 1000 a 3000, donde PEXT es lento |
| `portable` | Cualquier CPU x86-64, sin AVX2. En Windows analiza unas 7 veces menos nodos por segundo |

Si la CPU no admite la variante elegida, el motor avisa al arrancar, o se cierra sin mensaje si
tampoco tiene AVX. En ambos casos, usa `portable`. Los ejecutables de Windows no están firmados, así
que SmartScreen puede avisar al abrirlos.

## Compilar

Necesitas Visual Studio Build Tools con las herramientas de C++.

```bash
scripts\build.cmd
```

El ejecutable queda en `build\release\ayano.exe`. Si tienes una carpeta `build` de una versión
anterior a la 0.10.0, bórrala antes: conserva el tamaño de red antiguo y compilaría sin red integrada.

| Variante | Comando | Para qué |
|---|---|---|
| Normal | `scripts\build.cmd` | CPUs con AVX2 y BMI2 |
| AVX2 sin PEXT | `scripts\build.cmd avx2` | AMD Ryzen 1000 a 3000, donde PEXT es lento |
| Portable | `scripts\build.cmd portable` | CPUs sin AVX2 |
| Más rápida | `scripts\build_pgo.cmd` | Cerca de un 9 % más veloz gracias a un perfil real, en `build\pgo` |

En Linux bastan g++ y make. El ejecutable queda en `ayano`:

```bash
make -j4
```

Acepta `ARCH=avx2` y `ARCH=portable`, las mismas variantes de la tabla, y `STATIC=1` para un
binario que funcione en otras distribuciones. `make test` compila y pasa los tests. Se ha probado
con g++ 15 en WSL. Las herramientas solo se compilan con CMake, y `match` solo funciona en Windows.

## Jugar contra él

1. Instala una interfaz gratuita, como [Cute Chess](https://cutechess.com) o
   [Arena](http://www.playwitharena.de).
2. Añade `ayano.exe` como motor UCI.
3. Elige su fuerza con la opción **Skill Level**: 20 es el máximo y 0 el más flojo.

Se ha probado en Cute Chess 1.5.1, pero aún no en Arena ni en otras interfaces. No admite Chess960.

## Opciones UCI

| Opción | Por defecto | Descripción |
|---|---|---|
| `Hash` | 64 | Memoria de la tabla de transposición, en MB |
| `Threads` | 1 | Hilos de búsqueda |
| `MultiPV` | 1 | Número de líneas que analiza |
| `Skill Level` | 20 | Fuerza de juego, de 0 a 20 |
| `Move Overhead` | 10 | Margen por jugada en ms, para compensar retrasos de la interfaz; sin incremento, también fija cuánto reloj se guarda para el final |
| `Use NNUE` | true | Usa la red neuronal; si no, la evaluación clásica |
| `EvalFile` | | Carga una red desde un archivo y vacía la tabla hash |
| `Clear Hash` | | Vacía la tabla de transposición |

Además de los comandos UCI, entiende estos:

| Comando | Qué hace |
|---|---|
| `d` | Muestra el tablero |
| `eval` | Evaluación estática de la posición |
| `perft <n>` | Cuenta los nodos hasta la profundidad `n` |
| `bench [n]` | Prueba de velocidad con posiciones fijas |
| `wait` | Espera a que termine `go`, útil en scripts |

## Tests

```bash
build\release\ayano_tests.exe
```

```bash
build\release\ayano_tests.exe --slow
```

La opción `--slow` añade el perft completo, de unos 800 millones de nodos.

La compilación con AddressSanitizer (`scripts\build.cmd asan`) necesita el entorno de Visual
Studio para encontrar su biblioteca:

```bash
scripts\vs.cmd build\asan\ayano_tests.exe
```

## Herramientas

Están en `build\release`.

**Enfrentar dos versiones** y decidir con SPRT si una es mejor:

```bash
match --engine1 nuevo.exe --engine2 viejo.exe --games 2000 --concurrency 5 --tc 8+0.08 --sprt 0 5
```

Si un motor rechaza una opción, se cae, se cuelga o juega una jugada ilegal, `match` se detiene
con un código de error, porque el resultado ya no sería válido. Una derrota por tiempo se avisa
con todo su contexto y el match continúa. Al final indica el margen mínimo de reloj: lo más cerca
que se ha estado de perder por tiempo.

**Medir la fuerza contra Stockfish limitado.** Con un límite de nodos, la carga de la máquina no
altera el resultado:

```bash
match --engine1 ayano.exe --engine2 stockfish.exe --opt2 "UCI_LimitStrength=true;UCI_Elo=3000;Hash=64" --games 100 --concurrency 4 --nodes 1000000
```

Para medir con reloj, cambia `--nodes 1000000` por `--tc 120+1`, deja la máquina libre y juega
menos partidas a la vez que núcleos tenga.

**Medir la resolución de posiciones tácticas:**

```bash
solve --epd tests\data\bratko-kopec.epd --ms 5000 --threads 4
```

**Comprobar una red sin jugar partidas:**

```bash
netcheck --data data\gen0.bin --net nets\default.nnue
```

**Ajustar las tablas de la evaluación clásica** con partidas propias, desde cero:

```bash
tune_pst --data data\gen2.bin --data data\gen3.bin --output src\eval\pst.h
```

**Ajustar los parámetros de la búsqueda** con SPSA. Necesita la compilación `scripts\build.cmd tune`,
que los expone como opciones UCI:

```bash
python scripts\spsa.py --engine build\tune\ayano.exe --match build\tune\match.exe --state spsa.json --iterations 2500
```

Se puede interrumpir: con el mismo `--state` continúa donde iba. Al terminar imprime los valores para
`src\search\params.h`. SPSA mueve los parámetros aunque no haya nada que ganar, así que solo valen si
ganan después a los anteriores en un match aparte.

## Entrenar una red nueva

La red integrada se entrenó en GPU con [bullet](https://github.com/jw1912/bullet) y datos de
Leela Chess Zero. La receta completa está en [training/README.md](training/README.md).

Sin GPU sigue disponible el ciclo con el que se hicieron las redes hasta la 0.9.1: juega partidas
contra sí mismo, entrena en CPU y compara la red nueva con la actual. Es un solo comando:

```bash
powershell -File scripts\new_generation.ps1 -Name gen5 -Minutes 60
```

Cada generación necesita un nombre nuevo. Con la arquitectura actual, `datagen`, `train` y `match`
funcionan por separado, pero el ciclo no se ha usado para entrenar una red completa: en unas horas de
CPU no reuniría ni una fracción de los datos de la red integrada.

Si una red gana, cópiala a `nets\default.nnue` y recompila: el ejecutable la integrará. Mide también
cada red contra rivales externos, porque el duelo con la anterior puede exagerar la mejora.

La red forma parte del ejecutable y su tamaño de capa oculta es fijo al compilar. Para probar
otro, configura con `-DAYANO_HIDDEN=N`: esa compilación no integra ninguna red, así que hay
que cargarla con `EvalFile`.

## Cómo está hecho

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explica el diseño, la búsqueda, la red y cómo se
verifica cada parte.

## Cómo se hizo

Claude, de Anthropic, diseñó y programó el motor con Claude Code, entrenó las redes, midió la
fuerza y redactó esta documentación. Jonathan Garcia propuso el proyecto, aportó el equipo y
decidió qué medir y qué publicar.

Las técnicas están documentadas en la
[Chess Programming Wiki](https://www.chessprogramming.org) y en la comunidad de motores abiertos.
El código del motor se escribió sin abrir el de otros motores. Para la red de la 0.10.0 sí se leyó
documentación ajena: los README de Coda y de otros motores, `NET.md` de Tcheran y la documentación,
los ejemplos y parte del código del entrenador bullet. Una IA puede reproducir de memoria lo que vio
al entrenarse, así que el código se comparó con [scripts/similarity.py](scripts/similarity.py) con el
de 15 motores: Stockfish 19, Alexandria, Berserk, Caissa, Clover, Ethereal, Halogen, Koivisto,
Obsidian, PlentyChess, Reckless, Seer, Stormphrax, Viridithas y Weiss.

Coincide un 0,7 % de los tokens, en fragmentos de pocas líneas: definiciones básicas, como la lista
de casillas, y fórmulas habituales, como el ajuste de la ventana de aspiración. Entre las series de
números coinciden las que vienen dadas por su función, como los saltos del caballo y del rey, los
valores publicados de perft y las constantes de SplitMix64, un generador de dominio público. También
coincide una tabla de diseño: las 8 cubetas de rey de la red son idénticas a las de Halogen. Se
escribieron reduciendo las del ejemplo de bullet, sin abrir Halogen. La herramienta busca secuencias
iguales, así que no detectaría una lógica copiada y reescrita. Con Reckless y Viridithas, que están
en Rust, solo tiene sentido comparar los números.

Hasta la 0.9.1, las redes aprendieron solo de sus propias partidas. La de la 0.10.0 se entrenó con
partidas de [Leela Chess Zero](https://lczero.org), convertidas y filtradas por otras personas con
herramientas de Stockfish. El entrenador es [bullet](https://github.com/jw1912/bullet), de Jamie
Whiting. Ni el código de Stockfish ni sus partidas forman parte de Ayano. Mantissa, Frozenight, Carp
y Motor solo se usaron como rivales. Stockfish y Clover fueron también rivales y entraron en la
comparación de código, y Stockfish analizó las partidas del motor en Lichess.

Las primeras partidas de entrenamiento se jugaron con las tablas pieza-casilla de PeSTO, de Ronald
Friederich, publicadas en esa wiki. Esas tablas ya no están en el código: `tune_pst` ajustó las
actuales desde cero con partidas propias.

El nombre viene de Kiyotaka Ayanokoji, de *Classroom of the Elite*: piensa muchas jugadas por delante
y se formó aislado en la White Room, igual que las primeras redes, que aprendieron solo de sus
propias partidas. Durante su desarrollo, el motor se llamó Enroque.

## Licencia

GPL-3.0 o posterior. El texto está en [LICENSE](LICENSE).

La red integrada se entrenó con datos que exigen este aviso:

Contains information from
[Filtered LC0 T91 (min-v2.v6)](https://huggingface.co/datasets/adamtwiss/t91-binpacks-filtered),
which is made available here under the
[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).

## English summary

Ayano is a UCI chess engine written from scratch in C++20. It uses bitboards, alpha-beta search
with aggressive pruning, Lazy SMP and an NNUE network (768 inputs in 8 king buckets, 1024 hidden
neurons per side, 8 output buckets), trained on public Leela Chess Zero data.

**It was written by an AI.** Claude, by Anthropic, designed and programmed the engine with Claude
Code, trained the networks, measured its strength and wrote the documentation. Jonathan Garcia
proposed the project, provided the hardware and decided what to measure and publish.

**This is a preliminary version.** All its strength figures come from a single machine. It will
become 1.0 once it has an external measurement and has been tested on other machines. Version 0.8.0
was first published as 1.0, and its files keep that name.

No code from other engines was opened while writing the engine. For the 0.10.0 network, third-party
documentation was read: the READMEs of Coda and other engines, Tcheran's `NET.md`, and the
documentation, examples and part of the code of the bullet trainer. An AI can still reproduce from
memory what it saw in training, so the code was compared with
[scripts/similarity.py](scripts/similarity.py) against 15 engines, including Stockfish 19, Ethereal,
Berserk, Koivisto and Stormphrax. About 0.7 % of the tokens match, in
fragments of a few lines: basic definitions such as the list of squares, and common formulas such as
the aspiration window update. One design table also matches: the 8 king buckets of the network are
identical to Halogen's. They were written by reducing those of the bullet example, without opening
Halogen. The tool looks for identical sequences, so it would not detect copied logic that was
rewritten.

Up to version 0.9.1, the networks learned only from the engine's own games. The 0.10.0 network was
trained with [bullet](https://github.com/jw1912/bullet), by Jamie Whiting, on
[Leela Chess Zero](https://lczero.org) games, converted and filtered by other people with Stockfish
tools. No Stockfish code or games are part of Ayano. Mantissa, Frozenight, Carp and Motor were used
only as opponents. Stockfish and Clover were also opponents and were part of the code comparison,
and Stockfish analyzed the engine's Lichess games. The recipe is in
[training/README.md](training/README.md).

The name comes from Kiyotaka Ayanokoji, from *Classroom of the Elite*: he thinks many moves ahead
and was raised in isolation in the White Room, just like the first networks, which learned only from
their own games. During development the engine was called Enroque.

Strength was measured on an AMD Ryzen 5 5600 against engines on the
[CCRL Blitz](https://computerchess.org.uk/ccrl/404/) list of 31 October 2026, 100 games per opponent
at 120 s + 1 s, with one thread and 64 MB of hash per engine. Version 0.10.0 scored about 3659,
between 3627 and 3689, against Motor 0.9.0 and Clover 9.1. Version 0.9.1 scored about 3502, between
3473 and 3535, against Mantissa 3.7.2, Frozenight 6.0.0 and Carp 3.0.1. In a direct match at 5 s +
0.05, 0.10.0 beat 0.9.1 by +189 ± 42 Elo over 200 games. These are the project's own measurements,
not CCRL ratings. Earlier figures such as 3275 for 0.9.0 are on the `UCI_Elo` scale of limited
Stockfish and cannot be compared with these.

Version 0.9.1 fixed the problems found in Cute Chess 1.5.1: 0.9.0 ran out of time in 10 of 48 games
at 5 s per side without increment, and sent illegal moves in some analysis lines. It has been tested
in Cute Chess, but not yet in Arena or other interfaces, and it does not support Chess960.

Windows and Linux binaries are on the
[latest release](https://github.com/CloseNoJona/ayano/releases/latest). Pick `bmi2` for most CPUs,
`avx2` for AMD Ryzen 1000 to 3000, and `portable` for CPUs without AVX2, which on Windows searches
about 7 times fewer nodes per second. To build it yourself, use `scripts\build.cmd` on Windows or
`make -j4` on Linux. The license is GPL-3.0 or later.

The built-in network requires this notice: Contains information from
[Filtered LC0 T91 (min-v2.v6)](https://huggingface.co/datasets/adamtwiss/t91-binpacks-filtered),
which is made available here under the
[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).

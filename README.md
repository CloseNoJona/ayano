# Ayano

Motor de ajedrez UCI escrito desde cero en C++20, con las técnicas de los motores modernos:
bitboards, búsqueda alfa-beta con podas agresivas y evaluación con una red neuronal NNUE
entrenada solo con sus propias partidas.

Como Stockfish, es un programa de consola sin tablero: se conecta a una interfaz gráfica
mediante el protocolo UCI.

**Lo escribió una IA:** Claude, de Anthropic, programó el motor, entrenó sus redes y midió su
fuerza. Los detalles están en [Cómo se hizo](#cómo-se-hizo).

**Es una versión preliminar:** toda su fuerza se ha medido en un solo equipo y aún no ha jugado fuera
de él. Será la 1.0 cuando se haya probado contra otros rivales y en otros equipos. La versión 0.8.0
se publicó primero como 1.0, y sus archivos conservan ese nombre.

*English summary: [see below](#english-summary).*

## Resultados

Medido en un AMD Ryzen 5 5600 (6 núcleos, 12 hilos). Las cifras de Elo llevan su margen de
error del 95 %.

### Fuerza absoluta

La regla es la escala `UCI_Elo` de Stockfish 19. Ayano 0.9.0, con la red gen4 y la búsqueda ajustada,
ronda los **3080 ± 35 Elo** con un millón de nodos por jugada. La versión anterior, 0.8.0, con la red
gen3, da 3015 ± 35 en las mismas condiciones: la mejora es de **+68 Elo**, de +21 a +115.

| Rival, con un millón de nodos por jugada | 0.9.0 | 0.8.0 |
|---|---|---|
| Stockfish 19 con `UCI_Elo` 3000 | +95 =56 −49 en 200 partidas: +81 ± 42 | +77 =57 −66 en 200 partidas: +19 ± 41 |
| Stockfish 19 con `UCI_Elo` 3190, el máximo | +16 =38 −46 en 100 partidas: −108 ± 55 | +6 =38 −56 en 100 partidas: −191 ± 56 |

Cada versión jugó 300 partidas con las mismas aperturas. El resultado no depende de la carga de la
máquina y los dos niveles coinciden: 3081 contra 3000 y 3082 contra 3190.

La versión 0.8.0 también jugó a 120 s + 1 s, el ritmo al que Stockfish calibró la escala, y quedó
**entre 3155 y 3320**, con solo 80 partidas: +31 =7 −2 contra el nivel 3000 y +10 =16 −14 contra el
3190. La 0.9.0 aún no se ha medido a ese ritmo.

Son referencias, no ratings de CCRL ni Elo humano. Los márgenes solo recogen el azar de las
partidas. Stockfish sin limitar es mucho más fuerte y no se ha medido. El detalle está en
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#fuerza-absoluta).

### Fuerza relativa

Estas cifras comparan versiones del propio motor, así que no tienen por qué repetirse contra
otros rivales.

| Partidas con 5 s por bando y 0,05 s de incremento | Partidas | Resultado |
|---|---|---|
| Ayano 0.9.0 contra 0.8.0 | 1000 | **+113 ± 17 Elo** |
| Red gen3 contra la evaluación clásica, con sus tablas anteriores | 160 | **+364 ± 71 Elo** |
| Red gen3 contra la red gen2 | 1000 | **+39 ± 16 Elo** |
| 4 hilos contra 1 hilo | 100 | **+151 ± 53 Elo** |

Las tres últimas filas son anteriores a gen4.

### Otras medidas

| Medida | Resultado |
|---|---|
| Velocidad con la red | 2,5 millones de nodos por segundo y hilo |
| Profundidad completada desde la posición inicial | 26 en 5 segundos |
| Bratko-Kopec, 2 s y 2 hilos | 19 y 20 de 24, en dos pasadas |
| Perft | Exacto en toda la suite, unos 800 millones de nodos |

## Descargar

Los ejecutables para Windows y Linux están en la
[última versión](https://github.com/CloseNoJona/ayano/releases/latest). Llevan la red integrada y
no necesitan nada más. Elige según tu CPU:

| Archivo | Para qué CPU |
|---|---|
| `bmi2` | La mayoría: Intel desde 2013 y AMD Ryzen 5000 o posterior |
| `avx2` | AMD Ryzen 1000 a 3000, donde PEXT es lento |
| `portable` | Cualquier CPU x86-64, sin AVX2 |

Si la CPU no admite la variante elegida, el motor avisa al arrancar, o se cierra sin mensaje si
tampoco tiene AVX. En ambos casos, usa `portable`. Los ejecutables de Windows no están firmados, así
que SmartScreen puede avisar al abrirlos.

## Compilar

Necesitas Visual Studio Build Tools con las herramientas de C++.

```bash
scripts\build.cmd
```

El ejecutable queda en `build\release\ayano.exe`.

| Variante | Comando | Para qué |
|---|---|---|
| Normal | `scripts\build.cmd` | CPUs con AVX2 y BMI2 |
| AVX2 sin PEXT | `scripts\build.cmd avx2` | AMD Ryzen 1000 a 3000, donde PEXT es lento |
| Portable | `scripts\build.cmd portable` | CPUs sin AVX2 |
| Más rápida | `scripts\build_pgo.cmd` | Entre un 3 y un 4 % más veloz gracias a un perfil real, en `build\pgo` |

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

## Opciones UCI

| Opción | Por defecto | Descripción |
|---|---|---|
| `Hash` | 64 | Memoria de la tabla de transposición, en MB |
| `Threads` | 1 | Hilos de búsqueda |
| `MultiPV` | 1 | Número de líneas que analiza |
| `Skill Level` | 20 | Fuerza de juego, de 0 a 20 |
| `Move Overhead` | 10 | Margen por jugada en ms, para compensar retrasos de la interfaz |
| `Use NNUE` | true | Usa la red neuronal; si no, la evaluación clásica |
| `EvalFile` | | Carga una red desde un archivo |
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
netcheck --data data\gen0.bin --net nets\gen2.nnue
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

El ciclo completo (jugar, entrenar y comparar) es un solo comando:

```bash
powershell -File scripts\new_generation.ps1 -Name gen5 -Minutes 60
```

Cada generación necesita un nombre nuevo. Si la red nueva gana, cópiala a `nets\default.nnue` y
recompila: el ejecutable la integrará.
Las generaciones se pueden encadenar, porque cada una produce datos mejores que la anterior.
Mide también cada red contra Stockfish limitado: el duelo con la anterior puede exagerar la mejora.

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
El código se escribió sin abrir el de otros motores. Aun así, una IA puede reproducir de memoria lo
que vio al entrenarse, así que se comparó con [scripts/similarity.py](scripts/similarity.py) con el
de 15 motores: Stockfish 19, Alexandria, Berserk, Caissa, Clover, Ethereal, Halogen, Koivisto,
Obsidian, PlentyChess, Reckless, Seer, Stormphrax, Viridithas y Weiss.

Coincide un 0,5 % de los tokens, en fragmentos de pocas líneas: definiciones básicas, como la lista
de casillas, y fórmulas habituales, como el ajuste de la ventana de aspiración. Entre las series de
números, solo coinciden las que vienen dadas por su función, como los saltos del caballo y del rey,
los valores publicados de perft y las constantes de SplitMix64, un generador de dominio público. La
herramienta busca secuencias iguales, así que no detectaría una lógica copiada y reescrita. Con
Reckless y Viridithas, que están en Rust, solo tiene sentido comparar los números.

Ninguna red ni dato de entrenamiento procede de otro motor. Stockfish solo se usó como rival para
medir la fuerza y para esa comparación: ni su código ni sus partidas forman parte de Ayano.

Las primeras partidas de entrenamiento se jugaron con las tablas pieza-casilla de PeSTO, de Ronald
Friederich, publicadas en esa wiki. Esas tablas ya no están en el código: `tune_pst` ajustó las
actuales desde cero con partidas propias.

El nombre viene de Kiyotaka Ayanokoji, de *Classroom of the Elite*: piensa muchas jugadas por delante
y se formó aislado en la White Room, igual que esta red aprendió solo de sus propias partidas.
Durante su desarrollo, el motor se llamó Enroque.

## Licencia

GPL-3.0 o posterior. El texto está en [LICENSE](LICENSE).

## English summary

Ayano is a UCI chess engine written from scratch in C++20. It uses bitboards, alpha-beta search
with aggressive pruning, Lazy SMP and a small NNUE network (768 inputs, 512 hidden neurons per
side, 8 output buckets) trained only on its own self-play games, with no data from other engines.

**It was written by an AI.** Claude, by Anthropic, designed and programmed the engine with Claude
Code, trained the networks, measured its strength and wrote the documentation. Jonathan Garcia
proposed the project, provided the hardware and decided what to measure and publish.

**This is a preliminary version.** All its strength figures come from a single machine, and it has
not yet played anywhere else. It will become 1.0 once it has been tested against other opponents and
on other machines. Version 0.8.0 was first published as 1.0, and its files keep that name.

No code from other engines was opened while writing it. An AI can still reproduce from memory what
it saw in training, so the code was compared with [scripts/similarity.py](scripts/similarity.py)
against 15 engines, including Stockfish 19, Ethereal, Berserk, Koivisto and Stormphrax. About 0.5 %
of the tokens match, in fragments of a few lines: basic definitions such as the list of squares, and
common formulas such as the aspiration window update. The tool looks for identical sequences, so it
would not detect copied logic that was rewritten.

The name comes from Kiyotaka Ayanokoji, from *Classroom of the Elite*: he thinks many moves ahead
and was raised in isolation in the White Room, just as this network learned only from its own
games. During development the engine was called Enroque.

Strength was measured against Stockfish 19 with `UCI_LimitStrength`, one thread and 64 MB of hash
per engine, on an AMD Ryzen 5 5600. Ayano 0.9.0 scored about 3080 ± 35 on the `UCI_Elo` scale at one
million nodes per move, up from about 3015 ± 35 for version 0.8.0. Version 0.8.0 also scored between
3155 and 3320 at 120 s + 1 s, from only 80 games; 0.9.0 has not been measured at that time control.
These are not CCRL ratings.

Windows and Linux binaries are on the
[latest release](https://github.com/CloseNoJona/ayano/releases/latest). Pick `bmi2` for most CPUs,
`avx2` for AMD Ryzen 1000 to 3000, and `portable` for CPUs without AVX2. To build it yourself, use
`scripts\build.cmd` on Windows or `make -j4` on Linux. The license is GPL-3.0 or later.

# Enroque

Motor de ajedrez UCI escrito desde cero en C++20, con las técnicas de los motores modernos:
bitboards, búsqueda alfa-beta con podas agresivas y evaluación con una red neuronal NNUE
entrenada solo con sus propias partidas.

Como Stockfish, es un programa de consola sin tablero: se conecta a una interfaz gráfica
mediante el protocolo UCI.

## Resultados

Medido en un AMD Ryzen 5 5600 (6 núcleos, 12 hilos). Las cifras de Elo llevan su margen de
error del 95 %.

### Fuerza absoluta

Con la red gen3, Enroque ronda los **3015 ± 35 Elo** en la escala `UCI_Elo` de Stockfish 19.
Cada motor dispuso de un millón de nodos por jugada, unos 0,4 s para Enroque.

| Rival | Partidas | Resultado de Enroque | Diferencia |
|---|---|---|---|
| Stockfish 19 con `UCI_Elo` 3000 | 200 | +77 =57 −66 | +19 ± 41 |
| Stockfish 19 con `UCI_Elo` 3190, el máximo | 100 | +6 =38 −56 | −191 ± 56 |

Es una referencia, no un rating de CCRL ni un Elo humano. El margen solo recoge el azar de las
partidas: Stockfish calibró esa escala a 120 s + 1 s, no a esta velocidad. Stockfish sin limitar
es mucho más fuerte y no se ha medido.

Contra este rival, gen3 no se distingue de gen2, que da 3045 ± 35. La ventaja de gen3 solo se ha
comprobado en el duelo directo entre ambas. El detalle está en
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#fuerza-absoluta).

### Fuerza relativa

Estas cifras comparan versiones del propio motor, así que no tienen por qué repetirse contra
otros rivales.

| Partidas con 5 s por bando y 0,05 s de incremento | Partidas | Resultado |
|---|---|---|
| Red gen3 contra la evaluación clásica | 160 | **+364 ± 71 Elo** |
| Red gen3 contra la red gen2 | 1000 | **+39 ± 16 Elo** |
| 4 hilos contra 1 hilo | 100 | **+151 ± 53 Elo** |

### Otras medidas

| Medida | Resultado |
|---|---|
| Velocidad con la red | 2,5 millones de nodos por segundo y hilo |
| Profundidad completada desde la posición inicial | 23 en 5 segundos |
| Bratko-Kopec, 2 s y 2 hilos | 19 de 24 |
| Perft | Exacto en toda la suite, unos 800 millones de nodos |

## Compilar

Necesitas Visual Studio Build Tools con las herramientas de C++.

```bash
scripts\build.cmd
```

El ejecutable queda en `build\release\enroque.exe`.

| Variante | Comando | Para qué |
|---|---|---|
| Normal | `scripts\build.cmd` | CPUs con AVX2 y BMI2 |
| AVX2 sin PEXT | `scripts\build.cmd avx2` | AMD Ryzen 1000 a 3000, donde PEXT es lento |
| Portable | `scripts\build.cmd portable` | CPUs sin AVX2 |
| Más rápida | `scripts\build_pgo.cmd` | Entre un 3 y un 4 % más veloz gracias a un perfil real, en `build\pgo` |

## Jugar contra él

1. Instala una interfaz gratuita, como [Cute Chess](https://cutechess.com) o
   [Arena](http://www.playwitharena.de).
2. Añade `enroque.exe` como motor UCI.
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
build\release\enroque_tests.exe
```

```bash
build\release\enroque_tests.exe --slow
```

La opción `--slow` añade el perft completo, de unos 800 millones de nodos.

La compilación con AddressSanitizer (`scripts\build.cmd asan`) necesita el entorno de Visual
Studio para encontrar su biblioteca:

```bash
scripts\vs.cmd build\asan\enroque_tests.exe
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
match --engine1 enroque.exe --engine2 stockfish.exe --opt2 "UCI_LimitStrength=true;UCI_Elo=3000;Hash=64" --games 100 --concurrency 4 --nodes 1000000
```

**Medir la resolución de posiciones tácticas:**

```bash
solve --epd tests\data\bratko-kopec.epd --ms 5000 --threads 4
```

**Comprobar una red sin jugar partidas:**

```bash
netcheck --data data\gen0.bin --net nets\gen2.nnue
```

## Entrenar una red nueva

El ciclo completo (jugar, entrenar y comparar) es un solo comando:

```bash
powershell -File scripts\new_generation.ps1 -Name gen4 -Minutes 60
```

Cada generación necesita un nombre nuevo. Si la red nueva gana, cópiala a `nets\default.nnue` y
recompila: el ejecutable la integrará.
Las generaciones se pueden encadenar, porque cada una produce datos mejores que la anterior.
Mide también cada red contra Stockfish limitado: el duelo con la anterior puede exagerar la mejora.

La red forma parte del ejecutable y su tamaño de capa oculta es fijo al compilar. Para probar
otro, configura con `-DENROQUE_HIDDEN=N`: esa compilación no integra ninguna red, así que hay
que cargarla con `EvalFile`.

## Cómo está hecho

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explica el diseño, la búsqueda, la red y cómo se
verifica cada parte.

## Créditos

El código es original. Las técnicas están documentadas en la
[Chess Programming Wiki](https://www.chessprogramming.org) y en la comunidad de motores
abiertos. Las tablas pieza-casilla de la evaluación clásica son las de PeSTO, de Ronald
Friederich, publicadas en esa wiki. Ninguna red ni dato de entrenamiento procede de otro motor.
Stockfish solo se usó como rival para medir la fuerza: ni su código ni sus partidas forman parte
de Enroque.

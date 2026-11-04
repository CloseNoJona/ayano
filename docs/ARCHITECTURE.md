# Arquitectura de Enroque

Enroque es un motor UCI escrito desde cero en C++20. Usa las mismas técnicas que los
motores modernos de primer nivel: bitboards, búsqueda alfa-beta con podas agresivas y
evaluación con una red NNUE. El código es original y las técnicas están descritas
públicamente, sobre todo en la [Chess Programming Wiki](https://www.chessprogramming.org).

## Mapa del código

| Carpeta | Contenido |
|---|---|
| `src/core` | Tipos, bitboards, tablas de ataque (PEXT o magics), claves Zobrist |
| `src/board` | Posición inmutable, FEN, generación de jugadas, SEE, perft |
| `src/eval` | Evaluación clásica, usada como respaldo y para arrancar el entrenamiento |
| `src/nnue` | Red neuronal: inferencia AVX2, acumuladores, formato de datos |
| `src/train` | Entrenador de la red en CPU |
| `src/search` | Búsqueda, hilos, tabla de transposición, historiales, tiempo |
| `src/uci` | Protocolo UCI y benchmark |
| `tools` | `datagen`, `train`, `match`, `solve`, `netcheck` |
| `tests` | Tests unitarios y de propiedades |

## Decisiones de diseño

### Posición inmutable (copy-make)

Aplicar una jugada crea una posición nueva; la original nunca cambia. La búsqueda escribe
cada posición en su pila, de modo que no hace falta deshacer jugadas. Copiar unos 200 bytes
por nodo cuesta poco y elimina una clase entera de errores de estado. También hace trivial
que cada hilo trabaje por su cuenta.

### Jugadas pseudo-legales y generación por etapas

El generador produce jugadas pseudo-legales por etapas: primero capturas y después jugadas
tranquilas. `MovePicker` entrega la jugada de la tabla hash, las capturas buenas, las
refutaciones y por último el resto. Si una jugada temprana produce un corte, las demás ni
se generan. La legalidad se comprueba justo antes de jugar, usando las clavadas que cada
posición ya tiene calculadas.

### Tabla de transposición sin bloqueos

Cada entrada son dos enteros atómicos de 64 bits: los datos y `clave XOR datos`. Si dos
hilos escriben a la vez y una lectura mezcla mitades, la comprobación falla y la entrada se
ignora. Las operaciones `relaxed` compilan a instrucciones normales, así que no hay coste
y tampoco hay carrera de datos desde el punto de vista del estándar.

### Lazy SMP

Todos los hilos ejecutan la misma búsqueda iterativa y comparten solo la tabla de
transposición. Al terminar, votan la mejor jugada según profundidad y puntuación. Con reloj
real, 4 hilos ganan a 1 hilo por +151 ± 53 Elo.

### Tiempo con el reloj bajo

Cada jugada tiene un objetivo, que se ajusta según lo estable que sea la mejor jugada, y un
límite duro que la búsqueda comprueba cada 1024 nodos. Aun así, el sistema puede retrasar un
hilo: se midió un exceso de 16 ms sobre el límite, casi un tic del temporizador de Windows
(15,6 ms), con solo 86 ms en el reloj. Por eso el límite duro deja siempre un 30 % del reloj y
un margen fijo. Perder una partida por tiempo cuesta mucho más que pensar un poco menos.

| Margen mínimo de reloj en cada control | Antes | Ahora |
|---|---|---|
| 1 s + 0,01 | 13 ms | 29 ms |
| 5 s + 0,05 | 10 ms | 47 ms |

El cambio no cuesta fuerza: +5 ± 15 Elo contra la versión anterior a 1 s + 0,01, donde más
actúa. Antes de él, una de 200 partidas terminó con una anomalía cuyo motivo no quedó guardado;
no se repitió en las 3800 siguientes y `match` ahora registra el contexto completo de cada una.

## Búsqueda

Cada nodo sigue el mismo camino, separado en funciones pequeñas en `src/search/search.cpp`:

1. **Entrada** (`enter_node`): tablas por repetición, regla de 50 jugadas, material
   insuficiente, poda por distancia al mate.
2. **Tabla hash** (`probe_tt`): corte directo si la entrada es lo bastante profunda.
3. **Evaluación estática** (`compute_static_eval`): red o evaluación clásica, ajustada con
   el historial de corrección.
4. **Podas previas** (`forward_prune`, `probcut`): razoring, futilidad inversa, jugada nula
   con verificación, reducción interna y ProbCut.
5. **Bucle de jugadas** (`move_loop`): poda por conteo, futilidad, SEE e historial;
   extensiones singulares, dobles, triples y negativas; reducciones tardías (LMR) con
   re-búsqueda; ventana nula (PVS).
6. **Cierre** (`finish_node`): actualiza historiales, tabla hash e historial de corrección.

La búsqueda de quietud (`qsearch`) resuelve las capturas pendientes para que la evaluación
se aplique solo a posiciones tranquilas.

Historiales: mariposa (origen-destino), capturas, continuación a 1, 2, 4 y 6 medias
jugadas, estructura de peones, jugadas asesinas, respuesta a la jugada anterior y corrección
de la evaluación por peones, piezas menores y resto de piezas.

## Red NNUE

```
768 entradas por perspectiva (pieza x casilla, reflejadas)
   -> capa oculta de 512 neuronas con SCReLU, una por cada perspectiva
   -> 1024 valores -> 1 salida, elegida entre 8 cubetas según las piezas que quedan
```

* **Dos perspectivas:** la red ve el tablero desde cada bando, siempre con el bando propio
  abajo. La primera mitad de la salida corresponde a quien tiene el turno.
* **Reflejo horizontal:** si el rey de una perspectiva está en las columnas e-h, su tablero
  se refleja. La red solo aprende posiciones con el rey en el flanco de dama, y necesita
  menos datos para generalizar.
* **Actualización incremental:** una jugada cambia dos o tres entradas, así que la capa
  oculta se actualiza con sumas y restas de filas. El cálculo es perezoso: solo se hace para
  nodos que se evalúan de verdad, partiendo del antepasado más cercano ya calculado.
* **Cuantización:** enteros de 16 bits con escalas 255 y 64. El producto `x * w` cabe en 16
  bits, lo que permite calcular `x² * w` con `_mm256_madd_epi16`.
* **Escala propia:** cada archivo de red guarda los centipeones por unidad de salida, así que
  una red no puede usarse con una escala distinta de la que se entrenó.

## Cómo se entrena la red

```
evaluación clásica -> partidas contra sí mismo -> red 1 -> más partidas -> red 2 -> ...
```

1. `datagen` juega partidas desde aperturas aleatorias con un límite de nodos y guarda las
   posiciones tranquilas junto con la evaluación de la búsqueda y el resultado final.
2. `train` ajusta la red para predecir `sigmoide(evaluación / K)`. Usa Adam y varios hilos:
   la entrada dispersa hace que la CPU baste para esta arquitectura.
3. `match` comprueba con partidas que la red nueva es más fuerte que la anterior.

Cada generación juega mejor que la anterior, por lo que produce datos mejores. Ninguna red
ni dato procede de otro motor.

### Progreso por generación

| Red | Posiciones nuevas | Total de posiciones | Contra la anterior, a 10 000 nodos | Contra la anterior, a 5 s + 0,05 |
|---|---|---|---|---|
| gen1 | 5,1 millones | 7,9 millones | +126 ± 31 | Sin medir |
| gen2 | 9,7 millones | 17,6 millones | +75 ± 29 | +111 ± 16 |
| gen3 | 6,1 millones | 23,7 millones | +30 ± 29 | +39 ± 16 |

La ventaja de gen2 sobre gen1 crece con el tiempo de reflexión: +75 ± 29 a 10 000 nodos,
+90 ± 17 a 1 s + 0,01 y +111 ± 16 a 5 s + 0,05. Una evaluación mejor pesa más cuanto más
profunda es la búsqueda, así que las pruebas rápidas la infravaloran.

gen3 mejora menos que gen2. También aportó menos posiciones nuevas, pero no se ha medido cuánto
pesó eso. A 10 000 nodos su ventaja apenas supera el margen de error; fueron las 1000 partidas
con reloj las que la confirmaron. Contra Stockfish esa ventaja no se aprecia: ver
[Fuerza absoluta](#fuerza-absoluta).

### Qué se midió al entrenar

Cada decisión se tomó jugando partidas, no mirando la pérdida del entrenamiento. Las cifras
son de partidas a igual número de nodos, con margen de error del 95 %, salvo que se indique.

| Pregunta | Resultado | Decisión |
|---|---|---|
| ¿Mezclar el resultado de la partida en el objetivo? | Sin mezcla +142 ± 54 contra la evaluación clásica; con un 40 % de mezcla, −150 ± 56 | No mezclar |
| ¿Faltaban datos? | Con la mitad de los datos, −166 ± 58 frente a −150: casi igual | No era la causa |
| ¿Qué escala K? | K=140 gana a K=400 por +47 ± 41; K=100 y K=200 son indistinguibles de K=140 | K=200 |
| ¿Mezclar el resultado con K calibrada? | −61 ± 43 contra K=400 sin mezcla, unos −108 respecto a K=140 sin mezcla | No mezclar |
| ¿Cuántas neuronas? | 256: −23 ± 38. 1024: −22 ± 38 y un 25 % más lenta | 512 |
| ¿Usar también datos de generaciones anteriores? | +126 ± 31 con todos, +113 ± 32 solo con los nuevos | Usar todos |

La primera red mezclaba el resultado y perdía por 150 Elo pese a tener una pérdida baja. Hay
dos causas. La primera es de escala: en estas partidas una ventaja de 250 cp gana el 85 % de
las veces, mientras que `sigmoide(x / 400)` predice un 65 %, así que la mezcla inflaba las
evaluaciones y los márgenes de poda dejaban de tener sentido. La segunda es de cantidad: unos
millones de posiciones vienen de pocas decenas de miles de partidas, y todas las de una
partida comparten un único resultado, de modo que la red memoriza ruido. Con muchas más
partidas la mezcla podría volver a compensar.

## Cómo se verifica

| Comprobación | Qué garantiza |
|---|---|
| Perft contra valores publicados | Reglas exactas: enroque, al paso, promoción, clavadas |
| Hash incremental contra hash desde cero | Las claves no se desincronizan |
| `gives_check` contra la posición resultante | La detección de jaques es exacta |
| Acumulador incremental contra recálculo | La red evalúa igual por cualquier camino |
| Gradiente contra diferencias finitas | La retropropagación es correcta |
| Red cuantizada contra red flotante | La cuantización no altera la fórmula |
| Compilación con AddressSanitizer | No hay accesos inválidos a memoria |
| Compilación con `ENROQUE_VERIFY_NNUE` | La red no se desincroniza durante una búsqueda real |
| Compilación portable | El camino sin AVX2 ni PEXT también es correcto |
| `bench` | Huella que detecta cambios involuntarios en la búsqueda |
| `bench` en cada variante | Release, portable, AVX2 sin PEXT y PGO recorren los mismos nodos |
| `match` con SPRT | Cada mejora se confirma jugando, no por intuición |
| `match` contra Stockfish limitado | Sitúa al motor en una escala externa |

### Fuerza absoluta

Stockfish 19 limitado hace de regla: su documentación calibra `UCI_Elo` a 120 s + 1 s y la ancla
a CCRL 40/4. Las redes jugaron contra varios niveles, con aperturas al azar de 8 medias jugadas y
un millón de nodos por jugada para cada motor. Limitar por nodos hace que el resultado no dependa
de la carga de la máquina.

| Rival | gen3 | gen2 |
|---|---|---|
| `UCI_Elo` 2800 | Sin medir | +32 =5 −3 |
| `UCI_Elo` 3000, primera tanda | +40 =28 −32 | +22 =10 −8 |
| `UCI_Elo` 3000, las dos a la vez | +37 =29 −34 | +35 =33 −32 |
| `UCI_Elo` 3190 | +6 =38 −56 | +13 =31 −56 |
| **Elo estimado** | **3014** (de 2981 a 3047) | **3045** (de 3010 a 3080) |

Un ajuste por máxima verosimilitud, con las tablas como medio punto, busca el Elo que mejor
explica los resultados de cada red. El intervalo del 95 % sale de remuestrear las partidas. En
gen3 los dos niveles coinciden: por separado dan 3019 y 2999.

gen3 gana a gen2 por +39 ± 16 en duelo directo, pero contra Stockfish esa mejora no se ve: la
diferencia es de −31 ± 48. Cuando las dos jugaron a la vez, con las mismas aperturas, ambas
puntuaron un 51,5 %. Esos datos no bastan para decir que gen2 sea mejor, así que gen3 sigue
integrada. Una explicación posible es que gen3 aprendió de partidas de gen2 y puede ganarle sin
ganar lo mismo a un motor distinto, pero no se ha demostrado. Por eso conviene medir cada red
nueva también contra un rival externo.

Las tandas cortas engañan. La primera medida de gen2 dio 3068, con un 67,5 % en 40 partidas
contra el nivel 3000. Al repetirla con 100 partidas, ese porcentaje bajó al 51,5 %.

El intervalo solo recoge el azar de las partidas. Quedan dos incertidumbres mayores:

- **La velocidad.** La escala se calibró a 120 s + 1 s y aquí cada jugada duró unos 0,4 s. Con
  más tiempo la diferencia puede cambiar; no se ha medido.
- **El tipo de rival.** Stockfish limitado juega a propósito, con cierta probabilidad, una jugada
  peor. Eso no equivale a enfrentarse a un motor de esa fuerza, así que la cifra no es un rating
  de CCRL.

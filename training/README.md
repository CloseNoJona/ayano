# Cómo se entrena la red

La red integrada, `nets/default.nnue`, se entrenó en GPU con
[bullet](https://github.com/jw1912/bullet) y datos públicos de
[Leela Chess Zero](https://lczero.org), de su ciclo T91. xu-shawn los empaquetó en formato binpack, y
Adam Twiss los minimizó y los filtró con el fork de Stockfish de linrock. Aquí está lo necesario para
repetirlo.

| Archivo | Qué es |
|---|---|
| `ayano.rs` | El entrenador, un programa para bullet |
| `convert_net.py` | Pasa la salida de bullet al formato de Ayano y comprueba que ambos evalúan igual |
| `calibrate_scale.py` | Calcula la escala de una red nueva |
| `calibration.fen` | 712 posiciones, algunas repetidas, de las primeras partidas del motor en Lichess |
| `LICENSE.bullet` | Licencia MIT de bullet, de cuyo ejemplo parte `ayano.rs` |

## Qué hace falta

- Una GPU NVIDIA con CUDA. Se usó una RTX 3060 de 12 GB con CUDA 13.5.
- Rust estable. En Windows, también Visual Studio Build Tools.
- Python con `numpy` y `python-chess`.
- 13 GB de disco para los datos.

Todo esto solo se ha probado en Windows.

## Pasos

Los comandos parten de una carpeta de trabajo con `ayano` (este repositorio), `bullet` y
`data/t91`, y usan la sintaxis de bash.

1. **Baja bullet** y ponlo en la versión usada:

   ```bash
   git clone https://github.com/jw1912/bullet
   git -C bullet checkout 5d70177
   cp ayano/training/ayano.rs bullet/examples/
   ```

   Declara el ejemplo en `bullet/crates/bullet_lib/Cargo.toml`:

   ```toml
   [[example]]
   name = "ayano"
   path = "../../examples/ayano.rs"
   ```

2. **Baja los datos.** Son los cinco archivos `.binpack` de
   [`adamtwiss/t91-binpacks-filtered`](https://huggingface.co/datasets/adamtwiss/t91-binpacks-filtered),
   12,6 GB. Van en `data/t91`, y su página publica las sumas MD5 para comprobarlos.

3. **Compila y entrena.** La red `b1` tardó 6 h 40 min:

   ```bash
   (cd bullet && cargo build --release -p bullet_lib --example ayano --features cuda)
   AYANO_NET=b1 AYANO_SB=320 bullet/target/release/examples/ayano
   ```

   Guarda una copia cada 10 superlotes en `checkpoints`. El resultado final queda en
   `checkpoints/b1-320/quantised.bin`.

4. **Compila el motor.** En Windows, `scripts\build.cmd` deja `build\release\ayano.exe`. En Linux,
   `make` deja `ayano` en la carpeta del repositorio.

5. **Convierte la red y calibra su escala.** La escala hace que las evaluaciones midan como las de
   una red anterior, porque los márgenes de poda de la búsqueda se ajustaron con redes de esa escala.
   Aquí `motor` es el ejecutable que acabas de compilar y `motor-anterior` el de esa red, que para
   `b1` fue la 0.9.1:

   ```bash
   python ayano/training/convert_net.py checkpoints/b1-320/quantised.bin b1-400.nnue 400 motor
   python ayano/training/calibrate_scale.py motor-anterior motor b1-400.nnue 400
   python ayano/training/convert_net.py checkpoints/b1-320/quantised.bin b1.nnue 155 motor
   ```

   `b1` dio 145 por pendiente y 163 por desviación, y se eligió 155, un valor intermedio. No se ha
   medido con partidas si otro valor rinde más.

   Con un motor como último argumento, `convert_net.py` compara sus evaluaciones con las de una
   versión independiente en Python y se detiene si no coinciden.

6. **Mide la red antes de adoptarla**, contra la actual y contra rivales externos, con `match`. Si
   gana, cópiala a `nets/default.nnue` y recompila. Un duelo solo contra la red anterior puede
   exagerar la mejora.

## Qué posiciones se usan

`ayano.rs` descarta al leer las posiciones de las 16 primeras medias jugadas, las que están en jaque
y aquellas cuya jugada es una captura, una promoción, un enroque o una captura al paso. También
descarta las que tienen una evaluación de más de 10 000, lo que incluye las marcadas por el filtro
de los datos.

| Parámetro | Valor |
|---|---|
| Arquitectura | `(768x8hm -> 1024)x2 -> 8` |
| Lote | 16 384 posiciones, 6104 lotes por superlote |
| Superlotes | 320, unos 32 000 millones de posiciones vistas |
| Peso del resultado de la partida | De 0,2 a 0,4 |
| Tasa de aprendizaje | De 0,001 a 0,0000024, con caída coseno |
| Escala del entrenamiento | 400 |

Cambiar el tamaño de la red exige tocar `HIDDEN` en `ayano.rs`, las constantes de `convert_net.py` y
`AYANO_HIDDEN` al compilar. La disposición de las cubetas de rey debe coincidir en `ayano.rs`,
`convert_net.py` y `src/nnue/network.h`.

## Licencia de los datos

Contains information from
[Filtered LC0 T91 (min-v2.v6)](https://huggingface.co/datasets/adamtwiss/t91-binpacks-filtered),
which is made available here under the
[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).

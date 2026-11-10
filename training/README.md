# Cómo se entrena la red

La red integrada, `nets/default.nnue`, se entrenó en GPU con
[bullet](https://github.com/jw1912/bullet) y datos públicos de
[Leela Chess Zero](https://lczero.org), de su ejecución de prueba T80 de enero de 2024. Los publica
linrock, ya convertidos al formato binpack y filtrados (versión `min-v2.v6`). En las mediciones esta
red se llama c1. Aquí está lo necesario para repetirla.

La red de la 0.10.0, b1, salió de la misma receta con los datos del ciclo T91 que xu-shawn empaquetó
y Adam Twiss filtró. Con la búsqueda de la 0.11 y a igual número de nodos, c1 le gana por
+25,7 ± 12,4 Elo (1772 partidas).

| Archivo | Qué es |
|---|---|
| `ayano.rs` | El entrenador, un programa para bullet |
| `convert_net.py` | Pasa la salida de bullet al formato de Ayano y comprueba que ambos evalúan igual |
| `calibrate_scale.py` | Calcula la escala de una red nueva |
| `calibration.fen` | 712 posiciones, algunas repetidas, de las primeras partidas del motor en Lichess |
| `ayano_deep.rs`, `convert_deep_net.py` | Lo mismo para una red con cabeza profunda, experimental: ver [Cabeza profunda](#cabeza-profunda-experimental) |
| `LICENSE.bullet` | Licencia MIT de bullet, de cuyos ejemplos parten `ayano.rs` y `ayano_deep.rs` |

## Qué hace falta

- Una GPU NVIDIA con CUDA. Se usó una RTX 3060 de 12 GB con CUDA 13.5.
- Rust estable. En Windows, también Visual Studio Build Tools.
- Python con `numpy` y `python-chess`, y algo que descomprima zstd: `zstd` o Python 3.14, que lo
  trae en `compression.zstd`.
- 17 GB de disco mientras se descomprimen los datos, 9,2 GB después.

Todo esto solo se ha probado en Windows.

## Pasos

Los comandos parten de una carpeta de trabajo con `ayano` (este repositorio), `bullet` y
`data/t80`, y usan la sintaxis de bash.

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

2. **Baja los datos.** Es el archivo `test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst` de
   [`linrock/test80-2024`](https://huggingface.co/datasets/linrock/test80-2024), 7,7 GB. Su página
   publica la suma SHA-256 de cada archivo. Descomprimido ocupa 9,2 GB y va en `data/t80`:

   ```bash
   zstd -d test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst -o data/t80/test80-2024-01-jan-2tb7p.min-v2.v6.binpack
   ```

3. **Compila y entrena.** `AYANO_DATA` es la carpeta de los datos: el entrenador lee todos sus
   archivos. La red `c1` tardó 7 h 6 min, con partidas de prueba a la vez en la CPU:

   ```bash
   (cd bullet && cargo build --release -p bullet_lib --example ayano --features cuda)
   AYANO_NET=c1 AYANO_SB=320 AYANO_DATA=data/t80 bullet/target/release/examples/ayano
   ```

   Guarda una copia cada 10 superlotes en `checkpoints`. El resultado final queda en
   `checkpoints/c1-320/quantised.bin`.

4. **Compila el motor.** En Windows, `scripts\build.cmd` deja `build\release\ayano.exe`. En Linux,
   `make` deja `ayano` en la carpeta del repositorio.

5. **Convierte la red y calibra su escala.** La escala hace que las evaluaciones midan como las de
   la red anterior, porque los márgenes de poda de la búsqueda se ajustaron con redes de esa escala.
   Aquí `motor` es el ejecutable que acabas de compilar, con la red anterior integrada:

   ```bash
   python ayano/training/convert_net.py checkpoints/c1-320/quantised.bin c1-155.nnue 155 motor
   python ayano/training/calibrate_scale.py motor motor c1-155.nnue 155
   python ayano/training/convert_net.py checkpoints/c1-320/quantised.bin c1.nnue 144 motor
   ```

   Frente a b1, `c1` dio 141 por pendiente y 144 por desviación, y se eligió 144, que iguala la
   dispersión de las evaluaciones. Con b1 frente a la 0.9.1 salieron 145 y 163, y se eligió 155. No se
   ha medido con partidas si otro valor rinde más. Los dos motores deben evaluar igual salvo por la
   red: `eval` devuelve la evaluación que usa la búsqueda, con todos sus ajustes.

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

## Cabeza profunda (experimental)

El motor lee también, con `EvalFile`, redes con una cabeza más profunda: la capa oculta multiplica
sus neuronas por parejas y, en cada cubeta de salida, siguen capas de 16, 32 y 1 neuronas (formato
ENQ3). La primera, d1, perdió contra c1 a igual tiempo, así que no es la integrada: ver
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#cabeza-profunda-experimental).

Se entrena como la lineal, con `ayano_deep.rs` declarado también en `Cargo.toml` (`name =
"ayano_deep"`, `path = "../../examples/ayano_deep.rs"`). d1 usó los datos de enero y de febrero de
2024; el de febrero es `test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst`, de 6,9 GB (8,3 GB
descomprimido). Los dos van en otra carpeta, `data/t80-ene-feb`, para que la receta de c1 siga
leyendo solo enero:

```bash
(cd bullet && cargo build --release -p bullet_lib --example ayano_deep --features cuda)
AYANO_NET=d1 AYANO_SB=320 AYANO_DATA=data/t80-ene-feb bullet/target/release/examples/ayano_deep
python ayano/training/convert_deep_net.py checkpoints/d1-320/quantised.bin d1-400.nnue 400 motor
python ayano/training/calibrate_scale.py motor motor d1-400.nnue 400
python ayano/training/convert_deep_net.py checkpoints/d1-320/quantised.bin d1.nnue 152 motor
```

`convert_deep_net.py` reordena las neuronas por actividad, medida en `calibration.fen`, para que el
motor salte más grupos de entradas a cero; la red evalúa exactamente igual. Con un motor como último
argumento compara sus evaluaciones con las de una versión en Python, como `convert_net.py`.

## Licencia de los datos

Leela Chess Zero publica sus datos de entrenamiento con la Open Database License y su contenido con
la Database Contents License. La página de `linrock/test80-2024` no indica licencia.

Contains information from
[Leela Chess Zero training data, test80 2024, filtered (min-v2.v6)](https://huggingface.co/datasets/linrock/test80-2024),
which is made available here under the
[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).

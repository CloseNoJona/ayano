"""Calcula la escala de una red nueva para que sus evaluaciones midan como las de otra anterior.

Los margenes de poda se ajustaron con redes de una escala concreta: si la nueva da numeros mucho
mayores o menores, la busqueda poda de mas o de menos.

Uso: calibrate_scale.py motor_anterior motor_nuevo red_nueva.nnue escala_de_esa_red [fens]
`fens` es un archivo con una posicion por linea; por defecto, calibration.fen.
"""
import pathlib
import statistics as st
import subprocess
import sys
from collections.abc import Sequence

DEFAULT_FENS = pathlib.Path(__file__).with_name('calibration.fen')
MAX_OLD_EVAL = 1500
MAX_NEW_EVAL = 6000


def evals(engine: str, fens: list[str], options: Sequence[tuple[str, str]] = ()) -> list[int]:
    lines = [f'setoption name {name} value {value}' for name, value in options]
    for fen in fens:
        lines += [f'position fen {fen}', 'eval']
    lines.append('quit')
    result = subprocess.run([engine], input='\n'.join(lines) + '\n', capture_output=True,
                            text=True, timeout=300, check=True)
    if options and 'red cargada' not in result.stdout:
        sys.exit('el motor no cargo la red: compararia su red integrada consigo misma')
    values = [int(line.split(':')[1].split()[0]) for line in result.stdout.splitlines()
              if line.startswith('Evaluacion')]
    if len(values) != len(fens):
        sys.exit(f'{len(values)} evaluaciones para {len(fens)} posiciones')
    return values


def main() -> None:
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    previous_engine, new_engine, net, scale = sys.argv[1:5]
    scale = int(scale)
    path = pathlib.Path(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_FENS
    fens = path.read_text(encoding='utf-8').split('\n')
    fens = [fen for fen in fens if fen.strip()]
    old = evals(previous_engine, fens)
    new = evals(new_engine, fens, [('EvalFile', net)])
    pairs = [(a, b) for a, b in zip(old, new) if abs(a) < MAX_OLD_EVAL and abs(b) < MAX_NEW_EVAL]
    # Pendiente por minimos cuadrados con ordenada cero: red anterior sobre red nueva.
    slope = sum(a * b for a, b in pairs) / sum(b * b for _, b in pairs)
    spread_old = st.pstdev(a for a, _ in pairs)
    spread_new = st.pstdev(b for _, b in pairs)
    mean_old = st.mean(a for a, _ in pairs)
    mean_new = st.mean(b for _, b in pairs)
    covariance = sum((a - mean_old) * (b - mean_new) for a, b in pairs) / len(pairs)
    print(f'{len(pairs)} posiciones; correlacion {covariance / (spread_old * spread_new):.3f}')
    print(f'desviacion anterior {spread_old:.0f} nueva {spread_new:.0f}; pendiente {slope:.3f}')
    print(f'escala por pendiente {scale * slope:.0f}; por desviacion '
          f'{scale * spread_old / spread_new:.0f}')


if __name__ == '__main__':
    main()

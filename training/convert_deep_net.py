"""Convierte una red de cabeza profunda de bullet al formato ENQ3 de Ayano y comprueba que el
motor evalua igual que una referencia escrita aparte.

La referencia sigue las reglas del entrenador (tablero relativo al bando con turno, parejas de la
neurona i con la i + HIDDEN / 2) y repite en float32, en el mismo orden, las capas en coma flotante:
asi el motor debe coincidir al centipeon.
Uso: convert_deep_net.py checkpoints/x/quantised.bin salida.nnue [escala] [motor]
Con --sintetica en lugar del primer argumento genera una red aleatoria para probar el conversor.
"""
import pathlib
import struct
import sys

import chess
import numpy as np

# Posiciones, lectura del motor y entradas, de convert_net.py: sus posiciones tienen el contador
# de 50 jugadas a cero, porque el motor atenua la evaluacion.
from convert_net import TEST_FENS, engine_evals, perspective_features

BUCKETS, HIDDEN, L1, L2 = 8, 1024, 16, 32
INPUTS = 768 * BUCKETS
QA, Q1 = 255, 64
PAIR_SHIFT, PAIR_MAX = 9, (255 * 255) >> 9
MAGIC = b'ENQ3'
SECTIONS = [  # nombre, tipo, cantidad
    ('l0w', '<i2', INPUTS * HIDDEN),
    ('l0b', '<i2', HIDDEN),
    ('l1w', 'i1', BUCKETS * L1 * HIDDEN),
    ('l1b', '<f4', BUCKETS * L1),
    ('l2w', '<f4', BUCKETS * L2 * L1),
    ('l2b', '<f4', BUCKETS * L2),
    ('l3w', '<f4', BUCKETS * L2),
    ('l3b', '<f4', BUCKETS),
]


def section_bytes():
    return sum(np.dtype(kind).itemsize * count for _, kind, count in SECTIONS)


def read_sections(path):
    data = open(path, 'rb').read()
    total = section_bytes()
    assert total <= len(data) < total + 64, (total, len(data))
    sections, offset = {}, 0
    for name, kind, count in SECTIONS:
        sections[name] = np.frombuffer(data, dtype=kind, count=count, offset=offset)
        offset += np.dtype(kind).itemsize * count
    return sections


def synthetic_sections(seed=1):
    rng = np.random.default_rng(seed)
    return {
        'l0w': rng.integers(-120, 121, INPUTS * HIDDEN).astype('<i2'),
        'l0b': rng.integers(-200, 201, HIDDEN).astype('<i2'),
        'l1w': rng.integers(-128, 128, BUCKETS * L1 * HIDDEN).astype('i1'),
        'l1b': rng.uniform(-1, 1, BUCKETS * L1).astype('<f4'),
        'l2w': rng.uniform(-1.5, 1.5, BUCKETS * L2 * L1).astype('<f4'),
        'l2b': rng.uniform(-0.5, 0.5, BUCKETS * L2).astype('<f4'),
        'l3w': rng.uniform(-1.5, 1.5, BUCKETS * L2).astype('<f4'),
        'l3b': rng.uniform(-0.5, 0.5, BUCKETS).astype('<f4'),
    }


def screlu(x):
    clipped = min(max(x, np.float32(0)), np.float32(1))
    return clipped * clipped


def evaluate(sections, fen, scale):
    weights = sections['l0w'].reshape(INPUTS, HIDDEN).astype(np.int64)
    bias = sections['l0b'].astype(np.int64)
    board = chess.Board(fen)
    bucket = (len(board.piece_map()) - 2) // 4
    pairs = []
    for stm_view in (True, False):
        acc = bias.copy()
        for feature in perspective_features(board, stm_view):
            acc += weights[feature]
        clipped = np.clip(acc, 0, QA)
        pairs.append((clipped[:HIDDEN // 2] * clipped[HIDDEN // 2:]) >> PAIR_SHIFT)
    inputs = np.concatenate(pairs)
    sums = sections['l1w'].reshape(BUCKETS, L1, HIDDEN)[bucket].astype(np.int64) @ inputs
    f32 = np.float32
    first = [screlu(f32(int(s)) / f32(PAIR_MAX * Q1) + sections['l1b'][bucket * L1 + o])
             for o, s in enumerate(sums)]
    second = []
    for k in range(L2):
        total = sections['l2b'][bucket * L2 + k]
        for o in range(L1):
            total = total + sections['l2w'][(bucket * L2 + k) * L1 + o] * first[o]
        second.append(screlu(total))
    # Tercera capa como en el motor: ocho sumas parciales por resto modulo 8, juntadas de dos en
    # dos ((0+4) + (2+6)) + ((1+5) + (3+7)), y al final el sesgo.
    lanes = [f32(0)] * 8
    for k in range(L2):
        lanes[k % 8] = lanes[k % 8] + sections['l3w'][bucket * L2 + k] * second[k]
    total = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]))
    output = sections['l3b'][bucket] + total
    return int(output * f32(scale))


def write_network(sections, target, scale):
    # bullet guarda las capas como [cubeta][neurona][entrada]. El motor quiere la primera como
    # [cubeta][grupo de 4 entradas][neurona][entrada del grupo] y la segunda como
    # [cubeta][entrada][neurona]. Las demas secciones van tal cual.
    engine_layout = dict(sections)
    engine_layout['l1w'] = sections['l1w'].reshape(BUCKETS, L1, HIDDEN // 4, 4).transpose(0, 2, 1, 3)
    engine_layout['l2w'] = sections['l2w'].reshape(BUCKETS, L2, L1).transpose(0, 2, 1)
    with open(target, 'wb') as out:
        out.write(struct.pack('<4siiii', MAGIC, INPUTS, HIDDEN, BUCKETS, scale))
        out.write(struct.pack('<ii', L1, L2))
        for name, kind, _ in SECTIONS:
            out.write(np.ascontiguousarray(engine_layout[name], dtype=kind).tobytes())


def pair_activity(sections, fens):
    """Veces que cada pareja de neuronas da un producto distinto de cero, en ambas perspectivas."""
    weights = sections['l0w'].reshape(INPUTS, HIDDEN).astype(np.int64)
    bias = sections['l0b'].astype(np.int64)
    counts = np.zeros(HIDDEN // 2, dtype=np.int64)
    for fen in fens:
        board = chess.Board(fen)
        for stm_view in (True, False):
            acc = bias.copy()
            for feature in perspective_features(board, stm_view):
                acc += weights[feature]
            clipped = np.clip(acc, 0, QA)
            counts += ((clipped[:HIDDEN // 2] * clipped[HIDDEN // 2:]) >> PAIR_SHIFT) > 0
    return counts


def permute_pairs(sections, fens):
    """Ordena las parejas de la mas a la menos activa. La red evalua exactamente igual, pero las
    activas quedan juntas y el motor salta mas grupos de 4 entradas a cero (idea de Stockfish)."""
    order = np.argsort(-pair_activity(sections, fens), kind='stable')
    neurons = np.concatenate([order, order + HIDDEN // 2])
    permuted = dict(sections)
    permuted['l0w'] = sections['l0w'].reshape(INPUTS, HIDDEN)[:, neurons].reshape(-1)
    permuted['l0b'] = sections['l0b'][neurons]
    permuted['l1w'] = sections['l1w'].reshape(BUCKETS, L1, HIDDEN)[:, :, neurons].reshape(-1)
    return permuted


def active_groups(sections, fens):
    weights = sections['l0w'].reshape(INPUTS, HIDDEN).astype(np.int64)
    bias = sections['l0b'].astype(np.int64)
    total, views = 0, 0
    for fen in fens:
        board = chess.Board(fen)
        for stm_view in (True, False):
            acc = bias.copy()
            for feature in perspective_features(board, stm_view):
                acc += weights[feature]
            clipped = np.clip(acc, 0, QA)
            pairs = (clipped[:HIDDEN // 2] * clipped[HIDDEN // 2:]) >> PAIR_SHIFT
            total += np.count_nonzero(pairs.reshape(-1, 4).any(axis=1))
            views += 1
    return total / views


def sample_fens():
    """Posiciones de partidas reales de Ayano, las mismas que usa la calibracion de escala."""
    path = pathlib.Path(__file__).with_name('calibration.fen')
    return [fen for fen in path.read_text(encoding='utf-8').split('\n') if fen.strip()]


def main():
    source, target = sys.argv[1], sys.argv[2]
    scale = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    sections = synthetic_sections() if source == '--sintetica' else read_sections(source)
    for name in ('l1w', 'l1b', 'l2w', 'l3w'):
        values = sections[name]
        print(f'{name}: {float(values.min()):.3f} .. {float(values.max()):.3f}')
    if source != '--sintetica':
        fens = sample_fens()
        before = active_groups(sections, fens[1::2])
        sections = permute_pairs(sections, fens[::2])
        after = active_groups(sections, fens[1::2])
        print(f'grupos de 4 activos por perspectiva: {before:.1f} -> {after:.1f} de {HIDDEN // 8}')
    write_network(sections, target, scale)
    if len(sys.argv) > 4:
        got = engine_evals(sys.argv[4], target, TEST_FENS)
        want = [evaluate(sections, fen, scale) for fen in TEST_FENS]
        print('motor     ', got)
        print('referencia', want)
        print('COINCIDEN' if got == want else 'NO COINCIDEN')
        return 0 if got == want else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""Convierte la salida de bullet al formato de Ayano y comprueba que ambos evaluan igual.

La comprobacion usa las reglas del entrenador (tablero relativo al bando con turno), escritas
aparte del motor, y sus posiciones pasan por todas las cubetas de rey y por los dos flancos.

Uso: convert_net.py checkpoints/b1-320/quantised.bin salida.nnue [escala] [motor]
Necesita numpy y python-chess.
"""
import struct
import subprocess
import sys

import chess
import numpy as np

BUCKETS, HIDDEN, OUTPUTS = 8, 1024, 8
INPUTS = 768 * BUCKETS
QA, QB = 255, 64
LAYOUT = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6] + [7] * 16
MAGIC = b'ENQ2'
# Con el contador de 50 jugadas a cero, para que el motor no atenue la evaluacion.
TEST_FENS = [
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
    'r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 0 9',
    '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
    '8/8/4k3/8/8/3K4/4P3/8 b - - 0 1',
    '2r3k1/5pp1/p3p2p/1p1pP3/3P1P2/1P2K3/P5PP/2R5 w - - 0 30',
    '1rbq2k1/1Bp2pp1/P6p/1Q1PN3/3P1n2/2P3r1/1P4N1/R3R1K1 w - - 0 28',
    'r1b2r1k/ppq2pp1/4p2p/2b1PnB1/3pNP1P/3B3Q/PPP3P1/1K2R2R b - - 0 19',
    '2kr3r/ppp2ppp/8/8/8/8/PPP2PPP/2KR3R w - - 0 1',
    '8/1k6/8/3q4/3Q4/8/2K5/8 w - - 0 1',
    '8/5k2/8/8/4P3/8/6K1/8 b - - 0 1',
    'r4rk1/5ppp/8/8/8/8/PP6/1K1R3R w - - 0 1',
    '7k/8/8/8/3K4/8/4P3/8 w - - 0 1',
    'k7/8/8/4K3/8/8/4P3/8 b - - 0 1',
]


def read_sections(path: str) -> list[np.ndarray]:
    with open(path, 'rb') as source:
        data = source.read()
    sizes = (INPUTS * HIDDEN, HIDDEN, OUTPUTS * 2 * HIDDEN, OUTPUTS)
    total = 2 * sum(sizes)
    if not total <= len(data) < total + 64:
        sys.exit(f'tamano inesperado: {len(data)} bytes, se esperaban {total}')
    offset = 0
    sections = []
    for count in sizes:
        sections.append(np.frombuffer(data, dtype='<i2', count=count, offset=offset))
        offset += count * 2
    return sections


def bucket_of(square: int) -> int:
    """Casilla relativa al bando dueno del rey, como `ChessBucketsMirrored`."""
    file = square % 8
    return LAYOUT[(square // 8) * 4 + (file if file < 4 else 7 - file)]


def perspective_features(board: chess.Board, stm_view: bool) -> list[int]:
    """Entradas de una perspectiva, siguiendo el formato relativo al bando con turno."""
    turn = board.turn
    rel = (lambda s: s) if turn == chess.WHITE else (lambda s: s ^ 56)
    ours = rel(board.king(turn))
    theirs = rel(board.king(not turn)) ^ 56
    ksq = ours if stm_view else theirs
    flip = 7 if ksq % 8 > 3 else 0
    bucket = 768 * bucket_of(ksq)
    features = []
    for square, piece in board.piece_map().items():
        sq = rel(square)
        colour = 0 if piece.color == turn else 1
        kind = 64 * (piece.piece_type - 1)
        if stm_view:
            index = (0, 384)[colour] + kind + sq
        else:
            index = (384, 0)[colour] + kind + (sq ^ 56)
        features.append(bucket + (index ^ flip))
    return features


def screlu_sum(weights: np.ndarray, bias: np.ndarray, features: list[int],
               out_weights: np.ndarray) -> int:
    acc = bias.astype(np.int64).copy()
    for feature in features:
        acc += weights[feature]
    x = np.clip(acc, 0, QA)
    return int(np.sum(x * x * out_weights.astype(np.int64)))


def truncated_division(value: int, divisor: int) -> int:
    """Division entera hacia cero, como en C++."""
    quotient = abs(value) // divisor
    return quotient if value >= 0 else -quotient


def evaluate(sections: list[np.ndarray], fen: str, scale: int) -> int:
    weights, bias, out_weights, out_bias = sections
    weights = weights.reshape(INPUTS, HIDDEN)
    out_weights = out_weights.reshape(OUTPUTS, 2, HIDDEN)
    board = chess.Board(fen)
    bucket = (len(board.piece_map()) - 2) // 4
    total = screlu_sum(weights, bias, perspective_features(board, True), out_weights[bucket, 0])
    total += screlu_sum(weights, bias, perspective_features(board, False), out_weights[bucket, 1])
    scaled = truncated_division(total, QA) + int(out_bias[bucket])
    return truncated_division(scaled * scale, QA * QB)


def engine_evals(engine: str, net: str, fens: list[str]) -> list[int]:
    lines = [f'setoption name EvalFile value {net}']
    for fen in fens:
        lines += [f'position fen {fen}', 'eval']
    lines.append('quit')
    result = subprocess.run([engine], input='\n'.join(lines) + '\n', capture_output=True,
                            text=True, timeout=60, check=True)
    if 'red cargada' not in result.stdout:
        sys.exit('el motor no cargo la red')
    return [int(line.split(':')[1].split()[0]) for line in result.stdout.splitlines()
            if line.startswith('Evaluacion')]


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    source, target = sys.argv[1], sys.argv[2]
    scale = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    sections = read_sections(source)
    print('rango de pesos de entrada', int(sections[0].min()), int(sections[0].max()))
    print('rango de pesos de salida', int(sections[2].min()), int(sections[2].max()))
    with open(target, 'wb') as out:
        out.write(struct.pack('<4siiii', MAGIC, INPUTS, HIDDEN, OUTPUTS, scale))
        for section in sections:
            out.write(section.tobytes())
    if len(sys.argv) > 4:
        got = engine_evals(sys.argv[4], target, TEST_FENS)
        want = [evaluate(sections, fen, scale) for fen in TEST_FENS]
        print('motor     ', got)
        print('referencia', want)
        if got != want:
            sys.exit('NO COINCIDEN')
        print('COINCIDEN')


if __name__ == '__main__':
    main()

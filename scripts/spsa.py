"""Ajusta los parametros de la busqueda con SPSA, jugando a nodos fijos.

Uso: python scripts/spsa.py --engine build/tune/ayano.exe --match build/tune/match.exe
         --iterations 3000 --games 8 --nodes 30000 --workers 3 --state spsa.json

Cada iteracion enfrenta dos copias del motor con los parametros desplazados en sentidos
opuestos, y mueve los parametros hacia la que gana. Se puede interrumpir: con el mismo --state
continua donde iba. Al terminar imprime los valores para llevarlos a src/search/params.h.

SPSA siempre mueve los parametros, aunque no haya nada que ganar: el resultado solo vale si
gana despues a los valores originales en un match aparte.
"""
import argparse
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

PARAMS_FILE = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'search' / 'params.h'
PARAM_LINE = re.compile(r'X\((k\w+), (-?\d+), (-?\d+), (-?\d+)\)')
RESULT_LINE = re.compile(r'Partidas \d+ \| \+(\d+) =(\d+) -(\d+)')

MIN_RANGE = 20  # con menos rango, un paso no llega a cambiar el entero
FINAL_STEP_DIVISOR = 20  # el desplazamiento final es esta fraccion del rango
ALPHA = 0.602
GAMMA = 0.101
STABILITY = 0.1  # fraccion de las iteraciones que frena los primeros pasos
FINAL_RATE = 0.004
HASH_MB = 16
SAVE_PERIOD = 25
REPORT_PERIOD = 100


@dataclass(frozen=True)
class Param:
    name: str
    default: int
    low: int
    high: int

    @property
    def final_step(self) -> float:
        return (self.high - self.low) / FINAL_STEP_DIVISOR


def load_params() -> list[Param]:
    text = PARAMS_FILE.read_text(encoding='utf-8')
    params = [Param(name, int(value), int(low), int(high))
              for name, value, low, high in PARAM_LINE.findall(text)]
    return [param for param in params if param.high - param.low >= MIN_RANGE]


def schedule(iteration: int, total: int) -> tuple[float, float]:
    """Factores del desplazamiento y de la tasa respecto a sus valores finales."""
    stability = STABILITY * total
    shift = (total / (iteration + 1)) ** GAMMA
    rate = ((stability + total) / (stability + iteration + 1)) ** ALPHA
    return shift, rate


def shifted(param: Param, value: float, offset: float) -> int:
    return min(param.high, max(param.low, round(value + offset)))


def options_text(params: list[Param], values: dict[str, float], offsets: dict[str, float]) -> str:
    settings = [f'{param.name}={shifted(param, values[param.name], offsets[param.name])}'
                for param in params]
    return ';'.join([f'Hash={HASH_MB}', *settings])


def play(args: argparse.Namespace, options1: str, options2: str, seed: int) -> tuple[int, int, int]:
    command = [args.match, '--engine1', args.engine, '--engine2', args.engine,
               '--opt1', options1, '--opt2', options2, '--games', str(args.games),
               '--concurrency', '1', '--nodes', str(args.nodes), '--seed', str(seed)]
    run = subprocess.run(command, capture_output=True, text=True, errors='replace', check=False)
    results = RESULT_LINE.findall(run.stdout)
    if run.returncode != 0 or not results:
        raise RuntimeError(f'match fallo con codigo {run.returncode}:\n{run.stdout[-800:]}'
                           f'\n{run.stderr[-800:]}')
    wins, draws, losses = results[-1]
    return int(wins), int(draws), int(losses)


class Tuner:
    def __init__(self, args: argparse.Namespace, params: list[Param]) -> None:
        self.args = args
        self.params = params
        self.lock = threading.Lock()
        self.failure: str | None = None
        self.state = self.load_state()
        self.started = self.state['iteration']
        self.begin = time.monotonic()

    def load_state(self) -> dict:
        path = pathlib.Path(self.args.state)
        if not path.exists():
            return {'iteration': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                    'values': {param.name: float(param.default) for param in self.params}}
        state = json.loads(path.read_text(encoding='utf-8'))
        if set(state.get('values', {})) != {param.name for param in self.params}:
            raise ValueError(f'{path} no tiene los mismos parametros que {PARAMS_FILE}')
        return state

    def save_state(self) -> None:
        path = pathlib.Path(self.args.state)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=1), encoding='utf-8')
        os.replace(temporary, path)

    def next_job(self) -> tuple[int, dict[str, float]] | None:
        with self.lock:
            if self.failure is not None or self.started >= self.args.iterations:
                return None
            iteration = self.started
            self.started += 1
            return iteration, dict(self.state['values'])

    def apply(self, iteration: int, signs: dict[str, int], result: tuple[int, int, int]) -> None:
        wins, draws, losses = result
        shift, rate = schedule(iteration, self.args.iterations)
        with self.lock:
            for param in self.params:
                # La tasa va en unidades del paso: a / c, con a = FINAL_RATE * c_final^2.
                step = FINAL_RATE * param.final_step * rate / shift
                value = self.state['values'][param.name] + step * (wins - losses) * signs[param.name]
                self.state['values'][param.name] = min(param.high, max(param.low, value))
            self.state['iteration'] += 1
            self.state['wins'] += wins
            self.state['draws'] += draws
            self.state['losses'] += losses
            done = self.state['iteration']
            if done % SAVE_PERIOD == 0:
                self.save_state()
            if done % REPORT_PERIOD == 0:
                self.report()

    def report(self) -> None:
        minutes = (time.monotonic() - self.begin) / 60
        state = self.state
        moved = sorted(self.params, key=lambda param: -abs(
            state['values'][param.name] - param.default) / (param.high - param.low))[:6]
        changes = ' '.join(f'{param.name}:{param.default}->{state["values"][param.name]:.1f}'
                           for param in moved)
        print(f'{state["iteration"]}/{self.args.iterations} | {minutes:.1f} min | '
              f'+{state["wins"]} ={state["draws"]} -{state["losses"]} | {changes}', flush=True)

    def work(self, seed_base: int) -> None:
        rng = random.Random(seed_base)
        while (job := self.next_job()) is not None:
            iteration, values = job
            shift, _ = schedule(iteration, self.args.iterations)
            signs = {param.name: rng.choice((-1, 1)) for param in self.params}
            plus = {p.name: signs[p.name] * p.final_step * shift for p in self.params}
            minus = {name: -offset for name, offset in plus.items()}
            try:
                result = play(self.args, options_text(self.params, values, plus),
                              options_text(self.params, values, minus),
                              self.args.seed + iteration)
                self.apply(iteration, signs, result)
            except (RuntimeError, OSError) as error:
                with self.lock:
                    self.failure = str(error)
                return

    def join(self, workers: list[threading.Thread]) -> None:
        try:
            for worker in workers:
                worker.join()
        except KeyboardInterrupt:
            with self.lock:
                self.failure = 'Interrumpido: se guarda el progreso'
            for worker in workers:
                worker.join()

    def run(self) -> int:
        workers = [threading.Thread(target=self.work, args=(self.args.seed * 1000 + i,))
                   for i in range(self.args.workers)]
        for worker in workers:
            worker.start()
        self.join(workers)
        self.save_state()
        if self.failure is not None:
            print(self.failure, file=sys.stderr)
            return 1
        self.report()
        for param in self.params:
            value = round(self.state['values'][param.name])
            if value != param.default:
                print(f'{param.name} {param.default} -> {value}')
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Ajuste de parametros de la busqueda con SPSA')
    parser.add_argument('--engine', required=True, help='motor compilado con AYANO_TUNE')
    parser.add_argument('--match', required=True, help='herramienta match')
    parser.add_argument('--state', required=True, help='archivo JSON con el progreso')
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--games', type=int, default=8, help='partidas por iteracion, pares')
    parser.add_argument('--nodes', type=int, default=30000, help='nodos por jugada')
    parser.add_argument('--workers', type=int, default=3, help='matches a la vez')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    args.engine = str(pathlib.Path(args.engine).resolve())
    args.match = str(pathlib.Path(args.match).resolve())
    if args.iterations < 1 or args.workers < 1 or args.nodes < 1:
        parser.error('--iterations, --workers y --nodes deben ser positivos')
    if args.games < 2 or args.games % 2:
        parser.error('--games debe ser par: cada apertura se juega con los dos colores')
    return args


def main() -> int:
    args = parse_args()
    params = load_params()
    if not params:
        print(f'No hay parametros en {PARAMS_FILE}', file=sys.stderr)
        return 1
    print(f'{len(params)} parametros, {args.iterations} iteraciones de {args.games} partidas',
          flush=True)
    try:
        tuner = Tuner(args, params)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    return tuner.run()


if __name__ == '__main__':
    sys.exit(main())

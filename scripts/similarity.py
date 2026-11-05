"""Busca secuencias compartidas entre nuestro codigo y el de otros motores.

Uso: python -I similarity.py <carpetas propias separadas por comas> <carpeta ajena> [...]

Compara de tres formas: tokens tal cual, para copias literales; identificadores sustituidos,
para copias con nombres cambiados; y solo numeros, para tablas y constantes, que tambien
detecta coincidencias con codigo en otro lenguaje. Nunca imprime el codigo ajeno: solo
longitudes, nombres de archivo y nuestras propias lineas.
"""
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TOKEN = re.compile(r'''
    /\*.*?\*/ | //[^\n]* |
    "(?:\\.|[^"\\\n])*" | '(?:\\.|[^'\\\n])*' |
    [A-Za-z_][A-Za-z0-9_]* |
    0[xX][0-9a-fA-F'_]+(?:[uUiI](?:8|16|32|64|size)|[uUlL]*) |
    \d[\d.'_]*(?:[eE][+-]?\d+)?(?:[a-zA-Z][a-zA-Z0-9]*)? |
    <<=|>>=|\.\.\.|::|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^]= |
    [{}()\[\];,<>=+\-*/%&|^~!?:.]
''', re.X | re.S)

KEYWORDS = set('''
    if else for while do switch case default break continue return goto const constexpr
    static inline auto void bool int char short long unsigned signed float double struct
    class enum namespace template typename using public private protected virtual override
    true false nullptr this new delete sizeof noexcept
'''.split())

SUFFIXES = {'.cpp', '.h', '.hpp', '.c', '.cc', '.rs'}


def tokenize(path):
    """Devuelve dos listas paralelas, tokens y lineas, sin comentarios ni preprocesador."""
    text = path.read_text(encoding='utf-8', errors='replace')
    text = re.sub(r'^[ \t]*#.*(?:\\\n.*)*$', lambda m: '\n' * m.group(0).count('\n'), text,
                  flags=re.M)
    tokens = []
    lines = []
    line = 1
    position = 0
    for match in TOKEN.finditer(text):
        line += text.count('\n', position, match.start())
        position = match.start()
        value = match.group(0)
        if value.startswith(('//', '/*')):
            continue
        tokens.append(value)
        lines.append(line)
    return tokens, lines


def normalized(tokens):
    result = []
    for token in tokens:
        if token[0] in '"\'':
            result.append('S')
        elif (token[0].isalpha() or token[0] == '_') and token not in KEYWORDS:
            result.append('I')
        else:
            result.append(token)
    return result


def number_value(token):
    text = re.sub(r"['_]", '', token).lower()
    suffixes = 'u|l|ul|ull|ll|i8|i16|i32|i64|u8|u16|u32|u64|usize|isize'
    if not text.startswith('0x'):  # en hexadecimal, la f es una cifra
        suffixes += '|f'
    return re.sub(rf'(?<=[0-9a-f])({suffixes})$', '', text)


def numbers_with_lines(tokens, lines):
    pairs = [(number_value(token), line) for token, line in zip(tokens, lines)
             if token[0].isdigit()]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def load(root):
    root = pathlib.Path(root).resolve()
    files = []
    for path in sorted(root.rglob('*')):
        if path.suffix in SUFFIXES and path.is_file():
            tokens, lines = tokenize(path)
            files.append((f'{root.name}/{path.relative_to(root).as_posix()}', tokens, lines))
    return files


def build_index(streams, size):
    index = {}
    for file_id, stream in enumerate(streams):
        for i in range(len(stream) - size + 1):
            index.setdefault(hash(tuple(stream[i:i + size])), []).append((file_id, i))
    return index


def longest_match(ours, start, theirs, candidates, size):
    best_length = 0
    best_file = None
    for file_id, position in candidates[:64]:
        stream = theirs[file_id]
        if ours[start:start + size] != stream[position:position + size]:
            continue
        length = size
        while (start + length < len(ours) and position + length < len(stream)
               and ours[start + length] == stream[position + length]):
            length += 1
        if length > best_length:
            best_length = length
            best_file = file_id
    return best_length, best_file


def interesting_numbers(window):
    """Descarta rellenos y series triviales, que aparecen en cualquier programa."""
    if len(set(window)) < 5:
        return False
    try:
        values = [int(value, 0) for value in window]
    except ValueError:
        return True
    gaps = {b - a for a, b in zip(values, values[1:])}
    return len(gaps) > 2


def compare(own, foreign, size, make_stream, label, top, keep=None):
    theirs = [make_stream(tokens, lines)[0] for _name, tokens, lines in foreign]
    index = build_index(theirs, size)
    runs = []
    total = 0
    covered = 0
    for name, tokens, lines in own:
        stream, stream_lines = make_stream(tokens, lines)
        total += len(stream)
        i = 0
        while i <= len(stream) - size:
            candidates = index.get(hash(tuple(stream[i:i + size])))
            length = 0
            if candidates:
                length, file_id = longest_match(stream, i, theirs, candidates, size)
            if not length or (keep and not keep(stream[i:i + length])):
                i += 1
                continue
            runs.append((length, name, stream_lines[i], stream_lines[i + length - 1],
                         foreign[file_id][0]))
            covered += length
            i += length
    runs.sort(key=lambda run: -run[0])
    print(f'\n== {label}: ventana de {size}, {len(runs)} coincidencias, '
          f'{100 * covered / max(1, total):.2f} % de {total} propios')
    for length, name, first, last, other in runs[:top]:
        print(f'{length:5}  {name}:{first}-{last}  <->  {other}')


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    own = [item for root in sys.argv[1].split(',') for item in load(root)]
    foreign = [item for root in sys.argv[2:] for item in load(root)]
    print(f'Propios: {len(own)} archivos. Ajenos: {len(foreign)} archivos, '
          f'{sum(len(item[1]) for item in foreign)} tokens.')
    compare(own, foreign, 24, lambda tokens, lines: (tokens, lines), 'Tokens exactos', 40)
    compare(own, foreign, 60, lambda tokens, lines: (normalized(tokens), lines),
            'Nombres sustituidos', 25)
    compare(own, foreign, 8, numbers_with_lines, 'Solo numeros', 30, interesting_numbers)


main()

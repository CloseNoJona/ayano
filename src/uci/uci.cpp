#include "uci/uci.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <iostream>

#include "board/movegen.h"
#include "board/perft.h"
#include "core/attacks.h"
#include "nnue/network.h"
#include "search/params.h"
#include "uci/bench.h"

namespace ayano {

namespace {

constexpr const char* kEngineName = "Ayano 0.11.0";
constexpr const char* kEngineAuthor = "Jonathan Garcia y Claude";
constexpr int kMaxHashMb = 65536;
constexpr int kMaxThreads = 256;
constexpr int kMaxMultiPv = 256;
constexpr int kMaxOverheadMs = 5000;
constexpr int kMaxPerftDepth = 12;
constexpr std::int64_t kMinLimit = -1'000'000'000'000LL;
constexpr std::int64_t kMaxLimit = 1'000'000'000'000'000LL;
constexpr std::string_view kUtf8Bom = "\xEF\xBB\xBF";

// Convierte texto a entero y comprueba que este dentro del rango.
std::optional<std::int64_t> parse_int(const std::string& text, std::int64_t low,
                                      std::int64_t high) {
  std::int64_t value = 0;
  const char* end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(text.data(), end, value);
  if (ec != std::errc() || ptr != end || value < low || value > high) return std::nullopt;
  return value;
}

// Une las palabras siguientes hasta `stop`: los nombres de las opciones llevan espacios.
std::string join_words(std::istringstream& args, std::string_view stop) {
  std::string joined;
  for (std::string word; args >> word && word != stop;) {
    if (!joined.empty()) joined.push_back(' ');
    joined += word;
  }
  return joined;
}

std::string score_to_uci(Value score) {
  if (!is_mate_score(score)) return "cp " + std::to_string(score);
  const int plies = score > 0 ? VALUE_MATE - score : -VALUE_MATE - score;
  const int moves = plies > 0 ? (plies + 1) / 2 : plies / 2;
  return "mate " + std::to_string(moves);
}

std::string to_lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return char(std::tolower(c)); });
  return text;
}

}  // namespace

Uci::Uci() : position_(Position::startpos()) {
  engine_.set_listener(this);
  engine_.set_hash_mb(64);
}

void Uci::send(const std::string& text) {
  const std::lock_guard<std::mutex> lock(output_mutex_);
  std::cout << text << std::endl;
}

void Uci::send_error(const std::string& message) { send("info string error: " + message); }

void Uci::on_info(const SearchInfo& info) {
  if (in_bench_) return;
  std::ostringstream out;
  out << "info depth " << info.depth << " seldepth " << info.sel_depth << " multipv "
      << info.multi_pv << " score " << score_to_uci(info.score);
  if (info.lower_bound) out << " lowerbound";
  if (info.upper_bound) out << " upperbound";
  const std::int64_t elapsed = std::max<std::int64_t>(info.time_ms, 1);
  out << " nodes " << info.nodes << " nps " << info.nodes * 1000 / std::uint64_t(elapsed)
      << " hashfull " << info.hashfull << " time " << info.time_ms << " pv";
  for (const Move m : *info.pv) out << ' ' << m.to_uci();
  send(out.str());
}

void Uci::on_best_move(Move best, Move ponder) {
  if (in_bench_) return;
  std::string text = "bestmove " + best.to_uci();
  if (ponder != Move::none()) text += " ponder " + ponder.to_uci();
  send(text);
}

void Uci::stop_search() {
  engine_.stop();
  engine_.wait();
}

void Uci::handle_uci() {
  std::ostringstream out;
  out << "id name " << kEngineName << "\n"
      << "id author " << kEngineAuthor << "\n"
      << "option name Hash type spin default 64 min 1 max " << kMaxHashMb << "\n"
      << "option name Threads type spin default 1 min 1 max " << kMaxThreads << "\n"
      << "option name MultiPV type spin default 1 min 1 max " << kMaxMultiPv << "\n"
      << "option name Move Overhead type spin default 10 min 0 max " << kMaxOverheadMs << "\n"
      << "option name Skill Level type spin default " << kMaxSkillLevel << " min 0 max "
      << kMaxSkillLevel << "\n"
      << "option name Ponder type check default false\n"
      << "option name Use NNUE type check default true\n"
      << "option name EvalFile type string default <empty>\n"
      << "option name Clear Hash type button\n"
      << "uciok";
  send(out.str());
}

void Uci::handle_new_game() {
  stop_search();
  engine_.new_game();
  position_ = Position::startpos();
  game_keys_.clear();
}

void Uci::handle_position(std::istringstream& args) {
  stop_search();

  std::string token;
  args >> token;
  std::optional<Position> pos;
  if (token == "startpos") {
    pos = Position::startpos();
    args >> token;  // consume "moves", si esta
  } else if (token == "fen") {
    std::string fen;
    while (args >> token && token != "moves") fen += token + " ";
    std::string error;
    pos = Position::from_fen(fen, &error);
    if (!pos) {
      send_error("FEN invalido: " + error);
      return;
    }
  } else {
    send_error("se esperaba 'startpos' o 'fen'");
    return;
  }

  std::vector<Key> keys;
  while (args >> token) {
    const Move m = parse_uci_move(*pos, token);
    if (m == Move::none()) {
      send_error("jugada ilegal: " + token);
      return;
    }
    keys.push_back(pos->key());
    pos = pos->after(m);
  }
  position_ = *pos;
  game_keys_ = std::move(keys);
}

void Uci::apply_limit(SearchLimits& limits, const std::string& name, std::int64_t value) {
  // El reloj puede llegar negativo si la interfaz va con retraso: equivale a muy poco tiempo.
  const std::int64_t positive = std::max<std::int64_t>(value, 0);
  if (name == "wtime") limits.time_ms[WHITE] = std::max<std::int64_t>(value, 1);
  if (name == "btime") limits.time_ms[BLACK] = std::max<std::int64_t>(value, 1);
  if (name == "winc") limits.increment_ms[WHITE] = positive;
  if (name == "binc") limits.increment_ms[BLACK] = positive;
  if (name == "movetime") limits.move_time_ms = positive;
  if (name == "movestogo") limits.moves_to_go = int(std::min<std::int64_t>(positive, 1000));
  if (name == "depth") limits.depth = int(std::min<std::int64_t>(positive, MAX_PLY - 1));
  if (name == "nodes") limits.nodes = std::uint64_t(positive);
  if (name == "mate") limits.mate = int(std::min<std::int64_t>(positive, MAX_PLY / 2));
}

SearchLimits Uci::parse_limits(std::istringstream& args) {
  static const std::vector<std::string> kNumericLimits = {
      "wtime", "btime", "winc", "binc", "movetime", "movestogo", "depth", "nodes", "mate"};

  std::vector<std::string> tokens;
  for (std::string token; args >> token;) tokens.push_back(token);

  SearchLimits limits;
  bool malformed = false;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const std::string& token = tokens[i];
    if (token == "infinite") {
      limits.infinite = true;
    } else if (token == "ponder") {
      limits.ponder = true;
    } else if (token == "searchmoves") {
      // La lista termina en el primer texto que no sea una jugada legal.
      while (i + 1 < tokens.size()) {
        const Move m = parse_uci_move(position_, tokens[i + 1]);
        if (m == Move::none()) break;
        limits.search_moves.push_back(m);
        ++i;
      }
    } else if (std::find(kNumericLimits.begin(), kNumericLimits.end(), token) !=
               kNumericLimits.end()) {
      const auto value = i + 1 < tokens.size() ? parse_int(tokens[i + 1], kMinLimit, kMaxLimit)
                                               : std::nullopt;
      if (value) {
        apply_limit(limits, token, *value);
      } else {
        send_error("valor invalido para " + token);
        malformed = true;
      }
      ++i;
    }
  }

  // Una interfaz que espera `bestmove` perderia por tiempo: siempre se responde, aunque
  // sea con una busqueda minima.
  if (malformed) {
    limits = SearchLimits{};
    limits.depth = 1;
  }
  return limits;
}

void Uci::handle_go(std::istringstream& args) {
  stop_search();
  engine_.start(position_, game_keys_, parse_limits(args));
}

void Uci::apply_option(const std::string& name, const std::string& value) {
  const std::string key = to_lower(name);
  const auto read_int = [&](std::int64_t low, std::int64_t high) {
    const auto parsed = parse_int(value, low, high);
    if (!parsed) send_error("valor fuera de rango para " + name + ": " + value);
    return parsed;
  };

#ifdef AYANO_TUNE
  for (const params::TunableParam& param : params::kTunableParams) {
    if (to_lower(std::string(param.name)) != key) continue;
    if (const auto parsed = read_int(param.low, param.high)) {
      *param.value = int(*parsed);
      rebuild_search_tables();
    }
    return;
  }
#endif

  if (key == "hash") {
    if (const auto mb = read_int(1, kMaxHashMb)) {
      if (!engine_.set_hash_mb(std::size_t(*mb))) {
        send_error("no hay memoria para Hash " + value + "; se mantiene el tamano anterior");
      }
    }
  } else if (key == "threads") {
    if (const auto count = read_int(1, kMaxThreads)) {
      if (!engine_.set_threads(int(*count))) {
        send_error("no hay memoria para " + value + " hilos; se mantienen los anteriores");
      }
    }
  } else if (key == "multipv") {
    if (const auto lines = read_int(1, kMaxMultiPv)) engine_.set_multi_pv(int(*lines));
  } else if (key == "move overhead") {
    if (const auto ms = read_int(0, kMaxOverheadMs)) engine_.set_move_overhead(int(*ms));
  } else if (key == "skill level") {
    if (const auto level = read_int(0, kMaxSkillLevel)) engine_.set_skill_level(int(*level));
  } else if (key == "use nnue") {
    engine_.set_use_nnue(to_lower(value) == "true");
  } else if (key == "evalfile") {
    if (value.empty() || value == "<empty>") return;
    std::string error;
    if (nnue::load_network_file(value, &error)) {
      engine_.new_game();  // la tabla hash y las correcciones son de la red anterior
      send("info string red cargada: " + value);
    } else {
      send_error(error);
    }
  } else if (key == "clear hash") {
    engine_.clear_hash();
  } else if (key != "ponder") {
    send_error("opcion desconocida: " + name);
  }
}

void Uci::handle_setoption(std::istringstream& args) {
  stop_search();
  std::string keyword;
  args >> keyword;  // "name"
  const std::string name = join_words(args, "value");
  apply_option(name, join_words(args, {}));
}

void Uci::handle_perft(std::istringstream& args) {
  stop_search();
  std::string text;
  args >> text;
  const auto depth = parse_int(text, 1, kMaxPerftDepth);
  if (!depth) {
    send_error("uso: perft <profundidad 1-12>");
    return;
  }
  const auto start = std::chrono::steady_clock::now();
  const std::uint64_t nodes = perft_divide(position_, int(*depth));
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
  send("Tiempo " + std::to_string(elapsed) + " ms, " +
       std::to_string(nodes * 1000 / std::uint64_t(std::max<std::int64_t>(elapsed, 1))) +
       " nodos/s");
}

void Uci::handle_bench(std::istringstream& args) {
  stop_search();
  std::string text;
  int depth = kDefaultBenchDepth;
  if (args >> text) {
    const auto parsed = parse_int(text, 1, 40);
    if (!parsed) {
      send_error("uso: bench <profundidad 1-40>");
      return;
    }
    depth = int(*parsed);
  }
  in_bench_ = true;
  const BenchResult result = run_bench(engine_, depth);
  in_bench_ = false;
  send("Nodos: " + std::to_string(result.nodes) + "\nNPS: " + std::to_string(result.nps()) +
       "\nTiempo: " + std::to_string(result.time_ms) + " ms");
}

void Uci::handle_eval() {
  stop_search();
  const Value value = engine_.evaluate(position_);
  send("Evaluacion (bando con turno): " + std::to_string(value) + " cp");
}

void Uci::show_board() {
  std::ostringstream out;
  constexpr std::string_view kChars = "PpNnBbRrQqKk";
  for (int rank = 7; rank >= 0; --rank) {
    out << " +---+---+---+---+---+---+---+---+\n";
    for (int file = 0; file < 8; ++file) {
      const Piece pc = position_.piece_on(make_square(file, rank));
      out << " | " << (pc == NO_PIECE ? ' ' : kChars[pc]);
    }
    out << " | " << rank + 1 << "\n";
  }
  out << " +---+---+---+---+---+---+---+---+\n   a   b   c   d   e   f   g   h\n\n"
      << "Fen: " << position_.fen() << "\nClave: " << std::hex << position_.key() << std::dec;
  send(out.str());
}

bool Uci::execute(const std::string& line) {
  std::istringstream args(line);
  std::string command;
  if (!(args >> command)) return true;

  if (command == "quit") {
    stop_search();
    return false;
  }
  // Son parte de UCI, pero Ayano no tiene modo de depuracion ni necesita registro.
  if (command == "debug" || command == "register") return true;

  if (command == "uci") handle_uci();
  else if (command == "isready") send("readyok");
  else if (command == "ucinewgame") handle_new_game();
  else if (command == "position") handle_position(args);
  else if (command == "go") handle_go(args);
  else if (command == "stop") engine_.stop();
  else if (command == "wait") engine_.wait();  // para scripts: espera a que termine `go`
  else if (command == "ponderhit") engine_.ponder_hit();
  else if (command == "setoption") handle_setoption(args);
  else if (command == "perft") handle_perft(args);
  else if (command == "bench") handle_bench(args);
  else if (command == "eval") handle_eval();
  else if (command == "d") show_board();
  else send_error("comando desconocido: " + command);
  return true;
}

void Uci::loop() {
  std::string line;
  bool first_line = true;
  while (std::getline(std::cin, line)) {
    // Algunas consolas anteponen una marca de orden de bytes UTF-8 al primer comando.
    if (first_line && line.starts_with(kUtf8Bom)) line.erase(0, kUtf8Bom.size());
    first_line = false;
    if (!execute(line)) return;
  }
  stop_search();  // la interfaz cerro la entrada estandar
}

}  // namespace ayano

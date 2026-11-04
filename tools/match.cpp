// Enfrenta dos motores UCI y mide la diferencia de fuerza con Elo y SPRT.
//
// Uso: match --engine1 a.exe --engine2 b.exe --games 1000 --concurrency 5 --tc 8+0.08
//            [--opt1 "Use NNUE=false"] [--opt2 "EvalFile=x.nnue"] [--nodes 20000]
//            [--sprt 0 5] [--seed 1]
// Las opciones se separan con ';'. Cada apertura se juega dos veces con colores
// invertidos para cancelar la ventaja de la apertura.
#ifndef _WIN32
#error "match.cpp usa la API de procesos de Windows"
#endif

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "board/movegen.h"
#include "core/attacks.h"
#include "core/prng.h"
#include "core/zobrist.h"
#include "nnue/network.h"
#include "search/engine.h"

using namespace enroque;

namespace {

constexpr int kOpeningPlies = 8;
constexpr Value kMaxOpeningScore = 90;
constexpr int kMateScore = 30000;
constexpr int kWinAdjudicationScore = 900;
constexpr int kWinAdjudicationPlies = 8;
constexpr int kDrawAdjudicationScore = 6;
constexpr int kDrawAdjudicationPlies = 12;
constexpr int kDrawAdjudicationMinPly = 80;
constexpr int kMaxGamePlies = 500;
constexpr std::int64_t kTimeGraceMs = 2000;
constexpr std::int64_t kNodeLimitBudgetMs = 120000;
constexpr std::int64_t kNoMargin = std::numeric_limits<std::int64_t>::max();
constexpr const char* kEngineErrorPrefix = "info string error";
constexpr DWORD kPipeBufferBytes = 1 << 16;  // una partida larga no debe llenar la tuberia
constexpr int kReportEveryGames = 20;
constexpr double kSprtAlpha = 0.05;
constexpr double kSprtBeta = 0.05;

// Proceso hijo con su entrada y salida conectadas a tuberias.
class EngineProcess {
 public:
  explicit EngineProcess(const std::string& path) {
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
    HANDLE child_stdin_read = nullptr;
    HANDLE child_stdout_write = nullptr;
    CreatePipe(&stdout_read_, &child_stdout_write, &attributes, kPipeBufferBytes);
    CreatePipe(&child_stdin_read, &stdin_write_, &attributes, kPipeBufferBytes);
    SetHandleInformation(stdout_read_, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(stdin_write_, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = child_stdin_read;
    startup.hStdOutput = child_stdout_write;
    startup.hStdError = child_stdout_write;

    std::string command = "\"" + path + "\"";
    alive_ = CreateProcessA(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                            nullptr, nullptr, &startup, &process_) != 0;
    CloseHandle(child_stdin_read);
    CloseHandle(child_stdout_write);
    if (alive_) reader_ = std::thread([this] { read_loop(); });
  }

  ~EngineProcess() {
    if (alive_) {
      send("quit");
      if (WaitForSingleObject(process_.hProcess, 2000) != WAIT_OBJECT_0) {
        TerminateProcess(process_.hProcess, 1);
      }
      CloseHandle(process_.hProcess);
      CloseHandle(process_.hThread);
    }
    CloseHandle(stdin_write_);
    stop_reader();
    CloseHandle(stdout_read_);
  }

  EngineProcess(const EngineProcess&) = delete;
  EngineProcess& operator=(const EngineProcess&) = delete;

  bool started() const { return alive_; }

  // True si el proceso cerro su salida, es decir, si ya no esta en marcha.
  bool has_exited() {
    const std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

  void send(const std::string& line) {
    const std::string text = line + "\n";
    DWORD written = 0;
    WriteFile(stdin_write_, text.data(), DWORD(text.size()), &written, nullptr);
  }

  // Espera una linea durante `timeout_ms` como maximo.
  std::optional<std::string> read_line(std::int64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    const bool ready = available_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                           [this] { return !lines_.empty() || closed_; });
    if (!ready || lines_.empty()) return std::nullopt;
    std::string line = std::move(lines_.front());
    lines_.pop_front();
    return line;
  }

  // Lee hasta una linea con `prefix`. Si se pasa `error`, guarda en el el ultimo aviso de
  // error del motor visto por el camino.
  bool wait_for(const std::string& prefix, std::int64_t timeout_ms, std::string* error = nullptr) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
      const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 deadline - std::chrono::steady_clock::now())
                                 .count();
      if (remaining <= 0) return false;
      const auto line = read_line(remaining);
      if (!line) return false;
      if (line->rfind(prefix, 0) == 0) return true;
      if (error && line->rfind(kEngineErrorPrefix, 0) == 0) *error = *line;
    }
  }

 private:
  // Un proceso hijo del motor puede sobrevivirle con la tuberia abierta, y entonces la
  // lectura no terminaria nunca. Se cancela en bucle por si aun no habia empezado.
  void stop_reader() {
    if (!reader_.joinable()) return;
    while (WaitForSingleObject(reader_.native_handle(), 50) == WAIT_TIMEOUT) {
      CancelIoEx(stdout_read_, nullptr);
    }
    reader_.join();
  }

  void read_loop() {
    std::string buffer;
    std::array<char, 4096> chunk{};
    DWORD count = 0;
    while (ReadFile(stdout_read_, chunk.data(), DWORD(chunk.size()), &count, nullptr) && count > 0) {
      buffer.append(chunk.data(), count);
      std::size_t newline = 0;
      while ((newline = buffer.find('\n')) != std::string::npos) {
        std::string line = buffer.substr(0, newline);
        buffer.erase(0, newline + 1);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::lock_guard<std::mutex> lock(mutex_);
        lines_.push_back(std::move(line));
        available_.notify_one();
      }
    }
    const std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    available_.notify_all();
  }

  HANDLE stdin_write_ = nullptr;
  HANDLE stdout_read_ = nullptr;
  PROCESS_INFORMATION process_{};
  bool alive_ = false;
  std::thread reader_;
  std::mutex mutex_;
  std::condition_variable available_;
  std::deque<std::string> lines_;
  bool closed_ = false;
};

struct Config {
  std::string engine1;
  std::string engine2;
  std::string options1;
  std::string options2;
  int games = 100;
  int concurrency = 1;
  std::int64_t base_ms = 8000;
  std::int64_t increment_ms = 80;
  std::uint64_t nodes = 0;
  std::uint64_t seed = 1;
  bool use_sprt = false;
  double elo0 = 0;
  double elo1 = 5;
};

struct Opening {
  std::vector<Move> moves;
};

enum class Outcome { WhiteWins, BlackWins, Draw };

struct GameRecord {
  Outcome outcome = Outcome::Draw;
  bool anomaly = false;  // reloj agotado, motor mudo o caido, jugada ilegal
  bool invalid = false;  // la comparacion ya no sirve y hay que dejar de jugar
  std::string reason;
};

struct Totals {
  std::mutex mutex;
  int wins = 0;  // desde el punto de vista del motor 1
  int losses = 0;
  int draws = 0;
  int anomalies = 0;
  std::int64_t min_margin_ms = kNoMargin;  // lo mas cerca que se ha estado de perder por tiempo
  std::int64_t left_at_min_margin_ms = 0;
  std::atomic<int> next_game{0};
  std::atomic<bool> stop{false};
  std::atomic<bool> failed{false};
};

double elo_from_score(double score) {
  const double clipped = std::clamp(score, 1e-6, 1.0 - 1e-6);
  return -400.0 * std::log10(1.0 / clipped - 1.0);
}

double score_from_elo(double elo) { return 1.0 / (1.0 + std::pow(10.0, -elo / 400.0)); }

struct Stats {
  double elo;
  double margin;
  double llr;
};

Stats compute_stats(int wins, int losses, int draws, double elo0, double elo1) {
  const double games = double(wins + losses + draws);
  const double score = (wins + 0.5 * draws) / games;
  const double variance = (wins * (1.0 - score) * (1.0 - score) +
                           draws * (0.5 - score) * (0.5 - score) + losses * score * score) /
                          games;
  const double deviation = std::sqrt(std::max(variance, 1e-9) / games);
  const double low = elo_from_score(score - 1.96 * deviation);
  const double high = elo_from_score(score + 1.96 * deviation);

  const double s0 = score_from_elo(elo0);
  const double s1 = score_from_elo(elo1);
  const double llr = games * (s1 - s0) * (2.0 * score - s0 - s1) / (2.0 * std::max(variance, 1e-9));
  return {elo_from_score(score), (high - low) / 2.0, llr};
}

std::vector<std::string> split(const std::string& text, char separator) {
  std::vector<std::string> parts;
  std::istringstream stream(text);
  std::string part;
  while (std::getline(stream, part, separator)) {
    if (!part.empty()) parts.push_back(part);
  }
  return parts;
}

// Un error al aplicar una opcion invalida la comparacion: el motor seguiria con otra
// configuracion, por ejemplo con otra red, y el resultado pareceria bueno.
bool initialize_engine(EngineProcess& engine, const std::string& options) {
  if (!engine.started()) return false;
  std::string error;
  engine.send("uci");
  if (!engine.wait_for("uciok", 10000, &error)) return false;
  for (const std::string& option : split(options, ';')) {
    const std::size_t equals = option.find('=');
    if (equals == std::string::npos) {
      std::cout << "ERROR: opcion sin valor: " << option << std::endl;
      return false;
    }
    engine.send("setoption name " + option.substr(0, equals) + " value " +
                option.substr(equals + 1));
  }
  engine.send("isready");
  const bool ready = engine.wait_for("readyok", 30000, &error);
  if (!error.empty()) std::cout << "ERROR del motor: " << error << std::endl;
  return ready && error.empty();
}

std::vector<Opening> generate_openings(int count, std::uint64_t seed) {
  std::vector<Opening> openings;
  Prng rng(seed);
  Engine evaluator;
  evaluator.set_use_nnue(false);
  SearchLimits limits;
  limits.soft_nodes = 15000;
  limits.nodes = 100000;

  while (int(openings.size()) < count) {
    Opening opening;
    Position pos = Position::startpos();
    bool valid = true;
    for (int ply = 0; ply < kOpeningPlies && valid; ++ply) {
      const MoveList moves = legal_moves(pos);
      if (moves.empty()) {
        valid = false;
        break;
      }
      const Move m = moves[rng.below(moves.size())].move;
      opening.moves.push_back(m);
      pos = pos.after(m);
    }
    if (!valid || legal_moves(pos).empty()) continue;
    evaluator.new_game();
    if (std::abs(evaluator.search_blocking(pos, {}, limits).score) > kMaxOpeningScore) continue;
    openings.push_back(std::move(opening));
  }
  return openings;
}

bool is_threefold(const std::vector<Key>& keys, Key key, int rule50) {
  int seen = 0;
  const int size = int(keys.size());
  const int limit = std::min(rule50, size);
  for (int back = 2; back <= limit; back += 2) {
    if (keys[std::size_t(size - back)] == key && ++seen >= 2) return true;
  }
  return false;
}

// Lee la ultima puntuacion anunciada en una linea `info`.
void parse_score(const std::string& line, std::optional<int>& score) {
  const std::size_t at = line.find(" score ");
  if (at == std::string::npos) return;
  std::istringstream stream(line.substr(at + 7));
  std::string kind;
  int value = 0;
  stream >> kind >> value;
  if (kind == "cp") score = value;
  if (kind == "mate") score = value > 0 ? kMateScore : -kMateScore;
}

struct Clock {
  std::array<std::int64_t, COLOR_NB> remaining_ms;
};

std::string go_command(const Config& config, const Clock& clock) {
  if (config.nodes > 0) return "go nodes " + std::to_string(config.nodes);
  std::ostringstream command;
  command << "go wtime " << clock.remaining_ms[WHITE] << " btime " << clock.remaining_ms[BLACK]
          << " winc " << config.increment_ms << " binc " << config.increment_ms;
  return command.str();
}

// Pide una jugada y devuelve su texto, o vacio si el motor no responde a tiempo.
std::string request_move(EngineProcess& engine, std::int64_t budget_ms,
                         std::optional<int>& score) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(budget_ms);
  while (true) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                               deadline - std::chrono::steady_clock::now())
                               .count();
    if (remaining <= 0) return {};
    const auto line = engine.read_line(remaining);
    if (!line) return {};
    if (line->rfind("info", 0) == 0) parse_score(*line, score);
    if (line->rfind("bestmove", 0) == 0) {
      std::istringstream stream(*line);
      std::string word;
      std::string move;
      stream >> word >> move;
      return move;
    }
  }
}

GameRecord finish(Outcome outcome, const std::string& reason, bool anomaly = false) {
  return {outcome, anomaly, false, reason};
}

// Un motor caido, mudo o que juega ilegal deja de ser comparable: seguir solo acumularia
// derrotas falsas. Perder por tiempo, en cambio, forma parte de lo que se mide.
GameRecord invalidate(GameRecord record) {
  record.invalid = true;
  return record;
}

Outcome win_for(Color c) { return c == WHITE ? Outcome::WhiteWins : Outcome::BlackWins; }

std::optional<GameRecord> check_rules(const Position& pos, const std::vector<Key>& keys) {
  if (legal_moves(pos).empty()) {
    if (pos.in_check()) return finish(win_for(~pos.side_to_move()), "mate");
    return finish(Outcome::Draw, "ahogado");
  }
  if (pos.rule50() >= 100) return finish(Outcome::Draw, "50 jugadas");
  if (pos.is_insufficient_material()) return finish(Outcome::Draw, "material insuficiente");
  if (is_threefold(keys, pos.key(), pos.rule50())) return finish(Outcome::Draw, "repeticion");
  return std::nullopt;
}

// Respuesta de un motor y lo que tardo en darla.
struct Reply {
  std::string text;
  std::int64_t spent_ms = 0;
  std::optional<int> score;  // vacia si el motor no llego a anunciar ninguna
};

Reply ask_engine(EngineProcess& engine, const std::string& history, const Config& config,
                 const Clock& clock, Color us) {
  engine.send(history);
  engine.send(go_command(config, clock));
  const std::int64_t budget =
      config.nodes > 0 ? kNodeLimitBudgetMs : clock.remaining_ms[us] + kTimeGraceMs;
  const auto start = std::chrono::steady_clock::now();
  Reply reply;
  reply.text = request_move(engine, budget, reply.score);
  reply.spent_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - start)
                       .count();
  return reply;
}

// Partida perdida por un fallo del motor, con lo necesario para reproducirla.
GameRecord fault(Color us, const std::string& what, const Reply& reply, std::int64_t left_ms,
                 const std::string& history) {
  std::ostringstream text;
  text << what << " | juegan " << (us == WHITE ? "blancas" : "negras") << " | quedaban " << left_ms
       << " ms | gasto " << reply.spent_ms << " ms | " << history;
  return finish(win_for(~us), text.str(), true);
}

// Termina la partida cuando las puntuaciones llevan tiempo siendo concluyentes.
class Adjudicator {
 public:
  std::optional<GameRecord> update(int white_score, int ply) {
    if (std::abs(white_score) >= kWinAdjudicationScore) {
      const int direction = white_score > 0 ? 1 : -1;
      win_streak_ = win_streak_ * direction > 0 ? win_streak_ + direction : direction;
    } else {
      win_streak_ = 0;
    }
    if (std::abs(win_streak_) >= kWinAdjudicationPlies) {
      return finish(win_streak_ > 0 ? Outcome::WhiteWins : Outcome::BlackWins, "adjudicada");
    }
    draw_streak_ = std::abs(white_score) <= kDrawAdjudicationScore ? draw_streak_ + 1 : 0;
    if (ply >= kDrawAdjudicationMinPly && draw_streak_ >= kDrawAdjudicationPlies) {
      return finish(Outcome::Draw, "adjudicada");
    }
    return std::nullopt;
  }

 private:
  int win_streak_ = 0;
  int draw_streak_ = 0;
};

// Posicion en curso, con lo que hace falta para detectar repeticiones y para enviarla.
struct GameState {
  Position pos = Position::startpos();
  std::vector<Key> keys;
  std::string history = "position startpos moves";

  void play(Move m) {
    keys.push_back(pos.key());
    history += " " + m.to_uci();
    pos = pos.after(m);
  }
};

bool start_new_game(EngineProcess& white, EngineProcess& black) {
  for (EngineProcess* engine : {&white, &black}) {
    engine->send("ucinewgame");
    engine->send("isready");
    if (!engine->wait_for("readyok", 30000)) return false;
  }
  return true;
}

void note_margin(Totals& totals, std::int64_t left_ms, std::int64_t spent_ms) {
  const std::lock_guard<std::mutex> lock(totals.mutex);
  if (left_ms - spent_ms >= totals.min_margin_ms) return;
  totals.min_margin_ms = left_ms - spent_ms;
  totals.left_at_min_margin_ms = left_ms;
}

GameRecord play_game(EngineProcess& white, EngineProcess& black, const Opening& opening,
                     const Config& config, Totals& totals) {
  GameState game;
  for (const Move m : opening.moves) game.play(m);

  if (!start_new_game(white, black)) {
    return invalidate(finish(Outcome::Draw, "un motor no responde a isready", true));
  }

  Clock clock{{config.base_ms, config.base_ms}};
  Adjudicator adjudicator;
  for (int ply = 0; ply < kMaxGamePlies; ++ply) {
    if (const auto ended = check_rules(game.pos, game.keys)) return *ended;

    const Color us = game.pos.side_to_move();
    EngineProcess& engine = us == WHITE ? white : black;
    const std::int64_t left_ms = clock.remaining_ms[us];
    const Reply reply = ask_engine(engine, game.history, config, clock, us);

    if (reply.text.empty()) {
      const char* what = engine.has_exited() ? "motor caido" : "sin respuesta";
      return invalidate(fault(us, what, reply, left_ms, game.history));
    }
    if (config.nodes == 0) {
      if (reply.spent_ms > left_ms) {
        return fault(us, "tiempo agotado", reply, left_ms, game.history);
      }
      note_margin(totals, left_ms, reply.spent_ms);
      clock.remaining_ms[us] += config.increment_ms - reply.spent_ms;
    }

    const Move m = parse_uci_move(game.pos, reply.text);
    if (m == Move::none()) {
      return invalidate(fault(us, "jugada ilegal " + reply.text, reply, left_ms, game.history));
    }

    if (reply.score) {
      const int white_score = us == WHITE ? *reply.score : -*reply.score;
      if (const auto ended = adjudicator.update(white_score, ply)) return *ended;
    }

    game.play(m);
  }
  return finish(Outcome::Draw, "limite de jugadas");
}

void report_progress(Totals& totals, const Config& config) {
  const int played = totals.wins + totals.losses + totals.draws;
  const Stats stats = compute_stats(totals.wins, totals.losses, totals.draws, config.elo0,
                                    config.elo1);
  std::cout << "Partidas " << played << " | +" << totals.wins << " =" << totals.draws << " -"
            << totals.losses << " | Elo " << std::fixed << std::setprecision(1) << stats.elo
            << " +/- " << stats.margin;
  if (config.use_sprt) {
    std::cout << " | LLR " << std::setprecision(2) << stats.llr;
  }
  if (totals.anomalies) std::cout << " | anomalias " << totals.anomalies;
  std::cout << std::endl;
}

void record_game(Totals& totals, const Config& config, const GameRecord& record,
                 bool engine1_is_white) {
  const std::lock_guard<std::mutex> lock(totals.mutex);
  if (record.anomaly) {
    ++totals.anomalies;
    std::cout << "AVISO (motor 1 con " << (engine1_is_white ? "blancas" : "negras")
              << "): " << record.reason << std::endl;
  }
  if (record.invalid) {
    totals.failed = true;
    totals.stop = true;
  }
  if (record.outcome == Outcome::Draw) {
    ++totals.draws;
  } else if ((record.outcome == Outcome::WhiteWins) == engine1_is_white) {
    ++totals.wins;
  } else {
    ++totals.losses;
  }

  const int played = totals.wins + totals.losses + totals.draws;
  if (played % kReportEveryGames == 0) report_progress(totals, config);

  if (config.use_sprt && played >= 2 * kReportEveryGames) {
    const double upper = std::log((1.0 - kSprtBeta) / kSprtAlpha);
    const double lower = std::log(kSprtBeta / (1.0 - kSprtAlpha));
    const Stats stats = compute_stats(totals.wins, totals.losses, totals.draws, config.elo0,
                                      config.elo1);
    if (stats.llr >= upper || stats.llr <= lower) {
      std::cout << "SPRT terminado: " << (stats.llr >= upper ? "H1 aceptada" : "H0 aceptada")
                << std::endl;
      totals.stop = true;
    }
  }
}

void run_worker(const Config& config, const std::vector<Opening>& openings, Totals& totals) {
  EngineProcess first(config.engine1);
  EngineProcess second(config.engine2);
  if (!initialize_engine(first, config.options1) || !initialize_engine(second, config.options2)) {
    std::cout << "ERROR: no se pudo iniciar un motor" << std::endl;
    totals.failed = true;
    totals.stop = true;
    return;
  }

  while (!totals.stop.load()) {
    const int game = totals.next_game.fetch_add(1);
    if (game >= config.games) return;
    const Opening& opening = openings[std::size_t(game / 2)];
    const bool engine1_is_white = game % 2 == 0;
    EngineProcess& white = engine1_is_white ? first : second;
    EngineProcess& black = engine1_is_white ? second : first;
    const GameRecord record = play_game(white, black, opening, config, totals);
    record_game(totals, config, record, engine1_is_white);
  }
}

bool parse_time_control(const std::string& text, Config& config) {
  const std::size_t plus = text.find('+');
  if (plus == std::string::npos) return false;
  config.base_ms = std::int64_t(std::stod(text.substr(0, plus)) * 1000.0);
  config.increment_ms = std::int64_t(std::stod(text.substr(plus + 1)) * 1000.0);
  return config.base_ms > 0;
}

bool parse_args(int argc, char** argv, Config& config) {
  for (int i = 1; i < argc; ++i) {
    const std::string name = argv[i];
    if (name == "--sprt" && i + 2 < argc) {
      config.use_sprt = true;
      config.elo0 = std::stod(argv[++i]);
      config.elo1 = std::stod(argv[++i]);
      continue;
    }
    if (i + 1 >= argc) return false;
    const std::string value = argv[++i];
    if (name == "--engine1") config.engine1 = value;
    else if (name == "--engine2") config.engine2 = value;
    else if (name == "--opt1") config.options1 = value;
    else if (name == "--opt2") config.options2 = value;
    else if (name == "--games") config.games = std::stoi(value);
    else if (name == "--concurrency") config.concurrency = std::stoi(value);
    else if (name == "--nodes") config.nodes = std::stoull(value);
    else if (name == "--seed") config.seed = std::stoull(value);
    else if (name == "--tc") {
      if (!parse_time_control(value, config)) return false;
    } else {
      std::cerr << "Opcion desconocida: " << name << "\n";
      return false;
    }
  }
  return !config.engine1.empty() && !config.engine2.empty() && config.games > 0 &&
         config.concurrency > 0;
}

}  // namespace

int main(int argc, char** argv) {
  Config config;
  try {
    if (!parse_args(argc, argv, config)) {
      std::cerr << "Uso: match --engine1 a --engine2 b --games N --concurrency N --tc 8+0.08 "
                   "[--opt1 a=b;c=d] [--opt2 ...] [--nodes N] [--sprt elo0 elo1] [--seed N]\n";
      return 1;
    }
  } catch (const std::exception& error) {
    std::cerr << "Argumento invalido: " << error.what() << "\n";
    return 1;
  }

  attacks::init();
  zobrist::init();

  std::cout << "Generando aperturas..." << std::endl;
  const std::vector<Opening> openings = generate_openings((config.games + 1) / 2, config.seed);

  Totals totals;
  std::vector<std::thread> workers;
  for (int i = 0; i < config.concurrency; ++i) {
    workers.emplace_back(run_worker, std::cref(config), std::cref(openings), std::ref(totals));
  }
  for (std::thread& worker : workers) worker.join();

  const int played = totals.wins + totals.losses + totals.draws;
  std::cout << "--- Resultado final ---" << std::endl;
  if (played > 0) report_progress(totals, config);
  if (totals.min_margin_ms != kNoMargin) {
    std::cout << "Margen minimo de reloj: " << totals.min_margin_ms << " ms (quedaban "
              << totals.left_at_min_margin_ms << " ms)" << std::endl;
  }
  return played > 0 && !totals.failed ? 0 : 1;
}

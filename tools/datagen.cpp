// Genera datos de entrenamiento: el motor juega contra si mismo con un limite de
// nodos, y cada posicion tranquila se guarda con su evaluacion y el resultado final.
//
// Uso: datagen --out data/gen0.bin --threads 10 --nodes 5000 --seed 1
//              [--games N] [--minutes M] [--net nets/x.nnue]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "board/movegen.h"
#include "core/attacks.h"
#include "core/prng.h"
#include "core/zobrist.h"
#include "nnue/network.h"
#include "nnue/training_data.h"
#include "search/engine.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using namespace enroque;
using nnue::GameResult;
using nnue::PackedPosition;

namespace {

constexpr int kRandomPlies = 8;
constexpr Value kMaxOpeningScore = 600;
constexpr Value kWinAdjudication = 2500;
constexpr int kWinAdjudicationPlies = 6;
constexpr Value kDrawAdjudication = 10;
constexpr int kDrawAdjudicationPlies = 12;
constexpr int kDrawAdjudicationMinPly = 70;
constexpr int kMaxGamePlies = 400;
constexpr std::size_t kHashMb = 16;
constexpr Value kMaxRecordedScore = 10000;

struct Config {
  std::string out_path;
  std::string net_path;
  int threads = 1;
  std::uint64_t nodes = 5000;
  std::uint64_t seed = 1;
  std::uint64_t max_games = 0;
  int minutes = 0;
};

struct Progress {
  std::atomic<std::uint64_t> games{0};
  std::atomic<std::uint64_t> positions{0};
  std::atomic<std::uint64_t> white_wins{0};
  std::atomic<std::uint64_t> black_wins{0};
  std::atomic<std::uint64_t> draws{0};
  std::atomic<bool> stop{false};
};

struct Sample {
  Position pos;
  Value score;
};

class SharedWriter {
 public:
  explicit SharedWriter(const std::string& path) : out_(path, std::ios::binary | std::ios::app) {}
  bool ok() const { return bool(out_); }

  void write(const std::vector<PackedPosition>& records) {
    const std::lock_guard<std::mutex> lock(mutex_);
    out_.write(reinterpret_cast<const char*>(records.data()),
               std::streamsize(records.size() * sizeof(PackedPosition)));
    out_.flush();
  }

 private:
  std::ofstream out_;
  std::mutex mutex_;
};

bool is_threefold(const std::vector<Key>& keys, Key key, int rule50) {
  int seen = 0;
  const int size = int(keys.size());
  const int limit = std::min(rule50, size);
  for (int back = 2; back <= limit; back += 2) {
    if (keys[std::size_t(size - back)] == key && ++seen >= 2) return true;
  }
  return false;
}

// Aperturas aleatorias: dan variedad; las muy desequilibradas se descartan.
bool play_random_opening(Prng& rng, Engine& engine, Position& pos, std::vector<Key>& keys) {
  pos = Position::startpos();
  keys.clear();
  const int plies = kRandomPlies + int(rng.below(2));
  for (int ply = 0; ply < plies; ++ply) {
    const MoveList moves = legal_moves(pos);
    if (moves.empty()) return false;
    keys.push_back(pos.key());
    pos = pos.after(moves[rng.below(moves.size())].move);
  }
  if (legal_moves(pos).empty()) return false;

  SearchLimits limits;
  limits.soft_nodes = 2000;
  limits.nodes = 20000;
  const SearchResult check = engine.search_blocking(pos, keys, limits);
  return std::abs(check.score) <= kMaxOpeningScore;
}

// Devuelve el resultado desde el punto de vista de las blancas.
GameResult play_game(Engine& engine, const Config& config, Position pos, std::vector<Key> keys,
                     std::vector<Sample>& samples) {
  SearchLimits limits;
  limits.soft_nodes = config.nodes;
  limits.nodes = config.nodes * 10;

  int win_streak = 0;
  int draw_streak = 0;

  for (int ply = 0; ply < kMaxGamePlies; ++ply) {
    if (pos.rule50() >= 100 || pos.is_insufficient_material() ||
        is_threefold(keys, pos.key(), pos.rule50())) {
      return GameResult::Draw;
    }

    const SearchResult result = engine.search_blocking(pos, keys, limits);
    if (result.best == Move::none()) {
      if (!pos.in_check()) return GameResult::Draw;
      return pos.side_to_move() == WHITE ? GameResult::Loss : GameResult::Win;
    }

    const Value white_score = pos.side_to_move() == WHITE ? result.score : -result.score;
    win_streak = std::abs(white_score) >= kWinAdjudication
                     ? (win_streak == 0 || (white_score > 0) == (win_streak > 0)
                            ? win_streak + (white_score > 0 ? 1 : -1)
                            : (white_score > 0 ? 1 : -1))
                     : 0;
    if (std::abs(win_streak) >= kWinAdjudicationPlies) {
      return win_streak > 0 ? GameResult::Win : GameResult::Loss;
    }
    draw_streak = std::abs(white_score) <= kDrawAdjudication ? draw_streak + 1 : 0;
    if (ply >= kDrawAdjudicationMinPly && draw_streak >= kDrawAdjudicationPlies) {
      return GameResult::Draw;
    }

    // Solo interesan posiciones tranquilas: la red no debe aprender a ver tacticas
    // que la busqueda de quietud resolvera de todos modos.
    const bool quiet = !pos.in_check() && !pos.is_tactical(result.best);
    if (quiet && std::abs(result.score) < kMaxRecordedScore) {
      samples.push_back({pos, result.score});
    }

    keys.push_back(pos.key());
    pos = pos.after(result.best);
  }
  return GameResult::Draw;
}

GameResult relative_result(GameResult white_result, Color stm) {
  if (white_result == GameResult::Draw || stm == WHITE) return white_result;
  return white_result == GameResult::Win ? GameResult::Loss : GameResult::Win;
}

void worker_loop(int id, const Config& config, SharedWriter& writer, Progress& progress) {
  Prng rng(config.seed * 1000003ULL + std::uint64_t(id) * 7919ULL + 1);
  Engine engine;
  engine.set_hash_mb(kHashMb);

  std::vector<Sample> samples;
  std::vector<PackedPosition> records;
  std::vector<Key> keys;
  Position opening;

  while (!progress.stop.load(std::memory_order_relaxed)) {
    engine.new_game();
    if (!play_random_opening(rng, engine, opening, keys)) continue;

    samples.clear();
    engine.new_game();
    const GameResult result = play_game(engine, config, opening, keys, samples);

    records.clear();
    for (const Sample& sample : samples) {
      records.push_back(nnue::pack_position(sample.pos, sample.score,
                                            relative_result(result, sample.pos.side_to_move())));
    }
    if (!records.empty()) writer.write(records);

    progress.positions += records.size();
    if (result == GameResult::Win) ++progress.white_wins;
    else if (result == GameResult::Loss) ++progress.black_wins;
    else ++progress.draws;

    const std::uint64_t games = ++progress.games;
    if (config.max_games > 0 && games >= config.max_games) progress.stop = true;
  }
}

bool parse_args(int argc, char** argv, Config& config) {
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string name = argv[i];
    const std::string value = argv[i + 1];
    if (name == "--out") config.out_path = value;
    else if (name == "--net") config.net_path = value;
    else if (name == "--threads") config.threads = std::stoi(value);
    else if (name == "--nodes") config.nodes = std::stoull(value);
    else if (name == "--seed") config.seed = std::stoull(value);
    else if (name == "--games") config.max_games = std::stoull(value);
    else if (name == "--minutes") config.minutes = std::stoi(value);
    else {
      std::cerr << "Opcion desconocida: " << name << "\n";
      return false;
    }
  }
  return !config.out_path.empty() && config.threads > 0;
}

void report(const Progress& progress, std::int64_t seconds) {
  const std::uint64_t positions = progress.positions.load();
  std::cout << seconds << " s | partidas " << progress.games.load() << " | posiciones " << positions
            << " (" << positions / std::uint64_t(std::max<std::int64_t>(seconds, 1)) << "/s)"
            << " | +" << progress.white_wins.load() << " =" << progress.draws.load() << " -"
            << progress.black_wins.load() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  Config config;
  try {
    if (!parse_args(argc, argv, config)) {
      std::cerr << "Uso: datagen --out archivo --threads N --nodes N --seed N "
                   "[--games N] [--minutes M] [--net red]\n";
      return 1;
    }
  } catch (const std::exception& error) {
    std::cerr << "Argumento invalido: " << error.what() << "\n";
    return 1;
  }

#ifdef _WIN32
  // Prioridad baja: el PC sigue siendo usable mientras se generan datos.
  SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
#endif

  attacks::init();
  zobrist::init();
  if (!config.net_path.empty()) {
    std::string error;
    if (!nnue::load_network_file(config.net_path, &error)) {
      std::cerr << "Error: " << error << "\n";
      return 1;
    }
  }

  SharedWriter writer(config.out_path);
  if (!writer.ok()) {
    std::cerr << "No se pudo abrir " << config.out_path << "\n";
    return 1;
  }

  Progress progress;
  std::vector<std::thread> threads;
  for (int id = 0; id < config.threads; ++id) {
    threads.emplace_back(worker_loop, id, std::cref(config), std::ref(writer), std::ref(progress));
  }

  const auto start = std::chrono::steady_clock::now();
  std::int64_t next_report = 60;
  while (!progress.stop.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    const std::int64_t seconds =
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start)
            .count();
    if (seconds >= next_report) {
      report(progress, seconds);
      next_report += 60;
    }
    if (config.minutes > 0 && seconds >= std::int64_t(config.minutes) * 60) progress.stop = true;
  }

  for (std::thread& thread : threads) thread.join();
  const std::int64_t total =
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start)
          .count();
  report(progress, total);
  return 0;
}

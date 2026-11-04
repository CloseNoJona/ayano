#include "search/engine.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <new>

namespace enroque {

namespace {

// Margen de error permitido por nivel: a menor nivel, peores jugadas son aceptables.
constexpr int kSkillMarginPerLevel = 18;

std::uint64_t time_based_seed() {
  return std::uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
}

}  // namespace

Engine::Engine() : skill_rng_(time_based_seed()) { set_threads(1); }

Engine::~Engine() {
  stop();
  wait();
}

bool Engine::set_threads(int count) {
  wait();
  std::vector<std::unique_ptr<Worker>> created;
  try {
    for (int id = 0; id < count; ++id) {
      created.push_back(std::make_unique<Worker>(id, shared_));
    }
  } catch (const std::bad_alloc&) {
    return false;
  }
  workers_ = std::move(created);
  shared_.workers = workers_;
  return true;
}

bool Engine::set_hash_mb(std::size_t megabytes) {
  wait();
  return shared_.tt.resize(megabytes);
}

void Engine::new_game() {
  wait();
  shared_.tt.clear();
  for (auto& worker : workers_) worker->new_game();
}

void Engine::configure(const Position& root, std::span<const Key> game_keys,
                       const SearchLimits& limits) {
  root_ = root;
  game_keys_.assign(game_keys.begin(), game_keys.end());
  shared_.limits = limits;
  shared_.stop.store(false, std::memory_order_relaxed);
  shared_.pondering.store(limits.ponder, std::memory_order_relaxed);
  shared_.time.start(limits, root.side_to_move(), root.game_ply(), shared_.move_overhead_ms);
  shared_.tt.new_search();
  for (auto& worker : workers_) worker->prepare(root_, game_keys_);
}

void Engine::run_threads() {
  std::vector<std::thread> helpers;
  helpers.reserve(workers_.size() - 1);
  for (std::size_t i = 1; i < workers_.size(); ++i) {
    helpers.emplace_back([worker = workers_[i].get()] { worker->run(); });
  }
  workers_[0]->run();

  // UCI no permite responder mientras se medita o en modo infinito.
  while (!shared_.stop.load(std::memory_order_relaxed) &&
         (shared_.pondering.load(std::memory_order_relaxed) || shared_.limits.infinite)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  shared_.stop.store(true, std::memory_order_relaxed);
  for (std::thread& helper : helpers) helper.join();
}

void Engine::start(const Position& root, std::span<const Key> game_keys,
                   const SearchLimits& limits) {
  wait();
  configure(root, game_keys, limits);
  search_thread_ = std::thread([this] {
    run_threads();
    const SearchResult result = collect_result();
    if (shared_.listener) shared_.listener->on_best_move(result.best, result.ponder);
  });
}

SearchResult Engine::search_blocking(const Position& root, std::span<const Key> game_keys,
                                     const SearchLimits& limits) {
  wait();
  configure(root, game_keys, limits);
  run_threads();
  return collect_result();
}

void Engine::stop() { shared_.stop.store(true, std::memory_order_relaxed); }

void Engine::ponder_hit() { shared_.pondering.store(false, std::memory_order_relaxed); }

void Engine::wait() {
  if (search_thread_.joinable()) search_thread_.join();
}

// Con varios hilos cada uno puede preferir una jugada: votan segun la profundidad
// alcanzada y la puntuacion. Un mate demostrado gana directamente.
const Worker& Engine::pick_best_worker() const {
  const Worker* best = workers_[0].get();
  if (workers_.size() == 1 || shared_.effective_multi_pv() > 1) return *best;

  Value lowest = VALUE_INFINITE;
  for (const auto& worker : workers_) {
    if (worker->has_root_moves()) lowest = std::min(lowest, worker->best_root_move().score);
  }

  std::map<std::uint16_t, std::int64_t> votes;
  for (const auto& worker : workers_) {
    if (!worker->has_root_moves()) continue;
    const RootMove& root_move = worker->best_root_move();
    const std::int64_t weight = std::int64_t(root_move.score - lowest + 14) *
                                std::max(worker->completed_depth(), 1);
    votes[root_move.move.raw()] += weight;
  }

  for (const auto& worker : workers_) {
    if (!worker->has_root_moves()) continue;
    const RootMove& candidate = worker->best_root_move();
    const RootMove& current = best->best_root_move();
    if (candidate.score >= VALUE_MATE_IN_MAX_PLY) {
      if (candidate.score > current.score) best = worker.get();
    } else if (current.score < VALUE_MATE_IN_MAX_PLY &&
               votes[candidate.move.raw()] > votes[current.move.raw()]) {
      best = worker.get();
    }
  }
  return *best;
}

// Elige al azar entre las jugadas analizadas que no sean demasiado peores que la mejor.
Move Engine::pick_with_skill(const Worker& worker) const {
  const std::span<const RootMove> moves = worker.root_moves();
  const std::size_t analyzed =
      std::min(std::size_t(shared_.effective_multi_pv()), moves.size());
  const Value margin = (kMaxSkillLevel - shared_.skill_level) * kSkillMarginPerLevel;
  const Value top = moves[0].score;
  // Un mate a favor se juega siempre: fallarlo seria un error demasiado artificial.
  if (top >= VALUE_MATE_IN_MAX_PLY) return moves[0].move;

  std::size_t acceptable = 1;
  while (acceptable < analyzed && moves[acceptable].score != -VALUE_INFINITE &&
         moves[acceptable].score >= top - margin) {
    ++acceptable;
  }
  return moves[skill_rng_.below(acceptable)].move;
}

SearchResult Engine::collect_result() const {
  SearchResult result;
  result.nodes = shared_.total_nodes();
  if (!workers_[0]->has_root_moves()) return result;

  const Worker& worker = pick_best_worker();
  const RootMove& root_move = worker.best_root_move();
  result.best = root_move.move;
  result.score = root_move.score;
  result.depth = worker.completed_depth();
  if (root_move.pv.size() > 1) result.ponder = root_move.pv[1];
  if (shared_.skill_level < kMaxSkillLevel) {
    result.best = pick_with_skill(worker);
    if (result.best != root_move.move) result.ponder = Move::none();
  }
  return result;
}

Value Engine::evaluate(const Position& pos) {
  wait();
  workers_[0]->prepare(pos, {});
  return workers_[0]->evaluate_root();
}

}  // namespace enroque

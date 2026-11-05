#include "search/worker.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "board/movegen.h"
#include "eval/classical.h"
#include "nnue/network.h"
#include "search/params.h"

namespace ayano {

namespace {

constexpr std::int64_t kReportEveryMs = 3000;

template <typename Array>
void fill_zero(Array& array) {
  std::memset(static_cast<void*>(&array), 0, sizeof(array));
}

}  // namespace

// Con nivel reducido se analizan varias jugadas para poder elegir una peor.
int SearchShared::effective_multi_pv() const {
  constexpr int kSkillCandidates = 4;
  return skill_level < kMaxSkillLevel ? std::max(multi_pv, kSkillCandidates) : multi_pv;
}

int SearchShared::skill_depth_limit() const {
  return skill_level < kMaxSkillLevel ? 1 + skill_level / 2 : MAX_PLY;
}

std::uint64_t SearchShared::total_nodes() const {
  std::uint64_t total = 0;
  for (const auto& worker : workers) total += worker->nodes();
  return total;
}

Worker::Worker(int id, SearchShared& shared)
    : id_(id),
      shared_(shared),
      histories_(std::make_unique<Histories>()),
      accumulators_(std::make_unique<nnue::AccumulatorStack>()),
      stack_(std::size_t(MAX_PLY + kStackOffset + 4)),
      pv_table_(std::size_t(MAX_PLY + 2)) {
  new_game();
}

void Worker::new_game() {
  Histories& h = *histories_;
  fill_zero(h.main);
  fill_zero(h.capture);
  fill_zero(h.pawn);
  fill_zero(h.continuation);
  fill_zero(h.pawn_correction);
  fill_zero(h.minor_correction);
  fill_zero(h.non_pawn_correction);
  fill_zero(h.sentinel);
  for (auto& per_piece : h.counter_moves) per_piece.fill(Move::none());
}

void Worker::prepare(const Position& root, std::span<const Key> game_keys) {
  game_length_ = int(game_keys.size());
  key_history_.assign(game_keys.begin(), game_keys.end());
  key_history_.resize(game_keys.size() + MAX_PLY + 2);

  for (StackEntry& e : stack_) {
    e.cont_hist = &histories_->sentinel;
    e.move = Move::none();
    e.excluded = Move::none();
    e.killers = {Move::none(), Move::none()};
    e.static_eval = VALUE_NONE;
    e.cutoff_count = 0;
    e.in_check = false;
    e.tt_pv = false;
  }
  entry(0)->pos = root;
  accumulators_->reset();

  root_moves_.clear();
  const std::vector<Move>& allowed = shared_.limits.search_moves;
  for (const ExtMove& legal : legal_moves(root)) {
    if (!allowed.empty() && std::find(allowed.begin(), allowed.end(), legal.move) == allowed.end()) {
      continue;
    }
    RootMove root_move;
    root_move.move = legal.move;
    root_move.pv.push_back(legal.move);
    root_moves_.push_back(std::move(root_move));
  }

  nodes_.store(0, std::memory_order_relaxed);
  completed_depth_ = 0;
  nmp_min_ply_ = 0;
  pv_index_ = 0;
  calls_until_check_ = kCheckInterval;
  stable_best_move_ = Move::none();
  best_move_stability_ = 0;
  previous_iteration_score_ = VALUE_NONE;
}

void Worker::run() {
  if (root_moves_.empty()) return;
  iterate_depth();
}

Value Worker::evaluate_root() { return evaluate(0); }

bool Worker::root_contains(Move m) const {
  return std::any_of(root_moves_.begin() + std::ptrdiff_t(pv_index_), root_moves_.end(),
                     [m](const RootMove& root_move) { return root_move.move == m; });
}

void Worker::iterate_depth() {
  const SearchLimits& limits = shared_.limits;
  const std::size_t multi_pv =
      std::min(std::size_t(shared_.effective_multi_pv()), root_moves_.size());

  for (root_depth_ = 1; root_depth_ < MAX_PLY && !stopped(); ++root_depth_) {
    if (limits.depth > 0 && root_depth_ > limits.depth) break;
    if (root_depth_ > shared_.skill_depth_limit()) break;

    for (RootMove& root_move : root_moves_) root_move.previous_score = root_move.score;

    for (pv_index_ = 0; pv_index_ < multi_pv && !stopped(); ++pv_index_) {
      sel_depth_ = 0;
      aspiration_window();
      std::stable_sort(root_moves_.begin(), root_moves_.begin() + std::ptrdiff_t(pv_index_) + 1,
                       [](const RootMove& a, const RootMove& b) { return a.score > b.score; });
    }
    pv_index_ = 0;

    if (stopped()) {
      if (is_main()) report_pv(root_depth_);
      break;
    }

    completed_depth_ = root_depth_;
    if (is_main()) {
      report_pv(root_depth_);
      decide_stop_after_iteration();
    }
    if (limits.mate > 0 && root_moves_[0].score >= mate_in(2 * limits.mate)) break;
  }
}

void Worker::aspiration_window() {
  const Value previous = root_moves_[pv_index_].previous_score;
  Value alpha = -VALUE_INFINITE;
  Value beta = VALUE_INFINITE;
  int delta = params::kAspirationDelta + previous * previous / 16000;

  if (root_depth_ >= params::kAspirationMinDepth && previous != -VALUE_INFINITE) {
    alpha = std::max(previous - delta, -VALUE_INFINITE);
    beta = std::min(previous + delta, VALUE_INFINITE);
  }

  int failed_high_count = 0;
  while (true) {
    const int depth = std::max(1, root_depth_ - failed_high_count);
    const Value value = search<NodeType::Root>(0, alpha, beta, depth, false);

    std::stable_sort(root_moves_.begin() + std::ptrdiff_t(pv_index_), root_moves_.end(),
                     [](const RootMove& a, const RootMove& b) { return a.score > b.score; });

    if (stopped()) return;

    const bool outside_window = value <= alpha || value >= beta;
    if (is_main() && outside_window && shared_.time.elapsed_ms() > kReportEveryMs) {
      report_pv(root_depth_);
    }

    if (value <= alpha) {
      beta = (alpha + beta) / 2;
      alpha = std::max(value - delta, -VALUE_INFINITE);
      failed_high_count = 0;
    } else if (value >= beta) {
      beta = std::min(value + delta, VALUE_INFINITE);
      ++failed_high_count;
    } else {
      return;
    }
    delta += delta / 3;
  }
}

void Worker::report_pv(int depth) {
  if (!shared_.listener) return;
  const std::size_t lines =
      std::min(std::size_t(shared_.effective_multi_pv()), root_moves_.size());
  for (std::size_t i = 0; i < lines; ++i) {
    const RootMove& root_move = root_moves_[i];
    if (root_move.score == -VALUE_INFINITE) continue;  // aun sin resultado en esta iteracion
    SearchInfo info;
    info.depth = depth;
    info.sel_depth = root_move.sel_depth;
    info.multi_pv = int(i) + 1;
    info.score = root_move.score;
    info.lower_bound = root_move.lower_bound;
    info.upper_bound = root_move.upper_bound;
    info.nodes = shared_.total_nodes();
    info.time_ms = shared_.time.elapsed_ms();
    info.hashfull = shared_.tt.hashfull();
    info.pv = &root_move.pv;
    shared_.listener->on_info(info);
  }
}

void Worker::check_limits() {
  // Se estima con el contador propio: es barato y permite comprobar en cada nodo.
  const SearchLimits& limits = shared_.limits;
  if (limits.nodes > 0 && nodes() * shared_.workers.size() >= limits.nodes) {
    shared_.stop.store(true, std::memory_order_relaxed);
    return;
  }

  if (--calls_until_check_ > 0) return;
  calls_until_check_ = kCheckInterval;

  if (shared_.pondering.load(std::memory_order_relaxed)) return;
  if (shared_.time.has_limit() && shared_.time.elapsed_ms() >= shared_.time.maximum_ms()) {
    shared_.stop.store(true, std::memory_order_relaxed);
  }
}

// Decide si empezar otra iteracion: una mejor jugada estable y dominante necesita
// menos tiempo; un cambio de jugada o una caida de la evaluacion, mas.
void Worker::decide_stop_after_iteration() {
  const SearchLimits& limits = shared_.limits;
  const RootMove& best = root_moves_[0];

  if (best.move == stable_best_move_) {
    ++best_move_stability_;
  } else {
    stable_best_move_ = best.move;
    best_move_stability_ = 0;
  }

  if (limits.soft_nodes > 0 && shared_.total_nodes() >= limits.soft_nodes) {
    shared_.stop.store(true, std::memory_order_relaxed);
    return;
  }
  if (!shared_.time.has_limit()) return;

  // Un mate demostrado no mejora pensando mas: basta con haber buscado mas
  // profundo que la distancia al mate.
  if (is_mate_score(best.score) && root_depth_ > 2 * (VALUE_MATE - std::abs(best.score)) + 2 &&
      !shared_.pondering.load(std::memory_order_relaxed)) {
    shared_.stop.store(true, std::memory_order_relaxed);
    return;
  }
  if (limits.move_time_ms > 0) return;

  const double stability = 1.35 - 0.07 * std::min(best_move_stability_, 8);
  const double total = double(std::max<std::uint64_t>(nodes(), 1));
  const double best_fraction = double(best.nodes) / total;
  const double effort = (1.55 - best_fraction) * 1.25;

  double falling = 1.0;
  if (previous_iteration_score_ != VALUE_NONE) {
    falling = std::clamp(1.0 + double(previous_iteration_score_ - best.score) / 120.0, 0.85, 1.5);
  }
  previous_iteration_score_ = best.score;

  double scale = stability * effort * falling;
  if (root_moves_.size() == 1) scale = 0.15;

  const double budget = double(shared_.time.optimum_ms()) * scale;
  if (double(shared_.time.elapsed_ms()) >= budget) {
    if (shared_.pondering.load(std::memory_order_relaxed)) return;
    shared_.stop.store(true, std::memory_order_relaxed);
  }
}

bool Worker::is_repetition(const Position& pos, int ply) const {
  const int index = game_length_ + ply;
  const int limit = std::min(pos.rule50(), index);
  const Key key = pos.key();
  int previous_occurrences = 0;
  for (int back = 4; back <= limit; back += 2) {
    if (key_history_[std::size_t(index - back)] != key) continue;
    if (back < ply) return true;  // dentro del arbol basta con una repeticion
    if (++previous_occurrences >= 2) return true;
  }
  return false;
}

bool Worker::is_draw(const Position& pos, int ply) const {
  if (pos.rule50() >= 100) return !pos.in_check() || !legal_moves(pos).empty();
  return pos.is_insufficient_material() || is_repetition(pos, ply);
}

Value Worker::evaluate(int ply) {
  const Position& pos = entry(ply)->pos;
  const nnue::Network* network = shared_.use_nnue ? nnue::active_network() : nullptr;
  Value value = VALUE_ZERO;
  if (network) {
    value = accumulators_->evaluate(*network, ply,
                                    [this](int p) -> const StackEntry& { return *entry(p); });
#ifdef AYANO_VERIFY_NNUE
    // Diagnostico: el camino incremental debe coincidir siempre con el recalculo.
    if (value != nnue::evaluate_from_scratch(*network, pos)) {
      std::cerr << "NNUE desincronizada en ply " << ply << ": " << pos.fen() << std::endl;
      std::abort();
    }
#endif
  } else {
    value = classical::evaluate(pos);
  }
  // Acercarse a la regla de las 50 jugadas hace menos valiosa la ventaja.
  value = value * (200 - pos.rule50()) / 200;
  return std::clamp(value, VALUE_MATED_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);
}

}  // namespace ayano

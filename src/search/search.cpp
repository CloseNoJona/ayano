// Busqueda alfa-beta con ventana nula (PVS) y las podas modernas. Cada nodo sigue
// siempre el mismo camino: entrada, tabla hash, evaluacion estatica, podas antes de
// generar jugadas, bucle de jugadas y actualizacion de estadisticas.
#include <algorithm>
#include <array>
#include <cmath>

#include "search/node.h"
#include "search/params.h"
#include "search/worker.h"

namespace enroque {

namespace {

constexpr int kLmrUnit = 1024;

struct LmrTable {
  std::array<std::array<int, 64>, 64> values{};

  LmrTable() { rebuild(); }

  void rebuild() {
    const double base = params::kLmrBase / 100.0;
    const double divisor = params::kLmrDivisor / 100.0;
    for (int depth = 1; depth < 64; ++depth) {
      for (int moves = 1; moves < 64; ++moves) {
        const double plies = base + std::log(depth) * std::log(moves) / divisor;
        values[depth][moves] = int(plies * kLmrUnit);
      }
    }
  }
};

LmrTable lmr_table;

}  // namespace

void Worker::make_move(StackEntry* ss, int ply, Move m, bool capture) {
  ss->move = m;
  ss->cont_hist =
      &histories_->continuation[ss->in_check][capture][ss->pos.moved_piece(m)][m.to()];
  ss->pos.apply_to(m, ss[1].pos);
  shared_.tt.prefetch(ss[1].pos.key());
  accumulators_->on_move(ply + 1);
}

void Worker::make_null_move(StackEntry* ss, int ply) {
  ss->move = Move::null();
  ss->cont_hist = &histories_->sentinel;
  ss->pos.apply_null_to(ss[1].pos);
  accumulators_->on_null_move(ply + 1);
}

void Worker::update_pv(int ply, Move m) {
  auto& line = pv_table_[std::size_t(ply)];
  const auto& child = pv_table_[std::size_t(ply) + 1];
  const int child_length = pv_length_[std::size_t(ply) + 1];
  line[0] = m;
  std::copy_n(child.begin(), child_length, line.begin() + 1);
  pv_length_[std::size_t(ply)] = child_length + 1;
}

ContinuationRefs Worker::continuation_refs(const StackEntry* ss) const {
  return {ss[-1].cont_hist, ss[-2].cont_hist, ss[-4].cont_hist, ss[-6].cont_hist};
}

Value Worker::enter_node(Node& n) {
  const Position& pos = n.pos;
  StackEntry* ss = n.ss;

  count_node();
  if (is_main()) check_limits();
  if (n.pv_node) {
    pv_length_[std::size_t(n.ply)] = 0;
    sel_depth_ = std::max(sel_depth_, n.ply + 1);
  }

  n.in_check = ss->in_check = pos.in_check();
  key_history_[std::size_t(game_length_ + n.ply)] = pos.key();

  if (!n.root_node) {
    if (stopped()) return VALUE_ZERO;
    if (is_draw(pos, n.ply)) return draw_value();
    if (n.ply >= MAX_PLY - 1) return n.in_check ? VALUE_ZERO : evaluate(n.ply);

    // Un mate mas corto ya encontrado hace inutil seguir por esta rama.
    n.alpha = std::max(mated_in(n.ply), n.alpha);
    n.beta = std::min(mate_in(n.ply + 1), n.beta);
    if (n.alpha >= n.beta) return n.alpha;
  }

  ss[1].excluded = Move::none();
  ss[1].cutoff_count = 0;
  ss[2].killers = {Move::none(), Move::none()};
  return VALUE_NONE;
}

void Worker::probe_tt(Node& n) {
  const Position& pos = n.pos;
  StackEntry* ss = n.ss;

  n.excluded = ss->excluded;
  n.key = pos.key();
  n.tt_hit = n.excluded == Move::none() && shared_.tt.probe(n.key, n.tt);
  n.tt_value = n.tt_hit ? value_from_tt(n.tt.value, n.ply) : VALUE_NONE;
  if (n.root_node) {
    n.tt_move = root_moves_[pv_index_].pv[0];
  } else {
    n.tt_move = n.tt_hit ? n.tt.move : Move::none();
  }
  n.tt_capture = n.tt_move != Move::none() && pos.is_tactical(n.tt_move);
  if (n.excluded == Move::none()) ss->tt_pv = n.pv_node || (n.tt_hit && n.tt.pv);
}

bool Worker::can_cut_with_tt(const Node& n) const {
  return !n.pv_node && n.excluded == Move::none() && n.tt_hit && n.tt.depth >= n.depth &&
         n.tt_value != VALUE_NONE &&
         has_bound(n.tt.bound, n.tt_value >= n.beta ? Bound::Lower : Bound::Upper) &&
         n.pos.rule50() < 90;
}

void Worker::compute_static_eval(Node& n) {
  StackEntry* ss = n.ss;
  if (n.in_check) {
    ss->static_eval = n.eval = VALUE_NONE;
    n.improving = false;
    return;
  }

  if (n.excluded != Move::none()) {
    n.eval = ss->static_eval;  // ya calculada por el mismo nodo antes de la exclusion
  } else {
    const bool stored = n.tt_hit && n.tt.eval != VALUE_NONE;
    n.raw_eval = stored ? n.tt.eval : evaluate(n.ply);
    ss->static_eval = n.eval = corrected_eval(n.pos, n.raw_eval);
    if (!n.tt_hit) {
      shared_.tt.store(n.key, VALUE_NONE, n.raw_eval, -1, Bound::None, Move::none(), ss->tt_pv);
    }
    // El valor de la tabla hash viene de una busqueda: suele ser mejor estimacion.
    if (n.tt_hit && n.tt_value != VALUE_NONE &&
        has_bound(n.tt.bound, n.tt_value > n.eval ? Bound::Lower : Bound::Upper)) {
      n.eval = n.tt_value;
    }
    learn_from_eval_swing(n);
  }

  const Value two_back = ss[-2].static_eval;
  const Value four_back = ss[-4].static_eval;
  if (two_back != VALUE_NONE) {
    n.improving = ss->static_eval > two_back;
  } else {
    n.improving = four_back == VALUE_NONE || ss->static_eval > four_back;
  }
}

// Si la evaluacion salto tras una jugada tranquila, esa jugada era mejor o peor
// de lo que indicaba el historial: se ajusta sin esperar a un corte.
void Worker::learn_from_eval_swing(const Node& n) {
  const StackEntry* ss = n.ss;
  const Move prior = ss[-1].move;
  if (!prior.is_ok() || ss[-1].in_check || n.pos.captured_piece() != NO_PIECE) return;
  if (ss[-1].static_eval == VALUE_NONE) return;
  const int swing = -8 * (ss[-1].static_eval + ss->static_eval);
  update_history<kMainHistoryLimit>(histories_->main[~n.pos.side_to_move()][prior.from_to()],
                                    std::clamp(swing, -1000, 1000));
}

Value Worker::null_move_search(Node& n) {
  const int reduction =
      params::kNullMoveBase + n.depth / params::kNullMoveDepthDivisor +
      std::min((n.eval - n.beta) / params::kNullMoveEvalDivisor, params::kNullMoveMaxEvalReduction);

  make_null_move(n.ss, n.ply);
  const Value value =
      -search<NodeType::NonPv>(n.ply + 1, -n.beta, -n.beta + 1, n.depth - reduction, !n.cut_node);

  if (value < n.beta || is_mate_score(value)) return VALUE_NONE;
  if (nmp_min_ply_ > 0 || n.depth < params::kNullMoveVerifyDepth) return value;

  // A gran profundidad se verifica para no caer en zugzwang: durante la
  // verificacion no se permite otra jugada nula hasta bien entrado el arbol.
  nmp_min_ply_ = n.ply + 3 * (n.depth - reduction) / 4;
  const Value verified =
      search<NodeType::NonPv>(n.ply, n.beta - 1, n.beta, n.depth - reduction, false);
  nmp_min_ply_ = 0;
  return verified >= n.beta ? value : VALUE_NONE;
}

Value Worker::forward_prune(Node& n) {
  if (n.pv_node || n.in_check || n.excluded != Move::none()) return VALUE_NONE;

  const Position& pos = n.pos;
  const StackEntry* ss = n.ss;

  // Razoring: muy por debajo de alfa, se confirma con la busqueda de quietud.
  if (n.depth <= params::kRazorMaxDepth && n.eval + params::kRazorMargin * n.depth <= n.alpha) {
    const Value value = qsearch<false>(n.ply, n.alpha, n.alpha + 1);
    if (value <= n.alpha) return value;
  }

  // Poda de futilidad inversa: tan por encima de beta que se da por bueno.
  const int rfp_margin =
      params::kRfpMargin * n.depth - params::kRfpImprovingBonus * int(n.improving);
  if (n.depth <= params::kRfpMaxDepth && n.eval - rfp_margin >= n.beta &&
      !is_mate_score(n.eval) && !is_mate_score(n.beta)) {
    return (n.eval + n.beta) / 2;
  }

  // Jugada nula: si pasando el turno seguimos por encima de beta, se corta.
  if (n.depth >= params::kNullMoveMinDepth && ss[-1].move != Move::null() && n.eval >= n.beta &&
      ss->static_eval >= n.beta - 20 * n.depth + 180 &&
      pos.has_non_pawn_material(pos.side_to_move()) && n.ply >= nmp_min_ply_ &&
      !is_mate_score(n.beta)) {
    const Value value = null_move_search(n);
    if (value != VALUE_NONE) return value;
  }
  return VALUE_NONE;
}

// ProbCut: si una captura buena supera beta con margen en una busqueda reducida,
// lo mas probable es que tambien lo haga a profundidad completa.
Value Worker::probcut(Node& n) {
  if (n.pv_node || n.in_check || n.excluded != Move::none()) return VALUE_NONE;
  const int improving_bonus = params::kProbCutImprovingBonus * int(n.improving);
  const Value probcut_beta = n.beta + params::kProbCutMargin - improving_bonus;
  if (n.depth < params::kProbCutMinDepth || is_mate_score(n.beta)) return VALUE_NONE;
  if (n.tt_hit && n.tt.depth >= n.depth - 3 && n.tt_value != VALUE_NONE &&
      n.tt_value < probcut_beta) {
    return VALUE_NONE;
  }

  const Position& pos = n.pos;
  StackEntry* ss = n.ss;
  MovePicker picker(pos, n.tt_capture ? n.tt_move : Move::none(), probcut_beta - ss->static_eval,
                    *histories_);
  Move m = Move::none();
  while ((m = picker.next()) != Move::none()) {
    if (!pos.is_legal(m)) continue;
    make_move(ss, n.ply, m, true);
    Value value = -qsearch<false>(n.ply + 1, -probcut_beta, -probcut_beta + 1);
    if (value >= probcut_beta) {
      value = -search<NodeType::NonPv>(n.ply + 1, -probcut_beta, -probcut_beta + 1, n.depth - 4,
                                       !n.cut_node);
    }
    if (stopped()) return VALUE_ZERO;
    if (value >= probcut_beta) {
      shared_.tt.store(n.key, value_to_tt(value, n.ply), n.raw_eval, n.depth - 3, Bound::Lower, m,
                       ss->tt_pv);
      return value - (probcut_beta - n.beta);
    }
  }
  return VALUE_NONE;
}

bool Worker::prune_move(Node& n, Candidate& c, bool& skip_quiets) {
  const Position& pos = n.pos;
  const StackEntry* ss = n.ss;
  if (n.root_node || n.best_value <= VALUE_MATED_IN_MAX_PLY ||
      !pos.has_non_pawn_material(pos.side_to_move())) {
    return false;
  }

  // Tras muchas jugadas sin exito, las tranquilas restantes rara vez sirven.
  const int lmp_limit = (params::kLmpBase + n.depth * n.depth) / (2 - int(n.improving));
  if (n.move_count >= lmp_limit) skip_quiets = true;

  int lmr_depth = c.new_depth - lmr_reduction(n, c) / kLmrUnit;

  if (c.capture || c.gives_check) {
    const PieceType victim = pos.captured_type(c.move);
    const int capture_history = histories_->capture[c.piece][c.to][victim];
    if (!c.gives_check && !n.in_check && lmr_depth < params::kCaptureFutilityMaxDepth) {
      const Value futility = ss->static_eval + params::kCaptureFutilityBase +
                             params::kCaptureFutilityPerDepth * lmr_depth + see_value(victim) +
                             capture_history / params::kCaptureHistoryDivisor;
      if (futility <= n.alpha) return true;
    }
    return !pos.see_ge(c.move, -params::kSeeCaptureMargin * n.depth);
  }

  const ContinuationRefs refs = continuation_refs(ss);
  int history = (*refs[0])[c.piece][c.to] + (*refs[1])[c.piece][c.to] +
                histories_->pawn[pos.pawn_key() % kPawnHistorySize][c.piece][c.to];
  if (lmr_depth <= params::kHistoryPruneMaxDepth &&
      history < -params::kHistoryPruneMargin * n.depth) {
    return true;
  }

  history += 2 * histories_->main[pos.side_to_move()][c.move.from_to()];
  lmr_depth += history / params::kHistoryToLmrDepth;

  if (!n.in_check && lmr_depth < params::kFutilityMaxDepth) {
    const Value futility =
        ss->static_eval + params::kFutilityBase + params::kFutilityPerDepth * lmr_depth;
    if (futility <= n.alpha) {
      if (n.best_value < futility && !is_mate_score(n.best_value)) n.best_value = futility;
      return true;
    }
  }

  lmr_depth = std::max(lmr_depth, 0);
  return !pos.see_ge(c.move, -params::kSeeQuietMargin * lmr_depth * lmr_depth);
}

int Worker::extension(Node& n, const Candidate& c) {
  StackEntry* ss = n.ss;
  if (n.ply >= 2 * root_depth_) return 0;

  const bool singular_candidate =
      !n.root_node && n.depth >= params::kSingularMinDepth + int(ss->tt_pv) &&
      c.move == n.tt_move && n.excluded == Move::none() && n.tt_value != VALUE_NONE &&
      !is_mate_score(n.tt_value) && has_bound(n.tt.bound, Bound::Lower) &&
      n.tt.depth >= n.depth - params::kSingularTtDepthMargin;

  if (singular_candidate) {
    const Value singular_beta = n.tt_value - n.depth;
    ss->excluded = c.move;
    const Value value = search<NodeType::NonPv>(n.ply, singular_beta - 1, singular_beta,
                                                (n.depth - 1) / 2, n.cut_node);
    ss->excluded = Move::none();

    if (value < singular_beta) {
      if (n.pv_node) return 1;
      if (value < singular_beta - params::kTripleExtensionMargin && !c.capture) return 3;
      if (value < singular_beta - params::kDoubleExtensionMargin) return 2;
      return 1;
    }
    // Varias jugadas superan beta: nos fiamos del corte sin mas busqueda.
    if (singular_beta >= n.beta) {
      n.early_return = singular_beta;
      return 0;
    }
    if (n.tt_value >= n.beta) return -2;
    if (n.cut_node) return -1;
    return 0;
  }

  return c.gives_check && n.depth >= 6 && n.pv_node ? 1 : 0;
}

int Worker::lmr_reduction(const Node& n, const Candidate& c) const {
  const int depth = std::min(n.depth, 63);
  const int moves = std::min(n.move_count, 63);
  int r = lmr_table.values[depth][moves];
  if (!n.improving) r += kLmrUnit / 2;
  if (c.capture) r -= kLmrUnit / 2;
  return r;
}

// Calcula la reduccion final de una jugada segun lo que se sabe del nodo.
int Worker::adjusted_reduction(const Node& n, const Candidate& c) {
  StackEntry* ss = n.ss;
  const ContinuationRefs refs = continuation_refs(ss);
  const int stat_score =
      c.capture ? 2 * histories_->capture[c.piece][c.to][n.pos.captured_type(c.move)]
                : 2 * histories_->main[n.pos.side_to_move()][c.move.from_to()] +
                      (*refs[0])[c.piece][c.to] + (*refs[1])[c.piece][c.to] +
                      (*refs[2])[c.piece][c.to];

  int r = lmr_reduction(n, c);
  if (n.pv_node) r -= kLmrUnit;
  if (ss->tt_pv) r -= kLmrUnit;
  if (n.cut_node) r += 3 * kLmrUnit / 2;
  if (n.tt_capture) r += kLmrUnit / 2;
  if (ss[1].cutoff_count > 3) r += kLmrUnit;
  if (c.move == n.tt_move) r -= kLmrUnit;
  if (c.gives_check) r -= kLmrUnit / 2;
  return r - stat_score * kLmrUnit / params::kLmrStatScoreDivisor;
}

// Busca la jugada a profundidad reducida y repite si sorprende por encima de alfa.
Value Worker::reduced_search(Node& n, const Candidate& c, int& new_depth) {
  const Value alpha = n.alpha;
  const int ply = n.ply + 1;
  const int r = adjusted_reduction(n, c);
  const int reduced = std::clamp(new_depth - r / kLmrUnit, 1, std::max(1, new_depth + 1));
  Value value = -search<NodeType::NonPv>(ply, -(alpha + 1), -alpha, reduced, true);

  if (value > alpha && reduced < new_depth) {
    const bool deeper = value > n.best_value + params::kLmrDeeperMargin + 2 * new_depth;
    const bool shallower = value < n.best_value + params::kLmrShallowerMargin;
    new_depth += int(deeper) - int(shallower);
    if (reduced < new_depth) {
      value = -search<NodeType::NonPv>(ply, -(alpha + 1), -alpha, new_depth, !n.cut_node);
    }
    if (!c.capture) {
      const int bonus = value <= alpha ? -history_malus(new_depth) : history_bonus(new_depth);
      update_continuation(n.ss, c.piece, c.to, bonus);
    }
  }
  return value;
}

template <NodeType Type>
Value Worker::search_candidate(Node& n, const Candidate& c, int extended_depth) {
  constexpr bool kPv = Type != NodeType::NonPv;
  constexpr bool kRoot = Type == NodeType::Root;
  const Value alpha = n.alpha;
  const Value beta = n.beta;
  const int ply = n.ply + 1;
  int new_depth = extended_depth;
  Value value = -VALUE_INFINITE;

  if (n.depth >= 2 && n.move_count > 1 + int(kRoot)) {
    value = reduced_search(n, c, new_depth);
  } else if (!kPv || n.move_count > 1) {
    value = -search<NodeType::NonPv>(ply, -(alpha + 1), -alpha, new_depth, !n.cut_node);
  }

  if (kPv && (n.move_count == 1 || (value > alpha && (kRoot || value < beta)))) {
    pv_length_[std::size_t(ply)] = 0;
    value = -search<NodeType::Pv>(ply, -beta, -alpha, new_depth, false);
  }
  return value;
}

void Worker::update_root_move(Node& n, Move m, Value value) {
  RootMove& root_move = *std::find_if(root_moves_.begin(), root_moves_.end(),
                                      [m](const RootMove& rm) { return rm.move == m; });

  if (n.move_count == 1 || value > n.alpha) {
    root_move.score = value;
    root_move.sel_depth = sel_depth_;
    root_move.lower_bound = value >= n.beta;
    root_move.upper_bound = value <= n.alpha;
    root_move.pv.resize(1);
    const auto& line = pv_table_[1];
    root_move.pv.insert(root_move.pv.end(), line.begin(), line.begin() + pv_length_[1]);
  } else {
    root_move.score = -VALUE_INFINITE;  // solo importa el orden relativo
  }
}

void Worker::record_result(Node& n, const Candidate& c, Value value) {
  StackEntry* ss = n.ss;

  if (value > n.best_value) {
    n.best_value = value;
    if (value > n.alpha) {
      n.best_move = c.move;
      if (n.pv_node && !n.root_node) update_pv(n.ply, c.move);
      if (value >= n.beta) {
        ss->cutoff_count += 1 + int(n.tt_move == Move::none());
        return;
      }
      if (n.depth > 2 && n.depth < 14 && !is_mate_score(value)) n.depth -= 1;
      n.alpha = value;
    }
  }

  if (c.move != n.best_move) {
    if (c.capture && n.capture_count < kMaxTried) {
      n.captures_tried[std::size_t(n.capture_count++)] = c.move;
    } else if (!c.capture && n.quiet_count < kMaxTried) {
      n.quiets_tried[std::size_t(n.quiet_count++)] = c.move;
    }
  }
}

template <NodeType Type>
Value Worker::move_loop(Node& n) {
  constexpr bool kRoot = Type == NodeType::Root;
  const Position& pos = n.pos;
  StackEntry* ss = n.ss;

  const ContinuationRefs refs = continuation_refs(ss);
  const Move prior = ss[-1].move;
  const Move counter = prior.is_ok()
                           ? histories_->counter_moves[pos.piece_on(prior.to())][prior.to()]
                           : Move::none();
  MovePicker picker(pos, n.tt_move, *histories_, refs, {ss->killers[0], ss->killers[1], counter});

  bool skip_quiets = false;
  Move m = Move::none();
  while ((m = picker.next(skip_quiets)) != Move::none()) {
    if (m == n.excluded) continue;
    if (kRoot && !root_contains(m)) continue;
    if (!pos.is_legal(m)) continue;
    ++n.move_count;

    Candidate c{m, pos.moved_piece(m), m.to(), pos.is_tactical(m), pos.gives_check(m), n.depth - 1};
    if (prune_move(n, c, skip_quiets)) continue;

    const int extended = c.new_depth + extension(n, c);
    if (n.early_return != VALUE_NONE) return n.early_return;

    const std::uint64_t nodes_before = nodes();
    make_move(ss, n.ply, m, c.capture);
    const Value value = search_candidate<Type>(n, c, extended);

    if (stopped()) return VALUE_ZERO;
    if (kRoot) {
      update_root_move(n, m, value);
      auto root_move = std::find_if(root_moves_.begin(), root_moves_.end(),
                                    [m](const RootMove& rm) { return rm.move == m; });
      root_move->nodes += nodes() - nodes_before;
    }
    record_result(n, c, value);
    if (n.best_value >= n.beta) break;
  }
  return n.best_value;
}

Value Worker::finish_node(Node& n) {
  const Position& pos = n.pos;
  StackEntry* ss = n.ss;

  if (n.move_count == 0) {
    if (n.excluded != Move::none()) return n.alpha;
    return n.in_check ? mated_in(n.ply) : VALUE_DRAW;
  }

  if (n.best_move != Move::none()) {
    update_cutoff_stats(n);
  } else {
    update_prior_move_stats(n);
  }

  if (n.excluded != Move::none()) return n.best_value;

  // En MultiPV las lineas secundarias no son el valor real de la raiz.
  if (!(n.root_node && pv_index_ > 0)) {
    Bound bound = Bound::Upper;
    if (n.best_value >= n.beta) {
      bound = Bound::Lower;
    } else if (n.pv_node && n.best_move != Move::none()) {
      bound = Bound::Exact;
    }
    shared_.tt.store(n.key, value_to_tt(n.best_value, n.ply), n.raw_eval, n.depth, bound,
                     n.best_move, ss->tt_pv);

    const bool tactical_best = n.best_move != Move::none() && pos.is_tactical(n.best_move);
    const bool bound_disagrees = (bound == Bound::Lower && n.best_value <= ss->static_eval) ||
                                 (bound == Bound::Upper && n.best_value >= ss->static_eval);
    if (!n.in_check && !tactical_best && !bound_disagrees && !is_mate_score(n.best_value)) {
      update_correction(pos, n.depth, n.best_value - ss->static_eval);
    }
  }
  return n.best_value;
}

template <NodeType Type>
Value Worker::search(int ply, Value alpha, Value beta, int depth, bool cut_node) {
  constexpr bool kPv = Type != NodeType::NonPv;
  constexpr bool kRoot = Type == NodeType::Root;

  if (depth <= 0) return qsearch<kPv>(ply, alpha, beta);

  StackEntry* ss = entry(ply);
  Node n{ss, ss->pos, ply, std::min(depth, MAX_PLY - 1), alpha, beta, kPv, kRoot, cut_node};

  const Value early = enter_node(n);
  if (early != VALUE_NONE) return early;

  probe_tt(n);
  if (can_cut_with_tt(n)) {
    if (n.tt_value >= n.beta && n.tt_move != Move::none() && !n.tt_capture &&
        n.pos.is_pseudo_legal(n.tt_move)) {
      update_quiet_history(n.pos, ss, n.tt_move, history_bonus(n.depth));
    }
    return n.tt_value;
  }

  compute_static_eval(n);

  const Value pruned = forward_prune(n);
  if (pruned != VALUE_NONE) return pruned;

  // Sin jugada de la tabla hash el orden sera malo: se busca menos profundo y la
  // proxima visita ya tendra una jugada con la que empezar.
  if ((kPv || n.cut_node) && n.depth >= params::kIirMinDepth && n.tt_move == Move::none() &&
      n.excluded == Move::none()) {
    n.depth -= 1;
  }

  const Value cut = probcut(n);
  if (cut != VALUE_NONE) return cut;

  const Value loop_result = move_loop<Type>(n);
  if (n.early_return != VALUE_NONE) return n.early_return;
  if (stopped()) return loop_result;
  return finish_node(n);
}

template Value Worker::search<NodeType::Root>(int, Value, Value, int, bool);

void rebuild_search_tables() { lmr_table.rebuild(); }

}  // namespace enroque

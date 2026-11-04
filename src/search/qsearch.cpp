// Busqueda de quietud: resuelve las capturas pendientes para que la evaluacion se
// aplique solo a posiciones tranquilas.
#include <algorithm>

#include "search/node.h"
#include "search/params.h"
#include "search/worker.h"

namespace enroque {

struct QNode {
  StackEntry* ss;
  const Position& pos;
  int ply;
  Value alpha;
  Value beta;
  bool pv;

  bool in_check = false;
  bool tt_hit = false;
  bool tt_pv = false;
  TTData tt{};
  Value tt_value = VALUE_NONE;
  Move tt_move = Move::none();
  Value raw_eval = VALUE_NONE;
  Value best_value = -VALUE_INFINITE;
  Value futility_base = -VALUE_INFINITE;
  Move best_move = Move::none();
  int move_count = 0;
};

Value Worker::qsearch_enter(QNode& q) {
  count_node();
  if (is_main()) check_limits();
  if (q.pv) {
    pv_length_[std::size_t(q.ply)] = 0;
    sel_depth_ = std::max(sel_depth_, q.ply + 1);
  }

  q.in_check = q.ss->in_check = q.pos.in_check();
  key_history_[std::size_t(game_length_ + q.ply)] = q.pos.key();

  if (stopped()) return VALUE_ZERO;
  if (is_draw(q.pos, q.ply)) return draw_value();
  if (q.ply >= MAX_PLY - 1) return q.in_check ? VALUE_ZERO : evaluate(q.ply);
  return VALUE_NONE;
}

void Worker::qsearch_probe(QNode& q) {
  q.tt_hit = shared_.tt.probe(q.pos.key(), q.tt);
  q.tt_value = q.tt_hit ? value_from_tt(q.tt.value, q.ply) : VALUE_NONE;
  q.tt_move = q.tt_hit ? q.tt.move : Move::none();
  q.tt_pv = q.pv || (q.tt_hit && q.tt.pv);
}

// Valor de quedarse quieto. En jaque no existe esa opcion: hay que responder.
Value Worker::qsearch_stand_pat(QNode& q) {
  StackEntry* ss = q.ss;
  if (q.in_check) {
    ss->static_eval = VALUE_NONE;
    return VALUE_NONE;
  }

  q.raw_eval = q.tt_hit && q.tt.eval != VALUE_NONE ? q.tt.eval : evaluate(q.ply);
  ss->static_eval = q.best_value = corrected_eval(q.pos, q.raw_eval);
  if (q.tt_hit && q.tt_value != VALUE_NONE &&
      has_bound(q.tt.bound, q.tt_value > q.best_value ? Bound::Lower : Bound::Upper)) {
    q.best_value = q.tt_value;
  }

  if (q.best_value >= q.beta) {
    if (!q.tt_hit) {
      shared_.tt.store(q.pos.key(), value_to_tt(q.best_value, q.ply), q.raw_eval, -1, Bound::Lower,
                       Move::none(), q.tt_pv);
    }
    return is_mate_score(q.best_value) ? q.best_value : (q.best_value + q.beta) / 2;
  }

  q.alpha = std::max(q.alpha, q.best_value);
  q.futility_base = ss->static_eval + params::kQsFutilityMargin;
  return VALUE_NONE;
}

bool Worker::qsearch_prune(QNode& q, Move m) {
  if (q.best_value <= VALUE_MATED_IN_MAX_PLY || q.in_check) return false;

  // Una promocion vale mucho mas de lo que mide `futility`: nunca se poda aqui.
  if (!q.pos.gives_check(m) && m.kind() != MoveKind::Promotion &&
      q.futility_base > -VALUE_INFINITE) {
    if (q.move_count > 2) return true;
    const Value futility = q.futility_base + see_value(q.pos.captured_type(m));
    if (futility <= q.alpha) {
      q.best_value = std::max(q.best_value, futility);
      return true;
    }
    if (q.futility_base <= q.alpha && !q.pos.see_ge(m, 1)) {
      q.best_value = std::max(q.best_value, q.futility_base);
      return true;
    }
  }
  return !q.pos.see_ge(m, params::kQsSeeMargin);
}

template <bool Pv>
Value Worker::qsearch(int ply, Value alpha, Value beta) {
  StackEntry* ss = entry(ply);
  QNode q{ss, ss->pos, ply, alpha, beta, Pv};

  Value early = qsearch_enter(q);
  if (early != VALUE_NONE) return early;

  qsearch_probe(q);
  if (!Pv && q.tt_hit && q.tt_value != VALUE_NONE &&
      has_bound(q.tt.bound, q.tt_value >= q.beta ? Bound::Lower : Bound::Upper)) {
    return q.tt_value;
  }

  early = qsearch_stand_pat(q);
  if (early != VALUE_NONE) return early;

  const ContinuationRefs refs = continuation_refs(ss);
  MovePicker picker(q.pos, q.tt_move, *histories_, refs);
  Move m = Move::none();

  while ((m = picker.next()) != Move::none()) {
    if (!q.pos.is_legal(m)) continue;
    ++q.move_count;
    if (qsearch_prune(q, m)) continue;

    make_move(ss, ply, m, q.pos.is_tactical(m));
    const Value value = -qsearch<Pv>(ply + 1, -q.beta, -q.alpha);
    if (stopped()) return VALUE_ZERO;

    if (value > q.best_value) {
      q.best_value = value;
      if (value > q.alpha) {
        q.best_move = m;
        if (Pv) update_pv(ply, m);
        if (value >= q.beta) break;
        q.alpha = value;
      }
    }
  }

  if (q.in_check && q.move_count == 0) return mated_in(ply);

  const Bound bound = q.best_value >= q.beta ? Bound::Lower : Bound::Upper;
  shared_.tt.store(q.pos.key(), value_to_tt(q.best_value, ply), q.raw_eval, 0, bound, q.best_move,
                   q.tt_pv);
  return q.best_value;
}

template Value Worker::qsearch<true>(int, Value, Value);
template Value Worker::qsearch<false>(int, Value, Value);

}  // namespace enroque

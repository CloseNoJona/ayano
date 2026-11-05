// Actualizacion de los historiales y correccion de la evaluacion.
#include <algorithm>

#include "search/node.h"
#include "search/params.h"
#include "search/worker.h"

namespace ayano {

Value Worker::corrected_eval(const Position& pos, Value raw) const {
  const Color us = pos.side_to_move();
  const Histories& h = *histories_;
  const int pawn = h.pawn_correction[us][pos.pawn_key() % kCorrectionSize];
  const int minor = h.minor_correction[us][pos.minor_key() % kCorrectionSize];
  const int white = h.non_pawn_correction[WHITE][us][pos.non_pawn_key(WHITE) % kCorrectionSize];
  const int black = h.non_pawn_correction[BLACK][us][pos.non_pawn_key(BLACK) % kCorrectionSize];
  const int correction = params::kCorrectionPawnWeight * pawn +
                         params::kCorrectionMinorWeight * minor +
                         params::kCorrectionNonPawnWeight * (white + black);
  const Value value = raw + correction / (params::kCorrectionDivisor * 8);
  return std::clamp(value, VALUE_MATED_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);
}

void Worker::update_correction(const Position& pos, int depth, Value difference) {
  const Color us = pos.side_to_move();
  Histories& h = *histories_;
  const int bonus = std::clamp(difference * depth / 8, -params::kCorrectionScale,
                               params::kCorrectionScale);
  update_history<kCorrectionLimit>(h.pawn_correction[us][pos.pawn_key() % kCorrectionSize], bonus);
  update_history<kCorrectionLimit>(h.minor_correction[us][pos.minor_key() % kCorrectionSize],
                                   bonus);
  update_history<kCorrectionLimit>(
      h.non_pawn_correction[WHITE][us][pos.non_pawn_key(WHITE) % kCorrectionSize], bonus);
  update_history<kCorrectionLimit>(
      h.non_pawn_correction[BLACK][us][pos.non_pawn_key(BLACK) % kCorrectionSize], bonus);
}

void Worker::update_continuation(StackEntry* ss, Piece pc, Square to, int bonus) {
  for (const int back : {1, 2, 4, 6}) {
    if (ss->in_check && back > 2) break;
    if (ss[-back].move.is_ok()) {
      update_history<kContinuationLimit>((*ss[-back].cont_hist)[pc][to], bonus);
    }
  }
}

void Worker::update_quiet_history(const Position& pos, StackEntry* ss, Move m, int bonus) {
  const Piece pc = pos.moved_piece(m);
  Histories& h = *histories_;
  update_history<kMainHistoryLimit>(h.main[pos.side_to_move()][m.from_to()], bonus);
  update_history<kPawnHistoryLimit>(h.pawn[pos.pawn_key() % kPawnHistorySize][pc][m.to()], bonus);
  update_continuation(ss, pc, m.to(), bonus);
}

void Worker::update_capture_history(const Position& pos, Move m, int bonus) {
  update_history<kCaptureHistoryLimit>(
      histories_->capture[pos.moved_piece(m)][m.to()][pos.captured_type(m)], bonus);
}

void Worker::set_refutations(const Position& pos, StackEntry* ss, Move best) {
  if (ss->killers[0] != best) {
    ss->killers[1] = ss->killers[0];
    ss->killers[0] = best;
  }
  const Move previous = ss[-1].move;
  if (previous.is_ok()) {
    histories_->counter_moves[pos.piece_on(previous.to())][previous.to()] = best;
  }
}

void Worker::update_cutoff_stats(Node& n) {
  const Position& pos = n.pos;
  const int bonus = history_bonus(n.depth);
  const int malus = history_malus(n.depth);

  if (pos.is_tactical(n.best_move)) {
    update_capture_history(pos, n.best_move, bonus);
  } else {
    set_refutations(pos, n.ss, n.best_move);
    update_quiet_history(pos, n.ss, n.best_move, bonus);
    for (int i = 0; i < n.quiet_count; ++i) {
      update_quiet_history(pos, n.ss, n.quiets_tried[std::size_t(i)], -malus);
    }
  }
  for (int i = 0; i < n.capture_count; ++i) {
    update_capture_history(pos, n.captures_tried[std::size_t(i)], -malus);
  }
}

// Si ninguna jugada mejoro alfa, la jugada anterior del rival fue buena.
void Worker::update_prior_move_stats(Node& n) {
  const StackEntry* ss = n.ss;
  const Move prior = ss[-1].move;
  if (!prior.is_ok() || n.pos.captured_piece() != NO_PIECE) return;
  const int bonus = history_bonus(n.depth);
  update_continuation(n.ss - 1, n.pos.piece_on(prior.to()), prior.to(), bonus);
  update_history<kMainHistoryLimit>(histories_->main[~n.pos.side_to_move()][prior.from_to()],
                                    bonus / 2);
}

}  // namespace ayano

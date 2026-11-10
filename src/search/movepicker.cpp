#include "search/movepicker.h"

namespace ayano {

namespace {

// Una captura durante una evasion se prueba antes que cualquier jugada tranquila,
// cuyo valor de historial queda muy por debajo.
constexpr int kEvasionCaptureBase = 1 << 24;
constexpr int kCaptureVictimFactor = 7;
constexpr int kGoodCaptureDivisor = 18;
constexpr int kThreatEscapeBonus = 8000;

// Casillas atacadas por piezas rivales de menor valor que la pieza indicada.
struct Threats {
  Bitboard by_pawn;
  Bitboard by_minor;
  Bitboard by_rook;

  Bitboard against(PieceType pt) const {
    switch (pt) {
      case KNIGHT:
      case BISHOP: return by_pawn;
      case ROOK: return by_minor;
      case QUEEN: return by_rook;
      default: return 0;
    }
  }
};

Bitboard attacks_of(const Position& pos, Color c, PieceType pt) {
  Bitboard result = 0;
  Bitboard pieces = pos.pieces(c, pt);
  while (pieces) result |= attacks::of(pt, pop_lsb(pieces), pos.pieces());
  return result;
}

Threats compute_threats(const Position& pos, Color them) {
  const Bitboard pawns = pos.pieces(them, PAWN);
  Threats threats{};
  threats.by_pawn = them == WHITE ? shift<NORTH_EAST>(pawns) | shift<NORTH_WEST>(pawns)
                                  : shift<SOUTH_EAST>(pawns) | shift<SOUTH_WEST>(pawns);
  threats.by_minor =
      threats.by_pawn | attacks_of(pos, them, KNIGHT) | attacks_of(pos, them, BISHOP);
  threats.by_rook = threats.by_minor | attacks_of(pos, them, ROOK);
  return threats;
}

}  // namespace

MovePicker::MovePicker(const Position& pos, Move tt_move, const Histories& histories,
                       const ContinuationRefs& continuation, const Refutations& refutations,
                       Bitboard threats)
    : pos_(pos),
      histories_(histories),
      continuation_(&continuation),
      refutations_(refutations),
      tt_move_(Move::none()),
      threshold_(0),
      threats_(threats) {
  const bool usable = tt_move != Move::none() && pos.is_pseudo_legal(tt_move);
  if (usable) tt_move_ = tt_move;
  if (pos.in_check()) {
    stage_ = usable ? Stage::EvasionTT : Stage::EvasionInit;
  } else {
    stage_ = usable ? Stage::MainTT : Stage::CaptureInit;
  }
}

MovePicker::MovePicker(const Position& pos, Move tt_move, const Histories& histories,
                       const ContinuationRefs& continuation, Bitboard threats)
    : pos_(pos),
      histories_(histories),
      continuation_(&continuation),
      refutations_{Move::none(), Move::none(), Move::none()},
      tt_move_(Move::none()),
      threshold_(0),
      threats_(threats) {
  const bool in_check = pos.in_check();
  // Sin jaque solo se admiten jugadas tacticas: la quietud debe terminar.
  const bool usable = tt_move != Move::none() && pos.is_pseudo_legal(tt_move) &&
                      (in_check || pos.is_tactical(tt_move));
  if (usable) tt_move_ = tt_move;
  if (in_check) {
    stage_ = usable ? Stage::EvasionTT : Stage::EvasionInit;
  } else {
    stage_ = usable ? Stage::QuiescenceTT : Stage::QuiescenceInit;
  }
}

MovePicker::MovePicker(const Position& pos, Move tt_move, Value threshold,
                       const Histories& histories)
    : pos_(pos),
      histories_(histories),
      continuation_(nullptr),
      refutations_{Move::none(), Move::none(), Move::none()},
      tt_move_(Move::none()),
      threshold_(threshold),
      threats_(0) {
  const bool usable = tt_move != Move::none() && pos.is_pseudo_legal(tt_move) &&
                      pos.is_capture(tt_move) && pos.see_ge(tt_move, threshold);
  if (usable) tt_move_ = tt_move;
  stage_ = usable ? Stage::ProbCutTT : Stage::ProbCutInit;
}

void MovePicker::score_captures() {
  for (ExtMove* entry = cur_; entry < end_; ++entry) {
    const Move m = entry->move;
    const PieceType victim = pos_.captured_type(m);
    int material = see_value(victim);
    if (m.kind() == MoveKind::Promotion) {
      material += see_value(m.promotion_type()) - see_value(PAWN);
    }
    entry->value = kCaptureVictimFactor * material +
                   histories_.capture[pos_.moved_piece(m)][m.to()][victim];
  }
}

void MovePicker::score_quiets() {
  const Color us = pos_.side_to_move();
  const auto& pawn = histories_.pawn[pos_.pawn_key() % kPawnHistorySize];
  const ContinuationRefs& refs = *continuation_;
  const Threats threats = compute_threats(pos_, ~us);

  for (ExtMove* entry = cur_; entry < end_; ++entry) {
    const Move m = entry->move;
    const Piece pc = pos_.moved_piece(m);
    const Square to = m.to();

    int value = 2 * main_entry(histories_.main, us, threats_, m) + 2 * pawn[pc][to];
    value += (*refs[0])[pc][to] + (*refs[1])[pc][to];
    value += (*refs[2])[pc][to] + (*refs[3])[pc][to] / 2;

    const Bitboard danger = threats.against(type_of(pc));
    if (danger & square_bb(m.from())) value += kThreatEscapeBonus;
    if (danger & square_bb(to)) value -= kThreatEscapeBonus;
    entry->value = value;
  }
}

void MovePicker::score_evasions() {
  const Color us = pos_.side_to_move();
  const ContinuationRefs& refs = *continuation_;

  for (ExtMove* entry = cur_; entry < end_; ++entry) {
    const Move m = entry->move;
    const Piece pc = pos_.moved_piece(m);
    if (pos_.is_capture(m)) {
      entry->value = kEvasionCaptureBase + see_value(pos_.captured_type(m)) - int(type_of(pc));
    } else {
      entry->value = main_entry(histories_.main, us, threats_, m) + (*refs[0])[pc][m.to()];
    }
  }
}

ExtMove MovePicker::pop_best() {
  ExtMove* best = cur_;
  for (ExtMove* entry = cur_ + 1; entry < end_; ++entry) {
    if (entry->value > best->value) best = entry;
  }
  std::swap(*best, *cur_);
  return *cur_++;
}

bool MovePicker::is_refutation(Move m) const {
  return m == refutations_.killer1 || m == refutations_.killer2 || m == refutations_.counter;
}

bool MovePicker::is_usable_refutation(Move m) const {
  return m != Move::none() && m != tt_move_ && pos_.is_pseudo_legal(m) && !pos_.is_tactical(m);
}

Move MovePicker::next_good_capture() {
  while (cur_ < end_) {
    const ExtMove pick = pop_best();
    if (pick.move == tt_move_) continue;
    // SEE no sabe valorar una promocion: en esta etapa solo llegan las de dama.
    if (pick.move.kind() == MoveKind::Promotion ||
        pos_.see_ge(pick.move, -pick.value / kGoodCaptureDivisor)) {
      return pick.move;
    }
    *end_bad_++ = pick;  // cabe: end_bad_ nunca adelanta a cur_
  }
  return Move::none();
}

Move MovePicker::next_quiet() {
  while (cur_ < end_) {
    const ExtMove pick = pop_best();
    if (pick.move != tt_move_ && !is_refutation(pick.move)) return pick.move;
  }
  return Move::none();
}

Move MovePicker::next_other_than_tt() {
  while (cur_ < end_) {
    const ExtMove pick = pop_best();
    if (pick.move != tt_move_) return pick.move;
  }
  return Move::none();
}

Move MovePicker::next_probcut() {
  while (cur_ < end_) {
    const ExtMove pick = pop_best();
    if (pick.move != tt_move_ && pos_.see_ge(pick.move, threshold_)) return pick.move;
  }
  return Move::none();
}

Move MovePicker::next(bool skip_quiets) {
  while (true) {
    switch (stage_) {
      case Stage::MainTT:
      case Stage::EvasionTT:
      case Stage::ProbCutTT:
      case Stage::QuiescenceTT:
        advance_stage();
        return tt_move_;

      case Stage::CaptureInit:
      case Stage::ProbCutInit:
      case Stage::QuiescenceInit:
        cur_ = end_bad_ = list_.begin();
        end_ = generate<GenType::Captures>(pos_, cur_);
        score_captures();
        advance_stage();
        break;

      case Stage::GoodCapture: {
        const Move m = next_good_capture();
        if (m != Move::none()) return m;
        stage_ = Stage::Killer1;
        break;
      }

      case Stage::Killer1:
        advance_stage();
        if (!skip_quiets && is_usable_refutation(refutations_.killer1)) {
          return refutations_.killer1;
        }
        break;

      case Stage::Killer2:
        advance_stage();
        if (!skip_quiets && refutations_.killer2 != refutations_.killer1 &&
            is_usable_refutation(refutations_.killer2)) {
          return refutations_.killer2;
        }
        break;

      case Stage::Counter:
        advance_stage();
        if (!skip_quiets && refutations_.counter != refutations_.killer1 &&
            refutations_.counter != refutations_.killer2 &&
            is_usable_refutation(refutations_.counter)) {
          return refutations_.counter;
        }
        break;

      case Stage::QuietInit:
        cur_ = end_ = end_bad_;
        if (!skip_quiets) {
          end_ = generate<GenType::Quiets>(pos_, cur_);
          score_quiets();
        }
        advance_stage();
        break;

      case Stage::Quiet: {
        if (!skip_quiets) {
          const Move m = next_quiet();
          if (m != Move::none()) return m;
        }
        cur_ = list_.begin();
        end_ = end_bad_;
        advance_stage();
        break;
      }

      case Stage::BadCapture:
        if (cur_ < end_) return (cur_++)->move;
        stage_ = Stage::Done;
        break;

      case Stage::EvasionInit:
        cur_ = list_.begin();
        end_ = generate<GenType::Evasions>(pos_, cur_);
        score_evasions();
        advance_stage();
        break;

      case Stage::Evasion:
      case Stage::Quiescence: {
        const Move m = next_other_than_tt();
        if (m != Move::none()) return m;
        stage_ = Stage::Done;
        break;
      }

      case Stage::ProbCut: {
        const Move m = next_probcut();
        if (m != Move::none()) return m;
        stage_ = Stage::Done;
        break;
      }

      case Stage::Done:
        return Move::none();
    }
  }
}

}  // namespace ayano

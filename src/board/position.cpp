#include "board/position.h"

#include "board/movegen.h"
#include "core/zobrist.h"

namespace enroque {

namespace {

// Derechos que sobreviven cuando una pieza sale o llega a cada casilla.
constexpr std::array<std::uint8_t, SQUARE_NB> make_castling_masks() {
  std::array<std::uint8_t, SQUARE_NB> masks{};
  for (auto& mask : masks) mask = ALL_CASTLING;
  masks[A1] = ALL_CASTLING & ~WHITE_OOO;
  masks[H1] = ALL_CASTLING & ~WHITE_OO;
  masks[E1] = ALL_CASTLING & ~(WHITE_OO | WHITE_OOO);
  masks[A8] = ALL_CASTLING & ~BLACK_OOO;
  masks[H8] = ALL_CASTLING & ~BLACK_OO;
  masks[E8] = ALL_CASTLING & ~(BLACK_OO | BLACK_OOO);
  return masks;
}

constexpr auto kCastlingMasks = make_castling_masks();

}  // namespace

void Position::toggle_keys(Piece pc, Square s) {
  const Key k = zobrist::piece(pc, s);
  const PieceType pt = type_of(pc);
  key_ ^= k;
  if (pt == PAWN) {
    pawn_key_ ^= k;
    return;
  }
  non_pawn_key_[color_of(pc)] ^= k;
  if (pt == KNIGHT || pt == BISHOP || pt == KING) minor_key_ ^= k;
}

void Position::put_piece(Piece pc, Square s) {
  const Bitboard bb = square_bb(s);
  board_[s] = pc;
  by_type_[type_of(pc)] |= bb;
  by_color_[color_of(pc)] |= bb;
  toggle_keys(pc, s);
}

void Position::remove_piece(Square s) {
  const Piece pc = board_[s];
  const Bitboard bb = square_bb(s);
  board_[s] = NO_PIECE;
  by_type_[type_of(pc)] ^= bb;
  by_color_[color_of(pc)] ^= bb;
  toggle_keys(pc, s);
}

void Position::move_piece(Square from, Square to) {
  const Piece pc = board_[from];
  const Bitboard both = square_bb(from) | square_bb(to);
  board_[from] = NO_PIECE;
  board_[to] = pc;
  by_type_[type_of(pc)] ^= both;
  by_color_[color_of(pc)] ^= both;
  toggle_keys(pc, from);
  toggle_keys(pc, to);
}

void Position::apply_castling(Square from, Square to) {
  const bool king_side = to > from;
  const Square rook_from = make_square(king_side ? 7 : 0, rank_of(from));
  const Square rook_to = make_square(king_side ? 5 : 3, rank_of(from));
  move_piece(from, to);
  move_piece(rook_from, rook_to);
}

void Position::update_castling_rights(Square from, Square to) {
  const int remaining = castling_ & kCastlingMasks[from] & kCastlingMasks[to];
  key_ ^= zobrist::castling(castling_) ^ zobrist::castling(remaining);
  castling_ = std::uint8_t(remaining);
}

// Solo se registra la casilla al paso si un peon rival puede capturar: asi dos
// posiciones identicas a efectos de repeticion comparten clave.
void Position::set_en_passant_if_capturable(Square from, Square to) {
  const Color mover = color_of(board_[to]);
  const Square passed = Square((int(from) + int(to)) / 2);
  if (attacks::pawn(mover, passed) & pieces(~mover, PAWN)) {
    ep_ = passed;
    key_ ^= zobrist::en_passant(file_of(passed));
  }
}

Bitboard Position::slider_blockers(Color king_color) const {
  const Square ksq = king_sq_[king_color];
  const Color them = ~king_color;
  Bitboard snipers = ((attacks::rook(ksq, 0) & pieces(them, ROOK, QUEEN)) |
                      (attacks::bishop(ksq, 0) & pieces(them, BISHOP, QUEEN)));
  const Bitboard occupied = pieces() ^ snipers;
  Bitboard result = 0;
  while (snipers) {
    const Square sniper = pop_lsb(snipers);
    const Bitboard in_between = attacks::between(ksq, sniper) & occupied;
    if (in_between && !more_than_one(in_between)) result |= in_between;
  }
  return result;
}

void Position::refresh_check_info() {
  checkers_ = attackers_to(king_sq_[stm_]) & by_color_[~stm_];
  blockers_[WHITE] = slider_blockers(WHITE);
  blockers_[BLACK] = slider_blockers(BLACK);
}

Bitboard Position::attackers_to(Square s, Bitboard occupied) const {
  return (attacks::pawn(BLACK, s) & pieces(WHITE, PAWN)) |
         (attacks::pawn(WHITE, s) & pieces(BLACK, PAWN)) |
         (attacks::knight(s) & pieces(KNIGHT)) | (attacks::king(s) & pieces(KING)) |
         (attacks::rook(s, occupied) & pieces(ROOK, QUEEN)) |
         (attacks::bishop(s, occupied) & pieces(BISHOP, QUEEN));
}

void Position::apply_to(Move m, Position& out) const {
  out = *this;

  const Color us = stm_;
  const Square from = m.from();
  const Square to = m.to();
  const MoveKind kind = m.kind();
  const Piece moved = board_[from];
  const Square captured_sq = kind == MoveKind::EnPassant ? to - pawn_push(us) : to;
  const Piece captured = kind == MoveKind::Castling ? NO_PIECE : board_[captured_sq];

  out.key_ ^= zobrist::side();
  if (ep_ != NO_SQUARE) {
    out.key_ ^= zobrist::en_passant(file_of(ep_));
    out.ep_ = NO_SQUARE;
  }
  out.captured_ = captured;
  out.rule50_ = (type_of(moved) == PAWN || captured != NO_PIECE) ? 0 : rule50_ + 1;
  out.game_ply_ = game_ply_ + 1;

  if (kind == MoveKind::Castling) {
    out.apply_castling(from, to);
  } else {
    if (captured != NO_PIECE) out.remove_piece(captured_sq);
    if (kind == MoveKind::Promotion) {
      out.remove_piece(from);
      out.put_piece(make_piece(us, m.promotion_type()), to);
    } else {
      out.move_piece(from, to);
      if (type_of(moved) == PAWN && (int(to) ^ int(from)) == 16) {
        out.set_en_passant_if_capturable(from, to);
      }
    }
  }

  if (type_of(moved) == KING) out.king_sq_[us] = to;
  if (castling_) out.update_castling_rights(from, to);
  out.stm_ = ~us;
  out.refresh_check_info();
}

void Position::apply_null_to(Position& out) const {
  out = *this;
  out.key_ ^= zobrist::side();
  if (ep_ != NO_SQUARE) {
    out.key_ ^= zobrist::en_passant(file_of(ep_));
    out.ep_ = NO_SQUARE;
  }
  out.captured_ = NO_PIECE;
  out.rule50_ = rule50_ + 1;
  out.game_ply_ = game_ply_ + 1;
  out.stm_ = ~stm_;
  // Ningun rey esta en jaque: la jugada nula no se permite estando en jaque.
  out.checkers_ = 0;
}

bool Position::is_legal(Move m) const {
  const Color us = stm_;
  const Color them = ~us;
  const Square from = m.from();
  const Square to = m.to();
  const Square ksq = king_sq_[us];

  if (m.kind() == MoveKind::EnPassant) {
    const Square captured_sq = to - pawn_push(us);
    const Bitboard occupied =
        (pieces() ^ square_bb(from) ^ square_bb(captured_sq)) | square_bb(to);
    // Se miran todos los atacantes y no solo las piezas deslizantes: un FEN imposible
    // puede traer un jaque de caballo que el al paso no resuelve.
    return !(attackers_to(ksq, occupied) & by_color_[them] & occupied);
  }

  if (m.kind() == MoveKind::Castling) {
    const Direction step = to > from ? WEST : EAST;
    for (Square s = to; s != from; s = s + step) {
      if (attackers_to(s) & by_color_[them]) return false;
    }
    return true;
  }

  if (type_of(board_[from]) == KING) {
    return !(attackers_to(to, pieces() ^ square_bb(from)) & by_color_[them]);
  }

  return !(blockers_[us] & square_bb(from)) || attacks::aligned(from, to, ksq);
}

bool Position::is_pseudo_legal(Move m) const {
  const Color us = stm_;
  const Square from = m.from();
  const Square to = m.to();
  const Piece pc = board_[from];

  if (!m.is_ok()) return false;
  if (m.kind() != MoveKind::Normal) {
    return generate_moves(*this, in_check() ? GenType::Evasions : GenType::NonEvasions).contains(m);
  }
  if (m.has_promotion_bits()) return false;
  if (pc == NO_PIECE || color_of(pc) != us) return false;
  if (by_color_[us] & square_bb(to)) return false;

  if (type_of(pc) == PAWN) {
    if (relative_rank(us, to) == 7) return false;  // debe ser una jugada de promocion
    const Direction up = pawn_push(us);
    const bool capture = (attacks::pawn(us, from) & by_color_[~us] & square_bb(to)) != 0;
    const bool single_push = from + up == to && is_empty(to);
    const bool double_push = relative_rank(us, from) == 1 && Square(int(from) + 2 * int(up)) == to &&
                             is_empty(to) && is_empty(from + up);
    if (!capture && !single_push && !double_push) return false;
  } else if (!(attacks::of(type_of(pc), from, pieces()) & square_bb(to))) {
    return false;
  }

  if (!checkers_) return true;

  // En jaque, solo cuentan las jugadas que podrian resolverlo.
  if (type_of(pc) == KING) {
    return !(attackers_to(to, pieces() ^ square_bb(from)) & by_color_[~us]);
  }
  if (more_than_one(checkers_)) return false;
  const Bitboard block_or_capture = attacks::between(king_sq_[us], lsb(checkers_)) | checkers_;
  return (block_or_capture & square_bb(to)) != 0;
}

bool Position::gives_check(Move m) const {
  const Color us = stm_;
  const Color them = ~us;
  const Square from = m.from();
  const Square to = m.to();
  const Square ksq = king_sq_[them];
  const Bitboard king_bb = square_bb(ksq);

  // Jaque a la descubierta: la pieza abandona la linea hacia el rey rival.
  if ((blockers_[them] & square_bb(from)) && !attacks::aligned(from, to, ksq)) return true;

  const Bitboard occupied = (pieces() ^ square_bb(from)) | square_bb(to);
  switch (m.kind()) {
    case MoveKind::Normal: {
      const PieceType pt = type_of(board_[from]);
      if (pt == PAWN) return (attacks::pawn(us, to) & king_bb) != 0;
      return (attacks::of(pt, to, occupied) & king_bb) != 0;
    }
    case MoveKind::Promotion:
      return (attacks::of(m.promotion_type(), to, occupied) & king_bb) != 0;
    case MoveKind::EnPassant: {
      const Bitboard after_capture = occupied ^ square_bb(to - pawn_push(us));
      return (attacks::pawn(us, to) & king_bb) ||
             (attacks::rook(ksq, after_capture) & pieces(us, ROOK, QUEEN)) ||
             (attacks::bishop(ksq, after_capture) & pieces(us, BISHOP, QUEEN));
    }
    case MoveKind::Castling: {
      const bool king_side = to > from;
      const Square rook_from = make_square(king_side ? 7 : 0, rank_of(from));
      const Square rook_to = make_square(king_side ? 5 : 3, rank_of(from));
      const Bitboard after_castling = (occupied ^ square_bb(rook_from)) | square_bb(rook_to);
      return (attacks::rook(rook_to, after_castling) & king_bb) != 0;
    }
  }
  return false;
}

bool Position::is_insufficient_material() const {
  if (pieces(PAWN, ROOK) | pieces(QUEEN)) return false;
  const int minors = popcount(pieces(KNIGHT, BISHOP));
  if (minors <= 1) return true;
  if (pieces(KNIGHT)) return false;
  const Bitboard bishops = pieces(BISHOP);
  return !(bishops & DARK_SQUARES_BB) || !(bishops & ~DARK_SQUARES_BB);
}

}  // namespace enroque

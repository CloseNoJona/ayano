#include "board/movegen.h"

namespace ayano {

namespace {

template <GenType Type>
ExtMove* add_promotions(ExtMove* list, Square from, Square to) {
  if constexpr (Type != GenType::Quiets) {
    (list++)->move = Move::promotion(from, to, QUEEN);
  }
  if constexpr (Type != GenType::Captures) {
    (list++)->move = Move::promotion(from, to, KNIGHT);
    (list++)->move = Move::promotion(from, to, ROOK);
    (list++)->move = Move::promotion(from, to, BISHOP);
  }
  return list;
}

template <Direction D>
ExtMove* add_pawn_moves(ExtMove* list, Bitboard targets) {
  while (targets) {
    const Square to = pop_lsb(targets);
    (list++)->move = Move::normal(to - D, to);
  }
  return list;
}

template <GenType Type, Direction D>
ExtMove* add_all_promotions(ExtMove* list, Bitboard targets) {
  while (targets) {
    const Square to = pop_lsb(targets);
    list = add_promotions<Type>(list, to - D, to);
  }
  return list;
}

template <Color Us, GenType Type>
ExtMove* generate_pawn_pushes(const Position& pos, ExtMove* list, Bitboard pawns, Bitboard target) {
  constexpr Direction Up = pawn_push(Us);
  constexpr Bitboard ThirdRank = Us == WHITE ? RANK_3_BB : RANK_6_BB;
  const Bitboard empty = ~pos.pieces();

  Bitboard single = shift<Up>(pawns) & empty;
  Bitboard twice = shift<Up>(single & ThirdRank) & empty;
  if constexpr (Type == GenType::Evasions) {
    single &= target;
    twice &= target;
  }
  list = add_pawn_moves<Up>(list, single);
  while (twice) {
    const Square to = pop_lsb(twice);
    (list++)->move = Move::normal(to - Up - Up, to);
  }
  return list;
}

template <Color Us, GenType Type>
ExtMove* generate_pawn_promotions(const Position& pos, ExtMove* list, Bitboard pawns,
                                  Bitboard enemies, Bitboard target) {
  constexpr Direction Up = pawn_push(Us);
  constexpr Direction UpEast = Us == WHITE ? NORTH_EAST : SOUTH_EAST;
  constexpr Direction UpWest = Us == WHITE ? NORTH_WEST : SOUTH_WEST;

  Bitboard pushes = shift<Up>(pawns) & ~pos.pieces();
  if constexpr (Type == GenType::Evasions) pushes &= target;

  list = add_all_promotions<Type, UpEast>(list, shift<UpEast>(pawns) & enemies);
  list = add_all_promotions<Type, UpWest>(list, shift<UpWest>(pawns) & enemies);
  return add_all_promotions<Type, Up>(list, pushes);
}

template <Color Us>
ExtMove* generate_pawn_captures(const Position& pos, ExtMove* list, Bitboard pawns,
                                Bitboard enemies) {
  constexpr Direction UpEast = Us == WHITE ? NORTH_EAST : SOUTH_EAST;
  constexpr Direction UpWest = Us == WHITE ? NORTH_WEST : SOUTH_WEST;

  list = add_pawn_moves<UpEast>(list, shift<UpEast>(pawns) & enemies);
  list = add_pawn_moves<UpWest>(list, shift<UpWest>(pawns) & enemies);

  if (pos.ep_square() != NO_SQUARE) {
    Bitboard capturers = pawns & attacks::pawn(~Us, pos.ep_square());
    while (capturers) {
      (list++)->move = Move::en_passant(pop_lsb(capturers), pos.ep_square());
    }
  }
  return list;
}

template <Color Us, GenType Type>
ExtMove* generate_pawns(const Position& pos, ExtMove* list, Bitboard target) {
  constexpr Bitboard SeventhRank = Us == WHITE ? RANK_7_BB : RANK_2_BB;
  const Bitboard pawns = pos.pieces(Us, PAWN);
  const Bitboard promoting = pawns & SeventhRank;
  const Bitboard others = pawns & ~SeventhRank;
  const Bitboard enemies = Type == GenType::Evasions ? pos.checkers() : pos.pieces(~Us);

  if constexpr (Type != GenType::Captures) {
    list = generate_pawn_pushes<Us, Type>(pos, list, others, target);
  }
  if (promoting) {
    list = generate_pawn_promotions<Us, Type>(pos, list, promoting, enemies, target);
  }
  if constexpr (Type != GenType::Quiets) {
    list = generate_pawn_captures<Us>(pos, list, others, enemies);
  }
  return list;
}

template <PieceType Pt>
ExtMove* generate_piece(const Position& pos, ExtMove* list, Color us, Bitboard target) {
  const Bitboard occupied = pos.pieces();
  Bitboard pieces = pos.pieces(us, Pt);
  while (pieces) {
    const Square from = pop_lsb(pieces);
    Bitboard moves = attacks::of(Pt, from, occupied) & target;
    while (moves) (list++)->move = Move::normal(from, pop_lsb(moves));
  }
  return list;
}

template <Color Us>
ExtMove* generate_castling(const Position& pos, ExtMove* list) {
  constexpr Square KingHome = Us == WHITE ? E1 : E8;
  constexpr int KingSide = Us == WHITE ? WHITE_OO : BLACK_OO;
  constexpr int QueenSide = Us == WHITE ? WHITE_OOO : BLACK_OOO;
  constexpr Bitboard KingSidePath = (Bitboard(0x60)) << (Us == WHITE ? 0 : 56);
  constexpr Bitboard QueenSidePath = (Bitboard(0x0E)) << (Us == WHITE ? 0 : 56);

  const int rights = pos.castling_rights();
  if ((rights & KingSide) && !(pos.pieces() & KingSidePath)) {
    (list++)->move = Move::castling(KingHome, Us == WHITE ? G1 : G8);
  }
  if ((rights & QueenSide) && !(pos.pieces() & QueenSidePath)) {
    (list++)->move = Move::castling(KingHome, Us == WHITE ? C1 : C8);
  }
  return list;
}

template <GenType Type>
Bitboard target_squares(const Position& pos, Color us) {
  if constexpr (Type == GenType::Captures) return pos.pieces(~us);
  if constexpr (Type == GenType::Quiets) return ~pos.pieces();
  if constexpr (Type == GenType::NonEvasions) return ~pos.pieces(us);
  if constexpr (Type == GenType::Evasions) {
    const Square checker = lsb(pos.checkers());
    return attacks::between(pos.king_square(us), checker) | square_bb(checker);
  }
}

template <Color Us, GenType Type>
ExtMove* generate_for(const Position& pos, ExtMove* list) {
  const Square ksq = pos.king_square(Us);
  const bool double_check = Type == GenType::Evasions && more_than_one(pos.checkers());

  if (!double_check) {
    const Bitboard target = target_squares<Type>(pos, Us);
    list = generate_pawns<Us, Type>(pos, list, target);
    list = generate_piece<KNIGHT>(pos, list, Us, target);
    list = generate_piece<BISHOP>(pos, list, Us, target);
    list = generate_piece<ROOK>(pos, list, Us, target);
    list = generate_piece<QUEEN>(pos, list, Us, target);
  }

  Bitboard king_target = ~pos.pieces(Us);
  if constexpr (Type == GenType::Captures) king_target = pos.pieces(~Us);
  if constexpr (Type == GenType::Quiets) king_target = ~pos.pieces();
  Bitboard king_moves = attacks::king(ksq) & king_target;
  while (king_moves) (list++)->move = Move::normal(ksq, pop_lsb(king_moves));

  if constexpr (Type == GenType::Quiets || Type == GenType::NonEvasions) {
    list = generate_castling<Us>(pos, list);
  }
  return list;
}

}  // namespace

template <GenType Type>
ExtMove* generate(const Position& pos, ExtMove* list) {
  return pos.side_to_move() == WHITE ? generate_for<WHITE, Type>(pos, list)
                                     : generate_for<BLACK, Type>(pos, list);
}

template ExtMove* generate<GenType::Captures>(const Position&, ExtMove*);
template ExtMove* generate<GenType::Quiets>(const Position&, ExtMove*);
template ExtMove* generate<GenType::Evasions>(const Position&, ExtMove*);
template ExtMove* generate<GenType::NonEvasions>(const Position&, ExtMove*);

MoveList generate_moves(const Position& pos, GenType type) {
  MoveList list;
  ExtMove* end = list.begin();
  switch (type) {
    case GenType::Captures: end = generate<GenType::Captures>(pos, end); break;
    case GenType::Quiets: end = generate<GenType::Quiets>(pos, end); break;
    case GenType::Evasions: end = generate<GenType::Evasions>(pos, end); break;
    case GenType::NonEvasions: end = generate<GenType::NonEvasions>(pos, end); break;
  }
  list.resize_to(end);
  return list;
}

MoveList legal_moves(const Position& pos) {
  MoveList list = generate_moves(pos, pos.in_check() ? GenType::Evasions : GenType::NonEvasions);
  const ExtMove* end = std::remove_if(list.begin(), list.end(), [&pos](const ExtMove& entry) {
    return !pos.is_legal(entry.move);
  });
  list.resize_to(end);
  return list;
}

Move parse_uci_move(const Position& pos, std::string_view text) {
  if (text.size() != 4 && text.size() != 5) return Move::none();
  for (const ExtMove& entry : legal_moves(pos)) {
    if (entry.move.to_uci() == text) return entry.move;
  }
  return Move::none();
}

}  // namespace ayano

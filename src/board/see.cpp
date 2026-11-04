// Evaluacion estatica de intercambios (SEE): decide si una serie de capturas en
// una casilla gana al menos `threshold`, sin mover piezas de verdad.
#include "board/position.h"

namespace enroque {

namespace {

PieceType least_valuable(const Position& pos, Bitboard attackers) {
  for (const PieceType pt : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}) {
    if (attackers & pos.pieces(pt)) return pt;
  }
  return KING;
}

}  // namespace

bool Position::see_ge(Move m, Value threshold) const {
  if (m.kind() != MoveKind::Normal) return threshold <= 0;

  const Square from = m.from();
  const Square to = m.to();

  // `balance` es la ganancia del bando que acaba de capturar si todo termina aqui.
  Value balance = see_value(type_of(board_[to])) - threshold;
  if (balance < 0) return false;
  balance -= see_value(type_of(board_[from]));
  if (balance >= 0) return true;

  Bitboard occupied = (pieces() ^ square_bb(from)) | square_bb(to);
  Bitboard attackers = attackers_to(to, occupied) & occupied;
  const Bitboard diagonal_sliders = pieces(BISHOP, QUEEN);
  const Bitboard straight_sliders = pieces(ROOK, QUEEN);
  Color side = ~stm_;

  while (true) {
    // Aproximacion: una pieza clavada a su rey no participa en el intercambio.
    const Bitboard side_attackers = attackers & by_color_[side] & ~blockers_[side];
    if (!side_attackers) break;

    const PieceType attacker = least_valuable(*this, side_attackers);
    side = ~side;
    balance = -balance - 1 - see_value(attacker);

    if (balance >= 0) {
      // El rey no puede capturar si la casilla sigue defendida.
      if (attacker == KING && (attackers & by_color_[side])) side = ~side;
      break;
    }

    occupied ^= square_bb(lsb(side_attackers & pieces(attacker)));
    if (attacker == PAWN || attacker == BISHOP || attacker == QUEEN) {
      attackers |= attacks::bishop(to, occupied) & diagonal_sliders;
    }
    if (attacker == ROOK || attacker == QUEEN) {
      attackers |= attacks::rook(to, occupied) & straight_sliders;
    }
    attackers &= occupied;
  }

  return side != stm_;
}

}  // namespace enroque

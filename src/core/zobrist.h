// Claves Zobrist: identifican una posicion con un hash de 64 bits actualizable
// de forma incremental con XOR.
#pragma once

#include <array>

#include "core/types.h"

namespace ayano::zobrist {

struct Keys {
  std::array<std::array<Key, SQUARE_NB>, PIECE_NB> piece;
  std::array<Key, CASTLING_NB> castling;
  std::array<Key, 8> en_passant;
  Key side;
};

extern Keys keys;

void init();

inline Key piece(Piece pc, Square s) { return keys.piece[pc][s]; }
inline Key castling(int rights) { return keys.castling[rights]; }
inline Key en_passant(int file) { return keys.en_passant[file]; }
inline Key side() { return keys.side; }

}  // namespace ayano::zobrist

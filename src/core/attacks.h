// Tablas de ataque precalculadas. Las piezas deslizantes usan PEXT o bitboards
// magicos segun como se compile (ver ENROQUE_PEXT): se decide en compilacion. PEXT es
// rapido en Intel y en AMD desde Zen 3, pero muy lento en Zen 1 y Zen 2.
#pragma once

#include <array>

#include "core/bitboard.h"
#include "core/types.h"

#ifdef ENROQUE_USE_PEXT
#include <immintrin.h>
#endif

namespace enroque::attacks {

// Debe llamarse una vez antes de usar cualquier otra funcion de este modulo.
void init();

struct SliderEntry {
  Bitboard mask;
  Bitboard magic;
  const Bitboard* table;
  unsigned shift;

  unsigned index(Bitboard occupied) const {
#ifdef ENROQUE_USE_PEXT
    return unsigned(_pext_u64(occupied, mask));
#else
    return unsigned(((occupied & mask) * magic) >> shift);
#endif
  }
};

using SquareTable = std::array<Bitboard, SQUARE_NB>;

struct Tables {
  SquareTable knight;
  SquareTable king;
  std::array<SquareTable, COLOR_NB> pawn;
  std::array<SquareTable, SQUARE_NB> between;  // casillas estrictamente entre dos casillas
  std::array<SquareTable, SQUARE_NB> line;     // linea completa que une dos casillas
  std::array<SliderEntry, SQUARE_NB> rook;
  std::array<SliderEntry, SQUARE_NB> bishop;
};

extern Tables tables;

inline Bitboard knight(Square s) { return tables.knight[s]; }
inline Bitboard king(Square s) { return tables.king[s]; }
inline Bitboard pawn(Color c, Square s) { return tables.pawn[c][s]; }
inline Bitboard between(Square a, Square b) { return tables.between[a][b]; }
inline Bitboard line(Square a, Square b) { return tables.line[a][b]; }

inline Bitboard rook(Square s, Bitboard occupied) {
  const SliderEntry& entry = tables.rook[s];
  return entry.table[entry.index(occupied)];
}

inline Bitboard bishop(Square s, Bitboard occupied) {
  const SliderEntry& entry = tables.bishop[s];
  return entry.table[entry.index(occupied)];
}

inline Bitboard queen(Square s, Bitboard occupied) {
  return rook(s, occupied) | bishop(s, occupied);
}

inline bool aligned(Square a, Square b, Square c) { return (line(a, b) & square_bb(c)) != 0; }

// Ataques de cualquier pieza excepto peones.
inline Bitboard of(PieceType pt, Square s, Bitboard occupied) {
  switch (pt) {
    case KNIGHT: return knight(s);
    case BISHOP: return bishop(s, occupied);
    case ROOK: return rook(s, occupied);
    case QUEEN: return queen(s, occupied);
    case KING: return king(s);
    default: return 0;
  }
}

}  // namespace enroque::attacks

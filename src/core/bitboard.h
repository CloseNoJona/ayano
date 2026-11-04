// Operaciones sobre bitboards: un entero de 64 bits con un bit por casilla.
#pragma once

#include <bit>

#include "core/types.h"

namespace enroque {

inline constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
inline constexpr Bitboard FILE_B_BB = FILE_A_BB << 1;
inline constexpr Bitboard FILE_G_BB = FILE_A_BB << 6;
inline constexpr Bitboard FILE_H_BB = FILE_A_BB << 7;
inline constexpr Bitboard RANK_1_BB = 0xFFULL;
inline constexpr Bitboard RANK_2_BB = RANK_1_BB << 8;
inline constexpr Bitboard RANK_3_BB = RANK_1_BB << 16;
inline constexpr Bitboard RANK_4_BB = RANK_1_BB << 24;
inline constexpr Bitboard RANK_5_BB = RANK_1_BB << 32;
inline constexpr Bitboard RANK_6_BB = RANK_1_BB << 40;
inline constexpr Bitboard RANK_7_BB = RANK_1_BB << 48;
inline constexpr Bitboard RANK_8_BB = RANK_1_BB << 56;
inline constexpr Bitboard DARK_SQUARES_BB = 0xAA55AA55AA55AA55ULL;

constexpr Bitboard file_bb(int file) { return FILE_A_BB << file; }
constexpr Bitboard rank_bb(int rank) { return RANK_1_BB << (8 * rank); }
constexpr Bitboard square_bb(Square s) { return Bitboard(1) << s; }

constexpr int popcount(Bitboard b) { return std::popcount(b); }
constexpr bool more_than_one(Bitboard b) { return (b & (b - 1)) != 0; }
inline Square lsb(Bitboard b) { return Square(std::countr_zero(b)); }
inline Square msb(Bitboard b) { return Square(63 - std::countl_zero(b)); }

inline Square pop_lsb(Bitboard& b) {
  const Square s = lsb(b);
  b &= b - 1;
  return s;
}

template <Direction D>
constexpr Bitboard shift(Bitboard b) {
  if constexpr (D == NORTH) return b << 8;
  if constexpr (D == SOUTH) return b >> 8;
  if constexpr (D == EAST) return (b & ~FILE_H_BB) << 1;
  if constexpr (D == WEST) return (b & ~FILE_A_BB) >> 1;
  if constexpr (D == NORTH_EAST) return (b & ~FILE_H_BB) << 9;
  if constexpr (D == NORTH_WEST) return (b & ~FILE_A_BB) << 7;
  if constexpr (D == SOUTH_EAST) return (b & ~FILE_H_BB) >> 7;
  if constexpr (D == SOUTH_WEST) return (b & ~FILE_A_BB) >> 9;
}

}  // namespace enroque

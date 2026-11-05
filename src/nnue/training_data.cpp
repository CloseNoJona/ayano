#include "nnue/training_data.h"

#include <algorithm>
#include <sstream>
#include <string_view>

namespace ayano::nnue {

PackedPosition pack_position(const Position& pos, Value score, GameResult result) {
  PackedPosition packed{};
  packed.occupancy = pos.pieces();
  packed.score = std::int16_t(std::clamp(score, -32000, 32000));
  packed.result = std::uint8_t(result);
  packed.side_to_move = std::uint8_t(pos.side_to_move());

  int index = 0;
  Bitboard occupied = packed.occupancy;
  while (occupied) {
    const Piece pc = pos.piece_on(pop_lsb(occupied));
    packed.pieces[std::size_t(index / 2)] |= std::uint8_t(pc << (4 * (index % 2)));
    ++index;
  }
  return packed;
}

UnpackedPosition unpack_position(const PackedPosition& packed) {
  UnpackedPosition result;
  result.side_to_move = Color(packed.side_to_move);

  Bitboard occupied = packed.occupancy;
  while (occupied) {
    const int i = result.count;
    const Square s = pop_lsb(occupied);
    const Piece pc = Piece((packed.pieces[std::size_t(i / 2)] >> (4 * (i % 2))) & 0xF);
    result.pieces[std::size_t(i)] = pc;
    result.squares[std::size_t(i)] = s;
    if (type_of(pc) == KING) result.kings[color_of(pc)] = s;
    ++result.count;
  }
  return result;
}

std::string to_fen(const PackedPosition& packed) {
  const UnpackedPosition unpacked = unpack_position(packed);
  std::array<char, 64> board{};
  constexpr std::string_view kChars = "PpNnBbRrQqKk";
  for (int i = 0; i < unpacked.count; ++i) {
    board[unpacked.squares[std::size_t(i)]] = kChars[unpacked.pieces[std::size_t(i)]];
  }
  std::ostringstream fen;
  for (int rank = 7; rank >= 0; --rank) {
    int empty = 0;
    for (int file = 0; file < 8; ++file) {
      const char c = board[std::size_t(rank * 8 + file)];
      if (!c) {
        ++empty;
        continue;
      }
      if (empty) fen << empty;
      empty = 0;
      fen << c;
    }
    if (empty) fen << empty;
    if (rank) fen << '/';
  }
  fen << (unpacked.side_to_move == WHITE ? " w - - 0 1" : " b - - 0 1");
  return fen.str();
}

bool is_valid(const PackedPosition& packed) {
  const int count = popcount(packed.occupancy);
  if (count < 2 || count > 32) return false;
  if (packed.side_to_move > 1 || packed.result > 2) return false;

  std::array<int, COLOR_NB> kings{};
  for (int i = 0; i < count; ++i) {
    const int nibble = (packed.pieces[std::size_t(i / 2)] >> (4 * (i % 2))) & 0xF;
    if (nibble >= PIECE_NB) return false;
    if (type_of(Piece(nibble)) == KING) ++kings[color_of(Piece(nibble))];
  }
  return kings[WHITE] == 1 && kings[BLACK] == 1;
}

}  // namespace ayano::nnue

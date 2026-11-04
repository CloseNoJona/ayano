#include "nnue/training_data.h"

#include <algorithm>

namespace enroque::nnue {

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

}  // namespace enroque::nnue

#include "board/perft.h"

#include <iostream>

#include "board/movegen.h"

namespace ayano {

std::uint64_t perft(const Position& pos, int depth) {
  const MoveList moves = legal_moves(pos);
  if (depth <= 1) return depth == 1 ? moves.size() : 1;

  std::uint64_t nodes = 0;
  Position next;
  for (const ExtMove& entry : moves) {
    pos.apply_to(entry.move, next);
    nodes += perft(next, depth - 1);
  }
  return nodes;
}

std::uint64_t perft_divide(const Position& pos, int depth) {
  std::uint64_t total = 0;
  for (const ExtMove& entry : legal_moves(pos)) {
    const std::uint64_t nodes = perft(pos.after(entry.move), depth - 1);
    std::cout << entry.move.to_uci() << ": " << nodes << "\n";
    total += nodes;
  }
  std::cout << "\nNodos: " << total << std::endl;
  return total;
}

}  // namespace ayano

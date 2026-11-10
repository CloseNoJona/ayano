#include "board/cuckoo.h"

#include <array>
#include <cstddef>
#include <utility>

#include "core/attacks.h"
#include "core/bitboard.h"
#include "core/zobrist.h"

namespace ayano::cuckoo {

namespace {

// Con 3668 jugadas la tabla queda llena al 22 %: cada insercion acaba en pocos pasos.
constexpr std::size_t kSize = 16384;
constexpr int kMaxKicks = 1000;

std::array<Key, kSize> keys{};
std::array<Move, kSize> moves{};
bool ready = false;

std::size_t slot1(Key key) { return std::size_t(key & (kSize - 1)); }
std::size_t slot2(Key key) { return std::size_t((key >> 16) & (kSize - 1)); }

// Cada clave tiene dos casillas posibles. Si la suya esta ocupada, desaloja a la que habia,
// que pasa a su otra casilla, y asi hasta dar con una libre.
bool insert(Key key, Move move) {
  std::size_t i = slot1(key);
  for (int kick = 0; kick < kMaxKicks; ++kick) {
    std::swap(keys[i], key);
    std::swap(moves[i], move);
    if (move == Move::none()) return true;
    i = i == slot1(key) ? slot2(key) : slot1(key);
  }
  return false;
}

}  // namespace

int init() {
  keys.fill(0);
  moves.fill(Move::none());
  ready = false;

  int count = 0;
  bool complete = true;
  for (const Color c : {WHITE, BLACK}) {
    for (const PieceType pt : {KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
      const Piece pc = make_piece(c, pt);
      for (int a = 0; a < SQUARE_NB; ++a) {
        const Bitboard targets = attacks::of(pt, Square(a), 0);
        for (int b = a + 1; b < SQUARE_NB; ++b) {
          if (!(targets & square_bb(Square(b)))) continue;
          const Key key =
              zobrist::piece(pc, Square(a)) ^ zobrist::piece(pc, Square(b)) ^ zobrist::side();
          complete = insert(key, Move::normal(Square(a), Square(b))) && complete;
          ++count;
        }
      }
    }
  }
  ready = complete && count == kReversibleMoves;
  return ready ? count : 0;
}

Move find(Key delta) {
  if (!ready) return Move::none();
  if (keys[slot1(delta)] == delta) return moves[slot1(delta)];
  if (keys[slot2(delta)] == delta) return moves[slot2(delta)];
  return Move::none();
}

}  // namespace ayano::cuckoo

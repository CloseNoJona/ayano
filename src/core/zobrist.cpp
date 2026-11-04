#include "core/zobrist.h"

#include <mutex>

#include "core/prng.h"

namespace enroque::zobrist {

Keys keys;

void init() {
  static std::once_flag once;
  std::call_once(once, [] {
    Prng rng(0xE4120C0E5EED5EEDULL);
    for (auto& per_square : keys.piece) {
      for (Key& key : per_square) key = rng.next();
    }
    for (Key& key : keys.en_passant) key = rng.next();
    keys.side = rng.next();

    // Cada derecho tiene una clave y las combinaciones son su XOR.
    std::array<Key, 4> single{};
    for (Key& key : single) key = rng.next();
    for (int rights = 0; rights < CASTLING_NB; ++rights) {
      Key combined = 0;
      for (int bit = 0; bit < 4; ++bit) {
        if (rights & (1 << bit)) combined ^= single[bit];
      }
      keys.castling[rights] = combined;
    }
  });
}

}  // namespace enroque::zobrist

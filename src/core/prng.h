// Generador pseudoaleatorio SplitMix64: rapido, determinista y con semilla fija. Sigue la
// implementacion de referencia de Sebastiano Vigna, de dominio publico.
#pragma once

#include <cstdint>

namespace ayano {

class Prng {
 public:
  explicit constexpr Prng(std::uint64_t seed) : state_(seed) {}

  constexpr std::uint64_t next() {
    state_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  // Entero en [0, bound).
  constexpr std::uint64_t below(std::uint64_t bound) { return next() % bound; }

 private:
  std::uint64_t state_;
};

}  // namespace ayano

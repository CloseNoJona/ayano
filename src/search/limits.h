// Condiciones de parada de una busqueda, tal como llegan por UCI.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "core/types.h"

namespace enroque {

struct SearchLimits {
  std::array<std::int64_t, COLOR_NB> time_ms{};
  std::array<std::int64_t, COLOR_NB> increment_ms{};
  std::int64_t move_time_ms = 0;
  int moves_to_go = 0;
  int depth = 0;
  int mate = 0;
  std::uint64_t nodes = 0;       // limite estricto
  std::uint64_t soft_nodes = 0;  // termina la iteracion en curso al superarlo
  bool infinite = false;
  bool ponder = false;
  std::vector<Move> search_moves;

  bool uses_clock() const { return time_ms[WHITE] > 0 || time_ms[BLACK] > 0; }
};

}  // namespace enroque

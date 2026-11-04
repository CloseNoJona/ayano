// Reparto del tiempo: decide cuanto pensar en esta jugada.
#pragma once

#include <chrono>
#include <cstdint>

#include "search/limits.h"

namespace enroque {

class TimeManager {
 public:
  void start(const SearchLimits& limits, Color us, int game_ply, int move_overhead_ms);

  std::int64_t elapsed_ms() const;
  // Objetivo orientativo: se revisa al terminar cada iteracion.
  std::int64_t optimum_ms() const { return optimum_ms_; }
  // Limite duro: la busqueda se corta aunque este a mitad de una iteracion.
  std::int64_t maximum_ms() const { return maximum_ms_; }
  bool has_limit() const { return maximum_ms_ > 0; }

 private:
  std::chrono::steady_clock::time_point start_time_{};
  std::int64_t optimum_ms_ = 0;
  std::int64_t maximum_ms_ = 0;
};

}  // namespace enroque

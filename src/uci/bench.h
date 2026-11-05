// Prueba de rendimiento con posiciones fijas. Su numero de nodos actua como huella:
// si cambia sin haber tocado la busqueda, algo se ha roto.
#pragma once

#include <cstdint>

#include "search/engine.h"

namespace ayano {

inline constexpr int kDefaultBenchDepth = 13;

struct BenchResult {
  std::uint64_t nodes = 0;
  std::int64_t time_ms = 0;

  std::uint64_t nps() const {
    return nodes * 1000 / std::uint64_t(time_ms > 0 ? time_ms : 1);
  }
};

BenchResult run_bench(Engine& engine, int depth);

}  // namespace ayano

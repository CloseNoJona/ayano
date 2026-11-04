// Comprueba al arrancar que la CPU tiene las instrucciones usadas al compilar, para dar
// un mensaje claro en lugar de terminar con una instruccion ilegal.
#pragma once

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace enroque {

inline bool cpu_supports_this_build() {
#if defined(_MSC_VER) && (defined(__AVX2__) || defined(ENROQUE_USE_PEXT))
  constexpr int kAvx2Bit = 1 << 5;
  constexpr int kBmi2Bit = 1 << 8;
  int registers[4] = {};
  __cpuidex(registers, 7, 0);
  const int features = registers[1];
#if defined(__AVX2__)
  if (!(features & kAvx2Bit)) return false;
#endif
#if defined(ENROQUE_USE_PEXT)
  if (!(features & kBmi2Bit)) return false;
#endif
#endif
  return true;
}

}  // namespace enroque

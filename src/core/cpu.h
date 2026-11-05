// Comprueba al arrancar que la CPU tiene las instrucciones usadas al compilar, para dar
// un mensaje claro en lugar de terminar con una instruccion ilegal.
#pragma once

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace ayano {

inline bool cpu_supports_this_build() {
#if defined(_MSC_VER) && (defined(__AVX2__) || defined(AYANO_USE_PEXT))
  int registers[4] = {};
  __cpuidex(registers, 7, 0);
  const int features = registers[1];
#if defined(__AVX2__)
  constexpr int kAvx2Bit = 1 << 5;
  if (!(features & kAvx2Bit)) return false;
#endif
#if defined(AYANO_USE_PEXT)
  constexpr int kBmi2Bit = 1 << 8;
  if (!(features & kBmi2Bit)) return false;
#endif
#elif defined(__GNUC__)
#if defined(__AVX2__)
  if (!__builtin_cpu_supports("avx2")) return false;
#endif
#if defined(AYANO_USE_PEXT)
  if (!__builtin_cpu_supports("bmi2")) return false;
#endif
#endif
  return true;
}

}  // namespace ayano

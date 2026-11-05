// Version por defecto: sin red integrada. Cuando existe nets/default.nnue, CMake
// compila en su lugar un archivo generado con los datos de la red.
#include "nnue/network.h"

#ifndef AYANO_HAS_EMBEDDED_NET

namespace ayano::nnue {

std::span<const std::uint8_t> embedded_network() { return {}; }

}  // namespace ayano::nnue

#endif

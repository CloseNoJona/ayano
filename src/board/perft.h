// Perft: cuenta los nodos del arbol de jugadas legales hasta una profundidad.
#pragma once

#include <cstdint>

#include "board/position.h"

namespace ayano {

std::uint64_t perft(const Position& pos, int depth);

// Imprime el recuento por cada jugada de la raiz y devuelve el total.
std::uint64_t perft_divide(const Position& pos, int depth);

}  // namespace ayano

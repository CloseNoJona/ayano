// Evaluacion clasica escrita a mano. Tiene dos usos: generar los primeros datos de
// entrenamiento de la red NNUE y servir de respaldo si no hay red disponible.
#pragma once

#include "board/position.h"

namespace enroque::classical {

// Valor en centipeones desde el punto de vista del bando con turno.
Value evaluate(const Position& pos);

}  // namespace enroque::classical

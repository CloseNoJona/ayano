// Evaluacion clasica escrita a mano. Tiene dos usos: generar los primeros datos de
// entrenamiento de la red NNUE y servir de respaldo si no hay red disponible.
#pragma once

#include "board/position.h"

namespace ayano::classical {

inline constexpr int kMaxPhase = 24;
inline constexpr int kTempo = 12;

// Todo lo que no sale de las tablas de material y casillas, visto por las blancas.
struct PositionalTerms {
  int mg = 0;
  int eg = 0;
};

PositionalTerms positional_terms(const Position& pos);

// Piezas que quedan en el tablero: 0 en un final de peones, kMaxPhase con todas.
int game_phase(const Position& pos);

// Valor en centipeones desde el punto de vista del bando con turno.
Value evaluate(const Position& pos);

}  // namespace ayano::classical

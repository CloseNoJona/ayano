// Formato de los datos de entrenamiento: 32 bytes por posicion, compartido por el
// generador de partidas y el entrenador.
#pragma once

#include <array>
#include <cstdint>

#include "board/position.h"

namespace enroque::nnue {

enum class GameResult : std::uint8_t { Loss = 0, Draw = 1, Win = 2 };

struct PackedPosition {
  std::uint64_t occupancy;
  std::array<std::uint8_t, 16> pieces;  // una pieza por nibble, en orden de casilla
  std::int16_t score;                   // desde el punto de vista del bando con turno
  std::uint8_t result;                  // GameResult, tambien relativo al bando con turno
  std::uint8_t side_to_move;
  std::array<std::uint8_t, 4> reserved;
};
static_assert(sizeof(PackedPosition) == 32);

PackedPosition pack_position(const Position& pos, Value score, GameResult result);

// Una posicion desempaquetada, lista para calcular las entradas de la red.
struct UnpackedPosition {
  std::array<Piece, 32> pieces;
  std::array<Square, 32> squares;
  int count = 0;
  Color side_to_move = WHITE;
  std::array<Square, COLOR_NB> kings{};
};

UnpackedPosition unpack_position(const PackedPosition& packed);

// Comprueba un registro leido de un archivo antes de usarlo para indexar la red.
bool is_valid(const PackedPosition& packed);

}  // namespace enroque::nnue

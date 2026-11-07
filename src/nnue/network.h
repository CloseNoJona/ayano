// Red NNUE: 768 entradas por perspectiva (pieza x casilla) para cada cubeta del rey propio,
// una capa oculta con activacion SCReLU y una salida elegida segun las piezas que quedan.
// El tablero se refleja para que el rey propio este siempre en el flanco de dama.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>

#include "board/position.h"

namespace ayano::nnue {

inline constexpr int kPieceSquareInputs = 768;
inline constexpr int kKingBuckets = 8;
inline constexpr int kFeatures = kPieceSquareInputs * kKingBuckets;
#ifndef AYANO_HIDDEN
#define AYANO_HIDDEN 1024
#endif
inline constexpr int kHidden = AYANO_HIDDEN;
static_assert(kHidden > 0 && kHidden % 16 == 0, "AVX2 procesa las neuronas de 16 en 16");
inline constexpr int kOutputBuckets = 8;
inline constexpr int kQuantActivation = 255;  // escala de la capa de entrada (QA)
inline constexpr int kQuantWeight = 64;       // escala de la capa de salida (QB)
// Centipeones por unidad de salida. Cada red guarda el suyo, porque depende de como se
// entreno: una escala baja concentra la precision en las posiciones igualadas.
inline constexpr int kDefaultEvalScale = 200;
inline constexpr int kMaxEvalScale = 2000;

struct alignas(64) Network {
  std::array<std::int16_t, kFeatures * kHidden> feature_weights;  // [entrada][neurona]
  std::array<std::int16_t, kHidden> feature_bias;
  // [cubeta][bando con turno, bando sin turno][neurona]
  std::array<std::int16_t, kOutputBuckets * 2 * kHidden> output_weights;
  std::array<std::int16_t, kOutputBuckets> output_bias;
  std::int32_t eval_scale;
};

// Refleja la casilla si el rey de esa perspectiva esta en las columnas e-h.
constexpr bool is_mirrored(Square king) { return file_of(king) >= 4; }

// Cubeta segun la casilla del rey, vista desde su bando y reflejada al flanco de dama: una
// fila por rango y cuatro columnas. Distingue mas donde el rey pasa la mayor parte de la
// partida. Debe coincidir con la del entrenador. El archivo de red no la describe: si cambia
// con el mismo numero de cubetas, cambia tambien `kMagic` para rechazar las redes anteriores.
// Reduce a 8 cubetas la del ejemplo de bullet, y coincide con la de Halogen.
inline constexpr std::array<std::uint8_t, 32> kKingBucketLayout = {
    0, 1, 2, 3,  //
    4, 4, 5, 5,  //
    6, 6, 6, 6,  //
    6, 6, 6, 6,  //
    7, 7, 7, 7,  //
    7, 7, 7, 7,  //
    7, 7, 7, 7,  //
    7, 7, 7, 7,
};

constexpr int king_bucket(Color perspective, Square king) {
  const int square = king ^ (perspective == BLACK ? 56 : 0);
  const int file = square & 7;
  return kKingBucketLayout[std::size_t((square >> 3) * 4 + (file < 4 ? file : 7 - file))];
}

constexpr int feature_index(Color perspective, Square king, Piece pc, Square s) {
  int square = s ^ (perspective == BLACK ? 56 : 0);
  if (is_mirrored(king)) square ^= 7;
  const int side = color_of(pc) == perspective ? 0 : 6;
  return king_bucket(perspective, king) * kPieceSquareInputs + (side + type_of(pc)) * 64 + square;
}

// Si el rey cambia de flanco o de cubeta, todas las entradas de esa perspectiva cambian.
constexpr bool needs_refresh(Color perspective, Square before, Square after) {
  return is_mirrored(before) != is_mirrored(after) ||
         king_bucket(perspective, before) != king_bucket(perspective, after);
}

constexpr int output_bucket(int piece_count) { return (piece_count - 2) / 4; }

// Red en uso, o nullptr si solo hay evaluacion clasica.
const Network* active_network();

// Carga la red integrada en el ejecutable, si la hay. Devuelve false si existe pero no
// es valida: el motor seguiria funcionando, aunque solo con la evaluacion clasica.
bool init(std::string* error = nullptr);

bool load_network_file(const std::string& path, std::string* error);
bool save_network_file(const Network& network, const std::string& path, std::string* error);
bool parse_network(std::span<const std::uint8_t> bytes, Network& network, std::string* error);

// Datos de la red embebida al compilar; vacio si no se integro ninguna.
std::span<const std::uint8_t> embedded_network();

}  // namespace ayano::nnue

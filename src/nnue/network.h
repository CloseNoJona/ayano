// Red NNUE: 768 entradas por perspectiva (pieza x casilla) para cada cubeta del rey propio,
// una capa oculta y una cabeza elegida segun las piezas que quedan. El tablero se refleja
// para que el rey propio este siempre en el flanco de dama.
//
// Hay dos cabezas, y cada archivo de red dice cual usa:
// - lineal: SCReLU de la capa oculta y una salida por cubeta (formato ENQ2);
// - profunda: producto por parejas de la capa oculta y tres capas por cubeta, de 16, 32 y 1
//   neuronas (formato ENQ3). Sigue la idea del ejemplo 4_multi_layer de bullet (MIT).
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

// Cabeza profunda. La neurona i se multiplica por la i + kPairs, ambas recortadas a [0, QA],
// y el producto se lleva a [0, 127] para que maddubs sume dos productos u8 x i8 sin saturar.
inline constexpr int kPairs = kHidden / 2;
inline constexpr int kDeepInputs = 2 * kPairs;  // las parejas de las dos perspectivas
inline constexpr int kDeepL1 = 16;
inline constexpr int kDeepL2 = 32;
inline constexpr int kPairShift = 9;
inline constexpr int kPairMax = (kQuantActivation * kQuantActivation) >> kPairShift;
inline constexpr int kDeepWeightScale = 64;  // escala de los pesos int8 de la primera capa
static_assert(kPairMax <= 127, "maddubs suma dos productos u8 x i8 en 16 bits con signo");
static_assert(kPairs % 32 == 0, "AVX2 procesa las parejas de 32 en 32");

enum class Head : std::uint8_t { Linear, Deep };

struct alignas(64) Network {
  std::array<std::int16_t, kFeatures * kHidden> feature_weights;  // [entrada][neurona]
  std::array<std::int16_t, kHidden> feature_bias;
  // Cabeza lineal: [cubeta][bando con turno, bando sin turno][neurona]
  std::array<std::int16_t, kOutputBuckets * 2 * kHidden> output_weights;
  std::array<std::int16_t, kOutputBuckets> output_bias;
  // Cabeza profunda. Primera capa: [cubeta][grupo de 4 entradas][neurona][entrada del grupo],
  // con las entradas del bando con turno primero: asi cada grupo no nulo suma sus 4 entradas a
  // las 16 neuronas de una vez. Segunda: [cubeta][entrada][neurona], para sumar varias neuronas
  // a la vez sin cambiar el orden de las sumas de cada una.
  alignas(64) std::array<std::int8_t, kOutputBuckets * kDeepL1 * kDeepInputs> l1_weights;
  std::array<float, kOutputBuckets * kDeepL1> l1_bias;
  std::array<float, kOutputBuckets * kDeepL1 * kDeepL2> l2_weights;
  std::array<float, kOutputBuckets * kDeepL2> l2_bias;
  std::array<float, kOutputBuckets * kDeepL2> l3_weights;
  std::array<float, kOutputBuckets> l3_bias;
  std::int32_t eval_scale;
  Head head = Head::Linear;
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

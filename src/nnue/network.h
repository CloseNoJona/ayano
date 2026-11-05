// Red NNUE: 768 entradas por perspectiva (pieza x casilla), una capa oculta con
// activacion SCReLU y una salida elegida segun las piezas que quedan. El tablero se
// refleja para que el rey propio este siempre en el flanco de dama.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>

#include "board/position.h"

namespace ayano::nnue {

inline constexpr int kFeatures = 768;
#ifndef AYANO_HIDDEN
#define AYANO_HIDDEN 512
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

constexpr int feature_index(Color perspective, Square king, Piece pc, Square s) {
  int square = s ^ (perspective == BLACK ? 56 : 0);
  if (is_mirrored(king)) square ^= 7;
  const int side = color_of(pc) == perspective ? 0 : 6;
  return (side + type_of(pc)) * 64 + square;
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

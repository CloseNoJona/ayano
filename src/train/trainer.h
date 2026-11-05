// Entrenador de la red NNUE en CPU. La entrada es dispersa (unas 30 de 768 entradas
// activas por perspectiva), asi que la retropropagacion solo toca una fraccion de los
// pesos y varios hilos bastan para entrenar sin GPU.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "nnue/network.h"
#include "nnue/training_data.h"

namespace ayano::train {

// Pesos en coma flotante, antes de cuantizar.
struct FloatNetwork {
  std::vector<float> feature_weights;  // [entrada][neurona]
  std::vector<float> feature_bias;
  std::vector<float> output_weights;  // [cubeta][2 * neurona]
  std::vector<float> output_bias;

  FloatNetwork();
  void randomize(std::uint64_t seed);
  std::size_t parameter_count() const;
};

struct TrainConfig {
  int epochs = 10;
  std::size_t batch_size = 4096;
  int threads = 4;
  float learning_rate = 0.001f;
  float final_learning_rate = 0.00002f;
  // 0: solo evaluacion de la busqueda, 1: solo resultado de la partida. Con pocas
  // partidas conviene 0: muchas posiciones comparten un unico resultado y la red
  // acaba memorizando ruido.
  float wdl_weight = 0.0f;
  int eval_scale = nnue::kDefaultEvalScale;
  float weight_clip = 1.98f;
  std::uint64_t seed = 1;
  std::string output_prefix = "nets/run";
};

// Motivo por el que no se puede entrenar con esta configuracion, si lo hay.
std::optional<std::string> config_error(const TrainConfig& config);

// Entradas activas de una posicion para las dos perspectivas.
struct Features {
  std::array<std::uint16_t, 32> own;
  std::array<std::uint16_t, 32> other;
  int count = 0;
  int bucket = 0;
};

Features extract_features(const nnue::PackedPosition& packed);

// Valor esperado que debe predecir la red, en el intervalo [0, 1].
float training_target(const nnue::PackedPosition& packed, float wdl_weight,
                      int eval_scale = nnue::kDefaultEvalScale);

// Salida de la red: multiplicada por la escala da centipeones.
float forward(const FloatNetwork& network, const Features& features);

// Error cuadratico medio sobre `samples`.
double mean_loss(const FloatNetwork& network, std::span<const nnue::PackedPosition> samples,
                 float wdl_weight, int eval_scale = nnue::kDefaultEvalScale);

// Gradiente completo de la perdida de una sola posicion; se expone para poder
// comprobarlo numericamente en los tests.
void accumulate_gradient(const FloatNetwork& network, const Features& features, float target,
                         FloatNetwork& gradient);

std::unique_ptr<nnue::Network> quantize(const FloatNetwork& network,
                                        int eval_scale = nnue::kDefaultEvalScale);

// Entrena y escribe `<prefijo>-<epoca>.nnue` tras cada epoca. Devuelve false en cuanto
// una red no se puede guardar: seguir solo perderia horas de calculo.
bool train(FloatNetwork& network, std::vector<nnue::PackedPosition>& samples,
           const TrainConfig& config, std::string* error = nullptr);

}  // namespace ayano::train

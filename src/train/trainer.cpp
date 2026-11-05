#include "train/trainer.h"

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>

#include "core/prng.h"

namespace ayano::train {

using nnue::kFeatures;
using nnue::kHidden;
using nnue::kOutputBuckets;
using nnue::kQuantActivation;
using nnue::kQuantWeight;
using nnue::PackedPosition;

namespace {

constexpr float kAdamBeta1 = 0.9f;
constexpr float kAdamBeta2 = 0.999f;
constexpr float kAdamEpsilon = 1e-8f;
constexpr double kValidationFraction = 0.005;
constexpr std::size_t kMaxValidationSamples = 200000;

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float screlu(float x) {
  const float clipped = std::clamp(x, 0.0f, 1.0f);
  return clipped * clipped;
}

// Activaciones de una posicion: dos vectores de kHidden.
struct Activations {
  std::array<float, kHidden> own;
  std::array<float, kHidden> other;
};

void accumulate_rows(const FloatNetwork& network, const std::uint16_t* features, int count,
                     std::array<float, kHidden>& output) {
  std::copy(network.feature_bias.begin(), network.feature_bias.end(), output.begin());
  float* __restrict destination = output.data();
  for (int i = 0; i < count; ++i) {
    const float* __restrict row =
        network.feature_weights.data() + std::size_t(features[i]) * kHidden;
    for (int j = 0; j < kHidden; ++j) destination[j] += row[j];
  }
}

float network_output(const FloatNetwork& network, const Activations& acts, int bucket) {
  const float* own_weights = network.output_weights.data() + std::size_t(bucket) * 2 * kHidden;
  const float* other_weights = own_weights + kHidden;
  float sum = network.output_bias[std::size_t(bucket)];
  for (int j = 0; j < kHidden; ++j) {
    sum += screlu(acts.own[std::size_t(j)]) * own_weights[j];
    sum += screlu(acts.other[std::size_t(j)]) * other_weights[j];
  }
  return sum;
}

void compute_activations(const FloatNetwork& network, const Features& features, Activations& acts) {
  accumulate_rows(network, features.own.data(), features.count, acts.own);
  accumulate_rows(network, features.other.data(), features.count, acts.other);
}

// Propaga el error hacia una perspectiva y acumula el gradiente de su capa de entrada.
void backprop_side(const std::array<float, kHidden>& activation, const float* output_weights,
                   float* output_gradient, const std::uint16_t* features, int count,
                   float error, FloatNetwork& gradient) {
  std::array<float, kHidden> delta;
  for (int j = 0; j < kHidden; ++j) {
    const float x = activation[std::size_t(j)];
    const float clipped = std::clamp(x, 0.0f, 1.0f);
    output_gradient[j] += error * clipped * clipped;
    // La derivada de clamp(x)^2 es 2x dentro de (0, 1) y cero fuera.
    delta[std::size_t(j)] = (x > 0.0f && x < 1.0f) ? error * output_weights[j] * 2.0f * x : 0.0f;
  }

  float* __restrict bias = gradient.feature_bias.data();
  const float* __restrict values = delta.data();
  for (int j = 0; j < kHidden; ++j) bias[j] += values[j];
  for (int i = 0; i < count; ++i) {
    float* __restrict row = gradient.feature_weights.data() + std::size_t(features[i]) * kHidden;
    for (int j = 0; j < kHidden; ++j) row[j] += values[j];
  }
}

class Adam {
 public:
  explicit Adam(std::size_t size) : first_(size, 0.0f), second_(size, 0.0f) {}

  // Actualiza solo [begin, end), lo que permite repartir el trabajo entre hilos.
  void step(float* weights, const float* gradient, std::size_t offset, std::size_t begin,
            std::size_t end, float learning_rate, float bias1, float bias2, float clip) {
    for (std::size_t i = begin; i < end; ++i) {
      const std::size_t k = offset + i;
      const float g = gradient[i];
      first_[k] = kAdamBeta1 * first_[k] + (1.0f - kAdamBeta1) * g;
      second_[k] = kAdamBeta2 * second_[k] + (1.0f - kAdamBeta2) * g * g;
      const float update = (first_[k] / bias1) / (std::sqrt(second_[k] / bias2) + kAdamEpsilon);
      weights[i] = std::clamp(weights[i] - learning_rate * update, -clip, clip);
    }
  }

 private:
  std::vector<float> first_;
  std::vector<float> second_;
};

void clear(FloatNetwork& network) {
  std::fill(network.feature_weights.begin(), network.feature_weights.end(), 0.0f);
  std::fill(network.feature_bias.begin(), network.feature_bias.end(), 0.0f);
  std::fill(network.output_weights.begin(), network.output_weights.end(), 0.0f);
  std::fill(network.output_bias.begin(), network.output_bias.end(), 0.0f);
}

// Suma el rango [begin, end) de los gradientes de todos los hilos en el primero.
void reduce_range(std::vector<float>& total, const std::vector<FloatNetwork>& parts,
                  std::vector<float> FloatNetwork::* member, std::size_t begin, std::size_t end) {
  for (std::size_t t = 1; t < parts.size(); ++t) {
    const std::vector<float>& part = parts[t].*member;
    for (std::size_t i = begin; i < end; ++i) total[i] += part[i];
  }
}

struct Slice {
  std::size_t begin;
  std::size_t end;
};

Slice slice_of(std::size_t size, int index, int count) {
  return {size * std::size_t(index) / std::size_t(count),
          size * std::size_t(index + 1) / std::size_t(count)};
}

class TrainingRun {
 public:
  TrainingRun(FloatNetwork& network, const TrainConfig& config)
      : network_(network),
        config_(config),
        gradients_(std::size_t(config.threads)),
        adam_(network.parameter_count()),
        losses_(std::size_t(config.threads), 0.0) {}

  double run_epoch(std::span<const PackedPosition> data, const std::vector<std::uint32_t>& order,
                   int epoch);

 private:
  void process_batch(int thread, std::span<const PackedPosition> data,
                     std::span<const std::uint32_t> batch);
  void apply_gradients(int thread, float learning_rate, float batch_scale);

  FloatNetwork& network_;
  const TrainConfig& config_;
  std::vector<FloatNetwork> gradients_;
  Adam adam_;
  std::vector<double> losses_;
  std::uint64_t step_ = 0;
};

void TrainingRun::process_batch(int thread, std::span<const PackedPosition> data,
                                std::span<const std::uint32_t> batch) {
  FloatNetwork& gradient = gradients_[std::size_t(thread)];
  clear(gradient);
  const Slice mine = slice_of(batch.size(), thread, config_.threads);
  double loss = 0.0;
  for (std::size_t i = mine.begin; i < mine.end; ++i) {
    const PackedPosition& sample = data[batch[i]];
    const Features features = extract_features(sample);
    const float target = training_target(sample, config_.wdl_weight, config_.eval_scale);
    const float prediction = sigmoid(forward(network_, features));
    loss += double(prediction - target) * double(prediction - target);
    accumulate_gradient(network_, features, target, gradient);
  }
  losses_[std::size_t(thread)] += loss;
}

void TrainingRun::apply_gradients(int thread, float learning_rate, float batch_scale) {
  const float bias1 = 1.0f - std::pow(kAdamBeta1, float(step_));
  const float bias2 = 1.0f - std::pow(kAdamBeta2, float(step_));
  FloatNetwork& total = gradients_[0];

  std::size_t offset = 0;
  const auto update = [&](std::vector<float>& weights, std::vector<float>& gradient,
                          std::vector<float> FloatNetwork::* member, float clip) {
    const Slice mine = slice_of(weights.size(), thread, config_.threads);
    reduce_range(gradient, gradients_, member, mine.begin, mine.end);
    for (std::size_t i = mine.begin; i < mine.end; ++i) gradient[i] *= batch_scale;
    adam_.step(weights.data(), gradient.data(), offset, mine.begin, mine.end, learning_rate, bias1,
               bias2, clip);
    offset += weights.size();
  };

  const float clip = config_.weight_clip;
  constexpr float kNoClip = 1e6f;
  update(network_.feature_weights, total.feature_weights, &FloatNetwork::feature_weights, clip);
  update(network_.feature_bias, total.feature_bias, &FloatNetwork::feature_bias, kNoClip);
  update(network_.output_weights, total.output_weights, &FloatNetwork::output_weights, clip);
  update(network_.output_bias, total.output_bias, &FloatNetwork::output_bias, kNoClip);
}

double TrainingRun::run_epoch(std::span<const PackedPosition> data,
                              const std::vector<std::uint32_t>& order, int epoch) {
  const std::size_t batch_count = order.size() / config_.batch_size;
  const double total_batches = double(batch_count) * config_.epochs;
  std::fill(losses_.begin(), losses_.end(), 0.0);

  float learning_rate = config_.learning_rate;
  std::barrier sync(config_.threads);

  const auto worker = [&](int thread) {
    for (std::size_t b = 0; b < batch_count; ++b) {
      const auto batch = std::span(order).subspan(b * config_.batch_size, config_.batch_size);
      process_batch(thread, data, batch);
      sync.arrive_and_wait();

      if (thread == 0) {
        ++step_;
        // Decaimiento exponencial desde la tasa inicial hasta la final.
        const double progress = (double(epoch) * double(batch_count) + double(b)) / total_batches;
        learning_rate = float(config_.learning_rate *
                              std::pow(config_.final_learning_rate / config_.learning_rate,
                                       progress));
      }
      sync.arrive_and_wait();

      apply_gradients(thread, learning_rate, 1.0f / float(config_.batch_size));
      sync.arrive_and_wait();
    }
  };

  std::vector<std::thread> threads;
  for (int t = 1; t < config_.threads; ++t) threads.emplace_back(worker, t);
  worker(0);
  for (std::thread& thread : threads) thread.join();

  const double loss = std::accumulate(losses_.begin(), losses_.end(), 0.0);
  return loss / double(batch_count * config_.batch_size);
}

}  // namespace

std::optional<std::string> config_error(const TrainConfig& config) {
  if (config.epochs < 1 || config.threads < 1 || config.batch_size == 0) {
    return "las epocas, los hilos y el tamano de lote deben ser positivos";
  }
  const auto is_rate = [](float value) { return std::isfinite(value) && value > 0.0f; };
  if (!is_rate(config.learning_rate) || !is_rate(config.final_learning_rate)) {
    return "las tasas de aprendizaje deben ser positivas y finitas";
  }
  // Escrito asi para rechazar tambien NaN, que no cumple ninguna comparacion.
  if (!(config.wdl_weight >= 0.0f && config.wdl_weight <= 1.0f)) {
    return "el peso del resultado debe estar entre 0 y 1";
  }
  if (config.eval_scale < 1 || config.eval_scale > nnue::kMaxEvalScale) {
    return "la escala debe estar entre 1 y " + std::to_string(nnue::kMaxEvalScale);
  }
  return std::nullopt;
}

FloatNetwork::FloatNetwork()
    : feature_weights(std::size_t(kFeatures) * kHidden, 0.0f),
      feature_bias(kHidden, 0.0f),
      output_weights(std::size_t(kOutputBuckets) * 2 * kHidden, 0.0f),
      output_bias(kOutputBuckets, 0.0f) {}

void FloatNetwork::randomize(std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> feature_noise(0.0f, 0.06f);
  std::uniform_real_distribution<float> output_noise(-0.04f, 0.04f);
  for (float& w : feature_weights) w = feature_noise(rng);
  for (float& w : output_weights) w = output_noise(rng);
  std::fill(feature_bias.begin(), feature_bias.end(), 0.2f);
}

std::size_t FloatNetwork::parameter_count() const {
  return feature_weights.size() + feature_bias.size() + output_weights.size() + output_bias.size();
}

Features extract_features(const PackedPosition& packed) {
  const nnue::UnpackedPosition pos = nnue::unpack_position(packed);
  const Color us = pos.side_to_move;
  const Color them = ~us;
  Features features;
  features.count = pos.count;
  features.bucket = nnue::output_bucket(pos.count);
  for (int i = 0; i < pos.count; ++i) {
    const Piece pc = pos.pieces[std::size_t(i)];
    const Square s = pos.squares[std::size_t(i)];
    features.own[std::size_t(i)] = std::uint16_t(nnue::feature_index(us, pos.kings[us], pc, s));
    features.other[std::size_t(i)] =
        std::uint16_t(nnue::feature_index(them, pos.kings[them], pc, s));
  }
  return features;
}

float training_target(const PackedPosition& packed, float wdl_weight, int eval_scale) {
  const float from_score = sigmoid(float(packed.score) / float(eval_scale));
  const float from_result = float(packed.result) / 2.0f;
  return wdl_weight * from_result + (1.0f - wdl_weight) * from_score;
}

float forward(const FloatNetwork& network, const Features& features) {
  Activations acts;
  compute_activations(network, features, acts);
  return network_output(network, acts, features.bucket);
}

void accumulate_gradient(const FloatNetwork& network, const Features& features, float target,
                         FloatNetwork& gradient) {
  Activations acts;
  compute_activations(network, features, acts);
  const float output = network_output(network, acts, features.bucket);
  const float prediction = sigmoid(output);
  // d/dsalida de (sigmoide(salida) - objetivo)^2
  const float error = 2.0f * (prediction - target) * prediction * (1.0f - prediction);

  const std::size_t base = std::size_t(features.bucket) * 2 * kHidden;
  gradient.output_bias[std::size_t(features.bucket)] += error;
  backprop_side(acts.own, network.output_weights.data() + base,
                gradient.output_weights.data() + base, features.own.data(), features.count, error,
                gradient);
  backprop_side(acts.other, network.output_weights.data() + base + kHidden,
                gradient.output_weights.data() + base + kHidden, features.other.data(),
                features.count, error, gradient);
}

double mean_loss(const FloatNetwork& network, std::span<const PackedPosition> samples,
                 float wdl_weight, int eval_scale) {
  double total = 0.0;
  for (const PackedPosition& sample : samples) {
    const float prediction = sigmoid(forward(network, extract_features(sample)));
    const float difference = prediction - training_target(sample, wdl_weight, eval_scale);
    total += double(difference) * double(difference);
  }
  return samples.empty() ? 0.0 : total / double(samples.size());
}

std::unique_ptr<nnue::Network> quantize(const FloatNetwork& network, int eval_scale) {
  auto result = std::make_unique<nnue::Network>();
  result->eval_scale = eval_scale;
  const auto to_int16 = [](float value, int scale) {
    const long rounded = std::lround(value * float(scale));
    return std::int16_t(std::clamp<long>(rounded, -32767, 32767));
  };
  for (std::size_t i = 0; i < network.feature_weights.size(); ++i) {
    result->feature_weights[i] = to_int16(network.feature_weights[i], kQuantActivation);
  }
  for (std::size_t i = 0; i < network.feature_bias.size(); ++i) {
    result->feature_bias[i] = to_int16(network.feature_bias[i], kQuantActivation);
  }
  for (std::size_t i = 0; i < network.output_weights.size(); ++i) {
    result->output_weights[i] = to_int16(network.output_weights[i], kQuantWeight);
  }
  for (std::size_t i = 0; i < network.output_bias.size(); ++i) {
    result->output_bias[i] = to_int16(network.output_bias[i], kQuantActivation * kQuantWeight);
  }
  return result;
}

bool train(FloatNetwork& network, std::vector<PackedPosition>& samples, const TrainConfig& config,
           std::string* error) {
  const auto fail = [error](const std::string& message) {
    if (error) *error = message;
    return false;
  };
  if (const auto problem = config_error(config)) return fail(*problem);

  std::mt19937_64 rng(config.seed);
  std::shuffle(samples.begin(), samples.end(), rng);

  const std::size_t validation_size = std::min(
      kMaxValidationSamples, std::size_t(double(samples.size()) * kValidationFraction));
  const std::span<const PackedPosition> validation(samples.data(), validation_size);
  const std::span<const PackedPosition> data(samples.data() + validation_size,
                                             samples.size() - validation_size);
  if (data.size() < config.batch_size) return fail("hay menos posiciones que el tamano de lote");

  std::vector<std::uint32_t> order(data.size());
  std::iota(order.begin(), order.end(), 0U);

  TrainingRun run(network, config);
  for (int epoch = 0; epoch < config.epochs; ++epoch) {
    const auto start = std::chrono::steady_clock::now();
    std::shuffle(order.begin(), order.end(), rng);
    const double loss = run.run_epoch(data, order, epoch);
    if (!std::isfinite(loss)) return fail("la perdida no es finita: el entrenamiento divergio");
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    const std::string path = config.output_prefix + "-" + std::to_string(epoch + 1) + ".nnue";
    std::string save_error;
    if (!nnue::save_network_file(*quantize(network, config.eval_scale), path, &save_error)) {
      return fail(save_error);
    }

    std::cout << "epoca " << epoch + 1 << "/" << config.epochs << " | perdida " << loss
              << " | validacion "
              << mean_loss(network, validation, config.wdl_weight, config.eval_scale) << " | "
              << std::uint64_t(double(data.size()) / seconds) << " pos/s | " << path << std::endl;
  }
  return true;
}

}  // namespace ayano::train

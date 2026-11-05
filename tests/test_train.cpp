#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "board/movegen.h"
#include "core/prng.h"
#include "nnue/accumulator.h"
#include "nnue/training_data.h"
#include "test_framework.h"
#include "train/trainer.h"

using namespace ayano;
using namespace ayano::nnue;
using namespace ayano::train;

namespace {

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

double sample_loss(const FloatNetwork& network, const Features& features, float target) {
  const double difference = double(sigmoid(forward(network, features))) - double(target);
  return difference * difference;
}

PackedPosition sample_position(const char* fen, Value score, GameResult result) {
  const auto pos = Position::from_fen(fen);
  CHECK(pos.has_value());
  return pack_position(pos ? *pos : Position::startpos(), score, result);
}

constexpr const char* kKiwipete =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

// Posiciones tras seis jugadas al azar, con una evaluacion que depende del turno.
std::vector<PackedPosition> random_samples(int count) {
  std::vector<PackedPosition> samples;
  Prng rng(11);
  for (int i = 0; i < count; ++i) {
    Position pos = Position::startpos();
    for (int ply = 0; ply < 6; ++ply) {
      const MoveList moves = legal_moves(pos);
      if (moves.empty()) break;
      pos = pos.after(moves[rng.below(moves.size())].move);
    }
    const Value score = pos.side_to_move() == WHITE ? 300 : -300;
    samples.push_back(pack_position(pos, score, GameResult::Draw));
  }
  return samples;
}

template <typename Change>
bool is_rejected(Change change) {
  TrainConfig config;
  change(config);
  return config_error(config).has_value();
}

}  // namespace

TEST(empaquetar_conserva_piezas_y_turno) {
  const auto pos = Position::from_fen(kKiwipete);
  CHECK(pos.has_value());
  if (!pos) return;
  const PackedPosition packed = pack_position(*pos, -37, GameResult::Win);
  const UnpackedPosition unpacked = unpack_position(packed);

  CHECK_EQ(packed.score, std::int16_t(-37));
  CHECK_EQ(int(packed.result), int(GameResult::Win));
  CHECK_EQ(unpacked.count, popcount(pos->pieces()));
  CHECK_EQ(int(unpacked.side_to_move), int(pos->side_to_move()));
  CHECK_EQ(int(unpacked.kings[WHITE]), int(pos->king_square(WHITE)));
  CHECK_EQ(int(unpacked.kings[BLACK]), int(pos->king_square(BLACK)));
  for (int i = 0; i < unpacked.count; ++i) {
    const Square s = unpacked.squares[std::size_t(i)];
    CHECK_EQ(int(unpacked.pieces[std::size_t(i)]), int(pos->piece_on(s)));
  }
}

TEST(datos_corruptos_se_detectan) {
  const PackedPosition good = sample_position(kKiwipete, 10, GameResult::Draw);
  CHECK(is_valid(good));

  PackedPosition too_many = good;
  too_many.occupancy = ~std::uint64_t(0);
  CHECK(!is_valid(too_many));

  PackedPosition bad_piece = good;
  bad_piece.pieces[0] = 0xFF;
  CHECK(!is_valid(bad_piece));

  PackedPosition bad_result = good;
  bad_result.result = 3;
  CHECK(!is_valid(bad_result));

  PackedPosition no_king = sample_position("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0, GameResult::Draw);
  no_king.pieces[0] = std::uint8_t(W_PAWN | (B_PAWN << 4));
  CHECK(!is_valid(no_king));
}

TEST(entradas_del_entrenador_coinciden_con_las_del_motor) {
  const PackedPosition packed = sample_position(kKiwipete, 0, GameResult::Draw);
  const auto pos = Position::from_fen(kKiwipete);
  CHECK(pos.has_value());
  if (!pos) return;
  const Features features = extract_features(packed);
  CHECK_EQ(features.count, popcount(pos->pieces()));
  CHECK_EQ(features.bucket, output_bucket(popcount(pos->pieces())));

  const Color us = pos->side_to_move();
  int index = 0;
  Bitboard occupied = pos->pieces();
  while (occupied) {
    const Square s = pop_lsb(occupied);
    const int expected = feature_index(us, pos->king_square(us), pos->piece_on(s), s);
    CHECK_EQ(int(features.own[std::size_t(index)]), expected);
    ++index;
  }
}

TEST(objetivo_mezcla_evaluacion_y_resultado) {
  const PackedPosition win = sample_position(kKiwipete, 0, GameResult::Win);
  CHECK(std::abs(training_target(win, 1.0f) - 1.0f) < 1e-6f);
  CHECK(std::abs(training_target(win, 0.0f) - 0.5f) < 1e-6f);
  CHECK(std::abs(training_target(win, 0.5f) - 0.75f) < 1e-6f);
}

// Comprobacion clasica: el gradiente analitico debe coincidir con el numerico.
TEST(la_escala_cambia_cuanto_pesa_la_evaluacion) {
  const PackedPosition sample = sample_position(kKiwipete, 200, GameResult::Draw);
  CHECK(training_target(sample, 0.0f, 100) > training_target(sample, 0.0f, 400));
  CHECK(std::abs(training_target(sample, 0.0f, 200) - sigmoid(1.0f)) < 1e-6f);
}

TEST(gradiente_coincide_con_diferencias_finitas) {
  auto network = std::make_unique<FloatNetwork>();
  network->randomize(3);
  const PackedPosition packed = sample_position(kKiwipete, 120, GameResult::Win);
  const Features features = extract_features(packed);
  const float target = training_target(packed, 0.5f);

  auto gradient = std::make_unique<FloatNetwork>();
  accumulate_gradient(*network, features, target, *gradient);

  const auto check_parameter = [&](float& parameter, float analytic) {
    const float original = parameter;
    const float step = 1e-2f;
    parameter = original + step;
    const double above = sample_loss(*network, features, target);
    parameter = original - step;
    const double below = sample_loss(*network, features, target);
    parameter = original;
    const double numeric = (above - below) / (2.0 * step);
    const double tolerance = 1e-5 + 0.05 * std::abs(numeric);
    if (std::abs(numeric - double(analytic)) > tolerance) {
      testfw::fail(__FILE__, __LINE__,
                   "numerico " + std::to_string(numeric) + " analitico " +
                       std::to_string(analytic));
    }
  };

  Prng rng(1);
  const std::size_t base = std::size_t(features.bucket) * 2 * kHidden;
  for (int i = 0; i < 40; ++i) {
    const std::size_t j = base + rng.below(2 * kHidden);
    check_parameter(network->output_weights[j], gradient->output_weights[j]);
  }
  check_parameter(network->output_bias[std::size_t(features.bucket)],
                  gradient->output_bias[std::size_t(features.bucket)]);
  for (int i = 0; i < 40; ++i) {
    const std::size_t feature = features.own[rng.below(std::size_t(features.count))];
    const std::size_t j = feature * kHidden + rng.below(kHidden);
    check_parameter(network->feature_weights[j], gradient->feature_weights[j]);
  }
  for (int i = 0; i < 20; ++i) {
    const std::size_t j = rng.below(kHidden);
    check_parameter(network->feature_bias[j], gradient->feature_bias[j]);
  }
}

// Con pesos situados exactamente en la rejilla de cuantizacion no hay error de
// redondeo: si la formula de la red cuantizada es correcta, ambas redes coinciden.
TEST(cuantizar_conserva_la_evaluacion) {
  auto network = std::make_unique<FloatNetwork>();
  network->randomize(5);
  const auto snap = [](std::vector<float>& values, float scale) {
    for (float& value : values) value = std::round(value * scale) / scale;
  };
  snap(network->feature_weights, float(kQuantActivation));
  snap(network->feature_bias, float(kQuantActivation));
  snap(network->output_weights, float(kQuantWeight));
  snap(network->output_bias, float(kQuantActivation * kQuantWeight));
  const auto quantized = quantize(*network);

  for (const char* fen : {kKiwipete, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                          "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"}) {
    const auto pos = Position::from_fen(fen);
    CHECK(pos.has_value());
    if (!pos) continue;
    const Features features = extract_features(pack_position(*pos, 0, GameResult::Draw));
    const float expected = forward(*network, features) * float(quantized->eval_scale);
    const Value actual = evaluate_from_scratch(*quantized, *pos);
    if (std::abs(expected - float(actual)) > 2.0f) {
      testfw::fail(__FILE__, __LINE__,
                   std::string(fen) + ": flotante " + std::to_string(expected) + " cuantizada " +
                       std::to_string(actual));
    }
  }
}

TEST(configuracion_de_entrenamiento_se_valida) {
  CHECK(!config_error(TrainConfig{}).has_value());
  CHECK(is_rejected([](TrainConfig& c) { c.eval_scale = 0; }));
  CHECK(is_rejected([](TrainConfig& c) { c.eval_scale = -200; }));
  CHECK(is_rejected([](TrainConfig& c) { c.eval_scale = kMaxEvalScale + 1; }));
  CHECK(is_rejected([](TrainConfig& c) { c.learning_rate = 0.0f; }));
  CHECK(is_rejected([](TrainConfig& c) { c.final_learning_rate = -0.1f; }));
  constexpr float kInfinity = std::numeric_limits<float>::infinity();
  CHECK(is_rejected([](TrainConfig& c) { c.learning_rate = kInfinity; }));
  CHECK(is_rejected([](TrainConfig& c) { c.final_learning_rate = kInfinity; }));
  CHECK(is_rejected([](TrainConfig& c) { c.wdl_weight = 1.5f; }));
  CHECK(is_rejected([](TrainConfig& c) { c.wdl_weight = std::nanf(""); }));
  CHECK(is_rejected([](TrainConfig& c) { c.epochs = 0; }));
  CHECK(is_rejected([](TrainConfig& c) { c.threads = 0; }));
  CHECK(is_rejected([](TrainConfig& c) { c.batch_size = 0; }));
}

// Sin esto se perderia todo el entrenamiento y el proceso diria que ha ido bien.
TEST(entrenar_falla_si_no_puede_guardar_la_red) {
  std::vector<PackedPosition> samples = random_samples(512);
  auto network = std::make_unique<FloatNetwork>();
  TrainConfig config;
  config.epochs = 2;
  config.batch_size = 256;
  config.threads = 1;
  config.output_prefix = "carpeta_que_no_existe/red";
  std::string error;
  CHECK(!ayano::train::train(*network, samples, config, &error));
  CHECK(!error.empty());
}

// Lo que se aparta para validar puede dejar menos posiciones que un lote.
TEST(entrenar_falla_si_no_queda_un_lote_completo) {
  std::vector<PackedPosition> samples = random_samples(300);
  auto network = std::make_unique<FloatNetwork>();
  TrainConfig config;
  config.batch_size = 300;
  config.threads = 1;
  config.output_prefix = "test_entrenamiento_sin_lotes";
  std::string error;
  CHECK(!ayano::train::train(*network, samples, config, &error));
  CHECK(!error.empty());
}

// Una red con NaN se guardaria como un archivo valido que evalua basura.
TEST(entrenar_falla_si_la_perdida_no_es_finita) {
  std::vector<PackedPosition> samples = random_samples(512);
  auto network = std::make_unique<FloatNetwork>();
  std::fill(network->output_bias.begin(), network->output_bias.end(), std::nanf(""));
  TrainConfig config;
  config.epochs = 1;
  config.batch_size = 256;
  config.threads = 1;
  config.output_prefix = "test_entrenamiento_divergente";
  std::string error;
  CHECK(!ayano::train::train(*network, samples, config, &error));
  CHECK(!error.empty());
}

TEST(entrenar_reduce_la_perdida) {
  // Un conjunto minimo: la red debe aprender que tener una dama de mas es bueno.
  std::vector<PackedPosition> samples = random_samples(4096);

  auto network = std::make_unique<FloatNetwork>();
  network->randomize(1);
  const double before = mean_loss(*network, samples, 0.0f);

  TrainConfig config;
  config.epochs = 3;
  config.batch_size = 256;
  config.threads = 2;
  config.wdl_weight = 0.0f;
  config.output_prefix = "test_entrenamiento_temporal";
  std::string error;
  CHECK(ayano::train::train(*network, samples, config, &error));
  CHECK(error.empty());

  const double after = mean_loss(*network, samples, 0.0f);
  CHECK(after < before);
  for (int epoch = 1; epoch <= config.epochs; ++epoch) {
    std::remove((config.output_prefix + "-" + std::to_string(epoch) + ".nnue").c_str());
  }
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "board/movegen.h"
#include "core/prng.h"
#include "nnue/accumulator.h"
#include "nnue/network.h"
#include "test_framework.h"

using namespace ayano;
using namespace ayano::nnue;

namespace {

std::unique_ptr<Network> random_network(std::uint64_t seed) {
  auto network = std::make_unique<Network>();
  network->eval_scale = 400;
  Prng rng(seed);
  const auto small = [&rng](int range) {
    return std::int16_t(int(rng.below(std::uint64_t(2 * range + 1))) - range);
  };
  for (auto& w : network->feature_weights) w = small(120);
  for (auto& b : network->feature_bias) b = small(200);
  for (auto& w : network->output_weights) w = small(100);
  for (auto& b : network->output_bias) b = small(2000);
  return network;
}

std::unique_ptr<Network> random_deep_network(std::uint64_t seed) {
  auto network = random_network(seed);
  network->head = Head::Deep;
  Prng rng(seed * 7919 + 1);
  const auto uniform = [&rng](float range) {
    return (float(rng.below(20001)) / 10000.0f - 1.0f) * range;
  };
  // Pesos de la primera capa en todo el rango de int8, para cubrir los extremos de maddubs.
  for (auto& w : network->l1_weights) w = std::int8_t(int(rng.below(256)) - 128);
  for (auto& b : network->l1_bias) b = uniform(1.0f);
  for (auto& w : network->l2_weights) w = uniform(1.5f);
  for (auto& b : network->l2_bias) b = uniform(0.5f);
  for (auto& w : network->l3_weights) w = uniform(1.5f);
  for (auto& b : network->l3_bias) b = uniform(0.5f);
  return network;
}

// La cabeza profunda calculada de la forma mas directa, sin acumuladores ni SIMD.
Value reference_deep_eval(const Network& network, const Position& pos) {
  std::array<std::array<std::int32_t, kHidden>, COLOR_NB> hidden{};
  for (const Color perspective : {WHITE, BLACK}) {
    auto& values = hidden[std::size_t(perspective)];
    for (int i = 0; i < kHidden; ++i) values[std::size_t(i)] = network.feature_bias[std::size_t(i)];
    Bitboard occupied = pos.pieces();
    while (occupied) {
      const Square s = pop_lsb(occupied);
      const int feature = feature_index(perspective, pos.king_square(perspective), pos.piece_on(s), s);
      for (int i = 0; i < kHidden; ++i) {
        values[std::size_t(i)] += network.feature_weights[std::size_t(feature) * kHidden + std::size_t(i)];
      }
    }
  }
  std::array<int, kDeepInputs> inputs{};
  const Color order[2] = {pos.side_to_move(), ~pos.side_to_move()};
  for (int side = 0; side < 2; ++side) {
    const auto& values = hidden[std::size_t(order[side])];
    for (int i = 0; i < kPairs; ++i) {
      const int a = std::clamp(values[std::size_t(i)], 0, kQuantActivation);
      const int b = std::clamp(values[std::size_t(i + kPairs)], 0, kQuantActivation);
      inputs[std::size_t(side * kPairs + i)] = (a * b) >> kPairShift;
    }
  }
  const std::size_t bucket = std::size_t(output_bucket(popcount(pos.pieces())));
  const auto screlu = [](float x) {
    const float c = std::clamp(x, 0.0f, 1.0f);
    return c * c;
  };
  std::array<float, kDeepL1> first{};
  for (std::size_t o = 0; o < first.size(); ++o) {
    std::int32_t sum = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      // [cubeta][grupo de 4 entradas][neurona][entrada del grupo]
      sum += inputs[i] * network.l1_weights[((bucket * kDeepInputs / 4 + i / 4) * kDeepL1 + o) * 4 + i % 4];
    }
    first[o] = screlu(float(sum) / float(kPairMax * kDeepWeightScale) + network.l1_bias[bucket * kDeepL1 + o]);
  }
  std::array<float, kDeepL2> second{};
  for (std::size_t k = 0; k < second.size(); ++k) {
    float sum = network.l2_bias[bucket * kDeepL2 + k];
    for (std::size_t o = 0; o < first.size(); ++o) {
      sum += network.l2_weights[(bucket * kDeepL1 + o) * kDeepL2 + k] * first[o];
    }
    second[k] = screlu(sum);
  }
  // Ocho sumas parciales por resto modulo 8, juntadas de dos en dos, y al final el sesgo.
  std::array<float, 8> lanes{};
  for (std::size_t k = 0; k < second.size(); ++k) {
    lanes[k % 8] = lanes[k % 8] + network.l3_weights[bucket * kDeepL2 + k] * second[k];
  }
  const float sum = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
  const float output = network.l3_bias[bucket] + sum;
  return Value(static_cast<int>(output * float(network.eval_scale)));
}

struct Frame {
  Position pos;
  Move move;
};

}  // namespace

// El camino SIMD de la cabeza profunda debe dar exactamente lo mismo que el calculo directo.
TEST(cabeza_profunda_coincide_con_el_calculo_directo) {
  const std::vector<std::string> fens = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "8/8/4k3/8/8/3K4/4P3/8 b - - 0 1",
  };
  for (const std::uint64_t seed : {3u, 4u, 5u}) {
    const auto network = random_deep_network(seed);
    for (const std::string& fen : fens) {
      const auto pos = Position::from_fen(fen);
      CHECK(pos.has_value());
      if (!pos) continue;
      CHECK_EQ(evaluate_from_scratch(*network, *pos), reference_deep_eval(*network, *pos));
    }
  }
}

TEST(cabeza_profunda_incremental_coincide_con_recalculo) {
  const auto network = random_deep_network(17);
  std::mt19937 rng(5);
  auto stack = std::make_unique<AccumulatorStack>();
  for (int game = 0; game < 12; ++game) {
    std::vector<Frame> frames(1);
    frames[0] = {Position::startpos(), Move::none()};
    stack->reset();
    const auto get_entry = [&frames](int ply) -> const Frame& { return frames[std::size_t(ply)]; };
    for (int ply = 0; ply < 60; ++ply) {
      if (rng() % 2 == 0) {
        CHECK_EQ(stack->evaluate(*network, ply, get_entry),
                 evaluate_from_scratch(*network, frames[std::size_t(ply)].pos));
      }
      const MoveList moves = legal_moves(frames[std::size_t(ply)].pos);
      if (moves.empty()) break;
      const Move m = moves[rng() % moves.size()].move;
      frames[std::size_t(ply)].move = m;
      frames.push_back({frames[std::size_t(ply)].pos.after(m), Move::none()});
      stack->on_move(ply + 1);
    }
  }
}

TEST(cabeza_profunda_es_simetrica_al_invertir_colores) {
  const auto network = random_deep_network(23);
  const auto white = Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
  const auto black = Position::from_fen("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3");
  CHECK(white && black);
  if (!white || !black) return;
  CHECK_EQ(evaluate_from_scratch(*network, *white), evaluate_from_scratch(*network, *black));
}

TEST(cabeza_profunda_se_guarda_y_se_carga_igual) {
  const auto original = random_deep_network(8);
  original->eval_scale = 155;
  const std::string path = "test_red_profunda.nnue";
  std::string error;
  CHECK(save_network_file(*original, path, &error));
  CHECK(load_network_file(path, &error));
  const Network* loaded = active_network();
  CHECK(loaded != nullptr);
  if (loaded) {
    CHECK(loaded->head == Head::Deep);
    CHECK(loaded->feature_weights == original->feature_weights);
    CHECK(loaded->l1_weights == original->l1_weights);
    CHECK(loaded->l1_bias == original->l1_bias);
    CHECK(loaded->l2_weights == original->l2_weights);
    CHECK(loaded->l3_bias == original->l3_bias);
    CHECK_EQ(loaded->eval_scale, 155);
    CHECK_EQ(evaluate_from_scratch(*loaded, Position::startpos()),
             evaluate_from_scratch(*original, Position::startpos()));
  }
  std::remove(path.c_str());
  CHECK(init(&error));  // deja la red integrada para el resto de los tests
}

// Un NaN o un peso desmesurado darian evaluaciones sin sentido o un entero desbordado.
TEST(cabeza_profunda_con_valores_invalidos_se_rechaza) {
  const std::string path = "test_red_profunda_invalida.nnue";
  for (const float bad : {std::nanf(""), 1e30f, -20000.0f}) {
    auto network = random_deep_network(9);
    network->l2_weights[40] = bad;
    std::string error;
    CHECK(save_network_file(*network, path, &error));
    CHECK(!load_network_file(path, &error));
    CHECK(!error.empty());
  }
  std::remove(path.c_str());
}

// Si el camino incremental y el recalculo difieren en un solo bit, la red evaluaria
// distinto segun como se llego a la posicion.
TEST(acumulador_incremental_coincide_con_recalculo) {
  const auto network = random_network(7);
  const std::vector<std::string> fens = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      // Con reyes solos y libres, casi cada jugada cambia su cubeta o su flanco.
      "8/8/3k4/8/8/4K3/8/8 w - - 0 1",
  };

  std::mt19937 rng(99);
  auto stack = std::make_unique<AccumulatorStack>();
  for (const std::string& fen : fens) {
    for (int game = 0; game < 20; ++game) {
      const auto start = Position::from_fen(fen);
      CHECK(start.has_value());
      if (!start) return;

      std::vector<Frame> frames(1);
      frames[0].pos = *start;
      frames[0].move = Move::none();
      stack->reset();

      const auto get_entry = [&frames](int ply) -> const Frame& {
        return frames[std::size_t(ply)];
      };

      for (int ply = 0; ply < 80; ++ply) {
        // Se evalua solo a veces para ejercitar el calculo perezoso de varios niveles.
        if (rng() % 3 == 0) {
          CHECK_EQ(stack->evaluate(*network, ply, get_entry),
                   evaluate_from_scratch(*network, frames[std::size_t(ply)].pos));
        }
        const MoveList moves = legal_moves(frames[std::size_t(ply)].pos);
        if (moves.empty()) break;
        const Move m = moves[rng() % moves.size()].move;
        frames[std::size_t(ply)].move = m;
        frames.push_back({frames[std::size_t(ply)].pos.after(m), Move::none()});
        stack->on_move(ply + 1);
      }
    }
  }
}

// La cache guarda sumas de una red concreta: tras cambiar de red no debe reutilizarlas.
TEST(cache_de_acumuladores_no_sobrevive_a_otra_red) {
  const auto first = random_network(31);
  const auto second = random_network(32);
  auto stack = std::make_unique<AccumulatorStack>();
  std::vector<Frame> frames(1);
  frames[0] = {Position::startpos(), Move::none()};
  const auto get_entry = [&frames](int ply) -> const Frame& { return frames[std::size_t(ply)]; };

  stack->reset();
  (void)stack->evaluate(*first, 0, get_entry);
  stack->reset();
  CHECK_EQ(stack->evaluate(*second, 0, get_entry), evaluate_from_scratch(*second, frames[0].pos));
}

TEST(jugada_nula_conserva_el_acumulador) {
  const auto network = random_network(11);
  auto stack = std::make_unique<AccumulatorStack>();
  std::vector<Frame> frames(2);
  frames[0] = {Position::startpos(), Move::null()};
  frames[1] = {frames[0].pos.after_null(), Move::none()};
  const auto get_entry = [&frames](int ply) -> const Frame& { return frames[std::size_t(ply)]; };

  stack->reset();
  (void)stack->evaluate(*network, 0, get_entry);
  stack->on_null_move(1);
  CHECK_EQ(stack->evaluate(*network, 1, get_entry),
           evaluate_from_scratch(*network, frames[1].pos));
}

TEST(la_red_es_simetrica_al_invertir_colores) {
  const auto network = random_network(21);
  const auto white = Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
  const auto black = Position::from_fen("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3");
  CHECK(white && black);
  if (!white || !black) return;
  CHECK_EQ(evaluate_from_scratch(*network, *white), evaluate_from_scratch(*network, *black));
}

// Con toda la capa saturada y los pesos en su limite, la suma no cabe en 32 bits. El archivo
// de red admite estos pesos, asi que la evaluacion debe seguir siendo exacta.
TEST(salida_de_la_red_no_desborda_con_pesos_extremos) {
  auto network = std::make_unique<Network>();
  network->eval_scale = 1;
  network->feature_bias.fill(std::int16_t(kQuantActivation));
  for (const std::int16_t weight : {std::int16_t(128), std::int16_t(-128)}) {
    network->output_weights.fill(weight);
    CHECK_EQ(evaluate_from_scratch(*network, Position::startpos()), Value(kHidden * weight / 32));
  }
}

TEST(indice_de_entrada_refleja_cuando_el_rey_cruza) {
  CHECK(!is_mirrored(D1));
  CHECK(is_mirrored(E1));
  // Con el rey en el flanco de rey, a1 y h1 intercambian su indice.
  CHECK_EQ(feature_index(WHITE, E1, W_ROOK, A1), feature_index(WHITE, D1, W_ROOK, H1));
  // Mismo tipo de pieza propia desde la perspectiva negra, con el tablero invertido.
  CHECK_EQ(feature_index(WHITE, A1, W_PAWN, E2), feature_index(BLACK, A8, B_PAWN, E7));
}

// Debe coincidir con la disposicion del entrenador: si no, la red evaluaria otra posicion.
TEST(cubeta_de_rey_depende_de_su_casilla_relativa) {
  CHECK_EQ(king_bucket(WHITE, A1), 0);
  CHECK_EQ(king_bucket(WHITE, D1), 3);
  CHECK_EQ(king_bucket(WHITE, A2), 4);
  CHECK_EQ(king_bucket(WHITE, D2), 5);
  CHECK_EQ(king_bucket(WHITE, A3), 6);
  CHECK_EQ(king_bucket(WHITE, D4), 6);
  CHECK_EQ(king_bucket(WHITE, A5), 7);
  CHECK_EQ(king_bucket(WHITE, D8), 7);
}

TEST(cubeta_de_rey_es_simetrica) {
  for (int rank = 0; rank < 8; ++rank) {
    for (int file = 0; file < 8; ++file) {
      const Square s = make_square(file, rank);
      const int bucket = king_bucket(WHITE, s);
      CHECK(bucket >= 0 && bucket < kKingBuckets);
      CHECK_EQ(bucket, king_bucket(WHITE, make_square(7 - file, rank)));
      CHECK_EQ(bucket, king_bucket(BLACK, make_square(file, 7 - rank)));
    }
  }
}

TEST(indice_de_entrada_depende_de_la_cubeta_del_rey) {
  const int base = feature_index(WHITE, A1, W_PAWN, E2);
  CHECK_EQ(feature_index(WHITE, D1, W_PAWN, E2), base + 3 * kPieceSquareInputs);
  CHECK_EQ(feature_index(WHITE, D8, W_PAWN, E2), base + 7 * kPieceSquareInputs);
  CHECK(feature_index(WHITE, D8, B_KING, H8) < kFeatures);
}

TEST(cubetas_de_salida_cubren_todo_el_material) {
  CHECK_EQ(output_bucket(2), 0);
  CHECK_EQ(output_bucket(32), 7);
  for (int pieces = 2; pieces <= 32; ++pieces) {
    CHECK(output_bucket(pieces) >= 0 && output_bucket(pieces) < kOutputBuckets);
  }
}

TEST(red_rechaza_archivos_corruptos) {
  auto network = std::make_unique<Network>();
  std::string error;
  const std::vector<std::uint8_t> too_short(100, 0);
  CHECK(!parse_network(too_short, *network, &error));
  CHECK(!error.empty());
}

// Con pesos mayores la multiplicacion de 16 bits de SCReLU desbordaria sin avisar.
TEST(red_con_pesos_de_salida_fuera_de_rango_se_rechaza) {
  auto network = random_network(9);
  network->output_weights[17] = 500;
  const std::string path = "test_red_fuera_de_rango.nnue";
  std::string error;
  CHECK(save_network_file(*network, path, &error));
  CHECK(!load_network_file(path, &error));
  CHECK(!error.empty());
  std::remove(path.c_str());
}

// Una escala invalida produciria evaluaciones sin sentido sin ningun otro sintoma.
TEST(red_con_escala_fuera_de_rango_se_rechaza) {
  auto network = random_network(9);
  const std::string path = "test_red_escala.nnue";
  for (const int scale : {0, -200, kMaxEvalScale + 1}) {
    network->eval_scale = scale;
    std::string error;
    CHECK(save_network_file(*network, path, &error));
    CHECK(!load_network_file(path, &error));
    CHECK(!error.empty());
  }
  std::remove(path.c_str());
}

TEST(red_en_formato_antiguo_se_rechaza) {
  const auto network = random_network(9);
  const std::string path = "test_red_formato.nnue";
  std::string error;
  CHECK(save_network_file(*network, path, &error));
  std::ifstream in(path, std::ios::binary);
  std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(in),
                                  std::istreambuf_iterator<char>()};
  in.close();
  std::remove(path.c_str());
  CHECK(bytes.size() > 4);
  if (bytes.size() <= 4) return;

  bytes[3] = '1';  // ENQ2 -> ENQ1
  auto parsed = std::make_unique<Network>();
  CHECK(!parse_network(bytes, *parsed, &error));
  CHECK(!error.empty());
}

// Si la red integrada no encaja, el motor jugaria con la evaluacion clasica.
TEST(la_red_integrada_es_valida) {
  const auto embedded = embedded_network();
  if (embedded.empty()) return;  // compilacion sin red integrada
  auto network = std::make_unique<Network>();
  std::string error;
  CHECK(parse_network(embedded, *network, &error));
  CHECK(error.empty());
}

TEST(iniciar_con_la_red_integrada_no_da_error) {
  std::string error;
  CHECK(init(&error));
  CHECK(error.empty());
}

TEST(guardar_red_en_carpeta_inexistente_da_error) {
  const auto network = random_network(9);
  std::string error;
  CHECK(!save_network_file(*network, "carpeta_que_no_existe/red.nnue", &error));
  CHECK(!error.empty());
}

TEST(archivo_de_red_inexistente_da_error) {
  std::string error;
  CHECK(!load_network_file("no_existe_esta_red.nnue", &error));
  CHECK(!error.empty());
}

TEST(red_se_guarda_y_se_carga_igual) {
  const auto original = random_network(5);
  const std::string path = "test_red_temporal.nnue";
  std::string error;
  CHECK(save_network_file(*original, path, &error));
  CHECK(load_network_file(path, &error));
  const Network* loaded = active_network();
  CHECK(loaded != nullptr);
  if (loaded) {
    CHECK(loaded->feature_weights == original->feature_weights);
    CHECK(loaded->output_weights == original->output_weights);
    CHECK(loaded->output_bias == original->output_bias);
    CHECK_EQ(loaded->eval_scale, original->eval_scale);
  }
  std::remove(path.c_str());
}

// Compara redes y evaluacion clasica sobre datos de entrenamiento, sin jugar partidas:
// mide con que precision cada una predice el resultado y la evaluacion de la busqueda.
//
// Uso: netcheck --data data/gen0.bin --net nets/x.nnue [--net otra.nnue] [--samples 200000]
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/attacks.h"
#include "core/prng.h"
#include "core/zobrist.h"
#include "eval/classical.h"
#include "nnue/accumulator.h"
#include "nnue/network.h"
#include "nnue/training_data.h"

using namespace enroque;

namespace {

// Escala fija para que las metricas de redes distintas sean comparables.
constexpr double kScale = 200.0;

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::string to_fen(const nnue::PackedPosition& packed) {
  const nnue::UnpackedPosition unpacked = nnue::unpack_position(packed);
  std::array<char, 64> board{};
  constexpr std::string_view kChars = "PpNnBbRrQqKk";
  for (int i = 0; i < unpacked.count; ++i) {
    board[unpacked.squares[std::size_t(i)]] = kChars[unpacked.pieces[std::size_t(i)]];
  }
  std::ostringstream fen;
  for (int rank = 7; rank >= 0; --rank) {
    int empty = 0;
    for (int file = 0; file < 8; ++file) {
      const char c = board[std::size_t(rank * 8 + file)];
      if (!c) {
        ++empty;
        continue;
      }
      if (empty) fen << empty;
      empty = 0;
      fen << c;
    }
    if (empty) fen << empty;
    if (rank) fen << '/';
  }
  fen << (unpacked.side_to_move == WHITE ? " w - - 0 1" : " b - - 0 1");
  return fen.str();
}

struct Metrics {
  double result_error = 0;  // error cuadratico contra el resultado de la partida
  double score_error = 0;   // error cuadratico contra sigmoide(evaluacion de la busqueda)
  double absolute_cp = 0;   // diferencia media en centipeones con la busqueda
  std::uint64_t count = 0;

  void add(Value eval, const nnue::PackedPosition& sample) {
    const double predicted = sigmoid(double(eval) / kScale);
    const double result = double(sample.result) / 2.0;
    const double score = sigmoid(double(sample.score) / kScale);
    result_error += (predicted - result) * (predicted - result);
    score_error += (predicted - score) * (predicted - score);
    absolute_cp += std::abs(double(eval) - double(sample.score));
    ++count;
  }

  void print(const std::string& name) const {
    std::cout << std::left << std::setw(28) << name << std::fixed << std::setprecision(5)
              << " vs resultado " << result_error / double(count) << " | vs busqueda "
              << score_error / double(count) << " | dif media " << std::setprecision(1)
              << absolute_cp / double(count) << " cp" << std::endl;
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string data_path;
  std::vector<std::string> nets;
  std::uint64_t samples = 200000;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string name = argv[i];
    const std::string value = argv[i + 1];
    if (name == "--data") data_path = value;
    else if (name == "--net") nets.push_back(value);
    else if (name == "--samples") samples = std::stoull(value);
  }
  if (data_path.empty()) {
    std::cerr << "Uso: netcheck --data archivo [--net red]... [--samples N]\n";
    return 1;
  }

  attacks::init();
  zobrist::init();

  std::error_code code;
  const std::uint64_t total =
      std::filesystem::file_size(data_path, code) / sizeof(nnue::PackedPosition);
  if (code || total == 0) {
    std::cerr << "No se pudo leer " << data_path << "\n";
    return 1;
  }

  // Muestra aleatoria repartida por todo el archivo.
  std::vector<nnue::PackedPosition> sample;
  std::vector<Position> positions;
  std::ifstream in(data_path, std::ios::binary);
  Prng rng(12345);
  for (std::uint64_t i = 0; i < samples; ++i) {
    nnue::PackedPosition record;
    in.seekg(std::streamoff(rng.below(total) * sizeof(record)));
    in.read(reinterpret_cast<char*>(&record), sizeof(record));
    if (!in || !nnue::is_valid(record)) continue;
    const auto pos = Position::from_fen(to_fen(record));
    if (!pos) continue;
    sample.push_back(record);
    positions.push_back(*pos);
  }
  std::cout << "Posiciones analizadas: " << sample.size() << "\n";

  Metrics classical_metrics;
  for (std::size_t i = 0; i < sample.size(); ++i) {
    classical_metrics.add(classical::evaluate(positions[i]), sample[i]);
  }
  classical_metrics.print("clasica");

  for (const std::string& path : nets) {
    std::string error;
    if (!nnue::load_network_file(path, &error)) {
      std::cerr << "Error: " << error << "\n";
      return 1;
    }
    const nnue::Network* network = nnue::active_network();
    Metrics metrics;
    for (std::size_t i = 0; i < sample.size(); ++i) {
      metrics.add(nnue::evaluate_from_scratch(*network, positions[i]), sample[i]);
    }
    metrics.print(std::filesystem::path(path).filename().string());
  }
  return 0;
}

// Ajusta desde cero las tablas de material y casillas de la evaluacion clasica con partidas
// del propio motor: una regresion logistica hace que la evaluacion prediga la puntuacion de
// la busqueda. No parte de las tablas actuales, y su salida sustituye a src/eval/pst.h.
//
// Uso: tune_pst --data data/gen3.bin [--data otro.bin] --output src/eval/pst.h
//               [--samples 2000000] [--steps 1500] [--threads 6]
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "core/attacks.h"
#include "core/prng.h"
#include "core/zobrist.h"
#include "eval/classical.h"
#include "nnue/training_data.h"

using namespace ayano;

namespace {

constexpr int kPieceTypes = 6;
constexpr int kTableSize = kPieceTypes * 64;
constexpr int kMaterialMg = 2 * kTableSize;
constexpr int kMaterialEg = kMaterialMg + kPieceTypes;
constexpr int kParameterCount = kMaterialEg + kPieceTypes;

constexpr double kScale = 200.0;  // la misma K con la que se entrenan las redes
constexpr double kInitialRate = 2.0;
constexpr double kFinalRate = 0.05;
constexpr double kBeta1 = 0.9;
constexpr double kBeta2 = 0.999;
constexpr double kEpsilon = 1e-12;
constexpr double kTableDecay = 1e-9;   // acerca al material las casillas con pocos datos
constexpr std::uint64_t kValidationPeriod = 20;  // una posicion de cada 20 no se entrena
constexpr int kReportEverySteps = 100;
constexpr std::uint64_t kMinimumVisits = 200;  // por debajo, la casilla se queda sin bono
constexpr std::size_t kRecordsPerRead = 1 << 16;
constexpr std::uint64_t kMaxCount = 1'000'000'000;
constexpr std::uint64_t kMaxThreads = 256;

using Parameters = std::array<double, kParameterCount>;

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Options {
  std::vector<std::string> data;
  std::string output;
  std::uint64_t samples = 2000000;
  int steps = 1500;
  int threads = int(std::max(1u, std::thread::hardware_concurrency() / 2));
};

// Una pieza es el indice de su casilla en las tablas, o su complemento si es negra.
struct Sample {
  std::array<std::int16_t, 32> features{};
  float rest_mg = 0;
  float rest_eg = 0;
  float phase = 0;
  float target = 0;
  std::int8_t side = 1;
  std::uint8_t count = 0;
};

struct Dataset {
  std::vector<Sample> train;
  std::vector<Sample> validation;
  std::array<std::uint64_t, kTableSize> visits{};
  double current_error = 0;  // error de las tablas compiladas sobre las de validacion
};

Sample make_sample(const Position& pos, const nnue::PackedPosition& record) {
  Sample sample;
  for (int c = WHITE; c <= BLACK; ++c) {
    for (int pt = PAWN; pt <= KING; ++pt) {
      Bitboard pieces = pos.pieces(Color(c), PieceType(pt));
      while (pieces) {
        // Las tablas tienen la octava fila arriba: las blancas invierten la fila.
        const int index = pt * 64 + (pop_lsb(pieces) ^ (c == WHITE ? 56 : 0));
        sample.features[sample.count++] = std::int16_t(c == WHITE ? index : ~index);
      }
    }
  }
  const classical::PositionalTerms terms = classical::positional_terms(pos);
  sample.rest_mg = float(terms.mg);
  sample.rest_eg = float(terms.eg);
  sample.phase = float(classical::game_phase(pos)) / float(classical::kMaxPhase);
  sample.target = float(sigmoid(double(record.score) / kScale));
  sample.side = pos.side_to_move() == WHITE ? 1 : -1;
  return sample;
}

void add_record(const nnue::PackedPosition& record, Dataset& dataset, std::uint64_t& kept) {
  if (!nnue::is_valid(record)) return;
  const auto pos = Position::from_fen(nnue::to_fen(record));
  if (!pos) return;
  const Sample sample = make_sample(*pos, record);
  if (++kept % kValidationPeriod == 0) {
    const double error =
        sigmoid(double(classical::evaluate(*pos)) / kScale) - double(sample.target);
    dataset.current_error += error * error;
    dataset.validation.push_back(sample);
    return;
  }
  for (int i = 0; i < sample.count; ++i) {
    const int feature = sample.features[std::size_t(i)];
    ++dataset.visits[std::size_t(feature < 0 ? ~feature : feature)];
  }
  dataset.train.push_back(sample);
}

// Recorre los archivos una vez y conserva una muestra uniforme de todas las posiciones.
bool load_dataset(const Options& options, Dataset& dataset) {
  std::uint64_t total = 0;
  for (const std::string& path : options.data) {
    std::error_code code;
    const std::uintmax_t size = std::filesystem::file_size(path, code);
    if (code) {
      std::cerr << "No se pudo leer " << path << "\n";
      return false;
    }
    total += size / sizeof(nnue::PackedPosition);
  }
  if (total == 0) {
    std::cerr << "Los datos estan vacios\n";
    return false;
  }

  Prng rng(20261104);
  std::uint64_t kept = 0;
  std::vector<nnue::PackedPosition> block(kRecordsPerRead);
  for (const std::string& path : options.data) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      std::cerr << "No se pudo abrir " << path << "\n";
      return false;
    }
    while (in) {
      in.read(reinterpret_cast<char*>(block.data()),
              std::streamsize(block.size() * sizeof(nnue::PackedPosition)));
      const auto records = std::size_t(in.gcount()) / sizeof(nnue::PackedPosition);
      for (std::size_t i = 0; i < records; ++i) {
        if (rng.below(total) < options.samples) add_record(block[i], dataset, kept);
      }
    }
    if (in.bad()) {
      std::cerr << "Error de lectura en " << path << "\n";
      return false;
    }
  }
  if (dataset.train.empty() || dataset.validation.empty()) {
    std::cerr << "Hacen falta al menos " << kValidationPeriod << " posiciones validas\n";
    return false;
  }
  return true;
}

template <typename Visit>
void for_each_feature(const Sample& sample, Visit visit) {
  for (int i = 0; i < sample.count; ++i) {
    const int feature = sample.features[std::size_t(i)];
    const int index = feature < 0 ? ~feature : feature;
    visit(index, index / 64, feature < 0 ? -1.0 : 1.0);
  }
}

double predict(const Sample& sample, const Parameters& parameters) {
  double mg = sample.rest_mg;
  double eg = sample.rest_eg;
  for_each_feature(sample, [&](int index, int piece, double sign) {
    mg += sign * (parameters[std::size_t(index)] + parameters[std::size_t(kMaterialMg + piece)]);
    eg += sign * (parameters[std::size_t(kTableSize + index)] +
                  parameters[std::size_t(kMaterialEg + piece)]);
  });
  const double white_value = mg * double(sample.phase) + eg * (1.0 - double(sample.phase));
  return sigmoid((double(sample.side) * white_value + classical::kTempo) / kScale);
}

void add_gradient(const Sample& sample, double predicted, Parameters& gradient) {
  const double error = predicted - double(sample.target);
  const double slope =
      2.0 * error * predicted * (1.0 - predicted) * double(sample.side) / kScale;
  const double mg = slope * double(sample.phase);
  const double eg = slope * (1.0 - double(sample.phase));
  for_each_feature(sample, [&](int index, int piece, double sign) {
    gradient[std::size_t(index)] += sign * mg;
    gradient[std::size_t(kTableSize + index)] += sign * eg;
    gradient[std::size_t(kMaterialMg + piece)] += sign * mg;
    gradient[std::size_t(kMaterialEg + piece)] += sign * eg;
  });
}

// Gradiente medio sobre todas las posiciones, repartidas entre los hilos.
Parameters full_gradient(const std::vector<Sample>& samples, const Parameters& parameters,
                         int thread_count) {
  std::vector<Parameters> partial(std::size_t(thread_count), Parameters{});
  std::vector<std::thread> threads;
  const std::size_t slice = (samples.size() + std::size_t(thread_count) - 1) / std::size_t(thread_count);
  for (int t = 0; t < thread_count; ++t) {
    threads.emplace_back([&, t] {
      const std::size_t begin = std::min(samples.size(), std::size_t(t) * slice);
      const std::size_t end = std::min(samples.size(), begin + slice);
      for (std::size_t i = begin; i < end; ++i) {
        add_gradient(samples[i], predict(samples[i], parameters), partial[std::size_t(t)]);
      }
    });
  }
  for (std::thread& thread : threads) thread.join();

  Parameters gradient{};
  for (const Parameters& part : partial) {
    for (std::size_t i = 0; i < gradient.size(); ++i) gradient[i] += part[i];
  }
  for (double& value : gradient) value /= double(samples.size());
  for (std::size_t i = 0; i < std::size_t(kMaterialMg); ++i) {
    gradient[i] += kTableDecay * parameters[i];
  }
  return gradient;
}

double mean_error(const std::vector<Sample>& samples, const Parameters& parameters) {
  double total = 0;
  for (const Sample& sample : samples) {
    const double error = predict(sample, parameters) - double(sample.target);
    total += error * error;
  }
  return total / double(samples.size());
}

Parameters fit(const Dataset& dataset, const Options& options) {
  Parameters parameters{};
  Parameters first_moment{};
  Parameters second_moment{};
  for (int step = 1; step <= options.steps; ++step) {
    const Parameters gradient = full_gradient(dataset.train, parameters, options.threads);
    const double progress = double(step - 1) / double(std::max(1, options.steps - 1));
    const double rate = kInitialRate + (kFinalRate - kInitialRate) * progress;
    for (std::size_t i = 0; i < parameters.size(); ++i) {
      first_moment[i] = kBeta1 * first_moment[i] + (1 - kBeta1) * gradient[i];
      second_moment[i] = kBeta2 * second_moment[i] + (1 - kBeta2) * gradient[i] * gradient[i];
      const double mean = first_moment[i] / (1 - std::pow(kBeta1, step));
      const double variance = second_moment[i] / (1 - std::pow(kBeta2, step));
      parameters[i] -= rate * mean / (std::sqrt(variance) + kEpsilon);
    }
    if (step % kReportEverySteps == 0 || step == options.steps) {
      std::cout << "Paso " << step << ": error de validacion " << std::fixed
                << std::setprecision(6) << mean_error(dataset.validation, parameters)
                << std::endl;
    }
  }
  return parameters;
}

int clear_rare_squares(const Dataset& dataset, Parameters& parameters) {
  int cleared = 0;
  for (std::size_t i = 0; i < std::size_t(kTableSize); ++i) {
    if (dataset.visits[i] >= kMinimumVisits) continue;
    if (dataset.visits[i] > 0) ++cleared;
    parameters[i] = 0;
    parameters[std::size_t(kTableSize) + i] = 0;
  }
  return cleared;
}

void write_material(std::ostream& out, const char* name, const Parameters& parameters, int base) {
  out << "inline constexpr std::array<int, 6> " << name << " = {";
  for (int pt = 0; pt < kPieceTypes; ++pt) {
    out << std::lround(parameters[std::size_t(base + pt)]) << (pt + 1 < kPieceTypes ? ", " : "");
  }
  out << "};\n";
}

void write_tables(std::ostream& out, const char* name, const Parameters& parameters, int base) {
  constexpr std::array<const char*, kPieceTypes> kLabels = {"Peon",  "Caballo", "Alfil",
                                                            "Torre", "Dama",    "Rey"};
  out << "\ninline constexpr std::array<PstTable, 6> " << name << " = {{\n";
  for (int pt = 0; pt < kPieceTypes; ++pt) {
    out << "    // " << kLabels[std::size_t(pt)] << "\n    {";
    for (int square = 0; square < 64; ++square) {
      const double value = parameters[std::size_t(base + pt * 64 + square)];
      out << std::setw(4) << std::lround(value);
      if (square == 63) break;
      out << (square % 8 == 7 ? ",\n     " : ", ");
    }
    out << "},\n";
  }
  out << "}};\n";
}

bool write_header(const std::string& path, const Parameters& parameters) {
  std::ofstream out(path, std::ios::binary);
  out << "// Tablas de material y casillas para el medio juego y el final. Las genera\n"
      << "// tools/tune_pst a partir de partidas del propio motor: no se editan a mano.\n"
      << "// Estan escritas con la octava fila arriba, tal como se ve el tablero.\n"
      << "#pragma once\n\n#include <array>\n\nnamespace ayano::classical {\n\n"
      << "using PstTable = std::array<int, 64>;\n\n";
  write_material(out, "kMaterialMg", parameters, kMaterialMg);
  write_material(out, "kMaterialEg", parameters, kMaterialEg);
  write_tables(out, "kPstMg", parameters, 0);
  write_tables(out, "kPstEg", parameters, kTableSize);
  out << "\n}  // namespace ayano::classical\n";
  return bool(out);
}

// Lee un entero positivo y rechaza signos, decimales y texto sobrante.
std::optional<std::uint64_t> parse_count(const std::string& text, std::uint64_t maximum) {
  std::uint64_t value = 0;
  const char* end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(text.data(), end, value);
  if (ec != std::errc() || ptr != end || value == 0 || value > maximum) return std::nullopt;
  return value;
}

bool parse_options(int argc, char** argv, Options& options) {
  if (argc % 2 == 0) return false;  // cada opcion lleva su valor
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string name = argv[i];
    const std::string value = argv[i + 1];
    if (name == "--data") {
      options.data.push_back(value);
      continue;
    }
    if (name == "--output") {
      options.output = value;
      continue;
    }
    const auto number = parse_count(value, name == "--threads" ? kMaxThreads : kMaxCount);
    if (!number) return false;
    if (name == "--samples") options.samples = *number;
    else if (name == "--steps") options.steps = int(*number);
    else if (name == "--threads") options.threads = int(*number);
    else return false;
  }
  return !options.data.empty() && !options.output.empty();
}

int run(const Options& options) {
  Dataset dataset;
  if (!load_dataset(options, dataset)) return 1;
  std::cout << "Posiciones: " << dataset.train.size() << " para ajustar y "
            << dataset.validation.size() << " para validar\n"
            << "Error de las tablas compiladas: " << std::fixed << std::setprecision(6)
            << dataset.current_error / double(dataset.validation.size()) << std::endl;

  Parameters parameters = fit(dataset, options);
  const int cleared = clear_rare_squares(dataset, parameters);
  std::cout << "Casillas sin datos suficientes: " << cleared << "\n"
            << "Error final de validacion: " << mean_error(dataset.validation, parameters)
            << "\n";
  if (!write_header(options.output, parameters)) {
    std::cerr << "No se pudo escribir " << options.output << "\n";
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, options)) {
    std::cerr << "Uso: tune_pst --data archivo [--data otro] --output pst.h "
                 "[--samples N] [--steps N] [--threads N]\n";
    return 1;
  }
  attacks::init();
  zobrist::init();
  try {
    return run(options);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}

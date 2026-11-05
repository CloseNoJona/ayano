// Entrena una red a partir de archivos de datagen.
//
// Uso: train --data data/gen0.bin [--data otro.bin] --out nets/gen1 --epochs 15
//            --threads 10 [--lr 0.001] [--final-lr 0.00002] [--wdl 0] [--scale 200]
//            [--batch 4096] [--seed 1] [--max-positions N]
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "core/attacks.h"
#include "core/zobrist.h"
#include "train/trainer.h"

using namespace ayano;

namespace {

struct Options {
  train::TrainConfig config;
  std::vector<std::string> inputs;
  std::uint64_t max_positions = 0;
};

// Exige leer todo el texto: `stoi` convertiria "0.5" en 0 sin avisar.
template <typename Number>
Number parse_number(const std::string& text) {
  Number value{};
  const char* const end = text.data() + text.size();
  const auto [parsed_to, code] = std::from_chars(text.data(), end, value);
  if (code != std::errc{} || parsed_to != end) {
    throw std::invalid_argument("numero invalido: " + text);
  }
  return value;
}

void apply_option(Options& options, const std::string& name, const std::string& value) {
  train::TrainConfig& config = options.config;
  if (name == "--data") options.inputs.push_back(value);
  else if (name == "--out") config.output_prefix = value;
  else if (name == "--epochs") config.epochs = parse_number<int>(value);
  else if (name == "--threads") config.threads = parse_number<int>(value);
  else if (name == "--lr") config.learning_rate = parse_number<float>(value);
  else if (name == "--final-lr") config.final_learning_rate = parse_number<float>(value);
  else if (name == "--wdl") config.wdl_weight = parse_number<float>(value);
  else if (name == "--scale") config.eval_scale = parse_number<int>(value);
  else if (name == "--batch") config.batch_size = parse_number<std::size_t>(value);
  else if (name == "--seed") config.seed = parse_number<std::uint64_t>(value);
  else if (name == "--max-positions") options.max_positions = parse_number<std::uint64_t>(value);
  else throw std::invalid_argument("opcion desconocida: " + name);
}

bool parse_args(int argc, char** argv, Options& options) {
  try {
    if (argc % 2 == 0) throw std::invalid_argument("cada opcion necesita un valor");
    for (int i = 1; i + 1 < argc; i += 2) apply_option(options, argv[i], argv[i + 1]);
  } catch (const std::invalid_argument& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return false;
  }
  if (options.inputs.empty()) {
    std::cerr << "Uso: train --data archivo --out prefijo --epochs N --threads N\n";
    return false;
  }
  if (const auto problem = train::config_error(options.config)) {
    std::cerr << "Error: " << *problem << "\n";
    return false;
  }
  return true;
}

// Falla ahora y no tras la primera epoca, que puede tardar mucho.
bool prepare_output_folder(const std::string& prefix) {
  const std::filesystem::path folder = std::filesystem::path(prefix).parent_path();
  if (folder.empty()) return true;
  std::error_code error;
  std::filesystem::create_directories(folder, error);
  if (error) std::cerr << "Error: no se pudo crear " << folder.string() << "\n";
  return !error;
}

bool load_samples(const std::string& path, std::vector<nnue::PackedPosition>& samples,
                  std::uint64_t limit) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error) {
    std::cerr << "No se pudo leer " << path << "\n";
    return false;
  }
  if (size % sizeof(nnue::PackedPosition) != 0) {
    std::cerr << "Aviso: " << path << " tiene un registro incompleto al final\n";
  }
  std::uint64_t count = size / sizeof(nnue::PackedPosition);
  if (limit > 0 && samples.size() + count > limit) count = limit - samples.size();

  const std::size_t offset = samples.size();
  samples.resize(offset + std::size_t(count));
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char*>(samples.data() + offset),
          std::streamsize(count * sizeof(nnue::PackedPosition)));
  return bool(in);
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_args(argc, argv, options)) return 1;
  if (!prepare_output_folder(options.config.output_prefix)) return 1;

  attacks::init();
  zobrist::init();

  std::vector<nnue::PackedPosition> samples;
  for (const std::string& path : options.inputs) {
    if (!load_samples(path, samples, options.max_positions)) return 1;
  }
  const std::size_t discarded = std::erase_if(
      samples, [](const nnue::PackedPosition& sample) { return !nnue::is_valid(sample); });
  if (discarded > 0) std::cerr << "Aviso: " << discarded << " registros invalidos\n";
  std::cout << "Posiciones: " << samples.size() << std::endl;

  auto network = std::make_unique<train::FloatNetwork>();
  network->randomize(options.config.seed);
  std::string error;
  if (!train::train(*network, samples, options.config, &error)) {
    std::cerr << "Error: " << error << "\n";
    return 1;
  }
  return 0;
}

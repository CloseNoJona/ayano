#include "nnue/network.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace enroque::nnue {

namespace {

constexpr std::array<char, 4> kMagic = {'E', 'N', 'Q', '2'};

// Cabecera de archivo: permite rechazar redes de otra arquitectura en vez de
// leer basura.
struct Header {
  std::array<char, 4> magic;
  std::int32_t features;
  std::int32_t hidden;
  std::int32_t buckets;
  std::int32_t eval_scale;
};

constexpr std::size_t kFileSize =
    sizeof(Header) + sizeof(Network::feature_weights) + sizeof(Network::feature_bias) +
    sizeof(Network::output_weights) + sizeof(Network::output_bias);
constexpr std::int16_t kMaxOutputWeight = 128;

std::unique_ptr<Network> loaded_network;

template <typename Array>
void copy_section(Array& target, const std::uint8_t*& cursor) {
  std::memcpy(target.data(), cursor, sizeof(target));
  cursor += sizeof(target);
}

template <typename Array>
void write_section(const Array& source, std::ofstream& out) {
  out.write(reinterpret_cast<const char*>(source.data()), std::streamsize(sizeof(source)));
}

bool fail(std::string* error, const std::string& message) {
  if (error) *error = message;
  return false;
}

}  // namespace

const Network* active_network() { return loaded_network.get(); }

bool parse_network(std::span<const std::uint8_t> bytes, Network& network, std::string* error) {
  if (bytes.size() != kFileSize) return fail(error, "tamano de red incorrecto");

  Header header{};
  std::memcpy(&header, bytes.data(), sizeof(header));
  if (header.magic != kMagic) return fail(error, "el archivo no es una red de Enroque");
  if (header.features != kFeatures || header.hidden != kHidden ||
      header.buckets != kOutputBuckets) {
    return fail(error, "la red tiene otra arquitectura");
  }
  if (header.eval_scale < 1 || header.eval_scale > kMaxEvalScale) {
    return fail(error, "escala de evaluacion invalida");
  }
  network.eval_scale = header.eval_scale;

  const std::uint8_t* cursor = bytes.data() + sizeof(Header);
  copy_section(network.feature_weights, cursor);
  copy_section(network.feature_bias, cursor);
  copy_section(network.output_weights, cursor);
  copy_section(network.output_bias, cursor);

  // El producto de 16 bits de SCReLU solo es seguro con pesos en este rango.
  const auto out_of_range = [](std::int16_t w) { return w < -kMaxOutputWeight || w > kMaxOutputWeight; };
  if (std::any_of(network.output_weights.begin(), network.output_weights.end(), out_of_range)) {
    return fail(error, "la red tiene pesos de salida fuera de rango");
  }
  return true;
}

bool load_network_file(const std::string& path, std::string* error) {
  // Se comprueba el tamano antes de reservar memoria para el contenido.
  std::error_code code;
  const std::uintmax_t size = std::filesystem::file_size(path, code);
  if (code) return fail(error, "no se pudo abrir " + path);
  if (size != kFileSize) return fail(error, "tamano de red incorrecto: " + path);

  std::ifstream in(path, std::ios::binary);
  std::vector<std::uint8_t> bytes(kFileSize);
  if (!in.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(kFileSize))) {
    return fail(error, "no se pudo leer " + path);
  }

  auto network = std::make_unique<Network>();
  if (!parse_network(bytes, *network, error)) return false;
  loaded_network = std::move(network);
  return true;
}

bool save_network_file(const Network& network, const std::string& path, std::string* error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return fail(error, "no se pudo crear " + path);
  const Header header{kMagic, kFeatures, kHidden, kOutputBuckets, network.eval_scale};
  out.write(reinterpret_cast<const char*>(&header), std::streamsize(sizeof(header)));
  write_section(network.feature_weights, out);
  write_section(network.feature_bias, out);
  write_section(network.output_weights, out);
  write_section(network.output_bias, out);
  out.close();  // el ultimo volcado tambien puede fallar, por ejemplo con el disco lleno
  return !out.fail() || fail(error, "no se pudo escribir " + path);
}

bool init(std::string* error) {
  const std::span<const std::uint8_t> embedded = embedded_network();
  if (embedded.empty()) return true;
  auto network = std::make_unique<Network>();
  if (!parse_network(embedded, *network, error)) return false;
  loaded_network = std::move(network);
  return true;
}

}  // namespace enroque::nnue

#include "nnue/network.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace ayano::nnue {

namespace {

constexpr std::array<char, 4> kMagicLinear = {'E', 'N', 'Q', '2'};
constexpr std::array<char, 4> kMagicDeep = {'E', 'N', 'Q', '3'};

// Cabecera de archivo: permite rechazar redes de otra arquitectura en vez de
// leer basura.
struct Header {
  std::array<char, 4> magic;
  std::int32_t features;
  std::int32_t hidden;
  std::int32_t buckets;
  std::int32_t eval_scale;
};

// La cabeza profunda anade el tamano de sus dos capas intermedias.
struct DeepSizes {
  std::int32_t l1;
  std::int32_t l2;
};

constexpr std::size_t kInputLayerSize = sizeof(Network::feature_weights) + sizeof(Network::feature_bias);
constexpr std::size_t kLinearFileSize =
    sizeof(Header) + kInputLayerSize + sizeof(Network::output_weights) + sizeof(Network::output_bias);
constexpr std::size_t kDeepFileSize =
    sizeof(Header) + sizeof(DeepSizes) + kInputLayerSize + sizeof(Network::l1_weights) +
    sizeof(Network::l1_bias) + sizeof(Network::l2_weights) + sizeof(Network::l2_bias) +
    sizeof(Network::l3_weights) + sizeof(Network::l3_bias);
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

// Una red entrenada no pasa de 2 en valor absoluto. Con este limite la salida tampoco puede
// desbordar el entero de la evaluacion, ni siquiera con la escala maxima.
constexpr float kMaxDeepValue = 10000.0f;

template <typename Array>
bool all_bounded(const Array& values) {
  return std::all_of(values.begin(), values.end(),
                     [](float v) { return std::isfinite(v) && std::abs(v) <= kMaxDeepValue; });
}

bool fail(std::string* error, const std::string& message) {
  if (error) *error = message;
  return false;
}

bool parse_linear_head(const std::uint8_t* cursor, Network& network, std::string* error) {
  copy_section(network.output_weights, cursor);
  copy_section(network.output_bias, cursor);
  // El producto de 16 bits de SCReLU solo es seguro con pesos en este rango.
  const auto out_of_range = [](std::int16_t w) { return w < -kMaxOutputWeight || w > kMaxOutputWeight; };
  if (std::any_of(network.output_weights.begin(), network.output_weights.end(), out_of_range)) {
    return fail(error, "la red tiene pesos de salida fuera de rango");
  }
  return true;
}

bool parse_deep_head(const std::uint8_t* cursor, Network& network, std::string* error) {
  copy_section(network.l1_weights, cursor);
  copy_section(network.l1_bias, cursor);
  copy_section(network.l2_weights, cursor);
  copy_section(network.l2_bias, cursor);
  copy_section(network.l3_weights, cursor);
  copy_section(network.l3_bias, cursor);
  // Un NaN o un infinito se propagaria a todas las evaluaciones sin otro sintoma.
  if (!all_bounded(network.l1_bias) || !all_bounded(network.l2_weights) ||
      !all_bounded(network.l2_bias) || !all_bounded(network.l3_weights) ||
      !all_bounded(network.l3_bias)) {
    return fail(error, "la red tiene valores no finitos o fuera de rango");
  }
  return true;
}

}  // namespace

const Network* active_network() { return loaded_network.get(); }

bool parse_network(std::span<const std::uint8_t> bytes, Network& network, std::string* error) {
  if (bytes.size() < sizeof(Header)) return fail(error, "tamano de red incorrecto");

  Header header{};
  std::memcpy(&header, bytes.data(), sizeof(header));
  Head head = Head::Linear;
  if (header.magic == kMagicDeep) {
    head = Head::Deep;
  } else if (header.magic != kMagicLinear) {
    return fail(error, "el archivo no es una red de Ayano");
  }
  if (bytes.size() != (head == Head::Deep ? kDeepFileSize : kLinearFileSize)) {
    return fail(error, "tamano de red incorrecto");
  }
  if (header.features != kFeatures || header.hidden != kHidden ||
      header.buckets != kOutputBuckets) {
    return fail(error, "la red tiene otra arquitectura");
  }
  if (header.eval_scale < 1 || header.eval_scale > kMaxEvalScale) {
    return fail(error, "escala de evaluacion invalida");
  }

  const std::uint8_t* cursor = bytes.data() + sizeof(Header);
  if (head == Head::Deep) {
    DeepSizes sizes{};
    std::memcpy(&sizes, cursor, sizeof(sizes));
    cursor += sizeof(sizes);
    if (sizes.l1 != kDeepL1 || sizes.l2 != kDeepL2) return fail(error, "la red tiene otra arquitectura");
  }
  network.eval_scale = header.eval_scale;
  network.head = head;
  copy_section(network.feature_weights, cursor);
  copy_section(network.feature_bias, cursor);
  return head == Head::Deep ? parse_deep_head(cursor, network, error)
                            : parse_linear_head(cursor, network, error);
}

bool load_network_file(const std::string& path, std::string* error) {
  // Se comprueba el tamano antes de reservar memoria para el contenido.
  std::error_code code;
  const std::uintmax_t size = std::filesystem::file_size(path, code);
  if (code) return fail(error, "no se pudo abrir " + path);
  if (size != kLinearFileSize && size != kDeepFileSize) {
    return fail(error, "tamano de red incorrecto: " + path);
  }

  std::ifstream in(path, std::ios::binary);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (!in.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()))) {
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
  const bool deep = network.head == Head::Deep;
  const Header header{deep ? kMagicDeep : kMagicLinear, kFeatures, kHidden, kOutputBuckets,
                      network.eval_scale};
  out.write(reinterpret_cast<const char*>(&header), std::streamsize(sizeof(header)));
  if (deep) {
    const DeepSizes sizes{kDeepL1, kDeepL2};
    out.write(reinterpret_cast<const char*>(&sizes), std::streamsize(sizeof(sizes)));
  }
  write_section(network.feature_weights, out);
  write_section(network.feature_bias, out);
  if (deep) {
    write_section(network.l1_weights, out);
    write_section(network.l1_bias, out);
    write_section(network.l2_weights, out);
    write_section(network.l2_bias, out);
    write_section(network.l3_weights, out);
    write_section(network.l3_bias, out);
  } else {
    write_section(network.output_weights, out);
    write_section(network.output_bias, out);
  }
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

}  // namespace ayano::nnue

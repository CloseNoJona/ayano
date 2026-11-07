// Convierte un archivo de red en un fichero C++ con sus bytes. Los escribe como enteros de
// 64 bits porque el compilador tarda mucho mas con una tabla de bytes de este tamano.
// Uso: embed_net red.nnue embedded_net.cpp
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {

constexpr std::size_t kWordBytes = sizeof(std::uint64_t);
constexpr std::size_t kWordsPerLine = 8;

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "uso: embed_net red.nnue salida.cpp\n";
    return 2;
  }
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) {
    std::cerr << "no se pudo abrir " << argv[1] << "\n";
    return 1;
  }
  std::vector<char> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  const std::size_t size = bytes.size();
  bytes.resize((size + kWordBytes - 1) / kWordBytes * kWordBytes, 0);

  std::ofstream out(argv[2]);
  out << "// Generado automaticamente a partir de la red: no editar.\n"
      << "#include \"nnue/network.h\"\n\nnamespace ayano::nnue {\n\nnamespace {\n"
      << "constexpr std::size_t kEmbeddedBytes = " << size << ";\n"
      << "alignas(64) constexpr std::uint64_t kEmbeddedNetwork[] = {\n"
      << std::hex;
  for (std::size_t offset = 0, count = 0; offset < bytes.size(); offset += kWordBytes, ++count) {
    std::uint64_t word = 0;
    std::memcpy(&word, bytes.data() + offset, kWordBytes);
    out << "0x" << word << "ull," << ((count + 1) % kWordsPerLine == 0 ? "\n" : "");
  }
  out << "};\n}  // namespace\n\n"
      << "std::span<const std::uint8_t> embedded_network() {\n"
      << "  return {reinterpret_cast<const std::uint8_t*>(kEmbeddedNetwork), kEmbeddedBytes};\n"
      << "}\n\n}  // namespace ayano::nnue\n";
  out.close();
  if (out.fail()) {
    std::cerr << "no se pudo escribir " << argv[2] << "\n";
    return 1;
  }
  return 0;
}

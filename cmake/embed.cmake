# Convierte un archivo binario en un fichero C++ con sus bytes.
# Uso: cmake -DINPUT=red.nnue -DOUTPUT=embedded_net.cpp -P embed.cmake
file(READ "${INPUT}" hex_bytes HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," byte_list "${hex_bytes}")
file(WRITE "${OUTPUT}"
"// Generado automaticamente a partir de la red: no editar.
#include \"nnue/network.h\"

namespace enroque::nnue {

namespace {
constexpr std::uint8_t kEmbeddedNetwork[] = {${byte_list}};
}  // namespace

std::span<const std::uint8_t> embedded_network() { return kEmbeddedNetwork; }

}  // namespace enroque::nnue
")

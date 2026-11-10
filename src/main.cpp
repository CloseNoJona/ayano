#include <iostream>
#include <string>

#include "board/cuckoo.h"
#include "core/attacks.h"
#include "core/cpu.h"
#include "core/zobrist.h"
#include "nnue/network.h"
#include "uci/uci.h"

int main(int argc, char** argv) {
  if (!ayano::cpu_supports_this_build()) {
    std::cout << "info string error: esta CPU no tiene AVX2 o BMI2; usa la variante portable"
              << std::endl;
    return 1;
  }

  ayano::attacks::init();
  ayano::zobrist::init();
  ayano::cuckoo::init();
  // Sin este aviso el motor pasaria a la evaluacion clasica en silencio.
  std::string error;
  if (!ayano::nnue::init(&error)) {
    std::cout << "info string error: red integrada rechazada: " << error << std::endl;
  } else if (!ayano::nnue::active_network()) {
    std::cout << "info string sin red integrada: se usa la evaluacion clasica" << std::endl;
  }

  ayano::Uci uci;

  // Los argumentos se tratan como un comando, por ejemplo: ayano bench 13
  if (argc > 1) {
    std::string command;
    for (int i = 1; i < argc; ++i) {
      if (i > 1) command += ' ';
      command += argv[i];
    }
    uci.execute(command);
    uci.execute("quit");
    return 0;
  }

  uci.loop();
  return 0;
}

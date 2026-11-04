#include <iostream>
#include <string>

#include "core/attacks.h"
#include "core/cpu.h"
#include "core/zobrist.h"
#include "nnue/network.h"
#include "uci/uci.h"

int main(int argc, char** argv) {
  if (!enroque::cpu_supports_this_build()) {
    std::cout << "info string error: esta CPU no tiene AVX2 o BMI2; "
                 "compila la variante portable (scripts\\build.cmd portable)" << std::endl;
    return 1;
  }

  enroque::attacks::init();
  enroque::zobrist::init();
  // Sin este aviso el motor pasaria a la evaluacion clasica en silencio.
  std::string error;
  if (!enroque::nnue::init(&error)) {
    std::cout << "info string error: red integrada rechazada: " << error << std::endl;
  } else if (!enroque::nnue::active_network()) {
    std::cout << "info string sin red integrada: se usa la evaluacion clasica" << std::endl;
  }

  enroque::Uci uci;

  // Los argumentos se tratan como un comando, por ejemplo: enroque bench 13
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

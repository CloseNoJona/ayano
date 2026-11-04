// Mide cuantas posiciones de una suite EPD resuelve el motor en un tiempo fijo.
//
// Uso: solve --epd tests/data/bratko-kopec.epd --ms 5000 --threads 4 [--net red]
//            [--classical]
// Formato de cada linea: <FEN de 4 campos> bm <jugadas UCI>; id "nombre";
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "board/movegen.h"
#include "core/attacks.h"
#include "core/zobrist.h"
#include "nnue/network.h"
#include "search/engine.h"

using namespace enroque;

namespace {

struct TestPosition {
  Position pos;
  std::vector<Move> best_moves;
  std::string name;
};

bool parse_line(const std::string& line, TestPosition& test) {
  const std::size_t marker = line.find(" bm ");
  if (marker == std::string::npos) return false;
  const auto pos = Position::from_fen(line.substr(0, marker) + " 0 1");
  if (!pos) return false;
  test.pos = *pos;

  const std::size_t end = line.find(';', marker);
  std::istringstream moves(line.substr(marker + 4, end - marker - 4));
  std::string text;
  test.best_moves.clear();
  while (moves >> text) {
    const Move m = parse_uci_move(test.pos, text);
    if (m == Move::none()) return false;
    test.best_moves.push_back(m);
  }

  const std::size_t id = line.find("id \"");
  test.name = id == std::string::npos ? line.substr(0, 20)
                                      : line.substr(id + 4, line.find('"', id + 4) - id - 4);
  return !test.best_moves.empty();
}

}  // namespace

int main(int argc, char** argv) {
  std::string epd_path;
  std::string net_path;
  std::int64_t think_ms = 1000;
  int threads = 1;
  bool classical = false;

  for (int i = 1; i < argc; ++i) {
    const std::string name = argv[i];
    if (name == "--classical") {
      classical = true;
    } else if (i + 1 < argc) {
      const std::string value = argv[++i];
      if (name == "--epd") epd_path = value;
      else if (name == "--net") net_path = value;
      else if (name == "--ms") think_ms = std::stoll(value);
      else if (name == "--threads") threads = std::stoi(value);
    }
  }
  if (epd_path.empty()) {
    std::cerr << "Uso: solve --epd archivo --ms N --threads N [--net red] [--classical]\n";
    return 1;
  }

  attacks::init();
  zobrist::init();
  std::string error;
  if (!nnue::init(&error)) std::cerr << "Aviso: red integrada rechazada: " << error << "\n";
  if (!net_path.empty()) {
    if (!nnue::load_network_file(net_path, &error)) {
      std::cerr << "Error: " << error << "\n";
      return 1;
    }
  }

  std::ifstream in(epd_path);
  if (!in) {
    std::cerr << "No se pudo abrir " << epd_path << "\n";
    return 1;
  }

  Engine engine;
  engine.set_threads(threads);
  engine.set_hash_mb(256);
  engine.set_use_nnue(!classical);

  int solved = 0;
  int total = 0;
  std::string line;
  while (std::getline(in, line)) {
    TestPosition test;
    if (!parse_line(line, test)) continue;

    engine.new_game();
    SearchLimits limits;
    limits.move_time_ms = think_ms;
    const SearchResult result = engine.search_blocking(test.pos, {}, limits);

    const bool ok = std::find(test.best_moves.begin(), test.best_moves.end(), result.best) !=
                    test.best_moves.end();
    solved += ok;
    ++total;
    std::cout << (ok ? "[ok]    " : "[fallo] ") << test.name << " juega "
              << result.best.to_uci() << " (prof. " << result.depth << ", " << result.score
              << " cp)" << std::endl;
  }

  std::cout << "Resueltas: " << solved << "/" << total << std::endl;
  return 0;
}

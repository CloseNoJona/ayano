// Perft cuenta los nodos del arbol de movimientos legales. Los valores de
// referencia son los publicados en la Chess Programming Wiki: cualquier error
// de reglas (enroque, al paso, promocion, clavadas) cambia el numero.
#include <cstdint>
#include <vector>

#include "board/perft.h"
#include "board/position.h"
#include "test_framework.h"

using namespace ayano;

namespace {

struct PerftCase {
  const char* fen;
  std::vector<std::uint64_t> nodes;  // nodes[i] = perft(i + 1)
};

const std::vector<PerftCase>& perft_cases() {
  static const std::vector<PerftCase> cases = {
      {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
       {20, 400, 8902, 197281, 4865609, 119060324}},
      {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
       {48, 2039, 97862, 4085603, 193690690}},
      {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
       {14, 191, 2812, 43238, 674624, 11030083, 178633661}},
      {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
       {6, 264, 9467, 422333, 15833292}},
      {"r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
       {6, 264, 9467, 422333, 15833292}},
      {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
       {44, 1486, 62379, 2103487, 89941194}},
      {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
       {46, 2079, 89890, 3894594, 164075551}},
  };
  return cases;
}

void run_cases(std::uint64_t max_nodes) {
  for (const PerftCase& test_case : perft_cases()) {
    const auto pos = Position::from_fen(test_case.fen);
    CHECK(pos.has_value());
    if (!pos) continue;
    for (std::size_t i = 0; i < test_case.nodes.size(); ++i) {
      if (test_case.nodes[i] > max_nodes) break;
      CHECK_EQ(perft(*pos, static_cast<int>(i) + 1), test_case.nodes[i]);
    }
  }
}

}  // namespace

TEST(perft_hasta_cinco_millones_de_nodos) { run_cases(5'000'000); }

SLOW_TEST(perft_completo) { run_cases(200'000'000); }

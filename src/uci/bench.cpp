#include "uci/bench.h"

#include <array>
#include <chrono>
#include <optional>
#include <string_view>

namespace enroque {

namespace {

constexpr std::array<std::string_view, 20> kBenchFens = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 9",
    "2rq1rk1/pb1n1ppp/1p2pn2/2bp4/3P1B2/2NBPN2/PPQ2PPP/2R2RK1 w - - 0 13",
    "r2qr1k1/1p1bbppp/p1np1n2/4p3/4P3/1NN1BP2/PPPQB1PP/R4RK1 w - - 3 13",
    "3r2k1/pp3pp1/2p1bn1p/8/2P5/1P2BP2/P4KPP/3R4 w - - 0 24",
    "6k1/5p1p/4p1p1/3pP3/1r1P1P2/1P4P1/5K1P/2R5 b - - 2 33",
    "8/5pk1/5p1p/2R5/5K2/1r4P1/7P/8 w - - 8 50",
    "8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1",
    "8/8/1p1k4/1P6/2K5/8/8/8 w - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "5rk1/1ppb3p/p1pb4/6q1/3P1p1r/2P1R2P/PP1BQ1P1/5RKN w - - 0 1",
    "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
    "1r3rk1/1pnnq1bR/p1pp2B1/P2P1p2/1PP1pP2/2B3P1/5PK1/2Q4R w - - 0 1",
    "2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - - 1 1",
    "r1bqk2r/pp2bppp/2p5/3pP3/P2Q1P2/2N1B3/1PP3PP/R4RK1 b kq - 0 1",
};

}  // namespace

BenchResult run_bench(Engine& engine, int depth) {
  BenchResult result;
  engine.new_game();
  const auto start = std::chrono::steady_clock::now();

  SearchLimits limits;
  limits.depth = depth;
  for (const std::string_view fen : kBenchFens) {
    const std::optional<Position> pos = Position::from_fen(fen);
    if (!pos) continue;
    result.nodes += engine.search_blocking(*pos, {}, limits).nodes;
  }

  result.time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - start)
                       .count();
  return result;
}

}  // namespace enroque

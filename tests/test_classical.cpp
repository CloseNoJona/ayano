#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

#include "board/position.h"
#include "eval/classical.h"
#include "eval/pst.h"
#include "test_framework.h"

using namespace ayano;

namespace {

int evaluate_fen(const std::string& fen) {
  const auto pos = Position::from_fen(fen);
  CHECK(pos.has_value());
  return pos ? int(classical::evaluate(*pos)) : 0;
}

// Da la vuelta al tablero y cambia los colores: para el bando con turno es la misma posicion.
std::string mirrored_board(const std::string& board) {
  std::vector<std::string> ranks(1);
  for (const char c : board) {
    if (c == '/') {
      ranks.emplace_back();
    } else {
      const auto letter = static_cast<unsigned char>(c);
      ranks.back() += char(std::isupper(letter) ? std::tolower(letter) : std::toupper(letter));
    }
  }
  std::reverse(ranks.begin(), ranks.end());
  std::string mirrored = ranks.front();
  for (std::size_t i = 1; i < ranks.size(); ++i) mirrored += "/" + ranks[i];
  return mirrored;
}

}  // namespace

TEST(clasica_la_posicion_inicial_solo_vale_el_turno) {
  CHECK_EQ(int(classical::evaluate(Position::startpos())), classical::kTempo);
}

TEST(clasica_los_terminos_posicionales_se_anulan_en_la_inicial) {
  const classical::PositionalTerms terms = classical::positional_terms(Position::startpos());
  CHECK_EQ(terms.mg, 0);
  CHECK_EQ(terms.eg, 0);
}

TEST(clasica_la_fase_va_del_final_al_medio_juego) {
  CHECK_EQ(classical::game_phase(Position::startpos()), classical::kMaxPhase);
  const auto ending = Position::from_fen("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1");
  CHECK(ending.has_value());
  if (ending) CHECK_EQ(classical::game_phase(*ending), 0);
}

TEST(clasica_es_simetrica_al_invertir_los_colores) {
  const std::vector<std::string> boards = {
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R",
      "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R1BQ1RK1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8",
      "8/8/4k3/8/8/4K3/4P3/8",
  };
  for (const std::string& board : boards) {
    CHECK_EQ(evaluate_fen(board + " w - - 0 1"), evaluate_fen(mirrored_board(board) + " b - - 0 1"));
  }
}

TEST(clasica_ordena_el_material) {
  const int queen = evaluate_fen("4k3/pppp4/8/8/8/8/PPPP4/3QK3 w - - 0 1");
  const int rook = evaluate_fen("4k3/pppp4/8/8/8/8/PPPP4/3RK3 w - - 0 1");
  const int knight = evaluate_fen("4k3/pppp4/8/8/8/8/PPPP4/3NK3 w - - 0 1");
  const int pawn = evaluate_fen("4k3/pppp4/8/8/8/8/PPPPP3/4K3 w - - 0 1");
  const int equal = evaluate_fen("4k3/pppp4/8/8/8/8/PPPP4/4K3 w - - 0 1");
  CHECK(queen > rook);
  CHECK(rook > knight);
  CHECK(knight > pawn);
  CHECK(pawn > equal);
}

TEST(clasica_los_peones_no_tienen_valor_en_las_filas_extremas) {
  for (std::size_t file = 0; file < 8; ++file) {
    CHECK_EQ(classical::kPstMg[PAWN][file], 0);
    CHECK_EQ(classical::kPstMg[PAWN][56 + file], 0);
    CHECK_EQ(classical::kPstEg[PAWN][file], 0);
    CHECK_EQ(classical::kPstEg[PAWN][56 + file], 0);
  }
}

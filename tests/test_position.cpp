#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "board/movegen.h"
#include "board/position.h"
#include "test_framework.h"

using namespace enroque;

namespace {

const std::vector<std::string>& sample_fens() {
  static const std::vector<std::string> fens = {
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
      "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
      "4k3/8/8/8/8/8/8/4K2R w K - 12 40",
  };
  return fens;
}

// Recorre partidas aleatorias y aplica `visit` a cada posicion alcanzada.
template <typename Visitor>
void random_walks(int games, int max_plies, Visitor visit) {
  std::mt19937 rng(20261103);
  for (const std::string& fen : sample_fens()) {
    for (int game = 0; game < games; ++game) {
      auto start = Position::from_fen(fen);
      CHECK(start.has_value());
      if (!start) return;
      Position pos = *start;
      for (int ply = 0; ply < max_plies; ++ply) {
        const MoveList moves = legal_moves(pos);
        if (moves.empty()) break;
        visit(pos, moves);
        pos = pos.after(moves[rng() % moves.size()].move);
      }
    }
  }
}

}  // namespace

TEST(fen_ida_y_vuelta) {
  for (const std::string& fen : sample_fens()) {
    const auto pos = Position::from_fen(fen);
    CHECK(pos.has_value());
    if (pos) CHECK_EQ(pos->fen(), fen);
  }
}

TEST(fen_invalidos_se_rechazan_con_mensaje) {
  const std::vector<std::string> invalid = {
      "",
      "basura",
      "8/8/8/8/8/8/8/8 w - - 0 1",                                 // sin reyes
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",  // turno invalido
      "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  // fila demasiado larga
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1",           // faltan filas
      "kk6/8/8/8/8/8/8/K7 w - - 0 1",                              // dos reyes negros
      "P3k3/8/8/8/8/8/8/4K3 w - - 0 1",                            // peon en la octava
      "4k3/8/8/8/8/8/8/4K2R b K - 0 1x",                           // contador invalido
      "4k2R/8/8/8/8/8/8/4K3 w - - 0 1",                            // rey rival en jaque sin turno
  };
  for (const std::string& fen : invalid) {
    std::string error;
    const auto pos = Position::from_fen(fen, &error);
    CHECK(!pos.has_value());
    CHECK(!error.empty());
  }
}

TEST(fen_con_mas_de_dos_jaques_se_rechaza) {
  std::string error;
  const auto pos = Position::from_fen("4k3/8/8/8/8/3n1n2/8/r3K3 w - - 0 1", &error);
  CHECK(!pos.has_value());
  CHECK(!error.empty());
}

// Posicion imposible pero aceptada: el al paso no resuelve el jaque del caballo.
TEST(al_paso_no_es_legal_si_no_resuelve_un_jaque_de_caballo) {
  const auto pos = Position::from_fen("4k3/8/8/3pP3/8/5n2/8/4K3 w - d6 0 1");
  CHECK(pos.has_value());
  if (!pos) return;
  CHECK(pos->in_check());
  CHECK(!legal_moves(*pos).contains(Move::en_passant(E5, D6)));
}

TEST(aplicar_una_jugada_no_modifica_el_original) {
  random_walks(4, 60, [](const Position& pos, const MoveList& moves) {
    const std::string before = pos.fen();
    const Key key_before = pos.key();
    for (const ExtMove& entry : moves) (void)pos.after(entry.move);
    CHECK_EQ(pos.fen(), before);
    CHECK_EQ(pos.key(), key_before);
  });
}

TEST(hash_incremental_coincide_con_hash_desde_cero) {
  random_walks(8, 120, [](const Position& pos, const MoveList&) {
    const auto rebuilt = Position::from_fen(pos.fen());
    CHECK(rebuilt.has_value());
    if (!rebuilt) return;
    CHECK_EQ(pos.key(), rebuilt->key());
    CHECK_EQ(pos.pawn_key(), rebuilt->pawn_key());
    CHECK_EQ(pos.minor_key(), rebuilt->minor_key());
    CHECK_EQ(pos.non_pawn_key(WHITE), rebuilt->non_pawn_key(WHITE));
    CHECK_EQ(pos.non_pawn_key(BLACK), rebuilt->non_pawn_key(BLACK));
    CHECK_EQ(pos.checkers(), rebuilt->checkers());
    CHECK_EQ(pos.blockers(WHITE), rebuilt->blockers(WHITE));
    CHECK_EQ(pos.blockers(BLACK), rebuilt->blockers(BLACK));
  });
}

TEST(gives_check_coincide_con_la_posicion_resultante) {
  random_walks(8, 120, [](const Position& pos, const MoveList& moves) {
    for (const ExtMove& entry : moves) {
      CHECK_EQ(pos.gives_check(entry.move), pos.after(entry.move).in_check());
    }
  });
}

TEST(is_pseudo_legal_acepta_exactamente_las_jugadas_generadas) {
  random_walks(4, 100, [](const Position& pos, const MoveList& moves) {
    for (const ExtMove& entry : moves) CHECK(pos.is_pseudo_legal(entry.move));

    // Solo las jugadas de la lista deben pasar los dos filtros.
    std::vector<std::uint16_t> legal;
    for (const ExtMove& entry : moves) legal.push_back(entry.move.raw());
    for (int from = 0; from < 64; ++from) {
      for (int to = 0; to < 64; ++to) {
        const Move candidate = Move::normal(Square(from), Square(to));
        const bool listed = std::find(legal.begin(), legal.end(), candidate.raw()) != legal.end();
        const bool accepted = pos.is_pseudo_legal(candidate) && pos.is_legal(candidate);
        if (accepted != listed) {
          testfw::fail(__FILE__, __LINE__, pos.fen() + " jugada " + candidate.to_uci());
        }
      }
    }
  });
}

TEST(generadores_por_etapas_cubren_todas_las_jugadas) {
  random_walks(4, 100, [](const Position& pos, const MoveList& moves) {
    MoveList staged;
    if (pos.in_check()) {
      staged = generate_moves(pos, GenType::Evasions);
    } else {
      staged = generate_moves(pos, GenType::Captures);
      const MoveList quiets = generate_moves(pos, GenType::Quiets);
      for (const ExtMove& entry : quiets) staged.push_back(entry.move);
    }
    std::size_t legal_count = 0;
    for (const ExtMove& entry : staged) legal_count += pos.is_legal(entry.move);
    CHECK_EQ(legal_count, moves.size());
  });
}

TEST(jugada_nula_cambia_turno_y_borra_al_paso) {
  const auto pos = Position::from_fen("4k3/8/8/8/4Pp2/8/8/4K3 b - e3 0 1");
  CHECK(pos.has_value());
  if (!pos) return;
  CHECK_EQ(int(pos->ep_square()), int(E3));
  const Position after_null = pos->after_null();
  CHECK_EQ(int(after_null.side_to_move()), int(WHITE));
  CHECK_EQ(int(after_null.ep_square()), int(NO_SQUARE));
  const auto rebuilt = Position::from_fen(after_null.fen());
  CHECK(rebuilt.has_value());
  if (rebuilt) CHECK_EQ(after_null.key(), rebuilt->key());
}

TEST(al_paso_solo_cuenta_en_el_hash_si_se_puede_capturar) {
  const auto with_capture = Position::from_fen("4k3/8/8/8/4Pp2/8/8/4K3 b - e3 0 1");
  const auto without_ep = Position::from_fen("4k3/8/8/8/4Pp2/8/8/4K3 b - - 0 1");
  const auto useless_ep = Position::from_fen("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1");
  const auto no_ep_plain = Position::from_fen("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1");
  CHECK(with_capture && without_ep && useless_ep && no_ep_plain);
  if (!with_capture || !without_ep || !useless_ep || !no_ep_plain) return;
  CHECK(with_capture->key() != without_ep->key());
  CHECK_EQ(useless_ep->key(), no_ep_plain->key());
}

TEST(see_valores_conocidos) {
  struct SeeCase {
    const char* fen;
    const char* move;
    int value;  // resultado exacto del intercambio en la escala de see_value
  };
  const int pawn = see_value(PAWN);
  const int knight = see_value(KNIGHT);
  const int rook = see_value(ROOK);
  const std::vector<SeeCase> cases = {
      {"1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1", "e1e5", pawn},
      {"1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1", "d3e5", pawn - knight},
      {"4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1", "d1d5", pawn},
      {"4k3/8/2p5/3p4/8/8/8/3RK3 w - - 0 1", "d1d5", pawn - rook},
      {"4k3/8/8/8/8/8/3r4/3RK3 w - - 0 1", "d1d2", rook},
  };
  for (const SeeCase& test_case : cases) {
    const auto pos = Position::from_fen(test_case.fen);
    CHECK(pos.has_value());
    if (!pos) continue;
    const Move move = parse_uci_move(*pos, test_case.move);
    CHECK(move != Move::none());
    CHECK(pos->see_ge(move, test_case.value));
    CHECK(!pos->see_ge(move, test_case.value + 1));
  }
}

TEST(parse_uci_rechaza_jugadas_ilegales) {
  const Position pos = Position::startpos();
  CHECK(parse_uci_move(pos, "e2e4") != Move::none());
  CHECK(parse_uci_move(pos, "e2e5") == Move::none());
  CHECK(parse_uci_move(pos, "zzzz") == Move::none());
  CHECK(parse_uci_move(pos, "") == Move::none());
  CHECK(parse_uci_move(pos, "e7e8q") == Move::none());
}

TEST(promocion_por_uci) {
  const auto pos = Position::from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1");
  CHECK(pos.has_value());
  if (!pos) return;
  const Move move = parse_uci_move(*pos, "a7a8n");
  CHECK(move != Move::none());
  CHECK_EQ(move.to_uci(), std::string("a7a8n"));
  CHECK_EQ(int(pos->after(move).piece_on(A8)), int(make_piece(WHITE, KNIGHT)));
}

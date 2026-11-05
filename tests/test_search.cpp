#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "board/movegen.h"
#include "board/position.h"
#include "search/engine.h"
#include "search/movepicker.h"
#include "search/tt.h"
#include "test_framework.h"

using namespace ayano;

namespace {

SearchResult search_fen(const std::string& fen, int depth = 4) {
  const auto pos = Position::from_fen(fen);
  CHECK(pos.has_value());
  if (!pos) return {};
  Engine engine;
  engine.set_use_nnue(false);
  SearchLimits limits;
  limits.depth = depth;
  return engine.search_blocking(*pos, {}, limits);
}

void expect_mate(const std::string& fen, int depth, int moves_to_mate, const std::string& first) {
  const SearchResult result = search_fen(fen, depth);
  CHECK_EQ(result.score, mate_in(2 * moves_to_mate - 1));
  if (!first.empty()) CHECK_EQ(result.best.to_uci(), first);
}

}  // namespace

TEST(encuentra_mate_en_uno) {
  expect_mate("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 6, 1, "a1a8");
  expect_mate("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 6, 1, "h5f7");
}

TEST(encuentra_mate_en_dos) {
  // 1.Dh6+ Rxh6 2.Axf6++ (jaque doble de alfil y torre).
  expect_mate("r1bq2r1/b4pk1/p1pp1p2/1p2pP2/1P2P1PB/3P4/1PPQ2P1/R3K2R w KQ - 0 1", 12, 2, "d2h6");
}

TEST(encuentra_mate_en_tres) {
  expect_mate("r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1", 14, 3, "f6a6");
}

TEST(mate_de_dama_y_rey_contra_rey) {
  const SearchResult result = search_fen("8/8/8/8/8/5k2/8/4K2Q b - - 0 1", 22);
  CHECK(result.score <= VALUE_MATED_IN_MAX_PLY);
}

TEST(detecta_que_recibe_mate) {
  const SearchResult result = search_fen("7k/8/5K2/6Q1/8/8/8/8 b - - 0 1", 8);
  CHECK(result.score <= VALUE_MATED_IN_MAX_PLY);
}

TEST(sin_jugadas_legales_no_hay_jugada) {
  CHECK(search_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").best == Move::none());      // ahogado
  CHECK(search_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").best == Move::none());      // mate
}

TEST(con_dama_de_mas_da_mate_y_no_ahoga) {
  const SearchResult result = search_fen("7k/8/6K1/8/8/5Q2/8/8 w - - 0 1", 10);
  CHECK_EQ(result.score, mate_in(1));
}

TEST(material_insuficiente_es_tablas) {
  const SearchResult result = search_fen("8/8/4k3/8/8/4K3/4B3/8 w - - 0 1", 10);
  CHECK(result.score >= -2 && result.score <= 2);
}

TEST(perdiendo_busca_la_triple_repeticion) {
  // Torre contra dama: pierde, salvo que Td8+ repita por tercera vez una posicion.
  const auto pos = Position::from_fen("7k/8/8/8/8/1q6/8/K2R4 w - - 8 20");
  CHECK(pos.has_value());
  if (!pos) return;
  const Move repeating = parse_uci_move(*pos, "d1d8");
  const Key repeated = pos->after(repeating).key();
  const std::vector<Key> keys = {0, repeated, 2, 3, 4, repeated, 6, 7};

  Engine engine;
  engine.set_use_nnue(false);
  SearchLimits limits;
  limits.depth = 8;
  const SearchResult with_history = engine.search_blocking(*pos, keys, limits);
  CHECK(with_history.best == repeating);
  CHECK(with_history.score >= -2 && with_history.score <= 2);

  engine.new_game();
  const SearchResult without_history = engine.search_blocking(*pos, {}, limits);
  CHECK(without_history.score < -200);
}

TEST(una_repeticion_dentro_del_arbol_cuenta_como_tablas) {
  // Con rey y peon bloqueado no hay progreso: la busqueda debe terminar en tablas.
  const SearchResult result = search_fen("8/8/8/1k6/1p6/1P6/1K6/8 w - - 0 1", 20);
  CHECK(result.score >= -30 && result.score <= 30);
}

TEST(regla_de_cincuenta_jugadas) {
  // Con el contador a 99, cualquier jugada que no sea mate termina en tablas.
  const SearchResult result = search_fen("7k/8/8/8/8/8/8/KR6 w - - 99 80", 6);
  CHECK(result.score >= -2 && result.score <= 2);
  // Pero un mate en la ultima jugada permitida sigue siendo mate.
  const SearchResult mate = search_fen("7k/8/6K1/8/8/8/8/1R6 w - - 99 80", 6);
  CHECK_EQ(mate.score, mate_in(1));
}

TEST(respeta_el_limite_de_nodos) {
  Engine engine;
  engine.set_use_nnue(false);
  SearchLimits limits;
  limits.nodes = 20000;
  const SearchResult result = engine.search_blocking(Position::startpos(), {}, limits);
  CHECK(result.best != Move::none());
  CHECK(result.nodes < 40000);
}

TEST(multipv_devuelve_una_jugada_legal) {
  Engine engine;
  engine.set_use_nnue(false);
  engine.set_multi_pv(3);
  SearchLimits limits;
  limits.depth = 6;
  const Position pos = Position::startpos();
  const SearchResult result = engine.search_blocking(pos, {}, limits);
  CHECK(legal_moves(pos).contains(result.best));
}

TEST(varios_hilos_devuelven_una_jugada_legal) {
  Engine engine;
  engine.set_use_nnue(false);
  engine.set_threads(4);
  SearchLimits limits;
  limits.depth = 10;
  const auto pos = Position::from_fen(
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
  CHECK(pos.has_value());
  if (!pos) return;
  const SearchResult result = engine.search_blocking(*pos, {}, limits);
  CHECK(legal_moves(*pos).contains(result.best));
}

TEST(nivel_bajo_varia_las_jugadas_y_siempre_son_legales) {
  Engine engine;
  engine.set_use_nnue(false);
  engine.set_skill_level(0);
  const Position pos = Position::startpos();
  SearchLimits limits;
  std::vector<std::uint16_t> seen;
  for (int i = 0; i < 40; ++i) {
    const SearchResult result = engine.search_blocking(pos, {}, limits);
    CHECK(legal_moves(pos).contains(result.best));
    if (std::find(seen.begin(), seen.end(), result.best.raw()) == seen.end()) {
      seen.push_back(result.best.raw());
    }
  }
  CHECK(seen.size() > 1);
}

TEST(con_nivel_bajo_no_se_deja_escapar_un_mate) {
  const auto pos = Position::from_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1");
  CHECK(pos.has_value());
  if (!pos) return;
  Engine engine;
  engine.set_use_nnue(false);
  engine.set_skill_level(4);
  SearchLimits limits;
  CHECK_EQ(engine.search_blocking(*pos, {}, limits).best.to_uci(), std::string("a1a8"));
}

TEST(searchmoves_restringe_la_raiz) {
  Engine engine;
  engine.set_use_nnue(false);
  const Position pos = Position::startpos();
  SearchLimits limits;
  limits.depth = 5;
  const Move only = parse_uci_move(pos, "a2a3");
  limits.search_moves.push_back(only);
  CHECK(engine.search_blocking(pos, {}, limits).best == only);
}

TEST(limite_de_nodos_exacto_con_un_hilo) {
  Engine engine;
  engine.set_use_nnue(false);
  SearchLimits limits;
  limits.nodes = 300;
  const SearchResult result = engine.search_blocking(Position::startpos(), {}, limits);
  CHECK(result.best != Move::none());
  CHECK(result.nodes <= 302);
}

TEST(la_promocion_a_dama_se_prueba_antes_que_las_tranquilas) {
  const auto pos = Position::from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1");
  CHECK(pos.has_value());
  if (!pos) return;
  auto histories = std::make_unique<Histories>();
  std::memset(static_cast<void*>(histories.get()), 0, sizeof(Histories));
  // Un historial de capturas muy negativo no debe relegar la promocion al final.
  histories->capture[W_PAWN][A8][NO_PIECE_TYPE] = -8000;
  const ContinuationRefs refs = {&histories->sentinel, &histories->sentinel,
                                 &histories->sentinel, &histories->sentinel};
  MovePicker picker(*pos, Move::none(), *histories, refs,
                    {Move::none(), Move::none(), Move::none()});
  CHECK(picker.next() == Move::promotion(A7, A8, QUEEN));
}

// AddressSanitizer aborta ante una reserva tan grande en vez de dejar que falle.
#ifndef __SANITIZE_ADDRESS__
TEST(tabla_hash_sobrevive_a_un_tamano_imposible) {
  TranspositionTable table;
  table.resize(4);
  CHECK(!table.resize(std::size_t(1) << 40));
  const Key key = 77;
  table.store(key, 5, 6, 7, Bound::Exact, Move::normal(E2, E4), false);
  TTData data{};
  CHECK(table.probe(key, data));
}
#endif

TEST(tabla_hash_guarda_y_recupera) {
  TranspositionTable table;
  table.resize(1);
  const Key key = 0x1234567890ABCDEFULL;
  const Move move = Move::normal(E2, E4);
  table.store(key, -321, 45, 9, Bound::Lower, move, true);

  TTData data{};
  CHECK(table.probe(key, data));
  CHECK(data.move == move);
  CHECK_EQ(data.value, -321);
  CHECK_EQ(data.eval, 45);
  CHECK_EQ(data.depth, 9);
  CHECK(data.bound == Bound::Lower);
  CHECK(data.pv);

  TTData other{};
  CHECK(!table.probe(key ^ 1, other));
}

TEST(tabla_hash_no_pierde_la_jugada_al_actualizar) {
  TranspositionTable table;
  table.resize(1);
  const Key key = 42;
  const Move move = Move::normal(G1, F3);
  table.store(key, 10, 5, 4, Bound::Exact, move, false);
  table.store(key, 20, 5, 8, Bound::Upper, Move::none(), false);
  TTData data{};
  CHECK(table.probe(key, data));
  CHECK(data.move == move);
  CHECK_EQ(data.value, 20);
}

TEST(las_puntuaciones_de_mate_se_ajustan_por_altura) {
  const Value mate_from_root = mate_in(9);
  const int ply = 4;
  const Value stored = value_to_tt(mate_from_root, ply);
  CHECK_EQ(stored, mate_in(5));
  CHECK_EQ(value_from_tt(stored, ply), mate_from_root);
  CHECK_EQ(value_from_tt(stored, 6), mate_in(11));
  const Value mated = mated_in(8);
  CHECK_EQ(value_from_tt(value_to_tt(mated, 3), 3), mated);
}

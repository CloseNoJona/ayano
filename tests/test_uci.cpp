#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "board/movegen.h"
#include "board/position.h"
#include "search/timeman.h"
#include "test_framework.h"
#include "uci/uci.h"

using namespace ayano;

namespace {

// Ejecuta comandos UCI y captura lo que el motor escribe en la salida estandar.
class UciSession {
 public:
  UciSession() : saved_(std::cout.rdbuf(buffer_.rdbuf())) {}
  ~UciSession() {
    // Con la busqueda parada ya nadie escribe en el buffer que se retira.
    uci_.execute("stop");
    uci_.execute("wait");
    std::cout.rdbuf(saved_);
  }
  UciSession(const UciSession&) = delete;
  UciSession& operator=(const UciSession&) = delete;

  std::string run(const std::string& line) {
    buffer_.str("");
    uci_.execute(line);
    return buffer_.str();
  }

  // Envia un comando sin tocar el buffer: vale con una busqueda en marcha.
  void send(const std::string& line) { uci_.execute(line); }

  // Lanza una busqueda sin leer la salida: su hilo ya escribe en el buffer.
  void start(const std::string& go_line) {
    buffer_.str("");
    uci_.execute(go_line);
  }

  // Envia `line` a la busqueda en curso y espera a que termine. El buffer solo se toca
  // cuando el hilo ha acabado: vaciarlo antes podria borrar el `bestmove`.
  std::string finish(const std::string& line) {
    uci_.execute(line);
    uci_.execute("wait");
    return buffer_.str();
  }

  // Lanza `go` y espera a que termine la busqueda.
  std::string search(const std::string& go_line) {
    start(go_line);
    return finish("wait");
  }

 private:
  std::ostringstream buffer_;
  std::streambuf* saved_;
  Uci uci_;
};

bool contains(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

int count_lines(const std::string& text, const std::string& start) {
  std::istringstream lines(text);
  int count = 0;
  for (std::string line; std::getline(lines, line);) count += line.starts_with(start);
  return count;
}

constexpr const char* kStartFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Una interfaz convierte cada PV a notacion algebraica y se queja de la primera jugada ilegal.
void expect_legal_pvs(const std::string& fen, int depth) {
  const auto root = Position::from_fen(fen);
  CHECK(root.has_value());
  if (!root) return;

  UciSession session;
  session.run("setoption name Use NNUE value false");  // el arbol no depende de la red activa
  session.run("position fen " + fen);
  std::istringstream reply(session.search("go depth " + std::to_string(depth)));
  int lines = 0;
  for (std::string line; std::getline(reply, line);) {
    const std::size_t start = line.find(" pv ");
    if (start == std::string::npos) continue;
    ++lines;
    Position pos = *root;
    std::istringstream moves(line.substr(start + 4));
    for (std::string text; moves >> text;) {
      const Move move = parse_uci_move(pos, text);
      CHECK(move != Move::none());
      if (move == Move::none()) break;
      pos = pos.after(move);
    }
  }
  CHECK(lines > 0);
}

// Cuanto reloj queda tras `moves` jugadas sin incremento. La busqueda suele pasarse del
// objetivo, asi que se cuenta el doble, sin superar el maximo.
std::int64_t clock_after_game(std::int64_t start, int moves, std::int64_t latency_ms) {
  constexpr int kOverhead = 10;
  std::int64_t clock = start;
  for (int move = 0; move < moves && clock > 0; ++move) {
    SearchLimits limits;
    limits.time_ms[WHITE] = clock;
    TimeManager time;
    time.start(limits, WHITE, 2 * move, kOverhead);
    clock -= std::min(2 * time.optimum_ms(), time.maximum_ms()) + latency_ms;
  }
  return clock;
}

}  // namespace

TEST(uci_se_identifica_y_declara_opciones) {
  UciSession session;
  const std::string reply = session.run("uci");
  CHECK(contains(reply, "id name Ayano"));
  CHECK(contains(reply, "id author"));
  CHECK(contains(reply, "option name Hash"));
  CHECK(contains(reply, "option name Threads"));
  CHECK(contains(reply, "uciok"));
  CHECK(contains(session.run("isready"), "readyok"));
}

TEST(uci_aplica_jugadas_a_la_posicion) {
  UciSession session;
  session.run("position startpos moves e2e4 e7e5 g1f3");
  CHECK(contains(session.run("d"),
                 "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"));
}

TEST(uci_acepta_fen_con_jugadas) {
  UciSession session;
  session.run("position fen 4k3/8/8/8/8/8/4P3/4K3 w - - 0 1 moves e2e4");
  CHECK(contains(session.run("d"), "4k3/8/8/8/4P3/8/8/4K3 b - - 0 1"));
}

TEST(uci_rechaza_fen_invalido_y_conserva_la_posicion) {
  UciSession session;
  session.run("position startpos");
  CHECK(contains(session.run("position fen esto/no/es/un/fen w - - 0 1"), "error"));
  CHECK(contains(session.run("d"), kStartFen));
}

TEST(uci_rechaza_jugada_ilegal_y_conserva_la_posicion) {
  UciSession session;
  session.run("position startpos");
  CHECK(contains(session.run("position startpos moves e2e4 e2e5"), "error"));
  CHECK(contains(session.run("d"), kStartFen));
}

TEST(uci_position_sin_argumentos_es_un_error) {
  UciSession session;
  CHECK(contains(session.run("position"), "error"));
  CHECK(contains(session.run("position algo"), "error"));
}

TEST(uci_go_devuelve_una_jugada) {
  UciSession session;
  session.run("position startpos");
  const std::string reply = session.search("go depth 4");
  CHECK(contains(reply, "info depth"));
  CHECK(contains(reply, "bestmove"));
  CHECK(!contains(reply, "bestmove 0000"));
}

TEST(uci_go_con_reloj_responde_rapido) {
  UciSession session;
  session.run("position startpos");
  const std::string reply = session.search("go wtime 200 btime 200 winc 10 binc 10");
  CHECK(contains(reply, "bestmove"));
}

// Una interfaz que espera `bestmove` perderia por tiempo: siempre debe haber respuesta.
TEST(uci_go_con_valores_invalidos_avisa_y_responde_igualmente) {
  UciSession session;
  session.run("position startpos");
  const std::string reply = session.search("go depth muchos");
  CHECK(contains(reply, "error"));
  CHECK(contains(reply, "bestmove"));
  CHECK(!contains(reply, "bestmove 0000"));
}

TEST(uci_searchmoves_admite_mas_parametros_detras) {
  UciSession session;
  session.run("position startpos");
  const std::string reply = session.search("go searchmoves a2a3 h2h3 depth 3");
  CHECK(!contains(reply, "error"));
  CHECK(contains(reply, "bestmove a2a3") || contains(reply, "bestmove h2h3"));
}

TEST(uci_sin_jugadas_responde_0000) {
  UciSession session;
  session.run("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
  CHECK(contains(session.search("go depth 3"), "bestmove 0000"));
}

TEST(uci_mate_se_anuncia_en_jugadas) {
  UciSession session;
  session.run("position fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1");
  const std::string reply = session.search("go depth 6");
  CHECK(contains(reply, "score mate 1"));
  CHECK(contains(reply, "bestmove a1a8"));
}

TEST(uci_stop_termina_una_busqueda_infinita) {
  UciSession session;
  session.run("position startpos");
  session.start("go infinite");
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK(contains(session.finish("stop"), "bestmove"));
}

// Una interfaz puede cancelar justo despues de `go`: tambien merece una jugada valida.
TEST(uci_stop_inmediato_tambien_responde) {
  UciSession session;
  session.run("position startpos");
  for (int cycle = 0; cycle < 50; ++cycle) {
    session.start("go infinite");
    const std::string reply = session.finish("stop");
    CHECK(contains(reply, "bestmove"));
    CHECK(!contains(reply, "bestmove 0000"));
  }
}

// UCI prohibe responder mientras se medita: `readyok` debe llegar antes que `bestmove`.
TEST(uci_ponderhit_libera_la_respuesta) {
  UciSession session;
  session.run("position startpos");
  session.start("go ponder depth 2");
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  session.send("isready");
  const std::string reply = session.finish("ponderhit");
  const std::size_t ready = reply.find("readyok");
  const std::size_t best = reply.find("bestmove");
  CHECK(ready != std::string::npos);
  CHECK(best != std::string::npos);
  CHECK(ready < best);
}

TEST(uci_valida_las_opciones) {
  UciSession session;
  CHECK(contains(session.run("setoption name Hash value 0"), "error"));
  CHECK(contains(session.run("setoption name Hash value mucho"), "error"));
  CHECK(contains(session.run("setoption name Threads value 100000"), "error"));
  CHECK(contains(session.run("setoption name Inventada value 1"), "error"));
  CHECK(!contains(session.run("setoption name Hash value 8"), "error"));
  CHECK(!contains(session.run("setoption name Threads value 2"), "error"));
  CHECK(!contains(session.run("setoption name Move Overhead value 50"), "error"));
  CHECK(contains(session.run("setoption name EvalFile value no_existe.nnue"), "error"));
}

TEST(uci_hash_pequeno_permite_buscar) {
  UciSession session;
  session.run("setoption name Hash value 8");
  session.run("position startpos");
  CHECK(contains(session.search("go depth 3"), "bestmove"));
}

TEST(uci_comando_desconocido_no_rompe_nada) {
  UciSession session;
  CHECK(contains(session.run("foobar"), "error"));
  CHECK(session.run("").empty());
  CHECK(contains(session.run("isready"), "readyok"));
}

TEST(uci_acepta_debug_y_register_sin_quejarse) {
  UciSession session;
  CHECK_EQ(session.run("debug on"), std::string());
  CHECK_EQ(session.run("debug off"), std::string());
  CHECK_EQ(session.run("register later"), std::string());
  CHECK(contains(session.run("isready"), "readyok"));
}

// Si la ventana nula corta en un nodo PV no se repite la busqueda: pasa sobre todo tras un mate,
// porque la poda por distancia al mate baja beta. La linea no debe heredar restos de otra.
TEST(uci_las_pv_son_legales) {
  expect_legal_pvs("4R3/1k6/4P3/8/1pbP4/2n3P1/1KP4P/5r2 w - - 3 43", 14);
}

// Con muy poco reloj la interfaz tarda mas en leer las lineas que el motor en escribirlas, y
// ese retraso se descuenta del reloj: solo se informa del resultado final.
TEST(uci_con_poco_tiempo_solo_informa_al_final) {
  UciSession session;
  session.run("position startpos");
  const std::string reply = session.search("go wtime 50 btime 50");
  CHECK_EQ(count_lines(reply, "info depth"), 1);
  CHECK_EQ(count_lines(reply, "bestmove"), 1);
}

// Mientras medita el reloj del motor no corre: la interfaz puede leer sin prisa.
TEST(uci_mientras_medita_informa_aunque_quede_poco_reloj) {
  UciSession session;
  session.run("position startpos");
  session.start("go ponder wtime 50 btime 50");
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  CHECK(count_lines(session.finish("stop"), "info depth") > 1);
}

TEST(uci_con_tiempo_informa_de_cada_iteracion) {
  UciSession session;
  session.run("position startpos");
  CHECK(count_lines(session.search("go movetime 150"), "info depth") > 1);
  CHECK(count_lines(session.search("go depth 6"), "info depth") >= 6);
}

TEST(uci_perft_valida_la_profundidad) {
  UciSession session;
  CHECK(contains(session.run("perft 0"), "error"));
  CHECK(contains(session.run("perft x"), "error"));
  session.run("position startpos");
  CHECK(contains(session.run("perft 3"), "8902"));
}

TEST(uci_multipv_informa_varias_lineas) {
  UciSession session;
  session.run("setoption name MultiPV value 3");
  session.run("position startpos");
  const std::string reply = session.search("go depth 5");
  CHECK(contains(reply, "multipv 1"));
  CHECK(contains(reply, "multipv 3"));
}

TEST(tiempo_el_maximo_nunca_supera_el_reloj) {
  for (const std::int64_t remaining : {1LL, 5LL, 50LL, 1000LL, 60000LL, 3600000LL}) {
    for (const std::int64_t increment : {0LL, 10LL, 1000LL}) {
      SearchLimits limits;
      limits.time_ms[WHITE] = remaining;
      limits.increment_ms[WHITE] = increment;
      TimeManager time;
      time.start(limits, WHITE, 20, 10);
      CHECK(time.optimum_ms() >= 1);
      CHECK(time.optimum_ms() <= time.maximum_ms());
      CHECK(time.maximum_ms() <= std::max<std::int64_t>(remaining, 1));
    }
  }
}

// Windows puede retrasar un hilo un tic de 15,6 ms o mas: se midio un exceso de 16 ms sobre
// el limite con 86 ms en el reloj. Con el reloj bajo, esa parada no debe costar la partida.
TEST(tiempo_deja_colchon_con_el_reloj_bajo) {
  constexpr int kOverhead = 10;
  for (const std::int64_t remaining : {24LL, 50LL, 86LL, 120LL, 200LL, 500LL}) {
    for (const std::int64_t increment : {0LL, 10LL, 50LL, 100LL}) {
      SearchLimits limits;
      limits.time_ms[WHITE] = remaining;
      limits.increment_ms[WHITE] = increment;
      TimeManager time;
      time.start(limits, WHITE, 200, kOverhead);
      CHECK(remaining - time.maximum_ms() >= 2 * kOverhead);
      CHECK(time.maximum_ms() * 10 <= remaining * 7);
    }
  }
}

// Cada jugada cuesta reloj aunque no se piense: con Cute Chess se midieron de 3 a 6 ms entre
// `go` y `bestmove`. Sin incremento, el reloj debe aguantar una partida larga.
TEST(tiempo_sin_incremento_aguanta_una_partida_larga) {
  constexpr int kMoves = 120;
  constexpr std::int64_t kLatencyMs = 4;
  for (const std::int64_t start : {2000LL, 5000LL, 60000LL, 300000LL}) {
    CHECK(clock_after_game(start, kMoves, kLatencyMs) > 0);
  }
}

TEST(tiempo_sin_incremento_juega_al_momento_con_poco_reloj) {
  SearchLimits limits;
  limits.time_ms[WHITE] = 400;
  TimeManager time;
  time.start(limits, WHITE, 200, 10);
  CHECK_EQ(time.optimum_ms(), std::int64_t(1));
  CHECK_EQ(time.maximum_ms(), std::int64_t(1));
}

// Un margen grande suele ser una precaucion: la reserva no debe dejar la partida sin pensar.
TEST(tiempo_un_margen_grande_no_se_lleva_todo_el_reloj) {
  SearchLimits limits;
  limits.time_ms[WHITE] = 5000;
  TimeManager time;
  time.start(limits, WHITE, 20, 100);
  CHECK(time.optimum_ms() > 50);
}

// Con incremento o con control de jugadas, el reparto debe seguir siendo el que se midio.
TEST(tiempo_la_reserva_no_toca_el_reparto_medido) {
  SearchLimits long_game;
  long_game.time_ms[WHITE] = 60000;
  long_game.increment_ms[WHITE] = 1000;
  TimeManager with_increment;
  with_increment.start(long_game, WHITE, 20, 10);
  CHECK_EQ(with_increment.optimum_ms(), std::int64_t(2139));
  CHECK_EQ(with_increment.maximum_ms(), std::int64_t(12228));

  SearchLimits small_increment;
  small_increment.time_ms[WHITE] = 1000;
  small_increment.increment_ms[WHITE] = 10;
  TimeManager same_as_overhead;
  same_as_overhead.start(small_increment, WHITE, 200, 10);
  CHECK_EQ(same_as_overhead.optimum_ms(), std::int64_t(53));
  CHECK_EQ(same_as_overhead.maximum_ms(), std::int64_t(308));

  SearchLimits moves_control;
  moves_control.time_ms[WHITE] = 1000;
  moves_control.moves_to_go = 10;
  TimeManager with_moves;
  with_moves.start(moves_control, WHITE, 40, 10);
  CHECK_EQ(with_moves.optimum_ms(), std::int64_t(69));
  CHECK_EQ(with_moves.maximum_ms(), std::int64_t(396));
}

TEST(tiempo_movetime_es_exacto_menos_el_margen) {
  SearchLimits limits;
  limits.move_time_ms = 1000;
  TimeManager time;
  time.start(limits, WHITE, 0, 10);
  CHECK_EQ(time.maximum_ms(), std::int64_t(990));
  CHECK_EQ(time.optimum_ms(), std::int64_t(990));
}

TEST(tiempo_sin_reloj_no_hay_limite) {
  TimeManager time;
  time.start(SearchLimits{}, WHITE, 0, 10);
  CHECK(!time.has_limit());
}

TEST(tiempo_movestogo_reparte_el_reloj) {
  SearchLimits limits;
  limits.time_ms[BLACK] = 60000;
  limits.moves_to_go = 10;
  TimeManager few_moves;
  few_moves.start(limits, BLACK, 30, 0);
  limits.moves_to_go = 40;
  TimeManager many_moves;
  many_moves.start(limits, BLACK, 30, 0);
  CHECK(few_moves.optimum_ms() > many_moves.optimum_ms());
}

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "search/timeman.h"
#include "test_framework.h"
#include "uci/uci.h"

using namespace enroque;

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

constexpr const char* kStartFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

}  // namespace

TEST(uci_se_identifica_y_declara_opciones) {
  UciSession session;
  const std::string reply = session.run("uci");
  CHECK(contains(reply, "id name Enroque"));
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

// Protocolo UCI: el idioma comun entre motores e interfaces como Arena o CuteChess.
#pragma once

#include <iosfwd>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "board/position.h"
#include "search/engine.h"

namespace enroque {

class Uci : public SearchListener {
 public:
  Uci();

  // Lee comandos de la entrada estandar hasta recibir `quit`.
  void loop();
  // Ejecuta un comando; devuelve false si es `quit`.
  bool execute(const std::string& line);

  void on_info(const SearchInfo& info) override;
  void on_best_move(Move best, Move ponder) override;

 private:
  void send(const std::string& text);
  void send_error(const std::string& message);
  void handle_uci();
  void handle_position(std::istringstream& args);
  void handle_go(std::istringstream& args);
  void handle_setoption(std::istringstream& args);
  void handle_perft(std::istringstream& args);
  void handle_bench(std::istringstream& args);
  void handle_new_game();
  void handle_eval();
  void show_board();
  void apply_option(const std::string& name, const std::string& value);
  SearchLimits parse_limits(std::istringstream& args);
  static void apply_limit(SearchLimits& limits, const std::string& name, std::int64_t value);
  void stop_search();

  Position position_;
  std::vector<Key> game_keys_;
  std::mutex output_mutex_;
  bool in_bench_ = false;
  // El ultimo: su destructor espera a la busqueda, que usa todo lo anterior.
  Engine engine_;
};

}  // namespace enroque

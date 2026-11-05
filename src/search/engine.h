// Fachada de la busqueda: gestiona los hilos y expone una API sencilla, tanto
// asincrona (para UCI) como bloqueante (para generar datos y para los tests).
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "core/prng.h"
#include "search/worker.h"

namespace ayano {

struct SearchResult {
  Move best = Move::none();
  Move ponder = Move::none();
  Value score = VALUE_ZERO;
  int depth = 0;
  std::uint64_t nodes = 0;
};

class Engine {
 public:
  Engine();
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Ambas devuelven false si no hay memoria; entonces se conserva lo anterior.
  bool set_threads(int count);
  bool set_hash_mb(std::size_t megabytes);
  void set_multi_pv(int lines) { shared_.multi_pv = lines; }
  void set_skill_level(int level) { shared_.skill_level = level; }
  void set_move_overhead(int ms) { shared_.move_overhead_ms = ms; }
  void set_use_nnue(bool enabled) { shared_.use_nnue = enabled; }
  void set_listener(SearchListener* listener) { shared_.listener = listener; }
  void clear_hash() { shared_.tt.clear(); }
  void new_game();

  // Inicia la busqueda en otro hilo. `game_keys` son las claves de las posiciones
  // anteriores a `root`, necesarias para detectar repeticiones.
  void start(const Position& root, std::span<const Key> game_keys, const SearchLimits& limits);
  void stop();
  void ponder_hit();
  void wait();

  // Busca en el hilo actual y devuelve el resultado al terminar.
  SearchResult search_blocking(const Position& root, std::span<const Key> game_keys,
                               const SearchLimits& limits);

  Value evaluate(const Position& pos);

 private:
  void configure(const Position& root, std::span<const Key> game_keys, const SearchLimits& limits);
  void run_threads();
  const Worker& pick_best_worker() const;
  Move pick_with_skill(const Worker& worker) const;
  SearchResult collect_result() const;

  SearchShared shared_;
  std::vector<std::unique_ptr<Worker>> workers_;
  std::thread search_thread_;
  mutable Prng skill_rng_;
  Position root_;
  std::vector<Key> game_keys_;
};

}  // namespace ayano

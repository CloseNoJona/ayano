// Un Worker es un hilo de busqueda con su propia pila de posiciones e historiales.
// Todos los hilos comparten solo la tabla de transposicion (Lazy SMP).
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "board/position.h"
#include "nnue/accumulator.h"
#include "search/history.h"
#include "search/limits.h"
#include "search/movepicker.h"
#include "search/timeman.h"
#include "search/tt.h"

namespace ayano {

class Worker;

struct RootMove {
  Move move = Move::none();
  Value score = -VALUE_INFINITE;
  Value previous_score = -VALUE_INFINITE;
  int sel_depth = 0;
  bool lower_bound = false;
  bool upper_bound = false;
  std::uint64_t nodes = 0;
  std::vector<Move> pv;
};

// Datos de una iteracion completada, listos para mostrar.
struct SearchInfo {
  int depth = 0;
  int sel_depth = 0;
  int multi_pv = 1;
  Value score = VALUE_ZERO;
  bool lower_bound = false;
  bool upper_bound = false;
  std::uint64_t nodes = 0;
  std::int64_t time_ms = 0;
  int hashfull = 0;
  const std::vector<Move>* pv = nullptr;
};

inline constexpr int kMaxSkillLevel = 20;

class SearchListener {
 public:
  virtual ~SearchListener() = default;
  virtual void on_info(const SearchInfo& info) = 0;
  virtual void on_best_move(Move best, Move ponder) = 0;
};

struct SearchShared {
  TranspositionTable tt;
  TimeManager time;
  SearchLimits limits;
  std::atomic<bool> stop{false};
  std::atomic<bool> pondering{false};
  int multi_pv = 1;
  int move_overhead_ms = 10;
  int skill_level = kMaxSkillLevel;
  bool use_nnue = true;
  SearchListener* listener = nullptr;
  std::span<const std::unique_ptr<Worker>> workers;

  std::uint64_t total_nodes() const;
  int effective_multi_pv() const;
  int skill_depth_limit() const;
};

enum class NodeType { NonPv, Pv, Root };

struct StackEntry {
  Position pos;
  PieceToHistory* cont_hist;  // tabla de continuacion de la jugada en curso
  Move move;                  // jugada que se esta probando desde este nodo
  Move excluded;              // jugada excluida en la busqueda de singularidad
  std::array<Move, 2> killers;
  Value static_eval;
  int cutoff_count;
  int reduction;  // plies que la busqueda reducida quita al hijo; 0 fuera de ella
  Bitboard threats;  // casillas que ataca el rival; eligen la subtabla del historial principal
  bool in_check;
  bool tt_pv;
};

struct Node;
struct QNode;
struct Candidate;

// Recalcula las tablas derivadas de los parametros (solo para ajuste por UCI).
void rebuild_search_tables();

class Worker {
 public:
  Worker(int id, SearchShared& shared);

  void new_game();
  // `game_keys` contiene las claves de las posiciones anteriores a la raiz.
  void prepare(const Position& root, std::span<const Key> game_keys);
  void run();

  std::uint64_t nodes() const { return nodes_.load(std::memory_order_relaxed); }
  int completed_depth() const { return completed_depth_; }
  const RootMove& best_root_move() const { return root_moves_.front(); }
  std::span<const RootMove> root_moves() const { return root_moves_; }
  bool has_root_moves() const { return !root_moves_.empty(); }

  // Evaluacion estatica sin efectos sobre la busqueda (comando `eval`).
  Value evaluate_root();

 private:
  static constexpr int kStackOffset = 7;
  static constexpr int kCheckInterval = 1024;

  StackEntry* entry(int ply) { return &stack_[std::size_t(ply + kStackOffset)]; }
  bool is_main() const { return id_ == 0; }
  bool stopped() const { return shared_.stop.load(std::memory_order_relaxed); }
  void count_node() {
    nodes_.store(nodes_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
  }

  // Bucle de profundizacion iterativa (worker.cpp).
  void iterate_depth();
  void aspiration_window();
  void report_pv(int depth);
  void check_limits();
  void decide_stop_after_iteration();
  bool root_contains(Move m) const;

  // Busqueda (search.cpp).
  template <NodeType Type>
  Value search(int ply, Value alpha, Value beta, int depth, bool cut_node);
  template <bool Pv>
  Value qsearch(int ply, Value alpha, Value beta);
  template <NodeType Type>
  Value move_loop(Node& n);

  Value enter_node(Node& n);
  void probe_tt(Node& n);
  bool can_cut_with_tt(const Node& n) const;
  void compute_static_eval(Node& n);
  void adjust_depth_in_hindsight(Node& n) const;
  void learn_from_eval_swing(const Node& n);
  Value forward_prune(Node& n);
  Value null_move_search(Node& n);
  Value probcut(Node& n);
  bool prune_move(Node& n, Candidate& c, bool& skip_quiets);
  int extension(Node& n, const Candidate& c);
  int lmr_reduction(const Node& n, const Candidate& c) const;
  int adjusted_reduction(const Node& n, const Candidate& c);
  Value reduced_search(Node& n, const Candidate& c, int& new_depth);
  template <NodeType Type>
  Value search_candidate(Node& n, const Candidate& c, int extended_depth);
  void record_result(Node& n, const Candidate& c, Value value);
  void update_root_move(Node& n, Move m, Value value);
  Value finish_node(Node& n);

  // Busqueda de quietud (qsearch.cpp).
  Value qsearch_enter(QNode& q);
  void qsearch_probe(QNode& q);
  Value qsearch_stand_pat(QNode& q);
  bool qsearch_prune(QNode& q, Move m);

  void make_move(StackEntry* ss, int ply, Move m, bool capture);
  void make_null_move(StackEntry* ss, int ply);
  void update_pv(int ply, Move m);

  bool is_draw(const Position& pos, int ply) const;
  bool is_repetition(const Position& pos, int ply) const;
  bool has_upcoming_repetition(const Position& pos, int ply) const;
  bool claim_upcoming_draw(const Position& pos, int ply, Value& alpha, Value beta) const;
  Value draw_value() const { return VALUE_DRAW - 1 + Value(nodes() & 2); }

  Value evaluate(int ply);
  Value corrected_eval(const StackEntry* ss, Value raw) const;
  void update_correction(const StackEntry* ss, int depth, Value difference);
  HistoryValue* continuation_correction_entry(const StackEntry* ss) const;
  void update_continuation(StackEntry* ss, Piece pc, Square to, int bonus);
  void update_quiet_history(const Position& pos, StackEntry* ss, Move m, int bonus);
  void update_capture_history(const Position& pos, Move m, int bonus);
  void update_cutoff_stats(Node& n);
  void update_prior_move_stats(Node& n);
  void set_refutations(const Position& pos, StackEntry* ss, Move best);
  ContinuationRefs continuation_refs(const StackEntry* ss) const;

  const int id_;
  SearchShared& shared_;
  std::unique_ptr<Histories> histories_;
  std::unique_ptr<nnue::AccumulatorStack> accumulators_;
  std::vector<StackEntry> stack_;
  std::vector<std::array<Move, MAX_PLY + 1>> pv_table_;
  std::array<int, MAX_PLY + 2> pv_length_{};

  std::vector<RootMove> root_moves_;
  std::vector<Key> key_history_;
  int game_length_ = 0;
  std::size_t pv_index_ = 0;
  int root_depth_ = 0;
  int completed_depth_ = 0;
  int sel_depth_ = 0;
  int nmp_min_ply_ = 0;
  int calls_until_check_ = kCheckInterval;
  std::atomic<std::uint64_t> nodes_{0};

  // Estado para decidir cuando parar (solo hilo principal).
  Move stable_best_move_ = Move::none();
  int best_move_stability_ = 0;
  Value previous_iteration_score_ = VALUE_NONE;
};

}  // namespace ayano

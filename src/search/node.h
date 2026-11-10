// Estado interno de un nodo de la busqueda, compartido por los archivos de src/search.
#pragma once

#include <algorithm>
#include <array>

#include "search/params.h"
#include "search/worker.h"

namespace ayano {

inline constexpr int kMaxTried = 32;

inline int history_bonus(int depth) {
  return std::min(params::kHistoryBonusScale * depth - params::kHistoryBonusOffset,
                  params::kHistoryBonusMax);
}

inline int history_malus(int depth) {
  return std::min(params::kHistoryMalusScale * depth - params::kHistoryMalusOffset,
                  params::kHistoryMalusMax);
}

// Casillas que ataca el rival del bando al que le toca, para el historial principal. La idea
// de separar el historial segun las amenazas esta en Reckless, Stormphrax, Berserk y otros.
inline Bitboard history_threats(const Position& pos) {
  return pos.attacks_by(~pos.side_to_move());
}

// Estado de un nodo durante su busqueda. Vive en la pila de la funcion `search`.
struct Node {
  StackEntry* ss;
  const Position& pos;
  int ply;
  int depth;
  Value alpha;
  Value beta;
  bool pv_node;
  bool root_node;
  bool cut_node;

  bool in_check = false;
  bool improving = false;
  bool tt_hit = false;
  bool tt_capture = false;
  Key key = 0;
  TTData tt{};
  Value tt_value = VALUE_NONE;
  Move tt_move = Move::none();
  Move excluded = Move::none();
  Value raw_eval = VALUE_NONE;
  Value eval = VALUE_NONE;
  int complexity = 0;  // cuanto movio la correccion la evaluacion de la red

  Value best_value = -VALUE_INFINITE;
  Move best_move = Move::none();
  int move_count = 0;
  Value early_return = VALUE_NONE;
  int quiet_count = 0;
  int capture_count = 0;
  std::array<Move, kMaxTried> quiets_tried{};
  std::array<Move, kMaxTried> captures_tried{};
};

// Datos de la jugada que se esta considerando dentro del bucle.
struct Candidate {
  Move move;
  Piece piece;
  Square to;
  bool capture;
  bool gives_check;
  int new_depth;
};

}  // namespace ayano

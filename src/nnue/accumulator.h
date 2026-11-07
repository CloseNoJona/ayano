// Acumuladores NNUE: la suma de pesos de las piezas presentes se actualiza de forma
// incremental, porque una jugada solo cambia de dos a cuatro entradas de la red. El
// calculo es perezoso: solo se hace para los nodos que realmente se evaluan.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "board/position.h"
#include "nnue/network.h"

namespace ayano::nnue {

struct alignas(64) Accumulator {
  std::array<std::array<std::int16_t, kHidden>, COLOR_NB> values;
};

// Ultimo acumulador visto para una perspectiva, cubeta de rey y flanco, con las piezas que
// suma. Al cambiar de cubeta basta con corregir las piezas que difieren, no sumarlas todas.
struct alignas(64) RefreshEntry {
  std::array<std::int16_t, kHidden> values;
  std::array<Bitboard, PIECE_NB> pieces;
};

// Evaluacion completa de una posicion, sin estado incremental.
Value evaluate_from_scratch(const Network& network, const Position& pos);

class AccumulatorStack {
 public:
  AccumulatorStack() : stack_(std::size_t(MAX_PLY + 2)) { reset(); }

  void reset() {
    computed_.fill(false);
    null_move_.fill(false);
    cache_ready_ = false;  // la red activa pudo cambiar entre dos busquedas
  }
  void on_move(int ply) {
    computed_[std::size_t(ply)] = false;
    null_move_[std::size_t(ply)] = false;
  }
  void on_null_move(int ply) {
    computed_[std::size_t(ply)] = false;
    null_move_[std::size_t(ply)] = true;
  }

  // `get_entry(ply)` debe devolver algo con los miembros `pos` y `move`.
  template <typename Getter>
  Value evaluate(const Network& network, int ply, Getter&& get_entry) {
    int base = ply;
    while (base > 0 && !computed_[std::size_t(base)]) --base;
    if (!computed_[std::size_t(base)]) {
      refresh(network, get_entry(base).pos, base);
      computed_[std::size_t(base)] = true;
    }
    for (int p = base + 1; p <= ply; ++p) {
      if (null_move_[std::size_t(p)]) {
        stack_[std::size_t(p)] = stack_[std::size_t(p - 1)];
      } else {
        const auto& parent = get_entry(p - 1);
        update(network, parent.pos, parent.move, get_entry(p).pos, p);
      }
      computed_[std::size_t(p)] = true;
    }
    return output(network, stack_[std::size_t(ply)], get_entry(ply).pos);
  }

 private:
  void prepare_cache(const Network& network);
  void refresh_perspective(const Network& network, const Position& pos, Color perspective,
                           std::array<std::int16_t, kHidden>& values);
  void refresh(const Network& network, const Position& pos, int ply);
  void update(const Network& network, const Position& parent, Move m, const Position& child,
              int ply);
  static Value output(const Network& network, const Accumulator& accumulator, const Position& pos);

  using RefreshCache =
      std::array<std::array<std::array<RefreshEntry, 2>, kKingBuckets>, COLOR_NB>;

  std::vector<Accumulator> stack_;
  RefreshCache cache_;
  bool cache_ready_ = false;
  std::array<bool, MAX_PLY + 2> computed_{};
  std::array<bool, MAX_PLY + 2> null_move_{};
};

}  // namespace ayano::nnue

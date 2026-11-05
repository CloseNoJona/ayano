// Generacion de jugadas pseudo-legales por etapas. La busqueda genera solo lo que
// necesita (capturas primero) y comprueba la legalidad justo antes de jugar.
#pragma once

#include <algorithm>
#include <array>
#include <string_view>

#include "board/position.h"

namespace ayano {

struct ExtMove {
  Move move;
  int value;
};

enum class GenType { Captures, Quiets, Evasions, NonEvasions };

class MoveList {
 public:
  MoveList() : size_(0) {}

  void push_back(Move m) { items_[size_++] = {m, 0}; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  ExtMove& operator[](std::size_t i) { return items_[i]; }
  const ExtMove& operator[](std::size_t i) const { return items_[i]; }
  ExtMove* begin() { return items_.data(); }
  ExtMove* end() { return items_.data() + size_; }
  const ExtMove* begin() const { return items_.data(); }
  const ExtMove* end() const { return items_.data() + size_; }
  void resize_to(const ExtMove* end) { size_ = std::size_t(end - items_.data()); }

  bool contains(Move m) const {
    return std::any_of(begin(), end(), [m](const ExtMove& entry) { return entry.move == m; });
  }

 private:
  std::array<ExtMove, MAX_MOVES> items_;
  std::size_t size_;
};

// Escribe las jugadas desde `list` y devuelve el nuevo final. Requisitos: Evasions
// solo estando en jaque; Captures, Quiets y NonEvasions solo sin jaque.
template <GenType Type>
ExtMove* generate(const Position& pos, ExtMove* list);

MoveList generate_moves(const Position& pos, GenType type);
MoveList legal_moves(const Position& pos);

// Devuelve Move::none() si el texto no es una jugada legal en esta posicion.
Move parse_uci_move(const Position& pos, std::string_view text);

}  // namespace ayano

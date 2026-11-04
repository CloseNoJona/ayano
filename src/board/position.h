// Posicion de ajedrez inmutable: una jugada produce una posicion nueva
// ("copy-make") y la original no cambia. Asi la busqueda no necesita deshacer
// jugadas, cada hilo trabaja con su propia pila y se evita toda una clase de
// errores de estado. La copia cuesta unos 200 bytes por nodo.
#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>

#include "core/attacks.h"
#include "core/bitboard.h"
#include "core/types.h"

namespace enroque {

// Valores para el intercambio estatico (SEE), en centipeones.
constexpr Value see_value(PieceType pt) {
  constexpr std::array<Value, 7> values = {100, 320, 330, 520, 980, 0, 0};
  return values[pt];
}

class Position {
 public:
  Position() = default;

  static Position startpos();
  static std::optional<Position> from_fen(std::string_view fen, std::string* error = nullptr);
  std::string fen() const;

  Bitboard pieces() const { return by_color_[WHITE] | by_color_[BLACK]; }
  Bitboard pieces(Color c) const { return by_color_[c]; }
  Bitboard pieces(PieceType pt) const { return by_type_[pt]; }
  Bitboard pieces(PieceType a, PieceType b) const { return by_type_[a] | by_type_[b]; }
  Bitboard pieces(Color c, PieceType pt) const { return by_color_[c] & by_type_[pt]; }
  Bitboard pieces(Color c, PieceType a, PieceType b) const {
    return by_color_[c] & (by_type_[a] | by_type_[b]);
  }

  Piece piece_on(Square s) const { return board_[s]; }
  bool is_empty(Square s) const { return board_[s] == NO_PIECE; }
  Square king_square(Color c) const { return king_sq_[c]; }
  Color side_to_move() const { return stm_; }
  Square ep_square() const { return ep_; }
  int castling_rights() const { return castling_; }
  int rule50() const { return rule50_; }
  int game_ply() const { return game_ply_; }

  Key key() const { return key_; }
  Key pawn_key() const { return pawn_key_; }
  Key minor_key() const { return minor_key_; }
  Key non_pawn_key(Color c) const { return non_pawn_key_[c]; }

  Bitboard checkers() const { return checkers_; }
  bool in_check() const { return checkers_ != 0; }
  // Piezas (de ambos colores) que son lo unico entre una pieza deslizante
  // rival y el rey del color indicado.
  Bitboard blockers(Color king_color) const { return blockers_[king_color]; }
  // Pieza capturada por la jugada que llevo a esta posicion.
  Piece captured_piece() const { return captured_; }

  Piece moved_piece(Move m) const { return board_[m.from()]; }
  bool is_capture(Move m) const {
    return m.kind() == MoveKind::EnPassant ||
           (m.kind() != MoveKind::Castling && board_[m.to()] != NO_PIECE);
  }
  // Tipo de la pieza capturada; NO_PIECE_TYPE si la jugada no captura.
  PieceType captured_type(Move m) const {
    return m.kind() == MoveKind::EnPassant ? PAWN : type_of(board_[m.to()]);
  }
  // Las capturas y la promocion a dama se tratan como jugadas tacticas.
  bool is_tactical(Move m) const {
    return is_capture(m) || (m.kind() == MoveKind::Promotion && m.promotion_type() == QUEEN);
  }

  // Comprueba una jugada de origen desconocido (tabla hash, killers) sin
  // verificar si deja al rey en jaque; para eso esta is_legal.
  bool is_pseudo_legal(Move m) const;
  bool is_legal(Move m) const;  // requiere que la jugada sea pseudo-legal
  bool gives_check(Move m) const;
  // True si el intercambio iniciado por `m` gana al menos `threshold`.
  bool see_ge(Move m, Value threshold) const;

  Bitboard attackers_to(Square s, Bitboard occupied) const;
  Bitboard attackers_to(Square s) const { return attackers_to(s, pieces()); }
  bool has_non_pawn_material(Color c) const {
    return (pieces(c) ^ pieces(c, PAWN, KING)) != 0;
  }
  bool is_insufficient_material() const;

  // `out` debe ser un objeto distinto de *this. Es la variante que usa la
  // busqueda para escribir directamente en su pila de posiciones.
  void apply_to(Move m, Position& out) const;
  void apply_null_to(Position& out) const;

  Position after(Move m) const {
    Position next;
    apply_to(m, next);
    return next;
  }
  Position after_null() const {
    Position next;
    apply_null_to(next);
    return next;
  }

 private:
  bool parse_placement(std::string_view text, std::string* error);
  bool parse_castling(std::string_view text, std::string* error);
  bool parse_en_passant(std::string_view text, std::string* error);
  bool validate(std::string* error) const;
  void clear();

  void toggle_keys(Piece pc, Square s);
  void put_piece(Piece pc, Square s);
  void remove_piece(Square s);
  void move_piece(Square from, Square to);
  void apply_castling(Square from, Square to);
  void update_castling_rights(Square from, Square to);
  void set_en_passant_if_capturable(Square from, Square to);
  void refresh_check_info();
  Bitboard slider_blockers(Color king_color) const;

  std::array<Bitboard, PIECE_TYPE_NB> by_type_;
  std::array<Bitboard, COLOR_NB> by_color_;
  std::array<Bitboard, COLOR_NB> blockers_;
  Bitboard checkers_;
  Key key_;
  Key pawn_key_;
  Key minor_key_;
  std::array<Key, COLOR_NB> non_pawn_key_;
  std::array<Piece, SQUARE_NB> board_;
  std::array<Square, COLOR_NB> king_sq_;
  Square ep_;
  Color stm_;
  Piece captured_;
  std::uint8_t castling_;
  std::uint8_t rule50_;
  std::uint16_t game_ply_;
};

}  // namespace enroque

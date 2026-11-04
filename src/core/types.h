// Tipos basicos del motor: colores, piezas, casillas, jugadas y valores.
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace enroque {

using Bitboard = std::uint64_t;
using Key = std::uint64_t;
using Value = int;

enum Color : std::uint8_t { WHITE, BLACK };
inline constexpr int COLOR_NB = 2;
constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : std::uint8_t { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE_TYPE };
inline constexpr int PIECE_TYPE_NB = 6;

// pieza = tipo * 2 + color: extraer tipo y color cuesta una sola instruccion.
enum Piece : std::uint8_t {
  W_PAWN, B_PAWN, W_KNIGHT, B_KNIGHT, W_BISHOP, B_BISHOP,
  W_ROOK, B_ROOK, W_QUEEN, B_QUEEN, W_KING, B_KING, NO_PIECE
};
inline constexpr int PIECE_NB = 12;

constexpr Piece make_piece(Color c, PieceType pt) { return Piece(pt * 2 + c); }
constexpr PieceType type_of(Piece p) { return PieceType(p >> 1); }
constexpr Color color_of(Piece p) { return Color(p & 1); }

enum Square : std::uint8_t {
  A1, B1, C1, D1, E1, F1, G1, H1,
  A2, B2, C2, D2, E2, F2, G2, H2,
  A3, B3, C3, D3, E3, F3, G3, H3,
  A4, B4, C4, D4, E4, F4, G4, H4,
  A5, B5, C5, D5, E5, F5, G5, H5,
  A6, B6, C6, D6, E6, F6, G6, H6,
  A7, B7, C7, D7, E7, F7, G7, H7,
  A8, B8, C8, D8, E8, F8, G8, H8,
  NO_SQUARE
};
inline constexpr int SQUARE_NB = 64;

enum Direction : int {
  NORTH = 8, SOUTH = -8, EAST = 1, WEST = -1,
  NORTH_EAST = 9, NORTH_WEST = 7, SOUTH_EAST = -7, SOUTH_WEST = -9
};

constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
constexpr int file_of(Square s) { return s & 7; }
constexpr int rank_of(Square s) { return s >> 3; }
constexpr Square make_square(int file, int rank) { return Square(rank * 8 + file); }
constexpr Direction pawn_push(Color c) { return c == WHITE ? NORTH : SOUTH; }
constexpr int relative_rank(Color c, Square s) { return c == WHITE ? rank_of(s) : 7 - rank_of(s); }
constexpr Square relative_square(Color c, Square s) { return Square(s ^ (c * 56)); }

enum CastlingRight : std::uint8_t {
  NO_CASTLING = 0,
  WHITE_OO = 1,
  WHITE_OOO = 2,
  BLACK_OO = 4,
  BLACK_OOO = 8,
  ALL_CASTLING = 15
};
inline constexpr int CASTLING_NB = 16;

enum class MoveKind : std::uint16_t { Normal = 0, Promotion = 1, EnPassant = 2, Castling = 3 };

// Jugada en 16 bits: origen (0-5), destino (6-11), promocion (12-13), tipo (14-15).
// El constructor por defecto no inicializa a proposito: las listas de jugadas
// se crean millones de veces por segundo. Usa Move::none() para el valor vacio.
class Move {
 public:
  Move() = default;

  static constexpr Move none() { return Move(std::uint16_t(0)); }
  static constexpr Move null() { return Move(std::uint16_t(65)); }  // b1b1: origen == destino
  static constexpr Move from_raw(std::uint16_t raw) { return Move(raw); }

  static constexpr Move normal(Square from, Square to) {
    return Move(std::uint16_t(from | (to << 6)));
  }
  static constexpr Move promotion(Square from, Square to, PieceType promoted) {
    return Move(std::uint16_t(from | (to << 6) | ((promoted - KNIGHT) << 12) |
                              (int(MoveKind::Promotion) << 14)));
  }
  static constexpr Move en_passant(Square from, Square to) {
    return Move(std::uint16_t(from | (to << 6) | (int(MoveKind::EnPassant) << 14)));
  }
  static constexpr Move castling(Square from, Square to) {
    return Move(std::uint16_t(from | (to << 6) | (int(MoveKind::Castling) << 14)));
  }

  constexpr Square from() const { return Square(data_ & 63); }
  constexpr Square to() const { return Square((data_ >> 6) & 63); }
  constexpr MoveKind kind() const { return MoveKind(data_ >> 14); }
  constexpr PieceType promotion_type() const { return PieceType(((data_ >> 12) & 3) + KNIGHT); }
  constexpr bool has_promotion_bits() const { return (data_ >> 12 & 3) != 0; }
  constexpr std::uint16_t raw() const { return data_; }
  constexpr std::uint16_t from_to() const { return data_ & 0xFFF; }
  // Falso para none() y null(), las dos unicas jugadas con origen == destino.
  constexpr bool is_ok() const { return from() != to(); }

  friend constexpr bool operator==(Move a, Move b) { return a.data_ == b.data_; }
  friend constexpr bool operator!=(Move a, Move b) { return a.data_ != b.data_; }

  std::string to_uci() const;

 private:
  explicit constexpr Move(std::uint16_t data) : data_(data) {}
  std::uint16_t data_;
};

inline constexpr int MAX_PLY = 246;
inline constexpr int MAX_MOVES = 320;

inline constexpr Value VALUE_ZERO = 0;
inline constexpr Value VALUE_DRAW = 0;
inline constexpr Value VALUE_MATE = 32000;
inline constexpr Value VALUE_INFINITE = 32001;
inline constexpr Value VALUE_NONE = 32002;
inline constexpr Value VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY;
inline constexpr Value VALUE_MATED_IN_MAX_PLY = -VALUE_MATE_IN_MAX_PLY;

constexpr Value mate_in(int ply) { return VALUE_MATE - ply; }
constexpr Value mated_in(int ply) { return -VALUE_MATE + ply; }
constexpr bool is_mate_score(Value v) {
  return v >= VALUE_MATE_IN_MAX_PLY || v <= VALUE_MATED_IN_MAX_PLY;
}

std::string square_to_string(Square s);

}  // namespace enroque

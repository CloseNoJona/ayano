// Lectura y escritura de FEN. Un FEN viene de fuera del motor, asi que se valida
// por completo: una posicion imposible podria desbordar las listas de jugadas.
#include <charconv>
#include <sstream>
#include <vector>

#include "board/position.h"
#include "core/zobrist.h"

namespace enroque {

namespace {

constexpr std::string_view kPieceChars = "PpNnBbRrQqKk";
constexpr std::string_view kStartFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr int kMaxPiecesPerSide = 16;
constexpr int kMaxPawnsPerSide = 8;
constexpr int kMaxRule50 = 255;
constexpr int kMaxCheckers = 2;

bool fail(std::string* error, const std::string& message) {
  if (error) *error = message;
  return false;
}

std::vector<std::string_view> split_fields(std::string_view text) {
  std::vector<std::string_view> fields;
  std::size_t pos = 0;
  while (pos < text.size()) {
    const std::size_t start = text.find_first_not_of(" \t\r\n", pos);
    if (start == std::string_view::npos) break;
    std::size_t end = text.find_first_of(" \t\r\n", start);
    if (end == std::string_view::npos) end = text.size();
    fields.push_back(text.substr(start, end - start));
    pos = end;
  }
  return fields;
}

std::optional<int> parse_non_negative(std::string_view text) {
  int value = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, value);
  if (ec != std::errc() || ptr != end || value < 0) return std::nullopt;
  return value;
}

}  // namespace

void Position::clear() {
  by_type_.fill(0);
  by_color_.fill(0);
  blockers_.fill(0);
  board_.fill(NO_PIECE);
  non_pawn_key_.fill(0);
  king_sq_.fill(NO_SQUARE);
  checkers_ = 0;
  key_ = 0;
  pawn_key_ = 0;
  minor_key_ = 0;
  ep_ = NO_SQUARE;
  stm_ = WHITE;
  captured_ = NO_PIECE;
  castling_ = 0;
  rule50_ = 0;
  game_ply_ = 0;
}

bool Position::parse_placement(std::string_view text, std::string* error) {
  int file = 0;
  int rank = 7;
  for (const char c : text) {
    if (c == '/') {
      if (file != 8) return fail(error, "cada fila del FEN debe tener 8 casillas");
      if (--rank < 0) return fail(error, "el FEN tiene mas de 8 filas");
      file = 0;
    } else if (c >= '1' && c <= '8') {
      file += c - '0';
      if (file > 8) return fail(error, "fila del FEN demasiado larga");
    } else {
      const std::size_t index = kPieceChars.find(c);
      if (index == std::string_view::npos) {
        return fail(error, std::string("caracter invalido en el FEN: ") + c);
      }
      if (file > 7) return fail(error, "fila del FEN demasiado larga");
      put_piece(Piece(index), make_square(file++, rank));
    }
  }
  if (rank != 0 || file != 8) return fail(error, "el FEN debe tener exactamente 8 filas");
  return true;
}

bool Position::parse_castling(std::string_view text, std::string* error) {
  if (text == "-") return true;
  for (const char c : text) {
    Color color = WHITE;
    int right = 0;
    Square rook_square = H1;
    switch (c) {
      case 'K': right = WHITE_OO; rook_square = H1; break;
      case 'Q': right = WHITE_OOO; rook_square = A1; break;
      case 'k': right = BLACK_OO; rook_square = H8; color = BLACK; break;
      case 'q': right = BLACK_OOO; rook_square = A8; color = BLACK; break;
      default: return fail(error, std::string("derecho de enroque invalido: ") + c);
    }
    // Un derecho sin rey o torre en su casilla inicial no se puede ejercer: se ignora.
    const Square king_home = color == WHITE ? E1 : E8;
    if (board_[king_home] == make_piece(color, KING) &&
        board_[rook_square] == make_piece(color, ROOK)) {
      castling_ |= std::uint8_t(right);
    }
  }
  return true;
}

bool Position::parse_en_passant(std::string_view text, std::string* error) {
  if (text == "-") return true;
  if (text.size() != 2 || text[0] < 'a' || text[0] > 'h') {
    return fail(error, "casilla al paso invalida");
  }
  const int expected_rank = stm_ == WHITE ? 5 : 2;
  if (text[1] - '1' != expected_rank) return fail(error, "casilla al paso en fila incorrecta");
  const Square target = make_square(text[0] - 'a', expected_rank);
  const Square pushed = target - pawn_push(stm_);
  const bool pawn_is_there = board_[pushed] == make_piece(~stm_, PAWN);
  if (pawn_is_there && (attacks::pawn(~stm_, target) & pieces(stm_, PAWN))) ep_ = target;
  return true;
}

bool Position::validate(std::string* error) const {
  for (const Color c : {WHITE, BLACK}) {
    if (popcount(pieces(c, KING)) != 1) return fail(error, "cada bando necesita exactamente un rey");
    if (popcount(pieces(c)) > kMaxPiecesPerSide) return fail(error, "demasiadas piezas en un bando");
    if (popcount(pieces(c, PAWN)) > kMaxPawnsPerSide) return fail(error, "demasiados peones");
  }
  if (pieces(PAWN) & (RANK_1_BB | RANK_8_BB)) return fail(error, "peon en la primera u octava fila");
  return true;
}

std::optional<Position> Position::from_fen(std::string_view fen, std::string* error) {
  const std::vector<std::string_view> fields = split_fields(fen);
  if (fields.size() < 4 || fields.size() > 6) {
    fail(error, "un FEN necesita entre 4 y 6 campos");
    return std::nullopt;
  }

  Position pos;
  pos.clear();
  if (!pos.parse_placement(fields[0], error)) return std::nullopt;
  if (!pos.validate(error)) return std::nullopt;

  if (fields[1] != "w" && fields[1] != "b") {
    fail(error, "el turno debe ser 'w' o 'b'");
    return std::nullopt;
  }
  pos.stm_ = fields[1] == "w" ? WHITE : BLACK;
  pos.king_sq_[WHITE] = lsb(pos.pieces(WHITE, KING));
  pos.king_sq_[BLACK] = lsb(pos.pieces(BLACK, KING));

  if (!pos.parse_castling(fields[2], error)) return std::nullopt;
  if (!pos.parse_en_passant(fields[3], error)) return std::nullopt;

  int full_move = 1;
  if (fields.size() >= 5) {
    const auto rule50 = parse_non_negative(fields[4]);
    if (!rule50 || *rule50 > kMaxRule50) {
      fail(error, "contador de medias jugadas invalido");
      return std::nullopt;
    }
    pos.rule50_ = std::uint8_t(*rule50);
  }
  if (fields.size() == 6) {
    const auto parsed = parse_non_negative(fields[5]);
    if (!parsed || *parsed > 30000) {
      fail(error, "numero de jugada invalido");
      return std::nullopt;
    }
    full_move = *parsed == 0 ? 1 : *parsed;
  }
  pos.game_ply_ = std::uint16_t(2 * (full_move - 1) + (pos.stm_ == BLACK ? 1 : 0));

  if (pos.attackers_to(pos.king_sq_[~pos.stm_]) & pos.by_color_[pos.stm_]) {
    fail(error, "el bando sin turno esta en jaque");
    return std::nullopt;
  }

  if (pos.stm_ == BLACK) pos.key_ ^= zobrist::side();
  pos.key_ ^= zobrist::castling(pos.castling_);
  if (pos.ep_ != NO_SQUARE) pos.key_ ^= zobrist::en_passant(file_of(pos.ep_));
  pos.refresh_check_info();
  if (popcount(pos.checkers_) > kMaxCheckers) {
    fail(error, "ninguna jugada legal produce mas de dos jaques a la vez");
    return std::nullopt;
  }
  return pos;
}

Position Position::startpos() {
  std::optional<Position> pos = from_fen(kStartFen);
  return *pos;  // el FEN inicial es una constante valida
}

std::string Position::fen() const {
  std::ostringstream out;
  for (int rank = 7; rank >= 0; --rank) {
    int empty_count = 0;
    for (int file = 0; file < 8; ++file) {
      const Piece pc = board_[make_square(file, rank)];
      if (pc == NO_PIECE) {
        ++empty_count;
        continue;
      }
      if (empty_count) out << empty_count;
      empty_count = 0;
      out << kPieceChars[pc];
    }
    if (empty_count) out << empty_count;
    if (rank > 0) out << '/';
  }

  out << (stm_ == WHITE ? " w " : " b ");
  if (castling_ == 0) out << '-';
  if (castling_ & WHITE_OO) out << 'K';
  if (castling_ & WHITE_OOO) out << 'Q';
  if (castling_ & BLACK_OO) out << 'k';
  if (castling_ & BLACK_OOO) out << 'q';

  out << ' ' << (ep_ == NO_SQUARE ? std::string("-") : square_to_string(ep_));
  out << ' ' << int(rule50_) << ' ' << (game_ply_ / 2 + 1);
  return out.str();
}

}  // namespace enroque

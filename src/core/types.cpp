#include "core/types.h"

namespace ayano {

std::string square_to_string(Square s) {
  std::string text;
  text += char('a' + file_of(s));
  text += char('1' + rank_of(s));
  return text;
}

std::string Move::to_uci() const {
  if (*this == Move::none()) return "0000";
  if (*this == Move::null()) return "0000";
  std::string text = square_to_string(from()) + square_to_string(to());
  if (kind() == MoveKind::Promotion) text += " nbrq"[promotion_type()];
  return text;
}

}  // namespace ayano

#include "eval/classical.h"

#include <algorithm>
#include <cstdlib>

#include "eval/pst.h"

namespace enroque::classical {

namespace {

struct Score {
  int mg = 0;
  int eg = 0;

  Score& operator+=(Score other) {
    mg += other.mg;
    eg += other.eg;
    return *this;
  }
  Score& operator-=(Score other) {
    mg -= other.mg;
    eg -= other.eg;
    return *this;
  }
};

constexpr Score operator*(Score s, int factor) { return {s.mg * factor, s.eg * factor}; }

constexpr int kMaxPhase = 24;
constexpr std::array<int, 6> kPhaseWeight = {0, 1, 1, 2, 4, 0};

constexpr Score kBishopPair = {25, 45};
constexpr Score kIsolatedPawn = {-8, -12};
constexpr Score kDoubledPawn = {-6, -18};
constexpr Score kConnectedPawn = {6, 5};
constexpr Score kRookOpenFile = {22, 8};
constexpr Score kRookSemiOpenFile = {10, 5};
constexpr int kTempo = 12;

constexpr std::array<Score, 8> kPassedPawn = {{{0, 0}, {2, 8}, {4, 12}, {10, 25},
                                               {25, 50}, {50, 95}, {85, 150}, {0, 0}}};

// Bonus por casilla de movilidad a partir de una cifra central para cada pieza.
constexpr std::array<Score, 6> kMobilityWeight = {{{0, 0}, {4, 4}, {4, 5}, {2, 4}, {1, 2}, {0, 0}}};
constexpr std::array<int, 6> kMobilityCenter = {0, 4, 6, 7, 13, 0};
constexpr std::array<int, 6> kKingAttackWeight = {0, 2, 2, 3, 5, 0};

struct SideInfo {
  Bitboard pawn_attacks = 0;
  int king_attack_units = 0;
  int king_attackers = 0;
};

Bitboard pawn_attack_span(Color c, Bitboard pawns) {
  return c == WHITE ? shift<NORTH_EAST>(pawns) | shift<NORTH_WEST>(pawns)
                    : shift<SOUTH_EAST>(pawns) | shift<SOUTH_WEST>(pawns);
}

Bitboard forward_ranks(Color c, Square s) {
  const int rank = rank_of(s);
  if (c == WHITE) return rank == 7 ? 0 : ~Bitboard(0) << (8 * (rank + 1));
  return rank == 0 ? 0 : ~Bitboard(0) >> (8 * (8 - rank));
}

Bitboard adjacent_files(Square s) {
  const Bitboard file = file_bb(file_of(s));
  return shift<EAST>(file) | shift<WEST>(file);
}

Score material_and_position(const Position& pos, Color c) {
  Score score;
  for (int pt = PAWN; pt <= KING; ++pt) {
    Bitboard pieces = pos.pieces(c, PieceType(pt));
    while (pieces) {
      // Las tablas tienen la octava fila arriba: las blancas invierten la fila.
      const int index = pop_lsb(pieces) ^ (c == WHITE ? 56 : 0);
      score.mg += kMaterialMg[pt] + kPstMg[pt][index];
      score.eg += kMaterialEg[pt] + kPstEg[pt][index];
    }
  }
  return score;
}

Score passed_pawn_score(const Position& pos, Color c, Square s) {
  const int rank = relative_rank(c, s);
  Score score = kPassedPawn[rank];
  const Square ahead = s + pawn_push(c);
  const auto king_distance = [ahead](Square king) {
    return std::max(std::abs(file_of(king) - file_of(ahead)),
                    std::abs(rank_of(king) - rank_of(ahead)));
  };
  if (rank >= 3) {
    const int distance_gap = king_distance(pos.king_square(~c)) - king_distance(pos.king_square(c));
    score.eg += distance_gap * (rank - 2) * 4;
  }
  return score;
}

Score pawn_structure(const Position& pos, Color c) {
  Score score;
  const Bitboard own = pos.pieces(c, PAWN);
  const Bitboard enemy = pos.pieces(~c, PAWN);
  Bitboard pawns = own;
  while (pawns) {
    const Square s = pop_lsb(pawns);
    const Bitboard file = file_bb(file_of(s));
    const Bitboard neighbours = adjacent_files(s);
    const Bitboard ahead = forward_ranks(c, s);

    if (!(own & neighbours)) score += kIsolatedPawn;
    if (own & file & ahead) score += kDoubledPawn;
    if (attacks::pawn(~c, s) & own) score += kConnectedPawn;
    if (!(enemy & (file | neighbours) & ahead)) score += passed_pawn_score(pos, c, s);
  }
  return score;
}

Score piece_activity(const Position& pos, Color c, const SideInfo& enemy, SideInfo& info) {
  Score score;
  const Bitboard occupied = pos.pieces();
  const Bitboard mobility_area = ~(pos.pieces(c, PAWN, KING) | enemy.pawn_attacks);
  const Bitboard enemy_king_zone =
      attacks::king(pos.king_square(~c)) | square_bb(pos.king_square(~c));

  for (int pt = KNIGHT; pt <= QUEEN; ++pt) {
    Bitboard pieces = pos.pieces(c, PieceType(pt));
    while (pieces) {
      const Square s = pop_lsb(pieces);
      const Bitboard moves = attacks::of(PieceType(pt), s, occupied);
      score += kMobilityWeight[pt] * (popcount(moves & mobility_area) - kMobilityCenter[pt]);

      const Bitboard on_king = moves & enemy_king_zone;
      if (on_king) {
        info.king_attack_units += kKingAttackWeight[pt] * popcount(on_king);
        ++info.king_attackers;
      }
      if (pt == ROOK) {
        const Bitboard file = file_bb(file_of(s));
        if (!(pos.pieces(PAWN) & file)) {
          score += kRookOpenFile;
        } else if (!(pos.pieces(c, PAWN) & file)) {
          score += kRookSemiOpenFile;
        }
      }
    }
  }
  if (popcount(pos.pieces(c, BISHOP)) >= 2) score += kBishopPair;
  return score;
}

// Penalizacion cuadratica: un ataque coordinado vale mas que la suma de sus partes.
Score king_danger(const Position& pos, Color attacker, const SideInfo& info) {
  if (info.king_attackers < 2 || !pos.pieces(attacker, QUEEN)) return {};
  const int units = info.king_attack_units + 2 * info.king_attackers;
  return {std::min(units * units / 3, 500), units};
}

Score king_shelter(const Position& pos, Color c) {
  const Square king = pos.king_square(c);
  const Bitboard shield_zone = attacks::king(king) | shift<NORTH>(attacks::king(king)) |
                               shift<SOUTH>(attacks::king(king));
  const Bitboard in_front = shield_zone & forward_ranks(c, king);
  const int shield = std::min(popcount(in_front & pos.pieces(c, PAWN)), 3);
  return {(shield - 2) * 12, 0};
}

// Sin peones, una ventaja menor que una torre rara vez se convierte en victoria.
int endgame_scale(const Position& pos, Color strong) {
  if (pos.pieces(strong, PAWN)) return 64;
  const auto minor_or_major = [&pos](Color c) {
    return 3 * popcount(pos.pieces(c, KNIGHT, BISHOP)) + 5 * popcount(pos.pieces(c, ROOK)) +
           9 * popcount(pos.pieces(c, QUEEN));
  };
  return minor_or_major(strong) - minor_or_major(~strong) <= 3 ? 12 : 64;
}

int game_phase(const Position& pos) {
  int phase = 0;
  for (int pt = KNIGHT; pt <= QUEEN; ++pt) {
    phase += kPhaseWeight[pt] * popcount(pos.pieces(PieceType(pt)));
  }
  return std::min(phase, kMaxPhase);
}

}  // namespace

Value evaluate(const Position& pos) {
  std::array<SideInfo, COLOR_NB> info;
  info[WHITE].pawn_attacks = pawn_attack_span(WHITE, pos.pieces(WHITE, PAWN));
  info[BLACK].pawn_attacks = pawn_attack_span(BLACK, pos.pieces(BLACK, PAWN));

  Score score;
  score += material_and_position(pos, WHITE);
  score -= material_and_position(pos, BLACK);
  score += pawn_structure(pos, WHITE);
  score -= pawn_structure(pos, BLACK);
  score += piece_activity(pos, WHITE, info[BLACK], info[WHITE]);
  score -= piece_activity(pos, BLACK, info[WHITE], info[BLACK]);
  score += king_danger(pos, WHITE, info[WHITE]);
  score -= king_danger(pos, BLACK, info[BLACK]);
  score += king_shelter(pos, WHITE);
  score -= king_shelter(pos, BLACK);

  const int phase = game_phase(pos);
  int value = (score.mg * phase + score.eg * (kMaxPhase - phase)) / kMaxPhase;
  value = value * endgame_scale(pos, value >= 0 ? WHITE : BLACK) / 64;

  const int relative = pos.side_to_move() == WHITE ? value : -value;
  return relative + kTempo;
}

}  // namespace enroque::classical

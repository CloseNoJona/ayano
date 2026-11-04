#include "core/attacks.h"

#include <cstdlib>
#include <mutex>
#include <vector>

#include "core/prng.h"

namespace enroque::attacks {

Tables tables;

namespace {

constexpr std::size_t kRookTableSize = 102400;
constexpr std::size_t kBishopTableSize = 5248;

std::array<Bitboard, kRookTableSize> rook_table;
std::array<Bitboard, kBishopTableSize> bishop_table;

constexpr std::array<Direction, 4> kRookDirections = {NORTH, SOUTH, EAST, WEST};
constexpr std::array<Direction, 4> kBishopDirections = {NORTH_EAST, NORTH_WEST, SOUTH_EAST,
                                                        SOUTH_WEST};

bool can_step(Square s, Direction d) {
  const int target = int(s) + int(d);
  if (target < 0 || target >= SQUARE_NB) return false;
  return std::abs(file_of(Square(target)) - file_of(s)) <= 1;
}

Bitboard step_attack(Square s, std::initializer_list<int> offsets) {
  Bitboard result = 0;
  for (const int offset : offsets) {
    const int target = int(s) + offset;
    if (target < 0 || target >= SQUARE_NB) continue;
    if (std::abs(file_of(Square(target)) - file_of(s)) <= 2) result |= square_bb(Square(target));
  }
  return result;
}

Bitboard slide(Square s, Bitboard occupied, const std::array<Direction, 4>& directions) {
  Bitboard result = 0;
  for (const Direction d : directions) {
    Square current = s;
    while (can_step(current, d)) {
      current = current + d;
      result |= square_bb(current);
      if (occupied & square_bb(current)) break;
    }
  }
  return result;
}

void init_leapers() {
  for (int i = 0; i < SQUARE_NB; ++i) {
    const Square s = Square(i);
    tables.knight[s] = step_attack(s, {17, 15, 10, 6, -6, -10, -15, -17});
    tables.king[s] = step_attack(s, {9, 8, 7, 1, -1, -7, -8, -9});
    const Bitboard bb = square_bb(s);
    tables.pawn[WHITE][s] = shift<NORTH_EAST>(bb) | shift<NORTH_WEST>(bb);
    tables.pawn[BLACK][s] = shift<SOUTH_EAST>(bb) | shift<SOUTH_WEST>(bb);
  }
}

// Las casillas del borde no cambian el ataque, asi que no forman parte de la mascara.
Bitboard relevant_mask(Square s, const std::array<Direction, 4>& directions) {
  const Bitboard edges = ((RANK_1_BB | RANK_8_BB) & ~rank_bb(rank_of(s))) |
                         ((FILE_A_BB | FILE_H_BB) & ~file_bb(file_of(s)));
  return slide(s, 0, directions) & ~edges;
}

#ifndef ENROQUE_USE_PEXT
// Busca un multiplicador sin colisiones destructivas para esta casilla.
Bitboard find_magic(SliderEntry& entry, const std::vector<Bitboard>& occupancies,
                    const std::vector<Bitboard>& references, Bitboard* table, Prng& rng) {
  const std::size_t size = occupancies.size();
  std::vector<int> epoch(size, 0);
  int attempt = 0;
  while (true) {
    entry.magic = rng.next() & rng.next() & rng.next();
    if (popcount((entry.magic * entry.mask) >> 56) < 6) continue;
    ++attempt;
    bool collision = false;
    for (std::size_t i = 0; i < size && !collision; ++i) {
      const unsigned index = entry.index(occupancies[i]);
      if (epoch[index] < attempt) {
        epoch[index] = attempt;
        table[index] = references[i];
      } else if (table[index] != references[i]) {
        collision = true;
      }
    }
    if (!collision) return entry.magic;
  }
}
#endif

void init_sliders(std::array<SliderEntry, SQUARE_NB>& entries, Bitboard* table,
                  const std::array<Direction, 4>& directions) {
  [[maybe_unused]] Prng rng(0x9E3779B97F4A7C15ULL);
  std::vector<Bitboard> occupancies;
  std::vector<Bitboard> references;
  Bitboard* next_table = table;

  for (int i = 0; i < SQUARE_NB; ++i) {
    const Square s = Square(i);
    SliderEntry& entry = entries[s];
    entry.mask = relevant_mask(s, directions);
    entry.shift = 64 - unsigned(popcount(entry.mask));
    entry.magic = 0;
    entry.table = next_table;

    occupancies.clear();
    references.clear();
    Bitboard subset = 0;
    do {  // Recorre todos los subconjuntos de la mascara (carry-rippler).
      occupancies.push_back(subset);
      references.push_back(slide(s, subset, directions));
      subset = (subset - entry.mask) & entry.mask;
    } while (subset != 0);

#ifndef ENROQUE_USE_PEXT
    find_magic(entry, occupancies, references, next_table, rng);
#endif
    for (std::size_t j = 0; j < occupancies.size(); ++j) {
      next_table[entry.index(occupancies[j])] = references[j];
    }
    next_table += occupancies.size();
  }
}

void init_lines() {
  for (int a = 0; a < SQUARE_NB; ++a) {
    for (int b = 0; b < SQUARE_NB; ++b) {
      const Square sa = Square(a);
      const Square sb = Square(b);
      tables.line[sa][sb] = 0;
      tables.between[sa][sb] = 0;
      if (a == b) continue;
      for (const bool diagonal : {false, true}) {
        const auto& directions = diagonal ? kBishopDirections : kRookDirections;
        const Bitboard from_a = slide(sa, 0, directions);
        if (!(from_a & square_bb(sb))) continue;
        tables.line[sa][sb] =
            (from_a & slide(sb, 0, directions)) | square_bb(sa) | square_bb(sb);
        tables.between[sa][sb] =
            slide(sa, square_bb(sb), directions) & slide(sb, square_bb(sa), directions);
      }
    }
  }
}

}  // namespace

void init() {
  static std::once_flag once;
  std::call_once(once, [] {
    init_leapers();
    init_sliders(tables.rook, rook_table.data(), kRookDirections);
    init_sliders(tables.bishop, bishop_table.data(), kBishopDirections);
    init_lines();
  });
}

}  // namespace enroque::attacks

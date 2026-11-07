#include "nnue/accumulator.h"

#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ayano::nnue {

namespace {

struct PieceChange {
  Piece piece;
  Square square;
};

// Una jugada anade como mucho dos piezas y quita como mucho dos.
struct Delta {
  std::array<PieceChange, 2> added;
  std::array<PieceChange, 2> removed;
  int added_count = 0;
  int removed_count = 0;

  void add(Piece pc, Square s) { added[std::size_t(added_count++)] = {pc, s}; }
  void remove(Piece pc, Square s) { removed[std::size_t(removed_count++)] = {pc, s}; }
};

Delta compute_delta(const Position& parent, Move m) {
  Delta delta;
  const Color us = parent.side_to_move();
  const Square from = m.from();
  const Square to = m.to();
  const Piece moved = parent.piece_on(from);

  delta.remove(moved, from);
  switch (m.kind()) {
    case MoveKind::Normal:
      delta.add(moved, to);
      if (parent.piece_on(to) != NO_PIECE) delta.remove(parent.piece_on(to), to);
      break;
    case MoveKind::Promotion:
      delta.add(make_piece(us, m.promotion_type()), to);
      if (parent.piece_on(to) != NO_PIECE) delta.remove(parent.piece_on(to), to);
      break;
    case MoveKind::EnPassant:
      delta.add(moved, to);
      delta.remove(make_piece(~us, PAWN), to - pawn_push(us));
      break;
    case MoveKind::Castling: {
      const bool king_side = to > from;
      const Piece rook = make_piece(us, ROOK);
      delta.add(moved, to);
      delta.remove(rook, make_square(king_side ? 7 : 0, rank_of(from)));
      delta.add(rook, make_square(king_side ? 5 : 3, rank_of(from)));
      break;
    }
  }
  return delta;
}

const std::int16_t* feature_row(const Network& network, int feature) {
  return network.feature_weights.data() + std::size_t(feature) * kHidden;
}

void sum_all_pieces(const Network& network, const Position& pos, Color perspective,
                    std::array<std::int16_t, kHidden>& values) {
  values = network.feature_bias;
  const Square king = pos.king_square(perspective);
  Bitboard occupied = pos.pieces();
  while (occupied) {
    const Square s = pop_lsb(occupied);
    const int feature = feature_index(perspective, king, pos.piece_on(s), s);
    const std::int16_t* row = feature_row(network, feature);
    for (int i = 0; i < kHidden; ++i) values[std::size_t(i)] += row[i];
  }
}

#if defined(__AVX2__)

constexpr int kLanes = 16;

inline __m256i load_lanes(const std::int16_t* values) {
  return _mm256_load_si256(reinterpret_cast<const __m256i*>(values));
}

// El numero de filas va en la plantilla para que el bucle interno no tenga saltos: esta
// funcion recorre toda la capa oculta en cada nodo evaluado.
template <int AddCount, int SubCount>
void apply_fixed_rows(const std::int16_t* from, std::int16_t* to,
                      const std::array<const std::int16_t*, 2>& adds,
                      const std::array<const std::int16_t*, 2>& subs) {
  for (int i = 0; i < kHidden; i += kLanes) {
    __m256i value = load_lanes(from + i);
    if constexpr (AddCount >= 1) value = _mm256_add_epi16(value, load_lanes(adds[0] + i));
    if constexpr (AddCount >= 2) value = _mm256_add_epi16(value, load_lanes(adds[1] + i));
    if constexpr (SubCount >= 1) value = _mm256_sub_epi16(value, load_lanes(subs[0] + i));
    if constexpr (SubCount >= 2) value = _mm256_sub_epi16(value, load_lanes(subs[1] + i));
    _mm256_store_si256(reinterpret_cast<__m256i*>(to + i), value);
  }
}

void apply_rows(const std::int16_t* from, std::int16_t* to,
                const std::array<const std::int16_t*, 2>& adds, int add_count,
                const std::array<const std::int16_t*, 2>& subs, int sub_count) {
  switch (add_count * 3 + sub_count) {
    case 1: apply_fixed_rows<0, 1>(from, to, adds, subs); break;
    case 2: apply_fixed_rows<0, 2>(from, to, adds, subs); break;
    case 3: apply_fixed_rows<1, 0>(from, to, adds, subs); break;
    case 4: apply_fixed_rows<1, 1>(from, to, adds, subs); break;
    case 5: apply_fixed_rows<1, 2>(from, to, adds, subs); break;
    case 6: apply_fixed_rows<2, 0>(from, to, adds, subs); break;
    case 7: apply_fixed_rows<2, 1>(from, to, adds, subs); break;
    case 8: apply_fixed_rows<2, 2>(from, to, adds, subs); break;
    default: apply_fixed_rows<0, 0>(from, to, adds, subs); break;
  }
}

// Suma de clamp(x)^2 * w. El truco: (x * w) cabe en 16 bits porque x <= 255 y
// |w| <= 128, asi que madd puede multiplicar otra vez por x sin desbordar.
std::int64_t screlu_dot(const std::int16_t* values, const std::int16_t* weights) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i limit = _mm256_set1_epi16(kQuantActivation);
  __m256i sum = _mm256_setzero_si256();
  for (int i = 0; i < kHidden; i += kLanes) {
    __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(values + i));
    const __m256i w = _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + i));
    x = _mm256_min_epi16(_mm256_max_epi16(x, zero), limit);
    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_mullo_epi16(x, w), x));
  }
  // Cada carril cabe en 32 bits, pero la suma de los ocho puede no caber.
  const __m256i wide = _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum)),
                                        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum, 1)));
  const __m128i half = _mm_add_epi64(_mm256_castsi256_si128(wide),
                                     _mm256_extracti128_si256(wide, 1));
  return _mm_cvtsi128_si64(_mm_add_epi64(half, _mm_unpackhi_epi64(half, half)));
}

#else

void apply_rows(const std::int16_t* from, std::int16_t* to,
                const std::array<const std::int16_t*, 2>& adds, int add_count,
                const std::array<const std::int16_t*, 2>& subs, int sub_count) {
  for (int i = 0; i < kHidden; ++i) {
    int value = from[i];
    for (int a = 0; a < add_count; ++a) value += adds[std::size_t(a)][i];
    for (int s = 0; s < sub_count; ++s) value -= subs[std::size_t(s)][i];
    to[i] = std::int16_t(value);
  }
}

std::int64_t screlu_dot(const std::int16_t* values, const std::int16_t* weights) {
  std::int64_t sum = 0;
  for (int i = 0; i < kHidden; ++i) {
    const int x = std::clamp(int(values[i]), 0, kQuantActivation);
    sum += x * x * weights[i];
  }
  return sum;
}

#endif

// Lleva un acumulador de un conjunto de piezas a otro, de cuatro filas en cuatro.
class RowBatch {
 public:
  RowBatch(const Network& network, std::int16_t* values) : network_(network), values_(values) {}
  ~RowBatch() { flush(); }
  RowBatch(const RowBatch&) = delete;
  RowBatch& operator=(const RowBatch&) = delete;

  void add(int feature) {
    adds_[std::size_t(add_count_++)] = feature_row(network_, feature);
    if (add_count_ == 2) flush();
  }
  void remove(int feature) {
    subs_[std::size_t(sub_count_++)] = feature_row(network_, feature);
    if (sub_count_ == 2) flush();
  }

 private:
  void flush() {
    if (add_count_ + sub_count_ > 0) apply_rows(values_, values_, adds_, add_count_, subs_, sub_count_);
    add_count_ = sub_count_ = 0;
  }

  const Network& network_;
  std::int16_t* values_;
  std::array<const std::int16_t*, 2> adds_{};
  std::array<const std::int16_t*, 2> subs_{};
  int add_count_ = 0;
  int sub_count_ = 0;
};

Value network_output(const Network& network, const Accumulator& accumulator, const Position& pos) {
  const Color us = pos.side_to_move();
  const int bucket = output_bucket(popcount(pos.pieces()));
  const std::int16_t* weights = network.output_weights.data() + std::size_t(bucket) * 2 * kHidden;
  const std::int64_t sum = screlu_dot(accumulator.values[us].data(), weights) +
                           screlu_dot(accumulator.values[~us].data(), weights + kHidden);
  const std::int64_t scaled = sum / kQuantActivation + network.output_bias[std::size_t(bucket)];
  return Value(scaled * network.eval_scale / (kQuantActivation * kQuantWeight));
}

}  // namespace

void AccumulatorStack::prepare_cache(const Network& network) {
  for (auto& perspective : cache_) {
    for (auto& bucket : perspective) {
      for (RefreshEntry& entry : bucket) {
        entry.values = network.feature_bias;
        entry.pieces.fill(0);
      }
    }
  }
  cache_ready_ = true;
}

void AccumulatorStack::refresh_perspective(const Network& network, const Position& pos,
                                           Color perspective,
                                           std::array<std::int16_t, kHidden>& values) {
  if (!cache_ready_) prepare_cache(network);
  const Square king = pos.king_square(perspective);
  RefreshEntry& entry =
      cache_[perspective][std::size_t(king_bucket(perspective, king))][std::size_t(is_mirrored(king))];
  {
    RowBatch batch(network, entry.values.data());
    for (int index = 0; index < PIECE_NB; ++index) {
      const Piece pc = Piece(index);
      const Bitboard now = pos.pieces(color_of(pc), type_of(pc));
      Bitboard added = now & ~entry.pieces[std::size_t(index)];
      Bitboard removed = entry.pieces[std::size_t(index)] & ~now;
      while (added) batch.add(feature_index(perspective, king, pc, pop_lsb(added)));
      while (removed) batch.remove(feature_index(perspective, king, pc, pop_lsb(removed)));
      entry.pieces[std::size_t(index)] = now;
    }
  }
  values = entry.values;
}

void AccumulatorStack::refresh(const Network& network, const Position& pos, int ply) {
  Accumulator& accumulator = stack_[std::size_t(ply)];
  refresh_perspective(network, pos, WHITE, accumulator.values[WHITE]);
  refresh_perspective(network, pos, BLACK, accumulator.values[BLACK]);
}

void AccumulatorStack::update(const Network& network, const Position& parent, Move m,
                              const Position& child, int ply) {
  const Accumulator& from = stack_[std::size_t(ply - 1)];
  Accumulator& to = stack_[std::size_t(ply)];
  const Delta delta = compute_delta(parent, m);

  for (const Color perspective : {WHITE, BLACK}) {
    const Square king = child.king_square(perspective);
    if (needs_refresh(perspective, parent.king_square(perspective), king)) {
      refresh_perspective(network, child, perspective, to.values[perspective]);
      continue;
    }
    std::array<const std::int16_t*, 2> adds{};
    std::array<const std::int16_t*, 2> subs{};
    for (int i = 0; i < delta.added_count; ++i) {
      const PieceChange& change = delta.added[std::size_t(i)];
      adds[std::size_t(i)] = feature_row(
          network, feature_index(perspective, king, change.piece, change.square));
    }
    for (int i = 0; i < delta.removed_count; ++i) {
      const PieceChange& change = delta.removed[std::size_t(i)];
      subs[std::size_t(i)] = feature_row(
          network, feature_index(perspective, king, change.piece, change.square));
    }
    apply_rows(from.values[perspective].data(), to.values[perspective].data(), adds,
               delta.added_count, subs, delta.removed_count);
  }
}

Value AccumulatorStack::output(const Network& network, const Accumulator& accumulator,
                               const Position& pos) {
  return network_output(network, accumulator, pos);
}

Value evaluate_from_scratch(const Network& network, const Position& pos) {
  Accumulator accumulator;
  sum_all_pieces(network, pos, WHITE, accumulator.values[WHITE]);
  sum_all_pieces(network, pos, BLACK, accumulator.values[BLACK]);
  return network_output(network, accumulator, pos);
}

}  // namespace ayano::nnue

#include "nnue/accumulator.h"

#include <algorithm>
#include <bit>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// La cabeza profunda debe dar el mismo numero con y sin AVX2: nada de fundir multiplicacion y
// suma en FMA. GCC y Clang lo evitan con -ffp-contract=off (CMakeLists.txt y Makefile).
#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract(off)
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

// Parejas de una perspectiva en [0, kPairMax]. (a * b) >> 9 es mulhi(a << 7, b) sin perder
// nada, porque a << 7 <= 32640 cabe en 16 bits sin signo.
void pairwise(const std::int16_t* values, std::uint8_t* out) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i limit = _mm256_set1_epi16(kQuantActivation);
  const auto clipped = [&](const std::int16_t* p) {
    return _mm256_min_epi16(_mm256_max_epi16(load_lanes(p), zero), limit);
  };
  for (int i = 0; i < kPairs; i += 2 * kLanes) {
    const __m256i low = _mm256_mulhi_epu16(_mm256_slli_epi16(clipped(values + i), 16 - kPairShift),
                                           clipped(values + kPairs + i));
    const __m256i high =
        _mm256_mulhi_epu16(_mm256_slli_epi16(clipped(values + i + kLanes), 16 - kPairShift),
                           clipped(values + kPairs + i + kLanes));
    // packus intercala las mitades de 128 bits; el permute las devuelve a su orden.
    const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(low, high), 0b11011000);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), packed);
  }
}

// Primera capa de la cabeza. Tras el producto por parejas muchas entradas valen cero, asi que
// solo se recorren los grupos de 4 entradas con alguna distinta de cero. Cada grupo se repite
// en todo el registro y maddubs lo multiplica por sus pesos de las 16 neuronas: dos productos
// u8 x i8 en 16 bits, que no saturan porque las entradas no pasan de 127, y madd los lleva a
// 32 bits. Las sumas enteras no dependen del orden: el resultado es el de la version escalar.
void deep_first_layer(const std::uint8_t* inputs, const std::int8_t* weights,
                      std::array<std::int32_t, kDeepL1>& sums) {
  static_assert(kDeepL1 == 16, "dos registros de 8 neuronas");
  constexpr int kGroups = kDeepInputs / 4;
  std::array<std::uint16_t, kGroups> active;
  int count = 0;
  const __m256i zero = _mm256_setzero_si256();
  for (int i = 0; i < kDeepInputs; i += 32) {
    // Con bytes de 0 a 127, un grupo de 4 leido como int32 es positivo si no es todo cero.
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs + i));
    unsigned mask = unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(x, zero))));
    while (mask) {
      active[std::size_t(count++)] = std::uint16_t(i / 4 + std::countr_zero(mask));
      mask &= mask - 1;
    }
  }

  const __m256i ones = _mm256_set1_epi16(1);
  __m256i low = _mm256_setzero_si256();
  __m256i high = _mm256_setzero_si256();
  for (int k = 0; k < count; ++k) {
    const int group = active[std::size_t(k)];
    std::int32_t packed;
    std::memcpy(&packed, inputs + 4 * group, sizeof(packed));
    const __m256i x = _mm256_set1_epi32(packed);
    const std::int8_t* w = weights + std::size_t(group) * kDeepL1 * 4;
    const __m256i w_low = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
    const __m256i w_high = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32));
    low = _mm256_add_epi32(low, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w_low), ones));
    high = _mm256_add_epi32(high, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w_high), ones));
  }
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(sums.data()), low);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(sums.data() + 8), high);
}

// SCReLU en coma flotante, como en el entrenador: recorta a [0, 1] y eleva al cuadrado.
inline __m256 screlu_ps(__m256 x) {
  const __m256 clipped = _mm256_min_ps(_mm256_max_ps(x, _mm256_setzero_ps()), _mm256_set1_ps(1.0f));
  return _mm256_mul_ps(clipped, clipped);
}

// Capas en coma flotante de la cabeza, con el orden de sumas descrito en la version escalar.
// Multiplicar y sumar van por separado, como alli: sin FMA el resultado es identico.
float deep_float_layers(const Network& network, std::size_t bucket,
                        const std::array<std::int32_t, kDeepL1>& sums) {
  const __m256 scale = _mm256_set1_ps(float(kPairMax * kDeepWeightScale));
  const float* l1_bias = network.l1_bias.data() + bucket * kDeepL1;
  alignas(32) std::array<float, kDeepL1> first;
  for (std::size_t h = 0; h < kDeepL1; h += 8) {
    const __m256 sum =
        _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(sums.data() + h)));
    const __m256 value = _mm256_add_ps(_mm256_div_ps(sum, scale), _mm256_loadu_ps(l1_bias + h));
    _mm256_store_ps(first.data() + h, screlu_ps(value));
  }

  const float* l2_bias = network.l2_bias.data() + bucket * kDeepL2;
  const float* l2 = network.l2_weights.data() + bucket * kDeepL1 * kDeepL2;
  // Como argumento de plantilla, GCC descarta los atributos de __m256 y avisa
  // (-Wignored-attributes). Aqui no importa: el arreglo solo guarda registros y se usa como tal.
  // Envolver __m256 en un struct evita el aviso, pero MSVC lo compila un 2 % mas lento.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
  std::array<__m256, kDeepL2 / 8> second;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  for (std::size_t j = 0; j < second.size(); ++j) second[j] = _mm256_loadu_ps(l2_bias + 8 * j);
  for (std::size_t o = 0; o < kDeepL1; ++o) {
    const __m256 x = _mm256_set1_ps(first[o]);
    for (std::size_t j = 0; j < second.size(); ++j) {
      second[j] = _mm256_add_ps(second[j], _mm256_mul_ps(_mm256_loadu_ps(l2 + o * kDeepL2 + 8 * j), x));
    }
  }

  const float* l3 = network.l3_weights.data() + bucket * kDeepL2;
  __m256 lanes = _mm256_setzero_ps();
  for (std::size_t j = 0; j < second.size(); ++j) {
    lanes = _mm256_add_ps(lanes, _mm256_mul_ps(_mm256_loadu_ps(l3 + 8 * j), screlu_ps(second[j])));
  }
  const __m128 halves = _mm_add_ps(_mm256_castps256_ps128(lanes), _mm256_extractf128_ps(lanes, 1));
  const __m128 pairs = _mm_add_ps(halves, _mm_movehl_ps(halves, halves));
  const __m128 total = _mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 0x55));
  return network.l3_bias[bucket] + _mm_cvtss_f32(total);
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

void pairwise(const std::int16_t* values, std::uint8_t* out) {
  for (int i = 0; i < kPairs; ++i) {
    const int a = std::clamp(int(values[i]), 0, kQuantActivation);
    const int b = std::clamp(int(values[i + kPairs]), 0, kQuantActivation);
    out[i] = std::uint8_t((a * b) >> kPairShift);
  }
}

void deep_first_layer(const std::uint8_t* inputs, const std::int8_t* weights,
                      std::array<std::int32_t, kDeepL1>& sums) {
  sums.fill(0);
  for (int group = 0; group < kDeepInputs / 4; ++group) {
    const std::int8_t* w = weights + std::size_t(group) * kDeepL1 * 4;
    for (int o = 0; o < kDeepL1; ++o) {
      for (int j = 0; j < 4; ++j) sums[std::size_t(o)] += int(inputs[4 * group + j]) * int(w[4 * o + j]);
    }
  }
}

// Capas en coma flotante de la cabeza, en un orden de sumas fijo que la version AVX2 repite:
// asi todas las compilaciones dan el mismo numero y la firma de bench no depende de la CPU.
// - Primera: suma entera a coma flotante, sesgo y SCReLU (recorte a [0, 1] y cuadrado).
// - Segunda: cada neurona parte de su sesgo y suma sus 16 entradas en orden.
// - Tercera: ocho sumas parciales, la j con las entradas j, j + 8, j + 16 y j + 24 en ese orden;
//   luego la j con la j + 4, esas cuatro de dos en dos, y al final el sesgo.
float deep_float_layers(const Network& network, std::size_t bucket,
                        const std::array<std::int32_t, kDeepL1>& sums) {
  const auto screlu = [](float x) {
    const float clipped = std::min(std::max(x, 0.0f), 1.0f);
    return clipped * clipped;
  };
  const float scale = float(kPairMax * kDeepWeightScale);
  std::array<float, kDeepL1> first;
  for (std::size_t o = 0; o < first.size(); ++o) {
    first[o] = screlu(float(sums[o]) / scale + network.l1_bias[bucket * kDeepL1 + o]);
  }

  std::array<float, kDeepL2> second;
  std::copy_n(network.l2_bias.data() + bucket * kDeepL2, kDeepL2, second.begin());
  const float* l2 = network.l2_weights.data() + bucket * kDeepL1 * kDeepL2;
  for (std::size_t o = 0; o < first.size(); ++o) {
    for (std::size_t k = 0; k < second.size(); ++k) second[k] = second[k] + l2[o * kDeepL2 + k] * first[o];
  }

  const float* l3 = network.l3_weights.data() + bucket * kDeepL2;
  std::array<float, 8> lanes{};
  for (std::size_t k = 0; k < second.size(); ++k) lanes[k % 8] = lanes[k % 8] + l3[k] * screlu(second[k]);
  const float t0 = lanes[0] + lanes[4];
  const float t1 = lanes[1] + lanes[5];
  const float t2 = lanes[2] + lanes[6];
  const float t3 = lanes[3] + lanes[7];
  return network.l3_bias[bucket] + ((t0 + t2) + (t1 + t3));
}

#endif

Value deep_output(const Network& network, const Accumulator& accumulator, const Position& pos) {
  const Color us = pos.side_to_move();
  const std::size_t bucket = std::size_t(output_bucket(popcount(pos.pieces())));
  alignas(64) std::array<std::uint8_t, kDeepInputs> inputs;
  pairwise(accumulator.values[us].data(), inputs.data());
  pairwise(accumulator.values[~us].data(), inputs.data() + kPairs);

  std::array<std::int32_t, kDeepL1> sums;
  deep_first_layer(inputs.data(), network.l1_weights.data() + bucket * kDeepL1 * kDeepInputs, sums);
  const float output = deep_float_layers(network, bucket, sums);
  // Una red valida no se acerca a este limite; uno manipulado desbordaria despues el entero
  // de la evaluacion al aplicar la regla de las 50 jugadas.
  constexpr float kLimit = 30000.0f;
  return Value(static_cast<int>(std::clamp(output * float(network.eval_scale), -kLimit, kLimit)));
}

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
  if (network.head == Head::Deep) return deep_output(network, accumulator, pos);
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

#include "search/tt.h"

#include <algorithm>
#include <new>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <xmmintrin.h>
#endif

namespace enroque {

namespace {

// Disposicion de los 64 bits de datos.
constexpr int kValueShift = 16;
constexpr int kEvalShift = 32;
constexpr int kDepthShift = 48;
constexpr int kBoundShift = 56;
constexpr int kPvShift = 58;
constexpr int kGenerationShift = 59;

std::uint64_t pack(Move move, Value value, Value eval, int stored_depth, Bound bound, bool pv,
                   unsigned generation) {
  return std::uint64_t(move.raw()) |
         (std::uint64_t(std::uint16_t(std::int16_t(value))) << kValueShift) |
         (std::uint64_t(std::uint16_t(std::int16_t(eval))) << kEvalShift) |
         (std::uint64_t(std::uint8_t(stored_depth)) << kDepthShift) |
         (std::uint64_t(bound) << kBoundShift) | (std::uint64_t(pv) << kPvShift) |
         (std::uint64_t(generation) << kGenerationShift);
}

Move unpack_move(std::uint64_t data) { return Move::from_raw(std::uint16_t(data)); }
Value unpack_value(std::uint64_t data) { return std::int16_t(std::uint16_t(data >> kValueShift)); }
Value unpack_eval(std::uint64_t data) { return std::int16_t(std::uint16_t(data >> kEvalShift)); }
int unpack_depth(std::uint64_t data) { return int(std::uint8_t(data >> kDepthShift)); }
Bound unpack_bound(std::uint64_t data) { return Bound((data >> kBoundShift) & 3); }
bool unpack_pv(std::uint64_t data) { return ((data >> kPvShift) & 1) != 0; }
unsigned unpack_generation(std::uint64_t data) { return unsigned(data >> kGenerationShift); }

std::uint64_t multiply_high(std::uint64_t a, std::uint64_t b) {
#if defined(_MSC_VER)
  return __umulh(a, b);
#else
  return std::uint64_t((static_cast<unsigned __int128>(a) * b) >> 64);
#endif
}

}  // namespace

TranspositionTable::TranspositionTable() { resize(16); }

bool TranspositionTable::resize(std::size_t megabytes) {
  const std::size_t previous = bucket_count_;
  const std::size_t count = std::max<std::size_t>(1, megabytes * 1024 * 1024 / sizeof(Bucket));
  buckets_.reset();  // libera antes de reservar para no tener dos tablas a la vez
  bucket_count_ = 0;
  generation_ = 0;
  bool enough_memory = true;
  try {
    buckets_ = std::make_unique<Bucket[]>(count);
    bucket_count_ = count;
  } catch (const std::bad_alloc&) {
    enough_memory = false;
    bucket_count_ = std::max<std::size_t>(previous, 1);
    buckets_ = std::make_unique<Bucket[]>(bucket_count_);
  }
  return enough_memory;
}

void TranspositionTable::clear() {
  for (std::size_t i = 0; i < bucket_count_; ++i) {
    for (Entry& entry : buckets_[i].entries) {
      entry.key_xor_data.store(0, std::memory_order_relaxed);
      entry.data.store(0, std::memory_order_relaxed);
    }
  }
  generation_ = 0;
}

TranspositionTable::Bucket& TranspositionTable::bucket_for(Key key) const {
  return buckets_[multiply_high(key, bucket_count_)];
}

void TranspositionTable::prefetch(Key key) const {
  _mm_prefetch(reinterpret_cast<const char*>(&bucket_for(key)), _MM_HINT_T0);
}

bool TranspositionTable::probe(Key key, TTData& out) const {
  const Bucket& bucket = bucket_for(key);
  for (const Entry& entry : bucket.entries) {
    const std::uint64_t data = entry.data.load(std::memory_order_relaxed);
    const std::uint64_t mixed = entry.key_xor_data.load(std::memory_order_relaxed);
    if ((mixed ^ data) != key || data == 0) continue;
    out.move = unpack_move(data);
    out.value = unpack_value(data);
    out.eval = unpack_eval(data);
    out.depth = unpack_depth(data) - kDepthOffset;
    out.bound = unpack_bound(data);
    out.pv = unpack_pv(data);
    return true;
  }
  return false;
}

void TranspositionTable::store(Key key, Value value, Value eval, int depth, Bound bound, Move move,
                               bool pv) {
  Bucket& bucket = bucket_for(key);
  Entry* target = &bucket.entries[0];
  int worst_priority = 1 << 30;
  std::uint64_t target_data = 0;
  bool same_position = false;

  for (Entry& entry : bucket.entries) {
    const std::uint64_t data = entry.data.load(std::memory_order_relaxed);
    const std::uint64_t mixed = entry.key_xor_data.load(std::memory_order_relaxed);
    if (data != 0 && (mixed ^ data) == key) {
      target = &entry;
      target_data = data;
      same_position = true;
      break;
    }
    // Se sustituye la entrada menos profunda y mas antigua.
    const unsigned age = (generation_ - unpack_generation(data)) & kGenerationMask;
    const int priority = data == 0 ? -(1 << 20) : unpack_depth(data) - 8 * int(age);
    if (priority < worst_priority) {
      worst_priority = priority;
      target = &entry;
      target_data = data;
    }
  }

  const int stored_depth = std::clamp(depth + kDepthOffset, 0, 255);
  if (same_position) {
    if (move == Move::none()) move = unpack_move(target_data);
    // No se pisa una busqueda mas profunda con una superficial del mismo nodo.
    const bool shallower = stored_depth + 2 * int(pv) + 4 <= unpack_depth(target_data);
    if (bound != Bound::Exact && shallower) return;
  }

  const std::uint64_t data = pack(move, value, eval, stored_depth, bound, pv, generation_);
  target->data.store(data, std::memory_order_relaxed);
  target->key_xor_data.store(key ^ data, std::memory_order_relaxed);
}

int TranspositionTable::hashfull() const {
  const std::size_t sample = std::min<std::size_t>(250, bucket_count_);
  int used = 0;
  for (std::size_t i = 0; i < sample; ++i) {
    for (const Entry& entry : buckets_[i].entries) {
      const std::uint64_t data = entry.data.load(std::memory_order_relaxed);
      if (data != 0 && unpack_generation(data) == generation_) ++used;
    }
  }
  return int(used * 1000 / (sample * kBucketSize));
}

}  // namespace enroque

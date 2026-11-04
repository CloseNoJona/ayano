// Tabla de transposicion compartida por todos los hilos, sin bloqueos.
//
// Cada entrada son dos enteros atomicos de 64 bits: los datos y (clave XOR datos).
// Si dos hilos escriben a la vez y una lectura mezcla mitades, la comprobacion
// del XOR falla y la entrada simplemente no se usa. Las operaciones relajadas
// compilan a instrucciones normales, asi que la seguridad no cuesta velocidad.
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "core/types.h"

namespace enroque {

enum class Bound : std::uint8_t { None = 0, Upper = 1, Lower = 2, Exact = 3 };

constexpr bool has_bound(Bound value, Bound wanted) {
  return (std::uint8_t(value) & std::uint8_t(wanted)) != 0;
}

struct TTData {
  Move move;
  Value value;
  Value eval;
  int depth;
  Bound bound;
  bool pv;
};

class TranspositionTable {
 public:
  TranspositionTable();

  // Devuelve false si no hay memoria; en ese caso conserva el tamano anterior.
  bool resize(std::size_t megabytes);
  void clear();
  void new_search() { generation_ = (generation_ + 1) & kGenerationMask; }

  bool probe(Key key, TTData& data) const;
  void store(Key key, Value value, Value eval, int depth, Bound bound, Move move, bool pv);
  void prefetch(Key key) const;
  int hashfull() const;  // ocupacion en tantos por mil

 private:
  static constexpr int kBucketSize = 4;
  static constexpr unsigned kGenerationMask = 31;
  static constexpr int kDepthOffset = 8;

  struct Entry {
    std::atomic<std::uint64_t> key_xor_data;
    std::atomic<std::uint64_t> data;
  };

  struct alignas(64) Bucket {
    std::array<Entry, kBucketSize> entries;
  };

  Bucket& bucket_for(Key key) const;

  std::unique_ptr<Bucket[]> buckets_;
  std::size_t bucket_count_ = 0;
  unsigned generation_ = 0;
};

// Las puntuaciones de mate se guardan relativas a la posicion, no a la raiz.
constexpr Value value_to_tt(Value v, int ply) {
  if (v == VALUE_NONE) return v;
  if (v >= VALUE_MATE_IN_MAX_PLY) return v + ply;
  if (v <= VALUE_MATED_IN_MAX_PLY) return v - ply;
  return v;
}

constexpr Value value_from_tt(Value v, int ply) {
  if (v == VALUE_NONE) return v;
  if (v >= VALUE_MATE_IN_MAX_PLY) return v - ply;
  if (v <= VALUE_MATED_IN_MAX_PLY) return v + ply;
  return v;
}

}  // namespace enroque

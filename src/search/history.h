// Tablas de historial: recuerdan que jugadas produjeron cortes en busquedas
// anteriores para ordenar mejor y podar con mas criterio.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>

#include "core/types.h"

namespace enroque {

using HistoryValue = std::int16_t;

// Actualizacion con "gravedad": cuanto mas cerca esta del limite, menos sube.
// Mantiene los valores acotados sin tener que reescalar las tablas.
template <int Limit>
inline void update_history(HistoryValue& entry, int bonus) {
  const int clamped = std::clamp(bonus, -Limit, Limit);
  entry = HistoryValue(entry + clamped - entry * std::abs(clamped) / Limit);
}

inline constexpr int kMainHistoryLimit = 7183;
inline constexpr int kCaptureHistoryLimit = 10692;
inline constexpr int kContinuationLimit = 29952;
inline constexpr int kPawnHistoryLimit = 8192;
inline constexpr int kCorrectionLimit = 1024;

inline constexpr std::size_t kPawnHistorySize = 1024;
inline constexpr std::size_t kCorrectionSize = 16384;

template <typename T, std::size_t N>
using Table = std::array<T, N>;

using PieceToHistory = Table<Table<HistoryValue, SQUARE_NB>, PIECE_NB>;
using ContinuationTable = Table<Table<PieceToHistory, SQUARE_NB>, PIECE_NB>;
using CorrectionTable = Table<Table<HistoryValue, kCorrectionSize>, COLOR_NB>;

struct Histories {
  Table<Table<HistoryValue, 4096>, COLOR_NB> main;
  // El ultimo indice incluye NO_PIECE_TYPE para las promociones sin captura.
  Table<Table<Table<HistoryValue, PIECE_TYPE_NB + 1>, SQUARE_NB>, PIECE_NB> capture;
  Table<PieceToHistory, kPawnHistorySize> pawn;
  // Indexada por [estaba en jaque][fue captura].
  Table<Table<ContinuationTable, 2>, 2> continuation;
  Table<Table<Move, SQUARE_NB>, PIECE_NB> counter_moves;

  CorrectionTable pawn_correction;
  CorrectionTable minor_correction;
  Table<CorrectionTable, COLOR_NB> non_pawn_correction;
  PieceToHistory sentinel;  // destino de las referencias antes del primer nodo
};

}  // namespace enroque

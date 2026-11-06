#include "search/timeman.h"

#include <algorithm>

namespace ayano {

namespace {

constexpr std::int64_t kMinimumThinkMs = 1;
constexpr int kDefaultMovesToGo = 28;
constexpr int kMaxMovesToGo = 50;
constexpr int kReservedMoves = 50;
constexpr std::int64_t kMinimumReserveCapMs = 500;

// Cada jugada cuesta reloj aunque no se piense, por los retrasos de la interfaz y del sistema.
// Sin un incremento que lo cubra, repartir una fraccion del reloj lo lleva hasta casi cero y
// las partidas largas se pierden por tiempo: hay que guardar para las jugadas futuras. Un
// margen grande suele ser una precaucion, asi que no puede llevarse todo el reloj.
std::int64_t future_moves_reserve(std::int64_t clock_ms, std::int64_t increment_ms,
                                  int move_overhead_ms) {
  const std::int64_t unpaid_overhead = std::max<std::int64_t>(move_overhead_ms - increment_ms, 0);
  return std::min(unpaid_overhead * kReservedMoves, std::max(kMinimumReserveCapMs, clock_ms / 2));
}

}  // namespace

void TimeManager::start(const SearchLimits& limits, Color us, int game_ply, int move_overhead_ms) {
  start_time_ = std::chrono::steady_clock::now();
  optimum_ms_ = 0;
  maximum_ms_ = 0;

  if (limits.move_time_ms > 0) {
    optimum_ms_ = maximum_ms_ =
        std::max(kMinimumThinkMs, limits.move_time_ms - std::int64_t(move_overhead_ms));
    return;
  }
  if (!limits.uses_clock() || limits.infinite) return;

  const std::int64_t increment = limits.increment_ms[us];
  const std::int64_t reserve =
      limits.moves_to_go > 0
          ? 0
          : future_moves_reserve(limits.time_ms[us], increment, move_overhead_ms);
  const std::int64_t remaining =
      std::max(kMinimumThinkMs, limits.time_ms[us] - std::int64_t(move_overhead_ms) - reserve);

  // Sin control fijo se estiman las jugadas restantes: al principio de la partida
  // quedan mas, asi que se reserva algo mas de tiempo para el medio juego.
  int moves_left = kDefaultMovesToGo;
  if (limits.moves_to_go > 0) {
    moves_left = std::min(limits.moves_to_go, kMaxMovesToGo);
  } else {
    moves_left = std::max(kDefaultMovesToGo - game_ply / 8, 14);
  }

  const std::int64_t base = remaining / moves_left + increment * 3 / 4;
  // El sistema puede retrasar un hilo decenas de milisegundos y, con el reloj bajo, esa
  // parada costaria la partida: el limite duro deja un colchon proporcional y otro fijo.
  const std::int64_t hard_limit =
      std::max(kMinimumThinkMs, remaining * 7 / 10 - std::int64_t(move_overhead_ms));
  optimum_ms_ = std::clamp(base * 7 / 10, kMinimumThinkMs, remaining);
  maximum_ms_ = std::clamp(base * 4, kMinimumThinkMs, hard_limit);
  optimum_ms_ = std::min(optimum_ms_, maximum_ms_);
}

std::int64_t TimeManager::elapsed_ms() const {
  const auto elapsed = std::chrono::steady_clock::now() - start_time_;
  return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

}  // namespace ayano

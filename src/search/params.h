// Parametros numericos de la busqueda. Estan reunidos aqui para poder ajustarlos con
// partidas de prueba sin tocar la logica. En la compilacion normal son constantes; con
// ENROQUE_TUNE se convierten en variables modificables mediante opciones UCI.
#pragma once

#include <array>
#include <string_view>

// X(nombre, valor, minimo, maximo)
#define ENROQUE_SEARCH_PARAMS(X)         \
  /* Ventana de aspiracion */            \
  X(kAspirationDelta, 14, 5, 40)         \
  /* Poda antes de generar jugadas */    \
  X(kRazorMaxDepth, 4, 1, 8)             \
  X(kRazorMargin, 320, 100, 700)         \
  X(kRfpMaxDepth, 8, 4, 14)              \
  X(kRfpMargin, 78, 30, 160)             \
  X(kRfpImprovingBonus, 62, 0, 140)      \
  X(kNullMoveMinDepth, 3, 1, 6)          \
  X(kNullMoveBase, 4, 2, 6)              \
  X(kNullMoveDepthDivisor, 3, 2, 6)      \
  X(kNullMoveEvalDivisor, 200, 80, 400)  \
  X(kNullMoveMaxEvalReduction, 3, 1, 6)  \
  X(kNullMoveVerifyDepth, 14, 8, 24)     \
  X(kProbCutMinDepth, 5, 3, 9)           \
  X(kProbCutMargin, 200, 80, 400)        \
  X(kProbCutImprovingBonus, 50, 0, 120)  \
  X(kIirMinDepth, 4, 2, 8)               \
  /* Poda dentro del bucle de jugadas */ \
  X(kLmpBase, 3, 1, 8)                   \
  X(kFutilityMaxDepth, 9, 4, 14)         \
  X(kFutilityBase, 95, 20, 220)          \
  X(kFutilityPerDepth, 105, 40, 220)     \
  X(kCaptureFutilityMaxDepth, 7, 3, 12)  \
  X(kCaptureFutilityBase, 190, 60, 400)  \
  X(kCaptureFutilityPerDepth, 190, 60, 400) \
  X(kCaptureHistoryDivisor, 7, 3, 16)    \
  X(kSeeCaptureMargin, 95, 30, 220)      \
  X(kSeeQuietMargin, 28, 8, 80)          \
  X(kHistoryPruneMaxDepth, 5, 2, 9)      \
  X(kHistoryPruneMargin, 3200, 800, 8000) \
  X(kHistoryToLmrDepth, 7000, 2000, 16000) \
  /* Extensiones */                      \
  X(kSingularMinDepth, 6, 4, 10)         \
  X(kSingularTtDepthMargin, 3, 1, 5)     \
  X(kDoubleExtensionMargin, 18, 4, 60)   \
  X(kTripleExtensionMargin, 90, 30, 240) \
  /* Reducciones tardias (centesimas) */ \
  X(kLmrBase, 77, 20, 140)               \
  X(kLmrDivisor, 236, 150, 340)          \
  X(kLmrStatScoreDivisor, 11000, 4000, 24000) \
  X(kLmrDeeperMargin, 40, 10, 100)       \
  X(kLmrShallowerMargin, 10, 2, 40)      \
  /* Busqueda de quietud */              \
  X(kQsFutilityMargin, 180, 60, 400)     \
  X(kQsSeeMargin, -60, -200, 0)          \
  /* Historial */                        \
  X(kHistoryBonusScale, 170, 60, 400)    \
  X(kHistoryBonusOffset, 80, 0, 300)     \
  X(kHistoryBonusMax, 1700, 600, 4000)   \
  X(kHistoryMalusScale, 200, 60, 500)    \
  X(kHistoryMalusOffset, 90, 0, 300)     \
  X(kHistoryMalusMax, 1500, 600, 4000)   \
  /* Correccion de la evaluacion */      \
  X(kCorrectionScale, 256, 64, 512)      \
  X(kCorrectionPawnWeight, 66, 0, 160)   \
  X(kCorrectionMinorWeight, 45, 0, 160)  \
  X(kCorrectionNonPawnWeight, 55, 0, 160) \
  X(kCorrectionDivisor, 512, 128, 1024)

namespace enroque::params {

#ifdef ENROQUE_TUNE
#define ENROQUE_DECLARE_PARAM(name, value, low, high) inline int name = value;
#else
#define ENROQUE_DECLARE_PARAM(name, value, low, high) inline constexpr int name = value;
#endif

ENROQUE_SEARCH_PARAMS(ENROQUE_DECLARE_PARAM)

#undef ENROQUE_DECLARE_PARAM

// Margen de la ventana de aspiracion por debajo del cual no se usa.
inline constexpr int kAspirationMinDepth = 4;

#ifdef ENROQUE_TUNE
struct TunableParam {
  std::string_view name;
  int* value;
  int low;
  int high;
};

#define ENROQUE_LIST_PARAM(name, value, low, high) TunableParam{#name, &name, low, high},
inline const auto kTunableParams = std::array{ENROQUE_SEARCH_PARAMS(ENROQUE_LIST_PARAM)};
#undef ENROQUE_LIST_PARAM
#endif

}  // namespace enroque::params

// Parametros numericos de la busqueda. Estan reunidos aqui para poder ajustarlos con
// partidas de prueba sin tocar la logica. En la compilacion normal son constantes; con
// AYANO_TUNE se convierten en variables modificables mediante opciones UCI.
// scripts/spsa.py los ajusta jugando partidas, sin salirse del minimo y el maximo de cada uno.
#pragma once

#include <array>
#include <string_view>

// X(nombre, valor, minimo, maximo)
#define AYANO_SEARCH_PARAMS(X)         \
  /* Ventana de aspiracion */            \
  X(kAspirationDelta, 13, 5, 40)         \
  /* Poda antes de generar jugadas */    \
  X(kRazorMaxDepth, 4, 1, 8)             \
  X(kRazorMargin, 328, 100, 700)         \
  X(kRfpMaxDepth, 8, 4, 14)              \
  X(kRfpMargin, 59, 30, 160)             \
  X(kRfpImprovingBonus, 54, 0, 140)      \
  X(kNullMoveMinDepth, 3, 1, 6)          \
  X(kNullMoveBase, 4, 2, 6)              \
  X(kNullMoveDepthDivisor, 3, 2, 6)      \
  X(kNullMoveEvalDivisor, 194, 80, 400)  \
  X(kNullMoveMaxEvalReduction, 3, 1, 6)  \
  X(kNullMoveVerifyDepth, 14, 8, 24)     \
  X(kProbCutMinDepth, 5, 3, 9)           \
  X(kProbCutMargin, 191, 80, 400)        \
  X(kProbCutImprovingBonus, 51, 0, 120)  \
  X(kIirMinDepth, 4, 2, 8)               \
  /* Profundidad corregida a posteriori: reduccion minima del padre para alargar o acortar, */ \
  /* y margen de evaluacion para acortar. Con 20 en una reduccion, ese ajuste no se aplica. */ \
  X(kHindsightExtendReduction, 3, 1, 20) \
  X(kHindsightReduceReduction, 2, 1, 20) \
  X(kHindsightMargin, 94, 20, 300)      \
  /* Corte claro que cuenta como uno mas profundo, y peso de la correccion en las reducciones */ \
  X(kHistoryBoostMargin, 77, 30, 300)    \
  X(kComplexityLmrDivisor, 133, 40, 400) \
  /* Poda dentro del bucle de jugadas */ \
  X(kLmpBase, 3, 1, 8)                   \
  X(kFutilityMaxDepth, 9, 4, 14)         \
  X(kFutilityBase, 60, 20, 220)          \
  X(kFutilityPerDepth, 88, 40, 220)      \
  X(kCaptureFutilityMaxDepth, 7, 3, 12)  \
  X(kCaptureFutilityBase, 232, 60, 400)  \
  X(kCaptureFutilityPerDepth, 271, 60, 400) \
  X(kCaptureHistoryDivisor, 7, 3, 16)    \
  X(kSeeCaptureMargin, 89, 30, 220)      \
  X(kSeeQuietMargin, 25, 8, 80)          \
  X(kHistoryPruneMaxDepth, 5, 2, 9)      \
  X(kHistoryPruneMargin, 2861, 800, 8000) \
  X(kHistoryToLmrDepth, 6099, 2000, 16000) \
  /* Extensiones */                      \
  X(kSingularMinDepth, 6, 4, 10)         \
  X(kSingularTtDepthMargin, 3, 1, 5)     \
  X(kDoubleExtensionMargin, 23, 4, 60)   \
  X(kTripleExtensionMargin, 91, 30, 240) \
  /* Reducciones tardias (centesimas) */ \
  X(kLmrBase, 72, 20, 140)               \
  X(kLmrDivisor, 236, 150, 340)          \
  X(kLmrStatScoreDivisor, 13051, 4000, 24000) \
  X(kLmrDeeperMargin, 38, 10, 100)       \
  X(kLmrShallowerMargin, 12, 2, 40)      \
  /* Busqueda de quietud */              \
  X(kQsFutilityMargin, 164, 60, 400)     \
  X(kQsSeeMargin, -75, -200, 0)          \
  /* Historial */                        \
  X(kHistoryBonusScale, 133, 60, 400)    \
  X(kHistoryBonusOffset, 97, 0, 300)    \
  X(kHistoryBonusMax, 1870, 600, 4000)   \
  X(kHistoryMalusScale, 263, 60, 500)    \
  X(kHistoryMalusOffset, 69, 0, 300)     \
  X(kHistoryMalusMax, 748, 600, 4000)    \
  /* Correccion de la evaluacion */      \
  X(kCorrectionScale, 233, 64, 512)      \
  X(kCorrectionPawnWeight, 66, 0, 160)   \
  X(kCorrectionMinorWeight, 37, 0, 160)  \
  X(kCorrectionNonPawnWeight, 64, 0, 160) \
  X(kCorrectionContinuationWeight, 50, 0, 160) \
  X(kCorrectionDivisor, 504, 128, 1024)

namespace ayano::params {

#ifdef AYANO_TUNE
#define AYANO_DECLARE_PARAM(name, value, low, high) inline int name = value;
#else
#define AYANO_DECLARE_PARAM(name, value, low, high) inline constexpr int name = value;
#endif

AYANO_SEARCH_PARAMS(AYANO_DECLARE_PARAM)

#undef AYANO_DECLARE_PARAM

// Margen de la ventana de aspiracion por debajo del cual no se usa.
inline constexpr int kAspirationMinDepth = 4;

#ifdef AYANO_TUNE
struct TunableParam {
  std::string_view name;
  int* value;
  int low;
  int high;
};

#define AYANO_LIST_PARAM(name, value, low, high) TunableParam{#name, &name, low, high},
inline const auto kTunableParams = std::array{AYANO_SEARCH_PARAMS(AYANO_LIST_PARAM)};
#undef AYANO_LIST_PARAM
#endif

}  // namespace ayano::params

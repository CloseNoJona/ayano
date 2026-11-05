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
  X(kAspirationDelta, 14, 5, 40)         \
  /* Poda antes de generar jugadas */    \
  X(kRazorMaxDepth, 4, 1, 8)             \
  X(kRazorMargin, 348, 100, 700)         \
  X(kRfpMaxDepth, 8, 4, 14)              \
  X(kRfpMargin, 74, 30, 160)             \
  X(kRfpImprovingBonus, 64, 0, 140)      \
  X(kNullMoveMinDepth, 3, 1, 6)          \
  X(kNullMoveBase, 4, 2, 6)              \
  X(kNullMoveDepthDivisor, 3, 2, 6)      \
  X(kNullMoveEvalDivisor, 213, 80, 400)  \
  X(kNullMoveMaxEvalReduction, 3, 1, 6)  \
  X(kNullMoveVerifyDepth, 14, 8, 24)     \
  X(kProbCutMinDepth, 5, 3, 9)           \
  X(kProbCutMargin, 191, 80, 400)        \
  X(kProbCutImprovingBonus, 48, 0, 120)  \
  X(kIirMinDepth, 4, 2, 8)               \
  /* Poda dentro del bucle de jugadas */ \
  X(kLmpBase, 3, 1, 8)                   \
  X(kFutilityMaxDepth, 9, 4, 14)         \
  X(kFutilityBase, 73, 20, 220)          \
  X(kFutilityPerDepth, 103, 40, 220)     \
  X(kCaptureFutilityMaxDepth, 7, 3, 12)  \
  X(kCaptureFutilityBase, 195, 60, 400)  \
  X(kCaptureFutilityPerDepth, 232, 60, 400) \
  X(kCaptureHistoryDivisor, 7, 3, 16)    \
  X(kSeeCaptureMargin, 101, 30, 220)      \
  X(kSeeQuietMargin, 25, 8, 80)          \
  X(kHistoryPruneMaxDepth, 5, 2, 9)      \
  X(kHistoryPruneMargin, 3022, 800, 8000) \
  X(kHistoryToLmrDepth, 6604, 2000, 16000) \
  /* Extensiones */                      \
  X(kSingularMinDepth, 6, 4, 10)         \
  X(kSingularTtDepthMargin, 3, 1, 5)     \
  X(kDoubleExtensionMargin, 20, 4, 60)   \
  X(kTripleExtensionMargin, 86, 30, 240) \
  /* Reducciones tardias (centesimas) */ \
  X(kLmrBase, 76, 20, 140)               \
  X(kLmrDivisor, 233, 150, 340)          \
  X(kLmrStatScoreDivisor, 12000, 4000, 24000) \
  X(kLmrDeeperMargin, 42, 10, 100)       \
  X(kLmrShallowerMargin, 10, 2, 40)      \
  /* Busqueda de quietud */              \
  X(kQsFutilityMargin, 181, 60, 400)     \
  X(kQsSeeMargin, -69, -200, 0)          \
  /* Historial */                        \
  X(kHistoryBonusScale, 150, 60, 400)    \
  X(kHistoryBonusOffset, 93, 0, 300)     \
  X(kHistoryBonusMax, 1746, 600, 4000)   \
  X(kHistoryMalusScale, 252, 60, 500)    \
  X(kHistoryMalusOffset, 80, 0, 300)     \
  X(kHistoryMalusMax, 1255, 600, 4000)   \
  /* Correccion de la evaluacion */      \
  X(kCorrectionScale, 244, 64, 512)      \
  X(kCorrectionPawnWeight, 66, 0, 160)   \
  X(kCorrectionMinorWeight, 35, 0, 160)  \
  X(kCorrectionNonPawnWeight, 62, 0, 160) \
  X(kCorrectionDivisor, 501, 128, 1024)

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

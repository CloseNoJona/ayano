// Entrega las jugadas de una posicion por etapas, de la mas prometedora a la
// menos. Si la primera provoca un corte, las demas ni siquiera se generan.
#pragma once

#include <array>

#include "board/movegen.h"
#include "search/history.h"

namespace ayano {

// Referencias de continuacion: 1, 2, 4 y 6 medias jugadas atras.
using ContinuationRefs = std::array<const PieceToHistory*, 4>;

struct Refutations {
  Move killer1;
  Move killer2;
  Move counter;
};

class MovePicker {
 public:
  // Busqueda principal.
  MovePicker(const Position& pos, Move tt_move, const Histories& histories,
             const ContinuationRefs& continuation, const Refutations& refutations);
  // Busqueda de quietud: solo capturas (o evasiones si hay jaque).
  MovePicker(const Position& pos, Move tt_move, const Histories& histories,
             const ContinuationRefs& continuation);
  // ProbCut: capturas que superen `threshold` segun SEE.
  MovePicker(const Position& pos, Move tt_move, Value threshold, const Histories& histories);

  // Devuelve Move::none() cuando no quedan jugadas.
  Move next(bool skip_quiets = false);

 private:
  enum class Stage {
    MainTT, CaptureInit, GoodCapture, Killer1, Killer2, Counter, QuietInit, Quiet, BadCapture,
    EvasionTT, EvasionInit, Evasion,
    ProbCutTT, ProbCutInit, ProbCut,
    QuiescenceTT, QuiescenceInit, Quiescence,
    Done
  };

  void advance_stage() { stage_ = Stage(int(stage_) + 1); }
  void score_captures();
  void score_quiets();
  void score_evasions();
  ExtMove pop_best();
  bool is_refutation(Move m) const;
  bool is_usable_refutation(Move m) const;
  Move next_good_capture();
  Move next_quiet();
  Move next_other_than_tt();
  Move next_probcut();

  const Position& pos_;
  const Histories& histories_;
  const ContinuationRefs* continuation_;
  Refutations refutations_;
  Move tt_move_;
  Value threshold_;
  Stage stage_;

  ExtMove* cur_;
  ExtMove* end_;
  ExtMove* end_bad_;
  MoveList list_;
};

}  // namespace ayano

// Tabla "cuckoo" de las jugadas reversibles de pieza (todas menos las de peon). Dada la
// diferencia de clave Zobrist entre dos posiciones, dice si una sola de esas jugadas lleva de
// una a otra: asi la busqueda ve que un bando puede repetir una posicion antes de jugarla.
// La tecnica es de Marcel van Kervinck y la usan casi todos los motores abiertos fuertes.
#pragma once

#include "core/types.h"

namespace ayano::cuckoo {

// Numero de jugadas de pieza distintas en un tablero vacio, contando los dos colores.
inline constexpr int kReversibleMoves = 3668;

// Requiere haber llamado antes a attacks::init() y zobrist::init(). Devuelve cuantas jugadas
// guardo: si no son kReversibleMoves, la tabla no sirve y find() no encontrara nada.
int init();

// Jugada cuya diferencia de clave, cambio de turno incluido, es `delta`; Move::none() si no
// hay ninguna. La jugada se guarda con el origen en la casilla menor: puede ir en cualquier
// sentido.
Move find(Key delta);

}  // namespace ayano::cuckoo

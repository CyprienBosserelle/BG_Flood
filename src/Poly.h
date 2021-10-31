
#ifndef POLY_H
#define POLY_H

#include "General.h"
#include "Param.h"
#include "Input.h"
#include "Util_CPU.h"
#include "Forcing.h"
#include "Arrays.h"
#include "MemManagement.h"

template <class T> bool blockinpoly(T xo, T yo, T dx, int blkwidth, Polygon Poly);
template <class T> int wn_PnPoly(T Px, T Py, Polygon Poly);



// End of global definition
#endif

#ifndef IMPLICIT_H
#define IMPLICIT_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Multilayer.h"
#include "FlowGPU.h"
#include "Halo.h"


template <class T>
void solve_implicit_barotropic(Param& XParam, Loop<T>& XLoop, Model<T>& XModel);

#endif

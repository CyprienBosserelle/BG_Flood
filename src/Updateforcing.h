
#ifndef UPDATEFORCING_H
#define UPDATEFORCING_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "InitialConditions.h"
#include "MemManagement.h"
#include "ReadForcing.h"
#include "GridManip.h"

template <class T> void updateforcing(Param XParam, Loop<T> XLoop, Forcing<float>& XForcing);


#endif

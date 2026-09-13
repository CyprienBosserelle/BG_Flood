#ifndef FLOWMLGPU_H
#define FLOWMLGPU_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "Util_CPU.h"
#include "MemManagement.h"
#include "Multilayer.h"
#include "Implicit.h"
#include "FlowGPU.h"
#include "Advection.h"

template <class T> void FlowMLGPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel);

template <class T> void solveImplicit(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel);
template <class T> void AdvecML(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel,T dt);
template <class T> void solveEtaPCG(Param XParam, Model<T> XModel,T dt);
template <class T> void test_symetry(Param XParam, Model<T> XModel,T dt);


// End of global definition
#endif

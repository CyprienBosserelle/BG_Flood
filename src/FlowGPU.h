#ifndef FLOWGPU_H
#define FLOWGPU_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "Util_CPU.h"
#include "MemManagement.h"
#include "Gradients.h"
#include "Kurganov.h"
#include "Advection.h"
#include "Friction.h"
#include "Updateforcing.h"

template <class T> void FlowGPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel);

template <class T> __global__ void reset_var(int halowidth, int* active, T resetval, T* Var);
// End of global definition
#endif

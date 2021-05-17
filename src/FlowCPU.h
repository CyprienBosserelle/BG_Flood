#ifndef FLOWCPU_H
#define FLOWCPU_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "Util_CPU.h"
#include "MemManagement.h"
#include "Halo.h"
#include "GridManip.h"
#include "Gradients.h"
#include "Kurganov.h"
#include "Advection.h"
#include "Friction.h"
#include "Updateforcing.h"

// End of global definition
template <class T> void FlowCPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel);

#endif

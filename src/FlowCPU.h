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

// End of global definition
template <class T> void FlowCPU(Param XParam, Loop<T>& XLoop, Model<T> XModel);

#endif

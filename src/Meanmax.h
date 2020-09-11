
#ifndef MEANMAX_H
#define MEANMAX_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "FlowGPU.h"


template <class T> void Calcmeanmax(Param XParam, Loop<T>& XLoop, Model<T> XModel, Model<T> XModel_g);
template <class T> void resetmeanmax(Param XParam, Loop<T>& XLoop, Model<T> XModel, Model<T> XModel_g);
template <class T> void Initmeanmax(Param XParam, Loop<T> XLoop, Model<T> XModel, Model<T> XModel_g);
// End of global definition
#endif

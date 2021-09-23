
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


template <class T> __global__ void addavg_varGPU(Param XParam, BlockP<T> XBlock, T* Varmean, T* Var);
template <class T> __global__ void divavg_varGPU(Param XParam, BlockP<T> XBlock, T ntdiv, T* Varmean);
template <class T> __global__ void max_varGPU(Param XParam, BlockP<T> XBlock, T* Varmax, T* Var);

// End of global definition
#endif

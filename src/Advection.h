#ifndef ADVECTION_H
#define ADVECTION_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "MemManagement.h"

template <class T> __global__ void updateEVGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, FluxP<T> XFlux, AdvanceP<T> XAdv);
template <class T> __host__ void updateEVCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, FluxP<T> XFlux, AdvanceP<T> XAdv);

template <class T> __global__ void AdvkernelGPU(Param XParam, BlockP<T> XBlock, T dt, T* zb, EvolvingP<T> XEv, AdvanceP<T> XAdv, EvolvingP<T> XEv_o);
template <class T> __host__ void AdvkernelCPU(Param XParam, BlockP<T> XBlock, T dt, T* zb, EvolvingP<T> XEv, AdvanceP<T> XAdv, EvolvingP<T> XEv_o);

template <class T> __global__ void cleanupGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, EvolvingP<T> XEv_o);
template <class T> __host__ void cleanupCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, EvolvingP<T> XEv_o);

template <class T> __host__ T CalctimestepCPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, TimeP<T> XTime);
template <class T> __host__ T CalctimestepGPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, TimeP<T> XTime);



// End of global definition
#endif

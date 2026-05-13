
#ifndef GROUNDWATER_H
#define GROUNDWATER_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"

template <class T> void GroundwaterStepGPU(Param XParam, Loop<T>& XLoop, Model<T> XModel);

template <class T> __global__ void DarcyFluxXGPU(Param XParam, BlockP<T> XBlock, T* hgw, T* hsw, T* zb, T* K_gw, T* Aquifer_Depth, T* Qx);
template <class T> __global__ void DarcyFluxYGPU(Param XParam, BlockP<T> XBlock, T* hgw, T* hsw, T* zb, T* K_gw, T* Aquifer_Depth, T* Qy);
template <class T> __global__ void GroundwaterMassBalanceGPU(Param XParam, T dt, BlockP<T> XBlock, T* hgw, T* hsw, T* zs, T* topo, T* fs_gw, T* Sy_gw, T* Qx, T* Qy);

#endif

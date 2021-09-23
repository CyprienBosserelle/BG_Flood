#ifndef FRICTION_H
#define FRICTION_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "MemManagement.h"

template <class T> __global__ void bottomfrictionGPU(Param XParam, BlockP<T> XBlock,T dt, T* cf, EvolvingP<T> XEvolv);
template <class T> __host__ void bottomfrictionCPU(Param XParam, BlockP<T> XBlock,T dt, T* cf, EvolvingP<T> XEvolv);

template <class T> __host__ void XiafrictionCPU(Param XParam, BlockP<T> XBlock, T dt, T* cf, EvolvingP<T> XEvolv, EvolvingP<T> XEvolv_o);
template <class T> __global__ void XiafrictionGPU(Param XParam, BlockP<T> XBlock, T dt, T* cf, EvolvingP<T> XEvolv, EvolvingP<T> XEvolv_o);

template <class T> __host__ __device__ T smartfriction(T hi,T zo);
template <class T> __host__ __device__ T manningfriction(T g, T hi, T n);

template <class T> __global__ void TheresholdVelGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEvolv);
template <class T> __host__ void TheresholdVelCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEvolv);

// End of global definition
#endif

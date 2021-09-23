#ifndef KURGANOV_H
#define KURGANOV_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Util_CPU.h"

template <class T> __global__ void updateKurgXGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb);
template <class T> __global__ void AddSlopeSourceXGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* zb);

template <class T> __host__ void updateKurgXCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb);
template <class T> __host__ void AddSlopeSourceXCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* zb);


template <class T> __global__ void updateKurgYGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb);
template <class T> __global__ void AddSlopeSourceYGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* zb);

template <class T> __host__ void updateKurgYCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb);
template <class T> __host__ void AddSlopeSourceYCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* zb);

template <class T> __host__ __device__ T KurgSolver(T g, T delta, T epsi, T CFL, T cm, T fm, T hp, T hm, T up, T um, T& fh, T& fu);

#endif

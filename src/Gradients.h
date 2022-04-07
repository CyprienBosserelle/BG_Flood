#ifndef GRADIENTS_H
#define GRADIENTS_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "Util_CPU.h"
#include "Setup_GPU.h"
#include "MemManagement.h"
#include "Halo.h"

// CPU specific functions
template <class T> void gradientCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);
template <class T> void gradientC(Param XParam, BlockP<T> XBlock, T* a, T* dadx, T* dady);
template <class T> void gradientHalo(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady);

template <class T> void WetsloperesetCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);
template <class T> void WetsloperesetHaloLeftCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);
template <class T> void WetsloperesetHaloBotCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);
template <class T> void WetsloperesetHaloRightCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);
template <class T>  void WetsloperesetHaloTopCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);

// GPU specific functions
template <class T> void gradientGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);
template <class T> __global__ void gradient(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dadx, T* dady);
template <class T> void gradientHaloGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady);

template <class T> __global__ void WetsloperesetXGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);
template <class T> __global__ void WetsloperesetYGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);

template <class T> __global__ void WetsloperesetHaloLeftGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);
template <class T> __global__ void WetsloperesetHaloRightGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);
template <class T> __global__ void WetsloperesetHaloBotGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);
template <class T> __global__ void WetsloperesetHaloTopGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);


template <class T> __global__ void gradientHaloLeftGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady);
template <class T> __global__ void gradientHaloRightGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady);
template <class T> __global__ void gradientHaloTopGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady);
template <class T> __global__ void gradientHaloBotGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady);
// End of global definition
#endif

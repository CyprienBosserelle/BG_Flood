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

template <class T> void gradientC(Param XParam, BlockP<T> XBlock, T* a, T* dadx, T* dady);

template <class T> void gradientGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);
template <class T> void gradientCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb);

template <class T> __global__ void gradient(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dadx, T* dady);


// End of global definition
#endif

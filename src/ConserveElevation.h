
#ifndef CONSERVEELEVATION_H
#define CONSERVEELEVATION_H

#include "General.h"
#include "Param.h"
#include "Write_txtlog.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "MemManagement.h"

template <class T> void conserveElevation(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);

template <class T> void conserveElevationGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);


//template <class T> void conserveElevationGradHalo(Param XParam, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy);
template <class T> void conserveElevationGradHalo(Param XParam, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx, T* dhdy, T* dzsdy);
//template <class T> void conserveElevationGradHaloGPU(Param XParam, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy);
template <class T> void conserveElevationGradHaloGPU(Param XParam, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx, T* dhdy, T* dzsdy);

// End of global definition
#endif

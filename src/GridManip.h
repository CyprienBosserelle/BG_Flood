
#ifndef GRIDMANIP_H
#define GRIDMANIP_H

#include "General.h"
#include "Param.h"
#include "Util_CPU.h"
#include "Forcing.h"
#include "Arrays.h"
#include "MemManagement.h"

template <class T, class F> void CopyArrayBUQ(Param XParam, BlockP<F> XBlock, T* source, T*& dest);
template <class T> void CopyArrayBUQ(Param XParam, BlockP<T> XBlock, EvolvingP<T> source, EvolvingP<T>& dest);
template <class T, class F> void InitArrayBUQ(Param XParam, BlockP<F> XBlock, T initval, T*& Arr);
template <class T, class F> void InitBlkBUQ(Param XParam, BlockP<F> XBlock, T initval, T*& Arr);
template <class T>  void setedges(Param XParam, BlockP<T> XBlock, T*& zb);


template <class T> void interp2BUQ(Param XParam, BlockP<T> XBlock, std::vector<StaticForcingP<float>> forcing, T* z);
template <class T, class F> void interp2BUQ(Param XParam, BlockP<T> XBlock, F forcing, T*& z);
template <class T, class F> T interp2BUQ(T x, T y, F forcing);

template <class T, class F> void InterpstepCPU(int nx, int ny, int hdstep, F totaltime, F hddt, T*& Ux, T* Uo, T* Un);
template <class T> __global__ void InterpstepGPU(int nx, int ny, int hdstp, T totaltime, T hddt, T* Ux, T* Uo, T* Un);

template <class T> void Copy2CartCPU(int nx, int ny, T* dest, T* src);

// End of global definition
#endif

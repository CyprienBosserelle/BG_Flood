
#ifndef BOUNDARY_H
#define BOUNDARY_H
// includes, system

#include "General.h"
#include "MemManagement.h"
#include "Util_CPU.h"



template <class T> void Flowbnd(Param XParam, Loop<T>& XLoop, BlockP<T> XBlock, bndparam side, EvolvingP<T> XEv);
__host__ __device__ int Inside(int halowidth, int blkmemwidth, int isright, int istop, int ix, int iy, int ib);
__host__ __device__ bool isbnd(int isright, int istop, int blkwidth, int ix, int iy);

template <class T> __host__ void maskbnd(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb);
template <class T> __global__ void maskbndGPUleft(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb);
template <class T> __global__ void maskbndGPUtop(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb);
template <class T> __global__ void maskbndGPUright(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb);
template <class T> __global__ void maskbndGPUbot(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb);

// End of global definition
#endif

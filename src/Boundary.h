
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

template <class T> __global__ void bndGPU(Param XParam, bndparam side, BlockP<T> XBlock, float itime, T* zs, T* h, T* un, T* ut);


__device__ __host__ void findmaskside(int side, bool &isleftbot, bool& islefttop, bool& istopleft, bool& istopright, bool& isrighttop, bool& isrightbot, bool& isbotright, bool& isbotleft);
template <class T> __device__ __host__ void halowall(T zsinside, T& un, T& ut, T& zs, T& h,T&zb);
template <class T> __device__ __host__ void noslipbnd(T zsinside,T hinside,T &un, T &ut,T &zs, T &h);
template <class T> __device__ __host__ void ABS1D(T g, T sign, T zsbnd, T zsinside, T hinside, T utbnd,T unbnd, T& un, T& ut, T& zs, T& h);
template <class T> __device__ __host__ void Dirichlet1D(T g, T sign, T zsbnd, T zsinside, T hinside,  T uninside, T& un, T& ut, T& zs, T& h);

// End of global definition
#endif

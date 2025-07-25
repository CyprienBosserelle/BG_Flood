
#ifndef CONSERVEELEVATION_H
#define CONSERVEELEVATION_H

#include "General.h"
#include "Param.h"
#include "Write_txtlog.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "MemManagement.h"

template <class T> void conserveElevation(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb, int nblk_local_start = 0);

template <class T> void conserveElevationGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);


//template <class T> void conserveElevationGradHalo(Param XParam, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy);
template <class T> void conserveElevationGradHalo(Param XParam, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx, T* dhdy, T* dzsdy, int nblk_local_start = 0);
//template <class T> void conserveElevationGradHaloGPU(Param XParam, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy);
template <class T> void conserveElevationGradHaloGPU(Param XParam, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx, T* dhdy, T* dzsdy);

template <class T> void conserveElevationLeft(Param XParam, int ib, int ibLB, int ibLT, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);
template <class T> __global__ void conserveElevationLeft(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);

template <class T> void conserveElevationRight(Param XParam, int ib, int ibRB, int ibRT, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);
template <class T> __global__ void conserveElevationRight(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);

template <class T> void conserveElevationTop(Param XParam, int ib, int ibTL, int ibTR, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);
template <class T> __global__ void conserveElevationTop(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);

template <class T> void conserveElevationBot(Param XParam, int ib, int ibBL, int ibBR, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);
template <class T> __global__ void conserveElevationBot(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);

template <class T> void conserveElevationGHLeft(Param XParam, int ib, int ibLB, int ibLT, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx);
template <class T> __global__ void conserveElevationGHLeft(Param XParam, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx);

template <class T> void conserveElevationGHRight(Param XParam, int ib, int ibRB, int ibRT, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx);
template <class T> __global__ void conserveElevationGHRight(Param XParam, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx);

template <class T> void conserveElevationGHTop(Param XParam, int ib, int ibTL, int ibTR, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx);
template <class T> __global__ void conserveElevationGHTop(Param XParam, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx);

template <class T> void conserveElevationGHBot(Param XParam, int ib, int ibBL, int ibBR, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx);
template <class T> __global__ void conserveElevationGHBot(Param XParam, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx);

template <class T> void WetDryProlongationGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);
template <class T> void WetDryProlongation(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb, int nblk_local_start = 0);

template <class T> __global__ void WetDryProlongationGPURight(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);
template <class T> __global__ void WetDryProlongationGPUTop(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);
template <class T> __global__ void WetDryProlongationGPUBot(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);
template <class T> __global__ void WetDryProlongationGPULeft(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);

template <class T> void WetDryRestrictionGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);
template <class T> void WetDryRestriction(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb, int nblk_local_start = 0);

template <class T> __global__ void WetDryRestrictionGPULeft(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);
template <class T> __global__ void WetDryRestrictionGPUTop(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);
template <class T> __global__ void WetDryRestrictionGPUBot(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);
template <class T> __global__ void WetDryRestrictionGPURight(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);

template <class T> __host__ __device__ void wetdryrestriction(int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, T* h, T* zs, T* zb);
template <class T> __host__ __device__ void ProlongationElevation(int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int ip, int jp, T* h, T* zs, T* zb);

// End of global definition
#endif

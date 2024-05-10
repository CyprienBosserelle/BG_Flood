
#ifndef BOUNDARY_H
#define BOUNDARY_H
// includes, system

#include "General.h"
#include "MemManagement.h"
#include "Util_CPU.h"
#include "Updateforcing.h"



template <class T> void Flowbnd(Param XParam, Loop<T>& XLoop, BlockP<T> XBlock, bndparam side, DynForcingP<float> Atmp, EvolvingP<T> XEv);
__host__ __device__ int Inside(int halowidth, int blkmemwidth, int isright, int istop, int ix, int iy, int ib);
__host__ __device__ bool isbnd(int isright, int istop, int blkwidth, int ix, int iy);

template <class T> __host__ void maskbnd(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb);
template <class T> __global__ void maskbndGPUleft(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb);
template <class T> __global__ void maskbndGPUtop(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb);
template <class T> __global__ void maskbndGPUright(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb);
template <class T> __global__ void maskbndGPUbot(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb);

template <class T> __global__ void maskbndGPUFluxleft(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, FluxP<T> Flux);
template <class T> __global__ void maskbndGPUFluxtop(Param XParam, BlockP<T> XBlock, FluxP<T> Flux);
template <class T> __global__ void maskbndGPUFluxright(Param XParam, BlockP<T> XBlock, FluxP<T> Flux);
template <class T> __global__ void maskbndGPUFluxbot(Param XParam, BlockP<T> XBlock, FluxP<T> Flux);

template <class T> void FlowbndFlux(Param XParam, double totaltime, BlockP<T> XBlock, bndsegment bndseg, DynForcingP<float> Atmp, EvolvingP<T> XEv, FluxP<T> XFlux);
template <class T> void FlowbndFlux(Param XParam,double totaltime, BlockP<T> XBlock, bndparam side, DynForcingP<float> Atmp, EvolvingP<T> XEv, FluxP<T> XFlux);


template <class T> __global__ void bndFluxGPUSide(Param XParam, bndsegmentside side, BlockP<T> XBlock, DynForcingP<float> Atmp, DynForcingP<float> Zsmap, bool uniform, int type, float zsbnd, T* zs, T* h, T* un, T* ut, T* Fh, T* Fq, T* Ss);

template <class T> __global__ void bndGPU(Param XParam, bndparam side, BlockP<T> XBlock, DynForcingP<float> Atmp, float itime, T* zs, T* h, T* un, T* ut);
template <class T> __host__ void bndCPU(Param XParam, bndparam side, BlockP<T> XBlock, std::vector<double> zsbndvec, std::vector<double> uubndvec, std::vector<double> vvbndvec, DynForcingP<float> Atmp, T* zs, T* h, T* un, T* ut);


__device__ __host__ void findmaskside(int side, bool &isleftbot, bool& islefttop, bool& istopleft, bool& istopright, bool& isrighttop, bool& isrightbot, bool& isbotright, bool& isbotleft);
template <class T> __device__ __host__ void halowall(T zsinside, T& un, T& ut, T& zs, T& h,T&zb);
template <class T> __device__ __host__ void noslipbnd(T zsinside,T hinside,T &un, T &ut,T &zs, T &h);
template <class T> __device__ __host__ void noslipbndQ(T& F, T& G, T& S);
template <class T> __device__ __host__ void ABS1D(T g, T sign, T zsbnd, T zsinside, T hinside, T utbnd,T unbnd, T& un, T& ut, T& zs, T& h);
template <class T> __device__ __host__ void ABS1DQ(T g, T sign, T factime, T facrel, T zs, T zsbnd, T zsinside, T h, T& qmean, T& q, T& G, T& S);
template <class T> __device__ __host__ void Dirichlet1D(T g, T sign, T zsbnd, T zsinside, T hinside,  T uninside, T& un, T& ut, T& zs, T& h);

// End of global definition
#endif

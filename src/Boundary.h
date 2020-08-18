
#ifndef BOUNDARY_H
#define BOUNDARY_H
// includes, system

#include "General.h"
#include "MemManagement.h"
#include "Util_CPU.h"


template <class T> __global__ void noslipbnd(int halowidth, int isright, int istop, int* bndblck, T* zs, T* h, T* un);
template <class T> __host__ void noslipbnd(Param XParam, int isright, int istop, int* bndblck, T* zs, T* h, T* un);

template <class T> __global__ void ABS1D(int halowidth, int isright, int istop, int nbnd, T g, T dx, T xo, T yo, T xmax, T ymax, T itime, cudaTextureObject_t texZsBND, int* bndblck, int* level, T* blockxo, T* blockyo, T* zs, T* zb, T* h, T* un, T* ut);
template <class T> __host__ void ABS1D(Param XParam, std::vector<double> zsbndvec, int isright, int istop, int nbnd, T itime, BlockP<T> XBlock, int* bndblk, T* zs, T* zb, T* h, T* un, T* ut);


template <class T> void Flowbnd(Param XParam, Loop<T>& XLoop, bndparam side, int isright, int istop, EvolvingP<T> XEv, T* zb);
__host__ __device__ int Inside(int halowidth, int blkmemwidth, int isright, int istop, int ix, int iy, int ib);
__host__ __device__ bool isbnd(int isright, int istop, int blkwidth, int ix, int iy);

// End of global definition
#endif

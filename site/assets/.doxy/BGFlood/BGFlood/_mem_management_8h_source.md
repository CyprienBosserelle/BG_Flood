

# File MemManagement.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**MemManagement.h**](_mem_management_8h.md)

[Go to the documentation of this file](_mem_management_8h.md)


```C++

#ifndef MEMMANAGEMENT_H
#define MEMMANAGEMENT_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Setup_GPU.h"


template <class T> void AllocateCPU(int nx, int ny, T*& zb);
template <class T> void AllocateCPU(int nx, int ny, T*& zs, T*& h, T*& u, T*& v);
template <class T> void AllocateCPU(int nx, int ny, T*& zs, T*& h, T*& u, T*& v, T*& U, T*& hU);

template <class T> void AllocateCPU(int nx, int ny, GradientsP<T>& Grad);
template <class T> void AllocateCPU(int nblk, int blksize, EvolvingP<T> &Ev);
template <class T> void AllocateCPU(int nblk, int blksize, EvolvingP_M<T>& Ev);
template <class T> void AllocateCPU(int nblk, int blksize, Param XParam, Model<T>& XModel);


template <class T> void ReallocArray(int nblk, int blksize, T*& zb);
template <class T> void ReallocArray(int nblk, int blksize, T*& zs, T*& h, T*& u, T*& v);
template <class T> void ReallocArray(int nblk, int blksize, T*& zs, T*& h, T*& u, T*& v, T*& U, T*& hU);
template <class T> void ReallocArray(int nblk, int blksize, EvolvingP<T>& Ev);
template <class T> void ReallocArray(int nblk, int blksize, EvolvingP_M<T>& Ev);
template <class T> void ReallocArray(int nblk, int blksize, Param XParam, Model<T>& XModel);

template <class T> void AllocateMappedMemCPU(int nx, int ny, int gpudevice, T*& z);


template <class T> __host__ void FillCPU(int nx, int ny, T fillval, T*& zb);

int memloc(Param XParam, int i, int j, int ib);
//__device__ int memloc(int halowidth, int blkmemwidth, int  blksize, int i, int j, int ib);
__host__ __device__ int memloc(int halowidth, int blkmemwidth, int i, int j, int ib);

template <class T> void AllocateGPU(int nblk, int blksize, Param XParam, Model<T>& XModel);
template <class T> void AllocateGPU(int nx, int ny, T*& z_g);
template <class T> void AllocateMappedMemGPU(int nx, int ny, int gpudevice, T*& z_g, T* z);
// End of global definition
#endif
```





# File Reimann.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Reimann.h**](Reimann_8h.md)

[Go to the documentation of this file](Reimann_8h.md)


```C++
#ifndef REIMANN_H
#define REIMANN_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Util_CPU.h"

template <class T> __global__ void UpdateButtingerXGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb);
template <class T> __host__ void UpdateButtingerXCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb);
template <class T> __global__ void UpdateButtingerYGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb);
template <class T> __host__ void UpdateButtingerYCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb);
template <class T> __host__ __device__ T hllc(T g, T delta, T epsi, T CFL, T cm, T fm, T hm, T hp, T um, T up, T& fh, T& fq);
#endif
```



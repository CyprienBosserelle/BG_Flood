

# File Multilayer.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Multilayer.h**](Multilayer_8h.md)

[Go to the documentation of this file](Multilayer_8h.md)


```C++
#ifndef MULTILAYER_H
#define MULTILAYER_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Spherical.h"
#include "Util_CPU.h"

template <class T> __global__ void CalcfaceValX(T pdt, Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux, T* dtmax, T* zb);
template <class T> __global__ void CalcfaceValY(T pdt, Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux, T* dtmax, T* zb);

template <class T> __global__ void CheckadvecMLX(Param XParam, BlockP<T> XBlock, T dt, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux);
template <class T> __global__ void CheckadvecMLY(Param XParam, BlockP<T> XBlock, T dt, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux);
template <class T> __global__ void AdvecFluxML(Param XParam, BlockP<T> XBlock, T dt, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux);
template <class T> __global__ void AdvecEv(Param XParam, BlockP<T> XBlock, T dt, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux);
template <class T> __global__ void pressureML(Param XParam, BlockP<T> XBlock, T dt, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux);

template <class T> __global__ void CleanupML(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb);


// End of global definition
#endif
```



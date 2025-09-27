

# File Kurganov.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Kurganov.h**](Kurganov_8h.md)

[Go to the source code of this file](Kurganov_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "MemManagement.h"`
* `#include "Util_CPU.h"`
* `#include "Spherical.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**AddSlopeSourceXCPU**](#function-addslopesourcexcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* zb) <br> |
|  \_\_global\_\_ void | [**AddSlopeSourceXGPU**](#function-addslopesourcexgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* zb) <br> |
|  \_\_host\_\_ void | [**AddSlopeSourceYCPU**](#function-addslopesourceycpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* zb) <br> |
|  \_\_global\_\_ void | [**AddSlopeSourceYGPU**](#function-addslopesourceygpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* zb) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**KurgSolver**](#function-kurgsolver) (T g, T delta, T epsi, T CFL, T cm, T fm, T hp, T hm, T up, T um, T & fh, T & fu) <br> |
|  \_\_host\_\_ void | [**updateKurgXATMCPU**](#function-updatekurgxatmcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdx) <br> |
|  \_\_global\_\_ void | [**updateKurgXATMGPU**](#function-updatekurgxatmgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdx) <br> |
|  \_\_host\_\_ void | [**updateKurgXCPU**](#function-updatekurgxcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br> |
|  \_\_global\_\_ void | [**updateKurgXGPU**](#function-updatekurgxgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br> |
|  \_\_host\_\_ void | [**updateKurgYATMCPU**](#function-updatekurgyatmcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdy) <br> |
|  \_\_global\_\_ void | [**updateKurgYATMGPU**](#function-updatekurgyatmgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdy) <br> |
|  \_\_host\_\_ void | [**updateKurgYCPU**](#function-updatekurgycpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br> |
|  \_\_global\_\_ void | [**updateKurgYGPU**](#function-updatekurgygpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br> |




























## Public Functions Documentation




### function AddSlopeSourceXCPU 

```C++
template<class T>
__host__ void AddSlopeSourceXCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * zb
) 
```




<hr>



### function AddSlopeSourceXGPU 

```C++
template<class T>
__global__ void AddSlopeSourceXGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * zb
) 
```




<hr>



### function AddSlopeSourceYCPU 

```C++
template<class T>
__host__ void AddSlopeSourceYCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * zb
) 
```




<hr>



### function AddSlopeSourceYGPU 

```C++
template<class T>
__global__ void AddSlopeSourceYGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * zb
) 
```




<hr>



### function KurgSolver 

```C++
template<class T>
__host__ __device__ T KurgSolver (
    T g,
    T delta,
    T epsi,
    T CFL,
    T cm,
    T fm,
    T hp,
    T hm,
    T up,
    T um,
    T & fh,
    T & fu
) 
```




<hr>



### function updateKurgXATMCPU 

```C++
template<class T>
__host__ void updateKurgXATMCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb,
    T * Patm,
    T * dPdx
) 
```




<hr>



### function updateKurgXATMGPU 

```C++
template<class T>
__global__ void updateKurgXATMGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb,
    T * Patm,
    T * dPdx
) 
```




<hr>



### function updateKurgXCPU 

```C++
template<class T>
__host__ void updateKurgXCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb
) 
```




<hr>



### function updateKurgXGPU 

```C++
template<class T>
__global__ void updateKurgXGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb
) 
```




<hr>



### function updateKurgYATMCPU 

```C++
template<class T>
__host__ void updateKurgYATMCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb,
    T * Patm,
    T * dPdy
) 
```




<hr>



### function updateKurgYATMGPU 

```C++
template<class T>
__global__ void updateKurgYATMGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb,
    T * Patm,
    T * dPdy
) 
```




<hr>



### function updateKurgYCPU 

```C++
template<class T>
__host__ void updateKurgYCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb
) 
```




<hr>



### function updateKurgYGPU 

```C++
template<class T>
__global__ void updateKurgYGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Kurganov.h`


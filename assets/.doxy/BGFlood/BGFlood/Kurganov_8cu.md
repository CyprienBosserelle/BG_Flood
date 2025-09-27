

# File Kurganov.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Kurganov.cu**](Kurganov_8cu.md)

[Go to the source code of this file](Kurganov_8cu_source.md)



* `#include "Kurganov.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**AddSlopeSourceXCPU**](#function-addslopesourcexcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* zb) <br> |
|  template \_\_host\_\_ void | [**AddSlopeSourceXCPU&lt; double &gt;**](#function-addslopesourcexcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, double \* zb) <br> |
|  template \_\_host\_\_ void | [**AddSlopeSourceXCPU&lt; float &gt;**](#function-addslopesourcexcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, float \* zb) <br> |
|  \_\_global\_\_ void | [**AddSlopeSourceXGPU**](#function-addslopesourcexgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* zb) <br> |
|  template \_\_global\_\_ void | [**AddSlopeSourceXGPU&lt; double &gt;**](#function-addslopesourcexgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, double \* zb) <br> |
|  template \_\_global\_\_ void | [**AddSlopeSourceXGPU&lt; float &gt;**](#function-addslopesourcexgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, float \* zb) <br> |
|  \_\_host\_\_ void | [**AddSlopeSourceYCPU**](#function-addslopesourceycpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* zb) <br> |
|  template \_\_host\_\_ void | [**AddSlopeSourceYCPU&lt; double &gt;**](#function-addslopesourceycpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, double \* zb) <br> |
|  template \_\_host\_\_ void | [**AddSlopeSourceYCPU&lt; float &gt;**](#function-addslopesourceycpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, float \* zb) <br> |
|  \_\_global\_\_ void | [**AddSlopeSourceYGPU**](#function-addslopesourceygpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* zb) <br> |
|  template \_\_global\_\_ void | [**AddSlopeSourceYGPU&lt; double &gt;**](#function-addslopesourceygpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, double \* zb) <br> |
|  template \_\_global\_\_ void | [**AddSlopeSourceYGPU&lt; float &gt;**](#function-addslopesourceygpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, float \* zb) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**KurgSolver**](#function-kurgsolver) (T g, T delta, T epsi, T CFL, T cm, T fm, T hp, T hm, T up, T um, T & fh, T & fu) <br> |
|  \_\_host\_\_ void | [**updateKurgXATMCPU**](#function-updatekurgxatmcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdx) <br> |
|  template \_\_host\_\_ void | [**updateKurgXATMCPU&lt; double &gt;**](#function-updatekurgxatmcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb, double \* Patm, double \* dPdx) <br> |
|  template \_\_host\_\_ void | [**updateKurgXATMCPU&lt; float &gt;**](#function-updatekurgxatmcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb, float \* Patm, float \* dPdx) <br> |
|  \_\_global\_\_ void | [**updateKurgXATMGPU**](#function-updatekurgxatmgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdx) <br> |
|  template \_\_global\_\_ void | [**updateKurgXATMGPU&lt; double &gt;**](#function-updatekurgxatmgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb, double \* Patm, double \* dPdx) <br> |
|  template \_\_global\_\_ void | [**updateKurgXATMGPU&lt; float &gt;**](#function-updatekurgxatmgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb, float \* Patm, float \* dPdx) <br> |
|  \_\_host\_\_ void | [**updateKurgXCPU**](#function-updatekurgxcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br> |
|  template \_\_host\_\_ void | [**updateKurgXCPU&lt; double &gt;**](#function-updatekurgxcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  template \_\_host\_\_ void | [**updateKurgXCPU&lt; float &gt;**](#function-updatekurgxcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |
|  \_\_global\_\_ void | [**updateKurgXGPU**](#function-updatekurgxgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br> |
|  template \_\_global\_\_ void | [**updateKurgXGPU&lt; double &gt;**](#function-updatekurgxgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  template \_\_global\_\_ void | [**updateKurgXGPU&lt; float &gt;**](#function-updatekurgxgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |
|  \_\_host\_\_ void | [**updateKurgYATMCPU**](#function-updatekurgyatmcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdy) <br> |
|  template \_\_host\_\_ void | [**updateKurgYATMCPU&lt; double &gt;**](#function-updatekurgyatmcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb, double \* Patm, double \* dPdy) <br> |
|  template \_\_host\_\_ void | [**updateKurgYATMCPU&lt; float &gt;**](#function-updatekurgyatmcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb, float \* Patm, float \* dPdy) <br> |
|  \_\_global\_\_ void | [**updateKurgYATMGPU**](#function-updatekurgyatmgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdy) <br> |
|  template \_\_global\_\_ void | [**updateKurgYATMGPU&lt; double &gt;**](#function-updatekurgyatmgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb, double \* Patm, double \* dPdy) <br> |
|  template \_\_global\_\_ void | [**updateKurgYATMGPU&lt; float &gt;**](#function-updatekurgyatmgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb, float \* Patm, float \* dPdy) <br> |
|  \_\_host\_\_ void | [**updateKurgYCPU**](#function-updatekurgycpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br> |
|  template \_\_host\_\_ void | [**updateKurgYCPU&lt; double &gt;**](#function-updatekurgycpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  template \_\_host\_\_ void | [**updateKurgYCPU&lt; float &gt;**](#function-updatekurgycpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |
|  \_\_global\_\_ void | [**updateKurgYGPU**](#function-updatekurgygpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br> |
|  template \_\_global\_\_ void | [**updateKurgYGPU&lt; double &gt;**](#function-updatekurgygpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  template \_\_global\_\_ void | [**updateKurgYGPU&lt; float &gt;**](#function-updatekurgygpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |




























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



### function AddSlopeSourceXCPU&lt; double &gt; 

```C++
template __host__ void AddSlopeSourceXCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * zb
) 
```




<hr>



### function AddSlopeSourceXCPU&lt; float &gt; 

```C++
template __host__ void AddSlopeSourceXCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * zb
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



### function AddSlopeSourceXGPU&lt; double &gt; 

```C++
template __global__ void AddSlopeSourceXGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * zb
) 
```




<hr>



### function AddSlopeSourceXGPU&lt; float &gt; 

```C++
template __global__ void AddSlopeSourceXGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * zb
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



### function AddSlopeSourceYCPU&lt; double &gt; 

```C++
template __host__ void AddSlopeSourceYCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * zb
) 
```




<hr>



### function AddSlopeSourceYCPU&lt; float &gt; 

```C++
template __host__ void AddSlopeSourceYCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * zb
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



### function AddSlopeSourceYGPU&lt; double &gt; 

```C++
template __global__ void AddSlopeSourceYGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * zb
) 
```




<hr>



### function AddSlopeSourceYGPU&lt; float &gt; 

```C++
template __global__ void AddSlopeSourceYGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * zb
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



### function updateKurgXATMCPU&lt; double &gt; 

```C++
template __host__ void updateKurgXATMCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb,
    double * Patm,
    double * dPdx
) 
```




<hr>



### function updateKurgXATMCPU&lt; float &gt; 

```C++
template __host__ void updateKurgXATMCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb,
    float * Patm,
    float * dPdx
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



### function updateKurgXATMGPU&lt; double &gt; 

```C++
template __global__ void updateKurgXATMGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb,
    double * Patm,
    double * dPdx
) 
```




<hr>



### function updateKurgXATMGPU&lt; float &gt; 

```C++
template __global__ void updateKurgXATMGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb,
    float * Patm,
    float * dPdx
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



### function updateKurgXCPU&lt; double &gt; 

```C++
template __host__ void updateKurgXCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb
) 
```




<hr>



### function updateKurgXCPU&lt; float &gt; 

```C++
template __host__ void updateKurgXCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb
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



### function updateKurgXGPU&lt; double &gt; 

```C++
template __global__ void updateKurgXGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb
) 
```




<hr>



### function updateKurgXGPU&lt; float &gt; 

```C++
template __global__ void updateKurgXGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb
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



### function updateKurgYATMCPU&lt; double &gt; 

```C++
template __host__ void updateKurgYATMCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb,
    double * Patm,
    double * dPdy
) 
```




<hr>



### function updateKurgYATMCPU&lt; float &gt; 

```C++
template __host__ void updateKurgYATMCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb,
    float * Patm,
    float * dPdy
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



### function updateKurgYATMGPU&lt; double &gt; 

```C++
template __global__ void updateKurgYATMGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb,
    double * Patm,
    double * dPdy
) 
```




<hr>



### function updateKurgYATMGPU&lt; float &gt; 

```C++
template __global__ void updateKurgYATMGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb,
    float * Patm,
    float * dPdy
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



### function updateKurgYCPU&lt; double &gt; 

```C++
template __host__ void updateKurgYCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb
) 
```




<hr>



### function updateKurgYCPU&lt; float &gt; 

```C++
template __host__ void updateKurgYCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb
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



### function updateKurgYGPU&lt; double &gt; 

```C++
template __global__ void updateKurgYGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb
) 
```




<hr>



### function updateKurgYGPU&lt; float &gt; 

```C++
template __global__ void updateKurgYGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Kurganov.cu`


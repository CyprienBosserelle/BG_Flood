

# File Multilayer.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Multilayer.h**](Multilayer_8h.md)

[Go to the source code of this file](Multilayer_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "MemManagement.h"`
* `#include "Spherical.h"`
* `#include "Util_CPU.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_global\_\_ void | [**AdvecEv**](#function-advecev) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxMLP**](structFluxMLP.md)&lt; T &gt; XFlux) <br> |
|  \_\_global\_\_ void | [**AdvecFluxML**](#function-advecfluxml) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxMLP**](structFluxMLP.md)&lt; T &gt; XFlux) <br> |
|  \_\_global\_\_ void | [**CalcfaceValX**](#function-calcfacevalx) (T pdt, [**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxMLP**](structFluxMLP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br> |
|  \_\_global\_\_ void | [**CalcfaceValY**](#function-calcfacevaly) (T pdt, [**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxMLP**](structFluxMLP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br> |
|  \_\_global\_\_ void | [**CheckadvecMLX**](#function-checkadvecmlx) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxMLP**](structFluxMLP.md)&lt; T &gt; XFlux) <br> |
|  \_\_global\_\_ void | [**CheckadvecMLY**](#function-checkadvecmly) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxMLP**](structFluxMLP.md)&lt; T &gt; XFlux) <br> |
|  \_\_global\_\_ void | [**CleanupML**](#function-cleanupml) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**pressureML**](#function-pressureml) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxMLP**](structFluxMLP.md)&lt; T &gt; XFlux) <br> |




























## Public Functions Documentation




### function AdvecEv 

```C++
template<class T>
__global__ void AdvecEv (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux
) 
```




<hr>



### function AdvecFluxML 

```C++
template<class T>
__global__ void AdvecFluxML (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux
) 
```




<hr>



### function CalcfaceValX 

```C++
template<class T>
__global__ void CalcfaceValX (
    T pdt,
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux,
    T * dtmax,
    T * zb
) 
```




<hr>



### function CalcfaceValY 

```C++
template<class T>
__global__ void CalcfaceValY (
    T pdt,
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux,
    T * dtmax,
    T * zb
) 
```




<hr>



### function CheckadvecMLX 

```C++
template<class T>
__global__ void CheckadvecMLX (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux
) 
```




<hr>



### function CheckadvecMLY 

```C++
template<class T>
__global__ void CheckadvecMLY (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux
) 
```




<hr>



### function CleanupML 

```C++
template<class T>
__global__ void CleanupML (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function pressureML 

```C++
template<class T>
__global__ void pressureML (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Multilayer.h`


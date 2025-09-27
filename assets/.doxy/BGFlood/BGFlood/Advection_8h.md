

# File Advection.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Advection.h**](Advection_8h.md)

[Go to the source code of this file](Advection_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "MemManagement.h"`
* `#include "Spherical.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**AdvkernelCPU**](#function-advkernelcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_o) <br> |
|  \_\_global\_\_ void | [**AdvkernelGPU**](#function-advkernelgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_o) <br> |
|  \_\_host\_\_ T | [**CalctimestepCPU**](#function-calctimestepcpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**TimeP**](structTimeP.md)&lt; T &gt; XTime) <br> |
|  \_\_host\_\_ T | [**CalctimestepGPU**](#function-calctimestepgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**TimeP**](structTimeP.md)&lt; T &gt; XTime) <br> |
|  \_\_host\_\_ void | [**cleanupCPU**](#function-cleanupcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_o) <br> |
|  \_\_global\_\_ void | [**cleanupGPU**](#function-cleanupgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_o) <br> |
|  \_\_global\_\_ void | [**densify**](#function-densify) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* g\_idata, T \* g\_odata) <br> |
|  \_\_global\_\_ void | [**reducemin3**](#function-reducemin3) (T \* g\_idata, T \* g\_odata, unsigned int n) <br> |
|  \_\_host\_\_ T | [**timestepreductionCPU**](#function-timestepreductioncpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**TimeP**](structTimeP.md)&lt; T &gt; XTime) <br> |
|  \_\_host\_\_ void | [**updateEVCPU**](#function-updateevcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br> |
|  \_\_global\_\_ void | [**updateEVGPU**](#function-updateevgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br> |




























## Public Functions Documentation




### function AdvkernelCPU 

```C++
template<class T>
__host__ void AdvkernelCPU (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    T * zb,
    EvolvingP < T > XEv,
    AdvanceP < T > XAdv,
    EvolvingP < T > XEv_o
) 
```




<hr>



### function AdvkernelGPU 

```C++
template<class T>
__global__ void AdvkernelGPU (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    T * zb,
    EvolvingP < T > XEv,
    AdvanceP < T > XAdv,
    EvolvingP < T > XEv_o
) 
```




<hr>



### function CalctimestepCPU 

```C++
template<class T>
__host__ T CalctimestepCPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    TimeP < T > XTime
) 
```




<hr>



### function CalctimestepGPU 

```C++
template<class T>
__host__ T CalctimestepGPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    TimeP < T > XTime
) 
```




<hr>



### function cleanupCPU 

```C++
template<class T>
__host__ void cleanupCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    EvolvingP < T > XEv_o
) 
```




<hr>



### function cleanupGPU 

```C++
template<class T>
__global__ void cleanupGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    EvolvingP < T > XEv_o
) 
```




<hr>



### function densify 

```C++
template<class T>
__global__ void densify (
    Param XParam,
    BlockP < T > XBlock,
    T * g_idata,
    T * g_odata
) 
```




<hr>



### function reducemin3 

```C++
template<class T>
__global__ void reducemin3 (
    T * g_idata,
    T * g_odata,
    unsigned int n
) 
```




<hr>



### function timestepreductionCPU 

```C++
template<class T>
__host__ T timestepreductionCPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    TimeP < T > XTime
) 
```




<hr>



### function updateEVCPU 

```C++
template<class T>
__host__ void updateEVCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    FluxP < T > XFlux,
    AdvanceP < T > XAdv
) 
```




<hr>



### function updateEVGPU 

```C++
template<class T>
__global__ void updateEVGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    FluxP < T > XFlux,
    AdvanceP < T > XAdv
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Advection.h`


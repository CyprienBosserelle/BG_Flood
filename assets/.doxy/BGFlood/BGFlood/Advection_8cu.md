

# File Advection.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Advection.cu**](Advection_8cu.md)

[Go to the source code of this file](Advection_8cu_source.md)



* `#include "Advection.h"`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**SharedMemory**](structSharedMemory.md) &lt;class T&gt;<br> |
| struct | [**SharedMemory&lt; double &gt;**](structSharedMemory_3_01double_01_4.md) &lt;&gt;<br> |






















## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**AdvkernelCPU**](#function-advkernelcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_o) <br> |
|  template \_\_host\_\_ void | [**AdvkernelCPU&lt; double &gt;**](#function-advkernelcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double dt, double \* zb, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**AdvanceP**](structAdvanceP.md)&lt; double &gt; XAdv, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv\_o) <br> |
|  template \_\_host\_\_ void | [**AdvkernelCPU&lt; float &gt;**](#function-advkernelcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float dt, float \* zb, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**AdvanceP**](structAdvanceP.md)&lt; float &gt; XAdv, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv\_o) <br> |
|  \_\_global\_\_ void | [**AdvkernelGPU**](#function-advkernelgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_o) <br> |
|  template \_\_global\_\_ void | [**AdvkernelGPU&lt; double &gt;**](#function-advkernelgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double dt, double \* zb, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**AdvanceP**](structAdvanceP.md)&lt; double &gt; XAdv, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv\_o) <br> |
|  template \_\_global\_\_ void | [**AdvkernelGPU&lt; float &gt;**](#function-advkernelgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float dt, float \* zb, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**AdvanceP**](structAdvanceP.md)&lt; float &gt; XAdv, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv\_o) <br> |
|  \_\_host\_\_ T | [**CalctimestepCPU**](#function-calctimestepcpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**TimeP**](structTimeP.md)&lt; T &gt; XTime) <br> |
|  template \_\_host\_\_ double | [**CalctimestepCPU&lt; double &gt;**](#function-calctimestepcpu-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**TimeP**](structTimeP.md)&lt; double &gt; XTime) <br> |
|  template \_\_host\_\_ float | [**CalctimestepCPU&lt; float &gt;**](#function-calctimestepcpu-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**TimeP**](structTimeP.md)&lt; float &gt; XTime) <br> |
|  \_\_host\_\_ T | [**CalctimestepGPU**](#function-calctimestepgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**TimeP**](structTimeP.md)&lt; T &gt; XTime) <br> |
|  template \_\_host\_\_ double | [**CalctimestepGPU&lt; double &gt;**](#function-calctimestepgpu-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**TimeP**](structTimeP.md)&lt; double &gt; XTime) <br> |
|  template \_\_host\_\_ float | [**CalctimestepGPU&lt; float &gt;**](#function-calctimestepgpu-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**TimeP**](structTimeP.md)&lt; float &gt; XTime) <br> |
|  \_\_host\_\_ void | [**cleanupCPU**](#function-cleanupcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_o) <br> |
|  template \_\_host\_\_ void | [**cleanupCPU&lt; double &gt;**](#function-cleanupcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv\_o) <br> |
|  template \_\_host\_\_ void | [**cleanupCPU&lt; float &gt;**](#function-cleanupcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv\_o) <br> |
|  \_\_global\_\_ void | [**cleanupGPU**](#function-cleanupgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_o) <br> |
|  template \_\_global\_\_ void | [**cleanupGPU&lt; double &gt;**](#function-cleanupgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv\_o) <br> |
|  template \_\_global\_\_ void | [**cleanupGPU&lt; float &gt;**](#function-cleanupgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv\_o) <br> |
|  \_\_global\_\_ void | [**densify**](#function-densify) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* g\_idata, T \* g\_odata) <br> |
|  \_\_global\_\_ void | [**reducemin3**](#function-reducemin3) (T \* g\_idata, T \* g\_odata, unsigned int n) <br> |
|  \_\_host\_\_ T | [**timestepreductionCPU**](#function-timestepreductioncpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**TimeP**](structTimeP.md)&lt; T &gt; XTime) <br> |
|  template \_\_host\_\_ float | [**timestepreductionCPU**](#function-timestepreductioncpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**TimeP**](structTimeP.md)&lt; float &gt; XTime) <br> |
|  template \_\_host\_\_ double | [**timestepreductionCPU**](#function-timestepreductioncpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**TimeP**](structTimeP.md)&lt; double &gt; XTime) <br> |
|  \_\_host\_\_ void | [**updateEVCPU**](#function-updateevcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br> |
|  template \_\_host\_\_ void | [**updateEVCPU&lt; double &gt;**](#function-updateevcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, [**AdvanceP**](structAdvanceP.md)&lt; double &gt; XAdv) <br> |
|  template \_\_host\_\_ void | [**updateEVCPU&lt; float &gt;**](#function-updateevcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, [**AdvanceP**](structAdvanceP.md)&lt; float &gt; XAdv) <br> |
|  \_\_global\_\_ void | [**updateEVGPU**](#function-updateevgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br> |
|  template \_\_global\_\_ void | [**updateEVGPU&lt; double &gt;**](#function-updateevgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux, [**AdvanceP**](structAdvanceP.md)&lt; double &gt; XAdv) <br> |
|  template \_\_global\_\_ void | [**updateEVGPU&lt; float &gt;**](#function-updateevgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux, [**AdvanceP**](structAdvanceP.md)&lt; float &gt; XAdv) <br> |




























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



### function AdvkernelCPU&lt; double &gt; 

```C++
template __host__ void AdvkernelCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double dt,
    double * zb,
    EvolvingP < double > XEv,
    AdvanceP < double > XAdv,
    EvolvingP < double > XEv_o
) 
```




<hr>



### function AdvkernelCPU&lt; float &gt; 

```C++
template __host__ void AdvkernelCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float dt,
    float * zb,
    EvolvingP < float > XEv,
    AdvanceP < float > XAdv,
    EvolvingP < float > XEv_o
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



### function AdvkernelGPU&lt; double &gt; 

```C++
template __global__ void AdvkernelGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double dt,
    double * zb,
    EvolvingP < double > XEv,
    AdvanceP < double > XAdv,
    EvolvingP < double > XEv_o
) 
```




<hr>



### function AdvkernelGPU&lt; float &gt; 

```C++
template __global__ void AdvkernelGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float dt,
    float * zb,
    EvolvingP < float > XEv,
    AdvanceP < float > XAdv,
    EvolvingP < float > XEv_o
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



### function CalctimestepCPU&lt; double &gt; 

```C++
template __host__ double CalctimestepCPU< double > (
    Param XParam,
    Loop < double > XLoop,
    BlockP < double > XBlock,
    TimeP < double > XTime
) 
```




<hr>



### function CalctimestepCPU&lt; float &gt; 

```C++
template __host__ float CalctimestepCPU< float > (
    Param XParam,
    Loop < float > XLoop,
    BlockP < float > XBlock,
    TimeP < float > XTime
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



### function CalctimestepGPU&lt; double &gt; 

```C++
template __host__ double CalctimestepGPU< double > (
    Param XParam,
    Loop < double > XLoop,
    BlockP < double > XBlock,
    TimeP < double > XTime
) 
```




<hr>



### function CalctimestepGPU&lt; float &gt; 

```C++
template __host__ float CalctimestepGPU< float > (
    Param XParam,
    Loop < float > XLoop,
    BlockP < float > XBlock,
    TimeP < float > XTime
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



### function cleanupCPU&lt; double &gt; 

```C++
template __host__ void cleanupCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    EvolvingP < double > XEv_o
) 
```




<hr>



### function cleanupCPU&lt; float &gt; 

```C++
template __host__ void cleanupCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    EvolvingP < float > XEv_o
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



### function cleanupGPU&lt; double &gt; 

```C++
template __global__ void cleanupGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    EvolvingP < double > XEv_o
) 
```




<hr>



### function cleanupGPU&lt; float &gt; 

```C++
template __global__ void cleanupGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    EvolvingP < float > XEv_o
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



### function timestepreductionCPU 

```C++
template __host__ float timestepreductionCPU (
    Param XParam,
    Loop < float > XLoop,
    BlockP < float > XBlock,
    TimeP < float > XTime
) 
```




<hr>



### function timestepreductionCPU 

```C++
template __host__ double timestepreductionCPU (
    Param XParam,
    Loop < double > XLoop,
    BlockP < double > XBlock,
    TimeP < double > XTime
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



### function updateEVCPU&lt; double &gt; 

```C++
template __host__ void updateEVCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    FluxP < double > XFlux,
    AdvanceP < double > XAdv
) 
```




<hr>



### function updateEVCPU&lt; float &gt; 

```C++
template __host__ void updateEVCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    FluxP < float > XFlux,
    AdvanceP < float > XAdv
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



### function updateEVGPU&lt; double &gt; 

```C++
template __global__ void updateEVGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    FluxP < double > XFlux,
    AdvanceP < double > XAdv
) 
```




<hr>



### function updateEVGPU&lt; float &gt; 

```C++
template __global__ void updateEVGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    FluxP < float > XFlux,
    AdvanceP < float > XAdv
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Advection.cu`




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
|  \_\_host\_\_ void | [**AdvkernelCPU**](#function-advkernelcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_o) <br>_CPU routine for advancing the solution in time for each block and cell._  |
|  \_\_global\_\_ void | [**AdvkernelGPU**](#function-advkernelgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_o) <br>_GPU kernel for advancing the solution in time for each block and cell._  |
|  \_\_host\_\_ T | [**CalctimestepCPU**](#function-calctimestepcpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**TimeP**](structTimeP.md)&lt; T &gt; XTime) <br>_CPU routine to calculate the next time step for the simulation._  |
|  \_\_host\_\_ T | [**CalctimestepGPU**](#function-calctimestepgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**TimeP**](structTimeP.md)&lt; T &gt; XTime) <br>_GPU routine to calculate the next time step for the simulation._  |
|  \_\_host\_\_ void | [**cleanupCPU**](#function-cleanupcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_o) <br>_CPU routine to clean up evolving variables after advection step._  |
|  \_\_global\_\_ void | [**cleanupGPU**](#function-cleanupgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_o) <br>_GPU kernel to clean up evolving variables after advection step._  |
|  \_\_global\_\_ void | [**densify**](#function-densify) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* g\_idata, T \* g\_odata) <br>_GPU kernel to copy and densify data from block memory to output array._  |
|  \_\_global\_\_ void | [**reducemin3**](#function-reducemin3) (T \* g\_idata, T \* g\_odata, unsigned int n) <br>_GPU kernel to compute the minimum value in an array using parallel reduction._  |
|  \_\_host\_\_ T | [**timestepreductionCPU**](#function-timestepreductioncpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**TimeP**](structTimeP.md)&lt; T &gt; XTime) <br>_CPU routine to compute the minimum allowed time step across all blocks and cells._  |
|  \_\_host\_\_ void | [**updateEVCPU**](#function-updateevcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_CPU routine to update evolving variables (h, u, v, zs) for each block and cell._  |
|  \_\_global\_\_ void | [**updateEVGPU**](#function-updateevgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_GPU kernel to update evolving variables (h, u, v, zs) for each block and cell._  |




























## Public Functions Documentation




### function AdvkernelCPU 

_CPU routine for advancing the solution in time for each block and cell._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `dt` Time step 
* `zb` Bed elevation array 
* `XEv` Evolving variables 
* `XAdv` Advance variables 
* `XEv_o` Output evolving variables

Updates water height, velocity, and surface elevation for the next time step. 


        

<hr>



### function AdvkernelGPU 

_GPU kernel for advancing the solution in time for each block and cell._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `dt` Time step 
* `zb` Bed elevation array 
* `XEv` Evolving variables 
* `XAdv` Advance variables 
* `XEv_o` Output evolving variables

Updates water height, velocity, and surface elevation for the next time step. 


        

<hr>



### function CalctimestepCPU 

_CPU routine to calculate the next time step for the simulation._ 
```C++
template<class T>
__host__ T CalctimestepCPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    TimeP < T > XTime
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) control structure 
* `XBlock` Block data structure 
* `XTime` Time control structure 



**Returns:**

Computed time step 





        

<hr>



### function CalctimestepGPU 

_GPU routine to calculate the next time step for the simulation._ 
```C++
template<class T>
__host__ T CalctimestepGPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    TimeP < T > XTime
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) control structure 
* `XBlock` Block data structure 
* `XTime` Time control structure 



**Returns:**

Computed time step 





        

<hr>



### function cleanupCPU 

_CPU routine to clean up evolving variables after advection step._ 
```C++
template<class T>
__host__ void cleanupCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    EvolvingP < T > XEv_o
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables 
* `XEv_o` Output evolving variables 




        

<hr>



### function cleanupGPU 

_GPU kernel to clean up evolving variables after advection step._ 
```C++
template<class T>
__global__ void cleanupGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    EvolvingP < T > XEv_o
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables 
* `XEv_o` Output evolving variables 




        

<hr>



### function densify 

_GPU kernel to copy and densify data from block memory to output array._ 
```C++
template<class T>
__global__ void densify (
    Param XParam,
    BlockP < T > XBlock,
    T * g_idata,
    T * g_odata
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `g_idata` Input array 
* `g_odata` Output array 




        

<hr>



### function reducemin3 

_GPU kernel to compute the minimum value in an array using parallel reduction._ 
```C++
template<class T>
__global__ void reducemin3 (
    T * g_idata,
    T * g_odata,
    unsigned int n
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `g_idata` Input array 
* `g_odata` Output array (min per block) 
* `n` Number of elements 




        

<hr>



### function timestepreductionCPU 

_CPU routine to compute the minimum allowed time step across all blocks and cells._ 
```C++
template<class T>
__host__ T timestepreductionCPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    TimeP < T > XTime
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) control structure 
* `XBlock` Block data structure 
* `XTime` Time control structure 



**Returns:**

Minimum allowed time step 





        

<hr>



### function updateEVCPU 

_CPU routine to update evolving variables (h, u, v, zs) for each block and cell._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables 
* `XFlux` Flux variables 
* `XAdv` Advance variables

Computes new values for water height, velocity, and surface elevation using fluxes and advances. 


        

<hr>



### function updateEVGPU 

_GPU kernel to update evolving variables (h, u, v, zs) for each block and cell._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables 
* `XFlux` Flux variables 
* `XAdv` Advance variables

Computes new values for water height, velocity, and surface elevation using fluxes and advances. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Advection.h`


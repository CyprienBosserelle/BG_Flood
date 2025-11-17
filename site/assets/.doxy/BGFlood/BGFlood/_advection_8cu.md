

# File Advection.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Advection.cu**](_advection_8cu.md)

[Go to the source code of this file](_advection_8cu_source.md)



* `#include "Advection.h"`















## Classes

| Type | Name |
| ---: | :--- |
| struct | [**SharedMemory**](struct_shared_memory.md) &lt;class T&gt;<br> |
| struct | [**SharedMemory&lt; double &gt;**](struct_shared_memory_3_01double_01_4.md) &lt;&gt;<br> |






















## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**AdvkernelCPU**](#function-advkernelcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T dt, T \* zb, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**AdvanceP**](struct_advance_p.md)&lt; T &gt; XAdv, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv\_o) <br>_CPU routine for advancing the solution in time for each block and cell._  |
|  template \_\_host\_\_ void | [**AdvkernelCPU&lt; double &gt;**](#function-advkernelcpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double dt, double \* zb, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**AdvanceP**](struct_advance_p.md)&lt; double &gt; XAdv, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv\_o) <br> |
|  template \_\_host\_\_ void | [**AdvkernelCPU&lt; float &gt;**](#function-advkernelcpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float dt, float \* zb, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**AdvanceP**](struct_advance_p.md)&lt; float &gt; XAdv, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv\_o) <br> |
|  \_\_global\_\_ void | [**AdvkernelGPU**](#function-advkernelgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T dt, T \* zb, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**AdvanceP**](struct_advance_p.md)&lt; T &gt; XAdv, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv\_o) <br>_GPU kernel for advancing the solution in time for each block and cell._  |
|  template \_\_global\_\_ void | [**AdvkernelGPU&lt; double &gt;**](#function-advkernelgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double dt, double \* zb, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**AdvanceP**](struct_advance_p.md)&lt; double &gt; XAdv, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv\_o) <br> |
|  template \_\_global\_\_ void | [**AdvkernelGPU&lt; float &gt;**](#function-advkernelgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float dt, float \* zb, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**AdvanceP**](struct_advance_p.md)&lt; float &gt; XAdv, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv\_o) <br> |
|  \_\_host\_\_ T | [**CalctimestepCPU**](#function-calctimestepcpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**TimeP**](struct_time_p.md)&lt; T &gt; XTime) <br>_CPU routine to calculate the next time step for the simulation._  |
|  template \_\_host\_\_ double | [**CalctimestepCPU&lt; double &gt;**](#function-calctimestepcpu-double) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; double &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**TimeP**](struct_time_p.md)&lt; double &gt; XTime) <br> |
|  template \_\_host\_\_ float | [**CalctimestepCPU&lt; float &gt;**](#function-calctimestepcpu-float) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; float &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**TimeP**](struct_time_p.md)&lt; float &gt; XTime) <br> |
|  \_\_host\_\_ T | [**CalctimestepGPU**](#function-calctimestepgpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**TimeP**](struct_time_p.md)&lt; T &gt; XTime) <br>_GPU routine to calculate the next time step for the simulation._  |
|  template \_\_host\_\_ double | [**CalctimestepGPU&lt; double &gt;**](#function-calctimestepgpu-double) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; double &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**TimeP**](struct_time_p.md)&lt; double &gt; XTime) <br> |
|  template \_\_host\_\_ float | [**CalctimestepGPU&lt; float &gt;**](#function-calctimestepgpu-float) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; float &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**TimeP**](struct_time_p.md)&lt; float &gt; XTime) <br> |
|  \_\_host\_\_ void | [**cleanupCPU**](#function-cleanupcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv\_o) <br>_CPU routine to clean up evolving variables after advection step._  |
|  template \_\_host\_\_ void | [**cleanupCPU&lt; double &gt;**](#function-cleanupcpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv\_o) <br> |
|  template \_\_host\_\_ void | [**cleanupCPU&lt; float &gt;**](#function-cleanupcpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv\_o) <br> |
|  \_\_global\_\_ void | [**cleanupGPU**](#function-cleanupgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv\_o) <br>_GPU kernel to clean up evolving variables after advection step._  |
|  template \_\_global\_\_ void | [**cleanupGPU&lt; double &gt;**](#function-cleanupgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv\_o) <br> |
|  template \_\_global\_\_ void | [**cleanupGPU&lt; float &gt;**](#function-cleanupgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv\_o) <br> |
|  \_\_global\_\_ void | [**densify**](#function-densify) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* g\_idata, T \* g\_odata) <br>_GPU kernel to copy and densify data from block memory to output array._  |
|  \_\_global\_\_ void | [**reducemin3**](#function-reducemin3) (T \* g\_idata, T \* g\_odata, unsigned int n) <br>_GPU kernel to compute the minimum value in an array using parallel reduction._  |
|  \_\_host\_\_ T | [**timestepreductionCPU**](#function-timestepreductioncpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**TimeP**](struct_time_p.md)&lt; T &gt; XTime) <br>_CPU routine to compute the minimum allowed time step across all blocks and cells._  |
|  template \_\_host\_\_ float | [**timestepreductionCPU**](#function-timestepreductioncpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; float &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**TimeP**](struct_time_p.md)&lt; float &gt; XTime) <br> |
|  template \_\_host\_\_ double | [**timestepreductionCPU**](#function-timestepreductioncpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; double &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**TimeP**](struct_time_p.md)&lt; double &gt; XTime) <br> |
|  \_\_host\_\_ void | [**updateEVCPU**](#function-updateevcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, [**AdvanceP**](struct_advance_p.md)&lt; T &gt; XAdv) <br>_CPU routine to update evolving variables (h, u, v, zs) for each block and cell._  |
|  template \_\_host\_\_ void | [**updateEVCPU&lt; double &gt;**](#function-updateevcpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, [**AdvanceP**](struct_advance_p.md)&lt; double &gt; XAdv) <br> |
|  template \_\_host\_\_ void | [**updateEVCPU&lt; float &gt;**](#function-updateevcpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, [**AdvanceP**](struct_advance_p.md)&lt; float &gt; XAdv) <br> |
|  \_\_global\_\_ void | [**updateEVGPU**](#function-updateevgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, [**AdvanceP**](struct_advance_p.md)&lt; T &gt; XAdv) <br>_GPU kernel to update evolving variables (h, u, v, zs) for each block and cell._  |
|  template \_\_global\_\_ void | [**updateEVGPU&lt; double &gt;**](#function-updateevgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, [**AdvanceP**](struct_advance_p.md)&lt; double &gt; XAdv) <br> |
|  template \_\_global\_\_ void | [**updateEVGPU&lt; float &gt;**](#function-updateevgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, [**AdvanceP**](struct_advance_p.md)&lt; float &gt; XAdv) <br> |




























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


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `dt` Time step 
* `zb` Bed elevation array 
* `XEv` Evolving variables 
* `XAdv` Advance variables 
* `XEv_o` Output evolving variables

Updates water height, velocity, and surface elevation for the next time step. 


        

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


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `dt` Time step 
* `zb` Bed elevation array 
* `XEv` Evolving variables 
* `XAdv` Advance variables 
* `XEv_o` Output evolving variables

Updates water height, velocity, and surface elevation for the next time step. 


        

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


* `XParam` [**Model**](struct_model.md) parameters 
* `XLoop` [**Loop**](struct_loop.md) control structure 
* `XBlock` Block data structure 
* `XTime` Time control structure 



**Returns:**

Computed time step 





        

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


* `XParam` [**Model**](struct_model.md) parameters 
* `XLoop` [**Loop**](struct_loop.md) control structure 
* `XBlock` Block data structure 
* `XTime` Time control structure 



**Returns:**

Computed time step 





        

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


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables 
* `XEv_o` Output evolving variables 




        

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


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables 
* `XEv_o` Output evolving variables 




        

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


* `XParam` [**Model**](struct_model.md) parameters 
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


* `XParam` [**Model**](struct_model.md) parameters 
* `XLoop` [**Loop**](struct_loop.md) control structure 
* `XBlock` Block data structure 
* `XTime` Time control structure 



**Returns:**

Minimum allowed time step 





        

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


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables 
* `XFlux` Flux variables 
* `XAdv` Advance variables

Computes new values for water height, velocity, and surface elevation using fluxes and advances. 


        

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


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables 
* `XFlux` Flux variables 
* `XAdv` Advance variables

Computes new values for water height, velocity, and surface elevation using fluxes and advances. 


        

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


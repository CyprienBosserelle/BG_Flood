

# File FlowGPU.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**FlowGPU.cu**](FlowGPU_8cu.md)

[Go to the source code of this file](FlowGPU_8cu_source.md)



* `#include "FlowGPU.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**FlowGPU**](#function-flowgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  template void | [**FlowGPU&lt; double &gt;**](#function-flowgpu-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template void | [**FlowGPU&lt; float &gt;**](#function-flowgpu-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |
|  void | [**HalfStepGPU**](#function-halfstepgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  template void | [**HalfStepGPU&lt; double &gt;**](#function-halfstepgpu-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template void | [**HalfStepGPU&lt; float &gt;**](#function-halfstepgpu-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |
|  \_\_global\_\_ void | [**reset\_var**](#function-reset_var) (int halowidth, int \* active, T resetval, T \* Var) <br> |
|  template \_\_global\_\_ void | [**reset\_var&lt; double &gt;**](#function-reset_var-double) (int halowidth, int \* active, double resetval, double \* Var) <br> |
|  template \_\_global\_\_ void | [**reset\_var&lt; float &gt;**](#function-reset_var-float) (int halowidth, int \* active, float resetval, float \* Var) <br> |




























## Public Functions Documentation




### function FlowGPU 

```C++
template<class T>
void FlowGPU (
    Param XParam,
    Loop < T > & XLoop,
    Forcing < float > XForcing,
    Model < T > XModel
) 
```




<hr>



### function FlowGPU&lt; double &gt; 

```C++
template void FlowGPU< double > (
    Param XParam,
    Loop < double > & XLoop,
    Forcing < float > XForcing,
    Model < double > XModel
) 
```




<hr>



### function FlowGPU&lt; float &gt; 

```C++
template void FlowGPU< float > (
    Param XParam,
    Loop < float > & XLoop,
    Forcing < float > XForcing,
    Model < float > XModel
) 
```




<hr>



### function HalfStepGPU 

```C++
template<class T>
void HalfStepGPU (
    Param XParam,
    Loop < T > & XLoop,
    Forcing < float > XForcing,
    Model < T > XModel
) 
```




<hr>



### function HalfStepGPU&lt; double &gt; 

```C++
template void HalfStepGPU< double > (
    Param XParam,
    Loop < double > & XLoop,
    Forcing < float > XForcing,
    Model < double > XModel
) 
```




<hr>



### function HalfStepGPU&lt; float &gt; 

```C++
template void HalfStepGPU< float > (
    Param XParam,
    Loop < float > & XLoop,
    Forcing < float > XForcing,
    Model < float > XModel
) 
```




<hr>



### function reset\_var 

```C++
template<class T>
__global__ void reset_var (
    int halowidth,
    int * active,
    T resetval,
    T * Var
) 
```




<hr>



### function reset\_var&lt; double &gt; 

```C++
template __global__ void reset_var< double > (
    int halowidth,
    int * active,
    double resetval,
    double * Var
) 
```




<hr>



### function reset\_var&lt; float &gt; 

```C++
template __global__ void reset_var< float > (
    int halowidth,
    int * active,
    float resetval,
    float * Var
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/FlowGPU.cu`


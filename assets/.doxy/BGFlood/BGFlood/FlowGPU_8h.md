

# File FlowGPU.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**FlowGPU.h**](FlowGPU_8h.md)

[Go to the source code of this file](FlowGPU_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "Util_CPU.h"`
* `#include "MemManagement.h"`
* `#include "Gradients.h"`
* `#include "Kurganov.h"`
* `#include "Advection.h"`
* `#include "Friction.h"`
* `#include "Updateforcing.h"`
* `#include "Reimann.h"`
* `#include "Boundary.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**FlowGPU**](#function-flowgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  void | [**HalfStepGPU**](#function-halfstepgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  \_\_global\_\_ void | [**reset\_var**](#function-reset_var) (int halowidth, int \* active, T resetval, T \* Var) <br> |




























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

------------------------------
The documentation for this class was generated from the following file `src/FlowGPU.h`


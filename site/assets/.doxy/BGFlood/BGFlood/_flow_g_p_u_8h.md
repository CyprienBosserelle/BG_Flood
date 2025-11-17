

# File FlowGPU.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**FlowGPU.h**](_flow_g_p_u_8h.md)

[Go to the source code of this file](_flow_g_p_u_8h_source.md)



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
|  void | [**FlowGPU**](#function-flowgpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; & XLoop, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Main GPU flow solver for the flood model._  |
|  void | [**HalfStepGPU**](#function-halfstepgpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; & XLoop, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Debugging GPU flow step for the flood model._  |
|  \_\_global\_\_ void | [**reset\_var**](#function-reset_var) (int halowidth, int \* active, T resetval, T \* Var) <br>_CUDA kernel to reset a variable array for all active blocks._  |




























## Public Functions Documentation




### function FlowGPU 

_Main GPU flow solver for the flood model._ 
```C++
template<class T>
void FlowGPU (
    Param XParam,
    Loop < T > & XLoop,
    Forcing < float > XForcing,
    Model < T > XModel
) 
```



Executes predictor and corrector steps, applies atmospheric, wind, and river forcing, updates advection and friction terms, and manages halo and gradient reconstruction for all blocks using CUDA kernels and streams.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XLoop` [**Loop**](struct_loop.md) control and time stepping 
* `XForcing` [**Forcing**](struct_forcing.md) data (atmospheric, wind, river, rain) 
* `XModel` [**Model**](struct_model.md) data structure 




        

<hr>



### function HalfStepGPU 

_Debugging GPU flow step for the flood model._ 
```C++
template<class T>
void HalfStepGPU (
    Param XParam,
    Loop < T > & XLoop,
    Forcing < float > XForcing,
    Model < T > XModel
) 
```



Runs a simplified flow step for debugging the main engine, including forcing, advection, friction, and halo/gradient reconstruction using CUDA kernels and streams.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XLoop` [**Loop**](struct_loop.md) control and time stepping 
* `XForcing` [**Forcing**](struct_forcing.md) data (atmospheric, wind, river, rain) 
* `XModel` [**Model**](struct_model.md) data structure 




        

<hr>



### function reset\_var 

_CUDA kernel to reset a variable array for all active blocks._ 
```C++
template<class T>
__global__ void reset_var (
    int halowidth,
    int * active,
    T resetval,
    T * Var
) 
```



Sets all values in the variable array to the specified reset value for each block and cell.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `halowidth` Halo width 
* `active` Array of active block indices 
* `resetval` Value to set 
* `Var` Variable array to reset 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/FlowGPU.h`


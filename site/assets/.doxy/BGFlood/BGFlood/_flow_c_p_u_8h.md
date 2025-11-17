

# File FlowCPU.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**FlowCPU.h**](_flow_c_p_u_8h.md)

[Go to the source code of this file](_flow_c_p_u_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "Util_CPU.h"`
* `#include "MemManagement.h"`
* `#include "Halo.h"`
* `#include "GridManip.h"`
* `#include "Gradients.h"`
* `#include "Kurganov.h"`
* `#include "Advection.h"`
* `#include "Friction.h"`
* `#include "Updateforcing.h"`
* `#include "Reimann.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**FlowCPU**](#function-flowcpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; & XLoop, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Main CPU flow solver for the flood model._  |
|  void | [**HalfStepCPU**](#function-halfstepcpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; & XLoop, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Debugging flow step for the flood model._  |




























## Public Functions Documentation




### function FlowCPU 

_Main CPU flow solver for the flood model._ 
```C++
template<class T>
void FlowCPU (
    Param XParam,
    Loop < T > & XLoop,
    Forcing < float > XForcing,
    Model < T > XModel
) 
```



Executes predictor and corrector steps, applies atmospheric, wind, and river forcing, updates advection and friction terms, and manages halo and gradient reconstruction for all blocks.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XLoop` [**Loop**](struct_loop.md) control and time stepping 
* `XForcing` [**Forcing**](struct_forcing.md) data (atmospheric, wind, river, rain) 
* `XModel` [**Model**](struct_model.md) data structure 




        

<hr>



### function HalfStepCPU 

_Debugging flow step for the flood model._ 
```C++
template<class T>
void HalfStepCPU (
    Param XParam,
    Loop < T > & XLoop,
    Forcing < float > XForcing,
    Model < T > XModel
) 
```



Runs a simplified flow step for debugging the main engine, including forcing, advection, friction, and halo/gradient reconstruction.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XLoop` [**Loop**](struct_loop.md) control and time stepping 
* `XForcing` [**Forcing**](struct_forcing.md) data (atmospheric, wind, river, rain) 
* `XModel` [**Model**](struct_model.md) data structure

Debugging flow step This function was created to debug the main engine of the model 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/FlowCPU.h`


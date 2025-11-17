

# File Mainloop.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Mainloop.h**](_mainloop_8h.md)

[Go to the source code of this file](_mainloop_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "Mesh.h"`
* `#include "Write_netcdf.h"`
* `#include "InitialConditions.h"`
* `#include "MemManagement.h"`
* `#include "Boundary.h"`
* `#include "FlowGPU.h"`
* `#include "FlowCPU.h"`
* `#include "Meanmax.h"`
* `#include "Updateforcing.h"`
* `#include "FlowMLGPU.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**DebugLoop**](#function-debugloop) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel, [**Model**](struct_model.md)&lt; T &gt; & XModel\_g) <br>_Debugging loop for the flood model._  |
|  [**Loop**](struct_loop.md)&lt; T &gt; | [**InitLoop**](#function-initloop) ([**Param**](class_param.md) & XParam, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Initialize the simulation loop structure._  |
|  void | [**MainLoop**](#function-mainloop) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel, [**Model**](struct_model.md)&lt; T &gt; & XModel\_g) <br>_Main simulation loop for the flood model._  |
|  \_\_host\_\_ double | [**initdt**](#function-initdt) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; XLoop, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Initialize the simulation time step._  |
|  void | [**printstatus**](#function-printstatus) (T totaltime, T dt) <br>_Print the current simulation time and time step to the console._  |
|  \_\_global\_\_ void | [**storeTSout**](#function-storetsout) ([**Param**](class_param.md) XParam, int noutnodes, int outnode, int istep, int blknode, int inode, int jnode, int \* blkTS, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* store) <br>_CUDA kernel to store time series output for specified nodes._  |




























## Public Functions Documentation




### function DebugLoop 

_Debugging loop for the flood model._ 
```C++
template<class T>
void DebugLoop (
    Param & XParam,
    Forcing < float > XForcing,
    Model < T > & XModel,
    Model < T > & XModel_g
) 
```



Used to debug and wrap the debug flow engine. Runs a fixed number of steps and outputs diagnostic information. Handles both CPU and GPU execution paths.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data (float) 
* `XModel` [**Model**](struct_model.md) state (CPU) 
* `XModel_g` [**Model**](struct_model.md) state (GPU) 




        

<hr>



### function InitLoop 

_Initialize the simulation loop structure._ 
```C++
template<class T>
Loop < T > InitLoop (
    Param & XParam,
    Model < T > & XModel
) 
```



Sets up loop control variables, output buffers, and initial time step.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) state 



**Returns:**

Initialized loop control structure 





        

<hr>



### function MainLoop 

_Main simulation loop for the flood model._ 
```C++
template<class T>
void MainLoop (
    Param & XParam,
    Forcing < float > XForcing,
    Model < T > & XModel,
    Model < T > & XModel_g
) 
```



Advances the simulation in time, applying boundary conditions, forcing, core engine, output, and crash detection. Handles both CPU and GPU execution paths.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data (float) 
* `XModel` [**Model**](struct_model.md) state (CPU) 
* `XModel_g` [**Model**](struct_model.md) state (GPU) 




        

<hr>



### function initdt 

_Initialize the simulation time step._ 
```C++
template<class T>
__host__ double initdt (
    Param XParam,
    Loop < T > XLoop,
    Model < T > XModel
) 
```



Calculates the initial time step based on user input or model parameters. Uses either a user-specified value or computes a safe initial value based on water depth and cell resolution.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XLoop` [**Loop**](struct_loop.md) control structure 
* `XModel` [**Model**](struct_model.md) state 



**Returns:**

Initial time step value 





        

<hr>



### function printstatus 

_Print the current simulation time and time step to the console._ 
```C++
template<class T>
void printstatus (
    T totaltime,
    T dt
) 
```



Displays the total simulation time and current time step in a formatted manner.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `totaltime` Current simulation time 
* `dt` Current time step 




        

<hr>



### function storeTSout 

_CUDA kernel to store time series output for specified nodes._ 
```C++
template<class T>
__global__ void storeTSout (
    Param XParam,
    int noutnodes,
    int outnode,
    int istep,
    int blknode,
    int inode,
    int jnode,
    int * blkTS,
    EvolvingP < T > XEv,
    T * store
) 
```



Writes evolving variables for selected nodes and time steps to output storage array.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `noutnodes` Number of output nodes 
* `outnode` Output node index 
* `istep` Time step index 
* `blknode` Block index 
* `inode` Node i-index 
* `jnode` Node j-index 
* `blkTS` Block time series mapping 
* `XEv` Evolving state variables 
* `store` Output storage array 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Mainloop.h`


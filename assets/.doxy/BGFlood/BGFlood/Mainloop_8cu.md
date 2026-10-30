

# File Mainloop.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Mainloop.cu**](Mainloop_8cu.md)

[Go to the source code of this file](Mainloop_8cu_source.md)



* `#include "Mainloop.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**CalcInitdtCPU**](#function-calcinitdtcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv, T \* dtmax) <br>_Calculate initial time step values on the CPU for all blocks and nodes._  |
|  \_\_global\_\_ void | [**CalcInitdtGPU**](#function-calcinitdtgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv, T \* dtmax) <br>_CUDA kernel to calculate initial time step values on the GPU for all blocks and nodes._  |
|  void | [**CrashDetection**](#function-crashdetection) ([**Param**](classParam.md) & XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br>_Detects simulation crash due to small time steps and generates a crash report._  |
|  void | [**DebugLoop**](#function-debugloop) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel, [**Model**](structModel.md)&lt; T &gt; & XModel\_g) <br>_Debugging loop for the flood model._  |
|  template void | [**DebugLoop&lt; double &gt;**](#function-debugloop-double) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; double &gt; & XModel, [**Model**](structModel.md)&lt; double &gt; & XModel\_g) <br> |
|  template void | [**DebugLoop&lt; float &gt;**](#function-debugloop-float) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; float &gt; & XModel, [**Model**](structModel.md)&lt; float &gt; & XModel\_g) <br> |
|  [**Loop**](structLoop.md)&lt; T &gt; | [**InitLoop**](#function-initloop) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br>_Initialize the simulation loop structure._  |
|  void | [**MainLoop**](#function-mainloop) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel, [**Model**](structModel.md)&lt; T &gt; & XModel\_g) <br>_Main simulation loop for the flood model._  |
|  template void | [**MainLoop&lt; double &gt;**](#function-mainloop-double) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; double &gt; & XModel, [**Model**](structModel.md)&lt; double &gt; & XModel\_g) <br> |
|  template void | [**MainLoop&lt; float &gt;**](#function-mainloop-float) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; float &gt; & XModel, [**Model**](structModel.md)&lt; float &gt; & XModel\_g) <br> |
|  \_\_host\_\_ double | [**initdt**](#function-initdt) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Model**](structModel.md)&lt; T &gt; XModel) <br>_Initialize the simulation time step._  |
|  template \_\_host\_\_ double | [**initdt&lt; double &gt;**](#function-initdt-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template \_\_host\_\_ double | [**initdt&lt; float &gt;**](#function-initdt-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |
|  void | [**mapoutput**](#function-mapoutput) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Model**](structModel.md)&lt; T &gt; & XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br>_Output map data at specified simulation times._  |
|  void | [**pointoutputstep**](#function-pointoutputstep) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br>_Output time series data for specified nodes._  |
|  void | [**printstatus**](#function-printstatus) (T totaltime, T dt) <br>_Print the current simulation time and time step to the console._  |
|  \_\_global\_\_ void | [**storeTSout**](#function-storetsout) ([**Param**](classParam.md) XParam, int noutnodes, int outnode, int istep, int blknode, int inode, int jnode, int \* blkTS, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* store) <br>_CUDA kernel to store time series output for specified nodes._  |
|  void | [**updateBnd**](#function-updatebnd) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br>_Update boundary conditions for the simulation._  |




























## Public Functions Documentation




### function CalcInitdtCPU 

_Calculate initial time step values on the CPU for all blocks and nodes._ 
```C++
template<class T>
__host__ void CalcInitdtCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEvolv,
    T * dtmax
) 
```



Computes the maximum allowable time step for each cell based on local water depth and cell resolution.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data 
* `XEvolv` Evolving state variables 
* `dtmax` Output array for maximum time step per cell 




        

<hr>



### function CalcInitdtGPU 

_CUDA kernel to calculate initial time step values on the GPU for all blocks and nodes._ 
```C++
template<class T>
__global__ void CalcInitdtGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEvolv,
    T * dtmax
) 
```



Computes the maximum allowable time step for each cell using GPU parallelism.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data 
* `XEvolv` Evolving state variables 
* `dtmax` Output array for maximum time step per cell 




        

<hr>



### function CrashDetection 

_Detects simulation crash due to small time steps and generates a crash report._ 
```C++
template<class T>
void CrashDetection (
    Param & XParam,
    Loop < T > XLoop,
    Model < T > XModel,
    Model < T > XModel_g
) 
```



If the time step falls below the minimum allowed value before the simulation end time, stops the model and writes output variables to a crash report file. Handles both CPU and GPU data transfer for output variables.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters (may be modified) 
* `XLoop` [**Loop**](structLoop.md) control structure 
* `XModel` [**Model**](structModel.md) state (CPU) 
* `XModel_g` [**Model**](structModel.md) state (GPU) 




        

<hr>



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


* `XParam` [**Model**](structModel.md) parameters 
* `XForcing` [**Forcing**](structForcing.md) data (float) 
* `XModel` [**Model**](structModel.md) state (CPU) 
* `XModel_g` [**Model**](structModel.md) state (GPU) 




        

<hr>



### function DebugLoop&lt; double &gt; 

```C++
template void DebugLoop< double > (
    Param & XParam,
    Forcing < float > XForcing,
    Model < double > & XModel,
    Model < double > & XModel_g
) 
```




<hr>



### function DebugLoop&lt; float &gt; 

```C++
template void DebugLoop< float > (
    Param & XParam,
    Forcing < float > XForcing,
    Model < float > & XModel,
    Model < float > & XModel_g
) 
```




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


* `XParam` [**Model**](structModel.md) parameters 
* `XModel` [**Model**](structModel.md) state 



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


* `XParam` [**Model**](structModel.md) parameters 
* `XForcing` [**Forcing**](structForcing.md) data (float) 
* `XModel` [**Model**](structModel.md) state (CPU) 
* `XModel_g` [**Model**](structModel.md) state (GPU) 




        

<hr>



### function MainLoop&lt; double &gt; 

```C++
template void MainLoop< double > (
    Param & XParam,
    Forcing < float > XForcing,
    Model < double > & XModel,
    Model < double > & XModel_g
) 
```




<hr>



### function MainLoop&lt; float &gt; 

```C++
template void MainLoop< float > (
    Param & XParam,
    Forcing < float > XForcing,
    Model < float > & XModel,
    Model < float > & XModel_g
) 
```




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


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) control structure 
* `XModel` [**Model**](structModel.md) state 



**Returns:**

Initial time step value 





        

<hr>



### function initdt&lt; double &gt; 

```C++
template __host__ double initdt< double > (
    Param XParam,
    Loop < double > XLoop,
    Model < double > XModel
) 
```




<hr>



### function initdt&lt; float &gt; 

```C++
template __host__ double initdt< float > (
    Param XParam,
    Loop < float > XLoop,
    Model < float > XModel
) 
```




<hr>



### function mapoutput 

_Output map data at specified simulation times._ 
```C++
template<class T>
void mapoutput (
    Param XParam,
    Loop < T > & XLoop,
    Model < T > & XModel,
    Model < T > XModel_g
) 
```



Saves model state to NetCDF files when output times are reached, handling both CPU and GPU data transfer.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) control structure 
* `XModel` [**Model**](structModel.md) state (CPU) 
* `XModel_g` [**Model**](structModel.md) state (GPU) 




        

<hr>



### function pointoutputstep 

_Output time series data for specified nodes._ 
```C++
template<class T>
void pointoutputstep (
    Param XParam,
    Loop < T > & XLoop,
    Model < T > XModel,
    Model < T > XModel_g
) 
```



Collects and writes time series output for selected nodes, handling both CPU and GPU data paths and buffer management.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) control structure 
* `XModel` [**Model**](structModel.md) state (CPU) 
* `XModel_g` [**Model**](structModel.md) state (GPU) 




        

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


* `XParam` [**Model**](structModel.md) parameters 
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



### function updateBnd 

_Update boundary conditions for the simulation._ 
```C++
template<class T>
void updateBnd (
    Param XParam,
    Loop < T > XLoop,
    Forcing < float > XForcing,
    Model < T > XModel,
    Model < T > XModel_g
) 
```



Applies boundary flows for each segment, handling both CPU and GPU execution paths.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) control structure 
* `XForcing` [**Forcing**](structForcing.md) data (float) 
* `XModel` [**Model**](structModel.md) state (CPU) 
* `XModel_g` [**Model**](structModel.md) state (GPU) 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Mainloop.cu`


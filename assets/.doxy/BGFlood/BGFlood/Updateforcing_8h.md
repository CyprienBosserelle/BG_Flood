

# File Updateforcing.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Updateforcing.h**](Updateforcing_8h.md)

[Go to the source code of this file](Updateforcing_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "InitialConditions.h"`
* `#include "MemManagement.h"`
* `#include "ReadForcing.h"`
* `#include "GridManip.h"`
* `#include "Util_CPU.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_global\_\_ void | [**AddDeformGPU**](#function-adddeformgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**deformmap**](classdeformmap.md)&lt; float &gt; defmap, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T scale, T \* zb) <br>_Perform a deformation step on the model on the GPU. Applies deformation maps to the model based on the current simulation time and deformation parameters._  |
|  \_\_host\_\_ void | [**AddPatmforcingCPU**](#function-addpatmforcingcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; PAtm, [**Model**](structModel.md)&lt; T &gt; XModel) <br>_Add atmospheric pressure forcing to the model on the CPU. Adds atmospheric pressure forcing to the model based on pressure data and current simulation time._  |
|  \_\_global\_\_ void | [**AddPatmforcingGPU**](#function-addpatmforcinggpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; PAtm, [**Model**](structModel.md)&lt; T &gt; XModel) <br>_Add atmospheric pressure forcing to the model on the GPU. Adds atmospheric pressure forcing to the model based on pressure data and current simulation time._  |
|  \_\_host\_\_ void | [**AddRiverForcing**](#function-addriverforcing) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, std::vector&lt; [**River**](classRiver.md) &gt; XRivers, [**Model**](structModel.md)&lt; T &gt; XModel) <br>_Add river forcing to the model. Adds river discharge forcing to the model based on river data and current simulation time._  |
|  \_\_host\_\_ void | [**AddinfiltrationImplicitCPU**](#function-addinfiltrationimplicitcpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* il, T \* cl, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* hgw) <br>_Add infiltration forcing to the model implicitly on the CPU. Adds infiltration forcing to the model based on infiltration data and current water depth, updating water depth and surface elevation._  |
|  \_\_global\_\_ void | [**AddinfiltrationImplicitGPU**](#function-addinfiltrationimplicitgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* il, T \* cl, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* hgw) <br>_Add infiltration forcing to the model implicitly on the GPU. Adds infiltration forcing to the model based on infiltration data and current water depth, updating water depth and surface elevation._  |
|  \_\_host\_\_ void | [**AddrainforcingCPU**](#function-addrainforcingcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_Add rainfall forcing to the model on the CPU. Adds rainfall forcing to the model based on rainfall data and current simulation time._  |
|  \_\_global\_\_ void | [**AddrainforcingGPU**](#function-addrainforcinggpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_Add rainfall forcing to the model on the GPU. Adds rainfall forcing to the model based on rainfall data and current simulation time._  |
|  \_\_host\_\_ void | [**AddrainforcingImplicitCPU**](#function-addrainforcingimplicitcpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv) <br>_Add rainfall forcing to the model implicitly on the CPU. Adds rainfall forcing to the model based on rainfall data and current simulation time, updating water depth and surface elevation._  |
|  \_\_global\_\_ void | [**AddrainforcingImplicitGPU**](#function-addrainforcingimplicitgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv) <br>_Add rainfall forcing to the model implicitly on the GPU. Adds rainfall forcing to the model based on rainfall data and current simulation time, updating water depth and surface elevation._  |
|  \_\_host\_\_ void | [**AddwindforcingCPU**](#function-addwindforcingcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Uwind, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Vwind, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_Add wind forcing to the model on the CPU. Adds wind forcing to the model based on wind data and current simulation time._  |
|  \_\_global\_\_ void | [**AddwindforcingGPU**](#function-addwindforcinggpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Uwind, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Vwind, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_Add wind forcing to the model on the GPU. Adds wind forcing to the model based on wind data and current simulation time._  |
|  void | [**Forcingthisstep**](#function-forcingthisstep) ([**Param**](classParam.md) XParam, double totaltime, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; & XDynForcing) <br>_Update dynamic forcing for the current simulation step._  |
|  \_\_global\_\_ void | [**InjectRiverGPU**](#function-injectrivergpu) ([**Param**](classParam.md) XParam, [**River**](classRiver.md) XRiver, T qnow, int \* Riverblks, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_Inject river discharge into the model grid on the GPU. Injects river discharge into the model grid based on river geometry and discharge rate._  |
|  void | [**deformstep**](#function-deformstep) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, std::vector&lt; [**deformmap**](classdeformmap.md)&lt; float &gt; &gt; deform, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br>_Perform a deformation step on the model. Applies deformation maps to the model based on the current simulation time and deformation parameters. Overloaded function to handle both CPU and GPU models._  |
|  \_\_device\_\_ T | [**interpDyn2BUQ**](#function-interpdyn2buq) (T x, T y, [**TexSetP**](structTexSetP.md) Forcing) <br>_Interpolate dynamic forcing data at given coordinates on the GPU. Interpolates dynamic forcing data at specified coordinates using bilinear interpolation._  |
|  void | [**updateforcing**](#function-updateforcing) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing) <br>_Update dynamic forcings for the current simulation step. Updates the dynamic forcing data for the current simulation time step._  |




























## Public Functions Documentation




### function AddDeformGPU 

_Perform a deformation step on the model on the GPU. Applies deformation maps to the model based on the current simulation time and deformation parameters._ 
```C++
template<class T>
__global__ void AddDeformGPU (
    Param XParam,
    BlockP < T > XBlock,
    deformmap < float > defmap,
    EvolvingP < T > XEv,
    T scale,
    T * zb
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) structure containing time information 
* `deform` Vector of deformation maps 
* `XModel` [**Model**](structModel.md) data structure for GPU 




        

<hr>



### function AddPatmforcingCPU 

_Add atmospheric pressure forcing to the model on the CPU. Adds atmospheric pressure forcing to the model based on pressure data and current simulation time._ 
```C++
template<class T>
__host__ void AddPatmforcingCPU (
    Param XParam,
    BlockP < T > XBlock,
    DynForcingP < float > PAtm,
    Model < T > XModel
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `PAtm` Atmospheric pressure dynamic forcing structure 
* `XModel` [**Model**](structModel.md) data structure 




        

<hr>



### function AddPatmforcingGPU 

_Add atmospheric pressure forcing to the model on the GPU. Adds atmospheric pressure forcing to the model based on pressure data and current simulation time._ 
```C++
template<class T>
__global__ void AddPatmforcingGPU (
    Param XParam,
    BlockP < T > XBlock,
    DynForcingP < float > PAtm,
    Model < T > XModel
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `PAtm` Atmospheric pressure dynamic forcing structure 
* `XModel` [**Model**](structModel.md) data structure 




        

<hr>



### function AddRiverForcing 

_Add river forcing to the model. Adds river discharge forcing to the model based on river data and current simulation time._ 
```C++
template<class T>
__host__ void AddRiverForcing (
    Param XParam,
    Loop < T > XLoop,
    std::vector< River > XRivers,
    Model < T > XModel
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) structure containing time information 
* `XRivers` Vector of river data structures 
* `XModel` [**Model**](structModel.md) data structure 




        

<hr>



### function AddinfiltrationImplicitCPU 

_Add infiltration forcing to the model implicitly on the CPU. Adds infiltration forcing to the model based on infiltration data and current water depth, updating water depth and surface elevation._ 
```C++
template<class T>
__host__ void AddinfiltrationImplicitCPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    T * il,
    T * cl,
    EvolvingP < T > XEv,
    T * hgw
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) structure containing time information 
* `XBlock` Block data structure 
* `il` Initial infiltration rates array 
* `cl` Continuous infiltration rates array 
* `XEv` Evolving data structure 
* `hgw` Groundwater height array 




        

<hr>



### function AddinfiltrationImplicitGPU 

_Add infiltration forcing to the model implicitly on the GPU. Adds infiltration forcing to the model based on infiltration data and current water depth, updating water depth and surface elevation._ 
```C++
template<class T>
__global__ void AddinfiltrationImplicitGPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    T * il,
    T * cl,
    EvolvingP < T > XEv,
    T * hgw
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) structure containing time information 
* `XBlock` Block data structure 
* `il` Initial infiltration rates array 
* `cl` Continuous infiltration rates array 
* `XEv` Evolving data structure 
* `hgw` Groundwater height array 




        

<hr>



### function AddrainforcingCPU 

_Add rainfall forcing to the model on the CPU. Adds rainfall forcing to the model based on rainfall data and current simulation time._ 
```C++
template<class T>
__host__ void AddrainforcingCPU (
    Param XParam,
    BlockP < T > XBlock,
    DynForcingP < float > Rain,
    AdvanceP < T > XAdv
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `Rain` Rainfall dynamic forcing structure 
* `XAdv` Advance data structure 




        

<hr>



### function AddrainforcingGPU 

_Add rainfall forcing to the model on the GPU. Adds rainfall forcing to the model based on rainfall data and current simulation time._ 
```C++
template<class T>
__global__ void AddrainforcingGPU (
    Param XParam,
    BlockP < T > XBlock,
    DynForcingP < float > Rain,
    AdvanceP < T > XAdv
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `Rain` Rainfall dynamic forcing structure 
* `XAdv` Advance data structure 




        

<hr>



### function AddrainforcingImplicitCPU 

_Add rainfall forcing to the model implicitly on the CPU. Adds rainfall forcing to the model based on rainfall data and current simulation time, updating water depth and surface elevation._ 
```C++
template<class T>
__host__ void AddrainforcingImplicitCPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    DynForcingP < float > Rain,
    EvolvingP < T > XEv
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) structure containing time information 
* `XBlock` Block data structure 
* `Rain` Rainfall dynamic forcing structure 
* `XEv` Evolving data structure 




        

<hr>



### function AddrainforcingImplicitGPU 

_Add rainfall forcing to the model implicitly on the GPU. Adds rainfall forcing to the model based on rainfall data and current simulation time, updating water depth and surface elevation._ 
```C++
template<class T>
__global__ void AddrainforcingImplicitGPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    DynForcingP < float > Rain,
    EvolvingP < T > XEv
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) structure containing time information 
* `XBlock` Block data structure 
* `Rain` Rainfall dynamic forcing structure 
* `XEv` Evolving data structure 




        

<hr>



### function AddwindforcingCPU 

_Add wind forcing to the model on the CPU. Adds wind forcing to the model based on wind data and current simulation time._ 
```C++
template<class T>
__host__ void AddwindforcingCPU (
    Param XParam,
    BlockP < T > XBlock,
    DynForcingP < float > Uwind,
    DynForcingP < float > Vwind,
    AdvanceP < T > XAdv
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `Uwind` U-component of wind dynamic forcing structure 
* `Vwind` V-component of wind dynamic forcing structure 
* `XAdv` Advance data structure 




        

<hr>



### function AddwindforcingGPU 

_Add wind forcing to the model on the GPU. Adds wind forcing to the model based on wind data and current simulation time._ 
```C++
template<class T>
__global__ void AddwindforcingGPU (
    Param XParam,
    BlockP < T > XBlock,
    DynForcingP < float > Uwind,
    DynForcingP < float > Vwind,
    AdvanceP < T > XAdv
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `Uwind` U-component of wind dynamic forcing structure 
* `Vwind` V-component of wind dynamic forcing structure 
* `XAdv` Advance data structure 




        

<hr>



### function Forcingthisstep 

_Update dynamic forcing for the current simulation step._ 
```C++
void Forcingthisstep (
    Param XParam,
    double totaltime,
    DynForcingP < float > & XDynForcing
) 
```



Updates the dynamic forcing data for the current simulation time step, handling uniform and non-uniform cases.




**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `totaltime` Current simulation time 
* `XDynForcing` Dynamic forcing structure to update 




        

<hr>



### function InjectRiverGPU 

_Inject river discharge into the model grid on the GPU. Injects river discharge into the model grid based on river geometry and discharge rate._ 
```C++
template<class T>
__global__ void InjectRiverGPU (
    Param XParam,
    River XRiver,
    T qnow,
    int * Riverblks,
    BlockP < T > XBlock,
    AdvanceP < T > XAdv
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XRiver` [**River**](classRiver.md) data structure 
* `qnow` Current river discharge rate 
* `Riverblks` Array of blocks affected by the river 
* `XBlock` Block data structure 
* `XAdv` Advance data structure 




        

<hr>



### function deformstep 

_Perform a deformation step on the model. Applies deformation maps to the model based on the current simulation time and deformation parameters. Overloaded function to handle both CPU and GPU models._ 
```C++
template<class T>
void deformstep (
    Param XParam,
    Loop < T > XLoop,
    std::vector< deformmap < float > > deform,
    Model < T > XModel,
    Model < T > XModel_g
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) structure containing time information 
* `deform` Vector of deformation maps 
* `XModel` [**Model**](structModel.md) data structure 
* `XModel_g` [**Model**](structModel.md) data structure for GPU 




        

<hr>



### function interpDyn2BUQ 

_Interpolate dynamic forcing data at given coordinates on the GPU. Interpolates dynamic forcing data at specified coordinates using bilinear interpolation._ 
```C++
template<class T>
__device__ T interpDyn2BUQ (
    T x,
    T y,
    TexSetP Forcing
) 
```





**Parameters:**


* `x` X-coordinate 
* `y` Y-coordinate 
* [**Forcing**](structForcing.md) Dynamic forcing data structure 




        

<hr>



### function updateforcing 

_Update dynamic forcings for the current simulation step. Updates the dynamic forcing data for the current simulation time step._ 
```C++
template<class T>
void updateforcing (
    Param XParam,
    Loop < T > XLoop,
    Forcing < float > & XForcing
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) structure containing time information 
* `XForcing` [**Forcing**](structForcing.md) data structure to update 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Updateforcing.h`


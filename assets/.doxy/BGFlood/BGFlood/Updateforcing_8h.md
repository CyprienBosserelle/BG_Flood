

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
|  \_\_global\_\_ void | [**AddDeformGPU**](#function-adddeformgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**deformmap**](classdeformmap.md)&lt; float &gt; defmap, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T scale, T \* zb) <br> |
|  \_\_host\_\_ void | [**AddPatmforcingCPU**](#function-addpatmforcingcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; PAtm, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  \_\_global\_\_ void | [**AddPatmforcingGPU**](#function-addpatmforcinggpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; PAtm, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  \_\_host\_\_ void | [**AddRiverForcing**](#function-addriverforcing) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, std::vector&lt; [**River**](classRiver.md) &gt; XRivers, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  \_\_host\_\_ void | [**AddinfiltrationImplicitCPU**](#function-addinfiltrationimplicitcpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* il, T \* cl, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* hgw) <br> |
|  \_\_global\_\_ void | [**AddinfiltrationImplicitGPU**](#function-addinfiltrationimplicitgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* il, T \* cl, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* hgw) <br> |
|  \_\_host\_\_ void | [**AddrainforcingCPU**](#function-addrainforcingcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br> |
|  \_\_global\_\_ void | [**AddrainforcingGPU**](#function-addrainforcinggpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br> |
|  \_\_host\_\_ void | [**AddrainforcingImplicitCPU**](#function-addrainforcingimplicitcpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv) <br> |
|  \_\_global\_\_ void | [**AddrainforcingImplicitGPU**](#function-addrainforcingimplicitgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv) <br> |
|  \_\_host\_\_ void | [**AddwindforcingCPU**](#function-addwindforcingcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Uwind, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Vwind, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br> |
|  \_\_global\_\_ void | [**AddwindforcingGPU**](#function-addwindforcinggpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Uwind, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Vwind, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br> |
|  void | [**Forcingthisstep**](#function-forcingthisstep) ([**Param**](classParam.md) XParam, double totaltime, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; & XDynForcing) <br> |
|  \_\_global\_\_ void | [**InjectRiverGPU**](#function-injectrivergpu) ([**Param**](classParam.md) XParam, [**River**](classRiver.md) XRiver, T qnow, int \* Riverblks, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br> |
|  void | [**deformstep**](#function-deformstep) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, std::vector&lt; [**deformmap**](classdeformmap.md)&lt; float &gt; &gt; deform, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  \_\_device\_\_ T | [**interpDyn2BUQ**](#function-interpdyn2buq) (T x, T y, [**TexSetP**](structTexSetP.md) Forcing) <br> |
|  void | [**updateforcing**](#function-updateforcing) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing) <br> |




























## Public Functions Documentation




### function AddDeformGPU 

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




<hr>



### function AddPatmforcingCPU 

```C++
template<class T>
__host__ void AddPatmforcingCPU (
    Param XParam,
    BlockP < T > XBlock,
    DynForcingP < float > PAtm,
    Model < T > XModel
) 
```




<hr>



### function AddPatmforcingGPU 

```C++
template<class T>
__global__ void AddPatmforcingGPU (
    Param XParam,
    BlockP < T > XBlock,
    DynForcingP < float > PAtm,
    Model < T > XModel
) 
```




<hr>



### function AddRiverForcing 

```C++
template<class T>
__host__ void AddRiverForcing (
    Param XParam,
    Loop < T > XLoop,
    std::vector< River > XRivers,
    Model < T > XModel
) 
```




<hr>



### function AddinfiltrationImplicitCPU 

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




<hr>



### function AddinfiltrationImplicitGPU 

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




<hr>



### function AddrainforcingCPU 

```C++
template<class T>
__host__ void AddrainforcingCPU (
    Param XParam,
    BlockP < T > XBlock,
    DynForcingP < float > Rain,
    AdvanceP < T > XAdv
) 
```




<hr>



### function AddrainforcingGPU 

```C++
template<class T>
__global__ void AddrainforcingGPU (
    Param XParam,
    BlockP < T > XBlock,
    DynForcingP < float > Rain,
    AdvanceP < T > XAdv
) 
```




<hr>



### function AddrainforcingImplicitCPU 

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




<hr>



### function AddrainforcingImplicitGPU 

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




<hr>



### function AddwindforcingCPU 

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




<hr>



### function AddwindforcingGPU 

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




<hr>



### function Forcingthisstep 

```C++
void Forcingthisstep (
    Param XParam,
    double totaltime,
    DynForcingP < float > & XDynForcing
) 
```




<hr>



### function InjectRiverGPU 

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




<hr>



### function deformstep 

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




<hr>



### function interpDyn2BUQ 

```C++
template<class T>
__device__ T interpDyn2BUQ (
    T x,
    T y,
    TexSetP Forcing
) 
```




<hr>



### function updateforcing 

```C++
template<class T>
void updateforcing (
    Param XParam,
    Loop < T > XLoop,
    Forcing < float > & XForcing
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Updateforcing.h`


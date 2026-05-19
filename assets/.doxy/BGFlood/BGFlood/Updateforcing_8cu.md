

# File Updateforcing.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Updateforcing.cu**](Updateforcing_8cu.md)

[Go to the source code of this file](Updateforcing_8cu_source.md)



* `#include "Updateforcing.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**AddDeformCPU**](#function-adddeformcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**deformmap**](classdeformmap.md)&lt; float &gt; defmap, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T scale, T \* zb) <br>_Perform a deformation step on the model on the CPU. Applies deformation maps to the model based on the current simulation time and deformation parameters._  __ |
|  \_\_global\_\_ void | [**AddDeformGPU**](#function-adddeformgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**deformmap**](classdeformmap.md)&lt; float &gt; defmap, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T scale, T \* zb) <br>_Perform a deformation step on the model on the GPU. Applies deformation maps to the model based on the current simulation time and deformation parameters._  |
|  \_\_host\_\_ void | [**AddPatmforcingCPU**](#function-addpatmforcingcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; PAtm, [**Model**](structModel.md)&lt; T &gt; XModel) <br>_Add atmospheric pressure forcing to the model on the CPU. Adds atmospheric pressure forcing to the model based on pressure data and current simulation time._  |
|  template \_\_host\_\_ void | [**AddPatmforcingCPU&lt; double &gt;**](#function-addpatmforcingcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; PAtm, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template \_\_host\_\_ void | [**AddPatmforcingCPU&lt; float &gt;**](#function-addpatmforcingcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; PAtm, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |
|  \_\_global\_\_ void | [**AddPatmforcingGPU**](#function-addpatmforcinggpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; PAtm, [**Model**](structModel.md)&lt; T &gt; XModel) <br>_Add atmospheric pressure forcing to the model on the GPU. Adds atmospheric pressure forcing to the model based on pressure data and current simulation time._  |
|  template \_\_global\_\_ void | [**AddPatmforcingGPU&lt; double &gt;**](#function-addpatmforcinggpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; PAtm, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template \_\_global\_\_ void | [**AddPatmforcingGPU&lt; float &gt;**](#function-addpatmforcinggpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; PAtm, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |
|  \_\_host\_\_ void | [**AddRiverForcing**](#function-addriverforcing) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, std::vector&lt; [**River**](classRiver.md) &gt; XRivers, [**Model**](structModel.md)&lt; T &gt; XModel) <br>_Add river forcing to the model. Adds river discharge forcing to the model based on river data and current simulation time._  |
|  template \_\_host\_\_ void | [**AddRiverForcing&lt; double &gt;**](#function-addriverforcing-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, std::vector&lt; [**River**](classRiver.md) &gt; XRivers, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template \_\_host\_\_ void | [**AddRiverForcing&lt; float &gt;**](#function-addriverforcing-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, std::vector&lt; [**River**](classRiver.md) &gt; XRivers, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |
|  \_\_host\_\_ void | [**AddinfiltrationImplicitCPU**](#function-addinfiltrationimplicitcpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* il, T \* cl, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* hgw) <br>_Add infiltration forcing to the model implicitly on the CPU. Adds infiltration forcing to the model based on infiltration data and current water depth, updating water depth and surface elevation._  |
|  template \_\_host\_\_ void | [**AddinfiltrationImplicitCPU&lt; double &gt;**](#function-addinfiltrationimplicitcpu-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* il, double \* cl, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, double \* hgw) <br> |
|  template \_\_host\_\_ void | [**AddinfiltrationImplicitCPU&lt; float &gt;**](#function-addinfiltrationimplicitcpu-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* il, float \* cl, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, float \* hgw) <br> |
|  \_\_global\_\_ void | [**AddinfiltrationImplicitGPU**](#function-addinfiltrationimplicitgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* il, T \* cl, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* hgw) <br>_Add infiltration forcing to the model implicitly on the GPU. Adds infiltration forcing to the model based on infiltration data and current water depth, updating water depth and surface elevation._  |
|  template \_\_global\_\_ void | [**AddinfiltrationImplicitGPU&lt; double &gt;**](#function-addinfiltrationimplicitgpu-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* il, double \* cl, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, double \* hgw) <br> |
|  template \_\_global\_\_ void | [**AddinfiltrationImplicitGPU&lt; float &gt;**](#function-addinfiltrationimplicitgpu-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* il, float \* cl, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, float \* hgw) <br> |
|  \_\_host\_\_ void | [**AddrainforcingCPU**](#function-addrainforcingcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_Add rainfall forcing to the model on the CPU. Adds rainfall forcing to the model based on rainfall data and current simulation time._  |
|  template \_\_host\_\_ void | [**AddrainforcingCPU&lt; double &gt;**](#function-addrainforcingcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**AdvanceP**](structAdvanceP.md)&lt; double &gt; XAdv) <br> |
|  template \_\_host\_\_ void | [**AddrainforcingCPU&lt; float &gt;**](#function-addrainforcingcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**AdvanceP**](structAdvanceP.md)&lt; float &gt; XAdv) <br> |
|  \_\_global\_\_ void | [**AddrainforcingGPU**](#function-addrainforcinggpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_Add rainfall forcing to the model on the GPU. Adds rainfall forcing to the model based on rainfall data and current simulation time._  |
|  template \_\_global\_\_ void | [**AddrainforcingGPU&lt; double &gt;**](#function-addrainforcinggpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**AdvanceP**](structAdvanceP.md)&lt; double &gt; XAdv) <br> |
|  template \_\_global\_\_ void | [**AddrainforcingGPU&lt; float &gt;**](#function-addrainforcinggpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**AdvanceP**](structAdvanceP.md)&lt; float &gt; XAdv) <br> |
|  \_\_host\_\_ void | [**AddrainforcingImplicitCPU**](#function-addrainforcingimplicitcpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv) <br>_Add rainfall forcing to the model implicitly on the CPU. Adds rainfall forcing to the model based on rainfall data and current simulation time, updating water depth and surface elevation._  |
|  template \_\_host\_\_ void | [**AddrainforcingImplicitCPU&lt; double &gt;**](#function-addrainforcingimplicitcpu-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv) <br> |
|  template \_\_host\_\_ void | [**AddrainforcingImplicitCPU&lt; float &gt;**](#function-addrainforcingimplicitcpu-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv) <br> |
|  \_\_global\_\_ void | [**AddrainforcingImplicitGPU**](#function-addrainforcingimplicitgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv) <br>_Add rainfall forcing to the model implicitly on the GPU. Adds rainfall forcing to the model based on rainfall data and current simulation time, updating water depth and surface elevation._  |
|  template \_\_global\_\_ void | [**AddrainforcingImplicitGPU&lt; double &gt;**](#function-addrainforcingimplicitgpu-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv) <br> |
|  template \_\_global\_\_ void | [**AddrainforcingImplicitGPU&lt; float &gt;**](#function-addrainforcingimplicitgpu-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Rain, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv) <br> |
|  \_\_host\_\_ void | [**AddwindforcingCPU**](#function-addwindforcingcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Uwind, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Vwind, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_Add wind forcing to the model on the CPU. Adds wind forcing to the model based on wind data and current simulation time._  |
|  template \_\_host\_\_ void | [**AddwindforcingCPU&lt; double &gt;**](#function-addwindforcingcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Uwind, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Vwind, [**AdvanceP**](structAdvanceP.md)&lt; double &gt; XAdv) <br> |
|  template \_\_host\_\_ void | [**AddwindforcingCPU&lt; float &gt;**](#function-addwindforcingcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Uwind, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Vwind, [**AdvanceP**](structAdvanceP.md)&lt; float &gt; XAdv) <br> |
|  \_\_global\_\_ void | [**AddwindforcingGPU**](#function-addwindforcinggpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Uwind, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Vwind, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_Add wind forcing to the model on the GPU. Adds wind forcing to the model based on wind data and current simulation time._  |
|  template \_\_global\_\_ void | [**AddwindforcingGPU&lt; double &gt;**](#function-addwindforcinggpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Uwind, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Vwind, [**AdvanceP**](structAdvanceP.md)&lt; double &gt; XAdv) <br> |
|  template \_\_global\_\_ void | [**AddwindforcingGPU&lt; float &gt;**](#function-addwindforcinggpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Uwind, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Vwind, [**AdvanceP**](structAdvanceP.md)&lt; float &gt; XAdv) <br> |
|  void | [**Forcingthisstep**](#function-forcingthisstep) ([**Param**](classParam.md) XParam, double totaltime, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; & XDynForcing) <br>_Update dynamic forcing for the current simulation step._  |
|  \_\_global\_\_ void | [**InjectManyRiversGPU**](#function-injectmanyriversgpu) ([**Param**](classParam.md) XParam, int irib, [**RiverInfo**](structRiverInfo.md)&lt; T &gt; XRin, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_Inject multiple river discharges into the model grid on the GPU. Injects river discharges into the model grid based on river geometry and discharge rates. Optimisation by processing multiple rivers in a single kernel launch._  |
|  \_\_host\_\_ void | [**InjectRiverCPU**](#function-injectrivercpu) ([**Param**](classParam.md) XParam, [**River**](classRiver.md) XRiver, T qnow, int nblkriver, int \* Riverblks, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_Inject river discharge into the model grid on the CPU. Injects river discharge into the model grid based on river geometry and discharge rate._  |
|  template \_\_host\_\_ void | [**InjectRiverCPU&lt; double &gt;**](#function-injectrivercpu-double) ([**Param**](classParam.md) XParam, [**River**](classRiver.md) XRiver, double qnow, int nblkriver, int \* Riverblks, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**AdvanceP**](structAdvanceP.md)&lt; double &gt; XAdv) <br> |
|  template \_\_host\_\_ void | [**InjectRiverCPU&lt; float &gt;**](#function-injectrivercpu-float) ([**Param**](classParam.md) XParam, [**River**](classRiver.md) XRiver, float qnow, int nblkriver, int \* Riverblks, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**AdvanceP**](structAdvanceP.md)&lt; float &gt; XAdv) <br> |
|  \_\_global\_\_ void | [**InjectRiverGPU**](#function-injectrivergpu) ([**Param**](classParam.md) XParam, [**River**](classRiver.md) XRiver, T qnow, int \* Riverblks, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**AdvanceP**](structAdvanceP.md)&lt; T &gt; XAdv) <br>_Inject river discharge into the model grid on the GPU. Injects river discharge into the model grid based on river geometry and discharge rate._  |
|  template \_\_global\_\_ void | [**InjectRiverGPU&lt; double &gt;**](#function-injectrivergpu-double) ([**Param**](classParam.md) XParam, [**River**](classRiver.md) XRiver, double qnow, int \* Riverblks, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**AdvanceP**](structAdvanceP.md)&lt; double &gt; XAdv) <br> |
|  template \_\_global\_\_ void | [**InjectRiverGPU&lt; float &gt;**](#function-injectrivergpu-float) ([**Param**](classParam.md) XParam, [**River**](classRiver.md) XRiver, float qnow, int \* Riverblks, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**AdvanceP**](structAdvanceP.md)&lt; float &gt; XAdv) <br> |
|  void | [**deformstep**](#function-deformstep) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, std::vector&lt; [**deformmap**](classdeformmap.md)&lt; float &gt; &gt; deform, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br>_Perform a deformation step on the model. Applies deformation maps to the model based on the current simulation time and deformation parameters. Overloaded function to handle both CPU and GPU models._  |
|  void | [**deformstep**](#function-deformstep) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, std::vector&lt; [**deformmap**](classdeformmap.md)&lt; float &gt; &gt; deform, [**Model**](structModel.md)&lt; T &gt; XModel) <br>_Perform a deformation step on the model. Applies deformation maps to the model based on the current simulation time and deformation parameters._  |
|  template void | [**deformstep&lt; double &gt;**](#function-deformstep-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, std::vector&lt; [**deformmap**](classdeformmap.md)&lt; float &gt; &gt; deform, [**Model**](structModel.md)&lt; double &gt; XModel, [**Model**](structModel.md)&lt; double &gt; XModel\_g) <br> |
|  template void | [**deformstep&lt; float &gt;**](#function-deformstep-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, std::vector&lt; [**deformmap**](classdeformmap.md)&lt; float &gt; &gt; deform, [**Model**](structModel.md)&lt; float &gt; XModel, [**Model**](structModel.md)&lt; float &gt; XModel\_g) <br> |
|  \_\_device\_\_ T | [**interp2BUQ**](#function-interp2buq) (T x, T y, [**TexSetP**](structTexSetP.md) Forcing) <br>_Interpolate data at given coordinates on the GPU. Interpolates data at specified coordinates using bilinear interpolation._  |
|  template \_\_device\_\_ double | [**interp2BUQ&lt; double &gt;**](#function-interp2buq-double) (double x, double y, [**TexSetP**](structTexSetP.md) Forcing) <br> |
|  template \_\_device\_\_ float | [**interp2BUQ&lt; float &gt;**](#function-interp2buq-float) (float x, float y, [**TexSetP**](structTexSetP.md) Forcing) <br> |
|  \_\_device\_\_ T | [**interpDyn2BUQ**](#function-interpdyn2buq) (T x, T y, [**TexSetP**](structTexSetP.md) Forcing) <br>_Interpolate dynamic forcing data at given coordinates on the GPU. Interpolates dynamic forcing data at specified coordinates using bilinear interpolation._  |
|  template \_\_device\_\_ double | [**interpDyn2BUQ&lt; double &gt;**](#function-interpdyn2buq-double) (double x, double y, [**TexSetP**](structTexSetP.md) Forcing) <br> |
|  template \_\_device\_\_ float | [**interpDyn2BUQ&lt; float &gt;**](#function-interpdyn2buq-float) (float x, float y, [**TexSetP**](structTexSetP.md) Forcing) <br> |
|  void | [**updateforcing**](#function-updateforcing) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing) <br>_Update dynamic forcings for the current simulation step. Updates the dynamic forcing data for the current simulation time step._  |
|  template void | [**updateforcing&lt; double &gt;**](#function-updateforcing-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing) <br> |
|  template void | [**updateforcing&lt; float &gt;**](#function-updateforcing-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing) <br> |




























## Public Functions Documentation




### function AddDeformCPU 

_Perform a deformation step on the model on the CPU. Applies deformation maps to the model based on the current simulation time and deformation parameters._  __
```C++
template<class T>
__host__ void AddDeformCPU (
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
* `XBlock` Block data structure 
* `defmap` Deformation map 
* `XEv` Evolving data structure 
* `scale` Scaling factor for deformation 
* `zb` Bed elevation array 




        

<hr>



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



### function AddPatmforcingCPU&lt; double &gt; 

```C++
template __host__ void AddPatmforcingCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    DynForcingP < float > PAtm,
    Model < double > XModel
) 
```




<hr>



### function AddPatmforcingCPU&lt; float &gt; 

```C++
template __host__ void AddPatmforcingCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    DynForcingP < float > PAtm,
    Model < float > XModel
) 
```




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



### function AddPatmforcingGPU&lt; double &gt; 

```C++
template __global__ void AddPatmforcingGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    DynForcingP < float > PAtm,
    Model < double > XModel
) 
```




<hr>



### function AddPatmforcingGPU&lt; float &gt; 

```C++
template __global__ void AddPatmforcingGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    DynForcingP < float > PAtm,
    Model < float > XModel
) 
```




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



### function AddRiverForcing&lt; double &gt; 

```C++
template __host__ void AddRiverForcing< double > (
    Param XParam,
    Loop < double > XLoop,
    std::vector< River > XRivers,
    Model < double > XModel
) 
```




<hr>



### function AddRiverForcing&lt; float &gt; 

```C++
template __host__ void AddRiverForcing< float > (
    Param XParam,
    Loop < float > XLoop,
    std::vector< River > XRivers,
    Model < float > XModel
) 
```




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



### function AddinfiltrationImplicitCPU&lt; double &gt; 

```C++
template __host__ void AddinfiltrationImplicitCPU< double > (
    Param XParam,
    Loop < double > XLoop,
    BlockP < double > XBlock,
    double * il,
    double * cl,
    EvolvingP < double > XEv,
    double * hgw
) 
```




<hr>



### function AddinfiltrationImplicitCPU&lt; float &gt; 

```C++
template __host__ void AddinfiltrationImplicitCPU< float > (
    Param XParam,
    Loop < float > XLoop,
    BlockP < float > XBlock,
    float * il,
    float * cl,
    EvolvingP < float > XEv,
    float * hgw
) 
```




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



### function AddinfiltrationImplicitGPU&lt; double &gt; 

```C++
template __global__ void AddinfiltrationImplicitGPU< double > (
    Param XParam,
    Loop < double > XLoop,
    BlockP < double > XBlock,
    double * il,
    double * cl,
    EvolvingP < double > XEv,
    double * hgw
) 
```




<hr>



### function AddinfiltrationImplicitGPU&lt; float &gt; 

```C++
template __global__ void AddinfiltrationImplicitGPU< float > (
    Param XParam,
    Loop < float > XLoop,
    BlockP < float > XBlock,
    float * il,
    float * cl,
    EvolvingP < float > XEv,
    float * hgw
) 
```




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



### function AddrainforcingCPU&lt; double &gt; 

```C++
template __host__ void AddrainforcingCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    DynForcingP < float > Rain,
    AdvanceP < double > XAdv
) 
```




<hr>



### function AddrainforcingCPU&lt; float &gt; 

```C++
template __host__ void AddrainforcingCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    DynForcingP < float > Rain,
    AdvanceP < float > XAdv
) 
```




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



### function AddrainforcingGPU&lt; double &gt; 

```C++
template __global__ void AddrainforcingGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    DynForcingP < float > Rain,
    AdvanceP < double > XAdv
) 
```




<hr>



### function AddrainforcingGPU&lt; float &gt; 

```C++
template __global__ void AddrainforcingGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    DynForcingP < float > Rain,
    AdvanceP < float > XAdv
) 
```




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



### function AddrainforcingImplicitCPU&lt; double &gt; 

```C++
template __host__ void AddrainforcingImplicitCPU< double > (
    Param XParam,
    Loop < double > XLoop,
    BlockP < double > XBlock,
    DynForcingP < float > Rain,
    EvolvingP < double > XEv
) 
```




<hr>



### function AddrainforcingImplicitCPU&lt; float &gt; 

```C++
template __host__ void AddrainforcingImplicitCPU< float > (
    Param XParam,
    Loop < float > XLoop,
    BlockP < float > XBlock,
    DynForcingP < float > Rain,
    EvolvingP < float > XEv
) 
```




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



### function AddrainforcingImplicitGPU&lt; double &gt; 

```C++
template __global__ void AddrainforcingImplicitGPU< double > (
    Param XParam,
    Loop < double > XLoop,
    BlockP < double > XBlock,
    DynForcingP < float > Rain,
    EvolvingP < double > XEv
) 
```




<hr>



### function AddrainforcingImplicitGPU&lt; float &gt; 

```C++
template __global__ void AddrainforcingImplicitGPU< float > (
    Param XParam,
    Loop < float > XLoop,
    BlockP < float > XBlock,
    DynForcingP < float > Rain,
    EvolvingP < float > XEv
) 
```




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



### function AddwindforcingCPU&lt; double &gt; 

```C++
template __host__ void AddwindforcingCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    DynForcingP < float > Uwind,
    DynForcingP < float > Vwind,
    AdvanceP < double > XAdv
) 
```




<hr>



### function AddwindforcingCPU&lt; float &gt; 

```C++
template __host__ void AddwindforcingCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    DynForcingP < float > Uwind,
    DynForcingP < float > Vwind,
    AdvanceP < float > XAdv
) 
```




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



### function AddwindforcingGPU&lt; double &gt; 

```C++
template __global__ void AddwindforcingGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    DynForcingP < float > Uwind,
    DynForcingP < float > Vwind,
    AdvanceP < double > XAdv
) 
```




<hr>



### function AddwindforcingGPU&lt; float &gt; 

```C++
template __global__ void AddwindforcingGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    DynForcingP < float > Uwind,
    DynForcingP < float > Vwind,
    AdvanceP < float > XAdv
) 
```




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



### function InjectManyRiversGPU 

_Inject multiple river discharges into the model grid on the GPU. Injects river discharges into the model grid based on river geometry and discharge rates. Optimisation by processing multiple rivers in a single kernel launch._ 
```C++
template<class T>
__global__ void InjectManyRiversGPU (
    Param XParam,
    int irib,
    RiverInfo < T > XRin,
    BlockP < T > XBlock,
    AdvanceP < T > XAdv
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `irib` Index of the river being processed 
* `XRin` [**River**](classRiver.md) information structure 
* `XBlock` Block data structure 
* `XAdv` Advance data structure 




        

<hr>



### function InjectRiverCPU 

_Inject river discharge into the model grid on the CPU. Injects river discharge into the model grid based on river geometry and discharge rate._ 
```C++
template<class T>
__host__ void InjectRiverCPU (
    Param XParam,
    River XRiver,
    T qnow,
    int nblkriver,
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



### function InjectRiverCPU&lt; double &gt; 

```C++
template __host__ void InjectRiverCPU< double > (
    Param XParam,
    River XRiver,
    double qnow,
    int nblkriver,
    int * Riverblks,
    BlockP < double > XBlock,
    AdvanceP < double > XAdv
) 
```




<hr>



### function InjectRiverCPU&lt; float &gt; 

```C++
template __host__ void InjectRiverCPU< float > (
    Param XParam,
    River XRiver,
    float qnow,
    int nblkriver,
    int * Riverblks,
    BlockP < float > XBlock,
    AdvanceP < float > XAdv
) 
```




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



### function InjectRiverGPU&lt; double &gt; 

```C++
template __global__ void InjectRiverGPU< double > (
    Param XParam,
    River XRiver,
    double qnow,
    int * Riverblks,
    BlockP < double > XBlock,
    AdvanceP < double > XAdv
) 
```




<hr>



### function InjectRiverGPU&lt; float &gt; 

```C++
template __global__ void InjectRiverGPU< float > (
    Param XParam,
    River XRiver,
    float qnow,
    int * Riverblks,
    BlockP < float > XBlock,
    AdvanceP < float > XAdv
) 
```




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



### function deformstep 

_Perform a deformation step on the model. Applies deformation maps to the model based on the current simulation time and deformation parameters._ 
```C++
template<class T>
void deformstep (
    Param XParam,
    Loop < T > XLoop,
    std::vector< deformmap < float > > deform,
    Model < T > XModel
) 
```





**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XLoop` [**Loop**](structLoop.md) structure containing time information 
* `deform` Vector of deformation maps 
* `XModel` [**Model**](structModel.md) data structure 




        

<hr>



### function deformstep&lt; double &gt; 

```C++
template void deformstep< double > (
    Param XParam,
    Loop < double > XLoop,
    std::vector< deformmap < float > > deform,
    Model < double > XModel,
    Model < double > XModel_g
) 
```




<hr>



### function deformstep&lt; float &gt; 

```C++
template void deformstep< float > (
    Param XParam,
    Loop < float > XLoop,
    std::vector< deformmap < float > > deform,
    Model < float > XModel,
    Model < float > XModel_g
) 
```




<hr>



### function interp2BUQ 

_Interpolate data at given coordinates on the GPU. Interpolates data at specified coordinates using bilinear interpolation._ 
```C++
template<class T>
__device__ T interp2BUQ (
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



### function interp2BUQ&lt; double &gt; 

```C++
template __device__ double interp2BUQ< double > (
    double x,
    double y,
    TexSetP Forcing
) 
```




<hr>



### function interp2BUQ&lt; float &gt; 

```C++
template __device__ float interp2BUQ< float > (
    float x,
    float y,
    TexSetP Forcing
) 
```




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



### function interpDyn2BUQ&lt; double &gt; 

```C++
template __device__ double interpDyn2BUQ< double > (
    double x,
    double y,
    TexSetP Forcing
) 
```




<hr>



### function interpDyn2BUQ&lt; float &gt; 

```C++
template __device__ float interpDyn2BUQ< float > (
    float x,
    float y,
    TexSetP Forcing
) 
```




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



### function updateforcing&lt; double &gt; 

```C++
template void updateforcing< double > (
    Param XParam,
    Loop < double > XLoop,
    Forcing < float > & XForcing
) 
```




<hr>



### function updateforcing&lt; float &gt; 

```C++
template void updateforcing< float > (
    Param XParam,
    Loop < float > XLoop,
    Forcing < float > & XForcing
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Updateforcing.cu`


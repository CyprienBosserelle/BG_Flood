

# File BG\_Flood.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**BG\_Flood.h**](BG__Flood_8h.md)

[Go to the source code of this file](BG__Flood_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Write_txtlog.h"`
* `#include "ReadInput.h"`
* `#include "ReadForcing.h"`
* `#include "Setup_GPU.h"`
* `#include "Util_CPU.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "Mesh.h"`
* `#include "InitialConditions.h"`
* `#include "Adaptation.h"`
* `#include "Mainloop.h"`
* `#include "Testing.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**mainwork**](#function-mainwork) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br>_Main model setup and execution routine for BG\_Flood._  |




























## Public Functions Documentation




### function mainwork 

_Main model setup and execution routine for BG\_Flood._ 
```C++
template<class T>
int mainwork (
    Param XParam,
    Forcing < float > XForcing,
    Model < T > XModel,
    Model < T > XModel_g
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XForcing` [**Forcing**](structForcing.md) data 
* `XModel` [**Model**](structModel.md) structure (CPU) 
* `XModel_g` [**Model**](structModel.md) structure (GPU) 



**Returns:**

Status code (0 for success)


This function performs the main setup and execution steps for the BG\_Flood model:
* Reads and verifies input/forcing data
* Initializes mesh and initial conditions
* Performs initial adaptation
* Sets up GPU resources
* Runs the main simulation loop (MainLoop)
* Handles test mode if specified




Integrates all major model components and coordinates CPU/GPU execution. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/BG_Flood.h`


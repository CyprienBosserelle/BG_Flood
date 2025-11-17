

# File InitialConditions.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**InitialConditions.h**](_initial_conditions_8h.md)

[Go to the source code of this file](_initial_conditions_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Forcing.h"`
* `#include "MemManagement.h"`
* `#include "Util_CPU.h"`
* `#include "Arrays.h"`
* `#include "Write_txtlog.h"`
* `#include "GridManip.h"`
* `#include "InitEvolv.h"`
* `#include "Gradients.h"`
* `#include "Spherical.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Calcbndblks**](#function-calcbndblks) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock) <br>_Calculates the number of blocks on each boundary of the domain._  |
|  void | [**FindTSoutNodes**](#function-findtsoutnodes) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**BndblockP**](struct_bndblock_p.md)&lt; T &gt; & bnd) <br>_Finds and assigns output nodes to blocks for time series output._  |
|  void | [**Findbndblks**](#function-findbndblks) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing) <br>_Finds which blocks on the model edge belong to a side boundary._  |
|  void | [**InitRivers**](#function-initrivers) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Initializes river discharge areas and assigns river information to model blocks._  |
|  void | [**InitTSOutput**](#function-inittsoutput) ([**Param**](class_param.md) XParam) <br>_Initializes time series output files for specified nodes._  |
|  void | [**InitialConditions**](#function-initialconditions) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Initializes model parameters, bathymetry, friction, initial conditions, and output variables._  |
|  void | [**Initmaparray**](#function-initmaparray) ([**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Initializes output variable maps and metadata for the simulation._  |
|  void | [**Initoutzone**](#function-initoutzone) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock) <br>_Initializes output zones for the simulation domain._  |
|  void | [**InitzbgradientCPU**](#function-initzbgradientcpu) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Initializes bathymetry gradient and halo on CPU._  |
|  void | [**InitzbgradientGPU**](#function-initzbgradientgpu) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Initializes bathymetry gradient and halo on GPU._  |
|  void | [**calcactiveCellCPU**](#function-calcactivecellcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, T \* zb) <br>_Calculates active cells in the domain based on mask elevation and area of interest._  |
|  void | [**initOutputTimes**](#function-initoutputtimes) ([**Param**](class_param.md) XParam, std::vector&lt; double &gt; & OutputT, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock) <br>_Compiles and sorts output times for map outputs, including zone outputs._  |
|  void | [**initoutput**](#function-initoutput) ([**Param**](class_param.md) & XParam, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Initializes output arrays and maps for the simulation._  |




























## Public Functions Documentation




### function Calcbndblks 

_Calculates the number of blocks on each boundary of the domain._ 
```C++
template<class T>
void Calcbndblks (
    Param & XParam,
    Forcing < float > & XForcing,
    BlockP < T > XBlock
) 
```



Updates counts for left, right, top, and bottom boundaries and stores them in the forcing and parameter structures.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data (float) 
* `XBlock` Block parameters 




        

<hr>



### function FindTSoutNodes 

_Finds and assigns output nodes to blocks for time series output._ 
```C++
template<class T>
void FindTSoutNodes (
    Param & XParam,
    BlockP < T > XBlock,
    BndblockP < T > & bnd
) 
```



Determines which block each output node belongs to and updates the boundary block structure.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `bnd` Boundary block structure 




        

<hr>



### function Findbndblks 

_Finds which blocks on the model edge belong to a side boundary._ 
```C++
template<class T>
void Findbndblks (
    Param XParam,
    Model < T > XModel,
    Forcing < float > & XForcing
) 
```



Populates arrays for blocks on each side boundary and updates the forcing structure.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XModel` [**Model**](struct_model.md) data 
* `XForcing` [**Forcing**](struct_forcing.md) data (float)

Find which block on the model edge belongs to a "side boundary" 


        

<hr>



### function InitRivers 

_Initializes river discharge areas and assigns river information to model blocks._ 
```C++
template<class T>
void InitRivers (
    Param XParam,
    Forcing < float > & XForcing,
    Model < T > & XModel
) 
```



Identifies grid cells affected by river discharge, calculates discharge areas, and sets up river-block relationships.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data (float) 
* `XModel` [**Model**](struct_model.md) data 




        

<hr>



### function InitTSOutput 

_Initializes time series output files for specified nodes._ 
```C++
void InitTSOutput (
    Param XParam
) 
```



Creates and overwrites output files for each node in the time series output list.




**Parameters:**


* `XParam` Simulation parameters 




        

<hr>



### function InitialConditions 

_Initializes model parameters, bathymetry, friction, initial conditions, and output variables._ 
```C++
template<class T>
void InitialConditions (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < T > & XModel
) 
```



Sets up the initial state of the simulation, including bathymetry, friction maps, evolving variables, river forcing, boundary blocks, active cells, and output arrays.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data (float) 
* `XModel` [**Model**](struct_model.md) data 




        

<hr>



### function Initmaparray 

_Initializes output variable maps and metadata for the simulation._ 
```C++
template<class T>
void Initmaparray (
    Model < T > & XModel
) 
```



Sets up output variable names, units, and long names for all tracked quantities in the model.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XModel` [**Model**](struct_model.md) data 




        

<hr>



### function Initoutzone 

_Initializes output zones for the simulation domain._ 
```C++
template<class T>
void Initoutzone (
    Param & XParam,
    BlockP < T > & XBlock
) 
```



Sets up output zones based on user input or defaults to the full domain if none specified.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 




        

<hr>



### function InitzbgradientCPU 

_Initializes bathymetry gradient and halo on CPU._ 
```C++
template<class T>
void InitzbgradientCPU (
    Param XParam,
    Model < T > XModel
) 
```



Computes gradients and refines bathymetry for the model blocks on the CPU.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XModel` [**Model**](struct_model.md) data 




        

<hr>



### function InitzbgradientGPU 

_Initializes bathymetry gradient and halo on GPU._ 
```C++
template<class T>
void InitzbgradientGPU (
    Param XParam,
    Model < T > XModel
) 
```



Computes gradients and refines bathymetry for the model blocks using CUDA streams and kernels.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XModel` [**Model**](struct_model.md) data 




        

<hr>



### function calcactiveCellCPU 

_Calculates active cells in the domain based on mask elevation and area of interest._ 
```C++
template<class T>
void calcactiveCellCPU (
    Param XParam,
    BlockP < T > XBlock,
    Forcing < float > & XForcing,
    T * zb
) 
```



Sets the active cell flag for each cell, removing rain from masked and boundary cells as needed.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data (float) 
* `zb` Bathymetry array 




        

<hr>



### function initOutputTimes 

_Compiles and sorts output times for map outputs, including zone outputs._ 
```C++
template<class T>
void initOutputTimes (
    Param XParam,
    std::vector< double > & OutputT,
    BlockP < T > & XBlock
) 
```



Combines times from the main output structure and all zone outputs, sorts and removes duplicates, and assigns to output arrays.


Creation of a vector for times requiering a map output Compilations of vectors and independent times from the general input and the different zones outputs




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `OutputT` Output times vector 
* `XBlock` Block parameters 




        

<hr>



### function initoutput 

_Initializes output arrays and maps for the simulation._ 
```C++
template<class T>
void initoutput (
    Param & XParam,
    Model < T > & XModel
) 
```



Sets up storage for evolving parameters, output zones, and output files.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XModel` [**Model**](struct_model.md) data 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/InitialConditions.h`


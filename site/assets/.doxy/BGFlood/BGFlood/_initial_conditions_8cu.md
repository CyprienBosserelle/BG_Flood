

# File InitialConditions.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**InitialConditions.cu**](_initial_conditions_8cu.md)

[Go to the source code of this file](_initial_conditions_8cu_source.md)



* `#include "InitialConditions.h"`
* `#include "Input.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Calcbndblks**](#function-calcbndblks) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock) <br>_Calculates the number of blocks on each boundary of the domain._  |
|  void | [**FindTSoutNodes**](#function-findtsoutnodes) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**BndblockP**](struct_bndblock_p.md)&lt; T &gt; & bnd) <br>_Finds and assigns output nodes to blocks for time series output._  |
|  template void | [**FindTSoutNodes&lt; double &gt;**](#function-findtsoutnodes-double) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**BndblockP**](struct_bndblock_p.md)&lt; double &gt; & bnd) <br> |
|  template void | [**FindTSoutNodes&lt; float &gt;**](#function-findtsoutnodes-float) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**BndblockP**](struct_bndblock_p.md)&lt; float &gt; & bnd) <br> |
|  void | [**Findbndblks**](#function-findbndblks) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing) <br>_Finds which blocks on the model edge belong to a side boundary._  |
|  void | [**Findoutzoneblks**](#function-findoutzoneblks) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock) <br>_Finds and assigns blocks to output zones based on user-defined rectangular areas._  |
|  template void | [**Findoutzoneblks&lt; double &gt;**](#function-findoutzoneblks-double) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; & XBlock) <br> |
|  template void | [**Findoutzoneblks&lt; float &gt;**](#function-findoutzoneblks-float) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; & XBlock) <br> |
|  std::vector&lt; double &gt; | [**GetTimeOutput**](#function-gettimeoutput) ([**T\_output**](class_t__output.md) time\_info) <br>_Creates a vector of output times from the input time structure._  |
|  void | [**InitRivers**](#function-initrivers) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Initializes river discharge areas and assigns river information to model blocks._  |
|  template void | [**InitRivers&lt; double &gt;**](#function-initrivers-double) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; double &gt; & XModel) <br> |
|  template void | [**InitRivers&lt; float &gt;**](#function-initrivers-float) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; float &gt; & XModel) <br> |
|  void | [**InitTSOutput**](#function-inittsoutput) ([**Param**](class_param.md) XParam) <br>_Initializes time series output files for specified nodes._  |
|  void | [**Initbndblks**](#function-initbndblks) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock) <br>_Initializes boundary block assignments and segment information._  |
|  void | [**InitialConditions**](#function-initialconditions) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Initializes model parameters, bathymetry, friction, initial conditions, and output variables._  |
|  template void | [**InitialConditions&lt; double &gt;**](#function-initialconditions-double) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; double &gt; & XModel) <br> |
|  template void | [**InitialConditions&lt; float &gt;**](#function-initialconditions-float) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; float &gt; & XModel) <br> |
|  void | [**Initmaparray**](#function-initmaparray) ([**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Initializes output variable maps and metadata for the simulation._  |
|  template void | [**Initmaparray&lt; double &gt;**](#function-initmaparray-double) ([**Model**](struct_model.md)&lt; double &gt; & XModel) <br> |
|  template void | [**Initmaparray&lt; float &gt;**](#function-initmaparray-float) ([**Model**](struct_model.md)&lt; float &gt; & XModel) <br> |
|  void | [**Initoutzone**](#function-initoutzone) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock) <br>_Initializes output zones for the simulation domain._  |
|  template void | [**Initoutzone&lt; double &gt;**](#function-initoutzone-double) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; & XBlock) <br> |
|  template void | [**Initoutzone&lt; float &gt;**](#function-initoutzone-float) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; & XBlock) <br> |
|  void | [**InitzbgradientCPU**](#function-initzbgradientcpu) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Initializes bathymetry gradient and halo on CPU._  |
|  template void | [**InitzbgradientCPU&lt; double &gt;**](#function-initzbgradientcpu-double) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; double &gt; XModel) <br> |
|  template void | [**InitzbgradientCPU&lt; float &gt;**](#function-initzbgradientcpu-float) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; float &gt; XModel) <br> |
|  void | [**InitzbgradientGPU**](#function-initzbgradientgpu) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Initializes bathymetry gradient and halo on GPU._  |
|  template void | [**InitzbgradientGPU&lt; double &gt;**](#function-initzbgradientgpu-double) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; double &gt; XModel) <br> |
|  template void | [**InitzbgradientGPU&lt; float &gt;**](#function-initzbgradientgpu-float) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; float &gt; XModel) <br> |
|  void | [**RectCornerBlk**](#function-rectcornerblk) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock, double xo, double yo, double xmax, double ymax, bool isEps, std::vector&lt; int &gt; & cornerblk) <br>_Finds the blocks containing the corners of a rectangular box (for output zone definition)._  |
|  void | [**calcactiveCellCPU**](#function-calcactivecellcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, T \* zb) <br>_Calculates active cells in the domain based on mask elevation and area of interest._  |
|  \_\_global\_\_ void | [**calcactiveCellGPU**](#function-calcactivecellgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* zb) <br>_CUDA kernel to calculate active cells on the GPU based on mask elevation._  |
|  void | [**initOutputTimes**](#function-initoutputtimes) ([**Param**](class_param.md) XParam, std::vector&lt; double &gt; & OutputT, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock) <br>_Compiles and sorts output times for map outputs, including zone outputs._  |
|  void | [**initinfiltration**](#function-initinfiltration) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* initLoss, T \* hgw) <br>_Initializes infiltration loss array for each cell._  |
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



### function FindTSoutNodes&lt; double &gt; 

```C++
template void FindTSoutNodes< double > (
    Param & XParam,
    BlockP < double > XBlock,
    BndblockP < double > & bnd
) 
```




<hr>



### function FindTSoutNodes&lt; float &gt; 

```C++
template void FindTSoutNodes< float > (
    Param & XParam,
    BlockP < float > XBlock,
    BndblockP < float > & bnd
) 
```




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



### function Findoutzoneblks 

_Finds and assigns blocks to output zones based on user-defined rectangular areas._ 
```C++
template<class T>
void Findoutzoneblks (
    Param & XParam,
    BlockP < T > & XBlock
) 
```



Determines which blocks belong to each output zone, computes zone boundaries, and updates the block structure. Initialise all storage involving parameters of the outzone objects




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 




        

<hr>



### function Findoutzoneblks&lt; double &gt; 

```C++
template void Findoutzoneblks< double > (
    Param & XParam,
    BlockP < double > & XBlock
) 
```




<hr>



### function Findoutzoneblks&lt; float &gt; 

```C++
template void Findoutzoneblks< float > (
    Param & XParam,
    BlockP < float > & XBlock
) 
```




<hr>



### function GetTimeOutput 

_Creates a vector of output times from the input time structure._ 
```C++
std::vector< double > GetTimeOutput (
    T_output time_info
) 
```



Combines independent values and time steps from the input structure.




**Parameters:**


* `time_info` Time output structure 



**Returns:**

Vector of output times 





        

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



### function InitRivers&lt; double &gt; 

```C++
template void InitRivers< double > (
    Param XParam,
    Forcing < float > & XForcing,
    Model < double > & XModel
) 
```




<hr>



### function InitRivers&lt; float &gt; 

```C++
template void InitRivers< float > (
    Param XParam,
    Forcing < float > & XForcing,
    Model < float > & XModel
) 
```




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



### function Initbndblks 

_Initializes boundary block assignments and segment information._ 
```C++
template<class T>
void Initbndblks (
    Param & XParam,
    Forcing < float > & XForcing,
    BlockP < T > XBlock
) 
```



Finds boundary blocks, assigns them to segments, and allocates arrays for segment sides and flow.



* Initialise bnd blk assign block to their relevant segment allocate memory...


* Find all the boundary blocks(block with themselves as neighbours)
* make an array to store which segment they belong to




If any bnd segment was specified
* scan each block and find which (if any) segment they belong to For each segment Calculate bbox if inbbox calc inpoly if inpoly overwrite assingned segment with new one
* Calculate nblk per segment & allocate (do for each segment)
* fill segment and side arrays for each segments






**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data (float) 
* `XBlock` Block parameters 




        

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



### function InitialConditions&lt; double &gt; 

```C++
template void InitialConditions< double > (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < double > & XModel
) 
```




<hr>



### function InitialConditions&lt; float &gt; 

```C++
template void InitialConditions< float > (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < float > & XModel
) 
```




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



### function Initmaparray&lt; double &gt; 

```C++
template void Initmaparray< double > (
    Model < double > & XModel
) 
```




<hr>



### function Initmaparray&lt; float &gt; 

```C++
template void Initmaparray< float > (
    Model < float > & XModel
) 
```




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



### function Initoutzone&lt; double &gt; 

```C++
template void Initoutzone< double > (
    Param & XParam,
    BlockP < double > & XBlock
) 
```




<hr>



### function Initoutzone&lt; float &gt; 

```C++
template void Initoutzone< float > (
    Param & XParam,
    BlockP < float > & XBlock
) 
```




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



### function InitzbgradientCPU&lt; double &gt; 

```C++
template void InitzbgradientCPU< double > (
    Param XParam,
    Model < double > XModel
) 
```




<hr>



### function InitzbgradientCPU&lt; float &gt; 

```C++
template void InitzbgradientCPU< float > (
    Param XParam,
    Model < float > XModel
) 
```




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



### function InitzbgradientGPU&lt; double &gt; 

```C++
template void InitzbgradientGPU< double > (
    Param XParam,
    Model < double > XModel
) 
```




<hr>



### function InitzbgradientGPU&lt; float &gt; 

```C++
template void InitzbgradientGPU< float > (
    Param XParam,
    Model < float > XModel
) 
```




<hr>



### function RectCornerBlk 

_Finds the blocks containing the corners of a rectangular box (for output zone definition)._ 
```C++
template<class T>
void RectCornerBlk (
    Param & XParam,
    BlockP < T > & XBlock,
    double xo,
    double yo,
    double xmax,
    double ymax,
    bool isEps,
    std::vector< int > & cornerblk
) 
```



Returns indices of blocks through "cornerblk" at the corners of the rectangle, starting from bottom left and turning clockwise.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `xo` X start 
* `yo` Y start 
* `xmax` X end 
* `ymax` Y end 
* `isEps` Whether to use epsilon margin 
* `cornerblk` Vector to store corner block indices

Find the block containing the border of a rectangular box (used for the defining the output zones) The indice of the blocks are returned through "cornerblk" from bottom left turning in the clockwise direction 


        

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



### function calcactiveCellGPU 

_CUDA kernel to calculate active cells on the GPU based on mask elevation._ 
```C++
template<class T>
__global__ void calcactiveCellGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * zb
) 
```



Sets the active cell flag for each cell in the block using GPU parallelism.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
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



### function initinfiltration 

_Initializes infiltration loss array for each cell._ 
```C++
template<class T>
void initinfiltration (
    Param XParam,
    BlockP < T > XBlock,
    T * h,
    T * initLoss,
    T * hgw
) 
```



Sets initial infiltration loss to zero for wet cells.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `h` Water depth array 
* `initLoss` Initial loss array 
* `hgw` Groundwater head array 




        

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
The documentation for this class was generated from the following file `src/InitialConditions.cu`


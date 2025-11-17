

# File Mesh.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Mesh.h**](_mesh_8h.md)

[Go to the source code of this file](_mesh_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Forcing.h"`
* `#include "MemManagement.h"`
* `#include "Util_CPU.h"`
* `#include "Arrays.h"`
* `#include "Write_txtlog.h"`
* `#include "GridManip.h"`
* `#include "Poly.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**CalcInitnblk**](#function-calcinitnblk) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing) <br>_Calculates the initial number of blocks for the mesh._  |
|  int | [**CalcMaskblk**](#function-calcmaskblk) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock) <br>_Calculates the number of blocks with masked neighbors (for boundary handling)._  |
|  void | [**FindMaskblk**](#function-findmaskblk) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock) <br>_Identifies and stores blocks with masked sides for boundary processing._  |
|  void | [**InitBlockInfo**](#function-initblockinfo) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock) <br>_Initializes block information (active status, level, coordinates, neighbors)._  |
|  void | [**InitBlockadapt**](#function-initblockadapt) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**AdaptP**](struct_adapt_p.md) & XAdap) <br>_Initializes block adaptation arrays for mesh refinement/coarsening._  |
|  void | [**InitBlockneighbours**](#function-initblockneighbours) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock) <br>_Initializes neighbor relationships for each block in a uniform mesh._  |
|  void | [**InitBlockxoyo**](#function-initblockxoyo) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock) <br>_Initializes block coordinates and active status for the mesh._  |
|  void | [**InitMesh**](#function-initmesh) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Initializes the mesh and allocates memory for blocks._  |




























## Public Functions Documentation




### function CalcInitnblk 

_Calculates the initial number of blocks for the mesh._ 
```C++
int CalcInitnblk (
    Param XParam,
    Forcing < float > XForcing
) 
```





**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters (resolution, block size, etc.) 
* `XForcing` [**Forcing**](struct_forcing.md) data (bathymetry, AOI polygon, etc.) 



**Returns:**

Number of blocks to allocate for the mesh.


This function divides the domain into uniform blocks, checks masking and AOI, and counts blocks that are active for computation. 


        

<hr>



### function CalcMaskblk 

_Calculates the number of blocks with masked neighbors (for boundary handling)._ 
```C++
template<class T>
int CalcMaskblk (
    Param XParam,
    BlockP < T > XBlock
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 



**Returns:**

Number of blocks with masked neighbors. 





        

<hr>



### function FindMaskblk 

_Identifies and stores blocks with masked sides for boundary processing._ 
```C++
template<class T>
void FindMaskblk (
    Param XParam,
    BlockP < T > & XBlock
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure

Populates mask arrays for blocks with masked sides for later boundary condition handling. 


        

<hr>



### function InitBlockInfo 

_Initializes block information (active status, level, coordinates, neighbors)._ 
```C++
template<class T>
void InitBlockInfo (
    Param & XParam,
    Forcing < float > & XForcing,
    BlockP < T > & XBlock
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data 
* `XBlock` Block data structure 




        

<hr>



### function InitBlockadapt 

_Initializes block adaptation arrays for mesh refinement/coarsening._ 
```C++
template<class T>
void InitBlockadapt (
    Param & XParam,
    BlockP < T > XBlock,
    AdaptP & XAdap
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `XAdap` Adaptation data structure 




        

<hr>



### function InitBlockneighbours 

_Initializes neighbor relationships for each block in a uniform mesh._ 
```C++
template<class T>
void InitBlockneighbours (
    Param & XParam,
    Forcing < float > & XForcing,
    BlockP < T > & XBlock
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data 
* `XBlock` Block data structure

Sets up neighbor indices for each block (left, right, top, bottom, corners). 


        

<hr>



### function InitBlockxoyo 

_Initializes block coordinates and active status for the mesh._ 
```C++
template<class T>
void InitBlockxoyo (
    Param XParam,
    Forcing < float > XForcing,
    BlockP < T > & XBlock
) 
```





**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data 
* `XBlock` Block data structure

Sets block coordinates and marks active blocks based on mask and AOI polygon. Loops over all blocks, checks if each block is inside the area of interest (AOI), and if the mask threshold is met, sets the block as active and stores its coordinates. 


        

<hr>



### function InitMesh 

_Initializes the mesh and allocates memory for blocks._ 
```C++
template<class T>
void InitMesh (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < T > & XModel
) 
```





**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data 
* `XModel` [**Model**](struct_model.md) structure to hold mesh and block data

Allocates memory, initializes block info, adaptation info, and boundary masks. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Mesh.h`




# File Mesh.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Mesh.cu**](Mesh_8cu.md)

[Go to the source code of this file](Mesh_8cu_source.md)

_Mesh initialization and management routines for BG\_Flood GPU model._ [More...](#detailed-description)

* `#include "Mesh.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**CalcInitnblk**](#function-calcinitnblk) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing) <br>_Calculates the initial number of blocks for the mesh._  |
|  int | [**CalcMaskblk**](#function-calcmaskblk) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock) <br>_Calculates the number of blocks with masked neighbors (for boundary handling)._  |
|  template int | [**CalcMaskblk&lt; double &gt;**](#function-calcmaskblk-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock) <br> |
|  template int | [**CalcMaskblk&lt; float &gt;**](#function-calcmaskblk-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock) <br> |
|  void | [**FindMaskblk**](#function-findmaskblk) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br>_Identifies and stores blocks with masked sides for boundary processing._  |
|  template void | [**FindMaskblk&lt; double &gt;**](#function-findmaskblk-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; & XBlock) <br> |
|  template void | [**FindMaskblk&lt; float &gt;**](#function-findmaskblk-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; & XBlock) <br> |
|  void | [**InitBlockInfo**](#function-initblockinfo) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br>_Initializes block information (active status, level, coordinates, neighbors)._  |
|  void | [**InitBlockadapt**](#function-initblockadapt) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**AdaptP**](structAdaptP.md) & XAdap) <br>_Initializes block adaptation arrays for mesh refinement/coarsening._  |
|  template void | [**InitBlockadapt&lt; double &gt;**](#function-initblockadapt-double) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**AdaptP**](structAdaptP.md) & XAdap) <br> |
|  template void | [**InitBlockadapt&lt; float &gt;**](#function-initblockadapt-float) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**AdaptP**](structAdaptP.md) & XAdap) <br> |
|  void | [**InitBlockneighbours**](#function-initblockneighbours) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br>_Initializes neighbor relationships for each block in a uniform mesh._  |
|  template void | [**InitBlockneighbours&lt; double &gt;**](#function-initblockneighbours-double) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; double &gt; & XBlock) <br> |
|  template void | [**InitBlockneighbours&lt; float &gt;**](#function-initblockneighbours-float) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; float &gt; & XBlock) <br> |
|  void | [**InitBlockxoyo**](#function-initblockxoyo) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br>_Initializes block coordinates and active status for the mesh._  |
|  template void | [**InitBlockxoyo&lt; double &gt;**](#function-initblockxoyo-double) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**BlockP**](structBlockP.md)&lt; double &gt; & XBlockP) <br> |
|  template void | [**InitBlockxoyo&lt; float &gt;**](#function-initblockxoyo-float) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**BlockP**](structBlockP.md)&lt; float &gt; & XBlock) <br> |
|  void | [**InitMesh**](#function-initmesh) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br>_Initializes the mesh and allocates memory for blocks._  |
|  template void | [**InitMesh&lt; double &gt;**](#function-initmesh-double) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template void | [**InitMesh&lt; float &gt;**](#function-initmesh-float) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; float &gt; & XModel) <br> |




























## Detailed Description


Contains functions for block-based mesh setup, memory allocation, block adaptation, and block neighbor initialization. 


    
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


* `XParam` [**Model**](structModel.md) parameters (resolution, block size, etc.) 
* `XForcing` [**Forcing**](structForcing.md) data (bathymetry, AOI polygon, etc.) 



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


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 



**Returns:**

Number of blocks with masked neighbors. 





        

<hr>



### function CalcMaskblk&lt; double &gt; 

```C++
template int CalcMaskblk< double > (
    Param XParam,
    BlockP < double > XBlock
) 
```




<hr>



### function CalcMaskblk&lt; float &gt; 

```C++
template int CalcMaskblk< float > (
    Param XParam,
    BlockP < float > XBlock
) 
```




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


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure

Populates mask arrays for blocks with masked sides for later boundary condition handling. 


        

<hr>



### function FindMaskblk&lt; double &gt; 

```C++
template void FindMaskblk< double > (
    Param XParam,
    BlockP < double > & XBlock
) 
```




<hr>



### function FindMaskblk&lt; float &gt; 

```C++
template void FindMaskblk< float > (
    Param XParam,
    BlockP < float > & XBlock
) 
```




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


* `XParam` [**Model**](structModel.md) parameters 
* `XForcing` [**Forcing**](structForcing.md) data 
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


* `XParam` [**Model**](structModel.md) parameters 
* `XBlock` Block data structure 
* `XAdap` Adaptation data structure 




        

<hr>



### function InitBlockadapt&lt; double &gt; 

```C++
template void InitBlockadapt< double > (
    Param & XParam,
    BlockP < double > XBlock,
    AdaptP & XAdap
) 
```




<hr>



### function InitBlockadapt&lt; float &gt; 

```C++
template void InitBlockadapt< float > (
    Param & XParam,
    BlockP < float > XBlock,
    AdaptP & XAdap
) 
```




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


* `XParam` [**Model**](structModel.md) parameters 
* `XForcing` [**Forcing**](structForcing.md) data 
* `XBlock` Block data structure

Sets up neighbor indices for each block (left, right, top, bottom, corners). 


        

<hr>



### function InitBlockneighbours&lt; double &gt; 

```C++
template void InitBlockneighbours< double > (
    Param & XParam,
    Forcing < float > & XForcing,
    BlockP < double > & XBlock
) 
```




<hr>



### function InitBlockneighbours&lt; float &gt; 

```C++
template void InitBlockneighbours< float > (
    Param & XParam,
    Forcing < float > & XForcing,
    BlockP < float > & XBlock
) 
```




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


* `XParam` [**Model**](structModel.md) parameters 
* `XForcing` [**Forcing**](structForcing.md) data 
* `XBlock` Block data structure

Sets block coordinates and marks active blocks based on mask and AOI polygon. Loops over all blocks, checks if each block is inside the area of interest (AOI), and if the mask threshold is met, sets the block as active and stores its coordinates. 


        

<hr>



### function InitBlockxoyo&lt; double &gt; 

```C++
template void InitBlockxoyo< double > (
    Param XParam,
    Forcing < float > XForcing,
    BlockP < double > & XBlockP
) 
```




<hr>



### function InitBlockxoyo&lt; float &gt; 

```C++
template void InitBlockxoyo< float > (
    Param XParam,
    Forcing < float > XForcing,
    BlockP < float > & XBlock
) 
```




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


* `XParam` [**Model**](structModel.md) parameters 
* `XForcing` [**Forcing**](structForcing.md) data 
* `XModel` [**Model**](structModel.md) structure to hold mesh and block data

Allocates memory, initializes block info, adaptation info, and boundary masks. 


        

<hr>



### function InitMesh&lt; double &gt; 

```C++
template void InitMesh< double > (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < double > & XModel
) 
```




<hr>



### function InitMesh&lt; float &gt; 

```C++
template void InitMesh< float > (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < float > & XModel
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Mesh.cu`




# File Mesh.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Mesh.cu**](Mesh_8cu.md)

[Go to the source code of this file](Mesh_8cu_source.md)



* `#include "Mesh.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**CalcInitnblk**](#function-calcinitnblk) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing) <br> |
|  int | [**CalcMaskblk**](#function-calcmaskblk) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock) <br> |
|  template int | [**CalcMaskblk&lt; double &gt;**](#function-calcmaskblk-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock) <br> |
|  template int | [**CalcMaskblk&lt; float &gt;**](#function-calcmaskblk-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock) <br> |
|  void | [**FindMaskblk**](#function-findmaskblk) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  template void | [**FindMaskblk&lt; double &gt;**](#function-findmaskblk-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; & XBlock) <br> |
|  template void | [**FindMaskblk&lt; float &gt;**](#function-findmaskblk-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; & XBlock) <br> |
|  void | [**InitBlockInfo**](#function-initblockinfo) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  void | [**InitBlockadapt**](#function-initblockadapt) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**AdaptP**](structAdaptP.md) & XAdap) <br> |
|  template void | [**InitBlockadapt&lt; double &gt;**](#function-initblockadapt-double) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**AdaptP**](structAdaptP.md) & XAdap) <br> |
|  template void | [**InitBlockadapt&lt; float &gt;**](#function-initblockadapt-float) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**AdaptP**](structAdaptP.md) & XAdap) <br> |
|  void | [**InitBlockneighbours**](#function-initblockneighbours) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  template void | [**InitBlockneighbours&lt; double &gt;**](#function-initblockneighbours-double) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; double &gt; & XBlock) <br> |
|  template void | [**InitBlockneighbours&lt; float &gt;**](#function-initblockneighbours-float) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; float &gt; & XBlock) <br> |
|  void | [**InitBlockxoyo**](#function-initblockxoyo) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  template void | [**InitBlockxoyo&lt; double &gt;**](#function-initblockxoyo-double) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**BlockP**](structBlockP.md)&lt; double &gt; & XBlockP) <br> |
|  template void | [**InitBlockxoyo&lt; float &gt;**](#function-initblockxoyo-float) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**BlockP**](structBlockP.md)&lt; float &gt; & XBlock) <br> |
|  void | [**InitMesh**](#function-initmesh) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  template void | [**InitMesh&lt; double &gt;**](#function-initmesh-double) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template void | [**InitMesh&lt; float &gt;**](#function-initmesh-float) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; float &gt; & XModel) <br> |




























## Public Functions Documentation




### function CalcInitnblk 

```C++
int CalcInitnblk (
    Param XParam,
    Forcing < float > XForcing
) 
```




<hr>



### function CalcMaskblk 

```C++
template<class T>
int CalcMaskblk (
    Param XParam,
    BlockP < T > XBlock
) 
```




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

```C++
template<class T>
void FindMaskblk (
    Param XParam,
    BlockP < T > & XBlock
) 
```




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

```C++
template<class T>
void InitBlockInfo (
    Param & XParam,
    Forcing < float > & XForcing,
    BlockP < T > & XBlock
) 
```




<hr>



### function InitBlockadapt 

```C++
template<class T>
void InitBlockadapt (
    Param & XParam,
    BlockP < T > XBlock,
    AdaptP & XAdap
) 
```




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

```C++
template<class T>
void InitBlockneighbours (
    Param & XParam,
    Forcing < float > & XForcing,
    BlockP < T > & XBlock
) 
```




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

```C++
template<class T>
void InitBlockxoyo (
    Param XParam,
    Forcing < float > XForcing,
    BlockP < T > & XBlock
) 
```




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

```C++
template<class T>
void InitMesh (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < T > & XModel
) 
```




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


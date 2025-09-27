

# File Mesh.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Mesh.h**](Mesh_8h.md)

[Go to the source code of this file](Mesh_8h_source.md)



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
|  int | [**CalcInitnblk**](#function-calcinitnblk) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing) <br> |
|  int | [**CalcMaskblk**](#function-calcmaskblk) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock) <br> |
|  void | [**FindMaskblk**](#function-findmaskblk) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  void | [**InitBlockInfo**](#function-initblockinfo) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  void | [**InitBlockadapt**](#function-initblockadapt) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**AdaptP**](structAdaptP.md) & XAdap) <br> |
|  void | [**InitBlockneighbours**](#function-initblockneighbours) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  void | [**InitBlockxoyo**](#function-initblockxoyo) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  void | [**InitMesh**](#function-initmesh) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |




























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



### function FindMaskblk 

```C++
template<class T>
void FindMaskblk (
    Param XParam,
    BlockP < T > & XBlock
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

------------------------------
The documentation for this class was generated from the following file `src/Mesh.h`


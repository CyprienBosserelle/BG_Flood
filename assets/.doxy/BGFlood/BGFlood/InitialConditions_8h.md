

# File InitialConditions.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**InitialConditions.h**](InitialConditions_8h.md)

[Go to the source code of this file](InitialConditions_8h_source.md)



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
|  void | [**Calcbndblks**](#function-calcbndblks) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock) <br> |
|  void | [**FindTSoutNodes**](#function-findtsoutnodes) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**BndblockP**](structBndblockP.md)&lt; T &gt; & bnd) <br> |
|  void | [**Findbndblks**](#function-findbndblks) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing) <br> |
|  void | [**InitRivers**](#function-initrivers) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  void | [**InitTSOutput**](#function-inittsoutput) ([**Param**](classParam.md) XParam) <br> |
|  void | [**InitialConditions**](#function-initialconditions) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  void | [**Initmaparray**](#function-initmaparray) ([**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  void | [**Initoutzone**](#function-initoutzone) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  void | [**InitzbgradientCPU**](#function-initzbgradientcpu) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  void | [**InitzbgradientGPU**](#function-initzbgradientgpu) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  void | [**calcactiveCellCPU**](#function-calcactivecellcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, T \* zb) <br> |
|  void | [**initOutputTimes**](#function-initoutputtimes) ([**Param**](classParam.md) XParam, std::vector&lt; double &gt; & OutputT, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  void | [**initoutput**](#function-initoutput) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |




























## Public Functions Documentation




### function Calcbndblks 

```C++
template<class T>
void Calcbndblks (
    Param & XParam,
    Forcing < float > & XForcing,
    BlockP < T > XBlock
) 
```




<hr>



### function FindTSoutNodes 

```C++
template<class T>
void FindTSoutNodes (
    Param & XParam,
    BlockP < T > XBlock,
    BndblockP < T > & bnd
) 
```




<hr>



### function Findbndblks 

```C++
template<class T>
void Findbndblks (
    Param XParam,
    Model < T > XModel,
    Forcing < float > & XForcing
) 
```



Find which block on the model edge belongs to a "side boundary" 


        

<hr>



### function InitRivers 

```C++
template<class T>
void InitRivers (
    Param XParam,
    Forcing < float > & XForcing,
    Model < T > & XModel
) 
```




<hr>



### function InitTSOutput 

```C++
void InitTSOutput (
    Param XParam
) 
```




<hr>



### function InitialConditions 

```C++
template<class T>
void InitialConditions (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < T > & XModel
) 
```




<hr>



### function Initmaparray 

```C++
template<class T>
void Initmaparray (
    Model < T > & XModel
) 
```




<hr>



### function Initoutzone 

```C++
template<class T>
void Initoutzone (
    Param & XParam,
    BlockP < T > & XBlock
) 
```




<hr>



### function InitzbgradientCPU 

```C++
template<class T>
void InitzbgradientCPU (
    Param XParam,
    Model < T > XModel
) 
```




<hr>



### function InitzbgradientGPU 

```C++
template<class T>
void InitzbgradientGPU (
    Param XParam,
    Model < T > XModel
) 
```




<hr>



### function calcactiveCellCPU 

```C++
template<class T>
void calcactiveCellCPU (
    Param XParam,
    BlockP < T > XBlock,
    Forcing < float > & XForcing,
    T * zb
) 
```




<hr>



### function initOutputTimes 

```C++
template<class T>
void initOutputTimes (
    Param XParam,
    std::vector< double > & OutputT,
    BlockP < T > & XBlock
) 
```




<hr>



### function initoutput 

```C++
template<class T>
void initoutput (
    Param & XParam,
    Model < T > & XModel
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/InitialConditions.h`


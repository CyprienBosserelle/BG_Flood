

# File AdaptCriteria.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**AdaptCriteria.cu**](AdaptCriteria_8cu.md)

[Go to the source code of this file](AdaptCriteria_8cu_source.md)



* `#include "AdaptCriteria.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**AdaptCriteria**](#function-adaptcriteria) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  template int | [**AdaptCriteria&lt; double &gt;**](#function-adaptcriteria-double) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template int | [**AdaptCriteria&lt; float &gt;**](#function-adaptcriteria-float) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |
|  int | [**Thresholdcriteria**](#function-thresholdcriteria) ([**Param**](classParam.md) XParam, T threshold, T \* z, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  template int | [**Thresholdcriteria&lt; double &gt;**](#function-thresholdcriteria-double) ([**Param**](classParam.md) XParam, double threshold, double \* z, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  template int | [**Thresholdcriteria&lt; float &gt;**](#function-thresholdcriteria-float) ([**Param**](classParam.md) XParam, float threshold, float \* z, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  int | [**inrangecriteria**](#function-inrangecriteria) ([**Param**](classParam.md) XParam, T zmin, T zmax, T \* z, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  template int | [**inrangecriteria&lt; double &gt;**](#function-inrangecriteria-double) ([**Param**](classParam.md) XParam, double zmin, double zmax, double \* z, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  template int | [**inrangecriteria&lt; float &gt;**](#function-inrangecriteria-float) ([**Param**](classParam.md) XParam, float zmin, float zmax, float \* z, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  int | [**targetlevelcriteria**](#function-targetlevelcriteria) ([**Param**](classParam.md) XParam, [**StaticForcingP**](structStaticForcingP.md)&lt; int &gt; targetlevelmap, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  template int | [**targetlevelcriteria&lt; double &gt;**](#function-targetlevelcriteria-double) ([**Param**](classParam.md) XParam, [**StaticForcingP**](structStaticForcingP.md)&lt; int &gt; targetlevelmap, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  template int | [**targetlevelcriteria&lt; float &gt;**](#function-targetlevelcriteria-float) ([**Param**](classParam.md) XParam, [**StaticForcingP**](structStaticForcingP.md)&lt; int &gt; targetlevelmap, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, bool \* refine, bool \* coarsen) <br> |




























## Public Functions Documentation




### function AdaptCriteria 

```C++
template<class T>
int AdaptCriteria (
    Param XParam,
    Forcing < float > XForcing,
    Model < T > XModel
) 
```




<hr>



### function AdaptCriteria&lt; double &gt; 

```C++
template int AdaptCriteria< double > (
    Param XParam,
    Forcing < float > XForcing,
    Model < double > XModel
) 
```




<hr>



### function AdaptCriteria&lt; float &gt; 

```C++
template int AdaptCriteria< float > (
    Param XParam,
    Forcing < float > XForcing,
    Model < float > XModel
) 
```




<hr>



### function Thresholdcriteria 

```C++
template<class T>
int Thresholdcriteria (
    Param XParam,
    T threshold,
    T * z,
    BlockP < T > XBlock,
    bool * refine,
    bool * coarsen
) 
```




<hr>



### function Thresholdcriteria&lt; double &gt; 

```C++
template int Thresholdcriteria< double > (
    Param XParam,
    double threshold,
    double * z,
    BlockP < double > XBlock,
    bool * refine,
    bool * coarsen
) 
```




<hr>



### function Thresholdcriteria&lt; float &gt; 

```C++
template int Thresholdcriteria< float > (
    Param XParam,
    float threshold,
    float * z,
    BlockP < float > XBlock,
    bool * refine,
    bool * coarsen
) 
```




<hr>



### function inrangecriteria 

```C++
template<class T>
int inrangecriteria (
    Param XParam,
    T zmin,
    T zmax,
    T * z,
    BlockP < T > XBlock,
    bool * refine,
    bool * coarsen
) 
```




<hr>



### function inrangecriteria&lt; double &gt; 

```C++
template int inrangecriteria< double > (
    Param XParam,
    double zmin,
    double zmax,
    double * z,
    BlockP < double > XBlock,
    bool * refine,
    bool * coarsen
) 
```




<hr>



### function inrangecriteria&lt; float &gt; 

```C++
template int inrangecriteria< float > (
    Param XParam,
    float zmin,
    float zmax,
    float * z,
    BlockP < float > XBlock,
    bool * refine,
    bool * coarsen
) 
```




<hr>



### function targetlevelcriteria 

```C++
template<class T>
int targetlevelcriteria (
    Param XParam,
    StaticForcingP < int > targetlevelmap,
    BlockP < T > XBlock,
    bool * refine,
    bool * coarsen
) 
```




<hr>



### function targetlevelcriteria&lt; double &gt; 

```C++
template int targetlevelcriteria< double > (
    Param XParam,
    StaticForcingP < int > targetlevelmap,
    BlockP < double > XBlock,
    bool * refine,
    bool * coarsen
) 
```




<hr>



### function targetlevelcriteria&lt; float &gt; 

```C++
template int targetlevelcriteria< float > (
    Param XParam,
    StaticForcingP < int > targetlevelmap,
    BlockP < float > XBlock,
    bool * refine,
    bool * coarsen
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/AdaptCriteria.cu`


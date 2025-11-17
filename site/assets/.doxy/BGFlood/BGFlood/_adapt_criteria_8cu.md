

# File AdaptCriteria.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**AdaptCriteria.cu**](_adapt_criteria_8cu.md)

[Go to the source code of this file](_adapt_criteria_8cu_source.md)



* `#include "AdaptCriteria.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**AdaptCriteria**](#function-adaptcriteria) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Selects and applies the adaptation criteria for mesh refinement/coarsening._  |
|  template int | [**AdaptCriteria&lt; double &gt;**](#function-adaptcriteria-double) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; double &gt; XModel) <br> |
|  template int | [**AdaptCriteria&lt; float &gt;**](#function-adaptcriteria-float) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; float &gt; XModel) <br> |
|  int | [**Thresholdcriteria**](#function-thresholdcriteria) ([**Param**](class_param.md) XParam, T threshold, T \* z, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, bool \* refine, bool \* coarsen) <br>_Applies threshold-based adaptation criteria for mesh refinement/coarsening._  |
|  template int | [**Thresholdcriteria&lt; double &gt;**](#function-thresholdcriteria-double) ([**Param**](class_param.md) XParam, double threshold, double \* z, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  template int | [**Thresholdcriteria&lt; float &gt;**](#function-thresholdcriteria-float) ([**Param**](class_param.md) XParam, float threshold, float \* z, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  int | [**inrangecriteria**](#function-inrangecriteria) ([**Param**](class_param.md) XParam, T zmin, T zmax, T \* z, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, bool \* refine, bool \* coarsen) <br>_Applies in-range adaptation criteria for mesh refinement/coarsening._  |
|  template int | [**inrangecriteria&lt; double &gt;**](#function-inrangecriteria-double) ([**Param**](class_param.md) XParam, double zmin, double zmax, double \* z, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  template int | [**inrangecriteria&lt; float &gt;**](#function-inrangecriteria-float) ([**Param**](class_param.md) XParam, float zmin, float zmax, float \* z, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  int | [**targetlevelcriteria**](#function-targetlevelcriteria) ([**Param**](class_param.md) XParam, [**StaticForcingP**](struct_static_forcing_p.md)&lt; int &gt; targetlevelmap, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, bool \* refine, bool \* coarsen) <br>_Applies target level adaptation criteria for mesh refinement/coarsening._  |
|  template int | [**targetlevelcriteria&lt; double &gt;**](#function-targetlevelcriteria-double) ([**Param**](class_param.md) XParam, [**StaticForcingP**](struct_static_forcing_p.md)&lt; int &gt; targetlevelmap, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  template int | [**targetlevelcriteria&lt; float &gt;**](#function-targetlevelcriteria-float) ([**Param**](class_param.md) XParam, [**StaticForcingP**](struct_static_forcing_p.md)&lt; int &gt; targetlevelmap, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, bool \* refine, bool \* coarsen) <br> |




























## Public Functions Documentation




### function AdaptCriteria 

_Selects and applies the adaptation criteria for mesh refinement/coarsening._ 
```C++
template<class T>
int AdaptCriteria (
    Param XParam,
    Forcing < float > XForcing,
    Model < T > XModel
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data 
* `XModel` [**Model**](struct_model.md) structure 



**Returns:**

Success flag (0 or 1)


This function chooses the adaptation method ("Threshold", "Inrange", "Targetlevel") based on XParam.AdaptCrit, and applies it to the mesh using the corresponding criteria function. It sets the refine/coarsen flags for each block according to the selected method and arguments. For "Targetlevel", it loops over all target adaptation maps. 


        

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

_Applies threshold-based adaptation criteria for mesh refinement/coarsening._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `threshold` Threshold value for adaptation 
* `z` Array of variable values (e.g., water depth) 
* `XBlock` Block data structure 
* `refine` Array of refinement flags 
* `coarsen` Array of coarsening flags 



**Returns:**

Success flag (0 or 1)


Refines blocks where any cell value exceeds the threshold, coarsens otherwise.


Threshold criteria is a general form of wet dry criteria. Simple wet/dry refining criteria. If the block is wet -&gt; refine is true. If the block is dry -&gt; coarsen is true. 

**Warning:**

the refinement sanity check is meant to be done after running this function. 





        

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

_Applies in-range adaptation criteria for mesh refinement/coarsening._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `zmin` Minimum value for refinement 
* `zmax` Maximum value for refinement 
* `z` Array of variable values 
* `XBlock` Block data structure 
* `refine` Array of refinement flags 
* `coarsen` Array of coarsening flags 



**Returns:**

Success flag (0 or 1)


Refines blocks where any cell value is within [zmin, zmax], coarsens otherwise.


Simple in-range refining criteria. If any value of z (could be any variable) is zmin &lt;= z &lt;= zmax the block will try to refine. Otherwise, the block will try to coarsen. 

**Warning:**

the refinement sanity check is meant to be done after running this function. 





        

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

_Applies target level adaptation criteria for mesh refinement/coarsening._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `targetlevelmap` Map of target levels for adaptation 
* `XBlock` Block data structure 
* `refine` Array of refinement flags 
* `coarsen` Array of coarsening flags 



**Returns:**

Success flag (0 or 1)


Refines blocks where target level is greater than current, coarsens if equal or lower. 


        

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




# File AdaptCriteria.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**AdaptCriteria.h**](_adapt_criteria_8h.md)

[Go to the source code of this file](_adapt_criteria_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Write_txtlog.h"`
* `#include "Util_CPU.h"`
* `#include "Arrays.h"`
* `#include "Mesh.h"`
* `#include "Halo.h"`
* `#include "GridManip.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**AdaptCriteria**](#function-adaptcriteria) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Selects and applies the adaptation criteria for mesh refinement/coarsening._  |
|  int | [**Thresholdcriteria**](#function-thresholdcriteria) ([**Param**](class_param.md) XParam, T threshold, T \* z, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, bool \* refine, bool \* coarsen) <br>_Applies threshold-based adaptation criteria for mesh refinement/coarsening._  |
|  int | [**inrangecriteria**](#function-inrangecriteria) ([**Param**](class_param.md) XParam, T zmin, T zmax, T \* z, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, bool \* refine, bool \* coarsen) <br>_Applies in-range adaptation criteria for mesh refinement/coarsening._  |




























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

------------------------------
The documentation for this class was generated from the following file `src/AdaptCriteria.h`


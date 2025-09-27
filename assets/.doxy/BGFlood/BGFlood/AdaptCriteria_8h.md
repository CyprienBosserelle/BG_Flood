

# File AdaptCriteria.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**AdaptCriteria.h**](AdaptCriteria_8h.md)

[Go to the source code of this file](AdaptCriteria_8h_source.md)



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
|  int | [**AdaptCriteria**](#function-adaptcriteria) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  int | [**Thresholdcriteria**](#function-thresholdcriteria) ([**Param**](classParam.md) XParam, T threshold, T \* z, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, bool \* refine, bool \* coarsen) <br> |
|  int | [**inrangecriteria**](#function-inrangecriteria) ([**Param**](classParam.md) XParam, T zmin, T zmax, T \* z, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, bool \* refine, bool \* coarsen) <br> |




























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

------------------------------
The documentation for this class was generated from the following file `src/AdaptCriteria.h`




# File Meanmax.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Meanmax.h**](Meanmax_8h.md)

[Go to the source code of this file](Meanmax_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "MemManagement.h"`
* `#include "FlowGPU.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Calcmeanmax**](#function-calcmeanmax) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  void | [**Initmeanmax**](#function-initmeanmax) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  \_\_global\_\_ void | [**addUandhU\_GPU**](#function-adduandhu_gpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* u, T \* v, T \* U, T \* hU) <br> |
|  \_\_global\_\_ void | [**addavg\_varGPU**](#function-addavg_vargpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Varmean, T \* Var) <br> |
|  \_\_global\_\_ void | [**addwettime\_GPU**](#function-addwettime_gpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* wett, T \* h, T thresold, T time) <br> |
|  \_\_global\_\_ void | [**divavg\_varGPU**](#function-divavg_vargpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T ntdiv, T \* Varmean) <br> |
|  \_\_global\_\_ void | [**max\_Norm\_GPU**](#function-max_norm_gpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var1, T \* Var2) <br> |
|  \_\_global\_\_ void | [**max\_hU\_GPU**](#function-max_hu_gpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var1, T \* Var2, T \* Var3) <br> |
|  \_\_global\_\_ void | [**max\_varGPU**](#function-max_vargpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var) <br> |
|  void | [**resetmeanmax**](#function-resetmeanmax) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  void | [**resetvalGPU**](#function-resetvalgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& var, T val) <br> |




























## Public Functions Documentation




### function Calcmeanmax 

```C++
template<class T>
void Calcmeanmax (
    Param XParam,
    Loop < T > & XLoop,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>



### function Initmeanmax 

```C++
template<class T>
void Initmeanmax (
    Param XParam,
    Loop < T > XLoop,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>



### function addUandhU\_GPU 

```C++
template<class T>
__global__ void addUandhU_GPU (
    Param XParam,
    BlockP < T > XBlock,
    T * h,
    T * u,
    T * v,
    T * U,
    T * hU
) 
```




<hr>



### function addavg\_varGPU 

```C++
template<class T>
__global__ void addavg_varGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmean,
    T * Var
) 
```




<hr>



### function addwettime\_GPU 

```C++
template<class T>
__global__ void addwettime_GPU (
    Param XParam,
    BlockP < T > XBlock,
    T * wett,
    T * h,
    T thresold,
    T time
) 
```




<hr>



### function divavg\_varGPU 

```C++
template<class T>
__global__ void divavg_varGPU (
    Param XParam,
    BlockP < T > XBlock,
    T ntdiv,
    T * Varmean
) 
```




<hr>



### function max\_Norm\_GPU 

```C++
template<class T>
__global__ void max_Norm_GPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * Var1,
    T * Var2
) 
```




<hr>



### function max\_hU\_GPU 

```C++
template<class T>
__global__ void max_hU_GPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * Var1,
    T * Var2,
    T * Var3
) 
```




<hr>



### function max\_varGPU 

```C++
template<class T>
__global__ void max_varGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * Var
) 
```




<hr>



### function resetmeanmax 

```C++
template<class T>
void resetmeanmax (
    Param XParam,
    Loop < T > & XLoop,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>



### function resetvalGPU 

```C++
template<class T>
void resetvalGPU (
    Param XParam,
    BlockP < T > XBlock,
    T *& var,
    T val
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Meanmax.h`




# File Meanmax.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Meanmax.cu**](Meanmax_8cu.md)

[Go to the source code of this file](Meanmax_8cu_source.md)



* `#include "Meanmax.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Calcmeanmax**](#function-calcmeanmax) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  template void | [**Calcmeanmax&lt; double &gt;**](#function-calcmeanmax-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; & XLoop, [**Model**](structModel.md)&lt; double &gt; XModel, [**Model**](structModel.md)&lt; double &gt; XModel\_g) <br> |
|  template void | [**Calcmeanmax&lt; float &gt;**](#function-calcmeanmax-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; & XLoop, [**Model**](structModel.md)&lt; float &gt; XModel, [**Model**](structModel.md)&lt; float &gt; XModel\_g) <br> |
|  void | [**Initmeanmax**](#function-initmeanmax) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  template void | [**Initmeanmax&lt; double &gt;**](#function-initmeanmax-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**Model**](structModel.md)&lt; double &gt; XModel, [**Model**](structModel.md)&lt; double &gt; XModel\_g) <br> |
|  template void | [**Initmeanmax&lt; float &gt;**](#function-initmeanmax-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**Model**](structModel.md)&lt; float &gt; XModel, [**Model**](structModel.md)&lt; float &gt; XModel\_g) <br> |
|  \_\_host\_\_ void | [**addUandhU\_CPU**](#function-adduandhu_cpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* u, T \* v, T \* U, T \* hU) <br> |
|  \_\_global\_\_ void | [**addUandhU\_GPU**](#function-adduandhu_gpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* u, T \* v, T \* U, T \* hU) <br> |
|  \_\_host\_\_ void | [**addavg\_varCPU**](#function-addavg_varcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Varmean, T \* Var) <br> |
|  \_\_global\_\_ void | [**addavg\_varGPU**](#function-addavg_vargpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Varmean, T \* Var) <br> |
|  \_\_host\_\_ void | [**addwettime\_CPU**](#function-addwettime_cpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* wett, T \* h, T thresold, T time) <br> |
|  \_\_global\_\_ void | [**addwettime\_GPU**](#function-addwettime_gpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* wett, T \* h, T thresold, T time) <br> |
|  \_\_host\_\_ void | [**divavg\_varCPU**](#function-divavg_varcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T ntdiv, T \* Varmean) <br> |
|  \_\_global\_\_ void | [**divavg\_varGPU**](#function-divavg_vargpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T ntdiv, T \* Varmean) <br> |
|  \_\_host\_\_ void | [**max\_Norm\_CPU**](#function-max_norm_cpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var1, T \* Var2) <br> |
|  \_\_global\_\_ void | [**max\_Norm\_GPU**](#function-max_norm_gpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var1, T \* Var2) <br> |
|  \_\_host\_\_ void | [**max\_hU\_CPU**](#function-max_hu_cpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Varmax, T \* h, T \* u, T \* v) <br> |
|  \_\_global\_\_ void | [**max\_hU\_GPU**](#function-max_hu_gpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Varmax, T \* h, T \* u, T \* v) <br> |
|  \_\_host\_\_ void | [**max\_varCPU**](#function-max_varcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var) <br> |
|  \_\_global\_\_ void | [**max\_varGPU**](#function-max_vargpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var) <br> |
|  void | [**resetmaxCPU**](#function-resetmaxcpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; & XEv) <br> |
|  void | [**resetmaxGPU**](#function-resetmaxgpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; & XEv) <br> |
|  void | [**resetmeanCPU**](#function-resetmeancpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; & XEv) <br> |
|  template void | [**resetmeanCPU&lt; double &gt;**](#function-resetmeancpu-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; double &gt; & XEv) <br> |
|  template void | [**resetmeanCPU&lt; float &gt;**](#function-resetmeancpu-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; float &gt; & XEv) <br> |
|  void | [**resetmeanGPU**](#function-resetmeangpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; & XEv) <br> |
|  template void | [**resetmeanGPU&lt; double &gt;**](#function-resetmeangpu-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; double &gt; & XEv) <br> |
|  template void | [**resetmeanGPU&lt; float &gt;**](#function-resetmeangpu-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; float &gt; & XEv) <br> |
|  void | [**resetmeanmax**](#function-resetmeanmax) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  template void | [**resetmeanmax&lt; double &gt;**](#function-resetmeanmax-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; & XLoop, [**Model**](structModel.md)&lt; double &gt; XModel, [**Model**](structModel.md)&lt; double &gt; XModel\_g) <br> |
|  template void | [**resetmeanmax&lt; float &gt;**](#function-resetmeanmax-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; & XLoop, [**Model**](structModel.md)&lt; float &gt; XModel, [**Model**](structModel.md)&lt; float &gt; XModel\_g) <br> |
|  void | [**resetvalCPU**](#function-resetvalcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& var, T val) <br> |
|  template void | [**resetvalCPU&lt; double &gt;**](#function-resetvalcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \*& var, double val) <br> |
|  template void | [**resetvalCPU&lt; float &gt;**](#function-resetvalcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \*& var, float val) <br> |
|  void | [**resetvalGPU**](#function-resetvalgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& var, T val) <br> |
|  template void | [**resetvalGPU&lt; double &gt;**](#function-resetvalgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \*& var, double val) <br> |
|  template void | [**resetvalGPU&lt; float &gt;**](#function-resetvalgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \*& var, float val) <br> |




























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



### function Calcmeanmax&lt; double &gt; 

```C++
template void Calcmeanmax< double > (
    Param XParam,
    Loop < double > & XLoop,
    Model < double > XModel,
    Model < double > XModel_g
) 
```




<hr>



### function Calcmeanmax&lt; float &gt; 

```C++
template void Calcmeanmax< float > (
    Param XParam,
    Loop < float > & XLoop,
    Model < float > XModel,
    Model < float > XModel_g
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



### function Initmeanmax&lt; double &gt; 

```C++
template void Initmeanmax< double > (
    Param XParam,
    Loop < double > XLoop,
    Model < double > XModel,
    Model < double > XModel_g
) 
```




<hr>



### function Initmeanmax&lt; float &gt; 

```C++
template void Initmeanmax< float > (
    Param XParam,
    Loop < float > XLoop,
    Model < float > XModel,
    Model < float > XModel_g
) 
```




<hr>



### function addUandhU\_CPU 

```C++
template<class T>
__host__ void addUandhU_CPU (
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



### function addavg\_varCPU 

```C++
template<class T>
__host__ void addavg_varCPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmean,
    T * Var
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



### function addwettime\_CPU 

```C++
template<class T>
__host__ void addwettime_CPU (
    Param XParam,
    BlockP < T > XBlock,
    T * wett,
    T * h,
    T thresold,
    T time
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



### function divavg\_varCPU 

```C++
template<class T>
__host__ void divavg_varCPU (
    Param XParam,
    BlockP < T > XBlock,
    T ntdiv,
    T * Varmean
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



### function max\_Norm\_CPU 

```C++
template<class T>
__host__ void max_Norm_CPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * Var1,
    T * Var2
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



### function max\_hU\_CPU 

```C++
template<class T>
__host__ void max_hU_CPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * h,
    T * u,
    T * v
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
    T * h,
    T * u,
    T * v
) 
```




<hr>



### function max\_varCPU 

```C++
template<class T>
__host__ void max_varCPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * Var
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



### function resetmaxCPU 

```C++
template<class T>
void resetmaxCPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    EvolvingP_M < T > & XEv
) 
```




<hr>



### function resetmaxGPU 

```C++
template<class T>
void resetmaxGPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    EvolvingP_M < T > & XEv
) 
```




<hr>



### function resetmeanCPU 

```C++
template<class T>
void resetmeanCPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    EvolvingP_M < T > & XEv
) 
```




<hr>



### function resetmeanCPU&lt; double &gt; 

```C++
template void resetmeanCPU< double > (
    Param XParam,
    Loop < double > XLoop,
    BlockP < double > XBlock,
    EvolvingP_M < double > & XEv
) 
```




<hr>



### function resetmeanCPU&lt; float &gt; 

```C++
template void resetmeanCPU< float > (
    Param XParam,
    Loop < float > XLoop,
    BlockP < float > XBlock,
    EvolvingP_M < float > & XEv
) 
```




<hr>



### function resetmeanGPU 

```C++
template<class T>
void resetmeanGPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    EvolvingP_M < T > & XEv
) 
```




<hr>



### function resetmeanGPU&lt; double &gt; 

```C++
template void resetmeanGPU< double > (
    Param XParam,
    Loop < double > XLoop,
    BlockP < double > XBlock,
    EvolvingP_M < double > & XEv
) 
```




<hr>



### function resetmeanGPU&lt; float &gt; 

```C++
template void resetmeanGPU< float > (
    Param XParam,
    Loop < float > XLoop,
    BlockP < float > XBlock,
    EvolvingP_M < float > & XEv
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



### function resetmeanmax&lt; double &gt; 

```C++
template void resetmeanmax< double > (
    Param XParam,
    Loop < double > & XLoop,
    Model < double > XModel,
    Model < double > XModel_g
) 
```




<hr>



### function resetmeanmax&lt; float &gt; 

```C++
template void resetmeanmax< float > (
    Param XParam,
    Loop < float > & XLoop,
    Model < float > XModel,
    Model < float > XModel_g
) 
```




<hr>



### function resetvalCPU 

```C++
template<class T>
void resetvalCPU (
    Param XParam,
    BlockP < T > XBlock,
    T *& var,
    T val
) 
```




<hr>



### function resetvalCPU&lt; double &gt; 

```C++
template void resetvalCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double *& var,
    double val
) 
```




<hr>



### function resetvalCPU&lt; float &gt; 

```C++
template void resetvalCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float *& var,
    float val
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



### function resetvalGPU&lt; double &gt; 

```C++
template void resetvalGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double *& var,
    double val
) 
```




<hr>



### function resetvalGPU&lt; float &gt; 

```C++
template void resetvalGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float *& var,
    float val
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Meanmax.cu`


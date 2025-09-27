

# File Gradients.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Gradients.h**](Gradients_8h.md)

[Go to the source code of this file](Gradients_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "Util_CPU.h"`
* `#include "Setup_GPU.h"`
* `#include "MemManagement.h"`
* `#include "Halo.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_global\_\_ void | [**WetsloperesetHaloBotGPU**](#function-wetsloperesethalobotgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetsloperesetHaloLeftGPU**](#function-wetsloperesethaloleftgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetsloperesetHaloRightGPU**](#function-wetsloperesethalorightgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetsloperesetHaloTopGPU**](#function-wetsloperesethalotopgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetsloperesetXGPU**](#function-wetsloperesetxgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetsloperesetYGPU**](#function-wetsloperesetygpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**gradient**](#function-gradient) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br> |
|  void | [**gradientC**](#function-gradientc) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  void | [**gradientCPU**](#function-gradientcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  void | [**gradientGPU**](#function-gradientgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  void | [**gradientGPUnew**](#function-gradientgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  void | [**gradientHalo**](#function-gradienthalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloBotGPU**](#function-gradienthalobotgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloBotGPUnew**](#function-gradienthalobotgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  void | [**gradientHaloGPU**](#function-gradienthalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloLeftGPU**](#function-gradienthaloleftgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloLeftGPUnew**](#function-gradienthaloleftgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloRightGPU**](#function-gradienthalorightgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloRightGPUnew**](#function-gradienthalorightgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloTopGPU**](#function-gradienthalotopgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloTopGPUnew**](#function-gradienthalotopgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientSM**](#function-gradientsm) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientSMB**](#function-gradientsmb) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientSMC**](#function-gradientsmc) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientedgeX**](#function-gradientedgex) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx) <br> |
|  \_\_global\_\_ void | [**gradientedgeY**](#function-gradientedgey) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dady) <br> |




























## Public Functions Documentation




### function WetsloperesetHaloBotGPU 

```C++
template<class T>
__global__ void WetsloperesetHaloBotGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    T * zb
) 
```




<hr>



### function WetsloperesetHaloLeftGPU 

```C++
template<class T>
__global__ void WetsloperesetHaloLeftGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    T * zb
) 
```




<hr>



### function WetsloperesetHaloRightGPU 

```C++
template<class T>
__global__ void WetsloperesetHaloRightGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    T * zb
) 
```




<hr>



### function WetsloperesetHaloTopGPU 

```C++
template<class T>
__global__ void WetsloperesetHaloTopGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    T * zb
) 
```




<hr>



### function WetsloperesetXGPU 

```C++
template<class T>
__global__ void WetsloperesetXGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    T * zb
) 
```




<hr>



### function WetsloperesetYGPU 

```C++
template<class T>
__global__ void WetsloperesetYGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    T * zb
) 
```




<hr>



### function gradient 

```C++
template<class T>
__global__ void gradient (
    int halowidth,
    int * active,
    int * level,
    T theta,
    T dx,
    T * a,
    T * dadx,
    T * dady
) 
```



Device kernel for calculating grdients for an evolving poarameter using the minmod limiter 


        

<hr>



### function gradientC 

```C++
template<class T>
void gradientC (
    Param XParam,
    BlockP < T > XBlock,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientCPU 

```C++
template<class T>
void gradientCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    T * zb
) 
```




<hr>



### function gradientGPU 

```C++
template<class T>
void gradientGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    T * zb
) 
```



Wrapping function to calculate gradien of evolving variables on GPU This function is the entry point to the gradient functions on the GPU 


        

<hr>



### function gradientGPUnew 

```C++
template<class T>
void gradientGPUnew (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    T * zb
) 
```




<hr>



### function gradientHalo 

```C++
template<class T>
void gradientHalo (
    Param XParam,
    BlockP < T > XBlock,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientHaloBotGPU 

```C++
template<class T>
__global__ void gradientHaloBotGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientHaloBotGPUnew 

```C++
template<class T>
__global__ void gradientHaloBotGPUnew (
    Param XParam,
    BlockP < T > XBlock,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientHaloGPU 

```C++
template<class T>
void gradientHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientHaloLeftGPU 

```C++
template<class T>
__global__ void gradientHaloLeftGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientHaloLeftGPUnew 

```C++
template<class T>
__global__ void gradientHaloLeftGPUnew (
    Param XParam,
    BlockP < T > XBlock,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientHaloRightGPU 

```C++
template<class T>
__global__ void gradientHaloRightGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientHaloRightGPUnew 

```C++
template<class T>
__global__ void gradientHaloRightGPUnew (
    Param XParam,
    BlockP < T > XBlock,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientHaloTopGPU 

```C++
template<class T>
__global__ void gradientHaloTopGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientHaloTopGPUnew 

```C++
template<class T>
__global__ void gradientHaloTopGPUnew (
    Param XParam,
    BlockP < T > XBlock,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientSM 

```C++
template<class T>
__global__ void gradientSM (
    int halowidth,
    int * active,
    int * level,
    T theta,
    T dx,
    T * a,
    T * dadx,
    T * dady
) 
```



Depreciated shared memory version of Device kernel for calculating gradients Much slower than above 


        

<hr>



### function gradientSMB 

```C++
template<class T>
__global__ void gradientSMB (
    int halowidth,
    int * active,
    int * level,
    T theta,
    T dx,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientSMC 

```C++
template<class T>
__global__ void gradientSMC (
    int halowidth,
    int * active,
    int * level,
    T theta,
    T dx,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientedgeX 

```C++
template<class T>
__global__ void gradientedgeX (
    int halowidth,
    int * active,
    int * level,
    T theta,
    T dx,
    T * a,
    T * dadx
) 
```




<hr>



### function gradientedgeY 

```C++
template<class T>
__global__ void gradientedgeY (
    int halowidth,
    int * active,
    int * level,
    T theta,
    T dx,
    T * a,
    T * dady
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Gradients.h`


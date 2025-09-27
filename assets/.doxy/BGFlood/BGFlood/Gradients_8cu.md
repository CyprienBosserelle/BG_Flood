

# File Gradients.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Gradients.cu**](Gradients_8cu.md)

[Go to the source code of this file](Gradients_8cu_source.md)



* `#include "Gradients.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**WetsloperesetCPU**](#function-wetsloperesetcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  void | [**WetsloperesetHaloBotCPU**](#function-wetsloperesethalobotcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetsloperesetHaloBotGPU**](#function-wetsloperesethalobotgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  void | [**WetsloperesetHaloLeftCPU**](#function-wetsloperesethaloleftcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetsloperesetHaloLeftGPU**](#function-wetsloperesethaloleftgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  void | [**WetsloperesetHaloRightCPU**](#function-wetsloperesethalorightcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetsloperesetHaloRightGPU**](#function-wetsloperesethalorightgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  void | [**WetsloperesetHaloTopCPU**](#function-wetsloperesethalotopcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetsloperesetHaloTopGPU**](#function-wetsloperesethalotopgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetsloperesetXGPU**](#function-wetsloperesetxgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetsloperesetYGPU**](#function-wetsloperesetygpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  \_\_global\_\_ void | [**gradient**](#function-gradient) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br> |
|  template \_\_global\_\_ void | [**gradient&lt; double &gt;**](#function-gradient-double) (int halowidth, int \* active, int \* level, double theta, double dx, double \* a, double \* dadx, double \* dady) <br> |
|  template \_\_global\_\_ void | [**gradient&lt; float &gt;**](#function-gradient-float) (int halowidth, int \* active, int \* level, float theta, float dx, float \* a, float \* dadx, float \* dady) <br> |
|  void | [**gradientC**](#function-gradientc) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  template void | [**gradientC&lt; double &gt;**](#function-gradientc-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* a, double \* dadx, double \* dady) <br> |
|  template void | [**gradientC&lt; float &gt;**](#function-gradientc-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* a, float \* dadx, float \* dady) <br> |
|  void | [**gradientCPU**](#function-gradientcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  template void | [**gradientCPU&lt; double &gt;**](#function-gradientcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, double \* zb) <br> |
|  template void | [**gradientCPU&lt; float &gt;**](#function-gradientcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, float \* zb) <br> |
|  void | [**gradientGPU**](#function-gradientgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  template void | [**gradientGPU&lt; double &gt;**](#function-gradientgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, double \* zb) <br> |
|  template void | [**gradientGPU&lt; float &gt;**](#function-gradientgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, float \* zb) <br> |
|  void | [**gradientGPUnew**](#function-gradientgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br> |
|  template void | [**gradientGPUnew&lt; double &gt;**](#function-gradientgpunew-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, double \* zb) <br> |
|  template void | [**gradientGPUnew&lt; float &gt;**](#function-gradientgpunew-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, float \* zb) <br> |
|  void | [**gradientHalo**](#function-gradienthalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  void | [**gradientHaloBot**](#function-gradienthalobot) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, int ib, int ix, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloBotGPU**](#function-gradienthalobotgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloBotGPUnew**](#function-gradienthalobotgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  void | [**gradientHaloGPU**](#function-gradienthalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  void | [**gradientHaloGPUnew**](#function-gradienthalogpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  void | [**gradientHaloLeft**](#function-gradienthaloleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, int ib, int iy, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloLeftGPU**](#function-gradienthaloleftgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloLeftGPUnew**](#function-gradienthaloleftgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  void | [**gradientHaloRight**](#function-gradienthaloright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, int ib, int iy, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloRightGPU**](#function-gradienthalorightgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloRightGPUnew**](#function-gradienthalorightgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  void | [**gradientHaloTop**](#function-gradienthalotop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, int ib, int ix, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloTopGPU**](#function-gradienthalotopgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientHaloTopGPUnew**](#function-gradienthalotopgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br> |
|  \_\_global\_\_ void | [**gradientSM**](#function-gradientsm) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br> |
|  template \_\_global\_\_ void | [**gradientSM&lt; double &gt;**](#function-gradientsm-double) (int halowidth, int \* active, int \* level, double theta, double dx, double \* a, double \* dadx, double \* dady) <br> |
|  template \_\_global\_\_ void | [**gradientSM&lt; float &gt;**](#function-gradientsm-float) (int halowidth, int \* active, int \* level, float theta, float dx, float \* a, float \* dadx, float \* dady) <br> |
|  \_\_global\_\_ void | [**gradientSMB**](#function-gradientsmb) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br> |
|  template \_\_global\_\_ void | [**gradientSMB&lt; double &gt;**](#function-gradientsmb-double) (int halowidth, int \* active, int \* level, double theta, double dx, double \* a, double \* dadx, double \* dady) <br> |
|  template \_\_global\_\_ void | [**gradientSMB&lt; float &gt;**](#function-gradientsmb-float) (int halowidth, int \* active, int \* level, float theta, float dx, float \* a, float \* dadx, float \* dady) <br> |
|  \_\_global\_\_ void | [**gradientSMC**](#function-gradientsmc) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br> |
|  template \_\_global\_\_ void | [**gradientSMC&lt; double &gt;**](#function-gradientsmc-double) (int halowidth, int \* active, int \* level, double theta, double dx, double \* a, double \* dadx, double \* dady) <br> |
|  template \_\_global\_\_ void | [**gradientSMC&lt; float &gt;**](#function-gradientsmc-float) (int halowidth, int \* active, int \* level, float theta, float dx, float \* a, float \* dadx, float \* dady) <br> |
|  \_\_global\_\_ void | [**gradientedgeX**](#function-gradientedgex) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx) <br> |
|  template \_\_global\_\_ void | [**gradientedgeX&lt; double &gt;**](#function-gradientedgex-double) (int halowidth, int \* active, int \* level, double theta, double dx, double \* a, double \* dadx) <br> |
|  template \_\_global\_\_ void | [**gradientedgeX&lt; float &gt;**](#function-gradientedgex-float) (int halowidth, int \* active, int \* level, float theta, float dx, float \* a, float \* dadx) <br> |
|  \_\_global\_\_ void | [**gradientedgeY**](#function-gradientedgey) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dady) <br> |
|  template \_\_global\_\_ void | [**gradientedgeY&lt; double &gt;**](#function-gradientedgey-double) (int halowidth, int \* active, int \* level, double theta, double dx, double \* a, double \* dady) <br> |
|  template \_\_global\_\_ void | [**gradientedgeY&lt; float &gt;**](#function-gradientedgey-float) (int halowidth, int \* active, int \* level, float theta, float dx, float \* a, float \* dady) <br> |




























## Public Functions Documentation




### function WetsloperesetCPU 

```C++
template<class T>
void WetsloperesetCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    T * zb
) 
```




<hr>



### function WetsloperesetHaloBotCPU 

```C++
template<class T>
void WetsloperesetHaloBotCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    T * zb
) 
```




<hr>



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



### function WetsloperesetHaloLeftCPU 

```C++
template<class T>
void WetsloperesetHaloLeftCPU (
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



### function WetsloperesetHaloRightCPU 

```C++
template<class T>
void WetsloperesetHaloRightCPU (
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



### function WetsloperesetHaloTopCPU 

```C++
template<class T>
void WetsloperesetHaloTopCPU (
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



### function gradient&lt; double &gt; 

```C++
template __global__ void gradient< double > (
    int halowidth,
    int * active,
    int * level,
    double theta,
    double dx,
    double * a,
    double * dadx,
    double * dady
) 
```




<hr>



### function gradient&lt; float &gt; 

```C++
template __global__ void gradient< float > (
    int halowidth,
    int * active,
    int * level,
    float theta,
    float dx,
    float * a,
    float * dadx,
    float * dady
) 
```




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



### function gradientC&lt; double &gt; 

```C++
template void gradientC< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * a,
    double * dadx,
    double * dady
) 
```




<hr>



### function gradientC&lt; float &gt; 

```C++
template void gradientC< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * a,
    float * dadx,
    float * dady
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



### function gradientCPU&lt; double &gt; 

```C++
template void gradientCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    double * zb
) 
```




<hr>



### function gradientCPU&lt; float &gt; 

```C++
template void gradientCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    float * zb
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



### function gradientGPU&lt; double &gt; 

```C++
template void gradientGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    double * zb
) 
```




<hr>



### function gradientGPU&lt; float &gt; 

```C++
template void gradientGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    float * zb
) 
```




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



### function gradientGPUnew&lt; double &gt; 

```C++
template void gradientGPUnew< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    double * zb
) 
```




<hr>



### function gradientGPUnew&lt; float &gt; 

```C++
template void gradientGPUnew< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    float * zb
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



### function gradientHaloBot 

```C++
template<class T>
void gradientHaloBot (
    Param XParam,
    BlockP < T > XBlock,
    int ib,
    int ix,
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



### function gradientHaloGPUnew 

```C++
template<class T>
void gradientHaloGPUnew (
    Param XParam,
    BlockP < T > XBlock,
    T * a,
    T * dadx,
    T * dady
) 
```




<hr>



### function gradientHaloLeft 

```C++
template<class T>
void gradientHaloLeft (
    Param XParam,
    BlockP < T > XBlock,
    int ib,
    int iy,
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



### function gradientHaloRight 

```C++
template<class T>
void gradientHaloRight (
    Param XParam,
    BlockP < T > XBlock,
    int ib,
    int iy,
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



### function gradientHaloTop 

```C++
template<class T>
void gradientHaloTop (
    Param XParam,
    BlockP < T > XBlock,
    int ib,
    int ix,
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



### function gradientSM&lt; double &gt; 

```C++
template __global__ void gradientSM< double > (
    int halowidth,
    int * active,
    int * level,
    double theta,
    double dx,
    double * a,
    double * dadx,
    double * dady
) 
```




<hr>



### function gradientSM&lt; float &gt; 

```C++
template __global__ void gradientSM< float > (
    int halowidth,
    int * active,
    int * level,
    float theta,
    float dx,
    float * a,
    float * dadx,
    float * dady
) 
```




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



### function gradientSMB&lt; double &gt; 

```C++
template __global__ void gradientSMB< double > (
    int halowidth,
    int * active,
    int * level,
    double theta,
    double dx,
    double * a,
    double * dadx,
    double * dady
) 
```




<hr>



### function gradientSMB&lt; float &gt; 

```C++
template __global__ void gradientSMB< float > (
    int halowidth,
    int * active,
    int * level,
    float theta,
    float dx,
    float * a,
    float * dadx,
    float * dady
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



### function gradientSMC&lt; double &gt; 

```C++
template __global__ void gradientSMC< double > (
    int halowidth,
    int * active,
    int * level,
    double theta,
    double dx,
    double * a,
    double * dadx,
    double * dady
) 
```




<hr>



### function gradientSMC&lt; float &gt; 

```C++
template __global__ void gradientSMC< float > (
    int halowidth,
    int * active,
    int * level,
    float theta,
    float dx,
    float * a,
    float * dadx,
    float * dady
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



### function gradientedgeX&lt; double &gt; 

```C++
template __global__ void gradientedgeX< double > (
    int halowidth,
    int * active,
    int * level,
    double theta,
    double dx,
    double * a,
    double * dadx
) 
```




<hr>



### function gradientedgeX&lt; float &gt; 

```C++
template __global__ void gradientedgeX< float > (
    int halowidth,
    int * active,
    int * level,
    float theta,
    float dx,
    float * a,
    float * dadx
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



### function gradientedgeY&lt; double &gt; 

```C++
template __global__ void gradientedgeY< double > (
    int halowidth,
    int * active,
    int * level,
    double theta,
    double dx,
    double * a,
    double * dady
) 
```




<hr>



### function gradientedgeY&lt; float &gt; 

```C++
template __global__ void gradientedgeY< float > (
    int halowidth,
    int * active,
    int * level,
    float theta,
    float dx,
    float * a,
    float * dady
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Gradients.cu`




# File Gradients.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Gradients.cu**](Gradients_8cu.md)

[Go to the source code of this file](Gradients_8cu_source.md)



* `#include "Gradients.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**WetsloperesetCPU**](#function-wetsloperesetcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_CPU function to apply wet slope limiters to gradients of surface elevation. Adjusts gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  void | [**WetsloperesetHaloBotCPU**](#function-wetsloperesethalobotcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_CPU function to reset the wet slope limiter at the bottom halo boundary. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**WetsloperesetHaloBotGPU**](#function-wetsloperesethalobotgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_GPU kernel to reset the wet slope limiter at the bottom halo boundary. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  void | [**WetsloperesetHaloLeftCPU**](#function-wetsloperesethaloleftcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_CPU function to apply wet slope limiters to gradients of surface elevation at the left halo boundary. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**WetsloperesetHaloLeftGPU**](#function-wetsloperesethaloleftgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_Device kernel to apply wet slope limiters to gradients of surface elevation at the left halo boundary. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  void | [**WetsloperesetHaloRightCPU**](#function-wetsloperesethalorightcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_CPU function to apply wet slope limiters to gradients of surface elevation at the right halo boundary. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**WetsloperesetHaloRightGPU**](#function-wetsloperesethalorightgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_Device kernel to apply wet slope limiters to gradients of surface elevation at the right halo boundary. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  void | [**WetsloperesetHaloTopCPU**](#function-wetsloperesethalotopcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_CPU function to reset the wet slope limiter at the top halo boundary. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**WetsloperesetHaloTopGPU**](#function-wetsloperesethalotopgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_GPU kernel to reset the wet slope limiter at the top halo boundary. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**WetsloperesetXGPU**](#function-wetsloperesetxgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_Device kernel to apply wet slope limiters to gradients of surface elevation in the x-direction. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**WetsloperesetYGPU**](#function-wetsloperesetygpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_Device kernel to apply wet slope limiters to gradients of surface elevation in the y-direction. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**gradient**](#function-gradient) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br>_Device kernel for calculating gradients for an evolving parameter using the minmod limiter._  |
|  template \_\_global\_\_ void | [**gradient&lt; double &gt;**](#function-gradient-double) (int halowidth, int \* active, int \* level, double theta, double dx, double \* a, double \* dadx, double \* dady) <br> |
|  template \_\_global\_\_ void | [**gradient&lt; float &gt;**](#function-gradient-float) (int halowidth, int \* active, int \* level, float theta, float dx, float \* a, float \* dadx, float \* dady) <br> |
|  void | [**gradientC**](#function-gradientc) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_CPU function for calculating gradients using the minmod limiter. Computes spatial derivatives in x and y directions for a given variable._  |
|  template void | [**gradientC&lt; double &gt;**](#function-gradientc-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* a, double \* dadx, double \* dady) <br> |
|  template void | [**gradientC&lt; float &gt;**](#function-gradientc-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* a, float \* dadx, float \* dady) <br> |
|  void | [**gradientCPU**](#function-gradientcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_CPU function to compute gradients for all evolving parameters, handle halo regions, and apply wet-dry fixes. Calculates spatial derivatives for height, surface elevation, and velocity components. Also manages halo regions and applies wet-dry fixes if necessary._  |
|  template void | [**gradientCPU&lt; double &gt;**](#function-gradientcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, double \* zb) <br> |
|  template void | [**gradientCPU&lt; float &gt;**](#function-gradientcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, float \* zb) <br> |
|  void | [**gradientGPU**](#function-gradientgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_Entry point for gradient of evolving variables calculation on the GPU._  |
|  template void | [**gradientGPU&lt; double &gt;**](#function-gradientgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, double \* zb) <br> |
|  template void | [**gradientGPU&lt; float &gt;**](#function-gradientgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, float \* zb) <br> |
|  void | [**gradientGPUnew**](#function-gradientgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_Alternative GPU gradient calculation using shared memory kernels and CUDA streams._  |
|  template void | [**gradientGPUnew&lt; double &gt;**](#function-gradientgpunew-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad, double \* zb) <br> |
|  template void | [**gradientGPUnew&lt; float &gt;**](#function-gradientgpunew-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad, float \* zb) <br> |
|  void | [**gradientHalo**](#function-gradienthalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_CPU function to compute gradients at the halo boundaries of all active blocks. This function iterates over all active blocks and computes the gradients at their halo boundaries using finite difference approximations._  |
|  void | [**gradientHaloBot**](#function-gradienthalobot) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, int ib, int ix, T \* a, T \* dadx, T \* dady) <br>_CPU function to compute the gradient at the bottom halo boundary of a specific block. This function calculates the x and y derivatives at the bottom edge of the block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloBotGPU**](#function-gradienthalobotgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the bottom halo boundary of blocks. This kernel calculates the x and y derivatives at the bottom edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloBotGPUnew**](#function-gradienthalobotgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the bottom halo boundary of blocks. This kernel calculates the x and y derivatives at the bottom edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  void | [**gradientHaloGPU**](#function-gradienthalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU function to compute gradients at the halo boundaries of all active blocks. This function launches CUDA kernels to compute the gradients at the halo boundaries of all active blocks using parallel processing._  |
|  void | [**gradientHaloGPUnew**](#function-gradienthalogpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU function to compute gradients at the halo boundaries of all active blocks using multiple CUDA streams. This function launches CUDA kernels in separate streams to compute the gradients at the halo boundaries of all active blocks, allowing for concurrent execution and improved performance._  |
|  void | [**gradientHaloLeft**](#function-gradienthaloleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, int ib, int iy, T \* a, T \* dadx, T \* dady) <br>_CPU function to compute the gradient at the left halo boundary of a specific block. This function calculates the x and y derivatives at the left edge of the block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloLeftGPU**](#function-gradienthaloleftgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the left halo boundary of blocks. This kernel calculates the x and y derivatives at the left edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloLeftGPUnew**](#function-gradienthaloleftgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the left halo boundary of blocks. This kernel calculates the x and y derivatives at the left edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  void | [**gradientHaloRight**](#function-gradienthaloright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, int ib, int iy, T \* a, T \* dadx, T \* dady) <br>_CPU function to compute the gradient at the left halo boundary of a specific block. This function calculates the x and y derivatives at the left edge of the block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloRightGPU**](#function-gradienthalorightgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the right halo boundary of blocks. This kernel calculates the x and y derivatives at the right edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloRightGPUnew**](#function-gradienthalorightgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the right halo boundary of blocks. This kernel calculates the x and y derivatives at the right edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  void | [**gradientHaloTop**](#function-gradienthalotop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, int ib, int ix, T \* a, T \* dadx, T \* dady) <br>_CPU function to compute the gradient at the top halo boundary of a specific block. This function calculates the x and y derivatives at the top edge of the block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloTopGPU**](#function-gradienthalotopgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the top halo boundary of blocks. This kernel calculates the x and y derivatives at the top edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloTopGPUnew**](#function-gradienthalotopgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the top halo boundary of blocks. This kernel calculates the x and y derivatives at the top edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientSM**](#function-gradientsm) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br>_Deprecated shared memory device kernel for gradient calculation._  |
|  template \_\_global\_\_ void | [**gradientSM&lt; double &gt;**](#function-gradientsm-double) (int halowidth, int \* active, int \* level, double theta, double dx, double \* a, double \* dadx, double \* dady) <br> |
|  template \_\_global\_\_ void | [**gradientSM&lt; float &gt;**](#function-gradientsm-float) (int halowidth, int \* active, int \* level, float theta, float dx, float \* a, float \* dadx, float \* dady) <br> |
|  \_\_global\_\_ void | [**gradientSMB**](#function-gradientsmb) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br>_Shared memory device kernel for gradient calculation (variant B)._  |
|  template \_\_global\_\_ void | [**gradientSMB&lt; double &gt;**](#function-gradientsmb-double) (int halowidth, int \* active, int \* level, double theta, double dx, double \* a, double \* dadx, double \* dady) <br> |
|  template \_\_global\_\_ void | [**gradientSMB&lt; float &gt;**](#function-gradientsmb-float) (int halowidth, int \* active, int \* level, float theta, float dx, float \* a, float \* dadx, float \* dady) <br> |
|  \_\_global\_\_ void | [**gradientSMC**](#function-gradientsmc) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br>_Shared memory device kernel for gradient calculation (variant C)._  |
|  template \_\_global\_\_ void | [**gradientSMC&lt; double &gt;**](#function-gradientsmc-double) (int halowidth, int \* active, int \* level, double theta, double dx, double \* a, double \* dadx, double \* dady) <br> |
|  template \_\_global\_\_ void | [**gradientSMC&lt; float &gt;**](#function-gradientsmc-float) (int halowidth, int \* active, int \* level, float theta, float dx, float \* a, float \* dadx, float \* dady) <br> |
|  \_\_global\_\_ void | [**gradientedgeX**](#function-gradientedgex) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx) <br>_Device kernel for calculating gradients for an evolving parameter using the minmod limiter only at a fixed column (i.e. fixed ix)._  |
|  template \_\_global\_\_ void | [**gradientedgeX&lt; double &gt;**](#function-gradientedgex-double) (int halowidth, int \* active, int \* level, double theta, double dx, double \* a, double \* dadx) <br> |
|  template \_\_global\_\_ void | [**gradientedgeX&lt; float &gt;**](#function-gradientedgex-float) (int halowidth, int \* active, int \* level, float theta, float dx, float \* a, float \* dadx) <br> |
|  \_\_global\_\_ void | [**gradientedgeY**](#function-gradientedgey) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dady) <br>_Device kernel for calculating gradients for an evolving parameter using the minmod limiter only at a fixed row (i.e. fixed iy)._  |
|  template \_\_global\_\_ void | [**gradientedgeY&lt; double &gt;**](#function-gradientedgey-double) (int halowidth, int \* active, int \* level, double theta, double dx, double \* a, double \* dady) <br> |
|  template \_\_global\_\_ void | [**gradientedgeY&lt; float &gt;**](#function-gradientedgey-float) (int halowidth, int \* active, int \* level, float theta, float dx, float \* a, float \* dady) <br> |




























## Public Functions Documentation




### function WetsloperesetCPU 

_CPU function to apply wet slope limiters to gradients of surface elevation. Adjusts gradients to prevent non-physical slopes in wet-dry transition zones._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `XEv` Evolving parameters (height, surface elevation, velocities) 
* `XGrad` Gradient storage for the evolving parameters 
* `zb` Bathymetry array 




        

<hr>



### function WetsloperesetHaloBotCPU 

_CPU function to reset the wet slope limiter at the bottom halo boundary. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `XEv` Evolving parameters (height, surface elevation, velocities) 
* `XGrad` Gradient storage for the evolving parameters 
* `zb` Bathymetry array 



**Note:**

This function specifically handles the bottom halo boundary, where special care is needed due to the absence of neighboring blocks on that side. The logic accounts for various configurations of neighboring blocks to correctly compute the bottom surface elevation needed for the wet slope limiter. 





        

<hr>



### function WetsloperesetHaloBotGPU 

_GPU kernel to reset the wet slope limiter at the bottom halo boundary. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._ 
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





**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `XEv` Evolving parameters (height, surface elevation, velocities) 
* `XGrad` Gradient storage for the evolving parameters 
* `zb` Bathymetry array 



**Note:**

This kernel specifically handles the bottom halo boundary, where special care is needed due to the absence of neighboring blocks on that side. The logic accounts for various configurations of neighboring blocks to correctly compute the bottom surface elevation needed for the wet slope limiter. 





        

<hr>



### function WetsloperesetHaloLeftCPU 

_CPU function to apply wet slope limiters to gradients of surface elevation at the left halo boundary. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `XEv` Evolving parameters (height, surface elevation, velocities) 
* `XGrad` Gradient storage for the evolving parameters 
* `zb` Bathymetry array 



**Note:**

This function specifically handles the left halo boundary, where special care is needed due to the absence of neighboring blocks on that side. The logic accounts for various configurations of neighboring blocks to correctly compute the left surface elevation needed for 





        

<hr>



### function WetsloperesetHaloLeftGPU 

_Device kernel to apply wet slope limiters to gradients of surface elevation at the left halo boundary. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._ 
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





**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `XEv` Evolving parameters (height, surface elevation, velocities) 
* `XGrad` Gradient storage for the evolving parameters 
* `zb` Bathymetry array 



**Note:**

This kernel specifically handles the left halo boundary, where special care is needed due to the absence of neighboring blocks on that side. The logic accounts for various configurations of neighboring blocks to correctly compute the left surface elevation needed for the wet slope limiter. 





        

<hr>



### function WetsloperesetHaloRightCPU 

_CPU function to apply wet slope limiters to gradients of surface elevation at the right halo boundary. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `XEv` Evolving parameters (height, surface elevation, velocities) 
* `XGrad` Gradient storage for the evolving parameters 
* `zb` Bathymetry array 



**Note:**

This function specifically handles the right halo boundary, where special care is needed due to the absence of neighboring blocks on that side. The logic accounts for various configurations of neighboring blocks to correctly compute the right surface elevation needed for the wet slope limiter. 





        

<hr>



### function WetsloperesetHaloRightGPU 

_Device kernel to apply wet slope limiters to gradients of surface elevation at the right halo boundary. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._ 
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





**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `XEv` Evolving parameters (height, surface elevation, velocities) 
* `XGrad` Gradient storage for the evolving parameters 
* `zb` Bathymetry array 



**Note:**

This kernel specifically handles the right halo boundary, where special care is needed due to the absence of neighboring blocks on that side. The logic accounts for various configurations of neighboring blocks to correctly compute the right surface elevation needed for the wet slope limiter. 





        

<hr>



### function WetsloperesetHaloTopCPU 

_CPU function to reset the wet slope limiter at the top halo boundary. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `XEv` Evolving parameters (height, surface elevation, velocities) 
* `XGrad` Gradient storage for the evolving parameters 
* `zb` Bathymetry array 



**Note:**

This function specifically handles the top halo boundary, where special care is needed due to the absence of neighboring blocks on that side. The logic accounts for various configurations of neighboring blocks to correctly compute the top surface elevation needed for the wet slope limiter. 





        

<hr>



### function WetsloperesetHaloTopGPU 

_GPU kernel to reset the wet slope limiter at the top halo boundary. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._ 
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





**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `XEv` Evolving parameters (height, surface elevation, velocities) 
* `XGrad` Gradient storage for the evolving parameters 
* `zb` Bathymetry array 



**Note:**

This kernel specifically handles the top halo boundary, where special care is needed due to the absence of neighboring blocks on that side. The logic accounts for various configurations of neighboring blocks to correctly compute the top surface elevation needed for the wet slope limiter. 





        

<hr>



### function WetsloperesetXGPU 

_Device kernel to apply wet slope limiters to gradients of surface elevation in the x-direction. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._ 
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





**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `XEv` Evolving parameters (height, surface elevation, velocities) 
* `XGrad` Gradient storage for the evolving parameters 
* `zb` Bathymetry array 




        

<hr>



### function WetsloperesetYGPU 

_Device kernel to apply wet slope limiters to gradients of surface elevation in the y-direction. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._ 
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





**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `XEv` Evolving parameters (height, surface elevation, velocities) 
* `XGrad` Gradient storage for the evolving parameters 
* `zb` Bathymetry array 




        

<hr>



### function gradient 

_Device kernel for calculating gradients for an evolving parameter using the minmod limiter._ 
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



Computes spatial derivatives in x and y directions for a given variable.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `halowidth` Width of halo region 
* `active` Active block indices 
* `level` Block refinement levels 
* `theta` Limiter parameter 
* `dx` Grid spacing 
* `a` Input variable array 
* `dadx` Output gradient in x 
* `dady` Output gradient in y 




        

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

_CPU function for calculating gradients using the minmod limiter. Computes spatial derivatives in x and y directions for a given variable._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `a` Input variable array 
* `dadx` Output gradient in x 
* `dady` Output gradient in y 




        

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

_CPU function to compute gradients for all evolving parameters, handle halo regions, and apply wet-dry fixes. Calculates spatial derivatives for height, surface elevation, and velocity components. Also manages halo regions and applies wet-dry fixes if necessary._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `XEv` Evolving parameters (height, surface elevation, velocities) 
* `XGrad` Gradient storage for the evolving parameters 
* `zb` Bathymetry array 




        

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

_Entry point for gradient of evolving variables calculation on the GPU._ 
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



Calculates gradients of evolving variables using CUDA kernels and synchronizes device operations. Handles halo filling and elevation conservation if required.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradient variables 
* `zb` Bed elevation array

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

_Alternative GPU gradient calculation using shared memory kernels and CUDA streams._ 
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



Uses gradientSMC kernels and handles halo filling, elevation conservation, and wet/dry prolongation.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradient variables 
* `zb` Bed elevation array 




        

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

_CPU function to compute gradients at the halo boundaries of all active blocks. This function iterates over all active blocks and computes the gradients at their halo boundaries using finite difference approximations._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This function calls specific functions to handle each of the four halo boundaries (left, right, bottom, top) for each block. It ensures that gradients are accurately computed at the edges of the computational domain, which is crucial for maintaining solution accuracy and stability. The function assumes that the input arrays are properly allocated and sized according to the simulation parameters. The gradient computations are performed using central differences where possible, and one-sided differences at the boundaries. The function is templated to support different data types (e.g., float, double). 




**See also:** gradientHaloLeft, gradientHaloRight, gradientHaloBot, gradientHaloTop 



        

<hr>



### function gradientHaloBot 

_CPU function to compute the gradient at the bottom halo boundary of a specific block. This function calculates the x and y derivatives at the bottom edge of the block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `ib` Index of the block for which the bottom halo gradient is to be computed 
* `ix` x-index within the block 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This function handles various configurations of neighboring blocks, including cases where neighboring blocks are at different levels of refinement 





        

<hr>



### function gradientHaloBotGPU 

_GPU kernel to compute the gradient at the bottom halo boundary of blocks. This kernel calculates the x and y derivatives at the bottom edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This kernel handles various configurations of neighboring blocks, including cases where neighboring blocks are at different levels of refinement 





        

<hr>



### function gradientHaloBotGPUnew 

_GPU kernel to compute the gradient at the bottom halo boundary of blocks. This kernel calculates the x and y derivatives at the bottom edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This kernel handles various configurations of neighboring blocks, including cases where neighboring blocks are at different levels of refinement 





        

<hr>



### function gradientHaloGPU 

_GPU function to compute gradients at the halo boundaries of all active blocks. This function launches CUDA kernels to compute the gradients at the halo boundaries of all active blocks using parallel processing._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This function sets up the CUDA grid and block dimensions and launches specific kernels to handle each of the four halo boundaries (left, right, bottom, top) for all active blocks. It ensures that gradients are accurately computed at the edges of the computational domain, which is crucial for maintaining solution accuracy and stability. The function assumes that the input arrays are properly allocated and sized according to the simulation parameters. The gradient computations are performed using central differences where possible, and one-sided differences at the boundaries. The function is templated to support different data types (e.g., float, double). 




**See also:** gradientHaloLeftGPU, gradientHaloRightGPU, gradientHaloBotGPU, gradientHaloTopGPU 


**See also:** gradientHaloLeft, gradientHaloRight, gradientHaloBot, gradientHaloTop 



        

<hr>



### function gradientHaloGPUnew 

_GPU function to compute gradients at the halo boundaries of all active blocks using multiple CUDA streams. This function launches CUDA kernels in separate streams to compute the gradients at the halo boundaries of all active blocks, allowing for concurrent execution and improved performance._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This function sets up multiple CUDA streams and launches specific kernels to handle each of the four halo boundaries (left, right, bottom, top) for all active blocks in separate streams. It ensures that gradients are accurately computed at the edges of the computational domain, which is crucial for maintaining solution accuracy and stability. The function assumes that the input arrays are properly allocated and sized according to the simulation parameters. The gradient computations are performed using central differences where possible, and one-sided differences at the boundaries. The function is templated to support different data types (e.g., float, double). 




**See also:** gradientHaloLeftGPUnew, gradientHaloRightGPUnew, gradientHaloBotGPUnew, gradientHaloTopGPUnew 


**See also:** gradientHaloLeft, gradientHaloRight, gradientHaloBot, gradientHaloTop 



        

<hr>



### function gradientHaloLeft 

_CPU function to compute the gradient at the left halo boundary of a specific block. This function calculates the x and y derivatives at the left edge of the block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `ib` Index of the block for which the left halo gradient is to be computed 
* `iy` y-index within the block 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This function handles various configurations of neighboring blocks, including cases where neighboring blocks are at different levels of refinement 





        

<hr>



### function gradientHaloLeftGPU 

_GPU kernel to compute the gradient at the left halo boundary of blocks. This kernel calculates the x and y derivatives at the left edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This kernel handles various configurations of neighboring blocks, including cases where neighboring blocks are at different levels of refinement 





        

<hr>



### function gradientHaloLeftGPUnew 

_GPU kernel to compute the gradient at the left halo boundary of blocks. This kernel calculates the x and y derivatives at the left edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This kernel handles various configurations of neighboring blocks, including cases where neighboring blocks are at different levels of refinement 





        

<hr>



### function gradientHaloRight 

_CPU function to compute the gradient at the left halo boundary of a specific block. This function calculates the x and y derivatives at the left edge of the block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `ib` Index of the block for which the left halo gradient is to be computed 
* `iy` y-index within the block 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This function handles various configurations of neighboring blocks, including cases where neighboring blocks are at different levels of refinement 





        

<hr>



### function gradientHaloRightGPU 

_GPU kernel to compute the gradient at the right halo boundary of blocks. This kernel calculates the x and y derivatives at the right edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This kernel handles various configurations of neighboring blocks, including cases where neighboring blocks are at different levels of refinement 





        

<hr>



### function gradientHaloRightGPUnew 

_GPU kernel to compute the gradient at the right halo boundary of blocks. This kernel calculates the x and y derivatives at the right edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This kernel handles various configurations of neighboring blocks, including cases where neighboring blocks are at different levels of refinement 





        

<hr>



### function gradientHaloTop 

_CPU function to compute the gradient at the top halo boundary of a specific block. This function calculates the x and y derivatives at the top edge of the block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `ib` Index of the block for which the top halo gradient is to be computed 
* `ix` x-index within the block 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This function handles various configurations of neighboring blocks, including cases where neighboring blocks are at different levels of refinement 





        

<hr>



### function gradientHaloTopGPU 

_GPU kernel to compute the gradient at the top halo boundary of blocks. This kernel calculates the x and y derivatives at the top edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This kernel handles various configurations of neighboring blocks, including cases where neighboring blocks are at different levels of refinement 





        

<hr>



### function gradientHaloTopGPUnew 

_GPU kernel to compute the gradient at the top halo boundary of blocks. This kernel calculates the x and y derivatives at the top edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._ 
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





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block information 
* `a` Array containing the variable for which gradients are to be computed 
* `dadx` Array to store the computed x-derivative gradients 
* `dady` Array to store the computed y-derivative gradients 



**Note:**

This kernel handles various configurations of neighboring blocks, including cases where neighboring blocks are at different levels of refinement 





        

<hr>



### function gradientSM 

_Deprecated shared memory device kernel for gradient calculation._ 
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



Uses shared memory for stencil operations; slower than the standard kernel.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `halowidth` Width of halo region 
* `active` Active block indices 
* `level` Block refinement levels 
* `theta` Limiter parameter 
* `dx` Grid spacing 
* `a` Input variable array 
* `dadx` Output gradient in x 
* `dady` Output gradient in y 




        

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

_Shared memory device kernel for gradient calculation (variant B)._ 
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



Uses a fixed shared memory tile for stencil operations; only computes gradients for interior points.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `halowidth` Width of halo region 
* `active` Active block indices 
* `level` Block refinement levels 
* `theta` Limiter parameter 
* `dx` Grid spacing 
* `a` Input variable array 
* `dadx` Output gradient in x 
* `dady` Output gradient in y 




        

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

_Shared memory device kernel for gradient calculation (variant C)._ 
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



Uses a flat shared memory array for stencil operations; computes gradients for all points.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `halowidth` Width of halo region 
* `active` Active block indices 
* `level` Block refinement levels 
* `theta` Limiter parameter 
* `dx` Grid spacing 
* `a` Input variable array 
* `dadx` Output gradient in x 
* `dady` Output gradient in y 




        

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

_Device kernel for calculating gradients for an evolving parameter using the minmod limiter only at a fixed column (i.e. fixed ix)._ 
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



Computes x-derivative for a specific column using the minmod limiter.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `halowidth` Width of halo region 
* `active` Active block indices 
* `level` Block refinement levels 
* `theta` Limiter parameter 
* `dx` Grid spacing 
* `a` Input variable array 
* `dadx` Output gradient in x 




        

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

_Device kernel for calculating gradients for an evolving parameter using the minmod limiter only at a fixed row (i.e. fixed iy)._ 
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



Computes y-derivative for a specific row using the minmod limiter.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `halowidth` Width of halo region 
* `active` Active block indices 
* `level` Block refinement levels 
* `theta` Limiter parameter 
* `dx` Grid spacing 
* `a` Input variable array 
* `dady` Output gradient in y 




        

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




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
|  \_\_global\_\_ void | [**WetsloperesetHaloBotGPU**](#function-wetsloperesethalobotgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_GPU kernel to reset the wet slope limiter at the bottom halo boundary. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**WetsloperesetHaloLeftGPU**](#function-wetsloperesethaloleftgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_Device kernel to apply wet slope limiters to gradients of surface elevation at the left halo boundary. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**WetsloperesetHaloRightGPU**](#function-wetsloperesethalorightgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_Device kernel to apply wet slope limiters to gradients of surface elevation at the right halo boundary. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**WetsloperesetHaloTopGPU**](#function-wetsloperesethalotopgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_GPU kernel to reset the wet slope limiter at the top halo boundary. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**WetsloperesetXGPU**](#function-wetsloperesetxgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_Device kernel to apply wet slope limiters to gradients of surface elevation in the x-direction. Adjusts x-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**WetsloperesetYGPU**](#function-wetsloperesetygpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_Device kernel to apply wet slope limiters to gradients of surface elevation in the y-direction. Adjusts y-derivative gradients to prevent non-physical slopes in wet-dry transition zones._  |
|  \_\_global\_\_ void | [**gradient**](#function-gradient) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br>_Device kernel for calculating gradients for an evolving parameter using the minmod limiter._  |
|  void | [**gradientC**](#function-gradientc) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_CPU function for calculating gradients using the minmod limiter. Computes spatial derivatives in x and y directions for a given variable._  |
|  void | [**gradientCPU**](#function-gradientcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_CPU function to compute gradients for all evolving parameters, handle halo regions, and apply wet-dry fixes. Calculates spatial derivatives for height, surface elevation, and velocity components. Also manages halo regions and applies wet-dry fixes if necessary._  |
|  void | [**gradientGPU**](#function-gradientgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_Entry point for gradient of evolving variables calculation on the GPU._  |
|  void | [**gradientGPUnew**](#function-gradientgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, T \* zb) <br>_Alternative GPU gradient calculation using shared memory kernels and CUDA streams._  |
|  void | [**gradientHalo**](#function-gradienthalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_CPU function to compute gradients at the halo boundaries of all active blocks. This function iterates over all active blocks and computes the gradients at their halo boundaries using finite difference approximations._  |
|  \_\_global\_\_ void | [**gradientHaloBotGPU**](#function-gradienthalobotgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the bottom halo boundary of blocks. This kernel calculates the x and y derivatives at the bottom edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloBotGPUnew**](#function-gradienthalobotgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the bottom halo boundary of blocks. This kernel calculates the x and y derivatives at the bottom edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  void | [**gradientHaloGPU**](#function-gradienthalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU function to compute gradients at the halo boundaries of all active blocks. This function launches CUDA kernels to compute the gradients at the halo boundaries of all active blocks using parallel processing._  |
|  \_\_global\_\_ void | [**gradientHaloLeftGPU**](#function-gradienthaloleftgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the left halo boundary of blocks. This kernel calculates the x and y derivatives at the left edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloLeftGPUnew**](#function-gradienthaloleftgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the left halo boundary of blocks. This kernel calculates the x and y derivatives at the left edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloRightGPU**](#function-gradienthalorightgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the right halo boundary of blocks. This kernel calculates the x and y derivatives at the right edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloRightGPUnew**](#function-gradienthalorightgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the right halo boundary of blocks. This kernel calculates the x and y derivatives at the right edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloTopGPU**](#function-gradienthalotopgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the top halo boundary of blocks. This kernel calculates the x and y derivatives at the top edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientHaloTopGPUnew**](#function-gradienthalotopgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* a, T \* dadx, T \* dady) <br>_GPU kernel to compute the gradient at the top halo boundary of blocks. This kernel calculates the x and y derivatives at the top edge of each block using finite difference approximations, taking into account the presence of neighboring blocks and their levels._  |
|  \_\_global\_\_ void | [**gradientSM**](#function-gradientsm) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br>_Deprecated shared memory device kernel for gradient calculation._  |
|  \_\_global\_\_ void | [**gradientSMB**](#function-gradientsmb) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br>_Shared memory device kernel for gradient calculation (variant B)._  |
|  \_\_global\_\_ void | [**gradientSMC**](#function-gradientsmc) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx, T \* dady) <br>_Shared memory device kernel for gradient calculation (variant C)._  |
|  \_\_global\_\_ void | [**gradientedgeX**](#function-gradientedgex) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dadx) <br>_Device kernel for calculating gradients for an evolving parameter using the minmod limiter only at a fixed column (i.e. fixed ix)._  |
|  \_\_global\_\_ void | [**gradientedgeY**](#function-gradientedgey) (int halowidth, int \* active, int \* level, T theta, T dx, T \* a, T \* dady) <br>_Device kernel for calculating gradients for an evolving parameter using the minmod limiter only at a fixed row (i.e. fixed iy)._  |




























## Public Functions Documentation




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

------------------------------
The documentation for this class was generated from the following file `src/Gradients.h`


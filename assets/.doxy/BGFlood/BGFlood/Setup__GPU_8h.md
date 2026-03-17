

# File Setup\_GPU.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Setup\_GPU.h**](Setup__GPU_8h.md)

[Go to the source code of this file](Setup__GPU_8h_source.md)



* `#include "General.h"`
* `#include "Forcing.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "MemManagement.h"`
* `#include "Halo.h"`
* `#include "InitialConditions.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**AllocateBndTEX**](#function-allocatebndtex) ([**bndparam**](classbndparam.md) & side) <br>_Allocate boundary texture for GPU._  |
|  void | [**AllocateTEX**](#function-allocatetex) (int nx, int ny, [**TexSetP**](structTexSetP.md) & Tex, float \* input) <br>_Allocate and bind a CUDA texture object._  |
|  void | [**CUDA\_CHECK**](#function-cuda_check) (cudaError CUDerr) <br>_Check CUDA error status and print message if error occurs._  |
|  void | [**CopyGPUtoCPU**](#function-copygputocpu) (int nblk, int blksize, T \* z\_cpu, T \* z\_gpu) <br>_Copy data from GPU to CPU memory._  |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel\_cpu, [**Model**](structModel.md)&lt; T &gt; XModel\_gpu) <br>_Copy complex data structures from CPU to GPU._  |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, T \* z\_cpu, T \* z\_gpu) <br>_Copy data from CPU to GPU memory._  |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_cpu, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_gpu) <br>_Copy complex data structures from CPU to GPU._  |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; XEv\_cpu, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; XEv\_gpu) <br>_Copy complex data structures from CPU to GPU._  |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad\_cpu, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad\_gpu) <br>_Copy complex data structures from CPU to GPU._  |
|  void | [**SetupGPU**](#function-setupgpu) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel\_g) <br>_Setup and initialize GPU for simulation._  |




























## Public Functions Documentation




### function AllocateBndTEX 

_Allocate boundary texture for GPU._ 
```C++
void AllocateBndTEX (
    bndparam & side
) 
```



Allocates and binds boundary water level data as a CUDA texture for GPU use.




**Parameters:**


* `side` Boundary parameter structure 




        

<hr>



### function AllocateTEX 

_Allocate and bind a CUDA texture object._ 
```C++
void AllocateTEX (
    int nx,
    int ny,
    TexSetP & Tex,
    float * input
) 
```



Allocates a CUDA array and creates a texture object for use in GPU kernels.




**Parameters:**


* `nx` Number of x grid points 
* `ny` Number of y grid points 
* `Tex` Texture set structure 
* `input` Input data array 




        

<hr>



### function CUDA\_CHECK 

_Check CUDA error status and print message if error occurs._ 
```C++
void CUDA_CHECK (
    cudaError CUDerr
) 
```



Checks the CUDA error code and prints an error message if the code indicates failure.




**Parameters:**


* `CUDerr` CUDA error code 




        

<hr>



### function CopyGPUtoCPU 

_Copy data from GPU to CPU memory._ 
```C++
template<class T>
void CopyGPUtoCPU (
    int nblk,
    int blksize,
    T * z_cpu,
    T * z_gpu
) 
```



Copies an array from device (GPU) memory to host (CPU) memory using CUDA.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `z_cpu` Destination array (CPU) 
* `z_gpu` Source array (GPU) 




        

<hr>



### function CopytoGPU 

_Copy complex data structures from CPU to GPU._ 
```C++
template<class T>
void CopytoGPU (
    int nblk,
    int blksize,
    Param XParam,
    Model < T > XModel_cpu,
    Model < T > XModel_gpu
) 
```



This function copies the entire model structure from the host (CPU) to the device (GPU) memory.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `XParam` Simulation parameters 
* `XModel_cpu` Source model structure (CPU) 
* `XModel_gpu` Destination model structure (GPU) 




        

<hr>



### function CopytoGPU 

_Copy data from CPU to GPU memory._ 
```C++
template<class T>
void CopytoGPU (
    int nblk,
    int blksize,
    T * z_cpu,
    T * z_gpu
) 
```



Copies an array from host (CPU) memory to device (GPU) memory using CUDA.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `z_cpu` Source array (CPU) 
* `z_gpu` Destination array (GPU) 




        

<hr>



### function CopytoGPU 

_Copy complex data structures from CPU to GPU._ 
```C++
template<class T>
void CopytoGPU (
    int nblk,
    int blksize,
    EvolvingP < T > XEv_cpu,
    EvolvingP < T > XEv_gpu
) 
```



This function copies the evolving variables structure from the host (CPU) to the device (GPU) memory.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `XEv_cpu` Source evolving variables structure (CPU) 
* `XEv_gpu` Destination evolving variables structure (GPU) 




        

<hr>



### function CopytoGPU 

_Copy complex data structures from CPU to GPU._ 
```C++
template<class T>
void CopytoGPU (
    int nblk,
    int blksize,
    EvolvingP_M < T > XEv_cpu,
    EvolvingP_M < T > XEv_gpu
) 
```



This function copies the evolving variables with momentum structure from the host (CPU) to the device (GPU) memory.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `XEv_cpu` Source evolving variables with momentum structure (CPU) 
* `XEv_gpu` Destination evolving variables with momentum structure (GPU) 




        

<hr>



### function CopytoGPU 

_Copy complex data structures from CPU to GPU._ 
```C++
template<class T>
void CopytoGPU (
    int nblk,
    int blksize,
    GradientsP < T > XGrad_cpu,
    GradientsP < T > XGrad_gpu
) 
```



This function copies the gradients structure from the host (CPU) to the device (GPU) memory.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `XGrad_cpu` Source gradients structure (CPU) 
* `XGrad_gpu` Destination gradients structure (GPU) 




        

<hr>



### function SetupGPU 

_Setup and initialize GPU for simulation._ 
```C++
template<class T>
void SetupGPU (
    Param & XParam,
    Model < T > XModel,
    Forcing < float > & XForcing,
    Model < T > & XModel_g
) 
```



This function sets up the GPU device, allocates memory, and copies data from the host to the device. 
 

**Parameters:**


* `XParam` Simulation parameters 
* `XModel` Host model data structure 
* `XForcing` [**Forcing**](structForcing.md) data structure 
* `XModel_g` Device model data structure 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Setup_GPU.h`


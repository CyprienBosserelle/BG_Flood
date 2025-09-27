

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
|  void | [**AllocateBndTEX**](#function-allocatebndtex) ([**bndparam**](classbndparam.md) & side) <br> |
|  void | [**AllocateTEX**](#function-allocatetex) (int nx, int ny, [**TexSetP**](structTexSetP.md) & Tex, float \* input) <br> |
|  void | [**CUDA\_CHECK**](#function-cuda_check) (cudaError CUDerr) <br> |
|  void | [**CopyGPUtoCPU**](#function-copygputocpu) (int nblk, int blksize, T \* z\_cpu, T \* z\_gpu) <br> |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel\_cpu, [**Model**](structModel.md)&lt; T &gt; XModel\_gpu) <br> |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, T \* z\_cpu, T \* z\_gpu) <br> |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_cpu, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_gpu) <br> |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; XEv\_cpu, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; XEv\_gpu) <br> |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad\_cpu, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad\_gpu) <br> |
|  void | [**SetupGPU**](#function-setupgpu) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel\_g) <br> |




























## Public Functions Documentation




### function AllocateBndTEX 

```C++
void AllocateBndTEX (
    bndparam & side
) 
```




<hr>



### function AllocateTEX 

```C++
void AllocateTEX (
    int nx,
    int ny,
    TexSetP & Tex,
    float * input
) 
```




<hr>



### function CUDA\_CHECK 

```C++
void CUDA_CHECK (
    cudaError CUDerr
) 
```




<hr>



### function CopyGPUtoCPU 

```C++
template<class T>
void CopyGPUtoCPU (
    int nblk,
    int blksize,
    T * z_cpu,
    T * z_gpu
) 
```




<hr>



### function CopytoGPU 

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




<hr>



### function CopytoGPU 

```C++
template<class T>
void CopytoGPU (
    int nblk,
    int blksize,
    T * z_cpu,
    T * z_gpu
) 
```




<hr>



### function CopytoGPU 

```C++
template<class T>
void CopytoGPU (
    int nblk,
    int blksize,
    EvolvingP < T > XEv_cpu,
    EvolvingP < T > XEv_gpu
) 
```




<hr>



### function CopytoGPU 

```C++
template<class T>
void CopytoGPU (
    int nblk,
    int blksize,
    EvolvingP_M < T > XEv_cpu,
    EvolvingP_M < T > XEv_gpu
) 
```




<hr>



### function CopytoGPU 

```C++
template<class T>
void CopytoGPU (
    int nblk,
    int blksize,
    GradientsP < T > XGrad_cpu,
    GradientsP < T > XGrad_gpu
) 
```




<hr>



### function SetupGPU 

```C++
template<class T>
void SetupGPU (
    Param & XParam,
    Model < T > XModel,
    Forcing < float > & XForcing,
    Model < T > & XModel_g
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Setup_GPU.h`


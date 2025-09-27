

# File Setup\_GPU.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Setup\_GPU.cu**](Setup__GPU_8cu.md)

[Go to the source code of this file](Setup__GPU_8cu_source.md)



* `#include "Setup_GPU.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**AllocateBndTEX**](#function-allocatebndtex) ([**bndparam**](classbndparam.md) & side) <br> |
|  void | [**AllocateTEX**](#function-allocatetex) (int nx, int ny, [**TexSetP**](structTexSetP.md) & Tex, float \* input) <br> |
|  void | [**CUDA\_CHECK**](#function-cuda_check) (cudaError CUDerr) <br> |
|  void | [**CopyGPUtoCPU**](#function-copygputocpu) (int nblk, int blksize, T \* z\_cpu, T \* z\_gpu) <br> |
|  template void | [**CopyGPUtoCPU&lt; bool &gt;**](#function-copygputocpu-bool) (int nblk, int blksize, bool \* z\_cpu, bool \* z\_gpu) <br> |
|  template void | [**CopyGPUtoCPU&lt; double &gt;**](#function-copygputocpu-double) (int nblk, int blksize, double \* z\_cpu, double \* z\_gpu) <br> |
|  template void | [**CopyGPUtoCPU&lt; float &gt;**](#function-copygputocpu-float) (int nblk, int blksize, float \* z\_cpu, float \* z\_gpu) <br> |
|  template void | [**CopyGPUtoCPU&lt; int &gt;**](#function-copygputocpu-int) (int nblk, int blksize, int \* z\_cpu, int \* z\_gpu) <br> |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, T \* z\_cpu, T \* z\_gpu) <br> |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_cpu, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv\_gpu) <br> |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; XEv\_cpu, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; XEv\_gpu) <br> |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad\_cpu, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad\_gpu) <br> |
|  template void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad\_cpu, [**GradientsP**](structGradientsP.md)&lt; float &gt; XGrad\_gpu) <br> |
|  template void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad\_cpu, [**GradientsP**](structGradientsP.md)&lt; double &gt; XGrad\_gpu) <br> |
|  void | [**CopytoGPU**](#function-copytogpu) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel\_cpu, [**Model**](structModel.md)&lt; T &gt; XModel\_gpu) <br> |
|  template void | [**CopytoGPU&lt; bool &gt;**](#function-copytogpu-bool) (int nblk, int blksize, bool \* z\_cpu, bool \* z\_gpu) <br> |
|  template void | [**CopytoGPU&lt; double &gt;**](#function-copytogpu-double) (int nblk, int blksize, double \* z\_cpu, double \* z\_gpu) <br> |
|  template void | [**CopytoGPU&lt; double &gt;**](#function-copytogpu-double) (int nblk, int blksize, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv\_cpu, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv\_gpu) <br> |
|  template void | [**CopytoGPU&lt; double &gt;**](#function-copytogpu-double) (int nblk, int blksize, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; double &gt; XEv\_cpu, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; double &gt; XEv\_gpu) <br> |
|  template void | [**CopytoGPU&lt; double &gt;**](#function-copytogpu-double) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; double &gt; XModel\_cpu, [**Model**](structModel.md)&lt; double &gt; XModel\_gpu) <br> |
|  template void | [**CopytoGPU&lt; float &gt;**](#function-copytogpu-float) (int nblk, int blksize, float \* z\_cpu, float \* z\_gpu) <br> |
|  template void | [**CopytoGPU&lt; float &gt;**](#function-copytogpu-float) (int nblk, int blksize, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv\_cpu, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv\_gpu) <br> |
|  template void | [**CopytoGPU&lt; float &gt;**](#function-copytogpu-float) (int nblk, int blksize, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; float &gt; XEv\_cpu, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; float &gt; XEv\_gpu) <br> |
|  template void | [**CopytoGPU&lt; float &gt;**](#function-copytogpu-float) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; float &gt; XModel\_cpu, [**Model**](structModel.md)&lt; float &gt; XModel\_gpu) <br> |
|  template void | [**CopytoGPU&lt; int &gt;**](#function-copytogpu-int) (int nblk, int blksize, int \* z\_cpu, int \* z\_gpu) <br> |
|  void | [**SetupGPU**](#function-setupgpu) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel\_g) <br> |
|  template void | [**SetupGPU&lt; double &gt;**](#function-setupgpu-double) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; double &gt; XModel, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; double &gt; & XModel\_g) <br> |
|  template void | [**SetupGPU&lt; float &gt;**](#function-setupgpu-float) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; float &gt; XModel, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; float &gt; & XModel\_g) <br> |




























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



### function CopyGPUtoCPU&lt; bool &gt; 

```C++
template void CopyGPUtoCPU< bool > (
    int nblk,
    int blksize,
    bool * z_cpu,
    bool * z_gpu
) 
```




<hr>



### function CopyGPUtoCPU&lt; double &gt; 

```C++
template void CopyGPUtoCPU< double > (
    int nblk,
    int blksize,
    double * z_cpu,
    double * z_gpu
) 
```




<hr>



### function CopyGPUtoCPU&lt; float &gt; 

```C++
template void CopyGPUtoCPU< float > (
    int nblk,
    int blksize,
    float * z_cpu,
    float * z_gpu
) 
```




<hr>



### function CopyGPUtoCPU&lt; int &gt; 

```C++
template void CopyGPUtoCPU< int > (
    int nblk,
    int blksize,
    int * z_cpu,
    int * z_gpu
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



### function CopytoGPU 

```C++
template void CopytoGPU (
    int nblk,
    int blksize,
    GradientsP < float > XGrad_cpu,
    GradientsP < float > XGrad_gpu
) 
```




<hr>



### function CopytoGPU 

```C++
template void CopytoGPU (
    int nblk,
    int blksize,
    GradientsP < double > XGrad_cpu,
    GradientsP < double > XGrad_gpu
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



### function CopytoGPU&lt; bool &gt; 

```C++
template void CopytoGPU< bool > (
    int nblk,
    int blksize,
    bool * z_cpu,
    bool * z_gpu
) 
```




<hr>



### function CopytoGPU&lt; double &gt; 

```C++
template void CopytoGPU< double > (
    int nblk,
    int blksize,
    double * z_cpu,
    double * z_gpu
) 
```




<hr>



### function CopytoGPU&lt; double &gt; 

```C++
template void CopytoGPU< double > (
    int nblk,
    int blksize,
    EvolvingP < double > XEv_cpu,
    EvolvingP < double > XEv_gpu
) 
```




<hr>



### function CopytoGPU&lt; double &gt; 

```C++
template void CopytoGPU< double > (
    int nblk,
    int blksize,
    EvolvingP_M < double > XEv_cpu,
    EvolvingP_M < double > XEv_gpu
) 
```




<hr>



### function CopytoGPU&lt; double &gt; 

```C++
template void CopytoGPU< double > (
    int nblk,
    int blksize,
    Param XParam,
    Model < double > XModel_cpu,
    Model < double > XModel_gpu
) 
```




<hr>



### function CopytoGPU&lt; float &gt; 

```C++
template void CopytoGPU< float > (
    int nblk,
    int blksize,
    float * z_cpu,
    float * z_gpu
) 
```




<hr>



### function CopytoGPU&lt; float &gt; 

```C++
template void CopytoGPU< float > (
    int nblk,
    int blksize,
    EvolvingP < float > XEv_cpu,
    EvolvingP < float > XEv_gpu
) 
```




<hr>



### function CopytoGPU&lt; float &gt; 

```C++
template void CopytoGPU< float > (
    int nblk,
    int blksize,
    EvolvingP_M < float > XEv_cpu,
    EvolvingP_M < float > XEv_gpu
) 
```




<hr>



### function CopytoGPU&lt; float &gt; 

```C++
template void CopytoGPU< float > (
    int nblk,
    int blksize,
    Param XParam,
    Model < float > XModel_cpu,
    Model < float > XModel_gpu
) 
```




<hr>



### function CopytoGPU&lt; int &gt; 

```C++
template void CopytoGPU< int > (
    int nblk,
    int blksize,
    int * z_cpu,
    int * z_gpu
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



### function SetupGPU&lt; double &gt; 

```C++
template void SetupGPU< double > (
    Param & XParam,
    Model < double > XModel,
    Forcing < float > & XForcing,
    Model < double > & XModel_g
) 
```




<hr>



### function SetupGPU&lt; float &gt; 

```C++
template void SetupGPU< float > (
    Param & XParam,
    Model < float > XModel,
    Forcing < float > & XForcing,
    Model < float > & XModel_g
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Setup_GPU.cu`


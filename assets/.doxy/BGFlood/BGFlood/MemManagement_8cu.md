

# File MemManagement.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**MemManagement.cu**](MemManagement_8cu.md)

[Go to the source code of this file](MemManagement_8cu_source.md)



* `#include "MemManagement.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**AllocateCPU**](#function-allocatecpu) (int nx, int ny, T \*& zb) <br> |
|  \_\_host\_\_ void | [**AllocateCPU**](#function-allocatecpu) (int nx, int ny, T \*& zs, T \*& h, T \*& u, T \*& v) <br> |
|  \_\_host\_\_ void | [**AllocateCPU**](#function-allocatecpu) (int nx, int ny, T \*& zs, T \*& h, T \*& u, T \*& v, T \*& U, T \*& hU) <br> |
|  \_\_host\_\_ void | [**AllocateCPU**](#function-allocatecpu) (int nx, int ny, [**GradientsP**](structGradientsP.md)&lt; T &gt; & Grad) <br> |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nblk, int blksize, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & Ev) <br> |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nblk, int blksize, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; & Ev) <br> |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  template void | [**AllocateCPU&lt; double &gt;**](#function-allocatecpu-double) (int nx, int ny, double \*& zs, double \*& h, double \*& u, double \*& v) <br> |
|  template void | [**AllocateCPU&lt; double &gt;**](#function-allocatecpu-double) (int nx, int ny, double \*& zs, double \*& h, double \*& u, double \*& v, double \*& U, double \*& hU) <br> |
|  template void | [**AllocateCPU&lt; double &gt;**](#function-allocatecpu-double) (int nx, int ny, [**GradientsP**](structGradientsP.md)&lt; double &gt; & Grad) <br> |
|  template void | [**AllocateCPU&lt; double &gt;**](#function-allocatecpu-double) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template void | [**AllocateCPU&lt; float &gt;**](#function-allocatecpu-float) (int nx, int ny, float \*& zs, float \*& h, float \*& u, float \*& v) <br> |
|  template void | [**AllocateCPU&lt; float &gt;**](#function-allocatecpu-float) (int nx, int ny, float \*& zs, float \*& h, float \*& u, float \*& v, float \*& U, float \*& hU) <br> |
|  template void | [**AllocateCPU&lt; float &gt;**](#function-allocatecpu-float) (int nx, int ny, [**GradientsP**](structGradientsP.md)&lt; float &gt; & Grad) <br> |
|  template void | [**AllocateCPU&lt; float &gt;**](#function-allocatecpu-float) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; float &gt; & XModel) <br> |
|  template void | [**AllocateCPU&lt; int &gt;**](#function-allocatecpu-int) (int nx, int ny, int \*& zs, int \*& h, int \*& u, int \*& v) <br> |
|  template void | [**AllocateCPU&lt; int &gt;**](#function-allocatecpu-int) (int nx, int ny, int \*& zs, int \*& h, int \*& u, int \*& v, int \*& U, int \*& hU) <br> |
|  void | [**AllocateGPU**](#function-allocategpu) (int nx, int ny, T \*& z\_g) <br> |
|  void | [**AllocateGPU**](#function-allocategpu) (int nx, int ny, T \*& zs, T \*& h, T \*& u, T \*& v) <br> |
|  void | [**AllocateGPU**](#function-allocategpu) (int nx, int ny, T \*& zs, T \*& h, T \*& u, T \*& v, T \*& U, T \*& hU) <br> |
|  void | [**AllocateGPU**](#function-allocategpu) (int nx, int ny, [**GradientsP**](structGradientsP.md)&lt; T &gt; & Grad) <br> |
|  void | [**AllocateGPU**](#function-allocategpu) (int nblk, int blksize, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & Ev) <br> |
|  void | [**AllocateGPU**](#function-allocategpu) (int nblk, int blksize, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; & Ev) <br> |
|  void | [**AllocateGPU**](#function-allocategpu) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  template void | [**AllocateGPU&lt; double &gt;**](#function-allocategpu-double) (int nx, int ny, double \*& zs, double \*& h, double \*& u, double \*& v) <br> |
|  template void | [**AllocateGPU&lt; double &gt;**](#function-allocategpu-double) (int nx, int ny, double \*& zs, double \*& h, double \*& u, double \*& v, double \*& U, double \*& hU) <br> |
|  template void | [**AllocateGPU&lt; double &gt;**](#function-allocategpu-double) (int nx, int ny, [**GradientsP**](structGradientsP.md)&lt; double &gt; & Grad) <br> |
|  template void | [**AllocateGPU&lt; double &gt;**](#function-allocategpu-double) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template void | [**AllocateGPU&lt; float &gt;**](#function-allocategpu-float) (int nx, int ny, float \*& zs, float \*& h, float \*& u, float \*& v) <br> |
|  template void | [**AllocateGPU&lt; float &gt;**](#function-allocategpu-float) (int nx, int ny, float \*& zs, float \*& h, float \*& u, float \*& v, float \*& U, float \*& hU) <br> |
|  template void | [**AllocateGPU&lt; float &gt;**](#function-allocategpu-float) (int nx, int ny, [**GradientsP**](structGradientsP.md)&lt; float &gt; & Grad) <br> |
|  template void | [**AllocateGPU&lt; float &gt;**](#function-allocategpu-float) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; float &gt; & XModel) <br> |
|  template void | [**AllocateGPU&lt; int &gt;**](#function-allocategpu-int) (int nx, int ny, int \*& zs, int \*& h, int \*& u, int \*& v) <br> |
|  template void | [**AllocateGPU&lt; int &gt;**](#function-allocategpu-int) (int nx, int ny, int \*& zs, int \*& h, int \*& u, int \*& v, int \*& U, int \*& hU) <br> |
|  void | [**AllocateMappedMemCPU**](#function-allocatemappedmemcpu) (int nx, int ny, int gpudevice, T \*& z) <br> |
|  template void | [**AllocateMappedMemCPU&lt; double &gt;**](#function-allocatemappedmemcpu-double) (int nx, int ny, int gpudevice, double \*& z) <br> |
|  template void | [**AllocateMappedMemCPU&lt; float &gt;**](#function-allocatemappedmemcpu-float) (int nx, int ny, int gpudevice, float \*& z) <br> |
|  template void | [**AllocateMappedMemCPU&lt; int &gt;**](#function-allocatemappedmemcpu-int) (int nx, int ny, int gpudevice, int \*& z) <br> |
|  void | [**AllocateMappedMemGPU**](#function-allocatemappedmemgpu) (int nx, int ny, int gpudevice, T \*& z\_g, T \* z) <br> |
|  template void | [**AllocateMappedMemGPU&lt; double &gt;**](#function-allocatemappedmemgpu-double) (int nx, int ny, int gpudevice, double \*& z\_g, double \* z) <br> |
|  template void | [**AllocateMappedMemGPU&lt; float &gt;**](#function-allocatemappedmemgpu-float) (int nx, int ny, int gpudevice, float \*& z\_g, float \* z) <br> |
|  template void | [**AllocateMappedMemGPU&lt; int &gt;**](#function-allocatemappedmemgpu-int) (int nx, int ny, int gpudevice, int \*& z\_g, int \* z) <br> |
|  \_\_host\_\_ void | [**FillCPU**](#function-fillcpu) (int nx, int ny, T fillval, T \*& zb) <br> |
|  template void | [**FillCPU&lt; double &gt;**](#function-fillcpu-double) (int nx, int ny, double fillval, double \*& zb) <br> |
|  template void | [**FillCPU&lt; float &gt;**](#function-fillcpu-float) (int nx, int ny, float fillval, float \*& zb) <br> |
|  template void | [**FillCPU&lt; int &gt;**](#function-fillcpu-int) (int nx, int ny, int fillval, int \*& zb) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, T \*& zb) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, T \*& zs, T \*& h, T \*& u, T \*& v) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, T \*& zs, T \*& h, T \*& u, T \*& v, T \*& U, T \*& hU) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & Ev) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; & Ev) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  template void | [**ReallocArray&lt; double &gt;**](#function-reallocarray-double) (int nblk, int blksize, double \*& zs, double \*& h, double \*& u, double \*& v) <br> |
|  template void | [**ReallocArray&lt; double &gt;**](#function-reallocarray-double) (int nblk, int blksize, double \*& zs, double \*& h, double \*& u, double \*& v, double \*& U, double \*& hU) <br> |
|  template void | [**ReallocArray&lt; double &gt;**](#function-reallocarray-double) (int nblk, int blksize, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; & Ev) <br> |
|  template void | [**ReallocArray&lt; double &gt;**](#function-reallocarray-double) (int nblk, int blksize, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; double &gt; & Ev) <br> |
|  template void | [**ReallocArray&lt; double &gt;**](#function-reallocarray-double) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template void | [**ReallocArray&lt; float &gt;**](#function-reallocarray-float) (int nblk, int blksize, float \*& zs, float \*& h, float \*& u, float \*& v) <br> |
|  template void | [**ReallocArray&lt; float &gt;**](#function-reallocarray-float) (int nblk, int blksize, float \*& zs, float \*& h, float \*& u, float \*& v, float \*& U, float \*& hU) <br> |
|  template void | [**ReallocArray&lt; float &gt;**](#function-reallocarray-float) (int nblk, int blksize, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; & Ev) <br> |
|  template void | [**ReallocArray&lt; float &gt;**](#function-reallocarray-float) (int nblk, int blksize, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; float &gt; & Ev) <br> |
|  template void | [**ReallocArray&lt; float &gt;**](#function-reallocarray-float) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; float &gt; & XModel) <br> |
|  template void | [**ReallocArray&lt; int &gt;**](#function-reallocarray-int) (int nblk, int blksize, int \*& zs, int \*& h, int \*& u, int \*& v) <br> |
|  template void | [**ReallocArray&lt; int &gt;**](#function-reallocarray-int) (int nblk, int blksize, int \*& zs, int \*& h, int \*& u, int \*& v, int \*& U, int \*& hU) <br> |
|  \_\_host\_\_ int | [**memloc**](#function-memloc) ([**Param**](classParam.md) XParam, int i, int j, int ib) <br> |
|  \_\_host\_\_ \_\_device\_\_ int | [**memloc**](#function-memloc) (int halowidth, int blkmemwidth, int i, int j, int ib) <br> |



























## Macros

| Type | Name |
| ---: | :--- |
| define  | [**ALIGN\_UP**](MemManagement_8cu.md#define-align_up) (x, size) `( ((size\_t)x+(size-1))&(~(size-1)) )`<br> |
| define  | [**MEMORY\_ALIGNMENT**](MemManagement_8cu.md#define-memory_alignment)  `4096`<br> |

## Public Functions Documentation




### function AllocateCPU 

```C++
template<class T>
__host__ void AllocateCPU (
    int nx,
    int ny,
    T *& zb
) 
```




<hr>



### function AllocateCPU 

```C++
template<class T>
__host__ void AllocateCPU (
    int nx,
    int ny,
    T *& zs,
    T *& h,
    T *& u,
    T *& v
) 
```




<hr>



### function AllocateCPU 

```C++
template<class T>
__host__ void AllocateCPU (
    int nx,
    int ny,
    T *& zs,
    T *& h,
    T *& u,
    T *& v,
    T *& U,
    T *& hU
) 
```




<hr>



### function AllocateCPU 

```C++
template<class T>
__host__ void AllocateCPU (
    int nx,
    int ny,
    GradientsP < T > & Grad
) 
```




<hr>



### function AllocateCPU 

```C++
template<class T>
void AllocateCPU (
    int nblk,
    int blksize,
    EvolvingP < T > & Ev
) 
```




<hr>



### function AllocateCPU 

```C++
template<class T>
void AllocateCPU (
    int nblk,
    int blksize,
    EvolvingP_M < T > & Ev
) 
```




<hr>



### function AllocateCPU 

```C++
template<class T>
void AllocateCPU (
    int nblk,
    int blksize,
    Param XParam,
    Model < T > & XModel
) 
```




<hr>



### function AllocateCPU&lt; double &gt; 

```C++
template void AllocateCPU< double > (
    int nx,
    int ny,
    double *& zs,
    double *& h,
    double *& u,
    double *& v
) 
```




<hr>



### function AllocateCPU&lt; double &gt; 

```C++
template void AllocateCPU< double > (
    int nx,
    int ny,
    double *& zs,
    double *& h,
    double *& u,
    double *& v,
    double *& U,
    double *& hU
) 
```




<hr>



### function AllocateCPU&lt; double &gt; 

```C++
template void AllocateCPU< double > (
    int nx,
    int ny,
    GradientsP < double > & Grad
) 
```




<hr>



### function AllocateCPU&lt; double &gt; 

```C++
template void AllocateCPU< double > (
    int nblk,
    int blksize,
    Param XParam,
    Model < double > & XModel
) 
```




<hr>



### function AllocateCPU&lt; float &gt; 

```C++
template void AllocateCPU< float > (
    int nx,
    int ny,
    float *& zs,
    float *& h,
    float *& u,
    float *& v
) 
```




<hr>



### function AllocateCPU&lt; float &gt; 

```C++
template void AllocateCPU< float > (
    int nx,
    int ny,
    float *& zs,
    float *& h,
    float *& u,
    float *& v,
    float *& U,
    float *& hU
) 
```




<hr>



### function AllocateCPU&lt; float &gt; 

```C++
template void AllocateCPU< float > (
    int nx,
    int ny,
    GradientsP < float > & Grad
) 
```




<hr>



### function AllocateCPU&lt; float &gt; 

```C++
template void AllocateCPU< float > (
    int nblk,
    int blksize,
    Param XParam,
    Model < float > & XModel
) 
```




<hr>



### function AllocateCPU&lt; int &gt; 

```C++
template void AllocateCPU< int > (
    int nx,
    int ny,
    int *& zs,
    int *& h,
    int *& u,
    int *& v
) 
```




<hr>



### function AllocateCPU&lt; int &gt; 

```C++
template void AllocateCPU< int > (
    int nx,
    int ny,
    int *& zs,
    int *& h,
    int *& u,
    int *& v,
    int *& U,
    int *& hU
) 
```




<hr>



### function AllocateGPU 

```C++
template<class T>
void AllocateGPU (
    int nx,
    int ny,
    T *& z_g
) 
```




<hr>



### function AllocateGPU 

```C++
template<class T>
void AllocateGPU (
    int nx,
    int ny,
    T *& zs,
    T *& h,
    T *& u,
    T *& v
) 
```




<hr>



### function AllocateGPU 

```C++
template<class T>
void AllocateGPU (
    int nx,
    int ny,
    T *& zs,
    T *& h,
    T *& u,
    T *& v,
    T *& U,
    T *& hU
) 
```




<hr>



### function AllocateGPU 

```C++
template<class T>
void AllocateGPU (
    int nx,
    int ny,
    GradientsP < T > & Grad
) 
```




<hr>



### function AllocateGPU 

```C++
template<class T>
void AllocateGPU (
    int nblk,
    int blksize,
    EvolvingP < T > & Ev
) 
```




<hr>



### function AllocateGPU 

```C++
template<class T>
void AllocateGPU (
    int nblk,
    int blksize,
    EvolvingP_M < T > & Ev
) 
```




<hr>



### function AllocateGPU 

```C++
template<class T>
void AllocateGPU (
    int nblk,
    int blksize,
    Param XParam,
    Model < T > & XModel
) 
```




<hr>



### function AllocateGPU&lt; double &gt; 

```C++
template void AllocateGPU< double > (
    int nx,
    int ny,
    double *& zs,
    double *& h,
    double *& u,
    double *& v
) 
```




<hr>



### function AllocateGPU&lt; double &gt; 

```C++
template void AllocateGPU< double > (
    int nx,
    int ny,
    double *& zs,
    double *& h,
    double *& u,
    double *& v,
    double *& U,
    double *& hU
) 
```




<hr>



### function AllocateGPU&lt; double &gt; 

```C++
template void AllocateGPU< double > (
    int nx,
    int ny,
    GradientsP < double > & Grad
) 
```




<hr>



### function AllocateGPU&lt; double &gt; 

```C++
template void AllocateGPU< double > (
    int nblk,
    int blksize,
    Param XParam,
    Model < double > & XModel
) 
```




<hr>



### function AllocateGPU&lt; float &gt; 

```C++
template void AllocateGPU< float > (
    int nx,
    int ny,
    float *& zs,
    float *& h,
    float *& u,
    float *& v
) 
```




<hr>



### function AllocateGPU&lt; float &gt; 

```C++
template void AllocateGPU< float > (
    int nx,
    int ny,
    float *& zs,
    float *& h,
    float *& u,
    float *& v,
    float *& U,
    float *& hU
) 
```




<hr>



### function AllocateGPU&lt; float &gt; 

```C++
template void AllocateGPU< float > (
    int nx,
    int ny,
    GradientsP < float > & Grad
) 
```




<hr>



### function AllocateGPU&lt; float &gt; 

```C++
template void AllocateGPU< float > (
    int nblk,
    int blksize,
    Param XParam,
    Model < float > & XModel
) 
```




<hr>



### function AllocateGPU&lt; int &gt; 

```C++
template void AllocateGPU< int > (
    int nx,
    int ny,
    int *& zs,
    int *& h,
    int *& u,
    int *& v
) 
```




<hr>



### function AllocateGPU&lt; int &gt; 

```C++
template void AllocateGPU< int > (
    int nx,
    int ny,
    int *& zs,
    int *& h,
    int *& u,
    int *& v,
    int *& U,
    int *& hU
) 
```




<hr>



### function AllocateMappedMemCPU 

```C++
template<class T>
void AllocateMappedMemCPU (
    int nx,
    int ny,
    int gpudevice,
    T *& z
) 
```




<hr>



### function AllocateMappedMemCPU&lt; double &gt; 

```C++
template void AllocateMappedMemCPU< double > (
    int nx,
    int ny,
    int gpudevice,
    double *& z
) 
```




<hr>



### function AllocateMappedMemCPU&lt; float &gt; 

```C++
template void AllocateMappedMemCPU< float > (
    int nx,
    int ny,
    int gpudevice,
    float *& z
) 
```




<hr>



### function AllocateMappedMemCPU&lt; int &gt; 

```C++
template void AllocateMappedMemCPU< int > (
    int nx,
    int ny,
    int gpudevice,
    int *& z
) 
```




<hr>



### function AllocateMappedMemGPU 

```C++
template<class T>
void AllocateMappedMemGPU (
    int nx,
    int ny,
    int gpudevice,
    T *& z_g,
    T * z
) 
```




<hr>



### function AllocateMappedMemGPU&lt; double &gt; 

```C++
template void AllocateMappedMemGPU< double > (
    int nx,
    int ny,
    int gpudevice,
    double *& z_g,
    double * z
) 
```




<hr>



### function AllocateMappedMemGPU&lt; float &gt; 

```C++
template void AllocateMappedMemGPU< float > (
    int nx,
    int ny,
    int gpudevice,
    float *& z_g,
    float * z
) 
```




<hr>



### function AllocateMappedMemGPU&lt; int &gt; 

```C++
template void AllocateMappedMemGPU< int > (
    int nx,
    int ny,
    int gpudevice,
    int *& z_g,
    int * z
) 
```




<hr>



### function FillCPU 

```C++
template<class T>
__host__ void FillCPU (
    int nx,
    int ny,
    T fillval,
    T *& zb
) 
```




<hr>



### function FillCPU&lt; double &gt; 

```C++
template void FillCPU< double > (
    int nx,
    int ny,
    double fillval,
    double *& zb
) 
```




<hr>



### function FillCPU&lt; float &gt; 

```C++
template void FillCPU< float > (
    int nx,
    int ny,
    float fillval,
    float *& zb
) 
```




<hr>



### function FillCPU&lt; int &gt; 

```C++
template void FillCPU< int > (
    int nx,
    int ny,
    int fillval,
    int *& zb
) 
```




<hr>



### function ReallocArray 

```C++
template<class T>
void ReallocArray (
    int nblk,
    int blksize,
    T *& zb
) 
```




<hr>



### function ReallocArray 

```C++
template<class T>
void ReallocArray (
    int nblk,
    int blksize,
    T *& zs,
    T *& h,
    T *& u,
    T *& v
) 
```




<hr>



### function ReallocArray 

```C++
template<class T>
void ReallocArray (
    int nblk,
    int blksize,
    T *& zs,
    T *& h,
    T *& u,
    T *& v,
    T *& U,
    T *& hU
) 
```




<hr>



### function ReallocArray 

```C++
template<class T>
void ReallocArray (
    int nblk,
    int blksize,
    EvolvingP < T > & Ev
) 
```




<hr>



### function ReallocArray 

```C++
template<class T>
void ReallocArray (
    int nblk,
    int blksize,
    EvolvingP_M < T > & Ev
) 
```




<hr>



### function ReallocArray 

```C++
template<class T>
void ReallocArray (
    int nblk,
    int blksize,
    Param XParam,
    Model < T > & XModel
) 
```




<hr>



### function ReallocArray&lt; double &gt; 

```C++
template void ReallocArray< double > (
    int nblk,
    int blksize,
    double *& zs,
    double *& h,
    double *& u,
    double *& v
) 
```




<hr>



### function ReallocArray&lt; double &gt; 

```C++
template void ReallocArray< double > (
    int nblk,
    int blksize,
    double *& zs,
    double *& h,
    double *& u,
    double *& v,
    double *& U,
    double *& hU
) 
```




<hr>



### function ReallocArray&lt; double &gt; 

```C++
template void ReallocArray< double > (
    int nblk,
    int blksize,
    EvolvingP < double > & Ev
) 
```




<hr>



### function ReallocArray&lt; double &gt; 

```C++
template void ReallocArray< double > (
    int nblk,
    int blksize,
    EvolvingP_M < double > & Ev
) 
```




<hr>



### function ReallocArray&lt; double &gt; 

```C++
template void ReallocArray< double > (
    int nblk,
    int blksize,
    Param XParam,
    Model < double > & XModel
) 
```




<hr>



### function ReallocArray&lt; float &gt; 

```C++
template void ReallocArray< float > (
    int nblk,
    int blksize,
    float *& zs,
    float *& h,
    float *& u,
    float *& v
) 
```




<hr>



### function ReallocArray&lt; float &gt; 

```C++
template void ReallocArray< float > (
    int nblk,
    int blksize,
    float *& zs,
    float *& h,
    float *& u,
    float *& v,
    float *& U,
    float *& hU
) 
```




<hr>



### function ReallocArray&lt; float &gt; 

```C++
template void ReallocArray< float > (
    int nblk,
    int blksize,
    EvolvingP < float > & Ev
) 
```




<hr>



### function ReallocArray&lt; float &gt; 

```C++
template void ReallocArray< float > (
    int nblk,
    int blksize,
    EvolvingP_M < float > & Ev
) 
```




<hr>



### function ReallocArray&lt; float &gt; 

```C++
template void ReallocArray< float > (
    int nblk,
    int blksize,
    Param XParam,
    Model < float > & XModel
) 
```




<hr>



### function ReallocArray&lt; int &gt; 

```C++
template void ReallocArray< int > (
    int nblk,
    int blksize,
    int *& zs,
    int *& h,
    int *& u,
    int *& v
) 
```




<hr>



### function ReallocArray&lt; int &gt; 

```C++
template void ReallocArray< int > (
    int nblk,
    int blksize,
    int *& zs,
    int *& h,
    int *& u,
    int *& v,
    int *& U,
    int *& hU
) 
```




<hr>



### function memloc 

```C++
__host__ int memloc (
    Param XParam,
    int i,
    int j,
    int ib
) 
```




<hr>



### function memloc 

```C++
__host__ __device__ int memloc (
    int halowidth,
    int blkmemwidth,
    int i,
    int j,
    int ib
) 
```




<hr>
## Macro Definition Documentation





### define ALIGN\_UP 

```C++
#define ALIGN_UP (
    x,
    size
) `( ((size_t)x+(size-1))&(~(size-1)) )`
```




<hr>



### define MEMORY\_ALIGNMENT 

```C++
#define MEMORY_ALIGNMENT `4096`
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/MemManagement.cu`


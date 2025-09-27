

# File MemManagement.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**MemManagement.h**](MemManagement_8h.md)

[Go to the source code of this file](MemManagement_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Setup_GPU.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nx, int ny, T \*& zb) <br> |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nx, int ny, T \*& zs, T \*& h, T \*& u, T \*& v) <br> |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nx, int ny, T \*& zs, T \*& h, T \*& u, T \*& v, T \*& U, T \*& hU) <br> |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nx, int ny, [**GradientsP**](structGradientsP.md)&lt; T &gt; & Grad) <br> |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nblk, int blksize, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & Ev) <br> |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nblk, int blksize, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; & Ev) <br> |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  void | [**AllocateGPU**](#function-allocategpu) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  void | [**AllocateGPU**](#function-allocategpu) (int nx, int ny, T \*& z\_g) <br> |
|  void | [**AllocateMappedMemCPU**](#function-allocatemappedmemcpu) (int nx, int ny, int gpudevice, T \*& z) <br> |
|  void | [**AllocateMappedMemGPU**](#function-allocatemappedmemgpu) (int nx, int ny, int gpudevice, T \*& z\_g, T \* z) <br> |
|  \_\_host\_\_ void | [**FillCPU**](#function-fillcpu) (int nx, int ny, T fillval, T \*& zb) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, T \*& zb) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, T \*& zs, T \*& h, T \*& u, T \*& v) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, T \*& zs, T \*& h, T \*& u, T \*& v, T \*& U, T \*& hU) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & Ev) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; & Ev) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, [**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  int | [**memloc**](#function-memloc) ([**Param**](classParam.md) XParam, int i, int j, int ib) <br> |
|  \_\_host\_\_ \_\_device\_\_ int | [**memloc**](#function-memloc) (int halowidth, int blkmemwidth, int i, int j, int ib) <br> |




























## Public Functions Documentation




### function AllocateCPU 

```C++
template<class T>
void AllocateCPU (
    int nx,
    int ny,
    T *& zb
) 
```




<hr>



### function AllocateCPU 

```C++
template<class T>
void AllocateCPU (
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
void AllocateCPU (
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
void AllocateCPU (
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



### function memloc 

```C++
int memloc (
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

------------------------------
The documentation for this class was generated from the following file `src/MemManagement.h`


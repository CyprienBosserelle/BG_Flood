

# File MemManagement.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**MemManagement.cu**](_mem_management_8cu.md)

[Go to the source code of this file](_mem_management_8cu_source.md)



* `#include "MemManagement.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**AllocateCPU**](#function-allocatecpu) (int nx, int ny, T \*& zb) <br>_Allocate memory for a single array on the CPU._  |
|  \_\_host\_\_ void | [**AllocateCPU**](#function-allocatecpu) (int nx, int ny, T \*& zs, T \*& h, T \*& u, T \*& v) <br>_Allocate memory for multiple arrays (zs, h, u, v) on the CPU._  |
|  \_\_host\_\_ void | [**AllocateCPU**](#function-allocatecpu) (int nx, int ny, T \*& zs, T \*& h, T \*& u, T \*& v, T \*& U, T \*& hU) <br>_Allocate memory for extended arrays (zs, h, u, v, U, hU) on the CPU._  |
|  \_\_host\_\_ void | [**AllocateCPU**](#function-allocatecpu) (int nx, int ny, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; & Grad) <br>_Allocate memory for gradient arrays on the CPU._  |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nblk, int blksize, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & Ev) <br>_Allocate memory for evolving variables on the CPU._  |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nblk, int blksize, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; T &gt; & Ev) <br>_Allocate memory for extended evolving variables on the CPU._  |
|  void | [**AllocateCPU**](#function-allocatecpu) (int nblk, int blksize, [**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Allocate all model arrays on the CPU._  |
|  template void | [**AllocateCPU&lt; double &gt;**](#function-allocatecpu-double) (int nx, int ny, double \*& zs, double \*& h, double \*& u, double \*& v) <br> |
|  template void | [**AllocateCPU&lt; double &gt;**](#function-allocatecpu-double) (int nx, int ny, double \*& zs, double \*& h, double \*& u, double \*& v, double \*& U, double \*& hU) <br> |
|  template void | [**AllocateCPU&lt; double &gt;**](#function-allocatecpu-double) (int nx, int ny, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; & Grad) <br> |
|  template void | [**AllocateCPU&lt; double &gt;**](#function-allocatecpu-double) (int nblk, int blksize, [**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; double &gt; & XModel) <br> |
|  template void | [**AllocateCPU&lt; float &gt;**](#function-allocatecpu-float) (int nx, int ny, float \*& zs, float \*& h, float \*& u, float \*& v) <br> |
|  template void | [**AllocateCPU&lt; float &gt;**](#function-allocatecpu-float) (int nx, int ny, float \*& zs, float \*& h, float \*& u, float \*& v, float \*& U, float \*& hU) <br> |
|  template void | [**AllocateCPU&lt; float &gt;**](#function-allocatecpu-float) (int nx, int ny, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; & Grad) <br> |
|  template void | [**AllocateCPU&lt; float &gt;**](#function-allocatecpu-float) (int nblk, int blksize, [**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; float &gt; & XModel) <br> |
|  template void | [**AllocateCPU&lt; int &gt;**](#function-allocatecpu-int) (int nx, int ny, int \*& zs, int \*& h, int \*& u, int \*& v) <br> |
|  template void | [**AllocateCPU&lt; int &gt;**](#function-allocatecpu-int) (int nx, int ny, int \*& zs, int \*& h, int \*& u, int \*& v, int \*& U, int \*& hU) <br> |
|  void | [**AllocateGPU**](#function-allocategpu) (int nx, int ny, T \*& z\_g) <br>_Allocate memory on the GPU for a single array._  |
|  void | [**AllocateGPU**](#function-allocategpu) (int nx, int ny, T \*& zs, T \*& h, T \*& u, T \*& v) <br>_Allocate memory on the GPU for multiple arrays (zs, h, u, v)._  |
|  void | [**AllocateGPU**](#function-allocategpu) (int nx, int ny, T \*& zs, T \*& h, T \*& u, T \*& v, T \*& U, T \*& hU) <br>_Allocate memory on the GPU for extended arrays (zs, h, u, v, U, hU)._  |
|  void | [**AllocateGPU**](#function-allocategpu) (int nx, int ny, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; & Grad) <br>_Allocate memory on the GPU for gradient arrays._  |
|  void | [**AllocateGPU**](#function-allocategpu) (int nblk, int blksize, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & Ev) <br>_Allocate memory on the GPU for evolving variables structure._  |
|  void | [**AllocateGPU**](#function-allocategpu) (int nblk, int blksize, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; T &gt; & Ev) <br>_Allocate memory on the GPU for extended evolving variables structure._  |
|  void | [**AllocateGPU**](#function-allocategpu) (int nblk, int blksize, [**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Allocate all model arrays on the GPU._  |
|  template void | [**AllocateGPU&lt; double &gt;**](#function-allocategpu-double) (int nx, int ny, double \*& zs, double \*& h, double \*& u, double \*& v) <br> |
|  template void | [**AllocateGPU&lt; double &gt;**](#function-allocategpu-double) (int nx, int ny, double \*& zs, double \*& h, double \*& u, double \*& v, double \*& U, double \*& hU) <br> |
|  template void | [**AllocateGPU&lt; double &gt;**](#function-allocategpu-double) (int nx, int ny, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; & Grad) <br> |
|  template void | [**AllocateGPU&lt; double &gt;**](#function-allocategpu-double) (int nblk, int blksize, [**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; double &gt; & XModel) <br> |
|  template void | [**AllocateGPU&lt; float &gt;**](#function-allocategpu-float) (int nx, int ny, float \*& zs, float \*& h, float \*& u, float \*& v) <br> |
|  template void | [**AllocateGPU&lt; float &gt;**](#function-allocategpu-float) (int nx, int ny, float \*& zs, float \*& h, float \*& u, float \*& v, float \*& U, float \*& hU) <br> |
|  template void | [**AllocateGPU&lt; float &gt;**](#function-allocategpu-float) (int nx, int ny, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; & Grad) <br> |
|  template void | [**AllocateGPU&lt; float &gt;**](#function-allocategpu-float) (int nblk, int blksize, [**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; float &gt; & XModel) <br> |
|  template void | [**AllocateGPU&lt; int &gt;**](#function-allocategpu-int) (int nx, int ny, int \*& zs, int \*& h, int \*& u, int \*& v) <br> |
|  template void | [**AllocateGPU&lt; int &gt;**](#function-allocategpu-int) (int nx, int ny, int \*& zs, int \*& h, int \*& u, int \*& v, int \*& U, int \*& hU) <br> |
|  void | [**AllocateMappedMemCPU**](#function-allocatemappedmemcpu) (int nx, int ny, int gpudevice, T \*& z) <br>_Allocate mapped memory on the CPU for CUDA interop._  |
|  template void | [**AllocateMappedMemCPU&lt; double &gt;**](#function-allocatemappedmemcpu-double) (int nx, int ny, int gpudevice, double \*& z) <br> |
|  template void | [**AllocateMappedMemCPU&lt; float &gt;**](#function-allocatemappedmemcpu-float) (int nx, int ny, int gpudevice, float \*& z) <br> |
|  template void | [**AllocateMappedMemCPU&lt; int &gt;**](#function-allocatemappedmemcpu-int) (int nx, int ny, int gpudevice, int \*& z) <br> |
|  void | [**AllocateMappedMemGPU**](#function-allocatemappedmemgpu) (int nx, int ny, int gpudevice, T \*& z\_g, T \* z) <br>_Get device pointer for mapped host memory._  |
|  template void | [**AllocateMappedMemGPU&lt; double &gt;**](#function-allocatemappedmemgpu-double) (int nx, int ny, int gpudevice, double \*& z\_g, double \* z) <br> |
|  template void | [**AllocateMappedMemGPU&lt; float &gt;**](#function-allocatemappedmemgpu-float) (int nx, int ny, int gpudevice, float \*& z\_g, float \* z) <br> |
|  template void | [**AllocateMappedMemGPU&lt; int &gt;**](#function-allocatemappedmemgpu-int) (int nx, int ny, int gpudevice, int \*& z\_g, int \* z) <br> |
|  \_\_host\_\_ void | [**FillCPU**](#function-fillcpu) (int nx, int ny, T fillval, T \*& zb) <br>_Fill a CPU array with a specified value._  |
|  template void | [**FillCPU&lt; double &gt;**](#function-fillcpu-double) (int nx, int ny, double fillval, double \*& zb) <br> |
|  template void | [**FillCPU&lt; float &gt;**](#function-fillcpu-float) (int nx, int ny, float fillval, float \*& zb) <br> |
|  template void | [**FillCPU&lt; int &gt;**](#function-fillcpu-int) (int nx, int ny, int fillval, int \*& zb) <br> |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, T \*& zb) <br>_Reallocate memory for a single array._  |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, T \*& zs, T \*& h, T \*& u, T \*& v) <br>_Reallocate memory for multiple arrays (zs, h, u, v)._  |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, T \*& zs, T \*& h, T \*& u, T \*& v, T \*& U, T \*& hU) <br>_Reallocate memory for extended arrays (zs, h, u, v, U, hU)._  |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & Ev) <br>_Reallocate memory for evolving variables structure._  |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; T &gt; & Ev) <br>_Reallocate memory for extended evolving variables structure._  |
|  void | [**ReallocArray**](#function-reallocarray) (int nblk, int blksize, [**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Reallocate all model arrays._  |
|  template void | [**ReallocArray&lt; double &gt;**](#function-reallocarray-double) (int nblk, int blksize, double \*& zs, double \*& h, double \*& u, double \*& v) <br> |
|  template void | [**ReallocArray&lt; double &gt;**](#function-reallocarray-double) (int nblk, int blksize, double \*& zs, double \*& h, double \*& u, double \*& v, double \*& U, double \*& hU) <br> |
|  template void | [**ReallocArray&lt; double &gt;**](#function-reallocarray-double) (int nblk, int blksize, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; & Ev) <br> |
|  template void | [**ReallocArray&lt; double &gt;**](#function-reallocarray-double) (int nblk, int blksize, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; double &gt; & Ev) <br> |
|  template void | [**ReallocArray&lt; double &gt;**](#function-reallocarray-double) (int nblk, int blksize, [**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; double &gt; & XModel) <br> |
|  template void | [**ReallocArray&lt; float &gt;**](#function-reallocarray-float) (int nblk, int blksize, float \*& zs, float \*& h, float \*& u, float \*& v) <br> |
|  template void | [**ReallocArray&lt; float &gt;**](#function-reallocarray-float) (int nblk, int blksize, float \*& zs, float \*& h, float \*& u, float \*& v, float \*& U, float \*& hU) <br> |
|  template void | [**ReallocArray&lt; float &gt;**](#function-reallocarray-float) (int nblk, int blksize, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; & Ev) <br> |
|  template void | [**ReallocArray&lt; float &gt;**](#function-reallocarray-float) (int nblk, int blksize, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; float &gt; & Ev) <br> |
|  template void | [**ReallocArray&lt; float &gt;**](#function-reallocarray-float) (int nblk, int blksize, [**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; float &gt; & XModel) <br> |
|  template void | [**ReallocArray&lt; int &gt;**](#function-reallocarray-int) (int nblk, int blksize, int \*& zs, int \*& h, int \*& u, int \*& v) <br> |
|  template void | [**ReallocArray&lt; int &gt;**](#function-reallocarray-int) (int nblk, int blksize, int \*& zs, int \*& h, int \*& u, int \*& v, int \*& U, int \*& hU) <br> |
|  \_\_host\_\_ int | [**memloc**](#function-memloc) ([**Param**](class_param.md) XParam, int i, int j, int ib) <br>_Compute memory index for a cell in a block (using_ [_**Param**_](class_param.md) _)._ |
|  \_\_host\_\_ \_\_device\_\_ int | [**memloc**](#function-memloc) (int halowidth, int blkmemwidth, int i, int j, int ib) <br>_Compute memory index for a cell in a block (using explicit sizes)._  |



























## Macros

| Type | Name |
| ---: | :--- |
| define  | [**ALIGN\_UP**](_mem_management_8cu.md#define-align_up) (x, size) `( ((size\_t)x+(size-1))&(~(size-1)) )`<br> |
| define  | [**MEMORY\_ALIGNMENT**](_mem_management_8cu.md#define-memory_alignment)  `4096`<br> |

## Public Functions Documentation




### function AllocateCPU 

_Allocate memory for a single array on the CPU._ 
```C++
template<class T>
__host__ void AllocateCPU (
    int nx,
    int ny,
    T *& zb
) 
```



Allocates memory for the given array and checks for allocation failure.




**Template parameters:**


* `T` Data type (float or int) 



**Parameters:**


* `nx` Number of x elements 
* `ny` Number of y elements 
* `zb` Array to allocate 




        

<hr>



### function AllocateCPU 

_Allocate memory for multiple arrays (zs, h, u, v) on the CPU._ 
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



Allocates memory for the given arrays.




**Template parameters:**


* `T` Data type (float, double, int) 



**Parameters:**


* `nx` Number of x elements 
* `ny` Number of y elements 
* `zs` Array to allocate 
* `h` Array to allocate 
* `u` Array to allocate 
* `v` Array to allocate 




        

<hr>



### function AllocateCPU 

_Allocate memory for extended arrays (zs, h, u, v, U, hU) on the CPU._ 
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



Allocates memory for the given arrays.




**Template parameters:**


* `T` Data type (float, double, int) 



**Parameters:**


* `nx` Number of x elements 
* `ny` Number of y elements 
* `zs` Array to allocate 
* `h` Array to allocate 
* `u` Array to allocate 
* `v` Array to allocate 
* `U` Array to allocate 
* `hU` Array to allocate 




        

<hr>



### function AllocateCPU 

_Allocate memory for gradient arrays on the CPU._ 
```C++
template<class T>
__host__ void AllocateCPU (
    int nx,
    int ny,
    GradientsP < T > & Grad
) 
```



Allocates memory for all gradient arrays in [**GradientsP**](struct_gradients_p.md) structure.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nx` Number of x elements 
* `ny` Number of y elements 
* `Grad` [**GradientsP**](struct_gradients_p.md) structure to allocate 




        

<hr>



### function AllocateCPU 

_Allocate memory for evolving variables on the CPU._ 
```C++
template<class T>
void AllocateCPU (
    int nblk,
    int blksize,
    EvolvingP < T > & Ev
) 
```



Allocates memory for h, zs, u, v arrays in [**EvolvingP**](struct_evolving_p.md) structure.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `Ev` [**EvolvingP**](struct_evolving_p.md) structure to allocate 




        

<hr>



### function AllocateCPU 

_Allocate memory for extended evolving variables on the CPU._ 
```C++
template<class T>
void AllocateCPU (
    int nblk,
    int blksize,
    EvolvingP_M < T > & Ev
) 
```



Allocates memory for h, zs, u, v, U, hU arrays in [**EvolvingP\_M**](struct_evolving_p___m.md) structure.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `Ev` [**EvolvingP\_M**](struct_evolving_p___m.md) structure to allocate 




        

<hr>



### function AllocateCPU 

_Allocate all model arrays on the CPU._ 
```C++
template<class T>
void AllocateCPU (
    int nblk,
    int blksize,
    Param XParam,
    Model < T > & XModel
) 
```



Allocates memory for all arrays in the [**Model**](struct_model.md) structure, including blocks, gradients, fluxes, and output buffers.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) structure to allocate 




        

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

_Allocate memory on the GPU for a single array._ 
```C++
template<class T>
void AllocateGPU (
    int nx,
    int ny,
    T *& z_g
) 
```



Allocates device memory for the given array.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nx` Number of x elements 
* `ny` Number of y elements 
* `z_g` Device pointer output 




        

<hr>



### function AllocateGPU 

_Allocate memory on the GPU for multiple arrays (zs, h, u, v)._ 
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



Allocates device memory for the given arrays.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nx` Number of x elements 
* `ny` Number of y elements 
* `zs` Device pointer output 
* `h` Device pointer output 
* `u` Device pointer output 
* `v` Device pointer output 




        

<hr>



### function AllocateGPU 

_Allocate memory on the GPU for extended arrays (zs, h, u, v, U, hU)._ 
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



Allocates device memory for the given arrays.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nx` Number of x elements 
* `ny` Number of y elements 
* `zs` Device pointer output 
* `h` Device pointer output 
* `u` Device pointer output 
* `v` Device pointer output 
* `U` Device pointer output 
* `hU` Device pointer output 




        

<hr>



### function AllocateGPU 

_Allocate memory on the GPU for gradient arrays._ 
```C++
template<class T>
void AllocateGPU (
    int nx,
    int ny,
    GradientsP < T > & Grad
) 
```



Allocates device memory for all gradient arrays in [**GradientsP**](struct_gradients_p.md) structure.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nx` Number of x elements 
* `ny` Number of y elements 
* `Grad` [**GradientsP**](struct_gradients_p.md) structure to allocate 




        

<hr>



### function AllocateGPU 

_Allocate memory on the GPU for evolving variables structure._ 
```C++
template<class T>
void AllocateGPU (
    int nblk,
    int blksize,
    EvolvingP < T > & Ev
) 
```



Allocates device memory for all arrays in [**EvolvingP**](struct_evolving_p.md) structure.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `Ev` [**EvolvingP**](struct_evolving_p.md) structure to allocate 




        

<hr>



### function AllocateGPU 

_Allocate memory on the GPU for extended evolving variables structure._ 
```C++
template<class T>
void AllocateGPU (
    int nblk,
    int blksize,
    EvolvingP_M < T > & Ev
) 
```



Allocates device memory for all arrays in [**EvolvingP\_M**](struct_evolving_p___m.md) structure.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `Ev` [**EvolvingP\_M**](struct_evolving_p___m.md) structure to allocate 




        

<hr>



### function AllocateGPU 

_Allocate all model arrays on the GPU._ 
```C++
template<class T>
void AllocateGPU (
    int nblk,
    int blksize,
    Param XParam,
    Model < T > & XModel
) 
```



Allocates device memory for all arrays in the [**Model**](struct_model.md) structure, including blocks, gradients, fluxes, and output buffers.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) structure to allocate 




        

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

_Allocate mapped memory on the CPU for CUDA interop._ 
```C++
template<class T>
void AllocateMappedMemCPU (
    int nx,
    int ny,
    int gpudevice,
    T *& z
) 
```



Allocates pinned or mapped memory for CUDA host-device interoperation.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nx` Number of x elements 
* `ny` Number of y elements 
* `gpudevice` GPU device index 
* `z` Pointer to allocated memory 




        

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

_Get device pointer for mapped host memory._ 
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



Retrieves the device pointer for host memory mapped for CUDA interop.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nx` Number of x elements 
* `ny` Number of y elements 
* `gpudevice` GPU device index 
* `z_g` Device pointer output 
* `z` Host pointer input 




        

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

_Fill a CPU array with a specified value._ 
```C++
template<class T>
__host__ void FillCPU (
    int nx,
    int ny,
    T fillval,
    T *& zb
) 
```



Sets all elements of the array to the given fill value.




**Template parameters:**


* `T` Data type (float, double, int) 



**Parameters:**


* `nx` Number of x elements 
* `ny` Number of y elements 
* `fillval` Value to fill 
* `zb` Array to fill 




        

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

_Reallocate memory for a single array._ 
```C++
template<class T>
void ReallocArray (
    int nblk,
    int blksize,
    T *& zb
) 
```



Reallocates memory for the given array to match the new block and size.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `zb` Array to reallocate 




        

<hr>



### function ReallocArray 

_Reallocate memory for multiple arrays (zs, h, u, v)._ 
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



Reallocates memory for the given arrays to match the new block and size.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `zs` Array to reallocate 
* `h` Array to reallocate 
* `u` Array to reallocate 
* `v` Array to reallocate 




        

<hr>



### function ReallocArray 

_Reallocate memory for extended arrays (zs, h, u, v, U, hU)._ 
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



Reallocates memory for the given arrays to match the new block and size.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `zs` Array to reallocate 
* `h` Array to reallocate 
* `u` Array to reallocate 
* `v` Array to reallocate 
* `U` Array to reallocate 
* `hU` Array to reallocate 




        

<hr>



### function ReallocArray 

_Reallocate memory for evolving variables structure._ 
```C++
template<class T>
void ReallocArray (
    int nblk,
    int blksize,
    EvolvingP < T > & Ev
) 
```



Reallocates memory for all arrays in [**EvolvingP**](struct_evolving_p.md) structure.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `Ev` [**EvolvingP**](struct_evolving_p.md) structure to reallocate 




        

<hr>



### function ReallocArray 

_Reallocate memory for extended evolving variables structure._ 
```C++
template<class T>
void ReallocArray (
    int nblk,
    int blksize,
    EvolvingP_M < T > & Ev
) 
```



Reallocates memory for all arrays in [**EvolvingP\_M**](struct_evolving_p___m.md) structure.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `Ev` [**EvolvingP\_M**](struct_evolving_p___m.md) structure to reallocate 




        

<hr>



### function ReallocArray 

_Reallocate all model arrays._ 
```C++
template<class T>
void ReallocArray (
    int nblk,
    int blksize,
    Param XParam,
    Model < T > & XModel
) 
```



Reallocates memory for all arrays in the [**Model**](struct_model.md) structure, including blocks, gradients, fluxes, and output buffers.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `nblk` Number of blocks 
* `blksize` Block size 
* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) structure to reallocate 




        

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

_Compute memory index for a cell in a block (using_ [_**Param**_](class_param.md) _)._
```C++
__host__ int memloc (
    Param XParam,
    int i,
    int j,
    int ib
) 
```



Calculates the linear memory index for a cell in a block using model parameters.




**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `i` Cell x-index 
* `j` Cell y-index 
* `ib` Block index 



**Returns:**

Linear memory index 





        

<hr>



### function memloc 

_Compute memory index for a cell in a block (using explicit sizes)._ 
```C++
__host__ __device__ int memloc (
    int halowidth,
    int blkmemwidth,
    int i,
    int j,
    int ib
) 
```



Calculates the linear memory index for a cell in a block using explicit halo and block sizes.




**Parameters:**


* `halowidth` Halo width 
* `blkmemwidth` Block memory width 
* `i` Cell x-index 
* `j` Cell y-index 
* `ib` Block index 



**Returns:**

Linear memory index 





        

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


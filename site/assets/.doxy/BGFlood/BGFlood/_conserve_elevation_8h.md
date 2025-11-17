

# File ConserveElevation.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**ConserveElevation.h**](_conserve_elevation_8h.md)

[Go to the source code of this file](_conserve_elevation_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Write_txtlog.h"`
* `#include "Util_CPU.h"`
* `#include "Arrays.h"`
* `#include "MemManagement.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ \_\_device\_\_ void | [**ProlongationElevation**](#function-prolongationelevation) (int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int ip, int jp, T \* h, T \* zs, T \* zb) <br>_Prolongs elevation values from parent to child block halo cells, handling dry/wet logic._  |
|  void | [**WetDryProlongation**](#function-wetdryprolongation) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br>_Performs wet/dry prolongation for all block boundaries._  |
|  void | [**WetDryProlongationGPU**](#function-wetdryprolongationgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br>_Performs wet/dry prolongation for all block boundaries on the GPU using CUDA kernels._  |
|  \_\_global\_\_ void | [**WetDryProlongationGPUBot**](#function-wetdryprolongationgpubot) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryProlongationGPULeft**](#function-wetdryprolongationgpuleft) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryProlongationGPURight**](#function-wetdryprolongationgpuright) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryProlongationGPUTop**](#function-wetdryprolongationgputop) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**WetDryRestriction**](#function-wetdryrestriction) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br>_Performs wet/dry restriction for all block boundaries._  |
|  void | [**WetDryRestrictionGPU**](#function-wetdryrestrictiongpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br>_Performs wet/dry restriction for all block boundaries on the GPU using CUDA kernels._  |
|  \_\_global\_\_ void | [**WetDryRestrictionGPUBot**](#function-wetdryrestrictiongpubot) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryRestrictionGPULeft**](#function-wetdryrestrictiongpuleft) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryRestrictionGPURight**](#function-wetdryrestrictiongpuright) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryRestrictionGPUTop**](#function-wetdryrestrictiongputop) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevation**](#function-conserveelevation) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br>_Conserves elevation across all active blocks by applying elevation conservation on each block's boundaries._  |
|  void | [**conserveElevationBot**](#function-conserveelevationbot) ([**Param**](class_param.md) XParam, int ib, int ibBL, int ibBR, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**conserveElevationBot**](#function-conserveelevationbot) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevationGHBot**](#function-conserveelevationghbot) ([**Param**](class_param.md) XParam, int ib, int ibBL, int ibBR, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  \_\_global\_\_ void | [**conserveElevationGHBot**](#function-conserveelevationghbot) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  void | [**conserveElevationGHLeft**](#function-conserveelevationghleft) ([**Param**](class_param.md) XParam, int ib, int ibLB, int ibLT, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  \_\_global\_\_ void | [**conserveElevationGHLeft**](#function-conserveelevationghleft) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  void | [**conserveElevationGHRight**](#function-conserveelevationghright) ([**Param**](class_param.md) XParam, int ib, int ibRB, int ibRT, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  \_\_global\_\_ void | [**conserveElevationGHRight**](#function-conserveelevationghright) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  void | [**conserveElevationGHTop**](#function-conserveelevationghtop) ([**Param**](class_param.md) XParam, int ib, int ibTL, int ibTR, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  \_\_global\_\_ void | [**conserveElevationGHTop**](#function-conserveelevationghtop) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  void | [**conserveElevationGPU**](#function-conserveelevationgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br>_Conserves elevation on the GPU for all active blocks using CUDA kernels._  |
|  void | [**conserveElevationGradHalo**](#function-conserveelevationgradhalo) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx, T \* dhdy, T \* dzsdy) <br>_Conserves elevation gradients in halo cells for all active blocks._  |
|  void | [**conserveElevationGradHaloGPU**](#function-conserveelevationgradhalogpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx, T \* dhdy, T \* dzsdy) <br> |
|  void | [**conserveElevationLeft**](#function-conserveelevationleft) ([**Param**](class_param.md) XParam, int ib, int ibLB, int ibLT, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**conserveElevationLeft**](#function-conserveelevationleft) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevationRight**](#function-conserveelevationright) ([**Param**](class_param.md) XParam, int ib, int ibRB, int ibRT, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**conserveElevationRight**](#function-conserveelevationright) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevationTop**](#function-conserveelevationtop) ([**Param**](class_param.md) XParam, int ib, int ibTL, int ibTR, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**conserveElevationTop**](#function-conserveelevationtop) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_host\_\_ \_\_device\_\_ void | [**wetdryrestriction**](#function-wetdryrestriction) (int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, T \* h, T \* zs, T \* zb) <br> |




























## Public Functions Documentation




### function ProlongationElevation 

_Prolongs elevation values from parent to child block halo cells, handling dry/wet logic._ 
```C++
template<class T>
__host__ __device__ void ProlongationElevation (
    int halowidth,
    int blkmemwidth,
    T eps,
    int ib,
    int ibn,
    int ihalo,
    int jhalo,
    int ip,
    int jp,
    T * h,
    T * zs,
    T * zb
) 
```



Copies elevation and water surface values from parent to child halo if any neighbor is dry.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `halowidth` Halo width 
* `blkmemwidth` Block memory width 
* `eps` Dry threshold 
* `ib` Block index 
* `ibn` Neighbor block index 
* `ihalo` Halo i-index 
* `jhalo` Halo j-index 
* `ip` Parent i-index 
* `jp` Parent j-index 
* `h` Water depth array 
* `zs` Water surface array 
* `zb` Bed elevation array 




        

<hr>



### function WetDryProlongation 

_Performs wet/dry prolongation for all block boundaries._ 
```C++
template<class T>
void WetDryProlongation (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```



Applies prolongation logic to left, right, top, and bottom boundaries where block level is greater than neighbor.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables structure 
* `zb` Bed elevation array 




        

<hr>



### function WetDryProlongationGPU 

_Performs wet/dry prolongation for all block boundaries on the GPU using CUDA kernels._ 
```C++
template<class T>
void WetDryProlongationGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```



Launches CUDA kernels for left, right, top, and bottom boundaries, synchronizing after each.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables structure 
* `zb` Bed elevation array 




        

<hr>



### function WetDryProlongationGPUBot 

```C++
template<class T>
__global__ void WetDryProlongationGPUBot (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function WetDryProlongationGPULeft 

```C++
template<class T>
__global__ void WetDryProlongationGPULeft (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function WetDryProlongationGPURight 

```C++
template<class T>
__global__ void WetDryProlongationGPURight (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function WetDryProlongationGPUTop 

```C++
template<class T>
__global__ void WetDryProlongationGPUTop (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function WetDryRestriction 

_Performs wet/dry restriction for all block boundaries._ 
```C++
template<class T>
void WetDryRestriction (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```



Applies restriction logic to left, right, top, and bottom boundaries where block level is less than neighbor.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables structure 
* `zb` Bed elevation array 




        

<hr>



### function WetDryRestrictionGPU 

_Performs wet/dry restriction for all block boundaries on the GPU using CUDA kernels._ 
```C++
template<class T>
void WetDryRestrictionGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```



Launches CUDA kernels for left, right, top, and bottom boundaries, synchronizing after each.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables structure 
* `zb` Bed elevation array 




        

<hr>



### function WetDryRestrictionGPUBot 

```C++
template<class T>
__global__ void WetDryRestrictionGPUBot (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function WetDryRestrictionGPULeft 

```C++
template<class T>
__global__ void WetDryRestrictionGPULeft (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function WetDryRestrictionGPURight 

```C++
template<class T>
__global__ void WetDryRestrictionGPURight (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function WetDryRestrictionGPUTop 

```C++
template<class T>
__global__ void WetDryRestrictionGPUTop (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function conserveElevation 

_Conserves elevation across all active blocks by applying elevation conservation on each block's boundaries._ 
```C++
template<class T>
void conserveElevation (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```



Iterates over all active blocks and applies conservation routines for left, right, top, and bottom boundaries.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables structure 
* `zb` Bed elevation array 




        

<hr>



### function conserveElevationBot 

```C++
template<class T>
void conserveElevationBot (
    Param XParam,
    int ib,
    int ibBL,
    int ibBR,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function conserveElevationBot 

```C++
template<class T>
__global__ void conserveElevationBot (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function conserveElevationGHBot 

```C++
template<class T>
void conserveElevationGHBot (
    Param XParam,
    int ib,
    int ibBL,
    int ibBR,
    BlockP < T > XBlock,
    T * h,
    T * zs,
    T * zb,
    T * dhdx,
    T * dzsdx
) 
```




<hr>



### function conserveElevationGHBot 

```C++
template<class T>
__global__ void conserveElevationGHBot (
    Param XParam,
    BlockP < T > XBlock,
    T * h,
    T * zs,
    T * zb,
    T * dhdx,
    T * dzsdx
) 
```




<hr>



### function conserveElevationGHLeft 

```C++
template<class T>
void conserveElevationGHLeft (
    Param XParam,
    int ib,
    int ibLB,
    int ibLT,
    BlockP < T > XBlock,
    T * h,
    T * zs,
    T * zb,
    T * dhdx,
    T * dzsdx
) 
```




<hr>



### function conserveElevationGHLeft 

```C++
template<class T>
__global__ void conserveElevationGHLeft (
    Param XParam,
    BlockP < T > XBlock,
    T * h,
    T * zs,
    T * zb,
    T * dhdx,
    T * dzsdx
) 
```




<hr>



### function conserveElevationGHRight 

```C++
template<class T>
void conserveElevationGHRight (
    Param XParam,
    int ib,
    int ibRB,
    int ibRT,
    BlockP < T > XBlock,
    T * h,
    T * zs,
    T * zb,
    T * dhdx,
    T * dzsdx
) 
```




<hr>



### function conserveElevationGHRight 

```C++
template<class T>
__global__ void conserveElevationGHRight (
    Param XParam,
    BlockP < T > XBlock,
    T * h,
    T * zs,
    T * zb,
    T * dhdx,
    T * dzsdx
) 
```




<hr>



### function conserveElevationGHTop 

```C++
template<class T>
void conserveElevationGHTop (
    Param XParam,
    int ib,
    int ibTL,
    int ibTR,
    BlockP < T > XBlock,
    T * h,
    T * zs,
    T * zb,
    T * dhdx,
    T * dzsdx
) 
```




<hr>



### function conserveElevationGHTop 

```C++
template<class T>
__global__ void conserveElevationGHTop (
    Param XParam,
    BlockP < T > XBlock,
    T * h,
    T * zs,
    T * zb,
    T * dhdx,
    T * dzsdx
) 
```




<hr>



### function conserveElevationGPU 

_Conserves elevation on the GPU for all active blocks using CUDA kernels._ 
```C++
template<class T>
void conserveElevationGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```



Launches CUDA kernels for left, right, top, and bottom boundaries, synchronizing after each.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `XEv` Evolving variables structure 
* `zb` Bed elevation array 




        

<hr>



### function conserveElevationGradHalo 

_Conserves elevation gradients in halo cells for all active blocks._ 
```C++
template<class T>
void conserveElevationGradHalo (
    Param XParam,
    BlockP < T > XBlock,
    T * h,
    T * zs,
    T * zb,
    T * dhdx,
    T * dzsdx,
    T * dhdy,
    T * dzsdy
) 
```



Applies gradient conservation routines for left, right, top, and bottom boundaries.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `h` Water depth array 
* `zs` Water surface array 
* `zb` Bed elevation array 
* `dhdx` Water depth gradient x 
* `dzsdx` Water surface gradient x 
* `dhdy` Water depth gradient y 
* `dzsdy` Water surface gradient y 




        

<hr>



### function conserveElevationGradHaloGPU 

```C++
template<class T>
void conserveElevationGradHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * h,
    T * zs,
    T * zb,
    T * dhdx,
    T * dzsdx,
    T * dhdy,
    T * dzsdy
) 
```




<hr>



### function conserveElevationLeft 

```C++
template<class T>
void conserveElevationLeft (
    Param XParam,
    int ib,
    int ibLB,
    int ibLT,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function conserveElevationLeft 

```C++
template<class T>
__global__ void conserveElevationLeft (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function conserveElevationRight 

```C++
template<class T>
void conserveElevationRight (
    Param XParam,
    int ib,
    int ibRB,
    int ibRT,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function conserveElevationRight 

```C++
template<class T>
__global__ void conserveElevationRight (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function conserveElevationTop 

```C++
template<class T>
void conserveElevationTop (
    Param XParam,
    int ib,
    int ibTL,
    int ibTR,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function conserveElevationTop 

```C++
template<class T>
__global__ void conserveElevationTop (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function wetdryrestriction 

```C++
template<class T>
__host__ __device__ void wetdryrestriction (
    int halowidth,
    int blkmemwidth,
    T eps,
    int ib,
    int ibn,
    int ihalo,
    int jhalo,
    int i,
    int j,
    T * h,
    T * zs,
    T * zb
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/ConserveElevation.h`


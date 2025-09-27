

# File ConserveElevation.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**ConserveElevation.h**](ConserveElevation_8h.md)

[Go to the source code of this file](ConserveElevation_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Write_txtlog.h"`
* `#include "Util_CPU.h"`
* `#include "Arrays.h"`
* `#include "MemManagement.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ \_\_device\_\_ void | [**ProlongationElevation**](#function-prolongationelevation) (int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int ip, int jp, T \* h, T \* zs, T \* zb) <br> |
|  void | [**WetDryProlongation**](#function-wetdryprolongation) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**WetDryProlongationGPU**](#function-wetdryprolongationgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryProlongationGPUBot**](#function-wetdryprolongationgpubot) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryProlongationGPULeft**](#function-wetdryprolongationgpuleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryProlongationGPURight**](#function-wetdryprolongationgpuright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryProlongationGPUTop**](#function-wetdryprolongationgputop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**WetDryRestriction**](#function-wetdryrestriction) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**WetDryRestrictionGPU**](#function-wetdryrestrictiongpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryRestrictionGPUBot**](#function-wetdryrestrictiongpubot) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryRestrictionGPULeft**](#function-wetdryrestrictiongpuleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryRestrictionGPURight**](#function-wetdryrestrictiongpuright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryRestrictionGPUTop**](#function-wetdryrestrictiongputop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevation**](#function-conserveelevation) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevationBot**](#function-conserveelevationbot) ([**Param**](classParam.md) XParam, int ib, int ibBL, int ibBR, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**conserveElevationBot**](#function-conserveelevationbot) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevationGHBot**](#function-conserveelevationghbot) ([**Param**](classParam.md) XParam, int ib, int ibBL, int ibBR, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  \_\_global\_\_ void | [**conserveElevationGHBot**](#function-conserveelevationghbot) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  void | [**conserveElevationGHLeft**](#function-conserveelevationghleft) ([**Param**](classParam.md) XParam, int ib, int ibLB, int ibLT, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  \_\_global\_\_ void | [**conserveElevationGHLeft**](#function-conserveelevationghleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  void | [**conserveElevationGHRight**](#function-conserveelevationghright) ([**Param**](classParam.md) XParam, int ib, int ibRB, int ibRT, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  \_\_global\_\_ void | [**conserveElevationGHRight**](#function-conserveelevationghright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  void | [**conserveElevationGHTop**](#function-conserveelevationghtop) ([**Param**](classParam.md) XParam, int ib, int ibTL, int ibTR, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  \_\_global\_\_ void | [**conserveElevationGHTop**](#function-conserveelevationghtop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  void | [**conserveElevationGPU**](#function-conserveelevationgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevationGradHalo**](#function-conserveelevationgradhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx, T \* dhdy, T \* dzsdy) <br> |
|  void | [**conserveElevationGradHaloGPU**](#function-conserveelevationgradhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx, T \* dhdy, T \* dzsdy) <br> |
|  void | [**conserveElevationLeft**](#function-conserveelevationleft) ([**Param**](classParam.md) XParam, int ib, int ibLB, int ibLT, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**conserveElevationLeft**](#function-conserveelevationleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevationRight**](#function-conserveelevationright) ([**Param**](classParam.md) XParam, int ib, int ibRB, int ibRT, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**conserveElevationRight**](#function-conserveelevationright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevationTop**](#function-conserveelevationtop) ([**Param**](classParam.md) XParam, int ib, int ibTL, int ibTR, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**conserveElevationTop**](#function-conserveelevationtop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_host\_\_ \_\_device\_\_ void | [**wetdryrestriction**](#function-wetdryrestriction) (int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, T \* h, T \* zs, T \* zb) <br> |




























## Public Functions Documentation




### function ProlongationElevation 

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




<hr>



### function WetDryProlongation 

```C++
template<class T>
void WetDryProlongation (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function WetDryProlongationGPU 

```C++
template<class T>
void WetDryProlongationGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




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

```C++
template<class T>
void WetDryRestriction (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function WetDryRestrictionGPU 

```C++
template<class T>
void WetDryRestrictionGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




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

```C++
template<class T>
void conserveElevation (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




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

```C++
template<class T>
void conserveElevationGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function conserveElevationGradHalo 

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


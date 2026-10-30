

# File ConserveElevation.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**ConserveElevation.cu**](ConserveElevation_8cu.md)

[Go to the source code of this file](ConserveElevation_8cu_source.md)



* `#include "ConserveElevation.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ \_\_device\_\_ void | [**ProlongationElevation**](#function-prolongationelevation) (int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int ip, int jp, T \* h, T \* zs, T \* zb) <br>_Prolongs elevation values from parent to child block halo cells, handling dry/wet logic._  |
|  \_\_host\_\_ \_\_device\_\_ void | [**ProlongationElevationGH**](#function-prolongationelevationgh) (int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int ip, int jp, T \* h, T \* dhdx, T \* dzsdx) <br>_Prolongs gradient and water surface values from parent to child block halo cells, handling dry/wet logic._  |
|  \_\_host\_\_ \_\_device\_\_ void | [**RevertProlongationElevation**](#function-revertprolongationelevation) (int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int ip, int jp, int level, T dx, T \* h, T \* zb, T \* dzbdx, T \* dzbdy) <br>_Reverts prolongation for elevation, adjusting bed elevation using gradients._  |
|  void | [**WetDryProlongation**](#function-wetdryprolongation) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br>_Performs wet/dry prolongation for all block boundaries._  |
|  template void | [**WetDryProlongation&lt; double &gt;**](#function-wetdryprolongation-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, double \* zb) <br> |
|  template void | [**WetDryProlongation&lt; float &gt;**](#function-wetdryprolongation-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, float \* zb) <br> |
|  void | [**WetDryProlongationGPU**](#function-wetdryprolongationgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br>_Performs wet/dry prolongation for all block boundaries on the GPU using CUDA kernels._  |
|  template void | [**WetDryProlongationGPU&lt; double &gt;**](#function-wetdryprolongationgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, double \* zb) <br> |
|  template void | [**WetDryProlongationGPU&lt; float &gt;**](#function-wetdryprolongationgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, float \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryProlongationGPUBot**](#function-wetdryprolongationgpubot) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryProlongationGPULeft**](#function-wetdryprolongationgpuleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryProlongationGPURight**](#function-wetdryprolongationgpuright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryProlongationGPUTop**](#function-wetdryprolongationgputop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**WetDryRestriction**](#function-wetdryrestriction) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br>_Performs wet/dry restriction for all block boundaries._  |
|  template void | [**WetDryRestriction&lt; double &gt;**](#function-wetdryrestriction-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, double \* zb) <br> |
|  template void | [**WetDryRestriction&lt; float &gt;**](#function-wetdryrestriction-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, float \* zb) <br> |
|  void | [**WetDryRestrictionGPU**](#function-wetdryrestrictiongpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br>_Performs wet/dry restriction for all block boundaries on the GPU using CUDA kernels._  |
|  template void | [**WetDryRestrictionGPU&lt; double &gt;**](#function-wetdryrestrictiongpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, double \* zb) <br> |
|  template void | [**WetDryRestrictionGPU&lt; float &gt;**](#function-wetdryrestrictiongpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, float \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryRestrictionGPUBot**](#function-wetdryrestrictiongpubot) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryRestrictionGPULeft**](#function-wetdryrestrictiongpuleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryRestrictionGPURight**](#function-wetdryrestrictiongpuright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**WetDryRestrictionGPUTop**](#function-wetdryrestrictiongputop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevation**](#function-conserveelevation) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br>_Conserves elevation across all active blocks by applying elevation conservation on each block's boundaries._  |
|  \_\_host\_\_ \_\_device\_\_ void | [**conserveElevation**](#function-conserveelevation) (int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, T \* h, T \* zs, T \* zb) <br>_Conserves elevation for a single halo cell using neighbor values._  |
|  \_\_host\_\_ \_\_device\_\_ void | [**conserveElevation**](#function-conserveelevation) (T zb, T & zswet, T & hwet) <br> |
|  template void | [**conserveElevation&lt; double &gt;**](#function-conserveelevation-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, double \* zb) <br> |
|  template \_\_host\_\_ \_\_device\_\_ void | [**conserveElevation&lt; double &gt;**](#function-conserveelevation-double) (int halowidth, int blkmemwidth, double eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, double \* h, double \* zs, double \* zb) <br> |
|  template void | [**conserveElevation&lt; float &gt;**](#function-conserveelevation-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, float \* zb) <br> |
|  template \_\_host\_\_ \_\_device\_\_ void | [**conserveElevation&lt; float &gt;**](#function-conserveelevation-float) (int halowidth, int blkmemwidth, float eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, float \* h, float \* zs, float \* zb) <br> |
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
|  void | [**conserveElevationGPU**](#function-conserveelevationgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br>_Conserves elevation on the GPU for all active blocks using CUDA kernels._  |
|  template void | [**conserveElevationGPU&lt; double &gt;**](#function-conserveelevationgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, double \* zb) <br> |
|  template void | [**conserveElevationGPU&lt; float &gt;**](#function-conserveelevationgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, float \* zb) <br> |
|  void | [**conserveElevationGradHalo**](#function-conserveelevationgradhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx, T \* dhdy, T \* dzsdy) <br>_Conserves elevation gradients in halo cells for all active blocks._  |
|  \_\_host\_\_ \_\_device\_\_ void | [**conserveElevationGradHalo**](#function-conserveelevationgradhalo) (int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, T \* h, T \* dhdx, T \* dhdy) <br> |
|  template void | [**conserveElevationGradHalo&lt; double &gt;**](#function-conserveelevationgradhalo-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* h, double \* zs, double \* zb, double \* dhdx, double \* dzsdx, double \* dhdy, double \* dzsdy) <br> |
|  template void | [**conserveElevationGradHalo&lt; float &gt;**](#function-conserveelevationgradhalo-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* h, float \* zs, float \* zb, float \* dhdx, float \* dzsdx, float \* dhdy, float \* dzsdy) <br> |
|  \_\_host\_\_ \_\_device\_\_ void | [**conserveElevationGradHaloA**](#function-conserveelevationgradhaloa) (int halowidth, int blkmemwidth, int ib, int ibn, int ihalo, int jhalo, int ip, int jp, int iq, int jq, T theta, T delta, T \* h, T \* dhdx) <br> |
|  \_\_host\_\_ \_\_device\_\_ void | [**conserveElevationGradHaloB**](#function-conserveelevationgradhalob) (int halowidth, int blkmemwidth, int ib, int ibn, int ihalo, int jhalo, int ip, int jp, int iq, int jq, T theta, T delta, T eps, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx) <br> |
|  void | [**conserveElevationGradHaloGPU**](#function-conserveelevationgradhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* zs, T \* zb, T \* dhdx, T \* dzsdx, T \* dhdy, T \* dzsdy) <br> |
|  template void | [**conserveElevationGradHaloGPU&lt; double &gt;**](#function-conserveelevationgradhalogpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* h, double \* zs, double \* zb, double \* dhdx, double \* dzsdx, double \* dhdy, double \* dzsdy) <br> |
|  template void | [**conserveElevationGradHaloGPU&lt; float &gt;**](#function-conserveelevationgradhalogpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* h, float \* zs, float \* zb, float \* dhdx, float \* dzsdx, float \* dhdy, float \* dzsdy) <br> |
|  void | [**conserveElevationLeft**](#function-conserveelevationleft) ([**Param**](classParam.md) XParam, int ib, int ibLB, int ibLT, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**conserveElevationLeft**](#function-conserveelevationleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevationRight**](#function-conserveelevationright) ([**Param**](classParam.md) XParam, int ib, int ibRB, int ibRT, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**conserveElevationRight**](#function-conserveelevationright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  void | [**conserveElevationTop**](#function-conserveelevationtop) ([**Param**](classParam.md) XParam, int ib, int ibTL, int ibTR, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_global\_\_ void | [**conserveElevationTop**](#function-conserveelevationtop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* zb) <br> |
|  \_\_host\_\_ \_\_device\_\_ void | [**wetdryrestriction**](#function-wetdryrestriction) (int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, T \* h, T \* zs, T \* zb) <br> |
|  template \_\_host\_\_ \_\_device\_\_ void | [**wetdryrestriction&lt; double &gt;**](#function-wetdryrestriction-double) (int halowidth, int blkmemwidth, double eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, double \* h, double \* zs, double \* zb) <br> |
|  template \_\_host\_\_ \_\_device\_\_ void | [**wetdryrestriction&lt; float &gt;**](#function-wetdryrestriction-float) (int halowidth, int blkmemwidth, float eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, float \* h, float \* zs, float \* zb) <br> |




























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



### function ProlongationElevationGH 

_Prolongs gradient and water surface values from parent to child block halo cells, handling dry/wet logic._ 
```C++
template<class T>
__host__ __device__ void ProlongationElevationGH (
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
    T * dhdx,
    T * dzsdx
) 
```



Sets gradients to zero in child halo if any neighbor is dry.




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
* `dhdx` Water depth gradient x 
* `dzsdx` Water surface gradient x 




        

<hr>



### function RevertProlongationElevation 

_Reverts prolongation for elevation, adjusting bed elevation using gradients._ 
```C++
template<class T>
__host__ __device__ void RevertProlongationElevation (
    int halowidth,
    int blkmemwidth,
    T eps,
    int ib,
    int ibn,
    int ihalo,
    int jhalo,
    int ip,
    int jp,
    int level,
    T dx,
    T * h,
    T * zb,
    T * dzbdx,
    T * dzbdy
) 
```



Applies gradient-based correction to bed elevation in halo cells if any neighbor is dry.




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
* `level` Block refinement level 
* `dx` Cell size 
* `h` Water depth array 
* `zb` Bed elevation array 
* `dzbdx` Bed elevation gradient x 
* `dzbdy` Bed elevation gradient y 




        

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



### function WetDryProlongation&lt; double &gt; 

```C++
template void WetDryProlongation< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    double * zb
) 
```




<hr>



### function WetDryProlongation&lt; float &gt; 

```C++
template void WetDryProlongation< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    float * zb
) 
```




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



### function WetDryProlongationGPU&lt; double &gt; 

```C++
template void WetDryProlongationGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    double * zb
) 
```




<hr>



### function WetDryProlongationGPU&lt; float &gt; 

```C++
template void WetDryProlongationGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    float * zb
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



### function WetDryRestriction&lt; double &gt; 

```C++
template void WetDryRestriction< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    double * zb
) 
```




<hr>



### function WetDryRestriction&lt; float &gt; 

```C++
template void WetDryRestriction< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    float * zb
) 
```




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



### function WetDryRestrictionGPU&lt; double &gt; 

```C++
template void WetDryRestrictionGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    double * zb
) 
```




<hr>



### function WetDryRestrictionGPU&lt; float &gt; 

```C++
template void WetDryRestrictionGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    float * zb
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



### function conserveElevation 

_Conserves elevation for a single halo cell using neighbor values._ 
```C++
template<class T>
__host__ __device__ void conserveElevation (
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



Computes wet/dry averages and updates water depth and surface for the halo cell.




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
* `i` Parent i-index 
* `j` Parent j-index 
* `h` Water depth array 
* `zs` Water surface array 
* `zb` Bed elevation array 




        

<hr>



### function conserveElevation 

```C++
template<class T>
__host__ __device__ void conserveElevation (
    T zb,
    T & zswet,
    T & hwet
) 
```




<hr>



### function conserveElevation&lt; double &gt; 

```C++
template void conserveElevation< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    double * zb
) 
```




<hr>



### function conserveElevation&lt; double &gt; 

```C++
template __host__ __device__ void conserveElevation< double > (
    int halowidth,
    int blkmemwidth,
    double eps,
    int ib,
    int ibn,
    int ihalo,
    int jhalo,
    int i,
    int j,
    double * h,
    double * zs,
    double * zb
) 
```




<hr>



### function conserveElevation&lt; float &gt; 

```C++
template void conserveElevation< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    float * zb
) 
```




<hr>



### function conserveElevation&lt; float &gt; 

```C++
template __host__ __device__ void conserveElevation< float > (
    int halowidth,
    int blkmemwidth,
    float eps,
    int ib,
    int ibn,
    int ihalo,
    int jhalo,
    int i,
    int j,
    float * h,
    float * zs,
    float * zb
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



### function conserveElevationGPU&lt; double &gt; 

```C++
template void conserveElevationGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    double * zb
) 
```




<hr>



### function conserveElevationGPU&lt; float &gt; 

```C++
template void conserveElevationGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    float * zb
) 
```




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



### function conserveElevationGradHalo 

```C++
template<class T>
__host__ __device__ void conserveElevationGradHalo (
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
    T * dhdx,
    T * dhdy
) 
```




<hr>



### function conserveElevationGradHalo&lt; double &gt; 

```C++
template void conserveElevationGradHalo< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * h,
    double * zs,
    double * zb,
    double * dhdx,
    double * dzsdx,
    double * dhdy,
    double * dzsdy
) 
```




<hr>



### function conserveElevationGradHalo&lt; float &gt; 

```C++
template void conserveElevationGradHalo< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * h,
    float * zs,
    float * zb,
    float * dhdx,
    float * dzsdx,
    float * dhdy,
    float * dzsdy
) 
```




<hr>



### function conserveElevationGradHaloA 

```C++
template<class T>
__host__ __device__ void conserveElevationGradHaloA (
    int halowidth,
    int blkmemwidth,
    int ib,
    int ibn,
    int ihalo,
    int jhalo,
    int ip,
    int jp,
    int iq,
    int jq,
    T theta,
    T delta,
    T * h,
    T * dhdx
) 
```




<hr>



### function conserveElevationGradHaloB 

```C++
template<class T>
__host__ __device__ void conserveElevationGradHaloB (
    int halowidth,
    int blkmemwidth,
    int ib,
    int ibn,
    int ihalo,
    int jhalo,
    int ip,
    int jp,
    int iq,
    int jq,
    T theta,
    T delta,
    T eps,
    T * h,
    T * zs,
    T * zb,
    T * dhdx,
    T * dzsdx
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



### function conserveElevationGradHaloGPU&lt; double &gt; 

```C++
template void conserveElevationGradHaloGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * h,
    double * zs,
    double * zb,
    double * dhdx,
    double * dzsdx,
    double * dhdy,
    double * dzsdy
) 
```




<hr>



### function conserveElevationGradHaloGPU&lt; float &gt; 

```C++
template void conserveElevationGradHaloGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * h,
    float * zs,
    float * zb,
    float * dhdx,
    float * dzsdx,
    float * dhdy,
    float * dzsdy
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



### function wetdryrestriction&lt; double &gt; 

```C++
template __host__ __device__ void wetdryrestriction< double > (
    int halowidth,
    int blkmemwidth,
    double eps,
    int ib,
    int ibn,
    int ihalo,
    int jhalo,
    int i,
    int j,
    double * h,
    double * zs,
    double * zb
) 
```




<hr>



### function wetdryrestriction&lt; float &gt; 

```C++
template __host__ __device__ void wetdryrestriction< float > (
    int halowidth,
    int blkmemwidth,
    float eps,
    int ib,
    int ibn,
    int ihalo,
    int jhalo,
    int i,
    int j,
    float * h,
    float * zs,
    float * zb
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/ConserveElevation.cu`


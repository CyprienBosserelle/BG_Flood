

# File Halo.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Halo.h**](Halo_8h.md)

[Go to the source code of this file](Halo_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Write_txtlog.h"`
* `#include "Util_CPU.h"`
* `#include "Arrays.h"`
* `#include "Mesh.h"`
* `#include "MemManagement.h"`
* `#include "Boundary.h"`
* `#include "ConserveElevation.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_global\_\_ void | [**HaloFluxGPUBT**](#function-halofluxgpubt) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  \_\_global\_\_ void | [**HaloFluxGPUBTnew**](#function-halofluxgpubtnew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  \_\_global\_\_ void | [**HaloFluxGPULR**](#function-halofluxgpulr) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  \_\_global\_\_ void | [**HaloFluxGPULRnew**](#function-halofluxgpulrnew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  void | [**RecalculateZs**](#function-recalculatezs) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br>_Recalculate water surface after recalculating the values on the halo on the CPU._  |
|  \_\_global\_\_ void | [**RecalculateZsGPU**](#function-recalculatezsgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  void | [**Recalculatehh**](#function-recalculatehh) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  void | [**bndmaskGPU**](#function-bndmaskgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  void | [**fillBot**](#function-fillbot) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillBot**](#function-fillbot) (int halowidth, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, T \* a) <br> |
|  \_\_global\_\_ void | [**fillBotnew**](#function-fillbotnew) (int halowidth, int nblk, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, T \* a) <br> |
|  void | [**fillCorners**](#function-fillcorners) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  void | [**fillCorners**](#function-fillcorners) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  void | [**fillCorners**](#function-fillcorners) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & Xev) <br> |
|  \_\_global\_\_ void | [**fillCornersGPU**](#function-fillcornersgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev) <br> |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**GradientsP**](structGradientsP.md)&lt; T &gt; Grad) <br> |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  void | [**fillHaloC**](#function-fillhaloc) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating halos for each block of a single variable on CPU._  |
|  void | [**fillHaloF**](#function-fillhalof) ([**Param**](classParam.md) XParam, bool doProlongation, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating flux in the halos for a block and a single variable on CPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br>_Wrapping function for calculating halos for each block of a single variable on GPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev) <br> |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**GradientsP**](structGradientsP.md)&lt; T &gt; Grad) <br> |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  void | [**fillHaloGPUnew**](#function-fillhalogpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br> |
|  void | [**fillHaloTopRightC**](#function-fillhalotoprightc) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating flux for halos for each block of a single variable on GPU._  |
|  void | [**fillHaloTopRightGPU**](#function-fillhalotoprightgpu) ([**Param**](classParam.md) XParam, bool doprolong, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br> |
|  void | [**fillLeft**](#function-fillleft) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillLeft**](#function-fillleft) (int halowidth, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, T \* a) <br> |
|  \_\_global\_\_ void | [**fillLeftnew**](#function-fillleftnew) (int halowidth, int nblk, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, T \* a) <br> |
|  void | [**fillRight**](#function-fillright) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillRight**](#function-fillright) (int halowidth, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, T \* a) <br> |
|  void | [**fillRightFlux**](#function-fillrightflux) ([**Param**](classParam.md) XParam, bool doProlongation, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillRightFlux**](#function-fillrightflux) (int halowidth, bool doProlongation, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, T \* a) <br> |
|  \_\_global\_\_ void | [**fillRightnew**](#function-fillrightnew) (int halowidth, int nblk, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, T \* a) <br> |
|  void | [**fillTop**](#function-filltop) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillTop**](#function-filltop) (int halowidth, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, T \* a) <br> |
|  void | [**fillTopFlux**](#function-filltopflux) ([**Param**](classParam.md) XParam, bool doProlongation, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillTopFlux**](#function-filltopflux) (int halowidth, bool doProlongation, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, T \* a) <br> |
|  \_\_global\_\_ void | [**fillTopnew**](#function-filltopnew) (int halowidth, int nblk, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, T \* a) <br> |
|  void | [**refine\_linear**](#function-refine_linear) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  void | [**refine\_linearGPU**](#function-refine_lineargpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  void | [**refine\_linear\_Bot**](#function-refine_linear_bot) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdy) <br> |
|  void | [**refine\_linear\_Left**](#function-refine_linear_left) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  void | [**refine\_linear\_Right**](#function-refine_linear_right) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx) <br> |
|  void | [**refine\_linear\_Top**](#function-refine_linear_top) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdy) <br> |




























## Public Functions Documentation




### function HaloFluxGPUBT 

```C++
template<class T>
__global__ void HaloFluxGPUBT (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```




<hr>



### function HaloFluxGPUBTnew 

```C++
template<class T>
__global__ void HaloFluxGPUBTnew (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```




<hr>



### function HaloFluxGPULR 

```C++
template<class T>
__global__ void HaloFluxGPULR (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```




<hr>



### function HaloFluxGPULRnew 

```C++
template<class T>
__global__ void HaloFluxGPULRnew (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```




<hr>



### function RecalculateZs 

_Recalculate water surface after recalculating the values on the halo on the CPU._ 
```C++
template<class T>
void RecalculateZs (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```



Recalculate water surface after recalculating the values on the halo on the GPU.


### Description



Recalculate water surface after recalculating the values on the halo on the CPU. zb (bottom elevation) on each halo is calculated at the start of the loop or as part of the initial condition. When conserve-elevation is not required, only h is recalculated on the halo at ever 1/2 steps. 
 zs then needs to be recalculated to obtain a mass-conservative solution (if zs is conserved then mass conservation is not garanteed)



### Warning



This function calculate zs everywhere in the block... this is a bit unecessary. Instead it should recalculate only where there is a prolongation or a restiction



### Description



Recalculate water surface after recalculating the values on the halo on the CPU. zb (bottom elevation) on each halo is calculated at the start of the loop or as part of the initial condition. When conserve-elevation is not required, only h is recalculated on the halo at ever 1/2 steps. zs then needs to be recalculated to obtain a mass-conservative solution (if zs is conserved then mass conservation is not garanteed)



### Warning



This function calculate zs everywhere in the block... this is a bit unecessary. Instead it should recalculate only where there is a prolongation or a restiction 



        

<hr>



### function RecalculateZsGPU 

```C++
template<class T>
__global__ void RecalculateZsGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```




<hr>



### function Recalculatehh 

```C++
template<class T>
void Recalculatehh (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```




<hr>



### function bndmaskGPU 

```C++
template<class T>
void bndmaskGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    FluxP < T > Flux
) 
```




<hr>



### function fillBot 

```C++
template<class T>
void fillBot (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```




<hr>



### function fillBot 

```C++
template<class T>
__global__ void fillBot (
    int halowidth,
    int * active,
    int * level,
    int * botleft,
    int * botright,
    int * topleft,
    int * lefttop,
    int * righttop,
    T * a
) 
```




<hr>



### function fillBotnew 

```C++
template<class T>
__global__ void fillBotnew (
    int halowidth,
    int nblk,
    int * active,
    int * level,
    int * botleft,
    int * botright,
    int * topleft,
    int * lefttop,
    int * righttop,
    T * a
) 
```




<hr>



### function fillCorners 

```C++
template<class T>
void fillCorners (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```




<hr>



### function fillCorners 

```C++
template<class T>
void fillCorners (
    Param XParam,
    BlockP < T > XBlock,
    T *& z
) 
```




<hr>



### function fillCorners 

```C++
template<class T>
void fillCorners (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > & Xev
) 
```




<hr>



### function fillCornersGPU 

```C++
template<class T>
__global__ void fillCornersGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```




<hr>



### function fillHalo 

```C++
template<class T>
void fillHalo (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```




<hr>



### function fillHalo 

```C++
template<class T>
void fillHalo (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev
) 
```




<hr>



### function fillHalo 

```C++
template<class T>
void fillHalo (
    Param XParam,
    BlockP < T > XBlock,
    GradientsP < T > Grad
) 
```




<hr>



### function fillHalo 

```C++
template<class T>
void fillHalo (
    Param XParam,
    BlockP < T > XBlock,
    FluxP < T > Flux
) 
```




<hr>



### function fillHaloC 

_Wrapping function for calculating halos for each block of a single variable on CPU._ 
```C++
template<class T>
void fillHaloC (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```



### Description



This function is a wraping fuction of the halo functions on CPU. It is called from the main Halo CPU function. This is layer 2 of 3 wrap so the candy doesn't stick too much. 



        

<hr>



### function fillHaloF 

_Wrapping function for calculating flux in the halos for a block and a single variable on CPU._ 
```C++
template<class T>
void fillHaloF (
    Param XParam,
    bool doProlongation,
    BlockP < T > XBlock,
    T * z
) 
```



### Depreciated



This function is was never sucessful and will never be used. It is fundamentally flawed because is doesn't preserve the balance of fluxes on the restiction interface It should be deleted soon 



### Description




        

<hr>



### function fillHaloGPU 

_Wrapping function for calculating halos for each block of a single variable on GPU._ 
```C++
template<class T>
void fillHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```



### Description



This function is a wraping fuction of the halo functions on GPU. It is called from the main Halo GPU function. The present imnplementation is naive and slow one that calls the rather complex fillLeft type functions 



        

<hr>



### function fillHaloGPU 

```C++
template<class T>
void fillHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```




<hr>



### function fillHaloGPU 

```C++
template<class T>
void fillHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev
) 
```




<hr>



### function fillHaloGPU 

```C++
template<class T>
void fillHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```




<hr>



### function fillHaloGPU 

```C++
template<class T>
void fillHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    GradientsP < T > Grad
) 
```




<hr>



### function fillHaloGPU 

```C++
template<class T>
void fillHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    FluxP < T > Flux
) 
```




<hr>



### function fillHaloGPUnew 

```C++
template<class T>
void fillHaloGPUnew (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```




<hr>



### function fillHaloTopRightC 

_Wrapping function for calculating flux for halos for each block of a single variable on GPU._ 
```C++
template<class T>
void fillHaloTopRightC (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```



### Description



This function is a wraping function of the halo flux functions on GPU. It is called from the main Halo GPU function. The present imnplementation is naive and slow one that calls the rather complex fillLeft type functions 



        

<hr>



### function fillHaloTopRightGPU 

```C++
template<class T>
void fillHaloTopRightGPU (
    Param XParam,
    bool doprolong,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```




<hr>



### function fillLeft 

```C++
template<class T>
void fillLeft (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```




<hr>



### function fillLeft 

```C++
template<class T>
__global__ void fillLeft (
    int halowidth,
    int * active,
    int * level,
    int * leftbot,
    int * lefttop,
    int * rightbot,
    int * botright,
    int * topright,
    T * a
) 
```




<hr>



### function fillLeftnew 

```C++
template<class T>
__global__ void fillLeftnew (
    int halowidth,
    int nblk,
    int * active,
    int * level,
    int * leftbot,
    int * lefttop,
    int * rightbot,
    int * botright,
    int * topright,
    T * a
) 
```




<hr>



### function fillRight 

```C++
template<class T>
void fillRight (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```




<hr>



### function fillRight 

```C++
template<class T>
__global__ void fillRight (
    int halowidth,
    int * active,
    int * level,
    int * rightbot,
    int * righttop,
    int * leftbot,
    int * botleft,
    int * topleft,
    T * a
) 
```




<hr>



### function fillRightFlux 

```C++
template<class T>
void fillRightFlux (
    Param XParam,
    bool doProlongation,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```




<hr>



### function fillRightFlux 

```C++
template<class T>
__global__ void fillRightFlux (
    int halowidth,
    bool doProlongation,
    int * active,
    int * level,
    int * rightbot,
    int * righttop,
    int * leftbot,
    int * botleft,
    int * topleft,
    T * a
) 
```




<hr>



### function fillRightnew 

```C++
template<class T>
__global__ void fillRightnew (
    int halowidth,
    int nblk,
    int * active,
    int * level,
    int * rightbot,
    int * righttop,
    int * leftbot,
    int * botleft,
    int * topleft,
    T * a
) 
```




<hr>



### function fillTop 

```C++
template<class T>
void fillTop (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```




<hr>



### function fillTop 

```C++
template<class T>
__global__ void fillTop (
    int halowidth,
    int * active,
    int * level,
    int * topleft,
    int * topright,
    int * botleft,
    int * leftbot,
    int * rightbot,
    T * a
) 
```




<hr>



### function fillTopFlux 

```C++
template<class T>
void fillTopFlux (
    Param XParam,
    bool doProlongation,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```




<hr>



### function fillTopFlux 

```C++
template<class T>
__global__ void fillTopFlux (
    int halowidth,
    bool doProlongation,
    int * active,
    int * level,
    int * topleft,
    int * topright,
    int * botleft,
    int * leftbot,
    int * rightbot,
    T * a
) 
```




<hr>



### function fillTopnew 

```C++
template<class T>
__global__ void fillTopnew (
    int halowidth,
    int nblk,
    int * active,
    int * level,
    int * topleft,
    int * topright,
    int * botleft,
    int * leftbot,
    int * rightbot,
    T * a
) 
```




<hr>



### function refine\_linear 

```C++
template<class T>
void refine_linear (
    Param XParam,
    BlockP < T > XBlock,
    T * z,
    T * dzdx,
    T * dzdy
) 
```




<hr>



### function refine\_linearGPU 

```C++
template<class T>
void refine_linearGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * z,
    T * dzdx,
    T * dzdy
) 
```




<hr>



### function refine\_linear\_Bot 

```C++
template<class T>
void refine_linear_Bot (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T * z,
    T * dzdy
) 
```




<hr>



### function refine\_linear\_Left 

```C++
template<class T>
void refine_linear_Left (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T * z,
    T * dzdx,
    T * dzdy
) 
```




<hr>



### function refine\_linear\_Right 

```C++
template<class T>
void refine_linear_Right (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T * z,
    T * dzdx
) 
```




<hr>



### function refine\_linear\_Top 

```C++
template<class T>
void refine_linear_Top (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T * z,
    T * dzdy
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Halo.h`


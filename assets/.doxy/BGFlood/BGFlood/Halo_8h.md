

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
|  \_\_global\_\_ void | [**HaloFluxGPUBT**](#function-halofluxgpubt) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_GPU kernel for applying halo flux correction on the top and bottom boundaries of all active blocks._  |
|  \_\_global\_\_ void | [**HaloFluxGPUBTnew**](#function-halofluxgpubtnew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_GPU kernel for applying halo flux correction on the top and bottom boundaries of all active blocks._  |
|  \_\_global\_\_ void | [**HaloFluxGPULR**](#function-halofluxgpulr) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for applying halo flux correction on the left and right boundaries of all active blocks._  |
|  \_\_global\_\_ void | [**HaloFluxGPULRnew**](#function-halofluxgpulrnew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_GPU kernel for applying halo flux correction on the left and right boundaries of all active blocks._  |
|  void | [**RecalculateZs**](#function-recalculatezs) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br>_Recalculate water surface after recalculating the values on the halo on the CPU._  |
|  \_\_global\_\_ void | [**RecalculateZsGPU**](#function-recalculatezsgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  void | [**Recalculatehh**](#function-recalculatehh) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br>_Recalculate water depth after recalculating the values on the halo on the CPU._  |
|  void | [**bndmaskGPU**](#function-bndmaskgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br>_Wrapping function for applying boundary masks to flux variables on GPU._  |
|  void | [**fillBot**](#function-fillbot) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br>_Function to fill the bottom halo region of a block, handling various neighbor configurations._  |
|  \_\_global\_\_ void | [**fillBot**](#function-fillbot) (int halowidth, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, T \* a) <br>_CUDA kernel to fill the bottom halo region of blocks in parallel, handling various neighbor configurations._  |
|  \_\_global\_\_ void | [**fillBotnew**](#function-fillbotnew) (int halowidth, int nblk, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, T \* a) <br>_CUDA kernel to fill the bottom halo region of blocks in parallel for new refinement, handling various neighbor configurations, new version._  |
|  void | [**fillCorners**](#function-fillcorners) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br>_Function to fill the corner halo regions for a specific block, handling various neighbor configurations._  |
|  void | [**fillCorners**](#function-fillcorners) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br>_Function to fill the corner halo regions for all active blocks._  |
|  void | [**fillCorners**](#function-fillcorners) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & Xev) <br>_Function to fill the corner halo regions for all active blocks and all evolving variables._  |
|  \_\_global\_\_ void | [**fillCornersGPU**](#function-fillcornersgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_CUDA kernel to fill the corner halo regions for all active blocks in parallel, handling various neighbor configurations._  |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br>_Wrapping function for calculating halos for each block and each variable on CPU._  |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev) <br>_Wrapping function for calculating halos for each block and each variable on CPU._  |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**GradientsP**](structGradientsP.md)&lt; T &gt; Grad) <br>_Wrapping function for calculating halos for each block and each variable on CPU._  |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br>_Wrapping function for calculating flux halos for each block and each variable on CPU._  |
|  void | [**fillHaloC**](#function-fillhaloc) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating halos for each block of a single variable on CPU._  |
|  void | [**fillHaloF**](#function-fillhalof) ([**Param**](classParam.md) XParam, bool doProlongation, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating flux in the halos for a block and a single variable on CPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br>_Wrapping function for calculating halos for each block of a single variable on GPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating halos for each block of a single variable on GPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev) <br>_Wrapping function for calculating halos for each block and each variable on GPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br>_Wrapping function for calculating halos for each block and each variable on GPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**GradientsP**](structGradientsP.md)&lt; T &gt; Grad) <br>_Wrapping function for calculating halos for each block and each variable on GPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br>_Wrapping function for calculating flux halos for each block and each variable on GPU._  |
|  void | [**fillHaloGPUnew**](#function-fillhalogpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br>_Wrapping function for calculating halos for each block of a single variable on GPU. New version._  |
|  void | [**fillHaloTopRightC**](#function-fillhalotoprightc) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating flux for halos for each block of a single variable on GPU._  |
|  void | [**fillHaloTopRightGPU**](#function-fillhalotoprightgpu) ([**Param**](classParam.md) XParam, bool doprolong, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br> |
|  void | [**fillLeft**](#function-fillleft) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br>_Applying halo flux correction on the left boundaries of all active blocks on GPU._  |
|  \_\_global\_\_ void | [**fillLeft**](#function-fillleft) (int halowidth, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, T \* a) <br>_GPU kernel for applying halo flux correction on the left boundaries of all active blocks._  |
|  \_\_global\_\_ void | [**fillLeftnew**](#function-fillleftnew) (int halowidth, int nblk, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, T \* a) <br>_New way of filling the left halo 2 blocks at a time to maximize GPU occupancy._  |
|  void | [**fillRight**](#function-fillright) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br>_Fills the right halo region of a block._  |
|  \_\_global\_\_ void | [**fillRight**](#function-fillright) (int halowidth, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, T \* a) <br>_CUDA kernel to fill the right halo region of blocks in parallel._  |
|  void | [**fillRightFlux**](#function-fillrightflux) ([**Param**](classParam.md) XParam, bool doProlongation, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br>_Function to fill the right halo region of a block, handling various neighbor configurations._  |
|  \_\_global\_\_ void | [**fillRightFlux**](#function-fillrightflux) (int halowidth, bool doProlongation, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, T \* a) <br>_CUDA kernel to fill the right halo region of blocks in parallel for flux variables, handling various neighbor configurations._  |
|  \_\_global\_\_ void | [**fillRightnew**](#function-fillrightnew) (int halowidth, int nblk, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, T \* a) <br>_CUDA kernel to fill the right halo region of blocks in parallel (new version)._  |
|  void | [**fillTop**](#function-filltop) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br>_Fills the top halo region of a block, handling various neighbor configurations._  |
|  \_\_global\_\_ void | [**fillTop**](#function-filltop) (int halowidth, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, T \* a) <br>_CUDA kernel to fill the top halo region of blocks in parallel for new refinement, handling various neighbor configurations, new version._  |
|  void | [**fillTopFlux**](#function-filltopflux) ([**Param**](classParam.md) XParam, bool doProlongation, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br>_Function to fill the top halo region of a block for new refinement, handling various neighbor configurations._  |
|  \_\_global\_\_ void | [**fillTopFlux**](#function-filltopflux) (int halowidth, bool doProlongation, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, T \* a) <br>_CUDA kernel to fill the top halo region of blocks in parallel for new refinement, handling various neighbor configurations, new version._  |
|  \_\_global\_\_ void | [**fillTopnew**](#function-filltopnew) (int halowidth, int nblk, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, T \* a) <br> |
|  void | [**refine\_linear**](#function-refine_linear) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_Wrapping function for refining all sides of active blocks using linear reconstruction._  |
|  void | [**refine\_linearGPU**](#function-refine_lineargpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_Wrapping function for refining all sides of active blocks using linear reconstruction on GPU._  |
|  void | [**refine\_linear\_Bot**](#function-refine_linear_bot) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdy) <br> |
|  void | [**refine\_linear\_Left**](#function-refine_linear_left) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_Refine a block on the left side using linear reconstruction._  |
|  void | [**refine\_linear\_Right**](#function-refine_linear_right) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx) <br> |
|  void | [**refine\_linear\_Top**](#function-refine_linear_top) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdy) <br> |




























## Public Functions Documentation




### function HaloFluxGPUBT 

_GPU kernel for applying halo flux correction on the top and bottom boundaries of all active blocks._ 
```C++
template<class T>
__global__ void HaloFluxGPUBT (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 



**Template parameters:**


* `T` The data type (float or double) 




        

<hr>



### function HaloFluxGPUBTnew 

_GPU kernel for applying halo flux correction on the top and bottom boundaries of all active blocks._ 
```C++
template<class T>
__global__ void HaloFluxGPUBTnew (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 



**Template parameters:**


* `T` The data type (float or double) 




        

<hr>



### function HaloFluxGPULR 

_Wrapping function for applying halo flux correction on the left and right boundaries of all active blocks._ 
```C++
template<class T>
__global__ void HaloFluxGPULR (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 



**Template parameters:**


* `T` The data type (float or double) 




        

<hr>



### function HaloFluxGPULRnew 

_GPU kernel for applying halo flux correction on the left and right boundaries of all active blocks._ 
```C++
template<class T>
__global__ void HaloFluxGPULRnew (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 



**Template parameters:**


* `T` The data type (float or double) 




        

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


! 


### Description



Recalculate water surface after recalculating the values on the halo on the CPU. zb (bottom elevation) on each halo is calculated at the start of the loop or as part of the initial condition. When conserve-elevation is not required, only h is recalculated on the halo at ever 1/2 steps. 
 zs then needs to be recalculated to obtain a mass-conservative solution (if zs is conserved then mass conservation is not garanteed)




**Warning:**

This function calculate zs everywhere in the block... this is a bit unecessary. Instead it should recalculate only where there is a prolongation or a restiction




**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `Xev` The evolving structure containing the evolving variables 
* `zb` The bottom elevation variable 



**Template parameters:**


* `T` The data type (float or double)

! 



### Description



Recalculate water surface after recalculating the values on the halo on the CPU. zb (bottom elevation) on each halo is calculated at the start of the loop or as part of the initial condition. When conserve-elevation is not required, only h is recalculated on the halo at ever 1/2 steps. zs then needs to be recalculated to obtain a mass-conservative solution (if zs is conserved then mass conservation is not garanteed)




**Warning:**

This function calculate zs everywhere in the block... this is a bit unecessary. Instead it should recalculate only where there is a prolongation or a restiction 




**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `Xev` The evolving structure containing the evolving variables 
* `zb` The bottom elevation variable 



**Template parameters:**


* `T` The data type (float or double) 





        

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

_Recalculate water depth after recalculating the values on the halo on the CPU._ 
```C++
template<class T>
void Recalculatehh (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```



### Description



Recalculate water depth after recalculating the values on the halo on the CPU. zb (bottom elevation) on each halo is calculated at the start of the loop or as part of the initial condition. When conserve-elevation is not required, only h is recalculated on the halo at ever 1/2 steps. zs then needs to be recalculated to obtain a mass-conservative solution (if zs is conserved then mass conservation is not garanteed) 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `Xev` The evolving structure containing the evolving variables 
* `zb` The bottom elevation variable 



**Template parameters:**


* `T` The data type (float or double) 





        

<hr>



### function bndmaskGPU 

_Wrapping function for applying boundary masks to flux variables on GPU._ 
```C++
template<class T>
void bndmaskGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    FluxP < T > Flux
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `Xev` The evolving structure containing the evolving variables 
* `Flux` The flux structure containing the flux variables 



**Template parameters:**


* `T` The data type (float or double) 




        

<hr>



### function fillBot 

_Function to fill the bottom halo region of a block, handling various neighbor configurations._ 
```C++
template<class T>
void fillBot (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```





**Parameters:**


* `XParam` The parameters of the grid and blocks 
* `ib` The index of the current block 
* `XBlock` The block structure containing neighbor information 
* `z` The variable to be refined 
* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillBot 

_CUDA kernel to fill the bottom halo region of blocks in parallel, handling various neighbor configurations._ 
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





**Parameters:**


* `halowidth` The width of the halo region 
* `active` The array of active block indices 
* `level` The array of block levels 
* `botleft` The array of bottom left neighbor block indices 
* `botright` The array of bottom right neighbor block indices 
* `topleft` The array of top left neighbor block indices 
* `lefttop` The array of left top neighbor block indices 
* `righttop` The array of right top neighbor block indices 
* `a` The variable to be refined 
* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillBotnew 

_CUDA kernel to fill the bottom halo region of blocks in parallel for new refinement, handling various neighbor configurations, new version._ 
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





**Parameters:**


* `halowidth` The width of the halo region 
* `nblk` The number of active blocks 
* `active` The array of active block indices 
* `level` The array of block levels 
* `botleft` The array of bottom left neighbor block indices 
* `botright` The array of bottom right neighbor block indices 
* `topleft` The array of top left neighbor block indices 
* `lefttop` The array of left top neighbor block indices 
* `righttop` The array of right top neighbor block indices 
* `a` The variable to be refined 
* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillCorners 

_Function to fill the corner halo regions for a specific block, handling various neighbor configurations._ 
```C++
template<class T>
void fillCorners (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```





**Parameters:**


* `XParam` The parameters of the grid and blocks 
* `ib` The index of the block to be processed 
* `XBlock` The structure containing block neighbor information 
* `z` The variable to be processed 
* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillCorners 

_Function to fill the corner halo regions for all active blocks._ 
```C++
template<class T>
void fillCorners (
    Param XParam,
    BlockP < T > XBlock,
    T *& z
) 
```





**Parameters:**


* `XParam` The parameters of the grid and blocks 
* `XBlock` The structure containing block neighbor information 
* `z` The variable to be processed 



**Template parameters:**


* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillCorners 

_Function to fill the corner halo regions for all active blocks and all evolving variables._ 
```C++
template<class T>
void fillCorners (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > & Xev
) 
```





**Parameters:**


* `XParam` The parameters of the grid and blocks 
* `XBlock` The structure containing block neighbor information 
* `Xev` The structure containing evolving variables 



**Template parameters:**


* `T` The data type of the variables (e.g., float, double) 




        

<hr>



### function fillCornersGPU 

_CUDA kernel to fill the corner halo regions for all active blocks in parallel, handling various neighbor configurations._ 
```C++
template<class T>
__global__ void fillCornersGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```





**Parameters:**


* `XParam` The parameters of the grid and blocks 
* `XBlock` The structure containing block neighbor information 
* `z` The variable to be processed 
* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillHalo 

_Wrapping function for calculating halos for each block and each variable on CPU._ 
```C++
template<class T>
void fillHalo (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```



### Description



This function is a wraping fuction of the halo functions on CPU. It is called from the main Halo function. It uses multithreading to calculate the halos of the 4 variables in parallel. 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `Xev` The evolving structure containing the evolving variables 
* `zb` The bottom elevation variable 



**Template parameters:**


* `T` The data type (float or double) 





        

<hr>



### function fillHalo 

_Wrapping function for calculating halos for each block and each variable on CPU._ 
```C++
template<class T>
void fillHalo (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev
) 
```



### Description



This function is a wraping fuction of the halo functions on CPU. It is called from the main Halo function. It uses multithreading to calculate the halos of the 4 variables in parallel. 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `Xev` The evolving structure containing the evolving variables 



**Template parameters:**


* `T` The data type (float or double) 





        

<hr>



### function fillHalo 

_Wrapping function for calculating halos for each block and each variable on CPU._ 
```C++
template<class T>
void fillHalo (
    Param XParam,
    BlockP < T > XBlock,
    GradientsP < T > Grad
) 
```



### Description



This function is a wrapping function of the halo functions on CPU. It is called from the main Halo function. It uses multithreading to calculate the halos of the 4 variables in parallel. 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `Grad` The gradients structure containing the gradients 



**Template parameters:**


* `T` The data type (float or double) 





        

<hr>



### function fillHalo 

_Wrapping function for calculating flux halos for each block and each variable on CPU._ 
```C++
template<class T>
void fillHalo (
    Param XParam,
    BlockP < T > XBlock,
    FluxP < T > Flux
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `Flux` The flux structure containing the flux variables 



**Template parameters:**


* `T` The data type (float or double) 




        

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



! 


### Description



This function is a wraping fuction of the halo functions on CPU. It is called from the main Halo CPU function. This is layer 2 of 3 wrap so the candy doesn't stick too much. 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to work on 





        

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



! 

**Deprecated**

This function is was never sucessful and will never be used. It is fundamentally flawed because is doesn't preserve the balance of fluxes on the restiction interface. It should be deleted soon.




        

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



! 


### Description



This function is a wraping fuction of the halo functions on GPU. It is called from the main Halo GPU function. The present imnplementation is naive and slow one that calls the rather complex fillLeft type functions 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `stream` The cuda stream to use 
* `z` The variable to work on 





        

<hr>



### function fillHaloGPU 

_Wrapping function for calculating halos for each block of a single variable on GPU._ 
```C++
template<class T>
void fillHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```





**Deprecated**



**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `stream` The cuda stream to use 
* `z` The variable to work on 




        

<hr>



### function fillHaloGPU 

_Wrapping function for calculating halos for each block and each variable on GPU._ 
```C++
template<class T>
void fillHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `Xev` The evolving structure containing the evolving variables 



**Template parameters:**


* `T` The data type (float or double) 




        

<hr>



### function fillHaloGPU 

_Wrapping function for calculating halos for each block and each variable on GPU._ 
```C++
template<class T>
void fillHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```



### Description



This function is a wraping fuction of the halo functions on GPU. It is called from the main Halo GPU function. It uses multiple cuda streams to calculate the halos of the 4 variables in parallel. After filling the halos, it applies either the elevation conservation or wet-dry fix if enabled in parameters. Finally, it recalculates the surface elevation zs based on the updated water depth h and bottom elevation zb. 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `Xev` The evolving structure containing the evolving variables 
* `zb` The bottom elevation variable 



**Template parameters:**


* `T` The data type (float or double) 





        

<hr>



### function fillHaloGPU 

_Wrapping function for calculating halos for each block and each variable on GPU._ 
```C++
template<class T>
void fillHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    GradientsP < T > Grad
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `Grad` The gradients structure containing the gradients 



**Template parameters:**


* `T` The data type (float or double) 




        

<hr>



### function fillHaloGPU 

_Wrapping function for calculating flux halos for each block and each variable on GPU._ 
```C++
template<class T>
void fillHaloGPU (
    Param XParam,
    BlockP < T > XBlock,
    FluxP < T > Flux
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `Flux` The flux structure containing the flux variables 



**Template parameters:**


* `T` The data type (float or double) 




        

<hr>



### function fillHaloGPUnew 

_Wrapping function for calculating halos for each block of a single variable on GPU. New version._ 
```C++
template<class T>
void fillHaloGPUnew (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```



! 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `stream` The cuda stream to use 
* `z` The variable to work on 




        

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

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to work on 



**Note:**

For flux term and actually most terms, only top and right neighbours are needed! 






        

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

_Applying halo flux correction on the left boundaries of all active blocks on GPU._ 
```C++
template<class T>
void fillLeft (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```





**Parameters:**


* `XParam` The model parameters 
* `ib` The block index 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 



**Template parameters:**


* `T` The data type (float or double) 




        

<hr>



### function fillLeft 

_GPU kernel for applying halo flux correction on the left boundaries of all active blocks._ 
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





**Parameters:**


* `halowidth` The width of the halo region 
* `active` The array of active block indices 
* `level` The array of block levels 
* `leftbot` The array of left bottom neighbor block indices 
* `lefttop` The array of left top neighbor block indices 
* `rightbot` The array of right bottom neighbor block indices 
* `botright` The array of bottom right neighbor block indices 
* `topright` The array of top right neighbor block indices 
* `a` The variable to be refined 
* `T` The data type (float or double) 




        

<hr>



### function fillLeftnew 

_New way of filling the left halo 2 blocks at a time to maximize GPU occupancy._ 
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



### Description



This fuction is a wraping fuction of the halo functions for CPU. It is called from another wraping function to keep things clean. In a sense this is the third (and last) layer of wrapping




**Parameters:**


* `halowidth` The width of the halo region 
* `nblk` The number of active blocks 
* `active` The array of active block indices 
* `level` The array of block levels 
* `leftbot` The array of left bottom neighbor block indices 
* `lefttop` The array of left top neighbor block indices 
* `rightbot` The array of right bottom neighbor block indices 
* `botright` The array of bottom right neighbor block indices 
* `topright` The array of top right neighbor block indices 
* `a` The variable to be refined 





        

<hr>



### function fillRight 

_Fills the right halo region of a block._ 
```C++
template<class T>
void fillRight (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```





**Parameters:**


* `XParam` The simulation parameters 
* `ib` The index of the block to process 
* `XBlock` The block structure containing neighbor information 
* `z` The variable to be refined 




        

<hr>



### function fillRight 

_CUDA kernel to fill the right halo region of blocks in parallel._ 
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





**Parameters:**


* `halowidth` The width of the halo region 
* `active` The array of active block indices 
* `level` The array of block levels 
* `rightbot` The array of right bottom neighbor block indices 
* `righttop` The array of right top neighbor block indices 
* `leftbot` The array of left bottom neighbor block indices 
* `botleft` The array of bottom left neighbor block indices 
* `topleft` The array of top left neighbor block indices 
* `a` The variable to be refined 



**Template parameters:**


* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillRightFlux 

_Function to fill the right halo region of a block, handling various neighbor configurations._ 
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





**Parameters:**


* `XParam` The parameters of the grid and blocks 
* `doProlongation` Flag indicating whether to perform prolongation 
* `ib` The index of the current block 
* `XBlock` The block structure containing neighbor information 
* `z` The variable to be refined 



**Template parameters:**


* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillRightFlux 

_CUDA kernel to fill the right halo region of blocks in parallel for flux variables, handling various neighbor configurations._ 
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





**Parameters:**


* `halowidth` The width of the halo region 
* `doProlongation` Flag indicating whether to perform prolongation 
* `active` The array of active block indices 
* `level` The array of block levels 
* `rightbot` The array of right bottom neighbor block indices 
* `righttop` The array of right top neighbor block indices 
* `leftbot` The array of left bottom neighbor block indices 
* `botleft` The array of bottom left neighbor block indices 
* `topleft` The array of top left neighbor block indices 
* `a` The variable to be refined 
* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillRightnew 

_CUDA kernel to fill the right halo region of blocks in parallel (new version)._ 
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





**Parameters:**


* `halowidth` The width of the halo region 
* `nblk` The number of active blocks 
* `active` The array of active block indices 
* `level` The array of block levels 
* `rightbot` The array of right bottom neighbor block indices 
* `righttop` The array of right top neighbor block indices 
* `leftbot` The array of left bottom neighbor block indices 
* `botleft` The array of bottom left neighbor block indices 
* `topleft` The array of top left neighbor block indices 
* `a` The variable to be refined 



**Template parameters:**


* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillTop 

_Fills the top halo region of a block, handling various neighbor configurations._ 
```C++
template<class T>
void fillTop (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```





**Parameters:**


* `XParam` The parameters of the grid/block structure 
* `ib` The index of the current block 
* `XBlock` The block structure containing neighbor information 
* `z` The variable to be refined 
* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillTop 

_CUDA kernel to fill the top halo region of blocks in parallel for new refinement, handling various neighbor configurations, new version._ 
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





**Parameters:**


* `halowidth` The width of the halo region 
* `active` The array of active block indices 
* `level` The array of block levels 
* `topleft` The array of top left neighbor block indices 
* `topright` The array of top right neighbor block indices 
* `botleft` The array of bottom left neighbor block indices 
* `leftbot` The array of left bottom neighbor block indices 
* `rightbot` The array of right bottom neighbor block indices 
* `a` The variable to be refined 
* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillTopFlux 

_Function to fill the top halo region of a block for new refinement, handling various neighbor configurations._ 
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





**Parameters:**


* `XParam` The parameters of the grid and blocks 
* `doProlongation` Flag indicating whether to perform prolongation 
* `ib` The index of the block to be processed 
* `XBlock` The structure containing block neighbor information 
* `z` The variable to be refined 



**Template parameters:**


* `T` The data type of the variable (e.g., float, double) 




        

<hr>



### function fillTopFlux 

_CUDA kernel to fill the top halo region of blocks in parallel for new refinement, handling various neighbor configurations, new version._ 
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





**Parameters:**


* `halowidth` The width of the halo region 
* `doProlongation` Flag indicating whether to perform prolongation 
* `active` The array of active block indices 
* `level` The array of block levels 
* `topleft` The array of top left neighbor block indices 
* `topright` The array of top right neighbor block indices 
* `botleft` The array of bottom left neighbor block indices 
* `leftbot` The array of left bottom neighbor block indices 
* `rightbot` The array of right bottom neighbor block indices 
* `a` The variable to be refined 
* `T` The data type of the variable (e.g., float, double) 




        

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

_Wrapping function for refining all sides of active blocks using linear reconstruction._ 
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





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 
* `dzdx` The gradient of z in the x direction 
* `dzdy` The gradient of z in the y direction 



**Template parameters:**


* `T` The data type (float or double) 




        

<hr>



### function refine\_linearGPU 

_Wrapping function for refining all sides of active blocks using linear reconstruction on GPU._ 
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





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 
* `dzdx` The gradient of z in the x direction 
* `dzdy` The gradient of z in the y direction 



**Template parameters:**


* `T` The data type (float or double) 




        

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

_Refine a block on the left side using linear reconstruction._ 
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



### Description



This function refines a block on the left side using linear reconstruction. It checks if the neighboring block on the left is at a coarser level. If so, it calculates the new values for the left boundary of the current block using the gradients in the x and y directions. The new values are computed based on the distance to the neighboring block and the gradients, ensuring a smooth transition between different resolution levels. 

**Parameters:**


* `XParam` The model parameters 
* `ib` The index of the current block 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 
* `dzdx` The gradient of z in the x direction 
* `dzdy` The gradient of z in the y direction 



**Template parameters:**


* `T` The data type (float or double) 





        

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


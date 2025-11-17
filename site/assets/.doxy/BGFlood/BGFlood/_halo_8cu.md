

# File Halo.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Halo.cu**](_halo_8cu.md)

[Go to the source code of this file](_halo_8cu_source.md)



* `#include "Halo.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**HaloFluxCPUBT**](#function-halofluxcpubt) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for applying halo flux correction on the left and right boundaries of all active blocks on GPU._  |
|  void | [**HaloFluxCPULR**](#function-halofluxcpulr) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_CPU function for applying halo flux correction on the left and right boundaries._  |
|  \_\_global\_\_ void | [**HaloFluxGPUBT**](#function-halofluxgpubt) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_GPU kernel for applying halo flux correction on the top and bottom boundaries of all active blocks._  |
|  \_\_global\_\_ void | [**HaloFluxGPUBTnew**](#function-halofluxgpubtnew) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_GPU kernel for applying halo flux correction on the top and bottom boundaries of all active blocks._  |
|  \_\_global\_\_ void | [**HaloFluxGPULR**](#function-halofluxgpulr) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for applying halo flux correction on the left and right boundaries of all active blocks._  |
|  \_\_global\_\_ void | [**HaloFluxGPULRnew**](#function-halofluxgpulrnew) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_GPU kernel for applying halo flux correction on the left and right boundaries of all active blocks._  |
|  void | [**RecalculateZs**](#function-recalculatezs) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev, T \* zb) <br>_Recalculate water surface after recalculating the values on the halo on the CPU._  |
|  template void | [**RecalculateZs&lt; double &gt;**](#function-recalculatezs-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template void | [**RecalculateZs&lt; float &gt;**](#function-recalculatezs-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; Xev, float \* zb) <br> |
|  \_\_global\_\_ void | [**RecalculateZsGPU**](#function-recalculatezsgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev, T \* zb) <br> |
|  template \_\_global\_\_ void | [**RecalculateZsGPU&lt; double &gt;**](#function-recalculatezsgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template \_\_global\_\_ void | [**RecalculateZsGPU&lt; float &gt;**](#function-recalculatezsgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; Xev, float \* zb) <br> |
|  void | [**Recalculatehh**](#function-recalculatehh) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev, T \* zb) <br>_Recalculate water depth after recalculating the values on the halo on the CPU._  |
|  template void | [**Recalculatehh&lt; double &gt;**](#function-recalculatehh-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template void | [**Recalculatehh&lt; float &gt;**](#function-recalculatehh-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; Xev, float \* zb) <br> |
|  void | [**bndmaskGPU**](#function-bndmaskgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev, [**FluxP**](struct_flux_p.md)&lt; T &gt; Flux) <br>_Wrapping function for applying boundary masks to flux variables on GPU._  |
|  template void | [**bndmaskGPU&lt; double &gt;**](#function-bndmaskgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; Xev, [**FluxP**](struct_flux_p.md)&lt; double &gt; Flux) <br> |
|  template void | [**bndmaskGPU&lt; float &gt;**](#function-bndmaskgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; Xev, [**FluxP**](struct_flux_p.md)&lt; float &gt; Flux) <br> |
|  void | [**fillBot**](#function-fillbot) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& z) <br>_Function to fill the bottom halo region of a block, handling various neighbor configurations._  |
|  \_\_global\_\_ void | [**fillBot**](#function-fillbot) (int halowidth, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, T \* a) <br>_CUDA kernel to fill the bottom halo region of blocks in parallel, handling various neighbor configurations._  |
|  template \_\_global\_\_ void | [**fillBot&lt; double &gt;**](#function-fillbot-double) (int halowidth, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillBot&lt; float &gt;**](#function-fillbot-float) (int halowidth, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, float \* a) <br> |
|  void | [**fillBotFlux**](#function-fillbotflux) ([**Param**](class_param.md) XParam, bool doProlongation, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& z) <br>_Fills the bottom halo region of a block, handling various neighbor configurations._  |
|  \_\_global\_\_ void | [**fillBotnew**](#function-fillbotnew) (int halowidth, int nblk, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, T \* a) <br>_CUDA kernel to fill the bottom halo region of blocks in parallel for new refinement, handling various neighbor configurations, new version._  |
|  template \_\_global\_\_ void | [**fillBotnew&lt; double &gt;**](#function-fillbotnew-double) (int halowidth, int nblk, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillBotnew&lt; float &gt;**](#function-fillbotnew-float) (int halowidth, int nblk, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, float \* a) <br> |
|  void | [**fillCorners**](#function-fillcorners) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& z) <br>_Function to fill the corner halo regions for all active blocks._  |
|  void | [**fillCorners**](#function-fillcorners) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & Xev) <br>_Function to fill the corner halo regions for all active blocks and all evolving variables._  |
|  void | [**fillCorners**](#function-fillcorners) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& z) <br>_Function to fill the corner halo regions for a specific block, handling various neighbor configurations._  |
|  template void | [**fillCorners&lt; double &gt;**](#function-fillcorners-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \*& z) <br> |
|  template void | [**fillCorners&lt; double &gt;**](#function-fillcorners-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; & Xev) <br> |
|  template void | [**fillCorners&lt; double &gt;**](#function-fillcorners-double) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \*& z) <br> |
|  template void | [**fillCorners&lt; float &gt;**](#function-fillcorners-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \*& z) <br> |
|  template void | [**fillCorners&lt; float &gt;**](#function-fillcorners-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; & Xev) <br> |
|  template void | [**fillCorners&lt; float &gt;**](#function-fillcorners-float) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \*& z) <br> |
|  \_\_global\_\_ void | [**fillCornersGPU**](#function-fillcornersgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_CUDA kernel to fill the corner halo regions for all active blocks in parallel, handling various neighbor configurations._  |
|  template \_\_global\_\_ void | [**fillCornersGPU&lt; double &gt;**](#function-fillcornersgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template \_\_global\_\_ void | [**fillCornersGPU&lt; float &gt;**](#function-fillcornersgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev, T \* zb) <br>_Wrapping function for calculating halos for each block and each variable on CPU._  |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev) <br>_Wrapping function for calculating halos for each block and each variable on CPU._  |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; Grad) <br>_Wrapping function for calculating halos for each block and each variable on CPU._  |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**FluxP**](struct_flux_p.md)&lt; T &gt; Flux) <br>_Wrapping function for calculating flux halos for each block and each variable on CPU._  |
|  template void | [**fillHalo&lt; double &gt;**](#function-fillhalo-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template void | [**fillHalo&lt; double &gt;**](#function-fillhalo-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; Xev) <br> |
|  template void | [**fillHalo&lt; double &gt;**](#function-fillhalo-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; Grad) <br> |
|  template void | [**fillHalo&lt; double &gt;**](#function-fillhalo-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**FluxP**](struct_flux_p.md)&lt; double &gt; Flux) <br> |
|  template void | [**fillHalo&lt; float &gt;**](#function-fillhalo-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; Xev, float \* zb) <br> |
|  template void | [**fillHalo&lt; float &gt;**](#function-fillhalo-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; Xev) <br> |
|  template void | [**fillHalo&lt; float &gt;**](#function-fillhalo-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; Grad) <br> |
|  template void | [**fillHalo&lt; float &gt;**](#function-fillhalo-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**FluxP**](struct_flux_p.md)&lt; float &gt; Flux) <br> |
|  void | [**fillHaloBTFluxC**](#function-fillhalobtfluxc) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating flux for halos for each block of a single variable on GPU._  |
|  template void | [**fillHaloBTFluxC&lt; double &gt;**](#function-fillhalobtfluxc-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloBTFluxC&lt; float &gt;**](#function-fillhalobtfluxc-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHaloBotTopGPU**](#function-fillhalobottopgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br>_Wrapping function for calculating flux for halos for each block of a single variable on GPU._  |
|  template void | [**fillHaloBotTopGPU&lt; double &gt;**](#function-fillhalobottopgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloBotTopGPU&lt; float &gt;**](#function-fillhalobottopgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  void | [**fillHaloBotTopGPUnew**](#function-fillhalobottopgpunew) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br>_Wrapping function for calculating flux for halos for each block of a single variable on GPU. New version._  |
|  template void | [**fillHaloBotTopGPUnew&lt; double &gt;**](#function-fillhalobottopgpunew-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloBotTopGPUnew&lt; float &gt;**](#function-fillhalobottopgpunew-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  void | [**fillHaloC**](#function-fillhaloc) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating halos for each block of a single variable on CPU._  |
|  template void | [**fillHaloC&lt; double &gt;**](#function-fillhaloc-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloC&lt; float &gt;**](#function-fillhaloc-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHaloD**](#function-fillhalod) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating halos on CPU on every side of a block of a single variable._  |
|  template void | [**fillHaloD&lt; double &gt;**](#function-fillhalod-double) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloD&lt; float &gt;**](#function-fillhalod-float) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHaloF**](#function-fillhalof) ([**Param**](class_param.md) XParam, bool doProlongation, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating flux in the halos for a block and a single variable on CPU._  |
|  template void | [**fillHaloF&lt; double &gt;**](#function-fillhalof-double) ([**Param**](class_param.md) XParam, bool doProlongation, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloF&lt; float &gt;**](#function-fillhalof-float) ([**Param**](class_param.md) XParam, bool doProlongation, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br>_Wrapping function for calculating halos for each block of a single variable on GPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating halos for each block of a single variable on GPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev) <br>_Wrapping function for calculating halos for each block and each variable on GPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev, T \* zb) <br>_Wrapping function for calculating halos for each block and each variable on GPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; Grad) <br>_Wrapping function for calculating halos for each block and each variable on GPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**FluxP**](struct_flux_p.md)&lt; T &gt; Flux) <br>_Wrapping function for calculating flux halos for each block and each variable on GPU._  |
|  template void | [**fillHaloGPU&lt; double &gt;**](#function-fillhalogpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloGPU&lt; double &gt;**](#function-fillhalogpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloGPU&lt; double &gt;**](#function-fillhalogpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; Xev) <br> |
|  template void | [**fillHaloGPU&lt; double &gt;**](#function-fillhalogpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template void | [**fillHaloGPU&lt; double &gt;**](#function-fillhalogpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; Grad) <br> |
|  template void | [**fillHaloGPU&lt; double &gt;**](#function-fillhalogpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**FluxP**](struct_flux_p.md)&lt; double &gt; Flux) <br> |
|  template void | [**fillHaloGPU&lt; float &gt;**](#function-fillhalogpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  template void | [**fillHaloGPU&lt; float &gt;**](#function-fillhalogpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z) <br> |
|  template void | [**fillHaloGPU&lt; float &gt;**](#function-fillhalogpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; Xev) <br> |
|  template void | [**fillHaloGPU&lt; float &gt;**](#function-fillhalogpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; Xev, float \* zb) <br> |
|  template void | [**fillHaloGPU&lt; float &gt;**](#function-fillhalogpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; Grad) <br> |
|  template void | [**fillHaloGPU&lt; float &gt;**](#function-fillhalogpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**FluxP**](struct_flux_p.md)&lt; float &gt; Flux) <br> |
|  void | [**fillHaloGPUnew**](#function-fillhalogpunew) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br>_Wrapping function for calculating halos for each block of a single variable on GPU. New version._  |
|  template void | [**fillHaloGPUnew&lt; double &gt;**](#function-fillhalogpunew-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloGPUnew&lt; float &gt;**](#function-fillhalogpunew-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  void | [**fillHaloLRFluxC**](#function-fillhalolrfluxc) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating flux for halos for each block of a single variable on GPU._  |
|  template void | [**fillHaloLRFluxC&lt; double &gt;**](#function-fillhalolrfluxc-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloLRFluxC&lt; float &gt;**](#function-fillhalolrfluxc-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHaloLeftRightGPU**](#function-fillhaloleftrightgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br>_Wrapping function for calculating for halos for each block of a single variable on GPU._  |
|  template void | [**fillHaloLeftRightGPU&lt; double &gt;**](#function-fillhaloleftrightgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloLeftRightGPU&lt; float &gt;**](#function-fillhaloleftrightgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  void | [**fillHaloLeftRightGPUnew**](#function-fillhaloleftrightgpunew) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br>_Wrapping function for calculating for halos for each block of a single variable on GPU. New version._  |
|  template void | [**fillHaloLeftRightGPUnew&lt; double &gt;**](#function-fillhaloleftrightgpunew-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloLeftRightGPUnew&lt; float &gt;**](#function-fillhaloleftrightgpunew-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  void | [**fillHaloTopRightC**](#function-fillhalotoprightc) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating flux for halos for each block of a single variable on GPU._  |
|  template void | [**fillHaloTopRightC&lt; double &gt;**](#function-fillhalotoprightc-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloTopRightC&lt; float &gt;**](#function-fillhalotoprightc-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHaloTopRightGPU**](#function-fillhalotoprightgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br>_Wrapping function for calculating flux for halos for each block of a single variable on GPU._  |
|  template void | [**fillHaloTopRightGPU&lt; double &gt;**](#function-fillhalotoprightgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloTopRightGPU&lt; float &gt;**](#function-fillhalotoprightgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  void | [**fillLeft**](#function-fillleft) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& z) <br>_Applying halo flux correction on the left boundaries of all active blocks on GPU._  |
|  \_\_global\_\_ void | [**fillLeft**](#function-fillleft) (int halowidth, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, T \* a) <br>_GPU kernel for applying halo flux correction on the left boundaries of all active blocks._  |
|  template \_\_global\_\_ void | [**fillLeft&lt; double &gt;**](#function-fillleft-double) (int halowidth, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillLeft&lt; float &gt;**](#function-fillleft-float) (int halowidth, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, float \* a) <br> |
|  void | [**fillLeftFlux**](#function-fillleftflux) ([**Param**](class_param.md) XParam, bool doProlongation, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& z) <br>_CPU function for applying halo flux correction on the left boundaries of a specific block._  |
|  \_\_global\_\_ void | [**fillLeftnew**](#function-fillleftnew) (int halowidth, int nblk, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, T \* a) <br>_New way of filling the left halo 2 blocks at a time to maximize GPU occupancy._  |
|  template \_\_global\_\_ void | [**fillLeftnew&lt; double &gt;**](#function-fillleftnew-double) (int halowidth, int nblk, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillLeftnew&lt; float &gt;**](#function-fillleftnew-float) (int halowidth, int nblk, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, float \* a) <br> |
|  void | [**fillRight**](#function-fillright) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& z) <br>_Fills the right halo region of a block._  |
|  \_\_global\_\_ void | [**fillRight**](#function-fillright) (int halowidth, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, T \* a) <br>_CUDA kernel to fill the right halo region of blocks in parallel._  |
|  template \_\_global\_\_ void | [**fillRight&lt; double &gt;**](#function-fillright-double) (int halowidth, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillRight&lt; float &gt;**](#function-fillright-float) (int halowidth, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, float \* a) <br> |
|  void | [**fillRightFlux**](#function-fillrightflux) ([**Param**](class_param.md) XParam, bool doProlongation, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& z) <br>_Function to fill the right halo region of a block, handling various neighbor configurations._  |
|  \_\_global\_\_ void | [**fillRightFlux**](#function-fillrightflux) (int halowidth, bool doProlongation, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, T \* a) <br>_CUDA kernel to fill the right halo region of blocks in parallel for flux variables, handling various neighbor configurations._  |
|  template void | [**fillRightFlux&lt; double &gt;**](#function-fillrightflux-double) ([**Param**](class_param.md) XParam, bool doProlongation, int ib, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \*& z) <br> |
|  template \_\_global\_\_ void | [**fillRightFlux&lt; double &gt;**](#function-fillrightflux-double) (int halowidth, bool doProlongation, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, double \* a) <br> |
|  template void | [**fillRightFlux&lt; float &gt;**](#function-fillrightflux-float) ([**Param**](class_param.md) XParam, bool doProlongation, int ib, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \*& z) <br> |
|  template \_\_global\_\_ void | [**fillRightFlux&lt; float &gt;**](#function-fillrightflux-float) (int halowidth, bool doProlongation, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, float \* a) <br> |
|  \_\_global\_\_ void | [**fillRightnew**](#function-fillrightnew) (int halowidth, int nblk, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, T \* a) <br>_CUDA kernel to fill the right halo region of blocks in parallel (new version)._  |
|  template \_\_global\_\_ void | [**fillRightnew&lt; double &gt;**](#function-fillrightnew-double) (int halowidth, int nblk, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillRightnew&lt; float &gt;**](#function-fillrightnew-float) (int halowidth, int nblk, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, float \* a) <br> |
|  void | [**fillTop**](#function-filltop) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& z) <br>_Fills the top halo region of a block, handling various neighbor configurations._  |
|  \_\_global\_\_ void | [**fillTop**](#function-filltop) (int halowidth, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, T \* a) <br>_CUDA kernel to fill the top halo region of blocks in parallel for new refinement, handling various neighbor configurations, new version._  |
|  template \_\_global\_\_ void | [**fillTop&lt; double &gt;**](#function-filltop-double) (int halowidth, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillTop&lt; float &gt;**](#function-filltop-float) (int halowidth, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, float \* a) <br> |
|  void | [**fillTopFlux**](#function-filltopflux) ([**Param**](class_param.md) XParam, bool doProlongation, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& z) <br>_Function to fill the top halo region of a block for new refinement, handling various neighbor configurations._  |
|  \_\_global\_\_ void | [**fillTopFlux**](#function-filltopflux) (int halowidth, bool doProlongation, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, T \* a) <br>_CUDA kernel to fill the top halo region of blocks in parallel for new refinement, handling various neighbor configurations, new version._  |
|  template void | [**fillTopFlux&lt; double &gt;**](#function-filltopflux-double) ([**Param**](class_param.md) XParam, bool doProlongation, int ib, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \*& z) <br> |
|  template \_\_global\_\_ void | [**fillTopFlux&lt; double &gt;**](#function-filltopflux-double) (int halowidth, bool doProlongation, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, double \* a) <br> |
|  template void | [**fillTopFlux&lt; float &gt;**](#function-filltopflux-float) ([**Param**](class_param.md) XParam, bool doProlongation, int ib, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \*& z) <br> |
|  template \_\_global\_\_ void | [**fillTopFlux&lt; float &gt;**](#function-filltopflux-float) (int halowidth, bool doProlongation, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, float \* a) <br> |
|  \_\_global\_\_ void | [**fillTopnew**](#function-filltopnew) (int halowidth, int nblk, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, T \* a) <br> |
|  template \_\_global\_\_ void | [**fillTopnew&lt; double &gt;**](#function-filltopnew-double) (int halowidth, int nblk, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillTopnew&lt; float &gt;**](#function-filltopnew-float) (int halowidth, int nblk, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, float \* a) <br> |
|  void | [**refine\_linear**](#function-refine_linear) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_Wrapping function for refining all sides of active blocks using linear reconstruction._  |
|  template void | [**refine\_linear&lt; double &gt;**](#function-refine_linear-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template void | [**refine\_linear&lt; float &gt;**](#function-refine_linear-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  void | [**refine\_linearGPU**](#function-refine_lineargpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_Wrapping function for refining all sides of active blocks using linear reconstruction on GPU._  |
|  template void | [**refine\_linearGPU&lt; double &gt;**](#function-refine_lineargpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template void | [**refine\_linearGPU&lt; float &gt;**](#function-refine_lineargpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  void | [**refine\_linear\_Bot**](#function-refine_linear_bot) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_Refine a block on the bottom side using linear reconstruction._  |
|  template void | [**refine\_linear\_Bot&lt; double &gt;**](#function-refine_linear_bot-double) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template void | [**refine\_linear\_Bot&lt; float &gt;**](#function-refine_linear_bot-float) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  \_\_global\_\_ void | [**refine\_linear\_BotGPU**](#function-refine_linear_botgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_GPU kernel to refine a block on the bottom side using linear reconstruction._  |
|  template \_\_global\_\_ void | [**refine\_linear\_BotGPU&lt; double &gt;**](#function-refine_linear_botgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template \_\_global\_\_ void | [**refine\_linear\_BotGPU&lt; float &gt;**](#function-refine_linear_botgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  void | [**refine\_linear\_Left**](#function-refine_linear_left) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_Refine a block on the left side using linear reconstruction._  |
|  template void | [**refine\_linear\_Left&lt; double &gt;**](#function-refine_linear_left-double) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template void | [**refine\_linear\_Left&lt; float &gt;**](#function-refine_linear_left-float) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  \_\_global\_\_ void | [**refine\_linear\_LeftGPU**](#function-refine_linear_leftgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_GPU kernel to refine a block on the left side using linear reconstruction._  |
|  template \_\_global\_\_ void | [**refine\_linear\_LeftGPU&lt; double &gt;**](#function-refine_linear_leftgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template \_\_global\_\_ void | [**refine\_linear\_LeftGPU&lt; float &gt;**](#function-refine_linear_leftgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  void | [**refine\_linear\_Right**](#function-refine_linear_right) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_Refine a block on the right side using linear reconstruction._  |
|  template void | [**refine\_linear\_Right&lt; double &gt;**](#function-refine_linear_right-double) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template void | [**refine\_linear\_Right&lt; float &gt;**](#function-refine_linear_right-float) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  \_\_global\_\_ void | [**refine\_linear\_RightGPU**](#function-refine_linear_rightgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_GPU kernel to refine a block on the right side using linear reconstruction._  |
|  template \_\_global\_\_ void | [**refine\_linear\_RightGPU&lt; double &gt;**](#function-refine_linear_rightgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template \_\_global\_\_ void | [**refine\_linear\_RightGPU&lt; float &gt;**](#function-refine_linear_rightgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  void | [**refine\_linear\_Top**](#function-refine_linear_top) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_Refine a block on the top side using linear reconstruction._  |
|  template void | [**refine\_linear\_Top&lt; double &gt;**](#function-refine_linear_top-double) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template void | [**refine\_linear\_Top&lt; float &gt;**](#function-refine_linear_top-float) ([**Param**](class_param.md) XParam, int ib, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  \_\_global\_\_ void | [**refine\_linear\_TopGPU**](#function-refine_linear_topgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br>_GPU kernel to refine a block on the top side using linear reconstruction._  |
|  template \_\_global\_\_ void | [**refine\_linear\_TopGPU&lt; double &gt;**](#function-refine_linear_topgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template \_\_global\_\_ void | [**refine\_linear\_TopGPU&lt; float &gt;**](#function-refine_linear_topgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |




























## Public Functions Documentation




### function HaloFluxCPUBT 

_Wrapping function for applying halo flux correction on the left and right boundaries of all active blocks on GPU._ 
```C++
template<class T>
void HaloFluxCPUBT (
    Param XParam,
    int ib,
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



### function HaloFluxCPULR 

_CPU function for applying halo flux correction on the left and right boundaries._ 
```C++
template<class T>
void HaloFluxCPULR (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T * z
) 
```





**Parameters:**


* `XParam` The model parameters 
* `ib` The index of the block 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 




        

<hr>



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



### function RecalculateZs&lt; double &gt; 

```C++
template void RecalculateZs< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev,
    double * zb
) 
```




<hr>



### function RecalculateZs&lt; float &gt; 

```C++
template void RecalculateZs< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev,
    float * zb
) 
```




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



### function RecalculateZsGPU&lt; double &gt; 

```C++
template __global__ void RecalculateZsGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev,
    double * zb
) 
```




<hr>



### function RecalculateZsGPU&lt; float &gt; 

```C++
template __global__ void RecalculateZsGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev,
    float * zb
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



### function Recalculatehh&lt; double &gt; 

```C++
template void Recalculatehh< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev,
    double * zb
) 
```




<hr>



### function Recalculatehh&lt; float &gt; 

```C++
template void Recalculatehh< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev,
    float * zb
) 
```




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



### function bndmaskGPU&lt; double &gt; 

```C++
template void bndmaskGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev,
    FluxP < double > Flux
) 
```




<hr>



### function bndmaskGPU&lt; float &gt; 

```C++
template void bndmaskGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev,
    FluxP < float > Flux
) 
```




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



### function fillBot&lt; double &gt; 

```C++
template __global__ void fillBot< double > (
    int halowidth,
    int * active,
    int * level,
    int * botleft,
    int * botright,
    int * topleft,
    int * lefttop,
    int * righttop,
    double * a
) 
```




<hr>



### function fillBot&lt; float &gt; 

```C++
template __global__ void fillBot< float > (
    int halowidth,
    int * active,
    int * level,
    int * botleft,
    int * botright,
    int * topleft,
    int * lefttop,
    int * righttop,
    float * a
) 
```




<hr>



### function fillBotFlux 

_Fills the bottom halo region of a block, handling various neighbor configurations._ 
```C++
template<class T>
void fillBotFlux (
    Param XParam,
    bool doProlongation,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```





**Parameters:**


* `XParam` The parameters of the grid/block structure 
* `doProlongation` Flag indicating whether to perform prolongation 
* `ib` The index of the current block 
* `XBlock` The block structure containing neighbor information 
* `z` The variable to be refined 
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



### function fillBotnew&lt; double &gt; 

```C++
template __global__ void fillBotnew< double > (
    int halowidth,
    int nblk,
    int * active,
    int * level,
    int * botleft,
    int * botright,
    int * topleft,
    int * lefttop,
    int * righttop,
    double * a
) 
```




<hr>



### function fillBotnew&lt; float &gt; 

```C++
template __global__ void fillBotnew< float > (
    int halowidth,
    int nblk,
    int * active,
    int * level,
    int * botleft,
    int * botright,
    int * topleft,
    int * lefttop,
    int * righttop,
    float * a
) 
```




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



### function fillCorners&lt; double &gt; 

```C++
template void fillCorners< double > (
    Param XParam,
    BlockP < double > XBlock,
    double *& z
) 
```




<hr>



### function fillCorners&lt; double &gt; 

```C++
template void fillCorners< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > & Xev
) 
```




<hr>



### function fillCorners&lt; double &gt; 

```C++
template void fillCorners< double > (
    Param XParam,
    int ib,
    BlockP < double > XBlock,
    double *& z
) 
```




<hr>



### function fillCorners&lt; float &gt; 

```C++
template void fillCorners< float > (
    Param XParam,
    BlockP < float > XBlock,
    float *& z
) 
```




<hr>



### function fillCorners&lt; float &gt; 

```C++
template void fillCorners< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > & Xev
) 
```




<hr>



### function fillCorners&lt; float &gt; 

```C++
template void fillCorners< float > (
    Param XParam,
    int ib,
    BlockP < float > XBlock,
    float *& z
) 
```




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



### function fillCornersGPU&lt; double &gt; 

```C++
template __global__ void fillCornersGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z
) 
```




<hr>



### function fillCornersGPU&lt; float &gt; 

```C++
template __global__ void fillCornersGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z
) 
```




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



### function fillHalo&lt; double &gt; 

```C++
template void fillHalo< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev,
    double * zb
) 
```




<hr>



### function fillHalo&lt; double &gt; 

```C++
template void fillHalo< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev
) 
```




<hr>



### function fillHalo&lt; double &gt; 

```C++
template void fillHalo< double > (
    Param XParam,
    BlockP < double > XBlock,
    GradientsP < double > Grad
) 
```




<hr>



### function fillHalo&lt; double &gt; 

```C++
template void fillHalo< double > (
    Param XParam,
    BlockP < double > XBlock,
    FluxP < double > Flux
) 
```




<hr>



### function fillHalo&lt; float &gt; 

```C++
template void fillHalo< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev,
    float * zb
) 
```




<hr>



### function fillHalo&lt; float &gt; 

```C++
template void fillHalo< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev
) 
```




<hr>



### function fillHalo&lt; float &gt; 

```C++
template void fillHalo< float > (
    Param XParam,
    BlockP < float > XBlock,
    GradientsP < float > Grad
) 
```




<hr>



### function fillHalo&lt; float &gt; 

```C++
template void fillHalo< float > (
    Param XParam,
    BlockP < float > XBlock,
    FluxP < float > Flux
) 
```




<hr>



### function fillHaloBTFluxC 

_Wrapping function for calculating flux for halos for each block of a single variable on GPU._ 
```C++
template<class T>
void fillHaloBTFluxC (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to work on 



**Note:**

For flux term and actually most terms, only top and right neighbours are needed! 





        

<hr>



### function fillHaloBTFluxC&lt; double &gt; 

```C++
template void fillHaloBTFluxC< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z
) 
```




<hr>



### function fillHaloBTFluxC&lt; float &gt; 

```C++
template void fillHaloBTFluxC< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z
) 
```




<hr>



### function fillHaloBotTopGPU 

_Wrapping function for calculating flux for halos for each block of a single variable on GPU._ 
```C++
template<class T>
void fillHaloBotTopGPU (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```



### Description



This function is a wraping function of the halo flux functions on GPU. It is called from the main Halo GPU function. The present imnplementation is naive and slow one that calls the rather complex fillLeft type functions 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `stream` The cuda stream to use 
* `z` The variable to work on 





        

<hr>



### function fillHaloBotTopGPU&lt; double &gt; 

```C++
template void fillHaloBotTopGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    cudaStream_t stream,
    double * z
) 
```




<hr>



### function fillHaloBotTopGPU&lt; float &gt; 

```C++
template void fillHaloBotTopGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    cudaStream_t stream,
    float * z
) 
```




<hr>



### function fillHaloBotTopGPUnew 

_Wrapping function for calculating flux for halos for each block of a single variable on GPU. New version._ 
```C++
template<class T>
void fillHaloBotTopGPUnew (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```



### Description



This function is a wraping function of the halo flux functions on GPU. It is called from the main Halo GPU function. The present imnplementation is naive and slow one that calls the rather complex fillLeft type functions 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `stream` The cuda stream to use 
* `z` The variable to work on 





        

<hr>



### function fillHaloBotTopGPUnew&lt; double &gt; 

```C++
template void fillHaloBotTopGPUnew< double > (
    Param XParam,
    BlockP < double > XBlock,
    cudaStream_t stream,
    double * z
) 
```




<hr>



### function fillHaloBotTopGPUnew&lt; float &gt; 

```C++
template void fillHaloBotTopGPUnew< float > (
    Param XParam,
    BlockP < float > XBlock,
    cudaStream_t stream,
    float * z
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



!


### Description



This function is a wraping fuction of the halo functions on CPU. It is called from the main Halo CPU function. This is layer 2 of 3 wrap so the candy doesn't stick too much. 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to work on 





        

<hr>



### function fillHaloC&lt; double &gt; 

```C++
template void fillHaloC< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z
) 
```




<hr>



### function fillHaloC&lt; float &gt; 

```C++
template void fillHaloC< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z
) 
```




<hr>



### function fillHaloD 

_Wrapping function for calculating halos on CPU on every side of a block of a single variable._ 
```C++
template<class T>
void fillHaloD (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T * z
) 
```



!


### Description



This fuction is a wraping fuction of the halo functions for CPU. It is called from another wraping function to keep things clean. In a sense this is the third (and last) layer of wrapping




**Parameters:**


* `XParam` The model parameters 
* `ib` The block index to work on 
* `XBlock` The block structure containing the block information 
* `z` The variable to work on 





        

<hr>



### function fillHaloD&lt; double &gt; 

```C++
template void fillHaloD< double > (
    Param XParam,
    int ib,
    BlockP < double > XBlock,
    double * z
) 
```




<hr>



### function fillHaloD&lt; float &gt; 

```C++
template void fillHaloD< float > (
    Param XParam,
    int ib,
    BlockP < float > XBlock,
    float * z
) 
```




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



### function fillHaloF&lt; double &gt; 

```C++
template void fillHaloF< double > (
    Param XParam,
    bool doProlongation,
    BlockP < double > XBlock,
    double * z
) 
```




<hr>



### function fillHaloF&lt; float &gt; 

```C++
template void fillHaloF< float > (
    Param XParam,
    bool doProlongation,
    BlockP < float > XBlock,
    float * z
) 
```




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



### function fillHaloGPU&lt; double &gt; 

```C++
template void fillHaloGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    cudaStream_t stream,
    double * z
) 
```




<hr>



### function fillHaloGPU&lt; double &gt; 

```C++
template void fillHaloGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z
) 
```




<hr>



### function fillHaloGPU&lt; double &gt; 

```C++
template void fillHaloGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev
) 
```




<hr>



### function fillHaloGPU&lt; double &gt; 

```C++
template void fillHaloGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev,
    double * zb
) 
```




<hr>



### function fillHaloGPU&lt; double &gt; 

```C++
template void fillHaloGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    GradientsP < double > Grad
) 
```




<hr>



### function fillHaloGPU&lt; double &gt; 

```C++
template void fillHaloGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    FluxP < double > Flux
) 
```




<hr>



### function fillHaloGPU&lt; float &gt; 

```C++
template void fillHaloGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    cudaStream_t stream,
    float * z
) 
```




<hr>



### function fillHaloGPU&lt; float &gt; 

```C++
template void fillHaloGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z
) 
```




<hr>



### function fillHaloGPU&lt; float &gt; 

```C++
template void fillHaloGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev
) 
```




<hr>



### function fillHaloGPU&lt; float &gt; 

```C++
template void fillHaloGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev,
    float * zb
) 
```




<hr>



### function fillHaloGPU&lt; float &gt; 

```C++
template void fillHaloGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    GradientsP < float > Grad
) 
```




<hr>



### function fillHaloGPU&lt; float &gt; 

```C++
template void fillHaloGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    FluxP < float > Flux
) 
```




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



### function fillHaloGPUnew&lt; double &gt; 

```C++
template void fillHaloGPUnew< double > (
    Param XParam,
    BlockP < double > XBlock,
    cudaStream_t stream,
    double * z
) 
```




<hr>



### function fillHaloGPUnew&lt; float &gt; 

```C++
template void fillHaloGPUnew< float > (
    Param XParam,
    BlockP < float > XBlock,
    cudaStream_t stream,
    float * z
) 
```




<hr>



### function fillHaloLRFluxC 

_Wrapping function for calculating flux for halos for each block of a single variable on GPU._ 
```C++
template<class T>
void fillHaloLRFluxC (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to work on 



**Note:**

For flux term and actually most terms, only top and right neighbours are needed! 





        

<hr>



### function fillHaloLRFluxC&lt; double &gt; 

```C++
template void fillHaloLRFluxC< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z
) 
```




<hr>



### function fillHaloLRFluxC&lt; float &gt; 

```C++
template void fillHaloLRFluxC< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z
) 
```




<hr>



### function fillHaloLeftRightGPU 

_Wrapping function for calculating for halos for each block of a single variable on GPU._ 
```C++
template<class T>
void fillHaloLeftRightGPU (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `stream` The cuda stream to use 
* `z` The variable to work on 



**Note:**

For flux term and actually most terms, only top and right neighbours are needed! 





        

<hr>



### function fillHaloLeftRightGPU&lt; double &gt; 

```C++
template void fillHaloLeftRightGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    cudaStream_t stream,
    double * z
) 
```




<hr>



### function fillHaloLeftRightGPU&lt; float &gt; 

```C++
template void fillHaloLeftRightGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    cudaStream_t stream,
    float * z
) 
```




<hr>



### function fillHaloLeftRightGPUnew 

_Wrapping function for calculating for halos for each block of a single variable on GPU. New version._ 
```C++
template<class T>
void fillHaloLeftRightGPUnew (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `stream` The cuda stream to use 
* `z` The variable to work on 




        

<hr>



### function fillHaloLeftRightGPUnew&lt; double &gt; 

```C++
template void fillHaloLeftRightGPUnew< double > (
    Param XParam,
    BlockP < double > XBlock,
    cudaStream_t stream,
    double * z
) 
```




<hr>



### function fillHaloLeftRightGPUnew&lt; float &gt; 

```C++
template void fillHaloLeftRightGPUnew< float > (
    Param XParam,
    BlockP < float > XBlock,
    cudaStream_t stream,
    float * z
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

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to work on 



**Note:**

For flux term and actually most terms, only top and right neighbours are needed! 






        

<hr>



### function fillHaloTopRightC&lt; double &gt; 

```C++
template void fillHaloTopRightC< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z
) 
```




<hr>



### function fillHaloTopRightC&lt; float &gt; 

```C++
template void fillHaloTopRightC< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z
) 
```




<hr>



### function fillHaloTopRightGPU 

_Wrapping function for calculating flux for halos for each block of a single variable on GPU._ 
```C++
template<class T>
void fillHaloTopRightGPU (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```





**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
 
* `stream` The cuda stream to use 
* `z` The variable to work on 



**Note:**

For flux term and actually most terms, only top and right neighbours are needed! 





        

<hr>



### function fillHaloTopRightGPU&lt; double &gt; 

```C++
template void fillHaloTopRightGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    cudaStream_t stream,
    double * z
) 
```




<hr>



### function fillHaloTopRightGPU&lt; float &gt; 

```C++
template void fillHaloTopRightGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    cudaStream_t stream,
    float * z
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



### function fillLeft&lt; double &gt; 

```C++
template __global__ void fillLeft< double > (
    int halowidth,
    int * active,
    int * level,
    int * leftbot,
    int * lefttop,
    int * rightbot,
    int * botright,
    int * topright,
    double * a
) 
```




<hr>



### function fillLeft&lt; float &gt; 

```C++
template __global__ void fillLeft< float > (
    int halowidth,
    int * active,
    int * level,
    int * leftbot,
    int * lefttop,
    int * rightbot,
    int * botright,
    int * topright,
    float * a
) 
```




<hr>



### function fillLeftFlux 

_CPU function for applying halo flux correction on the left boundaries of a specific block._ 
```C++
template<class T>
void fillLeftFlux (
    Param XParam,
    bool doProlongation,
    int ib,
    BlockP < T > XBlock,
    T *& z
) 
```





**Parameters:**


* `XParam` The simulation parameters 
* `doProlongation` Flag indicating whether to perform prolongation 
* `ib` The index of the block to process 
* `XBlock` The block structure containing neighbor information 
* `z` The variable to be refined 




        

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



### function fillLeftnew&lt; double &gt; 

```C++
template __global__ void fillLeftnew< double > (
    int halowidth,
    int nblk,
    int * active,
    int * level,
    int * leftbot,
    int * lefttop,
    int * rightbot,
    int * botright,
    int * topright,
    double * a
) 
```




<hr>



### function fillLeftnew&lt; float &gt; 

```C++
template __global__ void fillLeftnew< float > (
    int halowidth,
    int nblk,
    int * active,
    int * level,
    int * leftbot,
    int * lefttop,
    int * rightbot,
    int * botright,
    int * topright,
    float * a
) 
```




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



### function fillRight&lt; double &gt; 

```C++
template __global__ void fillRight< double > (
    int halowidth,
    int * active,
    int * level,
    int * rightbot,
    int * righttop,
    int * leftbot,
    int * botleft,
    int * topleft,
    double * a
) 
```




<hr>



### function fillRight&lt; float &gt; 

```C++
template __global__ void fillRight< float > (
    int halowidth,
    int * active,
    int * level,
    int * rightbot,
    int * righttop,
    int * leftbot,
    int * botleft,
    int * topleft,
    float * a
) 
```




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



### function fillRightFlux&lt; double &gt; 

```C++
template void fillRightFlux< double > (
    Param XParam,
    bool doProlongation,
    int ib,
    BlockP < double > XBlock,
    double *& z
) 
```




<hr>



### function fillRightFlux&lt; double &gt; 

```C++
template __global__ void fillRightFlux< double > (
    int halowidth,
    bool doProlongation,
    int * active,
    int * level,
    int * rightbot,
    int * righttop,
    int * leftbot,
    int * botleft,
    int * topleft,
    double * a
) 
```




<hr>



### function fillRightFlux&lt; float &gt; 

```C++
template void fillRightFlux< float > (
    Param XParam,
    bool doProlongation,
    int ib,
    BlockP < float > XBlock,
    float *& z
) 
```




<hr>



### function fillRightFlux&lt; float &gt; 

```C++
template __global__ void fillRightFlux< float > (
    int halowidth,
    bool doProlongation,
    int * active,
    int * level,
    int * rightbot,
    int * righttop,
    int * leftbot,
    int * botleft,
    int * topleft,
    float * a
) 
```




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



### function fillRightnew&lt; double &gt; 

```C++
template __global__ void fillRightnew< double > (
    int halowidth,
    int nblk,
    int * active,
    int * level,
    int * rightbot,
    int * righttop,
    int * leftbot,
    int * botleft,
    int * topleft,
    double * a
) 
```




<hr>



### function fillRightnew&lt; float &gt; 

```C++
template __global__ void fillRightnew< float > (
    int halowidth,
    int nblk,
    int * active,
    int * level,
    int * rightbot,
    int * righttop,
    int * leftbot,
    int * botleft,
    int * topleft,
    float * a
) 
```




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



### function fillTop&lt; double &gt; 

```C++
template __global__ void fillTop< double > (
    int halowidth,
    int * active,
    int * level,
    int * topleft,
    int * topright,
    int * botleft,
    int * leftbot,
    int * rightbot,
    double * a
) 
```




<hr>



### function fillTop&lt; float &gt; 

```C++
template __global__ void fillTop< float > (
    int halowidth,
    int * active,
    int * level,
    int * topleft,
    int * topright,
    int * botleft,
    int * leftbot,
    int * rightbot,
    float * a
) 
```




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



### function fillTopFlux&lt; double &gt; 

```C++
template void fillTopFlux< double > (
    Param XParam,
    bool doProlongation,
    int ib,
    BlockP < double > XBlock,
    double *& z
) 
```




<hr>



### function fillTopFlux&lt; double &gt; 

```C++
template __global__ void fillTopFlux< double > (
    int halowidth,
    bool doProlongation,
    int * active,
    int * level,
    int * topleft,
    int * topright,
    int * botleft,
    int * leftbot,
    int * rightbot,
    double * a
) 
```




<hr>



### function fillTopFlux&lt; float &gt; 

```C++
template void fillTopFlux< float > (
    Param XParam,
    bool doProlongation,
    int ib,
    BlockP < float > XBlock,
    float *& z
) 
```




<hr>



### function fillTopFlux&lt; float &gt; 

```C++
template __global__ void fillTopFlux< float > (
    int halowidth,
    bool doProlongation,
    int * active,
    int * level,
    int * topleft,
    int * topright,
    int * botleft,
    int * leftbot,
    int * rightbot,
    float * a
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



### function fillTopnew&lt; double &gt; 

```C++
template __global__ void fillTopnew< double > (
    int halowidth,
    int nblk,
    int * active,
    int * level,
    int * topleft,
    int * topright,
    int * botleft,
    int * leftbot,
    int * rightbot,
    double * a
) 
```




<hr>



### function fillTopnew&lt; float &gt; 

```C++
template __global__ void fillTopnew< float > (
    int halowidth,
    int nblk,
    int * active,
    int * level,
    int * topleft,
    int * topright,
    int * botleft,
    int * leftbot,
    int * rightbot,
    float * a
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



### function refine\_linear&lt; double &gt; 

```C++
template void refine_linear< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z,
    double * dzdx,
    double * dzdy
) 
```




<hr>



### function refine\_linear&lt; float &gt; 

```C++
template void refine_linear< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z,
    float * dzdx,
    float * dzdy
) 
```




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



### function refine\_linearGPU&lt; double &gt; 

```C++
template void refine_linearGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z,
    double * dzdx,
    double * dzdy
) 
```




<hr>



### function refine\_linearGPU&lt; float &gt; 

```C++
template void refine_linearGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z,
    float * dzdx,
    float * dzdy
) 
```




<hr>



### function refine\_linear\_Bot 

_Refine a block on the bottom side using linear reconstruction._ 
```C++
template<class T>
void refine_linear_Bot (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T * z,
    T * dzdx,
    T * dzdy
) 
```



### Description



This function refines a block on the bottom side using linear reconstruction. It checks if the neighboring block on the bottom is at a coarser level. If so, it calculates the new values for the bottom boundary of the current block using the gradients in the x and y directions. The new values are computed based on the distance to the neighboring block and the gradients, ensuring a smooth transition between different resolution levels. 

**Parameters:**


* `XParam` The model parameters 
* `ib` The index of the current block 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 
* `dzdx` The gradient of z in the x direction 
* `dzdy` The gradient of z in the y direction 
 





        

<hr>



### function refine\_linear\_Bot&lt; double &gt; 

```C++
template void refine_linear_Bot< double > (
    Param XParam,
    int ib,
    BlockP < double > XBlock,
    double * z,
    double * dzdx,
    double * dzdy
) 
```




<hr>



### function refine\_linear\_Bot&lt; float &gt; 

```C++
template void refine_linear_Bot< float > (
    Param XParam,
    int ib,
    BlockP < float > XBlock,
    float * z,
    float * dzdx,
    float * dzdy
) 
```




<hr>



### function refine\_linear\_BotGPU 

_GPU kernel to refine a block on the bottom side using linear reconstruction._ 
```C++
template<class T>
__global__ void refine_linear_BotGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * z,
    T * dzdx,
    T * dzdy
) 
```



### Description



This GPU kernel refines a block on the bottom side using linear reconstruction. It checks if the neighboring block on the bottom is at a coarser level. If so, it calculates the new values for the bottom boundary of the current block using the gradients in the x and y directions. The new values are computed based on the distance to the neighboring block and the gradients, ensuring a smooth transition between different resolution levels. Each thread processes a specific column of the block, allowing for parallel computation across multiple blocks. 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 
* `dzdx` The gradient of z in the x direction 
* `dzdy` The gradient of z in the y direction 





        

<hr>



### function refine\_linear\_BotGPU&lt; double &gt; 

```C++
template __global__ void refine_linear_BotGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z,
    double * dzdx,
    double * dzdy
) 
```




<hr>



### function refine\_linear\_BotGPU&lt; float &gt; 

```C++
template __global__ void refine_linear_BotGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z,
    float * dzdx,
    float * dzdy
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



### function refine\_linear\_Left&lt; double &gt; 

```C++
template void refine_linear_Left< double > (
    Param XParam,
    int ib,
    BlockP < double > XBlock,
    double * z,
    double * dzdx,
    double * dzdy
) 
```




<hr>



### function refine\_linear\_Left&lt; float &gt; 

```C++
template void refine_linear_Left< float > (
    Param XParam,
    int ib,
    BlockP < float > XBlock,
    float * z,
    float * dzdx,
    float * dzdy
) 
```




<hr>



### function refine\_linear\_LeftGPU 

_GPU kernel to refine a block on the left side using linear reconstruction._ 
```C++
template<class T>
__global__ void refine_linear_LeftGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * z,
    T * dzdx,
    T * dzdy
) 
```



### Description



This GPU kernel refines a block on the left side using linear reconstruction. It checks if the neighboring block on the left is at a coarser level. If so, it calculates the new values for the left boundary of the current block using the gradients in the x and y directions. The new values are computed based on the distance to the neighboring block and the gradients, ensuring a smooth transition between different resolution levels. Each thread processes a specific row of the block, allowing for parallel computation across multiple blocks. 
 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 
* `dzdx` The gradient of z in the x direction 
* `dzdy` The gradient of z in the y direction 
* `T` The data type (float or double) 





        

<hr>



### function refine\_linear\_LeftGPU&lt; double &gt; 

```C++
template __global__ void refine_linear_LeftGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z,
    double * dzdx,
    double * dzdy
) 
```




<hr>



### function refine\_linear\_LeftGPU&lt; float &gt; 

```C++
template __global__ void refine_linear_LeftGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z,
    float * dzdx,
    float * dzdy
) 
```




<hr>



### function refine\_linear\_Right 

_Refine a block on the right side using linear reconstruction._ 
```C++
template<class T>
void refine_linear_Right (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T * z,
    T * dzdx,
    T * dzdy
) 
```



### Description



This function refines a block on the right side using linear reconstruction. It checks if the neighboring block on the right is at a coarser level. If so, it calculates the new values for the right boundary of the current block using the gradients in the x and y directions. The new values are computed based on the distance to the neighboring block and the gradients, ensuring a smooth transition between different resolution levels. 

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



### function refine\_linear\_Right&lt; double &gt; 

```C++
template void refine_linear_Right< double > (
    Param XParam,
    int ib,
    BlockP < double > XBlock,
    double * z,
    double * dzdx,
    double * dzdy
) 
```




<hr>



### function refine\_linear\_Right&lt; float &gt; 

```C++
template void refine_linear_Right< float > (
    Param XParam,
    int ib,
    BlockP < float > XBlock,
    float * z,
    float * dzdx,
    float * dzdy
) 
```




<hr>



### function refine\_linear\_RightGPU 

_GPU kernel to refine a block on the right side using linear reconstruction._ 
```C++
template<class T>
__global__ void refine_linear_RightGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * z,
    T * dzdx,
    T * dzdy
) 
```



### Description



This GPU kernel refines a block on the right side using linear reconstruction. It checks if the neighboring block on the right is at a coarser level. If so, it calculates the new values for the right boundary of the current block using the gradients in the x and y directions. The new values are computed based on the distance to the neighboring block and the gradients, ensuring a smooth transition between different resolution levels. Each thread processes a specific row of the block, allowing for parallel computation across multiple blocks. 
 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 
* `dzdx` The gradient of z in the x direction 
* `dzdy` The gradient of z in the y direction 
* `T` The data type (float or double) 





        

<hr>



### function refine\_linear\_RightGPU&lt; double &gt; 

```C++
template __global__ void refine_linear_RightGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z,
    double * dzdx,
    double * dzdy
) 
```




<hr>



### function refine\_linear\_RightGPU&lt; float &gt; 

```C++
template __global__ void refine_linear_RightGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z,
    float * dzdx,
    float * dzdy
) 
```




<hr>



### function refine\_linear\_Top 

_Refine a block on the top side using linear reconstruction._ 
```C++
template<class T>
void refine_linear_Top (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T * z,
    T * dzdx,
    T * dzdy
) 
```



### Description



This function refines a block on the top side using linear reconstruction. It checks if the neighboring block on the top is at a coarser level. If so, it calculates the new values for the top boundary of the current block using the gradients in the x and y directions. The new values are computed based on the distance to the neighboring block and the gradients, ensuring a smooth transition between different resolution levels. 

**Parameters:**


* `XParam` The model parameters 
* `ib` The index of the current block 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 
* `dzdx` The gradient of z in the x direction 
* `dzdy` The gradient of z in the y direction 
 





        

<hr>



### function refine\_linear\_Top&lt; double &gt; 

```C++
template void refine_linear_Top< double > (
    Param XParam,
    int ib,
    BlockP < double > XBlock,
    double * z,
    double * dzdx,
    double * dzdy
) 
```




<hr>



### function refine\_linear\_Top&lt; float &gt; 

```C++
template void refine_linear_Top< float > (
    Param XParam,
    int ib,
    BlockP < float > XBlock,
    float * z,
    float * dzdx,
    float * dzdy
) 
```




<hr>



### function refine\_linear\_TopGPU 

_GPU kernel to refine a block on the top side using linear reconstruction._ 
```C++
template<class T>
__global__ void refine_linear_TopGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * z,
    T * dzdx,
    T * dzdy
) 
```



### Description



This GPU kernel refines a block on the top side using linear reconstruction. It checks if the neighboring block on the top is at a coarser level. If so, it calculates the new values for the top boundary of the current block using the gradients in the x and y directions. The new values are computed based on the distance to the neighboring block and the gradients, ensuring a smooth transition between different resolution levels. Each thread processes a specific column of the block, allowing for parallel computation across multiple blocks. 

**Parameters:**


* `XParam` The model parameters 
* `XBlock` The block structure containing the block information 
* `z` The variable to be refined 
* `dzdx` The gradient of z in the x direction 
* `dzdy` The gradient of z in the y direction 
* `T` The data type (float or double) 





        

<hr>



### function refine\_linear\_TopGPU&lt; double &gt; 

```C++
template __global__ void refine_linear_TopGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z,
    double * dzdx,
    double * dzdy
) 
```




<hr>



### function refine\_linear\_TopGPU&lt; float &gt; 

```C++
template __global__ void refine_linear_TopGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z,
    float * dzdx,
    float * dzdy
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Halo.cu`


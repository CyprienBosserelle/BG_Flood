

# File Halo.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Halo.cu**](Halo_8cu.md)

[Go to the source code of this file](Halo_8cu_source.md)



* `#include "Halo.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**HaloFluxCPUBT**](#function-halofluxcpubt) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  void | [**HaloFluxCPULR**](#function-halofluxcpulr) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  \_\_global\_\_ void | [**HaloFluxGPUBT**](#function-halofluxgpubt) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  \_\_global\_\_ void | [**HaloFluxGPUBTnew**](#function-halofluxgpubtnew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  \_\_global\_\_ void | [**HaloFluxGPULR**](#function-halofluxgpulr) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  \_\_global\_\_ void | [**HaloFluxGPULRnew**](#function-halofluxgpulrnew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  void | [**RecalculateZs**](#function-recalculatezs) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br>_Recalculate water surface after recalculating the values on the halo on the CPU._  |
|  template void | [**RecalculateZs&lt; double &gt;**](#function-recalculatezs-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template void | [**RecalculateZs&lt; float &gt;**](#function-recalculatezs-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev, float \* zb) <br> |
|  \_\_global\_\_ void | [**RecalculateZsGPU**](#function-recalculatezsgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  template \_\_global\_\_ void | [**RecalculateZsGPU&lt; double &gt;**](#function-recalculatezsgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template \_\_global\_\_ void | [**RecalculateZsGPU&lt; float &gt;**](#function-recalculatezsgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev, float \* zb) <br> |
|  void | [**Recalculatehh**](#function-recalculatehh) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  template void | [**Recalculatehh&lt; double &gt;**](#function-recalculatehh-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template void | [**Recalculatehh&lt; float &gt;**](#function-recalculatehh-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev, float \* zb) <br> |
|  void | [**bndmaskGPU**](#function-bndmaskgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  template void | [**bndmaskGPU&lt; double &gt;**](#function-bndmaskgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev, [**FluxP**](structFluxP.md)&lt; double &gt; Flux) <br> |
|  template void | [**bndmaskGPU&lt; float &gt;**](#function-bndmaskgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev, [**FluxP**](structFluxP.md)&lt; float &gt; Flux) <br> |
|  void | [**fillBot**](#function-fillbot) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillBot**](#function-fillbot) (int halowidth, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, T \* a) <br> |
|  template \_\_global\_\_ void | [**fillBot&lt; double &gt;**](#function-fillbot-double) (int halowidth, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillBot&lt; float &gt;**](#function-fillbot-float) (int halowidth, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, float \* a) <br> |
|  void | [**fillBotFlux**](#function-fillbotflux) ([**Param**](classParam.md) XParam, bool doProlongation, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillBotnew**](#function-fillbotnew) (int halowidth, int nblk, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, T \* a) <br> |
|  template \_\_global\_\_ void | [**fillBotnew&lt; double &gt;**](#function-fillbotnew-double) (int halowidth, int nblk, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillBotnew&lt; float &gt;**](#function-fillbotnew-float) (int halowidth, int nblk, int \* active, int \* level, int \* botleft, int \* botright, int \* topleft, int \* lefttop, int \* righttop, float \* a) <br> |
|  void | [**fillCorners**](#function-fillcorners) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  void | [**fillCorners**](#function-fillcorners) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & Xev) <br> |
|  void | [**fillCorners**](#function-fillcorners) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  template void | [**fillCorners&lt; double &gt;**](#function-fillcorners-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \*& z) <br> |
|  template void | [**fillCorners&lt; double &gt;**](#function-fillcorners-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; & Xev) <br> |
|  template void | [**fillCorners&lt; double &gt;**](#function-fillcorners-double) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \*& z) <br> |
|  template void | [**fillCorners&lt; float &gt;**](#function-fillcorners-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \*& z) <br> |
|  template void | [**fillCorners&lt; float &gt;**](#function-fillcorners-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; & Xev) <br> |
|  template void | [**fillCorners&lt; float &gt;**](#function-fillcorners-float) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \*& z) <br> |
|  \_\_global\_\_ void | [**fillCornersGPU**](#function-fillcornersgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  template \_\_global\_\_ void | [**fillCornersGPU&lt; double &gt;**](#function-fillcornersgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template \_\_global\_\_ void | [**fillCornersGPU&lt; float &gt;**](#function-fillcornersgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev) <br> |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**GradientsP**](structGradientsP.md)&lt; T &gt; Grad) <br> |
|  void | [**fillHalo**](#function-fillhalo) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  template void | [**fillHalo&lt; double &gt;**](#function-fillhalo-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template void | [**fillHalo&lt; double &gt;**](#function-fillhalo-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev) <br> |
|  template void | [**fillHalo&lt; double &gt;**](#function-fillhalo-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**GradientsP**](structGradientsP.md)&lt; double &gt; Grad) <br> |
|  template void | [**fillHalo&lt; double &gt;**](#function-fillhalo-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**FluxP**](structFluxP.md)&lt; double &gt; Flux) <br> |
|  template void | [**fillHalo&lt; float &gt;**](#function-fillhalo-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev, float \* zb) <br> |
|  template void | [**fillHalo&lt; float &gt;**](#function-fillhalo-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev) <br> |
|  template void | [**fillHalo&lt; float &gt;**](#function-fillhalo-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**GradientsP**](structGradientsP.md)&lt; float &gt; Grad) <br> |
|  template void | [**fillHalo&lt; float &gt;**](#function-fillhalo-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**FluxP**](structFluxP.md)&lt; float &gt; Flux) <br> |
|  void | [**fillHaloBTFluxC**](#function-fillhalobtfluxc) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  template void | [**fillHaloBTFluxC&lt; double &gt;**](#function-fillhalobtfluxc-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloBTFluxC&lt; float &gt;**](#function-fillhalobtfluxc-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHaloBotTopGPU**](#function-fillhalobottopgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br> |
|  template void | [**fillHaloBotTopGPU&lt; double &gt;**](#function-fillhalobottopgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloBotTopGPU&lt; float &gt;**](#function-fillhalobottopgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  void | [**fillHaloBotTopGPUnew**](#function-fillhalobottopgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br> |
|  template void | [**fillHaloBotTopGPUnew&lt; double &gt;**](#function-fillhalobottopgpunew-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloBotTopGPUnew&lt; float &gt;**](#function-fillhalobottopgpunew-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  void | [**fillHaloC**](#function-fillhaloc) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating halos for each block of a single variable on CPU._  |
|  template void | [**fillHaloC&lt; double &gt;**](#function-fillhaloc-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloC&lt; float &gt;**](#function-fillhaloc-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHaloD**](#function-fillhalod) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating halos on CPU on every side of a block of a single variable._  |
|  template void | [**fillHaloD&lt; double &gt;**](#function-fillhalod-double) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloD&lt; float &gt;**](#function-fillhalod-float) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHaloF**](#function-fillhalof) ([**Param**](classParam.md) XParam, bool doProlongation, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating flux in the halos for a block and a single variable on CPU._  |
|  template void | [**fillHaloF&lt; double &gt;**](#function-fillhalof-double) ([**Param**](classParam.md) XParam, bool doProlongation, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloF&lt; float &gt;**](#function-fillhalof-float) ([**Param**](classParam.md) XParam, bool doProlongation, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br>_Wrapping function for calculating halos for each block of a single variable on GPU._  |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev) <br> |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**GradientsP**](structGradientsP.md)&lt; T &gt; Grad) <br> |
|  void | [**fillHaloGPU**](#function-fillhalogpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  template void | [**fillHaloGPU&lt; double &gt;**](#function-fillhalogpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloGPU&lt; double &gt;**](#function-fillhalogpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloGPU&lt; double &gt;**](#function-fillhalogpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev) <br> |
|  template void | [**fillHaloGPU&lt; double &gt;**](#function-fillhalogpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template void | [**fillHaloGPU&lt; double &gt;**](#function-fillhalogpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**GradientsP**](structGradientsP.md)&lt; double &gt; Grad) <br> |
|  template void | [**fillHaloGPU&lt; double &gt;**](#function-fillhalogpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**FluxP**](structFluxP.md)&lt; double &gt; Flux) <br> |
|  template void | [**fillHaloGPU&lt; float &gt;**](#function-fillhalogpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  template void | [**fillHaloGPU&lt; float &gt;**](#function-fillhalogpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z) <br> |
|  template void | [**fillHaloGPU&lt; float &gt;**](#function-fillhalogpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev) <br> |
|  template void | [**fillHaloGPU&lt; float &gt;**](#function-fillhalogpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev, float \* zb) <br> |
|  template void | [**fillHaloGPU&lt; float &gt;**](#function-fillhalogpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**GradientsP**](structGradientsP.md)&lt; float &gt; Grad) <br> |
|  template void | [**fillHaloGPU&lt; float &gt;**](#function-fillhalogpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**FluxP**](structFluxP.md)&lt; float &gt; Flux) <br> |
|  void | [**fillHaloGPUnew**](#function-fillhalogpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br> |
|  template void | [**fillHaloGPUnew&lt; double &gt;**](#function-fillhalogpunew-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloGPUnew&lt; float &gt;**](#function-fillhalogpunew-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  void | [**fillHaloLRFluxC**](#function-fillhalolrfluxc) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  template void | [**fillHaloLRFluxC&lt; double &gt;**](#function-fillhalolrfluxc-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloLRFluxC&lt; float &gt;**](#function-fillhalolrfluxc-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHaloLeftRightGPU**](#function-fillhaloleftrightgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br> |
|  template void | [**fillHaloLeftRightGPU&lt; double &gt;**](#function-fillhaloleftrightgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloLeftRightGPU&lt; float &gt;**](#function-fillhaloleftrightgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  void | [**fillHaloLeftRightGPUnew**](#function-fillhaloleftrightgpunew) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br> |
|  template void | [**fillHaloLeftRightGPUnew&lt; double &gt;**](#function-fillhaloleftrightgpunew-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloLeftRightGPUnew&lt; float &gt;**](#function-fillhaloleftrightgpunew-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  void | [**fillHaloTopRightC**](#function-fillhalotoprightc) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br>_Wrapping function for calculating flux for halos for each block of a single variable on GPU._  |
|  template void | [**fillHaloTopRightC&lt; double &gt;**](#function-fillhalotoprightc-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillHaloTopRightC&lt; float &gt;**](#function-fillhalotoprightc-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**fillHaloTopRightGPU**](#function-fillhalotoprightgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, cudaStream\_t stream, T \* z) <br> |
|  template void | [**fillHaloTopRightGPU&lt; double &gt;**](#function-fillhalotoprightgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, cudaStream\_t stream, double \* z) <br> |
|  template void | [**fillHaloTopRightGPU&lt; float &gt;**](#function-fillhalotoprightgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, cudaStream\_t stream, float \* z) <br> |
|  void | [**fillLeft**](#function-fillleft) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillLeft**](#function-fillleft) (int halowidth, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, T \* a) <br> |
|  template \_\_global\_\_ void | [**fillLeft&lt; double &gt;**](#function-fillleft-double) (int halowidth, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillLeft&lt; float &gt;**](#function-fillleft-float) (int halowidth, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, float \* a) <br> |
|  void | [**fillLeftFlux**](#function-fillleftflux) ([**Param**](classParam.md) XParam, bool doProlongation, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillLeftnew**](#function-fillleftnew) (int halowidth, int nblk, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, T \* a) <br> |
|  template \_\_global\_\_ void | [**fillLeftnew&lt; double &gt;**](#function-fillleftnew-double) (int halowidth, int nblk, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillLeftnew&lt; float &gt;**](#function-fillleftnew-float) (int halowidth, int nblk, int \* active, int \* level, int \* leftbot, int \* lefttop, int \* rightbot, int \* botright, int \* topright, float \* a) <br> |
|  void | [**fillRight**](#function-fillright) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillRight**](#function-fillright) (int halowidth, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, T \* a) <br> |
|  template \_\_global\_\_ void | [**fillRight&lt; double &gt;**](#function-fillright-double) (int halowidth, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillRight&lt; float &gt;**](#function-fillright-float) (int halowidth, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, float \* a) <br> |
|  void | [**fillRightFlux**](#function-fillrightflux) ([**Param**](classParam.md) XParam, bool doProlongation, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillRightFlux**](#function-fillrightflux) (int halowidth, bool doProlongation, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, T \* a) <br> |
|  template void | [**fillRightFlux&lt; double &gt;**](#function-fillrightflux-double) ([**Param**](classParam.md) XParam, bool doProlongation, int ib, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \*& z) <br> |
|  template \_\_global\_\_ void | [**fillRightFlux&lt; double &gt;**](#function-fillrightflux-double) (int halowidth, bool doProlongation, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, double \* a) <br> |
|  template void | [**fillRightFlux&lt; float &gt;**](#function-fillrightflux-float) ([**Param**](classParam.md) XParam, bool doProlongation, int ib, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \*& z) <br> |
|  template \_\_global\_\_ void | [**fillRightFlux&lt; float &gt;**](#function-fillrightflux-float) (int halowidth, bool doProlongation, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, float \* a) <br> |
|  \_\_global\_\_ void | [**fillRightnew**](#function-fillrightnew) (int halowidth, int nblk, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, T \* a) <br> |
|  template \_\_global\_\_ void | [**fillRightnew&lt; double &gt;**](#function-fillrightnew-double) (int halowidth, int nblk, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillRightnew&lt; float &gt;**](#function-fillrightnew-float) (int halowidth, int nblk, int \* active, int \* level, int \* rightbot, int \* righttop, int \* leftbot, int \* botleft, int \* topleft, float \* a) <br> |
|  void | [**fillTop**](#function-filltop) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillTop**](#function-filltop) (int halowidth, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, T \* a) <br> |
|  template \_\_global\_\_ void | [**fillTop&lt; double &gt;**](#function-filltop-double) (int halowidth, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillTop&lt; float &gt;**](#function-filltop-float) (int halowidth, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, float \* a) <br> |
|  void | [**fillTopFlux**](#function-filltopflux) ([**Param**](classParam.md) XParam, bool doProlongation, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& z) <br> |
|  \_\_global\_\_ void | [**fillTopFlux**](#function-filltopflux) (int halowidth, bool doProlongation, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, T \* a) <br> |
|  template void | [**fillTopFlux&lt; double &gt;**](#function-filltopflux-double) ([**Param**](classParam.md) XParam, bool doProlongation, int ib, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \*& z) <br> |
|  template \_\_global\_\_ void | [**fillTopFlux&lt; double &gt;**](#function-filltopflux-double) (int halowidth, bool doProlongation, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, double \* a) <br> |
|  template void | [**fillTopFlux&lt; float &gt;**](#function-filltopflux-float) ([**Param**](classParam.md) XParam, bool doProlongation, int ib, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \*& z) <br> |
|  template \_\_global\_\_ void | [**fillTopFlux&lt; float &gt;**](#function-filltopflux-float) (int halowidth, bool doProlongation, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, float \* a) <br> |
|  \_\_global\_\_ void | [**fillTopnew**](#function-filltopnew) (int halowidth, int nblk, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, T \* a) <br> |
|  template \_\_global\_\_ void | [**fillTopnew&lt; double &gt;**](#function-filltopnew-double) (int halowidth, int nblk, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, double \* a) <br> |
|  template \_\_global\_\_ void | [**fillTopnew&lt; float &gt;**](#function-filltopnew-float) (int halowidth, int nblk, int \* active, int \* level, int \* topleft, int \* topright, int \* botleft, int \* leftbot, int \* rightbot, float \* a) <br> |
|  void | [**refine\_linear**](#function-refine_linear) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  template void | [**refine\_linear&lt; double &gt;**](#function-refine_linear-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template void | [**refine\_linear&lt; float &gt;**](#function-refine_linear-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  void | [**refine\_linearGPU**](#function-refine_lineargpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  template void | [**refine\_linearGPU&lt; double &gt;**](#function-refine_lineargpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template void | [**refine\_linearGPU&lt; float &gt;**](#function-refine_lineargpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  void | [**refine\_linear\_Bot**](#function-refine_linear_bot) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  template void | [**refine\_linear\_Bot&lt; double &gt;**](#function-refine_linear_bot-double) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template void | [**refine\_linear\_Bot&lt; float &gt;**](#function-refine_linear_bot-float) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  \_\_global\_\_ void | [**refine\_linear\_BotGPU**](#function-refine_linear_botgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  template \_\_global\_\_ void | [**refine\_linear\_BotGPU&lt; double &gt;**](#function-refine_linear_botgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template \_\_global\_\_ void | [**refine\_linear\_BotGPU&lt; float &gt;**](#function-refine_linear_botgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  void | [**refine\_linear\_Left**](#function-refine_linear_left) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  template void | [**refine\_linear\_Left&lt; double &gt;**](#function-refine_linear_left-double) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template void | [**refine\_linear\_Left&lt; float &gt;**](#function-refine_linear_left-float) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  \_\_global\_\_ void | [**refine\_linear\_LeftGPU**](#function-refine_linear_leftgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  template \_\_global\_\_ void | [**refine\_linear\_LeftGPU&lt; double &gt;**](#function-refine_linear_leftgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template \_\_global\_\_ void | [**refine\_linear\_LeftGPU&lt; float &gt;**](#function-refine_linear_leftgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  void | [**refine\_linear\_Right**](#function-refine_linear_right) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  template void | [**refine\_linear\_Right&lt; double &gt;**](#function-refine_linear_right-double) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template void | [**refine\_linear\_Right&lt; float &gt;**](#function-refine_linear_right-float) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  \_\_global\_\_ void | [**refine\_linear\_RightGPU**](#function-refine_linear_rightgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  template \_\_global\_\_ void | [**refine\_linear\_RightGPU&lt; double &gt;**](#function-refine_linear_rightgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template \_\_global\_\_ void | [**refine\_linear\_RightGPU&lt; float &gt;**](#function-refine_linear_rightgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  void | [**refine\_linear\_Top**](#function-refine_linear_top) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  template void | [**refine\_linear\_Top&lt; double &gt;**](#function-refine_linear_top-double) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template void | [**refine\_linear\_Top&lt; float &gt;**](#function-refine_linear_top-float) ([**Param**](classParam.md) XParam, int ib, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |
|  \_\_global\_\_ void | [**refine\_linear\_TopGPU**](#function-refine_linear_topgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z, T \* dzdx, T \* dzdy) <br> |
|  template \_\_global\_\_ void | [**refine\_linear\_TopGPU&lt; double &gt;**](#function-refine_linear_topgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z, double \* dzdx, double \* dzdy) <br> |
|  template \_\_global\_\_ void | [**refine\_linear\_TopGPU&lt; float &gt;**](#function-refine_linear_topgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z, float \* dzdx, float \* dzdy) <br> |




























## Public Functions Documentation




### function HaloFluxCPUBT 

```C++
template<class T>
void HaloFluxCPUBT (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T * z
) 
```




<hr>



### function HaloFluxCPULR 

```C++
template<class T>
void HaloFluxCPULR (
    Param XParam,
    int ib,
    BlockP < T > XBlock,
    T * z
) 
```




<hr>



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

```C++
template<class T>
__global__ void fillCornersGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```




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

```C++
template<class T>
void fillHaloBTFluxC (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```




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

```C++
template<class T>
void fillHaloBotTopGPU (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```




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

```C++
template<class T>
void fillHaloBotTopGPUnew (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```




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



### Description



This function is a wraping fuction of the halo functions on CPU. It is called from the main Halo CPU function. This is layer 2 of 3 wrap so the candy doesn't stick too much. 



        

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



### Description



This fuction is a wraping fuction of the halo functions for CPU. It is called from another wraping function to keep things clean. In a sense this is the third (and last) layer of wrapping 



        

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



### Depreciated



This function is was never sucessful and will never be used. It is fundamentally flawed because is doesn't preserve the balance of fluxes on the restiction interface It should be deleted soon 



### Description




        

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

```C++
template<class T>
void fillHaloLRFluxC (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```




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

```C++
template<class T>
void fillHaloLeftRightGPU (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```




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

```C++
template<class T>
void fillHaloLeftRightGPUnew (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```




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

```C++
template<class T>
void fillHaloTopRightGPU (
    Param XParam,
    BlockP < T > XBlock,
    cudaStream_t stream,
    T * z
) 
```




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


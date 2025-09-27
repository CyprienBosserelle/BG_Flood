

# File Boundary.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Boundary.cu**](Boundary_8cu.md)

[Go to the source code of this file](Boundary_8cu_source.md)



* `#include "Boundary.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_device\_\_ \_\_host\_\_ void | [**ABS1D**](#function-abs1d) (T g, T sign, T zsbnd, T zsinside, T hinside, T utbnd, T unbnd, T & un, T & ut, T & zs, T & h) <br> |
|  \_\_device\_\_ \_\_host\_\_ void | [**ABS1DQ**](#function-abs1dq) (T g, T sign, T factime, T facrel, T zs, T zsbnd, T zsinside, T h, T & qmean, T & q, T & G, T & S) <br> |
|  \_\_device\_\_ \_\_host\_\_ void | [**Dirichlet1D**](#function-dirichlet1d) (T g, T sign, T zsbnd, T zsinside, T hinside, T uninside, T & un, T & ut, T & zs, T & h) <br> |
|  \_\_device\_\_ \_\_host\_\_ void | [**Dirichlet1Q**](#function-dirichlet1q) (T g, T sign, T zsbnd, T zsinside, T hinside, T uninside, T & q) <br> |
|  void | [**Flowbnd**](#function-flowbnd) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**bndparam**](classbndparam.md) side, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv) <br> |
|  template void | [**Flowbnd&lt; double &gt;**](#function-flowbnd-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; & XLoop, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**bndparam**](classbndparam.md) side, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv) <br> |
|  template void | [**Flowbnd&lt; float &gt;**](#function-flowbnd-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; & XLoop, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**bndparam**](classbndparam.md) side, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv) <br> |
|  void | [**FlowbndFlux**](#function-flowbndflux) ([**Param**](classParam.md) XParam, double totaltime, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**bndsegment**](classbndsegment.md) bndseg, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux) <br> |
|  template void | [**FlowbndFlux&lt; double &gt;**](#function-flowbndflux-double) ([**Param**](classParam.md) XParam, double totaltime, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**bndsegment**](classbndsegment.md) bndseg, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux) <br> |
|  template void | [**FlowbndFlux&lt; float &gt;**](#function-flowbndflux-float) ([**Param**](classParam.md) XParam, double totaltime, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**bndsegment**](classbndsegment.md) bndseg, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux) <br> |
|  void | [**FlowbndFluxML**](#function-flowbndfluxml) ([**Param**](classParam.md) XParam, double totaltime, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**bndsegment**](classbndsegment.md) bndseg, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**FluxMLP**](structFluxMLP.md)&lt; T &gt; XFlux) <br> |
|  template void | [**FlowbndFluxML&lt; double &gt;**](#function-flowbndfluxml-double) ([**Param**](classParam.md) XParam, double totaltime, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**bndsegment**](classbndsegment.md) bndseg, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**FluxMLP**](structFluxMLP.md)&lt; double &gt; XFlux) <br> |
|  template void | [**FlowbndFluxML&lt; float &gt;**](#function-flowbndfluxml-float) ([**Param**](classParam.md) XParam, double totaltime, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**bndsegment**](classbndsegment.md) bndseg, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**FluxMLP**](structFluxMLP.md)&lt; float &gt; XFlux) <br> |
|  void | [**FlowbndFluxold**](#function-flowbndfluxold) ([**Param**](classParam.md) XParam, double totaltime, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**bndparam**](classbndparam.md) side, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux) <br> |
|  template void | [**FlowbndFluxold&lt; double &gt;**](#function-flowbndfluxold-double) ([**Param**](classParam.md) XParam, double totaltime, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**bndparam**](classbndparam.md) side, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEv, [**FluxP**](structFluxP.md)&lt; double &gt; XFlux) <br> |
|  template void | [**FlowbndFluxold&lt; float &gt;**](#function-flowbndfluxold-float) ([**Param**](classParam.md) XParam, double totaltime, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**bndparam**](classbndparam.md) side, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEv, [**FluxP**](structFluxP.md)&lt; float &gt; XFlux) <br> |
|  \_\_host\_\_ \_\_device\_\_ int | [**Inside**](#function-inside) (int halowidth, int blkmemwidth, int isright, int istop, int ix, int iy, int ib) <br> |
|  \_\_host\_\_ void | [**bndCPU**](#function-bndcpu) ([**Param**](classParam.md) XParam, [**bndparam**](classbndparam.md) side, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, std::vector&lt; double &gt; zsbndvec, std::vector&lt; double &gt; uubndvec, std::vector&lt; double &gt; vvbndvec, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, T \* zs, T \* h, T \* un, T \* ut) <br> |
|  template \_\_host\_\_ void | [**bndCPU&lt; double &gt;**](#function-bndcpu-double) ([**Param**](classParam.md) XParam, [**bndparam**](classbndparam.md) side, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, std::vector&lt; double &gt; zsbndvec, std::vector&lt; double &gt; uubndvec, std::vector&lt; double &gt; vvbndvec, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, double \* zs, double \* h, double \* un, double \* ut) <br> |
|  template \_\_host\_\_ void | [**bndCPU&lt; float &gt;**](#function-bndcpu-float) ([**Param**](classParam.md) XParam, [**bndparam**](classbndparam.md) side, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, std::vector&lt; double &gt; zsbndvec, std::vector&lt; double &gt; uubndvec, std::vector&lt; double &gt; vvbndvec, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, float \* zs, float \* h, float \* un, float \* ut) <br> |
|  \_\_global\_\_ void | [**bndFluxGPUSide**](#function-bndfluxgpuside) ([**Param**](classParam.md) XParam, [**bndsegmentside**](classbndsegmentside.md) side, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Zsmap, bool uniform, int type, float zsbnd, T taper, T \* zs, T \* h, T \* un, T \* ut, T \* Fh, T \* Fq, T \* Ss) <br> |
|  void | [**bndFluxGPUSideCPU**](#function-bndfluxgpusidecpu) ([**Param**](classParam.md) XParam, [**bndsegmentside**](classbndsegmentside.md) side, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Zsmap, bool uniform, int type, float zsbnd, T taper, T \* zs, T \* h, T \* un, T \* ut, T \* Fh, T \* Fq, T \* Ss) <br> |
|  \_\_global\_\_ void | [**bndGPU**](#function-bndgpu) ([**Param**](classParam.md) XParam, [**bndparam**](classbndparam.md) side, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, float itime, T \* zs, T \* h, T \* un, T \* ut) <br> |
|  template \_\_global\_\_ void | [**bndGPU&lt; double &gt;**](#function-bndgpu-double) ([**Param**](classParam.md) XParam, [**bndparam**](classbndparam.md) side, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, float itime, double \* zs, double \* h, double \* un, double \* ut) <br> |
|  template \_\_global\_\_ void | [**bndGPU&lt; float &gt;**](#function-bndgpu-float) ([**Param**](classParam.md) XParam, [**bndparam**](classbndparam.md) side, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, float itime, float \* zs, float \* h, float \* un, float \* ut) <br> |
|  \_\_device\_\_ \_\_host\_\_ void | [**findmaskside**](#function-findmaskside) (int side, bool & isleftbot, bool & islefttop, bool & istopleft, bool & istopright, bool & isrighttop, bool & isrightbot, bool & isbotright, bool & isbotleft) <br> |
|  \_\_device\_\_ \_\_host\_\_ void | [**halowall**](#function-halowall) (T zsinside, T & un, T & ut, T & zs, T & h, T & zb) <br> |
|  \_\_host\_\_ \_\_device\_\_ bool | [**isbnd**](#function-isbnd) (int isright, int istop, int blkwidth, int ix, int iy) <br> |
|  \_\_host\_\_ void | [**maskbnd**](#function-maskbnd) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  template \_\_host\_\_ void | [**maskbnd&lt; double &gt;**](#function-maskbnd-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template \_\_host\_\_ void | [**maskbnd&lt; float &gt;**](#function-maskbnd-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev, float \* zb) <br> |
|  \_\_global\_\_ void | [**maskbndGPUFluxbot**](#function-maskbndgpufluxbot) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUFluxbot&lt; double &gt;**](#function-maskbndgpufluxbot-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**FluxP**](structFluxP.md)&lt; double &gt; Flux) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUFluxbot&lt; float &gt;**](#function-maskbndgpufluxbot-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**FluxP**](structFluxP.md)&lt; float &gt; Flux) <br> |
|  \_\_global\_\_ void | [**maskbndGPUFluxleft**](#function-maskbndgpufluxleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUFluxleft&lt; double &gt;**](#function-maskbndgpufluxleft-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev, [**FluxP**](structFluxP.md)&lt; double &gt; Flux) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUFluxleft&lt; float &gt;**](#function-maskbndgpufluxleft-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev, [**FluxP**](structFluxP.md)&lt; float &gt; Flux) <br> |
|  \_\_global\_\_ void | [**maskbndGPUFluxright**](#function-maskbndgpufluxright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUFluxright&lt; double &gt;**](#function-maskbndgpufluxright-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**FluxP**](structFluxP.md)&lt; double &gt; Flux) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUFluxright&lt; float &gt;**](#function-maskbndgpufluxright-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**FluxP**](structFluxP.md)&lt; float &gt; Flux) <br> |
|  \_\_global\_\_ void | [**maskbndGPUFluxtop**](#function-maskbndgpufluxtop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUFluxtop&lt; double &gt;**](#function-maskbndgpufluxtop-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**FluxP**](structFluxP.md)&lt; double &gt; Flux) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUFluxtop&lt; float &gt;**](#function-maskbndgpufluxtop-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**FluxP**](structFluxP.md)&lt; float &gt; Flux) <br> |
|  \_\_global\_\_ void | [**maskbndGPUbot**](#function-maskbndgpubot) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUbot&lt; double &gt;**](#function-maskbndgpubot-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUbot&lt; float &gt;**](#function-maskbndgpubot-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev, float \* zb) <br> |
|  \_\_global\_\_ void | [**maskbndGPUleft**](#function-maskbndgpuleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUleft&lt; double &gt;**](#function-maskbndgpuleft-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUleft&lt; float &gt;**](#function-maskbndgpuleft-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev, float \* zb) <br> |
|  \_\_global\_\_ void | [**maskbndGPUright**](#function-maskbndgpuright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUright&lt; double &gt;**](#function-maskbndgpuright-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUright&lt; float &gt;**](#function-maskbndgpuright-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev, float \* zb) <br> |
|  \_\_global\_\_ void | [**maskbndGPUtop**](#function-maskbndgputop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUtop&lt; double &gt;**](#function-maskbndgputop-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; Xev, double \* zb) <br> |
|  template \_\_global\_\_ void | [**maskbndGPUtop&lt; float &gt;**](#function-maskbndgputop-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; Xev, float \* zb) <br> |
|  \_\_device\_\_ \_\_host\_\_ void | [**noslipbnd**](#function-noslipbnd) (T zsinside, T hinside, T & un, T & ut, T & zs, T & h) <br> |
|  \_\_device\_\_ \_\_host\_\_ void | [**noslipbndQ**](#function-noslipbndq) (T & F, T & G, T & S) <br> |




























## Public Functions Documentation




### function ABS1D 

```C++
template<class T>
__device__ __host__ void ABS1D (
    T g,
    T sign,
    T zsbnd,
    T zsinside,
    T hinside,
    T utbnd,
    T unbnd,
    T & un,
    T & ut,
    T & zs,
    T & h
) 
```




<hr>



### function ABS1DQ 

```C++
template<class T>
__device__ __host__ void ABS1DQ (
    T g,
    T sign,
    T factime,
    T facrel,
    T zs,
    T zsbnd,
    T zsinside,
    T h,
    T & qmean,
    T & q,
    T & G,
    T & S
) 
```




<hr>



### function Dirichlet1D 

```C++
template<class T>
__device__ __host__ void Dirichlet1D (
    T g,
    T sign,
    T zsbnd,
    T zsinside,
    T hinside,
    T uninside,
    T & un,
    T & ut,
    T & zs,
    T & h
) 
```




<hr>



### function Dirichlet1Q 

```C++
template<class T>
__device__ __host__ void Dirichlet1Q (
    T g,
    T sign,
    T zsbnd,
    T zsinside,
    T hinside,
    T uninside,
    T & q
) 
```




<hr>



### function Flowbnd 

```C++
template<class T>
void Flowbnd (
    Param XParam,
    Loop < T > & XLoop,
    BlockP < T > XBlock,
    bndparam side,
    DynForcingP < float > Atmp,
    EvolvingP < T > XEv
) 
```




<hr>



### function Flowbnd&lt; double &gt; 

```C++
template void Flowbnd< double > (
    Param XParam,
    Loop < double > & XLoop,
    BlockP < double > XBlock,
    bndparam side,
    DynForcingP < float > Atmp,
    EvolvingP < double > XEv
) 
```




<hr>



### function Flowbnd&lt; float &gt; 

```C++
template void Flowbnd< float > (
    Param XParam,
    Loop < float > & XLoop,
    BlockP < float > XBlock,
    bndparam side,
    DynForcingP < float > Atmp,
    EvolvingP < float > XEv
) 
```




<hr>



### function FlowbndFlux 

```C++
template<class T>
void FlowbndFlux (
    Param XParam,
    double totaltime,
    BlockP < T > XBlock,
    bndsegment bndseg,
    DynForcingP < float > Atmp,
    EvolvingP < T > XEv,
    FluxP < T > XFlux
) 
```




<hr>



### function FlowbndFlux&lt; double &gt; 

```C++
template void FlowbndFlux< double > (
    Param XParam,
    double totaltime,
    BlockP < double > XBlock,
    bndsegment bndseg,
    DynForcingP < float > Atmp,
    EvolvingP < double > XEv,
    FluxP < double > XFlux
) 
```




<hr>



### function FlowbndFlux&lt; float &gt; 

```C++
template void FlowbndFlux< float > (
    Param XParam,
    double totaltime,
    BlockP < float > XBlock,
    bndsegment bndseg,
    DynForcingP < float > Atmp,
    EvolvingP < float > XEv,
    FluxP < float > XFlux
) 
```




<hr>



### function FlowbndFluxML 

```C++
template<class T>
void FlowbndFluxML (
    Param XParam,
    double totaltime,
    BlockP < T > XBlock,
    bndsegment bndseg,
    DynForcingP < float > Atmp,
    EvolvingP < T > XEv,
    FluxMLP < T > XFlux
) 
```




<hr>



### function FlowbndFluxML&lt; double &gt; 

```C++
template void FlowbndFluxML< double > (
    Param XParam,
    double totaltime,
    BlockP < double > XBlock,
    bndsegment bndseg,
    DynForcingP < float > Atmp,
    EvolvingP < double > XEv,
    FluxMLP < double > XFlux
) 
```




<hr>



### function FlowbndFluxML&lt; float &gt; 

```C++
template void FlowbndFluxML< float > (
    Param XParam,
    double totaltime,
    BlockP < float > XBlock,
    bndsegment bndseg,
    DynForcingP < float > Atmp,
    EvolvingP < float > XEv,
    FluxMLP < float > XFlux
) 
```




<hr>



### function FlowbndFluxold 

```C++
template<class T>
void FlowbndFluxold (
    Param XParam,
    double totaltime,
    BlockP < T > XBlock,
    bndparam side,
    DynForcingP < float > Atmp,
    EvolvingP < T > XEv,
    FluxP < T > XFlux
) 
```




<hr>



### function FlowbndFluxold&lt; double &gt; 

```C++
template void FlowbndFluxold< double > (
    Param XParam,
    double totaltime,
    BlockP < double > XBlock,
    bndparam side,
    DynForcingP < float > Atmp,
    EvolvingP < double > XEv,
    FluxP < double > XFlux
) 
```




<hr>



### function FlowbndFluxold&lt; float &gt; 

```C++
template void FlowbndFluxold< float > (
    Param XParam,
    double totaltime,
    BlockP < float > XBlock,
    bndparam side,
    DynForcingP < float > Atmp,
    EvolvingP < float > XEv,
    FluxP < float > XFlux
) 
```




<hr>



### function Inside 

```C++
__host__ __device__ int Inside (
    int halowidth,
    int blkmemwidth,
    int isright,
    int istop,
    int ix,
    int iy,
    int ib
) 
```




<hr>



### function bndCPU 

```C++
template<class T>
__host__ void bndCPU (
    Param XParam,
    bndparam side,
    BlockP < T > XBlock,
    std::vector< double > zsbndvec,
    std::vector< double > uubndvec,
    std::vector< double > vvbndvec,
    DynForcingP < float > Atmp,
    T * zs,
    T * h,
    T * un,
    T * ut
) 
```




<hr>



### function bndCPU&lt; double &gt; 

```C++
template __host__ void bndCPU< double > (
    Param XParam,
    bndparam side,
    BlockP < double > XBlock,
    std::vector< double > zsbndvec,
    std::vector< double > uubndvec,
    std::vector< double > vvbndvec,
    DynForcingP < float > Atmp,
    double * zs,
    double * h,
    double * un,
    double * ut
) 
```




<hr>



### function bndCPU&lt; float &gt; 

```C++
template __host__ void bndCPU< float > (
    Param XParam,
    bndparam side,
    BlockP < float > XBlock,
    std::vector< double > zsbndvec,
    std::vector< double > uubndvec,
    std::vector< double > vvbndvec,
    DynForcingP < float > Atmp,
    float * zs,
    float * h,
    float * un,
    float * ut
) 
```




<hr>



### function bndFluxGPUSide 

```C++
template<class T>
__global__ void bndFluxGPUSide (
    Param XParam,
    bndsegmentside side,
    BlockP < T > XBlock,
    DynForcingP < float > Atmp,
    DynForcingP < float > Zsmap,
    bool uniform,
    int type,
    float zsbnd,
    T taper,
    T * zs,
    T * h,
    T * un,
    T * ut,
    T * Fh,
    T * Fq,
    T * Ss
) 
```




<hr>



### function bndFluxGPUSideCPU 

```C++
template<class T>
void bndFluxGPUSideCPU (
    Param XParam,
    bndsegmentside side,
    BlockP < T > XBlock,
    DynForcingP < float > Atmp,
    DynForcingP < float > Zsmap,
    bool uniform,
    int type,
    float zsbnd,
    T taper,
    T * zs,
    T * h,
    T * un,
    T * ut,
    T * Fh,
    T * Fq,
    T * Ss
) 
```




<hr>



### function bndGPU 

```C++
template<class T>
__global__ void bndGPU (
    Param XParam,
    bndparam side,
    BlockP < T > XBlock,
    DynForcingP < float > Atmp,
    float itime,
    T * zs,
    T * h,
    T * un,
    T * ut
) 
```




<hr>



### function bndGPU&lt; double &gt; 

```C++
template __global__ void bndGPU< double > (
    Param XParam,
    bndparam side,
    BlockP < double > XBlock,
    DynForcingP < float > Atmp,
    float itime,
    double * zs,
    double * h,
    double * un,
    double * ut
) 
```




<hr>



### function bndGPU&lt; float &gt; 

```C++
template __global__ void bndGPU< float > (
    Param XParam,
    bndparam side,
    BlockP < float > XBlock,
    DynForcingP < float > Atmp,
    float itime,
    float * zs,
    float * h,
    float * un,
    float * ut
) 
```




<hr>



### function findmaskside 

```C++
__device__ __host__ void findmaskside (
    int side,
    bool & isleftbot,
    bool & islefttop,
    bool & istopleft,
    bool & istopright,
    bool & isrighttop,
    bool & isrightbot,
    bool & isbotright,
    bool & isbotleft
) 
```




<hr>



### function halowall 

```C++
template<class T>
__device__ __host__ void halowall (
    T zsinside,
    T & un,
    T & ut,
    T & zs,
    T & h,
    T & zb
) 
```




<hr>



### function isbnd 

```C++
__host__ __device__ bool isbnd (
    int isright,
    int istop,
    int blkwidth,
    int ix,
    int iy
) 
```




<hr>



### function maskbnd 

```C++
template<class T>
__host__ void maskbnd (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```




<hr>



### function maskbnd&lt; double &gt; 

```C++
template __host__ void maskbnd< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev,
    double * zb
) 
```




<hr>



### function maskbnd&lt; float &gt; 

```C++
template __host__ void maskbnd< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev,
    float * zb
) 
```




<hr>



### function maskbndGPUFluxbot 

```C++
template<class T>
__global__ void maskbndGPUFluxbot (
    Param XParam,
    BlockP < T > XBlock,
    FluxP < T > Flux
) 
```




<hr>



### function maskbndGPUFluxbot&lt; double &gt; 

```C++
template __global__ void maskbndGPUFluxbot< double > (
    Param XParam,
    BlockP < double > XBlock,
    FluxP < double > Flux
) 
```




<hr>



### function maskbndGPUFluxbot&lt; float &gt; 

```C++
template __global__ void maskbndGPUFluxbot< float > (
    Param XParam,
    BlockP < float > XBlock,
    FluxP < float > Flux
) 
```




<hr>



### function maskbndGPUFluxleft 

```C++
template<class T>
__global__ void maskbndGPUFluxleft (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    FluxP < T > Flux
) 
```




<hr>



### function maskbndGPUFluxleft&lt; double &gt; 

```C++
template __global__ void maskbndGPUFluxleft< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev,
    FluxP < double > Flux
) 
```




<hr>



### function maskbndGPUFluxleft&lt; float &gt; 

```C++
template __global__ void maskbndGPUFluxleft< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev,
    FluxP < float > Flux
) 
```




<hr>



### function maskbndGPUFluxright 

```C++
template<class T>
__global__ void maskbndGPUFluxright (
    Param XParam,
    BlockP < T > XBlock,
    FluxP < T > Flux
) 
```




<hr>



### function maskbndGPUFluxright&lt; double &gt; 

```C++
template __global__ void maskbndGPUFluxright< double > (
    Param XParam,
    BlockP < double > XBlock,
    FluxP < double > Flux
) 
```




<hr>



### function maskbndGPUFluxright&lt; float &gt; 

```C++
template __global__ void maskbndGPUFluxright< float > (
    Param XParam,
    BlockP < float > XBlock,
    FluxP < float > Flux
) 
```




<hr>



### function maskbndGPUFluxtop 

```C++
template<class T>
__global__ void maskbndGPUFluxtop (
    Param XParam,
    BlockP < T > XBlock,
    FluxP < T > Flux
) 
```




<hr>



### function maskbndGPUFluxtop&lt; double &gt; 

```C++
template __global__ void maskbndGPUFluxtop< double > (
    Param XParam,
    BlockP < double > XBlock,
    FluxP < double > Flux
) 
```




<hr>



### function maskbndGPUFluxtop&lt; float &gt; 

```C++
template __global__ void maskbndGPUFluxtop< float > (
    Param XParam,
    BlockP < float > XBlock,
    FluxP < float > Flux
) 
```




<hr>



### function maskbndGPUbot 

```C++
template<class T>
__global__ void maskbndGPUbot (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```




<hr>



### function maskbndGPUbot&lt; double &gt; 

```C++
template __global__ void maskbndGPUbot< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev,
    double * zb
) 
```




<hr>



### function maskbndGPUbot&lt; float &gt; 

```C++
template __global__ void maskbndGPUbot< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev,
    float * zb
) 
```




<hr>



### function maskbndGPUleft 

```C++
template<class T>
__global__ void maskbndGPUleft (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```




<hr>



### function maskbndGPUleft&lt; double &gt; 

```C++
template __global__ void maskbndGPUleft< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev,
    double * zb
) 
```




<hr>



### function maskbndGPUleft&lt; float &gt; 

```C++
template __global__ void maskbndGPUleft< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev,
    float * zb
) 
```




<hr>



### function maskbndGPUright 

```C++
template<class T>
__global__ void maskbndGPUright (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```




<hr>



### function maskbndGPUright&lt; double &gt; 

```C++
template __global__ void maskbndGPUright< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev,
    double * zb
) 
```




<hr>



### function maskbndGPUright&lt; float &gt; 

```C++
template __global__ void maskbndGPUright< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev,
    float * zb
) 
```




<hr>



### function maskbndGPUtop 

```C++
template<class T>
__global__ void maskbndGPUtop (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```




<hr>



### function maskbndGPUtop&lt; double &gt; 

```C++
template __global__ void maskbndGPUtop< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > Xev,
    double * zb
) 
```




<hr>



### function maskbndGPUtop&lt; float &gt; 

```C++
template __global__ void maskbndGPUtop< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > Xev,
    float * zb
) 
```




<hr>



### function noslipbnd 

```C++
template<class T>
__device__ __host__ void noslipbnd (
    T zsinside,
    T hinside,
    T & un,
    T & ut,
    T & zs,
    T & h
) 
```




<hr>



### function noslipbndQ 

```C++
template<class T>
__device__ __host__ void noslipbndQ (
    T & F,
    T & G,
    T & S
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Boundary.cu`


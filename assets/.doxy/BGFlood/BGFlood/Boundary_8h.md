

# File Boundary.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Boundary.h**](Boundary_8h.md)

[Go to the source code of this file](Boundary_8h_source.md)



* `#include "General.h"`
* `#include "MemManagement.h"`
* `#include "Util_CPU.h"`
* `#include "Updateforcing.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_device\_\_ \_\_host\_\_ void | [**ABS1D**](#function-abs1d) (T g, T sign, T zsbnd, T zsinside, T hinside, T utbnd, T unbnd, T & un, T & ut, T & zs, T & h) <br> |
|  \_\_device\_\_ \_\_host\_\_ void | [**ABS1DQ**](#function-abs1dq) (T g, T sign, T factime, T facrel, T zs, T zsbnd, T zsinside, T h, T & qmean, T & q, T & G, T & S) <br> |
|  \_\_device\_\_ \_\_host\_\_ void | [**Dirichlet1D**](#function-dirichlet1d) (T g, T sign, T zsbnd, T zsinside, T hinside, T uninside, T & un, T & ut, T & zs, T & h) <br> |
|  \_\_device\_\_ \_\_host\_\_ void | [**Dirichlet1Q**](#function-dirichlet1q) (T g, T sign, T zsbnd, T zsinside, T hinside, T uninside, T & q) <br> |
|  void | [**Flowbnd**](#function-flowbnd) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**bndparam**](classbndparam.md) side, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv) <br> |
|  void | [**FlowbndFlux**](#function-flowbndflux) ([**Param**](classParam.md) XParam, double totaltime, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**bndsegment**](classbndsegment.md) bndseg, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux) <br> |
|  void | [**FlowbndFlux**](#function-flowbndflux) ([**Param**](classParam.md) XParam, double totaltime, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**bndparam**](classbndparam.md) side, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux) <br> |
|  void | [**FlowbndFluxML**](#function-flowbndfluxml) ([**Param**](classParam.md) XParam, double totaltime, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**bndsegment**](classbndsegment.md) bndseg, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**FluxMLP**](structFluxMLP.md)&lt; T &gt; XFlux) <br> |
|  \_\_host\_\_ \_\_device\_\_ int | [**Inside**](#function-inside) (int halowidth, int blkmemwidth, int isright, int istop, int ix, int iy, int ib) <br> |
|  \_\_host\_\_ void | [**bndCPU**](#function-bndcpu) ([**Param**](classParam.md) XParam, [**bndparam**](classbndparam.md) side, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, std::vector&lt; double &gt; zsbndvec, std::vector&lt; double &gt; uubndvec, std::vector&lt; double &gt; vvbndvec, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, T \* zs, T \* h, T \* un, T \* ut) <br> |
|  \_\_global\_\_ void | [**bndFluxGPUSide**](#function-bndfluxgpuside) ([**Param**](classParam.md) XParam, [**bndsegmentside**](classbndsegmentside.md) side, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Zsmap, bool uniform, int type, float zsbnd, T taper, T \* zs, T \* h, T \* un, T \* ut, T \* Fh, T \* Fq, T \* Ss) <br> |
|  \_\_global\_\_ void | [**bndGPU**](#function-bndgpu) ([**Param**](classParam.md) XParam, [**bndparam**](classbndparam.md) side, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; Atmp, float itime, T \* zs, T \* h, T \* un, T \* ut) <br> |
|  \_\_device\_\_ \_\_host\_\_ void | [**findmaskside**](#function-findmaskside) (int side, bool & isleftbot, bool & islefttop, bool & istopleft, bool & istopright, bool & isrighttop, bool & isrightbot, bool & isbotright, bool & isbotleft) <br> |
|  \_\_device\_\_ \_\_host\_\_ void | [**halowall**](#function-halowall) (T zsinside, T & un, T & ut, T & zs, T & h, T & zb) <br> |
|  \_\_host\_\_ \_\_device\_\_ bool | [**isbnd**](#function-isbnd) (int isright, int istop, int blkwidth, int ix, int iy) <br> |
|  \_\_host\_\_ void | [**maskbnd**](#function-maskbnd) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  \_\_global\_\_ void | [**maskbndGPUFluxbot**](#function-maskbndgpufluxbot) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  \_\_global\_\_ void | [**maskbndGPUFluxleft**](#function-maskbndgpufluxleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  \_\_global\_\_ void | [**maskbndGPUFluxright**](#function-maskbndgpufluxright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  \_\_global\_\_ void | [**maskbndGPUFluxtop**](#function-maskbndgpufluxtop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**FluxP**](structFluxP.md)&lt; T &gt; Flux) <br> |
|  \_\_global\_\_ void | [**maskbndGPUbot**](#function-maskbndgpubot) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  \_\_global\_\_ void | [**maskbndGPUleft**](#function-maskbndgpuleft) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  \_\_global\_\_ void | [**maskbndGPUright**](#function-maskbndgpuright) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
|  \_\_global\_\_ void | [**maskbndGPUtop**](#function-maskbndgputop) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; Xev, T \* zb) <br> |
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



### function FlowbndFlux 

```C++
template<class T>
void FlowbndFlux (
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
The documentation for this class was generated from the following file `src/Boundary.h`


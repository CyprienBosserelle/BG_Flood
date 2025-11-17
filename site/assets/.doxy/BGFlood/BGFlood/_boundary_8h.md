

# File Boundary.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Boundary.h**](_boundary_8h.md)

[Go to the source code of this file](_boundary_8h_source.md)



* `#include "General.h"`
* `#include "MemManagement.h"`
* `#include "Util_CPU.h"`
* `#include "Updateforcing.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_device\_\_ \_\_host\_\_ void | [**ABS1D**](#function-abs1d) (T g, T sign, T zsbnd, T zsinside, T hinside, T utbnd, T unbnd, T & un, T & ut, T & zs, T & h) <br>_Device/host function for 1D absorbing boundary condition._  |
|  \_\_device\_\_ \_\_host\_\_ void | [**ABS1DQ**](#function-abs1dq) (T g, T sign, T factime, T facrel, T zs, T zsbnd, T zsinside, T h, T & qmean, T & q, T & G, T & S) <br>_Device/host function for 1D absorbing boundary condition for flux variables._  |
|  \_\_device\_\_ \_\_host\_\_ void | [**Dirichlet1D**](#function-dirichlet1d) (T g, T sign, T zsbnd, T zsinside, T hinside, T uninside, T & un, T & ut, T & zs, T & h) <br>_Device/host function for 1D Dirichlet boundary condition._  |
|  \_\_device\_\_ \_\_host\_\_ void | [**Dirichlet1Q**](#function-dirichlet1q) (T g, T sign, T zsbnd, T zsinside, T hinside, T uninside, T & q) <br>_Device/host function for 1D Dirichlet boundary condition for flux variables._  |
|  void | [**Flowbnd**](#function-flowbnd) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; & XLoop, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**bndparam**](classbndparam.md) side, [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; Atmp, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv) <br>_Applies boundary conditions for flow variables on a given side of the domain._  |
|  void | [**FlowbndFlux**](#function-flowbndflux) ([**Param**](class_param.md) XParam, double totaltime, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**bndsegment**](classbndsegment.md) bndseg, [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; Atmp, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux) <br>_Applies boundary conditions for flux variables on a given segment of the domain._  |
|  void | [**FlowbndFlux**](#function-flowbndflux) ([**Param**](class_param.md) XParam, double totaltime, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**bndparam**](classbndparam.md) side, [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; Atmp, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux) <br> |
|  void | [**FlowbndFluxML**](#function-flowbndfluxml) ([**Param**](class_param.md) XParam, double totaltime, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**bndsegment**](classbndsegment.md) bndseg, [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; Atmp, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**FluxMLP**](struct_flux_m_l_p.md)&lt; T &gt; XFlux) <br>_Applies boundary conditions for flux ML variables on a given segment of the domain._  |
|  \_\_host\_\_ \_\_device\_\_ int | [**Inside**](#function-inside) (int halowidth, int blkmemwidth, int isright, int istop, int ix, int iy, int ib) <br>_Helper to compute the index of the inside cell for a boundary cell._  |
|  \_\_host\_\_ void | [**bndCPU**](#function-bndcpu) ([**Param**](class_param.md) XParam, [**bndparam**](classbndparam.md) side, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, std::vector&lt; double &gt; zsbndvec, std::vector&lt; double &gt; uubndvec, std::vector&lt; double &gt; vvbndvec, [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; Atmp, T \* zs, T \* h, T \* un, T \* ut) <br>_CPU implementation for applying boundary conditions on a side._  |
|  \_\_global\_\_ void | [**bndFluxGPUSide**](#function-bndfluxgpuside) ([**Param**](class_param.md) XParam, [**bndsegmentside**](classbndsegmentside.md) side, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; Atmp, [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; Zsmap, bool uniform, int type, float zsbnd, T taper, T \* zs, T \* h, T \* un, T \* ut, T \* Fh, T \* Fq, T \* Ss) <br>_CUDA kernel for applying boundary fluxes on a segment side (GPU version)._  |
|  \_\_global\_\_ void | [**bndGPU**](#function-bndgpu) ([**Param**](class_param.md) XParam, [**bndparam**](classbndparam.md) side, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; Atmp, float itime, T \* zs, T \* h, T \* un, T \* ut) <br>_CUDA kernel for applying boundary conditions on a side (GPU version)._  |
|  \_\_device\_\_ \_\_host\_\_ void | [**findmaskside**](#function-findmaskside) (int side, bool & isleftbot, bool & islefttop, bool & istopleft, bool & istopright, bool & isrighttop, bool & isrightbot, bool & isbotright, bool & isbotleft) <br>_Helper to decode mask side bitfield into booleans for each boundary/corner._  |
|  \_\_device\_\_ \_\_host\_\_ void | [**halowall**](#function-halowall) (T zsinside, T & un, T & ut, T & zs, T & h, T & zb) <br>_Device/host function to apply wall boundary in halo region._  |
|  \_\_host\_\_ \_\_device\_\_ bool | [**isbnd**](#function-isbnd) (int isright, int istop, int blkwidth, int ix, int iy) <br>_Helper to check if a cell is at the boundary._  |
|  \_\_host\_\_ void | [**maskbnd**](#function-maskbnd) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev, T \* zb) <br>_CPU implementation for applying masked blocks boundary conditions (halo walls)._  |
|  \_\_global\_\_ void | [**maskbndGPUFluxbot**](#function-maskbndgpufluxbot) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**FluxP**](struct_flux_p.md)&lt; T &gt; Flux) <br>_CUDA kernel for applying masked flux boundary conditions on the bottom side._  |
|  \_\_global\_\_ void | [**maskbndGPUFluxleft**](#function-maskbndgpufluxleft) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev, [**FluxP**](struct_flux_p.md)&lt; T &gt; Flux) <br>_CUDA kernel for applying masked flux boundary conditions on the left side._  |
|  \_\_global\_\_ void | [**maskbndGPUFluxright**](#function-maskbndgpufluxright) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**FluxP**](struct_flux_p.md)&lt; T &gt; Flux) <br>_CUDA kernel for applying masked flux boundary conditions on the right side._  |
|  \_\_global\_\_ void | [**maskbndGPUFluxtop**](#function-maskbndgpufluxtop) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**FluxP**](struct_flux_p.md)&lt; T &gt; Flux) <br>_CUDA kernel for applying masked flux boundary conditions on the top side._  |
|  \_\_global\_\_ void | [**maskbndGPUbot**](#function-maskbndgpubot) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev, T \* zb) <br>_CUDA kernel for applying masked boundary conditions (halo walls) on the bottom side._  |
|  \_\_global\_\_ void | [**maskbndGPUleft**](#function-maskbndgpuleft) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev, T \* zb) <br>_CUDA kernel for applying masked boundary conditions (halo walls) on the left side._  |
|  \_\_global\_\_ void | [**maskbndGPUright**](#function-maskbndgpuright) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev, T \* zb) <br>_CUDA kernel for applying masked boundary conditions (halo walls) on the right side._  |
|  \_\_global\_\_ void | [**maskbndGPUtop**](#function-maskbndgputop) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; Xev, T \* zb) <br>_CUDA kernel for applying masked boundary conditions (halo walls) on the top side._  |
|  \_\_device\_\_ \_\_host\_\_ void | [**noslipbnd**](#function-noslipbnd) (T zsinside, T hinside, T & un, T & ut, T & zs, T & h) <br>_Device/host function to apply no-slip boundary condition._  |
|  \_\_device\_\_ \_\_host\_\_ void | [**noslipbndQ**](#function-noslipbndq) (T & F, T & G, T & S) <br>_Device/host function to apply no-slip boundary for flux variables._  |




























## Public Functions Documentation




### function ABS1D 

_Device/host function for 1D absorbing boundary condition._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `g` Gravity 
* `sign` Side sign 
* `zsbnd` Boundary zs value 
* `zsinside` Inside zs value 
* `hinside` Inside h value 
* `utbnd` Tangential boundary velocity 
* `unbnd` Normal boundary velocity 
* `un` Normal velocity (output) 
* `ut` Tangential velocity (output) 
* `zs` zs value (output) 
* `h` h value (output)

Computes absorbing boundary for normal/tangential velocity and updates zs, h. 


        

<hr>



### function ABS1DQ 

_Device/host function for 1D absorbing boundary condition for flux variables._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `g` Gravity 
* `sign` Side sign 
* `factime` Filter time factor 
* `facrel` Relaxation time factor 
* `zs` zs value 
* `zsbnd` Boundary zs value 
* `zsinside` Inside zs value 
* `h` h value 
* `qmean` Mean flux (output) 
* `q` Flux q (output) 
* `G` Flux G (output) 
* `S` Source term (output)

Computes absorbing boundary for flux variables and updates qmean, q, G, S. 


        

<hr>



### function Dirichlet1D 

_Device/host function for 1D Dirichlet boundary condition._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `g` Gravity 
* `sign` Side sign 
* `zsbnd` Boundary zs value 
* `zsinside` Inside zs value 
* `hinside` Inside h value 
* `uninside` Inside normal velocity 
* `un` Normal velocity (output) 
* `ut` Tangential velocity (output) 
* `zs` zs value (output) 
* `h` h value (output)

Computes Dirichlet boundary for normal/tangential velocity and updates zs, h. 


        

<hr>



### function Dirichlet1Q 

_Device/host function for 1D Dirichlet boundary condition for flux variables._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `g` Gravity 
* `sign` Side sign 
* `zsbnd` Boundary zs value 
* `zsinside` Inside zs value 
* `hinside` Inside h value 
* `uninside` Inside normal velocity 
* `q` Flux q (output)

Computes Dirichlet boundary for flux variable q. 


        

<hr>



### function Flowbnd 

_Applies boundary conditions for flow variables on a given side of the domain._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XLoop` [**Loop**](struct_loop.md) control structure 
* `XBlock` Block data structure 
* `side` Boundary parameter (side info) 
* `Atmp` Dynamic forcing data 
* `XEv` Evolving variables

Handles boundary values for water level, velocity, and applies interpolation in time and space. Integrates any existing comments and logic. 


        

<hr>



### function FlowbndFlux 

_Applies boundary conditions for flux variables on a given segment of the domain._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `totaltime` Current simulation time 
* `XBlock` Block data structure 
* `bndseg` Boundary segment info 
* `Atmp` Dynamic forcing data 
* `XEv` Evolving variables 
* `XFlux` Flux variables

Handles boundary fluxes, applies tapers, and manages GPU/CPU execution for boundary segments. Integrates any existing comments and logic. 


        

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

_Applies boundary conditions for flux ML variables on a given segment of the domain._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `totaltime` Current simulation time 
* `XBlock` Block data structure 
* `bndseg` Boundary segment info 
* `Atmp` Dynamic forcing data 
* `XEv` Evolving variables 
* `XFlux` Machine learning flux variables

Handles boundary fluxes for ML variables, applies tapers, and manages GPU/CPU execution for boundary segments. Integrates any existing comments and logic. 


        

<hr>



### function Inside 

_Helper to compute the index of the inside cell for a boundary cell._ 
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





**Parameters:**


* `halowidth` Halo width 
* `blkmemwidth` Block memory width 
* `isright` Side info 
* `istop` Top info 
* `ix` x-index 
* `iy` y-index 
* `ib` Block index 



**Returns:**

Index of the inside cell 





        

<hr>



### function bndCPU 

_CPU implementation for applying boundary conditions on a side._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `side` Boundary parameter info 
* `XBlock` Block data structure 
* `zsbndvec` Vector of boundary zs values 
* `uubndvec` Vector of boundary u values 
* `vvbndvec` Vector of boundary v values 
* `Atmp` Dynamic forcing data 
* `zs` Array of zs values 
* `h` Array of h values 
* `un` Array of normal velocities 
* `ut` Array of tangential velocities

Applies boundary conditions for each block/thread on the CPU, using provided boundary vectors and dynamic forcing data. Handles no-slip, Dirichlet, ABS, and Neumann boundary types. Integrates any existing comments and logic. 


        

<hr>



### function bndFluxGPUSide 

_CUDA kernel for applying boundary fluxes on a segment side (GPU version)._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `side` Boundary segment side info 
* `XBlock` Block data structure 
* `Atmp` Dynamic forcing data 
* `Zsmap` Dynamic forcing data for zs 
* `uniform` Whether boundary is uniform 
* `type` Boundary type 
* `zsbnd` Boundary zs value 
* `taper` Taper value for smoothing 
* `zs` Array of zs values 
* `h` Array of h values 
* `un` Array of normal velocities 
* `ut` Array of tangential velocities 
* `Fh` Array for flux h 
* `Fq` Array for flux q 
* `Ss` Array for source terms

Applies boundary conditions and fluxes for each thread/block on the GPU, handling tapers, Dirichlet, and ABS boundary types. 


        

<hr>



### function bndGPU 

_CUDA kernel for applying boundary conditions on a side (GPU version)._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `side` Boundary parameter info 
* `XBlock` Block data structure 
* `Atmp` Dynamic forcing data 
* `itime` Interpolated time for boundary data 
* `zs` Array of zs values 
* `h` Array of h values 
* `un` Array of normal velocities 
* `ut` Array of tangential velocities

Applies boundary conditions for each thread/block on the GPU, using interpolated time and dynamic forcing data. 


        

<hr>



### function findmaskside 

_Helper to decode mask side bitfield into booleans for each boundary/corner._ 
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





**Parameters:**


* `side` Bitfield encoding mask sides 
* `isleftbot` Is left-bottom active 
* `islefttop` Is left-top active 
* `istopleft` Is top-left active 
* `istopright` Is top-right active 
* `isrighttop` Is right-top active 
* `isrightbot` Is right-bottom active 
* `isbotright` Is bottom-right active 
* `isbotleft` Is bottom-left active 




        

<hr>



### function halowall 

_Device/host function to apply wall boundary in halo region._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `zsinside` Inside zs value 
* `un` Normal velocity (output) 
* `ut` Tangential velocity (output) 
* `zs` zs value (output) 
* `h` h value (output) 
* `zb` Mask value (output)

Sets normal/tangential velocity and h to zero, copies zsinside to zs and zb. 


        

<hr>



### function isbnd 

_Helper to check if a cell is at the boundary._ 
```C++
__host__ __device__ bool isbnd (
    int isright,
    int istop,
    int blkwidth,
    int ix,
    int iy
) 
```





**Parameters:**


* `isright` Side info 
* `istop` Top info 
* `blkwidth` Block width 
* `ix` x-index 
* `iy` y-index 



**Returns:**

True if cell is at the boundary, false otherwise 





        

<hr>



### function maskbnd 

_CPU implementation for applying masked blocks boundary conditions (halo walls)._ 
```C++
template<class T>
__host__ void maskbnd (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `Xev` Evolving variables 
* `zb` Array of mask values

Applies wall boundary conditions in the halo region for masked blocks, updating velocities, zs, h, and mask values. Handles all four sides and corners. Integrates any existing comments and logic. 


        

<hr>



### function maskbndGPUFluxbot 

_CUDA kernel for applying masked flux boundary conditions on the bottom side._ 
```C++
template<class T>
__global__ void maskbndGPUFluxbot (
    Param XParam,
    BlockP < T > XBlock,
    FluxP < T > Flux
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `Flux` Flux variables

Applies flux boundary conditions in the halo region for masked blocks on the bottom side. 


        

<hr>



### function maskbndGPUFluxleft 

_CUDA kernel for applying masked flux boundary conditions on the left side._ 
```C++
template<class T>
__global__ void maskbndGPUFluxleft (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    FluxP < T > Flux
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `Xev` Evolving variables 
* `Flux` Flux variables

Applies flux boundary conditions in the halo region for masked blocks on the left side. 


        

<hr>



### function maskbndGPUFluxright 

_CUDA kernel for applying masked flux boundary conditions on the right side._ 
```C++
template<class T>
__global__ void maskbndGPUFluxright (
    Param XParam,
    BlockP < T > XBlock,
    FluxP < T > Flux
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `Flux` Flux variables

Applies flux boundary conditions in the halo region for masked blocks on the right side. 


        

<hr>



### function maskbndGPUFluxtop 

_CUDA kernel for applying masked flux boundary conditions on the top side._ 
```C++
template<class T>
__global__ void maskbndGPUFluxtop (
    Param XParam,
    BlockP < T > XBlock,
    FluxP < T > Flux
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `Flux` Flux variables

Applies flux boundary conditions in the halo region for masked blocks on the top side. 


        

<hr>



### function maskbndGPUbot 

_CUDA kernel for applying masked boundary conditions (halo walls) on the bottom side._ 
```C++
template<class T>
__global__ void maskbndGPUbot (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `Xev` Evolving variables 
* `zb` Array of mask values

Applies wall boundary conditions in the halo region for masked blocks on the bottom side. 


        

<hr>



### function maskbndGPUleft 

_CUDA kernel for applying masked boundary conditions (halo walls) on the left side._ 
```C++
template<class T>
__global__ void maskbndGPUleft (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `Xev` Evolving variables 
* `zb` Array of mask values

Applies wall boundary conditions in the halo region for masked blocks on the left side. 


        

<hr>



### function maskbndGPUright 

_CUDA kernel for applying masked boundary conditions (halo walls) on the right side._ 
```C++
template<class T>
__global__ void maskbndGPUright (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `Xev` Evolving variables 
* `zb` Array of mask values

Applies wall boundary conditions in the halo region for masked blocks on the right side. 


        

<hr>



### function maskbndGPUtop 

_CUDA kernel for applying masked boundary conditions (halo walls) on the top side._ 
```C++
template<class T>
__global__ void maskbndGPUtop (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > Xev,
    T * zb
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `Xev` Evolving variables 
* `zb` Array of mask values

Applies wall boundary conditions in the halo region for masked blocks on the top side. 


        

<hr>



### function noslipbnd 

_Device/host function to apply no-slip boundary condition._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `zsinside` Inside zs value 
* `hinside` Inside h value 
* `un` Normal velocity (output) 
* `ut` Tangential velocity (output) 
* `zs` zs value (output) 
* `h` h value (output)

Sets normal velocity to zero, copies zsinside and hinside. 


        

<hr>



### function noslipbndQ 

_Device/host function to apply no-slip boundary for flux variables._ 
```C++
template<class T>
__device__ __host__ void noslipbndQ (
    T & F,
    T & G,
    T & S
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `F` Flux F (output) 
* `G` Flux G (input) 
* `S` Source term (output)

Sets F to zero, S to G. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Boundary.h`


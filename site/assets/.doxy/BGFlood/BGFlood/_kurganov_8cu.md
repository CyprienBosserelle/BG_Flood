

# File Kurganov.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Kurganov.cu**](_kurganov_8cu.md)

[Go to the source code of this file](_kurganov_8cu_source.md)



* `#include "Kurganov.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**AddSlopeSourceXCPU**](#function-addslopesourcexcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* zb) <br>_Host function for adding topographic slope source terms in X direction._  |
|  template \_\_host\_\_ void | [**AddSlopeSourceXCPU&lt; double &gt;**](#function-addslopesourcexcpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* zb) <br> |
|  template \_\_host\_\_ void | [**AddSlopeSourceXCPU&lt; float &gt;**](#function-addslopesourcexcpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* zb) <br> |
|  \_\_global\_\_ void | [**AddSlopeSourceXGPU**](#function-addslopesourcexgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* zb) <br>_CUDA kernel for adding topographic slope source terms in X direction._  |
|  template \_\_global\_\_ void | [**AddSlopeSourceXGPU&lt; double &gt;**](#function-addslopesourcexgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* zb) <br> |
|  template \_\_global\_\_ void | [**AddSlopeSourceXGPU&lt; float &gt;**](#function-addslopesourcexgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* zb) <br> |
|  \_\_host\_\_ void | [**AddSlopeSourceYCPU**](#function-addslopesourceycpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* zb) <br>_Host function for adding topographic slope source terms in Y direction._  |
|  template \_\_host\_\_ void | [**AddSlopeSourceYCPU&lt; double &gt;**](#function-addslopesourceycpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* zb) <br> |
|  template \_\_host\_\_ void | [**AddSlopeSourceYCPU&lt; float &gt;**](#function-addslopesourceycpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* zb) <br> |
|  \_\_global\_\_ void | [**AddSlopeSourceYGPU**](#function-addslopesourceygpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* zb) <br>_CUDA kernel for adding topographic slope source terms in Y direction._  |
|  template \_\_global\_\_ void | [**AddSlopeSourceYGPU&lt; double &gt;**](#function-addslopesourceygpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* zb) <br> |
|  template \_\_global\_\_ void | [**AddSlopeSourceYGPU&lt; float &gt;**](#function-addslopesourceygpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* zb) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**KurgSolver**](#function-kurgsolver) (T g, T delta, T epsi, T CFL, T cm, T fm, T hp, T hm, T up, T um, T & fh, T & fu) <br>_Kurganov-Petrova approximate Riemann solver for fluxes and time step._  |
|  \_\_host\_\_ void | [**updateKurgXATMCPU**](#function-updatekurgxatmcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdx) <br>_Host function for updating X-direction fluxes with atmospheric pressure effects._  |
|  template \_\_host\_\_ void | [**updateKurgXATMCPU&lt; double &gt;**](#function-updatekurgxatmcpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb, double \* Patm, double \* dPdx) <br> |
|  template \_\_host\_\_ void | [**updateKurgXATMCPU&lt; float &gt;**](#function-updatekurgxatmcpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb, float \* Patm, float \* dPdx) <br> |
|  \_\_global\_\_ void | [**updateKurgXATMGPU**](#function-updatekurgxatmgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdx) <br>_CUDA kernel for updating X-direction fluxes with atmospheric pressure effects._  |
|  template \_\_global\_\_ void | [**updateKurgXATMGPU&lt; double &gt;**](#function-updatekurgxatmgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb, double \* Patm, double \* dPdx) <br> |
|  template \_\_global\_\_ void | [**updateKurgXATMGPU&lt; float &gt;**](#function-updatekurgxatmgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb, float \* Patm, float \* dPdx) <br> |
|  \_\_host\_\_ void | [**updateKurgXCPU**](#function-updatekurgxcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_Host function for updating X-direction fluxes using the Kurganov scheme._  |
|  template \_\_host\_\_ void | [**updateKurgXCPU&lt; double &gt;**](#function-updatekurgxcpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  template \_\_host\_\_ void | [**updateKurgXCPU&lt; float &gt;**](#function-updatekurgxcpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |
|  \_\_global\_\_ void | [**updateKurgXGPU**](#function-updatekurgxgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_CUDA kernel for updating X-direction fluxes using the Kurganov scheme._  |
|  template \_\_global\_\_ void | [**updateKurgXGPU&lt; double &gt;**](#function-updatekurgxgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  template \_\_global\_\_ void | [**updateKurgXGPU&lt; float &gt;**](#function-updatekurgxgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |
|  \_\_host\_\_ void | [**updateKurgYATMCPU**](#function-updatekurgyatmcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdy) <br>_Host function for updating Y-direction fluxes with atmospheric pressure effects._  |
|  template \_\_host\_\_ void | [**updateKurgYATMCPU&lt; double &gt;**](#function-updatekurgyatmcpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb, double \* Patm, double \* dPdy) <br> |
|  template \_\_host\_\_ void | [**updateKurgYATMCPU&lt; float &gt;**](#function-updatekurgyatmcpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb, float \* Patm, float \* dPdy) <br> |
|  \_\_global\_\_ void | [**updateKurgYATMGPU**](#function-updatekurgyatmgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdy) <br>_CUDA kernel for updating Y-direction fluxes with atmospheric pressure effects._  |
|  template \_\_global\_\_ void | [**updateKurgYATMGPU&lt; double &gt;**](#function-updatekurgyatmgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb, double \* Patm, double \* dPdy) <br> |
|  template \_\_global\_\_ void | [**updateKurgYATMGPU&lt; float &gt;**](#function-updatekurgyatmgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb, float \* Patm, float \* dPdy) <br> |
|  \_\_host\_\_ void | [**updateKurgYCPU**](#function-updatekurgycpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_Host function for updating Y-direction fluxes using the Kurganov scheme._  |
|  template \_\_host\_\_ void | [**updateKurgYCPU&lt; double &gt;**](#function-updatekurgycpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  template \_\_host\_\_ void | [**updateKurgYCPU&lt; float &gt;**](#function-updatekurgycpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |
|  \_\_global\_\_ void | [**updateKurgYGPU**](#function-updatekurgygpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_CUDA kernel for updating Y-direction fluxes using the Kurganov scheme._  |
|  template \_\_global\_\_ void | [**updateKurgYGPU&lt; double &gt;**](#function-updatekurgygpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  template \_\_global\_\_ void | [**updateKurgYGPU&lt; float &gt;**](#function-updatekurgygpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |




























## Public Functions Documentation




### function AddSlopeSourceXCPU 

_Host function for adding topographic slope source terms in X direction._ 
```C++
template<class T>
__host__ void AddSlopeSourceXCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * zb
) 
```



Updates fluxes with slope source terms for well-balanced solutions on CPU (based on kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradients 
* `XFlux` Fluxes 
* `zb` Bathymetry array 




        

<hr>



### function AddSlopeSourceXCPU&lt; double &gt; 

```C++
template __host__ void AddSlopeSourceXCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * zb
) 
```




<hr>



### function AddSlopeSourceXCPU&lt; float &gt; 

```C++
template __host__ void AddSlopeSourceXCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * zb
) 
```




<hr>



### function AddSlopeSourceXGPU 

_CUDA kernel for adding topographic slope source terms in X direction._ 
```C++
template<class T>
__global__ void AddSlopeSourceXGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * zb
) 
```



Updates fluxes with slope source terms for well-balanced solutions (based on Kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradients 
* `XFlux` Fluxes 
* `zb` Bathymetry array 




        

<hr>



### function AddSlopeSourceXGPU&lt; double &gt; 

```C++
template __global__ void AddSlopeSourceXGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * zb
) 
```




<hr>



### function AddSlopeSourceXGPU&lt; float &gt; 

```C++
template __global__ void AddSlopeSourceXGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * zb
) 
```




<hr>



### function AddSlopeSourceYCPU 

_Host function for adding topographic slope source terms in Y direction._ 
```C++
template<class T>
__host__ void AddSlopeSourceYCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * zb
) 
```



Updates fluxes with slope source terms for well-balanced solutions in Y direction on CPU (based on kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradients 
* `XFlux` Fluxes 
* `zb` Bathymetry array 




        

<hr>



### function AddSlopeSourceYCPU&lt; double &gt; 

```C++
template __host__ void AddSlopeSourceYCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * zb
) 
```




<hr>



### function AddSlopeSourceYCPU&lt; float &gt; 

```C++
template __host__ void AddSlopeSourceYCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * zb
) 
```




<hr>



### function AddSlopeSourceYGPU 

_CUDA kernel for adding topographic slope source terms in Y direction._ 
```C++
template<class T>
__global__ void AddSlopeSourceYGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * zb
) 
```



Updates fluxes with slope source terms for well-balanced solutions in Y direction (based on kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradients 
* `XFlux` Fluxes 
* `zb` Bathymetry array 




        

<hr>



### function AddSlopeSourceYGPU&lt; double &gt; 

```C++
template __global__ void AddSlopeSourceYGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * zb
) 
```




<hr>



### function AddSlopeSourceYGPU&lt; float &gt; 

```C++
template __global__ void AddSlopeSourceYGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * zb
) 
```




<hr>



### function KurgSolver 

_Kurganov-Petrova approximate Riemann solver for fluxes and time step._ 
```C++
template<class T>
__host__ __device__ T KurgSolver (
    T g,
    T delta,
    T epsi,
    T CFL,
    T cm,
    T fm,
    T hp,
    T hm,
    T up,
    T um,
    T & fh,
    T & fu
) 
```



Computes fluxes and time step for the Kurganov scheme given left/right states and velocities (based on kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `g` Gravity 
* `delta` Cell size 
* `epsi` Small epsilon for stability 
* `CFL` CFL number 
* `cm` Metric coefficient 
* `fm` Flux metric 
* `hp` Water depth (plus side) 
* `hm` Water depth (minus side) 
* `up` Velocity (plus side) 
* `um` Velocity (minus side) 
* `fh` Output: flux for h 
* `fu` Output: flux for u 



**Returns:**

Time step 





        

<hr>



### function updateKurgXATMCPU 

_Host function for updating X-direction fluxes with atmospheric pressure effects._ 
```C++
template<class T>
__host__ void updateKurgXATMCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb,
    T * Patm,
    T * dPdx
) 
```



Computes fluxes and time step constraints for each cell in the X direction on CPU, including atmospheric pressure terms (based on kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradients 
* `XFlux` Fluxes 
* `dtmax` Maximum time step array 
* `zb` Bathymetry array 
* `Patm` Atmospheric pressure array 
* `dPdx` Pressure gradient array 




        

<hr>



### function updateKurgXATMCPU&lt; double &gt; 

```C++
template __host__ void updateKurgXATMCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb,
    double * Patm,
    double * dPdx
) 
```




<hr>



### function updateKurgXATMCPU&lt; float &gt; 

```C++
template __host__ void updateKurgXATMCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb,
    float * Patm,
    float * dPdx
) 
```




<hr>



### function updateKurgXATMGPU 

_CUDA kernel for updating X-direction fluxes with atmospheric pressure effects._ 
```C++
template<class T>
__global__ void updateKurgXATMGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb,
    T * Patm,
    T * dPdx
) 
```



Computes fluxes and time step constraints for each cell in the X direction, including atmospheric pressure terms (based on kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradients 
* `XFlux` Fluxes 
* `dtmax` Maximum time step array 
* `zb` Bathymetry array 
* `Patm` Atmospheric pressure array 
* `dPdx` Pressure gradient array 




        

<hr>



### function updateKurgXATMGPU&lt; double &gt; 

```C++
template __global__ void updateKurgXATMGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb,
    double * Patm,
    double * dPdx
) 
```




<hr>



### function updateKurgXATMGPU&lt; float &gt; 

```C++
template __global__ void updateKurgXATMGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb,
    float * Patm,
    float * dPdx
) 
```




<hr>



### function updateKurgXCPU 

_Host function for updating X-direction fluxes using the Kurganov scheme._ 
```C++
template<class T>
__host__ void updateKurgXCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb
) 
```



Computes fluxes and time step constraints for each cell in the X direction on CPU (based on kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradients 
* `XFlux` Fluxes 
* `dtmax` Maximum time step array 
* `zb` Bathymetry array 




        

<hr>



### function updateKurgXCPU&lt; double &gt; 

```C++
template __host__ void updateKurgXCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb
) 
```




<hr>



### function updateKurgXCPU&lt; float &gt; 

```C++
template __host__ void updateKurgXCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb
) 
```




<hr>



### function updateKurgXGPU 

_CUDA kernel for updating X-direction fluxes using the Kurganov scheme._ 
```C++
template<class T>
__global__ void updateKurgXGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb
) 
```



Computes fluxes and time step constraints for each cell in the X direction (based on kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradients 
* `XFlux` Fluxes 
* `dtmax` Maximum time step array 
* `zb` Bathymetry array 




        

<hr>



### function updateKurgXGPU&lt; double &gt; 

```C++
template __global__ void updateKurgXGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb
) 
```




<hr>



### function updateKurgXGPU&lt; float &gt; 

```C++
template __global__ void updateKurgXGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb
) 
```




<hr>



### function updateKurgYATMCPU 

_Host function for updating Y-direction fluxes with atmospheric pressure effects._ 
```C++
template<class T>
__host__ void updateKurgYATMCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb,
    T * Patm,
    T * dPdy
) 
```



Computes fluxes and time step constraints for each cell in the Y direction on CPU, including atmospheric pressure terms (based on kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradients 
* `XFlux` Fluxes 
* `dtmax` Maximum time step array 
* `zb` Bathymetry array 
* `Patm` Atmospheric pressure array 
* `dPdy` Pressure gradient array 




        

<hr>



### function updateKurgYATMCPU&lt; double &gt; 

```C++
template __host__ void updateKurgYATMCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb,
    double * Patm,
    double * dPdy
) 
```




<hr>



### function updateKurgYATMCPU&lt; float &gt; 

```C++
template __host__ void updateKurgYATMCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb,
    float * Patm,
    float * dPdy
) 
```




<hr>



### function updateKurgYATMGPU 

_CUDA kernel for updating Y-direction fluxes with atmospheric pressure effects._ 
```C++
template<class T>
__global__ void updateKurgYATMGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb,
    T * Patm,
    T * dPdy
) 
```



Computes fluxes and time step constraints for each cell in the Y direction, including atmospheric pressure terms (based on kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradients 
* `XFlux` Fluxes 
* `dtmax` Maximum time step array 
* `zb` Bathymetry array 
* `Patm` Atmospheric pressure array 
* `dPdy` Pressure gradient array 




        

<hr>



### function updateKurgYATMGPU&lt; double &gt; 

```C++
template __global__ void updateKurgYATMGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb,
    double * Patm,
    double * dPdy
) 
```




<hr>



### function updateKurgYATMGPU&lt; float &gt; 

```C++
template __global__ void updateKurgYATMGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb,
    float * Patm,
    float * dPdy
) 
```




<hr>



### function updateKurgYCPU 

_Host function for updating Y-direction fluxes using the Kurganov scheme._ 
```C++
template<class T>
__host__ void updateKurgYCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb
) 
```



Computes fluxes and time step constraints for each cell in the Y direction on CPU (based on kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradients 
* `XFlux` Fluxes 
* `dtmax` Maximum time step array 
* `zb` Bathymetry array 




        

<hr>



### function updateKurgYCPU&lt; double &gt; 

```C++
template __host__ void updateKurgYCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb
) 
```




<hr>



### function updateKurgYCPU&lt; float &gt; 

```C++
template __host__ void updateKurgYCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb
) 
```




<hr>



### function updateKurgYGPU 

_CUDA kernel for updating Y-direction fluxes using the Kurganov scheme._ 
```C++
template<class T>
__global__ void updateKurgYGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb
) 
```



Computes fluxes and time step constraints for each cell in the Y direction (based on kurganov and Petrova 2007).




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `XGrad` Gradients 
* `XFlux` Fluxes 
* `dtmax` Maximum time step array 
* `zb` Bathymetry array 




        

<hr>



### function updateKurgYGPU&lt; double &gt; 

```C++
template __global__ void updateKurgYGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxP < double > XFlux,
    double * dtmax,
    double * zb
) 
```




<hr>



### function updateKurgYGPU&lt; float &gt; 

```C++
template __global__ void updateKurgYGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxP < float > XFlux,
    float * dtmax,
    float * zb
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Kurganov.cu`


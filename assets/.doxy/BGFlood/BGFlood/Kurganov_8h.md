

# File Kurganov.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Kurganov.h**](Kurganov_8h.md)

[Go to the source code of this file](Kurganov_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "MemManagement.h"`
* `#include "Util_CPU.h"`
* `#include "Spherical.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**AddSlopeSourceXCPU**](#function-addslopesourcexcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* zb) <br>_Host function for adding topographic slope source terms in X direction._  |
|  \_\_global\_\_ void | [**AddSlopeSourceXGPU**](#function-addslopesourcexgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* zb) <br>_CUDA kernel for adding topographic slope source terms in X direction._  |
|  \_\_host\_\_ void | [**AddSlopeSourceYCPU**](#function-addslopesourceycpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* zb) <br>_Host function for adding topographic slope source terms in Y direction._  |
|  \_\_global\_\_ void | [**AddSlopeSourceYGPU**](#function-addslopesourceygpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* zb) <br>_CUDA kernel for adding topographic slope source terms in Y direction._  |
|  \_\_host\_\_ \_\_device\_\_ T | [**KurgSolver**](#function-kurgsolver) (T g, T delta, T epsi, T CFL, T cm, T fm, T hp, T hm, T up, T um, T & fh, T & fu) <br>_Kurganov-Petrova approximate Riemann solver for fluxes and time step._  |
|  \_\_host\_\_ void | [**updateKurgXATMCPU**](#function-updatekurgxatmcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdx) <br>_Host function for updating X-direction fluxes with atmospheric pressure effects._  |
|  \_\_global\_\_ void | [**updateKurgXATMGPU**](#function-updatekurgxatmgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdx) <br>_CUDA kernel for updating X-direction fluxes with atmospheric pressure effects._  |
|  \_\_host\_\_ void | [**updateKurgXCPU**](#function-updatekurgxcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_Host function for updating X-direction fluxes using the Kurganov scheme._  |
|  \_\_global\_\_ void | [**updateKurgXGPU**](#function-updatekurgxgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_CUDA kernel for updating X-direction fluxes using the Kurganov scheme._  |
|  \_\_host\_\_ void | [**updateKurgYATMCPU**](#function-updatekurgyatmcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdy) <br>_Host function for updating Y-direction fluxes with atmospheric pressure effects._  |
|  \_\_global\_\_ void | [**updateKurgYATMGPU**](#function-updatekurgyatmgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb, T \* Patm, T \* dPdy) <br>_CUDA kernel for updating Y-direction fluxes with atmospheric pressure effects._  |
|  \_\_host\_\_ void | [**updateKurgYCPU**](#function-updatekurgycpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_Host function for updating Y-direction fluxes using the Kurganov scheme._  |
|  \_\_global\_\_ void | [**updateKurgYGPU**](#function-updatekurgygpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_CUDA kernel for updating Y-direction fluxes using the Kurganov scheme._  |




























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

------------------------------
The documentation for this class was generated from the following file `src/Kurganov.h`


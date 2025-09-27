

# File Reimann.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Reimann.h**](Reimann_8h.md)

[Go to the source code of this file](Reimann_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "MemManagement.h"`
* `#include "Util_CPU.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**UpdateButtingerXCPU**](#function-updatebuttingerxcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_"Adaptive" second-order hydrostatic reconstruction. CPU version for the X-axis_  |
|  \_\_global\_\_ void | [**UpdateButtingerXGPU**](#function-updatebuttingerxgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_"Adaptive" second-order hydrostatic reconstruction. GPU version for t X-axis_  |
|  \_\_host\_\_ void | [**UpdateButtingerYCPU**](#function-updatebuttingerycpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_"Adaptive" second-order hydrostatic reconstruction. CPU version for the Y-axis_  |
|  \_\_global\_\_ void | [**UpdateButtingerYGPU**](#function-updatebuttingerygpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, [**GradientsP**](structGradientsP.md)&lt; T &gt; XGrad, [**FluxP**](structFluxP.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_"Adaptive" second-order hydrostatic reconstruction. GPU version for the Y-axis_  |
|  \_\_host\_\_ \_\_device\_\_ T | [**hllc**](#function-hllc) (T g, T delta, T epsi, T CFL, T cm, T fm, T hm, T hp, T um, T up, T & fh, T & fq) <br>_Calculate the Harten-Lax-van Leer-contact (HLLC) flux._  |




























## Public Functions Documentation




### function UpdateButtingerXCPU 

_"Adaptive" second-order hydrostatic reconstruction. CPU version for the X-axis_ 
```C++
template<class T>
__host__ void UpdateButtingerXCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb
) 
```



### Description



This function computes the flux term at the cell interface using the hydrostatic reconstruction from Buttinger et al (2019). This reconstruction is safe for steep slope with thin water depth and is well-balanced meaning that it conserve the "lake-at-rest" states.


For optimising the code on CPU and GPU there are 4 versions of this function: X or Y and CPU or GPU



### Where does this come from:



This scheme was adapted/modified from the Basilisk / B-Flood source code. I (CypB) changed the zr and zl term back to the Audusse type reconstruction [http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h](http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h)


Reference: Kirstetter, G., Delestre, O., Lagree, P.-Y., Popinet, S., and Josserand, C.: B-flood 1.0: an open-source Saint-Venant model for flash flood simulation using adaptive refinement, Geosci. [**Model**](structModel.md) Dev. Discuss. [preprint], [https://doi.org/10.5194/gmd-2021-15](https://doi.org/10.5194/gmd-2021-15), in review, 2021.\* Buttinger-Kreuzhuber, A., Horvath, Z., Noelle, S., Bloschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional structured grids over abrupt topography, Advances in water resources, 127, 89-108, 2019. 



        

<hr>



### function UpdateButtingerXGPU 

_"Adaptive" second-order hydrostatic reconstruction. GPU version for t X-axis_ 
```C++
template<class T>
__global__ void UpdateButtingerXGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb
) 
```



### Description



This function computes the flux term at the cell interface using the hydrostatic reconstruction from Buttinger et al (2019). This reconstruction is safe for steep slope with thin water depth and is well-balanced meaning that it conserve the "lake-at-rest" states.


For optimising the code on CPU and GPU there are 4 versions of this function: X or Y and CPU or GPU



### Where does this come from:



This scheme was adapted/modified from the Basilisk / B-Flood source code. I (CypB) changed the zr and zl term back to the Audusse type reconstruction [http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h](http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h)


Reference: Kirstetter, G., Delestre, O., Lagree, P.-Y., Popinet, S., and Josserand, C.: B-flood 1.0: an open-source Saint-Venant model for flash flood simulation using adaptive refinement, Geosci. [**Model**](structModel.md) Dev. Discuss. [preprint], [https://doi.org/10.5194/gmd-2021-15](https://doi.org/10.5194/gmd-2021-15), in review, 2021.\* Buttinger-Kreuzhuber, A., Horvath, Z., Noelle, S., Bloschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional structured grids over abrupt topography, Advances in water resources, 127, 89-108, 2019. 



        

<hr>



### function UpdateButtingerYCPU 

_"Adaptive" second-order hydrostatic reconstruction. CPU version for the Y-axis_ 
```C++
template<class T>
__host__ void UpdateButtingerYCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb
) 
```



### Description



This function computes the flux term at the cell interface using the hydrostatic reconstruction from Buttinger et al (2019). This reconstruction is safe for steep slope with thin water depth and is well-balanced meaning that it conserve the "lake-at-rest" states.


For optimising the code on CPU and GPU there are 4 versions of this function: X or Y and CPU or GPU



### Where does this come from:



This scheme was adapted/modified from the Basilisk / B-Flood source code. I (CypB) changed the zr and zl term back to the Audusse type reconstruction [http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h](http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h)


Reference: Kirstetter, G., Delestre, O., Lagree, P.-Y., Popinet, S., and Josserand, C.: B-flood 1.0: an open-source Saint-Venant model for flash flood simulation using adaptive refinement, Geosci. [**Model**](structModel.md) Dev. Discuss. [preprint], [https://doi.org/10.5194/gmd-2021-15](https://doi.org/10.5194/gmd-2021-15), in review, 2021.\* Buttinger-Kreuzhuber, A., Horvath, Z., Noelle, S., Bloschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional structured grids over abrupt topography, Advances in water resources, 127, 89-108, 2019. 



        

<hr>



### function UpdateButtingerYGPU 

_"Adaptive" second-order hydrostatic reconstruction. GPU version for the Y-axis_ 
```C++
template<class T>
__global__ void UpdateButtingerYGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxP < T > XFlux,
    T * dtmax,
    T * zb
) 
```



### Description



This function computes the flux term at the cell interface using the hydrostatic reconstruction from Buttinger et al (2019). This reconstruction is safe for steep slope with thin water depth and is well-balanced meaning that it conserve the "lake-at-rest" states.


For optimising the code on CPU and GPU there are 4 versions of this function: X or Y and CPU or GPU



### Where does this come from:



This scheme was adapted/modified from the Basilisk / B-Flood source code. I (CypB) changed the zr and zl term back to the Audusse type reconstruction [http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h](http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h)


Reference: Kirstetter, G., Delestre, O., Lagree, P.-Y., Popinet, S., and Josserand, C.: B-flood 1.0: an open-source Saint-Venant model for flash flood simulation using adaptive refinement, Geosci. [**Model**](structModel.md) Dev. Discuss. [preprint], [https://doi.org/10.5194/gmd-2021-15](https://doi.org/10.5194/gmd-2021-15), in review, 2021.\* Buttinger-Kreuzhuber, A., Horvath, Z., Noelle, S., Bloschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional structured grids over abrupt topography, Advances in water resources, 127, 89-108, 2019. 



        

<hr>



### function hllc 

_Calculate the Harten-Lax-van Leer-contact (HLLC) flux._ 
```C++
template<class T>
__host__ __device__ T hllc (
    T g,
    T delta,
    T epsi,
    T CFL,
    T cm,
    T fm,
    T hm,
    T hp,
    T um,
    T up,
    T & fh,
    T & fq
) 
```



### Description



This an implementation of the HLLC solver.



### Where does this come from:



This scheme was adapted/modified from the Basilisk source code. [http://basilisk.fr/src/riemann.h](http://basilisk.fr/src/riemann.h)


Reference: (Basilisk reference the scheme from Kurganov reference below) Kurganov, A., & Levy, D. (2002). Central-upwind schemes for the Saint-Venant system. Mathematical Modelling and Numerical Analysis, 36(3), 397-425. 



        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Reimann.h`


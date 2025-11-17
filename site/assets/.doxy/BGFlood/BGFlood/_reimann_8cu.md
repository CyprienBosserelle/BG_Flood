

# File Reimann.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Reimann.cu**](_reimann_8cu.md)

[Go to the source code of this file](_reimann_8cu_source.md)



* `#include "Reimann.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**UpdateButtingerXCPU**](#function-updatebuttingerxcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_"Adaptive" second-order hydrostatic reconstruction. CPU version for the X-axis_  |
|  template \_\_host\_\_ void | [**UpdateButtingerXCPU**](#function-updatebuttingerxcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |
|  template \_\_host\_\_ void | [**UpdateButtingerXCPU**](#function-updatebuttingerxcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  \_\_global\_\_ void | [**UpdateButtingerXGPU**](#function-updatebuttingerxgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_"Adaptive" second-order hydrostatic reconstruction. GPU version for the X-axis_  |
|  template \_\_global\_\_ void | [**UpdateButtingerXGPU**](#function-updatebuttingerxgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |
|  template \_\_global\_\_ void | [**UpdateButtingerXGPU**](#function-updatebuttingerxgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  \_\_host\_\_ void | [**UpdateButtingerYCPU**](#function-updatebuttingerycpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_"Adaptive" second-order hydrostatic reconstruction. CPU version for the Y-axis_  |
|  template \_\_host\_\_ void | [**UpdateButtingerYCPU**](#function-updatebuttingerycpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |
|  template \_\_host\_\_ void | [**UpdateButtingerYCPU**](#function-updatebuttingerycpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  \_\_global\_\_ void | [**UpdateButtingerYGPU**](#function-updatebuttingerygpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br>_"Adaptive" second-order hydrostatic reconstruction. GPU version for the Y-axis_  |
|  template \_\_global\_\_ void | [**UpdateButtingerYGPU**](#function-updatebuttingerygpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |
|  template \_\_global\_\_ void | [**UpdateButtingerYGPU**](#function-updatebuttingerygpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxP**](struct_flux_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
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



This function computes the flux term at the cell interface using the hydrostatic reconstruction from Buttinger et al (2019). This reconstruction is safe for steep slopes with thin water depth and is well-balanced, conserving "lake-at-rest" states.


For optimizing the code on CPU and GPU, there are 4 versions of this function: X or Y and CPU or GPU. 



### Where does this come from:



This scheme was adapted/modified from the Basilisk / B-Flood source code. I (CypB) changed the zr and zl term back to the Audusse type reconstruction [http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h](http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h)


Reference:
* Buttinger-Kreuzhuber, A., Horvath, Z., Noelle, S., Bloschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional structured grids over abrupt topography, Advances in water resources, 127, 89-108, 2019.
* Kirstetter, G., Delestre, O., Lagree, P.-Y., Popinet, S., and Josserand, C.: B-flood 1.0: an open-source Saint-Venant model for flash flood simulation using adaptive refinement, Geosci. [**Model**](struct_model.md) Dev. Discuss. [preprint], [https://doi.org/10.5194/gmd-2021-15](https://doi.org/10.5194/gmd-2021-15), in review, 2021. 

**Parameters:**


  * `XParam` Structure containing the simulation parameters 
  * `XBlock` Structure containing the block information 
  * `XEv` Structure containing the evolving variables 
  * `XGrad` Structure containing the gradients of the evolving variables 
  * `XFlux` Structure containing the fluxes to be updated 
  * `dtmax` Array to store the maximum allowable time step for each cell 
  * `zb` Array containing the bed elevation 



**Template parameters:**


  * `T` Data type, either float or double 



**Note:**

This function is designed to be run on the CPU and should be called within a loop over all blocks. 








        

<hr>



### function UpdateButtingerXCPU 

```C++
template __host__ void UpdateButtingerXCPU (
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



### function UpdateButtingerXCPU 

```C++
template __host__ void UpdateButtingerXCPU (
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



### function UpdateButtingerXGPU 

_"Adaptive" second-order hydrostatic reconstruction. GPU version for the X-axis_ 
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



This function computes the flux term at the cell interface using the hydrostatic reconstruction from Buttinger et al (2019). This reconstruction is safe for steep slopes with thin water depth and is well-balanced, conserving "lake-at-rest" states.


For optimising the code on CPU and GPU there are 4 versions of this function: X or Y and CPU or GPU



### Where does this come from:



This scheme was adapted/modified from the Basilisk / B-Flood source code. I (CypB) changed the zr and zl term back to the Audusse type reconstruction [http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h](http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h)



### Reference:



Kirstetter, G., Delestre, O., Lagree, P.-Y., Popinet, S., and Josserand, C.: B-flood 1.0: an open-source Saint-Venant model for flash flood simulation using adaptive refinement, Geosci. [**Model**](struct_model.md) Dev. Discuss. [preprint], [https://doi.org/10.5194/gmd-2021-15](https://doi.org/10.5194/gmd-2021-15), in review, 2021.\* Buttinger-Kreuzhuber, A., Horvath, Z., Noelle, S., Bloschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional structured grids over abrupt topography, Advances in water resources, 127, 89-108, 2019.




**Parameters:**


* `XParam` Structure containing the simulation parameters 
* `XBlock` Structure containing the block information 
* `XEv` Structure containing the evolving variables 
* `XGrad` Structure containing the gradients of the evolving variables 
* `XFlux` Structure containing the fluxes to be updated 
* `dtmax` Array to store the maximum allowable time step for each cell 
* `zb` Array containing the bed elevation 
* `T` Data type, either float or double 



**Note:**

This function is designed to be run on the GPU and should be launched with a grid and block configuration that matches the number of blocks and block size. 






        

<hr>



### function UpdateButtingerXGPU 

```C++
template __global__ void UpdateButtingerXGPU (
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



### function UpdateButtingerXGPU 

```C++
template __global__ void UpdateButtingerXGPU (
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


Reference: Kirstetter, G., Delestre, O., Lagree, P.-Y., Popinet, S., and Josserand, C.: B-flood 1.0: an open-source Saint-Venant model for flash flood simulation using adaptive refinement, Geosci. [**Model**](struct_model.md) Dev. Discuss. [preprint], [https://doi.org/10.5194/gmd-2021-15](https://doi.org/10.5194/gmd-2021-15), in review, 2021.\* Buttinger-Kreuzhuber, A., Horvath, Z., Noelle, S., Bloschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional structured grids over abrupt topography, Advances in water resources, 127, 89-108, 2019. 

**Parameters:**


* `XParam` Structure containing the simulation parameters 
* `XBlock` Structure containing the block information 
* `XEv` Structure containing the evolving variables 
* `XGrad` Structure containing the gradients of the evolving variables 
* `XFlux` Structure containing the fluxes to be updated 
* `dtmax` Array to store the maximum allowable time step for each cell 
* `zb` Array containing the bed elevation 
* `T` Data type, either float or double 



**Note:**

This function is designed to be run on the CPU and should be called within a loop over all blocks. 






        

<hr>



### function UpdateButtingerYCPU 

```C++
template __host__ void UpdateButtingerYCPU (
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



### function UpdateButtingerYCPU 

```C++
template __host__ void UpdateButtingerYCPU (
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


Reference: Kirstetter, G., Delestre, O., Lagree, P.-Y., Popinet, S., and Josserand, C.: B-flood 1.0: an open-source Saint-Venant model for flash flood simulation using adaptive refinement, Geosci. [**Model**](struct_model.md) Dev. Discuss. [preprint], [https://doi.org/10.5194/gmd-2021-15](https://doi.org/10.5194/gmd-2021-15), in review, 2021.\* Buttinger-Kreuzhuber, A., Horvath, Z., Noelle, S., Bloschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional structured grids over abrupt topography, Advances in water resources, 127, 89-108, 2019. 

**Parameters:**


* `XParam` Structure containing the simulation parameters 
* `XBlock` Structure containing the block information 
* `XEv` Structure containing the evolving variables 
* `XGrad` Structure containing the gradients of the evolving variables 
* `XFlux` Structure containing the fluxes to be updated 
* `dtmax` Array to store the maximum allowable time step for each cell 
* `zb` Array containing the bed elevation 



**Template parameters:**


* `T` Data type, either float or double 



**Note:**

This function is designed to be run on the GPU and should be launched with a grid and block configuration that matches the number of blocks and block size. 






        

<hr>



### function UpdateButtingerYGPU 

```C++
template __global__ void UpdateButtingerYGPU (
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



### function UpdateButtingerYGPU 

```C++
template __global__ void UpdateButtingerYGPU (
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




**Parameters:**


* `g` Gravitational acceleration 
* `delta` Grid resolution at the current level 
* `epsi` Small number to prevent division by zero 
* `CFL` Courant-Friedrichs-Lewy number for stability condition 
* `cm` Metric term for spherical coordinates (1.0 for Cartesian) 
* `fm` Metric term for spherical coordinates (1.0 for Cartesian) 
* `hm` Water depth on the left side of the interface 
* `hp` Water depth on the right side of the interface 
* `um` Velocity in the x-direction on the left side of the interface 
* `up` Velocity in the x-direction on the right side of the interface 
* `fh` Reference to store the computed flux for water depth 
* `fq` Reference to store the computed flux for momentum 



**Returns:**

The maximum allowable time step based on the wave speeds and CFL condition 




**Template parameters:**


* `T` Data type, either float or double 



**Note:**

This function is designed to be run on both CPU and GPU. 






        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Reimann.cu`


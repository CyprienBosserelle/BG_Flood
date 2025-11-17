

# File Multilayer.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Multilayer.cu**](_multilayer_8cu.md)

[Go to the source code of this file](_multilayer_8cu_source.md)



* `#include "Multilayer.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_global\_\_ void | [**AdvecEv**](#function-advecev) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T dt, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; T &gt; XFlux) <br> |
|  template \_\_global\_\_ void | [**AdvecEv&lt; double &gt;**](#function-advecev-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double dt, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; double &gt; XFlux) <br> |
|  template \_\_global\_\_ void | [**AdvecEv&lt; float &gt;**](#function-advecev-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float dt, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; float &gt; XFlux) <br> |
|  \_\_global\_\_ void | [**AdvecFluxML**](#function-advecfluxml) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T dt, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; T &gt; XFlux) <br> |
|  template \_\_global\_\_ void | [**AdvecFluxML&lt; double &gt;**](#function-advecfluxml-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double dt, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; double &gt; XFlux) <br> |
|  template \_\_global\_\_ void | [**AdvecFluxML&lt; float &gt;**](#function-advecfluxml-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float dt, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; float &gt; XFlux) <br> |
|  \_\_global\_\_ void | [**CalcfaceValX**](#function-calcfacevalx) (T pdt, [**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br> |
|  template \_\_global\_\_ void | [**CalcfaceValX&lt; double &gt;**](#function-calcfacevalx-double) (double pdt, [**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  template \_\_global\_\_ void | [**CalcfaceValX&lt; float &gt;**](#function-calcfacevalx-float) (float pdt, [**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |
|  \_\_global\_\_ void | [**CalcfaceValY**](#function-calcfacevaly) (T pdt, [**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; T &gt; XFlux, T \* dtmax, T \* zb) <br> |
|  template \_\_global\_\_ void | [**CalcfaceValY&lt; double &gt;**](#function-calcfacevaly-double) (double pdt, [**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; double &gt; XFlux, double \* dtmax, double \* zb) <br> |
|  template \_\_global\_\_ void | [**CalcfaceValY&lt; float &gt;**](#function-calcfacevaly-float) (float pdt, [**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; float &gt; XFlux, float \* dtmax, float \* zb) <br> |
|  \_\_global\_\_ void | [**CheckadvecMLX**](#function-checkadvecmlx) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T dt, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; T &gt; XFlux) <br> |
|  template \_\_global\_\_ void | [**CheckadvecMLX&lt; double &gt;**](#function-checkadvecmlx-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double dt, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; double &gt; XFlux) <br> |
|  template \_\_global\_\_ void | [**CheckadvecMLX&lt; float &gt;**](#function-checkadvecmlx-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float dt, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; float &gt; XFlux) <br> |
|  \_\_global\_\_ void | [**CheckadvecMLY**](#function-checkadvecmly) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T dt, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; T &gt; XFlux) <br> |
|  template \_\_global\_\_ void | [**CheckadvecMLY&lt; double &gt;**](#function-checkadvecmly-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double dt, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; double &gt; XFlux) <br> |
|  template \_\_global\_\_ void | [**CheckadvecMLY&lt; float &gt;**](#function-checkadvecmly-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float dt, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; float &gt; XFlux) <br> |
|  \_\_global\_\_ void | [**CleanupML**](#function-cleanupml) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, T \* zb) <br> |
|  template \_\_global\_\_ void | [**CleanupML&lt; double &gt;**](#function-cleanupml-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, double \* zb) <br> |
|  template \_\_global\_\_ void | [**CleanupML&lt; float &gt;**](#function-cleanupml-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, float \* zb) <br> |
|  \_\_global\_\_ void | [**pressureML**](#function-pressureml) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T dt, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; T &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; T &gt; XFlux) <br> |
|  template \_\_global\_\_ void | [**pressureML&lt; double &gt;**](#function-pressureml-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double dt, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; double &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; double &gt; XFlux) <br> |
|  template \_\_global\_\_ void | [**pressureML&lt; float &gt;**](#function-pressureml-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float dt, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEv, [**GradientsP**](struct_gradients_p.md)&lt; float &gt; XGrad, [**FluxMLP**](struct_flux_m_l_p.md)&lt; float &gt; XFlux) <br> |




























## Public Functions Documentation




### function AdvecEv 

```C++
template<class T>
__global__ void AdvecEv (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux
) 
```




<hr>



### function AdvecEv&lt; double &gt; 

```C++
template __global__ void AdvecEv< double > (
    Param XParam,
    BlockP < double > XBlock,
    double dt,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxMLP < double > XFlux
) 
```




<hr>



### function AdvecEv&lt; float &gt; 

```C++
template __global__ void AdvecEv< float > (
    Param XParam,
    BlockP < float > XBlock,
    float dt,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxMLP < float > XFlux
) 
```




<hr>



### function AdvecFluxML 

```C++
template<class T>
__global__ void AdvecFluxML (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux
) 
```




<hr>



### function AdvecFluxML&lt; double &gt; 

```C++
template __global__ void AdvecFluxML< double > (
    Param XParam,
    BlockP < double > XBlock,
    double dt,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxMLP < double > XFlux
) 
```




<hr>



### function AdvecFluxML&lt; float &gt; 

```C++
template __global__ void AdvecFluxML< float > (
    Param XParam,
    BlockP < float > XBlock,
    float dt,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxMLP < float > XFlux
) 
```




<hr>



### function CalcfaceValX 

```C++
template<class T>
__global__ void CalcfaceValX (
    T pdt,
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux,
    T * dtmax,
    T * zb
) 
```




<hr>



### function CalcfaceValX&lt; double &gt; 

```C++
template __global__ void CalcfaceValX< double > (
    double pdt,
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxMLP < double > XFlux,
    double * dtmax,
    double * zb
) 
```




<hr>



### function CalcfaceValX&lt; float &gt; 

```C++
template __global__ void CalcfaceValX< float > (
    float pdt,
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxMLP < float > XFlux,
    float * dtmax,
    float * zb
) 
```




<hr>



### function CalcfaceValY 

```C++
template<class T>
__global__ void CalcfaceValY (
    T pdt,
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux,
    T * dtmax,
    T * zb
) 
```




<hr>



### function CalcfaceValY&lt; double &gt; 

```C++
template __global__ void CalcfaceValY< double > (
    double pdt,
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxMLP < double > XFlux,
    double * dtmax,
    double * zb
) 
```




<hr>



### function CalcfaceValY&lt; float &gt; 

```C++
template __global__ void CalcfaceValY< float > (
    float pdt,
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxMLP < float > XFlux,
    float * dtmax,
    float * zb
) 
```




<hr>



### function CheckadvecMLX 

```C++
template<class T>
__global__ void CheckadvecMLX (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux
) 
```




<hr>



### function CheckadvecMLX&lt; double &gt; 

```C++
template __global__ void CheckadvecMLX< double > (
    Param XParam,
    BlockP < double > XBlock,
    double dt,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxMLP < double > XFlux
) 
```




<hr>



### function CheckadvecMLX&lt; float &gt; 

```C++
template __global__ void CheckadvecMLX< float > (
    Param XParam,
    BlockP < float > XBlock,
    float dt,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxMLP < float > XFlux
) 
```




<hr>



### function CheckadvecMLY 

```C++
template<class T>
__global__ void CheckadvecMLY (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux
) 
```




<hr>



### function CheckadvecMLY&lt; double &gt; 

```C++
template __global__ void CheckadvecMLY< double > (
    Param XParam,
    BlockP < double > XBlock,
    double dt,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxMLP < double > XFlux
) 
```




<hr>



### function CheckadvecMLY&lt; float &gt; 

```C++
template __global__ void CheckadvecMLY< float > (
    Param XParam,
    BlockP < float > XBlock,
    float dt,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxMLP < float > XFlux
) 
```




<hr>



### function CleanupML 

```C++
template<class T>
__global__ void CleanupML (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEv,
    T * zb
) 
```




<hr>



### function CleanupML&lt; double &gt; 

```C++
template __global__ void CleanupML< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEv,
    double * zb
) 
```




<hr>



### function CleanupML&lt; float &gt; 

```C++
template __global__ void CleanupML< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEv,
    float * zb
) 
```




<hr>



### function pressureML 

```C++
template<class T>
__global__ void pressureML (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    EvolvingP < T > XEv,
    GradientsP < T > XGrad,
    FluxMLP < T > XFlux
) 
```




<hr>



### function pressureML&lt; double &gt; 

```C++
template __global__ void pressureML< double > (
    Param XParam,
    BlockP < double > XBlock,
    double dt,
    EvolvingP < double > XEv,
    GradientsP < double > XGrad,
    FluxMLP < double > XFlux
) 
```




<hr>



### function pressureML&lt; float &gt; 

```C++
template __global__ void pressureML< float > (
    Param XParam,
    BlockP < float > XBlock,
    float dt,
    EvolvingP < float > XEv,
    GradientsP < float > XGrad,
    FluxMLP < float > XFlux
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Multilayer.cu`


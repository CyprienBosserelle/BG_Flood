

# File Friction.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Friction.cu**](Friction_8cu.md)

[Go to the source code of this file](Friction_8cu_source.md)



* `#include "Friction.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**TheresholdVelCPU**](#function-theresholdvelcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv) <br>_CPU routine for enforcing a velocity threshold._  |
|  template \_\_host\_\_ void | [**TheresholdVelCPU&lt; double &gt;**](#function-theresholdvelcpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEvolv) <br> |
|  template \_\_host\_\_ void | [**TheresholdVelCPU&lt; float &gt;**](#function-theresholdvelcpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEvolv) <br> |
|  \_\_global\_\_ void | [**TheresholdVelGPU**](#function-theresholdvelgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv) <br>_CUDA kernel for enforcing a velocity threshold. Function Used to prevent crazy velocity on the GPU. The function wraps the main function for the GPU. Updates velocity components to ensure they do not exceed a specified threshold._  |
|  template \_\_global\_\_ void | [**TheresholdVelGPU&lt; double &gt;**](#function-theresholdvelgpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEvolv) <br> |
|  template \_\_global\_\_ void | [**TheresholdVelGPU&lt; float &gt;**](#function-theresholdvelgpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEvolv) <br> |
|  \_\_host\_\_ \_\_device\_\_ bool | [**ThresholdVelocity**](#function-thresholdvelocity) (T Threshold, T & u, T & v) <br>_Function Used to prevent crazy velocity._  |
|  \_\_host\_\_ void | [**XiafrictionCPU**](#function-xiafrictioncpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* cf, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv\_o) <br>_CPU routine for applying bottom friction following Xia & Lang (2018)._  |
|  template \_\_host\_\_ void | [**XiafrictionCPU&lt; double &gt;**](#function-xiafrictioncpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double dt, double \* cf, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEvolv, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEvolv\_o) <br> |
|  template \_\_host\_\_ void | [**XiafrictionCPU&lt; float &gt;**](#function-xiafrictioncpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float dt, float \* cf, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEvolv, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEvolv\_o) <br> |
|  \_\_global\_\_ void | [**XiafrictionGPU**](#function-xiafrictiongpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* cf, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv\_o) <br>_CUDA kernel for applying Xia & Lang (2018) bottom friction to all active blocks._  |
|  template \_\_global\_\_ void | [**XiafrictionGPU&lt; double &gt;**](#function-xiafrictiongpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double dt, double \* cf, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEvolv, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEvolv\_o) <br> |
|  template \_\_global\_\_ void | [**XiafrictionGPU&lt; float &gt;**](#function-xiafrictiongpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float dt, float \* cf, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEvolv, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEvolv\_o) <br> |
|  \_\_host\_\_ void | [**bottomfrictionCPU**](#function-bottomfrictioncpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* cf, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv) <br>_CPU routine for applying bottom friction to all active blocks._  |
|  template \_\_host\_\_ void | [**bottomfrictionCPU&lt; double &gt;**](#function-bottomfrictioncpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double dt, double \* cf, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEvolv) <br> |
|  template \_\_host\_\_ void | [**bottomfrictionCPU&lt; float &gt;**](#function-bottomfrictioncpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float dt, float \* cf, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEvolv) <br> |
|  \_\_global\_\_ void | [**bottomfrictionGPU**](#function-bottomfrictiongpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* cf, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv) <br>_CUDA kernel for applying bottom friction to all active blocks._  |
|  template \_\_global\_\_ void | [**bottomfrictionGPU&lt; double &gt;**](#function-bottomfrictiongpu-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double dt, double \* cf, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEvolv) <br> |
|  template \_\_global\_\_ void | [**bottomfrictionGPU&lt; float &gt;**](#function-bottomfrictiongpu-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float dt, float \* cf, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEvolv) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**manningfriction**](#function-manningfriction) (T g, T hi, T n) <br>_Manning friction model._  |
|  \_\_host\_\_ \_\_device\_\_ T | [**smartfriction**](#function-smartfriction) (T hi, T zo) <br>_Smart friction model for roughness height._  |




























## Public Functions Documentation




### function TheresholdVelCPU 

_CPU routine for enforcing a velocity threshold._ 
```C++
template<class T>
__host__ void TheresholdVelCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEvolv
) 
```



Updates velocity components to ensure they do not exceed a specified threshold. Function Used to prevent crazy velocity on the CPU




**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `XEvolv` Evolving variables structure 




        

<hr>



### function TheresholdVelCPU&lt; double &gt; 

```C++
template __host__ void TheresholdVelCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEvolv
) 
```




<hr>



### function TheresholdVelCPU&lt; float &gt; 

```C++
template __host__ void TheresholdVelCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEvolv
) 
```




<hr>



### function TheresholdVelGPU 

_CUDA kernel for enforcing a velocity threshold. Function Used to prevent crazy velocity on the GPU. The function wraps the main function for the GPU. Updates velocity components to ensure they do not exceed a specified threshold._ 
```C++
template<class T>
__global__ void TheresholdVelGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEvolv
) 
```





**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `XEvolv` Evolving variables structure 




        

<hr>



### function TheresholdVelGPU&lt; double &gt; 

```C++
template __global__ void TheresholdVelGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > XEvolv
) 
```




<hr>



### function TheresholdVelGPU&lt; float &gt; 

```C++
template __global__ void TheresholdVelGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > XEvolv
) 
```




<hr>



### function ThresholdVelocity 

_Function Used to prevent crazy velocity._ 
```C++
template<class T>
__host__ __device__ bool ThresholdVelocity (
    T Threshold,
    T & u,
    T & v
) 
```



The function scale velocities so it doesn't exceeds a given threshold. Default threshold is/should be 16.0m/s




**Parameters:**


* `Threshold` Velocity threshold 
* `u` Velocity component in x direction 
* `v` Velocity component in y direction 



**Returns:**

true if velocity was above threshold and has been scaled down, false otherwise 





        

<hr>



### function XiafrictionCPU 

_CPU routine for applying bottom friction following Xia & Lang (2018)._ 
```C++
template<class T>
__host__ void XiafrictionCPU (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    T * cf,
    EvolvingP < T > XEvolv,
    EvolvingP < T > XEvolv_o
) 
```



Updates velocity components using the Xia & Lang friction model for each cell in all blocks, using both current and previous evolving variables. Reference: Xia and Lang (2018), [https://doi.org/10.1016/j.advwatres.2018.05.004](https://doi.org/10.1016/j.advwatres.2018.05.004)




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `dt` Time step 
* `cf` Friction coefficient array 
* `XEvolv` Current evolving variables structure 
* `XEvolv_o` Previous evolving variables structure 




        

<hr>



### function XiafrictionCPU&lt; double &gt; 

```C++
template __host__ void XiafrictionCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double dt,
    double * cf,
    EvolvingP < double > XEvolv,
    EvolvingP < double > XEvolv_o
) 
```




<hr>



### function XiafrictionCPU&lt; float &gt; 

```C++
template __host__ void XiafrictionCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float dt,
    float * cf,
    EvolvingP < float > XEvolv,
    EvolvingP < float > XEvolv_o
) 
```




<hr>



### function XiafrictionGPU 

_CUDA kernel for applying Xia & Lang (2018) bottom friction to all active blocks._ 
```C++
template<class T>
__global__ void XiafrictionGPU (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    T * cf,
    EvolvingP < T > XEvolv,
    EvolvingP < T > XEvolv_o
) 
```



Updates velocity components using the Xia & Lang friction model for each cell in all blocks, using both current and previous evolving variables.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `dt` Time step 
* `cf` Friction coefficient array 
* `XEvolv` Current evolving variables structure 
* `XEvolv_o` Previous evolving variables structure 




        

<hr>



### function XiafrictionGPU&lt; double &gt; 

```C++
template __global__ void XiafrictionGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double dt,
    double * cf,
    EvolvingP < double > XEvolv,
    EvolvingP < double > XEvolv_o
) 
```




<hr>



### function XiafrictionGPU&lt; float &gt; 

```C++
template __global__ void XiafrictionGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float dt,
    float * cf,
    EvolvingP < float > XEvolv,
    EvolvingP < float > XEvolv_o
) 
```




<hr>



### function bottomfrictionCPU 

_CPU routine for applying bottom friction to all active blocks._ 
```C++
template<class T>
__host__ void bottomfrictionCPU (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    T * cf,
    EvolvingP < T > XEvolv
) 
```



Updates velocity components using the specified friction model (default, smart, or Manning) for each cell in all blocks.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `dt` Time step 
* `cf` Friction coefficient array 
* `XEvolv` Evolving variables structure 




        

<hr>



### function bottomfrictionCPU&lt; double &gt; 

```C++
template __host__ void bottomfrictionCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double dt,
    double * cf,
    EvolvingP < double > XEvolv
) 
```




<hr>



### function bottomfrictionCPU&lt; float &gt; 

```C++
template __host__ void bottomfrictionCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float dt,
    float * cf,
    EvolvingP < float > XEvolv
) 
```




<hr>



### function bottomfrictionGPU 

_CUDA kernel for applying bottom friction to all active blocks._ 
```C++
template<class T>
__global__ void bottomfrictionGPU (
    Param XParam,
    BlockP < T > XBlock,
    T dt,
    T * cf,
    EvolvingP < T > XEvolv
) 
```



Updates velocity components using the specified friction model (default, smart, or Manning) for each cell in all blocks.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block data structure 
* `dt` Time step 
* `cf` Friction coefficient array 
* `XEvolv` Evolving variables structure 




        

<hr>



### function bottomfrictionGPU&lt; double &gt; 

```C++
template __global__ void bottomfrictionGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double dt,
    double * cf,
    EvolvingP < double > XEvolv
) 
```




<hr>



### function bottomfrictionGPU&lt; float &gt; 

```C++
template __global__ void bottomfrictionGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float dt,
    float * cf,
    EvolvingP < float > XEvolv
) 
```




<hr>



### function manningfriction 

_Manning friction model._ 
```C++
template<class T>
__host__ __device__ T manningfriction (
    T g,
    T hi,
    T n
) 
```



Computes friction coefficient using Manning's equation based on gravity, water depth, and Manning's n.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `g` Gravity 
* `hi` Water depth 
* `n` Manning's n 



**Returns:**

Friction coefficient 





        

<hr>



### function smartfriction 

_Smart friction model for roughness height._ 
```C++
template<class T>
__host__ __device__ T smartfriction (
    T hi,
    T zo
) 
```



Computes friction coefficient using a log-law based on water depth and roughness height.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `hi` Water depth 
* `zo` Roughness height 



**Returns:**

Friction coefficient 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Friction.cu`


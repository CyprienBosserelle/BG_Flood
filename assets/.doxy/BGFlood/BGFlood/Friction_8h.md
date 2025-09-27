

# File Friction.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Friction.h**](Friction_8h.md)

[Go to the source code of this file](Friction_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "MemManagement.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**TheresholdVelCPU**](#function-theresholdvelcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv) <br>_Function Used to prevent crazy velocity on the CPU._  |
|  \_\_global\_\_ void | [**TheresholdVelGPU**](#function-theresholdvelgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv) <br>_Function Used to prevent crazy velocity on the GPU._  |
|  \_\_host\_\_ \_\_device\_\_ bool | [**ThresholdVelocity**](#function-thresholdvelocity) (T Threshold, T & u, T & v) <br>_Function Used to prevent crazy velocity._  |
|  \_\_host\_\_ void | [**XiafrictionCPU**](#function-xiafrictioncpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* cf, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv\_o) <br> |
|  \_\_global\_\_ void | [**XiafrictionGPU**](#function-xiafrictiongpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* cf, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv\_o) <br> |
|  \_\_host\_\_ void | [**bottomfrictionCPU**](#function-bottomfrictioncpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* cf, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv) <br> |
|  \_\_global\_\_ void | [**bottomfrictionGPU**](#function-bottomfrictiongpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T dt, T \* cf, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**manningfriction**](#function-manningfriction) (T g, T hi, T n) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**smartfriction**](#function-smartfriction) (T hi, T zo) <br> |




























## Public Functions Documentation




### function TheresholdVelCPU 

_Function Used to prevent crazy velocity on the CPU._ 
```C++
template<class T>
__host__ void TheresholdVelCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEvolv
) 
```



The function wraps teh main functio for the CPU. 


        

<hr>



### function TheresholdVelGPU 

_Function Used to prevent crazy velocity on the GPU._ 
```C++
template<class T>
__global__ void TheresholdVelGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEvolv
) 
```



The function wraps the main function for the GPU. 


        

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


        

<hr>



### function XiafrictionCPU 

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




<hr>



### function XiafrictionGPU 

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




<hr>



### function bottomfrictionCPU 

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




<hr>



### function bottomfrictionGPU 

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




<hr>



### function manningfriction 

```C++
template<class T>
__host__ __device__ T manningfriction (
    T g,
    T hi,
    T n
) 
```




<hr>



### function smartfriction 

```C++
template<class T>
__host__ __device__ T smartfriction (
    T hi,
    T zo
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Friction.h`


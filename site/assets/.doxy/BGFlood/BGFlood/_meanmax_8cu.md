

# File Meanmax.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Meanmax.cu**](_meanmax_8cu.md)

[Go to the source code of this file](_meanmax_8cu_source.md)



* `#include "Meanmax.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Calcmeanmax**](#function-calcmeanmax) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; & XLoop, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Calculate mean and/or max of requested variables on GPU/CPU._  |
|  template void | [**Calcmeanmax&lt; double &gt;**](#function-calcmeanmax-double) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; double &gt; & XLoop, [**Model**](struct_model.md)&lt; double &gt; XModel, [**Model**](struct_model.md)&lt; double &gt; XModel\_g) <br> |
|  template void | [**Calcmeanmax&lt; float &gt;**](#function-calcmeanmax-float) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; float &gt; & XLoop, [**Model**](struct_model.md)&lt; float &gt; XModel, [**Model**](struct_model.md)&lt; float &gt; XModel\_g) <br> |
|  void | [**Initmeanmax**](#function-initmeanmax) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; XLoop, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Initialize mean and max statistics at the start of the simulation._  |
|  template void | [**Initmeanmax&lt; double &gt;**](#function-initmeanmax-double) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; double &gt; XLoop, [**Model**](struct_model.md)&lt; double &gt; XModel, [**Model**](struct_model.md)&lt; double &gt; XModel\_g) <br> |
|  template void | [**Initmeanmax&lt; float &gt;**](#function-initmeanmax-float) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; float &gt; XLoop, [**Model**](struct_model.md)&lt; float &gt; XModel, [**Model**](struct_model.md)&lt; float &gt; XModel\_g) <br> |
|  \_\_host\_\_ void | [**addUandhU\_CPU**](#function-adduandhu_cpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* u, T \* v, T \* U, T \* hU) <br>_Compute velocity magnitude and hU product on the CPU._  |
|  \_\_global\_\_ void | [**addUandhU\_GPU**](#function-adduandhu_gpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* u, T \* v, T \* U, T \* hU) <br>_CUDA kernel to compute velocity magnitude and hU product._  |
|  \_\_host\_\_ void | [**addavg\_varCPU**](#function-addavg_varcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Varmean, T \* Var) <br>_Accumulate mean values for a variable on the CPU._  |
|  \_\_global\_\_ void | [**addavg\_varGPU**](#function-addavg_vargpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Varmean, T \* Var) <br>_CUDA kernel to accumulate mean values for a variable._  |
|  \_\_host\_\_ void | [**addwettime\_CPU**](#function-addwettime_cpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* wett, T \* h, T thresold, T time) <br>_Accumulate wet duration for each cell on the CPU._  |
|  \_\_global\_\_ void | [**addwettime\_GPU**](#function-addwettime_gpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* wett, T \* h, T thresold, T time) <br>_CUDA kernel to accumulate wet duration for each cell._  |
|  \_\_host\_\_ void | [**divavg\_varCPU**](#function-divavg_varcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T ntdiv, T \* Varmean) <br>_Divide mean values by the number of steps on the CPU._  |
|  \_\_global\_\_ void | [**divavg\_varGPU**](#function-divavg_vargpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T ntdiv, T \* Varmean) <br>_CUDA kernel to divide mean values by the number of steps._  |
|  \_\_host\_\_ void | [**max\_Norm\_CPU**](#function-max_norm_cpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var1, T \* Var2) <br>_Compute max velocity magnitude on the CPU._  |
|  \_\_global\_\_ void | [**max\_Norm\_GPU**](#function-max_norm_gpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var1, T \* Var2) <br>_CUDA kernel to compute max velocity magnitude._  |
|  \_\_host\_\_ void | [**max\_hU\_CPU**](#function-max_hu_cpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Varmax, T \* h, T \* u, T \* v) <br>_Compute max hU value on the CPU._  |
|  \_\_global\_\_ void | [**max\_hU\_GPU**](#function-max_hu_gpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Varmax, T \* h, T \* u, T \* v) <br>_CUDA kernel to compute max hU value._  |
|  \_\_host\_\_ void | [**max\_varCPU**](#function-max_varcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var) <br>_Compute max value for a variable on the CPU._  |
|  \_\_global\_\_ void | [**max\_varGPU**](#function-max_vargpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var) <br>_CUDA kernel to compute max value for a variable._  |
|  void | [**resetmaxCPU**](#function-resetmaxcpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; T &gt; & XEv) <br>_Reset max statistics arrays on the CPU._  |
|  void | [**resetmaxGPU**](#function-resetmaxgpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; T &gt; & XEv) <br>_Reset max statistics arrays on the GPU._  |
|  void | [**resetmeanCPU**](#function-resetmeancpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; T &gt; & XEv) <br>_Reset mean statistics arrays on the CPU._  |
|  template void | [**resetmeanCPU&lt; double &gt;**](#function-resetmeancpu-double) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; double &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; double &gt; & XEv) <br> |
|  template void | [**resetmeanCPU&lt; float &gt;**](#function-resetmeancpu-float) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; float &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; float &gt; & XEv) <br> |
|  void | [**resetmeanGPU**](#function-resetmeangpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; T &gt; & XEv) <br>_Reset mean statistics arrays on the GPU._  |
|  template void | [**resetmeanGPU&lt; double &gt;**](#function-resetmeangpu-double) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; double &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; double &gt; & XEv) <br> |
|  template void | [**resetmeanGPU&lt; float &gt;**](#function-resetmeangpu-float) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; float &gt; XLoop, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; float &gt; & XEv) <br> |
|  void | [**resetmeanmax**](#function-resetmeanmax) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; & XLoop, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Reset mean and/or max statistics at output steps._  |
|  template void | [**resetmeanmax&lt; double &gt;**](#function-resetmeanmax-double) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; double &gt; & XLoop, [**Model**](struct_model.md)&lt; double &gt; XModel, [**Model**](struct_model.md)&lt; double &gt; XModel\_g) <br> |
|  template void | [**resetmeanmax&lt; float &gt;**](#function-resetmeanmax-float) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; float &gt; & XLoop, [**Model**](struct_model.md)&lt; float &gt; XModel, [**Model**](struct_model.md)&lt; float &gt; XModel\_g) <br> |
|  void | [**resetvalCPU**](#function-resetvalcpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& var, T val) <br>_Reset a variable array to a specified value on the CPU._  |
|  template void | [**resetvalCPU&lt; double &gt;**](#function-resetvalcpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \*& var, double val) <br> |
|  template void | [**resetvalCPU&lt; float &gt;**](#function-resetvalcpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \*& var, float val) <br> |
|  void | [**resetvalGPU**](#function-resetvalgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& var, T val) <br>_Reset a variable array to a specified value on the GPU._  |
|  template void | [**resetvalGPU&lt; double &gt;**](#function-resetvalgpu-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \*& var, double val) <br> |
|  template void | [**resetvalGPU&lt; float &gt;**](#function-resetvalgpu-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \*& var, float val) <br> |




























## Public Functions Documentation




### function Calcmeanmax 

_Calculate mean and/or max of requested variables on GPU/CPU._ 
```C++
template<class T>
void Calcmeanmax (
    Param XParam,
    Loop < T > & XLoop,
    Model < T > XModel,
    Model < T > XModel_g
) 
```



Computes mean, max, and wet duration statistics for evolving variables, handling both CPU and GPU execution paths.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XLoop` [**Loop**](struct_loop.md) control structure 
* `XModel` [**Model**](struct_model.md) state (CPU) 
* `XModel_g` [**Model**](struct_model.md) state (GPU) 




        

<hr>



### function Calcmeanmax&lt; double &gt; 

```C++
template void Calcmeanmax< double > (
    Param XParam,
    Loop < double > & XLoop,
    Model < double > XModel,
    Model < double > XModel_g
) 
```




<hr>



### function Calcmeanmax&lt; float &gt; 

```C++
template void Calcmeanmax< float > (
    Param XParam,
    Loop < float > & XLoop,
    Model < float > XModel,
    Model < float > XModel_g
) 
```




<hr>



### function Initmeanmax 

_Initialize mean and max statistics at the start of the simulation._ 
```C++
template<class T>
void Initmeanmax (
    Param XParam,
    Loop < T > XLoop,
    Model < T > XModel,
    Model < T > XModel_g
) 
```



Sets up/reset statistics arrays for the initial simulation step.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XLoop` [**Loop**](struct_loop.md) control structure 
* `XModel` [**Model**](struct_model.md) state (CPU) 
* `XModel_g` [**Model**](struct_model.md) state (GPU) 




        

<hr>



### function Initmeanmax&lt; double &gt; 

```C++
template void Initmeanmax< double > (
    Param XParam,
    Loop < double > XLoop,
    Model < double > XModel,
    Model < double > XModel_g
) 
```




<hr>



### function Initmeanmax&lt; float &gt; 

```C++
template void Initmeanmax< float > (
    Param XParam,
    Loop < float > XLoop,
    Model < float > XModel,
    Model < float > XModel_g
) 
```




<hr>



### function addUandhU\_CPU 

_Compute velocity magnitude and hU product on the CPU._ 
```C++
template<class T>
__host__ void addUandhU_CPU (
    Param XParam,
    BlockP < T > XBlock,
    T * h,
    T * u,
    T * v,
    T * U,
    T * hU
) 
```



Calculates the velocity magnitude and its product with water depth for each cell on the CPU.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `h` Water depth array 
* `u` U-velocity array 
* `v` V-velocity array 
* `U` Output velocity magnitude array 
* `hU` Output hU product array 




        

<hr>



### function addUandhU\_GPU 

_CUDA kernel to compute velocity magnitude and hU product._ 
```C++
template<class T>
__global__ void addUandhU_GPU (
    Param XParam,
    BlockP < T > XBlock,
    T * h,
    T * u,
    T * v,
    T * U,
    T * hU
) 
```



Calculates the velocity magnitude and its product with water depth for each cell on the GPU.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `h` Water depth array 
* `u` U-velocity array 
* `v` V-velocity array 
* `U` Output velocity magnitude array 
* `hU` Output hU product array 




        

<hr>



### function addavg\_varCPU 

_Accumulate mean values for a variable on the CPU._ 
```C++
template<class T>
__host__ void addavg_varCPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmean,
    T * Var
) 
```



Adds the current value to the running mean for each cell on the CPU.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `Varmean` Mean variable array 
* `Var` Source variable array 




        

<hr>



### function addavg\_varGPU 

_CUDA kernel to accumulate mean values for a variable._ 
```C++
template<class T>
__global__ void addavg_varGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmean,
    T * Var
) 
```



Adds the current value to the running mean for each cell on the GPU.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `Varmean` Mean variable array 
* `Var` Source variable array 




        

<hr>



### function addwettime\_CPU 

_Accumulate wet duration for each cell on the CPU._ 
```C++
template<class T>
__host__ void addwettime_CPU (
    Param XParam,
    BlockP < T > XBlock,
    T * wett,
    T * h,
    T thresold,
    T time
) 
```



Adds time to the wet duration for cells where water depth exceeds a threshold on the CPU.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `wett` Wet duration array 
* `h` Water depth array 
* `thresold` Wet threshold value 
* `time` Time increment 




        

<hr>



### function addwettime\_GPU 

_CUDA kernel to accumulate wet duration for each cell._ 
```C++
template<class T>
__global__ void addwettime_GPU (
    Param XParam,
    BlockP < T > XBlock,
    T * wett,
    T * h,
    T thresold,
    T time
) 
```



Adds time to the wet duration for cells where water depth exceeds a threshold on the GPU.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `wett` Wet duration array 
* `h` Water depth array 
* `thresold` Wet threshold value 
* `time` Time increment 




        

<hr>



### function divavg\_varCPU 

_Divide mean values by the number of steps on the CPU._ 
```C++
template<class T>
__host__ void divavg_varCPU (
    Param XParam,
    BlockP < T > XBlock,
    T ntdiv,
    T * Varmean
) 
```



Finalizes the mean calculation by dividing accumulated values by the number of time steps on the CPU.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `ntdiv` Number of time steps 
* `Varmean` Mean variable array 




        

<hr>



### function divavg\_varGPU 

_CUDA kernel to divide mean values by the number of steps._ 
```C++
template<class T>
__global__ void divavg_varGPU (
    Param XParam,
    BlockP < T > XBlock,
    T ntdiv,
    T * Varmean
) 
```



Finalizes the mean calculation by dividing accumulated values by the number of time steps on the GPU.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `ntdiv` Number of time steps 
* `Varmean` Mean variable array 




        

<hr>



### function max\_Norm\_CPU 

_Compute max velocity magnitude on the CPU._ 
```C++
template<class T>
__host__ void max_Norm_CPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * Var1,
    T * Var2
) 
```



Updates the max velocity magnitude for each cell by comparing with the current value using CPU routines.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `Varmax` Max variable array 
* `Var1` U-velocity array 
* `Var2` V-velocity array 




        

<hr>



### function max\_Norm\_GPU 

_CUDA kernel to compute max velocity magnitude._ 
```C++
template<class T>
__global__ void max_Norm_GPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * Var1,
    T * Var2
) 
```



Updates the max velocity magnitude for each cell by comparing with the current value on the GPU.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `Varmax` Max variable array 
* `Var1` U-velocity array 
* `Var2` V-velocity array 




        

<hr>



### function max\_hU\_CPU 

_Compute max hU value on the CPU._ 
```C++
template<class T>
__host__ void max_hU_CPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * h,
    T * u,
    T * v
) 
```



Updates the max hU value for each cell by comparing with the current value using CPU routines.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `Varmax` Max variable array 
* `h` Water depth array 
* `u` U-velocity array 
* `v` V-velocity array 




        

<hr>



### function max\_hU\_GPU 

_CUDA kernel to compute max hU value._ 
```C++
template<class T>
__global__ void max_hU_GPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * h,
    T * u,
    T * v
) 
```



Updates the max hU value for each cell by comparing with the current value on the GPU.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `Varmax` Max variable array 
* `h` Water depth array 
* `u` U-velocity array 
* `v` V-velocity array 




        

<hr>



### function max\_varCPU 

_Compute max value for a variable on the CPU._ 
```C++
template<class T>
__host__ void max_varCPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * Var
) 
```



Updates the max value for each cell by comparing with the current value using CPU routines.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `Varmax` Max variable array 
* `Var` Source variable array 




        

<hr>



### function max\_varGPU 

_CUDA kernel to compute max value for a variable._ 
```C++
template<class T>
__global__ void max_varGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * Var
) 
```



Updates the max value for each cell by comparing with the current value on the GPU.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `Varmax` Max variable array 
* `Var` Source variable array 




        

<hr>



### function resetmaxCPU 

_Reset max statistics arrays on the CPU._ 
```C++
template<class T>
void resetmaxCPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    EvolvingP_M < T > & XEv
) 
```



Sets all max arrays to a large negative value using CPU routines.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XLoop` [**Loop**](struct_loop.md) control structure 
* `XBlock` Block data 
* `XEv` Max evolving variables 




        

<hr>



### function resetmaxGPU 

_Reset max statistics arrays on the GPU._ 
```C++
template<class T>
void resetmaxGPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    EvolvingP_M < T > & XEv
) 
```



Sets all max arrays to a large negative value using a CUDA kernel.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XLoop` [**Loop**](struct_loop.md) control structure 
* `XBlock` Block data 
* `XEv` Max evolving variables 




        

<hr>



### function resetmeanCPU 

_Reset mean statistics arrays on the CPU._ 
```C++
template<class T>
void resetmeanCPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    EvolvingP_M < T > & XEv
) 
```



Sets all mean arrays to zero using CPU routines.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XLoop` [**Loop**](struct_loop.md) control structure 
* `XBlock` Block data 
* `XEv` Mean evolving variables 




        

<hr>



### function resetmeanCPU&lt; double &gt; 

```C++
template void resetmeanCPU< double > (
    Param XParam,
    Loop < double > XLoop,
    BlockP < double > XBlock,
    EvolvingP_M < double > & XEv
) 
```




<hr>



### function resetmeanCPU&lt; float &gt; 

```C++
template void resetmeanCPU< float > (
    Param XParam,
    Loop < float > XLoop,
    BlockP < float > XBlock,
    EvolvingP_M < float > & XEv
) 
```




<hr>



### function resetmeanGPU 

_Reset mean statistics arrays on the GPU._ 
```C++
template<class T>
void resetmeanGPU (
    Param XParam,
    Loop < T > XLoop,
    BlockP < T > XBlock,
    EvolvingP_M < T > & XEv
) 
```



Sets all mean arrays to zero using a CUDA kernel.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XLoop` [**Loop**](struct_loop.md) control structure 
* `XBlock` Block data 
* `XEv` Mean evolving variables 




        

<hr>



### function resetmeanGPU&lt; double &gt; 

```C++
template void resetmeanGPU< double > (
    Param XParam,
    Loop < double > XLoop,
    BlockP < double > XBlock,
    EvolvingP_M < double > & XEv
) 
```




<hr>



### function resetmeanGPU&lt; float &gt; 

```C++
template void resetmeanGPU< float > (
    Param XParam,
    Loop < float > XLoop,
    BlockP < float > XBlock,
    EvolvingP_M < float > & XEv
) 
```




<hr>



### function resetmeanmax 

_Reset mean and/or max statistics at output steps._ 
```C++
template<class T>
void resetmeanmax (
    Param XParam,
    Loop < T > & XLoop,
    Model < T > XModel,
    Model < T > XModel_g
) 
```



Resets mean, max, and wet duration arrays after output is produced, handling both CPU and GPU paths.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XLoop` [**Loop**](struct_loop.md) control structure 
* `XModel` [**Model**](struct_model.md) state (CPU) 
* `XModel_g` [**Model**](struct_model.md) state (GPU) 




        

<hr>



### function resetmeanmax&lt; double &gt; 

```C++
template void resetmeanmax< double > (
    Param XParam,
    Loop < double > & XLoop,
    Model < double > XModel,
    Model < double > XModel_g
) 
```




<hr>



### function resetmeanmax&lt; float &gt; 

```C++
template void resetmeanmax< float > (
    Param XParam,
    Loop < float > & XLoop,
    Model < float > XModel,
    Model < float > XModel_g
) 
```




<hr>



### function resetvalCPU 

_Reset a variable array to a specified value on the CPU._ 
```C++
template<class T>
void resetvalCPU (
    Param XParam,
    BlockP < T > XBlock,
    T *& var,
    T val
) 
```



Sets all elements of the array to the given value using CPU routines.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `var` Variable array to reset 
* `val` Value to set 




        

<hr>



### function resetvalCPU&lt; double &gt; 

```C++
template void resetvalCPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double *& var,
    double val
) 
```




<hr>



### function resetvalCPU&lt; float &gt; 

```C++
template void resetvalCPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float *& var,
    float val
) 
```




<hr>



### function resetvalGPU 

_Reset a variable array to a specified value on the GPU._ 
```C++
template<class T>
void resetvalGPU (
    Param XParam,
    BlockP < T > XBlock,
    T *& var,
    T val
) 
```



Sets all elements of the array to the given value using a CUDA kernel.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data 
* `var` Variable array to reset 
* `val` Value to set 




        

<hr>



### function resetvalGPU&lt; double &gt; 

```C++
template void resetvalGPU< double > (
    Param XParam,
    BlockP < double > XBlock,
    double *& var,
    double val
) 
```




<hr>



### function resetvalGPU&lt; float &gt; 

```C++
template void resetvalGPU< float > (
    Param XParam,
    BlockP < float > XBlock,
    float *& var,
    float val
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Meanmax.cu`


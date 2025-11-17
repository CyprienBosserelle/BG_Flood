

# File Meanmax.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Meanmax.h**](_meanmax_8h.md)

[Go to the source code of this file](_meanmax_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "MemManagement.h"`
* `#include "FlowGPU.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Calcmeanmax**](#function-calcmeanmax) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; & XLoop, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Calculate mean and/or max of requested variables on GPU/CPU._  |
|  void | [**Initmeanmax**](#function-initmeanmax) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; XLoop, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Initialize mean and max statistics at the start of the simulation._  |
|  \_\_global\_\_ void | [**addUandhU\_GPU**](#function-adduandhu_gpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* h, T \* u, T \* v, T \* U, T \* hU) <br>_CUDA kernel to compute velocity magnitude and hU product._  |
|  \_\_global\_\_ void | [**addavg\_varGPU**](#function-addavg_vargpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Varmean, T \* Var) <br>_CUDA kernel to accumulate mean values for a variable._  |
|  \_\_global\_\_ void | [**addwettime\_GPU**](#function-addwettime_gpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* wett, T \* h, T thresold, T time) <br>_CUDA kernel to accumulate wet duration for each cell._  |
|  \_\_global\_\_ void | [**divavg\_varGPU**](#function-divavg_vargpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T ntdiv, T \* Varmean) <br>_CUDA kernel to divide mean values by the number of steps._  |
|  \_\_global\_\_ void | [**max\_Norm\_GPU**](#function-max_norm_gpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var1, T \* Var2) <br>_CUDA kernel to compute max velocity magnitude._  |
|  \_\_global\_\_ void | [**max\_hU\_GPU**](#function-max_hu_gpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var1, T \* Var2, T \* Var3) <br>_CUDA kernel to compute max hU value._  |
|  \_\_global\_\_ void | [**max\_varGPU**](#function-max_vargpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Varmax, T \* Var) <br>_CUDA kernel to compute max value for a variable._  |
|  void | [**resetmeanmax**](#function-resetmeanmax) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; & XLoop, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Reset mean and/or max statistics at output steps._  |
|  void | [**resetvalGPU**](#function-resetvalgpu) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& var, T val) <br>_Reset a variable array to a specified value on the GPU._  |




























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



### function max\_hU\_GPU 

_CUDA kernel to compute max hU value._ 
```C++
template<class T>
__global__ void max_hU_GPU (
    Param XParam,
    BlockP < T > XBlock,
    T * Varmax,
    T * Var1,
    T * Var2,
    T * Var3
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

------------------------------
The documentation for this class was generated from the following file `src/Meanmax.h`


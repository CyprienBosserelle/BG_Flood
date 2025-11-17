

# File FlowMLGPU.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**FlowMLGPU.cu**](_flow_m_l_g_p_u_8cu.md)

[Go to the source code of this file](_flow_m_l_g_p_u_8cu_source.md)



* `#include "FlowMLGPU.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**FlowMLGPU**](#function-flowmlgpu) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; T &gt; & XLoop, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; XModel) <br> |
|  template void | [**FlowMLGPU&lt; double &gt;**](#function-flowmlgpu-double) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; double &gt; & XLoop, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; double &gt; XModel) <br> |
|  template void | [**FlowMLGPU&lt; float &gt;**](#function-flowmlgpu-float) ([**Param**](class_param.md) XParam, [**Loop**](struct_loop.md)&lt; float &gt; & XLoop, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; float &gt; XModel) <br> |




























## Public Functions Documentation




### function FlowMLGPU 

```C++
template<class T>
void FlowMLGPU (
    Param XParam,
    Loop < T > & XLoop,
    Forcing < float > XForcing,
    Model < T > XModel
) 
```




<hr>



### function FlowMLGPU&lt; double &gt; 

```C++
template void FlowMLGPU< double > (
    Param XParam,
    Loop < double > & XLoop,
    Forcing < float > XForcing,
    Model < double > XModel
) 
```




<hr>



### function FlowMLGPU&lt; float &gt; 

```C++
template void FlowMLGPU< float > (
    Param XParam,
    Loop < float > & XLoop,
    Forcing < float > XForcing,
    Model < float > XModel
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/FlowMLGPU.cu`


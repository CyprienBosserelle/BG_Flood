

# File FlowCPU.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**FlowCPU.cu**](FlowCPU_8cu.md)

[Go to the source code of this file](FlowCPU_8cu_source.md)



* `#include "FlowCPU.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**FlowCPU**](#function-flowcpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  template void | [**FlowCPU&lt; double &gt;**](#function-flowcpu-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template void | [**FlowCPU&lt; float &gt;**](#function-flowcpu-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |
|  void | [**HalfStepCPU**](#function-halfstepcpu) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  template void | [**HalfStepCPU&lt; double &gt;**](#function-halfstepcpu-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template void | [**HalfStepCPU&lt; float &gt;**](#function-halfstepcpu-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; & XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |




























## Public Functions Documentation




### function FlowCPU 

```C++
template<class T>
void FlowCPU (
    Param XParam,
    Loop < T > & XLoop,
    Forcing < float > XForcing,
    Model < T > XModel
) 
```




<hr>



### function FlowCPU&lt; double &gt; 

```C++
template void FlowCPU< double > (
    Param XParam,
    Loop < double > & XLoop,
    Forcing < float > XForcing,
    Model < double > XModel
) 
```




<hr>



### function FlowCPU&lt; float &gt; 

```C++
template void FlowCPU< float > (
    Param XParam,
    Loop < float > & XLoop,
    Forcing < float > XForcing,
    Model < float > XModel
) 
```




<hr>



### function HalfStepCPU 

```C++
template<class T>
void HalfStepCPU (
    Param XParam,
    Loop < T > & XLoop,
    Forcing < float > XForcing,
    Model < T > XModel
) 
```



Debugging flow step This function was crated to debug the main engine of the model 


        

<hr>



### function HalfStepCPU&lt; double &gt; 

```C++
template void HalfStepCPU< double > (
    Param XParam,
    Loop < double > & XLoop,
    Forcing < float > XForcing,
    Model < double > XModel
) 
```




<hr>



### function HalfStepCPU&lt; float &gt; 

```C++
template void HalfStepCPU< float > (
    Param XParam,
    Loop < float > & XLoop,
    Forcing < float > XForcing,
    Model < float > XModel
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/FlowCPU.cu`


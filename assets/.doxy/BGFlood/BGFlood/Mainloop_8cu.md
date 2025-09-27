

# File Mainloop.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Mainloop.cu**](Mainloop_8cu.md)

[Go to the source code of this file](Mainloop_8cu_source.md)



* `#include "Mainloop.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ void | [**CalcInitdtCPU**](#function-calcinitdtcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv, T \* dtmax) <br> |
|  \_\_global\_\_ void | [**CalcInitdtGPU**](#function-calcinitdtgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvolv, T \* dtmax) <br> |
|  void | [**CrashDetection**](#function-crashdetection) ([**Param**](classParam.md) & XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  void | [**DebugLoop**](#function-debugloop) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel, [**Model**](structModel.md)&lt; T &gt; & XModel\_g) <br> |
|  template void | [**DebugLoop&lt; double &gt;**](#function-debugloop-double) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; double &gt; & XModel, [**Model**](structModel.md)&lt; double &gt; & XModel\_g) <br> |
|  template void | [**DebugLoop&lt; float &gt;**](#function-debugloop-float) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; float &gt; & XModel, [**Model**](structModel.md)&lt; float &gt; & XModel\_g) <br> |
|  [**Loop**](structLoop.md)&lt; T &gt; | [**InitLoop**](#function-initloop) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  void | [**MainLoop**](#function-mainloop) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel, [**Model**](structModel.md)&lt; T &gt; & XModel\_g) <br> |
|  template void | [**MainLoop&lt; double &gt;**](#function-mainloop-double) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; double &gt; & XModel, [**Model**](structModel.md)&lt; double &gt; & XModel\_g) <br> |
|  template void | [**MainLoop&lt; float &gt;**](#function-mainloop-float) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; float &gt; & XModel, [**Model**](structModel.md)&lt; float &gt; & XModel\_g) <br> |
|  \_\_host\_\_ double | [**initdt**](#function-initdt) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  template \_\_host\_\_ double | [**initdt&lt; double &gt;**](#function-initdt-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template \_\_host\_\_ double | [**initdt&lt; float &gt;**](#function-initdt-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |
|  void | [**mapoutput**](#function-mapoutput) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Model**](structModel.md)&lt; T &gt; & XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  void | [**pointoutputstep**](#function-pointoutputstep) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; & XLoop, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  void | [**printstatus**](#function-printstatus) (T totaltime, T dt) <br> |
|  \_\_global\_\_ void | [**storeTSout**](#function-storetsout) ([**Param**](classParam.md) XParam, int noutnodes, int outnode, int istep, int blknode, int inode, int jnode, int \* blkTS, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* store) <br> |
|  void | [**updateBnd**](#function-updatebnd) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |




























## Public Functions Documentation




### function CalcInitdtCPU 

```C++
template<class T>
__host__ void CalcInitdtCPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEvolv,
    T * dtmax
) 
```




<hr>



### function CalcInitdtGPU 

```C++
template<class T>
__global__ void CalcInitdtGPU (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > XEvolv,
    T * dtmax
) 
```




<hr>



### function CrashDetection 

```C++
template<class T>
void CrashDetection (
    Param & XParam,
    Loop < T > XLoop,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>



### function DebugLoop 

```C++
template<class T>
void DebugLoop (
    Param & XParam,
    Forcing < float > XForcing,
    Model < T > & XModel,
    Model < T > & XModel_g
) 
```



Debugging loop This function was crated to debug to properly wrap the debug flow engine of the model 


        

<hr>



### function DebugLoop&lt; double &gt; 

```C++
template void DebugLoop< double > (
    Param & XParam,
    Forcing < float > XForcing,
    Model < double > & XModel,
    Model < double > & XModel_g
) 
```




<hr>



### function DebugLoop&lt; float &gt; 

```C++
template void DebugLoop< float > (
    Param & XParam,
    Forcing < float > XForcing,
    Model < float > & XModel,
    Model < float > & XModel_g
) 
```




<hr>



### function InitLoop 

```C++
template<class T>
Loop < T > InitLoop (
    Param & XParam,
    Model < T > & XModel
) 
```




<hr>



### function MainLoop 

```C++
template<class T>
void MainLoop (
    Param & XParam,
    Forcing < float > XForcing,
    Model < T > & XModel,
    Model < T > & XModel_g
) 
```




<hr>



### function MainLoop&lt; double &gt; 

```C++
template void MainLoop< double > (
    Param & XParam,
    Forcing < float > XForcing,
    Model < double > & XModel,
    Model < double > & XModel_g
) 
```




<hr>



### function MainLoop&lt; float &gt; 

```C++
template void MainLoop< float > (
    Param & XParam,
    Forcing < float > XForcing,
    Model < float > & XModel,
    Model < float > & XModel_g
) 
```




<hr>



### function initdt 

```C++
template<class T>
__host__ double initdt (
    Param XParam,
    Loop < T > XLoop,
    Model < T > XModel
) 
```




<hr>



### function initdt&lt; double &gt; 

```C++
template __host__ double initdt< double > (
    Param XParam,
    Loop < double > XLoop,
    Model < double > XModel
) 
```




<hr>



### function initdt&lt; float &gt; 

```C++
template __host__ double initdt< float > (
    Param XParam,
    Loop < float > XLoop,
    Model < float > XModel
) 
```




<hr>



### function mapoutput 

```C++
template<class T>
void mapoutput (
    Param XParam,
    Loop < T > & XLoop,
    Model < T > & XModel,
    Model < T > XModel_g
) 
```




<hr>



### function pointoutputstep 

```C++
template<class T>
void pointoutputstep (
    Param XParam,
    Loop < T > & XLoop,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>



### function printstatus 

```C++
template<class T>
void printstatus (
    T totaltime,
    T dt
) 
```




<hr>



### function storeTSout 

```C++
template<class T>
__global__ void storeTSout (
    Param XParam,
    int noutnodes,
    int outnode,
    int istep,
    int blknode,
    int inode,
    int jnode,
    int * blkTS,
    EvolvingP < T > XEv,
    T * store
) 
```




<hr>



### function updateBnd 

```C++
template<class T>
void updateBnd (
    Param XParam,
    Loop < T > XLoop,
    Forcing < float > XForcing,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Mainloop.cu`


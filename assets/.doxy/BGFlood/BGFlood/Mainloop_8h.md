

# File Mainloop.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Mainloop.h**](Mainloop_8h.md)

[Go to the source code of this file](Mainloop_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "Mesh.h"`
* `#include "Write_netcdf.h"`
* `#include "InitialConditions.h"`
* `#include "MemManagement.h"`
* `#include "Boundary.h"`
* `#include "FlowGPU.h"`
* `#include "FlowCPU.h"`
* `#include "Meanmax.h"`
* `#include "Updateforcing.h"`
* `#include "FlowMLGPU.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**DebugLoop**](#function-debugloop) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel, [**Model**](structModel.md)&lt; T &gt; & XModel\_g) <br> |
|  [**Loop**](structLoop.md)&lt; T &gt; | [**InitLoop**](#function-initloop) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  void | [**MainLoop**](#function-mainloop) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel, [**Model**](structModel.md)&lt; T &gt; & XModel\_g) <br> |
|  \_\_host\_\_ double | [**initdt**](#function-initdt) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  void | [**printstatus**](#function-printstatus) (T totaltime, T dt) <br> |
|  \_\_global\_\_ void | [**storeTSout**](#function-storetsout) ([**Param**](classParam.md) XParam, int noutnodes, int outnode, int istep, int blknode, int inode, int jnode, int \* blkTS, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEv, T \* store) <br> |




























## Public Functions Documentation




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

------------------------------
The documentation for this class was generated from the following file `src/Mainloop.h`




# File GridManip.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**GridManip.h**](GridManip_8h.md)

[Go to the source code of this file](GridManip_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Util_CPU.h"`
* `#include "Forcing.h"`
* `#include "Arrays.h"`
* `#include "MemManagement.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Copy2CartCPU**](#function-copy2cartcpu) (int nx, int ny, T \* dest, T \* src) <br> |
|  void | [**CopyArrayBUQ**](#function-copyarraybuq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; F &gt; XBlock, T \* source, T \*& dest) <br> |
|  void | [**CopyArrayBUQ**](#function-copyarraybuq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; source, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & dest) <br> |
|  void | [**CopyArrayBUQ**](#function-copyarraybuq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; source, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; & dest) <br> |
|  void | [**InitArrayBUQ**](#function-initarraybuq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; F &gt; XBlock, T initval, T \*& Arr) <br> |
|  void | [**InitBlkBUQ**](#function-initblkbuq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; F &gt; XBlock, T initval, T \*& Arr) <br> |
|  void | [**InterpstepCPU**](#function-interpstepcpu) (int nx, int ny, int hdstep, F totaltime, F hddt, T \*& Ux, T \* Uo, T \* Un) <br> |
|  \_\_global\_\_ void | [**InterpstepGPU**](#function-interpstepgpu) (int nx, int ny, T totaltime, T beforetime, T aftertime, T \* Ux, T \* Uo, T \* Un) <br> |
|  void | [**interp2BUQ**](#function-interp2buq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, std::vector&lt; [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; &gt; forcing, T \* z) <br> |
|  void | [**interp2BUQ**](#function-interp2buq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, F forcing, T \*& z) <br> |
|  T | [**interp2BUQ**](#function-interp2buq) (T x, T y, F forcing) <br> |
|  T | [**interp2BUQ**](#function-interp2buq) (T x, T y, T dx, F forcing) <br> |
|  void | [**setedges**](#function-setedges) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& zb) <br> |




























## Public Functions Documentation




### function Copy2CartCPU 

```C++
template<class T>
void Copy2CartCPU (
    int nx,
    int ny,
    T * dest,
    T * src
) 
```




<hr>



### function CopyArrayBUQ 

```C++
template<class T, class F>
void CopyArrayBUQ (
    Param XParam,
    BlockP < F > XBlock,
    T * source,
    T *& dest
) 
```




<hr>



### function CopyArrayBUQ 

```C++
template<class T>
void CopyArrayBUQ (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > source,
    EvolvingP < T > & dest
) 
```




<hr>



### function CopyArrayBUQ 

```C++
template<class T>
void CopyArrayBUQ (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > source,
    EvolvingP_M < T > & dest
) 
```




<hr>



### function InitArrayBUQ 

```C++
template<class T, class F>
void InitArrayBUQ (
    Param XParam,
    BlockP < F > XBlock,
    T initval,
    T *& Arr
) 
```




<hr>



### function InitBlkBUQ 

```C++
template<class T, class F>
void InitBlkBUQ (
    Param XParam,
    BlockP < F > XBlock,
    T initval,
    T *& Arr
) 
```




<hr>



### function InterpstepCPU 

```C++
template<class T, class F>
void InterpstepCPU (
    int nx,
    int ny,
    int hdstep,
    F totaltime,
    F hddt,
    T *& Ux,
    T * Uo,
    T * Un
) 
```




<hr>



### function InterpstepGPU 

```C++
template<class T>
__global__ void InterpstepGPU (
    int nx,
    int ny,
    T totaltime,
    T beforetime,
    T aftertime,
    T * Ux,
    T * Uo,
    T * Un
) 
```




<hr>



### function interp2BUQ 

```C++
template<class T>
void interp2BUQ (
    Param XParam,
    BlockP < T > XBlock,
    std::vector< StaticForcingP < float > > forcing,
    T * z
) 
```




<hr>



### function interp2BUQ 

```C++
template<class T, class F>
void interp2BUQ (
    Param XParam,
    BlockP < T > XBlock,
    F forcing,
    T *& z
) 
```




<hr>



### function interp2BUQ 

```C++
template<class T, class F>
T interp2BUQ (
    T x,
    T y,
    F forcing
) 
```




<hr>



### function interp2BUQ 

```C++
template<class T, class F>
T interp2BUQ (
    T x,
    T y,
    T dx,
    F forcing
) 
```




<hr>



### function setedges 

```C++
template<class T>
void setedges (
    Param XParam,
    BlockP < T > XBlock,
    T *& zb
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/GridManip.h`


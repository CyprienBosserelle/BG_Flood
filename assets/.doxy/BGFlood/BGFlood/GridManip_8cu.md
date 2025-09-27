

# File GridManip.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**GridManip.cu**](GridManip_8cu.md)

[Go to the source code of this file](GridManip_8cu_source.md)



* `#include "GridManip.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Copy2CartCPU**](#function-copy2cartcpu) (int nx, int ny, T \* dest, T \* src) <br> |
|  template void | [**Copy2CartCPU&lt; bool &gt;**](#function-copy2cartcpu-bool) (int nx, int ny, bool \* dest, bool \* src) <br> |
|  template void | [**Copy2CartCPU&lt; double &gt;**](#function-copy2cartcpu-double) (int nx, int ny, double \* dest, double \* src) <br> |
|  template void | [**Copy2CartCPU&lt; float &gt;**](#function-copy2cartcpu-float) (int nx, int ny, float \* dest, float \* src) <br> |
|  template void | [**Copy2CartCPU&lt; int &gt;**](#function-copy2cartcpu-int) (int nx, int ny, int \* dest, int \* src) <br> |
|  void | [**CopyArrayBUQ**](#function-copyarraybuq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; F &gt; XBlock, T \* source, T \*& dest) <br> |
|  void | [**CopyArrayBUQ**](#function-copyarraybuq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; source, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & dest) <br> |
|  void | [**CopyArrayBUQ**](#function-copyarraybuq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; source, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; T &gt; & dest) <br> |
|  template void | [**CopyArrayBUQ&lt; bool, double &gt;**](#function-copyarraybuq-bool-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, bool \* source, bool \*& dest) <br> |
|  template void | [**CopyArrayBUQ&lt; bool, float &gt;**](#function-copyarraybuq-bool-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, bool \* source, bool \*& dest) <br> |
|  template void | [**CopyArrayBUQ&lt; double &gt;**](#function-copyarraybuq-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; source, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; & dest) <br> |
|  template void | [**CopyArrayBUQ&lt; double &gt;**](#function-copyarraybuq-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; source, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; double &gt; & dest) <br> |
|  template void | [**CopyArrayBUQ&lt; double, double &gt;**](#function-copyarraybuq-double-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* source, double \*& dest) <br> |
|  template void | [**CopyArrayBUQ&lt; double, float &gt;**](#function-copyarraybuq-double-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, double \* source, double \*& dest) <br> |
|  template void | [**CopyArrayBUQ&lt; float &gt;**](#function-copyarraybuq-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; source, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; & dest) <br> |
|  template void | [**CopyArrayBUQ&lt; float &gt;**](#function-copyarraybuq-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; source, [**EvolvingP\_M**](structEvolvingP__M.md)&lt; float &gt; & dest) <br> |
|  template void | [**CopyArrayBUQ&lt; float, double &gt;**](#function-copyarraybuq-float-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, float \* source, float \*& dest) <br> |
|  template void | [**CopyArrayBUQ&lt; float, float &gt;**](#function-copyarraybuq-float-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* source, float \*& dest) <br> |
|  template void | [**CopyArrayBUQ&lt; int, double &gt;**](#function-copyarraybuq-int-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, int \* source, int \*& dest) <br> |
|  template void | [**CopyArrayBUQ&lt; int, float &gt;**](#function-copyarraybuq-int-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, int \* source, int \*& dest) <br> |
|  void | [**InitArrayBUQ**](#function-initarraybuq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; F &gt; XBlock, T initval, T \*& Arr) <br> |
|  template void | [**InitArrayBUQ&lt; bool, double &gt;**](#function-initarraybuq-bool-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, bool initval, bool \*& Arr) <br> |
|  template void | [**InitArrayBUQ&lt; bool, float &gt;**](#function-initarraybuq-bool-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, bool initval, bool \*& Arr) <br> |
|  template void | [**InitArrayBUQ&lt; double, double &gt;**](#function-initarraybuq-double-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double initval, double \*& Arr) <br> |
|  template void | [**InitArrayBUQ&lt; double, float &gt;**](#function-initarraybuq-double-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, double initval, double \*& Arr) <br> |
|  template void | [**InitArrayBUQ&lt; float, double &gt;**](#function-initarraybuq-float-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, float initval, float \*& Arr) <br> |
|  template void | [**InitArrayBUQ&lt; float, float &gt;**](#function-initarraybuq-float-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float initval, float \*& Arr) <br> |
|  template void | [**InitArrayBUQ&lt; int, double &gt;**](#function-initarraybuq-int-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, int initval, int \*& Arr) <br> |
|  template void | [**InitArrayBUQ&lt; int, float &gt;**](#function-initarraybuq-int-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, int initval, int \*& Arr) <br> |
|  void | [**InitBlkBUQ**](#function-initblkbuq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; F &gt; XBlock, T initval, T \*& Arr) <br> |
|  template void | [**InitBlkBUQ&lt; bool, double &gt;**](#function-initblkbuq-bool-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, bool initval, bool \*& Arr) <br> |
|  template void | [**InitBlkBUQ&lt; bool, float &gt;**](#function-initblkbuq-bool-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, bool initval, bool \*& Arr) <br> |
|  template void | [**InitBlkBUQ&lt; double, double &gt;**](#function-initblkbuq-double-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double initval, double \*& Arr) <br> |
|  template void | [**InitBlkBUQ&lt; double, float &gt;**](#function-initblkbuq-double-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, double initval, double \*& Arr) <br> |
|  template void | [**InitBlkBUQ&lt; float, double &gt;**](#function-initblkbuq-float-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, float initval, float \*& Arr) <br> |
|  template void | [**InitBlkBUQ&lt; float, float &gt;**](#function-initblkbuq-float-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float initval, float \*& Arr) <br> |
|  template void | [**InitBlkBUQ&lt; int, double &gt;**](#function-initblkbuq-int-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, int initval, int \*& Arr) <br> |
|  template void | [**InitBlkBUQ&lt; int, float &gt;**](#function-initblkbuq-int-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, int initval, int \*& Arr) <br> |
|  void | [**InterpstepCPU**](#function-interpstepcpu) (int nx, int ny, int hdstep, F totaltime, F hddt, T \*& Ux, T \* Uo, T \* Un) <br> |
|  template void | [**InterpstepCPU&lt; double, double &gt;**](#function-interpstepcpu-double-double) (int nx, int ny, int hdstep, double totaltime, double hddt, double \*& Ux, double \* Uo, double \* Un) <br> |
|  template void | [**InterpstepCPU&lt; double, float &gt;**](#function-interpstepcpu-double-float) (int nx, int ny, int hdstep, float totaltime, float hddt, double \*& Ux, double \* Uo, double \* Un) <br> |
|  template void | [**InterpstepCPU&lt; float, double &gt;**](#function-interpstepcpu-float-double) (int nx, int ny, int hdstep, double totaltime, double hddt, float \*& Ux, float \* Uo, float \* Un) <br> |
|  template void | [**InterpstepCPU&lt; float, float &gt;**](#function-interpstepcpu-float-float) (int nx, int ny, int hdstep, float totaltime, float hddt, float \*& Ux, float \* Uo, float \* Un) <br> |
|  template void | [**InterpstepCPU&lt; int, double &gt;**](#function-interpstepcpu-int-double) (int nx, int ny, int hdstep, double totaltime, double hddt, int \*& Ux, int \* Uo, int \* Un) <br> |
|  template void | [**InterpstepCPU&lt; int, float &gt;**](#function-interpstepcpu-int-float) (int nx, int ny, int hdstep, float totaltime, float hddt, int \*& Ux, int \* Uo, int \* Un) <br> |
|  \_\_global\_\_ void | [**InterpstepGPU**](#function-interpstepgpu) (int nx, int ny, T totaltime, T beforetime, T aftertime, T \* Ux, T \* Uo, T \* Un) <br> |
|  template \_\_global\_\_ void | [**InterpstepGPU&lt; double &gt;**](#function-interpstepgpu-double) (int nx, int ny, double totaltime, double beforetime, double aftertime, double \* Ux, double \* Uo, double \* Un) <br> |
|  template \_\_global\_\_ void | [**InterpstepGPU&lt; float &gt;**](#function-interpstepgpu-float) (int nx, int ny, float totaltime, float beforetime, float aftertime, float \* Ux, float \* Uo, float \* Un) <br> |
|  T | [**blockmean**](#function-blockmean) (T x, T y, T dx, F forcing) <br> |
|  void | [**interp2BUQ**](#function-interp2buq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, F forcing, T \*& z) <br> |
|  void | [**interp2BUQ**](#function-interp2buq) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, std::vector&lt; [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; &gt; forcing, T \* z) <br> |
|  T | [**interp2BUQ**](#function-interp2buq) (T x, T y, T dx, F forcing) <br> |
|  T | [**interp2BUQ**](#function-interp2buq) (T x, T y, F forcing) <br> |
|  template void | [**interp2BUQ&lt; double &gt;**](#function-interp2buq-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, std::vector&lt; [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; &gt; forcing, double \* z) <br> |
|  template void | [**interp2BUQ&lt; double, DynForcingP&lt; float &gt; &gt;**](#function-interp2buq-double-dynforcingp-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; forcing, double \*& z) <br> |
|  template double | [**interp2BUQ&lt; double, DynForcingP&lt; float &gt; &gt;**](#function-interp2buq-double-dynforcingp-float) (double x, double y, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; forcing) <br> |
|  template void | [**interp2BUQ&lt; double, StaticForcingP&lt; float &gt; &gt;**](#function-interp2buq-double-staticforcingp-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; forcing, double \*& z) <br> |
|  template double | [**interp2BUQ&lt; double, StaticForcingP&lt; float &gt; &gt;**](#function-interp2buq-double-staticforcingp-float) (double x, double y, [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; forcing) <br> |
|  template double | [**interp2BUQ&lt; double, StaticForcingP&lt; int &gt; &gt;**](#function-interp2buq-double-staticforcingp-int) (double x, double y, [**StaticForcingP**](structStaticForcingP.md)&lt; int &gt; forcing) <br> |
|  template void | [**interp2BUQ&lt; double, deformmap&lt; float &gt; &gt;**](#function-interp2buq-double-deformmap-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**deformmap**](classdeformmap.md)&lt; float &gt; forcing, double \*& z) <br> |
|  template double | [**interp2BUQ&lt; double, deformmap&lt; float &gt; &gt;**](#function-interp2buq-double-deformmap-float) (double x, double y, [**deformmap**](classdeformmap.md)&lt; float &gt; forcing) <br> |
|  template void | [**interp2BUQ&lt; float &gt;**](#function-interp2buq-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, std::vector&lt; [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; &gt; forcing, float \* z) <br> |
|  template void | [**interp2BUQ&lt; float, DynForcingP&lt; float &gt; &gt;**](#function-interp2buq-float-dynforcingp-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; forcing, float \*& z) <br> |
|  template float | [**interp2BUQ&lt; float, DynForcingP&lt; float &gt; &gt;**](#function-interp2buq-float-dynforcingp-float) (float x, float y, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; forcing) <br> |
|  template void | [**interp2BUQ&lt; float, StaticForcingP&lt; float &gt; &gt;**](#function-interp2buq-float-staticforcingp-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; forcing, float \*& z) <br> |
|  template float | [**interp2BUQ&lt; float, StaticForcingP&lt; float &gt; &gt;**](#function-interp2buq-float-staticforcingp-float) (float x, float y, [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; forcing) <br> |
|  template float | [**interp2BUQ&lt; float, StaticForcingP&lt; int &gt; &gt;**](#function-interp2buq-float-staticforcingp-int) (float x, float y, [**StaticForcingP**](structStaticForcingP.md)&lt; int &gt; forcing) <br> |
|  template void | [**interp2BUQ&lt; float, deformmap&lt; float &gt; &gt;**](#function-interp2buq-float-deformmap-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**deformmap**](classdeformmap.md)&lt; float &gt; forcing, float \*& z) <br> |
|  template float | [**interp2BUQ&lt; float, deformmap&lt; float &gt; &gt;**](#function-interp2buq-float-deformmap-float) (float x, float y, [**deformmap**](classdeformmap.md)&lt; float &gt; forcing) <br> |
|  void | [**setedges**](#function-setedges) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \*& zb) <br> |
|  template void | [**setedges&lt; double &gt;**](#function-setedges-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \*& zb) <br> |
|  template void | [**setedges&lt; float &gt;**](#function-setedges-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \*& zb) <br> |
|  void | [**setedgessideBT**](#function-setedgessidebt) ([**Param**](classParam.md) XParam, int ib, int blkA, int blkB, int jread, int jwrite, T \*& zb) <br> |
|  void | [**setedgessideLR**](#function-setedgessidelr) ([**Param**](classParam.md) XParam, int ib, int blkA, int blkB, int iread, int iwrite, T \*& zb) <br> |




























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



### function Copy2CartCPU&lt; bool &gt; 

```C++
template void Copy2CartCPU< bool > (
    int nx,
    int ny,
    bool * dest,
    bool * src
) 
```




<hr>



### function Copy2CartCPU&lt; double &gt; 

```C++
template void Copy2CartCPU< double > (
    int nx,
    int ny,
    double * dest,
    double * src
) 
```




<hr>



### function Copy2CartCPU&lt; float &gt; 

```C++
template void Copy2CartCPU< float > (
    int nx,
    int ny,
    float * dest,
    float * src
) 
```




<hr>



### function Copy2CartCPU&lt; int &gt; 

```C++
template void Copy2CartCPU< int > (
    int nx,
    int ny,
    int * dest,
    int * src
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



### function CopyArrayBUQ&lt; bool, double &gt; 

```C++
template void CopyArrayBUQ< bool, double > (
    Param XParam,
    BlockP < double > XBlock,
    bool * source,
    bool *& dest
) 
```




<hr>



### function CopyArrayBUQ&lt; bool, float &gt; 

```C++
template void CopyArrayBUQ< bool, float > (
    Param XParam,
    BlockP < float > XBlock,
    bool * source,
    bool *& dest
) 
```




<hr>



### function CopyArrayBUQ&lt; double &gt; 

```C++
template void CopyArrayBUQ< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > source,
    EvolvingP < double > & dest
) 
```




<hr>



### function CopyArrayBUQ&lt; double &gt; 

```C++
template void CopyArrayBUQ< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > source,
    EvolvingP_M < double > & dest
) 
```




<hr>



### function CopyArrayBUQ&lt; double, double &gt; 

```C++
template void CopyArrayBUQ< double, double > (
    Param XParam,
    BlockP < double > XBlock,
    double * source,
    double *& dest
) 
```




<hr>



### function CopyArrayBUQ&lt; double, float &gt; 

```C++
template void CopyArrayBUQ< double, float > (
    Param XParam,
    BlockP < float > XBlock,
    double * source,
    double *& dest
) 
```




<hr>



### function CopyArrayBUQ&lt; float &gt; 

```C++
template void CopyArrayBUQ< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > source,
    EvolvingP < float > & dest
) 
```




<hr>



### function CopyArrayBUQ&lt; float &gt; 

```C++
template void CopyArrayBUQ< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > source,
    EvolvingP_M < float > & dest
) 
```




<hr>



### function CopyArrayBUQ&lt; float, double &gt; 

```C++
template void CopyArrayBUQ< float, double > (
    Param XParam,
    BlockP < double > XBlock,
    float * source,
    float *& dest
) 
```




<hr>



### function CopyArrayBUQ&lt; float, float &gt; 

```C++
template void CopyArrayBUQ< float, float > (
    Param XParam,
    BlockP < float > XBlock,
    float * source,
    float *& dest
) 
```




<hr>



### function CopyArrayBUQ&lt; int, double &gt; 

```C++
template void CopyArrayBUQ< int, double > (
    Param XParam,
    BlockP < double > XBlock,
    int * source,
    int *& dest
) 
```




<hr>



### function CopyArrayBUQ&lt; int, float &gt; 

```C++
template void CopyArrayBUQ< int, float > (
    Param XParam,
    BlockP < float > XBlock,
    int * source,
    int *& dest
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



### function InitArrayBUQ&lt; bool, double &gt; 

```C++
template void InitArrayBUQ< bool, double > (
    Param XParam,
    BlockP < double > XBlock,
    bool initval,
    bool *& Arr
) 
```




<hr>



### function InitArrayBUQ&lt; bool, float &gt; 

```C++
template void InitArrayBUQ< bool, float > (
    Param XParam,
    BlockP < float > XBlock,
    bool initval,
    bool *& Arr
) 
```




<hr>



### function InitArrayBUQ&lt; double, double &gt; 

```C++
template void InitArrayBUQ< double, double > (
    Param XParam,
    BlockP < double > XBlock,
    double initval,
    double *& Arr
) 
```




<hr>



### function InitArrayBUQ&lt; double, float &gt; 

```C++
template void InitArrayBUQ< double, float > (
    Param XParam,
    BlockP < float > XBlock,
    double initval,
    double *& Arr
) 
```




<hr>



### function InitArrayBUQ&lt; float, double &gt; 

```C++
template void InitArrayBUQ< float, double > (
    Param XParam,
    BlockP < double > XBlock,
    float initval,
    float *& Arr
) 
```




<hr>



### function InitArrayBUQ&lt; float, float &gt; 

```C++
template void InitArrayBUQ< float, float > (
    Param XParam,
    BlockP < float > XBlock,
    float initval,
    float *& Arr
) 
```




<hr>



### function InitArrayBUQ&lt; int, double &gt; 

```C++
template void InitArrayBUQ< int, double > (
    Param XParam,
    BlockP < double > XBlock,
    int initval,
    int *& Arr
) 
```




<hr>



### function InitArrayBUQ&lt; int, float &gt; 

```C++
template void InitArrayBUQ< int, float > (
    Param XParam,
    BlockP < float > XBlock,
    int initval,
    int *& Arr
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



### function InitBlkBUQ&lt; bool, double &gt; 

```C++
template void InitBlkBUQ< bool, double > (
    Param XParam,
    BlockP < double > XBlock,
    bool initval,
    bool *& Arr
) 
```




<hr>



### function InitBlkBUQ&lt; bool, float &gt; 

```C++
template void InitBlkBUQ< bool, float > (
    Param XParam,
    BlockP < float > XBlock,
    bool initval,
    bool *& Arr
) 
```




<hr>



### function InitBlkBUQ&lt; double, double &gt; 

```C++
template void InitBlkBUQ< double, double > (
    Param XParam,
    BlockP < double > XBlock,
    double initval,
    double *& Arr
) 
```




<hr>



### function InitBlkBUQ&lt; double, float &gt; 

```C++
template void InitBlkBUQ< double, float > (
    Param XParam,
    BlockP < float > XBlock,
    double initval,
    double *& Arr
) 
```




<hr>



### function InitBlkBUQ&lt; float, double &gt; 

```C++
template void InitBlkBUQ< float, double > (
    Param XParam,
    BlockP < double > XBlock,
    float initval,
    float *& Arr
) 
```




<hr>



### function InitBlkBUQ&lt; float, float &gt; 

```C++
template void InitBlkBUQ< float, float > (
    Param XParam,
    BlockP < float > XBlock,
    float initval,
    float *& Arr
) 
```




<hr>



### function InitBlkBUQ&lt; int, double &gt; 

```C++
template void InitBlkBUQ< int, double > (
    Param XParam,
    BlockP < double > XBlock,
    int initval,
    int *& Arr
) 
```




<hr>



### function InitBlkBUQ&lt; int, float &gt; 

```C++
template void InitBlkBUQ< int, float > (
    Param XParam,
    BlockP < float > XBlock,
    int initval,
    int *& Arr
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



### function InterpstepCPU&lt; double, double &gt; 

```C++
template void InterpstepCPU< double, double > (
    int nx,
    int ny,
    int hdstep,
    double totaltime,
    double hddt,
    double *& Ux,
    double * Uo,
    double * Un
) 
```




<hr>



### function InterpstepCPU&lt; double, float &gt; 

```C++
template void InterpstepCPU< double, float > (
    int nx,
    int ny,
    int hdstep,
    float totaltime,
    float hddt,
    double *& Ux,
    double * Uo,
    double * Un
) 
```




<hr>



### function InterpstepCPU&lt; float, double &gt; 

```C++
template void InterpstepCPU< float, double > (
    int nx,
    int ny,
    int hdstep,
    double totaltime,
    double hddt,
    float *& Ux,
    float * Uo,
    float * Un
) 
```




<hr>



### function InterpstepCPU&lt; float, float &gt; 

```C++
template void InterpstepCPU< float, float > (
    int nx,
    int ny,
    int hdstep,
    float totaltime,
    float hddt,
    float *& Ux,
    float * Uo,
    float * Un
) 
```




<hr>



### function InterpstepCPU&lt; int, double &gt; 

```C++
template void InterpstepCPU< int, double > (
    int nx,
    int ny,
    int hdstep,
    double totaltime,
    double hddt,
    int *& Ux,
    int * Uo,
    int * Un
) 
```




<hr>



### function InterpstepCPU&lt; int, float &gt; 

```C++
template void InterpstepCPU< int, float > (
    int nx,
    int ny,
    int hdstep,
    float totaltime,
    float hddt,
    int *& Ux,
    int * Uo,
    int * Un
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



### function InterpstepGPU&lt; double &gt; 

```C++
template __global__ void InterpstepGPU< double > (
    int nx,
    int ny,
    double totaltime,
    double beforetime,
    double aftertime,
    double * Ux,
    double * Uo,
    double * Un
) 
```




<hr>



### function InterpstepGPU&lt; float &gt; 

```C++
template __global__ void InterpstepGPU< float > (
    int nx,
    int ny,
    float totaltime,
    float beforetime,
    float aftertime,
    float * Ux,
    float * Uo,
    float * Un
) 
```




<hr>



### function blockmean 

```C++
template<class T, class F>
T blockmean (
    T x,
    T y,
    T dx,
    F forcing
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
T interp2BUQ (
    T x,
    T y,
    T dx,
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
    F forcing
) 
```




<hr>



### function interp2BUQ&lt; double &gt; 

```C++
template void interp2BUQ< double > (
    Param XParam,
    BlockP < double > XBlock,
    std::vector< StaticForcingP < float > > forcing,
    double * z
) 
```




<hr>



### function interp2BUQ&lt; double, DynForcingP&lt; float &gt; &gt; 

```C++
template void interp2BUQ< double, DynForcingP< float > > (
    Param XParam,
    BlockP < double > XBlock,
    DynForcingP < float > forcing,
    double *& z
) 
```




<hr>



### function interp2BUQ&lt; double, DynForcingP&lt; float &gt; &gt; 

```C++
template double interp2BUQ< double, DynForcingP< float > > (
    double x,
    double y,
    DynForcingP < float > forcing
) 
```




<hr>



### function interp2BUQ&lt; double, StaticForcingP&lt; float &gt; &gt; 

```C++
template void interp2BUQ< double, StaticForcingP< float > > (
    Param XParam,
    BlockP < double > XBlock,
    StaticForcingP < float > forcing,
    double *& z
) 
```




<hr>



### function interp2BUQ&lt; double, StaticForcingP&lt; float &gt; &gt; 

```C++
template double interp2BUQ< double, StaticForcingP< float > > (
    double x,
    double y,
    StaticForcingP < float > forcing
) 
```




<hr>



### function interp2BUQ&lt; double, StaticForcingP&lt; int &gt; &gt; 

```C++
template double interp2BUQ< double, StaticForcingP< int > > (
    double x,
    double y,
    StaticForcingP < int > forcing
) 
```




<hr>



### function interp2BUQ&lt; double, deformmap&lt; float &gt; &gt; 

```C++
template void interp2BUQ< double, deformmap< float > > (
    Param XParam,
    BlockP < double > XBlock,
    deformmap < float > forcing,
    double *& z
) 
```




<hr>



### function interp2BUQ&lt; double, deformmap&lt; float &gt; &gt; 

```C++
template double interp2BUQ< double, deformmap< float > > (
    double x,
    double y,
    deformmap < float > forcing
) 
```




<hr>



### function interp2BUQ&lt; float &gt; 

```C++
template void interp2BUQ< float > (
    Param XParam,
    BlockP < float > XBlock,
    std::vector< StaticForcingP < float > > forcing,
    float * z
) 
```




<hr>



### function interp2BUQ&lt; float, DynForcingP&lt; float &gt; &gt; 

```C++
template void interp2BUQ< float, DynForcingP< float > > (
    Param XParam,
    BlockP < float > XBlock,
    DynForcingP < float > forcing,
    float *& z
) 
```




<hr>



### function interp2BUQ&lt; float, DynForcingP&lt; float &gt; &gt; 

```C++
template float interp2BUQ< float, DynForcingP< float > > (
    float x,
    float y,
    DynForcingP < float > forcing
) 
```




<hr>



### function interp2BUQ&lt; float, StaticForcingP&lt; float &gt; &gt; 

```C++
template void interp2BUQ< float, StaticForcingP< float > > (
    Param XParam,
    BlockP < float > XBlock,
    StaticForcingP < float > forcing,
    float *& z
) 
```




<hr>



### function interp2BUQ&lt; float, StaticForcingP&lt; float &gt; &gt; 

```C++
template float interp2BUQ< float, StaticForcingP< float > > (
    float x,
    float y,
    StaticForcingP < float > forcing
) 
```




<hr>



### function interp2BUQ&lt; float, StaticForcingP&lt; int &gt; &gt; 

```C++
template float interp2BUQ< float, StaticForcingP< int > > (
    float x,
    float y,
    StaticForcingP < int > forcing
) 
```




<hr>



### function interp2BUQ&lt; float, deformmap&lt; float &gt; &gt; 

```C++
template void interp2BUQ< float, deformmap< float > > (
    Param XParam,
    BlockP < float > XBlock,
    deformmap < float > forcing,
    float *& z
) 
```




<hr>



### function interp2BUQ&lt; float, deformmap&lt; float &gt; &gt; 

```C++
template float interp2BUQ< float, deformmap< float > > (
    float x,
    float y,
    deformmap < float > forcing
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



### function setedges&lt; double &gt; 

```C++
template void setedges< double > (
    Param XParam,
    BlockP < double > XBlock,
    double *& zb
) 
```




<hr>



### function setedges&lt; float &gt; 

```C++
template void setedges< float > (
    Param XParam,
    BlockP < float > XBlock,
    float *& zb
) 
```




<hr>



### function setedgessideBT 

```C++
template<class T>
void setedgessideBT (
    Param XParam,
    int ib,
    int blkA,
    int blkB,
    int jread,
    int jwrite,
    T *& zb
) 
```




<hr>



### function setedgessideLR 

```C++
template<class T>
void setedgessideLR (
    Param XParam,
    int ib,
    int blkA,
    int blkB,
    int iread,
    int iwrite,
    T *& zb
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/GridManip.cu`


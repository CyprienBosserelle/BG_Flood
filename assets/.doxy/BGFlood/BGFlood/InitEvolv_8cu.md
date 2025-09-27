

# File InitEvolv.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**InitEvolv.cu**](InitEvolv_8cu.md)

[Go to the source code of this file](InitEvolv_8cu_source.md)



* `#include "InitEvolv.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**AddZSoffset**](#function-addzsoffset) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv, T \* zb) <br> |
|  int | [**coldstart**](#function-coldstart) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv) <br> |
|  void | [**initevolv**](#function-initevolv) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv, T \*& zb) <br> |
|  template void | [**initevolv&lt; double &gt;**](#function-initevolv-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; & XEv, double \*& zb) <br> |
|  template void | [**initevolv&lt; float &gt;**](#function-initevolv-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; & XEv, float \*& zb) <br> |
|  int | [**readhotstartfile**](#function-readhotstartfile) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv, T \*& zb) <br> |
|  template int | [**readhotstartfile&lt; double &gt;**](#function-readhotstartfile-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; & XEv, double \*& zb) <br> |
|  template int | [**readhotstartfile&lt; float &gt;**](#function-readhotstartfile-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; & XEv, float \*& zb) <br> |
|  int | [**readhotstartfileBG**](#function-readhotstartfilebg) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv, T \*& zb) <br> |
|  void | [**warmstart**](#function-warmstart) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv) <br> |
|  void | [**warmstartold**](#function-warmstartold) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv) <br> |




























## Public Functions Documentation




### function AddZSoffset 

```C++
template<class T>
int AddZSoffset (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > & XEv,
    T * zb
) 
```




<hr>



### function coldstart 

```C++
template<class T>
int coldstart (
    Param XParam,
    BlockP < T > XBlock,
    T * zb,
    EvolvingP < T > & XEv
) 
```




<hr>



### function initevolv 

```C++
template<class T>
void initevolv (
    Param XParam,
    BlockP < T > XBlock,
    Forcing < float > XForcing,
    EvolvingP < T > & XEv,
    T *& zb
) 
```




<hr>



### function initevolv&lt; double &gt; 

```C++
template void initevolv< double > (
    Param XParam,
    BlockP < double > XBlock,
    Forcing < float > XForcing,
    EvolvingP < double > & XEv,
    double *& zb
) 
```




<hr>



### function initevolv&lt; float &gt; 

```C++
template void initevolv< float > (
    Param XParam,
    BlockP < float > XBlock,
    Forcing < float > XForcing,
    EvolvingP < float > & XEv,
    float *& zb
) 
```




<hr>



### function readhotstartfile 

```C++
template<class T>
int readhotstartfile (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > & XEv,
    T *& zb
) 
```




<hr>



### function readhotstartfile&lt; double &gt; 

```C++
template int readhotstartfile< double > (
    Param XParam,
    BlockP < double > XBlock,
    EvolvingP < double > & XEv,
    double *& zb
) 
```




<hr>



### function readhotstartfile&lt; float &gt; 

```C++
template int readhotstartfile< float > (
    Param XParam,
    BlockP < float > XBlock,
    EvolvingP < float > & XEv,
    float *& zb
) 
```




<hr>



### function readhotstartfileBG 

```C++
template<class T>
int readhotstartfileBG (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > & XEv,
    T *& zb
) 
```




<hr>



### function warmstart 

```C++
template<class T>
void warmstart (
    Param XParam,
    Forcing < float > XForcing,
    BlockP < T > XBlock,
    T * zb,
    EvolvingP < T > & XEv
) 
```




<hr>



### function warmstartold 

```C++
template<class T>
void warmstartold (
    Param XParam,
    Forcing < float > XForcing,
    BlockP < T > XBlock,
    T * zb,
    EvolvingP < T > & XEv
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/InitEvolv.cu`


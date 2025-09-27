

# File InitEvolv.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**InitEvolv.h**](InitEvolv_8h.md)

[Go to the source code of this file](InitEvolv_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Forcing.h"`
* `#include "MemManagement.h"`
* `#include "Util_CPU.h"`
* `#include "Arrays.h"`
* `#include "Write_txtlog.h"`
* `#include "GridManip.h"`
* `#include "Read_netcdf.h"`
* `#include "ReadForcing.h"`
* `#include "Updateforcing.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**AddZSoffset**](#function-addzsoffset) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv, T \* zb) <br> |
|  int | [**coldstart**](#function-coldstart) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv) <br> |
|  void | [**initevolv**](#function-initevolv) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv, T \*& zb) <br> |
|  int | [**readhotstartfile**](#function-readhotstartfile) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv, T \*& zb) <br> |
|  void | [**warmstart**](#function-warmstart) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv) <br> |




























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

------------------------------
The documentation for this class was generated from the following file `src/InitEvolv.h`


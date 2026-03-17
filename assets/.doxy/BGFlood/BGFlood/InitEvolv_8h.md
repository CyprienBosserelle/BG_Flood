

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
|  int | [**AddZSoffset**](#function-addzsoffset) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv, T \* zb) <br>_Add offset to surface elevation (zs) and update water depth (h)._  |
|  int | [**coldstart**](#function-coldstart) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv) <br>_Cold start initialization of evolving variables._  |
|  void | [**initevolv**](#function-initevolv) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv, T \*& zb) <br>_Initialize evolving variables for the simulation._  |
|  int | [**readhotstartfile**](#function-readhotstartfile) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv, T \*& zb) <br>_Read hotstart file and initialize evolving variables and bathymetry._  |
|  void | [**warmstart**](#function-warmstart) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* zb, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv) <br>_Warm start initialization using boundary conditions and interpolation._  |




























## Public Functions Documentation




### function AddZSoffset 

_Add offset to surface elevation (zs) and update water depth (h)._ 
```C++
template<class T>
int AddZSoffset (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > & XEv,
    T * zb
) 
```



Applies zsoffset to zs and updates h for all blocks where h &gt; eps.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables (input/output) 
* `zb` Bathymetry array 



**Returns:**

Success flag (1 if successful) 





        

<hr>



### function coldstart 

_Cold start initialization of evolving variables._ 
```C++
template<class T>
int coldstart (
    Param XParam,
    BlockP < T > XBlock,
    T * zb,
    EvolvingP < T > & XEv
) 
```



Sets initial water level, velocity, and bathymetry arrays for all blocks using specified zsinit and zsoffset.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `zb` Bathymetry array 
* `XEv` Evolving variables (output) 



**Returns:**

Success flag (1 if successful) 





        

<hr>



### function initevolv 

_Initialize evolving variables for the simulation._ 
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



Handles hotstart, coldstart, and warmstart initialization of water level, velocity, and bathymetry arrays. Applies offsets and boundary conditions as needed.




**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XForcing` Forcing/boundary conditions 
* `XEv` Evolving variables (output) 
* `zb` Bathymetry array (input/output) 




        

<hr>



### function readhotstartfile 

_Read hotstart file and initialize evolving variables and bathymetry._ 
```C++
template<class T>
int readhotstartfile (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > & XEv,
    T *& zb
) 
```



Reads NetCDF hotstart file, extracts variables, and fills arrays for all blocks. Handles missing variables and applies edge corrections.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables (output) 
* `zb` Bathymetry array (output) 



**Returns:**

Success flag (1 if successful, 0 if fallback to cold start) 





        

<hr>



### function warmstart 

_Warm start initialization using boundary conditions and interpolation._ 
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



Sets initial water level, velocity, and bathymetry arrays for all blocks using boundary segments and atmospheric pressure forcing.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XForcing` Forcing/boundary conditions 
* `XBlock` Block parameters 
* `zb` Bathymetry array 
* `XEv` Evolving variables (output) 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/InitEvolv.h`


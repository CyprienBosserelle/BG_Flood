

# File InitEvolv.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**InitEvolv.cu**](_init_evolv_8cu.md)

[Go to the source code of this file](_init_evolv_8cu_source.md)



* `#include "InitEvolv.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**AddZSoffset**](#function-addzsoffset) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & XEv, T \* zb) <br>_Add offset to surface elevation (zs) and update water depth (h)._  |
|  int | [**coldstart**](#function-coldstart) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* zb, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & XEv) <br>_Cold start initialization of evolving variables._  |
|  void | [**initevolv**](#function-initevolv) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & XEv, T \*& zb) <br>_Initialize evolving variables for the simulation._  |
|  template void | [**initevolv&lt; double &gt;**](#function-initevolv-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; & XEv, double \*& zb) <br> |
|  template void | [**initevolv&lt; float &gt;**](#function-initevolv-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; & XEv, float \*& zb) <br> |
|  int | [**readhotstartfile**](#function-readhotstartfile) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & XEv, T \*& zb) <br>_Read hotstart file and initialize evolving variables and bathymetry._  |
|  template int | [**readhotstartfile&lt; double &gt;**](#function-readhotstartfile-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; & XEv, double \*& zb) <br> |
|  template int | [**readhotstartfile&lt; float &gt;**](#function-readhotstartfile-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; & XEv, float \*& zb) <br> |
|  int | [**readhotstartfileBG**](#function-readhotstartfilebg) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & XEv, T \*& zb) <br>_Read BG\_Flood hotstart file and extract block attributes._  |
|  void | [**warmstart**](#function-warmstart) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* zb, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & XEv) <br>_Warm start initialization using boundary conditions and interpolation._  |
|  void | [**warmstartold**](#function-warmstartold) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* zb, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & XEv) <br>_Legacy warm start initialization using inverse distance to boundaries._  |




























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

_Read BG\_Flood hotstart file and extract block attributes._ 
```C++
template<class T>
int readhotstartfileBG (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > & XEv,
    T *& zb
) 
```



Opens NetCDF hotstart file, checks for BG\_Flood attribute, and closes file.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `XEv` Evolving variables 
* `zb` Bathymetry array 



**Returns:**

Status code 





        

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



### function warmstartold 

_Legacy warm start initialization using inverse distance to boundaries._ 
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



Sets initial water level, velocity, and bathymetry arrays for all blocks using inverse distance interpolation from boundaries.




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
The documentation for this class was generated from the following file `src/InitEvolv.cu`


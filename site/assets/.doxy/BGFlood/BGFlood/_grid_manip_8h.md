

# File GridManip.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**GridManip.h**](_grid_manip_8h.md)

[Go to the source code of this file](_grid_manip_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Util_CPU.h"`
* `#include "Forcing.h"`
* `#include "Arrays.h"`
* `#include "MemManagement.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Copy2CartCPU**](#function-copy2cartcpu) (int nx, int ny, T \* dest, T \* src) <br>_Copy values from source to destination for a Cartesian grid._  |
|  void | [**CopyArrayBUQ**](#function-copyarraybuq) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; F &gt; XBlock, T \* source, T \*& dest) <br>_Copy values from source array to destination array for all blocks._  |
|  void | [**CopyArrayBUQ**](#function-copyarraybuq) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; source, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & dest) <br>_Copy all evolving variables from source to destination._  |
|  void | [**CopyArrayBUQ**](#function-copyarraybuq) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; source, [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; T &gt; & dest) <br>_Copy all evolving variables and compute derived quantities (U, hU)._  |
|  void | [**InitArrayBUQ**](#function-initarraybuq) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; F &gt; XBlock, T initval, T \*& Arr) <br>_Initialize a block-structured array with a given value._  |
|  void | [**InitBlkBUQ**](#function-initblkbuq) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; F &gt; XBlock, T initval, T \*& Arr) <br>_Initialize a block-level array with a given value._  |
|  void | [**InterpstepCPU**](#function-interpstepcpu) (int nx, int ny, int hdstep, F totaltime, F hddt, T \*& Ux, T \* Uo, T \* Un) <br>_CPU routine for time interpolation of solution arrays._  |
|  \_\_global\_\_ void | [**InterpstepGPU**](#function-interpstepgpu) (int nx, int ny, T totaltime, T beforetime, T aftertime, T \* Ux, T \* Uo, T \* Un) <br>_GPU kernel for time interpolation of solution arrays._  |
|  void | [**interp2BUQ**](#function-interp2buq) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, std::vector&lt; [**StaticForcingP**](struct_static_forcing_p.md)&lt; float &gt;&gt; forcing, T \* z) <br>_Interpolate values from multiple forcing maps to block array using bilinear interpolation._  |
|  void | [**interp2BUQ**](#function-interp2buq) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, F forcing, T \*& z) <br>_Interpolate values from forcing map to block array using bilinear interpolation._  |
|  T | [**interp2BUQ**](#function-interp2buq) (T x, T y, F forcing) <br>_Bilinear interpolation for value at (x, y) from forcing map._  |
|  T | [**interp2BUQ**](#function-interp2buq) (T x, T y, T dx, F forcing) <br>_Interpolate value at (x, y) using either bilinear or blockmean interpolation._  |
|  void | [**setedges**](#function-setedges) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \*& zb) <br>_Set edge values for bathymetry array at domain boundaries._  |




























## Public Functions Documentation




### function Copy2CartCPU 

_Copy values from source to destination for a Cartesian grid._ 
```C++
template<class T>
void Copy2CartCPU (
    int nx,
    int ny,
    T * dest,
    T * src
) 
```



Copies all values from src to dest for a regular Cartesian grid.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `nx` Number of x grid points 
* `ny` Number of y grid points 
* `dest` Destination array 
* `src` Source array 




        

<hr>



### function CopyArrayBUQ 

_Copy values from source array to destination array for all blocks._ 
```C++
template<class T, class F>
void CopyArrayBUQ (
    Param XParam,
    BlockP < F > XBlock,
    T * source,
    T *& dest
) 
```



Copies all elements for each active block from source to dest.




**Template parameters:**


* `T` Data type 
* `F` Block type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `source` Source array 
* `dest` Destination array 




        

<hr>



### function CopyArrayBUQ 

_Copy all evolving variables from source to destination._ 
```C++
template<class T>
void CopyArrayBUQ (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > source,
    EvolvingP < T > & dest
) 
```



Copies h, u, v, zs arrays for all blocks.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `source` Source evolving variables 
* `dest` Destination evolving variables 




        

<hr>



### function CopyArrayBUQ 

_Copy all evolving variables and compute derived quantities (U, hU)._ 
```C++
template<class T>
void CopyArrayBUQ (
    Param XParam,
    BlockP < T > XBlock,
    EvolvingP < T > source,
    EvolvingP_M < T > & dest
) 
```



Copies h, u, v, zs arrays and computes U and hU for all blocks.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `source` Source evolving variables 
* `dest` Destination evolving variables (with derived quantities) 




        

<hr>



### function InitArrayBUQ 

_Initialize a block-structured array with a given value._ 
```C++
template<class T, class F>
void InitArrayBUQ (
    Param XParam,
    BlockP < F > XBlock,
    T initval,
    T *& Arr
) 
```



Sets all elements of Arr for each active block to initval.




**Template parameters:**


* `T` Data type (float, double, int, bool) 
* `F` Block type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `initval` Value to initialize 
* `Arr` Array to initialize 




        

<hr>



### function InitBlkBUQ 

_Initialize a block-level array with a given value._ 
```C++
template<class T, class F>
void InitBlkBUQ (
    Param XParam,
    BlockP < F > XBlock,
    T initval,
    T *& Arr
) 
```



Sets each block's entry in Arr to initval.




**Template parameters:**


* `T` Data type 
* `F` Block type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `initval` Value to initialize 
* `Arr` Array to initialize 




        

<hr>



### function InterpstepCPU 

_CPU routine for time interpolation of solution arrays._ 
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



Interpolates between Uo and Un to compute Ux at a given time step.




**Template parameters:**


* `T` Data type 
* `F` Time type 



**Parameters:**


* `nx` Number of x grid points 
* `ny` Number of y grid points 
* `hdstep` Time step index 
* `totaltime` Total simulation time 
* `hddt` Time step size 
* `Ux` Output array 
* `Uo` Previous solution array 
* `Un` Next solution array 




        

<hr>



### function InterpstepGPU 

_GPU kernel for time interpolation of solution arrays._ 
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



Interpolates between Uo and Un to compute Ux at a given time using shared memory.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `nx` Number of x grid points 
* `ny` Number of y grid points 
* `totaltime` Total simulation time 
* `beforetime` Previous time 
* `aftertime` Next time 
* `Ux` Output array 
* `Uo` Previous solution array 
* `Un` Next solution array 




        

<hr>



### function interp2BUQ 

_Interpolate values from multiple forcing maps to block array using bilinear interpolation._ 
```C++
template<class T>
void interp2BUQ (
    Param XParam,
    BlockP < T > XBlock,
    std::vector< StaticForcingP < float >> forcing,
    T * z
) 
```



Fills z array for each block using bilinear interpolation from multiple forcing maps.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `forcing` Vector of forcing maps 
* `z` Output array 




        

<hr>



### function interp2BUQ 

_Interpolate values from forcing map to block array using bilinear interpolation._ 
```C++
template<class T, class F>
void interp2BUQ (
    Param XParam,
    BlockP < T > XBlock,
    F forcing,
    T *& z
) 
```



Fills z array for each block using bilinear interpolation from forcing map(s).




**Template parameters:**


* `T` Data type 
* `F` [**Forcing**](struct_forcing.md) type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `forcing` [**Forcing**](struct_forcing.md) map(s) 
* `z` Output array 




        

<hr>



### function interp2BUQ 

_Bilinear interpolation for value at (x, y) from forcing map._ 
```C++
template<class T, class F>
T interp2BUQ (
    T x,
    T y,
    F forcing
) 
```



Performs bilinear interpolation using surrounding grid points in forcing map.




**Template parameters:**


* `T` Data type 
* `F` [**Forcing**](struct_forcing.md) type 



**Parameters:**


* `x` X coordinate 
* `y` Y coordinate 
* `forcing` [**Forcing**](struct_forcing.md) map 



**Returns:**

Interpolated value 





        

<hr>



### function interp2BUQ 

_Interpolate value at (x, y) using either bilinear or blockmean interpolation._ 
```C++
template<class T, class F>
T interp2BUQ (
    T x,
    T y,
    T dx,
    F forcing
) 
```



Chooses interpolation method based on grid spacing dx.




**Template parameters:**


* `T` Data type 
* `F` [**Forcing**](struct_forcing.md) type 



**Parameters:**


* `x` X coordinate 
* `y` Y coordinate 
* `dx` Grid spacing 
* `forcing` [**Forcing**](struct_forcing.md) map 



**Returns:**

Interpolated value 





        

<hr>



### function setedges 

_Set edge values for bathymetry array at domain boundaries._ 
```C++
template<class T>
void setedges (
    Param XParam,
    BlockP < T > XBlock,
    T *& zb
) 
```



Copies values from interior to boundary cells for blocks with no neighbor.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` Simulation parameters 
* `XBlock` Block parameters 
* `zb` Bathymetry array 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/GridManip.h`


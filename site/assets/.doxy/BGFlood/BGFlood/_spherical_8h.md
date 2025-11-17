

# File Spherical.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Spherical.h**](_spherical_8h.md)

[Go to the source code of this file](_spherical_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "MemManagement.h"`
* `#include "Util_CPU.h"`
* `#include "Kurganov.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ \_\_device\_\_ T | [**calcCM**](#function-calccm) (T Radius, T delta, T yo, int iy) <br>_Calculate the scale factor for the y face length in a spherical model. This function computes the scale factor based on the sphere's radius, grid spacing, origin offset, and index in the y direction. Scale factor for y face length (x face lengh scale is always 1 in spherical model assuming that lat long are entered)_  |
|  \_\_host\_\_ \_\_device\_\_ T | [**calcFM**](#function-calcfm) (T Radius, T delta, T yo, T iy) <br>_Calculate the scale factor for the y face length in a spherical model. This function computes the scale factor based on the sphere's radius, grid spacing, origin offset and index in the y direction. Scale factor for y face length (x face lengh scale is always 1 in spherical model assuming that lat long are entered)_  |
|  \_\_host\_\_ \_\_device\_\_ T | [**spharea**](#function-spharea) (T Radius, T lon, T lat, T dx) <br>_Calculate the surface area of a spherical cap._  |




























## Public Functions Documentation




### function calcCM 

_Calculate the scale factor for the y face length in a spherical model. This function computes the scale factor based on the sphere's radius, grid spacing, origin offset, and index in the y direction. Scale factor for y face length (x face lengh scale is always 1 in spherical model assuming that lat long are entered)_ 
```C++
template<class T>
__host__ __device__ T calcCM (
    T Radius,
    T delta,
    T yo,
    int iy
) 
```





**Parameters:**


* `Radius` Radius of the sphere 
* `delta` Grid spacing 
* `yo` Origin offset in the y direction 
* `iy` Index in the y direction 




        

<hr>



### function calcFM 

_Calculate the scale factor for the y face length in a spherical model. This function computes the scale factor based on the sphere's radius, grid spacing, origin offset and index in the y direction. Scale factor for y face length (x face lengh scale is always 1 in spherical model assuming that lat long are entered)_ 
```C++
template<class T>
__host__ __device__ T calcFM (
    T Radius,
    T delta,
    T yo,
    T iy
) 
```





**Parameters:**


* `Radius` Radius of the sphere 
* `delta` Grid spacing 
* `yo` Origin offset in the y direction 
* `iy` Index in the y direction 




        

<hr>



### function spharea 

_Calculate the surface area of a spherical cap._ 
```C++
template<class T>
__host__ __device__ T spharea (
    T Radius,
    T lon,
    T lat,
    T dx
) 
```





**Template parameters:**


* `T` Data type (float or double) 



**Parameters:**


* `Radius` Radius of the sphere 
* `lon` Longitude of the center of the cap (in degrees) 
* `lat` Latitude of the center of the cap (in degrees) 
* `dx` Grid spacing (in degrees) 



**Returns:**

Surface area of the spherical cap 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Spherical.h`




# File Spherical.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Spherical.h**](Spherical_8h.md)

[Go to the source code of this file](Spherical_8h_source.md)



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
|  \_\_host\_\_ \_\_device\_\_ T | [**calcCM**](#function-calccm) (T Radius, T delta, T yo, int iy) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**calcFM**](#function-calcfm) (T Radius, T delta, T yo, T iy) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**spharea**](#function-spharea) (T Radius, T lon, T lat, T dx) <br> |




























## Public Functions Documentation




### function calcCM 

```C++
template<class T>
__host__ __device__ T calcCM (
    T Radius,
    T delta,
    T yo,
    int iy
) 
```



Scale factor for y face length (x face lengh scale is always 1 in spherical model assuming that lat long are entered) 


        

<hr>



### function calcFM 

```C++
template<class T>
__host__ __device__ T calcFM (
    T Radius,
    T delta,
    T yo,
    T iy
) 
```




<hr>



### function spharea 

```C++
template<class T>
__host__ __device__ T spharea (
    T Radius,
    T lon,
    T lat,
    T dx
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Spherical.h`


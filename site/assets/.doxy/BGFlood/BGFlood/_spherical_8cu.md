

# File Spherical.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Spherical.cu**](_spherical_8cu.md)

[Go to the source code of this file](_spherical_8cu_source.md)



* `#include "Spherical.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ \_\_device\_\_ T | [**calcCM**](#function-calccm) (T Radius, T delta, T yo, int iy) <br>_Calculate the scale factor for the y face length in a spherical model. This function computes the scale factor based on the sphere's radius, grid spacing, origin offset, and index in the y direction. Scale factor for y face length (x face lengh scale is always 1 in spherical model assuming that lat long are entered)_  |
|  template \_\_host\_\_ \_\_device\_\_ double | [**calcCM**](#function-calccm) (double Radius, double delta, double yo, int iy) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**calcCM**](#function-calccm) (float Radius, float delta, float yo, int iy) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**calcFM**](#function-calcfm) (T Radius, T delta, T yo, T iy) <br>_Calculate the scale factor for the y face length in a spherical model. This function computes the scale factor based on the sphere's radius, grid spacing, origin offset and index in the y direction. Scale factor for y face length (x face lengh scale is always 1 in spherical model assuming that lat long are entered)_  |
|  template \_\_host\_\_ \_\_device\_\_ double | [**calcFM**](#function-calcfm) (double Radius, double delta, double yo, double iy) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**calcFM**](#function-calcfm) (float Radius, float delta, float yo, float iy) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**haversin**](#function-haversin) (T Radius, T lon1, T lat1, T lon2, T lat2) <br>_Classic Haversine formula to calculate great-circle distance between two points on a sphere. The function is too slow to use directly in BG\_flood engine but is more usable (i.e. naive) for model setup._  |
|  \_\_host\_\_ \_\_device\_\_ T | [**spharea**](#function-spharea) (T Radius, T lon, T lat, T dx) <br>_Calculate the surface area of a spherical cap._  |
|  template \_\_host\_\_ \_\_device\_\_ double | [**spharea**](#function-spharea) (double Radius, double lon, double lat, double dx) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**spharea**](#function-spharea) (float Radius, float lon, float lat, float dx) <br> |




























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



### function calcCM 

```C++
template __host__ __device__ double calcCM (
    double Radius,
    double delta,
    double yo,
    int iy
) 
```




<hr>



### function calcCM 

```C++
template __host__ __device__ float calcCM (
    float Radius,
    float delta,
    float yo,
    int iy
) 
```




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



### function calcFM 

```C++
template __host__ __device__ double calcFM (
    double Radius,
    double delta,
    double yo,
    double iy
) 
```




<hr>



### function calcFM 

```C++
template __host__ __device__ float calcFM (
    float Radius,
    float delta,
    float yo,
    float iy
) 
```




<hr>



### function haversin 

_Classic Haversine formula to calculate great-circle distance between two points on a sphere. The function is too slow to use directly in BG\_flood engine but is more usable (i.e. naive) for model setup._ 
```C++
template<class T>
__host__ __device__ T haversin (
    T Radius,
    T lon1,
    T lat1,
    T lon2,
    T lat2
) 
```





**Parameters:**


* `Radius` Radius of the sphere 
* `lon1` Longitude of the first point (in degrees) 
* `lat1` Latitude of the first point (in degrees) 
* `lon2` Longitude of the second point (in degrees) 
* `lat2` Latitude of the second point (in degrees) 



**Returns:**

Great-circle distance between the two points 





        

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



### function spharea 

```C++
template __host__ __device__ double spharea (
    double Radius,
    double lon,
    double lat,
    double dx
) 
```




<hr>



### function spharea 

```C++
template __host__ __device__ float spharea (
    float Radius,
    float lon,
    float lat,
    float dx
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Spherical.cu`


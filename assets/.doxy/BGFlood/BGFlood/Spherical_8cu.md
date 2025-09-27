

# File Spherical.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Spherical.cu**](Spherical_8cu.md)

[Go to the source code of this file](Spherical_8cu_source.md)



* `#include "Spherical.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ \_\_device\_\_ T | [**calcCM**](#function-calccm) (T Radius, T delta, T yo, int iy) <br> |
|  template \_\_host\_\_ \_\_device\_\_ double | [**calcCM**](#function-calccm) (double Radius, double delta, double yo, int iy) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**calcCM**](#function-calccm) (float Radius, float delta, float yo, int iy) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**calcFM**](#function-calcfm) (T Radius, T delta, T yo, T iy) <br> |
|  template \_\_host\_\_ \_\_device\_\_ double | [**calcFM**](#function-calcfm) (double Radius, double delta, double yo, double iy) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**calcFM**](#function-calcfm) (float Radius, float delta, float yo, float iy) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**haversin**](#function-haversin) (T Radius, T lon1, T lat1, T lon2, T lat2) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**spharea**](#function-spharea) (T Radius, T lon, T lat, T dx) <br> |
|  template \_\_host\_\_ \_\_device\_\_ double | [**spharea**](#function-spharea) (double Radius, double lon, double lat, double dx) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**spharea**](#function-spharea) (float Radius, float lon, float lat, float dx) <br> |




























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



Classic haversin function The function is too slow to use directly in BG\_flood engine but is more usable (i.e. naive) for model setup 


        

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


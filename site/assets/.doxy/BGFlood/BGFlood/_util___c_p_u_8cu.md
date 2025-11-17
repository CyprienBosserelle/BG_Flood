

# File Util\_CPU.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Util\_CPU.cu**](_util___c_p_u_8cu.md)

[Go to the source code of this file](_util___c_p_u_8cu_source.md)



* `#include "Util_CPU.h"`













## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**utils**](namespaceutils.md) <br> |
























## Public Functions

| Type | Name |
| ---: | :--- |
|  T | [**BarycentricInterpolation**](#function-barycentricinterpolation) (T q1, T x1, T y1, T q2, T x2, T y2, T q3, T x3, T y3, T x, T y) <br>_Barycentric interpolation within a triangle. Barycentric interpolation within a triangle defined by the vertices (x1, y1), (x2, y2), and (x3, y3). The values at the vertices are q1, q2, and q3. The function returns the interpolated value at the point (x, y)._  |
|  template float | [**BarycentricInterpolation**](#function-barycentricinterpolation) (float q1, float x1, float y1, float q2, float x2, float y2, float q3, float x3, float y3, float x, float y) <br> |
|  template double | [**BarycentricInterpolation**](#function-barycentricinterpolation) (double q1, double x1, double y1, double q2, double x2, double y2, double q3, double x3, double y3, double x, double y) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**BilinearInterpolation**](#function-bilinearinterpolation) (T q11, T q12, T q21, T q22, T x1, T x2, T y1, T y2, T x, T y) <br>_Bilinear interpolation within a rectangle. Bilinear interpolation within a rectangle defined by (x1, y1) and (x2, y2). The values at the corners of the rectangle are q11, q12, q21, and q22. The function returns the interpolated value at the point (x, y)._  __ |
|  template \_\_host\_\_ \_\_device\_\_ double | [**BilinearInterpolation&lt; double &gt;**](#function-bilinearinterpolation-double) (double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**BilinearInterpolation&lt; float &gt;**](#function-bilinearinterpolation-float) (float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y) <br> |
|  \_\_host\_\_ \_\_device\_\_ bool | [**OBBdetect**](#function-obbdetect) (T Axmin, T Axmax, T Aymin, T Aymax, T Bxmin, T Bxmax, T Bymin, T Bymax) <br>_Overlapping Bounding Box detection. Overlapping Bounding Box detection to determine if two axis-aligned bounding boxes overlap. The function takes the minimum and maximum coordinates of two bounding boxes (A and B). It returns true if the bounding boxes overlap, and false otherwise._  |
|  template \_\_host\_\_ \_\_device\_\_ bool | [**OBBdetect**](#function-obbdetect) (float Axmin, float Axmax, float Aymin, float Aymax, float Bxmin, float Bxmax, float Bymin, float Bymax) <br> |
|  template \_\_host\_\_ \_\_device\_\_ bool | [**OBBdetect**](#function-obbdetect) (double Axmin, double Axmax, double Aymin, double Aymax, double Bxmin, double Bxmax, double Bymin, double Bymax) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**calcres**](#function-calcres) (T dx, int level) <br>_Calculate the grid resolution at a given refinement level. Calculate the grid resolution at a given refinement level. If level is negative, the resolution is coarsened (doubled for each level). If level is positive, the resolution is refined (halved for each level)._  |
|  \_\_host\_\_ \_\_device\_\_ T | [**calcres**](#function-calcres) ([**Param**](class_param.md) XParam, T dx, int level) <br>_Calculate the grid resolution at a given refinement level, considering spherical coordinates. Calculate the grid resolution at a given refinement level, considering spherical coordinates. If level is negative, the resolution is coarsened (doubled for each level). If level is positive, the resolution is refined (halved for each level). If the grid is spherical, the resolution is adjusted by the Earth's radius and converted from degrees to meters._  |
|  template \_\_host\_\_ \_\_device\_\_ double | [**calcres&lt; double &gt;**](#function-calcres-double) (double dx, int level) <br> |
|  template \_\_host\_\_ \_\_device\_\_ double | [**calcres&lt; double &gt;**](#function-calcres-double) ([**Param**](class_param.md) XParam, double dx, int level) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**calcres&lt; float &gt;**](#function-calcres-float) (float dx, int level) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**calcres&lt; float &gt;**](#function-calcres-float) ([**Param**](class_param.md) XParam, float dx, int level) <br> |
|  int | [**ftoi**](#function-ftoi) (T value) <br>_Converts a floating-point number to the nearest integer. Converts a floating-point number to the nearest integer. The function rounds the value to the nearest integer, rounding halfway cases away from zero._  |
|  template int | [**ftoi&lt; double &gt;**](#function-ftoi-double) (double value) <br> |
|  template int | [**ftoi&lt; float &gt;**](#function-ftoi-float) (float value) <br> |
|  double | [**interptime**](#function-interptime) (double next, double prev, double timenext, double time) <br>_Linear interpolation between two values._  |
|  \_\_host\_\_ \_\_device\_\_ T | [**minmod2**](#function-minmod2) (T theta, T s0, T s1, T s2) <br>_Minmod limiter function for slope limiting in numerical schemes. Minmod limiter function for slope limiting in numerical schemes. The function takes a parameter theta and three slope values (s0, s1, s2). Theta is used to tune the limiting (theta=1 gives minmod, the most dissipative limiter, and theta=2 gives superbee, the least dissipative). The function returns the limited slope value based on the input slopes and theta. Usual value : float theta = 1.3f;._  |
|  template \_\_host\_\_ \_\_device\_\_ float | [**minmod2**](#function-minmod2) (float theta, float s0, float s1, float s2) <br> |
|  template \_\_host\_\_ \_\_device\_\_ double | [**minmod2**](#function-minmod2) (double theta, double s0, double s1, double s2) <br> |
|  unsigned int | [**nextPow2**](#function-nextpow2) (unsigned int x) <br>_Computes the next power of two greater than or equal to x. Computes the next power of two greater than or equal to x._  |
|  \_\_host\_\_ \_\_device\_\_ T | [**signof**](#function-signof) (T a) <br>_Returns the sign of a number. Returns the sign of a number._  |
|  template int | [**signof**](#function-signof) (int a) <br> |
|  template float | [**signof**](#function-signof) (float a) <br> |
|  template double | [**signof**](#function-signof) (double a) <br> |




























## Public Functions Documentation




### function BarycentricInterpolation 

_Barycentric interpolation within a triangle. Barycentric interpolation within a triangle defined by the vertices (x1, y1), (x2, y2), and (x3, y3). The values at the vertices are q1, q2, and q3. The function returns the interpolated value at the point (x, y)._ 
```C++
template<class T>
T BarycentricInterpolation (
    T q1,
    T x1,
    T y1,
    T q2,
    T x2,
    T y2,
    T q3,
    T x3,
    T y3,
    T x,
    T y
) 
```





**Parameters:**


* `q1` Value at (x1, y1) 
* `x1` x-coordinate of the first vertex 
* `y1` y-coordinate of the first vertex 
* `q2` Value at (x2, y2) 
* `x2` x-coordinate of the second vertex 
* `y2` y-coordinate of the second vertex 
* `q3` Value at (x3, y3) 
* `x3` x-coordinate of the third vertex 
* `y3` y-coordinate of the third vertex 
* `x` x-coordinate of the point to interpolate 
* `y` y-coordinate of the point to interpolate 



**Returns:**

Interpolated value at (x, y) 





        

<hr>



### function BarycentricInterpolation 

```C++
template float BarycentricInterpolation (
    float q1,
    float x1,
    float y1,
    float q2,
    float x2,
    float y2,
    float q3,
    float x3,
    float y3,
    float x,
    float y
) 
```




<hr>



### function BarycentricInterpolation 

```C++
template double BarycentricInterpolation (
    double q1,
    double x1,
    double y1,
    double q2,
    double x2,
    double y2,
    double q3,
    double x3,
    double y3,
    double x,
    double y
) 
```




<hr>



### function BilinearInterpolation 

_Bilinear interpolation within a rectangle. Bilinear interpolation within a rectangle defined by (x1, y1) and (x2, y2). The values at the corners of the rectangle are q11, q12, q21, and q22. The function returns the interpolated value at the point (x, y)._  __
```C++
template<class T>
__host__ __device__ T BilinearInterpolation (
    T q11,
    T q12,
    T q21,
    T q22,
    T x1,
    T x2,
    T y1,
    T y2,
    T x,
    T y
) 
```





**Parameters:**


* `q11` Value at (x1, y1) 
* `q12` Value at (x1, y2) 
* `q21` Value at (x2, y1) 
* `q22` Value at (x2, y2) 
* `x1` x-coordinate of the bottom-left corner 
* `x2` x-coordinate of the top-right corner 
* `y1` y-coordinate of the bottom-left corner 
* `y2` y-coordinate of the top-right corner 
* `x` x-coordinate of the point to interpolate 
* `y` y-coordinate of the point to interpolate 




        

<hr>



### function BilinearInterpolation&lt; double &gt; 

```C++
template __host__ __device__ double BilinearInterpolation< double > (
    double q11,
    double q12,
    double q21,
    double q22,
    double x1,
    double x2,
    double y1,
    double y2,
    double x,
    double y
) 
```




<hr>



### function BilinearInterpolation&lt; float &gt; 

```C++
template __host__ __device__ float BilinearInterpolation< float > (
    float q11,
    float q12,
    float q21,
    float q22,
    float x1,
    float x2,
    float y1,
    float y2,
    float x,
    float y
) 
```




<hr>



### function OBBdetect 

_Overlapping Bounding Box detection. Overlapping Bounding Box detection to determine if two axis-aligned bounding boxes overlap. The function takes the minimum and maximum coordinates of two bounding boxes (A and B). It returns true if the bounding boxes overlap, and false otherwise._ 
```C++
template<class T>
__host__ __device__ bool OBBdetect (
    T Axmin,
    T Axmax,
    T Aymin,
    T Aymax,
    T Bxmin,
    T Bxmax,
    T Bymin,
    T Bymax
) 
```





**Parameters:**


* `Axmin` Minimum x-coordinate of bounding box A. 
* `Axmax` Maximum x-coordinate of bounding box A. 
* `Aymin` Minimum y-coordinate of bounding box A. 
* `Aymax` Maximum y-coordinate of bounding box A. 
* `Bxmin` Minimum x-coordinate of bounding box B. 
* `Bxmax` Maximum x-coordinate of bounding box B. 
* `Bymin` Minimum y-coordinate of bounding box B. 
* `Bymax` Maximum y-coordinate of bounding box B. 



**Returns:**

True if the bounding boxes overlap, false otherwise.


Overlaping Bounding Box to detect which cell river falls into. It is the simplest version of the algorythm where the bounding box are paralle;l to the axis 


        

<hr>



### function OBBdetect 

```C++
template __host__ __device__ bool OBBdetect (
    float Axmin,
    float Axmax,
    float Aymin,
    float Aymax,
    float Bxmin,
    float Bxmax,
    float Bymin,
    float Bymax
) 
```




<hr>



### function OBBdetect 

```C++
template __host__ __device__ bool OBBdetect (
    double Axmin,
    double Axmax,
    double Aymin,
    double Aymax,
    double Bxmin,
    double Bxmax,
    double Bymin,
    double Bymax
) 
```




<hr>



### function calcres 

_Calculate the grid resolution at a given refinement level. Calculate the grid resolution at a given refinement level. If level is negative, the resolution is coarsened (doubled for each level). If level is positive, the resolution is refined (halved for each level)._ 
```C++
template<class T>
__host__ __device__ T calcres (
    T dx,
    int level
) 
```





**Parameters:**


* `dx` The base grid resolution. 
* `level` The refinement level (negative for coarsening, positive for refining). 



**Returns:**

The calculated grid resolution at the specified level. 





        

<hr>



### function calcres 

_Calculate the grid resolution at a given refinement level, considering spherical coordinates. Calculate the grid resolution at a given refinement level, considering spherical coordinates. If level is negative, the resolution is coarsened (doubled for each level). If level is positive, the resolution is refined (halved for each level). If the grid is spherical, the resolution is adjusted by the Earth's radius and converted from degrees to meters._ 
```C++
template<class T>
__host__ __device__ T calcres (
    Param XParam,
    T dx,
    int level
) 
```





**Parameters:**


* `XParam` The parameter object containing grid settings. 
* `dx` The base grid resolution. 
* `level` The refinement level (negative for coarsening, positive for refining). 



**Returns:**

The calculated grid resolution at the specified level, adjusted for spherical coordinates if applicable. 





        

<hr>



### function calcres&lt; double &gt; 

```C++
template __host__ __device__ double calcres< double > (
    double dx,
    int level
) 
```




<hr>



### function calcres&lt; double &gt; 

```C++
template __host__ __device__ double calcres< double > (
    Param XParam,
    double dx,
    int level
) 
```




<hr>



### function calcres&lt; float &gt; 

```C++
template __host__ __device__ float calcres< float > (
    float dx,
    int level
) 
```




<hr>



### function calcres&lt; float &gt; 

```C++
template __host__ __device__ float calcres< float > (
    Param XParam,
    float dx,
    int level
) 
```




<hr>



### function ftoi 

_Converts a floating-point number to the nearest integer. Converts a floating-point number to the nearest integer. The function rounds the value to the nearest integer, rounding halfway cases away from zero._ 
```C++
template<class T>
int ftoi (
    T value
) 
```





**Parameters:**


* `value` The floating-point number to convert. 



**Returns:**

The nearest integer. 





        

<hr>



### function ftoi&lt; double &gt; 

```C++
template int ftoi< double > (
    double value
) 
```




<hr>



### function ftoi&lt; float &gt; 

```C++
template int ftoi< float > (
    float value
) 
```




<hr>



### function interptime 

_Linear interpolation between two values._ 
```C++
double interptime (
    double next,
    double prev,
    double timenext,
    double time
) 
```




<hr>



### function minmod2 

_Minmod limiter function for slope limiting in numerical schemes. Minmod limiter function for slope limiting in numerical schemes. The function takes a parameter theta and three slope values (s0, s1, s2). Theta is used to tune the limiting (theta=1 gives minmod, the most dissipative limiter, and theta=2 gives superbee, the least dissipative). The function returns the limited slope value based on the input slopes and theta. Usual value : float theta = 1.3f;._ 
```C++
template<class T>
__host__ __device__ T minmod2 (
    T theta,
    T s0,
    T s1,
    T s2
) 
```





**Parameters:**


* `theta` The tuning parameter for the limiter (between 1 and 2). 
* `s0` The slope value at the left cell. 
* `s1` The slope value at the center cell. 
* `s2` The slope value at the right cell. 



**Returns:**

The limited slope value. 





        

<hr>



### function minmod2 

```C++
template __host__ __device__ float minmod2 (
    float theta,
    float s0,
    float s1,
    float s2
) 
```




<hr>



### function minmod2 

```C++
template __host__ __device__ double minmod2 (
    double theta,
    double s0,
    double s1,
    double s2
) 
```




<hr>



### function nextPow2 

_Computes the next power of two greater than or equal to x. Computes the next power of two greater than or equal to x._ 
```C++
unsigned int nextPow2 (
    unsigned int x
) 
```




<hr>



### function signof 

_Returns the sign of a number. Returns the sign of a number._ 
```C++
template<class T>
__host__ __device__ T signof (
    T a
) 
```




<hr>



### function signof 

```C++
template int signof (
    int a
) 
```




<hr>



### function signof 

```C++
template float signof (
    float a
) 
```




<hr>



### function signof 

```C++
template double signof (
    double a
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Util_CPU.cu`


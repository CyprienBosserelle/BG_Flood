

# File Util\_CPU.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Util\_CPU.cu**](Util__CPU_8cu.md)

[Go to the source code of this file](Util__CPU_8cu_source.md)



* `#include "Util_CPU.h"`













## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**utils**](namespaceutils.md) <br> |
























## Public Functions

| Type | Name |
| ---: | :--- |
|  T | [**BarycentricInterpolation**](#function-barycentricinterpolation) (T q1, T x1, T y1, T q2, T x2, T y2, T q3, T x3, T y3, T x, T y) <br> |
|  template float | [**BarycentricInterpolation**](#function-barycentricinterpolation) (float q1, float x1, float y1, float q2, float x2, float y2, float q3, float x3, float y3, float x, float y) <br> |
|  template double | [**BarycentricInterpolation**](#function-barycentricinterpolation) (double q1, double x1, double y1, double q2, double x2, double y2, double q3, double x3, double y3, double x, double y) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**BilinearInterpolation**](#function-bilinearinterpolation) (T q11, T q12, T q21, T q22, T x1, T x2, T y1, T y2, T x, T y) <br> |
|  template \_\_host\_\_ \_\_device\_\_ double | [**BilinearInterpolation&lt; double &gt;**](#function-bilinearinterpolation-double) (double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**BilinearInterpolation&lt; float &gt;**](#function-bilinearinterpolation-float) (float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y) <br> |
|  \_\_host\_\_ \_\_device\_\_ bool | [**OBBdetect**](#function-obbdetect) (T Axmin, T Axmax, T Aymin, T Aymax, T Bxmin, T Bxmax, T Bymin, T Bymax) <br> |
|  template \_\_host\_\_ \_\_device\_\_ bool | [**OBBdetect**](#function-obbdetect) (float Axmin, float Axmax, float Aymin, float Aymax, float Bxmin, float Bxmax, float Bymin, float Bymax) <br> |
|  template \_\_host\_\_ \_\_device\_\_ bool | [**OBBdetect**](#function-obbdetect) (double Axmin, double Axmax, double Aymin, double Aymax, double Bxmin, double Bxmax, double Bymin, double Bymax) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**calcres**](#function-calcres) (T dx, int level) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**calcres**](#function-calcres) ([**Param**](classParam.md) XParam, T dx, int level) <br> |
|  template \_\_host\_\_ \_\_device\_\_ double | [**calcres&lt; double &gt;**](#function-calcres-double) (double dx, int level) <br> |
|  template \_\_host\_\_ \_\_device\_\_ double | [**calcres&lt; double &gt;**](#function-calcres-double) ([**Param**](classParam.md) XParam, double dx, int level) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**calcres&lt; float &gt;**](#function-calcres-float) (float dx, int level) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**calcres&lt; float &gt;**](#function-calcres-float) ([**Param**](classParam.md) XParam, float dx, int level) <br> |
|  int | [**ftoi**](#function-ftoi) (T value) <br> |
|  template int | [**ftoi&lt; double &gt;**](#function-ftoi-double) (double value) <br> |
|  template int | [**ftoi&lt; float &gt;**](#function-ftoi-float) (float value) <br> |
|  double | [**interptime**](#function-interptime) (double next, double prev, double timenext, double time) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**minmod2**](#function-minmod2) (T theta, T s0, T s1, T s2) <br> |
|  template \_\_host\_\_ \_\_device\_\_ float | [**minmod2**](#function-minmod2) (float theta, float s0, float s1, float s2) <br> |
|  template \_\_host\_\_ \_\_device\_\_ double | [**minmod2**](#function-minmod2) (double theta, double s0, double s1, double s2) <br> |
|  unsigned int | [**nextPow2**](#function-nextpow2) (unsigned int x) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**signof**](#function-signof) (T a) <br> |
|  template int | [**signof**](#function-signof) (int a) <br> |
|  template float | [**signof**](#function-signof) (float a) <br> |
|  template double | [**signof**](#function-signof) (double a) <br> |




























## Public Functions Documentation




### function BarycentricInterpolation 

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

```C++
template<class T>
__host__ __device__ T calcres (
    T dx,
    int level
) 
```




<hr>



### function calcres 

```C++
template<class T>
__host__ __device__ T calcres (
    Param XParam,
    T dx,
    int level
) 
```




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

```C++
template<class T>
int ftoi (
    T value
) 
```




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

```C++
template<class T>
__host__ __device__ T minmod2 (
    T theta,
    T s0,
    T s1,
    T s2
) 
```




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

```C++
unsigned int nextPow2 (
    unsigned int x
) 
```




<hr>



### function signof 

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


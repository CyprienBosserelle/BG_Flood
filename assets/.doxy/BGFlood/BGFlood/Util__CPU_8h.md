

# File Util\_CPU.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Util\_CPU.h**](Util__CPU_8h.md)

[Go to the source code of this file](Util__CPU_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`













## Namespaces

| Type | Name |
| ---: | :--- |
| namespace | [**utils**](namespaceutils.md) <br> |
























## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ \_\_device\_\_ T | [**BarycentricInterpolation**](#function-barycentricinterpolation) (T q1, T x1, T y1, T q2, T x2, T y2, T q3, T x3, T y3, T x, T y) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**BilinearInterpolation**](#function-bilinearinterpolation) (T q11, T q12, T q21, T q22, T x1, T x2, T y1, T y2, T x, T y) <br> |
|  \_\_host\_\_ \_\_device\_\_ bool | [**OBBdetect**](#function-obbdetect) (T Axmin, T Axmax, T Aymin, T Aymax, T Bxmin, T Bxmax, T Bymin, T Bymax) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**calcres**](#function-calcres) (T dx, int level) <br> |
|  int | [**ftoi**](#function-ftoi) (T value) <br> |
|  double | [**interptime**](#function-interptime) (double next, double prev, double timenext, double time) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**minmod2**](#function-minmod2) (T theta, T s0, T s1, T s2) <br> |
|  unsigned int | [**nextPow2**](#function-nextpow2) (unsigned int x) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**signof**](#function-signof) (T a) <br> |




























## Public Functions Documentation




### function BarycentricInterpolation 

```C++
template<class T>
__host__ __device__ T BarycentricInterpolation (
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



### function calcres 

```C++
template<class T>
__host__ __device__ T calcres (
    T dx,
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

------------------------------
The documentation for this class was generated from the following file `src/Util_CPU.h`


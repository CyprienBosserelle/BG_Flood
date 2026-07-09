

# Namespace utils



[**Namespace List**](namespaces.md) **>** [**utils**](namespaceutils.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|  \_\_host\_\_ \_\_device\_\_ const T & | [**max**](#function-max) (const T & a, const T & b) <br>_Generic max function._  |
|  template \_\_host\_\_ \_\_device\_\_ const double & | [**max&lt; double &gt;**](#function-max-double) (const double & a, const double & b) <br> |
|  template \_\_host\_\_ \_\_device\_\_ const float & | [**max&lt; float &gt;**](#function-max-float) (const float & a, const float & b) <br> |
|  template \_\_host\_\_ \_\_device\_\_ const int & | [**max&lt; int &gt;**](#function-max-int) (const int & a, const int & b) <br> |
|  \_\_host\_\_ \_\_device\_\_ const T & | [**min**](#function-min) (const T & a, const T & b) <br>_Generic min function._  |
|  template \_\_host\_\_ \_\_device\_\_ const double & | [**min&lt; double &gt;**](#function-min-double) (const double & a, const double & b) <br> |
|  template \_\_host\_\_ \_\_device\_\_ const float & | [**min&lt; float &gt;**](#function-min-float) (const float & a, const float & b) <br> |
|  template \_\_host\_\_ \_\_device\_\_ const int & | [**min&lt; int &gt;**](#function-min-int) (const int & a, const int & b) <br> |
|  \_\_host\_\_ \_\_device\_\_ const T & | [**nearest**](#function-nearest) (const T & a, const T & b, const T & c) <br>_Generic nearest value function to a given value c._  |
|  \_\_host\_\_ \_\_device\_\_ const T & | [**nearest**](#function-nearest) (const T & a, const T & b) <br>_Generic nearest value function to 0.0 between 2 parameter._  |
|  template \_\_host\_\_ \_\_device\_\_ const double & | [**nearest&lt; double &gt;**](#function-nearest-double) (const double & a, const double & b, const double & c) <br> |
|  template \_\_host\_\_ \_\_device\_\_ const double & | [**nearest&lt; double &gt;**](#function-nearest-double) (const double & a, const double & b) <br> |
|  template \_\_host\_\_ \_\_device\_\_ const float & | [**nearest&lt; float &gt;**](#function-nearest-float) (const float & a, const float & b, const float & c) <br> |
|  template \_\_host\_\_ \_\_device\_\_ const float & | [**nearest&lt; float &gt;**](#function-nearest-float) (const float & a, const float & b) <br> |
|  template \_\_host\_\_ \_\_device\_\_ const int & | [**nearest&lt; int &gt;**](#function-nearest-int) (const int & a, const int & b, const int & c) <br> |
|  template \_\_host\_\_ \_\_device\_\_ const int & | [**nearest&lt; int &gt;**](#function-nearest-int) (const int & a, const int & b) <br> |
|  \_\_host\_\_ \_\_device\_\_ T | [**sq**](#function-sq) (T a) <br>_Generic squaring function._  |
|  template double \_\_host\_\_ \_\_device\_\_ | [**sq&lt; double &gt;**](#function-sq-double) (double a) <br> |
|  template float \_\_host\_\_ \_\_device\_\_ | [**sq&lt; float &gt;**](#function-sq-float) (float a) <br> |
|  template int \_\_host\_\_ \_\_device\_\_ | [**sq&lt; int &gt;**](#function-sq-int) (int a) <br> |




























## Public Functions Documentation




### function max 

_Generic max function._ 
```C++
template<class T>
__host__ __device__ const T & utils::max (
    const T & a,
    const T & b
) 
```



! 


        

<hr>



### function max&lt; double &gt; 

```C++
template __host__ __device__ const double & utils::max< double > (
    const double & a,
    const double & b
) 
```




<hr>



### function max&lt; float &gt; 

```C++
template __host__ __device__ const float & utils::max< float > (
    const float & a,
    const float & b
) 
```




<hr>



### function max&lt; int &gt; 

```C++
template __host__ __device__ const int & utils::max< int > (
    const int & a,
    const int & b
) 
```




<hr>



### function min 

_Generic min function._ 
```C++
template<class T>
__host__ __device__ const T & utils::min (
    const T & a,
    const T & b
) 
```



! 


        

<hr>



### function min&lt; double &gt; 

```C++
template __host__ __device__ const double & utils::min< double > (
    const double & a,
    const double & b
) 
```




<hr>



### function min&lt; float &gt; 

```C++
template __host__ __device__ const float & utils::min< float > (
    const float & a,
    const float & b
) 
```




<hr>



### function min&lt; int &gt; 

```C++
template __host__ __device__ const int & utils::min< int > (
    const int & a,
    const int & b
) 
```




<hr>



### function nearest 

_Generic nearest value function to a given value c._ 
```C++
template<class T>
__host__ __device__ const T & utils::nearest (
    const T & a,
    const T & b,
    const T & c
) 
```



! 


        

<hr>



### function nearest 

_Generic nearest value function to 0.0 between 2 parameter._ 
```C++
template<class T>
__host__ __device__ const T & utils::nearest (
    const T & a,
    const T & b
) 
```




<hr>



### function nearest&lt; double &gt; 

```C++
template __host__ __device__ const double & utils::nearest< double > (
    const double & a,
    const double & b,
    const double & c
) 
```




<hr>



### function nearest&lt; double &gt; 

```C++
template __host__ __device__ const double & utils::nearest< double > (
    const double & a,
    const double & b
) 
```




<hr>



### function nearest&lt; float &gt; 

```C++
template __host__ __device__ const float & utils::nearest< float > (
    const float & a,
    const float & b,
    const float & c
) 
```




<hr>



### function nearest&lt; float &gt; 

```C++
template __host__ __device__ const float & utils::nearest< float > (
    const float & a,
    const float & b
) 
```




<hr>



### function nearest&lt; int &gt; 

```C++
template __host__ __device__ const int & utils::nearest< int > (
    const int & a,
    const int & b,
    const int & c
) 
```




<hr>



### function nearest&lt; int &gt; 

```C++
template __host__ __device__ const int & utils::nearest< int > (
    const int & a,
    const int & b
) 
```




<hr>



### function sq 

_Generic squaring function._ 
```C++
template<class T>
__host__ __device__ T utils::sq (
    T a
) 
```



! 


        

<hr>



### function sq&lt; double &gt; 

```C++
template double __host__ __device__ utils::sq< double > (
    double a
) 
```




<hr>



### function sq&lt; float &gt; 

```C++
template float __host__ __device__ utils::sq< float > (
    float a
) 
```




<hr>



### function sq&lt; int &gt; 

```C++
template int __host__ __device__ utils::sq< int > (
    int a
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Util_CPU.cu`


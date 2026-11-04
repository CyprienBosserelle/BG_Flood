

# Struct Loop

**template &lt;class T&gt;**



[**ClassList**](annotated.md) **>** [**Loop**](structLoop.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  std::vector&lt; std::vector&lt; [**Pointout**](classPointout.md) &gt; &gt; | [**TSAllout**](#variable-tsallout)  <br> |
|  int | [**atmpstep**](#variable-atmpstep)   = `1`<br> |
|  T | [**atmpuni**](#variable-atmpuni)  <br> |
|  bool | [**atmpuniform**](#variable-atmpuniform)  <br> |
|  dim3 | [**blockDim**](#variable-blockdim)  <br> |
|  double | [**dt**](#variable-dt)  <br> |
|  double | [**dtmax**](#variable-dtmax)  <br> |
|  T | [**epsilon**](#variable-epsilon)  <br> |
|  dim3 | [**gridDim**](#variable-griddim)  <br> |
|  T | [**hugenegval**](#variable-hugenegval)  <br> |
|  T | [**hugeposval**](#variable-hugeposval)  <br> |
|  int | [**indNextoutputtime**](#variable-indnextoutputtime)   = `0`<br> |
|  int | [**nTSsteps**](#variable-ntssteps)   = `0`<br> |
|  double | [**nextoutputtime**](#variable-nextoutputtime)  <br> |
|  int | [**nstep**](#variable-nstep)   = `0`<br> |
|  int | [**nstepout**](#variable-nstepout)   = `0`<br> |
|  const int | [**num\_streams**](#variable-num_streams)   = `4`<br> |
|  int | [**rainstep**](#variable-rainstep)   = `1`<br> |
|  T | [**rainuni**](#variable-rainuni)   = `T(0.0)`<br> |
|  bool | [**rainuniform**](#variable-rainuniform)  <br> |
|  cudaStream\_t | [**streams**](#variable-streams)  <br> |
|  double | [**totaltime**](#variable-totaltime)  <br> |
|  T | [**uwinduni**](#variable-uwinduni)   = `T(0.0)`<br> |
|  T | [**vwinduni**](#variable-vwinduni)   = `T(0.0)`<br> |
|  int | [**windstep**](#variable-windstep)   = `1`<br> |
|  bool | [**winduniform**](#variable-winduniform)  <br> |












































## Public Attributes Documentation




### variable TSAllout 

```C++
std::vector< std::vector< Pointout > > Loop< T >::TSAllout;
```




<hr>



### variable atmpstep 

```C++
int Loop< T >::atmpstep;
```




<hr>



### variable atmpuni 

```C++
T Loop< T >::atmpuni;
```




<hr>



### variable atmpuniform 

```C++
bool Loop< T >::atmpuniform;
```




<hr>



### variable blockDim 

```C++
dim3 Loop< T >::blockDim;
```




<hr>



### variable dt 

```C++
double Loop< T >::dt;
```




<hr>



### variable dtmax 

```C++
double Loop< T >::dtmax;
```




<hr>



### variable epsilon 

```C++
T Loop< T >::epsilon;
```




<hr>



### variable gridDim 

```C++
dim3 Loop< T >::gridDim;
```




<hr>



### variable hugenegval 

```C++
T Loop< T >::hugenegval;
```




<hr>



### variable hugeposval 

```C++
T Loop< T >::hugeposval;
```




<hr>



### variable indNextoutputtime 

```C++
int Loop< T >::indNextoutputtime;
```




<hr>



### variable nTSsteps 

```C++
int Loop< T >::nTSsteps;
```




<hr>



### variable nextoutputtime 

```C++
double Loop< T >::nextoutputtime;
```




<hr>



### variable nstep 

```C++
int Loop< T >::nstep;
```




<hr>



### variable nstepout 

```C++
int Loop< T >::nstepout;
```




<hr>



### variable num\_streams 

```C++
const int Loop< T >::num_streams;
```




<hr>



### variable rainstep 

```C++
int Loop< T >::rainstep;
```




<hr>



### variable rainuni 

```C++
T Loop< T >::rainuni;
```




<hr>



### variable rainuniform 

```C++
bool Loop< T >::rainuniform;
```




<hr>



### variable streams 

```C++
cudaStream_t Loop< T >::streams[4];
```




<hr>



### variable totaltime 

```C++
double Loop< T >::totaltime;
```




<hr>



### variable uwinduni 

```C++
T Loop< T >::uwinduni;
```




<hr>



### variable vwinduni 

```C++
T Loop< T >::vwinduni;
```




<hr>



### variable windstep 

```C++
int Loop< T >::windstep;
```




<hr>



### variable winduniform 

```C++
bool Loop< T >::winduniform;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Arrays.h`


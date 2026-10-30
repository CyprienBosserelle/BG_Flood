

# Struct DynForcingP

**template &lt;class T&gt;**



[**ClassList**](annotated.md) **>** [**DynForcingP**](structDynForcingP.md)








Inherits the following classes: [forcingmap](classforcingmap.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  T \* | [**after**](#variable-after)  <br> |
|  T \* | [**after\_g**](#variable-after_g)  <br> |
|  T \* | [**before**](#variable-before)  <br> |
|  T \* | [**before\_g**](#variable-before_g)  <br> |
|  T | [**clampedge**](#variable-clampedge)   = `0.0`<br> |
|  T \* | [**now**](#variable-now)  <br> |
|  T \* | [**now\_g**](#variable-now_g)  <br> |
|  T \* | [**val**](#variable-val)  <br> |


## Public Attributes inherited from forcingmap

See [forcingmap](classforcingmap.md)

| Type | Name |
| ---: | :--- |
|  [**TexSetP**](structTexSetP.md) | [**GPU**](classforcingmap.md#variable-gpu)  <br> |
|  double | [**dt**](classforcingmap.md#variable-dt)  <br> |
|  std::string | [**inputfile**](classforcingmap.md#variable-inputfile)  <br> |
|  int | [**instep**](classforcingmap.md#variable-instep)   = `0`<br> |
|  double | [**nowvalue**](classforcingmap.md#variable-nowvalue)  <br> |
|  int | [**nt**](classforcingmap.md#variable-nt)  <br> |
|  double | [**tmax**](classforcingmap.md#variable-tmax)  <br> |
|  double | [**to**](classforcingmap.md#variable-to)  <br> |
|  std::vector&lt; [**Windin**](classWindin.md) &gt; | [**unidata**](classforcingmap.md#variable-unidata)  <br> |
|  bool | [**uniform**](classforcingmap.md#variable-uniform)   = `false`<br> |


## Public Attributes inherited from inputmap

See [inputmap](classinputmap.md)

| Type | Name |
| ---: | :--- |
|  double | [**denanval**](classinputmap.md#variable-denanval)   = `NAN`<br> |
|  double | [**dx**](classinputmap.md#variable-dx)   = `0.0`<br> |
|  double | [**dy**](classinputmap.md#variable-dy)   = `0.0`<br> |
|  std::string | [**extension**](classinputmap.md#variable-extension)  <br> |
|  bool | [**flipxx**](classinputmap.md#variable-flipxx)   = `false`<br> |
|  bool | [**flipyy**](classinputmap.md#variable-flipyy)   = `false`<br> |
|  double | [**grdalpha**](classinputmap.md#variable-grdalpha)   = `0.0`<br> |
|  std::string | [**inputfile**](classinputmap.md#variable-inputfile)  <br> |
|  int | [**nx**](classinputmap.md#variable-nx)   = `0`<br> |
|  int | [**ny**](classinputmap.md#variable-ny)   = `0`<br> |
|  std::string | [**varname**](classinputmap.md#variable-varname)  <br> |
|  double | [**xmax**](classinputmap.md#variable-xmax)   = `0.0`<br> |
|  double | [**xo**](classinputmap.md#variable-xo)   = `0.0`<br> |
|  double | [**ymax**](classinputmap.md#variable-ymax)   = `0.0`<br> |
|  double | [**yo**](classinputmap.md#variable-yo)   = `0.0`<br> |
































































































































## Public Attributes Documentation




### variable after 

```C++
T * DynForcingP< T >::after;
```




<hr>



### variable after\_g 

```C++
T * DynForcingP< T >::after_g;
```




<hr>



### variable before 

```C++
T* DynForcingP< T >::before;
```




<hr>



### variable before\_g 

```C++
T* DynForcingP< T >::before_g;
```




<hr>



### variable clampedge 

```C++
T DynForcingP< T >::clampedge;
```




<hr>



### variable now 

```C++
T* DynForcingP< T >::now;
```




<hr>



### variable now\_g 

```C++
T* DynForcingP< T >::now_g;
```




<hr>



### variable val 

```C++
T* DynForcingP< T >::val;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Forcing.h`


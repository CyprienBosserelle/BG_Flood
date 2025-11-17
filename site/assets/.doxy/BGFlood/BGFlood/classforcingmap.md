

# Class forcingmap



[**ClassList**](annotated.md) **>** [**forcingmap**](classforcingmap.md)








Inherits the following classes: [inputmap](classinputmap.md)


Inherited by the following classes: [DynForcingP](struct_dyn_forcing_p.md),  [DynForcingP](struct_dyn_forcing_p.md)




















## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**TexSetP**](struct_tex_set_p.md) | [**GPU**](#variable-gpu)  <br> |
|  double | [**dt**](#variable-dt)  <br> |
|  std::string | [**inputfile**](#variable-inputfile)  <br> |
|  int | [**instep**](#variable-instep)   = `0`<br> |
|  double | [**nowvalue**](#variable-nowvalue)  <br> |
|  int | [**nt**](#variable-nt)  <br> |
|  double | [**tmax**](#variable-tmax)  <br> |
|  double | [**to**](#variable-to)  <br> |
|  std::vector&lt; [**Windin**](class_windin.md) &gt; | [**unidata**](#variable-unidata)  <br> |
|  bool | [**uniform**](#variable-uniform)   = `false`<br> |


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




### variable GPU 

```C++
TexSetP forcingmap::GPU;
```




<hr>



### variable dt 

```C++
double forcingmap::dt;
```




<hr>



### variable inputfile 

```C++
std::string forcingmap::inputfile;
```




<hr>



### variable instep 

```C++
int forcingmap::instep;
```




<hr>



### variable nowvalue 

```C++
double forcingmap::nowvalue;
```




<hr>



### variable nt 

```C++
int forcingmap::nt;
```




<hr>



### variable tmax 

```C++
double forcingmap::tmax;
```




<hr>



### variable to 

```C++
double forcingmap::to;
```




<hr>



### variable unidata 

```C++
std::vector<Windin> forcingmap::unidata;
```




<hr>



### variable uniform 

```C++
bool forcingmap::uniform;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Forcing.h`


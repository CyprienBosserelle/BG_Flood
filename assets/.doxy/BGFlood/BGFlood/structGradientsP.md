

# Struct GradientsP

**template &lt;class T&gt;**



[**ClassList**](annotated.md) **>** [**GradientsP**](structGradientsP.md)



_Structure holding gradient arrays for physical variables._ [More...](#detailed-description)

* `#include <Arrays.h>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  T \* | [**dhdx**](#variable-dhdx)  <br> |
|  T \* | [**dhdy**](#variable-dhdy)  <br> |
|  T \* | [**dudx**](#variable-dudx)  <br> |
|  T \* | [**dudy**](#variable-dudy)  <br> |
|  T \* | [**dvdx**](#variable-dvdx)  <br> |
|  T \* | [**dvdy**](#variable-dvdy)  <br> |
|  T \* | [**dzbdx**](#variable-dzbdx)  <br> |
|  T \* | [**dzbdy**](#variable-dzbdy)  <br> |
|  T \* | [**dzsdx**](#variable-dzsdx)  <br> |
|  T \* | [**dzsdy**](#variable-dzsdy)  <br> |












































## Detailed Description




**Template parameters:**


* `T` Data type 




    
## Public Attributes Documentation




### variable dhdx 

```C++
T* GradientsP< T >::dhdx;
```



Water depth gradient in x-direction 


        

<hr>



### variable dhdy 

```C++
T* GradientsP< T >::dhdy;
```



Water depth gradient in y-direction 


        

<hr>



### variable dudx 

```C++
T* GradientsP< T >::dudx;
```



Velocity u gradient in x-direction 


        

<hr>



### variable dudy 

```C++
T* GradientsP< T >::dudy;
```



Velocity u gradient in y-direction 


        

<hr>



### variable dvdx 

```C++
T* GradientsP< T >::dvdx;
```



Velocity v gradient in x-direction 


        

<hr>



### variable dvdy 

```C++
T* GradientsP< T >::dvdy;
```



Velocity v gradient in y-direction 


        

<hr>



### variable dzbdx 

```C++
T* GradientsP< T >::dzbdx;
```



Bed elevation gradient in x-direction 


        

<hr>



### variable dzbdy 

```C++
T* GradientsP< T >::dzbdy;
```



Bed elevation gradient in y-direction 


        

<hr>



### variable dzsdx 

```C++
T* GradientsP< T >::dzsdx;
```



Surface elevation gradient in x-direction 


        

<hr>



### variable dzsdy 

```C++
T* GradientsP< T >::dzsdy;
```



Surface elevation gradient in y-direction 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Arrays.h`




# Struct GradientsMLP

**template &lt;class T&gt;**



[**ClassList**](annotated.md) **>** [**GradientsMLP**](struct_gradients_m_l_p.md)



_Structure holding gradient arrays (no z relative variables)._ [More...](#detailed-description)

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












































## Detailed Description




**Template parameters:**


* `T` Data type 




    
## Public Attributes Documentation




### variable dhdx 

```C++
T* GradientsMLP< T >::dhdx;
```



Water depth gradient in x-direction 


        

<hr>



### variable dhdy 

```C++
T* GradientsMLP< T >::dhdy;
```



Water depth gradient in y-direction 


        

<hr>



### variable dudx 

```C++
T* GradientsMLP< T >::dudx;
```



Velocity u gradient in x-direction 


        

<hr>



### variable dudy 

```C++
T* GradientsMLP< T >::dudy;
```



Velocity u gradient in y-direction 


        

<hr>



### variable dvdx 

```C++
T* GradientsMLP< T >::dvdx;
```



Velocity v gradient in x-direction 


        

<hr>



### variable dvdy 

```C++
T* GradientsMLP< T >::dvdy;
```



Velocity v gradient in y-direction 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Arrays.h`


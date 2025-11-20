

# Struct FluxP

**template &lt;class T&gt;**



[**ClassList**](annotated.md) **>** [**FluxP**](structFluxP.md)



_Structure holding flux variables for advection._ [More...](#detailed-description)

* `#include <Arrays.h>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  T \* | [**Fhu**](#variable-fhu)  <br> |
|  T \* | [**Fhv**](#variable-fhv)  <br> |
|  T \* | [**Fqux**](#variable-fqux)  <br> |
|  T \* | [**Fquy**](#variable-fquy)  <br> |
|  T \* | [**Fqvx**](#variable-fqvx)  <br> |
|  T \* | [**Fqvy**](#variable-fqvy)  <br> |
|  T \* | [**Su**](#variable-su)  <br> |
|  T \* | [**Sv**](#variable-sv)  <br> |












































## Detailed Description




**Template parameters:**


* `T` Data type 




    
## Public Attributes Documentation




### variable Fhu 

```C++
T* FluxP< T >::Fhu;
```



Flux of h in u-direction 


        

<hr>



### variable Fhv 

```C++
T* FluxP< T >::Fhv;
```



Flux of h in v-direction 


        

<hr>



### variable Fqux 

```C++
T* FluxP< T >::Fqux;
```



Flux of u in x-direction 


        

<hr>



### variable Fquy 

```C++
T* FluxP< T >::Fquy;
```



Flux of u in y-direction 


        

<hr>



### variable Fqvx 

```C++
T* FluxP< T >::Fqvx;
```



Flux of v in x-direction 


        

<hr>



### variable Fqvy 

```C++
T* FluxP< T >::Fqvy;
```



Flux of v in y-direction 


        

<hr>



### variable Su 

```C++
T* FluxP< T >::Su;
```



Source term for u 


        

<hr>



### variable Sv 

```C++
T* FluxP< T >::Sv;
```



Source term for v 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Arrays.h`


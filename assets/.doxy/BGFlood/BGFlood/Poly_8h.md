

# File Poly.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Poly.h**](Poly_8h.md)

[Go to the source code of this file](Poly_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Input.h"`
* `#include "Write_txtlog.h"`
* `#include "Util_CPU.h"`
* `#include "Forcing.h"`
* `#include "Arrays.h"`
* `#include "MemManagement.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  [**Polygon**](classPolygon.md) | [**CounterCWPoly**](#function-countercwpoly) ([**Polygon**](classPolygon.md) Poly) <br>_check polygon handedness and reverse if necessary._  |
|  bool | [**blockinpoly**](#function-blockinpoly) (T xo, T yo, T dx, int blkwidth, [**Polygon**](classPolygon.md) Poly) <br>_Check whether a block is inside or intersects a polygon._  |
|  int | [**wn\_PnPoly**](#function-wn_pnpoly) (T Px, T Py, [**Polygon**](classPolygon.md) Poly) <br>_winding number test for a point in a polygon_  |




























## Public Functions Documentation




### function CounterCWPoly 

_check polygon handedness and reverse if necessary._ 
```C++
Polygon CounterCWPoly (
    Polygon Poly
) 
```



### Description



check polygon handedness and enforce left-handesness (Counter-clockwise). This function is used to ensure the right polygon handedness for the winding number inpoly (using the isleft()) 



        

<hr>



### function blockinpoly 

_Check whether a block is inside or intersects a polygon._ 
```C++
template<class T>
bool blockinpoly (
    T xo,
    T yo,
    T dx,
    int blkwidth,
    Polygon Poly
) 
```



Determines if any corner of the block is inside the polygon or if the block intersects the polygon.




**Template parameters:**


* `T` Coordinate type 



**Parameters:**


* `xo` Block origin x 
* `yo` Block origin y 
* `dx` Block cell size 
* `blkwidth` Block width 
* `Poly` [**Polygon**](classPolygon.md) to test 



**Returns:**

True if block is inside or intersects polygon, false otherwise 





        

<hr>



### function wn\_PnPoly 

_winding number test for a point in a polygon_ 
```C++
template<class T>
int wn_PnPoly (
    T Px,
    T Py,
    Polygon Poly
) 
```



### Description



wn\_PnPoly(): winding number test for a point in a polygon Input: P = a point, V[] = vertex points of a polygon V[n+1] with V[n]=V[0] Return: wn = the winding number (=0 only when P is outside)



### Where does this come from:



Copyright 2000 softSurfer, 2012 Dan Sunday 


#### Original Licence



This code may be freely used and modified for any purpose providing that this copyright notice is included with it. SoftSurfer makes no warranty for this code, and cannot be held liable for any real or imagined damage resulting from its use. Users of this code must verify correctness for their application. Code modified to fit the use in DisperGPU 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Poly.h`


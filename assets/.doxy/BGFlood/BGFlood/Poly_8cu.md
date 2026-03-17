

# File Poly.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Poly.cu**](Poly_8cu.md)

[Go to the source code of this file](Poly_8cu_source.md)



* `#include "Poly.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  [**Polygon**](classPolygon.md) | [**CounterCWPoly**](#function-countercwpoly) ([**Polygon**](classPolygon.md) Poly) <br>_check polygon handedness and reverse if necessary._  |
|  bool | [**PolygonIntersect**](#function-polygonintersect) ([**Polygon**](classPolygon.md) P, [**Polygon**](classPolygon.md) Q) <br>_Intersection between two polygons._  |
|  bool | [**SegmentIntersect**](#function-segmentintersect) ([**Polygon**](classPolygon.md) P, [**Polygon**](classPolygon.md) Q) <br>_Intersection between segments._  |
|  [**Vertex**](classVertex.md) | [**VertAdd**](#function-vertadd) ([**Vertex**](classVertex.md) A, [**Vertex**](classVertex.md) B) <br>_Add two vertices._  |
|  [**Vertex**](classVertex.md) | [**VertSub**](#function-vertsub) ([**Vertex**](classVertex.md) A, [**Vertex**](classVertex.md) B) <br>_Subtract two vertices._  |
|  bool | [**blockinpoly**](#function-blockinpoly) (T xo, T yo, T dx, int blkwidth, [**Polygon**](classPolygon.md) Poly) <br>_Check whether a block is inside or intersects a polygon._  |
|  template bool | [**blockinpoly&lt; double &gt;**](#function-blockinpoly-double) (double xo, double yo, double dx, int blkwidth, [**Polygon**](classPolygon.md) Poly) <br> |
|  template bool | [**blockinpoly&lt; float &gt;**](#function-blockinpoly-float) (float xo, float yo, float dx, int blkwidth, [**Polygon**](classPolygon.md) Poly) <br> |
|  int | [**cn\_PnPoly**](#function-cn_pnpoly) (T Px, T Py, F \* Vx, F \* Vy, int n) <br>_Crossing number test for a point in a polygon._  |
|  double | [**dotprod**](#function-dotprod) ([**Vertex**](classVertex.md) A, [**Vertex**](classVertex.md) B) <br>_Compute dot product of two vertices._  |
|  T | [**isLeft**](#function-isleft) (T P0x, T P0y, T P1x, T P1y, T P2x, T P2y) <br>_Tests if a point is Left\|On\|Right of an infinite line._  |
|  bool | [**test\_SegmentIntersect**](#function-test_segmentintersect) () <br>_Test segment intersection function._  |
|  bool | [**test\_intersectpoly**](#function-test_intersectpoly) () <br>_Test polygon intersection function._  |
|  bool | [**test\_wninpoly**](#function-test_wninpoly) () <br>_Test winding number inpoly function._  |
|  int | [**wn\_PnPoly**](#function-wn_pnpoly) (T Px, T Py, T \* Vx, T \* Vy, unsigned int n) <br>_winding number test for a point in a polygon_  |
|  int | [**wn\_PnPoly**](#function-wn_pnpoly) (T Px, T Py, [**Polygon**](classPolygon.md) Poly) <br>_winding number test for a point in a polygon_  |
|  template int | [**wn\_PnPoly&lt; double &gt;**](#function-wn_pnpoly-double) (double Px, double Py, [**Polygon**](classPolygon.md) Poly) <br> |
|  template int | [**wn\_PnPoly&lt; float &gt;**](#function-wn_pnpoly-float) (float Px, float Py, [**Polygon**](classPolygon.md) Poly) <br> |
|  double | [**xprod**](#function-xprod) ([**Vertex**](classVertex.md) A, [**Vertex**](classVertex.md) B) <br>_Compute cross product of two vertices._  |




























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



### function PolygonIntersect 

_Intersection between two polygons._ 
```C++
bool PolygonIntersect (
    Polygon P,
    Polygon Q
) 
```



Checks whether two polygons intersect by testing all segment pairs. The function checks whether each segment of [**Polygon**](classPolygon.md) P intersect any segment of Poly Q. If an intersection is detected, returns true immediately.




**Parameters:**


* `P` First polygon 
* `Q` Second polygon 



**Returns:**

True if polygons intersect, false otherwise 





        

<hr>



### function SegmentIntersect 

_Intersection between segments._ 
```C++
bool SegmentIntersect (
    Polygon P,
    Polygon Q
) 
```



Checks whether two polygon segments intersect. [**Polygon**](classPolygon.md) P and Q are only 2 vertex long each. i.e. they represent a segment each.


### Where does this come from:



[https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect](https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect) Best answer from Gareth Rees




**Parameters:**


* `P` First segment as [**Polygon**](classPolygon.md) 
* `Q` Second segment as [**Polygon**](classPolygon.md) 



**Returns:**

True if segments intersect, false otherwise 






        

<hr>



### function VertAdd 

_Add two vertices._ 
```C++
Vertex VertAdd (
    Vertex A,
    Vertex B
) 
```



Returns the sum of two [**Vertex**](classVertex.md) objects.




**Parameters:**


* `A` First vertex 
* `B` Second vertex 



**Returns:**

Sum vertex 





        

<hr>



### function VertSub 

_Subtract two vertices._ 
```C++
Vertex VertSub (
    Vertex A,
    Vertex B
) 
```



Returns the difference of two [**Vertex**](classVertex.md) objects.




**Parameters:**


* `A` First vertex 
* `B` Second vertex 



**Returns:**

Difference vertex 





        

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



### function blockinpoly&lt; double &gt; 

```C++
template bool blockinpoly< double > (
    double xo,
    double yo,
    double dx,
    int blkwidth,
    Polygon Poly
) 
```




<hr>



### function blockinpoly&lt; float &gt; 

```C++
template bool blockinpoly< float > (
    float xo,
    float yo,
    float dx,
    int blkwidth,
    Polygon Poly
) 
```




<hr>



### function cn\_PnPoly 

_Crossing number test for a point in a polygon._ 
```C++
template<class T, class F>
int cn_PnPoly (
    T Px,
    T Py,
    F * Vx,
    F * Vy,
    int n
) 
```



Determines if a point is inside a polygon using the crossing number algorithm. cn\_PnPoly(): crossing number test for a point in a polygon Input: P = a point, V[] = vertex points of a polygon V[n+1] with V[n]=V[0] Return: 0 = outside, 1 = inside 


### Where does this come from:



Copyright 2000 softSurfer, 2012 Dan Sunday 


#### Original Licence



This code may be freely used and modified for any purpose providing that this copyright notice is included with it. SoftSurfer makes no warranty for this code, and cannot be held liable for any real or imagined damage resulting from its use. Users of this code must verify correctness for their application. Code modified to fit the use in DisperGPU


This code is patterned after [Franklin, 2000]




**Template parameters:**


* `T` Point coordinate type 
* `F` [**Vertex**](classVertex.md) coordinate type 



**Parameters:**


* `Px` X coordinate of point 
* `Py` Y coordinate of point 
* `Vx` Array of polygon vertex X coordinates 
* `Vy` Array of polygon vertex Y coordinates 
* `n` Number of vertices 



**Returns:**

1 if inside, 0 if outside 







        

<hr>



### function dotprod 

_Compute dot product of two vertices._ 
```C++
double dotprod (
    Vertex A,
    Vertex B
) 
```



Calculates the dot product of two [**Vertex**](classVertex.md) objects.




**Parameters:**


* `A` First vertex 
* `B` Second vertex 



**Returns:**

Dot product value 





        

<hr>



### function isLeft 

_Tests if a point is Left\|On\|Right of an infinite line._ 
```C++
template<class T>
T isLeft (
    T P0x,
    T P0y,
    T P1x,
    T P1y,
    T P2x,
    T P2y
) 
```



Returns &gt;0 for P2 left of the line through P0 and P1, =0 for P2 on the line, &lt;0 for P2 right of the line. See: Algorithm 1 "Area of Triangles and Polygons"


### Where does this come from:



Copyright 2000 softSurfer, 2012 Dan Sunday 


#### Original Licence



This code may be freely used and modified for any purpose providing that this copyright notice is included with it. SoftSurfer makes no warranty for this code, and cannot be held liable for any real or imagined damage resulting from its use. Users of this code must verify correctness for their application. Code modified to fit the use in DisperGPU




**Template parameters:**


* `T` Coordinate type 



**Parameters:**


* `P0x` X of first point 
* `P0y` Y of first point 
* `P1x` X of second point 
* `P1y` Y of second point 
* `P2x` X of test point 
* `P2y` Y of test point 



**Returns:**

Relative position value 







        

<hr>



### function test\_SegmentIntersect 

_Test segment intersection function._ 
```C++
bool test_SegmentIntersect () 
```



Tests the segment intersection function for known cases. 

**Returns:**

True if test passes, false otherwise 





        

<hr>



### function test\_intersectpoly 

_Test polygon intersection function._ 
```C++
bool test_intersectpoly () 
```



Tests the polygon intersection function for known cases. 

**Returns:**

True if test passes, false otherwise 





        

<hr>



### function test\_wninpoly 

_Test winding number inpoly function._ 
```C++
bool test_wninpoly () 
```



Tests the winding number function for a block polygon. 

**Returns:**

True if test passes, false otherwise 





        

<hr>



### function wn\_PnPoly 

_winding number test for a point in a polygon_ 
```C++
template<class T>
int wn_PnPoly (
    T Px,
    T Py,
    T * Vx,
    T * Vy,
    unsigned int n
) 
```



### Description



wn\_PnPoly(): winding number test for a point in a polygon Input: P = a point, V[] = vertex points of a polygon V[n+1] with V[n]=V[0] Return: wn = the winding number (=0 only when P is outside)



### Where does this come from:



Copyright 2000 softSurfer, 2012 Dan Sunday 


#### Original Licence



This code may be freely used and modified for any purpose providing that this copyright notice is included with it. SoftSurfer makes no warranty for this code, and cannot be held liable for any real or imagined damage resulting from its use. Users of this code must verify correctness for their application. Code modified to fit the use in DisperGPU 




        

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



### function wn\_PnPoly&lt; double &gt; 

```C++
template int wn_PnPoly< double > (
    double Px,
    double Py,
    Polygon Poly
) 
```




<hr>



### function wn\_PnPoly&lt; float &gt; 

```C++
template int wn_PnPoly< float > (
    float Px,
    float Py,
    Polygon Poly
) 
```




<hr>



### function xprod 

_Compute cross product of two vertices._ 
```C++
double xprod (
    Vertex A,
    Vertex B
) 
```



Calculates the cross product of two [**Vertex**](classVertex.md) objects.




**Parameters:**


* `A` First vertex 
* `B` Second vertex 



**Returns:**

Cross product value 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Poly.cu`


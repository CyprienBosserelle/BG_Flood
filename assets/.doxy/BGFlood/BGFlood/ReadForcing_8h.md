

# File ReadForcing.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**ReadForcing.h**](ReadForcing_8h.md)

[Go to the source code of this file](ReadForcing_8h_source.md)



* `#include "General.h"`
* `#include "Input.h"`
* `#include "Param.h"`
* `#include "Write_txtlog.h"`
* `#include "Read_netcdf.h"`
* `#include "Forcing.h"`
* `#include "Util_CPU.h"`
* `#include "Setup_GPU.h"`
* `#include "Poly.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  std::vector&lt; std::string &gt; | [**DelimLine**](#function-delimline) (std::string line, int n, char delim) <br> |
|  std::vector&lt; std::string &gt; | [**DelimLine**](#function-delimline) (std::string line, int n) <br> |
|  void | [**InitDynforcing**](#function-initdynforcing) (bool gpgpu, [**Param**](classParam.md) & XParam, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; & Dforcing) <br> |
|  void | [**InterpstepCPU**](#function-interpstepcpu) (int nx, int ny, int hdstep, float totaltime, float hddt, T \*& Ux, T \* Uo, T \* Un) <br> |
|  void | [**clampedges**](#function-clampedges) (int nx, int ny, T clamp, T \* z) <br> |
|  void | [**denan**](#function-denan) (int nx, int ny, float denanval, int \* z) <br> |
|  void | [**denan**](#function-denan) (int nx, int ny, float denanval, T \* z) <br> |
|  void | [**readDynforcing**](#function-readdynforcing) (bool gpgpu, double totaltime, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; & Dforcing) <br> |
|  std::vector&lt; [**Flowin**](classFlowin.md) &gt; | [**readFlowfile**](#function-readflowfile) (std::string Flowfilename, std::string & refdate) <br> |
|  std::vector&lt; [**Windin**](classWindin.md) &gt; | [**readINfileUNI**](#function-readinfileuni) (std::string filename, std::string & refdate) <br> |
|  std::vector&lt; [**SLTS**](classSLTS.md) &gt; | [**readNestfile**](#function-readnestfile) (std::string ncfile, std::string varname, int hor, double eps, double bndxo, double bndxmax, double bndy) <br> |
|  [**Polygon**](classPolygon.md) | [**readPolygon**](#function-readpolygon) (std::string filename) <br> |
|  std::vector&lt; [**SLTS**](classSLTS.md) &gt; | [**readWLfile**](#function-readwlfile) (std::string WLfilename, std::string & refdate) <br> |
|  std::vector&lt; [**Windin**](classWindin.md) &gt; | [**readWNDfileUNI**](#function-readwndfileuni) (std::string filename, std::string & refdate, double grdalpha) <br> |
|  void | [**readXBbathy**](#function-readxbbathy) (std::string filename, int nx, int ny, T \*& zb) <br> |
|  void | [**readbathyASCHead**](#function-readbathyaschead) (std::string filename, int & nx, int & ny, double & dx, double & xo, double & yo, double & grdalpha) <br> |
|  void | [**readbathyASCzb**](#function-readbathyasczb) (std::string filename, int nx, int ny, T \*& zb) <br> |
|  void | [**readbathyHeadMD**](#function-readbathyheadmd) (std::string filename, int & nx, int & ny, double & dx, double & grdalpha) <br> |
|  void | [**readbathyMD**](#function-readbathymd) (std::string filename, T \*& zb) <br> |
|  std::vector&lt; [**SLTS**](classSLTS.md) &gt; | [**readbndfile**](#function-readbndfile) (std::string filename, [**Param**](classParam.md) & XParam) <br> |
|  [**Polygon**](classPolygon.md) | [**readbndpolysegment**](#function-readbndpolysegment) ([**bndsegment**](classbndsegment.md) bnd, [**Param**](classParam.md) XParam) <br> |
|  void | [**readforcing**](#function-readforcing) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; T &gt; & XForcing) <br> |
|  void | [**readforcingdata**](#function-readforcingdata) (int step, T forcing) <br> |
|  void | [**readforcingdata**](#function-readforcingdata) (double totaltime, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; & forcing) <br> |
|  [**DynForcingP**](structDynForcingP.md)&lt; float &gt; | [**readforcinghead**](#function-readforcinghead) ([**DynForcingP**](structDynForcingP.md)&lt; float &gt; Fmap, [**Param**](classParam.md) XParam) <br> |
|  T | [**readforcinghead**](#function-readforcinghead) (T Fmap) <br> |
|  void | [**readstaticforcing**](#function-readstaticforcing) (T & Sforcing) <br> |
|  void | [**readstaticforcing**](#function-readstaticforcing) (int step, T & Sforcing) <br> |




























## Public Functions Documentation




### function DelimLine 

```C++
std::vector< std::string > DelimLine (
    std::string line,
    int n,
    char delim
) 
```




<hr>



### function DelimLine 

```C++
std::vector< std::string > DelimLine (
    std::string line,
    int n
) 
```




<hr>



### function InitDynforcing 

```C++
void InitDynforcing (
    bool gpgpu,
    Param & XParam,
    DynForcingP < float > & Dforcing
) 
```




<hr>



### function InterpstepCPU 

```C++
template<class T>
void InterpstepCPU (
    int nx,
    int ny,
    int hdstep,
    float totaltime,
    float hddt,
    T *& Ux,
    T * Uo,
    T * Un
) 
```




<hr>



### function clampedges 

```C++
template<class T>
void clampedges (
    int nx,
    int ny,
    T clamp,
    T * z
) 
```




<hr>



### function denan 

```C++
void denan (
    int nx,
    int ny,
    float denanval,
    int * z
) 
```




<hr>



### function denan 

```C++
template<class T>
void denan (
    int nx,
    int ny,
    float denanval,
    T * z
) 
```




<hr>



### function readDynforcing 

```C++
void readDynforcing (
    bool gpgpu,
    double totaltime,
    DynForcingP < float > & Dforcing
) 
```



This is a deprecated function! See InitDynforcing() instead 


        

<hr>



### function readFlowfile 

```C++
std::vector< Flowin > readFlowfile (
    std::string Flowfilename,
    std::string & refdate
) 
```




<hr>



### function readINfileUNI 

```C++
std::vector< Windin > readINfileUNI (
    std::string filename,
    std::string & refdate
) 
```




<hr>



### function readNestfile 

```C++
std::vector< SLTS > readNestfile (
    std::string ncfile,
    std::string varname,
    int hor,
    double eps,
    double bndxo,
    double bndxmax,
    double bndy
) 
```



Read boundary Nesting data 


        

<hr>



### function readPolygon 

```C++
Polygon readPolygon (
    std::string filename
) 
```




<hr>



### function readWLfile 

```C++
std::vector< SLTS > readWLfile (
    std::string WLfilename,
    std::string & refdate
) 
```




<hr>



### function readWNDfileUNI 

```C++
std::vector< Windin > readWNDfileUNI (
    std::string filename,
    std::string & refdate,
    double grdalpha
) 
```




<hr>



### function readXBbathy 

```C++
template<class T>
void readXBbathy (
    std::string filename,
    int nx,
    int ny,
    T *& zb
) 
```




<hr>



### function readbathyASCHead 

```C++
void readbathyASCHead (
    std::string filename,
    int & nx,
    int & ny,
    double & dx,
    double & xo,
    double & yo,
    double & grdalpha
) 
```



Read ASC file meta/header data 


        

<hr>



### function readbathyASCzb 

```C++
template<class T>
void readbathyASCzb (
    std::string filename,
    int nx,
    int ny,
    T *& zb
) 
```




<hr>



### function readbathyHeadMD 

```C++
void readbathyHeadMD (
    std::string filename,
    int & nx,
    int & ny,
    double & dx,
    double & grdalpha
) 
```



Read MD file header data 


        

<hr>



### function readbathyMD 

```C++
template<class T>
void readbathyMD (
    std::string filename,
    T *& zb
) 
```




<hr>



### function readbndfile 

```C++
std::vector< SLTS > readbndfile (
    std::string filename,
    Param & XParam
) 
```




<hr>



### function readbndpolysegment 

```C++
Polygon readbndpolysegment (
    bndsegment bnd,
    Param XParam
) 
```




<hr>



### function readforcing 

```C++
template<class T>
void readforcing (
    Param & XParam,
    Forcing < T > & XForcing
) 
```



wrapping function for reading all the forcing data 


        

<hr>



### function readforcingdata 

```C++
template<class T>
void readforcingdata (
    int step,
    T forcing
) 
```



Read static forcing data 


        

<hr>



### function readforcingdata 

```C++
void readforcingdata (
    double totaltime,
    DynForcingP < float > & forcing
) 
```



Read Dynamic forcing data 


        

<hr>



### function readforcinghead 

```C++
DynForcingP < float > readforcinghead (
    DynForcingP < float > Fmap,
    Param XParam
) 
```



Read Dynamic forcing meta/header data 


        

<hr>



### function readforcinghead 

```C++
template<class T>
T readforcinghead (
    T Fmap
) 
```



Read Static forcing meta/header data 


        

<hr>



### function readstaticforcing 

```C++
template<class T>
void readstaticforcing (
    T & Sforcing
) 
```



single parameter version of readstaticforcing(int step,T& Sforcing) readstaticforcing(0, Sforcing); 


        

<hr>



### function readstaticforcing 

```C++
template<class T>
void readstaticforcing (
    int step,
    T & Sforcing
) 
```



Allocate and read static (i.e. not varying in time) forcing Used for Bathy, roughness, deformation 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/ReadForcing.h`


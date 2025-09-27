

# File ReadForcing.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**ReadForcing.cu**](ReadForcing_8cu.md)

[Go to the source code of this file](ReadForcing_8cu_source.md)



* `#include "ReadForcing.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  std::vector&lt; std::string &gt; | [**DelimLine**](#function-delimline) (std::string line, int n, char delim) <br> |
|  std::vector&lt; std::string &gt; | [**DelimLine**](#function-delimline) (std::string line, int n) <br> |
|  void | [**InitDynforcing**](#function-initdynforcing) (bool gpgpu, [**Param**](classParam.md) & XParam, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; & Dforcing) <br> |
|  void | [**clampedges**](#function-clampedges) (int nx, int ny, T clamp, T \* z) <br> |
|  void | [**denan**](#function-denan) (int nx, int ny, float denanval, T \* z) <br> |
|  void | [**denan**](#function-denan) (int nx, int ny, float denanval, int \* z) <br> |
|  template void | [**denan&lt; double &gt;**](#function-denan-double) (int nx, int ny, float denanval, double \* z) <br> |
|  template void | [**denan&lt; float &gt;**](#function-denan-float) (int nx, int ny, float denanval, float \* z) <br> |
|  std::string | [**readCRSfrombathy**](#function-readcrsfrombathy) (std::string crs\_ref, [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; & Sforcing) <br> |
|  void | [**readDynforcing**](#function-readdynforcing) (bool gpgpu, double totaltime, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; & Dforcing) <br> |
|  std::vector&lt; [**Flowin**](classFlowin.md) &gt; | [**readFlowfile**](#function-readflowfile) (std::string Flowfilename, std::string & refdate) <br> |
|  std::vector&lt; [**Windin**](classWindin.md) &gt; | [**readINfileUNI**](#function-readinfileuni) (std::string filename, std::string & refdate) <br> |
|  std::vector&lt; [**SLTS**](classSLTS.md) &gt; | [**readNestfile**](#function-readnestfile) (std::string ncfile, std::string varname, int hor, double eps, double bndxo, double bndxmax, double bndy) <br> |
|  [**Polygon**](classPolygon.md) | [**readPolygon**](#function-readpolygon) (std::string filename) <br> |
|  std::vector&lt; [**SLTS**](classSLTS.md) &gt; | [**readWLfile**](#function-readwlfile) (std::string WLfilename, std::string & refdate) <br> |
|  std::vector&lt; [**Windin**](classWindin.md) &gt; | [**readWNDfileUNI**](#function-readwndfileuni) (std::string filename, std::string & refdate, double grdalpha) <br> |
|  void | [**readXBbathy**](#function-readxbbathy) (std::string filename, int nx, int ny, T \*& zb) <br> |
|  template void | [**readXBbathy&lt; float &gt;**](#function-readxbbathy-float) (std::string filename, int nx, int ny, float \*& zb) <br> |
|  template void | [**readXBbathy&lt; int &gt;**](#function-readxbbathy-int) (std::string filename, int nx, int ny, int \*& zb) <br> |
|  void | [**readbathyASCHead**](#function-readbathyaschead) (std::string filename, int & nx, int & ny, double & dx, double & xo, double & yo, double & grdalpha) <br> |
|  void | [**readbathyASCzb**](#function-readbathyasczb) (std::string filename, int nx, int ny, T \*& zb) <br> |
|  template void | [**readbathyASCzb&lt; float &gt;**](#function-readbathyasczb-float) (std::string filename, int nx, int ny, float \*& zb) <br> |
|  template void | [**readbathyASCzb&lt; int &gt;**](#function-readbathyasczb-int) (std::string filename, int nx, int ny, int \*& zb) <br> |
|  void | [**readbathyHeadMD**](#function-readbathyheadmd) (std::string filename, int & nx, int & ny, double & dx, double & grdalpha) <br> |
|  void | [**readbathyMD**](#function-readbathymd) (std::string filename, T \*& zb) <br> |
|  template void | [**readbathyMD&lt; float &gt;**](#function-readbathymd-float) (std::string filename, float \*& zb) <br> |
|  template void | [**readbathyMD&lt; int &gt;**](#function-readbathymd-int) (std::string filename, int \*& zb) <br> |
|  void | [**readbathydata**](#function-readbathydata) (int posdown, [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; & Sforcing) <br> |
|  std::vector&lt; [**SLTS**](classSLTS.md) &gt; | [**readbndfile**](#function-readbndfile) (std::string filename, [**Param**](classParam.md) & XParam) <br> |
|  [**Polygon**](classPolygon.md) | [**readbndpolysegment**](#function-readbndpolysegment) ([**bndsegment**](classbndsegment.md) bnd, [**Param**](classParam.md) XParam) <br> |
|  void | [**readforcing**](#function-readforcing) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; T &gt; & XForcing) <br> |
|  template void | [**readforcing&lt; float &gt;**](#function-readforcing-float) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing) <br> |
|  void | [**readforcingdata**](#function-readforcingdata) (int step, T forcing) <br> |
|  void | [**readforcingdata**](#function-readforcingdata) (double totaltime, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; & forcing) <br> |
|  template void | [**readforcingdata&lt; StaticForcingP&lt; float &gt; &gt;**](#function-readforcingdata-staticforcingp-float) (int step, [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; forcing) <br> |
|  template void | [**readforcingdata&lt; StaticForcingP&lt; int &gt; &gt;**](#function-readforcingdata-staticforcingp-int) (int step, [**StaticForcingP**](structStaticForcingP.md)&lt; int &gt; forcing) <br> |
|  template void | [**readforcingdata&lt; deformmap&lt; float &gt; &gt;**](#function-readforcingdata-deformmap-float) (int step, [**deformmap**](classdeformmap.md)&lt; float &gt; forcing) <br> |
|  [**DynForcingP**](structDynForcingP.md)&lt; float &gt; | [**readforcinghead**](#function-readforcinghead) ([**DynForcingP**](structDynForcingP.md)&lt; float &gt; Fmap, [**Param**](classParam.md) XParam) <br> |
|  T | [**readforcinghead**](#function-readforcinghead) (T ForcingParam) <br> |
|  template [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; | [**readforcinghead&lt; StaticForcingP&lt; float &gt; &gt;**](#function-readforcinghead-staticforcingp-float) ([**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; ForcingParam) <br> |
|  template [**forcingmap**](classforcingmap.md) | [**readforcinghead&lt; forcingmap &gt;**](#function-readforcinghead-forcingmap) ([**forcingmap**](classforcingmap.md) BathyParam) <br> |
|  template [**inputmap**](classinputmap.md) | [**readforcinghead&lt; inputmap &gt;**](#function-readforcinghead-inputmap) ([**inputmap**](classinputmap.md) BathyParam) <br> |
|  void | [**readstaticforcing**](#function-readstaticforcing) (T & Sforcing) <br> |
|  void | [**readstaticforcing**](#function-readstaticforcing) (int step, T & Sforcing) <br> |
|  template void | [**readstaticforcing&lt; StaticForcingP&lt; float &gt; &gt;**](#function-readstaticforcing-staticforcingp-float) ([**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; & Sforcing) <br> |
|  template void | [**readstaticforcing&lt; StaticForcingP&lt; float &gt; &gt;**](#function-readstaticforcing-staticforcingp-float) (int step, [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; & Sforcing) <br> |
|  template void | [**readstaticforcing&lt; StaticForcingP&lt; int &gt; &gt;**](#function-readstaticforcing-staticforcingp-int) ([**StaticForcingP**](structStaticForcingP.md)&lt; int &gt; & Sforcing) <br> |
|  template void | [**readstaticforcing&lt; StaticForcingP&lt; int &gt; &gt;**](#function-readstaticforcing-staticforcingp-int) (int step, [**StaticForcingP**](structStaticForcingP.md)&lt; int &gt; & Sforcing) <br> |
|  template void | [**readstaticforcing&lt; deformmap&lt; float &gt; &gt;**](#function-readstaticforcing-deformmap-float) ([**deformmap**](classdeformmap.md)&lt; float &gt; & Sforcing) <br> |
|  template void | [**readstaticforcing&lt; deformmap&lt; float &gt; &gt;**](#function-readstaticforcing-deformmap-float) (int step, [**deformmap**](classdeformmap.md)&lt; float &gt; & Sforcing) <br> |




























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
template<class T>
void denan (
    int nx,
    int ny,
    float denanval,
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



### function denan&lt; double &gt; 

```C++
template void denan< double > (
    int nx,
    int ny,
    float denanval,
    double * z
) 
```




<hr>



### function denan&lt; float &gt; 

```C++
template void denan< float > (
    int nx,
    int ny,
    float denanval,
    float * z
) 
```




<hr>



### function readCRSfrombathy 

```C++
std::string readCRSfrombathy (
    std::string crs_ref,
    StaticForcingP < float > & Sforcing
) 
```



Reading the CRS information from the bathymetry file (last one read); 


        

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



### function readXBbathy&lt; float &gt; 

```C++
template void readXBbathy< float > (
    std::string filename,
    int nx,
    int ny,
    float *& zb
) 
```




<hr>



### function readXBbathy&lt; int &gt; 

```C++
template void readXBbathy< int > (
    std::string filename,
    int nx,
    int ny,
    int *& zb
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



### function readbathyASCzb&lt; float &gt; 

```C++
template void readbathyASCzb< float > (
    std::string filename,
    int nx,
    int ny,
    float *& zb
) 
```




<hr>



### function readbathyASCzb&lt; int &gt; 

```C++
template void readbathyASCzb< int > (
    std::string filename,
    int nx,
    int ny,
    int *& zb
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



### function readbathyMD&lt; float &gt; 

```C++
template void readbathyMD< float > (
    std::string filename,
    float *& zb
) 
```




<hr>



### function readbathyMD&lt; int &gt; 

```C++
template void readbathyMD< int > (
    std::string filename,
    int *& zb
) 
```




<hr>



### function readbathydata 

```C++
void readbathydata (
    int posdown,
    StaticForcingP < float > & Sforcing
) 
```



special case of readstaticforcing(Sforcing); where the data 


        

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



### function readforcing&lt; float &gt; 

```C++
template void readforcing< float > (
    Param & XParam,
    Forcing < float > & XForcing
) 
```




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



### function readforcingdata&lt; StaticForcingP&lt; float &gt; &gt; 

```C++
template void readforcingdata< StaticForcingP< float > > (
    int step,
    StaticForcingP < float > forcing
) 
```




<hr>



### function readforcingdata&lt; StaticForcingP&lt; int &gt; &gt; 

```C++
template void readforcingdata< StaticForcingP< int > > (
    int step,
    StaticForcingP < int > forcing
) 
```




<hr>



### function readforcingdata&lt; deformmap&lt; float &gt; &gt; 

```C++
template void readforcingdata< deformmap< float > > (
    int step,
    deformmap < float > forcing
) 
```




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
    T ForcingParam
) 
```



Read Static forcing meta/header data 


        

<hr>



### function readforcinghead&lt; StaticForcingP&lt; float &gt; &gt; 

```C++
template StaticForcingP < float > readforcinghead< StaticForcingP< float > > (
    StaticForcingP < float > ForcingParam
) 
```




<hr>



### function readforcinghead&lt; forcingmap &gt; 

```C++
template forcingmap readforcinghead< forcingmap > (
    forcingmap BathyParam
) 
```




<hr>



### function readforcinghead&lt; inputmap &gt; 

```C++
template inputmap readforcinghead< inputmap > (
    inputmap BathyParam
) 
```




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



### function readstaticforcing&lt; StaticForcingP&lt; float &gt; &gt; 

```C++
template void readstaticforcing< StaticForcingP< float > > (
    StaticForcingP < float > & Sforcing
) 
```




<hr>



### function readstaticforcing&lt; StaticForcingP&lt; float &gt; &gt; 

```C++
template void readstaticforcing< StaticForcingP< float > > (
    int step,
    StaticForcingP < float > & Sforcing
) 
```




<hr>



### function readstaticforcing&lt; StaticForcingP&lt; int &gt; &gt; 

```C++
template void readstaticforcing< StaticForcingP< int > > (
    StaticForcingP < int > & Sforcing
) 
```




<hr>



### function readstaticforcing&lt; StaticForcingP&lt; int &gt; &gt; 

```C++
template void readstaticforcing< StaticForcingP< int > > (
    int step,
    StaticForcingP < int > & Sforcing
) 
```




<hr>



### function readstaticforcing&lt; deformmap&lt; float &gt; &gt; 

```C++
template void readstaticforcing< deformmap< float > > (
    deformmap < float > & Sforcing
) 
```




<hr>



### function readstaticforcing&lt; deformmap&lt; float &gt; &gt; 

```C++
template void readstaticforcing< deformmap< float > > (
    int step,
    deformmap < float > & Sforcing
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/ReadForcing.cu`




# File Write\_netcdf.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Write\_netcdf.cu**](Write__netcdf_8cu.md)

[Go to the source code of this file](Write__netcdf_8cu_source.md)



* `#include "Write_netcdf.h"`
* `#include "Util_CPU.h"`
* `#include "General.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  std::vector&lt; int &gt; | [**Calcactiveblockzone**](#function-calcactiveblockzone) ([**Param**](classParam.md) XParam, int \* activeblk, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  void | [**Calcnxny**](#function-calcnxny) ([**Param**](classParam.md) XParam, int level, int & nx, int & ny) <br> |
|  void | [**Calcnxnyzone**](#function-calcnxnyzone) ([**Param**](classParam.md) XParam, int level, int & nx, int & ny, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  void | [**InitSave2Netcdf**](#function-initsave2netcdf) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  template void | [**InitSave2Netcdf&lt; double &gt;**](#function-initsave2netcdf-double) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template void | [**InitSave2Netcdf&lt; float &gt;**](#function-initsave2netcdf-float) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; float &gt; & XModel) <br> |
|  void | [**Save2Netcdf**](#function-save2netcdf) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  template void | [**Save2Netcdf&lt; double &gt;**](#function-save2netcdf-double) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; double &gt; XLoop, [**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template void | [**Save2Netcdf&lt; float &gt;**](#function-save2netcdf-float) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; float &gt; XLoop, [**Model**](structModel.md)&lt; float &gt; & XModel) <br> |
|  void | [**create2dnc**](#function-create2dnc) (char \* filename, int nx, int ny, double \* xx, double \* yy, double \* var, char \* varname) <br> |
|  void | [**create3dnc**](#function-create3dnc) (char \* name, int nx, int ny, int nt, double \* xx, double \* yy, double \* theta, double \* var, char \* varname) <br> |
|  void | [**creatncfileBUQ**](#function-creatncfilebuq) ([**Param**](classParam.md) & XParam, int \* activeblk, int \* level, T \* blockxo, T \* blockyo, [**outzoneB**](structoutzoneB.md) & Xzone) <br> |
|  void | [**creatncfileBUQ**](#function-creatncfilebuq) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  template void | [**creatncfileBUQ&lt; double &gt;**](#function-creatncfilebuq-double) ([**Param**](classParam.md) & XParam, int \* activeblk, int \* level, double \* blockxo, double \* blockyo, [**outzoneB**](structoutzoneB.md) & Xzone) <br> |
|  template void | [**creatncfileBUQ&lt; double &gt;**](#function-creatncfilebuq-double) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; double &gt; & XBlock) <br> |
|  template void | [**creatncfileBUQ&lt; float &gt;**](#function-creatncfilebuq-float) ([**Param**](classParam.md) & XParam, int \* activeblk, int \* level, float \* blockxo, float \* blockyo, [**outzoneB**](structoutzoneB.md) & Xzone) <br> |
|  template void | [**creatncfileBUQ&lt; float &gt;**](#function-creatncfilebuq-float) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; float &gt; & XBlock) <br> |
|  void | [**defncvarBUQ**](#function-defncvarbuq) ([**Param**](classParam.md) XParam, int \* activeblk, int \* level, T \* blockxo, T \* blockyo, std::string varst, int vdim, T \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  void | [**defncvarBUQ**](#function-defncvarbuq) ([**Param**](classParam.md) XParam, int \* activeblk, int \* level, T \* blockxo, T \* blockyo, std::string varst, std::string longname, std::string stdname, std::string unit, int vdim, T \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  template void | [**defncvarBUQ&lt; double &gt;**](#function-defncvarbuq-double) ([**Param**](classParam.md) XParam, int \* activeblk, int \* level, double \* blockxo, double \* blockyo, std::string varst, int vdim, double \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  template void | [**defncvarBUQ&lt; float &gt;**](#function-defncvarbuq-float) ([**Param**](classParam.md) XParam, int \* activeblk, int \* level, float \* blockxo, float \* blockyo, std::string varst, int vdim, float \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  void | [**defncvarBUQlev**](#function-defncvarbuqlev) ([**Param**](classParam.md) XParam, int \* activeblk, int \* level, T \* blockxo, T \* blockyo, std::string varst, std::string longname, std::string stdname, std::string unit, int vdim, T \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  void | [**handle\_ncerror**](#function-handle_ncerror) (int status) <br> |
|  void | [**write2dvarnc**](#function-write2dvarnc) (int nx, int ny, double totaltime, double \* var) <br> |
|  void | [**write3dvarnc**](#function-write3dvarnc) (int nx, int ny, int nt, double totaltime, double \* var) <br> |
|  void | [**writenctimestep**](#function-writenctimestep) (std::string outfile, double totaltime) <br> |
|  void | [**writencvarstepBUQ**](#function-writencvarstepbuq) ([**Param**](classParam.md) XParam, int vdim, int \* activeblk, int \* level, T \* blockxo, T \* blockyo, std::string varst, T \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  template void | [**writencvarstepBUQ&lt; double &gt;**](#function-writencvarstepbuq-double) ([**Param**](classParam.md) XParam, int vdim, int \* activeblk, int \* level, double \* blockxo, double \* blockyo, std::string varst, double \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  template void | [**writencvarstepBUQ&lt; float &gt;**](#function-writencvarstepbuq-float) ([**Param**](classParam.md) XParam, int vdim, int \* activeblk, int \* level, float \* blockxo, float \* blockyo, std::string varst, float \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  void | [**writencvarstepBUQlev**](#function-writencvarstepbuqlev) ([**Param**](classParam.md) XParam, int vdim, int \* activeblk, int \* level, T \* blockxo, T \* blockyo, std::string varst, T \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  template void | [**writencvarstepBUQlev&lt; double &gt;**](#function-writencvarstepbuqlev-double) ([**Param**](classParam.md) XParam, int vdim, int \* activeblk, int \* level, double \* blockxo, double \* blockyo, std::string varst, double \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  template void | [**writencvarstepBUQlev&lt; float &gt;**](#function-writencvarstepbuqlev-float) ([**Param**](classParam.md) XParam, int vdim, int \* activeblk, int \* level, float \* blockxo, float \* blockyo, std::string varst, float \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |




























## Public Functions Documentation




### function Calcactiveblockzone 

```C++
std::vector< int > Calcactiveblockzone (
    Param XParam,
    int * activeblk,
    outzoneB Xzone
) 
```




<hr>



### function Calcnxny 

```C++
void Calcnxny (
    Param XParam,
    int level,
    int & nx,
    int & ny
) 
```




<hr>



### function Calcnxnyzone 

```C++
void Calcnxnyzone (
    Param XParam,
    int level,
    int & nx,
    int & ny,
    outzoneB Xzone
) 
```




<hr>



### function InitSave2Netcdf 

```C++
template<class T>
void InitSave2Netcdf (
    Param & XParam,
    Model < T > & XModel
) 
```




<hr>



### function InitSave2Netcdf&lt; double &gt; 

```C++
template void InitSave2Netcdf< double > (
    Param & XParam,
    Model < double > & XModel
) 
```




<hr>



### function InitSave2Netcdf&lt; float &gt; 

```C++
template void InitSave2Netcdf< float > (
    Param & XParam,
    Model < float > & XModel
) 
```




<hr>



### function Save2Netcdf 

```C++
template<class T>
void Save2Netcdf (
    Param XParam,
    Loop < T > XLoop,
    Model < T > & XModel
) 
```




<hr>



### function Save2Netcdf&lt; double &gt; 

```C++
template void Save2Netcdf< double > (
    Param XParam,
    Loop < double > XLoop,
    Model < double > & XModel
) 
```




<hr>



### function Save2Netcdf&lt; float &gt; 

```C++
template void Save2Netcdf< float > (
    Param XParam,
    Loop < float > XLoop,
    Model < float > & XModel
) 
```




<hr>



### function create2dnc 

```C++
void create2dnc (
    char * filename,
    int nx,
    int ny,
    double * xx,
    double * yy,
    double * var,
    char * varname
) 
```




<hr>



### function create3dnc 

```C++
void create3dnc (
    char * name,
    int nx,
    int ny,
    int nt,
    double * xx,
    double * yy,
    double * theta,
    double * var,
    char * varname
) 
```




<hr>



### function creatncfileBUQ 

```C++
template<class T>
void creatncfileBUQ (
    Param & XParam,
    int * activeblk,
    int * level,
    T * blockxo,
    T * blockyo,
    outzoneB & Xzone
) 
```




<hr>



### function creatncfileBUQ 

```C++
template<class T>
void creatncfileBUQ (
    Param & XParam,
    BlockP < T > & XBlock
) 
```




<hr>



### function creatncfileBUQ&lt; double &gt; 

```C++
template void creatncfileBUQ< double > (
    Param & XParam,
    int * activeblk,
    int * level,
    double * blockxo,
    double * blockyo,
    outzoneB & Xzone
) 
```




<hr>



### function creatncfileBUQ&lt; double &gt; 

```C++
template void creatncfileBUQ< double > (
    Param & XParam,
    BlockP < double > & XBlock
) 
```




<hr>



### function creatncfileBUQ&lt; float &gt; 

```C++
template void creatncfileBUQ< float > (
    Param & XParam,
    int * activeblk,
    int * level,
    float * blockxo,
    float * blockyo,
    outzoneB & Xzone
) 
```




<hr>



### function creatncfileBUQ&lt; float &gt; 

```C++
template void creatncfileBUQ< float > (
    Param & XParam,
    BlockP < float > & XBlock
) 
```




<hr>



### function defncvarBUQ 

```C++
template<class T>
void defncvarBUQ (
    Param XParam,
    int * activeblk,
    int * level,
    T * blockxo,
    T * blockyo,
    std::string varst,
    int vdim,
    T * var,
    outzoneB Xzone
) 
```




<hr>



### function defncvarBUQ 

```C++
template<class T>
void defncvarBUQ (
    Param XParam,
    int * activeblk,
    int * level,
    T * blockxo,
    T * blockyo,
    std::string varst,
    std::string longname,
    std::string stdname,
    std::string unit,
    int vdim,
    T * var,
    outzoneB Xzone
) 
```




<hr>



### function defncvarBUQ&lt; double &gt; 

```C++
template void defncvarBUQ< double > (
    Param XParam,
    int * activeblk,
    int * level,
    double * blockxo,
    double * blockyo,
    std::string varst,
    int vdim,
    double * var,
    outzoneB Xzone
) 
```




<hr>



### function defncvarBUQ&lt; float &gt; 

```C++
template void defncvarBUQ< float > (
    Param XParam,
    int * activeblk,
    int * level,
    float * blockxo,
    float * blockyo,
    std::string varst,
    int vdim,
    float * var,
    outzoneB Xzone
) 
```




<hr>



### function defncvarBUQlev 

```C++
template<class T>
void defncvarBUQlev (
    Param XParam,
    int * activeblk,
    int * level,
    T * blockxo,
    T * blockyo,
    std::string varst,
    std::string longname,
    std::string stdname,
    std::string unit,
    int vdim,
    T * var,
    outzoneB Xzone
) 
```




<hr>



### function handle\_ncerror 

```C++
void handle_ncerror (
    int status
) 
```




<hr>



### function write2dvarnc 

```C++
void write2dvarnc (
    int nx,
    int ny,
    double totaltime,
    double * var
) 
```




<hr>



### function write3dvarnc 

```C++
void write3dvarnc (
    int nx,
    int ny,
    int nt,
    double totaltime,
    double * var
) 
```




<hr>



### function writenctimestep 

```C++
void writenctimestep (
    std::string outfile,
    double totaltime
) 
```




<hr>



### function writencvarstepBUQ 

```C++
template<class T>
void writencvarstepBUQ (
    Param XParam,
    int vdim,
    int * activeblk,
    int * level,
    T * blockxo,
    T * blockyo,
    std::string varst,
    T * var,
    outzoneB Xzone
) 
```




<hr>



### function writencvarstepBUQ&lt; double &gt; 

```C++
template void writencvarstepBUQ< double > (
    Param XParam,
    int vdim,
    int * activeblk,
    int * level,
    double * blockxo,
    double * blockyo,
    std::string varst,
    double * var,
    outzoneB Xzone
) 
```




<hr>



### function writencvarstepBUQ&lt; float &gt; 

```C++
template void writencvarstepBUQ< float > (
    Param XParam,
    int vdim,
    int * activeblk,
    int * level,
    float * blockxo,
    float * blockyo,
    std::string varst,
    float * var,
    outzoneB Xzone
) 
```




<hr>



### function writencvarstepBUQlev 

```C++
template<class T>
void writencvarstepBUQlev (
    Param XParam,
    int vdim,
    int * activeblk,
    int * level,
    T * blockxo,
    T * blockyo,
    std::string varst,
    T * var,
    outzoneB Xzone
) 
```




<hr>



### function writencvarstepBUQlev&lt; double &gt; 

```C++
template void writencvarstepBUQlev< double > (
    Param XParam,
    int vdim,
    int * activeblk,
    int * level,
    double * blockxo,
    double * blockyo,
    std::string varst,
    double * var,
    outzoneB Xzone
) 
```




<hr>



### function writencvarstepBUQlev&lt; float &gt; 

```C++
template void writencvarstepBUQlev< float > (
    Param XParam,
    int vdim,
    int * activeblk,
    int * level,
    float * blockxo,
    float * blockyo,
    std::string varst,
    float * var,
    outzoneB Xzone
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Write_netcdf.cu`


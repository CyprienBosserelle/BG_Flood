

# File Write\_netcdf.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Write\_netcdf.h**](Write__netcdf_8h.md)

[Go to the source code of this file](Write__netcdf_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Write_txtlog.h"`
* `#include "ReadInput.h"`
* `#include "MemManagement.h"`
* `#include "Util_CPU.h"`
* `#include "Arrays.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**InitSave2Netcdf**](#function-initsave2netcdf) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  void | [**Save2Netcdf**](#function-save2netcdf) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  void | [**create2dnc**](#function-create2dnc) (char \* filename, int nx, int ny, double \* xx, double \* yy, double \* var, char \* varname) <br> |
|  void | [**create3dnc**](#function-create3dnc) (char \* name, int nx, int ny, int nt, double \* xx, double \* yy, double \* theta, double \* var, char \* varname) <br> |
|  void | [**creatncfileBUQ**](#function-creatncfilebuq) ([**Param**](classParam.md) & XParam, int \* activeblk, int \* level, T \* blockxo, T \* blockyo, [**outzoneB**](structoutzoneB.md) & Xzone) <br> |
|  void | [**creatncfileBUQ**](#function-creatncfilebuq) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  void | [**defncvarBUQ**](#function-defncvarbuq) ([**Param**](classParam.md) XParam, int \* activeblk, int \* level, T \* blockxo, T \* blockyo, std::string varst, int vdim, T \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |
|  void | [**handle\_ncerror**](#function-handle_ncerror) (int status) <br> |
|  void | [**write2dvarnc**](#function-write2dvarnc) (int nx, int ny, double totaltime, double \* var) <br> |
|  void | [**write3dvarnc**](#function-write3dvarnc) (int nx, int ny, int nt, double totaltime, double \* var) <br> |
|  void | [**writenctimestep**](#function-writenctimestep) (std::string outfile, double totaltime) <br> |
|  void | [**writencvarstepBUQ**](#function-writencvarstepbuq) ([**Param**](classParam.md) XParam, int vdim, int \* activeblk, int \* level, T \* blockxo, T \* blockyo, std::string varst, T \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br> |




























## Public Functions Documentation




### function InitSave2Netcdf 

```C++
template<class T>
void InitSave2Netcdf (
    Param & XParam,
    Model < T > & XModel
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

------------------------------
The documentation for this class was generated from the following file `src/Write_netcdf.h`


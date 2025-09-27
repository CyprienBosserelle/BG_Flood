

# File Read\_netcdf.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Read\_netcdf.h**](Read__netcdf_8h.md)

[Go to the source code of this file](Read__netcdf_8h_source.md)



* `#include "General.h"`
* `#include "Input.h"`
* `#include "ReadInput.h"`
* `#include "Write_txtlog.h"`
* `#include "Write_netcdf.h"`
* `#include "Util_CPU.h"`
* `#include "GridManip.h"`
* `#include "Forcing.h"`
* `#include "utctime.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  std::string | [**checkncvarname**](#function-checkncvarname) (int ncid, std::string stringA, std::string stringB, std::string stringC, std::string stringD, std::string stringE) <br> |
|  int | [**nc\_get\_var1\_T**](#function-nc_get_var1_t) (int ncid, int varid, const size\_t \* startp, float \* zsa) <br> |
|  int | [**nc\_get\_var1\_T**](#function-nc_get_var1_t) (int ncid, int varid, const size\_t \* startp, double \* zsa) <br> |
|  int | [**nc\_get\_var\_T**](#function-nc_get_var_t) (int ncid, int varid, float \*& zb) <br> |
|  int | [**nc\_get\_var\_T**](#function-nc_get_var_t) (int ncid, int varid, double \*& zb) <br> |
|  int | [**nc\_get\_vara\_T**](#function-nc_get_vara_t) (int ncid, int varid, const size\_t \* startp, const size\_t \* countp, float \*& zb) <br> |
|  int | [**nc\_get\_vara\_T**](#function-nc_get_vara_t) (int ncid, int varid, const size\_t \* startp, const size\_t \* countp, double \*& zb) <br> |
|  void | [**read2Dnc**](#function-read2dnc) (int nx, int ny, char ncfile, float \*& hh) <br> |
|  void | [**read3Dnc**](#function-read3dnc) (int nx, int ny, int ntheta, char ncfile, float \*& ee) <br> |
|  void | [**readATMstep**](#function-readatmstep) ([**forcingmap**](classforcingmap.md) ATMPmap, int steptoread, float \*& Po) <br> |
|  void | [**readWNDstep**](#function-readwndstep) ([**forcingmap**](classforcingmap.md) WNDUmap, [**forcingmap**](classforcingmap.md) WNDVmap, int steptoread, float \*& Uo, float \*& Vo) <br> |
|  void | [**readgridncsize**](#function-readgridncsize) (const std::string ncfilestr, const std::string varstr, std::string reftime, int & nx, int & ny, int & nt, double & dx, double & dy, double & dt, double & xo, double & yo, double & to, double & xmax, double & ymax, double & tmax, bool & flipx, bool & flipy) <br> |
|  void | [**readgridncsize**](#function-readgridncsize) ([**forcingmap**](classforcingmap.md) & Fmap, [**Param**](classParam.md) XParam) <br> |
|  void | [**readgridncsize**](#function-readgridncsize) (T & Imap) <br> |
|  int | [**readncslev1**](#function-readncslev1) (std::string filename, std::string varstr, size\_t indx, size\_t indy, size\_t indt, bool checkhh, double eps, T \*& zsa) <br> |
|  int | [**readnctime**](#function-readnctime) (std::string filename, double \*& time) <br> |
|  int | [**readnctime2**](#function-readnctime2) (int ncid, char \* timecoordname, std::string refdate, size\_t nt, double \*& time) <br> |
|  void | [**readnczb**](#function-readnczb) (int nx, int ny, std::string ncfile, float \*& zb) <br> |
|  int | [**readvardata**](#function-readvardata) (std::string filename, std::string Varname, int step, T \*& vardata, bool flipx, bool flipy) <br> |
|  int | [**readvarinfo**](#function-readvarinfo) (std::string filename, std::string Varname, size\_t \*& ddimU) <br> |




























## Public Functions Documentation




### function checkncvarname 

```C++
std::string checkncvarname (
    int ncid,
    std::string stringA,
    std::string stringB,
    std::string stringC,
    std::string stringD,
    std::string stringE
) 
```




<hr>



### function nc\_get\_var1\_T 

```C++
inline int nc_get_var1_T (
    int ncid,
    int varid,
    const size_t * startp,
    float * zsa
) 
```




<hr>



### function nc\_get\_var1\_T 

```C++
inline int nc_get_var1_T (
    int ncid,
    int varid,
    const size_t * startp,
    double * zsa
) 
```




<hr>



### function nc\_get\_var\_T 

```C++
inline int nc_get_var_T (
    int ncid,
    int varid,
    float *& zb
) 
```




<hr>



### function nc\_get\_var\_T 

```C++
inline int nc_get_var_T (
    int ncid,
    int varid,
    double *& zb
) 
```




<hr>



### function nc\_get\_vara\_T 

```C++
inline int nc_get_vara_T (
    int ncid,
    int varid,
    const size_t * startp,
    const size_t * countp,
    float *& zb
) 
```




<hr>



### function nc\_get\_vara\_T 

```C++
inline int nc_get_vara_T (
    int ncid,
    int varid,
    const size_t * startp,
    const size_t * countp,
    double *& zb
) 
```




<hr>



### function read2Dnc 

```C++
void read2Dnc (
    int nx,
    int ny,
    char ncfile,
    float *& hh
) 
```




<hr>



### function read3Dnc 

```C++
void read3Dnc (
    int nx,
    int ny,
    int ntheta,
    char ncfile,
    float *& ee
) 
```




<hr>



### function readATMstep 

```C++
void readATMstep (
    forcingmap ATMPmap,
    int steptoread,
    float *& Po
) 
```




<hr>



### function readWNDstep 

```C++
void readWNDstep (
    forcingmap WNDUmap,
    forcingmap WNDVmap,
    int steptoread,
    float *& Uo,
    float *& Vo
) 
```




<hr>



### function readgridncsize 

```C++
void readgridncsize (
    const std::string ncfilestr,
    const std::string varstr,
    std::string reftime,
    int & nx,
    int & ny,
    int & nt,
    double & dx,
    double & dy,
    double & dt,
    double & xo,
    double & yo,
    double & to,
    double & xmax,
    double & ymax,
    double & tmax,
    bool & flipx,
    bool & flipy
) 
```




<hr>



### function readgridncsize 

```C++
void readgridncsize (
    forcingmap & Fmap,
    Param XParam
) 
```




<hr>



### function readgridncsize 

```C++
template<class T>
void readgridncsize (
    T & Imap
) 
```




<hr>



### function readncslev1 

```C++
template<class T>
int readncslev1 (
    std::string filename,
    std::string varstr,
    size_t indx,
    size_t indy,
    size_t indt,
    bool checkhh,
    double eps,
    T *& zsa
) 
```




<hr>



### function readnctime 

```C++
int readnctime (
    std::string filename,
    double *& time
) 
```




<hr>



### function readnctime2 

```C++
int readnctime2 (
    int ncid,
    char * timecoordname,
    std::string refdate,
    size_t nt,
    double *& time
) 
```




<hr>



### function readnczb 

```C++
void readnczb (
    int nx,
    int ny,
    std::string ncfile,
    float *& zb
) 
```




<hr>



### function readvardata 

```C++
template<class T>
int readvardata (
    std::string filename,
    std::string Varname,
    int step,
    T *& vardata,
    bool flipx,
    bool flipy
) 
```




<hr>



### function readvarinfo 

```C++
int readvarinfo (
    std::string filename,
    std::string Varname,
    size_t *& ddimU
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Read_netcdf.h`


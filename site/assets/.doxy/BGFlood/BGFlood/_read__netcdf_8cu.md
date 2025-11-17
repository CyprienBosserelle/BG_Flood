

# File Read\_netcdf.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Read\_netcdf.cu**](_read__netcdf_8cu.md)

[Go to the source code of this file](_read__netcdf_8cu_source.md)



* `#include "Read_netcdf.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  std::string | [**checkncvarname**](#function-checkncvarname) (int ncid, std::string stringA, std::string stringB, std::string stringC, std::string stringD, std::string stringE) <br>_Check for the existence of NetCDF variable names and return the first found._  |
|  int | [**nc\_get\_var1\_T**](#function-nc_get_var1_t) (int ncid, int varid, const size\_t \* startp, float \* zsa) <br> |
|  int | [**nc\_get\_var1\_T**](#function-nc_get_var1_t) (int ncid, int varid, const size\_t \* startp, double \* zsa) <br> |
|  int | [**nc\_get\_var\_T**](#function-nc_get_var_t) (int ncid, int varid, float \*& zb) <br> |
|  int | [**nc\_get\_var\_T**](#function-nc_get_var_t) (int ncid, int varid, double \*& zb) <br> |
|  int | [**nc\_get\_var\_T**](#function-nc_get_var_t) (int ncid, int varid, int \*& zb) <br> |
|  int | [**nc\_get\_vara\_T**](#function-nc_get_vara_t) (int ncid, int varid, const size\_t \* startp, const size\_t \* countp, int \*& zb) <br> |
|  int | [**nc\_get\_vara\_T**](#function-nc_get_vara_t) (int ncid, int varid, const size\_t \* startp, const size\_t \* countp, float \*& zb) <br> |
|  int | [**nc\_get\_vara\_T**](#function-nc_get_vara_t) (int ncid, int varid, const size\_t \* startp, const size\_t \* countp, double \*& zb) <br> |
|  void | [**read2Dnc**](#function-read2dnc) (int nx, int ny, char ncfile, float \*& hh) <br> |
|  void | [**read3Dnc**](#function-read3dnc) (int nx, int ny, int ntheta, char ncfile, float \*& ee) <br> |
|  void | [**readATMstep**](#function-readatmstep) ([**forcingmap**](classforcingmap.md) ATMPmap, int steptoread, float \*& Po) <br>_Read atmospheric pressure data from NetCDF file for a specific time step._  |
|  void | [**readWNDstep**](#function-readwndstep) ([**forcingmap**](classforcingmap.md) WNDUmap, [**forcingmap**](classforcingmap.md) WNDVmap, int steptoread, float \*& Uo, float \*& Vo) <br>_Read wind data from NetCDF files for a specific time step._  |
|  void | [**readgridncsize**](#function-readgridncsize) (const std::string ncfilestr, const std::string varstr, std::string reftime, int & nx, int & ny, int & nt, double & dx, double & dy, double & dt, double & xo, double & yo, double & to, double & xmax, double & ymax, double & tmax, bool & flipx, bool & flipy) <br>_Read grid size and metadata from a NetCDF file._  |
|  void | [**readgridncsize**](#function-readgridncsize) ([**forcingmap**](classforcingmap.md) & Fmap, [**Param**](class_param.md) XParam) <br>_Read grid size and metadata for a forcing map._  |
|  void | [**readgridncsize**](#function-readgridncsize) (T & Imap) <br>_Read grid size and metadata for a generic map type._  |
|  template void | [**readgridncsize&lt; DynForcingP&lt; float &gt; &gt;**](#function-readgridncsize-dynforcingp-float) ([**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; & Imap) <br> |
|  template void | [**readgridncsize&lt; StaticForcingP&lt; float &gt; &gt;**](#function-readgridncsize-staticforcingp-float) ([**StaticForcingP**](struct_static_forcing_p.md)&lt; float &gt; & Imap) <br> |
|  template void | [**readgridncsize&lt; StaticForcingP&lt; int &gt; &gt;**](#function-readgridncsize-staticforcingp-int) ([**StaticForcingP**](struct_static_forcing_p.md)&lt; int &gt; & Imap) <br> |
|  template void | [**readgridncsize&lt; deformmap&lt; float &gt; &gt;**](#function-readgridncsize-deformmap-float) ([**deformmap**](classdeformmap.md)&lt; float &gt; & Imap) <br> |
|  template void | [**readgridncsize&lt; forcingmap &gt;**](#function-readgridncsize-forcingmap) ([**forcingmap**](classforcingmap.md) & Imap) <br> |
|  template void | [**readgridncsize&lt; inputmap &gt;**](#function-readgridncsize-inputmap) ([**inputmap**](classinputmap.md) & Imap) <br> |
|  int | [**readncslev1**](#function-readncslev1) (std::string filename, std::string varstr, size\_t indx, size\_t indy, size\_t indt, bool checkhh, double eps, T \*& zsa) <br>_Read a single level of data from a NetCDF file._  |
|  template int | [**readncslev1&lt; double &gt;**](#function-readncslev1-double) (std::string filename, std::string varstr, size\_t indx, size\_t indy, size\_t indt, bool checkhh, double eps, double \*& zsa) <br> |
|  template int | [**readncslev1&lt; float &gt;**](#function-readncslev1-float) (std::string filename, std::string varstr, size\_t indx, size\_t indy, size\_t indt, bool checkhh, double eps, float \*& zsa) <br> |
|  int | [**readnctime**](#function-readnctime) (std::string filename, double \*& time) <br>_Read time variable from a NetCDF file._  |
|  int | [**readnctime2**](#function-readnctime2) (int ncid, char \* timecoordname, std::string refdate, size\_t nt, double \*& time) <br>_Read time variable from a NetCDF file with reference date._  |
|  void | [**readnczb**](#function-readnczb) (int nx, int ny, std::string ncfile, float \*& zb) <br> |
|  int | [**readvardata**](#function-readvardata) (std::string filename, std::string Varname, int step, T \*& vardata, bool flipx, bool flipy) <br>_Read variable data from a NetCDF file for a specific time step._  |
|  template int | [**readvardata&lt; double &gt;**](#function-readvardata-double) (std::string filename, std::string Varname, int step, double \*& vardata, bool flipx, bool flipy) <br> |
|  template int | [**readvardata&lt; float &gt;**](#function-readvardata-float) (std::string filename, std::string Varname, int step, float \*& vardata, bool flipx, bool flipy) <br> |
|  template int | [**readvardata&lt; int &gt;**](#function-readvardata-int) (std::string filename, std::string Varname, int step, int \*& vardata, bool flipx, bool flipy) <br> |
|  int | [**readvarinfo**](#function-readvarinfo) (std::string filename, std::string Varname, size\_t \*& ddimU) <br>_Read variable dimension info from a NetCDF file._  |




























## Public Functions Documentation




### function checkncvarname 

_Check for the existence of NetCDF variable names and return the first found._ 
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



Checks up to five possible variable names in a NetCDF file and returns the first one that exists.




**Parameters:**


* `ncid` NetCDF file ID 
* `stringA` First variable name to check 
* `stringB` Second variable name to check 
* `stringC` Third variable name to check 
* `stringD` Fourth variable name to check 
* `stringE` Fifth variable name to check 



**Returns:**

The first variable name found in the NetCDF file, or an empty string if none are found. 





        

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



### function nc\_get\_var\_T 

```C++
inline int nc_get_var_T (
    int ncid,
    int varid,
    int *& zb
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
    int *& zb
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

_Read atmospheric pressure data from NetCDF file for a specific time step._ 
```C++
void readATMstep (
    forcingmap ATMPmap,
    int steptoread,
    float *& Po
) 
```



Reads atmospheric pressure data from a NetCDF file for a given time step. Atm pressure is same as wind we only read floats and that is plenty for real world application.




**Parameters:**


* `ATMPmap` [**Forcing**](struct_forcing.md) map for atmospheric pressure 
* `steptoread` Time step to read 
* `Po` Output array for pressure data 




        

<hr>



### function readWNDstep 

_Read wind data from NetCDF files for a specific time step._ 
```C++
void readWNDstep (
    forcingmap WNDUmap,
    forcingmap WNDVmap,
    int steptoread,
    float *& Uo,
    float *& Vo
) 
```



Reads U and V wind components from NetCDF files for a given time step. By default we want to read wind info as float because it will reside in a texture. the value is converted to the apropriate type only when it is used. so there is no need to template this function




**Parameters:**


* `WNDUmap` [**Forcing**](struct_forcing.md) map for U wind 
* `WNDVmap` [**Forcing**](struct_forcing.md) map for V wind 
* `steptoread` Time step to read 
* `Uo` Output array for U wind 
* `Vo` Output array for V wind 




        

<hr>



### function readgridncsize 

_Read grid size and metadata from a NetCDF file._ 
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



Reads dimensions, coordinates, and time information for a variable in a NetCDF file.




**Parameters:**


* `ncfilestr` NetCDF filename 
* `varstr` Variable name 
* `reftime` Reference time string 
* `nx` Number of x grid points 
* `ny` Number of y grid points 
* `nt` Number of time steps 
* `dx` Grid spacing in x 
* `dy` Grid spacing in y 
* `dt` Time step size 
* `xo` Origin x 
* `yo` Origin y 
* `to` Origin time 
* `xmax` Maximum x 
* `ymax` Maximum y 
* `tmax` Maximum time 
* `flipx` Flip x axis 
* `flipy` Flip y axis 




        

<hr>



### function readgridncsize 

_Read grid size and metadata for a forcing map._ 
```C++
void readgridncsize (
    forcingmap & Fmap,
    Param XParam
) 
```



Reads grid size and metadata for a forcing map using model parameters.




**Parameters:**


* `Fmap` [**Forcing**](struct_forcing.md) map structure 
* `XParam` [**Model**](struct_model.md) parameters 




        

<hr>



### function readgridncsize 

_Read grid size and metadata for a generic map type._ 
```C++
template<class T>
void readgridncsize (
    T & Imap
) 
```



Reads grid size and metadata for a generic map type (inputmap, forcingmap, etc.).




**Template parameters:**


* `T` Map type 



**Parameters:**


* `Imap` Map structure 




        

<hr>



### function readgridncsize&lt; DynForcingP&lt; float &gt; &gt; 

```C++
template void readgridncsize< DynForcingP< float > > (
    DynForcingP < float > & Imap
) 
```




<hr>



### function readgridncsize&lt; StaticForcingP&lt; float &gt; &gt; 

```C++
template void readgridncsize< StaticForcingP< float > > (
    StaticForcingP < float > & Imap
) 
```




<hr>



### function readgridncsize&lt; StaticForcingP&lt; int &gt; &gt; 

```C++
template void readgridncsize< StaticForcingP< int > > (
    StaticForcingP < int > & Imap
) 
```




<hr>



### function readgridncsize&lt; deformmap&lt; float &gt; &gt; 

```C++
template void readgridncsize< deformmap< float > > (
    deformmap < float > & Imap
) 
```




<hr>



### function readgridncsize&lt; forcingmap &gt; 

```C++
template void readgridncsize< forcingmap > (
    forcingmap & Imap
) 
```




<hr>



### function readgridncsize&lt; inputmap &gt; 

```C++
template void readgridncsize< inputmap > (
    inputmap & Imap
) 
```




<hr>



### function readncslev1 

_Read a single level of data from a NetCDF file._ 
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



Reads a single level of data for a variable from a NetCDF file.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `filename` NetCDF filename 
* `varstr` Variable name 
* `indx` X index 
* `indy` Y index 
* `indt` Time index 
* `checkhh` Check for missing values 
* `eps` Epsilon for missing value detection 
* `zsa` Output array for data 



**Returns:**

Status code 





        

<hr>



### function readncslev1&lt; double &gt; 

```C++
template int readncslev1< double > (
    std::string filename,
    std::string varstr,
    size_t indx,
    size_t indy,
    size_t indt,
    bool checkhh,
    double eps,
    double *& zsa
) 
```




<hr>



### function readncslev1&lt; float &gt; 

```C++
template int readncslev1< float > (
    std::string filename,
    std::string varstr,
    size_t indx,
    size_t indy,
    size_t indt,
    bool checkhh,
    double eps,
    float *& zsa
) 
```




<hr>



### function readnctime 

_Read time variable from a NetCDF file._ 
```C++
int readnctime (
    std::string filename,
    double *& time
) 
```



Reads the time variable from a NetCDF file into a double array.




**Parameters:**


* `filename` NetCDF filename 
* `time` Output array for time values 



**Returns:**

Status code 





        

<hr>



### function readnctime2 

_Read time variable from a NetCDF file with reference date._ 
```C++
int readnctime2 (
    int ncid,
    char * timecoordname,
    std::string refdate,
    size_t nt,
    double *& time
) 
```



Reads the time variable from a NetCDF file using a reference date and time coordinate name.




**Parameters:**


* `ncid` NetCDF file ID 
* `timecoordname` Time coordinate variable name 
* `refdate` Reference date string 
* `nt` Number of time steps 
* `time` Output array for time values 



**Returns:**

Status code 





        

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

_Read variable data from a NetCDF file for a specific time step._ 
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



Reads data for a variable from a NetCDF file for a given time step, with optional axis flipping.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `filename` NetCDF filename 
* `Varname` Variable name 
* `step` Time step to read 
* `vardata` Output array for data 
* `flipx` Flip x axis 
* `flipy` Flip y axis 



**Returns:**

Status code 





        

<hr>



### function readvardata&lt; double &gt; 

```C++
template int readvardata< double > (
    std::string filename,
    std::string Varname,
    int step,
    double *& vardata,
    bool flipx,
    bool flipy
) 
```




<hr>



### function readvardata&lt; float &gt; 

```C++
template int readvardata< float > (
    std::string filename,
    std::string Varname,
    int step,
    float *& vardata,
    bool flipx,
    bool flipy
) 
```




<hr>



### function readvardata&lt; int &gt; 

```C++
template int readvardata< int > (
    std::string filename,
    std::string Varname,
    int step,
    int *& vardata,
    bool flipx,
    bool flipy
) 
```




<hr>



### function readvarinfo 

_Read variable dimension info from a NetCDF file._ 
```C++
int readvarinfo (
    std::string filename,
    std::string Varname,
    size_t *& ddimU
) 
```



Reads the dimensions for a variable in a NetCDF file.




**Parameters:**


* `filename` NetCDF filename 
* `Varname` Variable name 
* `ddimU` Output array for dimension sizes 



**Returns:**

Number of dimensions 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Read_netcdf.cu`


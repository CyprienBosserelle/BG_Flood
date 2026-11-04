

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
|  void | [**InitSave2Netcdf**](#function-initsave2netcdf) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br>_Initialize NetCDF output files for the model. Initializes NetCDF output files for the model based on the provided parameters and model configuration. If output variables are specified in the parameters, it creates the necessary NetCDF files and defines the variables to be saved._  |
|  void | [**Save2Netcdf**](#function-save2netcdf) ([**Param**](classParam.md) XParam, [**Loop**](structLoop.md)&lt; T &gt; XLoop, [**Model**](structModel.md)&lt; T &gt; & XModel) <br>_Save model output to NetCDF files at specified output times. Saves model output to NetCDF files at specified output times based on the provided parameters. It checks if the current output time matches the next scheduled output time for each output zone, and if so, writes the relevant variables to the corresponding NetCDF files._  |
|  void | [**create2dnc**](#function-create2dnc) (char \* filename, int nx, int ny, double \* xx, double \* yy, double \* var, char \* varname) <br>_Create a NetCDF file containing a 2D variable, for testing for example. Creates a NetCDF file containing a 2D variable with the specified dimensions and data. If the file already exists, it will be overwritten._  |
|  void | [**create3dnc**](#function-create3dnc) (char \* name, int nx, int ny, int nt, double \* xx, double \* yy, double \* theta, double \* var, char \* varname) <br>_Create a NetCDF file containing a 3D variable, for testing for example._  _Creates a NetCDF file containing a 3D variable with the specified dimensions and data. If the file already exists, it will be overwritten._ |
|  void | [**creatncfileBUQ**](#function-creatncfilebuq) ([**Param**](classParam.md) & XParam, int \* activeblk, int \* level, T \* blockxo, T \* blockyo, [**outzoneB**](structoutzoneB.md) & Xzone) <br>_Create a NetCDF file for BG-Flood output. Creates a NetCDF file for BG-Flood output based on the provided parameters and zone information._  |
|  void | [**creatncfileBUQ**](#function-creatncfilebuq) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br>_Create NetCDF files for all output zones in a block. Creates NetCDF files for all output zones defined in the block using the provided parameters and block information._  |
|  void | [**defncvarBUQ**](#function-defncvarbuq) ([**Param**](classParam.md) XParam, int \* activeblk, int \* level, T \* blockxo, T \* blockyo, std::string varst, int vdim, T \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br>_Define a NetCDF variable for BG-Flood output. Defines a NetCDF variable for BG-Flood output based on the provided parameters, block information and zone information._  |
|  void | [**handle\_ncerror**](#function-handle_ncerror) (int status) <br>_Handle NetCDF errors._  |
|  void | [**write2dvarnc**](#function-write2dvarnc) (int nx, int ny, double totaltime, double \* var) <br>_Write a time step of a 2D variable to an existing NetCDF file, for testing for example. Writes a time step of a 2D variable to an existing NetCDF file by appending the provided variable data at the next available time index._  |
|  void | [**write3dvarnc**](#function-write3dvarnc) (int nx, int ny, int nt, double totaltime, double \* var) <br>_Write a time step of a 3D variable to an existing NetCDF file, for testing for example. Writes a time step of a 3D variable to an existing NetCDF file by appending the provided variable data at the next available time index._  |
|  void | [**writenctimestep**](#function-writenctimestep) (std::string outfile, double totaltime) <br>_Write the current time step to a NetCDF file. Writes the current time step to a NetCDF file by updating the "time" variable with the provided total time value._  |
|  void | [**writencvarstepBUQ**](#function-writencvarstepbuq) ([**Param**](classParam.md) XParam, int vdim, int \* activeblk, int \* level, T \* blockxo, T \* blockyo, std::string varst, T \* var, [**outzoneB**](structoutzoneB.md) Xzone) <br>_Write a time step of a NetCDF variable for BG-Flood output. Writes a time step of a NetCDF variable for BG-Flood output based on the provided parameters, block information, variable data, and zone information._  |




























## Public Functions Documentation




### function InitSave2Netcdf 

_Initialize NetCDF output files for the model. Initializes NetCDF output files for the model based on the provided parameters and model configuration. If output variables are specified in the parameters, it creates the necessary NetCDF files and defines the variables to be saved._ 
```C++
template<class T>
void InitSave2Netcdf (
    Param & XParam,
    Model < T > & XModel
) 
```





**Parameters:**


* `XParam` The parameter object containing grid settings and model parameters. 
* `XModel` The model object containing block information and output variable mappings. 



**Note:**

This is a templated function that can handle different data types for the model (e.g., float, double). 




**See also:** creatncfileBUQ for creating the NetCDF file and defining variables. 



        

<hr>



### function Save2Netcdf 

_Save model output to NetCDF files at specified output times. Saves model output to NetCDF files at specified output times based on the provided parameters. It checks if the current output time matches the next scheduled output time for each output zone, and if so, writes the relevant variables to the corresponding NetCDF files._ 
```C++
template<class T>
void Save2Netcdf (
    Param XParam,
    Loop < T > XLoop,
    Model < T > & XModel
) 
```





**Parameters:**


* `XParam` The parameter object containing grid settings and model parameters. 
* `XLoop` The loop object containing time-stepping information. 
* `XModel` The model object containing block information and output variable mappings. 



**Note:**

This is a templated function that can handle different data types for the model (e.g., float, double). 




**See also:** writenctimestep for writing the time step to the NetCDF file. 



        

<hr>



### function create2dnc 

_Create a NetCDF file containing a 2D variable, for testing for example. Creates a NetCDF file containing a 2D variable with the specified dimensions and data. If the file already exists, it will be overwritten._ 
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





**Parameters:**


* `filename` The name of the NetCDF file to be created.
* `nx` The number of grid points in the x-direction. 
* `ny` The number of grid points in the y-direction. 
* `xx` Pointer to an array containing the x-coordinates of the grid points. 
* `yy` Pointer to an array containing the y-coordinates of the grid points. 
* `var` Pointer to an array containing the 2D variable data to be stored in the NetCDF file. 
* `varname` The name of the variable to be stored in the NetCDF file. 



**Note:**

This function uses the NetCDF C library to create and write to the NetCDF file. 





        

<hr>



### function create3dnc 

_Create a NetCDF file containing a 3D variable, for testing for example._  _Creates a NetCDF file containing a 3D variable with the specified dimensions and data. If the file already exists, it will be overwritten._
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





**Parameters:**


* `name` The name of the NetCDF file to be created. 
* `nx` The number of grid points in the x-direction. 
* `ny` The number of grid points in the y-direction. 
* `nt` The number of time steps. 
* `xx` Pointer to an array containing the x-coordinates of the grid points. 
* `yy` Pointer to an array containing the y-coordinates of the grid points. 
* `theta` Pointer to an array containing the time values. 
* `var` Pointer to an array containing the 3D variable data to be stored in the NetCDF file. 
* `varname` The name of the variable to be stored in the NetCDF file. 



**Note:**

This function uses the NetCDF C library to create and write to the NetCDF file. 





        

<hr>



### function creatncfileBUQ 

_Create a NetCDF file for BG-Flood output. Creates a NetCDF file for BG-Flood output based on the provided parameters and zone information._ 
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





**Parameters:**


* `XParam` The parameter object containing grid settings and model parameters. 
* `activeblk` Pointer to an array of active block indices. 
* `level` Pointer to an array of block levels. 
* `blockxo` Pointer to an array of block x-coordinates. 
* `blockyo` Pointer to an array of block y-coordinates. 
* `Xzone` The output zone object defining the area and settings for the NetCDF file. 




        

<hr>



### function creatncfileBUQ 

_Create NetCDF files for all output zones in a block. Creates NetCDF files for all output zones defined in the block using the provided parameters and block information._ 
```C++
template<class T>
void creatncfileBUQ (
    Param & XParam,
    BlockP < T > & XBlock
) 
```





**Parameters:**


* `XParam` The parameter object containing grid settings and model parameters. 
* `XBlock` The block object containing block information and output zones. 




        

<hr>



### function defncvarBUQ 

_Define a NetCDF variable for BG-Flood output. Defines a NetCDF variable for BG-Flood output based on the provided parameters, block information and zone information._ 
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





**Parameters:**


* `XParam` The parameter object containing grid settings and model parameters. 
* `activeblk` Pointer to an array of active block indices. 
* `level` Pointer to an array of block levels. 
* `blockxo` Pointer to an array of block x-coordinates. 
* `blockyo` Pointer to an array of block y-coordinates. 
* `varst` The base name of the variable to be defined. 
* `vdim` The number of dimensions of the variable (2 or 3). 
* `var` Pointer to the array containing the variable data. 
* `Xzone` The output zone object defining the area and settings for the NetCDF variable. 



**Note:**

This is an overloaded function that provides a simpler interface when longname, stdname, and unit are not needed. 





        

<hr>



### function handle\_ncerror 

_Handle NetCDF errors._ 
```C++
void handle_ncerror (
    int status
) 
```




<hr>



### function write2dvarnc 

_Write a time step of a 2D variable to an existing NetCDF file, for testing for example. Writes a time step of a 2D variable to an existing NetCDF file by appending the provided variable data at the next available time index._ 
```C++
void write2dvarnc (
    int nx,
    int ny,
    double totaltime,
    double * var
) 
```





**Parameters:**


* `nx` The number of grid points in the x-direction. 
* `ny` The number of grid points in the y-direction. 
* `totaltime` The total time value to be written to the "time" variable. 
* `var` Pointer to an array containing the 2D variable data to be appended to the NetCDF file. 



**Note:**

This function assumes that the NetCDF file "3Dvar.nc" already exists and is open for writing. 





        

<hr>



### function write3dvarnc 

_Write a time step of a 3D variable to an existing NetCDF file, for testing for example. Writes a time step of a 3D variable to an existing NetCDF file by appending the provided variable data at the next available time index._ 
```C++
void write3dvarnc (
    int nx,
    int ny,
    int nt,
    double totaltime,
    double * var
) 
```





**Parameters:**


* `nx` The number of grid points in the x-direction. 
* `ny` The number of grid points in the y-direction. 
* `nt` The number of time steps. 
* `totaltime` The total time value to be written to the "time" variable. 
* `var` Pointer to an array containing the 3D variable data to be appended to the NetCDF file. 



**Note:**

This function assumes that the NetCDF file "3Dvar.nc" already exists and is open for writing. 





        

<hr>



### function writenctimestep 

_Write the current time step to a NetCDF file. Writes the current time step to a NetCDF file by updating the "time" variable with the provided total time value._ 
```C++
void writenctimestep (
    std::string outfile,
    double totaltime
) 
```





**Parameters:**


* `outfile` The name of the NetCDF output file. 
* `totaltime` The total time value to be written to the "time" variable. 




        

<hr>



### function writencvarstepBUQ 

_Write a time step of a NetCDF variable for BG-Flood output. Writes a time step of a NetCDF variable for BG-Flood output based on the provided parameters, block information, variable data, and zone information._ 
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





**Parameters:**


* `XParam` The parameter object containing grid settings and model parameters. 
* `vdim` The number of dimensions of the variable (2 or 3). 
* `activeblk` Pointer to an array of active block indices. 
* `level` Pointer to an array of block levels. 
* `blockxo` Pointer to an array of block x-coordinates. 
* `blockyo` Pointer to an array of block y-coordinates. 
* `varst` The base name of the variable to be written. 
* `var` Pointer to the array containing the variable data. 
* `Xzone` The output zone object defining the area and settings for the NetCDF variable. 



**Note:**

This is a templated function that can handle different data types for the variable (e.g., float, double). 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Write_netcdf.h`


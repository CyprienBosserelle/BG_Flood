

# File ReadForcing.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**ReadForcing.h**](_read_forcing_8h.md)

[Go to the source code of this file](_read_forcing_8h_source.md)



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
|  std::vector&lt; std::string &gt; | [**DelimLine**](#function-delimline) (std::string line, int n, char delim) <br>_Split a string into a vector of substrings based on a specified delimiter._  |
|  std::vector&lt; std::string &gt; | [**DelimLine**](#function-delimline) (std::string line, int n) <br>_Split a string into a vector of substrings based on common delimiters (tab, space, comma). Tries tab, space, and comma as delimiters and returns the first successful split with the expected number of elements._  |
|  void | [**InitDynforcing**](#function-initdynforcing) (bool gpgpu, [**Param**](class_param.md) & XParam, [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; & Dforcing) <br>_Initialize dynamic forcing data (Rain, wind, Atm pressure)._  |
|  void | [**InterpstepCPU**](#function-interpstepcpu) (int nx, int ny, int hdstep, float totaltime, float hddt, T \*& Ux, T \* Uo, T \* Un) <br> |
|  void | [**clampedges**](#function-clampedges) (int nx, int ny, T clamp, T \* z) <br>_Clamp the edges of a 2D array to a specified value. Sets the values at the edges of a 2D array to a specified clamp value._  |
|  void | [**denan**](#function-denan) (int nx, int ny, float denanval, int \* z) <br> |
|  void | [**denan**](#function-denan) (int nx, int ny, float denanval, T \* z) <br>_Replace NaN values in a 2D array with a specified value. Iterates through a 2D array and replaces any NaN values with the specified denanval._  |
|  void | [**readDynforcing**](#function-readdynforcing) (bool gpgpu, double totaltime, [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; & Dforcing) <br>_Deprecated function (!!!) to read dynamic forcing data for a given time._  |
|  std::vector&lt; [**Flowin**](class_flowin.md) &gt; | [**readFlowfile**](#function-readflowfile) (std::string Flowfilename, std::string & refdate) <br>_Read flow data from a specified file, for river forcing for example. Reads flow data from a specified file and extracts time and flow rate information. Applies reference date adjustment if provided._  |
|  std::vector&lt; [**Windin**](class_windin.md) &gt; | [**readINfileUNI**](#function-readinfileuni) (std::string filename, std::string & refdate) <br>_Read rain/atmospheric pressure data from a specified file for spatially uniform forcing. Reads rain/atmospheric pressure data from a specified file and extracts time and wind speed information. Applies reference date adjustment if provided._  |
|  std::vector&lt; [**SLTS**](class_s_l_t_s.md) &gt; | [**readNestfile**](#function-readnestfile) (std::string ncfile, std::string varname, int hor, double eps, double bndxo, double bndxmax, double bndy) <br>_Read boundary nesting data from a NetCDF file. Reads boundary nesting data from a specified NetCDF file and variable name. Supports both horizontal and vertical boundaries._  |
|  [**Polygon**](class_polygon.md) | [**readPolygon**](#function-readpolygon) (std::string filename) <br>_Read polygon from a specified file. Reads polygon vertices from a specified file and ensures the polygon is closed. Calculates bounding box of the polygon._  |
|  std::vector&lt; [**SLTS**](class_s_l_t_s.md) &gt; | [**readWLfile**](#function-readwlfile) (std::string WLfilename, std::string & refdate) <br>_Read water level boundary file. Reads water level boundary file and extracts time and water level data. Applies reference date adjustment if provided._  |
|  std::vector&lt; [**Windin**](class_windin.md) &gt; | [**readWNDfileUNI**](#function-readwndfileuni) (std::string filename, std::string & refdate, double grdalpha) <br>_Read wind data from a specified file for spatially uniform forcing. Reads wind data from a specified file and extracts time, wind speed, wind direction, and calculates u and v wind components. Applies reference date adjustment if provided._  __ |
|  void | [**readXBbathy**](#function-readxbbathy) (std::string filename, int nx, int ny, T \*& zb) <br>_Read bathymetry data from an XBeach-style .bot/.dep file into a provided array. Parses the file format and fills the provided array with bathymetry values._  |
|  void | [**readbathyASCHead**](#function-readbathyaschead) (std::string filename, int & nx, int & ny, double & dx, double & xo, double & yo, double & grdalpha) <br>_Read header information from an ASC (bathymetry) file. Extracts grid size (nx, ny), grid spacing (dx), origin (xo, yo), and grid rotation angle (grdalpha). Adjusts origin if the file uses corner registration._  |
|  void | [**readbathyASCzb**](#function-readbathyasczb) (std::string filename, int nx, int ny, T \*& zb) <br>_Read (bathymetry) data from an ASC file into a provided array. Parses the ASC file format and fills the provided array with bathymetry values._  |
|  void | [**readbathyHeadMD**](#function-readbathyheadmd) (std::string filename, int & nx, int & ny, double & dx, double & grdalpha) <br>_Read header information from an MD bathymetry file (or other MD input map). Extracts grid size (nx, ny), grid spacing (dx), and grid rotation angle (grdalpha)._  |
|  void | [**readbathyMD**](#function-readbathymd) (std::string filename, T \*& zb) <br>_Read bathymetry data from an MD file into a provided array. Parses the MD file format and fills the provided array with bathymetry values._  __ |
|  std::vector&lt; [**SLTS**](class_s_l_t_s.md) &gt; | [**readbndfile**](#function-readbndfile) (std::string filename, [**Param**](class_param.md) & XParam) <br>_Read boundary forcing files (water levels or nest files)._  |
|  [**Polygon**](class_polygon.md) | [**readbndpolysegment**](#function-readbndpolysegment) ([**bndsegment**](classbndsegment.md) bnd, [**Param**](class_param.md) XParam) <br>_Read boundary polygon segment and create polygon structure._  |
|  void | [**readforcing**](#function-readforcing) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; T &gt; & XForcing) <br>_Wrapping function for reading all the forcing data._  |
|  void | [**readforcingdata**](#function-readforcingdata) (int step, T forcing) <br>_Read static forcing data from various file formats based on the file extension. Supports reading from .md, .nc, .bot/.dep, and .asc files._  |
|  void | [**readforcingdata**](#function-readforcingdata) (double totaltime, [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; & forcing) <br>_Read dynamic forcing data from a NetCDF file based on the current simulation time. Interpolates between time steps to obtain the current forcing values. Handles NaN values and clamps edges if specified._  |
|  [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; | [**readforcinghead**](#function-readforcinghead) ([**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; Fmap, [**Param**](class_param.md) XParam) <br>_Read dynamic forcing metadata/header from a NetCDF file. Extracts grid size, spacing, origin, and time information._  |
|  T | [**readforcinghead**](#function-readforcinghead) (T Fmap) <br>_Read static forcing metadata/header from various file formats based on the file extension. Supports reading from .md, .nc, .bot/.dep, and .asc files. Extracts grid size, spacing, origin, and other relevant information._  __ |
|  void | [**readstaticforcing**](#function-readstaticforcing) (T & Sforcing) <br>_Single parameter version of readstaticforcing(int step, T& Sforcing)._  |
|  void | [**readstaticforcing**](#function-readstaticforcing) (int step, T & Sforcing) <br>_Allocate and read static (i.e. not varying in time) forcing data. Used for Bathymetry, roughness, deformation, etc._  |




























## Public Functions Documentation




### function DelimLine 

_Split a string into a vector of substrings based on a specified delimiter._ 
```C++
std::vector< std::string > DelimLine (
    std::string line,
    int n,
    char delim
) 
```





**Parameters:**


* `s` The input string to be split 
* `delim` The delimiter character used for splitting 



**Returns:**

Vector of substrings 





        

<hr>



### function DelimLine 

_Split a string into a vector of substrings based on common delimiters (tab, space, comma). Tries tab, space, and comma as delimiters and returns the first successful split with the expected number of elements._ 
```C++
std::vector< std::string > DelimLine (
    std::string line,
    int n
) 
```





**Parameters:**


* `line` The input string to be split 
* `n` The expected number of elements after splitting 



**Returns:**

Vector of substrings if successful; empty vector otherwise 





        

<hr>



### function InitDynforcing 

_Initialize dynamic forcing data (Rain, wind, Atm pressure)._ 
```C++
void InitDynforcing (
    bool gpgpu,
    Param & XParam,
    DynForcingP < float > & Dforcing
) 
```



Reads dynamic forcing header and allocates memory for dynamic forcing arrays.




**Parameters:**


* `gpgpu` Use GPU acceleration 
* `XParam` [**Model**](struct_model.md) parameters 
* `Dforcing` Dynamic forcing structure 




        

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

_Clamp the edges of a 2D array to a specified value. Sets the values at the edges of a 2D array to a specified clamp value._ 
```C++
template<class T>
void clampedges (
    int nx,
    int ny,
    T clamp,
    T * z
) 
```





**Parameters:**


* `nx` Number of grid points in the x-direction 
* `ny` Number of grid points in the y-direction 
* `clamp` Value to set at the edges 
* `z` Pointer to the 2D array (flattened as 1D) to be modified 




        

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

_Replace NaN values in a 2D array with a specified value. Iterates through a 2D array and replaces any NaN values with the specified denanval._ 
```C++
template<class T>
void denan (
    int nx,
    int ny,
    float denanval,
    T * z
) 
```





**Parameters:**


* `nx` Number of grid points in the x-direction 
* `ny` Number of grid points in the y-direction 
* `denanval` Value to replace NaN values with 
* `z` Pointer to the 2D array (flattened as 1D) to be modified 




        

<hr>



### function readDynforcing 

_Deprecated function (!!!) to read dynamic forcing data for a given time._ 
```C++
void readDynforcing (
    bool gpgpu,
    double totaltime,
    DynForcingP < float > & Dforcing
) 
```



Reads and allocates dynamic forcing arrays for the specified time.




**Parameters:**


* `gpgpu` Use GPU acceleration 
* `totaltime` Current simulation time 
* `Dforcing` Dynamic forcing structure

This is a deprecated function! See InitDynforcing() instead 


        

<hr>



### function readFlowfile 

_Read flow data from a specified file, for river forcing for example. Reads flow data from a specified file and extracts time and flow rate information. Applies reference date adjustment if provided._ 
```C++
std::vector< Flowin > readFlowfile (
    std::string Flowfilename,
    std::string & refdate
) 
```





**Parameters:**


* `Flowfilename` Name of the flow data file 
* `refdate` Reference date for time adjustment 



**Returns:**

Vector of [**Flowin**](class_flowin.md) structures containing time and flow rate data 





        

<hr>



### function readINfileUNI 

_Read rain/atmospheric pressure data from a specified file for spatially uniform forcing. Reads rain/atmospheric pressure data from a specified file and extracts time and wind speed information. Applies reference date adjustment if provided._ 
```C++
std::vector< Windin > readINfileUNI (
    std::string filename,
    std::string & refdate
) 
```





**Parameters:**


* `filename` Name of the rain/atmospheric pressure data file 
* `refdate` Reference date for time adjustment 



**Returns:**

Vector of [**Windin**](class_windin.md) structures containing time and wind speed data 





        

<hr>



### function readNestfile 

_Read boundary nesting data from a NetCDF file. Reads boundary nesting data from a specified NetCDF file and variable name. Supports both horizontal and vertical boundaries._ 
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





**Parameters:**


* `ncfile` Name of the NetCDF file 
* `varname` Name of the variable to read 
* `hor` If 1, read horizontal boundary (top/bottom); if 0 read vertical boundary (left/right) 
* `eps` Small value to avoid numerical issues 
* `bndxo` Starting coordinate of the boundary 
* `bndxmax` Ending coordinate of the boundary 
* `bndy` Fixed coordinate of the boundary 



**Returns:**

Vector of [**SLTS**](class_s_l_t_s.md) structures containing time and water level data 





        

<hr>



### function readPolygon 

_Read polygon from a specified file. Reads polygon vertices from a specified file and ensures the polygon is closed. Calculates bounding box of the polygon._ 
```C++
Polygon readPolygon (
    std::string filename
) 
```





**Parameters:**


* `filename` Name of the polygon file 



**Returns:**

[**Polygon**](class_polygon.md) structure containing vertices and bounding box information 





        

<hr>



### function readWLfile 

_Read water level boundary file. Reads water level boundary file and extracts time and water level data. Applies reference date adjustment if provided._ 
```C++
std::vector< SLTS > readWLfile (
    std::string WLfilename,
    std::string & refdate
) 
```





**Parameters:**


* `WLfilename` Name of the water level boundary file 
* `refdate` Reference date for time adjustment 
 



**Returns:**

Vector of [**SLTS**](class_s_l_t_s.md) structures containing time and water level data 





        

<hr>



### function readWNDfileUNI 

_Read wind data from a specified file for spatially uniform forcing. Reads wind data from a specified file and extracts time, wind speed, wind direction, and calculates u and v wind components. Applies reference date adjustment if provided._  __
```C++
std::vector< Windin > readWNDfileUNI (
    std::string filename,
    std::string & refdate,
    double grdalpha
) 
```





**Parameters:**


* `filename` Name of the wind data file 
* `refdate` Reference date for time adjustment 
* `grdalpha` Grid rotation angle in radians 



**Returns:**

Vector of [**Windin**](class_windin.md) structures containing time, wind speed, wind direction, and u/v components 





        

<hr>



### function readXBbathy 

_Read bathymetry data from an XBeach-style .bot/.dep file into a provided array. Parses the file format and fills the provided array with bathymetry values._ 
```C++
template<class T>
void readXBbathy (
    std::string filename,
    int nx,
    int ny,
    T *& zb
) 
```





**Parameters:**


* `filename` Name of the XBeach-style bathymetry file 
* `nx` Number of grid points in the x-direction 
* `ny` Number of grid points in the y-direction 
* `zb` Reference to the array to store bathymetry values 




        

<hr>



### function readbathyASCHead 

_Read header information from an ASC (bathymetry) file. Extracts grid size (nx, ny), grid spacing (dx), origin (xo, yo), and grid rotation angle (grdalpha). Adjusts origin if the file uses corner registration._ 
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





**Parameters:**


* `filename` Name of the ASC bathymetry file 
* `nx` Reference to store the number of grid points in the x-direction 
* `ny` Reference to store the number of grid points in the y-direction 
* `dx` Reference to store the grid spacing in the x-direction 
* `xo` Reference to store the x-coordinate of the grid origin 
* `yo` Reference to store the y-coordinate of the grid origin 
* `grdalpha` Reference to store the grid rotation angle in radians 




        

<hr>



### function readbathyASCzb 

_Read (bathymetry) data from an ASC file into a provided array. Parses the ASC file format and fills the provided array with bathymetry values._ 
```C++
template<class T>
void readbathyASCzb (
    std::string filename,
    int nx,
    int ny,
    T *& zb
) 
```





**Parameters:**


* `filename` Name of the ASC bathymetry file 
* `nx` Number of grid points in the x-direction 
* `ny` Number of grid points in the y-direction 
* `zb` Reference to the array to store bathymetry values 




        

<hr>



### function readbathyHeadMD 

_Read header information from an MD bathymetry file (or other MD input map). Extracts grid size (nx, ny), grid spacing (dx), and grid rotation angle (grdalpha)._ 
```C++
void readbathyHeadMD (
    std::string filename,
    int & nx,
    int & ny,
    double & dx,
    double & grdalpha
) 
```





**Parameters:**


* `filename` Name of the MD bathymetry file 
* `nx` Reference to store the number of grid points in the x-direction 
* `ny` Reference to store the number of grid points in the y-direction 
* `dx` Reference to store the grid spacing 
* `grdalpha` Reference to store the grid rotation angle in radians 




        

<hr>



### function readbathyMD 

_Read bathymetry data from an MD file into a provided array. Parses the MD file format and fills the provided array with bathymetry values._  __
```C++
template<class T>
void readbathyMD (
    std::string filename,
    T *& zb
) 
```





**Parameters:**


* `filename` Name of the MD bathymetry file 
* `zb` Reference to the array to store bathymetry values 




        

<hr>



### function readbndfile 

_Read boundary forcing files (water levels or nest files)._ 
```C++
std::vector< SLTS > readbndfile (
    std::string filename,
    Param & XParam
) 
```



Reads boundary forcing files based on their extension (.nc for nest files, others for water level files). Applies zsoffset correction if specified in model parameters. 

**Parameters:**


* `filename` Name of the boundary forcing file 
* `XParam` [**Model**](struct_model.md) parameters 



**Returns:**

Vector of [**SLTS**](class_s_l_t_s.md) structures containing boundary information 





        

<hr>



### function readbndpolysegment 

_Read boundary polygon segment and create polygon structure._ 
```C++
Polygon readbndpolysegment (
    bndsegment bnd,
    Param XParam
) 
```



Reads boundary polygon segment based on specified keywords or file input.




**Parameters:**


* `bnd` Boundary segment structure 
* `XParam` [**Model**](struct_model.md) parameters 



**Returns:**

[**Polygon**](class_polygon.md) structure representing the boundary segment 





        

<hr>



### function readforcing 

_Wrapping function for reading all the forcing data._ 
```C++
template<class T>
void readforcing (
    Param & XParam,
    Forcing < T > & XForcing
) 
```



Reads bathymetry and other forcing data into the provided [**Forcing**](struct_forcing.md) structure.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data structure

wrapping function for reading all the forcing data 


        

<hr>



### function readforcingdata 

_Read static forcing data from various file formats based on the file extension. Supports reading from .md, .nc, .bot/.dep, and .asc files._ 
```C++
template<class T>
void readforcingdata (
    int step,
    T forcing
) 
```





**Template parameters:**


* `T` Type of the forcing parameter structure (e.g., StaticForcingP&lt;float&gt;, deformmap&lt;float&gt;, etc.) 



**Parameters:**


* `step` Current time step for reading time-dependent data (if applicable) 
* `forcing` [**Forcing**](struct_forcing.md) parameter structure containing file information and data storage 




        

<hr>



### function readforcingdata 

_Read dynamic forcing data from a NetCDF file based on the current simulation time. Interpolates between time steps to obtain the current forcing values. Handles NaN values and clamps edges if specified._ 
```C++
void readforcingdata (
    double totaltime,
    DynForcingP < float > & forcing
) 
```





**Parameters:**


* `totaltime` Current simulation time 
* `forcing` Dynamic forcing parameter structure containing file information and data storage 




        

<hr>



### function readforcinghead 

_Read dynamic forcing metadata/header from a NetCDF file. Extracts grid size, spacing, origin, and time information._ 
```C++
DynForcingP < float > readforcinghead (
    DynForcingP < float > Fmap,
    Param XParam
) 
```





**Parameters:**


* `Fmap` Dynamic forcing parameter structure containing file information 
* `XParam` Simulation parameters (used for reference date) 



**Returns:**

Updated dynamic forcing parameter structure with metadata 





        

<hr>



### function readforcinghead 

_Read static forcing metadata/header from various file formats based on the file extension. Supports reading from .md, .nc, .bot/.dep, and .asc files. Extracts grid size, spacing, origin, and other relevant information._  __
```C++
template<class T>
T readforcinghead (
    T Fmap
) 
```





**Template parameters:**


* `T` Type of the forcing parameter structure (e.g., StaticForcingP&lt;float&gt;, deformmap&lt;float&gt;, etc.) 



**Parameters:**


* `ForcingParam` [**Forcing**](struct_forcing.md) parameter structure containing file information 



**Returns:**

Updated forcing parameter structure with metadata 





        

<hr>



### function readstaticforcing 

_Single parameter version of readstaticforcing(int step, T& Sforcing)._ 
```C++
template<class T>
void readstaticforcing (
    T & Sforcing
) 
```



Calls readstaticforcing with step set to 0.




**Template parameters:**


* `T` Data type 



**Parameters:**


* `Sforcing` Static forcing structure to be read and allocated

single parameter version of readstaticforcing(int step,T& Sforcing) readstaticforcing(0, Sforcing); 


        

<hr>



### function readstaticforcing 

_Allocate and read static (i.e. not varying in time) forcing data. Used for Bathymetry, roughness, deformation, etc._ 
```C++
template<class T>
void readstaticforcing (
    int step,
    T & Sforcing
) 
```





**Parameters:**


* `step` Time step (usually 0 for static data) 
* `Sforcing` Static forcing structure to be read and allocated

Allocate and read static (i.e. not varying in time) forcing Used for Bathy, roughness, deformation 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/ReadForcing.h`




# File Write\_txtlog.cpp



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Write\_txtlog.cpp**](Write__txtlog_8cpp.md)

[Go to the source code of this file](Write__txtlog_8cpp_source.md)



* `#include "Write_txtlog.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**SaveParamtolog**](#function-saveparamtolog) ([**Param**](classParam.md) XParam) <br>_Save model parameters to the log file. Saves the model parameters from the given_ [_**Param**_](classParam.md) _object to the log file "BG\_log.txt"._ |
|  void | [**create\_logfile**](#function-create_logfile) () <br>_Create a log file for BG-Flood. Creates a log file named "BG\_log.txt" for BG-Flood. The log file is reset if it already exists. The function also writes a header with the current date and time._  |
|  void | [**log**](#function-log) (std::string text) <br> |
|  void | [**saveparam2netCDF**](#function-saveparam2netcdf) (int ncid, int bgfid, [**Param**](classParam.md) XParam) <br>_Save model parameters to a NetCDF file. Saves the model parameters from the given_ [_**Param**_](classParam.md) _object to a NetCDF file._ |
|  void | [**write\_text\_to\_log\_file**](#function-write_text_to_log_file) (std::string text) <br>_Write text to the log file. Writes the given text to the log file "BG\_log.txt"._  |




























## Public Functions Documentation




### function SaveParamtolog 

_Save model parameters to the log file. Saves the model parameters from the given_ [_**Param**_](classParam.md) _object to the log file "BG\_log.txt"._
```C++
void SaveParamtolog (
    Param XParam
) 
```





**Parameters:**


* `XParam` The [**Param**](classParam.md) object containing the model parameters 




        

<hr>



### function create\_logfile 

_Create a log file for BG-Flood. Creates a log file named "BG\_log.txt" for BG-Flood. The log file is reset if it already exists. The function also writes a header with the current date and time._ 
```C++
void create_logfile () 
```




<hr>



### function log 

```C++
void log (
    std::string text
) 
```




<hr>



### function saveparam2netCDF 

_Save model parameters to a NetCDF file. Saves the model parameters from the given_ [_**Param**_](classParam.md) _object to a NetCDF file._
```C++
void saveparam2netCDF (
    int ncid,
    int bgfid,
    Param XParam
) 
```





**Parameters:**


* `ncid` The NetCDF file ID 
* `bgfid` The NetCDF group ID for the parameters 
* `XParam` The [**Param**](classParam.md) object containing the model parameters 




        

<hr>



### function write\_text\_to\_log\_file 

_Write text to the log file. Writes the given text to the log file "BG\_log.txt"._ 
```C++
void write_text_to_log_file (
    std::string text
) 
```





**Parameters:**


* `text` The text to write to the log file 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Write_txtlog.cpp`


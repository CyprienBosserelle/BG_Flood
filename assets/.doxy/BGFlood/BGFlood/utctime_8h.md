

# File utctime.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**utctime.h**](utctime_8h.md)

[Go to the source code of this file](utctime_8h_source.md)



* `#include "General.h"`
* `#include "ReadInput.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  double | [**date\_string\_to\_s**](#function-date_string_to_s) (std::string datetime, std::string refdate) <br>_Convert a date string to seconds from a reference date. Converts a date string in the format "YYYY-MM-DDTHH:MM:SS" or "YYYY/MM/DD HH:MM:SS" to seconds from the reference date._  |
|  long long | [**date\_string\_to\_time**](#function-date_string_to_time) (std::string date) <br>_Convert a date string to a Unix timestamp. Converts a date string in the format "YYYY-MM-DDTHH:MM:SS" or "YYYY/MM/DD HH:MM:SS" to a Unix timestamp (number of seconds since the beginning of 1970 CE)._  |
|  double | [**readinputtimetxt**](#function-readinputtimetxt) (std::string input, std::string & refdate) <br>_Read a time string and convert it to seconds. Reads a time string and converts it to seconds. If the string is a valid datetime string it returns the seconds from the reference date, otherwise it returns a float of seconds._  |
|  bool | [**testime1**](#function-testime1) (int hour) <br>_Test time calculation functions. Test time calculation functions._  |
|  bool | [**testime2**](#function-testime2) (int hour) <br>_Test time calculation functions for greater than comparison. Test time calculation functions for greater than comparison._  |




























## Public Functions Documentation




### function date\_string\_to\_s 

_Convert a date string to seconds from a reference date. Converts a date string in the format "YYYY-MM-DDTHH:MM:SS" or "YYYY/MM/DD HH:MM:SS" to seconds from the reference date._ 
```C++
double date_string_to_s (
    std::string datetime,
    std::string refdate
) 
```





**Parameters:**


* `datetime` The date string to convert 
* `refdate` The reference date string 



**Returns:**

The number of seconds from the reference date as double 





        

<hr>



### function date\_string\_to\_time 

_Convert a date string to a Unix timestamp. Converts a date string in the format "YYYY-MM-DDTHH:MM:SS" or "YYYY/MM/DD HH:MM:SS" to a Unix timestamp (number of seconds since the beginning of 1970 CE)._ 
```C++
long long date_string_to_time (
    std::string date
) 
```





**Parameters:**


* `date` The date string to convert 



**Returns:**

The corresponding Unix timestamp as long long 





        

<hr>



### function readinputtimetxt 

_Read a time string and convert it to seconds. Reads a time string and converts it to seconds. If the string is a valid datetime string it returns the seconds from the reference date, otherwise it returns a float of seconds._ 
```C++
double readinputtimetxt (
    std::string input,
    std::string & refdate
) 
```





**Parameters:**


* `input` The input time string 
* `refdate` The reference date string 



**Returns:**

The time in seconds as double 





        

<hr>



### function testime1 

_Test time calculation functions. Test time calculation functions._ 
```C++
bool testime1 (
    int hour
) 
```





**Parameters:**


* `hour` The hour to test 




        

<hr>



### function testime2 

_Test time calculation functions for greater than comparison. Test time calculation functions for greater than comparison._ 
```C++
bool testime2 (
    int hour
) 
```





**Parameters:**


* `hour` The hour to test 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/utctime.h`




# File utctime.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**utctime.cu**](utctime_8cu.md)

[Go to the source code of this file](utctime_8cu_source.md)



* `#include "utctime.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  double | [**date\_string\_to\_s**](#function-date_string_to_s) (std::string datetime, std::string refdate) <br>_Convert a date string to seconds from a reference date. Converts a date string in the format "YYYY-MM-DDTHH:MM:SS" or "YYYY/MM/DD HH:MM:SS" to seconds from the reference date._  |
|  long long | [**date\_string\_to\_time**](#function-date_string_to_time) (std::string date) <br>_Convert a date string to a Unix timestamp. Converts a date string in the format "YYYY-MM-DDTHH:MM:SS" or "YYYY/MM/DD HH:MM:SS" to a Unix timestamp (number of seconds since the beginning of 1970 CE)._  |
|  int | [**days\_from\_epoch**](#function-days_from_epoch) (int y, int m, int d) <br>_Calculate the number of days from the epoch (1970-01-01) to a given date. Algorithm:_ [http://howardhinnant.github.io/date\_algorithms.html](http://howardhinnant.github.io/date_algorithms.html) _._ |
|  struct tm \* | [**gmtime\_r**](#function-gmtime_r) (const time\_t \* timep, struct tm \* tm) <br>_Convert a Unix timestamp to a Gregorian civil date-time tuple._  |
|  double | [**readinputtimetxt**](#function-readinputtimetxt) (std::string input, std::string & refdate) <br>_Read a time string and convert it to seconds. Reads a time string and converts it to seconds. If the string is a valid datetime string it returns the seconds from the reference date, otherwise it returns a float of seconds._  |
|  bool | [**testime1**](#function-testime1) (int hour) <br>_Test time calculation functions. Test time calculation functions._  |
|  bool | [**testime2**](#function-testime2) (int hour) <br>_Test time calculation functions for greater than comparison. Test time calculation functions for greater than comparison._  |
|  long long | [**timegm**](#function-timegm) (struct tm const \* t) <br>_Convert a Gregorian civil date-time tuple to a Unix timestamp. Converts a Gregorian civil date-time tuple in GMT (UTC) time zone to a Unix timestamp._  |




























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



### function days\_from\_epoch 

_Calculate the number of days from the epoch (1970-01-01) to a given date. Algorithm:_ [http://howardhinnant.github.io/date\_algorithms.html](http://howardhinnant.github.io/date_algorithms.html) _._
```C++
int days_from_epoch (
    int y,
    int m,
    int d
) 
```





**Parameters:**


* `y` Year 
* `m` Month 
* `d` Day 




        

<hr>



### function gmtime\_r 

_Convert a Unix timestamp to a Gregorian civil date-time tuple._ 
```C++
struct tm * gmtime_r (
    const time_t * timep,
    struct tm * tm
) 
```



Converts a Unix timestamp (number of seconds since the beginning of 1970 CE) to a Gregorian civil date-time tuple in GMT (UTC) time zone.


This conforms to C89 (and C99...) and POSIX.


This implementation works, and doesn't overflow for any sizeof(time\_t). It doesn't check for overflow/underflow in tm-&gt;tm\_year output. Other than that, it never overflows or underflows. It assumes that that time\_t is signed.


This implements the inverse of the POSIX formula ([http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1\_chap04.html#tag\_04\_15](http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_15)) for all time\_t values, no matter the size, as long as tm-&gt;tm\_year doesn't overflow or underflow. The formula is: tm\_sec + tm\_min\*60 + tm\_hour\*3600
* tm\_yday\*86400 + (tm\_year-70)\*31536000 + ((tm\_year-69)/4)\*86400 - ((tm\_year-1)/100)\*86400 + ((tm\_year+299)/400)\*86400.






**Parameters:**


* `timep` Pointer to the Unix timestamp 
* `tm` Pointer to the struct tm to be filled 



**Returns:**

Pointer to the filled struct tm 





        

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



### function timegm 

_Convert a Gregorian civil date-time tuple to a Unix timestamp. Converts a Gregorian civil date-time tuple in GMT (UTC) time zone to a Unix timestamp._ 
```C++
long long timegm (
    struct tm const * t
) 
```





**Note:**

It does not modify broken-down time 




**Parameters:**


* `t` Pointer to the struct tm to be converted 



**Returns:**

The corresponding Unix timestamp 





        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/utctime.cu`


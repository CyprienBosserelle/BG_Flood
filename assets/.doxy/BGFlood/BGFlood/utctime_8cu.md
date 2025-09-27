

# File utctime.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**utctime.cu**](utctime_8cu.md)

[Go to the source code of this file](utctime_8cu_source.md)



* `#include "utctime.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  double | [**date\_string\_to\_s**](#function-date_string_to_s) (std::string datetime, std::string refdate) <br> |
|  long long | [**date\_string\_to\_time**](#function-date_string_to_time) (std::string date) <br> |
|  int | [**days\_from\_epoch**](#function-days_from_epoch) (int y, int m, int d) <br> |
|  struct tm \* | [**gmtime\_r**](#function-gmtime_r) (const time\_t \* timep, struct tm \* tm) <br> |
|  double | [**readinputtimetxt**](#function-readinputtimetxt) (std::string input, std::string & refdate) <br> |
|  bool | [**testime1**](#function-testime1) (int hour) <br> |
|  bool | [**testime2**](#function-testime2) (int hour) <br> |
|  long long | [**timegm**](#function-timegm) (struct tm const \* t) <br> |




























## Public Functions Documentation




### function date\_string\_to\_s 

```C++
double date_string_to_s (
    std::string datetime,
    std::string refdate
) 
```




<hr>



### function date\_string\_to\_time 

```C++
long long date_string_to_time (
    std::string date
) 
```




<hr>



### function days\_from\_epoch 

```C++
int days_from_epoch (
    int y,
    int m,
    int d
) 
```




<hr>



### function gmtime\_r 

```C++
struct tm * gmtime_r (
    const time_t * timep,
    struct tm * tm
) 
```




<hr>



### function readinputtimetxt 

```C++
double readinputtimetxt (
    std::string input,
    std::string & refdate
) 
```




<hr>



### function testime1 

```C++
bool testime1 (
    int hour
) 
```




<hr>



### function testime2 

```C++
bool testime2 (
    int hour
) 
```




<hr>



### function timegm 

```C++
long long timegm (
    struct tm const * t
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/utctime.cu`


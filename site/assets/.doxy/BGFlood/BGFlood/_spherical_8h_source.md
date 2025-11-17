

# File Spherical.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Spherical.h**](_spherical_8h.md)

[Go to the documentation of this file](_spherical_8h.md)


```C++
#ifndef SPHERICAL_H
#define SPHERICAL_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Util_CPU.h"
#include "Kurganov.h"

template <class T> __host__ __device__ T calcCM(T Radius, T delta, T yo, int iy);
template <class T> __host__ __device__  T calcFM(T Radius, T delta, T yo, T iy);
template <class T> __host__ __device__  T spharea(T Radius, T lon, T lat, T dx);


#endif
```



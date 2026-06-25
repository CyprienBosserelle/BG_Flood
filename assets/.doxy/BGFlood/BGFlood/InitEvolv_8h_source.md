

# File InitEvolv.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**InitEvolv.h**](InitEvolv_8h.md)

[Go to the documentation of this file](InitEvolv_8h.md)


```C++

#ifndef INITEVOLV_H
#define INITEVOLV_H

#include "General.h"
#include "Param.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "Write_txtlog.h"
#include "GridManip.h"
#include "Read_netcdf.h"
#include "ReadForcing.h"
#include "Updateforcing.h"


template <class T> void initevolv(Param XParam, BlockP<T> XBlock, Forcing<float> XForcing, EvolvingP<T>& XEv, T*& zb);
template <class T> int coldstart(Param XParam, BlockP<T> XBlock, T* zb, EvolvingP<T>& XEv);
template <class T> void warmstart(Param XParam, Forcing<float> XForcing, BlockP<T> XBlock, T* zb, EvolvingP<T>& XEv);
template <class T> int AddZSoffset(Param XParam, BlockP<T> XBlock, EvolvingP<T>& XEv, T* zb);

template <class T> int readhotstartfile(Param XParam, BlockP<T> XBlock, EvolvingP<T>& XEv, T*& zb);

// End of global definition;
#endif
```



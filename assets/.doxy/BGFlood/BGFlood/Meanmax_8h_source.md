

# File Meanmax.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Meanmax.h**](Meanmax_8h.md)

[Go to the documentation of this file](Meanmax_8h.md)


```C++

#ifndef MEANMAX_H
#define MEANMAX_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "FlowGPU.h"


template <class T> void Calcmeanmax(Param XParam, Loop<T>& XLoop, Model<T> XModel, Model<T> XModel_g);
template <class T> void resetmeanmax(Param XParam, Loop<T>& XLoop, Model<T> XModel, Model<T> XModel_g);
template <class T> void Initmeanmax(Param XParam, Loop<T> XLoop, Model<T> XModel, Model<T> XModel_g);
template <class T> void resetvalGPU(Param XParam, BlockP<T> XBlock, T*& var, T val);


template <class T> __global__ void addavg_varGPU(Param XParam, BlockP<T> XBlock, T* Varmean, T* Var);
template <class T> __global__ void divavg_varGPU(Param XParam, BlockP<T> XBlock, T ntdiv, T* Varmean);
template <class T> __global__ void addUandhU_GPU(Param XParam, BlockP<T> XBlock, T* h, T* u, T* v, T* U, T* hU);
template <class T> __global__ void max_varGPU(Param XParam, BlockP<T> XBlock, T* Varmax, T* Var);
template <class T> __global__ void max_Norm_GPU(Param XParam, BlockP<T> XBlock, T* Varmax, T* Var1, T* Var2);
template <class T> __global__ void max_hU_GPU(Param XParam, BlockP<T> XBlock, T* Varmax, T* Var1, T* Var2, T* Var3);
template <class T> __global__ void addwettime_GPU(Param XParam, BlockP<T> XBlock, T* wett, T* h, T thresold, T time);

// End of global definition
#endif
```



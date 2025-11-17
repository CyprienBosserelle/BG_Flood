

# File FlowGPU.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**FlowGPU.h**](_flow_g_p_u_8h.md)

[Go to the documentation of this file](_flow_g_p_u_8h.md)


```C++
#ifndef FLOWGPU_H
#define FLOWGPU_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "Util_CPU.h"
#include "MemManagement.h"
#include "Gradients.h"
#include "Kurganov.h"
#include "Advection.h"
#include "Friction.h"
#include "Updateforcing.h"
#include "Reimann.h"
#include "Boundary.h"

template <class T> void FlowGPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel);

template <class T> __global__ void reset_var(int halowidth, int* active, T resetval, T* Var);

template <class T> void HalfStepGPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel);

// End of global definition
#endif
```



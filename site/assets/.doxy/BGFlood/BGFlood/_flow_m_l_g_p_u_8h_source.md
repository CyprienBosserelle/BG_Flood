

# File FlowMLGPU.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**FlowMLGPU.h**](_flow_m_l_g_p_u_8h.md)

[Go to the documentation of this file](_flow_m_l_g_p_u_8h.md)


```C++
#ifndef FLOWMLGPU_H
#define FLOWMLGPU_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "Util_CPU.h"
#include "MemManagement.h"
#include "Multilayer.h"
#include "FlowGPU.h"
#include "Advection.h"

template <class T> void FlowMLGPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel);


// End of global definition
#endif
```



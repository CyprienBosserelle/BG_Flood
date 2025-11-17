

# File FlowCPU.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**FlowCPU.h**](_flow_c_p_u_8h.md)

[Go to the documentation of this file](_flow_c_p_u_8h.md)


```C++
#ifndef FLOWCPU_H
#define FLOWCPU_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "Util_CPU.h"
#include "MemManagement.h"
#include "Halo.h"
#include "GridManip.h"
#include "Gradients.h"
#include "Kurganov.h"
#include "Advection.h"
#include "Friction.h"
#include "Updateforcing.h"
#include "Reimann.h"

// End of global definition
template <class T> void FlowCPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel);
template <class T> void HalfStepCPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel);


#endif
```



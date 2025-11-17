

# File Mainloop.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Mainloop.h**](_mainloop_8h.md)

[Go to the documentation of this file](_mainloop_8h.md)


```C++
#ifndef MAINLOOP_H
#define MAINLOOP_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "Mesh.h"
#include "Write_netcdf.h"
#include "InitialConditions.h"
#include "MemManagement.h"
#include "Boundary.h"
#include "FlowGPU.h"
#include "FlowCPU.h"
#include "Meanmax.h"
#include "Updateforcing.h"
#include "FlowMLGPU.h"

template <class T> void MainLoop(Param& XParam, Forcing<float> XForcing, Model<T>& XModel, Model<T>& XModel_g);

template <class T> void DebugLoop(Param& XParam, Forcing<float> XForcing, Model<T>& XModel, Model<T>& XModel_g);

template <class T> __host__ double initdt(Param XParam, Loop<T> XLoop, Model<T> XModel);

template <class T> Loop<T> InitLoop(Param& XParam, Model<T>& XModel);

template <class T> void printstatus(T totaltime, T dt);


template <class T> __global__ void storeTSout(Param XParam, int noutnodes, int outnode, int istep, int blknode, int inode, int jnode, int* blkTS, EvolvingP<T> XEv, T* store);


// End of global definition
#endif
```



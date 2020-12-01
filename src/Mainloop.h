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

template <class T> void MainLoop(Param& XParam, Forcing<float> XForcing, Model<T>& XModel, Model<T>& XModel_g);

template <class T> __host__ double initdt(Param XParam, Loop<T> XLoop, Model<T> XModel);

template <class T> Loop<T> InitLoop(Param& XParam, Model<T>& XModel);

// End of global definition
#endif

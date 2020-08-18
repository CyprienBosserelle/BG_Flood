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

template <class T> void MainLoop(Param& XParam, Forcing<float> XForcing, Model<T>& XModel, Model<T>& XModel_g);





// End of global definition
#endif

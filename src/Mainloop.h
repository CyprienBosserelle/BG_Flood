#ifndef MAINLOOP_H
#define MAINLOOP_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "Mesh.h"
#include "Write_netcdf.h"
#include "InitialConditions.cu"

template <class T> void MainLoop(Param XParam, Forcing<float> XForcing, Model<T> XModel);

template <class T> __global__ void reset_var(unsigned int halowidth, unsigned int* active, T resetval, T* Var);

// End of global definition
#endif

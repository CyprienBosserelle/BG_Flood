#ifndef FRICTION_H
#define FRICTION_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "MemManagement.h"

template <class T> __global__ void bottomfrictionGPU(Param XParam, BlockP<T> XBlock, T* cf, EvolvingP<T> XEvolv);
template <class T> __host__ void bottomfrictionCPU(Param XParam, BlockP<T> XBlock, T* cf, EvolvingP<T> XEvolv);



// End of global definition
#endif

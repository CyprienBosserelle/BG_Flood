#ifndef FRICTION_H
#define FRICTION_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "MemManagement.h"

template <class T> __global__ void bottomfrictionGPU(Param XParam, BlockP<T> XBlock,T dt, T* cf, EvolvingP<T> XEvolv);
template <class T> __host__ void bottomfrictionCPU(Param XParam, BlockP<T> XBlock,T dt, T* cf, EvolvingP<T> XEvolv);

template <class T> __host__ void XiafrictionCPU(Param XParam, BlockP<T> XBlock, T dt, T* cf, EvolvingP<T> XEvolv, EvolvingP<T> XEvolv_o);
template <class T> __global__ void XiafrictionGPU(Param XParam, BlockP<T> XBlock, T dt, T* cf, EvolvingP<T> XEvolv, EvolvingP<T> XEvolv_o);


// End of global definition
#endif


#ifndef UPDATEFORCING_H
#define UPDATEFORCING_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "InitialConditions.h"
#include "MemManagement.h"
#include "ReadForcing.h"
#include "GridManip.h"
#include "Util_CPU.h"

template <class T> void updateforcing(Param XParam, Loop<T> XLoop, Forcing<float>& XForcing);

void Forcingthisstep(Param XParam, double totaltime, DynForcingP<float>& XDynForcing);

template <class T> __device__ T interpDyn2BUQ(T x, T y, TexSetP Forcing);

template <class T> __host__ void AddwindforcingCPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> Uwind, DynForcingP<float> Vwind, AdvanceP<T> XAdv);
template <class T> __global__ void AddwindforcingGPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> Uwind, DynForcingP<float> Vwind, AdvanceP<T> XAdv);

template <class T> __host__ void AddrainforcingCPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> Rain, AdvanceP<T> XAdv);
template <class T> __host__ void AddrainforcingImplicitCPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, DynForcingP<float> Rain, EvolvingP<T> XEv);
template <class T> __global__ void AddrainforcingGPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> Rain, AdvanceP<T> XAdv);
template <class T> __global__ void AddrainforcingImplicitGPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, DynForcingP<float> Rain, EvolvingP<T> XEv);

template <class T> __host__ void  AddinfiltrationImplicitCPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, T* il, T* cl, EvolvingP<T> XEv, T* hgw);
template <class T> __global__ void AddinfiltrationImplicitGPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, T* il, T* cl, EvolvingP<T> XEv, T* hgw);

template <class T> __global__ void AddPatmforcingGPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> PAtm, Model<T> XModel);
template <class T> __host__ void AddPatmforcingCPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> PAtm, Model<T> XModel);

template <class T> __host__ void AddRiverForcing(Param XParam, Loop<T> XLoop, std::vector<River> XRivers, Model<T> XModel);

template <class T> void deformstep(Param XParam, Loop<T> XLoop, std::vector<deformmap<float>> deform, Model<T> XModel, Model<T> XModel_g);

template <class T> __global__ void InjectRiverGPU(Param XParam, River XRiver, T qnow, int* Riverblks, BlockP<T> XBlock, AdvanceP<T> XAdv);
template <class T> __global__ void AddDeformGPU(Param XParam, BlockP<T> XBlock, deformmap<float> defmap, EvolvingP<T> XEv, T scale, T* zb);


#endif

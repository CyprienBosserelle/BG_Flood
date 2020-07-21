
#ifndef INITIALCONDITION_H
#define INITIALCONDITION_H

#include "General.h"
#include "Param.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Utils_GPU.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "Write_txt.h"
#include "GridManip.h"
#include "Read_netcdf.h"
#include "ReadForcing.h"


template <class T> void InitialConditions(Param XParam, Forcing<float> XForcing, Model<T> XModel);
template <class T> void initevolv(Param XParam, BlockP<T> XBlock, EvolvingP<T>& XEv, T*& zb);
template <class T> int coldstart(Param XParam, BlockP<T> XBlock, T* zb, EvolvingP<T>& XEv);
template <class T> void warmstart(Param XParam, BlockP<T> XBlock, T* zb, EvolvingP<T>& XEv);
template <class T> int AddZSoffset(Param XParam, BlockP<T> XBlock, EvolvingP<T>& XEv, T* zb);

template <class T> int readhotstartfile(Param XParam, BlockP<T> XBlock, EvolvingP<T>& XEv, T*& zb);

// End of global definition;
#endif

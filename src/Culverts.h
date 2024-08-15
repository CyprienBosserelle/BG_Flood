
#ifndef CULVERTS_H
#define CULVERTS_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "GridManip.h"
#include "Util_CPU.h"


template <class T> __global__ void AddCulvertsCPU(XParam, XLoop, XForcing.culverts, XModel);
template <class T> __global__ void AddCulvertsGPU(XParam, XLoop, XForcing.culverts, XModel);

#endif

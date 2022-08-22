#ifndef SPHERICAL_H
#define SPHERICAL_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Util_CPU.h"
#include "Kurganov.h"

template <class T> __host__ __device__ T calcCM(T Radius, T delta, T yo, int iy);
template <class T> __host__ __device__  T calcFM(T Radius, T delta, T yo, T iy);


#endif

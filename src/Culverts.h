
#ifndef CULVERTS_H
#define CULVERTS_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "GridManip.h"
#include "Util_CPU.h"


template <class T> __host__ void AddCulverts(Param XParam, double dt, std::vector<Culvert> XCulverts, Model<T> XModel);

template <class T> __global__ void InjectCulvertGPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T> XCulvertF, int* Culvertblks, BlockP<T> XBlock, AdvanceP<T>& XAdv);
template <class T> __host__ void InjectCulvertCPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T> XCulvertF, int nblkculvert, int* Culvertblks, BlockP<T> XBlock, AdvanceP<T>& XAdv);

template <class T> __global__ void GetCulvertElevGPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T>& XCulvertF, int* Culvertblks, BlockP<T> XBlock, EvolvingP<T> XEv);
template <class T> __host__ void GetCulvertElevCPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T>& XCulvertF, int* Culvertblks, BlockP<T> XBlock, EvolvingP<T> XEv);

#endif

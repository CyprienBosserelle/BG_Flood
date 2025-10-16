
#ifndef CULVERTS_H
#define CULVERTS_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "GridManip.h"
#include "Util_CPU.h"
#include "MemManagement.h"



template <class T> __host__ void AddCulverts(Param XParam, double dt, std::vector<Culvert> XCulverts, Model<T> XModel);

template <class T> __global__ void InjectCulvertGPU(Param XParam, int cc, Culvert XCulvert, CulvertF<T> XCulvertF, int* Culvertblks, BlockP<T> XBlock, AdvanceP<T> XAdv);
template <class T> __host__ void InjectCulvertCPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T>& XCulvertF, int nblkculvert, int* Culvertblks, BlockP<T> XBlock, AdvanceP<T> XAdv);

template <class T> __global__ void GetCulvertElevGPU(Param XParam, int cc, int b1, int ix1, int iy1, int b2, int ix2, int iy2, CulvertF<T>& XCulvertF, int* Culvertblks, EvolvingP<T> XEv);
template <class T> __host__ void GetCulvertElevCPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T>& XCulvertF, int nblkculvert, int* Culvertblks, BlockP<T> XBlock, EvolvingP<T> XEv);

template <class T> __host__ void DischargeCulvertCPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T>& XCulvertF, int nblkculvert, int* Culvertblks);
template <class T> __global__ void DischargeCulvertGPU(Param XParam, CulvertF<T> XCulvertF);

template <class T> __host__ __device__ void CulvertPump(double Qmax, double dx1, T h1, T h2, T zs1, T zs2, T& dq, double dt);

#endif

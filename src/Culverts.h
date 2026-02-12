
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

template <class T> __host__ void DischargeCulvertCPU(Param XParam, double dt, std::vector<Culvert> XCulverts, CulvertF<T>& XCulvertF, int nblkculvert, int* Culvertblks);
template <class T> __global__ void DischargeCulvertGPU(Param XParam, double dt, CulvertF<T> XCulvertF);

template <class T> __host__ __device__ void CulvertPump(double Qmax, double dx1, T h1, T h2, T zs1, T zs2, T& dq, double dt);

template <class T> __host__ __device__ void CulvertDischarge(int shape, T width, T height, T length, T zb1, T zb2, T k_ex, T k_en, T C_d, T n, T zs1, T zs2, T u1, T u2, T v1, T v2, T& Q);

__host__ __device__ double manningQ(double A, double R, double n, double S);
template <class T> __host__ __device__ void OutletControled(T& Q, double k_ex, double k_en, double A_wet, double g, double H_L, T u2, T v2, double L, double R_wet, double n);

template <class T> __host__ __device__ void rect_geom(T b, T h, T& A, T& P, T& R, T& A_wet, T& P_wet, T& R_wet, T h_wet);
template <class T>__host__ __device__ void circular_geom(T R, T& A, T& P, T& A_wet, T& P_wet, T& R_wet, T h_wet);
template <class T> __host__ __device__ void normal_depth(T Q, T h_culvert, T b, T n, T S, T& h_n, int shape);
template <class T> __host__ __device__ void critical_depth_circular(T Q, T h_culvert, T& h_critical);


#endif

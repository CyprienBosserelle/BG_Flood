
#ifndef SETUPGPU_H
#define SETUPGPU_H

#include "General.h"
#include "Forcing.h"
#include "Param.h"
#include "Arrays.h"
#include "MemManagement.h"
#include "Halo.h"
#include "InitialConditions.h"

void CUDA_CHECK(cudaError CUDerr);

template <class T> void SetupGPU(Param XParam, Model<T> XModel, Forcing<float>& XForcing, Model<T>& XModel_g);
void AllocateTEX(int nx, int ny, TexSetP& Tex, float* input);
void AllocateBndTEX(bndparam& side);

template <class T> void CopyGPUtoCPU(int nblk, int blksize, T* z_cpu, T* z_gpu);

template <class T> void CopytoGPU(int nblk, int blksize, Param XParam, Model<T> XModel_cpu, Model<T> XModel_gpu);
template <class T> void CopytoGPU(int nblk, int blksize, T* z_cpu, T* z_gpu);
template <class T> void CopytoGPU(int nblk, int blksize, EvolvingP<T> XEv_cpu, EvolvingP<T> XEv_gpu);
template <class T> void CopytoGPU(int nblk, int blksize, GradientsP<T> XGrad_cpu, GradientsP<T> XGrad_gpu);

// End of global definition
#endif


#ifndef SETUPGPU_H
#define SETUPGPU_H

#include "General.h"
#include "Forcing.h"
#include "Param.h"
#include "Arrays.h"
#include "MemManagement.h"
#include "Halo.h"

void CUDA_CHECK(cudaError CUDerr);

template <class T> void SetupGPU(Param XParam, Model<T> XModel, Forcing<float>& XForcing, Model<T>& XModel_g);
void AllocateTEX(int nx, int ny, TexSetP& Tex, float* input);
void AllocateBndTEX(bndparam& side);

template <class T> void CopyGPUtoCPU(int nblk, int blksize, T* z_cpu, T* z_gpu);
// End of global definition
#endif

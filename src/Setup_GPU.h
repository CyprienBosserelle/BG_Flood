
#ifndef SETUPGPU_H
#define SETUPGPU_H

#include "General.h"
#include "Forcing.h"
#include "Param.h"
#include "Arrays.h"
#include "MemManagement.h"

void CUDA_CHECK(cudaError CUDerr);

template <class T> void SetupGPU(Param XParam, Model<T> XModel, Model<T>& XModel_g);
void AllocateTEX(int nx, int ny, TexSetP& Tex, float* input);
void AllocateBndTEX(bndparam& side);
// End of global definition
#endif

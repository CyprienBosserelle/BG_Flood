
#ifndef SETUPGPU_H
#define SETUPGPU_H

#include "General.h"
#include "Forcing.h"


void CUDA_CHECK(cudaError CUDerr);
void AllocateTEX(int nx, int ny, TexSetP& Tex, float* input);
void AllocateBndTEX(bndparam& side);
// End of global definition
#endif

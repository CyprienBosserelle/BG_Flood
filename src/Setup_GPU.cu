
#include "Setup_GPU.h"

void CUDA_CHECK(cudaError CUDerr)
{


	if (cudaSuccess != CUDerr) {

		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \

			__FILE__, __LINE__, cudaGetErrorString(CUDerr));

		exit(EXIT_FAILURE);

	}
}

/*
void AllocateTEX(int nx, int ny, TexSetP& Tex, float* input)
{


	CUDA_CHECK(cudaMallocArray(&Tex.CudArr, &Tex.channelDesc, nx, ny));
	CUDA_CHECK(cudaMemcpyToArray(Tex.CudArr, 0, 0, input, nx * ny * sizeof(float), cudaMemcpyHostToDevice));


	memset(&Tex.texDesc, 0, sizeof(cudaTextureDesc));
	Tex.texDesc.addressMode[0] = cudaAddressModeClamp;
	Tex.texDesc.addressMode[1] = cudaAddressModeClamp;
	Tex.texDesc.filterMode = cudaFilterModeLinear;
	Tex.texDesc.normalizedCoords = false;

	memset(&Tex.resDesc, 0, sizeof(cudaResourceDesc));

	Tex.resDesc.resType = cudaResourceTypeArray;
	Tex.resDesc.res.array.array = Tex.CudArr;

	CUDA_CHECK(cudaCreateTextureObject(&Tex.tex, &Tex.resDesc, &Tex.texDesc, NULL));
	//CUDA_CHECK(cudaBindTextureToArray(Tex, zca, cCFD));


}
*/






#include "Setup_GPU.h"

void CUDA_CHECK(cudaError CUDerr)
{


	if (cudaSuccess != CUDerr) {

		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \

			__FILE__, __LINE__, cudaGetErrorString(CUDerr));

		exit(EXIT_FAILURE);

	}
}


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


void AllocateBndTEX(bndparam & side)
{
	int nbndtimes = (int)side.data.size();
	int nbndvec = (int)side.data[0].wlevs.size();
	
	float* lWLS;
	lWLS = (float*)malloc(nbndtimes * nbndvec * sizeof(float));

	for (int ibndv = 0; ibndv < nbndvec; ibndv++)
	{
		for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
		{
			//
			lWLS[ibndt + ibndv * nbndtimes] = (float)side.data[ibndt].wlevs[ibndv];
		}
	}
	AllocateTEX(nbndtimes, nbndvec, side.GPU.WLS, lWLS);
	
	// In case of Nesting U and V are also prescribed

	// If uu information is available in the boundary we can assume it is a nesting type of bnd
	int nbndvecuu = (int)side.data[0].uuvel.size();
	if (nbndvecuu == nbndvec)
	{
		//
		for (int ibndv = 0; ibndv < nbndvec; ibndv++)
		{
			for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
			{
				//
				lWLS[ibndt + ibndv * nbndtimes] = (float)side.data[ibndt].uuvel[ibndv];
			}
		}
		AllocateTEX(nbndtimes, nbndvec, side.GPU.Uvel, lWLS);
		
	}
	//V velocity side
	int nbndvecvv = (int)side.data[0].vvvel.size();

	if (nbndvecvv == nbndvec)
	{
		for (int ibndv = 0; ibndv < nbndvec; ibndv++)
		{
			for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
			{
				//
				lWLS[ibndt + ibndv * nbndtimes] = (float)side.data[ibndt].vvvel[ibndv];
			}
		}
		AllocateTEX(nbndtimes, nbndvec, side.GPU.Vvel, lWLS);
	}

	free(lWLS);


}





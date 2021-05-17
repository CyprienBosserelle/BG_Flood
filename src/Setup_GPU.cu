
#include "Setup_GPU.h"



template <class T> void SetupGPU(Param XParam, Model<T> XModel,Forcing<float> &XForcing, Model<T>& XModel_g)
{
	if (XParam.GPUDEVICE >= 0)
	{
		log("Setting up GPU");


		cudaSetDevice(XParam.GPUDEVICE);
		//Allocate memory for the model on the GPU
		AllocateGPU(XParam.nblkmem, XParam.blksize, XParam, XModel_g);
		
		// Copy arrays from CPU to GPU
		CopytoGPU(XParam.nblkmem, XParam.blksize,XParam, XModel, XModel_g);

		//
		fillHaloGPU(XParam, XModel_g.blocks, XModel_g.evolv);

		//=============================
		// Same for Bnds


		// Allocate memory for the boundary blk
		AllocateGPU(XForcing.left.nblk, 1, XForcing.left.blks_g);
		//copy bnd blk info on GPU
		CopytoGPU(XForcing.left.nblk, 1, XForcing.left.blks, XForcing.left.blks_g);

		AllocateGPU(XForcing.right.nblk, 1, XForcing.right.blks_g);
		CopytoGPU(XForcing.right.nblk, 1, XForcing.right.blks, XForcing.right.blks_g);

		AllocateGPU(XForcing.top.nblk, 1, XForcing.top.blks_g);
		CopytoGPU(XForcing.top.nblk, 1, XForcing.top.blks, XForcing.top.blks_g);

		AllocateGPU(XForcing.bot.nblk, 1, XForcing.bot.blks_g);
		CopytoGPU(XForcing.bot.nblk, 1, XForcing.bot.blks, XForcing.bot.blks_g);

		// Also for mask
		XModel_g.blocks.mask.nblk = XModel.blocks.mask.nblk;
		AllocateGPU(XModel_g.blocks.mask.nblk, 1, XModel_g.blocks.mask.side);
		AllocateGPU(XModel_g.blocks.mask.nblk, 1, XModel_g.blocks.mask.blks);
		CopytoGPU(XModel_g.blocks.mask.nblk, 1, XModel.blocks.mask.side, XModel_g.blocks.mask.side);
		CopytoGPU(XModel_g.blocks.mask.nblk, 1, XModel.blocks.mask.blks, XModel_g.blocks.mask.blks);


		// things are quite different for Time Series output. Why is that?.
		if (XModel.bndblk.nblkTs > 0)
		{

			AllocateGPU(XModel.bndblk.nblkTs, 1, XModel_g.bndblk.Tsout);
			CopytoGPU(XModel.bndblk.nblkTs, 1, XModel.bndblk.Tsout, XModel_g.bndblk.Tsout);

		}

		// River are a bit of a special case too
		if (XForcing.rivers.size() > 0)
		{
			//
			XModel_g.bndblk.nblkriver = XModel.bndblk.nblkriver;
			AllocateGPU(XModel.bndblk.nblkriver, 1, XModel_g.bndblk.river);
			CopytoGPU(XModel.bndblk.nblkriver, 1, XModel.bndblk.river, XModel_g.bndblk.river);
		}

		// Reset GPU mean and max arrays
		if (XParam.outmax)
		{
			//ResetmaxvarGPU(XParam);
		}
		if (XParam.outmean)
		{
			//ResetmeanvarGPU(XParam);
		}

		Initmaparray(XModel_g);
	}
}
template void SetupGPU<float>(Param XParam, Model<float> XModel, Forcing<float>& XForcing, Model<float>& XModel_g);
template void SetupGPU<double>(Param XParam, Model<double> XModel, Forcing<float>& XForcing, Model<double>& XModel_g);


void CUDA_CHECK(cudaError CUDerr)
{


	if (cudaSuccess != CUDerr) {

		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \

			__FILE__, __LINE__, cudaGetErrorString(CUDerr));

		exit(EXIT_FAILURE);

	}
}


template <class T> void CopytoGPU(int nblk, int blksize, T * z_cpu, T* z_gpu)
{
	CUDA_CHECK(cudaMemcpy(z_gpu, z_cpu, nblk * blksize * sizeof(T), cudaMemcpyHostToDevice));
}
template void CopytoGPU<bool>(int nblk, int blksize, bool* z_cpu, bool* z_gpu);
template void CopytoGPU<int>(int nblk, int blksize, int* z_cpu, int* z_gpu);
template void CopytoGPU<float>(int nblk, int blksize, float* z_cpu, float* z_gpu);
template void CopytoGPU<double>(int nblk, int blksize, double* z_cpu, double* z_gpu);

template <class T> void CopyGPUtoCPU(int nblk, int blksize, T* z_cpu, T* z_gpu)
{
	CUDA_CHECK(cudaMemcpy(z_cpu, z_gpu, nblk * blksize * sizeof(T), cudaMemcpyDeviceToHost));
}
template void CopyGPUtoCPU<bool>(int nblk, int blksize, bool* z_cpu, bool* z_gpu);
template void CopyGPUtoCPU<int>(int nblk, int blksize, int* z_cpu, int* z_gpu);
template void CopyGPUtoCPU<float>(int nblk, int blksize, float* z_cpu, float* z_gpu);
template void CopyGPUtoCPU<double>(int nblk, int blksize, double* z_cpu, double* z_gpu);

template <class T> void CopytoGPU(int nblk, int blksize, EvolvingP<T> XEv_cpu, EvolvingP<T> XEv_gpu)
{
	CopytoGPU(nblk, blksize, XEv_cpu.h, XEv_gpu.h);
	CopytoGPU(nblk, blksize, XEv_cpu.zs, XEv_gpu.zs);
	CopytoGPU(nblk, blksize, XEv_cpu.u, XEv_gpu.u);
	CopytoGPU(nblk, blksize, XEv_cpu.v, XEv_gpu.v);
}
template void CopytoGPU<float>(int nblk, int blksize, EvolvingP<float> XEv_cpu, EvolvingP<float> XEv_gpu);
template void CopytoGPU < double >(int nblk, int blksize, EvolvingP<double> XEv_cpu, EvolvingP < double >  XEv_gpu);


template <class T> void CopytoGPU(int nblk, int blksize, GradientsP<T> XGrad_cpu, GradientsP<T> XGrad_gpu)
{
	CopytoGPU(nblk, blksize, XGrad_cpu.dhdx, XGrad_gpu.dhdx);
	CopytoGPU(nblk, blksize, XGrad_cpu.dhdy, XGrad_gpu.dhdy);
	CopytoGPU(nblk, blksize, XGrad_cpu.dudx, XGrad_gpu.dudx);
	CopytoGPU(nblk, blksize, XGrad_cpu.dudy, XGrad_gpu.dudy);
	CopytoGPU(nblk, blksize, XGrad_cpu.dvdx, XGrad_gpu.dvdx);
	CopytoGPU(nblk, blksize, XGrad_cpu.dvdy, XGrad_gpu.dvdy);
	CopytoGPU(nblk, blksize, XGrad_cpu.dzsdx, XGrad_gpu.dzsdx);
	CopytoGPU(nblk, blksize, XGrad_cpu.dzsdy, XGrad_gpu.dzsdy);
}
template void CopytoGPU(int nblk, int blksize, GradientsP<float> XGrad_cpu, GradientsP<float> XGrad_gpu);
template void CopytoGPU(int nblk, int blksize, GradientsP<double> XGrad_cpu, GradientsP<double> XGrad_gpu);

template <class T> void CopytoGPU(int nblk, int blksize, Param XParam, Model<T> XModel_cpu, Model<T> XModel_gpu)
{
	CopytoGPU(nblk, blksize, XModel_cpu.zb, XModel_gpu.zb);

	CopytoGPU(nblk, blksize, XModel_cpu.evolv, XModel_gpu.evolv);
	//CopytoGPU(nblk, blksize, XModel_cpu.evolv_o, XModel_gpu.evolv_o);

	CopytoGPU(nblk, blksize, XModel_cpu.evolv_o, XModel_gpu.evolv_o);

	CopytoGPU(nblk, blksize, XModel_cpu.cf, XModel_gpu.cf);

	CopytoGPU(nblk, blksize, XModel_cpu.zb, XModel_gpu.zb);

	

	//Block info
	CopytoGPU(nblk, 1, XModel_cpu.blocks.active, XModel_gpu.blocks.active);
	CopytoGPU(nblk, 1, XModel_cpu.blocks.level, XModel_gpu.blocks.level);

	CopytoGPU(nblk, 1, XModel_cpu.blocks.xo, XModel_gpu.blocks.xo);
	CopytoGPU(nblk, 1, XModel_cpu.blocks.yo, XModel_gpu.blocks.yo);

	CopytoGPU(nblk, 1, XModel_cpu.blocks.BotLeft, XModel_gpu.blocks.BotLeft);
	CopytoGPU(nblk, 1, XModel_cpu.blocks.BotRight, XModel_gpu.blocks.BotRight);

	CopytoGPU(nblk, 1, XModel_cpu.blocks.TopLeft, XModel_gpu.blocks.TopLeft);
	CopytoGPU(nblk, 1, XModel_cpu.blocks.TopRight, XModel_gpu.blocks.TopRight);

	CopytoGPU(nblk, 1, XModel_cpu.blocks.LeftBot, XModel_gpu.blocks.LeftBot);
	CopytoGPU(nblk, 1, XModel_cpu.blocks.LeftTop, XModel_gpu.blocks.LeftTop);

	CopytoGPU(nblk, 1, XModel_cpu.blocks.RightBot, XModel_gpu.blocks.RightBot);
	CopytoGPU(nblk, 1, XModel_cpu.blocks.RightTop, XModel_gpu.blocks.RightTop);
	

	if (XParam.outmax)
	{
		CopytoGPU(nblk, blksize, XModel_cpu.evolv, XModel_gpu.evmax);
	}
	if (XParam.outmean)
	{
		CopytoGPU(nblk, blksize, XModel_cpu.evolv, XModel_gpu.evmean);
	}

}
template void CopytoGPU<float>(int nblk, int blksize, Param XParam, Model<float> XModel_cpu, Model<float> XModel_gpu);
template void CopytoGPU<double>(int nblk, int blksize, Param XParam, Model<double> XModel_cpu, Model<double> XModel_gpu);


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




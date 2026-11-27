
#include "Setup_GPU.h"


/**
 * @brief Setup and initialize GPU for simulation.
 *
 * This function sets up the GPU device, allocates memory, and copies data from the host to the device.	
 * @param XParam Simulation parameters
 * @param XModel Host model data structure
 * @param XForcing Forcing data structure
 * @param XModel_g Device model data structure
 * 
 */
template <class T> void SetupGPU(Param &XParam, Model<T> XModel,Forcing<float> &XForcing, Model<T>& XModel_g)
{
	if (XParam.GPUDEVICE >= 0)
	{
		log("Setting up GPU");

		size_t free_byte;

		size_t total_byte;

		cudaSetDevice(XParam.GPUDEVICE);

		CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));

		XParam.GPU_initmem_byte = total_byte - free_byte;


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


		for (int s = 0; s < XForcing.bndseg.size(); s++)
		{
			AllocateGPU(XForcing.bndseg[s].left.nblk, 1, XForcing.bndseg[s].left.blk_g);
			CopytoGPU(XForcing.bndseg[s].left.nblk, 1, XForcing.bndseg[s].left.blk, XForcing.bndseg[s].left.blk_g);

			AllocateGPU(XForcing.bndseg[s].right.nblk, 1, XForcing.bndseg[s].right.blk_g);
			CopytoGPU(XForcing.bndseg[s].right.nblk, 1, XForcing.bndseg[s].right.blk, XForcing.bndseg[s].right.blk_g);

			AllocateGPU(XForcing.bndseg[s].top.nblk, 1, XForcing.bndseg[s].top.blk_g);
			CopytoGPU(XForcing.bndseg[s].top.nblk, 1, XForcing.bndseg[s].top.blk, XForcing.bndseg[s].top.blk_g);

			AllocateGPU(XForcing.bndseg[s].bot.nblk, 1, XForcing.bndseg[s].bot.blk_g);
			CopytoGPU(XForcing.bndseg[s].bot.nblk, 1, XForcing.bndseg[s].bot.blk, XForcing.bndseg[s].bot.blk_g);

			AllocateGPU(XForcing.bndseg[s].left.nblk, XParam.blkwidth, XForcing.bndseg[s].left.qmean_g);
			CopytoGPU(XForcing.bndseg[s].left.nblk, XParam.blkwidth, XForcing.bndseg[s].left.qmean, XForcing.bndseg[s].left.qmean_g);

			AllocateGPU(XForcing.bndseg[s].right.nblk, XParam.blkwidth, XForcing.bndseg[s].right.qmean_g);
			CopytoGPU(XForcing.bndseg[s].right.nblk, XParam.blkwidth, XForcing.bndseg[s].right.qmean, XForcing.bndseg[s].right.qmean_g);

			AllocateGPU(XForcing.bndseg[s].top.nblk, XParam.blkwidth, XForcing.bndseg[s].top.qmean_g);
			CopytoGPU(XForcing.bndseg[s].top.nblk, XParam.blkwidth, XForcing.bndseg[s].top.qmean, XForcing.bndseg[s].top.qmean_g);

			AllocateGPU(XForcing.bndseg[s].bot.nblk, XParam.blkwidth, XForcing.bndseg[s].bot.qmean_g);
			CopytoGPU(XForcing.bndseg[s].bot.nblk, XParam.blkwidth, XForcing.bndseg[s].bot.qmean, XForcing.bndseg[s].bot.qmean_g);
		}


		// Also for mask
		XModel_g.blocks.mask.nblk = XModel.blocks.mask.nblk;
		AllocateGPU(XModel_g.blocks.mask.nblk, 1, XModel_g.blocks.mask.side);
		AllocateGPU(XModel_g.blocks.mask.nblk, 1, XModel_g.blocks.mask.blks);
		CopytoGPU(XModel_g.blocks.mask.nblk, 1, XModel.blocks.mask.side, XModel_g.blocks.mask.side);
		CopytoGPU(XModel_g.blocks.mask.nblk, 1, XModel.blocks.mask.blks, XModel_g.blocks.mask.blks);


		// things are quite different for Time Series output. Why is that?.
		if (XParam.TSnodesout.size() > 0)
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

			int nribmax = XModel.bndblk.Riverinfo.nribmax;
			int nburmax = XModel.bndblk.Riverinfo.nburmax;

			XModel_g.bndblk.Riverinfo.nribmax = nribmax;
			XModel_g.bndblk.Riverinfo.nburmax = nburmax;


			AllocateMappedMemGPU(XForcing.rivers.size(), 1,XParam.GPUDEVICE, XModel_g.bndblk.Riverinfo.qnow_g,XModel.bndblk.Riverinfo.qnow);
			XModel_g.bndblk.Riverinfo.qnow = XModel.bndblk.Riverinfo.qnow;


			AllocateGPU(nribmax, nburmax, XModel_g.bndblk.Riverinfo.Xbidir);
			AllocateGPU(nribmax, nburmax, XModel_g.bndblk.Riverinfo.Xridib);
			CopytoGPU(nribmax, nburmax, XModel.bndblk.Riverinfo.Xbidir, XModel_g.bndblk.Riverinfo.Xbidir);
			CopytoGPU(nribmax, nburmax, XModel.bndblk.Riverinfo.Xridib, XModel_g.bndblk.Riverinfo.Xridib);

			AllocateGPU(nribmax, nburmax, XModel_g.bndblk.Riverinfo.xstart);
			AllocateGPU(nribmax, nburmax, XModel_g.bndblk.Riverinfo.xend);
			AllocateGPU(nribmax, nburmax, XModel_g.bndblk.Riverinfo.ystart);
			AllocateGPU(nribmax, nburmax, XModel_g.bndblk.Riverinfo.yend);

			CopytoGPU(nribmax, nburmax, XModel.bndblk.Riverinfo.xstart, XModel_g.bndblk.Riverinfo.xstart);
			CopytoGPU(nribmax, nburmax, XModel.bndblk.Riverinfo.xend, XModel_g.bndblk.Riverinfo.xend);
			CopytoGPU(nribmax, nburmax, XModel.bndblk.Riverinfo.ystart, XModel_g.bndblk.Riverinfo.ystart);
			CopytoGPU(nribmax, nburmax, XModel.bndblk.Riverinfo.yend, XModel_g.bndblk.Riverinfo.yend);

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

		//InitzbgradientGPU(XParam, XModel_g);
		
		if (XForcing.culverts.size() > 0)
		{
			XModel_g.bndblk.nblkculvert = XModel.bndblk.nblkculvert;
			AllocateGPU(XModel.bndblk.nblkculvert, 1, XModel_g.bndblk.culvert);
			CopytoGPU(XModel.bndblk.nblkculvert, 1, XModel.bndblk.culvert, XModel_g.bndblk.culvert);

			AllocateGPU(XParam.nculverts, 1, XModel_g.culvertsF.dq);
			CopytoGPU(XParam.nculverts, 1, XModel.culvertsF.dq, XModel_g.culvertsF.dq);
			AllocateGPU(XParam.nculverts, 1, XModel_g.culvertsF.zs1);
			CopytoGPU(XParam.nculverts, 1, XModel.culvertsF.zs1, XModel_g.culvertsF.zs1);
			AllocateGPU(XParam.nculverts, 1, XModel_g.culvertsF.zs2);
			CopytoGPU(XParam.nculverts, 1, XModel.culvertsF.zs2, XModel_g.culvertsF.zs2);
			AllocateGPU(XParam.nculverts, 1, XModel_g.culvertsF.h1);
			CopytoGPU(XParam.nculverts, 1, XModel.culvertsF.h1, XModel_g.culvertsF.h1);
			AllocateGPU(XParam.nculverts, 1, XModel_g.culvertsF.h2);
			CopytoGPU(XParam.nculverts, 1, XModel.culvertsF.h2, XModel_g.culvertsF.h2);
			AllocateGPU(XParam.nculverts, 1, XModel_g.culvertsF.u1);
			CopytoGPU(XParam.nculverts, 1, XModel.culvertsF.u1, XModel_g.culvertsF.u1);
			AllocateGPU(XParam.nculverts, 1, XModel_g.culvertsF.u2);
			CopytoGPU(XParam.nculverts, 1, XModel.culvertsF.u2, XModel_g.culvertsF.u2);
			AllocateGPU(XParam.nculverts, 1, XModel_g.culvertsF.v1);
			CopytoGPU(XParam.nculverts, 1, XModel.culvertsF.v1, XModel_g.culvertsF.v1);
			AllocateGPU(XParam.nculverts, 1, XModel_g.culvertsF.v2);
			CopytoGPU(XParam.nculverts, 1, XModel.culvertsF.v2, XModel_g.culvertsF.v2);
			AllocateGPU(XParam.nculverts, 1, XModel_g.culvertsF.type);
			CopytoGPU(XParam.nculverts, 1, XModel.culvertsF.type, XModel_g.culvertsF.type);
			AllocateGPU(XParam.nculverts, 1, XModel_g.culvertsF.Qmax);
			CopytoGPU(XParam.nculverts, 1, XModel.culvertsF.Qmax, XModel_g.culvertsF.Qmax);
			AllocateGPU(XParam.nculverts, 1, XModel_g.culvertsF.dx1);
			CopytoGPU(XParam.nculverts, 1, XModel.culvertsF.dx1, XModel_g.culvertsF.dx1);
		}

	}
}
template void SetupGPU<float>(Param &XParam, Model<float> XModel, Forcing<float>& XForcing, Model<float>& XModel_g);
template void SetupGPU<double>(Param &XParam, Model<double> XModel, Forcing<float>& XForcing, Model<double>& XModel_g);


/**
 * @brief Check CUDA error status and print message if error occurs.
 *
 * Checks the CUDA error code and prints an error message if the code indicates failure.
 *
 * @param CUDerr CUDA error code
 */
void CUDA_CHECK(cudaError CUDerr)
{


	if (cudaSuccess != CUDerr) {

		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \

			__FILE__, __LINE__, cudaGetErrorString(CUDerr));

		exit(EXIT_FAILURE);

	}
}


/**
 * @brief Copy data from CPU to GPU memory.
 *
 * Copies an array from host (CPU) memory to device (GPU) memory using CUDA.
 *
 * @tparam T Data type
 * @param nblk Number of blocks
 * @param blksize Block size
 * @param z_cpu Source array (CPU)
 * @param z_gpu Destination array (GPU)
 */
template <class T> void CopytoGPU(int nblk, int blksize, T * z_cpu, T* z_gpu)
{
	CUDA_CHECK(cudaMemcpy(z_gpu, z_cpu, nblk * blksize * sizeof(T), cudaMemcpyHostToDevice));
}
template void CopytoGPU<bool>(int nblk, int blksize, bool* z_cpu, bool* z_gpu);
template void CopytoGPU<int>(int nblk, int blksize, int* z_cpu, int* z_gpu);
template void CopytoGPU<float>(int nblk, int blksize, float* z_cpu, float* z_gpu);
template void CopytoGPU<double>(int nblk, int blksize, double* z_cpu, double* z_gpu);

/**
 * @brief Copy data from GPU to CPU memory.
 *
 * Copies an array from device (GPU) memory to host (CPU) memory using CUDA.
 *
 * @tparam T Data type
 * @param nblk Number of blocks
 * @param blksize Block size
 * @param z_cpu Destination array (CPU)
 * @param z_gpu Source array (GPU)
 */
template <class T> void CopyGPUtoCPU(int nblk, int blksize, T* z_cpu, T* z_gpu)
{
	CUDA_CHECK(cudaMemcpy(z_cpu, z_gpu, nblk * blksize * sizeof(T), cudaMemcpyDeviceToHost));
}
template void CopyGPUtoCPU<bool>(int nblk, int blksize, bool* z_cpu, bool* z_gpu);
template void CopyGPUtoCPU<int>(int nblk, int blksize, int* z_cpu, int* z_gpu);
template void CopyGPUtoCPU<float>(int nblk, int blksize, float* z_cpu, float* z_gpu);
template void CopyGPUtoCPU<double>(int nblk, int blksize, double* z_cpu, double* z_gpu);

/**
 * @brief Copy complex data structures from CPU to GPU.
 *
 * This function copies the evolving variables structure from the host (CPU) to the device (GPU) memory.
 *
 * @tparam T Data type
 * @param nblk Number of blocks
 * @param blksize Block size
 * @param XEv_cpu Source evolving variables structure (CPU)
 * @param XEv_gpu Destination evolving variables structure (GPU)
 */
template <class T> void CopytoGPU(int nblk, int blksize, EvolvingP<T> XEv_cpu, EvolvingP<T> XEv_gpu)
{
	CopytoGPU(nblk, blksize, XEv_cpu.h, XEv_gpu.h);
	CopytoGPU(nblk, blksize, XEv_cpu.zs, XEv_gpu.zs);
	CopytoGPU(nblk, blksize, XEv_cpu.u, XEv_gpu.u);
	CopytoGPU(nblk, blksize, XEv_cpu.v, XEv_gpu.v);
}
template void CopytoGPU<float>(int nblk, int blksize, EvolvingP<float> XEv_cpu, EvolvingP<float> XEv_gpu);
template void CopytoGPU < double >(int nblk, int blksize, EvolvingP<double> XEv_cpu, EvolvingP<double> XEv_gpu);

/**
 * @brief Copy complex data structures from CPU to GPU.
 *
 * This function copies the evolving variables with momentum structure from the host (CPU) to the device (GPU) memory.
 *
 * @tparam T Data type
 * @param nblk Number of blocks
 * @param blksize Block size
 * @param XEv_cpu Source evolving variables with momentum structure (CPU)
 * @param XEv_gpu Destination evolving variables with momentum structure (GPU)
 */
template <class T> void CopytoGPU(int nblk, int blksize, EvolvingP_M<T> XEv_cpu, EvolvingP_M<T> XEv_gpu)
{
	CopytoGPU(nblk, blksize, XEv_cpu.h, XEv_gpu.h);
	CopytoGPU(nblk, blksize, XEv_cpu.zs, XEv_gpu.zs);
	CopytoGPU(nblk, blksize, XEv_cpu.u, XEv_gpu.u);
	CopytoGPU(nblk, blksize, XEv_cpu.v, XEv_gpu.v);
	CopytoGPU(nblk, blksize, XEv_cpu.U, XEv_gpu.U);
	CopytoGPU(nblk, blksize, XEv_cpu.hU, XEv_gpu.hU);
}
template void CopytoGPU<float>(int nblk, int blksize, EvolvingP_M<float> XEv_cpu, EvolvingP_M<float> XEv_gpu);
template void CopytoGPU < double >(int nblk, int blksize, EvolvingP_M<double> XEv_cpu, EvolvingP_M < double >  XEv_gpu);


/**
 * @brief Copy complex data structures from CPU to GPU.
 *
 * This function copies the gradients structure from the host (CPU) to the device (GPU) memory.
 *
 * @tparam T Data type
 * @param nblk Number of blocks
 * @param blksize Block size
 * @param XGrad_cpu Source gradients structure (CPU)
 * @param XGrad_gpu Destination gradients structure (GPU)
 */
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

/**
 * @brief Copy complex data structures from CPU to GPU.
 *
 * This function copies the entire model structure from the host (CPU) to the device (GPU) memory.
 *
 * @tparam T Data type
 * @param nblk Number of blocks
 * @param blksize Block size
 * @param XParam Simulation parameters
 * @param XModel_cpu Source model structure (CPU)
 * @param XModel_gpu Destination model structure (GPU)
 */
template <class T> void CopytoGPU(int nblk, int blksize, Param XParam, Model<T> XModel_cpu, Model<T> XModel_gpu)
{
	CopytoGPU(nblk, blksize, XModel_cpu.zb, XModel_gpu.zb);

	CopytoGPU(nblk, blksize, XModel_cpu.evolv, XModel_gpu.evolv);
	//CopytoGPU(nblk, blksize, XModel_cpu.evolv_o, XModel_gpu.evolv_o);

	CopytoGPU(nblk, blksize, XModel_cpu.evolv_o, XModel_gpu.evolv_o);

	CopytoGPU(nblk, blksize, XModel_cpu.cf, XModel_gpu.cf);

	CopytoGPU(nblk, blksize, XModel_cpu.grad.dzbdx, XModel_gpu.grad.dzbdx);
	CopytoGPU(nblk, blksize, XModel_cpu.grad.dzbdy, XModel_gpu.grad.dzbdy);
	

	//Block info
	CopytoGPU(nblk, 1, XModel_cpu.blocks.active, XModel_gpu.blocks.active);
	CopytoGPU(nblk, blksize, XModel_cpu.blocks.activeCell, XModel_gpu.blocks.activeCell);
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
	

	if (XParam.infiltration)
	{
		CopytoGPU(nblk, blksize, XModel_cpu.il, XModel_gpu.il);
		CopytoGPU(nblk, blksize, XModel_cpu.cl, XModel_gpu.cl);
		CopytoGPU(nblk, blksize, XModel_cpu.hgw, XModel_gpu.hgw);
	}

	if (XParam.outmax)
	{
		CopytoGPU(nblk, blksize, XModel_cpu.evmax, XModel_gpu.evmax);
	}
	if (XParam.outmean)
	{
		CopytoGPU(nblk, blksize, XModel_cpu.evmean, XModel_gpu.evmean);
	}
	if (XParam.outtwet)
	{
		CopytoGPU(nblk, blksize, XModel_cpu.wettime, XModel_gpu.wettime);
	}
	if (XParam.nculverts)
	{
		CopytoGPU(XParam.nculverts, 1, XModel_cpu.culvertsF.zs1, XModel_gpu.culvertsF.zs1);
		CopytoGPU(XParam.nculverts, 1, XModel_cpu.culvertsF.zs2, XModel_gpu.culvertsF.zs2);
		CopytoGPU(XParam.nculverts, 1, XModel_cpu.culvertsF.h1, XModel_gpu.culvertsF.h1);
		CopytoGPU(XParam.nculverts, 1, XModel_cpu.culvertsF.h2, XModel_gpu.culvertsF.h2);
		CopytoGPU(XParam.nculverts, 1, XModel_cpu.culvertsF.dq, XModel_gpu.culvertsF.dq);
	}
}
template void CopytoGPU<float>(int nblk, int blksize, Param XParam, Model<float> XModel_cpu, Model<float> XModel_gpu);
template void CopytoGPU<double>(int nblk, int blksize, Param XParam, Model<double> XModel_cpu, Model<double> XModel_gpu);


/**
 * @brief Copy the wave model structure from CPU to GPU.
 *
 * Wave equivalent of CopytoGPU(nblk, blksize, XParam, XModel_cpu, XModel_gpu).
 *
 * @tparam T Data type
 * @param nblk Number of blocks (allocated in memory, i.e. nblkmem)
 * @param blksize Block size
 * @param XWaveParam Wave model parameters
 * @param XWave_cpu Source wave model structure (CPU)
 * @param XWave_gpu Destination wave model structure (GPU)
 */
template <class T> void CopytoGPU(int nblk, int blksize, WaveParam XWaveParam, WaveModel<T> XWave_cpu, WaveModel<T> XWave_gpu)
{
	// Spectral grid
	CopytoGPU(XWaveParam.nfreq, 1, XWave_cpu.spec.sigma, XWave_gpu.spec.sigma);
	CopytoGPU(XWaveParam.nfreq, 1, XWave_cpu.spec.dsigma, XWave_gpu.spec.dsigma);
	CopytoGPU(XWaveParam.ndir, 1, XWave_cpu.spec.theta, XWave_gpu.spec.theta);
	CopytoGPU(XWaveParam.ndir, 1, XWave_cpu.spec.costh, XWave_gpu.spec.costh);
	CopytoGPU(XWaveParam.ndir, 1, XWave_cpu.spec.sinth, XWave_gpu.spec.sinth);

	// Spectral arrays
	CopytoGPU(nblk * XWaveParam.nspec, blksize, XWave_cpu.action.N, XWave_gpu.action.N);
	CopytoGPU(nblk * XWaveParam.nspec, blksize, XWave_cpu.action_o.N, XWave_gpu.action_o.N);
	CopytoGPU(nblk * XWaveParam.nspec, blksize, XWave_cpu.adv.dN, XWave_gpu.adv.dN);

	// Bulk parameters
	CopytoGPU(nblk, blksize, XWave_cpu.bulk.Hs, XWave_gpu.bulk.Hs);
	CopytoGPU(nblk, blksize, XWave_cpu.bulk.Tm01, XWave_gpu.bulk.Tm01);
	CopytoGPU(nblk, blksize, XWave_cpu.bulk.Tm02, XWave_gpu.bulk.Tm02);
	CopytoGPU(nblk, blksize, XWave_cpu.bulk.Tp, XWave_gpu.bulk.Tp);
	CopytoGPU(nblk, blksize, XWave_cpu.bulk.Dir, XWave_gpu.bulk.Dir);
	CopytoGPU(nblk, blksize, XWave_cpu.bulk.Dspr, XWave_gpu.bulk.Dspr);
	CopytoGPU(nblk, blksize, XWave_cpu.bulk.km, XWave_gpu.bulk.km);
	CopytoGPU(nblk, blksize, XWave_cpu.bulk.Urms, XWave_gpu.bulk.Urms);
	CopytoGPU(nblk, blksize, XWave_cpu.bulk.Qb, XWave_gpu.bulk.Qb);

	// Dissipation
	CopytoGPU(nblk, blksize, XWave_cpu.dissip.Dbr, XWave_gpu.dissip.Dbr);
	CopytoGPU(nblk, blksize, XWave_cpu.dissip.Dfr, XWave_gpu.dissip.Dfr);
	CopytoGPU(nblk, blksize, XWave_cpu.dissip.Dwc, XWave_gpu.dissip.Dwc);

	// Coupling arrays
	CopytoGPU(nblk, blksize, XWave_cpu.force.Sxx, XWave_gpu.force.Sxx);
	CopytoGPU(nblk, blksize, XWave_cpu.force.Sxy, XWave_gpu.force.Sxy);
	CopytoGPU(nblk, blksize, XWave_cpu.force.Syy, XWave_gpu.force.Syy);
	CopytoGPU(nblk, blksize, XWave_cpu.force.Fx, XWave_gpu.force.Fx);
	CopytoGPU(nblk, blksize, XWave_cpu.force.Fy, XWave_gpu.force.Fy);

	// Medium (the kinematics and work arrays are recalculated on the GPU)
	CopytoGPU(nblk, blksize, XWave_cpu.medium.h, XWave_gpu.medium.h);
	CopytoGPU(nblk, blksize, XWave_cpu.medium.ho, XWave_gpu.medium.ho);
	CopytoGPU(nblk, blksize, XWave_cpu.medium.dhdt, XWave_gpu.medium.dhdt);
	CopytoGPU(nblk, blksize, XWave_cpu.medium.u, XWave_gpu.medium.u);
	CopytoGPU(nblk, blksize, XWave_cpu.medium.v, XWave_gpu.medium.v);
}
template void CopytoGPU<float>(int nblk, int blksize, WaveParam XWaveParam, WaveModel<float> XWave_cpu, WaveModel<float> XWave_gpu);
template void CopytoGPU<double>(int nblk, int blksize, WaveParam XWaveParam, WaveModel<double> XWave_cpu, WaveModel<double> XWave_gpu);


/**
 * @brief Allocate and bind a CUDA texture object.
 *
 * Allocates a CUDA array and creates a texture object for use in GPU kernels.
 *
 * @param nx Number of x grid points
 * @param ny Number of y grid points
 * @param Tex Texture set structure
 * @param input Input data array
 */
void AllocateTEX(int nx, int ny, TexSetP& Tex, float* input)
{


	CUDA_CHECK(cudaMallocArray(&Tex.CudArr, &Tex.channelDesc, nx, ny));
	CUDA_CHECK(cudaMemcpyToArray(Tex.CudArr, 0, 0, input, nx * ny * sizeof(float), cudaMemcpyHostToDevice));


	memset(&Tex.texDesc, 0, sizeof(cudaTextureDesc));
	Tex.texDesc.addressMode[0] = cudaAddressModeClamp;
	Tex.texDesc.addressMode[1] = cudaAddressModeClamp;
	Tex.texDesc.filterMode = cudaFilterModeLinear;
	//Tex.texDesc.filterMode = cudaFilterModePoint;
	Tex.texDesc.normalizedCoords = false;

	memset(&Tex.resDesc, 0, sizeof(cudaResourceDesc));

	Tex.resDesc.resType = cudaResourceTypeArray;
	Tex.resDesc.res.array.array = Tex.CudArr;

	CUDA_CHECK(cudaCreateTextureObject(&Tex.tex, &Tex.resDesc, &Tex.texDesc, NULL));
	//CUDA_CHECK(cudaBindTextureToArray(Tex, zca, cCFD));


}


/**
 * @brief Allocate boundary texture for GPU.
 *
 * Allocates and binds boundary water level data as a CUDA texture for GPU use.
 *
 * @param side Boundary parameter structure
 */
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




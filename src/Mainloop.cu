#include "Mainloop.h"



template <class T> void MainLoopCPU(Param XParam, Forcing<float> XForcing, Model<T> XModel)
{
	//Define some useful variables 
	BlockP<T> XBlock = XModel.blocks;
	Loop<T> XLoop = InitLoop(XParam,XModel);

	while (XLoop.totaltime < XParam.endtime)
	{

	}

}

 
template <class T> Loop<T> InitLoop(Param XParam, Model<T> XModel)
{
	Loop<T> XLoop;
	XLoop.atmpuni = XParam.Paref;
	XLoop.totaltime = XParam.totaltime;
	XLoop.nextoutputtime = XParam.totaltime + XParam.outputtimestep;
	
	// Prepare output files
	InitSave2Netcdf(XParam, XModel);
	InitTSOutput(XParam);

	// GPU stuff
	if (XParam.GPUDEVICE >= 0)
	{
		XLoop.blockDim = make_int3(16, 16, 1);
		XLoop.gridDim= make_int3(XParam.nblk, 1, 1);
	}

	//Reset mean
	if (XParam.GPUDEVICE >= 0)
	{
		//
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, T(0.0), XModel.evmean.h);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, T(0.0), XModel.evmean.zs);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, T(0.0), XModel.evmean.u);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, T(0.0), XModel.evmean.v);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	else
	{
		InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.evmean.h);
		InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.evmean.zs);
		InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.evmean.u);
		InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.evmean.v);
	}

	//Reset Max
	if (XParam.GPUDEVICE >= 0)
	{
		//
		reset_var <<<XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, std::numeric_limits<T>::min(), XModel.evmax.h);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, std::numeric_limits<T>::min(), XModel.evmax.zs);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, std::numeric_limits<T>::min(), XModel.evmax.u);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, std::numeric_limits<T>::min(), XModel.evmax.v);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	else
	{
		InitArrayBUQ(XParam, XModel.blocks, std::numeric_limits<T>::min(), XModel.evmax.h);
		InitArrayBUQ(XParam, XModel.blocks, std::numeric_limits<T>::min(), XModel.evmax.zs);
		InitArrayBUQ(XParam, XModel.blocks, std::numeric_limits<T>::min(), XModel.evmax.u);
		InitArrayBUQ(XParam, XModel.blocks, std::numeric_limits<T>::min(), XModel.evmax.v);
	}

	return XLoop;

}



template <class T>
__global__ void reset_var(unsigned int halowidth, unsigned int * active, T resetval, T* Var)
{
	
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = active[ibl];

	int n = memloc(halowidth, blkmemwidth, blksize, ix, iy, ib);

	Var[n] = resetval;
}
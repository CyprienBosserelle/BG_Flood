#include "Mainloop.h"



template <class T> void MainLoop(Param XParam, Forcing<float> XForcing, Model<T> XModel)
{
	//Define some useful variables 
	BlockP<T> XBlock = XModel.blocks;
	Loop<T> XLoop = InitLoop(XParam,XModel);

	//while (XLoop.totaltime < XParam.endtime)
	{
		//
	}

}
template void MainLoop<float>(Param XParam, Forcing<float> XForcing, Model<float> XModel);
template void MainLoop<double>(Param XParam, Forcing<float> XForcing, Model<double> XModel);

 
template <class T> Loop<T> InitLoop(Param &XParam, Model<T> &XModel)
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
		XLoop.blockDim = (16, 16, 1);
		XLoop.gridDim = (XParam.nblk, 1, 1);
	}

	//Reset mean
	resetmean(XParam, XLoop, XModel.blocks, XModel.evmean);

	//Reset Max
	resetmax(XParam, XLoop, XModel.blocks, XModel.evmax);

	return XLoop;

}


template <class T> void resetmax(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, EvolvingP<T>& XEv)
{
	if (XParam.GPUDEVICE >= 0)
	{
		//
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XBlock.active, std::numeric_limits<T>::min(), XEv.h);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XBlock.active, std::numeric_limits<T>::min(), XEv.zs);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XBlock.active, std::numeric_limits<T>::min(), XEv.u);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XBlock.active, std::numeric_limits<T>::min(), XEv.v);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	else
	{
		InitArrayBUQ(XParam, XBlock, std::numeric_limits<T>::min(), XEv.h);
		InitArrayBUQ(XParam, XBlock, std::numeric_limits<T>::min(), XEv.zs);
		InitArrayBUQ(XParam, XBlock, std::numeric_limits<T>::min(), XEv.u);
		InitArrayBUQ(XParam, XBlock, std::numeric_limits<T>::min(), XEv.v);
	}
}


template <class T> void resetmean(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, EvolvingP<T> & XEv)
{
	if (XParam.GPUDEVICE >= 0)
	{
		//
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.h);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.zs);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.u);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.v);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	else
	{
		InitArrayBUQ(XParam, XBlock, T(0.0), XEv.h);
		InitArrayBUQ(XParam, XBlock, T(0.0), XEv.zs);
		InitArrayBUQ(XParam, XBlock, T(0.0), XEv.u);
		InitArrayBUQ(XParam, XBlock, T(0.0), XEv.v);
	}
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
template __global__ void reset_var<float>(unsigned int halowidth, unsigned int* active, float resetval, float* Var);
template __global__ void reset_var<double>(unsigned int halowidth, unsigned int* active, double resetval, double* Var);

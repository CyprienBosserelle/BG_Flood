#include "Mainloop.h"



template <class T> void MainLoop(Param &XParam, Forcing<float> XForcing, Model<T>& XModel, Model<T> &XModel_g)
{
	if (XParam.GPUDEVICE < 0)
	{
		MainLoop(XParam, XForcing, XModel);
	}
	else
	{
		MainLoop(XParam, XForcing, XModel_g);
	}

	

}
template void MainLoop<float>(Param& XParam, Forcing<float> XForcing, Model<float>& XModel, Model<float>& XModel_g);
template void MainLoop<double>(Param& XParam, Forcing<float> XForcing, Model<double>& XModel, Model<double>& XModel_g);




template <class T> void MainLoop(Param& XParam, Forcing<float> XForcing, Model<T>& XModel)
{
	//Define some useful variables 
	BlockP<T> XBlock = XModel.blocks;
	Loop<T> XLoop = InitLoop(XParam, XModel);

	while (XLoop.totaltime < XParam.endtime)
	{
		// Bnd stuff here

		// Forcing

		// Core engine

		// River 

		// Time keeping

		// Do Sum & Max variables Here

		// Check for TSoutput

		// Check for map output

	}



}
template void MainLoop<float>(Param &XParam, Forcing<float> XForcing, Model<float> &XModel);
template void MainLoop<double>(Param &XParam, Forcing<float> XForcing, Model<double> &XModel);

template <class T> void MainLoopGPU(Param& XParam, Forcing<float> XForcing, Model<T>& XModel)
{
	//Define some useful variables 
	BlockP<T> XBlock = XModel.blocks;
	Loop<T> XLoop = InitLoop(XParam, XModel);

	while (XLoop.totaltime < XParam.endtime)
	{
		// Bnd stuff here

		// Forcing

		// Core engine

		// River 

		// Time keeping

		// Do Sum & Max variables Here

		// Check for TSoutput

		// Check for map output

	}



}
template void MainLoopGPU<float>(Param& XParam, Forcing<float> XForcing, Model<float>& XModel);
template void MainLoopGPU<double>(Param& XParam, Forcing<float> XForcing, Model<double>& XModel);

 
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
	if (XParam.outmean)
	{
		resetmean(XParam, XLoop, XModel.blocks, XModel.evmean);
	}
	
	

	//Reset Max
	if (XParam.outmax)
	{
		resetmax(XParam, XLoop, XModel.blocks, XModel.evmax);
	}

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
template void resetmean<float>(Param XParam, Loop<float> XLoop, BlockP<float> XBlock, EvolvingP<float>& XEv);
template void resetmean<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, EvolvingP<double>& XEv);



template <class T> __global__ void reset_var(int halowidth, int * active, T resetval, T* Var)
{
	
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = active[ibl];

	int n = memloc(halowidth, blkmemwidth,  ix, iy, ib);
	//int n= (ix + halowidth) + (iy + halowidth) * blkmemwidth + ib * blksize;
	Var[n] = resetval;
}
template __global__ void reset_var<float>(int halowidth, int* active, float resetval, float* Var);
template __global__ void reset_var<double>(int halowidth, int* active, double resetval, double* Var);


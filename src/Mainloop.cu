#include "Mainloop.h"



template <class T> void MainLoop(Param &XParam, Forcing<float> XForcing, Model<T>& XModel, Model<T> &XModel_g)
{
	
	Loop<T> XLoop = InitLoop(XParam, XModel);

	//Define some useful variables 
	if (XParam.outmean)
	{
		resetmean(XParam, XLoop, XModel.blocks, XModel.evmean);
	}

	//Reset Max
	if (XParam.outmax)
	{
		resetmax(XParam, XLoop, XModel.blocks, XModel.evmax);
	}

	//while (XLoop.totaltime < XParam.endtime)
	{
		// Bnd stuff here
		Flowbnd(XParam, XLoop, XForcing.left, -1, 0, XModel.evolv, XModel.zb);
		Flowbnd(XParam, XLoop, XForcing.right, 1, 0, XModel.evolv, XModel.zb);
		Flowbnd(XParam, XLoop, XForcing.top, 0, 1, XModel.evolv, XModel.zb);
		Flowbnd(XParam, XLoop, XForcing.bot, 0, -1, XModel.evolv, XModel.zb);

		// Forcing
		if (XParam.GPUDEVICE >= 0)
		{
			FlowGPU(XParam, XLoop, XModel_g);
		}
		else
		{
			FlowCPU(XParam, XLoop, XModel);
		}
		

		// Core engine


		// River 

		// Time keeping

		// Do Sum & Max variables Here

		// Check for TSoutput

		// Check for map output

	}
	

	

}
template void MainLoop<float>(Param& XParam, Forcing<float> XForcing, Model<float>& XModel, Model<float>& XModel_g);
template void MainLoop<double>(Param& XParam, Forcing<float> XForcing, Model<double>& XModel, Model<double>& XModel_g);




 
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

	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();

	return XLoop;

}


template <class T> void resetmax(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, EvolvingP<T>& XEv)
{
	if (XParam.GPUDEVICE >= 0)
	{
		//
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.h);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.zs);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.u);
		reset_var << <XLoop.gridDim, XLoop.blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.v);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	else
	{
		InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.h);
		InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.zs);
		InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.u);
		InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.v);
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



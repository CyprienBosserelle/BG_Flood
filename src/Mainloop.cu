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


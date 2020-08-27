#include "FlowCPU.h"


template <class T> void FlowCPU(Param XParam, Loop<T>& XLoop, Model<T> XModel)
{
	
	//============================================
	//  
	fillHalo(XParam, XModel.blocks, XModel.evolv);

	//============================================
	// Reset DTmax
	InitArrayBUQ(XParam, XModel.blocks, XLoop.hugeposval, XModel.time.dtmax);
	

	//============================================
	// Calculate gradient for evolving parameters
	gradientCPU(XParam, XLoop, XModel.blocks, XModel.evolv, XModel.grad);


	printf("%f\n", XModel.grad.dhdx[memloc(XParam, 15, 0, 22)]);


	//============================================
	// First step in reimann solver
	updateKurgXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax);
	updateKurgYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax);

	//fillHalo(XParam, XModel.blocks, XModel.flux);
	
	XLoop.dt = double(CalctimestepCPU(XParam, XModel.blocks, XModel.time));
		
	if (ceil((XLoop.nextoutputtime - XLoop.totaltime) / XLoop.dt) > 0.0)
	{
		XLoop.dt = (XLoop.nextoutputtime - XLoop.totaltime) / ceil((XLoop.nextoutputtime - XLoop.totaltime) / XLoop.dt);
	}

	XModel.time.dt = T(XLoop.dt);

	updateEVCPU(XParam, XModel.blocks, XModel.evolv, XModel.flux, XModel.adv);

	AdvkernelCPU(XParam, XModel.blocks, XModel.time.dt * T(0.5), XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o);

	// Corrector step
	/*
	gradientCPU(XParam, XLoop, XModel.blocks, XModel.evolv_o, XModel.grad);

	updateKurgXCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax);
	updateKurgYCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax);

	//fillHalo(XParam, XModel.blocks, XModel.flux);

	updateEVCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.flux, XModel.adv);

	AdvkernelCPU(XParam, XModel.blocks, XModel.time.dt, XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o);
	*/

	cleanupCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.evolv);






}
template void FlowCPU<float>(Param XParam, Loop<float>& XLoop, Model<float> XModel);
template void FlowCPU<double>(Param XParam, Loop<double>& XLoop, Model<double> XModel);



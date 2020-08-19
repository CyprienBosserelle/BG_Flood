#include "FlowCPU.h"


template <class T> void FlowCPU(Param XParam, Loop<T>& XLoop, Model<T> XModel)
{
	


	//============================================
	// Reset DTmax
	InitArrayBUQ(XParam, XModel.blocks, XLoop.hugeposval, XModel.time.dtmax);
	

	//============================================
	// Calculate gradient for evolving parameters
	gradientCPU(XParam, XLoop, XModel.blocks, XModel.evolv, XModel.grad);

	//============================================
	// First step in reimann solver
	updateKurgXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax);










}
template void FlowCPU<float>(Param XParam, Loop<float>& XLoop, Model<float> XModel);
template void FlowCPU<double>(Param XParam, Loop<double>& XLoop, Model<double> XModel);



#include "FlowCPU.h"


template <class T> void FlowCPU(Param XParam, Loop<T>& XLoop, Model<T> XModel)
{
	//============================================
	// Predictor step in reimann solver
	//============================================

	//============================================
	//  Fill the halo for gradient reconstruction
	fillHalo(XParam, XModel.blocks, XModel.evolv);

	//============================================
	// Reset DTmax
	InitArrayBUQ(XParam, XModel.blocks, XLoop.hugeposval, XModel.time.dtmax);
	
	//============================================
	// Calculate gradient for evolving parameters
	gradientCPU(XParam, XLoop, XModel.blocks, XModel.evolv, XModel.grad);

	//============================================
	// Flux and Source term reconstruction
	// X- direction
	updateKurgXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);

	// Y- direction
	updateKurgYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);

	//============================================
	// Fill Halo for flux from fine to coarse
	fillHalo(XParam, XModel.blocks, XModel.flux);
	
	//============================================
	// Reduce minimum timestep
	XLoop.dt = double(CalctimestepCPU(XParam,XLoop, XModel.blocks, XModel.time));
	XModel.time.dt = T(XLoop.dt);

	//============================================
	// Update advection terms (dh dhu dhv) 
	updateEVCPU(XParam, XModel.blocks, XModel.evolv, XModel.flux, XModel.adv);

	//============================================
	//Update evolving variable by 1/2 time step
	AdvkernelCPU(XParam, XModel.blocks, XModel.time.dt * T(0.5), XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o);

	//============================================
	// Corrector step in reimann solver
	//============================================

	//============================================
	//  Fill the halo for gradient reconstruction
	fillHalo(XParam, XModel.blocks, XModel.evolv_o);

	//============================================
	// Calculate gradient for evolving parameters
	gradientCPU(XParam, XLoop, XModel.blocks, XModel.evolv_o, XModel.grad);

	//============================================
	// Flux and Source term reconstruction
	// X- direction
	updateKurgXCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceXCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.zb);
	
	// Y- direction
	updateKurgYCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceYCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.zb);

	//============================================
	// Fill Halo for flux from fine to coarse
	fillHalo(XParam, XModel.blocks, XModel.flux);

	//============================================
	// Update advection terms (dh dhu dhv) 
	updateEVCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.flux, XModel.adv);
	
	//============================================
	//Update evolving variable by 1 full time step
	AdvkernelCPU(XParam, XModel.blocks, XModel.time.dt, XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o);
	
	//============================================
	// Add bottom friction
	bottomfrictionCPU(XParam, XModel.blocks, XModel.time.dt, XModel.cf, XModel.evolv_o);

	//============================================
	//Copy updated evolving variable back
	cleanupCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.evolv);

	




}
template void FlowCPU<float>(Param XParam, Loop<float>& XLoop, Model<float> XModel);
template void FlowCPU<double>(Param XParam, Loop<double>& XLoop, Model<double> XModel);



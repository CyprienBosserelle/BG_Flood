#include "FlowCPU.h"


template <class T> void FlowCPU(Param XParam, Loop<T>& XLoop,Forcing<float> XForcing, Model<T> XModel)
{
	//============================================
	// Predictor step in reimann solver
	//============================================

	//============================================
	//  Fill the halo for gradient reconstruction
	fillHalo(XParam, XModel.blocks, XModel.evolv, XModel.zb);

	//============================================
	// Reset DTmax
	InitArrayBUQ(XParam, XModel.blocks, XLoop.hugeposval, XModel.time.dtmax);
	
	//============================================
	// Calculate gradient for evolving parameters
	gradientCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.zb);

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
	//fillHalo(XParam, XModel.blocks, XModel.flux);
	
	//============================================
	// Reduce minimum timestep
	XLoop.dt = double(CalctimestepCPU(XParam,XLoop, XModel.blocks, XModel.time));
	XLoop.dtmax = XLoop.dt;
	XModel.time.dt = T(XLoop.dt);

	//============================================
	// Update advection terms (dh dhu dhv) 
	updateEVCPU(XParam, XModel.blocks, XModel.evolv, XModel.flux, XModel.adv);

	//============================================
	// Add forcing (Rain, Wind)
	if (!XForcing.Rain.inputfile.empty())
	{
		AddrainforcingCPU(XParam, XModel.blocks, XForcing.Rain, XModel.adv);
	}
	if (!XForcing.UWind.inputfile.empty())//&& !XForcing.UWind.inputfile.empty()
	{
		AddwindforcingCPU(XParam, XModel.blocks, XForcing.UWind, XForcing.VWind, XModel.adv);
	}
	if (XForcing.rivers.size() > 0)
	{
		AddRiverForcing(XParam, XLoop, XForcing.rivers, XModel);
	}

	//============================================
	//Update evolving variable by 1/2 time step
	AdvkernelCPU(XParam, XModel.blocks, XModel.time.dt * T(0.5), XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o);
	
	
	//============================================
	// Corrector step in reimann solver
	//============================================

	//============================================
	//  Fill the halo for gradient reconstruction
	fillHalo(XParam, XModel.blocks, XModel.evolv_o,XModel.zb);

	//============================================
	// Calculate gradient for evolving parameters
	gradientCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.zb);

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
	//fillHalo(XParam, XModel.blocks, XModel.flux);

	//============================================
	// Update advection terms (dh dhu dhv) 
	updateEVCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.flux, XModel.adv);
	
	//============================================
	// Add forcing (Rain, Wind)
	if (!XForcing.Rain.inputfile.empty())
	{
		AddrainforcingCPU(XParam, XModel.blocks, XForcing.Rain, XModel.adv);
	}
	if (!XForcing.UWind.inputfile.empty())//&& !XForcing.UWind.inputfile.empty()
	{
		AddwindforcingCPU(XParam, XModel.blocks, XForcing.UWind, XForcing.VWind, XModel.adv);
	}
	if (XForcing.rivers.size() > 0)
	{
		AddRiverForcing(XParam, XLoop, XForcing.rivers, XModel);
	}

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
template void FlowCPU<float>(Param XParam, Loop<float>& XLoop, Forcing<float> XForcing, Model<float> XModel);
template void FlowCPU<double>(Param XParam, Loop<double>& XLoop, Forcing<float> XForcing, Model<double> XModel);



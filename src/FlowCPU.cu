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
	if (XParam.engine == 1)
	{
		// X- direction
		UpdateButtingerXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);

		// Y- direction
		UpdateButtingerYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	}
	else if (XParam.engine == 2)
	{
		// X- direction
		updateKurgXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
		//AddSlopeSourceXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);

		// Y- direction
		updateKurgYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
		//AddSlopeSourceYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);
	}
	else if (XParam.engine == 3)
	{
		// X- direction
		updateKurgXATMCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, XModel.Patm, XModel.datmpdx);
		//AddSlopeSourceXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);

		// Y- direction

		updateKurgYATMCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, XModel.Patm, XModel.datmpdy);
		//AddSlopeSourceYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);
	}
	

	//============================================
	// Fill Halo for flux from fine to coarse
	fillHalo(XParam, XModel.blocks, XModel.flux);
	
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
	//if (!XForcing.Rain.inputfile.empty())
	//{
	//	AddrainforcingCPU(XParam, XModel.blocks, XForcing.Rain, XModel.adv);
	//}
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
	if (XParam.engine == 1)
	{

		// X- direction
		UpdateButtingerXCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
		//updateKurgXCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
		//AddSlopeSourceXCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.zb);

		// Y- direction
		UpdateButtingerYCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
		//updateKurgYCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
		//AddSlopeSourceYCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.zb);
	}
	else if (XParam.engine == 2)
	{
		// X- direction
		updateKurgXCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
		//AddSlopeSourceXCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.zb);

		// Y- direction
		updateKurgYCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
		//AddSlopeSourceYCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.zb);
	}
	else if (XParam.engine == 3)
	{
		// X- direction
		//UpdateButtingerXCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
		updateKurgXATMCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, XModel.Patm, XModel.datmpdx);
		//AddSlopeSourceXCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.zb);

		// Y- direction
		//UpdateButtingerYCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
		updateKurgYATMCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, XModel.Patm, XModel.datmpdy);
		//AddSlopeSourceYCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.zb);
	}

	//============================================
	// Fill Halo for flux from fine to coarse
	fillHalo(XParam, XModel.blocks, XModel.flux);

	//============================================
	// Update advection terms (dh dhu dhv) 
	updateEVCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.flux, XModel.adv);
	
	//============================================
	// Add forcing (Rain, Wind)
	//if (!XForcing.Rain.inputfile.empty())
	//{
	//	AddrainforcingCPU(XParam, XModel.blocks, XForcing.Rain, XModel.adv);
	//}
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
	//XiafrictionCPU(XParam, XModel.blocks, XModel.time.dt, XModel.cf, XModel.evolv, XModel.evolv_o);


	//============================================
	//Copy updated evolving variable back
	cleanupCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.evolv);

	if (!XForcing.Rain.inputfile.empty())
	{
		AddrainforcingImplicitCPU(XParam, XLoop, XModel.blocks, XForcing.Rain, XModel.evolv);
	}

	if (XParam.VelThreshold > 0.0)
	{
		TheresholdVelCPU(XParam, XModel.blocks, XModel.evolv);
		
	}


}
template void FlowCPU<float>(Param XParam, Loop<float>& XLoop, Forcing<float> XForcing, Model<float> XModel);
template void FlowCPU<double>(Param XParam, Loop<double>& XLoop, Forcing<float> XForcing, Model<double> XModel);




/*! \fn  void HalfStepCPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel)
* Debugging flow step
* This function was crated to debug the main engine of the model
*/
template <class T> void HalfStepCPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel)
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
	UpdateButtingerXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//updateKurgXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);

	// Y- direction
	UpdateButtingerYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//updateKurgYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);

	//============================================
	// Fill Halo for flux from fine to coarse
	fillHalo(XParam, XModel.blocks, XModel.flux);

	//============================================
	// Reduce minimum timestep
	// Make only a half max step
	//XLoop.dt = double(CalctimestepCPU(XParam, XLoop, XModel.blocks, XModel.time)) * T(0.5);
	XLoop.dt = double(timestepreductionCPU(XParam, XLoop, XModel.blocks, XModel.time)) * T(0.5);
	XLoop.dtmax = XLoop.dt;
	XModel.time.dt = T(XLoop.dt);

	//============================================
	// Update advection terms (dh dhu dhv) 
	updateEVCPU(XParam, XModel.blocks, XModel.evolv, XModel.flux, XModel.adv);

	//============================================
	// Add forcing (Rain, Wind)
	//if (!XForcing.Rain.inputfile.empty())
	//{
	//	AddrainforcingCPU(XParam, XModel.blocks, XForcing.Rain, XModel.adv);
	//}
	if (!XForcing.UWind.inputfile.empty())//&& !XForcing.UWind.inputfile.empty()
	{
		AddwindforcingCPU(XParam, XModel.blocks, XForcing.UWind, XForcing.VWind, XModel.adv);
	}
	if (XForcing.rivers.size() > 0)
	{
		AddRiverForcing(XParam, XLoop, XForcing.rivers, XModel);
	}

	//============================================
	//Update evolving variable by 1 time step
	AdvkernelCPU(XParam, XModel.blocks, XModel.time.dt , XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o);


	//============================================
	// Add bottom friction
	bottomfrictionCPU(XParam, XModel.blocks, XModel.time.dt, XModel.cf, XModel.evolv_o);
	//XiafrictionCPU(XParam, XModel.blocks, XModel.time.dt, XModel.cf, XModel.evolv, XModel.evolv_o);

	//============================================
	//Copy updated evolving variable back
	cleanupCPU(XParam, XModel.blocks, XModel.evolv_o, XModel.evolv);

	if (!XForcing.Rain.inputfile.empty())
	{
		AddrainforcingImplicitCPU(XParam, XLoop, XModel.blocks, XForcing.Rain, XModel.evolv);
	}
}
template void HalfStepCPU<float>(Param XParam, Loop<float>& XLoop, Forcing<float> XForcing, Model<float> XModel);
template void HalfStepCPU<double>(Param XParam, Loop<double>& XLoop, Forcing<float> XForcing, Model<double> XModel);


#include "FlowCPU.h"


template <class T> void FlowCPU(Param XParam, Loop<T>& XLoop,Forcing<float> XForcing, Model<T> XModel)
{
	// Determine local block range for this MPI process
	int nblk_local_start, nblk_local_end;
	int nblk_per_process = XParam.nblk / XParam.size;
	int remainder_blks = XParam.nblk % XParam.size;

	if (XParam.rank < remainder_blks) {
		nblk_local_start = XParam.rank * (nblk_per_process + 1);
		nblk_local_end = nblk_local_start + nblk_per_process + 1;
	} else {
		nblk_local_start = XParam.rank * nblk_per_process + remainder_blks;
		nblk_local_end = nblk_local_start + nblk_per_process;
	}

	// Create a new Param object with local block counts for functions that iterate internally
	Param XParam_local = XParam;
	XParam_local.nblk = nblk_local_end - nblk_local_start;


	//============================================
	// Predictor step in reimann solver
	//============================================


	if (XParam.atmpforcing)
	{
		//Update atm press forcing
		AddPatmforcingCPU(XParam_local, XModel.blocks, XForcing.Atmp, XModel, nblk_local_start);
		
		//Fill atmp halo
		fillHaloC(XParam_local, XModel.blocks, XModel.Patm, nblk_local_start);
			

		//Calc dpdx and dpdy
		gradientC(XParam_local, XModel.blocks, XModel.Patm, XModel.datmpdx, XModel.datmpdy, nblk_local_start);
		gradientHalo(XParam_local, XModel.blocks, XModel.Patm, XModel.datmpdx, XModel.datmpdy, nblk_local_start);

		refine_linear(XParam_local, XModel.blocks, XModel.Patm, XModel.datmpdx, XModel.datmpdy, nblk_local_start);
		gradientHalo(XParam_local, XModel.blocks, XModel.Patm, XModel.datmpdx, XModel.datmpdy, nblk_local_start);
	}

	//============================================
	//  Fill the halo for gradient reconstruction
	//  This needs to be done across all processes before local computations
	fillHalo(XParam, XModel.blocks, XModel.evolv, XModel.zb); // Potentially MPI-aware version needed

	//============================================
	// Reset DTmax
	InitArrayBUQ(XParam_local, XModel.blocks, XLoop.hugeposval, XModel.time.dtmax, nblk_local_start);

	//============================================
	// Calculate gradient for evolving parameters
	gradientCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.zb, nblk_local_start);

	//============================================
	// Flux and Source term reconstruction
	if (XParam.engine == 1)
	{
		UpdateButtingerXCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, nblk_local_start);
		UpdateButtingerYCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, nblk_local_start);
	}
	else if (XParam.engine == 2)
	{
		updateKurgXCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, nblk_local_start);
		updateKurgYCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, nblk_local_start);
	}
	else if (XParam.engine == 3)
	{
		updateKurgXATMCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, XModel.Patm, XModel.datmpdx, nblk_local_start);
		updateKurgYATMCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, XModel.Patm, XModel.datmpdy, nblk_local_start);
	}
	
	//============================================
	// Fill Halo for flux from fine to coarse
	// This needs to be done across all processes
	fillHalo(XParam, XModel.blocks, XModel.flux); // Potentially MPI-aware version needed
	for (int iseg = 0; iseg < XForcing.bndseg.size(); iseg++)
	{
		// This function might need MPI awareness if bndseg spans across processes
		FlowbndFlux(XParam, XLoop.totaltime + XLoop.dt * 0.5, XModel.blocks, XForcing.bndseg[iseg], XForcing.Atmp, XModel.evolv, XModel.flux);
	}
	
	//============================================
	// Reduce minimum timestep - This requires global reduction (MPI_Allreduce)
	T local_dt_min = CalctimestepCPU(XParam_local, XLoop, XModel.blocks, XModel.time, nblk_local_start);
#ifdef USE_MPI
	MPI_Allreduce(&local_dt_min, &XLoop.dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); // Assuming T can be MPI_DOUBLE or MPI_FLOAT
#else
	XLoop.dt = local_dt_min;
#endif
	XLoop.dtmax = XLoop.dt;
	XModel.time.dt = T(XLoop.dt);

	//============================================
	// Update advection terms (dh dhu dhv) 
	updateEVCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.flux, XModel.adv, nblk_local_start);

	//============================================
	// Add forcing (Rain, Wind)
	if (!XForcing.UWind.inputfile.empty())
	{
		AddwindforcingCPU(XParam_local, XModel.blocks, XForcing.UWind, XForcing.VWind, XModel.adv, nblk_local_start);
	}
	if (XForcing.rivers.size() > 0)
	{
		// This function likely needs MPI awareness if rivers span across processes or if forcing is global
		AddRiverForcing(XParam, XLoop, XForcing.rivers, XModel);
	}

	//============================================
	//Update evolving variable by 1/2 time step
	AdvkernelCPU(XParam_local, XModel.blocks, XModel.time.dt * T(0.5), XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o, nblk_local_start);
	
	//============================================
	// Corrector step in reimann solver
	//============================================

	//============================================
	//  Fill the halo for gradient reconstruction
	//  This needs to be done across all processes
	fillHalo(XParam, XModel.blocks, XModel.evolv_o, XModel.zb); // Potentially MPI-aware version needed

	//============================================
	// Calculate gradient for evolving parameters
	gradientCPU(XParam_local, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.zb, nblk_local_start);

	//============================================
	// Flux and Source term reconstruction
	if (XParam.engine == 1)
	{
		UpdateButtingerXCPU(XParam_local, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, nblk_local_start);
		UpdateButtingerYCPU(XParam_local, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, nblk_local_start);
	}
	else if (XParam.engine == 2)
	{
		updateKurgXCPU(XParam_local, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, nblk_local_start);
		updateKurgYCPU(XParam_local, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, nblk_local_start);
	}
	else if (XParam.engine == 3)
	{
		updateKurgXATMCPU(XParam_local, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, XModel.Patm, XModel.datmpdx, nblk_local_start);
		updateKurgYATMCPU(XParam_local, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, XModel.Patm, XModel.datmpdy, nblk_local_start);
	}

	//============================================
	// Fill Halo for flux from fine to coarse
	// This needs to be done across all processes
	fillHalo(XParam, XModel.blocks, XModel.flux); // Potentially MPI-aware version needed
	for (int iseg = 0; iseg < XForcing.bndseg.size(); iseg++)
	{
		// This function might need MPI awareness
		FlowbndFlux(XParam, XLoop.totaltime + XLoop.dt , XModel.blocks, XForcing.bndseg[iseg], XForcing.Atmp, XModel.evolv, XModel.flux);
	}

	//============================================
	// Update advection terms (dh dhu dhv) 
	updateEVCPU(XParam_local, XModel.blocks, XModel.evolv_o, XModel.flux, XModel.adv, nblk_local_start);
	
	//============================================
	// Add forcing (Rain, Wind)
	if (!XForcing.UWind.inputfile.empty())
	{
		AddwindforcingCPU(XParam_local, XModel.blocks, XForcing.UWind, XForcing.VWind, XModel.adv, nblk_local_start);
	}
	if (XForcing.rivers.size() > 0)
	{
		// This function likely needs MPI awareness
		AddRiverForcing(XParam, XLoop, XForcing.rivers, XModel);
	}

	//============================================
	//Update evolving variable by 1 full time step
	AdvkernelCPU(XParam_local, XModel.blocks, XModel.time.dt, XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o, nblk_local_start);
	
	//============================================
	// Add bottom friction
	bottomfrictionCPU(XParam_local, XModel.blocks, XModel.time.dt, XModel.cf, XModel.evolv_o, nblk_local_start);

	//============================================
	//Copy updated evolving variable back
	cleanupCPU(XParam_local, XModel.blocks, XModel.evolv_o, XModel.evolv, nblk_local_start);

	if (!XForcing.Rain.inputfile.empty())
	{
		AddrainforcingImplicitCPU(XParam_local, XLoop, XModel.blocks, XForcing.Rain, XModel.evolv, nblk_local_start);
	}

	if (XParam.infiltration)
	{
		AddinfiltrationImplicitCPU(XParam_local, XLoop, XModel.blocks, XModel.il, XModel.cl, XModel.evolv, XModel.hgw, nblk_local_start);
	}

	if (XParam.VelThreshold > 0.0)
	{
		TheresholdVelCPU(XParam_local, XModel.blocks, XModel.evolv, nblk_local_start);
	}

	//============================================
	// Reset zb in halo from prolonggation injection
	// This might require MPI communication if it affects neighboring blocks on other processes
	if (XParam.conserveElevation)
	{
		refine_linear(XParam_local, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy, nblk_local_start);
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
	// Determine local block range for this MPI process
	int nblk_local_start, nblk_local_end;
	int nblk_per_process = XParam.nblk / XParam.size;
	int remainder_blks = XParam.nblk % XParam.size;

	if (XParam.rank < remainder_blks) {
		nblk_local_start = XParam.rank * (nblk_per_process + 1);
		nblk_local_end = nblk_local_start + nblk_per_process + 1;
	} else {
		nblk_local_start = XParam.rank * nblk_per_process + remainder_blks;
		nblk_local_end = nblk_local_start + nblk_per_process;
	}

	// Create a new Param object with local block counts for functions that iterate internally
	Param XParam_local = XParam;
	XParam_local.nblk = nblk_local_end - nblk_local_start;


	if (XParam.atmpforcing)
	{
		AddPatmforcingCPU(XParam_local, XModel.blocks, XForcing.Atmp, XModel, nblk_local_start);
		fillHaloC(XParam_local, XModel.blocks, XModel.Patm, nblk_local_start);
		gradientC(XParam_local, XModel.blocks, XModel.Patm, XModel.datmpdx, XModel.datmpdy, nblk_local_start);
		gradientHalo(XParam_local, XModel.blocks, XModel.Patm, XModel.datmpdx, XModel.datmpdy, nblk_local_start);
		refine_linear(XParam_local, XModel.blocks, XModel.Patm, XModel.datmpdx, XModel.datmpdy, nblk_local_start);
		gradientHalo(XParam_local, XModel.blocks, XModel.Patm, XModel.datmpdx, XModel.datmpdy, nblk_local_start);
	}
	
	//============================================
	//  Fill the halo for gradient reconstruction
	fillHalo(XParam, XModel.blocks, XModel.evolv, XModel.zb); // Potentially MPI-aware

	//============================================
	// Reset DTmax
	InitArrayBUQ(XParam_local, XModel.blocks, XLoop.hugeposval, XModel.time.dtmax, nblk_local_start);

	//============================================
	// Calculate gradient for evolving parameters
	gradientCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.zb, nblk_local_start);

	//============================================
	// Flux and Source term reconstruction
	if (XParam.engine == 1)
	{
		UpdateButtingerXCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, nblk_local_start);
		UpdateButtingerYCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, nblk_local_start);
	}
	else if (XParam.engine == 2)
	{
		updateKurgXCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, nblk_local_start);
		updateKurgYCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, nblk_local_start);
	}
	else if (XParam.engine == 3)
	{
		updateKurgXATMCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, XModel.Patm, XModel.datmpdx, nblk_local_start);
		updateKurgYATMCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb, XModel.Patm, XModel.datmpdy, nblk_local_start);
	}

	//============================================
	// Fill Halo for flux from fine to coarse
	fillHalo(XParam, XModel.blocks, XModel.flux); // Potentially MPI-aware

	//============================================
	// Reduce minimum timestep - Requires MPI_Allreduce
	T local_dt_min = CalctimestepCPU(XParam_local, XLoop, XModel.blocks, XModel.time, nblk_local_start);
#ifdef USE_MPI
	MPI_Allreduce(&local_dt_min, &XLoop.dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); // Assuming T can be MPI_DOUBLE or MPI_FLOAT
#else
	XLoop.dt = local_dt_min;
#endif
	XLoop.dtmax = XLoop.dt;
	XModel.time.dt = T(XLoop.dt);

	//============================================
	// Update advection terms (dh dhu dhv) 
	updateEVCPU(XParam_local, XModel.blocks, XModel.evolv, XModel.flux, XModel.adv, nblk_local_start);

	//============================================
	// Add forcing (Rain, Wind)
	if (!XForcing.UWind.inputfile.empty())
	{
		AddwindforcingCPU(XParam_local, XModel.blocks, XForcing.UWind, XForcing.VWind, XModel.adv, nblk_local_start);
	}
	if (XForcing.rivers.size() > 0)
	{
		AddRiverForcing(XParam, XLoop, XForcing.rivers, XModel); // Potentially MPI-aware
	}

	//============================================
	//Update evolving variable by 1/2 time step
	AdvkernelCPU(XParam_local, XModel.blocks, XModel.time.dt * T(0.5), XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o, nblk_local_start);

	//============================================
	// Add bottom friction
	bottomfrictionCPU(XParam_local, XModel.blocks, XModel.time.dt, XModel.cf, XModel.evolv_o, nblk_local_start);

	//============================================
	//Copy updated evolving variable back
	cleanupCPU(XParam_local, XModel.blocks, XModel.evolv_o, XModel.evolv, nblk_local_start);

	if (!XForcing.Rain.inputfile.empty())
	{
		AddrainforcingImplicitCPU(XParam_local, XLoop, XModel.blocks, XForcing.Rain, XModel.evolv, nblk_local_start);
	}
  if (XParam.infiltration)
	{
		AddinfiltrationImplicitCPU(XParam_local, XLoop, XModel.blocks, XModel.il, XModel.cl, XModel.evolv, XModel.hgw, nblk_local_start);
	}

	if (XParam.VelThreshold > 0.0)
	{
		TheresholdVelCPU(XParam_local, XModel.blocks, XModel.evolv, nblk_local_start);
	}

	//============================================
	// Reset zb in halo from prolonggation injection
	if (XParam.conserveElevation)
	{
		refine_linear(XParam_local, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy, nblk_local_start); // Potentially MPI-aware
	}
}
template void HalfStepCPU<float>(Param XParam, Loop<float>& XLoop, Forcing<float> XForcing, Model<float> XModel);
template void HalfStepCPU<double>(Param XParam, Loop<double>& XLoop, Forcing<float> XForcing, Model<double> XModel);


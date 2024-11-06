#include "FlowMLGPU.h"

template <class T> void FlowMLGPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, ModelML<T> XModel)
{
	
	//============================================
	//  Fill the halo for gradient reconstruction & Recalculate zs
	fillHaloGPU(XParam, XModel.blocks, XModel.evolv, XModel.zb);

	// Set max timestep

	//Calculate barotropic acceleration

	// Compute face value
		
	// Acceleration
	
	// Pressure

	// Advection

}

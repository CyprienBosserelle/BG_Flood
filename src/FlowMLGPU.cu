#include "FlowMLGPU.h"

template <class T> void FlowMLGPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel)
{
	
	//============================================
	//  Fill the halo for gradient reconstruction & Recalculate zs
	fillHaloGPU(XParam, XModel.blocks, XModel.evolv, XModel.zb);
	

	//============================================
	// Calculate gradient for evolving parameters for predictor step
	gradientGPUnew(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.zb);

	//============================================
	// Synchronise all ongoing streams
	CUDA_CHECK(cudaDeviceSynchronize());

	// Set max timestep
	reset_var <<< gridDim, blockDim, 0 >>> (XParam.halowidth, XModel.blocks.active, XLoop.hugeposval, XModel.time.dtmax);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Compute face value
	CalcfaceValX << < gridDim, blockDim, 0 >> > (XLoop.dt, XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.fluxml, XModel.time.dtmax, XModel.zb);
	CUDA_CHECK(cudaDeviceSynchronize());

	CalcfaceValY << < gridDim, blockDim, 0 >> > (XLoop.dt, XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.fluxml, XModel.time.dtmax, XModel.zb);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Timestep reduction
	XLoop.dt = double(CalctimestepGPU(XParam, XLoop, XModel.blocks, XModel.time));
	XLoop.dtmax = XLoop.dt;

	// Check hu/hv
	CheckadvecMLY << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XLoop.dt, XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

	CheckadvecMLX << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XLoop.dt, XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());


	
	// Acceleration
	// Pressure
	pressureML << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XLoop.dt, XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Advection
	AdvecFluxML << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XLoop.dt, XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

	AdvecEv << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XLoop.dt, XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

}
template void FlowMLGPU<float>(Param XParam, Loop<float>& XLoop, Forcing<float> XForcing, Model<float> XModel);
template void FlowMLGPU<double>(Param XParam, Loop<double>& XLoop, Forcing<float> XForcing, Model<double> XModel);


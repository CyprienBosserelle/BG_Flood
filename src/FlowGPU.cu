#include "FlowGPU.h"

template <class T> void FlowGPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel)
{
	//============================================
	// construct threads abnd block parameters
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	// for flux reconstruction the loop overlap the right(or top for the y direction) halo
	dim3 blockDimKX(XParam.blkwidth + XParam.halowidth, XParam.blkwidth, 1);
	dim3 blockDimKY(XParam.blkwidth, XParam.blkwidth + XParam.halowidth, 1);

	//============================================
	// Build cuda threads for multitasking on the GPU
	for (int i = 0; i < XLoop.num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&XLoop.streams[i]));
	}
	
	//============================================
	// Predictor step in reimann solver
	//============================================

	//============================================
	//  Fill the halo for gradient reconstruction
	fillHaloGPU(XParam, XModel.blocks, XModel.evolv, XModel.zb);


	//============================================
	// Reset DTmax
	reset_var <<< gridDim, blockDim, 0 >>> (XParam.halowidth,XModel.blocks.active,XLoop.hugeposval,XModel.time.dtmax);
	CUDA_CHECK(cudaDeviceSynchronize());

	//============================================
	// Calculate gradient for evolving parameters for predictor step
	gradientGPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.zb);
	
	//============================================
	// Synchronise all ongoing streams
	CUDA_CHECK(cudaDeviceSynchronize());

	
	
	//============================================
	// Flux and Source term reconstruction
	// X- direction
	updateKurgXGPU <<< gridDim, blockDimKX, 0 >>> (XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceXGPU <<< gridDim, blockDimKX, 0, XLoop.streams[0] >>> (XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);

	// Y- direction
	updateKurgYGPU <<< gridDim, blockDimKY, 0 >>> (XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceYGPU <<< gridDim, blockDimKY, 0, XLoop.streams[1] >>> (XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);
	//updateKurgY << < XLoop.gridDim, XLoop.blockDim, 0, XLoop.streams[0] >> > (XParam, XLoop.epsilon, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax);
	
	CUDA_CHECK(cudaDeviceSynchronize());

	//============================================
	// Fill Halo for flux from fine to coarse
	//fillHaloGPU(XParam, XModel.blocks, XModel.flux);

	//============================================
	// Reduce minimum timestep
	XLoop.dt = double(CalctimestepGPU(XParam, XLoop, XModel.blocks, XModel.time));
	XLoop.dtmax = XLoop.dt;
	

	XModel.time.dt = T(XLoop.dt);

	//============================================
	// Update advection terms (dh dhu dhv) 
	updateEVGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XModel.evolv, XModel.flux, XModel.adv);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	//============================================
	// Add forcing (Rain, Wind)
	if (!XForcing.Rain.inputfile.empty())
	{
		AddrainforcingGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XForcing.Rain, XModel.adv);
	}
	if (!XForcing.UWind.inputfile.empty())//&& !XForcing.UWind.inputfile.empty()
	{
		AddwindforcingGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XForcing.UWind, XForcing.VWind, XModel.adv);
	}
	if (XForcing.rivers.size() > 0)
	{
		AddRiverForcing(XParam, XLoop, XForcing.rivers, XModel);
	}

	//============================================
	//Update evolving variable by 1/2 time step
	AdvkernelGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XModel.time.dt*T(0.5), XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	
	//============================================
	// Corrector step in reimann solver
	//============================================

	//============================================
	//  Fill the halo for gradient reconstruction also wall boundary for masked block
	fillHaloGPU(XParam, XModel.blocks, XModel.evolv_o, XModel.zb);

	//============================================
	// Calculate gradient for evolving parameters
	gradientGPU(XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	//============================================
	// Flux and Source term reconstruction
	// X- direction
	updateKurgXGPU <<< gridDim, blockDimKX, 0 >>> (XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceXGPU <<< gridDim, blockDimKX, 0 >>> (XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.zb);

	// Y- direction
	updateKurgYGPU <<< gridDim, blockDimKY, 0 >>> (XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceYGPU <<< gridDim, blockDimKY, 0 >>> (XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.zb);
	CUDA_CHECK(cudaDeviceSynchronize());

	//============================================
	// Fill Halo for flux from fine to coarse
	//fillHaloGPU(XParam, XModel.blocks, XModel.flux);

	//============================================
	// Update advection terms (dh dhu dhv) 
	updateEVGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XModel.evolv_o, XModel.flux, XModel.adv);
	CUDA_CHECK(cudaDeviceSynchronize());
	

	//============================================
	// Add forcing (Rain, Wind)
	if (!XForcing.Rain.inputfile.empty())
	{
		AddrainforcingGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XForcing.Rain, XModel.adv);
	}
	if (!XForcing.UWind.inputfile.empty())//&& !XForcing.UWind.inputfile.empty()
	{
		AddwindforcingGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XForcing.UWind, XForcing.VWind, XModel.adv);
	}
	if (XForcing.rivers.size() > 0)
	{
		AddRiverForcing(XParam, XLoop, XForcing.rivers, XModel);
	}


	//============================================
	//Update evolving variable by 1 full time step
	AdvkernelGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XModel.time.dt, XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	
	//============================================
	// Add bottom friction
	bottomfrictionGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XModel.time.dt, XModel.cf, XModel.evolv_o);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	//============================================
	//Copy updated evolving variable back
	cleanupGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XModel.evolv_o, XModel.evolv);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	
	for (int i = 0; i < XLoop.num_streams; i++)
	{
		cudaStreamDestroy(XLoop.streams[i]);
	}

}
template void FlowGPU<float>(Param XParam, Loop<float>& XLoop, Forcing<float> XForcing, Model<float> XModel);
template void FlowGPU<double>(Param XParam, Loop<double>& XLoop, Forcing<float> XForcing, Model<double> XModel);



template <class T> __global__ void reset_var(int halowidth, int* active, T resetval, T* Var)
{

	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = active[ibl];

	int n = memloc(halowidth, blkmemwidth, ix, iy, ib);
	//int n= (ix + halowidth) + (iy + halowidth) * blkmemwidth + ib * blksize;
	Var[n] = resetval;
}
template __global__ void reset_var<float>(int halowidth, int* active, float resetval, float* Var);
template __global__ void reset_var<double>(int halowidth, int* active, double resetval, double* Var);



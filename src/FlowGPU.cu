#include "FlowGPU.h"

template <class T> void FlowGPU(Param XParam, Loop<T>& XLoop, Model<T> XModel)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	for (int i = 0; i < XLoop.num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&XLoop.streams[i]));
	}
	

	fillHaloGPU(XParam, XModel.blocks, XModel.evolv);


	//============================================
	// Reset DTmax
	reset_var <<< gridDim, blockDim, 0, XLoop.streams[0] >>> (XParam.halowidth,XModel.blocks.active,XLoop.hugeposval,XModel.time.dtmax);
	
	//============================================
	// Calculate gradient for evolving parameters for predictor step
	gradientGPU(XParam, XLoop, XModel.blocks, XModel.evolv, XModel.grad);
	
	//============================================
	// Synchronise all ongoing streams
	CUDA_CHECK(cudaDeviceSynchronize());

	dim3 blockDimKX(XParam.blkwidth+XParam.halowidth, XParam.blkwidth, 1);
	dim3 blockDimKY(XParam.blkwidth , XParam.blkwidth + XParam.halowidth, 1);
	//dim3 gridDim(XParam.nblk, 1, 1);

	updateKurgXGPU << < gridDim, blockDimKX, 0, XLoop.streams[0] >> > (XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax);
	updateKurgYGPU << < gridDim, blockDimKY, 0, XLoop.streams[1] >> > (XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax);
	//updateKurgY << < XLoop.gridDim, XLoop.blockDim, 0, XLoop.streams[0] >> > (XParam, XLoop.epsilon, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax);
	
	CUDA_CHECK(cudaDeviceSynchronize());

	//fillHaloGPU(XParam, XModel.blocks, XModel.flux);

	
	XLoop.dt = double(CalctimestepGPU(XParam, XModel.blocks, XModel.time));

	if (ceil((XLoop.nextoutputtime - XLoop.totaltime) / XLoop.dt) > 0.0)
	{
		XLoop.dt = (XLoop.nextoutputtime - XLoop.totaltime) / ceil((XLoop.nextoutputtime - XLoop.totaltime) / XLoop.dt);
	}

	XModel.time.dt = T(XLoop.dt);

	updateEVGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.evolv, XModel.flux, XModel.adv);
	CUDA_CHECK(cudaDeviceSynchronize());

	AdvkernelGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.time.dt*T(0.5), XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Corrector step
	
	gradientGPU(XParam, XLoop, XModel.blocks, XModel.evolv_o, XModel.grad);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	updateKurgXGPU << < gridDim, blockDimKX, 0 >> > (XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax);
	updateKurgYGPU << < gridDim, blockDimKY, 0 >> > (XParam, XModel.blocks, XModel.evolv_o, XModel.grad, XModel.flux, XModel.time.dtmax);
	CUDA_CHECK(cudaDeviceSynchronize());

	//fillHaloGPU(XParam, XModel.blocks, XModel.flux);

	updateEVGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.evolv_o, XModel.flux, XModel.adv);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	AdvkernelGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.time.dt, XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	cleanupGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.evolv_o, XModel.evolv);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	


}
template void FlowGPU<float>(Param XParam, Loop<float>& XLoop, Model<float> XModel);
template void FlowGPU<double>(Param XParam, Loop<double>& XLoop, Model<double> XModel);



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



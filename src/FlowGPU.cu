#include "FlowGPU.h"

template <class T> void FlowGPU(Param XParam, Loop<T>& XLoop, Model<T> XModel)
{
	for (int i = 0; i < XLoop.num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&XLoop.streams[i]));
	}
	

	//============================================
	// Reset DTmax
	reset_var <<< XLoop.gridDim, XLoop.blockDim, 0, XLoop.streams[0] >>> (XParam.halowidth,XModel.blocks.active,XLoop.hugeposval,XModel.time.dtmax);
	
	//============================================
	// Calculate gradient for evolving parameters
	gradientGPU(XParam, XLoop, XModel.blocks, XModel.evolv, XModel.grad);
	
	//============================================
	// Synchronise all ongoing streams
	CUDA_CHECK(cudaDeviceSynchronize());

	updateKurgXGPU << < XLoop.gridDim, XLoop.blockDim, 0, XLoop.streams[0] >> > (XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax);
	//updateKurgY << < XLoop.gridDim, XLoop.blockDim, 0, XLoop.streams[0] >> > (XParam, XLoop.epsilon, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax);
	
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



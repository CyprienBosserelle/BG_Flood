#include "FlowGPU.h"

template <class T> void FlowGPU(Param XParam, Loop<T>& XLoop, Model<T> XModel)
{
	for (int i = 0; i < XLoop.num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&XLoop.streams[i]));
	}
	// Fill Halo

	// reset dtmax
	reset_var <<< XLoop.gridDim, XLoop.blockDim, 0, XLoop.streams[0] >>> (XParam.halowidth,XModel.blocks.active,XLoop.hugeposval,XModel.time.dtmax);
	
	
	gradient << < XLoop.gridDim, XLoop.blockDim, 0, XLoop.streams[1] >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel.evolv.h, XModel.grad.dhdx, XModel.grad.dhdy);
	gradient << < XLoop.gridDim, XLoop.blockDim, 0, XLoop.streams[2] >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel.evolv.zs, XModel.grad.dzsdx, XModel.grad.dzsdy);
	gradient << < XLoop.gridDim, XLoop.blockDim, 0, XLoop.streams[3] >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel.evolv.u, XModel.grad.dudx, XModel.grad.dudy);
	gradient << < XLoop.gridDim, XLoop.blockDim, 0, XLoop.streams[0] >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel.evolv.v, XModel.grad.dvdx, XModel.grad.dvdy);

	CUDA_CHECK(cudaDeviceSynchronize());

	dim3 blockDimHaloLeft(1, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y

	//dim3 gridDim(XParam.nblk, 1, 1);

	fillLeft << <XLoop.gridDim, blockDimHaloLeft, 0 >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, XModel.blocks.LeftBot, XModel.blocks.LeftTop, XModel.blocks.RightBot, XModel.blocks.BotRight, XModel.blocks.TopRight, XModel.grad.dhdx);
	CUDA_CHECK(cudaDeviceSynchronize());
		
	// Fill Halo
	
	
	
	


}
template void FlowGPU<float>(Param XParam, Loop<float>& XLoop, Model<float> XModel);
template void FlowGPU<double>(Param XParam, Loop<double>& XLoop, Model<double> XModel);

template <class T> __global__ void gradient(int halowidth,int* active,int * level,T theta, T dx, T* a, T* dadx, T* dady)
{
	//int *leftblk,int *rightblk,int* topblk, int * botblk,

	//int ix = threadIdx.x+1;
	//int iy = threadIdx.y+1;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = active[ibl];

	int lev = level[ib];

	T delta = calcres(dx, lev);


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	int iright, ileft, itop, ibot;
	// shared array index to make the code bit more readable
	unsigned int sx = ix + halowidth;
	unsigned int sy = iy + halowidth;


	
	__shared__ T a_s[18][18];




	a_s[sx][sy] = a[i];
	//__syncthreads;
	//syncthread is needed here ?


	// read the halo around the tile
	if (threadIdx.x == blockDim.x - 1)
	{
		iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
		a_s[sx + 1][sy] = a[iright];
	}


	if (threadIdx.x == 0)
	{
		ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);;
		a_s[sx - 1][sy] = a[ileft];
	}


	if (threadIdx.y == blockDim.y - 1)
	{
		itop = memloc(halowidth, blkmemwidth, ix , iy + 1, ib);;
		a_s[sx][sy + 1] = a[itop];
	}

	if (threadIdx.y == 0)
	{
		ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
		a_s[sx][sy - 1] = a[ibot];
	}

	__syncthreads;
	

	dadx[i] = minmod2(theta, a_s[sx - 1][sy], a_s[sx][sy], a_s[sx + 1][sy]) / delta;
	dady[i] = minmod2(theta, a_s[sx][sy - 1], a_s[sx][sy], a_s[sx][sy + 1]) / delta;
	

}


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



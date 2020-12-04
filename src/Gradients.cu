#include "Gradients.h"



template <class T> void gradientGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad,T* zb)
{
	const int num_streams = 4;
	/*
	cudaStream_t streams[num_streams];

	for (int i = 0; i < num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}
	*/
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	//gradient << < gridDim, blockDim, 0, streams[1] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx, XGrad.dhdy);
	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx, XGrad.dhdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	//gradient << < gridDim, blockDim, 0, streams[2] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	//gradient << < gridDim, blockDim, 0, streams[3] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx, XGrad.dudy);
	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx, XGrad.dudy);
	CUDA_CHECK(cudaDeviceSynchronize());

	//gradient << < gridDim, blockDim, 0, streams[0] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx, XGrad.dvdy);
	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx, XGrad.dvdy);
	CUDA_CHECK(cudaDeviceSynchronize());


	//CUDA_CHECK(cudaDeviceSynchronize());
	/*
	for (int i = 0; i < num_streams; i++)
	{
		cudaStreamDestroy(streams[i]);
	}
	*/


	fillHaloGPU(XParam, XBlock, XGrad);

	conserveElevationGradHaloGPU(XParam, XBlock, XEv.h, XEv.zs, zb, XGrad.dhdx, XGrad.dzsdx, XGrad.dhdy, XGrad.dzsdy);
	//conserveElevationGradHaloGPU(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	//conserveElevationGradHaloGPU(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	//conserveElevationGradHaloGPU(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);

}
template void gradientGPU<float>(Param XParam, BlockP<float>XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, float * zb);
template void gradientGPU<double>(Param XParam,  BlockP<double>XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, double * zb);



template <class T> __global__ void gradient(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dadx, T* dady)
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


	T a_l, a_t, a_r, a_b,a_i;

	a_i = a[i];


	a_l = a[memloc(halowidth, blkmemwidth, ix - 1, iy, ib)];
	a_t = a[memloc(halowidth, blkmemwidth, ix , iy+1, ib)];
	a_r = a[memloc(halowidth, blkmemwidth, ix + 1, iy, ib)];
	a_b = a[memloc(halowidth, blkmemwidth, ix, iy-1, ib)];
	//__shared__ T a_s[18][18];



	__syncthreads();
	//__syncwarp;

	dadx[i] = minmod2(theta, a_l, a_i, a_r) / delta;
	
	dady[i] = minmod2(theta, a_b, a_i, a_t) / delta;


}

template <class T> __global__ void gradientSM(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dadx, T* dady)
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
	__syncthreads;
	//syncthread is needed here ?
		

	// read the halo around the tile
	if (threadIdx.x == blockDim.x - 1)
	{
		iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
		a_s[sx + 1][sy] = a[iright];
		__syncthreads;
	}
	

	if (threadIdx.x == 0)
	{
		ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);;
		a_s[sx - 1][sy] = a[ileft];
		__syncthreads;
	}
	

	if (threadIdx.y == blockDim.y - 1)
	{
		itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);;
		a_s[sx][sy + 1] = a[itop];
		__syncthreads;
	}
	
	if (threadIdx.y == 0)
	{
		ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
		a_s[sx][sy - 1] = a[ibot];
		__syncthreads;
	}

	__syncthreads;


	dadx[i] = minmod2(theta, a_s[sx - 1][sy], a_s[sx][sy], a_s[sx + 1][sy]) / delta;
	__syncthreads;
	dady[i] = minmod2(theta, a_s[sx][sy - 1], a_s[sx][sy], a_s[sx][sy + 1]) / delta;


}


template <class T> void gradientC(Param XParam, BlockP<T> XBlock, T* a, T* dadx, T* dady)
{

	int i,ib;
	int xplus, xminus, yplus, yminus;

	T delta;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		delta = calcres(T(XParam.dx), XBlock.level[ib]);
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				i = memloc(XParam, ix,iy,ib);
				
				//
				xplus = memloc(XParam, ix+1, iy, ib);
				xminus = memloc(XParam, ix-1, iy, ib);
				yplus = memloc(XParam, ix, iy+1, ib);
				yminus = memloc(XParam, ix, iy-1, ib);

				dadx[i] = minmod2(T(XParam.theta), a[xminus], a[i], a[xplus]) / delta;
				dady[i] = minmod2(T(XParam.theta), a[yminus], a[i], a[yplus]) / delta;
			}


		}


	}



}
template void gradientC<float>(Param XParam, BlockP<float> XBlock, float* a, float* dadx, float* dady);
template void gradientC<double>(Param XParam, BlockP<double> XBlock, double* a, double* dadx, double* dady);

template <class T> void gradientCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{


	std::thread t0(&gradientC<T>, XParam, XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
	std::thread t1(&gradientC<T>, XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	std::thread t2(&gradientC<T>, XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	std::thread t3(&gradientC<T>, XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);

	t0.join();
	t1.join();
	t2.join();
	t3.join();

	fillHalo(XParam, XBlock, XGrad);

	conserveElevationGradHalo(XParam, XBlock, XEv.h, XEv.zs, zb, XGrad.dhdx, XGrad.dzsdx, XGrad.dhdy, XGrad.dzsdy);
	//conserveElevationGradHalo(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	//conserveElevationGradHalo(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	//conserveElevationGradHalo(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);


}
template void gradientCPU<float>(Param XParam, BlockP<float>XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, float * zb);
template void gradientCPU<double>(Param XParam, BlockP<double>XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, double * zb);


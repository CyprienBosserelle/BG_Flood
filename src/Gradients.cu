#include "Gradients.h"


/*! \fn void gradientGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad,T* zb)
* Wrapping function to calculate gradien of evolving variables on GPU
* This function is the entry point to the gradient functions on the GPU
*/
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


	//fillHaloGPU(XParam, XBlock, XGrad);
	gradientHaloGPU(XParam, XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
	gradientHaloGPU(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	gradientHaloGPU(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	gradientHaloGPU(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);


	if (XParam.conserveElevation)
	{
		conserveElevationGradHaloGPU(XParam, XBlock, XEv.h, XEv.zs, zb, XGrad.dhdx, XGrad.dzsdx, XGrad.dhdy, XGrad.dzsdy);
	}
	//conserveElevationGradHaloGPU(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	//conserveElevationGradHaloGPU(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	//conserveElevationGradHaloGPU(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);

}
template void gradientGPU<float>(Param XParam, BlockP<float>XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, float * zb);
template void gradientGPU<double>(Param XParam,  BlockP<double>XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, double * zb);


/*! \fn void gradient(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dadx, T* dady)
* Device kernel for calculating grdients for an evolving poarameter using the minmod limiter
* 
*/
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

/*! \fn void gradientSM(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dadx, T* dady)
* Depreciated shared memory version of Device kernel for calculating gradients
* Much slower than above
*/
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

	//fillHalo(XParam, XBlock, XGrad);
	
	gradientHalo(XParam, XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
	gradientHalo(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	gradientHalo(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	gradientHalo(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);
	

	if (XParam.conserveElevation)
	{
		conserveElevationGradHalo(XParam, XBlock, XEv.h, XEv.zs, zb, XGrad.dhdx, XGrad.dzsdx, XGrad.dhdy, XGrad.dzsdy);

	}
	//conserveElevationGradHalo(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	//conserveElevationGradHalo(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	//conserveElevationGradHalo(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdyythhhhhhhhhg);


}
template void gradientCPU<float>(Param XParam, BlockP<float>XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, float * zb);
template void gradientCPU<double>(Param XParam, BlockP<double>XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, double * zb);


template <class T> void gradientHalo(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady)
{
	int i, ib;
	int xplus, xminus, yplus, yminus;

	T delta;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			gradientHaloLeft(XParam, XBlock, ib, iy, a, dadx, dady);
			gradientHaloRight(XParam, XBlock, ib, iy, a, dadx, dady);
		}
		for (int ix = 0; ix < XParam.blkwidth; ix++)
		{
			gradientHaloBot(XParam, XBlock, ib, ix, a, dadx, dady);
			gradientHaloTop(XParam, XBlock, ib, ix, a, dadx, dady);
		}
	}
}


template <class T> void gradientHaloGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady)
{
	dim3 blockDimL(1, XParam.blkwidth, 1);
	dim3 blockDimB(XParam.blkwidth, 1, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	
	gradientHaloLeftGPU << < gridDim, blockDimL, 0 >> > (XParam, XBlock, a, dadx, dady);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientHaloBotGPU << < gridDim, blockDimB, 0 >> > (XParam, XBlock, a, dadx, dady);
	CUDA_CHECK(cudaDeviceSynchronize());
		
	
}


template <class T> void gradientHaloLeft(Param XParam, BlockP<T>XBlock, int ib, int iy, T* a, T* dadx, T* dady)
{
	int i, j, ix, jj, ii, ir, it, itr;
	int xplus, read;
	
	T delta, aright, aleft, abot, atop;

	ix = -1;

	i = memloc(XParam, ix, iy, ib);
	xplus = memloc(XParam, ix + 1, iy, ib);
	

	aright = a[xplus];
	


	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.LeftBot[ib] == ib)//The lower half is a boundary 
	{
		if ( iy < (XParam.blkwidth / 2))
		{

			read = memloc(XParam, 0, iy, ib);// or memloc(XParam, -1, j, ib) but they should be the same
			
			aleft = a[read];
		}

		if (XBlock.LeftTop[ib] == ib) // boundary on the top half too
		{
			if ( iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, 0, iy, ib);
				
				aleft = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (iy >= (XParam.blkwidth / 2))
			{
				
				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, (XParam.blkwidth - 3), jj, XBlock.LeftTop[ib]);
				ir = memloc(XParam, (XParam.blkwidth - 4), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam, (XParam.blkwidth - 3), jj + 1, XBlock.LeftTop[ib]);
				itr = memloc(XParam, (XParam.blkwidth - 4), jj + 1, XBlock.LeftTop[ib]);

				aleft = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.LeftBot[ib]]) // LeftTop block does not exist
	{
		
			read = memloc(XParam, (XParam.blkwidth - 2), iy, XBlock.LeftBot[ib]);
			aleft = a[read];
		
	}
	else if (XBlock.level[XBlock.LeftBot[ib]] > XBlock.level[ib])
	{

		if (iy < (XParam.blkwidth / 2))
		{	

			jj = iy * 2;
			int bb = XBlock.LeftBot[ib];

			ii = memloc(XParam, (XParam.blkwidth - 3), jj, bb);
			ir = memloc(XParam, (XParam.blkwidth - 4), jj, bb);
			it = memloc(XParam, (XParam.blkwidth - 3), jj + 1, bb);
			itr = memloc(XParam, (XParam.blkwidth - 4), jj + 1, bb);

			aleft = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.LeftTop[ib] == ib)
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, 0, iy, ib);
				
				aleft = a[read];
			}
		}
		else
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//
				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, (XParam.blkwidth - 3), jj, XBlock.LeftTop[ib]);
				ir = memloc(XParam, (XParam.blkwidth - 4), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam, (XParam.blkwidth - 3), jj + 1, XBlock.LeftTop[ib]);
				itr = memloc(XParam, (XParam.blkwidth - 4), jj + 1, XBlock.LeftTop[ib]);

				aleft = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.LeftBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.RightBot[XBlock.LeftBot[ib]] == ib ? ceil(iy * (T)0.5) : ceil(iy * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(iy * (T)0.5) * 2 > iy ? T(0.75) : T(0.25);

		ii = memloc(XParam, (XParam.blkwidth - 1), jj, XBlock.LeftBot[ib]);
		ir = memloc(XParam, (XParam.blkwidth - 2), jj, XBlock.LeftBot[ib]);
		it = memloc(XParam, (XParam.blkwidth - 1), jj - 1, XBlock.LeftBot[ib]);
		itr = memloc(XParam, (XParam.blkwidth - 2), jj - 1, XBlock.LeftBot[ib]);

		aleft = BilinearInterpolation(a[itr], a[it], a[ir], a[ii], T(0.0), T(1.0), T(0.0), T(1.0), T(0.75), jr);
	}
	




	dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	//dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}


template <class T> void gradientHaloRight(Param XParam, BlockP<T>XBlock, int ib, int iy, T* a, T* dadx, T* dady)
{
	int i, j, ix, jj, ii, ir, it, itr;
	int xminus, read;

	T delta, aright, aleft, abot, atop;

	ix = 16;

	i = memloc(XParam, ix, iy, ib);
	xminus = memloc(XParam, ix - 1, iy, ib);


	aleft = a[xminus];



	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.RightBot[ib] == ib)//The lower half is a boundary 
	{
		if (iy < (XParam.blkwidth / 2))
		{

			read = memloc(XParam, XParam.blkwidth -1, iy, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			aright = a[read];
		}

		if (XBlock.RightTop[ib] == ib) // boundary on the top half too
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, XParam.blkwidth - 1, iy, ib);

				aright = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of righttopib is levelib+1
		{

			if (iy >= (XParam.blkwidth / 2))
			{

				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, 3, jj, XBlock.RightTop[ib]);
				ir = memloc(XParam, 2, jj, XBlock.RightTop[ib]);
				it = memloc(XParam, 3, jj + 1, XBlock.RightTop[ib]);
				itr = memloc(XParam, 2, jj + 1, XBlock.RightTop[ib]);

				aright = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.RightBot[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam, 1, iy, XBlock.RightBot[ib]);
		aright = a[read];

	}
	else if (XBlock.level[XBlock.RightBot[ib]] > XBlock.level[ib])
	{

		if (iy < (XParam.blkwidth / 2))
		{

			jj = iy * 2;
			int bb = XBlock.RightBot[ib];

			ii = memloc(XParam, 3, jj, bb);
			ir = memloc(XParam, 2, jj, bb);
			it = memloc(XParam, 3, jj + 1, bb);
			itr = memloc(XParam, 2, jj + 1, bb);

			aright = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.RightTop[ib] == ib)
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, XParam.blkwidth - 1, iy, ib);

				aright = a[read];
			}
		}
		else
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//
				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, 3, jj, XBlock.RightTop[ib]);
				ir = memloc(XParam, 2, jj, XBlock.RightTop[ib]);
				it = memloc(XParam, 3, jj + 1, XBlock.RightTop[ib]);
				itr = memloc(XParam, 2, jj + 1, XBlock.RightTop[ib]);

				aright = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.RightBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.LeftBot[XBlock.RightBot[ib]] == ib ? ceil(iy * (T)0.5) : ceil(iy * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(iy * (T)0.5) * 2 > iy ? T(0.75) : T(0.25);

		ii = memloc(XParam, 0, jj, XBlock.RightBot[ib]);
		ir = memloc(XParam, 1, jj, XBlock.RightBot[ib]);
		it = memloc(XParam, 0, jj - 1, XBlock.RightBot[ib]);
		itr = memloc(XParam, 1, jj - 1, XBlock.RightBot[ib]);

		aright = BilinearInterpolation(a[itr], a[it], a[ir], a[ii], T(0.0), T(1.0), T(0.0), T(1.0), T(0.75), jr);
	}





	dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	//dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}


template <class T> void gradientHaloBot(Param XParam, BlockP<T>XBlock, int ib, int ix, T* a, T* dadx, T* dady)
{
	int i, j, iy, jj, ii, ir, it, itr;
	int xplus, xminus, yplus, yminus, read;

	T delta, atop, abot;

	iy = -1;

	i = memloc(XParam, ix, iy, ib);
	yplus = memloc(XParam, ix , iy + 1, ib);
	



	atop = a[yplus];



	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.BotLeft[ib] == ib)//The lower half is a boundary 
	{
		if (ix < (XParam.blkwidth / 2))
		{

			read = memloc(XParam, ix, 0, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			abot = a[read];
			
		}

		if (XBlock.BotRight[ib] == ib) // boundary on the top half too
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, ix, 0, ib);

				abot = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (ix >= (XParam.blkwidth / 2))
			{

				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, jj, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				ir = memloc(XParam, jj, (XParam.blkwidth - 4), XBlock.BotRight[ib]);
				it = memloc(XParam, jj + 1, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				itr = memloc(XParam, jj + 1, (XParam.blkwidth - 4), XBlock.BotRight[ib]);

				abot = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.BotLeft[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam, ix, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);
		abot = a[read];

	}
	else if (XBlock.level[XBlock.BotLeft[ib]] > XBlock.level[ib])
	{

		if (ix < (XParam.blkwidth / 2))
		{

			jj = ix * 2;
			int bb = XBlock.BotLeft[ib];

			ii = memloc(XParam, jj, (XParam.blkwidth - 3), bb);
			ir = memloc(XParam, jj, (XParam.blkwidth - 4), bb);
			it = memloc(XParam, jj + 1, (XParam.blkwidth - 3), bb);
			itr = memloc(XParam, jj + 1, (XParam.blkwidth - 4), bb);

			abot = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.BotRight[ib] == ib)
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, ix, 0, ib);

				abot = a[read];
			}
		}
		else
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//
				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, jj, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				ir = memloc(XParam, jj, (XParam.blkwidth - 4), XBlock.BotRight[ib]);
				it = memloc(XParam, jj + 1, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				itr = memloc(XParam, jj + 1, (XParam.blkwidth - 4), XBlock.BotRight[ib]);

				abot = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.BotLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.TopLeft[XBlock.BotLeft[ib]] == ib ? ceil(ix * (T)0.5) : ceil(ix * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(ix * (T)0.5) * 2 > ix ? T(0.75) : T(0.25);

		ii = memloc(XParam, jj, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
		ir = memloc(XParam, jj, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);
		it = memloc(XParam, jj - 1, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
		itr = memloc(XParam, jj - 1, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);

		abot = BilinearInterpolation(a[itr], a[it], a[ir], a[ii], T(0.0), T(1.0), T(0.0), T(1.0), jr, T(0.75));
	}





	//dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}

template <class T> void gradientHaloTop(Param XParam, BlockP<T>XBlock, int ib, int ix, T* a, T* dadx, T* dady)
{
	int i, j, iy, jj, ii, ir, it, itr;
	int xplus, xminus, yplus, yminus, read;

	T delta, atop, abot;

	iy = XParam.blkwidth;

	i = memloc(XParam, ix, iy, ib);
	yminus = memloc(XParam, ix, XParam.blkwidth-1, ib);




	abot = a[yminus];



	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.TopLeft[ib] == ib)//The lower half is a boundary 
	{
		if (ix < (XParam.blkwidth / 2))
		{

			read = memloc(XParam, ix, XParam.blkwidth - 1, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			atop = a[read];

		}

		if (XBlock.TopRight[ib] == ib) // boundary on the top half too
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, ix, XParam.blkwidth - 1, ib);

				atop = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (ix >= (XParam.blkwidth / 2))
			{

				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, jj, 3, XBlock.TopRight[ib]);
				ir = memloc(XParam, jj, 2, XBlock.TopRight[ib]);
				it = memloc(XParam, jj + 1, 3, XBlock.TopRight[ib]);
				itr = memloc(XParam, jj + 1, 2, XBlock.TopRight[ib]);

				atop = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.TopLeft[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam, ix, 1, XBlock.TopLeft[ib]);
		atop = a[read];

	}
	else if (XBlock.level[XBlock.TopLeft[ib]] > XBlock.level[ib])
	{

		if (ix < (XParam.blkwidth / 2))
		{

			jj = ix * 2;
			int bb = XBlock.TopLeft[ib];

			ii = memloc(XParam, jj, 3, bb);
			ir = memloc(XParam, jj, 2, bb);
			it = memloc(XParam, jj + 1, 3, bb);
			itr = memloc(XParam, jj + 1, 2, bb);

			atop = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.TopRight[ib] == ib)
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, ix, XParam.blkwidth - 1, ib);

				atop = a[read];
			}
		}
		else
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//
				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, jj, 3, XBlock.TopRight[ib]);
				ir = memloc(XParam, jj, 2, XBlock.TopRight[ib]);
				it = memloc(XParam, jj + 1, 3, XBlock.TopRight[ib]);
				itr = memloc(XParam, jj + 1, 2, XBlock.TopRight[ib]);

				atop = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.TopLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.BotLeft[XBlock.TopLeft[ib]] == ib ? ceil(ix * (T)0.5) : ceil(ix * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(ix * (T)0.5) * 2 > ix ? T(0.75) : T(0.25);

		ii = memloc(XParam, jj, 0, XBlock.TopLeft[ib]);
		ir = memloc(XParam, jj, 1, XBlock.TopLeft[ib]);
		it = memloc(XParam, jj - 1, 0, XBlock.TopLeft[ib]);
		itr = memloc(XParam, jj - 1, 1, XBlock.TopLeft[ib]);

		atop = BilinearInterpolation(a[itr], a[it], a[ir], a[ii], T(0.0), T(1.0), T(0.0), T(1.0), jr, T(0.75));
	}





	//dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}



template <class T> __global__ void gradientHaloLeftGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int ix = -1;
	int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];
	int i, j,  jj, ii, ir, it, itr;
	int xplus, read;

	T delta, aright, aleft, abot, atop;

	

	i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);
	xplus = memloc(XParam.halowidth, blkmemwidth, ix + 1, iy, ib);


	aright = a[xplus];



	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.LeftBot[ib] == ib)//The lower half is a boundary 
	{
		if (iy < (XParam.blkwidth / 2))
		{

			read = memloc(XParam.halowidth, blkmemwidth, 0, iy, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			aleft = a[read];
		}

		if (XBlock.LeftTop[ib] == ib) // boundary on the top half too
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, 0, iy, ib);

				aleft = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (iy >= (XParam.blkwidth / 2))
			{

				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj, XBlock.LeftTop[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj + 1, XBlock.LeftTop[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj + 1, XBlock.LeftTop[ib]);

				aleft = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.LeftBot[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 2), iy, XBlock.LeftBot[ib]);
		aleft = a[read];

	}
	else if (XBlock.level[XBlock.LeftBot[ib]] > XBlock.level[ib])
	{

		if (iy < (XParam.blkwidth / 2))
		{

			jj = iy * 2;
			int bb = XBlock.LeftBot[ib];

			ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj, bb);
			ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj, bb);
			it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj + 1, bb);
			itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj + 1, bb);

			aleft = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.LeftTop[ib] == ib)
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, 0, iy, ib);

				aleft = a[read];
			}
		}
		else
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//
				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj, XBlock.LeftTop[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj + 1, XBlock.LeftTop[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj + 1, XBlock.LeftTop[ib]);

				aleft = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.LeftBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.RightBot[XBlock.LeftBot[ib]] == ib ? ceil(iy * (T)0.5) : ceil(iy * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(iy * (T)0.5) * 2 > iy ? T(0.75) : T(0.25);

		ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 1), jj, XBlock.LeftBot[ib]);
		ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 2), jj, XBlock.LeftBot[ib]);
		it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 1), jj - 1, XBlock.LeftBot[ib]);
		itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 2), jj - 1, XBlock.LeftBot[ib]);

		aleft = BilinearInterpolation(a[itr], a[it], a[ir], a[ii], T(0.0), T(1.0), T(0.0), T(1.0), T(0.75), jr);
	}





	dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	//dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}


template <class T> __global__ void gradientHaloBotGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int iy = -1;
	int ix = threadIdx.x;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];


	int i, j, jj, ii, ir, it, itr;
	int xplus, xminus, yplus, yminus, read;

	T delta, atop, abot;

	
	i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);
	yplus = memloc(XParam.halowidth, blkmemwidth, ix, iy + 1, ib);




	atop = a[yplus];



	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.BotLeft[ib] == ib)//The lower half is a boundary 
	{
		if (ix < (XParam.blkwidth / 2))
		{

			read = memloc(XParam.halowidth, blkmemwidth, ix, 0, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			abot = a[read];

		}

		if (XBlock.BotRight[ib] == ib) // boundary on the top half too
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, ix, 0, ib);

				abot = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (ix >= (XParam.blkwidth / 2))
			{

				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 4), XBlock.BotRight[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 4), XBlock.BotRight[ib]);

				abot = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.BotLeft[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam.halowidth, blkmemwidth, ix, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);
		abot = a[read];

		

	}
	else if (XBlock.level[XBlock.BotLeft[ib]] > XBlock.level[ib])
	{

		if (ix < (XParam.blkwidth / 2))
		{

			jj = ix * 2;
			int bb = XBlock.BotLeft[ib];

			ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 3), bb);
			ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 4), bb);
			it = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 3), bb);
			itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 4), bb);

			abot = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.BotRight[ib] == ib)
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, ix, 0, ib);

				abot = a[read];
			}
		}
		else
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//
				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 4), XBlock.BotRight[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 4), XBlock.BotRight[ib]);

				abot = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.BotLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.TopLeft[XBlock.BotLeft[ib]] == ib ? ceil(ix * (T)0.5) : ceil(ix * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(ix * (T)0.5) * 2 > ix ? T(0.75) : T(0.25);

		ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
		ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);
		it = memloc(XParam.halowidth, blkmemwidth, jj - 1, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
		itr = memloc(XParam.halowidth, blkmemwidth, jj - 1, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);

		abot = BilinearInterpolation(a[itr], a[it], a[ir], a[ii], T(0.0), T(1.0), T(0.0), T(1.0), jr, T(0.75));
	}





	//dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}



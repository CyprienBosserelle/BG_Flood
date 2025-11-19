#include "ConserveElevation.h"


template <class T> void conserveElevation(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		//int ii = memloc(XParam, -1, 5, 46);

		
		conserveElevationLeft(XParam, ib, XBlock.LeftBot[ib], XBlock.LeftTop[ib], XBlock, XEv, zb);
		
		conserveElevationRight(XParam, ib, XBlock.RightBot[ib], XBlock.RightTop[ib], XBlock, XEv, zb);
		
		conserveElevationTop(XParam, ib, XBlock.TopLeft[ib], XBlock.TopRight[ib], XBlock, XEv, zb);
		
		conserveElevationBot(XParam, ib, XBlock.BotLeft[ib], XBlock.BotRight[ib], XBlock, XEv, zb);
		
	}
}
template void conserveElevation<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, float* zb);
template void conserveElevation<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, double* zb);

template <class T> void conserveElevationGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	dim3 blockDimHaloLR(1, 16, 1);
	dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDim(XParam.nblk, 1, 1);


	conserveElevationLeft <<<gridDim, blockDimHaloLR, 0 >>> (XParam, XBlock, XEv, zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	conserveElevationRight <<<gridDim, blockDimHaloLR, 0 >>> (XParam, XBlock, XEv, zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	conserveElevationTop <<<gridDim, blockDimHaloBT, 0 >>> (XParam, XBlock, XEv, zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	conserveElevationBot <<<gridDim, blockDimHaloBT, 0 >>> (XParam, XBlock, XEv, zb);
	CUDA_CHECK(cudaDeviceSynchronize());

}
template void conserveElevationGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, float* zb);
template void conserveElevationGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, double* zb);

template <class T> void WetDryProlongation(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int ib, ibLB, ibTL, ibBL, ibRB,ibn;
	int ihalo, jhalo, ip, jp;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		ibLB = XBlock.LeftBot[ib];
		ibRB = XBlock.RightBot[ib];

		ibTL = XBlock.TopLeft[ib];
		ibBL = XBlock.BotLeft[ib];

		//Left side
		if (XBlock.level[ib] > XBlock.level[ibLB])
		{
			// Prolongation
			for (int j = 0; j < XParam.blkwidth; j++)
			{

				ihalo = -1;
				//
				jhalo = j;
				ibn = ibLB;

				//il = 0;
				//jl = j;




				ip = XParam.blkwidth - 1;
				jp = XBlock.RightBot[ibLB] == ib ? ftoi(floor(j * T(0.5))) : ftoi(floor(j * T(0.5)) + XParam.blkwidth / 2);

				//im = ip;
				//jm = ceil(j * T(0.5)) * 2 > j ? jp + 1 : jp - 1;

				ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);
			}
		}

		//Right side
		if (XBlock.level[ib] > XBlock.level[ibRB])
		{
			// Prolongation
			for (int j = 0; j < XParam.blkwidth; j++)
			{

				ihalo = XParam.blkwidth;
				//
				jhalo = j;
				ibn = ibRB;

				//il = 0;
				//jl = j;




				ip = 0;
				jp = XBlock.LeftBot[ibn] == ib ? ftoi(floor(j * T(0.5))) : ftoi(floor(j * T(0.5)) + XParam.blkwidth / 2);

				//im = ip;
				//jm = ceil(j * T(0.5)) * 2 > j ? jp + 1 : jp - 1;

				ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);
			}
		}

		// Top side
		if (XBlock.level[ib] > XBlock.level[ibTL])
		{
			//
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				jhalo = XParam.blkwidth;
				//
				ihalo = i;
				ibn = ibTL;

				//il = i;
				//jl = XParam.blkwidth - 1;

				jp = 0;
				ip = XBlock.BotLeft[ibn] == ib ? ftoi(floor(i * T(0.5))) : ftoi(floor(i * T(0.5)) + XParam.blkwidth / 2);

				//jm = jp;
				//im = ceil(i * T(0.5)) * 2 > i ? ip + 1 : ip - 1;

				ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);
			}

		}

		// Bot side
		if (XBlock.level[ib] > XBlock.level[ibBL])
		{
			//
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				//
				jhalo = -1;
				ihalo = i;
				ibn = ibBL;

				//il = i;
				//jl = 0;

				jp = XParam.blkwidth - 1;
				ip = XBlock.TopLeft[ibn] == ib ? ftoi(floor(i * T(0.5))) : ftoi(floor(i * T(0.5)) + XParam.blkwidth / 2);

				//jm = jp;
				//im = ceil(i * T(0.5)) * 2 > i ? ip + 1 : ip - 1;

				ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);
			}

		}

	}

}
template void WetDryProlongation<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, double* zb);
template void WetDryProlongation<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, float* zb);

template <class T> void WetDryRestriction(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int ib, ibLB, ibTL, ibBL, ibRB, ibLT, ibRT, ibTR, ibBR, ibn;
	int ihalo, jhalo, ir, jr, lev;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		ibLB = XBlock.LeftBot[ib];
		ibLT = XBlock.LeftTop[ib];

		ibRB = XBlock.RightBot[ib];
		ibRT = XBlock.RightTop[ib];

		ibTL = XBlock.TopLeft[ib];
		ibTR = XBlock.TopRight[ib];

		ibBL = XBlock.BotLeft[ib];
		ibBR = XBlock.BotRight[ib];

		lev = XBlock.level[ib];
		

		//Left side
		ir = XParam.blkwidth - 2;
		ihalo = -1;
		

		if (lev < XBlock.level[ibLB])
		{
			
			for (int iy = 0; iy < (XParam.blkwidth / 2); iy++)
			{
				jhalo = iy;
				
				ibn = ibLB;

				
				jr = iy * 2;

				wetdryrestriction(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
			}
		}
		if (lev < XBlock.level[ibLT])
		{
			for (int iy = (XParam.blkwidth / 2); iy < XParam.blkwidth; iy++)
			{
				jhalo = iy;
				ibn = ibLT;
				jr = (iy - (XParam.blkwidth / 2)) * 2;

				wetdryrestriction(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
			}
		}
		

		//Right side

		ihalo = XParam.blkwidth;
		ir = 0;

		if (lev < XBlock.level[ibRB] )
		{

			for (int iy = 0; iy < (XParam.blkwidth / 2); iy++)
			{
				jhalo = iy;
				ibn = ibRB;
				jr = iy * 2;

				wetdryrestriction(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
			}
		}
		if (lev < XBlock.level[ibRT] )
		{
			for (int iy = (XParam.blkwidth / 2); iy < XParam.blkwidth; iy++)
			{
				jhalo = iy;
				ibn = ibRT;
				jr = (iy - (XParam.blkwidth / 2)) * 2;

				wetdryrestriction(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
			}
		}

		// Top side
		jhalo = XParam.blkwidth;
		
		jr = 0;

		if (lev < XBlock.level[ibTL] )
		{
			for (int ix = 0; ix < (XParam.blkwidth / 2); ix++)
			{
				ihalo = ix;
				ibn = ibTL;

				ir = ix * 2;

				wetdryrestriction(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
			}
		}
		if (lev < XBlock.level[ibTR] )
		{
			for (int ix = (XParam.blkwidth / 2); ix < XParam.blkwidth; ix++)
			{
				ihalo = ix;
				ibn = ibTR;
				ir = (ix - (XParam.blkwidth / 2)) * 2;

				wetdryrestriction(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
			}
		}

		// Bot side
		jhalo = -1;
		
		jr = XParam.blkwidth - 2;

		if (lev < XBlock.level[ibBL] )
		{
			for (int ix = 0; ix < (XParam.blkwidth / 2); ix++)
			{
				ihalo = ix;
				ibn = ibBL;

				ir = ix * 2;

				wetdryrestriction(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
			}
		}
		if (lev < XBlock.level[ibBR] )
		{
			for (int ix = (XParam.blkwidth / 2); ix < XParam.blkwidth; ix++)
			{
				ihalo = ix;
				ibn = ibBR;
				ir = (ix - (XParam.blkwidth / 2)) * 2;

				wetdryrestriction(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
			}
		}

	}

}
template void WetDryRestriction<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, double* zb);
template void WetDryRestriction<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, float* zb);

template <class T> void WetDryProlongationGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	dim3 blockDimHaloLR(1, 16, 1);
	dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	//WetDryProlongationGPUBot

	WetDryProlongationGPULeft <<<gridDim, blockDimHaloLR, 0 >>> (XParam, XBlock, XEv, zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	WetDryProlongationGPURight <<<gridDim, blockDimHaloLR, 0 >>> (XParam, XBlock, XEv, zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	WetDryProlongationGPUTop <<<gridDim, blockDimHaloBT, 0 >>> (XParam, XBlock, XEv, zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	WetDryProlongationGPUBot <<<gridDim, blockDimHaloBT, 0 >>> (XParam, XBlock, XEv, zb);
	CUDA_CHECK(cudaDeviceSynchronize());

}
template void WetDryProlongationGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, double* zb);
template void WetDryProlongationGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, float* zb);

template <class T> void WetDryRestrictionGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	dim3 blockDimHaloLR(1, 16, 1);
	dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	//WetDryProlongationGPUBot

	WetDryRestrictionGPULeft <<<gridDim, blockDimHaloLR, 0 >>> (XParam, XBlock, XEv, zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	WetDryRestrictionGPURight <<<gridDim, blockDimHaloLR, 0 >>> (XParam, XBlock, XEv, zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	WetDryRestrictionGPUTop <<<gridDim, blockDimHaloBT, 0 >>> (XParam, XBlock, XEv, zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	WetDryRestrictionGPUBot <<<gridDim, blockDimHaloBT, 0 >>> (XParam, XBlock, XEv, zb);
	CUDA_CHECK(cudaDeviceSynchronize());

}
template void WetDryRestrictionGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, double* zb);
template void WetDryRestrictionGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, float* zb);

template <class T> __host__ __device__ void ProlongationElevation(int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo,  int ip, int jp, T* h, T* zs, T* zb)
{
	int  halo;
	//pp = memloc(halowidth, blkmemwidth, ip, jp, ibn);
	//ll = memloc(halowidth, blkmemwidth, il, jl, ib);
	//mm = memloc(halowidth, blkmemwidth, im, jm, ibn);

	halo = memloc(halowidth, blkmemwidth, ihalo, jhalo, ib);
	//Check if parent is dry or any of close neighbour
	int ii, left, right, top, bot;
	ii = memloc(halowidth, blkmemwidth, ip, jp, ibn);
	left = memloc(halowidth, blkmemwidth, ip - 1, jp, ibn);
	right = memloc(halowidth, blkmemwidth, ip + 1, jp, ibn);
	top = memloc(halowidth, blkmemwidth, ip, jp + 1, ibn);
	bot = memloc(halowidth, blkmemwidth, ip, jp - 1, ibn);


	//if (!(h[ll] > eps && h[halo]>eps && h[pp] > eps && h[mm] > eps))
	if (!(h[ii] > eps && h[left] > eps && h[right] > eps && h[top] > eps && h[bot] > eps))
	{
		
		//h[halo] = utils::max(T(0.0), zs[pp] - zb[halo]);
		//zs[halo] = h[halo] + zb[halo];
		h[halo] = h[ii];
		zb[halo] = zb[ii];
		zs[halo] = zs[ii];

	}
	


}


template <class T> __host__ __device__ void RevertProlongationElevation(int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int ip, int jp, int level, T dx, T* h, T* zb, T* dzbdx, T* dzbdy)
{
	int  halo;
	//pp = memloc(halowidth, blkmemwidth, ip, jp, ibn);
	//ll = memloc(halowidth, blkmemwidth, il, jl, ib);
	//mm = memloc(halowidth, blkmemwidth, im, jm, ibn);

	halo = memloc(halowidth, blkmemwidth, ihalo, jhalo, ib);
	//Check if parent is dry or any of close neighbour
	int ii, left, right, top, bot;
	ii = memloc(halowidth, blkmemwidth, ip, jp, ibn);
	left = memloc(halowidth, blkmemwidth, ip - 1, jp, ibn);
	right = memloc(halowidth, blkmemwidth, ip + 1, jp, ibn);
	top = memloc(halowidth, blkmemwidth, ip, jp + 1, ibn);
	bot = memloc(halowidth, blkmemwidth, ip, jp - 1, ibn);

	T ilevdx = calcres(dx, level) * T(0.5);

	T facbt, faclr;

	//if (!(h[ll] > eps && h[halo]>eps && h[pp] > eps && h[mm] > eps))
	if (!(h[ii] > eps && h[left] > eps && h[right] > eps && h[top] > eps && h[bot] > eps))
	{
		if (ihalo == -1)
		{
			faclr = 1.0;
			facbt = floor(jhalo * (T)0.5) * T(2.0) < (jhalo - T(0.01)) ? 1.0 : -1.0;
		}
		else if (ihalo == 16)
		{
			faclr = -1.0;
			facbt = floor(jhalo * (T)0.5) * T(2.0) < (jhalo - T(0.01)) ? 1.0 : -1.0;
		}
		if (jhalo == -1)
		{
			facbt = 1.0;
			facbt = floor(ihalo * (T)0.5) * T(2.0) < (ihalo - T(0.01)) ? 1.0 : -1.0;
		}
		else if (jhalo == 16)
		{
			facbt = -1.0;
			facbt = floor(ihalo * (T)0.5) * T(2.0) < (ihalo - T(0.01)) ? 1.0 : -1.0;
		}

		//h[halo] = utils::max(T(0.0), zs[pp] - zb[halo]);
		//zs[halo] = h[halo] + zb[halo];

		zb[halo] = zb[ii] + (faclr * dzbdx[ii] + facbt * dzbdy[ii]) * ilevdx;


	}



}

template <class T> __host__ __device__ void ProlongationElevationGH(int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int ip, int jp, T* h, T* dhdx, T* dzsdx)
{
	int halo;
	//pp = memloc(halowidth, blkmemwidth, ip, jp, ibn);
	//ll = memloc(halowidth, blkmemwidth, il, jl, ib);
	//mm = memloc(halowidth, blkmemwidth, im, jm, ibn);

	halo = memloc(halowidth, blkmemwidth, ihalo, jhalo, ib);
	//Check if parent is dry or any of close neighbour

	int ii, left, right, top, bot;
	ii = memloc(halowidth, blkmemwidth, ip, jp, ibn);
	left = memloc(halowidth, blkmemwidth, ip - 1, jp, ibn);
	right = memloc(halowidth, blkmemwidth, ip + 1, jp, ibn);
	top = memloc(halowidth, blkmemwidth, ip, jp + 1, ibn);
	bot = memloc(halowidth, blkmemwidth, ip, jp - 1, ibn);
	
	//if (!(h[ll] > eps && h[halo] > eps && h[pp] > eps && h[mm] > eps))
	if (!(h[ii] > eps && h[left] > eps && h[right] > eps && h[top] > eps && h[bot] > eps))
	{

		dhdx[halo] = T(0.0);
		dzsdx[halo] = T(0.0);
	}



}

template <class T> __host__ __device__ void conserveElevation(int halowidth,int blkmemwidth,T eps, int ib, int ibn,int ihalo, int jhalo ,int i,int j, T* h, T* zs, T * zb)
{
	int ii, ir, it, itr;
	T iiwet, irwet, itwet, itrwet;
	T zswet, hwet;

	int write;

	write = memloc(halowidth, blkmemwidth, ihalo, jhalo, ib);
	
	ii = memloc(halowidth, blkmemwidth, i, j, ibn);
	ir = memloc(halowidth, blkmemwidth, i + 1, j, ibn);
	it = memloc(halowidth, blkmemwidth, i, j + 1, ibn);
	itr = memloc(halowidth, blkmemwidth, i + 1, j + 1, ibn);

	iiwet = h[ii] > eps ? h[ii] : T(0.0);
	irwet = h[ir] > eps ? h[ir] : T(0.0);
	itwet = h[it] > eps ? h[it] : T(0.0);
	itrwet = h[itr] > eps ? h[itr] : T(0.0);

	hwet = (iiwet + irwet + itwet + itrwet);
	zswet = iiwet * (zb[ii] + h[ii]) + irwet * (zb[ir] + h[ir]) + itwet * (zb[it] + h[it]) + itrwet * (zb[itr] + h[itr]);

	//conserveElevation(zb[write], zswet, hwet);
	if (hwet > T(0.0))
	{
		zswet = zswet / hwet;
		hwet = utils::max(T(0.0), zswet - zb[write]);

	}
	else
	{
		hwet = T(0.0);

	}

	//zswet = hwet + zb;

	h[write] = hwet;
	zs[write] = hwet + zb[write];


}
template __host__ __device__ void conserveElevation<float>(int halowidth, int blkmemwidth, float eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, float* h, float* zs, float* zb);
template __host__ __device__ void conserveElevation<double>(int halowidth, int blkmemwidth, double eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, double* h, double* zs, double* zb);

template <class T> __host__ __device__ void wetdryrestriction(int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, T* h, T* zs, T* zb)
{
	int ii, ir, it, itr;
	T iiwet, irwet, itwet, itrwet;
	T zswet, hwet, cwet, zbw;

	int write;

	write = memloc(halowidth, blkmemwidth, ihalo, jhalo, ib);

	ii = memloc(halowidth, blkmemwidth, i, j, ibn);
	ir = memloc(halowidth, blkmemwidth, i + 1, j, ibn);
	it = memloc(halowidth, blkmemwidth, i, j + 1, ibn);
	itr = memloc(halowidth, blkmemwidth, i + 1, j + 1, ibn);

	T hii, hir, hit, hitr;

	hii = h[ii];
	hir = h[ir];
	hit = h[it];
	hitr = h[itr];

	zbw = zb[write];

	iiwet = hii > eps ? T(1.0) : T(0.0);
	irwet = hir > eps ? T(1.0) : T(0.0);
	itwet = hit > eps ? T(1.0) : T(0.0);
	itrwet = hitr > eps ? T(1.0) : T(0.0);

	cwet = (iiwet + irwet + itwet + itrwet);
	hwet = (iiwet*hii + irwet*hir + itwet*hit + itrwet*hitr);
	zswet = (iiwet*hii) * (zb[ii] + h[ii]) + (irwet*hir) * (zb[ir] + h[ir]) + (itwet*hit) * (zb[it] + h[it]) + (itrwet * hitr) * (zb[itr] + h[itr]);

	//conserveElevation(zb[write], zswet, hwet);
	if (cwet > T(0.0) && cwet < T(4.0))
	{
		zswet = zswet / hwet;
		hwet = utils::max(T(0.0), zswet - zbw);


		h[write] = hwet;
		//zs[write] = hwet + zbw;

	}
	

	//zswet = hwet + zb;



}
template __host__ __device__ void wetdryrestriction<float>(int halowidth, int blkmemwidth, float eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, float* h, float* zs, float* zb);
template __host__ __device__ void wetdryrestriction<double>(int halowidth, int blkmemwidth, double eps, int ib, int ibn, int ihalo, int jhalo, int i, int j, double* h, double* zs, double* zb);




template <class T> __host__ __device__ void conserveElevation(T zb, T& zswet, T& hwet)
{
	
	if (hwet > 0.0)
	{
		zswet = zswet / hwet;
		hwet = utils::max(T(0.0), zswet - zb);

	}
	else
	{
		hwet = T(0.0);
		
	}

	zswet = hwet + zb;
}

template <class T> void conserveElevationGradHalo(Param XParam, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx, T* dhdy, T* dzsdy)
{
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		conserveElevationGHLeft(XParam, ib, XBlock.LeftBot[ib], XBlock.LeftTop[ib], XBlock, h, zs, zb, dhdx, dzsdx);
		conserveElevationGHRight(XParam, ib, XBlock.RightBot[ib], XBlock.RightTop[ib], XBlock, h, zs, zb, dhdx, dzsdx);
		conserveElevationGHTop(XParam, ib, XBlock.TopLeft[ib], XBlock.TopRight[ib], XBlock, h, zs, zb, dhdy, dzsdy);
		conserveElevationGHBot(XParam, ib, XBlock.BotLeft[ib], XBlock.BotRight[ib], XBlock, h, zs, zb, dhdy, dzsdy);
	}
}
template void conserveElevationGradHalo<float>(Param XParam, BlockP<float> XBlock, float* h, float* zs, float* zb, float* dhdx, float* dzsdx, float* dhdy, float* dzsdy);
template void conserveElevationGradHalo<double>(Param XParam, BlockP<double> XBlock, double* h, double* zs, double* zb, double* dhdx, double* dzsdx, double* dhdy, double* dzsdy);

template <class T> void conserveElevationGradHaloGPU(Param XParam, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx, T* dhdy, T* dzsdy)
{
	dim3 blockDimHaloLR(1, 16, 1);
	dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	conserveElevationGHLeft <<<gridDim, blockDimHaloLR, 0 >>> (XParam, XBlock, h, zs, zb, dhdx, dzsdx);
	CUDA_CHECK(cudaDeviceSynchronize());

	conserveElevationGHRight <<<gridDim, blockDimHaloLR, 0 >>> (XParam, XBlock, h, zs, zb, dhdx, dzsdx);
	CUDA_CHECK(cudaDeviceSynchronize());

	conserveElevationGHTop <<<gridDim, blockDimHaloBT, 0 >>> (XParam, XBlock, h, zs, zb, dhdy, dzsdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	conserveElevationGHBot <<<gridDim, blockDimHaloBT, 0 >>> (XParam, XBlock, h, zs, zb, dhdy, dzsdy);
	CUDA_CHECK(cudaDeviceSynchronize());
	
}
template void conserveElevationGradHaloGPU<float>(Param XParam, BlockP<float> XBlock, float* h, float* zs, float* zb, float* dhdx, float* dzsdx, float* dhdy, float* dzsdy);
template void conserveElevationGradHaloGPU<double>(Param XParam, BlockP<double> XBlock, double* h, double* zs, double* zb, double* dhdx, double* dzsdx, double* dhdy, double* dzsdy);

template <class T> __host__ __device__ void conserveElevationGradHalo(int halowidth, int blkmemwidth, T eps, int ib, int ibn, int ihalo, int jhalo,int i, int j, T* h, T* dhdx, T* dhdy)
{
	int ii, ir, it, itr, jj;
	int write;
	write = memloc(halowidth, blkmemwidth, ihalo, jhalo, ib);

	ii = memloc(halowidth, blkmemwidth, i, j, ibn);
	ir = memloc(halowidth, blkmemwidth, i + 1, j, ibn);
	it = memloc(halowidth, blkmemwidth, i, j + 1, ibn);
	itr = memloc(halowidth, blkmemwidth, i + 1, j + 1, ibn);

	if (h[write] <= eps)
	{
		// Because of the slope limiter the average slope is not the slope of the averaged values
		// It seems that it should be the closest to zero instead... With conserve elevation This will work but maybe all prolongation need to be applied this way (?)
		dhdy[write] = utils::nearest(utils::nearest(utils::nearest(dhdy[ii], dhdy[ir]), dhdy[it]), dhdy[itr]);
		dhdx[write] = utils::nearest(utils::nearest(utils::nearest(dhdx[ii], dhdx[ir]), dhdx[it]), dhdx[itr]);
	}
}


template <class T> __host__ __device__ void conserveElevationGradHaloA(int halowidth, int blkmemwidth, int ib, int ibn, int ihalo, int jhalo, int ip, int jp, int iq, int jq, T theta, T delta, T* h, T* dhdx)
{
	//int pii, pir, pit, pitr;
	int qii, qir, qit, qitr;

	T p, q;
	T s0, s1, s2;

	int write, pii;
	write = memloc(halowidth, blkmemwidth, ihalo, jhalo, ib);
	pii = memloc(halowidth, blkmemwidth, ip, jp, ib);




	//pii = memloc(halowidth, blkmemwidth, ip, jp, ibn);
	//pir = memloc(halowidth, blkmemwidth, ip + 1, jp, ibn);
	//pit = memloc(halowidth, blkmemwidth, ip, jp + 1, ibn);
	//pitr = memloc(halowidth, blkmemwidth, ip + 1, jp + 1, ibn);

	qii = memloc(halowidth, blkmemwidth, iq, jq, ibn);
	qir = memloc(halowidth, blkmemwidth, iq + 1, jq, ibn);
	qit = memloc(halowidth, blkmemwidth, iq, jq + 1, ibn);
	qitr = memloc(halowidth, blkmemwidth, iq + 1, jq + 1, ibn);

	s1 = h[write];
	p = h[pii];
	q = T(0.25) * (h[qii] + h[qir] + h[qit] + h[qitr]);



	if (ip > ihalo || jp > jhalo)
	{
		s0 = q;
		s2 = p;
	}
	else
	{
		s2 = q;
		s0 = p;
	}

	dhdx[write] = minmod2(theta, s0, s1, s2) / delta;
	//dhdx[write] = utils::nearest(utils::nearest(utils::nearest(dhdx[ii], dhdx[ir]), dhdx[it]), dhdx[itr]);

}

template <class T> __host__ __device__ void conserveElevationGradHaloB(int halowidth, int blkmemwidth, int ib, int ibn, int ihalo, int jhalo, int ip, int jp, int iq, int jq, T theta, T delta, T eps, T* h, T* zs, T* zb, T* dhdx, T* dzsdx)
{
	//int pii, pir, pit, pitr;
	int qii, qir, qit, qitr;
	
	T hp, hq,zsp,zsq, zbq;
	T hs0, hs1, hs2,zss0, zss1, zss2;

	T hwet, zswet;
	int write, pii;
	T iiwet, irwet, itwet, itrwet;
	write = memloc(halowidth, blkmemwidth, ihalo, jhalo, ib);
	pii = memloc(halowidth, blkmemwidth, ip, jp, ib);
	
	//pii = memloc(halowidth, blkmemwidth, ip, jp, ibn);
	//pir = memloc(halowidth, blkmemwidth, ip + 1, jp, ibn);
	//pit = memloc(halowidth, blkmemwidth, ip, jp + 1, ibn);
	//pitr = memloc(halowidth, blkmemwidth, ip + 1, jp + 1, ibn);

	qii = memloc(halowidth, blkmemwidth, iq, jq, ibn);
	qir = memloc(halowidth, blkmemwidth, iq + 1, jq, ibn);
	qit = memloc(halowidth, blkmemwidth, iq, jq + 1, ibn);
	qitr = memloc(halowidth, blkmemwidth, iq + 1, jq + 1, ibn);

	

	
	zbq = T(0.25) * (zb[qii] + zb[qir] + zb[qit] + zb[qitr]);

	iiwet = h[qii] > eps ? h[qii] : T(0.0);
	irwet = h[qir] > eps ? h[qir] : T(0.0);
	itwet = h[qit] > eps ? h[qit] : T(0.0);
	itrwet = h[qitr] > eps ? h[qitr] : T(0.0);

	hwet = T(iiwet + irwet + itwet + itrwet);
	zswet = iiwet * (zb[qii] + h[qii]) + irwet * (zb[qir] + h[qir]) + itwet * (zb[qit] + h[qit]) + itrwet * (zb[qitr] + h[qitr]);
	
	if (hwet > T(0.0))
	{
		zswet = zswet / hwet;
		hq = utils::max(T(0.0), zswet - zbq);
		
	}
	else
	{
		hq = T(0.0);
	}

	hs1 = h[write];
	zss1= zs[write];
	hp = h[pii];
	zsp = zs[pii];
	zsq = hq + zbq;

	if (ip > ihalo || jp > jhalo )
	{
		hs0 = hq;
		hs2 = hp;
		zss0 = zsq;
		zss2 = zsp;
	}
	else
	{
		hs2 = hq;
		hs0 = hp;
		zss2 = zsq;
		zss0 = zsp;
	}

	dhdx[write] = minmod2(theta,hs0,hs1,hs2)/ delta;
	dzsdx[write] = minmod2(theta, zss0, zss1, zss2) / delta;
	//dhdx[write] = utils::nearest(utils::nearest(utils::nearest(dhdx[ii], dhdx[ir]), dhdx[it]), dhdx[itr]);
	
}

template <class T> void conserveElevationGHLeft(Param XParam, int ib, int ibLB, int ibLT, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx)
{
	int ibn;
	int ihalo, jhalo, ip, jp, iq, jq;
	T delta = calcres(T(XParam.delta), XBlock.level[ib]);
	ihalo = -1;
	ip = 0;


	if (XBlock.level[ib] < XBlock.level[ibLB])
	{

		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{
			jhalo = j;
			jp = j;
			iq = XParam.blkwidth - 4;
			jq = j * 2;
			ibn = ibLB;
			conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
			//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);
			//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibLB,  -1, j, XParam.blkwidth - 2, j * 2, h, dhdx, dhdy);
		}
	}
	if (XBlock.level[ib] < XBlock.level[ibLT])
	{
		for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
		{
			jhalo = j;
			jp = j;
			iq = XParam.blkwidth - 4;
			jq = (j - (XParam.blkwidth / 2)) * 2;
			ibn = ibLT;

			conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
			//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);

			//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibLT, -1, j, XParam.blkwidth - 2, (j - (XParam.blkwidth / 2)) * 2, h, dhdx, dhdy);
		}
	}

	// Prolongation part
	//int il, jl, im, jm;
	ihalo = -1;

	if (XBlock.level[ib] > XBlock.level[ibLB])
	{
		//
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//
			jhalo = j;
			ibn = ibLB;

			//il = 0;
			//jl = j;

			ip = XParam.blkwidth - 1;
			jp = XBlock.RightBot[ibLB] == ib ? ftoi(floor(j / 2)) : ftoi(floor(j / 2) + XParam.blkwidth / 2);

			//im = ip;
			//jm = ceil(j * T(0.5)) * 2 > j ? jp + 1 : jp - 1;

			ProlongationElevationGH(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, h, dhdx, dzsdx);
		}

	}

	

}

template <class T> __global__ void conserveElevationGHLeft(Param XParam, BlockP<T> XBlock, T* h, T*zs, T*zb, T* dhdx, T* dzsdx)
{
	
	
	
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int LB = XBlock.LeftBot[ib];
	int LT = XBlock.LeftTop[ib];

	int ip, jp, iq, jq;

	int ihalo, jhalo, ibn;
	T delta = calcres(XParam.delta, lev);


	ihalo = -1;
	jhalo = iy;
	iq = XParam.blkwidth - 4;
	ip = 0;
	jp = iy;
	if (lev < XBlock.level[LB] && iy < (blockDim.y / 2))
	{
		ibn = LB;
		jq = iy * 2;
		conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
		//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);

		//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}
	if (lev < XBlock.level[LT] && iy >= (blockDim.y / 2))
	{
		ibn = LT;
		jq = (iy - (blockDim.y / 2)) * 2;
		conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
		//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
		//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);
	}

	// Prolongation part
	//int il, jl, im, jm;
	

	if (XBlock.level[ib] > XBlock.level[LB])
	{
		//
		//
		
		ibn = LB;

		//il = 0;
		//jl = iy;

		ip = blockDim.y - 1;
		jp = XBlock.RightBot[LB] == ib ? int(floor(iy *T(0.5))) : int((floor(iy * T(0.5)) + blockDim.y / 2));
		//im = ip;
		//jm = ceil(iy * T(0.5)) * 2 > iy ? jp + 1 : jp - 1;

		ProlongationElevationGH(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, h, dhdx, dzsdx);
		

	}
}

template <class T> void conserveElevationGHRight(Param XParam, int ib, int ibRB, int ibRT, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx)
{
	int ibn;
	int ihalo, jhalo, ip, jp, iq, jq;
	T delta = calcres(T(XParam.delta), XBlock.level[ib]);
	ihalo = XParam.blkwidth;
	ip = XParam.blkwidth-1;

	if (XBlock.level[ib] < XBlock.level[ibRB])
	{
		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{
			jhalo = j;
			jp = j;
			iq = 2;
			jq = j * 2;
			ibn = ibRB;
			conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
			//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);
			//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibRB, XParam.blkwidth, j, 0, j * 2, h, dhdx, dhdy);
		}
	}
	if (XBlock.level[ib] < XBlock.level[ibRT])
	{
		for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
		{
			jhalo = j;
			jp = j;
			iq = 2;
			jq = (j - (XParam.blkwidth / 2)) * 2;
			ibn = ibRT;
			conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
			//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);

			//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibRT, XParam.blkwidth, j, 0, (j - (XParam.blkwidth / 2)) * 2, h, dhdx, dhdy);
		}
	}

	// Prolongation part
	//int il, jl, im, jm;
	

	if (XBlock.level[ib] > XBlock.level[ibRB])
	{
		//
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//
			jhalo = j;
			ibn = ibRB;

			//il = XParam.blkwidth-2;
			//jl = j;

			ip = 0;
			jp = XBlock.LeftBot[ibRB] == ib ? ftoi(floor(j / 2)) : ftoi(floor(j / 2) + XParam.blkwidth / 2);
			//im = ip;
			//jm = ceil(j * T(0.5)) * 2 > j ? jp + 1 : jp - 1;

			ProlongationElevationGH(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, h, dhdx, dzsdx);
		}

	}
}

template <class T> __global__ void conserveElevationGHRight(Param XParam, BlockP<T> XBlock, T* h, T*zs, T*zb, T* dhdx, T* dzsdx)
{
	
	
	
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int RB = XBlock.RightBot[ib];
	int RT = XBlock.RightTop[ib];

	

	int ihalo, jhalo, iq, jq, ip, jp, ibn;

	T delta = calcres(XParam.delta, lev);

	ihalo = blockDim.y;
	jhalo = iy;
	iq = blockDim.y - 4;
	ip = blockDim.y-1;
	jp = iy;

	if (XBlock.level[ib] < XBlock.level[RB] && iy < (blockDim.y / 2))
	{
		ibn = RB;
		jq = iy * 2;
		conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
		//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);

		//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}
	if (XBlock.level[ib] < XBlock.level[RT] && iy >= (blockDim.y / 2))
	{
		ibn = RT;
		jq = (iy - (XParam.blkwidth / 2)) * 2;
		conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
		//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);
		//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}

	// Prolongation part
	//int il, jl, im, jm;


	if (XBlock.level[ib] > XBlock.level[RB])
	{
		//
		
		//
		jhalo = iy;
		ibn = RB;

		//il = blockDim.y - 2;
		//jl = iy;

		ip = 0;
		jp = XBlock.LeftBot[RB] == ib ? int(floor(iy * T(0.5))) : int((floor(iy *T (0.5)) + blockDim.y / 2));
		
		//im = ip;
		//jm = ceil(iy * T(0.5)) * 2 > iy ? jp + 1 : jp - 1;

		ProlongationElevationGH(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, h, dhdx, dzsdx);
		

	}
}

template <class T> void conserveElevationGHTop(Param XParam, int ib, int ibTL, int ibTR, BlockP<T> XBlock, T* h, T*zs, T*zb, T* dhdx, T* dzsdx)
{
	int ibn;
	int ihalo, jhalo, ip, jp, iq, jq;
	T delta = calcres(T(XParam.delta), XBlock.level[ib]);
	jhalo = XParam.blkwidth;
	jp = XParam.blkwidth - 1;

	if (XBlock.level[ib] < XBlock.level[ibTL])
	{
		for (int i = 0; i < XParam.blkwidth / 2; i++)
		{
			ihalo = i;
			ip = i;
			jq = 2;
			iq = i * 2;
			ibn = ibTL;
			conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
			//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibTL, i, XParam.blkwidth, i * 2, 0, h, dhdx, dhdy);
		}
	}
	if (XBlock.level[ib] < XBlock.level[ibTR])
	{
		for (int i = (XParam.blkwidth / 2); i < (XParam.blkwidth); i++)
		{
			ihalo = i;
			ip = i;
			jq = 2;
			iq = (i - (XParam.blkwidth / 2)) * 2;
			ibn = ibTR;
			conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
			//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);

			//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibTR, i, XParam.blkwidth, (i - (XParam.blkwidth / 2)) * 2, 0, h, dhdx, dhdy);
		}
	}

	// Prolongation part
	//int il, jl, im, jm;


	if (XBlock.level[ib] > XBlock.level[ibTL])
	{
		//
		for (int i = 0; i < XParam.blkwidth; i++)
		{
			//
			ihalo = i;
			ibn = ibTL;

			//jl = XParam.blkwidth - 2;
			//il = i;

			jp = 0;
			ip = XBlock.BotLeft[ibTL] == ib ? ftoi(floor(i / 2)) : ftoi(floor(i / 2) + XParam.blkwidth / 2);
			//jm = jp;
			//im = ceil(i * T(0.5)) * 2 > i ? ip + 1 : ip - 1;

			ProlongationElevationGH(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, h, dhdx, dzsdx);
		}

	}
}

template <class T> __global__ void conserveElevationGHTop(Param XParam, BlockP<T> XBlock, T* h, T*zs, T*zb, T* dhdx, T* dzsdx)
{
	
	
	int iy = blockDim.x - 1;
	int ix = threadIdx.x;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int TL = XBlock.TopLeft[ib];
	int TR = XBlock.TopRight[ib];
	


	int ihalo, jhalo, iq, jq, ip, jp, ibn;
	T delta = calcres(XParam.delta, lev);

	ihalo = ix;
	jhalo = iy+1;
	jp = iy;
	ip = ix;
	jq = 2;

	if (XBlock.level[ib] < XBlock.level[TL] && ix < (blockDim.x / 2))
	{
		ibn = TL;
		iq = ix * 2;
		conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
		//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);

		//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}
	if (XBlock.level[ib] < XBlock.level[TR] && ix >= (blockDim.x / 2))
	{
		ibn = TR;
		iq = (ix - (blockDim.x / 2)) * 2;;
		conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
		//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);
		//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}

	// Prolongation part
	//int il, jl, im, jm;


	if (XBlock.level[ib] > XBlock.level[TL])
	{
		//
		//
		//ihalo = i;
		ibn = TL;

		//jl = blockDim.x - 2;
		//il = ix;

		jp = 0;
		ip = XBlock.BotLeft[TL] == ib ? int(floor(ix *T(0.0))) : int((floor(ix * T(0.0)) + blockDim.x / 2));
		//jm = jp;
		//im = ceil(ix * T(0.5)) * 2 > ix ? ip + 1 : ip - 1;

		ProlongationElevationGH(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, h, dhdx, dzsdx);
		

	}
}

template <class T> void conserveElevationGHBot(Param XParam, int ib, int ibBL, int ibBR, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx)
{
	int ibn;
	int ihalo, jhalo, ip, jp, iq, jq;
	T delta = calcres(T(XParam.delta), XBlock.level[ib]);
	jhalo = -1;
	jp = 0;

	if (XBlock.level[ib] < XBlock.level[ibBL])
	{
		for (int i = 0; i < XParam.blkwidth / 2; i++)
		{
			ihalo = i;
			ip = i;
			iq = i * 2;
			jq = XParam.blkwidth - 4;
			ibn = ibBL;
			conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
			//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);

			//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibBL, i, -1, i * 2, XParam.blkwidth - 2, h, dhdx, dhdy);
		}
	}
	if (XBlock.level[ib] < XBlock.level[ibBR])
	{
		for (int i = (XParam.blkwidth / 2); i < (XParam.blkwidth); i++)
		{
			ihalo = i;
			ip = i;
			iq = (i - (XParam.blkwidth / 2)) * 2;;
			jq = XParam.blkwidth - 4;
			ibn = ibBR;
			conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
			//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);

			//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibBR, i, -1, (i - (XParam.blkwidth / 2)) * 2, XParam.blkwidth - 2, h, dhdx, dhdy);
		}
	}
	// Prolongation part
	//int il, jl, im, jm;


	if (XBlock.level[ib] > XBlock.level[ibBL])
	{
		//
		for (int i = 0; i < XParam.blkwidth; i++)
		{
			//
			ihalo = i;
			ibn = ibBL;

			//jl = 0;
			//il = i;

			jp = XParam.blkwidth - 1;
			ip = XBlock.TopLeft[ibBL] == ib ? ftoi(floor(i / 2)) : ftoi(floor(i / 2) + XParam.blkwidth / 2);
			//jm = jp;
			//im = ceil(i * T(0.5)) * 2 > i ? ip + 1 : ip - 1;

			ProlongationElevationGH(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, h, dhdx, dzsdx);
		}

	}
}

template <class T> __global__ void conserveElevationGHBot(Param XParam, BlockP<T> XBlock, T* h, T* zs, T* zb, T* dhdx, T* dzsdx)
{
	
	
	
	int ix = threadIdx.x;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int BL = XBlock.BotLeft[ib];
	int BR = XBlock.BotRight[ib];

	int ip, jp, iq, jq;

	int ihalo, jhalo, ibn;
	T delta = calcres(XParam.delta, lev);

	ihalo = ix;
	jhalo = -1;
	jq = XParam.blkwidth - 4;
	jp = 0;
	ip = ix;

	if (XBlock.level[ib] < XBlock.level[BL] && ix < (blockDim.x / 2))
	{
		ibn = BL;
		iq = ix * 2;
		conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
		//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);
		//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}
	if (XBlock.level[ib] < XBlock.level[BR] && ix >= (blockDim.x / 2))
	{
		ibn = BR;
		iq = (ix - (blockDim.x / 2)) * 2;
		conserveElevationGradHaloB(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, T(XParam.eps), h, zs, zb, dhdx, dzsdx);
		//conserveElevationGradHaloA(XParam.halowidth, XParam.blkmemwidth, ib, ibn, ihalo, jhalo, ip, jp, iq, jq, T(XParam.theta), delta, h, dhdx);
		//conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}

	// Prolongation part
	//int il, jl, im, jm;


	if (XBlock.level[ib] > XBlock.level[BL])
	{
		//
		
		ihalo = ix;
		ibn = BL;

		//jl = 0;
		//il = ix;

		jp = blockDim.x - 1;
		ip = XBlock.TopLeft[BL] == ib ? int(floor(ix * T(0.0))) : int((floor(ix * T(0.0)) + blockDim.x / 2));

		//jm = jp;
		//im = ceil(ix * T(0.5)) * 2 > ix ? ip + 1 : ip - 1;

		ProlongationElevationGH(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, h, dhdx, dzsdx);
		

	}
}

template <class T> void conserveElevationLeft(Param XParam,int ib, int ibLB, int ibLT, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int ihalo,jhalo,ibn,ip, jp;
	
	// Restriction
	ihalo = -1;
	ip = XParam.blkwidth - 2;

	//int ii = memloc(XParam, -1, 5, 46);
	if (XBlock.level[ib] < XBlock.level[ibLB])
	{
		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{
			jhalo = j;
			jp = j * 2;
			ibn = ibLB;
			conserveElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);
		}

	}
	
	if (XBlock.level[ib] < XBlock.level[ibLT])
	{
		for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
		{
			jhalo = j;
			jp = (j - (XParam.blkwidth / 2)) * 2;
			ibn = ibLT;
			conserveElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);
		}

	}
	
	// Prolongation
	
	ihalo = -1;

	if (XBlock.level[ib] > XBlock.level[ibLB])
	{
		//
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//
			jhalo = j;
			ibn = ibLB;

			//il = 0;
			//jl = j;

			ip = XParam.blkwidth - 1;
			jp = XBlock.RightBot[ibLB] == ib ? ftoi(floor(j * T(0.5))) : ftoi(floor(j * T(0.5)) + XParam.blkwidth / 2);

			//im = ip;
			//jm = ceil(j * T(0.5)) * 2 > j ? jp + 1 : jp - 1;

			ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);
		}

	}
	
}

template <class T> __global__ void conserveElevationLeft(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int blkmemwidth = blockDim.y + XParam.halowidth * 2;
	
	
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int LB = XBlock.LeftBot[ib];
	int LT = XBlock.LeftTop[ib];

	

	int ihalo , jhalo, i, j, ibn;

	ihalo = -1;
	jhalo = iy;
	i = XParam.blkwidth - 2;

	if (lev < XBlock.level[LB] && iy < (blockDim.y / 2))
	{
		ibn = LB;
		j = iy * 2;

		conserveElevation(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, XEv.h, XEv.zs, zb);
	}
	if (lev < XBlock.level[LT] && iy >= (blockDim.y / 2))
	{
		ibn = LT;
		j = (iy - (blockDim.y / 2)) * 2;

		conserveElevation(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, XEv.h, XEv.zs, zb);
	}

	// Prolongation
	int ip, jp;
	ihalo = -1;

	if (XBlock.level[ib] > XBlock.level[LB])
	{
		//
		
		jhalo = iy;
		ibn = LB;

		//il = 0;
		//jl = iy;

		ip = XParam.blkwidth - 1;
		jp = XBlock.RightBot[ibn] == ib ? floor(iy * T(0.5)) : (floor(iy * T(0.5)) + blockDim.y / 2);


		//im = ip;
		//jm = ceil(iy * T(0.5)) * 2 > iy ? jp + 1 : jp - 1;

		ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);
		

	}
}

template <class T> __global__ void WetDryProlongationGPULeft(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int blkmemwidth = blockDim.y + XParam.halowidth * 2;


	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int LB = XBlock.LeftBot[ib];
	int LT = XBlock.LeftTop[ib];
	
	int ip, jp, ihalo, jhalo, ibn;
	
	jhalo = iy;
	ihalo = -1;


	if (lev > XBlock.level[LB])
	{
		//

		ibn = LB;
		

		//il = 0;
		//jl = iy;

		ip = XParam.blkwidth - 1;
		jp = XBlock.RightBot[ibn] == ib ? floor(iy * T(0.5)) : (floor(iy * T(0.5)) + blockDim.y / 2);


		//im = ip;
		//jm = ceil(iy * T(0.5)) * 2 > iy ? jp + 1 : jp - 1;

		ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);


	}
}

template <class T> __global__ void WetDryRestrictionGPULeft(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int blkmemwidth = blockDim.y + XParam.halowidth * 2;


	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int LB = XBlock.LeftBot[ib];
	int LT = XBlock.LeftTop[ib];

	int ihalo, jhalo, ibn, ir, jr;

	jhalo = iy;
	ihalo = -1;

	ir = XParam.blkwidth - 2;

	if (lev < XBlock.level[LB] && iy < (blockDim.y / 2))
	{
		ibn = LB;
		jr = iy * 2;

		wetdryrestriction(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
	}
	if (lev < XBlock.level[LT] && iy >= (blockDim.y / 2))
	{
		ibn = LT;
		jr = (iy - (blockDim.y / 2)) * 2;

		wetdryrestriction(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
	}


}


template <class T> void conserveElevationRight(Param XParam, int ib, int ibRB, int ibRT, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int ihalo, jhalo, ibn, ip, jp;

	if (XBlock.level[ib] < XBlock.level[ibRB])
	{
		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{
			conserveElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibRB, XParam.blkwidth, j, 0, j*2, XEv.h, XEv.zs, zb);
		}

	}
	if (XBlock.level[ib] < XBlock.level[ibRT])
	{
		for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
		{
			conserveElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibRT, XParam.blkwidth, j, 0, (j - (XParam.blkwidth / 2)) * 2, XEv.h, XEv.zs, zb);
		}

	}

	// Prolongation
	
	ihalo = XParam.blkwidth;

	if (XBlock.level[ib] > XBlock.level[ibRB])
	{
		//
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//
			jhalo = j;
			ibn = ibRB;

			//il = XParam.blkwidth-1;
			//jl = j;

			ip = 0;
			jp = XBlock.LeftBot[ibn] == ib ? ftoi(floor(j * T(0.5))) : ftoi(floor(j * T(0.5)) + XParam.blkwidth / 2);
			//im = ip;
			//jm = ceil(j * T(0.5)) * 2 > j ? jp + 1 : jp - 1;

			ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo,  ip, jp, XEv.h, XEv.zs, zb);
		}

	}
}

template <class T> __global__ void conserveElevationRight(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int blkmemwidth = blockDim.y + XParam.halowidth * 2;
	
	
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int RB = XBlock.RightBot[ib];
	int RT = XBlock.RightTop[ib];

	
	

	int ihalo, jhalo, i, j, ibn;

	ihalo = blockDim.y;
	jhalo = iy;

	i = 0;

	if (lev < XBlock.level[RB] && iy < (blockDim.y / 2))
	{
		ibn = RB;
		j = iy * 2;

		conserveElevation(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, XEv.h, XEv.zs, zb);
	}
	if (lev < XBlock.level[RT] && iy >= (blockDim.y / 2))
	{
		ibn = RT;
		j = (iy - (blockDim.y / 2)) * 2;

		conserveElevation(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, XEv.h, XEv.zs, zb);
	}
	
	// Prolongation
	int ip, jp;
	//ihalo = -1;

	if (lev > XBlock.level[RB])
	{
		//

		jhalo = iy;
		ibn = RB;

		//il = blockDim.y - 1;
		//jl = iy;

		ip = 0;
		jp = XBlock.LeftBot[ibn] == ib ? floor(iy * T(0.5)) : (floor(iy * T(0.5)) + blockDim.y / 2);

		//im = ip;
		//jm = ceil(iy * T(0.5)) * 2 > iy ? jp + 1 : jp - 1;

		ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);


	}
	
}

template <class T> __global__ void WetDryProlongationGPURight(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int blkmemwidth = blockDim.y + XParam.halowidth * 2;


	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int RB = XBlock.RightBot[ib];
	int RT = XBlock.RightTop[ib];

	int ip, jp, ihalo, jhalo, ibn;
	
	ihalo = blockDim.y;
	jhalo = iy;


	if (lev > XBlock.level[RB])
	{
		//

		
		ibn = RB;

		//il = blockDim.y - 1;
		//jl = iy;

		ip = 0;
		jp = XBlock.LeftBot[ibn] == ib ? floor(iy * T(0.5)) : (floor(iy * T(0.5)) + blockDim.y / 2);

		//im = ip;
		//jm = ceil(iy * T(0.5)) * 2 > iy ? jp + 1 : jp - 1;

		ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);


	}

}

template <class T> __global__ void WetDryRestrictionGPURight(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int blkmemwidth = blockDim.y + XParam.halowidth * 2;


	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int RB = XBlock.RightBot[ib];
	int RT = XBlock.RightTop[ib];

	int ihalo, jhalo, ibn, ir, jr;

	ihalo = blockDim.y;
	jhalo = iy;

	ir = 0;

	if (lev < XBlock.level[RB] && iy < (blockDim.y / 2))
	{
		ibn = RB;
		jr = iy * 2;

		wetdryrestriction(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
	}
	if (lev < XBlock.level[RT] && iy >= (blockDim.y / 2))
	{
		ibn = RT;
		jr = (iy - (blockDim.y / 2)) * 2;

		wetdryrestriction(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
	}

	

}

template <class T> void conserveElevationTop(Param XParam, int ib, int ibTL, int ibTR, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int ihalo, jhalo, ibn, ip, jp;

	

	if (XBlock.level[ib] < XBlock.level[ibTL])
	{
		for (int i = 0; i < XParam.blkwidth / 2; i++)
		{
			conserveElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibTL, i, XParam.blkwidth, i*2, 0, XEv.h, XEv.zs, zb);
		}

	}
	if (XBlock.level[ib] < XBlock.level[ibTR])
	{
		for (int i = (XParam.blkwidth / 2); i < (XParam.blkwidth); i++)
		{
			conserveElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibTR, i, XParam.blkwidth, (i - (XParam.blkwidth / 2)) * 2, 0, XEv.h, XEv.zs, zb);
		}

	}

	// Prolongation
	
	jhalo = XParam.blkwidth;

	if (XBlock.level[ib] > XBlock.level[ibTL])
	{
		//
		for (int i = 0; i < XParam.blkwidth; i++)
		{
			//
			ihalo = i;
			ibn = ibTL;

			//il = i;
			//jl = XParam.blkwidth - 1;

			jp = 0;
			ip = XBlock.BotLeft[ibn] == ib ? ftoi(floor(i * T(0.5))) : ftoi(floor(i * T(0.5)) + XParam.blkwidth / 2);

			//jm = jp;
			//im = ceil(i * T(0.5)) * 2 > i ? ip + 1 : ip - 1;

			ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);
		}

	}
}

template <class T> __global__ void conserveElevationTop(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int blkmemwidth = blockDim.x + XParam.halowidth * 2;
	
	
	int ix = threadIdx.x;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int TL = XBlock.TopLeft[ib];
	int TR = XBlock.TopRight[ib];

		

	int ihalo, jhalo, i, j, ibn;

	ihalo = ix;
	jhalo = blockDim.x;
	j = 0;

	if (lev < XBlock.level[TL] && ix < (blockDim.x / 2))
	{
		ibn = TL;
		
		i = ix * 2;

		conserveElevation(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, XEv.h, XEv.zs, zb);
	}
	if (lev < XBlock.level[TR] && ix >= (blockDim.x / 2))
	{
		ibn = TR;
		i = (ix - (blockDim.x / 2)) * 2;

		conserveElevation(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, XEv.h, XEv.zs, zb);
	}

	// Prolongation
	int ip, jp;
	

	if (lev > XBlock.level[TL])
	{
		//

		ihalo = ix;
		ibn = TL;

		//il = ix;
		//jl = blockDim.x - 1;

		jp = 0;
		ip = XBlock.BotLeft[ibn] == ib ? floor(ix * T(0.5)) : (floor(ix * T(0.5)) + blockDim.x / 2);

		//jm = jp;
		//im = ceil(ix * T(0.5)) * 2 > ix ? ip + 1 : ip - 1;

		ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);


	}
}

template <class T> __global__ void WetDryProlongationGPUTop(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int blkmemwidth = blockDim.x + XParam.halowidth * 2;


	int ix = threadIdx.x;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int TL = XBlock.TopLeft[ib];
	int TR = XBlock.TopRight[ib];
	// Prolongation
	int ip, jp,ihalo,jhalo,ibn;

	jhalo = blockDim.x;
	ihalo = ix;



	if (lev > XBlock.level[TL])
	{
		//
		
		ibn = TL;

		//il = ix;
		//jl = blockDim.x - 1;

		jp = 0;
		ip = XBlock.BotLeft[ibn] == ib ? floor(ix * T(0.5)) : (floor(ix * T(0.5)) + blockDim.x / 2);

		//jm = jp;
		//im = ceil(ix * T(0.5)) * 2 > ix ? ip + 1 : ip - 1;

		ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);


	}
}

template <class T> __global__ void WetDryRestrictionGPUTop(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int blkmemwidth = blockDim.x + XParam.halowidth * 2;


	int ix = threadIdx.x;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int TL = XBlock.TopLeft[ib];
	int TR = XBlock.TopRight[ib];
	// Prolongation
	int ihalo, jhalo, ibn, ir, jr;

	jhalo = blockDim.x;
	ihalo = ix;

	jr = 0;

	if (lev < XBlock.level[TL] && ix < (blockDim.x / 2))
	{
		ibn = TL;

		ir = ix * 2;

		wetdryrestriction(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
	}
	if (lev < XBlock.level[TR] && ix >= (blockDim.x / 2))
	{
		ibn = TR;
		ir = (ix - (blockDim.x / 2)) * 2;

		wetdryrestriction(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
	}



}

template <class T> void conserveElevationBot(Param XParam, int ib, int ibBL, int ibBR, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int ihalo, jhalo, ibn, ip, jp;

	

	if (XBlock.level[ib] < XBlock.level[ibBL])
	{
		for (int i = 0; i < XParam.blkwidth / 2; i++)
		{
			conserveElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibBL, i,-1, i * 2, XParam.blkwidth-2, XEv.h, XEv.zs, zb);
		}

	}
	if (XBlock.level[ib] < XBlock.level[ibBR])
	{
		for (int i = (XParam.blkwidth / 2); i < (XParam.blkwidth); i++)
		{
			conserveElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibBR, i, -1, (i - (XParam.blkwidth / 2)) * 2, XParam.blkwidth-2, XEv.h, XEv.zs, zb);
		}

	}


	// Prolongation
	
	jhalo = -1;

	if (XBlock.level[ib] > XBlock.level[ibBL])
	{
		//
		for (int i = 0; i < XParam.blkwidth; i++)
		{
			//
			ihalo = i;
			ibn = ibBL;

			//il = i;
			//jl = 0;

			jp = XParam.blkwidth - 1;
			ip = XBlock.TopLeft[ibn] == ib ? ftoi(floor(i * T(0.5))) : ftoi(floor(i * T(0.5)) + XParam.blkwidth / 2);

			//jm = jp;
			//im = ceil(i * T(0.5)) * 2 > i ? ip + 1 : ip - 1;

			ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);
		}

	}
}


template <class T> __global__ void conserveElevationBot(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int blkmemwidth = blockDim.x + XParam.halowidth * 2;
	
	
	int ix = threadIdx.x;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int BL = XBlock.BotLeft[ib];
	int BR = XBlock.BotRight[ib];

	
	

	int ihalo, jhalo, ibn;
	int i, j;

	ihalo = ix;
	jhalo = -1;
	j = blockDim.x-2;

	if (lev < XBlock.level[BL] && ix < (blockDim.x / 2))
	{
		ibn = BL;

		i = ix * 2;

		conserveElevation(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, XEv.h, XEv.zs, zb);
	}
	if (lev < XBlock.level[BR] && ix >= (blockDim.x / 2))
	{
		ibn = BR;
		i = (ix - (blockDim.x / 2)) * 2;

		conserveElevation(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, XEv.h, XEv.zs, zb);
	}

	// Prolongation
	int ip, jp;
	//int ip, jp, il, jl, im, jm;
	//jhalo = -1;

	if (lev > XBlock.level[BL])
	{
		//

		ihalo = ix;
		ibn = BL;

		//il = ix;
		//jl = 0;

		jp = blockDim.x - 1;
		ip = XBlock.TopLeft[ibn] == ib ? floor(ix *T(0.5)) : (floor(ix*T(0.5)) + blockDim.x / 2);

		//jm = jp;
		//im = ceil(ix * T(0.5)) * 2 > ix ? ip + 1 : ip - 1;

		ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp,  XEv.h, XEv.zs, zb);


	}

}

template <class T> __global__ void WetDryProlongationGPUBot(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int blkmemwidth = blockDim.x + XParam.halowidth * 2;


	int ix = threadIdx.x;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int BL = XBlock.BotLeft[ib];
	int BR = XBlock.BotRight[ib];




	int ihalo, jhalo, ibn;

	// Prolongation
	int ip, jp;
	//int ip, jp, il, jl, im, jm;
	//jhalo = -1;
	jhalo = -1;
	ihalo = ix;



	if (lev > XBlock.level[BL])
	{
		//
		
		ibn = BL;

		//il = ix;
		//jl = 0;

		jp = blockDim.x - 1;
		ip = XBlock.TopLeft[ibn] == ib ? floor(ix * T(0.5)) : (floor(ix * T(0.5)) + blockDim.x / 2);

		//jm = jp;
		//im = ceil(ix * T(0.5)) * 2 > ix ? ip + 1 : ip - 1;

		ProlongationElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ip, jp, XEv.h, XEv.zs, zb);


	}
}


template <class T> __global__ void WetDryRestrictionGPUBot(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int blkmemwidth = blockDim.x + XParam.halowidth * 2;


	int ix = threadIdx.x;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int BL = XBlock.BotLeft[ib];
	int BR = XBlock.BotRight[ib];




	int ihalo, jhalo, ibn;

	// Prolongation
	int ir, jr;
	//int ip, jp, il, jl, im, jm;
	//jhalo = -1;
	jhalo = -1;
	ihalo = ix;
	jr = blockDim.x - 2;

	if (lev < XBlock.level[BL] && ix < (blockDim.x / 2))
	{
		ibn = BL;

		ir = ix * 2;

		wetdryrestriction(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
	}
	if (lev < XBlock.level[BR] && ix >= (blockDim.x / 2))
	{
		ibn = BR;
		ir = (ix - (blockDim.x / 2)) * 2;

		wetdryrestriction(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, ir, jr, XEv.h, XEv.zs, zb);
	}

	
}

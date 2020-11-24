#include "ConserveElevation.h"


template <class T> void conserveElevation(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
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


		conserveElevationLeft<<<gridDim, blockDimHaloLR, 0>>> (XParam, XBlock, XEv, zb);
		CUDA_CHECK(cudaDeviceSynchronize());
		conserveElevationRight<<<gridDim, blockDimHaloLR, 0 >>> (XParam, XBlock, XEv, zb);
		CUDA_CHECK(cudaDeviceSynchronize());
		conserveElevationTop<<<gridDim, blockDimHaloBT, 0 >>> (XParam, XBlock, XEv, zb);
		CUDA_CHECK(cudaDeviceSynchronize());
		conserveElevationBot<<<gridDim, blockDimHaloBT, 0 >>> (XParam, XBlock, XEv, zb);
		CUDA_CHECK(cudaDeviceSynchronize());
	
}
template void conserveElevationGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, float* zb);
template void conserveElevationGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, double* zb);

template <class T> __host__ __device__ void conserveElevation(int halowidth,int blkmemwidth,T eps, int ib, int ibn,int ihalo, int jhalo ,int i,int j, T* h, T* zs, T * zb)
{
	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, hwet;

	int write;

	write = memloc(halowidth, blkmemwidth, ihalo, jhalo, ib);
	//jj = j * 2;
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

	conserveElevation(zb[write], zswet, hwet);

	h[write] = hwet;
	zs[write] = zswet;


}

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

template <class T> void conserveElevationGradHalo(Param XParam, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		conserveElevationGHLeft(XParam, ib, XBlock.LeftBot[ib], XBlock.LeftTop[ib], XBlock, h, dhdx, dhdy);
		conserveElevationGHRight(XParam, ib, XBlock.RightBot[ib], XBlock.RightTop[ib], XBlock, h, dhdx, dhdy);
		conserveElevationGHTop(XParam, ib, XBlock.TopLeft[ib], XBlock.TopRight[ib], XBlock, h, dhdx, dhdy);
		conserveElevationGHBot(XParam, ib, XBlock.BotLeft[ib], XBlock.BotRight[ib], XBlock, h, dhdx, dhdy);
	}
}
template void conserveElevationGradHalo<float>(Param XParam, BlockP<float> XBlock, float* h, float* dhdx, float* dhdy);
template void conserveElevationGradHalo<double>(Param XParam, BlockP<double> XBlock, double* h, double* dhdx, double* dhdy);

template <class T> void conserveElevationGradHaloGPU(Param XParam, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	dim3 blockDimHaloLR(1, 16, 1);
	dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	conserveElevationGHLeft <<<gridDim, blockDimHaloLR, 0 >>> (XParam, XBlock, h, dhdx, dhdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	conserveElevationGHRight <<<gridDim, blockDimHaloLR, 0 >>> (XParam, XBlock, h, dhdx, dhdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	conserveElevationGHTop <<<gridDim, blockDimHaloBT, 0 >>> (XParam, XBlock, h, dhdx, dhdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	conserveElevationGHBot <<<gridDim, blockDimHaloBT, 0 >>> (XParam, XBlock, h, dhdx, dhdy);
	CUDA_CHECK(cudaDeviceSynchronize());
	
}
template void conserveElevationGradHaloGPU<float>(Param XParam, BlockP<float> XBlock, float* h, float* dhdx, float* dhdy);
template void conserveElevationGradHaloGPU<double>(Param XParam, BlockP<double> XBlock, double* h, double* dhdx, double* dhdy);

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
template <class T> void conserveElevationGHLeft(Param XParam, int ib, int ibLB, int ibLT, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	if (XBlock.level[ib] < XBlock.level[ibLB])
	{
		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{
			conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibLB,  -1, j, XParam.blkwidth - 2, j * 2, h, dhdx, dhdy);
		}
	}
	if (XBlock.level[ib] < XBlock.level[ibLT])
	{
		for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
		{
			conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibLT, -1, j, XParam.blkwidth - 2, (j - (XParam.blkwidth / 2)) * 2, h, dhdx, dhdy);
		}
	}
}

template <class T> __global__ void conserveElevationGHLeft(Param XParam, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	unsigned int blkmemwidth = blockDim.y + XParam.halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = 0;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int LB = XBlock.LeftBot[ib];
	int LT = XBlock.LeftTop[ib];

	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, hwet;

	int ihalo, jhalo, i, j, ibn, write;

	ihalo = -1;
	jhalo = iy;
	i = XParam.blkwidth - 2;

	if (XBlock.level[ib] < XBlock.level[LB] && iy < (blockDim.y / 2))
	{
		ibn = LB;
		j = iy * 2;

		conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}
	if (XBlock.level[ib] < XBlock.level[LT] && iy >= (blockDim.y / 2))
	{
		ibn = LT;
		j = (iy - (blockDim.y / 2)) * 2;
		conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}
}

template <class T> void conserveElevationGHRight(Param XParam, int ib, int ibRB, int ibRT, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	if (XBlock.level[ib] < XBlock.level[ibRB])
	{
		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{
			conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibRB, XParam.blkwidth, j, 0, j * 2, h, dhdx, dhdy);
		}
	}
	if (XBlock.level[ib] < XBlock.level[ibRT])
	{
		for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
		{
			conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibRT, XParam.blkwidth, j, 0, (j - (XParam.blkwidth / 2)) * 2, h, dhdx, dhdy);
		}
	}
}

template <class T> __global__ void conserveElevationGHRight(Param XParam, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	unsigned int blkmemwidth = blockDim.y + XParam.halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = blockDim.y-1;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int RB = XBlock.RightBot[ib];
	int RT = XBlock.RightTop[ib];

	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, hwet;

	int ihalo, jhalo, i, j, ibn, write;

	ihalo = blockDim.y;
	jhalo = iy;
	i = 0;

	if (XBlock.level[ib] < XBlock.level[RB] && iy < (blockDim.y / 2))
	{
		ibn = RB;
		j = iy * 2;

		conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}
	if (XBlock.level[ib] < XBlock.level[RT] && iy >= (blockDim.y / 2))
	{
		ibn = RT;
		j = (iy - (blockDim.y / 2)) * 2;
		conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}
}

template <class T> void conserveElevationGHTop(Param XParam, int ib, int ibTL, int ibTR, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	if (XBlock.level[ib] < XBlock.level[ibTL])
	{
		for (int i = 0; i < XParam.blkwidth / 2; i++)
		{
			conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibTL, i, XParam.blkwidth, i * 2, 0, h, dhdx, dhdy);
		}
	}
	if (XBlock.level[ib] < XBlock.level[ibTR])
	{
		for (int i = (XParam.blkwidth / 2); i < (XParam.blkwidth); i++)
		{
			conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibTR, i, XParam.blkwidth, (i - (XParam.blkwidth / 2)) * 2, 0, h, dhdx, dhdy);
		}
	}
}

template <class T> __global__ void conserveElevationGHTop(Param XParam, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	unsigned int blkmemwidth = blockDim.y + XParam.halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int iy = blockDim.x - 1;
	unsigned int ix = threadIdx.x;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int TL = XBlock.TopLeft[ib];
	int TR = XBlock.TopRight[ib];

	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, hwet;

	int ihalo, jhalo, i, j, ibn, write;

	ihalo = ix;
	jhalo = iy+1;
	j = 0;

	if (XBlock.level[ib] < XBlock.level[TL] && ix < (blockDim.x / 2))
	{
		ibn = TL;
		i = ix * 2;

		conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}
	if (XBlock.level[ib] < XBlock.level[TR] && ix >= (blockDim.x / 2))
	{
		ibn = TR;
		i = (ix - (blockDim.x / 2)) * 2;;
		conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}
}

template <class T> void conserveElevationGHBot(Param XParam, int ib, int ibBL, int ibBR, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	if (XBlock.level[ib] < XBlock.level[ibBL])
	{
		for (int i = 0; i < XParam.blkwidth / 2; i++)
		{
			conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibBL, i, -1, i * 2, XParam.blkwidth - 2, h, dhdx, dhdy);
		}
	}
	if (XBlock.level[ib] < XBlock.level[ibBR])
	{
		for (int i = (XParam.blkwidth / 2); i < (XParam.blkwidth); i++)
		{
			conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibBR, i, -1, (i - (XParam.blkwidth / 2)) * 2, XParam.blkwidth - 2, h, dhdx, dhdy);
		}
	}
}

template <class T> __global__ void conserveElevationGHBot(Param XParam, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	unsigned int blkmemwidth = blockDim.y + XParam.halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int iy = blockDim.x - 1;
	unsigned int ix = threadIdx.x;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int BL = XBlock.BotLeft[ib];
	int BR = XBlock.BotRight[ib];

	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, hwet;

	int ihalo, jhalo, i, j, ibn, write;

	ihalo = ix;
	jhalo = -1;
	j = XParam.blkwidth - 2;;

	if (XBlock.level[ib] < XBlock.level[BL] && ix < (blockDim.x / 2))
	{
		ibn = BL;
		i = ix * 2;

		conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}
	if (XBlock.level[ib] < XBlock.level[BR] && ix >= (blockDim.x / 2))
	{
		ibn = BR;
		i = (ix - (blockDim.x / 2)) * 2;;
		conserveElevationGradHalo(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, h, dhdx, dhdy);
	}
}

template <class T> void conserveElevationLeft(Param XParam,int ib, int ibLB, int ibLT, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int ii, ir, it, itr,jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, writezs, writeh;
	
	int write;

	if (XBlock.level[ib] < XBlock.level[ibLB])
	{
		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{
			conserveElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibLB, -1, j, XParam.blkwidth-2, j*2, XEv.h, XEv.zs, zb);
		}

	}
	if (XBlock.level[ib] < XBlock.level[ibLT])
	{
		for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
		{
			conserveElevation(XParam.halowidth, XParam.blkmemwidth, T(XParam.eps), ib, ibLT, -1, j, XParam.blkwidth-2, (j - (XParam.blkwidth / 2)) * 2, XEv.h, XEv.zs, zb);
		}

	}
}

template <class T> __global__ void conserveElevationLeft(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	unsigned int blkmemwidth = blockDim.y + XParam.halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = 0;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int LB = XBlock.LeftBot[ib];
	int LT = XBlock.LeftTop[ib];

	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, hwet;

	int ihalo , jhalo, i, j, ibn, write;

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

}



template <class T> void conserveElevationRight(Param XParam, int ib, int ibRB, int ibRT, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, writezs, writeh;

	int write;

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
}

template <class T> __global__ void conserveElevationRight(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	unsigned int blkmemwidth = blockDim.y + XParam.halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = blockDim.y - 1;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int RB = XBlock.RightBot[ib];
	int RT = XBlock.RightTop[ib];

	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, hwet;

	int ihalo, jhalo, i, j, ibn, write;

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

}

template <class T> void conserveElevationTop(Param XParam, int ib, int ibTL, int ibTR, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, writezs, writeh;

	int write;

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
}

template <class T> __global__ void conserveElevationTop(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	unsigned int blkmemwidth = blockDim.x + XParam.halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int iy = blockDim.x - 1;
	unsigned int ix = threadIdx.x;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int TL = XBlock.TopLeft[ib];
	int TR = XBlock.TopRight[ib];

	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, hwet;

	int ihalo, jhalo, i, j, ibn, write;

	ihalo = ix;
	jhalo = blockDim.x;
	j = 0;

	if (lev < XBlock.level[TL] && iy < (blockDim.x / 2))
	{
		ibn = TL;
		
		i = ix * 2;

		conserveElevation(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, XEv.h, XEv.zs, zb);
	}
	if (lev < XBlock.level[TR] && iy >= (blockDim.x / 2))
	{
		ibn = TR;
		i = (ix - (blockDim.x / 2)) * 2;

		conserveElevation(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, XEv.h, XEv.zs, zb);
	}

}

template <class T> void conserveElevationBot(Param XParam, int ib, int ibBL, int ibBR, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, writezs, writeh;

	int write;

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
}


template <class T> __global__ void conserveElevationBot(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb)
{
	unsigned int blkmemwidth = blockDim.x + XParam.halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int iy = 0;
	unsigned int ix = threadIdx.x;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int TL = XBlock.TopLeft[ib];
	int TR = XBlock.TopRight[ib];

	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, hwet;

	int ihalo, jhalo, i, j, ibn, write;

	ihalo = ix;
	jhalo = -1;
	j = blockDim.x-2;

	if (lev < XBlock.level[TL] && iy < (blockDim.x / 2))
	{
		ibn = TL;

		i = ix * 2;

		conserveElevation(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, XEv.h, XEv.zs, zb);
	}
	if (lev < XBlock.level[TR] && iy >= (blockDim.x / 2))
	{
		ibn = TR;
		i = (ix - (blockDim.x / 2)) * 2;

		conserveElevation(XParam.halowidth, blkmemwidth, T(XParam.eps), ib, ibn, ihalo, jhalo, i, j, XEv.h, XEv.zs, zb);
	}

}

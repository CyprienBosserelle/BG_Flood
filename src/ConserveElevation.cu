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

template <class T> void conserveElevation(Param XParam, int ib, int ibn,int ihalo, int jhalo ,int i,int j, T* h, T* zs, T * zb)
{
	int ii, ir, it, itr, jj;
	T iiwet, irwet, itwet, itrwet;
	T zswet, writezs;

	int write;

	write = memloc(XParam.halowidth, XParam.blkmemwidth, ihalo, jhalo, ib);
	//jj = j * 2;
	ii = memloc(XParam.halowidth, XParam.blkmemwidth, i, j, ibn);
	ir = memloc(XParam.halowidth, XParam.blkmemwidth, i + 1, j, ibn);
	it = memloc(XParam.halowidth, XParam.blkmemwidth, i, j + 1, ibn);
	itr = memloc(XParam.halowidth, XParam.blkmemwidth, i + 1, j + 1, ibn);

	iiwet = h[ii] > XParam.eps ? h[ii] : T(0.0);
	irwet = h[ir] > XParam.eps ? h[ir] : T(0.0);
	itwet = h[it] > XParam.eps ? h[it] : T(0.0);
	itrwet = h[itr] > XParam.eps ? h[itr] : T(0.0);

	zswet = iiwet * (zb[ii] + h[ii]) + irwet * (zb[ir] + h[ir]) + itwet * (zb[it] + h[it]) + itrwet * (zb[itr] + h[itr]);

	if ((iiwet + irwet + itwet + itrwet) > T(0.0))//
	{

		//T swet = T(1.0 / (iiwet + irwet + itwet + itrwet)) * (zs[ii] * iiwet + zs[ir] * irwet + zs[it] * itwet + zs[itr] * itrwet);

		

		//writezs = zswet;// utils::max(zswet, zb[write]);
		//writezs = utils::max(zswet, zb[write]);
		writezs = zswet / (iiwet + irwet + itwet + itrwet);

		// Weighted averaged depth

		h[write] = utils::max(T(0.0), writezs - zb[write]);
		zs[write] =  h[write] + zb[write];

		//printf("i=%d; j=%d; ib=%d; h[]=%f; zswrite=%f; zs[]=%f\n", ihalo, jhalo, ib, h[write], writezs, zs[write]);

		//zs[write] = utils::max(writezs, zb[write]);
	}
	else
	{
		//zs[write] = zb[write];
		h[write] = T(0.0);
	}

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

template <class T> void conserveElevationGradHalo(Param XParam, int ib, int ibn, int ihalo, int jhalo,int i, int j, T* h, T* dhdx, T* dhdy)
{
	int ii, ir, it, itr, jj;
	int write;
	write = memloc(XParam.halowidth, XParam.blkmemwidth, ihalo, jhalo, ib);

	ii = memloc(XParam.halowidth, XParam.blkmemwidth, i, j, ibn);
	ir = memloc(XParam.halowidth, XParam.blkmemwidth, i + 1, j, ibn);
	it = memloc(XParam.halowidth, XParam.blkmemwidth, i, j + 1, ibn);
	itr = memloc(XParam.halowidth, XParam.blkmemwidth, i + 1, j + 1, ibn);

	if (h[write] <= XParam.eps)
	{
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
			conserveElevationGradHalo(XParam, ib, ibLB,  -1, j, XParam.blkwidth - 2, j * 2, h, dhdx, dhdy);
		}
	}
	if (XBlock.level[ib] < XBlock.level[ibLT])
	{
		for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
		{
			conserveElevationGradHalo(XParam, ib, ibLT, -1, j, XParam.blkwidth - 2, (j - (XParam.blkwidth / 2)) * 2, h, dhdx, dhdy);
		}
	}
}

template <class T> void conserveElevationGHRight(Param XParam, int ib, int ibRB, int ibRT, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	if (XBlock.level[ib] < XBlock.level[ibRB])
	{
		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{
			conserveElevationGradHalo(XParam, ib, ibRB, XParam.blkwidth, j, 0, j * 2, h, dhdx, dhdy);
		}
	}
	if (XBlock.level[ib] < XBlock.level[ibRT])
	{
		for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
		{
			conserveElevationGradHalo(XParam, ib, ibRT, XParam.blkwidth, j, 0, (j - (XParam.blkwidth / 2)) * 2, h, dhdx, dhdy);
		}
	}
}

template <class T> void conserveElevationGHTop(Param XParam, int ib, int ibTL, int ibTR, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	if (XBlock.level[ib] < XBlock.level[ibTL])
	{
		for (int i = 0; i < XParam.blkwidth / 2; i++)
		{
			conserveElevationGradHalo(XParam, ib, ibTL, i, XParam.blkwidth, i * 2, 0, h, dhdx, dhdy);
		}
	}
	if (XBlock.level[ib] < XBlock.level[ibTR])
	{
		for (int i = (XParam.blkwidth / 2); i < (XParam.blkwidth); i++)
		{
			conserveElevationGradHalo(XParam, ib, ibTR, i, XParam.blkwidth, (i - (XParam.blkwidth / 2)) * 2, 0, h, dhdx, dhdy);
		}
	}
}

template <class T> void conserveElevationGHBot(Param XParam, int ib, int ibBL, int ibBR, BlockP<T> XBlock, T* h, T* dhdx, T* dhdy)
{
	if (XBlock.level[ib] < XBlock.level[ibBL])
	{
		for (int i = 0; i < XParam.blkwidth / 2; i++)
		{
			conserveElevationGradHalo(XParam, ib, ibBL, i, -1, i * 2, XParam.blkwidth - 2, h, dhdx, dhdy);
		}
	}
	if (XBlock.level[ib] < XBlock.level[ibBR])
	{
		for (int i = (XParam.blkwidth / 2); i < (XParam.blkwidth); i++)
		{
			conserveElevationGradHalo(XParam, ib, ibBR, i, -1, (i - (XParam.blkwidth / 2)) * 2, XParam.blkwidth - 2, h, dhdx, dhdy);
		}
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
			conserveElevation(XParam, ib, ibLB, -1, j, XParam.blkwidth-2, j*2, XEv.h, XEv.zs, zb);
		}

	}
	if (XBlock.level[ib] < XBlock.level[ibLT])
	{
		for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
		{
			conserveElevation(XParam, ib, ibLT, -1, j, XParam.blkwidth-2, (j - (XParam.blkwidth / 2)) * 2, XEv.h, XEv.zs, zb);
		}

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
			conserveElevation(XParam, ib, ibRB, XParam.blkwidth, j, 0, j*2, XEv.h, XEv.zs, zb);
		}

	}
	if (XBlock.level[ib] < XBlock.level[ibRT])
	{
		for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
		{
			conserveElevation(XParam, ib, ibRT, XParam.blkwidth, j, 0, (j - (XParam.blkwidth / 2)) * 2, XEv.h, XEv.zs, zb);
		}

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
			conserveElevation(XParam, ib, ibTL, i, XParam.blkwidth, i*2, 0, XEv.h, XEv.zs, zb);
		}

	}
	if (XBlock.level[ib] < XBlock.level[ibTR])
	{
		for (int i = (XParam.blkwidth / 2); i < (XParam.blkwidth); i++)
		{
			conserveElevation(XParam, ib, ibTR, i, XParam.blkwidth, (i - (XParam.blkwidth / 2)) * 2, 0, XEv.h, XEv.zs, zb);
		}

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
			conserveElevation(XParam, ib, ibBL, i,-1, i * 2, XParam.blkwidth-2, XEv.h, XEv.zs, zb);
		}

	}
	if (XBlock.level[ib] < XBlock.level[ibBR])
	{
		for (int i = (XParam.blkwidth / 2); i < (XParam.blkwidth); i++)
		{
			conserveElevation(XParam, ib, ibBR, i, -1, (i - (XParam.blkwidth / 2)) * 2, XParam.blkwidth-2, XEv.h, XEv.zs, zb);
		}

	}
}


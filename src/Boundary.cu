#include "Boundary.h"


template <class T> void Flowbnd(Param XParam, Loop<T> &XLoop,BlockP<T> XBlock, bndparam side, EvolvingP<T> XEv)
{
	dim3 blockDim(XParam.blkwidth, 1, 1);
	dim3 gridDimBBND(side.nblk, 1, 1);

	T* un, * ut;

	double itime=0.0;

	std::vector<double> zsbndleft;
	std::vector<double> uubndleft;
	std::vector<double> vvbndleft;

	if (side.isright == 0)
	{
		//top or bottom
		un = XEv.v;
		ut = XEv.u;
	}
	else
	{
		un = XEv.u;
		ut = XEv.v;
	}

	if (side.on)
	{
		int SLstepinbnd = 1;

		double difft = side.data[SLstepinbnd].time - XLoop.totaltime;
		while (difft < 0.0)
		{
			SLstepinbnd++;
			difft = side.data[SLstepinbnd].time - XLoop.totaltime;
		}

		itime = SLstepinbnd - 1.0 + (XLoop.totaltime - side.data[SLstepinbnd - 1].time) / (side.data[SLstepinbnd].time - side.data[SLstepinbnd - 1].time);

		
		for (int n = 0; n < side.data[SLstepinbnd].wlevs.size(); n++)
		{
			zsbndleft.push_back(interptime(side.data[SLstepinbnd].wlevs[n], side.data[SLstepinbnd - 1].wlevs[n], side.data[SLstepinbnd].time - side.data[SLstepinbnd - 1].time, XLoop.totaltime - side.data[SLstepinbnd - 1].time));

		}
		// Repeat for u and v only if needed (otherwise values may not exist!)
		if (side.type == 4)
		{
			for (int n = 0; n < side.data[SLstepinbnd].uuvel.size(); n++)
			{
				uubndleft.push_back(interptime(side.data[SLstepinbnd].uuvel[n], side.data[SLstepinbnd - 1].uuvel[n], side.data[SLstepinbnd].time - side.data[SLstepinbnd - 1].time, XLoop.totaltime - side.data[SLstepinbnd - 1].time));

			}
			for (int n = 0; n < side.data[SLstepinbnd].vvvel.size(); n++)
			{
				vvbndleft.push_back(interptime(side.data[SLstepinbnd].vvvel[n], side.data[SLstepinbnd - 1].vvvel[n], side.data[SLstepinbnd].time - side.data[SLstepinbnd - 1].time, XLoop.totaltime - side.data[SLstepinbnd - 1].time));

			}
		}


	}
	if (XParam.GPUDEVICE >= 0)
	{
		bndGPU <<< gridDimBBND, blockDim, 0 >>> (XParam, side, XBlock, T(itime), XEv.zs, XEv.h, un, ut);
	}
	else
	{
		bndCPU(XParam, side, XBlock, zsbndleft, uubndleft, vvbndleft, XEv.zs, XEv.h, un, ut);
	}



}
template void Flowbnd<float>(Param XParam, Loop<float>& XLoop, BlockP<float> XBlock, bndparam side, EvolvingP<float> XEv);
template void Flowbnd<double>(Param XParam, Loop<double>& XLoop, BlockP<double> XBlock, bndparam side, EvolvingP<double> XEv);

template <class T> __global__ void bndGPU(Param XParam, bndparam side, BlockP<T> XBlock,float itime, T* zs, T* h, T* un, T* ut)
{
	//

	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	
	unsigned int ibl = blockIdx.x;
	unsigned int ix, iy;
	float itx;

	int ib = side.blks_g[ibl];
	int lev = XBlock.level[ib];


	T delta = calcres(T(XParam.dx), lev);

	if (side.isright == 0)
	{
		ix = threadIdx.x;
		iy = side.istop < 0 ? 0 : (blockDim.x - 1);
		//itx = (xx - XParam.xo) / (XParam.xmax - XParam.xo) * side.nbnd;
	}
	else
	{
		iy = threadIdx.x;
		ix = side.isright < 0 ? 0 : (blockDim.x - 1);
		//itx = (yy - XParam.yo) / (XParam.ymax - XParam.yo) * side.nbnd;
	}

	T sign = T(side.isright) + T(side.istop);
	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	
	T xx, yy;

	xx = XBlock.xo[ib] + ix * delta;
	yy = XBlock.yo[ib] + iy * delta;

	if (side.isright == 0)
	{
		itx = (xx) / (XParam.xmax - XParam.xo) * side.nbnd;
	}
	else
	{
		itx = (yy) / (XParam.ymax - XParam.yo) * side.nbnd;
	}


	

	int inside= Inside(halowidth, blkmemwidth, side.isright, side.istop, ix, iy, ib);
		   	
	T unnew, utnew, hnew, zsnew;
	T uninside, utinside, hinside, zsinside;
	T zsbnd;
	T unbnd = T(0.0);
	T utbnd = T(0.0);

	unnew = un[i];
	utnew = ut[i];
	hnew = h[i];
	zsnew = zs[i];

	zsinside = zs[inside];
	hinside = h[inside];
	uninside = un[inside];
	utinside = ut[inside];

	if (side.on)
	{
		
		zsbnd = tex2D<float>(side.GPU.WLS.tex, itime + 0.5f, itx + 0.5f);

		if (side.type == 4)
		{
			//un is V (top or bot bnd) or U (left or right bnd) depending on which side it's dealing with (same for ut)
			unbnd = side.isright == 0 ? tex2D<float>(side.GPU.Vvel.tex, itime + 0.5f, itx + 0.5f) : tex2D<float>(side.GPU.Vvel.tex, itime + 0.5f, itx + 0.5f);
			utbnd = side.isright == 0 ? tex2D<float>(side.GPU.Uvel.tex, itime + 0.5f, itx + 0.5f) : tex2D<float>(side.GPU.Uvel.tex, itime + 0.5f, itx + 0.5f);

		}
	}

	if (side.type == 0) // No slip == no friction wall
	{
		noslipbnd(zsinside, hinside, unnew, utnew, zsnew, hnew);
	}
	else if (side.type == 1) // neumann type
	{
		// Nothing to do here?
	}
	else if (side.type == 2)
	{
		Dirichlet1D(T(XParam.g), sign, zsbnd, zsinside, hinside, uninside, unnew, utnew, zsnew, hnew);
	}
	else if (side.type == 3)
	{
		ABS1D(T(XParam.g), sign, zsbnd, zsinside, hinside, utinside, unbnd, unnew, utnew, zsnew, hnew);
	}
	else if (side.type == 4)
	{
		
		ABS1D(T(XParam.g), sign, zsbnd, zsinside, hinside, utbnd, unbnd, unnew, utnew, zsnew, hnew);
	}
	
	un[i] = unnew;
	ut[i] = utnew;
	h[i] = hnew;
	zs[i] = zsnew;
}
template __global__ void bndGPU<float>(Param XParam, bndparam side, BlockP<float> XBlock, float itime, float* zs, float* h, float* un, float* ut);
template __global__ void bndGPU<double>(Param XParam, bndparam side, BlockP<double> XBlock, float itime, double* zs, double* h, double* un, double* ut);

template <class T> __host__ void bndCPU(Param XParam, bndparam side, BlockP<T> XBlock, std::vector<double> zsbndvec, std::vector<double> uubndvec, std::vector<double> vvbndvec, T* zs, T* h, T* un, T* ut)
{
	//
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = XParam.blkmemwidth;
	unsigned int blksize = blkmemwidth * blkmemwidth;

	for (int ibl = 0; ibl < side.nblk; ibl++)
	{

		int ib = side.blks[ibl];
		int lev = XBlock.level[ib];
		int nbnd = side.nbnd;
		T delta = calcres(T(XParam.dx), lev);

		for (int tx = 0; tx < XParam.blkwidth; tx++)
		{
			unsigned int ix, iy;
			double itx;
			T xx, yy;
			if (side.isright == 0)
			{
				ix = tx;
				iy = side.istop < 0 ? 0 : (XParam.blkwidth - 1);
				//itx = (xx - XParam.xo) / (XParam.xmax - XParam.xo) * side.nbnd;
			}
			else
			{
				iy = tx;
				ix = side.isright < 0 ? 0 : (XParam.blkwidth - 1);
				//itx = (yy - XParam.yo) / (XParam.ymax - XParam.yo) * side.nbnd;
			}
			xx = XBlock.xo[ib] + ix * delta;
			yy = XBlock.yo[ib] + iy * delta;

			T sign = T(side.isright) + T(side.istop);

			if (side.isright == 0)
			{
				itx = (xx) / (XParam.xmax - XParam.xo) * side.nbnd;
			}
			else
			{
				itx = (yy) / (XParam.ymax - XParam.yo) * side.nbnd;
			}


			int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
			int inside = Inside(halowidth, blkmemwidth, side.isright, side.istop, ix, iy, ib);

			T unnew, utnew, hnew, zsnew;
			T uninside, utinside, hinside, zsinside;
			T zsbnd;
			T unbnd = T(0.0);
			T utbnd = T(0.0);

			unnew = un[i];
			utnew = ut[i];
			hnew = h[i];
			zsnew = zs[i];

			zsinside = zs[inside];
			hinside = h[inside];
			uninside = un[inside];
			utinside = ut[inside];

			if (side.on)
			{
				int iprev = utils::max(int(floor(itx * (side.nbnd - 1))), 0);//min(max((int)ceil(itx / (1 / (side.nbnd - 1))), 0), (int)side.nbnd() - 2);
				int inext = iprev + 1;

				if (side.nbnd == 1)
				{
					zsbnd = zsbndvec[0];
					if (side.type == 4)
					{
						unbnd = side.isright == 0 ? vvbndvec[0] : uubndvec[0];
						utbnd = side.isright == 0 ? uubndvec[0] : vvbndvec[0];
					}
				}
				else
				{

					// here interp time is used to interpolate to the right node rather than in time...
					zsbnd = T(interptime(zsbndvec[inext], zsbndvec[iprev], 1.0, itx * (nbnd - 1) - iprev));

					if (side.type == 4)
					{
						unbnd = side.isright == 0 ? T(interptime(vvbndvec[inext], vvbndvec[iprev], 1.0, itx * (nbnd - 1) - iprev)) : T(interptime(uubndvec[inext], uubndvec[iprev], 1.0, itx * (nbnd - 1) - iprev));
						utbnd = side.isright == 0 ? T(interptime(uubndvec[inext], uubndvec[iprev], 1.0, itx * (nbnd - 1) - iprev)) : T(interptime(vvbndvec[inext], vvbndvec[iprev], 1.0, itx * (nbnd - 1) - iprev));
					}
				}



			}

			if (side.type == 0) // No slip == no friction wall
			{
				noslipbnd(zsinside, hinside, unnew, utnew, zsnew, hnew);
			}
			else if (side.type == 1) // neumann type
			{
				// Nothing to do here?
			}
			else if (side.type == 2)
			{
				Dirichlet1D(T(XParam.g), sign, zsbnd, zsinside, hinside, uninside, unnew, utnew, zsnew, hnew);
				
			}
			else if (side.type == 3)
			{
				ABS1D(T(XParam.g), sign, zsbnd, zsinside, hinside, utinside, unbnd, unnew, utnew, zsnew, hnew);
			}
			else if (side.type == 4)
			{

				ABS1D(T(XParam.g), sign, zsbnd, zsinside, hinside, utbnd, unbnd, unnew, utnew, zsnew, hnew);
			}

			un[i] = unnew;
			ut[i] = utnew;
			h[i] = hnew;
			zs[i] = zsnew;
		}
	}
}
template __host__ void bndCPU<float>(Param XParam, bndparam side, BlockP<float> XBlock, std::vector<double> zsbndvec, std::vector<double> uubndvec, std::vector<double> vvbndvec, float* zs, float* h, float* un, float* ut);
template __host__ void bndCPU<double>(Param XParam, bndparam side, BlockP<double> XBlock, std::vector<double> zsbndvec, std::vector<double> uubndvec, std::vector<double> vvbndvec, double* zs, double* h, double* un, double* ut);

// function to apply bnd at the boundary with masked blocked
// here a wall is applied in the halo 
template <class T> __host__ void maskbnd(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T*zb)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = XParam.blkmemwidth;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	int isright,istop;
	
	T zsinside,zsnew,hnew,vnew,unew,zbnew;

	bool isleftbot, islefttop, istopleft, istopright, isrighttop, isrightbot, isbotright, isbotleft;


	for (int ibl = 0; ibl <XBlock.mask.nblk ; ibl++)
	{

		int ib = XBlock.mask.blks[ibl];
		int lev = XBlock.level[ib];
		T delta = calcres(T(XParam.dx), lev);

		// find where the mask applies
		//
		findmaskside(XBlock.mask.side[ibl], isleftbot, islefttop, istopleft, istopright, isrighttop, isrightbot, isbotright, isbotleft);

		//leftside
		if (isleftbot | islefttop )//?
		{
			isright = -1;
			istop = 0;
					
			int ix = -1;

			int yst = isleftbot ? 0 : XParam.blkwidth * 0.5;
			int ynd = islefttop ? XParam.blkwidth : XParam.blkwidth * 0.5;

			for (int iy = yst; iy < ynd; iy++)
			{
				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int inside = Inside(halowidth, blkmemwidth, isright, istop, ix, iy, ib);

				zsinside = Xev.zs[inside];
				unew = Xev.u[i];
				vnew = Xev.v[i];
				zsnew = Xev.zs[i];
				hnew = Xev.zs[i];
				zbnew = zb[i];

				halowall(zsinside, unew, vnew, zsnew, hnew, zbnew);
				
				Xev.u[i]=unew;
				Xev.v[i]=vnew;
				Xev.zs[i]=zsnew;
				Xev.zs[i]=hnew;
				zb[i]=zbnew;

			}

		}
		//topside
		if (istopleft | istopright)//?
		{
			isright = 0;
			istop = 1;

			int iy = XParam.blkwidth;

			int xst = istopleft ? 0 : XParam.blkwidth * 0.5;
			int xnd = istopright ? XParam.blkwidth : XParam.blkwidth * 0.5;

			for (int ix = xst; ix < xnd; ix++)
			{
				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int inside = Inside(halowidth, blkmemwidth, isright, istop, ix, iy, ib);

				zsinside = Xev.zs[inside];
				unew = Xev.u[i];
				vnew = Xev.v[i];
				zsnew = Xev.zs[i];
				hnew = Xev.zs[i];
				zbnew = zb[i];

				halowall(zsinside, unew, vnew, zsnew, hnew, zbnew);

				Xev.u[i] = unew;
				Xev.v[i] = vnew;
				Xev.zs[i] = zsnew;
				Xev.zs[i] = hnew;
				zb[i] = zbnew;

			}

		}
		//rightside
		if (isrighttop | isrightbot)//?
		{
			isright = 1;
			istop = 0;

			int ix = XParam.blkwidth;

			int yst = isrightbot ? 0 : XParam.blkwidth * 0.5;
			int ynd = isrighttop ? XParam.blkwidth : XParam.blkwidth * 0.5;

			for (int iy = yst; iy < ynd; iy++)
			{
				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int inside = Inside(halowidth, blkmemwidth, isright, istop, ix, iy, ib);

				zsinside = Xev.zs[inside];
				unew = Xev.u[i];
				vnew = Xev.v[i];
				zsnew = Xev.zs[i];
				hnew = Xev.zs[i];
				zbnew = zb[i];

				halowall(zsinside, unew, vnew, zsnew, hnew, zbnew);

				Xev.u[i] = unew;
				Xev.v[i] = vnew;
				Xev.zs[i] = zsnew;
				Xev.zs[i] = hnew;
				zb[i] = zbnew;

			}

		}
		//botside
		if (isbotright | isbotleft)//?
		{
			isright = 0;
			istop = -1;

			int iy = -1;

			int xst = isbotleft ? 0 : XParam.blkwidth * 0.5;
			int xnd = isbotright ? XParam.blkwidth : XParam.blkwidth * 0.5;

			for (int ix = xst; ix < xnd; ix++)
			{
				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int inside = Inside(halowidth, blkmemwidth, isright, istop, ix, iy, ib);

				zsinside = Xev.zs[inside];
				unew = Xev.u[i];
				vnew = Xev.v[i];
				zsnew = Xev.zs[i];
				hnew = Xev.zs[i];
				zbnew = zb[i];

				halowall(zsinside, unew, vnew, zsnew, hnew, zbnew);

				Xev.u[i] = unew;
				Xev.v[i] = vnew;
				Xev.zs[i] = zsnew;
				Xev.zs[i] = hnew;
				zb[i] = zbnew;

			}

		}



	}
}
template __host__ void maskbnd<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> Xev, float* zb);
template __host__ void maskbnd<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> Xev, double* zb);

//For the GPU version we apply 4 separate global function in the hope to increase occupancy
template <class T> __global__ void maskbndGPUleft(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = XParam.blkmemwidth;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ibl = blockIdx.x;
	unsigned int ix, iy;

	int isright, istop;

	T zsinside, zsnew, hnew, vnew, unew, zbnew;

	bool isleftbot, islefttop, istopleft, istopright, isrighttop, isrightbot, isbotright, isbotleft;

	int ib = XBlock.mask.blks[ibl];
	//
	findmaskside(XBlock.mask.side[ibl], isleftbot, islefttop, istopleft, istopright, isrighttop, isrightbot, isbotright, isbotleft);

	//leftside
	if (isleftbot | islefttop)//?
	{
		isright = -1;
		istop = 0;

		ix = -1;
		iy = threadIdx.x;
		int yst = isleftbot ? 0 : XParam.blkwidth * 0.5;
		int ynd = islefttop ? XParam.blkwidth : XParam.blkwidth * 0.5;

		if(iy>=yst && iy<ynd)
		{
			int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
			int inside = Inside(halowidth, blkmemwidth, isright, istop, ix, iy, ib);

			zsinside = Xev.zs[inside];
			unew = Xev.u[i];
			vnew = Xev.v[i];
			zsnew = Xev.zs[i];
			hnew = Xev.zs[i];
			zbnew = zb[i];

			halowall(zsinside, unew, vnew, zsnew, hnew, zbnew);

			Xev.u[i] = unew;
			Xev.v[i] = vnew;
			Xev.zs[i] = zsnew;
			Xev.zs[i] = hnew;
			zb[i] = zbnew;

		}

	}

}
template __global__ void maskbndGPUleft<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> Xev, float* zb);
template __global__ void maskbndGPUleft<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> Xev, double* zb);

template <class T> __global__ void maskbndGPUtop(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = XParam.blkmemwidth;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ibl = blockIdx.x;
	unsigned int ix, iy;

	int isright, istop;

	T zsinside, zsnew, hnew, vnew, unew, zbnew;

	bool isleftbot, islefttop, istopleft, istopright, isrighttop, isrightbot, isbotright, isbotleft;

	int ib = XBlock.mask.blks[ibl];
	//
	findmaskside(XBlock.mask.side[ibl], isleftbot, islefttop, istopleft, istopright, isrighttop, isrightbot, isbotright, isbotleft);

	if (istopleft | istopright)//?
	{
		isright = 0;
		istop = 1;

		iy = XParam.blkwidth;
		ix = threadIdx.x;

		int xst = istopleft ? 0 : XParam.blkwidth * 0.5;
		int xnd = istopright ? XParam.blkwidth : XParam.blkwidth * 0.5;

		if(ix>=xst && ix<xnd)
		{
			int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
			int inside = Inside(halowidth, blkmemwidth, isright, istop, ix, iy, ib);

			zsinside = Xev.zs[inside];
			unew = Xev.u[i];
			vnew = Xev.v[i];
			zsnew = Xev.zs[i];
			hnew = Xev.zs[i];
			zbnew = zb[i];

			halowall(zsinside, unew, vnew, zsnew, hnew, zbnew);

			Xev.u[i] = unew;
			Xev.v[i] = vnew;
			Xev.zs[i] = zsnew;
			Xev.zs[i] = hnew;
			zb[i] = zbnew;

		}

	}
}
template __global__ void maskbndGPUtop<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> Xev, float* zb);
template __global__ void maskbndGPUtop<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> Xev, double* zb);

template <class T> __global__ void maskbndGPUright(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = XParam.blkmemwidth;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ibl = blockIdx.x;
	unsigned int ix, iy;

	int isright, istop;

	T zsinside, zsnew, hnew, vnew, unew, zbnew;

	bool isleftbot, islefttop, istopleft, istopright, isrighttop, isrightbot, isbotright, isbotleft;

	int ib = XBlock.mask.blks[ibl];
	//
	findmaskside(XBlock.mask.side[ibl], isleftbot, islefttop, istopleft, istopright, isrighttop, isrightbot, isbotright, isbotleft);

	if (isrighttop | isrightbot)//?
	{
		isright = 1;
		istop = 0;

		ix = XParam.blkwidth;

		iy = threadIdx.x;

		int yst = isrightbot ? 0 : XParam.blkwidth * 0.5;
		int ynd = isrighttop ? XParam.blkwidth : XParam.blkwidth * 0.5;

		if(iy>=yst && iy<ynd)
		{
			int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
			int inside = Inside(halowidth, blkmemwidth, isright, istop, ix, iy, ib);

			zsinside = Xev.zs[inside];
			unew = Xev.u[i];
			vnew = Xev.v[i];
			zsnew = Xev.zs[i];
			hnew = Xev.zs[i];
			zbnew = zb[i];

			halowall(zsinside, unew, vnew, zsnew, hnew, zbnew);

			Xev.u[i] = unew;
			Xev.v[i] = vnew;
			Xev.zs[i] = zsnew;
			Xev.zs[i] = hnew;
			zb[i] = zbnew;

		}

	}
}
template __global__ void maskbndGPUright<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> Xev, float* zb);
template __global__ void maskbndGPUright<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> Xev, double* zb);

template <class T> __global__ void maskbndGPUbot(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev, T* zb)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = XParam.blkmemwidth;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ibl = blockIdx.x;
	unsigned int ix, iy;

	int isright, istop;

	T zsinside, zsnew, hnew, vnew, unew, zbnew;

	bool isleftbot, islefttop, istopleft, istopright, isrighttop, isrightbot, isbotright, isbotleft;

	int ib = XBlock.mask.blks[ibl];
	//
	findmaskside(XBlock.mask.side[ibl], isleftbot, islefttop, istopleft, istopright, isrighttop, isrightbot, isbotright, isbotleft);
	
	if (isbotright | isbotleft)//?
	{
		isright = 0;
		istop = -1;

		iy = -1;
		ix = threadIdx.x;
		int xst = isbotleft ? 0 : XParam.blkwidth * 0.5;
		int xnd = isbotright ? XParam.blkwidth : XParam.blkwidth * 0.5;

		if(ix>=xst && ix<xnd)
		{
			int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
			int inside = Inside(halowidth, blkmemwidth, isright, istop, ix, iy, ib);

			zsinside = Xev.zs[inside];
			unew = Xev.u[i];
			vnew = Xev.v[i];
			zsnew = Xev.zs[i];
			hnew = Xev.zs[i];
			zbnew = zb[i];

			halowall(zsinside, unew, vnew, zsnew, hnew, zbnew);

			Xev.u[i] = unew;
			Xev.v[i] = vnew;
			Xev.zs[i] = zsnew;
			Xev.zs[i] = hnew;
			zb[i] = zbnew;

		}

	}

}
template __global__ void maskbndGPUbot<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> Xev, float* zb);
template __global__ void maskbndGPUbot<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> Xev, double* zb);

__device__ __host__ void findmaskside(int side, bool &isleftbot, bool& islefttop, bool& istopleft, bool& istopright, bool& isrighttop, bool& isrightbot, bool& isbotright, bool& isbotleft)
{
	int lb = 0b10000000;
	int lt = 0b01000000;
	int tl = 0b00100000;
	int tr = 0b00010000;
	int rt = 0b00001000;
	int rb = 0b00000100;
	int br = 0b00000010;
	int bl = 0b00000001;
	
	isleftbot = (side & lb) == lb;
	islefttop = (side & lt) == lt;

	istopleft = (side & tl) == tl;
	istopright = (side & tr) == tr;

	isrighttop = (side & rt) == rt;
	isrightbot = (side & rb) == rb;

	isbotright = (side & br) == br;
	isbotleft = (side & bl) == bl;
}




template <class T> __device__ __host__ void halowall(T zsinside, T& un, T& ut, T& zs, T& h,T&zb)
{
	// This function is used to make a wall on the halo to act as a wall
	// It may be more suitable/stable than the noslip as a wall boundary
	un = T(0.0);
	zs = zsinside;
	ut = T(0.0);
	h = T(0.0);
	zb = zsinside;

}


template <class T> __device__ __host__ void noslipbnd(T zsinside,T hinside,T &un, T &ut,T &zs, T &h)
{
	// Basic no slip bnd hs no normal velocity and leaves tanegtial velocity alone (maybe needs a wall friction added to it?)
	// 
	un = T(0.0);
	zs = zsinside;
	//ut[i] = ut[inside];//=0.0?
	h = hinside;//=0.0?

}


template <class T> __device__ __host__ void ABS1D(T g, T sign, T zsbnd, T zsinside, T hinside, T utbnd,T unbnd, T& un, T& ut, T& zs, T& h)
{
	// When nesting unbnd is read from file. when unbnd is not known assume 0. or the mean of un over a certain time 
	// For utbnd use utinside if no utbnd are known 
	un= sign * sqrt(g / h) * (zsinside - zsbnd) + T(unbnd);
	zs = zsinside;
	ut = T(utbnd);//ut[inside];
	h = hinside;
}

template <class T> __device__ __host__ void Dirichlet1D(T g, T sign, T zsbnd, T zsinside, T hinside,  T uninside, T& un, T& ut, T& zs, T& h)
{
	// Is this even the right formulation?.
	// I don't really like this formulation. while a bit less difssipative then abs1D with 0 unbnd (but worse if forcing uniside with 0) it is very reflective an not stable  
	T zbinside = zsinside - hinside;
	un = sign * T(2.0) * (sqrt(g * max(hinside, T(0.0))) - sqrt(g * max(zsbnd - zbinside, T(0.0)))) + uninside;
	ut = T(0.0);
	zs = zsinside;
	//ut[i] = ut[inside];
	h = hinside;
}



/*
template <class T> __global__ void ABS1D(int halowidth,int isright, int istop, int nbnd, T g, T dx, T xo, T yo, T xmax, T ymax, T itime, cudaTextureObject_t texZsBND, int* bndblck, int* level, T* blockxo, T* blockyo, T* zs, T* zb, T* h, T* un, T* ut)
{

	

	T xx, yy;
	
	T  sign, umean;
	float itx;

	sign = T(isright) + T(istop);




	//int xplus;
	//float hhi;
	float zsbnd;
	T zsinside;
	T levdx= calcres(dx, level[ib]);
	xx = blockxo[ib] + ix * levdx;
	yy = blockyo[ib] + iy * levdx;


	if (isright == 0) //leftside
	{
		
		itx = (xx - xo) / (xmax - xo) * (nbnd - 1);
	}
	else 
	{
		
		itx = (yy - yo) / (ymax - yo) * (nbnd - 1);
	}
	

	umean = T(0.0);
	zsbnd = tex2D(texZsBND, itime + 0.5f, itx + 0.5f);// textures use pixel registration so index of 0 is actually located at 0.5...

	if (isbnd(isright, istop, blockDim.x, ix, iy) && zsbnd > zb[i])
	{
		
		zsinside = zs[inside];
		
		un[i] = sign * sqrt(g / h[i]) * (zsinside - zsbnd) + umean;
		zs[i] = zsinside;
		ut[i] = ut[inside];
		h[i] = h[inside];
	}
}


template <class T> __host__ void ABS1D(Param XParam, std::vector<double> zsbndvec, int isright, int istop, int nbnd, T itime, BlockP<T> XBlock, int * bndblk, T* zs, T* zb, T* h, T* un, T* ut)
{

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = bndblk[ibl];


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
				int inside = Inside(XParam.halowidth, XParam.blkmemwidth, isright, istop, ix, iy, ib);

				// left bnd: isrigit = -1; istop=0;
				// right bnd: isright = 1; istop=0;
				// bottom bnd: isright = 0; istop=-1;
				// top bnd: isright = 0; istop=1;

				T xx, yy;

				T  sign, umean;
				float itx;

				sign = T(isright) + T(istop);




				//int xplus;
				//float hhi;
				float zsbnd;
				T zsinside;
				T levdx = calcres(dx, XBlock.level[ib]);
				xx = XBlock.xo[ib] + ix * levdx;
				yy = XBlock.yo[ib] + iy * levdx;
				int nbnd = zsbndvec.size();

				if (isright == 0) //leftside
				{

					itx = (xx - XParam.xo) / (XParam.xmax - XParam.xo) * (nbnd - 1);
				}
				else
				{

					itx = (yy - XParam.yo) / (XParam.ymax - XParam.yo) * (nbnd - 1);
				}


				umean = T(0.0);

				if (zsbndvec.size() == 1)
				{
					zsbnd = zsbndvec[0];
				}
				else
				{
					int iprev = utils::max(utils::min((int)floor(itx),nbnd-2),0);//utils::min(utils::max((int)ceil(itx / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
					int inext = iprev + 1;
					// here interp time is used to interpolate to the right node rather than in time...
					zsbnd = interptime(zsbndvec[inext], zsbndvec[iprev], 1.0, (itx - iprev));
				}

				if (isbnd(isright, istop, blockDim.x, ix, iy) && zsbnd > zb[i])
				{

					zsinside = zs[inside];

					un[i] = sign * sqrt(XParam.g / h[i]) * (zsinside - zsbnd) + umean;
					zs[i] = zsinside;
					ut[i] = ut[inside];
					h[i] = h[inside];
				}
			}
		}
	}
}
*/

__host__ __device__ int Inside(int halowidth, int blkmemwidth, int isright, int istop,int ix,int iy, int ib)
{
	//int bnd, bnd_c;
	int inside;

	if (isright < 0)
	{
		inside = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
		//bnd_c = 0;
		//bnd = ix;

	}
	else if (isright > 0)
	{
		inside = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
		//bnd_c = blockDim.x - 1;
		//bnd = ix;

	}
	else if (istop < 0)//isright must be ==0!
	{
		inside = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);
		//bnd_c = 0;
		//bnd = iy;

	}
	else // istop ==1 && isright ==0
	{
		inside = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
		//bnd_c = blockDim.y - 1;
		//bnd = iy;

	}
	return inside;
}


__host__ __device__ bool isbnd(int isright, int istop, int blkwidth, int ix, int iy)
{
	int bnd, bnd_c;
	//int inside;

	if (isright < 0 || istop < 0)
	{
		bnd_c = 0;
	}
	else
	{
		bnd_c = blkwidth - 1;
	}

	if (isright == 0)
	{
		bnd = iy;
	}
	else
	{
		bnd = ix;
	}


	return (bnd == bnd_c);
}
#include "Boundary.h"


template <class T> void Flowbnd(Param XParam, Loop<T> &XLoop, bndparam side, int isright, int istop, EvolvingP<T> XEv, T*zb)
{
	dim3 blockDim(16, 16, 1);// this should be only 16,1 or 1,16!!
	dim3 gridDimBBND(side.nblk, 1, 1);

	T* un, * ut;
	if (isright == 0)
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

		double itime = SLstepinbnd - 1.0 + (XLoop.totaltime - side.data[SLstepinbnd - 1].time) / (side.data[SLstepinbnd].time - side.data[SLstepinbnd - 1].time);

		if (side.type == 2)
		{
			// Dirichlet
		}
		else if (side.type == 3)
		{
			// Absorbing (normal only)
		}
		else if (side.type == 4)
		{
			// Nesting - Absorbing 
		}

	}
	else
	{
		if (side.type == 0)
		{

			
			// No slip wall
			noslipbnd << <gridDimBBND, blockDim, 0 >> > (XParam.halowidth,isright, istop, side.blks_g, XEv.zs, XEv.h, un);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else if (side.type == 1)
		{
			// Neumann (normal)
		}
	}



}

template void Flowbnd<float>(Param XParam, Loop<float>& XLoop, bndparam side, int isright, int istop, EvolvingP<float> XEv, float* zb);
template void Flowbnd<double>(Param XParam, Loop<double>& XLoop, bndparam side, int isright, int istop, EvolvingP<double> XEv, double* zb);

template <class T> __global__ void noslipbnd(int halowidth,int isright, int istop, int* bndblck, T* zs, T* h, T* un)
{
	//

	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int ib = bndblck[ibl];

	int i = memloc(halowidth, halowidth + blockDim.x, ix , iy , ib);
	int inside= Inside(halowidth, halowidth + blockDim.x, isright, istop, ix, iy, ib);


	if (isbnd(isright, istop, blockDim.x, ix, iy))
	{


		un[i] = T(0.0);
		zs[i] = zs[inside];
		//ut[i] = ut[inside];
		h[i] = h[inside];
	}

}

template <class T> __host__ void noslipbnd(Param XParam, int isright, int istop, int* bndblck, T* zs, T* h, T* un)
{

	//
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = bndblck[ibl];

		
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{



				int i = memloc(XParam, ix, iy, ib);
				int inside = Inside(XParam.halowidth, XParam.blkmemwidth, isright, istop, ix, iy, ib);

				
				if (isbnd(isright, istop, XParam.blkwidth, ix, iy))
				{


					un[i] = T(0.0);
					zs[i] = zs[inside];
					//ut[i] = ut[inside];
					h[i] = h[inside];
				}
			}
		}
	}

}



template <class T> __global__ void ABS1D(int halowidth,int isright, int istop, int nbnd, T g, T dx, T xo, T yo, T xmax, T ymax, T itime, cudaTextureObject_t texZsBND, int* bndblck, int* level, T* blockxo, T* blockyo, T* zs, T* zb, T* h, T* un, T* ut)
{

	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;

	int ib = bndblck[ibl];

	int i = memloc(halowidth, halowidth + blockDim.x, ix, iy, ib);
	int inside = Inside(halowidth, halowidth + blockDim.x, isright, istop, ix, iy, ib);

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
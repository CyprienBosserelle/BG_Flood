//////////////////////////////////////////////////////////////////////////////////
//						                                                        //
//Copyright (C) 2018 Bosserelle                                                 //
// This code contains an adaptation of the St Venant equation from Basilisk		//
// See																			//
// http://basilisk.fr/src/saint-venant.h and									//
// S. Popinet. Quadtree-adaptive tsunami modelling. Ocean Dynamics,				//
// doi: 61(9) : 1261 - 1285, 2011												//
//                                                                              //
//This program is free software: you can redistribute it and/or modify          //
//it under the terms of the GNU General Public License as published by          //
//the Free Software Foundation.                                                 //
//                                                                              //
//This program is distributed in the hope that it will be useful,               //
//but WITHOUT ANY WARRANTY; without even the implied warranty of                //    
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                 //
//GNU General Public License for more details.                                  //
//                                                                              //
//You should have received a copy of the GNU General Public License             //
//along with this program.  If not, see <http://www.gnu.org/licenses/>.         //
//////////////////////////////////////////////////////////////////////////////////


#include "GridManip.h"




template <class T,class F> void InitArrayBUQ(Param XParam, BlockP<F> XBlock,  T initval, T*& Arr)
{
	int ib, n;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				//n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				n = memloc(XParam, i, j, ib);
				Arr[n] = initval;
			}
		}
	}
}

template void InitArrayBUQ<float,float>(Param XParam, BlockP<float> XBlock, float initval, float*& Arr);
template void InitArrayBUQ<double, float>(Param XParam, BlockP<float> XBlock, double initval, double*& Arr);
template void InitArrayBUQ<int, float>(Param XParam, BlockP<float> XBlock, int initval, int*& Arr);
template void InitArrayBUQ<bool, float>(Param XParam, BlockP<float> XBlock, bool initval, bool*& Arr);

template void InitArrayBUQ<float, double>(Param XParam, BlockP<double> XBlock, float initval, float*& Arr);
template void InitArrayBUQ<double, double>(Param XParam, BlockP<double> XBlock, double initval, double*& Arr);
template void InitArrayBUQ<int, double>(Param XParam, BlockP<double> XBlock, int initval, int*& Arr);
template void InitArrayBUQ<bool, double>(Param XParam, BlockP<double> XBlock, bool initval, bool*& Arr);



template <class T, class F> void InitBlkBUQ(Param XParam, BlockP<F> XBlock, T initval, T*& Arr)
{
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		
				Arr[ib] = initval;
			
	}
}

template void InitBlkBUQ<bool, float>(Param XParam, BlockP<float> XBlock, bool initval, bool*& Arr);
template void InitBlkBUQ<int, float>(Param XParam, BlockP<float> XBlock, int initval, int*& Arr);
template void InitBlkBUQ<float, float>(Param XParam, BlockP<float> XBlock, float initval, float*& Arr);
template void InitBlkBUQ<double, float>(Param XParam, BlockP<float> XBlock, double initval, double*& Arr);

template void InitBlkBUQ<bool, double>(Param XParam, BlockP<double> XBlock, bool initval, bool*& Arr);
template void InitBlkBUQ<int, double>(Param XParam, BlockP<double> XBlock, int initval, int*& Arr);
template void InitBlkBUQ<float, double>(Param XParam, BlockP<double> XBlock, float initval, float*& Arr);
template void InitBlkBUQ<double, double>(Param XParam, BlockP<double> XBlock, double initval, double*& Arr);


template <class T,class F> void CopyArrayBUQ(Param XParam,BlockP<F> XBlock, T* source, T* & dest)
{
	int ib,n;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			
				
				dest[n] = source[n];
			}
		}
	}
}
template void CopyArrayBUQ<bool, float>(Param XParam, BlockP<float> XBlock, bool* source, bool*& dest);
template void CopyArrayBUQ<int, float>(Param XParam, BlockP<float> XBlock, int* source, int*& dest);
template void CopyArrayBUQ<float, float>(Param XParam, BlockP<float> XBlock, float* source, float*& dest);
template void CopyArrayBUQ<double, float>(Param XParam, BlockP<float> XBlock, double* source, double*& dest);

template void CopyArrayBUQ<bool, double>(Param XParam, BlockP<double> XBlock, bool* source, bool*& dest);
template void CopyArrayBUQ<int, double>(Param XParam, BlockP<double> XBlock, int* source, int*& dest);
template void CopyArrayBUQ<float, double>(Param XParam, BlockP<double> XBlock, float* source, float*& dest);
template void CopyArrayBUQ<double, double>(Param XParam, BlockP<double> XBlock, double* source, double*& dest);


template <class T> void CopyArrayBUQ(Param XParam, BlockP<T> XBlock, EvolvingP<T> source, EvolvingP<T>& dest)
{
	CopyArrayBUQ(XParam, XBlock, source.h, dest.h);
	CopyArrayBUQ(XParam, XBlock, source.u, dest.u);
	CopyArrayBUQ(XParam, XBlock, source.v, dest.v);
	CopyArrayBUQ(XParam, XBlock, source.zs, dest.zs);
}
template void CopyArrayBUQ<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> source, EvolvingP<float>& dest);
template void CopyArrayBUQ<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> source, EvolvingP<double>& dest);


template <class T>  void setedges(Param XParam, BlockP<T> XBlock, T *&zb)
{
	// template <class T> void setedges(int nblk, int nx, int ny, double xo, double yo, double dx, int * leftblk, int *rightblk, int * topblk, int* botblk, double *blockxo, double * blockyo, T *&zb)

	// here the bathy of the outter most cells of the domain are "set" to the same value as the second outter most.
	// this also applies to the blocks with no neighbour
	for (int bl = 0; bl < XParam.nblk; bl++)
	{
		int ib = XBlock.active[bl];
		// Now check each corner of each block
		

		// Left
		setedgessideLR(XParam, ib, XBlock.LeftBot[ib], XBlock.LeftTop[ib], 1, 0, zb);

		// Right
		setedgessideLR(XParam, ib, XBlock.RightBot[ib], XBlock.RightTop[ib], XParam.blkwidth - 2, XParam.blkwidth - 1, zb);

		// Top
		setedgessideBT(XParam, ib, XBlock.TopLeft[ib], XBlock.TopRight[ib], XParam.blkwidth - 2, XParam.blkwidth - 1, zb);

		// Bot
		setedgessideBT(XParam, ib, XBlock.BotLeft[ib], XBlock.BotRight[ib], 1, 0, zb);
		
		
	}
}
template void setedges<float>(Param XParam, BlockP<float> XBlock, float*& zb);
template void setedges<double>(Param XParam, BlockP<double> XBlock, double*& zb);

template <class T>  void setedgessideLR(Param XParam, int ib,int blkA, int blkB, int iread, int iwrite, T*& zb)
{
	if (blkA == ib || blkA == ib)
	{
		int n, k;
		int jstart, jend;
		jstart = (blkA == ib) ? 0 : XParam.blkwidth / 2;
		jend = (blkB == ib) ? XParam.blkwidth : XParam.blkwidth / 2;

		for (int j = jstart; j < jend; j++)
		{
			// read value at n and write at k
			n = (iread + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			k = (iwrite + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			zb[k] = zb[n];

		}
	}
}

template <class T>  void setedgessideBT(Param XParam, int ib, int blkA, int blkB, int jread, int jwrite, T*& zb)
{
	if (blkA == ib || blkA == ib)
	{
		int n, k;
		int istart, iend;
		istart = (blkA == ib) ? 0 : XParam.blkwidth / 2;
		iend = (blkB == ib) ? XParam.blkwidth : XParam.blkwidth / 2;

		for (int i = istart; i < iend; i++)
		{
			// read value at n and write at k
			n = (i + XParam.halowidth) + (jread + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			k = (i + XParam.halowidth) + (jwrite + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			zb[k] = zb[n];

		}
	}
}


template <class T, class F> void interp2BUQ(Param XParam, BlockP<T> XBlock, F forcing, T*& z)
{
	// This function interpolates the values in bathy maps or roughness map to cf using a bilinear interpolation

	T x, y;
	int n;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XBlock.active[ibl];
		
		double blkdx = calcres(XParam.dx, XBlock.level[ib]);
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				n = (i+XParam.halowidth) + (j+XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				x = XParam.xo + XBlock.xo[ib] + i * blkdx;
				y = XParam.yo + XBlock.yo[ib] + j * blkdx;

				z[n] = interp2BUQ(x, y, forcing);

			}
		}
	}
}
template void interp2BUQ<float, StaticForcingP<float>>(Param XParam, BlockP<float> XBlock, StaticForcingP<float> forcing, float*& z);
template void interp2BUQ<double, StaticForcingP<float>>(Param XParam, BlockP<double> XBlock, StaticForcingP<float> forcing, double*& z);
//template void interp2BUQ<float, StaticForcingP<float>>(Param XParam, BlockP<float> XBlock, std::vector<StaticForcingP<float>> forcing, float*& z);
template void interp2BUQ<double, StaticForcingP<float>>(Param XParam, BlockP<double> XBlock, StaticForcingP<float> forcing, double*& z);
template void interp2BUQ<float, deformmap<float>>(Param XParam, BlockP<float> XBlock, deformmap<float> forcing, float*& z);
template void interp2BUQ<double, deformmap<float>>(Param XParam, BlockP<double> XBlock, deformmap<float> forcing, double*& z);
template void interp2BUQ<float, DynForcingP<float>>(Param XParam, BlockP<float> XBlock, DynForcingP<float> forcing, float*& z);
template void interp2BUQ<double, DynForcingP<float>>(Param XParam, BlockP<double> XBlock, DynForcingP<float> forcing, double*& z);

template <class T> void interp2BUQ(Param XParam, BlockP<T> XBlock, std::vector<StaticForcingP<float>> forcing, T* z)
{
	// This function interpolates the values in bathy maps or roughness map to cf using a bilinear interpolation

	T x, y;
	int n;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XBlock.active[ibl];

		double blkdx = calcres(XParam.dx, XBlock.level[ib]);
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				x = XParam.xo + XBlock.xo[ib] + i * blkdx;
				y = XParam.yo + XBlock.yo[ib] + j * blkdx;
				
				// Interpolate to fill in values from the whole domain (even if the domain outspan the domain fo the bathy)
				z[n] = interp2BUQ(x, y, forcing[0]);

				// now interpolat to other grids
				for (int nf = 0; nf < forcing.size(); nf++)
				{
					if (x >= forcing[nf].xo && x <= forcing[nf].xmax && y >= forcing[nf].yo && y <= forcing[nf].ymax)
					{
						z[n] = interp2BUQ(x, y, forcing[nf]);
					}
				}
				

			}
		}
	}
}
template void interp2BUQ<float>(Param XParam, BlockP<float> XBlock, std::vector<StaticForcingP<float>> forcing, float* z);
template void interp2BUQ<double>(Param XParam, BlockP<double> XBlock, std::vector<StaticForcingP<float>> forcing, double* z);




template <class T, class F> T interp2BUQ(T x, T y, F forcing)
{
	//this is safer!

	double xi, yi;

	xi = utils::max(utils::min(double(x), forcing.xmax), forcing.xo);
	yi = utils::max(utils::min(double(y), forcing.ymax), forcing.yo);
	// cells that falls off this domain are assigned 
	double x1, x2, y1, y2;
	double q11, q12, q21, q22;
	int cfi, cfip, cfj, cfjp;



	cfi = utils::min(utils::max((int)floor((xi - forcing.xo) / forcing.dx), 0), forcing.nx - 2);
	cfip = cfi + 1;

	x1 = forcing.xo + forcing.dx * cfi;
	x2 = forcing.xo + forcing.dx * cfip;

	cfj = utils::min(utils::max((int)floor((yi - forcing.yo) / forcing.dx), 0), forcing.ny - 2);
	cfjp = cfj + 1;

	y1 = forcing.yo + forcing.dx * cfj;
	y2 = forcing.yo + forcing.dx * cfjp;

	q11 = forcing.val[cfi + cfj * forcing.nx];
	q12 = forcing.val[cfi + cfjp * forcing.nx];
	q21 = forcing.val[cfip + cfj * forcing.nx];
	q22 = forcing.val[cfip + cfjp * forcing.nx];

	return T(BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, xi, yi));
	//printf("x=%f\ty=%f\tcfi=%d\tcfj=%d\tn=%d\tzb_buq[n] = %f\n", x,y,cfi,cfj,n,zb_buq[n]);
}
template float interp2BUQ<float, StaticForcingP<float>>(float x, float y, StaticForcingP<float> forcing);
template double interp2BUQ<double, StaticForcingP<float>>(double x, double y, StaticForcingP<float> forcing);
template float interp2BUQ<float, StaticForcingP<int>>(float x, float y, StaticForcingP<int> forcing);
template double interp2BUQ<double, StaticForcingP<int>>(double x, double y, StaticForcingP<int> forcing);
template float interp2BUQ<float, deformmap<float>>(float x, float y, deformmap<float> forcing);
template double interp2BUQ<double, deformmap<float>>(double x, double y, deformmap<float> forcing);
template float interp2BUQ<float, DynForcingP<float>>(float x, float y, DynForcingP<float> forcing);
template double interp2BUQ<double, DynForcingP<float>>(double x, double y, DynForcingP<float> forcing);


template <class T, class F> void InterpstepCPU(int nx, int ny, int hdstep, F totaltime, F hddt, T *&Ux, T *Uo, T *Un)
{
	//float fac = 1.0;
	T Uxo, Uxn;

	/*Ums[tx]=Umask[ix];*/




	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			Uxo = Uo[i + nx*j];
			Uxn = Un[i + nx*j];

			Ux[i + nx*j] = Uxo + (totaltime - hddt*hdstep)*(Uxn - Uxo) / hddt;
		}
	}
}
template void InterpstepCPU<int,float>(int nx, int ny, int hdstep, float totaltime, float hddt, int *&Ux, int *Uo, int *Un);
template void InterpstepCPU<float, float>(int nx, int ny, int hdstep, float totaltime, float hddt, float *&Ux, float *Uo, float *Un);
template void InterpstepCPU<double, float>(int nx, int ny, int hdstep, float totaltime, float hddt, double *&Ux, double *Uo, double *Un);
template void InterpstepCPU<int, double>(int nx, int ny, int hdstep, double totaltime, double hddt, int*& Ux, int* Uo, int* Un);
template void InterpstepCPU<float, double>(int nx, int ny, int hdstep, double totaltime, double hddt, float*& Ux, float* Uo, float* Un);
template void InterpstepCPU<double, double>(int nx, int ny, int hdstep, double totaltime, double hddt, double*& Ux, double* Uo, double* Un);


template <class T> __global__ void InterpstepGPU(int nx, int ny, int hdstp, T totaltime, T hddt, T*Ux, T* Uo, T* Un)
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	__shared__ T Uxo[16][16];
	__shared__ T Uxn[16][16];
	//	__shared__ float Ums[16];



	if (ix < nx && iy < ny)
	{
		Uxo[tx][ty] = Uo[ix + nx * iy]/**Ums[tx]*/;
		Uxn[tx][ty] = Un[ix + nx * iy]/**Ums[tx]*/;

		Ux[ix + nx * iy] = Uxo[tx][ty] + (totaltime - hddt * hdstp) * (Uxn[tx][ty] - Uxo[tx][ty]) / hddt;
	}
}
//template __global__ void InterpstepGPU<int>(int nx, int ny, int hdstp, T totaltime, T hddt, T* Ux, T* Uo, T* Un);
template __global__ void InterpstepGPU<float>(int nx, int ny, int hdstp, float totaltime, float hddt, float* Ux, float* Uo, float* Un);
template __global__ void InterpstepGPU<double>(int nx, int ny, int hdstp, double totaltime, double hddt, double* Ux, double* Uo, double* Un);

template <class T> void Copy2CartCPU(int nx, int ny, T* dest, T* src)
{
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			dest[i + nx * j] = src[i + nx * j];
		}
	}
}
template void Copy2CartCPU<int>(int nx, int ny, int* dest, int* src);
template void Copy2CartCPU<bool>(int nx, int ny, bool* dest, bool* src);
template void Copy2CartCPU<float>(int nx, int ny, float* dest, float* src);
template void Copy2CartCPU<double>(int nx, int ny, double* dest, double* src);



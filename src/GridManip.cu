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




template <class T> void InitArrayBUQ(int nblk, int blkwidth, T initval, T*& Arr)
{
	int blksize = utils::sq(blkwidth);
	//inititiallise array with a single value
	for (int bl = 0; bl < nblk; bl++)
	{
		for (int j = 0; j < blkwidth; j++)
		{
			for (int i = 0; i < blkwidth; i++)
			{
				int n = i + j * blkwidth + bl * blksize;
				Arr[n] = initval;
			}
		}
	}
}

template <class T> void CopyArrayBUQ(int nblk, int blkwidth, T* source, T*& dest)
{
	int blksize = sq(blkwidth);
	//
	for (int bl = 0; bl < nblk; bl++)
	{
		for (int j = 0; j < blkwidth; j++)
		{
			for (int i = 0; i < blkwidth; i++)
			{
				int n = i + j * blkwidth + bl * blksize;
				dest[n] = source[n];
			}
		}
	}
}




template <class T>  void setedges(int nblk, int * leftblk, int *rightblk, int * topblk, int* botblk, T *&zb)
{
	// template <class T> void setedges(int nblk, int nx, int ny, double xo, double yo, double dx, int * leftblk, int *rightblk, int * topblk, int* botblk, double *blockxo, double * blockyo, T *&zb)

	// here the bathy of the outter most cells of the domain are "set" to the same value as the second outter most.
	// this also applies to the blocks with no neighbour
	for (int bl = 0; bl < nblk; bl++)
	{

		if (bl == leftblk[bl])//i.e. if a block refers to as it's onwn neighbour then it doesn't have a neighbour/// This also applies to block that are on the edge of the grid so the above is commentted
		{
			int i = 0;
			for (int j = 0; j < 16; j++)
			{

				zb[i + j * 16 + bl * 256] = zb[i + 1 + j * 16 + bl * 256];
			}
		}
		if (bl == rightblk[bl])
		{
			int i = 15;
			for (int j = 0; j < 16; j++)
			{

				zb[i + j * 16 + bl * 256] = zb[i - 1 + j * 16 + bl * 256];
			}
		}
		if (bl == topblk[bl])
		{
			int j = 15;
			for (int i = 0; i < 16; i++)
			{

				zb[i + j * 16 + bl * 256] = zb[i + (j - 1) * 16 + bl * 256];
			}
		}
		if (bl == botblk[bl])
		{
			int j = 0;
			for (int i = 0; i < 16; i++)
			{

				zb[i + j * 16 + bl * 256] = zb[i + (j + 1) * 16 + bl * 256];
			}
		}

	}
}

template void setedges<int>(int nblk, int * leftblk, int *rightblk, int * topblk, int* botblk, int *&zb);
template void setedges<float>(int nblk, int * leftblk, int *rightblk, int * topblk, int* botblk, float *&zb);
template void setedges<double>(int nblk, int * leftblk, int *rightblk, int * topblk, int* botblk, double *&zb);

template <class T, class F> void interp2BUQ(Param XParam, BlockP<T> XBlock, F forcing, T*& z)
{
	// This function interpolates the values in bathy maps or roughness map to cf using a bilinear interpolation

	double x, y;
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
				n = i + j * XParam.blkwidth + ib * XParam.blksize;
				x = XBlock.xo[ib] + i * blkdx;
				y = XBlock.yo[ib] + j * blkdx;

				//if (x >= xo && x <= xmax && y >= yo && y <= ymax)
				{
					//this is safer!
					x = utils::max(utils::min(x, forcing.xmax), forcing.xo);
					y = utils::max(utils::min(y, forcing.ymax), forcing.yo);
					// cells that falls off this domain are assigned 
					double x1, x2, y1, y2;
					double q11, q12, q21, q22;
					int cfi, cfip, cfj, cfjp;



					cfi = utils::min(utils::max((int)floor((x - forcing.xo) / forcing.dx), 0), forcing.nx - 2);
					cfip = cfi + 1;

					x1 = forcing.xo + forcing.dx * cfi;
					x2 = forcing.xo + forcing.dx * cfip;

					cfj = utils::min(utils::max((int)floor((y - forcing.yo) / forcing.dx), 0), forcing.ny - 2);
					cfjp = cfj + 1;

					y1 = forcing.yo + forcing.dx * cfj;
					y2 = forcing.yo + forcing.dx * cfjp;

					q11 = forcing.val[cfi + cfj * forcing.nx];
					q12 = forcing.val[cfi + cfjp * forcing.nx];
					q21 = forcing.val[cfip + cfj * forcing.nx];
					q22 = forcing.val[cfip + cfjp * forcing.nx];

					z[n] = (T)BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
					//printf("x=%f\ty=%f\tcfi=%d\tcfj=%d\tn=%d\tzb_buq[n] = %f\n", x,y,cfi,cfj,n,zb_buq[n]);
				}

			}
		}
	}
}


template void interp2BUQ<float, StaticForcingP<float>>(Param XParam, BlockP<float> XBlock, StaticForcingP<float> forcing, float*& z);
template void interp2BUQ<double, StaticForcingP<float>>(Param XParam, BlockP<double> XBlock, StaticForcingP<float> forcing, double*& z);
template void interp2BUQ<float, deformmap<float>>(Param XParam, BlockP<float> XBlock, deformmap<float> forcing, float*& z);
template void interp2BUQ<double, deformmap<float>>(Param XParam, BlockP<double> XBlock, deformmap<float> forcing, double*& z);



template <class T> void InterpstepCPU(int nx, int ny, int hdstep, T totaltime, T hddt, T *&Ux, T *Uo, T *Un)
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

template void InterpstepCPU<int>(int nx, int ny, int hdstep, int totaltime, int hddt, int *&Ux, int *Uo, int *Un);
template void InterpstepCPU<float>(int nx, int ny, int hdstep, float totaltime, float hddt, float *&Ux, float *Uo, float *Un);
template void InterpstepCPU<double>(int nx, int ny, int hdstep, double totaltime, double hddt, double *&Ux, double *Uo, double *Un);


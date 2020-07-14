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




template <class T> void InitArraySV(int nblk, int blksize, T initval, T * & Arr)
{
	//inititiallise array with a single value
	for (int bl = 0; bl < nblk; bl++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int i = 0; i < 16; i++)
			{
				int n = i + j * 16 + bl * blksize;
				Arr[n] = initval;
			}
		}
	}
}

template void InitArraySV<int>(int nblk, int blksize, int initval, int * & Arr);
template void InitArraySV<float>(int nblk, int blksize, float initval, float * & Arr);
template void InitArraySV<double>(int nblk, int blksize, double initval, double * & Arr);

template <class T> void CopyArray(int nblk, int blksize, T* source, T * & dest)
{
	//
	for (int bl = 0; bl < nblk; bl++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int i = 0; i < 16; i++)
			{
				int n = i + j * 16 + bl * blksize;
				dest[n] = source[n];
			}
		}
	}
}

template void CopyArray<int>(int nblk, int blksize, int* source, int * & dest);
template void CopyArray<float>(int nblk, int blksize, float* source, float * & dest);
template void CopyArray<double>(int nblk, int blksize, double* source, double * & dest);




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

template <class T> void carttoBUQ(int nblk, int nx, int ny, double xo, double yo, double dx, double* blockxo, double* blockyo, T * zb, T *&zb_buq)
{
	//
	int ix, iy;
	T x, y;
	for (int b = 0; b < nblk; b++)
	{

		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 16; j++)
			{
				x = blockxo[b] + i*dx;
				y = blockyo[b] + j*dx;
				ix = utils::min(utils::max((int)round((x - xo) / dx), 0), nx - 1); // min(max( part is overkill?
				iy = utils::min(utils::max((int)round((y - yo) / dx), 0), ny - 1);

				zb_buq[i + j * 16 + b * 256] = zb[ix + iy*nx];
				//printf("bid=%i\ti=%i\tj=%i\tix=%i\tiy=%i\tzb_buq[n]=%f\n", b,i,j,ix, iy, zb_buq[i + j * 16 + b * 256]);
			}
		}
	}
}

template void carttoBUQ<int>(int nblk, int nx, int ny, double xo, double yo, double dx, double* blockxo, double* blockyo, int * zb, int *&zb_buq);
template void carttoBUQ<float>(int nblk, int nx, int ny, double xo, double yo, double dx, double* blockxo, double* blockyo, float * zb, float *&zb_buq);
template void carttoBUQ<double>(int nblk, int nx, int ny, double xo, double yo, double dx, double* blockxo, double* blockyo, double * zb, double *&zb_buq);


template <class T> void interp2BUQ(int nblk, double blksize, double blkdx, double* blockxo, double* blockyo, int nx, int ny, double xo, double xmax, double yo, double ymax, double dx, T * zb, T *&zb_buq)
{
	// This function interpolates the values in bathy maps or roughness map to cf using a bilinear interpolation

	double x, y;
	int n;

	for (int bl = 0; bl < nblk; bl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		for (int j = 0; j < 16; j++)
		{
			for (int i = 0; i < 16; i++)
			{
				n = i + j * 16 + bl * blksize;
				x = blockxo[bl] + i*blkdx;
				y = blockyo[bl] + j*blkdx;

				//if (x >= xo && x <= xmax && y >= yo && y <= ymax)
				{
					//this is safer!
					x = utils::max(utils::min(x, xmax), xo);
					y = utils::max(utils::min(y, ymax), yo);
					// cells that falls off this domain are assigned 
					double x1, x2, y1, y2;
					double q11, q12, q21, q22;
					int cfi, cfip, cfj, cfjp;



					cfi = utils::min(utils::max((int)floor((x - xo) / dx), 0), nx - 2);
					cfip = cfi + 1;

					x1 = xo + dx*cfi;
					x2 = xo + dx*cfip;

					cfj = utils::min(utils::max((int)floor((y - yo) / dx), 0), ny - 2);
					cfjp = cfj + 1;

					y1 = yo + dx*cfj;
					y2 = yo + dx*cfjp;

					q11 = zb[cfi + cfj*nx];
					q12 = zb[cfi + cfjp*nx];
					q21 = zb[cfip + cfj*nx];
					q22 = zb[cfip + cfjp*nx];

					zb_buq[n] = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
					//printf("x=%f\ty=%f\tcfi=%d\tcfj=%d\tn=%d\tzb_buq[n] = %f\n", x,y,cfi,cfj,n,zb_buq[n]);
				}

			}
		}
	}
}

template void interp2BUQ<int>(int nblk, double blksize, double blkdx, double* blockxo, double* blockyo, int nx, int ny, double xo, double xmax, double yo, double ymax, double dx, int * zb, int *&zb_buq);
template void interp2BUQ<float>(int nblk, double blksize, double blkdx, double* blockxo, double* blockyo, int nx, int ny, double xo, double xmax, double yo, double ymax, double dx, float * zb, float *&zb_buq);
template void interp2BUQ<double>(int nblk, double blksize, double blkdx, double* blockxo, double* blockyo, int nx, int ny, double xo, double xmax, double yo, double ymax, double dx, double * zb, double *&zb_buq);



template <class T> void interp2BUQAda(int nblk, double blksize, double bdx, int * activeblk, int * level, double* blockxo, double* blockyo, int nx, int ny, double xo, double xmax, double yo, double ymax, double dx, T* zb, T*& zb_buq)
{
	// This function interpolates the values in bathy maps or roughness map to cf using a bilinear interpolation

	double x, y;
	int n;

	for (int ibl = 0; ibl < nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = activeblk[ibl];
		double blkdx = calcres(bdx, level[ib]);
		for (int j = 0; j < 16; j++)
		{
			for (int i = 0; i < 16; i++)
			{
				n = i + j * 16 + ib * blksize;
				x = blockxo[ib] + i * blkdx;
				y = blockyo[ib] + j * blkdx;

				//if (x >= xo && x <= xmax && y >= yo && y <= ymax)
				{
					//this is safer!
					x = utils::max(utils::min(x, xmax), xo);
					y = utils::max(utils::min(y, ymax), yo);
					// cells that falls off this domain are assigned 
					double x1, x2, y1, y2;
					double q11, q12, q21, q22;
					int cfi, cfip, cfj, cfjp;



					cfi = utils::min(utils::max((int)floor((x - xo) / dx), 0), nx - 2);
					cfip = cfi + 1;

					x1 = xo + dx * cfi;
					x2 = xo + dx * cfip;

					cfj = utils::min(utils::max((int)floor((y - yo) / dx), 0), ny - 2);
					cfjp = cfj + 1;

					y1 = yo + dx * cfj;
					y2 = yo + dx * cfjp;

					q11 = zb[cfi + cfj * nx];
					q12 = zb[cfi + cfjp * nx];
					q21 = zb[cfip + cfj * nx];
					q22 = zb[cfip + cfjp * nx];

					zb_buq[n] = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
					//printf("x=%f\ty=%f\tcfi=%d\tcfj=%d\tn=%d\tzb_buq[n] = %f\n", x,y,cfi,cfj,n,zb_buq[n]);
				}

			}
		}
	}
}

template void interp2BUQAda<int>(int nblk, double blksize, double bdx, int * activeblk, int * level, double* blockxo, double* blockyo, int nx, int ny, double xo, double xmax, double yo, double ymax, double dx, int* zb, int*& zb_buq);
template void interp2BUQAda<float>(int nblk, double blksize, double bdx, int * activeblk, int * level, double* blockxo, double* blockyo, int nx, int ny, double xo, double xmax, double yo, double ymax, double dx, float* zb, float*& zb_buq);
template void interp2BUQAda<double>(int nblk, double blksize, double bdx, int * activeblk, int * level, double* blockxo, double* blockyo, int nx, int ny, double xo, double xmax, double yo, double ymax, double dx, double* zb, double*& zb_buq);


template <class T> void interp2cf(Param XParam, float * cfin, T* blockxo, T* blockyo, T * &cf)
{
	// This function interpolates the values in cfmapin to cf using a bilinear interpolation

	double x, y;
	int n;

	for (int bl = 0; bl < XParam.nblk; bl++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int i = 0; i < 16; i++)
			{
				n = i + j * 16 + bl * XParam.blksize;

				x = blockxo[bl] + i*XParam.dx;
				y = blockyo[bl] + j*XParam.dx;

				if (x >= XParam.roughnessmap.xo && x <= XParam.roughnessmap.xmax && y >= XParam.roughnessmap.yo && y <= XParam.roughnessmap.ymax)
				{
					// cells that falls off this domain are assigned 
					double x1, x2, y1, y2;
					double q11, q12, q21, q22;
					int cfi, cfip, cfj, cfjp;



					cfi = utils::min(utils::max((int)floor((x - XParam.roughnessmap.xo) / XParam.roughnessmap.dx), 0), XParam.roughnessmap.nx - 2);
					cfip = cfi + 1;

					x1 = XParam.roughnessmap.xo + XParam.roughnessmap.dx*cfi;
					x2 = XParam.roughnessmap.xo + XParam.roughnessmap.dx*cfip;

					cfj = utils::min(utils::max((int)floor((y - XParam.roughnessmap.yo) / XParam.roughnessmap.dx), 0), XParam.roughnessmap.ny - 2);
					cfjp = cfj + 1;

					y1 = XParam.roughnessmap.yo + XParam.roughnessmap.dx*cfj;
					y2 = XParam.roughnessmap.yo + XParam.roughnessmap.dx*cfjp;

					q11 = cfin[cfi + cfj*XParam.roughnessmap.nx];
					q12 = cfin[cfi + cfjp*XParam.roughnessmap.nx];
					q21 = cfin[cfip + cfj*XParam.roughnessmap.nx];
					q22 = cfin[cfip + cfjp*XParam.roughnessmap.nx];

					cf[n] = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
				}

			}
		}
	}
}

template void interp2cf<int>(Param XParam, float * cfin, int* blockxo, int* blockyo, int * &cf);
template void interp2cf<float>(Param XParam, float * cfin, float* blockxo, float* blockyo, float * &cf);
template void interp2cf<double>(Param XParam, float * cfin, double* blockxo, double* blockyo, double * &cf);

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


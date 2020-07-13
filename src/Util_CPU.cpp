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


#include "Util_CPU.h"


template <class T> const T& max(const T& a, const T& b) {
	return (a < b) ? b : a;     // or: return comp(a,b)?b:a; for version (2)
}

template <class T> const T& min(const T& a, const T& b) {
	return !(b < a) ? a : b;     // or: return comp(a,b)?b:a; for version (2)
}

template <class T> T sq(T a) {
	return (a * a);
}

unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

template <class T> void AllocateCPU(int nx, int ny, T *&zb)
{
	zb = (T *)malloc(nx*ny * sizeof(T));
	if (!zb)
	{
		fprintf(stderr, "Memory allocation failure\n");

		exit(EXIT_FAILURE);
	}
}

template <class T> void AllocateCPU(int nx, int ny, T *&zs, T *&h, T *&u, T *&v)
{

	zs = AllocateCPU(nx, ny,zs);
	h = AllocateCPU(nx, ny, h);
	u = AllocateCPU(nx, ny, u);
	v = AllocateCPU(nx, ny, v);

}


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

template <class T> void ReallocArray(int nblk, int blksize, T* & zb)
{
	//
	
	zb = (T*)realloc(zb,nblk * blksize * sizeof(T));
	if (zb == NULL)
	{
		fprintf(stderr, "Memory reallocation failure\n");

		exit(EXIT_FAILURE);
	}
	//return nblkmem
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
				ix = min(max((int)round((x - xo) / dx), 0), nx - 1); // min(max( part is overkill?
				iy = min(max((int)round((y - yo) / dx), 0), ny - 1);

				zb_buq[i + j * 16 + b * 256] = zb[ix + iy*nx];
				//printf("bid=%i\ti=%i\tj=%i\tix=%i\tiy=%i\tzb_buq[n]=%f\n", b,i,j,ix, iy, zb_buq[i + j * 16 + b * 256]);
			}
		}
	}
}

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
					x = max(min(x, xmax), xo);
					y = max(min(y, ymax), yo);
					// cells that falls off this domain are assigned 
					double x1, x2, y1, y2;
					double q11, q12, q21, q22;
					int cfi, cfip, cfj, cfjp;



					cfi = min(max((int)floor((x - xo) / dx), 0), nx - 2);
					cfip = cfi + 1;

					x1 = xo + dx*cfi;
					x2 = xo + dx*cfip;

					cfj = min(max((int)floor((y - yo) / dx), 0), ny - 2);
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
					x = max(min(x, xmax), xo);
					y = max(min(y, ymax), yo);
					// cells that falls off this domain are assigned 
					double x1, x2, y1, y2;
					double q11, q12, q21, q22;
					int cfi, cfip, cfj, cfjp;



					cfi = min(max((int)floor((x - xo) / dx), 0), nx - 2);
					cfip = cfi + 1;

					x1 = xo + dx * cfi;
					x2 = xo + dx * cfip;

					cfj = min(max((int)floor((y - yo) / dx), 0), ny - 2);
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



					cfi = min(max((int)floor((x - XParam.roughnessmap.xo) / XParam.roughnessmap.dx), 0), XParam.roughnessmap.nx - 2);
					cfip = cfi + 1;

					x1 = XParam.roughnessmap.xo + XParam.roughnessmap.dx*cfi;
					x2 = XParam.roughnessmap.xo + XParam.roughnessmap.dx*cfip;

					cfj = min(max((int)floor((y - XParam.roughnessmap.yo) / XParam.roughnessmap.dx), 0), XParam.roughnessmap.ny - 2);
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


//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//Copyright (C) 2018 Bosserelle                                                 //
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


#include "Mesh.h"

int CalcInitnblk(Param XParam, Forcing<float> XForcing)
{

	////////////////////////////////////////////////
	// Rearrange the memory in uniform blocks
	////////////////////////////////////////////////


	//max nb of blocks is ceil(nx/16)*ceil(ny/16)
	int nblk = 0;
	int nmask = 0;
	int mloc = 0;

	double levdx = calcres(XParam.dx, XParam.initlevel);

	for (int nblky = 0; nblky < ceil(XForcing.Bathy.ny / XParam.blkwidth); nblky++)
	{
		for (int nblkx = 0; nblkx < ceil(XForcing.Bathy.nx / XParam.blkwidth); nblkx++)
		{
			nmask = 0;
			for (int i = 0; i < 16; i++)
			{
				for (int j = 0; j < 16; j++)
				{
					double x = XParam.xo + (i + XParam.blkwidth * nblkx) * levdx;
					double y = XParam.yo + (j + XParam.blkwidth * nblky) * levdx;

					if (x >= XForcing.Bathy.xo && x <= XForcing.Bathy.xmax && y >= XForcing.Bathy.yo && y <= XForcing.Bathy.ymax)
					{
						// cells that falls off this domain are assigned
						double x1, x2, y1, y2;
						double q11, q12, q21, q22, q;
						int cfi, cfip, cfj, cfjp;



						cfi = utils::min(utils::max((int)floor((x - XForcing.Bathy.xo) / XForcing.Bathy.dx), 0), XForcing.Bathy.nx - 2);
						cfip = cfi + 1;

						x1 = XForcing.Bathy.xo + XForcing.Bathy.dx * cfi;
						x2 = XForcing.Bathy.xo + XForcing.Bathy.dx * cfip;

						cfj = utils::min(utils::max((int)floor((y - XForcing.Bathy.yo) / XForcing.Bathy.dx), 0), XForcing.Bathy.ny - 2);
						cfjp = cfj + 1;

						y1 = XForcing.Bathy.yo + XForcing.Bathy.dx * cfj;
						y2 = XForcing.Bathy.yo + XForcing.Bathy.dx * cfjp;

						q11 = XForcing.Bathy.val[cfi + cfj * XForcing.Bathy.nx];
						q12 = XForcing.Bathy.val[cfi + cfjp * XForcing.Bathy.nx];
						q21 = XForcing.Bathy.val[cfip + cfj * XForcing.Bathy.nx];
						q22 = XForcing.Bathy.val[cfip + cfjp * XForcing.Bathy.nx];

						q = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
						//printf("q = %f\n", q);
						//printf("mloc: %i\n", mloc);
						if (q >= XParam.mask)
							nmask++;
					}
					else
					{
						//computational domnain is outside of the bathy domain
						nmask++;
					}

				}
			}
			if (nmask < XParam.blksize)
				nblk++;
		}
	}

	return nblk;
}

template <class T>
void InitMesh(Param &XParam, Forcing<float> XForcing, Model<T> &XModel)
{
	//=============================
	// Calculate an initial number of block
	int nblk;

	nblk= CalcInitnblk(XParam, XForcing);
		
	XParam.nblk = nblk;
	// allocate a few extra blocks for adaptation
	XParam.nblkmem = (int)ceil(nblk * XParam.membuffer); //5% buffer on the memory for adaptation 

	log("Initial number of blocks: " + std::to_string(nblk) + "; Will be allocating " + std::to_string(nblkmem) + " in memory.");

	//==============================
	// Allocate CPU memory for the whole model
	AllocateCPU(XParam.nblkmem, XParam.blksize, XParam, XModel);

	// Initialise activeblk array as all inactive ( = -1 )
	InitArrayBUQ(XParam.nblkmem, 1, -1, XModel.blocks.active);
	// Initialise level info
	InitArrayBUQ(XParam.nblkmem, 1, XParam.initlevel, XModel.blocks.level);


	
	
	
}

template void InitMesh<float>(Param &XParam, Forcing<float> XForcing, Model<float> &XModel);
template void InitMesh<double>(Param &XParam, Forcing<float> XForcing, Model<double> &XModel);

template <class T> void InitBlockInfo(Param XParam, Forcing<float> XForcing, Model<T> &XModel)
{
	nmask = 0;
	mloc = 0;
	int blkid = 0;
	double levdx = calcres(XParam.dx, XParam.initlevel);

	


	for (int nblky = 0; nblky < ceil(XParam.ny / ((T)XParam.blkwidth)); nblky++)
	{
		for (int nblkx = 0; nblkx < ceil(XParam.nx / ((T)XParam.blkwidth)); nblkx++)
		{
			nmask = 0;
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				for (int j = 0; j < XParam.blkwidth; j++)
				{
					double x = XParam.xo + (i + XParam.blkwidth * nblkx)*levdx;
					double y = XParam.yo + (j + XParam.blkwidth * nblky)*levdx;

					//x = max(min(x, XParam.Bathymetry.xmax), XParam.Bathymetry.xo);
					//y = max(min(y, XParam.Bathymetry.ymax), XParam.Bathymetry.yo);
					
					{
						x = utils::max(utils::min(x, XForcing.Bathy.xmax), XForcing.Bathy.xo);
						y = utils::max(utils::min(y, XForcing.Bathy.ymax), XForcing.Bathy.yo);
						// cells that falls off this domain are assigned
						double x1, x2, y1, y2;
						double q11, q12, q21, q22, q;
						int cfi, cfip, cfj, cfjp;



						cfi = min(max((int)floor((x - XForcing.Bathy.xo) / XForcing.Bathy.dx), 0), XForcing.Bathy.nx - 2);
						cfip = cfi + 1;

						x1 = XForcing.Bathy.xo + XForcing.Bathy.dx*cfi;
						x2 = XForcing.Bathy.xo + XForcing.Bathy.dx*cfip;

						cfj = min(max((int)floor((y - XForcing.Bathy.yo) / XForcing.Bathy.dx), 0), XForcing.Bathy.ny - 2);
						cfjp = cfj + 1;

						y1 = XForcing.Bathy.yo + XForcing.Bathy.dx*cfj;
						y2 = XForcing.Bathy.yo + XForcing.Bathy.dx*cfjp;

						q11 = XForcing.Bathy.val[cfi + cfj*XForcing.Bathy.nx];
						q12 = XForcing.Bathy.val[cfi + cfjp*XForcing.Bathy.nx];
						q21 = XForcing.Bathy.val[cfip + cfj*XForcing.Bathy.nx];
						q22 = XForcing.Bathy.val[cfip + cfjp*XForcing.Bathy.nx];

						q = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
						//printf("q = %f\t q11=%f\t, q12=%f\t, q21=%f\t, q22=%f\t, x1=%f\t, x2=%f\t, y1=%f\t, y2=%f\t, x=%f\t, y=%f\t\n", q, q11, q12, q21, q22, x1, x2, y1, y2, x, y);
						//printf("mloc: %i\n", mloc);
						if (q >= XParam.mask)
							nmask++;
					}
					else
					{
						//computational domnain is outside of the bathy domain
						nmask++;
					}

				}
			}
			if (nmask < 256)
			{
				//
				XModel.blocks.xo[blkid] = XParam.xo + nblkx * ((T)XParam.blkwidth) * levdx;
				XModel.blocks.xo[blkid] = XParam.yo + nblky * ((T)XParam.blkwidth) * levdx;
				XModel.blocks.activeblk[blkid] = blkid;
				//printf("blkxo=%f\tblkyo=%f\n", blockxo_d[blkid], blockyo_d[blkid]);
				blkid++;
			}
		}
	}
}
template void InitBlockInfo<float>(Param XParam, Forcing<float> XForcing, Model<float> &XModel);
template void InitBlockInfo<double>(Param XParam, Forcing<float> XForcing, Model<double> &XModel);


template <class T> void InitArrayBUQ(int nblk, int blkwidth, T initval, T * & Arr)
{
	int blksize = sq(blkwidth);
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

template <class T> void CopyArrayBUQ(int nblk, int blkwidth, T* source, T * & dest)
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


template <class T, class F> void interp2BUQ(Param XParam, BlockP<T> XBlock, F forcing, T*& z)
{
	// This function interpolates the values in bathy maps or roughness map to cf using a bilinear interpolation

	double x, y;
	int n;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XBlock.active[ibl];
		double blkdx = calcres(XParam.dx, level[ib]);
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
					x = utils::max(utils::min(x, F.xmax), F.xo);
					y = utils::max(utils::min(y, F.ymax), F.yo);
					// cells that falls off this domain are assigned 
					double x1, x2, y1, y2;
					double q11, q12, q21, q22;
					int cfi, cfip, cfj, cfjp;



					cfi = utils::min(utils::max((int)floor((x - F.xo) / F.dx), 0), F.nx - 2);
					cfip = cfi + 1;

					x1 = F.xo + F.dx * cfi;
					x2 = F.xo + F.dx * cfip;

					cfj = utils::min(utils::max((int)floor((y - F.yo) / F.dx), 0), F.ny - 2);
					cfjp = cfj + 1;

					y1 = F.yo + F.dx * cfj;
					y2 = F.yo + F.dx * cfjp;

					q11 = zb[cfi + cfj * F.nx];
					q12 = zb[cfi + cfjp * F.nx];
					q21 = zb[cfip + cfj * F.nx];
					q22 = zb[cfip + cfjp * F.nx];

					z[n] = (U)BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
					//printf("x=%f\ty=%f\tcfi=%d\tcfj=%d\tn=%d\tzb_buq[n] = %f\n", x,y,cfi,cfj,n,zb_buq[n]);
				}

			}
		}
	}
}

template void interp2BUQ<float, inputmap>(Param XParam, BlockP<float> XBlock, inputmap forcing, float*& z);
template void interp2BUQ<double, inputmap>(Param XParam, BlockP<double> XBlock, inputmap forcing, double*& z);
template void interp2BUQ<float, forcingmap>(Param XParam, BlockP<float> XBlock, forcingmap forcing, float*& z);
template void interp2BUQ<double, forcingmap>(Param XParam, BlockP<double> XBlock, forcingmap forcing, double*& z);
template void interp2BUQ<float, deformmap>(Param XParam, BlockP<float> XBlock, deformmap forcing, float*& z);
template void interp2BUQ<double, deformmap>(Param XParam, BlockP<double> XBlock, deformmap forcing, double*& z);


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
	//int mloc = 0;

	double levdx = calcres(XParam.dx, XParam.initlevel);

	int maxnbx = ceil(XParam.nx / (double)XParam.blkwidth);
	int maxnby = ceil(XParam.ny / (double)XParam.blkwidth);

	for (int nblky = 0; nblky < maxnby; nblky++)
	{
		for (int nblkx = 0; nblkx < maxnbx; nblkx++)
		{
			nmask = 0;
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				for (int j = 0; j < XParam.blkwidth; j++)
				{
					double x = XParam.xo + (i + XParam.blkwidth * nblkx) * levdx;
					double y = XParam.yo + (j + XParam.blkwidth * nblky) * levdx;

					//if (x >= XForcing.Bathy.xo && x <= XForcing.Bathy.xmax && y >= XForcing.Bathy.yo && y <= XForcing.Bathy.ymax)
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
					//else
					//{
						//computational domnain is outside of the bathy domain
					///	nmask++;
					//}

				}
			}
			if (nmask < (XParam.blkwidth* XParam.blkwidth))
				nblk++;
		}
	}

	return nblk;
}

template <class T>
void InitMesh(Param &XParam, Forcing<float> & XForcing, Model<T> &XModel)
{
	//=============================
	// Calculate an initial number of block
	int nblk;

	nblk= CalcInitnblk(XParam, XForcing);
		
	XParam.nblk = nblk;
	// allocate a few extra blocks for adaptation
	XParam.nblkmem = (int)ceil(nblk * XParam.membuffer); //5% buffer on the memory for adaptation 

	log("Initial number of blocks: " + std::to_string(nblk) + "; Will be allocating " + std::to_string(XParam.nblkmem) + " in memory.");

	//==============================
	// Allocate CPU memory for the whole model
	AllocateCPU(XParam.nblkmem, XParam.blksize, XParam, XModel);

	//==============================
	// Initialise blockinfo info
	InitBlockInfo(XParam, XForcing, XModel.blocks);

	//==============================
	// Init. adaptation info if needed
	if (XParam.maxlevel != XParam.minlevel)
	{
		
		InitBlockadapt(XParam, XModel.blocks, XModel.adapt);
	}

	//==============================
	// Reallocate array containing boundary blocks

	
	
	
}

template void InitMesh<float>(Param &XParam, Forcing<float>& XForcing, Model<float> &XModel);
template void InitMesh<double>(Param &XParam, Forcing<float>& XForcing, Model<double> &XModel);

template <class T> void InitBlockInfo(Param &XParam, Forcing<float> &XForcing, BlockP<T>& XBlock)
{
	//============================
	// Init active and level

	// Initialise activeblk array as all inactive ( = -1 )
	// Here we cannot yet use the InitBlkBUQ function since none of the blk are active
	//InitBlkBUQ(XParam, XBlock, XParam.initlevel, XBlock.level)
	for (int ib = 0; ib < XParam.nblkmem; ib++)
	{
		XBlock.active[ib] = -1;
		XBlock.level[ib] = XParam.initlevel;
	}
	
	

	//============================
	// Init xo, yo and active blk
	InitBlockxoyo(XParam, XForcing, XBlock);

	//============================
	// Init neighbours
	InitBlockneighbours(XParam, XForcing, XBlock);
	//Calcbndblks(XParam, XForcing, XBlock);

}

template <class T> void InitBlockadapt(Param &XParam, BlockP<T> XBlock, AdaptP& XAdap)
{
		InitBlkBUQ(XParam, XBlock, XParam.initlevel, XAdap.newlevel);
		InitBlkBUQ(XParam, XBlock, false, XAdap.coarsen);
		InitBlkBUQ(XParam, XBlock, false, XAdap.refine);
		//InitBlkBUQ(XParam, XBlock, XParam.initlevel, XBlock.level);
		//InitBlkBUQ(XParam, XBlock, XParam.initlevel, XBlock.level);
		//InitArrayBUQ(XParam.nblkmem, 1, 0, XParam.initlevel, XAdap.newlevel);
		//InitArrayBUQ(XParam.nblkmem, 1, 0, false, XAdap.coarsen);
		//InitArrayBUQ(XParam.nblkmem, 1, 0, false, XAdap.refine);


		for (int ibl = 0; ibl < (XParam.nblkmem - XParam.nblk); ibl++)
		{

			XAdap.availblk[ibl] = XParam.nblk + ibl;
			XParam.navailblk++;

		}
	
}
template void InitBlockadapt<float>(Param &XParam, BlockP<float> XBlock, AdaptP& XAdap);
template void InitBlockadapt<double>(Param &XParam, BlockP<double> XBlock, AdaptP& XAdap);



template <class T> void InitBlockxoyo(Param XParam, Forcing<float> XForcing, BlockP<T> &XBlock)
{

	int nmask = 0;
	//mloc = 0;
	int blkid = 0;
	double levdx = calcres(XParam.dx, XParam.initlevel);

	


	for (int nblky = 0; nblky < ceil(XParam.ny / ((double)XParam.blkwidth)); nblky++)
	{
		for (int nblkx = 0; nblkx < ceil(XParam.nx / ((double)XParam.blkwidth)); nblkx++)
		{
			nmask = 0;
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				for (int j = 0; j < XParam.blkwidth; j++)
				{
					double x = XParam.xo + (double(i) + XParam.blkwidth * nblkx)*levdx;
					double y = XParam.yo + (double(j) + XParam.blkwidth * nblky)*levdx;

					//x = max(min(x, XParam.Bathymetry.xmax), XParam.Bathymetry.xo);
					//y = max(min(y, XParam.Bathymetry.ymax), XParam.Bathymetry.yo);
					
					{
						x = utils::max(utils::min(x, XForcing.Bathy.xmax), XForcing.Bathy.xo);
						y = utils::max(utils::min(y, XForcing.Bathy.ymax), XForcing.Bathy.yo);
						// cells that falls off this domain are assigned
						double x1, x2, y1, y2;
						double q11, q12, q21, q22, q;
						int cfi, cfip, cfj, cfjp;



						cfi = utils::min(utils::max((int)floor((x - XForcing.Bathy.xo) / XForcing.Bathy.dx), 0), XForcing.Bathy.nx - 2);
						cfip = cfi + 1;

						x1 = XForcing.Bathy.xo + XForcing.Bathy.dx*cfi;
						x2 = XForcing.Bathy.xo + XForcing.Bathy.dx*cfip;

						cfj = utils::min(utils::max((int)floor((y - XForcing.Bathy.yo) / XForcing.Bathy.dx), 0), XForcing.Bathy.ny - 2);
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
					

				}
			}
			if (nmask < (XParam.blkwidth * XParam.blkwidth))
			{
				//
				XBlock.xo[blkid] = nblkx * ((T)XParam.blkwidth) * levdx;
				XBlock.yo[blkid] = nblky * ((T)XParam.blkwidth) * levdx;
				XBlock.active[blkid] = blkid;
				//printf("blkxo=%f\tblkyo=%f\n", blockxo_d[blkid], blockyo_d[blkid]);
				blkid++;
			}
		}
	}




}
template void InitBlockxoyo<float>(Param XParam, Forcing<float> XForcing, BlockP<float> &XBlock);
template void InitBlockxoyo<double>(Param XParam, Forcing<float> XForcing, BlockP<double> & XBlockP);

template <class T> void InitBlockneighbours(Param &XParam,Forcing<float> &XForcing,  BlockP<T>& XBlock)
{
	// This function will only work if the blocks are uniform
	// A separate function is used for adaptivity
	T leftxo, rightxo, topxo, botxo, leftyo, rightyo, topyo, botyo;

	//====================================
	// First setp up neighbours

	T levdx = calcres(XParam.dx, XParam.initlevel);
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{

		int bl = XBlock.active[ibl];
		T espdist = std::numeric_limits<T>::epsilon() * (T)10.0; // i.e. distances are calculated within 10x theoretical machine precision

		leftxo = XBlock.xo[bl] - ((T)XParam.blkwidth) * levdx;

		leftyo = XBlock.yo[bl];
		rightxo = XBlock.xo[bl] + ((T)XParam.blkwidth) * levdx;
		rightyo = XBlock.yo[bl];
		topxo = XBlock.xo[bl];
		topyo = XBlock.yo[bl] + ((T)XParam.blkwidth) * levdx;
		botxo = XBlock.xo[bl];
		botyo = XBlock.yo[bl] - ((T)XParam.blkwidth) * levdx;

		// by default neighbour block refer to itself. i.e. if the neighbour block is itself then there are no neighbour
		XBlock.LeftBot[bl] = bl;
		XBlock.LeftTop[bl] = bl;
		XBlock.RightBot[bl] = bl;
		XBlock.RightTop[bl] = bl;
		XBlock.TopLeft[bl] = bl;
		XBlock.TopRight[bl] = bl;
		XBlock.BotLeft[bl] = bl;
		XBlock.BotRight[bl] = bl;


		for (int iblb = 0; iblb < XParam.nblk; iblb++)
		{
			//
			int blb = XBlock.active[iblb];

			if (abs(XBlock.xo[blb] - leftxo) < espdist && abs(XBlock.yo[blb] - leftyo) < espdist)
			{
				XBlock.LeftBot[bl] = blb;
				XBlock.LeftTop[bl] = blb;
			}
			if (abs(XBlock.xo[blb] - rightxo) < espdist && abs(XBlock.yo[blb] - rightyo) < espdist)
			{
				XBlock.RightBot[bl] = blb;
				XBlock.RightTop[bl] = blb;
			}
			if (abs(XBlock.xo[blb] - topxo) < espdist && abs(XBlock.yo[blb] - topyo) < espdist)
			{
				XBlock.TopLeft[bl] = blb;
				XBlock.TopRight[bl] = blb;

			}
			if (abs(XBlock.xo[blb] - botxo) < espdist && abs(XBlock.yo[blb] - botyo) < espdist)
			{
				XBlock.BotLeft[bl] = blb;
				XBlock.BotRight[bl] = blb;
			}
		}
	}
		


	//
	

}
template void InitBlockneighbours<float>(Param &XParam,  Forcing<float>& XForcing, BlockP<float>& XBlock);
template void InitBlockneighbours<double>(Param &XParam, Forcing<float>& XForcing, BlockP<double>& XBlock);



template <class T> int CalcMaskblk(Param XParam, Model<T> XModel)
{
	int nmask = 0;
	bool neighbourmask = false;
	T leftxo, leftyo, rightxo, rightyo, topxo, topyo, botxo, botyo;
	T initlevdx = calcres(XParam.dx, XParam.initlevel);

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		T levdx = calcres(XParam.dx, XModel.blocks.level[ib]);

		leftxo = XModel.blocks.xo[ib]; // in adaptive this shoulbe be a range 

		leftyo = XModel.blocks.yo[ib];
		rightxo = XModel.blocks.xo[ib] + (XParam.blkwidth - 1) * levdx;
		rightyo = XModel.blocks.yo[ib];
		topxo = XModel.blocks.xo[ib];
		topyo = XModel.blocks.yo[ib] + (XParam.blkwidth - 1) * levdx;
		botxo = XModel.blocks.xo[ib];
		botyo = XModel.blocks.yo[ib];

		if ((XModel.blocks.LeftBot[ib] == ib || XModel.blocks.LeftTop[ib] == ib) && leftxo > initlevdx)
		{
			neighbourmask = true;
		}
		if ((XModel.blocks.BotLeft[ib] == ib || XModel.blocks.BotRight[ib] == ib) && botyo > initlevdx)
		{
			neighbourmask = true;
		}
		if ((XModel.blocks.TopLeft[ib] == ib || XModel.blocks.TopRight[ib] == ib) && ((topyo - (XParam.ymax - XParam.yo)) < (-1.0 * initlevdx)))
		{
			neighbourmask = true;
		}
		if ((XModel.blocks.RightBot[ib] == ib || XModel.blocks.RightBot[ib] == ib) && ((rightxo - (XParam.xmax - XParam.xo)) < (-1.0 * initlevdx)))
		{
			neighbourmask = true;
		}

		nmask++;

	}

	return nmask;
}


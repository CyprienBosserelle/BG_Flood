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


#include "InitialConditions.h"

template <class T> void InitialConditions(Param &XParam, Forcing<float> &XForcing, Model<T> &XModel)
{
	//=====================================
	// Initialise Bathy data

	interp2BUQ(XParam, XModel.blocks, XForcing.Bathy, XModel.zb);

	// Set edges
	setedges(XParam, XModel.blocks, XModel.zb);

	//=====================================
	// Initialise Friction map

	if (!XForcing.cf.inputfile.empty())
	{
		interp2BUQ(XParam, XModel.blocks, XForcing.cf, XModel.cf);
	}
	else
	{
		InitArrayBUQ(XParam, XModel.blocks, (T)XParam.cf, XModel.cf);
	}
	// Set edges of friction map
	setedges(XParam, XModel.blocks, XModel.cf);

	//=====================================
	// Initial Condition
	
	log("Initial condition:");
	// First calculate the initial values for Evolving parameters (i.e. zs, h, u and v)
	initevolv(XParam, XModel.blocks,XForcing, XModel.evolv, XModel.zb);
	CopyArrayBUQ(XParam, XModel.blocks, XModel.evolv, XModel.evolv_o);
	//=====================================
	// Initial forcing
	InitRivers(XParam, XForcing, XModel);

	//=====================================
	// Initial bndinfo
	Calcbndblks(XParam, XForcing, XModel.blocks);
	Findbndblks(XParam, XModel);

	//=====================================
	// Initialize output variables
	initoutput(XParam, XModel);

}

template void InitialConditions<float>(Param &XParam, Forcing<float> &XForcing, Model<float> &XModel);
template void InitialConditions<double>(Param &XParam, Forcing<float> &XForcing, Model<double> &XModel);

template <class T> void initoutput(Param &XParam, Model<T> &XModel)
{
	
	int ib;
	T levdx;
	FILE* fsSLTS;
	// Initialise all storage involving parameters
	//CopyArrayBUQ(XParam, XModel.blocks, XModel.evolv, XModel.evolv_o);
	if (XParam.outmax)
	{
		CopyArrayBUQ(XParam, XModel.blocks, XModel.evolv, XModel.evmax);
	}
	if (XParam.outmean)
	{
		CopyArrayBUQ(XParam, XModel.blocks, XModel.evolv, XModel.evmean);
	}
	
	if (XParam.TSnodesout.size() > 0)
	{
		FindTSoutNodes(XParam, XModel.blocks);

		for (int o = 0; o < XParam.TSnodesout.size(); o++)
		{
			// Add empty row for each output point
			XModel.TSallout.push_back(std::vector<Pointout>());
		}
	}
	

	//==============================
	// Init. map array
	Initmaparray(XModel);
	
	//==============================
	// Setup output netcdf file
	//XParam = creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo);


}

void InitTSOutput(Param XParam)
{
	for (int o = 0; o < XParam.TSnodesout.size(); o++)
	{
		FILE* fsSLTS;

		//Overwrite existing files
		fsSLTS = fopen(XParam.TSnodesout[o].outname.c_str(), "w");
		fprintf(fsSLTS, "# x=%f\ty=%f\ti=%d\tj=%d\tblock=%dt%s\n", XParam.TSnodesout[o].x, XParam.TSnodesout[o].y, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSnodesout[o].block, XParam.TSnodesout[o].outname.c_str());
		fclose(fsSLTS);
		

	}
}

template <class T> void FindTSoutNodes(Param& XParam, BlockP<T> XBlock)
{
	int ib;
	T levdx;
	
	// Initialise all storage involving parameters
	
	
	for (int o = 0; o < XParam.TSnodesout.size(); o++)
	{


		//find the block where point belongs
		for (int blk = 0; blk < XParam.nblk; blk++)
		{

			ib = XBlock.active[blk];
			levdx = calcres(XParam.dx,XBlock.level[ib]);
			if (XParam.TSnodesout[o].x >= XBlock.xo[ib] && XParam.TSnodesout[o].x <= (XBlock.xo[ib] + (T)(XParam.blkwidth - 1) * levdx) && XParam.TSnodesout[o].y >= XBlock.yo[o] && XParam.TSnodesout[o].y <= (XBlock.yo[ib] + (T)(XParam.blkwidth - 1) * levdx))
			{
				XParam.TSnodesout[o].block = ib;
				XParam.TSnodesout[o].i = min(max((int)round((XParam.TSnodesout[o].x - XBlock.xo[ib]) / levdx), 0), XParam.blkwidth - 1);
				XParam.TSnodesout[o].j = min(max((int)round((XParam.TSnodesout[o].y - XBlock.yo[ib]) / levdx), 0), XParam.blkwidth - 1);
				break;
			}
		}
	}
	

}
template void FindTSoutNodes<float>(Param& XParam, BlockP<float> XBlock);
template void FindTSoutNodes<double>(Param& XParam, BlockP<double> XBlock);



template <class T> void InitRivers(Param XParam, Forcing<float> &XForcing, Model<T> &XModel)
{
	//========================
	// River discharge

	if (XForcing.rivers.size() > 0)
	{
		//
		double xx, yy;
		int n,ib;
		double levdx;
		log("Initializing rivers");
		//For each rivers
		for (int Rin = 0; Rin < XForcing.rivers.size(); Rin++)
		{
			// find the cells where the river discharge will be applied
			std::vector<int> idis, jdis, blockdis;
			for (int ibl = 0; ibl < XParam.nblk; ibl++)
			{
				ib = XModel.blocks.active[ibl];
				levdx = calcres(XParam.dx, XModel.blocks.level[ib]);
				for (int j = 0; j < XParam.blkwidth; j++)
				{
					for (int i = 0; i < XParam.blkwidth; i++)
					{
						int n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
						
						
						xx = XModel.blocks.xo[ib] + i * levdx;
						yy = XModel.blocks.yo[ib] + j * levdx;
						// the conditions are that the discharge area as defined by the user have to include at least a model grid node
						// This could be really annoying and there should be a better way to deal wiith this like polygon intersection
						if (xx >= XForcing.rivers[Rin].xstart && xx <= XForcing.rivers[Rin].xend && yy >= XForcing.rivers[Rin].ystart && yy <= XForcing.rivers[Rin].yend)
						{

							// This cell belongs to the river discharge area
							idis.push_back(i);
							jdis.push_back(j);
							blockdis.push_back(ib);

						}
					}
				}

			}

			XForcing.rivers[Rin].i = idis;
			XForcing.rivers[Rin].j = jdis;
			XForcing.rivers[Rin].block = blockdis;
			XForcing.rivers[Rin].disarea = idis.size() * levdx * levdx; // That is not valid for spherical grids

			
		}
		//Now identify sort and unique blocks where rivers are being inserted
		std::vector<int> activeRiverBlk;

		for (int Rin = 0; Rin < XForcing.rivers.size(); Rin++)
		{

			activeRiverBlk.insert(std::end(activeRiverBlk), std::begin(XForcing.rivers[Rin].block), std::end(XForcing.rivers[Rin].block));
		}
		std::sort(activeRiverBlk.begin(), activeRiverBlk.end());
		activeRiverBlk.erase(std::unique(activeRiverBlk.begin(), activeRiverBlk.end()), activeRiverBlk.end());
		if (activeRiverBlk.size() > XModel.bndblk.nblkriver)
		{
			ReallocArray(activeRiverBlk.size(), 1, XModel.bndblk.river);
		}

		XModel.bndblk.nblkriver = activeRiverBlk.size();

		for (int b = 0; b < activeRiverBlk.size(); b++)
		{
			XModel.bndblk.river[b] = activeRiverBlk[b];
		}
		
	}


}

template void InitRivers<float>(Param XParam, Forcing<float> &XForcing, Model<float> &XModel);
template void InitRivers<double>(Param XParam, Forcing<float> &XForcing, Model<double> &XModel);


template<class T> void Initmaparray(Model<T>& XModel)
{
	XModel.OutputVarMap["zb"] = XModel.zb;

	XModel.OutputVarMap["u"] = XModel.evolv.u;

	XModel.OutputVarMap["v"] = XModel.evolv.v;

	XModel.OutputVarMap["zs"] = XModel.evolv.zs;

	XModel.OutputVarMap["h"] = XModel.evolv.h;

	XModel.OutputVarMap["hmean"] = XModel.evmean.h;

	XModel.OutputVarMap["hmax"] = XModel.evmax.h;

	XModel.OutputVarMap["zsmean"] = XModel.evmean.zs;

	XModel.OutputVarMap["zsmax"] = XModel.evmax.zs;

	XModel.OutputVarMap["umean"] = XModel.evmean.u;

	XModel.OutputVarMap["umax"] = XModel.evmax.u;

	XModel.OutputVarMap["vmean"] = XModel.evmean.v;

	XModel.OutputVarMap["vmax"] = XModel.evmax.v;

	XModel.OutputVarMap["vort"] = XModel.vort;

}

template void Initmaparray<float>(Model<float>& XModel);
template void Initmaparray<double>(Model<double>& XModel);



template <class T> void Calcbndblks(Param& XParam, Forcing<float>& XForcing, BlockP<T> XBlock)
{
	//=====================================
	// Find how many blocks are on each bnds
	int blbr = 0, blbb = 0, blbl = 0, blbt = 0;
	T leftxo, leftyo, rightxo, rightyo, topxo, topyo, botxo, botyo;

	T initlevdx = calcres(XParam.dx, XParam.initlevel);

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		double espdist = 0.00000001;///WARMING

		int ib = XBlock.active[ibl];

		T levdx = calcres(XParam.dx, XBlock.level[ib]);

		leftxo = XBlock.xo[ib]; // in adaptive this shoulbe be a range 

		leftyo = XBlock.yo[ib];
		rightxo = XBlock.xo[ib] + (XParam.blkwidth - 1) * levdx;
		rightyo = XBlock.yo[ib];
		topxo = XBlock.xo[ib];
		topyo = XBlock.yo[ib] + (XParam.blkwidth - 1) * levdx;
		botxo = XBlock.xo[ib];
		botyo = XBlock.yo[ib];

		if ((rightxo - XParam.xmax) > (-1.0 * initlevdx))
		{
			//
			blbr++;
			//bndrightblk[blbr] = bl;

		}

		if ((topyo - XParam.ymax) > (-1.0 * initlevdx))
		{
			//
			blbt++;
			//bndtopblk[blbt] = bl;

		}
		if ((XParam.yo - botyo) > (-1.0 * initlevdx))
		{
			//
			blbb++;
			//bndbotblk[blbb] = bl;

		}
		if ((XParam.xo - leftxo) > (-1.0 * initlevdx))
		{
			//
			blbl++;
			//bndleftblk[blbl] = bl;

		}
	}

	// fill
	XForcing.left.nblk = blbl;
	XForcing.right.nblk = blbr;
	XForcing.top.nblk = blbt;
	XForcing.bot.nblk = blbb;

	XParam.nbndblkleft = blbl;
	XParam.nbndblkright = blbr;
	XParam.nbndblktop = blbt;
	XParam.nbndblkbot = blbb;


}


template <class T> void Findbndblks(Param XParam, Model<T>& XModel)
{
	//=====================================
	// Find how many blocks are on each bnds
	int blbr = 0, blbb = 0, blbl = 0, blbt = 0;
	BlockP<T> XBlock = XModel.blocks;
	T initlevdx = calcres(XParam.dx, XParam.initlevel);
	T leftxo, leftyo, rightxo, rightyo, topxo, topyo, botxo, botyo;


	// Reallocate array if necessary
	ReallocArray(XParam.nbndblkleft, 1, XModel.bndblk.left);
	ReallocArray(XParam.nbndblkright, 1, XModel.bndblk.right);
	ReallocArray(XParam.nbndblktop, 1, XModel.bndblk.top);
	ReallocArray(XParam.nbndblkbot, 1, XModel.bndblk.bot);

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		double espdist = 0.00000001;///WARMING

		int ib = XBlock.active[ibl];
		T levdx = calcres(XParam.dx, XModel.blocks.level[ib]);


		leftxo = XBlock.xo[ib]; // in adaptive this shoulbe be a range 

		leftyo = XBlock.yo[ib];
		rightxo = XBlock.xo[ib] + (XParam.blkwidth - 1) * levdx;
		rightyo = XBlock.yo[ib];
		topxo = XBlock.xo[ib];
		topyo = XBlock.yo[ib] + (XParam.blkwidth - 1) * levdx;
		botxo = XBlock.xo[ib];
		botyo = XBlock.yo[ib];

		if ((rightxo - XParam.xmax) > (-1.0 * initlevdx))
		{
			//
			XModel.bndblk.right[blbr] = ib;
			blbr++;

		}

		if ((topyo - XParam.ymax) > (-1.0 * initlevdx))
		{
			//
			XModel.bndblk.top[blbt] = ib;
			blbt++;

		}
		if ((XParam.yo - botyo) > (-1.0 * initlevdx))
		{
			//
			XModel.bndblk.bot[blbb] = ib;
			blbb++;

		}
		if ((XParam.xo - leftxo) > (-1.0 * initlevdx))
		{
			//
			XModel.bndblk.left[blbl] = ib;
			blbl++;

		}
	}




}


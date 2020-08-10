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
	initForcing(XParam, XForcing, XModel);

	//=====================================
	// Initial bndinfo
	Initbnds(XParam, XForcing, XModel);
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
	CopyArrayBUQ(XParam, XModel.blocks, XModel.evolv, XModel.evolv_o);
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
		// Allocate storage memory
		AllocateCPU(XParam.maxTSstorage, 1.0, XModel.TSstore);
		for (int o = 0; o < XParam.TSnodesout.size(); o++)
		{


			//find the block where point belongs
			for (int blk = 0; blk < XParam.nblk; blk++)
			{
				
				ib = XModel.blocks.active[blk];
				levdx = calcres(XParam.dx, XModel.blocks.level[ib]);
				if (XParam.TSnodesout[o].x >= XModel.blocks.xo[ib] && XParam.TSnodesout[o].x <= (XModel.blocks.xo[ib] + (T)(XParam.blkwidth-1) * levdx) && XParam.TSnodesout[o].y >= XModel.blocks.yo[o] && XParam.TSnodesout[o].y <= (XModel.blocks.yo[ib] + (T)(XParam.blkwidth-1) * levdx))
				{
					XParam.TSnodesout[o].block = ib;
					XParam.TSnodesout[o].i = min(max((int)round((XParam.TSnodesout[o].x - XModel.blocks.xo[ib]) / levdx), 0), XParam.blkwidth - 1);
					XParam.TSnodesout[o].j = min(max((int)round((XParam.TSnodesout[o].y - XModel.blocks.yo[ib]) / levdx), 0), XParam.blkwidth - 1);
					break;
				}
			}
			//Overwrite existing files
			fsSLTS = fopen(XParam.TSnodesout[o].outname.c_str(), "w");
			fprintf(fsSLTS, "# x=%f\ty=%f\ti=%d\tj=%d\tblock=%dt%s\n", XParam.TSnodesout[o].x, XParam.TSnodesout[o].y, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSnodesout[o].block, XParam.TSnodesout[o].outname.c_str());
			fclose(fsSLTS);

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

template <class T> void initForcing(Param XParam, Forcing<float> &XForcing, Model<T> &XModel)
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
		
		AllocateCPU(activeRiverBlk.size(), 1, XModel.bndblk.river);

		XModel.bndblk.nriverblk = activeRiverBlk.size();

		for (int b = 0; b < activeRiverBlk.size(); b++)
		{
			XModel.bndblk.river[b] = activeRiverBlk[b];
		}
		
	}


}

template void initForcing<float>(Param XParam, Forcing<float> &XForcing, Model<float> &XModel);
template void initForcing<double>(Param XParam, Forcing<float> &XForcing, Model<double> &XModel);


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

template <class T> void Initbnds(Param XParam,Forcing<float> XForcing, Model<T>& XModel)
{
	// Initialise bnd block info 
	//XParam.leftbnd.nblk were calculated in 
	AllocateCPU(XForcing.left.nblk, 1, XModel.bndblk.left);
	AllocateCPU(XForcing.right.nblk, 1, XModel.bndblk.right);
	AllocateCPU(XForcing.top.nblk, 1, XModel.bndblk.top);
	AllocateCPU(XForcing.bot.nblk, 1, XModel.bndblk.bot);
	
	int blbr, blbb, blbl, blbt;
	int leftxo, leftyo, rightxo, rightyo, topxo, topyo, botxo, botyo;

	blbr = blbb = blbl = blbt = 0;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		double espdist = 0.00000001;///WARMING
		int ib = XModel.blocks.active[ibl];

		T levdx = calcres(XParam.dx, XModel.blocks.level[ib]);

		leftxo = XModel.blocks.xo[ib]; // in adaptive this shoulbe be a range
		leftyo = XModel.blocks.yo[ib];
		rightxo = XModel.blocks.xo[ib] + (XParam.blkwidth-1) * levdx;
		rightyo = XModel.blocks.yo[ib];
		topxo = XModel.blocks.xo[ib];
		topyo = XModel.blocks.yo[ib] + (XParam.blkwidth - 1) * levdx;
		botxo = XModel.blocks.xo[ib];
		botyo = XModel.blocks.yo[ib];

		if ((rightxo - XParam.xmax) > (-1.0 * levdx))
		{
			//

			XModel.bndblk.right[blbr] = ib;
			blbr++;

		}

		if ((topyo - XParam.ymax) > (-1.0 * levdx))
		{
			//

			XModel.bndblk.top[blbt] = ib;
			blbt++;

		}
		if ((XParam.yo - botyo) > (-1.0 * levdx))
		{
			//

			XModel.bndblk.bot[blbb] = ib;
			blbb++;

		}
		if ((XParam.xo - leftxo) > (-1.0 * levdx))
		{
			//

			XModel.bndblk.left[blbl] = ib;
			blbl++;
			

		}
	}
	

}
template void Initbnds<float>(Param XParam, Forcing<float> XForcing, Model<float>& XModel);
template void Initbnds<double>(Param XParam, Forcing<float> XForcing, Model<double>& XModel);

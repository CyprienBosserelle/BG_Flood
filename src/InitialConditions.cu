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
	
	log("\nInitial condition:");
	// First calculate the initial values for Evolving parameters (i.e. zs, h, u and v)
	initevolv(XParam, XModel.blocks,XForcing, XModel.evolv, XModel.zb);
	CopyArrayBUQ(XParam, XModel.blocks, XModel.evolv, XModel.evolv_o);
	//=====================================
	// Initial forcing
	InitRivers(XParam, XForcing, XModel);

	//=====================================
	// Initial bndinfo
	Calcbndblks(XParam, XForcing, XModel.blocks);
	Findbndblks(XParam, XModel, XForcing);

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
	//FILE* fsSLTS;
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
		FindTSoutNodes(XParam, XModel.blocks,XModel.bndblk);
	}
	

	//==============================
	// Init. map array
	Initmaparray(XModel);
	// Init. zones for output
	Initoutzone(XParam, XModel.blocks);
	
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
		fprintf(fsSLTS, "# x=%f\ty=%f\ti=%d\tj=%d\tblock=%d\t%s\n", XParam.TSnodesout[o].x, XParam.TSnodesout[o].y, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSnodesout[o].block, XParam.TSnodesout[o].outname.c_str());

		fprintf(fsSLTS, "# time[s]\tzs[m]\th[m]\tu[m/s]\tv[m/s]\n");

		fclose(fsSLTS);
		

	}
}

template <class T> void FindTSoutNodes(Param& XParam, BlockP<T> XBlock, BndblockP & bnd)
{
	int ib;
	T levdx;
	bnd.nblkTs = XParam.TSnodesout.size();

	AllocateCPU(bnd.nblkTs, 1, bnd.Tsout);

	// Initialise all storage involving parameters
	
	
	for (int o = 0; o < XParam.TSnodesout.size(); o++)
	{


		//find the block where point belongs
		for (int blk = 0; blk < XParam.nblk; blk++)
		{

			ib = XBlock.active[blk];
			levdx = calcres(XParam.dx,XBlock.level[ib]);
			if (XParam.TSnodesout[o].x >= (XParam.xo + XBlock.xo[ib]) && XParam.TSnodesout[o].x <= (XParam.xo + XBlock.xo[ib] + (T)(XParam.blkwidth - 1) * levdx) && XParam.TSnodesout[o].y >= (XParam.yo + XBlock.yo[ib]) && XParam.TSnodesout[o].y <= (XParam.yo + XBlock.yo[ib] + (T)(XParam.blkwidth - 1) * levdx))
			{
				XParam.TSnodesout[o].block = ib;
				XParam.TSnodesout[o].i = min(max((int)round((XParam.TSnodesout[o].x - (XParam.xo + XBlock.xo[ib])) / levdx), 0), XParam.blkwidth - 1);
				XParam.TSnodesout[o].j = min(max((int)round((XParam.TSnodesout[o].y - (XParam.yo + XBlock.yo[ib])) / levdx), 0), XParam.blkwidth - 1);
				break;
			}
		}
		bnd.Tsout[o] = ib;
	}
	

}
template void FindTSoutNodes<float>(Param& XParam, BlockP<float> XBlock, BndblockP& bnd);
template void FindTSoutNodes<double>(Param& XParam, BlockP<double> XBlock, BndblockP& bnd);



template <class T> void InitRivers(Param XParam, Forcing<float> &XForcing, Model<T> &XModel)
{
	//========================
	// River discharge

	if (XForcing.rivers.size() > 0)
	{
		//
		double xl, yb, xr, yt ;
		int n,ib;
		double levdx;
		double dischargeArea = 0.0;
		log("\tInitializing rivers");
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
						
						
						xl = XParam.xo + XModel.blocks.xo[ib] + i * levdx - 0.5 * levdx;
						yb = XParam.yo + XModel.blocks.yo[ib] + j * levdx - 0.5 * levdx;

						xr = XParam.xo + XModel.blocks.xo[ib] + i * levdx + 0.5 * levdx;
						yt = XParam.yo + XModel.blocks.yo[ib] + j * levdx + 0.5 * levdx;
						// the conditions are that the discharge area as defined by the user have to include at least a model grid node
						// This could be really annoying and there should be a better way to deal wiith this like polygon intersection
						//if (xx >= XForcing.rivers[Rin].xstart && xx <= XForcing.rivers[Rin].xend && yy >= XForcing.rivers[Rin].ystart && yy <= XForcing.rivers[Rin].yend)
						if (OBBdetect(xl, xr, yb, yt, XForcing.rivers[Rin].xstart, XForcing.rivers[Rin].xend, XForcing.rivers[Rin].ystart, XForcing.rivers[Rin].yend))
						{

							// This cell belongs to the river discharge area
							idis.push_back(i);
							jdis.push_back(j);
							blockdis.push_back(ib);
							dischargeArea = dischargeArea + levdx * levdx;
						}
					}
				}

			}

			XForcing.rivers[Rin].i = idis;
			XForcing.rivers[Rin].j = jdis;
			XForcing.rivers[Rin].block = blockdis;
			XForcing.rivers[Rin].disarea = dischargeArea; // That is not valid for spherical grids

			
		}
		//Now identify sort and unique blocks where rivers are being inserted
		std::vector<int> activeRiverBlk;

		for (int Rin = 0; Rin < XForcing.rivers.size(); Rin++)
		{

			activeRiverBlk.insert(std::end(activeRiverBlk), std::begin(XForcing.rivers[Rin].block), std::end(XForcing.rivers[Rin].block));
		}
		std::sort(activeRiverBlk.begin(), activeRiverBlk.end());
		activeRiverBlk.erase(std::unique(activeRiverBlk.begin(), activeRiverBlk.end()), activeRiverBlk.end());
		if (activeRiverBlk.size() > size_t(XModel.bndblk.nblkriver))
		{
			ReallocArray(activeRiverBlk.size(), 1, XModel.bndblk.river);
			XModel.bndblk.nblkriver = activeRiverBlk.size();
		}

		

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
	//Main Parameters
	XModel.OutputVarMap["zb"] = XModel.zb;

	XModel.OutputVarMap["u"] = XModel.evolv.u;

	XModel.OutputVarMap["v"] = XModel.evolv.v;

	XModel.OutputVarMap["zs"] = XModel.evolv.zs;

	XModel.OutputVarMap["h"] = XModel.evolv.h;

	//Mean Max parameters
	XModel.OutputVarMap["hmean"] = XModel.evmean.h;

	XModel.OutputVarMap["hmax"] = XModel.evmax.h;

	XModel.OutputVarMap["zsmean"] = XModel.evmean.zs;

	XModel.OutputVarMap["zsmax"] = XModel.evmax.zs;

	XModel.OutputVarMap["umean"] = XModel.evmean.u;

	XModel.OutputVarMap["umax"] = XModel.evmax.u;

	XModel.OutputVarMap["vmean"] = XModel.evmean.v;

	XModel.OutputVarMap["vmax"] = XModel.evmax.v;

	XModel.OutputVarMap["vort"] = XModel.vort;

	//others

	XModel.OutputVarMap["uo"] = XModel.evolv_o.u;

	XModel.OutputVarMap["vo"] = XModel.evolv_o.v;

	XModel.OutputVarMap["zso"] = XModel.evolv_o.zs;

	XModel.OutputVarMap["ho"] = XModel.evolv_o.h;

	// Gradients

	XModel.OutputVarMap["dhdx"] = XModel.grad.dhdx;

	XModel.OutputVarMap["dhdy"] = XModel.grad.dhdy;

	XModel.OutputVarMap["dudx"] = XModel.grad.dudx;

	XModel.OutputVarMap["dudy"] = XModel.grad.dudy;

	XModel.OutputVarMap["dvdx"] = XModel.grad.dvdx;

	XModel.OutputVarMap["dvdy"] = XModel.grad.dvdy;

	XModel.OutputVarMap["dzsdx"] = XModel.grad.dzsdx;

	XModel.OutputVarMap["dzsdy"] = XModel.grad.dzsdy;

	//Flux
	XModel.OutputVarMap["Fhu"] = XModel.flux.Fhu;

	XModel.OutputVarMap["Fhv"] = XModel.flux.Fhv;

	XModel.OutputVarMap["Fqux"] = XModel.flux.Fqux;

	XModel.OutputVarMap["Fqvy"] = XModel.flux.Fqvy;

	XModel.OutputVarMap["Fquy"] = XModel.flux.Fquy;

	XModel.OutputVarMap["Fqvx"] = XModel.flux.Fqvx;

	XModel.OutputVarMap["Su"] = XModel.flux.Su;

	XModel.OutputVarMap["Sv"] = XModel.flux.Sv;

	//Advance
	XModel.OutputVarMap["dh"] = XModel.adv.dh;

	XModel.OutputVarMap["dhu"] = XModel.adv.dhu;

	XModel.OutputVarMap["dhv"] = XModel.adv.dhv;

	XModel.OutputVarMap["cf"] = XModel.cf;
}

template void Initmaparray<float>(Model<float>& XModel);
template void Initmaparray<double>(Model<double>& XModel);


// Initialise all storage involving parameter the outzone objects
template <class T> void Findoutzoneblks(Param& XParam, BlockP<T> XBlock)
{
	int ib;
	T levdx;
	outzone Xzone;

	// Find the blocks to output and the corners of this area for each zone
	for (int o = 0; o < XParam.outzone.size(); o++)
	{

		Xzone = XParam.outzone[o];
		// Find the blocks to output for each zone (and the corner of this area) 
		//
		//We want the samller rectangular area, composed of full blocks, 
		//containing the area defined by the user. 
		//- If all the blocks have the same resolution, at least a part of the block
		//must be inside the user defined rectangular
		// -If there is blocks of different resolutions in the area, the corners of the area
		// must be defined first to have a rectangular zone. Then, a new pass through all blocks
		// identify the blocks inside this new defined zone.
		
		std::vector<int> blkzone;
		double xbl, ybl, xtl, ytl, xtr, ytr, xbr, ybr, xo, yo, xmax, ymax;
		double xl, xr, yb, yt;
		int ibl, itl, ibr, itr;

		int nblk = 0;

		//Getting the new area's corners
		for (int ibl = 0; ibl < XParam.nblk; ibl++)
		{
			ib = XBlock.active[ibl];
			levdx = calcres(XParam.dx, XBlock.level[ib]);

			// get the corners' locations of the block (center of the corner cell)
			xl = XParam.xo + XBlock.xo[ib];
			yb = XParam.yo + XBlock.yo[ib];
			xr = XParam.xo + XBlock.xo[ib] + (XParam.blkwidth - 1) * levdx;
			yt = XParam.yo + XBlock.yo[ib] + (XParam.blkwidth - 1) * levdx;

			// Getting the bottom left corner coordinate of the output area
			if (XParam.outzone[o].xstart >= xl && XParam.outzone[o].xstart <= xr && XParam.outzone[o].ystart >= yb && XParam.outzone[o].ystart <= yt)
			{
				xbl = XBlock.xo[ib];
				ybl = XBlock.yo[ib];
			}
			// Getting the top left corner coordinate of the output area
			if (XParam.outzone[o].xstart >= xl && XParam.outzone[o].xstart <= xr && XParam.outzone[o].yend >= yb && XParam.outzone[o].yend <= yt)
			{
				xtl = XBlock.xo[ib];
				ytl = XBlock.yo[ib];
			}
			// Getting the top right corner coordinate of the output area
			if (XParam.outzone[o].xend >= xl && XParam.outzone[o].xend <= xr && XParam.outzone[o].ystart >= yb && XParam.outzone[o].ystart <= yt)
			{
				xtr = XBlock.xo[ib];
				ytr = XBlock.yo[ib];
			}
			// Getting the bottom right corner coordinate of the output area
			if (XParam.outzone[o].xend >= xl && XParam.outzone[o].xend <= xr && XParam.outzone[o].ystart >= yb && XParam.outzone[o].ystart <= yt)
			{
				xbr = XBlock.xo[ib];
				ybr = XBlock.yo[ib];
			}
		}
		// get the minimal rectangle
		xo = min(xbl, xtl);
		yo = min(ybl, ybr);
		xmax = max(xtr, xbr);
		ymax = max(ytr, ytl);
		
		//This minimal rectangular can include only part of blocks depending of resolution
		//the blocks containing the corners are found and the lager block impose its border on each side
		for (int ibl = 0; ibl < XParam.nblk; ibl++)
		{
			ib = XBlock.active[ibl];
			levdx = calcres(XParam.dx, XBlock.level[ib]);

			// get the corners' locations of the block (center of the corner cell)
			xl = XParam.xo + XBlock.xo[ib];
			yb = XParam.yo + XBlock.yo[ib];
			xr = XParam.xo + XBlock.xo[ib] + (XParam.blkwidth - 1) * levdx;
			yt = XParam.yo + XBlock.yo[ib] + (XParam.blkwidth - 1) * levdx;

			// Getting the bottom left corner coordinate of the output area
			if (xo >= xl && xo <= xr && yo >= yb && yo <= yt)
			{
				ibl = ib;
			}
			// Getting the top left corner coordinate of the output area
			if (xo >= xl && xo <= xr && ymax >= yb && ymax <= yt)
			{
				itl = ib;
			}
			// Getting the top right corner coordinate of the output area
			if (xmax >= xl && xmax <= xr && yo >= yb && yo <= yt)
			{
				itr = ib;
			}
			// Getting the bottom right corner coordinate of the output area
			if (xmax >= xl && xmax <= xr && ymax >= yb && ymax <= yt)
			{
				ibr = ib;
			}
		}

		// for each side, the border is imposed by the larger block, of the "further out" one.
		XBlock.outZone[o].xo = min(XBlock.xo[ibl], XBlock.xo[itl]);
		XBlock.outZone[o].yo = min(XBlock.yo[ibl], XBlock.yo[ibr]);
		XBlock.outZone[o].xmax = max(XBlock.xo[itr], XBlock.xo[ibr]);
		XBlock.outZone[o].ymax = max(XBlock.yo[itr], XBlock.yo[itl]);
		
		// Get the list of all blocks in the zone
		for (int ibl = 0; ibl < XParam.nblk; ibl++)
		{
			double xbl, ybl, xtl, ytl, xtr, ytr, xbr, ybr;
			ib = XBlock.active[ibl];
			levdx = calcres(XParam.dx, XBlock.level[ib]);

			// get the corners' locations of the block (center of the corner cell)
			xl = XParam.xo + XBlock.xo[ib];
			yb = XParam.yo + XBlock.yo[ib];
			xr = XParam.xo + XBlock.xo[ib] + (XParam.blkwidth - 1) * levdx;
			yt = XParam.yo + XBlock.yo[ib] + (XParam.blkwidth - 1) * levdx;

			// Checking if at least one part of the a cell of the block is 
			// inside the area defined by the user.
			if (OBBdetect(xl, xr, yb, yt, XBlock.outZone[o].xo, XBlock.outZone[o].xmax, XBlock.outZone[o].yo, XBlock.outZone[o].ymax))
			{
				// This block belongs to the output zone defined by the user
				blkzone.push_back(ib);
				nblk++;
			}
		}
		XBlock.outZone[o].nblk = nblk;
		ReallocArray(blkzone.size(), 1, XBlock.outZone[o].blk);
		for (int b = 0; b < blkzone.size(); b++)
		{
			XBlock.outZone[o].blk[b] = blkzone[b];
		}
		XBlock.outZone[o].outname = XParam.outzone[o].outname;
	}

}
template void Findoutzoneblks<float>(Param& XParam, BlockP<float> XBlock);
template void Findoutzoneblks<double>(Param& XParam, BlockP<double> XBlock);

template <class T> void Initoutzone(Param& XParam, BlockP<T> XBlock)
{
	//The domain full domain is defined as the output zone by default 
	//(and the blocks have been initialised by default)
	// If a zone for the output has been requested by the user, the blocks in the 
	// zone and the corners are computed here:
	if (XParam.outzone.size() > 0)
	{
		Findoutzoneblks(XParam, XBlock);
	}
	else
	{
		std::vector<int> blksall;
		//Define the full domain as a zone
		XBlock.outZone[0].outname = XParam.outfile;
		XBlock.outZone[0].xo = XParam.xo;
		XBlock.outZone[0].yo = XParam.yo;
		XBlock.outZone[0].xmax = XParam.xmax;
		XBlock.outZone[0].ymax = XParam.ymax;
		XBlock.outZone[0].ymax = XParam.nblk;
		ReallocArray(XParam.nblk, 1, XBlock.outZone[0].blk);
		for (int ib = 0; ib < XParam.nblk; ib++)
		{
			XBlock.outZone[0].blk[ib]=ib;
		}
	}
}
template void Initoutzone<float>(Param& XParam, BlockP<float> XBlock);
template void Initoutzone<double>(Param& XParam, BlockP<double> XBlock);


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

		leftyo =XBlock.yo[ib];
		rightxo = XBlock.xo[ib] + (XParam.blkwidth - 1) * levdx;
		rightyo = XBlock.yo[ib];
		topxo = XBlock.xo[ib];
		topyo = XBlock.yo[ib] + (XParam.blkwidth - 1) * levdx;
		botxo = XBlock.xo[ib];
		botyo = XBlock.yo[ib];

		if ((rightxo - (XParam.xmax-XParam.xo)) > (-1.0 * levdx))
		{
			//
			blbr++;
			//bndrightblk[blbr] = bl;

		}

		if ((topyo - (XParam.ymax - XParam.yo)) > (-1.0 * levdx))
		{
			//
			blbt++;
			//bndtopblk[blbt] = bl;

		}
		if (botyo < levdx)
		{
			//
			blbb++;
			//bndbotblk[blbb] = bl;

		}
		if (leftxo < levdx)
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





template <class T> void Findbndblks(Param XParam, Model<T> XModel,Forcing<float> &XForcing)
{
	//=====================================
	// Find how many blocks are on each bnds
	int blbr = 0, blbb = 0, blbl = 0, blbt = 0;
	BlockP<T> XBlock = XModel.blocks;
	T initlevdx = calcres(XParam.dx, XParam.initlevel);
	T leftxo, leftyo, rightxo, rightyo, topxo, topyo, botxo, botyo;


	// Reallocate array if necessary
	ReallocArray(XParam.nbndblkleft, 1, XForcing.left.blks);
	ReallocArray(XParam.nbndblkright, 1, XForcing.right.blks);
	ReallocArray(XParam.nbndblktop, 1, XForcing.top.blks);
	ReallocArray(XParam.nbndblkbot, 1, XForcing.bot.blks);

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

		if ((rightxo - (XParam.xmax-XParam.xo)) > (-1.0 * levdx))
		{
			//
			XForcing.right.blks[blbr] = ib;
			blbr++;

		}

		if ((topyo - (XParam.ymax-XParam.yo)) > (-1.0 * levdx))
		{
			//
			XForcing.top.blks[blbt] = ib;
			blbt++;

		}
		if (botyo < levdx)
		{
			//
			XForcing.bot.blks[blbb] = ib;
			blbb++;

		}
		if (leftxo < levdx)
		{
			//
			XForcing.left.blks[blbl] = ib;
			blbl++;

		}
	}




}


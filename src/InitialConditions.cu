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

	// Initialise the topography slope and halo
	InitzbgradientCPU(XParam, XModel);

	//=====================================
	// Initial forcing
	InitRivers(XParam, XForcing, XModel);

	//=====================================
	// Initial bndinfo
	Calcbndblks(XParam, XForcing, XModel.blocks);
	Findbndblks(XParam, XModel, XForcing);

	//=====================================
	// Calculate Active cells
	calcactiveCellCPU(XParam, XModel.blocks, XForcing, XModel.zb);


	//=====================================
	// Initialize output variables
	initoutput(XParam, XModel);

}
template void InitialConditions<float>(Param &XParam, Forcing<float> &XForcing, Model<float> &XModel);
template void InitialConditions<double>(Param &XParam, Forcing<float> &XForcing, Model<double> &XModel);

template <class T> void InitzbgradientCPU(Param XParam, Model<T> XModel)
{
	

	fillHaloC(XParam, XModel.blocks, XModel.zb);
	gradientC(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
	gradientHalo(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);

	refine_linear(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
	gradientHalo(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);

	
}
template void InitzbgradientCPU<double>(Param XParam, Model<double> XModel);
template void InitzbgradientCPU<float>(Param XParam, Model<float> XModel);

template <class T> void InitzbgradientGPU(Param XParam, Model<T> XModel)
{
	const int num_streams = 4;
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);


	cudaStream_t streams[num_streams];


	CUDA_CHECK(cudaStreamCreate(&streams[0]));

	fillHaloGPU(XParam, XModel.blocks, streams[0], XModel.zb);

	cudaStreamDestroy(streams[0]);

	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientHaloGPU(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);

	refine_linearGPU(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);

	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientHaloGPU(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
}
template void InitzbgradientGPU<double>(Param XParam, Model<double> XModel);
template void InitzbgradientGPU<float>(Param XParam, Model<float> XModel);

template <class T> void initoutput(Param &XParam, Model<T> &XModel)
{
	

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
	if (XParam.outtwet)
	{
		InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.wettime);
	}

	if (XParam.TSnodesout.size() > 0)
	{
		FindTSoutNodes(XParam, XModel.blocks, XModel.bndblk);
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
	bnd.nblkTs = int(XParam.TSnodesout.size());

	AllocateCPU(bnd.nblkTs, 1, bnd.Tsout);

	// Initialise all storage involving parameters
	
	
	for (int o = 0; o < XParam.TSnodesout.size(); o++)
	{


		//find the block where point belongs
		for (int blk = 0; blk < XParam.nblk; blk++)
		{

			ib = XBlock.active[blk];
			levdx = T(calcres(XParam.dx,XBlock.level[ib]));
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
		int ib;
		double levdx;
		double dischargeArea;
		log("\tInitializing rivers");
		//For each rivers
		for (int Rin = 0; Rin < XForcing.rivers.size(); Rin++)
		{
			dischargeArea = 0.0;
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
						//int n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
						
						
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

		for (auto it = XForcing.rivers.begin(); it != XForcing.rivers.end(); it++)
		{

			if (it->disarea == 0.0)
			{
				log("Warning river outside active model domain found. This river has been removed!\n");
				XForcing.rivers.erase(it--);
			}
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
			XModel.bndblk.nblkriver = int(activeRiverBlk.size());
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

	XModel.OutputVarMap["Umean"] = XModel.evmean.U;

	XModel.OutputVarMap["Umax"] = XModel.evmax.U;

	XModel.OutputVarMap["hUmean"] = XModel.evmean.hU;

	XModel.OutputVarMap["hUmax"] = XModel.evmax.hU;

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

	XModel.OutputVarMap["Patm"] = XModel.Patm;
	XModel.OutputVarMap["datmpdx"] = XModel.datmpdx;
	XModel.OutputVarMap["datmpdy"] = XModel.datmpdy;

	//XModel.OutputVarMap["U"] = XModel.U;

	XModel.OutputVarMap["twet"] = XModel.wettime;

	//XModel.OutputVarMap["vort"] = XModel.vort;
}

template void Initmaparray<float>(Model<float>& XModel);
template void Initmaparray<double>(Model<double>& XModel);


// Initialise all storage involving parameters of the outzone objects
template <class T> void Findoutzoneblks(Param& XParam, BlockP<T>& XBlock)
{
	int ib, i;
	T levdx;
	std::vector<int> cornerblk; //index of the blocks at the corner of the zone 
	outzoneP Xzone; //info on outzone given by the user
	outzoneB XzoneB; //info on outzone computed and used actually for writing nc files
	double eps;

	// Find the blocks to output and the corners of this area for each zone
	for (int o = 0; o < XParam.outzone.size(); o++)
	{

		Xzone = XParam.outzone[o];
		cornerblk = { 0, 0, 0, 0 };
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
		double xl, xr, yb, yt;

		int nblk = 0;

		//Getting the new area's corners

		//Initialisation of the corners blocks on the domain boundaries
		//in case of the border given by user being out of the domain
		RectCornerBlk(XParam, XBlock, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, true, cornerblk);

		//Getting the corners blocks of the rectangle given by the user
		RectCornerBlk(XParam, XBlock, XParam.outzone[o].xstart, XParam.outzone[o].ystart, XParam.outzone[o].xend, XParam.outzone[o].yend, false, cornerblk);


		//left edge border
		int il = (XBlock.level[cornerblk[0]] < XBlock.level[cornerblk[1]]) ? cornerblk[0] : cornerblk[1];
		levdx = calcres(XParam.dx, XBlock.level[il]);
		XzoneB.xo = XParam.xo + XBlock.xo[il] - levdx / 2;
		//bottom edge border
		int ib = (XBlock.level[cornerblk[0]] < XBlock.level[cornerblk[3]]) ? cornerblk[0] : cornerblk[3];
		levdx = calcres(XParam.dx, XBlock.level[ib]);
		XzoneB.yo = XParam.yo + XBlock.yo[ib] - levdx / 2;
		//right edge border
		int ir = (XBlock.level[cornerblk[2]] < XBlock.level[cornerblk[3]]) ? cornerblk[2] : cornerblk[3];
		levdx = calcres(XParam.dx, XBlock.level[ir]);
		XzoneB.xmax = XParam.xo + XBlock.xo[ir] + (XParam.blkwidth - 1) * levdx + levdx/2;
		//top edge border
		int it = (XBlock.level[cornerblk[1]] < XBlock.level[cornerblk[2]]) ? cornerblk[1] : cornerblk[2];
		levdx = calcres(XParam.dx, XBlock.level[it]);
		XzoneB.ymax = XParam.yo + XBlock.yo[it] + (XParam.blkwidth - 1) * levdx + levdx/2;


		if (XParam.maxlevel != XParam.minlevel) //if adapatation
		{

			//This minimal rectangular can include only part of blocks depending of resolution.
			//the blocks containing the corners are found and the larger block impose its border on each side
			
			//In order of avoiding rounding error, a slightly smaller rectangular is used
			RectCornerBlk(XParam, XBlock, XzoneB.xo, XzoneB.yo, XzoneB.xmax, XzoneB.ymax, true, cornerblk);
	

			// for each side, the border is imposed by the larger block (the "further out" one) if adaptative,
			// if the grid is.

			//left edge border
			int il = (XBlock.level[cornerblk[0]] < XBlock.level[cornerblk[1]]) ? cornerblk[0] : cornerblk[1];
			levdx = calcres(XParam.dx, XBlock.level[il]);
			XzoneB.xo = XParam.xo + XBlock.xo[il] - levdx/2;
			//bottom edge border
			int ib = (XBlock.level[cornerblk[0]] < XBlock.level[cornerblk[3]]) ? cornerblk[0] : cornerblk[3];
			levdx = calcres(XParam.dx, XBlock.level[ib]);
			XzoneB.yo = XParam.yo + XBlock.yo[ib] - levdx/2;
			//right edge border
			int ir = (XBlock.level[cornerblk[2]] < XBlock.level[cornerblk[3]]) ? cornerblk[2] : cornerblk[3];
			levdx = calcres(XParam.dx, XBlock.level[ir]);
			XzoneB.xmax = XParam.xo + XBlock.xo[ir] + (XParam.blkwidth - 1) * levdx + levdx/2;
			//top edge border
			int it = (XBlock.level[cornerblk[1]] < XBlock.level[cornerblk[2]]) ? cornerblk[1] : cornerblk[2];
			levdx = calcres(XParam.dx, XBlock.level[it]);
			XzoneB.ymax = XParam.yo + XBlock.yo[it] + (XParam.blkwidth - 1) * levdx + levdx/2;
		}

		// Get the list of all blocks in the zone and the maximum and minimum level of refinement
		int maxlevel = XParam.minlevel;
		int minlevel = XParam.maxlevel;

		for (i = 0; i < XParam.nblk; i++)
		{
			ib = XBlock.active[i];
			levdx = calcres(XParam.dx, XBlock.level[ib]);

			// get the corners' locations of the block (center of the corner cell)
			xl = XParam.xo + XBlock.xo[ib];
			yb = XParam.yo + XBlock.yo[ib];
			xr = XParam.xo + XBlock.xo[ib] + (XParam.blkwidth - 1) * levdx;
			yt = XParam.yo + XBlock.yo[ib] + (XParam.blkwidth - 1) * levdx;

			// Checking if at least one part of the a cell of the block is 
			// inside the area defined by the user.
			if (OBBdetect(xl, xr, yb, yt, XzoneB.xo, XzoneB.xmax, XzoneB.yo, XzoneB.ymax))
			{
				// This block belongs to the output zone defined by the user
				blkzone.push_back(ib);
				nblk++;

				//min/max levels
				if (XBlock.level[ib] > maxlevel) { maxlevel = XBlock.level[ib]; }
				if (XBlock.level[ib] < minlevel) { minlevel = XBlock.level[ib]; }
			}
			

		}
		XzoneB.nblk = nblk;
		XzoneB.maxlevel = maxlevel;
		XzoneB.minlevel = minlevel;
		

		AllocateCPU(blkzone.size(), 1, XzoneB.blk);
		for (int b = 0; b < blkzone.size(); b++)
		{
			XzoneB.blk[b] = blkzone[b];
		}
		XzoneB.outname = XParam.outzone[o].outname;
		
		//All the zone informatin has been integrated in a outzoneB structure,
		// and pushed back to the initial variable.
		// If this variable has already be constructed and adjusted here (after adaptation for example),
		// just modify the variable

		if (XBlock.outZone.size() < XParam.outzone.size())
		{
			XBlock.outZone.push_back(XzoneB);
		}
		else
		{
			XBlock.outZone[o] = XzoneB;
		}
	}

}
template void Findoutzoneblks<float>(Param& XParam, BlockP<float>& XBlock);
template void Findoutzoneblks<double>(Param& XParam, BlockP<double>& XBlock);

template <class T> void Initoutzone(Param& XParam, BlockP<T>& XBlock)
{
	//The domain full domain is defined as the output zone by default 
	//(and the blocks have been initialised by default)
	// If a zone for the output has been requested by the user, the blocks in the 
	// zone and the corners are computed here:

	if (XParam.outzone.size() > 0)
	{
		XBlock.outZone.reserve(XParam.outzone.size()); //to avoid a change of location of memory if not enought space
		Findoutzoneblks(XParam, XBlock);
	}
	else
	{
		outzoneB XzoneB;
		std::vector<int> blksall;
		//Define the full domain as a zone
		XzoneB.outname = XParam.outfile; //.assign(XParam.outfile);
		XzoneB.xo = XParam.xo;
		XzoneB.yo = XParam.yo;
		XzoneB.xmax = XParam.xmax;
		XzoneB.ymax = XParam.ymax;
		XzoneB.nblk = XParam.nblk;
		XzoneB.maxlevel = XParam.maxlevel;
		XzoneB.minlevel = XParam.minlevel;
		AllocateCPU(XParam.nblk, 1, XzoneB.blk);
		int I = 0;
		for (int ib = 0; ib < XParam.nblk; ib++)
		{
			XzoneB.blk[ib] = XBlock.active[ib];
		}

		if (XBlock.outZone.size() > 0) //If adaptative, the zone need to be written over
		{
			XBlock.outZone[0] = XzoneB;
		}
		else
		{
			XBlock.outZone.push_back(XzoneB);
		}

	}
}
template void Initoutzone<float>(Param& XParam, BlockP<float>& XBlock);
template void Initoutzone<double>(Param& XParam, BlockP<double>& XBlock);


template <class T> void Calcbndblks(Param& XParam, Forcing<float>& XForcing, BlockP<T> XBlock)
{
	//=====================================
	// Find how many blocks are on each bnds
	int blbr = 0, blbb = 0, blbl = 0, blbt = 0;
	T leftxo, rightxo, topyo, botyo;

	T initlevdx = calcres(XParam.dx, XParam.initlevel);

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//double espdist = 0.00000001;///WARMING

		int ib = XBlock.active[ibl];

		T levdx = calcres(XParam.dx, XBlock.level[ib]);

		leftxo = XBlock.xo[ib]; // in adaptive this shoulbe be a range 

		//leftyo =XBlock.yo[ib];
		rightxo = XBlock.xo[ib] + (XParam.blkwidth - 1) * levdx;
		//rightyo = XBlock.yo[ib];
		//topxo = XBlock.xo[ib];
		topyo = XBlock.yo[ib] + (XParam.blkwidth - 1) * levdx;
		//botxo = XBlock.xo[ib];
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
	T leftxo, rightxo, topyo, botyo;


	// Reallocate array if necessary
	ReallocArray(XParam.nbndblkleft, 1, XForcing.left.blks);
	ReallocArray(XParam.nbndblkright, 1, XForcing.right.blks);
	ReallocArray(XParam.nbndblktop, 1, XForcing.top.blks);
	ReallocArray(XParam.nbndblkbot, 1, XForcing.bot.blks);

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//double espdist = 0.00000001;///WARMING

		int ib = XBlock.active[ibl];
		T levdx = calcres(XParam.dx, XModel.blocks.level[ib]);


		leftxo = XBlock.xo[ib]; // in adaptive this shoulbe be a range 

		//leftyo = XBlock.yo[ib];
		rightxo = XBlock.xo[ib] + (XParam.blkwidth - 1) * levdx;
		//rightyo = XBlock.yo[ib];
		//topxo = XBlock.xo[ib];
		topyo = XBlock.yo[ib] + (XParam.blkwidth - 1) * levdx;
		//botxo = XBlock.xo[ib];
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

/*! \fn RectCornerBlk(Param& XParam, BlockP<T>& XBlock, double xo, double yo, double xmax, double ymax, bool isEps, std::vector<int>& cornerblk)
	* Find the block containing the border of a rectangular box (used for the defining the output zones)
	* The indice of the blocks are returned through "cornerblk" from bottom left turning in the clockwise direction
	*/
template <class T> void RectCornerBlk(Param& XParam, BlockP<T>& XBlock, double xo, double yo, double xmax, double ymax, bool isEps, std::vector<int>& cornerblk)
{

	int ib;
	T levdx;
	double xl, yb, xr, yt;
	double eps = 0.0;

	for (int i = 0; i < XParam.nblk; i++)
	{
		ib = XBlock.active[i];
		levdx = calcres(XParam.dx, XBlock.level[ib]);

		// margin to search for block boundaries, to avoid machine error if rectangle corner are supposed to
		// be on blocks edges

		if (isEps == true)
		{
			eps = levdx/3;
		}

		// get the corners' locations of the block (edge of the corner cell)
		xl = XParam.xo + XBlock.xo[ib] - levdx/2;
		yb = XParam.yo + XBlock.yo[ib] - levdx/2;
		xr = XParam.xo + XBlock.xo[ib] + (XParam.blkwidth - 1) * levdx + levdx/2;
		yt = XParam.yo + XBlock.yo[ib] + (XParam.blkwidth - 1) * levdx + levdx/2;

		// Getting the bottom left corner coordinate of the output area
		if (xo + eps >= xl && xo + eps <= xr && yo + eps >= yb && yo + eps <= yt)
		{
			cornerblk[0] = ib;
		}
		// Getting the top left corner coordinate of the output area
		if (xo + eps >= xl && xo + eps <= xr && ymax - eps >= yb && ymax - eps <= yt)
		{
			cornerblk[1] = ib;
		}
		// Getting the top right corner coordinate of the output area
		if (xmax - eps >= xl && xmax - eps <= xr && ymax - eps >= yb && ymax - eps <= yt)
		{
			cornerblk[2] = ib;
		}
		// Getting the bottom right corner coordinate of the output area
		if (xmax - eps >= xl && xmax - eps <= xr && yo + eps >= yb && yo + eps <= yt)
		{
			cornerblk[3] = ib;
		}

	}

}

template <class T> void calcactiveCellCPU(Param XParam, BlockP<T> XBlock, Forcing<float>& XForcing, T* zb)
{
	int ib,n,wn;

	// Remove rain from area above mask elevatio
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				double levdx = calcres(XParam.dx, XBlock.level[ib]);
				double x = XParam.xo + XBlock.xo[ib] + i * levdx;
				double y = XParam.yo + XBlock.yo[ib] + j * levdx;
				wn = 1;
				if (XForcing.AOI.active)
				{
					wn = wn_PnPoly(x, y, XForcing.AOI.poly);
				}
				n = memloc(XParam, i, j, ib);
				if (zb[n] < XParam.mask && wn != 0)
				{
					XBlock.activeCell[n] = 1;
				}
				else
				{
					XBlock.activeCell[n] = 0;
				}
			}
		}
	}

	//bool Modif = false;
	if (XParam.rainbnd== false) {
		// Remove rain from boundary cells
		for (int ibl = 0; ibl < XParam.nbndblkleft; ibl++)
		{
			ib = XForcing.left.blks[ibl];
			for (int j = 0; j < XParam.blkwidth; j++)
			{
				n = memloc(XParam, 0, j, ib);
				XBlock.activeCell[n] = 0;

				n = memloc(XParam, 1, j, ib);
				XBlock.activeCell[n] = 0;
			}
		}
		for (int ibl = 0; ibl < XParam.nbndblkright; ibl++)
		{
			ib = XForcing.right.blks[ibl];
			for (int j = 0; j < XParam.blkwidth; j++)
			{
				n = memloc(XParam, XParam.blkwidth - 1, j, ib);
				XBlock.activeCell[n] = 0;

				n = memloc(XParam, XParam.blkwidth - 2, j, ib);
				XBlock.activeCell[n] = 0;
			}
		}
		for (int ibl = 0; ibl < XParam.nbndblkbot; ibl++)
		{
			ib = XForcing.bot.blks[ibl];
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				n = memloc(XParam, i, 0, ib);
				XBlock.activeCell[n] = 0;

				n = memloc(XParam, i, 1, ib);
				XBlock.activeCell[n] = 0;
			}
		}
		for (int ibl = 0; ibl < XParam.nbndblktop; ibl++)
		{
			ib = XForcing.top.blks[ibl];
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				n = memloc(XParam, i, XParam.blkwidth - 1, ib);
				XBlock.activeCell[n] = 0;

				n = memloc(XParam, i, XParam.blkwidth - 2, ib);
				XBlock.activeCell[n] = 0;
			}
		}
	}
	
}


template <class T> __global__ void calcactiveCellGPU(Param XParam, BlockP<T> XBlock, T *zb)
{
	unsigned int blkmemwidth = blockDim.x + XParam.halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int n = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);

	if (zb[n] < XParam.mask)
	{
		XBlock.activeCell[n] = 1;
	}
	else
	{
		XBlock.activeCell[n] = 0;
	}
}
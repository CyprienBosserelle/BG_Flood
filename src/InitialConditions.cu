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

	if (!XForcing.cf.empty())
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
	//Calcbndblks(XParam, XForcing, XModel.blocks);
	//Findbndblks(XParam, XModel, XForcing);
	Initbndblks(XParam, XForcing, XModel.blocks);


	//=====================================
	// Calculate Active cells
	calcactiveCellCPU(XParam, XModel.blocks, XForcing, XModel.zb);

	//=====================================
	// Initialise the rain losses map

	if (XParam.infiltration)
	{
		if (!XForcing.il.inputfile.empty())
		{
			interp2BUQ(XParam, XModel.blocks, XForcing.il, XModel.il);
		}
		else
		{
			InitArrayBUQ(XParam, XModel.blocks, (T)XParam.il, XModel.il);
		}
		if (!XForcing.cl.inputfile.empty())
		{
			interp2BUQ(XParam, XModel.blocks, XForcing.cl, XModel.cl);
		}
		else
		{
			InitArrayBUQ(XParam, XModel.blocks, (T)XParam.cl, XModel.cl);
		}
		// Set edges of friction map
		setedges(XParam, XModel.blocks, XModel.il);
		setedges(XParam, XModel.blocks, XModel.cl);
		InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.hgw);

		// Initialise infiltration to IL where h is already wet
		initinfiltration(XParam, XModel.blocks, XModel.evolv.h, XModel.il, XModel.hgw);

	}


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

	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.delta, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientHaloGPU(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);

	refine_linearGPU(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);

	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.delta, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
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

template <class T> void FindTSoutNodes(Param& XParam, BlockP<T> XBlock, BndblockP<T> & bnd)
{
	int ib;
	T levdx,x,y,blkxmin,blkxmax,blkymin,blkymax,dxblk;
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

			x = (T)XParam.TSnodesout[o].x;
			y = (T)XParam.TSnodesout[o].y;

			dxblk = (T)(XParam.blkwidth) * levdx;

			blkxmin = ((T)XParam.xo + XBlock.xo[ib] - T(0.5) * levdx);
			blkymin = ((T)XParam.yo + XBlock.yo[ib] - T(0.5) * levdx);

			blkxmax = (blkxmin + dxblk);
			blkymax = (blkymin + dxblk);


			if (x > blkxmin && x <= blkxmax && y > blkymin && y <= blkymax)
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
template void FindTSoutNodes<float>(Param& XParam, BlockP<float> XBlock, BndblockP<float>& bnd);
template void FindTSoutNodes<double>(Param& XParam, BlockP<double> XBlock, BndblockP<double>& bnd);



template <class T> void InitRivers(Param XParam, Forcing<float> &XForcing, Model<T> &XModel)
{
	//========================
	// River discharge

	if (XForcing.rivers.size() > 0)
	{
		//
		double xl, yb, xr, yt, xi,yi ;
		int ib;
		double levdx, levdelta;
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
				levdelta = calcres(XParam.delta, XModel.blocks.level[ib]);
				for (int j = 0; j < XParam.blkwidth; j++)
				{
					for (int i = 0; i < XParam.blkwidth; i++)
					{
						//int n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
						xi = XParam.xo + XModel.blocks.xo[ib] + i * levdx;
						yi = XParam.yo + XModel.blocks.yo[ib] + j * levdx;


						
						xl = xi - 0.5 * levdx;
						yb = yi - 0.5 * levdx;

						xr = xi + 0.5 * levdx;
						yt = yi + 0.5 * levdx;
						// the conditions are that the discharge area as defined by the user have to include at least a model grid node
						// This could be really annoying and there should be a better way to deal wiith this like polygon intersection
						//if (xx >= XForcing.rivers[Rin].xstart && xx <= XForcing.rivers[Rin].xend && yy >= XForcing.rivers[Rin].ystart && yy <= XForcing.rivers[Rin].yend)
						if (OBBdetect(xl, xr, yb, yt, XForcing.rivers[Rin].xstart, XForcing.rivers[Rin].xend, XForcing.rivers[Rin].ystart, XForcing.rivers[Rin].yend))
						{

							// This cell belongs to the river discharge area
							idis.push_back(i);
							jdis.push_back(j);
							blockdis.push_back(ib);
							if (XParam.spherical)
							{
								dischargeArea = dischargeArea + spharea(XParam.Radius, xi, yi, levdx);
							}
							else
							{
								dischargeArea = dischargeArea + levdelta * levdelta;
							}
						}
					}
				}

			}


			
				XForcing.rivers[Rin].i = idis;
				XForcing.rivers[Rin].j = jdis;
				XForcing.rivers[Rin].block = blockdis;
				XForcing.rivers[Rin].disarea = dischargeArea; // That is valid for spherical grids
			

			
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

		// Setup the river info

		int nburmax = activeRiverBlk.size();
		int nribmax = 0;
		for (int b = 0; b < activeRiverBlk.size(); b++)
		{
			int bur = activeRiverBlk[b];
			int nriverinblock = 0;
			for (int Rin = 0; Rin < XForcing.rivers.size(); Rin++)
			{
				std::vector<int> uniqblockforriver = XForcing.rivers[Rin].block;

				std::sort(uniqblockforriver.begin(), uniqblockforriver.end());
				uniqblockforriver.erase(std::unique(uniqblockforriver.begin(), uniqblockforriver.end()), uniqblockforriver.end());

				for (int bir = 0; bir < uniqblockforriver.size(); bir++)
				{
					if (uniqblockforriver[bir] == bur)
					{
						nriverinblock = nriverinblock + 1;
					}
				}

			}
			nribmax = max(nribmax, nriverinblock);
		}

		// Allocate Qnow as pinned memory
		AllocateMappedMemCPU(XForcing.rivers.size(), 1, XParam.GPUDEVICE,XModel.bndblk.Riverinfo.qnow);
		AllocateCPU(nribmax, nburmax, XModel.bndblk.Riverinfo.xstart, XModel.bndblk.Riverinfo.xend, XModel.bndblk.Riverinfo.ystart, XModel.bndblk.Riverinfo.yend);
		FillCPU(nribmax, nburmax, T(-1.0), XModel.bndblk.Riverinfo.xstart);
		FillCPU(nribmax, nburmax, T(-1.0), XModel.bndblk.Riverinfo.xend);
		FillCPU(nribmax, nburmax, T(-1.0), XModel.bndblk.Riverinfo.ystart);
		FillCPU(nribmax, nburmax, T(-1.0), XModel.bndblk.Riverinfo.yend);



		// Allocate XXbidir and Xridib
		ReallocArray(nribmax, nburmax, XModel.bndblk.Riverinfo.Xbidir);
		ReallocArray(nribmax, nburmax, XModel.bndblk.Riverinfo.Xridib);

		// Fill them with a flag value 
		FillCPU(nribmax, nburmax, -1, XModel.bndblk.Riverinfo.Xbidir);
		FillCPU(nribmax, nburmax, -1, XModel.bndblk.Riverinfo.Xridib);

		//Xbidir is an array that stores block id where n rivers apply
		//along the row of Xbidir block id is unique. meaning that a block id ith two river injection will appear on two seperate row of Xbidir
		//The number of column (size of row 1) in xbidir is nburmax = length(uniq(blockwith river injected))
		//

		//Xridib is an array that stores River id that a river is injected for the corresponding block id in Xbidir


		XModel.bndblk.Riverinfo.nribmax = nribmax;
		XModel.bndblk.Riverinfo.nburmax = nburmax;

		std::vector<RiverBlk> blocksalreadyin;
		RiverBlk emptyvec;
		for (int iblk = 0; iblk < nribmax; iblk++)
		{

			blocksalreadyin.push_back(emptyvec);
			
		}
		
		//(n, 10)
		// 
		std::vector<int> iriv(nribmax,0);
		for (int Rin = 0; Rin < XForcing.rivers.size(); Rin++)
		{
			std::vector<int> uniqblockforriver = XForcing.rivers[Rin].block;

			std::sort(uniqblockforriver.begin(), uniqblockforriver.end());
			uniqblockforriver.erase(std::unique(uniqblockforriver.begin(), uniqblockforriver.end()), uniqblockforriver.end());

			

			for (int bir = 0; bir < uniqblockforriver.size(); bir++)
			{

				for (int iribm = 0; iribm < nribmax; iribm++)
				{

					if (std::find(blocksalreadyin[iribm].block.begin(), blocksalreadyin[iribm].block.end(), uniqblockforriver[bir]) != blocksalreadyin[iribm].block.end())
					{
						//block found already listed in that line;

						continue;
					}
					else
					{
						//not found;
						// write to the array
						XModel.bndblk.Riverinfo.Xbidir[iriv[iribm] + iribm * nburmax] = uniqblockforriver[bir];
						XModel.bndblk.Riverinfo.Xridib[iriv[iribm] + iribm * nburmax] = Rin;

						iriv[iribm] = iriv[iribm] + 1;

						// add it to the list 
						blocksalreadyin[iribm].block.push_back(uniqblockforriver[bir]);

						

						break;
					}
				}
					
			}

		}
		for (int iribm = 0; iribm < nribmax; iribm++)
		{
			for (int ibur = 0; ibur < nburmax; ibur++)
			{
				int indx = ibur + iribm * nburmax;
				int Rin = XModel.bndblk.Riverinfo.Xridib[indx];
				if (Rin > -1)
				{
					XModel.bndblk.Riverinfo.xstart[indx] = XForcing.rivers[Rin].xstart;
					XModel.bndblk.Riverinfo.xend[indx] = XForcing.rivers[Rin].xend;
					XModel.bndblk.Riverinfo.ystart[indx] = XForcing.rivers[Rin].ystart;
					XModel.bndblk.Riverinfo.yend[indx] = XForcing.rivers[Rin].yend;
				}
			}
		}


		
		
	}


}

template void InitRivers<float>(Param XParam, Forcing<float> &XForcing, Model<float> &XModel);
template void InitRivers<double>(Param XParam, Forcing<float> &XForcing, Model<double> &XModel);


template<class T> void Initmaparray(Model<T>& XModel)
{
	//Main Parameters
	XModel.OutputVarMap["zb"] = XModel.zb;
	XModel.Outvarlongname["zb"] = "Ground elevation above datum";
	XModel.Outvarstdname["zb"] = "ground_elevation_above_datum";
	XModel.Outvarunits["zb"] = "m";


	XModel.OutputVarMap["u"] = XModel.evolv.u;
	XModel.Outvarlongname["u"] = "Water velocity in x-direction";// zonal
	XModel.Outvarstdname["u"] = "u_velocity";
	XModel.Outvarunits["u"] = "m s-1";

	XModel.OutputVarMap["v"] = XModel.evolv.v;
	XModel.Outvarlongname["v"] = "Velocity in y-direction";// meridional
	XModel.Outvarstdname["v"] = "v_velocity";
	XModel.Outvarunits["v"] = "m s-1";

	XModel.OutputVarMap["zs"] = XModel.evolv.zs;
	XModel.Outvarlongname["zs"] = "Water surface elevation above datum";
	XModel.Outvarstdname["zs"] = "water_surface_elevation";
	XModel.Outvarunits["zs"] = "m";

	XModel.OutputVarMap["h"] = XModel.evolv.h;
	XModel.Outvarlongname["h"] = "Water depth";
	XModel.Outvarstdname["h"] = "water_depth";
	XModel.Outvarunits["h"] = "m";

	//Mean Max parameters
	XModel.OutputVarMap["hmean"] = XModel.evmean.h;
	XModel.Outvarlongname["hmean"] = "Mean water depth since last output";
	XModel.Outvarstdname["hmean"] = "mean_water_depth";
	XModel.Outvarunits["hmean"] = "m";

	XModel.OutputVarMap["hmax"] = XModel.evmax.h;
	XModel.Outvarlongname["hmax"] = "Maximum water depth since simulation start";
	XModel.Outvarstdname["hmax"] = "maximum_water_depth";
	XModel.Outvarunits["hmax"] = "m";

	XModel.OutputVarMap["zsmean"] = XModel.evmean.zs;
	XModel.Outvarlongname["zsmean"] = "Mean water elevation above datum since last output";
	XModel.Outvarstdname["zsmean"] = "mean_water_elevation";
	XModel.Outvarunits["zsmean"] = "m";

	XModel.OutputVarMap["zsmax"] = XModel.evmax.zs;
	XModel.Outvarlongname["zsmax"] = "Maximum water elevation above datum since simulation start";
	XModel.Outvarstdname["zsmax"] = "maximum_water_elevation";
	XModel.Outvarunits["zsmax"] = "m";

	XModel.OutputVarMap["umean"] = XModel.evmean.u;
	XModel.Outvarlongname["umean"] = "Mean velocity in x-direction since last output";
	XModel.Outvarstdname["umean"] = "mean_u_velocity";
	XModel.Outvarunits["umean"] = "m s-1";

	XModel.OutputVarMap["umax"] = XModel.evmax.u;
	XModel.Outvarlongname["umax"] = "Maximum velocity in x-direction since simulation start";
	XModel.Outvarstdname["umax"] = "maximum_u_velocity";
	XModel.Outvarunits["umax"] = "m s-1";

	XModel.OutputVarMap["vmean"] = XModel.evmean.v;
	XModel.Outvarlongname["vmean"] = "Mean velocity in y-direction since last output";
	XModel.Outvarstdname["vmean"] = "mean_v_velocity";
	XModel.Outvarunits["vmean"] = "m s-1";

	XModel.OutputVarMap["vmax"] = XModel.evmax.v;
	XModel.Outvarlongname["vmax"] = "Maximum velocity in y-direction since simulation start";
	XModel.Outvarstdname["vmax"] = "maximum_v_velocity";
	XModel.Outvarunits["vmax"] = "m s-1";

	XModel.OutputVarMap["Umean"] = XModel.evmean.U;
	XModel.Outvarlongname["Umean"] = "Mean velocity magnitude since last output";
	XModel.Outvarstdname["Umean"] = "mean_velocity";
	XModel.Outvarunits["Umean"] = "m s-1";

	XModel.OutputVarMap["Umax"] = XModel.evmax.U;
	XModel.Outvarlongname["Umax"] = "Maximum velocity magnitude since simulation start";
	XModel.Outvarstdname["Umax"] = "maximum_velocity";
	XModel.Outvarunits["Umax"] = "m s-1";

	XModel.OutputVarMap["hUmean"] = XModel.evmean.hU;
	XModel.Outvarlongname["hUmean"] = "Mean depth times velocity since last output";
	XModel.Outvarstdname["hUmean"] = "mean_depth_velocity";
	XModel.Outvarunits["hUmean"] = "m2 s-1";

	XModel.OutputVarMap["hUmax"] = XModel.evmax.hU;
	XModel.Outvarlongname["hUmax"] = "Maximum depth times velocity since simulation start";
	XModel.Outvarstdname["hUmax"] = "maximum_depth_velocity";
	XModel.Outvarunits["hUmax"] = "m2 s-1";

	//others

	XModel.OutputVarMap["uo"] = XModel.evolv_o.u;
	XModel.Outvarlongname["uo"] = "Velocity in x-direction from previous half-step";
	XModel.Outvarstdname["uo"] = "previous_u_velocity";
	XModel.Outvarunits["uo"] = "m s-1";

	XModel.OutputVarMap["vo"] = XModel.evolv_o.v;
	XModel.Outvarlongname["vo"] = "Velocity in y-direction from previous half-step";
	XModel.Outvarstdname["vo"] = "previous_v_velocity";
	XModel.Outvarunits["vo"] = "m s-1";

	XModel.OutputVarMap["zso"] = XModel.evolv_o.zs;
	XModel.Outvarlongname["zso"] = "Water elevation above datum from previous half-step";
	XModel.Outvarstdname["zso"] = "previous_water_elevation";
	XModel.Outvarunits["zso"] = "m";

	XModel.OutputVarMap["ho"] = XModel.evolv_o.h;
	XModel.Outvarlongname["ho"] = "Water depth from previous half-step";
	XModel.Outvarstdname["ho"] = "previous_water_depth";
	XModel.Outvarunits["ho"] = "m";

	// Gradients

	XModel.OutputVarMap["dhdx"] = XModel.grad.dhdx;
	XModel.Outvarlongname["dhdx"] = "Water depth gradient in x-direction";
	XModel.Outvarstdname["dhdx"] = "water_depth_gradient_x_direction";
	XModel.Outvarunits["dhdx"] = "m/m";

	XModel.OutputVarMap["dhdy"] = XModel.grad.dhdy;
	XModel.Outvarlongname["dhdy"] = "Water depth gradient in y-direction";
	XModel.Outvarstdname["dhdy"] = "water_depth_gradient_y_direction";
	XModel.Outvarunits["dhdy"] = "m/m";

	XModel.OutputVarMap["dudx"] = XModel.grad.dudx;
	XModel.Outvarlongname["dudx"] = "u-velocity gradient in x-direction";
	XModel.Outvarstdname["dudx"] = "u_velocity_gradient_x_direction";
	XModel.Outvarunits["dudx"] = "m s-1/m";

	XModel.OutputVarMap["dudy"] = XModel.grad.dudy;
	XModel.Outvarlongname["dudy"] = "u-velocity gradient in y-direction";
	XModel.Outvarstdname["dudy"] = "u_velocity_gradient_y_direction";
	XModel.Outvarunits["dudy"] = "m s-1/m";

	XModel.OutputVarMap["dvdx"] = XModel.grad.dvdx;
	XModel.Outvarlongname["dvdx"] = "v-velocity gradient in x-direction";
	XModel.Outvarstdname["dvdx"] = "v_velocity_gradient_x_direction";
	XModel.Outvarunits["dvdx"] = "m s-1/m";

	XModel.OutputVarMap["dvdy"] = XModel.grad.dvdy;
	XModel.Outvarlongname["dvdy"] = "v-velocity gradient in y-direction";
	XModel.Outvarstdname["dvdy"] = "v_velocity_gradient_y_direction";
	XModel.Outvarunits["dvdy"] = "m s-1/m";

	XModel.OutputVarMap["dzsdx"] = XModel.grad.dzsdx;
	XModel.Outvarlongname["dzsdx"] = "Water surface gradient in x-direction";
	XModel.Outvarstdname["dzsdx"] = "water_surface_gradient_x_direction";
	XModel.Outvarunits["dzsdx"] = "m/m";

	XModel.OutputVarMap["dzsdy"] = XModel.grad.dzsdy;
	XModel.Outvarlongname["dzsdy"] = "Water surface gradient in y-direction";
	XModel.Outvarstdname["dzsdy"] = "water_surface_gradient_y_direction";
	XModel.Outvarunits["dzsdy"] = "m/m";

	XModel.OutputVarMap["dzbdx"] = XModel.grad.dzbdx;
	XModel.Outvarlongname["dzbdx"] = "ground elevation gradient in x-direction";
	XModel.Outvarstdname["dzbdx"] = "ground_surface_gradient_x_direction";
	XModel.Outvarunits["dzbdx"] = "m/m";

	XModel.OutputVarMap["dzbdy"] = XModel.grad.dzbdy;
	XModel.Outvarlongname["dzbdy"] = "ground slope in y-direction";
	XModel.Outvarstdname["dzbdy"] = "ground_surface_gradient_y_direction";
	XModel.Outvarunits["dzbdy"] = "m/m";

	//Flux
	XModel.OutputVarMap["Fhu"] = XModel.flux.Fhu;
	XModel.Outvarlongname["Fhu"] = "Fhu flux term in x-direction";
	XModel.Outvarstdname["Fhu"] = "Fh_x_direction";
	XModel.Outvarunits["Fhu"] = "m2 s-1";

	XModel.OutputVarMap["Fhv"] = XModel.flux.Fhv;
	XModel.Outvarlongname["Fhv"] = "Fhv flux term in y-direction";
	XModel.Outvarstdname["Fhv"] = "Fh_y_direction";
	XModel.Outvarunits["Fhv"] = "m2 s-1";

	XModel.OutputVarMap["Fqux"] = XModel.flux.Fqux;
	XModel.Outvarlongname["Fqux"] = "Fqux flux term in x-direction";
	XModel.Outvarstdname["Fqux"] = "Fqu_x_direction";
	XModel.Outvarunits["Fqux"] = "m2 s-1";

	XModel.OutputVarMap["Fqvy"] = XModel.flux.Fqvy;
	XModel.Outvarlongname["Fqvy"] = "Fqvy flux term in y-direction";
	XModel.Outvarstdname["Fqvy"] = "Fqv_y_direction";
	XModel.Outvarunits["Fqvy"] = "m2 s-1";

	XModel.OutputVarMap["Fquy"] = XModel.flux.Fquy;
	XModel.Outvarlongname["Fquy"] = "Fquy flux term in y-direction";
	XModel.Outvarstdname["Fquy"] = "Fqu_y_direction";
	XModel.Outvarunits["Fquy"] = "m2 s-1";

	XModel.OutputVarMap["Fqvx"] = XModel.flux.Fqvx;
	XModel.Outvarlongname["Fqvx"] = "Fqvx flux term in x-direction";
	XModel.Outvarstdname["Fqvx"] = "Fqv_x_direction";
	XModel.Outvarunits["Fqvx"] = "m2 s-1";

	XModel.OutputVarMap["Su"] = XModel.flux.Su;
	XModel.Outvarlongname["Su"] = "Topography source term un x-direction";
	XModel.Outvarstdname["Su"] = "Topo_source_x_direction";
	XModel.Outvarunits["Su"] = "m2 s-1";

	XModel.OutputVarMap["Sv"] = XModel.flux.Sv;
	XModel.Outvarlongname["Sv"] = "Topography source term un y-direction";
	XModel.Outvarstdname["Sv"] = "Topo_source_y_direction";
	XModel.Outvarunits["Sv"] = "m2 s-1";

	//Advance
	XModel.OutputVarMap["dh"] = XModel.adv.dh;
	XModel.Outvarlongname["dh"] = "rate of change in water depth";
	XModel.Outvarstdname["dh"] = "rate_change_water_depth";
	XModel.Outvarunits["dh"] = "m s-1";

	XModel.OutputVarMap["dhu"] = XModel.adv.dhu;
	XModel.Outvarlongname["dhu"] = "changes in flux n x-direction";
	XModel.Outvarstdname["dhu"] = "rate_change_flux_x_direction";
	XModel.Outvarunits["dhu"] = "m3 s-1/s";

	XModel.OutputVarMap["dhv"] = XModel.adv.dhv;
	XModel.Outvarlongname["dhv"] = "changes in flux n y-direction";
	XModel.Outvarstdname["dhv"] = "rate_change_flux_y_direction";
	XModel.Outvarunits["dhv"] = "m3 s-1/s";

	XModel.OutputVarMap["cf"] = XModel.cf;
	XModel.Outvarlongname["cf"] = "Roughness";
	XModel.Outvarunits["cf"] = "m";

	XModel.OutputVarMap["il"] = XModel.il;
	XModel.Outvarlongname["il"] = "Initial loss water from inflitration";
	XModel.Outvarunits["il"] = "mm";

	XModel.OutputVarMap["cl"] = XModel.cl;
	XModel.Outvarlongname["cl"] = "Continung loss water from inflitration";
	XModel.Outvarunits["cl"] = "mm h-1";

	XModel.OutputVarMap["hgw"] = XModel.hgw;
	XModel.Outvarlongname["hgw"] = "Groundwater height";
	XModel.Outvarunits["hgw"] = "m";

	XModel.OutputVarMap["Patm"] = XModel.Patm;
	XModel.Outvarlongname["Patm"] = "Atmospheric pressure";
	XModel.Outvarunits["Patm"] = "m";

	XModel.OutputVarMap["datmpdx"] = XModel.datmpdx;
	XModel.Outvarlongname["datmpdx"] = "Atmospheric pressure gradient in x-direction";
	XModel.Outvarunits["datmpdx"] = "m/m";

	XModel.OutputVarMap["datmpdy"] = XModel.datmpdy;
	XModel.Outvarlongname["datmpdy"] = "Atmospheric pressure gradient in y-direction";
	XModel.Outvarunits["datmpdy"] = "m/m";

	//XModel.OutputVarMap["U"] = XModel.U;

	XModel.OutputVarMap["twet"] = XModel.wettime;
	XModel.Outvarlongname["twet"] = "time since the cell has been wet";
	XModel.Outvarunits["twet"] = "s";
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
	//double eps;

	// Find the blocks to output and the corners of this area for each zone
	for (int o = 0; o < XParam.outzone.size(); o++)
	{

		Xzone = XParam.outzone[o];

		XzoneB.xo = Xzone.xstart;
		XzoneB.yo = Xzone.ystart;
		XzoneB.xmax = Xzone.xend;
		XzoneB.ymax = Xzone.yend;

		std::vector<int> blkzone;
		double xl, xr, yb, yt;

		int nblk = 0;
		/*
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
		*/
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
			if (OBBdetect(xl, xr, yb, yt, Xzone.xstart, Xzone.xend, Xzone.ystart, Xzone.yend))
			{
				// This block belongs to the output zone defined by the user
				blkzone.push_back(ib);
				nblk++;

				XzoneB.xo = min(XzoneB.xo,XParam.xo + XBlock.xo[ib] - levdx / 2);
				XzoneB.xmax = max(XzoneB.xmax, XParam.xo + XBlock.xo[ib] + (XParam.blkwidth - 1) * levdx + levdx / 2);
				XzoneB.yo = min(XzoneB.yo, XParam.yo + XBlock.yo[ib] - levdx / 2);
				XzoneB.ymax = max(XzoneB.ymax, XParam.yo + XBlock.yo[ib] + (XParam.blkwidth - 1) * levdx + levdx / 2);

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

/*
*  Initialise bnd blk assign block to their relevant segment allocate memory...
* 1. Find all the boundary blocks(block with themselves as neighbours)
*
* 2. make an array to store which segemnt they belong to
*
* If any bnd segment was specified
* 3. scan each block and find which (if any) segment they belong to
*	 For each segment
*		 Calculate bbox
*		 if inbbox calc inpoly
*		if inpoly overwrite assingned segment with new one
*
*
* 4. Calculate nblk per segment & allocate (do for each segment)
*
* 5. fill segmnent and side arrays for each segments
*/
template <class T> void Initbndblks(Param& XParam, Forcing<float>& XForcing, BlockP<T> XBlock)
{
	//if(XForcing.bndseg.size()>0)

	std::vector<int> bndblks;
	std::vector<int> bndsegment;
	// 1. Find all the boundary blocks (block with themselves as neighbours)
	
	
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];

		bool testbot = (XBlock.BotLeft[ib] == ib) || (XBlock.BotRight[ib] == ib) || (XBlock.TopLeft[ib] == ib) || (XBlock.TopRight[ib] == ib) || (XBlock.LeftTop[ib] == ib) || (XBlock.LeftBot[ib] == ib) || (XBlock.RightTop[ib] == ib) || (XBlock.RightBot[ib] == ib);
		if (testbot)
		{
			T dxlev = calcres(XParam.dx, XBlock.level[ib]);

			bndblks.push_back(ib);
			bndsegment.push_back(XForcing.bndseg.size()-1); // i.e. by default the block doesn't belong to a segment so it belongs to collector (last) segemnt
			//loop through all but the last bnd seg which is meant for block that are not in any segments
			for (int s = 0; s < XForcing.bndseg.size()-1; s++)
			{
				bool inpoly=blockinpoly(T(XParam.xo + XBlock.xo[ib]), T(XParam.yo + XBlock.yo[ib]), dxlev, XParam.blkwidth, XForcing.bndseg[s].poly);

				if (inpoly)
				{
					bndsegment.back() = s;
				}

			}



		}
		

	}

	
	for (int s = 0; s < XForcing.bndseg.size(); s++)
	{
		int segcount = 0;
		int leftcount = 0;
		int rightcount = 0;
		int topcount = 0;
		int botcount = 0;
		
		for (int ibl = 0; ibl < bndblks.size(); ibl++)
		{
			int ib = bndblks[ibl];
			if (bndsegment[ibl] == s)
			{
				segcount++;

				if ((XBlock.BotLeft[ib] == ib) || (XBlock.BotRight[ib] == ib))
				{
					botcount++;
				}
				if ((XBlock.TopLeft[ib] == ib) || (XBlock.TopRight[ib] == ib))
				{
					topcount++;
				}
				if ((XBlock.LeftBot[ib] == ib) || (XBlock.LeftTop[ib] == ib))
				{
					leftcount++;
				}
				if ((XBlock.RightBot[ib] == ib) || (XBlock.RightTop[ib] == ib))
				{
					rightcount++;
				}
			}
		}
		XForcing.bndseg[s].nblk = segcount;

		XForcing.bndseg[s].left.nblk = leftcount;
		XForcing.bndseg[s].right.nblk = rightcount;
		XForcing.bndseg[s].top.nblk = topcount;
		XForcing.bndseg[s].bot.nblk = botcount;

		//allocate array
		//ReallocArray(int nblk, int blksize, T * &zb)
		ReallocArray(leftcount, 1, XForcing.bndseg[s].left.blk);
		ReallocArray(rightcount, 1, XForcing.bndseg[s].right.blk);
		ReallocArray(topcount, 1, XForcing.bndseg[s].top.blk);
		ReallocArray(botcount, 1, XForcing.bndseg[s].bot.blk);

		ReallocArray(leftcount, XParam.blkwidth, XForcing.bndseg[s].left.qmean);
		ReallocArray(rightcount, XParam.blkwidth, XForcing.bndseg[s].right.qmean);
		ReallocArray(topcount, XParam.blkwidth, XForcing.bndseg[s].top.qmean);
		ReallocArray(botcount, XParam.blkwidth, XForcing.bndseg[s].bot.qmean);

		FillCPU(leftcount, XParam.blkwidth, 0.0f, XForcing.bndseg[s].left.qmean);
		FillCPU(rightcount, XParam.blkwidth, 0.0f, XForcing.bndseg[s].right.qmean);
		FillCPU(topcount, XParam.blkwidth, 0.0f, XForcing.bndseg[s].top.qmean);
		FillCPU(botcount, XParam.blkwidth, 0.0f, XForcing.bndseg[s].bot.qmean);

		leftcount = 0;
		rightcount = 0;
		topcount = 0;
		botcount = 0;

		for (int ibl = 0; ibl < bndblks.size(); ibl++)
		{
			int ib = bndblks[ibl];

			if (bndsegment[ibl] == s)
			{
				if ((XBlock.BotLeft[ib] == ib) || (XBlock.BotRight[ib] == ib))
				{
					XForcing.bndseg[s].bot.blk[botcount] = ib;
					botcount++;
				}
				if ((XBlock.TopLeft[ib] == ib) || (XBlock.TopRight[ib] == ib))
				{
					XForcing.bndseg[s].top.blk[topcount] = ib;
					topcount++;
				}
				if ((XBlock.LeftBot[ib] == ib) || (XBlock.LeftTop[ib] == ib))
				{
					XForcing.bndseg[s].left.blk[leftcount] = ib;
					leftcount++;
				}
				if ((XBlock.RightBot[ib] == ib) || (XBlock.RightTop[ib] == ib))
				{
					XForcing.bndseg[s].right.blk[rightcount] = ib;
					rightcount++;
				}

			}

		}



	}


}

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


/*! \fn Findbndblks(Param XParam, Model<T> XModel,Forcing<float> &XForcing)
* Find which block on the model edge belongs to a "side boundary"
* 
*/
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

template <class T>
void initinfiltration(Param XParam, BlockP<T> XBlock, T* h, T* initLoss ,T* hgw)
{
//Initialisation to 0 (cold or hot start)

	
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				int n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;

				if (h[n] > XParam.eps)
				{
					initLoss[n]=T(0.0);
				}
				
			}
		}
	}
}

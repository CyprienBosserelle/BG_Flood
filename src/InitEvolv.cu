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


#include "InitEvolv.h"

#include "InitEvolv.h"

/**
 * @brief Initialize evolving variables for the simulation.
 *
 * Handles hotstart, coldstart, and warmstart initialization of water level, velocity, and bathymetry arrays.
 * Applies offsets and boundary conditions as needed.
 *
 * @tparam T Data type (float or double)
 * @param XParam Simulation parameters
 * @param XBlock Block parameters
 * @param XForcing Forcing/boundary conditions
 * @param XEv Evolving variables (output)
 * @param zb Bathymetry array (input/output)
 */
template <class T> void initevolv(Param XParam, BlockP<T> XBlock,Forcing<float> XForcing, EvolvingP<T> &XEv,T* &zb)
{
	//move this to a subroutine
	int hotstartsucess = 0;
	if (!XParam.hotstartfile.empty())
	{
		// hotstart
		log("\tHotstart file used : " + XParam.hotstartfile);

		hotstartsucess = readhotstartfile(XParam, XBlock, XEv, zb);

		//add offset if present
		if (T(XParam.zsoffset) != T(0.0)) // apply specified zsoffset
		{
			printf("\t\tadd offset to zs and hh... ");
			//
			AddZSoffset(XParam, XBlock, XEv, zb);

		}


		if (hotstartsucess == 0)
		{
			printf("\t\tFailed...  ");
			write_text_to_log_file("\tHotstart failed switching to cold start");
		}
	}


	
	if (XParam.hotstartfile.empty() || hotstartsucess == 0)
	{
		//printf("Cold start  ");
		//log("Cold start");
		//Cold start
		// 2 options:
		//		(1) if zsinit is set, then apply zsinit everywhere
		//		(2) zsinit is not set so interpolate from boundaries. (if no boundaries were specified set zsinit to zeros and apply case (1))

		//Param defaultParam;
		//!leftWLbnd.empty()

		//case 0 (i.e. zsinint not specified by user and no boundaries were specified)
		bool bndison = false;
		for (int iseg = 0; iseg < XForcing.bndseg.size(); iseg++)
		{
			if (XForcing.bndseg[iseg].on)
			{
				bndison = true;
			}
		}


		
		if (std::isnan(XParam.zsinit) && (!bndison)) //zsinit is default
		{
			XParam.zsinit = 0.0; // better default value than nan
		}
		
		//case 1 cold start
		
		if (!std::isnan(XParam.zsinit)) // apply specified zsinit
		{
			log("\tCold start");
			int coldstartsucess = 0;
			coldstartsucess = coldstart(XParam, XBlock, zb, XEv);
			
		}
		// case 2 warm start
		else // lukewarm start i.e. inv. dist interpolation of zs at bnds // Argggh!
		{
			log("\tWarm start");
			warmstart(XParam, XForcing, XBlock, zb, XEv);
			
		}// end else
		
	}
}
template void initevolv<float>(Param XParam, BlockP<float> XBlock, Forcing<float> XForcing, EvolvingP<float> &XEv, float* &zb);
template void initevolv<double>(Param XParam, BlockP< double > XBlock, Forcing<float> XForcing, EvolvingP< double > &XEv, double* &zb);


/**
 * @brief Cold start initialization of evolving variables.
 *
 * Sets initial water level, velocity, and bathymetry arrays for all blocks using specified zsinit and zsoffset.
 *
 * @tparam T Data type
 * @param XParam Simulation parameters
 * @param XBlock Block parameters
 * @param zb Bathymetry array
 * @param XEv Evolving variables (output)
 * @return Success flag (1 if successful)
 */
template <class T>
int coldstart(Param XParam, BlockP<T> XBlock, T* zb, EvolvingP<T> & XEv)
{
	T zzini = std::isnan(XParam.zsinit)? T(0.0): T(XParam.zsinit);
	T zzoffset = T(XParam.zsoffset);
	

	
	int coldstartsucess = 0;
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				int n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				
				XEv.u[n] = T(0.0);
				XEv.v[n] = T(0.0);
				//zb[n] = 0.0f;
				XEv.zs[n] = utils::max(zzini + zzoffset, zb[n]);
				
				//if (i >= 64 && i < 82)
				//{
				//	zs[n] = max(zsbnd+0.2f, zb[i + j*nx]);
				//}
				XEv.h[n] = utils::max(XEv.zs[n] - zb[n], T(0.0));//0.0 or XParam.eps ??
			}
		}
	}
	
	coldstartsucess = 1;
	return coldstartsucess;
}

/**
 * @brief Warm start initialization using boundary conditions and interpolation.
 *
 * Sets initial water level, velocity, and bathymetry arrays for all blocks using boundary segments and atmospheric pressure forcing.
 *
 * @tparam T Data type
 * @param XParam Simulation parameters
 * @param XForcing Forcing/boundary conditions
 * @param XBlock Block parameters
 * @param zb Bathymetry array
 * @param XEv Evolving variables (output)
 */
template <class T>
void warmstart(Param XParam, Forcing<float> XForcing, BlockP<T> XBlock, T* zb, EvolvingP<T>& XEv)
{
	int nuni=0;
	int ndyn=0;

	T zsbnduni=T(0.0);
	T zsbnd;
	for (int iseg = 0; iseg < XForcing.bndseg.size(); iseg++)
	{
		if (XForcing.bndseg[iseg].on)
		{
			if (XForcing.bndseg[iseg].uniform)
			{
				nuni++;

				int SLstepinbnd = 1;

				double difft = XForcing.bndseg[iseg].data[SLstepinbnd].time - XParam.totaltime;
				while (difft < 0.0)
				{
					SLstepinbnd++;
					difft = XForcing.bndseg[iseg].data[SLstepinbnd].time - XParam.totaltime;
				}

				//itime = SLstepinbnd - 1.0 + (totaltime - bndseg.data[SLstepinbnd - 1].time) / (bndseg.data[SLstepinbnd].time - bndseg.data[SLstepinbnd - 1].time);
				zsbnduni = zsbnduni + interptime(XForcing.bndseg[iseg].data[SLstepinbnd].wspeed, XForcing.bndseg[iseg].data[SLstepinbnd - 1].wspeed, XForcing.bndseg[iseg].data[SLstepinbnd].time - XForcing.bndseg[iseg].data[SLstepinbnd - 1].time, XParam.totaltime - XForcing.bndseg[iseg].data[SLstepinbnd - 1].time);

			}
			else
			{
				ndyn++;
				Forcingthisstep(XParam, XParam.totaltime, XForcing.bndseg[iseg].WLmap);
			}
		}
	}
	if (nuni > 0)
	{
		zsbnduni = zsbnduni / nuni;
	}

	int ib;
	double xi, yi;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				int n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;

				double levdx = calcres(XParam.dx, XBlock.level[ib]);
				xi = XParam.xo + XBlock.xo[ib] + i * levdx;
				yi = XParam.yo + XBlock.yo[ib] + j * levdx;

				zsbnd = zsbnduni;

				if (ndyn > 0)
				{
					zsbnd = zsbnduni * nuni;
					
					for (int iseg = 0; iseg < XForcing.bndseg.size(); iseg++)
					{
						if (XForcing.bndseg[iseg].on && !XForcing.bndseg[iseg].uniform)
						{
							//
							zsbnd = zsbnd + float(interp2BUQ(xi, yi, XForcing.bndseg[iseg].WLmap));
						}
					}

					zsbnd = zsbnd / (nuni + ndyn);
				}

				if (XParam.atmpforcing)
				{
					float atmpi;

					if (XForcing.Atmp.uniform)
					{
						atmpi = float(XForcing.Atmp.nowvalue);
					}
					else
					{
						atmpi = float(interp2BUQ(xi, yi, XForcing.Atmp));
					}
					zsbnd = zsbnd - (atmpi - (T)XParam.Paref) * (T)XParam.Pa2m;
				}

				XEv.zs[n] = utils::max(zsbnd, zb[n]);
				XEv.h[n] = utils::max(XEv.zs[n] - zb[n], T(0.0));
				XEv.u[n] = T(0.0);
				XEv.v[n] = T(0.0);
			}
		}
	}

}


/**
 * @brief Legacy warm start initialization using inverse distance to boundaries.
 *
 * Sets initial water level, velocity, and bathymetry arrays for all blocks using inverse distance interpolation from boundaries.
 *
 * @tparam T Data type
 * @param XParam Simulation parameters
 * @param XForcing Forcing/boundary conditions
 * @param XBlock Block parameters
 * @param zb Bathymetry array
 * @param XEv Evolving variables (output)
 */
template <class T>
void warmstartold(Param XParam,Forcing<float> XForcing, BlockP<T> XBlock, T* zb, EvolvingP<T>& XEv)
{
	// This function read water level boundary if they have been setup and calculate the distance to the boundary 
	// toward the end the water level value is calculated as an inverse distance to the available boundaries.
	// While this may look convoluted its working quite simply.
	// look for each boundary side and calculate the closest water level value and the distance to that value

	double zsleft = 0.0;
	double zsright = 0.0;
	double zstop = 0.0;
	double zsbot = 0.0;
	T zsbnd = 0.0;

	double distleft, distright, disttop, distbot;

	double lefthere = 0.0;
	double righthere = 0.0;
	double tophere = 0.0;
	double bothere = 0.0;

	double xi, yi, jj, ii;
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				int n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;

				double levdx = calcres(XParam.dx, XBlock.level[ib]);
				xi = XParam.xo + XBlock.xo[ib] + i * levdx;
				yi = XParam.yo + XBlock.yo[ib] + j * levdx;

				disttop = max((XParam.ymax - yi) / levdx, 0.1);//max((double)(ny - 1) - j, 0.1);// WTF is that 0.1? // distleft cannot be 0 //theoretical minumun is 0.5?
				distbot = max((yi - XParam.yo) / levdx, 0.1);
				distleft = max((xi - XParam.xo) / levdx, 0.1);//max((double)i, 0.1);
				distright = max((XParam.xmax - xi) / levdx, 0.1);//max((double)(nx - 1) - i, 0.1);

				jj = (yi - XParam.yo) / (XParam.ymax - XParam.yo);
				ii = (xi - XParam.xo) / (XParam.xmax - XParam.xo);

				if (XForcing.left.on)
				{
					lefthere = 1.0;
					int SLstepinbnd = 1;



					// Do this for all the corners
					//Needs limiter in case WLbnd is empty
					double difft = XForcing.left.data[SLstepinbnd].time - XParam.totaltime;

					while (difft < 0.0)
					{
						SLstepinbnd++;
						difft = XForcing.left.data[SLstepinbnd].time - XParam.totaltime;
					}
					std::vector<double> zsbndvec;
					for (int k = 0; k < XForcing.left.data[SLstepinbnd].wlevs.size(); k++)
					{
						zsbndvec.push_back(interptime(XForcing.left.data[SLstepinbnd].wlevs[k], XForcing.left.data[SLstepinbnd - 1].wlevs[k], XForcing.left.data[SLstepinbnd].time - XForcing.left.data[SLstepinbnd - 1].time, XParam.totaltime - XForcing.left.data[SLstepinbnd - 1].time));

					}
					if (zsbndvec.size() == 1)
					{
						zsleft = zsbndvec[0];
					}
					else
					{
						int iprev = utils::min(utils::max((int)floor(jj * (zsbndvec.size() - 1)), 0), (int)zsbndvec.size() - 2);
						int inext = iprev + 1;
						// here interp time is used to interpolate to the right node rather than in time...
						zsleft = interptime(zsbndvec[inext], zsbndvec[iprev], 1.0, (double)(jj * (zsbndvec.size() - 1) - iprev));
					}

				}

				if (XForcing.right.on)
				{
					int SLstepinbnd = 1;
					righthere = 1.0;


					// Do this for all the corners
					//Needs limiter in case WLbnd is empty
					double difft = XForcing.right.data[SLstepinbnd].time - XParam.totaltime;

					while (difft < 0.0)
					{
						SLstepinbnd++;
						difft = XForcing.right.data[SLstepinbnd].time - XParam.totaltime;
					}
					std::vector<double> zsbndvec;
					for (int k = 0; k < XForcing.right.data[SLstepinbnd].wlevs.size(); k++)
					{
						zsbndvec.push_back(interptime(XForcing.right.data[SLstepinbnd].wlevs[k], XForcing.right.data[SLstepinbnd - 1].wlevs[k], XForcing.right.data[SLstepinbnd].time - XForcing.right.data[SLstepinbnd - 1].time, XParam.totaltime - XForcing.right.data[SLstepinbnd - 1].time));

					}
					if (zsbndvec.size() == 1)
					{
						zsright = zsbndvec[0];
					}
					else
					{
						int iprev = utils::min(utils::max((int)floor(jj * (zsbndvec.size() - 1)), 0), (int)zsbndvec.size() - 2);
						int inext = iprev + 1;
						// here interp time is used to interpolate to the right node rather than in time...
						zsright = interptime(zsbndvec[inext], zsbndvec[iprev], 1.0, (double)(jj * (zsbndvec.size() - 1) - iprev));
					}


				}
				if (XForcing.bot.on)
				{
					int SLstepinbnd = 1;
					bothere = 1.0;




					// Do this for all the corners
					//Needs limiter in case WLbnd is empty
					double difft = XForcing.bot.data[SLstepinbnd].time - XParam.totaltime;

					while (difft < 0.0)
					{
						SLstepinbnd++;
						difft = XForcing.bot.data[SLstepinbnd].time - XParam.totaltime;
					}
					std::vector<double> zsbndvec;
					for (int k = 0; k < XForcing.bot.data[SLstepinbnd].wlevs.size(); k++)
					{
						zsbndvec.push_back(interptime(XForcing.bot.data[SLstepinbnd].wlevs[k], XForcing.bot.data[SLstepinbnd - 1].wlevs[k], XForcing.bot.data[SLstepinbnd].time - XForcing.bot.data[SLstepinbnd - 1].time, XParam.totaltime - XForcing.bot.data[SLstepinbnd - 1].time));

					}
					if (zsbndvec.size() == 1)
					{
						zsbot = zsbndvec[0];
					}
					else
					{
						int iprev = utils::min(utils::max((int)floor(ii * (zsbndvec.size() - 1)), 0), (int)zsbndvec.size() - 2);
						int inext = iprev + 1;
						// here interp time is used to interpolate to the right node rather than in time...
						zsbot = interptime(zsbndvec[inext], zsbndvec[iprev], 1.0, (double)(ii * (zsbndvec.size() - 1) - iprev));
					}

				}
				if (XForcing.top.on)
				{
					int SLstepinbnd = 1;
					tophere = 1.0;




					// Do this for all the corners
					//Needs limiter in case WLbnd is empty
					double difft = XForcing.top.data[SLstepinbnd].time - XParam.totaltime;

					while (difft < 0.0)
					{
						SLstepinbnd++;
						difft = XForcing.top.data[SLstepinbnd].time - XParam.totaltime;
					}
					std::vector<double> zsbndvec;
					for (int k= 0; k < XForcing.top.data[SLstepinbnd].wlevs.size(); k++)
					{
						zsbndvec.push_back(interptime(XForcing.top.data[SLstepinbnd].wlevs[k], XForcing.top.data[SLstepinbnd - 1].wlevs[k], XForcing.top.data[SLstepinbnd].time - XForcing.top.data[SLstepinbnd - 1].time, XParam.totaltime - XForcing.top.data[SLstepinbnd - 1].time));

					}
					if (zsbndvec.size() == 1)
					{
						zstop = zsbndvec[0];
					}
					else
					{
						int iprev = utils::min(utils::max((int)floor(ii * (zsbndvec.size() - 1)), 0), (int)zsbndvec.size() - 2);
						int inext = iprev + 1;
						// here interp time is used to interpolate to the right node rather than in time...
						zstop = interptime(zsbndvec[inext], zsbndvec[iprev], 1.0, (double)(ii * (zsbndvec.size() - 1) - iprev));
					}

				}


				zsbnd = T(((zsleft / distleft) * lefthere + (zsright / distright) * righthere + (zstop / disttop) * tophere + (zsbot / distbot) * bothere) / ((1.0 / distleft) * lefthere + (1.0 / distright) * righthere + (1.0 / disttop) * tophere + (1.0 / distbot) * bothere));

				if (XParam.atmpforcing)
				{
					float atmpi;

					if (XForcing.Atmp.uniform)
					{
						atmpi = float(XForcing.Atmp.nowvalue);
					}
					else
					{
						atmpi = float(interp2BUQ(xi, yi, XForcing.Atmp));
					}
					zsbnd = zsbnd - (atmpi- (T)XParam.Paref) * (T)XParam.Pa2m;
				}

				XEv.zs[n] = utils::max(zsbnd, zb[n]);
				XEv.h[n] = utils::max(XEv.zs[n] - zb[n], T(0.0));
				XEv.u[n] = T(0.0);
				XEv.v[n] = T(0.0);



			}
		}
	}
}


/**
 * @brief Add offset to surface elevation (zs) and update water depth (h).
 *
 * Applies zsoffset to zs and updates h for all blocks where h > eps.
 *
 * @tparam T Data type
 * @param XParam Simulation parameters
 * @param XBlock Block parameters
 * @param XEv Evolving variables (input/output)
 * @param zb Bathymetry array
 * @return Success flag (1 if successful)
 */
template <class T>
int AddZSoffset(Param XParam, BlockP<T> XBlock, EvolvingP<T> &XEv, T*zb)
{
	int success = 1;
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				int n = memloc(XParam, i, j, ib);

				if (XEv.h[n] > XParam.eps)
				{

					XEv.zs[n] = max(XEv.zs[n] + T(XParam.zsoffset), zb[n]);

					XEv.h[n] = utils::max(XEv.zs[n] - zb[n], T(0.0));
				}
			}

		}
	}

	return success;
}


template <class T> static int readhotstartfileBG(Param XParam, BlockP<T> XBlock, EvolvingP<T>& XEv, T*& zb);

/**
 * @brief Read hotstart file and initialize evolving variables and bathymetry.
 *
 * Reads NetCDF hotstart file, extracts variables, and fills arrays for all blocks.
 * Handles missing variables and applies edge corrections.
 *
 * @tparam T Data type
 * @param XParam Simulation parameters
 * @param XBlock Block parameters
 * @param XEv Evolving variables (output)
 * @param zb Bathymetry array (output)
 * @return Success flag (1 if successful, 0 if fallback to cold start)
 */
template <class T> int readhotstartfile(Param XParam, BlockP<T> XBlock, EvolvingP<T>& XEv, T*& zb)
{
	// Try BG_Flood direct multires hotstart first
	int bgok = readhotstartfileBG(XParam, XBlock, XEv, zb);
	if (bgok == 1)
	{
		return 1;
	}

	// Existing legacy interpolation-based hotstart path below
	int status;
	int ncid;
	//int dimids[NC_MAX_VAR_DIMS];   // dimension IDs 
	int ib;
	//double scalefac = 1.0;
	//double offset = 0.0;
	
	std::string zbname, zsname, hname, uname, vname, xname, yname;
	// Open the file for read access
	//netCDF::NcFile dataFile(XParam.hotstartfile, NcFile::read);


	//Open NC file
	printf("Open file...");
	status = nc_open(XParam.hotstartfile.c_str(), NC_NOWRITE, &ncid);


	//bool isBG_Flood = false;

	// read ncfile attribute and see if BG_flood global attribute exists.

	//if it exist read each level separatly otherwise look for the following variables 

	if (status != NC_NOERR) handle_ncerror(status);
	zbname = checkncvarname(ncid, "zb", "z", "ZB", "Z", "zb_P0");
	zsname = checkncvarname(ncid, "zs", "eta", "ZS", "ETA", "zs_P0");
	hname = checkncvarname(ncid, "h", "hh", "hhh", "hhhh", "h_P0");
	uname = checkncvarname(ncid, "u", "uu", "uvel", "UVEL", "u_P0");
	vname = checkncvarname(ncid, "v", "vv", "vvel", "VVEL", "v_P0");

	//by default we assume that the x axis is called "xx" but that is not sure "x" shoudl be accepted and so does "lon" for spherical grid
	// The folowing section figure out which one is in the file and if none exits with the netcdf error
	// default name is "xx"
	//xname = checkncvarname(ncid, "x", "xx","lon","Lon");
	//yname = checkncvarname(ncid, "y", "yy", "lat", "Lat");

	status = nc_close(ncid);


	// First we should read x and y coordinates
	// Just as with other variables we expect the file follow the output naming convention of "xx" and "yy" both as a dimension and a variable
	StaticForcingP<float> zbhotstart, zshotstart, hhotstart, uhotstart, vhotstart;

	// Read hotstart block info if it exist
	// By default reuse mesh-layout
	// for now we pretend hotstart are just unifomr maesh layout



	//if hotstart has zb variable overright the previous ne
	//printf("Found variables: ");
	if (!zbname.empty())
	{
		//zb is set
		zbhotstart = readfileinfo(XParam.hotstartfile + "?" + zbname, zbhotstart);

		readstaticforcing(XParam.hotstep, zbhotstart);
		interp2BUQ(XParam, XBlock, zbhotstart, zb);

		//because we set the edges around empty blocks we need the edges for zs too
		// otherwise we create some gitantic waves at the edges of empty blocks
		setedges(XParam, XBlock, zb);



	}
	// second check if zs or hh are in the file


	//zs Section
	if (!zsname.empty())
	{
		log(" zs... ");

		zshotstart = readfileinfo(XParam.hotstartfile + "?" + zsname, zshotstart);
		//readforcingmaphead(zshotstart);
		readstaticforcing(XParam.hotstep, zshotstart);

		interp2BUQ(XParam, XBlock, zshotstart, XEv.zs);

		setedges(XParam, XBlock, XEv.zs);

		//setedges(XParam.nblk, leftblk, rightblk, topblk, botblk, zs);

		//check sanity
		for (int ibl = 0; ibl < XParam.nblk; ibl++)
		{
			ib = XBlock.active[ibl];
			for (int j = 0; j < XParam.blkwidth; j++)
			{
				for (int i = 0; i < XParam.blkwidth; i++)
				{
					int n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
					XEv.zs[n] = utils::max(XEv.zs[n], zb[n]);
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}


	}
	else
	{
		//Variable not found
		//It's ok if hh is specified
		log("zs not found in hotstart file. Looking for hh... ");

	}

	//hh section
	if (!hname.empty())
	{
		log("h... ");
		hhotstart = readfileinfo(XParam.hotstartfile + "?" + hname, hhotstart);
		//readforcingmaphead(zshotstart);
		readstaticforcing(XParam.hotstep, hhotstart);

		interp2BUQ(XParam, XBlock, hhotstart, XEv.h);

		setedges(XParam, XBlock, XEv.h);

		//if zs was not specified
		if (zsname.empty())
		{
			for (int ibl = 0; ibl < XParam.nblk; ibl++)
			{
				ib = XBlock.active[ibl];
				for (int j = 0; j < XParam.blkwidth; j++)
				{
					for (int i = 0; i < XParam.blkwidth; i++)
					{
						int n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
						XEv.zs[n] = zb[n] + XEv.h[n];
						//unpacked_value = packed_value * scale_factor + add_offset
					}
				}
			}

		}
		


	}
	else
	{
		//if both zs and h were not specified
		if (zsname.empty() && hname.empty())
		{
			//Variable not found
			//It's ok if hh is specified
			log("neither zs nor hh were found in hotstart file. this is not a valid hotstart file. using a cold start instead");
			return 0;
		}
		else
		{
			//zs was specified but not h
			for (int ibl = 0; ibl < XParam.nblk; ibl++)
			{
				ib = XBlock.active[ibl];
				for (int j = 0; j < XParam.blkwidth; j++)
				{
					for (int i = 0; i < XParam.blkwidth; i++)
					{
						int n = memloc(XParam, i, j, ib);


						XEv.h[n] = utils::max(XEv.zs[n] - zb[n], T(0.0));
					}

				}
			}

		}
	}

	//u Section

	if (!uname.empty())
	{
		log("u... ");
		uhotstart = readfileinfo(XParam.hotstartfile + "?" + uname, uhotstart);
		//readforcingmaphead(zshotstart);
		readstaticforcing(XParam.hotstep, uhotstart);

		interp2BUQ(XParam, XBlock, uhotstart, XEv.u);

		setedges(XParam, XBlock, XEv.u);

	}
	else
	{
		InitArrayBUQ(XParam, XBlock, (T)0.0, XEv.u);
	}

	//vv section

	if (!vname.empty())
	{
		log("v... ");
		vhotstart = readfileinfo(XParam.hotstartfile + "?" + vname, vhotstart);
		//readforcingmaphead(zshotstart);
		readstaticforcing(XParam.hotstep, vhotstart);

		interp2BUQ(XParam, XBlock, vhotstart, XEv.v);

		setedges(XParam, XBlock, XEv.v);


	}
	else
	{
		InitArrayBUQ(XParam,XBlock, (T)0.0, XEv.v);
	}
	//status = nc_get_var_float(ncid, hh_id, zb);
	status = nc_close(ncid);



	return 1;

}
template int readhotstartfile<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float>& XEv, float*& zb);
template int readhotstartfile<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double>& XEv, double*& zb);
//template int readhotstartfile<float>(Param XParam, int * leftblk, int *rightblk, int * topblk, int* botblk, double * blockxo, double * blockyo, float * &zs, float * &zb, float * &hh, float *&uu, float * &vv);

//template int readhotstartfile<double>(Param XParam, int * leftblk, int *rightblk, int * topblk, int* botblk, double * blockxo, double * blockyo, double * &zs, double * &zb, double * &hh, double *&uu, double * &vv);


template <class T>
static int readncpointscaled(int ncid, int varid, const std::vector<size_t>& start, T& outval)
{
	float raw = 0.0f;
	int status = nc_get_var1_float(ncid, varid, start.data(), &raw);
	if (status != NC_NOERR)
	{
		return status;
	}

	double scalefac = 1.0;
	double offset = 0.0;
	int sferr = nc_get_att_double(ncid, varid, "scale_factor", &scalefac);
	int oferr = nc_get_att_double(ncid, varid, "add_offset", &offset);

	double val = static_cast<double>(raw);
	if (sferr == NC_NOERR || oferr == NC_NOERR)
	{
		val = val * scalefac + offset;
	}

	outval = static_cast<T>(val);
	return NC_NOERR;
}

template <class T>
static int readhotstartfileBG(Param XParam, BlockP<T> XBlock, EvolvingP<T>& XEv, T*& zb)
{
	int status;
	int ncid;

	status = nc_open(XParam.hotstartfile.c_str(), NC_NOWRITE, &ncid);
	if (status != NC_NOERR)
	{
		return 0;
	}

	int bgVersion = -1;
	status = nc_get_att_int(ncid, NC_GLOBAL, "BG_Flood", &bgVersion);
	if (status != NC_NOERR || bgVersion < 0)
	{
		nc_close(ncid);
		return 0; // not a BG_Flood output
	}

	struct LevelInfo
	{
		bool ready = false;
		int nx = 0;
		int ny = 0;
		double x0 = 0.0;
		double y0 = 0.0;
		double dx = 0.0;
		double dy = 0.0;

		int zb_id = -1, zs_id = -1, h_id = -1, u_id = -1, v_id = -1;
		int zb_nd = 0, zs_nd = 0, h_nd = 0, u_nd = 0, v_nd = 0;
	};

	std::map<int, LevelInfo> levelCache;

	auto get_varid_if_exists = [&](const std::string& name) -> int
	{
		int varid = -1;
		int err = nc_inq_varid(ncid, name.c_str(), &varid);
		return (err == NC_NOERR) ? varid : -1;
	};

	auto get_ndims_if_exists = [&](int varid) -> int
	{
		if (varid < 0)
		{
			return 0;
		}
		int nd = 0;
		if (nc_inq_varndims(ncid, varid, &nd) != NC_NOERR)
		{
			return 0;
		}
		return nd;
	};

	int timedimid = -1;
	size_t ntime = 1;
	if (nc_inq_dimid(ncid, "time", &timedimid) == NC_NOERR)
	{
		nc_inq_dimlen(ncid, timedimid, &ntime);
	}
	const int hotstep = static_cast<int>(utils::min(utils::max(XParam.hotstep, 0), static_cast<int>(ntime) - 1));

	bool hasZbAny = false;
	bool hasZsAny = false;
	bool hasHAny = false;
	bool hasUAny = false;
	bool hasVAny = false;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		const int ib = XBlock.active[ibl];
		const int lev = XBlock.level[ib];

		LevelInfo& L = levelCache[lev];
		if (!L.ready)
		{
			const std::string sign = (lev < 0) ? "N" : "P";
			const std::string levtag = sign + std::to_string(abs(lev));

			const std::string xxname = "xx_" + levtag;
			const std::string yyname = "yy_" + levtag;

			int xxid = -1, yyid = -1;
			if (nc_inq_varid(ncid, xxname.c_str(), &xxid) != NC_NOERR ||
				nc_inq_varid(ncid, yyname.c_str(), &yyid) != NC_NOERR)
			{
				nc_close(ncid);
				return 0; // cannot do direct multires read without per-level coordinates
			}

			size_t nx = 0, ny = 0;
			int xndims = 0, yndims = 0;
			int xdimids[NC_MAX_VAR_DIMS], ydimids[NC_MAX_VAR_DIMS];

			nc_inq_varndims(ncid, xxid, &xndims);
			nc_inq_varndims(ncid, yyid, &yndims);
			nc_inq_vardimid(ncid, xxid, xdimids);
			nc_inq_vardimid(ncid, yyid, ydimids);

			nc_inq_dimlen(ncid, xdimids[0], &nx);
			nc_inq_dimlen(ncid, ydimids[0], &ny);

			std::vector<size_t> s0(1, 0), s1(1, 1);
			double x0 = 0.0, x1 = 0.0, y0 = 0.0, y1 = 0.0;
			nc_get_var1_double(ncid, xxid, s0.data(), &x0);
			nc_get_var1_double(ncid, yyid, s0.data(), &y0);
			if (nx > 1) nc_get_var1_double(ncid, xxid, s1.data(), &x1); else x1 = x0 + calcres(XParam.dx, lev);
			if (ny > 1) nc_get_var1_double(ncid, yyid, s1.data(), &y1); else y1 = y0 + calcres(XParam.dx, lev);

			L.nx = static_cast<int>(nx);
			L.ny = static_cast<int>(ny);
			L.x0 = x0;
			L.y0 = y0;
			L.dx = x1 - x0;
			L.dy = y1 - y0;

			L.zb_id = get_varid_if_exists("zb_" + levtag);
			L.zs_id = get_varid_if_exists("zs_" + levtag);
			L.h_id = get_varid_if_exists("h_" + levtag);
			L.u_id = get_varid_if_exists("u_" + levtag);
			L.v_id = get_varid_if_exists("v_" + levtag);

			L.zb_nd = get_ndims_if_exists(L.zb_id);
			L.zs_nd = get_ndims_if_exists(L.zs_id);
			L.h_nd = get_ndims_if_exists(L.h_id);
			L.u_nd = get_ndims_if_exists(L.u_id);
			L.v_nd = get_ndims_if_exists(L.v_id);

			hasZbAny = hasZbAny || (L.zb_id >= 0);
			hasZsAny = hasZsAny || (L.zs_id >= 0);
			hasHAny = hasHAny || (L.h_id >= 0);
			hasUAny = hasUAny || (L.u_id >= 0);
			hasVAny = hasVAny || (L.v_id >= 0);

			L.ready = true;
		}

		const double bx = XParam.xo + XBlock.xo[ib];
		const double by = XParam.yo + XBlock.yo[ib];

		const int io = static_cast<int>(round((bx - L.x0) / L.dx));
		const int jo = static_cast<int>(round((by - L.y0) / L.dy));

		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				const int ii = io + i;
				const int jj = jo + j;

				if (ii < 0 || ii >= L.nx || jj < 0 || jj >= L.ny)
				{
					continue;
				}

				const int n = memloc(XParam, i, j, ib);

				auto build_start = [&](int ndims) -> std::vector<size_t>
				{
					if (ndims == 3)
					{
						return { static_cast<size_t>(hotstep), static_cast<size_t>(jj), static_cast<size_t>(ii) };
					}
					else if (ndims == 2)
					{
						return { static_cast<size_t>(jj), static_cast<size_t>(ii) };
					}
					return {};
				};

				if (L.zb_id >= 0)
				{
					std::vector<size_t> s = build_start(L.zb_nd);
					if (!s.empty())
					{
						readncpointscaled(ncid, L.zb_id, s, zb[n]);
					}
				}
				if (L.zs_id >= 0)
				{
					std::vector<size_t> s = build_start(L.zs_nd);
					if (!s.empty())
					{
						readncpointscaled(ncid, L.zs_id, s, XEv.zs[n]);
					}
				}
				if (L.h_id >= 0)
				{
					std::vector<size_t> s = build_start(L.h_nd);
					if (!s.empty())
					{
						readncpointscaled(ncid, L.h_id, s, XEv.h[n]);
					}
				}
				if (L.u_id >= 0)
				{
					std::vector<size_t> s = build_start(L.u_nd);
					if (!s.empty())
					{
						readncpointscaled(ncid, L.u_id, s, XEv.u[n]);
					}
				}
				if (L.v_id >= 0)
				{
					std::vector<size_t> s = build_start(L.v_nd);
					if (!s.empty())
					{
						readncpointscaled(ncid, L.v_id, s, XEv.v[n]);
					}
				}
			}
		}
	}

	nc_close(ncid);

	if (!hasZsAny && !hasHAny)
	{
		log("BG hotstart: neither zs nor h found. Falling back to legacy hotstart reader.");
		return 0;
	}

	// Reconstruct missing state
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		const int ib = XBlock.active[ibl];
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				const int n = memloc(XParam, i, j, ib);

				if (!hasZsAny && hasHAny)
				{
					XEv.zs[n] = zb[n] + XEv.h[n];
				}
				if (hasZsAny && !hasHAny)
				{
					XEv.h[n] = utils::max(XEv.zs[n] - zb[n], T(0.0));
				}
				if (!hasUAny)
				{
					XEv.u[n] = T(0.0);
				}
				if (!hasVAny)
				{
					XEv.v[n] = T(0.0);
				}
			}
		}
	}

	// Restore edges/halo consistency
	setedges(XParam, XBlock, zb);
	setedges(XParam, XBlock, XEv.zs);
	setedges(XParam, XBlock, XEv.h);
	setedges(XParam, XBlock, XEv.u);
	setedges(XParam, XBlock, XEv.v);

	log("BG hotstart: direct multires read successful (no interpolation).");
	return 1;
}

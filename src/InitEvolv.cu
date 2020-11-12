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
		if (!std::isnan(XParam.zsoffset)) // apply specified zsoffset
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
		
		if (std::isnan(XParam.zsinit) && (!XParam.leftbnd && !XParam.rightbnd && !XParam.topbnd && !XParam.botbnd)) //zsinit is default
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


template <class T>
int coldstart(Param XParam, BlockP<T> XBlock, T* zb, EvolvingP<T> & XEv)
{
	T zzini = std::isnan(XParam.zsinit)? T(0.0): T(XParam.zsinit);
	T zzoffset = std::isnan(XParam.zsoffset) ? T(0.0) : T(XParam.zsoffset);
	

	
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
	return coldstartsucess = 1;
}


template <class T>
void warmstart(Param XParam,Forcing<float> XForcing, BlockP<T> XBlock, T* zb, EvolvingP<T>& XEv)
{
	// This function read water level boundary if they have nbeen setup and calculate thedistance to the boundary 
	// toward the end the water level value is calculated as an inverse distnace to the available boundaries.
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


				zsbnd = ((zsleft / distleft) * lefthere + (zsright / distright) * righthere + (zstop / disttop) * tophere + (zsbot / distbot) * bothere) / ((1.0 / distleft) * lefthere + (1.0 / distright) * righthere + (1.0 / disttop) * tophere + (1.0 / distbot) * bothere);



				XEv.zs[n] = utils::max(zsbnd, zb[n]);
				XEv.h[n] = utils::max(XEv.zs[n] - zb[n], T(0.0));
				XEv.u[n] = T(0.0);
				XEv.v[n] = T(0.0);



			}
		}
	}
}


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


template <class T>
int readhotstartfile(Param XParam, BlockP<T> XBlock, EvolvingP<T>& XEv, T*& zb)
{
	int status;
	int ncid, varid, ndims;
	//int dimids[NC_MAX_VAR_DIMS];   // dimension IDs 
	int ib;
	double scalefac = 1.0;
	double offset = 0.0;
	
	std::string zbname, zsname, hname, uname, vname, xname, yname;
	// Open the file for read access
	//netCDF::NcFile dataFile(XParam.hotstartfile, NcFile::read);


	//Open NC file
	printf("Open file...");
	status = nc_open(XParam.hotstartfile.c_str(), NC_NOWRITE, &ncid);
	if (status != NC_NOERR) handle_ncerror(status);
	zbname = checkncvarname(ncid, "zb", "z", "ZB", "Z");
	zsname = checkncvarname(ncid, "zs", "eta", "ZS", "ETA");
	hname = checkncvarname(ncid, "h", "hh", "hhh", "hhhh");
	uname = checkncvarname(ncid, "u", "uu", "uvel", "UVEL");
	vname = checkncvarname(ncid, "v", "vv", "vvel", "VVEL");

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

		//because we set the edges around empty blocks we need the set the edges for zs too
		// otherwise we create some gitantic waves at the edges of empty blocks
		setedges(XParam, XBlock, zb);



	}
	// second check if zs or hh are in teh file


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

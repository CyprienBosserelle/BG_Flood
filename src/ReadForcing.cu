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




#include "ReadForcing.h"


/*! \fn void readforcing(Param & XParam, Forcing<T> & XForcing)
* wrapping function for reading all the forcing data
* 
*/
template <class T>
void readforcing(Param & XParam, Forcing<T> & XForcing)
{
	int nt;

	//=================
	// Read Bathymetry
	log("\nReading bathymetry grid data...");
	for (int ib = 0; ib < XForcing.Bathy.size(); ib++)
	{
		readbathydata(XParam.posdown, XForcing.Bathy[ib]);

		if (ib == 0) // Fill Nan for only the first bathy listed, the others will use values from original bathy topo.
		{
			denan(XForcing.Bathy[ib].nx, XForcing.Bathy[ib].ny, T(0.0), XForcing.Bathy[ib].val);
		}
	}
	
	if (XForcing.Bathy[0].extension.compare("nc") == 0)
	{
		//Get_CRS information from last bathymetry file
		XParam.crs_ref = readCRSfrombathy(XParam.crs_ref, XForcing.Bathy[XForcing.Bathy.size() - 1]);

		//XParam.crs_ref = "test2";
	}
	bool gpgpu = XParam.GPUDEVICE >= 0;

	//=================
	// Read bnd files
	log("\nReading boundary data...");

	AllocateCPU(1, 1, XForcing.left.blks, XForcing.right.blks, XForcing.top.blks, XForcing.bot.blks);
	

	if (!XForcing.left.inputfile.empty())
	{
		//XParam.leftbnd.data = readWLfile(XParam.leftbnd.inputfile);
		XForcing.left.data = readbndfile(XForcing.left.inputfile, XParam, 0);

		XForcing.left.on = true; 
		XForcing.left.nbnd = int(XForcing.left.data[0].wlevs.size());

		if (gpgpu)
		{
			AllocateBndTEX(XForcing.left);
		}
		

	}
	if (!XForcing.right.inputfile.empty())
	{
		XForcing.right.data = readbndfile(XForcing.right.inputfile, XParam, 2);
		XForcing.right.on = true;
		XForcing.right.nbnd = int(XForcing.right.data[0].wlevs.size());
		if (gpgpu)
		{
			AllocateBndTEX(XForcing.right);
		}
	}
	if (!XForcing.top.inputfile.empty())
	{
		XForcing.top.data = readbndfile(XForcing.top.inputfile, XParam, 3);
		XForcing.top.on = true;
		XForcing.top.nbnd = int(XForcing.top.data[0].wlevs.size());
		if (gpgpu)
		{
			AllocateBndTEX(XForcing.top);
		}
	}
	if (!XForcing.bot.inputfile.empty())
	{
		XForcing.bot.data = readbndfile(XForcing.bot.inputfile, XParam, 1);
		XForcing.bot.on = true;
		XForcing.bot.nbnd = int(XForcing.bot.data[0].wlevs.size());
		if (gpgpu)
		{
			AllocateBndTEX(XForcing.bot);
		}
	}

	//Check that endtime is no longer than boundaries (if specified to other than wall or neumann)
	// Removed. This is better done in the sanity check!
	////XParam.endtime = setendtime(XParam, XForcing);

	//log("...done");

	//==================
	// Friction maps 
		
	if (!XForcing.cf.inputfile.empty())
	{
		XForcing.cf.denanval = 0.0000001;
		log("\nRead Roughness map (cf) data...");
		// roughness map was specified!
		readstaticforcing(XForcing.cf);

		//log("...done");
	}

	//==================
	// Rain losses maps

	if (!XForcing.il.inputfile.empty())
	{
		log("\nRead initial losses map (il) data...");
		XForcing.il.denanval = 0.0;

		readstaticforcing(XForcing.il);
		XParam.infiltration = true;
	}
	if (!XForcing.cl.inputfile.empty())
	{
		log("\nRead initial losses map (cl) data...");
		XForcing.cl.denanval = 0.0;
		readstaticforcing(XForcing.cl);
		XParam.infiltration = true;
	}
	  
	//=====================
	// Deformation (tsunami generation)
	if (XForcing.deform.size() > 0)
	{
		log("\nRead deform data...");
		// Deformation files was specified!

		for (int nd = 0; nd < XForcing.deform.size(); nd++)
		{
			XForcing.deform[nd].denanval = 0.0;
			// read the roughness map header
			readstaticforcing(XForcing.deform[nd]);
			//XForcing.deform[nd].grid = readcfmaphead(XForcing.deform[nd].grid);

			//Clamp edges to 0.0
			clampedges(XForcing.deform[nd].nx, XForcing.deform[nd].ny, 0.0f, XForcing.deform[nd].val);
			

			
			XParam.deformmaxtime = utils::max(XParam.deformmaxtime, XForcing.deform[nd].startime + XForcing.deform[nd].duration);

			AllocateTEX(XForcing.deform[nd].nx, XForcing.deform[nd].ny, XForcing.deform[nd].GPU, XForcing.deform[nd].val);

			// below might seem redundant but it simplifies the 
			// template <class T> __device__ T interpDyn2BUQ(T x, T y, TexSetP Forcing)
			// function
			XForcing.deform[nd].GPU.xo = float(XForcing.deform[nd].xo);
			XForcing.deform[nd].GPU.yo = float(XForcing.deform[nd].yo);
			XForcing.deform[nd].GPU.uniform = false;
			XForcing.deform[nd].GPU.dx = float(XForcing.deform[nd].dx);
		}
		//log("...done");

	}
	
	//=====================
	// Target level
	if (XParam.AdaptCrit.compare("Targetlevel") == 0)
	{
		log("\nRead Target level data...");
		for (int nd = 0; nd < XForcing.targetadapt.size(); nd++)
		{
			//
			readstaticforcing(XForcing.targetadapt[nd]);
		}
	}
	

	//======================
	// Rivers
	if (XForcing.rivers.size() > 0)
	{
		// This part of the code only reads the meta data and data 
		// the full initialisation and detection of river blocks is done in model initialisation
		log("\nPreparing rivers (" + std::to_string(XForcing.rivers.size()) + " rivers)");
		for (int Rin = 0; Rin < XForcing.rivers.size(); Rin++)
		{
			// Now read the discharge input and store to  
			XForcing.rivers[Rin].flowinput = readFlowfile(XForcing.rivers[Rin].Riverflowfile, XParam.reftime);

			//Check the time range of the river forcing
			nt = XForcing.rivers[Rin].flowinput.size();
			XForcing.rivers[Rin].to = XForcing.rivers[Rin].flowinput[0].time;
			XForcing.rivers[Rin].tmax = XForcing.rivers[Rin].flowinput[nt-1].time;
			if ( XForcing.rivers[Rin].tmax < XParam.endtime)
			{
				XParam.endtime = XForcing.rivers[Rin].tmax;
				log("\nWARNING: simulation endtime reduced to " + std::to_string(XParam.endtime) + " to fit the time range of the River number " + std::to_string(Rin));
			}
			if (XForcing.rivers[Rin].to > XParam.totaltime)
			{
				XParam.totaltime = XForcing.rivers[Rin].to;
				log("\nWARNING: simulation initial time increased to " + std::to_string(XParam.totaltime) + " to fit the time range of the River number " + std::to_string(Rin));
			}
		}
	}


	//======================
	// Wind file(s)
	if (!XForcing.UWind.inputfile.empty())
	{	
		log("\nPreparing Wind forcing");
		// This part of the code only reads the meta data and data for initial step
		// the full initialisation of the cuda array and texture is done in model initialisation
		if (XForcing.UWind.uniform == 1)
		{
			XForcing.VWind.uniform = true;

			// grid uniform time varying wind input: wlevs[0] is wind speed and wlev[1] is direction
			XForcing.VWind.inputfile = XForcing.UWind.inputfile;
			XForcing.UWind.unidata = readWNDfileUNI(XForcing.UWind.inputfile, XParam.reftime, XParam.grdalpha);
			XForcing.VWind.unidata = readWNDfileUNI(XForcing.VWind.inputfile, XParam.reftime, XParam.grdalpha);

			// this below is a bit ugly but it simplifies the other functions
			for (int n = 0; n < XForcing.VWind.unidata.size(); n++)
			{
				XForcing.VWind.unidata[n].wspeed = XForcing.VWind.unidata[n].vwind;
			}
			for (int n = 0; n < XForcing.UWind.unidata.size(); n++)
			{
				XForcing.UWind.unidata[n].wspeed = XForcing.UWind.unidata[n].uwind;
			}

			//Sanity check on the time range of the forcing
			nt = XForcing.UWind.unidata.size();
			if (XForcing.UWind.unidata[nt - 1].time < XParam.endtime || XForcing.VWind.unidata[nt - 1].time < XParam.endtime)
			{
				XParam.endtime = min(XForcing.UWind.unidata[nt - 1].time, XForcing.VWind.unidata[nt - 1].time);
				log("\nWARNING: simulation endtime reduced to " + std::to_string(XParam.endtime) + " to fit the time range of the wind forcing. ");
			}
			if (XForcing.UWind.unidata[0].time > XParam.totaltime || XForcing.VWind.unidata[0].time > XParam.totaltime)
			{
				XParam.totaltime = max(XForcing.UWind.unidata[0].time, XForcing.VWind.unidata[0].time);
				log("\nWARNING: simulation initial time increased to " + std::to_string(XParam.totaltime) + " to fit the time range of the wind forcing. ");
			}
			
		}
		else
		{
			//
			//readDynforcing(gpgpu, XParam.totaltime, XForcing.UWind);
			//readDynforcing(gpgpu, XParam.totaltime, XForcing.VWind);

			XForcing.UWind.denanval = 0.0;
			XForcing.VWind.denanval = 0.0;
			InitDynforcing(gpgpu, XParam, XForcing.UWind);
			InitDynforcing(gpgpu, XParam, XForcing.VWind);


			
		}

	}

	//======================
	// ATM file
	if (!XForcing.Atmp.inputfile.empty())
	{
		log("\nPreparing Atmospheric pressure forcing");
		// This part of the code only reads the meta data and data for initial step
		// the full initialisation of the cuda array and texture is done in model initialisation
		XForcing.Atmp.uniform = (XForcing.Atmp.extension.compare("nc") == 0) ? 0 : 1;
		if (XForcing.Atmp.uniform == 1)
		{
			// grid uniform time varying atm pressure input is pretty useless...
			XForcing.Atmp.unidata = readINfileUNI(XForcing.Atmp.inputfile, XParam.reftime);
		}
		else
		{
			XForcing.Atmp.denanval = XParam.Paref;
			InitDynforcing(gpgpu, XParam, XForcing.Atmp);
			//readDynforcing(gpgpu, XParam.totaltime, XForcing.Atmp);
		}
	}

	//======================
	// Rain file
	if (!XForcing.Rain.inputfile.empty())
	{
		log("\nPreparing Rain forcing");
		// This part of the code only reads the meta data and data for initial step
		// the full initialisation of the cuda array and texture is done in model initialisation
		if (XForcing.Rain.uniform == 1)
		{
			// grid uniform time varying rain input
			XForcing.Rain.unidata = readINfileUNI(XForcing.Rain.inputfile, XParam.reftime);
		}
		else
		{
			XForcing.Rain.denanval = 0.0;
			InitDynforcing(gpgpu, XParam, XForcing.Rain);
			//readDynforcing(gpgpu, XParam.totaltime, XForcing.Rain);
		}
	}

	//======================
	// Polygon data
	if (!XForcing.AOI.file.empty())
	{
		log("\nRead AOI polygon");

		//Polygon Poly;
		XForcing.AOI.poly = readPolygon(XForcing.AOI.file);
		
		// = CounterCWPoly(Poly);
		//
		
	}

	//======================
	// Done
	//======================
}

template void readforcing<float>(Param& XParam, Forcing<float>& XForcing);
//template void readforcing<double>(Param& XParam, Forcing<double>& XForcing);

/*! \fn  void readstaticforcing(T& Sforcing)
*  single parameter version of  readstaticforcing(int step,T& Sforcing)
* readstaticforcing(0, Sforcing);
*/
template <class T> void readstaticforcing(T& Sforcing)
{
	readstaticforcing(0, Sforcing);
}

template void readstaticforcing<deformmap<float>>(deformmap<float>& Sforcing);
template void readstaticforcing<StaticForcingP<float>>(StaticForcingP<float>& Sforcing);
template void readstaticforcing<StaticForcingP<int>>(StaticForcingP<int>& Sforcing);

/*! \fn  void readstaticforcing(int step,T& Sforcing)
* Allocate and read static (i.e. not varying in time) forcing
* Used for Bathy, roughness, deformation
*/
template <class T> void readstaticforcing(int step,T& Sforcing)
{
	Sforcing=readforcinghead(Sforcing);
	

	if (Sforcing.nx > 0 && Sforcing.ny > 0)
	{
		AllocateCPU(Sforcing.nx, Sforcing.ny, Sforcing.val);

		//readvarinfo(Sforcing.inputfile, Sforcing.varname, ddimU)
		// read the roughness map header
		//readvardata(0, Sforcing, Sforcing.val);
		readforcingdata(step,Sforcing);
		//readvardata(forcing.inputfile, forcing.varname, step, forcing.val);

		denan(Sforcing.nx, Sforcing.ny, float(Sforcing.denanval), Sforcing.val);

	}
	else
	{
		//Error message
		log("Error while reading forcing map file: " + Sforcing.inputfile);
	}
}
template void readstaticforcing<deformmap<float>>(int step, deformmap<float>& Sforcing);
template void readstaticforcing<StaticForcingP<float>>(int step, StaticForcingP<float>& Sforcing);
template void readstaticforcing<StaticForcingP<int>>(int step, StaticForcingP<int>& Sforcing);

/*! \fn void InitDynforcing(bool gpgpu,double totaltime,DynForcingP<float>& Dforcing)
* 
* Used for Rain, wind, Atm pressure
*/
void InitDynforcing(bool gpgpu,Param& XParam,DynForcingP<float>& Dforcing)
{
	Dforcing = readforcinghead(Dforcing, XParam);

	//Sanity check on the time range of the forcing
	if (Dforcing.tmax < XParam.endtime)
	{
		XParam.endtime = Dforcing.tmax;
		log("\nWARNING: simulation endtime reduced to " + std::to_string(XParam.endtime) + " to fit the time range provided in " + Dforcing.inputfile);
	}
	if (Dforcing.to > XParam.totaltime)
	{
		XParam.totaltime = Dforcing.to;
		log("\nWARNING: simulation initial time increased to " + std::to_string(XParam.totaltime) + " to fit the time provided in " + Dforcing.inputfile);
	}


	if (Dforcing.nx > 0 && Dforcing.ny > 0)
	{
		AllocateCPU(Dforcing.nx, Dforcing.ny, Dforcing.now, Dforcing.before, Dforcing.after, Dforcing.val);
		readforcingdata(XParam.totaltime, Dforcing);
		
		if (gpgpu)
		{ 
			AllocateGPU(Dforcing.nx, Dforcing.ny, Dforcing.now_g);
			AllocateGPU(Dforcing.nx, Dforcing.ny, Dforcing.before_g);
			AllocateGPU(Dforcing.nx, Dforcing.ny, Dforcing.after_g);
			CopytoGPU(Dforcing.nx, Dforcing.ny, Dforcing.now, Dforcing.now_g);
			CopytoGPU(Dforcing.nx, Dforcing.ny, Dforcing.before, Dforcing.before_g);
			CopytoGPU(Dforcing.nx, Dforcing.ny, Dforcing.after, Dforcing.after_g);

			// Allocate and bind textures
			AllocateTEX(Dforcing.nx, Dforcing.ny, Dforcing.GPU, Dforcing.now);

			// below might seem redundant but it simplifies the 
			// template <class T> __device__ T interpDyn2BUQ(T x, T y, TexSetP Forcing)
			// function
			Dforcing.GPU.xo = float(Dforcing.xo);
			Dforcing.GPU.yo = float(Dforcing.yo);
			Dforcing.GPU.uniform = Dforcing.uniform;
			Dforcing.GPU.dx = float(Dforcing.dx);
		}
		
	}
	else
	{
		//Error message
		log("Error while reading forcing map file: " + Dforcing.inputfile);
	}
}


/*! \fn void readDynforcing(bool gpgpu, double totaltime, DynForcingP<float>& Dforcing)
* This is a deprecated function! See InitDynforcing() instead
*
*/
void readDynforcing(bool gpgpu, double totaltime, DynForcingP<float>& Dforcing)
{
	Dforcing = readforcinghead(Dforcing);


	if (Dforcing.nx > 0 && Dforcing.ny > 0)
	{
		AllocateCPU(Dforcing.nx, Dforcing.ny, Dforcing.now, Dforcing.before, Dforcing.after, Dforcing.val);
		//
		readforcingdata(totaltime, Dforcing);
		if (gpgpu)
		{
			// Allocate and bind textures
			AllocateTEX(Dforcing.nx, Dforcing.ny, Dforcing.GPU, Dforcing.now);
		}

	}
	else
	{
		//Error message
		log("Error while reading forcing map file: " + Dforcing.inputfile);
	}
}


/*! \fn  void readbathydata(int posdown, StaticForcingP<float> &Sforcing)
* special case of readstaticforcing(Sforcing);
* where the data 
*/
void readbathydata(int posdown, StaticForcingP<float> &Sforcing)
{
	readstaticforcing(Sforcing);

	if (posdown == 1)
	{
		
		log("Bathy data is positive down...correcting");
		for (int j = 0; j < Sforcing.ny; j++)
		{
			for (int i = 0; i < Sforcing.nx; i++)
			{
				Sforcing.val[i + j * Sforcing.nx] = Sforcing.val[i + j * Sforcing.nx] * -1.0f;
				//printf("%f\n", zb[i + (j)*nx]);

			}
		}
	}
}

/*! \fn  void readCRSfrombathy(std::string crs_ref, StaticForcingP<float> &Sforcing)
* Reading the CRS information from the bathymetry file (last one read);
*/
std::string readCRSfrombathy(std::string crs_ref, StaticForcingP<float>& Sforcing)
{
	int ncid, ncvarid, ncAttid, status;
	size_t t_len;
	char* crs;
	char* crs_wkt;
	std::string crs_ref2;
	

	if (!Sforcing.inputfile.empty())
	{

		log("Reading CRS information from forcing metadata (file: " + Sforcing.inputfile + ")");


		/* Open the netCDF file */
		status = nc_open(Sforcing.inputfile.c_str(), NC_NOWRITE, &ncid);
		if (status != NC_NOERR) handle_ncerror(status);

		/* Get variable ID */
		status = nc_inq_varid(ncid, Sforcing.varname.c_str(), &ncvarid);
		if (status != NC_NOERR) handle_ncerror(status);

		/* Get the attribute ID */
		status = nc_inq_attid(ncid, ncvarid, "grid_mapping", &ncAttid);
		if (status == NC_NOERR)
		{




			/* Read CRS attribute from the variable */
			status = nc_inq_attlen(ncid, ncvarid, "grid_mapping", &t_len);
			if (status != NC_NOERR) handle_ncerror(status);

			crs = (char*)malloc(t_len + 1);

			/* Read CRS attribute from the variable */
			status = nc_get_att_text(ncid, ncvarid, "grid_mapping", crs);
			if (status != NC_NOERR) handle_ncerror(status);

			printf("grid info detected: %s\n", crs);


			/*Get associated CRS variable ID*/
			status = nc_inq_varid(ncid, crs, &ncvarid);
			if (status != NC_NOERR) handle_ncerror(status);

			std::vector<std::string> attnamevec = { "crs_wkt","spatial_ref" };

			int idatt = -1;

			for (int id = 0; id < attnamevec.size(); id++)
			{
				/* Get the attribute ID */
				status = nc_inq_attid(ncid, ncvarid, attnamevec[id].c_str(), &ncAttid);
				if (status == NC_NOERR)
				{
					idatt = id;
					break;
				}
			}

			if (idatt > -1)
			{

				/* Get the attribute ID */
				status = nc_inq_attid(ncid, ncvarid, attnamevec[idatt].c_str(), &ncAttid);
				if (status != NC_NOERR) handle_ncerror(status);


				/* Read CRS attribute from the variable */
				status = nc_inq_attlen(ncid, ncvarid, attnamevec[idatt].c_str(), &t_len);
				if (status != NC_NOERR) handle_ncerror(status);

				crs_wkt = (char*)malloc(t_len + 1);

				/* Read CRS attribute from the variable */
				status = nc_get_att_text(ncid, ncvarid, attnamevec[idatt].c_str(), crs_wkt);
				if (status != NC_NOERR) handle_ncerror(status);

				printf("CRS_info: %s\n", crs_wkt);

				//crs_ref = crs_wkt;
				//crs_ref2 = crs_wkt;

				//printf("CRS_info: %s\n", crs_ref2.c_str());
			}
			else
			{
				printf("CRS_info detected but not understood reverting to default CRS\n Rename attribute in grid-mapping variable\n");

				//crs_ref = "";
			}

		}
		status = nc_close(ncid);
		/* Close the netCDF file */
		if ( status != NC_NOERR) {
			fprintf(stderr, "Error: Failed to close file.\n");
		}
	}
	return crs_wkt;
}

/*! \fn std::vector<SLTS> readbndfile(std::string filename,Param XParam, int side)
* Read boundary forcing files
* 
*/
std::vector<SLTS> readbndfile(std::string filename,Param XParam, int side)
{
	// read bnd or nest file
	// side is for deciding whether we are talking about a left(side=0) bot (side =1) right (side=2) or top (side=3)
	// Warning just made this up and need to have some sort of convention in the model
	std::string fileext;
	std::vector<std::string> extvec = split(filename, '.');

	std::vector<std::string> nameelements;

	std::vector<SLTS> Bndinfo;

	//
	//printf("%d\n", side);
	/*
	double xxmax;
	int hor;
	switch (side)
	{
		case 0://Left bnd
		{
			//xxo = XParam.yo;
			xxmax = XParam.ymax;
			//yy = XParam.xo;
			hor = 0;
			break;
		}
		case 1://Bot bnd
		{
			//xxo = XParam.xo;
			xxmax = XParam.xmax;
			//yy = XParam.yo;
			hor = 1;
			break;
		}
		case 2://Right bnd
		{
			//xxo = XParam.yo;
			xxmax = XParam.ymax;
			//yy = XParam.xmax;
			hor = 0;
			break;
		}
		case 3://Top bnd
		{
			//xxo = XParam.xo;
			xxmax = XParam.xmax;
			//yy = XParam.ymax;
			hor = 1;
			break;
		}
	}
	*/

	//printf("%f\t%f\t%f\n", xxo, xxmax, yy);

	nameelements = split(extvec.back(), '?');
	if (nameelements.size() > 1)
	{
		
		fileext = nameelements[0];
	}
	else
	{
		fileext = extvec.back();
	}

	if (fileext.compare("nc") == 0)
	{
		//Bndinfo = readNestfile(filename);
		//Bndinfo = readNestfile(filename, hor, XParam.eps, xxo, xxmax, yy);
	}
	else
	{
		Bndinfo = readWLfile(filename);
	}

	// Add zsoffset
	for (int i = 0; i < Bndinfo.size(); i++)
	{
		for (int n = 0; n < Bndinfo[i].wlevs.size(); n++)
		{
			double addoffset = std::isnan(XParam.zsoffset) ? 0.0 : XParam.zsoffset;
			Bndinfo[i].wlevs[n] = Bndinfo[i].wlevs[n] + addoffset;
		}
	}


	return Bndinfo;
}


/*! \fn std::vector<SLTS> readWLfile(std::string WLfilename)
* Read boundary water level data
*
*/
std::vector<SLTS> readWLfile(std::string WLfilename)
{
	std::vector<SLTS> slbnd;

	std::ifstream fs(WLfilename);

	if (fs.fail()) {
		//std::cerr << WLfilename << " Water level bnd file could not be opened" << std::endl;
		log("ERROR: Water level bnd file could not be opened : " + WLfilename);
		exit(1);
	}

	std::string line;
	std::vector<std::string> lineelements;
	std::vector<double> WLS;
	SLTS slbndline;
	while (std::getline(fs, line))
	{
		//std::cout << line << std::endl;

		// skip empty lines and lines starting with #
		if (!line.empty() && line.substr(0, 1).compare("#") != 0)
		{
			//Data should be in the format : time,Water level 1,Water level 2,...Water level n
			//Location where the water level is 0:ny/(nwl-1):ny where nwl i the number of wlevnodes

			//by default we expect tab delimitation
			lineelements = split(line, '\t');
			if (lineelements.size() < 2)
			{
				// Is it space delimited?
				lineelements.clear();
				lineelements = split(line, ' ');
			}

			if (lineelements.size() < 2)
			{
				//Well it has to be comma delimited then
				lineelements.clear();
				lineelements = split(line, ',');
			}
			if (lineelements.size() < 2)
			{
				// Giving up now! Could not read the files
				//issue a warning and exit
				//std::cerr << WLfilename << "ERROR Water level bnd file format error. only " << lineelements.size() << " where at least 2 were expected. Exiting." << std::endl;
				log("ERROR:  Water level bnd file (" + WLfilename + ") format error. only " + std::to_string(lineelements.size()) + " where at least 2 were expected. Exiting.");
				log(line);
				exit(1);
			}


			slbndline.time = std::stod(lineelements[0]);

			for (int n = 1; n < lineelements.size(); n++)
			{
				WLS.push_back(std::stod(lineelements[n]));
			}



			slbndline.wlevs = WLS;
			
			

			//slbndline = readBSHline(line);
			slbnd.push_back(slbndline);
			//std::cout << line << std::endl;
			WLS.clear();
		}

	}
	fs.close();

	//std::cout << slbnd[0].wlev << std::endl;


	return slbnd;
}


/*! \fn std::vector<SLTS> readNestfile(std::string ncfile,std::string varname, int hor ,double eps, double bndxo, double bndxmax, double bndy)
* Read boundary Nesting data
*
*/
std::vector<SLTS> readNestfile(std::string ncfile,std::string varname, int hor ,double eps, double bndxo, double bndxmax, double bndy)
{
	// Prep boundary input vector from anorthe model output file
	//this function works for botom top bnd as written but flips x and y for left and right bnds
	// hor controls wheter the boundary is a top/botom bnd hor=1 or left/right hor=0 
	std::vector<SLTS> slbnd;
	SLTS slbndline;
	
	std::vector<double> WLS,Unest,Vnest;
	//Define NC file variables
	int nnx, nny, nt, nbndpts, indxx, indyy, indx, indy,nx, ny;
	double dx, xxo, yyo, to, xmax, ymax, tmax,xo,yo;
	double * ttt, *zsa;
	bool checkhh = false;
	int iswet;
	bool flipx = false;
	bool flipy = false;

	// Read NC info // 
	//readgridncsize(ncfile,varname, nnx, nny, nt, dx, xxo, yyo, to, xmax, ymax, tmax, flipx, flipy);
	
	if (hor == 0)
	{
		nx = nny;
		ny = nnx;
		xo = yyo;
		yo = xxo;

	}
	else
	{
		nx = nnx;
		ny = nny;
		xo = xxo;
		yo = yyo;
	}

	// Read time vector
	ttt=(double *)malloc(nt*sizeof(double));
	zsa = (double *)malloc(1*sizeof(double));
	readnctime(ncfile, ttt);


	

	nbndpts = (int)((bndxmax - bndxo) / dx)+1;

	//printf("%f\t%f\t%f\t%f\n", bndxmax, bndxo, xo, yo);
	//printf("%f\t%f\t%f\t%f\n", bndxmax, bndxo, xxo, yyo);
	//printf("%f\t%d\t%d\t%f\n", bndy, nx, ny, dx);

	//printf("%d\n", nbndpts);
	std::string ncfilestr;
	std::string varstr,varstruu,varstrvv;


	//char ncfile[]="ocean_ausnwsrstwq2.nc";
	std::vector<std::string> nameelements;
	nameelements = split(ncfile, '?');
	if (nameelements.size() > 1)
	{

		ncfilestr = nameelements[0];
		varstr = nameelements[1];
	}
	else
	{

		ncfilestr = ncfile;
		varstr = "zs";
		checkhh = true;
	}


	for (int it = 0; it < nt; it++)
	{
		slbndline.time = ttt[it];
		for (int ibnd = 0; ibnd < nbndpts; ibnd++)
		{
			//
			// Read// interpolate data for each bnds
			indxx = utils::max(utils::min((int)((bndxo+(dx*ibnd) - xo) / dx), nx - 1), 0);
			indyy = utils::max(utils::min((int)((bndy - yo) / dx), ny - 1), 0);

			if (hor == 0)
			{
				indy = indxx;
				indx = indyy;
			}
			else
			{
				indx = indxx;
				indy = indyy;
			}

			iswet=readncslev1(ncfile, varstr, indx, indy, it, checkhh,eps, zsa);
			//varstr
			//printf("%d\t%d\t%d\tzs=%f\t%d\n", it,indx, indy, zsa[0],iswet);

			if (iswet == 0)
			{
				if (WLS.size() >= 1)
				{
					zsa[0] = WLS.back();
				}
				else
				{
					zsa[0] = 0.0;
				}
			}

			WLS.push_back(zsa[0]);

			//printf("zs=%f\\n", zsa[0]);

			// If true nesting then uu and vv are expected to be present in the netcdf file 

			if (checkhh)
			{
				varstruu = "uu";
				iswet = readncslev1(ncfilestr, varstruu, indx, indy, it, checkhh, eps, zsa);
				//printf("%d\t%d\t%d\tuu=%f\t%d\n", it, indx, indy, zsa[0], iswet);
				//printf("%d\t%d\t%f\n", indx, indy, zsa[0]);

				if (iswet == 0)
				{

					if (Unest.size() >= 1)
					{
						zsa[0] = Unest.back();
					}
					else
					{
						zsa[0] = 0.0;
					}
				}

				Unest.push_back(zsa[0]);

				varstrvv = "vv";
				iswet = readncslev1(ncfile, varstrvv, indx, indy, it, checkhh, eps, zsa);
				//printf("%d\t%d\t%d\tvv=%f\t%d\n", it, indx, indy, zsa[0], iswet);
				//printf("%d\t%d\t%f\n", indx, indy, zsa[0]);

				if (iswet == 0)
				{
					if (Vnest.size() >= 1)
					{
						zsa[0] = Vnest.back();
					}
					else
					{
						zsa[0] = 0.0;
					}
				}

				Vnest.push_back(zsa[0]);
			}




		}
		slbndline.wlevs = WLS;
		WLS.clear();
		if (checkhh)
		{
			slbndline.uuvel = Unest;
			slbndline.vvvel = Vnest;
			Unest.clear();
			Vnest.clear();
		}

		slbnd.push_back(slbndline);
		//std::cout << line << std::endl;
		
	}
	///To Be continued
	
	free(ttt);
	free(zsa);
	return slbnd;
}

/*! \fn std::vector<Flowin> readFlowfile(std::string Flowfilename)
* Read flow data for river forcing
*
*/
std::vector<Flowin> readFlowfile(std::string Flowfilename, std::string refdate)
{
	std::vector<Flowin> slbnd;

	std::ifstream fs(Flowfilename);

	if (fs.fail()) {
		std::cerr << Flowfilename << " Flow file could not be opened" << std::endl;
		write_text_to_log_file("ERROR: Flow/River file could not be opened ");
		exit(1);
	}

	std::string line;
	std::vector<std::string> lineelements;
	//std::vector<double> WLS;
	Flowin slbndline;
	while (std::getline(fs, line))
	{
		//std::cout << line << std::endl;

		// skip empty lines and lines starting with #
		if (!line.empty() && line.substr(0, 1).compare("#") != 0)
		{
			//Data should be in the format : time,Water level 1,Water level 2,...Water level n
			//Location where the water level is 0:ny/(nwl-1):ny where nwl i the number of wlevnodes

			//by default we expect tab delimitation
			lineelements = split(line, '\t');
			if (lineelements.size() != 2)
			{
				// Is it space delimited?
				lineelements.clear();
				lineelements = split(line, ' ');
			}

			if (lineelements.size() != 2)
			{
				//Well it has to be comma delimited then
				lineelements.clear();
				lineelements = split(line, ',');
			}
			if (lineelements.size() != 2)
			{
				// Giving up now! Could not read the files
				//issue a warning and exit
				//std::cerr << Flowfilename << "ERROR flow file format error. only " << lineelements.size() << " where at least 2 were expected. Exiting." << std::endl;
				log("ERROR:  flow file (" + Flowfilename + ") format error. only " + std::to_string(lineelements.size()) + " where at least 2 were expected. Exiting.");
				log(line);
				exit(1);
			}

			slbndline.time = readinputtimetxt(lineelements[0], refdate);
			//slbndline.time = std::stod(lineelements[0]);

			



			slbndline.q = std::stod(lineelements[1]);;



			//slbndline = readBSHline(line);
			slbnd.push_back(slbndline);
			//std::cout << line << std::endl;
			//WLS.clear();
		}

	}
	fs.close();

	//std::cout << slbnd[0].wlev << std::endl;


	return slbnd;
}


/*! \fn std::vector<Windin> readINfileUNI(std::string filename)
* Read rain/atmpressure data for spatially uniform forcing
*
*/
std::vector<Windin> readINfileUNI(std::string filename, std::string refdate)
{
	std::vector<Windin> wndinput;

	std::ifstream fs(filename);

	if (fs.fail()) {
		//std::cerr << filename << "ERROR: Atm presssure / Rainfall file could not be opened" << std::endl;
		log("ERROR: Atm presssure / Rainfall file could not be opened : " + filename);
		exit(1);
	}

	std::string line;
	std::vector<std::string> lineelements;
	std::vector<double> WLS;
	Windin wndline;
	while (std::getline(fs, line))
	{
		// skip empty lines and lines starting with #
		if (!line.empty() && line.substr(0, 1).compare("#") != 0)
		{
			//Data should be in the format : time,wind speed, wind dir, uwind vwind
			//Location where the water level is 0:ny/(nwl-1):ny where nwl i the number of wlevnodes

			//by default we expect tab delimitation
			lineelements = split(line, '\t');
			if (lineelements.size() < 2)
			{
				// Is it space delimited?
				lineelements.clear();
				lineelements = split(line, ' ');
			}

			if (lineelements.size() < 2)
			{
				//Well it has to be comma delimited then
				lineelements.clear();
				lineelements = split(line, ',');
			}
			if (lineelements.size() < 2)
			{
				// Giving up now! Could not read the files
				//issue a warning and exit
				//std::cerr << filename << "ERROR Atm presssure / Rainfall  file format error. only " << lineelements.size() << " where at least 2 were expected. Exiting." << std::endl;
				log("ERROR:  Atm presssure / Rainfall file (" + filename + ") format error. only " + std::to_string(lineelements.size()) + " where at least 2 were expected. Exiting.");
				log(line);
				exit(1);
			}

			wndline.time = readinputtimetxt(lineelements[0], refdate);
			//wndline.time = std::stod(lineelements[0]);
			wndline.wspeed = std::stod(lineelements[1]);
			
			wndinput.push_back(wndline);
			

		}

	}
	fs.close();

	return wndinput;
}


/*! \fn std::vector<Windin> readWNDfileUNI(std::string filename, double grdalpha)
* Read wind data for spatially uniform forcing
*
*/
std::vector<Windin> readWNDfileUNI(std::string filename, std::string refdate, double grdalpha)
{
	// Warning grdapha is expected in radian here
	std::vector<Windin> wndinput;

	std::ifstream fs(filename);

	if (fs.fail()) {
		//std::cerr << filename << "ERROR: Wind file could not be opened" << std::endl;
		log("ERROR: Wind file could not be opened : "+ filename);
		exit(1);
	}

	std::string line;
	std::vector<std::string> lineelements;
	std::vector<double> WLS;
	Windin wndline;
	while (std::getline(fs, line))
	{
		//std::cout << line << std::endl;

		// skip empty lines and lines starting with #
		if (!line.empty() && line.substr(0, 1).compare("#") != 0)
		{
			//Data should be in the format : time,wind speed, wind dir, uwind vwind
			//Location where the water level is 0:ny/(nwl-1):ny where nwl i the number of wlevnodes

			//by default we expect tab delimitation
			lineelements = split(line, '\t');
			if (lineelements.size() < 3)
			{
				// Is it space delimited?
				lineelements.clear();
				lineelements = split(line, ' ');
			}

			if (lineelements.size() < 3)
			{
				//Well it has to be comma delimited then
				lineelements.clear();
				lineelements = split(line, ',');
			}
			if (lineelements.size() < 3)
			{
				// Giving up now! Could not read the files
				//issue a warning and exit
				//std::cerr << filename << "ERROR Wind  file format error. only " << lineelements.size() << " where at least 3 were expected. Exiting." << std::endl;
				log("ERROR:  Wind file (" + filename + ") format error. only " + std::to_string(lineelements.size()) + " where at least 3 were expected. Exiting.");
				log(line);
				exit(1);
			}

			wndline.time = readinputtimetxt(lineelements[0], refdate);
			//wndline.time = std::stod(lineelements[0]);
			if (lineelements.size() == 5)
			{
				// U and v are explicitelly stated
				wndline.wspeed = std::stod(lineelements[1]); // Actually his is a dummy 
				wndline.wdirection= std::stod(lineelements[2]); // Actually his is a dummy
				wndline.uwind = std::stod(lineelements[3]);
				wndline.vwind = std::stod(lineelements[4]);
			}
			else
			{
				// read speed and direction and directly convert to u and v
				wndline.wspeed = std::stod(lineelements[1]); // Actually his is a dummy 
				wndline.wdirection = std::stod(lineelements[2]);
				double theta = (1.5*pi - grdalpha) - wndline.wdirection*pi / 180;

				wndline.uwind = wndline.wspeed*cos(theta);
				wndline.vwind = wndline.wspeed*sin(theta);
			}
			//slbndline.wlevs = WLS;



			//slbndline = readBSHline(line);
			wndinput.push_back(wndline);
			//std::cout << line << std::endl;
			
		}

	}
	fs.close();

	//std::cout << slbnd[0].wlev << std::endl;


	return wndinput;
}



/*! \fn void read
* Read polygon
*
*/
Polygon readPolygon(std::string filename)
{
	Polygon poly, polyB;
	Vertex v;

	std::string line;
	std::vector<std::string> lineelements;

	std::ifstream fs(filename);

	if (fs.fail()) {
		//std::cerr << filename << "ERROR: Wind file could not be opened" << std::endl;
		log("ERROR: Polygon file could not be opened : " + filename);
		return poly;
	}
	
	while (std::getline(fs, line))
	{
		// skip empty lines and lines starting with #
		if (!line.empty() && line.substr(0, 1).compare("#") != 0)
		{
			//
			//line.substr(0, 1).compare(">") != 0
			//by default we expect tab delimitation
			lineelements = DelimLine(line, 2);
			v.x = std::stod(lineelements[0]);
			v.y = std::stod(lineelements[1]);

			poly.vertices.push_back(v);

		}
	}


	size_t nv = poly.vertices.size();

	// Make sure ploygon is closed
	double epsilon = std::numeric_limits<double>::epsilon() * 1000.0;

	if ( !(abs(poly.vertices[0].x - poly.vertices[nv - 1].x) < epsilon && abs(poly.vertices[0].y - poly.vertices[nv - 1].y) < epsilon) )
	{
		v.x = poly.vertices[0].x;
		v.y = poly.vertices[0].y;

		poly.vertices.push_back(v);
	}

	polyB = CounterCWPoly(poly);

	polyB.xmin = polyB.vertices[0].x;
	polyB.xmax = polyB.vertices[0].x;

	polyB.ymin = polyB.vertices[0].y;
	polyB.ymax = polyB.vertices[0].y;

	for (int i = 0; i < polyB.vertices.size(); i++)
	{
		polyB.xmin = utils::min(polyB.vertices[i].x, polyB.xmin);
		polyB.xmax = utils::max(polyB.vertices[i].x, polyB.xmax);
		polyB.ymin = utils::min(polyB.vertices[i].y, polyB.ymin);
		polyB.ymax = utils::max(polyB.vertices[i].y, polyB.ymax);
	}


	return polyB;
}


std::vector<std::string> DelimLine(std::string line, int n, char delim)
{
	std::vector<std::string> lineelements;
	lineelements = split(line, delim);
	if (lineelements.size() != n)
	{
		lineelements.clear();


	}
	return lineelements;
}

std::vector<std::string> DelimLine(std::string line, int n)
{
	std::vector<std::string> LeTab;
	std::vector<std::string> LeSpace;
	std::vector<std::string> LeComma;
	
	//std::vector<std::string> lineelements;

	LeTab = DelimLine(line, n, '\t');
	LeSpace = DelimLine(line, n, ' ');
	LeComma = DelimLine(line, n, ',');
	
	if (LeTab.size() == n)
	{
		return LeTab;
	}
	if (LeSpace.size() == n)
	{
		return LeSpace;
	}
	if (LeComma.size() == n)
	{
		return LeComma;
	}

	LeTab.clear();

	return LeTab;
		
}

/*! \fn void readforcingdata(int step,T forcing)
* Read static forcing data 
*
*/
template <class T>
void readforcingdata(int step,T forcing)
{
	// Check extension 
	std::string fileext;

	fileext = forcing.extension;
	//Now choose the right function to read the data

	if (fileext.compare("md") == 0)
	{
		readbathyMD(forcing.inputfile, forcing.val);
	}
	if (fileext.compare("nc") == 0)
	{
		readvardata(forcing.inputfile, forcing.varname,step, forcing.val, forcing.flipxx, forcing.flipyy);
	}
	if (fileext.compare("bot") == 0 || fileext.compare("dep") == 0)
	{
		readXBbathy(forcing.inputfile, forcing.nx, forcing.ny, forcing.val);
	}
	if (fileext.compare("asc") == 0)
	{
		//
		readbathyASCzb(forcing.inputfile, forcing.nx, forcing.ny, forcing.val);
	}

	//return 1;
}
template void readforcingdata<StaticForcingP<float>>(int step, StaticForcingP<float> forcing);
template void readforcingdata<deformmap<float>>(int step, deformmap<float> forcing);
template void readforcingdata<StaticForcingP<int>>(int step, StaticForcingP<int> forcing);
//template void readforcingdata<DynForcingP<float>>(int step, DynForcingP<float> forcing);

/*! \fn void readforcingdata(double totaltime, DynForcingP<float>& forcing)
* Read Dynamic forcing data
*
*/
void readforcingdata(double totaltime, DynForcingP<float>& forcing)
{
	int step = utils::min(utils::max((int)floor((totaltime - forcing.to) / forcing.dt), 0), forcing.nt - 2);
	readvardata(forcing.inputfile, forcing.varname, step, forcing.before, forcing.flipxx, forcing.flipyy);
	readvardata(forcing.inputfile, forcing.varname, step+1, forcing.after, forcing.flipxx, forcing.flipyy);

	denan(forcing.nx, forcing.ny, float(forcing.denanval), forcing.before);
	denan(forcing.nx, forcing.ny, float(forcing.denanval), forcing.after);
	
	clampedges(forcing.nx, forcing.ny, forcing.clampedge, forcing.before);
	clampedges(forcing.nx, forcing.ny, forcing.clampedge, forcing.after);

	InterpstepCPU(forcing.nx, forcing.ny, step, totaltime, forcing.dt, forcing.now, forcing.before, forcing.after);
	forcing.val = forcing.now;
}

/*! \fn DynForcingP<float> readforcinghead(DynForcingP<float> Fmap, Param XParam)
* Read Dynamic forcing meta/header data
*
*/
DynForcingP<float> readforcinghead(DynForcingP<float> Fmap, Param XParam)
{
	// Read critical parameter for the forcing map
	log("Forcing map was specified. Checking file... ");
	std::string fileext = Fmap.extension;
	//double dummy;
	

	if (fileext.compare("nc") == 0)
	{
		log("Reading Forcing file as netcdf file");
		//readgridncsize(Fmap.inputfile,Fmap.varname, Fmap.nx, Fmap.ny, Fmap.nt, Fmap.dx, Fmap.xo, Fmap.yo, Fmap.to, Fmap.xmax, Fmap.ymax, Fmap.tmax, Fmap.flipxx, Fmap.flipyy);
		readgridncsize(Fmap, XParam);
		
	}
	else
	{
		log("Forcing file needs to be a .nc file you also need to specify the netcdf variable name like this ncfile.nc?myvar");
	}


	return Fmap;
}



/*! \fn T readforcinghead(T ForcingParam)
* Read Static forcing meta/header data
*
*/
template<class T> T readforcinghead(T ForcingParam)
{
	//std::string fileext;

	//read bathy and perform sanity check

	if (!ForcingParam.inputfile.empty())
	{
		//printf("bathy: %s\n", BathyParam.inputfile.c_str());

		log("Reading forcing metadata. file: " + ForcingParam.inputfile + " extension: " + ForcingParam.extension);

		

		
		if (ForcingParam.extension.compare("md") == 0)
		{
			//log("'md' file");
			readbathyHeadMD(ForcingParam.inputfile, ForcingParam.nx, ForcingParam.ny, ForcingParam.dx, ForcingParam.grdalpha);
			ForcingParam.xo = 0.0;
			ForcingParam.yo = 0.0;
			ForcingParam.xmax = ForcingParam.xo + (double(ForcingParam.nx) - 1.0) * ForcingParam.dx;
			ForcingParam.ymax = ForcingParam.yo + (double(ForcingParam.ny) - 1.0) * ForcingParam.dx;

		}
		if (ForcingParam.extension.compare("nc") == 0)
		{
			int dummy;
			double dummyb, dummyc;
			//log("netcdf file");
			//readgridncsize(ForcingParam.inputfile, ForcingParam.varname, ForcingParam.nx, ForcingParam.ny, dummy, ForcingParam.dx, ForcingParam.xo, ForcingParam.yo, dummyb, ForcingParam.xmax, ForcingParam.ymax, dummyc, ForcingParam.flipxx, ForcingParam.flipyy);
			readgridncsize(ForcingParam);
			//log("For nc of bathy file please specify grdalpha in the BG_param.txt (default 0)");

			//Check that the x and y variable are in crescent order:
			if (ForcingParam.xmax < ForcingParam.xo)
			{
				log("FATAL ERROR:  x coordinate isn't in crescent order in file: " + ForcingParam.inputfile);
				exit(1);
			}
			if (ForcingParam.ymax < ForcingParam.yo)
			{
				log("FATAL ERROR:  y coordinate isn't in crescent order in file: " + ForcingParam.inputfile);
				exit(1);
			}

		}
		if (ForcingParam.extension.compare("dep") == 0 || ForcingParam.extension.compare("bot") == 0)
		{
			//XBeach style file
			log(ForcingParam.extension + " file");
			log("For this type of bathy file please specify nx, ny, dx, xo, yo and grdalpha in the XBG_param.txt");
		}
		if (ForcingParam.extension.compare("asc") == 0)
		{
			//
			//log("asc file");
			readbathyASCHead(ForcingParam.inputfile, ForcingParam.nx, ForcingParam.ny, ForcingParam.dx, ForcingParam.xo, ForcingParam.yo, ForcingParam.grdalpha);
			ForcingParam.xmax = ForcingParam.xo + (ForcingParam.nx-1)* ForcingParam.dx;
			ForcingParam.ymax = ForcingParam.yo + (ForcingParam.ny-1)* ForcingParam.dx;
			log("For asc of bathy file please specify grdalpha in the BG_param.txt (default 0)");
		}

		

		//XParam.nx = ceil(XParam.nx / 16) * 16;
		//XParam.ny = ceil(XParam.ny / 16) * 16;



		//printf("Bathymetry grid info: nx=%d\tny=%d\tdx=%lf\talpha=%f\txo=%lf\tyo=%lf\txmax=%lf\tymax=%lf\n", BathyParam.nx, BathyParam.ny, BathyParam.dx, BathyParam.grdalpha * 180.0 / pi, BathyParam.xo, BathyParam.yo, BathyParam.xmax, BathyParam.ymax);
		log("Forcing grid info: nx=" + std::to_string(ForcingParam.nx) + " ny=" + std::to_string(ForcingParam.ny) + " dx=" + std::to_string(ForcingParam.dx) + " grdalpha=" + std::to_string(ForcingParam.grdalpha*180.0 / pi) + " xo=" + std::to_string(ForcingParam.xo) + " xmax=" + std::to_string(ForcingParam.xmax) + " yo=" + std::to_string(ForcingParam.yo) + " ymax=" + std::to_string(ForcingParam.ymax));






	}
	else
	{
		std::cerr << "Fatal error: No bathymetry file specified. Please specify using 'bathy = Filename.bot'" << std::endl;
		log("Fatal error : No bathymetry file specified. Please specify using 'bathy = Filename.md'");
		exit(1);
	}
	return ForcingParam;
}
template inputmap readforcinghead<inputmap>(inputmap BathyParam);
template forcingmap readforcinghead<forcingmap>(forcingmap BathyParam);
//template StaticForcingP<float> readBathyhead<StaticForcingP<float>>(StaticForcingP<float> BathyParam);
template StaticForcingP<float> readforcinghead<StaticForcingP<float>>(StaticForcingP<float> ForcingParam);


/*! \fn void readbathyHeadMD(std::string filename, int &nx, int &ny, double &dx, double &grdalpha)
* Read MD file header data
*
*/
void readbathyHeadMD(std::string filename, int &nx, int &ny, double &dx, double &grdalpha)
{
	

	std::ifstream fs(filename);

	if (fs.fail()) {
		std::cerr << filename << " bathy file (md file) could not be opened" << std::endl;
		log("ERROR: bathy file could not be opened "+ filename);
		exit(1);
	}

	std::string line;
	std::vector<std::string> lineelements;

	std::getline(fs, line);
	// skip empty lines
	if (!line.empty())
	{

		//by default we expect tab delimitation
		lineelements = split(line, '\t');
		if (lineelements.size() < 5)
		{
			// Is it space delimited?
			lineelements.clear();
			lineelements = split(line, ' ');
		}

		if (lineelements.size() < 5)
		{
			//Well it has to be comma delimited then
			lineelements.clear();
			lineelements = split(line, ',');
		}
		if (lineelements.size() < 5)
		{
			// Giving up now! Could not read the files
			//issue a warning and exit
			std::cerr << filename << "ERROR Wind bnd file format error. only " << lineelements.size() << " where 5 were expected. Exiting." << std::endl;
			log("ERROR:  Wind bnd file (" + filename + ") format error. only " + std::to_string(lineelements.size()) + " where 3 were expected. Exiting.");
			log(line);
			exit(1);
		}

		nx = std::stoi(lineelements[0]);
		ny = std::stoi(lineelements[1]);
		dx = std::stod(lineelements[2]);
		grdalpha = std::stod(lineelements[4]);
	}

	fs.close();
}


/*! \fn void readbathyMD(std::string filename, float*& zb)
* Read MD file data
*
*/
template <class T> void readbathyMD(std::string filename, T*& zb)
{
	// Shit that doesn'y wor... Needs fixing 
	int nx;
	//int ny;
	//float dx;
	std::ifstream fs(filename);

	if (fs.fail()) {
		std::cerr << filename << " bathy file (md file) could not be opened" << std::endl;
		log("ERROR: bathy file could not be opened " + filename);
		exit(1);
	}

	std::string line;

	std::vector<std::string> lineelements;

	std::getline(fs, line);
	if (!line.empty() && line.substr(0, 1).compare("#") != 0)
	{
		//by default we expect tab delimitation
		lineelements = split(line, '\t');
		if (lineelements.size() < 5)
		{
			// Is it space delimited?
			lineelements.clear();
			lineelements = split(line, ' ');
		}

		if (lineelements.size() < 5)
		{
			//Well it has to be comma delimited then
			lineelements.clear();
			lineelements = split(line, ',');
		}
		if (lineelements.size() < 5)
		{
			// Giving up now! Could not read the files
			//issue a warning and exit
			std::cerr << filename << "ERROR Wind bnd file format error. only " << lineelements.size() << " where 5 were expected. Exiting." << std::endl;
			log("ERROR:  Wind bnd file (" + filename + ") format error. only " + std::to_string(lineelements.size()) + " where 3 were expected. Exiting.");
			log(line);
			exit(1);
		}

		nx = std::stoi(lineelements[0]);
		//ny = std::stoi(lineelements[1]);
		//dx = std::stof(lineelements[2]);
		//grdalpha = std::stof(lineelements[4]);
	}

	int j = 0;
	while (std::getline(fs, line))
	{
		//std::cout << line << std::endl;

		// skip empty lines and lines starting with #
		if (!line.empty() && line.substr(0, 1).compare("#") != 0)
		{
			lineelements = split(line, '\t');
			for (int i = 0; i < nx; i++)
			{
				zb[i + j * nx] = T(std::stof(lineelements[0]));
			}
			j++;
		}
	}

	fs.close();

}
template void readbathyMD<int>(std::string filename, int*& zb);
template void readbathyMD<float>(std::string filename, float*& zb);

/*! \fn  void readXBbathy(std::string filename, int nx,int ny, float *&zb)
* Read XBeach style file data
*
*/
template <class T> void readXBbathy(std::string filename, int nx,int ny, T *&zb)
{
	//read input data:
	//printf("bathy: %s\n", filename);
	
	
	//read md file
	 std::ifstream fs(filename);
	 std::string line;
	 std::vector<std::string> lineelements;

	 
	

	
	//int jreadzs;
	for (int jnod = 0; jnod < ny; jnod++)
	{

		std::getline(fs, line);

		for (int inod = 0; inod < nx; inod++)
		{
			//fscanf(fid, "%f", &zb[inod + (jnod)*nx]);
			zb[inod + jnod * nx] = T(std::stod(lineelements[0]));

		}
	}
	fs.close();
	//fclose(fid);
}
template void readXBbathy<int>(std::string filename, int nx, int ny, int*& zb);
template void readXBbathy<float>(std::string filename, int nx, int ny, float*& zb);


 /*! \fn  void readbathyASCHead(std::string filename, int &nx, int &ny, double &dx, double &xo, double &yo, double &grdalpha)
 * Read ASC file meta/header data
 *
 */
void readbathyASCHead(std::string filename, int &nx, int &ny, double &dx, double &xo, double &yo, double &grdalpha)
{
	std::ifstream fs(filename);

	if (fs.fail()) {
		std::cerr << filename << " bathy file (md file) could not be opened" << std::endl;
		log("ERROR: bathy file could not be opened " + filename);
		exit(1);
	}

	std::string line,left,right;
	std::vector<std::string> lineelements;
	//std::size_t found;
	//std::getline(fs, line);
	int linehead = 0;

	bool pixelreg = true;

	while (linehead < 6)
	{
		std::getline(fs, line);
		// skip empty lines
		if (!line.empty())
		{

			//by default we expect tab delimitation
			lineelements = split(line, ' ');
			if (lineelements.size() < 2)
			{
				lineelements = split(line, '\t');

			}




			left = trim(lineelements[0], " ");
			right = lineelements[1]; 
			//printf("left: %s ;right: %s\n", left.c_str(), right.c_str());
			//found = left.compare("ncols");// it needs to strictly compare
			if (left.compare("ncols") == 0) // found the parameter
			{

				//
				nx = std::stoi(right);

			}

			if (left.compare("nrows") == 0) // found the parameter
			{

				//
				ny = std::stoi(right);

			}
			if (left.compare("cellsize") == 0) // found the parameter
			{

				//
				dx = std::stod(right);

			}
			if (left.compare("xllcenter") == 0) // found the parameter
			{

				//
				xo = std::stod(right);

			}
			if (left.compare("yllcenter") == 0) // found the parameter
			{
				pixelreg = false;
				//
				yo = std::stod(right);

			}
			//if gridnode registration this should happen
			if (left.compare("xllcorner") == 0) // found the parameter
			{
				pixelreg = false;
				//
				xo = std::stod(right);

			}
			if (left.compare("yllcorner") == 0) // found the parameter
			{

				//
				yo = std::stod(right);
				//This should be:
				//yo = std::stod(right) + dx / 2.0;
				//but by the time xo and yo are found dx has not been setup... awkward...

			}
			linehead++;
		}
	}

	if (!pixelreg)
	{
		xo = xo + 0.5 * dx;
		yo = yo + 0.5 * dx;
	}

	grdalpha = 0.0;
	fs.close();

}


/*! \fn void readbathyASCzb(std::string filename,int nx, int ny, float* &zb)
* Read ASC file data
*
*/
template <class T> void readbathyASCzb(std::string filename,int nx, int ny, T* &zb)
{
	//
	std::ifstream fs(filename);
	int linehead = 0;
	std::string line;
	if (fs.fail()) {
		std::cerr << filename << " bathy file (md file) could not be opened" << std::endl;
		log("ERROR: bathy file could not be opened " + filename);
		exit(1);
	}
	while (linehead < 6)
	{
		//Skip header
		std::getline(fs, line);
		linehead++;
	}
	//int jreadzs;
	for (int jnod = ny-1; jnod >= 0; jnod--)
	{



		for (int inod = 0; inod < nx; inod++)
		{
			//fscanf(fid, "%f", &zb[inod + (jnod)*nx]);

			fs >> zb[inod + (jnod)*nx];
			//printf("%f\n", zb[inod + (jnod)*nx]);

		}
	}

	fs.close();
}
template void readbathyASCzb<int>(std::string filename, int nx, int ny, int*& zb);
template void readbathyASCzb<float>(std::string filename, int nx, int ny, float*& zb);

template <class T> void clampedges(int nx, int ny, T clamp, T* z)
{
	//
	int ii;
	for (int ix = 0; ix <nx; ix++)
	{
		ii = ix + 0 * nx;
		z[ii] = clamp;
		ii = ix + (ny - 1) * nx;
		z[ii] = clamp;
	}

	for (int iy = 0; iy < ny; iy++)
	{
		ii = 0 + iy * nx;
		z[ii] = clamp;
		ii = (nx - 1) + (iy)* nx;
		z[ii] = clamp;
	}
}

template <class T> void denan(int nx, int ny, float denanval, T* z)
{
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			if (isnan(z[i + j * nx]))
			{
				z[i + j * nx] = denanval;
			}
		}
	}
}
template void denan<float>(int nx, int ny, float denanval, float* z);
template void denan<double>(int nx, int ny, float denanval, double* z);

void denan(int nx, int ny, float denanval, int* z)
{
	//don't do nothing
	// This function exist for cleaner compiling requirement that NaN do not exist in int form
}

/*! \fn void InterpstepCPU(int nx, int ny, int hdstep, float totaltime, float hddt, float*& Ux, float* Uo, float* Un)
* linearly interpolate between 2 cartesian arrays (of the same size)
* This is used to interpolate dynamic forcing to a current time step 
*
*/
//template <class T> void InterpstepCPU(int nx, int ny, int hdstep, float totaltime, float hddt, T*& Ux, T* Uo, T* Un)
//{
//	//float fac = 1.0;
//	T Uxo, Uxn;
//
//	/*Ums[tx]=Umask[ix];*/
//
//
//
//
//	for (int i = 0; i < nx; i++)
//	{
//		for (int j = 0; j < ny; j++)
//		{
//			Uxo = Uo[i + nx * j];
//			Uxn = Un[i + nx * j];
//
//			Ux[i + nx * j] = Uxo + (totaltime - hddt * hdstep) * (Uxn - Uxo) / hddt;
//		}
//	}
//}
//template void InterpstepCPU<int>(int nx, int ny, int hdstep, float totaltime, float hddt, int*& Ux, int* Uo, int* Un);
//template void InterpstepCPU<float>(int nx, int ny, int hdstep, float totaltime, float hddt, float*& Ux, float* Uo, float* Un);




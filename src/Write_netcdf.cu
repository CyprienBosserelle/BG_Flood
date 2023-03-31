#include "Write_netcdf.h"
#include "Util_CPU.h"
#include "General.h"

void handle_ncerror(int status) {


	if (status != NC_NOERR) {
		fprintf(stderr, "Netcdf %s\n", nc_strerror(status));
		std::ostringstream stringStream;
		stringStream << nc_strerror(status);
		std::string copyOfStr = stringStream.str();
		log("Netcdf error:" + copyOfStr);
		//fprintf(logfile, "Netcdf: %s\n", nc_strerror(status));
		exit(2);
	}
}

void Calcnxny(Param XParam, int level, int& nx, int& ny)
{
	double ddx = calcres(XParam.dx, level);
	double dxp = calcres(XParam.dx, level + 1);
	double xxmax, xxmin, yymax, yymin;

	xxmax = XParam.xmax - dxp;
	yymax = XParam.ymax - dxp;

	xxmin = XParam.xo + dxp;
	yymin = XParam.yo + dxp;

	nx = round((xxmax - xxmin) / ddx + 1.0);
	ny = round((yymax - yymin) / ddx + 1.0);
}

void Calcnxnyzone(Param XParam, int level, int& nx, int& ny, outzoneB Xzone)
{
	double ddx = calcres(XParam.dx, level);
	double xxmax, xxmin, yymax, yymin;

	xxmax = Xzone.xmax;
	yymax = Xzone.ymax;

	xxmin = Xzone.xo;
	yymin = Xzone.yo;

	nx = ftoi((xxmax - xxmin) / ddx);
	ny = ftoi((yymax - yymin) / ddx);
}

std::vector<int> Calcactiveblockzone(Param XParam, int* activeblk, outzoneB Xzone)
{
	std::vector<int> actblkzone(Xzone.nblk, -1);
	int * inactive, * inblock;

	for (int ib = 0; ib < Xzone.nblk; ib++)
	{
		//printf("loop=%i \n", Xzone.blk[ib]);
		inactive = std::find (activeblk, activeblk + XParam.nblk, Xzone.blk[ib]);
		inblock = std::find (Xzone.blk, Xzone.blk + Xzone.nblk, Xzone.blk[ib]);
		//if ((inactive != activeblk + XParam.nblk) && (inblock != Xzone.blk + Xzone.nblk))
		if (inactive != activeblk + XParam.nblk)
		{
			//printf("active=%i \n", Xzone.blk[ib]);
			if (inblock != Xzone.blk + Xzone.nblk)
			{
				actblkzone[ib] = Xzone.blk[ib];
				//printf("block=%i \n", Xzone.blk[ib]);
			}
			else { actblkzone[ib] = -1; }
		}
		else
		{
			actblkzone[ib] = -1;
		}
	}
	return(actblkzone);
}

template<class T>
void creatncfileBUQ(Param &XParam,int * activeblk, int * level, T * blockxo, T * blockyo, outzoneB &Xzone)
{

	int status;
	int nx, ny;
	//double dx = XParam.dx;
	size_t nxx, nyy;
	int ncid, xx_dim, yy_dim, time_dim, blockid_dim, nblk;
	double * xval, *yval;

	//const int minlevzone = XParam.minlevel;
	//const int maxlevzone = XParam.maxlevel;

	std::vector<int> activeblkzone = Calcactiveblockzone(XParam, activeblk, Xzone);
	//Calclevelzone(XParam, minlevzone, maxlevzone, Xzone, level);
	nblk = Xzone.nblk;


	// create the netcdf dataset Xzone.outname.c_str()
	status = nc_create(Xzone.outname.c_str(), NC_NOCLOBBER|NC_NETCDF4, &ncid);
	if (status != NC_NOERR)
	{
		if (status == NC_EEXIST) // File already exist so automatically rename the output file 
		{
			//printf("Warning! Output file name already exist  ");
			log("Warning! Output file name already exist   ");
			int fileinc = 1;
			std::vector<std::string> extvec = split(Xzone.outname, '.');
			std::string bathyext = extvec.back();
			std::string newname;

			while (status == NC_EEXIST)
			{
				newname = extvec[0];
				for (int nstin = 1; nstin < extvec.size() - 1; nstin++)
				{
					// This is in case there are "." in the file name that do not relate to the file extension"
					newname = newname + "." + extvec[nstin];
				}
				newname = newname + "_" + std::to_string(fileinc) + "." + bathyext;
				Xzone.outname = newname;
				status = nc_create(Xzone.outname.c_str(), NC_NOCLOBBER|NC_NETCDF4, &ncid);
				fileinc++;
			}
			//printf("New file name: %s  ", Xzone.outname.c_str());
			log("New file name: " + Xzone.outname);

		}
		else
		{
			// Other error
			handle_ncerror(status);
		}
	}

	// status could be a new error after renaming the file
	if (status != NC_NOERR) handle_ncerror(status);
	
	double initdx = calcres(XParam.dx, XParam.initlevel);
	double xmin, xmax, ymin, ymax;

	xmin = Xzone.xo ;
	xmax = Xzone.xmax ;
	ymin = Xzone.yo ;
	ymax = Xzone.ymax ;

	// Define global attributes
	status = nc_put_att_int(ncid, NC_GLOBAL, "maxlevel", NC_INT, 1, &Xzone.maxlevel);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_put_att_int(ncid, NC_GLOBAL, "minlevel", NC_INT, 1, &Xzone.minlevel);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_put_att_double(ncid, NC_GLOBAL, "xmin", NC_DOUBLE, 1, &xmin);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_put_att_double(ncid, NC_GLOBAL, "xmax", NC_DOUBLE, 1, &xmax);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_put_att_double(ncid, NC_GLOBAL, "ymin", NC_DOUBLE, 1, &ymin);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_put_att_double(ncid, NC_GLOBAL, "ymax", NC_DOUBLE, 1, &ymax);
	if (status != NC_NOERR) handle_ncerror(status);


	// Define time variable 
	status = nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dim);
	if (status != NC_NOERR) handle_ncerror(status);

	int time_id, xx_id, yy_id;
	int tdim[] = { time_dim };
	
	//########################
	//static size_t tst[] = { 0 };
	size_t blkstart[] = { 0 }; // Xzone.blk[0]};
	size_t blkcount[] = { (size_t) Xzone.nblk };
	size_t xcount[] = { 0 };
	size_t ycount[] = { 0 };
	static size_t xstart[] = { 0 }; // start at first value
	static size_t ystart[] = { 0 };
	status = nc_def_var(ncid, "time", NC_FLOAT, 1, tdim, &time_id);
	if (status != NC_NOERR) handle_ncerror(status);

	static char txtname[] = "time";;
	status = nc_put_att_text(ncid, time_id, "standard_name", strlen(txtname), txtname );
	//status = nc_put_att_string(ncid, time_id, "standard_name", 1, "time");
	//units = "days since 1990-1-1 0:0:0";

	std::string timestr= "seconds since " + XParam.reftime;
	const char* timeunit = timestr.c_str();

	status = nc_put_att_text(ncid, time_id, "units", strlen(timeunit), timeunit);


	if (status != NC_NOERR) handle_ncerror(status);


	int crsid;
	std::string crsname;
	status = nc_def_var(ncid, "crs", NC_INT, 0, tdim, &crsid);
	if (XParam.spherical == 1)
	{
		crsname = "latitude_longitude";

		float primemeridian = 0.0f;
		float sma = 6378137.0f;
		float iflat = 298.257223563f;
		status = nc_put_att_text(ncid, crsid, "grid_mapping_name", crsname.size(), crsname.c_str());
		status = nc_put_att_float(ncid, crsid, "longitude_of_prime_meridian", NC_FLOAT, 1, &primemeridian);
		status = nc_put_att_float(ncid, crsid, "semi_major_axis", NC_FLOAT, 1, &sma);
		status = nc_put_att_float(ncid, crsid, "inverse_flattening", NC_FLOAT, 1, &iflat);
	}
	else
	{
		crsname = "projected";
		std::string proj = "";
		status = nc_put_att_text(ncid, crsid, "grid_mapping_name", crsname.size(), crsname.c_str());
		status = nc_put_att_text(ncid, crsid, "proj4", proj.size(), proj.c_str());
		//status = nc_put_att_float(ncid, crsid, "semi_major_axis", NC_FLOAT, 1, 6378137.0);
		//status = nc_put_att_float(ncid, crsid, "inverse_flattening", NC_FLOAT, 1, 298.257223563);
	}

	if (status != NC_NOERR) handle_ncerror(status);

	// Define dimensions and variables to store block id, status, level xo, yo

	status = nc_def_dim(ncid, "blockid", nblk, &blockid_dim);
	if (status != NC_NOERR) handle_ncerror(status);

	int biddim[] = { blockid_dim };
	int blkid_id, blkxo_id, blkyo_id, blklevel_id, blkwidth_id, blkstatus_id;

	status = nc_def_var(ncid, "blockid", NC_INT, 1, biddim, &blkid_id);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_def_var(ncid, "blockxo", NC_FLOAT, 1, biddim, &blkxo_id);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_def_var(ncid, "blockyo", NC_FLOAT, 1, biddim, &blkyo_id);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_def_var(ncid, "blockwidth", NC_FLOAT, 1, biddim, &blkwidth_id);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_def_var(ncid, "blocklevel", NC_INT, 1, biddim, &blklevel_id);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_def_var(ncid, "blockstatus", NC_INT, 1, biddim, &blkstatus_id);
	if (status != NC_NOERR) handle_ncerror(status);

	//int* levZone;

	// For each level Define xx yy 
	for (int lev = Xzone.minlevel; lev <= Xzone.maxlevel; lev++)
	{
		
		Calcnxnyzone(XParam, lev, nx, ny, Xzone);

		//printf("lev=%d;  xxmin=%f; xxmax=%f; nx=%d\n", lev, xmin, xmax, nx);
		//printf("lev=%d;  yymin=%f; yymax=%f; ny=%d\n", lev,  ymin, ymax, ny);

		//to change type from int to size_t
		nxx = nx;
		nyy = ny;

		//Define dimensions: Name and length
		std::string xxname, yyname, sign;

		lev < 0?sign="N":sign = "P";


		xxname = "xx_" + sign + std::to_string(abs(lev));
		yyname = "yy_" + sign + std::to_string(abs(lev));

		//printf("lev=%d; xxname=%s; yyname=%s;\n", lev, xxname.c_str(), yyname.c_str());
		//printf("ddx=%f; nxx=%d;\n", ddx, nxx);
		status = nc_def_dim(ncid, xxname.c_str(), nxx, &xx_dim);
		if (status != NC_NOERR) handle_ncerror(status);
		status = nc_def_dim(ncid, yyname.c_str(), nyy, &yy_dim);
		if (status != NC_NOERR) handle_ncerror(status);
		//status = nc_def_dim(ncid, "npart",nnpart,&p_dim);

		int xdim[] = { xx_dim };
		int ydim[] = { yy_dim };



		status = nc_def_var(ncid, xxname.c_str(), NC_DOUBLE, 1, xdim, &xx_id);
		if (status != NC_NOERR) handle_ncerror(status);
		status = nc_def_var(ncid, yyname.c_str(), NC_DOUBLE, 1, ydim, &yy_id);
		if (status != NC_NOERR) handle_ncerror(status);
		//End definitions: leave define mode
	}
	status = nc_enddef(ncid);
	if (status != NC_NOERR) handle_ncerror(status);

	//status = nc_close(ncid);
	//if (status != NC_NOERR) handle_ncerror(status);

	float* blkwidth;
	int* blkid;
	int ibl;


	AllocateCPU(1, nblk, blkwidth);
	AllocateCPU(1, nblk, blkid);

	printf("blockId:\n");
	for (int ib = 0; ib < nblk; ib++)
	{
		//int ibl = activeblk[Xzone.blk[ib]];
		ibl = activeblkzone[ib];
		blkwidth[ib] = (float)calcres(XParam.dx, level[ibl]);
		blkid[ib] = ibl;
	}
	

	status = nc_put_vara_int(ncid, blkid_id, blkstart, blkcount, blkid);
	if (status != NC_NOERR) handle_ncerror(status);

	//status = nc_put_vara_int(ncid, blkstatus_id, blkstart, blkcount, activeblk);
	status = nc_put_vara_float(ncid, blkwidth_id, blkstart, blkcount, blkwidth);
	if (status != NC_NOERR) handle_ncerror(status);


	// Reusing blkwidth/blkid for other array (for blkxo/blklevel and blkyo) to save memory space

	// This is needed because the blockxo array may be shuffled to memory block beyond nblk
	for (int ib = 0; ib < nblk; ib++)
	{
		//int ibl = activeblk[Xzone.blk[ib]];
		ibl = activeblkzone[ib];
		blkwidth[ib] = float(T(XParam.xo) + blockxo[ibl]);
		blkid[ib] = level[ibl];
		
	}

	status = nc_put_vara_float(ncid, blkxo_id, blkstart, blkcount, blkwidth);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_put_vara_int(ncid, blklevel_id, blkstart, blkcount, blkid);

	for (int ib = 0; ib < nblk; ib++)
	{
		//int ibl = activeblk[Xzone.blk[ib]];
		ibl = activeblkzone[ib];
		blkwidth[ib] = float(T(XParam.yo) + blockyo[ibl]);
	}

	status = nc_put_vara_float(ncid, blkyo_id, blkstart, blkcount, blkwidth);
	

	free(blkid);
	free(blkwidth);

	if (status != NC_NOERR) handle_ncerror(status);
	
	std::string xxname, yyname, sign;

for (int lev = Xzone.minlevel; lev <= Xzone.maxlevel; lev++)
{
	Calcnxnyzone(XParam, lev, nx, ny, Xzone);

	// start at first value
	//static size_t thstart[] = { 0 };
	xcount[0] = nx;
	ycount[0] = ny;
	//Recreat the x, y
	xval = (double*)malloc(nx * sizeof(double));
	yval = (double*)malloc(ny * sizeof(double));

	double ddx = calcres(XParam.dx, lev);
	double dxp = calcres(XParam.dx, lev + 1);
	double xxmax, xxmin, yymax, yymin;

	xxmax = Xzone.xmax - dxp;
	yymax = Xzone.ymax - dxp;

	xxmin = Xzone.xo + dxp;
	yymin = Xzone.yo + dxp;

	for (int i = 0; i < nx; i++)
	{
		xval[i] = xxmin + double(i) * ddx;
	}

	for (int i = 0; i < ny; i++)
	{
		yval[i] = yymin + double(i) * ddx;
	}


	lev < 0 ? sign = "N" : sign = "P";


	xxname = "xx_" + sign + std::to_string(abs(lev));
	yyname = "yy_" + sign + std::to_string(abs(lev));

	//printf("lev=%d; xxname=%s; yyname=%s;\n", lev, xxname.c_str(), yyname.c_str());

	status = nc_inq_varid(ncid, xxname.c_str(), &xx_id);
	if (status != NC_NOERR) handle_ncerror(status);
	status = nc_inq_varid(ncid, yyname.c_str(), &yy_id);
	if (status != NC_NOERR) handle_ncerror(status);

	//Provide values for variables

	status = nc_put_vara_double(ncid, xx_id, xstart, xcount, xval);
	if (status != NC_NOERR) handle_ncerror(status);
	status = nc_put_vara_double(ncid, yy_id, ystart, ycount, yval);
	if (status != NC_NOERR) handle_ncerror(status);

	free(xval);
	free(yval);
}

//close and save new file
status = nc_close(ncid);
if (status != NC_NOERR) handle_ncerror(status);




//return XParam;void
}

template void creatncfileBUQ<float>(Param& XParam, int* activeblk, int* level, float* blockxo, float* blockyo, outzoneB& Xzone);
template void creatncfileBUQ<double>(Param& XParam, int* activeblk, int* level, double* blockxo, double* blockyo, outzoneB& Xzone);


template<class T>
void creatncfileBUQ(Param& XParam, BlockP<T> &XBlock)
{
	for (int o = 0; o < XBlock.outZone.size(); o++)
	{
		creatncfileBUQ(XParam, XBlock.active, XBlock.level, XBlock.xo, XBlock.yo, XBlock.outZone[o]);
	}
}
template void creatncfileBUQ<float>(Param &XParam, BlockP<float> &XBlock);
template void creatncfileBUQ<double>(Param &XParam, BlockP<double> &XBlock);

template <class T> void defncvarBUQ(Param XParam, int* activeblk, int* level, T* blockxo, T* blockyo, std::string varst, int vdim, T* var, outzoneB Xzone)
{
	defncvarBUQ(XParam, activeblk, level, blockxo, blockyo, varst, "", "","", vdim, var, Xzone);
}

template <class T> void defncvarBUQ(Param XParam, int* activeblk, int* level, T* blockxo, T* blockyo, std::string varst, std::string longname, std::string stdname, std::string unit, int vdim, T* var, outzoneB Xzone)
{

	int smallnc = XParam.smallnc;
	float scalefactor = XParam.scalefactor;
	float addoffset = XParam.addoffset;
	//int nx = ceil(XParam.nx / 16.0) * 16.0;
	//int ny = ceil(XParam.ny / 16.0) * 16.0;
	int status;
	int ncid, var_id;
	int  var_dimid2D[2];
	int  var_dimid3D[3];
	//int  var_dimid4D[4];

	short* varblk_s;
	float* varblk;
	int recid, xid, yid;
	int bl, ibl, lev;
	//size_t ntheta;// nx and ny are stored in XParam not yet for ntheta

	float fillval = 9.9692e+36f;
	short fillval_s = (short)round((9.9692e+36f - addoffset) / scalefactor);
	//short Sfillval = 32767;
	//short fillval = 32767
	static size_t start2D[] = { 0, 0 }; // start at first value 
	//static size_t count2D[] = { ny, nx };
	static size_t count2D[] = { XParam.blkwidth, XParam.blkwidth };

	static size_t start3D[] = { 0, 0, 0 }; // start at first value 
	//static size_t count3D[] = { 1, ny, nx };
	static size_t count3D[] = { 1, XParam.blkwidth, XParam.blkwidth };
	//size_t count3D[3];
	//count3D[0] = 1;
	//count3D[1] = XParam.blkwidth;
	//count3D[2] = XParam.blkwidth;
	
	//int minlevzone, maxlevzone;

	std::string outfile = Xzone.outname;
	std::vector<int> activeblkzone = Calcactiveblockzone(XParam, activeblk, Xzone);
	//Calclevelzone(XParam, minlevzone, maxlevzone, Xzone, level);


	nc_type VarTYPE;

	if (smallnc > 0)
	{
		VarTYPE = NC_SHORT;
	}
	else
	{
		VarTYPE = NC_FLOAT;
	}

	//printf("\n ib=%d count3D=[%d,%d,%d]\n", count3D[0], count3D[1], count3D[2]);


	status = nc_open(outfile.c_str(), NC_WRITE, &ncid);
	if (status != NC_NOERR) handle_ncerror(status);
	status = nc_redef(ncid);
	if (status != NC_NOERR) handle_ncerror(status);
	//Inquire dimensions ids
	status = nc_inq_unlimdim(ncid, &recid);//time
	if (status != NC_NOERR) handle_ncerror(status);

	varblk = (float *)malloc(XParam.blkwidth* XParam.blkwidth * sizeof(float));
	if (smallnc > 0)
	{

		varblk_s = (short *)malloc(XParam.blkwidth * XParam.blkwidth * sizeof(short));
	}


	std::string xxname, yyname, varname,sign;

	//generate a different variable name for each level and add attribute as necessary
	for (lev = Xzone.minlevel; lev <= Xzone.maxlevel; lev++)
	{

		//std::string xxname, yyname, sign;

		lev < 0 ? sign = "N" : sign = "P";


		xxname = "xx_" + sign + std::to_string(abs(lev));
		yyname = "yy_" + sign + std::to_string(abs(lev));

		varname = varst + "_" + sign + std::to_string(abs(lev));


		//printf("lev=%d; xxname=%s; yyname=%s;\n", lev, xxname.c_str(), yyname.c_str());
		status = nc_inq_dimid(ncid, xxname.c_str(), &xid);
		if (status != NC_NOERR) handle_ncerror(status);
		status = nc_inq_dimid(ncid, yyname.c_str(), &yid);
		if (status != NC_NOERR) handle_ncerror(status);


		var_dimid2D[0] = yid;
		var_dimid2D[1] = xid;

		var_dimid3D[0] = recid;
		var_dimid3D[1] = yid;
		var_dimid3D[2] = xid;

		if (vdim == 2)
		{
			status = nc_def_var(ncid, varname.c_str(), VarTYPE, vdim, var_dimid2D, &var_id);
			if (status != NC_NOERR) handle_ncerror(status);
		}
		else if (vdim == 3)
		{
			status = nc_def_var(ncid, varname.c_str(), VarTYPE, vdim, var_dimid3D, &var_id);
			if (status != NC_NOERR) handle_ncerror(status);
		}

		if (smallnc > 0)
		{

			status = nc_put_att_short(ncid, var_id, "_FillValue", NC_SHORT, 1, &fillval_s);
			if (status != NC_NOERR) handle_ncerror(status);
			status = nc_put_att_short(ncid, var_id, "missingvalue", NC_SHORT, 1, &fillval_s);

			if (status != NC_NOERR) handle_ncerror(status);
		}
		else
		{
			status = nc_put_att_float(ncid, var_id, "_FillValue", NC_FLOAT, 1, &fillval);
			if (status != NC_NOERR) handle_ncerror(status);
			status = nc_put_att_float(ncid, var_id, "missingvalue", NC_FLOAT, 1, &fillval);

			if (status != NC_NOERR) handle_ncerror(status);
		}
		

		if (smallnc > 0)
		{
			
			status = nc_put_att_float(ncid, var_id, "scale_factor", NC_FLOAT, 1, &scalefactor);
			if (status != NC_NOERR) handle_ncerror(status);
			status = nc_put_att_float(ncid, var_id, "add_offset", NC_FLOAT, 1, &addoffset);
			if (status != NC_NOERR) handle_ncerror(status);
		}
		

		status = nc_put_att_text(ncid, var_id, "standard_name", stdname.size(), stdname.c_str());
		status = nc_put_att_text(ncid, var_id, "long_name", longname.size(), longname.c_str());
		status = nc_put_att_text(ncid, var_id, "units", unit.size(),unit.c_str());

		std::string crsstrname = "crs";
		status = nc_put_att_text(ncid, var_id, "grid_mapping", crsstrname.size(), crsstrname.c_str());
		

		int shuffle = 1;
		int deflate = 1;        // This switches compression on (1) or off (0).
		int deflate_level = 9;  // This is the compression level in range 1 (less) - 9 (more).
		nc_def_var_deflate(ncid, var_id, shuffle, deflate, deflate_level);

	}
	// End definition
	status = nc_enddef(ncid);
	if (status != NC_NOERR) handle_ncerror(status);

	//printf("\n ib=%d count3D=[%d,%d,%d]\n", count3D[0], count3D[1], count3D[2]);

	// Now write the initial value of the Variable out

	//std::vector<int> activeblkzone = Calcactiveblockzone(XParam, activeblk, Xzone);

	//####################
	for (ibl = 0; ibl < Xzone.nblk; ibl++)
	{
		
		//bl = activeblk[Xzone.blk[ibl]];
		bl = activeblkzone[ibl];
		lev = level[bl];


		double xxmax, yymax;
		double xxmin, yymin;
		double initdx = calcres(XParam.dx, XParam.initlevel);

		//xxmax = XParam.xmax + initdx / 2.0 - calcres(XParam.dx, lev )/2.0;
		//yymax = XParam.ymax + initdx / 2.0 - calcres(XParam.dx, lev )/2.0;

		xxmax = Xzone.xmax - calcres(XParam.dx, lev) / 2.0;
		yymax = Xzone.ymax - calcres(XParam.dx, lev )/2.0;

		//xxmin = XParam.xo - initdx / 2.0 + calcres(XParam.dx, lev )/2.0;
		//yymin = XParam.yo - initdx / 2.0 + calcres(XParam.dx, lev )/2.0;
		xxmin = Xzone.xo + calcres(XParam.dx, lev) / 2.0;
		yymin = Xzone.yo + calcres(XParam.dx, lev )/2.0;
		//printf("xxmin=%f, yymin=%f, lev=$d \n", xxmin, yymin, lev);

		//std::string xxname, yyname, sign;

		lev < 0 ? sign = "N" : sign = "P";


		xxname = "xx_" + sign + std::to_string(abs(lev));
		yyname = "yy_" + sign + std::to_string(abs(lev));

		varname = varst +  "_" + sign + std::to_string(abs(lev));
		status = nc_inq_dimid(ncid, xxname.c_str(), &xid);
		if (status != NC_NOERR) handle_ncerror(status);
		status = nc_inq_dimid(ncid, yyname.c_str(), &yid);
		if (status != NC_NOERR) handle_ncerror(status);
		status = nc_inq_varid(ncid, varname.c_str(), &var_id);
		if (status != NC_NOERR) handle_ncerror(status);

		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				int n = (i + XParam.halowidth + XParam.outishift) + (j + XParam.halowidth + XParam.outjshift) * XParam.blkmemwidth + bl * XParam.blksize;
				int r = i + j * XParam.blkwidth;
				if (smallnc > 0)
				{
					// packed_data_value = nint((unpacked_data_value - add_offset) / scale_factor)
					varblk_s[r] = (short)round((var[n] - addoffset) / scalefactor);
				}
				else
				{
					varblk[r] = (float)var[n];
				}
			}
		}
		if (vdim == 2)
		{
			start2D[0] = (size_t)round((XParam.yo + blockyo[bl] - yymin) / calcres(XParam.dx, lev));
			start2D[1] = (size_t)round((XParam.xo + blockxo[bl] - xxmin) / calcres(XParam.dx, lev));

			if (smallnc > 0)
			{
				status = nc_put_vara_short(ncid, var_id, start2D, count2D, varblk_s);
				if (status != NC_NOERR) handle_ncerror(status);
			}
			else
			{
				status = nc_put_vara_float(ncid, var_id, start2D, count2D, varblk);
				if (status != NC_NOERR) handle_ncerror(status);
			}
		}
		else if (vdim == 3)
		{
			//
			start3D[1] = (size_t)round((XParam.yo + blockyo[bl] - yymin) / calcres(XParam.dx, lev));
			start3D[2] = (size_t)round((XParam.xo + blockxo[bl] - xxmin) / calcres(XParam.dx, lev));
	
 			if (smallnc > 0)
			{
				status = nc_put_vara_short(ncid, var_id, start3D, count3D, varblk_s);
				if (status != NC_NOERR) handle_ncerror(status);
			}
			else
			{
				status = nc_put_vara_float(ncid, var_id, start3D, count3D, varblk);

				if (status != NC_NOERR)
				{
					//printf("\n ib=%d start=[%d,%d,%d]; initlevel=%d; initdx=%f; level=%d; xo=%f; yo=%f; blockxo[ib]=%f xxmin=%f blockyo[ib]=%f yymin=%f startfl=%f\n", bl, start3D[0], start3D[1], start3D[2], XParam.initlevel,initdx,lev, Xzone.xo, Xzone.yo, blockxo[bl], xxmin, blockyo[bl], yymin, (blockyo[bl] - yymin) / calcres(XParam.dx, lev));
					//printf("\n varblk[0]=%f varblk[255]=%f\n", varblk[0], varblk[255]);
					handle_ncerror(status);
				}
			}

		}

	}


	if (smallnc > 0)
	{

		free(varblk_s);
	}
	free(varblk);
	//close and save new file
	status = nc_close(ncid);
	if (status != NC_NOERR) handle_ncerror(status);

}

template void defncvarBUQ<float>(Param XParam, int* activeblk, int* level, float* blockxo, float* blockyo, std::string varst, int vdim, float* var, outzoneB Xzone);
template void defncvarBUQ<double>(Param XParam, int* activeblk, int* level, double* blockxo, double* blockyo, std::string varst, int vdim, double* var, outzoneB Xzone);



template <class T> void writencvarstepBUQ(Param XParam, int vdim, int * activeblk, int* level, T * blockxo, T *blockyo, std::string varst, T * var, outzoneB Xzone)
{
	int status, ncid, recid, var_id, ndims;
	static size_t nrec;
	short *varblk_s;
	float * varblk;
	int nx, ny;
	int dimids[NC_MAX_VAR_DIMS];
	size_t  *ddim, *start, *count;
	//XParam.outfile.c_str()

	static size_t start2D[] = { 0, 0 }; // start at first value 
	//static size_t count2D[] = { ny, nx };
	static size_t count2D[] = { XParam.blkwidth, XParam.blkwidth };

	static size_t start3D[] = { 0, 0, 0 }; // start at first value // This is updated to nrec-1 further down
	//static size_t count3D[] = { 1, ny, nx };
	static size_t count3D[] = { 1, XParam.blkwidth, XParam.blkwidth };

	int smallnc = XParam.smallnc;
	float scalefactor = XParam.scalefactor;
	float addoffset = XParam.addoffset;

	status = nc_open(Xzone.outname.c_str(), NC_WRITE, &ncid);
	if (status != NC_NOERR) handle_ncerror(status);
	//read id from time dimension
	status = nc_inq_unlimdim(ncid, &recid);
	if (status != NC_NOERR) handle_ncerror(status);
	status = nc_inq_dimlen(ncid, recid, &nrec);
	if (status != NC_NOERR) handle_ncerror(status);

	start3D[0] = nrec - 1;

	varblk = (float *)malloc(XParam.blkwidth* XParam.blkwidth * sizeof(float));
	if (smallnc > 0)
	{

		varblk_s = (short *)malloc(XParam.blkwidth * XParam.blkwidth * sizeof(short));
	}


	std::string xxname, yyname, varname, sign;
	std::vector<int> activeblkzone = Calcactiveblockzone(XParam, activeblk, Xzone);


	int lev, bl;
	for (int ibl = 0; ibl < Xzone.nblk; ibl++)
	{
		//bl = activeblk[Xzone.blk[ibl]];
	//for (int ibl = 0; ibl < XParam.nblk; ibl++)
	//{
		bl = activeblkzone[ibl];
		lev = level[bl];
		lev < 0 ? sign = "N" : sign = "P";
		double xxmax, xxmin, yymax, yymin;

		double initdx = calcres(XParam.dx, XParam.initlevel);

		//xxmax = XParam.xmax + initdx / 2.0 - calcres(XParam.dx, lev) / 2.0;
		//yymax = XParam.ymax + initdx / 2.0 - calcres(XParam.dx, lev) / 2.0;
		xxmax = Xzone.xmax - calcres(XParam.dx, lev) / 2.0;
		yymax = Xzone.ymax - calcres(XParam.dx, lev) / 2.0;

		//xxmin = XParam.xo - initdx / 2.0 + calcres(XParam.dx, lev) / 2.0;
		//yymin = XParam.yo - initdx / 2.0 + calcres(XParam.dx, lev) / 2.0;
		xxmin = Xzone.xo + calcres(XParam.dx, lev) / 2.0;
		yymin = Xzone.yo + calcres(XParam.dx, lev) / 2.0;

		varname = varst + "_" + sign + std::to_string(abs(lev));


		status = nc_inq_varid(ncid, varname.c_str(), &var_id);
		if (status != NC_NOERR) handle_ncerror(status);

		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				int n = (i + XParam.halowidth + XParam.outishift) + (j + XParam.halowidth + XParam.outjshift) * XParam.blkmemwidth + bl * XParam.blksize;
				int r = i + j * XParam.blkwidth;
				if (smallnc > 0)
				{
					// packed_data_value = nint((unpacked_data_value - add_offset) / scale_factor)
					varblk_s[r] = (short)round((var[n] - addoffset) / scalefactor);
				}
				else
				{
					varblk[r] = var[n];
				}
			}
		}
		if (vdim == 2)
		{
			start2D[0] = (size_t)round((XParam.yo + blockyo[bl] - yymin) / calcres(XParam.dx, lev));
			start2D[1] = (size_t)round((XParam.xo + blockxo[bl] - xxmin) / calcres(XParam.dx, lev));

			if (smallnc > 0)
			{


				status = nc_put_vara_short(ncid, var_id, start2D, count2D, varblk_s);
				if (status != NC_NOERR) handle_ncerror(status);
			}
			else
			{
				status = nc_put_vara_float(ncid, var_id, start2D, count2D, varblk);
				if (status != NC_NOERR) handle_ncerror(status);
			}
		}
		else if (vdim == 3)
		{
			start3D[1] = (size_t)round((XParam.yo + blockyo[bl] - yymin) / calcres(XParam.dx, lev));
			start3D[2] = (size_t)round((XParam.xo + blockxo[bl] - xxmin) / calcres(XParam.dx, lev));
			if (smallnc > 0)
			{
				status = nc_put_vara_short(ncid, var_id, start3D, count3D, varblk_s);
				if (status != NC_NOERR) handle_ncerror(status);
			}
			else
			{
				status = nc_put_vara_float(ncid, var_id, start3D, count3D, varblk);
				if (status != NC_NOERR) handle_ncerror(status);
				//printf("\n ib=%d start=[%d,%d,%d]; initlevel=%d; initdx=%f; level=%d; xo=%f; yo=%f; blockxo[ib]=%f xxmin=%f blockyo[ib]=%f yymin=%f startfl=%f\n", bl, start3D[0], start3D[1], start3D[2], XParam.initlevel, initdx, lev, Xzone.xo, Xzone.yo, blockxo[bl], xxmin, blockyo[bl], yymin, (blockyo[bl] - yymin) / calcres(XParam.dx, lev));
				//printf("\n varblk[0]=%f varblk[255]=%f\n", varblk[0], varblk[255]);
				//printf("\n ib=%d count3D=[%d,%d,%d]\n", count3D[0], count3D[1], count3D[2]);
				//printf("\n ib=%d; level=%d; blockxo[ib]=%f blockyo[ib]=%f \n", bl, lev, blockxo[bl], blockyo[bl]);
			}

		}

	}


	if (smallnc > 0)
	{

		free(varblk_s);
	}
	free(varblk);
	//close and save new file
	status = nc_close(ncid);
	if (status != NC_NOERR) handle_ncerror(status);
}

// Scope for compiler to know what function to compile

template void writencvarstepBUQ<float>(Param XParam, int vdim, int * activeblk, int* level, float * blockxo, float *blockyo, std::string varst, float * var, outzoneB Xzone);
template void writencvarstepBUQ<double>(Param XParam, int vdim, int * activeblk, int* level, double * blockxo, double *blockyo, std::string varst, double * var, outzoneB Xzone);

extern "C" void writenctimestep(std::string outfile, double totaltime)
{
	int status, ncid, recid, time_id;
	status = nc_open(outfile.c_str(), NC_WRITE, &ncid);
	if (status != NC_NOERR) handle_ncerror(status);
	static size_t nrec;
	static size_t tst[] = { 0 };
	//read id from time dimension
	status = nc_inq_unlimdim(ncid, &recid);
	if (status != NC_NOERR) handle_ncerror(status);
	status = nc_inq_dimlen(ncid, recid, &nrec);
	if (status != NC_NOERR) handle_ncerror(status);
	status = nc_inq_varid(ncid, "time", &time_id);
	if (status != NC_NOERR) handle_ncerror(status);
	tst[0] = nrec;
	status = nc_put_var1_double(ncid, time_id, tst, &totaltime);
	if (status != NC_NOERR) handle_ncerror(status);
	//close and save
	status = nc_close(ncid);
	if (status != NC_NOERR) handle_ncerror(status);
}

template <class T> void InitSave2Netcdf(Param &XParam, Model<T> &XModel)
{
	if (!XParam.outvars.empty())
	{
		log("Create netCDF output file...");
		creatncfileBUQ(XParam, XModel.blocks);
		//creatncfileBUQ(XParam);
		for (int o = 0; o < XModel.blocks.outZone.size(); o++)
		{
			writenctimestep(XModel.blocks.outZone[o].outname, XParam.totaltime);
			for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
			{
				std::string varstr = XParam.outvars[ivar];
				defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, varstr,XModel.Outvarlongname[varstr],XModel.Outvarstdname[varstr],XModel.Outvarunits[varstr], 3, XModel.OutputVarMap[varstr], XModel.blocks.outZone[o]);
			}
		}
	}
}
template void InitSave2Netcdf<float>(Param &XParam, Model<float> &XModel);
template void InitSave2Netcdf<double>(Param &XParam, Model<double> &XModel);


template <class T> void Save2Netcdf(Param XParam,Loop<T> XLoop, Model<T> XModel)
{
	if (!XParam.outvars.empty())
	{
		//creatncfileBUQ(XParam);
		for (int o = 0; o < XModel.blocks.outZone.size(); o++)
		{
			writenctimestep(XModel.blocks.outZone[o].outname, XLoop.totaltime);
			for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
			{
				writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, XParam.outvars[ivar], XModel.OutputVarMap[XParam.outvars[ivar]], XModel.blocks.outZone[o]);
			}
		}
	}
}
template void Save2Netcdf<float>(Param XParam, Loop<float> XLoop, Model<float> XModel);
template void Save2Netcdf<double>(Param XParam, Loop<double> XLoop, Model<double> XModel);


//The following functions are tools to create 2D or 3D netcdf files (for testing for example)

//
extern "C" void create2dnc(char* filename, int nx, int ny, double* xx, double* yy, double* var, char* varname)
{
	int status;
	int ncid, xx_dim, yy_dim, time_dim, p_dim, tvar_id;

	size_t nxx, nyy, ntt;
	static size_t start[] = { 0, 0 }; // start at first value
	static size_t count[] = { ny, nx };
	int time_id, xx_id, yy_id; //
	nxx = nx;
	nyy = ny;

	//create the netcdf dataset
	status = nc_create(filename, NC_NOCLOBBER, &ncid);

	//Define dimensions: Name and length

	status = nc_def_dim(ncid, "xx", nxx, &xx_dim);
	status = nc_def_dim(ncid, "yy", nyy, &yy_dim);
	int xdim[] = { xx_dim };
	int ydim[] = { yy_dim };

	//define variables: Name, Type,...
	int var_dimids[3];
	var_dimids[0] = yy_dim;
	var_dimids[1] = xx_dim;

	status = nc_def_var(ncid, "xx", NC_DOUBLE, 1, xdim, &xx_id);
	status = nc_def_var(ncid, "yy", NC_DOUBLE, 1, ydim, &yy_id);


	status = nc_def_var(ncid, varname, NC_DOUBLE, 2, var_dimids, &tvar_id);

	status = nc_enddef(ncid);

	static size_t xstart[] = { 0 }; // start at first value
	static size_t xcount[] = { nx };
	
	static size_t ystart[] = { 0 }; // start at first value
	static size_t ycount[] = { ny };



	//Provide values for variables
	status = nc_put_vara_double(ncid, xx_id, xstart, xcount, xx);
	status = nc_put_vara_double(ncid, yy_id, ystart, ycount, yy);

	status = nc_put_vara_double(ncid, tvar_id, start, count, var);
	status = nc_close(ncid);

}

//Create a ncdf file containing a 3D variable (the file is overwritten if it was existing before)
extern "C" void create3dnc(char* name, int nx, int ny, int nt, double* xx, double* yy, double* theta, double* var, char* varname)
{
	int status;
	int ncid, xx_dim, yy_dim, tt_dim, tvar_id;
	size_t nxx, nyy, ntt;
	static size_t start[] = { 0, 0, 0 }; // start at first value
	static size_t count[] = { nt, ny, nx };
	int xx_id, yy_id, tt_id; //
	nxx = nx;
	nyy = ny;
	ntt = nt;

	//create the netcdf dataset
	status = nc_create(name, NC_CLOBBER, &ncid);
	//Define dimensions: Name and length
	status = nc_def_dim(ncid, "xx", nxx, &xx_dim);
	status = nc_def_dim(ncid, "yy", nyy, &yy_dim);
	status = nc_def_dim(ncid, "time", ntt, &tt_dim);
	int xdim[] = { xx_dim };
	int ydim[] = { yy_dim };
	int tdim[] = { tt_dim };

	//define variables: Name, Type,...
	int var_dimids[3];
	var_dimids[0] = tt_dim;
	var_dimids[1] = yy_dim;
	var_dimids[2] = xx_dim;

	status = nc_def_var(ncid, "time", NC_DOUBLE, 1, tdim, &tt_id);
	status = nc_def_var(ncid, "xx", NC_DOUBLE, 1, xdim, &xx_id);
	status = nc_def_var(ncid, "yy", NC_DOUBLE, 1, ydim, &yy_id);

	status = nc_def_var(ncid, varname, NC_DOUBLE, 3, var_dimids, &tvar_id);

	status = nc_enddef(ncid);

	static size_t tst[] = { 0 };
	static size_t xstart[] = { 0 }; // start at first value
	static size_t xcount[] = { nx };
	static size_t ystart[] = { 0 }; // start at first value
	static size_t ycount[] = { ny };

	static size_t tstart[] = { 0 }; // start at first value
	static size_t tcount[] = { nt };

	//Provide values for variables
	status = nc_put_vara_double(ncid, xx_id, xstart, xcount, xx);
	status = nc_put_vara_double(ncid, yy_id, ystart, ycount, yy);
	status = nc_put_vara_double(ncid, tt_id, tstart, tcount, theta);

	status = nc_put_vara_double(ncid, tvar_id, start, count, var);
	status = nc_close(ncid);

}

extern "C" void write3dvarnc(int nx, int ny, int nt, double totaltime, double* var)
{
	int status;
	int ncid, time_dim, recid;
	size_t nxx, nyy;
	static size_t start[] = { 0, 0, 0, 0 }; // start at first value
	static size_t count[] = { 1, nt, ny, nx };
	static size_t tst[] = { 0 };
	int time_id, var_id;

	nxx = nx;
	nyy = ny;

	static size_t nrec;
	status = nc_open("3Dvar.nc", NC_WRITE, &ncid);
	//read id from time dimension
	status = nc_inq_unlimdim(ncid, &recid);
	status = nc_inq_dimlen(ncid, recid, &nrec);
	//printf("nrec=%d\n",nrec);

	 //read file for variable ids
	status = nc_inq_varid(ncid, "time", &time_id);
	status = nc_inq_varid(ncid, "3Dvar", &var_id);
	start[0] = nrec;//
	tst[0] = nrec;

	//Provide values for variables
	status = nc_put_var1_double(ncid, time_id, tst, &totaltime);
	status = nc_put_vara_double(ncid, var_id, start, count, var);
	status = nc_close(ncid);

}

extern "C" void write2dvarnc(int nx, int ny, double totaltime, double* var)
{
	int status;
	int ncid, time_dim, recid;
	size_t nxx, nyy;
	static size_t start[] = { 0, 0, 0 }; // start at first value
	static size_t count[] = { 1, ny, nx };
	static size_t tst[] = { 0 };
	int time_id, var_id;

	nxx = nx;
	nyy = ny;

	static size_t nrec;
	status = nc_open("3Dvar.nc", NC_WRITE, &ncid);

	//read id from time dimension
	status = nc_inq_unlimdim(ncid, &recid);
	status = nc_inq_dimlen(ncid, recid, &nrec);
	//printf("nrec=%d\n",nrec);

	 //read file for variable ids
	status = nc_inq_varid(ncid, "time", &time_id);
	status = nc_inq_varid(ncid, "3Dvar", &var_id);

	start[0] = nrec;//
	tst[0] = nrec;

	//Provide values for variables
	status = nc_put_var1_double(ncid, time_id, tst, &totaltime);
	status = nc_put_vara_double(ncid, var_id, start, count, var);
	status = nc_close(ncid);

}
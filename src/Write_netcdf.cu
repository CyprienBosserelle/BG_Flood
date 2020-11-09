
#include "Write_netcdf.h"

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

template<class T>
void creatncfileBUQ(Param &XParam,int * activeblk, int * level, T * blockxo, T * blockyo)
{

	int status;
	int nx, ny;
	//double dx = XParam.dx;
	size_t nxx, nyy;
	int ncid, xx_dim, yy_dim, time_dim, blockid_dim;
	double * xval, *yval;
	// create the netcdf datasetXParam.outfile.c_str()
	status = nc_create(XParam.outfile.c_str(), NC_NOCLOBBER|NC_NETCDF4, &ncid);
	if (status != NC_NOERR)
	{
		if (status == NC_EEXIST) // File already axist so automatically rename the output file 
		{
			//printf("Warning! Outut file name already exist  ");
			log("Warning! Outut file name already exist   ");
			int fileinc = 1;
			std::vector<std::string> extvec = split(XParam.outfile, '.');
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
				XParam.outfile = newname;
				status = nc_create(XParam.outfile.c_str(), NC_NOCLOBBER|NC_NETCDF4, &ncid);
				fileinc++;
			}
			//printf("New file name: %s  ", XParam.outfile.c_str());
			log("New file name: " + XParam.outfile);

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

	xmin = XParam.xo ;
	xmax = XParam.xmax ;
	ymin = XParam.yo ;
	ymax = XParam.ymax ;

	// Define global attributes
	status = nc_put_att_int(ncid, NC_GLOBAL, "maxlevel", NC_INT, 1, &XParam.maxlevel);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_put_att_int(ncid, NC_GLOBAL, "minlevel", NC_INT, 1, &XParam.minlevel);
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
	
	//static size_t tst[] = { 0 };
	size_t blkstart[] = { 0 };
	size_t blkcount[] = { (size_t) XParam.nblk };
	size_t xcount[] = { 0 };
	size_t ycount[] = { 0 };
	static size_t xstart[] = { 0 }; // start at first value
	static size_t ystart[] = { 0 };
	status = nc_def_var(ncid, "time", NC_FLOAT, 1, tdim, &time_id);
	if (status != NC_NOERR) handle_ncerror(status);

	// Define dimensions and variables to store block id,, status, level xo, yo

	status = nc_def_dim(ncid, "blockid", XParam.nblk, &blockid_dim);
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


	// For each level Define xx yy 
	for (int lev = XParam.minlevel; lev <= XParam.maxlevel; lev++)
	{

		double ddx = calcres(XParam.dx, lev);
		double initdx= calcres(XParam.dx, XParam.initlevel);
		double xxmax, xxmin, yymax, yymin;

		xxmax = XParam.xmax - calcres(XParam.dx, lev + 1);
		yymax = XParam.ymax - calcres(XParam.dx, lev + 1);

		xxmin = XParam.xo + calcres(XParam.dx, lev + 1);
		yymin = XParam.yo + calcres(XParam.dx, lev + 1);

		nx = (xxmax - xxmin) / ddx + 1;
		ny = (yymax - yymin) / ddx + 1;

		//printf("lev=%d; xxmax=%f; xxmin=%f; nx=%d\n", lev, xxmax, xxmin,nx);
		//printf("lev=%d; yymax=%f; yymin=%f; ny=%d\n", lev, yymax, yymin, ny);

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


	AllocateCPU(1, XParam.nblk, blkwidth);
	AllocateCPU(1, XParam.nblk, blkid);


	for (int ib = 0; ib < XParam.nblk; ib++)
	{
		int ibl = activeblk[ib];
		blkwidth[ib] = (float)calcres(XParam.dx, level[ibl]);
		blkid[ib] = ibl;
	}

	status = nc_put_vara_int(ncid, blkid_id, blkstart, blkcount, blkid);
	//status = nc_put_vara_int(ncid, blkstatus_id, blkstart, blkcount, activeblk);
	status = nc_put_vara_float(ncid, blkwidth_id, blkstart, blkcount, blkwidth);


	// Reusing blkwidth for other array
	// This is needed because the blockxo array may be shuffled to memory block beyond nblk
	for (int ib = 0; ib < XParam.nblk; ib++)
	{
		int ibl = activeblk[ib];
		blkwidth[ib] = XParam.xo + blockxo[ibl];
		blkid[ib] = level[ibl];
		
	}

	status = nc_put_vara_float(ncid, blkxo_id, blkstart, blkcount, blkwidth);
	status = nc_put_vara_int(ncid, blklevel_id, blkstart, blkcount, blkid);
	for (int ib = 0; ib < XParam.nblk; ib++)
	{
		int ibl = activeblk[ib];
		blkwidth[ib] = XParam.yo + blockyo[ibl];
	}

	status = nc_put_vara_float(ncid, blkyo_id, blkstart, blkcount, blkwidth);
	
	



	free(blkid);
	free(blkwidth);

	if (status != NC_NOERR) handle_ncerror(status);
	
	std::string xxname, yyname, sign;

	for (int lev = XParam.minlevel; lev <= XParam.maxlevel; lev++)
	{
		double ddx = calcres(XParam.dx, lev);
		double initdx = calcres(XParam.dx, XParam.initlevel);
		double xxmax, xxmin, yymax, yymin;

		xxmax = XParam.xmax - calcres(XParam.dx, lev + 1);
		yymax = XParam.ymax - calcres(XParam.dx, lev + 1);

		xxmin = XParam.xo + calcres(XParam.dx, lev + 1);
		yymin = XParam.yo  + calcres(XParam.dx, lev + 1);

		nx = (xxmax - xxmin) / ddx + 1;
		ny = (yymax - yymin) / ddx + 1;


		

		//printf("lev=%d; xxmax=%f; xxmin=%f; nx=%d\n", lev, xxmax, xxmin, nx);
		//printf("lev=%d; yymax=%f; yymin=%f; ny=%d\n", lev, yymax, yymin, ny);
		// start at first value
		//static size_t thstart[] = { 0 };
		xcount[0] = nx;
		ycount[0] = ny;
		//Recreat the x, y
		xval = (double *)malloc(nx*sizeof(double));
		yval = (double*)malloc(ny*sizeof(double));


		for (int i = 0; i < nx; i++)
		{
			xval[i] = xxmin + double(i)*ddx;
		}

		for (int i = 0; i < ny; i++)
		{
			yval[i] = yymin + double(i) * ddx;
		}


		//printf("yval[0]=%f\tyval[1]=%f\t yymin=%f\n", yval[0], yval[1], yymin);
		//std::string xxname, yyname, sign;

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

template void creatncfileBUQ<float>(Param &XParam, int* activeblk, int* level, float* blockxo, float* blockyo);
template void creatncfileBUQ<double>(Param &XParam, int* activeblk, int* level, double* blockxo, double* blockyo);


template<class T>
void creatncfileBUQ(Param &XParam, BlockP<T> XBlock)
{
	creatncfileBUQ(XParam, XBlock.active, XBlock.level, XBlock.xo, XBlock.yo);
}
template void creatncfileBUQ<float>(Param &XParam, BlockP<float> XBlock);
template void creatncfileBUQ<double>(Param &XParam, BlockP<double> XBlock);

template <class T> void defncvarBUQ(Param XParam, int * activeblk, int * level, T * blockxo, T *blockyo, std::string varst, int vdim, T * var)
{
	std::string outfile = XParam.outfile;
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

	short *varblk_s;
	float * varblk;
	int recid, xid, yid;
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

	nc_type VarTYPE;

	if (smallnc > 0)
	{
		VarTYPE = NC_SHORT;
	}
	else
	{
		VarTYPE = NC_FLOAT;
	}




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
	for (int lev = XParam.minlevel; lev <= XParam.maxlevel; lev++)
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
		




	}
	// End definition
	status = nc_enddef(ncid);
	if (status != NC_NOERR) handle_ncerror(status);

	// Now write the initial value of the Variable out
	int lev, bl;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		bl = activeblk[ibl];
		lev = level[bl];


		double xxmax, yymax;
		double xxmin, yymin;
		double initdx = calcres(XParam.dx, XParam.initlevel);

		//xxmax = XParam.xmax + initdx / 2.0 - calcres(XParam.dx, lev )/2.0;
		//yymax = XParam.ymax + initdx / 2.0 - calcres(XParam.dx, lev )/2.0;

		xxmax = XParam.xmax - calcres(XParam.dx, lev) / 2.0;
		yymax = XParam.ymax - calcres(XParam.dx, lev )/2.0;

		//xxmin = XParam.xo - initdx / 2.0 + calcres(XParam.dx, lev )/2.0;
		//yymin = XParam.yo - initdx / 2.0 + calcres(XParam.dx, lev )/2.0;
		xxmin = XParam.xo + calcres(XParam.dx, lev) / 2.0;
		yymin = XParam.yo + calcres(XParam.dx, lev )/2.0;



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
				int n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + bl * XParam.blksize;
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
			//printf("id=%d\tlev=%d\tblockxo=%f\tblockyo=%f\txxo=%f\tyyo=%f\n",bl, lev, blockxo[bl], blockyo[bl], round((blockxo[bl] - xxmin) / calcres(XParam.dx, lev)), round((blockyo[bl] - yymin) / calcres(XParam.dx, lev)));
			start3D[1] = (size_t)round((XParam.yo + blockyo[bl] - yymin) / calcres(XParam.dx, lev));
			start3D[2] = (size_t)round((XParam.xo + blockxo[bl] - xxmin) / calcres(XParam.dx, lev));
			//printf("id=%d\tlev=%d\tblockxo=%f\tblockyo=%f\txxo=%f\tyyo=%f\n", bl, lev, blockxo[bl], blockyo[bl], round((blockxo[bl] - xxmin) / calcres(XParam.dx, lev)), round((blockyo[bl] - yymin) / calcres(XParam.dx, lev)));
			//printf("id=%d\tlev=%d\tblockxo=%f\tblockyo=%f\txxo=%f\tyyo=%f\n", bl, lev, blockxo[bl], blockyo[bl], round((blockxo[bl] - xxmin) / calcres(XParam.dx, lev)), round((blockyo[bl] - yymin) / calcres(XParam.dx, lev)));


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
					//printf("\n ib=%d start=[%d,%d,%d]; initlevel=%d; initdx=%f; level=%d; xo=%f; yo=%f; blockxo[ib]=%f xxmin=%f blockyo[ib]=%f yymin=%f startfl=%f\n", bl, start3D[0], start3D[1], start3D[2], XParam.initlevel,initdx,lev, XParam.xo, XParam.yo,blockxo[bl],xxmin, blockyo[bl],yymin, (blockyo[bl] - yymin) / calcres(XParam.dx, lev));
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

template void defncvarBUQ<float>(Param XParam, int* activeblk, int* level, float* blockxo, float* blockyo, std::string varst, int vdim, float* var);
template void defncvarBUQ<double>(Param XParam, int* activeblk, int* level, double* blockxo, double* blockyo, std::string varst, int vdim, double* var);



template <class T> void writencvarstepBUQ(Param XParam, int vdim, int * activeblk, int* level, T * blockxo, T *blockyo, std::string varst, T * var)
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

	status = nc_open(XParam.outfile.c_str(), NC_WRITE, &ncid);
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

	int lev, bl;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		bl = activeblk[ibl];
		lev = level[bl];
		lev < 0 ? sign = "N" : sign = "P";
		double xxmax, xxmin, yymax, yymin;

		double initdx = calcres(XParam.dx, XParam.initlevel);

		//xxmax = XParam.xmax + initdx / 2.0 - calcres(XParam.dx, lev) / 2.0;
		//yymax = XParam.ymax + initdx / 2.0 - calcres(XParam.dx, lev) / 2.0;
		xxmax = XParam.xmax - calcres(XParam.dx, lev) / 2.0;
		yymax = XParam.ymax - calcres(XParam.dx, lev) / 2.0;

		//xxmin = XParam.xo - initdx / 2.0 + calcres(XParam.dx, lev) / 2.0;
		//yymin = XParam.yo - initdx / 2.0 + calcres(XParam.dx, lev) / 2.0;
		xxmin = XParam.xo + calcres(XParam.dx, lev) / 2.0;
		yymin = XParam.yo + calcres(XParam.dx, lev) / 2.0;

		varname = varst + "_" + sign + std::to_string(abs(lev));


		status = nc_inq_varid(ncid, varname.c_str(), &var_id);
		if (status != NC_NOERR) handle_ncerror(status);

		for (int j = 0; j < XParam.blkwidth; j++)
		{
			for (int i = 0; i < XParam.blkwidth; i++)
			{
				int n = (i + XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + bl * XParam.blksize;
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
				if (status != NC_NOERR) handle_ncerror(status);
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


template void writencvarstepBUQ<float>(Param XParam, int vdim, int * activeblk, int* level, float * blockxo, float *blockyo, std::string varst, float * var);
template void writencvarstepBUQ<double>(Param XParam, int vdim, int * activeblk, int* level, double * blockxo, double *blockyo, std::string varst, double * var);

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

template <class T> void InitSave2Netcdf(Param &XParam, Model<T> XModel)
{
	if (!XParam.outvars.empty())
	{
		log("Create netCDF output file...");
		creatncfileBUQ(XParam, XModel.blocks);
		//creatncfileBUQ(XParam);
		for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
		{

			defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, XParam.outvars[ivar], 3, XModel.OutputVarMap[XParam.outvars[ivar]]);

		}
	}
}
template void InitSave2Netcdf<float>(Param &XParam, Model<float> XModel);
template void InitSave2Netcdf<double>(Param &XParam, Model<double> XModel);


template <class T> void Save2Netcdf(Param XParam,Loop<T> XLoop, Model<T> XModel)
{
	if (!XParam.outvars.empty())
	{
		writenctimestep(XParam.outfile, XLoop.totaltime);
		//creatncfileBUQ(XParam);
		for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
		{
			writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, XParam.outvars[ivar], XModel.OutputVarMap[XParam.outvars[ivar]]);
		}
	}
}
template void Save2Netcdf<float>(Param XParam, Loop<float> XLoop, Model<float> XModel);
template void Save2Netcdf<double>(Param XParam, Loop<double> XLoop, Model<double> XModel);

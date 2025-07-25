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

#include "Header.cuh"
#define pi 3.14159265


extern "C" void readnczb(int nx, int ny, const std::string ncfile, float * &zb)
{
	int status;
	int ncid, hh_id;

	std::string varstr,ncfilestr;
	std::vector<std::string> nameelements;
	//by default we expect tab delimitation
	nameelements = split(ncfile, '?');
	if (nameelements.size() > 1)
	{
		//variable name for bathy is not given so it is assumed to be zb
		ncfilestr = nameelements[0];
		varstr = nameelements[1];
	}
	else
	{
		ncfilestr = ncfile;
		varstr = "zb";
	}


	status = nc_open(ncfilestr.c_str(), NC_NOWRITE, &ncid);
	if (status != NC_NOERR)	handle_error(status);
	status = nc_inq_varid(ncid, varstr.c_str(), &hh_id);
	if (status != NC_NOERR)	handle_error(status);
	status = nc_get_var_float(ncid, hh_id, zb);
	if (status != NC_NOERR)	handle_error(status);
	status = nc_close(ncid);
	if (status != NC_NOERR)	handle_error(status);


}
void readgridncsize(const std::string ncfile, int &nx, int &ny, int &nt, double &dx, double &xo, double &yo, double &to, double &xmax, double &ymax, double &tmax)
{
	//read the dimentions of grid, levels and time
	int status;
	int ncid, ndimshh, ndims;
	double *xcoord, *ycoord, *tcoord;
	int varid;

	//int ndimsp, nvarsp, nattsp, unlimdimidp;

	int dimids[NC_MAX_VAR_DIMS];   /* dimension IDs */
	char coordname[NC_MAX_NAME + 1];
	//char varname[NC_MAX_NAME + 1];
	size_t  *ddimhh;

	std::string ncfilestr;
	std::string varstr;


	//char ncfile[]="ocean_ausnwsrstwq2.nc";
	std::vector<std::string> nameelements;
	//by default we expect tab delimitation
	nameelements = split(ncfile, '?');
	if (nameelements.size() > 1)
	{
		//variable name for bathy is not given so it is assumed to be zb
		ncfilestr = nameelements[0];
		varstr = nameelements[1];
	}
	else
	{
		ncfilestr = ncfile;
		varstr = "zb";
	}
	//Open NC file
	printf("Open file\n");
	status = nc_open(ncfilestr.c_str(), NC_NOWRITE, &ncid);
	if (status != NC_NOERR) handle_error(status);


	//printf(" %s...\n", hhvar);
	status = nc_inq_varid(ncid, varstr.c_str(), &varid);
	if (status != NC_NOERR)	handle_error(status);



	status = nc_inq_varndims(ncid, varid, &ndimshh);
	if (status != NC_NOERR) handle_error(status);
	//printf("hhVar:%d dims\n", ndimshh);

	status = nc_inq_vardimid(ncid, varid, dimids);
	if (status != NC_NOERR) handle_error(status);

	ddimhh = (size_t *)malloc(ndimshh*sizeof(size_t));

	//Read dimensions nx_u ny_u
	for (int iddim = 0; iddim < ndimshh; iddim++)
	{
		status = nc_inq_dimlen(ncid, dimids[iddim], &ddimhh[iddim]);
		if (status != NC_NOERR) handle_error(status);

		//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
	}

	if (ndimshh > 2)
	{
		nt = (int) ddimhh[0];
		ny = (int) ddimhh[1];
		nx = (int) ddimhh[2];

	}
	else
	{
		nt = 0;
		ny = (int) ddimhh[0];
		nx = (int) ddimhh[1];
	}

	//allocate
	xcoord = (double *)malloc(nx*ny*sizeof(double));
	ycoord = (double *)malloc(nx*ny*sizeof(double));

	//inquire variable name for x dimension
	//aka x dim of hh
	int ycovar, xcovar, tcovar;

	if (ndimshh > 2)
	{
		tcovar = dimids[0];
		ycovar = dimids[1];
		xcovar = dimids[2];
	}
	else
	{
		ycovar = dimids[0];
		xcovar = dimids[1];
	}

	//ycoord
	status = nc_inq_dimname(ncid, ycovar, coordname);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varid(ncid, coordname, &varid);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varndims(ncid, varid, &ndims);
	if (status != NC_NOERR) handle_error(status);

	if (ndims < 2)
	{
		double * ytempvar;
		ytempvar = (double *)malloc(ny*sizeof(double));
		size_t start[] = { 0 };
		size_t count[] = { ny };
		status = nc_get_vara_double(ncid, varid, start, count, ytempvar);
		if (status != NC_NOERR) handle_error(status);

		for (int i = 0; i<nx; i++)
		{
			for (int j = 0; j<ny; j++)
			{

				ycoord[i + j*nx] = ytempvar[j];

			}
		}
		free(ytempvar);
	}
	else
	{
		size_t start[] = { 0, 0 };
		size_t count[] = { ny, nx };
		status = nc_get_vara_double(ncid, varid, start, count, ycoord);
		if (status != NC_NOERR) handle_error(status);

	}
	//xcoord
	status = nc_inq_dimname(ncid, xcovar, coordname);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varid(ncid, coordname, &varid);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varndims(ncid, varid, &ndims);
	if (status != NC_NOERR) handle_error(status);

	if (ndims < 2)
	{
		double * xtempvar;
		xtempvar = (double *)malloc(nx*sizeof(double));
		size_t start[] = { 0 };
		size_t count[] = { nx };
		status = nc_get_vara_double(ncid, varid, start, count, xtempvar);
		if (status != NC_NOERR) handle_error(status);

		for (int i = 0; i<nx; i++)
		{
			for (int j = 0; j<ny; j++)
			{

				xcoord[i + j*nx] = xtempvar[i];

			}
		}
		free(xtempvar);
	}
	else
	{
		size_t start[] = { 0, 0 };
		size_t count[] = { ny, nx };
		status = nc_get_vara_double(ncid, varid, start, count, xcoord);
		if (status != NC_NOERR) handle_error(status);

	}

	double dxx;
	//check dx
	dxx = (xcoord[nx - 1] - xcoord[0]) / (nx - 1.0);
	printf("xo=%lf; xmax=%lf; nx=%d; dxx=%lf\n", xcoord[0], xcoord[nx - 1], nx,dxx);
	//dyy = (float) abs(ycoord[0] - ycoord[(ny - 1)*nx]) / (ny - 1);


	//Read time dimension if any
	if (nt > 0)
	{
		//read dimension name
		status = nc_inq_dimname(ncid, tcovar, coordname);
		if (status != NC_NOERR) handle_error(status);

		//inquire variable id
		status = nc_inq_varid(ncid, coordname, &varid);
		if (status != NC_NOERR) handle_error(status);

		// read the dimension of time variable // yes it should be == 1
		status = nc_inq_varndims(ncid, varid, &ndims);
		if (status != NC_NOERR) handle_error(status);

		//allocate temporary array and read time vector
		double * ttempvar;
		ttempvar = (double *)malloc(nt * sizeof(double));
		size_t start[] = { 0 };
		size_t count[] = { nt };
		status = nc_get_vara_double(ncid, varid, start, count, ttempvar);

		to = ttempvar[0];
		tmax= ttempvar[nt-1];

		free(ttempvar);
	}
	else
	{
		//this is a 2d file so assign dummy values
		to = 0.0;
		tmax = 0.0;
	}

	dx = dxx;

	xo = xcoord[0];
	xmax = xcoord[nx - 1];
	yo= ycoord[0];
	ymax= ycoord[(ny - 1)*nx];



	status = nc_close(ncid);

	free(ddimhh);
	free(xcoord);
	free(ycoord);


}

int readvarinfo(std::string filename, std::string Varname, size_t *&ddimU)
{
	// This function reads the dimentions for each variables
	int status, varid;
	int ncid, ndims;
	int dimids[NC_MAX_VAR_DIMS];
	//Open NC file
	//printf("Open file\n");

	status = nc_open(filename.c_str(), 0, &ncid);
	if (status != NC_NOERR) handle_error(status);

	//inquire variable by name
	//printf("Reading information about %s...", Varname.c_str());
	status = nc_inq_varid(ncid, Varname.c_str(), &varid);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varndims(ncid, varid, &ndims);
	if (status != NC_NOERR) handle_error(status);


	status = nc_inq_vardimid(ncid, varid, dimids);
	if (status != NC_NOERR) handle_error(status);

	ddimU = (size_t *)malloc(ndims*sizeof(size_t));

	//Read dimensions nx_u ny_u
	for (int iddim = 0; iddim < ndims; iddim++)
	{
		status = nc_inq_dimlen(ncid, dimids[iddim], &ddimU[iddim]);
		if (status != NC_NOERR) handle_error(status);

		//printf("dim:%d=%d\n", iddim, ddimU[iddim]);
	}


	status = nc_close(ncid);

	return ndims;
}


int readnctime(std::string filename, double * &time)
{
	int status, ncid, varid;

	std::string ncfilestr;
	std::string varstr;


	//char ncfile[]="ocean_ausnwsrstwq2.nc";
	std::vector<std::string> nameelements;

	nameelements = split(filename, '?');
	if (nameelements.size() > 1)
	{
		//variable name for bathy is not given so it is assumed to be zb
		ncfilestr = nameelements[0];
		//varstr = nameelements[1];
	}
	else
	{
		ncfilestr = filename;
		//varstr = "time";
	}

	// Warning this could be more robust by taking the unlimited dimention if time does not exist!
	std::string Varname = "time";

	status = nc_open(ncfilestr.c_str(), 0, &ncid);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varid(ncid, Varname.c_str(), &varid);
	if (status != NC_NOERR) handle_error(status);

	status = nc_get_var_double(ncid, varid, time);
	if (status != NC_NOERR) handle_error(status);

	status = nc_close(ncid);

	return status;
}

int readncslev1(std::string filename, std::string varstr, size_t indx, size_t indy, size_t indt, bool checkhh, double eps, double * &zsa)
{
	int status, ncid, varid,ndims,sferr,oferr,misserr,fillerr, iderr, varerr;
	double scalefac, offset, missing, fillval;

	double hha,zza;

	//bool checkhh = false;

	int wet = 1;

	size_t *start, *count;
	//std::string Varname = "time";
	ndims = 3;

	start = (size_t *)malloc(ndims*sizeof(size_t));
	//count = (size_t *)malloc(ndims*sizeof(size_t));

	start[0] = indt;
	start[1] = indy;
	start[2] = indx;

	//std::string ncfilestr;
	//std::string varstr;


	//char ncfile[]="ocean_ausnwsrstwq2.nc";
	//std::vector<std::string> nameelements;

	/*nameelements = split(filename, '?');
	if (nameelements.size() > 1)
	{
		
		ncfilestr = nameelements[0];
		varstr = nameelements[1];
	}
	else
	{
		
		ncfilestr = filename;
		varstr = "zs";
		checkhh = true;
	}*/

	status = nc_open(filename.c_str(), 0, &ncid);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varid(ncid, varstr.c_str(), &varid);
	if (status != NC_NOERR) handle_error(status);

	status = nc_get_var1_double(ncid, varid, start, zsa);
	if (status != NC_NOERR) handle_error(status);

	

	sferr = nc_get_att_double(ncid, varid, "scale_factor", &scalefac);
	oferr = nc_get_att_double(ncid, varid, "add_offset", &offset);

	// Check if variable is a missing value

	misserr = nc_get_att_double(ncid, varid, "_FillValue", &missing);

	fillerr = nc_get_att_double(ncid, varid, "missingvalue", &fillval);

	if (misserr == NC_NOERR)
	{
		if (zsa[0] == missing)
		{
			zsa[0] = 0.0;
			wet = 0;
		}
	}
	if (fillerr == NC_NOERR)
	{
		if (zsa[0] == fillval)
		{
			zsa[0] = 0.0;
			wet = 0;
		}
	}




	if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
	{
		zsa[0] = zsa[0] * scalefac + offset;
	}

	if (checkhh)
	{
		zza = zsa[0];
		iderr = nc_inq_varid(ncid, "hh", &varid);
		if (iderr == NC_NOERR)
		{
			status = nc_get_var1_double(ncid, varid, start, zsa);
			sferr = nc_get_att_double(ncid, varid, "scale_factor", &scalefac);
			oferr = nc_get_att_double(ncid, varid, "add_offset", &offset);

			// Check if variable is a missing value

			misserr = nc_get_att_double(ncid, varid, "_FillValue", &missing);

			fillerr = nc_get_att_double(ncid, varid, "missingvalue", &fillval);

			if (misserr == NC_NOERR)
			{
				if (zsa[0] == missing)
				{
					zsa[0] = 0.0;
					wet = 0;
				}
			}
			if (fillerr == NC_NOERR)
			{
				if (zsa[0] == fillval)
				{
					zsa[0] = 0.0;
					wet = 0;
				}
			}




			if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
			{
				zsa[0] = zsa[0] * scalefac + offset;
			}

			hha = zsa[0];
			if (hha > eps)
			{
				zsa[0] = zza;
			}
			else
			{
				zsa[0] = 0.0;
				wet = 0;
			}
			
		}


	}



	status = nc_close(ncid);

	free(start);
	//free(count);


	return wet;
}




int readvardata(std::string filename, std::string Varname, int ndims, int hotstep, size_t * ddim, float * vardata)
{
	// function to standardise the way to read netCDF data off a file
	// The role of this function is to offload and simplify the rest of the code


	int nx, ny, nt, status, ncid, varid, sferr, oferr;
	size_t * start, * count;
	float scalefac, offset;

	start = (size_t *)malloc(ndims*sizeof(size_t));
	count = (size_t *)malloc(ndims*sizeof(size_t));


	//
	status = nc_open(filename.c_str(), 0, &ncid);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varid(ncid, Varname.c_str(), &varid);
	if (status != NC_NOERR) handle_error(status);

	if (ndims == 1)
	{
		nx = (int)ddim[0];
		start[0] = 0;
		count[0] =  nx ;



	}
	else if (ndims == 2)
	{
		ny = (int)ddim[0];
		nx = (int)ddim[1];
		start[0] = 0;
		start[1] = 0;

		count[0] = ny;
		count[1] = nx;


	}
	else //(ndim>2)
	{
		nt = (int)ddim[0];
		ny = (int)ddim[1];
		nx = (int)ddim[2];
		start[0] = min(hotstep, nt - 1);
		start[1] = 0;
		start[2] = 0;

		count[0] = 1;
		count[1] = ny;
		count[2] = nx;



	}
	status = nc_get_vara_float(ncid, varid, start, count, vardata);
	if (status != NC_NOERR) handle_error(status);

	if (ndims > 1)
	{
		sferr = nc_get_att_float(ncid, varid, "scale_factor", &scalefac);
		oferr = nc_get_att_float(ncid, varid, "add_offset", &offset);

		if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					vardata[i + j*nx] = vardata[i + j*nx] * scalefac + offset;
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}
	}



	//clean up
	free(start);
	free(count);

	status = nc_close(ncid);

	return status;

}


int readvardataD(std::string filename, std::string Varname, int ndims, int hotstep, size_t * ddim, double * vardata)
{
	// function to standardise the way to read netCDF data off a file
	// The role of this function is to offload and simplify the rest of the code


	int nx, ny, nt, status, ncid, varid, sferr, oferr;
	size_t * start, *count;
	double scalefac, offset;

	start = (size_t *)malloc(ndims*sizeof(size_t));
	count = (size_t *)malloc(ndims*sizeof(size_t));


	//
	status = nc_open(filename.c_str(), 0, &ncid);
	if (status != NC_NOERR) handle_error(status);

	status = nc_inq_varid(ncid, Varname.c_str(), &varid);
	if (status != NC_NOERR) handle_error(status);

	if (ndims == 1)
	{
		nx = (int)ddim[0];
		start[0] = 0;
		count[0] = nx;



	}
	else if (ndims == 2)
	{
		ny = (int)ddim[0];
		nx = (int)ddim[1];
		start[0] = 0;
		start[1] = 0;

		count[0] = ny;
		count[1] = nx;


	}
	else //(ndim>2)
	{
		nt = (int)ddim[0];
		ny = (int)ddim[1];
		nx = (int)ddim[2];
		start[0] = min(hotstep, nt - 1);
		start[1] = 0;
		start[2] = 0;

		count[0] = 1;
		count[1] = ny;
		count[2] = nx;



	}
	status = nc_get_vara_double(ncid, varid, start, count, vardata);
	if (status != NC_NOERR) handle_error(status);

	if (ndims > 1)
	{
		sferr = nc_get_att_double(ncid, varid, "scale_factor", &scalefac);
		oferr = nc_get_att_double(ncid, varid, "add_offset", &offset);

		if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					vardata[i + j*nx] = vardata[i + j*nx] * scalefac + offset;
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}
	}



	//clean up
	free(start);
	free(count);

	status = nc_close(ncid);

	return status;

}


int readhotstartfile(Param XParam, int * leftblk, int *rightblk, int * topblk, int* botblk, double * blockxo, double * blockyo, float * &zs, float * &zb, float * &hh, float *&uu, float * &vv)
{
	int status, zserror, hherror, uuerror, vverror, zberror, sferr, oferr, xerror, yerror;
	int ncid, varid, ndims;
	int dimids[NC_MAX_VAR_DIMS];   /* dimension IDs */
	int nx, ny, nt;
	float * xcoord, *ycoord, * varinfile; // Not necessarily identical to any prevoious ones
	float scalefac = 1.0;
	float offset = 0.0;
	size_t  *ddim;

	// Open the file for read access
	//netCDF::NcFile dataFile(XParam.hotstartfile, NcFile::read);


	//Open NC file
	printf("Open file...");
	status = nc_open(XParam.hotstartfile.c_str(), NC_NOWRITE, &ncid);
	if (status != NC_NOERR) handle_error(status);
	zberror = nc_inq_varid(ncid, "zb", &varid);
	zserror = nc_inq_varid(ncid, "zs", &varid);
	hherror = nc_inq_varid(ncid, "hh", &varid);
	uuerror = nc_inq_varid(ncid, "uu", &varid);
	vverror = nc_inq_varid(ncid, "vv", &varid);



	//by default we assume that the x axis is called "xx" but that is not sure "x" shoudl be accepted and so does "lon" for spherical grid
	// The folowing section figure out which one is in the file and if none exits with the netcdf error
	// default name is "xx"
	std::string xvname = "xx";

	xerror = nc_inq_varid(ncid, xvname.c_str(), &varid);
	if (xerror == -49) // Variable not found
	{
		xvname = "x";
		xerror = nc_inq_varid(ncid, xvname.c_str(), &varid);
		if (xerror == -49) // Variable not found
		{
			xvname = "lon";
			xerror = nc_inq_varid(ncid, xvname.c_str(), &varid);
			if (xerror != NC_NOERR) handle_error(status);
		}
		else if (xerror != NC_NOERR) handle_error(status);


	}
	else if (xerror != NC_NOERR) handle_error(status);

	std::string yvname = "yy";

	yerror = nc_inq_varid(ncid, xvname.c_str(), &varid);
	if (yerror == -49) // Variable not found
	{
		yvname = "y";
		yerror = nc_inq_varid(ncid, xvname.c_str(), &varid);
		if (yerror == -49) // Variable not found
		{
			yvname = "lat";
			xerror = nc_inq_varid(ncid, xvname.c_str(), &varid);
			if (yerror != NC_NOERR) handle_error(status);
		}
		else if (yerror != NC_NOERR) handle_error(status);


	}
	else if (yerror != NC_NOERR) handle_error(status);




	status = nc_close(ncid);


	// First we should read x and y coordinates
	// Just as with other variables we expect the file follow the output naming convention of "xx" and "yy" both as a dimension and a variable
	ndims = readvarinfo(XParam.hotstartfile.c_str(), xvname.c_str(), ddim);
	nx = ddim[0];
	xcoord = (float *)malloc(ddim[0]*sizeof(float)); /// Warning here we assume xx is 1D but may not be in the future

	status = readvardata(XParam.hotstartfile.c_str(), xvname.c_str(), ndims, 0, ddim, xcoord);
	if (status != NC_NOERR) handle_error(status);
	free(ddim);

	//Now read y coordinates
	ndims = readvarinfo(XParam.hotstartfile.c_str(), yvname.c_str(), ddim);
	ny = ddim[0];
	ycoord = (float *)malloc(ddim[0] * sizeof(float)); /// Warning here we assume xx is 1D but may not be in the future

	status = readvardata(XParam.hotstartfile.c_str(), yvname.c_str(), ndims, 0, ddim, ycoord);
	if (status != NC_NOERR) handle_error(status);
	free(ddim);

	//printf("xcoord[0]=%f, ycoord[0]=%f\n", xcoord[0], ycoord[0]);
	// Allocate var in file to temporarily store the variable stored
	varinfile = (float *)malloc(nx*ny * sizeof(float));

	//first check if hotstart has zb
	printf("Found variables: ");
	if (zberror == NC_NOERR)
	{
		printf("zb... ");
		//zb is set

		ndims = readvarinfo(XParam.hotstartfile.c_str(), "zb", ddim);

		status = readvardata(XParam.hotstartfile.c_str(), "zb", ndims, 0, ddim, varinfile);
		if (status != NC_NOERR) handle_error(status);
			//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, nx, ny, xcoord[0], xcoord[nx-1], ycoord[0], ycoord[ny-1], (xcoord[nx-1] - xcoord[0]) / (nx - 1), varinfile, zb);

		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo, blockyo, dummy, zb);

		//because we set the edges around empty blocks we need the set the edges for zs too
		// otherwise we create some gitantic waves at the edges of empty blocks
		setedges(XParam.nblk, leftblk, rightblk, topblk, botblk, zb);

		//status = nc_get_var_float(ncid, varid, zb);
		free(ddim);

	}
	// second check if zs or hh are in teh file


	//zs Section
	if (zserror == NC_NOERR)
	{
		printf(" zs... ");
		ndims = readvarinfo(XParam.hotstartfile.c_str(), "zs", ddim);

		status = readvardata(XParam.hotstartfile.c_str(), "zs", ndims, 0, ddim, varinfile);
		if (status != NC_NOERR) handle_error(status);
		//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
		//printf("zs hotstartfile[1407,435]=%f\n", varinfile[1407+435*nx]);
		//printf("xcoord[0]=%f \t xcoord[nx]=%f \t dx=%f \t nx=%d\n",xcoord[0],xcoord[nx],(xcoord[nx] - xcoord[0]) / (nx - 1),nx);
		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, nx, ny, xcoord[0], xcoord[nx-1], ycoord[0], ycoord[ny-1], (xcoord[nx-1] - xcoord[0]) / (nx - 1), varinfile, zs);

		//printf("zs interpolated[ 8 + 8 * 16 + 9000 * blksize]=%f\n", zs[0]);
		//printf("zs hotstartfile[1407,435]=%f\n", zs[1407 + 435 * nx]);
		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo, blockyo, dummy, zs);
		//because we set the edges around empty blocks we need the set the edges for zs too
		// otherwise we create some gitantic waves at the edges of empty blocks
		setedges(XParam.nblk, leftblk, rightblk, topblk, botblk, zs);

		//check sanity
		for (int bl = 0; bl < XParam.nblk; bl++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int i = 0; i < 16; i++)
				{
					int n = i + j * 16 + bl * 256;
					zs[n] = max(zs[n], zb[n]);
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}

		//status = nc_get_var_float(ncid, varid, zb);
		free(ddim);
	}
	else
	{
		if (zserror == -49)
		{
			//Variable not found
			//It's ok if hh is specified
			printf("zs not found in hotstart file. Looking for hh... ");
		}
		else
		{
			handle_error(zserror);
		}
	}

	//hh section
	if (hherror == NC_NOERR)
	{
		printf("hh... ");
		ndims = readvarinfo(XParam.hotstartfile.c_str(), "hh", ddim);

		status = readvardata(XParam.hotstartfile.c_str(), "hh", ndims, 0, ddim, varinfile);
		if (status != NC_NOERR) handle_error(status);
		//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, nx, ny, xcoord[0], xcoord[nx-1], ycoord[0], ycoord[ny-1], (xcoord[nx-1] - xcoord[0]) / (nx - 1), varinfile, hh);

		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo, blockyo, dummy, hh);

		//because we set the edges around empty blocks we need the set the edges for zs too
		// otherwise we create some gitantic waves at the edges of empty blocks
		setedges(XParam.nblk, leftblk, rightblk, topblk, botblk, hh);
		//if zs was not specified
		if (zserror == -49)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * 256;
						zs[n] = zb[n] + hh[n];
					}

				}
			}
		}
		free(ddim);


	}
	else
	{
		if (zserror == -49 && hherror == -49)
		{
			//Variable not found
			//It's ok if hh is specified
			printf("neither zs nor hh were found in hotstart file. this is not a valid hotstart file. using a cold start instead");
			return 0;
		}
		else
		{
			//hherror ==-49
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * 256;
						hh[n] = max(zs[n] - zb[n],(float) XParam.eps);
					}

				}
			}

		}
	}

	//uu Section

	if (uuerror == NC_NOERR)
	{
		printf("uu... ");
		ndims = readvarinfo(XParam.hotstartfile.c_str(), "uu", ddim);

		status = readvardata(XParam.hotstartfile.c_str(), "uu", ndims, 0, ddim, varinfile);
		if (status != NC_NOERR) handle_error(status);
		//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, nx, ny, xcoord[0], xcoord[nx-1], ycoord[0], ycoord[ny-1], (xcoord[nx-1] - xcoord[0]) / (nx - 1), varinfile, uu);


		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo, blockyo, dummy, uu);
		free(ddim);


	}
	else
	{
		if (uuerror == -49)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * 256;
						uu[n] = 0.0f;
					}
				}
			}
		}
		else
		{
			handle_error(uuerror);
		}
	}

	//vv section

	if (vverror == NC_NOERR)
	{
		printf("vv... ");
		ndims = readvarinfo(XParam.hotstartfile.c_str(), "vv", ddim);

		status = readvardata(XParam.hotstartfile.c_str(), "vv", ndims, 0, ddim, varinfile);
		if (status != NC_NOERR) handle_error(status);
		//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, nx, ny, xcoord[0], xcoord[nx-1], ycoord[0], ycoord[ny-1], (xcoord[nx-1] - xcoord[0]) / (nx - 1), varinfile, vv);


		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo, blockyo, dummy, vv);
		free(ddim);


	}
	else
	{
		if (vverror == -49)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * 256;
						vv[n] = 0.0f;
					}
				}
			}
		}
		else
		{
			handle_error(zserror);
		}
	}
	//status = nc_get_var_float(ncid, hh_id, zb);
	status = nc_close(ncid);

	

	return 1;

}

int readhotstartfileD(Param XParam, int * leftblk, int *rightblk, int * topblk, int* botblk, double * blockxo, double * blockyo, double * &zs, double * &zb, double * &hh, double *&uu, double * &vv)
{
	int status, zserror, hherror, uuerror, vverror, zberror, sferr, oferr;
	int ncid, varid, ndims;
	int dimids[NC_MAX_VAR_DIMS];   /* dimension IDs */
	int nx, ny, nt;
	double * xcoord, *ycoord, *varinfile; // Not necessarily identical to any prevoious ones

	size_t  *ddim;

	// Open the file for read access
	//netCDF::NcFile dataFile(XParam.hotstartfile, NcFile::read);


	//Open NC file
	printf("Open file\n");
	status = nc_open(XParam.hotstartfile.c_str(), NC_NOWRITE, &ncid);
	if (status != NC_NOERR) handle_error(status);
	zberror = nc_inq_varid(ncid, "zb", &varid);
	zserror = nc_inq_varid(ncid, "zs", &varid);
	hherror = nc_inq_varid(ncid, "hh", &varid);
	uuerror = nc_inq_varid(ncid, "uu", &varid);
	vverror = nc_inq_varid(ncid, "vv", &varid);
	status = nc_close(ncid);

	// First we should read x and y coordinates
	// Just as with other variables we expect the file follow the output naming convention of "xx" and "yy" both as a dimension and a variable
	ndims = readvarinfo(XParam.hotstartfile.c_str(), "xx", ddim);
	nx = ddim[0];
	xcoord = (double *)malloc(ddim[0] * sizeof(double)); /// Warning here we assume xx is 1D but may not be in the future

	status = readvardataD(XParam.hotstartfile.c_str(), "xx", ndims, 0, ddim, xcoord);
	if (status != NC_NOERR) handle_error(status);
	free(ddim);

	//Now read y coordinates
	ndims = readvarinfo(XParam.hotstartfile.c_str(), "yy", ddim);
	ny = ddim[0];
	ycoord = (double *)malloc(ddim[0] * sizeof(double)); /// Warning here we assume xx is 1D but may not be in the future

	status = readvardataD(XParam.hotstartfile.c_str(), "yy", ndims, 0, ddim, ycoord);
	if (status != NC_NOERR) handle_error(status);
	free(ddim);


	// Allocate var in file to temporarily store the variable stored
	varinfile = (double *)malloc(nx*ny * sizeof(double));

	//first check if hotstart has zb

	if (zberror == NC_NOERR)
	{
		//zb is set

		ndims = readvarinfo(XParam.hotstartfile.c_str(), "zb", ddim);

		status = readvardataD(XParam.hotstartfile.c_str(), "zb", ndims, 0, ddim, varinfile);
		if (status != NC_NOERR) handle_error(status);
		//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, nx, ny, xcoord[0], xcoord[nx-1], ycoord[0], ycoord[ny-1], (xcoord[nx-1] - xcoord[0]) / (nx - 1), varinfile, zb);

		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo, blockyo, dummy, zb);

		//because we set the edges around empty blocks we need the set the edges for zs too
		// otherwise we create some gitantic waves at the edges of empty blocks
		setedges(XParam.nblk, leftblk, rightblk, topblk, botblk, zb);

		//status = nc_get_var_float(ncid, varid, zb);
		free(ddim);

	}
	// second check if zs or hh are in teh file


	//zs Section
	if (zserror == NC_NOERR)
	{
		ndims = readvarinfo(XParam.hotstartfile.c_str(), "zs", ddim);

		status = readvardataD(XParam.hotstartfile.c_str(), "zs", ndims, 0, ddim, varinfile);
		if (status != NC_NOERR) handle_error(status);
		//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, nx, ny, xcoord[0], xcoord[nx-1], ycoord[0], ycoord[ny-1], (xcoord[nx-1] - xcoord[0]) / (nx - 1), varinfile, zs);

		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo, blockyo, dummy, zs);
		//because we set the edges around empty blocks we need the set the edges for zs too
		// otherwise we create some gitantic waves at the edges of empty blocks
		setedges(XParam.nblk, leftblk, rightblk, topblk, botblk, zs);

		//check sanity
		for (int bl = 0; bl < XParam.nblk; bl++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int i = 0; i < 16; i++)
				{
					int n = i + j * 16 + bl * 256;
					zs[n] = max(zs[n], zb[n]);
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}

		//status = nc_get_var_float(ncid, varid, zb);
		free(ddim);
	}
	else
	{
		if (zserror == -49)
		{
			//Variable not found
			//It's ok if hh is specified
			printf("zs not found in hotstart file. Looking for hh\n");
		}
		else
		{
			handle_error(zserror);
		}
	}

	//hh section
	if (hherror == NC_NOERR)
	{
		ndims = readvarinfo(XParam.hotstartfile.c_str(), "hh", ddim);

		status = readvardataD(XParam.hotstartfile.c_str(), "hh", ndims, 0, ddim, varinfile);
		if (status != NC_NOERR) handle_error(status);
		//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, nx, ny, xcoord[0], xcoord[nx-1], ycoord[0], ycoord[ny-1], (xcoord[nx-1] - xcoord[0]) / (nx - 1), varinfile, hh);

		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo, blockyo, dummy, hh);

		//because we set the edges around empty blocks we need the set the edges for zs too
		// otherwise we create some gitantic waves at the edges of empty blocks
		setedges(XParam.nblk, leftblk, rightblk, topblk, botblk, hh);
		//if zs was not specified
		if (zserror == -49)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * 256;
						zs[n] = zb[n] + hh[n];
					}

				}
			}
		}
		free(ddim);


	}
	else
	{
		if (zserror == -49 && hherror == -49)
		{
			//Variable not found
			//It's ok if hh is specified
			printf("neither zs nor hh were found in hotstart file. this is not a valid hotstart file. using a cold start instead");
			return 0;
		}
		else
		{
			//hherror ==-49
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * 256;
						hh[n] = max(zs[n] - zb[n], XParam.eps);
					}

				}
			}

		}
	}

	//uu Section

	if (uuerror == NC_NOERR)
	{
		ndims = readvarinfo(XParam.hotstartfile.c_str(), "uu", ddim);

		status = readvardataD(XParam.hotstartfile.c_str(), "uu", ndims, 0, ddim, varinfile);
		if (status != NC_NOERR) handle_error(status);
		//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, nx, ny, xcoord[0], xcoord[nx-1], ycoord[0], ycoord[ny-1], (xcoord[nx-1] - xcoord[0]) / (nx - 1), varinfile, uu);


		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo, blockyo, dummy, uu);
		free(ddim);


	}
	else
	{
		if (uuerror == -49)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * 256;
						uu[n] = 0.0f;
					}
				}
			}
		}
		else
		{
			handle_error(zserror);
		}
	}

	//vv section

	if (vverror == NC_NOERR)
	{
		ndims = readvarinfo(XParam.hotstartfile.c_str(), "vv", ddim);

		status = readvardataD(XParam.hotstartfile.c_str(), "vv", ndims, 0, ddim, varinfile);
		if (status != NC_NOERR) handle_error(status);
		//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, nx, ny, xcoord[0], xcoord[nx-1], ycoord[0], ycoord[ny-1], (xcoord[nx-1] - xcoord[0]) / (nx - 1), varinfile, vv);


		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo, blockyo, dummy, vv);
		free(ddim);


	}
	else
	{
		if (vverror == -49)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * 256;
						vv[n] = 0.0f;
					}
				}
			}
		}
		else
		{
			handle_error(zserror);
		}
	}
	//status = nc_get_var_float(ncid, hh_id, zb);
	status = nc_close(ncid);


	return 1;

}


void readWNDstep(forcingmap WNDUmap, forcingmap WNDVmap, int steptoread, float *&Uo, float *&Vo)
{
	//
	int status;
	int ncid;
	float NanValU = -9999, NanValV = -9999, NanValH = -9999;
	int uu_id, vv_id;
	// step to read should be adjusted in each variables so that it keeps using the last output and teh model keeps on going
	// right now the model will catch anexception
	printf("Reading Wind data step: %d ...", steptoread);
	//size_t startl[]={hdstep-1,lev,0,0};
	//size_t countlu[]={1,1,netau,nxiu};
	//size_t countlv[]={1,1,netav,nxiv};
	size_t startl[] = { steptoread, 0, 0 };
	size_t countlu[] = { 1, WNDUmap.ny, WNDUmap.nx };
	size_t countlv[] = { 1, WNDVmap.ny, WNDVmap.nx };

	//static ptrdiff_t stridel[]={1,1,1,1};
	static ptrdiff_t stridel[] = { 1, 1, 1 };

	std::string ncfilestrU, ncfilestrV;
	std::string Uvarstr, Vvarstr;


	//char ncfile[]="ocean_ausnwsrstwq2.nc";
	std::vector<std::string> nameelements;
	//by default we expect tab delimitation
	nameelements = split(WNDUmap.inputfile, '?');
	if (nameelements.size() > 1)
	{
		//variable name for bathy is not given so it is assumed to be zb
		ncfilestrU = nameelements[0];
		Uvarstr = nameelements[1];
	}
	else
	{
		ncfilestrU = WNDUmap.inputfile;
		Uvarstr = "uwnd";
	}

	nameelements = split(WNDVmap.inputfile, '?');
	if (nameelements.size() > 1)
	{
		//variable name for bathy is not given so it is assumed to be zb
		ncfilestrV = nameelements[0];
		Vvarstr = nameelements[1];
	}
	else
	{
		ncfilestrV = WNDVmap.inputfile;
		Vvarstr = "vwnd";
	}


	//Open NC file

	status = nc_open(ncfilestrU.c_str(), 0, &ncid);
	if (status != NC_NOERR) handle_error(status);

	//status = nc_inq_varid (ncid, "u", &uu_id);
	status = nc_inq_varid(ncid, Uvarstr.c_str(), &uu_id);
	if (status != NC_NOERR) handle_error(status);

	status = nc_get_vara_float(ncid, uu_id, startl, countlu, Uo);
	if (status != NC_NOERR) handle_error(status);

	//status = nc_get_att_float(ncid, uu_id, "_FillValue", &NanValU);
	//if (status != NC_NOERR) handle_error(status);

	status = nc_close(ncid);
	if (status != NC_NOERR) handle_error(status);

	status = nc_open(ncfilestrV.c_str(), 0, &ncid);
	if (status != NC_NOERR) handle_error(status);
	//status = nc_inq_varid (ncid, "v", &vv_id);
	status = nc_inq_varid(ncid, Vvarstr.c_str(), &vv_id);
	if (status != NC_NOERR) handle_error(status);

	status = nc_get_vara_float(ncid, vv_id, startl, countlv, Vo);
	if (status != NC_NOERR) handle_error(status);

	//status = nc_get_att_float(ncid, vv_id, "_FillValue", &NanValV);
	//if (status != NC_NOERR) handle_error(status);

	status = nc_close(ncid);
	if (status != NC_NOERR) handle_error(status);
	printf("Done!\n");

}

void readATMstep(forcingmap ATMPmap, int steptoread, float *&Po)
{
	//
	int status;
	int ncid;
	float NanValU = -9999, NanValV = -9999, NanValH = -9999;
	int uu_id, vv_id;
	// step to read should be adjusted in each variables so that it keeps using the last output and teh model keeps on going
	// right now the model will catch anexception
	printf("Reading atm pressure data. step: %d ...", steptoread);
	//size_t startl[]={hdstep-1,lev,0,0};
	//size_t countlu[]={1,1,netau,nxiu};
	//size_t countlv[]={1,1,netav,nxiv};
	size_t startl[] = { steptoread, 0, 0 };
	size_t countlu[] = { 1, ATMPmap.ny, ATMPmap.nx };
	//size_t countlv[] = { 1, WNDVmap.ny, WNDVmap.nx };

	//static ptrdiff_t stridel[]={1,1,1,1};
	static ptrdiff_t stridel[] = { 1, 1, 1 };

	std::string ncfilestr;
	std::string atmpvarstr;


	//char ncfile[]="ocean_ausnwsrstwq2.nc";
	std::vector<std::string> nameelements;
	//by default we expect tab delimitation
	nameelements = split(ATMPmap.inputfile, '?');
	if (nameelements.size() > 1)
	{
		//variable name for bathy is not given so it is assumed to be zb
		ncfilestr = nameelements[0];
		atmpvarstr = nameelements[1];
	}
	else
	{
		ncfilestr = ATMPmap.inputfile;
		atmpvarstr = "atmP";
	}


	//Open NC file

	status = nc_open(ncfilestr.c_str(), 0, &ncid);
	if (status != NC_NOERR) handle_error(status);

	//status = nc_inq_varid (ncid, "u", &uu_id);
	status = nc_inq_varid(ncid, atmpvarstr.c_str(), &uu_id);
	if (status != NC_NOERR) handle_error(status);

	status = nc_get_vara_float(ncid, uu_id, startl, countlu, Po);
	if (status != NC_NOERR) handle_error(status);

	//status = nc_get_att_float(ncid, uu_id, "_FillValue", &NanValU);
	//if (status != NC_NOERR) handle_error(status);

	status = nc_close(ncid);

	printf("Done!\n");

}

void InterpstepCPU(int nx, int ny,  int hdstep, float totaltime, float hddt, float *&Ux, float *Uo, float *Un)
{
	//float fac = 1.0;
	float Uxo, Uxn;

	/*Ums[tx]=Umask[ix];*/




	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			Uxo = Uo[i + nx*j];
			Uxn = Un[i + nx*j];

			Ux[i + nx*j] = Uxo + (totaltime - hddt*hdstep)*(Uxn - Uxo) / hddt;
		}
	}
}

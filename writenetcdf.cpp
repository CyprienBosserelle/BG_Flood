//////////////////////////////////////////////////////////////////////////////////
//XBeach_GPU                                                                    //
//Copyright (C) 2013 Bosserelle                                                 //
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


void handle_error(int status) {
	if (status != NC_NOERR) {
		fprintf(stderr, "Netcdf %s\n", nc_strerror(status));
		
		//fprintf(logfile, "Netcdf: %s\n", nc_strerror(status));
		exit(-1);
	}
}


extern "C" void creatncfileUD(std::string outfile, int nx, int ny, double dx, double totaltime)
{
	int status;
	
	//double dx = XParam.dx;
	size_t nxx, nyy, nth;
	int ncid, xx_dim, yy_dim, time_dim;
	float * xval, *yval, *thetaval;
	static size_t xcount[] = { nx };
	static size_t ycount[] = { ny };

	int time_id, xx_id, yy_id, th_id;

	static size_t tst[] = { 0 };
	static size_t xstart[] = { 0 }; // start at first value
	static size_t ystart[] = { 0 }; // start at first value 
	static size_t thstart[] = { 0 }; // start at first value
	nxx = nx;
	nyy = ny;
	

	//Recreat the x, y and theta array
	xval = (float *)malloc(nx*sizeof(float));
	yval = (float *)malloc(ny*sizeof(float));
	

	for (int i = 0; i<nx; i++)
	{
		xval[i] = i*dx;
	}
	for (int i = 0; i<ny; i++)
	{
		yval[i] = i*dx;
	}
	

	//create the netcdf datasetXParam.outfile.c_str()
	status = nc_create(outfile.c_str(), NC_NOCLOBBER, &ncid);

	//Define dimensions: Name and length

	status = nc_def_dim(ncid, "xx", nxx, &xx_dim);
	status = nc_def_dim(ncid, "yy", nyy, &yy_dim);
	
	//status = nc_def_dim(ncid, "npart",nnpart,&p_dim);
	status = nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dim);

	int tdim[] = { time_dim };
	int xdim[] = { xx_dim };
	int ydim[] = { yy_dim };
	
	
	status = nc_def_var(ncid, "time", NC_FLOAT, 1, tdim, &time_id);
	status = nc_def_var(ncid, "xx", NC_FLOAT, 1, xdim, &xx_id);
	status = nc_def_var(ncid, "yy", NC_FLOAT, 1, ydim, &yy_id);
	
	//End definitions: leave define mode
	status = nc_enddef(ncid);

	//Provide values for variables
	status = nc_put_var1_double(ncid, time_id, tst, &totaltime);
	status = nc_put_vara_float(ncid, xx_id, xstart, xcount, xval);
	status = nc_put_vara_float(ncid, yy_id, ystart, ycount, yval);
	
	//close and save new file
	status = nc_close(ncid);

	free(xval);
	free(yval);
	
}

extern "C" void defncvar(std::string outfile, int smallnc, float scalefactor, float addoffset, int nx, int ny, std::string varst, int vdim, float * var)
{
	int status;
	int ncid, var_id;
	int  var_dimid2D[2];
	int  var_dimid3D[3];
	int  var_dimid4D[4];

	int * var_s;
	int recid, xid, yid, thid;
	size_t ntheta;// nx and ny are stored in XParam not yet for ntheta

	status = nc_open(outfile.c_str(), NC_WRITE, &ncid);
	status = nc_redef(ncid);

	//Inquire dimensions ids
	status = nc_inq_unlimdim(ncid, &recid);//time
	status = nc_inq_dimid(ncid, "xx", &xid);
	status = nc_inq_dimid(ncid, "yy", &yid);
	
	

	var_dimid2D[0] = yid;
	var_dimid2D[1] = xid;

	var_dimid3D[0] = recid;
	var_dimid3D[1] = yid;
	var_dimid3D[2] = xid;

	float fillval = 9.9692e+36f;

	static size_t start2D[] = { 0, 0 }; // start at first value 
	static size_t count2D[] = { ny, nx };

	static size_t start3D[] = { 0, 0, 0 }; // start at first value 
	static size_t count3D[] = { 1, ny, nx };
	
	if (smallnc > 0)
	{
		//If saving as short than we first need to scale and shift the data
		var_s = (int *)malloc(nx*ny*sizeof(int));

		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				// packed_data_value = nint((unpacked_data_value - add_offset) / scale_factor)
				var_s[i + nx*j] = (int)round((var[i + nx*j] - addoffset) / scalefactor);
			}
		}
	}

	if (vdim == 2)
	{
		if (smallnc > 0)
		{
			status = nc_def_var(ncid, varst.c_str(), NC_INT, vdim, var_dimid2D, &var_id);

			status = nc_put_att_float(ncid, var_id, "scale_factor", NC_FLOAT, 1, &scalefactor);
			status = nc_put_att_float(ncid, var_id, "add_offset", NC_FLOAT, 1, &addoffset);
			status = nc_put_att_float(ncid, var_id, "_FillValue", NC_FLOAT, 1, &fillval);
			status = nc_put_att_float(ncid, var_id, "missingvalue", NC_FLOAT, 1, &fillval);
			status = nc_enddef(ncid);

			status = nc_put_vara_int(ncid, var_id, start2D, count2D, var_s);

		}
		else
		{
			status = nc_def_var(ncid, varst.c_str(), NC_FLOAT, vdim, var_dimid2D, &var_id);
			status = nc_put_att_float(ncid, var_id, "_FillValue", NC_FLOAT, 1, &fillval);
			status = nc_put_att_float(ncid, var_id, "missingvalue", NC_FLOAT, 1, &fillval);
			status = nc_enddef(ncid);
			status = nc_put_vara_float(ncid, var_id, start2D, count2D, var);
		}

		
	}
	if (vdim == 3)
	{
		if (smallnc > 0)
		{
			status = nc_def_var(ncid, varst.c_str(), NC_INT, vdim, var_dimid3D, &var_id);
			status = nc_put_att_float(ncid, var_id, "scale_factor", NC_FLOAT, 1, &scalefactor);
			status = nc_put_att_float(ncid, var_id, "add_offset", NC_FLOAT, 1, &addoffset);
			status = nc_put_att_float(ncid, var_id, "_FillValue", NC_FLOAT, 1, &fillval);
			status = nc_put_att_float(ncid, var_id, "missingvalue", NC_FLOAT, 1, &fillval);
			status = nc_enddef(ncid);

			status = nc_put_vara_int(ncid, var_id, start3D, count3D, var_s);

		}
		else
		{
			status = nc_def_var(ncid, varst.c_str(), NC_FLOAT, vdim, var_dimid3D, &var_id);
			status = nc_put_att_float(ncid, var_id, "_FillValue", NC_FLOAT, 1, &fillval);
			status = nc_put_att_float(ncid, var_id, "missingvalue", NC_FLOAT, 1, &fillval);
			status = nc_enddef(ncid);
			status = nc_put_vara_float(ncid, var_id, start3D, count3D, var);
		}
	}
	
	if (smallnc > 0)
	{
		free(var_s);
	}
	//close and save new file
	status = nc_close(ncid);

}


extern "C" void writenctimestep(std::string outfile,  double totaltime)
{
	int status, ncid, recid, time_id;
	status = nc_open(outfile.c_str(), NC_WRITE, &ncid);
	static size_t nrec;
	static size_t tst[] = { 0 };
	//read id from time dimension
	status = nc_inq_unlimdim(ncid, &recid);
	status = nc_inq_dimlen(ncid, recid, &nrec);
	status = nc_inq_varid(ncid, "time", &time_id);
	tst[0] = nrec;
	status = nc_put_var1_double(ncid, time_id, tst, &totaltime);
	//close and save
	status = nc_close(ncid);

}

extern "C" void writencvarstep(std::string outfile, int smallnc, float scalefactor, float addoffset, std::string varst, float * var)
{
	int status, ncid,time_dim, recid, var_id,ndims;
	static size_t nrec;
	short * var_s;
	int nx, ny;
	int dimids[NC_MAX_VAR_DIMS];
	size_t  *ddim, *start,*count;
//XParam.outfile.c_str()
	status = nc_open(outfile.c_str(), NC_WRITE, &ncid);

	//read id from time dimension
	status = nc_inq_unlimdim(ncid, &recid);
	status = nc_inq_dimlen(ncid, recid, &nrec);

	status = nc_inq_varid(ncid, varst.c_str(), &var_id);

	status = nc_inq_varndims(ncid, var_id, &ndims);
	if (status != NC_NOERR) handle_error(status);
	//printf("hhVar:%d dims\n", ndimshh);

	status = nc_inq_vardimid(ncid, var_id, dimids);
	if (status != NC_NOERR) handle_error(status);

	ddim = (size_t *)malloc(ndims*sizeof(size_t));
	start = (size_t *)malloc(ndims*sizeof(size_t));
	count = (size_t *)malloc(ndims*sizeof(size_t));

	//Read dimensions nx_u ny_u 
	for (int iddim = 0; iddim < ndims; iddim++)
	{
		status = nc_inq_dimlen(ncid, dimids[iddim], &ddim[iddim]);
		if (status != NC_NOERR) handle_error(status);
		start[iddim] = 0;
		count[iddim] = ddim[iddim];
		//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
	}

	start[0] = nrec-1;
	count[0] = 1;
	if (smallnc > 0)
	{
		nx = count[ndims - 1];
		ny = count[ndims - 2];//yuk!

		//printf("nx=%d\tny=%d\n", nx, ny);
		//If saving as short than we first need to scale and shift the data
		var_s = (short *)malloc(nx*ny*sizeof(short));

		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < ny; j++)
			{
				// packed_data_value = nint((unpacked_data_value - add_offset) / scale_factor)
				var_s[i + nx*j] = (short) round((var[i + nx*j] - addoffset) / scalefactor);
				//printf("var=%f\tvar_s=%d\n", var[i + nx*j],var_s[i + nx*j]);
			}
		}
		status = nc_put_vara_short(ncid, var_id, start, count, var_s);
		if (status != NC_NOERR) handle_error(status);
		free(var_s);
	}
	else
	{
		status = nc_put_vara_float(ncid, var_id, start, count, var);
		if (status != NC_NOERR) handle_error(status);
	}
	
	//close and save
	status = nc_close(ncid);

	free(ddim);
	free(start);
	free(count);
}


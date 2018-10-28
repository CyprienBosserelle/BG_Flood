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


void handle_error(int status) {
	
	
	if (status != NC_NOERR) {
		fprintf(stderr, "Netcdf %s\n", nc_strerror(status));
		std::ostringstream stringStream;
		stringStream << nc_strerror(status);
		std::string copyOfStr = stringStream.str();
		write_text_to_log_file("Netcdf error:" + copyOfStr);
		//fprintf(logfile, "Netcdf: %s\n", nc_strerror(status));
		exit(2);
	}
}


Param creatncfileUD(Param XParam)
{
	int status;
	int nx = ceil(XParam.nx / 16.0) * 16.0; // should be (xmax-xo)/dx but xmax is not known yet

	int ny = ceil(XParam.ny / 16.0) * 16.0;
	//double dx = XParam.dx;
	size_t nxx, nyy;
	int ncid, xx_dim, yy_dim, time_dim;
	float * xval, *yval;
	static size_t xcount[] = { nx };
	static size_t ycount[] = { ny };

	int time_id, xx_id, yy_id;

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
		xval[i] = (float) (XParam.xo+i*XParam.dx);
	}
	for (int i = 0; i<ny; i++)
	{
		yval[i] = (float) (XParam.yo + i*XParam.dx);
	}
	

	//create the netcdf datasetXParam.outfile.c_str()
	status = nc_create(XParam.outfile.c_str(), NC_NOCLOBBER, &ncid);
	if (status != NC_NOERR)
	{
		if (status == NC_EEXIST) // File already axist so automatically rename the output file 
		{
			printf("Warning! Outut file name already exist  ");
			write_text_to_log_file("Warning! Outut file name already exist   ");
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
					newname = newname+"."+extvec[nstin];
				}
				newname = newname + "(" + std::to_string(fileinc) + ")" + "." + bathyext;
				XParam.outfile = newname;
				status = nc_create(XParam.outfile.c_str(), NC_NOCLOBBER, &ncid);
				fileinc++;
			}
			printf("New file name: %s  ", XParam.outfile.c_str());
			write_text_to_log_file("New file name: " + XParam.outfile);
			
		}
		else
		{
			// Other error
			handle_error(status);
		}
	}

	// status could be a new error after renaming the file
	if (status != NC_NOERR) handle_error(status);
	//Define dimensions: Name and length

	status = nc_def_dim(ncid, "xx", nxx, &xx_dim);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_dim(ncid, "yy", nyy, &yy_dim);
	if (status != NC_NOERR) handle_error(status);
	//status = nc_def_dim(ncid, "npart",nnpart,&p_dim);
	status = nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dim);
	if (status != NC_NOERR) handle_error(status);
	int tdim[] = { time_dim };
	int xdim[] = { xx_dim };
	int ydim[] = { yy_dim };
	
	
	status = nc_def_var(ncid, "time", NC_FLOAT, 1, tdim, &time_id);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid, "xx", NC_FLOAT, 1, xdim, &xx_id);
	if (status != NC_NOERR) handle_error(status);
	status = nc_def_var(ncid, "yy", NC_FLOAT, 1, ydim, &yy_id);
	if (status != NC_NOERR) handle_error(status);
	//End definitions: leave define mode

	status = nc_enddef(ncid);
	if (status != NC_NOERR) handle_error(status);

	//Provide values for variables
	status = nc_put_var1_double(ncid, time_id, tst, &XParam.totaltime);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara_float(ncid, xx_id, xstart, xcount, xval);
	if (status != NC_NOERR) handle_error(status);
	status = nc_put_vara_float(ncid, yy_id, ystart, ycount, yval);
	if (status != NC_NOERR) handle_error(status);

	//close and save new file
	status = nc_close(ncid);
	if (status != NC_NOERR) handle_error(status);
	free(xval);
	free(yval);

	return XParam;
	
}

extern "C" void defncvar(Param XParam, double * blockxo, double *blockyo, std::string varst, int vdim, float * var)
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

	short * var_s,*varblk_s;
	float * varblk;
	int recid, xid, yid;
	//size_t ntheta;// nx and ny are stored in XParam not yet for ntheta

	status = nc_open(outfile.c_str(), NC_WRITE, &ncid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_redef(ncid);
	if (status != NC_NOERR) handle_error(status);
	//Inquire dimensions ids
	status = nc_inq_unlimdim(ncid, &recid);//time
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_dimid(ncid, "xx", &xid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_dimid(ncid, "yy", &yid);
	if (status != NC_NOERR) handle_error(status);
	

	var_dimid2D[0] = yid;
	var_dimid2D[1] = xid;

	var_dimid3D[0] = recid;
	var_dimid3D[1] = yid;
	var_dimid3D[2] = xid;

	float fillval = 9.9692e+36f;
	short Sfillval = 32767;
//short fillval = 32767
	static size_t start2D[] = { 0, 0 }; // start at first value 
	//static size_t count2D[] = { ny, nx };
	static size_t count2D[] = { 16, 16 };

	static size_t start3D[] = { 0, 0, 0 }; // start at first value 
	//static size_t count3D[] = { 1, ny, nx };
	static size_t count3D[] = { 1, 16, 16 };

	varblk = (float *)malloc(XParam.blksize * sizeof(float));
	
	if (smallnc > 0)
	{
		//If saving as short than we first need to scale and shift the data
		var_s = (short *)malloc(XParam.nblk*XParam.blksize *sizeof(short));
		varblk_s = (short *)malloc(XParam.blksize * sizeof(short));
		for (int bl = 0; bl < XParam.nblk; bl++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int i = 0; i < 16; i++)
				{
					int n = i + j * 16 + bl * XParam.blksize;
					// packed_data_value = nint((unpacked_data_value - add_offset) / scale_factor)
					var_s[n] = (short)round((var[n] - addoffset) / scalefactor);
				}
			}
		}
	}

	if (vdim == 2)
	{
		if (smallnc > 0)
		{
			
			status = nc_def_var(ncid, varst.c_str(), NC_SHORT, vdim, var_dimid2D, &var_id);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "scale_factor", NC_FLOAT, 1, &scalefactor);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "add_offset", NC_FLOAT, 1, &addoffset);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_short(ncid, var_id, "_FillValue", NC_SHORT, 1, &Sfillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_short(ncid, var_id, "missingvalue", NC_SHORT, 1, &Sfillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_enddef(ncid);
			if (status != NC_NOERR) handle_error(status);
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						varblk_s[i + j * 16] = var_s[n];
					}
				}
				//jstart
				start2D[0] = (size_t) round((blockyo[bl] - XParam.yo) / XParam.dx);
				start2D[1] = (size_t) round((blockxo[bl] - XParam.xo) / XParam.dx);
				status = nc_put_vara_short(ncid, var_id, start2D, count2D, varblk_s);
				if (status != NC_NOERR) handle_error(status);
			}
		}
		else
		{
			status = nc_def_var(ncid, varst.c_str(), NC_FLOAT, vdim, var_dimid2D, &var_id);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "_FillValue", NC_FLOAT, 1, &fillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "missingvalue", NC_FLOAT, 1, &fillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_enddef(ncid);
			if (status != NC_NOERR) handle_error(status);
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						varblk[i + j * 16] = var[n];
					}
				}
				start2D[0] = (size_t)round((blockyo[bl] - XParam.yo) / XParam.dx);
				start2D[1] = (size_t)round((blockxo[bl] - XParam.xo) / XParam.dx);
				status = nc_put_vara_float(ncid, var_id, start2D, count2D, varblk);
				if (status != NC_NOERR) handle_error(status);
			}
		}

		
	}
	if (vdim == 3)
	{
		if (smallnc > 0)
		{
			status = nc_def_var(ncid, varst.c_str(), NC_SHORT, vdim, var_dimid3D, &var_id);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "scale_factor", NC_FLOAT, 1, &scalefactor);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "add_offset", NC_FLOAT, 1, &addoffset);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_short(ncid, var_id, "_FillValue", NC_SHORT, 1, &Sfillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_short(ncid, var_id, "missingvalue", NC_SHORT, 1, &Sfillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_enddef(ncid);
			if (status != NC_NOERR) handle_error(status);
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						varblk_s[i + j * 16] = var_s[n];
					}
				}
				start3D[1] = (size_t)round((blockyo[bl] - XParam.yo) / XParam.dx);
				start3D[2] = (size_t)round((blockxo[bl] - XParam.xo) / XParam.dx);
				status = nc_put_vara_short(ncid, var_id, start3D, count3D, varblk_s);
				if (status != NC_NOERR) handle_error(status);
			}

		}
		else
		{
			status = nc_def_var(ncid, varst.c_str(), NC_FLOAT, vdim, var_dimid3D, &var_id);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "_FillValue", NC_FLOAT, 1, &fillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "missingvalue", NC_FLOAT, 1, &fillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_enddef(ncid);
			if (status != NC_NOERR) handle_error(status);
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						varblk[i + j * 16] = var[n];
					}
				}
				start3D[1] = (size_t)round((blockyo[bl] - XParam.yo) / XParam.dx);
				start3D[2] = (size_t)round((blockxo[bl] - XParam.xo) / XParam.dx);
				status = nc_put_vara_float(ncid, var_id, start3D, count3D, varblk);
				if (status != NC_NOERR) handle_error(status);
			}
		}
	}
	
	if (smallnc > 0)
	{
		free(var_s);
		free(varblk_s);
	}
	free(varblk);
	//close and save new file
	status = nc_close(ncid);
	if (status != NC_NOERR) handle_error(status);

}

extern "C" void defncvarD(Param XParam, double * blockxo, double *blockyo, std::string varst, int vdim, double * var)
{
	std::string outfile = XParam.outfile;
	int smallnc = XParam.smallnc;
	float scalefactor = XParam.scalefactor;
	float addoffset = XParam.addoffset;
	//int nx = XParam.nx;
	//int ny = XParam.ny;
	int status;
	int ncid, var_id;
	int  var_dimid2D[2];
	int  var_dimid3D[3];
	//int  var_dimid4D[4];

	short * var_s, *varblk_s;
	float * varblk;
	int recid, xid, yid;
	//size_t ntheta;// nx and ny are stored in XParam not yet for ntheta

	status = nc_open(outfile.c_str(), NC_WRITE, &ncid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_redef(ncid);
	if (status != NC_NOERR) handle_error(status);
	//Inquire dimensions ids
	status = nc_inq_unlimdim(ncid, &recid);//time
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_dimid(ncid, "xx", &xid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_dimid(ncid, "yy", &yid);
	if (status != NC_NOERR) handle_error(status);


	var_dimid2D[0] = yid;
	var_dimid2D[1] = xid;

	var_dimid3D[0] = recid;
	var_dimid3D[1] = yid;
	var_dimid3D[2] = xid;

	float fillval = 9.9692e+36f;
	short Sfillval = 32767;
	//short fillval = 32767
	static size_t start2D[] = { 0, 0 }; // start at first value 
										//static size_t count2D[] = { ny, nx };
	static size_t count2D[] = { 16, 16 };

	static size_t start3D[] = { 0, 0, 0 }; // start at first value 
										   //static size_t count3D[] = { 1, ny, nx };
	static size_t count3D[] = { 1, 16, 16 };

	varblk = (float *)malloc(XParam.blksize * sizeof(float));

	if (smallnc > 0)
	{
		//If saving as short than we first need to scale and shift the data
		var_s = (short *)malloc(XParam.nblk*XParam.blksize * sizeof(short));
		varblk_s = (short *)malloc(XParam.blksize * sizeof(short));
		for (int bl = 0; bl < XParam.nblk; bl++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int i = 0; i < 16; i++)
				{
					int n = i + j * 16 + bl * XParam.blksize;
					// packed_data_value = nint((unpacked_data_value - add_offset) / scale_factor)
					var_s[n] = (short)round((var[n] - addoffset) / scalefactor);
				}
			}
		}
	}

	if (vdim == 2)
	{
		if (smallnc > 0)
		{

			status = nc_def_var(ncid, varst.c_str(), NC_SHORT, vdim, var_dimid2D, &var_id);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "scale_factor", NC_FLOAT, 1, &scalefactor);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "add_offset", NC_FLOAT, 1, &addoffset);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_short(ncid, var_id, "_FillValue", NC_SHORT, 1, &Sfillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_short(ncid, var_id, "missingvalue", NC_SHORT, 1, &Sfillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_enddef(ncid);
			if (status != NC_NOERR) handle_error(status);
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						varblk_s[i + j * 16] = var_s[n];
					}
				}
				//jstart
				start2D[0] = (size_t)round((blockyo[bl] - XParam.yo) / XParam.dx);
				start2D[1] = (size_t)round((blockxo[bl] - XParam.xo) / XParam.dx);
				status = nc_put_vara_short(ncid, var_id, start2D, count2D, varblk_s);
				if (status != NC_NOERR) handle_error(status);
			}
		}
		else
		{
			status = nc_def_var(ncid, varst.c_str(), NC_FLOAT, vdim, var_dimid2D, &var_id);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "_FillValue", NC_FLOAT, 1, &fillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "missingvalue", NC_FLOAT, 1, &fillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_enddef(ncid);
			if (status != NC_NOERR) handle_error(status);
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						varblk[i + j * 16] = var[n];
					}
				}
				start2D[0] = (size_t)round((blockyo[bl] - XParam.yo) / XParam.dx);
				start2D[1] = (size_t)round((blockxo[bl] - XParam.xo) / XParam.dx);
				status = nc_put_vara_float(ncid, var_id, start2D, count2D, varblk);
				if (status != NC_NOERR) handle_error(status);
			}
		}


	}
	if (vdim == 3)
	{
		if (smallnc > 0)
		{
			status = nc_def_var(ncid, varst.c_str(), NC_SHORT, vdim, var_dimid3D, &var_id);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "scale_factor", NC_FLOAT, 1, &scalefactor);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "add_offset", NC_FLOAT, 1, &addoffset);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_short(ncid, var_id, "_FillValue", NC_SHORT, 1, &Sfillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_short(ncid, var_id, "missingvalue", NC_SHORT, 1, &Sfillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_enddef(ncid);
			if (status != NC_NOERR) handle_error(status);
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						varblk_s[i + j * 16] = var_s[n];
					}
				}
				start3D[1] = (size_t)round((blockyo[bl] - XParam.yo) / XParam.dx);
				start3D[2] = (size_t)round((blockxo[bl] - XParam.xo) / XParam.dx);
				status = nc_put_vara_short(ncid, var_id, start3D, count3D, varblk_s);
				if (status != NC_NOERR) handle_error(status);
			}

		}
		else
		{
			status = nc_def_var(ncid, varst.c_str(), NC_FLOAT, vdim, var_dimid3D, &var_id);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "_FillValue", NC_FLOAT, 1, &fillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_put_att_float(ncid, var_id, "missingvalue", NC_FLOAT, 1, &fillval);
			if (status != NC_NOERR) handle_error(status);
			status = nc_enddef(ncid);
			if (status != NC_NOERR) handle_error(status);
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						varblk[i + j * 16] = var[n];
					}
				}
				start3D[1] = (size_t)round((blockyo[bl] - XParam.yo) / XParam.dx);
				start3D[2] = (size_t)round((blockxo[bl] - XParam.xo) / XParam.dx);
				status = nc_put_vara_float(ncid, var_id, start3D, count3D, varblk);
				if (status != NC_NOERR) handle_error(status);
			}
		}
	}

	if (smallnc > 0)
	{
		free(var_s);
		free(varblk_s);
	}
	free(varblk);
	//close and save new file
	status = nc_close(ncid);
	if (status != NC_NOERR) handle_error(status);

}


extern "C" void writenctimestep(std::string outfile,  double totaltime)
{
	int status, ncid, recid, time_id;
	status = nc_open(outfile.c_str(), NC_WRITE, &ncid);
	if (status != NC_NOERR) handle_error(status);
	static size_t nrec;
	static size_t tst[] = { 0 };
	//read id from time dimension
	status = nc_inq_unlimdim(ncid, &recid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_dimlen(ncid, recid, &nrec);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid(ncid, "time", &time_id);
	if (status != NC_NOERR) handle_error(status);
	tst[0] = nrec;
	status = nc_put_var1_double(ncid, time_id, tst, &totaltime);
	if (status != NC_NOERR) handle_error(status);
	//close and save
	status = nc_close(ncid);
	if (status != NC_NOERR) handle_error(status);
}

extern "C" void writencvarstep(Param XParam, double * blockxo, double *blockyo, std::string varst, float * var)
{
	int status, ncid, recid, var_id,ndims;
	static size_t nrec;
	short * var_s, *varblk_s;
	float * varblk;
	int nx, ny;
	int dimids[NC_MAX_VAR_DIMS];
	size_t  *ddim, *start,*count;
//XParam.outfile.c_str()
	status = nc_open(XParam.outfile.c_str(), NC_WRITE, &ncid);
	if (status != NC_NOERR) handle_error(status);
	//read id from time dimension
	status = nc_inq_unlimdim(ncid, &recid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_dimlen(ncid, recid, &nrec);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid(ncid, varst.c_str(), &var_id);
	if (status != NC_NOERR) handle_error(status);
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
	count[1] = 16;
	count[2] = 16;
	varblk = (float *)malloc(XParam.blksize * sizeof(float));

	if (XParam.smallnc > 0)
	{
		//nx = (int) count[ndims - 1];
		//ny = (int) count[ndims - 2];//yuk!

		//printf("nx=%d\tny=%d\n", nx, ny);
		//If saving as short than we first need to scale and shift the data
		var_s = (short *)malloc(XParam.blksize*XParam.nblk*sizeof(short));
		varblk_s = (short *)malloc(XParam.blksize * sizeof(short));
		for (int bl = 0; bl < XParam.nblk; bl++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int i = 0; i < 16; i++)
				{
					int n = i + j * 16 + bl * XParam.blksize;
					// packed_data_value = nint((unpacked_data_value - add_offset) / scale_factor)
					var_s[n] = (short)round((var[n] - XParam.addoffset) / XParam.scalefactor);
				}
			}
		}
		for (int bl = 0; bl < XParam.nblk; bl++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int i = 0; i < 16; i++)
				{
					int n = i + j * 16 + bl * XParam.blksize;
					varblk_s[i + j * 16] = var_s[n];
				}
			}
			//jstart
			start[1] = (size_t)round((blockyo[bl] - XParam.yo) / XParam.dx);
			start[2] = (size_t)round((blockxo[bl] - XParam.xo) / XParam.dx);
			status = nc_put_vara_short(ncid, var_id, start, count, varblk_s);
			if (status != NC_NOERR) handle_error(status);
		}
		free(var_s);
		free(varblk_s);
	}
	else
	{
		for (int bl = 0; bl < XParam.nblk; bl++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int i = 0; i < 16; i++)
				{
					int n = i + j * 16 + bl * XParam.blksize;
					varblk[i + j * 16] = var[n];
				}
			}
			start[1] = (size_t)round((blockyo[bl] - XParam.yo) / XParam.dx);
			start[2] = (size_t)round((blockxo[bl] - XParam.xo) / XParam.dx);
			status = nc_put_vara_float(ncid, var_id, start, count, varblk);
			if (status != NC_NOERR) handle_error(status);
		}
	}
	
	//close and save
	status = nc_close(ncid);
	if (status != NC_NOERR) handle_error(status);
	free(ddim);
	free(start);
	free(count);
	free(varblk);
}

extern "C" void writencvarstepD(Param XParam, double * blockxo, double *blockyo, std::string varst, double * var_d)
{
	int status, ncid, recid, var_id, ndims;
	static size_t nrec;
	short * var_s, *varblk_s;
	float * varblk;
	int nx, ny;
	int dimids[NC_MAX_VAR_DIMS];
	size_t  *ddim, *start, *count;
	//XParam.outfile.c_str()
	status = nc_open(XParam.outfile.c_str(), NC_WRITE, &ncid);
	if (status != NC_NOERR) handle_error(status);
	//read id from time dimension
	status = nc_inq_unlimdim(ncid, &recid);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_dimlen(ncid, recid, &nrec);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varid(ncid, varst.c_str(), &var_id);
	if (status != NC_NOERR) handle_error(status);
	status = nc_inq_varndims(ncid, var_id, &ndims);
	if (status != NC_NOERR) handle_error(status);
	//printf("hhVar:%d dims\n", ndimshh);

	status = nc_inq_vardimid(ncid, var_id, dimids);
	if (status != NC_NOERR) handle_error(status);

	ddim = (size_t *)malloc(ndims * sizeof(size_t));
	start = (size_t *)malloc(ndims * sizeof(size_t));
	count = (size_t *)malloc(ndims * sizeof(size_t));

	//Read dimensions nx_u ny_u 
	for (int iddim = 0; iddim < ndims; iddim++)
	{
		status = nc_inq_dimlen(ncid, dimids[iddim], &ddim[iddim]);
		if (status != NC_NOERR) handle_error(status);
		start[iddim] = 0;
		count[iddim] = ddim[iddim];
		//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
	}

	start[0] = nrec - 1;
	count[0] = 1;
	count[1] = 16;
	count[2] = 16;
	varblk = (float *)malloc(XParam.blksize * sizeof(float));

	if (XParam.smallnc > 0)
	{
		nx = (int)count[ndims - 1];
		ny = (int)count[ndims - 2];//yuk!

								   //printf("nx=%d\tny=%d\n", nx, ny);
								   //If saving as short than we first need to scale and shift the data
		var_s = (short *)malloc(nx*ny * sizeof(short));
		varblk_s = (short *)malloc(XParam.blksize * sizeof(short));
		for (int bl = 0; bl < XParam.nblk; bl++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int i = 0; i < 16; i++)
				{
					int n = i + j * 16 + bl * XParam.blksize;
					// packed_data_value = nint((unpacked_data_value - add_offset) / scale_factor)
					var_s[n] = (short)round((var_d[n] - XParam.addoffset) / XParam.scalefactor);
				}
			}
		}
		for (int bl = 0; bl < XParam.nblk; bl++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int i = 0; i < 16; i++)
				{
					int n = i + j * 16 + bl * XParam.blksize;
					varblk_s[i + j * 16] = var_s[n];
				}
			}
			//jstart
			start[1] = (size_t)round((blockyo[bl] - XParam.yo) / XParam.dx);
			start[2] = (size_t)round((blockxo[bl] - XParam.xo) / XParam.dx);
			status = nc_put_vara_short(ncid, var_id, start, count, varblk_s);
			if (status != NC_NOERR) handle_error(status);
		}
		free(var_s);
		free(varblk_s);
	}
	else
	{
		for (int bl = 0; bl < XParam.nblk; bl++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int i = 0; i < 16; i++)
				{
					int n = i + j * 16 + bl * XParam.blksize;
					varblk[i + j * 16] = var_d[n];
				}
			}
			start[1] = (size_t)round((blockyo[bl] - XParam.yo) / XParam.dx);
			start[2] = (size_t)round((blockxo[bl] - XParam.xo) / XParam.dx);
			status = nc_put_vara_float(ncid, var_id, start, count, varblk);
			if (status != NC_NOERR) handle_error(status);
		}
	}

	//close and save
	status = nc_close(ncid);
	if (status != NC_NOERR) handle_error(status);
	free(ddim);
	free(start);
	free(count);
	free(varblk);
}

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

	float dxx;
	//check dx
	dxx = (float) abs(xcoord[0] - xcoord[nx - 1]) / (nx - 1);
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

int readhotstartfile(Param XParam, int * leftblk, int *rightblk, int * topblk, int* botblk, double * blockxo, double * blockyo, float * dummy, float * &zs, float * &zb, float * &hh, float *&uu, float * &vv)
{
	int status, zserror, hherror, uuerror, vverror, zberror, sferr, oferr;
	int ncid, varid, ndims;
	int dimids[NC_MAX_VAR_DIMS];   /* dimension IDs */
	int nx, ny, nt;
	float scalefac = 1.0;
	float offset = 0.0;
	size_t  *ddim;




	//Open NC file
	printf("Open file\n");
	status = nc_open(XParam.hotstartfile.c_str(), NC_NOWRITE, &ncid);
	if (status != NC_NOERR) handle_error(status);


	//first check if hotstart has zb 
	zberror = nc_inq_varid(ncid, "zb", &varid);
	if (zberror == NC_NOERR)
	{

		status = nc_inq_varndims(ncid, varid, &ndims);
		if (status != NC_NOERR) handle_error(status);
		//printf("hhVar:%d dims\n", ndimshh);

		status = nc_inq_vardimid(ncid, varid, dimids);
		if (status != NC_NOERR) handle_error(status);

		ddim = (size_t *)malloc(ndims * sizeof(size_t));

		//Read dimensions nx_u ny_u 
		for (int iddim = 0; iddim < ndims; iddim++)
		{
			status = nc_inq_dimlen(ncid, dimids[iddim], &ddim[iddim]);
			if (status != NC_NOERR) handle_error(status);

			//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
		}
		if (ndims > 2)
		{
			nt = (int)ddim[0];
			ny = (int)ddim[1];
			nx = (int)ddim[2];
			size_t start[] = { max(XParam.hotstep,nt - 1), 0, 0 };
			size_t count[] = { 1, ny, nx };
			status = nc_get_vara_float(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);


		}
		else
		{
			ny = (int)ddim[0];
			nx = (int)ddim[1];
			size_t start[] = { 0, 0 };
			size_t count[] = { ny, nx };
			status = nc_get_vara_float(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}

		sferr = nc_get_att_float(ncid, varid, "scale_factor", &scalefac);
		oferr = nc_get_att_float(ncid, varid, "add_offset", &offset);

		if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					dummy[i + j*nx] = dummy[i + j*nx] * scalefac + offset;
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}
		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, dummy, zb);

		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo, blockyo, dummy, zb);
		
		//because we set the edges around empty blocks we need the set the edges for zs too 
		// otherwise we create some gitantic waves at the edges of empty blocks
		setedges(XParam.nblk, leftblk, rightblk, topblk, botblk, zb);

		//status = nc_get_var_float(ncid, varid, zb);
		free(ddim);
	}
	// second check if zs or hh are in teh file 
	zserror = nc_inq_varid(ncid, "zs", &varid);
	if (zserror == NC_NOERR)
	{

		status = nc_inq_varndims(ncid, varid, &ndims);
		if (status != NC_NOERR) handle_error(status);
		//printf("hhVar:%d dims\n", ndimshh);

		status = nc_inq_vardimid(ncid, varid, dimids);
		if (status != NC_NOERR) handle_error(status);

		ddim = (size_t *)malloc(ndims * sizeof(size_t));

		//Read dimensions nx_u ny_u 
		for (int iddim = 0; iddim < ndims; iddim++)
		{
			status = nc_inq_dimlen(ncid, dimids[iddim], &ddim[iddim]);
			if (status != NC_NOERR) handle_error(status);

			//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
		}
		if (ndims > 2)
		{
			nt = (int)ddim[0];
			ny = (int)ddim[1];
			nx = (int)ddim[2];
			size_t start[] = { XParam.hotstep, 0, 0 };
			size_t count[] = { 1, ny, nx };
			status = nc_get_vara_float(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}
		else
		{
			ny = (int)ddim[0];
			nx = (int)ddim[1];
			size_t start[] = { 0, 0 };
			size_t count[] = { ny, nx };
			status = nc_get_vara_float(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}

		sferr = nc_get_att_float(ncid, varid, "scale_factor", &scalefac);
		oferr = nc_get_att_float(ncid, varid, "add_offset", &offset);

		if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					dummy[i + j*nx] = dummy[i + j*nx] * scalefac + offset;
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, dummy, zs);


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
	hherror = nc_inq_varid(ncid, "hh", &varid);
	if (hherror == NC_NOERR)
	{

		status = nc_inq_varndims(ncid, varid, &ndims);
		if (status != NC_NOERR) handle_error(status);
		//printf("hhVar:%d dims\n", ndimshh);

		status = nc_inq_vardimid(ncid, varid, dimids);
		if (status != NC_NOERR) handle_error(status);

		ddim = (size_t *)malloc(ndims * sizeof(size_t));

		//Read dimensions nx_u ny_u 
		for (int iddim = 0; iddim < ndims; iddim++)
		{
			status = nc_inq_dimlen(ncid, dimids[iddim], &ddim[iddim]);
			if (status != NC_NOERR) handle_error(status);

			//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
		}
		if (ndims > 2)
		{
			nt = (int)ddim[0];
			ny = (int)ddim[1];
			nx = (int)ddim[2];
			size_t start[] = { XParam.hotstep, 0, 0 };
			size_t count[] = { 1, ny, nx };
			status = nc_get_vara_float(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}
		else
		{
			ny = (int)ddim[0];
			nx = (int)ddim[1];
			size_t start[] = { 0, 0 };
			size_t count[] = { ny, nx };
			status = nc_get_vara_float(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}

		sferr = nc_get_att_float(ncid, varid, "scale_factor", &scalefac);
		oferr = nc_get_att_float(ncid, varid, "add_offset", &offset);

		if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					dummy[i + j*nx] = dummy[i + j*nx] * scalefac + offset;
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, dummy, hh);

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

	uuerror = nc_inq_varid(ncid, "uu", &varid);
	if (uuerror == NC_NOERR)
	{

		status = nc_inq_varndims(ncid, varid, &ndims);
		if (status != NC_NOERR) handle_error(status);
		//printf("hhVar:%d dims\n", ndimshh);

		status = nc_inq_vardimid(ncid, varid, dimids);
		if (status != NC_NOERR) handle_error(status);

		ddim = (size_t *)malloc(ndims * sizeof(size_t));

		//Read dimensions nx_u ny_u 
		for (int iddim = 0; iddim < ndims; iddim++)
		{
			status = nc_inq_dimlen(ncid, dimids[iddim], &ddim[iddim]);
			if (status != NC_NOERR) handle_error(status);

			//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
		}
		if (ndims > 2)
		{
			nt = (int)ddim[0];
			ny = (int)ddim[1];
			nx = (int)ddim[2];
			size_t start[] = { XParam.hotstep, 0, 0 };
			size_t count[] = { 1, ny, nx };
			status = nc_get_vara_float(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}
		else
		{
			ny = (int)ddim[0];
			nx = (int)ddim[1];
			size_t start[] = { 0, 0 };
			size_t count[] = { ny, nx };
			status = nc_get_vara_float(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}

		sferr = nc_get_att_float(ncid, varid, "scale_factor", &scalefac);
		oferr = nc_get_att_float(ncid, varid, "add_offset", &offset);

		if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					dummy[i + j*nx] = dummy[i + j*nx] * scalefac + offset;
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, dummy, uu);

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

	vverror = nc_inq_varid(ncid, "vv", &varid);
	if (vverror == NC_NOERR)
	{

		status = nc_inq_varndims(ncid, varid, &ndims);
		if (status != NC_NOERR) handle_error(status);
		//printf("hhVar:%d dims\n", ndimshh);

		status = nc_inq_vardimid(ncid, varid, dimids);
		if (status != NC_NOERR) handle_error(status);

		ddim = (size_t *)malloc(ndims * sizeof(size_t));

		//Read dimensions nx_u ny_u 
		for (int iddim = 0; iddim < ndims; iddim++)
		{
			status = nc_inq_dimlen(ncid, dimids[iddim], &ddim[iddim]);
			if (status != NC_NOERR) handle_error(status);

			//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
		}
		if (ndims > 2)
		{
			//nt = (int) ddim[0];
			ny = (int)ddim[1];
			nx = (int)ddim[2];
			size_t start[] = { XParam.hotstep, 0, 0 };
			size_t count[] = { 1, ny, nx };
			status = nc_get_vara_float(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}
		else
		{
			ny = (int)ddim[0];
			nx = (int)ddim[1];
			size_t start[] = { 0, 0 };
			size_t count[] = { ny, nx };
			status = nc_get_vara_float(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}

		sferr = nc_get_att_float(ncid, varid, "scale_factor", &scalefac);
		oferr = nc_get_att_float(ncid, varid, "add_offset", &offset);

		if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					dummy[i + j*nx] = dummy[i + j*nx] * scalefac + offset;
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, dummy, vv);


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

int readhotstartfileD(Param XParam, int * leftblk, int *rightblk, int * topblk, int* botblk, double * blockxo, double * blockyo, double * dummy, double * &zs, double * &zb, double * &hh, double *&uu, double * &vv)
{
	int status, zserror, hherror, uuerror, vverror, zberror, sferr, oferr;
	int ncid, varid, ndims;
	int dimids[NC_MAX_VAR_DIMS];   /* dimension IDs */
	int nx, ny, nt;
	double scalefac = 1.0;
	double offset = 0.0;
	size_t  *ddim;




	//Open NC file
	printf("Open file\n");
	status = nc_open(XParam.hotstartfile.c_str(), NC_NOWRITE, &ncid);
	if (status != NC_NOERR) handle_error(status);


	//first check if hotstart has zb 
	zberror = nc_inq_varid(ncid, "zb", &varid);
	if (zberror == NC_NOERR)
	{

		status = nc_inq_varndims(ncid, varid, &ndims);
		if (status != NC_NOERR) handle_error(status);
		//printf("hhVar:%d dims\n", ndimshh);

		status = nc_inq_vardimid(ncid, varid, dimids);
		if (status != NC_NOERR) handle_error(status);

		ddim = (size_t *)malloc(ndims * sizeof(size_t));

		//Read dimensions nx_u ny_u 
		for (int iddim = 0; iddim < ndims; iddim++)
		{
			status = nc_inq_dimlen(ncid, dimids[iddim], &ddim[iddim]);
			if (status != NC_NOERR) handle_error(status);

			//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
		}
		if (ndims > 2)
		{
			nt = (int)ddim[0];
			ny = (int)ddim[1];
			nx = (int)ddim[2];
			size_t start[] = { max(XParam.hotstep,nt - 1), 0, 0 };
			size_t count[] = { 1, ny, nx };
			status = nc_get_vara_double(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);

			
		}
		else
		{
			ny = (int)ddim[0];
			nx = (int)ddim[1];
			size_t start[] = { 0, 0 };
			size_t count[] = { ny, nx };
			status = nc_get_vara_double(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}

		sferr = nc_get_att_double(ncid, varid, "scale_factor", &scalefac);
		oferr = nc_get_att_double(ncid, varid, "add_offset", &offset);

		if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					dummy[i + j*nx] = dummy[i + j*nx] * scalefac + offset;
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, dummy, zb);

		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo, blockyo, dummy, zb);
		setedges(XParam.nblk, leftblk, rightblk, topblk, botblk, zb);
		//status = nc_get_var_float(ncid, varid, zb);
		free(ddim);
	}
	// second check if zs or hh are in teh file 
	zserror = nc_inq_varid(ncid, "zs", &varid);
	if (zserror == NC_NOERR)
	{

		status = nc_inq_varndims(ncid, varid, &ndims);
		if (status != NC_NOERR) handle_error(status);
		//printf("hhVar:%d dims\n", ndimshh);

		status = nc_inq_vardimid(ncid, varid, dimids);
		if (status != NC_NOERR) handle_error(status);

		ddim = (size_t *)malloc(ndims * sizeof(size_t));

		//Read dimensions nx_u ny_u 
		for (int iddim = 0; iddim < ndims; iddim++)
		{
			status = nc_inq_dimlen(ncid, dimids[iddim], &ddim[iddim]);
			if (status != NC_NOERR) handle_error(status);

			//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
		}
		if (ndims > 2)
		{
			nt = (int)ddim[0];
			ny = (int)ddim[1];
			nx = (int)ddim[2];
			size_t start[] = { XParam.hotstep, 0, 0 };
			size_t count[] = { 1, ny, nx };
			status = nc_get_vara_double(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}
		else
		{
			ny = (int)ddim[0];
			nx = (int)ddim[1];
			size_t start[] = { 0, 0 };
			size_t count[] = { ny, nx };
			status = nc_get_vara_double(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}

		sferr = nc_get_att_double(ncid, varid, "scale_factor", &scalefac);
		oferr = nc_get_att_double(ncid, varid, "add_offset", &offset);

		if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					dummy[i + j*nx] = dummy[i + j*nx] * scalefac + offset;
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}
		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, dummy, zs);


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
	hherror = nc_inq_varid(ncid, "hh", &varid);
	if (hherror == NC_NOERR)
	{

		status = nc_inq_varndims(ncid, varid, &ndims);
		if (status != NC_NOERR) handle_error(status);
		//printf("hhVar:%d dims\n", ndimshh);

		status = nc_inq_vardimid(ncid, varid, dimids);
		if (status != NC_NOERR) handle_error(status);

		ddim = (size_t *)malloc(ndims * sizeof(size_t));

		//Read dimensions nx_u ny_u 
		for (int iddim = 0; iddim < ndims; iddim++)
		{
			status = nc_inq_dimlen(ncid, dimids[iddim], &ddim[iddim]);
			if (status != NC_NOERR) handle_error(status);

			//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
		}
		if (ndims > 2)
		{
			nt = (int)ddim[0];
			ny = (int)ddim[1];
			nx = (int)ddim[2];
			size_t start[] = { XParam.hotstep, 0, 0 };
			size_t count[] = { 1, ny, nx };
			status = nc_get_vara_double(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}
		else
		{
			ny = (int)ddim[0];
			nx = (int)ddim[1];
			size_t start[] = { 0, 0 };
			size_t count[] = { ny, nx };
			status = nc_get_vara_double(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}

		sferr = nc_get_att_double(ncid, varid, "scale_factor", &scalefac);
		oferr = nc_get_att_double(ncid, varid, "add_offset", &offset);

		if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					dummy[i + j*nx] = dummy[i + j*nx] * scalefac + offset;
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, dummy, hh);


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

	uuerror = nc_inq_varid(ncid, "uu", &varid);
	if (uuerror == NC_NOERR)
	{

		status = nc_inq_varndims(ncid, varid, &ndims);
		if (status != NC_NOERR) handle_error(status);
		//printf("hhVar:%d dims\n", ndimshh);

		status = nc_inq_vardimid(ncid, varid, dimids);
		if (status != NC_NOERR) handle_error(status);

		ddim = (size_t *)malloc(ndims * sizeof(size_t));

		//Read dimensions nx_u ny_u 
		for (int iddim = 0; iddim < ndims; iddim++)
		{
			status = nc_inq_dimlen(ncid, dimids[iddim], &ddim[iddim]);
			if (status != NC_NOERR) handle_error(status);

			//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
		}
		if (ndims > 2)
		{
			nt = (int)ddim[0];
			ny = (int)ddim[1];
			nx = (int)ddim[2];
			size_t start[] = { XParam.hotstep, 0, 0 };
			size_t count[] = { 1, ny, nx };
			status = nc_get_vara_double(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}
		else
		{
			ny = (int)ddim[0];
			nx = (int)ddim[1];
			size_t start[] = { 0, 0 };
			size_t count[] = { ny, nx };
			status = nc_get_vara_double(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}

		sferr = nc_get_att_double(ncid, varid, "scale_factor", &scalefac);
		oferr = nc_get_att_double(ncid, varid, "add_offset", &offset);

		if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					dummy[i + j*nx] = dummy[i + j*nx] * scalefac + offset;
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}

		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, dummy, uu);

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
						uu[n] = 0.0;
					}
				}
			}
		}
		else
		{
			handle_error(zserror);
		}
	}

	vverror = nc_inq_varid(ncid, "vv", &varid);
	if (vverror == NC_NOERR)
	{

		status = nc_inq_varndims(ncid, varid, &ndims);
		if (status != NC_NOERR) handle_error(status);
		//printf("hhVar:%d dims\n", ndimshh);

		status = nc_inq_vardimid(ncid, varid, dimids);
		if (status != NC_NOERR) handle_error(status);

		ddim = (size_t *)malloc(ndims * sizeof(size_t));

		//Read dimensions nx_u ny_u 
		for (int iddim = 0; iddim < ndims; iddim++)
		{
			status = nc_inq_dimlen(ncid, dimids[iddim], &ddim[iddim]);
			if (status != NC_NOERR) handle_error(status);

			//printf("dim:%d=%d\n", iddim, ddimhh[iddim]);
		}
		if (ndims > 2)
		{
			//nt = (int) ddim[0];
			ny = (int)ddim[1];
			nx = (int)ddim[2];
			size_t start[] = { XParam.hotstep, 0, 0 };
			size_t count[] = { 1, ny, nx };
			status = nc_get_vara_double(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}
		else
		{
			ny = (int)ddim[0];
			nx = (int)ddim[1];
			size_t start[] = { 0, 0 };
			size_t count[] = { ny, nx };
			status = nc_get_vara_double(ncid, varid, start, count, dummy);
			if (status != NC_NOERR) handle_error(status);
		}

		sferr = nc_get_att_double(ncid, varid, "scale_factor", &scalefac);
		oferr = nc_get_att_double(ncid, varid, "add_offset", &offset);

		if (sferr == NC_NOERR || oferr == NC_NOERR) // data must be packed
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{
					dummy[i + j*nx] = dummy[i + j*nx] * scalefac + offset;
					//unpacked_value = packed_value * scale_factor + add_offset
				}
			}
		}
		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, dummy, vv);
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
						vv[n] = 0.0;
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
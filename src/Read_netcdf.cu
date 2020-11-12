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

#include "Read_netcdf.h"





inline int nc_get_var_T(int ncid, int varid, float * &zb)
{
	int status;
	status = nc_get_var_float(ncid, varid, zb);
	return status;
}
inline int nc_get_var_T(int ncid, int varid, double * &zb)
{
	int status;
	status = nc_get_var_double(ncid, varid, zb);
	return status;
}


inline int nc_get_vara_T(int ncid, int varid, const size_t* startp, const size_t* countp, int*& zb)
{
	int status;
	status = nc_get_vara_int(ncid, varid, startp, countp, zb);
	return status;

}
inline int nc_get_vara_T(int ncid, int varid, const size_t* startp, const size_t* countp, float * &zb)
{
	int status;
	status = nc_get_vara_float(ncid, varid, startp, countp, zb);
	return status;

}
inline int nc_get_vara_T(int ncid, int varid, const size_t* startp, const size_t* countp, double * &zb)
{
	int status;
	status = nc_get_vara_double(ncid, varid, startp, countp, zb);
	return status;

}

inline int nc_get_var1_T(int ncid, int varid, const size_t* startp, float * zsa)
{
	int status;
	status = nc_get_var1_float(ncid, varid, startp, zsa);
	return status;
}
inline int nc_get_var1_T(int ncid, int varid, const size_t* startp, double * zsa)
{
	int status;
	status = nc_get_var1_double(ncid, varid, startp, zsa);
	return status;
}



void readgridncsize(const std::string ncfilestr, const std::string varstr, int &nx, int &ny, int &nt, double &dx, double &xo, double &yo, double &to, double &xmax, double &ymax, double &tmax)
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

	
	//Open NC file
	//printf("Open file\n");
	status = nc_open(ncfilestr.c_str(), NC_NOWRITE, &ncid);
	if (status != NC_NOERR) handle_ncerror(status);


	//printf(" %s...\n", hhvar);
	status = nc_inq_varid(ncid, varstr.c_str(), &varid);
	if (status != NC_NOERR)	handle_ncerror(status);



	status = nc_inq_varndims(ncid, varid, &ndimshh);
	if (status != NC_NOERR) handle_ncerror(status);
	//printf("hhVar:%d dims\n", ndimshh);

	status = nc_inq_vardimid(ncid, varid, dimids);
	if (status != NC_NOERR) handle_ncerror(status);

	ddimhh = (size_t *)malloc(ndimshh*sizeof(size_t));

	//Read dimensions nx_u ny_u
	for (int iddim = 0; iddim < ndimshh; iddim++)
	{
		status = nc_inq_dimlen(ncid, dimids[iddim], &ddimhh[iddim]);
		if (status != NC_NOERR) handle_ncerror(status);

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
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_inq_varid(ncid, coordname, &varid);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_inq_varndims(ncid, varid, &ndims);
	if (status != NC_NOERR) handle_ncerror(status);

	if (ndims < 2)
	{
		double * ytempvar;
		ytempvar = (double *)malloc(ny*sizeof(double));
		size_t start[] = { 0 };
		size_t count[] = { ny };
		status = nc_get_vara_double(ncid, varid, start, count, ytempvar);
		if (status != NC_NOERR) handle_ncerror(status);

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
		if (status != NC_NOERR) handle_ncerror(status);

	}
	//xcoord
	status = nc_inq_dimname(ncid, xcovar, coordname);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_inq_varid(ncid, coordname, &varid);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_inq_varndims(ncid, varid, &ndims);
	if (status != NC_NOERR) handle_ncerror(status);

	if (ndims < 2)
	{
		double * xtempvar;
		xtempvar = (double *)malloc(nx*sizeof(double));
		size_t start[] = { 0 };
		size_t count[] = { nx };
		status = nc_get_vara_double(ncid, varid, start, count, xtempvar);
		if (status != NC_NOERR) handle_ncerror(status);

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
		if (status != NC_NOERR) handle_ncerror(status);

	}

	double dxx;
	//check dx
	dxx = (xcoord[nx - 1] - xcoord[0]) / (nx - 1.0);
	//log("xo=" + std::to_string(xcoord[0])+"; xmax="+ std::to_string(xcoord[nx - 1]) +"; nx="+ std::to_string(nx) +"; dxx=" +std::to_string(dxx));
	//dyy = (float) abs(ycoord[0] - ycoord[(ny - 1)*nx]) / (ny - 1);


	//Read time dimension if any
	if (nt > 0)
	{
		//read dimension name
		status = nc_inq_dimname(ncid, tcovar, coordname);
		if (status != NC_NOERR) handle_ncerror(status);

		//inquire variable id
		status = nc_inq_varid(ncid, coordname, &varid);
		if (status != NC_NOERR) handle_ncerror(status);

		// read the dimension of time variable // yes it should be == 1
		status = nc_inq_varndims(ncid, varid, &ndims);
		if (status != NC_NOERR) handle_ncerror(status);

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
	if (status != NC_NOERR) handle_ncerror(status);

	//inquire variable by name
	//printf("Reading information about %s...", Varname.c_str());
	status = nc_inq_varid(ncid, Varname.c_str(), &varid);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_inq_varndims(ncid, varid, &ndims);
	if (status != NC_NOERR) handle_ncerror(status);


	status = nc_inq_vardimid(ncid, varid, dimids);
	if (status != NC_NOERR) handle_ncerror(status);

	ddimU = (size_t *)malloc(ndims*sizeof(size_t));

	//Read dimensions nx_u ny_u
	for (int iddim = 0; iddim < ndims; iddim++)
	{
		status = nc_inq_dimlen(ncid, dimids[iddim], &ddimU[iddim]);
		if (status != NC_NOERR) handle_ncerror(status);

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
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_inq_varid(ncid, Varname.c_str(), &varid);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_get_var_double(ncid, varid, time);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_close(ncid);

	return status;
}

template <class T>
int readncslev1(std::string filename, std::string varstr, size_t indx, size_t indy, size_t indt, bool checkhh, double eps, T * &zsa)
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
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_inq_varid(ncid, varstr.c_str(), &varid);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_get_var1_T(ncid, varid, start, zsa);
	if (status != NC_NOERR) handle_ncerror(status);

	

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
			if (typeid(T) == typeid(float))
				status = nc_get_var1_T(ncid, varid, start, zsa);
			if (typeid(T) == typeid(double))
				status = nc_get_var1_T(ncid, varid, start, zsa);
			//status = nc_get_var1_double(ncid, varid, start, zsa);
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

template int readncslev1<float>(std::string filename, std::string varstr, size_t indx, size_t indy, size_t indt, bool checkhh, double eps, float * &zsa);
template int readncslev1<double>(std::string filename, std::string varstr, size_t indx, size_t indy, size_t indt, bool checkhh, double eps, double * &zsa);


template <class T>
int readvardata(std::string filename, std::string Varname, int step, T * &vardata)
{
	// function to standardise the way to read netCDF data off a file
	// The role of this function is to offload and simplify the rest of the code


	int nx, ny, nt, status, ncid, varid, sferr, oferr,ndims;
	size_t * start, * count, *ddim;
	double scalefac, offset;

	ndims = readvarinfo(filename, Varname, ddim);

	start = (size_t *)malloc(ndims*sizeof(size_t));
	count = (size_t *)malloc(ndims*sizeof(size_t));


	//
	status = nc_open(filename.c_str(), 0, &ncid);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_inq_varid(ncid, Varname.c_str(), &varid);
	if (status != NC_NOERR) handle_ncerror(status);

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
		start[0] = utils::min(step, nt - 1);
		start[1] = 0;
		start[2] = 0;

		count[0] = 1;
		count[1] = ny;
		count[2] = nx;



	}
	status = nc_get_vara_T(ncid, varid, start, count, vardata);
	if (status != NC_NOERR) handle_ncerror(status);

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
					vardata[i + j*nx] = vardata[i + j*nx] * (T)scalefac + (T)offset;
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
template int readvardata<int>(std::string filename, std::string Varname, int step, int*& vardata);
template int readvardata<float>(std::string filename, std::string Varname, int step, float * &vardata);
template int readvardata<double>(std::string filename, std::string Varname, int step, double * &vardata);




std::string checkncvarname(int ncid, std::string stringA, std::string stringB, std::string stringC, std::string stringD)
{
	int varid;
	int errorA, errorB,errorC,errorD;
	std::string outstring;

	//std::vector<std::string> teststr;

	//teststr.push_back((stringA))


	errorA = nc_inq_varid(ncid, stringA.c_str(), &varid);
	errorB = nc_inq_varid(ncid, stringB.c_str(), &varid);
	errorC = nc_inq_varid(ncid, stringC.c_str(), &varid);
	errorD = nc_inq_varid(ncid, stringD.c_str(), &varid);

	if (errorA == NC_NOERR)
	{
		outstring = stringA;
	}
	else if (errorB == NC_NOERR)
	{
		outstring = stringB;
	}
	else if (errorC == NC_NOERR)
	{
		outstring = stringC;
	}
	else if (errorD == NC_NOERR)
	{
		outstring = stringD;
	}


	return outstring;


}

//By default we want to read wind info as float because it will reside in a texture. the value is converted to the apropriate type only when it is used. so there is no need to template this function 
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
	if (status != NC_NOERR) handle_ncerror(status);

	//status = nc_inq_varid (ncid, "u", &uu_id);
	status = nc_inq_varid(ncid, Uvarstr.c_str(), &uu_id);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_get_vara_float(ncid, uu_id, startl, countlu, Uo);
	if (status != NC_NOERR) handle_ncerror(status);

	//status = nc_get_att_float(ncid, uu_id, "_FillValue", &NanValU);
	//if (status != NC_NOERR) handle_error(status);

	status = nc_close(ncid);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_open(ncfilestrV.c_str(), 0, &ncid);
	if (status != NC_NOERR) handle_ncerror(status);
	//status = nc_inq_varid (ncid, "v", &vv_id);
	status = nc_inq_varid(ncid, Vvarstr.c_str(), &vv_id);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_get_vara_float(ncid, vv_id, startl, countlv, Vo);
	if (status != NC_NOERR) handle_ncerror(status);

	//status = nc_get_att_float(ncid, vv_id, "_FillValue", &NanValV);
	//if (status != NC_NOERR) handle_error(status);

	status = nc_close(ncid);
	if (status != NC_NOERR) handle_ncerror(status);
	printf("Done!\n");

}

//Atm pressure is same as wind we on;ly read floats and that is plenty for real world application
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
	if (status != NC_NOERR) handle_ncerror(status);

	//status = nc_inq_varid (ncid, "u", &uu_id);
	status = nc_inq_varid(ncid, atmpvarstr.c_str(), &uu_id);
	if (status != NC_NOERR) handle_ncerror(status);

	status = nc_get_vara_float(ncid, uu_id, startl, countlu, Po);
	if (status != NC_NOERR) handle_ncerror(status);

	//status = nc_get_att_float(ncid, uu_id, "_FillValue", &NanValU);
	//if (status != NC_NOERR) handle_error(status);

	status = nc_close(ncid);

	printf("Done!\n");

}




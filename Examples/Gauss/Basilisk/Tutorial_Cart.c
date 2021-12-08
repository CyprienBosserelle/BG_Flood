#include "grid/cartesian.h"
#include "saint-venant.h"
#include "netcdf.h"
//#include "terrain.h"

int nx=256;
# define DRY 0.01
double delta;

char ncfile[]="output.nc";
// Declaring interpolation function for later
double interp1DMono(int nx, double *x, double *y, double xx);
double BiInterp(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y) ;

void create2dnc(char * ncfilename,int n, double len, double totaltime, scalar * outvars);
void write2varnc(char * ncfilename,int n, double totaltime, scalar * outvars);

//Needed because somehow gauge output function is not producting anyy output
//This is fixed now (no output when dry)
//FILE *fgauge1, *fgauge2, *fgauge3, *fSbnd;

event init (t = 0) {
  foreach()
	{
			double xo=128.0;
			double a=1.0;
			double yo=128.0;
			//double dd=(x[]-xo)*(x[]-xo) + (y[]-y0)*(y[]-y0);
			double cc = 32;

	    h[] = 0.1 + a*exp(-1.0*((x-xo)*(x-xo) + (y-yo)*(y-yo))/(2.0*cc*cc));
			// function Gaussian(x,a,b,c)
			//     u=x-b;
			//     up=-1.0*u*u/(2*c*c);
			//     return a*exp.(up);
			// end
			eta[] = zb[]+h[];
	}


	//Delta=1.0;
	//delta=Delta;
		scalar * outvars = {eta,h,u.x,u.y};
		create2dnc(ncfile,nx, L0, 0.0, outvars);
}

event end (t = 32) {
  printf ("i = %d t = %g\n", i, t);
}

int main() {
  origin (0.0, 0.0);
	dry = DRY;
	G = 9.81;

//	N = 1 << maxlevel;
	size(256);
	init_grid (nx);

	//DT = 0.0016;


  run();
}

event output_Maps (t+=1)
{
	printf("Output: t=%f DT=%f\n",t,CFL);

	scalar * outvars = {eta,h,u.x,u.y};
	write2varnc(ncfile,nx, t, outvars);
}


// Monotonic 1D interpolation
double interp1DMono(int nx, double *x, double *y, double xx)
{
	//  interpolation if x is monotonically increasing (i.e. the spacing is constant)
	double yy;
	double prevx = x[0];
	double nextx = x[1];
	double prevy = y[0];
 	double nexty = y[1];
 	double dx = nextx - prevx;
 	int indx = 0;
 	int indxp;

 	double diffx = max(xx-x[0],0.0);//This has to be positive!

 	indx = (int)floor(diffx / dx);
  	indxp = (int)min(indx*1.0 + 1, nx*1.0 - 1);
 	prevx = x[indx];
 	nextx = x[indxp];
 	prevy = y[indx];
 	nexty = y[indxp];

 	yy = prevy + (xx - prevx) / (nextx - prevx)*(nexty - prevy);
 	return yy;
}


double BiInterp(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y)
{
    double x2x1, y2y1, x2x, y2y, yy1, xx1;
    x2x1 = x2 - x1;
    y2y1 = y2 - y1;
    x2x = x2 - x;
    y2y = y2 - y;
    yy1 = y - y1;
    xx1 = x - x1;
    return 1.0 / (x2x1 * y2y1) * (
        q11 * x2x * y2y +
        q21 * xx1 * y2y +
        q12 * x2x * yy1 +
        q22 * xx1 * yy1
    );
}

void create2dnc(char * ncfilename,int n, double len, double totaltime, scalar * outvars)
{
	int status;
	int ncid, xx_dim, yy_dim, time_dim, p_dim;

	int * tvar_id;

	int nvars = list_len(outvars);

	tvar_id=(int*)malloc(nvars*sizeof(int));


	double *xx,*yy, *var;

	double dx=len/n;

	xx = (double *)malloc(n*sizeof(double));
	yy = (double *)malloc(n*sizeof(double));
	var = (double *)malloc(n*n*sizeof(double));

	for (int ix=0; ix<n; ix++)
	{
		xx[ix]=ix*dx;
		yy[ix]=ix*dx;
	}

	size_t nxx, nyy, ntt;
	size_t start[] = { 0, 0, 0 }; // start at first value
	size_t count[] = { 1, 1, 1 };
	count[1]=n;
	count[2]=n;


	int time_id, xx_id, yy_id, tt_id;	//
	nxx = n;
	nyy = n;


	//create the netcdf dataset
	status = nc_create(ncfilename, NC_NOCLOBBER, &ncid);

	//Define dimensions: Name and length

	status = nc_def_dim(ncid, "xx", nxx, &xx_dim);
	status = nc_def_dim(ncid, "yy", nyy, &yy_dim);

	status = nc_def_dim(ncid, "time", NC_UNLIMITED, &time_dim);
	int tdim[] = { time_dim };
	int xdim[] = { xx_dim };
	int ydim[] = { yy_dim };


	//define variables: Name, Type,...
	int  var_dimids[3];
	var_dimids[0] = time_dim;

	var_dimids[1] = yy_dim;
	var_dimids[2] = xx_dim;


	status = nc_def_var(ncid, "time", NC_DOUBLE, 1, tdim, &time_id);
	status = nc_def_var(ncid, "xx", NC_DOUBLE, 1, xdim, &xx_id);
	status = nc_def_var(ncid, "yy", NC_DOUBLE, 1, ydim, &yy_id);

	int kk=0;
	for (scalar s in outvars)
	{
		status = nc_def_var(ncid, s.name, NC_DOUBLE, 3, var_dimids, &tvar_id[kk++]);
	}



	status = nc_enddef(ncid);


	size_t tst[] = { 0 };
	size_t xstart[] = { 0 }; // start at first value
	size_t xcount[] = { 1 };
	xcount[0]=n;

	size_t ystart[] = { 0 }; // start at first value
	size_t ycount[] = { 1 };
	ycount[0]=n;





	//Provide values for variables
	status = nc_put_var1_double(ncid, time_id, tst, &totaltime);
	status = nc_put_vara_double(ncid, xx_id, xstart, xcount, xx);
	status = nc_put_vara_double(ncid, yy_id, ystart, ycount, yy);
	kk=0;
	for (scalar s in outvars)
	{
		for (int ii=0; ii<nxx; ii++)
		{
			for (int jj=0;jj<nyy;jj++)
			{
				var[ii+jj*nxx]=interpolate(s,ii*dx+ X0 + dx/2.,jj*dx+Y0+dx/2.);
			}

		}

		status = nc_put_vara_double(ncid, tvar_id[kk++], start, count, var);
	}
	status = nc_close(ncid);

	free(xx);
	free(yy);
	free(var);

}


void write2varnc(char * ncfilename,int n, double totaltime, scalar * outvars)
{
	int status;
	int ncid, time_dim, recid;
	size_t nxx, nyy;
	size_t start[] = { 0, 0, 0 }; // start at first value should be static?
	size_t count[] = { 1, n, n };

	// Is this absurd?
	count[1]=n;
	count[2]=n;

	static size_t tst[] = { 0 };
	int time_id, *var_id;

	int nvars = list_len(outvars);

	double * var;

	var_id=(int*)malloc(nvars*sizeof(int));

	var = (double *)malloc(n*n*sizeof(double));


	nxx = n;
	nyy = n;

	double dx= L0/n;

	static size_t nrec;
	status = nc_open(ncfilename, NC_WRITE, &ncid);

	//read id from time dimension
	status = nc_inq_unlimdim(ncid, &recid);
	status = nc_inq_dimlen(ncid, recid, &nrec);// What is the point of static definition if it is redefined here!
	//printf("nrec=%d\n",nrec);

	//read file for variable ids
	status = nc_inq_varid(ncid, "time", &time_id);
	int kk=0;
	for (scalar s in outvars)
	{
		status = nc_inq_varid(ncid, s.name, &var_id[kk++]);
	}

	start[0] = nrec;//
	tst[0] = nrec;

	//Provide values for variables
	status = nc_put_var1_double(ncid, time_id, tst, &totaltime);
	//status = nc_put_vara_double(ncid, var_id, start, count, var);
	kk=0;
	for (scalar s in outvars)
	{
		for (int ii=0; ii<nxx; ii++)
		{
			for (int jj=0;jj<nyy;jj++)
			{
				var[ii+jj*nxx]=interpolate(s,ii*dx+ X0 + dx/2.,jj*dx+Y0+dx/2.);
			}

		}

		status = nc_put_vara_double(ncid, var_id[kk++], start, count, var);
	}

	status = nc_close(ncid);

free(var);

}

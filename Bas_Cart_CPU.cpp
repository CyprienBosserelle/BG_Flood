//////////////////////////////////////////////////////////////////////////////////
//						                                                        //
//Copyright (C) 2017 Bosserelle                                                 //
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

// includes, system

#define pi 3.14159265

#define epsilon 1e-30


#include <stdio.h>
#include <math.h>
#include <cmath>
#include <ctime>
#include "netcdf.h"

double phi = (1.0f + sqrt(5.0f)) / 2;
double aphi = 1 / (phi + 1);
double bphi = phi / (phi + 1);
double twopi = 8 * atan(1.0f);

double g = 1.0;// 9.81;
double rho = 1025.0;
double eps = 0.0001;
double CFL = 0.5;

double totaltime = 0.0;
double dt, dx;
int nx, ny;

double delta;

double *x, *y;
double *x_g, *y_g;

double *zs, *hh, *zb, *uu, *vv;
double *zso, *hho, *uuo, *vvo;


double * dhdx, *dhdy, *dudx, *dudy, *dvdx, *dvdy;
double *dzsdx, *dzsdy;

double *fmu, *fmv, *Su, *Sv, *Fqux, *Fquy, *Fqvx, *Fqvy;

double * Fhu, *Fhv;

double * dh, *dhu, *dhv;

double dtmax = 1.0 / epsilon;


template <class T> T sq(T a) {
	return (a*a);
}

double sqd(double a) {
	return (a*a);
}

template <class T> const T& max(const T& a, const T& b) {
	return (a<b) ? b : a;     // or: return comp(a,b)?b:a; for version (2)
}

template <class T> const T& min(const T& a, const T& b) {
	return !(b<a) ? a : b;     // or: return comp(a,b)?b:a; for version (2)
}

double minmod2(double s0, double s1, double s2)
{
	double theta = 1.3;
	if (s0 < s1 && s1 < s2) {
		double d1 = theta*(s1 - s0), d2 = (s2 - s0) / 2., d3 = theta*(s2 - s1);
		if (d2 < d1) d1 = d2;
		return min(d1, d3);
	}
	if (s0 > s1 && s1 > s2) {
		double d1 = theta*(s1 - s0), d2 = (s2 - s0) / 2., d3 = theta*(s2 - s1);
		if (d2 > d1) d1 = d2;
		return max(d1, d3);
	}
	return 0.;
}

void gradient(int nx, int ny, double *a, double *&dadx, double * &dady)
{

	int i, xplus, yplus, xminus, yminus;

	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			//
			//
			xplus = min(ix + 1, nx - 1);
			xminus = max(ix - 1, 0);
			yplus = min(iy + 1, ny - 1);
			yminus = max(iy - 1, 0);
			i = ix + iy*nx;


			dadx[i] = minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
			dady[i] = minmod2(a[ix + yminus*nx], a[i], a[ix + yminus*nx]);


		}
	}

}

void kurganov(double hm, double hp, double um, double up, double Delta,	double * fh, double * fq, double * dtmax)
{
	double cp = sqrt(g*hp), cm = sqrt(g*hm);
	double ap = max(up + cp, um + cm); ap = max(ap, 0.);
	double am = min(up - cp, um - cm); am = min(am, 0.);
	double qm = hm*um, qp = hp*up;
	double a = max(ap, -am);
	if (a > epsilon) {
		*fh = (ap*qm - am*qp + ap*am*(hp - hm)) / (ap - am); // (4.5) of [1]
		*fq = (ap*(qm*um + g*sq(hm) / 2.) - am*(qp*up + g*sq(hp) / 2.) +
			ap*am*(qp - qm)) / (ap - am);
		double dt = CFL*Delta / a;
		if (dt < *dtmax)
			*dtmax = dt;
	}
	else
		*fh = *fq = 0.;
}


void neumannbnd(int nx, int ny, double*a)
{
	//
	int i, xplus, yplus, xminus, yminus;
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			i = ix + iy*nx;
			xplus = min(ix + 1, nx - 1);
			xminus = max(ix - 1, 0);
			yplus = min(iy + 1, ny - 1);
			yminus = max(iy - 1, 0);

			if (ix == 0)
			{
				a[i] = a[xplus + iy*nx];
			}
			if (ix = nx - 1)
			{
				a[i] = a[xminus + iy*nx];
			}

			if (iy == 0)
			{
				a[i] = a[ix + yplus*nx];
			}

			if (iy == ny-1)
			{
				a[i] = a[ix + yminus*nx];
			}


		}
	}

}

void update(int nx, int ny, double dt, double eps,double *hh, double *zs, double *uu, double *vv, double *&dh, double *&dhu, double *&dhv)
{
	int i, xplus, yplus, xminus, yminus;

	double hi;

	////calc gradient in h, eta, u and v
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			i = ix + iy*nx;
			xplus = min(ix + 1, nx - 1);
			xminus = max(ix - 1, 0);
			yplus = min(iy + 1, ny - 1);
			yminus = max(iy - 1, 0);


			//dhdx[i] = (hh[i] - hh[xplus + iy*nx]) / delta;
			//dhdy[i] = (hh[i] - hh[ix + yplus*nx]) / delta;

			//dzsdx[i] = (zs[i] - zs[xplus + iy*nx]) / delta;
			//dzsdy[i] = (zs[i] - zs[ix + yplus*nx]) / delta;

			//dudx[i] = (uu[i] - uu[xplus + iy*nx]) / delta;
			//dudy[i] = (uu[i] - uu[ix + yplus*nx]) / delta;

			//dvdx[i] = (vv[i] - vv[xplus + iy*nx]) / delta;
			//dvdy[i] = (vv[i] - vv[ix + yplus*nx]) / delta;

			//dtmax = min(dtmax, delta / (sqrt(g*hh[i])));

		}
	}
	
	dtmax = dt;



	gradient(nx, ny, hh, dhdx, dhdy);
	gradient(nx, ny, zs, dzsdx, dzsdy);
	gradient(nx, ny, uu, dudx, dudy);
	gradient(nx, ny, vv, dvdx, dvdy);
	/////if Hi is dry

	/////
	//for each face
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			i = ix + iy*nx;
			xplus = min(ix + 1, nx - 1);
			xminus = max(ix - 1, 0);
			yplus = min(iy + 1, ny - 1);
			yminus = max(iy - 1, 0);
			hi = hh[i];



			double hn = hh[xminus + iy*nx];


			if (hi > eps || hn > eps)
			{

				double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;

				// along X
				dx = delta / 2.;
				zi = zs[i] - hi;
				zl = zi - dx*(dzsdx[i] - dhdx[i]);
				zn = zs[xminus + iy*nx] - hn;
				zr = zn + dx*(dzsdx[xminus + iy*nx] - dhdx[xminus + iy*nx]);
				zlr = max(zl, zr);

				hl = hi - dx*dhdx[i];
				up = uu[i] - dx*dudx[i];
				hp = max(0., hl + zl - zlr);

				hr = hn + dx*dhdx[xminus + iy*nx];
				um = uu[xminus + iy*nx] + dx*dudx[xminus + iy*nx];
				hm = max(0., hr + zr - zlr);

				//// Reimann solver
				double fh, fu, fv;
				double cm = 0.1;

				//We can now call one of the approximate Riemann solvers to get the fluxes.
				kurganov(hm, hp, um, up, delta*cm / fmu[i], &fh, &fu, &dtmax);
				fv = (fh > 0. ? vv[ix + yminus*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;

				//// Topographic term

				/**
				#### Topographic source term

				In the case of adaptive refinement, care must be taken to ensure
				well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
				double sl = g / 2.*(sq(hp) - sq(hl) + (hl + hi)*(zi - zl));
				double sr = g / 2.*(sq(hm) - sq(hr) + (hr + hn)*(zn - zr));

				////Flux update

				Fhu[i] = fmu[i] * fh;
				Fqux[i] = fmu[i] * (fu - sl);
				Su[i] = fmu[i] * (fu - sr);
				Fqvx[i] = fmu[i] * fv;




				//Along Y
				dx = delta / 2.;
				zi = zs[i] - hi;
				zl = zi - dx*(dzsdy[i] - dhdy[i]);
				zn = zs[ix + yminus*nx] - hn;
				zr = zn + dx*(dzsdy[ix + yminus*nx] - dhdy[ix + yminus*nx]);
				zlr = max(zl, zr);

				hl = hi - dx*dhdy[i];
				up = vv[i] - dx*dvdy[i];
				hp = max(0., hl + zl - zlr);

				hr = hn + dx*dhdy[ix + yminus*nx];
				um = vv[ix + yminus*nx] + dx*dvdy[ix + yminus*nx];
				hm = max(0., hr + zr - zlr);

				//We can now call one of the approximate Riemann solvers to get the fluxes.
				kurganov(hm, hp, um, up, delta*cm / fmv[i], &fh, &fu, &dtmax);
				fv = (fh > 0. ? uu[xminus + iy*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;

				//// Topographic term

				/**
				#### Topographic source term

				In the case of adaptive refinement, care must be taken to ensure
				well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
				sl = g / 2.*(sq(hp) - sq(hl) + (hl + hi)*(zi - zl));
				sr = g / 2.*(sq(hm) - sq(hr) + (hr + hn)*(zn - zr));

				////Flux update

				Fhv[i] = fmv[i] * fh;
				Fqvy[i] = fmv[i] * (fu - sl);
				Sv[i] = fmv[i] * (fu - sr);
				Fquy[i] = fmv[i] * fv;
			}
			else
			{
				Fhu[i] = 0.0;
				Fqux[i] = 0.0;
				Su[i] = 0.0;
				Fqvx[i] = 0.0;

				Fhv[i] = 0.0;
				Fqvy[i] = 0.0;
				Sv[i] = 0.0;
				Fquy[i] = 0.0;
			}


		}
	}


	// UPDATES For evolving quantities

	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{

			i = ix + iy*nx;
			xplus = min(ix + 1, nx - 1);
			xminus = max(ix - 1, 0);
			yplus = min(iy + 1, ny - 1);
			yminus = max(iy - 1, 0);
			hi = hh[i];
			////
			//vector dhu = vector(updates[1 + dimension*l]);
			//foreach() {
			//	double dhl =
			//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
			//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
			//	foreach_dimension()
			//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
			double cm = 1.0;

			dh[i] = -1.0*(Fhu[xplus + iy*nx] - Fhu[i] + Fhv[ix + yplus*nx] - Fhv[i]) / (cm * delta);



			double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
			double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
			double fG = vv[i] * dmdl - uu[i] * dmdt;
			dhu[i] = (Fqux[i] + Fquy[i] - Su[xplus + iy*nx] - Fquy[ix + yplus*nx]) / (cm*delta);
			dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[xplus + iy*nx] - Fqvx[ix + yplus*nx]) / (cm*delta);
			//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
			dhu[i] += hi * (g*hi / 2.*dmdl + fG*vv[i]);
			dhv[i] += hi * (g*hi / 2.*dmdt - fG*uu[i]);

		}
	}
}


void advance(int nx, int ny, double dt, double eps, double *hh, double *zs, double *uu, double * vv, double * dh, double *dhu, double *dhv, double * &hho, double *&zso, double *&uuo, double *&vvo)
{
	//dim3 blockDim(16, 16, 1);
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);

	//adv_stvenant << <gridDim, blockDim, 0 >> > (nx, ny, dt, eps, zb_g, hh_g, zs_g, uu_g, vv_g, dh_g, dhu_g, dhv_g);


	//scalar hi = input[0], ho = output[0], dh = updates[0];
	//vector * uol = (vector *) &output[1];

	// new fields in ho[], uo[]
	//foreach() {
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			int i = ix + iy*nx;
			double hold = hh[i];
			double ho, uo, vo;
			ho = hold + dt*dh[i];
			zso[i] = zb[i] + ho;
			if (ho > eps) {
				//for (int l = 0; l < nl; l++) {
				//vector uo = vector(output[1 + dimension*l]);
				//vector ui = vector(input[1 + dimension*l]),
				//dhu = vector(updates[1 + dimension*l]);
				//foreach_dimension()
				uo = (hold*uu[i] + dt*dhu[i]) / ho;
				vo = (hold*vv[i] + dt*dhv[i]) / ho;
				//}


				//In the case of [multiplelayers](multilayer.h#viscous-friction-between-layers) we add the
				//viscous friction between layers.


			}
			else
			{// dry
			 //for (int l = 0; l < nl; l++) {
			 //vector uo = vector(output[1 + dimension*l]);
			 //foreach_dimension()
				uo = 0.;
				vo = 0.;
			}
			hho[i] = ho;
			uuo[i] = uo;
			vvo[i] = vo;
		}
	}

	// fixme: on trees eta is defined as eta = zb + h and not zb +
	// ho in the refine_eta() and restriction_eta() functions below
	//scalar * list = list_concat({ ho, eta }, (scalar *)uol);
	//boundary(list);
	//free(list);

	// Boundaries!!!!


}


void cleanup(int nx, int ny, double * hhi, double *zsi, double *uui, double *vvi, double * &hho, double *&zso, double *&uuo, double *&vvo)
{
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			int i = ix + iy*nx;
			hho[i] = hhi[i];
			zso[i] = zsi[i];
			uuo[i] = uui[i];
			vvo[i] = vvi[i];
		}
	}

}


extern "C" void create2dnc(int nx, int ny, double dx, double dy, double totaltime, double *xx, double *yy, double * var)
{
	int status;
	int ncid, xx_dim, yy_dim, time_dim, p_dim, tvar_id;

	size_t nxx, nyy, ntt;
	static size_t start[] = { 0, 0, 0 }; // start at first value 
	static size_t count[] = { 1, ny, nx };
	int time_id, xx_id, yy_id, tt_id;	//
	nxx = nx;
	nyy = ny;


	//create the netcdf dataset
	status = nc_create("2Dvar.nc", NC_NOCLOBBER, &ncid);

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



	status = nc_def_var(ncid, "2Dvar", NC_DOUBLE, 3, var_dimids, &tvar_id);


	status = nc_enddef(ncid);


	static size_t tst[] = { 0 };
	static size_t xstart[] = { 0 }; // start at first value 
	static size_t xcount[] = { nx };
	;
	static size_t ystart[] = { 0 }; // start at first value 
	static size_t ycount[] = { ny };






	//Provide values for variables
	status = nc_put_var1_double(ncid, time_id, tst, &totaltime);
	status = nc_put_vara_double(ncid, xx_id, xstart, xcount, xx);
	status = nc_put_vara_double(ncid, yy_id, ystart, ycount, yy);


	status = nc_put_vara_double(ncid, tvar_id, start, count, var);
	status = nc_close(ncid);

}

extern "C" void write2varnc(int nx, int ny, double totaltime, double * var)
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
	status = nc_open("2Dvar.nc", NC_WRITE, &ncid);

	//read id from time dimension
	status = nc_inq_unlimdim(ncid, &recid);
	status = nc_inq_dimlen(ncid, recid, &nrec);
	//printf("nrec=%d\n",nrec);

	//read file for variable ids
	status = nc_inq_varid(ncid, "time", &time_id);
	status = nc_inq_varid(ncid, "2Dvar", &var_id);

	start[0] = nrec;//
	tst[0] = nrec;

	//Provide values for variables
	status = nc_put_var1_double(ncid, time_id, tst, &totaltime);
	status = nc_put_vara_double(ncid, var_id, start, count, var);
	status = nc_close(ncid);

}

// Main loop that actually runs the model
void mainloop()
{

	// list of updates
	//scalar * updates = list_clone(evolving);
	//dt = dtnext(update(evolving, updates, DT));
	//if (gradient != zero) {
	//	/* 2nd-order time-integration */
	//	scalar * predictor = list_clone(evolving);
	//	/* predictor */
	//	advance(predictor, evolving, updates, dt / 2.);
	//	/* corrector */
	//	update(predictor, updates, dt);
	//	delete (predictor);
	//	free(predictor);
	//}
	//advance(evolving, evolving, updates, dt);
	//delete (updates);
	//free(updates);
	//update_perf();
	//iter = inext, t = tnext;
	
	dt = 0.0016;// CFL*delta / sqrt(g*5.0);
	
	
	//update(int nx, int ny, double dt, double eps,double *hh, double *zs, double *uu, double *vv, double *dh, double *dhu, double *dhv)
	update(nx, ny, dt, eps, hh,zs,uu,vv,dh,dhu,dhv);
	
	//predictor
	//advance(int nx, int ny, double dt, double eps, double *hh, double *zs, double *uu, double * vv, double * dh, double *dhu, double *dhv, double * &hho, double *&zso, double *&uuo, double *&vvo)
	advance(nx, ny, dt/2, eps,hh,zs,uu,vv,dh,dhu,dhv,hho,zso,uuo,vvo);

	//corrector
	update(nx, ny, dt, eps, hho, zso, uuo, vvo, dh, dhu, dhv);
	advance(nx, ny, dt, eps, hh, zs, uu, vv, dh, dhu, dhv, hho, zso, uuo, vvo);

	cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);






}





void flowbnd()
{


}


void flowstep()
{

	//advance

	//update

	//advance


}




int main(int argc, char **argv)
{
	//Model starts Here//

	//The main function setups all the init of the model and then calls the mainloop to actually run the model


	//First part reads the inputs to the model 
	//then allocate memory on GPU and CPU
	//Then prepare and initialise memory and arrays on CPU and GPU
	// Prepare output file
	// Run main loop
	// Clean up and close


	// Start timer to keep track of time 
	clock_t startcputime, endcputime;


	startcputime = clock();



	// This is just for temporary use
	nx = 256;
	ny = 256;
	double length = 1.0;
	delta = length / nx;
	

	double *xx, *yy;


	//
	//double * dhdx, *dhdy, *dudx, *dudy, *dvdx, *dvdy;
	//double *dzsdx, *dzsdy;

	//double *fmu, *fmv, *Su, *Sv, *Fqux, *Fquy, *Fqvx, *Fqvy;

	//double * dh, *dhu, *dhv;



	hh = (double *)malloc(nx*ny * sizeof(double));
	uu = (double *)malloc(nx*ny * sizeof(double));
	vv = (double *)malloc(nx*ny * sizeof(double));
	zs = (double *)malloc(nx*ny * sizeof(double));
	zb = (double *)malloc(nx*ny * sizeof(double));

	hho = (double *)malloc(nx*ny * sizeof(double));
	uuo = (double *)malloc(nx*ny * sizeof(double));
	vvo = (double *)malloc(nx*ny * sizeof(double));
	zso = (double *)malloc(nx*ny * sizeof(double));

	dhdx = (double *)malloc(nx*ny * sizeof(double));
	dhdy = (double *)malloc(nx*ny * sizeof(double));
	dudx = (double *)malloc(nx*ny * sizeof(double));
	dudy = (double *)malloc(nx*ny * sizeof(double));
	dvdx = (double *)malloc(nx*ny * sizeof(double));
	dvdy = (double *)malloc(nx*ny * sizeof(double));

	dzsdx = (double *)malloc(nx*ny * sizeof(double));
	dzsdy = (double *)malloc(nx*ny * sizeof(double));


	fmu = (double *)malloc(nx*ny * sizeof(double));
	fmv = (double *)malloc(nx*ny * sizeof(double));
	Su = (double *)malloc(nx*ny * sizeof(double));
	Sv = (double *)malloc(nx*ny * sizeof(double));
	Fqux = (double *)malloc(nx*ny * sizeof(double));
	Fquy = (double *)malloc(nx*ny * sizeof(double));
	Fqvx = (double *)malloc(nx*ny * sizeof(double));
	Fqvy = (double *)malloc(nx*ny * sizeof(double));
	Fhu = (double *)malloc(nx*ny * sizeof(double));
	Fhv = (double *)malloc(nx*ny * sizeof(double));

	dh = (double *)malloc(nx*ny * sizeof(double));
	dhu = (double *)malloc(nx*ny * sizeof(double));
	dhv = (double *)malloc(nx*ny * sizeof(double));

	x = (double *)malloc(nx*ny * sizeof(double));
	xx= (double *)malloc(nx * sizeof(double));
	y = (double *)malloc(nx*ny * sizeof(double));
	yy= (double *)malloc(ny * sizeof(double));

	//init variables
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			zb[i + j*nx] = 0.0;
			uu[i + j*nx] = 0.0;
			vv[i + j*nx] = 0.0;
			x[i + j*nx] = (i-nx/2)*delta+0.5*delta;
			xx[i] = (i - nx / 2)*delta+0.5*delta;
			yy[j] = (j - ny / 2)*delta+0.5*delta;
			y[i + j*nx] = (j-ny/2)*delta + 0.5*delta;
			fmu[i + j*nx] = 1.0;
			fmv[i + j*nx] = 1.0;
		}
	}

	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			double a, b;

			a = sq(x[i + j*nx]) + sq(y[i + j*nx]);
			//b =x[i + j*nx] * x[i + j*nx] + y[i + j*nx] * y[i + j*nx];


			//if (abs(a - b) > 0.00001)
			//{
			//	printf("%f\t%f\n", a, b);
			//}



			hh[i + j*nx] = 0.1 + 1.*exp(-200.*(a));

			zs[i + j*nx] = zb[i + j*nx] + hh[i + j*nx];
		}
	}


	create2dnc(nx, ny, dx, dx, 0.0, xx, yy, hh);
	
	while (totaltime < 0.1)
	{
		mainloop();
		totaltime = totaltime + dt;
		write2varnc(nx, ny, totaltime, hh);
	}
	

	



	endcputime = clock();
	printf("End Computation");
	printf("Total runtime= %d  seconds\n", (endcputime - startcputime) / CLOCKS_PER_SEC);











}


//////////////////////////////////////////////////////////////////////////////////
//						                                                        //
//Copyright (C) 2017 Bosserelle                                                 //
//                                                                              //
// This code is an adaptation of the St Venant equation from Basilisk			//
// See																			//
// http://basilisk.fr/src/saint-venant.h and									//
// S. Popinet. Quadtree-adaptive tsunami modelling. Ocean Dynamics,				//
// doi: 61(9) : 1261 - 1285, 2011												//	
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



#include "Header.cuh"




template <class T> T sq(T a) {
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
	//theta should be used as a global var 
	// can be used to tune the limiting (theta=1
	//gives minmod, the most dissipative limiter and theta = 2 gives
	//	superbee, the least dissipative).
	double theta = 1.3;
	if (s0 < s1 && s1 < s2) {
		double d1 = theta*(s1 - s0);
		double d2 = (s2 - s0) / 2.;
		double d3 = theta*(s2 - s1);
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

float minmod2f(float theta, float s0, float s1, float s2)
{
	//theta should be used as a global var 
	// can be used to tune the limiting (theta=1
	//gives minmod, the most dissipative limiter and theta = 2 gives
	//	superbee, the least dissipative).
	//float theta = 1.3f;
	if (s0 < s1 && s1 < s2) {
		float d1 = theta*(s1 - s0);
		float d2 = (s2 - s0) / 2.0f;
		float d3 = theta*(s2 - s1);
		if (d2 < d1) d1 = d2;
		return min(d1, d3);
	}
	if (s0 > s1 && s1 > s2) {
		float d1 = theta*(s1 - s0), d2 = (s2 - s0) / 2.0f, d3 = theta*(s2 - s1);
		if (d2 > d1) d1 = d2;
		return max(d1, d3);
	}
	return 0.;
}
double dtnext(double t, double tnext, double dt)
{
	//Function to make dt match output time step and prevent dt from chnaging too fast
	// tnext = t+dtp with dtp is the previous dt


	
	//COpied staright from BASILISK and it sucks a little
	//Maybe make this part of a model param structure...


	double TEPS = 1e-9;
	if (tnext != HUGE && tnext > t) {
		unsigned int n = (tnext - t) / dt;
		//assert(n < INT_MAX); // check that dt is not too small
		if (n == 0)
			dt = tnext - t;
		else {
			double dt1 = (tnext - t) / n;
			if (dt1 > dt + TEPS)
				dt = (tnext - t) / (n + 1);
			else if (dt1 < dt)
				dt = dt1;
			tnext = t + dt;
		}
	}
	else
		tnext = t + dt;
	return dt;
}

void gradient(int nx, int ny,float theta, double delta, float *a, float *&dadx, float * &dady)
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


			//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
			dadx[i] = minmod2f(theta,a[xminus+iy*nx], a[i], a[xplus+iy*nx])/delta;
			//dady[i] = (a[i] - a[ix + yminus*nx]) / delta;
			dady[i] = minmod2f(theta,a[ix + yminus*nx], a[i], a[ix + yplus*nx])/delta;


		}
	}

}

void kurganov(double g,double CFL,double hm, double hp, double um, double up, double Delta,	double * fh, double * fq, double * dtmax)
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
void kurganovf(float g, float CFL,float hm, float hp, float um, float up, float Delta, float * fh, float * fq, float * dtmax)
{
	float eps = epsilon;
	float cp = sqrtf(g*hp), cm = sqrtf(g*hm);
	float ap = max(up + cp, um + cm); ap = max(ap, 0.0f);
	float am = min(up - cp, um - cm); am = min(am, 0.0f);
	float qm = hm*um, qp = hp*up;
	float a = max(ap, -am);
	float ad = 1.0f / (ap - am);
	if (a > eps) {
		*fh = (ap*qm - am*qp + ap*am*(hp - hm)) *ad; // (4.5) of [1]
		*fq = (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) +
			ap*am*(qp - qm)) *ad;
		float dt = CFL*Delta / a;
		if (dt < *dtmax)
			*dtmax = dt;
	}
	else
		*fh = *fq = 0.0f;
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


//Warning all the g, dt etc shouyld all be float so the compiler does the conversion before running the 

void update(int nx, int ny, float theta, float dt, float eps, float g,float CFL, float delta,float *hh, float *zs, float *uu, float *vv, float *&dh, float *&dhu, float *&dhv)
{
	int i, xplus, yplus, xminus, yminus;

	float hi;

		
	dtmax = 1 / epsilon;
	float dtmaxtmp = dtmax;

	// calculate gradients
	gradient(nx, ny, theta, delta, hh, dhdx, dhdy);
	gradient(nx, ny, theta, delta, zs, dzsdx, dzsdy);
	gradient(nx, ny, theta, delta, uu, dudx, dudy);
	gradient(nx, ny, theta, delta, vv, dvdx, dvdy);
	
	float cm = 1.0;// 0.1;
	float fmu = 1.0;
	float fmv = 1.0;
	
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



			float hn = hh[xminus + iy*nx];


			if (hi > eps || hn > eps)
			{

				float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;

				// along X
				dx = delta / 2.0f;
				zi = zs[i] - hi;

				//printf("%f\n", zi);


				zl = zi - dx*(dzsdx[i] - dhdx[i]);
				//printf("%f\n", zl);

				zn = zs[xminus + iy*nx] - hn;

				//printf("%f\n", zn);
				zr = zn + dx*(dzsdx[xminus + iy*nx] - dhdx[xminus + iy*nx]);


				zlr = max(zl, zr);

				hl = hi - dx*dhdx[i];
				up = uu[i] - dx*dudx[i];
				hp = max(0.f, hl + zl - zlr);

				hr = hn + dx*dhdx[xminus + iy*nx];
				um = uu[xminus + iy*nx] + dx*dudx[xminus + iy*nx];
				hm = max(0.f, hr + zr - zlr);

				//// Reimann solver
				float fh, fu, fv;
				float dtmaxf= 1.0f / (float)epsilon;

				//We can now call one of the approximate Riemann solvers to get the fluxes.
				kurganovf(g,CFL,hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
				fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
				dtmax = dtmaxf;
				dtmaxtmp = min(dtmax, dtmaxtmp);
				float cpo = sqrtf(g*hp), cmo = sqrtf(g*hm);
				float ap = max(up + cpo, um + cmo); ap = max(ap, 0.0f);
				float am = min(up - cpo, um - cmo); am = min(am, 0.0f);
				float qm = hm*um, qp = hp*up;

				float fubis= (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) +	ap*am*(qp - qm)) / (ap - am);
				/*
				if (ix == 11 && iy == 0)
				{
					printf("a=%f\t b=%f\t c=%f\t d=%f\n", ap*(qm*um + g*sq(hm) / 2.0f), -am*(qp*up + g*sq(hp) / 2.0f), (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm)) / (ap - am),1 / (ap - am));
				}
				*/
				//printf("%f\t%f\t%f\n", x[i], y[i], fh);


				//// Topographic term

				/**
				#### Topographic source term

				In the case of adaptive refinement, care must be taken to ensure
				well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
				float sl =  g / 2.0f*(hp*hp - hl*hl + (hl + hi)*(zi - zl));
				float sr =  g / 2.0f*(hm*hm - hr*hr + (hr + hn)*(zn - zr));

				////Flux update

				Fhu[i] = fmu * fh;
				Fqux[i] = fmu * (fu - sl);
				Su[i] = fmu * (fu - sr);
				Fqvx[i] = fmu * fv;
			}
			else
			{
				Fhu[i] = 0.0f;
				Fqux[i] = 0.0f;
				Su[i] = 0.0f;
				Fqvx[i] = 0.0f;
			}

			}
		}
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

				float hn = hh[ix + yminus*nx];
				float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;
				


				if (hi > eps || hn > eps)
				{


				//Along Y

				hn = hh[ix + yminus*nx];
				dx = delta / 2.0f;
				zi = zs[i] - hi;
				zl = zi - dx*(dzsdy[i] - dhdy[i]);
				zn = zs[ix + yminus*nx] - hn;
				zr = zn + dx*(dzsdy[ix + yminus*nx] - dhdy[ix + yminus*nx]);
				zlr = max(zl, zr);

				hl = hi - dx*dhdy[i];
				up = vv[i] - dx*dvdy[i];
				hp = max(0.f, hl + zl - zlr);

				hr = hn + dx*dhdy[ix + yminus*nx];
				um = vv[ix + yminus*nx] + dx*dvdy[ix + yminus*nx];
				hm = max(0.f, hr + zr - zlr);

				//// Reimann solver
				float fh, fu, fv;
				float dtmaxf = 1 / (float)epsilon;
				//printf("%f\t%f\t%f\n", x[i], y[i], dhdy[i]);
				//printf("%f\n", hr);
				//We can now call one of the approximate Riemann solvers to get the fluxes.
				kurganovf(g,CFL,hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
				fv = (fh > 0. ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
				dtmax = dtmaxf;
				dtmaxtmp = min(dtmax, dtmaxtmp);
				//// Topographic term

				/**
				#### Topographic source term

				In the case of adaptive refinement, care must be taken to ensure
				well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
				float sl = g / 2.0f*(hp*hp - hl*hl + (hl + hi)*(zi - zl));
				float sr = g / 2.0f*(hm*hm - hr*hr + (hr + hn)*(zn - zr));

				////Flux update

				Fhv[i] = fmv * fh;
				Fqvy[i] = fmv * (fu - sl);
				Sv[i] = fmv * (fu - sr);
				Fquy[i] = fmv* fv;

				//printf("%f\t%f\t%f\n", x[i], y[i], Fhv[i]);
			}
			else
			{
				Fhv[i] = 0.0f;
				Fqvy[i] = 0.0f;
				Sv[i] = 0.0f;
				Fquy[i] = 0.0f;
			}

			//printf("%f\t%f\t%f\n", x[i], y[i], Fquy[i]);
		}
	}

	dtmax = dtmaxtmp;


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
			//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);
			float cm = 1.0f;
			float cmdel = 1.0f / (cm * delta);
			dh[i] = -1.0f*(Fhu[xplus + iy*nx] - Fhu[i] + Fhv[ix + yplus*nx] - Fhv[i]) *cmdel;
			//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


			//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
			//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
			float dmdl = (fmu - fmu) *cmdel;// absurd!
			float dmdt = (fmv - fmv) *cmdel;// absurd!
			float fG = vv[i] * dmdl - uu[i] * dmdt;
			dhu[i] = (Fqux[i] + Fquy[i] - Su[xplus + iy*nx] - Fquy[ix + yplus*nx]) *cmdel;
			dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[ix + yplus*nx] - Fqvx[xplus + iy*nx]) *cmdel;
			//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
			dhu[i] += hi * (g*hi / 2.*dmdl + fG*vv[i]);
			dhv[i] += hi * (g*hi / 2.*dmdt - fG*uu[i]);

			


			

		}
	}
}

void update_spherical(int nx, int ny, float theta, float dt, float eps, float g, float CFL, float delta,float yo,float Radius, float *hh, float *zs, float *uu, float *vv, float *&dh, float *&dhu, float *&dhv)
{
	int i, xplus, yplus, xminus, yminus;

	float hi;


	dtmax = 1 / epsilon;
	float dtmaxtmp = dtmax;

	// calculate gradients
	gradient(nx, ny, theta, delta, hh, dhdx, dhdy);
	gradient(nx, ny, theta, delta, zs, dzsdx, dzsdy);
	gradient(nx, ny, theta, delta, uu, dudx, dudy);
	gradient(nx, ny, theta, delta, vv, dvdx, dvdy);

	float cm = 1.0;// 0.1;
	float fmu = 1.0;
	float fmv = 1.0;
	float phi, dphi,y;
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

			y = yo + iy*delta;

			phi = y*(float) pi / 180.0f;

			dphi = delta / (2.0f*Radius);// dy*0.5f*pi/180.0f;

			cm = (sinf(phi + dphi) - sinf(phi - dphi)) / (2.0f*dphi);

			fmu = 1.0f;
			fmv = cosf(phi);


			float hn = hh[xminus + iy*nx];


			if (hi > eps || hn > eps)
			{

				float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;

				// along X
				dx = delta / 2.0f;
				zi = zs[i] - hi;

				//printf("%f\n", zi);


				zl = zi - dx*(dzsdx[i] - dhdx[i]);
				//printf("%f\n", zl);

				zn = zs[xminus + iy*nx] - hn;

				//printf("%f\n", zn);
				zr = zn + dx*(dzsdx[xminus + iy*nx] - dhdx[xminus + iy*nx]);


				zlr = max(zl, zr);

				hl = hi - dx*dhdx[i];
				up = uu[i] - dx*dudx[i];
				hp = max(0.f, hl + zl - zlr);

				hr = hn + dx*dhdx[xminus + iy*nx];
				um = uu[xminus + iy*nx] + dx*dudx[xminus + iy*nx];
				hm = max(0.f, hr + zr - zlr);

				//// Reimann solver
				float fh, fu, fv;
				float dtmaxf = 1.0f / (float)epsilon;

				//We can now call one of the approximate Riemann solvers to get the fluxes.
				kurganovf(g, CFL, hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
				fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
				dtmax = dtmaxf;
				dtmaxtmp = min(dtmax, dtmaxtmp);
				float cpo = sqrtf(g*hp), cmo = sqrtf(g*hm);
				float ap = max(up + cpo, um + cmo); ap = max(ap, 0.0f);
				float am = min(up - cpo, um - cmo); am = min(am, 0.0f);
				float qm = hm*um, qp = hp*up;

				float fubis = (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm)) / (ap - am);
				/*
				if (ix == 11 && iy == 0)
				{
				printf("a=%f\t b=%f\t c=%f\t d=%f\n", ap*(qm*um + g*sq(hm) / 2.0f), -am*(qp*up + g*sq(hp) / 2.0f), (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm)) / (ap - am),1 / (ap - am));
				}
				*/
				//printf("%f\t%f\t%f\n", x[i], y[i], fh);


				//// Topographic term

				/**
				#### Topographic source term

				In the case of adaptive refinement, care must be taken to ensure
				well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
				float sl = g / 2.0f*(hp*hp - hl*hl + (hl + hi)*(zi - zl));
				float sr = g / 2.0f*(hm*hm - hr*hr + (hr + hn)*(zn - zr));

				////Flux update

				Fhu[i] = fmu * fh;
				Fqux[i] = fmu * (fu - sl);
				Su[i] = fmu * (fu - sr);
				Fqvx[i] = fmu * fv;
			}
			else
			{
				Fhu[i] = 0.0f;
				Fqux[i] = 0.0f;
				Su[i] = 0.0f;
				Fqvx[i] = 0.0f;
			}

		}
	}
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

			float hn = hh[ix + yminus*nx];
			float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;
			y = yo + iy*delta;

			phi = y*(float)pi / 180.0f;

			dphi = delta / (2.0f*Radius);// dy*0.5f*pi/180.0f;

			cm = (sinf(phi + dphi) - sinf(phi - dphi)) / (2.0f*dphi);

			fmu = 1.0f;
			fmv = cosf(phi);


			if (hi > eps || hn > eps)
			{


				//Along Y

				hn = hh[ix + yminus*nx];
				dx = delta / 2.0f;
				zi = zs[i] - hi;
				zl = zi - dx*(dzsdy[i] - dhdy[i]);
				zn = zs[ix + yminus*nx] - hn;
				zr = zn + dx*(dzsdy[ix + yminus*nx] - dhdy[ix + yminus*nx]);
				zlr = max(zl, zr);

				hl = hi - dx*dhdy[i];
				up = vv[i] - dx*dvdy[i];
				hp = max(0.f, hl + zl - zlr);

				hr = hn + dx*dhdy[ix + yminus*nx];
				um = vv[ix + yminus*nx] + dx*dvdy[ix + yminus*nx];
				hm = max(0.f, hr + zr - zlr);

				//// Reimann solver
				float fh, fu, fv;
				float dtmaxf = 1 / (float)epsilon;
				//printf("%f\t%f\t%f\n", x[i], y[i], dhdy[i]);
				//printf("%f\n", hr);
				//We can now call one of the approximate Riemann solvers to get the fluxes.
				kurganovf(g, CFL, hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
				fv = (fh > 0. ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
				dtmax = dtmaxf;
				dtmaxtmp = min(dtmax, dtmaxtmp);
				//// Topographic term

				/**
				#### Topographic source term

				In the case of adaptive refinement, care must be taken to ensure
				well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
				float sl = g / 2.0f*(hp*hp - hl*hl + (hl + hi)*(zi - zl));
				float sr = g / 2.0f*(hm*hm - hr*hr + (hr + hn)*(zn - zr));

				////Flux update

				Fhv[i] = fmv * fh;
				Fqvy[i] = fmv * (fu - sl);
				Sv[i] = fmv * (fu - sr);
				Fquy[i] = fmv* fv;

				//printf("%f\t%f\t%f\n", x[i], y[i], Fhv[i]);
			}
			else
			{
				Fhv[i] = 0.0f;
				Fqvy[i] = 0.0f;
				Sv[i] = 0.0f;
				Fquy[i] = 0.0f;
			}

			//printf("%f\t%f\t%f\n", x[i], y[i], Fquy[i]);
		}
	}

	dtmax = dtmaxtmp;


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

			y = yo + iy*delta;

			phi = y*(float)pi / 180.0f;

			dphi = delta / (2.0f*Radius);// dy*0.5f*pi/180.0f;

			cm = (sinf(phi + dphi) - sinf(phi - dphi)) / (2.0f*dphi);

			fmu = 1.0f;
			fmv = cosf(phi);

			////
			//vector dhu = vector(updates[1 + dimension*l]);
			//foreach() {
			//	double dhl =
			//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
			//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
			//	foreach_dimension()
			//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
			//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);
			
			float cmdel = 1.0f / (cm * delta);
			dh[i] = -1.0f*(Fhu[xplus + iy*nx] - Fhu[i] + Fhv[ix + yplus*nx] - Fhv[i]) *cmdel;
			//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


			//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
			//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
			float dmdl = (fmu - fmu) *cmdel; 
			float dmdt = (fmv - fmv) *cmdel; 
			float fG = vv[i] * dmdl - uu[i] * dmdt;
			dhu[i] = (Fqux[i] + Fquy[i] - Su[xplus + iy*nx] - Fquy[ix + yplus*nx]) *cmdel;
			dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[ix + yplus*nx] - Fqvx[xplus + iy*nx]) *cmdel;
			//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
			dhu[i] += hi * (g*hi / 2.*dmdl + fG*vv[i]);
			dhv[i] += hi * (g*hi / 2.*dmdt - fG*uu[i]);






		}
	}
}


void advance(int nx, int ny, float dt, float eps, float *hh, float *zs, float *uu, float * vv, float * dh, float *dhu, float *dhv, float * &hho, float *&zso, float *&uuo, float *&vvo)
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
			float hold = hh[i];
			float ho, uo, vo;
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
				uo = 0.f;
				vo = 0.f;
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


void cleanup(int nx, int ny, float * hhi, float *zsi, float *uui, float *vvi, float * &hho, float *&zso, float *&uuo, float *&vvo)
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


extern "C" void create2dnc(int nx, int ny, double dx, double dy, double totaltime, double *xx, double *yy, float * var)
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



	status = nc_def_var(ncid, "2Dvar", NC_FLOAT, 3, var_dimids, &tvar_id);


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


	status = nc_put_vara_float(ncid, tvar_id, start, count, var);
	status = nc_close(ncid);

}

extern "C" void write2varnc(int nx, int ny, double totaltime, float * var)
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
	status = nc_put_vara_float(ncid, var_id, start, count, var);
	status = nc_close(ncid);

}

// Main loop that actually runs the model
float FlowCPU(Param XParam, float nextoutputtime)
{
	int nx = XParam.nx;
	int ny = XParam.ny;

	//forcing bnd update 
	//////////////////////////////
	//flowbnd();

	//update(int nx, int ny, double dt, double eps,double *hh, double *zs, double *uu, double *vv, double *dh, double *dhu, double *dhv)

	if (XParam.spherical == 1)
	{
		update_spherical(nx, ny, XParam.theta, XParam.dt, XParam.eps, XParam.g, XParam.CFL, XParam.delta, XParam.yo, XParam.Radius, hh, zs, uu, vv, dh, dhu, dhv);
	}
	else
	{
		update(nx, ny, XParam.theta, XParam.dt, XParam.eps, XParam.g, XParam.CFL, XParam.delta, hh, zs, uu, vv, dh, dhu, dhv);
	}
	
	//printf("dtmax=%f\n", dtmax);
	XParam.dt = dtmax;// dtnext(totaltime, totaltime + dt, dtmax);
	if (ceil((nextoutputtime - XParam.totaltime) / XParam.dt)> 0.0)
	{
		XParam.dt = (nextoutputtime - XParam.totaltime) / ceil((nextoutputtime - XParam.totaltime) / XParam.dt);
	}
	//printf("dt=%f\n", XParam.dt);
	//if (totaltime>0.0) //Fix this!
	{
		//predictor
		advance(nx, ny, XParam.dt*0.5, XParam.eps,  hh, zs, uu, vv, dh, dhu, dhv, hho, zso, uuo, vvo);

		//corrector
		if (XParam.spherical == 1)
		{
			update_spherical(nx, ny, XParam.theta, XParam.dt, XParam.eps, XParam.g, XParam.CFL, XParam.delta, XParam.yo, XParam.Radius, hho, zso, uuo, vvo, dh, dhu, dhv);
		}
		else
		{
			update(nx, ny, XParam.theta, XParam.dt, XParam.eps, XParam.g, XParam.CFL, XParam.delta, hho, zso, uuo, vvo, dh, dhu, dhv);
		}
		
	}
	//
	advance(nx, ny, XParam.dt, XParam.eps, hh, zs, uu, vv, dh, dhu, dhv, hho, zso, uuo, vvo);

	cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
	
	quadfrictionCPU(nx, ny, XParam.dt, XParam.eps, XParam.cf, hh, uu, vv);
	//write2varnc(nx, ny, totaltime, hh);

	//noslipbndallCPU(nx, ny, XParam.dt, XParam.eps, zb, zs, hh, uu, vv);
	return XParam.dt;


}


void leftdirichletCPU(int nx, int ny, float g, std::vector<float> zsbndvec, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	float zsbnd;
	for (int iy = 0; iy < ny; iy++)
	{
		if (zsbndvec.size() == 1)
		{
			zsbnd = zsbndvec[0];
		}
		else
		{
			int iprev = min(max((int) ceil(iy / (1 / (zsbndvec.size() - 1))),0), (int) zsbndvec.size()-2);
			int inext = iprev + 1;
			// here interp time is used to interpolate to the right node rather than in time...
			zsbnd = interptime(zsbndvec[inext], zsbndvec[iprev], (float)(inext - iprev), (float)(iy - iprev));
		}
		int ix = 0;
		int i = ix + iy*nx;
		int xplus;
		float hhi;
		if (ix == 0 && iy < ny)
		{
			xplus = min(ix + 1, nx - 1);
			hh[i] = zsbnd - zb[i];
			zs[i] = zsbnd;
			uu[i] = -2.0f*(sqrtf(g*max(hh[xplus + iy*nx], 0.0f)) - sqrtf(g*max(zsbnd - zb[xplus + iy*nx], 0.0f))) + uu[xplus + iy*nx];
			vv[i] = 0.0f;
			//if (iy == 0)
			//{
			//	printf("zsbnd=%f\t", zsbnd);
			//}
		}
		
	}
}


void rightdirichletCPU(int nx, int ny, float g, std::vector<float> zsbndvec, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	float zsbnd;
	for (int iy = 0; iy < ny; iy++)
	{
		if (zsbndvec.size() == 1)
		{
			zsbnd = zsbndvec[0];
		}
		else
		{
			int iprev = min(max((int)ceil(iy / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
			int inext = iprev + 1;
			// here interp time is used to interpolate to the right node rather than in time...
			zsbnd = interptime(zsbndvec[inext], zsbndvec[iprev], (float)(inext - iprev), (float)(iy - iprev));
		}
		int ix = nx-1;
		int i = ix + iy*nx;
		int xminus;
		float hhi;
		if ( iy < ny)
		{
			xminus = max(ix - 1, 0);
			hh[i] = zsbnd - zb[i];
			zs[i] = zsbnd;
			uu[i] = +2.0f*(sqrtf(g*max(hh[xminus + iy*nx], 0.0f)) - sqrtf(g*max(zsbnd - zb[xminus + iy*nx], 0.0f))) + uu[xminus + iy*nx];
			vv[i] = 0.0f;
		}

	}
}

void topdirichletCPU(int nx, int ny, float g, std::vector<float> zsbndvec, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	float zsbnd;
	for (int ix = 0; ix < nx; ix++)
	{
		if (zsbndvec.size() == 1)
		{
			zsbnd = zsbndvec[0];
		}
		else
		{
			int iprev = min(max((int)ceil(ix / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
			int inext = iprev + 1;
			// here interp time is used to interpolate to the right node rather than in time...
			zsbnd = interptime(zsbndvec[inext], zsbndvec[iprev], (float)(inext - iprev), (float)(ix - iprev));
		}
		int iy = ny - 1;
		int i = ix + iy*nx;
		int yminus;
		float hhi;
		if (iy < ny)
		{
			//xminus = max(ix - 1, 0);
			//xplus = min(ix + 1, nx - 1);
			//xminus = max(ix - 1, 0);
			//yplus = min(iy + 1, ny - 1);
			yminus = max(iy - 1, 0);
			hh[i] = zsbnd - zb[i];
			zs[i] = zsbnd;
			vv[i] = +2.0f*(sqrtf(g*max(hh[ix + yminus*nx], 0.0f)) - sqrtf(g*max(zsbnd - zb[ix + yminus*nx], 0.0f))) + vv[ix + yminus*nx];
			uu[i] = 0.0f;
		}

	}
}


void botdirichletCPU(int nx, int ny, float g, std::vector<float> zsbndvec, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	float zsbnd;
	for (int ix = 0; ix < nx; ix++)
	{
		if (zsbndvec.size() == 1)
		{
			zsbnd = zsbndvec[0];
		}
		else
		{
			int iprev = min(max((int)ceil(ix / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
			int inext = iprev + 1;
			// here interp time is used to interpolate to the right node rather than in time...
			zsbnd = interptime(zsbndvec[inext], zsbndvec[iprev], (float)(inext - iprev), (float)(ix - iprev));
		}
		int iy = 0;
		int i = ix + iy*nx;
		int yplus;
		float hhi;
		if (iy < ny)
		{
			//xminus = max(ix - 1, 0);
			//xplus = min(ix + 1, nx - 1);
			//xminus = max(ix - 1, 0);
			yplus = min(iy + 1, ny - 1);
			//yminus = max(iy - 1, 0);
			hh[i] = zsbnd - zb[i];
			zs[i] = zsbnd;
			vv[i] = -2.0f*(sqrtf(g*max(hh[ix + yplus*nx], 0.0f)) - sqrtf(g*max(zsbnd - zb[ix + yplus*nx], 0.0f))) + vv[ix + yplus*nx];
			uu[i] = 0.0f;
		}

	}
}

void quadfrictionCPU(int nx, int ny, float dt, float eps, float cf, float *hh, float *uu, float *vv)
{
	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			int i = ix + iy*nx;

			float normu, hhi;

			
				hhi = hh[i];
				if (hhi > eps)
				{
					normu = uu[i] * uu[i] + vv[i] * vv[i];
					float frc = (1.0 + dt*cf*normu / hhi);
					//u.x[] = h[]>dry ? u.x[] / (1 + dt*cf*norm(u) / h[]) : 0.;
					uu[i] = uu[i] / frc;
					vv[i] = vv[i] / frc;
				}

			
		}
	}

}

void noslipbndLeftCPU(int nx, int ny, float eps, float *zb, float *zs, float *hh, float *uu, float *vv)
{
	int i,ix;
	int  xplus, yplus, xminus, yminus;
	float normu, hhi;

	for (int iy = 0; iy < ny; iy++)
	{
		ix = 0;
		i = ix + iy*nx;
		xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

		
			uu[i] = 0.0f;
			zs[i] = zs[xplus + iy*nx];
			hh[i] = max(zs[xplus + iy*nx] - zb[i], eps);
		
	}
}

void noslipbndRightCPU(int nx, int ny, float eps, float *zb, float *zs, float *hh, float *uu, float *vv)
{

	int i,ix;
	int  xplus, yplus, xminus, yminus;
	float normu, hhi;

	for (int iy = 0; iy < ny; iy++)
	{
		ix = nx - 1;
		i = ix + iy*nx;
		//xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

			
		uu[i] = 0.0f;
		zs[i] = zs[xminus + iy*nx];
		hh[i] = max(zs[xminus + iy*nx] - zb[i], eps);

			


	}

}
void noslipbndTopCPU(int nx, int ny, float eps, float *zb, float *zs, float *hh, float *uu, float *vv)
{

	int i, iy;
	int  xplus, yplus, xminus, yminus;
	float normu, hhi;



	for (int ix = 0; ix < nx; ix++)
	{
		iy = ny - 1;
		i = ix + iy*nx;
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);


		vv[i] = 0.0f;
		zs[i] = zs[ix + yminus*nx];
		hh[i] = max(zs[ix + yminus*nx] - zb[i], eps);




	}

}

void noslipbndBotCPU(int nx, int ny, float eps, float *zb, float *zs, float *hh, float *uu, float *vv)
{

	int i, iy;
	int  xplus, yplus, xminus, yminus;
	float normu, hhi;

	for (int ix = 0; ix < nx; ix++)
	{
		iy = 0;
		i = ix + iy*nx;
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);


		vv[i] = 0.0f;
		zs[i] = zs[ix + yplus*nx];
		hh[i] = max(zs[ix + yplus*nx] - zb[i], eps);




	}

}

void noslipbndallCPU(int nx, int ny, float dt, float eps, float *zb, float *zs, float *hh, float *uu, float *vv)
{
	
	int i; 
	int  xplus, yplus, xminus, yminus;
	float normu, hhi;

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
				uu[i] = 0.0f;
				zs[i] = zs[xplus + iy*nx];
				hh[i] = max(zs[xplus + iy*nx] - zb[i], eps);
			}
			if (ix == nx - 1)
			{
				uu[i] = 0.0f;
				zs[i] = zs[xminus + iy*nx];
				hh[i] = max(zs[xminus + iy*nx] - zb[i], eps);

			}

			if (iy == 0)
			{
				vv[i] = 0.0f;
				zs[i] = zs[ix + yplus*nx];
				hh[i] = max(zs[ix + yplus*nx] - zb[i], eps);
			}
			if (iy == ny - 1)
			{
				vv[i] = 0.0f;
				zs[i] = zs[ix + yminus*nx];
				hh[i] = max(zs[ix + yminus*nx] - zb[i], eps);

			}

		}
	}

}
void warmstart(Param XParam, std::vector<SLTS> leftWLbnd, std::vector<SLTS> rightbnd, std::vector<SLTS> topbnd, std::vector<SLTS> botbnd)
{
	int nx = XParam.nx;
	int ny = XParam.ny;
	float zsbnd;




	if (!leftWLbnd.empty() && XParam.left == 1)
	{
		
		int SLstepinbnd = 1;



		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = leftWLbnd[SLstepinbnd].time - XParam.totaltime;

		while (difft < 0.0)
		{
			SLstepinbnd++;
			difft = leftWLbnd[SLstepinbnd].time - XParam.totaltime;
		}
		std::vector<float> leftbnd;
		for (int n = 0; n < leftWLbnd[SLstepinbnd].wlevs.size(); n++)
		{
			leftbnd.push_back(interptime(leftWLbnd[SLstepinbnd].wlevs[n], leftWLbnd[SLstepinbnd - 1].wlevs[n], leftWLbnd[SLstepinbnd].time - leftWLbnd[SLstepinbnd - 1].time, XParam.totaltime - leftWLbnd[SLstepinbnd - 1].time));

		}

		for (int iy = 0; iy < ny; iy++)
		{
			if (leftbnd.size() == 1)
			{
				zsbnd = leftbnd[0];
			}
			else
			{
				int iprev = min(max((int)ceil(iy / (1 / (leftbnd.size() - 1))), 0), (int)leftbnd.size() - 2);
				int inext = iprev + 1;
				// here interp time is used to interpolate to the right node rather than in time...
				zsbnd = interptime(leftbnd[inext], leftbnd[iprev], (float)(inext - iprev), (float)(iy - iprev));
			}
			int ix = 0;
			int i = ix + iy*nx;
			int xplus;
			float hhi;
			if (ix == 0 && iy < ny)
			{

				hh[i] = zsbnd - zb[i];
				zs[i] = zsbnd;

			}

		}
	}
}

void AddmeanCPU(Param XParam)
{
	int nx = XParam.nx;
	int ny = XParam.ny;

	if (XParam.outhhmean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				hhmean[i + j*nx] = hhmean[i + j*nx] + hh[i + j*nx];
			}
		}
	}

	if (XParam.outzsmean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				zsmean[i + j*nx] = zsmean[i + j*nx] + zs[i + j*nx];
			}
		}
	}

	if (XParam.outuumean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				uumean[i + j*nx] = uumean[i + j*nx] + uu[i + j*nx];
			}
		}
	}
	if (XParam.outvvmean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				vvmean[i + j*nx] = vvmean[i + j*nx] + vv[i + j*nx];
			}
		}
	}


}
void DivmeanCPU(Param XParam, float nstep)
{
	int nx = XParam.nx;
	int ny = XParam.ny;

	if (XParam.outhhmean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				hhmean[i + j*nx] = hhmean[i + j*nx] /nstep;
			}
		}
	}

	if (XParam.outzsmean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				zsmean[i + j*nx] = zsmean[i + j*nx] / nstep;
			}
		}
	}

	if (XParam.outuumean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				uumean[i + j*nx] = uumean[i + j*nx] / nstep;
			}
		}
	}
	if (XParam.outvvmean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				vvmean[i + j*nx] = vvmean[i + j*nx] / nstep;
			}
		}
	}


}

void ResetmeanCPU(Param XParam)
{
	int nx = XParam.nx;
	int ny = XParam.ny;

	if (XParam.outhhmean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				hhmean[i + j*nx] = 0.0;
			}
		}
	}

	if (XParam.outzsmean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				zsmean[i + j*nx] = 0.0;
			}
		}
	}

	if (XParam.outuumean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				uumean[i + j*nx] = 0.0;
			}
		}
	}
	if (XParam.outvvmean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				vvmean[i + j*nx] = 0.0;
			}
		}
	}


}

void maxallCPU(Param XParam)
{
	int nx = XParam.nx;
	int ny = XParam.ny;

	if (XParam.outhhmax == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				hhmax[i + j*nx] = max(hhmax[i + j*nx] , hh[i + j*nx]);
			}
		}
	}
	if (XParam.outzsmax == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				zsmax[i + j*nx] = max(zsmax[i + j*nx], zs[i + j*nx]);
			}
		}
	}
	if (XParam.outuumax == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				uumax[i + j*nx] = max(uumax[i + j*nx], uu[i + j*nx]);
			}
		}
	}
	if (XParam.outvvmax == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				vvmax[i + j*nx] = max(vvmax[i + j*nx], vv[i + j*nx]);
			}
		}
	}
}
void CalcVort(Param XParam)
{
	int nx = XParam.nx;
	int ny = XParam.ny;
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			vort[i + j*nx] = dvdy[i + j*nx] - dudx[i + j*nx];
		}
	}
}
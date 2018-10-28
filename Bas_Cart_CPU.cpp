//////////////////////////////////////////////////////////////////////////////////
//						                                                        //
//Copyright (C) 2017 Bosserelle                                                 //
//                                                                              //
// This code contains an adaptation of the St Venant equation from Basilisk		//
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


int findright(int blksize,int ix, int iy, int bid, int blrightid)
{
	int xplus;
	if (ix < 15)
	{
		xplus = (ix + 1) + iy * 16 + bid * blksize;
	}
	else
	{
		if (blrightid != bid)
		{
			xplus = 0 + iy * 16 + blrightid * blksize;
		}
		else
		{
			xplus = ix + iy * 16 + bid * blksize;
		}
	}
	return xplus;

}

int findleft(int blksize, int ix, int iy, int bid, int blleftid)
{
	int xminus;
	if (ix > 0)
	{
		xminus = (ix - 1) + iy * 16 + bid * blksize;
	}
	else
	{
		if (blleftid != bid)
		{
			xminus = 15 + iy * 16 + blleftid * blksize;
		}
		else
		{
			xminus = ix + iy * 16 + bid * blksize;
		}
	}
	return xminus;

}

int findtop(int blksize, int ix, int iy, int bid, int bltopid)
{
	int yplus;
	if (iy < 15)
	{
		yplus = ix  + (iy+1) * 16 + bid * blksize;
	}
	else
	{
		if (bltopid != bid)
		{
			yplus = ix + 0 * 16 + bltopid * blksize;
		}
		else
		{
			yplus = ix + iy * 16 + bid * blksize;
		}
	}
	return yplus;

}


int findbot(int blksize, int ix, int iy, int bid, int blbotid)
{
	int yminus;
	if (iy > 0)
	{
		yminus = ix + (iy - 1) * 16 + bid * blksize;
	}
	else
	{
		if (blbotid != bid)
		{
			yminus = ix + 15 * 16 + blbotid* blksize;
		}
		else
		{
			yminus = ix + iy * 16 + bid * blksize;
		}
	}
	return yminus;

}

template <class T> const T& max(const T& a, const T& b) {
	return (a<b) ? b : a;     // or: return comp(a,b)?b:a; for version (2)
}

template <class T> const T& min(const T& a, const T& b) {
	return !(b<a) ? a : b;     // or: return comp(a,b)?b:a; for version (2)
}

template <class T> T minmod2(T theta, T s0, T s1, T s2)
{
	//theta should be used as a global var 
	// can be used to tune the limiting (theta=1
	//gives minmod, the most dissipative limiter and theta = 2 gives
	//	superbee, the least dissipative).
	//float theta = 1.3f;
	

	if (s0 < s1 && s1 < s2) {
		T d1 = theta*(s1 - s0);
		T d2 = (s2 - s0) / T(2.0);
		T d3 = theta*(s2 - s1);
		if (d2 < d1) d1 = d2;
		return min(d1, d3);
	}
	if (s0 > s1 && s1 > s2) {
		T d1 = theta*(s1 - s0), d2 = (s2 - s0) / T(2.0), d3 = theta*(s2 - s1);
		if (d2 > d1) d1 = d2;
		return max(d1, d3);
	}
	return 0.;
}
/*double dtnext(double t, double tnext, double dt)
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
	*/

float interp2wnd(int wndnx, int wndny, float wnddx, float wndxo, float wndyo, float x, float y, float * U)
{
	// This function interpolates the values in cfmapin to cf using a bilinear interpolation

	

	// cells that falls off this domain are assigned 
	double x1, x2, y1, y2;
	double q11, q12, q21, q22;
	int cfi, cfip, cfj, cfjp;



	cfi = min(max((int)floor((x - wndxo) / wnddx), 0), wndnx - 2);
	cfip = cfi + 1;

	x1 = wndxo + wnddx*cfi;
	x2 = wndxo + wnddx*cfip;

	cfj = min(max((int)floor((y - wndyo) / wnddx), 0), wndny - 2);
	cfjp = cfj + 1;

	y1 = wndyo + wnddx*cfj;
	y2 = wndyo + wnddx*cfjp;

	q11 = U[cfi + cfj*wndnx];
	q12 = U[cfi + cfjp*wndnx];
	q21 = U[cfip + cfj*wndnx];
	q22 = U[cfip + cfjp*wndnx];

	return (float)BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
				
}
double interp2wnd(int wndnx, int wndny, double wnddx, double wndxo, double wndyo, double x, double y, float * U)
{
	// This function interpolates the values in cfmapin to cf using a bilinear interpolation



	// cells that falls off this domain are assigned 
	double x1, x2, y1, y2;
	double q11, q12, q21, q22;
	int cfi, cfip, cfj, cfjp;



	cfi = min(max((int)floor((x - wndxo) / wnddx), 0), wndnx - 2);
	cfip = cfi + 1;

	x1 = wndxo + wnddx*cfi;
	x2 = wndxo + wnddx*cfip;

	cfj = min(max((int)floor((y - wndyo) / wnddx), 0), wndny - 2);
	cfjp = cfj + 1;

	y1 = wndyo + wnddx*cfj;
	y2 = wndyo + wnddx*cfjp;

	q11 = U[cfi + cfj*wndnx];
	q12 = U[cfi + cfjp*wndnx];
	q21 = U[cfip + cfj*wndnx];
	q22 = U[cfip + cfjp*wndnx];

	return BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);

}

template <class T> void gradient(int nblk, int blksize, T theta, T delta, int * leftblk, int * rightblk, int * topblk, int * botblk, T *a, T *&dadx, T * &dady)
{

	int i, xplus, yplus, xminus, yminus;

	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;
				//
				//
				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				yminus = findbot(blksize, ix, iy, ib, botblk[ib]);
				


				//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
				dadx[i] = minmod2(theta, a[xminus], a[i], a[xplus]) / delta;
				//dady[i] = (a[i] - a[ix + yminus*nx]) / delta;
				dady[i] = minmod2(theta, a[yminus], a[i], a[yplus]) / delta;
			}


		}
	}

}

template <class T> void kurganov(T g, T CFL, T hm, T hp, T um, T up, T Delta, T  *fh, T  *fq, T *dtmax)
{
	
	T cp, cm, ap, am, qm, qp, a,dt, epsil;
	epsil = T(1e-30);
	cp = sqrt(g*hp);
	cm = sqrt(g*hm);
	ap = max(up + cp, um + cm);
	ap = max(ap, T(0.0));
	am = min(up - cp, um - cm);
	am = min(am, T(0.0));
	qm = hm*um;
	qp = hp*up;
	a = max(ap, -am);

	if (a > epsil) {
		*fh = (ap*qm - am*qp + ap*am*(hp - hm)) / (ap - am); // (4.5) of [1]
		*fq = (ap*(qm*um + g*sq(hm) / T(2.0)) - am*(qp*up + g*sq(hp) / T(2.0)) +
			ap*am*(qp - qm)) / (ap - am);
		dt = CFL*Delta / a;
		if (dt < *dtmax)
			*dtmax = dt;
	}
	else
		*fh = T(0.0);
		*fq = T(0.0);
}
void kurganovf(float g, float CFL,float hm, float hp, float um, float up, float Delta, float * fh, float * fq, float * dtmax)
{
	float eps = (float) epsilon; //this epsilon doesn't need to be a gloabl variable
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
void kurganovd(double g, double CFL, double hm, double hp, double um, double up, double Delta, double * fh, double * fq, double * dtmax)
{
	double eps = (double)epsilon; //this epsilon doesn't need to be a gloabl variable
	double cp = sqrt(g*hp), cm = sqrt(g*hm);
	double ap = max(up + cp, um + cm); ap = max(ap, 0.0);
	double am = min(up - cp, um - cm); am = min(am, 0.0);
	double qm = hm*um, qp = hp*up;
	double a = max(ap, -am);
	double ad = 1.0 / (ap - am);
	if (a > eps) {
		*fh = (ap*qm - am*qp + ap*am*(hp - hm)) *ad; // (4.5) of [1]
		*fq = (ap*(qm*um + g*sq(hm) / 2.0) - am*(qp*up + g*sq(hp) / 2.0) +
			ap*am*(qp - qm)) *ad;
		double dt = CFL*Delta / a;
		if (dt < *dtmax)
			*dtmax = dt;
	}
	else
		*fh = *fq = 0.0;
}


/*
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
*/

//Warning all the g, dt etc shouyld all be float so the compiler does the conversion before running the 

void update(int nblk, int blksize, float theta, float dt, float eps, float g,float CFL, float delta,float *hh, float *zs, float *uu, float *vv, float *&dh, float *&dhu, float *&dhv)
{
	int i, xplus, yplus, xminus, yminus;

	float hi;

		
	dtmax = (float) (1.0 / epsilon);
	float dtmaxtmp = dtmax;

	// calculate gradients
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, hh, dhdx, dhdy);
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, zs, dzsdx, dzsdy);
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, uu, dudx, dudy);
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, vv, dvdx, dvdy);
	
	float cm = 1.0;// 0.1;
	float fmu = 1.0;
	float fmv = 1.0;
	
	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;

				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				yminus = findbot(blksize, ix, iy, ib, botblk[ib]);
				hi = hh[i];



				float hn = hh[xminus];


				if (hi > eps || hn > eps)
				{

					float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;

					// along X
					dx = delta / 2.0f;
					zi = zs[i] - hi;

					//printf("%f\n", zi);


					zl = zi - dx*(dzsdx[i] - dhdx[i]);
					//printf("%f\n", zl);

					zn = zs[xminus ] - hn;

					//printf("%f\n", zn);
					zr = zn + dx*(dzsdx[xminus ] - dhdx[xminus]);


					zlr = max(zl, zr);

					hl = hi - dx*dhdx[i];
					up = uu[i] - dx*dudx[i];
					hp = max(0.f, hl + zl - zlr);

					hr = hn + dx*dhdx[xminus];
					um = uu[xminus ] + dx*dudx[xminus];
					hm = max(0.f, hr + zr - zlr);

					//// Reimann solver
					float fh, fu, fv;
					float dtmaxf = 1.0f / (float)epsilon;

					//We can now call one of the approximate Riemann solvers to get the fluxes.
					kurganovf(g, CFL, hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
					fv = (fh > 0.f ? vv[xminus ] + dx*dvdx[xminus ] : vv[i] - dx*dvdx[i])*fh;
					dtmax = dtmaxf;
					dtmaxtmp = min(dtmax, dtmaxtmp);
					//float cpo = sqrtf(g*hp), cmo = sqrtf(g*hm);
					//float ap = max(up + cpo, um + cmo); ap = max(ap, 0.0f);
					//float am = min(up - cpo, um - cmo); am = min(am, 0.0f);
					//float qm = hm*um, qp = hp*up;

					//float fubis= (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) +	ap*am*(qp - qm)) / (ap - am);
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
	}
	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;

				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				yminus = findbot(blksize, ix, iy, ib, botblk[ib]);

				
				hi = hh[i];

				float hn = hh[yminus];
				float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;



				if (hi > eps || hn > eps)
				{


					//Along Y

					hn = hh[yminus];
					dx = delta / 2.0f;
					zi = zs[i] - hi;
					zl = zi - dx*(dzsdy[i] - dhdy[i]);
					zn = zs[ yminus] - hn;
					zr = zn + dx*(dzsdy[yminus] - dhdy[yminus]);
					zlr = max(zl, zr);

					hl = hi - dx*dhdy[i];
					up = vv[i] - dx*dvdy[i];
					hp = max(0.f, hl + zl - zlr);

					hr = hn + dx*dhdy[yminus];
					um = vv[yminus] + dx*dvdy[yminus];
					hm = max(0.f, hr + zr - zlr);

					//// Reimann solver
					float fh, fu, fv;
					float dtmaxf = 1 / (float)epsilon;
					//printf("%f\t%f\t%f\n", x[i], y[i], dhdy[i]);
					//printf("%f\n", hr);
					//We can now call one of the approximate Riemann solvers to get the fluxes.
					kurganovf(g, CFL, hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
					fv = (fh > 0. ? uu[yminus] + dx*dudy[yminus] : uu[i] - dx*dudy[i])*fh;
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
			}

			//printf("%f\t%f\t%f\n", x[i], y[i], Fquy[i]);
		}
	}

	dtmax = dtmaxtmp;


	// UPDATES For evolving quantities

	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;

				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				//xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				//yminus = findbot(blksize, ix, iy, ib, botblk[ib]);


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
				dh[i] = -1.0f*(Fhu[xplus] - Fhu[i] + Fhv[yplus] - Fhv[i]) *cmdel;
				//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


				//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
				//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
				float dmdl = (fmu - fmu) *cmdel;// absurd!
				float dmdt = (fmv - fmv) *cmdel;// absurd!
				float fG = vv[i] * dmdl - uu[i] * dmdt;
				dhu[i] = (Fqux[i] + Fquy[i] - Su[xplus] - Fquy[yplus]) *cmdel;
				dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[yplus] - Fqvx[xplus]) *cmdel;
				//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
				dhu[i] += hi * (g*hi / 2.0f*dmdl + fG*vv[i]);
				dhv[i] += hi * (g*hi / 2.0f*dmdt - fG*uu[i]);



			}
			

		}
	}
}


void updateATM(int nblk, int blksize, int cstwind, int cstpress, int windnx, int windny, float winddx, float windxo, float windyo, float Uwndi, float Vwndi, float theta, float dt, float eps, float g, float CFL, float delta, float Cd, float Pa2m, float *hh, float *zs, float *uu, float *vv, float *&dh, float *&dhu, float *&dhv, float * Uwnd, float * Vwnd, float * Patm)
{
	int i, xplus, yplus, xminus, yminus;

	float hi;


	dtmax = (float)(1.0 / epsilon);
	float dtmaxtmp = dtmax;

	// calculate gradients
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, hh, dhdx, dhdy);
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, zs, dzsdx, dzsdy);
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, uu, dudx, dudy);
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, vv, dvdx, dvdy);

	if (cstpress == 0)
	{
		gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, Patm, dPdx, dPdy);
	}

	float cm = 1.0;// 0.1;
	float fmu = 1.0;
	float fmv = 1.0;

	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;

				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				yminus = findbot(blksize, ix, iy, ib, botblk[ib]);
				hi = hh[i];



				float hn = hh[xminus];


				if (hi > eps || hn > eps)
				{

					float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;

					// along X
					dx = delta / 2.0f;

					if (cstpress == 0)
					{
						zi = zs[i] - hi + Pa2m*Patm[i];

						//printf("%f\n", zi);


						zl = zi - dx*(dzsdx[i] - dhdx[i] + Pa2m*dPdx[i]);
						//printf("%f\n", zl);

						zn = zs[xminus] - hn + Pa2m * dPdx[xminus];

						//printf("%f\n", zn);
						zr = zn + dx*(dzsdx[xminus] - dhdx[xminus] + Pa2m*dPdx[xminus]);
					}
					else
					{
						zi = zs[i] - hi ;

						//printf("%f\n", zi);


						zl = zi - dx*(dzsdx[i] - dhdx[i] );
						//printf("%f\n", zl);

						zn = zs[xminus] - hn ;

						//printf("%f\n", zn);
						zr = zn + dx*(dzsdx[xminus] - dhdx[xminus] );
					}

					zlr = max(zl, zr);

					hl = hi - dx*dhdx[i];
					up = uu[i] - dx*dudx[i];
					hp = max(0.f, hl + zl - zlr);

					hr = hn + dx*dhdx[xminus];
					um = uu[xminus] + dx*dudx[xminus];
					hm = max(0.f, hr + zr - zlr);

					//// Reimann solver
					float fh, fu, fv;
					float dtmaxf = 1.0f / (float)epsilon;

					//We can now call one of the approximate Riemann solvers to get the fluxes.
					kurganovf(g, CFL, hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
					fv = (fh > 0.f ? vv[xminus] + dx*dvdx[xminus] : vv[i] - dx*dvdx[i])*fh;
					dtmax = dtmaxf;
					dtmaxtmp = min(dtmax, dtmaxtmp);
					//float cpo = sqrtf(g*hp), cmo = sqrtf(g*hm);
					//float ap = max(up + cpo, um + cmo); ap = max(ap, 0.0f);
					//float am = min(up - cpo, um - cmo); am = min(am, 0.0f);
					//float qm = hm*um, qp = hp*up;

					//float fubis= (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) +	ap*am*(qp - qm)) / (ap - am);
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
	}
	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;

				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				yminus = findbot(blksize, ix, iy, ib, botblk[ib]);


				hi = hh[i];

				float hn = hh[yminus];
				float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;



				if (hi > eps || hn > eps)
				{


					//Along Y

					hn = hh[yminus];
					dx = delta / 2.0f;

					if (cstpress == 0)
					{
						zi = zs[i] - hi + Pa2m*Patm[i];
						zl = zi - dx*(dzsdy[i] - dhdy[i] + Pa2m*dPdy[i]);
						zn = zs[yminus] - hn + Pa2m * Patm[yminus];
						zr = zn + dx*(dzsdy[yminus] - dhdy[yminus] + Pa2m * dPdy[yminus]);
					}
					else
					{
						zi = zs[i] - hi ;
						zl = zi - dx*(dzsdy[i] - dhdy[i] );
						zn = zs[yminus] - hn ;
						zr = zn + dx*(dzsdy[yminus] - dhdy[yminus] );
					}

					zlr = max(zl, zr);

					hl = hi - dx*dhdy[i];
					up = vv[i] - dx*dvdy[i];
					hp = max(0.f, hl + zl - zlr);

					hr = hn + dx*dhdy[yminus];
					um = vv[yminus] + dx*dvdy[yminus];
					hm = max(0.f, hr + zr - zlr);

					//// Reimann solver
					float fh, fu, fv;
					float dtmaxf = 1 / (float)epsilon;
					//printf("%f\t%f\t%f\n", x[i], y[i], dhdy[i]);
					//printf("%f\n", hr);
					//We can now call one of the approximate Riemann solvers to get the fluxes.
					kurganovf(g, CFL, hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
					fv = (fh > 0. ? uu[yminus] + dx*dudy[yminus] : uu[i] - dx*dudy[i])*fh;
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
			}

			//printf("%f\t%f\t%f\n", x[i], y[i], Fquy[i]);
		}
	}

	dtmax = dtmaxtmp;


	// UPDATES For evolving quantities

	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;

				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				yminus = findbot(blksize, ix, iy, ib, botblk[ib]);


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
				dh[i] = -1.0f*(Fhu[xplus] - Fhu[i] + Fhv[yplus] - Fhv[i]) *cmdel;
				//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


				//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
				//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
				float dmdl = (fmu - fmu) *cmdel;// absurd!
				float dmdt = (fmv - fmv) *cmdel;// absurd!
				float fG = vv[i] * dmdl - uu[i] * dmdt;

				//int windnx,int windny,float winddx, float windxo, float windyo,
				float x = blockxo[ib] + ix*delta;
				float y = blockyo[ib] + iy*delta;

				if (cstwind == 0)
				{
					Uwndi = interp2wnd(windnx, windny, winddx, windxo, windyo, x, y, Uwnd);
					Vwndi = interp2wnd(windnx, windny, winddx, windxo, windyo, x, y, Vwnd);
				}
				dhu[i] = (Fqux[i] + Fquy[i] - Su[xplus] - Fquy[yplus]) *cmdel + 0.00121951*Cd*Uwndi*abs(Uwndi);
				dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[yplus] - Fqvx[xplus]) *cmdel + 0.00121951*Cd*Vwndi*abs(Vwndi);
				//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
				dhu[i] += hi * (g*hi / 2.0f*dmdl + fG*vv[i]);
				dhv[i] += hi * (g*hi / 2.0f*dmdt - fG*uu[i]);



			}


		}
	}
}


void updateD(int nblk, int blksize, double theta, double dt, double eps, double g, double CFL, double delta, double *hh, double *zs, double *uu, double *vv, double *&dh, double *&dhu, double *&dhv)
{
	int i, xplus, yplus, xminus, yminus;

	double hi;


	dtmax = (1.0 / epsilon);
	double dtmaxtmp = dtmax;
	// calculate gradients
	// gradient(int nblk, int blksize, T theta, T delta, int * leftblk, int * rightblk, int * topblk, int * botblk, T *a, T *&dadx, T * &dady)
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, hh, dhdx_d, dhdy_d);
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, zs, dzsdx_d, dzsdy_d);
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, uu, dudx_d, dudy_d);
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, vv, dvdx_d, dvdy_d);

	double cm = 1.0;// 0.1;
	double fmu = 1.0;
	double fmv = 1.0;

	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;

				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				yminus = findbot(blksize, ix, iy, ib, botblk[ib]);
				hi = hh[i];



				double hn = hh[xminus];


				if (hi > eps || hn > eps)
				{

					double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;

					// along X
					dx = delta / 2.0;
					zi = zs[i] - hi;

					//printf("%f\n", zi);


					zl = zi - dx*(dzsdx_d[i] - dhdx_d[i]);
					//printf("%f\n", zl);

					zn = zs[xminus] - hn;

					//printf("%f\n", zn);
					zr = zn + dx*(dzsdx_d[xminus] - dhdx_d[xminus]);


					zlr = max(zl, zr);

					hl = hi - dx*dhdx_d[i];
					up = uu[i] - dx*dudx_d[i];
					hp = max(0.0, hl + zl - zlr);

					hr = hn + dx*dhdx_d[xminus];
					um = uu[xminus] + dx*dudx_d[xminus];
					hm = max(0.0, hr + zr - zlr);

					//// Reimann solver
					double fh, fu, fv;
					double dtmaxf = 1.0 / epsilon;

					//We can now call one of the approximate Riemann solvers to get the fluxes.
					kurganovd(g, CFL, hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
					fv = (fh > 0.0 ? vv[xminus] + dx*dvdx_d[xminus] : vv[i] - dx*dvdx_d[i])*fh;
					dtmax_d = dtmaxf;
					dtmaxtmp = min(dtmax_d, dtmaxtmp);
					//double cpo = sqrtf(g*hp), cmo = sqrtf(g*hm);
					//double ap = max(up + cpo, um + cmo); ap = max(ap, 0.0f);
					//double am = min(up - cpo, um - cmo); am = min(am, 0.0f);
					//double qm = hm*um, qp = hp*up;

					//double fubis= (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) +	ap*am*(qp - qm)) / (ap - am);
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
					double sl = g / 2.0*(hp*hp - hl*hl + (hl + hi)*(zi - zl));
					double sr = g / 2.0*(hm*hm - hr*hr + (hr + hn)*(zn - zr));

					////Flux update

					Fhu_d[i] = fmu * fh;
					Fqux_d[i] = fmu * (fu - sl);
					Su_d[i] = fmu * (fu - sr);
					Fqvx_d[i] = fmu * fv;
				}
				else
				{
					Fhu_d[i] = 0.0;
					Fqux_d[i] = 0.0;
					Su_d[i] = 0.0;
					Fqvx_d[i] = 0.0;
				}
			}

		}
	}
	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;

				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				yminus = findbot(blksize, ix, iy, ib, botblk[ib]);
				hi = hh[i];

				double hn = hh[yminus];
				double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;



				if (hi > eps || hn > eps)
				{


					//Along Y

					hn = hh[yminus];
					dx = delta / 2.0f;
					zi = zs[i] - hi;
					zl = zi - dx*(dzsdy_d[i] - dhdy_d[i]);
					zn = zs[yminus] - hn;
					zr = zn + dx*(dzsdy_d[ yminus] - dhdy_d[ yminus]);
					zlr = max(zl, zr);

					hl = hi - dx*dhdy_d[i];
					up = vv[i] - dx*dvdy_d[i];
					hp = max(0.0, hl + zl - zlr);

					hr = hn + dx*dhdy_d[yminus];
					um = vv[yminus] + dx*dvdy_d[yminus];
					hm = max(0.0, hr + zr - zlr);

					//// Reimann solver
					double fh, fu, fv;
					double dtmaxf = 1.0 / epsilon;
					//printf("%f\t%f\t%f\n", x[i], y[i], dhdy[i]);
					//printf("%f\n", hr);
					//We can now call one of the approximate Riemann solvers to get the fluxes.
					kurganovd(g, CFL, hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
					fv = (fh > 0. ? uu[yminus] + dx*dudy_d[yminus] : uu[i] - dx*dudy_d[i])*fh;
					dtmax_d = dtmaxf;
					dtmaxtmp = min(dtmax_d, dtmaxtmp);
					//// Topographic term

					/**
					#### Topographic source term

					In the case of adaptive refinement, care must be taken to ensure
					well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
					double sl = g / 2.0f*(hp*hp - hl*hl + (hl + hi)*(zi - zl));
					double sr = g / 2.0f*(hm*hm - hr*hr + (hr + hn)*(zn - zr));

					////Flux update

					Fhv_d[i] = fmv * fh;
					Fqvy_d[i] = fmv * (fu - sl);
					Sv_d[i] = fmv * (fu - sr);
					Fquy_d[i] = fmv* fv;

					//printf("%f\t%f\t%f\n", x[i], y[i], Fhv[i]);
				}
				else
				{
					Fhv_d[i] = 0.0;
					Fqvy_d[i] = 0.0;
					Sv_d[i] = 0.0;
					Fquy_d[i] = 0.0;
				}

				//printf("%f\t%f\t%f\n", x[i], y[i], Fquy[i]);
			}
		}
	}

	dtmax = dtmaxtmp;


	// UPDATES For evolving quantities

	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;

				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				yminus = findbot(blksize, ix, iy, ib, botblk[ib]);

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
				double cm = 1.0;
				double cmdel = 1.0 / (cm * delta);
				dh[i] = -1.0*(Fhu_d[xplus] - Fhu_d[i] + Fhv_d[yplus] - Fhv_d[i]) *cmdel;
				//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


				//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
				//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
				double dmdl = (fmu - fmu) *cmdel;// absurd!
				double dmdt = (fmv - fmv) *cmdel;// absurd!
				double fG = vv[i] * dmdl - uu[i] * dmdt;
				dhu[i] = (Fqux_d[i] + Fquy_d[i] - Su_d[xplus] - Fquy_d[yplus]) *cmdel;
				dhv[i] = (Fqvy_d[i] + Fqvx_d[i] - Sv_d[yplus] - Fqvx_d[xplus]) *cmdel;
				//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
				dhu[i] += hi * (g*hi / 2.0*dmdl + fG*vv[i]);
				dhv[i] += hi * (g*hi / 2.0*dmdt - fG*uu[i]);


			}



		}
	}
}
void update_spherical(int nblk, int blksize, double theta, double dt, double eps, double g, double CFL, double delta,double Radius,double ymax, double * blockyo, double *hh, double *zs, double *uu, double *vv, double *&dh, double *&dhu, double *&dhv)
{
	int i, xplus, yplus, xminus, yminus;

	double hi;


	dtmax =(1.0 / epsilon);
	double dtmaxtmp = dtmax;

	// calculate gradients
	// gradient(int nblk, int blksize, T theta, T delta, int * leftblk, int * rightblk, int * topblk, int * botblk, T *a, T *&dadx, T * &dady)
	gradient(nblk, blksize, theta, delta, leftblk,rightblk,topblk,botblk, hh, dhdx_d, dhdy_d);
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, zs, dzsdx_d, dzsdy_d);
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk,uu, dudx_d, dudy_d);
	gradient(nblk, blksize, theta, delta, leftblk, rightblk, topblk, botblk, vv, dvdx_d, dvdy_d);

	double cm = 1.0;// 0.1;
	double fmu = 1.0;
	double fmv = 1.0;
	double phi, dphi,y;
	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;
				
				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				yminus = findbot(blksize, ix, iy, ib, botblk[ib]);


				hi = hh[i];

				y = blockyo[ib] + iy*delta / Radius*180.0 / pi;

				phi = y*pi / 180.0;

				dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

				cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

				fmu = 1.0;
				fmv = cos(phi);


				double hn = hh[xminus];


				if (hi > eps || hn > eps)
				{

					double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;

					// along X
					dx = delta / 2.0;
					zi = zs[i] - hi;

					//printf("%f\n", zi);


					zl = zi - dx*(dzsdx_d[i] - dhdx_d[i]);
					//printf("%f\n", zl);

					zn = zs[xminus] - hn;

					//printf("%f\n", zn);
					zr = zn + dx*(dzsdx_d[xminus] - dhdx_d[xminus]);


					zlr = max(zl, zr);

					hl = hi - dx*dhdx_d[i];
					up = uu[i] - dx*dudx_d[i];
					hp = max(0.0, hl + zl - zlr);

					hr = hn + dx*dhdx_d[xminus];
					um = uu[xminus] + dx*dudx_d[xminus];
					hm = max(0.0, hr + zr - zlr);

					//// Reimann solver
					double fh, fu, fv;
					double dtmaxf = 1.0 / epsilon;

					//We can now call one of the approximate Riemann solvers to get the fluxes.
					kurganovd(g, CFL, hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
					fv = (fh > 0.0 ? vv[xminus] + dx*dvdx_d[xminus] : vv[i] - dx*dvdx_d[i])*fh;
					dtmax_d = dtmaxf;
					dtmaxtmp = min(dtmax_d, dtmaxtmp);
					//double cpo = sqrtf(g*hp), cmo = sqrtf(g*hm);
					//double ap = max(up + cpo, um + cmo); ap = max(ap, 0.0f);
					//double am = min(up - cpo, um - cmo); am = min(am, 0.0f);
					//double qm = hm*um, qp = hp*up;

					//double fubis = (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm)) / (ap - am);
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
					double sl = g / 2.0*(hp*hp - hl*hl + (hl + hi)*(zi - zl));
					double sr = g / 2.0*(hm*hm - hr*hr + (hr + hn)*(zn - zr));

					////Flux update

					Fhu_d[i] = fmu * fh;
					Fqux_d[i] = fmu * (fu - sl);
					Su_d[i] = fmu * (fu - sr);
					Fqvx_d[i] = fmu * fv;
				}
				else
				{
					Fhu_d[i] = 0.0;
					Fqux_d[i] = 0.0;
					Su_d[i] = 0.0;
					Fqvx_d[i] = 0.0;
				}
			}

		}
	}
	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;

				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				yminus = findbot(blksize, ix, iy, ib, botblk[ib]);


				hi = hh[i];

				double hn = hh[yminus];
				double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;

				y = blockyo[ib] + iy*delta / Radius*180.0 / pi;

				phi = y*pi / 180.0;

				dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

				cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

				fmu = 1.0;
				fmv = cos(phi);


				if (hi > eps || hn > eps)
				{


					//Along Y

					hn = hh[yminus];
					dx = delta / 2.0;
					zi = zs[i] - hi;
					zl = zi - dx*(dzsdy_d[i] - dhdy_d[i]);
					zn = zs[yminus] - hn;
					zr = zn + dx*(dzsdy_d[yminus] - dhdy_d[yminus]);
					zlr = max(zl, zr);

					hl = hi - dx*dhdy_d[i];
					up = vv[i] - dx*dvdy_d[i];
					hp = max(0.0, hl + zl - zlr);

					hr = hn + dx*dhdy_d[yminus];
					um = vv[yminus] + dx*dvdy_d[yminus];
					hm = max(0.0, hr + zr - zlr);

					//// Reimann solver
					double fh, fu, fv;
					double dtmaxf = 1 / (double)epsilon;
					//printf("%f\t%f\t%f\n", x[i], y[i], dhdy[i]);
					//printf("%f\n", hr);
					//We can now call one of the approximate Riemann solvers to get the fluxes.
					kurganovd(g, CFL, hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
					fv = (fh > 0.0 ? uu[yminus] + dx*dudy_d[yminus] : uu[i] - dx*dudy_d[i])*fh;
					dtmax_d = dtmaxf;
					dtmaxtmp = min(dtmax_d, dtmaxtmp);
					//// Topographic term

					/**
					#### Topographic source term

					In the case of adaptive refinement, care must be taken to ensure
					well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
					double sl = g / 2.0*(hp*hp - hl*hl + (hl + hi)*(zi - zl));
					double sr = g / 2.0*(hm*hm - hr*hr + (hr + hn)*(zn - zr));

					////Flux update

					Fhv_d[i] = fmv * fh;
					Fqvy_d[i] = fmv * (fu - sl);
					Sv_d[i] = fmv * (fu - sr);
					Fquy_d[i] = fmv* fv;

					//printf("%f\t%f\t%f\n", x[i], y[i], Fhv[i]);
				}
				else
				{
					Fhv_d[i] = 0.0;
					Fqvy_d[i] = 0.0;
					Sv_d[i] = 0.0;
					Fquy_d[i] = 0.0;
				}

				//printf("%f\t%f\t%f\n", x[i], y[i], Fquy[i]);
			}
		}
	}

	dtmax = dtmaxtmp;


	// UPDATES For evolving quantities

	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;

				xplus = findright(blksize, ix, iy, ib, rightblk[ib]);

				xminus = findleft(blksize, ix, iy, ib, leftblk[ib]);
				yplus = findtop(blksize, ix, iy, ib, topblk[ib]);
				yminus = findbot(blksize, ix, iy, ib, botblk[ib]);
				hi = hh[i];

				y = blockyo[ib] + iy*delta / Radius*180.0 / pi;
				//double yp = yo + min(iy + 1, ny - 1)*delta / Radius*180.0 / pi;

				double yp;
				if ((blockyo[ib] + (15.0 * delta / Radius*180.0 / pi)) == ymax)//if block is on the side
				{
					yp = blockyo[ib] + (min(iy+1,15))*delta / Radius*180.0 / pi;
				}
				else
				{
					yp = blockyo[ib] + (iy + 1)*delta / Radius*180.0 / pi; 
				}
					
				//yp= blockyo[ib] + (iy + 1)*delta / Radius*180.0 / pi; // Need so make this safer?

				phi = y*(double)pi / 180.0;

				dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

				cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

				fmu = 1.0;
				fmv = cos(phi);
				double fmvp = cosf(yp*pi / 180.0);
				////
				//vector dhu = vector(updates[1 + dimension*l]);
				//foreach() {
				//	double dhl =
				//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
				//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
				//	foreach_dimension()
				//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
				//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);

				double cmdel = 1.0 / (cm * delta);
				dh[i] = -1.0*(Fhu_d[xplus] - Fhu_d[i] + Fhv_d[yplus] - Fhv_d[i]) *cmdel;
				//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


				//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
				//but fmu is always ==1 event in spherical grids????
				//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
				double dmdl = (fmu - fmu) *cmdel;
				double dmdt = (fmvp - fmv) *cmdel;
				double fG = vv[i] * dmdl - uu[i] * dmdt;
				dhu[i] = (Fqux_d[i] + Fquy_d[i] - Su_d[xplus] - Fquy_d[yplus]) *cmdel;
				dhv[i] = (Fqvy_d[i] + Fqvx_d[i] - Sv_d[yplus] - Fqvx_d[xplus]) *cmdel;
				//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
				dhu[i] += hi * (g*hi / 2.0*dmdl + fG*vv[i]);
				dhv[i] += hi * (g*hi / 2.0*dmdt - fG*uu[i]);




			}

		}
	}
}



template <class T> void advance(int nblk, int blksize, T dt, T eps, T*zb, T *hh, T *zs, T *uu, T * vv, T * dh, T *dhu, T *dhv, T * &hho, T *&zso, T *&uuo, T *&vvo)
{
	//dim3 blockDim(16, 16, 1);
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);

	//adv_stvenant << <gridDim, blockDim, 0 >> > (nx, ny, dt, eps, zb_g, hh_g, zs_g, uu_g, vv_g, dh_g, dhu_g, dhv_g);


	//scalar hi = input[0], ho = output[0], dh = updates[0];
	//vector * uol = (vector *) &output[1];

	// new fields in ho[], uo[]
	//foreach() {
	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				int i = ix + iy * 16 + ib * blksize;
				T hold = hh[i];
				T ho, uo, vo;
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
					uo = T();
					vo = T();
				}



				hho[i] = ho;
				uuo[i] = uo;
				vvo[i] = vo;

			}
		}
	}

	// fixme: on trees eta is defined as eta = zb + h and not zb +
	// ho in the refine_eta() and restriction_eta() functions below
	//scalar * list = list_concat({ ho, eta }, (scalar *)uol);
	//boundary(list);
	//free(list);

	// Boundaries!!!!


}

template <class T> 
void cleanup(int nblk, int blksize, T * hhi, T *zsi, T *uui, T *vvi, T * &hho, T *&zso, T *&uuo, T *&vvo)
{
	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				int i = ix + iy * 16 + ib * blksize;
				hho[i] = hhi[i];
				zso[i] = zsi[i];
				uuo[i] = uui[i];
				vvo[i] = vvi[i];
			}
		}
	}

}





extern "C" void create2dnc(int nx, int ny, double dx, double dy, double totaltime, double *xx, double *yy, float * var)
{
	int status;
	int ncid, xx_dim, yy_dim, time_dim,  tvar_id;

	size_t nxx, nyy;
	static size_t start[] = { 0, 0, 0 }; // start at first value 
	static size_t count[] = { 1, ny, nx };
	int time_id, xx_id, yy_id;	//
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
	int ncid, recid;
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
double FlowCPU(Param XParam, double nextoutputtime)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	//forcing bnd update 
	//////////////////////////////
	//flowbnd();

	//update(int nx, int ny, double dt, double eps,double *hh, double *zs, double *uu, double *vv, double *dh, double *dhu, double *dhv)

	
	update(XParam.nblk, XParam.blksize, (float)XParam.theta, (float)XParam.dt, (float)XParam.eps, (float)XParam.g, (float)XParam.CFL, (float)XParam.delta, hh, zs, uu, vv, dh, dhu, dhv);
	//updateATM(int nblk, int blksize,int windnx,int windny,float winddx, float windxo, float windyo, float theta, float dt, float eps, float g, float CFL, float delta,float Cd, float *hh, float *zs, float *uu, float *vv, float *&dh, float *&dhu, float *&dhv, float * Uwnd, float * Vwnd)
	
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
		advance(XParam.nblk, XParam.blksize,(float)XParam.dt*0.5f, (float)XParam.eps, zb, hh, zs, uu, vv, dh, dhu, dhv, hho, zso, uuo, vvo);

		//corrector
		update(XParam.nblk, XParam.blksize, (float)XParam.theta, (float)XParam.dt, (float)XParam.eps, (float)XParam.g, (float)XParam.CFL, (float)XParam.delta, hho, zso, uuo, vvo, dh, dhu, dhv);
		
		
	}
	//
	advance(XParam.nblk, XParam.blksize, (float) XParam.dt, (float) XParam.eps, zb, hh, zs, uu, vv, dh, dhu, dhv, hho, zso, uuo, vvo);

	cleanup(XParam.nblk, XParam.blksize, hho, zso, uuo, vvo, hh, zs, uu, vv);
	
	bottomfrictionCPU(XParam.nblk, XParam.blksize, XParam.frictionmodel, (float)XParam.dt, (float) XParam.eps, cf, hh, uu, vv);
	//write2varnc(nx, ny, totaltime, hh);
	if (XParam.Rivers.size() > 1)
	{
		discharge_bnd_v_CPU(XParam, zs, hh);
	}
	//noslipbndallCPU(nx, ny, XParam.dt, XParam.eps, zb, zs, hh, uu, vv);
	return XParam.dt;


}

double FlowCPUATM(Param XParam, double nextoutputtime, int cstwind,int cstpress,float Uwindi, float Vwindi)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	//forcing bnd update 
	//////////////////////////////
	//flowbnd();

	//update(int nx, int ny, double dt, double eps,double *hh, double *zs, double *uu, double *vv, double *dh, double *dhu, double *dhv)


	//update(XParam.nblk, XParam.blksize, (float)XParam.theta, (float)XParam.dt, (float)XParam.eps, (float)XParam.g, (float)XParam.CFL, (float)XParam.delta, hh, zs, uu, vv, dh, dhu, dhv);
	updateATM(XParam.nblk, XParam.blksize,cstwind,cstpress, XParam.windU.nx, XParam.windU.ny, XParam.windU.dx, XParam.windU.xo, XParam.windU.yo, Uwindi, Vwindi, (float)XParam.theta, (float)XParam.dt, (float)XParam.eps, (float)XParam.g, (float)XParam.CFL, (float)XParam.delta,(float)XParam.Cd, (float)XParam.Pa2m, hh, zs, uu, vv, dh, dhu, dhv, Uwind, Vwind,Patm);

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
		advance(XParam.nblk, XParam.blksize, (float)XParam.dt*0.5f, (float)XParam.eps, zb, hh, zs, uu, vv, dh, dhu, dhv, hho, zso, uuo, vvo);

		//corrector
		//update(XParam.nblk, XParam.blksize, (float)XParam.theta, (float)XParam.dt, (float)XParam.eps, (float)XParam.g, (float)XParam.CFL, (float)XParam.delta, hho, zso, uuo, vvo, dh, dhu, dhv);
		updateATM(XParam.nblk, XParam.blksize, cstwind, cstpress, XParam.windU.nx, XParam.windU.ny, XParam.windU.dx, XParam.windU.xo, XParam.windU.yo, Uwindi, Vwindi, (float)XParam.theta, (float)XParam.dt, (float)XParam.eps, (float)XParam.g, (float)XParam.CFL, (float)XParam.delta, (float)XParam.Cd, (float)XParam.Pa2m, hho, zso, uuo, vvo, dh, dhu, dhv, Uwind, Vwind, Patm);


	}
	//
	advance(XParam.nblk, XParam.blksize, (float)XParam.dt, (float)XParam.eps, zb, hh, zs, uu, vv, dh, dhu, dhv, hho, zso, uuo, vvo);

	cleanup(XParam.nblk, XParam.blksize, hho, zso, uuo, vvo, hh, zs, uu, vv);

	bottomfrictionCPU(XParam.nblk, XParam.blksize, XParam.frictionmodel, (float)XParam.dt, (float)XParam.eps, cf, hh, uu, vv);
	//write2varnc(nx, ny, totaltime, hh);
	if (XParam.Rivers.size() > 1)
	{
		discharge_bnd_v_CPU(XParam, zs, hh);
	}
	//noslipbndallCPU(nx, ny, XParam.dt, XParam.eps, zb, zs, hh, uu, vv);
	return XParam.dt;


}

double FlowCPUSpherical(Param XParam, double nextoutputtime)
{
	//in spherical mode a special correction is made in update and all need to be in double to remove the 
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	//int nblk = XParam.nblk;
	//int blksize = XParam.blksize;
	//forcing bnd update 
	//////////////////////////////
	//flowbnd();

	//update(int nx, int ny, double dt, double eps,double *hh, double *zs, double *uu, double *vv, double *dh, double *dhu, double *dhv)

	//update_spherical(int nblk, int blksize, double theta, double dt, double eps, double g, double CFL, double delta,double Radius, double * blockyo, double *hh, double *zs, double *uu, double *vv, double *&dh, double *&dhu, double *&dhv)
	update_spherical(XParam.nblk, XParam.blksize, XParam.theta, XParam.dt, XParam.eps, XParam.g, XParam.CFL, XParam.delta, XParam.Radius, (XParam.yo + (ceil(XParam.ny / 16.0)*16.0 - 1)*XParam.dx), blockyo_d, hh_d, zs_d, uu_d, vv_d, dh_d, dhu_d, dhv_d);
	

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
		advance(XParam.nblk, XParam.blksize, XParam.dt*0.5, XParam.eps, zb_d,hh_d, zs_d, uu_d, vv_d, dh_d, dhu_d, dhv_d, hho_d, zso_d, uuo_d, vvo_d);

		//corrector
		update_spherical(XParam.nblk, XParam.blksize, XParam.theta, XParam.dt, XParam.eps, XParam.g, XParam.CFL, XParam.delta, XParam.Radius, (XParam.yo + (ceil(XParam.ny / 16.0)*16.0 - 1)*XParam.dx), blockyo_d, hho_d, zso_d, uuo_d, vvo_d, dh_d, dhu_d, dhv_d);
		

	}
	//
	advance(XParam.nblk, XParam.blksize, XParam.dt, XParam.eps, zb_d, hh_d, zs_d, uu_d, vv_d, dh_d, dhu_d, dhv_d, hho_d, zso_d, uuo_d, vvo_d);

	cleanup(XParam.nblk, XParam.blksize, hho_d, zso_d, uuo_d, vvo_d, hh_d, zs_d, uu_d, vv_d);

	bottomfrictionCPU(XParam.nblk, XParam.blksize, XParam.frictionmodel, XParam.dt, XParam.eps, cf_d, hh_d, uu_d, vv_d);
	//write2varnc(nx, ny, totaltime, hh);

	//noslipbndallCPU(nx, ny, XParam.dt, XParam.eps, zb, zs, hh, uu, vv);
	return XParam.dt;
}

double FlowCPUDouble(Param XParam, double nextoutputtime)
{
	//in spherical mode a special correction is made in update and all need to be in double to remove the 
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	//forcing bnd update 
	//////////////////////////////
	//flowbnd();

	//update(int nx, int ny, double dt, double eps,double *hh, double *zs, double *uu, double *vv, double *dh, double *dhu, double *dhv)


	updateD(XParam.nblk, XParam.blksize, XParam.theta, XParam.dt, XParam.eps, XParam.g, XParam.CFL, XParam.delta, hh_d, zs_d, uu_d, vv_d, dh_d, dhu_d, dhv_d);


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
		advance(XParam.nblk, XParam.blksize, XParam.dt*0.5, XParam.eps, zb_d, hh_d, zs_d, uu_d, vv_d, dh_d, dhu_d, dhv_d, hho_d, zso_d, uuo_d, vvo_d);

		//corrector
		updateD(XParam.nblk, XParam.blksize, XParam.theta, XParam.dt, XParam.eps, XParam.g, XParam.CFL, XParam.delta, hho_d, zso_d, uuo_d, vvo_d, dh_d, dhu_d, dhv_d);


	}
	//
	advance(XParam.nblk, XParam.blksize, XParam.dt, XParam.eps, zb_d, hh_d, zs_d, uu_d, vv_d, dh_d, dhu_d, dhv_d, hho_d, zso_d, uuo_d, vvo_d);

	cleanup(XParam.nblk, XParam.blksize, hho_d, zso_d, uuo_d, vvo_d, hh_d, zs_d, uu_d, vv_d);

	bottomfrictionCPU(XParam.nblk, XParam.blksize, XParam.frictionmodel, XParam.dt, XParam.eps,cf_d, hh_d, uu_d, vv_d);
	//write2varnc(nx, ny, totaltime, hh);
	if (XParam.Rivers.size() > 1)
	{
		discharge_bnd_v_CPU(XParam, zs_d, hh_d);
	}
	//noslipbndallCPU(nx, ny, XParam.dt, XParam.eps, zb, zs, hh, uu, vv);
	return XParam.dt;


}

void leftdirichletCPU_old(int nx, int ny, float g, std::vector<double> zsbndvec, float *zs, float *zb, float *hh, float *uu, float *vv)
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
			zsbnd = (float) interptime(zsbndvec[inext], zsbndvec[iprev], (float)(inext - iprev), (float)(iy - iprev));
		}
		int ix = 0;
		int i = ix + iy*nx;
		int xplus;
		
		//if (ix == 0 && iy < ny)
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

void leftdirichletCPU(int nblk, int blksize, float xo,float yo, float g, float dx, std::vector<double> zsbndvec, float * blockxo,float * blockyo, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	float zsbnd;
	float xi, yi,jj;

	for (int ib = 0; ib < nblk; ib++) //scan each block
	{
		if (blockxo[ib] == xo)//if block is on the side
		{
			int i = 0;
			for (int j = 0; j < 16; j++)
			{
				int n = i + j * 16 + ib * blksize;

				xi = blockxo[ib] + i*dx;
				yi = blockyo[ib] + j*dx;

				jj = (yi - yo) / dx;
				//ii = (xi - xo) / dx;
				if (zsbndvec.size() == 1)
				{
					zsbnd = zsbndvec[0];
				}
				else
				{
					int iprev = min(max((int)ceil(jj / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
					int inext = iprev + 1;
					// here interp time is used to interpolate to the right node rather than in time...
					zsbnd = (float)interptime(zsbndvec[inext], zsbndvec[iprev], (float)(inext - iprev), (float)(jj - iprev));
				}
				

				if (zsbnd>zb[n])
				{
					int nright = i+1 + j * 16 + ib * blksize;;
					hh[n] = zsbnd - zb[n];
					zs[n] = zsbnd;
					uu[n] = -2.0f*(sqrtf(g*max(hh[nright], 0.0f)) - sqrtf(g*max(zsbnd - zb[nright], 0.0f))) + uu[nright];
					vv[n] = 0.0f;
					//if (iy == 0)
					//{
					//	printf("zsbnd=%f\t", zsbnd);
					//}
				}
			}
		}
	}
}

void leftdirichletCPUD(int nblk, int blksize, double xo, double yo, double g, double dx, std::vector<double> zsbndvec, double * blockxo, double * blockyo, double *zs, double *zb, double *hh, double *uu, double *vv)
{
	double zsbnd;
	double xi, yi, jj;

	for (int ib = 0; ib < nblk; ib++) //scan each block
	{
		if (blockxo[ib] == xo)//if block is on the side
		{
			int i = 0;
			for (int j = 0; j < 16; j++)
			{
				int n = i + j * 16 + ib * blksize;

				xi = blockxo[ib] + i*dx;
				yi = blockyo[ib] + j*dx;

				jj = (yi - yo) / dx;
				//ii = (xi - xo) / dx;
				if (zsbndvec.size() == 1)
				{
					zsbnd = zsbndvec[0];
				}
				else
				{
					int iprev = min(max((int)ceil(jj / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
					int inext = iprev + 1;
					// here interp time is used to interpolate to the right node rather than in time...
					zsbnd = interptime(zsbndvec[inext], zsbndvec[iprev], (inext - iprev), (jj - iprev));
				}


				if (zsbnd>zb[n])
				{
					int nright = i + 1 + j * 16 + ib * blksize;;
					hh[n] = zsbnd - zb[n];
					zs[n] = zsbnd;
					uu[n] = -2.0*(sqrt(g*max(hh[nright], 0.0)) - sqrt(g*max(zsbnd - zb[nright], 0.0))) + uu[nright];
					vv[n] = 0.0;
					//if (iy == 0)
					//{
					//	printf("zsbnd=%f\t", zsbnd);
					//}
				}
			}
		}
	}
}



void rightdirichletCPU(int nblk, int blksize, int nx, float xo, float yo, float g, float dx, std::vector<double> zsbndvec, float * blockxo, float * blockyo, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	float zsbnd;
	float xi, yi, jj, ii;

	for (int ib = 0; ib < nblk; ib++) //scan each block
	{
		if (blockxo[ib] + (15 * dx) == xo + (ceil(nx / 16.0)*16.0 - 1)*dx)//if block is on the side
		{
			int i = 15;
			for (int j = 0; j < 16; j++)
			{
				int n = i + j * 16 + ib * blksize;

				xi = blockxo[ib] + i*dx;
				yi = blockyo[ib] + j*dx;

				jj = (yi - yo) / dx;
				ii = (xi - xo) / dx;
				if (zsbndvec.size() == 1)
				{
					zsbnd = zsbndvec[0];
				}
				else
				{
					int iprev = min(max((int)ceil(jj / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
					int inext = iprev + 1;
					// here interp time is used to interpolate to the right node rather than in time...
					zsbnd = interptime(zsbndvec[inext], zsbndvec[iprev], (inext - iprev), (jj - iprev));
				}


				if (zsbnd>zb[n])
				{
					int nleft = i - 1 + j * 16 + ib * blksize;;
					hh[n] = zsbnd - zb[n];
					zs[n] = zsbnd;
					uu[n] = +2.0f*(sqrtf(g*max(hh[nleft], 0.0f)) - sqrtf(g*max(zsbnd - zb[nleft], 0.0f))) + uu[nleft];
					vv[n] = 0.0f;
					//if (iy == 0)
					//{
					//	printf("zsbnd=%f\t", zsbnd);
					//}
				}
			}
		}
	}
}

void rightdirichletCPUD(int nblk, int blksize,int nx, double xo, double yo, double g, double dx, std::vector<double> zsbndvec,  double * blockxo, double * blockyo, double *zs, double *zb, double *hh, double *uu, double *vv)
{
	double zsbnd;
	double xi, yi, jj, ii;

	for (int ib = 0; ib < nblk; ib++) //scan each block
	{
		if (blockxo[ib] + (15 * dx) == xo + (ceil(nx / 16.0)*16.0 - 1)*dx)//if block is on the side
		{
			int i = 15;
			for (int j = 0; j < 16; j++)
			{
				int n = i + j * 16 + ib * blksize;

				xi = blockxo[ib] + i*dx;
				yi = blockyo[ib] + j*dx;

				jj = (yi - yo) / dx;
				ii = (xi - xo) / dx;
				if (zsbndvec.size() == 1)
				{
					zsbnd = zsbndvec[0];
				}
				else
				{
					int iprev = min(max((int)ceil(jj / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
					int inext = iprev + 1;
					// here interp time is used to interpolate to the right node rather than in time...
					zsbnd = interptime(zsbndvec[inext], zsbndvec[iprev], (inext - iprev), (jj - iprev));
				}


				if (zsbnd>zb[n])
				{
					int nleft = i - 1 + j * 16 + ib * blksize;;
					hh[n] = zsbnd - zb[n];
					zs[n] = zsbnd;
					uu[n] = +2.0*(sqrt(g*max(hh[nleft], 0.0)) - sqrt(g*max(zsbnd - zb[nleft], 0.0))) + uu[nleft];
					vv[n] = 0.0;
					//if (iy == 0)
					//{
					//	printf("zsbnd=%f\t", zsbnd);
					//}
				}
			}
		}
	}
}

void topdirichletCPU(int nblk, int blksize, int ny, float xo, float yo, float g, float dx, std::vector<double> zsbndvec, float * blockxo, float * blockyo, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	float zsbnd;
	float xi, yi, jj, ii;

	for (int ib = 0; ib < nblk; ib++) //scan each block
	{
		if (blockyo[ib] + (15 * dx) == yo + (ceil(ny / 16.0)*16.0 - 1)*dx)//if block is on the side
		{
			int j = 15;
			for (int i = 0; i < 16; i++)
			{
				int n = i + j * 16 + ib * blksize;

				xi = blockxo[ib] + i*dx;
				yi = blockyo[ib] + j*dx;

				jj = (yi - yo) / dx;
				ii = (xi - xo) / dx;
				if (zsbndvec.size() == 1)
				{
					zsbnd = zsbndvec[0];
				}
				else
				{
					int iprev = min(max((int)ceil(ii / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
					int inext = iprev + 1;
					// here interp time is used to interpolate to the right node rather than in time...
					zsbnd = interptime(zsbndvec[inext], zsbndvec[iprev], (inext - iprev), (ii - iprev));
				}


				if (zsbnd>zb[n])
				{
					int nbot = i + (j - 1) * 16 + ib * blksize;;
					hh[n] = zsbnd - zb[n];
					zs[n] = zsbnd;
					vv[n] = +2.0f*(sqrtf(g*max(hh[nbot], 0.0f)) - sqrtf(g*max(zsbnd - zb[nbot], 0.0f))) + vv[nbot];
					uu[n] = 0.0f;
					//if (iy == 0)
					//{
					//	printf("zsbnd=%f\t", zsbnd);
					//}
				}
			}
		}
	}
}

void topdirichletCPUD(int nblk, int blksize, int ny, double xo, double yo, double g, double dx, std::vector<double> zsbndvec, double * blockxo, double * blockyo, double *zs, double *zb, double *hh, double *uu, double *vv)
{
	double zsbnd;
	double xi, yi, jj, ii;

	for (int ib = 0; ib < nblk; ib++) //scan each block
	{
		if (blockyo[ib] + (15 * dx) == yo + (ceil(ny / 16.0)*16.0 - 1)*dx)//if block is on the side
		{
			int j = 15;
			for (int i = 0; i < 16; i++)
			{
				int n = i + j * 16 + ib * blksize;

				xi = blockxo[ib] + i*dx;
				yi = blockyo[ib] + j*dx;

				jj = (yi - yo) / dx;
				ii = (xi - xo) / dx;
				if (zsbndvec.size() == 1)
				{
					zsbnd = zsbndvec[0];
				}
				else
				{
					int iprev = min(max((int)ceil(ii/ (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
					int inext = iprev + 1;
					// here interp time is used to interpolate to the right node rather than in time...
					zsbnd = interptime(zsbndvec[inext], zsbndvec[iprev], (inext - iprev), (ii - iprev));
				}


				if (zsbnd>zb[n])
				{
					int nbot = i  + (j-1) * 16 + ib * blksize;;
					hh[n] = zsbnd - zb[n];
					zs[n] = zsbnd;
					vv[n] = +2.0*(sqrt(g*max(hh[nbot], 0.0)) - sqrt(g*max(zsbnd - zb[nbot], 0.0))) + vv[nbot];
					uu[n] = 0.0;
					//if (iy == 0)
					//{
					//	printf("zsbnd=%f\t", zsbnd);
					//}
				}
			}
		}
	}
}

void botdirichletCPU(int nblk, int blksize, int ny, float xo, float yo, float g, float dx, std::vector<double> zsbndvec, float * blockxo, float * blockyo, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	float zsbnd;
	float xi, yi, jj, ii;

	for (int ib = 0; ib < nblk; ib++) //scan each block
	{
		if (blockyo[ib] == yo)//if block is on the side
		{
			int j = 0;
			for (int i = 0; i < 16; i++)
			{
				int n = i + j * 16 + ib * blksize;

				xi = blockxo[ib] + i*dx;
				yi = blockyo[ib] + j*dx;

				jj = (yi - yo) / dx;
				ii = (xi - xo) / dx;
				if (zsbndvec.size() == 1)
				{
					zsbnd = zsbndvec[0];
				}
				else
				{
					int iprev = min(max((int)ceil(ii / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
					int inext = iprev + 1;
					// here interp time is used to interpolate to the right node rather than in time...
					zsbnd = interptime(zsbndvec[inext], zsbndvec[iprev], (inext - iprev), (ii - iprev));
				}


				if (zsbnd>zb[n])
				{
					int ntop = i + (j + 1) * 16 + ib * blksize;;
					hh[n] = zsbnd - zb[n];
					zs[n] = zsbnd;
					vv[n] = -2.0f*(sqrtf(g*max(hh[ntop], 0.0f)) - sqrtf(g*max(zsbnd - zb[ntop], 0.0f))) + vv[ntop];
					uu[n] = 0.0f;
					//if (iy == 0)
					//{
					//	printf("zsbnd=%f\t", zsbnd);
					//}
				}
			}
		}
	}
}

void botdirichletCPUD(int nblk, int blksize, int ny, double xo, double yo, double g, double dx, std::vector<double> zsbndvec, double * blockxo, double * blockyo, double *zs, double *zb, double *hh, double *uu, double *vv)
{
	double zsbnd;
	double xi, yi, jj, ii;

	for (int ib = 0; ib < nblk; ib++) //scan each block
	{
		if (blockyo[ib]  == yo )//if block is on the side
		{
			int j = 0;
			for (int i = 0; i < 16; i++)
			{
				int n = i + j * 16 + ib * blksize;

				xi = blockxo[ib] + i*dx;
				yi = blockyo[ib] + j*dx;

				jj = (yi - yo) / dx;
				ii = (xi - xo) / dx;
				if (zsbndvec.size() == 1)
				{
					zsbnd = zsbndvec[0];
				}
				else
				{
					int iprev = min(max((int)ceil(ii / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
					int inext = iprev + 1;
					// here interp time is used to interpolate to the right node rather than in time...
					zsbnd = interptime(zsbndvec[inext], zsbndvec[iprev], (inext - iprev), (ii - iprev));
				}


				if (zsbnd>zb[n])
				{
					int ntop = i + (j + 1) * 16 + ib * blksize;;
					hh[n] = zsbnd - zb[n];
					zs[n] = zsbnd;
					vv[n] = -2.0*(sqrt(g*max(hh[ntop], 0.0)) - sqrt(g*max(zsbnd - zb[ntop], 0.0))) + vv[ntop];
					uu[n] = 0.0;
					//if (iy == 0)
					//{
					//	printf("zsbnd=%f\t", zsbnd);
					//}
				}
			}
		}
	}
}


template <class T> 
void bottomfrictionCPU(int nblk, int blksize, int smart, T dt, T eps, T* cf, T *hh, T *uu, T *vv)
{
	T ee = T(2.71828182845905);

	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				int i = ix + iy * 16 + ib * blksize;




				T hhi = hh[i];
				if (hhi > eps)
				{
					T cfi = cf[i];
					if (smart == 1)
					{
						T zo = cf[i];
						if (hhi / zo < ee)
						{
							cfi = zo / (T(0.46)*hhi);
						}
						else
						{
							cfi = T(1.0) / (T(2.5)*(log(hhi / zo) - T(1.0) + T(1.359)*zo / hhi));
						}
						cfi=cfi*cfi;
					}
					T normu = uu[i] * uu[i] + vv[i] * vv[i];
					T tb = cfi * sqrt(normu) / hhi*dt;
					//u.x[] = h[]>dry ? u.x[] / (1 + dt*cf*norm(u) / h[]) : 0.;
					uu[i] = uu[i] / (T(1.0) + tb);
					vv[i] = vv[i] / (T(1.0) + tb);
				}
			}
			
		}
	}

}

template <class T>
void discharge_bnd_v_CPU(Param XParam,T*zs,T*hh)
{
	T qnow;
	for (int Rin = 0; Rin < XParam.Rivers.size(); Rin++)
	{
		
		//qnow = interptime(slbnd[SLstepinbnd].wlev0, slbnd[SLstepinbnd - 1].wlev0, slbnd[SLstepinbnd].time - slbnd[SLstepinbnd - 1].time, totaltime - slbnd[SLstepinbnd - 1].time);
		int bndstep = 0;
		double difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
		while (difft <= 0.0) // danger?
		{
			bndstep++;
			difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
		}
		
		qnow = interptime(XParam.Rivers[Rin].flowinput[bndstep].q, XParam.Rivers[Rin].flowinput[max(bndstep-1,0)].q, XParam.Rivers[Rin].flowinput[bndstep].time - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time,XParam.totaltime- XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time);
		
		for (int nc = 0; nc < XParam.Rivers[Rin].i.size(); nc++)
		{
			int i = XParam.Rivers[Rin].i[nc] + XParam.Rivers[Rin].j[nc] * 16 + XParam.Rivers[Rin].block[nc] *(XParam.blksize);
			T dzsdt = qnow*XParam.dt / XParam.Rivers[Rin].disarea;
			zs[i] = zs[i] + dzsdt;
			// Do hh[i] too although Im not sure it is worth it
			hh[i] = hh[i] + dzsdt;
		}

		
	}


	

}

void noslipbndLCPU(Param XParam)
{
	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{
		//noslipbndLeftCPU(XParam.nx, XParam.ny, XParam.eps, zb_d, zs_d, hh_d, uu_d, vv_d);
		noslipbndLeftCPU(XParam.nblk, XParam.blksize, XParam.xo, XParam.eps, blockxo_d, zb_d, zs_d, hh_d, uu_d, vv_d);
	}
	else
	{
		// Left Wall
		noslipbndLeftCPU(XParam.nblk, XParam.blksize, (float)XParam.xo, (float)XParam.eps, blockxo, zb, zs, hh, uu, vv);
	}
}
void noslipbndRCPU(Param XParam)
{
	//
	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{
		//noslipbndRightCPU(XParam.nx, XParam.ny, XParam.eps, zb_d, zs_d, hh_d, uu_d, vv_d);
		noslipbndRightCPU(XParam.nblk, XParam.blksize, XParam.xo, XParam.xmax, XParam.eps, XParam.dx, blockxo_d, zb_d, zs_d, hh_d, uu_d, vv_d);
	}
	else
	{
		// Right Wall
		noslipbndRightCPU(XParam.nblk, XParam.blksize, (float)XParam.xo, (float)XParam.xmax, (float)XParam.eps, (float)XParam.dx, blockxo, zb, zs, hh, uu, vv);
	}
}
void noslipbndTCPU(Param XParam)
{
	//
	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{
		//
		//noslipbndTopCPU(XParam.nx, XParam.ny, XParam.eps, zb_d, zs_d, hh_d, uu_d, vv_d);
		noslipbndTopCPU(XParam.nblk, XParam.blksize, XParam.yo, XParam.ymax, XParam.eps, XParam.dx, blockyo_d, zb_d, zs_d, hh_d, uu_d, vv_d);
	}
	else
	{
		// Top Wall
		noslipbndTopCPU(XParam.nblk, XParam.blksize, (float)XParam.yo, (float)XParam.ymax, (float)XParam.eps, (float)XParam.dx, blockyo, zb, zs, hh, uu, vv);
	}
}
void noslipbndBCPU(Param XParam)
{
	//
	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{
		//noslipbndBotCPU(XParam.nx, XParam.ny, XParam.eps, zb_d, zs_d, hh_d, uu_d, vv_d);
		noslipbndBotCPU(XParam.nblk, XParam.blksize, XParam.yo, XParam.eps, blockyo_d, zb_d, zs_d, hh_d, uu_d, vv_d);
	}
	else
	{
		// Bottom Wall
		noslipbndBotCPU(XParam.nblk, XParam.blksize, (float)XParam.yo, (float)XParam.eps, blockyo, zb, zs, hh, uu, vv);
	}
}

template <class T> void noslipbndLeftCPU(int nblk, int blksize, T xo, T eps,T* blockxo,  T *zb, T *zs, T *hh, T *uu, T *vv)
{
	
	for (int ib = 0; ib < nblk; ib++) //scan each block
	{
		if (blockxo[ib] == xo)//if block is on the side
		{
			int i = 0;
			for (int j = 0; j < 16; j++)
			{
				int n = i + j * 16 + ib * blksize;
				int nplus = i+1 + j * 16 + ib * blksize;
				//xminus = max(ix - 1, 0);
				//yplus = min(iy + 1, ny - 1);
				//yminus = max(iy - 1, 0);


				uu[n] = T(0.0);
				zs[n] = zs[nplus];
				hh[n] = max(zs[nplus] - zb[n], eps);
			}
		}
		
	}
}

template <class T> void noslipbndRightCPU(int nblk, int blksize, T xo, T xmax, T eps, T dx, T* blockxo, T *zb, T *zs, T *hh, T *uu, T *vv)
{

	for (int ib = 0; ib < nblk; ib++) //scan each block
	{
		if (blockxo[ib] + (15 * dx) == xmax)//if block is on the side
		{
			int i = 15;
			for (int j = 0; j < 16; j++)
			{
				int n = i + j * 16 + ib * blksize;
				int xminus = (i-1) + j * 16 + ib * blksize;
				//xplus = min(ix + 1, nx - 1);
				
				//yplus = min(iy + 1, ny - 1);
				//yminus = max(iy - 1, 0);


				uu[n] = T(0.0);
				zs[n] = zs[xminus];
				hh[n] = max(zs[xminus] - zb[n], eps);
			}
		}
	}
			


	

}
template <class T> void noslipbndTopCPU(int nblk, int blksize, T yo, T ymax, T eps, T dx, T* blockyo, T *zb, T *zs, T *hh, T *uu, T *vv)
{

	for (int ib = 0; ib < nblk; ib++) //scan each block
	{
		//printf("bymax=%f\tymax=%f\n", blockyo[ib] + (15.0 * dx), yo + (ceil(ny / 16.0)*16.0-1)*dx);

		if ((blockyo[ib] + (15.0 * dx)) == ymax)//if block is on the side
		{
			int j = 15;
			for (int i = 0; i < 16; i++)
			{
				int n = i + j * 16 + ib * blksize;
				int yminus = i  + (j - 1)* 16 + ib * blksize;
				
				

				vv[n] = T(0.0);
				zs[n] = zs[ yminus];
				hh[n] = max(zs[yminus] - zb[n], eps);

				//printf("zs[n]=%f\tzs[yminus]=%f\n", zs[n], zs[yminus]);

			}
		}




	}

}

template <class T> void noslipbndBotCPU(int nblk, int blksize,T yo, T eps, T* blockyo, T *zb, T *zs, T *hh, T *uu, T *vv)
{

	int n;
	int yplus;
	

	for (int ib = 0; ib < nblk; ib++) //scan each block
	{
		if (blockyo[ib] == yo)//if block is on the side
		{
			int j = 0;
			for (int i = 0; i < 16; i++)
			{
				n = i + j * 16 + ib * blksize;
				yplus = i  + (j+ 1) * 16 + ib * blksize;
				
				//xplus = min(ix + 1, nx - 1);
				//xminus = max(ix - 1, 0);
				//yplus = min(iy + 1, ny - 1);
				//yminus = max(iy - 1, 0);


				vv[n] = T(0.0);
				zs[n] = zs[yplus];
				hh[n] = max(zs[yplus] - zb[n], eps);
			}
		}




	}

}
/*
void noslipbndallCPU(int nx, int ny, float dt, float eps, float *zb, float *zs, float *hh, float *uu, float *vv)
{
	
	int i; 
	int  xplus, yplus, xminus, yminus;
	

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
*/

//Functions below use update global variable so we need two copies for the float and double cases

void AddmeanCPU(Param XParam)
{
	int nblk = XParam.nblk;
	int blksize = XParam.blksize;

	if (XParam.outhhmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					hhmean[i] = hhmean[i] + hh[i];
				}
			}
		}
	}

	if (XParam.outzsmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					zsmean[i] = zsmean[i] + zs[i];
				}
			}
		}
	}

	if (XParam.outuumean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					uumean[i] = uumean[i] + uu[i];
				}
			}
		}
	}
	if (XParam.outvvmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					vvmean[i] = vvmean[i] + vv[i];
				}
			}
		}
	}


}

void AddmeanCPUD(Param XParam)
{
	int nblk = XParam.nblk;
	int blksize = XParam.blksize;

	if (XParam.outhhmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					hhmean_d[i] = hhmean_d[i] + hh_d[i];
				}
			}
		}
	}

	if (XParam.outzsmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					zsmean_d[i] = zsmean_d[i] + zs_d[i];
				}
			}
		}
	}

	if (XParam.outuumean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					uumean_d[i] = uumean_d[i] + uu_d[i];
				}
			}
		}
	}
	if (XParam.outvvmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					vvmean_d[i] = vvmean_d[i] + vv_d[i];
				}
			}
		}
	}


}

void DivmeanCPU(Param XParam, float nstep)
{
	int nblk = XParam.nblk;
	int blksize = XParam.blksize;

	if (XParam.outhhmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					hhmean[i] = hhmean[i] / nstep;
				}
			}
		}
	}

	if (XParam.outzsmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					zsmean[i] = zsmean[i] / nstep;
				}
			}
		}
	}

	if (XParam.outuumean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					uumean[i] = uumean[i] / nstep;
				}
			}
		}
	}
	if (XParam.outvvmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					vvmean[i] = vvmean[i] / nstep;
				}
			}
		}
	}


}

void DivmeanCPUD(Param XParam, float nstep)
{
	int nblk = XParam.nblk;
	int blksize = XParam.blksize;

	if (XParam.outhhmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					hhmean_d[i] = hhmean_d[i] / nstep;
				}
			}
		}
	}

	if (XParam.outzsmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					zsmean_d[i] = zsmean_d[i] / nstep;
				}
			}
		}
	}

	if (XParam.outuumean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					uumean_d[i] = uumean_d[i] / nstep;
				}
			}
		}
	}
	if (XParam.outvvmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					vvmean_d[i] = vvmean_d[i] / nstep;
				}
			}
		}
	}


}

void ResetmeanCPU(Param XParam)
{
	int nblk = XParam.nblk;
	int blksize = XParam.blksize;

	if (XParam.outhhmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					hhmean[i] = 0.0;
				}
			}
		}
	}

	if (XParam.outzsmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					zsmean[i] = 0.0;
				}
			}
		}
	}

	if (XParam.outuumean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					uumean[i] = 0.0;
				}
			}
		}
	}
	if (XParam.outvvmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					vvmean[i] = 0.0;
				}
			}
		}
	}


}

void ResetmeanCPUD(Param XParam)
{
	int nblk = XParam.nblk;
	int blksize = XParam.blksize;

	if (XParam.outhhmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					hhmean_d[i] = 0.0;
				}
			}
		}
	}

	if (XParam.outzsmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					zsmean_d[i] = 0.0;
				}
			}
		}
	}

	if (XParam.outuumean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					uumean_d[i] = 0.0;
				}
			}
		}
	}
	if (XParam.outvvmean == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					vvmean_d[i] = 0.0;
				}
			}
		}
	}


}

void maxallCPU(Param XParam)
{
	int nblk = XParam.nblk;
	int blksize = XParam.blksize;

	if (XParam.outhhmax == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					hhmax[i] = max(hhmax[i], hh[i]);
				}
			}
		}
	}
	if (XParam.outzsmax == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					zsmax[i] = max(zsmax[i], zs[i]);
				}
			}
		}
	}
	if (XParam.outuumax == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					uumax[i] = max(uumax[i], uu[i]);
				}
			}
		}
	}
	if (XParam.outvvmax == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					vvmax[i] = max(vvmax[i], vv[i]);
				}
			}
		}
	}
}

void maxallCPUD(Param XParam)
{
	int nblk = XParam.nblk;
	int blksize = XParam.blksize;

	if (XParam.outhhmax == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					hhmax_d[i] = max(hhmax_d[i], hh_d[i]);
				}
			}
		}
	}
	if (XParam.outzsmax == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					zsmax_d[i] = max(zsmax_d[i], zs_d[i]);
				}
			}
		}
	}
	if (XParam.outuumax == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					uumax_d[i] = max(uumax_d[i], uu_d[i]);
				}
			}
		}
	}
	if (XParam.outvvmax == 1)
	{
		for (int ib = 0; ib < nblk; ib++)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					int i = ix + iy * 16 + ib * blksize;
					vvmax_d[i] = max(vvmax_d[i], vv_d[i]);
				}
			}
		}
	}
}

void CalcVort(Param XParam)
{
	int nblk = XParam.nblk;
	int blksize = XParam.blksize;
	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				int i = ix + iy * 16 + ib * blksize;
				vort[i] = dvdy[i] - dudx[i];
			}
		}
	}
}

void CalcVortD(Param XParam)
{
	int nblk = XParam.nblk;
	int blksize = XParam.blksize;
	for (int ib = 0; ib < nblk; ib++)
	{
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				int i = ix + iy * 16 + ib * blksize;
				vort_d[i] = dvdy_d[i] - dudx_d[i];
			}
		}
	}
}
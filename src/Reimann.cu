#include "Reimann.h"

template <class T> __global__ void UpdateButtingerXGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];

	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps) + epsi;
	T delta = calcres(T(XParam.dx), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);




	T dhdxi = XGrad.dhdx[i];
	T dhdxmin = XGrad.dhdx[ileft];
	T cm = T(1.0);
	T fmu = T(1.0);

	T hi = XEv.h[i];

	T hn = XEv.h[ileft];


	if (hi > eps || hn > eps)
	{
		T dx, zi, zn, hr, hl, etar, etal, zr, zl, zA, zCN, hCNr, hCNl;
		T ui, vi, uli, vli, dhdxi, dhdxil, dudxi, dudxil, dvdxi,dvdxil;
		// along X
		dx = delta * T(0.5);
		zi = zb[i];
		zn = zb[ileft];

		ui = XEv.u[i];
		vi = XEv.v[i];
		uli = XEv.u[ileft];
		vli = XEv.v[ileft];

		dhdxi = XGrad.dhdx[i];
		dhdxil = XGrad.dhdx[ileft];
		dudxi = XGrad.dudx[i];
		dudxil = XGrad.dudx[ileft];
		dvdxi = XGrad.dvdx[i];
		dvdxil = XGrad.dvdx[ileft];


		hr = hi - dx * dhdxi;
		hl = hn + dx * dhdxil;
		etar = XEv.zs[i] - dx * XGrad.dzsdx[i];
		etal = XEv.zs[ileft] + dx * XGrad.dzsdx[ileft];

		//define the topography term at the interfaces
		zr = zi - dx * dzbdx[i];
		zl = zn + dx * dzbdx[ileft];

		//define the Audusse terms
		zA = max(zr, zl);

		// Now the CN terms
		zCN = min(zA, min(etal, etar));
		hCNr = max(T(0.0), min(etar - zCN, hr));
		hCNl = max(T(0.0), min(etal - zCN, hl));
		
		//Velocity reconstruction
		//To avoid high velocities near dry cells, we reconstruct velocities according to Bouchut.
		T ul, ur, vl, vr;
		if (hi > eps) {
			ur = ui - (1. + dx * dhdxi / hi) * dx * dudxi;
			vr = vi - (1. + dx * dhdxi / hi) * dx * dvdxi;
		}
		else {
			ur = ui - dx * dudxi;
			vr = vi - dx * dvdxi;
		}
		if (hn > eps) {
			ul = uli + (1. - dx * dhdxil / hn) * dx * dudxil;
			vl = vli + (1. - dx * dhdxil / hn) * dx * dvdxil;
		}
		else {
			ul = uli + dx * dudxil;
			vl = vli + dx * dvdxil;
		}

	


		T fh, fu, fv, dt;


		//solver below also modifies fh and fu
		dt = hllc(g, delta, epsi, CFL, cm, fmu, hp, hm, up, um, fh, fu);

		if (dt < dtmax[i])
		{
			dtmax[i] = dt;
		}
		else
		{
			dtmax[i] = T(1.0) / epsi;
		}

		fv = (fh > 0. ? vl : vr) * fh;

	
		// Topographic source term

		// In the case of adaptive refinement, care must be taken to ensure
		// well-balancing at coarse/fine faces (see [notes/balanced.tm]()). 
		if ((ix == blockDim.y) && levRB < lev)//(ix==16) i.e. in the right halo
		{
			int jj = LBRB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + blockDim.y / 2;
			int iright = memloc(halowidth, blkmemwidth, 0, jj, RB);;
			hi = XEv.h[iright];
			zi = zb[iright];
		}
		if ((ix == 0) && levLB < lev)//(ix==16) i.e. in the right halo
		{
			int jj = RBLB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + blockDim.y / 2;
			int ilc = memloc(halowidth, blkmemwidth, blockDim.y - 1, jj, LB);
			hn = XEv.h[ilc];
			zn = zb[ilc];
		}

		sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi) * (zi - zl));
		sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn) * (zn - zr));

		////Flux update

		XFlux.Fhu[i] = fmu * fh;
		XFlux.Fqux[i] = fmu * (fu - sl);
		XFlux.Su[i] = fmu * (fu - sr);
		XFlux.Fqvx[i] = fmu * fv;
	}
	else
	{
		dtmax[i] = T(1.0) / epsi;
		XFlux.Fhu[i] = T(0.0);
		XFlux.Fqux[i] = T(0.0);
		XFlux.Su[i] = T(0.0);
		XFlux.Fqvx[i] = T(0.0);
	}

}



template <class T> __host__ __device__ T hllc(T g, T delta, T epsi, T CFL, T cm, T fm, T hm, T hp, T um, T up, T &fh, T &fq)
{
	T cp, cmo , dt, ustar, cstar, SL, SR, fhm, fum,fhp, fup,dlt;
	cmo = sqrt(g * hm);
	cp = sqrt(g * hp);
	ustar = (um + up) / 2. + cmo - cp;
	cstar = (cmo + cp) / 2. + (um - up) / 4.;
	SL = hm == 0. ? up - 2. * cp : min(um - cmo, ustar - cstar);
	SR = hp == 0. ? um + 2. * cmo : max(up + cp, ustar + cstar);

	if (0. <= SL) {
		fh = um * hm;
		fq = hm * (um * um + g * hm / 2.);
	}
	else if (0. >= SR) {
		fh = up * hp;
		fq = hp * (up * up + g * hp / 2.);
	}
	else {
		fhm = um * hm;
		fum = hm * (um * um + g * hm / 2.);
		fhp = up * hp;
		fup = hp * (up * up + g * hp / 2.);
		fh = (SR * fhm - SL * fhp + SL * SR * (hp - hm)) / (SR - SL);
		fq = (SR * fum - SL * fup + SL * SR * (hp * up - hm * um)) / (SR - SL);
	}

	double a = max(fabs(SL), fabs(SR));
	if (a > epsi) {
		dlt = delta * cm / fm;
		dt = CFL * dlt / a;
		
	}
	else
	{
		dt = T(1.0) / epsi;
	}
	return dt;
}
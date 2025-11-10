#include "Multilayer.h"

//template <class T> void calcAbaro()
//{
//
//	T gmetric = (2. * fm.x[i] / (cm[i] + cm[i - 1]))
//
//	a_baro[i] (G*gmetric*(eta[i-1] - eta[i])/Delta)
//	a_baro(eta, i) (gmetric(i)*(G*(eta[i-1] - eta[i])+sq(60.)*(100./1030.)*(panom[i-1]-panom[i]))/Delta)
//}

template <class T> __global__ void CalcfaceValX(T pdt,Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux, T* dtmax,T* zb,T* Patm)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	

	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps);// +epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	T CFL_H = T(0.5);

	T ybo = XParam.spherical ? T(XParam.yo + XBlock.yo[ib]) : T(1.0);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);

	T zsi = XEv.zs[i];

	T zsn = XEv.zs[ileft];

	T zbi = zb[i];
	T zbn = zb[ileft];


	T fmu = T(1.0);
	T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T cml = cm;
	//T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	//T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy)) : T(1.0);
	//T gmetric = T(1.0);// XParam.spherical ? (fmu / (cm)) : T(1.0);// (2. * fm.x[i] / (cm[i] + cm[i - 1]));

	/*if (XBlock.level[XBlock.LeftBot[ib]] > XBlock.level[ib])
	{
		cml = 0.5;
	}*/

	T gmetric =  (2. * fmu / (cm + cml));

	T ax = 0.0;
	
	if (XParam.atmpforcing)
	{
		ax = g * gmetric * ((zsn - zsi) + XParam.Pa2m * (Patm[ileft] - Patm[i])) / delta;
	}
	else
	{
		ax = (g * gmetric * (zsn - zsi) / delta);
	}
	//T ax = (g * gmetric* ((zsn + XParam.Pa2m * Patm[i-1]) - (zsi + XParam.Pa2m * Patm[i])) / delta);
	//T ax = (g * gmetric* ((eta[i-1] - eta[i])+sq(60.)*(100./1030.)*(panom[i-1]-panom[i]))/Delta)
	T H = T(0.0);
	T um = T(0.0);
	T Hr = T(0.0);
	T Hl = T(0.0);

	
	//foreach_layer() {
	{
		T hi = XEv.h[i];
		T hn = XEv.h[ileft];
		Hr += hi;
		Hl += hn;
		T hl = hn > dry ? hn : T(0.0);
		T hr = hi > dry ? hi : T(0.0);

		
		
		//XFlux.hu[i] = hl > 0. || hr > 0. ? (hl * XEv.u[ileft] + hr * XEvu[i]) / (hl + hr) : 0.;
		T hui = hl > T(0.0) || hr > T(0.0) ? (hl * XEv.u[ileft] + hr * XEv.u[i]) / (hl + hr) : T(0.0);
		
		T hff;

		if (Hl <= dry)
			hff = max(min(zbi + Hr - zbn, hi), T(0.0));
		else if (Hr <= dry)
			hff = max(min(zbn + Hl - zbi, hn), T(0.0));
		else
		{
			T un = pdt * (hui) / delta; //pdt * (hui + pdt * ax) / delta;
			T a =  signof(un);
			int iu = un > T(0.0) ? ileft : i;// -(a + 1.) / 2.;
			//double dhdx = h.gradient ? h.gradient(h[i - 1], h[i], h[i + 1]) / Delta : (h[i + 1] - h[i - 1]) / (2. * Delta);
			
			hff = XEv.h[iu] + a * (T(1.) - a * un) * XGrad.dhdx[iu] * delta / T(2.);
		}
		XFlux.hfu[i] = fmu * hff;

		if (fabs(hui) > um)
			um = fabs(hui);

		XFlux.hu[i] = hui* fmu * hff;
		XFlux.hau[i] = fmu * hff * ax;

		H += hff;
	}

	if (H > dry) {
		T c = um / CFL + sqrt(g*H) / CFL_H;//um / CFL + sqrt(g * (hydrostatic ? H : delta * tanh(H / delta))) / CFL_H;
		if (c > T(0.)) {
			dtmax[i] = min(cm*delta / (c * fmu),dtmax[i]);
			//if (dt < dtmax)
			//	dtmax = dt;
		}
	}
	//pdt = dt = dtnext(dtmax);
}
template __global__ void CalcfaceValX<float>(float pdt, Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxMLP<float> XFlux, float* dtmax, float* zb, float* Patm);
template __global__ void CalcfaceValX<double>(double pdt, Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxMLP<double> XFlux, double* dtmax, double* zb, double* Patm);

template <class T> __global__ void CalcfaceValY(T pdt, Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux, T* dtmax, T* zb,T* Patm)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];


	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps);// +epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);

	T ybo = XParam.spherical ? T(XParam.yo + XBlock.yo[ib]):T(1.0);

	T CFL = T(XParam.CFL);

	T CFL_H = T(0.5);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ibot = memloc(halowidth, blkmemwidth, ix, iy-1, ib);

	T zsi = XEv.zs[i];

	T zsn = XEv.zs[ibot];

	T zbi = zb[i];
	T zbn = zb[ibot];


	T fmu = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy)) : T(1.0);
	T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T cml = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy-1) : T(1.0);
	//T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	//T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy)) : T(1.0);
	T gmetric = XParam.spherical ? (2*fmu / (cml+cm)) : T(1.0);// (2. * fm.x[i] / (cm[i] + cm[i - 1]));

	T ax = 0.0;//(g * gmetric * (zsn - zsi) / delta);

	if (XParam.atmpforcing)
	{
		ax = g * gmetric * ((zsn - zsi) + XParam.Pa2m * (Patm[ibot] - Patm[i])) / delta;
	}
	else
	{
		ax = (g * gmetric * (zsn - zsi) / delta);
	}

	T H = T(0.0);
	T um = T(0.0);
	T Hr = T(0.0);
	T Hl = T(0.0);


	//foreach_layer() {
	{
		T hi = XEv.h[i];
		T hn = XEv.h[ibot];
		Hr += hi;
		Hl += hn;
		T hl = hn > dry ? hn : T(0.0);
		T hr = hi > dry ? hi : T(0.0);



		//XFlux.hu[i] = hl > 0. || hr > 0. ? (hl * XEv.u[ileft] + hr * XEvu[i]) / (hl + hr) : 0.;
		T hvi = hl > T(0.0) || hr > T(0.0) ? (hl * XEv.v[ibot] + hr * XEv.v[i]) / (hl + hr) : T(0.0);

		T hff;

		if (Hl <= dry)
			hff = max(min(zbi + Hr - zbn, hi), T(0.0));
		else if (Hr <= dry)
			hff = max(min(zbn + Hl - zbi, hn), T(0.0));
		else
		{
			T vn = pdt * (hvi) / delta;//pdt * (hvi + pdt * ax) / delta;
			T a = signof(vn);
			int iu = vn > T(0.0) ? ibot : i;// -(a + 1.) / 2.;
			//double dhdx = h.gradient ? h.gradient(h[i - 1], h[i], h[i + 1]) / Delta : (h[i + 1] - h[i - 1]) / (2. * Delta);

			hff = XEv.h[iu] + a * (T(1.0) - a * vn) * XGrad.dhdy[iu] * delta / T(2.0);
		}
		XFlux.hfv[i] = fmu * hff;

		if (fabs(hvi) > um)
			um = fabs(hvi);

		XFlux.hv[i] = hvi* fmu * hff;
		XFlux.hav[i] = fmu * hff * ax;

		H += hff;
	}

	if (H > dry) {
		T c = um / CFL + sqrt(g * H) / CFL_H;//um / CFL + sqrt(g * (hydrostatic ? H : delta * tanh(H / delta))) / CFL_H;
		if (c > T(0.0)) {
			dtmax[i] = min(min(cm,cml)*delta / (c * fmu), dtmax[i]);
			//if (dt < dtmax)
			//	dtmax = dt;
		}
	}
	//pdt = dt = dtnext(dtmax);
}
template __global__ void CalcfaceValY<float>(float pdt, Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxMLP<float> XFlux, float* dtmax, float* zb, float* Patm);
template __global__ void CalcfaceValY<double>(double pdt, Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxMLP<double> XFlux, double* dtmax, double* zb, double* Patm);



template <class T> __global__ void CheckadvecMLX(Param XParam, BlockP<T> XBlock,T dt, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];


	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps);// +epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	T CFL_H = T(0.5);

	T ybo = XParam.spherical ? T(XParam.yo + XBlock.yo[ib]) : T(1.0);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);

	//For each layer
	{
		T hul = XFlux.hu[i];
		T hi = XEv.h[i];
		T hn = XEv.h[ileft];

		T cmn = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);

		T cmi = cmn;// XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);

		if (hul * dt / (delta * cmn) > CFL * hn)
		{
			hul = CFL * hn * delta * cmn / dt;
		}
		else if (-hul * dt / (delta * cmi) > CFL * hi)
		{
			hul = -CFL * hi * delta * cmi / dt;
		}

		if (hul != XFlux.hu[i])
		{
			/*if (l < nl - 1)
			{
				hu.x[0, 0, 1] += hu.x[] - hul;
			}*/
			XFlux.hu[i] = hul;
		}
	}


}
template __global__ void CheckadvecMLX<float>(Param XParam, BlockP<float> XBlock, float dt, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxMLP<float> XFlux);
template __global__ void CheckadvecMLX<double>(Param XParam, BlockP<double> XBlock, double dt, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxMLP<double> XFlux);

template <class T> __global__ void CheckadvecMLY(Param XParam, BlockP<T> XBlock,T dt, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];


	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps);// +epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	T CFL_H = T(0.5);

	T ybo = XParam.spherical ? T(XParam.yo + XBlock.yo[ib]) : T(1.0);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ibot = memloc(halowidth, blkmemwidth, ix, iy-1, ib);

	//For each layer
	{
		T hvl = XFlux.hv[i];
		T hi = XEv.h[i];
		T hn = XEv.h[ibot];

		T cmn = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy-1) : T(1.0);
		T cmi = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);

		if (hvl * dt / (delta * cmn) > CFL * hn)
		{
			hvl = CFL * hn * delta * cmn / dt;
		}
		else if (-hvl * dt / (delta * cmi) > CFL * hi)
		{
			hvl = -CFL * hi * delta * cmi / dt;
		}

		if (hvl != XFlux.hv[i])
		{
			/*if (l < nl - 1)
			{
				hu.x[0, 0, 1] += hu.x[] - hul;
			}*/
			XFlux.hv[i] = hvl;
		}
	}


}
template __global__ void CheckadvecMLY<float>(Param XParam, BlockP<float> XBlock, float dt, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxMLP<float> XFlux);
template __global__ void CheckadvecMLY<double>(Param XParam, BlockP<double> XBlock, double dt, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxMLP<double> XFlux);



template <class T> __global__ void AdvecFluxML(Param XParam, BlockP<T> XBlock,T dt, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];


	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps);// +epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);


	//For each layer
	{
		T un = dt * XFlux.hu[i] / ((XFlux.hfu[i] + dry) * delta);
		//T vn = dt * XFlux.hu[i] / ((XFlux.hfu[i] + dry) * delta);
		T vn = dt * XFlux.hv[i] / ((XFlux.hfv[i] + dry) * delta);
		
		T au = signof(un);
		T av = signof(vn);

		int ixshft = un > T(0.0) ? -1: 0;
		int iyshft = vn > T(0.0) ? -1 : 0;
		//int iu = un >= 0.0 ? ileft : i;//-(a + 1.) / 2.;
		int iu = memloc(halowidth, blkmemwidth, ix + ixshft, iy, ib);

		int iut, iub;

		
		/*
		if (ix == 0 && iy == 15)
		{
			iut = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);
		}
		else
		{
			iut = memloc(halowidth, blkmemwidth, ix + ixshft, iy + 1, ib);
		}
		if (ix == 0 && iy == 0)
		{
			iub = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
		}
		else
		{
			iub = memloc(halowidth, blkmemwidth, ix + ixshft, iy - 1, ib);
		}
		*/
		iub = memloc(halowidth, blkmemwidth, ix + ixshft, iy - 1, ib);
		iut = memloc(halowidth, blkmemwidth, ix + ixshft, iy + 1, ib);


		int iv = memloc(halowidth, blkmemwidth, ix, iy + iyshft, ib);

		int ivr, ivl;
		/*
		if (iy == 0 && ix == 15)
		{
			ivr = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
		}
		else
		{
			ivr = memloc(halowidth, blkmemwidth, ix + 1, iy + iyshft, ib);
		}
		
		if (iy == 0 && ix == 0)
		{
			ivl = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
		}
		else
		{
			ivl = memloc(halowidth, blkmemwidth, ix - 1, iy+iyshft, ib);
		}
		*/
		ivr = memloc(halowidth, blkmemwidth, ix + 1, iy + iyshft, ib);
		ivl = memloc(halowidth, blkmemwidth, ix - 1, iy + iyshft, ib);

		
		T sux2 = XEv.u[iu] + au * (T(1.0) - au * un) * XGrad.dudx[iu] * delta / T(2.0);
		T suy2 = XEv.u[iv] + av * (T(1.0) - av * vn) * XGrad.dudy[iv] * delta / T(2.0);

		T svy2 = XEv.v[iv] + av * (T(1.0) - av * vn) * XGrad.dvdy[iv] * delta / T(2.0);
		T svx2 = XEv.v[iu] + au * (T(1.0) - au * un) * XGrad.dvdx[iu] * delta / T(2.0);
		if (XFlux.hfv[iu] + XFlux.hfv[iut] > dry)
		{
			T vvn = (XFlux.hv[iu] + XFlux.hv[iut]) / (XFlux.hfv[iu] + XFlux.hfv[iut]);
			T syy = XGrad.dudy[iu] * delta;// != 0.0 ? XGrad.dudy[iu] :*/ vvn < 0.0 ? XEv.u[iut] - XEv.u[iu] : XEv.u[iu] - XEv.u[iub];
			T sxx = XGrad.dvdy[iu] * delta;
			sux2 -= dt * vvn * syy / (T(2.) * delta);
			svx2 -= dt * vvn * sxx / (T(2.) * delta);
			
		}
		if (XFlux.hfu[iv] + XFlux.hfu[ivr] > dry)
		{
			T uun = (XFlux.hu[iv] + XFlux.hu[ivr]) / (XFlux.hfu[iv] + XFlux.hfu[ivr]);
			T syy = XGrad.dvdx[iv] * delta;// != 0.0 ? XGrad.dvdx[iv] : uun < 0.0 ? XEv.v[ivr] - XEv.v[iv] : XEv.v[iv] - XEv.v[ivl];
			T sxx = XGrad.dudx[iv] * delta;
			svy2 -= dt * uun * syy / (T(2.) * delta);
			suy2 -= dt * uun * sxx / (T(2.) * delta);
			//svx2 -= dt * vvn * syy / (2. * delta);
			//su2 -= dt * uun * syy / (2. * delta);
		}
	


		XFlux.Fux[i] = sux2 * XFlux.hu[i];
		XFlux.Fuy[i] = suy2 * XFlux.hv[i];// suy2* XFlux.hv[i];// su2*XFlux.hu[i];

		//XFlux.Fvx[i] = svy2 * XFlux.hv[i];// sv2*XFlux.hv[i];
		XFlux.Fvx[i] = svx2 * XFlux.hu[i];;// svx2* XFlux.hu[i];// sv2*XFlux.hv[i];
		XFlux.Fvy[i] = svy2 * XFlux.hv[i];
		// Confirmed equations
		//XFlux.Fux[i] = sux2 * XFlux.hu[i];
		
		//XFlux.Fvy[i] = svy2 * XFlux.hv[i];
		
		

	}
}
template __global__ void AdvecFluxML<float>(Param XParam, BlockP<float> XBlock, float dt, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxMLP<float> XFlux);
template __global__ void AdvecFluxML<double>(Param XParam, BlockP<double> XBlock, double dt, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxMLP<double> XFlux);


template <class T> __global__ void AdvecEv(Param XParam, BlockP<T> XBlock,T dt, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];


	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps);// +epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	T cmu = T(1.0);
	T cmv = T(1.0);

	if (XParam.spherical)
	{
		T ybo = T(XParam.yo + XBlock.yo[ib]);

		cmu = calcCM(T(XParam.Radius), delta, ybo, iy);
		cmv = calcCM(T(XParam.Radius), delta, ybo, iy);

	}


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
	int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
	int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);
	//For each layer
	{
		T uui = XEv.u[i];
		T vvi = XEv.v[i];
		T hi = XEv.h[i];

		

		uui *= hi;
		vvi *= hi;


		//for debugging
		//uui += XFlux.Fux[i];
		//vvi += XFlux.Fvx[i];
		

		//Below is correct
		
		uui += dt * (XFlux.Fux[i] - XFlux.Fux[iright]) / (delta * cmu);
		uui += dt * (XFlux.Fuy[i] - XFlux.Fuy[itop]) / (delta * cmv);

		vvi += dt * (XFlux.Fvx[i] - XFlux.Fvx[iright]) / (delta * cmu);
		vvi += dt * (XFlux.Fvy[i] - XFlux.Fvy[itop]) / (delta * cmv);
		

		

		T h1 = hi;
		h1 += dt * (XFlux.hu[i] - XFlux.hu[iright]) / (delta * cmu);
		h1 += dt * (XFlux.hv[i] - XFlux.hv[itop]) / (delta * cmv);

		XEv.h[i] = max(h1, T(0.0));

		if (h1 < dry)
		{
			uui = T(0.0);
			vvi = T(0.0);
		}
		else
		{
			uui /=  h1;
			vvi /=  h1;
		}
		XEv.u[i] = uui;
		XEv.v[i] = vvi;
	}

}
template __global__ void AdvecEv<float>(Param XParam, BlockP<float> XBlock, float dt, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxMLP<float> XFlux);
template __global__ void AdvecEv<double>(Param XParam, BlockP<double> XBlock, double dt, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxMLP<double> XFlux);



template <class T> __global__ void pressureML(Param XParam, BlockP<T> XBlock,T dt, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];


	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps);// +epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	T ybo = XParam.spherical ? T(XParam.yo + XBlock.yo[ib]) : T(1.0);


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
	int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
	int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

	T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T fmu = T(1.0);
	T fmv =  XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy)) : T(1.0);
	T fmup = T(1.0);
	T fmvp =  XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy+1)) : T(1.0);

	T cmdinv, ga;

	cmdinv = T(1.0) / (cm * delta);
	ga = T(0.5) * g;

	//For each layer
	{

		T uui = XEv.u[i];
		T vvi = XEv.v[i];
		//
		XFlux.hu[i] += dt * XFlux.hau[i];
		XFlux.hv[i] += dt * XFlux.hav[i];
		
		uui += dt * (XFlux.hau[i] + XFlux.hau[iright]) / (XFlux.hfu[i] + XFlux.hfu[iright] + dry);
		vvi += dt * (XFlux.hav[i] + XFlux.hav[itop]) / (XFlux.hfv[i] + XFlux.hfv[itop] + dry);

		T dmdl = (fmup - fmu) * cmdinv;// absurd if not spherical!dmdl = (fm.x[1,0] - fm.x[])/(cm[]*Delta);
		T dmdt = (fmvp - fmv) * cmdinv;// (fm.y[0,1] - fm.y[])/(cm[]*Delta);
		T fG = vvi * dmdl - uui * dmdt;

		uui += dt * fG * vvi;
		vvi -= dt * fG * uui;

		XEv.u[i] = uui;
		XEv.v[i] = vvi;
	}
	


}
template __global__ void pressureML<float>(Param XParam, BlockP<float> XBlock, float dt, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxMLP<float> XFlux);
template __global__ void pressureML<double>(Param XParam, BlockP<double> XBlock, double dt, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxMLP<double> XFlux);



template <class T> __global__ void CleanupML(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv,T* zb)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);


	XEv.zs[i] = zb[i] + max(XEv.h[i], T(0.0));
}
template __global__ void CleanupML<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, float* zb);
template __global__ void CleanupML<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, double* zb);

template <class T> __global__ void Updatewindandriver(Param XParam, BlockP<T> XBlock,T dt, EvolvingP<T> XEv, AdvanceP<T> XAdv)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	T hi = XEv.h[i];

	if (hi > T(XParam.eps)) {

		XEv.u[i] += (XAdv.dhu[i] / hi) * dt;
		XEv.v[i] += (XAdv.dhv[i] / hi) * dt;
	}

	XEv.h[i] = hi + XAdv.dh[i] * dt;

}
template __global__ void Updatewindandriver<float>(Param XParam, BlockP<float> XBlock, float dt, EvolvingP<float> XEv, AdvanceP<float> XAdv);
template __global__ void Updatewindandriver<double>(Param XParam, BlockP<double> XBlock, double dt, EvolvingP<double> XEv, AdvanceP<double> XAdv);
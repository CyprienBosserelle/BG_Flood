#include "Multilayer.h"

//template <class T> void calcAbaro()
//{
//
//	T gmetric = (2. * fm.x[i] / (cm[i] + cm[i - 1]))
//
//	a_baro[i] (G*gmetric*(eta[i-1] - eta[i])/Delta)
//}

template <class T> __global__ void CalcfaceValX(T pdt,Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux, T* dtmax,T* zb)
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
	T eps = T(XParam.eps) + epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	T CFL_H = T(0.5);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);

	T zsi = XEv.zs[i];

	T zsn = XEv.zs[ileft];

	T zbi = zb[i];
	T zbn = zb[ileft];


	T fmu = T(1.0);
	T cm = T(1.0);//T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T gmetric = T(1.0);// (2. * fm.x[i] / (cm[i] + cm[i - 1]));

	T ax = (G * gmetric * (zsn - zsi) / delta);

	T H = 0.;
	T um = 0.;
	T Hr = 0.;
	T Hl = 0.;

	
	//foreach_layer() {
	{
		T hi = XEv.h[i];
		T hn = XEv.h[ileft];
		Hr += hi;
		Hl += hn;
		T hl = hn > dry ? hn : 0.;
		T hr = hi > dry ? hi : 0.;

		
		
		//XFlux.hu[i] = hl > 0. || hr > 0. ? (hl * XEv.u[ileft] + hr * XEvu[i]) / (hl + hr) : 0.;
		T hui = hl > 0. || hr > 0. ? (hl * XEv.u[ileft] + hr * XEv.u[i]) / (hl + hr) : 0.;
		
		T hff;

		if (Hl <= dry)
			hff = fmax(fmin(zbi + Hr - zbi, hi), 0.);
		else if (Hr <= dry)
			hff = fmax(fmin(zbn + Hl - zbi, hn), 0.);
		else
		{
			T un = pdt * (hui + pdt * ax) / delta;
			auto a = sign(un);
			int iu = un >= 0.0 ? ileft : i;// -(a + 1.) / 2.;
			//double dhdx = h.gradient ? h.gradient(h[i - 1], h[i], h[i + 1]) / Delta : (h[i + 1] - h[i - 1]) / (2. * Delta);
			
			hff = h[iu] + a * (1. - a * un) * dhdx[iu] * delta / 2.;
		}
		XFlux.hfu[i] = fmu * hff;

		if (fabs(hui) > um)
			um = fabs(hui);

		XFlux.hu[i] *= XFlux.hfu[i];
		XFlux.hau[i] = XFlux.hfu[i] * ax;

		H += hff;
	}

	if (H > dry) {
		T c = um / CFL + sqrt(g*H) / CFL_H;//um / CFL + sqrt(g * (hydrostatic ? H : delta * tanh(H / delta))) / CFL_H;
		if (c > 0.) {
			double dtmax[i] = min(delta / (c * fmu),dtmax[i]);
			//if (dt < dtmax)
			//	dtmax = dt;
		}
	}
	//pdt = dt = dtnext(dtmax);
}

template <class T> __global__ void CalcfaceValY(T pdt, Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux, T* dtmax, T* zb)
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
	T eps = T(XParam.eps) + epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	T CFL_H = T(0.5);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ibot = memloc(halowidth, blkmemwidth, ix, iy-1, ib);

	T zsi = XEv.zs[i];

	T zsn = XEv.zs[ibot];

	T zbi = zb[i];
	T zbn = zb[ibot];


	T fmu = T(1.0);
	T cm = T(1.0);//T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T gmetric = T(1.0);// (2. * fm.x[i] / (cm[i] + cm[i - 1]));

	T ax = (G * gmetric * (zsn - zsi) / delta);

	T H = 0.;
	T um = 0.;
	T Hr = 0.;
	T Hl = 0.;


	//foreach_layer() {
	{
		T hi = XEv.h[i];
		T hn = XEv.h[ibot];
		Hr += hi;
		Hl += hn;
		T hl = hn > dry ? hn : 0.;
		T hr = hi > dry ? hi : 0.;



		//XFlux.hu[i] = hl > 0. || hr > 0. ? (hl * XEv.u[ileft] + hr * XEvu[i]) / (hl + hr) : 0.;
		T hvi = hl > 0. || hr > 0. ? (hl * XEv.v[ibot] + hr * XEv.v[i]) / (hl + hr) : 0.;

		T hff;

		if (Hl <= dry)
			hff = fmax(fmin(zbi + Hr - zbi, hi), 0.);
		else if (Hr <= dry)
			hff = fmax(fmin(zbn + Hl - zbi, hn), 0.);
		else
		{
			T vn = pdt * (hvi + pdt * ax) / delta;
			auto a = sign(vn);
			int iu = vn >= 0.0 ? ibot : i;// -(a + 1.) / 2.;
			//double dhdx = h.gradient ? h.gradient(h[i - 1], h[i], h[i + 1]) / Delta : (h[i + 1] - h[i - 1]) / (2. * Delta);

			hff = h[iu] + a * (1. - a * vn) * dhdy[iu] * delta / 2.;
		}
		XFlux.hfv[i] = fmu * hff;

		if (fabs(hvi) > um)
			um = fabs(hvi);

		XFlux.hv[i] *= XFlux.hfv[i];
		XFlux.hav[i] = XFlux.hfv[i] * ax;

		H += hff;
	}

	if (H > dry) {
		T c = um / CFL + sqrt(g * H) / CFL_H;//um / CFL + sqrt(g * (hydrostatic ? H : delta * tanh(H / delta))) / CFL_H;
		if (c > 0.) {
			double dtmax[i] = min(delta / (c * fmu), dtmax[i]);
			//if (dt < dtmax)
			//	dtmax = dt;
		}
	}
	//pdt = dt = dtnext(dtmax);
}

template <class T> __global__ void CheckadvecMLX(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux)
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
	T eps = T(XParam.eps) + epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	T CFL_H = T(0.5);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);

	//For each layer
	{
		T hul = XFlux.hu[i];
		T hi = XEv.h[i];
		T hn = XEv.h[ileft];

		T cmn = T(1.0);//cm[-1]
		T cmi = T(1.0);//cm[]

		if (hul * dt / (Delta * cmn) > CFL * hn)
		{
			hul = CFL * hn * Delta * cmn / dt;
		}
		else if (-hul * dt / (Delta * cm) > CFL * hi)
		{
			hul = -CFL * hn * Delta * cm / dt;
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
template <class T> __global__ void CheckadvecMLY(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux)
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
	T eps = T(XParam.eps) + epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	T CFL_H = T(0.5);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ibot = memloc(halowidth, blkmemwidth, ix, iy-1, ib);

	//For each layer
	{
		T hvl = XFlux.hv[i];
		T hi = XEv.h[i];
		T hn = XEv.h[ibot];

		T cmn = T(1.0);//cm[-1]
		T cmi = T(1.0);//cm[]

		if (hvl * dt / (delta * cmn) > CFL * hn)
		{
			hvl = CFL * hn * delta * cmn / dt;
		}
		else if (-hvl * dt / (delta * cm) > CFL * hi)
		{
			hvl = -CFL * hn * delta * cm / dt;
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


template <class T> __global__ void AdvecFluxML(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux)
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
	T eps = T(XParam.eps) + epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);


	//For each layer
	{
		T un = dt * XFlux.hu[i] / ((XFlux.hfu[i] + dry) * delta);
		T vn = dt * XFlux.hv[i] / ((XFlux.hfv[i] + dry) * delta);
		auto au = sign(un);
		auto av = sign(vn);

		int ixshft = un >= 0.0 ? -1, 0;
		int iyshft = vn >= 0.0 ? -1, 0;
		//int iu = un >= 0.0 ? ileft : i;//-(a + 1.) / 2.;
		int iu = memloc(halowidth, blkmemwidth, ix + ixshft, iy, ib);
		int iut = memloc(halowidth, blkmemwidth, ix + ixshft, iy + 1, ib);
		int iub = memloc(halowidth, blkmemwidth, ix + ixshft, iy - 1, ib);

		int iv = memloc(halowidth, blkmemwidth, ix, iy + iyshft, ib);
		int ivr = memloc(halowidth, blkmemwidth, ix +1, iy + iyshft, ib);
		int iul = memloc(halowidth, blkmemwidth, ix -1, iy + iyshft, ib);

		T su2 = XEv.u[iu] + au * (1. - au * un) * dudx[iu] * delta / 2.0;
		T sv2 = XEv.v[iv] + av * (1. - av * vn) * dvdy[iv] * delta / 2.0;
		if (XFlux.hfv[iu] + XFlux.hfv[iut] > dry)
		{
			T vvn = (XFlux.hv[iu] + XFlux.hv[iut]) / (XFlux.hfv[iu] + XFlux.hfv[iut]);
			T syy = dudy[iu] != 0.0 ? dudy[iu] : vn < 0.0 ? XEv.u[iut] - XEv.u[iu] : XEv.u[iu] - XEv.u[iub];
			su2 -= dt * vvn * syy / (2. * delta);
		}
		if (XFlux.hfu[iv] + XFlux.hfv[ivr] > dry)
		{
			T uun = (XFlux.hv[iv] + XFlux.hv[ivr]) / (XFlux.hfv[iv] + XFlux.hfv[ivr]);
			T syy = dvdx[iv] != 0.0 ? dvdx[iv] : uun < 0.0 ? XEv.v[ivr] - XEv.v[iv] : XEv.v[iv] - XEv.v[ivl];
			sv2 -= dt * uun * syy / (2. * delta);
		}

		XFlux.Fu[i] = su2 * XFlux.hu[i];
		XFlux.Fv[i] = sv2 * XFlux.hv[i];

	}
}

template <class T> __global__ void AdvecEv(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux)
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
	T eps = T(XParam.eps) + epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
	int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
	int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);
	//For each layer
	{
		T uui = XEv.u[i];
		T vvi = XEv.v[i];
		T hi = XEv.h[i];

		uui *= hhi;
		vvi *= hhi;

		uui += dt * (Flux.Fu[i] - Flux.Fu[iright]) / (delta * cmu);
		vvi += dt * (Flux.Fv[i] - Flux.Fv[itop]) / (delta * cmv);

		T h1 = hhi;
		h1 += dt * (XFlux.hu[i] - XFlux.hu[iright) / (delta * cmu);
		h1 += dt * (XFlux.hv[i] - XFlux.hv[itop) / (delta * cmv);

		XEv.h[i] = max(h1, T(0.0));

		if (h1 < dry)
		{
			uui = T(0.0);
			vvi = T(0.0);
		}
		else
		{
			uui /= h1;
			vvi /= h1;
		}
		XEv.u[i] = uui;
		XEv.v[i] = vvi;
	}

}

template <class T> __global__ void pressureML(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxMLP<T> XFlux)
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
	T eps = T(XParam.eps) + epsi;
	T dry = eps;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
	int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
	int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

	T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T fmu = T(1.0);
	T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, ydwn) : T(1.0);
	T fmup = T(1.0);
	T fmvp = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, yup) : T(1.0);

	T cmdinv, ga;

	cmdinv = T(1.0) / (cm * delta);
	ga = T(0.5) * g;

	//For each layer
	{

		T uui = XEv.u[i];
		T vvi = XEv.v[i]
		//
		XFlux.dh[i] += dt * (XFlux.hau[i]+XFlux.hav[i]);
		
		uui += dt * (XFlux.hau[i] + XFlux.hau[iright])/(XFlux.hfu[i]+XFlux.hfu[iright]+dry);
		vvi += dt * (XFlux.hav[i] + XFlux.hav[itop]) / (XFlux.hfv[i] + XFlux.hfv[itop] + dry);

		T dmdl = (fmup - fmu) * cmdinv;// absurd if not spherical!
		T dmdt = (fmvp - fmv) * cmdinv;
		T fG = vvi * dmdl - uui * dmdt;

		uui += dt * fG * vvi;
		vvi -= dt * fG * uui;

		XEv.u[i] = uui;
		XEv.v[i] = vvi;
	}
	


}


template <class T> __global__ void CleanupML()
{

}
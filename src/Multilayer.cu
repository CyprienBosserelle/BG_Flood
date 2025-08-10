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
	T eps = T(XParam.eps);// +epsi;
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

	T ax = (g * gmetric * (zsn - zsi) / delta);

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
			hff = max(min(zbi + Hr - zbn, hi), T(0.0));
		else if (Hr <= dry)
			hff = max(min(zbn + Hl - zbi, hn), T(0.0));
		else
		{
			T un = pdt * (hui + pdt * ax) / delta;
			T a =  signof(un);
			int iu = un > 0.0 ? ileft : i;// -(a + 1.) / 2.;
			//double dhdx = h.gradient ? h.gradient(h[i - 1], h[i], h[i + 1]) / Delta : (h[i + 1] - h[i - 1]) / (2. * Delta);
			
			hff = XEv.h[iu] + a * (1. - a * un) * XGrad.dhdx[iu] * delta / 2.;
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
		if (c > 0.) {
			dtmax[i] = min(delta / (c * fmu),dtmax[i]);
			//if (dt < dtmax)
			//	dtmax = dt;
		}
	}
	//pdt = dt = dtnext(dtmax);
}
template __global__ void CalcfaceValX<float>(float pdt, Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxMLP<float> XFlux, float* dtmax, float* zb);
template __global__ void CalcfaceValX<double>(double pdt, Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxMLP<double> XFlux, double* dtmax, double* zb);

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
	T eps = T(XParam.eps);// +epsi;
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

	T ax = (g * gmetric * (zsn - zsi) / delta);

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
			hff = max(min(zbi + Hr - zbn, hi), 0.);
		else if (Hr <= dry)
			hff = max(min(zbn + Hl - zbi, hn), 0.);
		else
		{
			T vn = pdt * (hvi + pdt * ax) / delta;
			T a = signof(vn);
			int iu = vn > 0.0 ? ibot : i;// -(a + 1.) / 2.;
			//double dhdx = h.gradient ? h.gradient(h[i - 1], h[i], h[i + 1]) / Delta : (h[i + 1] - h[i - 1]) / (2. * Delta);

			hff = XEv.h[iu] + a * (1. - a * vn) * XGrad.dhdy[iu] * delta / 2.;
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
		if (c > 0.) {
			dtmax[i] = min(delta / (c * fmu), dtmax[i]);
			//if (dt < dtmax)
			//	dtmax = dt;
		}
	}
	//pdt = dt = dtnext(dtmax);
}
template __global__ void CalcfaceValY<float>(float pdt, Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxMLP<float> XFlux, float* dtmax, float* zb);
template __global__ void CalcfaceValY<double>(double pdt, Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxMLP<double> XFlux, double* dtmax, double* zb);



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

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);

	//For each layer
	{
		T hul = XFlux.hu[i];
		T hi = XEv.h[i];
		T hn = XEv.h[ileft];

		T cmn = T(1.0);//cm[-1]
		T cmi = T(1.0);//cm[]

		if (hul * dt / (delta * cmn) > CFL * hn)
		{
			hul = CFL * hn * delta * cmn / dt;
		}
		else if (-hul * dt / (delta * cmi) > CFL * hi)
		{
			hul = -CFL * hn * delta * cmi / dt;
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
		else if (-hvl * dt / (delta * cmi) > CFL * hi)
		{
			hvl = -CFL * hn * delta * cmi / dt;
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
		T vn = dt * XFlux.hv[i] / ((XFlux.hfv[i] + dry) * delta);
		T au = signof(un);
		T av = signof(vn);

		int ixshft = un > 0.0 ? -1: 0;
		int iyshft = vn > 0.0 ? -1: 0;
		//int iu = un >= 0.0 ? ileft : i;//-(a + 1.) / 2.;
		int iu = memloc(halowidth, blkmemwidth, ix + ixshft, iy, ib);

		int iut, iub;
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

		int iv = memloc(halowidth, blkmemwidth, ix, iy + iyshft, ib);

		int ivr, ivl;

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

		T su2 = XEv.u[iu] + au * (1. - au * un) * XGrad.dudx[iu] * delta / 2.0;
		T sv2 = XEv.v[iv] + av * (1. - av * vn) * XGrad.dvdy[iv] * delta / 2.0;
		if (XFlux.hfv[iu] + XFlux.hfv[iut] > dry)
		{
			T vvn = (XFlux.hv[iu] + XFlux.hv[iut]) / (XFlux.hfv[iu] + XFlux.hfv[iut]);
			T syy = XGrad.dudy[iu] != 0.0 ? XGrad.dudy[iu] : vvn < 0.0 ? XEv.u[iut] - XEv.u[iu] : XEv.u[iu] - XEv.u[iub];
			su2 -= dt * vvn * syy / (2. * delta);
		}
		if (XFlux.hfu[iv] + XFlux.hfu[ivr] > dry)
		{
			T uun = (XFlux.hu[iv] + XFlux.hu[ivr]) / (XFlux.hfu[iv] + XFlux.hfu[ivr]);
			T syy = XGrad.dvdx[iv] != 0.0 ? XGrad.dvdx[iv] : uun < 0.0 ? XEv.v[ivr] - XEv.v[iv] : XEv.v[iv] - XEv.v[ivl];
			sv2 -= dt * uun * syy / (2. * delta);
		}

		XFlux.Fu[i] = su2 * XFlux.hu[i];
		XFlux.Fv[i] = sv2 * XFlux.hv[i];

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

		T cmu = T(1.0);
		T cmv = T(1.0);

		uui += dt * (XFlux.Fu[i] - XFlux.Fu[iright]) / (delta * cmu);
		vvi += dt * (XFlux.Fv[i] - XFlux.Fv[itop]) / (delta * cmv);

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
			uui /= h1;
			vvi /= h1;
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


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
	int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
	int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

	T cm = T(1.0);// XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T fmu = T(1.0);
	T fmv = T(1.0);// XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, ydwn) : T(1.0);
	T fmup = T(1.0);
	T fmvp = T(1.0);// XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, yup) : T(1.0);

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

		T dmdl = (fmup - fmu) * cmdinv;// absurd if not spherical!
		T dmdt = (fmvp - fmv) * cmdinv;
		T fG = vvi * dmdl - uui * dmdt;

		uui += dt * fG * vvi;
		vvi -= dt * fG * uui;

		XEv.u[i] = uui;
		XEv.v[i] = vvi;
	}
	


}
template __global__ void pressureML<float>(Param XParam, BlockP<float> XBlock, float dt, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxMLP<float> XFlux);
template __global__ void pressureML<double>(Param XParam, BlockP<double> XBlock, double dt, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxMLP<double> XFlux);



template <class T> __global__ void CleanupML()
{

}
#include "Kurganov.h"


template <class T> __global__ void updateKurgXGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T*zb)
{
	
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];
	int RB, LBRB, LB, RBLB, levRB, levLB;
	RB = XBlock.RightBot[ib];
	levRB = XBlock.level[RB];
	LBRB = XBlock.LeftBot[RB];

	LB = XBlock.LeftBot[ib];
	levLB = XBlock.level[LB];
	RBLB = XBlock.RightBot[LB];

	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps)+epsi;
	T delta = calcres(T(XParam.dx), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);
	// This is based on kurganov and Petrova 2007


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix-1, iy, ib);

	

	
	T dhdxi = XGrad.dhdx[i];
	T dhdxmin = XGrad.dhdx[ileft];
	T cm = T(1.0);
	T fmu = T(1.0);

	T hi = XEv.h[i];

	T hn = XEv.h[ileft];
	

	if (hi > eps || hn > eps)
	{
		T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, sl, sr,ga;

		// along X
		dx = delta * T(0.5);
		zi = XEv.zs[i] - hi;

		//printf("%f\n", zi);


		//zl = zi - dx*(dzsdx[i] - dhdx[i]);
		zl = zi - dx * (XGrad.dzsdx[i] - dhdxi);
		//printf("%f\n", zl);

		zn = XEv.zs[ileft] - hn;

		//printf("%f\n", zn);
		zr = zn + dx * (XGrad.dzsdx[ileft] - dhdxmin);


		zlr = max(zl, zr);

		//hl = hi - dx*dhdx[i];
		hl = hi - dx * dhdxi;
		up = XEv.u[i] - dx * XGrad.dudx[i];
		hp = max(T(0.0), hl + zl - zlr);

		hr = hn + dx * dhdxmin;
		um = XEv.u[ileft] + dx * XGrad.dudx[ileft];
		hm = max(T(0.0), hr + zr - zlr);

		ga = g * T(0.5);

		T fh, fu, fv, dt;

		
		//solver below also modifies fh and fu
		dt = KurgSolver(g, delta, epsi, CFL, cm, fmu, hp, hm, up, um, fh, fu);

		if (dt < dtmax[i])
		{
			dtmax[i] = dt;
		}
		else
		{
			dtmax[i] = T(1.0) / epsi;
		}
		


		if (fh > T(0.0))
		{
			fv = (XEv.v[ileft] + dx * XGrad.dvdx[ileft]) * fh;// Eq 3.7 third term? (X direction)
		}
		else
		{
			fv = (XEv.v[i] - dx * XGrad.dvdx[i]) * fh;
		}
		//fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
		//dtmax needs to be stored in an array and reduced at the end
		//dtmax = dtmaxf;
		//dtmaxtmp = min(dtmax, dtmaxtmp);
		/*if (ix == 11 && iy == 0)
		{
			printf("a=%f\t b=%f\t c=%f\t d=%f\n", ap*(qm*um + ga*hm2), -am*(qp*up + ga*hp2),( ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm) ) *ad/100.0f, ad);
		}
		*/
		/*
		#### Topographic source term
		
		In the case of adaptive refinement, care must be taken to ensure
		well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
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
template __global__ void updateKurgXGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float* zb);
template __global__ void updateKurgXGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double *zb);


template <class T> __global__ void AddSlopeSourceXGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T * zb)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];

	// neighbours for source term
	int RB, LBRB, LB, RBLB, levRB, levLB;
	RB = XBlock.RightBot[ib];
	levRB = XBlock.level[RB];
	LBRB = XBlock.LeftBot[RB];

	LB = XBlock.LeftBot[ib];
	levLB = XBlock.level[LB];
	RBLB = XBlock.RightBot[LB];



	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps) + epsi;
	T delta = calcres(T(XParam.dx), lev);
	T g = T(XParam.g);

	T ga = T(0.5) * g;

	

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);


	T dhdxi = XGrad.dhdx[i];
	T dhdxmin = XGrad.dhdx[ileft];
	T cm = T(1.0);
	T fmu = T(1.0);

	T dx, zi, zl, zn, zr, zlr, hl, hp, hr, hm;

	T hi = XEv.h[i];

	T hn = XEv.h[ileft];

	if (hi > eps || hn > eps)
	{

		// along X these are same as in Kurgannov
		dx = delta * T(0.5);
		zi = XEv.zs[i] - hi;

		zl = zi - dx * (XGrad.dzsdx[i] - dhdxi);

		zn = XEv.zs[ileft] - hn;

		zr = zn + dx * (XGrad.dzsdx[ileft] - dhdxmin);

		zlr = max(zl, zr);

		hl = hi - dx * dhdxi;
		hp = max(T(0.0), hl + zl - zlr);

		hr = hn + dx * dhdxmin;
		hm = max(T(0.0), hr + zr - zlr);


		//#### Topographic source term
		//In the case of adaptive refinement, care must be taken to ensure
		//	well - balancing at coarse / fine faces(see[notes / balanced.tm]()). * /

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

		T sl, sr;
		sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi) * (zi - zl));
		sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn) * (zn - zr));


		XFlux.Fqux[i] = XFlux.Fqux[i] - fmu * sl;
		XFlux.Su[i] = XFlux.Su[i] - fmu * sr;
	}
}
template __global__ void AddSlopeSourceXGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* zb);
template __global__ void AddSlopeSourceXGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* zb);

template <class T> __host__ void updateKurgXCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T*zb)
{

	
	T delta;
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);
	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps)+epsi;

	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	int RB, LBRB, LB, RBLB, levRB, levLB;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		int lev = XBlock.level[ib];
		delta = calcres(T(XParam.dx), lev);

		// neighbours for source term
		
		RB = XBlock.RightBot[ib];
		levRB = XBlock.level[RB];
		LBRB = XBlock.LeftBot[RB];

		LB = XBlock.LeftBot[ib];
		levLB = XBlock.level[LB];
		RBLB = XBlock.RightBot[LB];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < (XParam.blkwidth + XParam.halowidth); ix++)
			{




				// This is based on kurganov and Petrova 2007


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
					T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm,ga;

					// along X
					dx = delta * T(0.5);
					zi = XEv.zs[i] - hi;

					//printf("%f\n", zi);


					//zl = zi - dx*(dzsdx[i] - dhdx[i]);
					zl = zi - dx * (XGrad.dzsdx[i] - dhdxi);
					//printf("%f\n", zl);

					zn = XEv.zs[ileft] - hn;

					//printf("%f\n", zn);
					zr = zn + dx * (XGrad.dzsdx[ileft] - dhdxmin);


					zlr = max(zl, zr);

					//hl = hi - dx*dhdx[i];
					hl = hi - dx * dhdxi;
					up = XEv.u[i] - dx * XGrad.dudx[i];
					hp = max(T(0.0), hl + zl - zlr);

					hr = hn + dx * dhdxmin;
					um = XEv.u[ileft] + dx * XGrad.dudx[ileft];
					hm = max(T(0.0), hr + zr - zlr);

					ga = g * T(0.5);
					///// Reimann solver
					T fh, fu, fv, sl, sr, dt;

					//solver below also modifies fh and fu
					dt = KurgSolver(g, delta, epsi, CFL, cm, fmu, hp, hm, up, um, fh, fu);

					if (dt < dtmax[i])
					{
						dtmax[i] = dt;
					}
					else
					{
						dtmax[i] = T(1.0) / epsi;
					}



					if (fh > T(0.0))
					{
						fv = (XEv.v[ileft] + dx * XGrad.dvdx[ileft]) * fh;// Eq 3.7 third term? (X direction)
					}
					else
					{
						fv = (XEv.v[i] - dx * XGrad.dvdx[i]) * fh;
					}
					//fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
					//dtmax needs to be stored in an array and reduced at the end
					//dtmax = dtmaxf;
					//dtmaxtmp = min(dtmax, dtmaxtmp);
					/*if (ix == 11 && iy == 0)
					{
						printf("a=%f\t b=%f\t c=%f\t d=%f\n", ap*(qm*um + ga*hm2), -am*(qp*up + ga*hp2),( ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm) ) *ad/100.0f, ad);
					}
					*/
					/*
					#### Topographic source term

					In the case of adaptive refinement, care must be taken to ensure
					well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */

					if ((ix == XParam.blkwidth) && levRB < lev)//(ix==16) i.e. in the right halo
					{
						int jj = LBRB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + XParam.blkwidth / 2;
						int iright = memloc(halowidth, blkmemwidth, 0, jj, RB);;
						hi = XEv.h[iright];
						zi = zb[iright];
					}
					if ((ix == 0) && levLB < lev)//(ix==16) i.e. in the right halo if you 
					{
						int jj = RBLB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + XParam.blkwidth / 2;
						int ilc = memloc(halowidth, blkmemwidth, XParam.blkwidth - 1, jj, LB);
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
		}
	}


}
template __host__ void updateKurgXCPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float *zb);
template __host__ void updateKurgXCPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double *zb);


template <class T> __host__ void AddSlopeSourceXCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* zb)
{
	T delta;
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);
	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps) + epsi;

	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		int lev = XBlock.level[ib];
		delta = calcres(XParam.dx, lev);

		// neighbours for source term
		int RB, LBRB, LB, RBLB, levRB, levLB;
		RB = XBlock.RightBot[ib];
		levRB = XBlock.level[RB];
		LBRB = XBlock.LeftBot[RB];

		LB = XBlock.LeftBot[ib];
		levLB = XBlock.level[LB];
		RBLB = XBlock.RightBot[LB];



		T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
		T eps = T(XParam.eps) + epsi;

		T g = T(XParam.g);
		T ga = T(0.5) * g;

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < (XParam.blkwidth + XParam.halowidth); ix++)
			{

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);


				T dhdxi = XGrad.dhdx[i];
				T dhdxmin = XGrad.dhdx[ileft];
				T cm = T(1.0);
				T fmu = T(1.0);

				T dx, zi, zl, zn, zr, zlr, hl, hp, hr, hm;

				T hi = XEv.h[i];

				T hn = XEv.h[ileft];

				if (hi > eps || hn > eps)
				{

					// along X these are same as in Kurgannov
					dx = delta * T(0.5);
					zi = XEv.zs[i] - hi;

					zl = zi - dx * (XGrad.dzsdx[i] - dhdxi);

					zn = XEv.zs[ileft] - hn;

					zr = zn + dx * (XGrad.dzsdx[ileft] - dhdxmin);

					zlr = max(zl, zr);

					hl = hi - dx * dhdxi;
					hp = max(T(0.0), hl + zl - zlr);

					hr = hn + dx * dhdxmin;
					hm = max(T(0.0), hr + zr - zlr);


					//#### Topographic source term
					//In the case of adaptive refinement, care must be taken to ensure
					//	well - balancing at coarse / fine faces(see[notes / balanced.tm]()). * /

					if ((ix == XParam.blkwidth) && levRB < lev)//(ix==16) i.e. in the right halo
					{
						int jj = LBRB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + XParam.blkwidth / 2;
						int iright = memloc(halowidth, blkmemwidth, 0, jj, RB);;
						hi = XEv.h[iright];
						zi = zb[iright];
					}
					if ((ix == 0) && levLB < lev)//(ix==16) i.e. in the right halo if you 
					{
						int jj = RBLB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + XParam.blkwidth / 2;
						int ilc = memloc(halowidth, blkmemwidth, XParam.blkwidth - 1, jj, LB);
						hn = XEv.h[ilc];
						zn = zb[ilc];
					}

					T sl, sr;
					sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi) * (zi - zl));
					sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn) * (zn - zr));


					XFlux.Fqux[i] = XFlux.Fqux[i] - fmu * sl;
					XFlux.Su[i] = XFlux.Su[i] - fmu * sr;
				}
			}
		}
	}
}
template __host__ void AddSlopeSourceXCPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* zb);
template __host__ void AddSlopeSourceXCPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* zb);




template <class T> __global__ void updateKurgYGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];

	int TL, BLTL, BL, TLBL, levTL, levBL;
	TL = XBlock.TopLeft[ib];
	levTL = XBlock.level[TL];
	BLTL = XBlock.BotLeft[TL];

	BL = XBlock.BotLeft[ib];
	levBL = XBlock.level[BL];
	TLBL = XBlock.TopLeft[BL];

	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps)+epsi;
	T delta = calcres(T(XParam.dx), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);
	

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ibot = memloc(halowidth, blkmemwidth, ix , iy-1, ib);

	T cm = T(1.0);
	T fmv = T(1.0);
		
	T dhdyi = XGrad.dhdy[i];
	T dhdymin = XGrad.dhdy[ibot];
	T hi = XEv.h[i];
	T hn = XEv.h[ibot];
	T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm,ga;



	if (hi > eps || hn > eps)
	{
		hn = XEv.h[ibot];
		dx = delta * T(0.5);
		zi = XEv.zs[i] - hi;
		zl = zi - dx * (XGrad.dzsdy[i] - dhdyi);
		zn = XEv.zs[ibot] - hn;
		zr = zn + dx * (XGrad.dzsdy[ibot] - dhdymin);
		zlr = max(zl, zr);

		hl = hi - dx * dhdyi;
		up = XEv.v[i] - dx * XGrad.dvdy[i];
		hp = max(T(0.0), hl + zl - zlr);

		hr = hn + dx * dhdymin;
		um = XEv.v[ibot] + dx * XGrad.dvdy[ibot];
		hm = max(T(0.0), hr + zr - zlr);


		ga = g * T(0.5);

		//// Reimann solver
		T fh, fu, fv, sl, sr, dt;

		//solver below also modifies fh and fu
		dt = KurgSolver(g, delta, epsi, CFL, cm, fmv, hp, hm, up, um, fh, fu);

		if (dt < dtmax[i])
		{
			dtmax[i] = dt;
		}
		else
		{
			dtmax[i] = T(1.0) / epsi;
		}

		
		if (fh > T(0.0))
		{
			fv = (XEv.u[ibot] + dx * XGrad.dudy[ibot]) * fh;
		}
		else
		{
			fv = (XEv.u[i] - dx * XGrad.dudy[i]) * fh;
		}
		//fv = (fh > 0.f ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
		/**
		#### Topographic source term

		In the case of adaptive refinement, care must be taken to ensure
		well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
		//sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi) * (zi - zl));
		//sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn) * (zn - zr));

		//#### Topographic source term
		//In the case of adaptive refinement, care must be taken to ensure
		//	well - balancing at coarse / fine faces(see[notes / balanced.tm]()). * /

		if ((iy == blockDim.x) && levTL < lev)//(ix==16) i.e. in the right halo
		{
			int jj = BLTL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + blockDim.x / 2;
			int itop = memloc(halowidth, blkmemwidth, jj, 0, TL);;
			hi = XEv.h[itop];
			zi = zb[itop];
		}
		if ((iy == 0) && levBL < lev)//(ix==16) i.e. in the right halo
		{
			int jj = TLBL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + blockDim.x / 2;
			int ibc = memloc(halowidth, blkmemwidth, jj, blockDim.x - 1, BL);
			hn = XEv.h[ibc];
			zn = zb[ibc];
		}

		sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi) * (zi - zl));
		sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn) * (zn - zr));

		////Flux update

		XFlux.Fhv[i] = fmv * fh;
		XFlux.Fqvy[i] = fmv * (fu - sl);
		XFlux.Sv[i] = fmv * (fu - sr);
		XFlux.Fquy[i] = fmv * fv;
	}
	else
	{
		dtmax[i] = T(1.0) / epsi;
		XFlux.Fhv[i] = T(0.0);
		XFlux.Fqvy[i] = T(0.0);
		XFlux.Sv[i] = T(0.0);
		XFlux.Fquy[i] = T(0.0);
	}

}
template __global__ void updateKurgYGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float* zb);
template __global__ void updateKurgYGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double *zb);


template <class T> __global__ void AddSlopeSourceYGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* zb)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];

	// neighbours for source term
	int TL, BLTL, BL, TLBL, levTL, levBL;
	TL = XBlock.TopLeft[ib];
	levTL = XBlock.level[TL];
	BLTL = XBlock.BotLeft[TL];

	BL = XBlock.BotLeft[ib];
	levBL = XBlock.level[BL];
	TLBL = XBlock.TopLeft[BL];



	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps) + epsi;
	T delta = calcres(T(XParam.dx), lev);
	T g = T(XParam.g);
	T ga = T(0.5) * g;


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);


	
	T cm = T(1.0);
	T fmv = T(1.0);

	T dx, zi, zl, zn, zr, zlr, hl, hp, hr, hm;

	T dhdyi = XGrad.dhdy[i];
	T dhdymin = XGrad.dhdy[ibot];
	T hi = XEv.h[i];
	T hn = XEv.h[ibot];


	if (hi > eps || hn > eps)
	{

		// along X these are same as in Kurgannov
		dx = delta * T(0.5);
		zi = XEv.zs[i] - hi;

		zl = zi - dx * (XGrad.dzsdy[i] - dhdyi);
		zn = XEv.zs[ibot] - hn;
		zr = zn + dx * (XGrad.dzsdy[ibot] - dhdymin);
		zlr = max(zl, zr);

		hl = hi - dx * dhdyi;
		hp = max(T(0.0), hl + zl - zlr);

		hr = hn + dx * dhdymin;
		hm = max(T(0.0), hr + zr - zlr);


		//#### Topographic source term
		//In the case of adaptive refinement, care must be taken to ensure
		//	well - balancing at coarse / fine faces(see[notes / balanced.tm]()). * /

		if ((iy == blockDim.x) && levTL < lev)//(ix==16) i.e. in the right halo
		{
			int jj = BLTL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + blockDim.x / 2;
			int itop = memloc(halowidth, blkmemwidth, jj, 0, TL);;
			hi = XEv.h[itop];
			zi = zb[itop];
		}
		if ((iy == 0) && levBL < lev)//(ix==16) i.e. in the right halo
		{
			int jj = TLBL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + blockDim.x / 2;
			int ibc = memloc(halowidth, blkmemwidth, jj, blockDim.x - 1, BL);
			hn = XEv.h[ibc];
			zn = zb[ibc];
		}

		T sl, sr;
		sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi) * (zi - zl));
		sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn) * (zn - zr));


		XFlux.Fqvy[i] = XFlux.Fqvy[i] - fmv * sl;
		XFlux.Sv[i] = XFlux.Sv[i] - fmv * sr;
	}
}
template __global__ void AddSlopeSourceYGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* zb);
template __global__ void AddSlopeSourceYGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* zb);




template <class T> __host__ void updateKurgYCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax,T*zb)
{

	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps)+epsi;
	T delta;
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);
	

	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	int TL, BLTL, BL, TLBL, levTL, levBL, lev;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		


		
		TL = XBlock.TopLeft[ib];
		levTL = XBlock.level[TL];
		BLTL = XBlock.BotLeft[TL];

		BL = XBlock.BotLeft[ib];
		levBL = XBlock.level[BL];
		TLBL = XBlock.TopLeft[BL];

		lev = XBlock.level[ib];

		delta = calcres(XParam.dx, lev);

		for (int iy = 0; iy < (XParam.blkwidth + XParam.halowidth); iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);

				T cm = T(1.0);
				T fmv = T(1.0);

				T dhdyi = XGrad.dhdy[i];
				T dhdymin = XGrad.dhdy[ibot];
				T hi = XEv.h[i];
				T hn = XEv.h[ibot];
				T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, ga;


				

				if (hi > eps || hn > eps)
				{
					hn = XEv.h[ibot];
					dx = delta * T(0.5);
					zi = XEv.zs[i] - hi;
					zl = zi - dx * (XGrad.dzsdy[i] - dhdyi);
					zn = XEv.zs[ibot] - hn;
					zr = zn + dx * (XGrad.dzsdy[ibot] - dhdymin);
					zlr = max(zl, zr);

					hl = hi - dx * dhdyi;
					up = XEv.v[i] - dx * XGrad.dvdy[i];
					hp = max(T(0.0), hl + zl - zlr);

					hr = hn + dx * dhdymin;
					um = XEv.v[ibot] + dx * XGrad.dvdy[ibot];
					hm = max(T(0.0), hr + zr - zlr);


					ga = g * T(0.5);

					//// Reimann solver
					T fh, fu, fv, sl, sr, dt;

					//solver below also modifies fh and fu
					dt = KurgSolver(g, delta, epsi, CFL, cm, fmv, hp, hm, up, um, fh, fu);

					if (dt < dtmax[i])
					{
						dtmax[i] = dt;
					}
					else
					{
						dtmax[i] = T(1.0) / epsi;
					}


					if (fh > T(0.0))
					{
						fv = (XEv.u[ibot] + dx * XGrad.dudy[ibot]) * fh;
					}
					else
					{
						fv = (XEv.u[i] - dx * XGrad.dudy[i]) * fh;
					}
					//fv = (fh > 0.f ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
					/**
					#### Topographic source term

					In the case of adaptive refinement, care must be taken to ensure
					well-balancing at coarse/fine faces */

					if ((iy == XParam.blkwidth) && levTL < lev)//(ix==16) i.e. in the top halo
					{
						int jj = BLTL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + XParam.blkwidth / 2;
						int itop = memloc(halowidth, blkmemwidth, jj, 0, TL);
						hi = XEv.h[itop];
						zi = zb[itop];
					}
					if ((iy == 0) && levBL < lev)//(ix==16) i.e. in the bot halo
					{
						int jj = TLBL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + XParam.blkwidth / 2;
						int ibc = memloc(halowidth, blkmemwidth, jj, XParam.blkwidth - 1, BL);
						hn = XEv.h[ibc];
						zn = zb[ibc];
					}

					sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi) * (zi - zl));
					sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn) * (zn - zr));

					////Flux update

					XFlux.Fhv[i] = fmv * fh;
					XFlux.Fqvy[i] = fmv *  (fu - sl);
					XFlux.Sv[i] = fmv * (fu - sr);
					XFlux.Fquy[i] = fmv * fv;
				}
				else
				{
					dtmax[i] = T(1.0) / epsi;
					XFlux.Fhv[i] = T(0.0);
					XFlux.Fqvy[i] = T(0.0);
					XFlux.Sv[i] = T(0.0);
					XFlux.Fquy[i] = T(0.0);
				}
			}
		}
	}
}
template __host__ void updateKurgYCPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float *zb);
template __host__ void updateKurgYCPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double *zb);

template <class T> __host__ void AddSlopeSourceYCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* zb)
{
	T delta;
	T g = T(XParam.g);
	T ga = T(0.5) * g;
	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps) + epsi;

	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		

		int lev = XBlock.level[ib];
		delta = calcres(XParam.dx, lev);
		// neighbours for source term
		int TL, BLTL, BL, TLBL, levTL, levBL;
		TL = XBlock.TopLeft[ib];
		levTL = XBlock.level[TL];
		BLTL = XBlock.BotLeft[TL];

		BL = XBlock.BotLeft[ib];
		levBL = XBlock.level[BL];
		TLBL = XBlock.TopLeft[BL];





		for (int iy = 0; iy < (XParam.blkwidth + XParam.halowidth); iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);



				T cm = T(1.0);
				T fmv = T(1.0);

				T dx, zi, zl, zn, zr, zlr, hl, hp, hr, hm;

				T dhdyi = XGrad.dhdy[i];
				T dhdymin = XGrad.dhdy[ibot];
				T hi = XEv.h[i];
				T hn = XEv.h[ibot];


				if (hi > eps || hn > eps)
				{

					// along X these are same as in Kurgannov
					dx = delta * T(0.5);
					zi = XEv.zs[i] - hi;

					zl = zi - dx * (XGrad.dzsdy[i] - dhdyi);
					zn = XEv.zs[ibot] - hn;
					zr = zn + dx * (XGrad.dzsdy[ibot] - dhdymin);
					zlr = max(zl, zr);

					hl = hi - dx * dhdyi;
					hp = max(T(0.0), hl + zl - zlr);

					hr = hn + dx * dhdymin;
					hm = max(T(0.0), hr + zr - zlr);


					//#### Topographic source term
					//In the case of adaptive refinement, care must be taken to ensure
					//	well - balancing at coarse / fine faces(see[notes / balanced.tm]()). * /

					if ((iy == XParam.blkwidth) && levTL < lev)//(ix==16) i.e. in the right halo
					{
						int jj = BLTL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + XParam.blkwidth / 2;
						int itop = memloc(halowidth, blkmemwidth, jj, 0, TL);;
						hi = XEv.h[itop];
						zi = zb[itop];
					}
					if ((iy == 0) && levBL < lev)//(ix==16) i.e. in the right halo
					{
						int jj = TLBL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + XParam.blkwidth / 2;
						int ibc = memloc(halowidth, blkmemwidth, jj, XParam.blkwidth - 1, BL);
						hn = XEv.h[ibc];
						zn = zb[ibc];
					}

					T sl, sr;
					sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi) * (zi - zl));
					sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn) * (zn - zr));


					XFlux.Fqvy[i] = XFlux.Fqvy[i] - fmv * sl;
					XFlux.Sv[i] = XFlux.Sv[i] - fmv * sr;
				}
			}
		}
	}
}
template __host__ void AddSlopeSourceYCPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* zb);
template __host__ void AddSlopeSourceYCPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* zb);



template <class T> __host__ __device__ T KurgSolver(T g, T delta,T epsi, T CFL, T cm, T fm,  T hp, T hm, T up,T um, T &fh, T &fu)
{
	//// Reimann solver
	T dt;

	//We can now call one of the approximate Riemann solvers to get the fluxes.
	T cp, cmo, ap, am, qm, qp, a, dlt, ad, hm2, hp2, ga, apm;


	cp = sqrt(g * hp);
	cmo = sqrt(g * hm);

	ap = max(max(up + cp, um + cmo), T(0.0));
	//ap = max(ap, 0.0f);

	am = min(min(up - cp, um - cmo), T(0.0));
	//am = min(am, 0.0f);
	ad = T(1.0) / (ap - am);
	//Correct for spurious currents in really shallow depth
	qm = hm * um;
	qp = hp * up;
	//qm = hm*um*(sqrtf(2.0f) / sqrtf(1.0f + max(1.0f, powf(epsc / hm, 4.0f))));
	//qp = hp*up*(sqrtf(2.0f) / sqrtf(1.0f + max(1.0f, powf(epsc / hp, 4.0f))));

	hm2 = hm * hm;
	hp2 = hp * hp;
	a = max(ap, -am);
	ga = g * T(0.5);
	apm = ap * am;
	dlt = delta * cm / fm;

	if (a > epsi)
	{
		fh = (ap * qm - am * qp + apm * (hp - hm)) * ad;// H  in eq. 2.24 or eq 3.7 for F(h)
		fu = (ap * (qm * um + ga * hm2) - am * (qp * up + ga * hp2) + apm * (qp - qm)) * ad;// Eq 3.7 second term (Y direction)
		dt = CFL * dlt / a;
		

	}
	else
	{
		fh = T(0.0);
		fu = T(0.0);
		dt = T(1.0) / epsi;
	}
	return dt;
}

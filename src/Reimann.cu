#include "Reimann.h"


/*! \fn void UpdateButtingerXGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb)
* \brief "Adaptive" second-order hydrostatic reconstruction. GPU version for t X-axis
*
* ## Description
* This function computes the flux term at the cell interface using the hydrostatic reconstruction from Buttinger et al (2019).
* This reconstruction is safe for steep slope with thin water depth and is well-balanced meaning that it conserve the "lake-at-rest" states.
*
* For optimising the code on CPU and GPU there are 4 versions of this function: X or Y and CPU or GPU
*
* ## Where does this come from:
* This scheme was adapted/modified from the Basilisk / B-Flood source code. I (CypB) changed the zr and zl term back to the Audusse type reconstruction
* http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h
*
* Reference:
* Kirstetter, G., Delestre, O., Lagree, P.-Y., Popinet, S., and Josserand, C.: B-flood 1.0: an open-source Saint-Venant model for flash flood simulation using adaptive refinement, Geosci. Model Dev. Discuss. [preprint], https://doi.org/10.5194/gmd-2021-15, in review, 2021.*
* Buttinger-Kreuzhuber, A., Horvath, Z., Noelle, S., Bloschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional
* structured grids over abrupt topography, Advances in water resources, 127, 89-108, 2019.
*/
template <class T> __global__ void UpdateButtingerXGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
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
	T eps = T(XParam.eps) + epsi;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);

	T ybo = T(XParam.yo + XBlock.yo[ib]);


	//T dhdxi = XGrad.dhdx[i];
	//T dhdxmin = XGrad.dhdx[ileft];
	T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T fmu = T(1.0);

	T hi = XEv.h[i];

	T hn = XEv.h[ileft];


	if (hi > eps || hn > eps)
	{
		T dx, zi, zn, hr, hl, etar, etal, zr, zl, zA, zCN, hCNr, hCNl;
		T ui, vi, uli, vli, dhdxi, dhdxil, dudxi, dudxil, dvdxi,dvdxil;

		T ga = g * T(0.5);
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
		zr = etar - hr;// zi - dx * XGrad.dzbdx[i];
		zl = etal - hl;// zn + dx * XGrad.dzbdx[ileft];

		//define the Audusse terms
		zA = max(zr, zl);

		// Now the CN terms
		zCN = min(zA, min(etal, etar));
		hCNr = max(T(0.0), min(etar - zCN, hr));
		hCNl = max(T(0.0), min(etal - zCN, hl));
		
		//Velocity reconstruction
		//To avoid high velocities near dry cells, we reconstruct velocities according to Bouchut.
		T ul, ur, vl, vr,sl,sr;
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
		dt = hllc(g, delta, epsi, CFL, cm, fmu, hCNl, hCNr, ul, ur, fh, fu);
		//hllc(T g, T delta, T epsi, T CFL, T cm, T fm, T hm, T hp, T um, T up, T & fh, T & fq)

		if (dt < dtmax[i])
		{
			dtmax[i] = dt;
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

		sl = ga * (hi + hCNr) * (zi - zCN);
		sr = ga * (hCNl + hn) * (zn - zCN);

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
template __global__ void UpdateButtingerXGPU(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float* zb);
template __global__ void UpdateButtingerXGPU(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double* zb);


/*! \fn void UpdateButtingerXCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb)
* \brief "Adaptive" second-order hydrostatic reconstruction. CPU version for the X-axis
*
* ## Description
* This function computes the flux term at the cell interface using the hydrostatic reconstruction from Buttinger et al (2019).
* This reconstruction is safe for steep slope with thin water depth and is well-balanced meaning that it conserve the "lake-at-rest" states.
*
* For optimising the code on CPU and GPU there are 4 versions of this function: X or Y and CPU or GPU
*
* ## Where does this come from:
* This scheme was adapted/modified from the Basilisk / B-Flood source code. I (CypB) changed the zr and zl term back to the Audusse type reconstruction
* http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h
*
* Reference:
* Kirstetter, G., Delestre, O., Lagree, P.-Y., Popinet, S., and Josserand, C.: B-flood 1.0: an open-source Saint-Venant model for flash flood simulation using adaptive refinement, Geosci. Model Dev. Discuss. [preprint], https://doi.org/10.5194/gmd-2021-15, in review, 2021.*
* Buttinger-Kreuzhuber, A., Horvath, Z., Noelle, S., Bloschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional
* structured grids over abrupt topography, Advances in water resources, 127, 89-108, 2019.
*/
template <class T> __host__ void UpdateButtingerXCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb)
{


	T delta;
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);
	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps) + epsi;

	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	int RB, LBRB, LB, RBLB, levRB, levLB;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		int lev = XBlock.level[ib];
		delta = calcres(T(XParam.delta), lev);

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

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);

				T ybo = T(XParam.yo + XBlock.yo[ib]);


				//T dhdxi = XGrad.dhdx[i];
				//T dhdxmin = XGrad.dhdx[ileft];
				T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
				T fmu = T(1.0);

				T hi = XEv.h[i];

				T hn = XEv.h[ileft];


				if (hi > eps || hn > eps)
				{
					T dx, zi, zn, hr, hl, etar, etal, zr, zl, zA, zCN, hCNr, hCNl;
					T ui, vi, uli, vli, dhdxi, dhdxil, dudxi, dudxil, dvdxi, dvdxil;

					T ga = g * T(0.5);
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
					zr = etar - hr;// zi - dx * XGrad.dzbdx[i];
					zl = etal - hl;// zn + dx * XGrad.dzbdx[ileft];

					//define the Audusse terms
					zA = max(zr, zl);

					// Now the CN terms
					zCN = min(zA, min(etal, etar));
					hCNr = max(T(0.0), min(etar - zCN, hr));
					hCNl = max(T(0.0), min(etal - zCN, hl));

					//Velocity reconstruction
					//To avoid high velocities near dry cells, we reconstruct velocities according to Bouchut.
					T ul, ur, vl, vr, sl, sr;
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
					dt = hllc(g, delta, epsi, CFL, cm, fmu, hCNl, hCNr, ul, ur, fh, fu);
					//hllc(T g, T delta, T epsi, T CFL, T cm, T fm, T hm, T hp, T um, T up, T & fh, T & fq)

					if (dt < dtmax[i])
					{
						dtmax[i] = dt;
					}
					

					fv = (fh > 0. ? vl : vr) * fh;


					// Topographic source term

					// In the case of adaptive refinement, care must be taken to ensure
					// well-balancing at coarse/fine faces (see [notes/balanced.tm]()). 
					if ((ix == XParam.blkwidth) && levRB < lev)//(ix==16) i.e. in the right halo
					{
						int jj = LBRB == ib ? ftoi(floor(iy * (T)0.5)) : ftoi(floor(iy * (T)0.5) + XParam.blkwidth / 2);
						int iright = memloc(halowidth, blkmemwidth, 0, jj, RB);;
						hi = XEv.h[iright];
						zi = zb[iright];
					}
					if ((ix == 0) && levLB < lev)//(ix==16) i.e. in the right halo if you 
					{
						int jj = RBLB == ib ? ftoi(floor(iy * (T)0.5)) : ftoi(floor(iy * (T)0.5) + XParam.blkwidth / 2);
						int ilc = memloc(halowidth, blkmemwidth, XParam.blkwidth- 1, jj, LB);
						
						hn = XEv.h[ilc];
						zn = zb[ilc];
					}

					sl = ga * (hi + hCNr) * (zi - zCN);
					sr = ga * (hCNl + hn) * (zn - zCN);

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
template __host__ void UpdateButtingerXCPU(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float* zb);
template __host__ void UpdateButtingerXCPU(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double* zb);


/*! \fn void UpdateButtingerYGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb)
* \brief "Adaptive" second-order hydrostatic reconstruction. GPU version for the Y-axis
*
* ## Description
* This function computes the flux term at the cell interface using the hydrostatic reconstruction from Buttinger et al (2019).
* This reconstruction is safe for steep slope with thin water depth and is well-balanced meaning that it conserve the "lake-at-rest" states.
*
* For optimising the code on CPU and GPU there are 4 versions of this function: X or Y and CPU or GPU
*
* ## Where does this come from:
* This scheme was adapted/modified from the Basilisk / B-Flood source code. I (CypB) changed the zr and zl term back to the Audusse type reconstruction
* http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h
*
* Reference:
* Kirstetter, G., Delestre, O., Lagree, P.-Y., Popinet, S., and Josserand, C.: B-flood 1.0: an open-source Saint-Venant model for flash flood simulation using adaptive refinement, Geosci. Model Dev. Discuss. [preprint], https://doi.org/10.5194/gmd-2021-15, in review, 2021.*
* Buttinger-Kreuzhuber, A., Horvath, Z., Noelle, S., Bloschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional
* structured grids over abrupt topography, Advances in water resources, 127, 89-108, 2019.
*/
template <class T> __global__ void UpdateButtingerYGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
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
	T eps = T(XParam.eps) + epsi;
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);


	T ybo = T(XParam.yo + XBlock.yo[ib]);

	//T dhdyi = XGrad.dhdy[i];
	//T dhdymin = XGrad.dhdy[ibot];
	T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy)) : T(1.0);

	T hi = XEv.h[i];

	T hn = XEv.h[ibot];


	if (hi > eps || hn > eps)
	{
		T dx, zi, zn, hr, hl, etar, etal, zr, zl, zA, zCN, hCNr, hCNl;
		T ui, vi, uli, vli, dhdyi, dhdyil, dudyi, dudyil, dvdyi, dvdyil;

		T ga = g * T(0.5);
		// along X
		dx = delta * T(0.5);
		zi = zb[i];
		zn = zb[ibot];

		ui = XEv.u[i];
		vi = XEv.v[i];
		uli = XEv.u[ibot];
		vli = XEv.v[ibot];

		dhdyi = XGrad.dhdy[i];
		dhdyil = XGrad.dhdy[ibot];
		dudyi = XGrad.dudy[i];
		dudyil = XGrad.dudy[ibot];
		dvdyi = XGrad.dvdy[i];
		dvdyil = XGrad.dvdy[ibot];


		hr = hi - dx * dhdyi;
		hl = hn + dx * dhdyil;
		etar = XEv.zs[i] - dx * XGrad.dzsdy[i];
		etal = XEv.zs[ibot] + dx * XGrad.dzsdy[ibot];

		//define the topography term at the interfaces
		zr = etar - hr;// zi - dx * XGrad.dzbdy[i];
		zl = etal - hl;// zn + dx * XGrad.dzbdy[ibot];

		//define the Audusse terms
		zA = max(zr, zl);

		// Now the CN terms
		zCN = min(zA, min(etal, etar));
		hCNr = max(T(0.0), min(etar - zCN, hr));
		hCNl = max(T(0.0), min(etal - zCN, hl));

		//Velocity reconstruction
		//To avoid high velocities near dry cells, we reconstruct velocities according to Bouchut.
		T ul, ur, vl, vr, sl, sr;
		if (hi > eps) {
			ur = ui - (1. + dx * dhdyi / hi) * dx * dudyi;
			vr = vi - (1. + dx * dhdyi / hi) * dx * dvdyi;
		}
		else {
			ur = ui - dx * dudyi;
			vr = vi - dx * dvdyi;
		}
		if (hn > eps) {
			ul = uli + (1. - dx * dhdyil / hn) * dx * dudyil;
			vl = vli + (1. - dx * dhdyil / hn) * dx * dvdyil;
		}
		else {
			ul = uli + dx * dudyil;
			vl = vli + dx * dvdyil;
		}




		T fh, fu, fv, dt;


		//solver below also modifies fh and fu
		dt = hllc(g, delta, epsi, CFL, cm, fmv, hCNl, hCNr, vl, vr, fh, fu);
		//hllc(T g, T delta, T epsi, T CFL, T cm, T fm, T hm, T hp, T um, T up, T & fh, T & fq)

		if (dt < dtmax[i])
		{
			dtmax[i] = dt;
		}
		

		fv = (fh > 0. ? ul : ur) * fh;


		// Topographic source term

		// In the case of adaptive refinement, care must be taken to ensure
		// well-balancing at coarse/fine faces (see [notes/balanced.tm]()). 
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

		sl = ga * (hi + hCNr) * (zi - zCN);
		sr = ga * (hCNl + hn) * (zn - zCN);

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
template __global__ void UpdateButtingerYGPU(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float* zb);
template __global__ void UpdateButtingerYGPU(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double* zb);

/*! \fn void UpdateButtingerYCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb)
* \brief "Adaptive" second-order hydrostatic reconstruction. CPU version for the Y-axis
*
* ## Description
* This function computes the flux term at the cell interface using the hydrostatic reconstruction from Buttinger et al (2019).
* This reconstruction is safe for steep slope with thin water depth and is well-balanced meaning that it conserve the "lake-at-rest" states.
*
* For optimising the code on CPU and GPU there are 4 versions of this function: X or Y and CPU or GPU
*
* ## Where does this come from:
* This scheme was adapted/modified from the Basilisk / B-Flood source code. I (CypB) changed the zr and zl term back to the Audusse type reconstruction
* http://basilisk.fr/sandbox/b-flood/saint-venant-topo.h
*
* Reference:
* Kirstetter, G., Delestre, O., Lagree, P.-Y., Popinet, S., and Josserand, C.: B-flood 1.0: an open-source Saint-Venant model for flash flood simulation using adaptive refinement, Geosci. Model Dev. Discuss. [preprint], https://doi.org/10.5194/gmd-2021-15, in review, 2021.*
* Buttinger-Kreuzhuber, A., Horvath, Z., Noelle, S., Bloschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional
* structured grids over abrupt topography, Advances in water resources, 127, 89-108, 2019.
*/
template <class T> __host__ void UpdateButtingerYCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb)
{

	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps) + epsi;
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

		delta = calcres(T(XParam.delta), lev);

		for (int iy = 0; iy < (XParam.blkwidth + XParam.halowidth); iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);

				T ybo = T(XParam.yo + XBlock.yo[ib]);


				//T dhdyi = XGrad.dhdy[i];
				//T dhdymin = XGrad.dhdy[ibot];
				T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
				T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy)) : T(1.0);

				T hi = XEv.h[i];

				T hn = XEv.h[ibot];


				if (hi > eps || hn > eps)
				{
					T dx, zi, zn, hr, hl, etar, etal, zr, zl, zA, zCN, hCNr, hCNl;
					T ui, vi, uli, vli, dhdyi, dhdyil, dudyi, dudyil, dvdyi, dvdyil;

					T ga = g * T(0.5);
					// along X
					dx = delta * T(0.5);
					zi = zb[i];
					zn = zb[ibot];

					ui = XEv.u[i];
					vi = XEv.v[i];
					uli = XEv.u[ibot];
					vli = XEv.v[ibot];

					dhdyi = XGrad.dhdy[i];
					dhdyil = XGrad.dhdy[ibot];
					dudyi = XGrad.dudy[i];
					dudyil = XGrad.dudy[ibot];
					dvdyi = XGrad.dvdy[i];
					dvdyil = XGrad.dvdy[ibot];


					hr = hi - dx * dhdyi;
					hl = hn + dx * dhdyil;
					etar = XEv.zs[i] - dx * XGrad.dzsdy[i];
					etal = XEv.zs[ibot] + dx * XGrad.dzsdy[ibot];

					//define the topography term at the interfaces
					zr = etar - hr;// zi - dx * XGrad.dzbdy[i];
					zl = etal - hl;// zn + dx * XGrad.dzbdy[ibot];

					//define the Audusse terms
					zA = max(zr, zl);

					// Now the CN terms
					zCN = min(zA, min(etal, etar));
					hCNr = max(T(0.0), min(etar - zCN, hr));
					hCNl = max(T(0.0), min(etal - zCN, hl));

					//Velocity reconstruction
					//To avoid high velocities near dry cells, we reconstruct velocities according to Bouchut.
					T ul, ur, vl, vr, sl, sr;
					if (hi > eps) {
						ur = ui - (1. + dx * dhdyi / hi) * dx * dudyi;
						vr = vi - (1. + dx * dhdyi / hi) * dx * dvdyi;
					}
					else {
						ur = ui - dx * dudyi;
						vr = vi - dx * dvdyi;
					}
					if (hn > eps) {
						ul = uli + (1. - dx * dhdyil / hn) * dx * dudyil;
						vl = vli + (1. - dx * dhdyil / hn) * dx * dvdyil;
					}
					else {
						ul = uli + dx * dudyil;
						vl = vli + dx * dvdyil;
					}




					T fh, fu, fv, dt;


					//solver below also modifies fh and fu
					dt = hllc(g, delta, epsi, CFL, cm, fmv, hCNl, hCNr, vl, vr, fh, fu);
					//hllc(T g, T delta, T epsi, T CFL, T cm, T fm, T hm, T hp, T um, T up, T & fh, T & fq)

					if (dt < dtmax[i])
					{
						dtmax[i] = dt;
					}
					

					fv = (fh > 0. ? ul : ur) * fh;


					// Topographic source term

					// In the case of adaptive refinement, care must be taken to ensure
					// well-balancing at coarse/fine faces (see [notes/balanced.tm]()). 
					if ((iy == XParam.blkwidth) && levTL < lev)//(ix==16) i.e. in the top halo
					{
						int jj = BLTL == ib ? ftoi(floor(ix * (T)0.5)) : ftoi(floor(ix * (T)0.5) + XParam.blkwidth / 2);
						int itop = memloc(halowidth, blkmemwidth, jj, 0, TL);
						hi = XEv.h[itop];
						zi = zb[itop];
					}
					if ((iy == 0) && levBL < lev)//(ix==16) i.e. in the bot halo
					{
						int jj = TLBL == ib ? ftoi(floor(ix * (T)0.5)) : ftoi(floor(ix * (T)0.5) + XParam.blkwidth / 2);
						int ibc = memloc(halowidth, blkmemwidth, jj, XParam.blkwidth - 1, BL);
						// Warning I think the above is wrong and should be as below to be consistent with halo flux scheme:
						//int ibc = memloc(halowidth, blkmemwidth, jj, XParam.blkwidth, BL);
						hn = XEv.h[ibc];
						zn = zb[ibc];
					}

					sl = ga * (hi + hCNr) * (zi - zCN);
					sr = ga * (hCNl + hn) * (zn - zCN);

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
		}
	}
}
template __host__ void UpdateButtingerYCPU(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float* zb);
template __host__ void UpdateButtingerYCPU(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double* zb);




/*! \fn T hllc(T g, T delta, T epsi, T CFL, T cm, T fm, T hm, T hp, T um, T up, T &fh, T &fq)
* \brief Calculate the  Harten-Lax-van Leer-contact (HLLC) flux.
*
* ## Description
* This an implementation of the HLLC solver. 
* 
*
* ## Where does this come from:
* This scheme was adapted/modified from the Basilisk source code.
* http://basilisk.fr/src/riemann.h
* 
* Reference:
* (Basilisk reference the scheme from Kurganov reference below)
* Kurganov, A., & Levy, D. (2002). Central-upwind schemes for the
*    Saint-Venant system. Mathematical Modelling and Numerical
*    Analysis, 36(3), 397-425.
*
*/
template <class T> __host__ __device__ T hllc(T g, T delta, T epsi, T CFL, T cm, T fm, T hm, T hp, T um, T up, T &fh, T &fq)
{
	T cp, cmo , dt, ustar, cstar, SL, SR, fhm, fum,fhp, fup,dlt;
	cmo = sqrt(g * hm);
	cp = sqrt(g * hp);
	ustar = (um + up) / T(2.) + cmo - cp;
	cstar = (cmo + cp) / T(2.) + (um - up) / T(4.);
	SL = hm == T(0.) ? up - T(2.) * cp : min(um - cmo, ustar - cstar);
	SR = hp == T(0.) ? um + T(2.) * cmo : max(up + cp, ustar + cstar);

	if (T(0.) <= SL) {
		fh = um * hm;
		fq = hm * (um * um + g * hm / T(2.));
	}
	else if (T(0.) >= SR) {
		fh = up * hp;
		fq = hp * (up * up + g * hp / T(2.));
	}
	else {
		fhm = um * hm;
		fum = hm * (um * um + g * hm / T(2.));
		fhp = up * hp;
		fup = hp * (up * up + g * hp / T(2.));
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


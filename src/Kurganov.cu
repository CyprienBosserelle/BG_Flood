#include "Kurganov.h"

#define SHARED_MEM_HALO_WIDTH 1
#define STATIC_MAX_BLOCK_X 16
#define STATIC_MAX_BLOCK_Y 16

template <class T> __global__ void updateKurgXGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T*zb)
{
	// Shared memory declarations
	__shared__ T s_h[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_zs[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_u[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_v[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_dhdx[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_dzsdx[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_dudx[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_dvdx[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	
	unsigned int halowidth = XParam.halowidth; // Runtime halowidth
	unsigned int blkmemwidth = blockDim.y + halowidth * 2; // Keep original, used by memloc
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;

	// Safety check for static shared memory bounds
	if (ix >= STATIC_MAX_BLOCK_X || iy >= STATIC_MAX_BLOCK_Y) {
		return;
	}

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
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);
	// This is based on kurganov and Petrova 2007

	// Global memory indices
	int global_idx_center = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int global_idx_left_halo = memloc(halowidth, blkmemwidth, ix - 1, iy, ib); // ix-1 for halo

	// Shared memory indices: data for (ix,iy) at s_array[iy][ix+1], halo for (ix-1,iy) at s_array[iy][ix]
	int s_idx_current = ix + SHARED_MEM_HALO_WIDTH; // ix + 1
	int s_idx_halo = ix;                           // ix

	// Load data into shared memory
	s_h[iy][s_idx_current]    = XEv.h[global_idx_center];
	s_zs[iy][s_idx_current]   = XEv.zs[global_idx_center];
	s_u[iy][s_idx_current]    = XEv.u[global_idx_center];
	s_v[iy][s_idx_current]    = XEv.v[global_idx_center];
	s_dhdx[iy][s_idx_current] = XGrad.dhdx[global_idx_center];
	s_dzsdx[iy][s_idx_current]= XGrad.dzsdx[global_idx_center];
	s_dudx[iy][s_idx_current] = XGrad.dudx[global_idx_center];
	s_dvdx[iy][s_idx_current] = XGrad.dvdx[global_idx_center];

	if (ix == 0) {
		s_h[iy][s_idx_halo]    = XEv.h[global_idx_left_halo];
		s_zs[iy][s_idx_halo]   = XEv.zs[global_idx_left_halo];
		s_u[iy][s_idx_halo]    = XEv.u[global_idx_left_halo];
		s_v[iy][s_idx_halo]    = XEv.v[global_idx_left_halo];
		s_dhdx[iy][s_idx_halo] = XGrad.dhdx[global_idx_left_halo];
		s_dzsdx[iy][s_idx_halo]= XGrad.dzsdx[global_idx_left_halo];
		s_dudx[iy][s_idx_halo] = XGrad.dudx[global_idx_left_halo];
		s_dvdx[iy][s_idx_halo] = XGrad.dvdx[global_idx_left_halo];
	}

	__syncthreads();

	T ybo = T(XParam.yo + XBlock.yo[ib]);

	// Access data from shared memory
	T dhdxi   = s_dhdx[iy][s_idx_current];
	T dhdxmin = s_dhdx[iy][s_idx_halo]; 
	T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T fmu = T(1.0);

	T hi = s_h[iy][s_idx_current];
	T hn = s_h[iy][s_idx_halo]; 
	
	if (hi > eps || hn > eps)
	{
		T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, sl, sr,ga;

		dx = delta * T(0.5);
		zi = s_zs[iy][s_idx_current] - hi;
		zl = zi - dx * (s_dzsdx[iy][s_idx_current] - dhdxi);
		
		zn = s_zs[iy][s_idx_halo] - hn; 
		zr = zn + dx * (s_dzsdx[iy][s_idx_halo] - dhdxmin); 

		zlr = max(zl, zr);

		hl = hi - dx * dhdxi;
		up = s_u[iy][s_idx_current] - dx * s_dudx[iy][s_idx_current]; 
		hp = max(T(0.0), hl + zl - zlr);

		hr = hn + dx * dhdxmin;
		um = s_u[iy][s_idx_halo] + dx * s_dudx[iy][s_idx_halo]; 
		hm = max(T(0.0), hr + zr - zlr);

		ga = g * T(0.5);
		T fh, fu, fv, dt;
		
		dt = KurgSolver(g, delta, epsi, CFL, cm, fmu, hp, hm, up, um, fh, fu);
		
		// dtmax is an output array, write to global memory using global_idx_center
		if (dt < dtmax[global_idx_center]) 
		{
			dtmax[global_idx_center] = dt;
		}
		
		if (fh > T(0.0))
		{
			fv = (s_v[iy][s_idx_halo] + dx * s_dvdx[iy][s_idx_halo]) * fh;
		}
		else
		{
			fv = (s_v[iy][s_idx_current] - dx * s_dvdx[iy][s_idx_current]) * fh;
		}
		
		// Boundary condition accesses - keep as global.
		// Initialize boundary variables with values from shared memory (or computed from shared).
		T hi_boundary = hi; 
		T zi_boundary = zi; // zi is computed from shared memory values
		T hn_boundary = hn;
		T zn_boundary = zn; // zn is computed from shared memory values

		// Original code: `if ((ix == blockDim.y) && levRB < lev)`
		// Assuming blockDim.y was a placeholder for the extent of the block in x-direction.
		// Using STATIC_MAX_BLOCK_X - 1 for the rightmost thread in the current block.
		if ((ix == STATIC_MAX_BLOCK_X - 1) && levRB < lev)
		{
			int jj = LBRB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + STATIC_MAX_BLOCK_Y / 2;
			int iright = memloc(halowidth, blkmemwidth, 0, jj, RB); // Accesses cell 0 of neighbor block RB
			hi_boundary = XEv.h[iright]; 
			zi_boundary = zb[iright]; // Note: zb is a direct global pointer argument to the kernel   
		}
		// Original code: `if ((ix == 0) && levLB < lev)`
		if ((ix == 0) && levLB < lev)
		{
			int jj = RBLB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + STATIC_MAX_BLOCK_Y / 2;
			// Accesses last cell (STATIC_MAX_BLOCK_X - 1) of neighbor block LB
			int ilc = memloc(halowidth, blkmemwidth, STATIC_MAX_BLOCK_X - 1, jj, LB); 
			hn_boundary = XEv.h[ilc]; 
			zn_boundary = zb[ilc]; // Note: zb is a direct global pointer argument   
		}

		sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi_boundary) * (zi_boundary - zl));
		sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn_boundary) * (zn_boundary - zr));
		
		////Flux update (writing to global memory, using global_idx_center)
		XFlux.Fhu[global_idx_center] = fmu * fh;
		XFlux.Fqux[global_idx_center] = fmu * (fu - sl);
		XFlux.Su[global_idx_center] = fmu * (fu - sr);
		XFlux.Fqvx[global_idx_center] = fmu * fv;
	}
	else
	{
		// dtmax is an output array, write to global memory using global_idx_center
		dtmax[global_idx_center] = T(1.0) / epsi;
		XFlux.Fhu[global_idx_center] = T(0.0);
		XFlux.Fqux[global_idx_center] = T(0.0);
		XFlux.Su[global_idx_center] = T(0.0);
		XFlux.Fqvx[global_idx_center] = T(0.0);
	}


}
template __global__ void updateKurgXGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float* zb);
template __global__ void updateKurgXGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double *zb);

template <class T> __global__ void updateKurgXATMGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb, T* Patm, T*dPdx)
{
	// Shared memory declarations
	__shared__ T s_h[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_zs[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_u[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_v[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_dhdx[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_dzsdx[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_dudx[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_dvdx[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_Patm[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_dPdx[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];

	unsigned int halowidth = XParam.halowidth; // Runtime halowidth
	unsigned int blkmemwidth = blockDim.y + halowidth * 2; // Keep original, used by memloc
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;

	// Safety check for static shared memory bounds
	if (ix >= STATIC_MAX_BLOCK_X || iy >= STATIC_MAX_BLOCK_Y) {
		return;
	}

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
	// This is based on kurganov and Petrova 2007

	// Global memory indices
	int global_idx_center = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int global_idx_left_halo = memloc(halowidth, blkmemwidth, ix - 1, iy, ib); // ix-1 for halo

	// Shared memory indices
	int s_idx_current = ix + SHARED_MEM_HALO_WIDTH; // ix + 1
	int s_idx_halo = ix;                           // ix

	// Load data into shared memory
	s_h[iy][s_idx_current]     = XEv.h[global_idx_center];
	s_zs[iy][s_idx_current]    = XEv.zs[global_idx_center];
	s_u[iy][s_idx_current]     = XEv.u[global_idx_center];
	s_v[iy][s_idx_current]     = XEv.v[global_idx_center];
	s_dhdx[iy][s_idx_current]  = XGrad.dhdx[global_idx_center];
	s_dzsdx[iy][s_idx_current] = XGrad.dzsdx[global_idx_center];
	s_dudx[iy][s_idx_current]  = XGrad.dudx[global_idx_center];
	s_dvdx[iy][s_idx_current]  = XGrad.dvdx[global_idx_center];
	s_Patm[iy][s_idx_current]  = Patm[global_idx_center];
	s_dPdx[iy][s_idx_current]  = dPdx[global_idx_center];

	if (ix == 0) {
		s_h[iy][s_idx_halo]     = XEv.h[global_idx_left_halo];
		s_zs[iy][s_idx_halo]    = XEv.zs[global_idx_left_halo];
		s_u[iy][s_idx_halo]     = XEv.u[global_idx_left_halo];
		s_v[iy][s_idx_halo]     = XEv.v[global_idx_left_halo];
		s_dhdx[iy][s_idx_halo]  = XGrad.dhdx[global_idx_left_halo];
		s_dzsdx[iy][s_idx_halo] = XGrad.dzsdx[global_idx_left_halo];
		s_dudx[iy][s_idx_halo]  = XGrad.dudx[global_idx_left_halo];
		s_dvdx[iy][s_idx_halo]  = XGrad.dvdx[global_idx_left_halo];
		s_Patm[iy][s_idx_halo]  = Patm[global_idx_left_halo];
		s_dPdx[iy][s_idx_halo]  = dPdx[global_idx_left_halo];
	}

	__syncthreads();

	T ybo = T(XParam.yo + XBlock.yo[ib]);

	// Access data from shared memory
	T dhdxi   = s_dhdx[iy][s_idx_current];
	T dhdxmin = s_dhdx[iy][s_idx_halo];
	T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T fmu = T(1.0);

	T hi = s_h[iy][s_idx_current];
	T hn = s_h[iy][s_idx_halo];

	if (hi > eps || hn > eps)
	{
		T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, sl, sr, ga;

		dx = delta * T(0.5);
		zi = s_zs[iy][s_idx_current] - hi + XParam.Pa2m * s_Patm[iy][s_idx_current];
		zl = zi - dx * (s_dzsdx[iy][s_idx_current] - dhdxi + XParam.Pa2m * s_dPdx[iy][s_idx_current]);
		
		zn = s_zs[iy][s_idx_halo] - hn + XParam.Pa2m * s_Patm[iy][s_idx_halo];
		zr = zn + dx * (s_dzsdx[iy][s_idx_halo] - dhdxmin + XParam.Pa2m * s_dPdx[iy][s_idx_halo]);

		zlr = max(zl, zr);

		hl = hi - dx * dhdxi;
		up = s_u[iy][s_idx_current] - dx * s_dudx[iy][s_idx_current];
		hp = max(T(0.0), hl + zl - zlr);

		hr = hn + dx * dhdxmin;
		um = s_u[iy][s_idx_halo] + dx * s_dudx[iy][s_idx_halo];
		hm = max(T(0.0), hr + zr - zlr);

		ga = g * T(0.5);
		T fh, fu, fv, dt;

		dt = KurgSolver(g, delta, epsi, CFL, cm, fmu, hp, hm, up, um, fh, fu);

		if (dt < dtmax[global_idx_center]) // Output to global memory
		{
			dtmax[global_idx_center] = dt;
		}

		if (fh > T(0.0))
		{
			fv = (s_v[iy][s_idx_halo] + dx * s_dvdx[iy][s_idx_halo]) * fh;
		}
		else
		{
			fv = (s_v[iy][s_idx_current] - dx * s_dvdx[iy][s_idx_current]) * fh;
		}
		
		// Boundary condition handling: Initialize with shared/local values, then overwrite if boundary global access occurs.
		T hi_boundary = hi;
		T zi_boundary = zi; // zi already incorporates s_Patm for current cell
		T hn_boundary = hn;
		T zn_boundary = zn; // zn already incorporates s_Patm for halo cell

		if ((ix == STATIC_MAX_BLOCK_X - 1) && levRB < lev)
		{
			int jj = LBRB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + STATIC_MAX_BLOCK_Y / 2;
			int iright = memloc(halowidth, blkmemwidth, 0, jj, RB);
			hi_boundary = XEv.h[iright];
			// Patm[iright] is a global access from a neighboring block.
			zi_boundary = zb[iright] + XParam.Pa2m * Patm[iright]; 
		}
		if ((ix == 0) && levLB < lev)
		{
			int jj = RBLB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + STATIC_MAX_BLOCK_Y / 2;
			int ilc = memloc(halowidth, blkmemwidth, STATIC_MAX_BLOCK_X - 1, jj, LB);
			hn_boundary = XEv.h[ilc];
			// Patm[ilc] is a global access from a neighboring block.
			zn_boundary = zb[ilc] + XParam.Pa2m * Patm[ilc];
		}

		sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi_boundary) * (zi_boundary - zl));
		sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn_boundary) * (zn_boundary - zr));

		////Flux update (writing to global memory)
		XFlux.Fhu[global_idx_center] = fmu * fh;
		XFlux.Fqux[global_idx_center] = fmu * (fu - sl);
		XFlux.Su[global_idx_center] = fmu * (fu - sr);
		XFlux.Fqvx[global_idx_center] = fmu * fv;
	}
	else
	{
		dtmax[global_idx_center] = T(1.0) / epsi; // Output to global memory
		XFlux.Fhu[global_idx_center] = T(0.0);
		XFlux.Fqux[global_idx_center] = T(0.0);
		XFlux.Su[global_idx_center] = T(0.0);
		XFlux.Fqvx[global_idx_center] = T(0.0);
	}



}
template __global__ void updateKurgXATMGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float* zb, float* Patm, float* dPdx);
template __global__ void updateKurgXATMGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double* zb, double* Patm, double* dPdx);


template <class T> __global__ void AddSlopeSourceXGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T * zb)
{
	// Shared memory declarations
	__shared__ T s_h[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_zs[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_dhdx[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_dzsdx[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_Fqux[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];
	__shared__ T s_Su[STATIC_MAX_BLOCK_Y][STATIC_MAX_BLOCK_X + SHARED_MEM_HALO_WIDTH];

	unsigned int halowidth = XParam.halowidth; // Runtime halowidth
	unsigned int blkmemwidth = blockDim.y + halowidth * 2; // Keep original for memloc
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;

	// Safety check for static shared memory bounds
	if (ix >= STATIC_MAX_BLOCK_X || iy >= STATIC_MAX_BLOCK_Y) {
		return;
	}

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
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);

	T ga = T(0.5) * g;

	// Global memory indices
	int global_idx_center = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int global_idx_left_halo = memloc(halowidth, blkmemwidth, ix - 1, iy, ib); // ix-1 for halo

	// Shared memory indices
	int s_idx_current = ix + SHARED_MEM_HALO_WIDTH; // ix + 1
	int s_idx_halo = ix;                           // ix

	// Load data into shared memory
	s_h[iy][s_idx_current]     = XEv.h[global_idx_center];
	s_zs[iy][s_idx_current]    = XEv.zs[global_idx_center];
	s_dhdx[iy][s_idx_current]  = XGrad.dhdx[global_idx_center];
	s_dzsdx[iy][s_idx_current] = XGrad.dzsdx[global_idx_center];
	s_Fqux[iy][s_idx_current]  = XFlux.Fqux[global_idx_center]; // Read current flux value
	s_Su[iy][s_idx_current]    = XFlux.Su[global_idx_center];   // Read current flux value

	if (ix == 0) {
		s_h[iy][s_idx_halo]     = XEv.h[global_idx_left_halo];
		s_zs[iy][s_idx_halo]    = XEv.zs[global_idx_left_halo];
		s_dhdx[iy][s_idx_halo]  = XGrad.dhdx[global_idx_left_halo];
		s_dzsdx[iy][s_idx_halo] = XGrad.dzsdx[global_idx_left_halo];
		// No halo load for s_Fqux, s_Su as they are modified in place based on current cell and its halo
	}

	__syncthreads();
	
	// Access data from shared memory
	T dhdxi   = s_dhdx[iy][s_idx_current];
	T dhdxmin = s_dhdx[iy][s_idx_halo];
	T hi      = s_h[iy][s_idx_current];
	T hn      = s_h[iy][s_idx_halo];
	//T cm = T(1.0); // Not used in this kernel as per original
	T fmu = T(1.0);

	T dx, zi, zl, zn, zr, zlr, hl, hp, hr, hm;

	if (hi > eps || hn > eps)
	{
		dx = delta * T(0.5);
		zi = s_zs[iy][s_idx_current] - hi;
		zl = zi - dx * (s_dzsdx[iy][s_idx_current] - dhdxi);

		zn = s_zs[iy][s_idx_halo] - hn;
		zr = zn + dx * (s_dzsdx[iy][s_idx_halo] - dhdxmin);

		zlr = max(zl, zr);

		hl = hi - dx * dhdxi;
		hp = max(T(0.0), hl + zl - zlr);

		hr = hn + dx * dhdxmin;
		hm = max(T(0.0), hr + zr - zlr);

		// Boundary condition handling
		T hi_boundary = hi;
		T zi_boundary = zi;
		T hn_boundary = hn;
		T zn_boundary = zn;

		if ((ix == STATIC_MAX_BLOCK_X - 1) && levRB < lev)
		{
			int jj = LBRB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + STATIC_MAX_BLOCK_Y / 2;
			int iright = memloc(halowidth, blkmemwidth, 0, jj, RB);
			hi_boundary = XEv.h[iright];
			zi_boundary = zb[iright]; 
		}
		if ((ix == 0) && levLB < lev)
		{
			int jj = RBLB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + STATIC_MAX_BLOCK_Y / 2;
			int ilc = memloc(halowidth, blkmemwidth, STATIC_MAX_BLOCK_X - 1, jj, LB);
			hn_boundary = XEv.h[ilc];
			zn_boundary = zb[ilc]; 
		}

		T sl, sr;
		sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi_boundary) * (zi_boundary - zl));
		sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn_boundary) * (zn_boundary - zr));

		// Update flux components in shared memory
		s_Fqux[iy][s_idx_current] = s_Fqux[iy][s_idx_current] - fmu * sl;
		s_Su[iy][s_idx_current]   = s_Su[iy][s_idx_current] - fmu * sr;

		// Synchronize before writing back if there were cross-thread dependencies on these shared flux values.
		// In this specific logic, each thread modifies its own s_Fqux[iy][s_idx_current] and s_Su[iy][s_idx_current].
		// However, for clarity or future modifications, a syncthreads can be defensive.
		// __syncthreads(); // Optional here, as each thread writes to a distinct location based on its original global_idx_center.

		// Write results back to global memory
		XFlux.Fqux[global_idx_center] = s_Fqux[iy][s_idx_current];
		XFlux.Su[global_idx_center]   = s_Su[iy][s_idx_current];
	}
	// If (hi <= eps && hn <= eps), the original code does nothing to Fqux or Su,
	// so no "else" branch is needed to write back unmodified Fqux/Su from shared memory,
	// as they would have been loaded with their original values.
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
	T ybo;
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	int RB, LBRB, LB, RBLB, levRB, levLB;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		int lev = XBlock.level[ib];
		delta = calcres(T(XParam.delta), lev);

		ybo = T(XParam.yo + XBlock.yo[ib]);

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
				T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);;
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
						int jj = LBRB == ib ? ftoi(floor(iy * (T)0.5)) : ftoi(floor(iy * (T)0.5) + XParam.blkwidth / 2);
						int iright = memloc(halowidth, blkmemwidth, 0, jj, RB);;
						hi = XEv.h[iright];
						zi = zb[iright];
					}
					if ((ix == 0) && levLB < lev)//(ix==16) i.e. in the right halo if you 
					{
						int jj = RBLB == ib ? ftoi(floor(iy * (T)0.5)) : ftoi(floor(iy * (T)0.5) + XParam.blkwidth / 2);
						int ilc = memloc(halowidth, blkmemwidth, XParam.blkwidth - 1, jj, LB);
						//int ilc = memloc(halowidth, blkmemwidth, -1, iy, ib);
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

template <class T> __host__ void updateKurgXATMCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb,T* Patm,T*dPdx)
{


	T delta;
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);
	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps) + epsi;

	T ybo;

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

		ybo = T(XParam.yo + XBlock.yo[ib]);
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < (XParam.blkwidth + XParam.halowidth); ix++)
			{




				// This is based on kurganov and Petrova 2007


				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);


				T dhdxi = XGrad.dhdx[i];
				T dhdxmin = XGrad.dhdx[ileft];
				T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);;
				T fmu = T(1.0);

				T hi = XEv.h[i];

				T hn = XEv.h[ileft];


				if (hi > eps || hn > eps)
				{
					T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, ga;

					// along X
					dx = delta * T(0.5);
					zi = XEv.zs[i] - hi + T(XParam.Pa2m) * Patm[i];

					//printf("%f\n", zi);


					//zl = zi - dx*(dzsdx[i] - dhdx[i]);
					zl = zi - dx * (XGrad.dzsdx[i] - dhdxi + T(XParam.Pa2m) * dPdx[i]);
					//printf("%f\n", zl);

					zn = XEv.zs[ileft] - hn + T(XParam.Pa2m) * Patm[ileft];

					//printf("%f\n", zn);
					zr = zn + dx * (XGrad.dzsdx[ileft] - dhdxmin + T(XParam.Pa2m) * dPdx[ileft]);


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
						int jj = LBRB == ib ? ftoi(floor(iy * (T)0.5)) : ftoi(floor(iy * (T)0.5) + XParam.blkwidth / 2);
						int iright = memloc(halowidth, blkmemwidth, 0, jj, RB);;
						hi = XEv.h[iright];
						zi = zb[iright] + T(XParam.Pa2m) * Patm[iright];
					}
					if ((ix == 0) && levLB < lev)//(ix==16) i.e. in the right halo if you 
					{
						int jj = RBLB == ib ? ftoi(floor(iy * (T)0.5)) : ftoi(floor(iy * (T)0.5) + XParam.blkwidth / 2);
						int ilc = memloc(halowidth, blkmemwidth, XParam.blkwidth - 1, jj, LB);
						//int ilc = memloc(halowidth, blkmemwidth, -1, iy, ib);
						hn = XEv.h[ilc];
						zn = zb[ilc] + T(XParam.Pa2m) * Patm[ilc];
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
template __host__ void updateKurgXATMCPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float* zb, float* Patm, float* dPdx);
template __host__ void updateKurgXATMCPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double* zb, double* Patm, double* dPdx);


template <class T> __host__ void AddSlopeSourceXCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* zb)
{
	T delta;
	//T g = T(XParam.g);
	//T CFL = T(XParam.CFL);
	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps) + epsi;

	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		int lev = XBlock.level[ib];
		delta = T(calcres(XParam.delta, lev));

		// neighbours for source term
		int RB, LBRB, LB, RBLB, levRB, levLB;
		RB = XBlock.RightBot[ib];
		levRB = XBlock.level[RB];
		LBRB = XBlock.LeftBot[RB];

		LB = XBlock.LeftBot[ib];
		levLB = XBlock.level[LB];
		RBLB = XBlock.RightBot[LB];



		//T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
		//T eps = T(XParam.eps) + epsi;

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
				//T cm = T(1.0);
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
						int jj = LBRB == ib ? ftoi(floor(iy * (T)0.5)) : ftoi(floor(iy * (T)0.5) + XParam.blkwidth / 2);
						int iright = memloc(halowidth, blkmemwidth, 0, jj, RB);;
						hi = XEv.h[iright];
						zi = zb[iright];
					}
					if ((ix == 0) && levLB < lev)//(ix==16) i.e. in the right halo if you 
					{
						int jj = RBLB == ib ? ftoi(floor(iy * (T)0.5)) : ftoi(floor(iy * (T)0.5) + XParam.blkwidth / 2);
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
	// Shared memory declarations for Y-direction halo
	__shared__ T s_h[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_zs[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_u[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_v[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_dhdy[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_dzsdy[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_dudy[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_dvdy[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];

	unsigned int halowidth = XParam.halowidth; // Runtime halowidth
	unsigned int blkmemwidth = blockDim.x + halowidth * 2; // Keep original for memloc
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;

	// Safety check for static shared memory bounds
	if (ix >= STATIC_MAX_BLOCK_X || iy >= STATIC_MAX_BLOCK_Y) {
		return;
	}

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
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);
	
	T ybo = T(XParam.yo + XBlock.yo[ib]);

	// Global memory indices
	int global_idx_center = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int global_idx_bottom_halo = memloc(halowidth, blkmemwidth, ix, iy - 1, ib); // iy-1 for "bottom" halo

	// Shared memory indices
	int s_iy_current = iy + SHARED_MEM_HALO_WIDTH; // iy + 1
	int s_iy_halo    = iy;                       // iy (for halo at iy=0)

	// Load data into shared memory
	s_h[s_iy_current][ix]     = XEv.h[global_idx_center];
	s_zs[s_iy_current][ix]    = XEv.zs[global_idx_center];
	s_u[s_iy_current][ix]     = XEv.u[global_idx_center];
	s_v[s_iy_current][ix]     = XEv.v[global_idx_center];
	s_dhdy[s_iy_current][ix]  = XGrad.dhdy[global_idx_center];
	s_dzsdy[s_iy_current][ix] = XGrad.dzsdy[global_idx_center];
	s_dudy[s_iy_current][ix]  = XGrad.dudy[global_idx_center];
	s_dvdy[s_iy_current][ix]  = XGrad.dvdy[global_idx_center];

	if (iy == 0) { // Load "bottom" halo (iy-1 data)
		s_h[s_iy_halo][ix]     = XEv.h[global_idx_bottom_halo];
		s_zs[s_iy_halo][ix]    = XEv.zs[global_idx_bottom_halo];
		s_u[s_iy_halo][ix]     = XEv.u[global_idx_bottom_halo];
		s_v[s_iy_halo][ix]     = XEv.v[global_idx_bottom_halo];
		s_dhdy[s_iy_halo][ix]  = XGrad.dhdy[global_idx_bottom_halo];
		s_dzsdy[s_iy_halo][ix] = XGrad.dzsdy[global_idx_bottom_halo];
		s_dudy[s_iy_halo][ix]  = XGrad.dudy[global_idx_bottom_halo];
		s_dvdy[s_iy_halo][ix]  = XGrad.dvdy[global_idx_bottom_halo];
	}
	
	__syncthreads();

	T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy)) : T(1.0);
		
	// Access data from shared memory
	T dhdyi   = s_dhdy[s_iy_current][ix];
	T dhdymin = s_dhdy[s_iy_halo][ix];
	T hi      = s_h[s_iy_current][ix];
	T hn      = s_h[s_iy_halo][ix];
	T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm,ga;

	if (hi > eps || hn > eps)
	{
		// Note: 'hn' is already loaded from shared memory s_h[s_iy_halo][ix]
		dx = delta * T(0.5);
		zi = s_zs[s_iy_current][ix] - hi;
		zl = zi - dx * (s_dzsdy[s_iy_current][ix] - dhdyi);
		zn = s_zs[s_iy_halo][ix] - hn;
		zr = zn + dx * (s_dzsdy[s_iy_halo][ix] - dhdymin);
		zlr = max(zl, zr);

		hl = hi - dx * dhdyi;
		up = s_v[s_iy_current][ix] - dx * s_dvdy[s_iy_current][ix]; // Current thread's v and dvdy
		hp = max(T(0.0), hl + zl - zlr);

		hr = hn + dx * dhdymin;
		um = s_v[s_iy_halo][ix] + dx * s_dvdy[s_iy_halo][ix]; // Bottom halo's v and dvdy
		hm = max(T(0.0), hr + zr - zlr);

		ga = g * T(0.5);
		T fh, fu, fv, sl, sr, dt;

		dt = KurgSolver(g, delta, epsi, CFL, cm, fmv, hp, hm, up, um, fh, fu);

		if (dt < dtmax[global_idx_center]) // Output to global memory
		{
			dtmax[global_idx_center] = dt;
		}
		
		if (fh > T(0.0))
		{
			fv = (s_u[s_iy_halo][ix] + dx * s_dudy[s_iy_halo][ix]) * fh; // Bottom halo's u and dudy
		}
		else
		{
			fv = (s_u[s_iy_current][ix] - dx * s_dudy[s_iy_current][ix]) * fh; // Current thread's u and dudy
		}
		
		T hi_boundary = hi;
		T zi_boundary = zi;
		T hn_boundary = hn;
		T zn_boundary = zn;

		// Original: if ((iy == blockDim.x) && levTL < lev)
		// Assuming blockDim.x here means the Y-extent of the block (STATIC_MAX_BLOCK_Y).
		// This condition checks the "top" boundary of the current block.
		if ((iy == STATIC_MAX_BLOCK_Y - 1) && levTL < lev)
		{
			int jj = BLTL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + STATIC_MAX_BLOCK_X / 2;
			int itop = memloc(halowidth, blkmemwidth, jj, 0, TL); // Accesses cell 0 (in Y) of neighbor block TL
			hi_boundary = XEv.h[itop];
			zi_boundary = zb[itop];
		}
		// Original: if ((iy == 0) && levBL < lev)
		// This condition checks the "bottom" boundary of the current block.
		if ((iy == 0) && levBL < lev)
		{
			int jj = TLBL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + STATIC_MAX_BLOCK_X / 2;
			// Accesses last cell (STATIC_MAX_BLOCK_Y - 1 in Y) of neighbor block BL
			int ibc = memloc(halowidth, blkmemwidth, jj, STATIC_MAX_BLOCK_Y - 1, BL); 
			hn_boundary = XEv.h[ibc];
			zn_boundary = zb[ibc];
		}

		sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi_boundary) * (zi_boundary - zl));
		sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn_boundary) * (zn_boundary - zr));

		////Flux update (writing to global memory)
		XFlux.Fhv[global_idx_center] = fmv * fh;
		XFlux.Fqvy[global_idx_center] = fmv * (fu - sl);
		XFlux.Sv[global_idx_center] = fmv * (fu - sr);
		XFlux.Fquy[global_idx_center] = fmv * fv;
	}
	else
	{
		dtmax[global_idx_center] = T(1.0) / epsi; // Output to global memory
		XFlux.Fhv[global_idx_center] = T(0.0);
		XFlux.Fqvy[global_idx_center] = T(0.0);
		XFlux.Sv[global_idx_center] = T(0.0);
		XFlux.Fquy[global_idx_center] = T(0.0);
	}
}
template __global__ void updateKurgYGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float* zb);
template __global__ void updateKurgYGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double *zb);


template <class T> __global__ void updateKurgYATMGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb, T* Patm,T* dPdy)
{
	// Shared memory declarations for Y-direction halo
	__shared__ T s_h[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_zs[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_u[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_v[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_dhdy[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_dzsdy[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_dudy[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_dvdy[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_Patm[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_dPdy[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];

	unsigned int halowidth = XParam.halowidth; // Runtime halowidth
	unsigned int blkmemwidth = blockDim.x + halowidth * 2; // Keep original for memloc
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;

	// Safety check for static shared memory bounds
	if (ix >= STATIC_MAX_BLOCK_X || iy >= STATIC_MAX_BLOCK_Y) {
		return;
	}

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

	T ybo = T(XParam.yo + XBlock.yo[ib]);

	// Global memory indices
	int global_idx_center = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int global_idx_bottom_halo = memloc(halowidth, blkmemwidth, ix, iy - 1, ib); // iy-1 for "bottom" halo

	// Shared memory indices
	int s_iy_current = iy + SHARED_MEM_HALO_WIDTH; // iy + 1
	int s_iy_halo    = iy;                       // iy (for halo at iy=0)

	// Load data into shared memory
	s_h[s_iy_current][ix]     = XEv.h[global_idx_center];
	s_zs[s_iy_current][ix]    = XEv.zs[global_idx_center];
	s_u[s_iy_current][ix]     = XEv.u[global_idx_center];
	s_v[s_iy_current][ix]     = XEv.v[global_idx_center];
	s_dhdy[s_iy_current][ix]  = XGrad.dhdy[global_idx_center];
	s_dzsdy[s_iy_current][ix] = XGrad.dzsdy[global_idx_center];
	s_dudy[s_iy_current][ix]  = XGrad.dudy[global_idx_center];
	s_dvdy[s_iy_current][ix]  = XGrad.dvdy[global_idx_center];
	s_Patm[s_iy_current][ix]  = Patm[global_idx_center];
	s_dPdy[s_iy_current][ix]  = dPdy[global_idx_center];

	if (iy == 0) { // Load "bottom" halo (iy-1 data)
		s_h[s_iy_halo][ix]     = XEv.h[global_idx_bottom_halo];
		s_zs[s_iy_halo][ix]    = XEv.zs[global_idx_bottom_halo];
		s_u[s_iy_halo][ix]     = XEv.u[global_idx_bottom_halo];
		s_v[s_iy_halo][ix]     = XEv.v[global_idx_bottom_halo];
		s_dhdy[s_iy_halo][ix]  = XGrad.dhdy[global_idx_bottom_halo];
		s_dzsdy[s_iy_halo][ix] = XGrad.dzsdy[global_idx_bottom_halo];
		s_dudy[s_iy_halo][ix]  = XGrad.dudy[global_idx_bottom_halo];
		s_dvdy[s_iy_halo][ix]  = XGrad.dvdy[global_idx_bottom_halo];
		s_Patm[s_iy_halo][ix]  = Patm[global_idx_bottom_halo];
		s_dPdy[s_iy_halo][ix]  = dPdy[global_idx_bottom_halo];
	}

	__syncthreads();

	T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy)) : T(1.0);

	// Access data from shared memory
	T dhdyi   = s_dhdy[s_iy_current][ix];
	T dhdymin = s_dhdy[s_iy_halo][ix];
	T hi      = s_h[s_iy_current][ix];
	T hn      = s_h[s_iy_halo][ix];
	T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, ga;

	if (hi > eps || hn > eps)
	{
		// hn is already from shared s_h[s_iy_halo][ix]
		dx = delta * T(0.5);
		
		zi = s_zs[s_iy_current][ix] - hi + XParam.Pa2m * s_Patm[s_iy_current][ix];
		zl = zi - dx * (s_dzsdy[s_iy_current][ix] - dhdyi + XParam.Pa2m * s_dPdy[s_iy_current][ix]);
		zn = s_zs[s_iy_halo][ix] - hn + XParam.Pa2m * s_Patm[s_iy_halo][ix];
		zr = zn + dx * (s_dzsdy[s_iy_halo][ix] - dhdymin + XParam.Pa2m * s_dPdy[s_iy_halo][ix]);

		zlr = max(zl, zr);

		hl = hi - dx * dhdyi;
		up = s_v[s_iy_current][ix] - dx * s_dvdy[s_iy_current][ix];
		hp = max(T(0.0), hl + zl - zlr);

		hr = hn + dx * dhdymin;
		um = s_v[s_iy_halo][ix] + dx * s_dvdy[s_iy_halo][ix];
		hm = max(T(0.0), hr + zr - zlr);

		ga = g * T(0.5);
		T fh, fu, fv, sl, sr, dt;

		dt = KurgSolver(g, delta, epsi, CFL, cm, fmv, hp, hm, up, um, fh, fu);

		if (dt < dtmax[global_idx_center]) // Output to global memory
		{
			dtmax[global_idx_center] = dt;
		}

		if (fh > T(0.0))
		{
			fv = (s_u[s_iy_halo][ix] + dx * s_dudy[s_iy_halo][ix]) * fh;
		}
		else
		{
			fv = (s_u[s_iy_current][ix] - dx * s_dudy[s_iy_current][ix]) * fh;
		}
		
		// Boundary condition handling
		T hi_boundary = hi;
		T zi_boundary = zi; // zi already incorporates s_Patm for current cell
		T hn_boundary = hn;
		T zn_boundary = zn; // zn already incorporates s_Patm for halo cell

		// Original: if ((iy == blockDim.x) && levTL < lev)
		// Interpreted as top boundary of the current block.
		if ((iy == STATIC_MAX_BLOCK_Y - 1) && levTL < lev)
		{
			int jj = BLTL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + STATIC_MAX_BLOCK_X / 2;
			int itop = memloc(halowidth, blkmemwidth, jj, 0, TL);
			hi_boundary = XEv.h[itop];
			// Patm[itop] is a global access from a neighboring block (TL).
			zi_boundary = zb[itop] + XParam.Pa2m * Patm[itop]; 
		}
		// Original: if ((iy == 0) && levBL < lev)
		// Bottom boundary of the current block.
		if ((iy == 0) && levBL < lev)
		{
			int jj = TLBL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + STATIC_MAX_BLOCK_X / 2;
			int ibc = memloc(halowidth, blkmemwidth, jj, STATIC_MAX_BLOCK_Y - 1, BL);
			hn_boundary = XEv.h[ibc];
			// Patm[ibc] is a global access from a neighboring block (BL).
			zn_boundary = zb[ibc] + XParam.Pa2m * Patm[ibc];
		}

		sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi_boundary) * (zi_boundary - zl));
		sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn_boundary) * (zn_boundary - zr));

		////Flux update (writing to global memory)
		XFlux.Fhv[global_idx_center] = fmv * fh;
		XFlux.Fqvy[global_idx_center] = fmv * (fu - sl);
		XFlux.Sv[global_idx_center] = fmv * (fu - sr);
		XFlux.Fquy[global_idx_center] = fmv * fv;
	}
	else
	{
		dtmax[global_idx_center] = T(1.0) / epsi; // Output to global memory
		XFlux.Fhv[global_idx_center] = T(0.0);
		XFlux.Fqvy[global_idx_center] = T(0.0);
		XFlux.Sv[global_idx_center] = T(0.0);
		XFlux.Fquy[global_idx_center] = T(0.0);
	}
}
template __global__ void updateKurgYATMGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float* zb, float* Patm, float* dPdy);
template __global__ void updateKurgYATMGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double* zb, double* Patm, double* dPdy);



template <class T> __global__ void AddSlopeSourceYGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* zb)
{
	// Shared memory declarations for Y-direction halo
	__shared__ T s_h[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_zs[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_dhdy[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_dzsdy[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_Fqvy[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];
	__shared__ T s_Sv[STATIC_MAX_BLOCK_Y + SHARED_MEM_HALO_WIDTH][STATIC_MAX_BLOCK_X];

	unsigned int halowidth = XParam.halowidth; // Runtime halowidth
	unsigned int blkmemwidth = blockDim.x + halowidth * 2; // Keep original for memloc
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;

	// Safety check for static shared memory bounds
	if (ix >= STATIC_MAX_BLOCK_X || iy >= STATIC_MAX_BLOCK_Y) {
		return;
	}

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
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);
	T ga = T(0.5) * g;

	T ybo = T(XParam.yo + XBlock.yo[ib]);

	// Global memory indices
	int global_idx_center = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int global_idx_bottom_halo = memloc(halowidth, blkmemwidth, ix, iy - SHARED_MEM_HALO_WIDTH, ib); // iy-1

	// Shared memory indices
	int s_iy_current = iy + SHARED_MEM_HALO_WIDTH; // iy + 1
	int s_iy_halo    = iy;                       // iy (for halo at iy=0)

	// Load data into shared memory
	s_h[s_iy_current][ix]     = XEv.h[global_idx_center];
	s_zs[s_iy_current][ix]    = XEv.zs[global_idx_center];
	s_dhdy[s_iy_current][ix]  = XGrad.dhdy[global_idx_center];
	s_dzsdy[s_iy_current][ix] = XGrad.dzsdy[global_idx_center];
	s_Fqvy[s_iy_current][ix]  = XFlux.Fqvy[global_idx_center];
	s_Sv[s_iy_current][ix]    = XFlux.Sv[global_idx_center];

	if (iy == 0) { // Load "bottom" halo (iy-1 data)
		s_h[s_iy_halo][ix]     = XEv.h[global_idx_bottom_halo];
		s_zs[s_iy_halo][ix]    = XEv.zs[global_idx_bottom_halo];
		s_dhdy[s_iy_halo][ix]  = XGrad.dhdy[global_idx_bottom_halo];
		s_dzsdy[s_iy_halo][ix] = XGrad.dzsdy[global_idx_bottom_halo];
		// Fluxes s_Fqvy and s_Sv for halo are not strictly needed as they are outputs,
		// but loading them or zeroing might prevent uninitialized reads if logic changes.
		// For now, we'll match the pattern of loading inputs.
		s_Fqvy[s_iy_halo][ix]  = XFlux.Fqvy[global_idx_bottom_halo]; 
		s_Sv[s_iy_halo][ix]    = XFlux.Sv[global_idx_bottom_halo];
	}
	
	__syncthreads();
	
	//T cm = T(1.0); // Not used in this kernel
	T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy)) : T(1.0);

	T dx, zi, zl, zn, zr, zlr, hl, hp, hr, hm;

	// Access data from shared memory
	T dhdyi   = s_dhdy[s_iy_current][ix];
	T dhdymin = s_dhdy[s_iy_halo][ix];
	T hi      = s_h[s_iy_current][ix];
	T hn      = s_h[s_iy_halo][ix];

	if (hi > eps || hn > eps)
	{
		dx = delta * T(0.5);
		zi = s_zs[s_iy_current][ix] - hi;
		zl = zi - dx * (s_dzsdy[s_iy_current][ix] - dhdyi);

		zn = s_zs[s_iy_halo][ix] - hn;
		zr = zn + dx * (s_dzsdy[s_iy_halo][ix] - dhdymin);
		
		zlr = max(zl, zr);

		hl = hi - dx * dhdyi;
		hp = max(T(0.0), hl + zl - zlr);

		hr = hn + dx * dhdymin;
		hm = max(T(0.0), hr + zr - zlr);

		// Boundary condition handling
		T hi_boundary = hi;
		T zi_boundary = zi;
		T hn_boundary = hn;
		T zn_boundary = zn;

		// Original: if ((iy == blockDim.x) && levTL < lev)
		// Interpreted as top boundary of the current block.
		if ((iy == STATIC_MAX_BLOCK_Y - 1) && levTL < lev)
		{
			int jj = BLTL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + STATIC_MAX_BLOCK_X / 2;
			int itop = memloc(halowidth, blkmemwidth, jj, 0, TL);
			hi_boundary = XEv.h[itop];
			zi_boundary = zb[itop];
		}
		// Original: if ((iy == 0) && levBL < lev)
		// Bottom boundary of the current block.
		if ((iy == 0) && levBL < lev)
		{
			int jj = TLBL == ib ? floor(ix * (T)0.5) : floor(ix * (T)0.5) + STATIC_MAX_BLOCK_X / 2;
			int ibc = memloc(halowidth, blkmemwidth, jj, STATIC_MAX_BLOCK_Y - 1, BL);
			hn_boundary = XEv.h[ibc];
			zn_boundary = zb[ibc];
		}

		T sl, sr;
		sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi_boundary) * (zi_boundary - zl));
		sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn_boundary) * (zn_boundary - zr));

		// Update flux components in shared memory
		s_Fqvy[s_iy_current][ix] = s_Fqvy[s_iy_current][ix] - fmv * sl;
		s_Sv[s_iy_current][ix]   = s_Sv[s_iy_current][ix] - fmv * sr;

		__syncthreads(); // Synchronize before writing back to global memory

		// Write results back to global memory
		XFlux.Fqvy[global_idx_center] = s_Fqvy[s_iy_current][ix];
		XFlux.Sv[global_idx_center]   = s_Sv[s_iy_current][ix];
	}
	// If (hi <= eps && hn <= eps), the original code does nothing to Fqvy or Sv,
	// so no "else" branch is needed to write back unmodified values.
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
	
	T ybo;

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

		delta = T(calcres(XParam.delta, lev));

		ybo = T(XParam.yo + XBlock.yo[ib]);

		for (int iy = 0; iy < (XParam.blkwidth + XParam.halowidth); iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);

				T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
				T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy)) : T(1.0);

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
						int jj = BLTL == ib ? ftoi(floor(ix * (T)0.5)) : ftoi(floor(ix * (T)0.5) + XParam.blkwidth / 2);
						int itop = memloc(halowidth, blkmemwidth, jj, 0, TL);
						hi = XEv.h[itop];
						zi = zb[itop];
					}
					if ((iy == 0) && levBL < lev)//(ix==16) i.e. in the bot halo
					{
						int jj = TLBL == ib ? ftoi(floor(ix * (T)0.5)) : ftoi(floor(ix * (T)0.5) + XParam.blkwidth / 2);
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

template <class T> __host__ void updateKurgYATMCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* dtmax, T* zb, T* Patm, T* dPdy)
{

	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps) + epsi;
	T delta;
	T g = T(XParam.g);
	T CFL = T(XParam.CFL);

	T ybo;

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

		delta = T(calcres(XParam.delta, lev));

		ybo = T(XParam.yo + XBlock.yo[ib]);

		for (int iy = 0; iy < (XParam.blkwidth + XParam.halowidth); iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);

				T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
				T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy)) : T(1.0);

				T dhdyi = XGrad.dhdy[i];
				T dhdymin = XGrad.dhdy[ibot];
				T hi = XEv.h[i];
				T hn = XEv.h[ibot];
				T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, ga;




				if (hi > eps || hn > eps)
				{
					hn = XEv.h[ibot];
					dx = delta * T(0.5);


					//zi = XEv.zs[i] - hi;
					//zl = zi - dx * (XGrad.dzsdy[i] - dhdyi);
					//zn = XEv.zs[ibot] - hn;
					//zr = zn + dx * (XGrad.dzsdy[ibot] - dhdymin);

					zi = XEv.zs[i] - hi + T(XParam.Pa2m) * Patm[i];
					zl = zi - dx * (XGrad.dzsdy[i] - dhdyi + T(XParam.Pa2m) * dPdy[i]);
					zn = XEv.zs[ibot] - hn + T(XParam.Pa2m) * Patm[ibot];
					zr = zn + dx * (XGrad.dzsdy[ibot] - dhdymin + T(XParam.Pa2m) * dPdy[ibot]);

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
						int jj = BLTL == ib ? ftoi(floor(ix * (T)0.5)) : ftoi(floor(ix * (T)0.5) + XParam.blkwidth / 2);
						int itop = memloc(halowidth, blkmemwidth, jj, 0, TL);
						hi = XEv.h[itop];
						zi = zb[itop] + T(XParam.Pa2m) * Patm[itop];
					}
					if ((iy == 0) && levBL < lev)//(ix==16) i.e. in the bot halo
					{
						int jj = TLBL == ib ? ftoi(floor(ix * (T)0.5)) : ftoi(floor(ix * (T)0.5) + XParam.blkwidth / 2);
						int ibc = memloc(halowidth, blkmemwidth, jj, XParam.blkwidth - 1, BL);
						hn = XEv.h[ibc];
						zn = zb[ibc] + T(XParam.Pa2m) * Patm[ibc];
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
		}
	}
}
template __host__ void updateKurgYATMCPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, FluxP<float> XFlux, float* dtmax, float* zb, float* Patm, float* dPdy);
template __host__ void updateKurgYATMCPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, FluxP<double> XFlux, double* dtmax, double* zb, double* Patm, double* dPdy);


template <class T> __host__ void AddSlopeSourceYCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, FluxP<T> XFlux, T* zb)
{
	T delta;
	T g = T(XParam.g);
	T ga = T(0.5) * g;
	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps) + epsi;

	T ybo;

	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		

		int lev = XBlock.level[ib];
		delta = T(calcres(XParam.delta, lev));
		// neighbours for source term
		int TL, BLTL, BL, TLBL, levTL, levBL;
		TL = XBlock.TopLeft[ib];
		levTL = XBlock.level[TL];
		BLTL = XBlock.BotLeft[TL];

		BL = XBlock.BotLeft[ib];
		levBL = XBlock.level[BL];
		TLBL = XBlock.TopLeft[BL];

		ybo = T(XParam.yo + XBlock.yo[ib]);



		for (int iy = 0; iy < (XParam.blkwidth + XParam.halowidth); iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);



				//T cm = T(1.0);
				T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, T(iy)) : T(1.0);

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
						int jj = BLTL == ib ? ftoi(floor(ix * (T)0.5)) : ftoi(floor(ix * (T)0.5) + XParam.blkwidth / 2);
						int itop = memloc(halowidth, blkmemwidth, jj, 0, TL);;
						hi = XEv.h[itop];
						zi = zb[itop];
					}
					if ((iy == 0) && levBL < lev)//(ix==16) i.e. in the right halo
					{
						int jj = TLBL == ib ? ftoi(floor(ix * (T)0.5)) : ftoi(floor(ix * (T)0.5) + XParam.blkwidth / 2);
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

#include "Implicit.h"

// Define SharedMemory helper to avoid alignment issues (mirroring Advection.cu)
// template<class T>
// struct SharedMemory
// {
//     __device__ inline operator T* ()
//     {
//         extern __shared__ int __smem[];
//         return (T*)__smem;
//     }

//     __device__ inline operator const T* () const
//     {
//         extern __shared__ int __smem[];
//         return (T*)__smem;
//     }
// };

// template<>
// struct SharedMemory<double>
// {
//     __device__ inline operator double* ()
//     {
//         extern __shared__ double __smem_d[];
//         return (double*)__smem_d;
//     }

//     __device__ inline operator const double* () const
//     {
//         extern __shared__ double __smem_d[];
//         return (double*)__smem_d;
//     }
// };

// //#if __CUDA_ARCH__ < 600
// template <class T> __device__ double atomicAddC(T* address, T val)
// {
//     unsigned long long int* address_as_ull = (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;
//     do {
//         assumed = old;
//         old = atomicCAS(address_as_ull, assumed,
//                         __double_as_longlong(val + __longlong_as_double(assumed)));
//     } while (assumed != old);
//     return __longlong_as_double(old);
// }
//#endif



// ---------------------------------------------------------------------------
// 1. Face coefficient assembly:  alpha_face = -(theta_H*dt)^2 * hf_face
//    (equivalent of implicit.h's alpha_eta.x[] += hf.x[]; alpha_eta.x[] *= C)
// ---------------------------------------------------------------------------
template <class T> __global__ void assemble_alpha_kernel(Param XParam, BlockP<T> XBlock,FluxIMP<T> XFlux,T dt)
{
    int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];

    int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

    

    T C = -(XParam.theta_H * dt) * (XParam.theta_H * dt);

    
    XFlux.alpha_x[i] = C * XFlux.hfx[i];   // face at west side of (ix,iy)
    XFlux.alpha_y[i] = C * XFlux.hfy[i];   // face at south side of (ix,iy)
}

// ---------------------------------------------------------------------------
// 2. RHS assembly:  rhs_eta = eta_n - dt * div(F_star)
//    (equivalent of implicit.h's rhs_eta[] = eta[] - dt*(su.x[1]-su.x[])/Delta ...)
// ---------------------------------------------------------------------------
template <class T> __global__ void assemble_rhs_kernel(Param XParam, BlockP<T> XBlock,FluxMLP<T> XFlux,T dt)
{
    int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];

    int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
    int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
    int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

    T delta = calcres(T(XParam.delta), lev);

    T cm = T(1.0);//XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);

    T divF = (XFlux.hu[iright] - XFlux.hu[i]) / (delta*cm)
                + (XFlux.hv[itop] - XFlux.hv[i]) / (delta*cm);

    XFlux.rhs_eta[i] = XFlux.eta_n[i] - dt * divF;
}

template <class T> __global__ void matvec_kernel(Param XParam, BlockP<T> XBlock,T* x, T* Ax, FluxMLP<T> XFlux)
{
    int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];
    //T delta = calcres(T(XParam.delta), lev);
    int lev = XBlock.level[ib];
    T dx = calcres(T(XParam.delta), lev);

    int id   = memloc(halowidth, blkmemwidth, ix, iy, ib);
    int idxm = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);   // west neighbour
    int idxp = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);   // east neighbour
    int idym = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);   // south neighbour
    int idyp = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);   // north neighbour

    //int idxp_face = memloc(ix + 1, iy, ib);  // alpha_x stored at west face of (ix+1,iy) == east face of (ix,iy)
    //int idyp_face = memloc(ix, iy + 1, ib);

    T axW = XFlux.alpha_x[id];         // west face coeff of (ix,iy)
    T axE = XFlux.alpha_x[idxp];  // east face coeff of (ix,iy)
    T ayS = XFlux.alpha_y[id];
    T ayN = XFlux.alpha_y[idyp];

    T lap = axW * (x[idxm] - x[id]) - axE * (x[id] - x[idxp])
               + ayS * (x[idym] - x[id]) - ayN * (x[id] - x[idyp]);
    // NOTE the sign pattern mirrors relax_hydro's a_baro(eta,0)-a_baro(eta,1)
    // construction; double-check against your face convention.

    T invdx2 = 1.0 / (dx * dx);
    Ax[id] = x[id] - XParam.g * invdx2 * lap;
}


// ---------------------------------------------------------------------------
// 4. Jacobi preconditioner setup:  diagInv = 1 / (1 + (G/dx^2)*sum |alpha_face|)
// ---------------------------------------------------------------------------
template <class T> __global__ void jacobi_diag_kernel(Param XParam, BlockP<T> XBlock,FluxMLP<T> XFlux)
{
    int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];
    int lev = XBlock.level[ib];
    T dx = calcres(T(XParam.delta), lev);

    int id   = memloc(halowidth, blkmemwidth, ix, iy, ib);
    int idxp = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
    int idyp = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

    T axW = -XFlux.alpha_x[id];          // note: alpha <= 0, so -alpha >= 0
    T axE = -XFlux.alpha_x[idxp];
    T ayS = -XFlux.alpha_y[id];
    T ayN = -XFlux.alpha_y[idyp];

    T invdx2 = 1.0 / (dx * dx);
    T diag = 1.0 + XParam.g * invdx2 * (axW + axE + ayS + ayN);

    XFlux.diagInv[id] = 1.0 / diag;
}

template <class T> __global__ void jacobi_apply_kernel(Param XParam, BlockP<T> XBlock, T*  r, T*  z, T* diagInv)
{
    int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

    int id   = memloc(halowidth, blkmemwidth, ix, iy, ib);
    z[id] = r[id] * diagInv[id];
}
template __global__ void jacobi_apply_kernel<float>(Param XParam, BlockP<float> XBlock, float*  r, float*  z, float* diagInv);
template __global__ void jacobi_apply_kernel<double>(Param XParam, BlockP<double> XBlock, double*  r, double*  z, double* diagInv);

// ---------------------------------------------------------------------------
// 5. Simple vector kernels: axpy-style updates
// ---------------------------------------------------------------------------
template <class T> __global__ void axpy_kernel(Param XParam, BlockP<T> XBlock, T* y, T* x, T a)
{
    int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

    int id   = memloc(halowidth, blkmemwidth, ix, iy, ib);
    y[id] += a * x[id];
}
template __global__ void axpy_kernel<float>(Param XParam, BlockP<float> XBlock, float* y, float* x, float a);
template __global__ void axpy_kernel<double>(Param XParam, BlockP<double> XBlock, double* y, double* x, double a);

template <class T> __global__ void xpby_kernel(Param XParam, BlockP<T> XBlock, T* p, T* z, T beta)
{
    int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

    int id   = memloc(halowidth, blkmemwidth, ix, iy, ib);
    p[id] = z[id] + beta * p[id];
}
template __global__ void xpby_kernel<float>(Param XParam, BlockP<float> XBlock, float* p, float* z, float beta);
template __global__ void xpby_kernel<double>(Param XParam, BlockP<double> XBlock, double* p, double* z, double beta);


template <class T> __global__ void x2_kernel(Param XParam, BlockP<T> XBlock, T* x, T* y)
{
    int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

    int id   = memloc(halowidth, blkmemwidth, ix, iy, ib);
    y[id] = x[id];
}
template __global__ void x2_kernel<float>(Param XParam, BlockP<float> XBlock, float* x, float* y);
template __global__ void x2_kernel<double>(Param XParam, BlockP<double> XBlock, double* x, double* y);

template <class T> inline T half_advection_dt(Param XParam,T dt)
{
    return (1.0 - XParam.theta_H) * XParam.dt;
}

// foreach_face() body, x-direction. Call once with (ix,iy) swapped / hu_y etc.
// for the y-direction pass (mirrors Basilisk's foreach_dimension()).
template <class T> __global__ void acceleration_facex(Param XParam, BlockP<T> XBlock, FluxMLP<T> XFlux, FluxIMP<T> XImp, EvolvingP<T> XEv,T dt)
{
    int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];
    int lev = XBlock.level[ib];

    int id   = memloc(halowidth, blkmemwidth, ix, iy, ib);   // this face, "i=0" position
    int idm  = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);//memloc(ix - 1, iy, ib);   // cell to the west ("i-1")
    
    T theta_H = XParam.theta_H;

    T eps = XParam.eps;
    T g = XParam.g;

    
    T delta = calcres(T(XParam.delta), lev);

    T C = -(theta_H * dt) * (theta_H * dt);
    T a_baro = g * (XImp.eta_r[idm] -  XImp.eta_r[id]) / delta;
    T ax = theta_H * a_baro;   // a_baro(eta_r,0)

    T hl = XEv.h[idm] > eps ? XEv.h[idm] : 0.0;
    T hr = XEv.h[id]  > eps ? XEv.h[id]  : 0.0;
    T uf = (hl > 0.0 || hr > 0.0) ? (hl * XEv.u[idm] + hr * XEv.u[id]) / (hl + hr) : 0.0;

    T hu_new = (1.0 - theta_H) * (XFlux.hu[id] + dt * XFlux.hfu[id] * ax)
                   + theta_H * XFlux.hfu[id] * uf;
    hu_new += dt * (theta_H * XFlux.hau[id] - XFlux.hfu[id] * ax);

    XFlux.hau[id] -= XFlux.hfu[id] * ax;
    XFlux.hu[id]  = hu_new;

    XImp.su[id]        = hu_new;          // single layer: su.x[] += hu.x[] with su.x[] init 0
    XImp.alpha_x[id] = C * XFlux.hfu[id];       // single layer: alpha_eta.x[] += hf.x[]; then *= C
}
template __global__ void acceleration_facex<float>(Param XParam, BlockP<float> XBlock, FluxMLP<float> XFlux, FluxIMP<float> XImp, EvolvingP<float> XEv,float dt);
template __global__ void acceleration_facex<double>(Param XParam, BlockP<double> XBlock, FluxMLP<double> XFlux, FluxIMP<double> XImp, EvolvingP<double> XEv,double dt);

template <class T> __global__ void acceleration_facey(Param XParam, BlockP<T> XBlock,FluxMLP<T> XFlux, FluxIMP<T> XImp, EvolvingP<T> XEv,T dt)
{
    int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];
    int lev = XBlock.level[ib];

    int id   = memloc(halowidth, blkmemwidth, ix, iy, ib);   // this face, "i=0" position
    int idm  = memloc(halowidth, blkmemwidth, ix , iy -1, ib);//memloc(ix - 1, iy, ib);   // cell to the west ("i-1")

    T theta_H = XParam.theta_H;

    T eps = XParam.eps;

    T g = XParam.g;

    
    T delta = calcres(T(XParam.delta), lev);

    T C = -(theta_H * dt) * (theta_H * dt);
    T a_baro = g * (XImp.eta_r[idm] -  XImp.eta_r[id]) / delta;
    T ax = theta_H * a_baro;   // a_baro(eta_r,0)

    T hl = XEv.h[idm] > eps ? XEv.h[idm] : T(0.0);
    T hr = XEv.h[id]  > eps ? XEv.h[id]  : T(0.0);
    T uf = (hl > T(0.0) || hr > T(0.0)) ? (hl * XEv.v[idm] + hr * XEv.v[id]) / (hl + hr) : T(0.0);

    T hu_new = (1.0 - theta_H) * (XFlux.hv[id] + dt * XFlux.hfv[id] * ax)
                   + theta_H * XFlux.hfv[id] * uf;
    hu_new += dt * (theta_H * XFlux.hav[id] - XFlux.hfv[id] * ax);

    XFlux.hav[id] -= XFlux.hfv[id] * ax;
    XFlux.hv[id]  = hu_new;

    XImp.sv[id]        = hu_new;          // single layer: su.x[] += hu.x[] with su.x[] init 0
    XImp.alpha_y[id] = C * XFlux.hfv[id];       // single layer: alpha_eta.x[] += hf.x[]; then *= C
}
template __global__ void acceleration_facey<float>(Param XParam, BlockP<float> XBlock,FluxMLP<float> XFlux, FluxIMP<float> XImp, EvolvingP<float> XEv,float dt);
template __global__ void acceleration_facey<double>(Param XParam, BlockP<double> XBlock,FluxMLP<double> XFlux, FluxIMP<double> XImp, EvolvingP<double> XEv,double dt);


template <class T> __global__ void acceleration_rhs(Param XParam, BlockP<T> XBlock, FluxIMP<T> XImp, T dt)
{
    int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];
    int lev = XBlock.level[ib];
    T delta = calcres(T(XParam.delta), lev);

    int id   = memloc(halowidth, blkmemwidth, ix, iy, ib);//memloc(ix,     iy,     ib);
    int idxp = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);//memloc(ix + 1, iy,     ib);   // su.x[1]
    int idyp = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);//memloc(ix,     iy + 1, ib);   // su.y[1]

    T rhs = XParam.rigid ? T(0.0) : XImp.eta_r[id];
    rhs -= dt * (XImp.su[idxp] - XImp.su[id]) / delta;
    rhs -= dt * (XImp.sv[idyp] - XImp.sv[id]) / delta;
    XImp.rhs_eta[id] = rhs;
}
template __global__ void acceleration_rhs<float>(Param XParam, BlockP<float> XBlock, FluxIMP<float> XImp, float dt);
template __global__ void acceleration_rhs<double>(Param XParam, BlockP<double> XBlock, FluxIMP<double> XImp, double dt);

template <class T> __global__ void matvec_facefieldx(Param XParam, BlockP<T> XBlock,T* eta,T* g_x,T*alpha_x)
{
   int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];
    int lev = XBlock.level[ib];

    T delta = calcres(T(XParam.delta), lev);

    int id  = memloc(halowidth, blkmemwidth, ix, iy, ib);//memloc(ix,     iy, ib);
    int idm = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);//memloc(ix - 1, iy, ib);

    T a_baro = XParam.g * (eta[idm] - eta[id]) / delta;

    g_x[id] = alpha_x[id] * a_baro;   // a_baro(eta,0)
}
template __global__ void matvec_facefieldx<float>(Param XParam, BlockP<float> XBlock,float* eta,float* g_x,float*alpha_x);
template __global__ void matvec_facefieldx<double>(Param XParam, BlockP<double> XBlock,double* eta,double* g_x,double*alpha_x);

template <class T> __global__ void matvec_facefieldy(Param XParam, BlockP<T> XBlock,T* eta,T* g_y,T*alpha_y)
{
   int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];
    int lev = XBlock.level[ib];
    T delta = calcres(T(XParam.delta), lev);

    int id  = memloc(halowidth, blkmemwidth, ix, iy, ib);//memloc(ix,     iy, ib);
    int idm = memloc(halowidth, blkmemwidth, ix, iy-1, ib);//memloc(ix - 1, iy, ib);

    T a_baro = XParam.g * (eta[idm] - eta[id]) / delta;

    g_y[id] = alpha_y[id] * a_baro;   // a_baro(eta,0)
}
template __global__ void matvec_facefieldy<float>(Param XParam, BlockP<float> XBlock,float* eta,float* g_y,float*alpha_y);
template __global__ void matvec_facefieldy<double>(Param XParam, BlockP<double> XBlock,double* eta,double* g_y,double*alpha_y);

template <class T> __global__ void matvec_apply(Param XParam, BlockP<T> XBlock,T* eta,T* Aeta, T* g_x, T* g_y)
{
     int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];
    int lev = XBlock.level[ib];
    T delta = calcres(T(XParam.delta), lev);

    int id  = memloc(halowidth, blkmemwidth, ix, iy, ib);
    int idxp = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);//memloc(ix + 1, iy,     ib);
    int idyp = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);//memloc(ix,     iy + 1, ib);

    T Ax = XParam.rigid ? 0.0 : eta[id];
    Ax -= (g_x[idxp] - g_x[id]) / delta;
    Ax -= (g_y[idyp] - g_y[id]) / delta;
    Aeta[id] = Ax;
}
template __global__ void matvec_apply<float>(Param XParam, BlockP<float> XBlock,float* eta,float* Aeta, float* g_x, float* g_y);
template __global__ void matvec_apply<double>(Param XParam, BlockP<double> XBlock,double* eta,double* Aeta, double* g_x, double* g_y);


// ----------------------------------------------------------------------
// Jacobi preconditioner: diag(A) obtained by differentiating A(eta)[]
// w.r.t. eta[] itself (this is what Basilisk's diagonalize(eta) macro
// does automatically inside relax_hydro -- here it's written out by hand):
//
//   d(A)/d(eta[]) = 1 - (G/Delta^2)*(alpha_eta.x[0]+alpha_eta.x[1]
//                                    +alpha_eta.y[0]+alpha_eta.y[1])
// ----------------------------------------------------------------------
template <class T> __global__ void jacobi_diag(Param XParam, BlockP<T> XBlock, FluxIMP<T> XImp)
{
    int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

    int lev = XBlock.level[ib];
    T delta = calcres(T(XParam.delta), lev);

    T g = XParam.g;

    int id  = memloc(halowidth, blkmemwidth, ix, iy, ib);
    int idxp = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);//memloc(ix + 1, iy,     ib);
    int idyp = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);//memloc(ix,     iy + 1, ib);


    T sumAlpha = XImp.alpha_x[id] + XImp.alpha_x[idxp]
                     + XImp.alpha_y[id] + XImp.alpha_y[idyp];
    // alpha_eta <= 0, so this correctly increases the diagonal:
    T diag = (XParam.rigid ? 0.0 : 1.0) - (g / (delta * delta)) * sumAlpha;
    XImp.diagInv[id] = 1.0 / diag;
}
template __global__ void jacobi_diag<float>(Param XParam, BlockP<float> XBlock, FluxIMP<float> XImp);
template __global__ void jacobi_diag<double>(Param XParam, BlockP<double> XBlock, FluxIMP<double> XImp);

// foreach_face() flux-reconstruction block after the solve.
template <class T> __global__ void pressure_flux_reconstruction_facex(Param XParam, BlockP<T> XBlock,FluxMLP<T> XFlux, FluxIMP<T> XImp, EvolvingP<T> XEv,T dt)
{
     int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

    int lev = XBlock.level[ib];
    T delta = calcres(T(XParam.delta), lev);

    int id  = memloc(halowidth, blkmemwidth, ix, iy, ib);
    int idm = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);//memloc(ix - 1, iy, ib);

    T dry = XParam.eps;

    T abaro = XParam.g * (XImp.eta_r[idm] - XImp.eta_r[id]) / delta;

    T ax = XParam.theta_H * abaro;   // a_baro(eta_r,0)
    T newhau = XFlux.hau[id] + XFlux.hfu[id] * ax;

    T hl = XEv.h[idm] > dry ? XEv.h[idm] : 0.0;
    T hr = XEv.h[id]  > dry ? XEv.h[id]  : 0.0;
    T uf = (hl > 0.0 || hr > 0.0) ? (hl * XEv.u[idm] + hr * XEv.u[id]) / (hl + hr) : 0.0;

    // NOTE: what's stored here is theta_H*(hu)^{n+1}, not the full flux --
    // this is intentional (see implicit.h comment) because half_advection
    // already applied (1-theta_H)*dt worth of advection, so hydro.h's own
    // pressure event / advect() call only needs the remaining theta_H*dt.
    XFlux.hu[id] = XParam.theta_H * (XFlux.hfu[id] * uf + dt * newhau) - dt * newhau;
    XFlux.hau[id] = newhau;
}
template __global__ void pressure_flux_reconstruction_facex<float>(Param XParam, BlockP<float> XBlock,FluxMLP<float> XFlux, FluxIMP<float> XImp, EvolvingP<float> XEv,float dt);
template __global__ void pressure_flux_reconstruction_facex<double>(Param XParam, BlockP<double> XBlock,FluxMLP<double> XFlux, FluxIMP<double> XImp, EvolvingP<double> XEv,double dt);


template <class T> __global__ void pressure_flux_reconstruction_facey(Param XParam, BlockP<T> XBlock,FluxMLP<T> XFlux, FluxIMP<T> XImp, EvolvingP<T> XEv,T dt)
{
     int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.y + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

    int lev = XBlock.level[ib];
    T delta = calcres(T(XParam.delta), lev);

    int id  = memloc(halowidth, blkmemwidth, ix, iy, ib);
    int idm = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);//memloc(ix - 1, iy, ib);

    T dry = XParam.eps;

    T abaro = XParam.g * (XImp.eta_r[idm] - XImp.eta_r[id]) / delta;

    T ax = XParam.theta_H * abaro;   // a_baro(eta_r,0)
    //XFlux.hav[id] += XFlux.hfv[id] * ax;
    T newhav = XFlux.hav[id] + XFlux.hfv[id] * ax;

    T hl = XEv.h[idm] > dry ? XEv.h[idm] : 0.0;
    T hr = XEv.h[id]  > dry ? XEv.h[id]  : 0.0;
    T uf = (hl > 0.0 || hr > 0.0) ? (hl * XEv.v[idm] + hr * XEv.v[id]) / (hl + hr) : 0.0;

    // NOTE: what's stored here is theta_H*(hu)^{n+1}, not the full flux --
    // this is intentional (see implicit.h comment) because half_advection
    // already applied (1-theta_H)*dt worth of advection, so hydro.h's own
    // pressure event / advect() call only needs the remaining theta_H*dt.
    XFlux.hv[id] = XParam.theta_H * (XFlux.hfv[id] * uf + dt * newhav) - dt * newhav;
    XFlux.hav[id] = newhav;
}
template __global__ void pressure_flux_reconstruction_facey<float>(Param XParam, BlockP<float> XBlock,FluxMLP<float> XFlux, FluxIMP<float> XImp, EvolvingP<float> XEv,float dt);
template __global__ void pressure_flux_reconstruction_facey<double>(Param XParam, BlockP<double> XBlock,FluxMLP<double> XFlux, FluxIMP<double> XImp, EvolvingP<double> XEv,double dt);


// /**
//  * @brief Multigrid relaxation function (Red-Black Gauss-Seidel)
//  */
// template <class T>
// __global__ void relaxHydro(Param XParam,T* eta, T* rhs)
// {

//     int halowidth = XParam.halowidth;
// 	int blkmemwidth = blockDim.y + halowidth * 2;
// 	//unsigned int blksize = blkmemwidth * blkmemwidth;
// 	int ix = threadIdx.x;
// 	int iy = threadIdx.y;
// 	int ibl = blockIdx.x;
// 	int ib = XBlock.active[ibl];

// 	int lev = XBlock.level[ib];

//     int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

//     T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
// 	T eps = T(XParam.eps);// +epsi;
// 	T dry = eps;
// 	T delta = calcres(T(XParam.delta), lev);
// 	T g = T(XParam.g);
// 	T CFL = T(XParam.CFL);

//     T CFL_H=1e40;
//     bool rigid = false;

//     T cmu = T(1.0);
// 	T cmv = T(1.0);

// 	if (XParam.spherical)
// 	{
// 		T ybo = T(XParam.yo + XBlock.yo[ib]);

// 		cmu = calcCM(T(XParam.Radius), delta, ybo, iy);
// 		cmv = calcCM(T(XParam.Radius), delta, ybo, iy);

// 	}

//     double d = rigid ? 0.0 : - cmu*Delta;
//     double n = - cmu*Delta*rhs_eta[i];

//     eta[i]=T(0.0);


// }

// template <class T>
// __global__ void residualHydro()
// {
//     int halowidth = XParam.halowidth;
// 	int blkmemwidth = blockDim.y + halowidth * 2;
// 	//unsigned int blksize = blkmemwidth * blkmemwidth;
// 	int ix = threadIdx.x;
// 	int iy = threadIdx.y;
// 	int ibl = blockIdx.x;
// 	int ib = XBlock.active[ibl];

// 	int lev = XBlock.level[ib];

//     int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
// }


// /**
//  * @brief Multigrid relaxation function (Red-Black Gauss-Seidel)
//  */
// template <class T>
// __global__ void relax_implicit_eta(
//     T* eta, T* rhs, T* alpha_x, T* alpha_y, Param XParam, BlockP<T> XBlock,
//     int parity, T* diff_sum)
// {
//     T* s_diff = SharedMemory<T>();

//     int halowidth = XParam.halowidth;
//     int blkmemwidth = blockDim.x + halowidth * 2;
//     int ix = threadIdx.x;
//     int iy = threadIdx.y;
//     int tid = iy * blockDim.x + ix;

//     int ibl = blockIdx.x;
//     int ib = XBlock.active[ibl];
//     int lev = XBlock.level[ib];
//     T delta = calcres(T(XParam.delta), lev);

//     int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

//     T diff = T(0.0);

//     if ((ix + iy) % 2 == parity) {
//         int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
//         int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
//         int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
//         int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

//         T cmu = T(1.0);
//         T cmv = T(1.0);
//         if (XParam.spherical) {
//             T ybo = T(XParam.yo + XBlock.yo[ib]);
//             cmu = calcCM(T(XParam.Radius), delta, ybo, iy);
//             cmv = calcCM(T(XParam.Radius), delta, ybo, iy);
//         }

//         T d2 = delta * delta;
//         T n = rhs[i];
//         T d = T(1.0); // beta = 1

//         n -= (alpha_x[iright] * eta[iright] + alpha_x[i] * eta[ileft]) / (d2 * cmu * cmu);
//         n -= (alpha_y[itop] * eta[itop] + alpha_y[i] * eta[ibot]) / (d2 * cmv * cmv);

//         d -= (alpha_x[iright] + alpha_x[i]) / (d2 * cmu * cmu);
//         d -= (alpha_y[itop] + alpha_y[i]) / (d2 * cmv * cmv);

//         T old_val = eta[i];
//         T new_val = n / d;
//         eta[i] = new_val;
//         diff = (T)fabs(new_val - old_val);
//     }

//     // Shared memory reduction for convergence monitoring
//     s_diff[tid] = diff;
//     __syncthreads();

//     for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             s_diff[tid] += s_diff[tid + s];
//         }
//         __syncthreads();
//     }

//     if (tid == 0) {
//         atomicAddC(diff_sum, s_diff[0]);
//     }
// }

// template <class T>
// __global__ void compute_implicit_face_data(
//     T* alpha_x, T* alpha_y, T* su_x, T* su_y, Param XParam, BlockP<T> XBlock,
//     EvolvingP<T> XEv, FluxMLP<T> XFlux, T dt)
// {
//     int halowidth = XParam.halowidth;
//     int blkmemwidth = blockDim.x + halowidth * 2;
//     int ix = threadIdx.x;
//     int iy = threadIdx.y;
//     int ibl = blockIdx.x;
//     int ib = XBlock.active[ibl];
//     int lev = XBlock.level[ib];
//     T delta = calcres(T(XParam.delta), lev);
//     T g = T(XParam.g);
//     T theta = T(XParam.theta_imp);
//     T C = - theta * theta * dt * dt * g;

//     int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

//     alpha_x[i] = XFlux.hfu[i] * C;
//     alpha_y[i] = XFlux.hfv[i] * C;

//     su_x[i] = XFlux.hu[i] + theta * (T(1.0) - theta) * dt * XFlux.hau[i];
//     su_y[i] = XFlux.hv[i] + theta * (T(1.0) - theta) * dt * XFlux.hav[i];
// }

// template <class T>
// __global__ void compute_implicit_rhs(
//     T* rhs, T* su_x, T* su_y, Param XParam, BlockP<T> XBlock,
//     EvolvingP<T> XEv, T dt)
// {
//     int halowidth = XParam.halowidth;
//     int blkmemwidth = blockDim.x + halowidth * 2;
//     int ix = threadIdx.x;
//     int iy = threadIdx.y;
//     int ibl = blockIdx.x;
//     int ib = XBlock.active[ibl];
//     int lev = XBlock.level[ib];
//     T delta = calcres(T(XParam.delta), lev);

//     int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
//     int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
//     int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

//     T cmu = T(1.0);
//     T cmv = T(1.0);
//     if (XParam.spherical) {
//         T ybo = T(XParam.yo + XBlock.yo[ib]);
//         cmu = calcCM(T(XParam.Radius), delta, ybo, iy);
//         cmv = calcCM(T(XParam.Radius), delta, ybo, iy);
//     }

//     rhs[i] = XEv.zs[i] - dt * ( (su_x[iright] - su_x[i]) / (delta * cmu) + (su_y[itop] - su_y[i]) / (delta * cmv) );
// }

// template <class T>
// __global__ void project_implicit_velocities(
//     EvolvingP<T> XEv, T* eta_new, T* zb, Param XParam, BlockP<T> XBlock, T dt)
// {
//     int halowidth = XParam.halowidth;
//     int blkmemwidth = blockDim.x + halowidth * 2;
//     int ix = threadIdx.x;
//     int iy = threadIdx.y;
//     int ibl = blockIdx.x;
//     int ib = XBlock.active[ibl];
//     int lev = XBlock.level[ib];
//     T delta = calcres(T(XParam.delta), lev);
//     T theta = T(XParam.theta_imp);
//     T g = T(XParam.g);
//     T eps = T(XParam.eps);

//     int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
//     int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
//     int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
//     int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
//     int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

//     T cmu = T(1.0);
//     T cmv = T(1.0);
//     if (XParam.spherical) {
//         T ybo = T(XParam.yo + XBlock.yo[ib]);
//         cmu = calcCM(T(XParam.Radius), delta, ybo, iy);
//         cmv = calcCM(T(XParam.Radius), delta, ybo, iy);
//     }

//     T detadx = (eta_new[iright] - eta_new[ileft]) / (T(2.0) * delta * cmu);
//     T detady = (eta_new[itop] - eta_new[ibot]) / (T(2.0) * delta * cmv);

//     XEv.u[i] -= theta * dt * g * detadx;
//     XEv.v[i] -= theta * dt * g * detady;

//     XEv.h[i] = max(T(0.0), eta_new[i] - zb[i]);
//     XEv.zs[i] = zb[i] + XEv.h[i];

//     if (XEv.h[i] < eps) {
//         XEv.u[i] = T(0.0);
//         XEv.v[i] = T(0.0);
//     }
// }

// template <class T>
// void solve_implicit_barotropic(Param& XParam, Loop<T>& XLoop, Model<T>& XModel)
// {
//     dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
//     dim3 gridDim(XParam.nblk, 1, 1);
//     T dt = T(XLoop.dt);

//     compute_implicit_face_data<<<gridDim, blockDim>>>(
//         XModel.fluxml.alpha_x, XModel.fluxml.alpha_y, XModel.fluxml.su_x, XModel.fluxml.su_y,
//         XParam, XModel.blocks, XModel.evolv, XModel.fluxml, dt
//     );
//     CUDA_CHECK(cudaDeviceSynchronize());
//     fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.alpha_x);
//     fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.alpha_y);
//     fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.su_x);
//     fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.su_y);

//     compute_implicit_rhs<<<gridDim, blockDim>>>(
//         XModel.rhs, XModel.fluxml.su_x, XModel.fluxml.su_y, XParam, XModel.blocks, XModel.evolv, dt
//     );
//     CUDA_CHECK(cudaDeviceSynchronize());

//     T h_diff_sum;
//     T* d_diff_sum = XModel.time.arrmax;
//     int smemSize = blockDim.x * blockDim.y * sizeof(T);

//     for (int iter = 0; iter < XParam.mg_max_iter; ++iter) {
//         // Correct reset of convergence buffer
//         reset_var<<<gridDim, blockDim>>>(XParam.halowidth, XModel.blocks.active, T(0.0), d_diff_sum);
//         CUDA_CHECK(cudaDeviceSynchronize());

//         relax_implicit_eta<<<gridDim, blockDim, smemSize>>>(XModel.evolv.zs, XModel.rhs, XModel.fluxml.alpha_x, XModel.fluxml.alpha_y, XParam, XModel.blocks, 0, d_diff_sum);
//         CUDA_CHECK(cudaDeviceSynchronize());
//         fillHaloGPU(XParam, XModel.blocks, XModel.evolv.zs);
//         relax_implicit_eta<<<gridDim, blockDim, smemSize>>>(XModel.evolv.zs, XModel.rhs, XModel.fluxml.alpha_x, XModel.fluxml.alpha_y, XParam, XModel.blocks, 1, d_diff_sum);
//         CUDA_CHECK(cudaDeviceSynchronize());
//         fillHaloGPU(XParam, XModel.blocks, XModel.evolv.zs);

//         CUDA_CHECK(cudaMemcpy(&h_diff_sum, d_diff_sum, sizeof(T), cudaMemcpyDeviceToHost));
//         if (h_diff_sum / (XParam.nblk * XParam.blkwidth * XParam.blkwidth) < XParam.mg_tol) break;
//     }

//     project_implicit_velocities<<<gridDim, blockDim>>>(
//         XModel.evolv, XModel.evolv.zs, XModel.zb, XParam, XModel.blocks, dt
//     );
//     CUDA_CHECK(cudaDeviceSynchronize());
// }

// template void solve_implicit_barotropic<float>(Param& XParam, Loop<float>& XLoop, Model<float>& XModel);
// template void solve_implicit_barotropic<double>(Param& XParam, Loop<double>& XLoop, Model<double>& XModel);




#include "Implicit.h"

// Define SharedMemory helper to avoid alignment issues (mirroring Advection.cu)
template<class T>
struct SharedMemory
{
    __device__ inline operator T* ()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T* () const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

template<>
struct SharedMemory<double>
{
    __device__ inline operator double* ()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double* () const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

//#if __CUDA_ARCH__ < 600
template <class T> __device__ double atomicAddC(T* address, T val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
//#endif

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


/**
 * @brief Multigrid relaxation function (Red-Black Gauss-Seidel)
 */
template <class T>
__global__ void relax_implicit_eta(
    T* eta, T* rhs, T* alpha_x, T* alpha_y, Param XParam, BlockP<T> XBlock,
    int parity, T* diff_sum)
{
    T* s_diff = SharedMemory<T>();

    int halowidth = XParam.halowidth;
    int blkmemwidth = blockDim.x + halowidth * 2;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int tid = iy * blockDim.x + ix;

    int ibl = blockIdx.x;
    int ib = XBlock.active[ibl];
    int lev = XBlock.level[ib];
    T delta = calcres(T(XParam.delta), lev);

    int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

    T diff = T(0.0);

    if ((ix + iy) % 2 == parity) {
        int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
        int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
        int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
        int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

        T cmu = T(1.0);
        T cmv = T(1.0);
        if (XParam.spherical) {
            T ybo = T(XParam.yo + XBlock.yo[ib]);
            cmu = calcCM(T(XParam.Radius), delta, ybo, iy);
            cmv = calcCM(T(XParam.Radius), delta, ybo, iy);
        }

        T d2 = delta * delta;
        T n = rhs[i];
        T d = T(1.0); // beta = 1

        n -= (alpha_x[iright] * eta[iright] + alpha_x[i] * eta[ileft]) / (d2 * cmu * cmu);
        n -= (alpha_y[itop] * eta[itop] + alpha_y[i] * eta[ibot]) / (d2 * cmv * cmv);

        d -= (alpha_x[iright] + alpha_x[i]) / (d2 * cmu * cmu);
        d -= (alpha_y[itop] + alpha_y[i]) / (d2 * cmv * cmv);

        T old_val = eta[i];
        T new_val = n / d;
        eta[i] = new_val;
        diff = (T)fabs(new_val - old_val);
    }

    // Shared memory reduction for convergence monitoring
    s_diff[tid] = diff;
    __syncthreads();

    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_diff[tid] += s_diff[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAddC(diff_sum, s_diff[0]);
    }
}

template <class T>
__global__ void compute_implicit_face_data(
    T* alpha_x, T* alpha_y, T* su_x, T* su_y, Param XParam, BlockP<T> XBlock,
    EvolvingP<T> XEv, FluxMLP<T> XFlux, T dt)
{
    int halowidth = XParam.halowidth;
    int blkmemwidth = blockDim.x + halowidth * 2;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ibl = blockIdx.x;
    int ib = XBlock.active[ibl];
    int lev = XBlock.level[ib];
    T delta = calcres(T(XParam.delta), lev);
    T g = T(XParam.g);
    T theta = T(XParam.theta_imp);
    T C = - theta * theta * dt * dt * g;

    int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

    alpha_x[i] = XFlux.hfu[i] * C;
    alpha_y[i] = XFlux.hfv[i] * C;

    su_x[i] = XFlux.hu[i] + theta * (T(1.0) - theta) * dt * XFlux.hau[i];
    su_y[i] = XFlux.hv[i] + theta * (T(1.0) - theta) * dt * XFlux.hav[i];
}

template <class T>
__global__ void compute_implicit_rhs(
    T* rhs, T* su_x, T* su_y, Param XParam, BlockP<T> XBlock,
    EvolvingP<T> XEv, T dt)
{
    int halowidth = XParam.halowidth;
    int blkmemwidth = blockDim.x + halowidth * 2;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ibl = blockIdx.x;
    int ib = XBlock.active[ibl];
    int lev = XBlock.level[ib];
    T delta = calcres(T(XParam.delta), lev);

    int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
    int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
    int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

    T cmu = T(1.0);
    T cmv = T(1.0);
    if (XParam.spherical) {
        T ybo = T(XParam.yo + XBlock.yo[ib]);
        cmu = calcCM(T(XParam.Radius), delta, ybo, iy);
        cmv = calcCM(T(XParam.Radius), delta, ybo, iy);
    }

    rhs[i] = XEv.zs[i] - dt * ( (su_x[iright] - su_x[i]) / (delta * cmu) + (su_y[itop] - su_y[i]) / (delta * cmv) );
}

template <class T>
__global__ void project_implicit_velocities(
    EvolvingP<T> XEv, T* eta_new, T* zb, Param XParam, BlockP<T> XBlock, T dt)
{
    int halowidth = XParam.halowidth;
    int blkmemwidth = blockDim.x + halowidth * 2;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ibl = blockIdx.x;
    int ib = XBlock.active[ibl];
    int lev = XBlock.level[ib];
    T delta = calcres(T(XParam.delta), lev);
    T theta = T(XParam.theta_imp);
    T g = T(XParam.g);
    T eps = T(XParam.eps);

    int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
    int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
    int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
    int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
    int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

    T cmu = T(1.0);
    T cmv = T(1.0);
    if (XParam.spherical) {
        T ybo = T(XParam.yo + XBlock.yo[ib]);
        cmu = calcCM(T(XParam.Radius), delta, ybo, iy);
        cmv = calcCM(T(XParam.Radius), delta, ybo, iy);
    }

    T detadx = (eta_new[iright] - eta_new[ileft]) / (T(2.0) * delta * cmu);
    T detady = (eta_new[itop] - eta_new[ibot]) / (T(2.0) * delta * cmv);

    XEv.u[i] -= theta * dt * g * detadx;
    XEv.v[i] -= theta * dt * g * detady;

    XEv.h[i] = max(T(0.0), eta_new[i] - zb[i]);
    XEv.zs[i] = zb[i] + XEv.h[i];

    if (XEv.h[i] < eps) {
        XEv.u[i] = T(0.0);
        XEv.v[i] = T(0.0);
    }
}

template <class T>
void solve_implicit_barotropic(Param& XParam, Loop<T>& XLoop, Model<T>& XModel)
{
    dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
    dim3 gridDim(XParam.nblk, 1, 1);
    T dt = T(XLoop.dt);

    compute_implicit_face_data<<<gridDim, blockDim>>>(
        XModel.fluxml.alpha_x, XModel.fluxml.alpha_y, XModel.fluxml.su_x, XModel.fluxml.su_y,
        XParam, XModel.blocks, XModel.evolv, XModel.fluxml, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.alpha_x);
    fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.alpha_y);
    fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.su_x);
    fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.su_y);

    compute_implicit_rhs<<<gridDim, blockDim>>>(
        XModel.rhs, XModel.fluxml.su_x, XModel.fluxml.su_y, XParam, XModel.blocks, XModel.evolv, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    T h_diff_sum;
    T* d_diff_sum = XModel.time.arrmax;
    int smemSize = blockDim.x * blockDim.y * sizeof(T);

    for (int iter = 0; iter < XParam.mg_max_iter; ++iter) {
        // Correct reset of convergence buffer
        reset_var<<<gridDim, blockDim>>>(XParam.halowidth, XModel.blocks.active, T(0.0), d_diff_sum);
        CUDA_CHECK(cudaDeviceSynchronize());

        relax_implicit_eta<<<gridDim, blockDim, smemSize>>>(XModel.evolv.zs, XModel.rhs, XModel.fluxml.alpha_x, XModel.fluxml.alpha_y, XParam, XModel.blocks, 0, d_diff_sum);
        CUDA_CHECK(cudaDeviceSynchronize());
        fillHaloGPU(XParam, XModel.blocks, XModel.evolv.zs);
        relax_implicit_eta<<<gridDim, blockDim, smemSize>>>(XModel.evolv.zs, XModel.rhs, XModel.fluxml.alpha_x, XModel.fluxml.alpha_y, XParam, XModel.blocks, 1, d_diff_sum);
        CUDA_CHECK(cudaDeviceSynchronize());
        fillHaloGPU(XParam, XModel.blocks, XModel.evolv.zs);

        CUDA_CHECK(cudaMemcpy(&h_diff_sum, d_diff_sum, sizeof(T), cudaMemcpyDeviceToHost));
        if (h_diff_sum / (XParam.nblk * XParam.blkwidth * XParam.blkwidth) < XParam.mg_tol) break;
    }

    project_implicit_velocities<<<gridDim, blockDim>>>(
        XModel.evolv, XModel.evolv.zs, XModel.zb, XParam, XModel.blocks, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void solve_implicit_barotropic<float>(Param& XParam, Loop<float>& XLoop, Model<float>& XModel);
template void solve_implicit_barotropic<double>(Param& XParam, Loop<double>& XLoop, Model<double>& XModel);

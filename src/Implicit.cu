#include "Implicit.h"
#include "Multilayer.h"
#include "FlowGPU.h"
#include "Halo.h"

// #if __CUDA_ARCH__ < 600
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
// #endif

/**
 * @brief Setup the linear system for the semi-implicit barotropic solver.
 */
template <class T>
__global__ void setup_implicit_barotropic_system(
    T* rhs, T* diag, Param XParam, BlockP<T> XBlock,
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

    T divU = (XFlux.hu[iright] - XFlux.hu[i]) / (delta * cmu) +
             (XFlux.hv[itop] - XFlux.hv[i]) / (delta * cmv);

    T divG = (XFlux.hau[i] - XFlux.hau[iright]) / (delta * cmu) +
             (XFlux.hav[i] - XFlux.hav[itop]) / (delta * cmv);

    rhs[i] = XEv.zs[i] - dt * divU + theta * (T(1.0) - theta) * dt * dt * divG;

    T coeff_x = theta * theta * dt * dt * g / (delta * delta * cmu * cmu);
    T coeff_y = theta * theta * dt * dt * g / (delta * delta * cmv * cmv);

    diag[i] = T(1.0) + coeff_x * (XFlux.hfu[iright] + XFlux.hfu[i]) +
                       coeff_y * (XFlux.hfv[itop] + XFlux.hfv[i]);
}

/**
 * @brief Iterative solver step (Jacobi relaxation) with optimized reduction.
 */
template <class T>
__global__ void solve_implicit_eta_iteration(
    T* dst_eta, T* src_eta, T* rhs, T* diag, Param XParam, BlockP<T> XBlock,
    FluxMLP<T> XFlux, T dt, T* diff_sum)
{
    // Use fixed size shared memory for max possible block size (e.g., 16x16 = 256)
    __shared__ T s_diff[256];

    int halowidth = XParam.halowidth;
    int blkmemwidth = blockDim.x + halowidth * 2;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int tid = iy * blockDim.x + ix;
    int ibl = blockIdx.x;
    int ib = XBlock.active[ibl];
    int lev = XBlock.level[ib];

    T delta = calcres(T(XParam.delta), lev);
    T g = T(XParam.g);
    T theta = T(XParam.theta_imp);

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

    T coeff_x = theta * theta * dt * dt * g / (delta * delta * cmu * cmu);
    T coeff_y = theta * theta * dt * dt * g / (delta * delta * cmv * cmv);

    T sum_off = coeff_x * (XFlux.hfu[iright] * src_eta[iright] + XFlux.hfu[i] * src_eta[ileft]) +
                coeff_y * (XFlux.hfv[itop] * src_eta[itop] + XFlux.hfv[i] * src_eta[ibot]);

    T new_val = (rhs[i] + sum_off) / diag[i];
    T diff = fabs(new_val - src_eta[i]);

    dst_eta[i] = new_val;

    s_diff[tid] = diff;
    __syncthreads();

    // Reduction in shared memory
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

/**
 * @brief Project the converged free surface back to velocities and layer thicknesses.
 */
template <class T>
__global__ void project_implicit_velocities(
    EvolvingP<T> XEv, T* eta_new, T* eta_old, T* zb, Param XParam, BlockP<T> XBlock, FluxMLP<T> XFlux, T dt)
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

    T detadx_new = (eta_new[iright] - eta_new[ileft]) / (T(2.0) * delta * cmu);
    T detady_new = (eta_new[itop] - eta_new[ibot]) / (T(2.0) * delta * cmv);

    T detadx_old = (eta_old[iright] - eta_old[ileft]) / (T(2.0) * delta * cmu);
    T detady_old = (eta_old[itop] - eta_old[ibot]) / (T(2.0) * delta * cmv);

    XEv.u[i] -= dt * g * ((T(1.0) - theta) * detadx_old + theta * detadx_new);
    XEv.v[i] -= dt * g * ((T(1.0) - theta) * detady_old + theta * detady_new);

    XEv.h[i] = max(T(0.0), eta_new[i] - zb[i]);
    XEv.zs[i] = eta_new[i];

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

    CUDA_CHECK(cudaMemcpy(XModel.evolv_o.zs, XModel.evolv.zs,
                         XParam.nblkmem * XParam.blksize * sizeof(T),
                         cudaMemcpyDeviceToDevice));

    setup_implicit_barotropic_system<<<gridDim, blockDim>>>(
        XModel.rhs, XModel.diag, XParam, XModel.blocks,
        XModel.evolv, XModel.fluxml, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    T* src_eta = XModel.evolv.zs;
    T* dst_eta = XModel.zs_scratch;

    T h_diff_sum;
    T* d_diff_sum = XModel.time.arrmax;

    for (int iter = 0; iter < XParam.mg_max_iter; ++iter) {
        reset_var<<<1, 1>>>(0, XModel.blocks.active, T(0.0), d_diff_sum);

        solve_implicit_eta_iteration<<<gridDim, blockDim>>>(
            dst_eta, src_eta, XModel.rhs, XModel.diag, XParam, XModel.blocks,
            XModel.fluxml, dt, d_diff_sum
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_diff_sum, d_diff_sum, sizeof(T), cudaMemcpyDeviceToHost));
        if (h_diff_sum / (XParam.nblk * XParam.blkwidth * XParam.blkwidth) < XParam.mg_tol) {
            src_eta = dst_eta;
            break;
        }

        T* temp = src_eta;
        src_eta = dst_eta;
        dst_eta = temp;

        fillHaloGPU(XParam, XModel.blocks, src_eta);
    }

    if (src_eta != XModel.evolv.zs) {
        CUDA_CHECK(cudaMemcpy(XModel.evolv.zs, src_eta,
                             XParam.nblkmem * XParam.blksize * sizeof(T),
                             cudaMemcpyDeviceToDevice));
    }

    project_implicit_velocities<<<gridDim, blockDim>>>(
        XModel.evolv, XModel.evolv.zs, XModel.evolv_o.zs, XModel.zb, XParam, XModel.blocks, XModel.fluxml, dt
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void solve_implicit_barotropic<float>(Param& XParam, Loop<float>& XLoop, Model<float>& XModel);
template void solve_implicit_barotropic<double>(Param& XParam, Loop<double>& XLoop, Model<double>& XModel);

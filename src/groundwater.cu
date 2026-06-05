
#include "groundwater.h"
#include "MemManagement.h"
#include "Setup_GPU.h"
#include "Util_CPU.h"
#include "Halo.h"

/**
 * @brief CUDA kernel to calculate Darcy flux in the x-direction.
 */
template <class T> __global__ void DarcyFluxXGPU(Param XParam, BlockP<T> XBlock, T* hgw, T* hsw, T* zb, T* K_gw, T* Aquifer_Depth, T* Qx)
{
    int halowidth = XParam.halowidth;
    int blkmemwidth = blockDim.x + halowidth * 2;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ibl = blockIdx.x;
    int ib = XBlock.active[ibl];

    // Using Darcy flux at the interface between idx_L and idx_R
    // The kernel is launched with blockDimKX (blkwidth + 1)
    if (ix < XParam.blkwidth + 1) {
        int idx_L = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
        int idx_R = memloc(halowidth, blkmemwidth, ix, iy, ib);

        T levdx = calcres(T(XParam.dx), XBlock.level[ib]);

        T bed_L = zb[idx_L] - Aquifer_Depth[idx_L];
        T bed_R = zb[idx_R] - Aquifer_Depth[idx_R];

        T thick_L = utils::max(T(0.0), utils::min(hgw[idx_L], zb[idx_L]) - bed_L);
        T thick_R = utils::max(T(0.0), utils::min(hgw[idx_R], zb[idx_R]) - bed_R);
        T avg_thick = T(0.5) * (thick_L + thick_R);

        T K_avg = T(0.5) * (K_gw[idx_L] + K_gw[idx_R]);
        T H_eff_L = hgw[idx_L] + hsw[idx_L];
        T H_eff_R = hgw[idx_R] + hsw[idx_R];

        // Qx[idx_R] is the flux at the left face of cell idx_R
        Qx[idx_R] = -K_avg * avg_thick * (H_eff_R - H_eff_L) / levdx;
    }
}

/**
 * @brief CUDA kernel to calculate Darcy flux in the y-direction.
 */
template <class T> __global__ void DarcyFluxYGPU(Param XParam, BlockP<T> XBlock, T* hgw, T* hsw, T* zb, T* K_gw, T* Aquifer_Depth, T* Qy)
{
    int halowidth = XParam.halowidth;
    int blkmemwidth = blockDim.x + halowidth * 2;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ibl = blockIdx.x;
    int ib = XBlock.active[ibl];

    if (iy < XParam.blkwidth + 1) {
        int idx_B = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
        int idx_T = memloc(halowidth, blkmemwidth, ix, iy, ib);

        T levdx = calcres(T(XParam.dx), XBlock.level[ib]);

        T bed_B = zb[idx_B] - Aquifer_Depth[idx_B];
        T bed_T = zb[idx_T] - Aquifer_Depth[idx_T];

        T thick_B = utils::max(T(0.0), utils::min(hgw[idx_B], zb[idx_B]) - bed_B);
        T thick_T = utils::max(T(0.0), utils::min(hgw[idx_T], zb[idx_T]) - bed_T);
        T avg_thick = T(0.5) * (thick_B + thick_T);

        T K_avg = T(0.5) * (K_gw[idx_B] + K_gw[idx_T]);
        T H_eff_B = hgw[idx_B] + hsw[idx_B];
        T H_eff_T = hgw[idx_T] + hsw[idx_T];

        // Qy[idx_T] is the flux at the bottom face of cell idx_T
        Qy[idx_T] = -K_avg * avg_thick * (H_eff_T - H_eff_B) / levdx;
    }
}

/**
 * @brief CUDA kernel for groundwater CFL condition.
 */
template <class T> __global__ void GroundwaterCFLGPU(Param XParam, BlockP<T> XBlock, T* hgw, T* zb, T* K_gw, T* Sy_gw, T* Aquifer_Depth, T* dtmax)
{
    int halowidth = XParam.halowidth;
    int blkmemwidth = blockDim.x + halowidth * 2;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ibl = blockIdx.x;
    int ib = XBlock.active[ibl];

    int idx = memloc(halowidth, blkmemwidth, ix, iy, ib);

    T levdx = calcres(T(XParam.dx), XBlock.level[ib]);

    T bed = zb[idx] - Aquifer_Depth[idx];
    T thick = utils::max(T(0.0), utils::min(hgw[idx], zb[idx]) - bed);

    T diffusion = (K_gw[idx] * thick) / utils::max(Sy_gw[idx], T(1e-6));

    if (diffusion > T(1e-10)) {
        T dt_gw = T(0.5) * T(XParam.CFL) * levdx * levdx / diffusion;
        if (dt_gw < dtmax[idx]) {
            dtmax[idx] = dt_gw;
        }
    }
}

/**
 * @brief CUDA kernel for Groundwater Mass Balance and Seepage.
 */
template <class T> __global__ void GroundwaterMassBalanceGPU(Param XParam, T dt, BlockP<T> XBlock, T* h_gw, T* h_sw, T* zs, T* topo, T* fs_gw, T* Sy_gw, T* Qx, T* Qy)
{
    int halowidth = XParam.halowidth;
    int blkmemwidth = blockDim.x + halowidth * 2;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ibl = blockIdx.x;
    int ib = XBlock.active[ibl];

    int idx = memloc(halowidth, blkmemwidth, ix, iy, ib);
    int idx_r = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
    int idx_t = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

    T levdx = calcres(T(XParam.dx), XBlock.level[ib]);

    // Qx[idx] is flux at the left face, Qx[idx_r] is flux at the right face
    // Qy[idx] is flux at the bottom face, Qy[idx_t] is flux at the top face
    // net_lateral = -(dQx/dx + dQy/dy) = (Qx_in - Qx_out)/dx + (Qy_in - Qy_out)/dy
    T net_lateral = (Qx[idx] - Qx[idx_r]) / levdx + (Qy[idx] - Qy[idx_t]) / levdx;

    T actual_inf = utils::min(h_sw[idx], fs_gw[idx] * dt);

    h_sw[idx] -= actual_inf;

    T new_h_gw = h_gw[idx] + (net_lateral * dt / Sy_gw[idx]) + (actual_inf / Sy_gw[idx]);

    if (new_h_gw > topo[idx]) {
        T seepage = (new_h_gw - topo[idx]) * Sy_gw[idx];
        h_sw[idx] += seepage;
        new_h_gw = topo[idx];
    }

    h_gw[idx] = new_h_gw;
    zs[idx] = topo[idx] + h_sw[idx];
}

/**
 * @brief Host routine to orchestrate groundwater kernels.
 */
template <class T> void GroundwaterStepGPU(Param XParam, Loop<T>& XLoop, Model<T> XModel)
{
    dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
    dim3 gridDim(XParam.nblk, 1, 1);

    dim3 blockDimKX(XParam.blkwidth + 1, XParam.blkwidth, 1);
    dim3 blockDimKY(XParam.blkwidth, XParam.blkwidth + 1, 1);

    fillHaloGPU(XParam, XModel.blocks, XModel.hgw);
    CUDA_CHECK(cudaDeviceSynchronize());

    DarcyFluxXGPU<<<gridDim, blockDimKX, 0>>>(XParam, XModel.blocks, XModel.hgw, XModel.evolv.h, XModel.zb, XModel.K_gw, XModel.Aquifer_Depth, XModel.Qx);
    DarcyFluxYGPU<<<gridDim, blockDimKY, 0>>>(XParam, XModel.blocks, XModel.hgw, XModel.evolv.h, XModel.zb, XModel.K_gw, XModel.Aquifer_Depth, XModel.Qy);
    CUDA_CHECK(cudaDeviceSynchronize());

    fillHaloGPU(XParam, XModel.blocks, XModel.Qx);
    fillHaloGPU(XParam, XModel.blocks, XModel.Qy);
    CUDA_CHECK(cudaDeviceSynchronize());

    GroundwaterMassBalanceGPU<<<gridDim, blockDim, 0>>>(XParam, T(XLoop.dt), XModel.blocks, XModel.hgw, XModel.evolv.h, XModel.evolv.zs, XModel.zb, XModel.fs_gw, XModel.Sy_gw, XModel.Qx, XModel.Qy);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void GroundwaterStepGPU<float>(Param XParam, Loop<float>& XLoop, Model<float> XModel);
template void GroundwaterStepGPU<double>(Param XParam, Loop<double>& XLoop, Model<double> XModel);

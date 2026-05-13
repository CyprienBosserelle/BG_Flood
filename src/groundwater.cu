
#include "groundwater.h"


/**
 * @brief CUDA kernel to calculate Darcy flux in the x-direction.
 *
 * @tparam T Data type (float or double)
 * @param XParam Simulation parameters
 * @param XBlock Block data structure
 * @param hgw Groundwater elevation array
 * @param hsw Surface water depth array
 * @param zb Topography elevation array
 * @param K_gw Hydraulic conductivity array
 * @param Aquifer_Depth Aquifer depth array
 * @param Qx Output Darcy flux array (x-direction)
 */
template <class T> __global__ void DarcyFluxXGPU(Param XParam, BlockP<T> XBlock, T* hgw, T* hsw, T* zb, T* K_gw, T* Aquifer_Depth, T* Qx)
{
    int halowidth = XParam.halowidth;
    int blkmemwidth = blockDim.x + halowidth * 2;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ibl = blockIdx.x;
    int ib = XBlock.active[ibl];

    int idx = memloc(halowidth, blkmemwidth, ix, iy, ib);
    int idx_r = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);

    T levdx = calcres(T(XParam.dx), XBlock.level[ib]);

    // Z_bed = Topo - Aquifer_Depth
    T bed_L = zb[idx] - Aquifer_Depth[idx];
    T bed_R = zb[idx_r] - Aquifer_Depth[idx_r];

    // Saturated thickness (D) = max(0, min(H_gw, Topo) - Z_bed)
    T thick_L = max(T(0.0), min(hgw[idx], zb[idx]) - bed_L);
    T thick_R = max(T(0.0), min(hgw[idx_r], zb[idx_r]) - bed_R);
    T avg_thick = T(0.5) * (thick_L + thick_R);

    // Qx = -K * avg_thick * ((h_gw[idx+1] + h_sw[idx+1]) - (h_gw[idx] + h_sw[idx])) / dx
    T K_avg = T(0.5) * (K_gw[idx] + K_gw[idx_r]);
    T H_eff_L = hgw[idx] + hsw[idx];
    T H_eff_R = hgw[idx_r] + hsw[idx_r];

    Qx[idx] = -K_avg * avg_thick * (H_eff_R - H_eff_L) / levdx;
}

/**
 * @brief CUDA kernel to calculate Darcy flux in the y-direction.
 *
 * @tparam T Data type (float or double)
 * @param XParam Simulation parameters
 * @param XBlock Block data structure
 * @param hgw Groundwater elevation array
 * @param hsw Surface water depth array
 * @param zb Topography elevation array
 * @param K_gw Hydraulic conductivity array
 * @param Aquifer_Depth Aquifer depth array
 * @param Qy Output Darcy flux array (y-direction)
 */
template <class T> __global__ void DarcyFluxYGPU(Param XParam, BlockP<T> XBlock, T* hgw, T* hsw, T* zb, T* K_gw, T* Aquifer_Depth, T* Qy)
{
    int halowidth = XParam.halowidth;
    int blkmemwidth = blockDim.x + halowidth * 2;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ibl = blockIdx.x;
    int ib = XBlock.active[ibl];

    int idx = memloc(halowidth, blkmemwidth, ix, iy, ib);
    int idx_t = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

    T levdx = calcres(T(XParam.dx), XBlock.level[ib]);

    T bed_B = zb[idx] - Aquifer_Depth[idx];
    T bed_T = zb[idx_t] - Aquifer_Depth[idx_t];

    T thick_B = max(T(0.0), min(hgw[idx], zb[idx]) - bed_B);
    T thick_T = max(T(0.0), min(hgw[idx_t], zb[idx_t]) - bed_T);
    T avg_thick = T(0.5) * (thick_B + thick_T);

    T K_avg = T(0.5) * (K_gw[idx] + K_gw[idx_t]);
    T H_eff_B = hgw[idx] + hsw[idx];
    T H_eff_T = hgw[idx_t] + hsw[idx_t];

    Qy[idx] = -K_avg * avg_thick * (H_eff_T - H_eff_B) / levdx;
}

/**
 * @brief CUDA kernel for Groundwater Mass Balance and Seepage.
 *
 * @tparam T Data type (float or double)
 * @param XParam Simulation parameters
 * @param dt Time step
 * @param XBlock Block data structure
 * @param h_gw Groundwater elevation array
 * @param h_sw Surface water depth array
 * @param zs Surface water elevation array
 * @param topo Topography elevation array
 * @param fs_gw Saturated infiltration rate array
 * @param Sy_gw Specific yield array
 * @param Qx Darcy flux array (x-direction)
 * @param Qy Darcy flux array (y-direction)
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
    int idx_l = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
    int idx_b = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);

    T levdx = calcres(T(XParam.dx), XBlock.level[ib]);

    // net_lateral = -(dQx/dx + dQy/dy)
    T net_lateral = -( (Qx[idx] - Qx[idx_l]) / levdx + (Qy[idx] - Qy[idx_b]) / levdx );

    // actual_inf = min(h_sw, fs * dt)
    T actual_inf = min(h_sw[idx], fs_gw[idx]/1000/3600 * dt);

    // Update surface water depth
    h_sw[idx] -= actual_inf;

    // new_h_gw = h_gw[idx] + (net_lateral * dt / Sy) + (actual_inf / Sy);
    T new_h_gw = h_gw[idx] + (net_lateral * dt / Sy_gw[idx]) + (actual_inf / Sy_gw[idx]);

    // If H_gw > Topo, transfer volume to surface water (seepage)
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

    dim3 blockDimKX(XParam.blkwidth + XParam.halowidth, XParam.blkwidth, 1);
    dim3 blockDimKY(XParam.blkwidth, XParam.blkwidth + XParam.halowidth, 1);

    // Calculate Darcy Fluxes
    DarcyFluxXGPU<<<gridDim, blockDimKX, 0>>>(XParam, XModel.blocks, XModel.hgw, XModel.evolv.h, XModel.zb, XModel.K_gw, XModel.Aquifer_Depth, XModel.Qx);
    DarcyFluxYGPU<<<gridDim, blockDimKY, 0>>>(XParam, XModel.blocks, XModel.hgw, XModel.evolv.h, XModel.zb, XModel.K_gw, XModel.Aquifer_Depth, XModel.Qy);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Update Mass Balance and Seepage
    GroundwaterMassBalanceGPU<<<gridDim, blockDim, 0>>>(XParam, T(XLoop.dt), XModel.blocks, XModel.hgw, XModel.evolv.h, XModel.evolv.zs, XModel.zb, XModel.fs_gw, XModel.Sy_gw, XModel.Qx, XModel.Qy);
    CUDA_CHECK(cudaDeviceSynchronize());
}

template void GroundwaterStepGPU<float>(Param XParam, Loop<float>& XLoop, Model<float> XModel);
template void GroundwaterStepGPU<double>(Param XParam, Loop<double>& XLoop, Model<double> XModel);

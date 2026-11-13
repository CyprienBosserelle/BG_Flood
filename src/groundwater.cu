
#include "groundwater.h"



/**
 * @brief CUDA kernel to calculate groundwater CFL condition.
 *
 * Stability for diffusion: dt < 0.5 * dx^2 / (K * D / Sy)
 * where D is saturated thickness.
 *
 * @tparam T Data type (float or double)
 * @param XParam Simulation parameters
 * @param XBlock Block data structure
 * @param hgw Groundwater elevation array
 * @param zb Topography elevation array
 * @param K_gw Hydraulic conductivity array
 * @param Sy_gw Specific yield array
 * @param Aquifer_Depth Aquifer depth array
 * @param dtmax Maximum time step array
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

    T maxbed = zb[idx] - Aquifer_Depth[idx];
    T thick = utils::max(T(0.0), utils::min(hgw[idx],maxbed));

    T diffusion = (K_gw[idx] * thick) / utils::max(Sy_gw[idx], T(1e-6));

    if (diffusion > T(1e-10)) {
        T dt_gw = T(0.5) * T(XParam.CFL) * levdx * levdx / diffusion;
        if (dt_gw < dtmax[idx]) {
            dtmax[idx] = dt_gw;
        }
    }
}
template __global__ void GroundwaterCFLGPU<float>(Param XParam, BlockP<float> XBlock, float* hgw, float* zb, float* K_gw, float* Sy_gw, float* Aquifer_Depth, float* dtmax);
template __global__ void GroundwaterCFLGPU<double>(Param XParam, BlockP<double> XBlock, double* hgw, double* zb, double* K_gw, double* Sy_gw, double* Aquifer_Depth, double* dtmax);

template <class T> __global__ void Updatehgw(Param XParam, BlockP<T> XBlock, T* hgw, T* zbgw, T* zsgw)
{
    int halowidth = XParam.halowidth;
    int blkmemwidth = blockDim.x + halowidth * 2;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int ibl = blockIdx.x;
    int ib = XBlock.active[ibl];

    int idx = memloc(halowidth, blkmemwidth, ix, iy, ib);

    T levdx = calcres(T(XParam.dx), XBlock.level[ib]);

    hgw[idx] = max(T(0.0), zsgw[idx] - zbgw[idx]);

}


template <class T> __host__ T CalctimestepGWGPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, TimeP<T> XTime,T dtmax_surf)
{
    T* dummy;
    AllocateCPU(32, 1, dummy);

    // densify dtmax (i.e. remove empty block and halo that may sit in the middle of the memory structure)
    int s = XParam.nblk * (XParam.blkwidth * XParam.blkwidth); // Not blksize wich includes Halo

    dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
    dim3 gridDim(XParam.nblk, 1, 1);

    densify << < gridDim, blockDim, 0 >> > (XParam, XBlock, XTime.dtmax, XTime.arrmin);
    CUDA_CHECK(cudaDeviceSynchronize());


    CUDA_CHECK(cudaMemcpy(XTime.dtmax, XTime.arrmin, s * sizeof(T), cudaMemcpyDeviceToDevice));


    //GPU Harris reduction #3. 8.3x reduction #0  Note #7 if a lot faster
    // This was successfully tested with a range of grid size
    //reducemax3 <<<gridDimLine, blockDimLine, 64*sizeof(float) >>>(dtmax_g, arrmax_g, nx*ny)

    int maxThreads = 256;
    int threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
    int blocks = (s + (threads * 2 - 1)) / (threads * 2);
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    dim3 blockDimLine(threads, 1, 1);
    dim3 gridDimLine(blocks, 1, 1);


    reducemin3 << <gridDimLine, blockDimLine, smemSize >> > (XTime.dtmax, XTime.arrmin, s);
    CUDA_CHECK(cudaDeviceSynchronize());



    s = gridDimLine.x;
    while (s > 1)//cpuFinalThreshold
    {
        threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
        blocks = (s + (threads * 2 - 1)) / (threads * 2);

        smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

        dim3 blockDimLineS(threads, 1, 1);
        dim3 gridDimLineS(blocks, 1, 1);

        CUDA_CHECK(cudaMemcpy(XTime.dtmax, XTime.arrmin, s * sizeof(T), cudaMemcpyDeviceToDevice));

        reducemin3 << <gridDimLineS, blockDimLineS, smemSize >> > (XTime.dtmax, XTime.arrmin, s);
        CUDA_CHECK(cudaDeviceSynchronize());

        s = (s + (threads * 2 - 1)) / (threads * 2);
    }


    CUDA_CHECK(cudaMemcpy(dummy, XTime.arrmin, 32 * sizeof(T), cudaMemcpyDeviceToHost)); // replace 32 by word here?

    T dtmaxgw = dummy[0];


    int n_substeps = ceil(dtmax_surf / dtmaxgw);
    //float dt_gw = dtmax_surf / n_substeps
    free(dummy);



    return n_substeps;

    
}


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
template <class T> __global__ void DarcyFluxXGPU(Param XParam, BlockP<T> XBlock, T* hgw, T* hsw, T* zb, T* K_gw, T* z_aqb, T* Qx)
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

    // max gw thickness
    T bed_L = zb[idx] - z_aqb[idx];
    T bed_R = zb[idx_r] - z_aqb[idx_r];

    // Saturated thickness (D) = max(0, min(H_gw, Topo) - Z_bed)
    T thick_L = utils::max(T(0.0), utils::min(hgw[idx], bed_L));
    T thick_R = utils::max(T(0.0), utils::min(hgw[idx_r], bed_R));
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
 * @param hgw Groundwater depth array
 * @param hsw Surface water depth array
 * @param zb Topography elevation array
 * @param K_gw Hydraulic conductivity array
 * @param z_aqb Aquifer bed elevation array
 * @param Qy Output Darcy flux array (y-direction)
 */
template <class T> __global__ void DarcyFluxYGPU(Param XParam, BlockP<T> XBlock, T* hgw, T* hsw, T* zb, T* K_gw, T* z_aqb, T* Qy)
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

    T bed_B = zb[idx] - z_aqb[idx];
    T bed_T = zb[idx_t] - z_aqb[idx_t];

    T thick_B = utils::max(T(0.0), utils::min(hgw[idx], bed_B));
    T thick_T = utils::max(T(0.0), utils::min(hgw[idx_t], bed_T));
    T avg_thick = T(0.5) * (thick_B + thick_T);

    T K_avg = T(0.5) * (K_gw[idx] + K_gw[idx_t]);
    T H_eff_B = hgw[idx] + hsw[idx]; // Should be calculated with z!
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
template <class T> __global__ void GroundwaterMassBalanceGPU(Param XParam, T dt, BlockP<T> XBlock, T* h_gw,T* zs_gw,T* zb_gw, T* h_sw, T* zs, T* topo, T* fs_gw, T* Sy_gw, T* Qx, T* Qy)
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
    T net_lateral =  -((Qx[idx] - Qx[idx_l]) / levdx + (Qy[idx] - Qy[idx_b]) / levdx);

    T fsfac = T(1.0) / 1000 / 3600;

    // actual_inf = min(h_sw, fs * dt)
    T actual_inf = utils::min(h_sw[idx], fs_gw[idx] * fsfac * dt);

    // Update surface water depth
    h_sw[idx] -= actual_inf;

    // new_h_gw = h_gw[idx] + (net_lateral * dt / Sy) + (actual_inf / Sy);
    T new_h_gw = h_gw[idx] + (net_lateral * dt / Sy_gw[idx]) + (actual_inf / Sy_gw[idx]);

    // If H_gw > Topo, transfer volume to surface water (seepage)
    if ((new_h_gw + zb_gw[idx]) > topo[idx]) {
        T seepage = (new_h_gw + zb_gw[idx] - topo[idx]) * Sy_gw[idx];
        h_sw[idx] += seepage;
        new_h_gw = topo[idx] - zb_gw[idx];
    }

    h_gw[idx] = new_h_gw;
    zs[idx] = topo[idx] + h_sw[idx];

    zs_gw[idx] = min(topo[idx], zb_gw[idx] + h_gw[idx]);


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

    fillHaloGPU(XParam, XModel.blocks, XModel.gw.zb);
    fillHaloGPU(XParam, XModel.blocks, XModel.gw.K);


    
    // It is likely that the GW timestep is smaller then surface water the GW model is simple so we run only the GW model n time to keep up with surface water. 
    // We hope that the CFL condition don't change too drastically during this calculation. the CFL of 0.5 should help with that
    
    GroundwaterCFLGPU << <gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.gw.h, XModel.zb, XModel.gw.K, XModel.gw.Sy, XModel.gw.zb, XModel.time.dtmax);
    CUDA_CHECK(cudaDeviceSynchronize());

    int n_substeps = CalctimestepGWGPU(XParam, XLoop, XModel.blocks, XModel.time, T(XLoop.dtmax));

    T dtgw = XLoop.dtmax / n_substeps;

    for (int i = 0; i < n_substeps; i++) 
    {
        Updatehgw << <gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.gw.h, XModel.gw.zb, XModel.gw.zs);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Fill halo for groundwater head
        fillHaloGPU(XParam, XModel.blocks, XModel.gw.h);
        
        // Calculate Darcy Fluxes
        DarcyFluxXGPU << <gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.gw.h, XModel.evolv.h, XModel.zb, XModel.gw.K, XModel.gw.zb, XModel.gw.Qx);
        DarcyFluxYGPU << <gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.gw.h, XModel.evolv.h, XModel.zb, XModel.gw.K, XModel.gw.zb, XModel.gw.Qy);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Fill halo for groundwater fluxes
        fillHaloGPU(XParam, XModel.blocks, XModel.gw.Qx);
        fillHaloGPU(XParam, XModel.blocks, XModel.gw.Qy);
        
        // Update Mass Balance and Seepage
        GroundwaterMassBalanceGPU << <gridDim, blockDim, 0 >> > (XParam, dtgw, XModel.blocks, XModel.gw.h, XModel.gw.zs, XModel.gw.zb, XModel.evolv.h, XModel.evolv.zs, XModel.zb, XModel.gw.fs, XModel.gw.Sy, XModel.gw.Qx, XModel.gw.Qy);
        CUDA_CHECK(cudaDeviceSynchronize());
        
    }
    
}

template void GroundwaterStepGPU<float>(Param XParam, Loop<float>& XLoop, Model<float> XModel);
template void GroundwaterStepGPU<double>(Param XParam, Loop<double>& XLoop, Model<double> XModel);

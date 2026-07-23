#include "Advection.h"

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

// specialize for double to avoid unaligned memory
// access compile errors
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


/**
 * @brief GPU kernel to update evolving variables (h, u, v, zs) for each block and cell.
 * @tparam T Data type
 * @param XParam Model parameters
 * @param XBlock Block data structure
 * @param XEv Evolving variables
 * @param XFlux Flux variables
 * @param XAdv Advance variables
 *
 * Computes new values for water height, velocity, and surface elevation using fluxes and advances.
 */
template <class T>__global__ void updateEVGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, FluxP<T> XFlux, AdvanceP<T> XAdv)
{

	int halowidth = XParam.halowidth;
	int blkmemwidth = blockDim.x + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];

	//T eps = T(XParam.eps);
	T delta = calcres(T(XParam.delta), lev);
	T g = T(XParam.g);

	T ybo = T(XParam.yo + XBlock.yo[ib]);


	T fc = 0.0;// XParam.spherical ? sin((ybo + calcres(T(XParam.dx), lev) * iy) * pi / 180.0) * pi / T(21600.0) : sin(T(XParam.lat * pi / 180.0)) * pi / T(21600.0); // 2*(2*pi/24/3600)
	// fc should be pi / T(21600.0) * sin(phi)


	int iright, itop;

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
	itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

	T yup = T(iy) + T(1.0);
	T ydwn = T(iy);

	if (iy == XParam.blkwidth - 1)
	{
		if (XBlock.level[XBlock.TopLeft[ib]] > XBlock.level[ib])
		{
			yup = iy + 0.75;
		}
		//if (XBlock.level[XBlock.TopLeft[ib]] < XBlock.level[ib])
		//{
		//	yup = iy + 1.000;
		//}
	}

	if (iy == 0)
	{
		if (XBlock.level[XBlock.BotLeft[ib]] > XBlock.level[ib])
		{
			ydwn = iy - 0.25 ;
		}
		
	}





	T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
	T fmu = T(1.0);
	T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, ydwn) : T(1.0);
	T fmup = T(1.0);
	T fmvp = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, yup) : T(1.0);



	T hi = XEv.h[i];
	T uui = XEv.u[i];
	T vvi = XEv.v[i];


	T cmdinv, ga;

	cmdinv = T(1.0) / (cm * delta);
	ga = T(0.5) * g;


	XAdv.dh[i] = T(-1.0) * (XFlux.Fhu[iright] - XFlux.Fhu[i] + XFlux.Fhv[itop] - XFlux.Fhv[i]) * cmdinv;



	//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
	//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
	T dmdl = (fmup - fmu) * cmdinv;// absurd if not spherical!
	T dmdt = (fmvp - fmv) * cmdinv;;
	T fG = vvi * dmdl - uui * dmdt;
	XAdv.dhu[i] = (XFlux.Fqux[i] + XFlux.Fquy[i] - XFlux.Su[iright] - XFlux.Fquy[itop]) * cmdinv + fc * hi * vvi;
	XAdv.dhv[i] = (XFlux.Fqvy[i] + XFlux.Fqvx[i] - XFlux.Sv[itop] - XFlux.Fqvx[iright]) * cmdinv - fc * hi * uui;

	XAdv.dhu[i] += hi * (ga * hi * dmdl + fG * vvi);// This term is == 0 so should be commented here
	XAdv.dhv[i] += hi * (ga * hi * dmdt - fG * uui);// Need double checking before doing that

	
	
}
template __global__ void updateEVGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, FluxP<float> XFlux, AdvanceP<float> XAdv);
template __global__ void updateEVGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, FluxP<double> XFlux, AdvanceP<double> XAdv);


/**
 * @brief CPU routine to update evolving variables (h, u, v, zs) for each block and cell.
 * @tparam T Data type
 * @param XParam Model parameters
 * @param XBlock Block data structure
 * @param XEv Evolving variables
 * @param XFlux Flux variables
 * @param XAdv Advance variables
 *
 * Computes new values for water height, velocity, and surface elevation using fluxes and advances.
 */
template <class T>__host__ void updateEVCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, FluxP<T> XFlux, AdvanceP<T> XAdv)
{

	//T eps = T(XParam.eps);
	T delta;
	T g = T(XParam.g);

	T ybo;
	

	int ib,lev;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		lev = XBlock.level[ib];
		delta = calcres(T(XParam.delta), lev);

		ybo = (T)XParam.yo + XBlock.yo[ib];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				T fc = XParam.spherical ? sin((ybo + calcres(T(XParam.dx), lev) * iy) * pi / 180.0) * pi / T(21600.0) : sin(T(XParam.lat * pi / 180.0)) * pi / T(21600.0); // 2*(2*pi/24/3600)

				int iright, itop;

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
				itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);


				T yup = T(iy) + T(1.0);
				T ydwn = T(iy);

				if (iy == XParam.blkwidth - 1)
				{
					if (XBlock.level[XBlock.TopLeft[ib]] > XBlock.level[ib])
					{
						yup = iy + T(0.75);
					}
					
				}
				if (iy == 0)
				{
					if (XBlock.level[XBlock.BotLeft[ib]] > XBlock.level[ib])
					{
						ydwn = iy - T(0.25);
					}

				}



				T cm = XParam.spherical ? calcCM(T(XParam.Radius), delta, ybo, iy) : T(1.0);
				T fmu = T(1.0);
				T fmv = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, ydwn) : T(1.0);
				T fmup = T(1.0);
				T fmvp = XParam.spherical ? calcFM(T(XParam.Radius), delta, ybo, yup) : T(1.0);

				T hi = XEv.h[i];
				T uui = XEv.u[i];
				T vvi = XEv.v[i];


				T cmdinv, ga;

				cmdinv = T(1.0) / (cm * delta);
				ga = T(0.5) * g;


				XAdv.dh[i] = T(-1.0) * (XFlux.Fhu[iright] - XFlux.Fhu[i] + XFlux.Fhv[itop] - XFlux.Fhv[i]) * cmdinv;



				//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
				//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
				T dmdl = (fmup - fmu) / (cm * delta);// absurd if not spherical!
				T dmdt = (fmvp - fmv) / (cm * delta);
				T fG = vvi * dmdl - uui * dmdt;
				XAdv.dhu[i] = (XFlux.Fqux[i] + XFlux.Fquy[i] - XFlux.Su[iright] - XFlux.Fquy[itop]) * cmdinv + fc * hi * vvi;
				XAdv.dhv[i] = (XFlux.Fqvy[i] + XFlux.Fqvx[i] - XFlux.Sv[itop] - XFlux.Fqvx[iright]) * cmdinv - fc * hi * uui;

				
				XAdv.dhu[i] += hi * (ga * hi * dmdl + fG * vvi);// This term is == 0 so should be commented here
				XAdv.dhv[i] += hi * (ga * hi * dmdt - fG * uui);// Need double checking before doing that
			}
		}
	}
}
template __host__ void updateEVCPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, FluxP<float> XFlux, AdvanceP<float> XAdv);
template __host__ void updateEVCPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, FluxP<double> XFlux, AdvanceP<double> XAdv);


/**
 * @brief GPU kernel for advancing the solution in time for each block and cell.
 * @tparam T Data type
 * @param XParam Model parameters
 * @param XBlock Block data structure
 * @param dt Time step
 * @param zb Bed elevation array
 * @param XEv Evolving variables
 * @param XAdv Advance variables
 * @param XEv_o Output evolving variables
 *
 * Updates water height, velocity, and surface elevation for the next time step.
 */
template <class T> __global__ void AdvkernelGPU(Param XParam, BlockP<T> XBlock, T dt ,T* zb, EvolvingP<T> XEv, AdvanceP<T> XAdv, EvolvingP<T> XEv_o)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	T eps = T(XParam.eps);
	T hold = XEv.h[i];
	T ho, uo, vo;
	T dhi = XAdv.dh[i];

	T edt = XParam.ForceMassConserve ? dt : dhi >= T(0.0) ? dt : min(dt, max(hold, XParam.eps) / abs(dhi));
	
	//ho = max(hold + edt * dhi,T(0.0));
	ho = hold + edt * dhi;

	if (ho > eps) {
		//
		uo = (hold * XEv.u[i] + edt * XAdv.dhu[i]) / ho;
		vo = (hold * XEv.v[i] + edt * XAdv.dhv[i]) / ho;
		
	}
	else
	{// dry
		
		uo = T(0.0);
		vo = T(0.0);
	}


	XEv_o.zs[i] = zb[i] + ho;
	XEv_o.h[i] = ho;
	XEv_o.u[i] = uo;
	XEv_o.v[i] = vo;
	

}
template __global__ void AdvkernelGPU<float>(Param XParam, BlockP<float> XBlock, float dt, float* zb, EvolvingP<float> XEv, AdvanceP<float> XAdv, EvolvingP<float> XEv_o);
template __global__ void AdvkernelGPU<double>(Param XParam, BlockP<double> XBlock, double dt, double* zb, EvolvingP<double> XEv, AdvanceP<double> XAdv, EvolvingP<double> XEv_o);


/**
 * @brief CPU routine for advancing the solution in time for each block and cell.
 * @tparam T Data type
 * @param XParam Model parameters
 * @param XBlock Block data structure
 * @param dt Time step
 * @param zb Bed elevation array
 * @param XEv Evolving variables
 * @param XAdv Advance variables
 * @param XEv_o Output evolving variables
 *
 * Updates water height, velocity, and surface elevation for the next time step.
 */
template <class T> __host__ void AdvkernelCPU(Param XParam, BlockP<T> XBlock, T dt, T* zb, EvolvingP<T> XEv, AdvanceP<T> XAdv, EvolvingP<T> XEv_o)
{
	T eps = T(XParam.eps);
	


	int ib;
	//int halowidth = XParam.halowidth;
	//int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(XParam, ix, iy, ib);

				
				T hold = XEv.h[i];
				T ho, uo, vo, dhi;

				dhi = XAdv.dh[i];

				T edt = XParam.ForceMassConserve ? dt : dhi >= T(0.0) ? dt : min(dt, max(hold, XParam.eps) / abs(dhi));

				ho = hold + edt * dhi;


				if (ho > eps) {
					//
					uo = (hold * XEv.u[i] + edt * XAdv.dhu[i]) / ho;
					vo = (hold * XEv.v[i] + edt * XAdv.dhv[i]) / ho;

				}
				else
				{// dry

					uo = T(0.0);
					vo = T(0.0);
				}


				XEv_o.zs[i] = zb[i] + ho;
				XEv_o.h[i] = ho;
				XEv_o.u[i] = uo;
				XEv_o.v[i] = vo;
			}
		}
	}

}
template __host__ void AdvkernelCPU<float>(Param XParam, BlockP<float> XBlock, float dt, float* zb, EvolvingP<float> XEv, AdvanceP<float> XAdv, EvolvingP<float> XEv_o);
template __host__ void AdvkernelCPU<double>(Param XParam, BlockP<double> XBlock, double dt, double* zb, EvolvingP<double> XEv, AdvanceP<double> XAdv, EvolvingP<double> XEv_o);



/**
 * @brief GPU kernel to clean up evolving variables after advection step.
 * @tparam T Data type
 * @param XParam Model parameters
 * @param XBlock Block data structure
 * @param XEv Evolving variables
 * @param XEv_o Output evolving variables
 */
template <class T> __global__ void cleanupGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, EvolvingP<T> XEv_o)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	
	XEv_o.h[i] = XEv.h[i];
	XEv_o.zs[i] = XEv.zs[i];
	XEv_o.u[i] = XEv.u[i];
	XEv_o.v[i] = XEv.v[i];

}
template __global__ void cleanupGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, EvolvingP<float> XEv_o);
template __global__ void cleanupGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, EvolvingP<double> XEv_o);



/**
 * @brief CPU routine to clean up evolving variables after advection step.
 * @tparam T Data type
 * @param XParam Model parameters
 * @param XBlock Block data structure
 * @param XEv Evolving variables
 * @param XEv_o Output evolving variables
 */
template <class T> __host__ void cleanupCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, EvolvingP<T> XEv_o)
{
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				XEv_o.h[i] = XEv.h[i];
				XEv_o.zs[i] = XEv.zs[i];
				XEv_o.u[i] = XEv.u[i];
				XEv_o.v[i] = XEv.v[i];
			}
		}
	}

}
template __host__ void cleanupCPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, EvolvingP<float> XEv_o);
template __host__ void cleanupCPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, EvolvingP<double> XEv_o);


/**
 * @brief CPU routine to compute the minimum allowed time step across all blocks and cells.
 * @tparam T Data type
 * @param XParam Model parameters
 * @param XLoop Loop control structure
 * @param XBlock Block data structure
 * @param XTime Time control structure
 * @return Minimum allowed time step
 */
template <class T> __host__ T timestepreductionCPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, TimeP<T> XTime)
{
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);

	T dt = T(1.0) / epsi;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				//
				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				dt = utils::min(dt, XTime.dtmax[i]);

			}
		}
	}

	return dt;
}
template __host__ float timestepreductionCPU(Param XParam, Loop<float> XLoop, BlockP<float> XBlock, TimeP<float> XTime);
template __host__ double timestepreductionCPU(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, TimeP<double> XTime);

/**
 * @brief CPU routine to calculate the next time step for the simulation.
 * @tparam T Data type
 * @param XParam Model parameters
 * @param XLoop Loop control structure
 * @param XBlock Block data structure
 * @param XTime Time control structure
 * @return Computed time step
 */
template <class T> __host__ T CalctimestepCPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, TimeP<T> XTime)
{
	

	T dt= timestepreductionCPU(XParam,XLoop,XBlock,XTime);

	

	// also don't allow dt to be larger than 1.5*dtmax (usually the last time step or smallest delta/sqrt(gh) if the first step)
	if (dt > (1.5 * XLoop.dtmax))
	{
		dt = T(1.5 * XLoop.dtmax);
	}

	if (ceil((XLoop.nextoutputtime - XLoop.totaltime) / dt) > 0.0)
	{
		dt = T((XLoop.nextoutputtime - XLoop.totaltime) / ceil((XLoop.nextoutputtime - XLoop.totaltime) / dt));
	}

	

	return dt;

	
}
template __host__ float CalctimestepCPU<float>(Param XParam, Loop<float> XLoop, BlockP<float> XBlock, TimeP<float> XTime);
template __host__ double CalctimestepCPU<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, TimeP<double> XTime);


/**
 * @brief GPU routine to calculate the next time step for the simulation.
 * @tparam T Data type
 * @param XParam Model parameters
 * @param XLoop Loop control structure
 * @param XBlock Block data structure
 * @param XTime Time control structure
 * @return Computed time step
 */
template <class T> __host__ T CalctimestepGPU(Param XParam,Loop<T> XLoop, BlockP<T> XBlock, TimeP<T> XTime)
{
	T* dummy;
	AllocateCPU(32, 1, dummy);

	// densify dtmax (i.e. remove empty block and halo that may sit in the middle of the memory structure)
	int s = XParam.nblk * (XParam.blkwidth* XParam.blkwidth); // Not blksize wich includes Halo

	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	densify <<< gridDim, blockDim, 0 >>>(XParam, XBlock, XTime.dtmax, XTime.arrmin);
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

	
	reducemin3 <<<gridDimLine, blockDimLine, smemSize >>> (XTime.dtmax, XTime.arrmin, s);
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

		reducemin3 <<<gridDimLineS, blockDimLineS, smemSize >>> (XTime.dtmax, XTime.arrmin, s);
		CUDA_CHECK(cudaDeviceSynchronize());

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}


	CUDA_CHECK(cudaMemcpy(dummy, XTime.arrmin, 32 * sizeof(T), cudaMemcpyDeviceToHost)); // replace 32 by word here?

	if (dummy[0] > (1.5 * XLoop.dtmax))
	{
		dummy[0] = T(1.5 * XLoop.dtmax);
	}

	if (ceil((XLoop.nextoutputtime - XLoop.totaltime) / dummy[0]) > 0.0)
	{
		dummy[0] = T((XLoop.nextoutputtime - XLoop.totaltime) / ceil((XLoop.nextoutputtime - XLoop.totaltime) / dummy[0]));
	}


	return dummy[0];

	free(dummy);
}
template __host__ float CalctimestepGPU<float>(Param XParam,Loop<float> XLoop, BlockP<float> XBlock, TimeP<float> XTime);
template __host__ double CalctimestepGPU<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, TimeP<double> XTime);

template <class T> __host__ T reducedot(Param XParam, BlockP<T> XBlock, T* a, T* b, T* store)
{
	T* dummy;
	AllocateCPU(32, 1, dummy);

	// densify dtmax (i.e. remove empty block and halo that may sit in the middle of the memory structure)
	int s = XParam.nblk * (XParam.blkwidth* XParam.blkwidth); // Not blksize wich includes Halo

	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	densify <<< gridDim, blockDim, 0 >>>(XParam, XBlock, a, store);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(a, store, s * sizeof(T), cudaMemcpyDeviceToDevice));

	densify <<< gridDim, blockDim, 0 >>>(XParam, XBlock, b, store);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	CUDA_CHECK(cudaMemcpy(b, store, s * sizeof(T), cudaMemcpyDeviceToDevice));
	


	//GPU Harris reduction #3. 8.3x reduction #0  Note #7 if a lot faster
	// This was successfully tested with a range of grid size
	//reducemax3 <<<gridDimLine, blockDimLine, 64*sizeof(float) >>>(dtmax_g, arrmax_g, nx*ny)
	
	int maxThreads = 256;
	int threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
	int blocks = (s + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	dim3 blockDimLine(threads, 1, 1);
	dim3 gridDimLine(blocks, 1, 1);

	
	dotReduce3 <<<gridDimLine, blockDimLine, smemSize >>> (a, b, store, s);
	CUDA_CHECK(cudaDeviceSynchronize());



	s = gridDimLine.x;
	while (s > 1)//cpuFinalThreshold
	{
		threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);

		smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

		dim3 blockDimLineS(threads, 1, 1);
		dim3 gridDimLineS(blocks, 1, 1);

		CUDA_CHECK(cudaMemcpy(a, store, s * sizeof(T), cudaMemcpyDeviceToDevice));

		sumReduce3 <<<gridDimLineS, blockDimLineS, smemSize >>> (a, store, s);
		CUDA_CHECK(cudaDeviceSynchronize());

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}


	CUDA_CHECK(cudaMemcpy(dummy, store, 32 * sizeof(T), cudaMemcpyDeviceToHost)); // replace 32 by word here?


	return dummy[0];

	free(dummy);
}
template __host__ float reducedot<float>(Param XParam, BlockP<float> XBlock, float* a, float* b, float* store);
template __host__ double reducedot<double>(Param XParam, BlockP<double> XBlock, double* a, double* b, double* store);

template <class T> __host__ T reduceabsmaxold(Param XParam, BlockP<T> XBlock, T* a,T* store)
{
	T* dummy;
	AllocateCPU(32, 1, dummy);

	// densify dtmax (i.e. remove empty block and halo that may sit in the middle of the memory structure)
	int s = XParam.nblk * (XParam.blkwidth* XParam.blkwidth); // Not blksize wich includes Halo

	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	densify <<< gridDim, blockDim, 0 >>>(XParam, XBlock, a, store);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(a, store, s * sizeof(T), cudaMemcpyDeviceToDevice));

	


	//GPU Harris reduction #3. 8.3x reduction #0  Note #7 if a lot faster
	// This was successfully tested with a range of grid size
	//reducemax3 <<<gridDimLine, blockDimLine, 64*sizeof(float) >>>(dtmax_g, arrmax_g, nx*ny)
	
	int maxThreads = 256;
	int threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
	int blocks = (s + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	dim3 blockDimLine(threads, 1, 1);
	dim3 gridDimLine(blocks, 1, 1);

	
	absmaxReduce3 <<<gridDimLine, blockDimLine, smemSize >>> (a, store, s);
	CUDA_CHECK(cudaDeviceSynchronize());



	s = gridDimLine.x;
	while (s > 1)//cpuFinalThreshold
	{
		threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);

		smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

		dim3 blockDimLineS(threads, 1, 1);
		dim3 gridDimLineS(blocks, 1, 1);

		CUDA_CHECK(cudaMemcpy(a, store, s * sizeof(T), cudaMemcpyDeviceToDevice));

		maxReduce3 <<<gridDimLineS, blockDimLineS, smemSize >>> (a, store, s);
		CUDA_CHECK(cudaDeviceSynchronize());

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}


	CUDA_CHECK(cudaMemcpy(dummy, store, 32 * sizeof(T), cudaMemcpyDeviceToHost)); // replace 32 by word here?


	return dummy[0];

	free(dummy);
}
template __host__ float reduceabsmaxold<float>(Param XParam, BlockP<float> XBlock, float* a, float* store);
template __host__ double reduceabsmaxold<double>(Param XParam, BlockP<double> XBlock, double* a, double* store);

// ---------------------------------------------------------------------
// Stage 1: one CUDA block per ACTIVE memory block. Every thread reads
// exactly one interior cell, the whole block's 16x16=256 threads then
// do a Harris-style (kernel 3, sequential addressing) shared-memory
// reduction down to ONE max value for that block.
// ---------------------------------------------------------------------
template <class T> __global__ void absmaxReduceStage1(Param XParam, BlockP<T> XBlock, T* a,T* store)
{
    extern __shared__ T sdata[];

	 int blkmemwidth = XParam.blkwidth + 2 * XParam.halowidth;

    int ix  = threadIdx.x;
    int iy  = threadIdx.y;
    int ibl = blockIdx.x;
    int ib  = XBlock.active[ibl];          // real block id, only ACTIVE blocks get launched

    // Flatten the 2D thread index for the 1D shared-memory reduction below.
    unsigned int tid = iy * blockDim.x + ix;

    int id = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);
    sdata[tid] = fabs(a[id]);        // single, non-destructive read of `a`
    __syncthreads();

    // Sequential addressing (Harris kernel 3): halve the active range each
    // step, contiguous threads tid=0..s-1 do the work -> no bank conflicts,
    // no warp divergence. blockDim.x*blockDim.y (256) is a power of two,
    // so this cleanly reaches s=0.
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) store[ibl] = sdata[0];   // one value per active block
}

// ---------------------------------------------------------------------
// Stage 2: reduce the (small) per-block maxima down to one scalar.
// Same sequential-addressing technique, generic over array length via
// a multi-pass host loop (exactly the pattern used for reduceDot's
// Harris-3 version) since nblkactive isn't guaranteed to fit in one
// block's worth of threads.
// ---------------------------------------------------------------------
template <class T> __global__ void maxReduceStage2(const T* __restrict__ g_idata,
                                 T* __restrict__ g_odata,
                                 unsigned int n)
{
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // Cells beyond n contribute 0.0, the identity for max over |.| >= 0 data.
    sdata[tid] = (i < n) ? g_idata[i] : 0.0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------
// Host driver.
//   a              : the field to reduce (e.g. f.r, residual, etc.)
//   d_active       : device pointer to XBlock.active (compact list,
//                    length nblkactive)
//   nblkactive     : number of ACTIVE blocks (== grid size for stage 1)
//   halowidth      : XParam.halowidth (1 in your example)
//   blkdim         : interior tile size (16 in your example)
// ---------------------------------------------------------------------
//double reduceAbsMax(const double* a, const int* d_active, int nblkactive, int halowidth, int blkdim)
template <class T> T reduceAbsMax(Param XParam, BlockP<T> XBlock, T* a,T* store)
{
    int blkmemwidth = XParam.blkwidth + 2 * XParam.halowidth;

	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
    dim3 threads1(XParam.blkwidth, XParam.blkwidth);
    dim3 blocks1(XParam.nblk);
    size_t smem1 = XParam.blkwidth * XParam.blkwidth * sizeof(T);

    static T* d_blockMax = nullptr;
    static unsigned int allocatedBlocks = 0;
    if ((unsigned)XParam.nblk > allocatedBlocks) {
        if (d_blockMax) cudaFree(d_blockMax);
        cudaMalloc(&d_blockMax, XParam.nblk * sizeof(T));
        allocatedBlocks = XParam.nblk;
    }

	//absmaxReduceStage1(Param XParam, BlockP<T> XBlock, T* a,T* store)

    absmaxReduceStage1<<<blocks1, threads1, smem1>>>(XParam,XBlock,a,store);

    // --- Stage 2: multi-pass reduction of the nblkactive per-block maxima ---
    const unsigned int threads2 = 256;
    unsigned int n = (unsigned int)XParam.nblk;

    static T* d_bufA = nullptr;
    static T* d_bufB = nullptr;
    static unsigned int allocatedSize2 = 0;
    unsigned int neededSize = std::max((n + threads2 - 1) / threads2, 1u);
    if (neededSize > allocatedSize2) {
        if (d_bufA) cudaFree(d_bufA);
        if (d_bufB) cudaFree(d_bufB);
        cudaMalloc(&d_bufA, neededSize * sizeof(T));
        cudaMalloc(&d_bufB, neededSize * sizeof(T));
        allocatedSize2 = neededSize;
    }

    // First stage-2 pass reads directly from d_blockMax (the stage-1 output).
    unsigned int blocks2 = (n + threads2 - 1) / threads2;
    size_t smem2 = threads2 * sizeof(T);
    maxReduceStage2<<<blocks2, threads2, smem2>>>(d_blockMax, d_bufA, n);

    n = blocks2;
    T* src = d_bufA;
    T* dst = d_bufB;
    while (n > 1)
    {
        unsigned int nextBlocks = (n + threads2 - 1) / threads2;
        maxReduceStage2<<<nextBlocks, threads2, smem2>>>(src, dst, n);
        std::swap(src, dst);
        n = nextBlocks;
    }

    T h_result;
    cudaMemcpy(&h_result, src, sizeof(T), cudaMemcpyDeviceToHost);
    return h_result;
}
template float reduceAbsMax<float>(Param XParam, BlockP<float> XBlock, float* a,float* store);
template double reduceAbsMax<double>(Param XParam, BlockP<double> XBlock, double* a,double* store);

/**
 * @brief GPU kernel to compute the minimum value in an array using parallel reduction.
 * @tparam T Data type
 * @param g_idata Input array
 * @param g_odata Output array (min per block)
 * @param n Number of elements
 */
template <class T> __global__ void reducemin3(T* g_idata, T* g_odata, unsigned int n)
{
	//T *sdata = SharedMemory<T>();
	T* sdata = SharedMemory<T>();
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	T myMin = (i < n) ? g_idata[i] : T(1e30);

	if (i + blockDim.x < n)
		myMin = min(myMin, g_idata[i + blockDim.x]);

	sdata[tid] = myMin;
	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = myMin = min(myMin, sdata[tid + s]);
		}

		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = myMin;
}

// ---------------------------------------------------------------------
// Pass 1: fused a[i]*b[i], one thread per element, sequential-addressing
// reduction down to one partial sum per block.
// ---------------------------------------------------------------------
template <class T> __global__ void dotReduce3(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ g_odata, unsigned int n)
{
    T* sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // One global load (well, two: a and b) per thread -- no "first add
    // during load" here, that's kernel 4's optimization.
    sdata[tid] = (i < n) ? a[i] * b[i] : 0.0;
    __syncthreads();

    // Sequential addressing: s halves each step, active threads are
    // always the contiguous range tid = 0..s-1 -> no divergence, and
    // sdata[tid]/sdata[tid+s] accesses don't collide on shared-memory
    // banks (unlike kernel 2's `2*s*tid` indexing).
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();   // still needed every step -- no warp-level shortcut here
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------
// Subsequent passes: plain sum reduction over a contiguous buffer
// (the partial sums produced by the previous pass). Same technique,
// just without the a[]*b[] multiply.
// ---------------------------------------------------------------------
template <class T> __global__ void sumReduce3(const T* __restrict__ g_idata, T* __restrict__ g_odata, unsigned int n)
{
    T* sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T> __global__ void absmaxReduce3(const T* __restrict__ a, T* __restrict__ g_odata, unsigned int n)
{
    T* sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // Out-of-range threads contribute 0.0, the identity element for max
    // over non-negative values (fabs(...) is always >= 0).
    sdata[tid] = (i < n) ? fabs(a[i]) : 0.0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T> __global__ void maxReduce3(const T* __restrict__ g_idata, T* __restrict__ g_odata, unsigned int n)
{
    T* sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/**
 * @brief GPU kernel to copy and densify data from block memory to output array.
 * @tparam T Data type
 * @param XParam Model parameters
 * @param XBlock Block data structure
 * @param g_idata Input array
 * @param g_odata Output array
 */
template <class T> __global__ void densify(Param XParam, BlockP<T> XBlock, T* g_idata, T* g_odata)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int o = ix + iy * blockDim.x + ibl * (blockDim.x * blockDim.x);

	g_odata[o] = g_idata[i];
}


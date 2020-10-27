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


template <class T>__global__ void updateEVGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, FluxP<T> XFlux, AdvanceP<T> XAdv)
{
	
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	T eps = T(XParam.eps);
	T delta = calcres(T(XParam.dx), XBlock.level[ib]);
	T g = T(XParam.g);
	

	T fc = (T)XParam.lat * pi / T(21600.0);

	int iright, itop;
	
	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	
	iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
	itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

	

	T cm = T(1.0);// 0.1;
	T fmu = T(1.0);
	T fmv = T(1.0);

	T hi = XEv.h[i];
	T uui = XEv.u[i];
	T vvi = XEv.v[i];


	T cmdinv, ga;

	cmdinv = T(1.0)/ (cm * delta);
	ga = T(0.5) * g;
		

	XAdv.dh[i] = T(-1.0) * (XFlux.Fhu[iright] - XFlux.Fhu[i] + XFlux.Fhv[itop] - XFlux.Fhv[i]) * cmdinv;
		


	//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
	//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
	T dmdl = (fmu - fmu) / (cm * delta);// absurd if not spherical!
	T dmdt = (fmv - fmv) / (cm * delta);
	T fG = vvi * dmdl - uui * dmdt;
	XAdv.dhu[i] = (XFlux.Fqux[i] + XFlux.Fquy[i] - XFlux.Su[iright] - XFlux.Fquy[itop]) * cmdinv + fc * hi * vvi;
	XAdv.dhv[i] = (XFlux.Fqvy[i] + XFlux.Fqvx[i] - XFlux.Sv[itop] - XFlux.Fqvx[iright]) * cmdinv - fc * hi * uui;
		
	XAdv.dhu[i] += hi * (ga * hi * dmdl + fG * vvi);// This term is == 0 so should be commented here
	XAdv.dhv[i] += hi * (ga * hi * dmdt - fG * uui);// Need double checking before doing that
	
}
template __global__ void updateEVGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, FluxP<float> XFlux, AdvanceP<float> XAdv);
template __global__ void updateEVGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, FluxP<double> XFlux, AdvanceP<double> XAdv);


template <class T>__host__ void updateEVCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, FluxP<T> XFlux, AdvanceP<T> XAdv)
{

	T eps = T(XParam.eps);
	T delta;
	T g = T(XParam.g);
	

	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		delta = calcres(T(XParam.dx), XBlock.level[ib]);
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				T fc = (T)XParam.lat * pi / T(21600.0);

				int iright, itop;

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
				itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);



				T cm = T(1.0);// 0.1;
				T fmu = T(1.0);
				T fmv = T(1.0);

				T hi = XEv.h[i];
				T uui = XEv.u[i];
				T vvi = XEv.v[i];


				T cmdinv, ga;

				cmdinv = T(1.0) / (cm * delta);
				ga = T(0.5) * g;


				XAdv.dh[i] = T(-1.0) * (XFlux.Fhu[iright] - XFlux.Fhu[i] + XFlux.Fhv[itop] - XFlux.Fhv[i]) * cmdinv;



				//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
				//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
				T dmdl = (fmu - fmu) / (cm * delta);// absurd if not spherical!
				T dmdt = (fmv - fmv) / (cm * delta);
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


template <class T> __global__ void AdvkernelGPU(Param XParam, BlockP<T> XBlock, T dt ,T* zb, EvolvingP<T> XEv, AdvanceP<T> XAdv, EvolvingP<T> XEv_o)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	T eps = T(XParam.eps);
	T hold = XEv.h[i];
	T ho, uo, vo;
	ho = hold + dt * XAdv.dh[i];


	if (ho > eps) {
		//
		uo = (hold * XEv.u[i] + dt * XAdv.dhu[i]) / ho;
		vo = (hold * XEv.v[i] + dt * XAdv.dhv[i]) / ho;
		
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


template <class T> __host__ void AdvkernelCPU(Param XParam, BlockP<T> XBlock, T dt, T* zb, EvolvingP<T> XEv, AdvanceP<T> XAdv, EvolvingP<T> XEv_o)
{
	T eps = T(XParam.eps);
	


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

				int i = memloc(XParam, ix, iy, ib);

				
				T hold = XEv.h[i];
				T ho, uo, vo;
				ho = hold + dt * XAdv.dh[i];


				if (ho > eps) {
					//
					uo = (hold * XEv.u[i] + dt * XAdv.dhu[i]) / ho;
					vo = (hold * XEv.v[i] + dt * XAdv.dhv[i]) / ho;

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



template <class T> __global__ void cleanupGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, EvolvingP<T> XEv_o)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
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

template <class T> __host__ T CalctimestepCPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, TimeP<T> XTime)
{
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);

	T dt=T(1.0)/epsi;

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

	// also don't allow dt to be larger than 1.5*dtmax (usually the last time step or smallest delta/sqrt(gh) if the first step)
	if (dt > (1.5 * XLoop.dtmax))
	{
		dt = (1.5 * XLoop.dtmax);
	}

	if (ceil((XLoop.nextoutputtime - XLoop.totaltime) / dt) > 0.0)
	{
		dt = (XLoop.nextoutputtime - XLoop.totaltime) / ceil((XLoop.nextoutputtime - XLoop.totaltime) / dt);
	}

	

	return dt;

	
}
template __host__ float CalctimestepCPU<float>(Param XParam, Loop<float> XLoop, BlockP<float> XBlock, TimeP<float> XTime);
template __host__ double CalctimestepCPU<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, TimeP<double> XTime);


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
	//reducemax3 << <gridDimLine, blockDimLine, 64*sizeof(float) >> >(dtmax_g, arrmax_g, nx*ny)
	
	int maxThreads = 256;
	int threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
	int blocks = (s + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	dim3 blockDimLine(threads, 1, 1);
	dim3 gridDimLine(blocks, 1, 1);

	float mindtmaxB;

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

	if (dummy[0] > (1.5 * XLoop.dtmax))
	{
		dummy[0] = (1.5 * XLoop.dtmax);
	}

	if (ceil((XLoop.nextoutputtime - XLoop.totaltime) / dummy[0]) > 0.0)
	{
		dummy[0] = (XLoop.nextoutputtime - XLoop.totaltime) / ceil((XLoop.nextoutputtime - XLoop.totaltime) / dummy[0]);
	}


	return dummy[0];

	free(dummy);
}
template __host__ float CalctimestepGPU<float>(Param XParam,Loop<float> XLoop, BlockP<float> XBlock, TimeP<float> XTime);
template __host__ double CalctimestepGPU<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, TimeP<double> XTime);




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


template <class T> __global__ void densify(Param XParam, BlockP<T> XBlock, T* g_idata, T* g_odata)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int o = ix + iy * blockDim.x + ibl * (blockDim.x * blockDim.x);

	g_odata[o] = g_idata[i];
}


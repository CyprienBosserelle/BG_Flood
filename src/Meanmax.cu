
#include "Meanmax.h"


template <class T> void Calcmeanmax(Param XParam, Loop<T>& XLoop, Model<T> XModel, Model<T> XModel_g)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	
	if (XParam.outmean)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			addavg_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evmean.h, XModel_g.evolv.h);
			addavg_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evmean.zs, XModel_g.evolv.zs);
			addavg_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evmean.u, XModel_g.evolv.u);
			addavg_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evmean.v, XModel_g.evolv.v);
			addUandhU_GPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evolv.h, XModel_g.evolv.u, XModel_g.evolv.v, XModel_g.evmean.U, XModel_g.evmean.hU);

			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{

			addavg_varCPU(XParam, XModel.blocks, XModel.evmean.h, XModel.evolv.h);
			addavg_varCPU(XParam, XModel.blocks, XModel.evmean.zs, XModel.evolv.zs);
			addavg_varCPU(XParam, XModel.blocks, XModel.evmean.u, XModel.evolv.u);
			addavg_varCPU(XParam, XModel.blocks, XModel.evmean.v, XModel.evolv.v);
			addUandhU_CPU(XParam, XModel.blocks, XModel.evolv.h, XModel.evolv.u, XModel.evolv.v, XModel.evmean.U, XModel.evmean.hU);

		}


		XLoop.nstep++;

		if (XLoop.nextoutputtime - XLoop.totaltime <= XLoop.dt * T(0.00001))
		{
			// devide by number of steps
			if (XParam.GPUDEVICE >= 0)
			{
				divavg_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, T(XLoop.nstep), XModel_g.evmean.h);
				divavg_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, T(XLoop.nstep), XModel_g.evmean.zs);
				divavg_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, T(XLoop.nstep), XModel_g.evmean.u);
				divavg_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, T(XLoop.nstep), XModel_g.evmean.v);
				divavg_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, T(XLoop.nstep), XModel_g.evmean.U);
				divavg_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, T(XLoop.nstep), XModel_g.evmean.hU);
				CUDA_CHECK(cudaDeviceSynchronize());
			}
			else
			{
				divavg_varCPU(XParam, XModel.blocks, T(XLoop.nstep), XModel.evmean.h);
				divavg_varCPU(XParam, XModel.blocks, T(XLoop.nstep), XModel.evmean.zs);
				divavg_varCPU(XParam, XModel.blocks, T(XLoop.nstep), XModel.evmean.u);
				divavg_varCPU(XParam, XModel.blocks, T(XLoop.nstep), XModel.evmean.v);
				divavg_varCPU(XParam, XModel.blocks, T(XLoop.nstep), XModel.evmean.U);
				divavg_varCPU(XParam, XModel.blocks, T(XLoop.nstep), XModel.evmean.hU);
			}

			//XLoop.nstep will be reset after a save to the disk which occurs in a different function
		}

	}
	if (XParam.outmax)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			max_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evmax.h, XModel_g.evolv.h);
			max_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evmax.zs, XModel_g.evolv.zs);
			max_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evmax.u, XModel_g.evolv.u);
			max_varGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evmax.v, XModel_g.evolv.v);
			max_Norm_GPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evmax.U, XModel_g.evolv.u, XModel_g.evolv.v);
			max_hU_GPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evmax.hU, XModel_g.evolv.h, XModel_g.evolv.u, XModel_g.evolv.v);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			max_varCPU(XParam, XModel.blocks, XModel.evmax.h, XModel.evolv.h);
			max_varCPU(XParam, XModel.blocks, XModel.evmax.zs, XModel.evolv.zs);
			max_varCPU(XParam, XModel.blocks, XModel.evmax.u, XModel.evolv.u);
			max_varCPU(XParam, XModel.blocks, XModel.evmax.v, XModel.evolv.v);
			max_Norm_CPU(XParam, XModel.blocks, XModel.evmax.U, XModel.evolv.u, XModel.evolv.v);
			max_hU_CPU(XParam, XModel.blocks, XModel.evmax.hU, XModel.evolv.h, XModel.evolv.u, XModel.evolv.v);
		}
	}
	if (XParam.outtwet)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			// Add value GPU
			addwettime_GPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.wettime, XModel_g.evolv.h, T(0.1), T(XLoop.dt));

		}
		else
		{
			// Add value CPU
			addwettime_CPU(XParam, XModel.blocks, XModel.wettime, XModel.evolv.h, T(0.1), T(XLoop.dt));
		}
	}
}
template void Calcmeanmax<float>(Param XParam, Loop<float>& XLoop, Model<float> XModel, Model<float> XModel_g);
template void Calcmeanmax<double>(Param XParam, Loop<double>& XLoop, Model<double> XModel, Model<double> XModel_g);


template <class T> void resetmeanmax(Param XParam, Loop<T>& XLoop, Model<T> XModel, Model<T> XModel_g)
{
	// Reset mean and or max only at output steps
	//XLoop.nextoutputtime - XLoop.totaltime <= XLoop.dt * T(0.00001)
	if (XLoop.nstepout == 0) //This implis an output was just produced so need to reset
	{
		//Define some useful variables 
		if (XParam.outmean)
		{
			if (XParam.GPUDEVICE >= 0)
			{
				resetmeanGPU(XParam, XLoop, XModel_g.blocks, XModel_g.evmean);
			}
			else
			{
				resetmeanCPU(XParam, XLoop, XModel.blocks, XModel.evmean);
			}
			XLoop.nstep = 0;
		}

		//Reset Max 
		if (XParam.outmax && XParam.resetmax)
		{
			if (XParam.GPUDEVICE >= 0)
			{
				resetmaxGPU(XParam, XLoop, XModel_g.blocks, XModel_g.evmax);
			}
			else
			{
				resetmaxCPU(XParam, XLoop, XModel.blocks, XModel.evmax);

			}
		}

		//Reset Wet duration
		if (XParam.outtwet && XParam.resetmax)
		{
			if (XParam.GPUDEVICE >= 0)
			{
				resetvalGPU(XParam, XModel_g.blocks, XModel_g.wettime, T(0.0));
			}
			else
			{
				resetvalCPU(XParam, XModel.blocks, XModel.wettime, T(0.0));
			}
		}
	}
}
template void resetmeanmax<float>(Param XParam, Loop<float>& XLoop, Model<float> XModel, Model<float> XModel_g);
template void resetmeanmax<double>(Param XParam, Loop<double>& XLoop, Model<double> XModel, Model<double> XModel_g);

template <class T> void Initmeanmax(Param XParam, Loop<T> XLoop, Model<T> XModel, Model<T> XModel_g)
{
	//at the initial step overide the reset max to initialise the max variable (if needed)
	//this override is not preserved so wont affect the rest of the loop
	XParam.resetmax = true;
	XLoop.nextoutputtime = XLoop.totaltime;
	XLoop.dt = T(1.0);
	resetmeanmax(XParam, XLoop, XModel, XModel_g);
}
template void Initmeanmax<float>(Param XParam, Loop<float> XLoop, Model<float> XModel, Model<float> XModel_g);
template void Initmeanmax<double>(Param XParam, Loop<double> XLoop, Model<double> XModel, Model<double> XModel_g);

template <class T> void resetmaxGPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, EvolvingP_M<T>& XEv)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.h);
	CUDA_CHECK(cudaDeviceSynchronize());
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.zs);
	CUDA_CHECK(cudaDeviceSynchronize());
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.u);
	CUDA_CHECK(cudaDeviceSynchronize());
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.v);
	CUDA_CHECK(cudaDeviceSynchronize());
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.U);
	CUDA_CHECK(cudaDeviceSynchronize());
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.hU);
	CUDA_CHECK(cudaDeviceSynchronize());

}


template <class T> void resetmaxCPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, EvolvingP_M<T>& XEv)
{

	InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.h);
	InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.zs);
	InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.u);
	InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.v);
	InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.U);
	InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.hU);

}


template <class T> void resetmeanCPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, EvolvingP_M<T>& XEv)
{

	InitArrayBUQ(XParam, XBlock, T(0.0), XEv.h);
	InitArrayBUQ(XParam, XBlock, T(0.0), XEv.zs);
	InitArrayBUQ(XParam, XBlock, T(0.0), XEv.u);
	InitArrayBUQ(XParam, XBlock, T(0.0), XEv.v);
	InitArrayBUQ(XParam, XBlock, T(0.0), XEv.U);
	InitArrayBUQ(XParam, XBlock, T(0.0), XEv.hU);

}
template void resetmeanCPU<float>(Param XParam, Loop<float> XLoop, BlockP<float> XBlock, EvolvingP_M<float>& XEv);
template void resetmeanCPU<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, EvolvingP_M<double>& XEv);

template <class T> void resetmeanGPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, EvolvingP_M<T>& XEv)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	//
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.h);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.zs);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.u);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.v);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.U);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.hU);
	CUDA_CHECK(cudaDeviceSynchronize());


}
template void resetmeanGPU<float>(Param XParam, Loop<float> XLoop, BlockP<float> XBlock, EvolvingP_M<float>& XEv);
template void resetmeanGPU<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, EvolvingP_M<double>& XEv);


template <class T> void resetvalCPU(Param XParam, BlockP<T> XBlock, T*& var, T val)
{

	InitArrayBUQ(XParam, XBlock, val, var);

}
template void resetvalCPU<float>(Param XParam, BlockP<float> XBlock, float*& var, float val);
template void resetvalCPU<double>(Param XParam, BlockP<double> XBlock, double*& var, double val);

template <class T> void resetvalGPU(Param XParam, BlockP<T> XBlock, T*& var, T val)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, val, var);
	CUDA_CHECK(cudaDeviceSynchronize());

}
template void resetvalGPU<float>(Param XParam, BlockP<float> XBlock, float*& var, float val);
template void resetvalGPU<double>(Param XParam, BlockP<double> XBlock, double*& var, double val);



template <class T> __global__ void addavg_varGPU(Param XParam, BlockP<T> XBlock, T* Varmean, T* Var)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);


	Varmean[i] = Varmean[i] + Var[i];

}


template <class T> __host__ void addavg_varCPU(Param XParam, BlockP<T> XBlock, T* Varmean, T* Var)
{
	int ib, n;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

				Varmean[i] = Varmean[i] + Var[i];
			}
		}
	}

}

template <class T> __global__ void divavg_varGPU(Param XParam, BlockP<T> XBlock, T ntdiv, T* Varmean)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	Varmean[i] = Varmean[i] / ntdiv;

}

template <class T> __host__ void divavg_varCPU(Param XParam, BlockP<T> XBlock, T ntdiv, T* Varmean)
{
	int ib, n;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

				Varmean[i] = Varmean[i] / ntdiv;
			}
		}
	}

}

template <class T> __global__ void addUandhU_GPU(Param XParam, BlockP<T> XBlock, T * h, T * u, T * v, T* U, T* hU)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	U[i] = sqrt((u[i] * u[i]) + (v[i] * v[i]));
	hU[i] = h[i] * U[i];

}

template <class T> __host__ void addUandhU_CPU(Param XParam, BlockP<T> XBlock, T* h, T* u, T* v, T* U, T* hU)
{
	int ib, n;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

				U[i] = sqrt((u[i] * u[i]) + (v[i] * v[i]));
				hU[i] = h[i] * U[i];
			}
		}
	}

}

template <class T> __global__ void max_varGPU(Param XParam, BlockP<T> XBlock, T* Varmax, T* Var)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	Varmax[i] = max(Varmax[i], Var[i]);

}

template <class T> __global__ void max_Norm_GPU(Param XParam, BlockP<T> XBlock, T* Varmax, T* Var1, T* Var2)
{
	T Var_norm;
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	
	Var_norm = sqrt((Var1[i] * Var1[i]) + (Var2[i] * Var2[i]));
	Varmax[i] = max(Varmax[i], Var_norm);

}

template <class T> __global__ void max_hU_GPU(Param XParam, BlockP<T> XBlock, T* Varmax, T* h, T* u, T* v)
{
	T Var_hU;
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	Var_hU = h[i] * sqrt((u[i]*u[i])+(v[i]*v[i]));
	Varmax[i] = max(Varmax[i], Var_hU);

}

template <class T> __host__ void max_varCPU(Param XParam, BlockP<T> XBlock, T* Varmax, T* Var)
{
	int ib, n;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

				Varmax[i] = utils::max(Varmax[i], Var[i]);
			}
		}
	}

}

template <class T> __host__ void max_Norm_CPU(Param XParam, BlockP<T> XBlock, T* Varmax, T* Var1, T* Var2)
{
	int ib, n;
	T Var_norm;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
				Var_norm = sqrt((Var1[i] * Var1[i]) + (Var2[i] * Var2[i]));
				Varmax[i] = utils::max(Varmax[i], Var_norm);
			}
		}
	}

}

template <class T> __host__ void max_hU_CPU(Param XParam, BlockP<T> XBlock, T* Varmax, T* h, T* u, T* v)
{
	int ib, n;
	T Var_hU;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
				Var_hU = h[i] * sqrt((u[i] * u[i]) + (v[i] * v[i]));
				Varmax[i] = utils::max(Varmax[i], Var_hU);
			}
		}
	}

}

template <class T> __global__ void addwettime_GPU(Param XParam, BlockP<T> XBlock, T* wett, T* h, T thresold, T time)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	if (h[i] > thresold)
	{
		wett[i] = wett[i] + time;
	}

}


template <class T> __host__ void addwettime_CPU(Param XParam, BlockP<T> XBlock, T* wett, T* h, T thresold, T time)
{
	int ib, n;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

				if (h[i] > thresold)
				{
					wett[i] = wett[i] + time;
				}
			}
		}
	}

}
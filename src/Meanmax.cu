
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
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			addavg_varCPU(XParam, XModel.blocks, XModel.evmean.h, XModel.evolv.h);
			addavg_varCPU(XParam, XModel.blocks, XModel.evmean.zs, XModel.evolv.zs);
			addavg_varCPU(XParam, XModel.blocks, XModel.evmean.u, XModel.evolv.u);
			addavg_varCPU(XParam, XModel.blocks, XModel.evmean.v, XModel.evolv.v);
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
				CUDA_CHECK(cudaDeviceSynchronize());
			}
			else
			{
				divavg_varCPU(XParam, XModel.blocks, T(XLoop.nstep), XModel.evmean.h);
				divavg_varCPU(XParam, XModel.blocks, T(XLoop.nstep), XModel.evmean.zs);
				divavg_varCPU(XParam, XModel.blocks, T(XLoop.nstep), XModel.evmean.u);
				divavg_varCPU(XParam, XModel.blocks, T(XLoop.nstep), XModel.evmean.v);
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
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			max_varCPU(XParam, XModel.blocks, XModel.evmax.h, XModel.evolv.h);
			max_varCPU(XParam, XModel.blocks, XModel.evmax.zs, XModel.evolv.zs);
			max_varCPU(XParam, XModel.blocks, XModel.evmax.u, XModel.evolv.u);
			max_varCPU(XParam, XModel.blocks, XModel.evmax.v, XModel.evolv.v);
		}
	}
}
template void Calcmeanmax<float>(Param XParam, Loop<float>& XLoop, Model<float> XModel, Model<float> XModel_g);
template void Calcmeanmax<double>(Param XParam, Loop<double>& XLoop, Model<double> XModel, Model<double> XModel_g);


template <class T> void resetmeanmax(Param XParam, Loop<T>& XLoop, Model<T> XModel, Model<T> XModel_g)
{
	// Reset mean and or max only at output steps
	if (XLoop.nextoutputtime - XLoop.totaltime <= XLoop.dt * T(0.00001))
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
	}
}
template void resetmeanmax<float>(Param XParam, Loop<float>& XLoop, Model<float> XModel, Model<float> XModel_g);
template void resetmeanmax<double>(Param XParam, Loop<double>& XLoop, Model<double> XModel, Model<double> XModel_g);

template <class T> void Initmeanmax(Param XParam, Loop<T> XLoop, Model<T> XModel, Model<T> XModel_g)
{
	//at the initiial step overide the reset max to initialise the max variable (if needed)
	//this override is not preserved so wont affect the rest of the loop
	XParam.resetmax = true;
	XLoop.nextoutputtime = XLoop.totaltime;
	XLoop.dt = T(1.0);
	resetmeanmax(XParam, XLoop, XModel, XModel_g);
}
template void Initmeanmax<float>(Param XParam, Loop<float> XLoop, Model<float> XModel, Model<float> XModel_g);
template void Initmeanmax<double>(Param XParam, Loop<double> XLoop, Model<double> XModel, Model<double> XModel_g);

template <class T> void resetmaxGPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, EvolvingP<T>& XEv)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.h);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.zs);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.u);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XLoop.hugenegval, XEv.v);
	CUDA_CHECK(cudaDeviceSynchronize());

}


template <class T> void resetmaxCPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, EvolvingP<T>& XEv)
{

	InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.h);
	InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.zs);
	InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.u);
	InitArrayBUQ(XParam, XBlock, XLoop.hugenegval, XEv.v);

}


template <class T> void resetmeanCPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, EvolvingP<T>& XEv)
{

	InitArrayBUQ(XParam, XBlock, T(0.0), XEv.h);
	InitArrayBUQ(XParam, XBlock, T(0.0), XEv.zs);
	InitArrayBUQ(XParam, XBlock, T(0.0), XEv.u);
	InitArrayBUQ(XParam, XBlock, T(0.0), XEv.v);
}
template void resetmeanCPU<float>(Param XParam, Loop<float> XLoop, BlockP<float> XBlock, EvolvingP<float>& XEv);
template void resetmeanCPU<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, EvolvingP<double>& XEv);

template <class T> void resetmeanGPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, EvolvingP<T>& XEv)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	//
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.h);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.zs);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.u);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, T(0.0), XEv.v);
	CUDA_CHECK(cudaDeviceSynchronize());


}
template void resetmeanGPU<float>(Param XParam, Loop<float> XLoop, BlockP<float> XBlock, EvolvingP<float>& XEv);
template void resetmeanGPU<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, EvolvingP<double>& XEv);






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

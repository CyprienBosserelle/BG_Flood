#include "FlowMLGPU.h"

template <class T> void FlowMLGPU(Param XParam, Loop<T>& XLoop, Forcing<float> XForcing, Model<T> XModel)
{
	//============================================
	// construct threads abnd block parameters
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	// for flux reconstruction the loop overlap the right(or top for the y direction) halo
	dim3 blockDimKX(XParam.blkwidth + XParam.halowidth, XParam.blkwidth, 1);
	dim3 blockDimKY(XParam.blkwidth, XParam.blkwidth + XParam.halowidth, 1);

	// Fill halo for Fu and Fv
	dim3 blockDimHaloLR(2, XParam.blkwidth, 1);
	//dim3 blockDimHaloBT(16, 1, 1);
	dim3 gridDimHaloLR(ceil(XParam.nblk / 2), 1, 1);

	dim3 blockDimHaloBT(XParam.blkwidth, 2, 1);
	dim3 gridDimHaloBT(ceil(XParam.nblk / 2), 1, 1);


	// fill halo for zs,h,u and v 

	//============================================
	//  Fill the halo for gradient reconstruction & Recalculate zs
	fillHaloGPU(XParam, XModel.blocks, XModel.evolv, XModel.zb);
	
	// calculate grad for dhdx dhdy only
	
	//============================================
	// Calculate gradient for evolving parameters for predictor step
	gradientGPUnew(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.zb);
	//gradientSMC << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.delta, XModel.evolv.h, XModel.grad.dhdx, XModel.grad.dhdy);
	//CUDA_CHECK(cudaDeviceSynchronize());


	//============================================
	// Synchronise all ongoing streams
	CUDA_CHECK(cudaDeviceSynchronize());

	// Set max timestep
	reset_var <<< gridDim, blockDim, 0 >>> (XParam.halowidth, XModel.blocks.active, XLoop.hugeposval, XModel.time.dtmax);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Compute face value
	CalcfaceValX << < gridDim, blockDim, 0 >> > (T(XLoop.dtmax), XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.fluxml, XModel.time.dtmax, XModel.zb);
	CUDA_CHECK(cudaDeviceSynchronize());

	CalcfaceValY << < gridDim, blockDim, 0 >> > (T(XLoop.dtmax), XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.fluxml, XModel.time.dtmax, XModel.zb);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Timestep reduction
	XLoop.dt = double(CalctimestepGPU(XParam, XLoop, XModel.blocks, XModel.time));
	XLoop.dtmax = XLoop.dt;

	// Check hu/hv
	CheckadvecMLY << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, T(XLoop.dt), XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

	CheckadvecMLX << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, T(XLoop.dt), XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

	//Fill flux Halo for ha and hf
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hfu);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hfv);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hau);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hav);

	//HaloFluxGPULRnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfu);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUBTnew << <gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfv);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPULRnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hau);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUBTnew << <gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hav);
	//CUDA_CHECK(cudaDeviceSynchronize());

	// Acceleration
	// Pressure
	pressureML << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, T(XLoop.dt), XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Fill halo u and v calc grd for u and v and fill halo for hu and hv
	// 

	fillHaloGPU(XParam, XModel.blocks, XModel.evolv.u);
	fillHaloGPU(XParam, XModel.blocks, XModel.evolv.v);

	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hu);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.hv);
	//HaloFluxGPULRnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hu);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUBTnew << <gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hv);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPULRnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hv);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUBTnew << <gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hu);
	//CUDA_CHECK(cudaDeviceSynchronize());

	gradientSMC << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.delta, XModel.evolv.u, XModel.grad.dudx, XModel.grad.dudy);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientSMC << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.delta, XModel.evolv.v, XModel.grad.dvdx, XModel.grad.dvdy);
	CUDA_CHECK(cudaDeviceSynchronize());


	fillCornersGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XModel.fluxml.hu);
	CUDA_CHECK(cudaDeviceSynchronize());

	/*fillCornersGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hv);
	CUDA_CHECK(cudaDeviceSynchronize());

	fillCornersGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfu);
	CUDA_CHECK(cudaDeviceSynchronize());

	fillCornersGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.fluxml.hfv);
	CUDA_CHECK(cudaDeviceSynchronize());

	fillCornersGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.evolv.u);
	CUDA_CHECK(cudaDeviceSynchronize());

	fillCornersGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.evolv.v);
	CUDA_CHECK(cudaDeviceSynchronize());*/

	//hv hfv u hu hfu v
	
	// Advection
	AdvecFluxML << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, T(XLoop.dt), XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());

	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.Fu);
	fillHaloGPU(XParam, XModel.blocks, XModel.fluxml.Fv);

	//HaloFluxGPULRnew << < gridDimHaloLR, blockDimHaloLR, 0 >> > (XParam, XModel.blocks, XModel.fluxml.Fu);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//HaloFluxGPUBTnew << <gridDimHaloBT, blockDimHaloBT, 0 >> > (XParam, XModel.blocks, XModel.fluxml.Fv);
	//CUDA_CHECK(cudaDeviceSynchronize());

	AdvecEv << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, T(XLoop.dt), XModel.evolv, XModel.grad, XModel.fluxml);
	CUDA_CHECK(cudaDeviceSynchronize());


	if (XForcing.rivers.size() > 0)
	{
		//Add River ML
	}


	bottomfrictionGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.time.dt, XModel.cf, XModel.evolv);
	//XiafrictionGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XModel.time.dt, XModel.cf, XModel.evolv, XModel.evolv_o);


	CUDA_CHECK(cudaDeviceSynchronize());

	if (XForcing.rivers.size() > 0)
	{
		//Add River ML
	}


	if (!XForcing.Rain.inputfile.empty())
	{
		AddrainforcingImplicitGPU << < gridDim, blockDim, 0 >> > (XParam, XLoop, XModel.blocks, XForcing.Rain, XModel.evolv);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	if (XParam.infiltration)
	{
		AddinfiltrationImplicitGPU << < gridDim, blockDim, 0 >> > (XParam, XLoop, XModel.blocks, XModel.il, XModel.cl, XModel.evolv, XModel.hgw);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	if (XParam.VelThreshold > 0.0)
	{
		TheresholdVelGPU << < gridDim, blockDim, 0 >> > (XParam, XModel.blocks, XModel.evolv);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

}
template void FlowMLGPU<float>(Param XParam, Loop<float>& XLoop, Forcing<float> XForcing, Model<float> XModel);
template void FlowMLGPU<double>(Param XParam, Loop<double>& XLoop, Forcing<float> XForcing, Model<double> XModel);


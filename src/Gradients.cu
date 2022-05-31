#include "Gradients.h"


/*! \fn void gradientGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad,T* zb)
* Wrapping function to calculate gradien of evolving variables on GPU
* This function is the entry point to the gradient functions on the GPU
*/
template <class T> void gradientGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad,T* zb)
{
	//const int num_streams = 4;
	/*
	cudaStream_t streams[num_streams];

	for (int i = 0; i < num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}
	*/
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 blockDimLR(1, XParam.blkwidth, 1);
	dim3 blockDimBT(XParam.blkwidth, 1, 1);
	dim3 blockDimfull(XParam.blkmemwidth, XParam.blkmemwidth, 1);
	
	dim3 gridDim(XParam.nblk, 1, 1);

	//gradient << < gridDim, blockDim, 0, streams[1] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx, XGrad.dhdy);
	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx, XGrad.dhdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	//gradient << < gridDim, blockDim, 0, streams[2] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	//gradient << < gridDim, blockDim, 0, streams[3] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx, XGrad.dudy);
	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx, XGrad.dudy);
	CUDA_CHECK(cudaDeviceSynchronize());

	//gradient << < gridDim, blockDim, 0, streams[0] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx, XGrad.dvdy);
	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx, XGrad.dvdy);
	CUDA_CHECK(cudaDeviceSynchronize());


	//CUDA_CHECK(cudaDeviceSynchronize());
	/*
	for (int i = 0; i < num_streams; i++)
	{
		cudaStreamDestroy(streams[i]);
	}
	*/


	//fillHaloGPU(XParam, XBlock, XGrad);
	gradientHaloGPU(XParam, XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
	gradientHaloGPU(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	gradientHaloGPU(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	gradientHaloGPU(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);


	if (XParam.conserveElevation)
	{
		conserveElevationGradHaloGPU(XParam, XBlock, XEv.h, XEv.zs, zb, XGrad.dhdx, XGrad.dzsdx, XGrad.dhdy, XGrad.dzsdy);
	}
	else
	{
		refine_linearGPU(XParam, XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
		//refine_linearGPU(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
		refine_linearGPU(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
		refine_linearGPU(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);

		RecalculateZsGPU << < gridDim, blockDimfull, 0 >> > (XParam, XBlock, XEv, zb);
		CUDA_CHECK(cudaDeviceSynchronize());

		//gradient << < gridDim, blockDim, 0, streams[1] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx, XGrad.dhdy);
		gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx, XGrad.dhdy);
		CUDA_CHECK(cudaDeviceSynchronize());

		//gradient << < gridDim, blockDim, 0, streams[2] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
		gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
		CUDA_CHECK(cudaDeviceSynchronize());

		//gradient << < gridDim, blockDim, 0, streams[3] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx, XGrad.dudy);
		gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx, XGrad.dudy);
		CUDA_CHECK(cudaDeviceSynchronize());

		//gradient << < gridDim, blockDim, 0, streams[0] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx, XGrad.dvdy);
		gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx, XGrad.dvdy);
		CUDA_CHECK(cudaDeviceSynchronize());



		/*
		gradientedgeX << < gridDim, blockDimLR, 0 >> > (XParam.halowidth, 0, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx);
		gradientedgeX << < gridDim, blockDimLR, 0 >> > (XParam.halowidth, XParam.blkwidth-1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx);

		gradientedgeY << < gridDim, blockDimBT, 0 >> > (XParam.halowidth, 0, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdy);
		gradientedgeY << < gridDim, blockDimBT, 0 >> > (XParam.halowidth, XParam.blkwidth-1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdy);

		gradientedgeX << < gridDim, blockDimLR, 0 >> > (XParam.halowidth, 0, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx);
		gradientedgeX << < gridDim, blockDimLR, 0 >> > (XParam.halowidth, XParam.blkwidth - 1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx);

		gradientedgeY << < gridDim, blockDimBT, 0 >> > (XParam.halowidth, 0, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdy);
		gradientedgeY << < gridDim, blockDimBT, 0 >> > (XParam.halowidth, XParam.blkwidth - 1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdy);

		gradientedgeX << < gridDim, blockDimLR, 0 >> > (XParam.halowidth, 0, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx);
		gradientedgeX << < gridDim, blockDimLR, 0 >> > (XParam.halowidth, XParam.blkwidth - 1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx);

		gradientedgeY << < gridDim, blockDimBT, 0 >> > (XParam.halowidth, 0, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudy);
		gradientedgeY << < gridDim, blockDimBT, 0 >> > (XParam.halowidth, XParam.blkwidth - 1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudy);

		gradientedgeX << < gridDim, blockDimLR, 0 >> > (XParam.halowidth, 0, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx);
		gradientedgeX << < gridDim, blockDimLR, 0 >> > (XParam.halowidth, XParam.blkwidth - 1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx);

		gradientedgeY << < gridDim, blockDimBT, 0 >> > (XParam.halowidth, 0, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdy);
		gradientedgeY << < gridDim, blockDimBT, 0 >> > (XParam.halowidth, XParam.blkwidth - 1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdy);
		CUDA_CHECK(cudaDeviceSynchronize());
		*/


		gradientHaloGPU(XParam, XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
		gradientHaloGPU(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
		gradientHaloGPU(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
		gradientHaloGPU(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);

		if (XParam.engine == 1)
		{
			//  wet slope limiter
			WetsloperesetXGPU << < gridDim, blockDim, 0 >> > (XParam, XBlock, XEv, XGrad, zb);
			CUDA_CHECK(cudaDeviceSynchronize());

			WetsloperesetYGPU << < gridDim, blockDim, 0 >> > (XParam, XBlock, XEv, XGrad, zb);
			CUDA_CHECK(cudaDeviceSynchronize());

			// ALso do the slope limiter on the halo
			WetsloperesetHaloLeftGPU << < gridDim, blockDimLR, 0 >> > (XParam, XBlock, XEv, XGrad, zb);
			CUDA_CHECK(cudaDeviceSynchronize());

			WetsloperesetHaloRightGPU << < gridDim, blockDimLR, 0 >> > (XParam, XBlock, XEv, XGrad, zb);
			CUDA_CHECK(cudaDeviceSynchronize());

			WetsloperesetHaloBotGPU << < gridDim, blockDimBT, 0 >> > (XParam, XBlock, XEv, XGrad, zb);
			CUDA_CHECK(cudaDeviceSynchronize());

			WetsloperesetHaloTopGPU << < gridDim, blockDimBT, 0 >> > (XParam, XBlock, XEv, XGrad, zb);

			CUDA_CHECK(cudaDeviceSynchronize());
		}

	}
	//conserveElevationGradHaloGPU(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	//conserveElevationGradHaloGPU(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	//conserveElevationGradHaloGPU(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);

}
template void gradientGPU<float>(Param XParam, BlockP<float>XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, float * zb);
template void gradientGPU<double>(Param XParam,  BlockP<double>XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, double * zb);

template <class T> void gradientGPUnew(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{
	
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 blockDimLR(1, XParam.blkwidth, 1);
	dim3 blockDimBT(XParam.blkwidth, 1, 1);
	dim3 blockDimLR2(2, XParam.blkwidth, 1);
	dim3 blockDimBT2(XParam.blkwidth, 2, 1);
	dim3 blockDimfull(XParam.blkmemwidth, XParam.blkmemwidth, 1);

	dim3 gridDim(XParam.nblk, 1, 1);

	//gradient << < gridDim, blockDim, 0, streams[1] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx, XGrad.dhdy);
	gradientSM << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx, XGrad.dhdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	//gradient << < gridDim, blockDim, 0, streams[2] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	gradientSM << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	//gradient << < gridDim, blockDim, 0, streams[3] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx, XGrad.dudy);
	gradientSM << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx, XGrad.dudy);
	CUDA_CHECK(cudaDeviceSynchronize());

	//gradient << < gridDim, blockDim, 0, streams[0] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx, XGrad.dvdy);
	gradientSM << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx, XGrad.dvdy);
	CUDA_CHECK(cudaDeviceSynchronize());


	//CUDA_CHECK(cudaDeviceSynchronize());
	
	


	//fillHaloGPU(XParam, XBlock, XGrad);
	gradientHaloGPU(XParam, XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
	gradientHaloGPU(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	gradientHaloGPU(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	gradientHaloGPU(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);


	if (XParam.conserveElevation)
	{
		conserveElevationGradHaloGPU(XParam, XBlock, XEv.h, XEv.zs, zb, XGrad.dhdx, XGrad.dzsdx, XGrad.dhdy, XGrad.dzsdy);
	}
	else
	{
		refine_linearGPU(XParam, XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
		//refine_linearGPU(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
		refine_linearGPU(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
		refine_linearGPU(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);

		RecalculateZsGPU << < gridDim, blockDimfull, 0 >> > (XParam, XBlock, XEv, zb);
		CUDA_CHECK(cudaDeviceSynchronize());
		
		/*
		//gradient << < gridDim, blockDim, 0, streams[1] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx, XGrad.dhdy);
		gradientSM << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx, XGrad.dhdy);
		CUDA_CHECK(cudaDeviceSynchronize());

		//gradient << < gridDim, blockDim, 0, streams[2] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
		gradientSM << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
		CUDA_CHECK(cudaDeviceSynchronize());

		//gradient << < gridDim, blockDim, 0, streams[3] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx, XGrad.dudy);
		gradientSM << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx, XGrad.dudy);
		CUDA_CHECK(cudaDeviceSynchronize());

		//gradient << < gridDim, blockDim, 0, streams[0] >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx, XGrad.dvdy);
		gradientSM << < gridDim, blockDim, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx, XGrad.dvdy);
		CUDA_CHECK(cudaDeviceSynchronize());
		*/
		/*
		const int num_streams = 16;
		
		cudaStream_t streams[num_streams];

		for (int i = 0; i < num_streams; i++)
		{
			CUDA_CHECK(cudaStreamCreate(&streams[i]));
		}
		*/
		
		
		
		gradientedgeX << < gridDim, blockDimLR2, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx);
		//CUDA_CHECK(cudaDeviceSynchronize());
		//gradientedgeX << < gridDim, blockDimLR, 0 >> > (XParam.halowidth, XParam.blkwidth-1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdx);
		//CUDA_CHECK(cudaDeviceSynchronize());

		gradientedgeY << < gridDim, blockDimBT2, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdy);
		//CUDA_CHECK(cudaDeviceSynchronize());
		//gradientedgeY << < gridDim, blockDimBT, 0>> > (XParam.halowidth, XParam.blkwidth-1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.h, XGrad.dhdy);
		//CUDA_CHECK(cudaDeviceSynchronize());

		gradientedgeX << < gridDim, blockDimLR2, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx);
		//CUDA_CHECK(cudaDeviceSynchronize());
		//gradientedgeX << < gridDim, blockDimLR, 0 >> > (XParam.halowidth, XParam.blkwidth - 1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdx);
		//CUDA_CHECK(cudaDeviceSynchronize());

		gradientedgeY << < gridDim, blockDimBT2, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdy);
		//CUDA_CHECK(cudaDeviceSynchronize());
		//gradientedgeY << < gridDim, blockDimBT, 0 >> > (XParam.halowidth, XParam.blkwidth - 1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.zs, XGrad.dzsdy);
		//CUDA_CHECK(cudaDeviceSynchronize());

		gradientedgeX << < gridDim, blockDimLR2, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx);
		//CUDA_CHECK(cudaDeviceSynchronize());
		//gradientedgeX << < gridDim, blockDimLR, 0 >> > (XParam.halowidth, XParam.blkwidth - 1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudx);
		//CUDA_CHECK(cudaDeviceSynchronize());

		gradientedgeY << < gridDim, blockDimBT2, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudy);
		//CUDA_CHECK(cudaDeviceSynchronize());
		//gradientedgeY << < gridDim, blockDimBT, 0 >> > (XParam.halowidth, XParam.blkwidth - 1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.u, XGrad.dudy);
		//CUDA_CHECK(cudaDeviceSynchronize());

		gradientedgeX << < gridDim, blockDimLR2, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx);
		//CUDA_CHECK(cudaDeviceSynchronize());
		//gradientedgeX << < gridDim, blockDimLR, 0 >> > (XParam.halowidth, XParam.blkwidth - 1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdx);
		//CUDA_CHECK(cudaDeviceSynchronize());

		gradientedgeY << < gridDim, blockDimBT2, 0 >> > (XParam.halowidth, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdy);
		//CUDA_CHECK(cudaDeviceSynchronize());
		//gradientedgeY << < gridDim, blockDimBT, 0 >> > (XParam.halowidth, XParam.blkwidth - 1, XBlock.active, XBlock.level, (T)XParam.theta, (T)XParam.dx, XEv.v, XGrad.dvdy);
		CUDA_CHECK(cudaDeviceSynchronize());
		
		/*
		for (int i = 0; i < num_streams; i++)
		{
			cudaStreamDestroy(streams[i]);
		}
		*/
		gradientHaloGPU(XParam, XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
		gradientHaloGPU(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
		gradientHaloGPU(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
		gradientHaloGPU(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);

		if (XParam.engine == 1)
		{
			//  wet slope limiter
			WetsloperesetXGPU << < gridDim, blockDim, 0 >> > (XParam, XBlock, XEv, XGrad, zb);
			CUDA_CHECK(cudaDeviceSynchronize());

			WetsloperesetYGPU << < gridDim, blockDim, 0 >> > (XParam, XBlock, XEv, XGrad, zb);
			CUDA_CHECK(cudaDeviceSynchronize());

			// ALso do the slope limiter on the halo
			WetsloperesetHaloLeftGPU << < gridDim, blockDimLR, 0 >> > (XParam, XBlock, XEv, XGrad, zb);
			CUDA_CHECK(cudaDeviceSynchronize());

			WetsloperesetHaloRightGPU << < gridDim, blockDimLR, 0 >> > (XParam, XBlock, XEv, XGrad, zb);
			CUDA_CHECK(cudaDeviceSynchronize());

			WetsloperesetHaloBotGPU << < gridDim, blockDimBT, 0 >> > (XParam, XBlock, XEv, XGrad, zb);
			CUDA_CHECK(cudaDeviceSynchronize());

			WetsloperesetHaloTopGPU << < gridDim, blockDimBT, 0 >> > (XParam, XBlock, XEv, XGrad, zb);

			CUDA_CHECK(cudaDeviceSynchronize());
		}

	}
	//conserveElevationGradHaloGPU(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	//conserveElevationGradHaloGPU(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	//conserveElevationGradHaloGPU(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);

}
template void gradientGPUnew<float>(Param XParam, BlockP<float>XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, float* zb);
template void gradientGPUnew<double>(Param XParam, BlockP<double>XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, double* zb);


/*! \fn void gradient(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dadx, T* dady)
* Device kernel for calculating grdients for an evolving poarameter using the minmod limiter
* 
*/
template <class T> __global__ void gradient(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dadx, T* dady)
{
	//int *leftblk,int *rightblk,int* topblk, int * botblk,

	//int ix = threadIdx.x+1;
	//int iy = threadIdx.y+1;
	int blkmemwidth = blockDim.x + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = active[ibl];

	int lev = level[ib];

	T delta = calcres(dx, lev);


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	//int iright, ileft, ibot;
	// shared array index to make the code bit more readable
	//unsigned int sx = ix + halowidth;
	//unsigned int sy = iy + halowidth;


	T a_l, a_t, a_r, a_b,a_i;

	a_i = a[i];


	a_l = a[memloc(halowidth, blkmemwidth, ix - 1, iy, ib)];
	a_t = a[memloc(halowidth, blkmemwidth, ix , iy + 1, ib)];
	a_r = a[memloc(halowidth, blkmemwidth, ix + 1, iy, ib)];
	a_b = a[memloc(halowidth, blkmemwidth, ix, iy - 1, ib)];
	//__shared__ T a_s[18][18];



	//__syncthreads();
	//__syncwarp;

	dadx[i] = minmod2(theta, a_l, a_i, a_r) / delta;
	
	dady[i] = minmod2(theta, a_b, a_i, a_t) / delta;


}
template __global__ void gradient<float>(int halowidth, int* active, int* level, float theta, float dx, float* a, float* dadx, float* dady);
template __global__ void gradient<double>(int halowidth, int* active, int* level, double theta, double dx, double* a, double* dadx, double* dady);



/*! \fn void gradientSM(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dadx, T* dady)
* Depreciated shared memory version of Device kernel for calculating gradients
* Much slower than above
*/
template <class T> __global__ void gradientSM(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dadx, T* dady)
{
	//int *leftblk,int *rightblk,int* topblk, int * botblk,

	//int ix = threadIdx.x+1;
	//int iy = threadIdx.y+1;
	int blkmemwidth = blockDim.x + halowidth * 2;
	int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = active[ibl];

	int lev = level[ib];

	T delta = calcres(dx, lev);


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	int iright, ileft, itop, ibot;
	// shared array index to make the code bit more readable
	int sx = ix + halowidth;
	int sy = iy + halowidth;



	__shared__ T a_s[18][18];
	


	a_s[sx][sy] = a[i];
	
	//syncthread is needed here ?
		

	// read the halo around the tile
	if (threadIdx.x == blockDim.x - 1)
	{
		iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
		a_s[sx + 1][sy] = a[iright];
		
	}
	

	if (threadIdx.x == 0)
	{
		ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);;
		a_s[sx - 1][sy] = a[ileft];
		
	}

	if (threadIdx.y == blockDim.y - 1)
	{
		itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);;
		a_s[sx][sy + 1] = a[itop];
		
	}
	
	if (threadIdx.y == 0)
	{
		ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
		a_s[sx][sy - 1] = a[ibot];
		
	}

	__syncthreads();



	dadx[i] = minmod2(theta, a_s[sx - 1][sy], a_s[sx][sy], a_s[sx + 1][sy]) / delta;
	
	dady[i] = minmod2(theta, a_s[sx][sy - 1], a_s[sx][sy], a_s[sx][sy + 1]) / delta;


}
template __global__ void gradientSM<float>(int halowidth, int* active, int* level, float theta, float dx, float* a, float* dadx, float* dady);
template __global__ void gradientSM<double>(int halowidth, int* active, int* level, double theta, double dx, double* a, double* dadx, double* dady);

template <class T> __global__ void gradientSMB(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dadx, T* dady)
{
	//int *leftblk,int *rightblk,int* topblk, int * botblk,

	//int ix = threadIdx.x+1;
	//int iy = threadIdx.y+1;
	int blkmemwidth = 18;
	int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x-1;
	int iy = threadIdx.y-1;
	int ibl = blockIdx.x;
	int ib = active[ibl];

	int lev = level[ib];

	T delta = calcres(dx, lev);


	
	int iright, ileft, itop, ibot;
	// shared array index to make the code bit more readable
	int sx = ix + 1;
	int sy = iy + 1;

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	//int o = memloc(halowidth, blkmemwidth, sx, sy, ib);

	__shared__ T a_s[18][18];




	a_s[sx][sy] = a[i];
	__syncthreads();
	//syncthread is needed here ?

	T aleft, aright, atop, abot;
	aleft = a_s[sx - 1][sy];
	aright = a_s[sx + 1][sy];
	atop = a_s[sx][sy + 1];
	abot = a_s[sx][sy - 1];


	if (ix >= 0 && ix < 16 && iy >=0 && iy < 16)
	{

		dadx[i] = minmod2(theta, aleft, a_s[sx][sy], aright) / delta;

		dady[i] = minmod2(theta, abot, a_s[sx][sy], atop) / delta;
	}

}
template __global__ void gradientSMB<float>(int halowidth, int* active, int* level, float theta, float dx, float* a, float* dadx, float* dady);
template __global__ void gradientSMB<double>(int halowidth, int* active, int* level, double theta, double dx, double* a, double* dadx, double* dady);





/*! \fn void gradientedgeX(int halowidth, int ix, int* active, int* level, T theta, T dx, T* a, T* dadx)
* Device kernel for calculating gradients for an evolving parameter using the minmod limiter only at specific column (i.e. fixed ix)
*
*/
template <class T> __global__ void gradientedgeX(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dadx)
{
	//int *leftblk,int *rightblk,int* topblk, int * botblk,

	//int ix = threadIdx.x+1;
	//int iy = threadIdx.y+1;
	int blkmemwidth = blockDim.x + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	//unsigned int ix = threadIdx.x;
	int ix;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = active[ibl];

	if (threadIdx.x == 0)
	{
		ix = 0;
	}
	else
	{
		ix = blockDim.y - 1;
	}
		

	int lev = level[ib];

	T delta = calcres(dx, lev);


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	//int iright, ileft, ibot;
	// shared array index to make the code bit more readable
	//unsigned int sx = ix + halowidth;
	//unsigned int sy = iy + halowidth;


	T a_l, a_r, a_i;

	a_i = a[i];


	a_l = a[memloc(halowidth, blkmemwidth, ix - 1, iy, ib)];
	//a_t = a[memloc(halowidth, blkmemwidth, ix, iy + 1, ib)];
	a_r = a[memloc(halowidth, blkmemwidth, ix + 1, iy, ib)];
	//a_b = a[memloc(halowidth, blkmemwidth, ix, iy - 1, ib)];
	//__shared__ T a_s[18][18];



	//__syncthreads();
	//__syncwarp;

	dadx[i] = minmod2(theta, a_l, a_i, a_r) / delta;




}
template __global__ void gradientedgeX<float>(int halowidth,  int* active, int* level, float theta, float dx, float* a, float* dadx);
template __global__ void gradientedgeX<double>(int halowidth, int* active, int* level, double theta, double dx, double* a, double* dadx);



/*! \fn void gradientedgeY(int halowidth, int iy, int* active, int* level, T theta, T dx, T* a, T* dady)
* Device kernel for calculating gradients for an evolving parameter using the minmod limiter only at specific row (i.e. fixed iy)
*
*/
template <class T> __global__ void gradientedgeY(int halowidth, int* active, int* level, T theta, T dx, T* a, T* dady)
{
	//int *leftblk,int *rightblk,int* topblk, int * botblk,

	//int ix = threadIdx.x+1;
	//int iy = threadIdx.y+1;
	int blkmemwidth = blockDim.x + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	int ix = threadIdx.x;
	int iy;
	//unsigned int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = active[ibl];

	int lev = level[ib];

	T delta = calcres(dx, lev);

	if (threadIdx.y == 0)
	{
		iy = 0;
	}
	else
	{
		iy = blockDim.y - 1;
	}


	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	//int iright, ileft, ibot;
	// shared array index to make the code bit more readable
	//unsigned int sx = ix + halowidth;
	//unsigned int sy = iy + halowidth;


	T  a_t, a_b, a_i;

	a_i = a[i];


	//a_l = a[memloc(halowidth, blkmemwidth, ix - 1, iy, ib)];
	a_t = a[memloc(halowidth, blkmemwidth, ix, iy + 1, ib)];
	//a_r = a[memloc(halowidth, blkmemwidth, ix + 1, iy, ib)];
	a_b = a[memloc(halowidth, blkmemwidth, ix, iy - 1, ib)];
	//__shared__ T a_s[18][18];



	//__syncthreads();
	//__syncwarp;

	//dadx[i] = minmod2(theta, a_l, a_i, a_r) / delta;

	dady[i] = minmod2(theta, a_b, a_i, a_t) / delta;


}
template __global__ void gradientedgeY<float>(int halowidth, int* active, int* level, float theta, float dx, float* a, float* dady);
template __global__ void gradientedgeY<double>(int halowidth, int* active, int* level, double theta, double dx, double* a, double* dady);




template <class T> void gradientC(Param XParam, BlockP<T> XBlock, T* a, T* dadx, T* dady)
{

	int i,ib;
	int xplus, xminus, yplus, yminus;

	T delta;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		delta = calcres(T(XParam.dx), XBlock.level[ib]);
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				i = memloc(XParam, ix,iy,ib);
				
				//
				xplus = memloc(XParam, ix+1, iy, ib);
				xminus = memloc(XParam, ix-1, iy, ib);
				yplus = memloc(XParam, ix, iy+1, ib);
				yminus = memloc(XParam, ix, iy-1, ib);

				dadx[i] = minmod2(T(XParam.theta), a[xminus], a[i], a[xplus]) / delta;
				dady[i] = minmod2(T(XParam.theta), a[yminus], a[i], a[yplus]) / delta;
			}


		}


	}



}
template void gradientC<float>(Param XParam, BlockP<float> XBlock, float* a, float* dadx, float* dady);
template void gradientC<double>(Param XParam, BlockP<double> XBlock, double* a, double* dadx, double* dady);

template <class T> void gradientCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{


	std::thread t0(&gradientC<T>, XParam, XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
	std::thread t1(&gradientC<T>, XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	std::thread t2(&gradientC<T>, XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	std::thread t3(&gradientC<T>, XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);

	t0.join();
	t1.join();
	t2.join();
	t3.join();

	//fillHalo(XParam, XBlock, XGrad);
	
	gradientHalo(XParam, XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
	gradientHalo(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	gradientHalo(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	gradientHalo(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);
	
	if (XParam.conserveElevation)
	{
		conserveElevationGradHalo(XParam, XBlock, XEv.h, XEv.zs, zb, XGrad.dhdx, XGrad.dzsdx, XGrad.dhdy, XGrad.dzsdy);

	}
	
	
	refine_linear(XParam,XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
	refine_linear(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	refine_linear(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);

	RecalculateZs(XParam, XBlock, XEv, zb);
				

	gradientHalo(XParam, XBlock, XEv.h, XGrad.dhdx, XGrad.dhdy);
	gradientHalo(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	gradientHalo(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	gradientHalo(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdy);

	if (XParam.conserveElevation)
	{
		conserveElevationGradHalo(XParam, XBlock, XEv.h, XEv.zs, zb, XGrad.dhdx, XGrad.dzsdx, XGrad.dhdy, XGrad.dzsdy);

	}
	
	if (XParam.engine == 1)
	{
		WetsloperesetCPU(XParam, XBlock, XEv, XGrad, zb);

		WetsloperesetHaloLeftCPU(XParam, XBlock, XEv, XGrad, zb);
		WetsloperesetHaloRightCPU(XParam, XBlock, XEv, XGrad, zb);
		WetsloperesetHaloBotCPU(XParam, XBlock, XEv, XGrad, zb);
		WetsloperesetHaloTopCPU(XParam, XBlock, XEv, XGrad, zb);
	}


	


	//conserveElevationGradHalo(XParam, XBlock, XEv.zs, XGrad.dzsdx, XGrad.dzsdy);
	//conserveElevationGradHalo(XParam, XBlock, XEv.u, XGrad.dudx, XGrad.dudy);
	//conserveElevationGradHalo(XParam, XBlock, XEv.v, XGrad.dvdx, XGrad.dvdyythhhhhhhhhg);


}
template void gradientCPU<float>(Param XParam, BlockP<float>XBlock, EvolvingP<float> XEv, GradientsP<float> XGrad, float * zb);
template void gradientCPU<double>(Param XParam, BlockP<double>XBlock, EvolvingP<double> XEv, GradientsP<double> XGrad, double * zb);

template <class T> void WetsloperesetCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{
	int i, ib;
	int xplus, xminus, yminus;

	T delta;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		delta = calcres(T(XParam.dx), XBlock.level[ib]);
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				i = memloc(XParam, ix, iy, ib);

				//
				xplus = memloc(XParam, ix + 1, iy, ib);
				xminus = memloc(XParam, ix - 1, iy, ib);
				//yplus = memloc(XParam, ix, iy + 1, ib);
				yminus = memloc(XParam, ix, iy - 1, ib);

				T dzsdxi = XGrad.dzsdx[i];
				T dzsdyi = XGrad.dzsdy[i];




				//Do X axis
				if (utils::sq(dzsdxi) > utils::sq(XGrad.dzbdx[i]))
				{
					T leftzs, rightzs;
					leftzs = XEv.zs[i] - XEv.h[i] - delta * T(0.5) * (dzsdxi - XGrad.dhdx[i]);
					rightzs = XEv.zs[i] - XEv.h[i] + delta * T(0.5) * (dzsdxi - XGrad.dhdx[i]);

					if (leftzs > XEv.zs[xminus] || rightzs > XEv.zs[xplus])
					{
						XGrad.dzsdx[i] = XGrad.dhdx[i] + XGrad.dzbdx[i];
					}

				}

				//Do Y axis
				if (utils::sq(dzsdyi) > utils::sq(XGrad.dzbdy[i]))
				{
					T botzs, topzs;
					botzs = XEv.zs[i] - XEv.h[i] - delta * T(0.5) * (dzsdyi - XGrad.dhdy[i]);
					topzs = XEv.zs[i] - XEv.h[i] + delta * T(0.5) * (dzsdyi - XGrad.dhdy[i]);

					if (botzs > XEv.zs[yminus] || topzs > XEv.zs[yminus])
					{
						XGrad.dzsdy[i] = XGrad.dhdy[i] + XGrad.dzbdy[i];
					}

				}


			}
		}
	}
}

template <class T> __global__ void WetsloperesetXGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{
	unsigned int blkmemwidth = blockDim.x + XParam.halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];

	T delta = calcres(XParam.dx, lev);


	int i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);

	int iright, ileft;
	iright = memloc(XParam.halowidth, blkmemwidth, ix + 1, iy, ib);
	ileft = memloc(XParam.halowidth, blkmemwidth, ix - 1, iy, ib);

	T dzsdxi = XGrad.dzsdx[i];


	if (utils::sq(dzsdxi) > utils::sq(XGrad.dzbdx[i]))
	{
		T leftzs, rightzs;
		leftzs = XEv.zs[i] - XEv.h[i] - delta * T(0.5) * (dzsdxi - XGrad.dhdx[i]);
		rightzs = XEv.zs[i] - XEv.h[i] + delta * T(0.5) * (dzsdxi - XGrad.dhdx[i]);

		if (leftzs > XEv.zs[ileft] || rightzs > XEv.zs[iright])
		{
			XGrad.dzsdx[i] = XGrad.dhdx[i] + XGrad.dzbdx[i];
		}

	}
	

}

template <class T> __global__ void WetsloperesetYGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{
	unsigned int blkmemwidth = blockDim.x + XParam.halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];

	T delta = calcres(XParam.dx, lev);


	int i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);

	int itop, ibot;
	itop = memloc(XParam.halowidth, blkmemwidth, ix, iy + 1, ib);
	ibot = memloc(XParam.halowidth, blkmemwidth, ix, iy - 1, ib);

	T dzsdyi = XGrad.dzsdy[i];


	if (utils::sq(dzsdyi) > utils::sq(XGrad.dzbdy[i]))
	{
		T botzs, topzs;
		botzs = XEv.zs[i] - XEv.h[i] - delta * T(0.5) * (dzsdyi - XGrad.dhdy[i]);
		topzs = XEv.zs[i] - XEv.h[i] + delta * T(0.5) * (dzsdyi - XGrad.dhdy[i]);

		if (botzs > XEv.zs[ibot] || topzs > XEv.zs[itop])
		{
			XGrad.dzsdy[i] = XGrad.dhdy[i] + XGrad.dzbdy[i];
		}

	}


}


template <class T> __global__ void WetsloperesetHaloLeftGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	//unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int ix = -1;
	int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int lev = XBlock.level[ib];


	T delta = calcres(XParam.dx, lev);

	T zsi, zsright, zsleft;

	int i, iright;
	i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);
	iright = memloc(XParam.halowidth, blkmemwidth, ix + 1, iy, ib);

	zsi = XEv.zs[i];
	zsright = XEv.zs[iright];

	int read, jj, ii, ir, it, itr;

	if (XBlock.LeftBot[ib] == ib)//The lower half is a boundary 
	{
		if (iy < (XParam.blkwidth / 2))
		{

			read = memloc(XParam.halowidth, blkmemwidth, 0, iy, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			zsleft = XEv.zs[read];
		}

		if (XBlock.LeftTop[ib] == ib) // boundary on the top half too
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, 0, iy, ib);

				zsleft = XEv.zs[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (iy >= (XParam.blkwidth / 2))
			{

				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj, XBlock.LeftTop[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj + 1, XBlock.LeftTop[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj + 1, XBlock.LeftTop[ib]);

				zsleft = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.LeftBot[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 2), iy, XBlock.LeftBot[ib]);
		zsleft = XEv.zs[read];

	}
	else if (XBlock.level[XBlock.LeftBot[ib]] > XBlock.level[ib])
	{

		if (iy < (XParam.blkwidth / 2))
		{

			jj = iy * 2;
			int bb = XBlock.LeftBot[ib];

			ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj, bb);
			ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj, bb);
			it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj + 1, bb);
			itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj + 1, bb);

			zsleft = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.LeftTop[ib] == ib)
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, 0, iy, ib);

				zsleft = XEv.zs[read];
			}
		}
		else
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//
				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj, XBlock.LeftTop[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj + 1, XBlock.LeftTop[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj + 1, XBlock.LeftTop[ib]);

				zsleft = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.LeftBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.RightBot[XBlock.LeftBot[ib]] == ib ? ceil(iy * (T)0.5) : ceil(iy * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(iy * (T)0.5) * 2 > iy ? T(0.25) : T(0.75);

		ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 1), jj, XBlock.LeftBot[ib]);
		ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 2), jj, XBlock.LeftBot[ib]);
		it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 1), jj - 1, XBlock.LeftBot[ib]);
		itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 2), jj - 1, XBlock.LeftBot[ib]);

		zsleft = BilinearInterpolation(XEv.zs[itr], XEv.zs[ir], XEv.zs[it], XEv.zs[ii], T(0.0), T(1.0), T(0.0), T(1.0), T(0.75), jr);
	}
	

	T dzsdxi = XGrad.dzsdx[i];


	if (utils::sq(dzsdxi) > utils::sq(XGrad.dzbdx[i]))
	{
		T leftzs, rightzs;
		leftzs = zsi - XEv.h[i] - delta * T(0.5) * (dzsdxi - XGrad.dhdx[i]);
		rightzs = zsi - XEv.h[i] + delta * T(0.5) * (dzsdxi - XGrad.dhdx[i]);

		if (leftzs > zsleft || rightzs > zsright)
		{
			XGrad.dzsdx[i] = XGrad.dhdx[i] + XGrad.dzbdx[i];
		}

	}


}

template <class T> void WetsloperesetHaloLeftCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{


	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	//unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int ix = -1;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{

		unsigned int ib = XBlock.active[ibl];

		int lev = XBlock.level[ib];


		T delta = calcres(XParam.dx, lev);

		T zsi, zsright, zsleft;

		for (int iy = 0; iy <= XParam.blkwidth; iy++)
		{

			int i, iright;
			i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);
			iright = memloc(XParam.halowidth, blkmemwidth, ix + 1, iy, ib);

			zsi = XEv.zs[i];
			zsright = XEv.zs[iright];

			int read, jj, ii, ir, it, itr;

			if (XBlock.LeftBot[ib] == ib)//The lower half is a boundary 
			{
				if (iy < (XParam.blkwidth / 2))
				{

					read = memloc(XParam.halowidth, blkmemwidth, 0, iy, ib);// or memloc(XParam, -1, j, ib) but they should be the same

					zsleft = XEv.zs[read];
				}

				if (XBlock.LeftTop[ib] == ib) // boundary on the top half too
				{
					if (iy >= (XParam.blkwidth / 2))
					{
						//

						read = memloc(XParam.halowidth, blkmemwidth, 0, iy, ib);

						zsleft = XEv.zs[read];
					}
				}
				else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
				{

					if (iy >= (XParam.blkwidth / 2))
					{

						jj = (iy - XParam.blkwidth / 2) * 2;
						ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj, XBlock.LeftTop[ib]);
						ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj, XBlock.LeftTop[ib]);
						it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj + 1, XBlock.LeftTop[ib]);
						itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj + 1, XBlock.LeftTop[ib]);

						zsleft = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);

					}
				}
			}
			else if (XBlock.level[ib] == XBlock.level[XBlock.LeftBot[ib]]) // LeftTop block does not exist
			{

				read = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 2), iy, XBlock.LeftBot[ib]);
				zsleft = XEv.zs[read];

			}
			else if (XBlock.level[XBlock.LeftBot[ib]] > XBlock.level[ib])
			{

				if (iy < (XParam.blkwidth / 2))
				{

					jj = iy * 2;
					int bb = XBlock.LeftBot[ib];

					ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj, bb);
					ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj, bb);
					it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj + 1, bb);
					itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj + 1, bb);

					zsleft = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
				}
				//now find out aboy lefttop block
				if (XBlock.LeftTop[ib] == ib)
				{
					if (iy >= (XParam.blkwidth / 2))
					{
						//

						read = memloc(XParam.halowidth, blkmemwidth, 0, iy, ib);

						zsleft = XEv.zs[read];
					}
				}
				else
				{
					if (iy >= (XParam.blkwidth / 2))
					{
						//
						jj = (iy - XParam.blkwidth / 2) * 2;
						ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj, XBlock.LeftTop[ib]);
						ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj, XBlock.LeftTop[ib]);
						it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj + 1, XBlock.LeftTop[ib]);
						itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj + 1, XBlock.LeftTop[ib]);

						zsleft = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
					}
				}

			}
			else if (XBlock.level[XBlock.LeftBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
			{
				jj = XBlock.RightBot[XBlock.LeftBot[ib]] == ib ? ftoi(ceil(iy * (T)0.5)) : ftoi(ceil(iy * (T)0.5) + XParam.blkwidth / 2);
				T jr = ceil(iy * (T)0.5) * 2 > iy ? T(0.25) : T(0.75);

				ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 1), jj, XBlock.LeftBot[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 2), jj, XBlock.LeftBot[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 1), jj - 1, XBlock.LeftBot[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 2), jj - 1, XBlock.LeftBot[ib]);

				zsleft = BilinearInterpolation(XEv.zs[itr], XEv.zs[ir], XEv.zs[it], XEv.zs[ii], T(0.0), T(1.0), T(0.0), T(1.0), T(0.75), jr);
			}


			T dzsdxi = XGrad.dzsdx[i];


			if (utils::sq(dzsdxi) > utils::sq(XGrad.dzbdx[i]))
			{
				T leftzs, rightzs;
				leftzs = zsi - XEv.h[i] - delta * T(0.5) * (dzsdxi - XGrad.dhdx[i]);
				rightzs = zsi - XEv.h[i] + delta * T(0.5) * (dzsdxi - XGrad.dhdx[i]);

				if (leftzs > zsleft || rightzs > zsright)
				{
					XGrad.dzsdx[i] = XGrad.dhdx[i] + XGrad.dzbdx[i];
				}

			}
		}

	}
}


template <class T> __global__ void WetsloperesetHaloRightGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	//unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int ix = XParam.blkwidth;
	int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];
	int i, jj, ii, ir, it, itr;
	int read;

	int lev = XBlock.level[ib];

	T delta = calcres(XParam.dx, lev);


	i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);

	int ileft;
	
	ileft = memloc(XParam.halowidth, blkmemwidth, ix - 1, iy, ib);

	T zsi, zsleft, zsright;

	zsi = XEv.zs[i];
	zsleft = XEv.zs[ileft];
	
	T dzsdxi = XGrad.dzsdx[i];


	if (XBlock.RightBot[ib] == ib)//The lower half is a boundary 
	{
		if (iy < (XParam.blkwidth / 2))
		{

			read = memloc(XParam.halowidth, blkmemwidth, XParam.blkwidth - 1, iy, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			zsright = XEv.zs[read];;
		}

		if (XBlock.RightTop[ib] == ib) // boundary on the top half too
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, XParam.blkwidth - 1, iy, ib);

				zsright = XEv.zs[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (iy >= (XParam.blkwidth / 2))
			{

				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, 3, jj, XBlock.RightTop[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, 2, jj, XBlock.RightTop[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, 3, jj + 1, XBlock.RightTop[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, 2, jj + 1, XBlock.RightTop[ib]);

				zsright = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.RightBot[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam.halowidth, blkmemwidth, 1, iy, XBlock.RightBot[ib]);
		zsright = XEv.zs[read];

	}
	else if (XBlock.level[XBlock.RightBot[ib]] > XBlock.level[ib])
	{

		if (iy < (XParam.blkwidth / 2))
		{

			jj = iy * 2;
			int bb = XBlock.RightBot[ib];

			ii = memloc(XParam.halowidth, blkmemwidth, 3, jj, bb);
			ir = memloc(XParam.halowidth, blkmemwidth, 2, jj, bb);
			it = memloc(XParam.halowidth, blkmemwidth, 3, jj + 1, bb);
			itr = memloc(XParam.halowidth, blkmemwidth, 2, jj + 1, bb);

			zsright = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.RightTop[ib] == ib)
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, XParam.blkwidth - 1, iy, ib);

				zsright = XEv.zs[read];
			}
		}
		else
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//
				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, 3, jj, XBlock.RightTop[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, 2, jj, XBlock.RightTop[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, 3, jj + 1, XBlock.RightTop[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, 2, jj + 1, XBlock.RightTop[ib]);

				zsright = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.RightBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.LeftBot[XBlock.RightBot[ib]] == ib ? ceil(iy * (T)0.5) : ceil(iy * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(iy * (T)0.5) * 2 > iy ? T(0.25) : T(0.75);

		ii = memloc(XParam.halowidth, blkmemwidth, 0, jj, XBlock.RightBot[ib]);
		ir = memloc(XParam.halowidth, blkmemwidth, 1, jj, XBlock.RightBot[ib]);
		it = memloc(XParam.halowidth, blkmemwidth, 0, jj - 1, XBlock.RightBot[ib]);
		itr = memloc(XParam.halowidth, blkmemwidth, 1, jj - 1, XBlock.RightBot[ib]);

		zsright = BilinearInterpolation(XEv.zs[it], XEv.zs[ii], XEv.zs[itr], XEv.zs[ir], T(0.0), T(1.0), T(0.0), T(1.0), T(0.25), jr);
	}







	if (utils::sq(dzsdxi) > utils::sq(XGrad.dzbdx[i]))
	{
		T leftzs, rightzs;
		leftzs = zsi - XEv.h[i] - delta * T(0.5) * (dzsdxi - XGrad.dhdx[i]);
		rightzs = zsi - XEv.h[i] + delta * T(0.5) * (dzsdxi - XGrad.dhdx[i]);

		if (leftzs > zsleft || rightzs > zsright)
		{
			XGrad.dzsdx[i] = XGrad.dhdx[i] + XGrad.dzbdx[i];
		}

	}


}

template <class T> void WetsloperesetHaloRightCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	//unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int ix = XParam.blkwidth;
	
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{


			
			unsigned int ib = XBlock.active[ibl];
			int i, jj, ii, ir, it, itr;
			int read;

			int lev = XBlock.level[ib];

			T delta = calcres(XParam.dx, lev);


			i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);

			int  ileft;

			ileft = memloc(XParam.halowidth, blkmemwidth, ix - 1, iy, ib);

			T zsi, zsleft, zsright;

			zsi = XEv.zs[i];
			zsleft = XEv.zs[ileft];

			T dzsdxi = XGrad.dzsdx[i];


			if (XBlock.RightBot[ib] == ib)//The lower half is a boundary 
			{
				if (iy < (XParam.blkwidth / 2))
				{

					read = memloc(XParam.halowidth, blkmemwidth, XParam.blkwidth - 1, iy, ib);// or memloc(XParam, -1, j, ib) but they should be the same

					zsright = XEv.zs[read];;
				}

				if (XBlock.RightTop[ib] == ib) // boundary on the top half too
				{
					if (iy >= (XParam.blkwidth / 2))
					{
						//

						read = memloc(XParam.halowidth, blkmemwidth, XParam.blkwidth - 1, iy, ib);

						zsright = XEv.zs[read];
					}
				}
				else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
				{

					if (iy >= (XParam.blkwidth / 2))
					{

						jj = (iy - XParam.blkwidth / 2) * 2;
						ii = memloc(XParam.halowidth, blkmemwidth, 3, jj, XBlock.RightTop[ib]);
						ir = memloc(XParam.halowidth, blkmemwidth, 2, jj, XBlock.RightTop[ib]);
						it = memloc(XParam.halowidth, blkmemwidth, 3, jj + 1, XBlock.RightTop[ib]);
						itr = memloc(XParam.halowidth, blkmemwidth, 2, jj + 1, XBlock.RightTop[ib]);

						zsright = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);

					}
				}
			}
			else if (XBlock.level[ib] == XBlock.level[XBlock.RightBot[ib]]) // LeftTop block does not exist
			{

				read = memloc(XParam.halowidth, blkmemwidth, 1, iy, XBlock.RightBot[ib]);
				zsright = XEv.zs[read];

			}
			else if (XBlock.level[XBlock.RightBot[ib]] > XBlock.level[ib])
			{

				if (iy < (XParam.blkwidth / 2))
				{

					jj = iy * 2;
					int bb = XBlock.RightBot[ib];

					ii = memloc(XParam.halowidth, blkmemwidth, 3, jj, bb);
					ir = memloc(XParam.halowidth, blkmemwidth, 2, jj, bb);
					it = memloc(XParam.halowidth, blkmemwidth, 3, jj + 1, bb);
					itr = memloc(XParam.halowidth, blkmemwidth, 2, jj + 1, bb);

					zsright = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
				}
				//now find out aboy lefttop block
				if (XBlock.RightTop[ib] == ib)
				{
					if (iy >= (XParam.blkwidth / 2))
					{
						//

						read = memloc(XParam.halowidth, blkmemwidth, XParam.blkwidth - 1, iy, ib);

						zsright = XEv.zs[read];
					}
				}
				else
				{
					if (iy >= (XParam.blkwidth / 2))
					{
						//
						jj = (iy - XParam.blkwidth / 2) * 2;
						ii = memloc(XParam.halowidth, blkmemwidth, 3, jj, XBlock.RightTop[ib]);
						ir = memloc(XParam.halowidth, blkmemwidth, 2, jj, XBlock.RightTop[ib]);
						it = memloc(XParam.halowidth, blkmemwidth, 3, jj + 1, XBlock.RightTop[ib]);
						itr = memloc(XParam.halowidth, blkmemwidth, 2, jj + 1, XBlock.RightTop[ib]);

						zsright = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
					}
				}

			}
			else if (XBlock.level[XBlock.RightBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
			{
				jj = XBlock.LeftBot[XBlock.RightBot[ib]] == ib ? ftoi(ceil(iy * (T)0.5)) : ftoi(ceil(iy * (T)0.5) + XParam.blkwidth / 2);
				T jr = ceil(iy * (T)0.5) * 2 > iy ? T(0.25) : T(0.75);

				ii = memloc(XParam.halowidth, blkmemwidth, 0, jj, XBlock.RightBot[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, 1, jj, XBlock.RightBot[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, 0, jj - 1, XBlock.RightBot[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, 1, jj - 1, XBlock.RightBot[ib]);

				zsright = BilinearInterpolation(XEv.zs[it], XEv.zs[ii], XEv.zs[itr], XEv.zs[ir], T(0.0), T(1.0), T(0.0), T(1.0), T(0.25), jr);
			}







			if (utils::sq(dzsdxi) > utils::sq(XGrad.dzbdx[i]))
			{
				T leftzs, rightzs;
				leftzs = zsi - XEv.h[i] - delta * T(0.5) * (dzsdxi - XGrad.dhdx[i]);
				rightzs = zsi - XEv.h[i] + delta * T(0.5) * (dzsdxi - XGrad.dhdx[i]);

				if (leftzs > zsleft || rightzs > zsright)
				{
					XGrad.dzsdx[i] = XGrad.dhdx[i] + XGrad.dzbdx[i];
				}

			}
		}

	}
}


template <class T> __global__ void WetsloperesetHaloBotGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	//unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int iy = -1;
	int ix = threadIdx.x;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i, jj, ii, ir, it, itr;

	int lev = XBlock.level[ib];

	T delta = calcres(XParam.dx, lev);


	i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);

	int itop,read;
	itop = memloc(XParam.halowidth, blkmemwidth, ix, iy + 1, ib);
	
	T zsi, zstop, zsbot;

	T dzsdyi = XGrad.dzsdy[i];

	zsi = XEv.zs[i];
	zstop = XEv.zs[itop];


	if (XBlock.BotLeft[ib] == ib)//The lower half is a boundary 
	{
		if (ix < (XParam.blkwidth / 2))
		{

			read = memloc(XParam.halowidth, blkmemwidth, ix, 0, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			zsbot = XEv.zs[read];

		}

		if (XBlock.BotRight[ib] == ib) // boundary on the top half too
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, ix, 0, ib);

				zsbot = XEv.zs[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (ix >= (XParam.blkwidth / 2))
			{

				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 4), XBlock.BotRight[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 4), XBlock.BotRight[ib]);

				zsbot = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.BotLeft[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam.halowidth, blkmemwidth, ix, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);
		zsbot = XEv.zs[read];



	}
	else if (XBlock.level[XBlock.BotLeft[ib]] > XBlock.level[ib])
	{

		if (ix < (XParam.blkwidth / 2))
		{

			jj = ix * 2;
			int bb = XBlock.BotLeft[ib];

			ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 3), bb);
			ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 4), bb);
			it = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 3), bb);
			itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 4), bb);

			zsbot = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.BotRight[ib] == ib)
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, ix, 0, ib);

				zsbot = XEv.zs[read];
			}
		}
		else
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//
				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 4), XBlock.BotRight[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 4), XBlock.BotRight[ib]);

				zsbot = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.BotLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.TopLeft[XBlock.BotLeft[ib]] == ib ? ceil(ix * (T)0.5) : ceil(ix * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(ix * (T)0.5) * 2 > ix ? T(0.25) : T(0.75);

		ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
		ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);
		it = memloc(XParam.halowidth, blkmemwidth, jj - 1, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
		itr = memloc(XParam.halowidth, blkmemwidth, jj - 1, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);

		zsbot = BilinearInterpolation(XEv.zs[itr], XEv.zs[it], XEv.zs[ir], XEv.zs[ii], T(0.0), T(1.0), T(0.0), T(1.0), jr, T(0.75));
	}


	if (utils::sq(dzsdyi) > utils::sq(XGrad.dzbdy[i]))
	{
		T botzs, topzs;
		botzs = zsi - XEv.h[i] - delta * T(0.5) * (dzsdyi - XGrad.dhdy[i]);
		topzs = zsi - XEv.h[i] + delta * T(0.5) * (dzsdyi - XGrad.dhdy[i]);

		if (botzs > zsbot || topzs > zstop)
		{
			XGrad.dzsdy[i] = XGrad.dhdy[i] + XGrad.dzbdy[i];
		}

	}


}

template <class T> void WetsloperesetHaloBotCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	//unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int iy = -1;
	

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{

		
		unsigned int ib = XBlock.active[ibl];

		 

		int i, jj, ii, ir, it, itr;

		int lev = XBlock.level[ib];

		T delta = calcres(XParam.dx, lev);

		for (int ix = 0; ix < XParam.blkwidth; ix++)
		{


			i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);

			int itop, read;
			itop = memloc(XParam.halowidth, blkmemwidth, ix, iy + 1, ib);

			T zsi, zstop, zsbot;

			T dzsdyi = XGrad.dzsdy[i];

			zsi = XEv.zs[i];
			zstop = XEv.zs[itop];


			if (XBlock.BotLeft[ib] == ib)//The lower half is a boundary 
			{
				if (ix < (XParam.blkwidth / 2))
				{

					read = memloc(XParam.halowidth, blkmemwidth, ix, 0, ib);// or memloc(XParam, -1, j, ib) but they should be the same

					zsbot = XEv.zs[read];

				}

				if (XBlock.BotRight[ib] == ib) // boundary on the top half too
				{
					if (ix >= (XParam.blkwidth / 2))
					{
						//

						read = memloc(XParam.halowidth, blkmemwidth, ix, 0, ib);

						zsbot = XEv.zs[read];
					}
				}
				else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
				{

					if (ix >= (XParam.blkwidth / 2))
					{

						jj = (ix - XParam.blkwidth / 2) * 2;
						ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
						ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 4), XBlock.BotRight[ib]);
						it = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
						itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 4), XBlock.BotRight[ib]);

						zsbot = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);

					}
				}
			}
			else if (XBlock.level[ib] == XBlock.level[XBlock.BotLeft[ib]]) // LeftTop block does not exist
			{

				read = memloc(XParam.halowidth, blkmemwidth, ix, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);
				zsbot = XEv.zs[read];



			}
			else if (XBlock.level[XBlock.BotLeft[ib]] > XBlock.level[ib])
			{

				if (ix < (XParam.blkwidth / 2))
				{

					jj = ix * 2;
					int bb = XBlock.BotLeft[ib];

					ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 3), bb);
					ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 4), bb);
					it = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 3), bb);
					itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 4), bb);

					zsbot = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
				}
				//now find out aboy lefttop block
				if (XBlock.BotRight[ib] == ib)
				{
					if (ix >= (XParam.blkwidth / 2))
					{
						//

						read = memloc(XParam.halowidth, blkmemwidth, ix, 0, ib);

						zsbot = XEv.zs[read];
					}
				}
				else
				{
					if (ix >= (XParam.blkwidth / 2))
					{
						//
						jj = (ix - XParam.blkwidth / 2) * 2;
						ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
						ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 4), XBlock.BotRight[ib]);
						it = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
						itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 4), XBlock.BotRight[ib]);

						zsbot = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
					}
				}

			}
			else if (XBlock.level[XBlock.BotLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
			{
				jj = XBlock.TopLeft[XBlock.BotLeft[ib]] == ib ? ftoi(ceil(ix * (T)0.5)) : ftoi(ceil(ix * (T)0.5) + XParam.blkwidth / 2);
				T jr = ceil(ix * (T)0.5) * 2 > ix ? T(0.25) : T(0.75);

				ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, jj - 1, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, jj - 1, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);

				zsbot = BilinearInterpolation(XEv.zs[itr], XEv.zs[it], XEv.zs[ir], XEv.zs[ii], T(0.0), T(1.0), T(0.0), T(1.0), jr, T(0.75));
			}


			if (utils::sq(dzsdyi) > utils::sq(XGrad.dzbdy[i]))
			{
				T botzs, topzs;
				botzs = zsi - XEv.h[i] - delta * T(0.5) * (dzsdyi - XGrad.dhdy[i]);
				topzs = zsi - XEv.h[i] + delta * T(0.5) * (dzsdyi - XGrad.dhdy[i]);

				if (botzs > zsbot || topzs > zstop)
				{
					XGrad.dzsdy[i] = XGrad.dhdy[i] + XGrad.dzbdy[i];
				}

			}
		}

	}

}

template <class T> __global__ void WetsloperesetHaloTopGPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	//unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int iy = XParam.blkwidth;
	int ix = threadIdx.x;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i, jj, ii, ir, it, itr;

	int lev = XBlock.level[ib];

	T delta = calcres(XParam.dx, lev);


	i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);

	int ibot, read;
	
	ibot = memloc(XParam.halowidth, blkmemwidth, ix, iy - 1, ib);

	T zsi, zstop, zsbot;

	zsi = XEv.zs[i];
	zsbot = XEv.zs[ibot];

	T dzsdyi = XGrad.dzsdy[i];


	if (XBlock.TopLeft[ib] == ib)//The lower half is a boundary 
	{
		if (ix < (XParam.blkwidth / 2))
		{

			read = memloc(XParam.halowidth, blkmemwidth, ix, XParam.blkwidth - 1, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			zstop = XEv.zs[read];

		}

		if (XBlock.TopRight[ib] == ib) // boundary on the top half too
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, ix, XParam.blkwidth - 1, ib);

				zstop = XEv.zs[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (ix >= (XParam.blkwidth / 2))
			{

				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, jj, 3, XBlock.TopRight[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, jj, 2, XBlock.TopRight[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, jj + 1, 3, XBlock.TopRight[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, 2, XBlock.TopRight[ib]);

				zstop = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.TopLeft[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam.halowidth, blkmemwidth, ix, 1, XBlock.TopLeft[ib]);
		zstop = XEv.zs[read];



	}
	else if (XBlock.level[XBlock.TopLeft[ib]] > XBlock.level[ib])
	{

		if (ix < (XParam.blkwidth / 2))
		{

			jj = ix * 2;
			int bb = XBlock.TopLeft[ib];;

			ii = memloc(XParam.halowidth, blkmemwidth, jj, 3, bb);
			ir = memloc(XParam.halowidth, blkmemwidth, jj, 2, bb);
			it = memloc(XParam.halowidth, blkmemwidth, jj + 1, 3, bb);
			itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, 2, bb);

			zstop = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.TopRight[ib] == ib)
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, ix, XParam.blkwidth - 1, ib);

				zstop = XEv.zs[read];
			}
		}
		else
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//
				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, jj, 3, XBlock.TopRight[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, jj, 2, XBlock.TopRight[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, jj + 1, 3, XBlock.TopRight[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, 2, XBlock.TopRight[ib]);

				zstop = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.TopLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.BotLeft[XBlock.TopLeft[ib]] == ib ? ceil(ix * (T)0.5) : ceil(ix * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(ix * (T)0.5) * 2 > ix ? T(0.25) : T(0.75);

		ii = memloc(XParam.halowidth, blkmemwidth, jj, 0, XBlock.TopLeft[ib]);
		ir = memloc(XParam.halowidth, blkmemwidth, jj, 1, XBlock.TopLeft[ib]);
		it = memloc(XParam.halowidth, blkmemwidth, jj - 1, 0, XBlock.TopLeft[ib]);
		itr = memloc(XParam.halowidth, blkmemwidth, jj - 1, 1, XBlock.TopLeft[ib]);

		zstop = BilinearInterpolation(XEv.zs[it], XEv.zs[itr], XEv.zs[ii], XEv.zs[ir], T(0.0), T(1.0), T(0.0), T(1.0), jr, T(0.25));
	}


	if (utils::sq(dzsdyi) > utils::sq(XGrad.dzbdy[i]))
	{
		T botzs, topzs;
		botzs = zsi - XEv.h[i] - delta * T(0.5) * (dzsdyi - XGrad.dhdy[i]);
		topzs = zsi - XEv.h[i] + delta * T(0.5) * (dzsdyi - XGrad.dhdy[i]);

		if (botzs > zsbot || topzs > zstop)
		{
			XGrad.dzsdy[i] = XGrad.dhdy[i] + XGrad.dzbdy[i];
		}

	}


}



template <class T>  void WetsloperesetHaloTopCPU(Param XParam, BlockP<T>XBlock, EvolvingP<T> XEv, GradientsP<T> XGrad, T* zb)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	//unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int iy = XParam.blkwidth;
	

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		unsigned int ib = XBlock.active[ibl];

		int i, jj, ii, ir, it, itr;

		int lev = XBlock.level[ib];

		T delta = calcres(XParam.dx, lev);

		for (int ix = 0; ix < XParam.blkwidth; ix++)
		{

			i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);

			int ibot, read;

			ibot = memloc(XParam.halowidth, blkmemwidth, ix, iy - 1, ib);

			T zsi, zstop, zsbot;

			zsi = XEv.zs[i];
			zsbot = XEv.zs[ibot];

			T dzsdyi = XGrad.dzsdy[i];


			if (XBlock.TopLeft[ib] == ib)//The lower half is a boundary 
			{
				if (ix < (XParam.blkwidth / 2))
				{

					read = memloc(XParam.halowidth, blkmemwidth, ix, XParam.blkwidth - 1, ib);// or memloc(XParam, -1, j, ib) but they should be the same

					zstop = XEv.zs[read];

				}

				if (XBlock.TopRight[ib] == ib) // boundary on the top half too
				{
					if (ix >= (XParam.blkwidth / 2))
					{
						//

						read = memloc(XParam.halowidth, blkmemwidth, ix, XParam.blkwidth - 1, ib);

						zstop = XEv.zs[read];
					}
				}
				else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
				{

					if (ix >= (XParam.blkwidth / 2))
					{

						jj = (ix - XParam.blkwidth / 2) * 2;
						ii = memloc(XParam.halowidth, blkmemwidth, jj, 3, XBlock.TopRight[ib]);
						ir = memloc(XParam.halowidth, blkmemwidth, jj, 2, XBlock.TopRight[ib]);
						it = memloc(XParam.halowidth, blkmemwidth, jj + 1, 3, XBlock.TopRight[ib]);
						itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, 2, XBlock.TopRight[ib]);

						zstop = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);

					}
				}
			}
			else if (XBlock.level[ib] == XBlock.level[XBlock.TopLeft[ib]]) // LeftTop block does not exist
			{

				read = memloc(XParam.halowidth, blkmemwidth, ix, 1, XBlock.TopLeft[ib]);
				zstop = XEv.zs[read];



			}
			else if (XBlock.level[XBlock.TopLeft[ib]] > XBlock.level[ib])
			{

				if (ix < (XParam.blkwidth / 2))
				{

					jj = ix * 2;
					int bb = XBlock.TopLeft[ib];;

					ii = memloc(XParam.halowidth, blkmemwidth, jj, 3, bb);
					ir = memloc(XParam.halowidth, blkmemwidth, jj, 2, bb);
					it = memloc(XParam.halowidth, blkmemwidth, jj + 1, 3, bb);
					itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, 2, bb);

					zstop = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
				}
				//now find out aboy lefttop block
				if (XBlock.TopRight[ib] == ib)
				{
					if (ix >= (XParam.blkwidth / 2))
					{
						//

						read = memloc(XParam.halowidth, blkmemwidth, ix, XParam.blkwidth - 1, ib);

						zstop = XEv.zs[read];
					}
				}
				else
				{
					if (ix >= (XParam.blkwidth / 2))
					{
						//
						jj = (ix - XParam.blkwidth / 2) * 2;
						ii = memloc(XParam.halowidth, blkmemwidth, jj, 3, XBlock.TopRight[ib]);
						ir = memloc(XParam.halowidth, blkmemwidth, jj, 2, XBlock.TopRight[ib]);
						it = memloc(XParam.halowidth, blkmemwidth, jj + 1, 3, XBlock.TopRight[ib]);
						itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, 2, XBlock.TopRight[ib]);

						zstop = T(0.25) * (XEv.zs[ii] + XEv.zs[ir] + XEv.zs[it] + XEv.zs[itr]);
					}
				}

			}
			else if (XBlock.level[XBlock.TopLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
			{
				jj = XBlock.BotLeft[XBlock.TopLeft[ib]] == ib ? ftoi(ceil(ix * (T)0.5)) : ftoi(ceil(ix * (T)0.5) + XParam.blkwidth / 2);
				T jr = ceil(ix * (T)0.5) * 2 > ix ? T(0.25) : T(0.75);

				ii = memloc(XParam.halowidth, blkmemwidth, jj, 0, XBlock.TopLeft[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, jj, 1, XBlock.TopLeft[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, jj - 1, 0, XBlock.TopLeft[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, jj - 1, 1, XBlock.TopLeft[ib]);

				zstop = BilinearInterpolation(XEv.zs[it], XEv.zs[itr], XEv.zs[ii], XEv.zs[ir], T(0.0), T(1.0), T(0.0), T(1.0), jr, T(0.25));
			}


			if (utils::sq(dzsdyi) > utils::sq(XGrad.dzbdy[i]))
			{
				T botzs, topzs;
				botzs = zsi - XEv.h[i] - delta * T(0.5) * (dzsdyi - XGrad.dhdy[i]);
				topzs = zsi - XEv.h[i] + delta * T(0.5) * (dzsdyi - XGrad.dhdy[i]);

				if (botzs > zsbot || topzs > zstop)
				{
					XGrad.dzsdy[i] = XGrad.dhdy[i] + XGrad.dzbdy[i];
				}

			}
		}
	}


}


template <class T> void gradientHalo(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady)
{
	int ib;
	//int xplus;

	//T delta;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			gradientHaloLeft(XParam, XBlock, ib, iy, a, dadx, dady);
			gradientHaloRight(XParam, XBlock, ib, iy, a, dadx, dady);
		}
		for (int ix = 0; ix < XParam.blkwidth; ix++)
		{
			gradientHaloBot(XParam, XBlock, ib, ix, a, dadx, dady);
			gradientHaloTop(XParam, XBlock, ib, ix, a, dadx, dady);
		}
	}
}


template <class T> void gradientHaloGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady)
{
	dim3 blockDimL(1, XParam.blkwidth, 1);
	dim3 blockDimB(XParam.blkwidth, 1, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	
	gradientHaloLeftGPU << < gridDim, blockDimL, 0 >> > (XParam, XBlock, a, dadx, dady);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientHaloRightGPU << < gridDim, blockDimL, 0 >> > (XParam, XBlock, a, dadx, dady);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientHaloBotGPU << < gridDim, blockDimB, 0 >> > (XParam, XBlock, a, dadx, dady);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientHaloTopGPU << < gridDim, blockDimB, 0 >> > (XParam, XBlock, a, dadx, dady);
	CUDA_CHECK(cudaDeviceSynchronize());
		
	
}


template <class T> void gradientHaloLeft(Param XParam, BlockP<T>XBlock, int ib, int iy, T* a, T* dadx, T* dady)
{
	int i, ix, jj, ii, ir, it, itr;
	int xplus, read;
	
	T delta, aright, aleft;

	ix = -1;

	i = memloc(XParam, ix, iy, ib);
	xplus = memloc(XParam, ix + 1, iy, ib);
	

	aright = a[xplus];
	


	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.LeftBot[ib] == ib)//The lower half is a boundary 
	{
		if ( iy < (XParam.blkwidth / 2))
		{

			read = memloc(XParam, 0, iy, ib);// or memloc(XParam, -1, j, ib) but they should be the same
			
			aleft = a[read];
		}

		if (XBlock.LeftTop[ib] == ib) // boundary on the top half too
		{
			if ( iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, 0, iy, ib);
				
				aleft = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (iy >= (XParam.blkwidth / 2))
			{
				
				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, (XParam.blkwidth - 3), jj, XBlock.LeftTop[ib]);
				ir = memloc(XParam, (XParam.blkwidth - 4), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam, (XParam.blkwidth - 3), jj + 1, XBlock.LeftTop[ib]);
				itr = memloc(XParam, (XParam.blkwidth - 4), jj + 1, XBlock.LeftTop[ib]);

				aleft = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.LeftBot[ib]]) // LeftTop block does not exist
	{
		
			read = memloc(XParam, (XParam.blkwidth - 2), iy, XBlock.LeftBot[ib]);
			aleft = a[read];
		
	}
	else if (XBlock.level[XBlock.LeftBot[ib]] > XBlock.level[ib])
	{

		if (iy < (XParam.blkwidth / 2))
		{	

			jj = iy * 2;
			int bb = XBlock.LeftBot[ib];

			ii = memloc(XParam, (XParam.blkwidth - 3), jj, bb);
			ir = memloc(XParam, (XParam.blkwidth - 4), jj, bb);
			it = memloc(XParam, (XParam.blkwidth - 3), jj + 1, bb);
			itr = memloc(XParam, (XParam.blkwidth - 4), jj + 1, bb);

			aleft = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.LeftTop[ib] == ib)
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, 0, iy, ib);
				
				aleft = a[read];
			}
		}
		else
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//
				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, (XParam.blkwidth - 3), jj, XBlock.LeftTop[ib]);
				ir = memloc(XParam, (XParam.blkwidth - 4), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam, (XParam.blkwidth - 3), jj + 1, XBlock.LeftTop[ib]);
				itr = memloc(XParam, (XParam.blkwidth - 4), jj + 1, XBlock.LeftTop[ib]);

				aleft = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.LeftBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.RightBot[XBlock.LeftBot[ib]] == ib ? ftoi(ceil(iy * (T)0.5)) : ftoi(ceil(iy * (T)0.5) + XParam.blkwidth / 2);
		//T jr = ceil(iy * (T)0.5) * 2 > iy ? T(0.75) : T(0.25);// This is the wrong way around
		T jr = ceil(iy * (T)0.5) * 2 > iy ? T(0.25) : T(0.75); // This is right (e.g. at iy==0 use 0.75 at iy==1 use 0.25)

		ii = memloc(XParam, (XParam.blkwidth - 1), jj, XBlock.LeftBot[ib]);
		ir = memloc(XParam, (XParam.blkwidth - 2), jj, XBlock.LeftBot[ib]);
		it = memloc(XParam, (XParam.blkwidth - 1), jj - 1, XBlock.LeftBot[ib]);
		itr = memloc(XParam, (XParam.blkwidth - 2), jj - 1, XBlock.LeftBot[ib]);

		aleft = BilinearInterpolation(a[itr], a[ir], a[it], a[ii], T(0.0), T(1.0), T(0.0), T(1.0), T(0.75), jr);
	}
	




	dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	//dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}


template <class T> void gradientHaloRight(Param XParam, BlockP<T>XBlock, int ib, int iy, T* a, T* dadx, T* dady)
{
	int i, ix, jj, ii, ir, it, itr;
	int xminus, read;

	T delta, aright, aleft;

	ix = 16;

	i = memloc(XParam, ix, iy, ib);
	xminus = memloc(XParam, ix - 1, iy, ib);


	aleft = a[xminus];



	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.RightBot[ib] == ib)//The lower half is a boundary 
	{
		if (iy < (XParam.blkwidth / 2))
		{

			read = memloc(XParam, XParam.blkwidth -1, iy, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			aright = a[read];
		}

		if (XBlock.RightTop[ib] == ib) // boundary on the top half too
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, XParam.blkwidth - 1, iy, ib);

				aright = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of righttopib is levelib+1
		{

			if (iy >= (XParam.blkwidth / 2))
			{

				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, 3, jj, XBlock.RightTop[ib]);
				ir = memloc(XParam, 2, jj, XBlock.RightTop[ib]);
				it = memloc(XParam, 3, jj + 1, XBlock.RightTop[ib]);
				itr = memloc(XParam, 2, jj + 1, XBlock.RightTop[ib]);

				aright = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.RightBot[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam, 1, iy, XBlock.RightBot[ib]);
		aright = a[read];

	}
	else if (XBlock.level[XBlock.RightBot[ib]] > XBlock.level[ib])
	{

		if (iy < (XParam.blkwidth / 2))
		{

			jj = iy * 2;
			int bb = XBlock.RightBot[ib];

			ii = memloc(XParam, 3, jj, bb);
			ir = memloc(XParam, 2, jj, bb);
			it = memloc(XParam, 3, jj + 1, bb);
			itr = memloc(XParam, 2, jj + 1, bb);

			aright = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.RightTop[ib] == ib)
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, XParam.blkwidth - 1, iy, ib);

				aright = a[read];
			}
		}
		else
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//
				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, 3, jj, XBlock.RightTop[ib]);
				ir = memloc(XParam, 2, jj, XBlock.RightTop[ib]);
				it = memloc(XParam, 3, jj + 1, XBlock.RightTop[ib]);
				itr = memloc(XParam, 2, jj + 1, XBlock.RightTop[ib]);

				aright = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.RightBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.LeftBot[XBlock.RightBot[ib]] == ib ? ftoi(ceil(iy * (T)0.5)) : ftoi(ceil(iy * (T)0.5) + XParam.blkwidth / 2);
		T jr = ceil(iy * (T)0.5) * 2 > iy ? T(0.25) : T(0.75);

		ii = memloc(XParam, 0, jj, XBlock.RightBot[ib]);
		ir = memloc(XParam, 1, jj, XBlock.RightBot[ib]);
		it = memloc(XParam, 0, jj - 1, XBlock.RightBot[ib]);
		itr = memloc(XParam, 1, jj - 1, XBlock.RightBot[ib]);

		aright = BilinearInterpolation(a[it], a[ii], a[itr], a[ir], T(0.0), T(1.0), T(0.0), T(1.0), T(0.25), jr);
	}





	dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	//dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}


template <class T> void gradientHaloBot(Param XParam, BlockP<T>XBlock, int ib, int ix, T* a, T* dadx, T* dady)
{
	int i, iy, jj, ii, ir, it, itr;
	int yplus, read;

	T delta, atop, abot;

	iy = -1;

	i = memloc(XParam, ix, iy, ib);
	yplus = memloc(XParam, ix , iy + 1, ib);
	



	atop = a[yplus];



	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.BotLeft[ib] == ib)//The lower half is a boundary 
	{
		if (ix < (XParam.blkwidth / 2))
		{

			read = memloc(XParam, ix, 0, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			abot = a[read];
			
		}

		if (XBlock.BotRight[ib] == ib) // boundary on the top half too
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, ix, 0, ib);

				abot = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (ix >= (XParam.blkwidth / 2))
			{

				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, jj, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				ir = memloc(XParam, jj, (XParam.blkwidth - 4), XBlock.BotRight[ib]);
				it = memloc(XParam, jj + 1, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				itr = memloc(XParam, jj + 1, (XParam.blkwidth - 4), XBlock.BotRight[ib]);

				abot = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.BotLeft[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam, ix, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);
		abot = a[read];

	}
	else if (XBlock.level[XBlock.BotLeft[ib]] > XBlock.level[ib])
	{

		if (ix < (XParam.blkwidth / 2))
		{

			jj = ix * 2;
			int bb = XBlock.BotLeft[ib];

			ii = memloc(XParam, jj, (XParam.blkwidth - 3), bb);
			ir = memloc(XParam, jj, (XParam.blkwidth - 4), bb);
			it = memloc(XParam, jj + 1, (XParam.blkwidth - 3), bb);
			itr = memloc(XParam, jj + 1, (XParam.blkwidth - 4), bb);

			abot = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.BotRight[ib] == ib)
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, ix, 0, ib);

				abot = a[read];
			}
		}
		else
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//
				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, jj, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				ir = memloc(XParam, jj, (XParam.blkwidth - 4), XBlock.BotRight[ib]);
				it = memloc(XParam, jj + 1, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				itr = memloc(XParam, jj + 1, (XParam.blkwidth - 4), XBlock.BotRight[ib]);

				abot = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.BotLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.TopLeft[XBlock.BotLeft[ib]] == ib ? ftoi(ceil(ix * (T)0.5)) : ftoi(ceil(ix * (T)0.5) + XParam.blkwidth / 2);
		T jr = ceil(ix * (T)0.5) * 2 > ix ? T(0.25) : T(0.75);

		ii = memloc(XParam, jj, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
		ir = memloc(XParam, jj, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);
		it = memloc(XParam, jj - 1, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
		itr = memloc(XParam, jj - 1, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);

		abot = BilinearInterpolation(a[itr], a[it], a[ir], a[ii], T(0.0), T(1.0), T(0.0), T(1.0), jr, T(0.75));
	}





	//dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}

template <class T> void gradientHaloTop(Param XParam, BlockP<T>XBlock, int ib, int ix, T* a, T* dadx, T* dady)
{
	int i, iy, jj, ii, ir, it, itr;
	int yminus, read;

	T delta, atop, abot;

	iy = XParam.blkwidth;

	i = memloc(XParam, ix, iy, ib);
	yminus = memloc(XParam, ix, XParam.blkwidth-1, ib);




	abot = a[yminus];



	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.TopLeft[ib] == ib)//The lower half is a boundary 
	{
		if (ix < (XParam.blkwidth / 2))
		{

			read = memloc(XParam, ix, XParam.blkwidth - 1, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			atop = a[read];

		}

		if (XBlock.TopRight[ib] == ib) // boundary on the top half too
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, ix, XParam.blkwidth - 1, ib);

				atop = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (ix >= (XParam.blkwidth / 2))
			{

				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, jj, 3, XBlock.TopRight[ib]);
				ir = memloc(XParam, jj, 2, XBlock.TopRight[ib]);
				it = memloc(XParam, jj + 1, 3, XBlock.TopRight[ib]);
				itr = memloc(XParam, jj + 1, 2, XBlock.TopRight[ib]);

				atop = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.TopLeft[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam, ix, 1, XBlock.TopLeft[ib]);
		atop = a[read];

	}
	else if (XBlock.level[XBlock.TopLeft[ib]] > XBlock.level[ib])
	{

		if (ix < (XParam.blkwidth / 2))
		{

			jj = ix * 2;
			int bb = XBlock.TopLeft[ib];

			ii = memloc(XParam, jj, 3, bb);
			ir = memloc(XParam, jj, 2, bb);
			it = memloc(XParam, jj + 1, 3, bb);
			itr = memloc(XParam, jj + 1, 2, bb);

			atop = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.TopRight[ib] == ib)
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam, ix, XParam.blkwidth - 1, ib);

				atop = a[read];
			}
		}
		else
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//
				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam, jj, 3, XBlock.TopRight[ib]);
				ir = memloc(XParam, jj, 2, XBlock.TopRight[ib]);
				it = memloc(XParam, jj + 1, 3, XBlock.TopRight[ib]);
				itr = memloc(XParam, jj + 1, 2, XBlock.TopRight[ib]);

				atop = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.TopLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.BotLeft[XBlock.TopLeft[ib]] == ib ? ftoi(ceil(ix * (T)0.5)) : ftoi(ceil(ix * (T)0.5) + XParam.blkwidth / 2);
		T jr = ceil(ix * (T)0.5) * 2 > ix ? T(0.25) : T(0.75);

		ii = memloc(XParam, jj, 0, XBlock.TopLeft[ib]);
		ir = memloc(XParam, jj, 1, XBlock.TopLeft[ib]);
		it = memloc(XParam, jj - 1, 0, XBlock.TopLeft[ib]);
		itr = memloc(XParam, jj - 1, 1, XBlock.TopLeft[ib]);

		atop = BilinearInterpolation(a[it], a[itr], a[ii], a[ir], T(0.0), T(1.0), T(0.0), T(1.0), jr, T(0.25));
	}





	//dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}



template <class T> __global__ void gradientHaloLeftGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	//unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int ix = -1;
	int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];
	int i, jj, ii, ir, it, itr;
	int xplus, read;

	T delta, aright, aleft;

	

	i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);
	xplus = memloc(XParam.halowidth, blkmemwidth, ix + 1, iy, ib);


	aright = a[xplus];



	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.LeftBot[ib] == ib)//The lower half is a boundary 
	{
		if (iy < (XParam.blkwidth / 2))
		{

			read = memloc(XParam.halowidth, blkmemwidth, 0, iy, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			aleft = a[read];
		}

		if (XBlock.LeftTop[ib] == ib) // boundary on the top half too
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, 0, iy, ib);

				aleft = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (iy >= (XParam.blkwidth / 2))
			{

				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj, XBlock.LeftTop[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj + 1, XBlock.LeftTop[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj + 1, XBlock.LeftTop[ib]);

				aleft = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.LeftBot[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 2), iy, XBlock.LeftBot[ib]);
		aleft = a[read];

	}
	else if (XBlock.level[XBlock.LeftBot[ib]] > XBlock.level[ib])
	{

		if (iy < (XParam.blkwidth / 2))
		{

			jj = iy * 2;
			int bb = XBlock.LeftBot[ib];

			ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj, bb);
			ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj, bb);
			it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj + 1, bb);
			itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj + 1, bb);

			aleft = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.LeftTop[ib] == ib)
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, 0, iy, ib);

				aleft = a[read];
			}
		}
		else
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//
				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj, XBlock.LeftTop[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 3), jj + 1, XBlock.LeftTop[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 4), jj + 1, XBlock.LeftTop[ib]);

				aleft = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.LeftBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.RightBot[XBlock.LeftBot[ib]] == ib ? ceil(iy * (T)0.5) : ceil(iy * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(iy * (T)0.5) * 2 > iy ? T(0.25) : T(0.75);

		ii = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 1), jj, XBlock.LeftBot[ib]);
		ir = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 2), jj, XBlock.LeftBot[ib]);
		it = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 1), jj - 1, XBlock.LeftBot[ib]);
		itr = memloc(XParam.halowidth, blkmemwidth, (XParam.blkwidth - 2), jj - 1, XBlock.LeftBot[ib]);

		aleft = BilinearInterpolation(a[itr], a[ir], a[it], a[ii], T(0.0), T(1.0), T(0.0), T(1.0), T(0.75), jr);
	}





	dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	//dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}

template <class T> __global__ void gradientHaloRightGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	//unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int ix = XParam.blkwidth;
	int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];
	int i, jj, ii, ir, it, itr;
	int xminus, read;

	T delta, aright, aleft;



	i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);
	xminus = memloc(XParam.halowidth, blkmemwidth, ix - 1, iy, ib);


	aleft = a[xminus];



	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.RightBot[ib] == ib)//The lower half is a boundary 
	{
		if (iy < (XParam.blkwidth / 2))
		{

			read = memloc(XParam.halowidth, blkmemwidth, XParam.blkwidth - 1, iy, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			aright = a[read];;
		}

		if (XBlock.RightTop[ib] == ib) // boundary on the top half too
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, XParam.blkwidth - 1, iy, ib);

				aright = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (iy >= (XParam.blkwidth / 2))
			{

				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, 3, jj, XBlock.RightTop[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, 2, jj, XBlock.RightTop[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, 3, jj + 1, XBlock.RightTop[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, 2, jj + 1, XBlock.RightTop[ib]);

				aright = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.RightBot[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam.halowidth, blkmemwidth, 1, iy, XBlock.RightBot[ib]);
		aright = a[read];

	}
	else if (XBlock.level[XBlock.RightBot[ib]] > XBlock.level[ib])
	{

		if (iy < (XParam.blkwidth / 2))
		{

			jj = iy * 2;
			int bb = XBlock.RightBot[ib];

			ii = memloc(XParam.halowidth, blkmemwidth, 3, jj, bb);
			ir = memloc(XParam.halowidth, blkmemwidth, 2, jj, bb);
			it = memloc(XParam.halowidth, blkmemwidth, 3, jj + 1, bb);
			itr = memloc(XParam.halowidth, blkmemwidth, 2, jj + 1, bb);

			aright = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.RightTop[ib] == ib)
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, XParam.blkwidth - 1, iy, ib);

				aright = a[read];
			}
		}
		else
		{
			if (iy >= (XParam.blkwidth / 2))
			{
				//
				jj = (iy - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, 3, jj, XBlock.RightTop[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, 2, jj, XBlock.RightTop[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, 3, jj + 1, XBlock.RightTop[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, 2, jj + 1, XBlock.RightTop[ib]);

				aright = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.RightBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.LeftBot[XBlock.RightBot[ib]] == ib ? ceil(iy * (T)0.5) : ceil(iy * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(iy * (T)0.5) * 2 > iy ? T(0.25) : T(0.75);

		ii = memloc(XParam.halowidth, blkmemwidth, 0, jj, XBlock.RightBot[ib]);
		ir = memloc(XParam.halowidth, blkmemwidth, 1, jj, XBlock.RightBot[ib]);
		it = memloc(XParam.halowidth, blkmemwidth, 0, jj - 1, XBlock.RightBot[ib]);
		itr = memloc(XParam.halowidth, blkmemwidth, 1, jj - 1, XBlock.RightBot[ib]);

		aright = BilinearInterpolation(a[it], a[ii], a[itr], a[ir], T(0.0), T(1.0), T(0.0), T(1.0), T(0.25), jr);
	}





	dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	//dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}


template <class T> __global__ void gradientHaloBotGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	//unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int iy = -1;
	int ix = threadIdx.x;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];


	int i, jj, ii, ir, it, itr;
	int yplus, read;

	T delta, atop, abot;

	
	i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);
	yplus = memloc(XParam.halowidth, blkmemwidth, ix, iy + 1, ib);




	atop = a[yplus];



	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.BotLeft[ib] == ib)//The lower half is a boundary 
	{
		if (ix < (XParam.blkwidth / 2))
		{

			read = memloc(XParam.halowidth, blkmemwidth, ix, 0, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			abot = a[read];

		}

		if (XBlock.BotRight[ib] == ib) // boundary on the top half too
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, ix, 0, ib);

				abot = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (ix >= (XParam.blkwidth / 2))
			{

				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 4), XBlock.BotRight[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 4), XBlock.BotRight[ib]);

				abot = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.BotLeft[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam.halowidth, blkmemwidth, ix, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);
		abot = a[read];

		

	}
	else if (XBlock.level[XBlock.BotLeft[ib]] > XBlock.level[ib])
	{

		if (ix < (XParam.blkwidth / 2))
		{

			jj = ix * 2;
			int bb = XBlock.BotLeft[ib];

			ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 3), bb);
			ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 4), bb);
			it = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 3), bb);
			itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 4), bb);

			abot = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.BotRight[ib] == ib)
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, ix, 0, ib);

				abot = a[read];
			}
		}
		else
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//
				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 4), XBlock.BotRight[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 3), XBlock.BotRight[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, (XParam.blkwidth - 4), XBlock.BotRight[ib]);

				abot = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.BotLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.TopLeft[XBlock.BotLeft[ib]] == ib ? ceil(ix * (T)0.5) : ceil(ix * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(ix * (T)0.5) * 2 > ix ? T(0.25) : T(0.75);

		ii = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
		ir = memloc(XParam.halowidth, blkmemwidth, jj, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);
		it = memloc(XParam.halowidth, blkmemwidth, jj - 1, (XParam.blkwidth - 1), XBlock.BotLeft[ib]);
		itr = memloc(XParam.halowidth, blkmemwidth, jj - 1, (XParam.blkwidth - 2), XBlock.BotLeft[ib]);

		abot = BilinearInterpolation(a[itr], a[it], a[ir], a[ii], T(0.0), T(1.0), T(0.0), T(1.0), jr, T(0.75));
	}





	//dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}


template <class T> __global__ void gradientHaloTopGPU(Param XParam, BlockP<T>XBlock, T* a, T* dadx, T* dady)
{
	unsigned int blkmemwidth = XParam.blkwidth + XParam.halowidth * 2;
	//unsigned int blksize = XParam.blkmemwidth * XParam.blkmemwidth;
	int iy = XParam.blkwidth;
	int ix = threadIdx.x;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];


	int i, jj, ii, ir, it, itr;
	int yminus, read;

	T delta, atop, abot;


	i = memloc(XParam.halowidth, blkmemwidth, ix, iy, ib);
	yminus = memloc(XParam.halowidth, blkmemwidth, ix, XParam.blkwidth - 1, ib);




	abot = a[yminus];



	delta = calcres(T(XParam.dx), XBlock.level[ib]);


	if (XBlock.TopLeft[ib] == ib)//The lower half is a boundary 
	{
		if (ix < (XParam.blkwidth / 2))
		{

			read = memloc(XParam.halowidth, blkmemwidth, ix, XParam.blkwidth - 1, ib);// or memloc(XParam, -1, j, ib) but they should be the same

			atop = a[read];

		}

		if (XBlock.TopRight[ib] == ib) // boundary on the top half too
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, ix, XParam.blkwidth - 1, ib);

				atop = a[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			if (ix >= (XParam.blkwidth / 2))
			{

				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, jj, 3, XBlock.TopRight[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, jj, 2, XBlock.TopRight[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, jj + 1, 3, XBlock.TopRight[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, 2, XBlock.TopRight[ib]);

				atop = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[XBlock.TopLeft[ib]]) // LeftTop block does not exist
	{

		read = memloc(XParam.halowidth, blkmemwidth, ix, 1, XBlock.TopLeft[ib]);
		atop = a[read];



	}
	else if (XBlock.level[XBlock.TopLeft[ib]] > XBlock.level[ib])
	{

		if (ix < (XParam.blkwidth / 2))
		{

			jj = ix * 2;
			int bb = XBlock.TopLeft[ib];;

			ii = memloc(XParam.halowidth, blkmemwidth, jj, 3, bb);
			ir = memloc(XParam.halowidth, blkmemwidth, jj, 2, bb);
			it = memloc(XParam.halowidth, blkmemwidth, jj + 1, 3, bb);
			itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, 2, bb);

			atop = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.TopRight[ib] == ib)
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//

				read = memloc(XParam.halowidth, blkmemwidth, ix, XParam.blkwidth - 1, ib);

				atop = a[read];
			}
		}
		else
		{
			if (ix >= (XParam.blkwidth / 2))
			{
				//
				jj = (ix - XParam.blkwidth / 2) * 2;
				ii = memloc(XParam.halowidth, blkmemwidth, jj, 3, XBlock.TopRight[ib]);
				ir = memloc(XParam.halowidth, blkmemwidth, jj, 2, XBlock.TopRight[ib]);
				it = memloc(XParam.halowidth, blkmemwidth, jj + 1, 3, XBlock.TopRight[ib]);
				itr = memloc(XParam.halowidth, blkmemwidth, jj + 1, 2, XBlock.TopRight[ib]);

				atop = T(0.25) * (a[ii] + a[ir] + a[it] + a[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.TopLeft[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		jj = XBlock.BotLeft[XBlock.TopLeft[ib]] == ib ? ceil(ix * (T)0.5) : ceil(ix * (T)0.5) + XParam.blkwidth / 2;
		T jr = ceil(ix * (T)0.5) * 2 > ix ? T(0.25) : T(0.75);

		ii = memloc(XParam.halowidth, blkmemwidth, jj, 0, XBlock.TopLeft[ib]);
		ir = memloc(XParam.halowidth, blkmemwidth, jj, 1, XBlock.TopLeft[ib]);
		it = memloc(XParam.halowidth, blkmemwidth, jj - 1, 0, XBlock.TopLeft[ib]);
		itr = memloc(XParam.halowidth, blkmemwidth, jj - 1, 1, XBlock.TopLeft[ib]);

		atop = BilinearInterpolation(a[it], a[itr], a[ii], a[ir], T(0.0), T(1.0), T(0.0), T(1.0), jr, T(0.25));
	}





	//dadx[i] = minmod2(T(XParam.theta), aleft, a[i], aright) / delta;
	dady[i] = minmod2(T(XParam.theta), abot, a[i], atop) / delta;

}

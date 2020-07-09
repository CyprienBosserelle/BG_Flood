void mainloopGPUADA(Param XParam) // float, metric coordinate
{
	

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
							 //dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);


	//update step 1



	gradientGPUXYBUQADASM << <gridDim, blockDim, 0 >> >((float)XParam.theta, (float)XParam.delta,activeblk_g,level_g, leftblk_g, rightblk_g, topblk_g, botblk_g, zb_g, dzsdx_g, dzsdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

}

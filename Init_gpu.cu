
void CUDA_CHECK(cudaError CUDerr)
{


	if (cudaSuccess != CUDerr) {

		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \

			__FILE__, __LINE__, cudaGetErrorString(CUDerr));

		exit(EXIT_FAILURE);

	}
}



template <class T> void Allocate1GPU(int nx, int ny, T *&zb_g)
{
	CUDA_CHECK(cudaMalloc((void **)&zb_g, nx*ny * sizeof(T)));
}
template <class T> void Allocate4GPU(int nx, int ny, T *&zs_g, T *&hh_g, T *&uu_g, T *&vv_g)
{
	CUDA_CHECK(cudaMalloc((void **)&zs_g, nx*ny * sizeof(T)));
	CUDA_CHECK(cudaMalloc((void **)&hh_g, nx*ny * sizeof(T)));
	CUDA_CHECK(cudaMalloc((void **)&uu_g, nx*ny * sizeof(T)));
	CUDA_CHECK(cudaMalloc((void **)&vv_g, nx*ny * sizeof(T)));
}


void checkGradGPU(Param XParam)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	gradientGPUXYBUQ << <gridDim, blockDim, 0 >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hh_g, dhdx_g, dhdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradient(XParam.nblk, XParam.blksize, (float)XParam.theta, (float)XParam.delta, leftblk, rightblk, topblk, botblk, hh, dhdx, dhdy);

	CUDA_CHECK(cudaMemcpy(dummy, dhdx_g, XParam.nblk*XParam.blksize * sizeof(float), cudaMemcpyDeviceToHost));

	float mdiff = maxdiff(XParam.nblk*XParam.blksize, dhdx, dummy);
	float maxerr = 1e-11f;//1e-7f
	if (mdiff > maxerr)
	{
		printf("High error in dhdx: %f\n", mdiff);
	}
}


int AllocMemGPU(Param XParam)
{
	//function to allocate the memory on the GPU
	// Also prepare textures
	// Pointers are Global !
	//Need to add a sucess check for each call to malloc

	int nblk = XParam.nblk;
	int blksize = XParam.blksize;
	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{
		Allocate1GPU(nblk, blksize, zb_gd);
		Allocate4GPU(nblk, blksize, zs_gd, hh_gd, uu_gd, vv_gd);
		Allocate4GPU(nblk, blksize, zso_gd, hho_gd, uuo_gd, vvo_gd);
		Allocate4GPU(nblk, blksize, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd);
		Allocate4GPU(nblk, blksize, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd);

		Allocate4GPU(nblk, blksize, Su_gd, Sv_gd, Fhu_gd, Fhv_gd);
		Allocate4GPU(nblk, blksize, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd);

		Allocate4GPU(nblk, blksize, dh_gd, dhu_gd, dhv_gd, dtmax_gd);
		Allocate1GPU(nblk, blksize, cf_gd);
		Allocate1GPU(nblk, 1, blockxo_gd);
		Allocate1GPU(nblk, 1, blockyo_gd);



		arrmin_d = (double *)malloc(nblk* blksize * sizeof(double));
		CUDA_CHECK(cudaMalloc((void **)&arrmin_gd, nblk* blksize * sizeof(double)));
		CUDA_CHECK(cudaMalloc((void **)&arrmax_gd, nblk* blksize * sizeof(double)));

		if (XParam.outhhmax == 1)
		{
			Allocate1GPU(nblk, blksize, hhmax_gd);
		}
		if (XParam.outzsmax == 1)
		{
			Allocate1GPU(nblk, blksize, zsmax_gd);
		}
		if (XParam.outuumax == 1)
		{
			Allocate1GPU(nblk, blksize, uumax_gd);
		}
		if (XParam.outvvmax == 1)
		{
			Allocate1GPU(nblk, blksize, vvmax_gd);
		}
		if (XParam.outhhmean == 1)
		{
			Allocate1GPU(nblk, blksize, hhmean_gd);
		}
		if (XParam.outzsmean == 1)
		{
			Allocate1GPU(nblk, blksize, zsmean_gd);
		}
		if (XParam.outuumean == 1)
		{
			Allocate1GPU(nblk, blksize, uumean_gd);
		}
		if (XParam.outvvmean == 1)
		{
			Allocate1GPU(nblk, blksize, vvmean_gd);
		}

		if (XParam.outvort == 1)
		{
			Allocate1GPU(nblk, blksize, vort_gd);
		}

		if (XParam.TSnodesout.size() > 0)
		{
			// Allocate mmemory to store TSoutput in between writing to disk
			int nTS = 1; // Nb of points
			int nvts = 1; // NB of variables hh, zs, uu, vv
			int nstore = 2048; //store up to 2048 pts
			TSstore_d = (double *)malloc(nTS*nvts*nstore * sizeof(double));
			CUDA_CHECK(cudaMalloc((void **)&TSstore_gd, nTS*nvts*nstore * sizeof(double)));
			//Cpu part done differently because there are no latency issue (i.e. none that I care about)

		}
	}
	else
	{
		Allocate1GPU(nblk, blksize, zb_g);
		Allocate4GPU(nblk, blksize, zs_g, hh_g, uu_g, vv_g);
		Allocate4GPU(nblk, blksize, zso_g, hho_g, uuo_g, vvo_g);
		Allocate4GPU(nblk, blksize, dzsdx_g, dhdx_g, dudx_g, dvdx_g);
		Allocate4GPU(nblk, blksize, dzsdy_g, dhdy_g, dudy_g, dvdy_g);

		Allocate4GPU(nblk, blksize, Su_g, Sv_g, Fhu_g, Fhv_g);
		Allocate4GPU(nblk, blksize, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g);

		Allocate4GPU(nblk, blksize, dh_g, dhu_g, dhv_g, dtmax_g);
		Allocate1GPU(nblk, blksize, cf_g);

		Allocate1GPU(nblk, 1, blockxo_g);
		Allocate1GPU(nblk, 1, blockyo_g);

		arrmin = (float *)malloc(nblk*blksize * sizeof(float));
		CUDA_CHECK(cudaMalloc((void **)&arrmin_g, nblk*blksize * sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&arrmax_g, nblk*blksize * sizeof(float)));

		if (XParam.outhhmax == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&hhmax_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outzsmax == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&zsmax_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outuumax == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&uumax_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outvvmax == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&vvmax_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outhhmean == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&hhmean_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outzsmean == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&zsmean_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outuumean == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&uumean_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outvvmean == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&vvmean_g, nblk*blksize * sizeof(float)));
		}

		if (XParam.outvort == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&vort_g, nblk*blksize * sizeof(float)));
		}


		if (XParam.TSnodesout.size() > 0)
		{
			// Allocate mmemory to store TSoutput in between writing to disk
			int nTS = 1; // Nb of points
			int nvts = 1; // NB of variables hh, zs, uu, vv
			int nstore = 2048; //store up to 2048 pts
			TSstore = (float *)malloc(nTS*nvts*nstore * sizeof(float));
			CUDA_CHECK(cudaMalloc((void **)&TSstore_g, nTS*nvts*nstore * sizeof(float)));
			//Cpu part done differently because there are no latency issue (i.e. none that I care about)

		}
	}


	Allocate4GPU(nblk, 1, leftblk_g, rightblk_g, topblk_g, botblk_g);

	return 1;
}


int allocTexMem(bndparam bnd, cudaArray * &WLS, cudaArray * &Uvel, cudaArray * &Vvel, cudaChannelFormatDesc &CFDbndzs, cudaChannelFormatDesc &CFDbnduu, cudaChannelFormatDesc &CFDbndvv, texture<float, 2, cudaReadModeElementType> &TexZs, texture<float, 2, cudaReadModeElementType> &TexU, texture<float, 2, cudaReadModeElementType> &TexV)
{
	int nbndtimes = (int)bnd.data.size();
	int nbndvec = (int)bnd.data[0].wlevs.size();
	CUDA_CHECK(cudaMallocArray(&WLS, &CFDbndzs, nbndtimes, nbndvec));

	float * lWLS;
	lWLS = (float *)malloc(nbndtimes * nbndvec * sizeof(float));

	for (int ibndv = 0; ibndv < nbndvec; ibndv++)
	{
		for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
		{
			//
			lWLS[ibndt + ibndv*nbndtimes] = bnd.data[ibndt].wlevs[ibndv];
		}
	}

	CUDA_CHECK(cudaMemcpyToArray(WLS, 0, 0, lWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

	TexZs.addressMode[0] = cudaAddressModeClamp;
	TexZs.addressMode[1] = cudaAddressModeClamp;
	TexZs.filterMode = cudaFilterModeLinear;
	TexZs.normalized = false;


	CUDA_CHECK(cudaBindTextureToArray(TexZs, WLS, CFDbndzs));


	// In case of Nesting U and V are also prescribed

		// If uu information is available in the boundary we can assume it is a nesting type of bnd
	int nbndvecuu = (int)bnd.data[0].uuvel.size();

	if (nbndvecuu == nbndvec)
	{
		CUDA_CHECK(cudaMallocArray(&Uvel, &CFDbnduu, nbndtimes, nbndvec));

		for (int ibndv = 0; ibndv < nbndvec; ibndv++)
		{
			for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
			{
				//
				lWLS[ibndt + ibndv * nbndtimes] =bnd.data[ibndt].uuvel[ibndv];
			}
		}
		CUDA_CHECK(cudaMemcpyToArray(Uvel, 0, 0, lWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

		TexU.addressMode[0] = cudaAddressModeClamp;
		TexU.addressMode[1] = cudaAddressModeClamp;
		TexU.filterMode = cudaFilterModeLinear;
		TexU.normalized = false;

		CUDA_CHECK(cudaBindTextureToArray(TexU, Uvel, CFDbnduu));

	}


	//V velocity side
	int nbndvecvv = (int)bnd.data[0].vvvel.size();

	if (nbndvecvv == nbndvec)
	{

		CUDA_CHECK(cudaMallocArray(&Vvel, &CFDbndvv, nbndtimes, nbndvec));

		for (int ibndv = 0; ibndv < nbndvec; ibndv++)
		{
			for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
			{
				//
				lWLS[ibndt + ibndv * nbndtimes] = bnd.data[ibndt].vvvel[ibndv];
			}
		}
		CUDA_CHECK(cudaMemcpyToArray(Vvel, 0, 0, lWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

		TexV.addressMode[0] = cudaAddressModeClamp;
		TexV.addressMode[1] = cudaAddressModeClamp;
		TexV.filterMode = cudaFilterModeLinear;
		TexV.normalized = false;

		CUDA_CHECK(cudaBindTextureToArray(TexV, Vvel, CFDbndvv));

	}

	///BEWARE
	/// The cases above is not dealing with weird situation where nbndvecvv != nbndvec != nbndvecuu 

	free(lWLS);

	return 1;

}

int AllocMemGPUBND(Param XParam)
{
	// Allocate textures and bind arrays for boundary interpolation


	Allocate1GPU(XParam.leftbnd.nblk, 1, bndleftblk_g);
	Allocate1GPU(XParam.rightbnd.nblk, 1, bndrightblk_g);
	Allocate1GPU(XParam.topbnd.nblk, 1, bndtopblk_g);
	Allocate1GPU(XParam.botbnd.nblk, 1, bndbotblk_g);


	if (XParam.leftbnd.on)
	{
		allocTexMem(XParam.leftbnd, leftWLS_gp, leftUvel_gp, leftVvel_gp, channelDescleftbndzs, channelDescleftbnduu, channelDescleftbndvv, texLZsBND, texLUBND, texLVBND);

		/*
		//leftWLbnd = readWLfile(XParam.leftbndfile);
		//Flatten bnd to copy to cuda array
		int nbndtimes = (int)XParam.leftbnd.data.size();
		int nbndvec = (int)XParam.leftbnd.data[0].wlevs.size();
		CUDA_CHECK(cudaMallocArray(&leftWLS_gp, &channelDescleftbndzs, nbndtimes, nbndvec));
		// This below was float by default and probably should remain float as long as fetched floats are readily converted to double as needed
		float * leftWLS;
		leftWLS = (float *)malloc(nbndtimes * nbndvec * sizeof(float));

		for (int ibndv = 0; ibndv < nbndvec; ibndv++)
		{
			for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
			{
				//
				leftWLS[ibndt + ibndv*nbndtimes] = XParam.leftbnd.data[ibndt].wlevs[ibndv];
			}
		}
		CUDA_CHECK(cudaMemcpyToArray(leftWLS_gp, 0, 0, leftWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

		texLZsBND.addressMode[0] = cudaAddressModeClamp;
		texLZsBND.addressMode[1] = cudaAddressModeClamp;
		texLZsBND.filterMode = cudaFilterModeLinear;
		texLZsBND.normalized = false;


		CUDA_CHECK(cudaBindTextureToArray(texLZsBND, leftWLS_gp, channelDescleftbndzs));
		

		// In case of Nesting U and V are also prescribed

		// If uu information is available in the boundary we can assume it is a nesting type of bnd
		int nbndvecuu = (int)XParam.leftbnd.data[0].uuvel.size();

		if (nbndvecuu == nbndvec)
		{
			CUDA_CHECK(cudaMallocArray(&leftUvel_gp, &channelDescleftbnduu, nbndtimes, nbndvec));

			for (int ibndv = 0; ibndv < nbndvec; ibndv++)
			{
				for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
				{
					//
					leftWLS[ibndt + ibndv*nbndtimes] = XParam.leftbnd.data[ibndt].uuvel[ibndv];
				}
			}
			CUDA_CHECK(cudaMemcpyToArray(leftUvel_gp, 0, 0, leftWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

			texLUBND.addressMode[0] = cudaAddressModeClamp;
			texLUBND.addressMode[1] = cudaAddressModeClamp;
			texLUBND.filterMode = cudaFilterModeLinear;
			texLUBND.normalized = false;

			CUDA_CHECK(cudaBindTextureToArray(texLUBND, leftUvel_gp, channelDescleftbnduu));

		}


		//V velocity side
		int nbndvecvv = (int)XParam.leftbnd.data[0].vvvel.size();

		if (nbndvecvv == nbndvec )
		{

			CUDA_CHECK(cudaMallocArray(&leftVvel_gp, &channelDescleftbndvv, nbndtimes, nbndvec));

			for (int ibndv = 0; ibndv < nbndvec; ibndv++)
			{
				for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
				{
					//
					leftWLS[ibndt + ibndv*nbndtimes] = XParam.leftbnd.data[ibndt].vvvel[ibndv];
				}
			}
			CUDA_CHECK(cudaMemcpyToArray(leftVvel_gp, 0, 0, leftWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

			texLVBND.addressMode[0] = cudaAddressModeClamp;
			texLVBND.addressMode[1] = cudaAddressModeClamp;
			texLVBND.filterMode = cudaFilterModeLinear;
			texLVBND.normalized = false;

			CUDA_CHECK(cudaBindTextureToArray(texLVBND, leftVvel_gp, channelDescleftbndvv));

		}
		
		///BEWARE
		/// The cases above is not dealing with weird situation where nbndvecvv != nbndvec != nbndvecuu 


		free(leftWLS);
		*/

	}
	if (XParam.rightbnd.on)
	{
		allocTexMem(XParam.rightbnd, rightWLS_gp, rightUvel_gp, rightVvel_gp, channelDescrightbndzs, channelDescrightbnduu, channelDescrightbndvv, texRZsBND, texRUBND, texRVBND);

	}
	if (XParam.topbnd.on)
	{
		allocTexMem(XParam.topbnd, topWLS_gp, topUvel_gp, topVvel_gp, channelDesctopbndzs, channelDesctopbnduu, channelDesctopbndvv, texTZsBND, texTUBND, texTVBND);

	}
	if (XParam.botbnd.on)
	{
		allocTexMem(XParam.botbnd, botWLS_gp, botUvel_gp, botVvel_gp, channelDescbotbndzs, channelDescbotbnduu, channelDescbotbndvv, texBZsBND, texBUBND, texBVBND);


	}
	return 1;
}



void LeftFlowBnd(Param XParam)
{
	//
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	dim3 gridDimLBND(XParam.leftbnd.nblk, 1, 1);
	if (XParam.leftbnd.on)
	{
		int SLstepinbnd = 1;



		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = XParam.leftbnd.data[SLstepinbnd].time - XParam.totaltime;

		while (difft < 0.0)
		{
			SLstepinbnd++;
			difft = XParam.leftbnd.data[SLstepinbnd].time - XParam.totaltime;
		}




		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			double itime = SLstepinbnd - 1.0 + (XParam.totaltime - XParam.leftbnd.data[SLstepinbnd - 1].time) / (XParam.leftbnd.data[SLstepinbnd].time - XParam.leftbnd.data[SLstepinbnd - 1].time);
			if (XParam.leftbnd.type == 2 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				dirichlet << <gridDimLBND, blockDim, 0 >> > (-1, 0, (int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.xmax, XParam.yo, XParam.ymax, itime, bndleftblk_g, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.leftbnd.type == 2)
			{
				dirichlet << <gridDimLBND, blockDim, 0 >> > (-1, 0, (int)XParam.leftbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.xmax, (float)XParam.yo, (float)XParam.ymax, (float)itime, bndleftblk_g, rightblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);

				//leftdirichlet << <gridDimLBND, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.ymax, (float)itime, rightblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}

			if (XParam.leftbnd.type == 3 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1D << <gridDimLBND, blockDim, 0 >> > (-1, 0, (int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, bndleftblk_g,rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.leftbnd.type == 3)
			{
				ABS1D << <gridDimLBND, blockDim, 0 >> > (-1, 0, (int)XParam.leftbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, bndleftblk_g, rightblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}

			if (XParam.leftbnd.type == 4 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1DNEST << <gridDimLBND, blockDim, 0 >> > (-1, 0, (int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, bndleftblk_g, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.leftbnd.type == 4)
			{
				ABS1DNEST << <gridDimLBND, blockDim, 0 >> > (-1, 0, (int)XParam.leftbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, bndleftblk_g, rightblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}





			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			std::vector<double> zsbndleft;
			for (int n = 0; n < XParam.leftbnd.data[SLstepinbnd].wlevs.size(); n++)
			{
				zsbndleft.push_back(interptime(XParam.leftbnd.data[SLstepinbnd].wlevs[n], XParam.leftbnd.data[SLstepinbnd - 1].wlevs[n], XParam.leftbnd.data[SLstepinbnd].time - XParam.leftbnd.data[SLstepinbnd - 1].time, XParam.totaltime - XParam.leftbnd.data[SLstepinbnd - 1].time));

			}
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{

				leftdirichletCPUD(XParam.nblk, XParam.blksize, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndleft, blockxo_d, blockyo_d, zs_d, zb_d, hh_d, uu_d, vv_d);
			}
			else
			{
				//void leftdirichletCPU(int nblk, int blksize, float xo,float yo, float g, float dx, std::vector<double> zsbndvec, float * blockxo,float * blockyo, float *zs, float *zb, float *hh, float *uu, float *vv)
				//leftdirichletCPU(nx, ny, (float)XParam.g, zsbndleft, zs, zb, hh, uu, vv);
				leftdirichletCPU(XParam.nblk, XParam.blksize, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndleft, blockxo, blockyo, zs, zb, hh, uu, vv);
			}

		}
	}
	if (XParam.leftbnd.type == 0)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			//
			//dim3 blockDim(16, 16, 1);
			//dim3 gridDim(XParam.nblk, 1, 1);
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				noslipbnd << <gridDimLBND, blockDim, 0 >> >(-1, 0, bndleftblk_g, rightblk_g, zs_gd, hh_gd, uu_gd);
				//noslipbndLeft << <gridDim, blockDim, 0 >> > (XParam.xo, XParam.eps, rightblk_g, blockxo_gd, zb_gd, zs_gd, hh_gd, uu_gd, vv_gd);
			}
			else
			{
				noslipbnd << <gridDimLBND, blockDim, 0 >> >(-1, 0, bndleftblk_g, rightblk_g, zs_g, hh_g, uu_g);
				//noslipbndLeft << <gridDim, blockDim, 0 >> > ((float)XParam.xo, (float)XParam.eps, rightblk_g, blockxo_g, zb_g, zs_g, hh_g, uu_g, vv_g);
			}

			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			noslipbndLCPU(XParam);
		}
	}
	//else neumann bnd (is already built in the solver)
}

void RightFlowBnd(Param XParam)
{
	//
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	dim3 gridDimRBND(XParam.rightbnd.nblk, 1, 1);
	if (XParam.rightbnd.on)
	{
		int SLstepinbnd = 1;





		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = XParam.rightbnd.data[SLstepinbnd].time - XParam.totaltime;

		while (difft < 0.0)
		{
			SLstepinbnd++;
			difft = XParam.rightbnd.data[SLstepinbnd].time - XParam.totaltime;
		}




		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			double itime = SLstepinbnd - 1.0 + (XParam.totaltime - XParam.rightbnd.data[SLstepinbnd - 1].time) / (XParam.rightbnd.data[SLstepinbnd].time - XParam.rightbnd.data[SLstepinbnd - 1].time);
			if (XParam.rightbnd.type == 2 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				dirichlet << <gridDimRBND, blockDim, 0 >> > (1, 0, (int)XParam.rightbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.xmax, XParam.yo, XParam.ymax, itime, bndrightblk_g, leftblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);

				//rightdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.rightbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xmax, XParam.ymax, itime, leftblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.rightbnd.type == 2)
			{
				dirichlet << <gridDimRBND, blockDim, 0 >> > (1, 0, (int)XParam.rightbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.xmax, (float)XParam.yo, (float)XParam.ymax, (float)itime, bndrightblk_g, leftblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);

				//rightdirichlet << <gridDim, blockDim, 0 >> > ((int)XParam.rightbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xmax, (float)XParam.ymax, (float)itime, leftblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}
			else if (XParam.rightbnd.type == 3 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1D << <gridDimRBND, blockDim, 0 >> > (1, 0, (int)XParam.rightbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, bndrightblk_g, leftblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.rightbnd.type == 3)
			{
				ABS1D << <gridDimRBND, blockDim, 0 >> > (1, 0, (int)XParam.rightbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, bndrightblk_g, leftblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}
			if (XParam.rightbnd.type == 4 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1DNEST << <gridDimRBND, blockDim, 0 >> > (1, 0, (int)XParam.rightbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, bndrightblk_g, leftblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.rightbnd.type == 4)
			{
				ABS1DNEST << <gridDimRBND, blockDim, 0 >> > (1, 0, (int)XParam.rightbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, bndrightblk_g, leftblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}

			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			std::vector<double> zsbndright;
			for (int n = 0; n < XParam.rightbnd.data[SLstepinbnd].wlevs.size(); n++)
			{
				zsbndright.push_back(interptime(XParam.rightbnd.data[SLstepinbnd].wlevs[n], XParam.rightbnd.data[SLstepinbnd - 1].wlevs[n], XParam.rightbnd.data[SLstepinbnd].time - XParam.rightbnd.data[SLstepinbnd - 1].time, XParam.totaltime - XParam.rightbnd.data[SLstepinbnd - 1].time));

			}
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				rightdirichletCPUD(XParam.nblk, XParam.blksize, XParam.nx, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndright, blockxo_d, blockyo_d, zs_d, zb_d, hh_d, uu_d, vv_d);
			}
			else
			{
				//rightdirichletCPU(nx, ny, (float)XParam.g, zsbndright, zs, zb, hh, uu, vv);
				rightdirichletCPU(XParam.nblk, XParam.blksize, XParam.nx, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndright, blockxo, blockyo, zs, zb, hh, uu, vv);
			}
		}
	}
	if (XParam.rightbnd.type == 0)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			//
			dim3 blockDim(16, 16, 1);
			dim3 gridDim(XParam.nblk, 1, 1);
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				noslipbnd << <gridDimRBND, blockDim, 0 >> >(1, 0, bndrightblk_g, leftblk_g, zs_gd, hh_gd, uu_gd);
				//noslipbndRight << <gridDim, blockDim, 0 >> > (XParam.dx, XParam.xmax, XParam.eps, leftblk_g, blockxo_gd, zb_gd, zs_gd, hh_gd, uu_gd, vv_gd);
			}
			else
			{
				noslipbnd << <gridDimRBND, blockDim, 0 >> >(1, 0, bndrightblk_g, leftblk_g, zs_g, hh_g, uu_g);
				//noslipbndRight << <gridDim, blockDim, 0 >> > ((float)XParam.dx, (float)XParam.xmax, (float)XParam.eps, leftblk_g, blockxo_g, zb_g, zs_g, hh_g, uu_g, vv_g);
			}

			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			noslipbndRCPU(XParam);
		}
	}
	//else neumann bnd (is already built in the algorithm)
}

void TopFlowBnd(Param XParam)
{
	//

	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	dim3 gridDimTBND(XParam.topbnd.nblk, 1, 1);
	if (XParam.topbnd.on)
	{
		int SLstepinbnd = 1;





		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = XParam.topbnd.data[SLstepinbnd].time - XParam.totaltime;

		while (difft < 0.0)
		{
			SLstepinbnd++;
			difft = XParam.topbnd.data[SLstepinbnd].time - XParam.totaltime;
		}



		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			double itime = SLstepinbnd - 1.0 + (XParam.totaltime - XParam.topbnd.data[SLstepinbnd - 1].time) / (XParam.topbnd.data[SLstepinbnd].time - XParam.topbnd.data[SLstepinbnd - 1].time);
			if (XParam.topbnd.type == 2 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				dirichlet << <gridDimTBND, blockDim, 0 >> > (0, 1, (int)XParam.topbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.xmax, XParam.yo, XParam.ymax, itime, bndtopblk_g, botblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, vv_gd, uu_gd);

				//topdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.topbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xmax, XParam.ymax, itime, botblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.topbnd.type == 2)
			{
				dirichlet << <gridDimTBND, blockDim, 0 >> > (0, 1, (int)XParam.topbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.xmax, (float)XParam.yo, (float)XParam.ymax, (float)itime, bndtopblk_g, botblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, vv_g, uu_g);

				//topdirichlet << <gridDim, blockDim, 0 >> > ((int)XParam.topbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xmax, (float)XParam.ymax, (float)itime, botblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}
			else if (XParam.topbnd.type == 3 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1D << <gridDimTBND, blockDim, 0 >> > (0, 1, (int)XParam.topbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, bndtopblk_g, botblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.topbnd.type == 3)
			{
				ABS1D << <gridDimTBND, blockDim, 0 >> > (0, 1, (int)XParam.topbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, bndtopblk_g, botblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, vv_g, uu_g);
			}
			if (XParam.topbnd.type == 4 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1DNEST << <gridDimTBND, blockDim, 0 >> > (0, 1, (int)XParam.topbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, bndtopblk_g, botblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, vv_gd, uu_gd);
			}
			else if (XParam.topbnd.type == 4)
			{
				ABS1DNEST << <gridDimTBND, blockDim, 0 >> > (0, 1, (int)XParam.topbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, bndtopblk_g, botblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, vv_g, uu_g);
			}

			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			std::vector<double> zsbndtop;
			for (int n = 0; n < XParam.topbnd.data[SLstepinbnd].wlevs.size(); n++)
			{
				zsbndtop.push_back(interptime(XParam.topbnd.data[SLstepinbnd].wlevs[n], XParam.topbnd.data[SLstepinbnd - 1].wlevs[n], XParam.topbnd.data[SLstepinbnd].time - XParam.topbnd.data[SLstepinbnd - 1].time, XParam.totaltime - XParam.topbnd.data[SLstepinbnd - 1].time));

			}
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				topdirichletCPUD(XParam.nblk, XParam.blksize, XParam.ny, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndtop, blockxo_d, blockyo_d, zs_d, zb_d, hh_d, uu_d, vv_d);
			}
			else
			{

				//topdirichletCPU(nx, ny, (float)XParam.g, zsbndtop, zs, zb, hh, uu, vv);
				topdirichletCPU(XParam.nblk, XParam.blksize, XParam.ny, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndtop, blockxo, blockyo, zs, zb, hh, uu, vv);
			}
		}
	}
	if (XParam.topbnd.type == 0)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			//

			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				noslipbnd << <gridDimTBND, blockDim, 0 >> >(0, 1, bndtopblk_g, botblk_g, zs_gd, hh_gd, vv_gd);
				//noslipbndTop << <gridDim, blockDim, 0 >> > (XParam.dx, XParam.ymax, XParam.eps, botblk_g, blockyo_gd, zb_gd, zs_gd, hh_gd, uu_gd, vv_gd);
			}
			else
			{
				noslipbnd << <gridDimTBND, blockDim, 0 >> >(0, 1, bndtopblk_g, botblk_g, zs_g, hh_g, vv_g);
				//noslipbndTop << <gridDim, blockDim, 0 >> > ((float)XParam.dx, (float)XParam.ymax, (float)XParam.eps, botblk_g, blockyo_g, zb_g, zs_g, hh_g, uu_g, vv_g);
			}

			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			noslipbndTCPU(XParam);
		}
	}
	//else neumann bnd (is already built in the algorithm)
}

void BotFlowBnd(Param XParam)
{
	//
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	dim3 gridDimBBND(XParam.botbnd.nblk, 1, 1);
	if (XParam.botbnd.on)
	{
		int SLstepinbnd = 1;





		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = XParam.botbnd.data[SLstepinbnd].time - XParam.totaltime;

		while (difft < 0.0)
		{
			SLstepinbnd++;
			difft = XParam.botbnd.data[SLstepinbnd].time - XParam.totaltime;
		}




		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			double itime = SLstepinbnd - 1.0 + (XParam.totaltime - XParam.botbnd.data[SLstepinbnd - 1].time) / (XParam.botbnd.data[SLstepinbnd].time - XParam.botbnd.data[SLstepinbnd - 1].time);
			if (XParam.botbnd.type == 2 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				dirichlet << <gridDimBBND, blockDim, 0 >> > (0, -1, (int)XParam.botbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.xmax, XParam.yo, XParam.ymax, itime, bndbotblk_g, topblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, vv_gd, uu_gd);

				//botdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.botbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xmax, XParam.yo, itime, topblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.botbnd.type == 2)
			{
				dirichlet << <gridDimBBND, blockDim, 0 >> > (0, -1, (int)XParam.botbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.xmax, (float)XParam.yo, (float)XParam.ymax, (float)itime, bndbotblk_g, topblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, vv_g, uu_g);

				//botdirichlet << <gridDim, blockDim, 0 >> > ((int)XParam.botbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xmax, (float)XParam.yo, (float)itime, topblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}
			else if (XParam.botbnd.type == 3 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1D << <gridDimBBND, blockDim, 0 >> > (0, -1, (int)XParam.botbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, bndbotblk_g, topblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, vv_gd, uu_gd);
			}
			else if (XParam.botbnd.type == 3)
			{
				ABS1D << <gridDimBBND, blockDim, 0 >> > (0, -1, (int)XParam.botbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, bndbotblk_g, topblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, vv_g, uu_g);
			}
			if (XParam.botbnd.type == 4 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1DNEST << <gridDimBBND, blockDim, 0 >> > (0, -1, (int)XParam.botbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, bndbotblk_g, topblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, vv_gd, uu_gd);
			}
			else if (XParam.botbnd.type == 4)
			{
				ABS1DNEST << <gridDimBBND, blockDim, 0 >> > (0, -1, (int)XParam.botbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, bndbotblk_g, topblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, vv_g, uu_g);
			}

			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			std::vector<double> zsbndbot;
			for (int n = 0; n < XParam.botbnd.data[SLstepinbnd].wlevs.size(); n++)
			{
				zsbndbot.push_back(interptime(XParam.botbnd.data[SLstepinbnd].wlevs[n], XParam.botbnd.data[SLstepinbnd - 1].wlevs[n], XParam.botbnd.data[SLstepinbnd].time - XParam.botbnd.data[SLstepinbnd - 1].time, XParam.totaltime - XParam.botbnd.data[SLstepinbnd - 1].time));

			}
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				botdirichletCPUD(XParam.nblk, XParam.blksize, XParam.ny, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndbot, blockxo_d, blockyo_d, zs_d, zb_d, hh_d, uu_d, vv_d);
			}
			else
			{
				//botdirichletCPU(nx, ny, (float)XParam.g, zsbndbot, zs, zb, hh, uu, vv);
				botdirichletCPU(XParam.nblk, XParam.blksize, XParam.ny, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndbot, blockxo, blockyo, zs, zb, hh, uu, vv);
			}
		}
	}
	if (XParam.botbnd.type == 0)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			//

			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				noslipbnd << <gridDimBBND, blockDim, 0 >> >(0, -1, bndbotblk_g, topblk_g, zs_gd, hh_gd, vv_gd);
				//noslipbndBot << <gridDim, blockDim, 0 >> > (XParam.yo, XParam.eps, topblk_g, blockyo_gd, zb_gd, zs_gd, hh_gd, uu_gd, vv_gd);
			}
			else
			{
				noslipbnd << <gridDimBBND, blockDim, 0 >> >(0, -1, bndbotblk_g, topblk_g, zs_g, hh_g, vv_g);
				//noslipbndBot << <gridDim, blockDim, 0 >> > ((float)XParam.yo, (float)XParam.eps, topblk_g, blockyo_g, zb_g, zs_g, hh_g, uu_g, vv_g);
			}

			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			noslipbndBCPU(XParam);
		}
	}
	//else neumann bnd (is already built in the algorithm)
}





double FlowGPU(Param XParam, double nextoutputtime)
{
	int nx = XParam.nx;
	int ny = XParam.ny;

	const int num_streams = 2;

	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}



	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	dtmax = (float)(1.0 / epsilon);
	//float dtmaxtmp = dtmax;

	resetdtmax << <gridDim, blockDim, 0, streams[0] >> > (dtmax_g);
	//CUDA_CHECK(cudaDeviceSynchronize());
	//update step 1



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hh_g, dhdx_g, dhdy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zs_g, dzsdx_g, dzsdy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uu_g, dudx_g, dudy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vv_g, dvdx_g, dvdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//CUDA_CHECK(cudaStreamSynchronize(streams[0]));
	//normal cartesian case
	updateKurgX << <gridDim, blockDim, 0, streams[0] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, leftblk_g, hh_g, zs_g, uu_g, vv_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//CUDA_CHECK(cudaStreamSynchronize(streams[1]));
	updateKurgY << <gridDim, blockDim, 0, streams[1] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, botblk_g, hh_g, zs_g, uu_g, vv_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);

	CUDA_CHECK(cudaDeviceSynchronize());



	//GPU Harris reduction #3. 8.3x reduction #0  Note #7 if a lot faster
	// This was successfully tested with a range of grid size
	//reducemax3 << <gridDimLine, blockDimLine, 64*sizeof(float) >> >(dtmax_g, arrmax_g, nx*ny)
	int s = XParam.nblk*XParam.blksize;
	int maxThreads = 256;
	int threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
	int blocks = (s + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
	dim3 blockDimLine(threads, 1, 1);
	dim3 gridDimLine(blocks, 1, 1);

	float mindtmaxB;

	reducemin3 << <gridDimLine, blockDimLine, smemSize >> > (dtmax_g, arrmax_g, s);
	CUDA_CHECK(cudaDeviceSynchronize());



	s = gridDimLine.x;
	while (s > 1)//cpuFinalThreshold
	{
		threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);

		smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

		dim3 blockDimLineS(threads, 1, 1);
		dim3 gridDimLineS(blocks, 1, 1);

		CUDA_CHECK(cudaMemcpy(dtmax_g, arrmax_g, s * sizeof(float), cudaMemcpyDeviceToDevice));

		reducemin3 << <gridDimLineS, blockDimLineS, smemSize >> > (dtmax_g, arrmax_g, s);
		CUDA_CHECK(cudaDeviceSynchronize());

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}


	CUDA_CHECK(cudaMemcpy(dummy, arrmax_g, 32 * sizeof(float), cudaMemcpyDeviceToHost));
	mindtmaxB = dummy[0];

	//32 seem safe here bu I wonder why it is not 1 for the largers arrays...
	/*
	for (int i = 0; i < 32; i++)
	{
	mindtmaxB = min(dummy[i], mindtmaxB);
	printf("dt=%f\n", dummy[i]);

	}
	*/


	//float diffdt = mindtmaxB - mindtmax;
	XParam.dt = mindtmaxB;
	if (ceil((nextoutputtime - XParam.totaltime) / XParam.dt)> 0.0)
	{
		XParam.dt = (nextoutputtime - XParam.totaltime) / ceil((nextoutputtime - XParam.totaltime) / XParam.dt);
	}
	//printf("dt=%f\n", XParam.dt);


	updateEV << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.lat*pi / 21600.0f, rightblk_g, topblk_g, hh_g, uu_g, vv_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	CUDA_CHECK(cudaDeviceSynchronize());




	//predictor (advance 1/2 dt)
	Advkernel << <gridDim, blockDim, 0 >> >((float)XParam.dt*0.5f, (float)XParam.eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//corrector setp
	//update again


	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hho_g, dhdx_g, dhdy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zso_g, dzsdx_g, dzsdy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uuo_g, dudx_g, dudy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vvo_g, dvdx_g, dvdy_g);

	CUDA_CHECK(cudaDeviceSynchronize());



	updateKurgX << <gridDim, blockDim, 0, streams[0] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, leftblk_g, hho_g, zso_g, uuo_g, vvo_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
	//CUDA_CHECK(cudaDeviceSynchronize());


	updateKurgY << <gridDim, blockDim, 0, streams[1] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, botblk_g, hho_g, zso_g, uuo_g, vvo_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	// no reduction of dtmax during the corrector step


	updateEV << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.lat*pi / 21600.0f, rightblk_g, topblk_g, hho_g, uuo_g, vvo_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	CUDA_CHECK(cudaDeviceSynchronize());



	//
	Advkernel << <gridDim, blockDim, 0 >> >((float)XParam.dt, (float)XParam.eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
	cleanupGPU << <gridDim, blockDim, 0 >> >(hho_g, zso_g, uuo_g, vvo_g, hh_g, zs_g, uu_g, vv_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//Bottom friction
	bottomfriction << <gridDim, blockDim, 0 >> > (XParam.frictionmodel, (float)XParam.dt, (float)XParam.eps, cf_g, hh_g, uu_g, vv_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaStreamDestroy(streams[0]));
	CUDA_CHECK(cudaStreamDestroy(streams[1]));

	// Impose no slip condition by default
	//noslipbndall << <gridDim, blockDim, 0 >> > (nx, ny, XParam.dt, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
	//CUDA_CHECK(cudaDeviceSynchronize());




	return XParam.dt;
}


double FlowGPUATM(Param XParam, double nextoutputtime)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	const int num_streams = 2;

	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}



	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	dtmax = (float)(1.0 / epsilon);
	//float dtmaxtmp = dtmax;

	resetdtmax << <gridDim, blockDim, 0, streams[0] >> > (dtmax_g);
	//CUDA_CHECK(cudaDeviceSynchronize());
	//update step 1



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hh_g, dhdx_g, dhdy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zs_g, dzsdx_g, dzsdy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uu_g, dudx_g, dudy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vv_g, dvdx_g, dvdy_g);



	CUDA_CHECK(cudaDeviceSynchronize());

	//CUDA_CHECK(cudaStreamSynchronize(streams[0]));
	//normal cartesian case
	updateKurgX << <gridDim, blockDim, 0, streams[0] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, leftblk_g, hh_g, zs_g, uu_g, vv_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//CUDA_CHECK(cudaStreamSynchronize(streams[1]));
	updateKurgY << <gridDim, blockDim, 0, streams[1] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, botblk_g, hh_g, zs_g, uu_g, vv_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);

	CUDA_CHECK(cudaDeviceSynchronize());



	//GPU Harris reduction #3. 8.3x reduction #0  Note #7 if a lot faster
	// This was successfully tested with a range of grid size
	//reducemax3 << <gridDimLine, blockDimLine, 64*sizeof(float) >> >(dtmax_g, arrmax_g, nx*ny)
	int s = XParam.nblk*XParam.blksize;
	int maxThreads = 256;
	int threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
	int blocks = (s + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
	dim3 blockDimLine(threads, 1, 1);
	dim3 gridDimLine(blocks, 1, 1);

	float mindtmaxB;

	reducemin3 << <gridDimLine, blockDimLine, smemSize >> > (dtmax_g, arrmax_g, s);
	CUDA_CHECK(cudaDeviceSynchronize());



	s = gridDimLine.x;
	while (s > 1)//cpuFinalThreshold
	{
		threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);

		smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

		dim3 blockDimLineS(threads, 1, 1);
		dim3 gridDimLineS(blocks, 1, 1);

		CUDA_CHECK(cudaMemcpy(dtmax_g, arrmax_g, s * sizeof(float), cudaMemcpyDeviceToDevice));

		reducemin3 << <gridDimLineS, blockDimLineS, smemSize >> > (dtmax_g, arrmax_g, s);
		CUDA_CHECK(cudaDeviceSynchronize());

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}


	CUDA_CHECK(cudaMemcpy(dummy, arrmax_g, 32 * sizeof(float), cudaMemcpyDeviceToHost));
	mindtmaxB = dummy[0];

	//32 seem safe here bu I wonder why it is not 1 for the largers arrays...
	/*
	for (int i = 0; i < 32; i++)
	{
	mindtmaxB = min(dummy[i], mindtmaxB);
	printf("dt=%f\n", dummy[i]);

	}
	*/


	//float diffdt = mindtmaxB - mindtmax;
	XParam.dt = mindtmaxB;
	if (ceil((nextoutputtime - XParam.totaltime) / XParam.dt)> 0.0)
	{
		XParam.dt = (nextoutputtime - XParam.totaltime) / ceil((nextoutputtime - XParam.totaltime) / XParam.dt);
	}
	//printf("dt=%f\n", XParam.dt);


	//updateEV << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, rightblk_g, topblk_g, hh_g, uu_g, vv_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	updateEVATM << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, (float)(XParam.lat*pi / 21600.0f), (float)XParam.windU.xo, (float)XParam.windU.yo, (float)XParam.windU.dx, (float)XParam.Cd, rightblk_g, topblk_g, blockxo_g, blockyo_g, hh_g, uu_g, vv_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	CUDA_CHECK(cudaDeviceSynchronize());




	//predictor (advance 1/2 dt)
	Advkernel << <gridDim, blockDim, 0 >> >((float)XParam.dt*0.5f, (float)XParam.eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//corrector setp
	//update again


	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hho_g, dhdx_g, dhdy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zso_g, dzsdx_g, dzsdy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uuo_g, dudx_g, dudy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vvo_g, dvdx_g, dvdy_g);

	CUDA_CHECK(cudaDeviceSynchronize());



	updateKurgX << <gridDim, blockDim, 0, streams[0] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, leftblk_g, hho_g, zso_g, uuo_g, vvo_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
	//CUDA_CHECK(cudaDeviceSynchronize());


	updateKurgY << <gridDim, blockDim, 0, streams[1] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, botblk_g, hho_g, zso_g, uuo_g, vvo_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	// no reduction of dtmax during the corrector step


	//updateEV << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, rightblk_g, topblk_g, hho_g, uuo_g, vvo_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	updateEVATM << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, (float)(XParam.lat*pi / 21600.0f), (float)XParam.windU.xo, (float)XParam.windU.yo, (float)XParam.windU.dx, (float)XParam.Cd, rightblk_g, topblk_g, blockxo_g, blockyo_g, hho_g, uuo_g, vvo_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);

	CUDA_CHECK(cudaDeviceSynchronize());



	//
	Advkernel << <gridDim, blockDim, 0 >> >((float)XParam.dt, (float)XParam.eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
	cleanupGPU << <gridDim, blockDim, 0 >> >(hho_g, zso_g, uuo_g, vvo_g, hh_g, zs_g, uu_g, vv_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//Bottom friction
	bottomfriction << <gridDim, blockDim, 0 >> > (XParam.frictionmodel, (float)XParam.dt, (float)XParam.eps, cf_g, hh_g, uu_g, vv_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaStreamDestroy(streams[0]));
	CUDA_CHECK(cudaStreamDestroy(streams[1]));

	// Impose no slip condition by default
	//noslipbndall << <gridDim, blockDim, 0 >> > (nx, ny, XParam.dt, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
	//CUDA_CHECK(cudaDeviceSynchronize());

	if (XParam.Rivers.size() > 0)
	{
		//
		dim3 gridDimRiver(XParam.nriverblock, 1, 1);
		float qnow;
		for (int Rin = 0; Rin < XParam.Rivers.size(); Rin++)
		{

			//qnow = interptime(slbnd[SLstepinbnd].wlev0, slbnd[SLstepinbnd - 1].wlev0, slbnd[SLstepinbnd].time - slbnd[SLstepinbnd - 1].time, totaltime - slbnd[SLstepinbnd - 1].time);
			int bndstep = 0;
			double difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
			while (difft <= 0.0) // danger?
			{
				bndstep++;
				difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
			}

			qnow = interptime(XParam.Rivers[Rin].flowinput[bndstep].q, XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].q, XParam.Rivers[Rin].flowinput[bndstep].time - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time, XParam.totaltime - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time);



			discharge_bnd_v << <gridDimRiver, blockDim, 0 >> > ((float)XParam.Rivers[Rin].xstart, (float)XParam.Rivers[Rin].xend, (float)XParam.Rivers[Rin].ystart, (float)XParam.Rivers[Rin].yend, (float)XParam.dx, (float)XParam.dt, qnow, (float)XParam.Rivers[Rin].disarea, Riverblk_g, blockxo_g, blockyo_g, zs_g, hh_g);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	}



	return XParam.dt;
}


double FlowGPUSpherical(Param XParam, double nextoutputtime)
{


	const int num_streams = 2;

	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}



	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0) / blockDim.x), ceil((ny*1.0) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	dtmax = (float)(1.0 / epsilon);
	//float dtmaxtmp = dtmax;

	resetdtmax << <gridDim, blockDim, 0, streams[0] >> > (dtmax_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());
	//update step 1




	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hh_gd, dhdx_gd, dhdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zs_gd, dzsdx_gd, dzsdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());




	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uu_gd, dudx_gd, dudy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vv_gd, dvdx_gd, dvdy_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//CUDA_CHECK(cudaStreamSynchronize(streams[0]));
	//Spherical
	{
		//Spherical coordinates
		updateKurgXSPH << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, leftblk_g, blockyo_gd, XParam.Radius, hh_gd, zs_gd, uu_gd, vv_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);

		updateKurgYSPH << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, botblk_g, blockyo_gd, XParam.Radius, hh_gd, zs_gd, uu_gd, vv_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);

		CUDA_CHECK(cudaDeviceSynchronize());

	}

	/////////////////////////////////////////////////////
	// Reduction of dtmax
	/////////////////////////////////////////////////////

	// copy from GPU and do the reduction on the CPU  ///LAME!


	//GPU Harris reduction #3. 8.3x reduction #0  Note #7 if a lot faster
	// This was successfully tested with a range of grid size
	//reducemax3 << <gridDimLine, blockDimLine, 64*sizeof(float) >> >(dtmax_g, arrmax_g, nx*ny)
	int s = XParam.nblk*XParam.blksize;
	int maxThreads = 256;
	int threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
	int blocks = (s + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);
	dim3 blockDimLine(threads, 1, 1);
	dim3 gridDimLine(blocks, 1, 1);

	double mindtmaxB;

	reducemin3 << <gridDimLine, blockDimLine, smemSize >> > (dtmax_gd, arrmax_gd, s);
	CUDA_CHECK(cudaDeviceSynchronize());



	s = gridDimLine.x;
	while (s > 1)//cpuFinalThreshold
	{
		threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);

		smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

		dim3 blockDimLineS(threads, 1, 1);
		dim3 gridDimLineS(blocks, 1, 1);

		CUDA_CHECK(cudaMemcpy(dtmax_gd, arrmax_gd, s * sizeof(double), cudaMemcpyDeviceToDevice));

		reducemin3 << <gridDimLineS, blockDimLineS, smemSize >> > (dtmax_gd, arrmax_gd, s);
		CUDA_CHECK(cudaDeviceSynchronize());

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}


	CUDA_CHECK(cudaMemcpy(dummy_d, arrmax_gd, 32 * sizeof(double), cudaMemcpyDeviceToHost));
	mindtmaxB = dummy_d[0];
	/*
	//32 seem safe here bu I wonder why it is not 1 for the largers arrays...

	for (int i = 0; i < 32; i++)
	{
	mindtmaxB = min(dummy[i], mindtmaxB);
	printf("dt=%f\n", dummy[i]);

	}
	*/


	//float diffdt = mindtmaxB - mindtmax;
	XParam.dt = mindtmaxB;
	if (ceil((nextoutputtime - XParam.totaltime) / XParam.dt)> 0.0)
	{
		XParam.dt = (nextoutputtime - XParam.totaltime) / ceil((nextoutputtime - XParam.totaltime) / XParam.dt);
	}
	//printf("dt=%f\n", XParam.dt);

	//spherical
	{
		//if spherical corrdinate use this kernel with the right corrections
		updateEVSPH << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.yo, XParam.ymax, XParam.Radius, rightblk_g, topblk_g, blockyo_gd, hh_gd, uu_gd, vv_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}


	//predictor (advance 1/2 dt)
	Advkernel << <gridDim, blockDim, 0 >> >(XParam.dt*0.5, XParam.eps, hh_gd, zb_gd, uu_gd, vv_gd, dh_gd, dhu_gd, dhv_gd, zso_gd, hho_gd, uuo_gd, vvo_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//corrector setp
	//update again
	// calculate gradients
	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hho_g, dhdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hho_g, dhdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hho_gd, dhdx_gd, dhdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, zso_g, dzsdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, zso_g, dzsdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zso_gd, dzsdx_gd, dzsdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, uuo_g, dudx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, uuo_g, dudy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uuo_gd, dudx_gd, dudy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, vvo_g, dvdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, vvo_g, dvdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vvo_gd, dvdx_gd, dvdy_gd);

	CUDA_CHECK(cudaDeviceSynchronize());


	// Test whether it is better to have one here or later (are the instuctions overlap if occupancy and meme acess is available?)
	//CUDA_CHECK(cudaDeviceSynchronize());


	{
		//Spherical coordinates
		updateKurgXSPH << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, leftblk_g, blockyo_gd, XParam.Radius, hho_gd, zso_gd, uuo_gd, vvo_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);

		updateKurgYSPH << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, botblk_g, blockyo_gd, XParam.Radius, hho_gd, zso_gd, uuo_gd, vvo_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);

		CUDA_CHECK(cudaDeviceSynchronize());

	}
	// no reduction of dtmax during the corrector step


	{
		//if spherical corrdinate use this kernel with the right corrections
		updateEVSPH << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.yo, XParam.ymax, XParam.Radius, rightblk_g, topblk_g, blockyo_gd, hho_gd, uuo_gd, vvo_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	//
	Advkernel << <gridDim, blockDim, 0 >> >(XParam.dt, XParam.eps, hh_gd, zb_gd, uu_gd, vv_gd, dh_gd, dhu_gd, dhv_gd, zso_gd, hho_gd, uuo_gd, vvo_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
	cleanupGPU << <gridDim, blockDim, 0 >> >(hho_gd, zso_gd, uuo_gd, vvo_gd, hh_gd, zs_gd, uu_gd, vv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//Bottom friction
	bottomfriction << <gridDim, blockDim, 0 >> > (XParam.frictionmodel, XParam.dt, XParam.eps, cf_gd, hh_gd, uu_gd, vv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaStreamDestroy(streams[0]));
	CUDA_CHECK(cudaStreamDestroy(streams[1]));

	// Impose no slip condition by default
	//noslipbndall << <gridDim, blockDim, 0 >> > (nx, ny, XParam.dt, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
	//CUDA_CHECK(cudaDeviceSynchronize());
	return XParam.dt;
}


double FlowGPUDouble(Param XParam, double nextoutputtime)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	const int num_streams = 2;

	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}

	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);


	dtmax = (float)(1.0 / epsilon);
	//float dtmaxtmp = dtmax;

	resetdtmax << <gridDim, blockDim, 0, streams[0] >> > (dtmax_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());
	//update step 1




	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hh_gd, dhdx_gd, dhdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zs_gd, dzsdx_gd, dzsdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());




	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uu_gd, dudx_gd, dudy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vv_gd, dvdx_gd, dvdy_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//CUDA_CHECK(cudaStreamSynchronize(streams[0]));


	



	updateKurgXD << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, leftblk_g, hh_gd, zs_gd, uu_gd, vv_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);

	updateKurgYD << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, botblk_g, hh_gd, zs_gd, uu_gd, vv_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);

	CUDA_CHECK(cudaDeviceSynchronize());



	/////////////////////////////////////////////////////
	// Reduction of dtmax
	/////////////////////////////////////////////////////

	// copy from GPU and do the reduction on the CPU  ///LAME!


	//GPU Harris reduction #3. 8.3x reduction #0  Note #7 if a lot faster
	// This was successfully tested with a range of grid size
	//reducemax3 << <gridDimLine, blockDimLine, 64*sizeof(float) >> >(dtmax_g, arrmax_g, nx*ny)
	int s = XParam.nblk*XParam.blksize;
	int maxThreads = 256;
	int threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
	int blocks = (s + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);
	dim3 blockDimLine(threads, 1, 1);
	dim3 gridDimLine(blocks, 1, 1);

	double mindtmaxB;

	reducemin3 << <gridDimLine, blockDimLine, smemSize >> > (dtmax_gd, arrmax_gd, s);
	CUDA_CHECK(cudaDeviceSynchronize());



	s = gridDimLine.x;
	while (s > 1)//cpuFinalThreshold
	{
		threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);

		smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

		dim3 blockDimLineS(threads, 1, 1);
		dim3 gridDimLineS(blocks, 1, 1);

		CUDA_CHECK(cudaMemcpy(dtmax_gd, arrmax_gd, s * sizeof(double), cudaMemcpyDeviceToDevice));

		reducemin3 << <gridDimLineS, blockDimLineS, smemSize >> > (dtmax_gd, arrmax_gd, s);
		CUDA_CHECK(cudaDeviceSynchronize());

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}


	CUDA_CHECK(cudaMemcpy(dummy_d, arrmax_gd, 32 * sizeof(double), cudaMemcpyDeviceToHost));
	mindtmaxB = dummy_d[0];
	/*
	//32 seem safe here bu I wonder why it is not 1 for the largers arrays...

	for (int i = 0; i < 32; i++)
	{
	mindtmaxB = min(dummy[i], mindtmaxB);
	printf("dt=%f\n", dummy[i]);

	}
	*/


	//float diffdt = mindtmaxB - mindtmax;
	XParam.dt = mindtmaxB;
	if (ceil((nextoutputtime - XParam.totaltime) / XParam.dt)> 0.0)
	{
		XParam.dt = (nextoutputtime - XParam.totaltime) / ceil((nextoutputtime - XParam.totaltime) / XParam.dt);
	}
	//printf("dt=%f\n", XParam.dt);


	//if spherical corrdinate use this kernel with the right corrections
	updateEVD << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.lat*pi / 21600.0, rightblk_g, topblk_g, hh_gd, uu_gd, vv_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());



	//predictor (advance 1/2 dt)
	Advkernel << <gridDim, blockDim, 0 >> >(XParam.dt*0.5, XParam.eps, hh_gd, zb_gd, uu_gd, vv_gd, dh_gd, dhu_gd, dhv_gd, zso_gd, hho_gd, uuo_gd, vvo_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//uvcorr << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, hho_gd, uuo_gd, vvo_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//corrector setp
	//update again
	// calculate gradients
	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hho_g, dhdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hho_g, dhdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hho_gd, dhdx_gd, dhdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, zso_g, dzsdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, zso_g, dzsdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zso_gd, dzsdx_gd, dzsdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, uuo_g, dudx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, uuo_g, dudy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uuo_gd, dudx_gd, dudy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, vvo_g, dvdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, vvo_g, dvdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vvo_gd, dvdx_gd, dvdy_gd);

	CUDA_CHECK(cudaDeviceSynchronize());


	// Test whether it is better to have one here or later (are the instuctions overlap if occupancy and meme acess is available?)
	//CUDA_CHECK(cudaDeviceSynchronize());

	


	updateKurgXD << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, leftblk_g, hho_gd, zso_gd, uuo_gd, vvo_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);

	updateKurgYD << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, botblk_g, hho_gd, zso_gd, uuo_gd, vvo_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);

	CUDA_CHECK(cudaDeviceSynchronize());


	// no reduction of dtmax during the corrector step




	updateEVD << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.lat*pi / 21600.0, rightblk_g, topblk_g, hho_gd, uuo_gd, vvo_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());


	//
	Advkernel << <gridDim, blockDim, 0 >> >(XParam.dt, XParam.eps, hh_gd, zb_gd, uu_gd, vv_gd, dh_gd, dhu_gd, dhv_gd, zso_gd, hho_gd, uuo_gd, vvo_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//uvcorr << <gridDim, blockDim, 0 >> > (XParam.delta, hho_gd, uuo_gd, vvo_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
	cleanupGPU << <gridDim, blockDim, 0 >> >(hho_gd, zso_gd, uuo_gd, vvo_gd, hh_gd, zs_gd, uu_gd, vv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//Bottom friction
	bottomfriction << <gridDim, blockDim, 0 >> > (XParam.frictionmodel, XParam.dt, XParam.eps, cf_gd, hh_gd, uu_gd, vv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaStreamDestroy(streams[0]));
	CUDA_CHECK(cudaStreamDestroy(streams[1]));

	// Impose no slip condition by default
	//noslipbndall << <gridDim, blockDim, 0 >> > (nx, ny, XParam.dt, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
	//CUDA_CHECK(cudaDeviceSynchronize());

	if (XParam.Rivers.size() > 0)
	{
		dim3 gridDimRiver(XParam.nriverblock, 1, 1);
		//
		double qnow;
		for (int Rin = 0; Rin < XParam.Rivers.size(); Rin++)
		{

			//qnow = interptime(slbnd[SLstepinbnd].wlev0, slbnd[SLstepinbnd - 1].wlev0, slbnd[SLstepinbnd].time - slbnd[SLstepinbnd - 1].time, totaltime - slbnd[SLstepinbnd - 1].time);
			int bndstep = 0;
			double difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
			while (difft <= 0.0) // danger?
			{
				bndstep++;
				difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
			}

			qnow = interptime(XParam.Rivers[Rin].flowinput[bndstep].q, XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].q, XParam.Rivers[Rin].flowinput[bndstep].time - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time, XParam.totaltime - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time);



			discharge_bnd_v << <gridDimRiver, blockDim, 0 >> > (XParam.Rivers[Rin].xstart, XParam.Rivers[Rin].xend, XParam.Rivers[Rin].ystart, XParam.Rivers[Rin].yend, XParam.dx, XParam.dt, qnow, XParam.Rivers[Rin].disarea, Riverblk_g, blockxo_gd, blockyo_gd, zs_gd, hh_gd);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	}




	return XParam.dt;
}


void meanmaxvarGPU(Param XParam)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	if (XParam.outuumean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >(uumean_g, uu_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outvvmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >(vvmean_g, vv_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outhhmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >(hhmean_g, hh_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outzsmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >(zsmean_g, zs_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outzsmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(zsmax_g, zs_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outhhmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(hhmax_g, hh_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outuumax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(uumax_g, uu_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outvvmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(vvmax_g, vv_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

}


void meanmaxvarGPUD(Param XParam)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0) / blockDim.x), ceil((ny*1.0) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	if (XParam.outuumean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >(uumean_gd, uu_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outvvmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >(vvmean_gd, vv_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outhhmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >(hhmean_gd, hh_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outzsmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >(zsmean_gd, zs_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outzsmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(zsmax_gd, zs_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outhhmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(hhmax_gd, hh_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outuumax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(uumax_gd, uu_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outvvmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(vvmax_gd, vv_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

}


void DivmeanvarGPU(Param XParam, float nstep)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	if (XParam.outuumean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >(nstep, uumean_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outvvmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >(nstep, vvmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outhhmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >(nstep, hhmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outzsmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >(nstep, zsmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}



}


void DivmeanvarGPUD(Param XParam, double nstep)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0) / blockDim.x), ceil((ny*1.0) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	if (XParam.outuumean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >(nstep, uumean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outvvmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >(nstep, vvmean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outhhmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >(nstep, hhmean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outzsmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >(nstep, zsmean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}



}

void ResetmeanvarGPU(Param XParam)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	if (XParam.outuumean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(uumean_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outvvmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(vvmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outhhmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(hhmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outzsmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(zsmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}




void ResetmeanvarGPUD(Param XParam)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0) / blockDim.x), ceil((ny*1.0) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	if (XParam.outuumean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(uumean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outvvmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(vvmean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outhhmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(hhmean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outzsmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(zsmean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}
void ResetmaxvarGPU(Param XParam)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	if (XParam.outuumax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >(uumax_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outvvmax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >(vvmax_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outhhmax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >(hhmax_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outzsmax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >(zsmax_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}
void ResetmaxvarGPUD(Param XParam)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	if (XParam.outuumax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >(uumax_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outvvmax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >(vvmax_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outhhmax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >(hhmax_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outzsmax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >(zsmax_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}


void RiverSource(Param XParam)
{
	//
	dim3 gridDimRiver(XParam.nriverblock, 1, 1);
	dim3 blockDim(16, 16, 1);
	float qnow;
	for (int Rin = 0; Rin < XParam.Rivers.size(); Rin++)
	{

		//qnow = interptime(slbnd[SLstepinbnd].wlev0, slbnd[SLstepinbnd - 1].wlev0, slbnd[SLstepinbnd].time - slbnd[SLstepinbnd - 1].time, totaltime - slbnd[SLstepinbnd - 1].time);
		int bndstep = 0;
		double difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
		while (difft <= 0.0) // danger?
		{
			bndstep++;
			difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
		}

		qnow = interptime(XParam.Rivers[Rin].flowinput[bndstep].q, XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].q, XParam.Rivers[Rin].flowinput[bndstep].time - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time, XParam.totaltime - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time);


		discharge_bnd_v << <gridDimRiver, blockDim, 0 >> > ((float)XParam.Rivers[Rin].xstart, (float)XParam.Rivers[Rin].xend, (float)XParam.Rivers[Rin].ystart, (float)XParam.Rivers[Rin].yend, (float)XParam.dx, (float)XParam.dt, qnow, (float)XParam.Rivers[Rin].disarea, Riverblk_g, blockxo_g, blockyo_g, zs_g, hh_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
}

void RiverSourceD(Param XParam)
{
	//
	dim3 gridDimRiver(XParam.nriverblock, 1, 1);
	dim3 blockDim(16, 16, 1);
	double qnow;
	for (int Rin = 0; Rin < XParam.Rivers.size(); Rin++)
	{

		//qnow = interptime(slbnd[SLstepinbnd].wlev0, slbnd[SLstepinbnd - 1].wlev0, slbnd[SLstepinbnd].time - slbnd[SLstepinbnd - 1].time, totaltime - slbnd[SLstepinbnd - 1].time);
		int bndstep = 0;
		double difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
		while (difft <= 0.0) // danger?
		{
			bndstep++;
			difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
		}

		qnow = interptime(XParam.Rivers[Rin].flowinput[bndstep].q, XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].q, XParam.Rivers[Rin].flowinput[bndstep].time - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time, XParam.totaltime - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time);

		// no a great if statement

		discharge_bnd_v << <gridDimRiver, blockDim, 0 >> > (XParam.Rivers[Rin].xstart, XParam.Rivers[Rin].xend, XParam.Rivers[Rin].ystart, XParam.Rivers[Rin].yend, XParam.dx, XParam.dt, qnow, XParam.Rivers[Rin].disarea, Riverblk_g, blockxo_gd, blockyo_gd, zs_gd, hh_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
}


//external forcing functions prepartions

double Rainthisstep(Param XParam, dim3 gridDimRain, dim3 blockDimRain, int & rainstep)
{
	double rainuni = 0.0;
	if (XParam.Rainongrid.uniform == 1)
	{
		//
		int Rstepinbnd = 1;



		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = XParam.Rainongrid.data[Rstepinbnd].time - XParam.totaltime;

		while (difft < 0.0)
		{
			Rstepinbnd++;
			difft = XParam.Rainongrid.data[Rstepinbnd].time - XParam.totaltime;
		}

		rainuni = interptime(XParam.Rainongrid.data[Rstepinbnd].wspeed, XParam.Rainongrid.data[Rstepinbnd - 1].wspeed, XParam.Rainongrid.data[Rstepinbnd].time - XParam.Rainongrid.data[Rstepinbnd - 1].time, XParam.totaltime - XParam.Rainongrid.data[Rstepinbnd - 1].time);



	}
	else
	{
		int readfirststep = min(max((int)floor((XParam.totaltime - XParam.Rainongrid.to) / XParam.Rainongrid.dt), 0), XParam.Rainongrid.nt - 2);

		if (readfirststep + 1 > rainstep)
		{
			// Need to read a new step from the file
			NextHDstep << <gridDimRain, blockDimRain, 0 >> > (XParam.Rainongrid.nx, XParam.Rainongrid.ny, Rainbef_g, Rainaft_g);
			CUDA_CHECK(cudaDeviceSynchronize());


			readATMstep(XParam.Rainongrid, readfirststep + 1, Rainaft);
			CUDA_CHECK(cudaMemcpy(Rainaft_g, Rainaft, XParam.Rainongrid.nx*XParam.Rainongrid.ny * sizeof(float), cudaMemcpyHostToDevice));


			rainstep = readfirststep + 1;
		}

		HD_interp << < gridDimRain, blockDimRain, 0 >> > (XParam.Rainongrid.nx, XParam.Rainongrid.ny, 0, rainstep - 1, (float) XParam.totaltime, (float) XParam.Rainongrid.dt, Rainbef_g, Rainaft_g, Rain_g);
		CUDA_CHECK(cudaDeviceSynchronize());


		//InterpstepCPU(XParam.windU.nx, XParam.windU.ny, readfirststep, XParam.totaltime, XParam.windU.dt, Uwind, Uwbef, Uwaft);
		//InterpstepCPU(XParam.windV.nx, XParam.windV.ny, readfirststep, XParam.totaltime, XParam.windV.dt, Vwind, Vwbef, Vwaft);

		//below should be async so other streams can keep going
		CUDA_CHECK(cudaMemcpyToArray(Rain_gp, 0, 0, Rain_g, XParam.Rainongrid.nx*XParam.Rainongrid.ny * sizeof(float), cudaMemcpyDeviceToDevice));

	}

	return rainuni;
}

template <class T>
void Windthisstep(Param XParam, dim3 gridDimWND, dim3 blockDimWND, cudaStream_t stream, int & windstep, T & uwinduni, T & vwinduni)
{
	//
	if (XParam.windU.uniform == 1)
	{
		//
		int Wstepinbnd = 1;



		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = XParam.windU.data[Wstepinbnd].time - XParam.totaltime;

		while (difft < 0.0)
		{
			Wstepinbnd++;
			difft = XParam.windU.data[Wstepinbnd].time - XParam.totaltime;
		}

		uwinduni = interptime(XParam.windU.data[Wstepinbnd].uwind, XParam.windU.data[Wstepinbnd - 1].uwind, XParam.windU.data[Wstepinbnd].time - XParam.windU.data[Wstepinbnd - 1].time, XParam.totaltime - XParam.windU.data[Wstepinbnd - 1].time);
		vwinduni = interptime(XParam.windU.data[Wstepinbnd].vwind, XParam.windU.data[Wstepinbnd - 1].vwind, XParam.windU.data[Wstepinbnd].time - XParam.windU.data[Wstepinbnd - 1].time, XParam.totaltime - XParam.windU.data[Wstepinbnd - 1].time);
	}
	else
	{
		int readfirststep = min(max((int)floor((XParam.totaltime - XParam.windU.to) / XParam.windU.dt), 0), XParam.windU.nt - 2);

		if (readfirststep + 1 > windstep)
		{
			// Need to read a new step from the file
			NextHDstep << <gridDimWND, blockDimWND, 0, stream >> > (XParam.windU.nx, XParam.windU.ny, Uwbef_g, Uwaft_g);


			NextHDstep << <gridDimWND, blockDimWND, 0, stream >> > (XParam.windV.nx, XParam.windV.ny, Vwbef_g, Vwaft_g);
			CUDA_CHECK(cudaStreamSynchronize(stream));




			readWNDstep(XParam.windU, XParam.windV, readfirststep + 1, Uwaft, Vwaft);
			CUDA_CHECK(cudaMemcpy(Uwaft_g, Uwaft, XParam.windU.nx*XParam.windU.ny * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(Vwaft_g, Vwaft, XParam.windU.nx*XParam.windU.ny * sizeof(float), cudaMemcpyHostToDevice));

			windstep = readfirststep + 1;
		}

		HD_interp << < gridDimWND, blockDimWND, 0, stream >> > (XParam.windU.nx, XParam.windU.ny, 0, windstep - 1, (float)XParam.totaltime, (float) XParam.windU.dt, Uwbef_g, Uwaft_g, Uwind_g);


		HD_interp << <gridDimWND, blockDimWND, 0, stream >> > (XParam.windV.nx, XParam.windV.ny, 0, windstep - 1, (float)XParam.totaltime, (float) XParam.windU.dt, Vwbef_g, Vwaft_g, Vwind_g);
		CUDA_CHECK(cudaStreamSynchronize(stream));

		//InterpstepCPU(XParam.windU.nx, XParam.windU.ny, readfirststep, XParam.totaltime, XParam.windU.dt, Uwind, Uwbef, Uwaft);
		//InterpstepCPU(XParam.windV.nx, XParam.windV.ny, readfirststep, XParam.totaltime, XParam.windV.dt, Vwind, Vwbef, Vwaft);

		//below should be async so other streams can keep going
		CUDA_CHECK(cudaMemcpyToArray(Uwind_gp, 0, 0, Uwind_g, XParam.windU.nx*XParam.windU.ny * sizeof(float), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpyToArray(Vwind_gp, 0, 0, Vwind_g, XParam.windV.nx*XParam.windV.ny * sizeof(float), cudaMemcpyDeviceToDevice));
	}
}

//wind input is only in float (not double) because it is mapped to the texture hich is way wayu way faster in float.. who cares about doubel wind....
//void WindthisstepD(Param XParam, dim3 gridDimWND, dim3 blockDimWND, cudaStream_t stream, int & windstep, double & uwinduni, double & vwinduni)


void AtmPthisstep(Param XParam, dim3 gridDimATM, dim3 blockDimATM, int & atmpstep)
{
	//
	int readfirststep = min(max((int)floor((XParam.totaltime - XParam.atmP.to) / XParam.atmP.dt), 0), XParam.atmP.nt - 2);

	if (readfirststep + 1 > atmpstep)
	{
		// Need to read a new step from the file
		NextHDstep << <gridDimATM, blockDimATM, 0 >> > (XParam.atmP.nx, XParam.atmP.ny, Patmbef_g, Patmaft_g);
		CUDA_CHECK(cudaDeviceSynchronize());




		readATMstep(XParam.atmP, readfirststep + 1, Patmaft);
		CUDA_CHECK(cudaMemcpy(Patmaft_g, Patmaft, XParam.windU.nx*XParam.windU.ny * sizeof(float), cudaMemcpyHostToDevice));


		atmpstep = atmpstep + 1;
	}

	HD_interp << < gridDimATM, blockDimATM, 0 >> > (XParam.atmP.nx, XParam.atmP.ny, 0, atmpstep - 1, (float)XParam.totaltime, (float)XParam.atmP.dt, Patmbef_g, Patmaft_g, PatmX_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpyToArray(Patm_gp, 0, 0, PatmX_g, XParam.atmP.nx*XParam.atmP.ny * sizeof(float), cudaMemcpyDeviceToDevice));
}




void pointoutputstep(Param XParam, dim3 gridDim, dim3 blockDim, int & nTSsteps, std::vector< std::vector< Pointout > > & zsAllout)
{
	//
	FILE * fsSLTS;
	for (int o = 0; o < XParam.TSnodesout.size(); o++)
	{
		//
		Pointout stepread;

		stepread.time = XParam.totaltime;
		stepread.zs = 0.0;// a bit useless this
		stepread.hh = 0.0;
		stepread.uu = 0.0;
		stepread.vv = 0.0;
		zsAllout[o].push_back(stepread);

		if (XParam.spherical == 1 || XParam.doubleprecision == 1)
		{
			storeTSout << <gridDim, blockDim, 0 >> > ((int)XParam.TSnodesout.size(), o, nTSsteps, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSnodesout[o].block, zs_gd, hh_gd, uu_gd, vv_gd, TSstore_gd);
		}
		else
		{
			storeTSout << <gridDim, blockDim, 0 >> > ((int)XParam.TSnodesout.size(), o, nTSsteps, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSnodesout[o].block, zs_g, hh_g, uu_g, vv_g, TSstore_g);
		}

		CUDA_CHECK(cudaDeviceSynchronize());
	}
	nTSsteps++;

	if ((nTSsteps + 1)*XParam.TSnodesout.size() * 4 > 2048 || XParam.endtime - XParam.totaltime <= XParam.dt*0.00001f)
	{
		//Flush
		if (XParam.spherical == 1 || XParam.doubleprecision == 1)
		{
			CUDA_CHECK(cudaMemcpy(TSstore_d, TSstore_gd, 2048 * sizeof(double), cudaMemcpyDeviceToHost));
			for (int o = 0; o < XParam.TSnodesout.size(); o++)
			{
				fsSLTS = fopen(XParam.TSoutfile[o].c_str(), "a");


				for (int n = 0; n < nTSsteps; n++)
				{
					//


					fprintf(fsSLTS, "%f\t%.4f\t%.4f\t%.4f\t%.4f\n", zsAllout[o][n].time, TSstore_d[1 + o * 4 + n*XParam.TSnodesout.size() * 4], TSstore_d[0 + o * 4 + n*XParam.TSnodesout.size() * 4], TSstore_d[2 + o * 4 + n*XParam.TSnodesout.size() * 4], TSstore_d[3 + o * 4 + n*XParam.TSnodesout.size() * 4]);


				}
				fclose(fsSLTS);
				//reset zsout
				zsAllout[o].clear();
			}
		}
		else
		{
			CUDA_CHECK(cudaMemcpy(TSstore, TSstore_g, 2048 * sizeof(float), cudaMemcpyDeviceToHost));


		for (int o = 0; o < XParam.TSnodesout.size(); o++)
		{
			fsSLTS = fopen(XParam.TSoutfile[o].c_str(), "a");


			for (int n = 0; n < nTSsteps; n++)
			{
				//


				fprintf(fsSLTS, "%f\t%.4f\t%.4f\t%.4f\t%.4f\n", zsAllout[o][n].time, TSstore[1 + o * 4 + n*XParam.TSnodesout.size() * 4], TSstore[0 + o * 4 + n*XParam.TSnodesout.size() * 4], TSstore[2 + o * 4 + n*XParam.TSnodesout.size() * 4], TSstore[3 + o * 4 + n*XParam.TSnodesout.size() * 4]);


			}
			fclose(fsSLTS);
			//reset zsout
			zsAllout[o].clear();
		}
		}
		nTSsteps = 0;




	}
}


template <class T> double Calcmaxdt(Param XParam, T *dtmax, T *arrmax )
{
	//GPU Harris reduction #3. 8.3x reduction #0  Note #7 if a lot faster
	// This was successfully tested with a range of grid size
	//reducemax3 << <gridDimLine, blockDimLine, 64*sizeof(float) >> >(dtmax_g, arrmax_g, nx*ny)
	int s = XParam.nblk*XParam.blksize;
	int maxThreads = 256;
	int threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
	int blocks = (s + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
	dim3 blockDimLine(threads, 1, 1);
	dim3 gridDimLine(blocks, 1, 1);

	T mindtmaxB[32];

	reducemin3 << <gridDimLine, blockDimLine, smemSize >> > (dtmax, arrmax, s);
	CUDA_CHECK(cudaDeviceSynchronize());



	s = gridDimLine.x;
	while (s > 1)//cpuFinalThreshold
	{
		threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);

		smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

		dim3 blockDimLineS(threads, 1, 1);
		dim3 gridDimLineS(blocks, 1, 1);

		CUDA_CHECK(cudaMemcpy(dtmax, arrmax, s * sizeof(T), cudaMemcpyDeviceToDevice));

		reducemin3 << <gridDimLineS, blockDimLineS, smemSize >> > (dtmax, arrmax, s);
		CUDA_CHECK(cudaDeviceSynchronize());

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}


	CUDA_CHECK(cudaMemcpy(mindtmaxB, arrmax, 32 * sizeof(T), cudaMemcpyDeviceToHost));
	//mindtmaxB = dummy[0];

	//32 seem safe here bu I wonder why it is not 1 for the largers arrays...
	/*
	for (int i = 0; i < 32; i++)
	{
	mindtmaxB = min(dummy[i], mindtmaxB);
	printf("dt=%f\n", dummy[i]);

	}
	*/


	//float diffdt = mindtmaxB - mindtmax;
	return double(mindtmaxB[0]);
}


template <class T> void ApplyDeform(Param XParam,dim3 blockDim,dim3 gridDim, T *&dummy, T *&dh,T *&hh, T *&zs, T *&zb )
{
	float * def_f;
	T *def;
	//Check each deform input
	for (int nd = 0; nd < XParam.deform.size(); nd++)
	{
		Allocate1CPU(XParam.deform[nd].grid.nx, XParam.deform[nd].grid.ny, def);
		Allocate1CPU(XParam.deform[nd].grid.nx, XParam.deform[nd].grid.ny, def_f);
		if ((XParam.totaltime - XParam.deform[nd].startime) <= XParam.dt && (XParam.totaltime - XParam.deform[nd].startime)>0.0)
		{
			readmapdata(XParam.deform[nd].grid, def_f);

			//Should skip this part for float ops


			for (int k = 0; k<(XParam.deform[nd].grid.nx*XParam.deform[nd].grid.ny); k++)
			{
				def[k] = def_f[k];
			}



			interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.deform[nd].grid.nx, XParam.deform[nd].grid.ny, XParam.deform[nd].grid.xo, XParam.deform[nd].grid.xmax, XParam.deform[nd].grid.yo, XParam.deform[nd].grid.ymax, XParam.deform[nd].grid.dx, def, dummy);

			CUDA_CHECK(cudaMemcpy(dh, dummy, XParam.nblk*XParam.blksize * sizeof(T), cudaMemcpyHostToDevice));
			if (XParam.deform[nd].duration > 0.0)
			{

				//do zs=zs+dummy/duration *(XParam.totaltime - XParam.deform[nd].startime);
				Deform << <gridDim, blockDim, 0 >> > (T(1.0 / XParam.deform[nd].duration *(XParam.totaltime - XParam.deform[nd].startime)), dh, zs, zb);
				CUDA_CHECK(cudaDeviceSynchronize());
			}

			else
			{
				//do zs=zs+dummy;
				Deform << <gridDim, blockDim, 0 >> > (T(1.0), dh, zs, zb);
				CUDA_CHECK(cudaDeviceSynchronize());
			}

		}
		else if ((XParam.totaltime - XParam.deform[nd].startime) > XParam.dt && XParam.totaltime <= (XParam.deform[nd].startime + XParam.deform[nd].duration))
		{
			// read the data and store to dummy
			readmapdata(XParam.deform[nd].grid, def_f);
			for (int k = 0; k<(XParam.deform[nd].grid.nx*XParam.deform[nd].grid.ny); k++)
			{
				def[k] = def_f[k];
			}


				interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.deform[nd].grid.nx, XParam.deform[nd].grid.ny, XParam.deform[nd].grid.xo, XParam.deform[nd].grid.xmax, XParam.deform[nd].grid.yo, XParam.deform[nd].grid.ymax, XParam.deform[nd].grid.dx, def, dummy);
				CUDA_CHECK(cudaMemcpy(dh, dummy, XParam.nblk*XParam.blksize * sizeof(T), cudaMemcpyHostToDevice));

			// DO zs=zs+dummy/duration*dt
			Deform << <gridDim, blockDim, 0 >> > (T(1.0 / XParam.deform[nd].duration *XParam.dt), dh, zs, zb);
			CUDA_CHECK(cudaDeviceSynchronize());


		}
		free(def_f);
		free(def);

	}
}

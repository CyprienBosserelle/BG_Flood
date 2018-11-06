﻿
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


int AllocMemGPUBND(Param XParam)
{
	// Allocate textures and bind arrays for boundary interpolation
	if (XParam.leftbnd.on)
	{
		//leftWLbnd = readWLfile(XParam.leftbndfile);
		//Flatten bnd to copy to cuda array
		int nbndtimes = (int)XParam.leftbnd.data.size();
		int nbndvec = (int)XParam.leftbnd.data[0].wlevs.size();
		CUDA_CHECK(cudaMallocArray(&leftWLS_gp, &channelDescleftbnd, nbndtimes, nbndvec));
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

		texLBND.addressMode[0] = cudaAddressModeClamp;
		texLBND.addressMode[1] = cudaAddressModeClamp;
		texLBND.filterMode = cudaFilterModeLinear;
		texLBND.normalized = false;


		CUDA_CHECK(cudaBindTextureToArray(texLBND, leftWLS_gp, channelDescleftbnd));
		free(leftWLS);

	}
	if (XParam.rightbnd.on)
	{
		//leftWLbnd = readWLfile(XParam.leftbndfile);
		//Flatten bnd to copy to cuda array
		int nbndtimes = (int)XParam.rightbnd.data.size();
		int nbndvec = (int)XParam.rightbnd.data[0].wlevs.size();
		CUDA_CHECK(cudaMallocArray(&rightWLS_gp, &channelDescrightbnd, nbndtimes, nbndvec));

		float * rightWLS;
		rightWLS = (float *)malloc(nbndtimes * nbndvec * sizeof(float));

		for (int ibndv = 0; ibndv < nbndvec; ibndv++)
		{
			for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
			{
				//
				rightWLS[ibndt + ibndv*nbndtimes] = XParam.rightbnd.data[ibndt].wlevs[ibndv];
			}
		}
		CUDA_CHECK(cudaMemcpyToArray(rightWLS_gp, 0, 0, rightWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

		texRBND.addressMode[0] = cudaAddressModeClamp;
		texRBND.addressMode[1] = cudaAddressModeClamp;
		texRBND.filterMode = cudaFilterModeLinear;
		texRBND.normalized = false;


		CUDA_CHECK(cudaBindTextureToArray(texRBND, rightWLS_gp, channelDescrightbnd));
		free(rightWLS);

	}
	if (XParam.topbnd.on)
	{
		//leftWLbnd = readWLfile(XParam.leftbndfile);
		//Flatten bnd to copy to cuda array
		int nbndtimes = (int)XParam.topbnd.data.size();
		int nbndvec = (int)XParam.topbnd.data[0].wlevs.size();
		CUDA_CHECK(cudaMallocArray(&topWLS_gp, &channelDesctopbnd, nbndtimes, nbndvec));

		float * topWLS;
		topWLS = (float *)malloc(nbndtimes * nbndvec * sizeof(float));

		for (int ibndv = 0; ibndv < nbndvec; ibndv++)
		{
			for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
			{
				//
				topWLS[ibndt + ibndv*nbndtimes] = XParam.topbnd.data[ibndt].wlevs[ibndv];
			}
		}
		CUDA_CHECK(cudaMemcpyToArray(topWLS_gp, 0, 0, topWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

		texTBND.addressMode[0] = cudaAddressModeClamp;
		texTBND.addressMode[1] = cudaAddressModeClamp;
		texTBND.filterMode = cudaFilterModeLinear;
		texTBND.normalized = false;


		CUDA_CHECK(cudaBindTextureToArray(texTBND, topWLS_gp, channelDesctopbnd));
		free(topWLS);

	}
	if (XParam.botbnd.on)
	{
		//leftWLbnd = readWLfile(XParam.leftbndfile);
		//Flatten bnd to copy to cuda array
		int nbndtimes = (int)XParam.botbnd.data.size();
		int nbndvec = (int)XParam.botbnd.data[0].wlevs.size();
		CUDA_CHECK(cudaMallocArray(&botWLS_gp, &channelDescbotbnd, nbndtimes, nbndvec));

		float * botWLS;
		botWLS = (float *)malloc(nbndtimes * nbndvec * sizeof(float));

		for (int ibndv = 0; ibndv < nbndvec; ibndv++)
		{
			for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
			{
				//
				botWLS[ibndt + ibndv*nbndtimes] = XParam.botbnd.data[ibndt].wlevs[ibndv];
			}
		}
		CUDA_CHECK(cudaMemcpyToArray(botWLS_gp, 0, 0, botWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

		texBBND.addressMode[0] = cudaAddressModeClamp;
		texBBND.addressMode[1] = cudaAddressModeClamp;
		texBBND.filterMode = cudaFilterModeLinear;
		texBBND.normalized = false;


		CUDA_CHECK(cudaBindTextureToArray(texBBND, botWLS_gp, channelDescbotbnd));
		free(botWLS);

	}
	return 1;
}



void LeftFlowBnd(Param XParam)
{
	//
	int nx = XParam.nx;
	int ny = XParam.ny;
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



		dim3 blockDim(16, 16, 1);
		dim3 gridDim(XParam.nblk, 1, 1);
		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			double itime = SLstepinbnd - 1.0 + (XParam.totaltime - XParam.leftbnd.data[SLstepinbnd - 1].time) / (XParam.leftbnd.data[SLstepinbnd].time - XParam.leftbnd.data[SLstepinbnd - 1].time);
			if (XParam.leftbnd.type == 2 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.leftbnd.type == 2)
			{
				leftdirichlet << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.ymax, (float)itime, rightblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}

			if (XParam.leftbnd.type == 3 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1D << <gridDim, blockDim, 0 >> > (-1, 0, (int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.leftbnd.type == 3)
			{
				ABS1D << <gridDim, blockDim, 0 >> > (-1, 0, (int)XParam.leftbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, rightblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
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
			dim3 blockDim(16, 16, 1);
			dim3 gridDim(XParam.nblk, 1, 1);
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				noslipbndLeft << <gridDim, blockDim, 0 >> > (XParam.xo, XParam.eps, rightblk_g, blockxo_gd, zb_gd, zs_gd, hh_gd, uu_gd, vv_gd);
			}
			else
			{
				noslipbndLeft << <gridDim, blockDim, 0 >> > ((float)XParam.xo, (float)XParam.eps, rightblk_g, blockxo_g, zb_g, zs_g, hh_g, uu_g, vv_g);
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



		dim3 blockDim(16, 16, 1);
		dim3 gridDim(XParam.nblk, 1, 1);
		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			double itime = SLstepinbnd - 1.0 + (XParam.totaltime - XParam.rightbnd.data[SLstepinbnd - 1].time) / (XParam.rightbnd.data[SLstepinbnd].time - XParam.rightbnd.data[SLstepinbnd - 1].time);
			if (XParam.rightbnd.type == 2 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				rightdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.rightbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xmax, XParam.ymax, itime, leftblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.rightbnd.type == 2)
			{
				rightdirichlet << <gridDim, blockDim, 0 >> > ((int)XParam.rightbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xmax, (float)XParam.ymax, (float)itime, leftblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}
			else if (XParam.rightbnd.type == 3 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1D << <gridDim, blockDim, 0 >> > (1, 0, (int)XParam.rightbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, leftblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.rightbnd.type == 3)
			{
				ABS1D << <gridDim, blockDim, 0 >> > (1, 0, (int)XParam.rightbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, leftblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
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
				noslipbndRight << <gridDim, blockDim, 0 >> > (XParam.dx, XParam.xmax, XParam.eps, leftblk_g, blockxo_gd, zb_gd, zs_gd, hh_gd, uu_gd, vv_gd);
			}
			else
			{
				noslipbndRight << <gridDim, blockDim, 0 >> > ((float)XParam.dx, (float)XParam.xmax, (float)XParam.eps, leftblk_g, blockxo_g, zb_g, zs_g, hh_g, uu_g, vv_g);
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
	int nx = XParam.nx;
	int ny = XParam.ny;
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


		dim3 blockDim(16, 16, 1);
		dim3 gridDim(XParam.nblk, 1, 1);
		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			double itime = SLstepinbnd - 1.0 + (XParam.totaltime - XParam.topbnd.data[SLstepinbnd - 1].time) / (XParam.topbnd.data[SLstepinbnd].time - XParam.topbnd.data[SLstepinbnd - 1].time);
			if (XParam.topbnd.type == 2 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				topdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.topbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xmax, XParam.ymax, itime, botblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.topbnd.type == 2)
			{
				topdirichlet << <gridDim, blockDim, 0 >> > ((int)XParam.topbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xmax, (float)XParam.ymax, (float)itime, botblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}
			else if (XParam.topbnd.type == 3 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1D << <gridDim, blockDim, 0 >> > (0, 1, (int)XParam.topbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, botblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.topbnd.type == 3)
			{
				ABS1D << <gridDim, blockDim, 0 >> > (0, 1, (int)XParam.topbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, botblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, vv_g, uu_g);
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
			dim3 blockDim(16, 16, 1);
			dim3 gridDim(XParam.nblk, 1, 1);
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				noslipbndTop << <gridDim, blockDim, 0 >> > (XParam.dx, XParam.ymax, XParam.eps, botblk_g, blockyo_gd, zb_gd, zs_gd, hh_gd, uu_gd, vv_gd);
			}
			else
			{
				noslipbndTop << <gridDim, blockDim, 0 >> > ((float)XParam.dx, (float)XParam.ymax, (float)XParam.eps, botblk_g, blockyo_g, zb_g, zs_g, hh_g, uu_g, vv_g);
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
	int nx = XParam.nx;
	int ny = XParam.ny;
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



		dim3 blockDim(16, 16, 1);
		dim3 gridDim(XParam.nblk, 1, 1);
		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			double itime = SLstepinbnd - 1.0 + (XParam.totaltime - XParam.botbnd.data[SLstepinbnd - 1].time) / (XParam.botbnd.data[SLstepinbnd].time - XParam.botbnd.data[SLstepinbnd - 1].time);
			if (XParam.botbnd.type == 2 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				botdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.botbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xmax, XParam.yo, itime, topblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.botbnd.type == 2)
			{
				botdirichlet << <gridDim, blockDim, 0 >> > ((int)XParam.botbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xmax, (float)XParam.yo, (float)itime, topblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}
			else if (XParam.botbnd.type == 3 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)XParam.leftbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1D << <gridDim, blockDim, 0 >> > (0, -1, (int)XParam.botbnd.data[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, topblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.botbnd.type == 3)
			{
				ABS1D << <gridDim, blockDim, 0 >> > (0, -1, (int)XParam.botbnd.data[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, topblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, vv_g, uu_g);
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
			dim3 blockDim(16, 16, 1);
			dim3 gridDim(XParam.nblk, 1, 1);
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				noslipbndBot << <gridDim, blockDim, 0 >> > (XParam.yo, XParam.eps, topblk_g, blockyo_gd, zb_gd, zs_gd, hh_gd, uu_gd, vv_gd);
			}
			else
			{
				noslipbndBot << <gridDim, blockDim, 0 >> > ((float)XParam.yo, (float)XParam.eps, topblk_g, blockyo_g, zb_g, zs_g, hh_g, uu_g, vv_g);
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
	updateEVATM << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.lat*pi / 21600.0f, (float)XParam.windU.xo, (float)XParam.windU.yo, (float)XParam.windU.dx, (float)XParam.Cd, rightblk_g, topblk_g, blockxo_g, blockyo_g, hh_g, uu_g, vv_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
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
	updateEVATM << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.lat*pi / 21600.0f, (float)XParam.windU.xo, (float)XParam.windU.yo, (float)XParam.windU.dx, (float)XParam.Cd, rightblk_g, topblk_g, blockxo_g, blockyo_g, hho_g, uuo_g, vvo_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);

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

		CUDA_CHECK(cudaMemcpy(dtmax_gd, arrmax_gd, s * sizeof(float), cudaMemcpyDeviceToDevice));

		reducemin3 << <gridDimLineS, blockDimLineS, smemSize >> > (dtmax_gd, arrmax_gd, s);
		CUDA_CHECK(cudaDeviceSynchronize());

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}


	CUDA_CHECK(cudaMemcpy(dummy_d, arrmax_gd, 32 * sizeof(float), cudaMemcpyDeviceToHost));
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



	//Add rain on grid
	if (!XParam.Rainongrid.inputfile.empty())
	{
		//XParam.Rainongrid.xo
		Rain_on_grid << <gridDim, blockDim, 0 >> > (XParam.Rainongrid.xo, XParam.Rainongrid.yo, XParam.Rainongrid.dx, XParam.dx, blockxo_gd, blockyo_gd, XParam.dt, zs_gd, hh_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
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


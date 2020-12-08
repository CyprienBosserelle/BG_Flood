


#include "Testing.h"




/*! \fn bool testing(Param XParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g)
* Wrapping function for all the inbuilt test
* This function is the entry point to the software
*/
template <class T> void Testing(Param XParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g)
{
	

	log("\nRunning internal test(s):");

	

	if (XParam.test == 0)
	{
		bool bumptest;
		// Test 0 is pure bump test
		log("\t Gaussian wave on Cartesian grid");
		//set gpu is -1 for cpu test
		
		bumptest = GaussianHumptest(0.1, -1, false);
		std::string result = bumptest ? "successful" : "failed";
		log("\t\tCPU test: " + result);

		// If origiinal XParam tried to use GPU we try also
		if (XParam.GPUDEVICE >= 0)
		{
			bumptest = GaussianHumptest(0.1, XParam.GPUDEVICE,false);
			std::string result = bumptest ? "successful" : "failed";
			log("\t\tGPU test: " + result);
		}
	}
	if (XParam.test == 1)
	{
		bool rivertest;
		// Test 0 is pure bump test
		log("\t River Mass conservation grid");
		rivertest = Rivertest(0.1, -1);
		std::string result = rivertest ? "successful" : "failed";
		log("\t\tCPU test: " + result);
	}
	if (XParam.test == 2)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			bool GPUvsCPUtest;
			log("\t Gaussian wave on Cartesian grid: CPU vs GPU");
			GPUvsCPUtest = GaussianHumptest(0.1, XParam.GPUDEVICE, true);
			std::string result = GPUvsCPUtest ? "successful" : "failed";
			log("\t\tCPU vs GPU test: " + result);
		}
		else
		{
			log("Specify GPU device to run test 2 (CPU vs GPU comparison)");
		}
	}
	if (XParam.test == 3)
	{

		bool testresults;
		bool testreduction = true;

		// Iterate this test niter times:
		int niter = 1000;
		srand(time(0));
		log("\t Reduction Test");
		for (int iter = 0; iter < niter; iter++)
		{
			testresults = reductiontest(XParam, XModel, XModel_g);
			testreduction = testreduction && testresults;
		}

		std::string result = testreduction ? "successful" : "failed";
		log("\t\tReduction test: " + result);

	}

	if (XParam.test == 4)
	{
		//
		bool testresults;
		testresults = CPUGPUtest(XParam, XModel, XModel_g);
		exit(1);
	}
	if (XParam.test == 5)
	{
		log("\t Lake-at-rest Test");
		LakeAtRest(XParam, XModel);
	}

	

}
template void Testing<float>(Param XParam, Forcing<float> XForcing, Model<float> XModel, Model<float> XModel_g);
template void Testing<double>(Param XParam, Forcing<float> XForcing, Model<double> XModel, Model<double> XModel_g);

/*! \fn bool GaussianHumptest(T zsnit, int gpu, bool compare)
*
* This function tests the full hydrodynamics model and compares the results with pre-conmputed (Hard wired) values
*	The function creates it own model setup and mesh independantly to what the user might want to do
*	The setup consist of and centrally located gaussian hump radiating away
*	The test stops at an arbitrary time to compare with 8 values extracted from a identical run in basilisk
*	This function also compares the result of the GPU and CPU code (until they diverge) 
*/
template <class T> bool GaussianHumptest(T zsnit, int gpu, bool compare)
{
	log("#####");
	// this is a preplica of the tutorial case for Basilisk
	Param XParam;

	T x, y, delta;
	T cc = T(0.05);// Match the 200 in chracteristic radius used in Basilisk  1/(2*cc^2)=200


	T a = T(1.0); //Gaussian wave amplitude

	// Verification data
	// This is a transect across iy=15:16:127 at ix=127 (or vice versa because the solution is symetrical)
	// These values are based on single precision output from Netcdf file so are only accurate to 10-7 
	double ZsVerification[8] = { 0.100000000023, 0.100000063119, 0.100110376004, 0.195039970749, 0.136739044168, 0.0848024805994, 0.066275833049, 0.0637058445888 };
	//double ZsVerification[8] = { 0.100000008904, 0.187920326216, 0.152329657390, 0.117710230042, 0.0828616638138, 0.0483274739972, 0.0321501737555, 0.0307609731288 };

		
	


	// initialise domain and required resolution
	XParam.dx = 1.0 / ((1 << 8));
	XParam.xo = -0.5;
	XParam.yo = -0.5;

	XParam.xmax = 0.5;
	XParam.ymax = 0.5;
	//level 8 is 
	

	XParam.initlevel = 0;
	XParam.minlevel = 0;
	XParam.maxlevel = 0;

	XParam.zsinit = zsnit;
	XParam.zsoffset = 0.0;

	//Output times for comparisons
	XParam.endtime = 1.0;
	XParam.outputtimestep = 0.1;

	XParam.smallnc = 0;

	XParam.cf = 0.0;
	XParam.frictionmodel = 0;

	// Enforece GPU/CPU
	XParam.GPUDEVICE = gpu;

	std::string outvi[18] = { "zb","h","zs","u","v","Fqux","Fqvx","Fquy","Fqvy", "Fhu", "Fhv", "dh", "dhu", "dhv", "ho", "vo", "uo", "cf" };

	std::vector<std::string> outv;

	for (int nv = 0; nv < 18; nv++)
	{
		outv.push_back(outvi[nv]);
	}

	XParam.outvars = outv;

	// create Model setup
	Model<T> XModel;
	Model<T> XModel_g;

	Forcing<float> XForcing;
	StaticForcingP<float> bathy;

	XForcing.Bathy.push_back(bathy);
	// initialise forcing bathymetry to 0
	XForcing.Bathy[0].xo = -1.0;
	XForcing.Bathy[0].yo = -1.0;

	XForcing.Bathy[0].xmax = 1.0;
	XForcing.Bathy[0].ymax = 1.0;
	XForcing.Bathy[0].nx = 3;
	XForcing.Bathy[0].ny = 3;

	XForcing.Bathy[0].dx = 1.0;

	AllocateCPU(1, 1, XForcing.left.blks, XForcing.right.blks, XForcing.top.blks, XForcing.bot.blks);

	AllocateCPU(XForcing.Bathy[0].nx, XForcing.Bathy[0].ny, XForcing.Bathy[0].val);

	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = 0.0f;
		}
	}

	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	// Recreate the initia;l conditions
	//InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.zb);
	//InitArrayBUQ(XParam, XModel.blocks, zsnit, XModel.evolv.zs);
	//zs is initialised here:
	InitialConditions(XParam, XForcing, XModel);

	T xorigin = T(0.0);
	T yorigin = T(0.0);


	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XModel.blocks.active[ibl];
		delta = calcres(XParam.dx, XModel.blocks.level[ib]);


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				//
				int n = memloc(XParam, ix, iy, ib);
				x = XParam.xo + XModel.blocks.xo[ib] + ix * delta;
				y = XParam.yo + XModel.blocks.yo[ib] + iy * delta;
				XModel.evolv.zs[n] = XModel.evolv.zs[n] + a * exp(T(-1.0) * ((x - xorigin) * (x - xorigin) + (y - yorigin) * (y - yorigin)) / (2.0 * cc * cc));
				XModel.evolv.h[n] = utils::max(XModel.evolv.zs[n] - XModel.zb[n],T(0.0));
				
			}
		}
	}

	SetupGPU(XParam, XModel, XForcing, XModel_g);

	Loop<T> XLoop;
	Loop<T> XLoop_g;


	XLoop.hugenegval = std::numeric_limits<T>::min();
	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();
	XLoop.totaltime = 0.0;

	//InitSave2Netcdf(XParam, XModel);
	XLoop.nextoutputtime = XParam.outputtimestep;
	XLoop.dtmax = initdt(XParam, XLoop, XModel);

	//XLoop_g = XLoop;
	XLoop_g.hugenegval = std::numeric_limits<T>::min();
	XLoop_g.hugeposval = std::numeric_limits<T>::max();
	XLoop_g.epsilon = std::numeric_limits<T>::epsilon();
	XLoop_g.totaltime = 0.0;

	//InitSave2Netcdf(XParam, XModel);
	XLoop_g.nextoutputtime = XParam.outputtimestep;
	XLoop_g.dtmax = XLoop.dtmax;


	if (XParam.GPUDEVICE >= 0 && compare)
	{
		CompareCPUvsGPU(XParam, XModel, XModel_g, outv, false);
	}
	bool modelgood = true;

	while (XLoop.totaltime < XLoop.nextoutputtime)
	{

		if (XParam.GPUDEVICE >= 0)
		{
			FlowGPU(XParam, XLoop_g, XForcing, XModel_g);
			XLoop.dt = XLoop_g.dt;
		}
		else
		{
			FlowCPU(XParam, XLoop, XForcing, XModel);
		}
		if (XParam.GPUDEVICE >= 0 && compare)
		{
			FlowCPU(XParam, XLoop, XForcing, XModel);

			T diffdt = XLoop_g.dt - XLoop.dt;
			if (abs(diffdt) > T(100.0) * (XLoop.epsilon))
			{
				printf("Timestep Difference=%f\n", diffdt);

				compare = false;
			}
			CompareCPUvsGPU(XParam, XModel, XModel_g, outv,false);
		}

		//diffdh(XParam, XModel.blocks, XModel.flux.Su, diff, shuffle);
		//diffSource(XParam, XModel.blocks, XModel.flux.Fqux, XModel.flux.Su, diff);
		XLoop.totaltime = XLoop.totaltime + XLoop.dt;
		XLoop_g.totaltime = XLoop_g.totaltime + XLoop_g.dt;
		if (XLoop.nextoutputtime - XLoop.totaltime <= XLoop.dt * T(0.00001) && XParam.outputtimestep > 0.0)
		{
			if (XParam.GPUDEVICE >= 0)
			{
				for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
				{
					CUDA_CHECK(cudaMemcpy(XModel.OutputVarMap[XParam.outvars[ivar]], XModel_g.OutputVarMap[XParam.outvars[ivar]], XParam.nblkmem * XParam.blksize * sizeof(T), cudaMemcpyDeviceToHost));
				}
			}

			//Save2Netcdf(XParam, XModel);
			// Verify the Validity of results
			

			double diff;
			for (int iv = 0; iv < 8; iv++)
			{
				
				int ix, iy, ib, ii, jj,ibx,iby,nbx,nby;
				jj = 127;
				ii = (iv + 1) * 16 - 1;

				// Theoretical size is 255x255
				nbx = 256 / 16;
				nby = 256 / 16;

				ibx = floor(ii / 16);
				iby = floor(jj / 16);

				ib = (iby) * nbx + ibx;

				ix = ii - ibx*16;
				iy = jj - iby*16;

				int n = memloc(XParam, ix, iy, ib);

				diff = abs(T(XModel.evolv.zs[n]) - ZsVerification[iv]);



				if (diff > 1e-6)//Tolerance is 1e-6 or 1e-7/1e-8??
				{

					printf("ib=%d, ix=%d, iy=%d; simulated=%f; expected=%f; diff=%e\n", ib, ix, iy, XModel.evolv.zs[n], ZsVerification[iv], diff);
					modelgood = false;
				}



			}



			//XLoop.nextoutputtime = min(XLoop.nextoutputtime + XParam.outputtimestep, XParam.endtime);

		}
	}
	log("#####");
	return modelgood;
}
template bool GaussianHumptest<float>(float zsnit, int gpu, bool compare);
template bool GaussianHumptest<double>(double zsnit, int gpu, bool compare);

/*! \fn bool Rivertest(T zsnit, int gpu)
*
* This function tests the mass conservation of the vertical injection (used for rivers)
*	The function creates it own model setup and mesh independantly to what the user might want to do
*	This starts with a initial water level (zsnit=0 is dry) and runs for 0.1s before comparing results 
*	with zsnit=0.1 that is approx 20 steps
*/
template <class T> bool Rivertest(T zsnit, int gpu)
{
	log("#####");
	Param XParam;
	T delta, initVol, finalVol, TheoryInput;
	// initialise domain and required resolution
	XParam.dx = 1.0 / ((1 << 8));
	XParam.xo = -0.5;
	XParam.yo = -0.5;

	XParam.xmax = 0.5;
	XParam.ymax = 0.5;
	//level 8 is 


	XParam.initlevel = 0;
	XParam.minlevel = 0;
	XParam.maxlevel = 0;

	XParam.zsinit = zsnit;
	XParam.zsoffset = 0.0;

	//Output times for comparisons
	XParam.endtime = 1.0;
	XParam.outputtimestep = 0.1;

	XParam.smallnc = 0;

	XParam.cf = 0.0;
	XParam.frictionmodel = 0;

	// Enforece GPU/CPU
	XParam.GPUDEVICE = gpu;

	std::string outvi[16] = { "zb","h","zs","u","v","Fqux","Fqvx","Fquy","Fqvy", "Fhu", "Fhv", "dh", "dhu", "dhv", "Su", "Sv" };

	std::vector<std::string> outv;

	for (int nv = 0; nv < 15; nv++)
	{
		outv.push_back(outvi[nv]);
	}

	XParam.outvars = outv;
	// create Model setup
	Model<T> XModel;
	Model<T> XModel_g;

	Forcing<float> XForcing;

	StaticForcingP<float> bathy;

	XForcing.Bathy.push_back(bathy);

	// initialise forcing bathymetry to 0
	XForcing.Bathy[0].xo = -1.0;
	XForcing.Bathy[0].yo = -1.0;

	XForcing.Bathy[0].xmax = 1.0;
	XForcing.Bathy[0].ymax = 1.0;
	XForcing.Bathy[0].nx = 3;
	XForcing.Bathy[0].ny = 3;

	XForcing.Bathy[0].dx = 1.0;

	AllocateCPU(1, 1, XForcing.left.blks, XForcing.right.blks, XForcing.top.blks, XForcing.bot.blks);

	AllocateCPU(XForcing.Bathy[0].nx, XForcing.Bathy[0].ny, XForcing.Bathy[0].val);

	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = 0.0f;
		}
	}
	//
	//
	// 
	T Q = 0.001;
	TheoryInput = Q * XParam.outputtimestep;
	

	//Create a temporary file with river fluxes
	std::ofstream river_file(
		"testriver.tmp", std::ios_base::out | std::ios_base::trunc);
	river_file << "0.0 " + std::to_string(Q) << std::endl;
	river_file << "3600.0 " + std::to_string(Q) << std::endl;
	river_file.close(); //destructor implicitly does it

	River thisriver;
	thisriver.Riverflowfile = "testriver.tmp";
	thisriver.xstart = -1.0*XParam.dx*3.0;
	thisriver.xend = XParam.dx * 3.0;
	thisriver.ystart = -1.0 * XParam.dx * 3.0;
	thisriver.yend = XParam.dx * 3.0;

	XForcing.rivers.push_back(thisriver);


	XForcing.rivers[0].flowinput = readFlowfile(XForcing.rivers[0].Riverflowfile);


	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	InitialConditions(XParam, XForcing, XModel);

	SetupGPU(XParam, XModel, XForcing, XModel_g);

	Loop<T> XLoop;

	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();

	XLoop.totaltime = 0.0;
	
	//InitSave2Netcdf(XParam, XModel);
	XLoop.nextoutputtime = XParam.outputtimestep;
	XLoop.dtmax = initdt(XParam, XLoop, XModel);
	initVol = T(0.0);
	// Calculate initial water volume
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XModel.blocks.active[ibl];
		delta = calcres(XParam.dx, XModel.blocks.level[ib]);


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				//
				int n = memloc(XParam, ix, iy, ib);
				initVol = initVol+XModel.evolv.h[n] * delta * delta;
			}
		}
	}


	//InitSave2Netcdf(XParam, XModel);
	bool modelgood = true;

	while (XLoop.totaltime < XLoop.nextoutputtime)
	{

		if (XParam.GPUDEVICE >= 0)
		{
			FlowGPU(XParam, XLoop, XForcing, XModel_g);
		}
		else
		{
			FlowCPU(XParam, XLoop, XForcing, XModel);
		}
		XLoop.totaltime = XLoop.totaltime + XLoop.dt;
		//Save2Netcdf(XParam, XModel);

		if (XLoop.nextoutputtime - XLoop.totaltime <= XLoop.dt * T(0.00001) && XParam.outputtimestep > 0.0)
		{
			if (XParam.GPUDEVICE >= 0)
			{
				for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
				{
					CUDA_CHECK(cudaMemcpy(XModel.OutputVarMap[XParam.outvars[ivar]], XModel_g.OutputVarMap[XParam.outvars[ivar]], XParam.nblkmem * XParam.blksize * sizeof(T), cudaMemcpyDeviceToHost));
				}
			}

			//Save2Netcdf(XParam, XModel);
			// Verify the Validity of results
			finalVol = T(0.0);
			for (int ibl = 0; ibl < XParam.nblk; ibl++)
			{
				//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
				int ib = XModel.blocks.active[ibl];
				delta = calcres(XParam.dx, XModel.blocks.level[ib]);


				for (int iy = 0; iy < XParam.blkwidth; iy++)
				{
					for (int ix = 0; ix < XParam.blkwidth; ix++)
					{
						//
						int n = memloc(XParam, ix, iy, ib);
						finalVol = finalVol + XModel.evolv.h[n] * delta * delta;
					}
				}
			}
			T error = (finalVol - initVol) - TheoryInput;

			modelgood = error / TheoryInput < 0.05;
		}

		
	}
	log("#####");
	return modelgood;
}
template bool Rivertest<float>(float zsnit, int gpu);
template bool Rivertest<double>(double zsnit, int gpu);



/*! \fn reductiontest()
*
*	Test the algorithm for reducing the global time step on the user grid layout
*/
template <class T> bool reductiontest(Param XParam, Model<T> XModel,Model<T> XModel_g)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	//srand(seed);
	T mininput = T(rand())/T(RAND_MAX);
	bool test = true;

	Loop<T> XLoop;

	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();

	XLoop.totaltime = 0.0;

	//InitSave2Netcdf(XParam, XModel);
	XLoop.nextoutputtime = mininput*T(2.0);
	XLoop.dtmax = mininput * T(10.0);

	// Fill in dtmax with random values that are larger than  mininput
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
			
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
					//
				int n = memloc(XParam, ix, iy, ib);
				XModel.time.dtmax[n] = mininput * T(1.1) + utils::max(T(rand()) / T(RAND_MAX),T(0.0));
			}
		}
	}

	// randomly select a block a i and a j were the maximum value will be relocated
	int ibbl = floor(T(rand()) / T(RAND_MAX) * XParam.nblk);
	int ibb = XModel.blocks.active[ibbl];
	int ixx = floor(T(rand()) / T(RAND_MAX) * XParam.blkwidth);
	int iyy = floor(T(rand()) / T(RAND_MAX) * XParam.blkwidth);

	int nn= memloc(XParam, ixx, iyy, ibb);

	XModel.time.dtmax[nn] = mininput;

	T reducedt = CalctimestepCPU(XParam, XLoop, XModel.blocks, XModel.time);

	test = abs(reducedt - mininput) < T(100.0) * (XLoop.epsilon);
	bool testgpu;

	if (!test)
	{
		char buffer[256]; sprintf(buffer, "%e", abs(reducedt - mininput));
		std::string str(buffer);
		//log("\t\t CPU testfailed! : Expected=" + std::to_string(mininput) + ";  Reduced=" + std::to_string(reducedt)+ ";  error=" +str);
	}

	if (XParam.GPUDEVICE >= 0)
	{

		reset_var <<< gridDim, blockDim, 0>>> (XParam.halowidth, XModel_g.blocks.active, XLoop.hugeposval, XModel_g.time.dtmax);
		CUDA_CHECK(cudaDeviceSynchronize());

		CopytoGPU(XParam.nblkmem, XParam.blksize, XModel.time.dtmax, XModel_g.time.dtmax);
		T reducedtgpu=CalctimestepGPU(XParam, XLoop, XModel_g.blocks, XModel_g.time);
		testgpu = abs(reducedtgpu - mininput) < T(100.0) * (XLoop.epsilon);

		if (!testgpu)
		{
			char buffer[256]; sprintf(buffer, "%e", abs(reducedtgpu - mininput));
			std::string str(buffer);
			//log("\t\t GPU test failed! : Expected=" + std::to_string(mininput) + ";  Reduced=" + std::to_string(reducedtgpu) + ";  error=" + str);
		}

		if (abs(reducedtgpu - reducedt) > T(100.0) * (XLoop.epsilon))
		{
			char buffer[256]; sprintf(buffer, "%e", abs(reducedtgpu - reducedt));
			std::string str(buffer);
			log("\t\t CPU vs GPU test failed! : Expected=" + std::to_string(reducedt) + ";  Reduced=" + std::to_string(reducedtgpu) + ";  error=" + str);
		}

		test = test && testgpu;
	}
	

	return test;
}
template bool reductiontest<float>(Param XParam, Model<float> XModel, Model<float> XModel_g);
template bool reductiontest<double>(Param XParam, Model<double> XModel, Model<double> XModel_g);

/*! \fn CPUGPUtest(Param XParam, Model<float> XModel, Model<float> XModel_g)
*	Perform a series of test between the CPU and GPU Flow functions
*	This test only occurs if a valid GPU is specified by user
*/
template<class T> bool CPUGPUtest(Param XParam, Model<T> XModel, Model<T> XModel_g)
{
	bool test = true;

	T initdepth = T(0.1);
	T testamp = T(1.0);

	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	// for flux reconstruction the loop overlap the right(or top for the y direction) halo
	dim3 blockDimKX(XParam.blkwidth + XParam.halowidth, XParam.blkwidth, 1);
	dim3 blockDimKY(XParam.blkwidth, XParam.blkwidth + XParam.halowidth, 1);

	InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.zb);
	InitArrayBUQ(XParam, XModel.blocks, T(initdepth), XModel.evolv.zs);
	InitArrayBUQ(XParam, XModel.blocks, T(initdepth), XModel.evolv.h);
	InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.evolv.u);
	InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.evolv.v);


	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel_g.blocks.active, T(0.0), XModel_g.zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	// Create some usefull vectors
	std::string evolvst[4] = {"h","zs","u","v"};

	std::vector<std::string> evolvVar;

	for (int nv = 0; nv < 4; nv++)
	{
		evolvVar.push_back(evolvst[nv]);
	}

		
	// Check fillhalo function

	// fill with all evolv array with random value
	/*
	fillrandom(XParam, XModel.blocks, XModel.evolv.zs);
	fillrandom(XParam, XModel.blocks, XModel.evolv.h);
	fillrandom(XParam, XModel.blocks, XModel.evolv.u);
	fillrandom(XParam, XModel.blocks, XModel.evolv.v);
	*/
	fillgauss(XParam, XModel.blocks, testamp, XModel.evolv.zs);
	fillgauss(XParam, XModel.blocks, testamp, XModel.evolv.h);
	fillgauss(XParam, XModel.blocks, T(0.5* testamp), XModel.evolv.u);
	fillgauss(XParam, XModel.blocks, T(0.5* testamp), XModel.evolv.v);

	//copy to GPU
	CopytoGPU(XParam.nblkmem, XParam.blksize, XModel.evolv, XModel_g.evolv);

	//============================================
	//  Fill the halo for gradient reconstruction
	fillHalo(XParam, XModel.blocks, XModel.evolv, XModel.zb);
	fillHaloGPU(XParam, XModel_g.blocks, XModel_g.evolv, XModel.zb);

	CompareCPUvsGPU(XParam, XModel, XModel_g, evolvVar, true);

	//============================================
	//perform gradient reconstruction
	gradientCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.zb);
	gradientGPU(XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.grad, XModel.zb);

	std::string gradst[8] = { "dhdx","dzsdx","dudx","dvdx","dhdy","dzsdy","dudy","dvdy" };

	std::vector<std::string> gradVar;

	for (int nv = 0; nv < 8; nv++)
	{
		gradVar.push_back(gradst[nv]);
	}

	CompareCPUvsGPU(XParam, XModel, XModel_g, gradVar, false);

	//============================================
	// Kurganov scheme

	std::string fluxst[8] = { "Fhu","Su","Fqux","Fqvx","Fhv","Sv","Fqvy","Fquy" };

	std::vector<std::string> fluxVar;

	for (int nv = 0; nv < 8; nv++)
	{
		fluxVar.push_back(fluxst[nv]);
	}

	updateKurgXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
		
	//GPU part
	updateKurgXGPU <<< gridDim, blockDimKX, 0 >>> (XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.grad, XModel_g.flux, XModel_g.time.dtmax, XModel_g.zb);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	
	// Y- direction
	updateKurgYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);

	updateKurgYGPU <<< gridDim, blockDimKY, 0 >>> (XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.grad, XModel_g.flux, XModel_g.time.dtmax, XModel_g.zb);
	CUDA_CHECK(cudaDeviceSynchronize());

	CompareCPUvsGPU(XParam, XModel, XModel_g, fluxVar, false);


	fillHalo(XParam, XModel.blocks, XModel.flux);
	fillHaloGPU(XParam, XModel_g.blocks, XModel_g.flux);

	CompareCPUvsGPU(XParam, XModel, XModel_g, fluxVar, true);


	//============================================
	// Update step
	std::string advst[3] = { "dh","dhu","dhv"};

	std::vector<std::string> advVar;

	for (int nv = 0; nv < 3; nv++)
	{
		advVar.push_back(advst[nv]);
	}
	updateEVCPU(XParam, XModel.blocks, XModel.evolv, XModel.flux, XModel.adv);
	updateEVGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.flux, XModel_g.adv);
	CUDA_CHECK(cudaDeviceSynchronize());

	CompareCPUvsGPU(XParam, XModel, XModel_g, advVar, false);

	//============================================
	// Advance step
	std::string evost[4] = { "zso","ho","uo","vo" };

	std::vector<std::string> evoVar;

	for (int nv = 0; nv < 4; nv++)
	{
		evoVar.push_back(evost[nv]);
	}
	AdvkernelCPU(XParam, XModel.blocks, T(0.0005), XModel.zb, XModel.evolv, XModel.adv, XModel.evolv_o);
	AdvkernelGPU << < gridDim, blockDim, 0 >>> (XParam, XModel_g.blocks, T(0.0005), XModel_g.zb, XModel_g.evolv, XModel_g.adv, XModel_g.evolv_o);
	CUDA_CHECK(cudaDeviceSynchronize());

	CompareCPUvsGPU(XParam, XModel, XModel_g, evoVar, false);
	
	//============================================
	// Bottom friction

	bottomfrictionCPU(XParam, XModel.blocks, T(0.5), XModel.cf, XModel.evolv_o);

	bottomfrictionGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel_g.blocks, T(0.5), XModel_g.cf, XModel_g.evolv_o);
	CUDA_CHECK(cudaDeviceSynchronize());

	CompareCPUvsGPU(XParam, XModel, XModel_g, evoVar, false);


	//============================================
	// Repeat the full test
	Loop<T> XLoop;
	Loop<T> XLoop_g;

	XParam.endtime = utils::min(0.5 * (XParam.ymax - XParam.yo), 0.5 * (XParam.xmax - XParam.xo)) / (sqrt(XParam.g * (testamp + initdepth)));
	XParam.outputtimestep = XParam.endtime / 10.0;


	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();

	XLoop.totaltime = 0.0;

	//InitSave2Netcdf(XParam, XModel);
	XLoop.nextoutputtime = XParam.outputtimestep;
	XLoop.dtmax = initdt(XParam, XLoop, XModel);

	XLoop_g.hugenegval = std::numeric_limits<T>::min();

	XLoop_g.hugeposval = std::numeric_limits<T>::max();
	XLoop_g.epsilon = std::numeric_limits<T>::epsilon();

	XLoop_g.totaltime = 0.0;

	//InitSave2Netcdf(XParam, XModel);
	XLoop_g.nextoutputtime = XLoop.nextoutputtime;
	XLoop_g.dtmax = XLoop.dtmax;

	std::string outvi[18] = { "zb","h","zs","u","v","Fqux","Fqvx","Fquy","Fqvy", "Fhu", "Fhv", "dh", "dhu", "dhv", "ho", "vo", "uo", "cf" };

	std::vector<std::string> outv;

	for (int nv = 0; nv < 18; nv++)
	{
		outv.push_back(outvi[nv]);
	}

	
	InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.evolv.u);
	InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.evolv.v);
	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel_g.blocks.active, T(0.0), XModel_g.evolv.u);
	CUDA_CHECK(cudaDeviceSynchronize());

	reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel_g.blocks.active, T(0.0), XModel_g.evolv.v);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	Forcing<float> XForcing;
	while (XLoop.totaltime < XParam.endtime)
	{
		FlowGPU(XParam, XLoop_g, XForcing, XModel_g);
		FlowCPU(XParam, XLoop, XForcing, XModel);

		XLoop.totaltime = XLoop.totaltime + XLoop.dt;
		XLoop_g.totaltime = XLoop_g.totaltime + XLoop_g.dt;
		if (XLoop.nextoutputtime - XLoop.totaltime <= XLoop.dt * T(0.00001) && XParam.outputtimestep > 0.0)
		{
			CompareCPUvsGPU(XParam, XModel, XModel_g, outv, false);
			XLoop.nextoutputtime = min(XLoop.nextoutputtime + XParam.outputtimestep, XParam.endtime);
			XLoop_g.nextoutputtime = XLoop.nextoutputtime;
		}
	}
	

	return test;
}



/*! \fn 
*	Simulate the first predictive step and check wether the lake at rest is preserved
*	
*/
template <class T> bool LakeAtRest(Param XParam, Model<T> XModel)
{
	T epsi = 0.000001;
	int ib;

	bool test = true;


	Loop<T> XLoop = InitLoop(XParam, XModel);

	//============================================
	// Predictor step in reimann solver
	//============================================

	//============================================
	//  Fill the halo for gradient reconstruction
	fillHalo(XParam, XModel.blocks, XModel.evolv, XModel.zb);

	//============================================
	// Reset DTmax
	InitArrayBUQ(XParam, XModel.blocks, XLoop.hugeposval, XModel.time.dtmax);

	//============================================
	// Calculate gradient for evolving parameters
	gradientCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.zb);

	//============================================
	// Flux and Source term reconstruction
	// X- direction
	updateKurgXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);

	// Y- direction
	updateKurgYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);
	
	//============================================
	// Fill Halo for flux from fine to coarse
	//fillHalo(XParam, XModel.blocks, XModel.flux);


	// Check Fhu and Fhv (they should be zero)
	int i, ibot, itop, iright, ileft;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XModel.blocks.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < (XParam.blkwidth); ix++)
			{
				i = memloc(XParam, ix, iy, ib);
				iright = memloc(XParam, ix+1, iy, ib);
				ileft = memloc(XParam, ix-1, iy, ib);
				itop = memloc(XParam, ix, iy+1, ib);
				ibot = memloc(XParam, ix, iy-1, ib);

				if (abs(XModel.flux.Fhu[i]) > epsi)
				{
					log("Fhu is not zero. Lake at rest not preserved!!!");
					test = false;
				}
				
				if (abs(XModel.flux.Fhv[i]) > epsi)
				{
					log("Fhv is not zero. Lake at rest not preserved!!!");
					test = false;
				}

				T dhus = (XModel.flux.Fqux[i] - XModel.flux.Su[iright]);
				if (abs(dhus) > epsi)
				{
					test = false;

					log("dhu is not zero. Lake at rest not preserved!!!");

					printf("Fqux[i]=%f; Su[iright]=%f; Diff=%f \n",XModel.flux.Fqux[i], XModel.flux.Su[iright], (XModel.flux.Fqux[i] - XModel.flux.Su[iright]));

					printf(" At i: (ib=%d; ix=%d; iy=%d)\n", ib,ix,iy);
					testkurganovX(XParam, ib, ix, iy, XModel);

					printf(" At iright: (ib=%d; ix=%d; iy=%d)\n", ib, ix+1, iy);
					testkurganovX(XParam, ib, ix+1, iy, XModel);
				}

			}
		}
	}


	if (!test)
	{
		copyID2var(XParam, XModel.blocks, XModel.flux.Fhu);
		copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.LeftBot, XModel.grad.dhdx);
		copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.LeftTop, XModel.grad.dhdy);
		copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.TopLeft, XModel.grad.dzsdx);
		copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.TopRight, XModel.grad.dzsdy);
		copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.RightTop, XModel.grad.dudx);
		copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.RightBot, XModel.grad.dudy);
		copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.BotRight, XModel.grad.dvdx);
		copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.BotLeft, XModel.grad.dvdy);

		creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo);
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "blockID", 3, XModel.flux.Fhu);

		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "LeftBot", 3, XModel.grad.dhdx);
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "LeftTop", 3, XModel.grad.dhdy);

		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "TopLeft", 3, XModel.grad.dzsdx);
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "TopRight", 3, XModel.grad.dzsdy);

		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "RightTop", 3, XModel.grad.dudx);
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "RightBot", 3, XModel.grad.dudy);

		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "BotLeft", 3, XModel.grad.dvdx);
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "BotRight", 3, XModel.grad.dvdy);
	}

	
}


template <class T> void testkurganovX(Param XParam, int ib, int ix, int iy, Model<T> XModel)
{
	int RB, levRB, LBRB, LB, levLB, RBLB;
	int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
	int ileft = memloc(XParam.halowidth, XParam.blkmemwidth, ix - 1, iy, ib);

	int lev = XModel.blocks.level[ib];
	T delta = calcres(T(XParam.dx), lev);

	T g = T(XParam.g);
	T CFL = T(XParam.CFL);
	T epsi = nextafter(T(1.0), T(2.0)) - T(1.0);
	T eps = T(XParam.eps) + epsi;

	// neighbours for source term

	RB = XModel.blocks.RightBot[ib];
	levRB = XModel.blocks.level[RB];
	LBRB = XModel.blocks.LeftBot[RB];

	LB = XModel.blocks.LeftBot[ib];
	levLB = XModel.blocks.level[LB];
	RBLB = XModel.blocks.RightBot[LB];

	T dhdxi = XModel.grad.dhdx[i];
	T dhdxmin = XModel.grad.dhdx[ileft];
	T cm = T(1.0);
	T fmu = T(1.0);

	T hi = XModel.evolv.h[i];

	T hn = XModel.evolv.h[ileft];
	T dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, ga;

	// along X
	dx = delta * T(0.5);
	zi = XModel.evolv.zs[i] - hi;

	//printf("%f\n", zi);


	//zl = zi - dx*(dzsdx[i] - dhdx[i]);
	zl = zi - dx * (XModel.grad.dzsdx[i] - dhdxi);
	//printf("%f\n", zl);

	zn = XModel.evolv.zs[ileft] - hn;

	//printf("%f\n", zn);
	zr = zn + dx * (XModel.grad.dzsdx[ileft] - dhdxmin);


	zlr = max(zl, zr);

	//hl = hi - dx*dhdx[i];
	hl = hi - dx * dhdxi;
	up = XModel.evolv.u[i] - dx * XModel.grad.dudx[i];
	hp = max(T(0.0), hl + zl - zlr);

	hr = hn + dx * dhdxmin;
	um = XModel.evolv.u[ileft] + dx * XModel.grad.dudx[ileft];
	hm = max(T(0.0), hr + zr - zlr);

	ga = g * T(0.5);
	///// Reimann solver
	T fh, fu, fv, sl, sr, dt;

	//solver below also modifies fh and fu
	dt = KurgSolver(g, delta, epsi, CFL, cm, fmu, hp, hm, up, um, fh, fu);

	if ((ix == XParam.blkwidth) && levRB < lev)//(ix==16) i.e. in the right halo
	{
		int jj = LBRB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + XParam.blkwidth / 2;
		int iright = memloc(XParam.halowidth, XParam.blkmemwidth, 0, jj, RB);;
		hi = XModel.evolv.h[iright];
		zi = XModel.zb[iright];
	}
	if ((ix == 0) && levLB < lev)//(ix==16) i.e. in the right halo if you 
	{
		int jj = RBLB == ib ? floor(iy * (T)0.5) : floor(iy * (T)0.5) + XParam.blkwidth / 2;
		int ilc = memloc(XParam.halowidth, XParam.blkmemwidth, XParam.blkwidth - 1, jj, LB);
		hn = XModel.evolv.h[ilc];
		zn = XModel.zb[ilc];
	}

	
	sl = ga * (utils::sq(hp) - utils::sq(hl) + (hl + hi) * (zi - zl));
	sr = ga * (utils::sq(hm) - utils::sq(hr) + (hr + hn) * (zn - zr));

	////Flux update
	//Fhu[i] = fmu * fh;
	//Fqux[i] = fmu * (fu - sl);
	//Su[i] = fmu * (fu - sr);
	//Fqvx[i] = fmu * fv;

	printf("hi=%f; hn=%f,fh=%f; fu=%f; sl=%f; sr=%f; hp=%f; hm=%f; hr=%f; hl=%f; zr=%f; zl=%f;\n", XModel.evolv.h[i], XModel.evolv.h[ileft], fh, fu, sl, sr, hp, hm, hr, hl, zr, zl);



}


template <class T> void fillrandom(Param XParam, BlockP<T> XBlock, T * z)
{
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				//
				int n = memloc(XParam, ix, iy, ib);
				z[n] =  T(rand()) / T(RAND_MAX);
			}
		}
	}
}
template void fillrandom<float>(Param XParam, BlockP<float> XBlock, float* z);
template void fillrandom<double>(Param XParam, BlockP<double> XBlock, double* z);

template <class T> void fillgauss(Param XParam, BlockP<T> XBlock,T amp, T* z)
{
	T delta, x, y;
	T cc = T(0.05) * (XParam.xmax - XParam.xo);
	T xorigin = XParam.xo + T(0.5) * (XParam.xmax - XParam.xo);
	T yorigin = XParam.yo + T(0.5) * (XParam.ymax - XParam.yo);


	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XBlock.active[ibl];
		delta = calcres(XParam.dx, XBlock.level[ib]);


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				//
				int n = memloc(XParam, ix, iy, ib);
				x = XParam.xo + XBlock.xo[ib] + ix * delta;
				y = XParam.yo + XBlock.yo[ib] + iy * delta;
				z[n] = z[n] + amp * exp(T(-1.0) * ((x - xorigin) * (x - xorigin) + (y - yorigin) * (y - yorigin)) / (2.0 * cc * cc));
				

			}
		}
	}
}
template void fillgauss<float>(Param XParam, BlockP<float> XBlock, float amp, float* z);
template void fillgauss<double>(Param XParam, BlockP<double> XBlock, double amp, double* z);


/*! \fn TestingOutput(Param XParam, Model<T> XModel)
*  
*	OUTDATED?
*/
template <class T>
void TestingOutput(Param XParam, Model<T> XModel)
{
	std::string outvar;

	Loop<T> XLoop;
	// GPU stuff
	

	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();

	XLoop.totaltime = 0.0;

	XLoop.nextoutputtime = 0.2;

	Forcing<float> XForcing;

	//FlowCPU(XParam, XLoop, XModel);

	//log(std::to_string(XForcing.Bathy.val[50]));
	creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo);
	outvar = "h";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	outvar = "u";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	outvar = "v";
	//copyID2var(XParam, XModel.blocks, XModel.OutputVarMap[outvar]);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	outvar = "zb";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	outvar = "zs";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);


	FlowCPU(XParam, XLoop, XForcing, XModel);


	//outvar = "cf";
	//defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.cf);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhdx", 3, XModel.grad.dhdx);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhdy", 3, XModel.grad.dhdy);

	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fhv", 3, XModel.flux.Fhv);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fhu", 3, XModel.flux.Fhu);
	

	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqux", 3, XModel.flux.Fqux);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fquy", 3, XModel.flux.Fquy);

	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqvx", 3, XModel.flux.Fqvx);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqvy", 3, XModel.flux.Fqvy);


	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Su", 3, XModel.flux.Su);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Sv", 3, XModel.flux.Sv);


	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dh", 3, XModel.adv.dh);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhu", 3, XModel.adv.dhu);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhv", 3, XModel.adv.dhv);

	writenctimestep(XParam.outfile, XLoop.totaltime + XLoop.dt);
	

	outvar = "h";
	writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, XModel.OutputVarMap[outvar]);
	
	outvar = "zs";
	writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, XModel.OutputVarMap[outvar]);
	outvar = "u";
	writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, XModel.OutputVarMap[outvar]);
	outvar = "v";
	writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, XModel.OutputVarMap[outvar]);
	
}

template void TestingOutput<float>(Param XParam, Model<float> XModel);
template void TestingOutput<double>(Param XParam, Model<double> XModel);


template <class T> void copyID2var(Param XParam, BlockP<T> XBlock, T* z)
{
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int n = memloc(XParam, ix, iy, ib);
				z[n] = ib;
			}
		}
	}

}

template void copyID2var<float>(Param XParam, BlockP<float> XBlock, float* z);
template void copyID2var<double>(Param XParam, BlockP<double> XBlock, double* z);

template <class T> void copyBlockinfo2var(Param XParam, BlockP<T> XBlock, int* blkinfo, T* z)
{
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		int info = blkinfo[ib];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int n = memloc(XParam, ix, iy, ib);
				z[n] = T(info);
			}
		}
	}

}
template void copyBlockinfo2var<float>(Param XParam, BlockP<float> XBlock, int* blkinfo, float* z);
template void copyBlockinfo2var<double>(Param XParam, BlockP<double> XBlock, int* blkinfo, double* z);



template <class T> void CompareCPUvsGPU(Param XParam, Model<T> XModel, Model<T> XModel_g, std::vector<std::string> varlist, bool checkhalo)
{
	Loop<T> XLoop;
	// GPU stuff
	

	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();

	XLoop.totaltime = 0.0;

	XLoop.nextoutputtime = 3600.0;


	T* gpureceive;
	T* diff;

	//Forcing<float> XForcing;

	AllocateCPU(XParam.nblkmem, XParam.blksize, gpureceive);
	AllocateCPU(XParam.nblkmem, XParam.blksize, diff);


	//============================================
	// Compare gradients for evolving parameters
	
	// calculate difference
	//diffArray(XParam, XLoop, XModel.blocks, XModel.evolv.h, XModel_g.evolv.h, XModel.evolv_o.u);

	creatncfileBUQ(XParam, XModel.blocks);

	for (int ivar = 0; ivar < varlist.size(); ivar++)
	{
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, varlist[ivar], 3, XModel.OutputVarMap[varlist[ivar]]);
	}

	/*
	std::string varname = "dt";
	if (abs(dtgpu - dtcpu) < (XLoop.epsilon * 2))
	{
		log(varname + " PASS");
	}
	else
	{
		log(varname + " FAIL: " + " GPU(" + std::to_string(dtgpu) + ") - CPU("+std::to_string(dtcpu) +") =  difference: "+  std::to_string(abs(dtgpu - dtcpu)) + " Eps: " + std::to_string(XLoop.epsilon));
		
	}
	*/
	//Check variable
	for (int ivar = 0; ivar < varlist.size(); ivar++)
	{
		diffArray(XParam, XLoop, XModel.blocks, varlist[ivar], checkhalo, XModel.OutputVarMap[varlist[ivar]], XModel_g.OutputVarMap[varlist[ivar]], gpureceive, diff);
	}
	
	
	
	free(gpureceive);
	free(diff);
	
}
template void CompareCPUvsGPU<float>(Param XParam, Model<float> XModel, Model<float> XModel_g, std::vector<std::string> varlist, bool checkhalo);
template void CompareCPUvsGPU<double>(Param XParam, Model<double> XModel, Model<double> XModel_g, std::vector<std::string> varlist, bool checkhalo);


template <class T> void diffdh(Param XParam, BlockP<T> XBlock, T* input, T* output,T* shuffle)
{
	int iright, itop;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XBlock.active[ibl];


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

				iright = memloc(XParam.halowidth, XParam.blkmemwidth, ix + 1, iy, ib);
				itop = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy + 1, ib);

				output[i] = input[iright] - input[i];
				shuffle[i] = input[iright];
			}
		}
	}
}

template <class T> void diffSource(Param XParam, BlockP<T> XBlock, T* Fqux, T* Su, T* output)
{
	int iright, itop;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XBlock.active[ibl];


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

				iright = memloc(XParam.halowidth, XParam.blkmemwidth, ix + 1, iy, ib);
				itop = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy + 1, ib);

				output[i] = Fqux[i]  - Su[iright];
				//shuffle[i] = input[iright];
			}
		}
	}
}


template <class T> void diffArray(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, std::string varname,bool checkhalo, T* cpu, T* gpu, T* dummy, T* out)
{
	T diff, maxdiff, rmsdiff;
	unsigned int nit = 0;
	//copy GPU back to the CPU (store in dummy)
	CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, dummy, gpu);

	rmsdiff = T(0.0);
	maxdiff = XLoop.hugenegval;
	// calculate difference
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XBlock.active[ibl];

		int yst = checkhalo ? -1 : 0;
		int ynd = checkhalo ? XParam.blkwidth + 1 : XParam.blkwidth;

		int xst = checkhalo ? -1 : 0;
		int xnd = checkhalo ? XParam.blkwidth + 1 : XParam.blkwidth;

		for (int iy = yst; iy < ynd; iy++)
		{
			for (int ix = xst; ix < xnd; ix++)
			{
				int n = memloc(XParam, ix, iy, ib);
				diff = dummy[n] - cpu[n];
				maxdiff = utils::max(abs(diff), maxdiff);
				rmsdiff = rmsdiff + utils::sq(diff);
				nit++;
				out[n] = diff;
			}
		}

	}


	rmsdiff = rmsdiff / nit;

	

	if (maxdiff <= T(100.0)*(XLoop.epsilon))
	{
		log(varname + " PASS");
	}
	else
	{
		log(varname + " FAIL: " + " Max difference: " + std::to_string(maxdiff) + " RMS difference: " + std::to_string(rmsdiff) + " Eps: " + std::to_string(XLoop.epsilon));
		defncvarBUQ(XParam, XBlock.active, XBlock.level, XBlock.xo, XBlock.yo, varname + "_CPU", 3, cpu);
		defncvarBUQ(XParam, XBlock.active, XBlock.level, XBlock.xo, XBlock.yo, varname + "_GPU", 3, dummy);
		defncvarBUQ(XParam, XBlock.active, XBlock.level, XBlock.xo, XBlock.yo, varname + "_diff", 3, out);
	}
	



}



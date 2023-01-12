
#include "Testing.h"




/*! \fn bool testing(Param XParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g)
* Wrapping function for all the inbuilt test
* This function is the entry point to other function below.
*
* Test 0 is a gausian hump propagating on a flat uniorm cartesian mesh (both GPU and CPU version tested)
* Test 1 is vertical discharge on a flat uniorm cartesian mesh (GPU or CPU version)
* Test 2 Gaussian wave on Cartesian grid (same as test 0): CPU vs GPU (GPU required)
* Test 3 Test Reduction algorithm
* Test 4 Boundary condition test
* Test 5 Lake at rest test for Ardusse/kurganov reconstruction/scheme
* Test 6 Mass conservation on a slope
* Test 7 Mass conservation with rain fall on grid
* Test 8 Rain Map forcing (comparison map and Time Serie and test case with slope and non-uniform rain map)
* Test 9 Zoned output (test zoned outputs with adaptative grid)

* Test 99 Run all the test with test number < 99.

The following test are not independant, they are tools to check or debug a personnal case
* Test 998 Compare resuts between the CPU and GPU Flow functions (GPU required)
* Test 999 Run the main loop and engine in debug mode
*/
template <class T> bool Testing(Param XParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g)
{

	bool isfailed = false;
	std::string result;

	log("\nRunning internal test(s):");

	int mytest;
	mytest = XParam.test;
	if (XParam.test == 99)
	{
		mytest = 0;
	}

	while (mytest <= XParam.test)
	{
		if (mytest == 0)
		{
			bool bumptest, bumptestComp;
			bool bumptestGPU = true;
			// Test 0 is pure bump test
			log("\t ### Gaussian wave on Cartesian grid ###");
			//set gpu is -1 for cpu test

			bumptest = GaussianHumptest(0.1, -1, false);
			result = bumptest ? "successful" : "failed";
			log("\t\tCPU test: " + result);

			// If original XParam tried to use GPU we try also
			if (XParam.GPUDEVICE >= 0)
			{
				bumptestGPU = GaussianHumptest(0.1, XParam.GPUDEVICE, false);
				result = bumptestGPU ? "successful" : "failed";
				log("\t\tGPU test: " + result);

				if (!bumptestGPU)
				{
					bumptestComp = GaussianHumptest(0.1, XParam.GPUDEVICE, true);
				}
			}
			isfailed = ((bumptest == true) && (bumptestGPU == true)) ? false : true;
		}
		if (mytest == 1)
		{
			bool rivertest;
			// Test 1 is vertical discharge on a flat uniorm cartesian mesh (GPU and CU version)
			log("\t ### River Mass conservation grid ###");
			rivertest = Rivertest(0.1, -1);
			result = rivertest ? "successful" : "failed";
			log("\t\tCPU test: " + result);
			isfailed = (!rivertest || isfailed) ? true : false;

			log(" \t\t\t GPU device= " + XParam.GPUDEVICE);

			if (XParam.GPUDEVICE >= 0)
			{
				rivertest = Rivertest(0.1, XParam.GPUDEVICE);
				result = rivertest ? "successful" : "failed";
				log("\t\tGPU test: " + result);
				isfailed = (!rivertest || isfailed) ? true : false;
			}

			rivertest=RiverVolumeAdapt(XParam, T(0.4));
			result = rivertest ? "successful" : "failed";
			log("\t\tRiver Volume Adapt: " + result);
			isfailed = (!rivertest || isfailed) ? true : false;

		}
		if (mytest == 2)
		{
			if (XParam.GPUDEVICE >= 0)
			{
				bool GPUvsCPUtest;
				log("\t### Gaussian wave on Cartesian grid: CPU vs GPU ###");
				GPUvsCPUtest = GaussianHumptest(0.1, XParam.GPUDEVICE, true);
				result = GPUvsCPUtest ? "successful" : "failed";
				log("\t\tCPU vs GPU test: " + result);
				isfailed = (!GPUvsCPUtest || isfailed) ? true : false;
			}
			else
			{
				log("Specify GPU device to run test 2 (CPU vs GPU comparison)");
			}
		}
		if (mytest == 3)
		{

			bool testresults;
			bool testreduction = true;

			// Iterate this test niter times:
			int niter = 1000;
			srand(time(0));
			log("\t### Reduction Test ###");
			for (int iter = 0; iter < niter; iter++)
			{
				testresults = reductiontest(XParam, XModel, XModel_g);
				testreduction = testreduction && testresults;
			}

			result = testreduction ? "successful" : "failed";
			log("\t\tReduction test: " + result);
			isfailed = (!testreduction || isfailed) ? true : false;

		}
		if (mytest == 4)
		{
			log("\t### Boundary Test ###");
			bool testBound = testboundaries(XParam, T(0.1));
			result = testBound ? "successful" : "failed";
			isfailed = (!testBound || isfailed) ? true : false;
			log("\t\tboundaries test: " + result);
		}
		if (mytest == 5)
		{
			log("\t### Lake-at-rest Test ###");
			bool testTLAR = ThackerLakeAtRest(XParam, T(0.0));
			result = testTLAR ? "successful" : "failed";
			isfailed = (!testTLAR || isfailed) ? true : false;
			log("\t\tThaker lake-at-rest test: " + result);
			testTLAR = LakeAtRest(XParam, XModel);
			isfailed = (!testTLAR || isfailed) ? true : false;
			log("\t\tLake-at-rest test: " + result);
		}
		if (mytest == 6)
		{
			log("\t### Mass conservation Test ###");
			bool testSteepSlope = MassConserveSteepSlope(XParam.zsinit, XParam.GPUDEVICE);
			result = testSteepSlope ? "successful" : "failed";
			isfailed = (!testSteepSlope || isfailed) ? true : false;
			log("\t\tMass conservation test: " + result);
		}
		if (mytest == 7)
		{
			bool testrainGPU, testrainCPU;
			/* Test 7 is homogeneous rain on a uniform slope for cartesian mesh (GPU and CU version)
			 The input parameters are :
					- the initial water level (zs)
					- GPU option
					- the slope (%)
			*/
			log("\t### Homogeneous rain on grid Mass conservation test ###");
			testrainGPU = Raintest(0.0, 0, 10);
			result = testrainGPU ? "successful" : "failed";
			log("\t\tHomogeneous rain on grid test GPU: " + result);
			testrainCPU = Raintest(0.0, -1, 10);
			result = testrainCPU ? "successful" : "failed";
			log("\t\tHomogeneous rain on grid test CPU: " + result);
			isfailed = (!testrainCPU || !testrainGPU || isfailed) ? true : false;
		}
		if (XParam.test == 8)
		{
			bool raintest2;
			/* Test 8 is non-homogeneous rain on a non-uniform slope for cartesian mesh (GPU and CPU version)
			 It is based on a teste case from litterature (Iwagaki1955) and tests the different
			 rain inputs (time serie for 1D input or netCDF file).
			*/

			log("\t non-uniform rain forcing on slope based on Aureli2020");
			int gpu = 0;
			raintest2 = Raintestinput(gpu);
			result = raintest2 ? "successful" : "failed";
			log("\t\tNon-uniform rain forcing : " + result);
		}
		if (mytest == 9)
		{
			bool testzoneOutDef, testzoneOutUser;
			/* Test 9 is basic configuration to test the zoned outputs, with different resolutions.
			 The default (without zoned defined by user) configuration is tested.
			 Then, the creation of 3 zones is then tested(whole, zoned complexe, zoned with part of the levels).
			 The size of the created nc files is used to verified this fonctionnality.
			 Parameter: nbzones: number of zones for output defined by the user
						zsinit: initial water elevation
			*/

			log("\t### Test zoned output ###");
			int nbzones = 0;
			T zsinit = 0.01;
			testzoneOutDef = ZoneOutputTest(nbzones, zsinit);
			result = testzoneOutDef ? "successful" : "failed";
			log("\n\nDefault zoned Outputs: " + result);
			nbzones = 3; // 3 only
			testzoneOutUser = ZoneOutputTest(nbzones, zsinit);
			result = testzoneOutUser ? "successful" : "failed";
			log("\n\nUser defined zones Outputs: " + result);
			isfailed = (!testzoneOutDef || !testzoneOutUser || isfailed) ? true : false;
		}
		if (mytest == 10)
		{
			bool instab;
			log("\t### Wet/dry Instability test with Conserve Elevation ###");
			instab=TestInstability(XParam, XModel, XModel_g);
			result = instab ? "successful" : "failed";
			log("\t\tWet/dry Instability test : " + result);
		}

		if (mytest == 995)
		{
			TestFirsthalfstep(XParam, XForcing, XModel, XModel_g);
		}
		if (mytest == 996)
		{
			TestHaloSpeed(XParam,XModel,XModel_g);
		}
		if (mytest == 997)
		{
			TestGradientSpeed(XParam, XModel, XModel_g);
		}
		if (mytest == 998)
		{
			//
			bool testresults;
			log("\t### CPU vs GPU Test ###");
			testresults = CPUGPUtest(XParam, XModel, XModel_g);
			isfailed = (!testresults || isfailed) ? true : false;

			if (testresults)
			{
				exit(0);
			}
			else
			{
				exit(1);
			}
		}
		if (XParam.test == 999)
		{
			//
			DebugLoop(XParam, XForcing, XModel, XModel_g);
		}
		mytest++;
	}
	return(isfailed);
}
template bool Testing<float>(Param XParam, Forcing<float> XForcing, Model<float> XModel, Model<float> XModel_g);
template bool Testing<double>(Param XParam, Forcing<float> XForcing, Model<double> XModel, Model<double> XModel_g);


/*! \fn bool GaussianHumptest(T zsnit, int gpu, bool compare)
*
* This function tests the full hydrodynamics model and compares the results with pre-conmputed (Hard wired) values
*	The function creates it own model setup and mesh independantly to what the user might want to do
*	The setup consist of a centrally located gaussian hump radiating away
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
	//double ZsVerifKurganov[8] = { 0.100000000023, 0.100000063119, 0.100110376004, 0.195039970749, 0.136739044168, 0.0848024805994, 0.066275833049, 0.0637058445888 };
	//double ZsVerification[8] = { 0.100000008904, 0.187920326216, 0.152329657390, 0.117710230042, 0.0828616638138, 0.0483274739972, 0.0321501737555, 0.0307609731288 };
	double ZsVerifButtinger[8] = { 0.100000000023, 0.100000063119, 0.100093580546, 0.195088199869, 0.136767978925, 0.0850706353898, 0.0663028448129, 0.063727949607 };




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
		delta = T(calcres(XParam.dx, XModel.blocks.level[ib]));


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				//
				int n = memloc(XParam, ix, iy, ib);
				x = T(XParam.xo) + XModel.blocks.xo[ib] + ix * delta;
				y = T(XParam.yo) + XModel.blocks.yo[ib] + iy * delta;
				XModel.evolv.zs[n] = XModel.evolv.zs[n] + a * exp(T(-1.0) * ((x - xorigin) * (x - xorigin) + (y - yorigin) * (y - yorigin)) / (T(2.0) * cc * cc));
				XModel.evolv.h[n] = utils::max(XModel.evolv.zs[n] - XModel.zb[n], T(0.0));

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

	fillHaloC(XParam, XModel.blocks, XModel.zb);
	gradientC(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
	gradientHalo(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);

	refine_linear(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
	gradientHalo(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);

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

			T diffdt = T(XLoop_g.dt - XLoop.dt);
			if (abs(diffdt) > T(100.0) * (XLoop.epsilon))
			{
				printf("Timestep Difference=%f\n", diffdt);

				compare = false;
			}
			CompareCPUvsGPU(XParam, XModel, XModel_g, outv, false);
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

				int ix, iy, ib, ii, jj, ibx, iby, nbx;
				jj = 127;
				ii = (iv + 1) * 16 - 1;

				// Theoretical size is 255x255
				nbx = 256 / 16;
				

				ibx = ftoi(floor(ii / XParam.blkwidth));
				iby = ftoi(floor(jj / XParam.blkwidth));

				ib = (iby)*nbx + ibx;

				ix = ii - ibx * XParam.blkwidth;
				iy = jj - iby * XParam.blkwidth;

				int n = memloc(XParam, ix, iy, ib);

				diff = abs(T(XModel.evolv.zs[n]) - ZsVerifButtinger[iv]);



				if (diff > 1e-6)//Tolerance is 1e-6 or 1e-7/1e-8??
				{

					printf("ib=%d, ix=%d, iy=%d; simulated=%f; expected=%f; diff=%e\n", ib, ix, iy, XModel.evolv.zs[n], ZsVerifButtinger[iv], diff);
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
	T delta = 0;
	T initVol = 0;
	T finalVol = 0;
	T TheoryInput = 0;

	// initialise domain and required resolution
	XParam.dx = 1.0 / ((1 << 4));
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

	std::vector<std::string> outv = { "zb","h","zs","u","v","Fqux","Fqvx","Fquy","Fqvy", "Fhu", "Fhv", "dh", "dhu", "dhv", "Su", "Sv","dhdx", "dhdy", "dudx", "dvdx", "dzsdx", "twet", "hUmax", "Umean"};
	XParam.outvars = outv;

	XParam.outmax = true;
	XParam.outmean = true;
	XParam.outtwet = true;

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
	T Q = T(0.001);
	TheoryInput = Q * T(XParam.outputtimestep);


	//Create a temporary file with river fluxes
	std::ofstream river_file(
		"testriver.tmp", std::ios_base::out | std::ios_base::trunc);
	river_file << "0.0 " + std::to_string(Q) << std::endl;
	river_file << "3600.0 " + std::to_string(Q) << std::endl;
	river_file.close(); //destructor implicitly does it

	River thisriver;
	thisriver.Riverflowfile = "testriver.tmp";
	thisriver.xstart = -1.0 * XParam.dx * 3.0;
	thisriver.xend = XParam.dx * 3.0;
	thisriver.ystart = -1.0 * XParam.dx * 3.0;
	thisriver.yend = XParam.dx * 3.0;

	XForcing.rivers.push_back(thisriver);


	XForcing.rivers[0].flowinput = readFlowfile(XForcing.rivers[0].Riverflowfile);


	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	InitialConditions(XParam, XForcing, XModel);
	InitialAdaptation(XParam, XForcing, XModel);

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

	fillHaloC(XParam, XModel.blocks, XModel.zb);

	// Calculate initial water volume
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XModel.blocks.active[ibl];
		delta = calcres(T(XParam.dx), XModel.blocks.level[ib]);


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				//
				int n = memloc(XParam, ix, iy, ib);
				//printf("h[%d] = %f\n", n, XModel.evolv.h[n]);
				initVol = initVol + XModel.evolv.h[n] * delta * delta;
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
			printf("h[1] = %f\n", XModel.evolv.h[1]);
			FlowCPU(XParam, XLoop, XForcing, XModel);
		}
		XLoop.totaltime = XLoop.totaltime + XLoop.dt;
		//Save2Netcdf(XParam, XLoop, XModel);

		if (XLoop.nextoutputtime - XLoop.totaltime <= XLoop.dt * T(0.00001) && XParam.outputtimestep > 0.0)
		{
			if (XParam.GPUDEVICE >= 0)
			{
				for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
				{
					CUDA_CHECK(cudaMemcpy(XModel.OutputVarMap[XParam.outvars[ivar]], XModel_g.OutputVarMap[XParam.outvars[ivar]], XParam.nblkmem * XParam.blksize * sizeof(T), cudaMemcpyDeviceToHost));
				}
			}

			//Save2Netcdf(XParam, XLoop, XModel);
			// Verify the Validity of results
			finalVol = T(0.0);
			for (int ibl = 0; ibl < XParam.nblk; ibl++)
			{
				//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
				int ib = XModel.blocks.active[ibl];
				delta = calcres(T(XParam.dx), XModel.blocks.level[ib]);


				for (int iy = 0; iy < XParam.blkwidth; iy++)
				{
					for (int ix = 0; ix < XParam.blkwidth; ix++)
					{
						//
						int n = memloc(XParam, ix, iy, ib);
						//printf("h[%d] = %f\n", n, XModel.evolv.h[n]);
						finalVol = finalVol + XModel.evolv.h[n] * delta * delta;
					}
				}
			}
			T error = ((finalVol - initVol) - TheoryInput) / TheoryInput;
			printf("error = %g %%, initial volume=%4.4f; final Volume=%4.4f; abs. difference=%4.4f, Theoretical  input=%4.4f \n", error, initVol, finalVol, abs(finalVol - initVol), TheoryInput);


			modelgood = abs(error) < 0.05;
		}



	}

	if (!modelgood)
	{
		InitSave2Netcdf(XParam, XModel);

	}


	log("#####");
	return modelgood;
}
template bool Rivertest<float>(float zsnit, int gpu);
template bool Rivertest<double>(double zsnit, int gpu);



/*! \fn bool MassConserveSteepSlope(T zsnit, int gpu)
*
* This function tests the mass conservation of the vertical injection (used for rivers)
*	The function creates it own model setup and mesh independantly to what the user might want to do
*	This starts with a initial water level (zsnit=0 is dry) and runs for 0.1s before comparing results
*	with zsnit=0.1 that is approx 20 steps
*/
template <class T> bool MassConserveSteepSlope(T zsnit, int gpu)
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
	XParam.minlevel = -1;
	XParam.maxlevel = 1;

	XParam.AdaptCrit = "Threshold";
	XParam.Adapt_arg1 = "3.5";
	XParam.Adapt_arg2 = "zb";

	XParam.zsinit = zsnit;
	XParam.zsoffset = 0.0;

	//Output times for comparisons
	XParam.endtime = 1.0;
	XParam.outputtimestep = 0.04;//0.035;

	XParam.smallnc = 0;

	XParam.cf = 0.001;
	XParam.frictionmodel = 1;

	XParam.conserveElevation = false;

	// Enforece GPU/CPU
	XParam.GPUDEVICE = gpu;
	std::vector<std::string> outv = { "zb","h","zs","u","v","Fqux","Fqvx","Fquy","Fqvy", "Fhu", "Fhv", "dh", "dhu", "dhv", "Su", "Sv","dhdx", "dhdy" };


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
			XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = T(i * 4);
		}
	}
	//
	//
	// 
	T Q = T(0.10);
	TheoryInput = Q * T(XParam.outputtimestep);


	//Create a temporary file with river fluxes
	std::ofstream river_file(
		"testriver.tmp", std::ios_base::out | std::ios_base::trunc);
	river_file << "0.0 " + std::to_string(Q) << std::endl;
	river_file << "3600.0 " + std::to_string(Q) << std::endl;
	river_file.close(); //destructor implicitly does it

	River thisriver;
	thisriver.Riverflowfile = "testriver.tmp";
	thisriver.xstart = -1.0 * XParam.dx * 3.0;
	thisriver.xend = XParam.dx * 3.0;
	thisriver.ystart = -1.0 * XParam.dx * 3.0;
	thisriver.yend = XParam.dx * 3.0;

	XForcing.rivers.push_back(thisriver);


	XForcing.rivers[0].flowinput = readFlowfile(XForcing.rivers[0].Riverflowfile);


	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	InitialConditions(XParam, XForcing, XModel);

	InitialAdaptation(XParam, XForcing, XModel);

	SetupGPU(XParam, XModel, XForcing, XModel_g);

	Loop<T> XLoop;

	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();

	XLoop.totaltime = 0.0;



	InitSave2Netcdf(XParam, XModel);
	XLoop.nextoutputtime = XParam.outputtimestep;
	XLoop.dtmax = 0.025;// initdt(XParam, XLoop, XModel);




	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XModel.blocks.active[ibl];
		//delta = calcres(XParam.dx, XModel.blocks.level[ib]);


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				//
				int n = memloc(XParam, ix, iy, ib);
				if (XModel.zb[n] < XParam.eps)
				{
					printf("ix=%d, iy=%d, ib=%d, n=%d; zb=%f \n", ix, iy, ib, n, XModel.zb[n]);
				}
			}
		}
	}

	if (XParam.GPUDEVICE >= 0)
	{
		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));

		fillHaloGPU(XParam, XModel_g.blocks, stream, XModel_g.zb);

		cudaStreamDestroy(stream);
	}
	else
	{
		fillHaloC(XParam, XModel.blocks, XModel.zb);
	}

	initVol = T(0.0);
	// Calculate initial water volume
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XModel.blocks.active[ibl];
		delta = calcres(T(XParam.dx), XModel.blocks.level[ib]);


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				//
				int n = memloc(XParam, ix, iy, ib);
				initVol = initVol + XModel.evolv.h[n] * delta * delta;
			}
		}
	}


	//InitSave2Netcdf(XParam, XModel);+



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
		//Save2Netcdf(XParam, XLoop, XModel);

		if (XLoop.nextoutputtime - XLoop.totaltime <= XLoop.dt * T(0.00001) && XParam.outputtimestep > 0.0)
		{
			if (XParam.GPUDEVICE >= 0)
			{
				for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
				{
					CUDA_CHECK(cudaMemcpy(XModel.OutputVarMap[XParam.outvars[ivar]], XModel_g.OutputVarMap[XParam.outvars[ivar]], XParam.nblkmem * XParam.blksize * sizeof(T), cudaMemcpyDeviceToHost));
				}
			}

			Save2Netcdf(XParam, XLoop, XModel);
			// Verify the Validity of results
			finalVol = T(0.0);
			for (int ibl = 0; ibl < XParam.nblk; ibl++)
			{
				//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
				int ib = XModel.blocks.active[ibl];
				delta = calcres(T(XParam.dx), XModel.blocks.level[ib]);


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
template bool MassConserveSteepSlope<float>(float zsnit, int gpu);
template bool MassConserveSteepSlope<double>(double zsnit, int gpu);


/*! \fn T reductiontest(Param XParam, Model<T> XModel, Model<T> XModel_g)
*
*	Test the algorithm for reducing the global time step on the user grid layout
*/
template <class T> bool reductiontest(Param XParam, Model<T> XModel, Model<T> XModel_g)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	//srand(seed);
	T mininput = T(rand()) / T(RAND_MAX);
	bool test = true;

	Loop<T> XLoop;

	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();

	XLoop.totaltime = 0.0;

	//InitSave2Netcdf(XParam, XModel);
	XLoop.nextoutputtime = mininput * T(2.0);
	XLoop.dtmax = mininput * T(2.01);

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
				XModel.time.dtmax[n] = mininput * T(1.1) + utils::max(T(rand()) / T(RAND_MAX), T(0.0));
			}
		}
	}

	// randomly select a block a i and a j were the maximum value will be relocated
	int ibbl = ftoi(floor(T(rand()) / T(RAND_MAX) * XParam.nblk));
	int ibb = XModel.blocks.active[ibbl];
	int ixx = ftoi(floor(T(rand()) / T(RAND_MAX) * XParam.blkwidth));
	int iyy = ftoi(floor(T(rand()) / T(RAND_MAX) * XParam.blkwidth));

	int nn = memloc(XParam, ixx, iyy, ibb);

	XModel.time.dtmax[nn] = mininput;

	T reducedt = CalctimestepCPU(XParam, XLoop, XModel.blocks, XModel.time);

	test = abs(reducedt - mininput) < T(100.0) * (XLoop.epsilon);
	bool testgpu;

	if (!test)
	{
		char buffer[256]; sprintf(buffer, "%e", abs(reducedt - mininput));
		std::string str(buffer);
		log("\t\t CPU test failed! : Expected=" + std::to_string(mininput) + ";  Reduced=" + std::to_string(reducedt)+ ";  error=" +str);
	}

	if (XParam.GPUDEVICE >= 0)
	{

		reset_var << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel_g.blocks.active, XLoop.hugeposval, XModel_g.time.dtmax);
		CUDA_CHECK(cudaDeviceSynchronize());

		CopytoGPU(XParam.nblkmem, XParam.blksize, XModel.time.dtmax, XModel_g.time.dtmax);
		T reducedtgpu = CalctimestepGPU(XParam, XLoop, XModel_g.blocks, XModel_g.time);
		testgpu = abs(reducedtgpu - mininput) < T(100.0) * (XLoop.epsilon);

		if (!testgpu)
		{
			char buffer[256]; sprintf(buffer, "%e", abs(reducedtgpu - mininput));
			std::string str(buffer);
			log("\t\t GPU test failed! : Expected=" + std::to_string(mininput) + ";  Reduced=" + std::to_string(reducedtgpu) + ";  error=" + str);
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
	std::string evolvst[4] = { "h","zs","u","v" };

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
	fillgauss(XParam, XModel.blocks, T(0.5 * testamp), XModel.evolv.u);
	fillgauss(XParam, XModel.blocks, T(0.5 * testamp), XModel.evolv.v);

	//copy to GPU
	CopytoGPU(XParam.nblkmem, XParam.blksize, XModel.evolv, XModel_g.evolv);

	//============================================
	//  Fill the halo for gradient reconstruction
	fillHalo(XParam, XModel.blocks, XModel.evolv, XModel.zb);
	fillHaloGPU(XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.zb);

	CompareCPUvsGPU(XParam, XModel, XModel_g, evolvVar, true);

	//============================================
	//perform gradient reconstruction
	//gradientCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.zb);
	//gradientGPU(XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.grad, XModel.zb);


	// CPU gradients
	std::thread t0(&gradientC<T>, XParam, XModel.blocks, XModel.evolv.h, XModel.grad.dhdx, XModel.grad.dhdy);
	std::thread t1(&gradientC<T>, XParam, XModel.blocks, XModel.evolv.zs, XModel.grad.dzsdx, XModel.grad.dzsdy);
	std::thread t2(&gradientC<T>, XParam, XModel.blocks, XModel.evolv.u, XModel.grad.dudx, XModel.grad.dudy);
	std::thread t3(&gradientC<T>, XParam, XModel.blocks, XModel.evolv.v, XModel.grad.dvdx, XModel.grad.dvdy);

	t0.join();
	t1.join();
	t2.join();
	t3.join();

	//GPU gradients

	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel_g.blocks.active, XModel_g.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel_g.evolv.h, XModel_g.grad.dhdx, XModel_g.grad.dhdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel_g.blocks.active, XModel_g.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel_g.evolv.zs, XModel_g.grad.dzsdx, XModel_g.grad.dzsdy);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel_g.blocks.active, XModel_g.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel_g.evolv.u, XModel_g.grad.dudx, XModel_g.grad.dudy);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel_g.blocks.active, XModel_g.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel_g.evolv.v, XModel_g.grad.dvdx, XModel_g.grad.dvdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	std::string gradst[8] = { "dhdx","dzsdx","dudx","dvdx","dhdy","dzsdy","dudy","dvdy" };

	std::vector<std::string> gradVar;

	for (int nv = 0; nv < 8; nv++)
	{
		gradVar.push_back(gradst[nv]);
	}

	CompareCPUvsGPU(XParam, XModel, XModel_g, gradVar, false);

	// Gradient in Halo

	// CPU
	gradientHalo(XParam, XModel.blocks, XModel.evolv.h, XModel.grad.dhdx, XModel.grad.dhdy);
	gradientHalo(XParam, XModel.blocks, XModel.evolv.zs, XModel.grad.dzsdx, XModel.grad.dzsdy);
	gradientHalo(XParam, XModel.blocks, XModel.evolv.u, XModel.grad.dudx, XModel.grad.dudy);
	gradientHalo(XParam, XModel.blocks, XModel.evolv.v, XModel.grad.dvdx, XModel.grad.dvdy);

	// GPU
	gradientHaloGPU(XParam, XModel_g.blocks, XModel_g.evolv.h, XModel_g.grad.dhdx, XModel_g.grad.dhdy);
	gradientHaloGPU(XParam, XModel_g.blocks, XModel_g.evolv.zs, XModel_g.grad.dzsdx, XModel_g.grad.dzsdy);
	gradientHaloGPU(XParam, XModel_g.blocks, XModel_g.evolv.u, XModel_g.grad.dudx, XModel_g.grad.dudy);
	gradientHaloGPU(XParam, XModel_g.blocks, XModel_g.evolv.v, XModel_g.grad.dvdx, XModel_g.grad.dvdy);

	CompareCPUvsGPU(XParam, XModel, XModel_g, gradVar, true);

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
	updateKurgXGPU << < gridDim, blockDimKX, 0 >> > (XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.grad, XModel_g.flux, XModel_g.time.dtmax, XModel_g.zb);
	CUDA_CHECK(cudaDeviceSynchronize());


	// Y- direction
	updateKurgYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);

	updateKurgYGPU << < gridDim, blockDimKY, 0 >> > (XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.grad, XModel_g.flux, XModel_g.time.dtmax, XModel_g.zb);
	CUDA_CHECK(cudaDeviceSynchronize());

	CompareCPUvsGPU(XParam, XModel, XModel_g, fluxVar, false);


	fillHalo(XParam, XModel.blocks, XModel.flux);
	fillHaloGPU(XParam, XModel_g.blocks, XModel_g.flux);

	CompareCPUvsGPU(XParam, XModel, XModel_g, fluxVar, true);


	//============================================
	// Update step
	std::string advst[3] = { "dh","dhu","dhv" };

	std::vector<std::string> advVar;

	for (int nv = 0; nv < 3; nv++)
	{
		advVar.push_back(advst[nv]);
	}
	updateEVCPU(XParam, XModel.blocks, XModel.evolv, XModel.flux, XModel.adv);
	updateEVGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.flux, XModel_g.adv);
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
	AdvkernelGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, T(0.0005), XModel_g.zb, XModel_g.evolv, XModel_g.adv, XModel_g.evolv_o);
	CUDA_CHECK(cudaDeviceSynchronize());

	CompareCPUvsGPU(XParam, XModel, XModel_g, evoVar, false);

	//============================================
	// Bottom friction

	bottomfrictionCPU(XParam, XModel.blocks, T(0.5), XModel.cf, XModel.evolv_o);

	bottomfrictionGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, T(0.5), XModel_g.cf, XModel_g.evolv_o);
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

/*! \fn T ValleyBathy(T x, T y, T slope, T center)
* \brief	create V shape Valley basin
*
* This function creates a simple V shape Valley basin
*
*
*/
template <class T> T ValleyBathy(T x, T y, T slope, T center)
{


	T bathy;

	bathy = (abs(x - center) + y) * slope;


	return bathy;
}


/*! \fn T ThackerBathy(T x, T y, T L, T D)
* \brief	create a parabolic bassin
*
* This function creates a parabolic bassin. The function returns a single value of the bassin
*
* Borrowed from Buttinger et al. 2019.
*
* ### Reference
* Buttinger-Kreuzhuber, A., Horváth, Z., Noelle, S., Blöschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional
* structured grids over abrupt topography, Advances in water resources, 127, 89–108, 2019.
*/
template <class T> T ThackerBathy(T x, T y, T L, T D)
{


	T bathy = D * ((x * x + y * y) / (L * L) - 1.0);


	return bathy;
}

/*! \fn
* \brief	Simulate the Lake-at-rest in a parabolic bassin
* 
* This function creates a parabolic bassin filled to a given level and run the modle for a while and checks that the velocities in the lake remain very small
* thus verifying the well-balancedness of teh engine and the Lake-at-rest condition.
*
* Borrowed from Buttinger et al. 2019.
*
* ### Reference
* Buttinger-Kreuzhuber, A., Horváth, Z., Noelle, S., Blöschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional
* structured grids over abrupt topography, Advances in water resources, 127, 89–108, 2019.
*/
template <class T> bool ThackerLakeAtRest(Param XParam,T zsinit)
{
	bool test = true;
	// Make a Parabolic bathy
	
	auto modeltype = XParam.doubleprecision < 1 ? float() : double();
	Model<decltype(modeltype)> XModel; // For CPU pointers
	Model<decltype(modeltype)> XModel_g; // For GPU pointers

	Forcing<float> XForcing;

	StaticForcingP<float> bathy;

	XForcing.Bathy.push_back(bathy);

	// initialise forcing bathymetry to 0

	T Lo = T(2500.0);
	T Do = T(1.0);

	T x, y;



	XForcing.Bathy[0].xo = -4000.0;
	XForcing.Bathy[0].yo = -4000.0;

	XForcing.Bathy[0].xmax = 4000.0;
	XForcing.Bathy[0].ymax = 4000.0;
	XForcing.Bathy[0].nx = 64;
	XForcing.Bathy[0].ny = 64;

	XForcing.Bathy[0].dx = 126.0;

	AllocateCPU(1, 1, XForcing.left.blks, XForcing.right.blks, XForcing.top.blks, XForcing.bot.blks);

	AllocateCPU(XForcing.Bathy[0].nx, XForcing.Bathy[0].ny, XForcing.Bathy[0].val);

	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			x = T(XForcing.Bathy[0].xo + i * XForcing.Bathy[0].dx);
			y = T(XForcing.Bathy[0].yo + j * XForcing.Bathy[0].dx);
			XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = float(ThackerBathy(x, y, Lo, Do));
		}
	}

	// Overrule whatever may be set in the param file
	XParam.xmax = XForcing.Bathy[0].xmax;
	XParam.ymax = XForcing.Bathy[0].ymax;
	XParam.xo = XForcing.Bathy[0].xo;
	XParam.yo = XForcing.Bathy[0].yo;

	XParam.dx = XForcing.Bathy[0].dx;

	XParam.zsinit = zsinit;
	XParam.endtime = 1390.0;

	XParam.outputtimestep = XParam.endtime; 

	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	InitialConditions(XParam, XForcing, XModel);

	InitialAdaptation(XParam, XForcing, XModel);

	
	SetupGPU(XParam, XModel, XForcing, XModel_g);

	MainLoop(XParam, XForcing, XModel, XModel_g);


	// Check Lake at rest state?
	// all velocities should be very small
	T smallvel = T(1e-6);
	int i;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < (XParam.blkwidth); ix++)
			{
				i = memloc(XParam, ix, iy, ib);
				if (abs(XModel.evolv.u[i]) > smallvel || abs(XModel.evolv.v[i]) > smallvel)
				{
					log("Lake at rest state not acheived!");
					test = false;
				}
			}
		}
	}

	return test;
}
template bool ThackerLakeAtRest<float>(Param XParam,float zsinit);
template bool ThackerLakeAtRest<double>(Param XParam, double zsinit);



/*! \fn bool RiverVolumeAdapt(Param XParam)
* \brief	Wraping function for RiverVolumeAdapt(Param XParam, T slope, bool bottop, bool flip)
*
* The function calls it's child function with different adaptation set in XParam so needs to be rerun to account for the different scenarios:
* * uniform level
* * flow from coasrse to fine
* * flow from fine to coarse
*
* and account for different flow direction
* 
*/
template <class T> bool RiverVolumeAdapt(Param XParam, T maxslope)
{
	//T maxslope = 0.45; // tthe mass conservation is better with smaller slopes 

	bool UnitestA, UnitestB, UnitestC, UnitestD;
	bool ctofA, ctofB, ctofC, ctofD;
	bool ftocA, ftocB, ftocC, ftocD;

	std::string details;

	XParam.minlevel = 1;
	XParam.maxlevel = 1;
	XParam.initlevel = 1;
	
	
	UnitestA=RiverVolumeAdapt(XParam, maxslope, false, false);
	UnitestB=RiverVolumeAdapt(XParam, maxslope, true, false);
	UnitestC=RiverVolumeAdapt(XParam, maxslope, false, true);
	UnitestD=RiverVolumeAdapt(XParam, maxslope, true, true);

	if (UnitestA && UnitestB && UnitestC && UnitestD)
	{
		log("River Volume Conservation Test: Uniform mesh: Success");
	}
	else
	{
		log("River Volume Conservation Test: Uniform mesh: Failed");
		details = UnitestA ? "successful" : "failed";
		log("\t Uniform mesh A :"+ details);
		details = UnitestB ? "successful" : "failed";
		log("\t Uniform mesh B :" + details);
		details = UnitestC ? "successful" : "failed";
		log("\t Uniform mesh C :" + details);
		details = UnitestD ? "successful" : "failed";
		log("\t Uniform mesh D :" + details);
	}

	XParam.minlevel = 0;
	XParam.maxlevel = 1;
	XParam.initlevel = 0;

	//Fine to coarse
	// Change arg 1 and 2 if the slope is changed
	XParam.AdaptCrit = "Inrange";
	XParam.Adapt_arg1 = "28.0";
	XParam.Adapt_arg2 = "40.0";
	XParam.Adapt_arg3 = "zb";

	ftocA = RiverVolumeAdapt(XParam, maxslope, false, false);
	ftocB = RiverVolumeAdapt(XParam, maxslope, true, false);
	ftocC = RiverVolumeAdapt(XParam, maxslope, false, true);
	ftocD = RiverVolumeAdapt(XParam, maxslope, true, true);
	if (ftocA && ftocB && ftocC && ftocD)
	{
		log("River Volume Conservation Test: Flow from fine to coarse adapted mesh: Success");
	}
	else
	{
		log("River Volume Conservation Test: Flow from fine to coarse adapted mesh: Failed");
		details = ftocA ? "successful" : "failed";
		log("\t Flow from fine to coarse adapted mesh A :" + details);
		details = ftocB ? "successful" : "failed";
		log("\t Flow from fine to coarse adapted mesh B :" + details);
		details = ftocC ? "successful" : "failed";
		log("\t Flow from fine to coarse adapted mesh C :" + details);
		details = ftocD ? "successful" : "failed";
		log("\t Flow from fine to coarse adapted mesh D :" + details);
	}

	//coarse to fine
	// Change arg 1 and 2 if the slope is changed
	XParam.AdaptCrit = "Inrange";
	XParam.Adapt_arg1 = "0.0";
	XParam.Adapt_arg2 = "2.0";
	XParam.Adapt_arg3 = "zb";

	ctofA = RiverVolumeAdapt(XParam, maxslope, false, false);
	ctofB = RiverVolumeAdapt(XParam, maxslope, true, false);
	ctofC = RiverVolumeAdapt(XParam, maxslope, false, true);
	ctofD = RiverVolumeAdapt(XParam, maxslope, true, true);
	if (ctofA && ctofB && ctofC && ctofD)
	{
		log("River Volume Conservation Test: Flow from coarse to fine adapted mesh: Success");
	}
	else
	{
		log("River Volume Conservation Test: Flow from coarse to fine adapted: Failed");
		details = ctofA ? "successful" : "failed";
		log("\t Flow from coarse to fine adapted mesh A :" + details);
		details = ctofB ? "successful" : "failed";
		log("\t Flow from coarse to fine adapted mesh B :" + details);
		details = ctofC ? "successful" : "failed";
		log("\t Flow from coarse to fine adapted mesh C :" + details);
		details = ctofD ? "successful" : "failed";
		log("\t Flow from coarse to fine adapted mesh D :" + details);

	}

	return (UnitestA * UnitestB * UnitestC * UnitestD * ctofA * ctofB * ctofC * ctofD * ftocA * ftocB * ftocC * ftocD);
}


/*! \fn bool RiverVolumeAdapt(Param XParam, T slope, bool bottop, bool flip)
* \brief	Simulate a river flowing in a steep valley
* and heck the Volume conservation
*
* This function creates a dry steep valley topography to a given level and run the model for a while and checks that the Volume matches the theory.
*
* The function can test the water volume for 4 scenario each time:
* * left to right: bottop=false & flip=true;
* * right to left: bottop=false & flip=false;
* * bottom to top: bottop=true & flip=true;
* * top to bottom: bottop=true & flip=false;
*
* The function inherits the adaptation set in XParam so needs to be rerun to accnout for the different scenarios:
* * uniform level
* * flow from coasrse to fine
* * flow from fine to coarse
* This is done in the higher level wrapping function
*/
template <class T> bool RiverVolumeAdapt(Param XParam, T slope, bool bottop, bool flip)
{
	//bool test = true;
	//

	auto modeltype = XParam.doubleprecision < 1 ? float() : double();
	Model<decltype(modeltype)> XModel; // For CPU pointers
	Model<decltype(modeltype)> XModel_g; // For GPU pointers

	Forcing<float> XForcing;

	XForcing = MakValleyBathy(XParam, slope, bottop, flip);

	T x, y;
	T center = T(10.5);

	float maxtopo = std::numeric_limits<float>::min();
	float mintopo = std::numeric_limits<float>::max();

	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			maxtopo = max(XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx], maxtopo);
			mintopo = min(XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx], mintopo);
		}
	}

	

	// Overrule whatever is set in the river forcing
	T Q = T(1.0);
	
	double upstream = !flip ? 24.0 : 8;
	double riverx = !bottop ? upstream : center;
	double rivery = !bottop ? center : upstream;

	//Create a temporary file with river fluxes
	std::ofstream river_file(
		"testriver.tmp", std::ios_base::out | std::ios_base::trunc);
	river_file << "0.0 " + std::to_string(Q) << std::endl;
	river_file << "3600.0 " + std::to_string(Q) << std::endl;
	river_file.close(); //destructor implicitly does it

	River thisriver;
	thisriver.Riverflowfile = "testriver.tmp";
	thisriver.xstart = riverx - 1.0;
	thisriver.xend = riverx + 1.0;
	thisriver.ystart = rivery - 1.0;
	thisriver.yend = rivery + 1.0;

	XForcing.rivers.push_back(thisriver);


	XForcing.rivers[0].flowinput = readFlowfile(XForcing.rivers[0].Riverflowfile);


	// Overrule whatever may be set in the param file
	XParam.xmax = XForcing.Bathy[0].xmax;
	XParam.ymax = XForcing.Bathy[0].ymax;
	XParam.xo = XForcing.Bathy[0].xo;
	XParam.yo = XForcing.Bathy[0].yo;

	XParam.dx = XForcing.Bathy[0].dx;

	XParam.zsinit = mintopo+0.5;// Had a small amount of water to avoid a huge first step that would surely break the setup
	XParam.endtime = 20.0;

	XParam.outputtimestep = XParam.endtime;

	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	InitialConditions(XParam, XForcing, XModel);

	InitialAdaptation(XParam, XForcing, XModel);


	SetupGPU(XParam, XModel, XForcing, XModel_g);
	T initVol = T(0.0);
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		T delta = calcres(XParam.dx, XModel.blocks.level[ib]);
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < (XParam.blkwidth); ix++)
			{
				int i = memloc(XParam, ix, iy, ib);
				initVol = initVol + T(XModel.evolv.h[i]) * delta * delta;
			}
		}
	}


	MainLoop(XParam, XForcing, XModel, XModel_g);
	
	T TheoryInput = Q * XParam.endtime;


	T SimulatedVolume = T(0.0);
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		T delta = calcres(XParam.dx, XModel.blocks.level[ib]);
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < (XParam.blkwidth); ix++)
			{
				int i = memloc(XParam, ix, iy, ib);
				SimulatedVolume = SimulatedVolume + XModel.evolv.h[i] * delta * delta;
			}
		}
	}

	SimulatedVolume = SimulatedVolume - initVol;

	T error = abs(SimulatedVolume - TheoryInput);

	return error / TheoryInput < 0.05;

}



/*! \fn bool testboundaries(T maxslope)
* \brief	Wraping function for Boundary(Param XParam, T slope, bool bottop, bool flip)
*
* This function test the 3 types of boundaries (0: Wall/1: Neumann/3: non-reflexive)
* and on all orientations
*
*/
template <class T> bool testboundaries(Param XParam,T maxslope)
{
	//T maxslope = 0.45; // the mass conservation is better with smaller slopes 

	bool Wall_B;// , Wall_R, Wall_L, Wall_T;
	//bool ctofA, ctofB, ctofC, ctofD;
	//bool ftocA, ftocB, ftocC, ftocD;


	std::string details;
	int Bound_type;

	
	XParam.GPUDEVICE = 0;
	maxslope = 0.0;
	//Dir = 3;
	Bound_type = -1;
	Wall_B = RiverOnBoundary(XParam, maxslope, 3, Bound_type);
	//Wall_R = RiverOnBoundary(XParam, maxslope, 0, 0);
	//Wall_L = RiverOnBoundary(XParam, maxslope, 1, 0);
	//Wall_T = RiverOnBoundary(XParam, maxslope, 2, 0);
	/*

	if (UnitestA && UnitestB && UnitestC && UnitestD)
	{
		log("River Volume Conservation Test: Uniform mesh: Success");
	}
	else
	{
		log("River Volume Conservation Test: Uniform mesh: Failed");
		details = UnitestA ? "successful" : "failed";
		log("\t Uniform mesh A :" + details);
		details = UnitestB ? "successful" : "failed";
		log("\t Uniform mesh B :" + details);
		details = UnitestC ? "successful" : "failed";
		log("\t Uniform mesh C :" + details);
		details = UnitestD ? "successful" : "failed";
		log("\t Uniform mesh D :" + details);
	}

	XParam.minlevel = 0;
	XParam.maxlevel = 1;
	XParam.initlevel = 0;

	//Fine to coarse
	// Change arg 1 and 2 if the slope is changed
	XParam.AdaptCrit = "Inrange";
	XParam.Adapt_arg1 = "28.0";
	XParam.Adapt_arg2 = "40.0";
	XParam.Adapt_arg3 = "zb";

	ftocA = RiverVolumeAdapt(XParam, maxslope, false, false);
	ftocB = RiverVolumeAdapt(XParam, maxslope, true, false);
	ftocC = RiverVolumeAdapt(XParam, maxslope, false, true);
	ftocD = RiverVolumeAdapt(XParam, maxslope, true, true);
	if (ftocA && ftocB && ftocC && ftocD)
	{
		log("River Volume Conservation Test: Flow from fine to coarse adapted mesh: Success");
	}
	else
	{
		log("River Volume Conservation Test: Flow from fine to coarse adapted mesh: Failed");
		details = ftocA ? "successful" : "failed";
		log("\t Flow from fine to coarse adapted mesh A :" + details);
		details = ftocB ? "successful" : "failed";
		log("\t Flow from fine to coarse adapted mesh B :" + details);
		details = ftocC ? "successful" : "failed";
		log("\t Flow from fine to coarse adapted mesh C :" + details);
		details = ftocD ? "successful" : "failed";
		log("\t Flow from fine to coarse adapted mesh D :" + details);
	}

	//coarse to fine
	// Change arg 1 and 2 if the slope is changed
	XParam.AdaptCrit = "Inrange";
	XParam.Adapt_arg1 = "0.0";
	XParam.Adapt_arg2 = "2.0";
	XParam.Adapt_arg3 = "zb";

	ctofA = RiverVolumeAdapt(XParam, maxslope, false, false);
	ctofB = RiverVolumeAdapt(XParam, maxslope, true, false);
	ctofC = RiverVolumeAdapt(XParam, maxslope, false, true);
	ctofD = RiverVolumeAdapt(XParam, maxslope, true, true);
	if (ctofA && ctofB && ctofC && ctofD)
	{
		log("River Volume Conservation Test: Flow from coarse to fine adapted mesh: Success");
	}
	else
	{
		log("River Volume Conservation Test: Flow from coarse to fine adapted: Failed");
		details = ctofA ? "successful" : "failed";
		log("\t Flow from coarse to fine adapted mesh A :" + details);
		details = ctofB ? "successful" : "failed";
		log("\t Flow from coarse to fine adapted mesh B :" + details);
		details = ctofC ? "successful" : "failed";
		log("\t Flow from coarse to fine adapted mesh C :" + details);
		details = ctofD ? "successful" : "failed";
		log("\t Flow from coarse to fine adapted mesh D :" + details);
	}*/

	//return (UnitestA * UnitestB * UnitestC * UnitestD * ctofA * ctofB * ctofC * ctofD * ftocA * ftocB * ftocC * ftocD);
	return(Wall_B);
}


/*! \fn bool RiverOnBoundary(T slope, bool bottop, bool flip)
* \brief	Simulate a river flowing in a (steep) valley
* and check the Volume conservation
*
* This function creates a half dry steep valley topography to a given level and run the model for a while and checks that the Volume matches the theory.
* A wall is located in the center of the valley.
*
* The function can test the water volume for 4 scenario each time:
* * flowing to the right: Dir=0;
* * flowing to the left: Dir=1;
* * flowing to the top: Dir=2;
* * flowing to the bottom: Dir=3;
*
*/
template <class T> bool RiverOnBoundary(Param XParam,T slope, int Dir, int Bound_type)
{
	//bool test = true;
	// Make a Parabolic bathy

	//Param XParam;
	XParam.GPUDEVICE = -1;

	auto modeltype = XParam.doubleprecision < 1 ? float() : double();
	Model<decltype(modeltype)> XModel; // For CPU pointers
	Model<decltype(modeltype)> XModel_g; // For GPU pointers

	Forcing<float> XForcing;

	StaticForcingP<float> bathy;

	float* dummybathy;

	//Boundary conditions
	XForcing.top.type = 0;
	XForcing.bot.type = 0;
	XForcing.right.type = 0;
	XForcing.left.type = 0;

	//Physical wall boundary condition
	bool PhysWall = 0;
	if (Bound_type == -1)
	{
		PhysWall = 1;
		Bound_type = 0;
	}

	if (Dir == 0) //To right
	{
		XForcing.right.type = Bound_type;
		XForcing.top.type = 0;
	}
	else if (Dir == 1) //To left
	{
		XForcing.left.type = Bound_type;
		XForcing.bot.type = 0;
	}
	else if (Dir == 2) //To top
	{
		XForcing.top.type = Bound_type;
		XForcing.left.type = 0;
	}
	else if (Dir == 3) //To bottom
	{
		XForcing.bot.type = Bound_type;
		XForcing.right.type = 0;
	}

	XForcing.Bathy.push_back(bathy);

	XForcing.Bathy[0].xo = 0.0;
	XForcing.Bathy[0].yo = 0.0;
	XForcing.Bathy[0].xmax = 31.0;
	XForcing.Bathy[0].ymax = 31.0;
	XForcing.Bathy[0].nx = 32;
	XForcing.Bathy[0].ny = 32;

	XForcing.Bathy[0].dx = 1.0;

	T x, y;
	T center = T(31.0);

	AllocateCPU(1, 1, XForcing.left.blks, XForcing.right.blks, XForcing.top.blks, XForcing.bot.blks);

	AllocateCPU(XForcing.Bathy[0].nx, XForcing.Bathy[0].ny, XForcing.Bathy[0].val);
	AllocateCPU(XForcing.Bathy[0].nx, XForcing.Bathy[0].ny, dummybathy);


	//float maxtopo = std::numeric_limits<float>::min();
	float mintopo = 1000000000000.0f;
	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			x = T(XForcing.Bathy[0].xo + i * XForcing.Bathy[0].dx);
			y = T(XForcing.Bathy[0].yo + j * XForcing.Bathy[0].dx);


			dummybathy[i + j * XForcing.Bathy[0].nx] = float(ValleyBathy(y, x, slope, center));

			//Add physical walls
			if (PhysWall == 1)
			{
				//if (j < 3)
				//{
				//	dummybathy[i + j * XForcing.Bathy[0].nx] = 100.0;
				//}
				if (j > XForcing.Bathy[0].ny - 3)
				{
					dummybathy[i + j * XForcing.Bathy[0].nx] = 100.0;
				}
				if (i > XForcing.Bathy[0].nx - 3)
				{
					dummybathy[i + j * XForcing.Bathy[0].nx] = 100.0;
				}
				if (i < 17)
				{
					dummybathy[i + j * XForcing.Bathy[0].nx] = 1000.0;
				}
			}

			mintopo = utils::min(dummybathy[i + j * XForcing.Bathy[0].nx], mintopo);
			//maxtopo = max(dummybathy[i + j * XForcing.Bathy[0].nx], maxtopo);

		}
	}

	// Flip or rotate the bathy according to what is requested
	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			if (Dir == 1) //left wise
			{
				XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = dummybathy[i + j * XForcing.Bathy[0].nx];
			}
			else if (Dir == 0) //right wise
			{
				XForcing.Bathy[0].val[(XForcing.Bathy[0].nx - 1 - i) + j * XForcing.Bathy[0].nx] = dummybathy[i + j * XForcing.Bathy[0].nx];
			}
			else if (Dir == 3) //bottom wise
			{
				XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = dummybathy[j + i * XForcing.Bathy[0].nx];
			}
			else if (Dir == 2) //top wise
			{
				XForcing.Bathy[0].val[i + (XForcing.Bathy[0].ny - 1 - j) * XForcing.Bathy[0].nx] = dummybathy[j + i * XForcing.Bathy[0].nx];
			}
		}
	}

	free(dummybathy);

	// Overrule whatever is set in the river forcing
	T Q = T(1.0);

	double riverx = (Dir == 0 | Dir == 2)? 6.0 : 25.0; //Dir=1 =>leftward
	double rivery = (Dir == 2 | Dir == 1)? 6.0 : 25.0; //Dir=2 =>topward
	
	//Create a temporary file with river fluxes
	std::ofstream river_file(
		"testriver.tmp", std::ios_base::out | std::ios_base::trunc);
	river_file << "0.0 " + std::to_string(Q) << std::endl;
	river_file << "3600.0 " + std::to_string(Q) << std::endl;
	river_file.close(); //destructor implicitly does it

	River thisriver;
	thisriver.Riverflowfile = "testriver.tmp";
	thisriver.xstart = riverx - 1.0;
	thisriver.xend = riverx + 1.0;
	thisriver.ystart = rivery - 1.0;
	thisriver.yend = rivery + 1.0;

	XForcing.rivers.push_back(thisriver);


	XForcing.rivers[0].flowinput = readFlowfile(XForcing.rivers[0].Riverflowfile);


	// Overrule whatever may be set in the param file
	XParam.xmax = XForcing.Bathy[0].xmax;
	XParam.ymax = XForcing.Bathy[0].ymax;
	XParam.xo = XForcing.Bathy[0].xo;
	XParam.yo = XForcing.Bathy[0].yo;

	XParam.dx = XForcing.Bathy[0].dx;

	XParam.zsinit = mintopo + 0.5;// Had a small amount of water to avoid a huge first step that would surely break the setup
	//XParam.zsoffset = 0.2;
	XParam.endtime = 50.0;
	XParam.dtinit = 0.1;
	XParam.mask = 999.0;
	XParam.outishift = 0;
	XParam.outjshift = 0;


	XParam.outputtimestep = 10.0;// XParam.endtime;

	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	InitialConditions(XParam, XForcing, XModel);

	InitialAdaptation(XParam, XForcing, XModel);

	InitSave2Netcdf(XParam, XModel);

	SetupGPU(XParam, XModel, XForcing, XModel_g);
	T initVol = T(0.0);
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		T delta = calcres(XParam.dx, XModel.blocks.level[ib]);
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < (XParam.blkwidth); ix++)
			{
				int i = memloc(XParam, ix, iy, ib);
				initVol = initVol + XModel.evolv.h[i] * delta * delta;
			}
		}
	}


	MainLoop(XParam, XForcing, XModel, XModel_g);

	T TheoryInput = Q * XParam.endtime;


	T SimulatedVolume = T(0.0);
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		T delta = calcres(XParam.dx, XModel.blocks.level[ib]);
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < (XParam.blkwidth); ix++)
			{
				int i = memloc(XParam, ix, iy, ib);
				SimulatedVolume = SimulatedVolume + XModel.evolv.h[i] * delta * delta;
			}
		}
	}


	printf("End Volume : %f \n", SimulatedVolume);
	printf("Init Volume : %f \n", initVol);

	SimulatedVolume = SimulatedVolume - initVol;

	printf("End Volume - Init volume : %f \n", SimulatedVolume);

	T error = abs(SimulatedVolume - TheoryInput);

	printf("error : %f \n", error);
	printf("Theory input : %f \n", TheoryInput);
	printf("return : %f \n", (error/TheoryInput));


	return error / TheoryInput < 0.01;

}



/*! \fn bool LakeAtRest(Param XParam, Model<T> XModel)
*	This function simulates the first predictive step and check whether the lake at rest is preserved
*	otherwise it prints out to screen the cells (and neighbour) where the test fails
*/
template <class T> bool LakeAtRest(Param XParam, Model<T> XModel)
{
	T epsi = T(1e-5);
	int ib;

	bool test = true;


	Loop<T> XLoop = InitLoop(XParam, XModel);

	fillHaloC(XParam, XModel.blocks, XModel.zb);

	gradientC(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
	gradientHalo(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);

	refine_linear(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
	gradientHalo(XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
	



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
	//updateKurgXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	UpdateButtingerXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);

	// Y- direction
	//updateKurgYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	UpdateButtingerYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax, XModel.zb);
	//AddSlopeSourceYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.zb);

	//============================================
	// Fill Halo for flux from fine to coarse
	fillHalo(XParam, XModel.blocks, XModel.flux);

	// Do we need to check also before fill halo part?

	// Check Fhu and Fhv (they should be zero)
	int i, iright;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XModel.blocks.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < (XParam.blkwidth); ix++)
			{
				i = memloc(XParam, ix, iy, ib);
				iright = memloc(XParam, ix + 1, iy, ib);
				//ileft = memloc(XParam, ix - 1, iy, ib);
				//itop = memloc(XParam, ix, iy + 1, ib);
				//ibot = memloc(XParam, ix, iy - 1, ib);

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


					printf("Fhu[i]=%f\n", XModel.flux.Fhu[i]);

					printf("Fqux[i]=%f; Su[iright]=%f; Diff=%f \n",XModel.flux.Fqux[i], XModel.flux.Su[iright], (XModel.flux.Fqux[i] - XModel.flux.Su[iright]));

					printf(" At i: (ib=%d; ix=%d; iy=%d)\n", ib,ix,iy);
					testButtingerX(XParam, ib, ix, iy, XModel);

					printf(" At iright: (ib=%d; ix=%d; iy=%d)\n", ib, ix+1, iy);
					testButtingerX(XParam, ib, ix+1, iy, XModel);

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

		creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, XModel.blocks.outZone[0]);
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "blockID", 3, XModel.flux.Fhu, XModel.blocks.outZone[0]);

		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "LeftBot", 3, XModel.grad.dhdx, XModel.blocks.outZone[0]);
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "LeftTop", 3, XModel.grad.dhdy, XModel.blocks.outZone[0]);

		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "TopLeft", 3, XModel.grad.dzsdx, XModel.blocks.outZone[0]);
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "TopRight", 3, XModel.grad.dzsdy, XModel.blocks.outZone[0]);

		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "RightTop", 3, XModel.grad.dudx, XModel.blocks.outZone[0]);
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "RightBot", 3, XModel.grad.dudy, XModel.blocks.outZone[0]);

		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "BotLeft", 3, XModel.grad.dvdx, XModel.blocks.outZone[0]);
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "BotRight", 3, XModel.grad.dvdy, XModel.blocks.outZone[0]);
	}

	return test;
}


/*! \fn  void testButtingerX(Param XParam, int ib, int ix, int iy, Model<T> XModel)
*
* This function goes through the Buttinger scheme but instead of the normal output just prints all teh usefull values
* This function is/was used in the lake-at-rest verification
*
* See also: void testkurganovX(Param XParam, int ib, int ix, int iy, Model<T> XModel)
*/
template <class T> void testButtingerX(Param XParam, int ib, int ix, int iy, Model<T> XModel)
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

	
	T cm = T(1.0);
	T fmu = T(1.0);

	T hi = XModel.evolv.h[i];

	T hn = XModel.evolv.h[ileft];


	//if (hi > eps || hn > eps)
	{
		T dx, zi, zn, hr, hl, etar, etal, zr, zl, zA, zCN, hCNr, hCNl;
		T ui, vi, uli, vli, dhdxi, dhdxil, dudxi, dudxil, dvdxi, dvdxil;

		T ga = g * T(0.5);
		// along X
		dx = delta * T(0.5);
		zi = XModel.zb[i];
		zn = XModel.zb[ileft];

		ui = XModel.evolv.u[i];
		vi = XModel.evolv.v[i];
		uli = XModel.evolv.u[ileft];
		vli = XModel.evolv.v[ileft];

		dhdxi = XModel.grad.dhdx[i];
		dhdxil = XModel.grad.dhdx[ileft];
		dudxi = XModel.grad.dudx[i];
		dudxil = XModel.grad.dudx[ileft];
		dvdxi = XModel.grad.dvdx[i];
		dvdxil = XModel.grad.dvdx[ileft];


		hr = hi - dx * dhdxi;
		hl = hn + dx * dhdxil;
		etar = XModel.evolv.zs[i] - dx * XModel.grad.dzsdx[i];
		etal = XModel.evolv.zs[ileft] + dx * XModel.grad.dzsdx[ileft];

		//define the topography term at the interfaces
		zr = zi - dx * XModel.grad.dzbdx[i];
		zl = zn + dx * XModel.grad.dzbdx[ileft];

		//define the Audusse terms
		zA = max(zr, zl);

		// Now the CN terms
		zCN = min(zA, min(etal, etar));
		hCNr = max(T(0.0), min(etar - zCN, hr));
		hCNl = max(T(0.0), min(etal - zCN, hl));

		//Velocity reconstruction
		//To avoid high velocities near dry cells, we reconstruct velocities according to Bouchut.
		T ul, ur, vl, vr, sl, sr;
		if (hi > eps) {
			ur = ui - (T(1.) + dx * dhdxi / hi) * dx * dudxi;
			vr = vi - (T(1.) + dx * dhdxi / hi) * dx * dvdxi;
		}
		else {
			ur = ui - dx * dudxi;
			vr = vi - dx * dvdxi;
		}
		if (hn > eps) {
			ul = uli + (T(1.) - dx * dhdxil / hn) * dx * dudxil;
			vl = vli + (T(1.) - dx * dhdxil / hn) * dx * dvdxil;
		}
		else {
			ul = uli + dx * dudxil;
			vl = vli + dx * dvdxil;
		}




		T fh, fu, fv, dt;


		//solver below also modifies fh and fu
		dt = hllc(g, delta, epsi, CFL, cm, fmu, hCNl, hCNr, ul, ur, fh, fu);
		//hllc(T g, T delta, T epsi, T CFL, T cm, T fm, T hm, T hp, T um, T up, T & fh, T & fq)

		

		fv = (fh > 0. ? vl : vr) * fh;


		// Topographic source term

		// In the case of adaptive refinement, care must be taken to ensure
		// well-balancing at coarse/fine faces (see [notes/balanced.tm]()). 
		if ((ix == XParam.blkwidth) && levRB < lev)//(ix==16) i.e. in the right halo
		{
			int jj = LBRB == ib ? ftoi(floor(iy * (T)0.5)) : ftoi(floor(iy * (T)0.5) + XParam.blkwidth / 2);
			int iright = memloc(XParam.halowidth, XParam.blkmemwidth, 0, jj, RB);;
			hi = XModel.evolv.h[iright];
			zi = XModel.zb[iright];
		}
		if ((ix == 0) && levLB < lev)//(ix==16) i.e. in the right halo if you 
		{
			int jj = RBLB == ib ? ftoi(floor(iy * (T)0.5)) : ftoi(floor(iy * (T)0.5) + XParam.blkwidth / 2);
			int ilc = memloc(XParam.halowidth, XParam.blkmemwidth, XParam.blkwidth - 1, jj, LB);
			//int ilc = memloc(halowidth, blkmemwidth, -1, iy, ib);
			hn = XModel.evolv.h[ilc];
			zn = XModel.zb[ilc];
		}

		sl = ga * (hi + hCNr) * (zi - zCN);
		sr = ga * (hCNl + hn) * (zn - zCN);


		printf("dt=%f; etar=%f; etal=%f; zCN=%f; zi=%f; zn=%f; zA=%f, zr=%f, zl=%f\n",dt, etar,etal,zCN,zi,zn,zA, zr,zl);


		printf("hi=%f; hn=%f,fh=%f; fu=%f; sl=%f; sr=%f; hCNl=%f; hCNr=%f; hr=%f; hl=%f; zr=%f; zl=%f;\n", hi, hn, fh, fu, sl, sr, hCNl, hCNr, hr, hl, zr, zl);

		printf("h[i]=%f; h[ileft]=%f dhdx[i]=%f, dhdx[ileft]=%f, zs[i]=%f, zs[ileft]=%f, dzsdx[i]=%f, dzsdx[ileft]=%f, dzbdx[i]=%f, dzbdx[ileft]=%f\n\n", XModel.evolv.h[i], XModel.evolv.h[ileft], XModel.grad.dhdx[i], XModel.grad.dhdx[ileft], XModel.evolv.zs[i], XModel.evolv.zs[ileft], XModel.grad.dzsdx[i], XModel.grad.dzsdx[ileft], XModel.grad.dzbdx[i], XModel.grad.dzbdx[ileft]);
	}
}


/*! \fn  void testkurganovX(Param XParam, int ib, int ix, int iy, Model<T> XModel)
*
* This function goes through the Kurganov scheme but instead of the normal output just prints all teh usefull values
* This function is/was used in the lake-at-rest verification
*/
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

	printf("hi=%f; hn=%f,fh=%f; fu=%f; sl=%f; sr=%f; hp=%f; hm=%f; hr=%f; hl=%f; zr=%f; zl=%f;\n", hi, hn, fh, fu, sl, sr, hp, hm, hr, hl, zr, zl);

	printf("h[i]=%f; h[ileft]=%f dhdx[i]=%f, dhdx[ileft]=%f, zs[i]=%f, zs[ileft]=%f, dzsdx[i]=%f, dzsdx[ileft]=%f\n", XModel.evolv.h[i], XModel.evolv.h[ileft], XModel.grad.dhdx[i], XModel.grad.dhdx[ileft], XModel.evolv.zs[i], XModel.evolv.zs[ileft], XModel.grad.dzsdx[i], XModel.grad.dzsdx[ileft]);

}

/*! \fn bool Raintest(T zsnit, int gpu, float alpha)
*
* This function tests the mass conservation of the spacial injection (used to model rain on grid)
*	The function creates its own model setup and mesh independantly to what the user inputs.
*	This starts with a initial water level (zsnit=0.0 is dry) and runs for 0.1s before comparing results
*	with zsnit=0.1 that is approx 20 steps
*/
template <class T> bool Raintest(T zsnit, int gpu, float alpha)
{
	log("#####");
	Param XParam;
	T initVol, TheoryInput;
	TheoryInput = T(0.0);
	// initialise domain and required resolution
	XParam.dx = 1.0 / ((1 << 6)); //1<<8  = 2^8
	XParam.xo = -0.5;
	XParam.yo = -0.5;
	XParam.xmax = 0.5;
	XParam.ymax = 0.5;

	//XParam.initlevel = 0;
	//XParam.minlevel = 0;
	//XParam.maxlevel = 0;

	XParam.zsinit = zsnit;
	//XParam.zsoffset = 0.0;

	//Output times for comparisons
	XParam.endtime = 1.0;
	XParam.outputtimestep = 0.1;

	XParam.smallnc = 0;

	XParam.cf = 0.01;
	XParam.frictionmodel = 0;

	//Specification of the test
	//XParam.test = 7;
	XParam.rainforcing = true;

	// Enforce GPU/CPU
	XParam.GPUDEVICE = gpu;
	XParam.rainbnd = true;
	//output vars
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
			XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = i * alpha / 100;
		}
	}

	// Add wall boundary conditions
	XForcing.right.type = 0;
	XForcing.left.type = 0;
	XForcing.top.type = 0;
	XForcing.bot.type = 0;


	//Value definition for surface rain fall
	T Q = 300; // mm/hr
	std::cout << "# Theoretical volume of water input during the simulation in m3: " << TheoryInput << ", from a rain input of: " << Q << "mm/hr." << std::endl;
	//Create a temporary file with rain fluxes
	std::ofstream rain_file(
		"testrain.tmp", std::ios_base::out | std::ios_base::trunc);
	rain_file << "0.0 " + std::to_string(Q) << std::endl;
	rain_file << "3600.0 " + std::to_string(Q) << std::endl;
	rain_file.close(); //destructor implicitly does it

	XForcing.Rain.inputfile = "testrain.tmp";
	XForcing.Rain.uniform = true;

	// Reading rain forcing from file for CPU and unifor rain
	XForcing.Rain.unidata = readINfileUNI(XForcing.Rain.inputfile);

	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	InitialConditions(XParam, XForcing, XModel);

	InitialAdaptation(XParam, XForcing, XModel);

	SetupGPU(XParam, XModel, XForcing, XModel_g);
	initVol = T(0.0);
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		T delta = calcres(XParam.dx, XModel.blocks.level[ib]);
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < (XParam.blkwidth); ix++)
			{
				int i = memloc(XParam, ix, iy, ib);
				initVol = initVol + XModel.evolv.h[i] * delta * delta;
			}
		}
	}


	MainLoop(XParam, XForcing, XModel, XModel_g);

	TheoryInput = Q/ T(1000.0) / T(3600.0) * XParam.endtime;


	T SimulatedVolume = T(0.0);
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		T delta = calcres(XParam.dx, XModel.blocks.level[ib]);
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < (XParam.blkwidth); ix++)
			{
				int i = memloc(XParam, ix, iy, ib);
				SimulatedVolume = SimulatedVolume + XModel.evolv.h[i] * delta * delta;
			}
		}
	}

	SimulatedVolume = SimulatedVolume - initVol;

	T error = abs(SimulatedVolume - TheoryInput);

	T modelgood= error / TheoryInput < 0.05;

	//log("#####");
	return modelgood;
}


/*! \fn bool Raintestinput(int gpu)
*
* This function tests the different inputs for rain forcing.
* This test is based on the paper Aureli2020, the 3 slopes test
* with regional rain. The experiment has been presented in Iwagaki1955.
* The first test compares a time varying rain input using a uniform time serie 
* forcing and a time varying 2D field (with same value).
* The second test check the 3D rain forcing (comparing it to expected values).
*/
bool Raintestinput(int gpu)
{
	//Results of the experiment of Aureli, interpolated to output values
	bool modelgood1, modelgood2;
	std::string result;
	//int dim_flux;
	std::vector<float> Flux1D, Flux3DUni, Flux3D, Flux_obs;
	float diff, ref, error;
	
	
	//Comparison between the 1D forcing and the 3D hommgeneous forcing
	Flux1D = Raintestmap(gpu, 1, -0.03);
	Flux3DUni = Raintestmap(gpu, 31, -0.03);
	ref = 0.0;
	diff = 0.0;
	for (int i = 0; i < Flux1D.size(); i++)
	{
		diff = diff + Flux1D[i] - Flux3DUni[i];
		ref = ref + Flux1D[i];
	}

	error = abs(diff / ref);
	printf("Error %f \n", error);

	modelgood1 = error < 0.005;
	result = modelgood1 ? "successful" : "failed";
	log("\t\tRain test input 1D vs 3D homogeneous: " + result);

	//Comparison between the 3D forcing and the observations from Iwagaki1955.

	//From Observations
	//Flux_obs = { 1.75136262,  4.31856716, 24.36350225, 32.02235696, 32.41207121,
	//   31.68632601, 29.8140878 , 47.9632521 , 68.78608061, 57.03656989 };

	//From BG_run of the testcase
	Flux_obs = { 4.003079, 12.664897, 25.376514, 33.214722, 34.987427, 34.054474,
		32.696472, 30.718161, 89.497993, 58.156021 };

	Flux3D = Raintestmap(gpu, 3, -0.03);
	ref = 0.0;
	diff = 0.0;
	for (int i = 0; i < Flux3D.size(); i++)
	{
		diff = diff + Flux_obs[i] - Flux3D[i];
		ref = ref + Flux3D[i];
	}

	error = abs(diff / ref);
	printf("Error %f \n", error);

	modelgood2 = error < 0.00005;
	result = modelgood2 ? "successful" : "failed";
	log("\t\tRain test input 3D map vs Iwagaki1955: " + result);

	return (modelgood1 * modelgood2);
}

/*! \fnstd::vector<float> Raintestmap(int gpu, int dimf, T zinit)
*
* This function return the flux at the bottom of the 3 part slope
* for different types of rain forcings using the test case based on Iwagaki1955
*/
template <class T> std::vector<float> Raintestmap(int gpu, int dimf, T zinit)
{
	log("#####");
	int k;
	T rainDuration = 10.0;
	int NX = 2502;
	int NY = 22;
	int NT;
	double* xRain;
	double* yRain;
	double* tRain;
	double* rainForcing;
	

	Param XParam;
	T delta;

	// initialise domain and required resolution
	XParam.xo = 0;
	XParam.yo = 0;
	XParam.ymax = 0.196;
	XParam.dx=(XParam.ymax - XParam.yo) / (1 << 1);
	double Xmax_exp = 28.0; //minimum Xmax position (adjust to have a "full blocks" config)
	//Calculating xmax to have full blocs with at least a full block behaving as a reservoir
	XParam.xmax = XParam.xo + (16 * XParam.dx) * std::ceil((Xmax_exp - XParam.xo) / (16 * XParam.dx)) + (16 * XParam.dx);
	//Surf = T((XParam.xmax - XParam.xo) * (XParam.ymax - XParam.yo));
	XParam.nblk = ftoi(((XParam.xmax - XParam.xo) / XParam.dx / 16) * ((XParam.ymax - XParam.yo) / XParam.dx / 16));
	XParam.rainbnd = true;
	XParam.zsinit = zinit;

	//Output times for comparisons
	XParam.endtime = 30.0;
	XParam.outputtimestep = 3.0;

	XParam.smallnc = 0;

	//Specification of the test
	XParam.test = 8;

	// Enforce GPU/CPU
	XParam.GPUDEVICE = gpu;

	//Bottom friction
	XParam.frictionmodel = -1; //Manning model
	XParam.cf = 0.009; //n in m^(-1/3)s

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
	XForcing.Bathy[0].xmax = 28.0;
	XForcing.Bathy[0].ymax = 1.0;
	XForcing.Bathy[0].dx = 0.1;
	XForcing.Bathy[0].nx = ftoi((XForcing.Bathy[0].xmax - XForcing.Bathy[0].xo) / XForcing.Bathy[0].dx + 1);
	XForcing.Bathy[0].ny = ftoi((XForcing.Bathy[0].ymax - XForcing.Bathy[0].yo) / XForcing.Bathy[0].dx + 1);


	AllocateCPU(1, 1, XForcing.left.blks, XForcing.right.blks, XForcing.top.blks, XForcing.bot.blks);

	AllocateCPU(XForcing.Bathy[0].nx, XForcing.Bathy[0].ny, XForcing.Bathy[0].val);

	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = -10.0;
			if (i < (9 / XForcing.Bathy[0].dx + 1))
			{
				XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = T(0.2 + (9.0 - i * XForcing.Bathy[0].dx) * 2.0 / 100.0);
			}
			else if (i < (17 / XForcing.Bathy[0].dx + 1))
			{
				XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = T(0.08 + (17.0 - i * XForcing.Bathy[0].dx) * 1.5 / 100.0);
			}
			else if (i < (25 / XForcing.Bathy[0].dx + 1))
			{
				XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = T(0.0 + (25.0 - i * XForcing.Bathy[0].dx) * 1.0 / 100.0);
			}
		}
	}

	// Add wall boundary conditions but at the bottom of the slope
	//XForcing.right.type = 0;
	XForcing.left.type = 0;
	//XForcing.top.type = 0;
	//XForcing.bot.type = 0;

	//Value definition for surface rain fall
	T r1 = T(3888.0); // mm/hr
	T r2 = T(2296.8); //mm/hr
	T r3 = T(2880.0); //mm/hr
	T Q = (r1 + r2 + r3) / 3;
	//TheoryInput = Q * XParam.outputtimestep / T(1000.0) / T(3600.0) * Surf; //m3/s
	//printf("# Theoretical volume of water input during the simulation in m3: %f , from a mean rain input of: %f mm/hr.\n", TheoryInput, Q);
	double eps = 0.0001;

	// Create the rain forcing file
	if (dimf == 1)
	{
		//Create a temporary file with rain fluxes for uniform rain
		std::ofstream rain_file(
			"testrain.tmp", std::ios_base::out | std::ios_base::trunc);
		rain_file << "0.0 " + std::to_string(Q) << std::endl;
		rain_file << std::to_string(rainDuration) + " " + std::to_string(Q) << std::endl;
		rain_file << std::to_string(rainDuration + eps) + " 0.0" << std::endl;
		rain_file << std::to_string(rainDuration + 360000) + " 0.0" << std::endl;
		rain_file.close(); //destructor implicitly does it

		XForcing.Rain.inputfile = "testrain.tmp";
		XForcing.Rain.uniform = true;

		// Reading rain forcing from file for CPU and uniform rain
		XForcing.Rain.unidata = readINfileUNI(XForcing.Rain.inputfile);
		printf("1D rain forcing read\n");
	}
	else //non-uniform forcing
	{
		XForcing.Rain.uniform = false;

		//X Y variables

		xRain = (double*)malloc(sizeof(double) * NX);
		yRain = (double*)malloc(sizeof(double) * NY);

		for (int i = 0; i < NX; i++) { xRain[i] = -0.005 + 0.01 * i; }
		for (int j = 0; j < NY; j++) { yRain[j] = -0.01 + 0.01 * j; }

		NT = 601;
		tRain = (double*)malloc(sizeof(double) * NT);
		for (int tt = 0; tt < NT; tt++) { tRain[tt] = XParam.endtime / (NT - 1) * tt; }

		rainForcing = (double*)malloc(sizeof(double) * NT * NY * NX);

		//Create a non-uniform time-variable rain forcing
		if (dimf == 3)
		{
			//Create the rain forcing:
			for (k = 0; k < NT; k++)
			{
				for (int j = 0; j < NY; j++)
				{
					for (int i = 0; i < NX; i++)
					{
						if (tRain[k] < rainDuration+eps)
						{
							if (xRain[i] < 8.0)
							{
								rainForcing[k * (NX * NY) + j * NX + i] = r1;
							}
							else if (xRain[i] < 16.0)
							{
								rainForcing[k * (NX * NY) + j * NX + i] = r2;
							}
							else
							{
								rainForcing[k * (NX * NY) + j * NX + i] = r3;
							}
						}
						else
						{
							rainForcing[k * (NX * NY) + i * NY + j] = 0.0;
						}
					}
				}
			}

			//Write the netcdf file
			create3dnc("rainTemp.nc", NX, NY, NT, xRain, yRain, tRain, rainForcing, "myrainforcing");

			printf("non-uniform forcing\n");

			//End creation of the nc file for rain forcing
		}
		//Create a uniform time-variable rain forcing using a map forcing (nc file)
		else if (dimf == 31)
		{
			//Create the rain forcing:
			for (k = 0; k < NT; k++)
			{
				for (int j = 0; j < NY; j++)
				{
					for (int i = 0; i < NX; i++)
					{
						if (tRain[k] < rainDuration + eps)
						{
							rainForcing[k * (NX * NY) + j * NX + i] = Q;
						}
						else
						{
							rainForcing[k * (NX * NY) + i * NY + j] = 0.0;
						}
					}
				}
			}

			//Write the netcdf file
			create3dnc("rainTemp.nc", NX, NY, NT, xRain, yRain, tRain, rainForcing, "myrainforcing");

			printf("non-uniform forcing 31\n");
			//End creation of the nc file for rain forcing
		}
		/*
		//2D forcing (map without time variation is not working)
		else if (dimf == 2)//dimf==2 for rain forcing 
		{

			//Create a non-uniform time-constant rain forcing 
			rainForcing = (double*)malloc(sizeof(double) * NY * NX);

			//Create the rain forcing:

			for (int j = 0; j < NY; j++)
			{
				for (int i = 0; i < NX; i++)
				{

					if (xRain[i] < 8.0)
					{
						rainForcing[j * NX + i] = r1;
					}
					else if (xRain[i] < 16.0)
					{
						rainForcing[j * NX + i] = r2;
					}
					else
					{
						rainForcing[j * NX + i] = r3;
					}

				}
			}

			create2dnc("rainTempt.nc", NX, NY, xRain, yRain, rainForcing, "myrainforcing");

			//End creation of the nc file for rain forcing
		}
		*/
		else { printf("Error in rain forcing dimension (should be in [1,3,31])\n"); }

		//Reading non-unform forcing
		bool gpgpu = 0;
		if (XParam.GPUDEVICE != -1)
		{
			gpgpu = 1;
		}

		XForcing.Rain = readfileinfo("rainTemp.nc", XForcing.Rain);
		XForcing.Rain.uniform = 0;
		XForcing.Rain.varname = "myrainforcing";
		

		InitDynforcing(gpgpu, XParam.totaltime, XForcing.Rain);

		//readDynforcing(gpgpu, XParam.totaltime, XForcing.Rain);


		free(rainForcing);
		free(xRain);
		free(yRain);
		free(tRain);
	}


	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	InitialConditions(XParam, XForcing, XModel);

	InitialAdaptation(XParam, XForcing, XModel);

	SetupGPU(XParam, XModel, XForcing, XModel_g);

	log("Initialising model main loop");

	Loop<T> XLoop = InitLoop(XParam, XModel);

	//Define some useful variables 
	Initmeanmax(XParam, XLoop, XModel, XModel_g);


	log("\t\tCompleted");
	log("Model Running...");
	std::vector<float> Flux;

	while (XLoop.totaltime < XParam.endtime)
	{

		// Calculate Forcing at this step
		updateforcing(XParam, XLoop, XForcing);

		// Core engine
		if (XParam.GPUDEVICE >= 0)
		{
			FlowGPU(XParam, XLoop, XForcing, XModel_g);
		}
		else
		{
			FlowCPU(XParam, XLoop, XForcing, XModel);
		}

		// Time keeping
		XLoop.totaltime = XLoop.totaltime + XLoop.dt;
		//printf("\tTime = %f \n", XLoop.totaltime);

		//if Toutput, calculate the flux at x=24m;


		// Getting the coordinate for the flux calculation
		int bl, ixx, ibl, ix, ib;
		T dist = T(1000000000.0);
		for (ibl = 0; ibl < XParam.nblk; ibl++)
		{
			ib = XModel.blocks.active[ibl];
			delta = calcres(T(XParam.dx), XModel.blocks.level[ib]);
			for (ix = 0; ix < XParam.blkwidth; ix++)
			{
				//n = memloc(XParam, ix, 0, ib);
				if (abs(XModel.blocks.xo[ibl] + ix * delta - 24.0) < dist)
				{
					ixx = ix;
					bl = ibl;
					dist = T(abs(XModel.blocks.xo[ibl] + ix * delta - 24.0));
				}
			}
		}

		if (XLoop.nextoutputtime - XLoop.totaltime <= XLoop.dt * T(0.00001) && XParam.outputtimestep > 0.0)
		{
			T finalFlux = T(0.0);
			if (XParam.GPUDEVICE >= 0)
			{
				for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
				{
					CUDA_CHECK(cudaMemcpy(XModel.OutputVarMap[XParam.outvars[ivar]], XModel_g.OutputVarMap[XParam.outvars[ivar]], XParam.nblkmem * XParam.blksize * sizeof(T), cudaMemcpyDeviceToHost));
				}
			}

			Save2Netcdf(XParam, XLoop, XModel);
			

			//Calculation of the flux at the bottom of the slope (x=24m)
			ib = XModel.blocks.active[bl];
			delta = calcres(T(XParam.dx), XModel.blocks.level[ib]);

			for (int iy = 0; iy < XParam.blkwidth; iy++)
			{
				int n = memloc(XParam, ixx, iy, ib);
				finalFlux = finalFlux + XModel.evolv.h[n] * XModel.evolv.u[n] * delta;
			}
			finalFlux = finalFlux / float(XParam.ymax - XParam.yo)*100.0f*100.0f;
			Flux.push_back(finalFlux);
			XLoop.nextoutputtime = XLoop.nextoutputtime + XParam.outputtimestep;
			printf("\tTime = %f, Flux at bottom end of slope : %f \n", XLoop.totaltime, finalFlux);
		}
	}
	/*
	for (int n = 0; n < Flux.size(); n++)
	{
		printf("Flux at %i : %f \n", n, Flux[n]);
	}
	*/

	return Flux;
}
template std::vector<float> Raintestmap<float>(int gpu, int dimf, float Zsinit);
template std::vector<float> Raintestmap<double>(int gpu, int dimf, double Zsinit);


/*! \fn bool testzoneOutDef = ZoneOutputTest(int nzones, T zsinit)
*
* This function test the zoned output for a basic configuration
*/
template <class T> bool ZoneOutputTest(int nzones, T zsinit)
//template bool ZoneOutputTest<float>(int nzones, float zsinit);
{
	log("#####");

	Param XParam;
	Forcing<float> XForcing; 

	
	if (nzones  == 3)
	{
		// read param file
		readforcing(XParam, XForcing);
		outzoneP zone;
		zone.outname = "whole.nc";
		zone.xstart = -10;
		zone.xend = 10;
		zone.ystart = -10;
		zone.yend = 10;
		XParam.outzone.push_back(zone);
		zone.outname = "zoomed.nc";
		zone.xstart =1;
		zone.xend =2;
		zone.ystart = -2;
		zone.yend = 2;
		XParam.outzone.push_back(zone);
		zone.outname = "zoomed2.nc";
		zone.xstart = -2;
		zone.xend = 2;
		zone.ystart = -4;
		zone.yend = 2;
		XParam.outzone.push_back(zone);
	}

	// initialise domain and required resolution
	XParam.dx = 1.0 / ((1 << 6)); //1<<8  = 2^8
	XParam.xo = -5;
	XParam.yo = -5;
	XParam.xmax = 5;
	XParam.ymax = 5;

	XParam.initlevel = 0;
	XParam.minlevel = -1;
	XParam.maxlevel = 1;

	XParam.zsinit = zsinit;
	//XParam.zsoffset = 0.0;

	//Output times for comparisons
	XParam.endtime = 1.0;
	XParam.outputtimestep = 0.5;

	XParam.smallnc = 0;

	XParam.cf = 0.0001;
	XParam.frictionmodel = 1;

	//Specification of the test
	//XParam.test = 7;
	XParam.rainforcing = true;

	// Enforce GPU/CPU
	//XParam.GPUDEVICE = gpu;
	//XParam.rainbnd = true;

	// create Model setup
	Model<T> XModel;
	Model<T> XModel_g;

	StaticForcingP<float> bathy;

	XForcing.Bathy.push_back(bathy);

	// initialise forcing bathymetry to a central hill
	XForcing.Bathy[0].xo = -10.0;
	XForcing.Bathy[0].yo = -10.0;
	XForcing.Bathy[0].xmax = 10.0;
	XForcing.Bathy[0].ymax = 10.0;
	XForcing.Bathy[0].nx = 501;
	XForcing.Bathy[0].ny = 501;

	XForcing.Bathy[0].dx = 0.1;

	AllocateCPU(1, 1, XForcing.left.blks, XForcing.right.blks, XForcing.top.blks, XForcing.bot.blks);

	AllocateCPU(XForcing.Bathy[0].nx, XForcing.Bathy[0].ny, XForcing.Bathy[0].val);
	
	float rs, x, y, r, hm;
	rs = 20; //hill radio 
	hm = 5; //hill top
	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			x = XForcing.Bathy[0].xo + i * XForcing.Bathy[0].dx;
			y = XForcing.Bathy[0].yo + j * XForcing.Bathy[0].dx;
			r = sqrt(x * x + y * y);
			if (r < rs)
			{
				XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = hm*(1-r/rs);
			}
			if (x < -4.7 | x > 4.7 | y < -4.7 | y > 4.7)
			{
				XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = 10;
			}
		}
	}

	//Adaptation
	XParam.AdaptCrit = "Targetlevel";
	
	StaticForcingP<int> Target;
	XForcing.targetadapt.push_back(Target);

	XForcing.targetadapt[0].xo = -10;
	XForcing.targetadapt[0].yo = -10;
	XForcing.targetadapt[0].xmax = 10.0;
	XForcing.targetadapt[0].ymax = 10.0;
	XForcing.targetadapt[0].nx = 501;
	XForcing.targetadapt[0].ny = 501;

	XForcing.targetadapt[0].dx = 0.1;

	AllocateCPU(XForcing.targetadapt[0].nx, XForcing.targetadapt[0].ny, XForcing.targetadapt[0].val);

	for (int j = 0; j < XForcing.targetadapt[0].ny; j++)
	{
		for (int i = 0; i < XForcing.targetadapt[0].nx; i++)
		{
			x = XForcing.targetadapt[0].xo + i * XForcing.targetadapt[0].dx;
			y = XForcing.targetadapt[0].yo + j * XForcing.targetadapt[0].dx;
			if (x < 0.0)
			{
				XForcing.targetadapt[0].val[i + j * XForcing.targetadapt[0].nx] = -1;
			}
			else
			{
				if (y < 0.0)
				{
					XForcing.targetadapt[0].val[i + j * XForcing.targetadapt[0].nx] = 0;
				}
				else
				{
					XForcing.targetadapt[0].val[i + j * XForcing.targetadapt[0].nx] = 1;
				}
			}
		}
	}

	// Add wall boundary conditions
	XForcing.right.type = 0;
	XForcing.left.type = 0;
	XForcing.top.type = 0;
	XForcing.bot.type = 0;


	//Create a temporary file with river fluxes
	float Q = 1;
	std::ofstream river_file(
		"testriver.tmp", std::ios_base::out | std::ios_base::trunc);
	river_file << "0.0 " + std::to_string(Q) << std::endl;
	river_file << "3600.0 " + std::to_string(Q) << std::endl;
	river_file.close(); //destructor implicitly does it

	River thisriver;
	thisriver.Riverflowfile = "testriver.tmp";
	thisriver.xstart = -0.2;
	thisriver.xend = 0.2;
	thisriver.ystart = -0.2;
	thisriver.yend = 0.2;

	XForcing.rivers.push_back(thisriver);


	XForcing.rivers[0].flowinput = readFlowfile(XForcing.rivers[0].Riverflowfile);


	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	InitialConditions(XParam, XForcing, XModel);

	InitialAdaptation(XParam, XForcing, XModel);

	SetupGPU(XParam, XModel, XForcing, XModel_g);

	MainLoop(XParam, XForcing, XModel, XModel_g);

	//Test if file exist and can be open:
	int error = 1;
	std::vector<int> observedSize{ 473251462,23304761,130802886 };
	for (int o = 0; o < XModel.blocks.outZone.size(); o++)
	{
		std::ifstream fs(XModel.blocks.outZone[o].outname);
		if (fs.fail()) 
		{
			error++;
		}
		else
		{
			//Calculate the size of the file in bytes
			std::ifstream in_file(XModel.blocks.outZone[o].outname, std::ios::binary);
			in_file.seekg(0, std::ios::end);
			int file_size = in_file.tellg();
			printf("sizes : %i in bytes\n", file_size);
			error = error * (observedSize[o] / file_size);
		}
	}

	bool modelgood = (1-abs(error)) < 0.05;

	//log("#####");
	return modelgood;
}
template bool ZoneOutputTest<float>(int nzones, float zsinit);
template bool ZoneOutputTest<double>(int nzones, double zsinit);


/*! \fn bool testzoneOutDef = ZoneOutputTest(int nzones, T zsinit)
*
* This function test the spped and accuracy of a new gradient function
* gradient are only calculated for zb but assigned to different gradient variable for storage
*/
template <class T> int TestGradientSpeed(Param XParam, Model<T> XModel, Model<T> XModel_g)
{
	//
	int fastest = 1;
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	// for flux reconstruction the loop overlap the right(or top for the y direction) halo
	dim3 blockDimX2(XParam.blkwidth + XParam.halowidth*2, XParam.blkwidth + XParam.halowidth * 2, 1);



	// Allocate CUDA events that we'll use for timing
	cudaEvent_t startA, startB, startC, startG, startGnew;
	cudaEvent_t stopA, stopB, stopC, stopG, stopGnew;

	fillHalo(XParam, XModel.blocks, XModel.evolv, XModel.zb);

	std::thread t0(&gradientC<T>, XParam, XModel.blocks, XModel.zb, XModel.grad.dzbdx, XModel.grad.dzbdy);
	t0.join();


	Loop<T> XLoop;
	// GPU stuff


	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();

	XLoop.totaltime = 0.0;

	XLoop.nextoutputtime = 3600.0;


	cudaEventCreate(&startA);

	
	cudaEventCreate(&stopA);

	// Record the start event
	cudaEventRecord(startA, NULL);
	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel_g.blocks.active, XModel_g.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel_g.zb, XModel_g.grad.dzbdx, XModel_g.grad.dzbdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Record the stop event
	cudaEventRecord(stopA, NULL);

	// Wait for the stop event to complete
	cudaEventSynchronize(stopA);

	float msecTotalGrad = 0.0f;
	cudaEventElapsedTime(&msecTotalGrad, startA, stopA);

	cudaEventDestroy(startA);
	cudaEventDestroy(stopA);


	cudaEventCreate(&startB);


	cudaEventCreate(&stopB);

	// Record the start event
	cudaEventRecord(startB, NULL);
	gradientSM << < gridDim, blockDim >> > (XParam.halowidth, XModel_g.blocks.active, XModel_g.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel_g.zb, XModel_g.grad.dzsdx, XModel_g.grad.dzsdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Record the stop event
	cudaEventRecord(stopB, NULL);

	// Wait for the stop event to complete
	cudaEventSynchronize(stopB);

	float msecTotalSM = 0.0f;
	cudaEventElapsedTime(&msecTotalSM, startB, stopB);

	cudaEventDestroy(startB);
	cudaEventDestroy(stopB);


	cudaEventCreate(&startC);


	cudaEventCreate(&stopC);

	// Record the start event
	cudaEventRecord(startC, NULL);
	gradientSMC << < gridDim, blockDim >> > (XParam.halowidth, XModel_g.blocks.active, XModel_g.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel_g.zb, XModel_g.grad.dhdx, XModel_g.grad.dhdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Record the stop event
	cudaEventRecord(stopC, NULL);

	// Wait for the stop event to complete
	cudaEventSynchronize(stopC);

	float msecTotalSMB = 0.0f;
	cudaEventElapsedTime(&msecTotalSMB, startC, stopC);

	cudaEventDestroy(startC);
	cudaEventDestroy(stopC);


	

	CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.grad.dudx, XModel_g.grad.dzbdx);
	CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.grad.dudy, XModel_g.grad.dzbdy);

	CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.grad.dzsdx, XModel_g.grad.dzsdx);
	CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.grad.dzsdy, XModel_g.grad.dzsdy);

	CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.grad.dhdx, XModel_g.grad.dhdx);
	CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.grad.dhdy, XModel_g.grad.dhdy);

	printf("Runtime : normal=%f, shared mem=%f, SharedmemB=%f in msec\n", msecTotalGrad, msecTotalSM, msecTotalSMB);

	/*
	creatncfileBUQ(XParam, XModel.blocks);

	std::vector<std::string> varlist = { "zb", "dzbdx", "dzbdy" };

	for (int ivar = 0; ivar < varlist.size(); ivar++)
	{
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, varlist[ivar], 3, XModel.OutputVarMap[varlist[ivar]], XModel.blocks.outZone[0]);
	}

	diffArray(XParam, XLoop, XModel.blocks, "SMdx", false, XModel.grad.dzbdx, XModel_g.grad.dzsdx, XModel.time.arrmax, XModel.grad.dzsdx);
	

	diffArray(XParam, XLoop, XModel.blocks, "SMBdx", false, XModel.grad.dzbdx, XModel_g.grad.dhdx, XModel.time.arrmax, XModel.grad.dhdx);

	diffArray(XParam, XLoop, XModel.blocks, "SMBdy", false, XModel.grad.dzbdy, XModel_g.grad.dhdy, XModel.time.arrmax, XModel.grad.dhdy);
	diffArray(XParam, XLoop, XModel.blocks, "SMdy", false, XModel.grad.dzbdy, XModel_g.grad.dzsdy, XModel.time.arrmax, XModel.grad.dzsdy);
	*/
	T maxdiffx, maxdiffy;
	maxdiffx = T(0.0);
	maxdiffy = T(0.0);
	T maxdiffsmx, maxdiffsmy;
	maxdiffsmx = T(0.0);
	maxdiffsmy = T(0.0);
	T maxdiffsmbx, maxdiffsmby;
	maxdiffsmbx = T(0.0);
	maxdiffsmby = T(0.0);
	T diffsm, diffsmb;

	

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

				diffsm = abs(XModel.grad.dzbdx[i] - XModel.grad.dudx[i]);

				maxdiffx = max(maxdiffx, diffsm);

				diffsm = abs(XModel.grad.dzbdy[i] - XModel.grad.dudy[i]);

				maxdiffx = max(maxdiffx, diffsm);

				diffsm = abs(XModel.grad.dzbdx[i] - XModel.grad.dzsdx[i]);
				
				maxdiffsmx = max(maxdiffsmx, diffsm);

				diffsm = abs(XModel.grad.dzbdy[i] - XModel.grad.dzsdy[i]);

				maxdiffsmy = max(maxdiffsmy, diffsm);

				diffsm = abs(XModel.grad.dzbdx[i] - XModel.grad.dhdx[i]);
				maxdiffsmbx = max(maxdiffsmbx, diffsm);

				diffsm =  abs(XModel.grad.dzbdy[i] - XModel.grad.dhdy[i]);
				maxdiffsmby = max(maxdiffsmby, diffsm);
				//
			}
		}
	}

	
	printf("max error : normx=%e, normy=%e, smx=%e, smy=%e,  smbx=%e, smby=%e in m\n", maxdiffx, maxdiffy, maxdiffsmx, maxdiffsmy, maxdiffsmbx, maxdiffsmby);


	gradientCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.zb);


	cudaEventCreate(&startG);


	cudaEventCreate(&stopG);

	cudaEventRecord(startG, NULL);
	gradientGPU(XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.grad, XModel_g.zb);
	cudaEventRecord(stopG, NULL);

	// Wait for the stop event to complete
	cudaEventSynchronize(stopG);

	float msecTotalG = 0.0f;
	cudaEventElapsedTime(&msecTotalG, startG, stopG);

	cudaEventDestroy(startG);
	cudaEventDestroy(stopG);

	CompareCPUvsGPU(XParam, XModel, XModel_g, { "dhdx","dhdy", "dzsdx","dzsdy","dudx","dudy","dvdx","dvdy" }, true);

	//CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.grad.dzbdx, XModel_g.grad.dzbdx);
	//CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.grad.dzbdy, XModel_g.grad.dzbdy);

	//CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.grad.dzsdx, XModel_g.grad.dzsdx);
	//CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.grad.dzsdy, XModel_g.grad.dzsdy);

	//CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.grad.dhdx, XModel_g.grad.dhdx);
	//CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.grad.dhdy, XModel_g.grad.dhdy);

	cudaEventCreate(&startGnew);


	cudaEventCreate(&stopGnew);

	cudaEventRecord(startGnew, NULL);
	gradientGPUnew(XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.grad, XModel_g.zb);
	cudaEventRecord(stopGnew, NULL);

	// Wait for the stop event to complete
	cudaEventSynchronize(stopGnew);

	float msecTotalGnew = 0.0f;
	cudaEventElapsedTime(&msecTotalGnew, startGnew, stopGnew);

	cudaEventDestroy(startGnew);
	cudaEventDestroy(stopGnew);

	CompareCPUvsGPU(XParam, XModel, XModel_g, { "dhdx","dhdy", "dzsdx","dzsdy","dudx","dudy","dvdx","dvdy" }, true);

	printf("Runtime : old gradient=%f, new Gradient=%f in msec\n", msecTotalG, msecTotalGnew);
	
	return fastest;

}

/*! \fn bool testzoneOutDef = ZoneOutputTest(int nzones, T zsinit)
*
* This function test the spped and accuracy of a new gradient function
* gradient are only calculated for zb but assigned to different gradient variable for storage
*/
template <class T> bool TestHaloSpeed(Param XParam, Model<T> XModel, Model<T> XModel_g)
{
	Forcing<float> XForcing;

	XForcing = MakValleyBathy(XParam, T(0.4), true, true);

	float maxtopo = std::numeric_limits<float>::min();
	float mintopo = std::numeric_limits<float>::max();

	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			maxtopo = max(XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx], maxtopo);
			mintopo = min(XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx], mintopo);
		}
	}

	// Overrule whatever may be set in the param file
	XParam.xmax = XForcing.Bathy[0].xmax;
	XParam.ymax = XForcing.Bathy[0].ymax;
	XParam.xo = XForcing.Bathy[0].xo;
	XParam.yo = XForcing.Bathy[0].yo;

	XParam.dx = XForcing.Bathy[0].dx;

	XParam.zsinit = mintopo + 0.5;// Had a small amount of water to avoid a huge first step that would surely break the setup
	XParam.endtime = 20.0;

	XParam.outputtimestep = XParam.endtime;

	XParam.minlevel = 0;
	XParam.maxlevel = 1;
	XParam.initlevel = 0;

	//coarse to fine
	// Change arg 1 and 2 if the slope is changed
	XParam.AdaptCrit = "Inrange";
	XParam.Adapt_arg1 = "0.0";
	XParam.Adapt_arg2 = "2.0";
	XParam.Adapt_arg3 = "zb";

	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	InitialConditions(XParam, XForcing, XModel);

	InitialAdaptation(XParam, XForcing, XModel);

	SetupGPU(XParam, XModel, XForcing, XModel_g);

	


	// Copy zs from CPU to GPU ... again
	CopytoGPU(XParam.nblkmem, XParam.blksize, XModel.evolv.zs, XModel_g.evolv_o.zs);
	CopytoGPU(XParam.nblkmem, XParam.blksize, XModel.evolv.zs, XModel_g.evolv.zs);

	cudaStream_t streams[2];
	CUDA_CHECK(cudaStreamCreate(&streams[0]));
	CUDA_CHECK(cudaStreamCreate(&streams[1]));


	fillHaloC(XParam, XModel.blocks, XModel.evolv.zs);
	fillHaloGPU(XParam, XModel_g.blocks, streams[0], XModel_g.evolv.zs);
	fillHaloGPUnew(XParam, XModel_g.blocks, streams[1], XModel_g.evolv_o.zs);

	CUDA_CHECK(cudaDeviceSynchronize());

	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);


	//CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.evolv.u, XModel_g.evolv.zs);
	//CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, XModel.evolv.v, XModel_g.evolvo.zs);

	diffArray(XParam, XModel.blocks, "GPU_old", true, XModel.evolv.zs, XModel_g.evolv.zs, XModel.evolv.u, XModel.evolv_o.u);
	diffArray(XParam, XModel.blocks, "GPU_new", true, XModel.evolv.zs, XModel_g.evolv_o.zs, XModel.evolv.v, XModel.evolv_o.v);

	return true;
}

template <class T> int TestInstability(Param XParam, Model<T> XModel, Model<T> XModel_g)
{
	Forcing<float> XForcing;

	XForcing = MakValleyBathy(XParam, T(0.4), true, true);

	XParam.conserveElevation = true;

	float maxtopo = std::numeric_limits<float>::min();
	float mintopo = std::numeric_limits<float>::max();

	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			maxtopo = max(XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx], maxtopo);
			mintopo = min(XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx], mintopo);
		}
	}

	// Overrule whatever may be set in the param file
	XParam.xmax = XForcing.Bathy[0].xmax;
	XParam.ymax = XForcing.Bathy[0].ymax;
	XParam.xo = XForcing.Bathy[0].xo;
	XParam.yo = XForcing.Bathy[0].yo;

	XParam.dx = XForcing.Bathy[0].dx;

	XParam.zsinit = mintopo + 6.9;// Had a water level so that the wet and dry affects the 
	XParam.endtime = 20.0;

	XParam.outputtimestep = XParam.endtime;

	XParam.minlevel = 0;
	XParam.maxlevel = 2;
	XParam.initlevel = 0;

	// coarse to fine
	// Change arg 1 and 2 if the slope is changed
	XParam.AdaptCrit = "Targetlevel";
	XParam.Adapt_arg1 = "";
	XParam.Adapt_arg2 = "";
	XParam.Adapt_arg3 = "";

	StaticForcingP<int> targetlevel;
	XForcing.targetadapt.push_back(targetlevel);

	XForcing.targetadapt[0].xo = 0.0;
	XForcing.targetadapt[0].yo = 0.0;

	XForcing.targetadapt[0].xmax = 31.0;
	XForcing.targetadapt[0].ymax = 31.0;
	XForcing.targetadapt[0].nx = 32;
	XForcing.targetadapt[0].ny = 32;

	XForcing.targetadapt[0].dx = 1.0;

	AllocateCPU(XForcing.Bathy[0].nx, XForcing.Bathy[0].ny, XForcing.targetadapt[0].val);

	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			XForcing.targetadapt[0].val[i + j * XForcing.Bathy[0].nx] = 1;
		}
	}

	XForcing.targetadapt[0].val[12 + 12 * XForcing.Bathy[0].nx] = 2;


	// Setup Model(s)

	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	InitialConditions(XParam, XForcing, XModel);

	InitialAdaptation(XParam, XForcing, XModel);

	SetupGPU(XParam, XModel, XForcing, XModel_g);

	// Run first full step (i.e. 2 half steps)

	Loop<T> XLoop = InitLoop(XParam, XModel);
	
	//FlowCPU(XParam, XLoop, XForcing, XModel);
	HalfStepCPU(XParam, XLoop, XForcing, XModel);

	T maxu = std::numeric_limits<float>::min();
	T maxv = std::numeric_limits<float>::min();

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

				maxu = max(maxu, abs(XModel.evolv.u[i]));
				maxv = max(maxv, abs(XModel.evolv.v[i]));
			}
		}
	}

	bool test = false;

	if (maxu > T(std::numeric_limits<T>::epsilon() * T(1000.0)) || maxv > T(std::numeric_limits<T>::epsilon() * T(1000.0)))
	{
		//test = true;
		XParam.outvars = { "zb","h","zs","u","v","Fqux","Fqvx","Fquy","Fqvy", "Fhu", "Fhv", "dh", "dhu", "dhv", "Su", "Sv","dhdx", "dhdy", "dzsdx", "dzsdy" };
		InitSave2Netcdf(XParam, XModel);

	}
	else
	{
		test = true;
	}


	return test;

}

template <class T> int TestFirsthalfstep(Param XParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g)
{
	
	// Setup Model(s)

	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	InitialConditions(XParam, XForcing, XModel);

	InitialAdaptation(XParam, XForcing, XModel);

	SetupGPU(XParam, XModel, XForcing, XModel_g);

	// Run first full step (i.e. 2 half steps)

	Loop<T> XLoop = InitLoop(XParam, XModel);

	//FlowCPU(XParam, XLoop, XForcing, XModel);
	HalfStepCPU(XParam, XLoop, XForcing, XModel);

	T maxu = std::numeric_limits<float>::min();
	T maxv = std::numeric_limits<float>::min();

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

				maxu = max(maxu, abs(XModel.evolv.u[i]));
				maxv = max(maxv, abs(XModel.evolv.v[i]));
			}
		}
	}

	bool test = false;

	//test = true;
	XParam.outvars = { "zb","h","zs","u","v","Fqux","Fqvx","Fquy","Fqvy", "Fhu", "Fhv", "dh", "dhu", "dhv", "Su", "Sv","dhdx", "dhdy", "dzsdx", "dzsdy" };
	InitSave2Netcdf(XParam, XModel);

}


template <class T> Forcing<float> MakValleyBathy(Param XParam, T slope, bool bottop, bool flip)
{
	//

	Forcing<float> XForcing;

	StaticForcingP<float> bathy;

	float* dummybathy;

	XForcing.Bathy.push_back(bathy);

	XForcing.Bathy[0].xo = 0.0;
	XForcing.Bathy[0].yo = 0.0;

	XForcing.Bathy[0].xmax = 31.0;
	XForcing.Bathy[0].ymax = 31.0;
	XForcing.Bathy[0].nx = 32;
	XForcing.Bathy[0].ny = 32;

	XForcing.Bathy[0].dx = 1.0;

	T x, y;
	T center = T(10.5);

	AllocateCPU(1, 1, XForcing.left.blks, XForcing.right.blks, XForcing.top.blks, XForcing.bot.blks);

	AllocateCPU(XForcing.Bathy[0].nx, XForcing.Bathy[0].ny, XForcing.Bathy[0].val);
	AllocateCPU(XForcing.Bathy[0].nx, XForcing.Bathy[0].ny, dummybathy);


	float maxtopo = std::numeric_limits<float>::min();
	float mintopo = std::numeric_limits<float>::max();
	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			x = T(XForcing.Bathy[0].xo + i * XForcing.Bathy[0].dx);
			y = T(XForcing.Bathy[0].yo + j * XForcing.Bathy[0].dx);


			dummybathy[i + j * XForcing.Bathy[0].nx] = float(ValleyBathy(y, x, slope, center));

			maxtopo = max(dummybathy[i + j * XForcing.Bathy[0].nx], maxtopo);


		}
	}

	// Make surrounding wall
	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{

		dummybathy[0 + j * XForcing.Bathy[0].nx] = maxtopo + 5.0f;
		dummybathy[1 + j * XForcing.Bathy[0].nx] = maxtopo + 5.0f;

		dummybathy[j + 0 * XForcing.Bathy[0].nx] = maxtopo + 5.0f;
		dummybathy[j + 1 * XForcing.Bathy[0].nx] = maxtopo + 5.0f;

		dummybathy[(XForcing.Bathy[0].nx - 1) + j * XForcing.Bathy[0].nx] = maxtopo + 5.0f;
		dummybathy[(XForcing.Bathy[0].nx - 2) + j * XForcing.Bathy[0].nx] = maxtopo + 5.0f;

		dummybathy[j + (XForcing.Bathy[0].ny - 1) * XForcing.Bathy[0].nx] = maxtopo + 5.0f;
		dummybathy[j + (XForcing.Bathy[0].ny - 2) * XForcing.Bathy[0].nx] = maxtopo + 5.0f;


	}

	// make a specially elevated spot 

	dummybathy[(XForcing.Bathy[0].nx - 1) + 0 * XForcing.Bathy[0].nx] = maxtopo + 10.0f;
	dummybathy[(XForcing.Bathy[0].nx - 2) + 0 * XForcing.Bathy[0].nx] = maxtopo + 10.0f;

	dummybathy[(XForcing.Bathy[0].nx - 1) + 1 * XForcing.Bathy[0].nx] = maxtopo + 10.0f;
	dummybathy[(XForcing.Bathy[0].nx - 2) + 1 * XForcing.Bathy[0].nx] = maxtopo + 10.0f;

	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			mintopo = min(dummybathy[i + j * XForcing.Bathy[0].nx], mintopo);
		}
	}

	// Flip or rotate the bathy according to what is requested
	for (int j = 0; j < XForcing.Bathy[0].ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy[0].nx; i++)
		{
			if (!flip && !bottop)
			{
				XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = dummybathy[i + j * XForcing.Bathy[0].nx];
			}
			else if (flip && !bottop)
			{
				XForcing.Bathy[0].val[(XForcing.Bathy[0].nx - 1 - i) + j * XForcing.Bathy[0].nx] = dummybathy[i + j * XForcing.Bathy[0].nx];
			}
			else if (!flip && bottop)
			{
				XForcing.Bathy[0].val[i + j * XForcing.Bathy[0].nx] = dummybathy[j + i * XForcing.Bathy[0].nx];
			}
			else if (flip && bottop)
			{
				XForcing.Bathy[0].val[i + (XForcing.Bathy[0].ny - 1 - j) * XForcing.Bathy[0].nx] = dummybathy[j + i * XForcing.Bathy[0].nx];
			}
		}
	}

	free(dummybathy);

	return XForcing;

}


/*! \fn void alloc_init2Darray(float** arr, int NX, int NY)
* This function allocates and fills a 2D array with zero values
*
*
*/
void alloc_init2Darray(float** arr, int NX, int NY)
{
	int i, j;
	//Allocation
	arr = (float**)malloc(sizeof(float*) * NX);
	for (i = 0; i < NX; i++) {
		arr[i] = (float*)malloc(sizeof(float) * NY);
	}

	//arr = (int **)malloc(sizeof(int *) * NX);
	//for (i = 0; i < NX; i++) {
	//	arr[i] = (int *)malloc(sizeof(int) * NY);
	//}
	//Initialisation
	for (i = 0; i < NX; i++) {
		for (j = 0; j < NY; j++) {
			arr[i][j] = 0;
		}
	}
}

/*! \fn void init3Darray(float*** arr, int rows, int cols, int depths)
* This function fill a 3D array with zero values 
*
*
*/
void init3Darray(float*** arr, int rows, int cols, int depths)
{
	int i, j, k;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			for (k = 0; k < depths; k++)
			{
				arr[i][j][k] = 0;
			}
		}
	}
}

/*! \fn void fillrandom(Param XParam, BlockP<T> XBlock, T* z)
* This function fill an array with random values (0 - 1)
*
* 
*/
template <class T> void fillrandom(Param XParam, BlockP<T> XBlock, T* z)
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
				z[n] = T(rand()) / T(RAND_MAX);
			}
		}
	}
}
template void fillrandom<float>(Param XParam, BlockP<float> XBlock, float* z);
template void fillrandom<double>(Param XParam, BlockP<double> XBlock, double* z);

/*! \fn void fillgauss(Param XParam, BlockP<T> XBlock, T amp, T* z)
* This function fill an array with a gaussian bump
* 
* borrowed/adapted from Basilisk test (?)
*/
template <class T> void fillgauss(Param XParam, BlockP<T> XBlock, T amp, T* z)
{
	T delta, x, y;
	T cc = T(0.05) * (XParam.xmax - XParam.xo);
	T xorigin = XParam.xo + T(0.5) * (XParam.xmax - XParam.xo);
	T yorigin = XParam.yo + T(0.5) * (XParam.ymax - XParam.yo);


	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XBlock.active[ibl];
		delta = calcres(T(XParam.dx), XBlock.level[ib]);


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				//
				int n = memloc(XParam, ix, iy, ib);
				x = T(XParam.xo + XBlock.xo[ib] + ix * delta);
				y = T(XParam.yo + XBlock.yo[ib] + iy * delta);
				z[n] = z[n] + amp * exp(T(-1.0) * T(((x - xorigin) * (x - xorigin) + (y - yorigin) * (y - yorigin)) / (2.0 * cc * cc)));


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
	creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, XModel.blocks.outZone[0]);
	outvar = "h";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar], XModel.blocks.outZone[0]);
	outvar = "u";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar], XModel.blocks.outZone[0]);
	outvar = "v";
	//copyID2var(XParam, XModel.blocks, XModel.OutputVarMap[outvar]);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar], XModel.blocks.outZone[0]);
	outvar = "zb";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar], XModel.blocks.outZone[0]);
	outvar = "zs";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar], XModel.blocks.outZone[0]);


	FlowCPU(XParam, XLoop, XForcing, XModel);


	//outvar = "cf";
	//defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.cf);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhdx", 3, XModel.grad.dhdx, XModel.blocks.outZone[0]);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhdy", 3, XModel.grad.dhdy, XModel.blocks.outZone[0]);

	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fhv", 3, XModel.flux.Fhv, XModel.blocks.outZone[0]);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fhu", 3, XModel.flux.Fhu, XModel.blocks.outZone[0]);


	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqux", 3, XModel.flux.Fqux, XModel.blocks.outZone[0]);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fquy", 3, XModel.flux.Fquy, XModel.blocks.outZone[0]);

	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqvx", 3, XModel.flux.Fqvx, XModel.blocks.outZone[0]);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqvy", 3, XModel.flux.Fqvy, XModel.blocks.outZone[0]);


	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Su", 3, XModel.flux.Su, XModel.blocks.outZone[0]);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Sv", 3, XModel.flux.Sv, XModel.blocks.outZone[0]);


	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dh", 3, XModel.adv.dh, XModel.blocks.outZone[0]);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhu", 3, XModel.adv.dhu, XModel.blocks.outZone[0]);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhv", 3, XModel.adv.dhv, XModel.blocks.outZone[0]);

	writenctimestep(XParam.outfile, XLoop.totaltime + XLoop.dt);


	outvar = "h";
	writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, XModel.OutputVarMap[outvar], XModel.blocks.outZone[0]);

	outvar = "zs";
	writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, XModel.OutputVarMap[outvar], XModel.blocks.outZone[0]);
	outvar = "u";
	writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, XModel.OutputVarMap[outvar], XModel.blocks.outZone[0]);
	outvar = "v";
	writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, XModel.OutputVarMap[outvar], XModel.blocks.outZone[0]);

}

template void TestingOutput<float>(Param XParam, Model<float> XModel);
template void TestingOutput<double>(Param XParam, Model<double> XModel);

/*! \fn void copyID2var(Param XParam, BlockP<T> XBlock, T* z)
* This function copies block info to an output variable
* This function is somewhat useful when checking bugs in the mesh refinement or coarsening
* one needs to provide a pointer(z) allocated on the CPU to store the clockinfo
* This fonction only works on CPU
*
*/
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
				z[n] = T(ib);
			}
		}
	}

}

template void copyID2var<float>(Param XParam, BlockP<float> XBlock, float* z);
template void copyID2var<double>(Param XParam, BlockP<double> XBlock, double* z);


/*! \fn void copyBlockinfo2var(Param XParam, BlockP<T> XBlock, int* blkinfo, T* z)
* This function copies blick info to an output variable
* This function is somewhat useful when checking bugs in the mesh refinement or coarsening
* one needs to provide a pointer(z) allocated on the CPU to store the clockinfo
* This fonction only works on CPU
*
*/
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


/*! \fn void CompareCPUvsGPU(Param XParam, Model<T> XModel, Model<T> XModel_g, std::vector<std::string> varlist, bool checkhalo)
* This function compares the Valiables in a CPU model and a GPU models
* This function is quite useful when checking both are identical enough
* one needs to provide a list (vector<string>) of variable to check
* 
*/
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
	/*
	creatncfileBUQ(XParam, XModel.blocks);

	for (int ivar = 0; ivar < varlist.size(); ivar++)
	{
		defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, varlist[ivar], 3, XModel.OutputVarMap[varlist[ivar]], XModel.blocks.outZone[0]);
	}
	*/
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
		diffArray(XParam, XModel.blocks, varlist[ivar], checkhalo, XModel.OutputVarMap[varlist[ivar]], XModel_g.OutputVarMap[varlist[ivar]], gpureceive, diff);
	}



	free(gpureceive);
	free(diff);

}
template void CompareCPUvsGPU<float>(Param XParam, Model<float> XModel, Model<float> XModel_g, std::vector<std::string> varlist, bool checkhalo);
template void CompareCPUvsGPU<double>(Param XParam, Model<double> XModel, Model<double> XModel_g, std::vector<std::string> varlist, bool checkhalo);


/*! \fn void diffdh(Param XParam, BlockP<T> XBlock, T* input, T* output, T* shuffle)
* This function Calculates The difference in left and right flux terms.
* This function is quite useful when checking for Lake-at-Rest states
* This function requires a preallocated output and a shuffle (right side term) CPU pointers to save the result of teh calculation
*/
template <class T> void diffdh(Param XParam, BlockP<T> XBlock, T* input, T* output, T* shuffle)
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

/*! \fn void diffSource(Param XParam, BlockP<T> XBlock, T* Fqux, T* Su, T* output)
* This function Calculate The source term of the equation. 
* This function is quite useful when checking for Lake-at-Rest states
* This function requires an outputCPU pointers to save the result of teh calculation
*/
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

				output[i] = Fqux[i] - Su[iright];
				//shuffle[i] = input[iright];
			}
		}
	}
}

/*! \fn void diffArray(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, std::string varname, bool checkhalo, T* cpu, T* gpu, T* dummy, T* out)
* Calculate and output the difference between a CPU and a GPU array
* This function is quite usefull to spot inconsistencies between the GPU and CPU algorithmes.
* This function requires two (dummy and an output) CPU pointers to transition the GPU data on the CU RAM for comparison and saving to the disk
*/
template <class T> void diffArray(Param XParam, BlockP<T> XBlock, std::string varname, bool checkhalo, T* cpu, T* gpu, T* dummy, T* out)
{
	T diff, maxdiff, rmsdiff;
	unsigned int nit = 0;
	int ixmd, iymd, ibmd;
	//copy GPU back to the CPU (store in dummy)
	CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, dummy, gpu);


	T hugeposval = std::numeric_limits<T>::max();
	T hugenegval = T(-1.0) * hugeposval;
	T epsilon = std::numeric_limits<T>::epsilon();

	rmsdiff = T(0.0);
	maxdiff = hugenegval;
	ixmd = 0;
	iymd = 0;
	ibmd = 0;

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

				if (abs(diff) >= maxdiff)
				{
					maxdiff = utils::max(abs(diff), maxdiff);
					ixmd = ix;
					iymd = iy;
					ibmd = ib;
				}

				rmsdiff = rmsdiff + utils::sq(diff);
				nit++;
				out[n] = diff;
			}
		}

	}


	rmsdiff = rmsdiff / nit;



	if (maxdiff <= T(10000.0) * (epsilon))
	{
		log(varname + " PASS");
	}
	else
	{
		creatncfileBUQ(XParam, XBlock);
		log(varname + " FAIL: " + " Max difference: " + std::to_string(maxdiff) + " (at: ix = " + std::to_string(ixmd) + " iy = " + std::to_string(iymd) + " ib = " + std::to_string(ibmd) + ") RMS difference: " + std::to_string(rmsdiff) + " Eps: " + std::to_string(epsilon));
		defncvarBUQ(XParam, XBlock.active, XBlock.level, XBlock.xo, XBlock.yo, varname + "_CPU", 3, cpu, XBlock.outZone[0]);
		defncvarBUQ(XParam, XBlock.active, XBlock.level, XBlock.xo, XBlock.yo, varname + "_GPU", 3, dummy, XBlock.outZone[0]);
		defncvarBUQ(XParam, XBlock.active, XBlock.level, XBlock.xo, XBlock.yo, varname + "_diff", 3, out, XBlock.outZone[0]);
	}




}



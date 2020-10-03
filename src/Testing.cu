


#include "Testing.h"




/*! \fn bool testing(Param XParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g)
* Wrapping function for all the inbuilt test
* This function is the entry point to the software
*/
template <class T> void Testing(Param XParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g)
{
	bool toto;
	if (XParam.test == 0)
	{
		// Test 0 is pure bump test
		
		
		toto=GaussianHumptest(0.1f);


	}

	

}
template void Testing<float>(Param XParam, Forcing<float> XForcing, Model<float> XModel, Model<float> XModel_g);
template void Testing<double>(Param XParam, Forcing<float> XForcing, Model<double> XModel, Model<double> XModel_g);


template <class T> bool GaussianHumptest(T zsnit)
{
	
	// this is a preplica of the tutorial case for Basilisk
	Param XParam;

	T x, y, delta;
	T cc = T(0.05);// Match the 200 in chracteristic radius used in Basilisk  1/(2*cc^2)=200
	

	T a = T(1.0); //Gaussian wave amplitude

	// initialise domain and required resolution
	XParam.xo = -0.5;
	XParam.yo = -0.5;

	XParam.xmax = 0.5;
	XParam.ymax = 0.5;
	//level 8 is 
	XParam.dx = 1.0 / ((1<<8)-1);

	XParam.initlevel = 0;
	XParam.minlevel = 0;
	XParam.maxlevel = 0;

	XParam.zsinit = zsnit;
	XParam.zsoffset = 0.0;

	//Output times for comparisons
	XParam.endtime = 30.0;
	XParam.outputtimestep = 1.0;

	// Enforece GPU/CPU
	XParam.GPUDEVICE = -1;

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

	// initialise forcing bathymetry to 0
	XForcing.Bathy.xo = -1.0;
	XForcing.Bathy.yo = -1.0;

	XForcing.Bathy.xmax = 1.0;
	XForcing.Bathy.ymax = 1.0;
	XForcing.Bathy.nx = 3;
	XForcing.Bathy.ny = 3;

	XForcing.Bathy.dx = 1.0;

	AllocateCPU(1, 1, XForcing.left.blks, XForcing.right.blks, XForcing.top.blks, XForcing.bot.blks);

	AllocateCPU(XForcing.Bathy.nx, XForcing.Bathy.ny, XForcing.Bathy.val);

	for (int j = 0; j < XForcing.Bathy.ny; j++)
	{
		for (int i = 0; i < XForcing.Bathy.nx; i++)
		{
			XForcing.Bathy.val[i + j * XForcing.Bathy.nx] = 0.0f;
		}
	}

	checkparamsanity(XParam, XForcing);

	InitMesh(XParam, XForcing, XModel);

	// Recreate the initia;l conditions
	//InitArrayBUQ(XParam, XModel.blocks, T(0.0), XModel.zb);
	//InitArrayBUQ(XParam, XModel.blocks, zsnit, XModel.evolv.zs);

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
				x = XModel.blocks.xo[ib] + ix * delta;
				y = XModel.blocks.yo[ib] + iy * delta;
				XModel.evolv.zs[n] = XModel.evolv.zs[n] + a * exp(T(-1.0) * ((x - xorigin) * (x - xorigin) + (y - yorigin) * (y - yorigin)) / (2.0 * cc * cc));

				
			}
		}
	}

	Loop<T> XLoop;

	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();

	XLoop.totaltime = 0.0;

	InitSave2Netcdf(XParam, XModel);
	XLoop.nextoutputtime = XParam.outputtimestep;

	while (XLoop.totaltime < XParam.endtime)
	{

		if (XParam.GPUDEVICE >= 0)
		{
			FlowGPU(XParam, XLoop, XForcing, XModel_g);
		}
		else
		{
			FlowCPU(XParam, XLoop, XForcing, XModel);
		}



		//diffdh(XParam, XModel.blocks, XModel.flux.Su, diff, shuffle);
		//diffSource(XParam, XModel.blocks, XModel.flux.Fqux, XModel.flux.Su, diff);
		XLoop.totaltime = XLoop.totaltime + XLoop.dt;

		if (XLoop.nextoutputtime - XLoop.totaltime <= XLoop.dt * T(0.00001) && XParam.outputtimestep > 0.0)
		{
			if (XParam.GPUDEVICE >= 0)
			{
				for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
				{
					CUDA_CHECK(cudaMemcpy(XModel.OutputVarMap[XParam.outvars[ivar]], XModel_g.OutputVarMap[XParam.outvars[ivar]], XParam.nblkmem * XParam.blksize * sizeof(T), cudaMemcpyDeviceToHost));
				}
			}

			Save2Netcdf(XParam, XModel);


			XLoop.nextoutputtime = min(XLoop.nextoutputtime + XParam.outputtimestep, XParam.endtime);

		}
	}
	
	return true;
}
template bool GaussianHumptest<float>(float zsnit);
template bool GaussianHumptest<double>(double zsnit);



/*! \fn TestingOutput(Param XParam, Model<T> XModel)
*  
* 
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




/*template <class T> void Gaussianhump(Param  XParam, Model<T> XModel)
{
	T x, y,delta;
	T cc = 100.0;
	

	T a = 0.2;

	T* diff,*shuffle;

	AllocateCPU(XParam.nblkmem, XParam.blksize, diff);
	AllocateCPU(XParam.nblkmem, XParam.blksize, shuffle);

	T xorigin = XParam.xo + 0.5 * (XParam.xmax - XParam.xo);
	T yorigin = XParam.yo + 0.5 * (XParam.ymax - XParam.yo);
	Loop<T> XLoop;
	
	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();

	XLoop.totaltime = 0.0;

	
	XLoop.nextoutputtime = XParam.outputtimestep;

	
	//InitArrayBUQ(XParam, XModel.blocks, T(-1.0), XModel.zb);
	
	// make an empty forcing
	Forcing<float> XForcing;



	if (XParam.GPUDEVICE >= 0)
	{
		CopytoGPU(XParam.nblkmem, XParam.blksize, XParam, XModel, XModel_g);
	}

	InitSave2Netcdf(XParam, XModel);

	//defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "diff", 3, diff);
	//defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "shuffle", 3, shuffle);
	

	while (XLoop.totaltime < XParam.endtime)
	{
		
		if (XParam.GPUDEVICE >= 0)
		{
			FlowGPU(XParam, XLoop, XForcing, XModel_g);
		}
		else
		{
			FlowCPU(XParam, XLoop, XForcing, XModel);
		}
		

		
		//diffdh(XParam, XModel.blocks, XModel.flux.Su, diff, shuffle);
		//diffSource(XParam, XModel.blocks, XModel.flux.Fqux, XModel.flux.Su, diff);
		XLoop.totaltime = XLoop.totaltime + XLoop.dt;

		if (XLoop.nextoutputtime - XLoop.totaltime <= XLoop.dt * T(0.00001) && XParam.outputtimestep > 0.0)
		{
			if (XParam.GPUDEVICE >= 0)
			{
				for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
				{
					CUDA_CHECK(cudaMemcpy(XModel.OutputVarMap[XParam.outvars[ivar]], XModel_g.OutputVarMap[XParam.outvars[ivar]], XParam.nblkmem * XParam.blksize * sizeof(T), cudaMemcpyDeviceToHost));
				}
			}
			
			Save2Netcdf(XParam, XModel);
			

			XLoop.nextoutputtime = min(XLoop.nextoutputtime + XParam.outputtimestep, XParam.endtime);
		}
	}
	
	
	
	free(shuffle);
	free(diff);
}
template void Gaussianhump<float>(Param XParam, Model<float> XModel, Model<float> XModel_g);
template void Gaussianhump<double>(Param XParam, Model<double> XModel, Model<double> XModel_g);
*/


template <class T> void CompareCPUvsGPU(Param XParam, Model<T> XModel, Model<T> XModel_g)
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

	Forcing<float> XForcing;

	AllocateCPU(XParam.nblkmem, XParam.blksize, gpureceive);
	AllocateCPU(XParam.nblkmem, XParam.blksize, diff);


	//============================================
	// Compare gradients for evolving parameters
	
	// GPU
	FlowGPU(XParam, XLoop, XForcing, XModel_g);
	T dtgpu = XLoop.dt;
	// CPU
	FlowCPU(XParam, XLoop, XForcing, XModel);
	T dtcpu = XLoop.dt;
	// calculate difference
	//diffArray(XParam, XLoop, XModel.blocks, XModel.evolv.h, XModel_g.evolv.h, XModel.evolv_o.u);

	creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo);

	
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "h", 3, XModel.evolv_o.h);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "u", 3, XModel.evolv_o.u);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "v", 3, XModel.evolv_o.v);

	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqux", 3, XModel.flux.Fqux);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fquy", 3, XModel.flux.Fquy);

	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqvx", 3, XModel.flux.Fqvx);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqvy", 3, XModel.flux.Fqvy);


	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Su", 3, XModel.flux.Su);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Sv", 3, XModel.flux.Sv);


	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dh", 3, XModel.adv.dh);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhu", 3, XModel.adv.dhu);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhv", 3, XModel.adv.dhv);

	std::string varname = "dt";
	if (abs(dtgpu - dtcpu) < (XLoop.epsilon * 2))
	{
		log(varname + " PASS");
	}
	else
	{
		log(varname + " FAIL: " + " GPU(" + std::to_string(dtgpu) + ") - CPU("+std::to_string(dtcpu) +") =  difference: "+  std::to_string(abs(dtgpu - dtcpu)) + " Eps: " + std::to_string(XLoop.epsilon));
		
	}

	//Check evolving param
	diffArray(XParam, XLoop, XModel.blocks, "h", XModel.evolv_o.h, XModel_g.evolv_o.h, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "zs", XModel.evolv_o.zs, XModel_g.evolv_o.zs, gpureceive, diff);

	diffArray(XParam, XLoop, XModel.blocks, "u", XModel.evolv_o.u, XModel_g.evolv_o.u, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "v", XModel.evolv_o.v, XModel_g.evolv_o.v, gpureceive, diff);
	

	
	//check gradients
	diffArray(XParam, XLoop, XModel.blocks, "dhdx", XModel.grad.dhdx, XModel_g.grad.dhdx, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "dhdy", XModel.grad.dhdy, XModel_g.grad.dhdy, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "dzsdx", XModel.grad.dzsdx, XModel_g.grad.dzsdx, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "dzsdy", XModel.grad.dzsdy, XModel_g.grad.dzsdy, gpureceive, diff);

	//Check Kurganov
	diffArray(XParam, XLoop, XModel.blocks,"Fhu", XModel.flux.Fhu, XModel_g.flux.Fhu, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "Fqux", XModel.flux.Fqux, XModel_g.flux.Fqux, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "Su", XModel.flux.Su, XModel_g.flux.Su, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "Fqvx", XModel.flux.Fqvx, XModel_g.flux.Fqvx, gpureceive, diff);

	diffArray(XParam, XLoop, XModel.blocks, "Fhv", XModel.flux.Fhv, XModel_g.flux.Fhv, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "Fqvy", XModel.flux.Fqvy, XModel_g.flux.Fqvy, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "Sv", XModel.flux.Sv, XModel_g.flux.Sv, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "Fquy", XModel.flux.Fquy, XModel_g.flux.Fquy, gpureceive, diff);

	diffArray(XParam, XLoop, XModel.blocks, "dh", XModel.adv.dh, XModel_g.adv.dh, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "dhu", XModel.adv.dhu, XModel_g.adv.dhu, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "dhv", XModel.adv.dhv, XModel_g.adv.dhv, gpureceive, diff);



	
	free(gpureceive);
	free(diff);
	
}
template void CompareCPUvsGPU<float>(Param XParam, Model<float> XModel, Model<float> XModel_g);
template void CompareCPUvsGPU<double>(Param XParam,  Model<double> XModel, Model<double> XModel_g);

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


template <class T> void diffArray(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, std::string varname, T* cpu, T* gpu, T* dummy, T* out)
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


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
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
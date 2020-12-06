#include "Mainloop.h"



template <class T> void MainLoop(Param &XParam, Forcing<float> XForcing, Model<T>& XModel, Model<T> &XModel_g)
{
	
	log("Initialising model main loop");
	
	Loop<T> XLoop = InitLoop(XParam, XModel);

	//Define some useful variables 
	Initmeanmax(XParam, XLoop, XModel, XModel_g);

	// fill halo for zb
	// only need to do that once 
	fillHaloC(XParam, XModel.blocks, XModel.zb);
	if (XParam.GPUDEVICE >= 0)
	{
		CUDA_CHECK(cudaStreamCreate(&XLoop.streams[0]));
		fillHaloGPU(XParam, XModel_g.blocks, XLoop.streams[0], XModel_g.zb);

		cudaStreamDestroy(XLoop.streams[0]);
	}



	log("\t\tCompleted");
	log("Model Running...");
	while (XLoop.totaltime < XParam.endtime)
	{
		// Bnd stuff here
		updateBnd(XParam, XLoop, XForcing, XModel, XModel_g);
		

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

		// Apply tsunami deformation if any (this needs to happen after totaltime has been incremented)
		deformstep(XParam, XLoop, XForcing.deform, XModel, XModel_g);

		// Do Sum & Max variables Here
		Calcmeanmax(XParam, XLoop, XModel, XModel_g);

		// Check & collect TSoutput
		pointoutputstep(XParam, XLoop, XModel, XModel_g);

		// Check for map output
		mapoutput(XParam, XLoop, XModel, XModel_g);

		// Reset mean/Max if needed
		resetmeanmax(XParam, XLoop, XModel, XModel_g);

		printstatus(XLoop.totaltime, XLoop.dt);
	}
	

	

}
template void MainLoop<float>(Param& XParam, Forcing<float> XForcing, Model<float>& XModel, Model<float>& XModel_g);
template void MainLoop<double>(Param& XParam, Forcing<float> XForcing, Model<double>& XModel, Model<double>& XModel_g);




 
template <class T> Loop<T> InitLoop(Param &XParam, Model<T> &XModel)
{
	Loop<T> XLoop;
	XLoop.atmpuni = XParam.Paref;
	XLoop.totaltime = XParam.totaltime;
	XLoop.nextoutputtime = XParam.totaltime + XParam.outputtimestep;
	
	// Prepare output files
	InitSave2Netcdf(XParam, XModel);
	InitTSOutput(XParam);
	// Add empty row for each output point
	// This will allow for the loop to each point to work later
	for (int o = 0; o < XParam.TSnodesout.size(); o++)
	{
		XLoop.TSAllout.push_back(std::vector<Pointout>());
	}

	// GPU stuff
	if (XParam.GPUDEVICE >= 0)
	{
		XLoop.blockDim = (16, 16, 1);
		XLoop.gridDim = (XParam.nblk, 1, 1);
	}

	//XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.hugenegval = T(-1.0)* XLoop.hugeposval;
	XLoop.epsilon = std::numeric_limits<T>::epsilon();


	XLoop.dtmax = initdt(XParam, XLoop, XModel);

	return XLoop;

}

template <class T> void updateBnd(Param XParam, Loop<T> XLoop, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g)
{
	if (XParam.GPUDEVICE >= 0)
	{
		Flowbnd(XParam, XLoop, XModel_g.blocks, XForcing.left, XModel_g.evolv);
		Flowbnd(XParam, XLoop, XModel_g.blocks, XForcing.right, XModel_g.evolv);
		Flowbnd(XParam, XLoop, XModel_g.blocks, XForcing.top, XModel_g.evolv);
		Flowbnd(XParam, XLoop, XModel_g.blocks, XForcing.bot, XModel_g.evolv);
	}
	else
	{
		Flowbnd(XParam, XLoop, XModel.blocks, XForcing.left, XModel.evolv);
		Flowbnd(XParam, XLoop, XModel.blocks, XForcing.right, XModel.evolv);
		Flowbnd(XParam, XLoop, XModel.blocks, XForcing.top, XModel.evolv);
		Flowbnd(XParam, XLoop, XModel.blocks, XForcing.bot, XModel.evolv);
	}
}




template <class T> void mapoutput(Param XParam, Loop<T> &XLoop,Model<T> XModel, Model<T> XModel_g)
{
	XLoop.nstepout++;

	if (XLoop.nextoutputtime - XLoop.totaltime <= XLoop.dt * T(0.00001) && XParam.outputtimestep > 0.0)
	{
		char buffer[256];
		sprintf(buffer, "%e", XParam.outputtimestep / XLoop.nstepout);
		std::string str(buffer);

		log("Output to map. Totaltime = "+ std::to_string(XLoop.totaltime) +" s; Mean dt = " + str + " s");
		if (XParam.GPUDEVICE >= 0)
		{
			for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
			{
				CUDA_CHECK(cudaMemcpy(XModel.OutputVarMap[XParam.outvars[ivar]], XModel_g.OutputVarMap[XParam.outvars[ivar]], XParam.nblkmem * XParam.blksize * sizeof(T), cudaMemcpyDeviceToHost));
			}
		}
		
		Save2Netcdf(XParam, XLoop, XModel);


		XLoop.nextoutputtime = min(XLoop.nextoutputtime + XParam.outputtimestep, XParam.endtime);

		XLoop.nstepout = 0;
	}
}

template <class T> void pointoutputstep(Param XParam, Loop<T> &XLoop, Model<T> XModel, Model<T> XModel_g)
{
	//
	dim3 blockDim = (XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim = (XModel.bndblk.nblkTs, 1, 1);
	FILE* fsSLTS;
	if (XParam.GPUDEVICE>=0)
	{

		for (int o = 0; o < XParam.TSnodesout.size(); o++)
		{
			//
			Pointout stepread;
		
			stepread.time = XParam.totaltime;
			stepread.zs = 0.0;// That is a bit useless
			stepread.h = 0.0;
			stepread.u = 0.0;
			stepread.v = 0.0;
			XLoop.TSAllout[o].push_back(stepread);
					
			
			storeTSout << <gridDim, blockDim, 0 >> > (XParam,(int)XParam.TSnodesout.size(), o, XLoop.nTSsteps, XParam.TSnodesout[o].block, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XModel.bndblk.Tsout, XModel_g.evolv, XModel_g.TSstore);
		}
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	else
	{
		for (int o = 0; o < XParam.TSnodesout.size(); o++)
		{
			//
			Pointout stepread;

			int i = memloc(XParam.halowidth, XParam.blkmemwidth, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSnodesout[o].block);

			stepread.time = XParam.totaltime;
			stepread.zs = XModel.evolv.zs[i];
			stepread.h = XModel.evolv.h[i];;
			stepread.u = XModel.evolv.u[i];;
			stepread.v = XModel.evolv.v[i];;
			XLoop.TSAllout[o].push_back(stepread);

		}
	}
	XLoop.nTSsteps++;

	// if the buffer is full or if the model is complete
	if ((XLoop.nTSsteps + 1) * XParam.TSnodesout.size() * 4 > XParam.maxTSstorage || XParam.endtime - XLoop.totaltime <= XLoop.dt * 0.00001f)
	{

		//Flush to disk
		if (XParam.GPUDEVICE >= 0 && XParam.TSnodesout.size() > 0)
		{
			CUDA_CHECK(cudaMemcpy(XModel.TSstore, XModel_g.TSstore, XParam.maxTSstorage * sizeof(T), cudaMemcpyDeviceToHost));
			int oo;
			
			for (int o = 0; o < XParam.TSnodesout.size(); o++)
			{
				for (int istep = 0; istep < XLoop.TSAllout[o].size(); istep++)
				{
					oo = o * 4 + istep * XParam.TSnodesout.size() * 4;
					//
					XLoop.TSAllout[o][istep].h = XModel.TSstore[0 + oo];
					XLoop.TSAllout[o][istep].zs = XModel.TSstore[1 + oo];
					XLoop.TSAllout[o][istep].u = XModel.TSstore[2 + oo];
					XLoop.TSAllout[o][istep].v = XModel.TSstore[3 + oo];
				}
			}

		}
		for (int o = 0; o < XParam.TSnodesout.size(); o++)
		{
			fsSLTS = fopen(XParam.TSnodesout[o].outname.c_str(), "a");


			for (int n = 0; n < XLoop.nTSsteps; n++)
			{
				//


				fprintf(fsSLTS, "%f\t%.4f\t%.4f\t%.4f\t%.4f\n", XLoop.TSAllout[o][n].time, XLoop.TSAllout[o][n].zs, XLoop.TSAllout[o][n].h, XLoop.TSAllout[o][n].u, XLoop.TSAllout[o][n].v);


			}
			fclose(fsSLTS);
			//reset output buffer
			XLoop.TSAllout[o].clear();
		}
		// Reset buffer counter
		XLoop.nTSsteps = 0;




	}
}


template <class T> __global__ void storeTSout(Param XParam,int noutnodes, int outnode, int istep,int blknode, int inode,int jnode, int * blkTS, EvolvingP<T> XEv, T* store)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = blkTS[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);


	if (ib == blknode && ix == inode && iy == jnode)
	{
		store[0 + outnode * 4 + istep * noutnodes * 4] = XEv.h[i];
		store[1 + outnode * 4 + istep * noutnodes * 4] = XEv.zs[i];
		store[2 + outnode * 4 + istep * noutnodes * 4] = XEv.u[i];
		store[3 + outnode * 4 + istep * noutnodes * 4] = XEv.v[i];
	}
}


template <class T> __host__ double initdt(Param XParam, Loop<T> XLoop, Model<T> XModel)
{
	//dim3 blockDim = (XParam.blkwidth, XParam.blkwidth, 1);
	//dim3 gridDim = (XParam.nblk, 1, 1);

	double initdt;

	XLoop.dtmax = XLoop.hugeposval;


	BlockP<T> XBlock = XModel.blocks;

	/*
	if (XParam.GPUDEVICE >= 0)
	{
		CalcInitdtGPU <<< gridDim, blockDim, 0 >>> (XParam, XModel.blocks, XModel.evolv, XModel.time.dtmax);
		initdt = double(CalctimestepGPU(XParam, XLoop, XModel.blocks, XModel.time));
	}
	else
	{
	*/
		CalcInitdtCPU(XParam, XModel.blocks, XModel.evolv, XModel.time.dtmax);
		initdt = double(CalctimestepCPU(XParam, XLoop, XModel.blocks, XModel.time));

	//}

	
	return initdt;
}
template __host__ double initdt<float>(Param XParam, Loop<float> XLoop, Model<float> XModel);
template __host__ double initdt<double>(Param XParam, Loop<double> XLoop, Model<double> XModel);

template <class T> __host__ void CalcInitdtCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEvolv, T* dtmax)
{
	int ib;
	T delta;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		delta = calcres(XParam.dx, XBlock.level[ib]);

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

				dtmax[i] = delta / sqrt(XParam.g * std::max(XEvolv.h[i],T(XParam.eps)));
			}
		}
	}
}

template <class T> __global__ void CalcInitdtGPU(Param XParam, BlockP<T> XBlock,EvolvingP<T> XEvolv, T* dtmax)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.y + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	T delta = calcres(XParam.dx, XBlock.level[ib]);

	dtmax[i] = delta / sqrt(XParam.g * max(XEvolv.h[i],T(XParam.eps)));
}


template <class T> void printstatus(T totaltime, T dt)
{
	std::cout << "\r\e[K" << std::flush;
	std::cout << "\rtotaltime = "<< std::to_string(totaltime) << "   dt = " << std::to_string(dt) << std::flush;
	std::cout << "\r" << std::flush;
	//std::cout << std::endl; // all done
}



#include "Updateforcing.h"

template <class T> void updateforcing(Param XParam, Loop<T> XLoop, Forcing<float> &XForcing)
{
	// Update forcing for all possible dynamic forcing. 
	//if a file is declared that implies that the dynamic forcing is applicable
	if (!XForcing.Rain.inputfile.empty())
	{
		Forcingthisstep(XParam, double(XLoop.totaltime), XForcing.Rain);
	}
	if (!XForcing.Atmp.inputfile.empty())
	{
		Forcingthisstep(XParam, double(XLoop.totaltime), XForcing.Atmp);
	}
	if (!XForcing.UWind.inputfile.empty())//&& !XForcing.UWind.inputfile.empty()
	{
		Forcingthisstep(XParam, double(XLoop.totaltime), XForcing.UWind);
		Forcingthisstep(XParam, double(XLoop.totaltime), XForcing.VWind);
	}

	
}
template void updateforcing<float>(Param XParam, Loop<float> XLoop, Forcing<float>& XForcing);
template void updateforcing<double>(Param XParam, Loop<double> XLoop, Forcing<float>& XForcing);




void Forcingthisstep(Param XParam, double totaltime, DynForcingP<float> &XDynForcing)
{
	dim3 blockDimDF(16, 16, 1);
	dim3 gridDimDF((int)ceil((float)XDynForcing.nx / (float)blockDimDF.x), (int)ceil((float)XDynForcing.ny / (float)blockDimDF.y), 1);



	if (XDynForcing.uniform == 1)
	{
		//
		int Rstepinbnd = 1;

		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = XDynForcing.unidata[Rstepinbnd].time - totaltime;

		while (difft < 0.0)
		{
			Rstepinbnd++;
			difft = XDynForcing.unidata[Rstepinbnd].time - totaltime;
		}

		XDynForcing.nowvalue =interptime(XDynForcing.unidata[Rstepinbnd].wspeed, XDynForcing.unidata[Rstepinbnd - 1].wspeed, XDynForcing.unidata[Rstepinbnd].time - XDynForcing.unidata[Rstepinbnd - 1].time, totaltime - XDynForcing.unidata[Rstepinbnd - 1].time);



	}
	else
	{
		int readfirststep = std::min(std::max((int)floor((totaltime - XDynForcing.to) / XDynForcing.dt), 0), XDynForcing.nt - 2);

		if (readfirststep + 1 > XDynForcing.instep)
		{
			// Need to read a new step from the file

			// First copy the forward (aft) step to become the previous step
			if (XParam.GPUDEVICE >= 0)
			{
				CUDA_CHECK(cudaMemcpy(XDynForcing.before_g, XDynForcing.after_g, XDynForcing.nx * XDynForcing.ny * sizeof(float), cudaMemcpyDeviceToDevice));
			}
			else
			{
				Copy2CartCPU(XDynForcing.nx, XDynForcing.ny, XDynForcing.before, XDynForcing.after);
			}
			
			
			//NextHDstep << <gridDimRain, blockDimRain, 0 >> > (XParam.Rainongrid.nx, XParam.Rainongrid.ny, Rainbef_g, Rainaft_g);
			//CUDA_CHECK(cudaDeviceSynchronize());

			// Read the actual file data

			readvardata(XDynForcing.inputfile, XDynForcing.varname, readfirststep + 1, XDynForcing.after, XDynForcing.flipxx, XDynForcing.flipyy);
			if (XParam.GPUDEVICE >= 0)
			{
				CUDA_CHECK(cudaMemcpy(XDynForcing.after_g, XDynForcing.after, XDynForcing.nx * XDynForcing.ny * sizeof(float), cudaMemcpyHostToDevice));
			}

			XDynForcing.instep = readfirststep + 1;
		}

		// Interpolate the forcing array to this time 
		if (XParam.GPUDEVICE >= 0)
		{
			InterpstepGPU << <gridDimDF, blockDimDF, 0 >> > (XDynForcing.nx, XDynForcing.ny, XDynForcing.instep - 1, float(totaltime), float(XDynForcing.dt), XDynForcing.now_g, XDynForcing.before_g, XDynForcing.after_g);
			CUDA_CHECK(cudaDeviceSynchronize());

			CUDA_CHECK(cudaMemcpyToArray(XDynForcing.GPU.CudArr, 0, 0, XDynForcing.now_g, XDynForcing.nx * XDynForcing.ny * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		else
		{
			InterpstepCPU(XDynForcing.nx, XDynForcing.ny, XDynForcing.instep - 1, totaltime, XDynForcing.dt, XDynForcing.val, XDynForcing.before, XDynForcing.after);
		}
		//InterpstepCPU(XParam.windU.nx, XParam.windU.ny, readfirststep, XParam.totaltime, XParam.windU.dt, Uwind, Uwbef, Uwaft);
		//InterpstepCPU(XParam.windV.nx, XParam.windV.ny, readfirststep, XParam.totaltime, XParam.windV.dt, Vwind, Vwbef, Vwaft);

		

	}

	//return rainuni;
}

template <class T> __host__ void AddRiverForcing(Param XParam, Loop<T> XLoop, std::vector<River> XRivers, Model<T> XModel)
{
	dim3 gridDimRiver(XModel.bndblk.nblkriver, 1, 1);
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	T qnow;
	for (int Rin = 0; Rin < XRivers.size(); Rin++)
	{
		//
		int bndstep = 0;
		double difft = XRivers[Rin].flowinput[bndstep].time - XLoop.totaltime;
		while (difft <= 0.0) // danger?
		{
			bndstep++;
			difft = XRivers[Rin].flowinput[bndstep].time - XLoop.totaltime;
		}

		qnow = T(interptime(XRivers[Rin].flowinput[bndstep].q, XRivers[Rin].flowinput[max(bndstep - 1, 0)].q, XRivers[Rin].flowinput[bndstep].time - XRivers[Rin].flowinput[max(bndstep - 1, 0)].time, XLoop.totaltime - XRivers[Rin].flowinput[max(bndstep - 1, 0)].time));

		if (XParam.GPUDEVICE >= 0)
		{
			InjectRiverGPU <<<gridDimRiver, blockDim, 0 >>> (XParam, XRivers[Rin], qnow, XModel.bndblk.river, XModel.blocks, XModel.adv);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			InjectRiverCPU(XParam, XRivers[Rin], qnow, XModel.bndblk.nblkriver, XModel.bndblk.river, XModel.blocks, XModel.adv);
		}
	}
}
template __host__ void AddRiverForcing<float>(Param XParam, Loop<float> XLoop, std::vector<River> XRivers, Model<float> XModel);
template __host__ void AddRiverForcing<double>(Param XParam, Loop<double> XLoop, std::vector<River> XRivers, Model<double> XModel);


template <class T> __global__ void InjectRiverGPU(Param XParam,River XRiver, T qnow, int* Riverblks, BlockP<T> XBlock, AdvanceP<T> XAdv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = Riverblks[ibl];




	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	T delta = calcres(T(XParam.dx), XBlock.level[ib]);
	T xl, yb, xr, yt, xllo, yllo;
	xllo = XParam.xo + XBlock.xo[ib];
	yllo = XParam.yo + XBlock.yo[ib];

	xl = xllo + ix * delta - 0.5 * delta;
	yb = yllo + iy * delta - 0.5 * delta;

	xr = xllo + ix * delta + 0.5 * delta;
	yt = yllo + iy * delta + 0.5 * delta;
	// the conditions are that the discharge area as defined by the user have to include at least a model grid node
	// This could be really annoying and there should be a better way to deal wiith this like polygon intersection
	//if (xx >= XForcing.rivers[Rin].xstart && xx <= XForcing.rivers[Rin].xend && yy >= XForcing.rivers[Rin].ystart && yy <= XForcing.rivers[Rin].yend)
	if (OBBdetect(xl, xr, yb, yt, T(XRiver.xstart), T(XRiver.xend), T(XRiver.ystart), T(XRiver.yend)))
	{

		XAdv.dh[i] += qnow  / XRiver.disarea;
		
	}



}
template __global__ void InjectRiverGPU<float>(Param XParam, River XRiver, float qnow, int* Riverblks, BlockP<float> XBlock, AdvanceP<float> XAdv);
template __global__ void InjectRiverGPU<double>(Param XParam, River XRiver, double qnow, int* Riverblks, BlockP<double> XBlock, AdvanceP<double> XAdv);

template <class T> __host__ void InjectRiverCPU(Param XParam, River XRiver, T qnow, int nblkriver, int* Riverblks, BlockP<T> XBlock, AdvanceP<T> XAdv)
{
	unsigned int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	T xllo, yllo, xl, yb, xr, yt, levdx;

	for (int ibl = 0; ibl < nblkriver; ibl++)
	{
		ib = Riverblks[ibl];

		levdx = calcres(T(XParam.dx), XBlock.level[ib]);

		xllo = T(XParam.xo + XBlock.xo[ib]);
		yllo = T(XParam.yo + XBlock.yo[ib]);



		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				//T delta = calcres(T(XParam.dx), XBlock.level[ib]);

				//T x = XParam.xo + XBlock.xo[ib] + ix * delta;
				//T y = XParam.yo + XBlock.yo[ib] + iy * delta;

				//if (x >= XRiver.xstart && x <= XRiver.xend && y >= XRiver.ystart && y <= XRiver.yend)
				xl = xllo + ix * levdx - T(0.5) * levdx;
				yb = yllo + iy * levdx - T(0.5) * levdx;

				xr = xllo + ix * levdx + T(0.5) * levdx;
				yt = yllo + iy * levdx + T(0.5) * levdx;
				// the conditions are that the discharge area as defined by the user have to include at least a model grid node
				// This could be really annoying and there should be a better way to deal wiith this like polygon intersection
				//if (xx >= XForcing.rivers[Rin].xstart && xx <= XForcing.rivers[Rin].xend && yy >= XForcing.rivers[Rin].ystart && yy <= XForcing.rivers[Rin].yend)
				if (OBBdetect(xl, xr, yb, yt, T(XRiver.xstart),T(XRiver.xend), T(XRiver.ystart), T(XRiver.yend)))
				{
					XAdv.dh[i] += qnow / T(XRiver.disarea);

				}
			}
		}
	}


}
template __host__ void InjectRiverCPU<float>(Param XParam, River XRiver, float qnow, int nblkriver, int* Riverblks, BlockP<float> XBlock, AdvanceP<float> XAdv);
template __host__ void InjectRiverCPU<double>(Param XParam, River XRiver, double qnow, int nblkriver, int* Riverblks, BlockP<double> XBlock, AdvanceP<double> XAdv);

template <class T> __global__ void AddrainforcingGPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> Rain, AdvanceP<T> XAdv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	T delta = calcres(T(XParam.dx), XBlock.level[ib]);

	T Rainhh;

	T x = XParam.xo + XBlock.xo[ib] + ix * delta;
	T y = XParam.yo + XBlock.yo[ib] + iy * delta;
	if (Rain.uniform)
	{
		Rainhh = Rain.nowvalue;
	}
	else
	{
		Rainhh = T(interpDyn2BUQ(x, y, Rain.GPU));
	}


	Rainhh = Rainhh / T(1000.0) / T(3600.0); // convert from mm/hrs to m/s

	XAdv.dh[i] += Rainhh * XBlock.activeCell[i];
}
template __global__ void AddrainforcingGPU<float>(Param XParam, BlockP<float> XBlock, DynForcingP<float> Rain, AdvanceP<float> XAdv);
template __global__ void AddrainforcingGPU<double>(Param XParam, BlockP<double> XBlock, DynForcingP<float> Rain, AdvanceP<double> XAdv);


template <class T> __global__ void AddrainforcingImplicitGPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, DynForcingP<float> Rain, EvolvingP<T> XEv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	T delta = calcres(T(XParam.dx), XBlock.level[ib]);

	T Rainhh;

	T x = XParam.xo + XBlock.xo[ib] + ix * delta;
	T y = XParam.yo + XBlock.yo[ib] + iy * delta;
	if (Rain.uniform)
	{
		Rainhh = Rain.nowvalue;
	}
	else
	{
		Rainhh = T(interpDyn2BUQ(x, y, Rain.GPU));
	}

	

	Rainhh = max(Rainhh / T(1000.0) / T(3600.0) * XLoop.dt,T(0.0)); // convert from mm/hrs to m/s

	
	XEv.h[i] += Rainhh * XBlock.activeCell[i];
	XEv.zs[i] += Rainhh * XBlock.activeCell[i];

}
template __global__ void AddrainforcingImplicitGPU<float>(Param XParam, Loop<float> XLoop, BlockP<float> XBlock, DynForcingP<float> Rain, EvolvingP<float> XEv);
template __global__ void AddrainforcingImplicitGPU<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, DynForcingP<float> Rain, EvolvingP<double> XEv);


template <class T> __host__ void AddrainforcingCPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> Rain, AdvanceP<T> XAdv)
{
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				T delta = calcres(T(XParam.dx), XBlock.level[ib]);

				T Rainhh;

				T x = T(XParam.xo) + XBlock.xo[ib] + ix * delta;
				T y = T(XParam.yo) + XBlock.yo[ib] + iy * delta;

				if (Rain.uniform)
				{
					Rainhh = T(Rain.nowvalue);
				}
				else
				{
					Rainhh = interp2BUQ(x, y, Rain);
				}
				



				Rainhh = Rainhh / T(1000.0) / T(3600.0); // convert from mm/hrs to m/s

				

				XAdv.dh[i] += Rainhh * XBlock.activeCell[i];
			}
		}
	}
}
template __host__ void AddrainforcingCPU<float>(Param XParam, BlockP<float> XBlock, DynForcingP<float> Rain, AdvanceP<float> XAdv);
template __host__ void AddrainforcingCPU<double>(Param XParam, BlockP<double> XBlock, DynForcingP<float> Rain, AdvanceP<double> XAdv);

template <class T> __host__ void AddrainforcingImplicitCPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, DynForcingP<float> Rain, EvolvingP<T> XEv)
{
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				T delta = calcres(T(XParam.dx), XBlock.level[ib]);

				T Rainhh;

				T x = T(XParam.xo) + XBlock.xo[ib] + ix * delta;
				T y = T(XParam.yo) + XBlock.yo[ib] + iy * delta;

				if (Rain.uniform)
				{
					Rainhh = T(Rain.nowvalue);
				}
				else
				{
					Rainhh = interp2BUQ(x, y, Rain);
				}


				Rainhh = max(Rainhh / T(1000.0) / T(3600.0) * T(XLoop.dt), T(0.0)); // convert from mm/hrs to m/s

				XEv.h[i] += Rainhh * XBlock.activeCell[i];
				XEv.zs[i] += Rainhh * XBlock.activeCell[i];
			}
		}
	}
}
template __host__ void AddrainforcingImplicitCPU<float>(Param XParam, Loop<float> XLoop, BlockP<float> XBlock, DynForcingP<float> Rain, EvolvingP<float> XEv);
template __host__ void AddrainforcingImplicitCPU<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, DynForcingP<float> Rain, EvolvingP<double> XEv);

template <class T> __host__ void AddinfiltrationImplicitCPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, T* il, T* cl, EvolvingP<T> XEv, T* hgw)
{
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;
	int p = 0;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				T waterOut = XEv.h[i];
				T infiltrationLoc = 0.0;
				T availinitialinfiltration;

				if (waterOut > 0)
				{
					//Computation of the initial loss
					availinitialinfiltration = il[i] / T(1000.0) - hgw[i];
					infiltrationLoc = min(waterOut, availinitialinfiltration);
					waterOut -= infiltrationLoc;

					//Computation of the continuous loss
					T continuousloss = cl[i] / T(1000.0) / T(3600.0) * T(XLoop.dt); //convert from mm/hs to m/s
					infiltrationLoc += min(continuousloss, waterOut);

					hgw[i] += infiltrationLoc;

				}

				XEv.h[i] -= max(infiltrationLoc * XBlock.activeCell[i],T(0.0));
				XEv.zs[i] -= max(infiltrationLoc * XBlock.activeCell[i],T(0.0));
			}
		}
	}
}
template __host__ void AddinfiltrationImplicitCPU<float>(Param XParam, Loop<float> XLoop, BlockP<float> XBlock, float* il, float* cl, EvolvingP<float> XEv, float* hgw);
template __host__ void AddinfiltrationImplicitCPU<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, double* il, double* cl, EvolvingP<double> XEv, double* hgw);

template <class T> __global__ void AddinfiltrationImplicitGPU(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, T* il, T* cl, EvolvingP<T> XEv, T* hgw)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	T waterOut = XEv.h[i];
	T infiltrationLoc = 0.0;
	T availinitialinfiltration;

	if (waterOut > 0)
	{
		//Computation of the initial loss
		availinitialinfiltration = max(il[i] / T(1000.0) - hgw[i],T(0.0));
		infiltrationLoc = min(waterOut, availinitialinfiltration);
		waterOut -= infiltrationLoc;

		//Computation of the continuous loss
		T continuousloss = cl[i] / T(1000.0) / T(3600.0) * T(XLoop.dt); //convert from mm/hs to m
		infiltrationLoc += min(continuousloss, waterOut);
	}

	hgw[i] += infiltrationLoc;

	XEv.h[i] -= infiltrationLoc * XBlock.activeCell[i];
	XEv.zs[i] -= infiltrationLoc * XBlock.activeCell[i];

}
template __global__ void AddinfiltrationImplicitGPU<float>(Param XParam, Loop<float> XLoop, BlockP<float> XBlock, float* il, float* cl, EvolvingP<float> XEv, float* hgw);
template __global__ void AddinfiltrationImplicitGPU<double>(Param XParam, Loop<double> XLoop, BlockP<double> XBlock, double* il, double* cl, EvolvingP<double> XEv, double* hgw);



template <class T> __global__ void AddwindforcingGPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> Uwind, DynForcingP<float> Vwind, AdvanceP<T> XAdv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	T delta = calcres(T(XParam.dx), XBlock.level[ib]);

	T uwindi, vwindi;

	T x = XParam.xo + XBlock.xo[ib] + ix * delta;
	T y = XParam.yo + XBlock.yo[ib] + iy * delta;

	T rhoairrhowater = T(0.00121951); // density ratio rho(air)/rho(water) 

	if (Uwind.uniform)
	{
		uwindi = T(Uwind.nowvalue);
	}
	else
	{
		uwindi = interpDyn2BUQ(x, y, Uwind.GPU);
	}
	if (Vwind.uniform)
	{
		vwindi = T(Vwind.nowvalue);
	}
	else
	{
		vwindi = interpDyn2BUQ(x, y, Vwind.GPU);
	}

	XAdv.dhu[i] += rhoairrhowater * T(XParam.Cd) * uwindi * abs(uwindi);
	XAdv.dhv[i] += rhoairrhowater * T(XParam.Cd) * vwindi * abs(vwindi);

	//Rainhh = Rainhh / T(1000.0) / T(3600.0); // convert from mm/hrs to m/s

	//XAdv.dh[i] += Rainhh;
}
template __global__ void AddwindforcingGPU<float>(Param XParam, BlockP<float> XBlock, DynForcingP<float> Uwind, DynForcingP<float> Vwind, AdvanceP<float> XAdv);
template __global__ void AddwindforcingGPU<double>(Param XParam, BlockP<double> XBlock, DynForcingP<float> Uwind, DynForcingP<float> Vwind, AdvanceP<double> XAdv);

template <class T> __global__ void AddPatmforcingGPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> PAtm, Model<T> XModel)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	T delta = calcres(T(XParam.dx), XBlock.level[ib]);

	T atmpi;

	T x = XParam.xo + XBlock.xo[ib] + ix * delta;
	T y = XParam.yo + XBlock.yo[ib] + iy * delta;

	

	atmpi = interpDyn2BUQ(x, y, PAtm.GPU);
	

	XModel.Patm[i] = atmpi - XParam.Paref;
	
}
template __global__ void AddPatmforcingGPU<float>(Param XParam, BlockP<float> XBlock, DynForcingP<float> PAtm, Model<float> XModel);
template __global__ void AddPatmforcingGPU<double>(Param XParam, BlockP<double> XBlock, DynForcingP<float> PAtm, Model<double> XModel);


template <class T> __host__ void AddwindforcingCPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> Uwind, DynForcingP<float> Vwind, AdvanceP<T> XAdv)
{
	//
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				T delta = calcres(T(XParam.dx), XBlock.level[ib]);
				T uwindi, vwindi;

				T x = T(XParam.xo) + XBlock.xo[ib] + ix * delta;
				T y = T(XParam.yo) + XBlock.yo[ib] + iy * delta;

				T rhoairrhowater = T(0.00121951); // density ratio rho(air)/rho(water) 
				if (Uwind.uniform)
				{
					uwindi = T(Uwind.nowvalue);
				}
				else
				{
					uwindi = interp2BUQ(x, y, Uwind);
				}
				if (Vwind.uniform)
				{
					vwindi = T(Vwind.nowvalue);
				}
				else
				{
					vwindi = interp2BUQ(x, y, Vwind);
				}

				XAdv.dhu[i] += rhoairrhowater * T(XParam.Cd) * uwindi * abs(uwindi);
				XAdv.dhv[i] += rhoairrhowater * T(XParam.Cd) * vwindi * abs(vwindi);

			}
		}
	}
}
template __host__ void AddwindforcingCPU<float>(Param XParam, BlockP<float> XBlock, DynForcingP<float> Uwind, DynForcingP<float> Vwind, AdvanceP<float> XAdv);
template __host__ void AddwindforcingCPU<double>(Param XParam, BlockP<double> XBlock, DynForcingP<float> Uwind, DynForcingP<float> Vwind, AdvanceP<double> XAdv);



template <class T> __host__ void AddPatmforcingCPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> PAtm, Model<T> XModel)
{
	//
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				T delta = calcres(T(XParam.dx), XBlock.level[ib]);
				T atmpi;

				T x = XParam.xo + XBlock.xo[ib] + ix * delta;
				T y = XParam.yo + XBlock.yo[ib] + iy * delta;

				
				if (PAtm.uniform)
				{
					atmpi = T(PAtm.nowvalue);
				}
				else
				{
					atmpi = interp2BUQ(x, y, PAtm);
				}
				

				XModel.Patm[i] = atmpi;
				

			}
		}
	}
}
template __host__ void AddPatmforcingCPU<float>(Param XParam, BlockP<float> XBlock, DynForcingP<float> PAtm, Model<float> XModel);
template __host__ void AddPatmforcingCPU<double>(Param XParam, BlockP<double> XBlock, DynForcingP<float> PAtm, Model<double> XModel);




template <class T> __device__ T interpDyn2BUQ(T x, T y, TexSetP Forcing)
{
	T read;
	if (Forcing.uniform)
	{
		read = T(Forcing.nowvalue);
	}
	else
	{
		read = interp2BUQ(x, y, Forcing);
	}
	return read;
}
template __device__ float interpDyn2BUQ<float>(float x, float y, TexSetP Forcing);
template __device__ double interpDyn2BUQ<double>(double x, double y, TexSetP Forcing);


template <class T> __device__ T interp2BUQ(T x, T y, TexSetP Forcing)
{
	T read;
	
	float ivw = float((x - T(Forcing.xo)) / T(Forcing.dx) + T(0.5));
	float jvw = float((y - T(Forcing.yo)) / T(Forcing.dy) + T(0.5));
	read = tex2D<float>(Forcing.tex, ivw, jvw);
	
	return read;
}
template __device__ float interp2BUQ<float>(float x, float y, TexSetP Forcing);
template __device__ double interp2BUQ<double>(double x, double y, TexSetP Forcing);


template <class T> void deformstep(Param XParam, Loop<T> XLoop, std::vector<deformmap<float>> deform, Model<T> XModel, Model<T> XModel_g)
{
	if (XParam.GPUDEVICE < 0)
	{
		deformstep(XParam, XLoop, deform, XModel);
		InitzbgradientCPU(XParam, XModel); // need to recalculate the zb halo and gradients to avoid blow up in topographic terms
	}
	else
	{
		deformstep(XParam, XLoop, deform, XModel_g);
		InitzbgradientGPU(XParam, XModel_g);
	}
}
template void deformstep<float>(Param XParam, Loop<float> XLoop, std::vector<deformmap<float>> deform, Model<float> XModel, Model<float> XModel_g);
template void deformstep<double>(Param XParam, Loop<double> XLoop, std::vector<deformmap<float>> deform, Model<double> XModel, Model<double> XModel_g);

template <class T> void deformstep(Param XParam, Loop<T> XLoop, std::vector<deformmap<float>> deform, Model<T> XModel)
{
	dim3 gridDim(XParam.nblk, 1, 1);
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);

	bool updatezbhalo = false;

	for (int nd = 0; nd < deform.size(); nd++)
	{
		// if deformation happend in the last computational step
		if (((deform[nd].startime + deform[nd].duration) >= (XLoop.totaltime - XLoop.dt)) && (deform[nd].startime < XLoop.totaltime))
		{
			
			updatezbhalo = true;

			T dtdef = min(XLoop.dt, XLoop.totaltime - deform[nd].startime);
			if (XLoop.totaltime > deform[nd].startime + deform[nd].duration)
			{
				dtdef = (T)min(XLoop.dt, XLoop.totaltime - (deform[nd].startime + deform[nd].duration));
			}
				

			T scale = (deform[nd].duration > 0.0) ? T(1.0 / deform[nd].duration * dtdef) : T(1.0);

			//log("Applying deform: " + std::to_string(scale));

			if (XParam.GPUDEVICE < 0)
			{
				AddDeformCPU(XParam, XModel.blocks, deform[nd], scale, XModel.evolv.zs, XModel.zb);
			}
			else
			{
				AddDeformGPU <<<gridDim, blockDim, 0 >>> (XParam, XModel.blocks, deform[nd], scale, XModel.evolv.zs, XModel.zb);
				CUDA_CHECK(cudaDeviceSynchronize());
			}


		}

	}
	//Redo the halo if needed
	if (updatezbhalo)
	{
		
		if (XParam.GPUDEVICE >= 0)
		{
			CUDA_CHECK(cudaStreamCreate(&XLoop.streams[0]));
			fillHaloGPU(XParam, XModel.blocks, XLoop.streams[0], XModel.zb);

			cudaStreamDestroy(XLoop.streams[0]);
		}
		else
		{
			fillHaloC(XParam, XModel.blocks, XModel.zb);
		}
	}


}


template <class T> __global__ void AddDeformGPU(Param XParam, BlockP<T> XBlock, deformmap<float> defmap, T scale, T* zs, T* zb)
{
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];
	int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

	T zss, zbb;
	T def;
	T delta = calcres(T(XParam.dx), XBlock.level[ib]);

	T x = XParam.xo + XBlock.xo[ib] + ix * delta;
	T y = XParam.yo + XBlock.yo[ib] + iy * delta;

	def= interpDyn2BUQ(x, y, defmap.GPU);

	//if (x > 42000 && x < 43000 && y>7719000 && y < 7721000)
	//{
	//	printf("x=%f, y=%f, def=%f\n ", x, y, def);
	//}

	zss = zs[i] + def * scale;
	if (defmap.iscavity == true)
	{
		zbb = min(zss, zb[i]);
	}
	else
	{
		zbb = zb[i] + def * scale;
	}

	zs[i] = zss;
	zb[i] = zbb;

	//zs[i] = zs[i] + def * scale;
	//zb[i] = zb[i] + def * scale;



}

template <class T> __host__ void AddDeformCPU(Param XParam, BlockP<T> XBlock, deformmap<float> defmap, T scale, T* zs, T* zb)
{
	int ib;
	
	T zbb,zss;

	T def;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
				T delta = calcres(T(XParam.dx), XBlock.level[ib]);
				

				T x = T(XParam.xo) + XBlock.xo[ib] + ix * delta;
				T y = T(XParam.yo) + XBlock.yo[ib] + iy * delta;

				def = interp2BUQ(x, y, defmap);

				zss = zs[i] + def * scale;
				if (defmap.iscavity == true)
				{
					zbb = min(zss, zb[i]);
				}
				else
				{
					zbb = zb[i] + def * scale;
				}

				zs[i] = zss;
				zb[i] = zbb;
			}
		}
	}

}




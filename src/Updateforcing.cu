

#include "Updateforcing.h"

template <class T> void updateforcing(Param XParam, Loop<T> XLoop, Forcing<float> &XForcing)
{
	// Update forcing for all possible dynamic forcing. 
	//if a file is declared that implies that the dynamic forcing is applicable
	if (!XForcing.Rain.inputfile.empty())
	{
		Forcingthisstep(XParam, XLoop, XForcing.Rain);
	}
	if (!XForcing.Atmp.inputfile.empty())
	{
		Forcingthisstep(XParam, XLoop, XForcing.Atmp);
	}
	if (!XForcing.UWind.inputfile.empty())//&& !XForcing.UWind.inputfile.empty()
	{
		Forcingthisstep(XParam, XLoop, XForcing.UWind);
		Forcingthisstep(XParam, XLoop, XForcing.VWind);
	}

	
}
template void updateforcing<float>(Param XParam, Loop<float> XLoop, Forcing<float>& XForcing);
template void updateforcing<double>(Param XParam, Loop<double> XLoop, Forcing<float>& XForcing);




template <class T> void Forcingthisstep(Param XParam, Loop<T> XLoop, DynForcingP<float> &XDynForcing)
{
	dim3 blockDimDF(16, 16, 1);
	dim3 gridDimDF((int)ceil((float)XDynForcing.nx / (float)blockDimDF.x), (int)ceil((float)XDynForcing.ny / (float)blockDimDF.y), 1);



	if (XDynForcing.uniform == 1)
	{
		//
		int Rstepinbnd = 1;

		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = XDynForcing.unidata[Rstepinbnd].time - XLoop.totaltime;

		while (difft < 0.0)
		{
			Rstepinbnd++;
			difft = XDynForcing.unidata[Rstepinbnd].time - XParam.totaltime;
		}

		XDynForcing.nowvalue = T(interptime(XDynForcing.unidata[Rstepinbnd].wspeed, XDynForcing.unidata[Rstepinbnd - 1].wspeed, XDynForcing.unidata[Rstepinbnd].time - XDynForcing.unidata[Rstepinbnd - 1].time, XLoop.totaltime - XDynForcing.unidata[Rstepinbnd - 1].time));



	}
	else
	{
		int readfirststep = min(max((int)floor((XLoop.totaltime - XDynForcing.to) / XDynForcing.dt), 0), XDynForcing.nt - 2);

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

			readvardata(XDynForcing.inputfile, XDynForcing.varname, readfirststep + 1, XDynForcing.after);
			if (XParam.GPUDEVICE >= 0)
			{
				CUDA_CHECK(cudaMemcpy(XDynForcing.after_g, XDynForcing.after, XDynForcing.nx * XDynForcing.ny * sizeof(float), cudaMemcpyHostToDevice));
			}

			XDynForcing.instep = readfirststep + 1;
		}

		// Interpolate the forcing array to this time 
		if (XParam.GPUDEVICE >= 0)
		{
			InterpstepGPU << <gridDimDF, blockDimDF, 0 >> > (XDynForcing.nx, XDynForcing.ny, XDynForcing.instep - 1, float(XLoop.totaltime), float(XDynForcing.dt), XDynForcing.now_g, XDynForcing.before_g, XDynForcing.after_g);
			CUDA_CHECK(cudaDeviceSynchronize());

			CUDA_CHECK(cudaMemcpyToArray(XDynForcing.GPU.CudArr, 0, 0, XDynForcing.now_g, XDynForcing.nx * XDynForcing.ny * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		else
		{
			InterpstepCPU(XDynForcing.nx, XDynForcing.ny, XDynForcing.instep - 1, XLoop.totaltime, XDynForcing.dt, XDynForcing.now, XDynForcing.before, XDynForcing.after);
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
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = Riverblks[ibl];




	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	T delta = calcres(T(XParam.dx), XBlock.level[ib]);

	T x = XParam.xo + XBlock.xo[ib] + ix * delta;
	T y = XParam.yo + XBlock.yo[ib] + iy * delta;

	if (x >= XRiver.xstart && x <= XRiver.xend && y >= XRiver.ystart && y <= XRiver.yend)
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

	for (int ibl = 0; ibl < nblkriver; ibl++)
	{
		ib = Riverblks[ibl];

		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				T delta = calcres(T(XParam.dx), XBlock.level[ib]);

				T Rainhh;

				T x = XParam.xo + XBlock.xo[ib] + ix * delta;
				T y = XParam.yo + XBlock.yo[ib] + iy * delta;

				if (x >= XRiver.xstart && x <= XRiver.xend && y >= XRiver.ystart && y <= XRiver.yend)
				{
					XAdv.dh[i] += qnow / XRiver.disarea;

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
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	T delta = calcres(T(XParam.dx), XBlock.level[ib]);

	T Rainhh;

	T x = XParam.xo + XBlock.xo[ib] + ix * delta;
	T y = XParam.yo + XBlock.yo[ib] + iy * delta;

	Rainhh = T(interpDyn2BUQ(x, y, Rain.GPU));



	Rainhh = Rainhh / T(1000.0) / T(3600.0); // convert from mm/hrs to m/s

	XAdv.dh[i] += Rainhh;
}
template __global__ void AddrainforcingGPU<float>(Param XParam, BlockP<float> XBlock, DynForcingP<float> Rain, AdvanceP<float> XAdv);
template __global__ void AddrainforcingGPU<double>(Param XParam, BlockP<double> XBlock, DynForcingP<float> Rain, AdvanceP<double> XAdv);


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

				T x = XParam.xo + XBlock.xo[ib] + ix * delta;
				T y = XParam.yo + XBlock.yo[ib] + iy * delta;

				if (Rain.uniform)
				{
					Rainhh = Rain.nowvalue;
				}
				else
				{
					Rainhh = interp2BUQ(x, y, Rain);
				}
				



				Rainhh = Rainhh / T(1000.0) / T(3600.0); // convert from mm/hrs to m/s

				XAdv.dh[i] += Rainhh;
			}
		}
	}
}
template __host__ void AddrainforcingCPU<float>(Param XParam, BlockP<float> XBlock, DynForcingP<float> Rain, AdvanceP<float> XAdv);
template __host__ void AddrainforcingCPU<double>(Param XParam, BlockP<double> XBlock, DynForcingP<float> Rain, AdvanceP<double> XAdv);


template <class T> __global__ void AddwindforcingGPU(Param XParam, BlockP<T> XBlock, DynForcingP<float> Uwind, DynForcingP<float> Vwind, AdvanceP<T> XAdv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
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

	uwindi = interpDyn2BUQ(x, y, Uwind.GPU);
	vwindi = interpDyn2BUQ(x, y, Vwind.GPU);

	XAdv.dhu[i] += rhoairrhowater * T(XParam.Cd) * uwindi * abs(uwindi);
	XAdv.dhv[i] += rhoairrhowater * T(XParam.Cd) * vwindi * abs(vwindi);

	//Rainhh = Rainhh / T(1000.0) / T(3600.0); // convert from mm/hrs to m/s

	//XAdv.dh[i] += Rainhh;
}
template __global__ void AddwindforcingGPU<float>(Param XParam, BlockP<float> XBlock, DynForcingP<float> Uwind, DynForcingP<float> Vwind, AdvanceP<float> XAdv);
template __global__ void AddwindforcingGPU<double>(Param XParam, BlockP<double> XBlock, DynForcingP<float> Uwind, DynForcingP<float> Vwind, AdvanceP<double> XAdv);



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

				T x = XParam.xo + XBlock.xo[ib] + ix * delta;
				T y = XParam.yo + XBlock.yo[ib] + iy * delta;

				T rhoairrhowater = T(0.00121951); // density ratio rho(air)/rho(water) 
				if (Uwind.uniform)
				{
					uwindi = Uwind.nowvalue;
				}
				else
				{
					uwindi = interp2BUQ(x, y, Uwind);
				}
				if (Vwind.uniform)
				{
					vwindi = Vwind.nowvalue;
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
	float jvw = float((y - T(Forcing.yo)) / T(Forcing.dx) + T(0.5));
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
	}
	else
	{
		deformstep(XParam, XLoop, deform, XModel_g);
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
		//
		if ((XLoop.totaltime - deform[nd].startime) <= XLoop.dt && (XLoop.totaltime - deform[nd].startime) > 0.0)
		{
			updatezbhalo = true;

			T scale = (deform[nd].duration > 0.0) ? T(1.0 / deform[nd].duration * (XLoop.totaltime - deform[nd].startime)) : T(1.0);

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

	T def;
	T delta = calcres(T(XParam.dx), XBlock.level[ib]);

	T x = XParam.xo + XBlock.xo[ib] + ix * delta;
	T y = XParam.yo + XBlock.yo[ib] + iy * delta;

	def= interpDyn2BUQ(x, y, defmap.GPU);

	zs[i] = zs[i] + def * scale;
	zb[i] = zb[i] + def * scale;



}

template <class T> __host__ void AddDeformCPU(Param XParam, BlockP<T> XBlock, deformmap<float> defmap, T scale, T* zs, T* zb)
{
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

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
				

				T x = XParam.xo + XBlock.xo[ib] + ix * delta;
				T y = XParam.yo + XBlock.yo[ib] + iy * delta;

				def = interp2BUQ(x, y, defmap);

				zs[i] = zs[i] + def * scale;
				zb[i] = zb[i] + def * scale;
			}
		}
	}

}




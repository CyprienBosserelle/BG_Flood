

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
			InterpstepCPU(XDynForcing.nx, XDynForcing.ny, XDynForcing.instep - 1, float(XLoop.totaltime), float(XDynForcing.dt), XDynForcing.now, XDynForcing.before, XDynForcing.after);
		}
		//InterpstepCPU(XParam.windU.nx, XParam.windU.ny, readfirststep, XParam.totaltime, XParam.windU.dt, Uwind, Uwbef, Uwaft);
		//InterpstepCPU(XParam.windV.nx, XParam.windV.ny, readfirststep, XParam.totaltime, XParam.windV.dt, Vwind, Vwbef, Vwaft);

		

	}

	//return rainuni;
}



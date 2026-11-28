#include "Friction.h"



/**
 * @brief CUDA kernel for applying bottom friction to all active blocks.
 *
 * Updates velocity components using the specified friction model (default, smart, or Manning) for each cell in all blocks.
 *
 * @tparam T Data type (float or double)
 * @param XParam Simulation parameters
 * @param XBlock Block data structure
 * @param dt Time step
 * @param cf Friction coefficient array
 * @param XEvolv Evolving variables structure
 */
template <class T> __global__ void bottomfrictionGPU(Param XParam, BlockP<T> XBlock, T dt, T* cf,EvolvingP<T> XEvolv)
{
	// Shear stress equation:
	// Taub=cf*rho*U*sqrt(U^2+V^2)
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	T eps = T(XParam.eps);
	T g = T(XParam.g);

	int frictionmodel = XParam.frictionmodel;

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);


	T normu, hi, ui, vi;
	
	

	hi = XEvolv.h[i];
	ui = XEvolv.u[i];
	vi = XEvolv.v[i];
	if (hi > eps)
	{
		normu = sqrt(ui * ui + vi * vi);
			
		T cfi;
		if (frictionmodel == 0)
		{
			cfi = cf[i];
		}
		else if (frictionmodel == 1)//Smart friction formulation
		{
			cfi = smartfriction(hi, cf[i]);

		}
		else if (frictionmodel == -1)// Manning friction formulation
		{
			cfi = manningfriction(g, hi, cf[i]);

		}

		T tb = cfi * normu / hi * dt;
		XEvolv.u[i] = ui / (T(1.0) + tb);
		XEvolv.v[i] = vi / (T(1.0) + tb);
	}

	

}
template __global__ void bottomfrictionGPU<float>(Param XParam, BlockP<float> XBlock,float dt, float* cf, EvolvingP<float> XEvolv);
template __global__ void bottomfrictionGPU<double>(Param XParam, BlockP<double> XBlock,double dt, double* cf, EvolvingP<double> XEvolv);


/**
 * @brief Bottom friction and wave force on one cell: simple semi-implicit step (shared CPU/GPU).
 *
 * u^{n+1} = (u* + dt Fx/(rho h)) / (1 + dt cf |u*| / h)   (same for v with Fy)
 *
 * The wave force (radiation stress gradient, from the wave model) is added as an acceleration Fx/(rho h) in the
 * implicit bottom friction step. Both terms scale with 1/h so the velocity stays bounded in very shallow water
 * (u tends to the balance between the wave force and the bottom friction).
 */
template <class T> __host__ __device__ void bottomfrictionWaveCell(const Param& XParam, int i, T dt, T* cf, T* Fx, T* Fy, const EvolvingP<T>& XEvolv)
{
	T eps = T(XParam.eps);
	T g = T(XParam.g);
	T hi = XEvolv.h[i];
	if (hi > eps)
	{
		T ui = XEvolv.u[i];
		T vi = XEvolv.v[i];
		T normu = sqrt(ui * ui + vi * vi);
		T cfi;
		if (XParam.frictionmodel == 1)
		{
			cfi = smartfriction(hi, cf[i]);
		}
		else if (XParam.frictionmodel == -1)
		{
			cfi = manningfriction(g, hi, cf[i]);
		}
		else
		{
			cfi = cf[i];
		}
		T tb = cfi * normu / hi * dt;
		T acc = dt / (T(XParam.rho) * hi);
		XEvolv.u[i] = (ui + acc * Fx[i]) / (T(1.0) + tb);
		XEvolv.v[i] = (vi + acc * Fy[i]) / (T(1.0) + tb);
	}
}
template __host__ __device__ void bottomfrictionWaveCell<float>(const Param& XParam, int i, float dt, float* cf, float* Fx, float* Fy, const EvolvingP<float>& XEvolv);
template __host__ __device__ void bottomfrictionWaveCell<double>(const Param& XParam, int i, double dt, double* cf, double* Fx, double* Fy, const EvolvingP<double>& XEvolv);

/**
 * @brief Bottom friction with the wave force (GPU). See bottomfrictionWaveCell.
 */
template <class T> __global__ void bottomfrictionWaveGPU(Param XParam, BlockP<T> XBlock, T dt, T* cf, T* Fx, T* Fy, EvolvingP<T> XEvolv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int ib = XBlock.active[blockIdx.x];
	int i = memloc(halowidth, blkmemwidth, threadIdx.x, threadIdx.y, ib);
	bottomfrictionWaveCell(XParam, i, dt, cf, Fx, Fy, XEvolv);
}
template __global__ void bottomfrictionWaveGPU<float>(Param XParam, BlockP<float> XBlock, float dt, float* cf, float* Fx, float* Fy, EvolvingP<float> XEvolv);
template __global__ void bottomfrictionWaveGPU<double>(Param XParam, BlockP<double> XBlock, double dt, double* cf, double* Fx, double* Fy, EvolvingP<double> XEvolv);

/**
 * @brief Bottom friction with the wave force (CPU). See bottomfrictionWaveCell.
 */
template <class T> __host__ void bottomfrictionWaveCPU(Param XParam, BlockP<T> XBlock, T dt, T* cf, T* Fx, T* Fy, EvolvingP<T> XEvolv)
{
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				bottomfrictionWaveCell(XParam, memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib), dt, cf, Fx, Fy, XEvolv);
			}
		}
	}
}
template __host__ void bottomfrictionWaveCPU<float>(Param XParam, BlockP<float> XBlock, float dt, float* cf, float* Fx, float* Fy, EvolvingP<float> XEvolv);
template __host__ void bottomfrictionWaveCPU<double>(Param XParam, BlockP<double> XBlock, double dt, double* cf, double* Fx, double* Fy, EvolvingP<double> XEvolv);



/**
 * @brief CPU routine for applying bottom friction to all active blocks.
 *
 * Updates velocity components using the specified friction model (default, smart, or Manning) for each cell in all blocks.
 *
 * @tparam T Data type (float or double)
 * @param XParam Simulation parameters
 * @param XBlock Block data structure
 * @param dt Time step
 * @param cf Friction coefficient array
 * @param XEvolv Evolving variables structure
 */
template <class T> __host__ void bottomfrictionCPU(Param XParam, BlockP<T> XBlock,T dt, T* cf, EvolvingP<T> XEvolv)
{
	T eps = T(XParam.eps);
	T g = T(XParam.g);

	

	T hi, ui, vi,normu;

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

				hi = XEvolv.h[i];
				ui = XEvolv.u[i];
				vi = XEvolv.v[i];
				if (hi > eps)
				{
					normu = sqrt(ui * ui + vi * vi);

					T cfi;
					if (XParam.frictionmodel == 0)
					{
						cfi = cf[i];
					}
					else if (XParam.frictionmodel == 1)//Smart friction formulation
					{
						
						cfi = smartfriction(hi, cf[i]);

					}
					else if (XParam.frictionmodel == -1)// Manning friction formulation
					{
						T n = cf[i];
						cfi = manningfriction(g, hi, n);


					}

					T tb = cfi * normu / hi * dt;
					XEvolv.u[i] = ui / (T(1.0) + tb);
					XEvolv.v[i] = vi / (T(1.0) + tb);
				}
			}
		}
	}


}
template __host__ void bottomfrictionCPU<float>(Param XParam, BlockP<float> XBlock,float dt, float* cf, EvolvingP<float> XEvolv);
template __host__ void bottomfrictionCPU<double>(Param XParam, BlockP<double> XBlock,double dt, double* cf, EvolvingP<double> XEvolv);

/**
 * @brief CPU routine for applying bottom friction following Xia & Lang (2018).
 *
 * Updates velocity components using the Xia & Lang friction model for each cell in all blocks, using both current and previous evolving variables.
 * Reference: Xia and Lang (2018), https://doi.org/10.1016/j.advwatres.2018.05.004
 *
 * @tparam T Data type (float or double)
 * @param XParam Simulation parameters
 * @param XBlock Block data structure
 * @param dt Time step
 * @param cf Friction coefficient array
 * @param XEvolv Current evolving variables structure
 * @param XEvolv_o Previous evolving variables structure
 */
template <class T> __host__ void XiafrictionCPU(Param XParam, BlockP<T> XBlock, T dt, T* cf, EvolvingP<T> XEvolv, EvolvingP<T> XEvolv_o)
{
	T eps = T(XParam.eps);
	T g = T(XParam.g);



	T hi, ho, ui, vi, normu;

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
				ho = XEvolv.h[i];
				hi = XEvolv_o.h[i];
				ui = XEvolv_o.u[i];
				vi = XEvolv_o.v[i];
				if (hi > eps)
				{
					normu = sqrt(ui * ui + vi * vi);

					T cfi = cf[i]; //if (XParam.frictionmodel == 0)
					if (XParam.frictionmodel == 1)//Smart friction formulation
					{

						cfi = smartfriction(hi, cf[i]);

					}
					else if (XParam.frictionmodel == -1)// Manning friction formulation
					{
						T n = cf[i];
						cfi = manningfriction(g, hi, n);


					}

					T tb = cfi * normu * hi/(ho*ho) * dt;
					if (tb >= T(1e-10))
					{
						XEvolv_o.u[i] = (ui - ui * sqrt(T(1.0) + T(4.0) * tb)) / (T(-2.0) * tb);
						XEvolv_o.v[i] = (vi - vi * sqrt(T(1.0) + T(4.0) * tb)) / (T(-2.0) * tb);
					}
					
				}
			}
		}
	}


}

template __host__ void XiafrictionCPU<float>(Param XParam, BlockP<float> XBlock,float dt, float* cf, EvolvingP<float> XEvolv, EvolvingP<float> XEvolv_o);
template __host__ void XiafrictionCPU<double>(Param XParam, BlockP<double> XBlock,double dt, double* cf, EvolvingP<double> XEvolv, EvolvingP<double> XEvolv_o);

/**
 * @brief CUDA kernel for applying Xia & Lang (2018) bottom friction to all active blocks.
 *
 * Updates velocity components using the Xia & Lang friction model for each cell in all blocks, using both current and previous evolving variables.
 *
 * @tparam T Data type (float or double)
 * @param XParam Simulation parameters
 * @param XBlock Block data structure
 * @param dt Time step
 * @param cf Friction coefficient array
 * @param XEvolv Current evolving variables structure
 * @param XEvolv_o Previous evolving variables structure
 */
template <class T> __global__ void XiafrictionGPU(Param XParam, BlockP<T> XBlock, T dt, T* cf, EvolvingP<T> XEvolv, EvolvingP<T> XEvolv_o)
{
	// Shear stress equation:
	// Taub=cf*rho*U*sqrt(U^2+V^2)
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	T eps = T(XParam.eps);
	T g = T(XParam.g);

	//int frictionmodel = XParam.frictionmodel;

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);


	T normu,ho, hi, ui, vi;



	
	ho = XEvolv.h[i];
	hi = XEvolv_o.h[i];
	ui = XEvolv_o.u[i];
	vi = XEvolv_o.v[i];
	if (hi > eps) //SHould this be both ho and hi >eps ?
	{
		normu = sqrt(ui * ui + vi * vi);

		T cfi = cf[i]; //if (XParam.frictionmodel == 0)
		if (XParam.frictionmodel == 1)//Smart friction formulation
		{

			cfi = smartfriction(hi, cf[i]);

		}
		else if (XParam.frictionmodel == -1)// Manning friction formulation
		{
			T n = cf[i];
			cfi = manningfriction(g, hi, n);


		}

		T tb = cfi * normu * hi / (ho * ho) * dt;
		if (tb >= T(1e-10))
		{
			XEvolv_o.u[i] = (ui - ui * sqrt(T(1.0) + T(4.0) * tb)) / (T(-2.0) * tb);
			XEvolv_o.v[i] = (vi - vi * sqrt(T(1.0) + T(4.0) * tb)) / (T(-2.0) * tb);
		}

	}



}
template __global__ void XiafrictionGPU<float>(Param XParam, BlockP<float> XBlock, float dt, float* cf, EvolvingP<float> XEvolv, EvolvingP<float> XEvolv_o);
template __global__ void XiafrictionGPU<double>(Param XParam, BlockP<double> XBlock, double dt, double* cf, EvolvingP<double> XEvolv, EvolvingP<double> XEvolv_o);


/**
 * @brief Smart friction model for roughness height.
 *
 * Computes friction coefficient using a log-law based on water depth and roughness height.
 *
 * @tparam T Data type (float or double)
 * @param hi Water depth
 * @param zo Roughness height
 * @return Friction coefficient
 */
template <class T> __host__ __device__ T smartfriction(T hi,T zo)
{
	T cfi;
	T ee = T(2.71828182845905);

	T Hbar = hi / zo;
	if (Hbar <= ee)
	{
		cfi = T(1.0) / (T(0.46) * Hbar);
	}
	else
	{
		cfi = T(1.0) / (T(2.5) * (log(Hbar) - T(1.0) + T(1.359) / Hbar));
	}
	cfi = cfi * cfi; //

	return cfi;
}

/**
 * @brief Manning friction model.
 *
 * Computes friction coefficient using Manning's equation based on gravity, water depth, and Manning's n.
 *
 * @tparam T Data type (float or double)
 * @param g Gravity
 * @param hi Water depth
 * @param n Manning's n
 * @return Friction coefficient
 */
template <class T> __host__ __device__ T manningfriction(T g, T hi, T n)
{
	T cfi= g * n * n / cbrt(hi);
	return cfi;
}


/**
 * @brief CUDA kernel for enforcing a velocity threshold.
 * Function Used to prevent crazy velocity on the GPU.
 * The function wraps the main function for the GPU.
 * Updates velocity components to ensure they do not exceed a specified threshold.
 * @param XParam Simulation parameters
 * @param XBlock Block data structure
 * @param XEvolv Evolving variables structure
 *
 */
template <class T> __global__ void TheresholdVelGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEvolv)
{
	
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	//unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	bool bustedThreshold = false;


	T ui, vi;

	
	ui = XEvolv.u[i];
	vi = XEvolv.v[i];

	bustedThreshold = ThresholdVelocity(T(XParam.VelThreshold), ui, vi);

	if (bustedThreshold)
	{
		XEvolv.u[i] = ui;
		XEvolv.v[i] = vi;
	}


	

}
template __global__ void TheresholdVelGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEvolv);
template __global__ void TheresholdVelGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEvolv);

/**
 * @brief CPU routine for enforcing a velocity threshold.
 *
 * Updates velocity components to ensure they do not exceed a specified threshold.
 * Function Used to prevent crazy velocity on the CPU
 * 
 * @param XParam Simulation parameters
 * @param XBlock Block data structure
 * @param XEvolv Evolving variables structure
 *
 */

template <class T> __host__ void TheresholdVelCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEvolv)
{

	T ui, vi;

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
				bool bustedThreshold = false;

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				ui = XEvolv.u[i];

				vi = XEvolv.v[i];

				bustedThreshold = ThresholdVelocity(T(XParam.VelThreshold), ui, vi);

				if (bustedThreshold)
				{
					log("Velocity Threshold exceeded!");
				}
				XEvolv.u[i] = ui;

				XEvolv.v[i] = vi;
			}
		}
	}
}
template __host__ void TheresholdVelCPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEvolv);
template __host__ void TheresholdVelCPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEvolv);

/** 
 * 
 * @brief Function Used to prevent crazy velocity
 * 
 * The function scale velocities so it doesn't exceeds a given threshold. 
 * Default threshold is/should be 16.0m/s
 * 
 * @param Threshold Velocity threshold
 * @param u Velocity component in x direction
 * @param v Velocity component in y direction
 * @return true if velocity was above threshold and has been scaled down, false otherwise
 * 
*/
template <class T> __host__ __device__ bool ThresholdVelocity(T Threshold, T& u, T& v)
{
	T normvel = sqrt(u * u + v * v);

	bool alert = normvel > Threshold;

	if (alert)
	{
		u /= normvel / Threshold;
		v /= normvel / Threshold;
	}
	return alert;
}



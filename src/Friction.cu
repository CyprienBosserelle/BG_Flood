#include "Friction.h"



template <class T> __global__ void bottomfrictionGPU(Param XParam, BlockP<T> XBlock, T dt, T* cf,EvolvingP<T> XEvolv)
{
	// Shear stress equation:
	// Taub=cf*rho*U*sqrt(U^2+V^2)
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	T eps = T(XParam.eps);
	T g = T(XParam.g);

	int smart = XParam.frictionmodel;

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);


	T normu, hi, ui, vi;
	
	

	hi = XEvolv.h[i];
	ui = XEvolv.u[i];
	vi = XEvolv.v[i];
	if (hi > eps)
	{
		normu = sqrt(ui * ui + vi * vi);
			
		T cfi;
		if (smart == 1)//Smart friction formulation
		{
			cfi = smartfriction(hi, cf[i]);

		}
		if (smart == -1)// Manning friction formulation
		{
			cfi = manningfriction(g, hi, cf[i]);


		}

		T tb = cfi * normu / hi * dt;
		XEvolv.u[i] = ui / (T(1.0) + tb);
		XEvolv.v[i] = vi / (T(1.0) + tb);
	}

	

}
template __global__ void bottomfrictionGPU<float>(Param XParam, BlockP<float> XBlock, float* cf, EvolvingP<float> XEvolv);
template __global__ void bottomfrictionGPU<double>(Param XParam, BlockP<double> XBlock, double* cf, EvolvingP<double> XEvolv);



template <class T> __host__ void bottomfrictionCPU(Param XParam, BlockP<T> XBlock,T dt, T* cf, EvolvingP<T> XEvolv)
{
	T eps = T(XParam.eps);
	T g = T(XParam.g);

	int smart = XParam.frictionmodel;

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
					if (smart == 1)//Smart friction formulation
					{
						
						cfi = smartfriction(hi, cf[i]);

					}
					if (smart == -1)// Manning friction formulation
					{
						T n = cfi;
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
template __host__ void bottomfrictionCPU<float>(Param XParam, BlockP<float> XBlock, float* cf, EvolvingP<float> XEvolv);
template __host__ void bottomfrictionCPU<double>(Param XParam, BlockP<double> XBlock, double* cf, EvolvingP<double> XEvolv);



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

template <class T> __host__ __device__ T manningfriction(T g, T hi, T n)
{
	T cfi= g * n * n / cbrt(hi);
	return cfi;
}

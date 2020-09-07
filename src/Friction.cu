#include "Friction.h"



template <class T> __global__ void bottomfriction(Param XParam, BlockP<T> XBlock, T* cf,EvolvingP<T> XEvolv)
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
	T ee = T(2.71828182845905);
	

	hi = XEvolv.h[i];
	ui = XEvolv.u[i];
	vi = XEvolv.v[i];
	if (hi > eps)
	{
		normu = sqrt(ui * ui + vi * vi);
			
		T cfi = cf[i];
		if (smart == 1)//Smart friction formulation
		{
			T zo = cfi;
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
		}
		if (smart == -1)// Manning friction formulation
		{
			T n = cfi;
			cfi = g * n * n / cbrt(hi);

		}

		T tb = cfi * normu / hi * dt;
		XEvolv.u[i] = ui / (T(1.0) + tb);
		XEvolv.v[i] = vi / (T(1.0) + tb);
	}

	

}
template __global__ void bottomfriction<float>(Param XParam, BlockP<float> XBlock, float* cf, EvolvingP<float> XEvolv);
template __global__ void bottomfriction<double>(Param XParam, BlockP<double> XBlock, double* cf, EvolvingP<double> XEvolv);



template <class T> __host__ void bottomfriction(Param XParam, BlockP<T> XBlock, T* cf, EvolvingP<T> XEvolv)
{
	T eps = T(XParam.eps);
	T delta;
	T g = T(XParam.g);


	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		delta = calcres(XParam.dx, XBlock.level[ib]);
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

					T cfi = cf[i];
					if (smart == 1)//Smart friction formulation
					{
						T zo = cfi;
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
					}
					if (smart == -1)// Manning friction formulation
					{
						T n = cfi;
						cfi = g * n * n / cbrt(hi);

					}

					T tb = cfi * normu / hi * dt;
					XEvolv.u[i] = ui / (T(1.0) + tb);
					XEvolv.v[i] = vi / (T(1.0) + tb);
				}
			}
		}
	}


}
template __host__ void bottomfriction<float>(Param XParam, BlockP<float> XBlock, float* cf, EvolvingP<float> XEvolv);
template __host__ void bottomfriction<double>(Param XParam, BlockP<double> XBlock, double* cf, EvolvingP<double> XEvolv);

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
template __host__ void bottomfrictionCPU<float>(Param XParam, BlockP<float> XBlock,float dt, float* cf, EvolvingP<float> XEvolv);
template __host__ void bottomfrictionCPU<double>(Param XParam, BlockP<double> XBlock,double dt, double* cf, EvolvingP<double> XEvolv);

/*!\fn void XiafrictionCPU(Param XParam, BlockP<T> XBlock, T dt, T* cf, EvolvingP<T> XEvolv)
* apply bottom friction follwing the procedure from Xia and Lang 2018
* https://doi.org/10.1016/j.advwatres.2018.05.004
* 
*
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
						T n = cfi;
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

template <class T> __global__ void XiafrictionGPU(Param XParam, BlockP<T> XBlock, T dt, T* cf, EvolvingP<T> XEvolv, EvolvingP<T> XEvolv_o)
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

	int frictionmodel = XParam.frictionmodel;

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
			T n = cfi;
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

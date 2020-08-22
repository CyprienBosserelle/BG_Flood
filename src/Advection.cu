#include "Advection.h"


template <class T>__global__ void updateEVGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, FluxP<T> XFlux, AdvanceP<T> XAdv)
{
	
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	T eps = T(XParam.eps);
	T delta = calcres(T(XParam.dx), XBlock.level[ib]);
	T g = T(XParam.g);
	

	T fc = (T)XParam.lat * pi / T(21600.0);

	int iright, itop;
	
	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	
	iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
	itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

	

	T cm = T(1.0);// 0.1;
	T fmu = T(1.0);
	T fmv = T(1.0);

	T hi = XEv.h[i];
	T uui = XEv.u[i];
	T vvi = XEv.v[i];


	T cmdinv, ga;

	cmdinv = T(1.0)/ (cm * delta);
	ga = T(0.5) * g;
		

	XAdv.dh[i] = T(-1.0) * (XFlux.Fhu[iright] - XFlux.Fhu[i] + XFlux.Fhv[itop] - XFlux.Fhv[i]) * cmdinv;
		


	//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
	//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
	T dmdl = (fmu - fmu) / (cm * delta);// absurd if not spherical!
	T dmdt = (fmv - fmv) / (cm * delta);
	T fG = vvi * dmdl - uui * dmdt;
	XAdv.dhu[i] = (XFlux.Fqux[i] + XFlux.Fquy[i] - XFlux.Su[iright] - XFlux.Fquy[itop]) * cmdinv + fc * hi * vvi;
	XAdv.dhv[i] = (XFlux.Fqvy[i] + XFlux.Fqvx[i] - XFlux.Sv[itop] - XFlux.Fqvx[iright]) * cmdinv - fc * hi * uui;
		
	XAdv.dhu[i] += hi * (ga * hi * dmdl + fG * vvi);// This term is == 0 so should be commented here
	XAdv.dhv[i] += hi * (ga * hi * dmdt - fG * uui);// Need double checking before doing that
	
}
template __global__ void updateEVGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, FluxP<float> XFlux, AdvanceP<float> XAdv);
template __global__ void updateEVGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, FluxP<double> XFlux, AdvanceP<double> XAdv);


template <class T>__host__ void updateEVCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, FluxP<T> XFlux, AdvanceP<T> XAdv)
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

				T fc = (T)XParam.lat * pi / T(21600.0);

				int iright, itop;

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
				itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);



				T cm = T(1.0);// 0.1;
				T fmu = T(1.0);
				T fmv = T(1.0);

				T hi = XEv.h[i];
				T uui = XEv.u[i];
				T vvi = XEv.v[i];


				T cmdinv, ga;

				cmdinv = T(1.0) / (cm * delta);
				ga = T(0.5) * g;


				XAdv.dh[i] = T(-1.0) * (XFlux.Fhu[iright] - XFlux.Fhu[i] + XFlux.Fhv[itop] - XFlux.Fhv[i]) * cmdinv;



				//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
				//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
				T dmdl = (fmu - fmu) / (cm * delta);// absurd if not spherical!
				T dmdt = (fmv - fmv) / (cm * delta);
				T fG = vvi * dmdl - uui * dmdt;
				XAdv.dhu[i] = (XFlux.Fqux[i] + XFlux.Fquy[i] - XFlux.Su[iright] - XFlux.Fquy[itop]) * cmdinv + fc * hi * vvi;
				XAdv.dhv[i] = (XFlux.Fqvy[i] + XFlux.Fqvx[i] - XFlux.Sv[itop] - XFlux.Fqvx[iright]) * cmdinv - fc * hi * uui;

				XAdv.dhu[i] += hi * (ga * hi * dmdl + fG * vvi);// This term is == 0 so should be commented here
				XAdv.dhv[i] += hi * (ga * hi * dmdt - fG * uui);// Need double checking before doing that
			}
		}
	}
}
template __host__ void updateEVCPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, FluxP<float> XFlux, AdvanceP<float> XAdv);
template __host__ void updateEVCPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, FluxP<double> XFlux, AdvanceP<double> XAdv);


template <class T> __global__ void AdvkernelGPU(Param XParam, BlockP<T> XBlock, T dt ,T* zb, EvolvingP<T> XEv, AdvanceP<T> XAdv, EvolvingP<T> XEv_o)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	T eps = T(XParam.eps);
	T hold = XEv.h[i];
	T ho, uo, vo;
	ho = hold + dt * XAdv.dh[i];


	if (ho > eps) {
		//
		uo = (hold * XEv.u[i] + dt * XAdv.dhu[i]) / ho;
		vo = (hold * XEv.v[i] + dt * XAdv.dhv[i]) / ho;
		
	}
	else
	{// dry
		
		uo = T(0.0);
		vo = T(0.0);
	}


	XEv_o.zs[i] = zb[i] + ho;
	XEv_o.h[i] = ho;
	XEv_o.u[i] = uo;
	XEv_o.v[i] = vo;
	

}
template __global__ void AdvkernelGPU<float>(Param XParam, BlockP<float> XBlock, float dt, float* zb, EvolvingP<float> XEv, AdvanceP<float> XAdv, EvolvingP<float> XEv_o);
template __global__ void AdvkernelGPU<double>(Param XParam, BlockP<double> XBlock, double dt, double* zb, EvolvingP<double> XEv, AdvanceP<double> XAdv, EvolvingP<double> XEv_o);


template <class T> __host__ void AdvkernelCPU(Param XParam, BlockP<T> XBlock, T dt, T* zb, EvolvingP<T> XEv, AdvanceP<T> XAdv, EvolvingP<T> XEv_o)
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

				
				T hold = XEv.h[i];
				T ho, uo, vo;
				ho = hold + dt * XAdv.dh[i];


				if (ho > eps) {
					//
					uo = (hold * XEv.u[i] + dt * XAdv.dhu[i]) / ho;
					vo = (hold * XEv.v[i] + dt * XAdv.dhv[i]) / ho;

				}
				else
				{// dry

					uo = T(0.0);
					vo = T(0.0);
				}


				XEv_o.zs[i] = zb[i] + ho;
				XEv_o.h[i] = ho;
				XEv_o.u[i] = uo;
				XEv_o.v[i] = vo;
			}
		}
	}

}
template __host__ void AdvkernelCPU<float>(Param XParam, BlockP<float> XBlock, float dt, float* zb, EvolvingP<float> XEv, AdvanceP<float> XAdv, EvolvingP<float> XEv_o);
template __host__ void AdvkernelCPU<double>(Param XParam, BlockP<double> XBlock, double dt, double* zb, EvolvingP<double> XEv, AdvanceP<double> XAdv, EvolvingP<double> XEv_o);



template <class T> __global__ void cleanupGPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, EvolvingP<T> XEv_o)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	
	XEv_o.h[i] = XEv.h[i];
	XEv_o.zs[i] = XEv.zs[i];
	XEv_o.u[i] = XEv.u[i];
	XEv_o.v[i] = XEv.v[i];

}
template __global__ void cleanupGPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, EvolvingP<float> XEv_o);
template __global__ void cleanupGPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, EvolvingP<double> XEv_o);

template <class T> __host__ void cleanupCPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, EvolvingP<T> XEv_o)
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

				XEv_o.h[i] = XEv.h[i];
				XEv_o.zs[i] = XEv.zs[i];
				XEv_o.u[i] = XEv.u[i];
				XEv_o.v[i] = XEv.v[i];
			}
		}
	}

}
template __host__ void cleanupCPU<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> XEv, EvolvingP<float> XEv_o);
template __host__ void cleanupCPU<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> XEv, EvolvingP<double> XEv_o);


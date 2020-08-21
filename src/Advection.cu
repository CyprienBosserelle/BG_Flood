#include "Advection.h"

/*

template <class T>__global__ void updateEV(float delta, float g, float fc, int* rightblk, int* topblk, float* hh, float* uu, float* vv, float* Fhu, float* Fhv, float* Su, float* Sv, float* Fqux, float* Fquy, float* Fqvx, float* Fqvy, float* dh, float* dhu, float* dhv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int blksize = blkmemwidth * blkmemwidth;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];




	int i = ix + iy * blockDim.x + ibl * (blockDim.x * blockDim.y);




	int iright, itop;

	
	iright = findrightG(ix, iy, rightblk[ibl], ibl, blockDim.x);
	itop = findtopG(ix, iy, topblk[ibl], ibl, blockDim.x);
	

	

		float cm = 1.0f;// 0.1;
		float fmu = 1.0f;
		float fmv = 1.0f;

		float hi = hh[i];
		float uui = uu[i];
		float vvi = vv[i];


		float cmdinv, ga;

		cmdinv = 1.0f / (cm * delta);
		ga = 0.5f * g;
		

		dh[i] = -1.0f * (Fhu[iright] - Fhu[i] + Fhv[itop] - Fhv[i]) * cmdinv;
		


		//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
		//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
		float dmdl = (fmu - fmu) / (cm * delta);// absurd if not spherical!
		float dmdt = (fmv - fmv) / (cm * delta);
		float fG = vvi * dmdl - uui * dmdt;
		dhu[i] = (Fqux[i] + Fquy[i] - Su[iright] - Fquy[itop]) * cmdinv + fc * hi * vvi;
		dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[itop] - Fqvx[iright]) * cmdinv - fc * hi * uui;
		
		dhu[i] += hi * (ga * hi * dmdl + fG * vvi);// This term is == 0 so should be commented here
		dhv[i] += hi * (ga * hi * dmdt - fG * uui);// Need double checking before doing that
	
}
*/


#include "Culverts.h"


template <class T> __host__ void AddCulverts(Param XParam, double dt, std::vector<Culvert> XCulverts, Model<T> XModel)
{
	dim3 gridDimCulvert(XModel.bndblk.nblkculvert, 1, 1);
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	T Qmax, Vol1, Q;
	int cc;


	// Get the elevation/water column for each culvert edge and put it in the culvert structure (loop on concerned blocks)
	
	if (XParam.GPUDEVICE >= 0)
	{
		for (cc = 0; cc < XCulverts.size(); cc++)
		{
			GetCulvertElevGPU <<< gridDimCulvert, blockDim, 0 >>> (XParam, cc, XCulverts[cc], XModel.culvertsF, XModel.bndblk.culvert, XModel.blocks, XModel.evolv);
		}
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	else
	{
		GetCulvertElevCPU(XParam, XCulverts, XModel.culvertsF, XModel.bndblk.nblkculvert, XModel.bndblk.culvert, XModel.blocks, XModel.evolv);
	}


	// Calculation of the transfert of water (depending of the type of culvert)(loop on culverts)
	for (cc = 0; cc < XCulverts.size(); cc++)
	{

		//ib1 = XCulverts[cc].block1;

		//Pump system
		if (XCulverts[cc].type == 0)
		{
			Qmax = T(XCulverts[cc].Qmax);
			Vol1 = XModel.culvertsF.h1[cc];// *XCulverts[cc].dx1* XCulverts[cc].dx1;
			Q = T(Vol1 * dt);
			if (Q > Qmax)
			{
				XModel.culvertsF.dq[cc] = Qmax;
			}
			else
			{
				XModel.culvertsF.dq[cc] = Q;
			}
		}
		/*
		//One way (clapped) culvert
		if (XCulverts.type == 1)
		
		//Basic 2way culvert
		if (XCulverts.type == 2)
		*/
	}

	/*

	Application of the result to h:
	(Loop on blocks)

	*/

	if (XParam.GPUDEVICE >= 0)
	{
		for (cc = 0; cc < XCulverts.size(); cc++)
		{
			InjectCulvertGPU <<<gridDimCulvert, blockDim, 0 >>> (XParam, cc, XCulverts[cc], XModel.culvertsF, XModel.bndblk.culvert, XModel.blocks, XModel.adv);
		}
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	else
	{
		InjectCulvertCPU(XParam, XCulverts, XModel.culvertsF, XModel.bndblk.nblkculvert, XModel.bndblk.culvert, XModel.blocks, XModel.adv);
	}
}
template __host__ void AddCulverts<float>(Param XParam, double dt, std::vector<Culvert> XCulverts, Model<float> XModel);
template __host__ void AddCulverts<double>(Param XParam, double dt, std::vector<Culvert> XCulverts, Model<double> XModel);


template <class T> __global__ void InjectCulvertGPU(Param XParam, int cc, Culvert XCulvert, CulvertF<T> XCulvertF, int* Culvertblks, BlockP<T> XBlock, AdvanceP<T>& XAdv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = Culvertblks[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	if (i == XCulvert.i1)
	{
		XAdv.dh[i] -= XCulvertF.dq[cc] / (XCulvert.dx1 * XCulvert.dx1);
	}
	if (i == XCulvert.i2)
	{
		XAdv.dh[i] += XCulvertF.dq[cc] / (XCulvert.dx2 * XCulvert.dx2);
	}

}
template __global__ void InjectCulvertGPU<float>(Param XParam, int cc, Culvert XCulvert, CulvertF<float> XCulvertF, int* Culvertblks, BlockP<float> XBlock, AdvanceP<float>& XAdv);
template __global__ void InjectCulvertGPU<double>(Param XParam, int cc, Culvert XCulvert, CulvertF<double> XCulvertF, int* Culvertlks, BlockP<double> XBlock, AdvanceP<double>& XAdv);


template <class T> __host__ void InjectCulvertCPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T> XCulvertF, int nblkculvert, int* Culvertblks, BlockP<T> XBlock, AdvanceP<T> XAdv)
{

	T delta1, delta2;
	int cc;

	for (cc = 0; cc < XCulverts.size(); cc++)
	{
		delta1 = calcres(T(XParam.dx), XBlock.level[XCulverts[cc].block1]);
		delta2 = calcres(T(XParam.dx), XBlock.level[XCulverts[cc].block2]);
		XAdv.dh[XCulverts[cc].i1] -= XCulvertF.dq[cc] / (delta1 * delta1);
		XAdv.dh[XCulverts[cc].i2] += XCulvertF.dq[cc] / (delta2 * delta2);
	}

}
template __host__ void InjectCulvertCPU<float>(Param XParam, std::vector<Culvert> XCulverts, CulvertF<float> XCulvertF, int nblkculvert, int* Culvertblks, BlockP<float> XBlock, AdvanceP<float> XAdv);
template __host__ void InjectCulvertCPU<double>(Param XParam, std::vector<Culvert> XCulverts, CulvertF<double> XCulvertF, int nblkculvert, int* Culvertblks, BlockP<double> XBlock, AdvanceP<double> XAdv);



template <class T> __global__ void GetCulvertElevGPU(Param XParam, int cc, Culvert XCulvert, CulvertF<T>& XCulvertF, int* Culvertblks, BlockP<T> XBlock, EvolvingP<T> XEv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = Culvertblks[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);


	if (i == XCulvert.i1)
	{
		XCulvertF.h1[cc] = XEv.h[i];
		XCulvertF.zs1[cc] = XEv.zs[i];
	}
	if (i == XCulvert.i2)
	{
		XCulvertF.h2[cc] = XEv.h[i];
		XCulvertF.zs2[cc] = XEv.zs[i];
	}
}
template __global__ void GetCulvertElevGPU<float>(Param XParam, int cc, Culvert XCulvert, CulvertF<float>& XCulvertF, int* Culvertblks, BlockP<float> XBlock, EvolvingP<float> XEv);
template __global__ void GetCulvertElevGPU<double>(Param XParam, int cc, Culvert XCulvert, CulvertF<double>& XCulvertF, int* Culvertlks, BlockP<double> XBlock, EvolvingP<double> XEv);


template <class T> __host__ void GetCulvertElevCPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T>& XCulvertF, int nblkculvert, int* Culvertblks, BlockP<T> XBlock, EvolvingP<T> XEv)
{
	int cc;

	for (cc = 0; cc < XCulverts.size(); cc++)
	{
		XCulvertF.h1[cc] = XEv.h[XCulverts[cc].i1];
		XCulvertF.zs1[cc] = XEv.zs[XCulverts[cc].i1];

		XCulvertF.h2[cc] = XEv.h[XCulverts[cc].i2];
		XCulvertF.zs2[cc] = XEv.zs[XCulverts[cc].i2];
	}

}
template __host__ void GetCulvertElevCPU<float>(Param XParam, std::vector<Culvert> XCulverts, CulvertF<float>& XCulvertF, int nblkculvert, int* Culvertblks, BlockP<float> XBlock, EvolvingP<float> XEv);
template __host__ void GetCulvertElevCPU<double>(Param XParam, std::vector<Culvert> XCulverts, CulvertF<double>& XCulvertF,  int nblkculvert, int* Culvertblks, BlockP<double> XBlock, EvolvingP<double> XEv);



#include "Culverts.h"


template <class T> __host__ void AddCulverts(Param XParam, Loop<T> XLoop, std::vector<Culvert> XCulverts, Model<T> XModel)
{
	dim3 gridDimCulvert(XModel.bndblk.nblkcluvert, 1, 1);
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	T Qmax, Vol1, delta1, Q;

	unsigned int ib = Culvertblks[ibl];

	// Get the elevation for each culvert edge and put it in the culvert structure



	// Calculation of the transfert of water (depending of the type of culvert)
	for (int cc = 0; cc < XCulverts.size(); cc++)
	{
		XCulvertF = XModel.culvertsF[cc];

		//Pump system
		if (XCulverts.type == 0)
		{
			Qmax = XCulverts.Qmax;
			delta1 = calcres(T(XParam.dx), XBlock.level[ib]);
			Vol1 = XCulvertF.h1 * delta1 * delta1;
			Q = Vol1 * XLoop.dt;
			if (Q > Qmax)
			{
				XCulvertF.dq = Qmax;
			}
			else
			{
				XCulvertF.dq = Q;
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
	Loop on blocks first or on culverts first?

	*/

	for (int cc = 0; cc < XCulverts.size(); cc++)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			InjectCulvertGPU << <gridDimCulvert, blockDim, 0 >> > (XParam, XCulverts[cc], XModel.bndblk.culvert, XModel.blocks, XModel.adv);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			InjectCulvertCPU(XParam, XCulvert[cc], XModel.bndblk.nblkculvert, XModel.bndblk.culvert, XModel.blocks, XModel.adv);
		}
	}
}
template __host__ void AddCulverts<float>(Param XParam, Loop<float> XLoop, std::vector<Culvert> XRivers, Model<float> XModel);
template __host__ void AddCulverts<double>(Param XParam, Loop<double> XLoop, std::vector<Culvert> XRivers, Model<double> XModel);


template <class T> __global__ void InjectCulvertGPU(Param XParam, Culvert XCulvert, int* Culvertblks, BlockP<T> XBlock, AdvanceP<T> XAdv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = Culvertblks[ibl];



	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	T delta = calcres(T(XParam.dx), XBlock.level[ib]);



	if (i == XCulvert.i1)
	{
		XAdv.dh[i] -= CulvertF.dq/(delta*delta);
	}
	if (i == XCulvert.i2)
	{
		XAdv.dh[i] += CulvertF.dq/(delta*delta);
	}



}
template __global__ void InjectCulvertGPU<float>(Param XParam, Culvert XCulvert,  int* Culvertblks, BlockP<float> XBlock, AdvanceP<float> XAdv);
template __global__ void InjectCulvertGPU<double>(Param XParam, Culvert XCulvert,  int* Culvertlks, BlockP<double> XBlock, AdvanceP<double> XAdv);

template <class T> __host__ void InjectCulvertCPU(Param XParam, Culvert XCulvert, int nblkculvert, int* Culvertblks, BlockP<T> XBlock, AdvanceP<T> XAdv)
{
	unsigned int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	T delta, levdx;
	int i, ix, iy, ibl;

	for (ibl = 0; ibl < nblkculvert; ibl++)
	{
		ib = Culvertblks[ibl];

		levdx = calcres(T(XParam.dx), XBlock.level[ib]);

		for (iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (ix = 0; ix < XParam.blkwidth; ix++)
			{

				i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				delta = calcres(T(XParam.dx), XBlock.level[ib]);

				if (i == XCulvert.i1)
				{
					XAdv.dh[i] -= CulvertF.dq / (delta * delta);
				}
				if (i == XCulvert.i2)
				{
					XAdv.dh[i] += CulvertF.dq / (delta * delta);
				}

			}
		}
	}


}
template __host__ void InjectCulvertCPU<float>(Param XParam, Culvert XCulvert, int nblkculvert, int* Culvertblks, BlockP<float> XBlock, AdvanceP<float> XAdv);
template __host__ void InjectCulvertCPU<double>(Param XParam, Culvert XCulvert, int nblkculvert, int* Culvertblks, BlockP<double> XBlock, AdvanceP<double> XAdv);

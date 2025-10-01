
#include "Culverts.h"


template <class T> void AddCulverts(Param XParam, double dt, std::vector<Culvert> XCulverts, Model<T> XModel)
{
	dim3 gridDimCulvert(XModel.bndblk.nblkculvert, 1, 1);
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	T Qmax, Vol1, Q;
	int cc;

	printf("nblkculvert = %d, XCulverts.size() = %zu\n", XModel.bndblk.nblkculvert, XCulverts.size());
	// Get the elevation/water column for each culvert edge and put it in the culvert structure (loop on concerned blocks)
	
	if (XParam.GPUDEVICE >= 0)
	{
		for (cc = 0; cc < XCulverts.size(); cc++)
		{
		    GetCulvertElevGPU <<< gridDimCulvert, blockDim, 0 >>> (XParam, cc, XCulverts[cc], XModel.culvertsF, XModel.bndblk.culvert, XModel.evolv);
			//CUDA_CHECK(cudaDeviceSynchronize());
			//CopyGPUtoCPU   (XParam,BlockP, )
			//CopyGPUtoCPU(int , int blksize, T * z_cpu, T * z_gpu)
		}
		//CUDA_CHECK(cudaDeviceSynchronize());
	}
	else
	{
		GetCulvertElevCPU(XParam, XCulverts, XModel.culvertsF, XModel.bndblk.nblkculvert, XModel.bndblk.culvert, XModel.blocks, XModel.evolv);
	}


	// Calculation of the transfert of water (depending of the type of culvert)(loop on culverts)

	if (XParam.GPUDEVICE >= 0)
	{
		for (cc = 0; cc < XCulverts.size(); cc++)
		{
			DischargeCulvertGPU <<< gridDimCulvert, blockDim, 0 >>> (XParam, cc, XCulverts[cc], XModel.culvertsF, XModel.bndblk.culvert);
		}
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	else
	{
		DischargeCulvertCPU(XParam, XCulverts, XModel.culvertsF, XModel.bndblk.nblkculvert, XModel.bndblk.culvert);
	}


	/*
	for (cc = 0; cc < XCulverts.size(); cc++)
	{




		//ib1 = XCulverts[cc].block1;
		printf("XCulverts size=%f\n", XCulverts.size());
		printf("XCulverts size dx=%f\n", XCulverts[cc].dx1);

		//printf("XCulvertsF size h1=%i\n", XModel.culvertsF.h1.size());


		//Pump system
		if (XCulverts[cc].type == 0)
		{
			DischargeCulvert(XCulverts[cc], XModel.culvertsF.h1[cc], XModel.culvertsF.h2[cc], XModel.culvertsF.zs1[cc], XModel.culvertsF.zs2[cc])
				 
			printf("Qmax=%f\n", XCulverts[cc].Qmax);
			Qmax = XCulverts[cc].Qmax;
			printf("dx1=%f\n", XCulverts[cc].dx1);
			printf("h1=%f\n", XModel.culvertsF.h1[cc]);
			Vol1 = XModel.culvertsF.h1[cc] * XCulverts[cc].dx1 * XCulverts[cc].dx1;
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
		printf("H1=%f , H2=%f, DQ=%f \n", XModel.culvertsF.h1[cc], XModel.culvertsF.h2[cc], XModel.culvertsF.dq[cc]);
		
		//One way (clapped) culvert
		if (XCulverts[cc].type == 1)
		{
			if (XModel.culvertsF.zs1[cc] >= XModel.culvertsF.zs2[cc] && XModel.culvertsF.h1[cc] > 0.0)
			{
				XModel.culvertsF.dq[cc] = 0;
			}
			else
			{
				XModel.culvertsF.dq[cc] = 0;
			}
			
			
			
			Qmax = T(XCulverts[cc].Qmax);
			Vol1 = XModel.culvertsF.h1[cc] * XCulverts[cc].dx1 * XCulverts[cc].dx1;
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
		printf("H1=%f , H2=%f, DQ=%f \n", XModel.culvertsF.h1[cc], XModel.culvertsF.h2[cc], XModel.culvertsF.dq[cc]);
		//Basic 2way culvert
		if (XCulverts[cc].type == 2)
		{
			//TODO: to be completed
		}

	}
	*/
	/*

	Application of the result to h:
	(Loop on blocks)

	*/

	if (XParam.GPUDEVICE >= 0)
	{
		for (cc = 0; cc < XCulverts.size(); cc++)
		{
			InjectCulvertGPU <<<gridDimCulvert, blockDim, 0 >>> (XParam, cc, XCulverts[cc], XModel.culvertsF, XModel.bndblk.culvert, XModel.blocks, XModel.adv);
			CUDA_CHECK(cudaDeviceSynchronize());
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


template <class T> __global__ void InjectCulvertGPU(Param XParam, int cc, Culvert XCulvert, CulvertF<T>& XCulvertF, int* Culvertblks, BlockP<T> XBlock, AdvanceP<T>& XAdv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = Culvertblks[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	//for (int cc = 0; cc < XCulvert.size(); cc++)
	//{
		if (ix == XCulvert.ix1 && iy == XCulvert.iy1)
		{
			//printf("Before adding: h=%f\n", XAdv.dh[i]);
			XAdv.dh[i] -= XCulvertF.dq[cc] / (XCulvert.dx1 * XCulvert.dx1);
			//printf("After adding: h = % f\n", XAdv.dh[i]);

		}
		if (ix == XCulvert.ix2 && iy == XCulvert.iy2)
		{
			//printf("Before adding: h=%f\n", XAdv.dh[i]);
			XAdv.dh[i] += XCulvertF.dq[cc] / (XCulvert.dx2 * XCulvert.dx2);
			//printf("After adding: h=%f\n", XAdv.dh[i]);

		}
	//}

}
template __global__ void InjectCulvertGPU<float>(Param XParam, int cc, Culvert XCulvert, CulvertF<float>& XCulvertF, int* Culvertblks, BlockP<float> XBlock, AdvanceP<float>& XAdv);
template __global__ void InjectCulvertGPU<double>(Param XParam, int cc, Culvert XCulvert, CulvertF<double>& XCulvertF, int* Culvertlks, BlockP<double> XBlock, AdvanceP<double>& XAdv);


template <class T> __host__ void InjectCulvertCPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T>& XCulvertF, int nblkculvert, int* Culvertblks, BlockP<T> XBlock, AdvanceP<T> XAdv)
{

	T delta1, delta2;
	int cc, i1, i2;

	for (cc = 0; cc < XCulverts.size(); cc++)
	{
		i1 = memloc(XParam, XCulverts[cc].ix1, XCulverts[cc].iy1, XBlock.active[XCulverts[cc].block1]);
		i2 = memloc(XParam, XCulverts[cc].ix2, XCulverts[cc].iy2, XBlock.active[XCulverts[cc].block2]);
		//printf("before adding: i1=%i, b1=%i, x1=%i, y1=%i\n", i1, XCulverts[cc].block1, XCulverts[cc].ix1, XCulverts[cc].iy1);
		//printf("before adding: i2=%i, b2=%i, x2=%i, y2=%i\n", i2, XCulverts[cc].block2, XCulverts[cc].ix2, XCulverts[cc].iy2);

		delta1 = XCulverts[cc].dx1; // calcres(T(XParam.dx), XBlock.level[XCulverts[cc].block1]);
		delta2 = XCulverts[cc].dx2; // calcres(T(XParam.dx), XBlock.level[XCulverts[cc].block2]);
		//printf("before adding: dh1=%f, dh2=%f\n", XAdv.dh[i1], XAdv.dh[i2]);

		XAdv.dh[i1] -= XCulvertF.dq[cc] / (delta1 * delta1);
		XAdv.dh[i2] += XCulvertF.dq[cc] / (delta2 * delta2);
		//printf("Vars: dq=%f, delta2=%f, delta2=%f\n", XCulvertF.dq[cc], delta2, delta1);
		//printf("After adding: dh1=%f, dh2=%f\n", XAdv.dh[i1], XAdv.dh[i2]);
	}

}
template __host__ void InjectCulvertCPU<float>(Param XParam, std::vector<Culvert> XCulverts, CulvertF<float>& XCulvertF, int nblkculvert, int* Culvertblks, BlockP<float> XBlock, AdvanceP<float> XAdv);
template __host__ void InjectCulvertCPU<double>(Param XParam, std::vector<Culvert> XCulverts, CulvertF<double>& XCulvertF, int nblkculvert, int* Culvertblks, BlockP<double> XBlock, AdvanceP<double> XAdv);


template <class T> __global__ void GetCulvertElevGPU(Param XParam, int cc, Culvert XCulvert, CulvertF<T>& XCulvertF, int* Culvertblks, EvolvingP<T> XEv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = Culvertblks[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	//printf("culvert read h= %f, block=%i, i=%i, ix=%i, iy=%i, ix1=%i, iy1=%i\n", XEv.h[i], ib, i, ix, iy, XCulvert.ix1, XCulvert.iy1);
	//printf("culvert read ix1=%i, iy1=%i\n", XCulvert.ix1, XCulvert.iy1);
	//for (int cc = 0; cc < XCulverts.size(); cc++)
	//{
		if (ibl == XCulvert.block1 || ibl == XCulvert.block2)
		{
			if (ix == XCulvert.ix1 && iy == XCulvert.iy1)
			{
				XCulvertF.h1[cc] = XEv.h[i];
				XCulvertF.zs1[cc] = XEv.zs[i];
			}
			if (ix == XCulvert.ix2 && iy == XCulvert.iy2)
			{
				XCulvertF.h2[cc] = XEv.h[i];
				XCulvertF.zs2[cc] = XEv.zs[i];
			}
		}
	//}

}
template __global__ void GetCulvertElevGPU<float>(Param XParam, int cc, Culvert XCulvert, CulvertF<float>& XCulvertF, int* Culvertblks, EvolvingP<float> XEv);
template __global__ void GetCulvertElevGPU<double>(Param XParam, int cc, Culvert XCulvert, CulvertF<double>& XCulvertF, int* Culvertlks, EvolvingP<double> XEv);


template <class T> __host__ void GetCulvertElevCPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T>& XCulvertF, int nblkculvert, int* Culvertblks, BlockP<T> XBlock, EvolvingP<T> XEv)
{
	int cc, i1, i2;

	for (cc = 0; cc < XCulverts.size(); cc++)
	{
		//printf("GetCulvertElevCPU: block=%i, ix1=%i, iy1=%i, ix2=%i, iy2=%i\n", XCulverts[cc].block1, XCulverts[cc].ix1, XCulverts[cc].iy1, XCulverts[cc].ix2, XCulverts[cc].iy2);
		i1 = memloc(XParam, XCulverts[cc].ix1, XCulverts[cc].iy1, XCulverts[cc].block1);
		i2 = memloc(XParam, XCulverts[cc].ix2, XCulverts[cc].iy2, XCulverts[cc].block2);

		XCulvertF.h1[cc] = XEv.h[i1];
		XCulvertF.zs1[cc] = XEv.zs[i1];


		XCulvertF.h2[cc] = XEv.h[i2];
		XCulvertF.zs2[cc] = XEv.zs[i2];
	}

}
template __host__ void GetCulvertElevCPU<float>(Param XParam, std::vector<Culvert> XCulverts, CulvertF<float>& XCulvertF, int nblkculvert, int* Culvertblks, BlockP<float> XBlock, EvolvingP<float> XEv);
template __host__ void GetCulvertElevCPU<double>(Param XParam, std::vector<Culvert> XCulverts, CulvertF<double>& XCulvertF,  int nblkculvert, int* Culvertblks, BlockP<double> XBlock, EvolvingP<double> XEv);


template <class T> __global__ void DischargeCulvertGPU(Param XParam, int cc, Culvert XCulvert, CulvertF<T>& XCulvertF, int* Culvertblks)
{
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = Culvertblks[ibl];

	//for (int cc = 0; cc < XCulverts.size(); cc++)
	//{
		if (XCulvert.type == 0)
		{
			CulvertPump(XCulvert, XCulvertF.h1[cc], XCulvertF.h2[cc], XCulvertF.zs1[cc], XCulvertF.zs2[cc], XCulvertF.dq[cc], XParam.dt);
		}
	//}
}

template __global__ void DischargeCulvertGPU<float>(Param XParam, int cc, Culvert XCulvert, CulvertF<float>& XCulvertF, int* Culvertblks);
template __global__ void DischargeCulvertGPU<double>(Param XParam, int cc, Culvert XCulvert, CulvertF<double>& XCulvertF, int* Culvertlks);

template <class T> __host__ void DischargeCulvertCPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T>& XCulvertF, int nblkculvert, int* Culvertblks)
{
	T Qmax, Vol1, Q;
	int cc;
	for (cc = 0; cc < XCulverts.size(); cc++)
	{

	//Pump system	
		if (XCulverts[cc].type == 0)
		{
			CulvertPump(XCulverts[cc], XCulvertF.h1[cc], XCulvertF.h2[cc], XCulvertF.zs1[cc], XCulvertF.zs2[cc], XCulvertF.dq[cc], XParam.dt);
		}
		/*
		//printf("H1=%f , H2=%f, DQ=%f \n", XModel.culvertsF.h1[cc], XModel.culvertsF.h2[cc], XModel.culvertsF.dq[cc]);
		//One way (clapped) culvert
		if (XCulverts[cc].type == 1)
		{
			if (XCulvertF.zs1[cc] >= XCulvertF.zs2[cc] && XCulvertF.h1[cc] > 0.0)
			{
				XCulvertF.dq[cc] = 0;
			}
			else
			{
				XCulvertF.dq[cc] = 0;
			}
			Qmax = T(XCulverts[cc].Qmax);
			Vol1 = XCulvertF.h1[cc] * XCulverts[cc].dx1 * XCulverts[cc].dx1;
			Q = T(Vol1 * XParam.dt);
			if (Q > Qmax)
			{
				XCulvertF.dq[cc] = Qmax;
			}
			else
			{
				XCulvertF.dq[cc] = Q;
			}
		}*/
	}
}

template __host__ void DischargeCulvertCPU<float>(Param XParam, std::vector<Culvert> XCulverts, CulvertF<float>& XCulvertF, int nblkculvert, int* Culvertblks);
template __host__ void DischargeCulvertCPU<double>(Param XParam, std::vector<Culvert> XCulverts, CulvertF<double>& XCulvertF, int nblkculvert, int* Culvertblks);


template <class T> __host__ __device__ void CulvertPump(Culvert XCulvert, T h1, T h2, T zs1, T zs2, T & dq, double dt)
{
	T Qmax, Vol1, Q;

	Qmax = XCulvert.Qmax;
	Vol1 = h1 * XCulvert.dx1 * XCulvert.dx1;
	Q = T(Vol1 * dt);
	if (Q > Qmax)
	{
		dq = Qmax;
	}
	else
	{
		dq = Q;
	}
}
template __host__ __device__ void CulvertPump<float>(Culvert XCulvert, float h1, float h2, float zs1, float zs2, float& q, double dt);
template __host__ __device__ void CulvertPump<double>(Culvert XCulvert, double h1, double h2, double zs1, double zs2, double& q, double dt);

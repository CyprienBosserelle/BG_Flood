#include "Culverts.h"


template <class T> void AddCulverts(Param XParam, double dt, std::vector<Culvert> XCulverts, Model<T> XModel)
{
	dim3 gridDimCulvert(XCulverts.size(), 1, 1);
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	T Qmax, Vol1, Q;
	int cc;
	int i;

	//printf("nblkculvert = %d, XCulverts.size() = %zu\n", XModel.bndblk.nblkculvert, XCulverts.size());
	// Get the elevation/water column for each culvert edge and put it in the culvert structure (loop on concerned blocks)
	
	if (XParam.GPUDEVICE >= 0)
	{
		for (cc = 0; cc < XCulverts.size(); cc++)
		{
		    //GetCulvertElevGPU <<< gridDimCulvert, blockDim, 0 >>> (XParam, cc, XCulverts[cc].block1, XCulverts[cc].ix1, XCulverts[cc].iy1, XCulverts[cc].block2, XCulverts[cc].ix2, XCulverts[cc].iy2, XModel.culvertsF, XModel.bndblk.culvert, XModel.evolv);
			//CUDA_CHECK(cudaDeviceSynchronize());
			//XCulvertF.h1[cc] = XEv.h[i];
			i = memloc(XParam.halowidth, XParam.blkmemwidth, XCulverts[cc].ix1, XCulverts[cc].iy1, XCulverts[cc].block1);
			CUDA_CHECK(cudaMemcpy(XModel.culvertsF.h1 + cc, XModel.evolv.h+i, 1 * sizeof(T), cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(XModel.culvertsF.zs1 + cc, XModel.evolv.zs+i, 1 * sizeof(T), cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(XModel.culvertsF.u1 + cc, XModel.evolv.u + i, 1 * sizeof(T), cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(XModel.culvertsF.v1 + cc, XModel.evolv.v + i, 1 * sizeof(T), cudaMemcpyDeviceToDevice));

			i = memloc(XParam.halowidth, XParam.blkmemwidth, XCulverts[cc].ix2, XCulverts[cc].iy2, XCulverts[cc].block2);
			CUDA_CHECK(cudaMemcpy(XModel.culvertsF.h2 + cc, XModel.evolv.h+i, 1 * sizeof(T), cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(XModel.culvertsF.zs2 + cc, XModel.evolv.h+i, 1 * sizeof(T), cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(XModel.culvertsF.u2 + cc, XModel.evolv.u + i, 1 * sizeof(T), cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(XModel.culvertsF.v2 + cc, XModel.evolv.v + i, 1 * sizeof(T), cudaMemcpyDeviceToDevice));
		}
	}
	else
	{
		GetCulvertElevCPU(XParam, XCulverts, XModel.culvertsF, XModel.bndblk.nblkculvert, XModel.bndblk.culvert, XModel.blocks, XModel.evolv);
	}

	// Calculation of the transfert of water (depending of the type of culvert)(loop on culverts)

	if (XParam.GPUDEVICE >= 0)
	{
		//for (cc = 0; cc < XCulverts.size(); cc++)
		{
			int cc = 0;
			DischargeCulvertGPU << < gridDimCulvert, blockDim, 0 >> > (XParam, dt, XModel.culvertsF, XCulverts[cc]);
			//DischargeCulvertGPU <<< gridDimCulvert, blockDim, 0 >>> (XParam, XCulverts, XModel.culvertsF);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	}
	else
	{
		DischargeCulvertCPU(XParam, dt, XCulverts, XModel.culvertsF, XModel.bndblk.nblkculvert, XModel.bndblk.culvert);
	}

	
//	reducemin3 << <gridDimLine, blockDimLine, smemSize >> > (XTime.dtmax, XTime.arrmin, s);


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


template <class T> __global__ void InjectCulvertGPU(Param XParam, int cc, Culvert XCulvert, CulvertF<T> XCulvertF, int* Culvertblks, BlockP<T> XBlock, AdvanceP<T> XAdv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = Culvertblks[ibl];

	int iinlet = memloc(halowidth, blkmemwidth, XCulvert.ix1, XCulvert.iy1, XCulvert.block1);
	int ioutlet = memloc(halowidth, blkmemwidth, XCulvert.ix2, XCulvert.iy2, XCulvert.block2);

	//for (int cc = 0; cc < XCulvert.size(); cc++)
	//{
		if (ix == XCulvert.ix1 && iy == XCulvert.iy1)
		{
			//printf("Before adding: h=%f\n", XAdv.dh[i]);
			XAdv.dh[iinlet] -= XCulvertF.dq[cc] / (XCulvert.dx1 * XCulvert.dx1);
			//printf("After adding: h = % f\n", XAdv.dh[i]);

		}
		if (ix == XCulvert.ix2 && iy == XCulvert.iy2)
		{
			//printf("Before adding: h=%f\n", XAdv.dh[i]);
			XAdv.dh[ioutlet] += XCulvertF.dq[cc] / (XCulvert.dx2 * XCulvert.dx2);
			//printf("After adding: h=%f\n", XAdv.dh[i]);

		}
	//}

}
template __global__ void InjectCulvertGPU<float>(Param XParam, int cc, Culvert XCulvert, CulvertF<float> XCulvertF, int* Culvertblks, BlockP<float> XBlock, AdvanceP<float> XAdv);
template __global__ void InjectCulvertGPU<double>(Param XParam, int cc, Culvert XCulvert, CulvertF<double> XCulvertF, int* Culvertlks, BlockP<double> XBlock, AdvanceP<double> XAdv);


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


template <class T> __global__ void GetCulvertElevGPU(Param XParam, int cc, int b1, int ix1, int iy1, int b2, int ix2,int iy2, CulvertF<T>& XCulvertF, int* Culvertblks, EvolvingP<T> XEv)
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
		if (ibl == b1 || ibl == b2)
		{
			if (ix == ix1 && iy == iy1)
			{
				XCulvertF.h1[cc] = XEv.h[i];
				XCulvertF.zs1[cc] = XEv.zs[i];
			}
			if (ix == ix2 && iy == iy2)
			{
				XCulvertF.h2[cc] = XEv.h[i];
				XCulvertF.zs2[cc] = XEv.zs[i];
			}
		}
	//}

}
template __global__ void GetCulvertElevGPU<float>(Param XParam, int cc, int b1, int ix1, int iy1, int b2, int ix2, int iy2, CulvertF<float>& XCulvertF, int* Culvertblks, EvolvingP<float> XEv);
template __global__ void GetCulvertElevGPU<double>(Param XParam, int cc, int b1, int ix1, int iy1, int b2, int ix2, int iy2, CulvertF<double>& XCulvertF, int* Culvertlks, EvolvingP<double> XEv);


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


template <class T> __global__ void DischargeCulvertGPU(Param XParam, double dt, CulvertF<T> XCulvertF, Culvert XCulvert)
{
	//unsigned int ix = threadIdx.x;
	//unsigned int iy = threadIdx.y;
	int cci = blockIdx.x;
	int cc = 0;
	//unsigned int ib = Culvertblks[ibl];

	//for (int cc = 0; cc < XCulverts.size(); cc++)
	//{
	//printf("DischargeCulvertGPU:");
	//printf("DischargeCulvertGPU: cc=%i, type=%i, Qmax=%f, dx1=%f, h1=%f\n", cci, XCulvertF.type[cci], XCulvertF.Qmax[cci], XCulvertF.dx1[cci], XCulvertF.h1[cci]);
	//, h2 = % f, zs1 = % f, zs2 = % f
	//XCulvertF.h1[cci], XCulvertF.h2[cci], XCulvertF.zs1[cci], XCulvertF.zs2[cci]

		if (XCulvertF.type[cci] == 0)
		{
			CulvertPump(XCulvertF.Qmax[cci], XCulvertF.dx1[cci], XCulvertF.h1[cci], XCulvertF.h2[cci], XCulvertF.zs1[cci], XCulvertF.zs2[cci], XCulvertF.dq[cci], dt);
		}
		if (XCulvertF.type[cci] == 1)
		{
			T Q = T(0.0);
			if (XCulvertF.zs1[cc] >= XCulvertF.zs2[cc] && XCulvertF.h1[cc] > 0.0)
			{
				CulvertDischarge(XCulvert.shape, T(XCulvert.width), T(XCulvert.height), T(XCulvert.length), T(XCulvert.zb1), T(XCulvert.zb2), T(XCulvert.k_ex), T(XCulvert.k_en), T(XCulvert.C_d), T(XCulvert.n), XCulvertF.zs1[cc], XCulvertF.zs2[cc], XCulvertF.u1[cc], XCulvertF.u2[cc], XCulvertF.v1[cc], XCulvertF.v2[cc], Q);
			}
			

			T Vol1 = XCulvertF.h1[cc] * XCulvert.dx1 * XCulvert.dx1;
			T Qmax = T(Vol1 / XParam.dt);
			XCulvertF.dq[cc] = min(Qmax, Q);
		}
		//printf("DischargeCulvertGPU after: q=%f\n", XCulvertF.dq[cci]);

	//}
}

template __global__ void DischargeCulvertGPU<float>(Param XParam, double dt, CulvertF<float>  XCulvertF, Culvert XCulvert);
template __global__ void DischargeCulvertGPU<double>(Param XParam, double dt, CulvertF<double>  XCulvertF, Culvert XCulvert);

template <class T> __host__ void DischargeCulvertCPU(Param XParam, double dt, std::vector<Culvert> XCulverts, CulvertF<T>& XCulvertF, int nblkculvert, int* Culvertblks)
{
	T Qmax, Vol1, Q;
	int cc;

	for (cc = 0; cc < XCulverts.size(); cc++)
	{

		//Pump system	
		if (XCulverts[cc].type == 0)
		{
			CulvertPump(XCulverts[cc].Qmax, XCulverts[cc].dx1, XCulvertF.h1[cc], XCulvertF.h2[cc], XCulvertF.zs1[cc], XCulvertF.zs2[cc], XCulvertF.dq[cc], dt);
		}

		//One way (clapped) culvert
		if (XCulverts[cc].type == 1)
		{
			if (XCulvertF.zs1[cc] >= XCulvertF.zs2[cc] && XCulvertF.h1[cc] > 0.0)
			{
				CulvertDischarge(XCulverts[cc].shape, T(XCulverts[cc].width), T(XCulverts[cc].height), T(XCulverts[cc].length), T(XCulverts[cc].zb1), T(XCulverts[cc].zb2), T(XCulverts[cc].k_ex), T(XCulverts[cc].k_en), T(XCulverts[cc].C_d), T(XCulverts[cc].n), XCulvertF.zs1[cc], XCulvertF.zs2[cc], XCulvertF.u1[cc], XCulvertF.u2[cc], XCulvertF.v1[cc], XCulvertF.v2[cc], Q);
			}
			else
			{
				Q = 0;
			}

			Vol1 = XCulvertF.h1[cc] * XCulverts[cc].dx1 * XCulverts[cc].dx1;
			Qmax = T(Vol1 * XParam.dt);
			XCulvertF.dq[cc] = min(Qmax,Q);
		}
		// Basic 2way culvert
		if (XCulverts[cc].type == 2)
		{
			//NEED TO BE COMPLETED BY ADDED A change in 1/2 location if the flow is reversed.
			CulvertDischarge(XCulverts[cc].shape, T(XCulverts[cc].width), T(XCulverts[cc].height), T(XCulverts[cc].length), T(XCulverts[cc].zb1), T(XCulverts[cc].zb2), T(XCulverts[cc].k_ex), T(XCulverts[cc].k_en), T(XCulverts[cc].C_d), T(XCulverts[cc].n), XCulvertF.zs1[cc], XCulvertF.zs2[cc], XCulvertF.u1[cc], XCulvertF.u2[cc], XCulvertF.v1[cc], XCulvertF.v2[cc], Q);
			Vol1 = XCulvertF.h1[cc] * XCulverts[cc].dx1 * XCulverts[cc].dx1;
			Qmax = T(Vol1 * XParam.dt);
			XCulvertF.dq[cc] = min(Qmax, Q);
		}
	}
}


template <class T> __host__ __device__ void CulvertPump(double Qmax, double dx1, T h1, T h2, T zs1, T zs2, T & dq, double dt)
{
	T Vol1, Q;
	Vol1 = h1 * T(dx1 * dx1);
	Q =  T(Vol1 / T(dt));
	//printf("DischargeCulvertGPU before: q=%f, Q=%f, Vol1=%f, h1=%f, dx1=%f, dt=%f\n", dq, Q, Vol1, h1, dx1, dt);

	if (Q > Qmax)
	{
		dq = Qmax;
	}
	else
	{
		dq = Q;
	}
	//printf("DischargeCulvertGPU after: q=%f, Q=%f, Vol1=%f, h1=%f, dx1=%f, dt=%f, Qmax=%f\n", dq, Q, Vol1, h1, dx1, dt,Qmax);

}
template __host__ __device__ void CulvertPump<float>(double Qmax, double dx1, float h1, float h2, float zs1, float zs2, float& dq, double dt);
template __host__ __device__ void CulvertPump<double>(double Qmax, double dx1, double h1, double h2, double zs1, double zs2, double& dq, double dt);

//template <class T> __host__ __device__ void CulvertOneWay(double Qmax, double dx1, T h1, T h2, T zs1, T zs2, T& dq, double dt)
//{
//	//One way (clapped) culvert
//	Vol1 = XCulvertF.h1[cc] * XCulverts[cc].dx1 * XCulverts[cc].dx1;
//	Q = T(Vol1 * XParam.dt);
//	if (zs1 >= zs2 && h1 > 0.0)
//	{
//		if (Q > Qmax)
//		{
//			dq = Qmax;
//		}
//		else
//		{
//			dq = Q;
//		}
//	}
//	else
//	{
//		dq = 0;
//	}
//}
//template __host__ __device__ void CulvertOneWay<float>(double Qmax, double dx1, float h1, float h2, float zs1, float zs2, float& dq, double dt);
//template __host__ __device__ void CulvertOneWay<double>(double Qmax, double dx1, double h1, double h2, double zs1, double zs2, double& dq, double dt);
//
//template <class T> __host__ __device__ void CulvertTwoWay(double Qmax, double dx1, T h1, T h2, T zs1, T zs2, T& dq, double dt)
//{
//	//Basic 2way culvert flow computation
//
//	if zs1 >0.0 || zs2 > 0.0
//	{
//		if (zs2 > zs1)
//		{
//			Vol1 = h1 * T(dx1 * dx1);
//			Qmax = T(Vol1 * T(dt));
//			
//			if (Q > Qmax)
//			{
//				dq = Qmax;
//			}
//			else
//			{
//				dq = Q;
//			}
//		}
//		else
//		{
//			Vol2 = h2 * T(dx2 * dx2);
//			Q = T(Vol2 * T(dt));
//			if (Q > Qmax)
//			{
//				dq = -Qmax;
//			}
//			else
//			{
//				dq = -Q;
//			}
//		}
//	}
//	else
//	{
//		dq = 0;
//	}
//}

template <class T> __host__ __device__ void CulvertDischarge(int shape, T width,T height, T length, T zb1, T zb2, T k_ex, T k_en, T C_d, T n, T zs1, T zs2, T u1, T u2, T v1, T v2, T & Q)
{
	// A simplified method is used to calculate the discharge through the culvert.
	// 1- The wetted area and hydraulic radius are estimated based on the inlet depth.
	// 2- The discharge is calculated using the inlet and outlet controled methods.
	// 3- The minimum of the two discharges is taken to compute the normal depth as well as the critical depth (as a simplified form) at the outlet.
	// 4- A decision logic based on the normal and critical depth is used to determine the flow regime (submerged or unsubmerged).
	// 5- The final discharge is calculated based on the flow regime.
	// 

	T h_wet;
	T A_wet=T(0.0), P_wet= T(0.0), R_wet= T(0.0), A= T(0.0), P= T(0.0), R=T(height/2);
	T Q_inlet= T(0.0), Q_outlet= T(0.0), Q_estimated= T(0.0);
	T S; //slope
	T H_L; //head loss
	T g = T(9.81); //gravity
	T H_inlet, H_outlet;
	T h_n; //normal depth
	T h_c; //critical depth


	//Base geometry
	h_wet = min(T(zs1 - zb1), height); //Assuming inlet controled, the wetted height is equal to the inlet water depth
	if (shape == 0) //rectangular
	{
		rect_geom(width, height, A, P, R, A_wet, P_wet, R_wet, h_wet);
	}
	else if (shape == 1) //circular
	{
		height = width;
		circular_geom(width/T(2.0), A, P, A_wet, P_wet, R_wet, h_wet);
	}

	//Inlet controled discharge
	//Test if inlet is submerged or not
	H_inlet = T(zs1 - zb1 + (u1 * u1 + v1 * v1) / (2 * g));
	H_outlet = T(zs2 - zb1 + (u2 * u2 + v2 * v2) / (2 * g));
	H_L = H_inlet - H_outlet; //head loss

	if (zs1 < (zb1 + height))
	{
		//Unsubmerged
		Q_inlet = k_en * A * sqrt(2 * g * H_inlet);
	}
	else
	{
		//Submerged
		Q_inlet = C_d * A * sqrt(2 * g * H_L);
	}

	//Outlet controled discharge (full flow)
	OutletControled(Q_outlet, k_ex, k_en, A_wet, g, H_L, u2, v2, length, R_wet, n);


	//Normal depth at the outlet (suppose infinite flow)
	//using Manning's equation, solve for depth where Q = (1/n)A R^(2/3) S^(1/2)
	// Assume slope S = (d_inlet - d_outlet) / culvert_length
	// Use the minmum discharge (from inlet controled / outlet controled estimation) to calculate the normal depth
	S = max(T(1e-6), (min(zs1, zb1 + height) - min(zs2, zb2 + height)) / length);
	Q_estimated = min(Q_inlet, Q_outlet);

	//An iterative approched to minimize the energy is used to find the normal depth 
	normal_depth(Q_estimated, height, width, n, S, h_n, shape);


		//Critical depth at the outlet (corresponding to Fr=1)
	if (shape == 0) //rectangular
	{
		h_c = pow(Q_estimated * Q_estimated / g / width / width, 1.0 / 3.0);
	}
	if (shape == 1) //circular
	{
		//An iterative approched to minimize the energy is used to find the critical depth in a circular culvert
		critical_depth_circular(Q_estimated, height, h_c);
	}

	//Decision logic to determine the flow regime
	// - If headwater is high and tailwater is low, inlet control
    // - If tailwater is high (above critical and normal depth), outlet control
	if ((zs2 - zb2) < min(h_c, h_n))
	{
		//Outlet controled
		Q = Q_outlet;
	}
	else if (((zs1 - zb1) > h_c) && (zs2 - zb2) > h_c)
	{
		//Inlet controled
		Q = Q_inlet;
	}
	else
	{
		//Transitional or ambiguous case, take the minimum (often inlit controled)
		Q = min(Q_inlet, Q_outlet);
	}

}
template __host__ __device__ void CulvertDischarge<float>(int shape, float width, float height, float length, float zb1, float zb2, float k_ex, float k_en, float C_d, float n, float zs1, float zs2, float u1, float u2, float v1, float v2, float& Q);
template __host__ __device__ void CulvertDischarge<double>(int shape, double width, double height, double length, double zb1, double zb2, double k_ex, double k_en, double C_d, double n, double zs1, double zs2, double u1, double u2, double v1, double v2, double& Q);


__host__ __device__ double manningQ(double A, double R, double n, double S)
{
    return (1.0 / n) * A * pow(R, 2.0 / 3.0) * pow(S, 0.5);
}


template <class T> __host__ __device__ void OutletControled(T& Q, double k_ex, double k_en, double A_wet, double g, double H_L, T u2, T v2, double L, double R_wet, double n)
{
	//Calculation based on energy equation where the discharge is calculated from the head losses.
	T V22 = u2 * u2 + v2 * v2; //V2 square

	if (H_L > 0.0)
	{
		Q = A_wet * sqrt(H_L + k_ex / (2 * g) * V22 / (n * n * L / pow(R_wet, 4.0 / 3.0) + (k_en + k_ex) / (2 * g)));
	}
	else
	{
		Q = 0.0;
	}
}
template __host__ __device__ void OutletControled<float>(float& Q, double k_ex, double k_en, double A_wet, double g, double H_L, float u2, float v2, double L, double R_wet, double n);
template __host__ __device__ void OutletControled<double>(double& Q, double k_ex, double k_en, double A_wet, double g, double H_L, double u2, double v2, double L, double R_wet, double n);

//\template <class T> __host__ __device__ void InletControledUnsubmerged(T& Q, double k_en, double A, double g, T zs1, T u1, T v1)
//{
//	// Inlet controled, unsubmerged
//	Q = K_en * A * sqrt(2 * g * H_1);
//	
//}

//\template <class T> __host__ __device__ void InletControledSubmerged(T& Q, double C_d, double A, double g, T H_L)
//{
//	// Inlet controled, submerged
//	Q = C_d * A * sqrt(2 * g * H_L);
//}

//__host__ __device__ double normal_flow_area(double V, double S, double n)
//{
//	// Calculate the "normal flow" wetted area given the velocity of the flow
//	//using the normal flow formula: $$Q = \frac{1}{n} A_wet R_wet^{2/3} S^{1/2}*1.49 = A_wet*V$$
//	double h_wet;
//	double R_wet;
//
//	R_wet = pow(V * n / 1.49 / pow(S, 0.5), 2.3);
//
//	if rect:
//	A_wet = find_h_rect(b, R_wet);
//    //if cicle:
//
//	return A_wet
//}
//
//__host__ __device__ double find_A_wet_rect(double b, double R_wet)
//{
//	//Calculate the water depth given a hydraulic radius, width (b)
//	
//	double h_wet;
//	double A_wet;
//	h_wet = 2 * r_wet * b / (b - 2 * R_wet);
//	A_wet = h_wet * b;
//	return A_wet
//}



template <class T> __host__ __device__ void rect_geom(T b, T h, T & A, T & P, T & R, T& A_wet, T& P_wet, T& R_wet, T h_wet)
{ 
	//Calculate the full area, perimeter, wetted area and wetted perimeter of a rectangular culvert
	A = b * h;
	P = 2* b + 2 * h;
	R = A / P;
	A_wet = b * h_wet;
	P_wet = 2* b + 2 * h_wet;
	R_wet = A_wet / P_wet;
}
template __host__ __device__ void rect_geom<float>(float b, float h, float& A, float& P, float& R, float& A_wet, float& P_wet, float& R_wet, float h_wet);
template __host__ __device__ void rect_geom<double>(double b, double h, double& A, double& P, double& R, double& A_wet, double& P_wet, double& R_wet, double h_wet);


template <class T>__host__ __device__ void circular_geom(T R ,T & A, T & P, T& A_wet, T& P_wet, T& R_wet, T h_wet)
{
	T K;
	T theta;

	//Calculate the full area, perimeter, wetted area and wetted perimeter of a cylindic culvert, based on https://support.tygron.com/wiki/Culvert_formula_%28Water_Overlay%29
	A = T(3.14159) * R * R;
	P = 2 * T(3.14159) * R;
	R = A / P;
	if (h_wet > 0)
	{
		theta = 2 * acos((R - h_wet) / R); //angle corresponding to the water height
		K = R * R/2 * (theta - sin(theta));
		if (h_wet > R) //more than half full
		{
			A_wet = A - K;
			P_wet = P - R * theta;
		}
		{
			A_wet = K;
			P_wet = R * theta;
		}
		R_wet = A_wet / P_wet;
	}
	else
	{
		A_wet = 0;
		P_wet = 0;
		R_wet = 0;
	}
}
template __host__ __device__ void circular_geom<float>(float R, float& A, float& P, float& A_wet, float& P_wet, float& R_wet, float h_wet);
template __host__ __device__ void circular_geom<double>(double R, double& A, double& P, double& A_wet, double& P_wet, double& R_wet, double h_wet);

template <class T> __host__ __device__ void normal_depth(T Q, T h_culvert, T b, T n, T S,T & h_n, int shape)
{
	//An iterative approched to minimize the energy is used to find the normal depth in a rectangular and circular culvert
	T h_guess;
	int max_iter = 100;
	T Q_calc;
	T A_wet = T(0.0), P_wet = T(0.0), R_wet = T(0.0), A = T(0.0), P = T(0.0), R = h_culvert / 2;

	for (int iter = 0; iter < max_iter; iter++)
	{
		h_guess = iter * h_culvert / max_iter;
		//Calculate A_wet, P_wet, R_wet based on current depth
		if (shape == 0) //rectangular
		{
			rect_geom(b, h_culvert, A, P, R, A_wet, P_wet, R_wet, h_guess);
		}
		else if (shape == 1) //circular
		{
			circular_geom(h_culvert/2, A, P, A_wet, P_wet, R_wet, h_guess);
		}
		//Calculate Q based on Manning's equation
		Q_calc = manningQ(A_wet, R_wet, n, S);
		//Update depth based on difference between calculated Q and target Q
		if (Q_calc >= Q)
		{
			break;
		}
	}
	h_n = h_guess;
}
template __host__ __device__ void normal_depth<float>(float Q, float h_culvert, float b, float n, float S, float& h_n, int shape);
template __host__ __device__ void normal_depth<double>(double Q, double h_culvert, double b, double n, double S, double& h_n, int shape);


template <class T> __host__ __device__ void critical_depth_circular(T Q, T h_culvert, T & h_critical)
{
	//An iterative approched to minimize the energy is used to find the critical depth in a circular culvert
	T h_guess;
	int max_iter = 100;
	T min_E = 0;
	T g = T(9.81);
	T R = h_culvert / T(2.0);
	T E, V ;
	T A_wet = 0, P_wet = 0, R_wet = 0, A = 0, P = 0;

	for (int iter = 0; iter < max_iter; iter++)
	{
		h_guess = iter * h_culvert / max_iter;
		//Calculate A_wet, P_wet, R_wet based on current depth
		circular_geom(R, A, P, A_wet, P_wet, R_wet, h_guess);

		// Test on energy
		if (A_wet > 0)
		{
			V = Q / A_wet;
			E = h_guess + V * V / (2 * g);
			if ((E < min_E) || (iter == 0))
			{
				min_E = E;
				h_critical = h_guess;
			}
		}
	}
}
template __host__ __device__ void critical_depth_circular<float>(float Q, float h_culvert, float& h_critical);
template __host__ __device__ void critical_depth_circular<double>(double Q, double h_culvert, double& h_critical);

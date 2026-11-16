#include "Culverts.h"

// Add near top of file (before kernels)
__device__ inline float atomicAddCompat(float* address, float val)
{
	return atomicAdd(address, val);
}

__device__ inline double atomicAddCompat(double* address, double val)
{
#if __CUDA_ARCH__ >= 600
	return atomicAdd(address, val);
#else
	unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
	unsigned long long int old = *address_as_ull, assumed;

	do
	{
		assumed = old;
		old = atomicCAS(
			address_as_ull,
			assumed,
			__double_as_longlong(val + __longlong_as_double(assumed))
		);
	} while (assumed != old);

	return __longlong_as_double(old);
#endif
}


template <class T> void AddCulverts(Param XParam, double dt, std::vector<Culvert> XCulverts, Model<T> XModel)
{
	dim3 gridDimCulvert(1, 1, 1);
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	int cc;
	int i;

	const int nculverts = static_cast<int>(XCulverts.size());
	if (nculverts == 0)
	{
		return;
	}

	//=====================================
	// 1) Gather culvert endpoint states
	if (XParam.GPUDEVICE >= 0)
	{
		const int threads = 128;
		const int blocks = (nculverts + threads - 1) / threads;

		GatherCulvertStateGPU<<<blocks, threads>>>(nculverts, XModel.culvertsF, XModel.evolv);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	else
	{
		GetCulvertElevCPU(XParam, XCulverts, XModel.culvertsF, XModel.bndblk.nblkculvert, XModel.bndblk.culvert, XModel.blocks, XModel.evolv);
	}

	//=====================================
	// 2) Discharge computation (keep existing physics)
	if (XParam.GPUDEVICE >= 0)
	{
		const int threads = 128;
		const int blocks = (nculverts + threads - 1) / threads;

		DischargeManyCulvertsGPU<<<blocks, threads>>>(XParam, nculverts, dt, XModel.culvertsF);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	else
	{
		DischargeCulvertCPU(XParam, dt, XCulverts, XModel.culvertsF, XModel.bndblk.nblkculvert, XModel.bndblk.culvert);
	}

	//=====================================
	// 3) Apply discharge to model
	if (XParam.GPUDEVICE >= 0)
	{
		const int threads = 128;
		const int blocks = (nculverts + threads - 1) / threads;

		if (XParam.engine == 5)
		{
			ApplyCulvertsImplicitGPU<<<blocks, threads>>>(nculverts, T(dt), XModel.culvertsF, XModel.evolv);
		}
		else
		{
			ApplyCulvertsExplicitGPU<<<blocks, threads>>>(nculverts, XModel.culvertsF, XModel.adv);
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

template <class T> __global__ void InjectCulvertImplicitGPU(Param XParam, T dt, int cc, Culvert XCulvert, CulvertF<T> XCulvertF, int* Culvertblks, BlockP<T> XBlock, EvolvingP<T> XEv)
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
		XEv.h[iinlet] -= XCulvertF.dq[cc] * dt / (XCulvert.dx1 * XCulvert.dx1);
		//printf("After adding: h = % f\n", XAdv.dh[i]);

	}
	if (ix == XCulvert.ix2 && iy == XCulvert.iy2)
	{
		//printf("Before adding: h=%f\n", XAdv.dh[i]);
		XEv.h[ioutlet] += XCulvertF.dq[cc] * dt / (XCulvert.dx2 * XCulvert.dx2);
		//printf("After adding: h=%f\n", XAdv.dh[i]);

	}
	//}

}
template __global__ void InjectCulvertImplicitGPU<float>(Param XParam, float dt, int cc, Culvert XCulvert, CulvertF<float> XCulvertF, int* Culvertblks, BlockP<float> XBlock, EvolvingP<float> XEv);
template __global__ void InjectCulvertImplicitGPU<double>(Param XParam, double dt, int cc, Culvert XCulvert, CulvertF<double> XCulvertF, int* Culvertlks, BlockP<double> XBlock, EvolvingP<double> XEv);



template <class T> __host__ void InjectCulvertCPU(Param XParam, std::vector<Culvert> XCulverts, CulvertF<T>& XCulvertF, int nblkculvert, int* Culvertblks, BlockP<T> XBlock, AdvanceP<T> XAdv)
{
	for (int cc = 0; cc < static_cast<int>(XCulverts.size()); cc++)
	{
		const int i1 = XCulverts[cc].mem1;
		const int i2 = XCulverts[cc].mem2;

		XAdv.dh[i1] -= XCulvertF.dq[cc] * T(XCulverts[cc].invA1);
		XAdv.dh[i2] += XCulvertF.dq[cc] * T(XCulverts[cc].invA2);
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
	for (int cc = 0; cc < static_cast<int>(XCulverts.size()); cc++)
	{
		const int i1 = XCulverts[cc].mem1;
		const int i2 = XCulverts[cc].mem2;

		XCulvertF.h1[cc] = XEv.h[i1];
		XCulvertF.zs1[cc] = XEv.zs[i1];
		XCulvertF.h2[cc] = XEv.h[i2];
		XCulvertF.zs2[cc] = XEv.zs[i2];
	}
}
template __host__ void GetCulvertElevCPU<float>(Param XParam, std::vector<Culvert> XCulverts, CulvertF<float>& XCulvertF, int nblkculvert, int* Culvertblks, BlockP<float> XBlock, EvolvingP<float> XEv);
template __host__ void GetCulvertElevCPU<double>(Param XParam, std::vector<Culvert> XCulverts, CulvertF<double>& XCulvertF,  int nblkculvert, int* Culvertblks, BlockP<double> XBlock, EvolvingP<double> XEv);


template <class T> __global__ void DischargeCulvertGPU(Param XParam, int cc, double dt, CulvertF<T> XCulvertF, Culvert XCulvert)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	//int cci = blockIdx.x;
	//int cc = 0;
	//unsigned int ib = Culvertblks[ibl];

	//for (int cc = 0; cc < XCulverts.size(); cc++)
	if(ix==0 && iy==0)
	{
	//printf("DischargeCulvertGPU:");
	//printf("DischargeCulvertGPU: cc=%i, type=%i, Qmax=%f, dx1=%f, h1=%f\n", cci, XCulvertF.type[cci], XCulvertF.Qmax[cci], XCulvertF.dx1[cci], XCulvertF.h1[cci]);
	//, h2 = % f, zs1 = % f, zs2 = % f
	//XCulvertF.h1[cci], XCulvertF.h2[cci], XCulvertF.zs1[cci], XCulvertF.zs2[cci]

		if (XCulvertF.type[cc] == 0)
		{
			CulvertPump(XCulvertF.Qmax[cc], XCulvertF.dx1[cc], XCulvertF.h1[cc], XCulvertF.h2[cc], XCulvertF.zs1[cc], XCulvertF.zs2[cc], XCulvertF.dq[cc], dt);
		}
		else if (XCulvertF.type[cc] == 1)
		{
			T Q = T(0.0);
			if (XCulvertF.zs1[cc] > XCulvertF.zs2[cc] && XCulvertF.h1[cc] > 0.0)
			{
				CulvertDischarge(XCulvert.shape, T(XCulvert.width), T(XCulvert.height), T(XCulvert.length), T(XCulvert.zb1), T(XCulvert.zb2), T(XCulvert.k_ex), T(XCulvert.k_en), T(XCulvert.C_d), T(XCulvert.n), XCulvertF.zs1[cc], XCulvertF.zs2[cc], XCulvertF.u1[cc], XCulvertF.u2[cc], XCulvertF.v1[cc], XCulvertF.v2[cc], Q);
			
			}
			
			//printf("zs1 = %f; zs2 = %f; h1 = %f; Q = %f\n", XCulvertF.zs1[cc], XCulvertF.zs2[cc], XCulvertF.h1[cc],  Q );

			T Vol1 = XCulvertF.h1[cc] * XCulvert.dx1 * XCulvert.dx1;
			T Qmax = T(Vol1 / XParam.dt);
			XCulvertF.dq[cc] = min(Qmax, max(Q,T(0.0)));
		}
		else if (XCulvertF.type[cc] == 2)
		{
			T Q = T(0.0);
			T Vol, Qmax;
			if (XCulvertF.zs1[cc] > XCulvertF.zs2[cc] && XCulvertF.h1[cc] > 0.0)
			{
				Vol = XCulvertF.h1[cc] * XCulvert.dx1 * XCulvert.dx1;
				Qmax = T(Vol / XParam.dt);
				//printf("Sanitycheck: h1=%f zs1=%f, h2=%f, zs2=%f\n", XCulvertF.h1[cc], XCulvertF.zs1[cc], XCulvertF.h2[cc], XCulvertF.zs2[cc]);
				CulvertDischarge(XCulvert.shape, T(XCulvert.width), T(XCulvert.height), T(XCulvert.length), T(XCulvert.zb1), T(XCulvert.zb2), T(XCulvert.k_ex), T(XCulvert.k_en), T(XCulvert.C_d), T(XCulvert.n), XCulvertF.zs1[cc], XCulvertF.zs2[cc], XCulvertF.u1[cc], XCulvertF.u2[cc], XCulvertF.v1[cc], XCulvertF.v2[cc], Q);
				Q = min(Qmax, max(Q, T(0.0)));
				///printf("Qfinal = %f\n", Q);
			}
			else if (XCulvertF.zs2[cc] > XCulvertF.zs1[cc] && XCulvertF.h2[cc] > 0.0)
			{
				Vol = XCulvertF.h2[cc] * XCulvert.dx2 * XCulvert.dx2;
			 Qmax = T(Vol / XParam.dt);
				CulvertDischarge(XCulvert.shape, T(XCulvert.width), T(XCulvert.height), T(XCulvert.length), T(XCulvert.zb2), T(XCulvert.zb1), T(XCulvert.k_ex), T(XCulvert.k_en), T(XCulvert.C_d), T(XCulvert.n), XCulvertF.zs2[cc], XCulvertF.zs1[cc], XCulvertF.u2[cc], XCulvertF.u1[cc], XCulvertF.v2[cc], XCulvertF.v1[cc], Q);
				Q = -1*min(Qmax, max(Q, T(0.0)));
			}

			
			XCulvertF.dq[cc] = Q * XCulvert.fac;
		}
		//printf("DischargeCulvertGPU after: q=%f\n", XCulvertF.dq[cci]);

	}
}

template __global__ void DischargeCulvertGPU<float>(Param XParam, int cc, double dt, CulvertF<float>  XCulvertF, Culvert XCulvert);
template __global__ void DischargeCulvertGPU<double>(Param XParam, int cc, double dt, CulvertF<double>  XCulvertF, Culvert XCulvert);

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
			Qmax = T(Vol1 / T(XParam.dt));
			XCulvertF.dq[cc] = min(Qmax,Q * XCulverts[cc].fac);
		}
		// Basic 2way culvert
		if (XCulverts[cc].type == 2)
		{
			//NEED TO BE COMPLETED BY ADDED A change in 1/2 location if the flow is reversed.
			CulvertDischarge(XCulverts[cc].shape, T(XCulverts[cc].width), T(XCulverts[cc].height), T(XCulverts[cc].length), T(XCulverts[cc].zb1), T(XCulverts[cc].zb2), T(XCulverts[cc].k_ex), T(XCulverts[cc].k_en), T(XCulverts[cc].C_d), T(XCulverts[cc].n), XCulvertF.zs1[cc], XCulvertF.zs2[cc], XCulvertF.u1[cc], XCulvertF.u2[cc], XCulvertF.v1[cc], XCulvertF.v2[cc], Q);
			Vol1 = XCulvertF.h1[cc] * XCulverts[cc].dx1 * XCulverts[cc].dx1;
			Qmax = T(Vol1 / T(XParam.dt));
			XCulvertF.dq[cc] = min(Qmax, Q * XCulverts[cc].fac);
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
//
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
//		 Q = T(Vol2 * T(dt));
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

	//printf("height = %f\n", height);

	T h_wet;
	T A_wet = T(0.0), P_wet = T(0.0), R_wet = T(0.0), A = T(0.0), P = T(0.0), R = T(width / 2);
	T Q_inlet= T(0.0), Q_outlet= T(0.0), Q_estimated= T(0.0);
	T S; //slope
	T H_L; //head loss
	T g = T(9.81); //gravity
	T H_inlet, H_outlet;
	T h_n; //normal depth
	T h_c; //critical depth


	//printf("zs1=%f, zs2=%f\n ", zs1, zs2);

	//Base geometry
	h_wet = max(min(T(zs1 - zb1), height),0.0); //Assuming inlet controled, the wetted height is equal to the inlet water depth
	if (shape == 0) //rectangular
	{
		rect_geom(width, height, A, P, R, A_wet, P_wet, R_wet, h_wet);
	}
	else if (shape == 1) //circular
	{
		height = width;
		circular_geom(width/T(2.0), A, P, A_wet, P_wet, R_wet, h_wet);
	}

	//printf("hwet = %f; Awet = %f\n", h_wet, A_wet);

	//Inlet controled discharge
	//Test if inlet is submerged or not
	H_inlet = T(zs1 - zb1 + (u1 * u1 + v1 * v1) / (2 * g));
	H_outlet = T(zs2 - zb2 + (u2 * u2 + v2 * v2) / (2 * g));
	H_L = H_inlet - H_outlet; //head loss
	T HLL = T(zs1 - zb1) - T(zs2 - zb2);

	if (zs1 < (zb1 + height))
	{
		//Unsubmerged
		Q_inlet = k_en * A * sqrt(2 * g * H_inlet);
		//printf("Inlet control unsub h_wet=%f; A_wet = %f; Qius=%f\n", h_wet, A_wet, Q_inlet);
	}
	else
	{
		//Submerged
		Q_inlet = C_d * A * sqrt(2 * g * H_L);
	}

	//Outlet controled discharge (full flow)
	OutletControled(Q_outlet, k_ex, k_en, A_wet, g, HLL, u2, v2, length, R_wet, n);


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
		if (h_wet > 0.0 && h_wet < height)
		{
			critical_depth_circular(Q_estimated, height, h_c);
		}
		else
		{
			h_c = h_wet;
		}

		//h_c = pow(Q_estimated * Q_estimated / g / width / width, 1.0 / 3.0);
	}
	//T num;
	//if (((H_inlet)/height)<1.2)//zs1 < (zb1 + height)) //HW / D<1.2
	//{
	//	//Unsubmerged
	//	//Q_inlet = k_en * A * sqrt(2 * g * (H_inlet-h_c));
	//	// Unsubmerged - Form 1
	//	//	# Q = A * D ^ 0.5 * [((HW / D) - (Hc / D) + 0.5 * S) / K] ^ (1 / M)
	//	//	numerator = (HW / D) - (Hc / D) + (0.5 * S)
	//	num = max(H_inlet / height - h_c / height + 0.5 * S, 0.0);
	//	

	//	//Q = A * sqrt(D) * (numerator / K) ^ (1 / M)
	//	T K = 0.0098;
	//	T M = 2.0;
	//	Q_inlet = A_wet * sqrt(h_wet) * pow(num / K, 1 / M);
	//	printf("Inlet control unsub h_wet=%f; A_wet = %f; num=%f; h_c=%f,h_n=%f,  Qius=%f\n", h_wet, A_wet,num,h_c,h_n,Q_inlet);
	//}
	//else
	//{
	//	//Submerged
	//	//Q_inlet = C_d * A * sqrt(2 * g * H_L);
	//	//numerator = (HW / D) - Y + (0.5 * S)
	//	//Q = A * sqrt(D) * sqrt(numerator / c)
	//	T c = 0.0398;
	//	T Y = 0.67;
	//	num = max(H_inlet / height - Y + 0.5 * S,0.0);
	//	Q_inlet = A_wet * sqrt(height) * sqrt(num / c);

	//}

	//Decision logic to determine the flow regime
	// - If headwater is high and tailwater is low, inlet control
    // - If tailwater is high (above critical and normal depth), outlet control
	//if ((zs2 - zb2) < min(h_c, h_n))
	//{
	//	//Outlet controled
	//	Q = Q_outlet;
	//	printf("Outlet control\n");
	//}
	//else if (((zs1 - zb1) > h_c) && (zs2 - zb2) > h_c)
	//{
	//	//Inlet controled
	//	Q = Q_inlet;
	//	printf("Inlet control: Q=%f\n", Q_inlet);
	//}
	//else
	//{
	//	//Transitional or ambiguous case, take the minimum (often inlet controled)
	//	Q = min(Q_inlet, Q_outlet);
	//	printf("Transition flow\n");
	//}

	//printf("Sanity check h_wet=%f; A_wet = %f;  Rwet=%f,h_n=%f, HLL=%f;  Qius=%f\n", h_wet, A_wet, R_wet, h_n,HLL, Q_inlet);
	if ((zs2 - zb2) > height)
	{
		Q = Q_outlet;
		//printf("Outlet control:Q=%f\n", Q_outlet);
	}
	else if (h_n <= h_c)
	{
		Q = min(Q_inlet, Q_outlet);
		//printf("Mixed control: Q=%f\n", Q);
	}
	else
	{
		
		Q = Q_outlet;
		//printf("Outlet control:Q=%f\n", Q_outlet);
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
	T C = T(0.87);
	T friction_const =T(19.62);
	if (H_L > 0.0)
	{
		//Alice Eq? Something's wrong but maybe because Q should not depend on V22 ?
		//Q = A_wet * sqrt(2 * g * (H_inlet-h_c));
		// Unsubmerged - Form 1
		//	# Q = A * D ^ 0.5 * [((HW / D) - (Hc / D) + 0.5 * S) / K] ^ (1 / M)
		//	numerator = (HW / D) - (Hc / D) + (0.5 * S)
		//T numerator = (H_L + k_ex / (2 * g) * V22 / (n * n * L / pow(R_wet, 4.0 / 3.0) + (k_en + k_ex) / (2 * g)));
		//printf("Outletcontrol: Hl=%f\n", H_L);
		//printf("Outletcontrol: Q=%f, Awet=%f\n", Q, A_wet);
		//Q = Q * pow(H_L / (1 + T(0.87) * pow(numerator, -2.0)), 0.5);
		//Q = A_wet * sqrt(H_L / (1 + (friction_const * n * n * L) / (pow(R_wet, T(4.0 / 3.0))) + (k_ex + k_en) / (2 * g)));


		//USGS type 4 formula seem to work OK
		//T friction_term = (T(2.0) * g * n *n * L) / (pow(R_wet, T(4.0 / 3.0)));
		//Q = C * A_wet * sqrt((T(2.0) * g * H_L) / (1 + friction_term));

		// HDS-5 Outlet Control
		T loss_sum = k_en + k_ex + (friction_const * n * n * L) / (pow(R_wet, T(4.0 / 3.0)));
		Q = A_wet * sqrt((2 * g * H_L) / loss_sum);

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
	//R = A / P;
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

	/*function calculate_dc_circular(Q, D; g = 9.81, tol = 1e-5)
		# Target value for the section factor : Z = Q / sqrt(g)
		target_Z = Q / sqrt(g)

		# Helper to calculate Z(y) = A ^ 1.5 / sqrt(T)
		function section_factor(y, D)
		if y <= 0 return 0.0 end
			if y >= D return Inf end # Full pipe has no "critical depth" in this model

				theta = 2 * acos(1 - 2 * y / D)
				area = (D ^ 2 / 8) * (theta - sin(theta))
				top_width = D * sin(theta / 2)

				return (area ^ 1.5) / sqrt(top_width)
				end

				# Bisection search bounds
				low = 0.001 * D
				high = 0.999 * D
				mid = (low + high) / 2

				# Iterate until the depth converges
				for i in 1:100
					mid = (low + high) / 2
					z_mid = section_factor(mid, D)

					if abs(z_mid - target_Z) < tol
						return mid
						end

						if z_mid < target_Z
							low = mid
						else
							high = mid
							end
							end

							return mid
							end*/

	int max_iter = 100;
	T g = T(9.81);
	T target_Z = Q / sqrt(g);
	T A_wet = 0, P_wet = 0, R_wet = 0, A = 0, P = 0;
	T tol = 1e-5;
	T Z, theta, top_width;
	T D = h_culvert;///WARNING this is not intuitive
	T R = D / T(2.0);

	T low = 0.001 * D;
	T high = 0.999 * D;
	T mid;

	for (int iter = 0; iter < max_iter; iter++)
	{
		mid = (low + high) / 2;
		theta = 2 * acos(1 - 2 * mid / D);
		top_width = D * sin(theta / 2);
		circular_geom(R, A, P, A_wet, P_wet, R_wet, mid);
		Z = (pow(A_wet , 1.5)) / sqrt(top_width);
		if (abs(Z - target_Z) < tol)
		{
			break;
		}
		else
		{
			if (Z < target_Z)
			{
				low = mid;
			}
			else
			{
				high = mid;
			}
		}

	}

	h_critical = mid;



	// Alice's Algo here
	//T h_guess;
	//int max_iter = 100;
	//T min_E = 0;
	//T g = T(9.81);
	//T R = h_culvert / T(2.0);
	//T E, V ;
	//T A_wet = 0, P_wet = 0, R_wet = 0, A = 0, P = 0;

	//for (int iter = 0; iter < max_iter; iter++)
	//{
	//	h_guess = iter * h_culvert / max_iter;
	//	//Calculate A_wet, P_wet, R_wet based on current depth
	//	circular_geom(R, A, P, A_wet, P_wet, R_wet, h_guess);

	//	// Test on energy
	//	if (A_wet > 0)
	//	{
	//		V = Q / A_wet;
	//		E = h_guess + V * V / (2 * g);
	//		if ((E < min_E) || (iter == 0))
	//		{
	//			min_E = E;
	//			h_critical = h_guess;
	//		}
	//	}
	//}
}
template __host__ __device__ void critical_depth_circular<float>(float Q, float h_culvert, float& h_critical);
template __host__ __device__ void critical_depth_circular<double>(double Q, double h_culvert, double& h_critical);

template <class T> __global__ void GatherCulvertStateGPU(int nculverts, CulvertF<T> XCulvertF, EvolvingP<T> XEv)
{
	int cc = blockIdx.x * blockDim.x + threadIdx.x;
	if (cc >= nculverts) return;

	const int i1 = XCulvertF.mem1[cc];
	const int i2 = XCulvertF.mem2[cc];

	XCulvertF.h1[cc] = XEv.h[i1];
	XCulvertF.zs1[cc] = XEv.zs[i1];
	XCulvertF.u1[cc] = XEv.u[i1];
	XCulvertF.v1[cc] = XEv.v[i1];

	XCulvertF.h2[cc] = XEv.h[i2];
	XCulvertF.zs2[cc] = XEv.zs[i2];
	XCulvertF.u2[cc] = XEv.u[i2];
	XCulvertF.v2[cc] = XEv.v[i2];
}

template <class T> __global__ void ApplyCulvertsExplicitGPU(int nculverts, CulvertF<T> XCulvertF, AdvanceP<T> XAdv)
{
	int cc = blockIdx.x * blockDim.x + threadIdx.x;
	if (cc >= nculverts) return;

	const T dq = XCulvertF.dq[cc];
	atomicAddCompat(&XAdv.dh[XCulvertF.mem1[cc]], -dq * XCulvertF.invA1[cc]);
	atomicAddCompat(&XAdv.dh[XCulvertF.mem2[cc]],  dq * XCulvertF.invA2[cc]);
}

template <class T> __global__ void ApplyCulvertsImplicitGPU(int nculverts, T dt, CulvertF<T> XCulvertF, EvolvingP<T> XEv)
{
	int cc = blockIdx.x * blockDim.x + threadIdx.x;
	if (cc >= nculverts) return;

	const T vol = XCulvertF.dq[cc] * dt;
	atomicAddCompat(&XEv.h[XCulvertF.mem1[cc]], -vol * XCulvertF.invA1[cc]);
	atomicAddCompat(&XEv.h[XCulvertF.mem2[cc]],  vol * XCulvertF.invA2[cc]);
}

template <class T>
__global__ void DischargeManyCulvertsGPU(Param XParam, int nculverts, double dt, CulvertF<T> XCulvertF)
{
	int cc = blockIdx.x * blockDim.x + threadIdx.x;
	if (cc >= nculverts) return;

	const int type = XCulvertF.type[cc];

	if (type == 0)
	{
		CulvertPump(XCulvertF.Qmax[cc], XCulvertF.dx1[cc], XCulvertF.h1[cc], XCulvertF.h2[cc], XCulvertF.zs1[cc], XCulvertF.zs2[cc], XCulvertF.dq[cc], dt);
	}
	else if (type == 1)
	{
		T Q = T(0.0);
		if (XCulvertF.zs1[cc] > XCulvertF.zs2[cc] && XCulvertF.h1[cc] > 0.0)
		{
			CulvertDischarge(
				XCulvertF.shape[cc],
				T(XCulvertF.width[cc]),
				T(XCulvertF.height[cc]),
				T(XCulvertF.length[cc]),
				T(XCulvertF.zb1[cc]),
				T(XCulvertF.zb2[cc]),
				T(XCulvertF.k_ex[cc]),
				T(XCulvertF.k_en[cc]),
				T(XCulvertF.C_d[cc]),
				T(XCulvertF.n[cc]),
				XCulvertF.zs1[cc], XCulvertF.zs2[cc],
				XCulvertF.u1[cc], XCulvertF.u2[cc],
				XCulvertF.v1[cc], XCulvertF.v2[cc], Q);
		}
		T Vol1 = XCulvertF.h1[cc] * T(XCulvertF.dx1[cc] * XCulvertF.dx1[cc]);
		T Qmax = T(Vol1 / T(XParam.dt));
		XCulvertF.dq[cc] = min(Qmax, max(Q, T(0.0)));
	}
	else if (type == 2)
	{
		T Q = T(0.0);
		T Vol, Qmax;
		if (XCulvertF.zs1[cc] > XCulvertF.zs2[cc] && XCulvertF.h1[cc] > 0.0)
		{
			Vol = XCulvertF.h1[cc] * T(XCulvertF.dx1[cc] * XCulvertF.dx1[cc]);
			Qmax = T(Vol / T(XParam.dt));
			CulvertDischarge(
				XCulvertF.shape[cc],
				T(XCulvertF.width[cc]),
				T(XCulvertF.height[cc]),
				T(XCulvertF.length[cc]),
				T(XCulvertF.zb1[cc]),
				T(XCulvertF.zb2[cc]),
				T(XCulvertF.k_ex[cc]),
				T(XCulvertF.k_en[cc]),
				T(XCulvertF.C_d[cc]),
				T(XCulvertF.n[cc]),
				XCulvertF.zs1[cc], XCulvertF.zs2[cc],
				XCulvertF.u1[cc], XCulvertF.u2[cc],
				XCulvertF.v1[cc], XCulvertF.v2[cc], Q);
			Q = min(Qmax, max(Q, T(0.0)));
		}
		else if (XCulvertF.zs2[cc] > XCulvertF.zs1[cc] && XCulvertF.h2[cc] > 0.0)
		{
			Vol = XCulvertF.h2[cc] * T(XCulvertF.dx2[cc] * XCulvertF.dx2[cc]);
			Qmax = T(Vol / T(XParam.dt));
			CulvertDischarge(
				XCulvertF.shape[cc],
				T(XCulvertF.width[cc]),
				T(XCulvertF.height[cc]),
				T(XCulvertF.length[cc]),
				T(XCulvertF.zb2[cc]),
				T(XCulvertF.zb1[cc]),
				T(XCulvertF.k_ex[cc]),
				T(XCulvertF.k_en[cc]),
				T(XCulvertF.C_d[cc]),
				T(XCulvertF.n[cc]),
				XCulvertF.zs2[cc], XCulvertF.zs1[cc],
				XCulvertF.u2[cc], XCulvertF.u1[cc],
				XCulvertF.v2[cc], XCulvertF.v1[cc], Q);
			Q = -min(Qmax, max(Q, T(0.0)));
		}
		XCulvertF.dq[cc] = Q * T(XCulvertF.fac[cc]);
	}
}
template __global__ void DischargeManyCulvertsGPU<float>(Param XParam, int nculverts, double dt, CulvertF<float> XCulvertF);
template __global__ void DischargeManyCulvertsGPU<double>(Param XParam, int nculverts, double dt, CulvertF<double> XCulvertF);



#include "Culverts.h"


template <class T> __host__ void AddCulverts(Param XParam, Loop<T> XLoop, std::vector<Culvert> XCulverts, Model<T> XModel)
{
	dim3 gridDimRiver(XModel.bndblk.nblkriver, 1, 1);
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	T qnow;

	// Get the elevation for each culvert edge

	// Calculation of the transfert of water
	// Application of the result to h


	for (int cc = 0; cc < XCulverts.size(); cc++)
	{
		//
		int bndstep = 0;
		double difft = XRivers[Rin].flowinput[bndstep].time - XLoop.totaltime;
		while (difft <= 0.0) // danger?
		{
			bndstep++;
			difft = XRivers[Rin].flowinput[bndstep].time - XLoop.totaltime;
		}

		qnow = T(interptime(XRivers[Rin].flowinput[bndstep].q, XRivers[Rin].flowinput[max(bndstep - 1, 0)].q, XRivers[Rin].flowinput[bndstep].time - XRivers[Rin].flowinput[max(bndstep - 1, 0)].time, XLoop.totaltime - XRivers[Rin].flowinput[max(bndstep - 1, 0)].time));

		if (XParam.GPUDEVICE >= 0)
		{
			InjectRiverGPU << <gridDimRiver, blockDim, 0 >> > (XParam, XRivers[Rin], qnow, XModel.bndblk.river, XModel.blocks, XModel.adv);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			InjectRiverCPU(XParam, XRivers[Rin], qnow, XModel.bndblk.nblkriver, XModel.bndblk.river, XModel.blocks, XModel.adv);
		}
	}
}
template __host__ void AddRiverForcing<float>(Param XParam, Loop<float> XLoop, std::vector<River> XRivers, Model<float> XModel);
template __host__ void AddRiverForcing<double>(Param XParam, Loop<double> XLoop, std::vector<River> XRivers, Model<double> XModel);


template <class T> __global__ void InjectRiverGPU(Param XParam, River XRiver, T qnow, int* Riverblks, BlockP<T> XBlock, AdvanceP<T> XAdv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;

	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = Riverblks[ibl];



	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	T delta = calcres(T(XParam.dx), XBlock.level[ib]);
	T xl, yb, xr, yt, xllo, yllo;
	xllo = XParam.xo + XBlock.xo[ib];
	yllo = XParam.yo + XBlock.yo[ib];

	xl = xllo + ix * delta - 0.5 * delta;
	yb = yllo + iy * delta - 0.5 * delta;

	xr = xllo + ix * delta + 0.5 * delta;
	yt = yllo + iy * delta + 0.5 * delta;
	// the conditions are that the discharge area as defined by the user have to include at least a model grid node
	// This could be really annoying and there should be a better way to deal wiith this like polygon intersection
	//if (xx >= XForcing.rivers[Rin].xstart && xx <= XForcing.rivers[Rin].xend && yy >= XForcing.rivers[Rin].ystart && yy <= XForcing.rivers[Rin].yend)
	if (OBBdetect(xl, xr, yb, yt, T(XRiver.xstart), T(XRiver.xend), T(XRiver.ystart), T(XRiver.yend)))
	{

		XAdv.dh[i] += qnow / XRiver.disarea;

	}



}
template __global__ void InjectRiverGPU<float>(Param XParam, River XRiver, float qnow, int* Riverblks, BlockP<float> XBlock, AdvanceP<float> XAdv);
template __global__ void InjectRiverGPU<double>(Param XParam, River XRiver, double qnow, int* Riverblks, BlockP<double> XBlock, AdvanceP<double> XAdv);

template <class T> __host__ void InjectRiverCPU(Param XParam, River XRiver, T qnow, int nblkriver, int* Riverblks, BlockP<T> XBlock, AdvanceP<T> XAdv)
{
	unsigned int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	T xllo, yllo, xl, yb, xr, yt, levdx;

	for (int ibl = 0; ibl < nblkriver; ibl++)
	{
		ib = Riverblks[ibl];

		levdx = calcres(T(XParam.dx), XBlock.level[ib]);

		xllo = T(XParam.xo + XBlock.xo[ib]);
		yllo = T(XParam.yo + XBlock.yo[ib]);



		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

				//T delta = calcres(T(XParam.dx), XBlock.level[ib]);

				//T x = XParam.xo + XBlock.xo[ib] + ix * delta;
				//T y = XParam.yo + XBlock.yo[ib] + iy * delta;

				//if (x >= XRiver.xstart && x <= XRiver.xend && y >= XRiver.ystart && y <= XRiver.yend)
				xl = xllo + ix * levdx - T(0.5) * levdx;
				yb = yllo + iy * levdx - T(0.5) * levdx;

				xr = xllo + ix * levdx + T(0.5) * levdx;
				yt = yllo + iy * levdx + T(0.5) * levdx;
				// the conditions are that the discharge area as defined by the user have to include at least a model grid node
				// This could be really annoying and there should be a better way to deal wiith this like polygon intersection
				//if (xx >= XForcing.rivers[Rin].xstart && xx <= XForcing.rivers[Rin].xend && yy >= XForcing.rivers[Rin].ystart && yy <= XForcing.rivers[Rin].yend)
				if (OBBdetect(xl, xr, yb, yt, T(XRiver.xstart), T(XRiver.xend), T(XRiver.ystart), T(XRiver.yend)))
				{
					XAdv.dh[i] += qnow / T(XRiver.disarea);

				}
			}
		}
	}


}
template __host__ void InjectRiverCPU<float>(Param XParam, River XRiver, float qnow, int nblkriver, int* Riverblks, BlockP<float> XBlock, AdvanceP<float> XAdv);
template __host__ void InjectRiverCPU<double>(Param XParam, River XRiver, double qnow, int nblkriver, int* Riverblks, BlockP<double> XBlock, AdvanceP<double> XAdv);

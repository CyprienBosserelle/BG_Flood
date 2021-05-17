
#include "MemManagement.h"

__host__ int memloc(Param XParam, int i, int j, int ib)
{
	return (i+XParam.halowidth) + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
}


__host__ __device__ int memloc(int halowidth, int blkmemwidth, int i, int j, int ib)
{
	return (i + halowidth) + (j + halowidth) * blkmemwidth + ib * (blkmemwidth* blkmemwidth);
}

template <class T> __host__ void AllocateCPU(int nx, int ny, T *&zb)
{
	zb = (T *)malloc(nx*ny * sizeof(T));
	if (!zb)
	{
		fprintf(stderr, "Memory allocation failure\n");

		exit(EXIT_FAILURE);
	}
}


template <class T> __host__ void AllocateCPU(int nx, int ny, T *&zs, T *&h, T *&u, T *&v)
{

	AllocateCPU(nx, ny, zs);
	AllocateCPU(nx, ny, h);
	AllocateCPU(nx, ny, u);
	AllocateCPU(nx, ny, v);

}

template void AllocateCPU<double>(int nx, int ny, double *&zs, double *&h, double *&u, double *&v);
template void AllocateCPU<float>(int nx, int ny, float *&zs, float *&h, float *&u, float *&v);
template void AllocateCPU<int>(int nx, int ny, int *&zs, int *&h, int *&u, int *&v);

template <class T> __host__
void AllocateCPU(int nx, int ny, GradientsP<T>& Grad)
{
	AllocateCPU(nx, ny, Grad.dhdx, Grad.dzsdx, Grad.dudx, Grad.dvdx);
	AllocateCPU(nx, ny, Grad.dhdy, Grad.dzsdy, Grad.dudy, Grad.dvdy);
}
template void AllocateCPU<float>(int nx, int ny, GradientsP<float>& Grad);
template void AllocateCPU<double>(int nx, int ny, GradientsP<double>& Grad);

template <class T> void AllocateCPU(int nblk, int blksize, EvolvingP<T> & Ev)
{
	AllocateCPU(nblk, blksize, Ev.h, Ev.zs, Ev.u, Ev.v);
}



template <class T>
void AllocateCPU(int nblk, int blksize, Param XParam, Model<T>& XModel)
{
	// Allocate blocks data 
	AllocateCPU(nblk, blksize, XModel.evolv);
	AllocateCPU(nblk, blksize, XModel.evolv_o);

	AllocateCPU(nblk, blksize, XModel.grad.dhdy, XModel.grad.dzsdy, XModel.grad.dudy, XModel.grad.dvdy);
	AllocateCPU(nblk, blksize, XModel.grad.dhdx, XModel.grad.dzsdx, XModel.grad.dudx, XModel.grad.dvdx);

	AllocateCPU(nblk, blksize, XModel.flux.Fhu, XModel.flux.Fhv, XModel.flux.Fqux, XModel.flux.Fquy);

	AllocateCPU(nblk, blksize, XModel.flux.Fqvx, XModel.flux.Fqvy, XModel.flux.Su, XModel.flux.Sv);

	AllocateCPU(nblk, blksize, XModel.zb, XModel.adv.dh, XModel.adv.dhu, XModel.adv.dhv);

	AllocateCPU(nblk, blksize, XModel.cf, XModel.time.arrmax, XModel.time.arrmin, XModel.time.dtmax);
	

	//Allocate block info
	AllocateCPU(nblk, 1, XModel.blocks.active);
	AllocateCPU(nblk, 1, XModel.blocks.level);

	AllocateCPU(nblk, 1, XModel.blocks.BotLeft, XModel.blocks.BotRight, XModel.blocks.LeftBot, XModel.blocks.LeftTop);
	AllocateCPU(nblk, 1, XModel.blocks.RightBot, XModel.blocks.RightTop, XModel.blocks.TopLeft, XModel.blocks.TopRight);

	AllocateCPU(nblk, 1, XModel.blocks.xo);
	AllocateCPU(nblk, 1, XModel.blocks.yo);

	// do allocate 1 mask block (block with at least 1 empty neighbourhood) 
	// this will be reallocated eventually
	AllocateCPU(1, 1, XModel.blocks.mask.side);
	AllocateCPU(1, 1, XModel.blocks.mask.blks);

	// If no adatptation ignore this!
	if (XParam.maxlevel != XParam.minlevel)
	{
		AllocateCPU(nblk, 1, XModel.adapt.availblk, XModel.adapt.csumblk, XModel.adapt.invactive, XModel.adapt.newlevel);
		AllocateCPU(nblk, 1, XModel.adapt.coarsen);
		AllocateCPU(nblk, 1, XModel.adapt.refine);
	}
	

	if (XParam.atmpforcing)
	{
		AllocateCPU(nblk, blksize, XModel.datmpdx);
		AllocateCPU(nblk, blksize, XModel.datmpdy);
	}


	if (XParam.outmax)
	{
		AllocateCPU(nblk, blksize, XModel.evmax);
	}
	if (XParam.outmean)
	{
		AllocateCPU(nblk, blksize, XModel.evmean);
	}
	
	if (XParam.outvort)
	{
		AllocateCPU(nblk, blksize, XModel.vort);
	}

	if (XParam.TSnodesout.size() > 0)
	{
		// Timeseries output temporary storage
		int storage = XParam.maxTSstorage;
		AllocateCPU(storage, 1, XModel.TSstore);
	}
	if (XParam.nrivers > 0)
	{
		//this will be eventually reallocated later
		AllocateCPU(1, 1, XModel.bndblk.river);
		XModel.bndblk.nblkriver = 1;
	}
	// preallocate 1 block along all bnds
	//this will be eventually reallocated later
	//AllocateCPU(1, 1, XModel.bndblk.left);
	//AllocateCPU(1, 1, XModel.bndblk.right);
	//AllocateCPU(1, 1, XModel.bndblk.top);
	//AllocateCPU(1, 1, XModel.bndblk.bot);

}

template void AllocateCPU<float>(int nblk, int blksize, Param XParam, Model<float>& XModel);
template void AllocateCPU<double>(int nblk, int blksize, Param XParam, Model<double>& XModel);




template <class T> void ReallocArray(int nblk, int blksize, T* & zb)
{
	//
	if (nblk > 0)
	{
		zb = (T*)realloc(zb, nblk * blksize * sizeof(T));
		if (zb == NULL)
		{
			fprintf(stderr, "Memory reallocation failure\n");

			exit(EXIT_FAILURE);
		}
	}
	//return nblkmem
}

template <class T> void ReallocArray(int nblk, int blksize, T*& zs, T*& h, T*& u, T*& v)
{
	//

	ReallocArray(nblk, blksize, zs);
	ReallocArray(nblk, blksize, h);
	ReallocArray(nblk, blksize, u);
	ReallocArray(nblk, blksize, v);
	//return nblkmem
}

template void ReallocArray<int>(int nblk, int blksize, int* & zs, int*& h, int*& u, int*& v);
template void ReallocArray<float>(int nblk, int blksize, float* & zs, float*& h, float*& u, float*& v );
template void ReallocArray<double>(int nblk, int blksize, double* & zs, double*& h, double*& u, double*& v);

template <class T> void ReallocArray(int nblk, int blksize, EvolvingP<T>& Ev)
{
	ReallocArray(nblk, blksize, Ev.zs, Ev.h, Ev.u, Ev.v);
}
template void ReallocArray<float>(int nblk, int blksize, EvolvingP<float>& Ev);
template void ReallocArray<double>(int nblk, int blksize, EvolvingP<double>& Ev);


template <class T>
void ReallocArray(int nblk, int blksize, Param XParam, Model<T>& XModel)
{
	// Allocate blocks data 
	ReallocArray(nblk, blksize, XModel.evolv);
	ReallocArray(nblk, blksize, XModel.evolv_o);

	ReallocArray(nblk, blksize, XModel.grad.dhdy, XModel.grad.dzsdy, XModel.grad.dudy, XModel.grad.dvdy);
	ReallocArray(nblk, blksize, XModel.grad.dhdx, XModel.grad.dzsdx, XModel.grad.dudx, XModel.grad.dvdx);

	ReallocArray(nblk, blksize, XModel.flux.Fhu, XModel.flux.Fhv, XModel.flux.Fqux, XModel.flux.Fquy);

	ReallocArray(nblk, blksize, XModel.flux.Fqvx, XModel.flux.Fqvy, XModel.flux.Su, XModel.flux.Sv);

	ReallocArray(nblk, blksize, XModel.zb, XModel.adv.dh, XModel.adv.dhu, XModel.adv.dhv);

	ReallocArray(nblk, blksize, XModel.cf, XModel.time.arrmax, XModel.time.arrmin, XModel.time.dtmax);


	//Allocate block info
	ReallocArray(nblk, 1, XModel.blocks.active);
	ReallocArray(nblk, 1, XModel.blocks.level);

	ReallocArray(nblk, 1, XModel.blocks.BotLeft, XModel.blocks.BotRight, XModel.blocks.LeftBot, XModel.blocks.LeftTop);
	ReallocArray(nblk, 1, XModel.blocks.RightBot, XModel.blocks.RightTop, XModel.blocks.TopLeft, XModel.blocks.TopRight);

	ReallocArray(nblk, 1, XModel.blocks.xo);
	ReallocArray(nblk, 1, XModel.blocks.yo);

	// If no adatptation ignore this!
	if (XParam.maxlevel != XParam.minlevel)
	{
		ReallocArray(nblk, 1, XModel.adapt.availblk, XModel.adapt.csumblk, XModel.adapt.invactive, XModel.adapt.newlevel);
		ReallocArray(nblk, 1, XModel.adapt.coarsen);
		ReallocArray(nblk, 1, XModel.adapt.refine);
	}


	if (XParam.atmpforcing)
	{
		ReallocArray(nblk, blksize, XModel.datmpdx);
		ReallocArray(nblk, blksize, XModel.datmpdy);
	}


	if (XParam.outmax)
	{
		ReallocArray(nblk, blksize, XModel.evmax);
	}
	if (XParam.outmean)
	{
		ReallocArray(nblk, blksize, XModel.evmean);
	}

	//ReallocArray(nx, ny, XModel.);



}

template void ReallocArray<float>(int nblk, int blksize, Param XParam, Model<float>& XModel);
template void ReallocArray<double>(int nblk, int blksize, Param XParam, Model<double>& XModel);


template <class T> void AllocateGPU(int nx, int ny, T*& z_g)
{
	CUDA_CHECK(cudaMalloc((void**)& z_g, nx * ny * sizeof(T)));
}

template <class T> void AllocateGPU(int nx, int ny, T*& zs, T*& h, T*& u, T*& v)
{

	AllocateGPU(nx, ny, zs);
	AllocateGPU(nx, ny, h);
	AllocateGPU(nx, ny, u);
	AllocateGPU(nx, ny, v);

}

template void AllocateGPU<double>(int nx, int ny, double*& zs, double*& h, double*& u, double*& v);
template void AllocateGPU<float>(int nx, int ny, float*& zs, float*& h, float*& u, float*& v);
template void AllocateGPU<int>(int nx, int ny, int*& zs, int*& h, int*& u, int*& v);

template <class T> 
void AllocateGPU(int nx, int ny, GradientsP<T>& Grad)
{
	AllocateGPU(nx, ny, Grad.dhdx, Grad.dzsdx, Grad.dudx, Grad.dvdx);
	AllocateGPU(nx, ny, Grad.dhdy, Grad.dzsdy, Grad.dudy, Grad.dvdy);
}
template void AllocateGPU<float>(int nx, int ny, GradientsP<float>& Grad);
template void AllocateGPU<double>(int nx, int ny, GradientsP<double>& Grad);

template <class T> void AllocateGPU(int nblk, int blksize, EvolvingP<T>& Ev)
{
	AllocateGPU(nblk, blksize, Ev.h, Ev.zs, Ev.u, Ev.v);
}

template <class T>
void AllocateGPU(int nblk, int blksize, Param XParam, Model<T>& XModel)
{
	// Allocate blocks data 
	AllocateGPU(nblk, blksize, XModel.evolv);
	AllocateGPU(nblk, blksize, XModel.evolv_o);

	AllocateGPU(nblk, blksize, XModel.grad);
	AllocateGPU(nblk, blksize, XModel.flux.Fhu, XModel.flux.Fhv, XModel.flux.Fqux, XModel.flux.Fquy);

	AllocateGPU(nblk, blksize, XModel.flux.Fqvx, XModel.flux.Fqvy, XModel.flux.Su, XModel.flux.Sv);

	AllocateGPU(nblk, blksize, XModel.zb, XModel.adv.dh, XModel.adv.dhu, XModel.adv.dhv);

	AllocateGPU(nblk, blksize, XModel.cf, XModel.time.arrmax, XModel.time.arrmin, XModel.time.dtmax);


	//Allocate block info
	AllocateGPU(nblk, 1, XModel.blocks.active);
	AllocateGPU(nblk, 1, XModel.blocks.level);

	AllocateGPU(nblk, 1, XModel.blocks.BotLeft, XModel.blocks.BotRight, XModel.blocks.LeftBot, XModel.blocks.LeftTop);
	AllocateGPU(nblk, 1, XModel.blocks.RightBot, XModel.blocks.RightTop, XModel.blocks.TopLeft, XModel.blocks.TopRight);

	AllocateGPU(nblk, 1, XModel.blocks.xo);
	AllocateGPU(nblk, 1, XModel.blocks.yo);

	// If no adatptation ignore this!
	/*
	if (XParam.maxlevel != XParam.minlevel)
	{
		AllocateGPU(nblk, 1, XModel.adapt.availblk, XModel.adapt.csumblk, XModel.adapt.invactive, XModel.adapt.newlevel);
		AllocateGPU(nblk, 1, XModel.adapt.coarsen);
		AllocateGPU(nblk, 1, XModel.adapt.refine);
	}
	*/

	

	if (XParam.atmpforcing)
	{
		AllocateGPU(nblk, blksize, XModel.datmpdx);
		AllocateGPU(nblk, blksize, XModel.datmpdy);
	}


	if (XParam.outmax)
	{
		AllocateGPU(nblk, blksize, XModel.evmax);
	}
	if (XParam.outmean)
	{
		AllocateGPU(nblk, blksize, XModel.evmean);
	}

	
	if (XParam.outvort)
	{
		AllocateGPU(nblk, blksize, XModel.vort);
	}

	if (XParam.TSnodesout.size() > 0)
	{
		// Timeseries output temporary storage
		int storage = XParam.maxTSstorage;
		AllocateGPU(storage, 1, XModel.TSstore);
	}

	// Allocate textures for boundary and forcing is done in init forcing



}

template void AllocateGPU<float>(int nblk, int blksize, Param XParam, Model<float>& XModel);
template void AllocateGPU<double>(int nblk, int blksize, Param XParam, Model<double>& XModel);


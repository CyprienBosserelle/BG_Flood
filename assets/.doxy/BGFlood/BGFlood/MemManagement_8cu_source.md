

# File MemManagement.cu

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**MemManagement.cu**](MemManagement_8cu.md)

[Go to the documentation of this file](MemManagement_8cu.md)


```C++

#include "MemManagement.h"


#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

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

template <class T> __host__ void FillCPU(int nx, int ny,T fillval, T*& zb)
{
    for (int ix = 0; ix < nx; ix++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            zb[iy * nx + ix] = fillval;
        }
    }
}
template void FillCPU<double>(int nx, int ny, double fillval, double*& zb);
template void FillCPU<float>(int nx, int ny, float fillval, float*& zb);
template void FillCPU<int>(int nx, int ny, int fillval, int*& zb);

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

template <class T> __host__ void AllocateCPU(int nx, int ny, T*& zs, T*& h, T*& u, T*& v, T*& U, T*& hU)
{

    AllocateCPU(nx, ny, zs);
    AllocateCPU(nx, ny, h);
    AllocateCPU(nx, ny, u);
    AllocateCPU(nx, ny, v);
    AllocateCPU(nx, ny, U);
    AllocateCPU(nx, ny, hU);
}

template void AllocateCPU<double>(int nx, int ny, double*& zs, double*& h, double*& u, double*& v, double*& U, double*& hU);
template void AllocateCPU<float>(int nx, int ny, float*& zs, float*& h, float*& u, float*& v, float*& U, float*& hU);
template void AllocateCPU<int>(int nx, int ny, int*& zs, int*& h, int*& u, int*& v, int*& U, int*& hU);

template <class T> __host__
void AllocateCPU(int nx, int ny, GradientsP<T>& Grad)
{
    AllocateCPU(nx, ny, Grad.dhdx, Grad.dzsdx, Grad.dudx, Grad.dvdx);
    AllocateCPU(nx, ny, Grad.dhdy, Grad.dzsdy, Grad.dudy, Grad.dvdy);
    
    AllocateCPU(nx, ny, Grad.dzbdx);
    AllocateCPU(nx, ny, Grad.dzbdy);
}
template void AllocateCPU<float>(int nx, int ny, GradientsP<float>& Grad);
template void AllocateCPU<double>(int nx, int ny, GradientsP<double>& Grad);

template <class T> void AllocateCPU(int nblk, int blksize, EvolvingP<T> & Ev)
{
    AllocateCPU(nblk, blksize, Ev.h, Ev.zs, Ev.u, Ev.v);
}

template <class T> void AllocateCPU(int nblk, int blksize, EvolvingP_M<T>& Ev)
{
    AllocateCPU(nblk, blksize, Ev.h, Ev.zs, Ev.u, Ev.v, Ev.U, Ev.hU);

}

template <class T>
void AllocateCPU(int nblk, int blksize, Param XParam, Model<T>& XModel)
{
    // Allocate blocks data 
    AllocateCPU(nblk, blksize, XModel.evolv);
    AllocateCPU(nblk, blksize, XModel.evolv_o);

    AllocateCPU(nblk, blksize, XModel.grad.dhdy, XModel.grad.dzsdy, XModel.grad.dudy, XModel.grad.dvdy);
    AllocateCPU(nblk, blksize, XModel.grad.dhdx, XModel.grad.dzsdx, XModel.grad.dudx, XModel.grad.dvdx);

    AllocateCPU(nblk, blksize, XModel.grad.dzbdx);
    AllocateCPU(nblk, blksize, XModel.grad.dzbdy);
    if (XParam.engine==5)
    {
        AllocateCPU(nblk, blksize, XModel.fluxml.Fux, XModel.fluxml.Fvy, XModel.fluxml.Fuy, XModel.fluxml.Fvx);
        AllocateCPU(nblk, blksize, XModel.fluxml.hfu, XModel.fluxml.hfv, XModel.fluxml.hu, XModel.fluxml.hv);
        AllocateCPU(nblk, blksize, XModel.fluxml.hau);
        AllocateCPU(nblk, blksize, XModel.fluxml.hav);
    }
    else
    {
        AllocateCPU(nblk, blksize, XModel.flux.Fhu, XModel.flux.Fhv, XModel.flux.Fqux, XModel.flux.Fquy);

        AllocateCPU(nblk, blksize, XModel.flux.Fqvx, XModel.flux.Fqvy, XModel.flux.Su, XModel.flux.Sv);
    }
    AllocateCPU(nblk, blksize, XModel.zb, XModel.adv.dh, XModel.adv.dhu, XModel.adv.dhv);

    AllocateCPU(nblk, blksize, XModel.cf, XModel.time.arrmax, XModel.time.arrmin, XModel.time.dtmax);
    

    //Allocate block info
    AllocateCPU(nblk, 1, XModel.blocks.active);
    AllocateCPU(nblk, blksize, XModel.blocks.activeCell);
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
    
    // do allocate 1 outzone block, this will be reallocated eventually
    //AllocateCPU(1, 1, XModel.blocks.outZone[0].blk);
    //if (XParam.outzone.size() > 0)
    //{
    //  //XModel.blocks.outZone.resize(XParam.outzone.size())
    //  for (int o = 1; o < XParam.outzone.size(); o++)
    //  {
    //      AllocateCPU(1, 1, XModel.blocks.outZone[o].blk);
    //  }
    //}

    

    if (XParam.TSnodesout.size() > 0)
    {
        // Timeseries output temporary storage
        int storage = XParam.maxTSstorage;
        AllocateCPU(storage, 1, XModel.TSstore);
    }

    if (XParam.atmpforcing)
    {
        AllocateCPU(nblk, blksize, XModel.Patm);
        AllocateCPU(nblk, blksize, XModel.datmpdx);
        AllocateCPU(nblk, blksize, XModel.datmpdy);
    }

    if (XParam.infiltration)
    {
        AllocateCPU(nblk, blksize, XModel.il);
        AllocateCPU(nblk, blksize, XModel.cl);
        AllocateCPU(nblk, blksize, XModel.hgw);
    }

    if (XParam.outmax)
    {
        AllocateCPU(nblk, blksize, XModel.evmax);
    }
    if (XParam.outmean)
    {
        AllocateCPU(nblk, blksize, XModel.evmean);
    }
    if (XParam.outtwet)
    {
        AllocateCPU(nblk, blksize, XModel.wettime);
    }

    /*if (XParam.outvort)
    {
        AllocateCPU(nblk, blksize, XModel.vort);
    }
    if (XParam.outU)
    {
        AllocateCPU(nblk, blksize, XModel.U);
    }*/

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

        AllocateCPU(1, 1, XModel.bndblk.Riverinfo.Xbidir);
        AllocateCPU(1, 1, XModel.bndblk.Riverinfo.Xridib);
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
template void ReallocArray<int>(int nblk, int blksize, int*& zs, int*& h, int*& u, int*& v);
template void ReallocArray<float>(int nblk, int blksize, float*& zs, float*& h, float*& u, float*& v);
template void ReallocArray<double>(int nblk, int blksize, double*& zs, double*& h, double*& u, double*& v);

template <class T> void ReallocArray(int nblk, int blksize, T*& zs, T*& h, T*& u, T*& v, T*& U, T*& hU)
{
    //

    ReallocArray(nblk, blksize, zs);
    ReallocArray(nblk, blksize, h);
    ReallocArray(nblk, blksize, u);
    ReallocArray(nblk, blksize, v);
    ReallocArray(nblk, blksize, U);
    ReallocArray(nblk, blksize, hU);
    //return nblkmem
}

template void ReallocArray<int>(int nblk, int blksize, int* & zs, int*& h, int*& u, int*& v, int*& U, int*& hU);
template void ReallocArray<float>(int nblk, int blksize, float* & zs, float*& h, float*& u, float*& v, float*& U, float*& hU);
template void ReallocArray<double>(int nblk, int blksize, double* & zs, double*& h, double*& u, double*& v, double*& U, double*& hU);

template <class T> void ReallocArray(int nblk, int blksize, EvolvingP<T>& Ev)
{
    ReallocArray(nblk, blksize, Ev.zs, Ev.h, Ev.u, Ev.v);
}
template void ReallocArray<float>(int nblk, int blksize, EvolvingP<float>& Ev);
template void ReallocArray<double>(int nblk, int blksize, EvolvingP<double>& Ev);

template <class T> void ReallocArray(int nblk, int blksize, EvolvingP_M<T>& Ev)
{
    ReallocArray(nblk, blksize, Ev.zs, Ev.h, Ev.u, Ev.v, Ev.U, Ev.hU);
}
template void ReallocArray<float>(int nblk, int blksize, EvolvingP_M<float>& Ev);
template void ReallocArray<double>(int nblk, int blksize, EvolvingP_M<double>& Ev);

template <class T>
void ReallocArray(int nblk, int blksize, Param XParam, Model<T>& XModel)
{
    // Allocate blocks data 
    ReallocArray(nblk, blksize, XModel.evolv);
    ReallocArray(nblk, blksize, XModel.evolv_o);

    ReallocArray(nblk, blksize, XModel.grad.dhdy, XModel.grad.dzsdy, XModel.grad.dudy, XModel.grad.dvdy);
    ReallocArray(nblk, blksize, XModel.grad.dhdx, XModel.grad.dzsdx, XModel.grad.dudx, XModel.grad.dvdx);

    ReallocArray(nblk, blksize, XModel.grad.dzbdx);
    ReallocArray(nblk, blksize, XModel.grad.dzbdy);
    if (XParam.engine == 5)
    {
        ReallocArray(nblk, blksize, XModel.fluxml.Fux, XModel.fluxml.Fvy, XModel.fluxml.Fuy, XModel.fluxml.Fvx);
        ReallocArray(nblk, blksize, XModel.fluxml.hfu, XModel.fluxml.hfv, XModel.fluxml.hu, XModel.fluxml.hv);
        ReallocArray(nblk, blksize, XModel.fluxml.hau);
        ReallocArray(nblk, blksize, XModel.fluxml.hav);
    }
    else
    {
        ReallocArray(nblk, blksize, XModel.flux.Fhu, XModel.flux.Fhv, XModel.flux.Fqux, XModel.flux.Fquy);

        ReallocArray(nblk, blksize, XModel.flux.Fqvx, XModel.flux.Fqvy, XModel.flux.Su, XModel.flux.Sv);
    }

    ReallocArray(nblk, blksize, XModel.zb, XModel.adv.dh, XModel.adv.dhu, XModel.adv.dhv);

    ReallocArray(nblk, blksize, XModel.cf, XModel.time.arrmax, XModel.time.arrmin, XModel.time.dtmax);


    //Allocate block info
    ReallocArray(nblk, 1, XModel.blocks.active);
    ReallocArray(nblk, blksize, XModel.blocks.activeCell);
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
        ReallocArray(nblk, blksize, XModel.Patm);
        ReallocArray(nblk, blksize, XModel.datmpdx);
        ReallocArray(nblk, blksize, XModel.datmpdy);
    }

    if (XParam.infiltration)
    {
        ReallocArray(nblk, blksize, XModel.il);
        ReallocArray(nblk, blksize, XModel.cl);
        ReallocArray(nblk, blksize, XModel.hgw);

    }

    if (XParam.outmax)
    {
        ReallocArray(nblk, blksize, XModel.evmax);
    }
    if (XParam.outmean)
    {
        ReallocArray(nblk, blksize, XModel.evmean);
    }
    if (XParam.outtwet)
    {
        ReallocArray(nblk, blksize, XModel.wettime);
    }
    //ReallocArray(nx, ny, XModel.);



}

template void ReallocArray<float>(int nblk, int blksize, Param XParam, Model<float>& XModel);
template void ReallocArray<double>(int nblk, int blksize, Param XParam, Model<double>& XModel);




template <class T> void AllocateMappedMemCPU(int nx, int ny,int gpudevice, T*& z)
{

    bool bPinGenericMemory;
    cudaDeviceProp deviceProp;
#if defined(__APPLE__) || defined(MACOSX)
    bPinGenericMemory = false;  // Generic Pinning of System Paged memory is not currently supported on Mac OSX
#else
    bPinGenericMemory = true;
#endif

    // Here there should be a limit for cudar version less than 4.000


    if (bPinGenericMemory)
    {
        //printf("> Using Generic System Paged Memory (malloc)\n");
    }
    else
    {
        //printf("> Using CUDA Host Allocated (cudaHostAlloc)\n");
    }
    if (gpudevice >= 0)
    {
        cudaGetDeviceProperties(&deviceProp, gpudevice);

        if (!deviceProp.canMapHostMemory)
        {
            fprintf(stderr, "Device %d does not support mapping CPU host memory!\n", gpudevice);
            bPinGenericMemory = false;
        }
    }
    size_t bytes = nx * ny * sizeof(T);
    if (bPinGenericMemory)
    {

        

        T* a_UA = (T*)malloc(bytes + MEMORY_ALIGNMENT);
        

        // We need to ensure memory is aligned to 4K (so we will need to padd memory accordingly)
        z = (T*)ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
        
        if (gpudevice >= 0)
        {
            CUDA_CHECK(cudaHostRegister(z, bytes, cudaHostRegisterMapped));
        }

    }
    else
    {

        //flags = cudaHostAllocMapped;
        CUDA_CHECK(cudaHostAlloc((void**)&z, bytes, cudaHostAllocMapped));
        

    }


}
template void AllocateMappedMemCPU<int>(int nx, int ny, int gpudevice, int*& z);
template void AllocateMappedMemCPU<float>(int nx, int ny, int gpudevice, float*& z);
template void AllocateMappedMemCPU<double>(int nx, int ny, int gpudevice, double*& z);

template <class T> void AllocateMappedMemGPU(int nx, int ny, int gpudevice, T*& z_g, T* z)
{
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&z_g, (void*)z, 0));
}
template void AllocateMappedMemGPU<int>(int nx, int ny, int gpudevice, int*& z_g, int* z);
template void AllocateMappedMemGPU<float>(int nx, int ny, int gpudevice,float*& z_g, float* z);
template void AllocateMappedMemGPU<double>(int nx, int ny, int gpudevice, double*& z_g, double* z);


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

template <class T> void AllocateGPU(int nx, int ny, T*& zs, T*& h, T*& u, T*& v, T*& U, T*& hU)
{

    AllocateGPU(nx, ny, zs);
    AllocateGPU(nx, ny, h);
    AllocateGPU(nx, ny, u);
    AllocateGPU(nx, ny, v);
    AllocateGPU(nx, ny, U);
    AllocateGPU(nx, ny, hU);

}
template void AllocateGPU<double>(int nx, int ny, double*& zs, double*& h, double*& u, double*& v, double*& U, double*& hU);
template void AllocateGPU<float>(int nx, int ny, float*& zs, float*& h, float*& u, float*& v, float*& U, float*& hU);
template void AllocateGPU<int>(int nx, int ny, int*& zs, int*& h, int*& u, int*& v, int*& U, int*& hU);

template <class T> 
void AllocateGPU(int nx, int ny, GradientsP<T>& Grad)
{
    AllocateGPU(nx, ny, Grad.dhdx, Grad.dzsdx, Grad.dudx, Grad.dvdx);
    AllocateGPU(nx, ny, Grad.dhdy, Grad.dzsdy, Grad.dudy, Grad.dvdy);
    AllocateGPU(nx, ny, Grad.dzbdy);
    AllocateGPU(nx, ny, Grad.dzbdx);
}
template void AllocateGPU<float>(int nx, int ny, GradientsP<float>& Grad);
template void AllocateGPU<double>(int nx, int ny, GradientsP<double>& Grad);

template <class T> void AllocateGPU(int nblk, int blksize, EvolvingP<T>& Ev)
{
    AllocateGPU(nblk, blksize, Ev.h, Ev.zs, Ev.u, Ev.v);
}

template <class T> void AllocateGPU(int nblk, int blksize, EvolvingP_M<T>& Ev)
{
    AllocateGPU(nblk, blksize, Ev.h, Ev.zs, Ev.u, Ev.v, Ev.U, Ev.hU);
}

template <class T>
void AllocateGPU(int nblk, int blksize, Param XParam, Model<T>& XModel)
{
    // Allocate blocks data 
    AllocateGPU(nblk, blksize, XModel.evolv);
    AllocateGPU(nblk, blksize, XModel.evolv_o);

    AllocateGPU(nblk, blksize, XModel.grad);
    if (XParam.engine == 5)
    {
        AllocateGPU(nblk, blksize, XModel.fluxml.Fux, XModel.fluxml.Fvy, XModel.fluxml.hau, XModel.fluxml.hav);
        AllocateGPU(nblk, blksize, XModel.fluxml.hfu, XModel.fluxml.hfv, XModel.fluxml.hu, XModel.fluxml.hv);
        AllocateGPU(nblk, blksize, XModel.fluxml.Fuy);
        AllocateGPU(nblk, blksize, XModel.fluxml.Fvx);
    }
    else
    {
        AllocateGPU(nblk, blksize, XModel.flux.Fhu, XModel.flux.Fhv, XModel.flux.Fqux, XModel.flux.Fquy);

        AllocateGPU(nblk, blksize, XModel.flux.Fqvx, XModel.flux.Fqvy, XModel.flux.Su, XModel.flux.Sv);
    }
    AllocateGPU(nblk, blksize, XModel.zb, XModel.adv.dh, XModel.adv.dhu, XModel.adv.dhv);

    AllocateGPU(nblk, blksize, XModel.cf, XModel.time.arrmax, XModel.time.arrmin, XModel.time.dtmax);


    //Allocate block info
    AllocateGPU(nblk, 1, XModel.blocks.active);
    AllocateGPU(nblk, blksize, XModel.blocks.activeCell);
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
        AllocateGPU(nblk, blksize, XModel.Patm);
        AllocateGPU(nblk, blksize, XModel.datmpdx);
        AllocateGPU(nblk, blksize, XModel.datmpdy);
    }

    if (XParam.infiltration)
    {
        AllocateGPU(nblk, blksize, XModel.il);
        AllocateGPU(nblk, blksize, XModel.cl);
        AllocateGPU(nblk, blksize, XModel.hgw);
    }

    if (XParam.outmax)
    {
        AllocateGPU(nblk, blksize, XModel.evmax);
    }
    if (XParam.outmean)
    {
        AllocateGPU(nblk, blksize, XModel.evmean);
    }
    if (XParam.outtwet)
    {
        AllocateGPU(nblk, blksize, XModel.wettime);
    }

    
    /*if (XParam.outvort)
    {
        AllocateGPU(nblk, blksize, XModel.vort);
    }
    if (XParam.outU)
    {
        AllocateGPU(nblk, blksize, XModel.U);
    }*/

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

```



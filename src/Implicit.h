#ifndef IMPLICIT_H
#define IMPLICIT_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Spherical.h"
#include "Util_CPU.h"
#include "Multilayer.h"
#include "FlowGPU.h"
#include "Halo.h"



template <class T> inline T half_advection_dt(Param XParam,T dt);

// Not really used but kept alive for now
template <class T> __global__ void assemble_alpha_kernel(Param XParam, BlockP<T> XBlock,FluxMLP<T> XFlux,T dt);
template <class T> __global__ void assemble_rhs_kernel(Param XParam, BlockP<T> XBlock,FluxMLP<T> XFlux, T dt);
template <class T> __global__ void matvec_kernel(Param XParam, BlockP<T> XBlock,T* x, T* Ax, FluxMLP<T> XFlux);
template <class T> __global__ void jacobi_diag_kernel(Param XParam, BlockP<T> XBlock,FluxMLP<T> XFlux);
template <class T> __global__ void jacobi_apply_kernel(Param XParam, BlockP<T> XBlock, T*  r, T*  z, T* diagInv);

// Yes these are useful!
template <class T> __global__ void axpy_kernel(Param XParam, BlockP<T> XBlock, T* y, T* x, T a);
template <class T> __global__ void xpby_kernel(Param XParam, BlockP<T> XBlock, T* p, T* z, T beta);

template <class T> __global__ void x2_kernel(Param XParam, BlockP<T> XBlock, T* x, T* y);

template <class T> __global__ void acceleration_facex(Param XParam, BlockP<T> XBlock, FluxMLP<T> XFlux, FluxIMP<T> XImp, EvolvingP<T> XEv,T dt);
template <class T> __global__ void acceleration_facey(Param XParam, BlockP<T> XBlock,FluxMLP<T> XFlux, FluxIMP<T> XImp, EvolvingP<T> XEv,T dt);
template <class T> __global__ void acceleration_rhs(Param XParam, BlockP<T> XBlock, FluxIMP<T> XImp, T dt);
template <class T> __global__ void matvec_facefieldx(Param XParam, BlockP<T> XBlock,T* eta,T* g_x,T*alpha_x);
template <class T> __global__ void matvec_facefieldy(Param XParam, BlockP<T> XBlock,T* eta,T* g_y,T*alpha_y);
template <class T> __global__ void matvec_apply(Param XParam, BlockP<T> XBlock,T* eta,T* Aeta, T* g_x, T* g_y);

template <class T> __global__ void jacobi_diag(Param XParam, BlockP<T> XBlock, FluxIMP<T> XImp);
template <class T> __global__ void pressure_flux_reconstruction_facex(Param XParam, BlockP<T> XBlock,FluxMLP<T> XFlux, FluxIMP<T> XImp, EvolvingP<T> XEv,T dt);
template <class T> __global__ void pressure_flux_reconstruction_facey(Param XParam, BlockP<T> XBlock,FluxMLP<T> XFlux, FluxIMP<T> XImp, EvolvingP<T> XEv,T dt);

template <class T> __global__  void HaloFluxGPURMLclamp(Param XParam, BlockP<T> XBlock, T* z,T val);
template <class T> __global__  void HaloFluxGPUTMLclamp(Param XParam, BlockP<T> XBlock, T* z,T val);
#endif

#ifndef WAVECPU_H
#define WAVECPU_H

#include "General.h"
#include "Param.h"
#include "WaveParam.h"
#include "Arrays.h"
#include "WaveArrays.h"
#include "Forcing.h"
#include "Util_CPU.h"
#include "MemManagement.h"
#include "Halo.h"
#include "GridManip.h"
#include "Advection.h"
#include "Gradients.h"

// Index of a spectral bin value in a spectral array (see WaveArrays.h for the memory layout)
__host__ __device__ inline size_t specloc(int specstride, int ispec, int i)
{
	return size_t(ispec) * size_t(specstride) + size_t(i);
}

// Shared (CPU/GPU) wave physics helpers
template <class T> __host__ __device__ T WaveDispersion(T sigma, T h, T g);
template <class T> __host__ __device__ T WaveGroupVelocity(T sigma, T k, T h);
template <class T> __host__ __device__ T WaveDepth(const Param& XParam, const WaveParam& XWaveParam, int i, const EvolvingP<T>& XEv, T* zb);
template <class T> __host__ __device__ void CalcWaveBulkCell(const Param& XParam, const WaveParam& XWaveParam, int i, T h, const WaveModel<T>& XWave);
template <class T> __host__ __device__ T CalcWavedtCell(const Param& XParam, const WaveParam& XWaveParam, int i, int level, T h, const EvolvingP<T>& XEv, const WaveModel<T>& XWave);

// Wave time stepping
template <class T> double CalcWavetimestep(Param XParam, Loop<T> XLoop, WaveModel<T> XWave);
template <class T> double CalcWavedtmaxCPU(Param XParam, WaveParam XWaveParam, Model<T> XModel, WaveModel<T> XWave);
template <class T> Loop<T> WaveDummyLoop();

// Shared (CPU/GPU) propagation helpers
Param WaveLightParam(Param XParam, double theta);
template <class T> Model<T> WaveLightModel(Model<T> XModel);
double WaveBndTaper(Param XParam, WaveParam XWaveParam, double time);
template <class T> __host__ __device__ void WaveMediumCell(const Param& XParam, const WaveParam& XWaveParam, int i, T invtadv, const EvolvingP<T>& XEv, T* zb, const WaveModel<T>& XWave);
template <class T> __host__ __device__ void WaveKinCell(const Param& XParam, const WaveParam& XWaveParam, int i, const WaveModel<T>& XWave);
template <class T> __host__ __device__ T WaveUpwindFaceFlux(T cm, T cp, T Nm, T Np, T sm, T sp, bool wetm, bool wetp);
template <class T> __host__ __device__ void WaveFluxXCell(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int ispec, int woff, const BlockP<T>& XBlock, const WaveModel<T>& XWave);
template <class T> __host__ __device__ void WaveFluxYCell(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int ispec, int woff, const BlockP<T>& XBlock, const WaveModel<T>& XWave);
template <class T> __host__ __device__ void WaveGeoDivCell(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int ispec, int woff, const BlockP<T>& XBlock, const WaveModel<T>& XWave);
template <class T> __host__ __device__ T WaveSpecFaceFlux(T cf, T Na, T Nb, T Nam, T Nbp, bool order2, T theta, bool slopea, bool slopeb);
template <class T> __host__ __device__ void WaveSpecPropCell(const Param& XParam, const WaveParam& XWaveParam, int i, T dt, const WaveModel<T>& XWave);
template <class T> __host__ __device__ void WaveJonswapNorm(const WaveParam& XWaveParam, const WaveSpecGridP<T>& XSpec, T Tp, T Dir, T Dspr, T gamma, T& m0, T& Dsum);
template <class T> __host__ __device__ T WaveJonswapNbin(const WaveParam& XWaveParam, const WaveSpecGridP<T>& XSpec, int ispec, T Hs, T Tp, T Dir, T Dspr, T gamma, T m0, T Dsum);
template <class T> __host__ __device__ T WaveJonswapN(const WaveParam& XWaveParam, const WaveSpecGridP<T>& XSpec, int ispec, T Hs, T Tp, T Dir, T Dspr, T gamma);
template <class T> __host__ __device__ void WavebndFaceCell(const Param& XParam, const WaveParam& XWaveParam, int isright, int istop, int ib, int tx, int ispec, int woff, int type, T Hs, T Tp, T Dir, T Dspr, T gamma, T m0n, T Dsumn, T bndfac, const BlockP<T>& XBlock, const WaveModel<T>& XWave);
void WaveBndValues(const wavebndsegment& seg, double time, double gammadefault, double& Hs, double& Tp, double& Dir, double& Dspr, double& gamma);
template <class T> __host__ __device__ void WaveUpdateCell(const WaveParam& XWaveParam, int i, T dt, T alpha, const WaveModel<T>& XWave);

// Wave model CPU engine
template <class T> void WaveCPU(Param XParam, WaveParam XWaveParam, T dt, double time, bool dobulk, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave);
template <class T> void WavePrepCPU(Param XParam, WaveParam XWaveParam, double tadv, Model<T> XModel, WaveModel<T> XWave);
template <class T> void WaveRHSCPU(Param XParam, Param XParamN, WaveParam XWaveParam, T dt, double time, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave);
template <class T> void fillHaloWave(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveSpectrumP<T> XAction);
template <class T> void ResetWaveAdvCPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveAdvanceP<T> XAdv);
template <class T> void WaveGeoPropCPU(Param XParam, Param XParamN, WaveParam XWaveParam, const std::vector<wavebndsegment>& segs, double time, T bndfac, BlockP<T> XBlock, WaveModel<T> XWave);
template <class T> void WavebndFluxCPU(Param XParam, WaveParam XWaveParam, const wavebndsegment& seg, double time, T bndfac, int ispec0, int nbins, BlockP<T> XBlock, WaveModel<T> XWave);
template <class T> void WaveSpecPropCPU(Param XParam, WaveParam XWaveParam, T dt, BlockP<T> XBlock, WaveModel<T> XWave);
template <class T> void UpdateWaveActionCPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, T dt, T alpha, WaveModel<T> XWave);
template <class T> void CalcWaveBulkCPU(Param XParam, WaveParam XWaveParam, Model<T> XModel, WaveModel<T> XWave);

// End of global definition
#endif

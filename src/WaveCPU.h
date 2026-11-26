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

// Index of a spectral bin value in a spectral array (see WaveArrays.h for the memory layout)
__host__ __device__ inline size_t specloc(int specstride, int ispec, int i)
{
	return size_t(ispec) * size_t(specstride) + size_t(i);
}

// Shared (CPU/GPU) wave physics helpers
template <class T> __host__ __device__ T WaveDispersion(T sigma, T h, T g);
template <class T> __host__ __device__ T WaveGroupVelocity(T sigma, T k, T h);
template <class T> __host__ __device__ T WaveDepth(Param XParam, WaveParam XWaveParam, int i, EvolvingP<T> XEv, T* zb);
template <class T> __host__ __device__ void CalcWaveBulkCell(Param XParam, WaveParam XWaveParam, int i, T h, WaveModel<T> XWave);
template <class T> __host__ __device__ T CalcWavedtCell(Param XParam, WaveParam XWaveParam, int i, int level, T h, EvolvingP<T> XEv, WaveModel<T> XWave);

// Wave time stepping
template <class T> double CalcWavetimestep(Param XParam, Loop<T> XLoop, WaveModel<T> XWave);
template <class T> double CalcWavedtmaxCPU(Param XParam, WaveParam XWaveParam, Model<T> XModel, WaveModel<T> XWave);
template <class T> Loop<T> WaveDummyLoop();

// Wave model CPU engine
template <class T> void WaveCPU(Param XParam, WaveParam XWaveParam, T dt, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave);
template <class T> void fillHaloWave(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveSpectrumP<T> XAction);
template <class T> void ResetWaveAdvCPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveAdvanceP<T> XAdv);
template <class T> void UpdateWaveActionCPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, T dt, WaveAdvanceP<T> XAdv, WaveSpectrumP<T> XAction);
template <class T> void CalcWaveBulkCPU(Param XParam, WaveParam XWaveParam, Model<T> XModel, WaveModel<T> XWave);

// End of global definition
#endif

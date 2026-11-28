#ifndef WAVESOURCE_H
#define WAVESOURCE_H

#include "General.h"
#include "Param.h"
#include "WaveParam.h"
#include "Arrays.h"
#include "WaveArrays.h"
#include "Forcing.h"
#include "Updateforcing.h"
#include "WaveCPU.h"

/**
 * @brief Integral quantities of the spectrum of one cell needed by the source terms (computed from the current action density).
 */
template <class T>
struct WaveSrcCell
{
	bool wet;
	T h;
	T m0;     // variance (m2)
	T sigt;   // SWAN mean frequency sigma~ = m0 / m-1 (rad/s), whitecapping
	T kt;     // SWAN mean wave number k~ = (sum(k^-1/2 E)/m0)^-2 (rad/m), whitecapping
	T sigm;   // mean frequency m1/m0 (rad/s), breaking
	T Gw;     // whitecapping coefficient Cds (s~/s~PM)^p
	T Qb;     // fraction of breaking waves
	T Dbr;    // breaking dissipation rate (1/s), applied to all the bins (Battjes & Janssen)
	T Cfr;    // bottom friction coefficient (m2/s3)
	T ustar;  // wind friction velocity (m/s)
	T thw;    // wind direction (rad, cartesian, going to)
};

template <class T> __host__ __device__ bool WaveAnySource(const WaveParam& XWaveParam);
template <class T> __host__ __device__ void WaveSourceIntegrals(const Param& XParam, const WaveParam& XWaveParam, int i, const WaveModel<T>& XWave, WaveSrcCell<T>& sc);
template <class T> __host__ __device__ void WaveSourceBin(const Param& XParam, const WaveParam& XWaveParam, const WaveModel<T>& XWave, const WaveSrcCell<T>& sc, int i, int ifreq, int idir, T& Dsink, T& Bgrow, T& Alin);
template <class T> __host__ __device__ T WaveLimiterN(const WaveParam& XWaveParam, const WaveModel<T>& XWave, int i, int ifreq);
template <class T> __host__ __device__ void WaveSourceStepCell(const Param& XParam, const WaveParam& XWaveParam, int i, T dt, const WaveModel<T>& XWave);
template <class T> __host__ __device__ void WaveSourceDiagCell(const Param& XParam, const WaveParam& XWaveParam, int i, const WaveModel<T>& XWave);

template <class T> void WaveWindCPU(Param XParam, WaveParam XWaveParam, Forcing<float> XForcing, BlockP<T> XBlock, WaveModel<T> XWave);
template <class T> __global__ void WaveWindGPU(Param XParam, WaveParam XWaveParam, DynForcingP<float> Uwind, DynForcingP<float> Vwind, BlockP<T> XBlock, WaveModel<T> XWave);
template <class T> void WaveSourceStepCPU(Param XParam, WaveParam XWaveParam, T dt, BlockP<T> XBlock, WaveModel<T> XWave);
template <class T> __global__ void WaveSourceStepGPU(Param XParam, WaveParam XWaveParam, T dt, BlockP<T> XBlock, WaveModel<T> XWave);

// End of global definition
#endif

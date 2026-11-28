#ifndef WAVEIMPLICIT_H
#define WAVEIMPLICIT_H

#include "General.h"
#include "Param.h"
#include "WaveParam.h"
#include "Arrays.h"
#include "WaveArrays.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Setup_GPU.h"
#include "Halo.h"
#include "Updateforcing.h"
#include "Write_txtlog.h"
#include "WaveCPU.h"

// Setup
template <class T> void InitWaveImplicit(Param XParam, WaveParam& XWaveParam, Forcing<float>& XForcing, Model<T>& XModel, WaveModel<T>& XWave);
template <class T> void SetupWaveImplicitGPU(Param XParam, WaveParam XWaveParam, WaveModel<T>& XWave, WaveModel<T>& XWave_g);

// Shared (CPU/GPU) cell functions
template <class T> __host__ __device__ void WaveImplicitCell(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int sweep, T invdt, const BlockP<T>& XBlock, const WaveModel<T>& XWave);
template <class T> __host__ __device__ void WaveImplicitConvCell(const WaveParam& XWaveParam, int i, const WaveModel<T>& XWave);
template <class T> __host__ __device__ void WaveImplicitBndFace(const WaveParam& XWaveParam, int bw, int slot, int tx, int type, T Hs, T Tp, T Dir, T Dspr, T gamma, T m0n, T Dsumn, T bndfac, const WaveModel<T>& XWave);

// Solvers (return the number of iterations, convpct is the percentage of converged wet cells)
template <class T> int WaveImplicitCPU(Param XParam, WaveParam XWaveParam, double tadv, double time, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave, double& convpct);
template <class T> int WaveImplicitGPU(Param XParam, WaveParam XWaveParam, double tadv, double time, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave, double& convpct);

// GPU kernels
template <class T> __global__ void WaveImplicitLevelGPU(Param XParam, WaveParam XWaveParam, int sweep, T invdt, int* blks, BlockP<T> XBlock, WaveModel<T> XWave);
template <class T> __global__ void WaveImplicitConvGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveModel<T> XWave);
template <class T> __global__ void WaveImplicitBndGPU(Param XParam, WaveParam XWaveParam, int slot0, int type, bool interpmaps, T Hs, T Tp, T Dir, T Dspr, T gamma, T m0n, T Dsumn, TexSetP texHs, TexSetP texTp, TexSetP texDir, TexSetP texDspr, T bndfac, BlockP<T> XBlock, WaveModel<T> XWave);

// End of global definition
#endif

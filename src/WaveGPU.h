#ifndef WAVEGPU_H
#define WAVEGPU_H

#include "General.h"
#include "Param.h"
#include "WaveParam.h"
#include "Arrays.h"
#include "WaveArrays.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Halo.h"
#include "Advection.h"
#include "WaveCPU.h"

template <class T> void WaveGPU(Param XParam, WaveParam XWaveParam, T dt, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave);
template <class T> double CalcWavedtmaxGPU(Param XParam, WaveParam XWaveParam, Model<T> XModel, WaveModel<T> XWave);

template <class T> void fillHaloWaveGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveSpectrumP<T> XAction);

template <class T> __global__ void ResetWaveAdvGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveAdvanceP<T> XAdv);
template <class T> __global__ void UpdateWaveActionGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, T dt, WaveAdvanceP<T> XAdv, WaveSpectrumP<T> XAction);
template <class T> __global__ void CalcWaveBulkGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb, WaveModel<T> XWave);
template <class T> __global__ void CalcWavedtGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb, WaveModel<T> XWave);

// End of global definition
#endif

#ifndef WAVEINIT_H
#define WAVEINIT_H

#include "General.h"
#include "Param.h"
#include "WaveParam.h"
#include "Forcing.h"
#include "Arrays.h"
#include "WaveArrays.h"
#include "MemManagement.h"
#include "Setup_GPU.h"
#include "GridManip.h"
#include "Write_txtlog.h"
#include "WaveCPU.h"
#include "WaveGPU.h"

template <class T> void InitWaveModel(Param XParam, WaveParam& XWaveParam, Forcing<float>& XForcing, Model<T>& XModel, WaveModel<T>& XWave);
template <class T> void SetupWaveGPU(Param XParam, WaveParam XWaveParam, Model<T>& XModel_g, WaveModel<T>& XWave, WaveModel<T>& XWave_g);

template <class T> void InitWaveSpecGrid(WaveParam XWaveParam, WaveSpecGridP<T>& XSpec);
template <class T> void JonswapSpectrum(WaveParam XWaveParam, WaveSpecGridP<T> XSpec, double Hs, double Tp, double Dir, double Dspr, double gamma, std::vector<T>& Nspec);
template <class T> void InitWaveConditions(Param XParam, WaveParam XWaveParam, Model<T> XModel, WaveModel<T>& XWave);
template <class T> void InitWavemaparray(Model<T>& XModel, WaveModel<T> XWave);

// End of global definition
#endif


#ifndef TESTING_H
#define TESTING_H

#include "General.h"
#include "Param.h"
#include "Write_txtlog.h"
#include "ReadInput.h"
#include "ReadForcing.h"

#include "Util_CPU.h"
#include "Arrays.h"
#include "Forcing.h"
#include "Mesh.h"

#include "Setup_GPU.h"
#include "Mainloop.h"
#include "FlowCPU.h"
#include "FlowGPU.h"

#include "Adaptation.h"

#include "utctime.h"

template <class T> bool Testing(Param XParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g);

template <class T> bool GaussianHumptest(T zsnit, int gpu,bool compare);


template <class T> void TestingOutput(Param XParam, Model<T> XModel);
template <class T> void copyID2var(Param XParam, BlockP<T> XBlock, T* z);
template <class T> void copyBlockinfo2var(Param XParam, BlockP<T> XBlock, int* blkinfo, T* z);
template <class T> void CompareCPUvsGPU(Param XParam, Model<T> XModel, Model<T> XModel_g, std::vector<std::string> varlist, bool checkhalo);
//template <class T> void Gaussianhump(Param XParam, Model<T> XModel, Model<T> XModel_g);
template <class T> std::vector<float> Raintestmap(int gpu, int dimf, T zinit);
bool Raintestinput(int gpu);
template <class T> bool Rivertest(T zsnit, int gpu);
template <class T> bool MassConserveSteepSlope(T zsnit, int gpu);
template <class T> bool Raintest(T zsnit, int gpu, float alpha,int engine);
template <class T> bool testboundaries(Param XParam, T maxslope);
template <class T> bool ZoneOutputTest(int nzones, T zsinit);
template <class T> bool Rainlossestest(T zsnit, int gpu, float alpha);
template <class T> bool TestMultiBathyRough(int gpu, T ref, int secnario);
template <class T> bool TestFlexibleOutputTimes(int gpu, T ref, int scenario);

// End of global definition
#endif

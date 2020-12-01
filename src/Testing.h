
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



template <class T> void Testing(Param XParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g);

template <class T> bool GaussianHumptest(T zsnit, int gpu,bool compare);


template <class T> void TestingOutput(Param XParam, Model<T> XModel);
template <class T> void copyID2var(Param XParam, BlockP<T> XBlock, T* z);
template <class T> void copyBlockinfo2var(Param XParam, BlockP<T> XBlock, int* blkinfo, T* z);
template <class T> void CompareCPUvsGPU(Param XParam, Model<T> XModel, Model<T> XModel_g, std::vector<std::string> varlist, bool checkhalo);
//template <class T> void Gaussianhump(Param XParam, Model<T> XModel, Model<T> XModel_g);
// End of global definition
#endif

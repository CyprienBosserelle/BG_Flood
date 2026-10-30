

# File InitialConditions.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**InitialConditions.h**](InitialConditions_8h.md)

[Go to the documentation of this file](InitialConditions_8h.md)


```C++

#ifndef INITIALCONDITION_H
#define INITIALCONDITION_H

#include "General.h"
#include "Param.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "Write_txtlog.h"
#include "GridManip.h"
#include "InitEvolv.h"
#include "Gradients.h"
#include "Spherical.h"


template <class T> void InitialConditions(Param &XParam, Forcing<float> &XForcing, Model<T> &XModel);

template <class T> void InitRivers(Param XParam, Forcing<float> &XForcing, Model<T> &XModel);
template<class T> void Initmaparray(Model<T> &XModel);
template <class T> void initoutput(Param &XParam, Model<T>& XModel);
void InitTSOutput(Param XParam);
//template <class T> void Initbnds(Param XParam, Forcing<float> XForcing, Model<T>& XModel);

template <class T> void FindTSoutNodes(Param& XParam, BlockP<T> XBlock, BndblockP<T>& bnd);
template <class T> void Calcbndblks(Param& XParam, Forcing<float>& XForcing, BlockP<T> XBlock);
template <class T> void Findbndblks(Param XParam, Model<T> XModel, Forcing<float>& XForcing);
template <class T> void Initoutzone(Param& XParam, BlockP<T>& XBlock);

template <class T> void InitzbgradientCPU(Param XParam, Model<T> XModel);
template <class T> void InitzbgradientGPU(Param XParam, Model<T> XModel);

template <class T> void calcactiveCellCPU(Param XParam, BlockP<T> XBlock, Forcing<float>& XForcing, T* zb);

template <class T> void initOutputTimes(Param XParam, std::vector<double>& OutputT, BlockP<T>& XBlock);
// End of global definition;
#endif
```



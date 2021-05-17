
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


template <class T> void InitialConditions(Param &XParam, Forcing<float> &XForcing, Model<T> &XModel);

template <class T> void InitRivers(Param XParam, Forcing<float> &XForcing, Model<T> &XModel);
template<class T> void Initmaparray(Model<T> &XModel);
template <class T> void initoutput(Param &XParam, Model<T>& XModel);
void InitTSOutput(Param XParam);
//template <class T> void Initbnds(Param XParam, Forcing<float> XForcing, Model<T>& XModel);

template <class T> void FindTSoutNodes(Param& XParam, BlockP<T> XBlock, BndblockP& bnd);
template <class T> void Calcbndblks(Param& XParam, Forcing<float>& XForcing, BlockP<T> XBlock);
template <class T> void Findbndblks(Param XParam, Model<T> XModel, Forcing<float>& XForcing);

// End of global definition;
#endif

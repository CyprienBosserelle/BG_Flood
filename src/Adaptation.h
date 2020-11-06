
#ifndef ADAPTATION_H
#define ADAPTATION_H

#include "General.h"
#include "Param.h"
#include "Write_txtlog.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "Mesh.h"
#include "AdaptCriteria.h"
#include "Halo.h"
#include "InitialConditions.h"
#include "Testing.h"

template <class T> void Adaptation(Param& XParam, Forcing<float> XForcing, Model<T>& XModel);
template <class T> void InitialAdaptation(Param& XParam, Forcing<float> &XForcing, Model<T>& XModel);
template <class T> bool refinesanitycheck(Param XParam, BlockP<T> XBlock, bool*& refine, bool*& coarsen);
int checkneighbourrefine(int neighbourib, int levelib, int levelneighbour, bool*& refine, bool*& coarsen);

template <class T> bool checkBUQsanity(Param XParam, BlockP<T> XBlock);
bool checklevel(int ib, int levelib, int neighbourib, int levelneighbour);

template <class T> void Adapt(Param& XParam, Forcing<float> XForcing, Model<T>& XModel);

template <class T> int CalcAvailblk(Param& XParam, BlockP<T> XBlock, AdaptP& XAdapt);
template <class T> int AddBlocks(int nnewblk, Param& XParam, Model<T>& XModel);
template <class T> void coarsen(Param XParam, BlockP<T>& XBlock, AdaptP& XAdapt, EvolvingP<T> XEvo, EvolvingP<T>& XEv);
template <class T> void refine(Param XParam, BlockP<T>& XBlock, AdaptP& XAdapt, EvolvingP<T> XEvo, EvolvingP<T>& XEv);
template <class T> void Adaptationcleanup(Param& XParam, BlockP<T>& XBlock, AdaptP& XAdapt);

// End of global definition
#endif


#ifndef ADAPTATION_H
#define ADAPTATION_H

#include "General.h"
#include "Param.h"
#include "Write_txt.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "Mesh.h"
#include "AdaptCriteria.h"

template <class T> bool refinesanitycheck(Param XParam, BlockP<T> XBlock, bool*& refine, bool*& coarsen);

int checkneighbourrefine(int neighbourib, int levelib, int levelneighbour, bool*& refine, bool*& coarsen);

template <class T> int CalcAvailblk(Param& XParam, BlockP<T> XBlock, AdaptP& XAdapt);
template <class T> int AddBlocks(int nnewblk, Param& XParam, Model<T>& XModel);
template <class T> void coarsen(Param XParam, BlockP<T>& XBlock, AdaptP& XAdapt, EvolvingP<T> XEvo, EvolvingP<T>& XEv);


// End of global definition
#endif

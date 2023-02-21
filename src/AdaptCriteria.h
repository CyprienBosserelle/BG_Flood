
#ifndef ADAPTCRITERIA_H
#define ADAPTCRITERIA_H

#include "General.h"
#include "Param.h"
#include "Write_txtlog.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "Mesh.h"
#include "Halo.h"
#include "GridManip.h"

template <class T> int inrangecriteria(Param XParam, T zmin, T zmax, T* z, BlockP<T> XBlock, bool* refine, bool* coarsen);
template <class T> int Thresholdcriteria(Param XParam, T threshold, T* z, BlockP<T> XBlock, bool* refine, bool* coarsen);
template <class T> int AdaptCriteria(Param XParam, Forcing<float> XForcing, Model<T> XModel);




// End of global definition
#endif

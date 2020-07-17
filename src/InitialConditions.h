
#ifndef INITIALCONDITION_H
#define INITIALCONDITION_H

#include "General.h"
#include "Param.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Utils_GPU.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "Write_txt.h"
#include "GridManip.h"


template <class T> void InitialConditions(Param XParam, Forcing<float> XForcing, Model<T> XModel);


// End of global definition;
#endif

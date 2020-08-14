
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

template <class T> void TestingOutput(Param XParam, Model<T> XModel);
template <class T> void copyID2var(Param XParam, BlockP<T> XBlock, T* z);


// End of global definition
#endif

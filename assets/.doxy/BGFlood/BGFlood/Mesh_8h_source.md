

# File Mesh.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Mesh.h**](Mesh_8h.md)

[Go to the documentation of this file](Mesh_8h.md)


```C++

#ifndef MESH_H
#define MESH_H

#include "General.h"
#include "Param.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "Write_txtlog.h"
#include "GridManip.h"
#include "Poly.h"


int CalcInitnblk(Param XParam, Forcing<float> XForcing);


template <class T> void InitMesh(Param& XParam, Forcing<float> &XForcing, Model<T>& XModel);
template <class T> void InitBlockInfo(Param &XParam, Forcing<float> &XForcing, BlockP<T>& XBlock);
template <class T> void InitBlockadapt(Param &XParam, BlockP<T> XBlock, AdaptP& XAdap);
template <class T> void InitBlockxoyo(Param XParam, Forcing<float> XForcing, BlockP<T>& XBlock);
template <class T> void InitBlockneighbours(Param& XParam, Forcing<float>& XForcing, BlockP<T>& XBlock);

template <class T> int CalcMaskblk(Param XParam, BlockP<T> XBlock);
template <class T> void FindMaskblk(Param XParam, BlockP<T> &XBlock);

// End of global definition;
#endif
```



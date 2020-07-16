
#ifndef MESH_H
#define MESH_H

#include "General.h"
#include "Param.h"
#include "Forcing.h"
#include "MemManagement.h"
#include "Utils_GPU.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "Write_txt.h"
#include "GridManip.h"



int CalcInitnblk(Param XParam, Forcing<float> XForcing);


template <class T>void InitMesh(Param& XParam, Forcing<float> XForcing, Model<T>& XModel);
template <class T> void InitBlockinfo(Param XParam, Forcing<float> XForcing, BlockP<T>& XBlock);
void InitBlockadapt(Param XParam, AdaptP& XAdap);
template <class T> void InitBlockxoyo(Param XParam, Forcing<float> XForcing, BlockP<T>& XBlock);
template <class T> void InitBlockneighbours(Param XParam, BlockP<T>& XBlock);


// End of global definition;
#endif

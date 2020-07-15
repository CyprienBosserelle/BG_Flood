
#ifndef MESH_H
#define MESH_H

#include "General.h"
#include "Param.h"
#include "Forcing.h"
#include "Utils_GPU.h"
#include "Util_CPU.h"
#include "Arrays.h"
int CalcInitnblk(Param XParam, Forcing<float> XForcing);


void InitMesh(Param &XParam, Forcing<float> XForcing);
// End of global definition;
#endif

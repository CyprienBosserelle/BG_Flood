
#ifndef WRITENETCDF_H
#define WRITENETCDF_H

#include "General.h"
#include "Param.h"
#include "Write_txt.h"
#include "ReadInput.h"
#include "MemManagement.h"
#include "Util_CPU.h"

void handle_ncerror(int status);
template<class T> Param creatncfileBUQ(Param XParam, int* activeblk, int* level, T* blockxo, T* blockyo);
template <class T> void defncvarBUQ(Param XParam, int * activeblk, int * level, double * blockxo, double *blockyo, std::string varst, int vdim, T * var);
template <class T> void writencvarstepBUQ(Param XParam, int vdim, int * activeblk, int* level, double * blockxo, double *blockyo, std::string varst, T * var);


// End of global definition
#endif

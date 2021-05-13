
#ifndef WRITENETCDF_H
#define WRITENETCDF_H

#include "General.h"
#include "Param.h"
#include "Write_txtlog.h"
#include "ReadInput.h"
#include "MemManagement.h"
#include "Util_CPU.h"

void handle_ncerror(int status);
template<class T> void creatncfileBUQ(Param &XParam, int* activeblk, int* level, T* blockxo, T* blockyo);
template<class T> void creatncfileBUQ(Param &XParam, BlockP<T> XBlock);
template <class T> void defncvarBUQ(Param XParam, int * activeblk, int * level, T * blockxo, T *blockyo, std::string varst, int vdim, T * var);
template <class T> void writencvarstepBUQ(Param XParam, int vdim, int * activeblk, int* level, T * blockxo, T *blockyo, std::string varst, T * var);
template <class T> void InitSave2Netcdf(Param &XParam, Model<T> XModel);
extern "C" void writenctimestep(std::string outfile, double totaltime);
template <class T> void Save2Netcdf(Param XParam, Loop<T> XLoop, Model<T> XModel);
// End of global definition
#endif

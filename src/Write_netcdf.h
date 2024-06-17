
#ifndef WRITENETCDF_H
#define WRITENETCDF_H

#include "General.h"
#include "Param.h"
#include "Write_txtlog.h"
#include "ReadInput.h"
#include "MemManagement.h"
#include "Util_CPU.h"
#include "Arrays.h"

void handle_ncerror(int status);
template<class T> void creatncfileBUQ(Param &XParam, int* activeblk, int* level, T* blockxo, T* blockyo, outzoneB &Xzone);
template<class T> void creatncfileBUQ(Param &XParam, BlockP<T> &XBlock);
template <class T> void defncvarBUQ(Param XParam, int * activeblk, int * level, T * blockxo, T *blockyo, std::string varst, int vdim, T * var, outzoneB Xzone);
template <class T> void writencvarstepBUQ(Param XParam, int vdim, int * activeblk, int* level, T * blockxo, T *blockyo, std::string varst, T * var, outzoneB Xzone);
template <class T> void InitSave2Netcdf(Param &XParam, Model<T> &XModel);
extern "C" void writenctimestep(std::string outfile, double totaltime);
template <class T> void Save2Netcdf(Param XParam, Loop<T> XLoop, Model<T> XModel);

extern "C" void create2dnc(char* filename, int nx, int ny, double* xx, double* yy, double* var, char* varname);
extern "C" void create3dnc(char* name, int nx, int ny, int nt, double* xx, double* yy, double* theta, double* var, char* varname);
extern "C" void write3dvarnc(int nx, int ny, int nt, double totaltime, double* var);
extern "C" void write2dvarnc(int nx, int ny, double totaltime, double* var);

// End of global definition
#endif

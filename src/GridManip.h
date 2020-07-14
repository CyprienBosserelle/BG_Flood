
#ifndef GRIDMANIP_H
#define GRIDMANIP_H

#include "General.h"
#include "Param.h"
#include "Util_CPU.h"
#include "Utils_GPU.h"

template <class T> void InitArraySV(int nblk, int blksize, T initval, T*& Arr);

template <class T>  void setedges(int nblk, int* leftblk, int* rightblk, int* topblk, int* botblk, T*& zb);

template <class T> void CopyArray(int nblk, int blksize, T* source, T*& dest);

template <class T> void carttoBUQ(int nblk, int nx, int ny, double xo, double yo, double dx, double* blockxo, double* blockyo, T* zb, T*& zb_buq);


template <class T> void interp2BUQ(int nblk, double blksize, double blkdx, double* blockxo, double* blockyo, int nx, int ny, double xo, double xmax, double yo, double ymax, double dx, T* zb, T*& zb_buq);
template <class T> void interp2cf(Param XParam, float* cfin, T* blockxo, T* blockyo, T*& cf);

template <class T> void InterpstepCPU(int nx, int ny, int hdstep, T totaltime, T hddt, T *&Ux, T *Uo, T *Un);



// End of global definition
#endif

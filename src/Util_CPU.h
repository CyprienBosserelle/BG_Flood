
#ifndef UTILCPU_H
#define UTILCPU_H

#include "General.h"
#include "Param.h"

unsigned int nextPow2(unsigned int x);

template <class T> void AllocateCPU(int nx, int ny, T*& zb);
template <class T> void AllocateCPU(int nx, int ny, T*& zs, T*& h, T*& u, T*& v);

template <class T> void ReallocArray(int nblk, int blksize, T*& zb);

template <class T> void InitArraySV(int nblk, int blksize, T initval, T*& Arr);

template <class T>  void setedges(int nblk, int* leftblk, int* rightblk, int* topblk, int* botblk, T*& zb);

template <class T> void CopyArray(int nblk, int blksize, T* source, T*& dest);

template <class T> void carttoBUQ(int nblk, int nx, int ny, double xo, double yo, double dx, double* blockxo, double* blockyo, T* zb, T*& zb_buq);


template <class T> void interp2BUQ(int nblk, double blksize, double blkdx, double* blockxo, double* blockyo, int nx, int ny, double xo, double xmax, double yo, double ymax, double dx, T* zb, T*& zb_buq);
template <class T> void interp2cf(Param XParam, float* cfin, T* blockxo, T* blockyo, T*& cf);





// End of global definition
#endif

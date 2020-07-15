
#ifndef MEMMANAGEMENT_H
#define MEMMANAGEMENT_H

#include "General.h"
#include "Param.h"
#include "Arrays.h"



template <class T> void AllocateCPU(int nx, int ny, T*& zb);
template <class T> void AllocateCPU(int nx, int ny, T*& zs, T*& h, T*& u, T*& v);

template <class T> void AllocateCPU(int nx, int ny, GradientsP<T>& Grad);
template <class T> void AllocateCPU(int nblk, int blksize, EvolvingP<T> Ev);
template <class T> void AllocateCPU(int nblk, int blksize, Param XParam, Model<T>& XModel);


template <class T> void ReallocArray(int nblk, int blksize, T*& zb);


// End of global definition
#endif

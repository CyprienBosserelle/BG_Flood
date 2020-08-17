
#ifndef GRIDMANIP_H
#define GRIDMANIP_H

#include "General.h"
#include "Param.h"
#include "Util_CPU.h"
#include "Forcing.h"
#include "Arrays.h"
#include "MemManagement.h"

template <class T, class F> void CopyArrayBUQ(Param XParam, BlockP<F> XBlock, T* source, T*& dest);
template <class T> void CopyArrayBUQ(Param XParam, BlockP<T> XBlock, EvolvingP<T> source, EvolvingP<T>& dest);
template <class T, class F> void InitArrayBUQ(Param XParam, BlockP<F> XBlock, T initval, T*& Arr);
template <class T, class F> void InitBlkBUQ(Param XParam, BlockP<F> XBlock, T initval, T*& Arr);
template <class T>  void setedges(Param XParam, BlockP<T> XBlock, T*& zb);



template <class T, class F> void interp2BUQ(Param XParam, BlockP<T> XBlock, F forcing, T*& z);

template <class T> void InterpstepCPU(int nx, int ny, int hdstep, T totaltime, T hddt, T *&Ux, T *Uo, T *Un);



// End of global definition
#endif

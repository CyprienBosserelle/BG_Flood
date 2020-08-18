
#ifndef HALO_H
#define HALO_H

#include "General.h"
#include "Param.h"
#include "Write_txtlog.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "Mesh.h"
#include "MemManagement.h"

template <class T> void fillHalo(Param XParam, int ib, BlockP<T> XBlock, T*& z);
template <class T> void fillHalo(Param XParam, BlockP<T> XBlock, EvolvingP<T>& Xev);

template <class T> void fillHalo(Param XParam, BlockP<T> XBlock, GradientsP<T>& Grad);
template <class T> void fillHalo(Param XParam, BlockP<T> XBlock, FluxP<T>& Flux);

template <class T> void fillHaloTopRight(Param XParam, int ib, BlockP<T> XBlock, T*& z);

template <class T> void fillLeft(Param XParam, int ib, BlockP<T> XBlock, T*& z);
template <class T> void fillRight(Param XParam, int ib, BlockP<T> XBlock, T*& z);
template <class T> void fillBot(Param XParam, int ib, BlockP<T> XBlock, T*& z);
template <class T> void fillTop(Param XParam, int ib, BlockP<T> XBlock, T*& z);

template <class T> void fillCorners(Param XParam, int ib, BlockP<T> XBlock, T*& z);

template <class T> void fillCorners(Param XParam, BlockP<T> XBlock, T*& z);
template <class T> void fillCorners(Param XParam, BlockP<T> XBlock, EvolvingP<T>& Xev);


// GPU versions
template <class T> __global__ void fillLeft(int halowidth, int* active, int* level, int* leftbot, int* lefttop, int* rightbot, int* botright, int* topright, T* a);
template <class T> __global__ void fillRight(int halowidth, int* active, int* level, int* rightbot, int* righttop, int* leftbot, int* botleft, int* topleft, T* a);
// End of global definition
#endif

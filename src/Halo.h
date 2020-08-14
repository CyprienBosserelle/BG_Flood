
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


// End of global definition
#endif

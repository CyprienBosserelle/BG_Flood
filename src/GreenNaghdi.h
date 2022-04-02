
#ifndef GREENNAGHDI_H
#define GREENNAGHDI_H


#include "General.h"
#include "Param.h"
#include "Arrays.h"
#include "MemManagement.h"
#include "Util_CPU.h"


template <class T>
struct GNP
{
	T* Dx;
	T* Dy;
	T* bx;
	T* by;
	T* resx;
	T* resy;
};


// End of global definition
#endif

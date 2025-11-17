

# File Poly.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Poly.h**](_poly_8h.md)

[Go to the documentation of this file](_poly_8h.md)


```C++

#ifndef POLY_H
#define POLY_H

#include "General.h"
#include "Param.h"
#include "Input.h"
#include "Write_txtlog.h"
#include "Util_CPU.h"
#include "Forcing.h"
#include "Arrays.h"
#include "MemManagement.h"

template <class T> bool blockinpoly(T xo, T yo, T dx, int blkwidth, Polygon Poly);
template <class T> int wn_PnPoly(T Px, T Py, Polygon Poly);
Polygon CounterCWPoly(Polygon Poly);


// End of global definition
#endif
```



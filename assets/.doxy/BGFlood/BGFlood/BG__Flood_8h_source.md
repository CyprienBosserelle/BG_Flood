

# File BG\_Flood.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**BG\_Flood.h**](BG__Flood_8h.md)

[Go to the documentation of this file](BG__Flood_8h.md)


```C++

#ifndef BGFLOOD_H
#define BGFLOOD_H
// includes, system

#include "General.h"
#include "Param.h"
#include "Write_txtlog.h"
#include "ReadInput.h"
#include "ReadForcing.h"
#include "Setup_GPU.h"
#include "Util_CPU.h"
#include "Arrays.h"
#include "Forcing.h"
#include "Mesh.h"
#include "InitialConditions.h"
#include "Adaptation.h"
#include "Setup_GPU.h"
#include "Mainloop.h"

#include "Testing.h"


template < class T > int mainwork(Param XParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g);

// End of global definition
#endif
```



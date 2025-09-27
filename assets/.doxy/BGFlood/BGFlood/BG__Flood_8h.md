

# File BG\_Flood.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**BG\_Flood.h**](BG__Flood_8h.md)

[Go to the source code of this file](BG__Flood_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Write_txtlog.h"`
* `#include "ReadInput.h"`
* `#include "ReadForcing.h"`
* `#include "Setup_GPU.h"`
* `#include "Util_CPU.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "Mesh.h"`
* `#include "InitialConditions.h"`
* `#include "Adaptation.h"`
* `#include "Mainloop.h"`
* `#include "Testing.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**mainwork**](#function-mainwork) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |




























## Public Functions Documentation




### function mainwork 

```C++
template<class T>
int mainwork (
    Param XParam,
    Forcing < float > XForcing,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/BG_Flood.h`


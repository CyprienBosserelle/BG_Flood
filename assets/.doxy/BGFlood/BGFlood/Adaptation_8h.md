

# File Adaptation.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Adaptation.h**](Adaptation_8h.md)

[Go to the source code of this file](Adaptation_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Write_txtlog.h"`
* `#include "Util_CPU.h"`
* `#include "Arrays.h"`
* `#include "Mesh.h"`
* `#include "AdaptCriteria.h"`
* `#include "Halo.h"`
* `#include "InitialConditions.h"`
* `#include "Testing.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Adapt**](#function-adapt) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  void | [**Adaptation**](#function-adaptation) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  void | [**Adaptationcleanup**](#function-adaptationcleanup) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock, [**AdaptP**](structAdaptP.md) & XAdapt) <br> |
|  int | [**AddBlocks**](#function-addblocks) (int nnewblk, [**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  int | [**CalcAvailblk**](#function-calcavailblk) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**AdaptP**](structAdaptP.md) & XAdapt) <br> |
|  void | [**InitialAdaptation**](#function-initialadaptation) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  bool | [**checkBUQsanity**](#function-checkbuqsanity) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock) <br> |
|  bool | [**checklevel**](#function-checklevel) (int ib, int levelib, int neighbourib, int levelneighbour) <br> |
|  bool | [**checkneighbourdistance**](#function-checkneighbourdistance) (double dx, int ib, int levelib, T blocko, int neighbourib, int levelneighbour, T neighbourblocko, bool rightortop) <br> |
|  int | [**checkneighbourrefine**](#function-checkneighbourrefine) (int neighbourib, int levelib, int levelneighbour, bool \*& refine, bool \*& coarsen) <br> |
|  void | [**coarsen**](#function-coarsen) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock, [**AdaptP**](structAdaptP.md) & XAdapt, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvo, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv) <br> |
|  void | [**refine**](#function-refine) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock, [**AdaptP**](structAdaptP.md) & XAdapt, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvo, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv) <br> |
|  bool | [**refinesanitycheck**](#function-refinesanitycheck) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, bool \*& refine, bool \*& coarsen) <br> |




























## Public Functions Documentation




### function Adapt 

```C++
template<class T>
void Adapt (
    Param & XParam,
    Forcing < float > XForcing,
    Model < T > & XModel
) 
```




<hr>



### function Adaptation 

```C++
template<class T>
void Adaptation (
    Param & XParam,
    Forcing < float > XForcing,
    Model < T > & XModel
) 
```




<hr>



### function Adaptationcleanup 

```C++
template<class T>
void Adaptationcleanup (
    Param & XParam,
    BlockP < T > & XBlock,
    AdaptP & XAdapt
) 
```




<hr>



### function AddBlocks 

```C++
template<class T>
int AddBlocks (
    int nnewblk,
    Param & XParam,
    Model < T > & XModel
) 
```




<hr>



### function CalcAvailblk 

```C++
template<class T>
int CalcAvailblk (
    Param & XParam,
    BlockP < T > XBlock,
    AdaptP & XAdapt
) 
```




<hr>



### function InitialAdaptation 

```C++
template<class T>
void InitialAdaptation (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < T > & XModel
) 
```




<hr>



### function checkBUQsanity 

```C++
template<class T>
bool checkBUQsanity (
    Param XParam,
    BlockP < T > XBlock
) 
```




<hr>



### function checklevel 

```C++
bool checklevel (
    int ib,
    int levelib,
    int neighbourib,
    int levelneighbour
) 
```




<hr>



### function checkneighbourdistance 

```C++
template<class T>
bool checkneighbourdistance (
    double dx,
    int ib,
    int levelib,
    T blocko,
    int neighbourib,
    int levelneighbour,
    T neighbourblocko,
    bool rightortop
) 
```




<hr>



### function checkneighbourrefine 

```C++
int checkneighbourrefine (
    int neighbourib,
    int levelib,
    int levelneighbour,
    bool *& refine,
    bool *& coarsen
) 
```




<hr>



### function coarsen 

```C++
template<class T>
void coarsen (
    Param XParam,
    BlockP < T > & XBlock,
    AdaptP & XAdapt,
    EvolvingP < T > XEvo,
    EvolvingP < T > & XEv
) 
```




<hr>



### function refine 

```C++
template<class T>
void refine (
    Param XParam,
    BlockP < T > & XBlock,
    AdaptP & XAdapt,
    EvolvingP < T > XEvo,
    EvolvingP < T > & XEv
) 
```




<hr>



### function refinesanitycheck 

```C++
template<class T>
bool refinesanitycheck (
    Param XParam,
    BlockP < T > XBlock,
    bool *& refine,
    bool *& coarsen
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Adaptation.h`


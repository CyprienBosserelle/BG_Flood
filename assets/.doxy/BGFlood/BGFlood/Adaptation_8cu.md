

# File Adaptation.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Adaptation.cu**](Adaptation_8cu.md)

[Go to the source code of this file](Adaptation_8cu_source.md)



* `#include "Adaptation.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Adapt**](#function-adapt) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  void | [**Adaptation**](#function-adaptation) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  template void | [**Adaptation&lt; double &gt;**](#function-adaptation-double) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template void | [**Adaptation&lt; float &gt;**](#function-adaptation-float) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; float &gt; & XModel) <br> |
|  void | [**Adaptationcleanup**](#function-adaptationcleanup) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock, [**AdaptP**](structAdaptP.md) & XAdapt) <br> |
|  template void | [**Adaptationcleanup&lt; double &gt;**](#function-adaptationcleanup-double) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; double &gt; & XBlock, [**AdaptP**](structAdaptP.md) & XAdapt) <br> |
|  template void | [**Adaptationcleanup&lt; float &gt;**](#function-adaptationcleanup-float) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; float &gt; & XBlock, [**AdaptP**](structAdaptP.md) & XAdapt) <br> |
|  int | [**AddBlocks**](#function-addblocks) (int nnewblk, [**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  template int | [**AddBlocks&lt; double &gt;**](#function-addblocks-double) (int nnewblk, [**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template int | [**AddBlocks&lt; float &gt;**](#function-addblocks-float) (int nnewblk, [**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; float &gt; & XModel) <br> |
|  int | [**CalcAvailblk**](#function-calcavailblk) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**AdaptP**](structAdaptP.md) & XAdapt) <br> |
|  template int | [**CalcAvailblk&lt; double &gt;**](#function-calcavailblk-double) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**AdaptP**](structAdaptP.md) & XAdapt) <br> |
|  template int | [**CalcAvailblk&lt; float &gt;**](#function-calcavailblk-float) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**AdaptP**](structAdaptP.md) & XAdapt) <br> |
|  void | [**InitialAdaptation**](#function-initialadaptation) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  template void | [**InitialAdaptation&lt; double &gt;**](#function-initialadaptation-double) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template void | [**InitialAdaptation&lt; float &gt;**](#function-initialadaptation-float) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; float &gt; & XModel) <br> |
|  bool | [**checkBUQsanity**](#function-checkbuqsanity) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock) <br> |
|  template bool | [**checkBUQsanity&lt; double &gt;**](#function-checkbuqsanity-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock) <br> |
|  template bool | [**checkBUQsanity&lt; float &gt;**](#function-checkbuqsanity-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock) <br> |
|  bool | [**checklevel**](#function-checklevel) (int ib, int levelib, int neighbourib, int levelneighbour) <br> |
|  bool | [**checkneighbourdistance**](#function-checkneighbourdistance) (double dx, int ib, int levelib, T blocko, int neighbourib, int levelneighbour, T neighbourblocko, bool rightortop) <br> |
|  int | [**checkneighbourrefine**](#function-checkneighbourrefine) (int neighbourib, int levelib, int levelneighbour, bool \*& refine, bool \*& coarsen) <br> |
|  void | [**coarsen**](#function-coarsen) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock, [**AdaptP**](structAdaptP.md) & XAdapt, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvo, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv) <br> |
|  template void | [**coarsen&lt; double &gt;**](#function-coarsen-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; & XBlock, [**AdaptP**](structAdaptP.md) & XAdapt, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEvo, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; & XEv) <br> |
|  template void | [**coarsen&lt; float &gt;**](#function-coarsen-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; & XBlock, [**AdaptP**](structAdaptP.md) & XAdapt, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEvo, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; & XEv) <br> |
|  void | [**refine**](#function-refine) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock, [**AdaptP**](structAdaptP.md) & XAdapt, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; XEvo, [**EvolvingP**](structEvolvingP.md)&lt; T &gt; & XEv) <br> |
|  template void | [**refine&lt; double &gt;**](#function-refine-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; & XBlock, [**AdaptP**](structAdaptP.md) & XAdapt, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; XEvo, [**EvolvingP**](structEvolvingP.md)&lt; double &gt; & XEv) <br> |
|  template void | [**refine&lt; float &gt;**](#function-refine-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; & XBlock, [**AdaptP**](structAdaptP.md) & XAdapt, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; XEvo, [**EvolvingP**](structEvolvingP.md)&lt; float &gt; & XEv) <br> |
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



### function Adaptation&lt; double &gt; 

```C++
template void Adaptation< double > (
    Param & XParam,
    Forcing < float > XForcing,
    Model < double > & XModel
) 
```




<hr>



### function Adaptation&lt; float &gt; 

```C++
template void Adaptation< float > (
    Param & XParam,
    Forcing < float > XForcing,
    Model < float > & XModel
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



### function Adaptationcleanup&lt; double &gt; 

```C++
template void Adaptationcleanup< double > (
    Param & XParam,
    BlockP < double > & XBlock,
    AdaptP & XAdapt
) 
```




<hr>



### function Adaptationcleanup&lt; float &gt; 

```C++
template void Adaptationcleanup< float > (
    Param & XParam,
    BlockP < float > & XBlock,
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



### function AddBlocks&lt; double &gt; 

```C++
template int AddBlocks< double > (
    int nnewblk,
    Param & XParam,
    Model < double > & XModel
) 
```




<hr>



### function AddBlocks&lt; float &gt; 

```C++
template int AddBlocks< float > (
    int nnewblk,
    Param & XParam,
    Model < float > & XModel
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



### function CalcAvailblk&lt; double &gt; 

```C++
template int CalcAvailblk< double > (
    Param & XParam,
    BlockP < double > XBlock,
    AdaptP & XAdapt
) 
```




<hr>



### function CalcAvailblk&lt; float &gt; 

```C++
template int CalcAvailblk< float > (
    Param & XParam,
    BlockP < float > XBlock,
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



### function InitialAdaptation&lt; double &gt; 

```C++
template void InitialAdaptation< double > (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < double > & XModel
) 
```




<hr>



### function InitialAdaptation&lt; float &gt; 

```C++
template void InitialAdaptation< float > (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < float > & XModel
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



### function checkBUQsanity&lt; double &gt; 

```C++
template bool checkBUQsanity< double > (
    Param XParam,
    BlockP < double > XBlock
) 
```




<hr>



### function checkBUQsanity&lt; float &gt; 

```C++
template bool checkBUQsanity< float > (
    Param XParam,
    BlockP < float > XBlock
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



### function coarsen&lt; double &gt; 

```C++
template void coarsen< double > (
    Param XParam,
    BlockP < double > & XBlock,
    AdaptP & XAdapt,
    EvolvingP < double > XEvo,
    EvolvingP < double > & XEv
) 
```




<hr>



### function coarsen&lt; float &gt; 

```C++
template void coarsen< float > (
    Param XParam,
    BlockP < float > & XBlock,
    AdaptP & XAdapt,
    EvolvingP < float > XEvo,
    EvolvingP < float > & XEv
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



### function refine&lt; double &gt; 

```C++
template void refine< double > (
    Param XParam,
    BlockP < double > & XBlock,
    AdaptP & XAdapt,
    EvolvingP < double > XEvo,
    EvolvingP < double > & XEv
) 
```




<hr>



### function refine&lt; float &gt; 

```C++
template void refine< float > (
    Param XParam,
    BlockP < float > & XBlock,
    AdaptP & XAdapt,
    EvolvingP < float > XEvo,
    EvolvingP < float > & XEv
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
The documentation for this class was generated from the following file `src/Adaptation.cu`




# File Adaptation.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Adaptation.cu**](_adaptation_8cu.md)

[Go to the source code of this file](_adaptation_8cu_source.md)



* `#include "Adaptation.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Adapt**](#function-adapt) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Applies adaptation (refinement/coarsening) to the mesh and updates variables._  |
|  void | [**Adaptation**](#function-adaptation) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Performs mesh adaptation (refinement/coarsening) for the model._  |
|  template void | [**Adaptation&lt; double &gt;**](#function-adaptation-double) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; double &gt; & XModel) <br> |
|  template void | [**Adaptation&lt; float &gt;**](#function-adaptation-float) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; float &gt; & XModel) <br> |
|  void | [**Adaptationcleanup**](#function-adaptationcleanup) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt) <br>_Cleans up and updates block lists after adaptation._  |
|  template void | [**Adaptationcleanup&lt; double &gt;**](#function-adaptationcleanup-double) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; & XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt) <br> |
|  template void | [**Adaptationcleanup&lt; float &gt;**](#function-adaptationcleanup-float) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; & XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt) <br> |
|  int | [**AddBlocks**](#function-addblocks) (int nnewblk, [**Param**](class_param.md) & XParam, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Adds new blocks to the mesh for adaptation._  |
|  template int | [**AddBlocks&lt; double &gt;**](#function-addblocks-double) (int nnewblk, [**Param**](class_param.md) & XParam, [**Model**](struct_model.md)&lt; double &gt; & XModel) <br> |
|  template int | [**AddBlocks&lt; float &gt;**](#function-addblocks-float) (int nnewblk, [**Param**](class_param.md) & XParam, [**Model**](struct_model.md)&lt; float &gt; & XModel) <br> |
|  int | [**CalcAvailblk**](#function-calcavailblk) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt) <br>_Calculates the number of available blocks for refinement._  |
|  template int | [**CalcAvailblk&lt; double &gt;**](#function-calcavailblk-double) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt) <br> |
|  template int | [**CalcAvailblk&lt; float &gt;**](#function-calcavailblk-float) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt) <br> |
|  void | [**InitialAdaptation**](#function-initialadaptation) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Performs initial mesh adaptation and reruns initial conditions._  |
|  template void | [**InitialAdaptation&lt; double &gt;**](#function-initialadaptation-double) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; double &gt; & XModel) <br> |
|  template void | [**InitialAdaptation&lt; float &gt;**](#function-initialadaptation-float) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; float &gt; & XModel) <br> |
|  bool | [**checkBUQsanity**](#function-checkbuqsanity) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock) <br>_Checks the consistency and sanity of the block uniform quadtree mesh._  |
|  template bool | [**checkBUQsanity&lt; double &gt;**](#function-checkbuqsanity-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock) <br> |
|  template bool | [**checkBUQsanity&lt; float &gt;**](#function-checkbuqsanity-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock) <br> |
|  bool | [**checklevel**](#function-checklevel) (int ib, int levelib, int neighbourib, int levelneighbour) <br> |
|  bool | [**checkneighbourdistance**](#function-checkneighbourdistance) (double dx, int ib, int levelib, T blocko, int neighbourib, int levelneighbour, T neighbourblocko, bool rightortop) <br>_Checks if the distance between a block and its neighbor is consistent with their levels._  |
|  int | [**checkneighbourrefine**](#function-checkneighbourrefine) (int neighbourib, int levelib, int levelneighbour, bool \*& refine, bool \*& coarsen) <br> |
|  void | [**coarsen**](#function-coarsen) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEvo, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & XEv) <br>_Coarsens mesh blocks and updates conserved variables._  |
|  template void | [**coarsen&lt; double &gt;**](#function-coarsen-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; & XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEvo, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; & XEv) <br> |
|  template void | [**coarsen&lt; float &gt;**](#function-coarsen-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; & XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEvo, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; & XEv) <br> |
|  void | [**refine**](#function-refine) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEvo, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & XEv) <br>_Refines mesh blocks and interpolates conserved variables._  |
|  template void | [**refine&lt; double &gt;**](#function-refine-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; & XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; XEvo, [**EvolvingP**](struct_evolving_p.md)&lt; double &gt; & XEv) <br> |
|  template void | [**refine&lt; float &gt;**](#function-refine-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; & XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; XEvo, [**EvolvingP**](struct_evolving_p.md)&lt; float &gt; & XEv) <br> |
|  bool | [**refinesanitycheck**](#function-refinesanitycheck) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, bool \*& refine, bool \*& coarsen) <br> |




























## Public Functions Documentation




### function Adapt 

_Applies adaptation (refinement/coarsening) to the mesh and updates variables._ 
```C++
template<class T>
void Adapt (
    Param & XParam,
    Forcing < float > XForcing,
    Model < T > & XModel
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data 
* `XModel` [**Model**](struct_model.md) structure 




        

<hr>



### function Adaptation 

_Performs mesh adaptation (refinement/coarsening) for the model._ 
```C++
template<class T>
void Adaptation (
    Param & XParam,
    Forcing < float > XForcing,
    Model < T > & XModel
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data 
* `XModel` [**Model**](struct_model.md) structure

Iteratively refines or coarsens the mesh based on adaptation criteria, updating block info and variables. 


        

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

_Cleans up and updates block lists after adaptation._ 
```C++
template<class T>
void Adaptationcleanup (
    Param & XParam,
    BlockP < T > & XBlock,
    AdaptP & XAdapt
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `XAdapt` Adaptation data structure

Updates block levels, reorders active block list, and finalizes adaptation. 


        

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

_Adds new blocks to the mesh for adaptation._ 
```C++
template<class T>
int AddBlocks (
    int nnewblk,
    Param & XParam,
    Model < T > & XModel
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `nnewblk` Number of new blocks to add 
* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) structure 



**Returns:**

New total number of blocks in memory. 





        

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

_Calculates the number of available blocks for refinement._ 
```C++
template<class T>
int CalcAvailblk (
    Param & XParam,
    BlockP < T > XBlock,
    AdaptP & XAdapt
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `XAdapt` Adaptation data structure 



**Returns:**

Number of available blocks for refinement. 





        

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

_Performs initial mesh adaptation and reruns initial conditions._ 
```C++
template<class T>
void InitialAdaptation (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < T > & XModel
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data 
* `XModel` [**Model**](struct_model.md) structure 




        

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

_Checks the consistency and sanity of the block uniform quadtree mesh._ 
```C++
template<class T>
bool checkBUQsanity (
    Param XParam,
    BlockP < T > XBlock
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 



**Returns:**

True if mesh is sane, false otherwise. 





        

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

_Checks if the distance between a block and its neighbor is consistent with their levels._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `dx` Base grid spacing 
* `ib` Block index 
* `levelib` Block level 
* `blocko` Block coordinate 
* `neighbourib` Neighbor block index 
* `levelneighbour` Neighbor block level 
* `neighbourblocko` Neighbor block coordinate 
* `rightortop` True if neighbor is right/top, false if left/bottom 



**Returns:**

True if distance is consistent, false otherwise. 





        

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

_Coarsens mesh blocks and updates conserved variables._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `XAdapt` Adaptation data structure 
* `XEvo` Old evolving variables 
* `XEv` New evolving variables 




        

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

_Refines mesh blocks and interpolates conserved variables._ 
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





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XBlock` Block data structure 
* `XAdapt` Adaptation data structure 
* `XEvo` Old evolving variables 
* `XEv` New evolving variables 




        

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


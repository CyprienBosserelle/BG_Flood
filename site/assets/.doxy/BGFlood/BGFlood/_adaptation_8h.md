

# File Adaptation.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Adaptation.h**](_adaptation_8h.md)

[Go to the source code of this file](_adaptation_8h_source.md)



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
|  void | [**Adapt**](#function-adapt) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Applies adaptation (refinement/coarsening) to the mesh and updates variables._  |
|  void | [**Adaptation**](#function-adaptation) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Performs mesh adaptation (refinement/coarsening) for the model._  |
|  void | [**Adaptationcleanup**](#function-adaptationcleanup) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt) <br>_Cleans up and updates block lists after adaptation._  |
|  int | [**AddBlocks**](#function-addblocks) (int nnewblk, [**Param**](class_param.md) & XParam, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Adds new blocks to the mesh for adaptation._  |
|  int | [**CalcAvailblk**](#function-calcavailblk) ([**Param**](class_param.md) & XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt) <br>_Calculates the number of available blocks for refinement._  |
|  void | [**InitialAdaptation**](#function-initialadaptation) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, [**Model**](struct_model.md)&lt; T &gt; & XModel) <br>_Performs initial mesh adaptation and reruns initial conditions._  |
|  bool | [**checkBUQsanity**](#function-checkbuqsanity) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock) <br>_Checks the consistency and sanity of the block uniform quadtree mesh._  |
|  bool | [**checklevel**](#function-checklevel) (int ib, int levelib, int neighbourib, int levelneighbour) <br> |
|  bool | [**checkneighbourdistance**](#function-checkneighbourdistance) (double dx, int ib, int levelib, T blocko, int neighbourib, int levelneighbour, T neighbourblocko, bool rightortop) <br>_Checks if the distance between a block and its neighbor is consistent with their levels._  |
|  int | [**checkneighbourrefine**](#function-checkneighbourrefine) (int neighbourib, int levelib, int levelneighbour, bool \*& refine, bool \*& coarsen) <br> |
|  void | [**coarsen**](#function-coarsen) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEvo, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & XEv) <br>_Coarsens mesh blocks and updates conserved variables._  |
|  void | [**refine**](#function-refine) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; & XBlock, [**AdaptP**](struct_adapt_p.md) & XAdapt, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; XEvo, [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; & XEv) <br>_Refines mesh blocks and interpolates conserved variables._  |
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


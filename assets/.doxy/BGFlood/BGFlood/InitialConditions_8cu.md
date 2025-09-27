

# File InitialConditions.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**InitialConditions.cu**](InitialConditions_8cu.md)

[Go to the source code of this file](InitialConditions_8cu_source.md)



* `#include "InitialConditions.h"`
* `#include "Input.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**Calcbndblks**](#function-calcbndblks) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock) <br> |
|  void | [**FindTSoutNodes**](#function-findtsoutnodes) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**BndblockP**](structBndblockP.md)&lt; T &gt; & bnd) <br> |
|  template void | [**FindTSoutNodes&lt; double &gt;**](#function-findtsoutnodes-double) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, [**BndblockP**](structBndblockP.md)&lt; double &gt; & bnd) <br> |
|  template void | [**FindTSoutNodes&lt; float &gt;**](#function-findtsoutnodes-float) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, [**BndblockP**](structBndblockP.md)&lt; float &gt; & bnd) <br> |
|  void | [**Findbndblks**](#function-findbndblks) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing) <br> |
|  void | [**Findoutzoneblks**](#function-findoutzoneblks) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  template void | [**Findoutzoneblks&lt; double &gt;**](#function-findoutzoneblks-double) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; double &gt; & XBlock) <br> |
|  template void | [**Findoutzoneblks&lt; float &gt;**](#function-findoutzoneblks-float) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; float &gt; & XBlock) <br> |
|  std::vector&lt; double &gt; | [**GetTimeOutput**](#function-gettimeoutput) ([**T\_output**](classT__output.md) time\_info) <br> |
|  void | [**InitRivers**](#function-initrivers) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  template void | [**InitRivers&lt; double &gt;**](#function-initrivers-double) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template void | [**InitRivers&lt; float &gt;**](#function-initrivers-float) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; float &gt; & XModel) <br> |
|  void | [**InitTSOutput**](#function-inittsoutput) ([**Param**](classParam.md) XParam) <br> |
|  void | [**Initbndblks**](#function-initbndblks) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock) <br> |
|  void | [**InitialConditions**](#function-initialconditions) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  template void | [**InitialConditions&lt; double &gt;**](#function-initialconditions-double) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template void | [**InitialConditions&lt; float &gt;**](#function-initialconditions-float) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, [**Model**](structModel.md)&lt; float &gt; & XModel) <br> |
|  void | [**Initmaparray**](#function-initmaparray) ([**Model**](structModel.md)&lt; T &gt; & XModel) <br> |
|  template void | [**Initmaparray&lt; double &gt;**](#function-initmaparray-double) ([**Model**](structModel.md)&lt; double &gt; & XModel) <br> |
|  template void | [**Initmaparray&lt; float &gt;**](#function-initmaparray-float) ([**Model**](structModel.md)&lt; float &gt; & XModel) <br> |
|  void | [**Initoutzone**](#function-initoutzone) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  template void | [**Initoutzone&lt; double &gt;**](#function-initoutzone-double) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; double &gt; & XBlock) <br> |
|  template void | [**Initoutzone&lt; float &gt;**](#function-initoutzone-float) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; float &gt; & XBlock) <br> |
|  void | [**InitzbgradientCPU**](#function-initzbgradientcpu) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  template void | [**InitzbgradientCPU&lt; double &gt;**](#function-initzbgradientcpu-double) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template void | [**InitzbgradientCPU&lt; float &gt;**](#function-initzbgradientcpu-float) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |
|  void | [**InitzbgradientGPU**](#function-initzbgradientgpu) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  template void | [**InitzbgradientGPU&lt; double &gt;**](#function-initzbgradientgpu-double) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template void | [**InitzbgradientGPU&lt; float &gt;**](#function-initzbgradientgpu-float) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |
|  void | [**RectCornerBlk**](#function-rectcornerblk) ([**Param**](classParam.md) & XParam, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock, double xo, double yo, double xmax, double ymax, bool isEps, std::vector&lt; int &gt; & cornerblk) <br> |
|  void | [**calcactiveCellCPU**](#function-calcactivecellcpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, T \* zb) <br> |
|  \_\_global\_\_ void | [**calcactiveCellGPU**](#function-calcactivecellgpu) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* zb) <br> |
|  void | [**initOutputTimes**](#function-initoutputtimes) ([**Param**](classParam.md) XParam, std::vector&lt; double &gt; & OutputT, [**BlockP**](structBlockP.md)&lt; T &gt; & XBlock) <br> |
|  void | [**initinfiltration**](#function-initinfiltration) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* h, T \* initLoss, T \* hgw) <br> |
|  void | [**initoutput**](#function-initoutput) ([**Param**](classParam.md) & XParam, [**Model**](structModel.md)&lt; T &gt; & XModel) <br> |




























## Public Functions Documentation




### function Calcbndblks 

```C++
template<class T>
void Calcbndblks (
    Param & XParam,
    Forcing < float > & XForcing,
    BlockP < T > XBlock
) 
```




<hr>



### function FindTSoutNodes 

```C++
template<class T>
void FindTSoutNodes (
    Param & XParam,
    BlockP < T > XBlock,
    BndblockP < T > & bnd
) 
```




<hr>



### function FindTSoutNodes&lt; double &gt; 

```C++
template void FindTSoutNodes< double > (
    Param & XParam,
    BlockP < double > XBlock,
    BndblockP < double > & bnd
) 
```




<hr>



### function FindTSoutNodes&lt; float &gt; 

```C++
template void FindTSoutNodes< float > (
    Param & XParam,
    BlockP < float > XBlock,
    BndblockP < float > & bnd
) 
```




<hr>



### function Findbndblks 

```C++
template<class T>
void Findbndblks (
    Param XParam,
    Model < T > XModel,
    Forcing < float > & XForcing
) 
```



Find which block on the model edge belongs to a "side boundary" 


        

<hr>



### function Findoutzoneblks 

```C++
template<class T>
void Findoutzoneblks (
    Param & XParam,
    BlockP < T > & XBlock
) 
```




<hr>



### function Findoutzoneblks&lt; double &gt; 

```C++
template void Findoutzoneblks< double > (
    Param & XParam,
    BlockP < double > & XBlock
) 
```




<hr>



### function Findoutzoneblks&lt; float &gt; 

```C++
template void Findoutzoneblks< float > (
    Param & XParam,
    BlockP < float > & XBlock
) 
```




<hr>



### function GetTimeOutput 

```C++
std::vector< double > GetTimeOutput (
    T_output time_info
) 
```




<hr>



### function InitRivers 

```C++
template<class T>
void InitRivers (
    Param XParam,
    Forcing < float > & XForcing,
    Model < T > & XModel
) 
```




<hr>



### function InitRivers&lt; double &gt; 

```C++
template void InitRivers< double > (
    Param XParam,
    Forcing < float > & XForcing,
    Model < double > & XModel
) 
```




<hr>



### function InitRivers&lt; float &gt; 

```C++
template void InitRivers< float > (
    Param XParam,
    Forcing < float > & XForcing,
    Model < float > & XModel
) 
```




<hr>



### function InitTSOutput 

```C++
void InitTSOutput (
    Param XParam
) 
```




<hr>



### function Initbndblks 

```C++
template<class T>
void Initbndblks (
    Param & XParam,
    Forcing < float > & XForcing,
    BlockP < T > XBlock
) 
```




<hr>



### function InitialConditions 

```C++
template<class T>
void InitialConditions (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < T > & XModel
) 
```




<hr>



### function InitialConditions&lt; double &gt; 

```C++
template void InitialConditions< double > (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < double > & XModel
) 
```




<hr>



### function InitialConditions&lt; float &gt; 

```C++
template void InitialConditions< float > (
    Param & XParam,
    Forcing < float > & XForcing,
    Model < float > & XModel
) 
```




<hr>



### function Initmaparray 

```C++
template<class T>
void Initmaparray (
    Model < T > & XModel
) 
```




<hr>



### function Initmaparray&lt; double &gt; 

```C++
template void Initmaparray< double > (
    Model < double > & XModel
) 
```




<hr>



### function Initmaparray&lt; float &gt; 

```C++
template void Initmaparray< float > (
    Model < float > & XModel
) 
```




<hr>



### function Initoutzone 

```C++
template<class T>
void Initoutzone (
    Param & XParam,
    BlockP < T > & XBlock
) 
```




<hr>



### function Initoutzone&lt; double &gt; 

```C++
template void Initoutzone< double > (
    Param & XParam,
    BlockP < double > & XBlock
) 
```




<hr>



### function Initoutzone&lt; float &gt; 

```C++
template void Initoutzone< float > (
    Param & XParam,
    BlockP < float > & XBlock
) 
```




<hr>



### function InitzbgradientCPU 

```C++
template<class T>
void InitzbgradientCPU (
    Param XParam,
    Model < T > XModel
) 
```




<hr>



### function InitzbgradientCPU&lt; double &gt; 

```C++
template void InitzbgradientCPU< double > (
    Param XParam,
    Model < double > XModel
) 
```




<hr>



### function InitzbgradientCPU&lt; float &gt; 

```C++
template void InitzbgradientCPU< float > (
    Param XParam,
    Model < float > XModel
) 
```




<hr>



### function InitzbgradientGPU 

```C++
template<class T>
void InitzbgradientGPU (
    Param XParam,
    Model < T > XModel
) 
```




<hr>



### function InitzbgradientGPU&lt; double &gt; 

```C++
template void InitzbgradientGPU< double > (
    Param XParam,
    Model < double > XModel
) 
```




<hr>



### function InitzbgradientGPU&lt; float &gt; 

```C++
template void InitzbgradientGPU< float > (
    Param XParam,
    Model < float > XModel
) 
```




<hr>



### function RectCornerBlk 

```C++
template<class T>
void RectCornerBlk (
    Param & XParam,
    BlockP < T > & XBlock,
    double xo,
    double yo,
    double xmax,
    double ymax,
    bool isEps,
    std::vector< int > & cornerblk
) 
```



Find the block containing the border of a rectangular box (used for the defining the output zones) The indice of the blocks are returned through "cornerblk" from bottom left turning in the clockwise direction 


        

<hr>



### function calcactiveCellCPU 

```C++
template<class T>
void calcactiveCellCPU (
    Param XParam,
    BlockP < T > XBlock,
    Forcing < float > & XForcing,
    T * zb
) 
```




<hr>



### function calcactiveCellGPU 

```C++
template<class T>
__global__ void calcactiveCellGPU (
    Param XParam,
    BlockP < T > XBlock,
    T * zb
) 
```




<hr>



### function initOutputTimes 

```C++
template<class T>
void initOutputTimes (
    Param XParam,
    std::vector< double > & OutputT,
    BlockP < T > & XBlock
) 
```




<hr>



### function initinfiltration 

```C++
template<class T>
void initinfiltration (
    Param XParam,
    BlockP < T > XBlock,
    T * h,
    T * initLoss,
    T * hgw
) 
```




<hr>



### function initoutput 

```C++
template<class T>
void initoutput (
    Param & XParam,
    Model < T > & XModel
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/InitialConditions.cu`


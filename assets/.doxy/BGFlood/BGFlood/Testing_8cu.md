

# File Testing.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Testing.cu**](Testing_8cu.md)

[Go to the source code of this file](Testing_8cu_source.md)



* `#include "Testing.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  bool | [**CPUGPUtest**](#function-cpugputest) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  void | [**CompareCPUvsGPU**](#function-comparecpuvsgpu) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g, std::vector&lt; std::string &gt; varlist, bool checkhalo) <br> |
|  template void | [**CompareCPUvsGPU&lt; double &gt;**](#function-comparecpuvsgpu-double) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; double &gt; XModel, [**Model**](structModel.md)&lt; double &gt; XModel\_g, std::vector&lt; std::string &gt; varlist, bool checkhalo) <br> |
|  template void | [**CompareCPUvsGPU&lt; float &gt;**](#function-comparecpuvsgpu-float) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; float &gt; XModel, [**Model**](structModel.md)&lt; float &gt; XModel\_g, std::vector&lt; std::string &gt; varlist, bool checkhalo) <br> |
|  bool | [**GaussianHumptest**](#function-gaussianhumptest) (T zsnit, int gpu, bool compare) <br> |
|  template bool | [**GaussianHumptest&lt; double &gt;**](#function-gaussianhumptest-double) (double zsnit, int gpu, bool compare) <br> |
|  template bool | [**GaussianHumptest&lt; float &gt;**](#function-gaussianhumptest-float) (float zsnit, int gpu, bool compare) <br> |
|  bool | [**LakeAtRest**](#function-lakeatrest) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  [**Forcing**](structForcing.md)&lt; float &gt; | [**MakValleyBathy**](#function-makvalleybathy) ([**Param**](classParam.md) XParam, T slope, bool bottop, bool flip) <br> |
|  bool | [**MassConserveSteepSlope**](#function-massconservesteepslope) (T zsnit, int gpu) <br> |
|  template bool | [**MassConserveSteepSlope&lt; double &gt;**](#function-massconservesteepslope-double) (double zsnit, int gpu) <br> |
|  template bool | [**MassConserveSteepSlope&lt; float &gt;**](#function-massconservesteepslope-float) (float zsnit, int gpu) <br> |
|  bool | [**Rainlossestest**](#function-rainlossestest) (T zsinit, int gpu, float alpha) <br> |
|  bool | [**Raintest**](#function-raintest) (T zsnit, int gpu, float alpha, int engine) <br> |
|  bool | [**Raintestinput**](#function-raintestinput) (int gpu) <br> |
|  std::vector&lt; float &gt; | [**Raintestmap**](#function-raintestmap) (int gpu, int dimf, T zinit) <br> |
|  template std::vector&lt; float &gt; | [**Raintestmap&lt; double &gt;**](#function-raintestmap-double) (int gpu, int dimf, double Zsinit) <br> |
|  template std::vector&lt; float &gt; | [**Raintestmap&lt; float &gt;**](#function-raintestmap-float) (int gpu, int dimf, float Zsinit) <br> |
|  bool | [**RiverOnBoundary**](#function-riveronboundary) ([**Param**](classParam.md) XParam, T slope, int Dir, int Bound\_type) <br> |
|  bool | [**RiverVolumeAdapt**](#function-rivervolumeadapt) ([**Param**](classParam.md) XParam, T maxslope) <br> |
|  bool | [**RiverVolumeAdapt**](#function-rivervolumeadapt) ([**Param**](classParam.md) XParam, T slope, bool bottop, bool flip) <br>_Simulate a river flowing in a steep valley and heck the Volume conservation._  |
|  bool | [**Rivertest**](#function-rivertest) (T zsnit, int gpu) <br> |
|  template bool | [**Rivertest&lt; double &gt;**](#function-rivertest-double) (double zsnit, int gpu) <br> |
|  template bool | [**Rivertest&lt; float &gt;**](#function-rivertest-float) (float zsnit, int gpu) <br> |
|  int | [**TestAIObnd**](#function-testaiobnd) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g, bool bottop, bool flip, bool withaoi) <br> |
|  void | [**TestFirsthalfstep**](#function-testfirsthalfstep) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  bool | [**TestFlexibleOutputTimes**](#function-testflexibleoutputtimes) (int gpu, T ref, int scenario) <br> |
|  int | [**TestGradientSpeed**](#function-testgradientspeed) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  bool | [**TestHaloSpeed**](#function-testhalospeed) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  int | [**TestInstability**](#function-testinstability) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  bool | [**TestMultiBathyRough**](#function-testmultibathyrough) (int gpu, T ref, int scenario) <br> |
|  int | [**TestPinMem**](#function-testpinmem) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  template int | [**TestPinMem&lt; double &gt;**](#function-testpinmem-double) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; double &gt; XModel, [**Model**](structModel.md)&lt; double &gt; XModel\_g) <br> |
|  template int | [**TestPinMem&lt; float &gt;**](#function-testpinmem-float) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; float &gt; XModel, [**Model**](structModel.md)&lt; float &gt; XModel\_g) <br> |
|  bool | [**Testing**](#function-testing) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  template bool | [**Testing&lt; double &gt;**](#function-testing-double) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; double &gt; XModel, [**Model**](structModel.md)&lt; double &gt; XModel\_g) <br> |
|  template bool | [**Testing&lt; float &gt;**](#function-testing-float) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; float &gt; XModel, [**Model**](structModel.md)&lt; float &gt; XModel\_g) <br> |
|  void | [**TestingOutput**](#function-testingoutput) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  template void | [**TestingOutput&lt; double &gt;**](#function-testingoutput-double) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; double &gt; XModel) <br> |
|  template void | [**TestingOutput&lt; float &gt;**](#function-testingoutput-float) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; float &gt; XModel) <br> |
|  void | [**Testzbinit**](#function-testzbinit) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  T | [**ThackerBathy**](#function-thackerbathy) (T x, T y, T L, T D) <br>_create a parabolic bassin_  |
|  bool | [**ThackerLakeAtRest**](#function-thackerlakeatrest) ([**Param**](classParam.md) XParam, T zsinit) <br> |
|  template bool | [**ThackerLakeAtRest&lt; double &gt;**](#function-thackerlakeatrest-double) ([**Param**](classParam.md) XParam, double zsinit) <br> |
|  template bool | [**ThackerLakeAtRest&lt; float &gt;**](#function-thackerlakeatrest-float) ([**Param**](classParam.md) XParam, float zsinit) <br> |
|  T | [**ValleyBathy**](#function-valleybathy) (T x, T y, T slope, T center) <br>_create V shape Valley basin_  |
|  bool | [**ZoneOutputTest**](#function-zoneoutputtest) (int nzones, T zsinit) <br> |
|  template bool | [**ZoneOutputTest&lt; double &gt;**](#function-zoneoutputtest-double) (int nzones, double zsinit) <br> |
|  template bool | [**ZoneOutputTest&lt; float &gt;**](#function-zoneoutputtest-float) (int nzones, float zsinit) <br> |
|  void | [**alloc\_init2Darray**](#function-alloc_init2darray) (float \*\* arr, int NX, int NY) <br> |
|  void | [**copyBlockinfo2var**](#function-copyblockinfo2var) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, int \* blkinfo, T \* z) <br> |
|  template void | [**copyBlockinfo2var&lt; double &gt;**](#function-copyblockinfo2var-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, int \* blkinfo, double \* z) <br> |
|  template void | [**copyBlockinfo2var&lt; float &gt;**](#function-copyblockinfo2var-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, int \* blkinfo, float \* z) <br> |
|  void | [**copyID2var**](#function-copyid2var) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  template void | [**copyID2var&lt; double &gt;**](#function-copyid2var-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**copyID2var&lt; float &gt;**](#function-copyid2var-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**diffArray**](#function-diffarray) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, std::string varname, bool checkhalo, T \* cpu, T \* gpu, T \* dummy, T \* out) <br> |
|  void | [**diffSource**](#function-diffsource) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* Fqux, T \* Su, T \* output) <br> |
|  void | [**diffdh**](#function-diffdh) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* input, T \* output, T \* shuffle) <br> |
|  void | [**fillgauss**](#function-fillgauss) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T amp, T \* z) <br> |
|  template void | [**fillgauss&lt; double &gt;**](#function-fillgauss-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double amp, double \* z) <br> |
|  template void | [**fillgauss&lt; float &gt;**](#function-fillgauss-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float amp, float \* z) <br> |
|  void | [**fillrandom**](#function-fillrandom) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  template void | [**fillrandom&lt; double &gt;**](#function-fillrandom-double) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillrandom&lt; float &gt;**](#function-fillrandom-float) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**init3Darray**](#function-init3darray) (float \*\*\* arr, int rows, int cols, int depths) <br> |
|  bool | [**reductiontest**](#function-reductiontest) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  template bool | [**reductiontest&lt; double &gt;**](#function-reductiontest-double) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; double &gt; XModel, [**Model**](structModel.md)&lt; double &gt; XModel\_g) <br> |
|  template bool | [**reductiontest&lt; float &gt;**](#function-reductiontest-float) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; float &gt; XModel, [**Model**](structModel.md)&lt; float &gt; XModel\_g) <br> |
|  void | [**testButtingerX**](#function-testbuttingerx) ([**Param**](classParam.md) XParam, int ib, int ix, int iy, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  bool | [**testboundaries**](#function-testboundaries) ([**Param**](classParam.md) XParam, T maxslope) <br> |
|  void | [**testkurganovX**](#function-testkurganovx) ([**Param**](classParam.md) XParam, int ib, int ix, int iy, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  \_\_global\_\_ void | [**vectoroffsetGPU**](#function-vectoroffsetgpu) (int nx, T offset, T \* z) <br> |




























## Public Functions Documentation




### function CPUGPUtest 

```C++
template<class T>
bool CPUGPUtest (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>



### function CompareCPUvsGPU 

```C++
template<class T>
void CompareCPUvsGPU (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g,
    std::vector< std::string > varlist,
    bool checkhalo
) 
```



This function compares the Valiables in a CPU model and a GPU models This function is quite useful when checking both are identical enough one needs to provide a list (vector&lt;string&gt;) of variable to check 


        

<hr>



### function CompareCPUvsGPU&lt; double &gt; 

```C++
template void CompareCPUvsGPU< double > (
    Param XParam,
    Model < double > XModel,
    Model < double > XModel_g,
    std::vector< std::string > varlist,
    bool checkhalo
) 
```




<hr>



### function CompareCPUvsGPU&lt; float &gt; 

```C++
template void CompareCPUvsGPU< float > (
    Param XParam,
    Model < float > XModel,
    Model < float > XModel_g,
    std::vector< std::string > varlist,
    bool checkhalo
) 
```




<hr>



### function GaussianHumptest 

```C++
template<class T>
bool GaussianHumptest (
    T zsnit,
    int gpu,
    bool compare
) 
```



This function tests the full hydrodynamics model and compares the results with pre-conmputed (Hard wired) values The function creates it own model setup and mesh independantly to what the user might want to do The setup consist of a centrally located gaussian hump radiating away The test stops at an arbitrary time to compare with 8 values extracted from a identical run in basilisk This function also compares the result of the GPU and CPU code (until they diverge) 


        

<hr>



### function GaussianHumptest&lt; double &gt; 

```C++
template bool GaussianHumptest< double > (
    double zsnit,
    int gpu,
    bool compare
) 
```




<hr>



### function GaussianHumptest&lt; float &gt; 

```C++
template bool GaussianHumptest< float > (
    float zsnit,
    int gpu,
    bool compare
) 
```




<hr>



### function LakeAtRest 

```C++
template<class T>
bool LakeAtRest (
    Param XParam,
    Model < T > XModel
) 
```



This function simulates the first predictive step and check whether the lake at rest is preserved otherwise it prints out to screen the cells (and neighbour) where the test fails 


        

<hr>



### function MakValleyBathy 

```C++
template<class T>
Forcing < float > MakValleyBathy (
    Param XParam,
    T slope,
    bool bottop,
    bool flip
) 
```




<hr>



### function MassConserveSteepSlope 

```C++
template<class T>
bool MassConserveSteepSlope (
    T zsnit,
    int gpu
) 
```



This function tests the mass conservation of the vertical injection (used for rivers) The function creates it own model setup and mesh independantly to what the user might want to do This starts with a initial water level (zsnit=0 is dry) and runs for 0.1s before comparing results with zsnit=0.1 that is approx 20 steps 


        

<hr>



### function MassConserveSteepSlope&lt; double &gt; 

```C++
template bool MassConserveSteepSlope< double > (
    double zsnit,
    int gpu
) 
```




<hr>



### function MassConserveSteepSlope&lt; float &gt; 

```C++
template bool MassConserveSteepSlope< float > (
    float zsnit,
    int gpu
) 
```




<hr>



### function Rainlossestest 

```C++
template<class T>
bool Rainlossestest (
    T zsinit,
    int gpu,
    float alpha
) 
```



This function tests the Initial Losses and Continuous Losses implementation a plain domain, under constant rain. The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsinit=0.0 is dry) and runs for 1s comparing results every 0.1s (that is approx 20 steps) 


        

<hr>



### function Raintest 

```C++
template<class T>
bool Raintest (
    T zsnit,
    int gpu,
    float alpha,
    int engine
) 
```



This function tests the mass conservation of the spacial injection (used to model rain on grid) The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsnit=0.0 is dry) and runs for 0.1s before comparing results with zsnit=0.1 that is approx 20 steps 


        

<hr>



### function Raintestinput 

```C++
bool Raintestinput (
    int gpu
) 
```



This function tests the different inputs for rain forcing. This test is based on the paper Aureli2020, the 3 slopes test with regional rain. The experiment has been presented in Iwagaki1955. The first test compares a time varying rain input using a uniform time serie forcing and a time varying 2D field (with same value). The second test check the 3D rain forcing (comparing it to expected values). 


        

<hr>



### function Raintestmap 

```C++
template<class T>
std::vector< float > Raintestmap (
    int gpu,
    int dimf,
    T zinit
) 
```



\fnstdvector&lt;float&gt; Raintestmap(int gpu, int dimf, T zinit)


This function return the flux at the bottom of the 3 part slope for different types of rain forcings using the test case based on Iwagaki1955 


        

<hr>



### function Raintestmap&lt; double &gt; 

```C++
template std::vector< float > Raintestmap< double > (
    int gpu,
    int dimf,
    double Zsinit
) 
```




<hr>



### function Raintestmap&lt; float &gt; 

```C++
template std::vector< float > Raintestmap< float > (
    int gpu,
    int dimf,
    float Zsinit
) 
```




<hr>



### function RiverOnBoundary 

```C++
template<class T>
bool RiverOnBoundary (
    Param XParam,
    T slope,
    int Dir,
    int Bound_type
) 
```




<hr>



### function RiverVolumeAdapt 

```C++
template<class T>
bool RiverVolumeAdapt (
    Param XParam,
    T maxslope
) 
```




<hr>



### function RiverVolumeAdapt 

_Simulate a river flowing in a steep valley and heck the Volume conservation._ 
```C++
template<class T>
bool RiverVolumeAdapt (
    Param XParam,
    T slope,
    bool bottop,
    bool flip
) 
```



This function creates a dry steep valley topography to a given level and run the model for a while and checks that the Volume matches the theory.


The function can test the water volume for 4 scenario each time:
* left to right: bottop=false & flip=true;
* right to left: bottop=false & flip=false;
* bottom to top: bottop=true & flip=true;
* top to bottom: bottop=true & flip=false;




The function inherits the adaptation set in XParam so needs to be rerun to accnout for the different scenarios:
* uniform level
* flow from coasrse to fine
* flow from fine to coarse This is done in the higher level wrapping function 




        

<hr>



### function Rivertest 

```C++
template<class T>
bool Rivertest (
    T zsnit,
    int gpu
) 
```



This function tests the mass conservation of the vertical injection (used for rivers) The function creates it own model setup and mesh independantly to what the user might want to do This starts with a initial water level (zsnit=0 is dry) and runs for 0.1s before comparing results with zsnit=0.1 that is approx 20 steps 


        

<hr>



### function Rivertest&lt; double &gt; 

```C++
template bool Rivertest< double > (
    double zsnit,
    int gpu
) 
```




<hr>



### function Rivertest&lt; float &gt; 

```C++
template bool Rivertest< float > (
    float zsnit,
    int gpu
) 
```




<hr>



### function TestAIObnd 

```C++
template<class T>
int TestAIObnd (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g,
    bool bottop,
    bool flip,
    bool withaoi
) 
```




<hr>



### function TestFirsthalfstep 

```C++
template<class T>
void TestFirsthalfstep (
    Param XParam,
    Forcing < float > XForcing,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>



### function TestFlexibleOutputTimes 

```C++
template<class T>
bool TestFlexibleOutputTimes (
    int gpu,
    T ref,
    int scenario
) 
```




<hr>



### function TestGradientSpeed 

```C++
template<class T>
int TestGradientSpeed (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g
) 
```



This function fill an array with random values (0 - 1)


This function test the spped and accuracy of a new gradient function gradient are only calculated for zb but assigned to different gradient variable for storage 


        

<hr>



### function TestHaloSpeed 

```C++
template<class T>
bool TestHaloSpeed (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>



### function TestInstability 

```C++
template<class T>
int TestInstability (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>



### function TestMultiBathyRough 

```C++
template<class T>
bool TestMultiBathyRough (
    int gpu,
    T ref,
    int scenario
) 
```




<hr>



### function TestPinMem 

```C++
template<class T>
int TestPinMem (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>



### function TestPinMem&lt; double &gt; 

```C++
template int TestPinMem< double > (
    Param XParam,
    Model < double > XModel,
    Model < double > XModel_g
) 
```




<hr>



### function TestPinMem&lt; float &gt; 

```C++
template int TestPinMem< float > (
    Param XParam,
    Model < float > XModel,
    Model < float > XModel_g
) 
```




<hr>



### function Testing 

```C++
template<class T>
bool Testing (
    Param XParam,
    Forcing < float > XForcing,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>



### function Testing&lt; double &gt; 

```C++
template bool Testing< double > (
    Param XParam,
    Forcing < float > XForcing,
    Model < double > XModel,
    Model < double > XModel_g
) 
```




<hr>



### function Testing&lt; float &gt; 

```C++
template bool Testing< float > (
    Param XParam,
    Forcing < float > XForcing,
    Model < float > XModel,
    Model < float > XModel_g
) 
```




<hr>



### function TestingOutput 

```C++
template<class T>
void TestingOutput (
    Param XParam,
    Model < T > XModel
) 
```



OUTDATED? 


        

<hr>



### function TestingOutput&lt; double &gt; 

```C++
template void TestingOutput< double > (
    Param XParam,
    Model < double > XModel
) 
```




<hr>



### function TestingOutput&lt; float &gt; 

```C++
template void TestingOutput< float > (
    Param XParam,
    Model < float > XModel
) 
```




<hr>



### function Testzbinit 

```C++
template<class T>
void Testzbinit (
    Param XParam,
    Forcing < float > XForcing,
    Model < T > XModel,
    Model < T > XModel_g
) 
```




<hr>



### function ThackerBathy 

_create a parabolic bassin_ 
```C++
template<class T>
T ThackerBathy (
    T x,
    T y,
    T L,
    T D
) 
```



This function creates a parabolic bassin. The function returns a single value of the bassin


Borrowed from Buttinger et al. 2019.


#### Reference



Buttinger-Kreuzhuber, A., Horváth, Z., Noelle, S., Blöschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional structured grids over abrupt topography, Advances in water resources, 127, 89–108, 2019. 



        

<hr>



### function ThackerLakeAtRest 

```C++
template<class T>
bool ThackerLakeAtRest (
    Param XParam,
    T zsinit
) 
```




<hr>



### function ThackerLakeAtRest&lt; double &gt; 

```C++
template bool ThackerLakeAtRest< double > (
    Param XParam,
    double zsinit
) 
```




<hr>



### function ThackerLakeAtRest&lt; float &gt; 

```C++
template bool ThackerLakeAtRest< float > (
    Param XParam,
    float zsinit
) 
```




<hr>



### function ValleyBathy 

_create V shape Valley basin_ 
```C++
template<class T>
T ValleyBathy (
    T x,
    T y,
    T slope,
    T center
) 
```



This function creates a simple V shape Valley basin 


        

<hr>



### function ZoneOutputTest 

```C++
template<class T>
bool ZoneOutputTest (
    int nzones,
    T zsinit
) 
```



This function test the zoned output for a basic configuration


This function test the spped and accuracy of a new gradient function gradient are only calculated for zb but assigned to different gradient variable for storage 


        

<hr>



### function ZoneOutputTest&lt; double &gt; 

```C++
template bool ZoneOutputTest< double > (
    int nzones,
    double zsinit
) 
```




<hr>



### function ZoneOutputTest&lt; float &gt; 

```C++
template bool ZoneOutputTest< float > (
    int nzones,
    float zsinit
) 
```




<hr>



### function alloc\_init2Darray 

```C++
void alloc_init2Darray (
    float ** arr,
    int NX,
    int NY
) 
```



This function allocates and fills a 2D array with zero values 


        

<hr>



### function copyBlockinfo2var 

```C++
template<class T>
void copyBlockinfo2var (
    Param XParam,
    BlockP < T > XBlock,
    int * blkinfo,
    T * z
) 
```



This function copies blick info to an output variable This function is somewhat useful when checking bugs in the mesh refinement or coarsening one needs to provide a pointer(z) allocated on the CPU to store the clockinfo This fonction only works on CPU 


        

<hr>



### function copyBlockinfo2var&lt; double &gt; 

```C++
template void copyBlockinfo2var< double > (
    Param XParam,
    BlockP < double > XBlock,
    int * blkinfo,
    double * z
) 
```




<hr>



### function copyBlockinfo2var&lt; float &gt; 

```C++
template void copyBlockinfo2var< float > (
    Param XParam,
    BlockP < float > XBlock,
    int * blkinfo,
    float * z
) 
```




<hr>



### function copyID2var 

```C++
template<class T>
void copyID2var (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```



This function copies block info to an output variable This function is somewhat useful when checking bugs in the mesh refinement or coarsening one needs to provide a pointer(z) allocated on the CPU to store the clockinfo This fonction only works on CPU 


        

<hr>



### function copyID2var&lt; double &gt; 

```C++
template void copyID2var< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z
) 
```




<hr>



### function copyID2var&lt; float &gt; 

```C++
template void copyID2var< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z
) 
```




<hr>



### function diffArray 

```C++
template<class T>
void diffArray (
    Param XParam,
    BlockP < T > XBlock,
    std::string varname,
    bool checkhalo,
    T * cpu,
    T * gpu,
    T * dummy,
    T * out
) 
```




<hr>



### function diffSource 

```C++
template<class T>
void diffSource (
    Param XParam,
    BlockP < T > XBlock,
    T * Fqux,
    T * Su,
    T * output
) 
```



This function Calculate The source term of the equation. This function is quite useful when checking for Lake-at-Rest states This function requires an outputCPU pointers to save the result of teh calculation 


        

<hr>



### function diffdh 

```C++
template<class T>
void diffdh (
    Param XParam,
    BlockP < T > XBlock,
    T * input,
    T * output,
    T * shuffle
) 
```



This function Calculates The difference in left and right flux terms. This function is quite useful when checking for Lake-at-Rest states This function requires a preallocated output and a shuffle (right side term) CPU pointers to save the result of teh calculation 


        

<hr>



### function fillgauss 

```C++
template<class T>
void fillgauss (
    Param XParam,
    BlockP < T > XBlock,
    T amp,
    T * z
) 
```



This function fill an array with a gaussian bump


borrowed/adapted from Basilisk test (?) 


        

<hr>



### function fillgauss&lt; double &gt; 

```C++
template void fillgauss< double > (
    Param XParam,
    BlockP < double > XBlock,
    double amp,
    double * z
) 
```




<hr>



### function fillgauss&lt; float &gt; 

```C++
template void fillgauss< float > (
    Param XParam,
    BlockP < float > XBlock,
    float amp,
    float * z
) 
```




<hr>



### function fillrandom 

```C++
template<class T>
void fillrandom (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```



This function fill an array with random values (0 - 1) 


        

<hr>



### function fillrandom&lt; double &gt; 

```C++
template void fillrandom< double > (
    Param XParam,
    BlockP < double > XBlock,
    double * z
) 
```




<hr>



### function fillrandom&lt; float &gt; 

```C++
template void fillrandom< float > (
    Param XParam,
    BlockP < float > XBlock,
    float * z
) 
```




<hr>



### function init3Darray 

```C++
void init3Darray (
    float *** arr,
    int rows,
    int cols,
    int depths
) 
```



This function fill a 3D array with zero values 


        

<hr>



### function reductiontest 

```C++
template<class T>
bool reductiontest (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g
) 
```



Test the algorithm for reducing the global time step on the user grid layout 


        

<hr>



### function reductiontest&lt; double &gt; 

```C++
template bool reductiontest< double > (
    Param XParam,
    Model < double > XModel,
    Model < double > XModel_g
) 
```




<hr>



### function reductiontest&lt; float &gt; 

```C++
template bool reductiontest< float > (
    Param XParam,
    Model < float > XModel,
    Model < float > XModel_g
) 
```




<hr>



### function testButtingerX 

```C++
template<class T>
void testButtingerX (
    Param XParam,
    int ib,
    int ix,
    int iy,
    Model < T > XModel
) 
```



This function goes through the Buttinger scheme but instead of the normal output just prints all teh usefull values This function is/was used in the lake-at-rest verification


See also: void testkurganovX(Param XParam, int ib, int ix, int iy, Model&lt;T&gt; XModel) 


        

<hr>



### function testboundaries 

```C++
template<class T>
bool testboundaries (
    Param XParam,
    T maxslope
) 
```




<hr>



### function testkurganovX 

```C++
template<class T>
void testkurganovX (
    Param XParam,
    int ib,
    int ix,
    int iy,
    Model < T > XModel
) 
```



This function goes through the Kurganov scheme but instead of the normal output just prints all teh usefull values This function is/was used in the lake-at-rest verification 


        

<hr>



### function vectoroffsetGPU 

```C++
template<class T>
__global__ void vectoroffsetGPU (
    int nx,
    T offset,
    T * z
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Testing.cu`


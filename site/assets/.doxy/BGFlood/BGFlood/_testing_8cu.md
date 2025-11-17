

# File Testing.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Testing.cu**](_testing_8cu.md)

[Go to the source code of this file](_testing_8cu_source.md)



* `#include "Testing.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  bool | [**CPUGPUtest**](#function-cpugputest) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br> |
|  void | [**CompareCPUvsGPU**](#function-comparecpuvsgpu) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g, std::vector&lt; std::string &gt; varlist, bool checkhalo) <br>_Compares the Variables in a CPU model and a GPU models This function is quite useful when checking both are identical enough one needs to provide a list (vector&lt;string&gt;) of variable to check._  |
|  template void | [**CompareCPUvsGPU&lt; double &gt;**](#function-comparecpuvsgpu-double) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; double &gt; XModel, [**Model**](struct_model.md)&lt; double &gt; XModel\_g, std::vector&lt; std::string &gt; varlist, bool checkhalo) <br> |
|  template void | [**CompareCPUvsGPU&lt; float &gt;**](#function-comparecpuvsgpu-float) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; float &gt; XModel, [**Model**](struct_model.md)&lt; float &gt; XModel\_g, std::vector&lt; std::string &gt; varlist, bool checkhalo) <br> |
|  bool | [**GaussianHumptest**](#function-gaussianhumptest) (T zsnit, int gpu, bool compare) <br>_Gaussian hump propagation test._  |
|  template bool | [**GaussianHumptest&lt; double &gt;**](#function-gaussianhumptest-double) (double zsnit, int gpu, bool compare) <br> |
|  template bool | [**GaussianHumptest&lt; float &gt;**](#function-gaussianhumptest-float) (float zsnit, int gpu, bool compare) <br> |
|  bool | [**LakeAtRest**](#function-lakeatrest) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Test the lake at rest condition This function simulates the first predictive step and check whether the lake at rest is preserved otherwise it prints out to screen the cells (and neighbour) where the test fails._  |
|  [**Forcing**](struct_forcing.md)&lt; float &gt; | [**MakValleyBathy**](#function-makvalleybathy) ([**Param**](class_param.md) XParam, T slope, bool bottop, bool flip) <br>_Creates a valley bathymetry This function creates a valley bathymetry with a given slope and center It also adds a wall around the domain to avoid boundary effects._  |
|  bool | [**MassConserveSteepSlope**](#function-massconservesteepslope) (T zsnit, int gpu) <br>[_**River**_](class_river.md) _inflow mass conservation test on steep slope._ |
|  template bool | [**MassConserveSteepSlope&lt; double &gt;**](#function-massconservesteepslope-double) (double zsnit, int gpu) <br> |
|  template bool | [**MassConserveSteepSlope&lt; float &gt;**](#function-massconservesteepslope-float) (float zsnit, int gpu) <br> |
|  bool | [**Rainlossestest**](#function-rainlossestest) (T zsinit, int gpu, float alpha) <br>_Test the Initial and Continuous losses implementation This function tests the Initial Losses and Continuous Losses implementation a plain domain, under constant rain. The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsinit=0.0 is dry) and runs for 1s comparing results every 0.1s (that is approx 20 steps)_  |
|  bool | [**Raintest**](#function-raintest) (T zsnit, int gpu, float alpha, int engine) <br>_Test the rain input and mass conservation This function tests the mass conservation of the spacial injection (used to model rain on grid) The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsnit=0.0 is dry) and runs for 0.1s before comparing results with zsnit=0.1 that is approx 20 steps._  |
|  bool | [**Raintestinput**](#function-raintestinput) (int gpu) <br>_Test the rain input options This function tests the different inputs for rain forcing. This test is based on the paper Aureli2020, the 3 slopes test with regional rain. The experiment has been presented in Iwagaki1955. The first test compares a time varying rain input using a uniform time serie forcing and a time varying 2D field (with same value). The second test check the 3D rain forcing (comparing it to expected values)._  |
|  std::vector&lt; float &gt; | [**Raintestmap**](#function-raintestmap) (int gpu, int dimf, T zinit) <br>_Test the rain input options and return the flux at the bottom of the slope This function return the flux at the bottom of the 3 part slope for different types of rain forcings using the test case based on Iwagaki1955._  |
|  template std::vector&lt; float &gt; | [**Raintestmap&lt; double &gt;**](#function-raintestmap-double) (int gpu, int dimf, double Zsinit) <br> |
|  template std::vector&lt; float &gt; | [**Raintestmap&lt; float &gt;**](#function-raintestmap-float) (int gpu, int dimf, float Zsinit) <br> |
|  bool | [**RiverOnBoundary**](#function-riveronboundary) ([**Param**](class_param.md) XParam, T slope, int Dir, int Bound\_type) <br> |
|  bool | [**RiverVolumeAdapt**](#function-rivervolumeadapt) ([**Param**](class_param.md) XParam, T maxslope) <br> |
|  bool | [**RiverVolumeAdapt**](#function-rivervolumeadapt) ([**Param**](class_param.md) XParam, T slope, bool bottop, bool flip) <br>_Simulate a river flowing in a steep valley and heck the Volume conservation._  |
|  bool | [**Rivertest**](#function-rivertest) (T zsnit, int gpu) <br>[_**River**_](class_river.md) _inflow mass conservation test._ |
|  template bool | [**Rivertest&lt; double &gt;**](#function-rivertest-double) (double zsnit, int gpu) <br> |
|  template bool | [**Rivertest&lt; float &gt;**](#function-rivertest-float) (float zsnit, int gpu) <br> |
|  int | [**TestAIObnd**](#function-testaiobnd) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g, bool bottop, bool flip, bool withaoi) <br>_Test the aoibnd option of the model This function tests the aoibnd option of the model on a valley domain with a small amount of water The function creates its own model setup and mesh independantly to what the user inputs._  _This starts with a initial water level (zsinit=0.0 is dry) and runs for 20s._ |
|  void | [**TestFirsthalfstep**](#function-testfirsthalfstep) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Test the first half step of the model_  _This function tests the first half step of the model on a sloping domain with a small amount of water The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsinit=0.0 is dry) and runs for 1s comparing results every 0.1s._ |
|  bool | [**TestFlexibleOutputTimes**](#function-testflexibleoutputtimes) (int gpu, T ref, int scenario) <br>_Test the reading of flexible output times._  |
|  int | [**TestGradientSpeed**](#function-testgradientspeed) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Test the speed of different gradient functions This function fill an array with random values (0 - 1)_  |
|  bool | [**TestHaloSpeed**](#function-testhalospeed) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Test the speed of different halo filling functions This function test the speed and accuracy of a new gradient function gradient are only calculated for zb but assigned to different gradient variable for storage._  |
|  int | [**TestInstability**](#function-testinstability) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Test the stability of the model This function tests the stability of the model on a sloping domain with a small amount of water The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsinit=0.0 is dry) and runs for 1s comparing results every 0.1s._  |
|  bool | [**TestMultiBathyRough**](#function-testmultibathyrough) (int gpu, T ref, int scenario) <br>_Test the reading of multiple bathymetry and roughness files._  |
|  int | [**TestPinMem**](#function-testpinmem) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Test the pin memory allocation and transfer between CPU and GPU This function allocates a pinned memory array on the CPU, fills it with values, transfers it to the GPU, modifies it there, and transfers it back to the CPU. It then checks that the values have been correctly modified._  |
|  template int | [**TestPinMem&lt; double &gt;**](#function-testpinmem-double) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; double &gt; XModel, [**Model**](struct_model.md)&lt; double &gt; XModel\_g) <br> |
|  template int | [**TestPinMem&lt; float &gt;**](#function-testpinmem-float) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; float &gt; XModel, [**Model**](struct_model.md)&lt; float &gt; XModel\_g) <br> |
|  bool | [**Testing**](#function-testing) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Wrapping function for all the inbuilt test This function is the entry point to other function below._  |
|  template bool | [**Testing&lt; double &gt;**](#function-testing-double) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; double &gt; XModel, [**Model**](struct_model.md)&lt; double &gt; XModel\_g) <br> |
|  template bool | [**Testing&lt; float &gt;**](#function-testing-float) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; float &gt; XModel, [**Model**](struct_model.md)&lt; float &gt; XModel\_g) <br> |
|  void | [**TestingOutput**](#function-testingoutput) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_OUTDATED ?Test the output functions of the model OUTDATED? This function tests the output functions of the model by running a simple simulation and writing the output to a netcdf file._  |
|  template void | [**TestingOutput&lt; double &gt;**](#function-testingoutput-double) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; double &gt; XModel) <br> |
|  template void | [**TestingOutput&lt; float &gt;**](#function-testingoutput-float) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; float &gt; XModel) <br> |
|  void | [**Testzbinit**](#function-testzbinit) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Test the zbinit option of the model This function tests the zbinit option of the model on a sloping domain with a small amount of water The function creates its own model setup and mesh independantly to what the user inputs._  _This starts with a initial water level (zsinit=0.0 is dry) and runs for 1s comparing results every 0.1s._ |
|  T | [**ThackerBathy**](#function-thackerbathy) (T x, T y, T L, T D) <br>_create a parabolic bassin_  |
|  bool | [**ThackerLakeAtRest**](#function-thackerlakeatrest) ([**Param**](class_param.md) XParam, T zsinit) <br>_Simulate the Lake-at-rest in a parabolic bassin._  |
|  template bool | [**ThackerLakeAtRest&lt; double &gt;**](#function-thackerlakeatrest-double) ([**Param**](class_param.md) XParam, double zsinit) <br> |
|  template bool | [**ThackerLakeAtRest&lt; float &gt;**](#function-thackerlakeatrest-float) ([**Param**](class_param.md) XParam, float zsinit) <br> |
|  T | [**ValleyBathy**](#function-valleybathy) (T x, T y, T slope, T center) <br>_create V shape Valley basin_  |
|  bool | [**ZoneOutputTest**](#function-zoneoutputtest) (int nzones, T zsinit) <br>_Test the zoned output This function test the zoned output for a basic configuration._  |
|  template bool | [**ZoneOutputTest&lt; double &gt;**](#function-zoneoutputtest-double) (int nzones, double zsinit) <br> |
|  template bool | [**ZoneOutputTest&lt; float &gt;**](#function-zoneoutputtest-float) (int nzones, float zsinit) <br> |
|  void | [**alloc\_init2Darray**](#function-alloc_init2darray) (float \*\* arr, int NX, int NY) <br>_Allocates and initializes a 2D array This function allocates and fills a 2D array with zero values._  |
|  void | [**copyBlockinfo2var**](#function-copyblockinfo2var) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, int \* blkinfo, T \* z) <br>_Copies block info to an output variable This function copies block info to an output variable This function is somewhat useful when checking bugs in the mesh refinement or coarsening one needs to provide a pointer(z) allocated on the CPU to store the clockinfo This fonction only works on CPU._  |
|  template void | [**copyBlockinfo2var&lt; double &gt;**](#function-copyblockinfo2var-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, int \* blkinfo, double \* z) <br> |
|  template void | [**copyBlockinfo2var&lt; float &gt;**](#function-copyblockinfo2var-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, int \* blkinfo, float \* z) <br> |
|  void | [**copyID2var**](#function-copyid2var) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_Copies block ID to an output variable This function copies block info to an output variable This function is somewhat useful when checking bugs in the mesh refinement or coarsening one needs to provide a pointer(z) allocated on the CPU to store the clockinfo This fonction only works on CPU._  |
|  template void | [**copyID2var&lt; double &gt;**](#function-copyid2var-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**copyID2var&lt; float &gt;**](#function-copyid2var-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**diffArray**](#function-diffarray) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, std::string varname, bool checkhalo, T \* cpu, T \* gpu, T \* dummy, T \* out) <br> |
|  void | [**diffSource**](#function-diffsource) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* Fqux, T \* Su, T \* output) <br>_Calculate The source term of the equation This function Calculate The source term of the equation. This function is quite useful when checking for Lake-at-Rest states This function requires an outputCPU pointers to save the result of the calculation._  |
|  void | [**diffdh**](#function-diffdh) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* input, T \* output, T \* shuffle) <br>_Calculate The difference between adjacent cells in an array This function Calculates The difference in left and right flux terms. This function is quite useful when checking for Lake-at-Rest states This function requires a preallocated output and a shuffle (right side term) CPU pointers to save the result of teh calculation._  |
|  void | [**fillgauss**](#function-fillgauss) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T amp, T \* z) <br>_Fill an array with a gaussian bump This function fill an array with a gaussian bump._  |
|  template void | [**fillgauss&lt; double &gt;**](#function-fillgauss-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double amp, double \* z) <br> |
|  template void | [**fillgauss&lt; float &gt;**](#function-fillgauss-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float amp, float \* z) <br> |
|  void | [**fillrandom**](#function-fillrandom) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_Fill an array with random values This function fill an array with random values (0 - 1)_  |
|  template void | [**fillrandom&lt; double &gt;**](#function-fillrandom-double) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; double &gt; XBlock, double \* z) <br> |
|  template void | [**fillrandom&lt; float &gt;**](#function-fillrandom-float) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; float &gt; XBlock, float \* z) <br> |
|  void | [**init3Darray**](#function-init3darray) (float \*\*\* arr, int rows, int cols, int depths) <br>_Initializes a 3D array This function fill a 3D array with zero values._  |
|  bool | [**reductiontest**](#function-reductiontest) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Reduction test Test the algorithm for reducing the global time step on the user grid layout._  |
|  template bool | [**reductiontest&lt; double &gt;**](#function-reductiontest-double) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; double &gt; XModel, [**Model**](struct_model.md)&lt; double &gt; XModel\_g) <br> |
|  template bool | [**reductiontest&lt; float &gt;**](#function-reductiontest-float) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; float &gt; XModel, [**Model**](struct_model.md)&lt; float &gt; XModel\_g) <br> |
|  void | [**testButtingerX**](#function-testbuttingerx) ([**Param**](class_param.md) XParam, int ib, int ix, int iy, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Test the Buttinger scheme in X direction This function goes through the Buttinger scheme but instead of the normal output just prints all teh usefull values This function is/was used in the lake-at-rest verification._  |
|  bool | [**testboundaries**](#function-testboundaries) ([**Param**](class_param.md) XParam, T maxslope) <br> |
|  void | [**testkurganovX**](#function-testkurganovx) ([**Param**](class_param.md) XParam, int ib, int ix, int iy, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_Test the Kurganov scheme in X direction This function goes through the Kurganov scheme but instead of the normal output just prints all teh usefull values This function is/was used in the lake-at-rest verification See also: void testButtingerX(Param XParam, int ib, int ix, int iy, Model&lt;T&gt; XModel)_  |
|  \_\_global\_\_ void | [**vectoroffsetGPU**](#function-vectoroffsetgpu) (int nx, T offset, T \* z) <br>_A simple kernel to add an offset to a vector This is used to test the pin memory allocation and transfer between CPU and GPU._  |




























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

_Compares the Variables in a CPU model and a GPU models This function is quite useful when checking both are identical enough one needs to provide a list (vector&lt;string&gt;) of variable to check._ 
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



! 

**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) structure (CPU) 
* `XModel_g` [**Model**](struct_model.md) structure (GPU) 
* `varlist` List of variable names to check (as in OutputVarMap) 
* `checkhalo` true if halo cells should be checked, false otherwise 




        

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

_Gaussian hump propagation test._ 
```C++
template<class T>
bool GaussianHumptest (
    T zsnit,
    int gpu,
    bool compare
) 
```



!


This function tests the full hydrodynamics model and compares the results with pre-conmputed (Hard wired) values The function creates it own model setup and mesh independantly to what the user might want to do The setup consist of a centrally located gaussian hump radiating away The test stops at an arbitrary time to compare with 8 values extracted from a identical run in basilisk This function also compares the result of the GPU and CPU code (until they diverge) 

**Parameters:**


* `zsnit` Initial water surface elevation at the centre of the domain 
* `gpu` GPU device number to use (-1 for CPU only) 
* `compare` If true, compare GPU and CPU results (GPU required) 



**Returns:**

true if the test passed (results within 1e-6 of reference values) 





        

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

_Test the lake at rest condition This function simulates the first predictive step and check whether the lake at rest is preserved otherwise it prints out to screen the cells (and neighbour) where the test fails._ 
```C++
template<class T>
bool LakeAtRest (
    Param XParam,
    Model < T > XModel
) 
```



!


The function inherits the adaptation set in XParam so needs to be rerun to accnout for the different scenarios:
* uniform level
* flow from coasrse to fine
* flow from fine to coarse This is done in the higher level wrapping function 

**Parameters:**


  * `XParam` [**Model**](struct_model.md) parameters 
  * `XModel` [**Model**](struct_model.md) variables 



**Returns:**

true if test passed 







        

<hr>



### function MakValleyBathy 

_Creates a valley bathymetry This function creates a valley bathymetry with a given slope and center It also adds a wall around the domain to avoid boundary effects._ 
```C++
template<class T>
Forcing < float > MakValleyBathy (
    Param XParam,
    T slope,
    bool bottop,
    bool flip
) 
```





**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters (only used to set the mesh size) 
* `slope` Slope of the valley 
* `bottop` true if the valley is oriented in the y direction (i.e. bottom/top boundaries), false if in the x direction (i.e. left/right boundaries) 
* `flip` true if the valley is oriented towards top or right, false if towards bottom or left 



**Returns:**

A [**Forcing**](struct_forcing.md) structure containing the bathymetry 





        

<hr>



### function MassConserveSteepSlope 

[_**River**_](class_river.md) _inflow mass conservation test on steep slope._
```C++
template<class T>
bool MassConserveSteepSlope (
    T zsnit,
    int gpu
) 
```



!


This function tests the mass conservation of the vertical injection (used for rivers) The function creates it own model setup and mesh independantly to what the user might want to do This starts with a initial water level (zsnit=0 is dry) and runs for 0.1s before comparing results with zsnit=0.1 that is approx 20 steps 

**Parameters:**


* `zsnit` Initial water surface elevation at the centre of the domain 
* `gpu` GPU device number to use (-1 for CPU only) 



**Returns:**

true if the test passed (mass conservation within 5%) 





        

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

_Test the Initial and Continuous losses implementation This function tests the Initial Losses and Continuous Losses implementation a plain domain, under constant rain. The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsinit=0.0 is dry) and runs for 1s comparing results every 0.1s (that is approx 20 steps)_ 
```C++
template<class T>
bool Rainlossestest (
    T zsinit,
    int gpu,
    float alpha
) 
```



! 

**Parameters:**


* `zsinit` Initial water level 
* `gpu` GPU device to use (or -1 for CPU) 
* `alpha` Tolerance for the test (relative error) 



**Returns:**

true if test passed 





        

<hr>



### function Raintest 

_Test the rain input and mass conservation This function tests the mass conservation of the spacial injection (used to model rain on grid) The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsnit=0.0 is dry) and runs for 0.1s before comparing results with zsnit=0.1 that is approx 20 steps._ 
```C++
template<class T>
bool Raintest (
    T zsnit,
    int gpu,
    float alpha,
    int engine
) 
```



! 

**Parameters:**


* `zsnit` Initial water level 
* `gpu` GPU device to use (or -1 for CPU) 
* `alpha` Slope of the bathymetry in % 
* `engine` Engine to use (0=non-hydrostatic, 1=hydrostatic) 



**Returns:**

true if test passed 





        

<hr>



### function Raintestinput 

_Test the rain input options This function tests the different inputs for rain forcing. This test is based on the paper Aureli2020, the 3 slopes test with regional rain. The experiment has been presented in Iwagaki1955. The first test compares a time varying rain input using a uniform time serie forcing and a time varying 2D field (with same value). The second test check the 3D rain forcing (comparing it to expected values)._ 
```C++
bool Raintestinput (
    int gpu
) 
```



! 

**Parameters:**


* `gpu` GPU device to use (or -1 for CPU) 



**Returns:**

true if test passed 





        

<hr>



### function Raintestmap 

_Test the rain input options and return the flux at the bottom of the slope This function return the flux at the bottom of the 3 part slope for different types of rain forcings using the test case based on Iwagaki1955._ 
```C++
template<class T>
std::vector< float > Raintestmap (
    int gpu,
    int dimf,
    T zinit
) 
```



! \fnstd::vector&lt;float&gt; Raintestmap(int gpu, int dimf, T zinit) 

**Parameters:**


* `gpu` GPU device to use (or -1 for CPU) 
* `dimf` Dimension of the rain forcing (1=uniform, 3=2 
* `zinit` Initial water level 



**Returns:**

vector of flux at the bottom of the slope 





        

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



!


This function creates a dry steep valley topography to a given level and run the model for a while and checks that the Volume matches the theory.


The function can test the water volume for 4 scenario each time:
* left to right: bottop=false & flip=true;
* right to left: bottop=false & flip=false;
* bottom to top: bottop=true & flip=true;
* top to bottom: bottop=true & flip=false;




The function inherits the adaptation set in XParam so needs to be rerun to account for the different scenarios:
* uniform level
* flow from coarse to fine
* flow from fine to coarse This is done in the higher level wrapping function 

**Parameters:**


  * `XParam` [**Model**](struct_model.md) parameters 
  * `slope` slope of the valley sides 
  * `bottop` if true the river flows bottom to top, if false left to right 
  * `flip` if true the river flows right to left or top to bottom, if false left to right or bottom to top 



**Returns:**

true if test passed 







        

<hr>



### function Rivertest 

[_**River**_](class_river.md) _inflow mass conservation test._
```C++
template<class T>
bool Rivertest (
    T zsnit,
    int gpu
) 
```



!


This function tests the mass conservation of the vertical injection (used for rivers) The function creates it own model setup and mesh independantly to what the user might want to do This starts with a initial water level (zsnit=0 is dry) and runs for 0.1s before comparing results with zsnit=0.1 that is approx 20 steps 

**Parameters:**


* `zsnit` Initial water surface elevation at the centre of the domain 
* `gpu` GPU device number to use (-1 for CPU only) 



**Returns:**

true if the test is successful (mass is conserved within 0.1% of the theoretical value) 





        

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

_Test the aoibnd option of the model This function tests the aoibnd option of the model on a valley domain with a small amount of water The function creates its own model setup and mesh independantly to what the user inputs._  _This starts with a initial water level (zsinit=0.0 is dry) and runs for 20s._
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





**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) structure (CPU) 
* `XModel_g` [**Model**](struct_model.md) structure (GPU) 
* `bottop` true if the boundary condition to test is bottom/top, false for left/right 
* `flip` true if the boundary condition to test is top or right, false for bottom or left 
* `withaoi` true if the AOI is to be used, false otherwise 



**Returns:**

1 if the test passed (i.e. the model runs without crashing and gives a reasonable result), 0 otherwise 





        

<hr>



### function TestFirsthalfstep 

_Test the first half step of the model_  _This function tests the first half step of the model on a sloping domain with a small amount of water The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsinit=0.0 is dry) and runs for 1s comparing results every 0.1s._
```C++
template<class T>
void TestFirsthalfstep (
    Param XParam,
    Forcing < float > XForcing,
    Model < T > XModel,
    Model < T > XModel_g
) 
```





**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) structure (CPU) 
* `XModel_g` [**Model**](struct_model.md) structure (GPU) 




        

<hr>



### function TestFlexibleOutputTimes 

_Test the reading of flexible output times._ 
```C++
template<class T>
bool TestFlexibleOutputTimes (
    int gpu,
    T ref,
    int scenario
) 
```



This function creates a case set-up with a param file, read it. It tests the reading and default values used for times outputs. It checks the vectors for time outputs. 

**Parameters:**


* `gpu` GPU to use (-1 for CPU only) 
* `ref` Reference elevation for the bathymetry files 
* `scenario` Scenario to test (not used here but could be used to test different input cases) 



**Returns:**

true if test passed (i.e. the model runs without crashing and gives a reasonable result) 





        

<hr>



### function TestGradientSpeed 

_Test the speed of different gradient functions This function fill an array with random values (0 - 1)_ 
```C++
template<class T>
int TestGradientSpeed (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g
) 
```



!


This function test the spped and accuracy of a new gradient function gradient are only calculated for zb but assigned to different gradient variable for storage 

**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) structure (CPU) 
* `XModel_g` [**Model**](struct_model.md) structure (GPU) 



**Returns:**

1 if test passed 





        

<hr>



### function TestHaloSpeed 

_Test the speed of different halo filling functions This function test the speed and accuracy of a new gradient function gradient are only calculated for zb but assigned to different gradient variable for storage._ 
```C++
template<class T>
bool TestHaloSpeed (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g
) 
```



! 

**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) structure (CPU) 
* `XModel_g` [**Model**](struct_model.md) structure (GPU) 



**Returns:**

true if test passed 





        

<hr>



### function TestInstability 

_Test the stability of the model This function tests the stability of the model on a sloping domain with a small amount of water The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsinit=0.0 is dry) and runs for 1s comparing results every 0.1s._ 
```C++
template<class T>
int TestInstability (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g
) 
```





**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) structure (CPU) 
* `XModel_g` [**Model**](struct_model.md) structure (GPU) 



**Returns:**

0 if test failed (i.e. unstable), 1 if test passed 





        

<hr>



### function TestMultiBathyRough 

_Test the reading of multiple bathymetry and roughness files._ 
```C++
template<class T>
bool TestMultiBathyRough (
    int gpu,
    T ref,
    int scenario
) 
```



!


This function creates bathy and roughtness files and tests their reading (and interpolation) The objectif is particularly to test multi bathy/roughness inputs and value/file input.




**Parameters:**


* `gpu` GPU to use (-1 for CPU only) 
* `ref` Reference elevation for the bathymetry files 
* `scenario` Scenario to test (0: R1 in the middle of the domain, 1: R1 covering the whole domain) 



**Returns:**

true if test passed (i.e. the model runs without crashing and gives a reasonable result) 





        

<hr>



### function TestPinMem 

_Test the pin memory allocation and transfer between CPU and GPU This function allocates a pinned memory array on the CPU, fills it with values, transfers it to the GPU, modifies it there, and transfers it back to the CPU. It then checks that the values have been correctly modified._ 
```C++
template<class T>
int TestPinMem (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g
) 
```





**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters (only GPUDEVICE is used) 
* `XModel` [**Model**](struct_model.md) structure (CPU) 
* `XModel_g` [**Model**](struct_model.md) structure (GPU) 



**Returns:**

1 if the test passed (i.e. the values are as expected), 0 otherwise 





        

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

_Wrapping function for all the inbuilt test This function is the entry point to other function below._ 
```C++
template<class T>
bool Testing (
    Param XParam,
    Forcing < float > XForcing,
    Model < T > XModel,
    Model < T > XModel_g
) 
```



Test 0 is a gausian hump propagating on a flat uniorm cartesian mesh (both GPU and CPU version tested) Test 1 is vertical discharge on a flat uniorm cartesian mesh (GPU or CPU version) Test 2 Gaussian wave on Cartesian grid (same as test 0): CPU vs GPU (GPU required) Test 3 Test Reduction algorithm Test 4 Boundary condition test Test 5 Lake at rest test for Ardusse/kurganov reconstruction/scheme Test 6 Mass conservation on a slope Test 7 Mass conservation with rain fall on grid Test 8 Rain Map forcing (comparison map and Time Serie and test case with slope and non-uniform rain map) Test 9 Zoned output (test zoned outputs with adaptative grid) Test 10 Initial Loss / Continuous Loss on a slope, under uniform rain Test 11 Wet/dry Instability test with Conserve Elevation Test 12 Calendar time to second conversion Test 13 Multi bathy and roughness map input Test 14 Test AOI bnds aswall to start with Test 15 Flexible times reading


Test 99 Run all the test with test number &lt; 99.


The following test are not independant, they are tools to check or debug a personnal case Test 998 Compare resuts between the CPU and GPU Flow functions (GPU required) Test 999 Run the main loop and engine in debug mode




**Parameters:**


* `XParam` Simulation parameters 
* `XForcing` [**Forcing**](struct_forcing.md) data structure 
* `XModel` Host model data structure 
* `XModel_g` Device model data structure 



**Returns:**

true if all the tests passed 





        

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

_OUTDATED ?Test the output functions of the model OUTDATED? This function tests the output functions of the model by running a simple simulation and writing the output to a netcdf file._ 
```C++
template<class T>
void TestingOutput (
    Param XParam,
    Model < T > XModel
) 
```



! 

**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) structure (CPU) 




        

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

_Test the zbinit option of the model This function tests the zbinit option of the model on a sloping domain with a small amount of water The function creates its own model setup and mesh independantly to what the user inputs._  _This starts with a initial water level (zsinit=0.0 is dry) and runs for 1s comparing results every 0.1s._
```C++
template<class T>
void Testzbinit (
    Param XParam,
    Forcing < float > XForcing,
    Model < T > XModel,
    Model < T > XModel_g
) 
```





**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` [**Model**](struct_model.md) structure (CPU) 
* `XModel_g` [**Model**](struct_model.md) structure (GPU) 




        

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



!


This function creates a parabolic bassin. The function returns a single value of the bassin


Borrowed from Buttinger et al. 2019.


#### Reference



Buttinger-Kreuzhuber, A., Horváth, Z., Noelle, S., Blöschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional structured grids over abrupt topography, Advances in water resources, 127, 89–108, 2019. 

**Parameters:**


* `x` x coordinate 
* `y` y coordinate 
* `L` characteristic length scale of the bassin 
* `D` characteristic depth of the bassin 



**Returns:**

the depth of the basin at point (x,y) 






        

<hr>



### function ThackerLakeAtRest 

_Simulate the Lake-at-rest in a parabolic bassin._ 
```C++
template<class T>
bool ThackerLakeAtRest (
    Param XParam,
    T zsinit
) 
```



This function creates a parabolic bassin filled to a given level and run the modle for a while and checks that the velocities in the lake remain very small thus verifying the well-balancedness of teh engine and the Lake-at-rest condition.


Borrowed from Buttinger et al. 2019.


#### Reference



Buttinger-Kreuzhuber, A., Horváth, Z., Noelle, S., Blöschl, G., and Waser, J.: A fast second-order shallow water scheme on two-dimensional structured grids over abrupt topography, Advances in water resources, 127, 89–108, 2019. 

**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `zsinit` initial water surface elevation 



**Returns:**

true if test passed 






        

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



!


This function creates a simple V shape Valley basin




**Parameters:**


* `x` x coordinate 
* `y` y coordinate 
* `slope` slope of the valley sides 
* `center` x coordinate of the valley center 



**Returns:**

the depth of the basin at point (x,y) 





        

<hr>



### function ZoneOutputTest 

_Test the zoned output This function test the zoned output for a basic configuration._ 
```C++
template<class T>
bool ZoneOutputTest (
    int nzones,
    T zsinit
) 
```



! 

**Parameters:**


* `nzones` Number of zones to test (1 or 3) 
* `zsinit` Initial water level 



**Returns:**

true if test passed 





        

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

_Allocates and initializes a 2D array This function allocates and fills a 2D array with zero values._ 
```C++
void alloc_init2Darray (
    float ** arr,
    int NX,
    int NY
) 
```



! 

**Parameters:**


* `arr` Pointer to the 2D array 
* `NX` Number of rows 
* `NY` Number of columns 




        

<hr>



### function copyBlockinfo2var 

_Copies block info to an output variable This function copies block info to an output variable This function is somewhat useful when checking bugs in the mesh refinement or coarsening one needs to provide a pointer(z) allocated on the CPU to store the clockinfo This fonction only works on CPU._ 
```C++
template<class T>
void copyBlockinfo2var (
    Param XParam,
    BlockP < T > XBlock,
    int * blkinfo,
    T * z
) 
```



!




**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters (only nblk, blkwidth, xo, yo, xmax, ymax and dx are used) 
* `XBlock` Block parameters (only active is used) 
* `blkinfo` Block information array (CPU) 
* `z` Array to fill 




        

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

_Copies block ID to an output variable This function copies block info to an output variable This function is somewhat useful when checking bugs in the mesh refinement or coarsening one needs to provide a pointer(z) allocated on the CPU to store the clockinfo This fonction only works on CPU._ 
```C++
template<class T>
void copyID2var (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```



!




**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters (only nblk, blkwidth, xo, yo, xmax, ymax and dx are used) 
* `XBlock` Block parameters (only active is used) 
* `z` Array to fill 




        

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

_Calculate The source term of the equation This function Calculate The source term of the equation. This function is quite useful when checking for Lake-at-Rest states This function requires an outputCPU pointers to save the result of the calculation._ 
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



! 

**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters (only nblk and blkwidth are used) 
* `XBlock` Block parameters (only active are used) 
* `Fqux` Input array 
* `Su` Input array 
* `output` Output array (source term) 




        

<hr>



### function diffdh 

_Calculate The difference between adjacent cells in an array This function Calculates The difference in left and right flux terms. This function is quite useful when checking for Lake-at-Rest states This function requires a preallocated output and a shuffle (right side term) CPU pointers to save the result of teh calculation._ 
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



! 

**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters (only nblk and blkwidth are used) 
* `XBlock` Block parameters (only active are used) 
* `input` Input array 
* `output` Output array (difference) 
* `shuffle` Output array (right side term) 




        

<hr>



### function fillgauss 

_Fill an array with a gaussian bump This function fill an array with a gaussian bump._ 
```C++
template<class T>
void fillgauss (
    Param XParam,
    BlockP < T > XBlock,
    T amp,
    T * z
) 
```



!


borrowed/adapted from Basilisk test (?) 

**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters (only nblk, blkwidth, xo, yo, xmax, ymax and dx are used) 
* `XBlock` Block parameters (only active, level, xo and yo are used) 
* `amp` Amplitude of the gaussian bump 
* `z` Array to fill 




        

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

_Fill an array with random values This function fill an array with random values (0 - 1)_ 
```C++
template<class T>
void fillrandom (
    Param XParam,
    BlockP < T > XBlock,
    T * z
) 
```



! 

**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters (only nblk and blkwidth are used) 
* `XBlock` Block parameters (only active are used) 
* `z` Array to fill 




        

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

_Initializes a 3D array This function fill a 3D array with zero values._ 
```C++
void init3Darray (
    float *** arr,
    int rows,
    int cols,
    int depths
) 
```



! 

**Parameters:**


* `arr` Pointer to the 3D array 
* `rows` Number of rows 
* `cols` Number of columns 
* `depths` Number of depths 




        

<hr>



### function reductiontest 

_Reduction test Test the algorithm for reducing the global time step on the user grid layout._ 
```C++
template<class T>
bool reductiontest (
    Param XParam,
    Model < T > XModel,
    Model < T > XModel_g
) 
```



! 

**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `XModel` CPU model 
* `XModel_g` GPU model 



**Returns:**

true if test passed 





        

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

_Test the Buttinger scheme in X direction This function goes through the Buttinger scheme but instead of the normal output just prints all teh usefull values This function is/was used in the lake-at-rest verification._ 
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



!


See also: void testkurganovX(Param XParam, int ib, int ix, int iy, Model&lt;T&gt; XModel) 

**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `ib` Block index 
* `ix` X index in the block 
* `iy` Y index in the block 
* `XModel` [**Model**](struct_model.md) variables 




        

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

_Test the Kurganov scheme in X direction This function goes through the Kurganov scheme but instead of the normal output just prints all teh usefull values This function is/was used in the lake-at-rest verification See also: void testButtingerX(Param XParam, int ib, int ix, int iy, Model&lt;T&gt; XModel)_ 
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



! 

**Parameters:**


* `XParam` [**Model**](struct_model.md) parameters 
* `ib` Block index 
* `ix` X index in the block 
* `iy` Y index in the block 
* `XModel` [**Model**](struct_model.md) variables 




        

<hr>



### function vectoroffsetGPU 

_A simple kernel to add an offset to a vector This is used to test the pin memory allocation and transfer between CPU and GPU._ 
```C++
template<class T>
__global__ void vectoroffsetGPU (
    int nx,
    T offset,
    T * z
) 
```





**Parameters:**


* `nx` Number of elements in the vector 
* `offset` Offset to add 
* `z` Vector to modify (input and output) 



**Template parameters:**


* `T` Data type of the vector (float or double) 




        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/Testing.cu`


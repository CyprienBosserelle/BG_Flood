

# File Testing.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Testing.h**](_testing_8h.md)

[Go to the source code of this file](_testing_8h_source.md)



* `#include "General.h"`
* `#include "Param.h"`
* `#include "Write_txtlog.h"`
* `#include "ReadInput.h"`
* `#include "ReadForcing.h"`
* `#include "Util_CPU.h"`
* `#include "Arrays.h"`
* `#include "Forcing.h"`
* `#include "Mesh.h"`
* `#include "Setup_GPU.h"`
* `#include "Mainloop.h"`
* `#include "FlowCPU.h"`
* `#include "FlowGPU.h"`
* `#include "Adaptation.h"`
* `#include "utctime.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**CompareCPUvsGPU**](#function-comparecpuvsgpu) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g, std::vector&lt; std::string &gt; varlist, bool checkhalo) <br>_Compares the Variables in a CPU model and a GPU models This function is quite useful when checking both are identical enough one needs to provide a list (vector&lt;string&gt;) of variable to check._  |
|  bool | [**GaussianHumptest**](#function-gaussianhumptest) (T zsnit, int gpu, bool compare) <br>_Gaussian hump propagation test._  |
|  bool | [**MassConserveSteepSlope**](#function-massconservesteepslope) (T zsnit, int gpu) <br>[_**River**_](class_river.md) _inflow mass conservation test on steep slope._ |
|  bool | [**Rainlossestest**](#function-rainlossestest) (T zsnit, int gpu, float alpha) <br>_Test the Initial and Continuous losses implementation This function tests the Initial Losses and Continuous Losses implementation a plain domain, under constant rain. The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsinit=0.0 is dry) and runs for 1s comparing results every 0.1s (that is approx 20 steps)_  |
|  bool | [**Raintest**](#function-raintest) (T zsnit, int gpu, float alpha, int engine) <br>_Test the rain input and mass conservation This function tests the mass conservation of the spacial injection (used to model rain on grid) The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsnit=0.0 is dry) and runs for 0.1s before comparing results with zsnit=0.1 that is approx 20 steps._  |
|  bool | [**Raintestinput**](#function-raintestinput) (int gpu) <br>_Test the rain input options This function tests the different inputs for rain forcing. This test is based on the paper Aureli2020, the 3 slopes test with regional rain. The experiment has been presented in Iwagaki1955. The first test compares a time varying rain input using a uniform time serie forcing and a time varying 2D field (with same value). The second test check the 3D rain forcing (comparing it to expected values)._  |
|  std::vector&lt; float &gt; | [**Raintestmap**](#function-raintestmap) (int gpu, int dimf, T zinit) <br>_Test the rain input options and return the flux at the bottom of the slope This function return the flux at the bottom of the 3 part slope for different types of rain forcings using the test case based on Iwagaki1955._  |
|  bool | [**Rivertest**](#function-rivertest) (T zsnit, int gpu) <br>[_**River**_](class_river.md) _inflow mass conservation test._ |
|  bool | [**TestFlexibleOutputTimes**](#function-testflexibleoutputtimes) (int gpu, T ref, int scenario) <br>_Test the reading of flexible output times._  |
|  bool | [**TestMultiBathyRough**](#function-testmultibathyrough) (int gpu, T ref, int secnario) <br>_Test the reading of multiple bathymetry and roughness files._  |
|  bool | [**Testing**](#function-testing) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing, [**Model**](struct_model.md)&lt; T &gt; XModel, [**Model**](struct_model.md)&lt; T &gt; XModel\_g) <br>_Wrapping function for all the inbuilt test This function is the entry point to other function below._  |
|  void | [**TestingOutput**](#function-testingoutput) ([**Param**](class_param.md) XParam, [**Model**](struct_model.md)&lt; T &gt; XModel) <br>_OUTDATED ?Test the output functions of the model OUTDATED? This function tests the output functions of the model by running a simple simulation and writing the output to a netcdf file._  |
|  bool | [**ZoneOutputTest**](#function-zoneoutputtest) (int nzones, T zsinit) <br>_Test the zoned output This function test the zoned output for a basic configuration._  |
|  void | [**copyBlockinfo2var**](#function-copyblockinfo2var) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, int \* blkinfo, T \* z) <br>_Copies block info to an output variable This function copies block info to an output variable This function is somewhat useful when checking bugs in the mesh refinement or coarsening one needs to provide a pointer(z) allocated on the CPU to store the clockinfo This fonction only works on CPU._  |
|  void | [**copyID2var**](#function-copyid2var) ([**Param**](class_param.md) XParam, [**BlockP**](struct_block_p.md)&lt; T &gt; XBlock, T \* z) <br>_Copies block ID to an output variable This function copies block info to an output variable This function is somewhat useful when checking bugs in the mesh refinement or coarsening one needs to provide a pointer(z) allocated on the CPU to store the clockinfo This fonction only works on CPU._  |
|  bool | [**testboundaries**](#function-testboundaries) ([**Param**](class_param.md) XParam, T maxslope) <br> |




























## Public Functions Documentation




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



### function Rainlossestest 

_Test the Initial and Continuous losses implementation This function tests the Initial Losses and Continuous Losses implementation a plain domain, under constant rain. The function creates its own model setup and mesh independantly to what the user inputs. This starts with a initial water level (zsinit=0.0 is dry) and runs for 1s comparing results every 0.1s (that is approx 20 steps)_ 
```C++
template<class T>
bool Rainlossestest (
    T zsnit,
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



### function TestMultiBathyRough 

_Test the reading of multiple bathymetry and roughness files._ 
```C++
template<class T>
bool TestMultiBathyRough (
    int gpu,
    T ref,
    int secnario
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



### function testboundaries 

```C++
template<class T>
bool testboundaries (
    Param XParam,
    T maxslope
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Testing.h`




# File Testing.h



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Testing.h**](Testing_8h.md)

[Go to the source code of this file](Testing_8h_source.md)



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
|  void | [**CompareCPUvsGPU**](#function-comparecpuvsgpu) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g, std::vector&lt; std::string &gt; varlist, bool checkhalo) <br> |
|  bool | [**GaussianHumptest**](#function-gaussianhumptest) (T zsnit, int gpu, bool compare) <br> |
|  bool | [**MassConserveSteepSlope**](#function-massconservesteepslope) (T zsnit, int gpu) <br> |
|  bool | [**Rainlossestest**](#function-rainlossestest) (T zsnit, int gpu, float alpha) <br> |
|  bool | [**Raintest**](#function-raintest) (T zsnit, int gpu, float alpha, int engine) <br> |
|  bool | [**Raintestinput**](#function-raintestinput) (int gpu) <br> |
|  std::vector&lt; float &gt; | [**Raintestmap**](#function-raintestmap) (int gpu, int dimf, T zinit) <br> |
|  bool | [**Rivertest**](#function-rivertest) (T zsnit, int gpu) <br> |
|  bool | [**TestFlexibleOutputTimes**](#function-testflexibleoutputtimes) (int gpu, T ref, int scenario) <br> |
|  bool | [**TestMultiBathyRough**](#function-testmultibathyrough) (int gpu, T ref, int secnario) <br> |
|  bool | [**Testing**](#function-testing) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br> |
|  void | [**TestingOutput**](#function-testingoutput) ([**Param**](classParam.md) XParam, [**Model**](structModel.md)&lt; T &gt; XModel) <br> |
|  bool | [**ZoneOutputTest**](#function-zoneoutputtest) (int nzones, T zsinit) <br> |
|  void | [**copyBlockinfo2var**](#function-copyblockinfo2var) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, int \* blkinfo, T \* z) <br> |
|  void | [**copyID2var**](#function-copyid2var) ([**Param**](classParam.md) XParam, [**BlockP**](structBlockP.md)&lt; T &gt; XBlock, T \* z) <br> |
|  bool | [**testboundaries**](#function-testboundaries) ([**Param**](classParam.md) XParam, T maxslope) <br> |




























## Public Functions Documentation




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



### function Rainlossestest 

```C++
template<class T>
bool Rainlossestest (
    T zsnit,
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



### function TestMultiBathyRough 

```C++
template<class T>
bool TestMultiBathyRough (
    int gpu,
    T ref,
    int secnario
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


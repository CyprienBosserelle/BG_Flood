

# Class Param



[**ClassList**](annotated.md) **>** [**Param**](class_param.md)



[More...](#detailed-description)

* `#include <Param.h>`





















## Public Attributes

| Type | Name |
| ---: | :--- |
|  std::string | [**AdaptCrit**](#variable-adaptcrit)  <br> |
|  int \* | [**AdaptCrit\_funct\_pointer**](#variable-adaptcrit_funct_pointer)  <br> |
|  std::string | [**Adapt\_arg1**](#variable-adapt_arg1)  <br> |
|  std::string | [**Adapt\_arg2**](#variable-adapt_arg2)  <br> |
|  std::string | [**Adapt\_arg3**](#variable-adapt_arg3)  <br> |
|  std::string | [**Adapt\_arg4**](#variable-adapt_arg4)  <br> |
|  std::string | [**Adapt\_arg5**](#variable-adapt_arg5)  <br> |
|  double | [**CFL**](#variable-cfl)   = `0.5`<br> |
|  double | [**Cd**](#variable-cd)   = `0.002`<br> |
|  bool | [**ForceMassConserve**](#variable-forcemassconserve)   = `false`<br> |
|  int | [**GPUDEVICE**](#variable-gpudevice)   = `0`<br> |
|  size\_t | [**GPU\_initmem\_byte**](#variable-gpu_initmem_byte)  <br> |
|  size\_t | [**GPU\_totalmem\_byte**](#variable-gpu_totalmem_byte)  <br> |
|  double | [**Pa2m**](#variable-pa2m)   = `0.00009916`<br> |
|  double | [**Paref**](#variable-paref)   = `101300.0`<br> |
|  double | [**Radius**](#variable-radius)   = `6371220.`<br> |
|  std::vector&lt; [**TSoutnode**](class_t_soutnode.md) &gt; | [**TSnodesout**](#variable-tsnodesout)  <br> |
|  [**T\_output**](class_t__output.md) | [**Toutput**](#variable-toutput)  <br> |
|  double | [**VelThreshold**](#variable-velthreshold)   = `-1.0`<br> |
|  int | [**adaptmaxiteration**](#variable-adaptmaxiteration)   = `20`<br> |
|  float | [**addoffset**](#variable-addoffset)   = `0.0f`<br> |
|  int | [**aoibnd**](#variable-aoibnd)   = `0`<br> |
|  bool | [**atmpforcing**](#variable-atmpforcing)   = `false`<br> |
|  int | [**blkmemwidth**](#variable-blkmemwidth)   = `0`<br> |
|  int | [**blksize**](#variable-blksize)   = `0`<br> |
|  int | [**blkwidth**](#variable-blkwidth)   = `16`<br> |
|  double | [**bndfiltertime**](#variable-bndfiltertime)   = `60.0`<br> |
|  double | [**bndrelaxtime**](#variable-bndrelaxtime)   = `3600.0`<br> |
|  double | [**bndtaper**](#variable-bndtaper)   = `0.0`<br> |
|  bool | [**botbnd**](#variable-botbnd)   = `false`<br> |
|  double | [**cf**](#variable-cf)   = `0.0001`<br> |
|  double | [**cl**](#variable-cl)   = `0.0`<br> |
|  bool | [**conserveElevation**](#variable-conserveelevation)   = `false`<br> |
|  std::string | [**crs\_ref**](#variable-crs_ref)   = `"no\_crs"`<br> |
|  double | [**deformmaxtime**](#variable-deformmaxtime)   = `0.0`<br> |
|  double | [**delta**](#variable-delta)  <br> |
|  int | [**doubleprecision**](#variable-doubleprecision)   = `0`<br> |
|  double | [**dt**](#variable-dt)   = `0.0`<br> |
|  double | [**dtinit**](#variable-dtinit)   = `-1`<br> |
|  double | [**dtmin**](#variable-dtmin)   = `0.0005`<br> |
|  double | [**dx**](#variable-dx)   = `nan("")`<br> |
|  clock\_t | [**endcputime**](#variable-endcputime)  <br> |
|  double | [**endtime**](#variable-endtime)   = `std::numeric\_limits&lt;double&gt;::max()`<br> |
|  int | [**engine**](#variable-engine)   = `1`<br> |
|  double | [**eps**](#variable-eps)   = `0.0001`<br> |
|  int | [**frictionmodel**](#variable-frictionmodel)   = `0`<br> |
|  double | [**g**](#variable-g)   = `9.81`<br> |
|  double | [**grdalpha**](#variable-grdalpha)   = `nan("")`<br> |
|  int | [**halowidth**](#variable-halowidth)   = `1`<br> |
|  std::string | [**hotstartfile**](#variable-hotstartfile)  <br> |
|  int | [**hotstep**](#variable-hotstep)   = `0`<br> |
|  double | [**il**](#variable-il)   = `0.0`<br> |
|  bool | [**infiltration**](#variable-infiltration)   = `false`<br> |
|  int | [**initlevel**](#variable-initlevel)   = `0`<br> |
|  double | [**inittime**](#variable-inittime)   = `0.0`<br> |
|  double | [**lat**](#variable-lat)   = `0.0`<br> |
|  bool | [**leftbnd**](#variable-leftbnd)   = `false`<br> |
|  double | [**mask**](#variable-mask)   = `9999.0`<br> |
|  int | [**maxTSstorage**](#variable-maxtsstorage)   = `16384`<br> |
|  int | [**maxlevel**](#variable-maxlevel)   = `-99999`<br> |
|  double | [**membuffer**](#variable-membuffer)   = `1.05`<br> |
|  int | [**minlevel**](#variable-minlevel)   = `-99999`<br> |
|  int | [**navailblk**](#variable-navailblk)   = `0`<br> |
|  int | [**nblk**](#variable-nblk)   = `0`<br> |
|  int | [**nblkmem**](#variable-nblkmem)   = `0`<br> |
|  int | [**nblkriver**](#variable-nblkriver)   = `0`<br> |
|  int | [**nbndblkbot**](#variable-nbndblkbot)   = `0`<br> |
|  int | [**nbndblkleft**](#variable-nbndblkleft)   = `0`<br> |
|  int | [**nbndblkright**](#variable-nbndblkright)   = `0`<br> |
|  int | [**nbndblktop**](#variable-nbndblktop)   = `0`<br> |
|  int | [**nmaskblk**](#variable-nmaskblk)   = `0`<br> |
|  int | [**nrivers**](#variable-nrivers)   = `0`<br> |
|  int | [**nx**](#variable-nx)   = `0`<br> |
|  int | [**ny**](#variable-ny)   = `0`<br> |
|  std::string | [**outfile**](#variable-outfile)   = `"Output.nc"`<br> |
|  int | [**outishift**](#variable-outishift)   = `0`<br> |
|  int | [**outjshift**](#variable-outjshift)   = `0`<br> |
|  bool | [**outmax**](#variable-outmax)   = `false`<br> |
|  bool | [**outmean**](#variable-outmean)   = `false`<br> |
|  double | [**outputtimestep**](#variable-outputtimestep)   = `0.0`<br> |
|  bool | [**outtwet**](#variable-outtwet)   = `false`<br> |
|  std::vector&lt; std::string &gt; | [**outvars**](#variable-outvars)  <br> |
|  std::vector&lt; [**outzoneP**](classoutzone_p.md) &gt; | [**outzone**](#variable-outzone)  <br> |
|  int | [**posdown**](#variable-posdown)   = `0`<br> |
|  bool | [**rainbnd**](#variable-rainbnd)   = `false`<br> |
|  bool | [**rainforcing**](#variable-rainforcing)   = `false`<br> |
|  std::string | [**reftime**](#variable-reftime)   = `""`<br> |
|  bool | [**resetmax**](#variable-resetmax)   = `false`<br> |
|  double | [**rho**](#variable-rho)   = `1025.0`<br> |
|  bool | [**rightbnd**](#variable-rightbnd)   = `false`<br> |
|  bool | [**savebyblk**](#variable-savebyblk)   = `true`<br> |
|  float | [**scalefactor**](#variable-scalefactor)   = `0.01f`<br> |
|  clock\_t | [**setupcputime**](#variable-setupcputime)  <br> |
|  int | [**smallnc**](#variable-smallnc)   = `1`<br> |
|  bool | [**spherical**](#variable-spherical)   = `0`<br> |
|  clock\_t | [**startcputime**](#variable-startcputime)  <br> |
|  int | [**test**](#variable-test)   = `-1`<br> |
|  double | [**theta**](#variable-theta)   = `1.3`<br> |
|  bool | [**topbnd**](#variable-topbnd)   = `false`<br> |
|  double | [**totaltime**](#variable-totaltime)   = `0.0`<br> |
|  double | [**wet\_threshold**](#variable-wet_threshold)   = `0.1`<br> |
|  bool | [**wetdryfix**](#variable-wetdryfix)   = `true`<br> |
|  bool | [**windforcing**](#variable-windforcing)   = `false`<br> |
|  double | [**xmax**](#variable-xmax)   = `nan("")`<br> |
|  double | [**xo**](#variable-xo)   = `nan("")`<br> |
|  double | [**ymax**](#variable-ymax)   = `nan("")`<br> |
|  double | [**yo**](#variable-yo)   = `nan("")`<br> |
|  double | [**zsinit**](#variable-zsinit)   = `nan("")`<br> |
|  double | [**zsoffset**](#variable-zsoffset)   = `nan("")`<br> |












































## Detailed Description


A class. A class for holding model parameters. 


    
## Public Attributes Documentation




### variable AdaptCrit 

```C++
std::string Param::AdaptCrit;
```




<hr>



### variable AdaptCrit\_funct\_pointer 

```C++
int* Param::AdaptCrit_funct_pointer;
```




<hr>



### variable Adapt\_arg1 

```C++
std::string Param::Adapt_arg1;
```




<hr>



### variable Adapt\_arg2 

```C++
std::string Param::Adapt_arg2;
```




<hr>



### variable Adapt\_arg3 

```C++
std::string Param::Adapt_arg3;
```




<hr>



### variable Adapt\_arg4 

```C++
std::string Param::Adapt_arg4;
```




<hr>



### variable Adapt\_arg5 

```C++
std::string Param::Adapt_arg5;
```




<hr>



### variable CFL 

```C++
double Param::CFL;
```




<hr>



### variable Cd 

```C++
double Param::Cd;
```




<hr>



### variable ForceMassConserve 

```C++
bool Param::ForceMassConserve;
```




<hr>



### variable GPUDEVICE 

```C++
int Param::GPUDEVICE;
```




<hr>



### variable GPU\_initmem\_byte 

```C++
size_t Param::GPU_initmem_byte;
```




<hr>



### variable GPU\_totalmem\_byte 

```C++
size_t Param::GPU_totalmem_byte;
```




<hr>



### variable Pa2m 

```C++
double Param::Pa2m;
```




<hr>



### variable Paref 

```C++
double Param::Paref;
```




<hr>



### variable Radius 

```C++
double Param::Radius;
```




<hr>



### variable TSnodesout 

```C++
std::vector<TSoutnode> Param::TSnodesout;
```




<hr>



### variable Toutput 

```C++
T_output Param::Toutput;
```




<hr>



### variable VelThreshold 

```C++
double Param::VelThreshold;
```




<hr>



### variable adaptmaxiteration 

```C++
int Param::adaptmaxiteration;
```




<hr>



### variable addoffset 

```C++
float Param::addoffset;
```




<hr>



### variable aoibnd 

```C++
int Param::aoibnd;
```




<hr>



### variable atmpforcing 

```C++
bool Param::atmpforcing;
```




<hr>



### variable blkmemwidth 

```C++
int Param::blkmemwidth;
```




<hr>



### variable blksize 

```C++
int Param::blksize;
```




<hr>



### variable blkwidth 

```C++
int Param::blkwidth;
```




<hr>



### variable bndfiltertime 

```C++
double Param::bndfiltertime;
```




<hr>



### variable bndrelaxtime 

```C++
double Param::bndrelaxtime;
```




<hr>



### variable bndtaper 

```C++
double Param::bndtaper;
```




<hr>



### variable botbnd 

```C++
bool Param::botbnd;
```




<hr>



### variable cf 

```C++
double Param::cf;
```




<hr>



### variable cl 

```C++
double Param::cl;
```




<hr>



### variable conserveElevation 

```C++
bool Param::conserveElevation;
```




<hr>



### variable crs\_ref 

```C++
std::string Param::crs_ref;
```




<hr>



### variable deformmaxtime 

```C++
double Param::deformmaxtime;
```




<hr>



### variable delta 

```C++
double Param::delta;
```




<hr>



### variable doubleprecision 

```C++
int Param::doubleprecision;
```




<hr>



### variable dt 

```C++
double Param::dt;
```




<hr>



### variable dtinit 

```C++
double Param::dtinit;
```




<hr>



### variable dtmin 

```C++
double Param::dtmin;
```




<hr>



### variable dx 

```C++
double Param::dx;
```




<hr>



### variable endcputime 

```C++
clock_t Param::endcputime;
```




<hr>



### variable endtime 

```C++
double Param::endtime;
```




<hr>



### variable engine 

```C++
int Param::engine;
```




<hr>



### variable eps 

```C++
double Param::eps;
```




<hr>



### variable frictionmodel 

```C++
int Param::frictionmodel;
```




<hr>



### variable g 

```C++
double Param::g;
```




<hr>



### variable grdalpha 

```C++
double Param::grdalpha;
```




<hr>



### variable halowidth 

```C++
int Param::halowidth;
```




<hr>



### variable hotstartfile 

```C++
std::string Param::hotstartfile;
```




<hr>



### variable hotstep 

```C++
int Param::hotstep;
```




<hr>



### variable il 

```C++
double Param::il;
```




<hr>



### variable infiltration 

```C++
bool Param::infiltration;
```




<hr>



### variable initlevel 

```C++
int Param::initlevel;
```




<hr>



### variable inittime 

```C++
double Param::inittime;
```




<hr>



### variable lat 

```C++
double Param::lat;
```




<hr>



### variable leftbnd 

```C++
bool Param::leftbnd;
```




<hr>



### variable mask 

```C++
double Param::mask;
```




<hr>



### variable maxTSstorage 

```C++
int Param::maxTSstorage;
```




<hr>



### variable maxlevel 

```C++
int Param::maxlevel;
```




<hr>



### variable membuffer 

```C++
double Param::membuffer;
```




<hr>



### variable minlevel 

```C++
int Param::minlevel;
```




<hr>



### variable navailblk 

```C++
int Param::navailblk;
```




<hr>



### variable nblk 

```C++
int Param::nblk;
```




<hr>



### variable nblkmem 

```C++
int Param::nblkmem;
```




<hr>



### variable nblkriver 

```C++
int Param::nblkriver;
```




<hr>



### variable nbndblkbot 

```C++
int Param::nbndblkbot;
```




<hr>



### variable nbndblkleft 

```C++
int Param::nbndblkleft;
```




<hr>



### variable nbndblkright 

```C++
int Param::nbndblkright;
```




<hr>



### variable nbndblktop 

```C++
int Param::nbndblktop;
```




<hr>



### variable nmaskblk 

```C++
int Param::nmaskblk;
```




<hr>



### variable nrivers 

```C++
int Param::nrivers;
```




<hr>



### variable nx 

```C++
int Param::nx;
```




<hr>



### variable ny 

```C++
int Param::ny;
```




<hr>



### variable outfile 

```C++
std::string Param::outfile;
```




<hr>



### variable outishift 

```C++
int Param::outishift;
```




<hr>



### variable outjshift 

```C++
int Param::outjshift;
```




<hr>



### variable outmax 

```C++
bool Param::outmax;
```




<hr>



### variable outmean 

```C++
bool Param::outmean;
```




<hr>



### variable outputtimestep 

```C++
double Param::outputtimestep;
```




<hr>



### variable outtwet 

```C++
bool Param::outtwet;
```




<hr>



### variable outvars 

```C++
std::vector<std::string> Param::outvars;
```




<hr>



### variable outzone 

```C++
std::vector<outzoneP> Param::outzone;
```




<hr>



### variable posdown 

```C++
int Param::posdown;
```




<hr>



### variable rainbnd 

```C++
bool Param::rainbnd;
```




<hr>



### variable rainforcing 

```C++
bool Param::rainforcing;
```




<hr>



### variable reftime 

```C++
std::string Param::reftime;
```




<hr>



### variable resetmax 

```C++
bool Param::resetmax;
```




<hr>



### variable rho 

```C++
double Param::rho;
```




<hr>



### variable rightbnd 

```C++
bool Param::rightbnd;
```




<hr>



### variable savebyblk 

```C++
bool Param::savebyblk;
```




<hr>



### variable scalefactor 

```C++
float Param::scalefactor;
```




<hr>



### variable setupcputime 

```C++
clock_t Param::setupcputime;
```




<hr>



### variable smallnc 

```C++
int Param::smallnc;
```




<hr>



### variable spherical 

```C++
bool Param::spherical;
```




<hr>



### variable startcputime 

```C++
clock_t Param::startcputime;
```




<hr>



### variable test 

```C++
int Param::test;
```




<hr>



### variable theta 

```C++
double Param::theta;
```




<hr>



### variable topbnd 

```C++
bool Param::topbnd;
```




<hr>



### variable totaltime 

```C++
double Param::totaltime;
```




<hr>



### variable wet\_threshold 

```C++
double Param::wet_threshold;
```




<hr>



### variable wetdryfix 

```C++
bool Param::wetdryfix;
```




<hr>



### variable windforcing 

```C++
bool Param::windforcing;
```




<hr>



### variable xmax 

```C++
double Param::xmax;
```




<hr>



### variable xo 

```C++
double Param::xo;
```




<hr>



### variable ymax 

```C++
double Param::ymax;
```




<hr>



### variable yo 

```C++
double Param::yo;
```




<hr>



### variable zsinit 

```C++
double Param::zsinit;
```




<hr>



### variable zsoffset 

```C++
double Param::zsoffset;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Param.h`


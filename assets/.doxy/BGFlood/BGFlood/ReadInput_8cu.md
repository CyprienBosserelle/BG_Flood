

# File ReadInput.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**ReadInput.cu**](ReadInput_8cu.md)

[Go to the source code of this file](ReadInput_8cu_source.md)



* `#include "ReadInput.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**InitialiseToutput**](#function-initialisetoutput) ([**T\_output**](classT__output.md) & Toutput\_loc, [**Param**](classParam.md) XParam) <br> |
|  std::vector&lt; double &gt; | [**ReadTRangestr**](#function-readtrangestr) (std::vector&lt; std::string &gt; timestr, double start, double end, std::string reftime) <br> |
|  std::vector&lt; std::string &gt; | [**ReadToutSTR**](#function-readtoutstr) (std::string paramstr) <br> |
|  std::vector&lt; double &gt; | [**ReadToutput**](#function-readtoutput) (std::vector&lt; std::string &gt; paramstr, [**Param**](classParam.md) XParam) <br> |
|  double | [**ReadTvalstr**](#function-readtvalstr) (std::string timestr, double start, double end, std::string reftime) <br> |
|  void | [**Readparamfile**](#function-readparamfile) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing, std::string Paramfile) <br> |
|  std::size\_t | [**case\_insensitive\_compare**](#function-case_insensitive_compare) (std::string s1, std::string s2) <br> |
|  std::size\_t | [**case\_insensitive\_compare**](#function-case_insensitive_compare) (std::string s1, std::vector&lt; std::string &gt; vecstr) <br> |
|  void | [**checkparamsanity**](#function-checkparamsanity) ([**Param**](classParam.md) & XParam, [**Forcing**](structForcing.md)&lt; float &gt; & XForcing) <br> |
|  std::string | [**findparameter**](#function-findparameter) (std::vector&lt; std::string &gt; parameterstr, std::string line) <br> |
|  std::string | [**findparameter**](#function-findparameter) (std::string parameterstr, std::string line) <br> |
|  double | [**readApproxtimestr**](#function-readapproxtimestr) (std::string input) <br> |
|  [**bndsegment**](classbndsegment.md) | [**readbndline**](#function-readbndline) (std::string parametervalue) <br> |
|  [**bndsegment**](classbndsegment.md) | [**readbndlineside**](#function-readbndlineside) (std::string parametervalue, std::string side) <br> |
|  T | [**readfileinfo**](#function-readfileinfo) (std::string input, T outinfo) <br> |
|  template [**DynForcingP**](structDynForcingP.md)&lt; float &gt; | [**readfileinfo&lt; DynForcingP&lt; float &gt; &gt;**](#function-readfileinfo-dynforcingp-float) (std::string input, [**DynForcingP**](structDynForcingP.md)&lt; float &gt; outinfo) <br> |
|  template [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; | [**readfileinfo&lt; StaticForcingP&lt; float &gt; &gt;**](#function-readfileinfo-staticforcingp-float) (std::string input, [**StaticForcingP**](structStaticForcingP.md)&lt; float &gt; outinfo) <br> |
|  template [**deformmap**](classdeformmap.md)&lt; float &gt; | [**readfileinfo&lt; deformmap&lt; float &gt; &gt;**](#function-readfileinfo-deformmap-float) (std::string input, [**deformmap**](classdeformmap.md)&lt; float &gt; outinfo) <br> |
|  template [**forcingmap**](classforcingmap.md) | [**readfileinfo&lt; forcingmap &gt;**](#function-readfileinfo-forcingmap) (std::string input, [**forcingmap**](classforcingmap.md) outinfo) <br> |
|  template [**inputmap**](classinputmap.md) | [**readfileinfo&lt; inputmap &gt;**](#function-readfileinfo-inputmap) (std::string input, [**inputmap**](classinputmap.md) outinfo) <br> |
|  bool | [**readparambool**](#function-readparambool) (std::string paramstr, bool defaultval) <br> |
|  [**Param**](classParam.md) | [**readparamstr**](#function-readparamstr) (std::string line, [**Param**](classParam.md) param) <br> |
|  [**Forcing**](structForcing.md)&lt; T &gt; | [**readparamstr**](#function-readparamstr) (std::string line, [**Forcing**](structForcing.md)&lt; T &gt; forcing) <br> |
|  double | [**setendtime**](#function-setendtime) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing) <br> |
|  void | [**split**](#function-split) (const std::string & s, char delim, std::vector&lt; std::string &gt; & elems) <br> |
|  std::vector&lt; std::string &gt; | [**split**](#function-split) (const std::string & s, char delim) <br> |
|  std::vector&lt; std::string &gt; | [**split**](#function-split) (const std::string s, const std::string delim) <br> |
|  void | [**split\_full**](#function-split_full) (const std::string & s, char delim, std::vector&lt; std::string &gt; & elems) <br> |
|  std::vector&lt; std::string &gt; | [**split\_full**](#function-split_full) (const std::string & s, char delim) <br> |
|  std::string | [**trim**](#function-trim) (const std::string & str, const std::string & whitespace) <br> |




























## Public Functions Documentation




### function InitialiseToutput 

```C++
void InitialiseToutput (
    T_output & Toutput_loc,
    Param XParam
) 
```




<hr>



### function ReadTRangestr 

```C++
std::vector< double > ReadTRangestr (
    std::vector< std::string > timestr,
    double start,
    double end,
    std::string reftime
) 
```




<hr>



### function ReadToutSTR 

```C++
std::vector< std::string > ReadToutSTR (
    std::string paramstr
) 
```




<hr>



### function ReadToutput 

```C++
std::vector< double > ReadToutput (
    std::vector< std::string > paramstr,
    Param XParam
) 
```




<hr>



### function ReadTvalstr 

```C++
double ReadTvalstr (
    std::string timestr,
    double start,
    double end,
    std::string reftime
) 
```




<hr>



### function Readparamfile 

```C++
void Readparamfile (
    Param & XParam,
    Forcing < float > & XForcing,
    std::string Paramfile
) 
```




<hr>



### function case\_insensitive\_compare 

```C++
std::size_t case_insensitive_compare (
    std::string s1,
    std::string s2
) 
```




<hr>



### function case\_insensitive\_compare 

```C++
std::size_t case_insensitive_compare (
    std::string s1,
    std::vector< std::string > vecstr
) 
```




<hr>



### function checkparamsanity 

```C++
void checkparamsanity (
    Param & XParam,
    Forcing < float > & XForcing
) 
```



Check the Sanity of both [**Param**](classParam.md) and [**Forcing**](structForcing.md) class If required some parameter are infered 


        

<hr>



### function findparameter 

```C++
std::string findparameter (
    std::vector< std::string > parameterstr,
    std::string line
) 
```




<hr>



### function findparameter 

```C++
std::string findparameter (
    std::string parameterstr,
    std::string line
) 
```



separate parameter from value 


        

<hr>



### function readApproxtimestr 

```C++
double readApproxtimestr (
    std::string input
) 
```




<hr>



### function readbndline 

```C++
bndsegment readbndline (
    std::string parametervalue
) 
```




<hr>



### function readbndlineside 

```C++
bndsegment readbndlineside (
    std::string parametervalue,
    std::string side
) 
```




<hr>



### function readfileinfo 

```C++
template<class T>
T readfileinfo (
    std::string input,
    T outinfo
) 
```



convert file name into name and extension This is used for various input classes


template inputmap readfileinfo&lt;inputmap&gt;(std::string input, inputmap outinfo); template forcingmap readfileinfo&lt;forcingmap&gt;(std::string input, forcingmap outinfo); template StaticForcingP&lt;float&gt; readfileinfo&lt;StaticForcingP&lt;float&gt;&gt;(std::string input, StaticForcingP&lt;float&gt; outinfo); template [**DynForcingP&lt;float&gt;**](structDynForcingP.md) readfileinfo&lt;DynForcingP&lt;float&gt;&gt;(std::string input, DynForcingP&lt;float&gt; outinfo); template deformmap&lt;float&gt; readfileinfo&lt;deformmap&lt;float&gt;&gt;(std::string input, deformmap&lt;float&gt; outinfo); 


        

<hr>



### function readfileinfo&lt; DynForcingP&lt; float &gt; &gt; 

```C++
template DynForcingP < float > readfileinfo< DynForcingP< float > > (
    std::string input,
    DynForcingP < float > outinfo
) 
```




<hr>



### function readfileinfo&lt; StaticForcingP&lt; float &gt; &gt; 

```C++
template StaticForcingP < float > readfileinfo< StaticForcingP< float > > (
    std::string input,
    StaticForcingP < float > outinfo
) 
```




<hr>



### function readfileinfo&lt; deformmap&lt; float &gt; &gt; 

```C++
template deformmap < float > readfileinfo< deformmap< float > > (
    std::string input,
    deformmap < float > outinfo
) 
```




<hr>



### function readfileinfo&lt; forcingmap &gt; 

```C++
template forcingmap readfileinfo< forcingmap > (
    std::string input,
    forcingmap outinfo
) 
```




<hr>



### function readfileinfo&lt; inputmap &gt; 

```C++
template inputmap readfileinfo< inputmap > (
    std::string input,
    inputmap outinfo
) 
```




<hr>



### function readparambool 

```C++
bool readparambool (
    std::string paramstr,
    bool defaultval
) 
```




<hr>



### function readparamstr 

```C++
Param readparamstr (
    std::string line,
    Param param
) 
```



Read BG\_param.txt line and convert parameter to the righ parameter in the class retrun an updated [**Param**](classParam.md) class 


        

<hr>



### function readparamstr 

```C++
template<class T>
Forcing < T > readparamstr (
    std::string line,
    Forcing < T > forcing
) 
```



Read BG\_param.txt line and convert parameter to the righ parameter in the class return an updated [**Param**](classParam.md) class 


        

<hr>



### function setendtime 

```C++
double setendtime (
    Param XParam,
    Forcing < float > XForcing
) 
```



Calculate/modify endtime based on maximum time in forcing 


        

<hr>



### function split 

```C++
void split (
    const std::string & s,
    char delim,
    std::vector< std::string > & elems
) 
```



split string based in character 


        

<hr>



### function split 

```C++
std::vector< std::string > split (
    const std::string & s,
    char delim
) 
```



split string based in character 


        

<hr>



### function split 

```C++
std::vector< std::string > split (
    const std::string s,
    const std::string delim
) 
```




<hr>



### function split\_full 

```C++
void split_full (
    const std::string & s,
    char delim,
    std::vector< std::string > & elems
) 
```



split string based in character, conserving empty item 


        

<hr>



### function split\_full 

```C++
std::vector< std::string > split_full (
    const std::string & s,
    char delim
) 
```



split string based in character, conserving empty items 


        

<hr>



### function trim 

```C++
std::string trim (
    const std::string & str,
    const std::string & whitespace
) 
```



remove leading and trailing space in a string 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/ReadInput.cu`


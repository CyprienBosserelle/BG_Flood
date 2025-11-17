

# File ReadInput.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**ReadInput.cu**](_read_input_8cu.md)

[Go to the source code of this file](_read_input_8cu_source.md)



* `#include "ReadInput.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  void | [**InitialiseToutput**](#function-initialisetoutput) ([**T\_output**](class_t__output.md) & Toutput\_loc, [**Param**](class_param.md) XParam) <br>_Initialise the Toutput structure with output times. This function reads the output times from a specified input string and ensures that the times are within the simulation time range. If no valid times are provided, it defaults to using the total time and end time._  |
|  std::vector&lt; double &gt; | [**ReadTRangestr**](#function-readtrangestr) (std::vector&lt; std::string &gt; timestr, double start, double end, std::string reftime) <br>_Read and interpret a time range string, converting it to a vector of doubles within specified bounds. This function interprets a time range string formatted as "t\_init:t\_step:t\_end", where each component can be a specific time value or a keyword representing the start or end of the overall time range. It converts the range into a vector of double values representing discrete time steps, ensuring all values fall within the provided start and end bounds._  |
|  std::vector&lt; std::string &gt; | [**ReadToutSTR**](#function-readtoutstr) (std::string paramstr) <br>_Split a comma-separated parameter string into a vector of strings._  |
|  std::vector&lt; double &gt; | [**ReadToutput**](#function-readtoutput) (std::vector&lt; std::string &gt; paramstr, [**Param**](class_param.md) XParam) <br>_Read and interpret output time specifications from a vector of parameter strings. This function processes a vector of parameter strings that specify output times,_  _which can include individual time values or ranges defined by a start, step, and end. It converts these specifications into a vector of double values representing the output times, ensuring all times fall within the simulation's total time and end time._ |
|  double | [**ReadTvalstr**](#function-readtvalstr) (std::string timestr, double start, double end, std::string reftime) <br>_Read and interpret a time value string, converting it to a double within specified bounds. This function interprets a time value string, which can represent specific keywords ("start", "end"), relative times, or absolute date-time strings. It converts the_  _string to a double value representing time, ensuring it falls within the provided start and end bounds._ |
|  void | [**Readparamfile**](#function-readparamfile) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing, std::string Paramfile) <br>_Read and parse the parameter file. Opens the specified parameter file (default: BG\_param.txt file), reads its contents, and updates the provided parameter structures:_ [_**Param**_](class_param.md) _class (XParam) and_[_**Forcing**_](struct_forcing.md) _class (XForcing)._ |
|  std::size\_t | [**case\_insensitive\_compare**](#function-case_insensitive_compare) (std::string s1, std::string s2) <br>_Perform a non case-insensitive comparison between two strings or a string and a vector of strings. This function converts both strings to lowercase and compares them. If a vector of strings is provided, it compares the first string against each string in the vector._  |
|  std::size\_t | [**case\_insensitive\_compare**](#function-case_insensitive_compare) (std::string s1, std::vector&lt; std::string &gt; vecstr) <br>_Perform a non case-insensitive comparison between a string and a vector of strings. This function converts the first string to lowercase and compares it against each string in the vector._  |
|  void | [**checkparamsanity**](#function-checkparamsanity) ([**Param**](class_param.md) & XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; & XForcing) <br>_Check and adjust the sanity of model parameters and forcing data. This function checks the sanity of the model parameters and forcing data. It adjusts parameters as needed, ensuring they are within acceptable ranges and consistent with each other._  |
|  std::string | [**findparameter**](#function-findparameter) (std::vector&lt; std::string &gt; parameterstr, std::string line) <br>_Find and extract the value of a specified parameter from a configuration line. This function searches for a specified parameter in a given line of text, and extracts its associated value if found. It handles comments and whitespace appropriately._  |
|  std::string | [**findparameter**](#function-findparameter) (std::string parameterstr, std::string line) <br>_Find and extract the value of a specified parameter from a configuration line. This function searches for a specified parameter in a given line of text, and extracts its associated value if found. It handles comments and whitespace appropriately._  |
|  double | [**readApproxtimestr**](#function-readapproxtimestr) (std::string input) <br>_Convert an approximate time string to a double value in seconds. This function interprets a time string that may include a numeric value followed by a time unit (e.g., "seconds", "minutes", "hours", "days", "months", "years") and converts it to a double value representing the equivalent time in seconds. If the unit is not recognized, it defaults to seconds._  |
|  [**bndsegment**](classbndsegment.md) | [**readbndline**](#function-readbndline) (std::string parametervalue) <br>_Read boundary segment information from a parameter value string. This function parses a parameter value string to extract boundary segment information, including type, input file, and polygon file. It also reads file information and sets the expected type of input based on the file extension._  |
|  [**bndsegment**](classbndsegment.md) | [**readbndlineside**](#function-readbndlineside) (std::string parametervalue, std::string side) <br>_Read boundary segment information from a parameter value string for a specific side. This function parses a parameter value string to extract boundary segment information, including type, input file, and polygon file for a specified side. It also reads file information and sets the expected type of input based on the file extension._  |
|  T | [**readfileinfo**](#function-readfileinfo) (std::string input, T outinfo) <br>_Parse a parameter string and update the parameter structure. Parses a line from the parameter file and updates the given parameter structure. Convert file name into name and extension. This is used for various input classes template inputmap readfileinfo&lt;inputmap&gt;(std::string input, inputmap outinfo); template forcingmap readfileinfo&lt;forcingmap&gt;(std::string input, forcingmap outinfo); template StaticForcingP&lt;float&gt; readfileinfo&lt;StaticForcingP&lt;float&gt;&gt;(std::string input, StaticForcingP&lt;float&gt; outinfo); template_ [_**DynForcingP&lt;float&gt;**_](struct_dyn_forcing_p.md) _readfileinfo&lt;_[_**DynForcingP&lt;float&gt;**_](struct_dyn_forcing_p.md) _&gt;(std::string input,_[_**DynForcingP&lt;float&gt;**_](struct_dyn_forcing_p.md) _outinfo); template deformmap&lt;float&gt; readfileinfo&lt;deformmap&lt;float&gt;&gt;(std::string input, deformmap&lt;float&gt; outinfo);._ |
|  template [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; | [**readfileinfo&lt; DynForcingP&lt; float &gt; &gt;**](#function-readfileinfo-dynforcingp-float) (std::string input, [**DynForcingP**](struct_dyn_forcing_p.md)&lt; float &gt; outinfo) <br> |
|  template [**StaticForcingP**](struct_static_forcing_p.md)&lt; float &gt; | [**readfileinfo&lt; StaticForcingP&lt; float &gt; &gt;**](#function-readfileinfo-staticforcingp-float) (std::string input, [**StaticForcingP**](struct_static_forcing_p.md)&lt; float &gt; outinfo) <br> |
|  template [**deformmap**](classdeformmap.md)&lt; float &gt; | [**readfileinfo&lt; deformmap&lt; float &gt; &gt;**](#function-readfileinfo-deformmap-float) (std::string input, [**deformmap**](classdeformmap.md)&lt; float &gt; outinfo) <br> |
|  template [**forcingmap**](classforcingmap.md) | [**readfileinfo&lt; forcingmap &gt;**](#function-readfileinfo-forcingmap) (std::string input, [**forcingmap**](classforcingmap.md) outinfo) <br> |
|  template [**inputmap**](classinputmap.md) | [**readfileinfo&lt; inputmap &gt;**](#function-readfileinfo-inputmap) (std::string input, [**inputmap**](classinputmap.md) outinfo) <br> |
|  bool | [**readparambool**](#function-readparambool) (std::string paramstr, bool defaultval) <br>_Convert a parameter string to a boolean value, with a default fallback. This function interprets a parameter string as a boolean value, returning true for recognized true values and false for recognized false values. If the string does not match any known values, it returns a specified default value._  __ |
|  [**Param**](class_param.md) | [**readparamstr**](#function-readparamstr) (std::string line, [**Param**](class_param.md) param) <br>_Parse a parameter string and update the parameter structure._  |
|  [**Forcing**](struct_forcing.md)&lt; T &gt; | [**readparamstr**](#function-readparamstr) (std::string line, [**Forcing**](struct_forcing.md)&lt; T &gt; forcing) <br>_Parse a parameter string and update the forcing structure. Parses a line from the parameter file and updates the given forcing structure. Read BG\_param.txt line and convert parameter to the right parameter in the class_  _Return an updated_[_**Forcing**_](struct_forcing.md) _class._ |
|  double | [**setendtime**](#function-setendtime) ([**Param**](class_param.md) XParam, [**Forcing**](struct_forcing.md)&lt; float &gt; XForcing) <br>_Adjust the simulation "endtime" based on maximum time in forcings. This function checks the end times of boundary forcing data and adjusts the simulation end time if any boundary forcing ends before the specified end time. A warning is logged if the end time is reduced._  |
|  void | [**split**](#function-split) (const std::string & s, char delim, std::vector&lt; std::string &gt; & elems) <br>_Split a string into tokens based on a specified delimiter, skipping empty tokens. This function takes a string and splits it into a vector of substrings using the specified delimiter. Empty tokens resulting from consecutive delimiters are skipped._  |
|  std::vector&lt; std::string &gt; | [**split**](#function-split) (const std::string & s, char delim) <br>_Split a string into tokens based on a specified character delimiter, skipping empty tokens. This function takes a string and splits it into a vector of substrings using the specified delimiter. Empty tokens resulting from consecutive delimiters are skipped._  |
|  std::vector&lt; std::string &gt; | [**split**](#function-split) (const std::string s, const std::string delim) <br>_Split a string into tokens based on a specified substring delimiter. This function takes a string and splits it into a vector of substrings using the specified substring delimiter._  |
|  void | [**split\_full**](#function-split_full) (const std::string & s, char delim, std::vector&lt; std::string &gt; & elems) <br>_Split a string into tokens based on a specified delimiter, preserving empty tokens. This function takes a string and splits it into a vector of substrings using the specified delimiter. Unlike the standard split function, this version preserves empty tokens that result from consecutive delimiters._  __ |
|  std::vector&lt; std::string &gt; | [**split\_full**](#function-split_full) (const std::string & s, char delim) <br>_Split a string into tokens based on a specified character delimiter, preserving empty tokens. This function takes a string and splits it into a vector of substrings using the specified delimiter. Unlike the standard split function, this version preserves empty tokens that result from consecutive delimiters._  __ |
|  std::string | [**trim**](#function-trim) (const std::string & str, const std::string & whitespace) <br>_Trim leading and trailing whitespace from a string. This function removes all leading and trailing characters from the input string that are present in the specified whitespace string._  |




























## Public Functions Documentation




### function InitialiseToutput 

_Initialise the Toutput structure with output times. This function reads the output times from a specified input string and ensures that the times are within the simulation time range. If no valid times are provided, it defaults to using the total time and end time._ 
```C++
void InitialiseToutput (
    T_output & Toutput_loc,
    Param XParam
) 
```





**Parameters:**


* `Toutput_loc` Reference to the [**T\_output**](class_t__output.md) structure to be initialised. 
* `XParam` The [**Param**](class_param.md) structure containing simulation parameters. 




        

<hr>



### function ReadTRangestr 

_Read and interpret a time range string, converting it to a vector of doubles within specified bounds. This function interprets a time range string formatted as "t\_init:t\_step:t\_end", where each component can be a specific time value or a keyword representing the start or end of the overall time range. It converts the range into a vector of double values representing discrete time steps, ensuring all values fall within the provided start and end bounds._ 
```C++
std::vector< double > ReadTRangestr (
    std::vector< std::string > timestr,
    double start,
    double end,
    std::string reftime
) 
```





**Parameters:**


* `timestr` A vector of strings representing the time range components: [t\_init, t\_step, t\_end]. 
* `start` The start time bound. 
* `end` The end time bound. 
* `reftime` The reference time for interpreting absolute date-time strings. 



**Returns:**

A vector of double values representing the interpreted time steps within the specified range. 
 





        

<hr>



### function ReadToutSTR 

_Split a comma-separated parameter string into a vector of strings._ 
```C++
std::vector< std::string > ReadToutSTR (
    std::string paramstr
) 
```



This function takes a parameter string containing values separated by commas and splits it into a vector of individual strings. 

**Parameters:**


* `paramstr` The parameter string to be split. 



**Returns:**

A vector of strings obtained by splitting the input string at commas. 





        

<hr>



### function ReadToutput 

_Read and interpret output time specifications from a vector of parameter strings. This function processes a vector of parameter strings that specify output times,_  _which can include individual time values or ranges defined by a start, step, and end. It converts these specifications into a vector of double values representing the output times, ensuring all times fall within the simulation's total time and end time._
```C++
std::vector< double > ReadToutput (
    std::vector< std::string > paramstr,
    Param XParam
) 
```





**Parameters:**


* `paramstr` A vector of strings specifying output times or ranges. 
* `XParam` The [**Param**](class_param.md) structure containing simulation parameters, including total time and end time. 



**Returns:**

A vector of double values representing the interpreted output times. 





        

<hr>



### function ReadTvalstr 

_Read and interpret a time value string, converting it to a double within specified bounds. This function interprets a time value string, which can represent specific keywords ("start", "end"), relative times, or absolute date-time strings. It converts the_  _string to a double value representing time, ensuring it falls within the provided start and end bounds._
```C++
double ReadTvalstr (
    std::string timestr,
    double start,
    double end,
    std::string reftime
) 
```





**Parameters:**


* `timestr` The time value string to be interpreted. 
* `start` The start time bound. 
* `end` The end time bound. 
* `reftime` The reference time for interpreting absolute date-time strings. 



**Returns:**

A double value representing the interpreted time, constrained within the start and end bounds. 





        

<hr>



### function Readparamfile 

_Read and parse the parameter file. Opens the specified parameter file (default: BG\_param.txt file), reads its contents, and updates the provided parameter structures:_ [_**Param**_](class_param.md) _class (XParam) and_[_**Forcing**_](struct_forcing.md) _class (XForcing)._
```C++
void Readparamfile (
    Param & XParam,
    Forcing < float > & XForcing,
    std::string Paramfile
) 
```





**Parameters:**


* `XParam` Reference to the parameter structure to be updated 
* `XForcing` Reference to the forcing structure to be updated 
* `Paramfile` Name of the parameter file to read 




        

<hr>



### function case\_insensitive\_compare 

_Perform a non case-insensitive comparison between two strings or a string and a vector of strings. This function converts both strings to lowercase and compares them. If a vector of strings is provided, it compares the first string against each string in the vector._ 
```C++
std::size_t case_insensitive_compare (
    std::string s1,
    std::string s2
) 
```





**Parameters:**


* `s1` The first string to compare. 
* `s2` The second string to compare, or a vector of strings to compare against 




        

<hr>



### function case\_insensitive\_compare 

_Perform a non case-insensitive comparison between a string and a vector of strings. This function converts the first string to lowercase and compares it against each string in the vector._ 
```C++
std::size_t case_insensitive_compare (
    std::string s1,
    std::vector< std::string > vecstr
) 
```





**Parameters:**


* `s1` The first string to compare. 
* `vecstr` The vector of strings to compare against. 




        

<hr>



### function checkparamsanity 

_Check and adjust the sanity of model parameters and forcing data. This function checks the sanity of the model parameters and forcing data. It adjusts parameters as needed, ensuring they are within acceptable ranges and consistent with each other._ 
```C++
void checkparamsanity (
    Param & XParam,
    Forcing < float > & XForcing
) 
```





**Parameters:**


* `XParam` Reference to the model parameters structure to be checked and adjusted. 
* `XForcing` Reference to the forcing data structure to be checked and adjusted. 




        

<hr>



### function findparameter 

_Find and extract the value of a specified parameter from a configuration line. This function searches for a specified parameter in a given line of text, and extracts its associated value if found. It handles comments and whitespace appropriately._ 
```C++
std::string findparameter (
    std::vector< std::string > parameterstr,
    std::string line
) 
```





**Parameters:**


* `parameterstr` The parameter name to search for. 
* `line` The line of text to search within. 



**Returns:**

The extracted parameter value as a string, or an empty string if the parameter is not found. 





        

<hr>



### function findparameter 

_Find and extract the value of a specified parameter from a configuration line. This function searches for a specified parameter in a given line of text, and extracts its associated value if found. It handles comments and whitespace appropriately._ 
```C++
std::string findparameter (
    std::string parameterstr,
    std::string line
) 
```





**Parameters:**


* `parameterstr` The parameter name to search for. 
* `line` The line of text to search within. 



**Returns:**

The extracted parameter value as a string, or an empty string if the parameter is not found. 





        

<hr>



### function readApproxtimestr 

_Convert an approximate time string to a double value in seconds. This function interprets a time string that may include a numeric value followed by a time unit (e.g., "seconds", "minutes", "hours", "days", "months", "years") and converts it to a double value representing the equivalent time in seconds. If the unit is not recognized, it defaults to seconds._ 
```C++
double readApproxtimestr (
    std::string input
) 
```





**Parameters:**


* `input` The approximate time string to be converted (e.g., "10 minutes", "2.5 hours"). 



**Returns:**

A double value representing the time in seconds. 





        

<hr>



### function readbndline 

_Read boundary segment information from a parameter value string. This function parses a parameter value string to extract boundary segment information, including type, input file, and polygon file. It also reads file information and sets the expected type of input based on the file extension._ 
```C++
bndsegment readbndline (
    std::string parametervalue
) 
```





**Parameters:**


* `parametervalue` The parameter value string containing boundary segment information. 




        

<hr>



### function readbndlineside 

_Read boundary segment information from a parameter value string for a specific side. This function parses a parameter value string to extract boundary segment information, including type, input file, and polygon file for a specified side. It also reads file information and sets the expected type of input based on the file extension._ 
```C++
bndsegment readbndlineside (
    std::string parametervalue,
    std::string side
) 
```





**Parameters:**


* `parametervalue` The parameter value string containing boundary segment information. 
* `side` The side (e.g., "left", "right", "top", "bot") for which the boundary segment is defined. 




        

<hr>



### function readfileinfo 

_Parse a parameter string and update the parameter structure. Parses a line from the parameter file and updates the given parameter structure. Convert file name into name and extension. This is used for various input classes template inputmap readfileinfo&lt;inputmap&gt;(std::string input, inputmap outinfo); template forcingmap readfileinfo&lt;forcingmap&gt;(std::string input, forcingmap outinfo); template StaticForcingP&lt;float&gt; readfileinfo&lt;StaticForcingP&lt;float&gt;&gt;(std::string input, StaticForcingP&lt;float&gt; outinfo); template_ [_**DynForcingP&lt;float&gt;**_](struct_dyn_forcing_p.md) _readfileinfo&lt;_[_**DynForcingP&lt;float&gt;**_](struct_dyn_forcing_p.md) _&gt;(std::string input,_[_**DynForcingP&lt;float&gt;**_](struct_dyn_forcing_p.md) _outinfo); template deformmap&lt;float&gt; readfileinfo&lt;deformmap&lt;float&gt;&gt;(std::string input, deformmap&lt;float&gt; outinfo);._
```C++
template<class T>
T readfileinfo (
    std::string input,
    T outinfo
) 
```





**Parameters:**


* `line` Input line from parameter file 
* `param` Parameter structure to update 



**Returns:**

Updated parameter structure


convert file name into name and extension This is used for various input classes


template inputmap readfileinfo&lt;inputmap&gt;(std::string input, inputmap outinfo); template forcingmap readfileinfo&lt;forcingmap&gt;(std::string input, forcingmap outinfo); template StaticForcingP&lt;float&gt; readfileinfo&lt;StaticForcingP&lt;float&gt;&gt;(std::string input, StaticForcingP&lt;float&gt; outinfo); template [**DynForcingP&lt;float&gt;**](struct_dyn_forcing_p.md) readfileinfo&lt;[**DynForcingP&lt;float&gt;**](struct_dyn_forcing_p.md)&gt;(std::string input, [**DynForcingP&lt;float&gt;**](struct_dyn_forcing_p.md) outinfo); template deformmap&lt;float&gt; readfileinfo&lt;deformmap&lt;float&gt;&gt;(std::string input, deformmap&lt;float&gt; outinfo); 


        

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

_Convert a parameter string to a boolean value, with a default fallback. This function interprets a parameter string as a boolean value, returning true for recognized true values and false for recognized false values. If the string does not match any known values, it returns a specified default value._  __
```C++
bool readparambool (
    std::string paramstr,
    bool defaultval
) 
```





**Parameters:**


* `paramstr` The parameter string to be interpreted. 
* `defaultval` The default boolean value to return if the string does not match known values 




        

<hr>



### function readparamstr 

_Parse a parameter string and update the parameter structure._ 
```C++
Param readparamstr (
    std::string line,
    Param param
) 
```



Parses a line from the parameter file and updates the given parameter structure. Read BG\_param.txt line and convert parameter to the right parameter in the class Return an updated [**Param**](class_param.md) class




**Parameters:**


* `line` Input line from parameter file 
* `param` Parameter structure to update 



**Returns:**

Updated parameter structure


Read BG\_param.txt line and convert parameter to the righ parameter in the class return an updated [**Param**](class_param.md) class 


        

<hr>



### function readparamstr 

_Parse a parameter string and update the forcing structure. Parses a line from the parameter file and updates the given forcing structure. Read BG\_param.txt line and convert parameter to the right parameter in the class_  _Return an updated_[_**Forcing**_](struct_forcing.md) _class._
```C++
template<class T>
Forcing < T > readparamstr (
    std::string line,
    Forcing < T > forcing
) 
```





**Parameters:**


* `line` Input line from parameter file 
* `forcing` [**Forcing**](struct_forcing.md) structure to update 



**Returns:**

Updated forcing structure


Read BG\_param.txt line and convert parameter to the righ parameter in the class return an updated [**Param**](class_param.md) class 


        

<hr>



### function setendtime 

_Adjust the simulation "endtime" based on maximum time in forcings. This function checks the end times of boundary forcing data and adjusts the simulation end time if any boundary forcing ends before the specified end time. A warning is logged if the end time is reduced._ 
```C++
double setendtime (
    Param XParam,
    Forcing < float > XForcing
) 
```





**Parameters:**


* `XParam` The [**Param**](class_param.md) structure containing simulation parameters, including the initial end time. 
* `XForcing` The [**Forcing**](struct_forcing.md) structure containing boundary forcing data. 




        

<hr>



### function split 

_Split a string into tokens based on a specified delimiter, skipping empty tokens. This function takes a string and splits it into a vector of substrings using the specified delimiter. Empty tokens resulting from consecutive delimiters are skipped._ 
```C++
void split (
    const std::string & s,
    char delim,
    std::vector< std::string > & elems
) 
```





**Parameters:**


* `s` The input string to be split. 
* `delim` The character used as the delimiter for splitting the string. 
* `elems` A reference to a vector where the resulting substrings will be stored. 




        

<hr>



### function split 

_Split a string into tokens based on a specified character delimiter, skipping empty tokens. This function takes a string and splits it into a vector of substrings using the specified delimiter. Empty tokens resulting from consecutive delimiters are skipped._ 
```C++
std::vector< std::string > split (
    const std::string & s,
    char delim
) 
```





**Parameters:**


* `s` The input string to be split. 
* `delim` The character used as the delimiter for splitting the string. 



**Returns:**

A vector containing the resulting substrings. 





        

<hr>



### function split 

_Split a string into tokens based on a specified substring delimiter. This function takes a string and splits it into a vector of substrings using the specified substring delimiter._ 
```C++
std::vector< std::string > split (
    const std::string s,
    const std::string delim
) 
```





**Parameters:**


* `s` The input string to be split. 
* `delim` The substring used as the delimiter for splitting the string. 



**Returns:**

A vector containing the resulting substrings. 





        

<hr>



### function split\_full 

_Split a string into tokens based on a specified delimiter, preserving empty tokens. This function takes a string and splits it into a vector of substrings using the specified delimiter. Unlike the standard split function, this version preserves empty tokens that result from consecutive delimiters._  __
```C++
void split_full (
    const std::string & s,
    char delim,
    std::vector< std::string > & elems
) 
```





**Parameters:**


* `s` The input string to be split. 
* `delim` The character used as the delimiter for splitting the string. 
* `elems` A reference to a vector where the resulting substrings will be stored.

split string based in character, conserving empty item 


        

<hr>



### function split\_full 

_Split a string into tokens based on a specified character delimiter, preserving empty tokens. This function takes a string and splits it into a vector of substrings using the specified delimiter. Unlike the standard split function, this version preserves empty tokens that result from consecutive delimiters._  __
```C++
std::vector< std::string > split_full (
    const std::string & s,
    char delim
) 
```





**Parameters:**


* `s` The input string to be split. 
* `delim` The character used as the delimiter for splitting the string.

split string based in character, conserving empty items 


        

<hr>



### function trim 

_Trim leading and trailing whitespace from a string. This function removes all leading and trailing characters from the input string that are present in the specified whitespace string._ 
```C++
std::string trim (
    const std::string & str,
    const std::string & whitespace
) 
```





**Parameters:**


* `str` The input string to be trimmed. 
* `whitespace` A string containing all characters considered as whitespace.

remove leading and trailing space in a string 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/ReadInput.cu`


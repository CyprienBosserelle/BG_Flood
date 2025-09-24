# Install BG_Flood

## Windows 10 - 11

On windows OS you should be able to use the binaries/executable we make available in each release.
simply download and unzip the file in a suitable directory and either add the folder to your PATH or move the dll and .exe around where you what to run. 

### Build from source
To build BG_Flood from source on Windows you will need to have pre install:
* Visual Studio Community with C++ component installed
* Compatible Cuda toolkit
* Downloaded/cloned/forked source of the repo
* Netcdf developer install (i.e. netcdf.h and netcdf.lib)

#### Setup Visual Studio
1. start a new empty project
1. add CUDA build dependencies to the project
1. add NetCDF folder(s) to the include and library directories in the project properties
1. add "netcdf.lib" to the input (Properties -> Linker -> Input)
1. switch the "Generate Relocatable device code" to Yes (Properties -> CUDA C/C++ -> Common)
1. disable deprecation add _CRT_SECURE_NO_WARNINGS to preprocessor definition (Properties -> C/C++ -> Preprocessor)

## Linux OS

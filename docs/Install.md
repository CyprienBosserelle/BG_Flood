# Installation of BG_Flood

!!! warning
     BG_Flood has been written in CUDA language, C++ based language created by NVIDIA to interact directly with their GPUs.
     Even in the code can run on CPU, it is design to run on NVIDIA GPUs.

The code has two main dependencies:
- CUDA
- netcdf


## Windows 10 - 11

On windows OS you should be able to use the binaries/executable we make available in each [release](https://github.com/CyprienBosserelle/BG_Flood/releases/latest).
Simply download and unzip the file in a suitable directory and either add the folder to your PATH or move the dll and .exe around where you want to run. 

### Build from source
To build BG_Flood from source on Windows you will need to have pre-install:

- Visual Studio Community with C++ component installed
- Compatible (Cuda toolkit)[https://developer.nvidia.com/cuda-toolkit]
- Downloaded/cloned/forked source of the repo
- Netcdf developer install (i.e. netcdf.h and netcdf.lib)

#### Setup Visual Studio
1. start a new empty project
1. add CUDA build dependencies to the project
1. add NetCDF folder(s) to the include and library directories in the project properties
1. add "netcdf.lib" to the input (Properties -> Linker -> Input)
1. switch the "Generate Relocatable device code" to Yes (Properties -> CUDA C/C++ -> Common)
1. disable deprecation add _CRT_SECURE_NO_WARNINGS to preprocessor definition (Properties -> C/C++ -> Preprocessor)


## Linux OS

Make sure you have latest [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), g++ and NetCDF libraries installed.

```{bash}
sudo apt-get install nvidia-cuda-dev
sudo apt-get install g++
sudo apt-get install libnetcdf-dev
```

Also make sure the GPU driver being used is the Nvidia driver!

Do a quick comand line test to see if nvcc (CUDA compiler) is available from here.
If not, You may need to modify the cuda path in the makefile (line 155)
```{.bash}
NVCC          := nvcc -ccbin $(HOST_COMPILER)
```


The code can compile for multiple GPU architecture but later compiler do not support old GPU (2.0 is no longer supported)
Remove unsupported architecture in line 213 of the makefile

Then just type 
```{.bash} 
make```

Many warning will show up but that is OK.




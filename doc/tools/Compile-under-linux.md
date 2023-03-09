@page Compile-under-linux Compilation on Linux


Make sure you have latest CUDA Toolkit, g++ and NetCDF libraries installed.

```{.bash}
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



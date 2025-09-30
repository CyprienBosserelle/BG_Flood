# Installation of BG_Flood

!!! warning
     BG_Flood has been written in CUDA language, C++ based language created by NVIDIA to interact directly with their GPUs.
     Even if the code can run on CPU (for testing purposes for example), it will be performant on NVIDIA GPUs. The best performances
     are observed on large NVIDIA GPUs on supercomputers.

The code has only two dependencies:

- CUDA
- netcdf


## :fontawesome-brands-windows: Windows 10 - 11

On windows OS you should be able to use the binaries/executable we make available in each [release](https://github.com/CyprienBosserelle/BG_Flood/releases/latest).
Simply download and unzip the file in a suitable directory and either add the folder to your PATH or move the dlls and .exe around where you want to run. 

### Build from source
To build BG_Flood from source on Windows you will need to have pre-install:

- Visual Studio Community with C++ component installed
- Compatible (Cuda toolkit)[https://developer.nvidia.com/cuda-toolkit]
- Downloaded/cloned/forked source of the repo
- Netcdf developer install (i.e. netcdf.h and netcdf.lib)


!!! Tip "Setup on Visual Studio"
    
    - start a new empty project
    - add CUDA build dependencies to the project
    - add NetCDF folder(s) to the include and library directories in the project properties
    - add "netcdf.lib" to the input (Properties -> Linker -> Input)
    - switch the "Generate Relocatable device code" to Yes (Properties -> CUDA C/C++ -> Common)
    - disable deprecation add _CRT_SECURE_NO_WARNINGS to preprocessor definition (Properties -> C/C++ -> Preprocessor)


## :simple-linux: Linux

Make sure you have latest [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), g++ and NetCDF libraries installed.

```{bash}
sudo apt-get install nvidia-cuda-dev
sudo apt-get install g++
sudo apt-get install libnetcdf-dev
```

!!! note
    Make sure the GPU driver being used is the Nvidia driver!

Do a quick comand line test to see if nvcc (CUDA compiler) is available from here.

If not, you may need to modify the cuda path in the makefile (line 155) :
```{bash}
NVCC          := nvcc -ccbin $(HOST_COMPILER)
```


!!! warning
    The code can compile for multiple GPU architecture but later compiler do not support old GPU (2.0 is no longer supported).
    If needed, remove unsupported architecture in line 213 of the makefile.

Then just type 
```{bash} 
make
```

!!! success
    Many warning will show up but that is OK...

## :material-server: Supercomputers

The code can be run on local machines with NVIDIA GPU but it will get better performance by running on large GPU.Below are example of installation and running procedures on HPC the develloper used.

### ESNZ supercomputer: Cascade
This machine is set-up using stack and all tools need to be install through it before compiling/running the code.
The PBS job manager is used.

#### Compiling the code
```{bash}
. $(ls /opt/niwa/profile/spack_* | tail -1)
spack load netcdf-c@4.9.2%gcc@11.5.0 cuda@12.8.0
nclib=`nc-config --libdir`
export LD_LIBRARY_PATH="${nclib}:$LD_LIBRARY_PATH"

cd BG_Flood_Folder

make -j 10

```
!!! note
    Spack load doesn't set LD_LIBRARY_PATH so running
    executable won't find libnetcdf.  Also it doesn't set
    LDFLAGS=-Wl,-rpath (and Makefile doesn't honour LDFLAGS
    anyway), so libnetcdf path isn't linked in.  So hack this
    in for now.

#### Running the code
```{bash}

#!/bin/bash

#PBS -N *my_pbs_job_name*
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb
#PBS -l walltime=72:00:00
#PBS -q a100q=""
#PBS -A *myaccount*
#PBS -W umask=027

# Change to running directory if required
cd *my_case_dir*

# Launch of the solver
BG_Flood BG_param.txt

```


### NESI
!!! danger "Depreciated"
    NESI supercomputer has now been closed and replaced by REANZ new generation of machines.

The code is actually running on [New Zealand eScience Infrastructure (NeSI)](https://www.nesi.org.nz).
This national center uses a module systems associated to the slurm job manager.

#### Compiling the code
The Code needs to be compile on the machine, using the sources from the github repository.
Due to the code dependency to CUDA and netCDF, two modules need to be loaded:

- On Maui:
```{bash} 
module load CUDA\11.4.1
module load netCDF-C++4/4.3.0-GCC-7.1.0
```

- On Mahuika:
```{bash}
module load CUDA/11.4.1
module load netCDF-C++4/4.3.1-gimpi-2020a
```

#### Running the code

- Example of a slurm file on Maui:
```{bash}
#!/bin/bash
#SBATCH --job-name=MY-TEST-NAME
#SBATCH --time=8:00:00
#SBATCH --account=MY-NESI-ACCOUNT
#SBATCH --partition=nesi_gpu
#SBATCH --gres=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5GB

#Running directory (to be completed)
BGFLOOD=/nesi/project/XXXXXXXXXXXXXXX

cd ${BGFLOOD}

module load CUDA/11.4.1
module load netCDF-C++4/4.3.0-GCC-7.1.0

# Launching the executable
srun ./BG_Flood_Maui

echo "output_file = Output/${testname}/BGoutput-${reftime}.nc"

echo "end of setup_run_BG.sh"
```


- Example of a slurm file on Mahuika:
```{bash}
#!/bin/bash
#SBATCH --job-name=MY-TEST-NAME
#SBATCH --time=05:00:00
#SBATCH --account=MY-NESI-ACCOUNT
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB

#Running directory (to be completed)
BGFLOOD=/nesi/project/XXXXXXXXXXXXXXX

cd ${BGFLOOD}

#module load netCDF-C++4/4.3.0-gimkl-2017a
module load netCDF-C++4/4.3.1-gimpi-2020a
module load CUDA/11.4.1

# Launching the executable
srun ./BG_Flood_Mahuika

echo "output_file = Output/${testname}/BGoutput-${reftime}.nc"

echo "end of setup_run_BG.sh"
```



@page Use-On-SuperComputer Use on SuperComputer

The code can be run on local machines with NVIDIA GPU but it will get better performance by running on large GPU.

The code is actually running on [New Zealand eScience Infrastructure (NeSI)](https://www.nesi.org.nz).

## Compiling the code
The Code needs to be compile on the machine, using the sources grom the github repository.
Due to the code dependency to CUDA and netCDF, two modules need to be loaded:
* On Maui:
 ```{bash} 
module load CUDA\11.4.1
module load netCDF-C++4/4.3.0-GCC-7.1.0```
* On Mahuika:
 ```{bash}
module load CUDA/11.4.1
module load netCDF-C++4/4.3.1-gimpi-2020a```

## Running the code
* Example of a slurm file on Maui:
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


* Example of a slurm file on Mahuika:
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



@page tutorialRiver River flooding tutorial

The objectif of this tutorial is to explain how to use the code BG_Flood to model river flooding.
As the code allows rain on grid, we will look at pluvial and fluvial flooding.

This testcase aims to reproduce the flooding event that occured from the 16 to the 18 of July 2021 (a 1 in 50 years flood event). 
During this event, the Buller river, passing through the town of Westport, on the West Coast of the South Island of Aotearoa New Zealand, get out of its primary bed and flooded the Westport area.

Before begining this tutoral, the user is expected to have downloaded the windows executable or compiled the sources (version > 0.8).

# Param file
The interface with BG_flood software is done using only a text file:
```{bash} 
BG_param.txt
```
(This is the name of the input parameter file by defauls, an other name can be use as first variable when lauching BG_Flood.)

This file consists in a list of key words and inputs, separated by a "=" sign. 
For a full list of the available key words and a further description of this file, please refere to the [manual]{#Manual}.

# Preparation of the topography/bathymetry (DEM: Digital Elevation Model)
The DEM, topography or bathymetry file is the only necessary file to be added to BG_param file to run the model.
 ```{text}
DEM = Wesport_DEM_8m.nc?z;
 ``` 

![bathy](./figure/Westport_topo.png)

Different formats can be used for this file ( ASCII: .asc or NetCDF .nc principally).
As Netcdf files can contain different variables, the "?" follows by the name of the variable is needed.

Without further information, the code will run will default values for physical parameters and initialisation, Neumann boundaries and no external forcing:

![shell](./figure/Shell_output.png)


A result files: "output.nc" is created (here opened with the PyNcView) 

![outputs](./figure/outputs.png)

It contains 2D fields saved regularly in time.
It details the blocs information, the time 1D variable, the xx and yy axis and the basic outputs (h,u,v,zb,zs) for a shallow water model (see manual for further description of the variables).

A log file: BG_log.txt (very similaire to the shell outputs) is also created:
![logfile](./figure/log_file.png)

# Basic fluvial flooding set-up


## River discharge
The river are, at this stage, forced by a vertical discharge on a user defined rectagular area:
```{txt} 
river = river_discharge_TeKuha2.txt,1490249,1490427,5367640,5367805;
```
where the four final numbers are: \f$ x_1, x_2, y_1, y_2 \f$, to define the area for the vertical discharge and a text file containing the time-serie of the discharge (first column: time (\f$s\f$) from reference time, second column: river discharge in \f$m^3s^{-1}\f$).
![riverfile](./figure/river_discharge.png)
This file is from an observed hydrograph, with data saved every 5min:
![logfile](./figure/TE_Kuha_hydrograph.png)

For each new river, just add the a similar river input line in the parameter file.

## Timekeeping parameters
In this code, the time is defined in second, relative to some reference or the start of the simulation by default.

The end of the simulation is prescribed in second as :
```{txt} 
endtime = 21600;
```

The time steps can't be defined by the used, it will be automatically computed as the more restrictive one in the domain by the solver, using the prescribe CFL (Current Friedrich Limiter) value, \f$ CFL=0.5000 \f$ by default.

The simulation begin, by default at \f$ t=0 (s)\f$, but this can modify using "totaltime": 
```{txt} 
totaltime = 3600;
```
to begin one hour after the reference time ( used in the forcings for example).

## Outputs
There is two types of outputs:
 - map outputs of 2D variables regularly through time
 - time-series output of basic values, at a chosen position, at each time step.

### Map outputs
 By default, there is only a map output at the begining and end of the simulation.

 The map output can be modify by:
 - defining a timestep (in s) for these outputs:
```{txt} 
outputtimestep = 3600.0;
```
- changing the set of variables in the output file (from the list given in the manual)
```{txt} 
outvars = zs,h,u,v,zb,hmax,Umax,hUmax,twet;
```
The "max" variables will be the maximum value during the whole simulation. To reset it between the outputs, see the resetmax variable.
- changing the name of the output file:
```{txt} 
outfile = Results_tuto_basicRun.nc;
```
- saving the output as float (variables are saved as short integer by default.):
```{txt} 
smallnc = 0;
```

### Time-Serie outputs
For each TS output needed, a line with the destination file and the postition is needed:

```{txt} 
TSnodesout=Offshore.txt,1482120,53814890;
```
The file contains 5 colums \f$(t, zs, h, u,v)\f$ with the value at the nearest grid point (to the position defined by the user).

## Resolution
For a first test, we will modify the resolution and set it to 40m to decrease the computational time:
```{txt} 
dx=40;
```

## Basic fluvial innundation results
This the shell output:
![shell](./figure/Shell_output2.png)
It shows that 1 river has been added to the model, and also the time progression with 5 map outputs (in addition to the initial time step).




# Completing the set-up
## Adding boundary conditions


Presentation
### Tidal forcing

## Bottom friction
Different models from bottom friction are available.
You can choose the model using the keyword:
```{txt} 
frictionmodel=1;
```
frictionmodel=1;
cfmap=z0_100423_rec3.asc;
#cf=0.01;

Attention: Extrapolation outside of the map

## Initialisation
By default, the model is initialised by a plane water surface located at \f$z=0.0\f$.

This water level can be modify, depending of the local mean sea level and the vertical projection used to create the DEM, using:
```{txt} 
zsinit=-1.39; #in metre
```
The model can also be initialised using a restart/hot start.
A file containing a least the files zb, h or zs, u and v must be provided, with the steps (and no the time value) to use for the restart.
```{txt} 
hotstartfile = output_4.nc;
hotstep=5;
```

## Model controls

## Outputs

# ... Adding the rain

## Rain forcing
=> through TS or Time-varying maps (nc files)
=> Rain only on the part of the domain we are interested in
=> AOI or rain maps with masked area


=> results


## Ground infiltration losses (Basic ILCL model)


# Refining the grid in area of interest

## More DEMS
## Map of levels of resolution
## Adding the variable resolution in the code
Show adding a second high reso DEM (ALSO add mention of possibility to use path (relative or absolute))

=> quick look at shell outputs

=> results


#
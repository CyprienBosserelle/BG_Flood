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

![riverF](./figure/TE_Kuha_hydrograph.png)

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
There is also special variables for risk assesment after inundation (Umax, hmax, hUmax, twet)
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
![shell2](./figure/Shell_output2.png)
It shows that 1 river has been added to the model, and also the time progression with 5 map outputs (in addition to the initial time step).

In the outputs, we get the different fields requested, for 6 different times.
![output2](./figure/outputs_v2.png)

The Time-Serie output is:
![TS2](./figure/offshore.png)


# Completing the set-up
## Adding boundary conditions
Boundaries condition Boundaries are refered by their position, using "top/bottom/right/left" keywords. They are associated to a boundary type ( 0:wall; 1: Neumann (Default); 2:Dirichlet (zs); 3: abs1d) and possibly a file containing a time serie. In this case, the file name is placed before the type, coma-separated. 

In this case, we will use tide boundaries at when at least a part of the boundary is open on the sea, i.e. for the top, left and right boundaries.
At the bottom, we will conserve the default value: 1.
```{txt} 
left = tide_westport.txt,2; 
right = tide_westport.txt,2; 
top = tide_westport.txt,2; 
```

In this case, as the boundaries are relatively small compared to the time wave length, we will used the same value along all the boundaries. We will then have only two columns in the file: Time and one elevation.
tide_file
![tide_file](./figure/tide_file.png)

They correspond to a classic time Serie observed offshore of the river mouth.

![Tide](./figure/tide_westport.png)

(If more values, they will be regularly spread along the boundary and the forcing will be the linear interpolation of these values). 


## Bottom friction
Different models from bottom friction are available.
By default, the model used is -1 corresponding to a Manning model.
Here, we will use the model 1 corresponding to a roughness length (see manual for more information on the Bottom friction models).
The associated field (ASC or netCDF) or value must be enter with the key word
```{txt} 
frictionmodel=1;
cf=z0_100423_rec3.asc; #cf=0.01;  #If using a uniform value
```

![Friction](./figure/Bottom_friction_zo.png)

**Warning**: The model allows a roughness heigh / manning number map smaller than the computational domain and will extrapolate outside of the map.

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
Some variables can be used to adjust the model (see Manual for more details):
- run on CPU (or choose a GPU to run on):
```{txt} 
gpudevice=0;
```
By default, the code will detect if there is a suitable GPU on the machine.
- Double precision instead of a float precision during the computation:
```{txt} 
doubleprecision = 1;
```
- Minmod limiter parameter (to tune momentum dissipation \f$\in [1,2]\f$)
```{txt} 
theta=1.3; #default value=1.3
```
- Minimum heigh to concidere a cell wet (m)
```{txt} 
eps = 0.00010000; #default=0.0001
```

## Outputs



# ... Adding the rain

The model allows rain on grid forcing to model pluvial inundations.

## Rain forcing
A rain intensity in \f$ mm.h^{-1} \f$, time and space varying can be forced in the model.

The rain can be forced with a time serie (with uniform values on the domain) or a netCDF file if a spacial file is available:
- Time serie forcing:
```{txt} 
rainfall=rain_westport.txt
```
- Spacial file forcing:
```{txt} 
rainfile=VCSN_buller_202107_dailynzcsmcov_disaggdaily_500m_nztm_clipped.nc?depth;
```
Here, we will use a time serie:
![RainTS](./figure/rain_westport.png)

If the data is given in "rain height", a post-processing to turn it in rain intensity will be needed (at least at this stage of development).

Using the rain on grid forcing will activate all the cells of the domain and will increase the computational time of the simulation. 
Part of the domain can be "de-activate" (the blocs memory will not be allocated for this area) using different methods:
 - using the x0 / xmax / y0 / ymax keyword to restrain the extend of the computational domain (but still rectangular)
  ```{txt} 
x0=1475000; #m
```
 - a manual mask with values 999 in the bathymetry will be read by the code as "non-active" area
 - masking all the bloc with all cells having an elevation superior to some value:
 ```{txt} 
mask=250; #m
```
 - using a shape file to define a "area of interest" (**Method advised**):
 ```{txt} 
AOI=Domain_buffered-sea2.gmt;
```


# Refining the grid in area of interest
The code is based on a Block-uniform quadtree mesh. Each block, actually a 16 by 16 cells, is one unit of computation in the GPU.
These blocks can have different resolutions (but resolution does not change during the computation at this stage).

The initial resolution of the grid is the resolution of the bathymetry/topographic data. To refine or coarsen the grid, you can weather use the "dx" key word and choose a new resolution for the whole domain; wether use the levels of resolution. The reference level, correponding to the bathymetry resolution will be the level 0. Levels of resolution are then defined in relation to the reference levels using positive integers to increase the resolution or refine and negative integer to coarsen the grid by a multiple of two. For a given level  \f$𝑛\f$ , the resolution  \f$𝑑𝑥_𝑛\f$
  will be:
$$𝑑𝑥_𝑛=\frac{𝑑𝑥_0}{2^𝑛}$$
 
with  \f$𝑑𝑥_0\f$ the resolution at level 0. 

When refinning using the level implementation, different key words are expected:

- Initlevel: level used to create the first mesh created by the code
- Maxlevel: maximim level of refinement (over-ruling other commands)
- Minlevel: minimum level of refinement (over-ruling other commands)

The grid can also be unregular with an adaptition of the grid to the model (variables at initialisation step or user-defined refinement map). In this case, the cells will be devided in 4 cells for refinement, or 4 cells merged in one for coarsening. The code will ensure a progressive change of resolution (no cell should have a neighbour with more than 1 level of resolution of difference.)

The different methods of refinement available in the code are called using the key word "Adaptation". The refinement can be based on a classical input variable or a variable calculated during the initialisation:

- 'Threshold': impose a threshold for a different level of resolution
- 'Inrange': impose a range for a different level of resolutuion or it can be defined using a map of target levels for refinement:
- 'Targetlevel': the levels of resolution will be targeted but will be overruled by the maxlevel, minlevel entrance.

To refine the grid for this case, we will use the *former coarse simulation* and create a map for values where hmax is strictly positive (and/or umax,vmax different from zero), after removing the sea area. 

Here, the bathymetry map resolution is a 10m resolution (\f$ 𝑑𝑥_0=10𝑚\f$). We will impose:
- a level -3 resolution ( \f$𝑑𝑥_{−3}=80𝑚\f$) in the background domain, 
- a level 1 resolution ( \f$𝑑𝑥_1=5𝑚\f$) in the flooded area
- a level 2 resolution ( \f$𝑑𝑥_2=2.5𝑚\f$)  in the main river bed area. 

![coarseRun](./figure/coarse.png)
![refine](./figure/refine.png)

![meshrefine](./figure/test_1.png)

To use it, the following options need to be added to the param file:
```{txt} 
initlevel = -3 ;
maxlevel = 2 ;
minlevel = -3 ;
Adaptation = Targetlevel,refin_mask.nc?z ;
```


**Note_1**: In order to optimise the refinement, different bathymetry files can be provided. A higher resolution one with smaller extend can be provided as a second DEM and will be use by the code when refining the grid (the code will use the last entered, having the info available):
 ```{text}
DEM = Wesport_DEM_8m.nc?z;
DEM = Wesport_DEM_2m_zoomed.nc?z;
 ``` 


**Note_2**: The code also allows for zone outputs: the output of different zones defined by the user (outputname,x1,x2,y1,y2):
 ```{text}
outzone=zoomed.nc,5.3,5.4,0.5,0.8;
 ``` 


## Ground infiltration losses (Basic ILCL model)
An Initial Loss, Continuous Loss model has been implemented in the code. To use it, provide maps for the two coefficient as follow:
 ```{text}
initialloss=InitialLoss.asc;
continuousloss=ContinuousLoss.asc;
 ``` 

# Refining the grid in area of interest

## More DEMS
## Map of levels of resolution
## Adding the variable resolution in the code
Show adding a second high reso DEM (ALSO add mention of possibility to use path (relative or absolute))



#
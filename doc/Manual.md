# Manual {#Manual}

- [Input Parameters](#input-parameters)
  * [`BG_param.txt`](#-bg-paramtxt-)
    + [How to use the `BG_param.txt` file](#how-to-use-the--bg-paramtxt--file)
  * [List of Parameters](#list-of-parameters)
  * [Bathymetry/topography files](#bathymetrytopography-files)
    + [Masking](#masking)
  * [Boundaries](#boundaries)
  * [Bottom friction](#bottom-friction)
  * [Rivers and Area discharge](#rivers-and-area-discharge)
  * [Wind atm pressure forcing](#wind-atm-pressure-forcing)
  * [Output variables](#output-variables)


# Input Parameters
## `BG_param.txt`
All the model inputs are controlled by the BG_param.txt file. It is a simple text file that contains the parameters of the model that the user wishes to change.

### How to use the `BG_param.txt` file
Model parameters can be modified in the file by specifying the parameter name, the equal sign, the desired value(s) for the parameter and optionally a semi-column ;
```{txt}
#My Dummy BG_param files
#Any lines starting with the Hash tag will be ignored
#You can also leave blank lines

# Parameter order does not really matter but if you specify a parameter twice the first value will be overwritten

# You can only put one parameter per line
bathy = mybathyfile.nc

#Any number of leading space or space between the parameter name and the equal sign will be accepted. Tabs are not acceptable
    theta       =      1.0
#Obviously you have to put the right name down (spelling and not case sensitive) otherwise the line will be ignored
tteettaa = 1.22

#If you want to add a comment at the end of a line with a parameter you can do it after putting a semi column e.g.:
CFL = 9; Was 0.5 but I don't know what I'm doing

#The model has only one required parameter and that is the (level 0) bathymetry grid name
```
### General comment about input files
Some parameters expect a filename (e.g. `bathy`). While the model will accept any extension there are a couple of requirement. For spatially varying inputs, the extension controls how the file is read. You can use ESRI ascii grids `.asc`, mid-depth files `.md` or NetCDF `.nc`. NetCDF files are recommended and expected to have the `.nc` extension. Since NetCDF files can have several variables you can enforce (recommended) the variable being read by appending `?var` to the file name (e.g. `bathy = mybathyfile.nc?z`).

For spatially and temporally varying input (e.g. wind or atmospheric pressure) a/multiple NetCDF file with a 3D variable is expected (see [here](#wind-atm-pressure-forcing)).

Where the file input is a timeserie (e.g. boundary forcing), then the file extension is ignored but a text file is expected.
  
## List of Parameters
Remember that the only required parameter to run the model is the bathymetry file. But if you want to run something fun you will need to specify boundary conditions, initial conditions and/or some forcings.

### Input
| Parameter (synonyms)                       | Definition          | Default Value  |
| ------------------------ |:---------------------------------------:| -------------------:|
| bathy (or depfile)       | Bathy file name. See below for details  | _No Default values_ |
| gpudevice (or GPUDEVICE) | Define which GPU device to use (use -1 for CPU)     |   0 (default GPU)     |
| doubleprecision    |  if == 1 use double precision solver and math otherwise use float (much faster)  |    0  |
| mask    |  remove blocks from computation where the bathy/topo is larger than mask value (default no area are masked). see below for more details  |    9999.0  |
| nx    | Initial/input grid size (number of nodes) in x direction |  0  |
| ny    | Initial/input grid size (number of nodes) in y direction |  1  |
| dx    | Initial/input grid resolution ([m] for metric grid  or [decimal degree] for spherical grids) |  0.0  |
| grdalpha    | grid rotation [degrees clockwise of North for Y axis] |  0.0  |
| xo    | grid origin in x direction [m] or [decimal degree] if spherical ==1 |  0.0  |
| yo    | grid origin in y direction [m] or [decimal degree] if spherical ==1 |  0.0  |

### Forcing/Boundary
| Parameter (synonyms)                       | Definition          | Default Value  |
| ------------------------ |:---------------------------------------:| -------------------:|
| left    | Specify left boundary type (see below for details) |  1  |
| right    | Specify right boundary type (see below for details) |  1  |
| bot    | Specify bottom boundary type (see below for details) |  1  |
| top    | Specify top boundary type (see below for details) |  1  |
| initzs (or zsinit)    | Initial water level. The initial water level is calculated as a inverse distance to the value at the boundary use this parameter to define the initial water level if no boundaries or hotstart file are specified (e.g. for a lake) |  -999  |
| hotstartfile    | Netcdf file containing initial conditions either or a combination of u ,v ,zs ,hh and/or zb |  no file  |
| deformfile    | Netcdf file containing initial deformation from a tsunami. this deformation is applied to both the topography and water levels (Not Yet Implemented)  |  no file  |
| hotstep    | Step (in the time dimension?) to read hotstart condition in hotstart file. |  0 (i.e. first)  |
| frictionmodel    | Flag to tell the model which friction model is used. 1: Quadratic (is cf is fixed); 2: Smart 2017 manning style bottom friction safe for very shallow water (in this case the parameter cf is actually z0) |  0  |
| cfmap or roughnessmap    | Input grid of  either cf or z0 values (the same dimension as the input bathymetry) |  default value  |
| windfiles   | Wind forcing files containing wind speed in u and v direction. See below for details |  no files  |
| atmpfile    | Atmospheric forcing file (in netcdf format) containing atmospheric pressure in Pa.  |  no file  |
| Pa2m    | Conversion between atmospheric pressure changes to water level changes  |  0.00009916  |
| Paref    | Reference atmospheric pressure in Pa.  |  10130.0  |

### Hydrodynamics
| Parameter (synonyms)                       | Definition          | Default Value  |
| ------------------------ |:---------------------------------------:| -------------------:|
| eps    |  model drying height [m]  |    0.0001  |
| cf    |  model bottom friction (see below for details)  |    0.0001  |
| Cd    |  Wind drag coefficient  |    0.002  |
| CFL    |  CFL criterium between 0 and 1. Higher values may make the model unstable  |    0.5  |
| river |  5 parameters (comma separated) to define Point/area discharge in  the model see below for details |No rivers  |

### Time keeping
| Parameter (synonyms)                       | Definition          | Default Value  |
| ------------------------ |:---------------------------------------:| -------------------:|
| CFL    |  CFL criterium between 0 and 1. Higher values may make the model unstable  |    0.5  |
| outputtimestep (or  outtimestep)  |  time step between model output in s  |    0.0 (no output)  |
| endtime    |  time when the model stops s  |    0.0 (model will initialise but not run)  |
| totaltime    |  time at the start of the model s  |    0.0  |


### Output
| Parameter (synonyms)                       | Definition          | Default Value  |
| ------------------------ |:---------------------------------------:| -------------------:|
| outfile    |  netcdf output file name (if the file already exist a number will be appended to the filename and no file will be overwritten)  |    Output.nc  |
| outvars   |  List (comma separated) of variable to output to the netcdf file See below for details |    no variable  |
| TSOfile    |  Filename for timeseries output  |    no output  |
| resetmax    | if ==1 reset max variables after each output  |  0  |
| smallnc    | Flag for saving output as short integer. This reduces the size of output file (see scalefactor and addoffset below) |  1  |
| scalefactor    | Scale factor applied to output before saving as short integer. This follows the COARDS convention: `packed_data_value = nint((unpacked_data_value - add_offset) / scale_factor)`  |  0.01  |
| addoffset    | Offset applied to output before saving as short integer. This follows the COARDS convention: `packed_data_value = nint((unpacked_data_value - add_offset) / scale_factor)` |  0.0  |
| posdown    | switch to tell the modle that input grid is positive down. if podown==1 the input grid will be multiplied by -1.0 to transform to positive up |  0  |

### Miscelanious
| Parameter (synonyms)                       | Definition          | Default Value  |
| ------------------------ |:---------------------------------------:| -------------------:|
| g    | acceleration of gravity [m/s2]|  9.81  |
| rho    | fluid density [kg/m3] |  1025.0  |
| spherical    | switch to run the model in spherical (geographical) coordinates. This implies that the computation will occur in double precision  |  0  |
| Radius    | Earth radius used to calculate spherical grid corrections [m] |  6371220.0  |



## Bathymetry/topography files
Input bathymetry data as a regular grid using `.asc`, `.nc`, `.md`. This is a required parameter that ultimately also defines the extend of the model domain and model resolution.

`bathy = My_bathy_file.asc`

the correct way of reading the file will be dedicated by the file extension. For `.nc` (netcdf) files you can specify the name of the variable in the file as: `mybathyfile.nc?myvar` (will look for `zb` if none are specified). 

This input file is critical as it defines the extent of the model grid and the base resolution in the model. (The model is not adaptive yet but this is still relevant for the extent of the model.)

## Conputational mesh
BG-Flood generates its own mesh. By default, it is a quad regular mesh based on the DEM (Digital Elevation Model) extend and resolution. 
The extend of the mesh and resolution of the mesh can be modify by the user. The mesh can also be refined/coarsen in areas or following patterns prescribed by the user.

### Adaptative mesh

### Masking
Parts of the input bathymetry can be masked and excluded from the computation. The model extent (and locations of side boundaries) will remain the same but entire blocks can be removed from the computational memory. These area will appear as NaN in the output file. The input grid is first devided in blocks of 16x16. If all the value within a block exceed the mask value (9999 as default) then that block is excluded from memory and no computation will occur there.

There are no fancy treatment of the boundaries of masked blocks so it is safer to select a mask threshold (`mask`) to keep most of the masked are dry. If water tries to cross to a masked block, The boundary of blocks are treated as Neumann (no gradient) boundaries so the bathymetry of the 2 cells adjacent to a masked block are set to the same value (the value of the second cell removed from the masked block). 

## Boundaries
Four type of boundaries can be applied at the edge of the model. By default neumann (no perpendicular gradient) is applied. 


0. No slip (wall)
1. Neumann
2. dirichlets (water level )
3. Absorbing (only 1D/perpendicular absorbion is implemented)
4. ~~ 2D Absorbing~~ (Not implemented yet)

For Boundary type 2 and 3 (Dirichlet and Absorbing) the level at the boundary level is imposed from a file so a file needs to be sepcified:
```{txt}
    left = 1;
    right = 3;
    rightbndfile = mybndfile.txt
    top = 3;
    topbndfile = mybndfile.txt
```

### Boundary file (for type 2 or 3)
Water level boundary file are needed to type 2 nd 3 boundaries. The files are 2 (or more) column one with time in the first column and water level is the other(s). Note that the time step in the file doesn't need to be constant. The model will linearly interpolated between steps in the file. The file can be either comma separated or tab separated. This is automatically detected.

#### Uniform boundary file
For uniform boundary condition (along the boundary axis) the files needs to be 2 column:
```{txt}
    # Water level boundary
    0.0  1.2
    3600.0  1.2
    36000.0  1.22
    36060.0  1.25
    36120.0  1.24
  ```

#### Variable boundary file
For Variable boundary condition (along the boundary axis) the files needs to be 3+ columns. The boundary axis is then divided by the number of boundary column and the water level is interpolated between location. for example:
```{txt}
    # Water level boundary
    0.0  1.2 0.0
    3600.0  1.2 0.0
    36000.0  1.22 0.0
    36060.0  1.25 0.0
    36120.0  1.24 0.0
```

Here the the water level on the near-end of the boundary (left for bottom and top bnds; bottom for left and right bnds) axis will vary between 1.2 and 1.25 but the far-end (right for bottom and top bnd and top for left and right bnds) will remain constant at 0.0.

Here is an example with 3 water level column:
```{txt}
     # Water level boundary
    0.0 0.0 1.2 0.0
    3600.0 0.0 1.2 0.0
    36000.0 0.0 1.2 0.0
    36060.0 0.0 1.2 0.0
    36120.0 0.0 1.2 0.0 
```

In this case both near and far end of the boundary axis will remain zero and the center of the boundary axis will be 1.2m. 

## Bottom friction
Bottom friction is applied implicitly in the model (applied to velocities after momentum and continuity equations are solved). There are 3 friction equations implemented defined in `BG_param.txt` as `frictionmodel = `:

* 0 Quadratic
* 1 Smart
* -1 Mannings

`frictionmodel = 0` (i.e. Quadratic friction is the default with a uniform friction coefficient `cf = 0.001`. If a uniform friction is required add `cf=` to your `BG_param.txt` with the desired value. `cf` keyword is also used for the z0 of Smart formulation and n of the manning formulation. 

For non-uniform friction parameter use the keyword  `cfmap` or `roughnessmap` and assign an `.asc` or `.nc` file. for nc files you may need to supply the netcdf variable name. e.g. `cfmap=final_rough.nc?zo`. The roughness grid does not need to match the model grid dimension and coarser friction grid will be interpolated to the model grid cells and model cells outside of the forcing domain will be extrapolated (nearest value).

## Rivers and Area discharge
At this stage river can only be added to the model as a vertical discharge where the water is added to a rectangle on the model with no velocity.
To add rivers add a line per river with the parameters: `river = Fluxfile,xstart,xend,ystart,yend;` where `Fluxfile` is a 2 column text file containing time and dicharge in m3/s; `xstart` is the left coordinate of the square where the vertical discharge is applied, `xend` is the right coordinate of the square, `ystart` is the bottom coordinate of the square and `yend` is the top coordinate of the square. Example:
```{txt}
    river = Votualevu_R.txt,1867430,1867455,3914065,3914090;
    river = Mulomulo_R.txt,1867052,1867072,3911853,3911873;
```

## Wind atm pressure forcing
### Wind forcing (may contain bugs)
The hydrodynamics can be forced using a linear wind drag. the linear drag can be influenced with the keyword `Cd`. wind input is defined with the keyword `windfiles=`. There are several ways to use the keyword.
#### spatially uniform txt file: `windfiles=mywind.txt`
where `mywind.txt` is a text file with 3 column (time (in s), wind speed (m/s) and wind direction(degree North)). If the gris is in acoordinate system rotated from the north and a `grdalpha` is specified, the wind will be automatically rotated to the grid orientation. 

#### Spatially and time varying input `windfiles=mywind.nc?uw,mywind.nc?vw`
Here two arguments separated with a comma are expected. The first argument is the netcdf file and variable name containing the U component of the wind (Along the X axis) and the second argument is the netcdf file and variable name containing the V component of the wind (Along the Y axis). Both can be in the same netcdf file as in the example. Separate netcdf file name and variable name with a `?` similarly to other netcdf input option. The dimension of the wind forcing grid does not need to match the model grid dimension and coarser forcing will be interpolated to the model grid cells and model cells outside of the forcing domain will be extrapolated (nearest value). 


### Atmospheric pressure forcing
Spatially constant atmospheric pressure forcing is not relevant so only spatially varying forcing is feasable. like for the wind this is done through a netcdf file:
```{txt}
atmpfile=myncfile.nc?atmpres
```
The forcing pressure is expected to be in Pa and the effect of the atmospheric pressure gradient is calculated as the difference to a reference pressure `Paref=101300.0` converted to a height using `Pa2m=0..00009916`. If using hPa your will need to also change the reference pressure to `Paref=1013.0` and the conversion parameter to `Pa2m=0.009916`. As with the  wind forcing, the forcing grid does not need to match the model grid dimension and coarser forcing will be interpolated to the model grid cells and model cells outside of the forcing domain will be extrapolated (nearest value). 
 
## Output variables (Not up to date!!!)

### Snapshot outputs
| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| u       | U  velocity (at cell center) zonal velocity positive right | [m/s] |
| v       | V  velocity (at cell center) meridional velocity positive right | [m/s] |
| vort       | Vorticity | [rotation/s] |
| h       | water depth at cell center | [m] |
| zs       | Water level elevation above datum | [m] |
| zb       | Topography elevation above datum  | [m] |

### Mean/averaged output between output steps
This is for averaging variables in between output steps, useful for mean tidal flow calculation that averages out vortices. The average time is `outtimestep`

| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| umean       | Averaged U  velocity (at cell center) zonal velocity positive right | [m/s] |
| vmean       | Averaged V  velocity (at cell center) meridional velocity positive right | [m/s] |
| hmean       | Averaged water depth at cell center | [m] |
| zsmean       | Averaged Water level elevation above datum | [m] |

### Max output
The max can be calculated for the overall simulation (default) or between output steps (`resetmax = 1;`)

| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| umax       | Maximum U  velocity (at cell center) zonal velocity positive right | [m/s] |
| vmax       | Maximum V  velocity (at cell center) meridional velocity positive right | [m/s] |
| hmax       | Maximum water depth at cell center | [m] |
| zsmax     | Maximum Water level elevation above datum | [m] |

### Risk assesment related output
These variable are used to evaluate the damage resulting from the innundation.

put Udh / twet / ...

| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| umax       | Maximum U  velocity (at cell center) zonal velocity positive right | [m/s] |
| vmax       | Maximum V  velocity (at cell center) meridional velocity positive right | [m/s] |
| hmax       | Maximum water depth at cell center | [m] |
| zsmax     | Maximum Water level elevation above datum | [m] |

### Infiltration outputs
The quatity of water that infiltrate in the ground using the ILCL model is saved, as a cumulated variable in:

put Udh / twet / ...

| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| umax       | Maximum U  velocity (at cell center) zonal velocity positive right | [m/s] |
| vmax       | Maximum V  velocity (at cell center) meridional velocity positive right | [m/s] |
| hmax       | Maximum water depth at cell center | [m] |
| zsmax     | Maximum Water level elevation above datum | [m] |

### Other gradients and intermediate terms of the equations

 "dhdx","dhdy","dzsdx","dzsdy","dudx","dudy","dvdx","dvdy","Fhu","Fhv","Fqux","Fqvy","Fquy","Fqvx","Su","Sv","dh","dhu","dhv","cf"<br>|


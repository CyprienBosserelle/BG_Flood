# Manual

BG_Flood is a hydrodynamics model for simulating 2D depth-averaged flow. The following manual describes how to use the model but does not go into details about the model equations.

!!! Warning
    This manual is under revision and not up-to-date with the last version of the code.
    Please, use the Parameters and Forcings list as a reference

<!--
- [Input Parameters](#input-parameters)
  * [`BG_param.txt`](#-bg-paramtxt-)
    + [How to use the `BG_param.txt` file](#how-to-use-the--bg-paramtxt--file)
  * [List of Parameters](#list-of-parameters)
  * [Bathymetry/topography files](#bathymetrytopography-files)
  * [Computational mesh](#masking)
  * [Boundaries](#boundaries)
  * [Bottom friction](#bottom-friction)
  * [Rivers and Area discharge](#rivers-and-area-discharge)
  * [Wind and atmospheric pressure forcing](#wind-atm-pressure-forcing)
  * [Outputs](#output-variables)
-->

## Model controls
BG_Flood model reads an instruction file (or *param* file) that tells the model what to do and what forcing files to read. when instructions are not explicitely given the model will use the default value. How to use the param file is important for making BG_Flood do what you want it to do.

By default this instruction file is called ```BG_param.txt```and is the file BG_Flood will be looking for if it is launched without aregument. You can specify any filename you like if you launch BG_Flood this way:

```
./BG_Flood my_param_file.txt
```


### BG_param.txt
All the model inputs are controlled by the ```BG_param.txt``` file. It is a simple text file that contains the parameters of the model that the user wishes to change. While it is easy to build such file manually but also it is easy to use script and/or programming language to generate and modify text file to automatically to build and run many simulations.

#### How to use the BG_param.txt file
Model parameters can be modified in the file by specifying the parameter name, the equal sign, the desired value(s) for the parameter and optionally a semi-column :
```{txt}
#My Dummy BG_param files
#Any lines starting with the Hash tag will be ignored
#You can also leave blank lines

# Parameter order does not really matter but if you specify a parameter twice the first value will be overwritten

# You can only put one parameter per line
bathy = mybathyfile.nc

#Any number of leading space or space between the parameter name and the equal sign will be accepted. Tabs are not acceptable
    theta       =      1.0
#Obviously you have to put the right name down (spelling and not case sensitive) otherwise the line will be ignored without warning
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

[Full list of the parameters](ParametersList-py.md)


## Bathymetry/topography files
Input bathymetry data as a regular grid using `.asc`, `.nc`, `.md`. This is a required parameter that ultimately also defines the extend of the model domain and model resolution (if not defined by the user).

`bathy = My_bathy_file.asc`

The correct way of reading the file will be dedicated by the file extension. For `.nc` (netcdf) files you can specify the name of the variable in the file as: `mybathyfile.nc?myvar` (will look for `zb` if none are specified). 

This input file is critical as it defines the extent of the model grid and the base resolution in the model.

*Note*: 
!!! Note
    Different files can be provided to the code (using the instruction line multiple times). The code will use the last one having information at a given location when interpolating the data to create the bottom elevation (`zb`) variable.

## Computational mesh


!!! info "BG_Flood generates its own mesh"

BG_Flood generates its own  mesh based on the a referrence point, size and resolution ( by default extracted from the DEM (Digital Elevation Model) extent and resolution). 
The extent of the mesh and resolution of the mesh can be overridden by the user via several options. The mesh can also be refined/coarsen in areas or following patterns prescribed by the user.

The mesh is a Block Uniform Quadtree. Meaning that it is a domain similar to a quadtree except that each element of the quadtree is a regular cartesian grid of 16x16 elements.


### Mesh domain/extent

By default, the mesh origin, extent, and resolution are read from the DEM. when several DEMs are used (see #Bathymetry/topography files) the model domain is extracted from the first(closer to the top of the file) dem specified.

THe domain can be adjusted by specifying (one or more) coordinates of the bounding box of the domain such as:

```
xo = 1.0;
yo = 0.0;
xmax = 6.0;
ymax = 5.0;
```

In the case above BG_Flood will create a grid that has a bottom left corner at the ```(xo,yo)``` coordinate and extend at least all the way to coordinate ```(xmax,ymax)```.

### Resolution

The base (or reference) resolution is specified by `dx. By default it is the same as the DEM but can be overule. This is useful when testing the model with a coarser mesh for a quick run.

BG_Flood can build a refine mesh. In this case `dx` is the resolution of the reference level of refinement (level 0). Level 1 of refinement has a resolution of `0.5*dx`, level 2 is `0.25*dx`, level -1 is `2*dx` etc...

BG_Flood will automatically calculate the topography values at each mesh nodes from the DEM. This step can require interpolation (when DEM is coarser or out of phase with the DEM) or averaging (when the mesh is coarser than the DEM). 
!!! danger "Beware of aliasing"
  interpolating and averaging can lead to aliasing so 
 
#### top and right domain extend will not necessarily be what you specify
The ```(xmax,ymax)``` is at least reached because the final extent of the domain depends on what resolution (`dx`) is specified and the mesh block size so that the domain covers a round number of blocks. The mesh requires strictly compatible extent, resolution and round number of block in both `x` and `y` directions.
While this is sometimes confusing to acheive exactly, BG_Flood automatically adjust `xmax` and `ymax` to produce the mesh domain compatible with `dx` and the block size. Thus `xo`, `yo` and `dx` are always enforced.

!!! danger "Beware"
    When specifying `xo`, `yo`, `xmax`, `ymax` outside of the DEM extent, BG_Flood will extrapolate the values of forcing and DEM by repeating the values on the edge of the DEM domain. 

#### Example



### Area of Interest: polygonal extent

Instsead of a rectangular extent BG_Flood can build a mesh based on a polygon domain.

By specifying ```aoi = mydomain.xy``` BG_Flood will first build a virtual mesh but only keep the block that are inside or intersect the specified polygon.

!!! note
    `xo` and `yo` are still used as reference origin of the mesh using either the default DEM values or the ones specified by the user.  Since they are not trivially know at the start of the mesh generation `xmax` and `ymax` are also used to define the maximum extent but this is only noticeable by the user in the extent of the output layers.



### Topography masking
Parts of the input bathymetry can be masked and excluded from the computation. The model extent (and locations of side boundaries) will remain the same but entire blocks can be removed from the computational memory. These area will appear as NaN in the output file. The input grid is first devided in blocks of 16x16. If all the value within a block exceed the mask value (9999 as default) then that block is excluded from memory and no computation will occur there. An "area of interest" (AOI) can also be used to select a part of the domain. If none of the cells of a block is located in this area, that block will be excluded from memory.
The AOI is prefered other the mask method as the later can create waterfalls on the borders of the domain, specially if the rain-on-grid method is used.

There are no fancy treatment of the boundaries of masked blocks so it is safer to select a mask threshold (`mask`) to keep most of the masked are dry. If water tries to cross to a masked block, The boundary of blocks are treated as Neumann (no gradient) boundaries so the bathymetry of the 2 cells adjacent to a masked block are set to the same value (the value of the second cell removed from the masked block). 

#### Example
Below is an example of di

## Boundaries
Four type of boundaries can be applied at the edge of the model. By default neumann (no perpendicular gradient) is applied. 


- 0: No slip (wall)
- 1: Neumann
- 2: dirichlets (water level )
- 3: Absorbing (only 1D/perpendicular absorbion is implemented)
- (4: 2D Absorbing (Not implemented yet))

For Boundary type 2 and 3 (Dirichlet and Absorbing) the level at the boundary level is imposed from a file so a file needs to be sepcified:
```{txt}
    left = 1;
    right = mybndfile.txt,3;
    top = mybndfile.txt,3;
```
You can also specify boundary segments.

### Boundary file (for type 2 or 3)
Water level boundary file are needed to type 2 and 3 boundaries. The files are 2 (or more) columns, one with time in the first column and water level is the other(s). Note that the time step in the file doesn't need to be constant. The model will linearly interpolated between steps in the file. The file can be either comma separated or tab separated. This is automatically detected.

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
For Variable boundary condition (along the boundary axis) the files needs to be 3+ columns. The boundary axis is then divided by the number of boundary column and the water level is interpolated between location, for example:
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

There is no restriction in the number of columns. These values from each column will be forced uniformly spaced on the boundary and forcing in between will be linearly interpolated.


## Bottom friction
Bottom friction is applied implicitly in the model (applied to velocities after momentum and continuity equations are solved). 
There are 3 friction equations implemented defined in `BG_param.txt` as:

* ```frictionmodel = 0```: Basic Quadratic friction law (expecting the non-dimensional friction coefficient $c_f$ )
* ```frictionmodel = 1```: Roughtness length model from Smart (expecting a $z_0$ parameter in $c_f$)
* ```frictionmodel = -1```: Manning's friction model (expecting a Manning n parameter $c_f$)
Quadratic friction is the default, with a uniform friction coefficient:

```{txt}
  frictionmodel = 0
  cf = 0.001
``` 
If a uniform friction is required add `cf=` to your `BG_param.txt` with the desired value. `cf` keyword is also used for the $z0$ of Smart formulation and $n$ of the Manning formulation. 

For non-uniform friction parameter use the keyword  `cfmap` or `roughnessmap` and assign an `.asc` or `.nc` file. For nc files you may need to supply the netcdf variable name: e.g. `cfmap=final_rough.nc?zo`. The roughness grid does not need to match the model grid dimension and coarser friction grid will be interpolated to the model grid cells and model cells outside of the forcing domain will be extrapolated (nearest value).

## Rivers and Area discharge
At this stage river can only be added to the model as a vertical discharge where the water is added to a rectangle on the model with no velocity.
To add rivers add a line per river with the parameters: `river = Fluxfile,xstart,xend,ystart,yend;` where `Fluxfile` is a 2 column text file containing time and dicharge in $m^3/s$; `xstart` is the left coordinate of the rectangle where the vertical discharge is applied, `xend` is the right coordinate of the rectangle, `ystart` is the bottom coordinate of the rectangle and `yend` is the top coordinate of the rectangle. 
Example:
```{txt}
    river = Votualevu_R.txt,1867430,1867455,3914065,3914090;
    river = Mulomulo_R.txt,1867052,1867072,3911853,3911873;
```

## Wind atm pressure forcing
### Wind forcing (may contain bugs)
The hydrodynamics can be forced using a linear wind drag. the linear drag can be influenced with the keyword `Cd`. Wind input is defined with the keyword `windfiles`. There are several ways to use the keyword.
#### spatially uniform txt file: 
```{txt}
windfiles=mywind.txt
```
where `mywind.txt` is a text file with 3 column (time ($s$), wind speed ($m/s$) and wind direction ($\,^{{\circ}}$ North)). If the grid is in acoordinate system rotated from the north and a `grdalpha` is specified, the wind will be automatically rotated to the grid orientation. 

#### Spatially and time varying input
```{txt}
windfiles=mywind.nc?uw,mywind.nc?vw
```
Here two arguments separated with a comma are expected. The first argument is the netcdf file and variable name containing the U component of the wind (along the X axis) and the second argument is the netcdf file and variable name containing the V component of the wind (along the Y axis). Both can be in the same netcdf file as in the example or in separate netcdf files (add the variable name with a `?` similarly to other netcdf input options). The dimension of the wind forcing grid does not need to match the model grid dimension and coarser forcing will be interpolated to the model grid cells and model cells outside of the forcing domain will be extrapolated (nearest value). 


### Atmospheric pressure forcing
Spatially constant atmospheric pressure forcing is not relevant so only spatially varying forcing is feasable. Like for the wind this is done through a netcdf file:
```{txt}
atmpfile=myncfile.nc?atmpres
```
The forcing pressure is expected to be in Pa and the effect of the atmospheric pressure gradient is calculated as the difference to a reference pressure ```Paref=101300.0``` converted to a height using ```Pa2m=0.00009916```. If using hPa your will need to also change the reference pressure to ```Paref=1013.0``` and the conversion parameter to ```Pa2m=0.009916```. As with the  wind forcing, the forcing grid does not need to match the model grid dimension and coarser forcing will be interpolated to the model grid cells and model cells outside of the forcing domain will be extrapolated (nearest value). 
 
## Outputs
There is two types of outputs:
 - map outputs of 2D variables regularly through time.
 - time-series output of basic values, at a chosen position, at each time step.

### Map outputs
These maps are output as a nc file, with information on coordinates and blocks.

The map output can be modify by:
- defining a timestep (in $s$) for these outputs:
```{txt} 
outputtimestep = 3600.0;
```
- changing the set of variables in the output file (from the list given in the manual)
```{txt} 
outvars = zs,h,u,v,zb,hmax,Umax,hUmax,twet;
```
- changing the name of the output file:
```{txt} 
outfile = Results_tuto_basicRun.nc;
```

- choosing one or more zones to outputs (by default, the full domain is output):
 ```{txt}
outzone=MyZoneName.nc,x1,x2,y1,y2;
outzone=MyZoneNameb.nc,x1b,x2b,y1b,y2b;
 ``` 
- saving the output as float (variables are saved as short integer by default.):
```{txt} 
smallnc = 0;
```


By default, the variables outputs are the one listed in the following paragraph: Default snapshot outputs.
#### Default snapshot outputs
| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| u       | U  velocity (at cell center) zonal velocity positive right | [m/s] |
| v       | V  velocity (at cell center) meridional velocity positive right | [m/s] |
| h       | water depth at cell center | [m] |
| zs       | Water level elevation above datum | [m] |
| zb       | Topography elevation above datum  | [m] |

#### Complementary variables
| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| vort    | Vorticity | [rotation/s] |
| cf    | Bottom friction coefficient (Manning n or z0) | varies with model used |



#### Mean/averaged output between output steps
This is for averaging variables in between output steps, useful for mean tidal flow calculation that averages out vortices. The average time is `outtimestep`.

| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| umean       | Averaged u  velocity (at cell center) zonal velocity positive right | [m/s] |
| vmean       | Averaged v  velocity (at cell center) meridional velocity positive right | [m/s] |
| hmean       | Averaged water depth at cell center | [m] |
| zsmean       | Averaged Water level elevation above datum | [m] |

#### Max output
The max can be calculated for the overall simulation (default) or between output steps ( if `resetmax = 1;`)

| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| umax       | Maximum u  velocity (at cell center) zonal velocity positive right | [m/s] |
| vmax       | Maximum v  velocity (at cell center) meridional velocity positive right | [m/s] |
| hmax       | Maximum water depth at cell center | [m] |
| zsmax     | Maximum Water level elevation above datum | [m] |

#### Risk assesment related output
These variables are used to evaluate the damage resulting from the innundation (as a complement to hmax for example).

| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| hUmax       | Maximum of h time the velocity (U for amplitude of (u,v)) | [m2/s] |
| Umax       | Maximum of the velocity (U for amplitude of (u,v)) | [m/s] |
| twet       | Duration innundation of the cell in s (h>0.1m)  | [s] |

#### Model related outputs
These outputs  will be produce only if the associated model/forcing is used.

If an atmospheric forcing is used:

| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| Patm       | Atmospheric pressure | [Pa] |
| datmpdx/datmpdy       | Gradients of atmospheric pressure | [Pa/m] |

If the infiltration model (ILCL) is used, the quantity of water that infiltrate in the ground is saved, as a cumulated value in hgw.

| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| il       | Initial loss coefficient | [mm] |
| cl       | Continuous loss coefficient | [mm/hr] |
| hgw      | Cumulated height of infiltrated water in the ground | [m] |

#### Other gradients and intermediate terms of the equations
Terms of the equation can also been output such as the gradients (for error tracking mainly):

| Parameter                 | Definition          |     Unit         |
| ------------------------ |:----------------:|---------------------:|
| dhdx / dhdy       | Gradient of water elevation (h) in the x and y direction respectively  | [] |
| dzsdx / dzsdy       | Gradient of the water surface (zs) in the x and y direction respectively  | [] |
| dudx / dudy       | Gradient of x-velocity (u) in the x and y direction respectively  | [s-1] |
| dvdx / dvdy       | Gradient of y-velocity (v) in the x and y direction respectively  | [s-1] |
| Fhu / Fhv       | Flux of h time u in the x and y direction respectively  | [m2/s2] |
| Fqux / Fqvx       |  XXXXXXXXXXXX | [XXX] |
| Su / Sv       | Topographic source term  | [XXX] |
| dh       | Variation in elevation  | [m] |
| du / dv       | Variation of the x- and y-velocity respectively  | [m/s] |


### Point or Time-Serie output

For each Time-Serie output needed, a line with the destination file and the postition is needed:

``` txt
TSnodesout=Offshore.txt,xloc,yloc;
```
The file contains 5 colums $(t, zs, h, u,v)$ with the value at the nearest grid point (to the position defined by the user).

## Adaptative grid
At the stage of development, the code will adapt the grid only before the computation but not along the calcul.

The code is based on a Block-uniform quadtree mesh. Each block, actually a 16 by 16 cells, is one unit of computation in the GPU.
These blocks can have different resolutions (but the resolution does not change during the computation at this stage).

By default, the initial resolution of the grid is the resolution of the bathymetry/topographic data. To refine or coarsen the grid, you can weather use the ```dx``` key word and choose a new resolution for the whole domain; wether use different levels of resolution. 
The reference level, correponding to the bathymetry resolution or $dx$ if defined by the user, will be the level 0. Levels of resolution are then defined in relation to the reference levels using positive integers to increase the resolution or refine and negative integer to coarsen the grid by a multiple of two. For a given level  $n$ , the resolution  $dx_n$
  will be:
$$
dx_n=$rac{dx_0}{2^n}
$$
 
with  $dx_0$ the resolution at level 0. 

When refinning using the level implementation, different key words are expected:

- ```Initlevel```: level used to create the first mesh created by the code in the mesh refinement process (only a technical information)
- ```Maxlevel```: maximim level of refinement (over-ruling other commands)
- ```Minlevel```: minimum level of refinement (over-ruling other commands)

The grid can also be unregular with an adaptition of the grid to the model (variables at initialisation step or user-defined refinement map). In this case, the cells will be devided in 4 cells for refinement, or 4 cells merged in one for coarsening. The code will ensure a progressive change of resolution (no cell should have a neighbour with more than 1 level of resolution of difference.)

The different methods of refinement available in the code are called using the key word "Adaptation". The refinement can be based on a classical input variable or a variable calculated during the initialisation:

- ```Threshold```: impose a threshold for a different level of resolution
- ```Inrange```: impose a range for a different level of resolution
- ```Targetlevel```: the levels of resolution will be targeted but will be overruled by the maxlevel, minlevel entrance.

For example, for the adaptation with targeted levels:
```{txt} 
initlevel = init ;
maxlevel =  max ;
minlevel = min ;
Adaptation = Targetlevel,MyLevelMap.nc?levels ;
```
Where max and min represent the range of level expected, and init is a number in this range (it is advice to use the min level). MyLevelMap is a netcdf 2D map of levels, that can have a different resolution and dimension from the computational grid. The amplitude of the levels on the map can also be larger than than min/max. All these levels are positive or negative integer.

For a bathymetry map of $10$m resolution ( or `dx=10`), we can use `minlevel=-3`, `maxlevel=2` and `initlevel=-3` to create a grid where coarser cell will be $10/2^{-3}=80$m and the thinner $10/2^2=2.5$m. The level file would contains a 2D map with integer values from -3 to 2.


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

Different formats can be used for this file ( ASCII: .asc or NetCDF .nc principally).
As Netcdf files can contain different variables, the "?" follows by the name of the variable is needed.

Without further information, the code will run will default values for physical parameters and initialisation, Neumann boundaries and no external forcing:



=> Without further information, the model will run with the default values.

=>results

# Basic fluvial flooding set-up

## River discharge

## Time parameters

## Resolution

=>results

# Completing the set-up
## Adding boundary conditions
Presentation
### Tidal forcing

## Bottom friction
Attention: Extrapolation outside of the map

## Initialisation

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
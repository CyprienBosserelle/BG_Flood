# Groundwater module
Groundwater directly interact with surface water and can be a driver for inundation. This is a useful functionality for BG_Flood. 
## Motivation
BG_Flood simple infiltration module is often a limitation when simulating inundation where a shallow aquifer may be full and prevent infiltration or where a shallow aquifer may be pushed up by the tide and cause groundwater flooding. Having a groundwater module is quite critical.

Groundwater simulation is a difficult endeavour because sparse information lead to large uncertainties in the forcing of the model. In addition, groundwater processes often operate at much larger time-scale than inundation timescale. 

Hence the groundwater module of BG_Flood is not intended as a state of the art groundwater model but instead as a very simple module that allow to account for basic interaction between groundwater,surface water. This module in intended for the shallow part of teh groundwater and ignores the complex 3d natyure of groundwater flow. 

### Simple depth-averaged boussinesq groundwater model
This groundwater module simulates 2D transient water table dynamics within an unconfined aquifer by solving the linearized Boussinesq equation. It couples vertical surface infiltration (the source term) with lateral Darcy routing to track how groundwater mounds grow under localized recharge basins and subsequently decay or flatten out across the simulation grid over time.

### Time step consideration
When running an explicit 2D unconfined groundwater model like this one, choosing the correct time step ($\Delta t$) is critical to keep the simulation from crashing or leaking mass. Because the module uses an explicit finite-difference scheme to solve the Boussinesq equation, it is bound by the Courant-Friedrichs-Lewy (CFL) stability criterion, which requires that the numerical wave speed does not propagate faster than one grid cell per time step. Critically this time step could be smaller than the hydrodynamic time step. In that case each hydrodynamic time step will be bound to one or more groundwater model time step. 

## No unsaturated flow
The water that infiltrates is directly added to the surface of the aquifer. This is an OK assumption when the aquifere surface very close to the ground surface but can be an absurd assumption if the groundwater surface is deep.  

### Need more inputs
the groundwater module require a bunch of additional inputs. These are required to be given as maps:

* **zb_gw : Aquifer floor elevation** : Defines the bottom of the aquifer. In reality this could be really deep but using the true aquifer floor might results in debilitating small time steps. We would recommend artificially limit the dept of the aquifer to a safe distance below ground maybe as a __hard__ (i.e. no-flow) bourdary with the deeper aquifer.
* **Sy_gw : Aquifer porosity** The depth-averaged porosity of the aquifer. Typically between 20% and 40%. Ask your favorite hydrogeologist!
* **K_gw : Aquifer conductivity** | The depth-averaged hydraulic conductivity of the aquifer [m/s]. Sometimes expressed in m/day by our hydogeologist friends so can be a very small number. If you aquifer has a small conductivity it may not be worth having a groundwater module!
* **fs_gw : Infiltration rates** | fs_gw : This is the soil infltration rates in mm/h. This only applies to infiltration and not exfiltration (exfiltration rate not used, all groundwater head above ground becomes surface flow)
* **zs_gw_init : Aquifer surface level** | This is the elevation of the groundwater water level [m above the DEM datum]. This could be extracted from a groundwater model or interpolated from bore data. By default the model will not allow it to be higher than ground level. 


# Usage

the module operates as an optional step in the main model loop. 

For example:
```
groundwater=true

K_gw = GW_hydroCond.nc?z
fs_gw = GW_InfRate.nc?z
Sy_gw = GW_Porosity.nc?z
zb_gw = zb_gw_init.nc?z
zs_gw_init = zs_gw_init.nc?z
```

# Test
There are not test for groundwater module yet. 


# Validation
Hantush (1967) derived an analytical, linearized solution to the 2D Boussinesq equation to simulate how an unconfined water table rises and falls beneath a rectangular recharge basin. By utilizing a specialized double error function integral ($S^*$), the model provides a mathematical benchmark for predicting transient groundwater mound growth during steady infiltration and its subsequent decay via lateral Darcy routing.

## Setup

### Groundwater parameter
We are running a very simple inoput. flat topography, constant  K(500m/day), Fs (50mm/h) and Sy (30%)
We set the bottom of the aquifere at 10m deep and the aquifere (top) 2m below the surface.

```
K_gw = K_gw_test_500.nc?z
fs_gw = fs_gw_50mmph.nc?z
Sy_gw = Sy_gw_30p.nc?z
zb_gw = zb_gw_init.nc?z
zs_gw_init = zs_gw_h2m.nc?z
```

### Recharge mechanism.
We could setup a rainfall grid but it is easier to use a river injection in place. We just need to make sure the flow injected matches the rainfall rate:  50 mm/h over a 50x50 area is 0.034 m3/s

```
river=river.tmp,100.0,150.0,100.0,150.0
```

with `river.tmp` looking like this:
```
0.000000,0.0347200
3600.000000,0.0347200
3600.1,0.0
36000.0,0.0
```

## Results and comparison
Results are excellent. At the center of the injection, the model and analytical solution are virtually indistinguishable. 
<img width="702" height="477" alt="image" src="https://github.com/user-attachments/assets/a9be09d3-9909-4b6e-a0d6-c1f44e418349" />

Where the model and solution start to deviate are near the boundary condition or after a long time where bnd are influencing the results.

<img width="702" height="400" alt="image" src="https://github.com/user-attachments/assets/cd3bc01e-7e1b-41ad-ab00-2f7e88ebff3c" />
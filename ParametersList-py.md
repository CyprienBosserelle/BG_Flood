# Paramter and Forcing list for BG_Flood

BG_flood user interface consists in a text file, associating key words to user chosen parameters and forcings.
## List of the Parameters' input

|_Reference_|_Keys_|_default_|_Explanation_|
|---|---|---|---|
|test|test| -1|-1: no test, 99: run all independent tests, X: run test X|
|GPUDEVICE| GPUDEVICE , gpu | 0|0: first available GPU, -1: CPU single core, 2+: other GPU|
|doubleprecision|doubleprecision| 0|0: float precision, 1: double precision|
|maxlevel|maxlevel| 0|Maximum level for grid adaptation (overwrite the adaptation map if use) |
|minlevel|minlevel| 0|Minumim level for grid adaptation (overwrite the adaptation map if use) |
|initlevel|initlevel| 0|Initial level of grid adaptation (based on dx if defined by the user or on the resolution of the topography/bathymetry input)|
|conserveElevation|conserveElevation| false||
|membuffer|membuffer| 1.05|needs to allocate more memory than initially needed so adaptation can happen without memory reallocation|
|eps|eps| 0.0001|Drying height in m (under eps, the surface is concidered dry)|
|VelThreshold| VelThreshold , vthresh , vmax , velmax | -1.0|Using Velocity threshold if the the velocuity exceeds that threshold. Advice value of 16.0 to use or negative value (-1) to turn off|
|Cd|Cd| 0.002|Wind drag coefficient|
|Pa2m|Pa2m| 0.00009916|XXXX in Pa (if unit is hPa then user should use 0.009916)|
|Paref|Paref| 101300.0|Reference pressure in Pa (if unit is hPa then user should use 1013.0)|
|mask|mask| 9999.0|mask any zb above this value. if the entire Block is masked then it is not allocated in the memory|
|dt|dt| 0.0|Model time step in s.|
|CFL|CFL| 0.5|Current Freidrich Limiter|
|theta|theta| 1.3|Minmod limiter parameter, theta in [1,2]. <br>Can be used to tune the momentum dissipation (theta=1 gives minmod the most dissipative limiter and theta = 2 gives	superbee, the least dissipative).|
|outputtimestep| outputtimestep , outtimestep , outputstep | 0.0|Number of seconds between netCDF outputs, 0.0 for none|
|endtime|endtime| 0.0|Total runtime in s, will be calculated based on bnd input as min(length of the shortest time series, user defined) and should be shorter than any time-varying forcing|
|totaltime| totaltime , inittime | 0.0|Total simulation time in s|
|dtinit|dtinit| -1|Maximum initial time steps in s (should be positive, advice 0.1 if dry domain initialement) |
|outfile|outfile|||
|TSnodesout| TSnodesout , TSOutput |None<br>|Time serie output, giving a file name and a (x,y) position <br>(which will be converted to nearest grid position). <br>This keyword can be used multiple times to extract time series at different locations.<br>The data is stocked for each timestep and written by flocs.<br>The resulting file contains (t,zs,h,u,v)<br>Example: "TSnodesout: Offshore.txt,3101.00,4982.57" (*filename,x,y*)<br>|
|outvars|outvars|"zb", "zs", "u", "v", "h"<br>|list of names of the variables to output (for 2D maps),<br><br> supported variables = "zb", "zs", "u", "v", "h", "hmean", "zsmean", "umean", "vmean", "hmax", "zsmax", "umax", "vmax" ,"vort","dhdx","dhdy","dzsdx","dzsdy","dudx","dudy","dvdx","dvdy","Fhu","Fhv","Fqux","Fqvy","Fquy","Fqvx","Su","Sv","dh","dhu","dhv","cf"<br>|
|resetmax|resetmax| false||
|outishift|outishift| 0|DEBUGGING ONLY: allow cell shift (1 or -1) in x direction to visualise the halo around blocks in the output |
|outjshift|outjshift| 0|DEBUGGING ONLY: allow cell shift (1 or -1) in y direction to visualise the halo around blocks in the output |
|nx|nx|0|Initial grid size|
|ny|ny| 0|Initial grid size|
|dx|dx| nan("")|Grid resolution in the coordinate system unit in m.|
|grdalpha|grdalpha|||
|xo| xo , xmin | nan("")|Grid x origin (if not alter by the user, will be defined based on the topography/bathymetry input map)|
|yo| yo , ymin | nan("")|Grid y origin (if not alter by the user, will be defined based on the topography/bathymetry input map)|
|xmax|xmax| nan("")|Grid xmax (if not alter by the user, will be defined based on the topography/bathymetry input map)|
|ymax|ymax| nan("")|Grid ymax (if not alter by the user, will be defined based on the topography/bathymetry input map)|
|g|g| 9.81|Gravity in m.s-2|
|rho|rho| 1025.0|Fluid density in kg/m-3|
|smallnc|smallnc| 1|Short integer conversion for netcdf outputs. 1: save as short integer for the netcdf file, if 0 then save all variables as float|
|scalefactor|scalefactor| 0.01f|Scale factor used for the short integer conversion for netcdf outputs|
|addoffset|addoffset| 0.0f|Offset add during the short integer conversion for netcdf outputs|
|posdown|posdown| 0|Flag for bathy input. Model requirement is positive up  so if posdown ==1 then zb=zb*-1.0f|
|use_catalyst|use_catalyst| 0|Switch to use ParaView Catalyst|
|catalyst_python_pipeline|catalyst_python_pipeline| 0|Pipeline to use ParaView Catalyst|
|vtk_output_frequency|vtk_output_frequency| 0|Output frequency for ParaView Catalyst|
|vtk_output_time_interval|vtk_output_time_interval| 1.0|XXX  for ParaView Catalyst|
|vtk_outputfile_root|vtk_outputfile_root|||
|python_pipeline|python_pipeline|||
|zsinit| zsinit , initzs | nan("")|Init zs for cold start in m. If not specified by user and no bnd file = 1 then sanity check will set it to 0.0|
|zsoffset|zsoffset| nan("")|Add a water level offset in m to initial conditions and boundaries (0.0 by default)|
|hotstartfile|hotstartfile||Allow to hotstart (or restart) the computation providing a netcdf file containing at least zb, h or zs, u and v<br>Default= None<br>|
|hotstep|hotstep| 0|Step to read if hotstart file has multiple steps (step and not (computation) time)|
|spherical|spherical| 0|Flag for geographical coordinate. Can be activated by using the keyword geographic|
|Radius|Radius| 6371220.|Earth radius [m]|
|frictionmodel|frictionmodel| 0|Bottom friction model (-1: Manning model, 0: quadratic, 1: Smart model [REF])|
---

## List of the Forcings' inputs

|_Reference_|_Keys_|_default_|_Example_|_Explanation_|
|---|---|---|---|---|
|cf|cf|0.0001 (constant)|cf=0.001;<br>cfmap=bottom_friction.nc?bfc;|Bottom friction coefficient map (associated to the chosen bottom friction model)<br>NEED TO BE MODIFIED TO HAVE THE GOOD KEYS|
|Bathy| Bathy , bathyfile , bathymetry , depfile , depthfile , topofile , topo , DEM |None but input NECESSARY|"bathy=Westport_DEM_2020.nc?z" or "topo=Westport_DEM_2020.asc"<br>"bathy=South_Island_200.nc?z, West_Coast_100.nc?z, Westport_10.nc?z"| Bathymetry/Topography input, ONLY NECESSARY INPUT<br>Different format are accepted: .asc, .nc, .md. , the grid must be regular with growing coordinate.<br>This grid will define the extend of the model domain and model resolution (if not inform by the user).<br>The coordinate can be cartesian or spheric (To be check).<br>A list of file can also be use to provide a thiner resolution localy for example.<br>The first file will be use to define the domain area and base resolution but the following file<br>will be used during the refinement process.|
|left| left , leftbndfile , leftbnd |1|left = 0;<br>left = 2,leftBnd.txt;| 0:Wall (no slip); 1:neumann (zeros gradient) [Default]; 2:sealevel dirichlet; 3: Absorbing 1D 4: Absorbing 2D (not yet implemented)<br>For type 2 and 3 boundary, a file need to be added to determine the vaules at the boundary. This file will consist in a first time<br>column (with possibly variable time steps) and levels in the following columns (1 column correspond to a constant value along the boundary,<br>2 column will correspond to values at boundary edges with linear evolution in between, n colums will correspond to n regularly space<br>applied values along the boundary)|
|right| right , rightbndfile , rightbnd |1|right = 0;<br>right = 2,rightBnd.txt;|Same as left boundary|
|top| top , topbndfile , topbnd |1|top = 0;<br>top = 2,topBnd.txt;|Same as left boundary|
|bot| bot , botbndfile , botbnd , bottom |1|bot = 0;<br>bot = 2,botBnd.txt;|Same as left boundary|
|deform|deform|None|XXXXXXXXXXXXXXXX|Deform are maps to applie to both zs and zb; this is often co-seismic vertical deformation used to generate tsunami initial wave<br>Here you can spread the deformation across a certain amount of time and apply it at any point in the model|
|rivers| rivers , river |None|river = Votualevu_R.txt,1867430,1867455,3914065,3914090;|The river is added as a vertical discharge on a chosen area (the user input consisting in a Time serie and a rectangular area definition: river = Fluxfile,xstart,xend,ystart,yend).<br>The whole cells containing the corners of the area will be included in the area, no horizontal velocity is applied.<br>To add multiple rivers, just add different lines in the input file (one by river).|
|Atmp| Atmp , atmpfile ||||
|Rain| Rain , rainfile |None|For a uniform rain: "rain=rain_forcing.txt" (2 column values, time (not necessary unformly distributed) and rain intensity)<br>For a non-uniform rain: "rain=rain_forcing.nc?rain" (to define the entry netcdf file and the variable associated to the rain "rain", after the "?")| Rain dynamic forcing: This allow to force a time varying, space varying rain on the model, in mm/h.<br>The rain can also be forced using a time serie (rain will be considered uniform on the domain)|
---

## List of the non-identified inputs

|_Reference_|_Keys_|
|---|---|
|Adaptation|Adaptation|
|cfmap| cfmap , roughnessmap |
|Wind| Wind , windfiles |
---

*Note* : The keys are not case sensitive.

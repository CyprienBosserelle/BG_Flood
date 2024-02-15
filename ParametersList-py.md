# Paramter and Forcing list for BG_Flood

BG_flood user interface consists in a text file, associating key words to user chosen parameters and forcings.
## List of the Parameters' input

### General parameters
|_Reference_|_Keys_|_default_|_Explanation_|
|---|---|---|---|
|test|test| -1|-1: no test, 99: run all independent tests, X: run test X|
|g|g| 9.81|Gravity in m.s-2|
|rho|rho| 1025.0|Fluid density in kg.m-3|
|eps|eps| 0.0001|Drying height in m (if h<eps, the surface is concidered dry)|
|dt|dt| 0.0|Model time step in s.|
|CFL|CFL| 0.5|Current Freidrich Limiter|
|theta|theta| 1.3|Minmod limiter parameter, theta in [1,2]. <br>Can be used to tune the momentum dissipation (theta=1 gives minmod the most dissipative limiter and theta = 2 gives	superbee, the least dissipative).|
|VelThreshold| VelThreshold , vthresh , vmax , velmax | -1.0|Using Velocity threshold if the the velocuity exceeds that threshold. Advice value of 16.0 to use or negative value (-1) to turn off|
|frictionmodel|frictionmodel| 0|Bottom friction model (-1: Manning model, 0: quadratic, 1: Smart model)|
|savebyblk| savebyblk , writebyblk , saveperblk , writeperblk | 0|Bottom friction model (-1: Manning model, 0: quadratic, 1: Smart model)|
|cf| cf , roughness , cfmap | 0.0001|Bottom friction coefficient for flow model (if constant)|
|Cd|Cd| 0.002|Wind drag coefficient|
|conserveElevation|conserveElevation| false|Switch to force the conservation of zs instead of h at the interface between coarse and fine blocks|
|wetdryfix| wetdryfix , reminstab | true|Switch to remove wet/dry instability (i.e. true reoves instability and false leaves the model as is)|
|Pa2m|Pa2m| 0.00009916|Conversion between atmospheric pressure changes to water level changes in Pa (if unit is hPa then user should use 0.009916)|
|Paref|Paref| 101300.0|Reference pressure in Pa (if unit is hPa then user should use 1013.0)|
|GPUDEVICE| GPUDEVICE , gpu | 0|0: first available GPU, -1: CPU single core, 2+: other GPU|
|doubleprecision|doubleprecision| 0|0: float precision, 1: double precision|
|engine|engine| 1|1: Buttinger-Kreuzhuber et al. 2019, 2: Kurganov (Popinet 2011), 3: KurganovATMP same as Kurganov but with atmospheric forcing terms |
### Grid parameters
|_Reference_|_Keys_|_default_|_Explanation_|
|---|---|---|---|
|dx|dx| nan("")|Grid resolution in the coordinate system unit in m.|
|nx|nx| 0|Initial grid size in x direction|
|ny|ny| 0|Initial grid size in y direction|
|xo| xo , xmin | nan("")|Grid x origin (if not alter by the user, will be defined based on the topography/bathymetry input map)|
|yo| yo , ymin | nan("")|Grid y origin (if not alter by the user, will be defined based on the topography/bathymetry input map)|
|ymax|ymax| nan("")|Grid ymax (if not alter by the user, will be defined based on the topography/bathymetry input map)|
|xmax|xmax| nan("")|Grid xmax (if not alter by the user, will be defined based on the topography/bathymetry input map)|
|grdalpha|grdalpha| nan("")|Grid rotation Y axis from the North input in degrees but later converted to rad|
|posdown|posdown| 0|Flag for bathy input. Model requirement is positive up  so if posdown ==1 then zb=zb*-1.0f|
|spherical| spherical , geo | 0|Flag for sperical coordinate (still in development)|
|Radius|Radius| 6371220.|Earth radius [m]|
|mask|mask| 9999.0|Mask any zb above this value. If the entire Block is masked then it is not allocated in the memory|
### Adaptation
|_Reference_|_Keys_|_default_|_Explanation_|
|---|---|---|---|
|initlevel|initlevel| 0|Initial level of grid adaptation (based on dx if defined by the user or on the resolution of the topography/bathymetry input)|
|maxlevel|maxlevel| -99999|Maximum level for grid adaptation (overwrite the adaptation map if use) |
|minlevel|minlevel| -99999|Minumim level for grid adaptation (overwrite the adaptation map if use) |
|membuffer|membuffer| 1.05|Needs to allocate more memory than initially needed so adaptation can happen without memory reallocation|
### Timekeeping
|_Reference_|_Keys_|_default_|_Explanation_|
|---|---|---|---|
|outputtimestep| outputtimestep , outtimestep , outputstep | 0.0|Number of seconds between netCDF outputs, 0.0 for none|
|endtime| endtime , stoptime , end , stop , end_time , stop_time | 0.0|Number of seconds between netCDF outputs, 0.0 for none|
|totaltime| totaltime , inittime , starttime , start_time , init_time , start , init | 0.0|Total simulation time in s|
|dtinit|dtinit| -1|Maximum initial time steps in s (should be positive, advice 0.1 if dry domain initialement) |
|dtmin|dtmin| 0.0005|Minimum accepted time steps in s (a lower value will be concidered a crash of the code, and stop the run)|
###  Initialisation
|_Reference_|_Keys_|_default_|_Explanation_|
|---|---|---|---|
|zsinit| zsinit , initzs | nan("")|Init zs for cold start in m. If not specified by user and no bnd file = 1 then sanity check will set it to 0.0|
|zsoffset|zsoffset| nan("")|Add a water level offset in m to initial conditions and boundaries (0.0 by default)|
|hotstartfile|hotstartfile|None<br>|Allow to hotstart (or restart) the computation providing a netcdf file containing at least zb, h or zs, u and v<br>|
|hotstep|hotstep| 0|Step to read if hotstart file has multiple steps (step and not (computation) time)|
### Outputs
|_Reference_|_Keys_|_default_|_Explanation_|
|---|---|---|---|
|TSnodesout| TSnodesout , TSOutput |None<br>|Time serie output, giving a file name and a (x,y) position <br>(which will be converted to nearest grid position). <br>This keyword can be used multiple times to extract time series at different locations.<br>The data is stocked for each timestep and written by flocs.<br>The resulting file contains (t,zs,h,u,v)<br>Example: "TSnodesout = Offshore.txt,3101.00,4982.57" (*filename,x,y*)<br>|
|outfile|outfile| "Output.nc"|netcdf output file name|
|outvars|outvars|"zb", "zs", "u", "v", "h"<br>|List of names of the variables to output (for 2D maps)<br>Supported variables = "zb", "zs", "u", "v", "h", "hmean", "zsmean", "umean", "vmean", "hUmean", "Umean", "hmax", "zsmax", "umax", "vmax", "hUmax", "Umax", "twet", "dhdx","dhdy","dzsdx","dzsdy","dudx","dudy","dvdx","dvdy","Fhu","Fhv","Fqux","Fqvy","Fquy","Fqvx","Su","Sv","dh","dhu","dhv","cf","Patm", "datmpdx","datmpdy","il","cl","hgw";<br>|
|outzone|outzone|Full domain<br>|Zoned output (netcdf file), giving a file name and the position of two corner points<br>(which will be converted to a rectagle containing full blocks).<br>This keyword can be used multiple times to output maps of different areas.<br>Example: "outzone=zoomed.nc,5.3,5.4,0.5,0.8;" (*filename,x1,x2,y1,y2*)<br>|
|resetmax|resetmax| false|Switch to reset the "max" outputs after each output|
|outishift|outishift| 0|DEBUGGING ONLY: allow cell shift (1 or -1) in x direction to visualise the halo around blocks in the output |
|outjshift|outjshift| 0|DEBUGGING ONLY: allow cell shift (1 or -1) in y direction to visualise the halo around blocks in the output |
### Netcdf parameters
|_Reference_|_Keys_|_default_|_Explanation_|
|---|---|---|---|
|smallnc|smallnc| 1|Short integer conversion for netcdf outputs. 1: save as short integer for the netcdf file, if 0 then save all variables as float|
|scalefactor|scalefactor| 0.01f|Scale factor used for the short integer conversion for netcdf outputs|
|addoffset|addoffset| 0.0f|Offset add during the short integer conversion for netcdf outputs|
###  ParaView Catalyst parameters (SPECIAL USE WITH PARAVIEW)
|_Reference_|_Keys_|_default_|_Explanation_|
|---|---|---|---|
|use_catalyst|use_catalyst| 0|Switch to use ParaView Catalyst|
|catalyst_python_pipeline|catalyst_python_pipeline| 0|Pipeline to use ParaView Catalyst|
|vtk_output_frequency|vtk_output_frequency| 0|Output frequency for ParaView Catalyst|
|vtk_output_time_interval|vtk_output_time_interval| 1.0|Output time step for ParaView Catalyst|
|vtk_outputfile_root|vtk_outputfile_root| "bg_out"|output file name for ParaView Catalyst|
|python_pipeline|python_pipeline| "coproc.py"|python pipeline for ParaView Catalyst|
|rainbnd| rainbnd , rainonbnd | false|when false it force the rain foring on the bnd cells to be null.|
|adaptmaxiteration| adaptmaxiteration , maxiterationadapt | 20|Maximum number of iteration for adaptation. default 20|
|reftime| reftime , referencetime , timeref | ""|Reference time string as yyyy-mm-ddTHH:MM:SS|
---

## List of the Forcings' inputs

|_Reference_|_Keys_|_default_|_Example_|_Explanation_|
|---|---|---|---|---|
|cf| cf , roughness , cfmap |(see constant in parameters)|cf=0.001;<br>cf=bottom_friction.nc?bfc;|Bottom friction coefficient map (associated to the chosen bottom friction model)|
|Bathy| Bathy , bathyfile , bathymetry , depfile , depthfile , topofile , topo , DEM |None but input NECESSARY|bathy=Westport_DEM_2020.nc?z<br>topo=Westport_DEM_2020.asc| Bathymetry/Topography input, ONLY NECESSARY INPUT<br>Different format are accepted: .asc, .nc, .md. , the grid must be regular with growing coordinate.<br>This grid will define the extend of the model domain and model resolution (if not inform by the user).<br>The coordinate can be cartesian or spherical (still in development).<br>A list of file can also be use to provide a thiner resolution localy by using the key word each time on a different line.<br>The first file will be use to define the domain area and base resolution but the following file<br>will be used during the refinement process.|
|AOI| AOI , aoipoly |N/A|AOI=myarea.gmt;|Area of interest polygon<br>the input file is a text file with 2 columns containing the coordinate of a closed polygon (last line==first line)|
|left| left , leftbndfile , leftbnd |1|left = 0;<br>left = leftBnd.txt,2;| 0:Wall (no slip); 1:neumann (zeros gradient) [Default]; 2:sealevel dirichlet; 3: Absorbing 1D 4: Absorbing 2D (not yet implemented)<br>For type 2 and 3 boundary, a file need to be added to determine the vaules at the boundary. This file will consist in a first column containing time (with possibly variable time steps) and forcing values in the following columns (1 column of values corresponding to a constant value along the boundary, 2 columns correspond to values at boundary edges with linear evolution in between, n columns correspond to n regularly spaced values applied along the boundary)|
|right| right , rightbndfile , rightbnd |1|right = 0;<br>right = rightBnd.txt,2;|Same as left boundary|
|top| top , topbndfile , topbnd |1|top = 0;<br>top = topBnd.txt,2;|Same as left boundary|
|bot| bot , botbndfile , botbnd , bottom |1|bot = 0;<br>bot = botBnd.txt,2;|Same as left boundary|
|deform|deform|None|deform = myDeform.nc?z_def,3.0,10.0;<br>deform = *filename*, *time of initial rupture*, *rising time*;|Deform are maps to apply to both zs and zb; this is often co-seismic vertical deformation used to generate tsunami initial wave<br>Here you can spread the deformation across a certain amount of time and apply it at any point in the model.|
|rivers| rivers , river |None|river = Votualevu_R.txt,1867430,1867455,3914065,3914090;<br>river = *Fluxfile*, *xstart*, *xend*, *ystart*, *yend*;|The river is added as a vertical discharge on a chosen area (the user input consisting in a Time serie and a rectangular area definition).<br>The whole cells containing the corners of the area will be included in the area, no horizontal velocity is applied.<br>To add multiple rivers, just add different lines in the input file (one by river).|
|il| il , Rain_il , initialloss |(see constant in parameters)|il=rain_loss.nc?initial_loss;|Initial Rain loss coefficient map (in mm)|
|cl| cl , Rain_cl , continuousloss |(see constant in parameters)|cl=rain_loss.nc?continuous_loss;|Continuous Rain loss coefficient map (in mm/h)|
|Wind| Wind , windfiles |None|Wind = mywind.nc?uw,mywind.nc?vw<br>Wind = MyWind.txt|Spacially varying: 2 files are given, 1st file is U wind and second is V wind ( no rotation of the data is performed)<br>Spacially uniform: 1 file is given then a 3 column file is expected, showing time, windspeed and direction.<br>Wind direction is rotated (later) to the grid direction (using grdalpha input parameter)|
|Atmp| Atmp , atmpfile |None|Atmp=AtmosphericPressure.nc?p| The forcing pressure is expected to be in Pa and the effect of the atmospheric pressure gradient is calculated as the difference to a reference pressure Paref, converted to a height using Pa2.|
|Rain| Rain , rainfile |None|rain=rain_forcing.txt <br>rain=rain_forcing.nc?RainIntensity| This allow to force a time varying, space varying rain intensity on the model, in mm/h.<br>Spacially varrying (rain map), a netcdf file is expected (with the variable associated to the rain after "?").<br>Spacially uniform: the rain is forced using a time serie using a 2 column values table containing time (not necessary unformly distributed) and rain.|
---

## List of the non-identified inputs

|_Reference_|_Keys_|
|---|---|
|Adaptation|Adaptation|
|crs| crs , spatialref , spatial_ref , wtk , crsinfo , crs_info |
|bathy|bathy|
|bathyfile|bathyfile|
|bathymetry|bathymetry|
|depfile|depfile|
|cavity|cavity|
---

*Note* : The keys are not case sensitive.

# Paramter and Forcing list for BG_Flood

BG_flood user interface consists in a text file, associating key words to user chosen parameters and forcings.
## List of the Parameters' input

|_Reference_|_Keys_|_default_|_Explanation_|
|----|---|---|---|
|test|test|-1|-1:no test; 0:some test; 1:test 0 and XX test|
|doubleprecision|doubleprecision|0||
|maxlevel|maxlevel|0||
|minlevel|minlevel|0||
|initlevel|initlevel|0||
|conserveElevation|conserveElevation|false||
|membuffer|membuffer|1.05|needs to allocate more memory than initially needed so adaptation can happen without memory reallocation|
|eps|eps|0.0001|//drying height in m|
|Cd|Cd|0.002|Wind drag coeff|
|Pa2m|Pa2m|0.00009916|if unit is hPa then user should use 0.009916;|
|Paref|Paref|101300.0|if unit is hPa then user should use 1013.0|
|mask|mask|9999.0|mask any zb above this value. if the entire Block is masked then it is not allocated in the memory|
|dt|dt|0.0|Model time step in s.|
|CFL|CFL|0.5|Current Freidrich Limiter|
|theta|theta|1.3|minmod limiter can be used to tune the momentum dissipation (theta=1 gives minmod, the most dissipative limiter and theta = 2 gives	superbee, the least dissipative).|
|endtime|endtime|0.0|Total runtime in s will be calculated based on bnd input as min(length of the shortest time series, user defined)|
|outfile|outfile|"Output.nc"|netcdf output file name|
|outvars|outvars|DD|CC|
|resetmax|resetmax|false||
|outishift|outishift|0||
|outjshift|outjshift|0||
|nx|nx|0|Initial grid size|
|ny|ny|0|Initial grid size|
|dx|dx|nan("")|grid resolution in the coordinate system unit in m.|
|grdalpha|grdalpha|nan("")|grid rotation Y axis from the North input in degrees but later converted to rad|
|xmax|xmax|nan("")||
|ymax|ymax|nan("")||
|g|g|false||
|rho|rho|1025.0|fluid density in kg/m-3|
|smallnc|smallnc|1|default save as short integer if smallnc=0 then save all variables as float|
|scalefactor|scalefactor|0.01f||
|addoffset|addoffset|0.0f||
|posdown|posdown|0|flag for bathy input. model requirement is positive up  so if posdown ==1 then zb=zb*-1.0f|
|use_catalyst|use_catalyst|0||
|catalyst_python_pipeline|catalyst_python_pipeline|0||
|vtk_output_frequency|vtk_output_frequency|0||
|vtk_output_time_interval|vtk_output_time_interval|1.0||
|vtk_outputfile_root|vtk_outputfile_root|"bg_out"||
|python_pipeline|python_pipeline|"coproc.py"||
|zsinit|zsinit|nan("")|init zs for cold start. if not specified by user and no bnd file =1 then sanity check will set to 0.0|
|zsoffset|zsoffset|nan("")||
|hotstartfile|hotstartfile|DD|CC|
|hotstep|hotstep|0|step to read if hotstart file has multiple steps|
|spherical|spherical|0|flag for geographical coordinate. can be activated by using teh keyword geographic|
|Radius|Radius|6371220.|Earth radius [m]|
|frictionmodel|frictionmodel|0||
---

## List of the Forcings' inputs

|_Reference_|_Keys_|_default_|_Example_|_Explanation_|
|----|---|---|---|---|
|cf|cf|01||om friction for flow model cf|
|left|left||||
|right|right||||
|top|top||||
|bot|bot||||
|deform|deform|tata|toto|Deform are maps to applie to both zs and zb; this is often co-seismic vertical deformation used to generate tsunami initial wave<br>Here you can spread the deformation across a certain amount of time and apply it at any point in the model|
---


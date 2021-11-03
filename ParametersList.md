# Paramter and Forcing list for BG_Flood
 
BG_flood user interface consists in a text file, associating key words to user chosen parameters and forcings.
 
## List of the Parameters
|_Reference_|_Keys_|_default_|_Explanation_|
|----|---|---|---|
|test|"test"| -1|-1:no test; 0:some test; 1:test 0 and XX test|
|GPUDEVICE|"GPUDEVICE","gpu"|0| 0: first available GPU; -1: CPU single core; 2+: other GPU|
|doubleprecision|"doubleprecision"| 0|	int doubleprecision = 0;|
|maxlevel|"maxlevel"| 0|	int maxlevel = 0;|
|minlevel|"minlevel"| 0|	int minlevel = 0;|
|initlevel|"initlevel"| 0|	int initlevel = 0;|
|conserveElevation|"conserveElevation"| false|	bool conserveElevation = false;|
|membuffer|"membuffer"| 1.05|needs to allocate more memory than initially needed so adaptation can happen without memory reallocation|
|eps|"eps"| 0.0001| |
|cf|"cf"|0.0001| bottom friction for flow model cf|
|Cd|"Cd"|0.002| Wind drag coeff|
|Pa2m|"Pa2m"| 0.00009916| if unit is hPa then user should use 0.009916;|
|Paref|"Paref"| 101300.0| if unit is hPa then user should use 1013.0|
|mask|"mask"| 9999.0|mask any zb above this value. if the entire Block is masked then it is not allocated in the memory|
|dt|"dt"|0.0| Model time step in s.|
|CFL|"CFL"|0.5| Current Freidrich Limiter|
|theta|"theta"|1.3| minmod limiter can be used to tune the momentum dissipation (theta=1 gives minmod, the most dissipative limiter and theta = 2 gives	superbee, the least dissipative).|
|outputtimestep|"outputtimestep","outtimestep","outputstep"|0.0|number of seconds between output 0.0 for none|
|endtime|"endtime"|0.0| Total runtime in s will be calculated based on bnd input as min(length of the shortest time series, user defined)|
|totaltime|"totaltime","inittime"| 0.0||
|outfile|"outfile"|"Output.nc"| netcdf output file name|
|outvars|"outvars"|||
|resetmax|"resetmax"| false|	bool resetmax = false;|
|outishift|"outishift"| 0|	int outishift = 0;|
|outjshift|"outjshift"| 0|	int outjshift = 0;|
|nx|"nx"|0| Initial grid size|
|ny|"ny"|0|Initial grid size|
|dx|"dx"| nan("")| grid resolution in the coordinate system unit in m.|
|grdalpha|"grdalpha"| nan("")| grid rotation Y axis from the North input in degrees but later converted to rad|
|xo|"xo","xmin"| nan("")| originally defined has nan to check whether the user alter these values when runing the model|
|yo|"yo","ymin"| nan("")| grid origin|
|xmax|"xmax"| nan("")|	double xmax = nan("");|
|ymax|"ymax"| nan("")|	double ymax = nan("");|
|g|"g"|9.81| Gravity in m.s-2|
|rho|"rho"|1025.0| fluid density in kg|
|smallnc|"smallnc"| 1|default save as short integer if smallnc=0 then save all variables as float|
|scalefactor|"scalefactor"| 0.01f|	float scalefactor = 0.01f;|
|addoffset|"addoffset"| 0.0f|	float addoffset = 0.0f;|
|posdown|"posdown"| 0| flag for bathy input. model requirement is positive up  so if posdown ==1 then zb=zb*-1.0f|
|use_catalyst|"use_catalyst"| 0|        int use_catalyst = 0;|
|catalyst_python_pipeline|"catalyst_python_pipeline"| 0|        int catalyst_python_pipeline = 0;|
|vtk_output_frequency|"vtk_output_frequency"| 0|        int vtk_output_frequency = 0;|
|vtk_output_time_interval|"vtk_output_time_interval"| 1.0|        double vtk_output_time_interval = 1.0;|
|vtk_outputfile_root|"vtk_outputfile_root"| "bg_out"|        std::string vtk_outputfile_root = "bg_out";|
|python_pipeline|"python_pipeline"| "coproc.py"|        std::string python_pipeline = "coproc.py";|
|zsinit|"zsinit"| nan("")|init zs for cold start. if not specified by user and no bnd file =1 then sanity check will set to 0.0|
|zsoffset|"zsoffset"| nan("")|	double zsoffset = nan("");|
|hotstartfile|"hotstartfile"|||
|hotstep|"hotstep"| 0|step to read if hotstart file has multiple steps|
|spherical|"spherical"| 0| flag for geographical coordinate. can be activated by using teh keyword geographic|
|Radius|"Radius"| 6371220.|Earth radius [m]|
|frictionmodel|"frictionmodel"|0||
|rivers|"rivers","river"|||
---
&nbsp;
 
## List of the Forcings
|_Reference_|_Keys_|_default_|_example_|_Explanation_|
|----|---|---|---|---|
|Bathy|"Bathy","bathyfile","bathymetry","depfile","depthfile","topofile","topo","DEM"|None|EEE|FFF|
|left|"left","leftbndfile","leftbnd"|None|EEE|FFF|
|right|"right","rightbndfile","rightbnd"|None|EEE|FFF|
|top|"top","topbndfile","topbnd"|None|EEE|FFF|
|bot|"bot","botbndfile","botbnd","bottom"|None|EEE|FFF|
|deform|"deform"|None|EEE|FFF|
|Atmp|"Atmp","atmpfile"|None|EEE|FFF|
|Rain|"Rain","rainfile"|None|EEE|FFF|
---
&nbsp;
 
## List of the Unidentificated entries
|_Reference_|_Keys_|
|----|---|
|velthreshold|"velthreshold","vthresh","vmax","velmax"|
|Adaptation|"Adaptation"|
|bathymetry|"bathymetry"|
|depfile|"depfile"|
|cfmap|"cfmap","roughnessmap"|
|Adaptation|"Adaptation"|
---
 
*Note* : The keys are not case sensitive.
# Paramter and Forcing list for BG_Flood
 
BG_flood user interface consists in a text file, associating key words to user chosen parameters and forcings.
 

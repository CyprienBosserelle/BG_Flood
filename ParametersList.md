__Paramter list for BG_Flood__
\n
|_Reference_|_Keys_| _Variable name_|_default_|_Explanation_|
|----|---|---|---|---|
|test|"test"|Param.test| -1|-1:no test; 0:some test; 1:test 0 and XX test
|GPUDEVICE|"GPUDEVICE","gpu"|Param.GPUDEVICE|0| 0: first available GPU; -1: CPU single core; 2+: other GPU
|doubleprecision|"doubleprecision"|Param.doubleprecision| 0|	int doubleprecision = 0;
|maxlevel|"maxlevel"|Param.maxlevel| 0|	int maxlevel = 0;
|minlevel|"minlevel"|Param.minlevel| 0|	int minlevel = 0;
|initlevel|"initlevel"|Param.initlevel| 0|	int initlevel = 0;
|conserveElevation|"conserveElevation"|Param.conserveElevation| false|	bool conserveElevation = false;
|membuffer|"membuffer"|Param.membuffer| 1.05|needs to allocate more memory than initially needed so adaptation can happen without memory reallocation
|eps|"eps"|Param.eps| 0.0001| 
|cf|"cf"|Param.cf|0.0001| bottom friction for flow model cf
|Cd|"Cd"|Param.Cd|0.002| Wind drag coeff
|Pa2m|"Pa2m"|Param.Pa2m| 0.00009916| if unit is hPa then user should use 0.009916;
|Paref|"Paref"|Param.Paref| 101300.0| if unit is hPa then user should use 1013.0
|mask|"mask"|Param.mask| 9999.0|mask any zb above this value. if the entire Block is masked then it is not allocated in the memory
|dt|"dt"|Param.dt|0.0| Model time step in s.
|CFL|"CFL"|Param.CFL|0.5| Current Freidrich Limiter
|theta|"theta"|Param.theta|1.3| minmod limiter can be used to tune the momentum dissipation (theta=1 gives minmod, the most dissipative limiter and theta = 2 gives	superbee, the least dissipative).
|outputtimestep|"outputtimestep","outtimestep","outputstep"|Param.outputtimestep|0.0|number of seconds between output 0.0 for none
|endtime|"endtime"|Param.endtime|0.0| Total runtime in s will be calculated based on bnd input as min(length of the shortest time series, user defined)
|totaltime|"totaltime","inittime"|Param.totaltime| 0.0|
|outfile|"outfile"|Param.outfile|"Output.nc"| netcdf output file name
|outvars|"outvars"|Param.outvars||
|resetmax|"resetmax"|Param.resetmax| false|	bool resetmax = false;
|outishift|"outishift"|Param.outishift| 0|	int outishift = 0;
|outjshift|"outjshift"|Param.outjshift| 0|	int outjshift = 0;
|nx|"nx"|Param.nx|0| Initial grid size
|ny|"ny"|Param.ny|0|Initial grid size
|dx|"dx"|Param.dx| nan("")| grid resolution in the coordinate system unit in m.
|grdalpha|"grdalpha"|Param.grdalpha| nan("")| grid rotation Y axis from the North input in degrees but later converted to rad
|xo|"xo","xmin"|Param.xo| nan("")| originally defined has nan to check whether the user alter these values when runing the model
|yo|"yo","ymin"|Param.yo| nan("")| grid origin
|xmax|"xmax"|Param.xmax| nan("")|	double xmax = nan("");
|ymax|"ymax"|Param.ymax| nan("")|	double ymax = nan("");
|g|"g"|Param.g|9.81| Gravity in m.s-2
|rho|"rho"|Param.rho|1025.0| fluid density in kg
|smallnc|"smallnc"|Param.smallnc| 1|default save as short integer if smallnc=0 then save all variables as float
|scalefactor|"scalefactor"|Param.scalefactor| 0.01f|	float scalefactor = 0.01f;
|addoffset|"addoffset"|Param.addoffset| 0.0f|	float addoffset = 0.0f;
|posdown|"posdown"|Param.posdown| 0| flag for bathy input. model requirement is positive up  so if posdown ==1 then zb=zb*-1.0f
|use_catalyst|"use_catalyst"|Param.use_catalyst| 0|        int use_catalyst = 0;
|catalyst_python_pipeline|"catalyst_python_pipeline"|Param.catalyst_python_pipeline| 0|        int catalyst_python_pipeline = 0;
|vtk_output_frequency|"vtk_output_frequency"|Param.vtk_output_frequency| 0|        int vtk_output_frequency = 0;
|vtk_output_time_interval|"vtk_output_time_interval"|Param.vtk_output_time_interval| 1.0|        double vtk_output_time_interval = 1.0;
|vtk_outputfile_root|"vtk_outputfile_root"|Param.vtk_outputfile_root| "bg_out"|        std::string vtk_outputfile_root = "bg_out";
|python_pipeline|"python_pipeline"|Param.python_pipeline| "coproc.py"|        std::string python_pipeline = "coproc.py";
|zsinit|"zsinit"|Param.zsinit| nan("")|init zs for cold start. if not specified by user and no bnd file =1 then sanity check will set to 0.0
|zsoffset|"zsoffset"|Param.zsoffset| nan("")|	double zsoffset = nan("");
|hotstartfile|"hotstartfile"|Param.hotstartfile||
|hotstep|"hotstep"|Param.hotstep| 0|step to read if hotstart file has multiple steps
|spherical|"spherical"|Param.spherical| 0| flag for geographical coordinate. can be activated by using teh keyword geographic
|Radius|"Radius"|Param.Radius| 6371220.|Earth radius [m]
|frictionmodel|"frictionmodel"|Param.frictionmodel|0|
|Adaptation|"Adaptation"|Nan.Adaptation|NNdef|NNNN
|bathy|"bathy","bathyfile","bathymetry","depfile","depthfile","topofile","topo","DEM"|Nan.bathy|NNdef|NNNN
|bathymetry|"bathymetry"|Nan.bathymetry|NNdef|NNNN
|depfile|"depfile"|Nan.depfile|NNdef|NNNN
|left|"left","leftbndfile","leftbnd"|Forcing.left|Def|FFF
|right|"right","rightbndfile","rightbnd"|Forcing.right|Def|FFF
|top|"top","topbndfile","topbnd"|Forcing.top|Def|FFF
|bot|"bot","botbndfile","botbnd","bottom"|Forcing.bot|Def|FFF
|deform|"deform"|Forcing.deform|Def|FFF
|rivers|"rivers","river"|Param.rivers||
|cfmap|"cfmap","roughnessmap"|Nan.cfmap|NNdef|NNNN
|Atmp|"Atmp","atmpfile"|Forcing.Atmp|Def|FFF
|Rain|"Rain","rainfile"|Forcing.Rain|Def|FFF
|Adaptation|"Adaptation"|Nan.Adaptation|NNdef|NNNN
 
*Note* : The keys are not case sensitive.

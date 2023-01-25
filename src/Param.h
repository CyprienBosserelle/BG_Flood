#ifndef PARAM_H
#define PARAM_H

#include "General.h"
#include "Input.h"

/**
 *  A class. A class for holding model parameters.
 */
class Param {
public:

	//*General parameters
	int test = -1; //-1: no test, 99: run all independent tests, X: run test X
	double g = 9.81; // Gravity in m.s-2
	double rho = 1025.0; // Fluid density in kg.m-3
	double eps = 0.0001; // Drying height in m (if h<eps, the surface is concidered dry)
	double dt = 0.0; // Model time step in s.
	double CFL = 0.5; // Current Freidrich Limiter
	double theta = 1.3; // Minmod limiter parameter, theta in [1,2]. <br>Can be used to tune the momentum dissipation (theta=1 gives minmod the most dissipative limiter and theta = 2 gives	superbee, the least dissipative).
	double VelThreshold = -1.0; // Using Velocity threshold if the the velocuity exceeds that threshold. Advice value of 16.0 to use or negative value (-1) to turn off
	int frictionmodel = 0; // Bottom friction model (-1: Manning model, 0: quadratic, 1: Smart model)
	double cf = 0.0001; // Bottom friction coefficient for flow model (if constant)
	double Cd = 0.002; // Wind drag coefficient
	bool windforcing = false; //not working yet
	bool atmpforcing = false;
	bool rainforcing = false;
	bool infiltration = false;

	bool conserveElevation = false; //Switch to force the conservation of zs instead of h at the interface between coarse and fine blocks
	bool wetdryfix = true; // Switch to remove wet/dry instability (i.e. true reoves instability and false leaves the model as is)

	bool leftbnd = false; // bnd is forced (i.e. not a wall or neuman)
	bool rightbnd = false; // bnd is forced (i.e. not a wall or neuman)
	bool topbnd = false; // bnd is forced (i.e. not a wall or neuman)
	bool botbnd = false; // bnd is forced (i.e. not a wall or neuman)

	double Pa2m = 0.00009916; // Conversion between atmospheric pressure changes to water level changes in Pa (if unit is hPa then user should use 0.009916)
	double Paref = 101300.0; // Reference pressure in Pa (if unit is hPa then user should use 1013.0)
	double lat = 0.0; // Model latitude. This is ignored in spherical case
	int GPUDEVICE = 0; // 0: first available GPU, -1: CPU single core, 2+: other GPU

	int doubleprecision = 0; // 0: float precision, 1: double precision

	int engine = 1; // 1: Buttinger-Kreuzhuber et al. 2019, 2: Kurganov (Popinet 2011), 3: KurganovATMP same as Kurganov but with atmospheric forcing terms 

	//*Grid parameters
	double dx = nan(""); // Grid resolution in the coordinate system unit in m.
	double delta; // Grid resolution for the model. in Spherical coordinates this is dx * Radius*pi / 180.0
	int nx = 0; // Initial grid size in x direction
	int ny = 0; //Initial grid size in y direction
	int nblk = 0; // Number of compute blocks
	int blkwidth = 16; //Block width in number of cells
	int blkmemwidth = 0; // Calculated in sanity check as blkwidth+2*halowidth
	int blksize = 0; // Calculated in sanity check as blkmemwidth*blkmemwidth
	int halowidth = 1; // Use a halo around the blocks default is 1 cell: the memory for each blk is 18x18 when blkwidth is 16

	double xo = nan(""); // Grid x origin (if not alter by the user, will be defined based on the topography/bathymetry input map)
	double yo = nan(""); // Grid y origin (if not alter by the user, will be defined based on the topography/bathymetry input map)
	double ymax = nan(""); // Grid ymax (if not alter by the user, will be defined based on the topography/bathymetry input map)
	double xmax = nan(""); // Grid xmax (if not alter by the user, will be defined based on the topography/bathymetry input map)
	double grdalpha = nan(""); // Grid rotation Y axis from the North input in degrees but later converted to rad
	int posdown = 0; // Flag for bathy input. Model requirement is positive up  so if posdown ==1 then zb=zb*-1.0f
	bool spherical = 0; // Flag for sperical coordinate (still in development)
	double Radius = 6371220.; //Earth radius [m]
	double mask = 9999.0; //Mask any zb above this value. If the entire Block is masked then it is not allocated in the memory

	//*Adaptation
	int initlevel = 0; //Initial level of grid adaptation (based on dx if defined by the user or on the resolution of the topography/bathymetry input)
	int maxlevel = 0; //Maximum level for grid adaptation (overwrite the adaptation map if use) 
	int minlevel = 0; //Minumim level for grid adaptation (overwrite the adaptation map if use) 
	int nblkmem = 0;
	int navailblk = 0;
	double membuffer = 1.05; //Needs to allocate more memory than initially needed so adaptation can happen without memory reallocation



	//*Timekeeping
	double outputtimestep = 0.0; //Number of seconds between netCDF outputs, 0.0 for none
	double endtime = 0.0; // Total runtime in s, will be calculated based on bnd input as min(length of the shortest time series, user defined) and should be shorter than any time-varying forcing
	double totaltime = 0.0; // Total simulation time in s
	double dtinit = -1; // Maximum initial time steps in s (should be positive, advice 0.1 if dry domain initialement) 
	double dtmin = 0.0005; //Minimum accepted time steps in s (a lower value will be concidered a crash of the code, and stop the run)

	//* Initialisation
	double zsinit = nan(""); //Init zs for cold start in m. If not specified by user and no bnd file = 1 then sanity check will set it to 0.0

	double zsoffset = nan(""); //Add a water level offset in m to initial conditions and boundaries (0.0 by default)

	std::string hotstartfile;
	/*Allow to hotstart (or restart) the computation providing a netcdf file containing at least zb, h or zs, u and v
	Default: None
	*/
	//std::string deformfile;
	int hotstep = 0; //Step to read if hotstart file has multiple steps (step and not (computation) time)
	//other
	clock_t startcputime, endcputime, setupcputime;
	size_t GPU_initmem_byte, GPU_totalmem_byte;


	//*Outputs
	//std::string Bathymetryfile;// bathymetry file name
	//inputmap Bathymetry;


	//Timeseries output (save as a vector containing information for each Time Serie output)
	std::vector<TSoutnode> TSnodesout; 
	/*Time serie output, giving a file name and a (x,y) position 
	(which will be converted to nearest grid position). 
	This keyword can be used multiple times to extract time series at different locations.
	The data is stocked for each timestep and written by flocs.
	The resulting file contains (t,zs,h,u,v)
	Example: "TSnodesout = Offshore.txt,3101.00,4982.57" (*filename,x,y*)
	Default: None
	*/

	std::string outfile = "Output.nc"; // netcdf output file name
	std::vector<std::string> outvars; 
	/*List of names of the variables to output (for 2D maps)
	Supported variables = "zb", "zs", "u", "v", "h", "hmean", "zsmean", "umean", "vmean", "hUmean", "Umean", "hmax", "zsmax", "umax", "vmax", "hUmax", "Umax", "twet", "dhdx","dhdy","dzsdx","dzsdy","dudx","dudy","dvdx","dvdy","Fhu","Fhv","Fqux","Fqvy","Fquy","Fqvx","Su","Sv","dh","dhu","dhv","cf","Patm", "datmpdx","datmpdy","il","cl","hgw";
	Default: "zb", "zs", "u", "v", "h"
	*/
	double wet_threshold = 0.1; //in m. Limit to consider a cell wet for the twet output (duration of inundation (s))
	
	std::vector<outzoneP> outzone; 
	/*Zoned output (netcdf file), giving a file name and the position of two corner points
	(which will be converted to a rectagle containing full blocks).
	This keyword can be used multiple times to output maps of different areas.
	Example: "outzone=zoomed.nc,5.3,5.4,0.5,0.8;" (*filename,x1,x2,y1,y2*)
	Default: Full domain
	*/

	int maxTSstorage = 16384; //maximum strorage (nTSnodes*4*nTSsteps) before time series output are flushed to disk [2^14]




	// Output switch controls
	bool resetmax = false; //Switch to reset the "max" outputs after each output
	bool outmax = false;
	bool outmean = false;
	//bool outvort = false;
	bool outtwet = false;
	//bool outU = false;

	// WARNING FOR DEBUGGING PURPOSE ONLY
// For debugging one can shift the output by 1 or -1 in the i and j direction.
// this will save the value in the halo to the output file allowing debugging of values there.
	int outishift = 0; //DEBUGGING ONLY: allow cell shift (1 or -1) in x direction to visualise the halo around blocks in the output 
	int outjshift = 0; //DEBUGGING ONLY: allow cell shift (1 or -1) in y direction to visualise the halo around blocks in the output 


	//Rivers
	//std::vector<River> Rivers; // empty vector to hold river location and discharge time series
	int nrivers = 0;
	int nblkriver = 0;

	// length of bnd blk, redundant from XForcing but useful
	int nbndblkleft = 0;
	int nbndblkright = 0;
	int nbndblktop = 0;
	int nbndblkbot = 0;

	int nmaskblk = 0;



	//*Netcdf parameters
	int smallnc = 1; //Short integer conversion for netcdf outputs. 1: save as short integer for the netcdf file, if 0 then save all variables as float
	float scalefactor = 0.01f; //Scale factor used for the short integer conversion for netcdf outputs
	float addoffset = 0.0f; //Offset add during the short integer conversion for netcdf outputs

#ifdef USE_CATALYST
        //* ParaView Catalyst parameters (SPECIAL USE WITH PARAVIEW)
        int use_catalyst = 0; // Switch to use ParaView Catalyst
        int catalyst_python_pipeline = 0; //Pipeline to use ParaView Catalyst
        int vtk_output_frequency = 0; // Output frequency for ParaView Catalyst
        double vtk_output_time_interval = 1.0; // Output time step for ParaView Catalyst
        std::string vtk_outputfile_root = "bg_out"; //output file name for ParaView Catalyst
        std::string python_pipeline = "coproc.py"; //python pipeline for ParaView Catalyst
#endif




	// info of the mapped cf
	//inputmap roughnessmap;

	//forcingmap windU;
	//forcingmap windV;
	//forcingmap atmP;
	//forcingmap Rainongrid;

	// deformation forcing for tsunami generation
	//std::vector<deformmap> deform;
	double deformmaxtime = 0.0; // time (s) after which no deformation occurs (internal parameter to cut some of the loops)
	bool rainbnd = false; // when false it force the rain foring on the bnd cells to be null.

	// This here should be stored in a structure at a later stage

	std::string AdaptCrit;
	int* AdaptCrit_funct_pointer;

	std::string Adapt_arg1, Adapt_arg2, Adapt_arg3, Adapt_arg4, Adapt_arg5;
	int adaptmaxiteration = 20; // Maximum number of iteration for adaptation. default 20

};



// End of global definition
#endif
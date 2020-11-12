
#ifndef PARAM_H
#define PARAM_H

#include "General.h"
#include "input.h"

class Param {
public:
	//general parameters
	int test = -1;//-1:no test; 0:some test; 1:test 0 and XX test
	double g=9.81; // Gravity
	double rho=1025.0; // fluid density
	double eps= 0.0001; // //drying height in m
	double dt=0.0; // Model time step in s.
	double CFL=0.5; // Current Freidrich Limiter
	double theta=1.3; // minmod limiter can be used to tune the momentum dissipation (theta=1 gives minmod, the most dissipative limiter and theta = 2 gives	superbee, the least dissipative).
	int frictionmodel=0; // 
	double cf=0.0001; // bottom friction for flow model cf 
	double Cd=0.002; // Wind drag coeff
	bool windforcing = false;
	bool atmpforcing = false;
	bool rainforcing = false;

	bool leftbnd = false; // bnd is forced (i.e. not a wall or neuman)
	bool rightbnd = false; // bnd is forced (i.e. not a wall or neuman)
	bool topbnd = false; // bnd is forced (i.e. not a wall or neuman)
	bool botbnd = false; // bnd is forced (i.e. not a wall or neuman)

	double Pa2m = 0.00009916; // if unit is hPa then user should use 0.009916;
	double Paref = 101300.0; // if unit is hPa then user should use 1013.0 
	double lat = 0.0; // Model latitude. This is ignored in spherical case
	int GPUDEVICE=0; // 0: first available GPU; -1: CPU single core; 2+: other GPU

	int doubleprecision = 0;

	//grid parameters
	double dx= nan(""); // grid resolution in the coordinate system unit. 
	double delta; // grid resolution for the model. in Spherical coordinates this is dx * Radius*pi / 180.0
	int nx=0; // Initial grid size
	int ny=0; //Initial grid size
	int nblk=0; // number of compute blocks
	int blkwidth = 16;
	int blkmemwidth = 0; // Calculated in sanity check as blkwidth+2*halowidth
	int blksize = 0; // Calculated in sanity check as blkmemwidth*blkmemwidth
	int halowidth = 1; // use a halo around the blocks default is 1 cell: the memory for each blk is 18x18 when blkwidth is 16

	double xo = nan(""); // originally defined has nan to check whether the user alter these values when runing the model
	double yo = nan(""); // grid origin
	double ymax = nan("");
	double xmax = nan("");
	double grdalpha= nan(""); // grid rotation Y axis from the North input in degrees but later converted to rad
	int posdown = 0; // flag for bathy input. model requirement is positive up  so if posdown ==1 then zb=zb*-1.0f
	int spherical = 0; // flag for geographical coordinate. can be activated by using teh keyword geographic
	double Radius = 6371220.; //Earth radius [m]

	//Adaptation
	int initlevel = 0;
	int maxlevel = 0;
	int minlevel = 0;
	int nblkmem = 0;
	int navailblk = 0;
	double membuffer = 1.05; //needs to allocate more memory than initially needed so adaptation can happen without memory reallocation

	double mask = 9999.0; //mask any zb above this value. if the entire Block is masked then it is not allocated in the memory
	//files
	//std::string Bathymetryfile;// bathymetry file name
	//inputmap Bathymetry;
	std::string outfile="Output.nc"; // netcdf output file name
	
	//Timekeeping
	double outputtimestep=0.0; //number of seconds between output 0.0 for none
	double endtime=0.0; // Total runtime in s will be calculated based on bnd input as min(length of the shortest time series, user defined)
	double totaltime = 0.0; //

	//Timeseries output
	std::vector<TSoutnode> TSnodesout; // vector containing i and j of each variables
	int maxTSstorage = 16384; //maximum strorage (nTSnodes*4*nTSsteps) before time series output are flushed to disk [2^14]

	std::vector<std::string> outvars; //list of names of teh variables to output


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

	//hot start
	double zsinit = nan(""); //init zs for cold start. if not specified by user and no bnd file =1 then sanity check will set to 0.0

	//Add a water level offset to initial conditions and bnds
	double zsoffset = nan("");


	std::string hotstartfile;
	//std::string deformfile;
	int hotstep = 0; //step to read if hotstart file has multiple steps
	//other
	clock_t startcputime, endcputime, setupcputime;
	
	//Netcdf parameters
	int smallnc = 1;//default save as short integer if smallnc=0 then save all variables as float
	float scalefactor = 0.01f;
	float addoffset = 0.0f;

#ifdef USE_CATALYST
        // ParaView Catalyst parameters
        int use_catalyst = 0;
        int catalyst_python_pipeline = 0;
        int vtk_output_frequency = 0;
        double vtk_output_time_interval = 1.0;
        std::string vtk_outputfile_root = "bg_out";
        std::string python_pipeline = "coproc.py";
#endif

	// Output switch controls
	bool resetmax = false;
	bool outmax = false;
	

	bool outmean = false;
	

	bool outvort = false;

	// info of the mapped cf
	//inputmap roughnessmap;

	//forcingmap windU;
	//forcingmap windV;
	//forcingmap atmP;
	//forcingmap Rainongrid;

	// deformation forcing for tsunami generation
	//std::vector<deformmap> deform;
	double deformmaxtime = 0.0; // time after which no deformation occurs (internal parameter to cut some of the loops)


	// This here should be stored in a structure at a later stage
	std::string AdatpCrit;
	int* AdatpCrit_funct_pointer;
	std::string Adapt_arg1, Adapt_arg2, Adapt_arg3, Adapt_arg4, Adapt_arg5;


};



// End of global definition
#endif

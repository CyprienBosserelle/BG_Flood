
#ifndef CSVGPU_H
#define CSVGPU_H
// includes, system

#define pi 3.14159265

#define epsilone 1e-30

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <map>
#include <netcdf.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

class TSnode {
public:
	int i, j, block;
	double x, y;
};

class Flowin {
public:
	double time, q;
};

class River {
public:
	std::vector<int> i, j, block; // one river can spring across multiple cells
	double disarea; // discharge area
	double xstart,xend, ystart,yend; // location of the discharge as a rectangle
	std::string Riverflowfile; // river flow input time[s] flow in m3/s
	std::vector<Flowin> flowinput; // vector to store the data of the river flow input file
	
};

class inputmap {
public:
	int nx = 0;
	int ny= 0;
	double xo = 0.0;
	double yo = 0.0;
	double xmax = 0.0;
	double ymax = 0.0;
	double dx = 0.0;
	double grdalpha=0.0;
	std::string inputfile;
};

class SLTS {
public:
	double time;
	std::vector<double> wlevs;


};

class Windin {
public:
	double time;
	double wspeed;
	double wdirection;
	double uwind;
	double vwind;


};

class forcingmap {
public:
	int nx, ny;
	int nt;
	int uniform = 0;
	double to, tmax;
	double xo, yo;
	double xmax, ymax;
	double dx;
	double dt;
	std::string inputfile;
	std::vector<Windin> data; // only used if uniform forcing

};



class bndparam {
public:
	std::vector<SLTS> data;
	bool on = false;
	int type = 1; // 0:Wall (no slip); 1:neumann (zeros gredient) [Default]; 2:sealevel dirichlet; 3: Absorbing 1D 4: Absorbing 2D (not yet implemented)
	std::string inputfile;
	int nblk = 0; //number of blocks where this bnd applies
	int side = 0; // 0: top bnd, 1: rightbnd, 2: bot bnd, 3, Left bnd
};

class Param {
public:
	//general parameters
	double g=9.81; // Gravity
	double rho=1025.0; // fluid density
	double eps= 0.0001; // //drying height in m
	double dt=0.0; // Model time step in s.
	double CFL=0.5; // Current Freidrich Limiter
	double theta=1.3; // minmod limiter can be used to tune the momentum dissipation (theta=1 gives minmod, the most dissipative limiter and theta = 2 gives	superbee, the least dissipative).
	int frictionmodel=0; // 
	double cf=0.0001; // bottom friction for flow model cf 
	double Cd=0.002; // Wind drag coeff
	double Pa2m = 0.00009916; // if unit is hPa then user should use 0.009916;
	double Paref = 101300.0; // if unit is hPa then user should use 1013.0 
	double lat = 0.0; // Model latitude. This is ignored in spherical case
	int GPUDEVICE=0; // 0: first available GPU; -1: CPU single core; 2+: other GPU

	int doubleprecision = 0;

	//grid parameters
	double dx=0.0; // grid resolution in the coordinate system unit. 
	double delta; // grid resolution for the model. in Spherical coordinates this is dx * Radius*pi / 180.0
	int nx=0; // Initial grid size
	int ny=0; //Initial grid size
	int nblk=0; // number of compute blocks
	int blksize = 256; //16x16 blocks
	double xo = 0.0; // grid origin
	double yo = 0.0; // grid origin
	double ymax = 0.0;
	double xmax = 0.0;
	double grdalpha=0.0; // grid rotation Y axis from the North input in degrees but later converted to rad
	int posdown = 0; // flag for bathy input. model requirement is positive up  so if posdown ==1 then zb=zb*-1.0f
	int spherical = 0; // flag for geographical coordinate. can be activated by using teh keyword geographic
	double Radius = 6371220.; //Earth radius [m]

	double mask = 9999.0; //mask any zb above this value. if the entire Block is masked then it is not allocated in the memory
	//files
	//std::string Bathymetryfile;// bathymetry file name
	inputmap Bathymetry;
	std::string outfile="Output.nc"; // netcdf output file name
	
	//Timekeeping
	double outputtimestep=0.0; //number of seconds between output 0.0 for none
	double endtime=0.0; // Total runtime in s will be calculated based on bnd input as min(length of the shortest time series, user defined)
	double totaltime = 0.0; //

	//Timeseries output
	std::vector<std::string> TSoutfile; //filename of output time series (Only save time, H U,V and zs)
	std::vector<TSnode> TSnodesout; // vector containing i and j of each variables
									//Output variables
	std::vector<std::string> outvars; //list of names of teh variables to output


	//Rivers
	std::vector<River> Rivers; // empty vector to hold river location and discharge time series
	int nriverblock = 0;

	//bnd
	
	
	bndparam rightbnd;
	bndparam leftbnd;
	bndparam topbnd;
	bndparam botbnd;
	


	//hot start
	double zsinit = -999.0; //init zs for cold start. if not specified by user and no bnd file =1 then sanity check will set to 0.0

	std::string hotstartfile;
	std::string deformfile;
	int hotstep = 0; //step to read if hotstart file has multiple steps
	//other
	clock_t startcputime, endcputime;
	
	//Netcdf parameters
	int smallnc = 1;//default save as short integer if smallnc=0 then save all variables as float
	float scalefactor = 0.01f;
	float addoffset = 0.0f;

	// This is controlled by the sanity checker not directly by the user
	int resetmax = 0;
	int outhhmax = 0;
	int outzsmax = 0;
	int outuumax = 0;
	int outvvmax = 0;

	int outhhmean = 0;
	int outzsmean = 0;
	int outuumean = 0;
	int outvvmean = 0;

	int outvort = 0;

	// info of the mapped cf
	inputmap roughnessmap;

	forcingmap windU;
	forcingmap windV;
	forcingmap atmP;
	forcingmap Rainongrid;
};






class Pointout {
public:
	double time, zs, hh, uu,vv;
};


extern double epsilon;
//extern double g;
//extern double rho;
//extern double eps;
//extern double CFL;
//
//extern std::string outfile;
////Output variables
//extern std::vector<std::string> outvars; //list of names of the variables to output
//
//// Map arrays: this is to link variable name as a string to the pointers holding the data
//extern std::map<std::string, float *> OutputVarMapCPU;
//
//
//extern double dt, dx;
//extern int nx, ny;
//
//extern double delta;

extern double *x, *y;
extern double *x_g, *y_g;

extern float *zs, *hh, *zb, *uu, *vv;
extern double *zs_d, *hh_d, *zb_d, *uu_d, *vv_d; // double array only allocated instead of thge float if requested
extern float *zs_g, *hh_g, *zb_g, *uu_g, *vv_g; // for GPU
extern double *zs_gd, *hh_gd, *zb_gd, *uu_gd, *vv_gd; // double array for GPU

extern float *zso, *hho, *uuo, *vvo;
extern double *zso_d, *hho_d, *uuo_d, *vvo_d;
extern float *zso_g, *hho_g, *uuo_g, *vvo_g; // for GPU
extern double *zso_gd, *hho_gd, *uuo_gd, *vvo_gd;

extern float * dhdx, *dhdy, *dudx, *dudy, *dvdx, *dvdy;
extern float *dzsdx, *dzsdy;

extern double * dhdx_d, *dhdy_d, *dudx_d, *dudy_d, *dvdx_d, *dvdy_d;
extern double *dzsdx_d, *dzsdy_d;

//GPU
extern float * dhdx_g, *dhdy_g, *dudx_g, *dudy_g, *dvdx_g, *dvdy_g;
extern float *dzsdx_g, *dzsdy_g;
extern double * dhdx_gd, *dhdy_gd, *dudx_gd, *dudy_gd, *dvdx_gd, *dvdy_gd;
extern double *dzsdx_gd, *dzsdy_gd;
//double *fmu, *fmv;

extern float *Su, *Sv, *Fqux, *Fquy, *Fqvx, *Fqvy;
extern float * Fhu, *Fhv;
extern float * dh, *dhu, *dhv;

extern double *Su_d, *Sv_d, *Fqux_d, *Fquy_d, *Fqvx_d, *Fqvy_d;
extern double * Fhu_d, *Fhv_d;
extern double * dh_d, *dhu_d, *dhv_d;

//GPU
extern float *Su_g, *Sv_g, *Fqux_g, *Fquy_g, *Fqvx_g, *Fqvy_g;
extern float * Fhu_g, *Fhv_g;
extern float * dh_g, *dhu_g, *dhv_g;

extern double *Su_gd, *Sv_gd, *Fqux_gd, *Fquy_gd, *Fqvx_gd, *Fqvy_gd;
extern double * Fhu_gd, *Fhv_gd;
extern double * dh_gd, *dhu_gd, *dhv_gd;

extern float * hhmean, *uumean, *vvmean, *zsmean;
extern float * hhmean_g, *uumean_g, *vvmean_g, *zsmean_g;

extern double * hhmean_d, *uumean_d, *vvmean_d, *zsmean_d;
extern double * hhmean_gd, *uumean_gd, *vvmean_gd, *zsmean_gd;

extern float * hhmax, *uumax, *vvmax, *zsmax;
extern float * hhmax_g, *uumax_g, *vvmax_g, *zsmax_g;

extern double * hhmax_d, *uumax_d, *vvmax_d, *zsmax_d;
extern double * hhmax_gd, *uumax_gd, *vvmax_gd, *zsmax_gd;

extern float * vort, *vort_g;// Vorticity output
extern double * vort_d, *vort_gd;// Vorticity output

extern float dtmax;
extern float * dtmax_g;
extern float * TSstore, *TSstore_g;
extern float * dummy;

extern double dtmax_d;
extern double * dtmax_gd;
extern double * TSstore_d, *TSstore_gd;
extern double * dummy_d;


extern float * cf;
extern float * cf_g;
extern double * cf_d;
extern double * cf_gd;

// wind array storage
extern float * Uwind, *Uwbef, *Uwaft;
extern float * Vwind, *Vwbef, *Vwaft;
extern float *PatmX, *Patmbef, *Patmaft;
extern float * Patm, *dPdx, *dPdy;
extern double * Patm_d, *dPdx_d, *dPdy_d;

extern float * Uwind_g, *Uwbef_g, *Uwaft_g;
extern float * Vwind_g, *Vwbef_g, *Vwaft_g;
extern float * PatmX_g, *Patmbef_g, *Patmaft_g;

extern float * Patm_g, *dPdx_g, *dPdy_g;
extern double * Patm_gd, *dPdx_gd, *dPdy_gd;

//rain on grid
// Need double for computation but not for input if it is in mm/hrs rather than m/s as expected if we were to respect IS units
extern float *Rain, *Rainbef, *Rainaft;
extern float *Rain_g, *Rainbef_g, *Rainaft_g;

// Block info
extern double * blockxo_d, *blockyo_d;
extern float * blockxo, *blockyo;
extern int * leftblk, *rightblk, *topblk, *botblk;
extern int * bndleftblk, *bndrightblk, *bndtopblk, *bndbotblk;

extern float * blockxo_g, *blockyo_g;
extern double * blockxo_gd, *blockyo_gd;
extern int * leftblk_g, *rightblk_g, *topblk_g, *botblk_g;
extern int * bndleftblk_g, *bndrightblk_g, *bndtopblk_g, *bndbotblk_g;

// id of blocks with riverinput:
extern int * Riverblk, *Riverblk_g;


//Cuda Array to pre-store Water level boundary on the GPU and interpolate through the texture fetch
extern cudaArray* leftWLS_gp; // Cuda array to pre-store HD vel data before converting to textures
extern cudaArray* rightWLS_gp;
extern cudaArray* topWLS_gp;
extern cudaArray* botWLS_gp;

// store wind data in cuda array before sending to texture memory
extern cudaArray* Uwind_gp;
extern cudaArray* Vwind_gp;
extern cudaArray* Patm_gp;
extern cudaArray* Rain_gp;

// Below create channels between cuda arrays (see above) and textures
extern cudaChannelFormatDesc channelDescleftbnd;// = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
extern cudaChannelFormatDesc channelDescrightbnd;// = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
extern cudaChannelFormatDesc channelDescbotbnd;// = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
extern cudaChannelFormatDesc channelDesctopbnd;// = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

extern cudaChannelFormatDesc channelDescUwind; 
extern cudaChannelFormatDesc channelDescVwind; 
extern cudaChannelFormatDesc channelDescPatm;
extern cudaChannelFormatDesc channelDescRain;

template <class T> T sq(T a);

template <class T> const T& max(const T& a, const T& b);
template <class T> const T& min(const T& a, const T& b);

template <class T> void carttoBUQ(int nblk, int nx, int ny, double xo, double yo, double dx, double* blockxo, double* blockyo, T * zb, T *&zb_buq);
template <class T> void interp2BUQ(int nblk, double blksize, double blkdx, double* blockxo, double* blockyo, int nx, int ny, double xo, double xmax, double yo, double ymax, double dx, T * zb, T *&zb_buq);

//General CPU functions //Unecessary to declare here?
//void mainloopCPU(Param XParam, std::vector<SLTS> leftWLbnd, std::vector<SLTS> rightWLbnd, std::vector<SLTS> topWLbnd, std::vector<SLTS> botWLbnd);
double FlowCPU(Param XParam, double nextoutputtime);
double FlowCPUDouble(Param XParam, double nextoutputtime);
double FlowCPUSpherical(Param XParam, double nextoutputtime);
double FlowCPUATM(Param XParam, double nextoutputtime, int cstwind, int cstpress, float Uwindi, float Vwindi);
//float demoloopCPU(Param XParam);
template <class T>  void setedges(int nblk, int * leftblk, int *rightblk, int * topblk, int* botblk, T *&zb);
//void setedges(int nblk, int * leftblk, int *rightblk, int * topblk, int* botblk, double *&zb);
//void setedges(int nblk, int * leftblk, int *rightblk, int * topblk, int* botblk, float *&zb);

void update(int nx, int ny, float theta, float dt, float eps, float g, float CFL, float delta, float *hh, float *zs, float *uu, float *vv, float *&dh, float *&dhu, float *&dhv);
//void advance(int nx, int ny, float dt, float eps, float *hh, float *zs, float *uu, float * vv, float * dh, float *dhu, float *dhv, float * &hho, float *&zso, float *&uuo, float *&vvo);
//void cleanup(int nx, int ny, float * hhi, float *zsi, float *uui, float *vvi, float * &hho, float *&zso, float *&uuo, float *&vvo);
template <class T> void advance(int nx, int ny, T dt, T eps, T*zb, T *hh, T *zs, T *uu, T * vv, T * dh, T *dhu, T *dhv, T * &hho, T *&zso, T *&uuo, T *&vvo);
template <class T> void cleanup(int nx, int ny, T * hhi, T *zsi, T *uui, T *vvi, T * &hho, T *&zso, T *&uuo, T *&vvo);
template <class T> void gradient(int nblk, int blksize, T theta, T delta, int * leftblk, int * rightblk, int * topblk, int * botblk, T *a, T *&dadx, T * &dady);



//Bnd functions
//void neumannbnd(int nx, int ny, double*a);

void leftdirichletCPU(int nblk, int blksize, float xo, float yo, float g, float dx, std::vector<double> zsbndvec, float * blockxo, float * blockyo, float *zs, float *zb, float *hh, float *uu, float *vv);
void rightdirichletCPU(int nblk, int blksize, int nx, float xo, float yo, float g, float dx, std::vector<double> zsbndvec, float * blockxo, float * blockyo, float *zs, float *zb, float *hh, float *uu, float *vv);
void topdirichletCPU(int nblk, int blksize, int ny, float xo, float yo, float g, float dx, std::vector<double> zsbndvec, float * blockxo, float * blockyo, float *zs, float *zb, float *hh, float *uu, float *vv);
void botdirichletCPU(int nblk, int blksize, int ny, float xo, float yo, float g, float dx, std::vector<double> zsbndvec, float * blockxo, float * blockyo, float *zs, float *zb, float *hh, float *uu, float *vv);

void leftdirichletCPUD(int nblk, int blksize, double xo, double yo, double g, double dx, std::vector<double> zsbndvec, double * blockxo, double * blockyo, double *zs, double *zb, double *hh, double *uu, double *vv);
void rightdirichletCPUD(int nblk, int blksize, int nx, double xo, double yo, double g, double dx, std::vector<double> zsbndvec, double * blockxo, double * blockyo, double *zs, double *zb, double *hh, double *uu, double *vv);
void topdirichletCPUD(int nblk, int blksize, int ny, double xo, double yo, double g, double dx, std::vector<double> zsbndvec, double * blockxo, double * blockyo, double *zs, double *zb, double *hh, double *uu, double *vv);
void botdirichletCPUD(int nblk, int blksize, int ny, double xo, double yo, double g, double dx, std::vector<double> zsbndvec, double * blockxo, double * blockyo, double *zs, double *zb, double *hh, double *uu, double *vv);

void noslipbndLCPU(Param XParam);
void noslipbndRCPU(Param XParam);
void noslipbndTCPU(Param XParam);
void noslipbndBCPU(Param XParam);

//template <class T> void noslipbndallCPU(int nx, int ny, T dt, T eps, T *zb, T *zs, T *hh, T *uu, T *vv);
template <class T> void noslipbndLeftCPU(int nblk, int blksize, T xo, T eps, T* blockxo, T *zb, T *zs, T *hh, T *uu, T *vv);
template <class T> void noslipbndRightCPU(int nblk, int blksize, T xo, T xmax,T eps, T dx, T* blockxo, T *zb, T *zs, T *hh, T *uu, T *vv);
template <class T> void noslipbndBotCPU(int nblk, int blksize, T yo, T eps, T* blockyo, T *zb, T *zs, T *hh, T *uu, T *vv);
template <class T> void noslipbndTopCPU(int nblk, int blksize, T yo, T ymax, T eps, T dx, T* blockyo, T *zb, T *zs, T *hh, T *uu, T *vv);






template <class T> void bottomfrictionCPU(int nblk, int blksize, int smart, T dt, T eps, T* cf, T *hh, T *uu, T *vv);
template <class T> void discharge_bnd_v_CPU(Param XParam, T*zs, T*hh);
// CPU mean max
void AddmeanCPU(Param XParam);
void AddmeanCPUD(Param XParam);
void DivmeanCPU(Param XParam, float nstep);
void DivmeanCPUD(Param XParam, float nstep);
void ResetmeanCPU(Param XParam);
void ResetmeanCPUD(Param XParam);
void maxallCPU(Param XParam);
void maxallCPUD(Param XParam);
void CalcVort(Param XParam);
void CalcVortD(Param XParam);

//Output functions
extern "C" void create2dnc(int nx, int ny, double dx, double dy, double totaltime, double *xx, double *yy, float * var);
extern "C" void write2varnc(int nx, int ny, double totaltime, float * var);

// Netcdf functions
Param creatncfileUD(Param XParam);

extern "C" void defncvar(Param XParam, double * blockxo, double *blockyo, std::string varst, int vdim, float * var);
extern "C" void defncvarD(Param XParam, double * blockxo, double *blockyo, std::string varst, int vdim, double * var);
extern "C" void writenctimestep(std::string outfile, double totaltime);
extern "C" void writencvarstep(Param XParam, double * blockxo, double *blockyo, std::string varst, float * var);
extern "C" void writencvarstepD(Param XParam, double * blockxo, double *blockyo, std::string varst, double * var_d);//templating should have been fine here!
void readgridncsize(const std::string ncfile, int &nx, int &ny, int &nt, double &dx, double &xo, double &yo, double &to, double &xmax, double &ymax, double &tmax);
extern "C" void readnczb(int nx, int ny, std::string ncfile, float * &zb);
int readhotstartfile(Param XParam, int * leftblk, int *rightblk, int * topblk, int* botblk, double * blockxo, double * blockyo, float * dummy, float * &zs, float * &zb, float * &hh, float *&uu, float * &vv);
int readhotstartfileD(Param XParam, int * leftblk, int *rightblk, int * topblk, int* botblk, double * blockxo, double * blockyo, double * dummy, double * &zs, double * &zb, double * &hh, double *&uu, double * &vv);
void readWNDstep(forcingmap WNDUmap, forcingmap WNDVmap, int steptoread, float *&Uo, float *&Vo);
void readATMstep(forcingmap ATMPmap, int steptoread, float *&Po);
void InterpstepCPU(int nx, int ny, int hdstep, float totaltime, float hddt, float *&Ux, float *Uo, float *Un);
float interp2wnd(int wndnx, int wndny, float wnddx, float wndxo, float wndyo, float x, float y, float * U);
double interp2wnd(int wndnx, int wndny, double wnddx, double wndxo, double wndyo, double x, double y, float * U);


// I/O
inputmap readBathyhead(inputmap Bathy);
inputmap readcfmaphead(inputmap Roughmap);
forcingmap readforcingmaphead(forcingmap Fmap);
std::vector<Windin> readWNDfileUNI(std::string filename, double grdalpha);
extern "C" void readbathyMD(std::string filename, float *&zb);
void readbathyHeadMD(std::string filename, int &nx, int &ny, double &dx, double &grdalpha);
std::vector<SLTS> readWLfile(std::string WLfilename);
std::vector<Flowin> readFlowfile(std::string Flowfilename);
std::vector<Windin> readINfileUNI(std::string filename);
void readbathyASCHead(std::string filename, int &nx, int &ny, double &dx, double &xo, double &yo, double &grdalpha);
void readbathyASCzb(std::string filename, int nx, int ny, float* &zb);
extern "C" void readXBbathy(std::string filename, int nx, int ny, float *&zb);

Param readparamstr(std::string line, Param param);
std::string findparameter(std::string parameterstr, std::string line);
void split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, char delim);
std::string trim(const std::string& str, const std::string& whitespace);

Param checkparamsanity(Param XParam);
double setendtime(Param XParam);
void write_text_to_log_file(std::string text);
void SaveParamtolog(Param XParam);

//
double interptime(double next, double prev, double timenext, double time);
double BilinearInterpolation(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y);


// End of global definition
#endif

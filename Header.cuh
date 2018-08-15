
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

#include <iostream>
#include <fstream>
#include <sstream>

class TSnode {
public:
	int i, j;
	double x, y;
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
	int frictionmodel=0; // Not implemented yet 0: No friction; 
	double cf=0.0001; // bottom friction for flow model cf 
	double Cd=0.002; // Wind drag coeff
	int GPUDEVICE=0; // 0: first available GPU; -1: CPU single core; 2+: other GPU

	int doubleprecision = 0;

	//grid parameters
	double dx=0.0; // grid resolution
	double delta; // 
	int nx=0; // grid size
	int ny=0; //grid size
	double xo = 0.0; // grid origin
	double yo = 0.0; // grid origin
	double grdalpha=0.0; // grid rotation Y axis from the North input in degrees but later converted to rad
	int posdown = 0; // flag for bathy input. model requirement is positive up  so if posdown ==1 then zb=zb*-1.0f
	int spherical = 0; // flag for geographical coordinate. can be activated by using teh keyword geographic
	double Radius = 6371220.; //Earth radius [m]

	//files
	std::string Bathymetryfile;// bathymetry file name
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

	//bnd
	// 0:Wall; 1:sealevel dirichlet; 2:neumann (zeros gredient)
	int right = 2;
	int left = 2;
	int top = 2;
	int bot = 2;

	std::string rightbndfile;
	std::string leftbndfile;
	std::string topbndfile;
	std::string botbndfile;
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

};


class SLTS {
public:
	double time;
	std::vector<double> wlevs;
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

//Cuda Array to pre-store Water level boundary on the GPU and interpolate through the texture fetch
extern cudaArray* leftWLS_gp; // Cuda array to pre-store HD vel data before converting to textures
extern cudaArray* rightWLS_gp;
extern cudaArray* topWLS_gp;
extern cudaArray* botWLS_gp;

// Below create channels between cuda arrays (see above) and textures
extern cudaChannelFormatDesc channelDescleftbnd;// = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
extern cudaChannelFormatDesc channelDescrightbnd;// = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
extern cudaChannelFormatDesc channelDescbotbnd;// = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
extern cudaChannelFormatDesc channelDesctopbnd;// = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);



template <class T> T sq(T a);

template <class T> const T& max(const T& a, const T& b);
template <class T> const T& min(const T& a, const T& b);

//General CPU functions //Unecessary to declare here?
void mainloopCPU(Param XParam, std::vector<SLTS> leftWLbnd, std::vector<SLTS> rightWLbnd, std::vector<SLTS> topWLbnd, std::vector<SLTS> botWLbnd);
double FlowCPU(Param XParam, double nextoutputtime);
double FlowCPUDouble(Param XParam, double nextoutputtime);
double FlowCPUSpherical(Param XParam, double nextoutputtime);
//float demoloopCPU(Param XParam);

void update(int nx, int ny, float theta, float dt, float eps, float g, float CFL, float delta, float *hh, float *zs, float *uu, float *vv, float *&dh, float *&dhu, float *&dhv);
//void advance(int nx, int ny, float dt, float eps, float *hh, float *zs, float *uu, float * vv, float * dh, float *dhu, float *dhv, float * &hho, float *&zso, float *&uuo, float *&vvo);
//void cleanup(int nx, int ny, float * hhi, float *zsi, float *uui, float *vvi, float * &hho, float *&zso, float *&uuo, float *&vvo);
template <class T> void advance(int nx, int ny, T dt, T eps, T*zb, T *hh, T *zs, T *uu, T * vv, T * dh, T *dhu, T *dhv, T * &hho, T *&zso, T *&uuo, T *&vvo);
template <class T> void cleanup(int nx, int ny, T * hhi, T *zsi, T *uui, T *vvi, T * &hho, T *&zso, T *&uuo, T *&vvo);
//Bnd functions
//void neumannbnd(int nx, int ny, double*a);

void leftdirichletCPU(int nx, int ny, float g, std::vector<double> zsbndvec, float *zs, float *zb, float *hh, float *uu, float *vv);
void rightdirichletCPU(int nx, int ny, float g, std::vector<double> zsbndvec, float *zs, float *zb, float *hh, float *uu, float *vv);
void topdirichletCPU(int nx, int ny, float g, std::vector<double> zsbndvec, float *zs, float *zb, float *hh, float *uu, float *vv);
void botdirichletCPU(int nx, int ny, float g, std::vector<double> zsbndvec, float *zs, float *zb, float *hh, float *uu, float *vv);

void leftdirichletCPUD(int nx, int ny, double g, std::vector<double> zsbndvec, double *zs, double *zb, double *hh, double *uu, double *vv);
void rightdirichletCPUD(int nx, int ny, double g, std::vector<double> zsbndvec, double *zs, double *zb, double *hh, double *uu, double *vv);
void topdirichletCPUD(int nx, int ny, double g, std::vector<double> zsbndvec, double *zs, double *zb, double *hh, double *uu, double *vv);
void botdirichletCPUD(int nx, int ny, double g, std::vector<double> zsbndvec, double *zs, double *zb, double *hh, double *uu, double *vv);

void noslipbndLCPU(Param XParam);
void noslipbndRCPU(Param XParam);
void noslipbndTCPU(Param XParam);
void noslipbndBCPU(Param XParam);

//template <class T> void noslipbndallCPU(int nx, int ny, T dt, T eps, T *zb, T *zs, T *hh, T *uu, T *vv);
template <class T> void noslipbndLeftCPU(int nx, int ny, T eps, T *zb, T *zs, T *hh, T *uu, T *vv);
template <class T> void noslipbndRightCPU(int nx, int ny, T eps, T *zb, T *zs, T *hh, T *uu, T *vv);
template <class T> void noslipbndBotCPU(int nx, int ny, T eps, T *zb, T *zs, T *hh, T *uu, T *vv);
template <class T> void noslipbndTopCPU(int nx, int ny, T eps, T *zb, T *zs, T *hh, T *uu, T *vv);






template <class T> void quadfrictionCPU(int nx, int ny, T dt, T eps, T cf, T *hh, T *uu, T *vv);
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

extern "C" void defncvar(std::string outfile, int smallnc, float scalefactor, float addoffset, int nx, int ny, std::string varst, int vdim, float * var);
extern "C" void defncvarD(std::string outfile, int smallnc, float scalefactor, float addoffset, int nx, int ny, std::string varst, int vdim, double * var_d);
extern "C" void writenctimestep(std::string outfile, double totaltime);
extern "C" void writencvarstep(std::string outfile, int smallnc, float scalefactor, float addoffset, std::string varst, float * var);
extern "C" void writencvarstepD(std::string outfile, int smallnc, float scalefactor, float addoffset, std::string varst, double * var_d);
void readgridncsize(std::string ncfile, int &nx, int &ny, double &dx);
extern "C" void readnczb(int nx, int ny, std::string ncfile, float * &zb);
int readhotstartfile(Param XParam, float * &zs, float * &zb, float * &hh, float *&uu, float * &vv);
int readhotstartfileD(Param XParam, double * &zs, double * &zb, double * &hh, double *&uu, double * &vv);

// I/O
extern "C" void readbathy(std::string filename, float *&zb);
void readbathyHead(std::string filename, int &nx, int &ny, double &dx, double &grdalpha);
std::vector<SLTS> readWLfile(std::string WLfilename);
void readbathyASCHead(std::string filename, int &nx, int &ny, double &dx, double &xo, double &yo, double &grdalpha);
void readbathyASCzb(std::string filename, int nx, int ny, float* &zb);
extern "C" void readXBbathy(std::string filename, int nx, int ny, float *&zb);

Param readparamstr(std::string line, Param param);
std::string findparameter(std::string parameterstr, std::string line);
void split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, char delim);
std::string trim(const std::string& str, const std::string& whitespace);

Param checkparamsanity(Param XParam);
double setendtime(Param XParam, std::vector<SLTS> leftWLbnd, std::vector<SLTS> rightWLbnd, std::vector<SLTS> topWLbnd, std::vector<SLTS> botWLbnd);
void write_text_to_log_file(std::string text);
void SaveParamtolog(Param XParam);

//
double interptime(double next, double prev, double timenext, double time);

// End of global definition
#endif

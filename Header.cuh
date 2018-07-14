
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
	float x, y;
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

	//grid parameters
	double dx=0.0; // grid resolution
	double delta; // 
	int nx=0; // grid size
	int ny=0; //grid size
	double xo = 0.0; // grid origin
	double yo = 0.0; // grid origin
	double grdalpha=0.0; // grid rotation Y axis from the North input in degrees but later converted to rad
	int posdown = 0; // flag for bathy input. model requirement is positive up  so if posdown ==1 then zb=zb*-1.0f

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
	// 0:Wall; 1:sealevel dirichlet
	int right = 0;
	int left = 0;
	int top = 0;
	int bot = 0;

	std::string rightbndfile;
	std::string leftbndfile;
	std::string topbndfile;
	std::string botbndfile;
									  
	//other
	clock_t startcputime, endcputime;
	
	//Netcdf parameters
	int smallnc = 1;//default save as short integer if smallnc=0 then save all variables as float
	float scalefactor = 0.01;
	float addoffset = 0.0;

	// This is controlled by the sanity checker not directly by the user
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
	double time, wlev;
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
extern float *zs_g, *hh_g, *zb_g, *uu_g, *vv_g; // for GPU

extern float *zso, *hho, *uuo, *vvo;
extern float *zso_g, *hho_g, *uuo_g, *vvo_g; // for GPU

extern float * dhdx, *dhdy, *dudx, *dudy, *dvdx, *dvdy;
extern float *dzsdx, *dzsdy;

//GPU
extern float * dhdx_g, *dhdy_g, *dudx_g, *dudy_g, *dvdx_g, *dvdy_g;
extern float *dzsdx_g, *dzsdy_g;
//double *fmu, *fmv;

extern float *Su, *Sv, *Fqux, *Fquy, *Fqvx, *Fqvy;
extern float * Fhu, *Fhv;
extern float * dh, *dhu, *dhv;

//GPU
extern float *Su_g, *Sv_g, *Fqux_g, *Fquy_g, *Fqvx_g, *Fqvy_g;
extern float * Fhu_g, *Fhv_g;
extern float * dh_g, *dhu_g, *dhv_g;

extern float * hhmean, *uumean, *vvmean, *zsmean;
extern float * hhmean_g, *uumean_g, *vvmean_g, *zsmean_g;

extern float * hhmax, *uumax, *vvmax, *zsmax;
extern float * hhmax_g, *uumax_g, *vvmax_g, *zsmax_g;

extern float * vort, *vort_g;// Vorticity output

extern float dtmax;
extern float * dtmax_g;
extern float * TSstore, *TSstore_g;
extern float * dummy;

template <class T> T sq(T a);
double sqd(double a);
template <class T> const T& max(const T& a, const T& b);
template <class T> const T& min(const T& a, const T& b);

//General CPU functions //Unecessary to declare here?
void mainloopCPU(Param XParam, std::vector<SLTS> leftWLbnd, std::vector<SLTS> rightWLbnd, std::vector<SLTS> topWLbnd, std::vector<SLTS> botWLbnd);
float FlowCPU(Param XParam);
float demoloopCPU(Param XParam);

void update(int nx, int ny, float theta, double dt, double eps, double g, double CFL, double delta, float *hh, float *zs, float *uu, float *vv, float *&dh, float *&dhu, float *&dhv);
void advance(int nx, int ny, float dt, float eps, float *hh, float *zs, float *uu, float * vv, float * dh, float *dhu, float *dhv, float * &hho, float *&zso, float *&uuo, float *&vvo);
void cleanup(int nx, int ny, float * hhi, float *zsi, float *uui, float *vvi, float * &hho, float *&zso, float *&uuo, float *&vvo);
void leftdirichletCPU(int nx, int ny, float g, float zsbnd, float *zs, float *zb, float *hh, float *uu, float *vv);
void quadfrictionCPU(int nx, int ny, float dt, float eps, float cf, float *hh, float *uu, float *vv);
void noslipbndallCPU(int nx, int ny, float dt, float eps, float *zb, float *zs, float *hh, float *uu, float *vv);

//Bnd functions
void neumannbnd(int nx, int ny, double*a);

// CPU mean max
void AddmeanCPU(Param XParam);
void DivmeanCPU(Param XParam, float nstep);
void ResetmeanCPU(Param XParam);
void maxallCPU(Param XParam);
void CalcVort(Param XParam);
//Output functions
extern "C" void create2dnc(int nx, int ny, double dx, double dy, double totaltime, double *xx, double *yy, float * var);
extern "C" void write2varnc(int nx, int ny, double totaltime, float * var);

// Netcdf functions
extern "C" void creatncfileUD(std::string outfile, int nx, int ny, double dx, double totaltime);

extern "C" void defncvar(std::string outfile, int smallnc, float scalefactor, float addoffset, int nx, int ny, std::string varst, int vdim, float * var);
extern "C" void writenctimestep(std::string outfile, double totaltime);
extern "C" void writencvarstep(std::string outfile, int smallnc, float scalefactor, float addoffset, std::string varst, float * var);
void readgridncsize(std::string ncfile, int &nx, int &ny, double &dx);
extern "C" void readnczb(int nx, int ny, std::string ncfile, float * &zb);

// I/O
extern "C" void readbathy(std::string filename, float *&zb);
void readbathyHead(std::string filename, int &nx, int &ny, double &dx, double &grdalpha);
std::vector<SLTS> readWLfile(std::string WLfilename);
void readbathyASCHead(std::string filename, int &nx, int &ny, double &dx, double &grdalpha);
void readbathyASCzb(std::string filename, int nx, int ny, float* &zb);
extern "C" void readXBbathy(std::string filename, int nx, int ny, float *&zb);

Param readparamstr(std::string line, Param param);
std::string findparameter(std::string parameterstr, std::string line);
void split(const std::string &s, char delim, std::vector<std::string> &elems);
std::vector<std::string> split(const std::string &s, char delim);
std::string trim(const std::string& str, const std::string& whitespace);

Param checkparamsanity(Param XParam);
void write_text_to_log_file(std::string text);
void SaveParamtolog(Param XParam);

//
double interptime(double next, double prev, double timenext, double time);

// End of global definition
#endif

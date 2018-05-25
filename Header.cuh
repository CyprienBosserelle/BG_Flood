
#ifndef CSVGPU_H
#define CSVGPU_H
// includes, system

#define pi 3.14159265

#define epsilon 1e-30

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <ctime>
#include <netcdf.h>


extern double g;
extern double rho;
extern double eps;
extern double CFL;

extern double dt, dx;
extern int nx, ny;

extern double delta;

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

extern float dtmax;
extern float * dtmax_g;

template <class T> T sq(T a);
double sqd(double a);
template <class T> const T& max(const T& a, const T& b);
template <class T> const T& min(const T& a, const T& b);

//General CPU functions
void mainloopCPU();
double minmod2(double s0, double s1, double s2);
double dtnext(double t, double tnext, double dt);
//void gradient(int nx, int ny, double delta, double *a, double *&dadx, double * &dady);
void gradient(int nx, int ny, double delta, float *a, float *&dadx, float * &dady);
void kurganov(double hm, double hp, double um, double up, double Delta, double * fh, double * fq, double * dtmax);
void update(int nx, int ny, double dt, double eps, float *hh, float *zs, float *uu, float *vv, float *&dh, float *&dhu, float *&dhv);
void advance(int nx, int ny, double dt, double eps, float *hh, float *zs, float *uu, float * vv, float * dh, float *dhu, float *dhv, float * &hho, float *&zso, float *&uuo, float *&vvo);
void cleanup(int nx, int ny, float * hhi, float *zsi, float *uui, float *vvi, float * &hho, float *&zso, float *&uuo, float *&vvo);


//Bnd functions
void neumannbnd(int nx, int ny, double*a);

//Output functions
extern "C" void create2dnc(int nx, int ny, double dx, double dy, double totaltime, double *xx, double *yy, float * var);
extern "C" void write2varnc(int nx, int ny, double totaltime, float * var);

// End of global definition
#endif

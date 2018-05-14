
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
#include "netcdf.h"


extern double g;
extern double rho;
extern double eps;
extern double CFL;

extern double dt, dx;
extern int nx, ny;

extern double delta;

extern double *x, *y;
extern double *x_g, *y_g;

extern double *zs, *hh, *zb, *uu, *vv;
extern double *zso, *hho, *uuo, *vvo;


extern double * dhdx, *dhdy, *dudx, *dudy, *dvdx, *dvdy;
extern double *dzsdx, *dzsdy;
//double *fmu, *fmv;

extern double *Su, *Sv, *Fqux, *Fquy, *Fqvx, *Fqvy;

extern double * Fhu, *Fhv;

extern double * dh, *dhu, *dhv;

extern double dtmax;


template <class T> T sq(T a);
double sqd(double a);
template <class T> const T& max(const T& a, const T& b);
template <class T> const T& min(const T& a, const T& b);

//General CPU functions
void mainloopCPU();
double minmod2(double s0, double s1, double s2);
double dtnext(double t, double tnext, double dt);
void gradient(int nx, int ny, double delta, double *a, double *&dadx, double * &dady);
void kurganov(double hm, double hp, double um, double up, double Delta, double * fh, double * fq, double * dtmax);
void update(int nx, int ny, double dt, double eps, double *hh, double *zs, double *uu, double *vv, double *&dh, double *&dhu, double *&dhv);
void advance(int nx, int ny, double dt, double eps, double *hh, double *zs, double *uu, double * vv, double * dh, double *dhu, double *dhv, double * &hho, double *&zso, double *&uuo, double *&vvo);
void cleanup(int nx, int ny, double * hhi, double *zsi, double *uui, double *vvi, double * &hho, double *&zso, double *&uuo, double *&vvo);


//Bnd functions
void neumannbnd(int nx, int ny, double*a);

//Output functions
extern "C" void create2dnc(int nx, int ny, double dx, double dy, double totaltime, double *xx, double *yy, double * var);
extern "C" void write2varnc(int nx, int ny, double totaltime, double * var);

// End of global definition
#endif

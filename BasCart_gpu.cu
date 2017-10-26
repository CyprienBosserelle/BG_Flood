//////////////////////////////////////////////////////////////////////////////////
//						                                                        //
//Copyright (C) 2017 Bosserelle                                                 //
//                                                                              //
//This program is free software: you can redistribute it and/or modify          //
//it under the terms of the GNU General Public License as published by          //
//the Free Software Foundation.                                                 //
//                                                                              //
//This program is distributed in the hope that it will be useful,               //
//but WITHOUT ANY WARRANTY; without even the implied warranty of                //    
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                 //
//GNU General Public License for more details.                                  //
//                                                                              //
//You should have received a copy of the GNU General Public License             //
//along with this program.  If not, see <http://www.gnu.org/licenses/>.         //
//////////////////////////////////////////////////////////////////////////////////

// includes, system
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define pi 3.14159265

#include <stdio.h>
#include <math.h>
#include <cmath>
#include <ctime>

double phi = (1.0f + sqrt(5.0f)) / 2;
double aphi = 1 / (phi + 1);
double bphi = phi / (phi + 1);
double twopi = 8 * atan(1.0f);

double g = 9.81f;
double rho = 1025.0f;

double delta;

double *x, *y;
double *x_g, *y_g;

double *zs, *hh, *zb, *uu,*vv;
double *zs_g, *hh_g, *zb_g, *uu_g, *vv_g;

double * dhdx_g, *dhdy_g, *dudx_g, *dudy_g, *dvdx_g, *dvdy_g;
double *dzsdx_g, *dzsdy_g;

double *fmu_g, *fmv_g, *Su_g, *Sv_g, *Fqux_g, *Fquy_g, *Fqvx_g, *Fqvy_g;

double * dh_g, *dhu_g, *dhv_g;



#include "Flow_kernel.cu"


template <class T> const T& max(const T& a, const T& b) {
	return (a<b) ? b : a;     // or: return comp(a,b)?b:a; for version (2)
}

template <class T> const T& min(const T& a, const T& b) {
	return !(b<a) ? a : b;     // or: return comp(a,b)?b:a; for version (2)
}


void CUDA_CHECK(cudaError CUDerr)
{


	if (cudaSuccess != CUDerr) {

		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \

			__FILE__, __LINE__, cudaGetErrorString(CUDerr));

		exit(EXIT_FAILURE);

	}
}


// Main loop that actually runs the model
void mainloopGPU()
{
	
	

	
	
}





void flowbnd()
{
	
	
}


void flowstep()
{

	//advance

	//update

	//advance
	

}

void update(int nx, int ny, double dt, double eps)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);

	////calc gradient in h, eta, u and v

	/////if Hi is dry

	/////
	//double dx = delta/2.;
	//double zi = eta[] - hi;
	//double zl = zi - dx*(geta.x[] - gh.x[]);
	//double zn = eta[-1] - hn;
	//double zr = zn + dx*(geta.x[-1] - gh.x[-1]);
	//double zlr = max(zl, zr);

	//double hl = hi - dx*gh.x[];
	//double up = u.x[] - dx*gu.x.x[];
	//double hp = max(0., hl + zl - zlr);

	//double hr = hn + dx*gh.x[-1];
	//double um = u.x[-1] + dx*gu.x.x[-1];
	//double hm = max(0., hr + zr - zlr);

	//// Reimann solver
	//double fh, fu, fv;
	//kurganov(hm, hp, um, up, Δ*cm[] / fm.x[], &fh, &fu, &dtmax);
	//fv = (fh > 0. ? u.y[-1] + dx*gu.y.x[-1] : u.y[] - dx*gu.y.x[])*fh;

	////
	//double sl = G / 2.*(sq(hp) - sq(hl) + (hl + hi)*(zi - zl));
	//double sr = G / 2.*(sq(hm) - sq(hr) + (hr + hn)*(zn - zr));

	////Flux update

	//Fh.x[] = fm.x[] * fh;
	//Fq.x.x[] = fm.x[] * (fu - sl);
	//S.x[] = fm.x[] * (fu - sr);
	//Fq.y.x[] = fm.x[] * fv;

	////
	//vector dhu = vector(updates[1 + dimension*l]);
	//foreach() {
	//	double dhl =
	//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
	//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
	//	foreach_dimension()
	//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);

	

	//__global__ void MetricTerm(int nx, int ny, double delta, double G, double *h, double *u, double *v, double * fmu, double * fmv, double* dhu, double *dhv, double *Su, double *Sv, double * Fqux, double * Fquy, double * Fqvx, double * Fqvy)
	MetricTerm << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, hh_g, uu_g, vv_g, fmu_g, fmv_g, dhu_g, dhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g);

}


void advance(int nx, int ny, double dt, double eps)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);

	adv_stvenant << <gridDim, blockDim, 0 >> > (nx, ny, dt, eps, zb_g, hh_g, zs_g, uu_g, vv_g, dh_g, dhu_g, dhv_g);

	/*
	scalar hi = input[0], ho = output[0], dh = updates[0];
	vector * uol = (vector *) &output[1];

	// new fields in ho[], uo[]
	foreach() {
	double hold = hi[];
	ho[] = hold + dt*dh[];
	eta[] = zb[] + ho[];
	if (ho[] > dry) {
	for (int l = 0; l < nl; l++) {
	vector uo = vector(output[1 + dimension*l]);
	vector ui = vector(input[1 + dimension*l]),
	dhu = vector(updates[1 + dimension*l]);
	foreach_dimension()
	uo.x[] = (hold*ui.x[] + dt*dhu.x[])/ho[];
	}

	
	//In the case of [multiple
	layers](multilayer.h#viscous-friction-between-layers) we add the
	viscous friction between layers. 

	
}
	else // dry
		for (int l = 0; l < nl; l++) {
			vector uo = vector(output[1 + dimension*l]);
			foreach_dimension()
				uo.x[] = 0.;
		}
  }

  // fixme: on trees eta is defined as eta = zb + h and not zb +
  // ho in the refine_eta() and restriction_eta() functions below
  scalar * list = list_concat({ ho, eta }, (scalar *)uol);
  boundary(list);
  free(list);
	
	*/
}



int main(int argc, char **argv)
{
	//Model starts Here//

	//The main function setups all the init of the model and then calls the mainloop to actually run the model


	//First part reads the inputs to the model 
	//then allocate memory on GPU and CPU
	//Then prepare and initialise memory and arrays on CPU and GPU
	// Prepare output file
	// Run main loop
	// Clean up and close


	// Start timer to keep track of time 
	clock_t startcputime, endcputime;


	startcputime = clock();

	

	// This is just for temporary use
	int nx = 32;
	int ny = 32;
	double length = 1.0;
	delta = length / nx;
	double dx;
	double dt;
	
	hh = (double *)malloc(nx*ny * sizeof(double));
	uu = (double *)malloc(nx*ny * sizeof(double));
	vv = (double *)malloc(nx*ny * sizeof(double));
	zs = (double *)malloc(nx*ny * sizeof(double));
	zb = (double *)malloc(nx*ny * sizeof(double));
	x = (double *)malloc(nx*ny * sizeof(double));
	y = (double *)malloc(nx*ny * sizeof(double));

	//init variables
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			zb[i + j*nx] = 0.0;
			uu[i + j*nx] = 0.0;
			vv[i + j*nx] = 0.0;
			x[i + j*nx] = i*delta;
			y[i + j*nx] = j*delta;
		}
	}

	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			hh[i + j*nx] = 0.1 + 1.*exp(-200.*(x[i + j*nx] *x[i + j*nx] + y[i + j*nx] *y[i + j*nx]));;
			zs[i + j*nx] = zb[i + j*nx] + hh[i + j*nx];
		}
	}

	int nDevices;
	cudaGetDeviceCount(&nDevices);
	cudaDeviceProp prop;
	int GPUDEVICE = 0;

	if (GPUDEVICE > (nDevices - 1))
	{
		// 
		GPUDEVICE = 0;
	}

	cudaGetDeviceProperties(&prop,GPUDEVICE);
	printf("There are %d GPU devices on this machine\n", nDevices);
	printf("Using Device : %s\n", prop.name);

	CUDA_CHECK(cudaSetDevice(GPUDEVICE));

	//Allocate GPU memory
	CUDA_CHECK(cudaMalloc((void **)&hh_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&zb_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&zs_g, nx*ny * sizeof(double)));

	CUDA_CHECK(cudaMalloc((void **)&uu_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&vv_g, nx*ny * sizeof(double)));

	CUDA_CHECK(cudaMalloc((void **)&dh_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&dhu_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&dhv_g, nx*ny * sizeof(double)));

	CUDA_CHECK(cudaMalloc((void **)&fmu_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&fmv_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&Su_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&Sv_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&Fqux_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&Fquy_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&Fqvx_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&Fqvy_g, nx*ny * sizeof(double)));

	//i don't think x and y are needed here
	CUDA_CHECK(cudaMalloc((void **)&x_g, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **)&y_g, nx*ny * sizeof(double)));


	CUDA_CHECK(cudaMemcpy(hh_g, hh, nx*ny * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(zb_g, zb, nx*ny * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(zs_g, zs, nx*ny * sizeof(double), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemset(uu_g, 0.0f, nx*ny* sizeof(double)));
	CUDA_CHECK(cudaMemset(vv_g, 0.0f, nx*ny * sizeof(double)));

	CUDA_CHECK(cudaMemset(dh_g, 0.0f, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMemset(dhu_g, 0.0f, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMemset(dhv_g, 0.0f, nx*ny * sizeof(double)));

	CUDA_CHECK(cudaMemset(fmu_g, 0.0f, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMemset(fmv_g, 0.0f, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMemset(Su_g, 0.0f, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMemset(Sv_g, 0.0f, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMemset(Fqux_g, 0.0f, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMemset(Fquy_g, 0.0f, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMemset(Fqvx_g, 0.0f, nx*ny * sizeof(double)));
	CUDA_CHECK(cudaMemset(Fqvy_g, 0.0f, nx*ny * sizeof(double)));

	CUDA_CHECK(cudaMemcpy(x_g, x, nx*ny * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(y_g, y, nx*ny * sizeof(double), cudaMemcpyHostToDevice));



	mainloopGPU();
	



	
	endcputime = clock();
	printf("End Computation");
	printf("Total runtime= %d  seconds\n", (endcputime - startcputime) / CLOCKS_PER_SEC);
	
	cudaDeviceReset();











}


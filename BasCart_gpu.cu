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


#include "Header.cuh"



double phi = (1.0f + sqrt(5.0f)) / 2;
double aphi = 1 / (phi + 1);
double bphi = phi / (phi + 1);
double twopi = 8 * atan(1.0f);

double g = 1.0;// 9.81;
double rho = 1025.0;
double eps = 0.0001;
double CFL = 0.5;

double totaltime = 0.0;


double dt, dx;
int nx, ny;

double delta;

double *x, *y;
double *x_g, *y_g;

float *zs, *hh, *zb, *uu, *vv;//for CPU
float *zs_g, *hh_g, *zb_g, *uu_g, *vv_g; // for GPU
float *zso, *hho, *uuo, *vvo;
float *zso_g, *hho_g, *uuo_g, *vvo_g; // for GPU
//CPU
float * dhdx, *dhdy, *dudx, *dudy, *dvdx, *dvdy;
float *dzsdx, *dzsdy;
//GPU
float * dhdx_g, *dhdy_g, *dudx_g, *dudy_g, *dvdx_g, *dvdy_g;
float *dzsdx_g, *dzsdy_g;
//double *fmu, *fmv;

float *Su, *Sv, *Fqux, *Fquy, *Fqvx, *Fqvy;
float * Fhu, *Fhv;
float * dh, *dhu, *dhv;
//GPU
float *Su_g, *Sv_g, *Fqux_g, *Fquy_g, *Fqvx_g, *Fqvy_g;
float * Fhu_g, *Fhv_g;
float * dh_g, *dhu_g, *dhv_g;

float dtmax = 1.0 / epsilon;


#include "Flow_kernel.cu"

void CUDA_CHECK(cudaError CUDerr)
{


	if (cudaSuccess != CUDerr) {

		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \

			__FILE__, __LINE__, cudaGetErrorString(CUDerr));

		exit(EXIT_FAILURE);

	}
}




void updateGPU()
{
	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);

	int i, xplus, yplus, xminus, yminus;

	float hi;


	dtmax = 1 / epsilon;
	float dtmaxtmp = dtmax;

	// calculate gradients
	gradientGPUX <<<gridDim, blockDim, 0 >>>(nx, ny, delta, hh_g, dhdx_g);
	gradientGPUY <<<gridDim, blockDim, 0 >>>(nx, ny, delta, hh_g, dhdy_g);

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, zs_g, dzsdx_g);
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, zs_g, dzsdy_g);

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, uu_g, dudx_g);
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, uu_g, dudy_g);

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, vv_g, dvdx_g);
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, vv_g, dvdy_g);
	// Test whether it is better to have one here or later (are the instuctions overlap if occupancy and meme acess is available?)
	CUDA_CHECK(cudaDeviceSynchronize());

	float cm = 1.0;// 0.1;
	float fmu = 1.0;
	float fmv = 1.0;
}







void advanceGPU(int nx, int ny, double dt, double eps)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);

	
}

// Main loop that actually runs the model
void mainloopGPU()
{
	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	//update
	updateGPU();



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

	int GPUDEVICE = -1; //CPU by default

	startcputime = clock();



	// This is just for temporary use
	nx = 32;
	ny = 32;
	double length = 1.0;
	delta = length / nx;


	double *xx, *yy;
	dt = 0.0;// Will be resolved in update




	hh = (float *)malloc(nx*ny * sizeof(float));
	uu = (float *)malloc(nx*ny * sizeof(float));
	vv = (float *)malloc(nx*ny * sizeof(float));
	zs = (float *)malloc(nx*ny * sizeof(float));
	zb = (float *)malloc(nx*ny * sizeof(float));

	hho = (float *)malloc(nx*ny * sizeof(float));
	uuo = (float *)malloc(nx*ny * sizeof(float));
	vvo = (float *)malloc(nx*ny * sizeof(float));
	zso = (float *)malloc(nx*ny * sizeof(float));

	dhdx = (float *)malloc(nx*ny * sizeof(float));
	dhdy = (float *)malloc(nx*ny * sizeof(float));
	dudx = (float *)malloc(nx*ny * sizeof(float));
	dudy = (float *)malloc(nx*ny * sizeof(float));
	dvdx = (float *)malloc(nx*ny * sizeof(float));
	dvdy = (float *)malloc(nx*ny * sizeof(float));

	dzsdx = (float *)malloc(nx*ny * sizeof(float));
	dzsdy = (float *)malloc(nx*ny * sizeof(float));




	//fmu = (double *)malloc(nx*ny * sizeof(double));
	//fmv = (double *)malloc(nx*ny * sizeof(double));
	Su = (float *)malloc(nx*ny * sizeof(float));
	Sv = (float *)malloc(nx*ny * sizeof(float));
	Fqux = (float *)malloc(nx*ny * sizeof(float));
	Fquy = (float *)malloc(nx*ny * sizeof(float));
	Fqvx = (float *)malloc(nx*ny * sizeof(float));
	Fqvy = (float *)malloc(nx*ny * sizeof(float));
	Fhu = (float *)malloc(nx*ny * sizeof(float));
	Fhv = (float *)malloc(nx*ny * sizeof(float));

	dh = (float *)malloc(nx*ny * sizeof(float));
	dhu = (float *)malloc(nx*ny * sizeof(float));
	dhv = (float *)malloc(nx*ny * sizeof(float));

	//x = (double *)malloc(nx*ny * sizeof(double));
	xx = (double *)malloc(nx * sizeof(double));
	//y = (double *)malloc(nx*ny * sizeof(double));
	yy = (double *)malloc(ny * sizeof(double));

	if (GPUDEVICE >= 0)
	{
		// Init GPU
		// This should be in the sanity check
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		cudaDeviceProp prop;

		if (GPUDEVICE > (nDevices - 1))
		{
			// 
			GPUDEVICE = (nDevices - 1);
		}

	}

	// Now that we checked that there was indeed a GPU available
	if (GPUDEVICE >= 0)
	{
		CUDA_CHECK(cudaMalloc((void **)&hh_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&uu_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&vv_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&zb_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&zs_g, nx*ny*sizeof(float)));

		CUDA_CHECK(cudaMalloc((void **)&hho_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&uuo_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&vvo_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&zso_g, nx*ny*sizeof(float)));

		CUDA_CHECK(cudaMalloc((void **)&dhdx_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&dhdy_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&dudx_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&dudy_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&dvdx_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&dvdy_g, nx*ny*sizeof(float)));

		CUDA_CHECK(cudaMalloc((void **)&dzsdx_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&dzsdy_g, nx*ny*sizeof(float)));

		CUDA_CHECK(cudaMalloc((void **)&Su_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&Sv_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&Fqux_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&Fquy_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&Fqvx_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&Fqvy_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&Fhu_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&Fhv_g, nx*ny*sizeof(float)));

		CUDA_CHECK(cudaMalloc((void **)&dh_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&dhu_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&dhv_g, nx*ny*sizeof(float)));

		
	}



	//init variables
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			zb[i + j*nx] = 0.0f;
			uu[i + j*nx] = 0.0f;
			vv[i + j*nx] = 0.0f;
			//x[i + j*nx] = (i-nx/2)*delta+0.5*delta;
			xx[i] = (i - nx / 2)*delta + 0.5*delta;
			yy[j] = (j - ny / 2)*delta + 0.5*delta;
			//y[i + j*nx] = (j-ny/2)*delta + 0.5*delta;
			//fmu[i + j*nx] = 1.0;
			//fmv[i + j*nx] = 1.0;
		}
	}

	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			double a;

			a = sq(xx[i]) + sq(yy[j]);
			//b =x[i + j*nx] * x[i + j*nx] + y[i + j*nx] * y[i + j*nx];


			//if (abs(a - b) > 0.00001)
			//{
			//	printf("%f\t%f\n", a, b);
			//}



			hh[i + j*nx] = 0.1 + 1.*exp(-200.*(a));

			zs[i + j*nx] = zb[i + j*nx] + hh[i + j*nx];
		}
	}

	if (GPUDEVICE >= 0)
	{
		CUDA_CHECK(cudaMemcpy(zb_g, zb, nx*ny*sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(hh_g, hh, nx*ny*sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(uu_g, uu, nx*ny*sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(vv_g, vv, nx*ny*sizeof(float), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(zs_g, zs, nx*ny*sizeof(float), cudaMemcpyHostToDevice));
	}



	create2dnc(nx, ny, dx, dx, 0.0, xx, yy, hh);

	//while (totaltime < 10.0)
	for (int i = 0; i <10; i++)
	{
		if (GPUDEVICE >= 0)
		{
			mainloopCPU();
		}
		else
		{
			mainloopCPU();
		}
		
		totaltime = totaltime + dt;
		write2varnc(nx, ny, totaltime, hh);
		//write2varnc(nx, ny, totaltime, dhdx);
	}






	endcputime = clock();
	printf("End Computation totaltime=%f\n", totaltime);
	printf("Total runtime= %d  seconds\n", (endcputime - startcputime) / CLOCKS_PER_SEC);
	//if GPU?
	cudaDeviceReset();









}


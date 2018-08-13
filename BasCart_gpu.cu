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
double epsilon = 1e-30;
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
float * dtmax_g;
float *arrmax_g, float *arrmin_g;
float *arrmin;

float * dummy;

std::string outfile = "output.nc";
std::vector<std::string> outvars;
std::map<std::string, float *> OutputVarMapCPU;
std::map<std::string, float *> OutputVarMapGPU;
std::map<std::string, int> OutputVarMaplen;
/*
//constructor cant be global
OutputVarMapCPU["zb"] = zb;
OutputVarMapGPU["zb"] = zb_g;
OutputVarMaplen["zb"] = nx*ny;

OutputVarMapCPU["uu"] = uu;
OutputVarMapGPU["uu"] = uu_g;
OutputVarMaplen["uu"] = nx*ny;

OutputVarMapCPU["vv"] = vv;
OutputVarMapGPU["vv"] = vv_g;
OutputVarMaplen["vv"] = nx*ny;

OutputVarMapCPU["zs"] = zs;
OutputVarMapGPU["zs"] = zs_g;
OutputVarMaplen["zs"] = nx*ny;

OutputVarMapCPU["hh"] = hh;
OutputVarMapGPU["hh"] = hh_g;
OutputVarMaplen["hh"] = nx*ny;
*/

#include "Flow_kernel.cu"

void CUDA_CHECK(cudaError CUDerr)
{


	if (cudaSuccess != CUDerr) {

		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \

			__FILE__, __LINE__, cudaGetErrorString(CUDerr));

		exit(EXIT_FAILURE);

	}
}

unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}



float maxdiff(int nxny, float * ref, float * pred)
{
	float maxd = 0.0f;
	for (int i = 0; i < nxny; i++)
	{
		maxd = max(abs(pred[i] - ref[i]), maxd);
	}
	return maxd;
}

void checkloopGPU()
{
	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);

	dim3 blockDimLine(32, 1, 1);
	dim3 gridDimLine(ceil((nx*ny*1.0f) / blockDimLine.x), 1, 1);

	int i, xplus, yplus, xminus, yminus;

	float maxerr = 1e-11f;//1e-7f

	float hi;

	float maxdiffer;

	dtmax = 1 / epsilon;
	float dtmaxtmp = dtmax;

	resetdtmax << <gridDim, blockDim, 0 >> > (nx, ny, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(dummy, dtmax_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	float mindtmax = 1.0f / 1e-30f;
	for (int i = 0; i < nx*ny; i++)
	{
		mindtmax = min(dummy[i], mindtmax);
	}


	//update step 1

	// calculate gradients
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, hh_g, dhdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, hh_g, dhdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, zs_g, dzsdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, zs_g, dzsdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, uu_g, dudx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, uu_g, dudy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, vv_g, dvdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, vv_g, dvdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//update(int nx, int ny, double dt, double eps,double *hh, double *zs, double *uu, double *vv, double *dh, double *dhu, double *dhv)
	update(nx, ny, dt, eps, hh, zs, uu, vv, dh, dhu, dhv);



	CUDA_CHECK(cudaMemcpy(dummy, hh_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, hh, dummy);
	if (maxdiffer > 1e-7f)
	{
		printf("High error in dhdx: %f\n", maxdiffer);
	}


	
	// check gradients

	CUDA_CHECK(cudaMemcpy(dummy, dhdx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dhdx, dummy);
	if (maxdiffer > maxerr)
	{ 
		printf("High error in dhdx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dhdy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dhdy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dhdy: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dzsdx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dzsdx, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dzsdx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dzsdy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dzsdy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dzsdy: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dudx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dudx, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dudx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dudy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dudy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dudy: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dvdx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dvdx, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dvdx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dvdy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dvdy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dvdy: %f\n", maxdiffer);
	}


	// All good so far continuing

	updateKurgX << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, eps, CFL, hh_g, zs_g, uu_g, vv_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());


	updateKurgY << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, eps, CFL, hh_g, zs_g, uu_g, vv_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(dummy, Fhu_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Fhu, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fhu: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fhv_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Fhv, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fhv: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fqux_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Fqux, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fqux: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fqvx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Fqvx, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fqvx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fqvy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Fqvy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fqvy: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fquy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Fquy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fquy: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Su_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Su, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Su: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Sv_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Sv, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Sv: %f\n", maxdiffer);
	}

	// All good so far continuing
	///////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////
	minmaxKernel << <gridDimLine, blockDimLine, 0 >> >(nx*ny, arrmax_g, arrmin_g, dtmax_g);
	//CUT_CHECK_ERROR("UpdateZom execution failed\n");
	CUDA_CHECK(cudaDeviceSynchronize());

	finalminmaxKernel << <1, blockDimLine, 0 >> >(arrmax_g, arrmin_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//CUDA_CHECK(cudaMemcpy(arrmax, arrmax_g, nx*ny*sizeof(DECNUM), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(arrmin, arrmin_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));

	maxdiffer = abs(dtmax-arrmin[0]);


	CUDA_CHECK(cudaMemcpy(dummy, dtmax_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));


	
	mindtmax=1.0f/1e-30f;
	for (int i = 0; i < nx*ny; i++)
	{
		mindtmax=min(dummy[i], mindtmax);
	}
	maxdiffer = abs(dtmax - mindtmax);

	updateEV << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, hh_g, uu_g, vv_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(dummy, dh_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dh, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dh: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dhu_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dhu, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dhu: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dhv_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dhv, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dhv: %f\n", maxdiffer);
	}

	// All good so far continuing
	///////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////
	dt = arrmin[0];
	
	//predictor (advance 1/2 dt)
	Advkernel << <gridDim, blockDim, 0 >> >(nx, ny, dt*0.5, eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//predictor
	advance(nx, ny, dt*0.5, eps, hh, zs, uu, vv, dh, dhu, dhv, hho, zso, uuo, vvo);

	CUDA_CHECK(cudaMemcpy(dummy, zso_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, zso, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in zso: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, hho_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, hho, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in hho: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, uuo_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, uuo, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in uuo: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, vvo_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, vvo, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in vvo: %f\n", maxdiffer);
	}

	// All good so far continuing
	///////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////

	//corrector
	update(nx, ny, dt, eps, hho, zso, uuo, vvo, dh, dhu, dhv);

	//corrector setp
	//update again
	// calculate gradients
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, hho_g, dhdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, hho_g, dhdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, zso_g, dzsdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, zso_g, dzsdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, uuo_g, dudx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, uuo_g, dudy_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, vvo_g, dvdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, vvo_g, dvdy_g);
	// Test whether it is better to have one here or later (are the instuctions overlap if occupancy and meme acess is available?)
	CUDA_CHECK(cudaDeviceSynchronize());

	// check gradients

	CUDA_CHECK(cudaMemcpy(dummy, dhdx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dhdx, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dhdx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dhdy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dhdy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dhdy: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dzsdx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dzsdx, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dzsdx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dzsdy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dzsdy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dzsdy: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dudx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dudx, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dudx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dudy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dudy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dudy: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dvdx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dvdx, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dvdx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dvdy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dvdy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dvdy: %f\n", maxdiffer);
	}



	updateKurgX << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, eps, CFL, hho_g, zso_g, uuo_g, vvo_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());


	updateKurgY << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, eps, CFL, hho_g, zso_g, uuo_g, vvo_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	// no reduction of dtmax during the corrector step


	CUDA_CHECK(cudaMemcpy(dummy, Fhu_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Fhu, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fhu: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fhv_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Fhv, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fhv: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fqux_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Fqux, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fqux: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fqvx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Fqvx, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fqvx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fqvy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Fqvy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fqvy: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fquy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Fquy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fquy: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Su_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Su, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Su: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Sv_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, Sv, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Sv: %f\n", maxdiffer);
	}


	updateEV << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, hho_g, uuo_g, vvo_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	CUDA_CHECK(cudaDeviceSynchronize());


	CUDA_CHECK(cudaMemcpy(dummy, dh_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dh, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dh: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dhu_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dhu, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dhu: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dhv_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, dhv, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dhv: %f\n", maxdiffer);
	}


	advance(nx, ny, dt, eps, hh, zs, uu, vv, dh, dhu, dhv, hho, zso, uuo, vvo);

	//
	Advkernel << <gridDim, blockDim, 0 >> >(nx, ny, dt, eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(dummy, zso_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, zso, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in zso: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, hho_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, hho, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in hho: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, uuo_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, uuo, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in uuo: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, vvo_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, vvo, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in vvo: %f\n", maxdiffer);
	}

	cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);

	cleanupGPU << <gridDim, blockDim, 0 >> >(nx, ny, hho_g, zso_g, uuo_g, vvo_g, hh_g, zs_g, uu_g, vv_g);
	CUDA_CHECK(cudaDeviceSynchronize());



	CUDA_CHECK(cudaMemcpy(dummy, zs_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, zs, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in zs: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, hh_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, hh, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in hh: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, uu_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, uu, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in uu: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, vv_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiff(nx*ny, vv, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in vv: %f\n", maxdiffer);
	}


}



void FlowGPU()
{
	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);

	

	int i, xplus, yplus, xminus, yminus;

	float hi;


	dtmax = 1 / epsilon;
	float dtmaxtmp = dtmax;

	resetdtmax << <gridDim, blockDim, 0 >> > (nx, ny, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	//update step 1

	// calculate gradients
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, hh_g, dhdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, hh_g, dhdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, zs_g, dzsdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, zs_g, dzsdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, uu_g, dudx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, uu_g, dudy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, vv_g, dvdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, vv_g, dvdy_g);
	// Test whether it is better to have one here or later (are the instuctions overlap if occupancy and meme acess is available?)
	CUDA_CHECK(cudaDeviceSynchronize());


	updateKurgX << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, eps, CFL, hh_g, zs_g, uu_g, vv_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());


	updateKurgY << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, eps, CFL, hh_g, zs_g, uu_g, vv_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());


	/////////////////////////////////////////////////////
	// Reduction of dtmax
	/////////////////////////////////////////////////////

	// copy from GPU and do the reduction on the CPU  ///LAME!
	/*
	CUDA_CHECK(cudaMemcpy(dummy, dtmax_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	float mindtmax = 1.0f / 1e-30f;
	for (int i = 0; i < nx*ny; i++)
	{
		mindtmax = min(dummy[i], mindtmax);
	}
	dt = mindtmax;
	*/


	//GPU but it doesn't work
	/*
	minmaxKernel << <gridDimLine, blockDimLine, 0 >> >(nx*ny, arrmax_g, arrmin_g, dtmax_g);
	//CUT_CHECK_ERROR("UpdateZom execution failed\n");
	CUDA_CHECK(cudaDeviceSynchronize());

	finalminmaxKernel << <1, blockDimLine, 0 >> >(arrmax_g, arrmin_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	

	//CUDA_CHECK(cudaMemcpy(arrmax, arrmax_g, nx*ny*sizeof(DECNUM), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(arrmin, arrmin_g, nx*ny*sizeof(float), cudaMemcpyDeviceToHost));
	
	dt = arrmin[0];
	float diffdt = arrmin[0] - mindtmax;
	*/

	//GPU Harris reduction #3. 8.3x reduction #0  Note #7 if a lot faster
	// This was successfully tested with a range of grid size
	//reducemax3 << <gridDimLine, blockDimLine, 64*sizeof(float) >> >(dtmax_g, arrmax_g, nx*ny)
	int s = nx*ny;
	int maxThreads = 256;
	int threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
	int blocks = (s + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);
	dim3 blockDimLine(threads, 1, 1);
	dim3 gridDimLine(blocks, 1, 1);

	float mindtmaxB;

	reducemin3 << <gridDimLine, blockDimLine, smemSize >> > (dtmax_g, arrmax_g, nx*ny);
	CUDA_CHECK(cudaDeviceSynchronize());

	

	s = gridDimLine.x;
	while (s > 1)//cpuFinalThreshold
	{
		threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);

		smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

		dim3 blockDimLineS(threads, 1, 1);
		dim3 gridDimLineS(blocks, 1, 1);

		CUDA_CHECK(cudaMemcpy(dtmax_g, arrmax_g, s * sizeof(float), cudaMemcpyDeviceToDevice));

		reducemin3 << <gridDimLineS, blockDimLineS, smemSize >> > (dtmax_g, arrmax_g, s);
		CUDA_CHECK(cudaDeviceSynchronize());

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}

	
	CUDA_CHECK(cudaMemcpy(dummy, arrmax_g, 32*sizeof(float), cudaMemcpyDeviceToHost));
	mindtmaxB = dummy[0];
	/*
	//32 seem safe here bu I wonder why it is not 1 for the largers arrays...
	
	for (int i = 0; i < 32; i++)
	{
		mindtmaxB = min(dummy[i], mindtmaxB);
		printf("dt=%f\n", dummy[i]);
		
	}
	*/
	

	float diffdt = mindtmaxB - mindtmax;
	dt = mindtmaxB;

	printf("dt=%f\n", dt);


	updateEV << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, hh_g, uu_g, vv_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	


	//predictor (advance 1/2 dt)
	Advkernel << <gridDim, blockDim, 0 >> >(nx, ny, dt*0.5, eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//corrector setp
	//update again
	// calculate gradients
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, hho_g, dhdx_g);
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, hho_g, dhdy_g);

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, zso_g, dzsdx_g);
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, zso_g, dzsdy_g);

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, uuo_g, dudx_g);
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, uuo_g, dudy_g);

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, delta, vvo_g, dvdx_g);
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, delta, vvo_g, dvdy_g);
	// Test whether it is better to have one here or later (are the instuctions overlap if occupancy and meme acess is available?)
	CUDA_CHECK(cudaDeviceSynchronize());


	updateKurgX << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, eps, CFL, hho_g, zso_g, uuo_g, vvo_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	
	updateKurgY << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, eps, CFL, hho_g, zso_g, uuo_g, vvo_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	// no reduction of dtmax during the corrector step

	updateEV << <gridDim, blockDim, 0 >> >(nx, ny, delta, g, hho_g, uuo_g, vvo_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//
	Advkernel << <gridDim, blockDim, 0 >> >(nx, ny, dt, eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
	cleanupGPU << <gridDim, blockDim, 0 >> >(nx, ny, hho_g, zso_g, uuo_g, vvo_g, hh_g, zs_g, uu_g, vv_g);
	CUDA_CHECK(cudaDeviceSynchronize());
}

// Main loop that actually runs the model
void mainloopGPU()
{
	FlowGPU();
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

	int GPUDEVICE = 0; //-1:CPU 0:default GPU (first available) 1+:other GPU  [0]

	startcputime = clock();



	// This is just for temporary use
	nx = 32;
	ny = 32;
	double length = 1.0;
	delta = length / nx;


	double *xx, *yy;
	dt = 0.0;// Will be resolved in update

	std::vector<std::string> SupportedVarNames = { "zb", "zs", "uu", "vv", "hh" };
	for (int isup = 0; isup < SupportedVarNames.size(); isup++)
	{
		outvars.push_back(SupportedVarNames[isup]);

	}


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

	dummy = (float *)malloc(nx*ny * sizeof(float));

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

		CUDA_CHECK(cudaMalloc((void **)&dtmax_g, nx*ny*sizeof(float)));

		arrmin = (float *)malloc(nx*ny * sizeof(float));
		CUDA_CHECK(cudaMalloc((void **)&arrmin_g, nx*ny*sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&arrmax_g, nx*ny*sizeof(float)));
		
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
		dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
		dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);

		initdtmax << <gridDim, blockDim, 0 >> >(nx, ny, (float) epsilon, dtmax_g);

	}

	

	OutputVarMapCPU["zb"] = zb;
	OutputVarMapGPU["zb"] = zb_g;
	OutputVarMaplen["zb"] = nx*ny;

	OutputVarMapCPU["uu"] = uu;
	OutputVarMapGPU["uu"] = uu_g;
	OutputVarMaplen["uu"] = nx*ny;

	OutputVarMapCPU["vv"] = vv;
	OutputVarMapGPU["vv"] = vv_g;
	OutputVarMaplen["vv"] = nx*ny;

	OutputVarMapCPU["zs"] = zs;
	OutputVarMapGPU["zs"] = zs_g;
	OutputVarMaplen["zs"] = nx*ny;

	OutputVarMapCPU["hh"] = hh;
	OutputVarMapGPU["hh"] = hh_g;
	OutputVarMaplen["hh"] = nx*ny;
	//create nc file with no variables


	creatncfileUD(outfile, nx, ny, delta, 0.0);
	for (int ivar = 0; ivar < outvars.size(); ivar++)
	{
		//Create definition for each variable and store it
		defncvar(outfile, 0,1.0f,0.0f,nx,ny,outvars[ivar], 3, OutputVarMapCPU[outvars[ivar]]);
	}
	//create2dnc(nx, ny, dx, dx, 0.0, xx, yy, hh);

	//while (totaltime < 10.0)
	for (int i = 0; i <10; i++)
	{
		if (GPUDEVICE >= 0)
		{
			mainloopGPU();
			//CUDA_CHECK(cudaMemcpy(hh, hh_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
			//checkloopGPU();
		}
		else
		{
			mainloopCPU();
		}
		
		totaltime = totaltime + dt;
		//void creatncfileUD(std::string outfile, int nx, int ny, double dx, double totaltime);
		//void defncvar(std::string outfile, int smallnc, float scalefactor, float addoffset, int nx, int ny, std::string varst, int vdim, float * var);
		//void writenctimestep(std::string outfile, double totaltime);
		//void writencvarstep(std::string outfile, int smallnc, float scalefactor, float addoffset, std::string varst, float * var);
		writenctimestep(outfile, totaltime);

		for (int ivar = 0; ivar < outvars.size(); ivar++)
		{
			if (OutputVarMaplen[outvars[ivar]] > 0)
			{
				if (GPUDEVICE >= 0)
				{
					//Should be async
					CUDA_CHECK(cudaMemcpy(OutputVarMapCPU[outvars[ivar]], OutputVarMapGPU[outvars[ivar]], OutputVarMaplen[outvars[ivar]] * sizeof(float), cudaMemcpyDeviceToHost));

				}
				//Create definition for each variable and store it
				writencvarstep(outfile, 0,1.0f,0.0f,outvars[ivar], OutputVarMapCPU[outvars[ivar]]);
			}
		}
		//write2varnc(nx, ny, totaltime, hh);
		//write2varnc(nx, ny, totaltime, dhdx);
	}






	endcputime = clock();
	printf("End Computation totaltime=%f\n", totaltime);
	printf("Total runtime= %d  seconds\n", (endcputime - startcputime) / CLOCKS_PER_SEC);
	//if GPU?
	cudaDeviceReset();









}


//////////////////////////////////////////////////////////////////////////////////
//						                                                        //
//Copyright (C) 2018 Bosserelle                                                 //
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



//double phi = (1.0f + sqrt(5.0f)) / 2;
//double aphi = 1 / (phi + 1);
//double bphi = phi / (phi + 1);
//double twopi = 8 * atan(1.0f);
double epsilon = 1e-30;
//double g = 1.0;// 9.81;
//double rho = 1025.0;
//double eps = 0.0001;
//double CFL = 0.5;
//
//double totaltime = 0.0;
//
//
//double dt, dx;
//int nx, ny;
//
//double delta;

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

float * TSstore, *TSstore_g;
float * hhmean, *uumean, *vvmean, *zsmean;
float * hhmean_g, *uumean_g, *vvmean_g, *zsmean_g;

float * hhmax, *uumax, *vvmax, *zsmax;
float * hhmax_g, *uumax_g, *vvmax_g, *zsmax_g;

float * vort, *vort_g;// Vorticity output

float dtmax = 1.0 / epsilon;
float * dtmax_g;
float *arrmax_g;
float *arrmin_g;
float *arrmin;

float * dummy;

//std::string outfile = "output.nc";
//std::vector<std::string> outvars;
std::map<std::string, float *> OutputVarMapCPU;
std::map<std::string, float *> OutputVarMapGPU;
std::map<std::string, int> OutputVarMaplen;

cudaArray* leftWLS_gp; // Cuda array to pre-store HD vel data before converting to textures
cudaArray* rightWLS_gp;
cudaArray* topWLS_gp;
cudaArray* botWLS_gp;

cudaChannelFormatDesc channelDescleftbnd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescrightbnd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescbotbnd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDesctopbnd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

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

float maxdiffID(int nx, int ny, int &im, int &jm,  float * ref, float * pred)
{
	float maxd = 0.0f;
	
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			if (abs(pred[i] - ref[i]) > maxd)
			{
				im = i;
				jm = j;
				maxd = abs(pred[i] - ref[i]);
			}
		}
	}
	return maxd;
}

void checkloopGPU(Param XParam)
{
	int nx = XParam.nx;
	int ny = XParam.ny;
	float delta = XParam.delta;
	float eps = XParam.eps;
	float CFL = XParam.CFL;
	float g = XParam.g;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);

	dim3 blockDimLine(32, 1, 1);
	dim3 gridDimLine(ceil((nx*ny*1.0f) / blockDimLine.x), 1, 1);

	int i, xplus, yplus, xminus, yminus;

	float maxerr = 1e-11f;//1e-7f

	float hi;

	float maxdiffer;

	int imax = 0;
	int jmax = 0;

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
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, hh_g, dhdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, hh_g, dhdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, zs_g, dzsdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, zs_g, dzsdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, uu_g, dudx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, uu_g, dudy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, vv_g, dvdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, vv_g, dvdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//update(int nx, int ny, double dt, double eps, double g, double CFL, double delta, float *hh, float *zs, float *uu, float *vv, float *&dh, float *&dhu, float *&dhv);
	update(nx, ny, XParam.theta, XParam.dt, eps, XParam.g, XParam.CFL, XParam.delta, hh, zs, uu, vv, dh, dhu, dhv);



	CUDA_CHECK(cudaMemcpy(dummy, hh_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, hh, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dhdx: %f\n", maxdiffer);
	}


	
	// check gradients

	CUDA_CHECK(cudaMemcpy(dummy, dhdx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, dhdx, dummy);
	if (maxdiffer > maxerr)
	{ 
		printf("High error in dhdx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dhdy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, dhdy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dhdy: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dzsdx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, dzsdx, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dzsdx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dzsdy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, dzsdy, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dzsdy: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dudx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, dudx, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in dudx: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dudy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, dudy, dummy);
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
	
	
	maxdiffer = maxdiffID(nx, ny, imax, jmax, Fqux, dummy);
	if (maxdiffer > maxerr)
	{
		printf("High error in Fqux (%f) in i=%d, j=%d\n", maxdiffer,imax,jmax);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fqvx_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, Fqvx, dummy);;
	if (maxdiffer > maxerr)
	{
		printf("High error in Fqvx (%f) in i=%d, j=%d\n", maxdiffer, imax, jmax);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fqvy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, Fqvy, dummy);;
	if (maxdiffer > maxerr)
	{
		printf("High error in Fqvy (%f) in i=%d, j=%d\n", maxdiffer, imax, jmax);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Fquy_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, Fquy, dummy);;
	if (maxdiffer > maxerr)
	{
		printf("High error in Fquy (%f) in i=%d, j=%d\n", maxdiffer, imax, jmax);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Su_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, Su, dummy);;
	if (maxdiffer > maxerr)
	{
		printf("High error in Su (%f) in i=%d, j=%d\n", maxdiffer, imax, jmax);
	}

	CUDA_CHECK(cudaMemcpy(dummy, Sv_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, Sv, dummy);;
	if (maxdiffer > maxerr)
	{
		printf("High error in Sv (%f) in i=%d, j=%d\n", maxdiffer, imax, jmax);
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
	maxdiffer = maxdiffID(nx, ny, imax, jmax, dh, dummy);;
	if (maxdiffer > maxerr)
	{
		printf("High error in dh: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dhu_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, dhu, dummy);;
	if (maxdiffer > maxerr)
	{
		printf("High error in dhu: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, dhv_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, dhv, dummy);;
	if (maxdiffer > maxerr)
	{
		printf("High error in dhv: %f\n", maxdiffer);
	}

	// All good so far continuing
	///////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////
	XParam.dt = arrmin[0];
	
	//predictor (advance 1/2 dt)
	Advkernel << <gridDim, blockDim, 0 >> >(nx, ny, XParam.dt*0.5, eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//predictor
	advance(nx, ny, XParam.dt*0.5, eps, hh, zs, uu, vv, dh, dhu, dhv, hho, zso, uuo, vvo);

	CUDA_CHECK(cudaMemcpy(dummy, zso_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, zso, dummy);;
	if (maxdiffer > maxerr)
	{
		printf("High error in zso: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, hho_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, hho, dummy);;
	if (maxdiffer > maxerr)
	{
		printf("High error in hho: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, uuo_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, uuo, dummy);;
	if (maxdiffer > maxerr)
	{
		printf("High error in uuo: %f\n", maxdiffer);
	}

	CUDA_CHECK(cudaMemcpy(dummy, vvo_g, nx*ny * sizeof(float), cudaMemcpyDeviceToHost));
	maxdiffer = maxdiffID(nx, ny, imax, jmax, vvo, dummy);;
	if (maxdiffer > maxerr)
	{
		printf("High error in vvo: %f\n", maxdiffer);
	}

	// All good so far continuing
	///////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////

	//corrector
	update(nx, ny, XParam.theta, XParam.dt, eps, XParam.g, XParam.CFL, XParam.delta, hho, zso, uuo, vvo, dh, dhu, dhv);

	//corrector setp
	//update again
	// calculate gradients
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, hho_g, dhdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, hho_g, dhdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, zso_g, dzsdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, zso_g, dzsdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, uuo_g, dudx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, uuo_g, dudy_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, vvo_g, dvdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, delta, vvo_g, dvdy_g);
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


	advance(nx, ny, XParam.dt, eps, hh, zs, uu, vv, dh, dhu, dhv, hho, zso, uuo, vvo);

	//
	Advkernel << <gridDim, blockDim, 0 >> >(nx, ny, XParam.dt, eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
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




void LeftFlowBnd(Param XParam, std::vector<SLTS> leftWLbnd)
{
	//
	int nx = XParam.nx;
	int ny = XParam.ny;
	if (XParam.left == 1 && !leftWLbnd.empty())
	{
		int SLstepinbnd = 1;

		

		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = leftWLbnd[SLstepinbnd].time - XParam.totaltime;

		while (difft < 0.0)
		{
			SLstepinbnd++;
			difft = leftWLbnd[SLstepinbnd].time - XParam.totaltime;
		}

		

		dim3 blockDim(16, 1, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
		dim3 gridDim(ceil((ny*1.0f) / blockDim.x), 1, 1);
		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			float itime = SLstepinbnd - 1.0 + (XParam.totaltime - leftWLbnd[SLstepinbnd - 1].time) / (leftWLbnd[SLstepinbnd].time - leftWLbnd[SLstepinbnd - 1].time);

			leftdirichlet << <gridDim, blockDim, 0 >> > (nx, ny, leftWLbnd[0].wlevs.size(), XParam.g, itime, zs_g, zb_g, hh_g, uu_g, vv_g);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			std::vector<float> zsbndleft;
			for (int n = 0; n < leftWLbnd[SLstepinbnd].wlevs.size(); n++)
			{
				zsbndleft.push_back(interptime(leftWLbnd[SLstepinbnd].wlevs[n], leftWLbnd[SLstepinbnd - 1].wlevs[n], leftWLbnd[SLstepinbnd].time - leftWLbnd[SLstepinbnd - 1].time, XParam.totaltime - leftWLbnd[SLstepinbnd - 1].time));

			}

			leftdirichletCPU(nx, ny, XParam.g, zsbndleft, zs, zb, hh, uu, vv);
		}
	}
	if (XParam.left == 0)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			//
			dim3 blockDim(16, 1, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
			dim3 gridDim(ceil((ny*1.0f) / blockDim.x), 1, 1);
			noslipbndLeft << <gridDim, blockDim, 0 >> > (nx, ny, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			// Left Wall
			noslipbndLeftCPU(XParam.nx, XParam.ny, XParam.eps, zb, zs, hh, uu, vv);
		}
	}
	//else neumann bnd (is already built in the solver)
}

void RightFlowBnd(Param XParam, std::vector<SLTS> rightWLbnd)
{
	//
	int nx = XParam.nx;
	int ny = XParam.ny;
	if (XParam.right == 1 && !rightWLbnd.empty())
	{
		int SLstepinbnd = 1;

		



		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = rightWLbnd[SLstepinbnd].time - XParam.totaltime;

		while (difft < 0.0)
		{
			SLstepinbnd++;
			difft = rightWLbnd[SLstepinbnd].time - XParam.totaltime;
		}

		

		dim3 blockDim(16, 1, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
		dim3 gridDim(ceil((ny*1.0f) / blockDim.x), 1, 1);
		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			float itime = SLstepinbnd - 1.0 + (XParam.totaltime - rightWLbnd[SLstepinbnd - 1].time) / (rightWLbnd[SLstepinbnd].time - rightWLbnd[SLstepinbnd - 1].time);

			rightdirichlet << <gridDim, blockDim, 0 >> > (nx, ny, rightWLbnd[0].wlevs.size(), XParam.g, itime, zs_g, zb_g, hh_g, uu_g, vv_g);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			std::vector<float> zsbndright;
			for (int n = 0; n < rightWLbnd[SLstepinbnd].wlevs.size(); n++)
			{
				zsbndright.push_back(interptime(rightWLbnd[SLstepinbnd].wlevs[n], rightWLbnd[SLstepinbnd - 1].wlevs[n], rightWLbnd[SLstepinbnd].time - rightWLbnd[SLstepinbnd - 1].time, XParam.totaltime - rightWLbnd[SLstepinbnd - 1].time));

			}

			rightdirichletCPU(nx, ny, XParam.g, zsbndright, zs, zb, hh, uu, vv);
		}
	}
	if (XParam.right == 0)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			//
			dim3 blockDim(16, 1, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
			dim3 gridDim(ceil((ny*1.0f) / blockDim.x), 1, 1);
			noslipbndRight << <gridDim, blockDim, 0 >> > (nx, ny, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			// Left Wall
			noslipbndRightCPU(XParam.nx, XParam.ny, XParam.eps, zb, zs, hh, uu, vv);
		}
	}
	//else neumann bnd (is already built in the algorithm)
}

void TopFlowBnd(Param XParam, std::vector<SLTS> topWLbnd)
{
	//
	int nx = XParam.nx;
	int ny = XParam.ny;
	if (XParam.top == 1 && !topWLbnd.empty())
	{
		int SLstepinbnd = 1;





		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = topWLbnd[SLstepinbnd].time - XParam.totaltime;

		while (difft < 0.0)
		{
			SLstepinbnd++;
			difft = topWLbnd[SLstepinbnd].time - XParam.totaltime;
		}


		dim3 blockDim(16, 1, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
		dim3 gridDim(ceil((nx*1.0f) / blockDim.x), 1, 1);
		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			float itime = SLstepinbnd - 1.0 + (XParam.totaltime - topWLbnd[SLstepinbnd - 1].time) / (topWLbnd[SLstepinbnd].time - topWLbnd[SLstepinbnd - 1].time);

			topdirichlet << <gridDim, blockDim, 0 >> > (nx, ny, topWLbnd[0].wlevs.size(), XParam.g, itime, zs_g, zb_g, hh_g, uu_g, vv_g);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			std::vector<float> zsbndtop;
			for (int n = 0; n < topWLbnd[SLstepinbnd].wlevs.size(); n++)
			{
				zsbndtop.push_back(interptime(topWLbnd[SLstepinbnd].wlevs[n], topWLbnd[SLstepinbnd - 1].wlevs[n], topWLbnd[SLstepinbnd].time - topWLbnd[SLstepinbnd - 1].time, XParam.totaltime - topWLbnd[SLstepinbnd - 1].time));

			}

			topdirichletCPU(nx, ny, XParam.g, zsbndtop, zs, zb, hh, uu, vv);
		}
	}
	if (XParam.top == 0)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			//
			dim3 blockDim(16, 1, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
			dim3 gridDim(ceil((nx*1.0f) / blockDim.x), 1, 1);
			noslipbndTop << <gridDim, blockDim, 0 >> > (nx, ny, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			// Left Wall
			noslipbndTopCPU(XParam.nx, XParam.ny, XParam.eps, zb, zs, hh, uu, vv);
		}
	}
	//else neumann bnd (is already built in the algorithm)
}

void BotFlowBnd(Param XParam, std::vector<SLTS> botWLbnd)
{
	//
	int nx = XParam.nx;
	int ny = XParam.ny;
	if (XParam.bot == 1 && !botWLbnd.empty())
	{
		int SLstepinbnd = 1;





		// Do this for all the corners
		//Needs limiter in case WLbnd is empty
		double difft = botWLbnd[SLstepinbnd].time - XParam.totaltime;

		while (difft < 0.0)
		{
			SLstepinbnd++;
			difft = botWLbnd[SLstepinbnd].time - XParam.totaltime;
		}

		

		dim3 blockDim(16, 1, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
		dim3 gridDim(ceil((nx*1.0f) / blockDim.x), 1, 1);
		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			float itime = SLstepinbnd - 1.0 + (XParam.totaltime - botWLbnd[SLstepinbnd - 1].time) / (botWLbnd[SLstepinbnd].time - botWLbnd[SLstepinbnd - 1].time);

			botdirichlet << <gridDim, blockDim, 0 >> > (nx, ny, botWLbnd[0].wlevs.size(), XParam.g, itime, zs_g, zb_g, hh_g, uu_g, vv_g);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			std::vector<float> zsbndbot;
			for (int n = 0; n < botWLbnd[SLstepinbnd].wlevs.size(); n++)
			{
				zsbndbot.push_back(interptime(botWLbnd[SLstepinbnd].wlevs[n], botWLbnd[SLstepinbnd - 1].wlevs[n], botWLbnd[SLstepinbnd].time - botWLbnd[SLstepinbnd - 1].time, XParam.totaltime - botWLbnd[SLstepinbnd - 1].time));

			}

			botdirichletCPU(nx, ny, XParam.g, zsbndbot, zs, zb, hh, uu, vv);
		}
	}
	if (XParam.bot == 0)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			//
			dim3 blockDim(16, 1, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
			dim3 gridDim(ceil((nx*1.0f) / blockDim.x), 1, 1);
			noslipbndBot << <gridDim, blockDim, 0 >> > (nx, ny, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			// Left Wall
			noslipbndBotCPU(XParam.nx, XParam.ny, XParam.eps, zb, zs, hh, uu, vv);
		}
	}
	//else neumann bnd (is already built in the algorithm)
}

float FlowGPU(Param XParam, float nextoutputtime)
{
	int nx = XParam.nx;
	int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);

	
	dtmax = 1 / epsilon;
	float dtmaxtmp = dtmax;

	resetdtmax << <gridDim, blockDim, 0 >> > (nx, ny, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	//update step 1

	// calculate gradients
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hh_g, dhdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hh_g, dhdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, zs_g, dzsdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, zs_g, dzsdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, uu_g, dudx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, uu_g, dudy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, vv_g, dvdx_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, vv_g, dvdy_g);
	// Test whether it is better to have one here or later (are the instuctions overlap if occupancy and meme acess is available?)
	CUDA_CHECK(cudaDeviceSynchronize());


	updateKurgX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.delta, XParam.g, XParam.eps, XParam.CFL, hh_g, zs_g, uu_g, vv_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());


	updateKurgY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.delta, XParam.g, XParam.eps, XParam.CFL, hh_g, zs_g, uu_g, vv_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);
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
	

	//float diffdt = mindtmaxB - mindtmax;
	XParam.dt = mindtmaxB;
	if (ceil((nextoutputtime - XParam.totaltime) / XParam.dt)> 0.0)
	{
		XParam.dt = (nextoutputtime - XParam.totaltime) / ceil((nextoutputtime - XParam.totaltime) / XParam.dt);
	}
	//printf("dt=%f\n", XParam.dt);


	updateEV << <gridDim, blockDim, 0 >> >(nx, ny, XParam.delta, XParam.g, hh_g, uu_g, vv_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	


	//predictor (advance 1/2 dt)
	Advkernel << <gridDim, blockDim, 0 >> >(nx, ny, XParam.dt*0.5, XParam.eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//corrector setp
	//update again
	// calculate gradients
	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hho_g, dhdx_g);
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hho_g, dhdy_g);

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, zso_g, dzsdx_g);
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, zso_g, dzsdy_g);

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, uuo_g, dudx_g);
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, uuo_g, dudy_g);

	gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, vvo_g, dvdx_g);
	gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, vvo_g, dvdy_g);
	// Test whether it is better to have one here or later (are the instuctions overlap if occupancy and meme acess is available?)
	CUDA_CHECK(cudaDeviceSynchronize());


	updateKurgX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.delta, XParam.g, XParam.eps, XParam.CFL, hho_g, zso_g, uuo_g, vvo_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	
	updateKurgY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.delta, XParam.g, XParam.eps, XParam.CFL, hho_g, zso_g, uuo_g, vvo_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	// no reduction of dtmax during the corrector step

	updateEV << <gridDim, blockDim, 0 >> >(nx, ny, XParam.delta, XParam.g, hho_g, uuo_g, vvo_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//
	Advkernel << <gridDim, blockDim, 0 >> >(nx, ny, XParam.dt, XParam.eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
	cleanupGPU << <gridDim, blockDim, 0 >> >(nx, ny, hho_g, zso_g, uuo_g, vvo_g, hh_g, zs_g, uu_g, vv_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//Bottom friction
	quadfriction << <gridDim, blockDim, 0 >> > (nx, ny, XParam.dt, XParam.eps, XParam.cf, hh_g, uu_g, vv_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Impose no slip condition by default
	//noslipbndall << <gridDim, blockDim, 0 >> > (nx, ny, XParam.dt, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
	//CUDA_CHECK(cudaDeviceSynchronize());
	return XParam.dt;
}

void meanmaxvarGPU(Param XParam)
{
	int nx = XParam.nx;
	int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	if (XParam.outuumean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >(nx, ny, uumean_g, uu_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outvvmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >(nx, ny, vvmean_g, vv_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outhhmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >(nx, ny, hhmean_g, hh_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outzsmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >(nx, ny, zsmean_g, zs_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outzsmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(nx, ny, zsmax_g, zs_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outhhmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(nx, ny, hhmax_g, hh_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outuumax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(nx, ny, uumax_g, uu_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outvvmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(nx, ny, vvmax_g, vv_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

}

void DivmeanvarGPU(Param XParam, float nstep)
{
	int nx = XParam.nx;
	int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	if (XParam.outuumean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >(nx, ny, nstep, uumean_g);
		CUDA_CHECK(cudaDeviceSynchronize());

		
	}
	if (XParam.outvvmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >(nx, ny, nstep, vvmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());

		
	}
	if (XParam.outhhmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >(nx, ny, nstep, hhmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());

		
	}
	if (XParam.outzsmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >(nx, ny, nstep, zsmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	
	

}
void ResetmeanvarGPU(Param XParam)
{
	int nx = XParam.nx;
	int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	if (XParam.outuumean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(nx, ny, uumean_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outvvmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(nx, ny,  vvmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outhhmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(nx, ny, hhmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outzsmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(nx, ny, zsmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}
// Main loop that actually runs the model
void mainloopGPU(Param XParam, std::vector<SLTS> leftWLbnd, std::vector<SLTS> rightWLbnd, std::vector<SLTS> topWLbnd, std::vector<SLTS> botWLbnd)
{
	float nextoutputtime = XParam.outputtimestep;
	int nstep = 0;
	int nTSsteps = 0;

	std::vector<Pointout> zsout;

	std::vector< std::vector< Pointout > > zsAllout;

	Pointout stepread;

	FILE * fsSLTS;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	dim3 gridDim(ceil((XParam.nx*1.0f) / blockDim.x), ceil((XParam.ny*1.0f) / blockDim.y), 1);
	


	for (int o = 0; o < XParam.TSoutfile.size(); o++)
	{
		//Overwrite existing files
		fsSLTS = fopen(XParam.TSoutfile[o].c_str(), "w");
		fprintf(fsSLTS, "# x=%f\ty=%f\ti=%d\tj=%d\t%s\n", XParam.TSnodesout[o].x, XParam.TSnodesout[o].y, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSoutfile[o].c_str());
		fclose(fsSLTS);

		// Add empty row for each output point
		zsAllout.push_back(std::vector<Pointout>());
	}



	while (XParam.totaltime < XParam.endtime)
	{
		// Bnd stuff here
		LeftFlowBnd(XParam, leftWLbnd);
		RightFlowBnd(XParam, rightWLbnd);
		TopFlowBnd(XParam, topWLbnd);
		BotFlowBnd(XParam, botWLbnd);

		// Run the model step
		XParam.dt = FlowGPU(XParam, nextoutputtime);
		
		//Time keeping
		XParam.totaltime = XParam.totaltime + XParam.dt;
		nstep++;
		
		// Do Sum & Max variables Here
		meanmaxvarGPU(XParam);


		//Check for TSoutput
		if (XParam.TSnodesout.size() > 0)
		{
			for (int o = 0; o < XParam.TSnodesout.size(); o++)
			{
				//
				stepread.time = XParam.totaltime;
				stepread.zs = 0.0;// a bit useless this
				stepread.hh = 0.0;
				stepread.uu = 0.0;
				stepread.vv = 0.0;
				zsAllout[o].push_back(stepread);

				storeTSout << <gridDim, blockDim, 0 >> > (XParam.nx, XParam.ny, XParam.TSnodesout.size(), o, nTSsteps, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, zs_g, hh_g, uu_g, vv_g, TSstore_g);
				CUDA_CHECK(cudaDeviceSynchronize());
			}
			nTSsteps++;
			
			if ((nTSsteps+1)*XParam.TSnodesout.size() * 4 > 2048 || XParam.endtime-XParam.totaltime <= XParam.dt*0.00001f)
			{
				//Flush
				CUDA_CHECK(cudaMemcpy(TSstore, TSstore_g, 2048 * sizeof(float), cudaMemcpyDeviceToHost));
				for (int o = 0; o < XParam.TSnodesout.size(); o++)
				{
					fsSLTS = fopen(XParam.TSoutfile[o].c_str(), "a");
					for (int n = 0; n < nTSsteps; n++)
					{
						//
						
						
							fprintf(fsSLTS, "%f\t%.4f\t%.4f\t%.4f\t%.4f\n", zsAllout[o][n].time, TSstore[1 + o * 4 + n*XParam.TSnodesout.size() * 4], TSstore[0 + o * 4 + n*XParam.TSnodesout.size() * 4], TSstore[2 + o * 4 + n*XParam.TSnodesout.size() * 4], TSstore[3 + o * 4 + n*XParam.TSnodesout.size() * 4]);
						
						
					}
					fclose(fsSLTS);
					//reset zsout
					zsAllout[o].clear();
				}
				nTSsteps = 0;

			}
			

		}

		if (nextoutputtime - XParam.totaltime <= XParam.dt*0.00001f  && XParam.outputtimestep > 0)
		{
			// Avg var sum here
			DivmeanvarGPU(XParam, nstep);

			if (XParam.outvort == 1)
			{
				CalcVorticity << <gridDim, blockDim, 0 >> > (XParam.nx, XParam.ny, vort_g, dvdx_g, dudy_g);
				CUDA_CHECK(cudaDeviceSynchronize());
			}

			if (!XParam.outvars.empty())
			{
				writenctimestep(XParam.outfile, XParam.totaltime);

				for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
				{
					if (OutputVarMaplen[XParam.outvars[ivar]] > 0)
					{
						if (XParam.GPUDEVICE >= 0)
						{
							//Should be async
							CUDA_CHECK(cudaMemcpy(OutputVarMapCPU[XParam.outvars[ivar]], OutputVarMapGPU[XParam.outvars[ivar]], OutputVarMaplen[XParam.outvars[ivar]] * sizeof(float), cudaMemcpyDeviceToHost));

						}
						//Create definition for each variable and store it
						writencvarstep(XParam.outfile, XParam.smallnc, XParam.scalefactor,XParam.addoffset, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
					}
				}
			}
			nextoutputtime = min(nextoutputtime + XParam.outputtimestep,XParam.endtime);

			printf("Writing output, totaltime:%f s, Mean dt=%f\n", XParam.totaltime, XParam.outputtimestep / nstep);
			write_text_to_log_file("Writing outputs, totaltime: " + std::to_string(XParam.totaltime) + ", Mean dt= " + std::to_string(XParam.outputtimestep / nstep));

			//.Reset Avg Variables
			ResetmeanvarGPU(XParam);


			//

			// Reset nstep
			nstep = 0;
		}

	}
}




void mainloopCPU(Param XParam, std::vector<SLTS> leftWLbnd, std::vector<SLTS> rightWLbnd, std::vector<SLTS> topWLbnd, std::vector<SLTS> botWLbnd)
{
	float nextoutputtime = XParam.outputtimestep;
	int nstep = 0;

	int nTSstep = 0;


	std::vector<Pointout> zsout;

	std::vector< std::vector< Pointout > > zsAllout;

	Pointout stepread;

	FILE * fsSLTS;

	for (int o = 0; o < XParam.TSoutfile.size(); o++)
	{
		//Overwrite existing files
		fsSLTS = fopen(XParam.TSoutfile[o].c_str(), "w");
		fprintf(fsSLTS, "# x=%f\ty=%f\ti=%d\tj=%d\t%s\n", XParam.TSnodesout[o].x, XParam.TSnodesout[o].y, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSoutfile[o].c_str());
		fclose(fsSLTS);

		// Add empty row for each output point
		zsAllout.push_back(std::vector<Pointout>());
	}

	while (XParam.totaltime < XParam.endtime)
	{
		// Bnd stuff here
		LeftFlowBnd(XParam, leftWLbnd);
		RightFlowBnd(XParam, rightWLbnd);
		TopFlowBnd(XParam, topWLbnd);
		BotFlowBnd(XParam, botWLbnd);


		// Run the model step
		XParam.dt = FlowCPU(XParam, nextoutputtime);

		//Time keeping
		XParam.totaltime = XParam.totaltime + XParam.dt;
		nstep++;

		// Do Sum & Max variables Here
		AddmeanCPU(XParam);
		maxallCPU(XParam);
		//Check for TSoutput
		if (XParam.TSnodesout.size() > 0)
		{
			for (int o = 0; o < XParam.TSnodesout.size(); o++)
			{
				//
				stepread.time = XParam.totaltime;
				stepread.zs = zs[XParam.TSnodesout[o].i + XParam.TSnodesout[o].j*XParam.nx];
				stepread.hh = hh[XParam.TSnodesout[o].i + XParam.TSnodesout[o].j*XParam.nx];
				stepread.uu = uu[XParam.TSnodesout[o].i + XParam.TSnodesout[o].j*XParam.nx];
				stepread.vv = vv[XParam.TSnodesout[o].i + XParam.TSnodesout[o].j*XParam.nx];
				zsAllout[o].push_back(stepread);

			}
			nTSstep++;

		}
		// CHeck for grid output
		if (nextoutputtime - XParam.totaltime <= XParam.dt*0.00001f  && XParam.outputtimestep > 0)
		{
			// Avg var sum here
			DivmeanCPU(XParam, (float)nstep);
			// Check for and calculate Vorticity if required
			if (XParam.outvort == 1)
			{
				CalcVort(XParam);
			}

			if (!XParam.outvars.empty())
			{
				writenctimestep(XParam.outfile, XParam.totaltime);

				for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
				{
					if (OutputVarMaplen[XParam.outvars[ivar]] > 0)
					{
						
						//Create definition for each variable and store it
						writencvarstep(XParam.outfile, XParam.smallnc, XParam.scalefactor, XParam.addoffset, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
					}
				}
			}
			nextoutputtime = min(nextoutputtime + XParam.outputtimestep, XParam.endtime);

			printf("Writing output, totaltime:%f s, Mean dt=%f\n", XParam.totaltime, XParam.outputtimestep / nstep);
			write_text_to_log_file("Writing outputs, totaltime: " + std::to_string(XParam.totaltime) + ", Mean dt= " + std::to_string(XParam.outputtimestep / nstep));

			//.Reset Avg Variables
			ResetmeanCPU(XParam);

			//
			if (!XParam.TSoutfile.empty())
			{
				for (int o = 0; o < XParam.TSoutfile.size(); o++)
				{
					//Overwrite existing files
					fsSLTS = fopen(XParam.TSoutfile[o].c_str(), "a");
					for (int n = 0; n < zsAllout[o].size(); n++)
					{
						fprintf(fsSLTS, "%f\t%.4f\t%.4f\t%.4f\t%.4f\n", zsAllout[o][n].time, zsAllout[o][n].zs, zsAllout[o][n].hh, zsAllout[o][n].uu, zsAllout[o][n].vv);
					}
					fclose(fsSLTS);
					//reset zsout
					zsAllout[o].clear();
					//zsAllout.push_back(std::vector<SLBnd>());
				}
			}
			// Reset nstep
			nstep = 0;
		}

		

	}
}




int main(int argc, char **argv)
{

	
	//Model starts Here//
	Param XParam;
	//The main function setups all the init of the model and then calls the mainloop to actually run the model


	//First part reads the inputs to the model 
	//then allocate memory on GPU and CPU
	//Then prepare and initialise memory and arrays on CPU and GPU
	// Prepare output file
	// Run main loop
	// Clean up and close


	// Start timer to keep track of time 
	XParam.startcputime = clock();



	// Reset the log file 
	FILE * flog;
	flog = fopen("BG_log.txt", "w"); //Find better name
	fclose(flog);

	//Logfile header
	time_t rawtime, dstart;
	struct tm * timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, 80, "%d-%m-%Y %H:%M:%S", timeinfo);
	std::string strtimenow(buffer);
	write_text_to_log_file("#################################");
	write_text_to_log_file("Basilisk-like Cartesian GPU v0.0");
	write_text_to_log_file("#################################");
	write_text_to_log_file("model started at " + strtimenow);


	//////////////////////////////////////////////////////
	/////             Read Operational file          /////
	//////////////////////////////////////////////////////


	std::ifstream fs("BG_param.txt");

	if (fs.fail()) {
		std::cerr << "BG_param.txt file could not be opened" << std::endl;
		write_text_to_log_file("ERROR: BG_param.txt file could not be opened...use this log file to create a file named BG_param.txt");
		SaveParamtolog(XParam);

		exit(1);
		
	}
	else
	{
		// Read and interpret each line of the BG_param.txt
		std::string line;
		while (std::getline(fs, line))
		{
			
			//Get param or skip empty lines
			if (!line.empty() && line.substr(0, 1).compare("#") != 0)
			{
				XParam = readparamstr(line, XParam);
				//std::cout << line << std::endl;
			}

		}
		fs.close();

		
	}





	std::string bathyext;
	
	//read bathy and perform sanity check
		
	if (!XParam.Bathymetryfile.empty())
	{
		printf("bathy: %s\n", XParam.Bathymetryfile.c_str());

		write_text_to_log_file("bathy: " + XParam.Bathymetryfile);

		std::vector<std::string> extvec = split(XParam.Bathymetryfile, '.');

		std::vector<std::string> nameelements;
		//by default we expect tab delimitation
		nameelements = split(extvec.back(), '?');
		if (nameelements.size() > 1)
		{
			//variable name for bathy is not given so it is assumed to be zb
			bathyext = nameelements[0];
		}
		else
		{
			bathyext = extvec.back();
		}

		
		write_text_to_log_file("bathy extension: " + bathyext);
		if (bathyext.compare("md") == 0)
		{
			write_text_to_log_file("Reading 'md' file");
			readbathyHead(XParam.Bathymetryfile, XParam.nx, XParam.ny, XParam.dx, XParam.grdalpha);
			
		}
		if (bathyext.compare("nc") == 0)
		{
			write_text_to_log_file("Reading bathy netcdf file");
			readgridncsize(XParam.Bathymetryfile, XParam.nx, XParam.ny, XParam.dx);
			write_text_to_log_file("For nc of bathy file please specify grdalpha in the BG_param.txt (default 0)");
			

		}
		if (bathyext.compare("dep") == 0 || bathyext.compare("bot") == 0)
		{
			//XBeach style file
			write_text_to_log_file("Reading " + bathyext + " file");
			write_text_to_log_file("For this type of bathy file please specify nx, ny, dx, xo, yo and grdalpha in the XBG_param.txt");
		}
		if (bathyext.compare("asc") == 0)
		{
			//
			write_text_to_log_file("Reading bathy asc file");
			readbathyASCHead(XParam.Bathymetryfile, XParam.nx, XParam.ny, XParam.dx, XParam.xo, XParam.yo, XParam.grdalpha);
			write_text_to_log_file("For asc of bathy file please specify grdalpha in the BG_param.txt (default 0)");
		}

		XParam.delta = XParam.dx;
		XParam.grdalpha = XParam.grdalpha*pi / 180; // grid rotation

													//fid = fopen(XParam.Bathymetryfile.c_str(), "r");
													//fscanf(fid, "%u\t%u\t%lf\t%*f\t%lf", &XParam.nx, &XParam.ny, &XParam.dx, &XParam.grdalpha);
		printf("nx=%d\tny=%d\tdx=%f\talpha=%f\n", XParam.nx, XParam.ny, XParam.dx, XParam.grdalpha * 180 / pi);
		write_text_to_log_file("nx=" + std::to_string(XParam.nx) + " ny=" + std::to_string(XParam.ny) + " dx=" + std::to_string(XParam.dx) + " grdalpha=" + std::to_string(XParam.grdalpha*180.0 / pi));


		/////////////////////////////////////////////////////
		////// CHECK PARAMETER SANITY
		/////////////////////////////////////////////////////
		XParam = checkparamsanity(XParam);





	}
	else
	{
		std::cerr << "Fatal error: No bathymetry file specified. Please specify using 'bathy = Filename.bot'" << std::endl;
		write_text_to_log_file("Fatal error : No bathymetry file specified. Please specify using 'bathy = Filename.md'");
		exit(1);
	}

	//////////////////////////////////////////////////
	////// Preprare Bnd
	//////////////////////////////////////////////////

	// So far bnd are limited to be cst along an edge
	// Read Bnd file if/where needed
	std::vector<SLTS> leftWLbnd;
	std::vector<SLTS> rightWLbnd;
	std::vector<SLTS> topWLbnd;
	std::vector<SLTS> botWLbnd;

	if (!XParam.leftbndfile.empty())
	{
		leftWLbnd = readWLfile(XParam.leftbndfile);
		
	}
	if (!XParam.rightbndfile.empty())
	{
		rightWLbnd = readWLfile(XParam.rightbndfile);
	}
	if (!XParam.topbndfile.empty())
	{
		topWLbnd = readWLfile(XParam.topbndfile);
	}
	if (!XParam.botbndfile.empty())
	{
		botWLbnd = readWLfile(XParam.botbndfile);
	}

	XParam.endtime = setendtime(XParam, leftWLbnd, rightWLbnd, topWLbnd, botWLbnd);

	XParam.dt = 0.0;// Will be resolved in update

	////////////////////////////////////////////////
	///// Allocate memory on CPU
	////////////////////////////////////////////////

	int nx = XParam.nx;
	int ny = XParam.ny;

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
	//not allocating below may be usefull

	if (XParam.outhhmax == 1)
	{
		hhmax = (float *)malloc(nx*ny * sizeof(float));
	}
	if (XParam.outuumax == 1)
	{
		uumax = (float *)malloc(nx*ny * sizeof(float));
	}
	if (XParam.outvvmax == 1)
	{
		vvmax = (float *)malloc(nx*ny * sizeof(float));
	}
	if (XParam.outzsmax == 1)
	{
		zsmax = (float *)malloc(nx*ny * sizeof(float));
	}
	
	if (XParam.outhhmean == 1)
	{
		hhmean = (float *)malloc(nx*ny * sizeof(float));
	}
	if (XParam.outzsmean == 1)
	{
		zsmean = (float *)malloc(nx*ny * sizeof(float));
	}
	if (XParam.outuumean == 1)
	{
		uumean = (float *)malloc(nx*ny * sizeof(float));
	}
	if (XParam.outvvmean == 1)
	{
		vvmean = (float *)malloc(nx*ny * sizeof(float));
	}

	if (XParam.outvort == 1)
	{
		vort = (float *)malloc(nx*ny * sizeof(float));
	}




	if (XParam.GPUDEVICE >= 0)
	{
		// Init GPU
		// This should be in the sanity check
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		cudaDeviceProp prop;

		if (XParam.GPUDEVICE > (nDevices - 1))
		{
			// 
			XParam.GPUDEVICE = (nDevices - 1);
		}

	}

	// Now that we checked that there was indeed a GPU available
	////////////////////////////////////////
	//////// ALLLOCATE GPU memory
	////////////////////////////////////////
	if (XParam.GPUDEVICE >= 0)
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
		

		if (XParam.outhhmax == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&hhmax_g, nx*ny*sizeof(float)));
		}
		if (XParam.outzsmax == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&zsmax_g, nx*ny*sizeof(float)));
		}
		if (XParam.outuumax == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&uumax_g, nx*ny*sizeof(float)));
		}
		if (XParam.outvvmax == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&vvmax_g, nx*ny*sizeof(float)));
		}
		if (XParam.outhhmean == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&hhmean_g, nx*ny*sizeof(float)));
		}
		if (XParam.outzsmean == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&zsmean_g, nx*ny*sizeof(float)));
		}
		if (XParam.outuumean == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&uumean_g, nx*ny*sizeof(float)));
		}
		if (XParam.outvvmean == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&vvmean_g, nx*ny*sizeof(float)));
		}

		if (XParam.outvort == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&vort_g, nx*ny*sizeof(float)));
		}

		if (XParam.TSnodesout.size() > 0)
		{
			// Allocate mmemory to store TSoutput in between writing to disk
			int nTS = 1; // Nb of points
			int nvts = 1; // NB of variables hh, zs, uu, vv
			int nstore = 2048; //store up to 2048 pts
			TSstore = (float *)malloc(nTS*nvts*nstore * sizeof(float));
			CUDA_CHECK(cudaMalloc((void **)&TSstore_g, nTS*nvts*nstore*sizeof(float)));
			//Cpu part done differently because there are no latency issue (i.e. none that I care about) 

		}

		if (!XParam.leftbndfile.empty())
		{
			//leftWLbnd = readWLfile(XParam.leftbndfile);
			//Flatten bnd to copy to cuda array
			int nbndtimes = leftWLbnd.size();
			int nbndvec = leftWLbnd[0].wlevs.size();
			CUDA_CHECK(cudaMallocArray(&leftWLS_gp, &channelDescleftbnd, nbndtimes, nbndvec));

			float * leftWLS;
			leftWLS=(float *)malloc(nbndtimes * nbndvec * sizeof(float));

			for (int ibndv = 0; ibndv < nbndvec; ibndv++)
			{
				for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
				{
					//
					leftWLS[ibndt + ibndv*nbndtimes] = leftWLbnd[ibndt].wlevs[ibndv];
				}
			}
			CUDA_CHECK(cudaMemcpyToArray(leftWLS_gp, 0, 0, leftWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

			texLBND.addressMode[0] = cudaAddressModeClamp;
			texLBND.addressMode[1] = cudaAddressModeClamp;
			texLBND.filterMode = cudaFilterModeLinear;
			texLBND.normalized = false;


			CUDA_CHECK(cudaBindTextureToArray(texLBND, leftWLS_gp, channelDescleftbnd));
			free(leftWLS);

		}
		if (!XParam.rightbndfile.empty())
		{
			//leftWLbnd = readWLfile(XParam.leftbndfile);
			//Flatten bnd to copy to cuda array
			int nbndtimes = rightWLbnd.size();
			int nbndvec = rightWLbnd[0].wlevs.size();
			CUDA_CHECK(cudaMallocArray(&rightWLS_gp, &channelDescrightbnd, nbndtimes, nbndvec));

			float * rightWLS;
			rightWLS = (float *)malloc(nbndtimes * nbndvec * sizeof(float));

			for (int ibndv = 0; ibndv < nbndvec; ibndv++)
			{
				for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
				{
					//
					rightWLS[ibndt + ibndv*nbndtimes] = rightWLbnd[ibndt].wlevs[ibndv];
				}
			}
			CUDA_CHECK(cudaMemcpyToArray(rightWLS_gp, 0, 0, rightWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

			texRBND.addressMode[0] = cudaAddressModeClamp;
			texRBND.addressMode[1] = cudaAddressModeClamp;
			texRBND.filterMode = cudaFilterModeLinear;
			texRBND.normalized = false;


			CUDA_CHECK(cudaBindTextureToArray(texRBND, rightWLS_gp, channelDescrightbnd));
			free(rightWLS);

		}
		if (!XParam.topbndfile.empty())
		{
			//leftWLbnd = readWLfile(XParam.leftbndfile);
			//Flatten bnd to copy to cuda array
			int nbndtimes = topWLbnd.size();
			int nbndvec = topWLbnd[0].wlevs.size();
			CUDA_CHECK(cudaMallocArray(&topWLS_gp, &channelDesctopbnd, nbndtimes, nbndvec));

			float * topWLS;
			topWLS = (float *)malloc(nbndtimes * nbndvec * sizeof(float));

			for (int ibndv = 0; ibndv < nbndvec; ibndv++)
			{
				for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
				{
					//
					topWLS[ibndt + ibndv*nbndtimes] = topWLbnd[ibndt].wlevs[ibndv];
				}
			}
			CUDA_CHECK(cudaMemcpyToArray(topWLS_gp, 0, 0, topWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

			texTBND.addressMode[0] = cudaAddressModeClamp;
			texTBND.addressMode[1] = cudaAddressModeClamp;
			texTBND.filterMode = cudaFilterModeLinear;
			texTBND.normalized = false;


			CUDA_CHECK(cudaBindTextureToArray(texTBND, topWLS_gp, channelDesctopbnd));
			free(topWLS);

		}
		if (!XParam.botbndfile.empty())
		{
			//leftWLbnd = readWLfile(XParam.leftbndfile);
			//Flatten bnd to copy to cuda array
			int nbndtimes = botWLbnd.size();
			int nbndvec = botWLbnd[0].wlevs.size();
			CUDA_CHECK(cudaMallocArray(&botWLS_gp, &channelDescbotbnd, nbndtimes, nbndvec));

			float * botWLS;
			botWLS = (float *)malloc(nbndtimes * nbndvec * sizeof(float));

			for (int ibndv = 0; ibndv < nbndvec; ibndv++)
			{
				for (int ibndt = 0; ibndt < nbndtimes; ibndt++)
				{
					//
					botWLS[ibndt + ibndv*nbndtimes] = botWLbnd[ibndt].wlevs[ibndv];
				}
			}
			CUDA_CHECK(cudaMemcpyToArray(botWLS_gp, 0, 0, botWLS, nbndtimes * nbndvec * sizeof(float), cudaMemcpyHostToDevice));

			texBBND.addressMode[0] = cudaAddressModeClamp;
			texBBND.addressMode[1] = cudaAddressModeClamp;
			texBBND.filterMode = cudaFilterModeLinear;
			texBBND.normalized = false;


			CUDA_CHECK(cudaBindTextureToArray(texBBND, botWLS_gp, channelDescbotbnd));
			free(botWLS);

		}

	}

	if (bathyext.compare("md") == 0)
	{
		readbathy(XParam.Bathymetryfile, zb);
	}
	if (bathyext.compare("nc") == 0)
	{
		readnczb(XParam.nx, XParam.ny, XParam.Bathymetryfile, zb);
	}
	if (bathyext.compare("bot") == 0 || bathyext.compare("dep") == 0)
	{
		readXBbathy(XParam.Bathymetryfile, XParam.nx, XParam.ny, zb);
	}
	if (bathyext.compare("asc") == 0)
	{
		//
		readbathyASCzb(XParam.Bathymetryfile, XParam.nx, XParam.ny, zb);
	}

	//printf("%f\n", zb[0]);
	//printf("%f\n", zb[(nx - 1) + (0)*nx]);
	//printf("%f\n", zb[(0) + (ny-1)*nx]);
	//printf("%f\n", zb[(nx - 1) + (ny - 1)*nx]);
	

	//init variables
	if (XParam.posdown == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				zb[i + j*nx] = zb[i + j*nx] * -1.0f;
				//printf("%f\n", zb[i + (j)*nx]);
				
			}
		}
	}

	/////////////////////////////////////////////////////
	// Initial Condition
	/////////////////////////////////////////////////////


	int hotstartsucess = 0;
	if (!XParam.hotstartfile.empty())
	{
		// hotstart
		hotstartsucess = readhotstartfile(XParam, zs, zb, hh, uu, vv);
	}
	if (XParam.hotstartfile.empty() || hotstartsucess == 0)
	{
		//Cold start
		// 2 options: 
		//		(1) if zsinit is set, then apply zsinit everywhere
		//		(2) zsinit is not set so interpolate from boundaries. (if no boundaries were specified set zsinit to zeros and apply case (1))

		Param defaultParam;
		//!leftWLbnd.empty()
		
		//case 2b (i.e. zsinint and no boundaries were specified)
		if ((abs(XParam.zsinit - defaultParam.zsinit) <= epsilon) && (leftWLbnd.empty() && rightWLbnd.empty() && topWLbnd.empty() && botWLbnd.empty()) ) //zsinit is default
		{
			XParam.zsinit = 0.0; // better default value
		}

		//case(1)
		if (abs(XParam.zsinit - defaultParam.zsinit) > epsilon) // apply specified zsinit
		{
			for (int j = 0; j < ny; j++)
			{
				for (int i = 0; i < nx; i++)
				{

					uu[i + j*nx] = 0.0f;
					vv[i + j*nx] = 0.0f;
					//zb[i + j*nx] = 0.0f;
					zs[i + j*nx] = max((float)XParam.zsinit, zb[i + j*nx]);
					//if (i >= 64 && i < 82)
					//{
					//	zs[i + j*nx] = max(zsbnd+0.2f, zb[i + j*nx]);
					//}
					hh[i + j*nx] = max(zs[i + j*nx] - zb[i + j*nx], (float)XParam.eps);


				}
			}

		}
		else // lukewarm start i.e. bilinear interpolation of zs
		{
			float zsleft = 0.0;
			float zsright = 0.0;
			float zstop = 0.0;
			float zsbot = 0.0;
			float zsbnd = 0.0;

			float distleft, distright, disttop, distbot;

			float lefthere = 0.0f;
			float righthere = 0.0f;
			float tophere = 0.0f;
			float bothere = 0.0f;


			for (int j = 0; j < ny; j++)
			{
				disttop = max((float)(ny - 1) - j, 0.1f);
				
				distbot = max((float) j, 0.1f);

				if (XParam.left == 1 && !leftWLbnd.empty())
				{
					lefthere = 1.0f;
					int SLstepinbnd = 1;



					// Do this for all the corners
					//Needs limiter in case WLbnd is empty
					double difft = leftWLbnd[SLstepinbnd].time - XParam.totaltime;

					while (difft < 0.0)
					{
						SLstepinbnd++;
						difft = leftWLbnd[SLstepinbnd].time - XParam.totaltime;
					}
					std::vector<float> zsbndvec;
					for (int n = 0; n < leftWLbnd[SLstepinbnd].wlevs.size(); n++)
					{
						zsbndvec.push_back(interptime(leftWLbnd[SLstepinbnd].wlevs[n], leftWLbnd[SLstepinbnd - 1].wlevs[n], leftWLbnd[SLstepinbnd].time - leftWLbnd[SLstepinbnd - 1].time, XParam.totaltime - leftWLbnd[SLstepinbnd - 1].time));

					}
					if (zsbndvec.size() == 1)
					{
						zsleft = zsbndvec[0];
					}
					else
					{
						int iprev = min(max((int)ceil(j / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
						int inext = iprev + 1;
						// here interp time is used to interpolate to the right node rather than in time...
						zsleft = interptime(zsbndvec[inext], zsbndvec[iprev], (float)(inext - iprev), (float)(j - iprev));
					}

				}
				
				if (XParam.right == 1 && !rightWLbnd.empty())
				{
					int SLstepinbnd = 1;
					righthere = 1.0f;


					// Do this for all the corners
					//Needs limiter in case WLbnd is empty
					double difft = rightWLbnd[SLstepinbnd].time - XParam.totaltime;

					while (difft < 0.0)
					{
						SLstepinbnd++;
						difft = rightWLbnd[SLstepinbnd].time - XParam.totaltime;
					}
					std::vector<float> zsbndvec;
					for (int n = 0; n < rightWLbnd[SLstepinbnd].wlevs.size(); n++)
					{
						zsbndvec.push_back(interptime(rightWLbnd[SLstepinbnd].wlevs[n], rightWLbnd[SLstepinbnd - 1].wlevs[n], rightWLbnd[SLstepinbnd].time - rightWLbnd[SLstepinbnd - 1].time, XParam.totaltime - rightWLbnd[SLstepinbnd - 1].time));

					}
					if (zsbndvec.size() == 1)
					{
						zsright = zsbndvec[0];
					}
					else
					{
						int iprev = min(max((int)ceil(j / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
						int inext = iprev + 1;
						// here interp time is used to interpolate to the right node rather than in time...
						zsright = interptime(zsbndvec[inext], zsbndvec[iprev], (float)(inext - iprev), (float)(j - iprev));
					}


				}
				
				
				
				

				for (int i = 0; i < nx; i++)
				{
					distleft = max((float)i,0.1f);
					distright = max((float)(nx - 1) - i, 0.1f);

					if (XParam.bot == 1 && !botWLbnd.empty())
					{
						int SLstepinbnd = 1;
						bothere = 1.0;




						// Do this for all the corners
						//Needs limiter in case WLbnd is empty
						double difft = botWLbnd[SLstepinbnd].time - XParam.totaltime;

						while (difft < 0.0)
						{
							SLstepinbnd++;
							difft = botWLbnd[SLstepinbnd].time - XParam.totaltime;
						}
						std::vector<float> zsbndvec;
						for (int n = 0; n < botWLbnd[SLstepinbnd].wlevs.size(); n++)
						{
							zsbndvec.push_back(interptime(botWLbnd[SLstepinbnd].wlevs[n], botWLbnd[SLstepinbnd - 1].wlevs[n], botWLbnd[SLstepinbnd].time - botWLbnd[SLstepinbnd - 1].time, XParam.totaltime - botWLbnd[SLstepinbnd - 1].time));

						}
						if (zsbndvec.size() == 1)
						{
							zsbot = zsbndvec[0];
						}
						else
						{
							int iprev = min(max((int)ceil(i / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
							int inext = iprev + 1;
							// here interp time is used to interpolate to the right node rather than in time...
							zsbot = interptime(zsbndvec[inext], zsbndvec[iprev], (float)(inext - iprev), (float)(i - iprev));
						}

					}
					if (XParam.top == 1 && !topWLbnd.empty())
					{
						int SLstepinbnd = 1;
						tophere = 1.0f;




						// Do this for all the corners
						//Needs limiter in case WLbnd is empty
						double difft = topWLbnd[SLstepinbnd].time - XParam.totaltime;

						while (difft < 0.0)
						{
							SLstepinbnd++;
							difft = topWLbnd[SLstepinbnd].time - XParam.totaltime;
						}
						std::vector<float> zsbndvec;
						for (int n = 0; n < topWLbnd[SLstepinbnd].wlevs.size(); n++)
						{
							zsbndvec.push_back(interptime(topWLbnd[SLstepinbnd].wlevs[n], topWLbnd[SLstepinbnd - 1].wlevs[n], topWLbnd[SLstepinbnd].time - topWLbnd[SLstepinbnd - 1].time, XParam.totaltime - topWLbnd[SLstepinbnd - 1].time));

						}
						if (zsbndvec.size() == 1)
						{
							zstop = zsbndvec[0];
						}
						else
						{
							int iprev = min(max((int)ceil(i / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
							int inext = iprev + 1;
							// here interp time is used to interpolate to the right node rather than in time...
							zstop = interptime(zsbndvec[inext], zsbndvec[iprev], (float)(inext - iprev), (float)(i - iprev));
						}

					}
				
										

					//if (XParam.top == 1 && !topWLbnd.empty() && XParam.bot == 1 && !botWLbnd.empty() && XParam.left == 1 && !leftWLbnd.empty() && XParam.right == 1 && !rightWLbnd.empty())
					//{
					//	zsbnd = (zsleft*(1 / i) + zsright * 1 / (nx - i) + zsbot * 1 / j + zstop * 1 / (ny - j)) / ((1 / i) + 1 / (nx - i) + 1 / j + 1 / (ny - j));
					//}
					
					zsbnd = ((zsleft * 1 / distleft)*lefthere + (zsright * 1 / distright)*righthere + (zstop * 1 / disttop)*tophere + (zsbot * 1 / distbot)*bothere) / ((1 / distleft)*lefthere + (1 / distright)*righthere + (1 / disttop)*tophere + (1 / distbot)*bothere);
					
					zs[i + j*nx] = max(zsbnd, zb[i + j*nx]);
					hh[i + j*nx] = max(zs[i + j*nx] - zb[i + j*nx], (float)XParam.eps);
					uu[i + j*nx] = 0.0;
					vv[i + j*nx] = 0.0;

				}
			}


		}

		



		
	}
	// Below is not succint but way faster than one loop that checks teh if statemenst each time
	if (XParam.outhhmax == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				hhmax[i + j*nx] = hh[i + j*nx];
			}
		}
	}

	if (XParam.outhhmean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				hhmean[i + j*nx] = 0.0;
			}
		}
	}
	if (XParam.outzsmax == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				zsmax[i + j*nx] = zs[i + j*nx];
			}
		}
	}

	if (XParam.outzsmean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				zsmean[i + j*nx] = 0.0;
			}
		}
	}

	if (XParam.outuumax == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				uumax[i + j*nx] = uu[i + j*nx];
			}
		}
	}

	if (XParam.outuumean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				uumean[i + j*nx] = 0.0;
			}
		}
	}
	if (XParam.outvvmax == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				vvmax[i + j*nx] = vv[i + j*nx];
			}
		}
	}

	if (XParam.outvvmean == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				vvmean[i + j*nx] = 0.0;
			}
		}
	}
	if (XParam.outvort == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				vort[i + j*nx] = 0.0;
			}
		}
	}

	if (XParam.GPUDEVICE >= 0)
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

	// Here map array to their name as a string. it makes it super easy to convert user define variables to the array it represents.
	// COul add more to output gradients etc...
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

	OutputVarMapCPU["hhmean"] = hhmean;
	OutputVarMapGPU["hhmean"] = hhmean_g;
	OutputVarMaplen["hhmean"] = nx*ny;

	OutputVarMapCPU["hhmax"] = hhmax;
	OutputVarMapGPU["hhmax"] = hhmax_g;
	OutputVarMaplen["hhmax"] = nx*ny;

	OutputVarMapCPU["zsmean"] = zsmean;
	OutputVarMapGPU["zsmean"] = zsmean_g;
	OutputVarMaplen["zsmean"] = nx*ny;

	OutputVarMapCPU["zsmax"] = zsmax;
	OutputVarMapGPU["zsmax"] = zsmax_g;
	OutputVarMaplen["zsmax"] = nx*ny;

	OutputVarMapCPU["uumean"] = uumean;
	OutputVarMapGPU["uumean"] = uumean_g;
	OutputVarMaplen["uumean"] = nx*ny;

	OutputVarMapCPU["uumax"] = uumax;
	OutputVarMapGPU["uumax"] = uumax_g;
	OutputVarMaplen["uumax"] = nx*ny;

	OutputVarMapCPU["vvmean"] = vvmean;
	OutputVarMapGPU["vvmean"] = vvmean_g;
	OutputVarMaplen["vvmean"] = nx*ny;

	OutputVarMapCPU["vvmax"] = vvmax;
	OutputVarMapGPU["vvmax"] = vvmax_g;
	OutputVarMaplen["vvmax"] = nx*ny;

	OutputVarMapCPU["vort"] = vort;
	OutputVarMapGPU["vort"] = vort_g;
	OutputVarMaplen["vort"] = nx*ny;


	//create nc file with no variables
	XParam=creatncfileUD(XParam);
	for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
	{
		//Create definition for each variable and store it
		//defncvar(std::string outfile, int smallnc, float scalefactor, float addoffset, int nx, int ny, std::string varst, int vdim, float * var)
		defncvar(XParam.outfile, XParam.smallnc, XParam.scalefactor, XParam.addoffset, nx, ny, XParam.outvars[ivar], 3, OutputVarMapCPU[XParam.outvars[ivar]]);
	}
	//create2dnc(nx, ny, dx, dx, 0.0, xx, yy, hh);

	
		if (XParam.GPUDEVICE >= 0)
		{
			mainloopGPU(XParam, leftWLbnd, rightWLbnd, topWLbnd, botWLbnd);
			//checkloopGPU(XParam);
			
		}
		else
		{
			mainloopCPU(XParam, leftWLbnd, rightWLbnd, topWLbnd, botWLbnd);
		}

	
	



	XParam.endcputime = clock();
	printf("End Computation \n");
	write_text_to_log_file("End Computation" );

	printf("Total runtime= %d  seconds\n", (XParam.endcputime - XParam.startcputime) / CLOCKS_PER_SEC);
	write_text_to_log_file("Total runtime= " + std::to_string((XParam.endcputime - XParam.startcputime) / CLOCKS_PER_SEC) + "  seconds" );

	//if GPU?
	free(hh);
	free(uu);
	free(vv);
	free(zb);
	free(zs);

	free(hho);
	free(uuo);
	free(vvo);
	free(zso);

	free(dhdx);
	free(dhdy);
	free(dudx);
	free(dudy);
	free(dvdx);
	free(dvdy);

	free(dzsdx);
	free(dzsdy);

	free(Su);
	free(Sv);
	free(Fqux);
	free(Fquy);
	free(Fqvx);
	free(Fqvy);
	free(Fhu);
	free(Fhv);

	free(dh);
	free(dhu);
	free(dhv);

	




	if (XParam.GPUDEVICE >= 0)
	{
		cudaFree(hh_g);
		cudaFree(uu_g);
		cudaFree(vv_g);
		cudaFree(zb_g);
		cudaFree(zs_g);

		cudaFree(hho_g);
		cudaFree(uuo_g);
		cudaFree(vvo_g);
		cudaFree(zso_g);

		cudaFree(dhdx_g);
		cudaFree(dhdy_g);
		cudaFree(dudx_g);
		cudaFree(dudy_g);
		cudaFree(dvdx_g);
		cudaFree(dvdy_g);

		cudaFree(dzsdx_g);
		cudaFree(dzsdy_g);

		cudaFree(Su_g);
		cudaFree(Sv_g);
		cudaFree(Fqux_g);
		cudaFree(Fquy_g);
		cudaFree(Fqvx_g);
		cudaFree(Fqvy_g);
		cudaFree(Fhu_g);
		cudaFree(Fhv_g);

		cudaFree(dh_g);
		cudaFree(dhu_g);
		cudaFree(dhv_g);

		cudaFree(dtmax_g);

		
		cudaFree(arrmin_g);
		cudaFree(arrmax_g);

		cudaDeviceReset();

	}


	








	exit(0);
}


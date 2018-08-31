//////////////////////////////////////////////////////////////////////////////////
//						                                                        //
//Copyright (C) 2018 Bosserelle                                                 //
// This code contains an adaptation of the St Venant equation from Basilisk		//
// See																			//
// http://basilisk.fr/src/saint-venant.h and									//
// S. Popinet. Quadtree-adaptive tsunami modelling. Ocean Dynamics,				//
// doi: 61(9) : 1261 - 1285, 2011												//
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
double *zs_d, *hh_d, *zb_d, *uu_d, *vv_d; // double array only allocated instead of thge float if requested
float *zs_g, *hh_g, *zb_g, *uu_g, *vv_g; // for GPU
double *zs_gd, *hh_gd, *zb_gd, *uu_gd, *vv_gd;

float *zso, *hho, *uuo, *vvo;
double *zso_d, *hho_d, *uuo_d, *vvo_d;
float *zso_g, *hho_g, *uuo_g, *vvo_g; // for GPU
double *zso_gd, *hho_gd, *uuo_gd, *vvo_gd;
//CPU
float * dhdx, *dhdy, *dudx, *dudy, *dvdx, *dvdy;
float *dzsdx, *dzsdy;
//GPU
float * dhdx_g, *dhdy_g, *dudx_g, *dudy_g, *dvdx_g, *dvdy_g;
float *dzsdx_g, *dzsdy_g;
//double *fmu, *fmv;

double * dhdx_d, *dhdy_d, *dudx_d, *dudy_d, *dvdx_d, *dvdy_d;
double *dzsdx_d, *dzsdy_d;

double * dhdx_gd, *dhdy_gd, *dudx_gd, *dudy_gd, *dvdx_gd, *dvdy_gd;
double *dzsdx_gd, *dzsdy_gd;

float *Su, *Sv, *Fqux, *Fquy, *Fqvx, *Fqvy;
float * Fhu, *Fhv;
float * dh, *dhu, *dhv;

double *Su_d, *Sv_d, *Fqux_d, *Fquy_d, *Fqvx_d, *Fqvy_d;
double * Fhu_d, *Fhv_d;
double * dh_d, *dhu_d, *dhv_d;

//GPU
float *Su_g, *Sv_g, *Fqux_g, *Fquy_g, *Fqvx_g, *Fqvy_g;
float * Fhu_g, *Fhv_g;
float * dh_g, *dhu_g, *dhv_g;

double *Su_gd, *Sv_gd, *Fqux_gd, *Fquy_gd, *Fqvx_gd, *Fqvy_gd;
double * Fhu_gd, *Fhv_gd;
double * dh_gd, *dhu_gd, *dhv_gd;

float * TSstore, *TSstore_g;
double * TSstore_d, *TSstore_gd;

float * hhmean, *uumean, *vvmean, *zsmean;
float * hhmean_g, *uumean_g, *vvmean_g, *zsmean_g;
double * hhmean_d, *uumean_d, *vvmean_d, *zsmean_d;
double * hhmean_gd, *uumean_gd, *vvmean_gd, *zsmean_gd;

float * hhmax, *uumax, *vvmax, *zsmax;
float * hhmax_g, *uumax_g, *vvmax_g, *zsmax_g;
double * hhmax_d, *uumax_d, *vvmax_d, *zsmax_d;
double * hhmax_gd, *uumax_gd, *vvmax_gd, *zsmax_gd;

float * vort, *vort_g;// Vorticity output
double * vort_d, *vort_gd;

float dtmax = (float) (1.0 / epsilon);
double dtmax_d = 1.0 / epsilon;

double * dtmax_gd;
float * dtmax_g;

float *arrmax_g;
float *arrmin_g;
float *arrmin;

double *arrmax_gd;
double *arrmin_gd;
double *arrmin_d;

float * dummy;
double * dummy_d;

// Block info
float * blockxo, *blockyo;
double * blockxo_d, *blockyo_d;
int * leftblk, *rightblk, *topblk, *botblk;

double * blockxo_gd, *blockyo_gd;
float * blockxo_g, *blockyo_g;
int * leftblk_g, *rightblk_g, *topblk_g, *botblk_g;

//std::string outfile = "output.nc";
//std::vector<std::string> outvars;
std::map<std::string, float *> OutputVarMapCPU;
std::map<std::string, double *> OutputVarMapCPUD;
std::map<std::string, float *> OutputVarMapGPU;
std::map<std::string, double *> OutputVarMapGPUD;
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

template <class T> void Allocate1GPU(int nx, int ny, T *&zb_g)
{
	CUDA_CHECK(cudaMalloc((void **)&zb_g, nx*ny * sizeof(T)));
}
template <class T> void Allocate4GPU(int nx, int ny, T *&zs_g, T *&hh_g, T *&uu_g, T *&vv_g)
{
	CUDA_CHECK(cudaMalloc((void **)&zs_g, nx*ny * sizeof(T)));
	CUDA_CHECK(cudaMalloc((void **)&hh_g, nx*ny * sizeof(T)));
	CUDA_CHECK(cudaMalloc((void **)&uu_g, nx*ny * sizeof(T)));
	CUDA_CHECK(cudaMalloc((void **)&vv_g, nx*ny * sizeof(T)));
}

template <class T> void Allocate1CPU(int nx, int ny, T *&zb)
{
	zb = (T *)malloc(nx*ny * sizeof(T));
}

template <class T> void Allocate4CPU(int nx, int ny, T *&zs, T *&hh, T *&uu, T *&vv)
{
	
	zs = (T *)malloc(nx*ny * sizeof(T));
	hh = (T *)malloc(nx*ny * sizeof(T));
	uu = (T *)malloc(nx*ny * sizeof(T));
	vv = (T *)malloc(nx*ny * sizeof(T));
}

template <class T> void setedges(int nblk, int * leftblk, int *rightblk, int * topblk, int* botblk,  T *&zb)
{
	// template <class T> void setedges(int nblk, int nx, int ny, double xo, double yo, double dx, int * leftblk, int *rightblk, int * topblk, int* botblk, double *blockxo, double * blockyo, T *&zb)

	// here the bathy of the outter most cells of the domain are "set" to the same value as the second outter most.
	// this also applies to the blocks with no neighbour
	for (int bl = 0; bl < nblk; bl++)
	{
		/*
		if (blockxo[bl] == xo)//safe? in adaptive this should be xo-x+0.5dx*(2^lev-1) <= tiny
		{
			int i = 0;
			for (int j = 0; j < 16; j++)
			{

				zb[i + j * 16 + bl * 256] = zb[i+1 + j * 16 + bl * 256];
			}
		}
		if (blockxo[bl]+(15*dx) == xo+ (ceil(nx / 16.0)*16.0 - 1)*dx)//safe? in adaptive this should be xo-x+0.5dx*(2^lev-1) <= tiny
		{
			int i = 15;
			for (int j = 0; j < 16; j++)
			{

				zb[i + j * 16 + bl * 256] = zb[i - 1 + j * 16 + bl * 256];
			}
		}
		if (blockyo[bl] == yo)//safe? in adaptive this should be xo-x+0.5dx*(2^lev-1) <= tiny
		{
			int j = 0;
			for (int i = 0; i < 16; i++)
			{

				zb[i + j * 16 + bl * 256] = zb[i  + (j+1) * 16 + bl * 256];
			}
		}
		if (blockyo[bl] + (15 * dx) == yo + (ceil(ny / 16.0)*16.0 - 1)*dx)//safe? in adaptive this should be xo-x+0.5dx*(2^lev-1) <= tiny
		{
			int j = 15;
			for (int i = 0; i < 16; i++)
			{

				zb[i + j * 16 + bl * 256] = zb[i + (j-1) * 16 + bl * 256];
			}
		}
		*/
		if (bl == leftblk[bl])//i.e. if a block refers to as it's onwn neighbour then it doesn't have a neighbour/// This also applies to block that are on the edge of the grid so the above is commentted
		{
			int i = 0;
			for (int j = 0; j < 16; j++)
			{

				zb[i + j * 16 + bl * 256] = zb[i + 1 + j * 16 + bl * 256];
			}
		}
		if (bl == rightblk[bl])
		{
			int i = 15;
			for (int j = 0; j < 16; j++)
			{

				zb[i + j * 16 + bl * 256] = zb[i - 1 + j * 16 + bl * 256];
			}
		}
		if (bl == topblk[bl])
		{
			int j = 15;
			for (int i = 0; i < 16; i++)
			{

				zb[i + j * 16 + bl * 256] = zb[i + (j - 1) * 16 + bl * 256];
			}
		}
		if (bl == botblk[bl])
		{
			int j = 0;
			for (int i = 0; i < 16; i++)
			{

				zb[i + j * 16 + bl * 256] = zb[i + (j + 1) * 16 + bl * 256];
			}
		}

	}
}

template <class T> void carttoBUQ(int nblk, int nx,int ny, double xo,double yo, double dx, double* blockxo, double* blockyo,  T * zb, T *&zb_buq)
{
	//
	int ix, iy;
	T x, y;
	for (int b = 0; b < nblk; b++)
	{

		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 16; j++)
			{
				x = blockxo[b] + i*dx;
				y = blockyo[b] + j*dx;
				ix = min(max((int)round((x-xo) / dx),0),nx-1); // min(max( part is overkill?
				iy = min(max((int)round((y-yo) / dx), 0), ny - 1);
				
				zb_buq[i + j * 16 + b * 256] = zb[ix + iy*nx];
				//printf("bid=%i\ti=%i\tj=%i\tix=%i\tiy=%i\tzb_buq[n]=%f\n", b,i,j,ix, iy, zb_buq[i + j * 16 + b * 256]);
			}
		}
	}
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


void checkGradGPU(Param XParam)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	gradientGPUXYBUQ << <gridDim, blockDim, 0 >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hh_g, dhdx_g, dhdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradient(XParam.nblk, XParam.blksize, (float)XParam.theta, (float)XParam.delta, leftblk, rightblk, topblk, botblk, hh, dhdx, dhdy);

	CUDA_CHECK(cudaMemcpy(dummy, dhdx_g, XParam.nblk*XParam.blksize * sizeof(float), cudaMemcpyDeviceToHost));

	float mdiff = maxdiff(XParam.nblk*XParam.blksize, dhdx, dummy);
	float maxerr = 1e-11f;//1e-7f
	if (mdiff > maxerr)
	{
		printf("High error in dhdx: %f\n", mdiff);
	}
}

int AllocMemCPU(Param XParam)
{
	//function to allocate the memory on the CPU
	// Pointers are Global !
	//Need to add a sucess check for each call to malloc

	int nblk = XParam.nblk;
	int blksize = XParam.blksize;


	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{
		//allocate double *arrays
		Allocate1CPU(nblk, blksize, zb_d);
		Allocate4CPU(nblk, blksize, zs_d, hh_d, uu_d, vv_d);
		Allocate4CPU(nblk, blksize, zso_d, hho_d, uuo_d, vvo_d);
		Allocate4CPU(nblk, blksize, dzsdx_d, dhdx_d, dudx_d, dvdx_d);
		Allocate4CPU(nblk, blksize, dzsdy_d, dhdy_d, dudy_d, dvdy_d);

		Allocate4CPU(nblk, blksize, Su_d, Sv_d, Fhu_d, Fhv_d);
		Allocate4CPU(nblk, blksize, Fqux_d, Fquy_d, Fqvx_d, Fqvy_d);

		//Allocate4CPU(nblk, blksize, dh_d, dhu_d, dhv_d, dummy_d);
		Allocate1CPU(nblk, blksize, dh_d);
		Allocate1CPU(nblk, blksize, dhu_d);
		Allocate1CPU(nblk, blksize, dhv_d);

		Allocate1CPU(nblk, blksize, cf_d);


		//not allocating below may be usefull

		if (XParam.outhhmax == 1)
		{
			Allocate1CPU(nblk, blksize, hhmax_d);
		}
		if (XParam.outuumax == 1)
		{
			Allocate1CPU(nblk, blksize, uumax_d);
		}
		if (XParam.outvvmax == 1)
		{
			Allocate1CPU(nblk, blksize, vvmax_d);
		}
		if (XParam.outzsmax == 1)
		{
			Allocate1CPU(nblk, blksize, zsmax_d);
		}

		if (XParam.outhhmean == 1)
		{
			Allocate1CPU(nblk, blksize, hhmean_d);
		}
		if (XParam.outzsmean == 1)
		{
			Allocate1CPU(nblk, blksize, zsmean_d);
		}
		if (XParam.outuumean == 1)
		{
			Allocate1CPU(nblk, blksize, uumean_d);
		}
		if (XParam.outvvmean == 1)
		{
			Allocate1CPU(nblk, blksize, vvmean_d);
		}

		if (XParam.outvort == 1)
		{
			Allocate1CPU(nblk, blksize, vort);
		}

	}
	else
	{
		// allocate float *arrays (same template functions but different pointers)
		Allocate1CPU(nblk, blksize, zb);
		Allocate4CPU(nblk, blksize, zs, hh, uu, vv);
		Allocate4CPU(nblk, blksize, zso, hho, uuo, vvo);
		Allocate4CPU(nblk, blksize, dzsdx, dhdx, dudx, dvdx);
		Allocate4CPU(nblk, blksize, dzsdy, dhdy, dudy, dvdy);

		Allocate4CPU(nblk, blksize, Su, Sv, Fhu, Fhv);
		Allocate4CPU(nblk, blksize, Fqux, Fquy, Fqvx, Fqvy);

		//Allocate4CPU(nx, ny, dh, dhu, dhv, dummy);
		Allocate1CPU(nblk, blksize, dh);
		Allocate1CPU(nblk, blksize, dhu);
		Allocate1CPU(nblk, blksize, dhv);
		Allocate1CPU(nblk, blksize, cf);
		//not allocating below may be usefull

		if (XParam.outhhmax == 1)
		{
			Allocate1CPU(nblk, blksize, hhmax);
		}
		if (XParam.outuumax == 1)
		{
			Allocate1CPU(nblk, blksize, uumax);
		}
		if (XParam.outvvmax == 1)
		{
			Allocate1CPU(nblk, blksize, vvmax);
		}
		if (XParam.outzsmax == 1)
		{
			Allocate1CPU(nblk, blksize, zsmax);
		}

		if (XParam.outhhmean == 1)
		{
			Allocate1CPU(nblk, blksize, hhmean);
		}
		if (XParam.outzsmean == 1)
		{
			Allocate1CPU(nblk, blksize, zsmean);
		}
		if (XParam.outuumean == 1)
		{
			Allocate1CPU(nblk, blksize, uumean);
		}
		if (XParam.outvvmean == 1)
		{
			Allocate1CPU(nblk, blksize, vvmean);
		}

		if (XParam.outvort == 1)
		{
			Allocate1CPU(nblk, blksize, vort);
		}

	}
	return 1; //Need a real test here
}

int AllocMemGPU(Param XParam)
{
	//function to allocate the memory on the GPU
	// Also prepare textures
	// Pointers are Global !
	//Need to add a sucess check for each call to malloc

	int nblk = XParam.nblk;
	int blksize = XParam.blksize;
	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{
		Allocate1GPU(nblk, blksize, zb_gd);
		Allocate4GPU(nblk, blksize, zs_gd, hh_gd, uu_gd, vv_gd);
		Allocate4GPU(nblk, blksize, zso_gd, hho_gd, uuo_gd, vvo_gd);
		Allocate4GPU(nblk, blksize, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd);
		Allocate4GPU(nblk, blksize, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd);

		Allocate4GPU(nblk, blksize, Su_gd, Sv_gd, Fhu_gd, Fhv_gd);
		Allocate4GPU(nblk, blksize, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd);

		Allocate4GPU(nblk, blksize, dh_gd, dhu_gd, dhv_gd, dtmax_gd);
		Allocate1GPU(nblk, blksize, cf_gd);
		Allocate1GPU(nblk, 1, blockxo_gd);
		Allocate1GPU(nblk, 1, blockyo_gd);



		arrmin_d = (double *)malloc(nblk* blksize * sizeof(double));
		CUDA_CHECK(cudaMalloc((void **)&arrmin_gd, nblk* blksize * sizeof(double)));
		CUDA_CHECK(cudaMalloc((void **)&arrmax_gd, nblk* blksize * sizeof(double)));

		if (XParam.outhhmax == 1)
		{
			Allocate1GPU(nblk, blksize, hhmax_gd);
		}
		if (XParam.outzsmax == 1)
		{
			Allocate1GPU(nblk, blksize, zsmax_gd);
		}
		if (XParam.outuumax == 1)
		{
			Allocate1GPU(nblk, blksize, uumax_gd);
		}
		if (XParam.outvvmax == 1)
		{
			Allocate1GPU(nblk, blksize, vvmax_gd);
		}
		if (XParam.outhhmean == 1)
		{
			Allocate1GPU(nblk, blksize, hhmean_gd);
		}
		if (XParam.outzsmean == 1)
		{
			Allocate1GPU(nblk, blksize, zsmean_gd);
		}
		if (XParam.outuumean == 1)
		{
			Allocate1GPU(nblk, blksize, uumean_gd);
		}
		if (XParam.outvvmean == 1)
		{
			Allocate1GPU(nblk, blksize, vvmean_gd);
		}

		if (XParam.outvort == 1)
		{
			Allocate1GPU(nblk, blksize, vort_gd);
		}

		if (XParam.TSnodesout.size() > 0)
		{
			// Allocate mmemory to store TSoutput in between writing to disk
			int nTS = 1; // Nb of points
			int nvts = 1; // NB of variables hh, zs, uu, vv
			int nstore = 2048; //store up to 2048 pts
			TSstore_d = (double *)malloc(nTS*nvts*nstore * sizeof(double));
			CUDA_CHECK(cudaMalloc((void **)&TSstore_gd, nTS*nvts*nstore * sizeof(double)));
			//Cpu part done differently because there are no latency issue (i.e. none that I care about) 

		}
	}
	else
	{
		Allocate1GPU(nblk, blksize, zb_g);
		Allocate4GPU(nblk, blksize, zs_g, hh_g, uu_g, vv_g);
		Allocate4GPU(nblk, blksize, zso_g, hho_g, uuo_g, vvo_g);
		Allocate4GPU(nblk, blksize, dzsdx_g, dhdx_g, dudx_g, dvdx_g);
		Allocate4GPU(nblk, blksize, dzsdy_g, dhdy_g, dudy_g, dvdy_g);

		Allocate4GPU(nblk, blksize, Su_g, Sv_g, Fhu_g, Fhv_g);
		Allocate4GPU(nblk, blksize, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g);

		Allocate4GPU(nblk, blksize, dh_g, dhu_g, dhv_g, dtmax_g);
		Allocate1GPU(nblk, blksize, cf_g);

		Allocate1GPU(nblk, 1, blockxo_g);
		Allocate1GPU(nblk, 1, blockyo_g);

		arrmin = (float *)malloc(nblk*blksize * sizeof(float));
		CUDA_CHECK(cudaMalloc((void **)&arrmin_g, nblk*blksize * sizeof(float)));
		CUDA_CHECK(cudaMalloc((void **)&arrmax_g, nblk*blksize * sizeof(float)));

		if (XParam.outhhmax == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&hhmax_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outzsmax == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&zsmax_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outuumax == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&uumax_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outvvmax == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&vvmax_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outhhmean == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&hhmean_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outzsmean == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&zsmean_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outuumean == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&uumean_g, nblk*blksize * sizeof(float)));
		}
		if (XParam.outvvmean == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&vvmean_g, nblk*blksize * sizeof(float)));
		}

		if (XParam.outvort == 1)
		{
			CUDA_CHECK(cudaMalloc((void **)&vort_g, nblk*blksize * sizeof(float)));
		}


		if (XParam.TSnodesout.size() > 0)
		{
			// Allocate mmemory to store TSoutput in between writing to disk
			int nTS = 1; // Nb of points
			int nvts = 1; // NB of variables hh, zs, uu, vv
			int nstore = 2048; //store up to 2048 pts
			TSstore = (float *)malloc(nTS*nvts*nstore * sizeof(float));
			CUDA_CHECK(cudaMalloc((void **)&TSstore_g, nTS*nvts*nstore * sizeof(float)));
			//Cpu part done differently because there are no latency issue (i.e. none that I care about) 

		}
	}


	Allocate4GPU(nblk, 1, leftblk_g, rightblk_g, topblk_g, botblk_g);

	return 1;
}

int AllocMemGPUBND(Param XParam, std::vector<SLTS> leftWLbnd, std::vector<SLTS> rightWLbnd, std::vector<SLTS> topWLbnd, std::vector<SLTS> botWLbnd)
{
	// Allocate textures and bind arrays for boundary interpolation
	if (!XParam.leftbndfile.empty())
	{
		//leftWLbnd = readWLfile(XParam.leftbndfile);
		//Flatten bnd to copy to cuda array
		int nbndtimes = (int)leftWLbnd.size();
		int nbndvec = (int)leftWLbnd[0].wlevs.size();
		CUDA_CHECK(cudaMallocArray(&leftWLS_gp, &channelDescleftbnd, nbndtimes, nbndvec));
		// This below was float by default and probably should remain float as long as fetched floats are readily converted to double as needed
		float * leftWLS;
		leftWLS = (float *)malloc(nbndtimes * nbndvec * sizeof(float));

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
		int nbndtimes = (int)rightWLbnd.size();
		int nbndvec = (int)rightWLbnd[0].wlevs.size();
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
		int nbndtimes = (int)topWLbnd.size();
		int nbndvec = (int)topWLbnd[0].wlevs.size();
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
		int nbndtimes = (int)botWLbnd.size();
		int nbndvec = (int)botWLbnd[0].wlevs.size();
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
	return 1;
}


void LeftFlowBnd(Param XParam, std::vector<SLTS> leftWLbnd)
{
	//
	int nx = XParam.nx;
	int ny = XParam.ny;
	if (XParam.left > 1 && !leftWLbnd.empty())
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

		

		dim3 blockDim(16, 16, 1);
		dim3 gridDim(XParam.nblk, 1, 1);
		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			double itime = SLstepinbnd - 1.0 + (XParam.totaltime - leftWLbnd[SLstepinbnd - 1].time) / (leftWLbnd[SLstepinbnd].time - leftWLbnd[SLstepinbnd - 1].time);
			if (XParam.left==2 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				leftdirichletD << <gridDim, blockDim, 0 >> > ( (int)leftWLbnd[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax,itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.left == 2)
			{
				leftdirichlet << <gridDim, blockDim, 0 >> > ( (int)leftWLbnd[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.ymax, (float)itime, rightblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}

			if (XParam.left == 3 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)leftWLbnd[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1D << <gridDim, blockDim, 0 >> > (-1, 0, (int)leftWLbnd[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.left == 3)
			{
				ABS1D << <gridDim, blockDim, 0 >> > (-1, 0, (int)leftWLbnd[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, rightblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}




			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			std::vector<double> zsbndleft;
			for (int n = 0; n < leftWLbnd[SLstepinbnd].wlevs.size(); n++)
			{
				zsbndleft.push_back( interptime(leftWLbnd[SLstepinbnd].wlevs[n], leftWLbnd[SLstepinbnd - 1].wlevs[n], leftWLbnd[SLstepinbnd].time - leftWLbnd[SLstepinbnd - 1].time, XParam.totaltime - leftWLbnd[SLstepinbnd - 1].time));

			}
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{

				leftdirichletCPUD(XParam.nblk, XParam.blksize, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndleft, blockxo_d, blockyo_d, zs_d, zb_d, hh_d, uu_d, vv_d);
			}
			else
			{
				//void leftdirichletCPU(int nblk, int blksize, float xo,float yo, float g, float dx, std::vector<double> zsbndvec, float * blockxo,float * blockyo, float *zs, float *zb, float *hh, float *uu, float *vv)
				//leftdirichletCPU(nx, ny, (float)XParam.g, zsbndleft, zs, zb, hh, uu, vv);
				leftdirichletCPU(XParam.nblk, XParam.blksize, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndleft, blockxo, blockyo, zs, zb, hh, uu, vv);
			}
			
		}
	}
	if (XParam.left == 0)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			//
			dim3 blockDim(16, 16, 1);
			dim3 gridDim(XParam.nblk, 1, 1);
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				noslipbndLeft << <gridDim, blockDim, 0 >> > (XParam.xo, XParam.eps, rightblk_g, blockxo_gd, zb_gd, zs_gd, hh_gd, uu_gd, vv_gd);
			}
			else
			{
				noslipbndLeft << <gridDim, blockDim, 0 >> > ((float)XParam.xo, (float)XParam.eps, rightblk_g, blockxo_g, zb_g, zs_g, hh_g, uu_g, vv_g);
			}
			
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			noslipbndLCPU(XParam);
		}
	}
	//else neumann bnd (is already built in the solver)
}

void RightFlowBnd(Param XParam, std::vector<SLTS> rightWLbnd)
{
	//
	int nx = XParam.nx;
	int ny = XParam.ny;
	if (XParam.right > 1 && !rightWLbnd.empty())
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

		

		dim3 blockDim(16, 16, 1);
		dim3 gridDim(XParam.nblk, 1, 1);
		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			double itime = SLstepinbnd - 1.0 + (XParam.totaltime - rightWLbnd[SLstepinbnd - 1].time) / (rightWLbnd[SLstepinbnd].time - rightWLbnd[SLstepinbnd - 1].time);
			if (XParam.right == 2 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				rightdirichletD << <gridDim, blockDim, 0 >> > ( (int)rightWLbnd[0].wlevs.size(), XParam.g, XParam.dx, XParam.xmax, XParam.ymax, itime, leftblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.right == 2)
			{
				rightdirichlet << <gridDim, blockDim, 0 >> > ( (int)rightWLbnd[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xmax, (float)XParam.ymax, (float)itime, leftblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}
			else if (XParam.right == 3 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)leftWLbnd[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1D << <gridDim, blockDim, 0 >> > (1, 0, (int)rightWLbnd[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, leftblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.right == 3)
			{
				ABS1D << <gridDim, blockDim, 0 >> > (1, 0, (int)rightWLbnd[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, leftblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			std::vector<double> zsbndright;
			for (int n = 0; n < rightWLbnd[SLstepinbnd].wlevs.size(); n++)
			{
				zsbndright.push_back( interptime(rightWLbnd[SLstepinbnd].wlevs[n], rightWLbnd[SLstepinbnd - 1].wlevs[n], rightWLbnd[SLstepinbnd].time - rightWLbnd[SLstepinbnd - 1].time, XParam.totaltime - rightWLbnd[SLstepinbnd - 1].time));

			}
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				rightdirichletCPUD(XParam.nblk, XParam.blksize, XParam.nx, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndright, blockxo_d, blockyo_d, zs_d, zb_d, hh_d, uu_d, vv_d);
			}
			else
			{
				//rightdirichletCPU(nx, ny, (float)XParam.g, zsbndright, zs, zb, hh, uu, vv);
				rightdirichletCPU(XParam.nblk, XParam.blksize, XParam.nx, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndright, blockxo, blockyo, zs, zb, hh, uu, vv);
			}
		}
	}
	if (XParam.right == 0)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			//
			dim3 blockDim(16, 16, 1);
			dim3 gridDim(XParam.nblk, 1, 1);
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				noslipbndRight << <gridDim, blockDim, 0 >> > (XParam.dx, XParam.xmax, XParam.eps, leftblk_g, blockxo_gd, zb_gd, zs_gd, hh_gd, uu_gd, vv_gd);
			}
			else
			{
				noslipbndRight << <gridDim, blockDim, 0 >> > ((float)XParam.dx, (float)XParam.xmax, (float)XParam.eps, leftblk_g, blockxo_g, zb_g, zs_g, hh_g, uu_g, vv_g);
			}
			
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			noslipbndRCPU(XParam);
		}
	}
	//else neumann bnd (is already built in the algorithm)
}

void TopFlowBnd(Param XParam, std::vector<SLTS> topWLbnd)
{
	//
	int nx = XParam.nx;
	int ny = XParam.ny;
	if (XParam.top > 1 && !topWLbnd.empty())
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


		dim3 blockDim(16, 16, 1);
		dim3 gridDim(XParam.nblk, 1, 1);
		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			double itime = SLstepinbnd - 1.0 + (XParam.totaltime - topWLbnd[SLstepinbnd - 1].time) / (topWLbnd[SLstepinbnd].time - topWLbnd[SLstepinbnd - 1].time);
			if (XParam.top == 2 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				topdirichletD << <gridDim, blockDim, 0 >> > ( (int)topWLbnd[0].wlevs.size(), XParam.g, XParam.dx, XParam.xmax, XParam.ymax, itime, botblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.top == 2)
			{
				topdirichlet << <gridDim, blockDim, 0 >> > ((int)topWLbnd[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xmax, (float)XParam.ymax, (float)itime, botblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}
			else if (XParam.top == 3 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)leftWLbnd[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1D << <gridDim, blockDim, 0 >> > (0, 1, (int)topWLbnd[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, botblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.top == 3)
			{
				ABS1D << <gridDim, blockDim, 0 >> > (0, 1, (int)topWLbnd[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, botblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, vv_g, uu_g);
			}
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			std::vector<double> zsbndtop;
			for (int n = 0; n < topWLbnd[SLstepinbnd].wlevs.size(); n++)
			{
				zsbndtop.push_back( interptime(topWLbnd[SLstepinbnd].wlevs[n], topWLbnd[SLstepinbnd - 1].wlevs[n], topWLbnd[SLstepinbnd].time - topWLbnd[SLstepinbnd - 1].time, XParam.totaltime - topWLbnd[SLstepinbnd - 1].time));

			}
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				topdirichletCPUD(XParam.nblk, XParam.blksize, XParam.ny, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndtop, blockxo_d, blockyo_d, zs_d, zb_d, hh_d, uu_d, vv_d);
			}
			else
			{

				//topdirichletCPU(nx, ny, (float)XParam.g, zsbndtop, zs, zb, hh, uu, vv);
				topdirichletCPU(XParam.nblk, XParam.blksize, XParam.ny, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndtop, blockxo, blockyo, zs, zb, hh, uu, vv);
			}
		}
	}
	if (XParam.top == 0)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			//
			dim3 blockDim(16, 16, 1);
			dim3 gridDim(XParam.nblk, 1, 1);
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				noslipbndTop << <gridDim, blockDim, 0 >> > (XParam.dx, XParam.ymax, XParam.eps, botblk_g, blockyo_gd, zb_gd, zs_gd, hh_gd, uu_gd, vv_gd);
			}
			else
			{
				noslipbndTop << <gridDim, blockDim, 0 >> > ((float)XParam.dx, (float)XParam.ymax, (float)XParam.eps, botblk_g, blockyo_g, zb_g, zs_g, hh_g, uu_g, vv_g);
			}
			
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			noslipbndTCPU(XParam);
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

		

		dim3 blockDim(16, 16, 1);
		dim3 gridDim(XParam.nblk, 1, 1);
		if (XParam.GPUDEVICE >= 0)
		{
			//leftdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
			double itime = SLstepinbnd - 1.0 + (XParam.totaltime - botWLbnd[SLstepinbnd - 1].time) / (botWLbnd[SLstepinbnd].time - botWLbnd[SLstepinbnd - 1].time);
			if (XParam.bot == 2 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				botdirichletD << <gridDim, blockDim, 0 >> > ( (int)botWLbnd[0].wlevs.size(), XParam.g, XParam.dx, XParam.xmax, XParam.yo, itime, topblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.bot == 2)
			{
				botdirichlet << <gridDim, blockDim, 0 >> > ( (int)botWLbnd[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xmax, (float)XParam.yo, (float)itime, topblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, uu_g, vv_g);
			}
			else if (XParam.bot == 3 && (XParam.doubleprecision == 1 || XParam.spherical == 1))
			{
				//leftdirichletD << <gridDim, blockDim, 0 >> > ((int)leftWLbnd[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.ymax, itime, rightblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
				ABS1D << <gridDim, blockDim, 0 >> > (0, -1, (int)botWLbnd[0].wlevs.size(), XParam.g, XParam.dx, XParam.xo, XParam.yo, XParam.xmax, XParam.ymax, itime, topblk_g, blockxo_gd, blockyo_gd, zs_gd, zb_gd, hh_gd, uu_gd, vv_gd);
			}
			else if (XParam.bot == 3)
			{
				ABS1D << <gridDim, blockDim, 0 >> > (0, -1, (int)botWLbnd[0].wlevs.size(), (float)XParam.g, (float)XParam.dx, (float)XParam.xo, (float)XParam.yo, (float)XParam.xmax, (float)XParam.ymax, (float)itime, topblk_g, blockxo_g, blockyo_g, zs_g, zb_g, hh_g, vv_g, uu_g);
			}
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			std::vector<double> zsbndbot;
			for (int n = 0; n < botWLbnd[SLstepinbnd].wlevs.size(); n++)
			{
				zsbndbot.push_back( interptime(botWLbnd[SLstepinbnd].wlevs[n], botWLbnd[SLstepinbnd - 1].wlevs[n], botWLbnd[SLstepinbnd].time - botWLbnd[SLstepinbnd - 1].time, XParam.totaltime - botWLbnd[SLstepinbnd - 1].time));

			}
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				botdirichletCPUD(XParam.nblk, XParam.blksize, XParam.ny, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndbot, blockxo_d, blockyo_d, zs_d, zb_d, hh_d, uu_d, vv_d);
			}
			else
			{
				//botdirichletCPU(nx, ny, (float)XParam.g, zsbndbot, zs, zb, hh, uu, vv);
				botdirichletCPU(XParam.nblk, XParam.blksize, XParam.ny, XParam.xo, XParam.yo, XParam.g, XParam.dx, zsbndbot, blockxo, blockyo, zs, zb, hh, uu, vv);
			}
		}
	}
	if (XParam.bot == 0)
	{
		if (XParam.GPUDEVICE >= 0)
		{
			//
			dim3 blockDim(16, 16, 1);
			dim3 gridDim(XParam.nblk, 1, 1);
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				noslipbndBot << <gridDim, blockDim, 0 >> > (XParam.yo,  XParam.eps, topblk_g, blockyo_gd, zb_gd, zs_gd, hh_gd, uu_gd, vv_gd);
			}
			else
			{
				noslipbndBot << <gridDim, blockDim, 0 >> > ((float)XParam.yo, (float)XParam.eps, topblk_g, blockyo_g, zb_g, zs_g, hh_g, uu_g, vv_g);
			}
			
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		else
		{
			noslipbndBCPU(XParam);
		}
	}
	//else neumann bnd (is already built in the algorithm)
}

double FlowGPU(Param XParam, double nextoutputtime)
{
	int nx = XParam.nx;
	int ny = XParam.ny;

	const int num_streams = 2;

	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++) 
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}



	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	
	dtmax = (float) (1.0 / epsilon);
	//float dtmaxtmp = dtmax;

	resetdtmax << <gridDim, blockDim,0, streams[0] >> > (dtmax_g);
	//CUDA_CHECK(cudaDeviceSynchronize());
	//update step 1

	

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >( (float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hh_g, dhdx_g, dhdy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());

	

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zs_g, dzsdx_g, dzsdy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uu_g, dudx_g, dudy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());

	

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >( (float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vv_g, dvdx_g, dvdy_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//CUDA_CHECK(cudaStreamSynchronize(streams[0]));
	//normal cartesian case
	updateKurgX << <gridDim, blockDim, 0, streams[0] >> > ( (float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, leftblk_g, hh_g, zs_g, uu_g, vv_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//CUDA_CHECK(cudaStreamSynchronize(streams[1]));
	updateKurgY << <gridDim, blockDim, 0, streams[1] >> > ( (float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, botblk_g, hh_g, zs_g, uu_g, vv_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);

	CUDA_CHECK(cudaDeviceSynchronize());
	


	//GPU Harris reduction #3. 8.3x reduction #0  Note #7 if a lot faster
	// This was successfully tested with a range of grid size
	//reducemax3 << <gridDimLine, blockDimLine, 64*sizeof(float) >> >(dtmax_g, arrmax_g, nx*ny)
	int s = XParam.nblk*XParam.blksize;
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

	
	updateEV << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, rightblk_g, topblk_g, hh_g, uu_g, vv_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	


	//predictor (advance 1/2 dt)
	Advkernel << <gridDim, blockDim, 0 >> >((float)XParam.dt*0.5f, (float)XParam.eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//corrector setp
	//update again
	

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hho_g, dhdx_g, dhdy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());

	

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >( (float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zso_g, dzsdx_g, dzsdy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());

	

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >( (float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uuo_g, dudx_g, dudy_g);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >( (float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vvo_g, dvdx_g, dvdy_g);
	
	CUDA_CHECK(cudaDeviceSynchronize());


	
	updateKurgX << <gridDim, blockDim, 0, streams[0] >> > ( (float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, leftblk_g, hho_g, zso_g, uuo_g, vvo_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
	//CUDA_CHECK(cudaDeviceSynchronize());


	updateKurgY << <gridDim, blockDim, 0, streams[1] >> > ( (float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, botblk_g, hho_g, zso_g, uuo_g, vvo_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	// no reduction of dtmax during the corrector step

	
	updateEV << <gridDim, blockDim, 0 >> > ( (float)XParam.delta, (float)XParam.g, rightblk_g, topblk_g, hho_g, uuo_g, vvo_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
	CUDA_CHECK(cudaDeviceSynchronize());
	
	

	//
	Advkernel << <gridDim, blockDim, 0 >> >( (float)XParam.dt, (float)XParam.eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
	cleanupGPU << <gridDim, blockDim, 0 >> >( hho_g, zso_g, uuo_g, vvo_g, hh_g, zs_g, uu_g, vv_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	//Bottom friction
	quadfriction << <gridDim, blockDim, 0 >> > ( (float)XParam.dt, (float)XParam.eps, cf_g, hh_g, uu_g, vv_g);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaStreamDestroy(streams[0]));
	CUDA_CHECK(cudaStreamDestroy(streams[1]));

	// Impose no slip condition by default
	//noslipbndall << <gridDim, blockDim, 0 >> > (nx, ny, XParam.dt, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
	//CUDA_CHECK(cudaDeviceSynchronize());

	if (XParam.Rivers.size() > 1)
	{
		//
		float qnow;
		for (int Rin = 0; Rin < XParam.Rivers.size(); Rin++)
		{

			//qnow = interptime(slbnd[SLstepinbnd].wlev0, slbnd[SLstepinbnd - 1].wlev0, slbnd[SLstepinbnd].time - slbnd[SLstepinbnd - 1].time, totaltime - slbnd[SLstepinbnd - 1].time);
			int bndstep = 0;
			double difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
			while (difft <= 0.0) // danger?
			{
				bndstep++;
				difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
			}

			qnow = interptime(XParam.Rivers[Rin].flowinput[bndstep].q, XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].q, XParam.Rivers[Rin].flowinput[bndstep].time - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time, XParam.totaltime - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time);



			discharge_bnd_v << <gridDim, blockDim, 0 >> > ((float)XParam.Rivers[Rin].xstart, (float)XParam.Rivers[Rin].xend, (float)XParam.Rivers[Rin].ystart, (float)XParam.Rivers[Rin].yend, (float)XParam.dx, (float)XParam.dt, qnow, (float)XParam.Rivers[Rin].disarea, blockxo_g, blockyo_g, zs_g, hh_g);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	}



	return XParam.dt;
}

double FlowGPUSpherical(Param XParam, double nextoutputtime)
{
	

	const int num_streams = 2;

	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}



	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0) / blockDim.x), ceil((ny*1.0) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	dtmax = (float)(1.0 / epsilon);
	//float dtmaxtmp = dtmax;

	resetdtmax << <gridDim, blockDim, 0, streams[0] >> > ( dtmax_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());
	//update step 1

	


	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hh_gd, dhdx_gd, dhdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zs_gd, dzsdx_gd, dzsdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());


	

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uu_gd, dudx_gd, dudy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vv_gd, dvdx_gd, dvdy_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//CUDA_CHECK(cudaStreamSynchronize(streams[0]));
	//Spherical
	{
		//Spherical coordinates 
		updateKurgXSPH << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, leftblk_g, blockyo_gd, XParam.Radius, hh_gd, zs_gd, uu_gd, vv_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);

		updateKurgYSPH << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, botblk_g, blockyo_gd, XParam.Radius, hh_gd, zs_gd, uu_gd, vv_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);

		CUDA_CHECK(cudaDeviceSynchronize());

	}

	/////////////////////////////////////////////////////
	// Reduction of dtmax
	/////////////////////////////////////////////////////

	// copy from GPU and do the reduction on the CPU  ///LAME!
	

	//GPU Harris reduction #3. 8.3x reduction #0  Note #7 if a lot faster
	// This was successfully tested with a range of grid size
	//reducemax3 << <gridDimLine, blockDimLine, 64*sizeof(float) >> >(dtmax_g, arrmax_g, nx*ny)
	int s = XParam.nblk*XParam.blksize;
	int maxThreads = 256;
	int threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
	int blocks = (s + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);
	dim3 blockDimLine(threads, 1, 1);
	dim3 gridDimLine(blocks, 1, 1);

	double mindtmaxB;

	reducemin3 << <gridDimLine, blockDimLine, smemSize >> > (dtmax_gd, arrmax_gd, s);
	CUDA_CHECK(cudaDeviceSynchronize());



	s = gridDimLine.x;
	while (s > 1)//cpuFinalThreshold
	{
		threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);

		smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

		dim3 blockDimLineS(threads, 1, 1);
		dim3 gridDimLineS(blocks, 1, 1);

		CUDA_CHECK(cudaMemcpy(dtmax_gd, arrmax_gd, s * sizeof(double), cudaMemcpyDeviceToDevice));

		reducemin3 << <gridDimLineS, blockDimLineS, smemSize >> > (dtmax_gd, arrmax_gd, s);
		CUDA_CHECK(cudaDeviceSynchronize());

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}


	CUDA_CHECK(cudaMemcpy(dummy_d, arrmax_gd, 32 * sizeof(double), cudaMemcpyDeviceToHost));
	mindtmaxB = dummy_d[0];
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

	//spherical
	{
		//if spherical corrdinate use this kernel with the right corrections
		updateEVSPH << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.yo, XParam.ymax, XParam.Radius, rightblk_g, topblk_g, blockyo_gd, hh_gd, uu_gd, vv_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}


	//predictor (advance 1/2 dt)
	Advkernel << <gridDim, blockDim, 0 >> >( XParam.dt*0.5, XParam.eps, hh_gd, zb_gd, uu_gd, vv_gd, dh_gd, dhu_gd, dhv_gd, zso_gd, hho_gd, uuo_gd, vvo_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//corrector setp
	//update again
	// calculate gradients
	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hho_g, dhdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hho_g, dhdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >( XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hho_gd, dhdx_gd, dhdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, zso_g, dzsdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, zso_g, dzsdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zso_gd, dzsdx_gd, dzsdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, uuo_g, dudx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, uuo_g, dudy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uuo_gd, dudx_gd, dudy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, vvo_g, dvdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, vvo_g, dvdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vvo_gd, dvdx_gd, dvdy_gd);

	CUDA_CHECK(cudaDeviceSynchronize());


	// Test whether it is better to have one here or later (are the instuctions overlap if occupancy and meme acess is available?)
	//CUDA_CHECK(cudaDeviceSynchronize());

	
	{
		//Spherical coordinates 
		updateKurgXSPH << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, leftblk_g, blockyo_gd, XParam.Radius, hho_gd, zso_gd, uuo_gd, vvo_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);

		updateKurgYSPH << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, botblk_g, blockyo_gd, XParam.Radius, hho_gd, zso_gd, uuo_gd, vvo_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);

		CUDA_CHECK(cudaDeviceSynchronize());

	}
	// no reduction of dtmax during the corrector step

	
	{
		//if spherical corrdinate use this kernel with the right corrections
		updateEVSPH << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.yo, XParam.ymax, XParam.Radius, rightblk_g, topblk_g, blockyo_gd, hho_gd, uuo_gd, vvo_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	//
	Advkernel << <gridDim, blockDim, 0 >> >( XParam.dt, XParam.eps, hh_gd, zb_gd, uu_gd, vv_gd, dh_gd, dhu_gd, dhv_gd, zso_gd, hho_gd, uuo_gd, vvo_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
	cleanupGPU << <gridDim, blockDim, 0 >> >( hho_gd, zso_gd, uuo_gd, vvo_gd, hh_gd, zs_gd, uu_gd, vv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//Bottom friction
	quadfriction << <gridDim, blockDim, 0 >> > ( XParam.dt, XParam.eps, cf_gd, hh_gd, uu_gd, vv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaStreamDestroy(streams[0]));
	CUDA_CHECK(cudaStreamDestroy(streams[1]));

	// Impose no slip condition by default
	//noslipbndall << <gridDim, blockDim, 0 >> > (nx, ny, XParam.dt, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
	//CUDA_CHECK(cudaDeviceSynchronize());
	return XParam.dt;
}


double FlowGPUDouble(Param XParam, double nextoutputtime)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	const int num_streams = 2;

	cudaStream_t streams[num_streams];
	for (int i = 0; i < num_streams; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
	}

	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);


	dtmax = (float)(1.0 / epsilon);
	//float dtmaxtmp = dtmax;

	resetdtmax << <gridDim, blockDim, 0, streams[0] >> > ( dtmax_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());
	//update step 1




	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >( XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hh_gd, dhdx_gd, dhdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >( XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zs_gd, dzsdx_gd, dzsdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());




	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >( XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uu_gd, dudx_gd, dudy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());



	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >( XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vv_gd, dvdx_gd, dvdy_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//CUDA_CHECK(cudaStreamSynchronize(streams[0]));
	
	
		
	updateKurgXD << <gridDim, blockDim, 0, streams[0] >> > ( XParam.delta, XParam.g, XParam.eps, XParam.CFL, leftblk_g, hh_gd, zs_gd, uu_gd, vv_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);

	updateKurgYD << <gridDim, blockDim, 0, streams[1] >> > ( XParam.delta, XParam.g, XParam.eps, XParam.CFL, botblk_g, hh_gd, zs_gd, uu_gd, vv_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);

	CUDA_CHECK(cudaDeviceSynchronize());

	

	/////////////////////////////////////////////////////
	// Reduction of dtmax
	/////////////////////////////////////////////////////

	// copy from GPU and do the reduction on the CPU  ///LAME!


	//GPU Harris reduction #3. 8.3x reduction #0  Note #7 if a lot faster
	// This was successfully tested with a range of grid size
	//reducemax3 << <gridDimLine, blockDimLine, 64*sizeof(float) >> >(dtmax_g, arrmax_g, nx*ny)
	int s = XParam.nblk*XParam.blksize;
	int maxThreads = 256;
	int threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
	int blocks = (s + (threads * 2 - 1)) / (threads * 2);
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);
	dim3 blockDimLine(threads, 1, 1);
	dim3 gridDimLine(blocks, 1, 1);

	double mindtmaxB;

	reducemin3 << <gridDimLine, blockDimLine, smemSize >> > (dtmax_gd, arrmax_gd, s);
	CUDA_CHECK(cudaDeviceSynchronize());



	s = gridDimLine.x;
	while (s > 1)//cpuFinalThreshold
	{
		threads = (s < maxThreads * 2) ? nextPow2((s + 1) / 2) : maxThreads;
		blocks = (s + (threads * 2 - 1)) / (threads * 2);

		smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

		dim3 blockDimLineS(threads, 1, 1);
		dim3 gridDimLineS(blocks, 1, 1);

		CUDA_CHECK(cudaMemcpy(dtmax_gd, arrmax_gd, s * sizeof(float), cudaMemcpyDeviceToDevice));

		reducemin3 << <gridDimLineS, blockDimLineS, smemSize >> > (dtmax_gd, arrmax_gd, s);
		CUDA_CHECK(cudaDeviceSynchronize());

		s = (s + (threads * 2 - 1)) / (threads * 2);
	}


	CUDA_CHECK(cudaMemcpy(dummy_d, arrmax_gd, 32 * sizeof(float), cudaMemcpyDeviceToHost));
	mindtmaxB = dummy_d[0];
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

	
		//if spherical corrdinate use this kernel with the right corrections
	updateEVD << <gridDim, blockDim, 0 >> > ( XParam.delta, XParam.g, rightblk_g, topblk_g, hh_gd, uu_gd, vv_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());
	


	//predictor (advance 1/2 dt)
	Advkernel << <gridDim, blockDim, 0 >> >( XParam.dt*0.5, XParam.eps, hh_gd, zb_gd, uu_gd, vv_gd, dh_gd, dhu_gd, dhv_gd, zso_gd, hho_gd, uuo_gd, vvo_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//corrector setp
	//update again
	// calculate gradients
	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hho_g, dhdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hho_g, dhdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >( XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hho_gd, dhdx_gd, dhdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, zso_g, dzsdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, zso_g, dzsdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >( XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zso_gd, dzsdx_gd, dzsdy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, uuo_g, dudx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, uuo_g, dudy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >( XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uuo_gd, dudx_gd, dudy_gd);
	//CUDA_CHECK(cudaDeviceSynchronize());

	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, vvo_g, dvdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, vvo_g, dvdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >( XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vvo_gd, dvdx_gd, dvdy_gd);

	CUDA_CHECK(cudaDeviceSynchronize());


	// Test whether it is better to have one here or later (are the instuctions overlap if occupancy and meme acess is available?)
	//CUDA_CHECK(cudaDeviceSynchronize());


	
	updateKurgXD << <gridDim, blockDim, 0, streams[0] >> > ( XParam.delta, XParam.g, XParam.eps, XParam.CFL, leftblk_g, hho_gd, zso_gd, uuo_gd, vvo_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);

	updateKurgYD << <gridDim, blockDim, 0, streams[1] >> > ( XParam.delta, XParam.g, XParam.eps, XParam.CFL, botblk_g, hho_gd, zso_gd, uuo_gd, vvo_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);

	CUDA_CHECK(cudaDeviceSynchronize());

	
	// no reduction of dtmax during the corrector step


	
	
	updateEVD << <gridDim, blockDim, 0 >> > ( XParam.delta, XParam.g, rightblk_g, topblk_g, hho_gd, uuo_gd, vvo_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());
	

	//
	Advkernel << <gridDim, blockDim, 0 >> >( XParam.dt, XParam.eps, hh_gd, zb_gd, uu_gd, vv_gd, dh_gd, dhu_gd, dhv_gd, zso_gd, hho_gd, uuo_gd, vvo_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
	cleanupGPU << <gridDim, blockDim, 0 >> >( hho_gd, zso_gd, uuo_gd, vvo_gd, hh_gd, zs_gd, uu_gd, vv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//Bottom friction
	quadfriction << <gridDim, blockDim, 0 >> > ( XParam.dt, XParam.eps, cf_gd, hh_gd, uu_gd, vv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaStreamDestroy(streams[0]));
	CUDA_CHECK(cudaStreamDestroy(streams[1]));

	// Impose no slip condition by default
	//noslipbndall << <gridDim, blockDim, 0 >> > (nx, ny, XParam.dt, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
	//CUDA_CHECK(cudaDeviceSynchronize());

	if (XParam.Rivers.size() > 1)
	{
		//
		double qnow;
		for (int Rin = 0; Rin < XParam.Rivers.size(); Rin++)
		{

			//qnow = interptime(slbnd[SLstepinbnd].wlev0, slbnd[SLstepinbnd - 1].wlev0, slbnd[SLstepinbnd].time - slbnd[SLstepinbnd - 1].time, totaltime - slbnd[SLstepinbnd - 1].time);
			int bndstep = 0;
			double difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
			while (difft <= 0.0) // danger?
			{
				bndstep++;
				difft = XParam.Rivers[Rin].flowinput[bndstep].time - XParam.totaltime;
			}

			qnow = interptime(XParam.Rivers[Rin].flowinput[bndstep].q, XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].q, XParam.Rivers[Rin].flowinput[bndstep].time - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time, XParam.totaltime - XParam.Rivers[Rin].flowinput[max(bndstep - 1, 0)].time);



			discharge_bnd_v << <gridDim, blockDim, 0 >> > (XParam.Rivers[Rin].xstart, XParam.Rivers[Rin].xend, XParam.Rivers[Rin].ystart, XParam.Rivers[Rin].yend, XParam.dx, XParam.dt, qnow, XParam.Rivers[Rin].disarea, blockxo_gd, blockyo_gd, zs_gd, hh_gd);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
	}

	return XParam.dt;
}


void meanmaxvarGPU(Param XParam)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	if (XParam.outuumean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >( uumean_g, uu_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outvvmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >( vvmean_g, vv_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outhhmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >( hhmean_g, hh_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outzsmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >( zsmean_g, zs_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outzsmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >( zsmax_g, zs_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outhhmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >( hhmax_g, hh_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outuumax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >(uumax_g, uu_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outvvmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >( vvmax_g, vv_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

}


void meanmaxvarGPUD(Param XParam)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0) / blockDim.x), ceil((ny*1.0) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	if (XParam.outuumean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >( uumean_gd, uu_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outvvmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >( vvmean_gd, vv_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outhhmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >( hhmean_gd, hh_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outzsmean == 1)
	{
		addavg_var << <gridDim, blockDim, 0 >> >( zsmean_gd, zs_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outzsmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >( zsmax_gd, zs_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outhhmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >( hhmax_gd, hh_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outuumax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >( uumax_gd, uu_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	if (XParam.outvvmax == 1)
	{
		max_var << <gridDim, blockDim, 0 >> >( vvmax_gd, vv_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

}


void DivmeanvarGPU(Param XParam, float nstep)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	if (XParam.outuumean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >( nstep, uumean_g);
		CUDA_CHECK(cudaDeviceSynchronize());

		
	}
	if (XParam.outvvmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >( nstep, vvmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());

		
	}
	if (XParam.outhhmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >( nstep, hhmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());

		
	}
	if (XParam.outzsmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >( nstep, zsmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	
	

}


void DivmeanvarGPUD(Param XParam, double nstep)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0) / blockDim.x), ceil((ny*1.0) / blockDim.y), 1);
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	if (XParam.outuumean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >( nstep, uumean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outvvmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >( nstep, vvmean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outhhmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >( nstep, hhmean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outzsmean == 1)
	{
		divavg_var << <gridDim, blockDim, 0 >> >( nstep, zsmean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}



}

void ResetmeanvarGPU(Param XParam)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	if (XParam.outuumean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(uumean_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outvvmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(vvmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outhhmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(hhmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outzsmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(zsmean_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}




void ResetmeanvarGPUD(Param XParam)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0) / blockDim.x), ceil((ny*1.0) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	if (XParam.outuumean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(uumean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outvvmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >( vvmean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outhhmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >(hhmean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outzsmean == 1)
	{
		resetavg_var << <gridDim, blockDim, 0 >> >( zsmean_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}
void ResetmaxvarGPU(Param XParam)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	if (XParam.outuumax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >( uumax_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outvvmax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >( vvmax_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outhhmax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >( hhmax_g);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outzsmax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >( zsmax_g);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}
void ResetmaxvarGPUD(Param XParam)
{
	//int nx = XParam.nx;
	//int ny = XParam.ny;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	if (XParam.outuumax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >(uumax_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outvvmax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >(vvmax_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outhhmax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >(hhmax_gd);
		CUDA_CHECK(cudaDeviceSynchronize());


	}
	if (XParam.outzsmax == 1)
	{
		resetmax_var << <gridDim, blockDim, 0 >> >(zsmax_gd);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}

// Main loop that actually runs the model
void mainloopGPU(Param XParam, std::vector<SLTS> leftWLbnd, std::vector<SLTS> rightWLbnd, std::vector<SLTS> topWLbnd, std::vector<SLTS> botWLbnd)
{
	double nextoutputtime = XParam.outputtimestep;
	int nstep = 0;
	int nTSsteps = 0;

	std::vector<Pointout> zsout;

	std::vector< std::vector< Pointout > > zsAllout;

	Pointout stepread;

	FILE * fsSLTS;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);


	for (int o = 0; o < XParam.TSoutfile.size(); o++)
	{
		//Overwrite existing files
		fsSLTS = fopen(XParam.TSoutfile[o].c_str(), "w");
		fprintf(fsSLTS, "# x=%f\ty=%f\ti=%d\tj=%d\t%s\n", XParam.TSnodesout[o].x, XParam.TSnodesout[o].y, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSoutfile[o].c_str());
		fclose(fsSLTS);

		// Add empty row for each output point
		zsAllout.push_back(std::vector<Pointout>());
	}
	// Reset GPU mean and max arrays
	if (XParam.spherical == 1 || XParam.doubleprecision == 1)
	{
		ResetmeanvarGPUD(XParam);
		ResetmaxvarGPUD(XParam);
	}
	else
	{
		ResetmeanvarGPU(XParam);
		ResetmaxvarGPU(XParam);
	}

	while (XParam.totaltime < XParam.endtime)
	{
		// Bnd stuff here
		LeftFlowBnd(XParam, leftWLbnd);
		RightFlowBnd(XParam, rightWLbnd);
		TopFlowBnd(XParam, topWLbnd);
		BotFlowBnd(XParam, botWLbnd);

		// Run the model step
		if (XParam.spherical == 1)
		{
			XParam.dt = FlowGPUSpherical(XParam, nextoutputtime);
		}
		else
		{
			if(XParam.doubleprecision==1)
			{
				XParam.dt = FlowGPUDouble(XParam, nextoutputtime);
			}
			else
			{
				XParam.dt = FlowGPU(XParam, nextoutputtime);
			}
			
		}
		
		
		//Time keeping
		XParam.totaltime = XParam.totaltime + XParam.dt;
		nstep++;
		
		// Do Sum & Max variables Here
		if (XParam.spherical == 1 || XParam.doubleprecision == 1)
		{
			meanmaxvarGPUD(XParam);
		}
		else
		{
			meanmaxvarGPU(XParam);
		}
		


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

				if (XParam.spherical == 1 || XParam.doubleprecision == 1)
				{
					storeTSout << <gridDim, blockDim, 0 >> > ( (int)XParam.TSnodesout.size(), o, nTSsteps, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSnodesout[o].block, zs_gd, hh_gd, uu_gd, vv_gd, TSstore_gd);
				}
				else
				{
					storeTSout << <gridDim, blockDim, 0 >> > ( (int)XParam.TSnodesout.size(), o, nTSsteps, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSnodesout[o].block, zs_g, hh_g, uu_g, vv_g, TSstore_g);
				}
				
				CUDA_CHECK(cudaDeviceSynchronize());
			}
			nTSsteps++;
			
			if ((nTSsteps+1)*XParam.TSnodesout.size() * 4 > 2048 || XParam.endtime-XParam.totaltime <= XParam.dt*0.00001f)
			{
				//Flush
				if (XParam.spherical == 1 || XParam.doubleprecision == 1)
				{
					CUDA_CHECK(cudaMemcpy(TSstore_d, TSstore_gd, 2048 * sizeof(double), cudaMemcpyDeviceToHost));
					for (int o = 0; o < XParam.TSnodesout.size(); o++)
					{
						fsSLTS = fopen(XParam.TSoutfile[o].c_str(), "a");
						for (int n = 0; n < nTSsteps; n++)
						{
							//


							fprintf(fsSLTS, "%f\t%.4f\t%.4f\t%.4f\t%.4f\n", zsAllout[o][n].time, TSstore_d[1 + o * 4 + n*XParam.TSnodesout.size() * 4], TSstore_d[0 + o * 4 + n*XParam.TSnodesout.size() * 4], TSstore_d[2 + o * 4 + n*XParam.TSnodesout.size() * 4], TSstore_d[3 + o * 4 + n*XParam.TSnodesout.size() * 4]);


						}
						fclose(fsSLTS);
						//reset zsout
						zsAllout[o].clear();
					}
					nTSsteps = 0;
				}
				else
				{

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
			

		}

		if (nextoutputtime - XParam.totaltime <= XParam.dt*0.00001f  && XParam.outputtimestep > 0)
		{
			if (XParam.spherical == 1 || XParam.doubleprecision == 1)
			{
				DivmeanvarGPUD(XParam, nstep);

				if (XParam.outvort == 1)
				{
					CalcVorticity << <gridDim, blockDim, 0 >> > ( vort_gd, dvdx_gd, dudy_gd);
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
								CUDA_CHECK(cudaMemcpy(OutputVarMapCPUD[XParam.outvars[ivar]], OutputVarMapGPUD[XParam.outvars[ivar]], OutputVarMaplen[XParam.outvars[ivar]] * sizeof(double), cudaMemcpyDeviceToHost));

							}
							//Create definition for each variable and store it
							writencvarstepD(XParam,blockxo_d,blockyo_d, XParam.outvars[ivar], OutputVarMapCPUD[XParam.outvars[ivar]]);
						}
					}
				}
			}
			else
			{

				// Avg var sum here
				DivmeanvarGPU(XParam, nstep*1.0f);

				if (XParam.outvort == 1)
				{
					CalcVorticity << <gridDim, blockDim, 0 >> > ( vort_g, dvdx_g, dudy_g);
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
							writencvarstep(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
						}
					}
				}
			}
			nextoutputtime = min(nextoutputtime + XParam.outputtimestep,XParam.endtime);

			printf("Writing output, totaltime:%f s, Mean dt=%f\n", XParam.totaltime, XParam.outputtimestep / nstep);
			write_text_to_log_file("Writing outputs, totaltime: " + std::to_string(XParam.totaltime) + ", Mean dt= " + std::to_string(XParam.outputtimestep / nstep));

			//.Reset Avg Variables
			if (XParam.spherical == 1 || XParam.doubleprecision == 1)
			{
				ResetmeanvarGPUD(XParam);
				if (XParam.resetmax == 1)
				{
					ResetmaxvarGPUD(XParam);
				}
			}
			else
			{
				ResetmeanvarGPU(XParam);
				if (XParam.resetmax == 1)
				{
					ResetmaxvarGPU(XParam);
				}
			}
			


			//

			// Reset nstep
			nstep = 0;
		} // End of output part

	} //Main while loop
}




void mainloopCPU(Param XParam, std::vector<SLTS> leftWLbnd, std::vector<SLTS> rightWLbnd, std::vector<SLTS> topWLbnd, std::vector<SLTS> botWLbnd)
{
	double nextoutputtime = XParam.outputtimestep;
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
		if (XParam.spherical == 1)
		{
			XParam.dt = FlowCPUSpherical(XParam, nextoutputtime);
		}
		else
		{
			if (XParam.doubleprecision==1)
			{
				XParam.dt = FlowCPUDouble(XParam, nextoutputtime);
			}
			else
			{
				XParam.dt = FlowCPU(XParam, nextoutputtime);
			}
		}

		//Time keeping
		XParam.totaltime = XParam.totaltime + XParam.dt;
		nstep++;

		// Do Sum & Max variables Here
		if (XParam.doubleprecision == 1 || XParam.spherical == 1)
		{
			AddmeanCPUD(XParam);
			maxallCPUD(XParam);
		}
		else
		{
			AddmeanCPU(XParam);
			maxallCPU(XParam);
		}
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

			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				DivmeanCPUD(XParam, (double)nstep);
				if (XParam.outvort == 1)
				{
					CalcVortD(XParam);
				}
			}
			else
			{
				DivmeanCPU(XParam, (float)nstep);
				if (XParam.outvort == 1)
				{
					CalcVort(XParam);
				}
			}
			
			// Check for and calculate Vorticity if required
			

			if (!XParam.outvars.empty())
			{
				writenctimestep(XParam.outfile, XParam.totaltime);

				for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
				{
					if (OutputVarMaplen[XParam.outvars[ivar]] > 0)
					{
						
						//write output step for each variable 
						if (XParam.doubleprecision == 1 || XParam.spherical == 1)
						{
							writencvarstepD(XParam,blockxo_d,blockyo_d, XParam.outvars[ivar], OutputVarMapCPUD[XParam.outvars[ivar]]);
						}
						else
						{
							writencvarstep(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
						}
						
					}
				}
			}
			nextoutputtime = min(nextoutputtime + XParam.outputtimestep, XParam.endtime);

			printf("Writing output, totaltime:%f s, Mean dt=%f\n", XParam.totaltime, XParam.outputtimestep / nstep);
			write_text_to_log_file("Writing outputs, totaltime: " + std::to_string(XParam.totaltime) + ", Mean dt= " + std::to_string(XParam.outputtimestep / nstep));

			//.Reset Avg Variables

			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				ResetmeanCPUD(XParam);
			}
			else
			{
				ResetmeanCPU(XParam);
			}
			

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
	time_t rawtime;
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

	///////////////////////////////////////////
	//  Read Bathy header
	///////////////////////////////////////////

	//this sets nx ny dx delta xo yo etc...

	XParam = readBathyhead(XParam);

	

	//////////////////////////////////////////////////
	////// Preprare Bnd
	//////////////////////////////////////////////////

	// So far bnd are limited to be cst along an edge
	// Read Bnd file if/where needed
	printf("Reading and preparing Boundaries...");
	write_text_to_log_file("Reading and preparing Boundaries");

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


	//Check that endtime is no longer than boundaries (if specified to other than wall or neumann)
	XParam.endtime = setendtime(XParam, leftWLbnd, rightWLbnd, topWLbnd, botWLbnd);


	printf("...done!\n");
	write_text_to_log_file("Done Reading and preparing Boundaries");

	XParam.dt = 0.0;// Will be resolved in update

	int nx = XParam.nx;
	int ny = XParam.ny;



	////////////////////////////////////////////////
	// read the bathy file (and store to dummy for now)
	////////////////////////////////////////////////
	Allocate1CPU(XParam.nx, XParam.ny, dummy);
	Allocate1CPU(XParam.nx, XParam.ny, dummy_d);

	printf("Read Bathy data...");
	write_text_to_log_file("Read Bathy data");


	// Check bathy extension 
	std::string bathyext;

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

	//Now choose the right function to read the data

	if (bathyext.compare("md") == 0)
	{
		readbathyMD(XParam.Bathymetryfile, dummy);
	}
	if (bathyext.compare("nc") == 0)
	{
		readnczb(XParam.nx, XParam.ny, XParam.Bathymetryfile, dummy);
	}
	if (bathyext.compare("bot") == 0 || bathyext.compare("dep") == 0)
	{
		readXBbathy(XParam.Bathymetryfile, XParam.nx, XParam.ny, dummy);
	}
	if (bathyext.compare("asc") == 0)
	{
		//
		readbathyASCzb(XParam.Bathymetryfile, XParam.nx, XParam.ny, dummy);
	}



	//printf("%f\n", zb[0]);
	//printf("%f\n", zb[(nx - 1) + (0)*nx]);
	//printf("%f\n", zb[(0) + (ny-1)*nx]);
	//printf("%f\n", zb[(nx - 1) + (ny - 1)*nx]);


	//init variables
	if (XParam.posdown == 1)
	{
		printf("Bathy data is positive down...correcting ...");
		write_text_to_log_file("Bathy data is positive down...correcting");
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				dummy[i + j*nx] = dummy[i + j*nx] * -1.0f;
				//printf("%f\n", zb[i + (j)*nx]);

			}
		}
	}
	printf("...done\n");
	////////////////////////////////////////////////
	// Rearrange the memory in uniform blocks
	////////////////////////////////////////////////
	
	//max nb of blocks is ceil(nx/16)*ceil(ny/16)
	int nblk = 0;
	int nmask = 0;
	int mloc = 0;
	for (int nblky = 0; nblky < ceil(ny / 16.0); nblky++)
	{
		for (int nblkx = 0; nblkx < ceil(nx / 16.0); nblkx++)
		{
			nmask = 0;
			for (int i = 0; i < 16; i++)
			{
				for (int j = 0; j < 16; j++)
				{
					int ix = min(i + 16 * nblkx, nx-1);
					int iy = min(j + nblky * 16 , ny-1);
					mloc = ix + iy*nx ;
					//printf("mloc: %i\n", mloc);
					if (dummy[mloc] >= XParam.mask)
						nmask++;

				}
			}
			if (nmask < 256)
				nblk++;
		}
	}

	XParam.nblk = nblk;
	
	int blksize = XParam.blksize; //useful below
	printf("Number of blocks: %i\n",nblk);

	////////////////////////////////////////////////
	///// Allocate and arrange blocks
	////////////////////////////////////////////////
	// caluculate the Block xo yo and what are its neighbour
	

	Allocate1CPU(nblk, 1, blockxo);
	Allocate1CPU(nblk, 1, blockyo);
	Allocate1CPU(nblk, 1, blockxo_d);
	Allocate1CPU(nblk, 1, blockyo_d);
	Allocate4CPU(nblk, 1, leftblk, rightblk, topblk, botblk);

	nmask = 0;
	mloc = 0;
	int blkid = 0;
	for (int nblky = 0; nblky < ceil(ny / 16.0); nblky++)
	{
		for (int nblkx = 0; nblkx < ceil(nx / 16.0); nblkx++)
		{
			nmask = 0;
			for (int i = 0; i < 16; i++)
			{
				for (int j = 0; j < 16; j++)
				{
					int ix = min(i + 16 * nblkx, nx - 1);
					int iy = min(j + nblky * 16, ny - 1);
					mloc = ix + iy*nx;
					//printf("mloc: %i\n", mloc);
					if (dummy[mloc] >= XParam.mask)
						nmask++;

				}
			}
			if (nmask < 256)
			{
				//
				blockxo_d[blkid] = XParam.xo + nblkx * 16.0 * XParam.dx;
				blockyo_d[blkid] = XParam.yo + nblky * 16 * XParam.dx;
				blkid++;
			}
		}
	}

	double leftxo, rightxo, topxo, botxo, leftyo, rightyo, topyo, botyo;
	for (int bl = 0; bl < nblk; bl++)
	{
		leftxo = blockxo_d[bl] - 16.0 * XParam.dx; // in adaptive this shoulbe be a range 
		leftyo = blockyo_d[bl];
		rightxo = blockxo_d[bl] + 16.0 * XParam.dx;
		rightyo = blockyo_d[bl];
		topxo = blockxo_d[bl];
		topyo = blockyo_d[bl] + 16.0 * XParam.dx;
		botxo = blockxo_d[bl];
		botyo = blockyo_d[bl] - 16.0 * XParam.dx;

		// by default neighbour block refer to itself. i.e. if the neighbour block is itself then there are no neighbour 
		leftblk[bl] = bl;
		rightblk[bl] = bl;
		topblk[bl] = bl;
		botblk[bl] = bl;
		for (int blb = 0; blb < nblk; blb++)
		{
			//
			if (blockxo_d[blb] == leftxo && blockyo_d[blb] == leftyo)
			{
				leftblk[bl] = blb;
			}
			if (blockxo_d[blb] == rightxo && blockyo_d[blb] == rightyo)
			{
				rightblk[bl] = blb;
			}
			if (blockxo_d[blb] == topxo && blockyo_d[blb] == topyo)
			{
				topblk[bl] = blb;
			}
			if (blockxo_d[blb] == botxo && blockyo_d[blb] == botyo)
			{
				botblk[bl] = blb;
			}
		}

	}

	for (int bl = 0; bl < nblk; bl++)
	{
		blockxo[bl] = blockxo_d[bl];
		blockyo[bl] = blockyo_d[bl];
	}


	// Also recalculate xmax and ymax here
	//xo + (ceil(nx / 16.0)*16.0 - 1)*dx
	XParam.xmax = XParam.xo + (ceil(nx / 16.0) * 16.0 - 1)*XParam.dx;
	XParam.ymax = XParam.yo + (ceil(ny / 16.0) * 16.0 - 1)*XParam.dx;

	////////////////////////////////////////////////
	///// Allocate memory on CPU
	////////////////////////////////////////////////

	printf("Allocate CPU memory...");
	write_text_to_log_file("Allocate CPU memory...");
	int check;
	
	check = AllocMemCPU(XParam);



	printf("...done!\n");
	write_text_to_log_file("Done");

	////////////////////////////////////////////////
	///// Find and prepare GPU device
	////////////////////////////////////////////////

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
		cudaGetDeviceProperties(&prop, XParam.GPUDEVICE);
		printf("There are %d GPU devices on this machine\n", nDevices);
		printf("Using Device : %s\n", prop.name);


		write_text_to_log_file("There are " + std::to_string(nDevices) + "GPU devices on this machine");
		write_text_to_log_file("There are " + std::string(prop.name) + "GPU devices on this machine");

	}

	// Now that we checked that there was indeed a GPU available
	////////////////////////////////////////
	//////// ALLLOCATE GPU memory
	////////////////////////////////////////
	if (XParam.GPUDEVICE >= 0)
	{
		printf("Allocating GPU memory...");
		write_text_to_log_file("Allocating GPU memory");
		int check;
		check = AllocMemGPU(XParam);
		check = AllocMemGPUBND(XParam, leftWLbnd, rightWLbnd, topWLbnd, botWLbnd);
		
		printf("Done\n");
		write_text_to_log_file("Done");

	}

	
	////////////////////////////////////////
	//////// Copy initial cartesian bathy array to BUQ array
	////////////////////////////////////////
	printf("Copy bathy to BUQ array...");
	write_text_to_log_file("Copy bathy to BUQ array...");
	// Copy dummy to zb
	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				dummy_d[i + j*nx] = dummy[i + j*nx] * 1.0;
			}
		}

		carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo_d, blockyo_d, dummy_d, zb_d);
	}
	else
	{
		carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo_d, blockyo_d, dummy, zb);
	}


	// free dummy and dummy_d because they are of size nx*ny but we want them nblk*blksize since we can't predict if one is larger then the other I'd rather free and malloc rather the realloc
	free(dummy);
	free(dummy_d);

	Allocate1CPU(XParam.nblk, XParam.blksize, dummy);
	Allocate1CPU(XParam.nblk, XParam.blksize, dummy_d);

	printf("Done\n");
	write_text_to_log_file("Done");


	// set grid edges. this is necessary for boundary conditions to work
	// Shouldn't this be done after the hotstarts es?
	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{
		//setedges(nx, ny, zb_d);
		//setedges(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo_d, blockyo_d, zb_d);
		setedges(XParam.nblk, leftblk, rightblk, topblk, botblk, zb_d);
	}
	else
	{
		setedges(XParam.nblk, leftblk, rightblk, topblk, botblk, zb);
	}
	

	/////////////////////////////////////////////////////
	// Prep River discharge
	/////////////////////////////////////////////////////
	
	if (XParam.Rivers.size() > 1)
	{
		double xx, yy;
		printf("Preparing rivers ");
		write_text_to_log_file("Preparing rivers");
		//For each rivers
		for (int Rin = 0; Rin < XParam.Rivers.size(); Rin++)
		{
			// find the cells where the river discharge will be applied
			std::vector<int> idis, jdis, blockdis;
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						xx = blockxo_d[bl] + i*XParam.dx;
						yy = blockyo_d[bl] + j*XParam.dx;
						// the conditions are that the discharge area as defined by the user have to include at least a model grid node
						// This could be really annoying and there should be a better way to deal wiith this like polygon intersection
						if (xx >= XParam.Rivers[Rin].xstart && xx <= XParam.Rivers[Rin].xend && yy >= XParam.Rivers[Rin].ystart && yy <= XParam.Rivers[Rin].yend)
						{
							
							// This cell belongs to the river discharge area
							idis.push_back(i);
							jdis.push_back(j);
							blockdis.push_back(bl);

						}
					}
				}
				
			}

			XParam.Rivers[Rin].i = idis;
			XParam.Rivers[Rin].j = jdis;
			XParam.Rivers[Rin].block = blockdis;
			XParam.Rivers[Rin].disarea = idis.size()*XParam.dx; // That is not valid for spherical grids

			// Now read the discharge input and store to  
			XParam.Rivers[Rin].flowinput = readFlowfile(XParam.Rivers[Rin].Riverflowfile);
		}
	}

	/////////////////////////////////////////////////////
	// Initial Condition
	/////////////////////////////////////////////////////
	printf("Initial condition: ");
	write_text_to_log_file("Initial condition:");

	//move this to a subroutine 
	int hotstartsucess = 0;
	if (!XParam.hotstartfile.empty())
	{
		// hotstart
		printf("Hotstart "); 
		write_text_to_log_file("Hotstart");
		if (XParam.doubleprecision == 1 || XParam.spherical == 1)
		{
			hotstartsucess = readhotstartfileD(XParam, blockxo_d, blockyo_d, dummy_d, zs_d, zb_d, hh_d, uu_d, vv_d);
		}
		else
		{
			hotstartsucess = readhotstartfile(XParam, blockxo_d, blockyo_d, dummy, zs, zb, hh, uu, vv);
		}
		
		if (hotstartsucess == 0)
		{
			printf("Failed...  ");
			write_text_to_log_file("Hotstart failed switching to cold start");
		}
	}
	if (XParam.hotstartfile.empty() || hotstartsucess == 0)
	{
		printf("Cold start  ");
		write_text_to_log_file("Cold start");
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
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				for (int bl = 0; bl < XParam.nblk; bl++)
				{
					for (int j = 0; j < 16; j++)
					{
						for (int i = 0; i < 16; i++)
						{
							int n = i + j * 16 + bl * XParam.blksize;

							uu_d[n] = 0.0;
							vv_d[n] = 0.0;
							//zb[n] = 0.0f;
							zs_d[n] = max(XParam.zsinit, zb_d[n]);
							//if (i >= 64 && i < 82)
							//{
							//	zs[n] = max(zsbnd+0.2f, zb[i + j*nx]);
							//}
							hh_d[n] = max(zs_d[n] - zb_d[n], XParam.eps);//0.0?

						}

					}
				}
			}
			else
			{
				for (int bl = 0; bl < XParam.nblk; bl++)
				{
					for (int j = 0; j < 16; j++)
					{
						for (int i = 0; i < 16; i++)
						{
							int n = i + j * 16 + bl * XParam.blksize;
							uu[n] = 0.0f;
							vv[n] = 0.0f;
							//zb[i + j*nx] = 0.0f;
							zs[n] = max((float)XParam.zsinit, zb[n]);
							//if (i >= 64 && i < 82)
							//{
							//	zs[i + j*nx] = max(zsbnd+0.2f, zb[i + j*nx]);
							//}
							hh[n] = max(zs[n] - zb[n], (float)XParam.eps);//0.0f?
						}

					}
				}
			}

		}
		else // lukewarm start i.e. bilinear interpolation of zs at bnds // Argggh!
		{
			double zsleft = 0.0;
			double zsright = 0.0;
			double zstop = 0.0;
			double zsbot = 0.0;
			double zsbnd = 0.0;

			double distleft, distright, disttop, distbot;

			double lefthere = 0.0;
			double righthere = 0.0;
			double tophere = 0.0;
			double bothere = 0.0;

			double xi, yi, jj, ii;

			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						xi = blockxo_d[bl] + i*XParam.dx;
						yi = blockyo_d[bl] + j*XParam.dx;

						disttop = max((XParam.yo + (ny - 1)*XParam.dx - yi) / XParam.dx, 0.1);//max((double)(ny - 1) - j, 0.1);// WTF is that 0.1? // distleft cannot be 0 
						distbot = max((yi - XParam.yo) / XParam.dx, 0.1);
						distleft = max((xi - XParam.xo) / XParam.dx, 0.1);//max((double)i, 0.1);
						distright = max((XParam.xo + (nx - 1)*XParam.dx - xi) / XParam.dx, 0.1);//max((double)(nx - 1) - i, 0.1);

						jj = (yi - XParam.yo) / XParam.dx;
						ii = (xi - XParam.xo) / XParam.dx;

						if (XParam.left == 1 && !leftWLbnd.empty())
						{
							lefthere = 1.0;
							int SLstepinbnd = 1;



							// Do this for all the corners
							//Needs limiter in case WLbnd is empty
							double difft = leftWLbnd[SLstepinbnd].time - XParam.totaltime;

							while (difft < 0.0)
							{
								SLstepinbnd++;
								difft = leftWLbnd[SLstepinbnd].time - XParam.totaltime;
							}
							std::vector<double> zsbndvec;
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
								int iprev = min(max((int)ceil(jj / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
								int inext = iprev + 1;
								// here interp time is used to interpolate to the right node rather than in time...
								zsleft = interptime(zsbndvec[inext], zsbndvec[iprev], (double)(inext - iprev), (double)(jj - iprev));
							}

						}

						if (XParam.right == 1 && !rightWLbnd.empty())
						{
							int SLstepinbnd = 1;
							righthere = 1.0;


							// Do this for all the corners
							//Needs limiter in case WLbnd is empty
							double difft = rightWLbnd[SLstepinbnd].time - XParam.totaltime;

							while (difft < 0.0)
							{
								SLstepinbnd++;
								difft = rightWLbnd[SLstepinbnd].time - XParam.totaltime;
							}
							std::vector<double> zsbndvec;
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
								int iprev = min(max((int)ceil(jj / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
								int inext = iprev + 1;
								// here interp time is used to interpolate to the right node rather than in time...
								zsright = interptime(zsbndvec[inext], zsbndvec[iprev], (double)(inext - iprev), (double)(jj - iprev));
							}


						}
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
							std::vector<double> zsbndvec;
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
								int iprev = min(max((int)ceil(ii / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
								int inext = iprev + 1;
								// here interp time is used to interpolate to the right node rather than in time...
								zsbot = interptime(zsbndvec[inext], zsbndvec[iprev], (double)(inext - iprev), (double)(ii - iprev));
							}

						}
						if (XParam.top == 1 && !topWLbnd.empty())
						{
							int SLstepinbnd = 1;
							tophere = 1.0;




							// Do this for all the corners
							//Needs limiter in case WLbnd is empty
							double difft = topWLbnd[SLstepinbnd].time - XParam.totaltime;

							while (difft < 0.0)
							{
								SLstepinbnd++;
								difft = topWLbnd[SLstepinbnd].time - XParam.totaltime;
							}
							std::vector<double> zsbndvec;
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
								int iprev = min(max((int)ceil(ii / (1 / (zsbndvec.size() - 1))), 0), (int)zsbndvec.size() - 2);
								int inext = iprev + 1;
								// here interp time is used to interpolate to the right node rather than in time...
								zstop = interptime(zsbndvec[inext], zsbndvec[iprev], (double)(inext - iprev), (double)(ii - iprev));
							}

						}


						zsbnd = ((zsleft * 1 / distleft)*lefthere + (zsright * 1 / distright)*righthere + (zstop * 1 / disttop)*tophere + (zsbot * 1 / distbot)*bothere) / ((1 / distleft)*lefthere + (1 / distright)*righthere + (1 / disttop)*tophere + (1 / distbot)*bothere);


						if (XParam.doubleprecision == 1 || XParam.spherical == 1)
						{
							zs_d[n] = max(zsbnd, zb_d[n]);
							hh_d[n] = max(zs_d[n] - zb_d[n], XParam.eps);
							uu_d[n] = 0.0;
							vv_d[n] = 0.0;
						}
						else
						{
							zs[n] = max((float)zsbnd, zb[n]);
							hh[n] = max(zs[i + j*nx] - zb[n], (float)XParam.eps);
							uu[n] = 0.0f;
							vv[n] = 0.0f;
							
						}

					}
				}
			}
		}
		
	}
	printf("done \n  ");
	write_text_to_log_file("Done");


	

	// Below is not succint but way faster than one loop that checks the if statemenst each time
	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{
		// Set default cf 
		for (int bl = 0; bl < XParam.nblk; bl++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int i = 0; i < 16; i++)
				{
					int n = i + j * 16 + bl * XParam.blksize;
					cf_d[n] = XParam.cf;
				}
			}
		}
		if (XParam.outhhmax == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						hhmax_d[n] = hh_d[n];
					}
				}
			}
		}

		if (XParam.outhhmean == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						hhmean_d[n] = 0.0;
					}
				}
			}
		}
		if (XParam.outzsmax == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						zsmax_d[n] = zs_d[n];
					}
				}
			}
		}

		if (XParam.outzsmean == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						zsmean_d[n] = 0.0;
					}
				}
			}
		}

		if (XParam.outuumax == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						uumax_d[n] = uu_d[n];
					}
				}
			}
		}

		if (XParam.outuumean == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						uumean_d[n] = 0.0;
					}
				}
			}
		}
		if (XParam.outvvmax == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						vvmax_d[n] = vv_d[n];
					}
				}
			}
		}

		if (XParam.outvvmean == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						vvmean_d[n] = 0.0;
					}
				}
			}
		}
		if (XParam.outvort == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						vort_d[n] = 0.0;
					}
				}
			}
		}
	}
	else //Using Float *
	{
		// Set default cf 
		for (int bl = 0; bl < XParam.nblk; bl++)
		{
			for (int j = 0; j < 16; j++)
			{
				for (int i = 0; i < 16; i++)
				{
					int n = i + j * 16 + bl * XParam.blksize;
					cf[n] = (float)XParam.cf;
				}
			}
		}
		
		if (XParam.outhhmax == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						hhmax[n] = hh[n];
					}
				}
			}
		}

		if (XParam.outhhmean == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						hhmean[n] = 0.0;
					}
				}
			}
		}
		if (XParam.outzsmax == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						zsmax[n] = zs[n];
					}
				}
			}
		}

		if (XParam.outzsmean == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						zsmean[n] = 0.0;
					}
				}
			}
		}

		if (XParam.outuumax == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						uumax[n] = uu[n];
					}
				}
			}
		}

		if (XParam.outuumean == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						uumean[n] = 0.0;
					}
				}
			}
		}
		if (XParam.outvvmax == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						vvmax[n] = vv[n];
					}
				}
			}
		}

		if (XParam.outvvmean == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						vvmean[n] = 0.0;
					}
				}
			}
		}
		if (XParam.outvort == 1)
		{
			for (int bl = 0; bl < XParam.nblk; bl++)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int i = 0; i < 16; i++)
					{
						int n = i + j * 16 + bl * XParam.blksize;
						vort[n] = 0.0;
					}
				}
			}
		}
	}
	

	if (XParam.GPUDEVICE >= 0)
	{
		printf("Init data on GPU ");
		write_text_to_log_file("Init data on GPU ");

		dim3 blockDim(16, 16, 1);
		dim3 gridDim(nblk, 1, 1);

		if (XParam.doubleprecision == 1 || XParam.spherical == 1)
		{
			CUDA_CHECK(cudaMemcpy(zb_gd, zb_d, nblk*blksize * sizeof(double), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(hh_gd, hh_d, nblk*blksize * sizeof(double), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(uu_gd, uu_d, nblk*blksize * sizeof(double), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(vv_gd, vv_d, nblk*blksize * sizeof(double), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(zs_gd, zs_d, nblk*blksize * sizeof(double), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(cf_gd, cf_d, nblk*blksize * sizeof(double), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(blockxo_gd, blockxo_d, nblk * sizeof(double), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(blockyo_gd, blockyo_d, nblk * sizeof(double), cudaMemcpyHostToDevice));
			
			initdtmax << <gridDim, blockDim, 0 >> >(epsilon, dtmax_gd);
		}
		else
		{
			CUDA_CHECK(cudaMemcpy(zb_g, zb, nblk*blksize * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(hh_g, hh, nblk*blksize * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(uu_g, uu, nblk*blksize * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(vv_g, vv, nblk*blksize * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(zs_g, zs, nblk*blksize * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(cf_g, cf, nblk*blksize * sizeof(double), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(blockxo_g, blockxo, nblk * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(blockyo_g, blockyo, nblk * sizeof(float), cudaMemcpyHostToDevice));
			initdtmax << <gridDim, blockDim, 0 >> >( (float)epsilon, dtmax_g);
		}
		
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(leftblk_g, leftblk, nblk * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(rightblk_g, rightblk, nblk * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(topblk_g, topblk, nblk * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(botblk_g, botblk, nblk * sizeof(int), cudaMemcpyHostToDevice));
		printf("...Done\n ");
		write_text_to_log_file("Done ");

	}

	// Here map array to their name as a string. it makes it super easy to convert user define variables to the array it represents.
	// COul add more to output gradients etc...
	OutputVarMapCPU["zb"] = zb;
	OutputVarMapCPUD["zb"] = zb_d;
	OutputVarMapGPU["zb"] = zb_g;
	OutputVarMapGPUD["zb"] = zb_gd;
	OutputVarMaplen["zb"] = nblk*blksize;

	OutputVarMapCPU["uu"] = uu;
	OutputVarMapCPUD["uu"] = uu_d;
	OutputVarMapGPU["uu"] = uu_g;
	OutputVarMapGPUD["uu"] = uu_gd;
	OutputVarMaplen["uu"] = nblk*blksize;

	OutputVarMapCPU["vv"] = vv;
	OutputVarMapCPUD["vv"] = vv_d;
	OutputVarMapGPU["vv"] = vv_g;
	OutputVarMapGPUD["vv"] = vv_gd;
	OutputVarMaplen["vv"] = nblk*blksize;

	OutputVarMapCPU["zs"] = zs;
	OutputVarMapCPUD["zs"] = zs_d;
	OutputVarMapGPU["zs"] = zs_g;
	OutputVarMapGPUD["zs"] = zs_gd;
	OutputVarMaplen["zs"] = nblk*blksize;

	OutputVarMapCPU["hh"] = hh;
	OutputVarMapCPUD["hh"] = hh_d;
	OutputVarMapGPU["hh"] = hh_g;
	OutputVarMapGPUD["hh"] = hh_gd;
	OutputVarMaplen["hh"] = nblk*blksize;

	OutputVarMapCPU["hhmean"] = hhmean;
	OutputVarMapCPUD["hhmean"] = hhmean_d;
	OutputVarMapGPU["hhmean"] = hhmean_g;
	OutputVarMapGPUD["hhmean"] = hhmean_gd;
	OutputVarMaplen["hhmean"] = nblk*blksize;

	OutputVarMapCPU["hhmax"] = hhmax;
	OutputVarMapCPUD["hhmax"] = hhmax_d;
	OutputVarMapGPU["hhmax"] = hhmax_g;
	OutputVarMapGPUD["hhmax"] = hhmax_gd;
	OutputVarMaplen["hhmax"] = nblk*blksize;

	OutputVarMapCPU["zsmean"] = zsmean;
	OutputVarMapCPUD["zsmean"] = zsmean_d;
	OutputVarMapGPU["zsmean"] = zsmean_g;
	OutputVarMapGPUD["zsmean"] = zsmean_gd;
	OutputVarMaplen["zsmean"] = nblk*blksize;

	OutputVarMapCPU["zsmax"] = zsmax;
	OutputVarMapCPUD["zsmax"] = zsmax_d;
	OutputVarMapGPU["zsmax"] = zsmax_g;
	OutputVarMapGPUD["zsmax"] = zsmax_gd;
	OutputVarMaplen["zsmax"] = nblk*blksize;

	OutputVarMapCPU["uumean"] = uumean;
	OutputVarMapCPUD["uumean"] = uumean_d;
	OutputVarMapGPU["uumean"] = uumean_g;
	OutputVarMapGPUD["uumean"] = uumean_gd;
	OutputVarMaplen["uumean"] = nblk*blksize;

	OutputVarMapCPU["uumax"] = uumax;
	OutputVarMapCPUD["uumax"] = uumax_d;
	OutputVarMapGPU["uumax"] = uumax_g;
	OutputVarMapGPUD["uumax"] = uumax_gd;
	OutputVarMaplen["uumax"] = nblk*blksize;

	OutputVarMapCPU["vvmean"] = vvmean;
	OutputVarMapCPUD["vvmean"] = vvmean_d;
	OutputVarMapGPU["vvmean"] = vvmean_g;
	OutputVarMapGPUD["vvmean"] = vvmean_gd;
	OutputVarMaplen["vvmean"] = nblk*blksize;

	OutputVarMapCPU["vvmax"] = vvmax;
	OutputVarMapCPUD["vvmax"] = vvmax_d;
	OutputVarMapGPU["vvmax"] = vvmax_g;
	OutputVarMapGPUD["vvmax"] = vvmax_gd;
	OutputVarMaplen["vvmax"] = nblk*blksize;

	OutputVarMapCPU["vort"] = vort;
	OutputVarMapCPUD["vort"] = vort_d;
	OutputVarMapGPU["vort"] = vort_g;
	OutputVarMapGPUD["vort"] = vort_gd;
	OutputVarMaplen["vort"] = nblk*blksize;


	printf("Create netCDF output file ");
	write_text_to_log_file("Create netCDF output file ");
	//create nc file with no variables
	XParam=creatncfileUD(XParam);
	for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
	{
		//Create definition for each variable and store it
		if (XParam.doubleprecision == 1 || XParam.spherical == 1)
		{
			//defncvarD(XParam.outfile, XParam.smallnc, XParam.scalefactor, XParam.addoffset, nx, ny, XParam.outvars[ivar], 3, OutputVarMapCPUD[XParam.outvars[ivar]]);
			defncvarD(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], 3, OutputVarMapCPUD[XParam.outvars[ivar]]);
		}
		else
		{
			defncvar(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], 3, OutputVarMapCPU[XParam.outvars[ivar]]);
		}
		
	}
	//create2dnc(nx, ny, dx, dx, 0.0, xx, yy, hh);

	printf("done \n ");
	write_text_to_log_file("Done ");

	
	SaveParamtolog(XParam);


	printf("Starting Model.\n ");
	write_text_to_log_file("Starting Model. ");

	if (XParam.GPUDEVICE >= 0)
	{
		mainloopGPU(XParam, leftWLbnd, rightWLbnd, topWLbnd, botWLbnd);
		//checkGradGPU(XParam);
			
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

	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{
		free(hh_d);
		free(uu_d);
		free(vv_d);
		free(zb_d);
		free(zs_d);

		free(hho_d);
		free(uuo_d);
		free(vvo_d);
		free(zso_d);

		free(dhdx_d);
		free(dhdy_d);
		free(dudx_d);
		free(dudy_d);
		free(dvdx_d);
		free(dvdy_d);

		free(dzsdx_d);
		free(dzsdy_d);

		free(Su_d);
		free(Sv_d);
		free(Fqux_d);
		free(Fquy_d);
		free(Fqvx_d);
		free(Fqvy_d);
		free(Fhu_d);
		free(Fhv_d);

		free(dh_d);
		free(dhu_d);
		free(dhv_d);

		if (XParam.outhhmax == 1)
		{
			free(hhmax_d);
		}

		if (XParam.outzsmax == 1)
		{
			free(zsmax_d);
		}
		if (XParam.outuumax == 1)
		{
			free(uumax_d);
		}
		if (XParam.outvvmax == 1)
		{
			free(vvmax_d);
		}
		if (XParam.outhhmean == 1)
		{
			free(hhmean_d);
		}
		if (XParam.outzsmean == 1)
		{
			free(zsmean_d);
		}
		if (XParam.outuumean == 1)
		{
			free(uumean_d);
		}
		if (XParam.outvvmean == 1)
		{
			free(vvmax_d);
		}

		if (XParam.outvort == 1)
		{
			free(vort_d);
		}

		if (XParam.GPUDEVICE >= 0)
		{
			cudaFree(hh_gd);
			cudaFree(uu_gd);
			cudaFree(vv_gd);
			cudaFree(zb_gd);
			cudaFree(zs_gd);

			cudaFree(hho_gd);
			cudaFree(uuo_gd);
			cudaFree(vvo_gd);
			cudaFree(zso_gd);

			cudaFree(dhdx_gd);
			cudaFree(dhdy_gd);
			cudaFree(dudx_gd);
			cudaFree(dudy_gd);
			cudaFree(dvdx_gd);
			cudaFree(dvdy_gd);

			cudaFree(dzsdx_gd);
			cudaFree(dzsdy_gd);

			cudaFree(Su_gd);
			cudaFree(Sv_gd);
			cudaFree(Fqux_gd);
			cudaFree(Fquy_gd);
			cudaFree(Fqvx_gd);
			cudaFree(Fqvy_gd);
			cudaFree(Fhu_gd);
			cudaFree(Fhv_gd);

			cudaFree(dh_gd);
			cudaFree(dhu_gd);
			cudaFree(dhv_gd);

			cudaFree(dtmax_gd);


			cudaFree(arrmin_gd);
			cudaFree(arrmax_gd);

			if (XParam.outhhmax == 1)
			{
				cudaFree(hhmax_gd);
			}

			if (XParam.outzsmax == 1)
			{
				cudaFree(zsmax_gd);
			}
			if (XParam.outuumax == 1)
			{
				cudaFree(uumax_gd);
			}
			if (XParam.outvvmax == 1)
			{
				cudaFree(vvmax_gd);
			}
			if (XParam.outhhmean == 1)
			{
				cudaFree(hhmean_gd);
			}
			if (XParam.outzsmean == 1)
			{
				cudaFree(zsmean_gd);
			}
			if (XParam.outuumean == 1)
			{
				cudaFree(uumean_gd);
			}
			if (XParam.outvvmean == 1)
			{
				cudaFree(vvmax_gd);
			}

			if (XParam.outvort == 1)
			{
				cudaFree(vort_gd);
			}

			cudaDeviceReset();

		}
	}
	else
	{
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

		if (XParam.outhhmax == 1)
		{
			free(hhmax);
		}

		if (XParam.outzsmax == 1)
		{
			free(zsmax);
		}
		if (XParam.outuumax == 1)
		{
			free(uumax);
		}
		if (XParam.outvvmax == 1)
		{
			free(vvmax);
		}
		if (XParam.outhhmean == 1)
		{
			free(hhmean);
		}
		if (XParam.outzsmean == 1)
		{
			free(zsmean);
		}
		if (XParam.outuumean == 1)
		{
			free(uumean);
		}
		if (XParam.outvvmean == 1)
		{
			free(vvmax);
		}

		if (XParam.outvort == 1)
		{
			free(vort);
		}




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

			if (XParam.outhhmax == 1)
			{
				cudaFree(hhmax_g);
			}

			if (XParam.outzsmax == 1)
			{
				cudaFree(zsmax_g);
			}
			if (XParam.outuumax == 1)
			{
				cudaFree(uumax_g);
			}
			if (XParam.outvvmax == 1)
			{
				cudaFree(vvmax_g);
			}
			if (XParam.outhhmean == 1)
			{
				cudaFree(hhmean_g);
			}
			if (XParam.outzsmean == 1)
			{
				cudaFree(zsmean_g);
			}
			if (XParam.outuumean == 1)
			{
				cudaFree(uumean_g);
			}
			if (XParam.outvvmean == 1)
			{
				cudaFree(vvmax_g);
			}

			if (XParam.outvort == 1)
			{
				cudaFree(vort_g);
			}

			cudaDeviceReset();

		}
	}


	








	exit(0);
}


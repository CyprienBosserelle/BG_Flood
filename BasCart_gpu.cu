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


#ifdef USE_CATALYST
#include "catalyst_adaptor.h"
#endif

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

float* bathydata;

float * cf;
float * cf_g;
double * cf_d;
double * cf_gd;

// Block info
float * blockxo, *blockyo;
double * blockxo_d, *blockyo_d;
int * leftblk, *rightblk, *topblk, *botblk;
int * bndleftblk, *bndrightblk, *bndtopblk, *bndbotblk;

double * blockxo_gd, *blockyo_gd;
float * blockxo_g, *blockyo_g;
int * leftblk_g, *rightblk_g, *topblk_g, *botblk_g;
int * bndleftblk_g, *bndrightblk_g, *bndtopblk_g, *bndbotblk_g;

//River stuff
int * Riverblk, *Riverblk_g;

// Wind arrays
float * Uwind, *Uwbef, *Uwaft;
float * Vwind, *Vwbef, *Vwaft;
float * PatmX, *Patmbef, *Patmaft;
float * Patm, *dPdx, *dPdy;
double * Patm_d, *dPdx_d, *dPdy_d;

float * Uwind_g, *Uwbef_g, *Uwaft_g;
float * Vwind_g, *Vwbef_g, *Vwaft_g;
float * PatmX_g, *Patmbef_g, *Patmaft_g;
float * Patm_g, *dPdx_g, *dPdy_g;
double * Patm_gd, *dPdx_gd, *dPdy_gd;

//rain on grid
float *Rain, *Rainbef, *Rainaft;
float *Rain_g, *Rainbef_g, *Rainaft_g;

// Adaptivity
int * level, *level_g, *newlevel, *newlevel_g, *activeblk, *availblk, * invactive, *activeblk_g, *availblk_g, *csumblk, *csumblk_g,* invactive_g ;

bool* coarsen, * refine;;

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

cudaArray* leftUvel_gp;
cudaArray* rightUvel_gp;
cudaArray* topUvel_gp;
cudaArray* botUvel_gp;

cudaArray* leftVvel_gp;
cudaArray* rightVvel_gp;
cudaArray* topVvel_gp;
cudaArray* botVvel_gp;

// store wind data in cuda array before sending to texture memory
cudaArray* Uwind_gp;
cudaArray* Vwind_gp;
cudaArray* Patm_gp;
cudaArray* Rain_gp;

cudaChannelFormatDesc channelDescleftbndzs = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescrightbndzs = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescbotbndzs = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDesctopbndzs = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

cudaChannelFormatDesc channelDescleftbnduu = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescrightbnduu = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescbotbnduu = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDesctopbnduu = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

cudaChannelFormatDesc channelDescleftbndvv = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescrightbndvv = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescbotbndvv = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDesctopbndvv = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

cudaChannelFormatDesc channelDescUwind = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescVwind = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescPatm = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
cudaChannelFormatDesc channelDescRain = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);


// Below file are included rather than compiled separately because this allow to use Template for function
// Otherwise I would need to keep a double and a single precision copy of almost most function wich would be impossible to manage
#include "Flow_kernel.cu"
#include "Init.cpp" // excluded from direct buil to move the template out of the main source
#include "Init_gpu.cu"
#include "Adapt_Flow_kernel.cu"
#include "Adapt_gpu.cu"
#include "write_output.cu"

#include "Mainloop_Adapt.cu"


// Main loop that actually runs the model.
void mainloopGPUDB(Param XParam)
{
	double nextoutputtime = XParam.totaltime + XParam.outputtimestep;
	int nstep = 0;
	int nTSsteps = 0;
	int rainstep = 1;

	double rainuni = 0.0;
	std::vector<Pointout> zsout;

	std::vector< std::vector< Pointout > > zsAllout;

	Pointout stepread;

	FILE * fsSLTS;


	dim3 blockDimRain(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDimRain((int)ceil((float)XParam.Rainongrid.nx / (float)blockDimRain.x), (int)ceil((float)XParam.Rainongrid.ny / (float)blockDimRain.y), 1);

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
	ResetmeanvarGPUD(XParam);
	ResetmaxvarGPUD(XParam);
	while (XParam.totaltime < XParam.endtime)
	{
		// Bnd stuff here
		LeftFlowBnd(XParam);
		RightFlowBnd(XParam);
		TopFlowBnd(XParam);
		BotFlowBnd(XParam);


		// External forcing
		if (!XParam.Rainongrid.inputfile.empty())
		{
			//(Param XParam, dim3 gridDimRain, dim3 blockDimRain, int & rainstep, double rainuni)
			//this function moves in the forcing file for both inuform and variable input
			rainuni = Rainthisstep(XParam, gridDimRain, blockDimRain, rainstep);

		}



		// Core
		XParam.dt = FlowGPUDouble(XParam, nextoutputtime);
		//add rivers

		if (XParam.Rivers.size() > 0)
		{
			RiverSourceD(XParam);
		}

		// add rain
		if (!XParam.Rainongrid.inputfile.empty())
		{
			if (XParam.Rainongrid.uniform == 1)
			{
				Rain_on_gridUNI << <gridDim, blockDim, 0 >> > (XParam.mask, rainuni, XParam.dt, zs_gd, hh_gd);
				CUDA_CHECK(cudaDeviceSynchronize());
			}
			else
			{
				//(int unirain, float xorain, float yorain, float dxrain, double delta, double*blockxo, double *blockyo, double dt,  T * zs, T *hh)
				Rain_on_grid << <gridDim, blockDim, 0 >> > (XParam.mask, XParam.Rainongrid.xo, XParam.Rainongrid.yo, XParam.Rainongrid.dx, XParam.delta, blockxo_gd, blockyo_gd, XParam.dt, zs_gd, hh_gd);
				CUDA_CHECK(cudaDeviceSynchronize());
			}
		}



		//Time keeping
		XParam.totaltime = XParam.totaltime + XParam.dt;
		nstep++;

		// Do Sum & Max variables Here
		meanmaxvarGPUD(XParam);

		//check, store Timeseries output
		if (XParam.TSnodesout.size() > 0)
		{

			pointoutputstep(XParam, gridDim, blockDim, nTSsteps, zsAllout);
		}

		if (nextoutputtime - XParam.totaltime <= XParam.dt*0.00001f  && XParam.outputtimestep > 0.0)
		{
			// Save output step
			DivmeanvarGPUD(XParam, nstep);

			if (XParam.outvort == 1)
			{
				CalcVorticity << <gridDim, blockDim, 0 >> > (vort_gd, dvdx_gd, dudy_gd);
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
						//writencvarstep(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPUD[XParam.outvars[ivar]]);
						writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPUD[XParam.outvars[ivar]]);
					}
				}
			}

			// Log
			nextoutputtime = min(nextoutputtime + XParam.outputtimestep, XParam.endtime);

			printf("Writing output, totaltime:%f s, Mean dt=%f\n", XParam.totaltime, XParam.outputtimestep / nstep);
			write_text_to_log_file("Writing outputs, totaltime: " + std::to_string(XParam.totaltime) + ", Mean dt= " + std::to_string(XParam.outputtimestep / nstep));

			//Reset Avg Variables
			ResetmeanvarGPUD(XParam);
			if (XParam.resetmax == 1)
			{
				ResetmaxvarGPUD(XParam);
			}

			// Reset nstep
			nstep = 0;
		}
	}
}
void mainloopGPUDATM(Param XParam) // float, metric coordinate
{
	double nextoutputtime = XParam.totaltime + XParam.outputtimestep;
	int nstep = 0;
	int nTSsteps = 0;

	int windstep = 1;
	int atmpstep = 1;
	int rainstep = 1;

	double rainuni = 0.0;
	double uwinduni = 0.0;
	double vwinduni = 0.0;
	double atmpuni = XParam.Paref;
	std::vector<Pointout> zsout;

	std::vector< std::vector< Pointout > > zsAllout;

	Pointout stepread;

	FILE * fsSLTS;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);


	dim3 blockDimWND(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDimWND((int)ceil((float)XParam.windU.nx / (float)blockDimWND.x), (int)ceil((float)XParam.windU.ny / (float)blockDimWND.y), 1);

	dim3 blockDimATM(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDimATM((int)ceil((float)XParam.atmP.nx / (float)blockDimATM.x), (int)ceil((float)XParam.atmP.ny / (float)blockDimATM.y), 1);

	dim3 blockDimRain(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDimRain((int)ceil((float)XParam.Rainongrid.nx / (float)blockDimRain.x), (int)ceil((float)XParam.Rainongrid.ny / (float)blockDimRain.y), 1);


	int winduniform = XParam.windU.uniform;
	int atmpuniform = XParam.atmP.uniform;

	if (XParam.windU.inputfile.empty())// this is should be true here so not really needed (?)
	{
		// set as uniform run 0 wind input below
		winduniform = 1;
	}
	if (XParam.atmP.inputfile.empty())// this is should be true here so not really needed (?)
	{
		atmpuniform = 1;
	}


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
	ResetmeanvarGPU(XParam);
	ResetmaxvarGPU(XParam);


	while (XParam.totaltime < XParam.endtime)
	{
		// Bnd stuff here
		LeftFlowBnd(XParam);
		RightFlowBnd(XParam);
		TopFlowBnd(XParam);
		BotFlowBnd(XParam);




		// Core engine





		//XParam.dt = FlowGPUATM(XParam, nextoutputtime);

		const int num_streams = 3;

		cudaStream_t streams[num_streams];
		for (int i = 0; i < num_streams; i++)
		{
			CUDA_CHECK(cudaStreamCreate(&streams[i]));
		}



		//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
		//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);
		dim3 blockDim(16, 16, 1);
		dim3 gridDim(XParam.nblk, 1, 1);

		dtmax = (1.0 / epsilon);
		//float dtmaxtmp = dtmax;



		// Check the atm Pressure forcing before starting

		if (!XParam.atmP.inputfile.empty() && atmpuniform == 1) //this is a gven
		{

			//
			AtmPthisstep(XParam, gridDimATM, blockDimATM, atmpstep);

			interp2ATMP << <gridDim, blockDim, 0 >> > (XParam.atmP.xo, XParam.atmP.yo, XParam.atmP.dx, XParam.delta, XParam.Paref, blockxo_gd, blockyo_gd, Patm_gd);
			CUDA_CHECK(cudaDeviceSynchronize());


		}

		// External forcing
		if (!XParam.Rainongrid.inputfile.empty())
		{
			//(Param XParam, dim3 gridDimRain, dim3 blockDimRain, int & rainstep, double rainuni)
			//this function moves in the forcing file for both inuform and variable input
			rainuni = Rainthisstep(XParam, gridDimRain, blockDimRain, rainstep);

		}

		resetdtmax << <gridDim, blockDim, 0, streams[0] >> > (dtmax_gd);
		//CUDA_CHECK(cudaDeviceSynchronize());
		//update step 1



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hh_gd, dhdx_gd, dhdy_gd);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zs_gd, dzsdx_gd, dzsdy_gd);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uu_gd, dudx_gd, dudy_gd);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> > (XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vv_gd, dvdx_gd, dvdy_gd);

		if (atmpuni == 0)
		{
			gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> > (XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, Patm_gd, dPdx_gd, dPdy_gd);
		}

		// Check the wind forcing at the same time here

		if (!XParam.windU.inputfile.empty())
		{

			Windthisstep(XParam, gridDimWND, blockDimWND, streams[2], windstep, uwinduni, vwinduni);
		}



		CUDA_CHECK(cudaDeviceSynchronize());

		//CUDA_CHECK(cudaStreamSynchronize(streams[0]));

		if (atmpuni == 1)
		{
			updateKurgXD << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, leftblk_g, hh_gd, zs_gd, uu_gd, vv_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);
			//CUDA_CHECK(cudaDeviceSynchronize());

			//CUDA_CHECK(cudaStreamSynchronize(streams[1]));
			updateKurgYD << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, botblk_g, hh_gd, zs_gd, uu_gd, vv_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);
		}
		else
		{
			updateKurgXATM << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, XParam.Pa2m, leftblk_g, hh_gd, zs_gd, uu_gd, vv_gd, Patm_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, dPdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);
			//CUDA_CHECK(cudaDeviceSynchronize());

			//CUDA_CHECK(cudaStreamSynchronize(streams[1]));
			updateKurgYATM << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, XParam.Pa2m, botblk_g, hh_gd, zs_gd, uu_gd, vv_gd, Patm_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, dPdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);


		}
		CUDA_CHECK(cudaDeviceSynchronize());



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

		//32 seem safe here bu I wonder why it is not 1 for the largers arrays...
		/*
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


		if (winduniform == 1)
		{
			// simpler input if wind is uniform
			updateEVATMWUNI << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.lat*pi / 21600.0, uwinduni, vwinduni, XParam.Cd, rightblk_g, topblk_g, hh_gd, uu_gd, vv_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);

		}
		else
		{
			//updateEV << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, rightblk_g, topblk_g, hh_g, uu_g, vv_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
			updateEVATM << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.lat*pi / 21600.0, XParam.windU.xo, XParam.windU.yo, XParam.windU.dx, XParam.Cd, rightblk_g, topblk_g, blockxo_gd, blockyo_gd, hh_gd, uu_gd, vv_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);

		}
		CUDA_CHECK(cudaDeviceSynchronize());




		//predictor (advance 1/2 dt)
		Advkernel << <gridDim, blockDim, 0 >> >(XParam.dt*0.5, XParam.eps, hh_gd, zb_gd, uu_gd, vv_gd, dh_gd, dhu_gd, dhv_gd, zso_gd, hho_gd, uuo_gd, vvo_gd);
		CUDA_CHECK(cudaDeviceSynchronize());

		//corrector setp
		//update again


		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hho_gd, dhdx_gd, dhdy_gd);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zso_gd, dzsdx_gd, dzsdy_gd);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uuo_gd, dudx_gd, dudy_gd);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vvo_gd, dvdx_gd, dvdy_gd);


		// No need to recalculate the gradient at this stage. (I'm not sure of that... we could reinterpolate the Patm 0.5dt foreward in time but that seems unecessary)

		CUDA_CHECK(cudaDeviceSynchronize());

		if (atmpuni == 1)
		{

			updateKurgXD << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, leftblk_g, hho_gd, zso_gd, uuo_gd, vvo_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);
			//CUDA_CHECK(cudaDeviceSynchronize());


			updateKurgYD << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, botblk_g, hho_gd, zso_gd, uuo_gd, vvo_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);
		}
		else
		{
			updateKurgXATM << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, XParam.Pa2m, leftblk_g, hho_gd, zso_gd, uuo_gd, vvo_gd, Patm_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, dPdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);
			//CUDA_CHECK(cudaDeviceSynchronize());


			updateKurgYATM << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, XParam.Pa2m, botblk_g, hho_gd, zso_gd, uuo_gd, vvo_gd, Patm_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, dPdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);
		}

		CUDA_CHECK(cudaDeviceSynchronize());

		// no reduction of dtmax during the corrector step

		if (winduniform == 1)
		{
			//
			updateEVATMWUNI << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.lat*pi / 21600.0, uwinduni, vwinduni, XParam.Cd, rightblk_g, topblk_g, hho_gd, uuo_gd, vvo_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);

		}
		else
		{
			//updateEV << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, rightblk_g, topblk_g, hho_g, uuo_g, vvo_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
			updateEVATM << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.lat*pi / 21600.0, XParam.windU.xo, XParam.windU.yo, XParam.windU.dx, XParam.Cd, rightblk_g, topblk_g, blockxo_gd, blockyo_gd, hho_gd, uuo_gd, vvo_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);
		}
		CUDA_CHECK(cudaDeviceSynchronize());



		//
		Advkernel << <gridDim, blockDim, 0 >> >(XParam.dt, XParam.eps, hh_gd, zb_gd, uu_gd, vv_gd, dh_gd, dhu_gd, dhv_gd, zso_gd, hho_gd, uuo_gd, vvo_gd);
		CUDA_CHECK(cudaDeviceSynchronize());

		//cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
		cleanupGPU << <gridDim, blockDim, 0 >> >(hho_gd, zso_gd, uuo_gd, vvo_gd, hh_gd, zs_gd, uu_gd, vv_gd);
		CUDA_CHECK(cudaDeviceSynchronize());

		//Bottom friction
		bottomfriction << <gridDim, blockDim, 0 >> > (XParam.frictionmodel, XParam.dt, XParam.eps, cf_gd, hh_gd, uu_gd, vv_gd);
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaStreamDestroy(streams[0]));
		CUDA_CHECK(cudaStreamDestroy(streams[1]));

		// Impose no slip condition by default
		//noslipbndall << <gridDim, blockDim, 0 >> > (nx, ny, XParam.dt, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
		//CUDA_CHECK(cudaDeviceSynchronize());

		// River
		if (XParam.Rivers.size() > 0)
		{
			RiverSourceD(XParam);
		}

		// add rain
		if (!XParam.Rainongrid.inputfile.empty())
		{
			if (XParam.Rainongrid.uniform == 1)
			{
				Rain_on_gridUNI << <gridDim, blockDim, 0 >> > (XParam.mask, rainuni, XParam.dt, zs_gd, hh_gd);
				CUDA_CHECK(cudaDeviceSynchronize());
			}
			else
			{
				//(int unirain, float xorain, float yorain, float dxrain, double delta, double*blockxo, double *blockyo, double dt,  T * zs, T *hh)
				Rain_on_grid << <gridDim, blockDim, 0 >> > (XParam.mask, XParam.Rainongrid.xo, XParam.Rainongrid.yo, XParam.Rainongrid.dx, XParam.delta, blockxo_gd, blockyo_gd, XParam.dt, zs_gd, hh_gd);
				CUDA_CHECK(cudaDeviceSynchronize());
			}
		}


		//Time keeping
		XParam.totaltime = XParam.totaltime + XParam.dt;
		nstep++;

		// Do Sum & Max variables Here
		meanmaxvarGPUD(XParam);




		//Check for TSoutput
		if (XParam.TSnodesout.size() > 0)
		{
			pointoutputstep(XParam, gridDim, blockDim, nTSsteps, zsAllout);

		}

		if (nextoutputtime - XParam.totaltime <= XParam.dt*0.00001f  && XParam.outputtimestep > 0.0)
		{
			// Avg var sum here
			DivmeanvarGPUD(XParam, nstep*1.0f);

			if (XParam.outvort == 1)
			{
				CalcVorticity << <gridDim, blockDim, 0 >> > (vort_gd, dvdx_gd, dudy_gd);
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
						//writencvarstep(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
						writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPUD[XParam.outvars[ivar]]);
					}
				}
			}

			nextoutputtime = min(nextoutputtime + XParam.outputtimestep, XParam.endtime);

			printf("Writing output, totaltime:%f s, Mean dt=%f\n", XParam.totaltime, XParam.outputtimestep / nstep);
			write_text_to_log_file("Writing outputs, totaltime: " + std::to_string(XParam.totaltime) + ", Mean dt= " + std::to_string(XParam.outputtimestep / nstep));

			//.Reset Avg Variables
			ResetmeanvarGPUD(XParam);
			if (XParam.resetmax == 1)
			{
				ResetmaxvarGPUD(XParam);
			}




			//

			// Reset nstep
			nstep = 0;
		} // End of output part

	} //Main while loop
}

void mainloopGPUDSPH(Param XParam)// double precision and spherical coordinate system
{
	double nextoutputtime = XParam.totaltime + XParam.outputtimestep;
	int nstep = 0;

	int rainstep = 1;

	double rainuni = 0.0;

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
	ResetmeanvarGPUD(XParam);
	ResetmaxvarGPUD(XParam);
	while (XParam.totaltime < XParam.endtime)
	{
		// Bnd stuff here
		LeftFlowBnd(XParam);
		RightFlowBnd(XParam);
		TopFlowBnd(XParam);
		BotFlowBnd(XParam);




		// Core
		XParam.dt = FlowGPUSpherical(XParam, nextoutputtime);

		//Time keeping
		XParam.totaltime = XParam.totaltime + XParam.dt;
		nstep++;

		// Add deformation?
		if (XParam.deform.size() > 0 && (XParam.totaltime - XParam.dt ) <= XParam.deformmaxtime)
		{
			ApplyDeform(XParam,blockDim,gridDim,dummy_d, dh_gd,hh_gd, zs_gd, zb_gd );
			// float * def;
			// double *def_d;
			// //Check each deform input
			// for (int nd = 0; nd < XParam.deform.size(); nd++)
			// {
			// 	Allocate1CPU(XParam.deform[nd].grid.nx, XParam.deform[nd].grid.ny, def_d);
			// 	Allocate1CPU(XParam.deform[nd].grid.nx, XParam.deform[nd].grid.ny, def);
			// 	if ((XParam.totaltime - XParam.deform[nd].startime) <= XParam.dt && (XParam.totaltime - XParam.deform[nd].startime)>0.0)
			// 	{
			// 		readmapdata(XParam.deform[nd].grid, def);
			//
			// 		for (int k = 0; k<(XParam.deform[nd].grid.nx*XParam.deform[nd].grid.ny); k++)
			// 		{
			// 			def_d[k] = def[k];
			// 		}
			//
			//
			// 		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.deform[nd].grid.nx, XParam.deform[nd].grid.ny, XParam.deform[nd].grid.xo, XParam.deform[nd].grid.xmax, XParam.deform[nd].grid.yo, XParam.deform[nd].grid.ymax, XParam.deform[nd].grid.dx, def_d, dummy_d);
			// 		CUDA_CHECK(cudaMemcpy(dh_gd, dummy_d, XParam.nblk*XParam.blksize * sizeof(double), cudaMemcpyHostToDevice));
			//
			// 		if (XParam.deform[nd].duration > 0.0)
			// 		{
			//
			// 			//do zs=zs+dummy/duration *(XParam.totaltime - XParam.deform[nd].startime);
			// 			Deform << <gridDim, blockDim, 0 >> > (1.0 / XParam.deform[nd].duration *(XParam.totaltime - XParam.deform[nd].startime), dh_gd, zs_gd, zb_gd);
			// 			CUDA_CHECK(cudaDeviceSynchronize());
			// 		}
			//
			// 		else
			// 		{
			// 			//do zs=zs+dummy;
			// 			Deform << <gridDim, blockDim, 0 >> > (1.0, dh_gd, zs_gd, zb_gd);
			// 			CUDA_CHECK(cudaDeviceSynchronize());
			// 		}
			//
			// 	}
			// 	else if ((XParam.totaltime - XParam.deform[nd].startime) > XParam.dt && XParam.totaltime <= (XParam.deform[nd].startime + XParam.deform[nd].duration))
			// 	{
			// 		// read the data and store to dummy
			// 		readmapdata(XParam.deform[nd].grid, def);
			// 		for (int k = 0; k<(XParam.deform[nd].grid.nx*XParam.deform[nd].grid.ny); k++)
			// 		{
			// 			def_d[k] = def[k];
			// 		}
			//
			//
			// 		interp2BUQ(XParam.nblk, XParam.blksize, XParam.dx, blockxo_d, blockyo_d, XParam.deform[nd].grid.nx, XParam.deform[nd].grid.ny, XParam.deform[nd].grid.xo, XParam.deform[nd].grid.xmax, XParam.deform[nd].grid.yo, XParam.deform[nd].grid.ymax, XParam.deform[nd].grid.dx, def_d, dummy_d);
			// 		CUDA_CHECK(cudaMemcpy(dh_gd, dummy_d, XParam.nblk*XParam.blksize * sizeof(double), cudaMemcpyHostToDevice));
			//
			// 		// DO zs=zs+dummy/duration*dt
			// 		Deform << <gridDim, blockDim, 0 >> > (1.0 / XParam.deform[nd].duration *XParam.dt, dh_gd, zs_gd, zb_gd);
			// 		CUDA_CHECK(cudaDeviceSynchronize());
			//
			//
			// 	}
			// 	free(def_d);
			//
			// }


		}

		// add rain ?
		if (!XParam.Rainongrid.inputfile.empty())
		{
			if (XParam.Rainongrid.uniform == 1)
			{
				Rain_on_gridUNI << <gridDim, blockDim, 0 >> > (XParam.mask, rainuni, XParam.dt, zs_gd, hh_gd);
				CUDA_CHECK(cudaDeviceSynchronize());
			}
			else
			{
				//(int unirain, float xorain, float yorain, float dxrain, double delta, double*blockxo, double *blockyo, double dt,  T * zs, T *hh)
				Rain_on_grid << <gridDim, blockDim, 0 >> > (XParam.mask, XParam.Rainongrid.xo, XParam.Rainongrid.yo, XParam.Rainongrid.dx, XParam.delta, blockxo_gd, blockyo_gd, XParam.dt, zs_gd, hh_gd);
				CUDA_CHECK(cudaDeviceSynchronize());
			}
		}

		// Do Sum & Max variables Here
		meanmaxvarGPUD(XParam);

		//check, store Timeseries output
		if (XParam.TSnodesout.size() > 0)
		{

			pointoutputstep(XParam, gridDim, blockDim, nTSsteps, zsAllout);
		}

		if (nextoutputtime - XParam.totaltime <= XParam.dt*0.00001f  && XParam.outputtimestep > 0.0)
		{
			// Save output step
			DivmeanvarGPUD(XParam, nstep);

			if (XParam.outvort == 1)
			{
				CalcVorticity << <gridDim, blockDim, 0 >> > (vort_gd, dvdx_gd, dudy_gd);
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
						//writencvarstep(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPUD[XParam.outvars[ivar]]);
						writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPUD[XParam.outvars[ivar]]);
					}
				}
			}

			// Log
			nextoutputtime = min(nextoutputtime + XParam.outputtimestep, XParam.endtime);

			printf("Writing output, totaltime:%f s, Mean dt=%f\n", XParam.totaltime, XParam.outputtimestep / nstep);
			write_text_to_log_file("Writing outputs, totaltime: " + std::to_string(XParam.totaltime) + ", Mean dt= " + std::to_string(XParam.outputtimestep / nstep));

			//Reset Avg Variables
			ResetmeanvarGPUD(XParam);
			if (XParam.resetmax == 1)
			{
				ResetmaxvarGPUD(XParam);
			}

			// Reset nstep
			nstep = 0;
		}
	}
}

void mainloopGPUDSPHATM(Param XParam)// double precision and spherical coordinate system
{
	double nextoutputtime = XParam.totaltime + XParam.outputtimestep;
	int nstep = 0;

	int rainstep = 1;

	double rainuni = 0.0;

	int windstep = 1;
	int atmpstep = 1;

	double uwinduni = 0.0f;
	double vwinduni = 0.0f;
	double atmpuni = XParam.Paref;


	int nTSsteps = 0;

	std::vector<Pointout> zsout;

	std::vector< std::vector< Pointout > > zsAllout;

	Pointout stepread;

	FILE * fsSLTS;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	dim3 blockDimWND(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDimWND((int)ceil(XParam.windU.nx / blockDimWND.x), (int)ceil(XParam.windU.ny / blockDimWND.y), 1);

	dim3 blockDimATM(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
	//dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDimATM((int)ceil(XParam.atmP.nx / blockDimATM.x), (int)ceil(XParam.atmP.ny / blockDimATM.y), 1);

	int winduniform = XParam.windU.uniform;
	int atmpuniform = XParam.atmP.uniform;

	if (XParam.windU.inputfile.empty())// this is should be true here so not really needed (?)
	{
		// set as uniform run 0 wind input below
		winduniform = 1;
	}
	if (XParam.atmP.inputfile.empty())// this is should be true here so not really needed (?)
	{
		atmpuniform = 1;
	}

	for (int o = 0; o < XParam.TSoutfile.size(); o++)
	{
		//Overwrite existing files
		fsSLTS = fopen(XParam.TSoutfile[o].c_str(), "w");
		fprintf(fsSLTS, "# x=%f\ty=%f\ti=%d\tj=%d\t%s\n", XParam.TSnodesout[o].x, XParam.TSnodesout[o].y, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSoutfile[o].c_str());
		fclose(fsSLTS);

		// Add empty row for each output point
		zsAllout.push_back(std::vector<Pointout>());
	}
	ResetmeanvarGPUD(XParam);
	ResetmaxvarGPUD(XParam);
	while (XParam.totaltime < XParam.endtime)
	{
		// Bnd stuff here
		LeftFlowBnd(XParam);
		RightFlowBnd(XParam);
		TopFlowBnd(XParam);
		BotFlowBnd(XParam);

		// Core
		//XParam.dt = FlowGPUSpherical(XParam, nextoutputtime);

		const int num_streams = 3;

		cudaStream_t streams[num_streams];
		for (int i = 0; i < num_streams; i++)
		{
			CUDA_CHECK(cudaStreamCreate(&streams[i]));
		}

		//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
		//dim3 gridDim(ceil((nx*1.0) / blockDim.x), ceil((ny*1.0) / blockDim.y), 1);
		dim3 blockDim(16, 16, 1);
		dim3 gridDim(XParam.nblk, 1, 1);


		if (!XParam.atmP.inputfile.empty() && atmpuniform == 1) //this is a gven
		{

			//
			AtmPthisstep(XParam, gridDimATM, blockDimATM, atmpstep);

			interp2ATMP << <gridDim, blockDim, 0 >> > ((float)XParam.atmP.xo, (float)XParam.atmP.yo, (float)XParam.atmP.dx, (float)XParam.delta, (float)XParam.Paref, blockxo_g, blockyo_g, Patm_g);
			CUDA_CHECK(cudaDeviceSynchronize());


		}

		dtmax = (float)(1.0 / epsilon);
		//float dtmaxtmp = dtmax;

		resetdtmax << <gridDim, blockDim, 0, streams[0] >> > (dtmax_gd);
		//CUDA_CHECK(cudaDeviceSynchronize());
		//update step 1




		gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hh_gd, dhdx_gd, dhdy_gd);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zs_gd, dzsdx_gd, dzsdy_gd);
		//CUDA_CHECK(cudaDeviceSynchronize());




		gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uu_gd, dudx_gd, dudy_gd);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[1] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vv_gd, dvdx_gd, dvdy_gd);



		if (atmpuni == 0)
		{
			gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> > (XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, Patm_gd, dPdx_gd, dPdy_gd);
		}

		// Check the wind forcing at the same time here

		if (!XParam.windU.inputfile.empty())
		{

			Windthisstep(XParam, gridDimWND, blockDimWND, streams[2], windstep, uwinduni, vwinduni);
		}


		CUDA_CHECK(cudaDeviceSynchronize());

		//CUDA_CHECK(cudaStreamSynchronize(streams[0]));
		if (atmpuni == 1)
		{
			//Spherical coordinates
			updateKurgXSPH << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, leftblk_g, blockyo_gd, XParam.Radius, hh_gd, zs_gd, uu_gd, vv_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);

			updateKurgYSPH << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, botblk_g, blockyo_gd, XParam.Radius, hh_gd, zs_gd, uu_gd, vv_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);



		}
		else
		{
			updateKurgXSPHATM << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, XParam.Pa2m, leftblk_g, blockyo_gd, XParam.Radius, hh_gd, zs_gd, uu_gd, vv_gd, Patm_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, dPdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);

			updateKurgYSPHATM << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, XParam.Pa2m, botblk_g, blockyo_gd, XParam.Radius, hh_gd, zs_gd, uu_gd, vv_gd, Patm_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, dPdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);


		}
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
		if (winduniform == 1)
		{
			//if spherical corrdinate use this kernel with the right corrections
			updateEVSPHATMUNI << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.yo, XParam.ymax, XParam.Radius, uwinduni, vwinduni, XParam.Cd, rightblk_g, topblk_g, blockyo_gd, hh_gd, uu_gd, vv_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);

		}
		else
		{
			updateEVSPHATM << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.yo, XParam.ymax, XParam.Radius, XParam.windU.xo, XParam.windU.yo, XParam.windU.dx, XParam.Cd, rightblk_g, topblk_g, blockxo_gd, blockyo_gd, hh_gd, uu_gd, vv_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);


		}
		CUDA_CHECK(cudaDeviceSynchronize());

	//predictor (advance 1/2 dt)
	Advkernel << <gridDim, blockDim, 0 >> >(XParam.dt*0.5, XParam.eps, hh_gd, zb_gd, uu_gd, vv_gd, dh_gd, dhu_gd, dhv_gd, zso_gd, hho_gd, uuo_gd, vvo_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//corrector setp
	//update again
	// calculate gradients
	//gradientGPUX << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hho_g, dhdx_g);
	//gradientGPUY << <gridDim, blockDim, 0 >> >(nx, ny, XParam.theta, XParam.delta, hho_g, dhdy_g);

	gradientGPUXYBUQ << <gridDim, blockDim, 0, streams[0] >> >(XParam.theta, XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hho_gd, dhdx_gd, dhdy_gd);
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


	if (atmpuni == 1)
	{
		//Spherical coordinates
		updateKurgXSPH << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, leftblk_g, blockyo_gd, XParam.Radius, hho_gd, zso_gd, uuo_gd, vvo_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);

		updateKurgYSPH << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, botblk_g, blockyo_gd, XParam.Radius, hho_gd, zso_gd, uuo_gd, vvo_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);



	}
	else
	{
		updateKurgXSPHATM << <gridDim, blockDim, 0, streams[0] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, XParam.Pa2m, leftblk_g, blockyo_gd, XParam.Radius, hho_gd, zso_gd, uuo_gd, vvo_gd, Patm_gd, dzsdx_gd, dhdx_gd, dudx_gd, dvdx_gd, dPdx_gd, Fhu_gd, Fqux_gd, Fqvx_gd, Su_gd, dtmax_gd);

		updateKurgYSPHATM << <gridDim, blockDim, 0, streams[1] >> > (XParam.delta, XParam.g, XParam.eps, XParam.CFL, XParam.Pa2m, botblk_g, blockyo_gd, XParam.Radius, hho_gd, zso_gd, uuo_gd, vvo_gd, Patm_gd, dzsdy_gd, dhdy_gd, dudy_gd, dvdy_gd, dPdy_gd, Fhv_gd, Fqvy_gd, Fquy_gd, Sv_gd, dtmax_gd);


	}
	CUDA_CHECK(cudaDeviceSynchronize());
	// no reduction of dtmax during the corrector step


	//spherical
	if (winduniform == 1)
	{
		//if spherical corrdinate use this kernel with the right corrections
		updateEVSPHATMUNI << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.yo, XParam.ymax, XParam.Radius, uwinduni, vwinduni, XParam.Cd, rightblk_g, topblk_g, blockyo_gd, hho_gd, uuo_gd, vvo_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);

	}
	else
	{
		updateEVSPHATM << <gridDim, blockDim, 0 >> > (XParam.delta, XParam.g, XParam.yo, XParam.ymax, XParam.Radius, XParam.windU.xo, XParam.windU.yo, XParam.windU.dx, XParam.Cd, rightblk_g, topblk_g, blockxo_gd, blockyo_gd, hho_gd, uuo_gd, vvo_gd, Fhu_gd, Fhv_gd, Su_gd, Sv_gd, Fqux_gd, Fquy_gd, Fqvx_gd, Fqvy_gd, dh_gd, dhu_gd, dhv_gd);


	}
	CUDA_CHECK(cudaDeviceSynchronize());

	//
	Advkernel << <gridDim, blockDim, 0 >> >(XParam.dt, XParam.eps, hh_gd, zb_gd, uu_gd, vv_gd, dh_gd, dhu_gd, dhv_gd, zso_gd, hho_gd, uuo_gd, vvo_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
	cleanupGPU << <gridDim, blockDim, 0 >> >(hho_gd, zso_gd, uuo_gd, vvo_gd, hh_gd, zs_gd, uu_gd, vv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	//Bottom friction
	bottomfriction << <gridDim, blockDim, 0 >> > (XParam.frictionmodel, XParam.dt, XParam.eps, cf_gd, hh_gd, uu_gd, vv_gd);
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaStreamDestroy(streams[0]));
	CUDA_CHECK(cudaStreamDestroy(streams[1]));


		//Time keeping
		XParam.totaltime = XParam.totaltime + XParam.dt;
		nstep++;


		// add rain ?
		if (!XParam.Rainongrid.inputfile.empty())
		{
			if (XParam.Rainongrid.uniform == 1)
			{
				Rain_on_gridUNI << <gridDim, blockDim, 0 >> > (XParam.mask, rainuni, XParam.dt, zs_gd, hh_gd);
				CUDA_CHECK(cudaDeviceSynchronize());
			}
			else
			{
				//(int unirain, float xorain, float yorain, float dxrain, double delta, double*blockxo, double *blockyo, double dt,  T * zs, T *hh)
				Rain_on_grid << <gridDim, blockDim, 0 >> > (XParam.mask, XParam.Rainongrid.xo, XParam.Rainongrid.yo, XParam.Rainongrid.dx, XParam.delta, blockxo_gd, blockyo_gd, XParam.dt, zs_gd, hh_gd);
				CUDA_CHECK(cudaDeviceSynchronize());
			}
		}

		// Do Sum & Max variables Here
		meanmaxvarGPUD(XParam);

		//check, store Timeseries output
		if (XParam.TSnodesout.size() > 0)
		{

			pointoutputstep(XParam, gridDim, blockDim, nTSsteps, zsAllout);
		}

		if (nextoutputtime - XParam.totaltime <= XParam.dt*0.00001f  && XParam.outputtimestep > 0.0)
		{
			// Save output step
			DivmeanvarGPUD(XParam, nstep);

			if (XParam.outvort == 1)
			{
				CalcVorticity << <gridDim, blockDim, 0 >> > (vort_gd, dvdx_gd, dudy_gd);
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
						//writencvarstep(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPUD[XParam.outvars[ivar]]);
						writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPUD[XParam.outvars[ivar]]);
					}
				}
			}

			// Log
			nextoutputtime = min(nextoutputtime + XParam.outputtimestep, XParam.endtime);

			printf("Writing output, totaltime:%f s, Mean dt=%f\n", XParam.totaltime, XParam.outputtimestep / nstep);
			write_text_to_log_file("Writing outputs, totaltime: " + std::to_string(XParam.totaltime) + ", Mean dt= " + std::to_string(XParam.outputtimestep / nstep));

			//Reset Avg Variables
			ResetmeanvarGPUD(XParam);
			if (XParam.resetmax == 1)
			{
				ResetmaxvarGPUD(XParam);
			}

			// Reset nstep
			nstep = 0;
		}
	}
}

void mainloopGPU(Param XParam) // float, metric coordinate
{
	double nextoutputtime = XParam.totaltime + XParam.outputtimestep;
	int nstep = 0;
	int nTSsteps = 0;

	int windstep = 1;
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
		fprintf(fsSLTS, "# x=%f\ty=%f\tblk=%d\ti=%d\tj=%d\t%s\n", XParam.TSnodesout[o].x, XParam.TSnodesout[o].y, XParam.TSnodesout[o].block, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSoutfile[o].c_str());
		fclose(fsSLTS);

		// Add empty row for each output point
		zsAllout.push_back(std::vector<Pointout>());
	}
	// Reset GPU mean and max arrays
	ResetmeanvarGPU(XParam);
	ResetmaxvarGPU(XParam);


	while (XParam.totaltime < XParam.endtime)
	{
		// Bnd stuff here
		LeftFlowBnd(XParam);
		RightFlowBnd(XParam);
		TopFlowBnd(XParam);
		BotFlowBnd(XParam);



		// Core engine
		XParam.dt = FlowGPU(XParam, nextoutputtime);

		// River
		if (XParam.Rivers.size() > 0)
		{
			RiverSource(XParam);
		}

		//Time keeping
		XParam.totaltime = XParam.totaltime + XParam.dt;
		nstep++;

		if (XParam.deform.size() > 0 && (XParam.totaltime - XParam.dt ) <= XParam.deformmaxtime)
		{
			ApplyDeform(XParam,blockDim,gridDim, dummy, dh_g, hh_g, zs_g, zb_g );
		}

		// Do Sum & Max variables Here
		meanmaxvarGPU(XParam);




		//Check for TSoutput
		if (XParam.TSnodesout.size() > 0)
		{
			pointoutputstep(XParam, gridDim, blockDim, nTSsteps, zsAllout);


		}

		if (nextoutputtime - XParam.totaltime <= XParam.dt*0.00001f  && XParam.outputtimestep > 0.0)
		{
			// Avg var sum here
			DivmeanvarGPU(XParam, nstep*1.0f);

			if (XParam.outvort == 1)
			{
				CalcVorticity << <gridDim, blockDim, 0 >> > (vort_g, dvdx_g, dudy_g);
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
						//writencvarstep(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
						writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
					}
				}
			}
			/*
			CUDA_CHECK(cudaMemcpy(Fhu, Fhu_g, XParam.nblk * XParam.blksize * sizeof(float), cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaMemcpy(Fhv, Fhv_g, XParam.nblk * XParam.blksize * sizeof(float), cudaMemcpyDeviceToHost));

			CUDA_CHECK(cudaMemcpy(dhdx, dhdx_g, XParam.nblk * XParam.blksize * sizeof(float), cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaMemcpy(dhdy, dhdy_g, XParam.nblk * XParam.blksize * sizeof(float), cudaMemcpyDeviceToHost));

			CUDA_CHECK(cudaMemcpy(dzsdx, dzsdx_g, XParam.nblk * XParam.blksize * sizeof(float), cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaMemcpy(dzsdy, dzsdy_g, XParam.nblk * XParam.blksize * sizeof(float), cudaMemcpyDeviceToHost));

			writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, "dhdx", dhdx);
			writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, "dhdy", dhdy);

			writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, "dzsdx", dzsdx);
			writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, "dzsdy", dzsdy);


			writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, "Fhu", Fhu);
			writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, "Fhv", Fhv);
			*/

			nextoutputtime = min(nextoutputtime + XParam.outputtimestep, XParam.endtime);

			printf("Writing output, totaltime:%f s, Mean dt=%f\n", XParam.totaltime, XParam.outputtimestep / nstep);
			write_text_to_log_file("Writing outputs, totaltime: " + std::to_string(XParam.totaltime) + ", Mean dt= " + std::to_string(XParam.outputtimestep / nstep));

			//.Reset Avg Variables
			ResetmeanvarGPU(XParam);
			if (XParam.resetmax == 1)
			{
				ResetmaxvarGPU(XParam);
			}




			//

			// Reset nstep
			nstep = 0;
		} // End of output part

	} //Main while loop
}


void mainloopGPUATM(Param XParam) // float, metric coordinate
{
	double nextoutputtime = XParam.totaltime + XParam.outputtimestep;
	int nstep = 0;
	int nTSsteps = 0;

	int windstep = 1;
	int atmpstep = 1;

	float uwinduni = 0.0f;
	float vwinduni = 0.0f;
	float atmpuni = XParam.Paref;
	std::vector<Pointout> zsout;

	std::vector< std::vector< Pointout > > zsAllout;

	Pointout stepread;

	FILE * fsSLTS;

	dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
							 //dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDim(XParam.nblk, 1, 1);


	dim3 blockDimWND(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
							 //dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDimWND((int)ceil((float)XParam.windU.nx / (float)blockDimWND.x), (int)ceil((float)XParam.windU.ny / (float)blockDimWND.y), 1);

	dim3 blockDimATM(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
								//dim3 gridDim(ceil((XParam.nx*1.0) / blockDim.x), ceil((XParam.ny*1.0) / blockDim.y), 1);
	dim3 gridDimATM((int)ceil((float)XParam.atmP.nx / (float)blockDimATM.x), (int)ceil((float)XParam.atmP.ny / (float)blockDimATM.y), 1);


	int winduniform = XParam.windU.uniform;
	int atmpuniform = XParam.atmP.uniform;

	if (XParam.windU.inputfile.empty())// this is should be true here so not really needed (?)
	{
		// set as uniform run 0 wind input below
		winduniform = 1;
	}
	if (XParam.atmP.inputfile.empty())// this is should be true here so not really needed (?)
	{
		atmpuniform = 1;
	}


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
	ResetmeanvarGPU(XParam);
	ResetmaxvarGPU(XParam);


	//Prep model
	//dim3 blockDim(16, 16, 1);
	//dim3 gridDim(XParam.nblk, 1, 1);


	const int num_streams = 4;

	cudaStream_t streams[num_streams];


	while (XParam.totaltime < XParam.endtime)
	{
		// Bnd stuff here

		LeftFlowBnd(XParam);
		RightFlowBnd(XParam);
		TopFlowBnd(XParam);
		BotFlowBnd(XParam);

		for (int i = 0; i < num_streams; i++)
		{
			CUDA_CHECK(cudaStreamCreate(&streams[i]));
		}


		// Core engine

		//dim3 blockDim(16, 16, 1);
		//dim3 gridDim(XParam.nblk, 1, 1);



		//XParam.dt = FlowGPUATM(XParam, nextoutputtime);


		//dim3 blockDim(16, 16, 1);// The grid has a better ocupancy when the size is a factor of 16 on both x and y
		//dim3 gridDim(ceil((nx*1.0f) / blockDim.x), ceil((ny*1.0f) / blockDim.y), 1);


		dtmax = (float)(1.0 / epsilon);
		//float dtmaxtmp = dtmax;



		// Check the atm Pressure forcing before starting

		if (!XParam.atmP.inputfile.empty() && atmpuniform == 0) //this is a gven
		{

			//
			AtmPthisstep(XParam, gridDimATM, blockDimATM, atmpstep);

			interp2ATMP << <gridDim, blockDim, 0 >> > ((float)XParam.atmP.xo, (float)XParam.atmP.yo, (float)XParam.atmP.dx, (float)XParam.delta, (float)XParam.Paref, blockxo_g, blockyo_g, Patm_g);
			CUDA_CHECK(cudaDeviceSynchronize());


		}



		resetdtmax << <gridDim, blockDim, 0, streams[0] >> > (dtmax_g);
		//CUDA_CHECK(cudaDeviceSynchronize());
		//update step 1



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hh_g, dhdx_g, dhdy_g);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[2] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zs_g, dzsdx_g, dzsdy_g);
		CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uu_g, dudx_g, dudy_g);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> > ((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vv_g, dvdx_g, dvdy_g);

		if (atmpuniform == 0)
		{
			gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[2] >> > ((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, Patm_g, dPdx_g, dPdy_g);
		}

		// Check the wind forcing at the same time here

		if (!XParam.windU.inputfile.empty())
		{

			Windthisstep(XParam, gridDimWND, blockDimWND, streams[0], windstep, uwinduni, vwinduni);
		}



		CUDA_CHECK(cudaDeviceSynchronize());

		//CUDA_CHECK(cudaStreamSynchronize(streams[0]));

		if (atmpuniform == 1)
		{
			updateKurgX << <gridDim, blockDim, 0, streams[0] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, leftblk_g, hh_g, zs_g, uu_g, vv_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
			//CUDA_CHECK(cudaDeviceSynchronize());

			//CUDA_CHECK(cudaStreamSynchronize(streams[1]));
			updateKurgY << <gridDim, blockDim, 0, streams[1] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, botblk_g, hh_g, zs_g, uu_g, vv_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);
		}
		else
		{
			updateKurgXATM << <gridDim, blockDim, 0, streams[0] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, (float)XParam.Pa2m, leftblk_g, hh_g, zs_g, uu_g, vv_g, Patm_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, dPdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
			//CUDA_CHECK(cudaDeviceSynchronize());

			//CUDA_CHECK(cudaStreamSynchronize(streams[1]));
			updateKurgYATM << <gridDim, blockDim, 0, streams[1] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, (float)XParam.Pa2m, botblk_g, hh_g, zs_g, uu_g, vv_g, Patm_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, dPdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);


		}
		CUDA_CHECK(cudaDeviceSynchronize());


		/*
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

		reducemin3 << <gridDimLine, blockDimLine, smemSize >> > (dtmax_g, arrmax_g, s);
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


		CUDA_CHECK(cudaMemcpy(dummy, arrmax_g, 32 * sizeof(float), cudaMemcpyDeviceToHost));
		mindtmaxB = dummy[0];

		//32 seem safe here bu I wonder why it is not 1 for the largers arrays...

		//for (int i = 0; i < 32; i++)
		//{
		//mindtmaxB = min(dummy[i], mindtmaxB);
		//printf("dt=%f\n", dummy[i]);

		//}



		//float diffdt = mindtmaxB - mindtmax;
		XParam.dt = mindtmaxB;
		*/
		XParam.dt = Calcmaxdt(XParam, dtmax_g, arrmax_g);

		if (ceil((nextoutputtime - XParam.totaltime) / XParam.dt)> 0.0)
		{
			XParam.dt = (nextoutputtime - XParam.totaltime) / ceil((nextoutputtime - XParam.totaltime) / XParam.dt);
		}
		//printf("dt=%f\n", XParam.dt);


		if (winduniform == 1)
		{
			// simpler input if wind is uniform
			updateEVATMWUNI << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, (float)(XParam.lat*pi / 21600.0f), uwinduni, vwinduni, (float)XParam.Cd, rightblk_g, topblk_g, hh_g, uu_g, vv_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);

		}
		else
		{
			//updateEV << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, rightblk_g, topblk_g, hh_g, uu_g, vv_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
			updateEVATM << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, (float)(XParam.lat*pi/21600.0f), (float)XParam.windU.xo, (float)XParam.windU.yo, (float)XParam.windU.dx, (float)XParam.Cd, rightblk_g, topblk_g, blockxo_g, blockyo_g, hh_g, uu_g, vv_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);

		}
		CUDA_CHECK(cudaDeviceSynchronize());




		//predictor (advance 1/2 dt)
		Advkernel << <gridDim, blockDim, 0 >> >((float)XParam.dt*0.5f, (float)XParam.eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
		CUDA_CHECK(cudaDeviceSynchronize());

		//corrector setp
		//update again


		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, hho_g, dhdx_g, dhdy_g);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, zso_g, dzsdx_g, dzsdy_g);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[0] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, uuo_g, dudx_g, dudy_g);
		//CUDA_CHECK(cudaDeviceSynchronize());



		gradientGPUXYBUQSM << <gridDim, blockDim, 0, streams[1] >> >((float)XParam.theta, (float)XParam.delta, leftblk_g, rightblk_g, topblk_g, botblk_g, vvo_g, dvdx_g, dvdy_g);


		// No need to recalculate the gradient at this stage. (I'm not sure of that... we could reinterpolate the Patm 0.5dt foreward in time but that seems unecessary)

		CUDA_CHECK(cudaDeviceSynchronize());

		if (atmpuniform == 1)
		{

			updateKurgX << <gridDim, blockDim, 0, streams[0] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, leftblk_g, hho_g, zso_g, uuo_g, vvo_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
			//CUDA_CHECK(cudaDeviceSynchronize());


			updateKurgY << <gridDim, blockDim, 0, streams[1] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, botblk_g, hho_g, zso_g, uuo_g, vvo_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);
		}
		else
		{
			updateKurgXATM << <gridDim, blockDim, 0, streams[0] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, (float)XParam.Pa2m, leftblk_g, hho_g, zso_g, uuo_g, vvo_g, Patm_g, dzsdx_g, dhdx_g, dudx_g, dvdx_g, dPdx_g, Fhu_g, Fqux_g, Fqvx_g, Su_g, dtmax_g);
			//CUDA_CHECK(cudaDeviceSynchronize());


			updateKurgYATM << <gridDim, blockDim, 0, streams[1] >> > ((float)XParam.delta, (float)XParam.g, (float)XParam.eps, (float)XParam.CFL, (float)XParam.Pa2m, botblk_g, hho_g, zso_g, uuo_g, vvo_g, Patm_g, dzsdy_g, dhdy_g, dudy_g, dvdy_g, dPdy_g, Fhv_g, Fqvy_g, Fquy_g, Sv_g, dtmax_g);
		}

		CUDA_CHECK(cudaDeviceSynchronize());

		// no reduction of dtmax during the corrector step

		if (winduniform == 1)
		{
			//
			updateEVATMWUNI << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, (float)(XParam.lat*pi / 21600.0f), uwinduni, vwinduni, (float)XParam.Cd, rightblk_g, topblk_g, hho_g, uuo_g, vvo_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);

		}
		else
		{
			//updateEV << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, rightblk_g, topblk_g, hho_g, uuo_g, vvo_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
			updateEVATM << <gridDim, blockDim, 0 >> > ((float)XParam.delta, (float)XParam.g, (float)(XParam.lat*pi / 21600.0f), (float)XParam.windU.xo, (float)XParam.windU.yo, (float)XParam.windU.dx, (float)XParam.Cd, rightblk_g, topblk_g, blockxo_g, blockyo_g, hho_g, uuo_g, vvo_g, Fhu_g, Fhv_g, Su_g, Sv_g, Fqux_g, Fquy_g, Fqvx_g, Fqvy_g, dh_g, dhu_g, dhv_g);
		}
		CUDA_CHECK(cudaDeviceSynchronize());



		//
		Advkernel << <gridDim, blockDim, 0 >> >((float)XParam.dt, (float)XParam.eps, hh_g, zb_g, uu_g, vv_g, dh_g, dhu_g, dhv_g, zso_g, hho_g, uuo_g, vvo_g);
		CUDA_CHECK(cudaDeviceSynchronize());

		//cleanup(nx, ny, hho, zso, uuo, vvo, hh, zs, uu, vv);
		cleanupGPU << <gridDim, blockDim, 0 >> >(hho_g, zso_g, uuo_g, vvo_g, hh_g, zs_g, uu_g, vv_g);
		CUDA_CHECK(cudaDeviceSynchronize());



		//Bottom friction
		bottomfriction << <gridDim, blockDim, 0 >> > (XParam.frictionmodel, (float)XParam.dt, (float)XParam.eps, cf_g, hh_g, uu_g, vv_g);
		CUDA_CHECK(cudaDeviceSynchronize());


		//Destroy streams
		for (int i = 0; i < num_streams; i++)
		{
			CUDA_CHECK(cudaStreamDestroy(streams[i]));
		}
		//CUDA_CHECK(cudaStreamDestroy(streams[0]));
		//CUDA_CHECK(cudaStreamDestroy(streams[1]));

		// Impose no slip condition by default
		//noslipbndall << <gridDim, blockDim, 0 >> > (nx, ny, XParam.dt, XParam.eps, zb_g, zs_g, hh_g, uu_g, vv_g);
		//CUDA_CHECK(cudaDeviceSynchronize());




		// River
		if (XParam.Rivers.size() > 0)
		{
			RiverSource(XParam);
		}




		//Time keeping
		XParam.totaltime = XParam.totaltime + XParam.dt;
		nstep++;

		// Do Sum & Max variables Here
		meanmaxvarGPU(XParam);




		//Check for TSoutput
		if (XParam.TSnodesout.size() > 0)
		{
			pointoutputstep(XParam, gridDim, blockDim, nTSsteps,zsAllout);

		}

		if (nextoutputtime - XParam.totaltime <= XParam.dt*0.00001f  && XParam.outputtimestep > 0.0)
		{
			// Avg var sum here
			DivmeanvarGPU(XParam, nstep*1.0f);

			if (XParam.outvort == 1)
			{
				CalcVorticity << <gridDim, blockDim, 0 >> > (vort_g, dvdx_g, dudy_g);
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
						//writencvarstep(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
						writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
					}
				}
			}

			nextoutputtime = min(nextoutputtime + XParam.outputtimestep, XParam.endtime);

			printf("Writing output, totaltime:%f s, Mean dt=%f\n", XParam.totaltime, XParam.outputtimestep / nstep);
			write_text_to_log_file("Writing outputs, totaltime: " + std::to_string(XParam.totaltime) + ", Mean dt= " + std::to_string(XParam.outputtimestep / nstep));

			//.Reset Avg Variables
			ResetmeanvarGPU(XParam);
			if (XParam.resetmax == 1)
			{
				ResetmaxvarGPU(XParam);
			}




			//

			// Reset nstep
			nstep = 0;
		} // End of output part


	} //Main while loop
}


void mainloopGPUold(Param XParam)
{
	double nextoutputtime = XParam.totaltime + XParam.outputtimestep;
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
		LeftFlowBnd(XParam);
		RightFlowBnd(XParam);
		TopFlowBnd(XParam);
		BotFlowBnd(XParam);

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

		if (nextoutputtime - XParam.totaltime <= XParam.dt*0.00001f  && XParam.outputtimestep > 0.0)
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
							writencvarstep(XParam,blockxo_d,blockyo_d, XParam.outvars[ivar], OutputVarMapCPUD[XParam.outvars[ivar]]);
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
							//writencvarstep(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
							writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
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




void mainloopCPU(Param XParam)
{
	double nextoutputtime = XParam.totaltime + XParam.outputtimestep;
	int nstep = 0;

	int nTSstep = 0;

	int windstep = 1;
	int atmpstep = 1;
	float uwinduni = 0.0f;
	float vwinduni = 0.0f;

	int cstwind = 1;
	int cstpress = 1;

	std::vector<Pointout> zsout;

	std::vector< std::vector< Pointout > > zsAllout;

	Pointout stepread;

	FILE * fsSLTS;

	for (int o = 0; o < XParam.TSoutfile.size(); o++)
	{
		//Overwrite existing files
		fsSLTS = fopen(XParam.TSoutfile[o].c_str(), "w");
		fprintf(fsSLTS, "# x=%f\ty=%f\tblk=%d\ti=%d\tj=%d\t%s\n", XParam.TSnodesout[o].x, XParam.TSnodesout[o].y, XParam.TSnodesout[o].block, XParam.TSnodesout[o].i, XParam.TSnodesout[o].j, XParam.TSoutfile[o].c_str());
		fclose(fsSLTS);

		// Add empty row for each output point
		zsAllout.push_back(std::vector<Pointout>());
	}

#ifdef USE_CATALYST
        int catalystTimeStep = 0;
        if (XParam.use_catalyst)
        {
                // Retrieve adaptor object and add vtkUniformGrid patch for each 16x16 block
                catalystAdaptor& adaptor = catalystAdaptor::getInstance();
                for (int blockId = 0; blockId < XParam.nblk; blockId++)
                {
                        // Hardcoding 16x16 patch size - may need to be replaced with parameter
                        if (adaptor.addPatch(blockId, 0, 16, 16, XParam.dx, XParam.dx, blockxo[blockId], blockyo[blockId]))
                        {
                                fprintf(stderr, "catalystAdaptor::addPatch failed");
                        }
                }
        }
#endif

	while (XParam.totaltime < XParam.endtime)
	{
		// Bnd stuff here
		LeftFlowBnd(XParam);
		RightFlowBnd(XParam);
		TopFlowBnd(XParam);
		BotFlowBnd(XParam);

		if (!XParam.atmP.inputfile.empty())
		{
			if (XParam.atmP.uniform == 1)
			{
				//zeros
				for (int ib = 0; ib < XParam.nblk; ib++)
				{
					for (int iy = 0; iy < 16; iy++)
					{
						for (int ix = 0; ix < 16; ix++)
						{
							int i = ix + iy * 16 + ib * XParam.blksize;
							Patm[i] = 0.0;
							dPdx[i] = 0.0;
							dPdy[i] = 0.0;
						}
					}
				}

			}
			else
			{
				cstpress = 0;
				int readfirststep = min(max((int)floor((XParam.totaltime - XParam.atmP.to) / XParam.atmP.dt), 0), XParam.atmP.nt - 2);

				if (readfirststep + 1 > atmpstep)
				{
					// Need to read a new step from the file
					for (int iw = 0; iw < XParam.atmP.nx*XParam.atmP.ny; iw++)
					{
						//
						Patmbef[iw] = Patmaft[iw];


					}

					readATMstep(XParam.atmP, readfirststep + 1, Patmaft);
					atmpstep = readfirststep + 1;
				}
				InterpstepCPU(XParam.atmP.nx, XParam.atmP.ny, readfirststep, XParam.totaltime, XParam.atmP.dt, PatmX, Patmbef, Patmaft);

				for (int ib = 0; ib < XParam.nblk; ib++)
				{
					for (int iy = 0; iy < 16; iy++)
					{
						for (int ix = 0; ix < 16; ix++)
						{
							int i = ix + iy * 16 + ib * XParam.blksize;
							float x = blockxo[ib] + ix*XParam.delta;
							float y = blockyo[ib] + iy*XParam.delta;
							Patm[i] = interp2wnd((float)XParam.atmP.nx, (float)XParam.atmP.ny, (float)XParam.atmP.dx, (float)XParam.atmP.xo, (float)XParam.atmP.yo, x, y, PatmX)-XParam.Paref;
						}
					}
				}
				//float x = blockxo[ib] + ix*delta;
				//float y = blockyo[ib] + iy*delta;


				//float Uwndi = interp2wnd(windnx, windny, winddx, windxo, windyo, x, y, Uwnd);


			}




		}
		// Interpolate to wind step if needed
		if (!XParam.windU.inputfile.empty())
		{
			if (XParam.windU.uniform == 1)
			{
				//
				int Wstepinbnd = 1;



				// Do this for all the corners
				//Needs limiter in case WLbnd is empty
				double difft = XParam.windU.data[Wstepinbnd].time - XParam.totaltime;

				while (difft < 0.0)
				{
					Wstepinbnd++;
					difft = XParam.windU.data[Wstepinbnd].time - XParam.totaltime;
				}

				uwinduni = interptime(XParam.windU.data[Wstepinbnd].uwind, XParam.windU.data[Wstepinbnd - 1].uwind, XParam.windU.data[Wstepinbnd].time - XParam.windU.data[Wstepinbnd - 1].time, XParam.totaltime - XParam.windU.data[Wstepinbnd - 1].time);
				vwinduni = interptime(XParam.windU.data[Wstepinbnd].vwind, XParam.windU.data[Wstepinbnd - 1].vwind, XParam.windU.data[Wstepinbnd].time - XParam.windU.data[Wstepinbnd - 1].time, XParam.totaltime - XParam.windU.data[Wstepinbnd - 1].time);


			}
			else
			{
				cstwind = 0;
				int readfirststep = min(max((int)floor((XParam.totaltime - XParam.windU.to) / XParam.windU.dt), 0), XParam.windU.nt - 2);

				if (readfirststep + 1 > windstep)
				{
					// Need to read a new step from the file
					for (int iw = 0; iw < XParam.windU.nx*XParam.windU.ny; iw++)
					{
						//
						Uwbef[iw] = Uwaft[iw];
						Vwbef[iw] = Vwaft[iw];

					}

					readWNDstep(XParam.windU, XParam.windV, readfirststep + 1, Uwaft, Vwaft);
					windstep = readfirststep + 1;
				}



				InterpstepCPU(XParam.windU.nx, XParam.windU.ny, readfirststep, XParam.totaltime, XParam.windU.dt, Uwind, Uwbef, Uwaft);
				InterpstepCPU(XParam.windV.nx, XParam.windV.ny, readfirststep, XParam.totaltime, XParam.windV.dt, Vwind, Vwbef, Vwaft);
			}
		}

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

				if (!XParam.windU.inputfile.empty() || !XParam.atmP.inputfile.empty())
				{
					XParam.dt = FlowCPUATM(XParam, nextoutputtime, cstwind, cstpress, uwinduni, uwinduni);
				}
				else
				{
					XParam.dt = FlowCPU(XParam, nextoutputtime);
				}

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
				stepread.zs = zs[XParam.TSnodesout[o].i + XParam.TSnodesout[o].j*16 +XParam.TSnodesout[o].block*XParam.blksize];
				stepread.hh = hh[XParam.TSnodesout[o].i + XParam.TSnodesout[o].j* 16 + XParam.TSnodesout[o].block*XParam.blksize];
				stepread.uu = uu[XParam.TSnodesout[o].i + XParam.TSnodesout[o].j* 16 + XParam.TSnodesout[o].block*XParam.blksize];
				stepread.vv = vv[XParam.TSnodesout[o].i + XParam.TSnodesout[o].j* 16 + XParam.TSnodesout[o].block*XParam.blksize];
				zsAllout[o].push_back(stepread);

			}
			nTSstep++;

		}

#ifdef USE_CATALYST
                // Could use existing global time step counter here
                catalystTimeStep += 1;
                if (XParam.use_catalyst)
                {
                        // Check if Catalyst should run at this simulation time or time step
                        catalystAdaptor& adaptor = catalystAdaptor::getInstance();
                        if (adaptor.requestDataDescription(XParam.totaltime, catalystTimeStep))
                        {
                                // Use same output mechanism as netCDF output for updating VTK data fields
                                for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
                                {
                                        if (OutputVarMaplen[XParam.outvars[ivar]] > 0)
                                        {
                                                for (int blockId = 0; blockId < XParam.nblk; blockId++)
                                                {
                                                        int adaptorStatus = 0;
                                                        if (XParam.doubleprecision == 1 || XParam.spherical == 1)
                                                        {
                                                                // Find memory address for this block - hardcoding 16x16 block size here
                                                                double * dataptr = OutputVarMapCPUD[XParam.outvars[ivar]] + blockId*16*16;
                                                                adaptorStatus = adaptor.updateFieldDouble(blockId, XParam.outvars[ivar], dataptr);
                                                        }
                                                        else
                                                        {
                                                                float * dataptr = OutputVarMapCPU[XParam.outvars[ivar]] + blockId*16*16;
                                                                adaptorStatus = adaptor.updateFieldSingle(blockId, XParam.outvars[ivar], dataptr);
                                                        }
                                                        if (adaptorStatus) fprintf(stderr, "catalystAdaptor::updateField failed");
                                                }
                                        }
                                }
                                // Run Catalyst pipeline
                                if (adaptor.runCoprocessor()) fprintf(stderr, "catalystAdaptor::runCoprocessor failed");
                        }
                }
#endif

		// CHeck for grid output
		if (nextoutputtime - XParam.totaltime <= XParam.dt*0.00001f  && XParam.outputtimestep > 0.0)
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
							//writencvarstep(XParam,blockxo_d,blockyo_d, XParam.outvars[ivar], OutputVarMapCPUD[XParam.outvars[ivar]]);
							writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPUD[XParam.outvars[ivar]]);
						}
						else
						{
							//writencvarstep(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
							writencvarstepBUQ(XParam, 3, activeblk, level, blockxo_d, blockyo_d, XParam.outvars[ivar], OutputVarMapCPU[XParam.outvars[ivar]]);
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
	Param defaultParam; // This is used later 

	//The main function setups all the init of the model and then calls the mainloop to actually run the model

	// Theire are many (12) mainloops depending whether the model runs on the GPU/CPU and whether the implementation is float/double or spherical coordinate (double only)


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


	//////////////////////////////////////////////////////
	/////             Sanity check                   /////
	//////////////////////////////////////////////////////

	//put this  in separate function

	//force double for Rain on grid cases
	if (!XParam.Rainongrid.inputfile.empty())
	{
		XParam.doubleprecision = 1;
	}





	///////////////////////////////////////////
	//  Read Bathy header
	///////////////////////////////////////////

	//this sets nx ny dx delta xo yo etc...

	XParam.Bathymetry = readBathyhead(XParam.Bathymetry);

	//Here if xo, xmax, yo, ymax dx and grdalfa have not been set by the user use the values specified by the bathymetry grid
	if (XParam.xo == XParam.xmax)
	{
		XParam.xo = XParam.Bathymetry.xo;
		XParam.xmax = XParam.Bathymetry.xmax;
	}

	if (XParam.yo == XParam.ymax)
	{
		XParam.yo = XParam.Bathymetry.yo;
		XParam.ymax = XParam.Bathymetry.ymax;
	}

	if (XParam.dx == 0.0)
	{
		XParam.dx = XParam.Bathymetry.dx;

	}
	if (XParam.grdalpha == 0.0) // This default value sucks because 0.0 should be enforcable from the input parameter file
	{
		XParam.grdalpha = XParam.Bathymetry.grdalpha;

	}


	double levdx = calcres(XParam.dx ,XParam.initlevel);// true grid resolution as in dx/2^(initlevel)
	printf("levdx=%f;1 << XParam.initlevel=%f\n", levdx, calcres(1.0, XParam.initlevel));

	XParam.nx = (XParam.xmax - XParam.xo) / (levdx)+1;
	XParam.ny = (XParam.ymax - XParam.yo) / (levdx)+1; //+1?


	if (XParam.spherical < 1)
	{
		XParam.delta = XParam.dx;
		XParam.grdalpha = XParam.grdalpha*pi / 180.0; // grid rotation

	}
	else
	{
		//Geo grid
		XParam.delta = XParam.dx * XParam.Radius*pi / 180.0;
		printf("Using spherical coordinate; delta=%f rad\n", XParam.delta);
		write_text_to_log_file("Using spherical coordinate; delta=" + std::to_string(XParam.delta));
		if (XParam.grdalpha != 0.0)
		{
			printf("grid rotation in spherical coordinate is not supported yet. grdalpha=%f rad\n", XParam.grdalpha);
			write_text_to_log_file("grid rotation in spherical coordinate is not supported yet. grdalpha=" + std::to_string(XParam.grdalpha*180.0 / pi));
		}
	}



	/////////////////////////////////////////////////////
	////// CHECK PARAMETER SANITY
	/////////////////////////////////////////////////////
	XParam = checkparamsanity(XParam);



	int nx = XParam.nx;
	int ny = XParam.ny;

	//printf("Model domain info: nx=%d\tny=%d\tdx=%f\talpha=%f\txo=%f\txmax=%f\tyo=%f\tymax=%f\n", XParam.nx, XParam.ny, XParam.dx, XParam.grdalpha * 180.0 / pi, XParam.xo, XParam.xmax, XParam.yo, XParam.ymax);



	////////////////////////////////////////////////
	// read the bathy file (and store to dummy for now)
	////////////////////////////////////////////////
	Allocate1CPU(XParam.Bathymetry.nx, XParam.Bathymetry.ny, bathydata);
	Allocate1CPU(XParam.Bathymetry.nx, XParam.Bathymetry.ny, dummy);
	Allocate1CPU(XParam.Bathymetry.nx, XParam.Bathymetry.ny, dummy_d);

	printf("Read Bathy data...");
	write_text_to_log_file("Read Bathy data");


	// Check bathy extension
	std::string bathyext;

	std::vector<std::string> extvec = split(XParam.Bathymetry.inputfile, '.');

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
		readbathyMD(XParam.Bathymetry.inputfile, dummy);
	}
	if (bathyext.compare("nc") == 0)
	{
		readnczb(XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.inputfile, dummy);
	}
	if (bathyext.compare("bot") == 0 || bathyext.compare("dep") == 0)
	{
		readXBbathy(XParam.Bathymetry.inputfile, XParam.Bathymetry.nx, XParam.Bathymetry.ny, dummy);
	}
	if (bathyext.compare("asc") == 0)
	{
		//
		readbathyASCzb(XParam.Bathymetry.inputfile, XParam.Bathymetry.nx, XParam.Bathymetry.ny, dummy);
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
		for (int j = 0; j <XParam.Bathymetry.ny; j++)
		{
			for (int i = 0; i < XParam.Bathymetry.nx; i++)
			{
				dummy[i + j*XParam.Bathymetry.nx] = dummy[i + j*XParam.Bathymetry.nx] * -1.0f;
				//printf("%f\n", zb[i + (j)*nx]);

			}
		}
	}

	for (int j = 0; j < XParam.Bathymetry.ny; j++)
	{
		for (int i = 0; i < XParam.Bathymetry.nx; i++)
		{
			bathydata[i + j * XParam.Bathymetry.nx] = dummy[i + j * XParam.Bathymetry.nx];
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
					double x = XParam.xo + (i + 16 * nblkx)*levdx;
					double y = XParam.yo + (j + 16 * nblky)*levdx;

					if (x >= XParam.Bathymetry.xo && x <= XParam.Bathymetry.xmax && y >= XParam.Bathymetry.yo && y <= XParam.Bathymetry.ymax)
					{
						// cells that falls off this domain are assigned
						double x1, x2, y1, y2;
						double q11, q12, q21, q22, q;
						int cfi, cfip, cfj, cfjp;



						cfi = min(max((int)floor((x - XParam.Bathymetry.xo) / XParam.Bathymetry.dx), 0), XParam.Bathymetry.nx - 2);
						cfip = cfi + 1;

						x1 = XParam.Bathymetry.xo + XParam.Bathymetry.dx*cfi;
						x2 = XParam.Bathymetry.xo + XParam.Bathymetry.dx*cfip;

						cfj = min(max((int)floor((y - XParam.Bathymetry.yo) / XParam.Bathymetry.dx), 0), XParam.Bathymetry.ny - 2);
						cfjp = cfj + 1;

						y1 = XParam.Bathymetry.yo + XParam.Bathymetry.dx*cfj;
						y2 = XParam.Bathymetry.yo + XParam.Bathymetry.dx*cfjp;

						q11 = dummy[cfi + cfj*XParam.Bathymetry.nx];
						q12 = dummy[cfi + cfjp*XParam.Bathymetry.nx];
						q21 = dummy[cfip + cfj*XParam.Bathymetry.nx];
						q22 = dummy[cfip + cfjp*XParam.Bathymetry.nx];

						q = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
						//printf("q = %f\n", q);
						//printf("mloc: %i\n", mloc);
						if (q >= XParam.mask)
							nmask++;
					}
					else
					{
						//computational domnain is outside of the bathy domain
						nmask++;
					}

				}
			}
			if (nmask < 256)
				nblk++;
		}
	}

	XParam.nblk = nblk;

	XParam.nblkmem = (int)ceil(nblk*XParam.membuffer); //5% buffer on the memory for adaptation 


	int blksize = XParam.blksize; //useful below
	printf("Number of blocks: %i\n",nblk);

	////////////////////////////////////////////////
	///// Allocate and arrange blocks
	////////////////////////////////////////////////
	// caluculate the Block xo yo and what are its neighbour


	Allocate1CPU(XParam.nblkmem, 1, blockxo);
	Allocate1CPU(XParam.nblkmem, 1, blockyo);
	Allocate1CPU(XParam.nblkmem, 1, blockxo_d);
	Allocate1CPU(XParam.nblkmem, 1, blockyo_d);
	Allocate4CPU(XParam.nblkmem, 1, leftblk, rightblk, topblk, botblk);

	Allocate1CPU(XParam.nblkmem, 1, level);
	Allocate1CPU(XParam.nblkmem, 1, newlevel);
	Allocate1CPU(XParam.nblkmem, 1, activeblk);
	Allocate1CPU(XParam.nblkmem, 1, availblk);
	Allocate1CPU(XParam.nblkmem, 1, csumblk);
	Allocate1CPU(XParam.nblkmem, 1, coarsen);
	Allocate1CPU(XParam.nblkmem, 1, refine);
	Allocate1CPU(XParam.nblkmem, 1, invactive);

	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		activeblk[ibl] = -1;
		invactive[ibl] = -1;
	}


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
					double x = XParam.xo + (i + 16 * nblkx)*levdx;
					double y = XParam.yo + (j + 16 * nblky)*levdx;

					//x = max(min(x, XParam.Bathymetry.xmax), XParam.Bathymetry.xo);
					//y = max(min(y, XParam.Bathymetry.ymax), XParam.Bathymetry.yo);
					if (x >= XParam.Bathymetry.xo && x <= XParam.Bathymetry.xmax && y >= XParam.Bathymetry.yo && y <= XParam.Bathymetry.ymax)
					{
						// cells that falls off this domain are assigned
						double x1, x2, y1, y2;
						double q11, q12, q21, q22, q;
						int cfi, cfip, cfj, cfjp;



						cfi = min(max((int)floor((x - XParam.Bathymetry.xo) / XParam.Bathymetry.dx), 0), XParam.Bathymetry.nx - 2);
						cfip = cfi + 1;

						x1 = XParam.Bathymetry.xo + XParam.Bathymetry.dx*cfi;
						x2 = XParam.Bathymetry.xo + XParam.Bathymetry.dx*cfip;

						cfj = min(max((int)floor((y - XParam.Bathymetry.yo) / XParam.Bathymetry.dx), 0), XParam.Bathymetry.ny - 2);
						cfjp = cfj + 1;

						y1 = XParam.Bathymetry.yo + XParam.Bathymetry.dx*cfj;
						y2 = XParam.Bathymetry.yo + XParam.Bathymetry.dx*cfjp;

						q11 = dummy[cfi + cfj*XParam.Bathymetry.nx];
						q12 = dummy[cfi + cfjp*XParam.Bathymetry.nx];
						q21 = dummy[cfip + cfj*XParam.Bathymetry.nx];
						q22 = dummy[cfip + cfjp*XParam.Bathymetry.nx];

						q = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
						//printf("q = %f\t q11=%f\t, q12=%f\t, q21=%f\t, q22=%f\t, x1=%f\t, x2=%f\t, y1=%f\t, y2=%f\t, x=%f\t, y=%f\t\n", q, q11, q12, q21, q22, x1, x2, y1, y2, x, y);
						//printf("mloc: %i\n", mloc);
						if (q >= XParam.mask)
							nmask++;
					}
					else
					{
						//computational domnain is outside of the bathy domain
						nmask++;
					}

				}
			}
			if (nmask < 256)
			{
				//
				blockxo_d[blkid] = XParam.xo + nblkx * 16.0 * levdx;
				blockyo_d[blkid] = XParam.yo + nblky * 16.0 * levdx;
				activeblk[blkid] = blkid;
				//printf("blkxo=%f\tblkyo=%f\n", blockxo_d[blkid], blockyo_d[blkid]);
				blkid++;
			}
		}
	}

	

	double leftxo, rightxo, topxo, botxo, leftyo, rightyo, topyo, botyo;
	for (int ibl = 0; ibl < nblk; ibl++)
	{
		int bl = activeblk[ibl];
		double espdist = 0.00000001;///WARMING

		leftxo = blockxo_d[bl] - 16.0 * levdx; // in adaptive this shoulbe be a range 

		leftyo = blockyo_d[bl];
		rightxo = blockxo_d[bl] + 16.0 * levdx;
		rightyo = blockyo_d[bl];
		topxo = blockxo_d[bl];
		topyo = blockyo_d[bl] + 16.0 * levdx;
		botxo = blockxo_d[bl];
		botyo = blockyo_d[bl] - 16.0 * levdx;

		// by default neighbour block refer to itself. i.e. if the neighbour block is itself then there are no neighbour
		leftblk[bl] = bl;
		rightblk[bl] = bl;
		topblk[bl] = bl;
		botblk[bl] = bl;
		for (int iblb = 0; iblb < nblk; iblb++)
		{
			//
			int blb = activeblk[iblb];

			if (abs(blockxo_d[blb] - leftxo) < espdist  && abs(blockyo_d[blb] - leftyo) < espdist)
			{
				leftblk[bl] = blb;
			}
			if (abs(blockxo_d[blb] - rightxo)  < espdist && abs(blockyo_d[blb] - rightyo) < espdist)
			{
				rightblk[bl] = blb;
			}
			if (abs(blockxo_d[blb] - topxo) < espdist && abs(blockyo_d[blb] - topyo) < espdist)
			{
				topblk[bl] = blb;

			}
			if (abs(blockxo_d[blb] - botxo) < espdist && abs(blockyo_d[blb] - botyo) < espdist)
			{
				botblk[bl] = blb;
			}
		}

		//printf("leftxo=%f\t leftyo=%f\t rightxo=%f\t rightyo=%f\t botxo=%f\t botyo=%f\t topxo=%f\t topyo=%f\n", leftxo, leftyo, rightxo, rightyo, botxo, botyo, topxo, topyo);
		//printf("blk=%d\t blockxo=%f\t blockyo=%f\t leftblk=%d\t rightblk=%d\t botblk=%d\t topblk=%d\n",bl, blockxo_d[bl], blockyo_d[bl], leftblk[bl], rightblk[bl], botblk[bl], topblk[bl]);


	}

	for (int ibl = 0; ibl < nblk; ibl++)
	{
		int bl = activeblk[ibl];
		blockxo[bl] = blockxo_d[bl];
		blockyo[bl] = blockyo_d[bl];
		level[bl] = XParam.initlevel;
		newlevel[bl] = 0;
		coarsen[bl] = false;
		refine[bl] = false;
		
	}

	for (int ibl = 0; ibl < (XParam.nblkmem - XParam.nblk); ibl++)
	{
		
		availblk[ibl] = XParam.nblk + ibl;
		XParam.navailblk++;

	}


	// Also recalculate xmax and ymax here
	//xo + (ceil(nx / 16.0)*16.0 - 1)*dx
	XParam.xmax = XParam.xo + (ceil(XParam.nx / 16.0) * 16.0 - 1)*levdx;
	XParam.ymax = XParam.yo + (ceil(XParam.ny / 16.0) * 16.0 - 1)*levdx;


	printf("Model domain info: nx=%d\tny=%d\tlevdx(dx)=%f(%f)\talpha=%f\txo=%f\txmax=%f\tyo=%f\tymax=%f\n", XParam.nx, XParam.ny, levdx, XParam.dx, XParam.grdalpha * 180.0 / pi, XParam.xo, XParam.xmax, XParam.yo, XParam.ymax);




	//////////////////////////////////////////////////
	////// Preprare Bnd
	//////////////////////////////////////////////////

	// So far bnd are limited to be cst along an edge
	// Read Bnd file if/where needed
	printf("Reading and preparing Boundaries...");
	write_text_to_log_file("Reading and preparing Boundaries");

	if (!XParam.leftbnd.inputfile.empty())
	{
		//XParam.leftbnd.data = readWLfile(XParam.leftbnd.inputfile);
		XParam.leftbnd.data = readbndfile(XParam.leftbnd.inputfile,XParam,0);

		XParam.leftbnd.on = 1; // redundant?
	}
	if (!XParam.rightbnd.inputfile.empty())
	{
		XParam.rightbnd.data = readbndfile(XParam.rightbnd.inputfile, XParam, 2);
		XParam.rightbnd.on = 1;
	}
	if (!XParam.topbnd.inputfile.empty())
	{
		XParam.topbnd.data = readbndfile(XParam.topbnd.inputfile, XParam, 3);
		XParam.topbnd.on = 1;
	}
	if (!XParam.botbnd.inputfile.empty())
	{
		XParam.botbnd.data = readbndfile(XParam.botbnd.inputfile, XParam, 1);
		XParam.botbnd.on = 1;
	}


	//Check that endtime is no longer than boundaries (if specified to other than wall or neumann)
	XParam.endtime = setendtime(XParam);


	printf("...done!\n");
	write_text_to_log_file("Done Reading and preparing Boundaries");

	XParam.dt = 0.0;// Will be resolved in update


	// Find how many blocks are on each bnds
	int blbr = 0, blbb = 0, blbl = 0, blbt = 0;
	for (int ibl = 0; ibl < nblk; ibl++)
	{
		double espdist = 0.00000001;///WARMING

		int bl = activeblk[ibl];
		leftxo = blockxo_d[bl]; // in adaptive this shoulbe be a range 

		leftyo = blockyo_d[bl];
		rightxo = blockxo_d[bl] + 15.0 * levdx;
		rightyo = blockyo_d[bl];
		topxo = blockxo_d[bl];
		topyo = blockyo_d[bl] + 15.0 * levdx;
		botxo = blockxo_d[bl];
		botyo = blockyo_d[bl];

		if ((rightxo - XParam.xmax) > (-1.0*levdx))
		{
			//
			blbr++;
			//bndrightblk[blbr] = bl;

		}

		if ((topyo - XParam.ymax) > (-1.0*levdx))
		{
			//
			blbt++;
			//bndtopblk[blbt] = bl;

		}
		if ((XParam.yo - botyo) > (-1.0*levdx))
		{
			//
			blbb++;
			//bndbotblk[blbb] = bl;

		}
		if ((XParam.xo - leftxo) > (-1.0*levdx))
		{
			//
			blbl++;
			//bndleftblk[blbl] = bl;

		}
	}

	//
	XParam.leftbnd.nblk = blbl;
	XParam.rightbnd.nblk = blbr;
	XParam.topbnd.nblk = blbt;
	XParam.botbnd.nblk = blbb;

	//
	Allocate1CPU(blbl, 1, bndleftblk);
	Allocate1CPU(blbr, 1, bndrightblk);
	Allocate1CPU(blbt, 1, bndtopblk);
	Allocate1CPU(blbb, 1, bndbotblk);

	blbr = blbb = blbl = blbt = 0;
	for (int ibl = 0; ibl < nblk; ibl++)
	{
		double espdist = 0.00000001;///WARMING
		int bl = activeblk[ibl];

		leftxo = blockxo_d[bl] ; // in adaptive this shoulbe be a range
		leftyo = blockyo_d[bl];
		rightxo = blockxo_d[bl] + 15.0 * levdx;
		rightyo = blockyo_d[bl];
		topxo = blockxo_d[bl];
		topyo = blockyo_d[bl] + 15.0 * levdx;
		botxo = blockxo_d[bl];
		botyo = blockyo_d[bl];

		if ((rightxo - XParam.xmax) > (-1.0*levdx))
		{
			//

			bndrightblk[blbr] = bl;
			blbr++;

		}

		if ((topyo - XParam.ymax) > (-1.0*levdx))
		{
			//

			bndtopblk[blbt] = bl;
			blbt++;

		}
		if ((XParam.yo - botyo) > (-1.0*levdx))
		{
			//

			bndbotblk[blbb] = bl;
			blbb++;

		}
		if ((XParam.xo - leftxo) > (-1.0*levdx))
		{
			//

			bndleftblk[blbl] = bl;
			blbl++;
			//printf("bl_left=%d\n", bl);

		}
	}




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
			//  if no GPU device are present then use the CPU (GPUDEVICE = -1)
			XParam.GPUDEVICE = (nDevices - 1);
		}
		cudaGetDeviceProperties(&prop, XParam.GPUDEVICE);
		printf("There are %d GPU devices on this machine\n", nDevices);
		write_text_to_log_file("There are " + std::to_string(nDevices) + "GPU devices on this machine");

		if (XParam.GPUDEVICE >= 0)
		{
			printf("Using Device : %s\n", prop.name);
			write_text_to_log_file("Using Device: " + std::string(prop.name));
		}
		else
		{
			printf("Warning ! No GPU device were detected on this machine... Using CPU instead");
			write_text_to_log_file("Warning ! No GPU device were detected on this machine... Using CPU instead");
		}

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
		check = AllocMemGPUBND(XParam);

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
		for (int j = 0; j < XParam.Bathymetry.ny; j++)
		{
			for (int i = 0; i < XParam.Bathymetry.nx; i++)
			{
				dummy_d[i + j*XParam.Bathymetry.nx] = dummy[i + j*XParam.Bathymetry.nx] * 1.0; //*1.0 elevates to double safely
			}
		}
		interp2BUQ(XParam.nblk, XParam.blksize, levdx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, dummy_d, zb_d);

		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo_d, blockyo_d, dummy_d, zb_d);
	}
	else
	{
		interp2BUQ(XParam.nblk, XParam.blksize, levdx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, dummy, zb);

		//carttoBUQ(XParam.nblk, XParam.nx, XParam.ny, XParam.xo, XParam.yo, XParam.dx, blockxo_d, blockyo_d, dummy, zb);
	}




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
			hotstartsucess = readhotstartfileD(XParam, leftblk, rightblk,topblk, botblk, blockxo_d, blockyo_d,  zs_d, zb_d, hh_d, uu_d, vv_d);
		}
		else
		{
			hotstartsucess = readhotstartfile(XParam, leftblk, rightblk, topblk,  botblk, blockxo_d, blockyo_d,  zs, zb, hh, uu, vv);
		}
		//add offset if present
		if (abs(XParam.zsoffset - defaultParam.zsoffset) > epsilon) // apply specified zsoffset
		{
			printf("add offset to zs and hh... ");
			//
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				AddZSoffset(XParam, zb_d, zs_d, hh_d);
			}
			else
			{
				AddZSoffset(XParam, zb, zs, hh);
			}

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

		//Param defaultParam;
		//!leftWLbnd.empty()

		//case 2b (i.e. zsinint and no boundaries were specified)
		if ((abs(XParam.zsinit - defaultParam.zsinit) <= epsilon) && (!XParam.leftbnd.on && !XParam.rightbnd.on && !XParam.topbnd.on && !XParam.botbnd.on)) //zsinit is default
		{
			XParam.zsinit = 0.0; // better default value
		}

		//case(1)
		if (abs(XParam.zsinit - defaultParam.zsinit) > epsilon) // apply specified zsinit
		{
			int coldstartsucess = 0;
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				coldstartsucess = coldstart(XParam, zb_d, uu_d, vv_d, zs_d, hh_d);
				printf("Cold start  ");
				write_text_to_log_file("Cold start");
			}
			else
			{
				coldstartsucess = coldstart(XParam, zb, uu, vv, zs, hh);
				printf("Cold start  ");
				write_text_to_log_file("Cold start");
			}

		}
		else // lukewarm start i.e. bilinear interpolation of zs at bnds // Argggh!
		{
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				warmstart(XParam,  zb_d, uu_d, vv_d, zs_d, hh_d);
				printf("Warm start  ");
				write_text_to_log_file("Warm start");
			}
			else
			{
				warmstart(XParam,  zb, uu, vv, zs, hh);
				printf("Warm start  ");
				write_text_to_log_file("Warm start");

			}
		}// end else

	}

	


	printf("done \n");
	write_text_to_log_file("Done");

	//////////////////////////////////////////////////////
	// Init other variables
	/////////////////////////////////////////////////////
	// free dummy and dummy_d because they are of size nx*ny but we want them nblk*blksize since we can't predict if one is larger then the other I'd rather free and malloc rather the realloc
	free(dummy);
	free(dummy_d);

	Allocate1CPU(XParam.nblkmem, XParam.blksize, dummy);
	Allocate1CPU(XParam.nblkmem, XParam.blksize, dummy_d);






	// Below is not succint but way faster than one loop that checks the if statemenst each time 
	// relocate this to the Allocate CPU part?
	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{

		CopyArray(XParam.nblk, XParam.blksize, hh_d, hho_d);
		CopyArray(XParam.nblk, XParam.blksize, uu_d, uuo_d);
		CopyArray(XParam.nblk, XParam.blksize, vv_d, vvo_d);
		CopyArray(XParam.nblk, XParam.blksize, zs_d, zso_d);

		// Set default cf
		InitArraySV(XParam.nblk, XParam.blksize, XParam.cf, cf_d);

		if (XParam.outhhmax == 1)
		{
			CopyArray(XParam.nblk, XParam.blksize, hh_d, hhmax_d);
		}

		if (XParam.outhhmean == 1)
		{
			InitArraySV(XParam.nblk, XParam.blksize, 0.0, hhmean_d);
		}
		if (XParam.outzsmax == 1)
		{
			CopyArray(XParam.nblk, XParam.blksize, zs_d, zsmax_d);
		}

		if (XParam.outzsmean == 1)
		{
			InitArraySV(XParam.nblk, XParam.blksize, 0.0, zsmean_d);
		}

		if (XParam.outuumax == 1)
		{
			CopyArray(XParam.nblk, XParam.blksize, uu_d, uumax_d);
		}

		if (XParam.outuumean == 1)
		{
			InitArraySV(XParam.nblk, XParam.blksize, 0.0, uumean_d);
		}
		if (XParam.outvvmax == 1)
		{
			CopyArray(XParam.nblk, XParam.blksize, vv_d, vvmax_d);
		}

		if (XParam.outvvmean == 1)
		{
			InitArraySV(XParam.nblk, XParam.blksize, 0.0, vvmean_d);
		}
		if (XParam.outvort == 1)
		{
			InitArraySV(XParam.nblk, XParam.blksize, 0.0, vort_d);
		}
	}
	else //Using Float *
	{

		CopyArray(XParam.nblk, XParam.blksize, hh, hho);
		CopyArray(XParam.nblk, XParam.blksize, uu, uuo);
		CopyArray(XParam.nblk, XParam.blksize, vv, vvo);
		CopyArray(XParam.nblk, XParam.blksize, zs, zso);

		// Set default cf
		InitArraySV(XParam.nblk, XParam.blksize,(float) XParam.cf, cf);

		if (XParam.outhhmax == 1)
		{
			CopyArray(XParam.nblk, XParam.blksize, hh, hhmax);
		}

		if (XParam.outhhmean == 1)
		{
			InitArraySV(XParam.nblk, XParam.blksize, 0.0f, hhmean);
		}
		if (XParam.outzsmax == 1)
		{
			CopyArray(XParam.nblk, XParam.blksize, zs, zsmax);
		}

		if (XParam.outzsmean == 1)
		{
			InitArraySV(XParam.nblk, XParam.blksize, 0.0f, zsmean);
		}

		if (XParam.outuumax == 1)
		{
			CopyArray(XParam.nblk, XParam.blksize, uu, uumax);
		}

		if (XParam.outuumean == 1)
		{
			InitArraySV(XParam.nblk, XParam.blksize, 0.0f, uumean);
		}
		if (XParam.outvvmax == 1)
		{
			CopyArray(XParam.nblk, XParam.blksize, vv, vvmax);
		}

		if (XParam.outvvmean == 1)
		{
			InitArraySV(XParam.nblk, XParam.blksize, 0.0f, vvmean);
		}
		if (XParam.outvort == 1)
		{
			InitArraySV(XParam.nblk, XParam.blksize, 0.0f, vort);
		}

	}

	///////////////////////////////////////////////////
	// Friction maps
	///////////////////////////////////////////////////

	if (!XParam.roughnessmap.inputfile.empty())
	{
		// roughness map was specified!

		// read the roughness map header
		XParam.roughnessmap = readcfmaphead(XParam.roughnessmap);

		// Quick Sanity check if nx and ny are not read properly just ignore cfmap
		if (XParam.roughnessmap.nx > 0 && XParam.roughnessmap.ny > 0)
		{

			// Allocate memory to read roughness map file content
			float * cfmapinput; // init as a float because the bathy subroutine expect a float
			Allocate1CPU(XParam.roughnessmap.nx, XParam.roughnessmap.ny, cfmapinput);

			// read the roughness map data
			readmapdata(XParam.roughnessmap, cfmapinput);

			// Interpolate data to the roughness array
			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{
				//
				double * cfmapinput_d; // init as a float because the bathy subroutine expect a float
				Allocate1CPU(XParam.roughnessmap.nx, XParam.roughnessmap.ny, cfmapinput_d);
				for (int j = 0; j < XParam.roughnessmap.ny; j++)
				{
					for (int i = 0; i < XParam.roughnessmap.nx; i++)
					{
						cfmapinput_d[i + j*XParam.roughnessmap.nx] = cfmapinput[i + j*XParam.roughnessmap.nx] * 1.0;
					}
				}

				interp2BUQ(XParam.nblk, XParam.blksize, levdx, blockxo_d, blockyo_d, XParam.roughnessmap.nx, XParam.roughnessmap.ny, XParam.roughnessmap.xo, XParam.roughnessmap.xmax, XParam.roughnessmap.yo, XParam.roughnessmap.ymax, XParam.roughnessmap.dx, cfmapinput_d, cf_d);

				//interp2cf(XParam, cfmapinput, blockxo_d, blockyo_d, cf_d);
				free(cfmapinput_d);
			}
			else
			{
				//
				interp2BUQ(XParam.nblk, XParam.blksize, levdx, blockxo_d, blockyo_d, XParam.roughnessmap.nx, XParam.roughnessmap.ny, XParam.roughnessmap.xo, XParam.roughnessmap.xmax, XParam.roughnessmap.yo, XParam.roughnessmap.ymax, XParam.roughnessmap.dx, cfmapinput, cf);

				//interp2cf(XParam, cfmapinput, blockxo, blockyo, cf);
			}

			// cleanup
			free(cfmapinput);
		}
		else
		{
			//Error message
			printf("Error while reading roughness map. Using constant roughness instead ");
			write_text_to_log_file("Error while reading roughness map. Using constant roughness instead ");
		}
	}



	if (XParam.deform.size()>0)
	{
		// Deformation files was specified!

		for (int nd = 0; nd < XParam.deform.size(); nd++)
		{
			// read the roughness map header
			XParam.deform[nd].grid = readcfmaphead(XParam.deform[nd].grid);
			// deform data is read and allocatted and applied only when needed so it doesn't use any unecessary memory
			// On the other hand applying deformation over long duration will be slow (it will read teh file at every step for teh duration of teh deformation)
			XParam.deformmaxtime = max(XParam.deformmaxtime, XParam.deform[nd].startime + XParam.deform[nd].duration);
		}


	}



	///////////////////////////////////////////////////
	// GPU data init
	///////////////////////////////////////////////////

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
			CUDA_CHECK(cudaMemcpy(cf_g, cf, nblk*blksize * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(blockxo_g, blockxo, nblk * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(blockyo_g, blockyo, nblk * sizeof(float), cudaMemcpyHostToDevice));
			initdtmax << <gridDim, blockDim, 0 >> >( (float)epsilon, dtmax_g);
		}

		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(leftblk_g, leftblk, nblk * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(rightblk_g, rightblk, nblk * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(topblk_g, topblk, nblk * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(botblk_g, botblk, nblk * sizeof(int), cudaMemcpyHostToDevice));


		CUDA_CHECK(cudaMemcpy(bndleftblk_g, bndleftblk, XParam.leftbnd.nblk * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(bndrightblk_g, bndrightblk, XParam.rightbnd.nblk * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(bndtopblk_g, bndtopblk, XParam.topbnd.nblk * sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(bndbotblk_g, bndbotblk, XParam.botbnd.nblk * sizeof(int), cudaMemcpyHostToDevice));



		printf("...Done\n");
		write_text_to_log_file("Done ");

	}


	//////////////////////////////////////////////////////////////////////////////////////////
	// Prep wind  / atm / rain forcing
	/////////////////////////////////////////////////////////////////////////////////////////



	/////////////////////////////////////////////////////
	// Prep River discharge
	/////////////////////////////////////////////////////

	if (XParam.Rivers.size() > 0)
	{
		double xx, yy;
		printf("Preparing rivers... ");
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
						xx = blockxo_d[bl] + i*levdx;
						yy = blockyo_d[bl] + j*levdx;
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
			XParam.Rivers[Rin].disarea = idis.size()*levdx*levdx; // That is not valid for spherical grids

			// Now read the discharge input and store to  
			XParam.Rivers[Rin].flowinput = readFlowfile(XParam.Rivers[Rin].Riverflowfile);
		}
		//Now identify sort unique blocks where rivers are being inserted
		std::vector<int> activeRiverBlk;

		for (int Rin = 0; Rin < XParam.Rivers.size(); Rin++)
		{

			activeRiverBlk.insert(std::end(activeRiverBlk), std::begin(XParam.Rivers[Rin].block), std::end(XParam.Rivers[Rin].block));
		}
		std::sort(activeRiverBlk.begin(), activeRiverBlk.end());
		activeRiverBlk.erase(std::unique(activeRiverBlk.begin(), activeRiverBlk.end()), activeRiverBlk.end());
		Allocate1CPU(activeRiverBlk.size(), 1, Riverblk);

		XParam.nriverblock = activeRiverBlk.size();

		for (int b = 0; b < activeRiverBlk.size(); b++)
		{
			Riverblk[b] = activeRiverBlk[b];
		}


		if (XParam.GPUDEVICE >= 0)
		{
			Allocate1GPU(activeRiverBlk.size(), 1, Riverblk_g);
			CUDA_CHECK(cudaMemcpy(Riverblk_g, Riverblk, activeRiverBlk.size() * sizeof(int), cudaMemcpyHostToDevice));

		}
		printf("Done\n");
	}

	/////////////////////////////////////////////////////
	// Prep Wind input
	/////////////////////////////////////////////////////
	

	if (!XParam.windU.inputfile.empty())
	{
		//windfile is present
		if (XParam.windU.uniform == 1)
		{
			// grid uniform time varying wind input
			// wlevs[0] is wind speed and wlev[1] is direction
			XParam.windU.data = readWNDfileUNI(XParam.windU.inputfile, XParam.grdalpha);
		}
		else
		{
			// grid and time varying wind input
			// read parameters fro the size of wind input
			XParam.windU = readforcingmaphead(XParam.windU);
			XParam.windV = readforcingmaphead(XParam.windV);

			Allocate1CPU(XParam.windU.nx, XParam.windU.ny, Uwind);
			Allocate1CPU(XParam.windU.nx, XParam.windU.ny, Vwind);

			Allocate4CPU(XParam.windU.nx, XParam.windU.ny, Uwbef, Uwaft, Vwbef, Vwaft);



			XParam.windU.dt = abs(XParam.windU.to - XParam.windU.tmax) / (XParam.windU.nt - 1);
			XParam.windV.dt = abs(XParam.windV.to - XParam.windV.tmax) / (XParam.windV.nt - 1);

			int readfirststep = min(max((int)floor((XParam.totaltime - XParam.windU.to) / XParam.windU.dt), 0), XParam.windU.nt - 2);



			readWNDstep(XParam.windU, XParam.windV, readfirststep, Uwbef, Vwbef);
			readWNDstep(XParam.windU, XParam.windV, readfirststep + 1, Uwaft, Vwaft);

			InterpstepCPU(XParam.windU.nx, XParam.windU.ny, readfirststep, XParam.totaltime, XParam.windU.dt, Uwind, Uwbef, Uwaft);
			InterpstepCPU(XParam.windV.nx, XParam.windV.ny, readfirststep, XParam.totaltime, XParam.windV.dt, Vwind, Vwbef, Vwaft);


			if (XParam.GPUDEVICE >= 0)
			{
				//setup GPU texture to streamline interpolation between the two array
				Allocate1GPU(XParam.windU.nx, XParam.windU.ny, Uwind_g);
				Allocate1GPU(XParam.windU.nx, XParam.windU.ny, Vwind_g);

				Allocate4GPU(XParam.windU.nx, XParam.windU.ny, Uwbef_g, Uwaft_g, Vwbef_g, Vwaft_g);


				CUDA_CHECK(cudaMemcpy(Uwind_g, Uwind, XParam.windU.nx*XParam.windU.ny * sizeof(float), cudaMemcpyHostToDevice));
				CUDA_CHECK(cudaMemcpy(Vwind_g, Vwind, XParam.windU.nx*XParam.windU.ny * sizeof(float), cudaMemcpyHostToDevice));
				CUDA_CHECK(cudaMemcpy(Uwbef_g, Uwbef, XParam.windU.nx*XParam.windU.ny * sizeof(float), cudaMemcpyHostToDevice));
				CUDA_CHECK(cudaMemcpy(Vwbef_g, Vwbef, XParam.windU.nx*XParam.windU.ny * sizeof(float), cudaMemcpyHostToDevice));
				CUDA_CHECK(cudaMemcpy(Uwaft_g, Uwaft, XParam.windU.nx*XParam.windU.ny * sizeof(float), cudaMemcpyHostToDevice));
				CUDA_CHECK(cudaMemcpy(Vwaft_g, Vwaft, XParam.windU.nx*XParam.windU.ny * sizeof(float), cudaMemcpyHostToDevice));

				//U-wind
				CUDA_CHECK(cudaMallocArray(&Uwind_gp, &channelDescUwind, XParam.windU.nx, XParam.windU.ny));


				CUDA_CHECK(cudaMemcpyToArray(Uwind_gp, 0, 0, Uwind, XParam.windU.nx * XParam.windU.ny * sizeof(float), cudaMemcpyHostToDevice));

				texUWND.addressMode[0] = cudaAddressModeClamp;
				texUWND.addressMode[1] = cudaAddressModeClamp;
				texUWND.filterMode = cudaFilterModeLinear;
				texUWND.normalized = false;


				CUDA_CHECK(cudaBindTextureToArray(texUWND, Uwind_gp, channelDescUwind));

				//V-wind
				CUDA_CHECK(cudaMallocArray(&Vwind_gp, &channelDescVwind, XParam.windV.nx, XParam.windV.ny));


				CUDA_CHECK(cudaMemcpyToArray(Vwind_gp, 0, 0, Vwind, XParam.windV.nx * XParam.windV.ny * sizeof(float), cudaMemcpyHostToDevice));

				texVWND.addressMode[0] = cudaAddressModeClamp;
				texVWND.addressMode[1] = cudaAddressModeClamp;
				texVWND.filterMode = cudaFilterModeLinear;
				texVWND.normalized = false;


				CUDA_CHECK(cudaBindTextureToArray(texVWND, Vwind_gp, channelDescVwind));




			}
		}




	}

	if (!XParam.atmP.inputfile.empty())
	{
		// read file extension; if .txt then it is applied uniformly else it is variable
		std::string ffext;

		std::vector<std::string> extvec = split(XParam.atmP.inputfile, '.');

		std::vector<std::string> nameelements;
		//by default we expect tab delimitation
		nameelements = split(extvec.back(), '?');
		if (nameelements.size() > 1)
		{
			//variable name for bathy is not given so it is assumed to be zb
			ffext = nameelements[0];
		}
		else
		{
			ffext = extvec.back();
		}


		XParam.atmP.uniform = (ffext.compare("nc") == 0) ? 0 : 1;




		if (XParam.atmP.uniform == 1)
		{
			// grid uniform time varying wind input
			//This is pointless?
			XParam.atmP.data = readINfileUNI(XParam.atmP.inputfile);;
		}
		else
		{
			// atm pressure is treated differently then wind and we need 3 arrays to store the actual data and 3 arrays for the computation (size of nblocks*blocksize)
			XParam.atmP = readforcingmaphead(XParam.atmP);
			Allocate1CPU(XParam.atmP.nx, XParam.atmP.ny, PatmX);
			Allocate1CPU(XParam.atmP.nx, XParam.atmP.ny, Patmbef);
			Allocate1CPU(XParam.atmP.nx, XParam.atmP.ny, Patmaft);

			CUDA_CHECK(cudaMemcpy(PatmX_g, PatmX, XParam.atmP.nx*XParam.atmP.ny * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(Patmbef_g, Patmbef, XParam.atmP.nx*XParam.atmP.ny * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(Patmaft_g, Patmaft, XParam.atmP.nx*XParam.atmP.ny * sizeof(float), cudaMemcpyHostToDevice));




			if (XParam.doubleprecision == 1 || XParam.spherical == 1)
			{

				Allocate1CPU(XParam.nblkmem, XParam.blksize, Patm_d);
				Allocate1CPU(XParam.nblkmem, XParam.blksize, dPdx_d);
				Allocate1CPU(XParam.nblkmem, XParam.blksize, dPdy_d);
				


			}
			else
			{
				Allocate1CPU(XParam.nblkmem, XParam.blksize, Patm);
				Allocate1CPU(XParam.nblkmem, XParam.blksize, dPdx);
				Allocate1CPU(XParam.nblkmem, XParam.blksize, dPdy);
			}

			// read the first 2 stepd of the data

			XParam.atmP.dt = abs(XParam.atmP.to - XParam.atmP.tmax) / (XParam.atmP.nt - 1);

			int readfirststep = min(max((int)floor((XParam.totaltime - XParam.atmP.to) / XParam.atmP.dt), 0), XParam.atmP.nt - 2);

			readATMstep(XParam.atmP, readfirststep, Patmbef);
			readATMstep(XParam.atmP, readfirststep+1, Patmaft);

			InterpstepCPU(XParam.atmP.nx, XParam.atmP.ny, readfirststep, XParam.totaltime, XParam.atmP.dt, PatmX, Patmbef, Patmaft);

			if (XParam.GPUDEVICE >= 0)
			{
				//setup GPU texture to streamline interpolation between the two array
				Allocate1GPU(XParam.atmP.nx, XParam.atmP.ny, PatmX_g);
				Allocate1GPU(XParam.atmP.nx, XParam.atmP.ny, Patmbef_g);
				Allocate1GPU(XParam.atmP.nx, XParam.atmP.ny, Patmaft_g);
				if (XParam.doubleprecision == 1 || XParam.spherical == 1)
				{
					Allocate1GPU(XParam.nblkmem, XParam.blksize, Patm_gd);
					Allocate1GPU(XParam.nblkmem, XParam.blksize, dPdx_gd);
					Allocate1GPU(XParam.nblkmem, XParam.blksize, dPdy_gd);
				}
				else
				{

					Allocate1GPU(XParam.nblkmem, XParam.blksize, Patm_gd);
					Allocate1GPU(XParam.nblkmem, XParam.blksize, dPdx_gd);
					Allocate1GPU(XParam.nblkmem, XParam.blksize, dPdy_gd);

				}
				CUDA_CHECK(cudaMallocArray(&Patm_gp, &channelDescPatm, XParam.atmP.nx, XParam.atmP.ny));


				CUDA_CHECK(cudaMemcpyToArray(Patm_gp, 0, 0, PatmX, XParam.atmP.nx * XParam.atmP.ny * sizeof(float), cudaMemcpyHostToDevice));

				texPATM.addressMode[0] = cudaAddressModeClamp;
				texPATM.addressMode[1] = cudaAddressModeClamp;
				texPATM.filterMode = cudaFilterModeLinear;
				texPATM.normalized = false;


				CUDA_CHECK(cudaBindTextureToArray(texPATM, Patm_gp, channelDescPatm));

			}


		}
	}

	////////////////////////////////////////
	//           Rain on grid         //////
	////////////////////////////////////////
	if (!XParam.Rainongrid.inputfile.empty())
	{
		//rain file is present
		if (XParam.Rainongrid.uniform == 1)
		{
			// grid uniform time varying wind input
			// wlevs[0] is wind speed and wlev[1] is direction
			XParam.Rainongrid.data = readINfileUNI(XParam.Rainongrid.inputfile);
		}
		else
		{
			// grid and time varying wind input
			// read parameters fro the size of wind input
			XParam.Rainongrid = readforcingmaphead(XParam.Rainongrid);

			Allocate1CPU(XParam.Rainongrid.nx, XParam.Rainongrid.ny, Rain);
			Allocate1CPU(XParam.Rainongrid.nx, XParam.Rainongrid.ny, Rainbef);
			Allocate1CPU(XParam.Rainongrid.nx, XParam.Rainongrid.ny, Rainaft);

			XParam.Rainongrid.dt = abs(XParam.Rainongrid.to - XParam.Rainongrid.tmax) / (XParam.Rainongrid.nt - 1);


			int readfirststep = min(max((int)floor((XParam.totaltime - XParam.Rainongrid.to) / XParam.Rainongrid.dt), 0), XParam.Rainongrid.nt - 2);



			readATMstep(XParam.Rainongrid, readfirststep, Rainbef);
			readATMstep(XParam.Rainongrid, readfirststep + 1, Rainaft);

			InterpstepCPU(XParam.Rainongrid.nx, XParam.Rainongrid.ny, readfirststep, XParam.totaltime, XParam.Rainongrid.dt, Rain, Rainbef, Rainaft);


			if (XParam.GPUDEVICE >= 0)
			{
				//setup GPU texture to streamline interpolation between the two array
				Allocate1GPU(XParam.Rainongrid.nx, XParam.Rainongrid.ny, Rain_g);
				Allocate1GPU(XParam.Rainongrid.nx, XParam.Rainongrid.ny, Rainbef_g);
				Allocate1GPU(XParam.Rainongrid.nx, XParam.Rainongrid.ny, Rainaft_g);



				CUDA_CHECK(cudaMemcpy(Rain_g, Rain, XParam.Rainongrid.nx*XParam.Rainongrid.ny * sizeof(float), cudaMemcpyHostToDevice));
				CUDA_CHECK(cudaMemcpy(Rainaft_g, Rainaft, XParam.Rainongrid.nx*XParam.Rainongrid.ny * sizeof(float), cudaMemcpyHostToDevice));
				CUDA_CHECK(cudaMemcpy(Rainbef_g, Rainbef, XParam.Rainongrid.nx*XParam.Rainongrid.ny * sizeof(float), cudaMemcpyHostToDevice));


				//
				CUDA_CHECK(cudaMallocArray(&Rain_gp, &channelDescRain, XParam.Rainongrid.nx, XParam.Rainongrid.ny));


				CUDA_CHECK(cudaMemcpyToArray(Rain_gp, 0, 0, Rain, XParam.Rainongrid.nx * XParam.Rainongrid.ny * sizeof(float), cudaMemcpyHostToDevice));

				texRAIN.addressMode[0] = cudaAddressModeClamp;
				texRAIN.addressMode[1] = cudaAddressModeClamp;
				texRAIN.filterMode = cudaFilterModeLinear;
				texRAIN.normalized = false;


				CUDA_CHECK(cudaBindTextureToArray(texRAIN, Rain_gp, channelDescRain));






			}
		}




	}

	//////////////////////////////////////////////////////////////////////////////////
	// Initial adaptation
	//////////////////////////////////////////////////////////////////////////////////

	int oldnblk = 0;
	if (XParam.maxlevel != XParam.minlevel)
	{
		while (oldnblk != XParam.nblk)
		//for (int i=0; i<1;i++)
		{
			oldnblk = XParam.nblk;
			//wetdrycriteria(XParam, refine, coarsen);
			inrangecriteria(XParam, -10.0f, 10.0f, refine, coarsen, zb);
			refinesanitycheck(XParam, refine, coarsen);
			XParam = adapt(XParam);
			if (!checkBUQsanity(XParam))
			{
				printf("Bad BUQ mesh layout\n");
				exit(2);
				//break;
			}

			
		}
		
	}
	//gradientADA(XParam.nblk, XParam.blksize, (float)XParam.theta, (float)XParam.dx, activeblk, level, leftblk, rightblk, topblk, botblk, zb, uu, vv);
	// Debugging...
	/*
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				int i = ix + iy * 16 + ib * XParam.blksize;

				hh[i] = ib;
				zs[i] = leftblk[ib];
				zb[i] = botblk[ib];
				uu[i] = rightblk[ib];
				vv[i] = topblk[ib];


			}
		}
	}
	
	*/

	///////////////////////////////////////////////////////////////////////////////////
	// Prepare various model outputs
	///////////////////////////////////////////////////////////////////////////////////


	//Check that if timeseries output nodes are specified that they are within nx and ny
	if (XParam.TSnodesout.size() > 0)
	{
		for (int o = 0; o < XParam.TSnodesout.size(); o++)
		{


			//find the block where point belongs
			for (int blk = 0; blk < XParam.nblk; blk++)
			{
				//
				if (XParam.TSnodesout[o].x >= blockxo_d[blk] && XParam.TSnodesout[o].x <= (blockxo_d[blk] + 16.0*levdx) && XParam.TSnodesout[o].y >= blockyo_d[o] && XParam.TSnodesout[o].y <= (blockyo_d[blk] + 16.0*levdx))
				{
					XParam.TSnodesout[o].block = blk;
					XParam.TSnodesout[o].i = min(max((int)round((XParam.TSnodesout[o].x - blockxo_d[blk]) / levdx), 0), 15);
					XParam.TSnodesout[o].j = min(max((int)round((XParam.TSnodesout[o].y - blockyo_d[blk]) / levdx), 0), 15);
					break;
				}
			}

		}

	}

	// Here map array to their name as a string. it makes it super easy to convert user define variables to the array it represents.
	// One could add more to output gradients etc...
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
	//XParam=creatncfileUD(XParam);
	XParam = creatncfileBUQ(XParam);
	//creatncfileBUQ(XParam);
	for (int ivar = 0; ivar < XParam.outvars.size(); ivar++)
	{
		//Create definition for each variable and store it
		if (XParam.doubleprecision == 1 || XParam.spherical == 1)
		{
			//defncvarD(XParam.outfile, XParam.smallnc, XParam.scalefactor, XParam.addoffset, nx, ny, XParam.outvars[ivar], 3, OutputVarMapCPUD[XParam.outvars[ivar]]);
			//defncvar(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], 3, OutputVarMapCPUD[XParam.outvars[ivar]]);
			defncvarBUQ(XParam, activeblk,level,blockxo_d, blockyo_d, XParam.outvars[ivar], 3, OutputVarMapCPUD[XParam.outvars[ivar]]);
		}
		else
		{
			//defncvar(XParam, blockxo_d, blockyo_d, XParam.outvars[ivar], 3, OutputVarMapCPU[XParam.outvars[ivar]]);
			defncvarBUQ(XParam, activeblk, level, blockxo_d, blockyo_d, XParam.outvars[ivar], 3, OutputVarMapCPU[XParam.outvars[ivar]]);
		}

	}
	/*
	defncvarBUQ(XParam, activeblk, level, blockxo_d, blockyo_d, "dhdx", 3, dhdx);
	defncvarBUQ(XParam, activeblk, level, blockxo_d, blockyo_d, "dhdy", 3, dhdy);
	defncvarBUQ(XParam, activeblk, level, blockxo_d, blockyo_d, "dzsdx", 3, dzsdx);
	defncvarBUQ(XParam, activeblk, level, blockxo_d, blockyo_d, "dzsdy", 3, dzsdy);
	defncvarBUQ(XParam, activeblk, level, blockxo_d, blockyo_d, "dudx", 3, dudx);
	defncvarBUQ(XParam, activeblk, level, blockxo_d, blockyo_d, "dudy", 3, dudy);
	

	defncvarBUQ(XParam, activeblk, level, blockxo_d, blockyo_d, "Fhu", 3, Fhu);
	defncvarBUQ(XParam, activeblk, level, blockxo_d, blockyo_d, "Fhv", 3, Fhv);
	*/

	//create2dnc(nx, ny, dx, dx, 0.0, xx, yy, hh);

	printf("done \n");
	write_text_to_log_file("Done ");


	SaveParamtolog(XParam);

#ifdef USE_CATALYST
        if (XParam.use_catalyst)
        {
                // Retrieve adaptor and initialise visualisation/VTK output pipeline
                catalystAdaptor& adaptor = catalystAdaptor::getInstance();
                if (XParam.catalyst_python_pipeline)
                {
                        if (adaptor.initialiseWithPython(XParam.python_pipeline))
                        {
                                fprintf(stderr, "catalystAdaptor::initialiseWithPython failed");
                        }
                }
                else
                {
                        if (adaptor.initialiseVTKOutput(XParam.vtk_output_frequency, XParam.vtk_output_time_interval, XParam.vtk_outputfile_root))
                        {
                                fprintf(stderr, "catalystAdaptor::initialiseVTKOutput failed");
                        }
                }
        }
#endif

	/////////////////////////////////////
	////      STARTING MODEL     ////////
	/////////////////////////////////////

	if (XParam.maxlevel == XParam.minlevel)
	{
		XParam.delta = levdx;

		printf("\n###################################\n   Starting Model.\n###################################\n \n");
		write_text_to_log_file("Starting Model. ");

		if (XParam.GPUDEVICE >= 0)
		{
			if (XParam.spherical == 1)
			{
				mainloopGPUDSPH(XParam);
			}
			else if (XParam.doubleprecision == 1)
			{
				mainloopGPUDB(XParam);
			}
			else
			{
				if (!XParam.windU.inputfile.empty())
				{
					//
					mainloopGPUATM(XParam);
				}
				else
				{
					mainloopGPU(XParam);
				}
			}
			//checkGradGPU(XParam);

		}
		else
		{
			mainloopCPU(XParam);
		}


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

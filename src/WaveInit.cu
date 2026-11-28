//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
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

#include "WaveInit.h"


/**
 * @brief Initialise the wave model on the CPU (memory, spectral grid, initial conditions, time step and outputs).
 *
 * Wave equivalent of InitMesh + InitialConditions for the flow. It must be called after the flow initial conditions
 * and the initial adaptation (the wave model uses the final flow mesh, bathymetry and water depth).
 *
 * @tparam T Data type (float or double)
 * @param XParam Flow model parameters (mesh information)
 * @param XWaveParam Wave model parameters (specstride is set here)
 * @param XForcing Forcing structure
 * @param XModel Flow model (CPU)
 * @param XWave Wave model (CPU)
 */
template <class T> void InitWaveModel(Param XParam, WaveParam& XWaveParam, Forcing<float>& XForcing, Model<T>& XModel, WaveModel<T>& XWave)
{
	if (!XWaveParam.wavemodel)
	{
		return;
	}

	log("\nInitializing wave model");

	//==============================
	// Memory layout and size check
	XWaveParam.specstride = XParam.nblkmem * XParam.blksize;

	size_t nspecmem = size_t(XWaveParam.nspec) * size_t(XWaveParam.specstride);
	if (nspecmem > size_t(std::numeric_limits<int>::max()))
	{
		log("ERROR: The wave spectral arrays are too large (" + std::to_string(nspecmem) + " values). Reduce the number of spectral bins (nfreq, ndir) or the size of the domain.");
		exit(1);
	}

	// 3 spectral arrays + 3 frequency arrays (k, cg, dsdd) + 4 batch work arrays + 34 2D arrays
	double memMB = double(3 * nspecmem + (3 * size_t(XWaveParam.nfreq) + 4 * size_t(XWaveParam.wavemodel == 2 ? XWaveParam.nchunk : 1) + 38) * size_t(XWaveParam.specstride)) * sizeof(T) / 1024.0 / 1024.0;
	log("\tSpectral grid: " + std::to_string(XWaveParam.nfreq) + " frequencies x " + std::to_string(XWaveParam.ndir) + " directions; " + std::string(XWaveParam.wavemodel == 1 ? "implicit solver" : "explicit solver, " + std::to_string(XWaveParam.nchunk) + " bins propagated per batch") + "; Wave model memory: " + std::to_string(memMB) + " MB");
	if (XWaveParam.staticmedium)
	{
		log("\tStatic medium: the wave kinematics and time step are only calculated once");
	}

	//==============================
	// Allocate CPU memory
	AllocateCPU(XParam.nblkmem, XParam.blksize, XWaveParam, XWave);

	//==============================
	// Spectral grid
	InitWaveSpecGrid(XWaveParam, XWave.spec);
	XWave.spec_h = XWave.spec;

	//==============================
	// Virtual blocks for the batches of spectral bins
	InitWaveVirtualBlocks(XParam, XWaveParam, XModel.blocks, XWave.vblocks);

	//==============================
	// Boundary blocks of each wave boundary segment (same function as for the flow boundaries)
	Initsegbndblks(XParam, XForcing.wavebndseg, XModel.blocks, "Wave boundary segment");
	for (int iseg = 0; iseg < XForcing.wavebndseg.size(); iseg++)
	{
		wavebndsegment& seg = XForcing.wavebndseg[iseg];
		std::vector<std::string> typestr = { "absorbing", "Neumann", "JONSWAP" };
		log("\tWave boundary segment " + std::to_string(iseg) + " (" + seg.polyfile + "): " + typestr[seg.type] + (seg.on ? (seg.uniform ? " (time series: " : " (maps: ") + seg.inputfile + ")" : std::string("")));
	}

	//==============================
	// Implicit solver (sweep orders, boundary faces)
	if (XWaveParam.wavemodel == 1)
	{
		InitWaveImplicit(XParam, XWaveParam, XForcing, XModel, XWave);
	}

	//==============================
	// Initial conditions
	InitWaveConditions(XParam, XWaveParam, XModel, XWave);

	//==============================
	// Time keeping
	XWave.clock.totaltime = XParam.totaltime;
	// The implicit solver is first run at the first step (stationary solution at the start), then every dtcouple
	XWave.clock.nextcoupletime = (XWaveParam.wavemodel == 1) ? XParam.totaltime : XParam.totaltime + XWaveParam.dtcouple;
	XWave.clock.nstep = 0;
	// The implicit solver is unconditionally stable: its time step is the coupling interval
	XWave.clock.dtmax = (XWaveParam.wavemodel == 1) ? XWaveParam.dtcouple : CalcWavedtmaxCPU(XParam, XWaveParam, XModel, XWave);
	XWave.clock.dt = XWave.clock.dtmax;

	log("\tWave model initial (max) time step: " + std::to_string(XWave.clock.dtmax) + " s");

	//==============================
	// Register the wave variables for map outputs (see Initmaparray in InitialConditions.cu)
	Initmaparray(XModel, XWave);

}
template void InitWaveModel<float>(Param XParam, WaveParam& XWaveParam, Forcing<float>& XForcing, Model<float>& XModel, WaveModel<float>& XWave);
template void InitWaveModel<double>(Param XParam, WaveParam& XWaveParam, Forcing<float>& XForcing, Model<double>& XModel, WaveModel<double>& XWave);


/**
 * @brief Build the virtual block topology used to process a batch of nchunk spectral bins in a single call.
 *
 * In the action density array, the bin j (relative to the first bin of a batch) of block ib has exactly the layout of
 * a flow variable in the block ib + j*nblkmem. vblocks replicates the block information nchunk times with the
 * neighbours shifted by j*nblkmem, and its active list holds the nchunk copies of the active blocks
 * (active[j*nblk + ibl] = active[ibl] + j*nblkmem). With XParam.nblk = nblk*nbins, the existing block functions
 * (fillHaloC/GPU, gradientC/gradient, gradientHalo, fillHaloLRFluxC, ...) then process nbins bins at once.
 * The domain boundaries (self neighbours) are preserved.
 */
template <class T> void InitWaveVirtualBlocks(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, BlockP<T>& vblocks)
{
	int nb = XParam.nblkmem;
	int nc = XWaveParam.nchunk;

	AllocateCPU(XParam.nblk * nc, 1, vblocks.active);
	AllocateCPU(nb * nc, 1, vblocks.level);
	AllocateCPU(nb * nc, 1, vblocks.xo);
	AllocateCPU(nb * nc, 1, vblocks.yo);
	AllocateCPU(nb * nc, 1, vblocks.BotLeft, vblocks.BotRight, vblocks.TopLeft, vblocks.TopRight);
	AllocateCPU(nb * nc, 1, vblocks.LeftBot, vblocks.LeftTop, vblocks.RightBot, vblocks.RightTop);

	for (int j = 0; j < nc; j++)
	{
		for (int ibl = 0; ibl < XParam.nblk; ibl++)
		{
			vblocks.active[j * XParam.nblk + ibl] = XBlock.active[ibl] + j * nb;
		}
		for (int ib = 0; ib < nb; ib++)
		{
			int iv = ib + j * nb;
			vblocks.level[iv] = XBlock.level[ib];
			vblocks.xo[iv] = XBlock.xo[ib];
			vblocks.yo[iv] = XBlock.yo[ib];
			vblocks.BotLeft[iv] = XBlock.BotLeft[ib] + j * nb;
			vblocks.BotRight[iv] = XBlock.BotRight[ib] + j * nb;
			vblocks.TopLeft[iv] = XBlock.TopLeft[ib] + j * nb;
			vblocks.TopRight[iv] = XBlock.TopRight[ib] + j * nb;
			vblocks.LeftBot[iv] = XBlock.LeftBot[ib] + j * nb;
			vblocks.LeftTop[iv] = XBlock.LeftTop[ib] + j * nb;
			vblocks.RightBot[iv] = XBlock.RightBot[ib] + j * nb;
			vblocks.RightTop[iv] = XBlock.RightTop[ib] + j * nb;
		}
	}
}
template void InitWaveVirtualBlocks<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, BlockP<float>& vblocks);
template void InitWaveVirtualBlocks<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, BlockP<double>& vblocks);


/**
 * @brief Allocate and copy the wave model to the GPU (bypassed if no GPU is used).
 *
 * Wave equivalent of SetupGPU. Must be called after SetupGPU (since SetupGPU resets XModel_g.OutputVarMap).
 *
 * @tparam T Data type (float or double)
 * @param XParam Flow model parameters
 * @param XWaveParam Wave model parameters
 * @param XForcing Forcing (the wave boundary segment block lists are copied to the GPU)
 * @param XModel_g Flow model (GPU) the wave output variables are registered in
 * @param XWave Wave model (CPU)
 * @param XWave_g Wave model (GPU)
 */
template <class T> void SetupWaveGPU(Param XParam, WaveParam XWaveParam, Forcing<float>& XForcing, Model<T>& XModel_g, WaveModel<T>& XWave, WaveModel<T>& XWave_g)
{
	if (!XWaveParam.wavemodel || XParam.GPUDEVICE < 0)
	{
		return;
	}
	log("Setting up wave model on GPU");

	AllocateGPU(XParam.nblkmem, XParam.blksize, XWaveParam, XWave_g);
	CopytoGPU(XParam.nblkmem, XParam.blksize, XWaveParam, XWave, XWave_g);

	XWave_g.clock = XWave.clock;
	XWave_g.spec_h = XWave.spec;

	if (XWaveParam.wavemodel == 1)
	{
		SetupWaveImplicitGPU(XParam, XWaveParam, XWave, XWave_g);
	}

	// Virtual blocks for the batches of spectral bins
	int nv = XParam.nblkmem * XWaveParam.nchunk;
	AllocateGPU(XParam.nblk * XWaveParam.nchunk, 1, XWave_g.vblocks.active);
	CopytoGPU(XParam.nblk * XWaveParam.nchunk, 1, XWave.vblocks.active, XWave_g.vblocks.active);
	AllocateGPU(nv, 1, XWave_g.vblocks.level);
	CopytoGPU(nv, 1, XWave.vblocks.level, XWave_g.vblocks.level);
	AllocateGPU(nv, 1, XWave_g.vblocks.xo);
	CopytoGPU(nv, 1, XWave.vblocks.xo, XWave_g.vblocks.xo);
	AllocateGPU(nv, 1, XWave_g.vblocks.yo);
	CopytoGPU(nv, 1, XWave.vblocks.yo, XWave_g.vblocks.yo);
	int** nbcpu[8] = { &XWave.vblocks.BotLeft, &XWave.vblocks.BotRight, &XWave.vblocks.TopLeft, &XWave.vblocks.TopRight, &XWave.vblocks.LeftBot, &XWave.vblocks.LeftTop, &XWave.vblocks.RightBot, &XWave.vblocks.RightTop };
	int** nbgpu[8] = { &XWave_g.vblocks.BotLeft, &XWave_g.vblocks.BotRight, &XWave_g.vblocks.TopLeft, &XWave_g.vblocks.TopRight, &XWave_g.vblocks.LeftBot, &XWave_g.vblocks.LeftTop, &XWave_g.vblocks.RightBot, &XWave_g.vblocks.RightTop };
	for (int in = 0; in < 8; in++)
	{
		AllocateGPU(nv, 1, *nbgpu[in]);
		CopytoGPU(nv, 1, *nbcpu[in], *nbgpu[in]);
	}

	// Wave boundary segments block lists (as for the flow bndseg in SetupGPU)
	for (int s = 0; s < XForcing.wavebndseg.size(); s++)
	{
		bndsegmentside* sides[4] = { &XForcing.wavebndseg[s].left, &XForcing.wavebndseg[s].right, &XForcing.wavebndseg[s].top, &XForcing.wavebndseg[s].bot };
		for (int is = 0; is < 4; is++)
		{
			AllocateGPU(utils::max(sides[is]->nblk, 1), 1, sides[is]->blk_g);
			if (sides[is]->nblk > 0)
			{
				CopytoGPU(sides[is]->nblk, 1, sides[is]->blk, sides[is]->blk_g);
			}
		}
	}

	Initmaparray(XModel_g, XWave_g);
}
template void SetupWaveGPU<float>(Param XParam, WaveParam XWaveParam, Forcing<float>& XForcing, Model<float>& XModel_g, WaveModel<float>& XWave, WaveModel<float>& XWave_g);
template void SetupWaveGPU<double>(Param XParam, WaveParam XWaveParam, Forcing<float>& XForcing, Model<double>& XModel_g, WaveModel<double>& XWave, WaveModel<double>& XWave_g);


/**
 * @brief Initialise the spectral grid.
 *
 * Frequencies are logarithmically distributed: f_i = fmin * freqfactor^i (as in SWAN and WW3)
 * and the bin width is dsigma_i = 0.5 * sigma_i * (freqfactor - 1/freqfactor) (as in WW3).
 * If nfreq == 1 (monochromatic / single representative frequency) dsigma = 1.0 so that E(theta) = sigma*N(theta).
 *
 * Directions are stored in cartesian convention (direction of propagation, counter-clockwise from the x axis):
 * theta_j = theta_0 + j*dtheta where theta_0 is the cartesian equivalent of the nautical dirstart.
 *
 * @tparam T Data type (float or double)
 * @param XWaveParam Wave model parameters
 * @param XSpec Spectral grid (CPU)
 */
template <class T> void InitWaveSpecGrid(WaveParam XWaveParam, WaveSpecGridP<T>& XSpec)
{
	double twopi = 2.0 * pi;
	double xfr = XWaveParam.freqfactor;
	for (int ifreq = 0; ifreq < XWaveParam.nfreq; ifreq++)
	{
		double f = XWaveParam.fmin * pow(xfr, double(ifreq));
		XSpec.sigma[ifreq] = T(twopi * f);
		XSpec.dsigma[ifreq] = (XWaveParam.nfreq > 1) ? T(0.5 * twopi * f * (xfr - 1.0 / xfr)) : T(1.0);
	}

	double theta0 = (270.0 - XWaveParam.dirstart) * pi / 180.0;
	for (int idir = 0; idir < XWaveParam.ndir; idir++)
	{
		double th = theta0 + idir * XWaveParam.dtheta;
		th = th - twopi * floor(th / twopi);
		XSpec.theta[idir] = T(th);
		XSpec.costh[idir] = T(cos(th));
		XSpec.sinth[idir] = T(sin(th));
	}
}
template void InitWaveSpecGrid<float>(WaveParam XWaveParam, WaveSpecGridP<float>& XSpec);
template void InitWaveSpecGrid<double>(WaveParam XWaveParam, WaveSpecGridP<double>& XSpec);


/**
 * @brief Build a discrete JONSWAP action density spectrum N(sigma,theta) on the model spectral grid.
 *
 * E(f) ~ f^-5 exp(-5/4 (fp/f)^4) gamma^exp(-(f-fp)^2/(2 s^2 fp^2)) (s = 0.07 for f<=fp and 0.09 for f>fp)
 * D(theta) ~ cos^2s((theta-theta_m)/2) with s = 2/dspr^2 - 1 (dspr in rad, Kuik et al. 1988)
 * The discrete spectrum is scaled so that 4*sqrt(m0) = Hs exactly on the model spectral grid.
 * This is used for the initial condition (waveinit=1). It uses WaveJonswapN, the function used for the JONSWAP boundaries.
 *
 * @tparam T Data type (float or double)
 * @param XWaveParam Wave model parameters
 * @param XSpec Spectral grid (CPU)
 * @param Hs Significant wave height (m)
 * @param Tp Peak period (s)
 * @param Dir Mean direction (degrees nautical, coming from)
 * @param Dspr Directional spreading (degrees)
 * @param gamma JONSWAP peak enhancement factor
 * @param Nspec Output action density for each spectral bin, size(nspec)
 */
template <class T> void JonswapSpectrum(WaveParam XWaveParam, WaveSpecGridP<T> XSpec, double Hs, double Tp, double Dir, double Dspr, double gamma, std::vector<T>& Nspec)
{
	// Same function as the one used for the JONSWAP boundaries (WaveJonswapN)
	Nspec.assign(XWaveParam.nspec, T(0.0));
	for (int ispec = 0; ispec < XWaveParam.nspec; ispec++)
	{
		Nspec[ispec] = WaveJonswapN(XWaveParam, XSpec, ispec, T(Hs), T(Tp), T(Dir), T(Dspr), T(gamma));
	}
}
template void JonswapSpectrum<float>(WaveParam XWaveParam, WaveSpecGridP<float> XSpec, double Hs, double Tp, double Dir, double Dspr, double gamma, std::vector<float>& Nspec);
template void JonswapSpectrum<double>(WaveParam XWaveParam, WaveSpecGridP<double> XSpec, double Hs, double Tp, double Dir, double Dspr, double gamma, std::vector<double>& Nspec);


/**
 * @brief Initialise the wave model arrays (action density, bulk parameters, dissipation and coupling arrays).
 *
 * waveinit = 0: calm sea (N = 0)
 * waveinit = 1: uniform JONSWAP spectrum (Hsinit, Tpinit, Dirinit, Dsprinit) in all wet cells
 *
 * @tparam T Data type (float or double)
 */
template <class T> void InitWaveConditions(Param XParam, WaveParam XWaveParam, Model<T> XModel, WaveModel<T>& XWave)
{
	int nblkmem = XParam.nblkmem;
	int blksize = XParam.blksize;
	int nspec = XWaveParam.nspec;

	//==============================
	// Set everything (including halo and unused blocks) to zero
	FillCPU(nblkmem * nspec, blksize, T(0.0), XWave.action.N);
	FillCPU(nblkmem * nspec, blksize, T(0.0), XWave.adv.dN);

	std::vector<T*> arr2D = { XWave.bulk.Hs, XWave.bulk.Tm01, XWave.bulk.Tm02, XWave.bulk.Tp, XWave.bulk.Dir, XWave.bulk.Dspr, XWave.bulk.km, XWave.bulk.Urms, XWave.bulk.Qb, XWave.bulk.Tmm10, XWave.bulk.Dp,
		XWave.dissip.Dbr, XWave.dissip.Dfr, XWave.dissip.Dwc,
		XWave.force.Sxx, XWave.force.Sxy, XWave.force.Syy, XWave.force.Fx, XWave.force.Fy,
		XWave.time.dtmax, XWave.time.arrmin,
		XWave.medium.h, XWave.medium.ho, XWave.medium.dhdt, XWave.medium.u, XWave.medium.v,
		XWave.medium.dhdx, XWave.medium.dhdy, XWave.medium.dudx, XWave.medium.dudy, XWave.medium.dvdx, XWave.medium.dvdy,
		XWave.medium.uwind, XWave.medium.vwind };
	for (int ia = 0; ia < arr2D.size(); ia++)
	{
		FillCPU(nblkmem, blksize, T(0.0), arr2D[ia]);
	}
	std::vector<T*> arrchunk = { XWave.work.dNdx, XWave.work.dNdy, XWave.work.Fx, XWave.work.Fy };
	int nwork = (XWaveParam.wavemodel == 2) ? XWaveParam.nchunk : 1;
	for (int ia = 0; ia < arrchunk.size(); ia++)
	{
		FillCPU(nblkmem * nwork, blksize, T(0.0), arrchunk[ia]);
	}
	FillCPU(nblkmem * XWaveParam.nfreq, blksize, T(0.0), XWave.kin.k);
	FillCPU(nblkmem * XWaveParam.nfreq, blksize, T(0.0), XWave.kin.cg);
	FillCPU(nblkmem * XWaveParam.nfreq, blksize, T(0.0), XWave.kin.dsdd);

	//==============================
	// Medium (depth, currents) and kinematics: dh/dt = 0 at the start
	Param XParamW = WaveLightParam(XParam, XParam.theta);
	WavePrepCPU(XParamW, XWaveParam, 0.0, XModel, XWave);


	//==============================
	// Initial spectrum
	if (XWaveParam.waveinit == 1)
	{
		log("\tWave initial condition: uniform JONSWAP spectrum Hs=" + std::to_string(XWaveParam.Hsinit) + "m, Tp=" + std::to_string(XWaveParam.Tpinit) + "s, Dir=" + std::to_string(XWaveParam.Dirinit) + "deg, Dspr=" + std::to_string(XWaveParam.Dsprinit) + "deg");

		std::vector<T> Nspec;
		JonswapSpectrum(XWaveParam, XWave.spec, XWaveParam.Hsinit, XWaveParam.Tpinit, XWaveParam.Dirinit, XWaveParam.Dsprinit, XWaveParam.gammajsp, Nspec);

		int ib;
		for (int ibl = 0; ibl < XParam.nblk; ibl++)
		{
			ib = XModel.blocks.active[ibl];
			for (int iy = 0; iy < XParam.blkwidth; iy++)
			{
				for (int ix = 0; ix < XParam.blkwidth; ix++)
				{
					int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
					T h = WaveDepth(XParam, XWaveParam, i, XModel.evolv, XModel.zb);
					if (h > T(XWaveParam.wavedepthmin))
					{
						for (int ispec = 0; ispec < nspec; ispec++)
						{
							XWave.action.N[specloc(XWaveParam.specstride, ispec, i)] = Nspec[ispec];
						}
					}
				}
			}
		}
	}
	else
	{
		log("\tWave initial condition: calm sea");
	}

	memcpy(XWave.action_o.N, XWave.action.N, size_t(nspec) * size_t(XWaveParam.specstride) * sizeof(T));

	//==============================
	// Initial bulk parameters
	CalcWaveBulkCPU(XParam, XWaveParam, XModel, XWave);

}
template void InitWaveConditions<float>(Param XParam, WaveParam XWaveParam, Model<float> XModel, WaveModel<float>& XWave);
template void InitWaveConditions<double>(Param XParam, WaveParam XWaveParam, Model<double> XModel, WaveModel<double>& XWave);



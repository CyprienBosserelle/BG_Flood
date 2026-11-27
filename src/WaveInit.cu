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

	// 3 spectral arrays + 2 frequency arrays (k, cg) + 36 2D arrays
	double memMB = double(3 * nspecmem + 2 * size_t(XWaveParam.nfreq) * size_t(XWaveParam.specstride) + 36 * size_t(XWaveParam.specstride)) * sizeof(T) / 1024.0 / 1024.0;
	log("\tSpectral grid: " + std::to_string(XWaveParam.nfreq) + " frequencies x " + std::to_string(XWaveParam.ndir) + " directions; Wave model memory: " + std::to_string(memMB) + " MB");

	//==============================
	// Allocate CPU memory
	AllocateCPU(XParam.nblkmem, XParam.blksize, XWaveParam, XWave);

	//==============================
	// Spectral grid
	InitWaveSpecGrid(XWaveParam, XWave.spec);

	//==============================
	// Initial conditions
	InitWaveConditions(XParam, XWaveParam, XModel, XWave);

	//==============================
	// Time keeping
	XWave.clock.totaltime = XParam.totaltime;
	XWave.clock.nextcoupletime = XParam.totaltime + XWaveParam.dtcouple;
	XWave.clock.nstep = 0;
	XWave.clock.dtmax = CalcWavedtmaxCPU(XParam, XWaveParam, XModel, XWave);
	XWave.clock.dt = XWave.clock.dtmax;

	log("\tWave model initial (max) time step: " + std::to_string(XWave.clock.dtmax) + " s");

	//==============================
	// Register the wave variables for map outputs
	InitWavemaparray(XModel, XWave);

}
template void InitWaveModel<float>(Param XParam, WaveParam& XWaveParam, Forcing<float>& XForcing, Model<float>& XModel, WaveModel<float>& XWave);
template void InitWaveModel<double>(Param XParam, WaveParam& XWaveParam, Forcing<float>& XForcing, Model<double>& XModel, WaveModel<double>& XWave);


/**
 * @brief Allocate and copy the wave model to the GPU (bypassed if no GPU is used).
 *
 * Wave equivalent of SetupGPU. Must be called after SetupGPU (since SetupGPU resets XModel_g.OutputVarMap).
 *
 * @tparam T Data type (float or double)
 * @param XParam Flow model parameters
 * @param XWaveParam Wave model parameters
 * @param XModel_g Flow model (GPU) the wave output variables are registered in
 * @param XWave Wave model (CPU)
 * @param XWave_g Wave model (GPU)
 */
template <class T> void SetupWaveGPU(Param XParam, WaveParam XWaveParam, Model<T>& XModel_g, WaveModel<T>& XWave, WaveModel<T>& XWave_g)
{
	if (!XWaveParam.wavemodel || XParam.GPUDEVICE < 0)
	{
		return;
	}
	log("Setting up wave model on GPU");

	AllocateGPU(XParam.nblkmem, XParam.blksize, XWaveParam, XWave_g);
	CopytoGPU(XParam.nblkmem, XParam.blksize, XWaveParam, XWave, XWave_g);

	XWave_g.clock = XWave.clock;

	InitWavemaparray(XModel_g, XWave_g);
}
template void SetupWaveGPU<float>(Param XParam, WaveParam XWaveParam, Model<float>& XModel_g, WaveModel<float>& XWave, WaveModel<float>& XWave_g);
template void SetupWaveGPU<double>(Param XParam, WaveParam XWaveParam, Model<double>& XModel_g, WaveModel<double>& XWave, WaveModel<double>& XWave_g);


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
 * This is used for the initial condition (waveinit=1) and will be used for parametric boundary conditions.
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
	int nfreq = XWaveParam.nfreq;
	int ndir = XWaveParam.ndir;
	double dth = XWaveParam.dtheta;
	double twopi = 2.0 * pi;

	Nspec.assign(XWaveParam.nspec, T(0.0));

	if (Hs <= 0.0 || Tp <= 0.0)
	{
		return;
	}

	// Frequency distribution E(sigma) (unscaled)
	double fp = 1.0 / Tp;
	std::vector<double> Esig(nfreq, 0.0);
	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	{
		double f = double(XSpec.sigma[ifreq]) / twopi;
		double sj = (f <= fp) ? 0.07 : 0.09;
		double r = exp(-1.0 * (f - fp) * (f - fp) / (2.0 * sj * sj * fp * fp));
		double Ef = pow(f, -5.0) * exp(-1.25 * pow(fp / f, 4.0)) * pow(gamma, r);
		Esig[ifreq] = Ef / twopi;
	}
	if (nfreq == 1)
	{
		Esig[0] = 1.0; // single representative frequency: all the energy in the only bin
	}

	// Directional distribution D(theta) normalised so that sum(D dtheta) = 1
	double thm = (270.0 - Dir) * pi / 180.0;
	double dsprrad = Dspr * pi / 180.0;
	double spow = utils::max(2.0 / (dsprrad * dsprrad) - 1.0, 0.0);
	std::vector<double> D(ndir, 0.0);
	double Dsum = 0.0;
	for (int idir = 0; idir < ndir; idir++)
	{
		D[idir] = pow(std::abs(cos(0.5 * (double(XSpec.theta[idir]) - thm))), 2.0 * spow);
		Dsum += D[idir] * dth;
	}
	if (Dsum <= 0.0)
	{
		return;
	}

	// Scale to Hs
	double m0 = 0.0;
	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	{
		m0 += Esig[ifreq] * double(XSpec.dsigma[ifreq]);
	}
	double scale = (Hs * Hs / 16.0) / m0;

	for (int ifreq = 0; ifreq < nfreq; ifreq++)
	{
		for (int idir = 0; idir < ndir; idir++)
		{
			double E = scale * Esig[ifreq] * D[idir] / Dsum;
			Nspec[ifreq * ndir + idir] = T(E / double(XSpec.sigma[ifreq]));
		}
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

	std::vector<T*> arr2D = { XWave.bulk.Hs, XWave.bulk.Tm01, XWave.bulk.Tm02, XWave.bulk.Tp, XWave.bulk.Dir, XWave.bulk.Dspr, XWave.bulk.km, XWave.bulk.Urms, XWave.bulk.Qb,
		XWave.dissip.Dbr, XWave.dissip.Dfr, XWave.dissip.Dwc,
		XWave.force.Sxx, XWave.force.Sxy, XWave.force.Syy, XWave.force.Fx, XWave.force.Fy,
		XWave.time.dtmax, XWave.time.arrmin,
		XWave.medium.h, XWave.medium.ho, XWave.medium.dhdt, XWave.medium.u, XWave.medium.v,
		XWave.medium.dhdx, XWave.medium.dhdy, XWave.medium.dudx, XWave.medium.dudy, XWave.medium.dvdx, XWave.medium.dvdy,
		XWave.work.dNdx, XWave.work.dNdy, XWave.work.Fx, XWave.work.Fy };
	for (int ia = 0; ia < arr2D.size(); ia++)
	{
		FillCPU(nblkmem, blksize, T(0.0), arr2D[ia]);
	}
	FillCPU(nblkmem * XWaveParam.nfreq, blksize, T(0.0), XWave.kin.k);
	FillCPU(nblkmem * XWaveParam.nfreq, blksize, T(0.0), XWave.kin.cg);

	//==============================
	// Medium (depth, currents) and kinematics: dh/dt = 0 at the start
	Param XParamW = WaveLightParam(XParam, XParam.theta);
	WavePrepCPU(XParamW, XWaveParam, 0.0, XModel, XWave);

	//==============================
	// Boundary spectrum (parametric JONSWAP)
	{
		std::vector<T> Nbnd;
		JonswapSpectrum(XWaveParam, XWave.spec, XWaveParam.Hsbnd, XWaveParam.Tpbnd, XWaveParam.Dirbnd, XWaveParam.Dsprbnd, XWaveParam.gammajsp, Nbnd);
		for (int ispec = 0; ispec < nspec; ispec++)
		{
			XWave.bnd.Nbnd[ispec] = Nbnd[ispec];
		}
		std::vector<int> bndtypes = { XWaveParam.leftwavebnd, XWaveParam.rightwavebnd, XWaveParam.topwavebnd, XWaveParam.botwavebnd };
		if (std::find(bndtypes.begin(), bndtypes.end(), 1) != bndtypes.end())
		{
			log("\tWave boundary: JONSWAP spectrum Hs=" + std::to_string(XWaveParam.Hsbnd) + "m, Tp=" + std::to_string(XWaveParam.Tpbnd) + "s, Dir=" + std::to_string(XWaveParam.Dirbnd) + "deg, Dspr=" + std::to_string(XWaveParam.Dsprbnd) + "deg (left/right/top/bot type: " + std::to_string(XWaveParam.leftwavebnd) + "/" + std::to_string(XWaveParam.rightwavebnd) + "/" + std::to_string(XWaveParam.topwavebnd) + "/" + std::to_string(XWaveParam.botwavebnd) + ")");
		}
	}

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


/**
 * @brief Register the wave (2D) variables in the flow output maps so the existing netcdf output machinery can be used.
 *
 * Wave equivalent of Initmaparray. Must be called again if the wave arrays are reallocated.
 *
 * @tparam T Data type (float or double)
 * @param XModel Flow model (CPU or GPU) holding the output maps
 * @param XWave Wave model (CPU or GPU)
 */
template <class T> void InitWavemaparray(Model<T>& XModel, WaveModel<T> XWave)
{
	XModel.OutputVarMap["Hs"] = XWave.bulk.Hs;
	XModel.Outvarlongname["Hs"] = "Significant wave height";
	XModel.Outvarstdname["Hs"] = "sea_surface_wave_significant_height";
	XModel.Outvarunits["Hs"] = "m";

	XModel.OutputVarMap["Tm01"] = XWave.bulk.Tm01;
	XModel.Outvarlongname["Tm01"] = "Mean wave period Tm01";
	XModel.Outvarstdname["Tm01"] = "sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment";
	XModel.Outvarunits["Tm01"] = "s";

	XModel.OutputVarMap["Tm02"] = XWave.bulk.Tm02;
	XModel.Outvarlongname["Tm02"] = "Mean wave period Tm02";
	XModel.Outvarstdname["Tm02"] = "sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment";
	XModel.Outvarunits["Tm02"] = "s";

	XModel.OutputVarMap["Tp"] = XWave.bulk.Tp;
	XModel.Outvarlongname["Tp"] = "Peak wave period";
	XModel.Outvarstdname["Tp"] = "sea_surface_wave_period_at_variance_spectral_density_maximum";
	XModel.Outvarunits["Tp"] = "s";

	XModel.OutputVarMap["Dir"] = XWave.bulk.Dir;
	XModel.Outvarlongname["Dir"] = "Mean wave direction (nautical, coming from)";
	XModel.Outvarstdname["Dir"] = "sea_surface_wave_from_direction";
	XModel.Outvarunits["Dir"] = "degree";

	XModel.OutputVarMap["Dspr"] = XWave.bulk.Dspr;
	XModel.Outvarlongname["Dspr"] = "Directional spreading";
	XModel.Outvarstdname["Dspr"] = "sea_surface_wave_directional_spread";
	XModel.Outvarunits["Dspr"] = "degree";

	XModel.OutputVarMap["km"] = XWave.bulk.km;
	XModel.Outvarlongname["km"] = "Mean wave number (based on Tm01)";
	XModel.Outvarstdname["km"] = "mean_wave_number";
	XModel.Outvarunits["km"] = "rad/m";

	XModel.OutputVarMap["Urms"] = XWave.bulk.Urms;
	XModel.Outvarlongname["Urms"] = "Root mean square near bottom orbital velocity";
	XModel.Outvarstdname["Urms"] = "rms_bottom_orbital_velocity";
	XModel.Outvarunits["Urms"] = "m/s";

	XModel.OutputVarMap["Qb"] = XWave.bulk.Qb;
	XModel.Outvarlongname["Qb"] = "Fraction of breaking waves";
	XModel.Outvarstdname["Qb"] = "fraction_of_breaking_waves";
	XModel.Outvarunits["Qb"] = "-";

	XModel.OutputVarMap["Dbr"] = XWave.dissip.Dbr;
	XModel.Outvarlongname["Dbr"] = "Wave energy dissipation by depth induced breaking";
	XModel.Outvarstdname["Dbr"] = "wave_dissipation_breaking";
	XModel.Outvarunits["Dbr"] = "W/m2";

	XModel.OutputVarMap["Dfr"] = XWave.dissip.Dfr;
	XModel.Outvarlongname["Dfr"] = "Wave energy dissipation by bottom friction";
	XModel.Outvarstdname["Dfr"] = "wave_dissipation_bottom_friction";
	XModel.Outvarunits["Dfr"] = "W/m2";

	XModel.OutputVarMap["Dwc"] = XWave.dissip.Dwc;
	XModel.Outvarlongname["Dwc"] = "Wave energy dissipation by whitecapping";
	XModel.Outvarstdname["Dwc"] = "wave_dissipation_whitecapping";
	XModel.Outvarunits["Dwc"] = "W/m2";

	XModel.OutputVarMap["Sxx"] = XWave.force.Sxx;
	XModel.Outvarlongname["Sxx"] = "Radiation stress component Sxx";
	XModel.Outvarstdname["Sxx"] = "radiation_stress_xx";
	XModel.Outvarunits["Sxx"] = "N/m";

	XModel.OutputVarMap["Sxy"] = XWave.force.Sxy;
	XModel.Outvarlongname["Sxy"] = "Radiation stress component Sxy";
	XModel.Outvarstdname["Sxy"] = "radiation_stress_xy";
	XModel.Outvarunits["Sxy"] = "N/m";

	XModel.OutputVarMap["Syy"] = XWave.force.Syy;
	XModel.Outvarlongname["Syy"] = "Radiation stress component Syy";
	XModel.Outvarstdname["Syy"] = "radiation_stress_yy";
	XModel.Outvarunits["Syy"] = "N/m";

	XModel.OutputVarMap["Fwx"] = XWave.force.Fx;
	XModel.Outvarlongname["Fwx"] = "Wave force in x-direction";
	XModel.Outvarstdname["Fwx"] = "wave_force_x";
	XModel.Outvarunits["Fwx"] = "N/m2";

	XModel.OutputVarMap["Fwy"] = XWave.force.Fy;
	XModel.Outvarlongname["Fwy"] = "Wave force in y-direction";
	XModel.Outvarstdname["Fwy"] = "wave_force_y";
	XModel.Outvarunits["Fwy"] = "N/m2";
}
template void InitWavemaparray<float>(Model<float>& XModel, WaveModel<float> XWave);
template void InitWavemaparray<double>(Model<double>& XModel, WaveModel<double> XWave);

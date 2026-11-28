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

#include "WaveCPU.h"


/**
 * @brief Solve the linear dispersion relation sigma^2 = g k tanh(k h) for the wave number k.
 *
 * Uses the explicit approximation of Fenton & McKee (1990) followed by 2 Newton-Raphson iterations
 * (relative error < 1e-10 for all depths).
 *
 * @tparam T Data type (float or double)
 * @param sigma Radian frequency (rad/s)
 * @param h Water depth (m)
 * @param g Acceleration of gravity (m/s2)
 * @return Wave number k (rad/m). 0 if h<=0
 */
template <class T> __host__ __device__ T WaveDispersion(T sigma, T h, T g)
{
	if (h <= T(0.0) || sigma <= T(0.0))
	{
		return T(0.0);
	}
	T kh0 = sigma * sigma * h / g; // deep water k0*h

	// Fenton & McKee (1990)
	T kh = kh0 * pow(T(1.0) / tanh(pow(kh0, T(0.75))), T(2.0 / 3.0));

	// Newton-Raphson iterations on f(kh) = kh*tanh(kh) - kh0
	for (int it = 0; it < 2; it++)
	{
		T th = tanh(kh);
		T f = kh * th - kh0;
		T df = th + kh * (T(1.0) - th * th);
		kh = kh - f / df;
	}

	return kh / h;
}
template __host__ __device__ float WaveDispersion<float>(float sigma, float h, float g);
template __host__ __device__ double WaveDispersion<double>(double sigma, double h, double g);


/**
 * @brief Linear wave group velocity cg = n*sigma/k with n = 0.5*(1 + 2kh/sinh(2kh)).
 *
 * @tparam T Data type (float or double)
 * @param sigma Radian frequency (rad/s)
 * @param k Wave number (rad/m)
 * @param h Water depth (m)
 * @return Group velocity (m/s)
 */
template <class T> __host__ __device__ T WaveGroupVelocity(T sigma, T k, T h)
{
	if (k <= T(0.0))
	{
		return T(0.0);
	}
	T kh = k * h;
	T n = (kh > T(20.0)) ? T(0.5) : T(0.5) * (T(1.0) + T(2.0) * kh / sinh(T(2.0) * kh));
	return n * sigma / k;
}
template __host__ __device__ float WaveGroupVelocity<float>(float sigma, float k, float h);
template __host__ __device__ double WaveGroupVelocity<double>(double sigma, double k, double h);


/**
 * @brief Water depth seen by the wave model at cell i.
 *
 * If wavewaterlevel is true (default) the flow model water depth is used (two-way water level coupling),
 * otherwise the depth is calculated from the still water level zsinit and the bathymetry.
 *
 * @tparam T Data type (float or double)
 * @param XParam Flow model parameters
 * @param XWaveParam Wave model parameters
 * @param i Cell memory index
 * @param XEv Flow evolving variables
 * @param zb Bed elevation
 * @return Water depth (m)
 */
template <class T> __host__ __device__ T WaveDepth(const Param& XParam, const WaveParam& XWaveParam, int i, const EvolvingP<T>& XEv, T* zb)
{
	T h;
	if (XWaveParam.wavewaterlevel)
	{
		h = XEv.h[i];
	}
	else
	{
		T zsref = (XParam.zsinit != XParam.zsinit) ? T(0.0) : T(XParam.zsinit); // (x != x) is true for NaN
		h = utils::max(zsref - zb[i], T(0.0));
	}
	return h;
}
template __host__ __device__ float WaveDepth<float>(const Param& XParam, const WaveParam& XWaveParam, int i, const EvolvingP<float>& XEv, float* zb);
template __host__ __device__ double WaveDepth<double>(const Param& XParam, const WaveParam& XWaveParam, int i, const EvolvingP<double>& XEv, double* zb);


/**
 * @brief Calculate the integrated (bulk) wave parameters and radiation stresses for a single cell.
 *
 * This function is shared between the CPU (CalcWaveBulkCPU) and the GPU (CalcWaveBulkGPU) versions.
 * With E(sigma,theta) = sigma*N(sigma,theta) and the moments m_n = sum(sigma^n E dsigma dtheta):
 * - Hs = 4 sqrt(m0)
 * - Tm01 = 2pi m0/m1 ; Tm02 = 2pi sqrt(m0/m2) ; Tm-1,0 = 2pi m-1/m0
 * - Tp = 2pi/sigma_p where sigma_p is the frequency of the peak of the 1D frequency spectrum
 * - Dp: peak direction, mean direction of the peak frequency band (as WW3 dp)
 * - Dir: mean direction (Kuik et al. 1988) in nautical convention (coming from)
 * - Dspr: directional spreading (Kuik et al. 1988)
 * - Urms = sqrt(sum(sigma^2/sinh^2(kh) E dsigma dtheta))
 * - Radiation stresses Sxx, Sxy, Syy (Longuet-Higgins & Stewart 1964)
 *
 * @tparam T Data type (float or double)
 * @param XParam Flow model parameters
 * @param XWaveParam Wave model parameters
 * @param i Cell memory index
 * @param h Water depth (m)
 * @param XWave Wave model structure
 */
template <class T> __host__ __device__ void CalcWaveBulkCell(const Param& XParam, const WaveParam& XWaveParam, int i, T h, const WaveModel<T>& XWave)
{
	const int nfreq = XWaveParam.nfreq;
	const int ndir = XWaveParam.ndir;
	const int stride = XWaveParam.specstride;

	const T g = T(XParam.g);
	const T rho = T(XParam.rho);
	const T dth = T(XWaveParam.dtheta);
	const T twopi = T(2.0 * pi);

	T m0 = T(0.0), m1 = T(0.0), m2 = T(0.0), mm1 = T(0.0);
	T Exp = T(0.0), Eyp = T(0.0); // directional moments of the peak frequency band
	T Ex = T(0.0), Ey = T(0.0);
	T Efmax = T(0.0), sigp = T(0.0);
	T Uorb2 = T(0.0);
	T Sxx = T(0.0), Sxy = T(0.0), Syy = T(0.0);

	if (h > T(XWaveParam.wavedepthmin))
	{
		for (int ifreq = 0; ifreq < nfreq; ifreq++)
		{
			T sig = XWave.spec.sigma[ifreq];
			T dsig = XWave.spec.dsigma[ifreq];

			// k and cg are cached in XWave.kin (recalculated only when the medium changes), n = cg/c = cg k / sigma
			size_t ik = specloc(stride, ifreq, i);
			T k = XWave.kin.k[ik];
			T kh = k * h;
			T n = (k > T(0.0)) ? XWave.kin.cg[ik] * k / sig : T(0.5);

			T Ef = T(0.0);
			T Exf = T(0.0), Eyf = T(0.0);
			for (int idir = 0; idir < ndir; idir++)
			{
				int ispec = ifreq * ndir + idir;
				T E = XWave.action.N[specloc(stride, ispec, i)] * sig * dsig * dth;
				T c = XWave.spec.costh[idir];
				T s = XWave.spec.sinth[idir];

				Ef += E;
				Exf += E * c;
				Eyf += E * s;

				Sxx += E * (n * (c * c + T(1.0)) - T(0.5));
				Sxy += E * (n * c * s);
				Syy += E * (n * (s * s + T(1.0)) - T(0.5));
			}

			Ex += Exf;
			Ey += Eyf;
			m0 += Ef;
			m1 += Ef * sig;
			m2 += Ef * sig * sig;
			mm1 += Ef / sig;

			if (kh < T(20.0))
			{
				T sh = sinh(kh);
				Uorb2 += sig * sig / (sh * sh) * Ef;
			}

			// Peak of the frequency spectrum (density, not bin energy, since bins are not uniform)
			if (Ef / dsig > Efmax)
			{
				Efmax = Ef / dsig;
				sigp = sig;
				Exp = Exf;
				Eyp = Eyf;
			}
		}
	}

	T Hs = T(4.0) * sqrt(m0);
	T Tm01 = (m1 > T(0.0)) ? twopi * m0 / m1 : T(0.0);
	T Tm02 = (m2 > T(0.0)) ? twopi * sqrt(m0 / m2) : T(0.0);
	T Tp = (sigp > T(0.0)) ? twopi / sigp : T(0.0);
	T Tmm10 = (m0 > T(0.0)) ? twopi * mm1 / m0 : T(0.0);

	T Dir = T(0.0), Dspr = T(0.0), Dp = T(0.0);
	if (m0 > T(0.0))
	{
		// Cartesian "going to" direction to nautical "coming from" direction
		T thm = atan2(Ey, Ex) * T(180.0) / T(pi);
		Dir = T(270.0) - thm;
		Dir = Dir - T(360.0) * floor(Dir / T(360.0));

		T thp = atan2(Eyp, Exp) * T(180.0) / T(pi);
		Dp = T(270.0) - thp;
		Dp = Dp - T(360.0) * floor(Dp / T(360.0));

		T R = sqrt(Ex * Ex + Ey * Ey) / m0;
		Dspr = sqrt(utils::max(T(2.0) * (T(1.0) - R), T(0.0))) * T(180.0) / T(pi);
	}

	XWave.bulk.Hs[i] = Hs;
	XWave.bulk.Tm01[i] = Tm01;
	XWave.bulk.Tm02[i] = Tm02;
	XWave.bulk.Tp[i] = Tp;
	XWave.bulk.Tmm10[i] = Tmm10;
	XWave.bulk.Dp[i] = Dp;
	XWave.bulk.Dir[i] = Dir;
	XWave.bulk.Dspr[i] = Dspr;
	XWave.bulk.km[i] = (Tm01 > T(0.0)) ? WaveDispersion(twopi / Tm01, h, g) : T(0.0);
	XWave.bulk.Urms[i] = sqrt(Uorb2);

	XWave.force.Sxx[i] = rho * g * Sxx;
	XWave.force.Sxy[i] = rho * g * Sxy;
	XWave.force.Syy[i] = rho * g * Syy;
}
template __host__ __device__ void CalcWaveBulkCell<float>(const Param& XParam, const WaveParam& XWaveParam, int i, float h, const WaveModel<float>& XWave);
template __host__ __device__ void CalcWaveBulkCell<double>(const Param& XParam, const WaveParam& XWaveParam, int i, double h, const WaveModel<double>& XWave);


/**
 * @brief Maximum stable explicit wave time step in a single cell.
 *
 * Based on the geographic propagation CFL: dt = CFL * delta / (sqrt(2) * (cg_max + |U|))
 * where cg_max is the group velocity of the lowest frequency (the fastest component for a given depth)
 * and sqrt(2) accounts for the unsplit 2D upwind scheme (|cos|+|sin| <= sqrt(2)).
 *
 * The spectral space Courant numbers (refraction and frequency shift) are not included here: the spectral
 * propagation velocities are clipped so that their Courant number does not exceed ctmax (see WaveSpecPropCell).
 *
 * @tparam T Data type (float or double)
 * @return Time step (s). Huge value for dry cells
 */
template <class T> __host__ __device__ T CalcWavedtCell(const Param& XParam, const WaveParam& XWaveParam, int i, int level, T h, const EvolvingP<T>& XEv, const WaveModel<T>& XWave)
{
	T dt = T(1.0e30); // "infinite" time step for dry cells (fits in float)

	if (h > T(XWaveParam.wavedepthmin))
	{
		T g = T(XParam.g);
		T sigmin = XWave.spec.sigma[0];
		T k = WaveDispersion(sigmin, h, g);
		T cg = WaveGroupVelocity(sigmin, k, h);
		T U = XWaveParam.wavecurrent ? sqrt(XEv.u[i] * XEv.u[i] + XEv.v[i] * XEv.v[i]) : T(0.0);
		T delta = calcres(T(XParam.delta), level);

		dt = T(XWaveParam.waveCFL) * delta / (T(1.41421356) * (cg + U));
	}
	return dt;
}
template __host__ __device__ float CalcWavedtCell<float>(const Param& XParam, const WaveParam& XWaveParam, int i, int level, float h, const EvolvingP<float>& XEv, const WaveModel<float>& XWave);
template __host__ __device__ double CalcWavedtCell<double>(const Param& XParam, const WaveParam& XWaveParam, int i, int level, double h, const EvolvingP<double>& XEv, const WaveModel<double>& XWave);


/**
 * @brief Create a dummy Loop structure so the flow time step reduction (CalctimestepGPU) can be reused
 * without its output time and 1.5*dtmax clamping.
 */
template <class T> Loop<T> WaveDummyLoop()
{
	Loop<T> XLoop;
	XLoop.totaltime = 0.0;
	XLoop.nextoutputtime = 0.0;
	XLoop.dtmax = std::numeric_limits<double>::max();
	return XLoop;
}
template Loop<float> WaveDummyLoop<float>();
template Loop<double> WaveDummyLoop<double>();


/**
 * @brief Calculate the maximum stable wave time step over the whole domain on the CPU.
 *
 * If the user prescribed a wave time step (wavedt > 0) it is used instead.
 * The per cell time step is stored in XWave.time.dtmax and reduced using the flow reduction function (timestepreductionCPU).
 *
 * @return Maximum wave time step (s)
 */
template <class T> double CalcWavedtmaxCPU(Param XParam, WaveParam XWaveParam, Model<T> XModel, WaveModel<T> XWave)
{
	if (XWaveParam.wavedt > 0.0)
	{
		return XWaveParam.wavedt;
	}

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
				XWave.time.dtmax[i] = CalcWavedtCell(XParam, XWaveParam, i, XModel.blocks.level[ib], h, XModel.evolv, XWave);
			}
		}
	}

	Loop<T> XLoopdummy = WaveDummyLoop<T>();
	return double(timestepreductionCPU(XParam, XLoopdummy, XModel.blocks, XWave.time));
}
template double CalcWavedtmaxCPU<float>(Param XParam, WaveParam XWaveParam, Model<float> XModel, WaveModel<float> XWave);
template double CalcWavedtmaxCPU<double>(Param XParam, WaveParam XWaveParam, Model<double> XModel, WaveModel<double> XWave);


/**
 * @brief Time step for the main loop when the flow model is turned off (wave only simulation).
 *
 * Uses the maximum stable wave time step (XWave.clock.dtmax) and adjusts it so that the next output time is reached exactly
 * (same approach as CalctimestepCPU for the flow).
 *
 * @return Main loop time step (s)
 */
template <class T> double CalcWavetimestep(Param XParam, Loop<T> XLoop, WaveModel<T> XWave)
{
	double dt = XWave.clock.dtmax;

	double remain = XLoop.nextoutputtime - XLoop.totaltime;
	if (remain > 0.0 && ceil(remain / dt) > 0.0)
	{
		dt = remain / ceil(remain / dt);
	}
	// Don't overshoot the end of the simulation
	if ((XParam.endtime - XLoop.totaltime) > 0.0 && dt > (XParam.endtime - XLoop.totaltime))
	{
		dt = XParam.endtime - XLoop.totaltime;
	}
	return dt;
}
template double CalcWavetimestep<float>(Param XParam, Loop<float> XLoop, WaveModel<float> XWave);
template double CalcWavetimestep<double>(Param XParam, Loop<double> XLoop, WaveModel<double> XWave);


/**
 * @brief Make a light copy of the flow parameters for the wave model.
 *
 * Param is passed by value to (nearly) all the block functions (halo, gradients, ...) and holds strings and vectors
 * (output variables, time series, zones...) that are deep copied each time. The wave model calls these functions for each
 * spectral bin so the copies become very expensive (on the CPU). This light copy keeps all the numerical parameters
 * and drops the containers that are not needed by the mesh functions.
 *
 * @param XParam Flow model parameters
 * @param theta Slope limiter parameter to use in the copy (used by the gradient functions)
 * @return Light copy of XParam
 */
Param WaveLightParam(Param XParam, double theta)
{
	Param P = XParam;
	P.theta = theta;
	P.reftime.clear(); P.reftime.shrink_to_fit();
	P.crs_ref.clear(); P.crs_ref.shrink_to_fit();
	P.hotstartfile.clear(); P.hotstartfile.shrink_to_fit();
	P.outfile.clear(); P.outfile.shrink_to_fit();
	P.AdaptCrit.clear(); P.AdaptCrit.shrink_to_fit();
	P.Adapt_arg1.clear(); P.Adapt_arg2.clear(); P.Adapt_arg3.clear(); P.Adapt_arg4.clear(); P.Adapt_arg5.clear();
	P.Adapt_arg1.shrink_to_fit(); P.Adapt_arg2.shrink_to_fit(); P.Adapt_arg3.shrink_to_fit(); P.Adapt_arg4.shrink_to_fit(); P.Adapt_arg5.shrink_to_fit();
	std::vector<TSoutnode>().swap(P.TSnodesout);
	std::vector<std::string>().swap(P.outvars);
	std::vector<outzoneP>().swap(P.outzone);
	std::vector<std::string>().swap(P.Toutput.inputstr);
	std::vector<double>().swap(P.Toutput.val);
#ifdef USE_CATALYST
	P.vtk_outputfile_root.clear(); P.vtk_outputfile_root.shrink_to_fit();
	P.python_pipeline.clear(); P.python_pipeline.shrink_to_fit();
#endif
	return P;
}


/**
 * @brief Make a light copy of the flow model structure for the wave model.
 *
 * Same reason as WaveLightParam: the block structure (BlockP) holds the output zones (with strings and vectors)
 * and is passed by value to the block functions. The light copy shares all the array pointers with XModel but
 * drops the output containers.
 */
template <class T> Model<T> WaveLightModel(Model<T> XModel)
{
	Model<T> M = XModel;
	std::vector<outzoneB>().swap(M.blocks.outZone);
	std::map<std::string, T*>().swap(M.OutputVarMap);
	std::map<std::string, std::string>().swap(M.Outvarlongname);
	std::map<std::string, std::string>().swap(M.Outvarstdname);
	std::map<std::string, std::string>().swap(M.Outvarunits);
	std::vector<double>().swap(M.OutputT);
	return M;
}
template Model<float> WaveLightModel<float>(Model<float> XModel);
template Model<double> WaveLightModel<double>(Model<double> XModel);


/**
 * @brief Update the medium seen by the wave model in one cell (depth, rate of change of depth and currents).
 *
 * @param invtadv 1/(time since the last update) used to calculate dh/dt (0 for no time variation)
 */
template <class T> __host__ __device__ void WaveMediumCell(const Param& XParam, const WaveParam& XWaveParam, int i, T invtadv, const EvolvingP<T>& XEv, T* zb, const WaveModel<T>& XWave)
{
	T h = WaveDepth(XParam, XWaveParam, i, XEv, zb);
	bool wet = h > T(XWaveParam.wavedepthmin);

	XWave.medium.dhdt[i] = (h - XWave.medium.ho[i]) * invtadv;
	XWave.medium.ho[i] = h;
	XWave.medium.h[i] = h;
	XWave.medium.u[i] = (XWaveParam.wavecurrent && wet) ? XEv.u[i] : T(0.0);
	XWave.medium.v[i] = (XWaveParam.wavecurrent && wet) ? XEv.v[i] : T(0.0);
}
template __host__ __device__ void WaveMediumCell<float>(const Param& XParam, const WaveParam& XWaveParam, int i, float invtadv, const EvolvingP<float>& XEv, float* zb, const WaveModel<float>& XWave);
template __host__ __device__ void WaveMediumCell<double>(const Param& XParam, const WaveParam& XWaveParam, int i, double invtadv, const EvolvingP<double>& XEv, double* zb, const WaveModel<double>& XWave);


/**
 * @brief Wave number, group velocity and dsigma/dh of each frequency in one cell (from the wave depth medium.h).
 * These are only recalculated when the medium changes (see WavePrep).
 */
template <class T> __host__ __device__ void WaveKinCell(const Param& XParam, const WaveParam& XWaveParam, int i, const WaveModel<T>& XWave)
{
	T h = XWave.medium.h[i];
	T g = T(XParam.g);
	for (int ifreq = 0; ifreq < XWaveParam.nfreq; ifreq++)
	{
		size_t ik = specloc(XWaveParam.specstride, ifreq, i);
		T sig = XWave.spec.sigma[ifreq];
		T k = WaveDispersion(sig, h, g);
		T kh = k * h;
		XWave.kin.k[ik] = k;
		XWave.kin.cg[ik] = WaveGroupVelocity(sig, k, h);
		XWave.kin.dsdd[ik] = (kh < T(20.0) && k > T(0.0)) ? k * sig / sinh(T(2.0) * kh) : T(0.0);
	}
}
template __host__ __device__ void WaveKinCell<float>(const Param& XParam, const WaveParam& XWaveParam, int i, const WaveModel<float>& XWave);
template __host__ __device__ void WaveKinCell<double>(const Param& XParam, const WaveParam& XWaveParam, int i, const WaveModel<double>& XWave);


/**
 * @brief Upwind (first or second order) flux of action density through a face of the geographic grid.
 *
 * The face is between the "minus" cell (left or bottom) and the "plus" cell (right or top).
 * The velocity at the face is the average of the cell velocities (or the velocity of the wet cell if one of the cells is dry).
 * Second order: the upwind cell value is extrapolated to the face using the limited gradient (MUSCL).
 * Dry cells hold no action density: waves propagating towards a dry cell are absorbed (as in SWAN).
 * The faces on the domain boundary are overwritten afterwards by the wave boundary conditions (WavebndFaceCell).
 *
 * @param cm, cp Propagation velocity normal to the face in the minus and plus cells
 * @param Nm, Np Action density in the minus and plus cells
 * @param sm, sp Half cell slope correction (0.5*delta*dN/dx) of the minus and plus cells (0 for first order)
 * @param wetm, wetp Wet flags of the minus and plus cells
 */
template <class T> __host__ __device__ T WaveUpwindFaceFlux(T cm, T cp, T Nm, T Np, T sm, T sp, bool wetm, bool wetp)
{
	if (!wetm && !wetp)
	{
		return T(0.0);
	}
	T cf = (wetm && wetp) ? T(0.5) * (cm + cp) : (wetm ? cm : cp);
	T Nmf = wetm ? Nm + sm : T(0.0);
	T Npf = wetp ? Np - sp : T(0.0);

	return (cf > T(0.0)) ? cf * Nmf : cf * Npf;
}
template __host__ __device__ float WaveUpwindFaceFlux<float>(float cm, float cp, float Nm, float Np, float sm, float sp, bool wetm, bool wetp);
template __host__ __device__ double WaveUpwindFaceFlux<double>(double cm, double cp, double Nm, double Np, double sm, double sp, bool wetm, bool wetp);


/**
 * @brief Action density flux through the left face of cell (ix,iy) of block ib for spectral bin ispec.
 * ix goes from 0 to blkwidth (the face at ix=blkwidth is the right face of the block, stored in the right halo).
 */
template <class T> __host__ __device__ void WaveFluxXCell(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int ispec, int woff, const BlockP<T>& XBlock, const WaveModel<T>& XWave)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;
	int bw = XParam.blkwidth;
	int stride = XWaveParam.specstride;
	int ifreq = ispec / XWaveParam.ndir;
	int idir = ispec - ifreq * XWaveParam.ndir;
	T hmin = T(XWaveParam.wavedepthmin);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int il = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);

	T delta = calcres(T(XParam.delta), XBlock.level[ib]);
	T cth = XWave.spec.costh[idir];

	bool wetm = XWave.medium.h[il] > hmin;
	bool wetp = XWave.medium.h[i] > hmin;

	T cm = XWave.kin.cg[specloc(stride, ifreq, il)] * cth + XWave.medium.u[il];
	T cp = XWave.kin.cg[specloc(stride, ifreq, i)] * cth + XWave.medium.u[i];

	T Nm = XWave.action.N[specloc(stride, ispec, il)];
	T Np = XWave.action.N[specloc(stride, ispec, i)];

	T sm = T(0.0), sp = T(0.0);
	if (XWaveParam.wavesolver == 2)
	{
		sm = T(0.5) * delta * XWave.work.dNdx[il + woff];
		sp = T(0.5) * delta * XWave.work.dNdx[i + woff];
	}

	XWave.work.Fx[i + woff] = WaveUpwindFaceFlux(cm, cp, Nm, Np, sm, sp, wetm, wetp);
}
template __host__ __device__ void WaveFluxXCell<float>(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int ispec, int woff, const BlockP<float>& XBlock, const WaveModel<float>& XWave);
template __host__ __device__ void WaveFluxXCell<double>(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int ispec, int woff, const BlockP<double>& XBlock, const WaveModel<double>& XWave);


/**
 * @brief Action density flux through the bottom face of cell (ix,iy) of block ib for spectral bin ispec.
 * iy goes from 0 to blkwidth (the face at iy=blkwidth is the top face of the block, stored in the top halo).
 */
template <class T> __host__ __device__ void WaveFluxYCell(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int ispec, int woff, const BlockP<T>& XBlock, const WaveModel<T>& XWave)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;
	int bw = XParam.blkwidth;
	int stride = XWaveParam.specstride;
	int ifreq = ispec / XWaveParam.ndir;
	int idir = ispec - ifreq * XWaveParam.ndir;
	T hmin = T(XWaveParam.wavedepthmin);

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);

	T delta = calcres(T(XParam.delta), XBlock.level[ib]);
	T sth = XWave.spec.sinth[idir];

	bool wetm = XWave.medium.h[ibot] > hmin;
	bool wetp = XWave.medium.h[i] > hmin;

	T cm = XWave.kin.cg[specloc(stride, ifreq, ibot)] * sth + XWave.medium.v[ibot];
	T cp = XWave.kin.cg[specloc(stride, ifreq, i)] * sth + XWave.medium.v[i];

	T Nm = XWave.action.N[specloc(stride, ispec, ibot)];
	T Np = XWave.action.N[specloc(stride, ispec, i)];

	T sm = T(0.0), sp = T(0.0);
	if (XWaveParam.wavesolver == 2)
	{
		sm = T(0.5) * delta * XWave.work.dNdy[ibot + woff];
		sp = T(0.5) * delta * XWave.work.dNdy[i + woff];
	}

	XWave.work.Fy[i + woff] = WaveUpwindFaceFlux(cm, cp, Nm, Np, sm, sp, wetm, wetp);
}
template __host__ __device__ void WaveFluxYCell<float>(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int ispec, int woff, const BlockP<float>& XBlock, const WaveModel<float>& XWave);
template __host__ __device__ void WaveFluxYCell<double>(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int ispec, int woff, const BlockP<double>& XBlock, const WaveModel<double>& XWave);


/**
 * @brief Add the divergence of the geographic fluxes of spectral bin ispec to the tendency dN (interior cell).
 * Same convention as updateEVCPU for the flow: dN = -(Fx[right] - Fx[i] + Fy[top] - Fy[i]) / delta.
 */
template <class T> __host__ __device__ void WaveGeoDivCell(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int ispec, int woff, const BlockP<T>& XBlock, const WaveModel<T>& XWave)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
	int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

	T delta = calcres(T(XParam.delta), XBlock.level[ib]);

	XWave.adv.dN[specloc(XWaveParam.specstride, ispec, i)] -= (XWave.work.Fx[iright + woff] - XWave.work.Fx[i + woff] + XWave.work.Fy[itop + woff] - XWave.work.Fy[i + woff]) / delta;
}
template __host__ __device__ void WaveGeoDivCell<float>(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int ispec, int woff, const BlockP<float>& XBlock, const WaveModel<float>& XWave);
template __host__ __device__ void WaveGeoDivCell<double>(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int ispec, int woff, const BlockP<double>& XBlock, const WaveModel<double>& XWave);


/**
 * @brief Upwind flux in spectral space (theta or sigma) through the face between bins a and b (b = a+1).
 * @param cf Propagation velocity at the face (positive from a to b)
 * @param Na, Nb Action density of bins a and b
 * @param Nam Action density of the bin before a (used for the second order slope of a)
 * @param Nbp Action density of the bin after b (used for the second order slope of b)
 * @param order2 Use the second order (limited, MUSCL) reconstruction
 * @param theta Limiter parameter (1: minmod, 2: superbee)
 * @param slopea, slopeb false to force a zero slope (first order) in bin a or b (e.g. first/last frequency)
 */
template <class T> __host__ __device__ T WaveSpecFaceFlux(T cf, T Na, T Nb, T Nam, T Nbp, bool order2, T theta, bool slopea, bool slopeb)
{
	if (cf > T(0.0))
	{
		T sa = (order2 && slopea) ? minmod2(theta, Nam, Na, Nb) : T(0.0);
		return cf * (Na + T(0.5) * sa);
	}
	else
	{
		T sb = (order2 && slopeb) ? minmod2(theta, Na, Nb, Nbp) : T(0.0);
		return cf * (Nb - T(0.5) * sb);
	}
}
template __host__ __device__ float WaveSpecFaceFlux<float>(float cf, float Na, float Nb, float Nam, float Nbp, bool order2, float theta, bool slopea, bool slopeb);
template __host__ __device__ double WaveSpecFaceFlux<double>(double cf, double Na, double Nb, double Nam, double Nbp, bool order2, double theta, bool slopea, bool slopeb);


/**
 * @brief Clip a value to [-cmax, cmax].
 */
template <class T> __host__ __device__ T WaveClip(T c, T cmax)
{
	return utils::min(utils::max(c, T(-1.0) * cmax), cmax);
}


/**
 * @brief Propagation in spectral space (refraction and frequency shift) for one cell. Adds to the tendency dN.
 *
 * Velocities in spectral space (cartesian, relative frequency sigma, e.g. SWAN technical manual):
 *
 *   ctheta = sigma/sinh(2kh) (sin(theta) dh/dx - cos(theta) dh/dy)
 *          + cos(theta) (sin(theta) du/dx - cos(theta) du/dy) + sin(theta) (sin(theta) dv/dx - cos(theta) dv/dy)
 *
 *   csigma = k sigma/sinh(2kh) (dh/dt + u dh/dx + v dh/dy)
 *          - cg k (cos^2(theta) du/dx + cos(theta) sin(theta) (du/dy + dv/dx) + sin^2(theta) dv/dy)
 *
 * The fluxes are calculated at the faces between bins with an upwind scheme (second order MUSCL for wavesolver 2)
 * - theta is periodic
 * - sigma: no incoming flux through the lowest and highest frequency boundaries, outgoing flux is allowed (energy leaves the spectral domain)
 * The face velocities are clipped so that the spectral Courant number is <= ctmax (as the WW3 refraction limiter).
 *
 * @param dt Time step of the (stage of the) time integration, used by the Courant number clipping
 */
template <class T> __host__ __device__ void WaveSpecPropCell(const Param& XParam, const WaveParam& XWaveParam, int i, T dt, const WaveModel<T>& XWave)
{
	T h = XWave.medium.h[i];
	if (h <= T(XWaveParam.wavedepthmin))
	{
		return;
	}

	const int nf = XWaveParam.nfreq;
	const int nd = XWaveParam.ndir;
	const int stride = XWaveParam.specstride;
	const bool order2 = (XWaveParam.wavesolver == 2);
	const T theta = T(XWaveParam.wavetheta);
	const T dth = T(XWaveParam.dtheta);
	const T ctmax = T(XWaveParam.ctmax);

	const bool dorefrac = XWaveParam.refraction && (nd > 1);
	const bool dofshift = XWaveParam.freqshift && (nf > 1);

	if (!dorefrac && !dofshift)
	{
		return;
	}

	T dhdx = XWave.medium.dhdx[i];
	T dhdy = XWave.medium.dhdy[i];
	T dhdt = XWave.medium.dhdt[i];
	T u = XWave.medium.u[i];
	T v = XWave.medium.v[i];
	T dudx = XWave.medium.dudx[i];
	T dudy = XWave.medium.dudy[i];
	T dvdx = XWave.medium.dvdx[i];
	T dvdy = XWave.medium.dvdy[i];

	T* N = XWave.action.N;
	T* dN = XWave.adv.dN;

	//============================================
	// Refraction (theta space, periodic)
	if (dorefrac)
	{
		T cthmax = ctmax * dth / dt;
		for (int ifreq = 0; ifreq < nf; ifreq++)
		{
			size_t ik = specloc(stride, ifreq, i);
			T k = XWave.kin.k[ik];
			if (k <= T(0.0))
			{
				continue;
			}
			T sdh = XWave.kin.dsdd[ik] / k; // (dsigma/dh)/k = sigma/sinh(2kh)

			int base = ifreq * nd;
			// ctheta of the first bin (the value of bin b is carried over to the next face)
			T c0 = XWave.spec.costh[0], s0 = XWave.spec.sinth[0];
			T cth0 = sdh * (s0 * dhdx - c0 * dhdy) + c0 * (s0 * dudx - c0 * dudy) + s0 * (s0 * dvdx - c0 * dvdy);
			T ctha = cth0;
			for (int ia = 0; ia < nd; ia++)
			{
				int ib = (ia + 1) % nd;
				int iam = (ia - 1 + nd) % nd;
				int ibp = (ib + 1) % nd;

				T cthb = cth0;
				if (ib != 0)
				{
					T cb = XWave.spec.costh[ib], sb = XWave.spec.sinth[ib];
					cthb = sdh * (sb * dhdx - cb * dhdy) + cb * (sb * dudx - cb * dudy) + sb * (sb * dvdx - cb * dvdy);
				}

				T cf = WaveClip(T(0.5) * (ctha + cthb), cthmax);

				size_t sa_i = specloc(stride, base + ia, i);
				size_t sb_i = specloc(stride, base + ib, i);

				T G = WaveSpecFaceFlux(cf, N[sa_i], N[sb_i], N[specloc(stride, base + iam, i)], N[specloc(stride, base + ibp, i)], order2, theta, true, true);

				dN[sa_i] -= G / dth;
				dN[sb_i] += G / dth;

				ctha = cthb;
			}
		}
	}

	//============================================
	// Frequency shift (sigma space)
	if (dofshift)
	{
		T depterm = dhdt + u * dhdx + v * dhdy;
		for (int idir = 0; idir < nd; idir++)
		{
			T c = XWave.spec.costh[idir];
			T s = XWave.spec.sinth[idir];
			T curterm = c * c * dudx + c * s * (dudy + dvdx) + s * s * dvdy;

			// csigma in each frequency bin is calculated on the fly
			// lower boundary (below the first frequency): only outgoing flux
			{
				size_t ik = specloc(stride, 0, i);
				T cs = XWave.kin.dsdd[ik] * depterm - XWave.kin.cg[ik] * XWave.kin.k[ik] * curterm;
				T dsig = XWave.spec.dsigma[0];
				if (cs < T(0.0))
				{
					size_t is = specloc(stride, idir, i);
					T F = WaveClip(cs, ctmax * dsig / dt) * N[is];
					dN[is] += F / dsig;
				}
			}

			T csa = XWave.kin.dsdd[specloc(stride, 0, i)] * depterm - XWave.kin.cg[specloc(stride, 0, i)] * XWave.kin.k[specloc(stride, 0, i)] * curterm;
			for (int ja = 0; ja < nf - 1; ja++)
			{
				int jb = ja + 1;
				size_t ikb = specloc(stride, jb, i);
				T csb = XWave.kin.dsdd[ikb] * depterm - XWave.kin.cg[ikb] * XWave.kin.k[ikb] * curterm;

				T dsiga = XWave.spec.dsigma[ja];
				T dsigb = XWave.spec.dsigma[jb];
				T cf = WaveClip(T(0.5) * (csa + csb), ctmax * utils::min(dsiga, dsigb) / dt);

				size_t sa_i = specloc(stride, ja * nd + idir, i);
				size_t sb_i = specloc(stride, jb * nd + idir, i);
				T Nam = (ja > 0) ? N[specloc(stride, (ja - 1) * nd + idir, i)] : T(0.0);
				T Nbp = (jb < nf - 1) ? N[specloc(stride, (jb + 1) * nd + idir, i)] : T(0.0);

				T F = WaveSpecFaceFlux(cf, N[sa_i], N[sb_i], Nam, Nbp, order2, theta, ja > 0, jb < nf - 1);

				dN[sa_i] -= F / dsiga;
				dN[sb_i] += F / dsigb;

				csa = csb;
			}

			// upper boundary (above the last frequency): only outgoing flux
			{
				int jl = nf - 1;
				size_t ik = specloc(stride, jl, i);
				T cs = XWave.kin.dsdd[ik] * depterm - XWave.kin.cg[ik] * XWave.kin.k[ik] * curterm;
				T dsig = XWave.spec.dsigma[jl];
				if (cs > T(0.0))
				{
					size_t is = specloc(stride, jl * nd + idir, i);
					T F = WaveClip(cs, ctmax * dsig / dt) * N[is];
					dN[is] -= F / dsig;
				}
			}
		}
	}
}
template __host__ __device__ void WaveSpecPropCell<float>(const Param& XParam, const WaveParam& XWaveParam, int i, float dt, const WaveModel<float>& XWave);
template __host__ __device__ void WaveSpecPropCell<double>(const Param& XParam, const WaveParam& XWaveParam, int i, double dt, const WaveModel<double>& XWave);


/**
 * @brief Update of the action density for one cell: N = alpha*N_o + (1-alpha)*(N + dt*dN), N >= 0 and N = 0 in dry cells.
 * alpha = 0: forward Euler step (or first stage of Heun / SSP-RK2); alpha = 0.5: second stage of Heun / SSP-RK2.
 */
template <class T> __host__ __device__ void WaveUpdateCell(const WaveParam& XWaveParam, int i, T dt, T alpha, const WaveModel<T>& XWave)
{
	bool wet = XWave.medium.h[i] > T(XWaveParam.wavedepthmin);
	for (int ispec = 0; ispec < XWaveParam.nspec; ispec++)
	{
		size_t is = specloc(XWaveParam.specstride, ispec, i);
		T Nn = XWave.action.N[is] + dt * XWave.adv.dN[is];
		Nn = alpha * XWave.action_o.N[is] + (T(1.0) - alpha) * Nn;
		XWave.action.N[is] = wet ? utils::max(Nn, T(0.0)) : T(0.0);
	}
}
template __host__ __device__ void WaveUpdateCell<float>(const WaveParam& XWaveParam, int i, float dt, float alpha, const WaveModel<float>& XWave);
template __host__ __device__ void WaveUpdateCell<double>(const WaveParam& XWaveParam, int i, double dt, double alpha, const WaveModel<double>& XWave);


/**
 * @brief Boundary taper factor (0 to 1 over wavebndtaper seconds from the start of the simulation).
 */
double WaveBndTaper(Param XParam, WaveParam XWaveParam, double time)
{
	if (XWaveParam.wavebndtaper <= 0.0)
	{
		return 1.0;
	}
	return utils::min(utils::max((time - XParam.inittime) / XWaveParam.wavebndtaper, 0.0), 1.0);
}


/**
 * @brief Action density of one spectral bin of a discrete JONSWAP spectrum (shared CPU/GPU).
 *
 * E(f) ~ f^-5 exp(-5/4 (fp/f)^4) gamma^exp(-(f-fp)^2/(2 s^2 fp^2)) (s = 0.07 for f<=fp and 0.09 for f>fp)
 * D(theta) ~ cos^2s((theta-theta_m)/2) with s = 2/dspr^2 - 1 (dspr in rad, Kuik et al. 1988)
 * The discrete spectrum is scaled so that 4*sqrt(m0) = Hs exactly on the model spectral grid
 * (the normalisations are recalculated here so the function only needs the bulk parameters).
 *
 * @param ispec Spectral bin
 * @param Hs Significant wave height (m)
 * @param Tp Peak period (s)
 * @param Dir Mean direction (degrees nautical, coming from)
 * @param Dspr Directional spreading (degrees)
 * @param gamma JONSWAP peak enhancement factor
 * @return Action density N of the bin
 */
template <class T> __host__ __device__ T WaveJonswapEf(int nfreq, T sigma, T fp, T gamma)
{
	// Unscaled JONSWAP frequency distribution E(sigma)
	if (nfreq == 1)
	{
		return T(1.0); // single representative frequency: all the energy in the only bin
	}
	const T twopi = T(2.0 * pi);
	T f = sigma / twopi;
	T sj = (f <= fp) ? T(0.07) : T(0.09);
	T r = exp(T(-1.0) * (f - fp) * (f - fp) / (T(2.0) * sj * sj * fp * fp));
	T fpf = fp / f;
	return pow(f, T(-5.0)) * exp(T(-1.25) * fpf * fpf * fpf * fpf) * pow(gamma, r) / twopi;
}

template <class T> __host__ __device__ void WaveJonswapShape(T Dir, T Dspr, T& thm, T& spow)
{
	thm = (T(270.0) - Dir) * T(pi) / T(180.0);
	T dsprrad = Dspr * T(pi) / T(180.0);
	spow = utils::max(T(2.0) / (dsprrad * dsprrad) - T(1.0), T(0.0));
}

/**
 * @brief Normalisation integrals of the discrete JONSWAP spectrum: m0 = sum(E(sigma) dsigma) and Dsum = sum(D(theta) dtheta).
 * They only depend on Tp, gamma, Dir and Dspr, so for a spatially uniform boundary they are calculated once per time on the host.
 */
template <class T> __host__ __device__ void WaveJonswapNorm(const WaveParam& XWaveParam, const WaveSpecGridP<T>& XSpec, T Tp, T Dir, T Dspr, T gamma, T& m0, T& Dsum)
{
	m0 = T(0.0);
	Dsum = T(0.0);
	if (Tp <= T(0.0))
	{
		return;
	}
	T fp = T(1.0) / Tp;
	for (int jf = 0; jf < XWaveParam.nfreq; jf++)
	{
		m0 += WaveJonswapEf(XWaveParam.nfreq, XSpec.sigma[jf], fp, gamma) * XSpec.dsigma[jf];
	}
	T thm, spow;
	WaveJonswapShape(Dir, Dspr, thm, spow);
	for (int jd = 0; jd < XWaveParam.ndir; jd++)
	{
		Dsum += pow(fabs(cos(T(0.5) * (XSpec.theta[jd] - thm))), T(2.0) * spow) * T(XWaveParam.dtheta);
	}
}
template __host__ __device__ void WaveJonswapNorm<float>(const WaveParam& XWaveParam, const WaveSpecGridP<float>& XSpec, float Tp, float Dir, float Dspr, float gamma, float& m0, float& Dsum);
template __host__ __device__ void WaveJonswapNorm<double>(const WaveParam& XWaveParam, const WaveSpecGridP<double>& XSpec, double Tp, double Dir, double Dspr, double gamma, double& m0, double& Dsum);

/**
 * @brief Action density of one spectral bin of a discrete JONSWAP spectrum with known normalisation integrals (see WaveJonswapNorm).
 */
template <class T> __host__ __device__ T WaveJonswapNbin(const WaveParam& XWaveParam, const WaveSpecGridP<T>& XSpec, int ispec, T Hs, T Tp, T Dir, T Dspr, T gamma, T m0, T Dsum)
{
	if (Hs <= T(0.0) || Tp <= T(0.0) || m0 <= T(0.0) || Dsum <= T(0.0))
	{
		return T(0.0);
	}
	int ifreq = ispec / XWaveParam.ndir;
	int idir = ispec - ifreq * XWaveParam.ndir;
	T thm, spow;
	WaveJonswapShape(Dir, Dspr, thm, spow);
	T Ebin = WaveJonswapEf(XWaveParam.nfreq, XSpec.sigma[ifreq], T(1.0) / Tp, gamma);
	T Dbin = pow(fabs(cos(T(0.5) * (XSpec.theta[idir] - thm))), T(2.0) * spow);
	return (Hs * Hs / T(16.0)) / m0 * Ebin * Dbin / Dsum / XSpec.sigma[ifreq];
}
template __host__ __device__ float WaveJonswapNbin<float>(const WaveParam& XWaveParam, const WaveSpecGridP<float>& XSpec, int ispec, float Hs, float Tp, float Dir, float Dspr, float gamma, float m0, float Dsum);
template __host__ __device__ double WaveJonswapNbin<double>(const WaveParam& XWaveParam, const WaveSpecGridP<double>& XSpec, int ispec, double Hs, double Tp, double Dir, double Dspr, double gamma, double m0, double Dsum);

template <class T> __host__ __device__ T WaveJonswapN(const WaveParam& XWaveParam, const WaveSpecGridP<T>& XSpec, int ispec, T Hs, T Tp, T Dir, T Dspr, T gamma)
{
	T m0, Dsum;
	WaveJonswapNorm(XWaveParam, XSpec, Tp, Dir, Dspr, gamma, m0, Dsum);
	return WaveJonswapNbin(XWaveParam, XSpec, ispec, Hs, Tp, Dir, Dspr, gamma, m0, Dsum);
}
template __host__ __device__ float WaveJonswapN<float>(const WaveParam& XWaveParam, const WaveSpecGridP<float>& XSpec, int ispec, float Hs, float Tp, float Dir, float Dspr, float gamma);
template __host__ __device__ double WaveJonswapN<double>(const WaveParam& XWaveParam, const WaveSpecGridP<double>& XSpec, int ispec, double Hs, double Tp, double Dir, double Dspr, double gamma);


/**
 * @brief Overwrite the flux of action density through one boundary face (wave equivalent of the flow bndFluxGPUSide).
 *
 * The face is defined as in the flow boundary functions: side.isright/side.istop give the side, tx the cell along the side.
 * - outgoing waves (propagation towards the outside): upwind flux from the interior cell (all types)
 * - incoming waves: type 0 (absorbing): no flux; type 1 (Neumann): flux of the interior action density (zero gradient);
 *   type 2 (JONSWAP): flux of the JONSWAP spectrum given by Hs, Tp, Dir, Dspr, gamma (tapered by bndfac)
 * Faces of half-boundary blocks that are not on the boundary are left untouched.
 * woff is the offset of the bin in the batch work arrays (j*specstride for the j-th bin of the batch).
 */
template <class T> __host__ __device__ void WavebndFaceCell(const Param& XParam, const WaveParam& XWaveParam, int isright, int istop, int ib, int tx, int ispec, int woff, int type, T Hs, T Tp, T Dir, T Dspr, T gamma, T m0n, T Dsumn, T bndfac, const BlockP<T>& XBlock, const WaveModel<T>& XWave)
{
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;
	int bw = XParam.blkwidth;
	int stride = XWaveParam.specstride;
	int ifreq = ispec / XWaveParam.ndir;
	int idir = ispec - ifreq * XWaveParam.ndir;

	int ix, iy, ixin, iyin;
	bool onbnd;
	if (isright == 0)
	{
		// bottom or top side
		ix = tx; ixin = tx;
		iy = (istop < 0) ? 0 : bw;
		iyin = (istop < 0) ? 0 : bw - 1;
		bool lefthalf = ix < (bw / 2);
		if (istop < 0)
		{
			onbnd = (lefthalf ? XBlock.BotLeft[ib] : XBlock.BotRight[ib]) == ib;
		}
		else
		{
			onbnd = (lefthalf ? XBlock.TopLeft[ib] : XBlock.TopRight[ib]) == ib;
		}
	}
	else
	{
		// left or right side
		iy = tx; iyin = tx;
		ix = (isright < 0) ? 0 : bw;
		ixin = (isright < 0) ? 0 : bw - 1;
		bool lowerhalf = iy < (bw / 2);
		if (isright < 0)
		{
			onbnd = (lowerhalf ? XBlock.LeftBot[ib] : XBlock.LeftTop[ib]) == ib;
		}
		else
		{
			onbnd = (lowerhalf ? XBlock.RightBot[ib] : XBlock.RightTop[ib]) == ib;
		}
	}
	if (!onbnd)
	{
		return;
	}

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib); // where the face flux is stored
	int iin = memloc(halowidth, blkmemwidth, ixin, iyin, ib); // interior cell next to the face

	T F = T(0.0);
	if (XWave.medium.h[iin] > T(XWaveParam.wavedepthmin))
	{
		T cg = XWave.kin.cg[specloc(stride, ifreq, iin)];
		T c = (isright == 0) ? cg * XWave.spec.sinth[idir] + XWave.medium.v[iin] : cg * XWave.spec.costh[idir] + XWave.medium.u[iin];
		T Nin = XWave.action.N[specloc(stride, ispec, iin)];

		bool incoming = (isright < 0 || istop < 0) ? (c > T(0.0)) : (c < T(0.0));
		if (incoming)
		{
			T Nbnd = T(0.0);
			if (type == 1)
			{
				Nbnd = Nin;
			}
			else if (type == 2)
			{
				// m0n, Dsumn: JONSWAP normalisation integrals (precalculated for uniform segments, <=0 to calculate them here)
				if (m0n <= T(0.0))
				{
					WaveJonswapNorm(XWaveParam, XWave.spec, Tp, Dir, Dspr, gamma, m0n, Dsumn);
				}
				Nbnd = bndfac * WaveJonswapNbin(XWaveParam, XWave.spec, ispec, Hs, Tp, Dir, Dspr, gamma, m0n, Dsumn);
			}
			F = c * Nbnd;
		}
		else
		{
			F = c * Nin;
		}
	}

	if (isright == 0)
	{
		XWave.work.Fy[i + woff] = F;
	}
	else
	{
		XWave.work.Fx[i + woff] = F;
	}
}
template __host__ __device__ void WavebndFaceCell<float>(const Param& XParam, const WaveParam& XWaveParam, int isright, int istop, int ib, int tx, int ispec, int woff, int type, float Hs, float Tp, float Dir, float Dspr, float gamma, float m0n, float Dsumn, float bndfac, const BlockP<float>& XBlock, const WaveModel<float>& XWave);
template __host__ __device__ void WavebndFaceCell<double>(const Param& XParam, const WaveParam& XWaveParam, int isright, int istop, int ib, int tx, int ispec, int woff, int type, double Hs, double Tp, double Dir, double Dspr, double gamma, double m0n, double Dsumn, double bndfac, const BlockP<double>& XBlock, const WaveModel<double>& XWave);


/**
 * @brief Bulk wave parameters of a spatially uniform wave boundary segment at a given time (linear interpolation in time).
 * The direction is interpolated as a vector to avoid the 0/360 discontinuity. Outside the time range of the file the
 * first/last values are used. If the file has no gamma column, gammadefault (gammajsp) is used.
 */
void WaveBndValues(const wavebndsegment& seg, double time, double gammadefault, double& Hs, double& Tp, double& Dir, double& Dspr, double& gamma)
{
	const std::vector<WaveBndIn>& d = seg.data;
	int n = int(d.size());
	int ib = 0;
	double w = 0.0;
	if (n == 1 || time <= d[0].time)
	{
		ib = 0; w = 0.0;
	}
	else if (time >= d[n - 1].time)
	{
		ib = n - 2; w = 1.0;
	}
	else
	{
		while (ib < n - 2 && d[ib + 1].time < time)
		{
			ib++;
		}
		w = (time - d[ib].time) / (d[ib + 1].time - d[ib].time);
	}
	const WaveBndIn& a = d[ib];
	const WaveBndIn& b = d[utils::min(ib + 1, n - 1)];

	Hs = a.Hs + w * (b.Hs - a.Hs);
	Tp = a.Tp + w * (b.Tp - a.Tp);
	Dspr = a.Dspr + w * (b.Dspr - a.Dspr);
	double ga = a.gamma < 0.0 ? gammadefault : a.gamma;
	double gb = b.gamma < 0.0 ? gammadefault : b.gamma;
	gamma = ga + w * (gb - ga);

	double xa = cos(a.Dir * pi / 180.0), ya = sin(a.Dir * pi / 180.0);
	double xb = cos(b.Dir * pi / 180.0), yb = sin(b.Dir * pi / 180.0);
	Dir = atan2(ya + w * (yb - ya), xa + w * (xb - xa)) * 180.0 / pi;
}


//==============================================================================
// CPU functions
//==============================================================================


/**
 * @brief Fill the halo of every spectral bin of the action density (CPU).
 *
 * Each spectral bin has the same layout as a flow variable so the flow halo function (fillHaloC) is reused.
 */
template <class T> void fillHaloWave(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveSpectrumP<T> XAction)
{
	for (int ispec = 0; ispec < XWaveParam.nspec; ispec++)
	{
		fillHaloC(XParam, XBlock, XAction.N + specloc(XWaveParam.specstride, ispec, 0));
	}
}
template void fillHaloWave<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, WaveSpectrumP<float> XAction);
template void fillHaloWave<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, WaveSpectrumP<double> XAction);


/**
 * @brief Prepare the medium (depth, currents and gradients) and the kinematics (k, cg) for the wave model (CPU).
 *
 * Called every time the wave model is advanced (the flow may have changed the water depth and the currents).
 *
 * @param tadv Time since the last call (used to calculate dh/dt), 0 for no time variation
 */
template <class T> void WavePrepCPU(Param XParam, WaveParam XWaveParam, double tadv, Model<T> XModel, WaveModel<T> XWave)
{
	T invtadv = (tadv > 0.0) ? T(1.0 / tadv) : T(0.0);
	int ib;

	// Medium in the interior cells
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XModel.blocks.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
				WaveMediumCell(XParam, XWaveParam, i, invtadv, XModel.evolv, XModel.zb, XWave);
			}
		}
	}

	// Halo (reusing the flow functions)
	fillHaloC(XParam, XModel.blocks, XWave.medium.h);
	fillHaloC(XParam, XModel.blocks, XWave.medium.u);
	fillHaloC(XParam, XModel.blocks, XWave.medium.v);

	// Gradients (reusing the flow functions)
	gradientC(XParam, XModel.blocks, XWave.medium.h, XWave.medium.dhdx, XWave.medium.dhdy);
	gradientC(XParam, XModel.blocks, XWave.medium.u, XWave.medium.dudx, XWave.medium.dudy);
	gradientC(XParam, XModel.blocks, XWave.medium.v, XWave.medium.dvdx, XWave.medium.dvdy);

	// Kinematics in all cells including the halo (needed for the face velocities)
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XModel.blocks.active[ibl];
		for (int iy = -XParam.halowidth; iy < XParam.blkwidth + XParam.halowidth; iy++)
		{
			for (int ix = -XParam.halowidth; ix < XParam.blkwidth + XParam.halowidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
				WaveKinCell(XParam, XWaveParam, i, XWave);
			}
		}
	}
}
template void WavePrepCPU<float>(Param XParam, WaveParam XWaveParam, double tadv, Model<float> XModel, WaveModel<float> XWave);
template void WavePrepCPU<double>(Param XParam, WaveParam XWaveParam, double tadv, Model<double> XModel, WaveModel<double> XWave);


/**
 * @brief Reset the action density tendency (dN) to zero on the CPU.
 *
 * Note: this is a direct loop rather than calling InitArrayBUQ for each spectral bin because Param is passed by value
 * (and holds strings/vectors) so calling the flow functions nspec times per step is expensive on the CPU.
 */
template <class T> void ResetWaveAdvCPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveAdvanceP<T> XAdv)
{
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
				for (int ispec = 0; ispec < XWaveParam.nspec; ispec++)
				{
					XAdv.dN[specloc(XWaveParam.specstride, ispec, i)] = T(0.0);
				}
			}
		}
	}
}
template void ResetWaveAdvCPU<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, WaveAdvanceP<float> XAdv);
template void ResetWaveAdvCPU<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, WaveAdvanceP<double> XAdv);


/**
 * @brief Geographic propagation (CPU): add -d(cx N)/dx - d(cy N)/dy to dN, by batches of nchunk spectral bins.
 *
 * For each batch, using the virtual blocks (XWave.vblocks, see InitWaveVirtualBlocks) so the flow block functions
 * process all the bins of the batch in a single call:
 * 1. fill the halo of the bins (fillHaloC, as for any flow variable)
 * 2. second order: limited gradients of the bins and their halo (gradientC / gradientHalo, as for the flow)
 * 3. upwind fluxes through the left (ix = 0..blkwidth) and bottom (iy = 0..blkwidth) faces of the cells
 * 3b. boundary conditions: the fluxes through the faces on the domain boundary are overwritten for each wave boundary segment (WavebndFluxCPU)
 * 4. coarse/fine flux correction in the halo (fillHaloLRFluxC / fillHaloBTFluxC, as for the flow fluxes) so the scheme is conservative across resolution changes
 * 5. flux divergence
 *
 * @param XParamN Light copy of XParam with theta = wavetheta (used for the limited gradients of N)
 */
template <class T> void WaveGeoPropCPU(Param XParam, Param XParamN, WaveParam XWaveParam, const std::vector<wavebndsegment>& segs, double time, T bndfac, BlockP<T> XBlock, WaveModel<T> XWave)
{
	int bw = XParam.blkwidth;
	int nb = XParam.nblkmem;
	int stride = XWaveParam.specstride;

	Param XParamV = XParam;
	Param XParamNV = XParamN;

	for (int ispec0 = 0; ispec0 < XWaveParam.nspec; ispec0 += XWaveParam.nchunk)
	{
		int nbins = utils::min(XWaveParam.nchunk, XWaveParam.nspec - ispec0);
		T* Nchunk = XWave.action.N + specloc(stride, ispec0, 0);
		XParamV.nblk = XParam.nblk * nbins;
		XParamNV.nblk = XParam.nblk * nbins;

		fillHaloC(XParamV, XWave.vblocks, Nchunk);

		if (XWaveParam.wavesolver == 2)
		{
			gradientC(XParamNV, XWave.vblocks, Nchunk, XWave.work.dNdx, XWave.work.dNdy);
			gradientHalo(XParamNV, XWave.vblocks, Nchunk, XWave.work.dNdx, XWave.work.dNdy);
		}

		for (int ibl = 0; ibl < XParamV.nblk; ibl++)
		{
			int ibv = XWave.vblocks.active[ibl];
			int ib = ibv % nb;
			int j = ibv / nb;
			int ispec = ispec0 + j;
			int woff = j * stride;
			for (int iy = 0; iy < bw; iy++)
			{
				for (int ix = 0; ix < (bw + XParam.halowidth); ix++)
				{
					WaveFluxXCell(XParam, XWaveParam, ix, iy, ib, ispec, woff, XBlock, XWave);
				}
			}
			for (int iy = 0; iy < (bw + XParam.halowidth); iy++)
			{
				for (int ix = 0; ix < bw; ix++)
				{
					WaveFluxYCell(XParam, XWaveParam, ix, iy, ib, ispec, woff, XBlock, XWave);
				}
			}
		}

		// Boundary conditions: overwrite the fluxes through the faces on the domain boundary (as FlowbndFlux for the flow)
		for (int iseg = 0; iseg < segs.size(); iseg++)
		{
			WavebndFluxCPU(XParam, XWaveParam, segs[iseg], time, bndfac, ispec0, nbins, XBlock, XWave);
		}

		fillHaloLRFluxC(XParamV, XWave.vblocks, XWave.work.Fx);
		fillHaloBTFluxC(XParamV, XWave.vblocks, XWave.work.Fy);

		for (int ibl = 0; ibl < XParamV.nblk; ibl++)
		{
			int ibv = XWave.vblocks.active[ibl];
			int ib = ibv % nb;
			int j = ibv / nb;
			for (int iy = 0; iy < bw; iy++)
			{
				for (int ix = 0; ix < bw; ix++)
				{
					WaveGeoDivCell(XParam, XWaveParam, ix, iy, ib, ispec0 + j, j * stride, XBlock, XWave);
				}
			}
		}
	}
}
template void WaveGeoPropCPU<float>(Param XParam, Param XParamN, WaveParam XWaveParam, const std::vector<wavebndsegment>& segs, double time, float bndfac, BlockP<float> XBlock, WaveModel<float> XWave);
template void WaveGeoPropCPU<double>(Param XParam, Param XParamN, WaveParam XWaveParam, const std::vector<wavebndsegment>& segs, double time, double bndfac, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Wave boundary conditions for one segment and a batch of spectral bins (CPU). Wave equivalent of FlowbndFlux.
 *
 * Uniform segments (text file): the bulk parameters are interpolated in time (WaveBndValues).
 * Spatially varying segments (netcdf file): the bulk parameters are interpolated from the forcing maps at each face (interp2BUQ),
 * once per face for the whole batch.
 *
 * @param ispec0 First spectral bin of the batch
 * @param nbins Number of bins in the batch
 */
template <class T> void WavebndFluxCPU(Param XParam, WaveParam XWaveParam, const wavebndsegment& seg, double time, T bndfac, int ispec0, int nbins, BlockP<T> XBlock, WaveModel<T> XWave)
{
	double Hs = 0.0, Tp = 10.0, Dir = 0.0, Dspr = 30.0, gamma = XWaveParam.gammajsp;
	if (seg.type == 2 && seg.on && seg.uniform)
	{
		WaveBndValues(seg, time, XWaveParam.gammajsp, Hs, Tp, Dir, Dspr, gamma);
	}
	bool interpmaps = (seg.type == 2 && seg.on && !seg.uniform);

	// JONSWAP normalisation: once per call for a uniform segment, at each face for maps
	T m0n = T(-1.0), Dsumn = T(-1.0);
	if (seg.type == 2 && !interpmaps)
	{
		WaveJonswapNorm(XWaveParam, XWave.spec, T(Tp), T(Dir), T(Dspr), T(gamma), m0n, Dsumn);
	}

	const bndsegmentside* sides[4] = { &seg.left, &seg.right, &seg.top, &seg.bot };
	for (int is = 0; is < 4; is++)
	{
		const bndsegmentside& side = *sides[is];
		for (int ibl = 0; ibl < side.nblk; ibl++)
		{
			int ib = side.blk[ibl];
			T delta = calcres(T(XParam.dx), XBlock.level[ib]);
			for (int tx = 0; tx < XParam.blkwidth; tx++)
			{
				if (interpmaps)
				{
					int ix = (side.isright == 0) ? tx : ((side.isright < 0) ? 0 : XParam.blkwidth);
					int iy = (side.isright == 0) ? ((side.istop < 0) ? 0 : XParam.blkwidth) : tx;
					T xx = T(XParam.xo) + XBlock.xo[ib] + ix * delta;
					T yy = T(XParam.yo) + XBlock.yo[ib] + iy * delta;
					Hs = interp2BUQ(xx, yy, seg.Hs);
					Tp = interp2BUQ(xx, yy, seg.Tp);
					Dir = interp2BUQ(xx, yy, seg.Dir);
					Dspr = interp2BUQ(xx, yy, seg.Dspr);
				}
				T m0f = m0n, Dsumf = Dsumn;
				if (interpmaps)
				{
					WaveJonswapNorm(XWaveParam, XWave.spec, T(Tp), T(Dir), T(Dspr), T(gamma), m0f, Dsumf);
				}
				for (int j = 0; j < nbins; j++)
				{
					WavebndFaceCell(XParam, XWaveParam, side.isright, side.istop, ib, tx, ispec0 + j, j * XWaveParam.specstride, seg.type, T(Hs), T(Tp), T(Dir), T(Dspr), T(gamma), m0f, Dsumf, bndfac, XBlock, XWave);
				}
			}
		}
	}
}
template void WavebndFluxCPU<float>(Param XParam, WaveParam XWaveParam, const wavebndsegment& seg, double time, float bndfac, int ispec0, int nbins, BlockP<float> XBlock, WaveModel<float> XWave);
template void WavebndFluxCPU<double>(Param XParam, WaveParam XWaveParam, const wavebndsegment& seg, double time, double bndfac, int ispec0, int nbins, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Spectral propagation (refraction and frequency shift) for all active cells (CPU).
 */
template <class T> void WaveSpecPropCPU(Param XParam, WaveParam XWaveParam, T dt, BlockP<T> XBlock, WaveModel<T> XWave)
{
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
				WaveSpecPropCell(XParam, XWaveParam, i, dt, XWave);
			}
		}
	}
}
template void WaveSpecPropCPU<float>(Param XParam, WaveParam XWaveParam, float dt, BlockP<float> XBlock, WaveModel<float> XWave);
template void WaveSpecPropCPU<double>(Param XParam, WaveParam XWaveParam, double dt, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Explicit update N = alpha*N_o + (1-alpha)*(N + dt*dN) on the CPU.
 */
template <class T> void UpdateWaveActionCPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, T dt, T alpha, WaveModel<T> XWave)
{
	int ib;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
				WaveUpdateCell(XWaveParam, i, dt, alpha, XWave);
			}
		}
	}
}
template void UpdateWaveActionCPU<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, float dt, float alpha, WaveModel<float> XWave);
template void UpdateWaveActionCPU<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, double dt, double alpha, WaveModel<double> XWave);


/**
 * @brief Calculate the bulk wave parameters and radiation stresses for all active cells on the CPU.
 */
template <class T> void CalcWaveBulkCPU(Param XParam, WaveParam XWaveParam, Model<T> XModel, WaveModel<T> XWave)
{
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
				CalcWaveBulkCell(XParam, XWaveParam, i, h, XWave);
			}
		}
	}
}
template void CalcWaveBulkCPU<float>(Param XParam, WaveParam XWaveParam, Model<float> XModel, WaveModel<float> XWave);
template void CalcWaveBulkCPU<double>(Param XParam, WaveParam XWaveParam, Model<double> XModel, WaveModel<double> XWave);


/**
 * @brief Right hand side of the action balance equation dN = L(N) (CPU): propagation (+ sources, to be implemented).
 */
template <class T> void WaveRHSCPU(Param XParam, Param XParamN, WaveParam XWaveParam, T dt, double time, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave)
{
	T bndfac = T(WaveBndTaper(XParam, XWaveParam, time));

	ResetWaveAdvCPU(XParam, XWaveParam, XModel.blocks, XWave.adv);

	// Geographic propagation: dN -= d(cx N)/dx + d(cy N)/dy
	WaveGeoPropCPU(XParam, XParamN, XWaveParam, XForcing.wavebndseg, time, bndfac, XModel.blocks, XWave);

	// Spectral propagation: dN -= d(csigma N)/dsigma + d(ctheta N)/dtheta
	WaveSpecPropCPU(XParam, XWaveParam, dt, XModel.blocks, XWave);

	// Source terms: dN += Stot/sigma
	// TODO (wind input uses XForcing.UWind / XForcing.VWind)
}
template void WaveRHSCPU<float>(Param XParam, Param XParamN, WaveParam XWaveParam, float dt, double time, Forcing<float> XForcing, Model<float> XModel, WaveModel<float> XWave);
template void WaveRHSCPU<double>(Param XParam, Param XParamN, WaveParam XWaveParam, double dt, double time, Forcing<float> XForcing, Model<double> XModel, WaveModel<double> XWave);


/**
 * @brief Main CPU wave solver: advance the action density by one explicit wave time step dt.
 *
 * This is the wave equivalent of FlowCPU. The medium (depth, currents, k, cg) must have been prepared (WavePrepCPU).
 * - wavesolver = 1: first order upwind in space, forward Euler in time
 * - wavesolver = 2: second order upwind (MUSCL with limiter) in space, Heun (SSP-RK2) in time:
 *       N* = N^n + dt L(N^n) ;  N^{n+1} = 0.5 N^n + 0.5 (N* + dt L(N*))
 * Then the bulk parameters and radiation stresses are updated.
 *
 * @tparam T Data type (float or double)
 * @param XParam Light copy of the flow model parameters (see WaveLightParam)
 * @param XWaveParam Wave model parameters
 * @param dt Wave time step (s)
 * @param time Wave model time at the start of the step (s), used for the boundary conditions
 * @param dobulk Calculate the bulk parameters at the end of the step
 * @param XForcing Forcing (wind is used by the wave model)
 * @param XModel Flow model (mesh, bathy, water depth and currents)
 * @param XWave Wave model
 */
template <class T> void WaveCPU(Param XParam, WaveParam XWaveParam, T dt, double time, bool dobulk, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave)
{
	Param XParamN = WaveLightParam(XParam, XWaveParam.wavetheta);

	//============================================
	// Keep a copy of the action density at the start of the step
	memcpy(XWave.action_o.N, XWave.action.N, size_t(XWaveParam.nspec) * size_t(XWaveParam.specstride) * sizeof(T));

	//============================================
	// First stage (forward Euler)
	WaveRHSCPU(XParam, XParamN, XWaveParam, dt, time, XForcing, XModel, XWave);
	UpdateWaveActionCPU(XParam, XWaveParam, XModel.blocks, dt, T(0.0), XWave);

	//============================================
	// Second stage (Heun / SSP-RK2)
	if (XWaveParam.wavesolver == 2)
	{
		WaveRHSCPU(XParam, XParamN, XWaveParam, dt, time + double(dt), XForcing, XModel, XWave);
		UpdateWaveActionCPU(XParam, XWaveParam, XModel.blocks, dt, T(0.5), XWave);
	}

	//============================================
	// Bulk parameters and radiation stresses (only when needed: output or coupling)
	if (dobulk)
	{
		CalcWaveBulkCPU(XParam, XWaveParam, XModel, XWave);
	}

	//============================================
	// Wave forces on the flow (Fx, Fy) from radiation stress gradients
	// TODO (only needed when XWaveParam.waveflowcoupling)
}
template void WaveCPU<float>(Param XParam, WaveParam XWaveParam, float dt, double time, bool dobulk, Forcing<float> XForcing, Model<float> XModel, WaveModel<float> XWave);
template void WaveCPU<double>(Param XParam, WaveParam XWaveParam, double dt, double time, bool dobulk, Forcing<float> XForcing, Model<double> XModel, WaveModel<double> XWave);

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
template <class T> __host__ __device__ T WaveDepth(Param XParam, WaveParam XWaveParam, int i, EvolvingP<T> XEv, T* zb)
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
template __host__ __device__ float WaveDepth<float>(Param XParam, WaveParam XWaveParam, int i, EvolvingP<float> XEv, float* zb);
template __host__ __device__ double WaveDepth<double>(Param XParam, WaveParam XWaveParam, int i, EvolvingP<double> XEv, double* zb);


/**
 * @brief Calculate the integrated (bulk) wave parameters and radiation stresses for a single cell.
 *
 * This function is shared between the CPU (CalcWaveBulkCPU) and the GPU (CalcWaveBulkGPU) versions.
 * With E(sigma,theta) = sigma*N(sigma,theta) and the moments m_n = sum(sigma^n E dsigma dtheta):
 * - Hs = 4 sqrt(m0)
 * - Tm01 = 2pi m0/m1 ; Tm02 = 2pi sqrt(m0/m2)
 * - Tp = 2pi/sigma_p where sigma_p is the frequency of the peak of the 1D frequency spectrum
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
template <class T> __host__ __device__ void CalcWaveBulkCell(Param XParam, WaveParam XWaveParam, int i, T h, WaveModel<T> XWave)
{
	const int nfreq = XWaveParam.nfreq;
	const int ndir = XWaveParam.ndir;
	const int stride = XWaveParam.specstride;

	const T g = T(XParam.g);
	const T rho = T(XParam.rho);
	const T dth = T(XWaveParam.dtheta);
	const T twopi = T(2.0 * pi);

	T m0 = T(0.0), m1 = T(0.0), m2 = T(0.0);
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

			T k = WaveDispersion(sig, h, g);
			T kh = k * h;
			T n = (kh > T(20.0)) ? T(0.5) : T(0.5) * (T(1.0) + T(2.0) * kh / sinh(T(2.0) * kh));

			T Ef = T(0.0);
			for (int idir = 0; idir < ndir; idir++)
			{
				int ispec = ifreq * ndir + idir;
				T E = XWave.action.N[specloc(stride, ispec, i)] * sig * dsig * dth;
				T c = XWave.spec.costh[idir];
				T s = XWave.spec.sinth[idir];

				Ef += E;
				Ex += E * c;
				Ey += E * s;

				Sxx += E * (n * (c * c + T(1.0)) - T(0.5));
				Sxy += E * (n * c * s);
				Syy += E * (n * (s * s + T(1.0)) - T(0.5));
			}

			m0 += Ef;
			m1 += Ef * sig;
			m2 += Ef * sig * sig;

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
			}
		}
	}

	T Hs = T(4.0) * sqrt(m0);
	T Tm01 = (m1 > T(0.0)) ? twopi * m0 / m1 : T(0.0);
	T Tm02 = (m2 > T(0.0)) ? twopi * sqrt(m0 / m2) : T(0.0);
	T Tp = (sigp > T(0.0)) ? twopi / sigp : T(0.0);

	T Dir = T(0.0), Dspr = T(0.0);
	if (m0 > T(0.0))
	{
		// Cartesian "going to" direction to nautical "coming from" direction
		T thm = atan2(Ey, Ex) * T(180.0) / T(pi);
		Dir = T(270.0) - thm;
		Dir = Dir - T(360.0) * floor(Dir / T(360.0));

		T R = sqrt(Ex * Ex + Ey * Ey) / m0;
		Dspr = sqrt(utils::max(T(2.0) * (T(1.0) - R), T(0.0))) * T(180.0) / T(pi);
	}

	XWave.bulk.Hs[i] = Hs;
	XWave.bulk.Tm01[i] = Tm01;
	XWave.bulk.Tm02[i] = Tm02;
	XWave.bulk.Tp[i] = Tp;
	XWave.bulk.Dir[i] = Dir;
	XWave.bulk.Dspr[i] = Dspr;
	XWave.bulk.km[i] = (Tm01 > T(0.0)) ? WaveDispersion(twopi / Tm01, h, g) : T(0.0);
	XWave.bulk.Urms[i] = sqrt(Uorb2);

	XWave.force.Sxx[i] = rho * g * Sxx;
	XWave.force.Sxy[i] = rho * g * Sxy;
	XWave.force.Syy[i] = rho * g * Syy;
}
template __host__ __device__ void CalcWaveBulkCell<float>(Param XParam, WaveParam XWaveParam, int i, float h, WaveModel<float> XWave);
template __host__ __device__ void CalcWaveBulkCell<double>(Param XParam, WaveParam XWaveParam, int i, double h, WaveModel<double> XWave);


/**
 * @brief Maximum stable explicit wave time step in a single cell.
 *
 * Based on the geographic propagation CFL: dt = CFL * delta / (sqrt(2) * (cg_max + |U|))
 * where cg_max is the group velocity of the lowest frequency (the fastest component for a given depth)
 * and sqrt(2) accounts for the unsplit 2D upwind scheme (|cos|+|sin| <= sqrt(2)).
 *
 * @todo add the spectral space CFL (refraction ctheta and frequency shift csigma) once propagation is implemented.
 *
 * @tparam T Data type (float or double)
 * @return Time step (s). Huge value for dry cells
 */
template <class T> __host__ __device__ T CalcWavedtCell(Param XParam, WaveParam XWaveParam, int i, int level, T h, EvolvingP<T> XEv, WaveModel<T> XWave)
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
template __host__ __device__ float CalcWavedtCell<float>(Param XParam, WaveParam XWaveParam, int i, int level, float h, EvolvingP<float> XEv, WaveModel<float> XWave);
template __host__ __device__ double CalcWavedtCell<double>(Param XParam, WaveParam XWaveParam, int i, int level, double h, EvolvingP<double> XEv, WaveModel<double> XWave);


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
 * @brief Explicit update of the action density N = max(N + dt*dN, 0) on the CPU.
 */
template <class T> void UpdateWaveActionCPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, T dt, WaveAdvanceP<T> XAdv, WaveSpectrumP<T> XAction)
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
					size_t is = specloc(XWaveParam.specstride, ispec, i);
					XAction.N[is] = utils::max(XAction.N[is] + dt * XAdv.dN[is], T(0.0));
				}
			}
		}
	}
}
template void UpdateWaveActionCPU<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, float dt, WaveAdvanceP<float> XAdv, WaveSpectrumP<float> XAction);
template void UpdateWaveActionCPU<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, double dt, WaveAdvanceP<double> XAdv, WaveSpectrumP<double> XAction);


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
 * @brief Main CPU wave solver: advance the action density by one wave time step dt.
 *
 * This is the wave equivalent of FlowCPU. The structure of the step is:
 * 1. Store the action density of the previous step (action_o)
 * 2. Fill the halo of each spectral bin (reusing the flow halo functions)
 * 3. Apply the wave boundary conditions                                 [to be implemented]
 * 4. Propagation in geographic space (cx, cy incl. currents)            [to be implemented]
 * 5. Propagation in spectral space (refraction ctheta, freq. shift csigma) [to be implemented]
 * 6. Source terms (wind, whitecapping, quadruplets, triads, breaking, friction) [to be implemented]
 * 7. Explicit update N = N + dt*dN (+ limiter)
 * 8. Bulk parameters and radiation stresses
 * 9. Wave forces for the flow model (radiation stress gradients)        [to be implemented]
 *
 * @tparam T Data type (float or double)
 * @param XParam Flow model parameters (mesh information)
 * @param XWaveParam Wave model parameters
 * @param dt Wave time step (s)
 * @param XForcing Forcing (wind is used by the wave model)
 * @param XModel Flow model (mesh, bathy, water depth and currents)
 * @param XWave Wave model
 */
template <class T> void WaveCPU(Param XParam, WaveParam XWaveParam, T dt, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave)
{
	//============================================
	// 1. Keep a copy of the action density at the previous step
	memcpy(XWave.action_o.N, XWave.action.N, size_t(XWaveParam.nspec) * size_t(XWaveParam.specstride) * sizeof(T));

	//============================================
	// 2. Fill the halo for each spectral bin
	fillHaloWave(XParam, XWaveParam, XModel.blocks, XWave.action);

	//============================================
	// 3. Wave boundary conditions
	// TODO: impose the parametric (JONSWAP) or spectral boundary on the halo of the boundary blocks

	//============================================
	// Reset tendency
	ResetWaveAdvCPU(XParam, XWaveParam, XModel.blocks, XWave.adv);

	//============================================
	// 4. Geographic propagation: dN -= d(cx N)/dx + d(cy N)/dy
	// TODO

	//============================================
	// 5. Spectral propagation: dN -= d(csigma N)/dsigma + d(ctheta N)/dtheta
	// TODO

	//============================================
	// 6. Source terms: dN += Stot/sigma
	// TODO (wind input uses XForcing.UWind / XForcing.VWind)

	//============================================
	// 7. Update action density
	UpdateWaveActionCPU(XParam, XWaveParam, XModel.blocks, dt, XWave.adv, XWave.action);

	//============================================
	// 8. Bulk parameters and radiation stresses
	CalcWaveBulkCPU(XParam, XWaveParam, XModel, XWave);

	//============================================
	// 9. Wave forces on the flow (Fx, Fy) from radiation stress gradients
	// TODO (only needed when XWaveParam.waveflowcoupling)
}
template void WaveCPU<float>(Param XParam, WaveParam XWaveParam, float dt, Forcing<float> XForcing, Model<float> XModel, WaveModel<float> XWave);
template void WaveCPU<double>(Param XParam, WaveParam XWaveParam, double dt, Forcing<float> XForcing, Model<double> XModel, WaveModel<double> XWave);

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

/*
* Wave source terms (SWAN GEN3 KOMEN formulations)
* ================================================
* For each bin the source terms are written as  S/sigma = A + (B - D) N  (action density):
*  - wind input (Komen et al. 1984 after Snyder et al. 1981): B = max(0, 0.25 rho_a/rho_w (28 ustar/c cos(theta - theta_w) - 1)) sigma,
*    ustar from U10 with the drag coefficient of Wu (1982); optional linear growth A (Cavaleri & Malanotte-Rizzoli 1981, SWAN AGROW)
*  - whitecapping (Komen et al. 1984): D = Cds ((1-delta) + delta k/k~) (s~/s~PM)^p sigma~ (k/k~)^powk,
*    s~ = k~ sqrt(m0), p = 2 powst, s~PM^2 = stpm, sigma~ = m0/m-1, k~ = (sum(k^-1/2 E)/m0)^-2
*  - depth induced breaking (Battjes & Janssen 1978): Dtot = 1/4 alpha Qb (sigma_m/2pi) Hmax^2, Hmax = gamma h,
*    Qb from (1-Qb)/ln(Qb) = -(Hrms/Hmax)^2 (SWAN approximation), distributed as D = Dtot/m0 over all the bins
*  - bottom friction: D = Cf sigma^2/(g^2 sinh^2(kh)), Cf = cfjon (JONSWAP) or cfw g Urms (Collins)
* The sinks (D) are treated implicitly and the wind growth (A, B) explicitly in both solvers.
*/

#include "WaveSource.h"


/**
 * @brief True if at least one source term is active.
 */
template <class T> __host__ __device__ bool WaveAnySource(const WaveParam& XWaveParam)
{
	return XWaveParam.windinput > 0 || XWaveParam.whitecapping > 0 || XWaveParam.breaking > 0 || XWaveParam.wavefriction > 0;
}
template __host__ __device__ bool WaveAnySource<float>(const WaveParam& XWaveParam);
template __host__ __device__ bool WaveAnySource<double>(const WaveParam& XWaveParam);


/**
 * @brief Integral quantities of the spectrum of cell i used by the source terms.
 */
template <class T> __host__ __device__ void WaveSourceIntegrals(const Param& XParam, const WaveParam& XWaveParam, int i, const WaveModel<T>& XWave, WaveSrcCell<T>& sc)
{
	const int nf = XWaveParam.nfreq;
	const int nd = XWaveParam.ndir;
	const int stride = XWaveParam.specstride;
	const T g = T(XParam.g);
	const T dth = T(XWaveParam.dtheta);

	sc.h = XWave.medium.h[i];
	sc.wet = sc.h > T(XWaveParam.wavedepthmin);
	sc.m0 = T(0.0); sc.sigt = T(0.0); sc.kt = T(0.0); sc.sigm = T(0.0);
	sc.Gw = T(0.0); sc.Qb = T(0.0); sc.Dbr = T(0.0); sc.Cfr = T(0.0);
	sc.ustar = T(0.0); sc.thw = T(0.0);
	if (!sc.wet)
	{
		return;
	}

	T m0 = T(0.0), m1 = T(0.0), mm1 = T(0.0), mk = T(0.0), ub = T(0.0);
	for (int ifreq = 0; ifreq < nf; ifreq++)
	{
		T sig = XWave.spec.sigma[ifreq];
		T sd = sig * XWave.spec.dsigma[ifreq] * dth;
		T Ef = T(0.0);
		for (int idir = 0; idir < nd; idir++)
		{
			Ef += XWave.action.N[specloc(stride, ifreq * nd + idir, i)] * sd;
		}
		T k = XWave.kin.k[specloc(stride, ifreq, i)];
		T kh = k * sc.h;
		m0 += Ef;
		m1 += Ef * sig;
		mm1 += Ef / sig;
		if (k > T(0.0))
		{
			mk += Ef / sqrt(k);
		}
		if (k > T(0.0) && kh < T(20.0))
		{
			T sh = sinh(kh);
			ub += sig * sig / (sh * sh) * Ef;
		}
	}
	sc.m0 = m0;
	if (m0 > T(0.0))
	{
		sc.sigt = (mm1 > T(0.0)) ? m0 / mm1 : T(0.0);
		sc.kt = (mk > T(0.0)) ? (m0 / mk) * (m0 / mk) : T(0.0);
		sc.sigm = m1 / m0;
	}

	// Wind (friction velocity from U10, Wu 1982 drag coefficient)
	if (XWaveParam.windinput > 0)
	{
		T uw = XWave.medium.uwind[i], vw = XWave.medium.vwind[i];
		T U10 = sqrt(uw * uw + vw * vw);
		T Cd = (U10 < T(7.5)) ? T(1.2875e-3) : (T(0.8) + T(0.065) * U10) * T(1.0e-3);
		sc.ustar = sqrt(Cd) * U10;
		sc.thw = (U10 > T(0.0)) ? atan2(vw, uw) : T(0.0);
	}

	// Whitecapping (Komen et al. 1984)
	if (XWaveParam.whitecapping > 0 && m0 > T(0.0) && sc.kt > T(0.0))
	{
		T st2 = sc.kt * sc.kt * m0; // s~^2
		sc.Gw = T(XWaveParam.cds) * pow(st2 / T(XWaveParam.stpm), T(XWaveParam.powst));
	}

	// Depth induced breaking (Battjes & Janssen 1978, SWAN approximation of Qb)
	if (XWaveParam.breaking > 0 && m0 > T(0.0))
	{
		T Hmax = T(XWaveParam.gammabr) * sc.h;
		T beta = sqrt(T(8.0) * m0) / Hmax;
		T Qb;
		if (beta < T(0.2))
		{
			Qb = T(0.0);
		}
		else if (beta >= T(1.0))
		{
			Qb = T(1.0);
		}
		else
		{
			T b2 = beta * beta;
			T Q0 = (beta <= T(0.5)) ? T(0.0) : (T(2.0) * beta - T(1.0)) * (T(2.0) * beta - T(1.0));
			T ex = exp((Q0 - T(1.0)) / b2);
			Qb = Q0 - b2 * (Q0 - ex) / (b2 - ex);
			Qb = utils::min(utils::max(Qb, T(0.0)), T(1.0));
		}
		sc.Qb = Qb;
		T Dtot = T(0.25) * T(XWaveParam.alphabr) * Qb * (sc.sigm / T(2.0 * pi)) * Hmax * Hmax;
		sc.Dbr = Dtot / m0;
	}

	// Bottom friction
	if (XWaveParam.wavefriction == 1)
	{
		sc.Cfr = T(XWaveParam.cfjon);
	}
	else if (XWaveParam.wavefriction == 2)
	{
		sc.Cfr = T(XWaveParam.cfw) * g * sqrt(ub);
	}
}
template __host__ __device__ void WaveSourceIntegrals<float>(const Param& XParam, const WaveParam& XWaveParam, int i, const WaveModel<float>& XWave, WaveSrcCell<float>& sc);
template __host__ __device__ void WaveSourceIntegrals<double>(const Param& XParam, const WaveParam& XWaveParam, int i, const WaveModel<double>& XWave, WaveSrcCell<double>& sc);


/**
 * @brief Source term coefficients of one bin: S/sigma = Alin + (Bgrow - Dsink) N.
 */
template <class T> __host__ __device__ void WaveSourceBin(const Param& XParam, const WaveParam& XWaveParam, const WaveModel<T>& XWave, const WaveSrcCell<T>& sc, int i, int ifreq, int idir, T& Dsink, T& Bgrow, T& Alin)
{
	Dsink = T(0.0); Bgrow = T(0.0); Alin = T(0.0);
	if (!sc.wet)
	{
		return;
	}
	const T g = T(XParam.g);
	T sig = XWave.spec.sigma[ifreq];
	T k = XWave.kin.k[specloc(XWaveParam.specstride, ifreq, i)];
	if (k <= T(0.0))
	{
		return;
	}
	T kh = k * sc.h;

	// Whitecapping
	if (XWaveParam.whitecapping > 0 && sc.kt > T(0.0))
	{
		T kk = k / sc.kt;
		T dl = T(XWaveParam.wcdelta);
		Dsink += sc.Gw * ((T(1.0) - dl) + dl * kk) * sc.sigt * pow(kk, T(XWaveParam.powk));
	}

	// Depth induced breaking
	Dsink += sc.Dbr;

	// Bottom friction
	if (XWaveParam.wavefriction > 0 && kh < T(20.0))
	{
		T sh = sinh(kh);
		Dsink += sc.Cfr * sig * sig / (g * g * sh * sh);
	}

	// Wind input
	if (XWaveParam.windinput > 0 && sc.ustar > T(0.0))
	{
		T c = sig / k;
		T cosd = cos(XWave.spec.theta[idir] - sc.thw);
		Bgrow = utils::max(T(0.0), T(0.25) * T(XWaveParam.rhoair / XParam.rho) * (T(28.0) * sc.ustar / c * cosd - T(1.0))) * sig;
		if (XWaveParam.lineargrowth)
		{
			T sigpm = T(2.0 * pi) * T(0.13) * g / (T(28.0) * sc.ustar);
			T r = sigpm / sig;
			T H = exp(T(-1.0) * r * r * r * r);
			T uc = sc.ustar * utils::max(cosd, T(0.0));
			T A = T(1.5e-3) / (T(2.0 * pi) * g * g) * uc * uc * uc * uc * H;
			Alin = A / sig;
		}
	}
}
template __host__ __device__ void WaveSourceBin<float>(const Param& XParam, const WaveParam& XWaveParam, const WaveModel<float>& XWave, const WaveSrcCell<float>& sc, int i, int ifreq, int idir, float& Dsink, float& Bgrow, float& Alin);
template __host__ __device__ void WaveSourceBin<double>(const Param& XParam, const WaveParam& XWaveParam, const WaveModel<double>& XWave, const WaveSrcCell<double>& sc, int i, int ifreq, int idir, double& Dsink, double& Bgrow, double& Alin);


/**
 * @brief Maximum change of action density allowed by the limiter: limiterfac * Phillips level N = 0.0081/(2 sigma k^3 cg).
 * Returns a negative value when the limiter is not active.
 */
template <class T> __host__ __device__ T WaveLimiterN(const WaveParam& XWaveParam, const WaveModel<T>& XWave, int i, int ifreq)
{
	if (!XWaveParam.wavelimiter || XWaveParam.windinput <= 0)
	{
		return T(-1.0);
	}
	size_t ik = specloc(XWaveParam.specstride, ifreq, i);
	T k = XWave.kin.k[ik];
	T cg = XWave.kin.cg[ik];
	if (k <= T(0.0) || cg <= T(0.0))
	{
		return T(-1.0);
	}
	return T(XWaveParam.limiterfac) * T(0.0081) / (T(2.0) * XWave.spec.sigma[ifreq] * k * k * k * cg);
}
template __host__ __device__ float WaveLimiterN<float>(const WaveParam& XWaveParam, const WaveModel<float>& XWave, int i, int ifreq);
template __host__ __device__ double WaveLimiterN<double>(const WaveParam& XWaveParam, const WaveModel<double>& XWave, int i, int ifreq);


/**
 * @brief Source term step of the explicit solver for one cell (operator splitting after the propagation):
 * N = (N + dt (Alin + Bgrow N)) / (1 + dt Dsink), sinks implicit (unconditionally stable), wind growth explicit, with limiter.
 */
template <class T> __host__ __device__ void WaveSourceStepCell(const Param& XParam, const WaveParam& XWaveParam, int i, T dt, const WaveModel<T>& XWave)
{
	WaveSrcCell<T> sc;
	WaveSourceIntegrals(XParam, XWaveParam, i, XWave, sc);
	if (!sc.wet)
	{
		return;
	}
	for (int ifreq = 0; ifreq < XWaveParam.nfreq; ifreq++)
	{
		T lim = WaveLimiterN(XWaveParam, XWave, i, ifreq);
		for (int idir = 0; idir < XWaveParam.ndir; idir++)
		{
			size_t is = specloc(XWaveParam.specstride, ifreq * XWaveParam.ndir + idir, i);
			T Dsink, Bgrow, Alin;
			WaveSourceBin(XParam, XWaveParam, XWave, sc, i, ifreq, idir, Dsink, Bgrow, Alin);
			T N0 = XWave.action.N[is];
			T Nn = (N0 + dt * (Alin + Bgrow * N0)) / (T(1.0) + dt * Dsink);
			if (lim > T(0.0))
			{
				Nn = utils::min(utils::max(Nn, N0 - lim), N0 + lim);
			}
			XWave.action.N[is] = utils::max(Nn, T(0.0));
		}
	}
}
template __host__ __device__ void WaveSourceStepCell<float>(const Param& XParam, const WaveParam& XWaveParam, int i, float dt, const WaveModel<float>& XWave);
template __host__ __device__ void WaveSourceStepCell<double>(const Param& XParam, const WaveParam& XWaveParam, int i, double dt, const WaveModel<double>& XWave);


/**
 * @brief Dissipation diagnostics of one cell (W/m2): Dbr (breaking), Dwc (whitecapping), Dfr (bottom friction), and Qb.
 */
template <class T> __host__ __device__ void WaveSourceDiagCell(const Param& XParam, const WaveParam& XWaveParam, int i, const WaveModel<T>& XWave)
{
	WaveSrcCell<T> sc;
	WaveSourceIntegrals(XParam, XWaveParam, i, XWave, sc);
	T Dwc = T(0.0), Dfr = T(0.0);
	if (sc.wet && sc.m0 > T(0.0))
	{
		const T g = T(XParam.g);
		for (int ifreq = 0; ifreq < XWaveParam.nfreq; ifreq++)
		{
			T sig = XWave.spec.sigma[ifreq];
			T sd = sig * XWave.spec.dsigma[ifreq] * T(XWaveParam.dtheta);
			T k = XWave.kin.k[specloc(XWaveParam.specstride, ifreq, i)];
			T kh = k * sc.h;
			T wc = T(0.0), fr = T(0.0);
			if (XWaveParam.whitecapping > 0 && sc.kt > T(0.0) && k > T(0.0))
			{
				T kk = k / sc.kt;
				T dl = T(XWaveParam.wcdelta);
				wc = sc.Gw * ((T(1.0) - dl) + dl * kk) * sc.sigt * pow(kk, T(XWaveParam.powk));
			}
			if (XWaveParam.wavefriction > 0 && k > T(0.0) && kh < T(20.0))
			{
				T sh = sinh(kh);
				fr = sc.Cfr * sig * sig / (g * g * sh * sh);
			}
			T Ef = T(0.0);
			for (int idir = 0; idir < XWaveParam.ndir; idir++)
			{
				Ef += XWave.action.N[specloc(XWaveParam.specstride, ifreq * XWaveParam.ndir + idir, i)] * sd;
			}
			Dwc += wc * Ef;
			Dfr += fr * Ef;
		}
	}
	T rg = T(XParam.rho) * T(XParam.g);
	XWave.dissip.Dbr[i] = rg * sc.Dbr * sc.m0;
	XWave.dissip.Dwc[i] = rg * Dwc;
	XWave.dissip.Dfr[i] = rg * Dfr;
	XWave.bulk.Qb[i] = sc.Qb;
}
template __host__ __device__ void WaveSourceDiagCell<float>(const Param& XParam, const WaveParam& XWaveParam, int i, const WaveModel<float>& XWave);
template __host__ __device__ void WaveSourceDiagCell<double>(const Param& XParam, const WaveParam& XWaveParam, int i, const WaveModel<double>& XWave);


/**
 * @brief 10 m wind seen by the wave model (CPU), from the flow model wind forcing (as AddwindforcingCPU).
 */
template <class T> void WaveWindCPU(Param XParam, WaveParam XWaveParam, Forcing<float> XForcing, BlockP<T> XBlock, WaveModel<T> XWave)
{
	bool haswind = (XWaveParam.windinput > 0) && !XForcing.UWind.inputfile.empty();
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		T delta = calcres(T(XParam.dx), XBlock.level[ib]);
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
				T uw = T(0.0), vw = T(0.0);
				if (haswind)
				{
					T x = T(XParam.xo) + XBlock.xo[ib] + ix * delta;
					T y = T(XParam.yo) + XBlock.yo[ib] + iy * delta;
					uw = XForcing.UWind.uniform ? T(XForcing.UWind.nowvalue) : interp2BUQ(x, y, XForcing.UWind);
					vw = XForcing.VWind.uniform ? T(XForcing.VWind.nowvalue) : interp2BUQ(x, y, XForcing.VWind);
				}
				XWave.medium.uwind[i] = uw;
				XWave.medium.vwind[i] = vw;
			}
		}
	}
}
template void WaveWindCPU<float>(Param XParam, WaveParam XWaveParam, Forcing<float> XForcing, BlockP<float> XBlock, WaveModel<float> XWave);
template void WaveWindCPU<double>(Param XParam, WaveParam XWaveParam, Forcing<float> XForcing, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief 10 m wind seen by the wave model (GPU), from the flow model wind forcing (as AddwindforcingGPU).
 */
template <class T> __global__ void WaveWindGPU(Param XParam, WaveParam XWaveParam, DynForcingP<float> Uwind, DynForcingP<float> Vwind, BlockP<T> XBlock, WaveModel<T> XWave)
{
	int ib = XBlock.active[blockIdx.x];
	int ix = threadIdx.x, iy = threadIdx.y;
	int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);
	T delta = calcres(T(XParam.dx), XBlock.level[ib]);
	T x = T(XParam.xo) + XBlock.xo[ib] + ix * delta;
	T y = T(XParam.yo) + XBlock.yo[ib] + iy * delta;
	XWave.medium.uwind[i] = Uwind.uniform ? T(Uwind.nowvalue) : interpDyn2BUQ(x, y, Uwind.GPU);
	XWave.medium.vwind[i] = Vwind.uniform ? T(Vwind.nowvalue) : interpDyn2BUQ(x, y, Vwind.GPU);
}
template __global__ void WaveWindGPU<float>(Param XParam, WaveParam XWaveParam, DynForcingP<float> Uwind, DynForcingP<float> Vwind, BlockP<float> XBlock, WaveModel<float> XWave);
template __global__ void WaveWindGPU<double>(Param XParam, WaveParam XWaveParam, DynForcingP<float> Uwind, DynForcingP<float> Vwind, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Source term step of the explicit solver for all active cells (CPU).
 */
template <class T> void WaveSourceStepCPU(Param XParam, WaveParam XWaveParam, T dt, BlockP<T> XBlock, WaveModel<T> XWave)
{
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				WaveSourceStepCell(XParam, XWaveParam, memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib), dt, XWave);
			}
		}
	}
}
template void WaveSourceStepCPU<float>(Param XParam, WaveParam XWaveParam, float dt, BlockP<float> XBlock, WaveModel<float> XWave);
template void WaveSourceStepCPU<double>(Param XParam, WaveParam XWaveParam, double dt, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Source term step of the explicit solver (GPU), one thread per cell.
 */
template <class T> __global__ void WaveSourceStepGPU(Param XParam, WaveParam XWaveParam, T dt, BlockP<T> XBlock, WaveModel<T> XWave)
{
	int ib = XBlock.active[blockIdx.x];
	int i = memloc(XParam.halowidth, XParam.blkmemwidth, threadIdx.x, threadIdx.y, ib);
	WaveSourceStepCell(XParam, XWaveParam, i, dt, XWave);
}
template __global__ void WaveSourceStepGPU<float>(Param XParam, WaveParam XWaveParam, float dt, BlockP<float> XBlock, WaveModel<float> XWave);
template __global__ void WaveSourceStepGPU<double>(Param XParam, WaveParam XWaveParam, double dt, BlockP<double> XBlock, WaveModel<double> XWave);

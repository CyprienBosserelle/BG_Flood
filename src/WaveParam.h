#ifndef WAVEPARAM_H
#define WAVEPARAM_H

#include "General.h"

/**
 *  A class. A class for holding the wave (spectral action balance) model parameters.
 *
 *  This is the wave-side equivalent of the Param class (flow side).
 *  The wave model solves the action balance equation
 *
 *      dN/dt + d(cx N)/dx + d(cy N)/dy + d(csigma N)/dsigma + d(ctheta N)/dtheta = Stot/sigma
 *
 *  for the action density N(x,y,sigma,theta,t) = E(x,y,sigma,theta,t)/sigma
 *  on the same (adaptive) block mesh as the flow model.
 *
 *  Default values follow SWAN (41.x) and WAVEWATCH III (v6/v7) recommendations:
 *  - Spectral discretisation: logarithmic frequencies (WW3 default: f1=0.0418 Hz, factor 1.1, 25 freqs) and 36 directions (10deg)
 *  - Depth induced breaking: Battjes & Janssen (1978), alpha=1.0, gamma=0.73 (SWAN default)
 *  - Bottom friction: JONSWAP, Cfjon=0.038 m2/s3 (SWAN >= 41.31 default)
 *  - Whitecapping: Komen et al. (1984) Cds=2.36e-5, stpm=3.02e-3 (SWAN GEN3 KOMEN default)
 *  - Quadruplets: DIA, lambda=0.25, Cnl4=3.0e7 (SWAN/WW3 default)
 *  - Triads: LTA, trfac=0.8, cutfr=2.5 (SWAN default, off by default)
 *  - Minimum depth: 0.05 m (SWAN DEPMIN)
 *
 *  All keywords used in BG_param.txt for the wave model are listed in ReadInput.cu (readparamstr(line, WaveParam)).
 *  The class purposely only contains plain data so it can be passed by value to CUDA kernels (as Param is).
 */
class WaveParam {
public:

	//*Wave model switches
	bool wavemodel = false; // Switch to run the spectral wave model (false: flow only [Default], true: run the wave model). Flow model can be turned off using "flowmodel = false"
	bool wavecurrent = true; // Include wave-current interaction (Doppler shift and current refraction) using the flow model velocities u,v
	bool wavewaterlevel = true; // Use the flow model water depth (h) for wave propagation. If false the depth is calculated using zsinit (or 0.0) and zb
	bool waveflowcoupling = true; // Feed the wave forces (radiation stress gradient) back to the flow model (only relevant when both models are running)
	int wavesolver = 2; // Explicit propagation scheme for the action balance equation (1: first order upwind with forward Euler time stepping, 2: second order upwind (MUSCL reconstruction with a limiter) with second order (Heun / SSP-RK2) time stepping [Default])
	double wavetheta = 1.0; // Slope limiter parameter for the second order upwind scheme, wavetheta in [1,2] (1: minmod, the most diffusive but most robust [Default], 2: superbee, the least diffusive)
	bool refraction = true; // Switch for depth and current induced refraction (propagation in theta-space, SWAN REFRAC)
	bool freqshift = true; // Switch for the frequency shift due to time varying depth and currents (propagation in sigma-space, SWAN FSHIFT)
	double ctmax = 0.5; // Maximum Courant number allowed for the refraction and frequency shift (cth*dt/dtheta and csig*dt/dsigma are clipped to this value, similar to the WW3 CTMAX refraction limiter)

	//*Spectral grid
	int nfreq = 25; // Number of (logarithmically distributed) frequency bins
	int ndir = 36; // Number of directional bins (full circle). 36 gives a 10 degree resolution
	double fmin = 0.0418; // Lowest discrete frequency in Hz (WW3 default 0.0418 Hz)
	double fmax = 0.4117; // Highest discrete frequency in Hz (WW3 default 0.0418*1.1^24 = 0.4117 Hz)
	double dirstart = 0.0; // Direction of the first directional bin (in degrees, nautical convention i.e. direction waves are coming from, clockwise from North)
	int nspec = 0; // Calculated in sanity check as nfreq*ndir (number of spectral bins)
	double freqfactor = 1.1; // Calculated in sanity check as (fmax/fmin)^(1/(nfreq-1))
	double dtheta = 0.0; // Calculated in sanity check as 2pi/ndir (in rad)
	int specstride = 0; // Calculated when allocating memory: nblkmem*blksize i.e. the memory distance between 2 spectral bins in the action density array

	//*Time stepping
	double wavedt = 0.0; // Wave model time step in s. If 0.0 (default) the time step is calculated based on waveCFL
	double waveCFL = 0.5; // Courant-Friedrichs-Lewy criterium for the explicit wave propagation (between 0 and 1)
	double dtcouple = 0.0; // Coupling interval (s) between the wave and the flow model. 0.0 (default) means the wave model is updated every flow time step
	double wavedepthmin = 0.05; // Minimum water depth (m) for the wave model. Cells with h < wavedepthmin are considered dry for the wave model (SWAN DEPMIN)

	//*Physics (source terms)
	// Wind input (Uses the wind forcing given to the flow model: "Wind" keyword)
	int windinput = 1; // Wind input formulation (0: off, 1: Komen et al. (1984) / Snyder (1981) [Default], 2: Janssen (1991) - to be implemented)
	bool lineargrowth = false; // Cavaleri & Malanotte-Rizzoli (1981) linear growth term (SWAN AGROW)

	// Whitecapping
	int whitecapping = 1; // Whitecapping formulation (0: off, 1: Komen et al. 1984 [Default])
	double cds = 2.36e-5; // Komen whitecapping rate coefficient (SWAN cds2)
	double stpm = 3.02e-3; // Komen Pierson-Moskowitz steepness (SWAN stpm)
	double powst = 2.0; // Komen power of steepness normalized with PM steepness (SWAN powst)
	double wcdelta = 1.0; // Komen weighting between linear and quadratic wavenumber dependency (SWAN delta)
	double powk = 1.0; // Komen power of wavenumber normalized with the mean wave number (SWAN powk)

	// Quadruplet wave-wave interaction
	int quadruplets = 1; // Quadruplet formulation (0: off, 1: Discrete Interaction Approximation (DIA) [Default])
	double dialambda = 0.25; // DIA lambda (SWAN lambda, WW3 LAMBDA)
	double diaCnl4 = 3.0e7; // DIA proportionality coefficient (SWAN Cnl4, WW3 NLPROP=2.78e7)

	// Triad wave-wave interaction
	int triads = 0; // Triad formulation (0: off [Default], 1: Lumped Triad Approximation (LTA))
	double trfac = 0.8; // LTA proportionality coefficient (SWAN trfac)
	double cutfr = 2.5; // LTA maximum frequency as a factor of the mean frequency (SWAN cutfr)

	// Depth induced breaking
	int breaking = 1; // Depth induced breaking formulation (0: off, 1: Battjes & Janssen (1978) [Default])
	double alphabr = 1.0; // Breaking dissipation proportionality coefficient (SWAN alpha)
	double gammabr = 0.73; // Breaker index Hmax/d (SWAN gamma)

	// Bottom friction
	int wavefriction = 1; // Bottom friction formulation (0: off, 1: JONSWAP (Hasselmann et al. 1973) [Default], 2: Collins (1972), 3: Madsen et al. (1988))
	double cfjon = 0.038; // JONSWAP bottom friction coefficient in m2/s3 (SWAN default 0.038, WW3 SBT1 0.067)
	double cfw = 0.015; // Collins friction coefficient
	double kn = 0.05; // Madsen equivalent roughness length (m)

	// Action density limiter
	bool wavelimiter = true; // Limit the change of action density per time step (SWAN/WW3 style limiter, Hersbach & Janssen 1999)
	double limiterfac = 0.1; // Fraction of the Phillips (PM) level used by the limiter (SWAN qb=0.1)

	//*Initial conditions
	int waveinit = 0; // Wave initial condition (0: calm sea, N=0 [Default], 1: uniform parametric JONSWAP spectrum using Hsinit, Tpinit, Dirinit, Dsprinit)
	double Hsinit = 0.0; // Initial significant wave height (m) for waveinit=1
	double Tpinit = 10.0; // Initial peak period (s) for waveinit=1
	double Dirinit = 0.0; // Initial mean wave direction (degrees nautical, coming from) for waveinit=1
	double Dsprinit = 30.0; // Initial directional spreading (degrees, directional standard deviation) for waveinit=1

	//*Boundaries
	// Wave boundary type for each side (0: no incoming waves (fully absorbing) [Default]; 1: parametric JONSWAP spectrum (Hsbnd,Tpbnd,Dirbnd,Dsprbnd); 2: spectral file (to be implemented))
	int leftwavebnd = 0;
	int rightwavebnd = 0;
	int topwavebnd = 0;
	int botwavebnd = 0;
	double Hsbnd = 0.0; // Boundary significant wave height (m)
	double Tpbnd = 10.0; // Boundary peak period (s)
	double Dirbnd = 0.0; // Boundary mean wave direction (degrees nautical, coming from)
	double Dsprbnd = 30.0; // Boundary directional spreading (degrees, directional standard deviation)
	double gammajsp = 3.3; // JONSWAP peak enhancement factor used for parametric spectra (initial and boundary conditions)
	double wavebndtaper = 0.0; // Number of seconds to taper the wave boundary from 0 to full value (similar to bndtaper for the flow)

};


/**
 * @brief List of the (2D) wave output variables that can be requested with the "outvars" keyword.
 * Hs: significant wave height; Tm01, Tm02: mean periods; Tp: peak period; Dir: mean direction (nautical);
 * Dspr: directional spreading; km: mean wave number; Urms: rms bottom orbital velocity; Qb: fraction of breaking waves;
 * Dbr, Dfr, Dwc: dissipation by breaking, bottom friction and whitecapping; Sxx, Sxy, Syy: radiation stresses;
 * Fwx, Fwy: wave forces.
 */
inline std::vector<std::string> WaveOutputVarNames()
{
	return { "Hs","Tm01","Tm02","Tp","Dir","Dspr","km","Urms","Qb","Dbr","Dfr","Dwc","Sxx","Sxy","Syy","Fwx","Fwy" };
}


// End of global definition
#endif

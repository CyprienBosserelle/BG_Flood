#ifndef WAVEPARAM_H
#define WAVEPARAM_H

#include "General.h"

// Maximum number of direction bins in one sweep quadrant of the implicit solver (local arrays size)
#define WAVE_MAXQDIR 130

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
	int wavemodel = 0; // Spectral wave model (0: off, flow only [Default], 1: implicit SWAN-like solver (four-sweep Gauss-Seidel, unconditionally stable) run every dtcouple seconds (loose coupling, e.g. every 10 min to 1 h of model time), 2: explicit solver run every flow time step or every dtcouple (tight coupling)). "wavemodel = true" is the same as 1. The flow model can be turned off using "flowmodel = false"
	bool wavecurrent = true; // Include wave-current interaction (Doppler shift and current refraction) using the flow model velocities u,v
	bool wavewaterlevel = true; // Use the flow model water depth (h) for wave propagation. If false the depth is calculated using zsinit (or 0.0) and zb
	bool waveflowcoupling = true; // Feed the wave forces (radiation stress gradient) back to the flow model (only relevant when both models are running)
	int wavesolver = 2; // (wavemodel = 2 only) Explicit propagation scheme for the action balance equation (1: first order upwind with forward Euler time stepping, 2: second order upwind (MUSCL reconstruction with a limiter) with second order (Heun / SSP-RK2) time stepping [Default])
	double wavetheta = 1.0; // Slope limiter parameter for the second order upwind scheme, wavetheta in [1,2] (1: minmod, the most diffusive but most robust [Default], 2: superbee, the least diffusive)
	bool refraction = true; // Switch for depth and current induced refraction (propagation in theta-space, SWAN REFRAC)
	bool freqshift = true; // Switch for the frequency shift due to time varying depth and currents (propagation in sigma-space, SWAN FSHIFT)
	int wavebinchunk = 0; // Number of spectral bins propagated together in one batch (one kernel launch) in geographic space. 0 (default): ndir bins (one frequency). Larger values are faster on the GPU but need 4*wavebinchunk extra 2D arrays of memory (wavebinchunk = nspec is the fastest)
	int nchunk = 0; // Calculated in sanity check from wavebinchunk (number of bins per batch)
	bool staticmedium = false; // Calculated in sanity check: true when the depth and currents seen by the wave model do not change (no flow model, or no water level and current coupling) so k, cg, the medium gradients and the time step are only calculated once
	double ctmax = 0.5; // (wavemodel = 2 only) Maximum Courant number allowed for the refraction and frequency shift (cth*dt/dtheta and csig*dt/dsigma are clipped to this value, similar to the WW3 CTMAX refraction limiter)

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
	double dtcouple = 0.0; // Coupling interval (s) between the wave and the flow model: the wave model is updated every dtcouple seconds of model time (with the flow depth and currents at that time) and the wave field is kept until the next update. For wavemodel = 2, 0.0 (default) means every flow time step. For wavemodel = 1, 0.0 means 600 s (10 min). In wave only simulations this is the wave time step of the implicit solver
	bool wavestationary = true; // (wavemodel = 1 only) true: stationary solution at each coupling time (SWAN MODE STATIONARY) [Default], false: nonstationary solution with an implicit (backward Euler) time step equal to the coupling interval (SWAN MODE NONSTATIONARY)
	double wavecdd = 1.0; // (wavemodel = 1 only) Blending of central (0) and first order upwind (1) differences for the refraction (SWAN CDD). 1 (first order upwind) [Default] is the most robust, the SWAN default is 0.5 (less diffusive but not suited to very narrow directional spectra)
	double wavecss = 1.0; // (wavemodel = 1 only) Blending of central (0) and first order upwind (1) differences for the frequency shift (SWAN CSS). 1 (first order upwind) [Default] is the most robust, the SWAN default is 0.5 (can slow down the convergence with strong currents)
	bool wavewarmstart = false; // (wavemodel = 1 only) Warm start: the first wave update (at the start of the simulation) is a stationary solution and all the following updates are nonstationary (implies wavestationary = false)
	int wavemaxiter = 50; // (wavemodel = 1 only) Maximum number of iterations (each iteration is 4 sweeps) of the implicit solver per coupling (SWAN MXITST)
	double wavedrel = 0.01; // (wavemodel = 1 only) Convergence criterion: relative change of Hs between 2 iterations (SWAN DREL)
	double wavedhabs = 0.005; // (wavemodel = 1 only) Convergence criterion: absolute change of Hs (m) between 2 iterations (SWAN DHABS)
	double wavenpnts = 99.5; // (wavemodel = 1 only) Convergence criterion: percentage of wet cells where the Hs criteria must be met to stop the iterations (SWAN NPNTS)
	int qstart[4] = { 0, 0, 0, 0 }; // Calculated at initialisation: first direction bin of each sweep quadrant (implicit solver)
	int qlen[4] = { 0, 0, 0, 0 }; // Calculated at initialisation: number of direction bins in each sweep quadrant (implicit solver)
	double wavedepthmin = 0.05; // Minimum water depth (m) for the wave model. Cells with h < wavedepthmin are considered dry for the wave model (SWAN DEPMIN)

	//*Physics (source terms)
	// Wind input (Uses the wind forcing given to the flow model: "Wind" keyword)
	int windinput = 1; // Wind input formulation (0: off, 1: Komen et al. (1984) / Snyder (1981) [Default]). Uses the flow model wind forcing ("Wind" keyword), turned off if no wind is given
	double rhoair = 1.225; // Air density (kg/m3) used by the wind input
	bool lineargrowth = true; // Cavaleri & Malanotte-Rizzoli (1981) linear growth term (SWAN AGROW). On by default so that waves can grow from a calm sea (the exponential wind input is proportional to the existing energy)

	// Whitecapping
	int whitecapping = 1; // Whitecapping formulation (0: off, 1: Komen et al. 1984 [Default])
	double cds = 2.36e-5; // Komen whitecapping rate coefficient (SWAN cds2)
	double stpm = 3.02e-3; // Komen Pierson-Moskowitz steepness (SWAN stpm)
	double powst = 2.0; // Komen power of steepness normalized with PM steepness (SWAN powst)
	double wcdelta = 1.0; // Komen weighting between linear and quadratic wavenumber dependency (SWAN delta)
	double powk = 1.0; // Komen power of wavenumber normalized with the mean wave number (SWAN powk)

	// Quadruplet wave-wave interaction
	int quadruplets = 0; // Quadruplet formulation (0: off [Default], 1: Discrete Interaction Approximation (DIA) - not implemented yet)
	double dialambda = 0.25; // DIA lambda (SWAN lambda, WW3 LAMBDA)
	double diaCnl4 = 3.0e7; // DIA proportionality coefficient (SWAN Cnl4, WW3 NLPROP=2.78e7)

	// Triad wave-wave interaction
	int triads = 0; // Triad formulation (0: off [Default], 1: Lumped Triad Approximation (LTA) - not implemented yet)
	double trfac = 0.8; // LTA proportionality coefficient (SWAN trfac)
	double cutfr = 2.5; // LTA maximum frequency as a factor of the mean frequency (SWAN cutfr)

	// Depth induced breaking
	int breaking = 1; // Depth induced breaking formulation (0: off, 1: Battjes & Janssen (1978) [Default])
	double alphabr = 1.0; // Breaking dissipation proportionality coefficient (SWAN alpha)
	double gammabr = 0.73; // Breaker index Hmax/d (SWAN gamma)

	// Bottom friction
	int wavefriction = 1; // Bottom friction formulation (0: off, 1: JONSWAP (Hasselmann et al. 1973) [Default], 2: Collins (1972))
	double cfjon = 0.038; // JONSWAP bottom friction coefficient in m2/s3 (SWAN default 0.038, WW3 SBT1 0.067)
	double cfw = 0.015; // Collins friction coefficient
	double kn = 0.05; // Madsen equivalent roughness length (m)

	// Action density limiter
	bool wavelimiter = true; // Limit the change of action density per time step / iteration when the wind input is active (SWAN/WW3 style limiter, Hersbach & Janssen 1999): |dN| <= limiterfac * Phillips level
	double limiterfac = 0.1; // Fraction of the Phillips (PM) level used by the limiter (SWAN qb=0.1)

	//*Initial conditions
	int waveinit = 0; // Wave initial condition (0: calm sea, N=0 [Default], 1: uniform parametric JONSWAP spectrum using Hsinit, Tpinit, Dirinit, Dsprinit)
	double Hsinit = 0.0; // Initial significant wave height (m) for waveinit=1
	double Tpinit = 10.0; // Initial peak period (s) for waveinit=1
	double Dirinit = 0.0; // Initial mean wave direction (degrees nautical, coming from) for waveinit=1
	double Dsprinit = 30.0; // Initial directional spreading (degrees, directional standard deviation) for waveinit=1

	//*Boundaries
	// The wave boundaries are defined in the forcing (wavebndseg, waveleft, waveright, wavetop, wavebot keywords, see Forcing.h)
	int waveaoibnd = 0; // Wave boundary type for the boundary blocks that are not in any wave boundary segment (0: absorbing i.e. no incoming waves [Default], 1: Neumann i.e. zero gradient)
	double gammajsp = 3.3; // JONSWAP peak enhancement factor used for parametric spectra (initial condition, and boundary conditions when the forcing file does not provide gamma)
	double wavebndtaper = 0.0; // Number of seconds to taper the wave boundary from 0 to full value (similar to bndtaper for the flow)

};


/**
 * @brief List of the (2D) wave output variables that can be requested with the "outvars" keyword.
 * Hs: significant wave height; Tm01, Tm02: mean periods; Tmm10: energy period Tm-1,0; Tp: peak period; Dir: mean direction (nautical); Dp: peak direction;
 * Dspr: directional spreading; km: mean wave number; Urms: rms bottom orbital velocity; Qb: fraction of breaking waves;
 * Dbr, Dfr, Dwc: dissipation by breaking, bottom friction and whitecapping; Sxx, Sxy, Syy: radiation stresses;
 * Fwx, Fwy: wave forces.
 * Keep in sync with the supported output names in readparamstr (ReadInput.cu).
 */
inline std::vector<std::string> WaveOutputVarNames()
{
	return { "Hs","Tm01","Tm02","Tmm10","Tp","Dir","Dp","Dspr","km","Urms","Qb","Dbr","Dfr","Dwc","Sxx","Sxy","Syy","Fwx","Fwy" };
}


// End of global definition
#endif

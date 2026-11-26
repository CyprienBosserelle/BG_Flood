#ifndef WAVEARRAYS_H
#define WAVEARRAYS_H

#include "General.h"
#include "Arrays.h"

/*
* Wave model memory structures
* ============================
* This is the wave-side equivalent of Arrays.h (Model<T> for the flow side).
*
* The wave model uses the SAME block mesh as the flow model (XModel.blocks: block neighbours, level, xo, yo, active blocks)
* so the wave model does not hold any block information. Similarly the bathymetry (zb), the water depth (h) and currents (u,v)
* are read from the flow Model<T> structure.
*
* Memory layout of spectral arrays
* --------------------------------
* Spectral arrays (e.g. the action density N) hold nspec = nfreq*ndir bins for each cell.
* They are stored as nspec consecutive "flow-like" 2D arrays (each of size nblkmem*blksize):
*
*     N[ispec * specstride + i]   with   ispec = ifreq * ndir + idir,   specstride = nblkmem * blksize
*                                 and    i = memloc(halowidth, blkmemwidth, ix, iy, ib)
*
* This means a pointer to a single spectral bin (N + ispec*specstride) has exactly the same layout as any other
* flow variable (e.g. XModel.evolv.h) so all existing block functions (halo filling, gradients, refine/coarsen,
* interpolation, output...) can be reused directly on each bin. This layout is also coalesced on the GPU
* (neighbouring threads access neighbouring cells of the same spectral bin).
*
* Bulk (2D) arrays have the same layout as the flow variables and can be outputted using the existing output machinery
* (they are registered in XModel.OutputVarMap).
*/


/**
 * @brief Spectral grid (frequency and direction bins). Small 1D arrays.
 * @tparam T Data type
 */
template <class T>
struct WaveSpecGridP
{
	/** Radian frequency of each frequency bin (rad/s) size(nfreq) */
	T* sigma;
	/** Width of each frequency bin (rad/s) size(nfreq) */
	T* dsigma;
	/** Cartesian direction of propagation (rad, counter-clockwise from x-axis, "going to") size(ndir) */
	T* theta;
	/** cos(theta) size(ndir) */
	T* costh;
	/** sin(theta) size(ndir) */
	T* sinth;
};

/**
 * @brief Spectral evolving variable (action density). Equivalent of EvolvingP for the wave model.
 * @tparam T Data type
 */
template <class T>
struct WaveSpectrumP
{
	/** Action density N(sigma,theta) = E(sigma,theta)/sigma in m2.s/rad2 size(nspec*nblkmem*blksize) */
	T* N;
};

/**
 * @brief Spectral advance variable (rate of change of action density). Equivalent of AdvanceP for the wave model.
 * @tparam T Data type
 */
template <class T>
struct WaveAdvanceP
{
	/** Rate of change of N (propagation + sources) size(nspec*nblkmem*blksize) */
	T* dN;
};

/**
 * @brief Integrated (bulk) wave parameters. Equivalent of EvolvingP_M for the wave model (2D arrays).
 * @tparam T Data type
 */
template <class T>
struct WaveBulkP
{
	/** Significant wave height Hm0 (m) */
	T* Hs;
	/** Mean wave period Tm01 (s) */
	T* Tm01;
	/** Mean wave period Tm02 (s) */
	T* Tm02;
	/** Peak period (s) */
	T* Tp;
	/** Mean wave direction (degrees nautical, coming from) */
	T* Dir;
	/** Directional spreading (degrees) */
	T* Dspr;
	/** Mean wave number based on Tm01 (rad/m) */
	T* km;
	/** Root mean square near bottom orbital velocity (m/s) (SWAN URMS; SWAN UBOT = sqrt(2)*Urms) */
	T* Urms;
	/** Fraction of breaking waves (-) */
	T* Qb;
};

/**
 * @brief Integrated dissipation terms (2D arrays). Diagnostic and also used for coupling (e.g. wave-enhanced friction, roller).
 * @tparam T Data type
 */
template <class T>
struct WaveDissipP
{
	/** Energy dissipation by depth induced breaking (W/m2) */
	T* Dbr;
	/** Energy dissipation by bottom friction (W/m2) */
	T* Dfr;
	/** Energy dissipation by whitecapping (W/m2) */
	T* Dwc;
};

/**
 * @brief Wave forcing on the flow (coupling wave -> flow) (2D arrays).
 * @tparam T Data type
 */
template <class T>
struct WaveFlowForcingP
{
	/** Radiation stress component Sxx (N/m) */
	T* Sxx;
	/** Radiation stress component Sxy (N/m) */
	T* Sxy;
	/** Radiation stress component Syy (N/m) */
	T* Syy;
	/** Wave force in x direction per unit area (N/m2) i.e. -(dSxx/dx + dSxy/dy) */
	T* Fx;
	/** Wave force in y direction per unit area (N/m2) i.e. -(dSxy/dx + dSyy/dy) */
	T* Fy;
};

/**
 * @brief Wave time keeping (scalar only, kept on the CPU structure).
 */
struct WaveTimeP
{
	/** Wave model time (s) */
	double totaltime = 0.0;
	/** Last wave time step (s) */
	double dt = 0.0;
	/** Maximum stable (or user prescribed) wave time step (s) */
	double dtmax = 0.0;
	/** Next time the wave model needs to be updated (coupling time) (s) */
	double nextcoupletime = 0.0;
	/** Number of wave steps done */
	int nstep = 0;
};


/**
 * @brief Main wave model structure holding all the wave memory pointers. Equivalent of Model<T> for the flow side.
 *
 * Like the flow model there is a CPU (XWave) and a GPU (XWave_g) version of this structure.
 * @tparam T Data type
 */
template <class T>
struct WaveModel
{
	// Spectral grid
	WaveSpecGridP<T> spec;

	// Spectral arrays
	WaveSpectrumP<T> action;
	WaveSpectrumP<T> action_o; // action density at the previous step (for predictor/corrector, iterative or limiter)
	WaveAdvanceP<T> adv;

	// 2D arrays
	WaveBulkP<T> bulk;
	WaveDissipP<T> dissip;
	WaveFlowForcingP<T> force;

	// Time keeping
	// time reuses the flow TimeP structure: time.dtmax (per cell maximum stable wave time step) and time.arrmin
	// (work array) so the existing time step reduction functions (timestepreductionCPU / CalctimestepGPU) can be reused
	TimeP<T> time;
	WaveTimeP clock; // wave model clock (scalars, only meaningful in the CPU structure)
};



// End of global definition
#endif

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
* Implicit wave solver (wavemodel = 1)
* ====================================
* SWAN-like implicit solver of the action balance equation for loose coupling (the wave model is updated every
* dtcouple seconds, typically 10 min to 1 h, with the flow depth and currents at that time).
*
* - Geographic space: first order upwind (SWAN BSBT), implicit, solved with the four-sweep Gauss-Seidel technique
*   (Zijlema & van der Westhuysen 2005). In sweep s only the directions of the quadrant s are updated
*   (s=0: 0-90deg, s=1: 90-180deg, s=2: 180-270deg, s=3: 270-360deg, cartesian) so that, for each of these directions,
*   the upwind cells have already been updated in the same sweep.
* - Spectral space (in each cell): implicit first order upwind refraction in theta solved with a tridiagonal (Thomas)
*   algorithm over the directions of the quadrant (as in SWAN) and implicit first order upwind frequency shift
*   (Gauss-Seidel in sigma from the low to the high frequencies).
* - Time: stationary (no time derivative, SWAN default) or nonstationary (backward Euler, dt = coupling interval).
* - The sweeps (4 per iteration) are repeated until the SWAN convergence criterion on Hs is met (wavedrel, wavedhabs,
*   wavenpnts) or wavemaxiter iterations.
*
* Adaptive block mesh / GPU: in each sweep the blocks are processed by "levels" (a block only depends on its upwind
* neighbour blocks, so all the blocks of a level are independent); the halo of the blocks of a level is filled
* (reusing the flow halo functions on the virtual blocks of all the spectral bins) just before the level is processed.
* Inside a block the cells are processed in upwind order (on the GPU: diagonal wavefront, one thread per cell).
*/

#include "WaveImplicit.h"


// Sweep directions: s=0 (+x,+y), s=1 (-x,+y), s=2 (-x,-y), s=3 (+x,-y)
__host__ __device__ inline int WaveSweepSx(int s) { return (s == 0 || s == 3) ? 1 : -1; }
__host__ __device__ inline int WaveSweepSy(int s) { return (s < 2) ? 1 : -1; }


/**
 * @brief Set up the implicit solver: sweep quadrants, block levels of each sweep, boundary face slots.
 *
 * Must be called after the mesh is final (after the adaptation) and after the wave boundary segments block lists are known.
 */
template <class T> void InitWaveImplicit(Param XParam, WaveParam& XWaveParam, Forcing<float>& XForcing, Model<T>& XModel, WaveModel<T>& XWave)
{
	int nd = XWaveParam.ndir;
	int nb = XParam.nblkmem;
	int nspec = XWaveParam.nspec;
	WaveImplicitP<T>& imp = XWave.imp;

	//==============================
	// Direction bins of each sweep quadrant (contiguous in theta, the directions on the axes belong to 2 quadrants)
	for (int s = 0; s < 4; s++)
	{
		double sx = WaveSweepSx(s), sy = WaveSweepSy(s);
		std::vector<bool> inq(nd);
		int n = 0;
		for (int d = 0; d < nd; d++)
		{
			inq[d] = (XWave.spec.costh[d] * sx > -1e-6) && (XWave.spec.sinth[d] * sy > -1e-6);
			n += inq[d] ? 1 : 0;
		}
		int d0 = 0;
		if (n > 0 && n < nd)
		{
			for (int d = 0; d < nd; d++)
			{
				if (inq[d] && !inq[(d - 1 + nd) % nd])
				{
					d0 = d;
					break;
				}
			}
		}
		XWaveParam.qstart[s] = d0;
		XWaveParam.qlen[s] = n;
		if (n + 2 > WAVE_MAXQDIR)
		{
			log("ERROR: too many directions in a sweep quadrant of the implicit wave solver");
			exit(1);
		}
	}

	//==============================
	// Block levels for each sweep: a block only depends on its upwind neighbours
	std::vector<int> lev(nb, 0);
	for (int s = 0; s < 4; s++)
	{
		int sx = WaveSweepSx(s), sy = WaveSweepSy(s);
		for (int ibl = 0; ibl < XParam.nblk; ibl++)
		{
			lev[XModel.blocks.active[ibl]] = 0;
		}
		bool changed = true;
		int npass = 0;
		while (changed)
		{
			changed = false;
			npass++;
			for (int ibl = 0; ibl < XParam.nblk; ibl++)
			{
				int ib = XModel.blocks.active[ibl];
				int up[4];
				up[0] = sx > 0 ? XModel.blocks.LeftBot[ib] : XModel.blocks.RightBot[ib];
				up[1] = sx > 0 ? XModel.blocks.LeftTop[ib] : XModel.blocks.RightTop[ib];
				up[2] = sy > 0 ? XModel.blocks.BotLeft[ib] : XModel.blocks.TopLeft[ib];
				up[3] = sy > 0 ? XModel.blocks.BotRight[ib] : XModel.blocks.TopRight[ib];
				int l = 0;
				for (int u = 0; u < 4; u++)
				{
					if (up[u] != ib)
					{
						l = utils::max(l, lev[up[u]] + 1);
					}
				}
				if (l != lev[ib])
				{
					lev[ib] = l;
					changed = true;
				}
			}
			if (npass > XParam.nblk + 1)
			{
				log("ERROR: could not order the blocks for the implicit wave solver sweeps");
				exit(1);
			}
		}

		int nlev = 0;
		for (int ibl = 0; ibl < XParam.nblk; ibl++)
		{
			nlev = utils::max(nlev, lev[XModel.blocks.active[ibl]] + 1);
		}
		imp.nlevels[s] = nlev;
		AllocateCPU(nlev + 1, 1, imp.levelstart[s]);
		AllocateCPU(XParam.nblk, 1, imp.order[s]);
		AllocateCPU(XParam.nblk * nspec, 1, imp.vorder[s]);

		int pos = 0;
		for (int l = 0; l < nlev; l++)
		{
			imp.levelstart[s][l] = pos;
			for (int ibl = 0; ibl < XParam.nblk; ibl++)
			{
				int ib = XModel.blocks.active[ibl];
				if (lev[ib] == l)
				{
					imp.order[s][pos] = ib;
					pos++;
				}
			}
			// virtual blocks of all the bins for the blocks of this level
			int nbl = pos - imp.levelstart[s][l];
			for (int j = 0; j < nspec; j++)
			{
				for (int k = 0; k < nbl; k++)
				{
					imp.vorder[s][imp.levelstart[s][l] * nspec + j * nbl + k] = imp.order[s][imp.levelstart[s][l] + k] + j * nb;
				}
			}
		}
		imp.levelstart[s][nlev] = pos;
	}
	log("\tImplicit wave solver: " + std::to_string(imp.nlevels[0]) + "/" + std::to_string(imp.nlevels[1]) + "/" + std::to_string(imp.nlevels[2]) + "/" + std::to_string(imp.nlevels[3]) + " block levels in the 4 sweeps");

	//==============================
	// Boundary face slots (one per block side on the domain boundary, ordered by wave boundary segment)
	AllocateCPU(4 * nb, 1, imp.bslot);
	for (int n = 0; n < 4 * nb; n++)
	{
		imp.bslot[n] = -1;
	}
	int nseg = XForcing.wavebndseg.size();
	AllocateCPU(nseg + 1, 1, imp.segslot);
	int nslot = 0;
	for (int iseg = 0; iseg < nseg; iseg++)
	{
		const wavebndsegment& seg = XForcing.wavebndseg[iseg];
		nslot += seg.left.nblk + seg.right.nblk + seg.bot.nblk + seg.top.nblk;
	}
	imp.nslot = nslot;
	AllocateCPU(utils::max(nslot, 1), 1, imp.slottype);
	AllocateCPU(utils::max(nslot, 1), 1, imp.slotblk);
	AllocateCPU(utils::max(nslot, 1), 1, imp.slotside);
	AllocateCPU(utils::max(nslot, 1) * nspec, XParam.blkwidth, imp.Nbf);
	FillCPU(utils::max(nslot, 1) * nspec, XParam.blkwidth, T(0.0), imp.Nbf);

	int slot = 0;
	for (int iseg = 0; iseg < nseg; iseg++)
	{
		const wavebndsegment& seg = XForcing.wavebndseg[iseg];
		imp.segslot[iseg] = slot;
		const bndsegmentside* sides[4] = { &seg.left, &seg.right, &seg.bot, &seg.top };
		for (int is = 0; is < 4; is++)
		{
			for (int ibl = 0; ibl < sides[is]->nblk; ibl++)
			{
				int ib = sides[is]->blk[ibl];
				imp.bslot[is * nb + ib] = slot;
				imp.slottype[slot] = seg.type;
				imp.slotblk[slot] = ib;
				imp.slotside[slot] = is;
				slot++;
			}
		}
	}
	imp.segslot[nseg] = slot;

	//==============================
	// Convergence arrays
	AllocateCPU(nb, XParam.blksize, imp.Hsold);
	AllocateCPU(nb, XParam.blksize, imp.conv);
	FillCPU(nb, XParam.blksize, T(0.0), imp.Hsold);
	FillCPU(nb, XParam.blksize, T(-1.0), imp.conv);
}
template void InitWaveImplicit<float>(Param XParam, WaveParam& XWaveParam, Forcing<float>& XForcing, Model<float>& XModel, WaveModel<float>& XWave);
template void InitWaveImplicit<double>(Param XParam, WaveParam& XWaveParam, Forcing<float>& XForcing, Model<double>& XModel, WaveModel<double>& XWave);


/**
 * @brief Copy the implicit solver arrays to the GPU (the level start indices stay on the host).
 */
template <class T> void SetupWaveImplicitGPU(Param XParam, WaveParam XWaveParam, WaveModel<T>& XWave, WaveModel<T>& XWave_g)
{
	int nb = XParam.nblkmem;
	int nspec = XWaveParam.nspec;
	WaveImplicitP<T>& h = XWave.imp;
	WaveImplicitP<T>& g = XWave_g.imp;

	for (int s = 0; s < 4; s++)
	{
		AllocateGPU(XParam.nblk, 1, g.order[s]);
		CopytoGPU(XParam.nblk, 1, h.order[s], g.order[s]);
		AllocateGPU(XParam.nblk * nspec, 1, g.vorder[s]);
		CopytoGPU(XParam.nblk * nspec, 1, h.vorder[s], g.vorder[s]);
		g.levelstart[s] = h.levelstart[s]; // host memory
		g.nlevels[s] = h.nlevels[s];
	}
	int ns = utils::max(h.nslot, 1);
	AllocateGPU(4 * nb, 1, g.bslot);
	CopytoGPU(4 * nb, 1, h.bslot, g.bslot);
	AllocateGPU(ns, 1, g.slottype);
	CopytoGPU(ns, 1, h.slottype, g.slottype);
	AllocateGPU(ns, 1, g.slotblk);
	CopytoGPU(ns, 1, h.slotblk, g.slotblk);
	AllocateGPU(ns, 1, g.slotside);
	CopytoGPU(ns, 1, h.slotside, g.slotside);
	AllocateGPU(ns * nspec, XParam.blkwidth, g.Nbf);
	CopytoGPU(ns * nspec, XParam.blkwidth, h.Nbf, g.Nbf);
	g.segslot = h.segslot; // host memory
	g.nslot = h.nslot;

	AllocateGPU(nb, XParam.blksize, g.Hsold);
	CopytoGPU(nb, XParam.blksize, h.Hsold, g.Hsold);
	AllocateGPU(nb, XParam.blksize, g.conv);
	CopytoGPU(nb, XParam.blksize, h.conv, g.conv);
}
template void SetupWaveImplicitGPU<float>(Param XParam, WaveParam XWaveParam, WaveModel<float>& XWave, WaveModel<float>& XWave_g);
template void SetupWaveImplicitGPU<double>(Param XParam, WaveParam XWaveParam, WaveModel<double>& XWave, WaveModel<double>& XWave_g);


/**
 * @brief Incoming action density of all the bins on one boundary face (slot, tx), stored in XWave.imp.Nbf[(slot*blkwidth + tx)*nspec + ispec].
 * Type 2 (JONSWAP): spectrum from the bulk parameters (tapered by bndfac); type 0 (absorbing): 0;
 * type 1 (Neumann) is handled directly in the cell solver (incoming action density = interior action density).
 */
template <class T> __host__ __device__ void WaveImplicitBndFace(const WaveParam& XWaveParam, int bw, int slot, int tx, int type, T Hs, T Tp, T Dir, T Dspr, T gamma, T m0n, T Dsumn, T bndfac, const WaveModel<T>& XWave)
{
	int nspec = XWaveParam.nspec;
	size_t base = (size_t(slot) * size_t(bw) + size_t(tx)) * size_t(nspec);
	if (type == 2 && m0n <= T(0.0))
	{
		WaveJonswapNorm(XWaveParam, XWave.spec, Tp, Dir, Dspr, gamma, m0n, Dsumn);
	}
	for (int ispec = 0; ispec < nspec; ispec++)
	{
		XWave.imp.Nbf[base + ispec] = (type == 2) ? bndfac * WaveJonswapNbin(XWaveParam, XWave.spec, ispec, Hs, Tp, Dir, Dspr, gamma, m0n, Dsumn) : T(0.0);
	}
}
template __host__ __device__ void WaveImplicitBndFace<float>(const WaveParam& XWaveParam, int bw, int slot, int tx, int type, float Hs, float Tp, float Dir, float Dspr, float gamma, float m0n, float Dsumn, float bndfac, const WaveModel<float>& XWave);
template __host__ __device__ void WaveImplicitBndFace<double>(const WaveParam& XWaveParam, int bw, int slot, int tx, int type, double Hs, double Tp, double Dir, double Dspr, double gamma, double m0n, double Dsumn, double bndfac, const WaveModel<double>& XWave);


/**
 * @brief Implicit update of the action density of one cell for the directions of sweep quadrant "sweep" (shared CPU/GPU).
 *
 * For each bin (frequency j, direction d of the quadrant), first order upwind implicit discretisation:
 *
 *   (1/dt + Out_geo + Out_theta + Out_sigma) N = N_old/dt + In_geo + In_sigma + (theta coupling with the neighbouring directions)
 *
 * - Out_geo/In_geo: outgoing/incoming fluxes through the 4 faces (face velocity = mean of the two cells, the incoming action
 *   density is the latest value of the upwind neighbour, boundary faces use the boundary condition)
 * - theta: implicit tridiagonal system over the directions of the quadrant (Thomas algorithm), the directions just outside
 *   the quadrant are taken at their latest value (as in SWAN)
 * - sigma: Gauss-Seidel from the low to the high frequencies (the lower frequency is already updated)
 * - theta and sigma face values blend first order upwind and central differences (SWAN CDD and CSS: wavecdd, wavecss;
 *   1 = upwind). If the blended system is not diagonally dominant in a cell/frequency, pure upwind is used there.
 * Dry cells hold no action density.
 *
 * @param invdt 1/dt for the nonstationary mode, 0 for the stationary mode
 */
template <class T> __host__ __device__ void WaveImplicitCell(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int sweep, T invdt, const BlockP<T>& XBlock, const WaveModel<T>& XWave)
{
	const int halowidth = XParam.halowidth;
	const int blkmemwidth = XParam.blkmemwidth;
	const int bw = XParam.blkwidth;
	const int nb = XParam.nblkmem;
	const int stride = XWaveParam.specstride;
	const int nf = XWaveParam.nfreq;
	const int nd = XWaveParam.ndir;
	const int nspec = XWaveParam.nspec;
	const int qs = XWaveParam.qstart[sweep];
	const int ql = XWaveParam.qlen[sweep];
	const T hmin = T(XWaveParam.wavedepthmin);
	const T dth = T(XWaveParam.dtheta);
	const bool dorefrac = XWaveParam.refraction && (nd > 1);
	const bool dofshift = XWaveParam.freqshift && (nf > 1);

	if (ql <= 0)
	{
		return;
	}

	T* N = XWave.action.N;
	T* No = XWave.action_o.N;

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	if (XWave.medium.h[i] <= hmin)
	{
		for (int ifreq = 0; ifreq < nf; ifreq++)
		{
			for (int m = 0; m < ql; m++)
			{
				N[specloc(stride, ifreq * nd + (qs + m) % nd, i)] = T(0.0);
			}
		}
		return;
	}

	T delta = calcres(T(XParam.delta), XBlock.level[ib]);
	T invdelta = T(1.0) / delta;

	int inb[4];
	inb[0] = memloc(halowidth, blkmemwidth, ix - 1, iy, ib); // left
	inb[1] = memloc(halowidth, blkmemwidth, ix + 1, iy, ib); // right
	inb[2] = memloc(halowidth, blkmemwidth, ix, iy - 1, ib); // bottom
	inb[3] = memloc(halowidth, blkmemwidth, ix, iy + 1, ib); // top

	// Domain boundary faces
	bool lowerhalf = iy < (bw / 2);
	bool lefthalf = ix < (bw / 2);
	bool bnd[4];
	bnd[0] = (ix == 0) && ((lowerhalf ? XBlock.LeftBot[ib] : XBlock.LeftTop[ib]) == ib);
	bnd[1] = (ix == bw - 1) && ((lowerhalf ? XBlock.RightBot[ib] : XBlock.RightTop[ib]) == ib);
	bnd[2] = (iy == 0) && ((lefthalf ? XBlock.BotLeft[ib] : XBlock.BotRight[ib]) == ib);
	bnd[3] = (iy == bw - 1) && ((lefthalf ? XBlock.TopLeft[ib] : XBlock.TopRight[ib]) == ib);
	int slot[4], btype[4], tx[4];
	tx[0] = iy; tx[1] = iy; tx[2] = ix; tx[3] = ix;
	bool wetn[4];
	for (int f = 0; f < 4; f++)
	{
		slot[f] = bnd[f] ? XWave.imp.bslot[f * nb + ib] : -1;
		btype[f] = (slot[f] >= 0) ? XWave.imp.slottype[slot[f]] : 0;
		wetn[f] = XWave.medium.h[inb[f]] > hmin;
	}

	T u = XWave.medium.u[i], v = XWave.medium.v[i];
	T un[4], vn[4];
	for (int f = 0; f < 4; f++)
	{
		un[f] = XWave.medium.u[inb[f]];
		vn[f] = XWave.medium.v[inb[f]];
	}
	T dhdx = XWave.medium.dhdx[i], dhdy = XWave.medium.dhdy[i], dhdt = XWave.medium.dhdt[i];
	T dudx = XWave.medium.dudx[i], dudy = XWave.medium.dudy[i];
	T dvdx = XWave.medium.dvdx[i], dvdy = XWave.medium.dvdy[i];
	T depterm = dhdt + u * dhdx + v * dhdy;

	T Dg[WAVE_MAXQDIR], Lg[WAVE_MAXQDIR], Ug[WAVE_MAXQDIR], Rg[WAVE_MAXQDIR];
	T cth[WAVE_MAXQDIR + 2];

	for (int ifreq = 0; ifreq < nf; ifreq++)
	{
		size_t ik = specloc(stride, ifreq, i);
		T k = XWave.kin.k[ik];
		T cg = XWave.kin.cg[ik];
		T dsdd = XWave.kin.dsdd[ik];
		T sdh = (k > T(0.0)) ? dsdd / k : T(0.0);
		T dsig = XWave.spec.dsigma[ifreq];

		T cgn[4];
		for (int f = 0; f < 4; f++)
		{
			cgn[f] = XWave.kin.cg[specloc(stride, ifreq, inb[f])];
		}

		// sigma velocity ingredients of the neighbouring frequencies (same cell)
		T kjm = T(0.0), cgjm = T(0.0), dsddjm = T(0.0), kjp = T(0.0), cgjp = T(0.0), dsddjp = T(0.0);
		if (dofshift && ifreq > 0)
		{
			size_t ikm = specloc(stride, ifreq - 1, i);
			kjm = XWave.kin.k[ikm]; cgjm = XWave.kin.cg[ikm]; dsddjm = XWave.kin.dsdd[ikm];
		}
		if (dofshift && ifreq < nf - 1)
		{
			size_t ikp = specloc(stride, ifreq + 1, i);
			kjp = XWave.kin.k[ikp]; cgjp = XWave.kin.cg[ikp]; dsddjp = XWave.kin.dsdd[ikp];
		}

		// theta velocity of the quadrant directions and of the 2 directions just outside the quadrant
		for (int m = -1; m <= ql; m++)
		{
			int d = (qs + m + nd) % nd;
			T c = XWave.spec.costh[d], s = XWave.spec.sinth[d];
			cth[m + 1] = dorefrac ? sdh * (s * dhdx - c * dhdy) + c * (s * dudx - c * dudy) + s * (s * dvdx - c * dvdy) : T(0.0);
		}

		// Two passes: with the SWAN central/upwind blending (wavecdd, wavecss) and, if the system is not diagonally
		// dominant in this cell/frequency, with pure first order upwind (always stable)
		for (int pass = 0; pass < 2; pass++)
		{
			const T wth = (pass == 0) ? T(XWaveParam.wavecdd) : T(1.0);
			const T wsg = (pass == 0) ? T(XWaveParam.wavecss) : T(1.0);

			for (int m = 0; m < ql; m++)
			{
				int d = (qs + m) % nd;
				int ispec = ifreq * nd + d;
				size_t is = specloc(stride, ispec, i);
				T c = XWave.spec.costh[d], s = XWave.spec.sinth[d];

				//===========================
				// Geographic space (first order upwind, SWAN BSBT)
				T cxi = cg * c + u;
				T cyi = cg * s + v;
				T cf[4];
				cf[0] = wetn[0] ? T(0.5) * (cxi + cgn[0] * c + un[0]) : cxi; // left face (positive towards +x)
				cf[1] = wetn[1] ? T(0.5) * (cxi + cgn[1] * c + un[1]) : cxi; // right face
				cf[2] = wetn[2] ? T(0.5) * (cyi + cgn[2] * s + vn[2]) : cyi; // bottom face (positive towards +y)
				cf[3] = wetn[3] ? T(0.5) * (cyi + cgn[3] * s + vn[3]) : cyi; // top face

				T Nn[4];
				for (int f = 0; f < 4; f++)
				{
					if (bnd[f])
					{
						// boundary condition: 0 absorbing, 1 Neumann (latest interior value), 2 JONSWAP
						Nn[f] = (btype[f] == 1) ? N[is] : ((btype[f] == 2) ? XWave.imp.Nbf[(size_t(slot[f]) * size_t(bw) + size_t(tx[f])) * size_t(nspec) + ispec] : T(0.0));
					}
					else
					{
						Nn[f] = wetn[f] ? N[specloc(stride, ispec, inb[f])] : T(0.0);
					}
				}

				T outgeo = (utils::max(cf[1], T(0.0)) - utils::min(cf[0], T(0.0)) + utils::max(cf[3], T(0.0)) - utils::min(cf[2], T(0.0))) * invdelta;
				T ingeo = (utils::max(cf[0], T(0.0)) * Nn[0] - utils::min(cf[1], T(0.0)) * Nn[1] + utils::max(cf[2], T(0.0)) * Nn[2] - utils::min(cf[3], T(0.0)) * Nn[3]) * invdelta;

				Dg[m] = invdt + outgeo;
				Rg[m] = invdt * No[is] + ingeo;
				Lg[m] = T(0.0);
				Ug[m] = T(0.0);

				//===========================
				// Spectral space: sigma (Gauss-Seidel: lower frequency already updated, higher frequency lagged)
				if (dofshift)
				{
					T cur = c * c * dudx + c * s * (dudy + dvdx) + s * s * dvdy;
					T csj = dsdd * depterm - cg * k * cur;
					// face value of a face between bins a (lower) and b (upper) with velocity cfs: Nf = am*Na + ap*Nb
					// lower face of bin j
					if (ifreq > 0)
					{
						T cfl = T(0.5) * (csj + dsddjm * depterm - cgjm * kjm * cur);
						T am = (cfl > T(0.0)) ? wsg + T(0.5) * (T(1.0) - wsg) : T(0.5) * (T(1.0) - wsg);
						T ap = T(1.0) - am;
						Dg[m] += T(-1.0) * cfl * ap / dsig;
						Rg[m] += cfl * am * N[specloc(stride, ispec - nd, i)] / dsig;
					}
					else if (csj < T(0.0))
					{
						Dg[m] += T(-1.0) * csj / dsig; // outgoing flux through the lowest frequency
					}
					// upper face of bin j
					if (ifreq < nf - 1)
					{
						T cfu = T(0.5) * (csj + dsddjp * depterm - cgjp * kjp * cur);
						T am = (cfu > T(0.0)) ? wsg + T(0.5) * (T(1.0) - wsg) : T(0.5) * (T(1.0) - wsg);
						T ap = T(1.0) - am;
						Dg[m] += cfu * am / dsig;
						Rg[m] += T(-1.0) * cfu * ap * N[specloc(stride, ispec + nd, i)] / dsig;
					}
					else if (csj > T(0.0))
					{
						Dg[m] += csj / dsig; // outgoing flux through the highest frequency
					}
				}
			}

			//===========================
			// Spectral space: theta (implicit, faces between the directions m and m+1, m = -1 ... ql-1)
			for (int m = -1; m < ql; m++)
			{
				T cfa = T(0.5) * (cth[m + 1] + cth[m + 2]);
				T am = (cfa > T(0.0)) ? wth + T(0.5) * (T(1.0) - wth) : T(0.5) * (T(1.0) - wth);
				T ap = T(1.0) - am;
				// flux F = cfa*(am*N[m] + ap*N[m+1]) leaves m and enters m+1
				if (m >= 0)
				{
					Dg[m] += cfa * am / dth;
					Ug[m] += T(-1.0) * cfa * ap / dth;
				}
				else
				{
					// direction m = -1 is outside the quadrant: lagged value
					Rg[0] += cfa * am * N[specloc(stride, ifreq * nd + (qs - 1 + nd) % nd, i)] / dth;
				}
				if (m + 1 < ql)
				{
					Dg[m + 1] += T(-1.0) * cfa * ap / dth;
					Lg[m + 1] += cfa * am / dth;
				}
				else
				{
					// direction m + 1 = ql is outside the quadrant: lagged value
					Rg[ql - 1] += T(-1.0) * cfa * ap * N[specloc(stride, ifreq * nd + (qs + ql) % nd, i)] / dth;
				}
			}

			// Check the diagonal dominance (M-matrix) of the blended system, otherwise use the upwind system
			bool ok = true;
			for (int m = 0; m < ql; m++)
			{
				T offd = ((Lg[m] < T(0.0)) ? T(-1.0) * Lg[m] : Lg[m]) + ((Ug[m] < T(0.0)) ? T(-1.0) * Ug[m] : Ug[m]);
				if (Dg[m] <= T(1.0e-20) || Dg[m] < offd || Rg[m] < T(0.0))
				{
					ok = false;
				}
			}
			if (!ok && pass == 0)
			{
				continue;
			}
			if (!ok)
			{
				break; // no outgoing flux at all (should not happen in a wet cell): keep the values
			}

			// Thomas algorithm: Dg[m] x[m] - Lg[m] x[m-1] - Ug[m] x[m+1] = Rg[m]
			for (int m = 1; m < ql; m++)
			{
				T w = Lg[m] / Dg[m - 1];
				Dg[m] = Dg[m] - w * Ug[m - 1];
				Rg[m] = Rg[m] + w * Rg[m - 1];
			}
			T x = Rg[ql - 1] / Dg[ql - 1];
			N[specloc(stride, ifreq * nd + (qs + ql - 1) % nd, i)] = utils::max(x, T(0.0));
			for (int m = ql - 2; m >= 0; m--)
			{
				x = (Rg[m] + Ug[m] * x) / Dg[m];
				N[specloc(stride, ifreq * nd + (qs + m) % nd, i)] = utils::max(x, T(0.0));
			}
			break;
		}
	}
}
template __host__ __device__ void WaveImplicitCell<float>(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int sweep, float invdt, const BlockP<float>& XBlock, const WaveModel<float>& XWave);
template __host__ __device__ void WaveImplicitCell<double>(const Param& XParam, const WaveParam& XWaveParam, int ix, int iy, int ib, int sweep, double invdt, const BlockP<double>& XBlock, const WaveModel<double>& XWave);


/**
 * @brief Convergence check of one cell (SWAN criterion): |Hs - Hs_previous| <= max(drel*Hs, dhabs).
 * conv = 1 (converged), 0 (not converged), -1 (dry). Hsold is updated.
 */
template <class T> __host__ __device__ void WaveImplicitConvCell(const WaveParam& XWaveParam, int i, const WaveModel<T>& XWave)
{
	if (XWave.medium.h[i] <= T(XWaveParam.wavedepthmin))
	{
		XWave.imp.conv[i] = T(-1.0);
		XWave.imp.Hsold[i] = T(0.0);
		return;
	}
	T m0 = T(0.0);
	for (int ifreq = 0; ifreq < XWaveParam.nfreq; ifreq++)
	{
		T sd = XWave.spec.sigma[ifreq] * XWave.spec.dsigma[ifreq] * T(XWaveParam.dtheta);
		for (int idir = 0; idir < XWaveParam.ndir; idir++)
		{
			m0 += XWave.action.N[specloc(XWaveParam.specstride, ifreq * XWaveParam.ndir + idir, i)] * sd;
		}
	}
	T Hs = T(4.0) * sqrt(m0);
	T dH = Hs - XWave.imp.Hsold[i];
	dH = dH < T(0.0) ? T(-1.0) * dH : dH;
	XWave.imp.conv[i] = (dH <= utils::max(T(XWaveParam.wavedrel) * Hs, T(XWaveParam.wavedhabs))) ? T(1.0) : T(0.0);
	XWave.imp.Hsold[i] = Hs;
}
template __host__ __device__ void WaveImplicitConvCell<float>(const WaveParam& XWaveParam, int i, const WaveModel<float>& XWave);
template __host__ __device__ void WaveImplicitConvCell<double>(const WaveParam& XWaveParam, int i, const WaveModel<double>& XWave);


/**
 * @brief Percentage of converged wet cells (from the conv array on the CPU).
 * The halo, unused blocks and dry cells hold -1 and are ignored.
 */
template <class T> double WaveConvPercent(size_t n, T* conv)
{
	size_t nwet = 0, nconv = 0;
	for (size_t i = 0; i < n; i++)
	{
		if (conv[i] >= T(0.0))
		{
			nwet++;
			nconv += (conv[i] > T(0.5)) ? 1 : 0;
		}
	}
	return (nwet > 0) ? 100.0 * double(nconv) / double(nwet) : 100.0;
}


/**
 * @brief Boundary values (incoming action density on the boundary faces) at a given time (CPU).
 */
template <class T> void WaveImplicitBndCPU(Param XParam, WaveParam XWaveParam, double time, Forcing<float> XForcing, BlockP<T> XBlock, WaveModel<T> XWave)
{
	T bndfac = T(WaveBndTaper(XParam, XWaveParam, time));
	for (int iseg = 0; iseg < XForcing.wavebndseg.size(); iseg++)
	{
		const wavebndsegment& seg = XForcing.wavebndseg[iseg];
		if (seg.type != 2)
		{
			continue;
		}
		double Hs = 0.0, Tp = 10.0, Dir = 0.0, Dspr = 30.0, gamma = XWaveParam.gammajsp;
		bool interpmaps = seg.on && !seg.uniform;
		T m0n = T(-1.0), Dsumn = T(-1.0);
		if (!interpmaps)
		{
			WaveBndValues(seg, time, XWaveParam.gammajsp, Hs, Tp, Dir, Dspr, gamma);
			WaveJonswapNorm(XWaveParam, XWave.spec, T(Tp), T(Dir), T(Dspr), T(gamma), m0n, Dsumn);
		}
		for (int sl = XWave.imp.segslot[iseg]; sl < XWave.imp.segslot[iseg + 1]; sl++)
		{
			int ib = XWave.imp.slotblk[sl];
			int side = XWave.imp.slotside[sl];
			T delta = calcres(T(XParam.dx), XBlock.level[ib]);
			for (int tx = 0; tx < XParam.blkwidth; tx++)
			{
				T Hsf = T(Hs), Tpf = T(Tp), Dirf = T(Dir), Dsprf = T(Dspr);
				if (interpmaps)
				{
					int ix = (side < 2) ? ((side == 0) ? -1 : XParam.blkwidth) : tx;
					int iy = (side < 2) ? tx : ((side == 2) ? -1 : XParam.blkwidth);
					T xx = T(XParam.xo) + XBlock.xo[ib] + ix * delta;
					T yy = T(XParam.yo) + XBlock.yo[ib] + iy * delta;
					Hsf = interp2BUQ(xx, yy, seg.Hs);
					Tpf = interp2BUQ(xx, yy, seg.Tp);
					Dirf = interp2BUQ(xx, yy, seg.Dir);
					Dsprf = interp2BUQ(xx, yy, seg.Dspr);
				}
				WaveImplicitBndFace(XWaveParam, XParam.blkwidth, sl, tx, seg.type, Hsf, Tpf, Dirf, Dsprf, T(gamma), m0n, Dsumn, bndfac, XWave);
			}
		}
	}
}


/**
 * @brief Implicit solver (CPU): four-sweep Gauss-Seidel iterations until convergence.
 *
 * @param XParam Light copy of the flow model parameters
 * @param tadv Time since the previous wave update (used as the implicit time step in the nonstationary mode)
 * @param time Time of the solution (boundary conditions)
 * @param convpct Percentage of converged wet cells at the end
 * @return Number of iterations
 */
template <class T> int WaveImplicitCPU(Param XParam, WaveParam XWaveParam, double tadv, double time, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave, double& convpct)
{
	int bw = XParam.blkwidth;
	int nspec = XWaveParam.nspec;
	T invdt = (XWaveParam.wavestationary || tadv <= 0.0) ? T(0.0) : T(1.0 / tadv);

	memcpy(XWave.action_o.N, XWave.action.N, size_t(nspec) * size_t(XWaveParam.specstride) * sizeof(T));

	WaveImplicitBndCPU(XParam, XWaveParam, time, XForcing, XModel.blocks, XWave);

	auto convcheck = [&]() {
		for (int ibl = 0; ibl < XParam.nblk; ibl++)
		{
			int ib = XModel.blocks.active[ibl];
			for (int iy = 0; iy < bw; iy++)
			{
				for (int ix = 0; ix < bw; ix++)
				{
					WaveImplicitConvCell(XWaveParam, memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib), XWave);
				}
			}
		}
		return WaveConvPercent(size_t(XParam.nblkmem) * XParam.blksize, XWave.imp.conv);
	};
	convcheck(); // initialise Hsold

	int it;
	convpct = 0.0;
	for (it = 1; it <= XWaveParam.wavemaxiter; it++)
	{
		for (int s = 0; s < 4; s++)
		{
			if (XWaveParam.qlen[s] <= 0)
			{
				continue;
			}
			int sx = WaveSweepSx(s), sy = WaveSweepSy(s);
			for (int l = 0; l < XWave.imp.nlevels[s]; l++)
			{
				int l0 = XWave.imp.levelstart[s][l];
				int nbl = XWave.imp.levelstart[s][l + 1] - l0;

				// Halo of the blocks of the level (all the spectral bins, reusing the flow halo function on the virtual blocks)
				Param XParamL = XParam;
				XParamL.nblk = nbl * nspec;
				BlockP<T> XBlockL = XWave.vblocks;
				XBlockL.active = XWave.imp.vorder[s] + size_t(l0) * nspec;
				fillHaloC(XParamL, XBlockL, XWave.action.N);

				for (int k = 0; k < nbl; k++)
				{
					int ib = XWave.imp.order[s][l0 + k];
					for (int jy = 0; jy < bw; jy++)
					{
						int iy = (sy > 0) ? jy : bw - 1 - jy;
						for (int jx = 0; jx < bw; jx++)
						{
							int ix = (sx > 0) ? jx : bw - 1 - jx;
							WaveImplicitCell(XParam, XWaveParam, ix, iy, ib, s, invdt, XModel.blocks, XWave);
						}
					}
				}
			}
		}
		convpct = convcheck();
		if (convpct >= XWaveParam.wavenpnts)
		{
			break;
		}
	}
	return utils::min(it, XWaveParam.wavemaxiter);
}
template int WaveImplicitCPU<float>(Param XParam, WaveParam XWaveParam, double tadv, double time, Forcing<float> XForcing, Model<float> XModel, WaveModel<float> XWave, double& convpct);
template int WaveImplicitCPU<double>(Param XParam, WaveParam XWaveParam, double tadv, double time, Forcing<float> XForcing, Model<double> XModel, WaveModel<double> XWave, double& convpct);


//==============================================================================
// GPU
//==============================================================================

/**
 * @brief Implicit update of all the blocks of one level for one sweep (GPU).
 * One thread block per model block, one thread per cell; the cells are processed in upwind order by diagonals (wavefront).
 */
template <class T> __global__ void WaveImplicitLevelGPU(Param XParam, WaveParam XWaveParam, int sweep, T invdt, int* blks, BlockP<T> XBlock, WaveModel<T> XWave)
{
	int ib = blks[blockIdx.x];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = XParam.blkwidth;
	int ix = (WaveSweepSx(sweep) > 0) ? tx : bw - 1 - tx;
	int iy = (WaveSweepSy(sweep) > 0) ? ty : bw - 1 - ty;

	for (int d = 0; d < 2 * bw - 1; d++)
	{
		if (tx + ty == d)
		{
			WaveImplicitCell(XParam, XWaveParam, ix, iy, ib, sweep, invdt, XBlock, XWave);
		}
		__syncthreads();
	}
}
template __global__ void WaveImplicitLevelGPU<float>(Param XParam, WaveParam XWaveParam, int sweep, float invdt, int* blks, BlockP<float> XBlock, WaveModel<float> XWave);
template __global__ void WaveImplicitLevelGPU<double>(Param XParam, WaveParam XWaveParam, int sweep, double invdt, int* blks, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Convergence check (GPU). See WaveImplicitConvCell.
 */
template <class T> __global__ void WaveImplicitConvGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveModel<T> XWave)
{
	int ib = XBlock.active[blockIdx.x];
	int i = memloc(XParam.halowidth, XParam.blkmemwidth, threadIdx.x, threadIdx.y, ib);
	WaveImplicitConvCell(XWaveParam, i, XWave);
}
template __global__ void WaveImplicitConvGPU<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, WaveModel<float> XWave);
template __global__ void WaveImplicitConvGPU<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Boundary values on the faces of the slots of one wave boundary segment (GPU). gridDim = number of slots, blockDim = blkwidth.
 */
template <class T> __global__ void WaveImplicitBndGPU(Param XParam, WaveParam XWaveParam, int slot0, int type, bool interpmaps, T Hs, T Tp, T Dir, T Dspr, T gamma, T m0n, T Dsumn, TexSetP texHs, TexSetP texTp, TexSetP texDir, TexSetP texDspr, T bndfac, BlockP<T> XBlock, WaveModel<T> XWave)
{
	int sl = slot0 + blockIdx.x;
	int tx = threadIdx.x;
	if (interpmaps)
	{
		int ib = XWave.imp.slotblk[sl];
		int side = XWave.imp.slotside[sl];
		T delta = calcres(T(XParam.dx), XBlock.level[ib]);
		int ix = (side < 2) ? ((side == 0) ? -1 : XParam.blkwidth) : tx;
		int iy = (side < 2) ? tx : ((side == 2) ? -1 : XParam.blkwidth);
		T xx = T(XParam.xo) + XBlock.xo[ib] + ix * delta;
		T yy = T(XParam.yo) + XBlock.yo[ib] + iy * delta;
		Hs = interpDyn2BUQ(xx, yy, texHs);
		Tp = interpDyn2BUQ(xx, yy, texTp);
		Dir = interpDyn2BUQ(xx, yy, texDir);
		Dspr = interpDyn2BUQ(xx, yy, texDspr);
		m0n = T(-1.0);
	}
	WaveImplicitBndFace(XWaveParam, XParam.blkwidth, sl, tx, type, Hs, Tp, Dir, Dspr, gamma, m0n, Dsumn, bndfac, XWave);
}
template __global__ void WaveImplicitBndGPU<float>(Param XParam, WaveParam XWaveParam, int slot0, int type, bool interpmaps, float Hs, float Tp, float Dir, float Dspr, float gamma, float m0n, float Dsumn, TexSetP texHs, TexSetP texTp, TexSetP texDir, TexSetP texDspr, float bndfac, BlockP<float> XBlock, WaveModel<float> XWave);
template __global__ void WaveImplicitBndGPU<double>(Param XParam, WaveParam XWaveParam, int slot0, int type, bool interpmaps, double Hs, double Tp, double Dir, double Dspr, double gamma, double m0n, double Dsumn, TexSetP texHs, TexSetP texTp, TexSetP texDir, TexSetP texDspr, double bndfac, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Implicit solver (GPU). Same algorithm as WaveImplicitCPU.
 */
template <class T> int WaveImplicitGPU(Param XParam, WaveParam XWaveParam, double tadv, double time, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave, double& convpct)
{
	int bw = XParam.blkwidth;
	int nspec = XWaveParam.nspec;
	T invdt = (XWaveParam.wavestationary || tadv <= 0.0) ? T(0.0) : T(1.0 / tadv);
	dim3 blockDim(bw, bw, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	CUDA_CHECK(cudaMemcpy(XWave.action_o.N, XWave.action.N, size_t(nspec) * size_t(XWaveParam.specstride) * sizeof(T), cudaMemcpyDeviceToDevice));

	// Boundary values
	T bndfac = T(WaveBndTaper(XParam, XWaveParam, time));
	dim3 blockDimB(bw, 1, 1);
	for (int iseg = 0; iseg < XForcing.wavebndseg.size(); iseg++)
	{
		const wavebndsegment& seg = XForcing.wavebndseg[iseg];
		int nsl = XWave.imp.segslot[iseg + 1] - XWave.imp.segslot[iseg];
		if (seg.type != 2 || nsl <= 0)
		{
			continue;
		}
		double Hs = 0.0, Tp = 10.0, Dir = 0.0, Dspr = 30.0, gamma = XWaveParam.gammajsp;
		bool interpmaps = seg.on && !seg.uniform;
		T m0n = T(-1.0), Dsumn = T(-1.0);
		if (!interpmaps)
		{
			WaveBndValues(seg, time, XWaveParam.gammajsp, Hs, Tp, Dir, Dspr, gamma);
			WaveJonswapNorm(XWaveParam, XWave.spec_h, T(Tp), T(Dir), T(Dspr), T(gamma), m0n, Dsumn);
		}
		dim3 gridDimB(nsl, 1, 1);
		WaveImplicitBndGPU <<< gridDimB, blockDimB, 0 >>> (XParam, XWaveParam, XWave.imp.segslot[iseg], seg.type, interpmaps, T(Hs), T(Tp), T(Dir), T(Dspr), T(gamma), m0n, Dsumn, seg.Hs.GPU, seg.Tp.GPU, seg.Dir.GPU, seg.Dspr.GPU, bndfac, XModel.blocks, XWave);
	}
	CUDA_CHECK(cudaDeviceSynchronize());

	// Convergence arrays (host copy for the count)
	T* conv_h;
	AllocateCPU(XParam.nblkmem, XParam.blksize, conv_h);
	auto convcheck = [&]() {
		WaveImplicitConvGPU <<< gridDim, blockDim, 0 >>> (XParam, XWaveParam, XModel.blocks, XWave);
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(conv_h, XWave.imp.conv, size_t(XParam.nblkmem) * XParam.blksize * sizeof(T), cudaMemcpyDeviceToHost));
		return WaveConvPercent(size_t(XParam.nblkmem) * XParam.blksize, conv_h);
	};
	convcheck();

	int it;
	convpct = 0.0;
	for (it = 1; it <= XWaveParam.wavemaxiter; it++)
	{
		for (int s = 0; s < 4; s++)
		{
			if (XWaveParam.qlen[s] <= 0)
			{
				continue;
			}
			for (int l = 0; l < XWave.imp.nlevels[s]; l++)
			{
				int l0 = XWave.imp.levelstart[s][l];
				int nbl = XWave.imp.levelstart[s][l + 1] - l0;

				Param XParamL = XParam;
				XParamL.nblk = nbl * nspec;
				BlockP<T> XBlockL = XWave.vblocks;
				XBlockL.active = XWave.imp.vorder[s] + size_t(l0) * nspec;
				fillHaloGPU(XParamL, XBlockL, XWave.action.N);

				dim3 gridDimL(nbl, 1, 1);
				WaveImplicitLevelGPU <<< gridDimL, blockDim, 0 >>> (XParam, XWaveParam, s, invdt, XWave.imp.order[s] + l0, XModel.blocks, XWave);
			}
		}
		CUDA_CHECK(cudaDeviceSynchronize());
		convpct = convcheck();
		if (convpct >= XWaveParam.wavenpnts)
		{
			break;
		}
	}
	free(conv_h);
	return utils::min(it, XWaveParam.wavemaxiter);
}
template int WaveImplicitGPU<float>(Param XParam, WaveParam XWaveParam, double tadv, double time, Forcing<float> XForcing, Model<float> XModel, WaveModel<float> XWave, double& convpct);
template int WaveImplicitGPU<double>(Param XParam, WaveParam XWaveParam, double tadv, double time, Forcing<float> XForcing, Model<double> XModel, WaveModel<double> XWave, double& convpct);

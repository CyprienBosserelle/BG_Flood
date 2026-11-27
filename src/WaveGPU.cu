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

#include "WaveGPU.h"


/**
 * @brief Fill the halo of every spectral bin of the action density (GPU).
 *
 * Each spectral bin has the same layout as a flow variable so the flow halo function (fillHaloGPU) is reused.
 */
template <class T> void fillHaloWaveGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveSpectrumP<T> XAction)
{
	for (int ispec = 0; ispec < XWaveParam.nspec; ispec++)
	{
		fillHaloGPU(XParam, XBlock, XAction.N + specloc(XWaveParam.specstride, ispec, 0));
	}
}
template void fillHaloWaveGPU<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, WaveSpectrumP<float> XAction);
template void fillHaloWaveGPU<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, WaveSpectrumP<double> XAction);


/**
 * @brief Reset the action density tendency (dN) to zero (GPU). One thread per cell, looping over the spectral bins.
 */
template <class T> __global__ void ResetWaveAdvGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveAdvanceP<T> XAdv)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	for (int ispec = 0; ispec < XWaveParam.nspec; ispec++)
	{
		XAdv.dN[specloc(XWaveParam.specstride, ispec, i)] = T(0.0);
	}
}
template __global__ void ResetWaveAdvGPU<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, WaveAdvanceP<float> XAdv);
template __global__ void ResetWaveAdvGPU<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, WaveAdvanceP<double> XAdv);


/**
 * @brief Explicit update N = alpha*N_o + (1-alpha)*(N + dt*dN) (GPU). See WaveUpdateCell.
 */
template <class T> __global__ void UpdateWaveActionGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, T dt, T alpha, WaveModel<T> XWave)
{
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

	WaveUpdateCell(XWaveParam, i, dt, alpha, XWave);
}
template __global__ void UpdateWaveActionGPU<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, float dt, float alpha, WaveModel<float> XWave);
template __global__ void UpdateWaveActionGPU<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, double dt, double alpha, WaveModel<double> XWave);


/**
 * @brief Update the medium seen by the wave model (interior cells) (GPU). See WaveMediumCell.
 */
template <class T> __global__ void WaveMediumGPU(Param XParam, WaveParam XWaveParam, T invtadv, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb, WaveModel<T> XWave)
{
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

	WaveMediumCell(XParam, XWaveParam, i, invtadv, XEv, zb, XWave);
}
template __global__ void WaveMediumGPU<float>(Param XParam, WaveParam XWaveParam, float invtadv, BlockP<float> XBlock, EvolvingP<float> XEv, float* zb, WaveModel<float> XWave);
template __global__ void WaveMediumGPU<double>(Param XParam, WaveParam XWaveParam, double invtadv, BlockP<double> XBlock, EvolvingP<double> XEv, double* zb, WaveModel<double> XWave);


/**
 * @brief Wave number and group velocity in all the cells of the blocks including the halo (GPU).
 * Launched with blockDim = (blkmemwidth, blkmemwidth). See WaveKinCell.
 */
template <class T> __global__ void WaveKinGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, WaveModel<T> XWave)
{
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = ix + iy * XParam.blkmemwidth + ib * XParam.blksize;

	WaveKinCell(XParam, XWaveParam, i, XWave);
}
template __global__ void WaveKinGPU<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, WaveModel<float> XWave);
template __global__ void WaveKinGPU<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Action density fluxes through the left faces (GPU). Launched with blockDim = (blkwidth + halowidth, blkwidth).
 */
template <class T> __global__ void WaveFluxXGPU(Param XParam, WaveParam XWaveParam, int ispec, T bndfac, BlockP<T> XBlock, WaveModel<T> XWave)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	WaveFluxXCell(XParam, XWaveParam, ix, iy, ib, ispec, bndfac, XBlock, XWave);
}
template __global__ void WaveFluxXGPU<float>(Param XParam, WaveParam XWaveParam, int ispec, float bndfac, BlockP<float> XBlock, WaveModel<float> XWave);
template __global__ void WaveFluxXGPU<double>(Param XParam, WaveParam XWaveParam, int ispec, double bndfac, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Action density fluxes through the bottom faces (GPU). Launched with blockDim = (blkwidth, blkwidth + halowidth).
 */
template <class T> __global__ void WaveFluxYGPU(Param XParam, WaveParam XWaveParam, int ispec, T bndfac, BlockP<T> XBlock, WaveModel<T> XWave)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	WaveFluxYCell(XParam, XWaveParam, ix, iy, ib, ispec, bndfac, XBlock, XWave);
}
template __global__ void WaveFluxYGPU<float>(Param XParam, WaveParam XWaveParam, int ispec, float bndfac, BlockP<float> XBlock, WaveModel<float> XWave);
template __global__ void WaveFluxYGPU<double>(Param XParam, WaveParam XWaveParam, int ispec, double bndfac, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Divergence of the geographic fluxes of one spectral bin added to dN (GPU).
 */
template <class T> __global__ void WaveGeoDivGPU(Param XParam, WaveParam XWaveParam, int ispec, BlockP<T> XBlock, WaveModel<T> XWave)
{
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ibl = blockIdx.x;
	int ib = XBlock.active[ibl];

	WaveGeoDivCell(XParam, XWaveParam, ix, iy, ib, ispec, XBlock, XWave);
}
template __global__ void WaveGeoDivGPU<float>(Param XParam, WaveParam XWaveParam, int ispec, BlockP<float> XBlock, WaveModel<float> XWave);
template __global__ void WaveGeoDivGPU<double>(Param XParam, WaveParam XWaveParam, int ispec, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Spectral propagation (refraction and frequency shift) (GPU). One thread per cell. See WaveSpecPropCell.
 */
template <class T> __global__ void WaveSpecPropGPU(Param XParam, WaveParam XWaveParam, T dt, BlockP<T> XBlock, WaveModel<T> XWave)
{
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(XParam.halowidth, XParam.blkmemwidth, ix, iy, ib);

	WaveSpecPropCell(XParam, XWaveParam, i, dt, XWave);
}
template __global__ void WaveSpecPropGPU<float>(Param XParam, WaveParam XWaveParam, float dt, BlockP<float> XBlock, WaveModel<float> XWave);
template __global__ void WaveSpecPropGPU<double>(Param XParam, WaveParam XWaveParam, double dt, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Calculate the bulk wave parameters and radiation stresses (GPU). Uses the same cell function as the CPU version.
 */
template <class T> __global__ void CalcWaveBulkGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb, WaveModel<T> XWave)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	T h = WaveDepth(XParam, XWaveParam, i, XEv, zb);
	CalcWaveBulkCell(XParam, XWaveParam, i, h, XWave);
}
template __global__ void CalcWaveBulkGPU<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, EvolvingP<float> XEv, float* zb, WaveModel<float> XWave);
template __global__ void CalcWaveBulkGPU<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, EvolvingP<double> XEv, double* zb, WaveModel<double> XWave);


/**
 * @brief Calculate the maximum stable wave time step in each cell (GPU). Uses the same cell function as the CPU version.
 */
template <class T> __global__ void CalcWavedtGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb, WaveModel<T> XWave)
{
	unsigned int halowidth = XParam.halowidth;
	unsigned int blkmemwidth = blockDim.x + halowidth * 2;
	unsigned int ix = threadIdx.x;
	unsigned int iy = threadIdx.y;
	unsigned int ibl = blockIdx.x;
	unsigned int ib = XBlock.active[ibl];

	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);

	T h = WaveDepth(XParam, XWaveParam, i, XEv, zb);
	XWave.time.dtmax[i] = CalcWavedtCell(XParam, XWaveParam, i, XBlock.level[ib], h, XEv, XWave);
}
template __global__ void CalcWavedtGPU<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, EvolvingP<float> XEv, float* zb, WaveModel<float> XWave);
template __global__ void CalcWavedtGPU<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, EvolvingP<double> XEv, double* zb, WaveModel<double> XWave);


/**
 * @brief Calculate the maximum stable wave time step over the whole domain on the GPU.
 *
 * The per cell time step is stored in XWave.time.dtmax and reduced using the flow GPU reduction (CalctimestepGPU).
 *
 * @return Maximum wave time step (s)
 */
template <class T> double CalcWavedtmaxGPU(Param XParam, WaveParam XWaveParam, Model<T> XModel, WaveModel<T> XWave)
{
	if (XWaveParam.wavedt > 0.0)
	{
		return XWaveParam.wavedt;
	}

	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	CalcWavedtGPU <<< gridDim, blockDim, 0 >>> (XParam, XWaveParam, XModel.blocks, XModel.evolv, XModel.zb, XWave);
	CUDA_CHECK(cudaDeviceSynchronize());

	Loop<T> XLoopdummy = WaveDummyLoop<T>();
	return double(CalctimestepGPU(XParam, XLoopdummy, XModel.blocks, XWave.time));
}
template double CalcWavedtmaxGPU<float>(Param XParam, WaveParam XWaveParam, Model<float> XModel, WaveModel<float> XWave);
template double CalcWavedtmaxGPU<double>(Param XParam, WaveParam XWaveParam, Model<double> XModel, WaveModel<double> XWave);


/**
 * @brief Prepare the medium (depth, currents and gradients) and the kinematics (k, cg) for the wave model (GPU).
 * See WavePrepCPU.
 */
template <class T> void WavePrepGPU(Param XParam, WaveParam XWaveParam, double tadv, Model<T> XModel, WaveModel<T> XWave)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 blockDimFull(XParam.blkmemwidth, XParam.blkmemwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	T invtadv = (tadv > 0.0) ? T(1.0 / tadv) : T(0.0);

	WaveMediumGPU <<< gridDim, blockDim, 0 >>> (XParam, XWaveParam, invtadv, XModel.blocks, XModel.evolv, XModel.zb, XWave);
	CUDA_CHECK(cudaDeviceSynchronize());

	fillHaloGPU(XParam, XModel.blocks, XWave.medium.h);
	fillHaloGPU(XParam, XModel.blocks, XWave.medium.u);
	fillHaloGPU(XParam, XModel.blocks, XWave.medium.v);
	CUDA_CHECK(cudaDeviceSynchronize());

	gradient <<< gridDim, blockDim, 0 >>> (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.delta, XWave.medium.h, XWave.medium.dhdx, XWave.medium.dhdy);
	gradient <<< gridDim, blockDim, 0 >>> (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.delta, XWave.medium.u, XWave.medium.dudx, XWave.medium.dudy);
	gradient <<< gridDim, blockDim, 0 >>> (XParam.halowidth, XModel.blocks.active, XModel.blocks.level, (T)XParam.theta, (T)XParam.delta, XWave.medium.v, XWave.medium.dvdx, XWave.medium.dvdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	WaveKinGPU <<< gridDim, blockDimFull, 0 >>> (XParam, XWaveParam, XModel.blocks, XWave);
	CUDA_CHECK(cudaDeviceSynchronize());
}
template void WavePrepGPU<float>(Param XParam, WaveParam XWaveParam, double tadv, Model<float> XModel, WaveModel<float> XWave);
template void WavePrepGPU<double>(Param XParam, WaveParam XWaveParam, double tadv, Model<double> XModel, WaveModel<double> XWave);


/**
 * @brief Geographic propagation (GPU), one spectral bin at a time. See WaveGeoPropCPU for the steps.
 *
 * @param XParamN Light copy of XParam with theta = wavetheta (used for the limited gradients of N)
 */
template <class T> void WaveGeoPropGPU(Param XParam, Param XParamN, WaveParam XWaveParam, T bndfac, BlockP<T> XBlock, WaveModel<T> XWave)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 blockDimKX(XParam.blkwidth + XParam.halowidth, XParam.blkwidth, 1);
	dim3 blockDimKY(XParam.blkwidth, XParam.blkwidth + XParam.halowidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	for (int ispec = 0; ispec < XWaveParam.nspec; ispec++)
	{
		T* Nbin = XWave.action.N + specloc(XWaveParam.specstride, ispec, 0);

		fillHaloGPU(XParam, XBlock, Nbin);
		CUDA_CHECK(cudaDeviceSynchronize());

		if (XWaveParam.wavesolver == 2)
		{
			gradient <<< gridDim, blockDim, 0 >>> (XParam.halowidth, XBlock.active, XBlock.level, (T)XWaveParam.wavetheta, (T)XParam.delta, Nbin, XWave.work.dNdx, XWave.work.dNdy);
			CUDA_CHECK(cudaDeviceSynchronize());
			gradientHaloGPU(XParamN, XBlock, Nbin, XWave.work.dNdx, XWave.work.dNdy);
		}

		WaveFluxXGPU <<< gridDim, blockDimKX, 0 >>> (XParam, XWaveParam, ispec, bndfac, XBlock, XWave);
		WaveFluxYGPU <<< gridDim, blockDimKY, 0 >>> (XParam, XWaveParam, ispec, bndfac, XBlock, XWave);
		CUDA_CHECK(cudaDeviceSynchronize());

		fillHaloLeftRightGPUnew(XParam, XBlock, 0, XWave.work.Fx);
		fillHaloBotTopGPUnew(XParam, XBlock, 0, XWave.work.Fy);
		CUDA_CHECK(cudaDeviceSynchronize());

		WaveGeoDivGPU <<< gridDim, blockDim, 0 >>> (XParam, XWaveParam, ispec, XBlock, XWave);
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}
template void WaveGeoPropGPU<float>(Param XParam, Param XParamN, WaveParam XWaveParam, float bndfac, BlockP<float> XBlock, WaveModel<float> XWave);
template void WaveGeoPropGPU<double>(Param XParam, Param XParamN, WaveParam XWaveParam, double bndfac, BlockP<double> XBlock, WaveModel<double> XWave);


/**
 * @brief Right hand side of the action balance equation dN = L(N) (GPU): propagation (+ sources, to be implemented).
 */
template <class T> void WaveRHSGPU(Param XParam, Param XParamN, WaveParam XWaveParam, T dt, T bndfac, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	ResetWaveAdvGPU <<< gridDim, blockDim, 0 >>> (XParam, XWaveParam, XModel.blocks, XWave.adv);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Geographic propagation
	WaveGeoPropGPU(XParam, XParamN, XWaveParam, bndfac, XModel.blocks, XWave);

	// Spectral propagation
	WaveSpecPropGPU <<< gridDim, blockDim, 0 >>> (XParam, XWaveParam, dt, XModel.blocks, XWave);
	CUDA_CHECK(cudaDeviceSynchronize());

	// Source terms
	// TODO
}
template void WaveRHSGPU<float>(Param XParam, Param XParamN, WaveParam XWaveParam, float dt, float bndfac, Forcing<float> XForcing, Model<float> XModel, WaveModel<float> XWave);
template void WaveRHSGPU<double>(Param XParam, Param XParamN, WaveParam XWaveParam, double dt, double bndfac, Forcing<float> XForcing, Model<double> XModel, WaveModel<double> XWave);


/**
 * @brief Main GPU wave solver: advance the action density by one explicit wave time step dt.
 *
 * This is the wave equivalent of FlowGPU. See WaveCPU for the description of the time integration.
 *
 * @tparam T Data type (float or double)
 * @param XParam Light copy of the flow model parameters (see WaveLightParam)
 * @param XWaveParam Wave model parameters
 * @param dt Wave time step (s)
 * @param bndfac Boundary taper factor (0 to 1)
 * @param XForcing Forcing (wind is used by the wave model)
 * @param XModel Flow model GPU structure (mesh, bathy, water depth and currents)
 * @param XWave Wave model GPU structure
 */
template <class T> void WaveGPU(Param XParam, WaveParam XWaveParam, T dt, T bndfac, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	Param XParamN = WaveLightParam(XParam, XWaveParam.wavetheta);

	//============================================
	// Keep a copy of the action density at the start of the step
	CUDA_CHECK(cudaMemcpy(XWave.action_o.N, XWave.action.N, size_t(XWaveParam.nspec) * size_t(XWaveParam.specstride) * sizeof(T), cudaMemcpyDeviceToDevice));

	//============================================
	// First stage (forward Euler)
	WaveRHSGPU(XParam, XParamN, XWaveParam, dt, bndfac, XForcing, XModel, XWave);
	UpdateWaveActionGPU <<< gridDim, blockDim, 0 >>> (XParam, XWaveParam, XModel.blocks, dt, T(0.0), XWave);
	CUDA_CHECK(cudaDeviceSynchronize());

	//============================================
	// Second stage (Heun / SSP-RK2)
	if (XWaveParam.wavesolver == 2)
	{
		WaveRHSGPU(XParam, XParamN, XWaveParam, dt, bndfac, XForcing, XModel, XWave);
		UpdateWaveActionGPU <<< gridDim, blockDim, 0 >>> (XParam, XWaveParam, XModel.blocks, dt, T(0.5), XWave);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	//============================================
	// Bulk parameters and radiation stresses
	CalcWaveBulkGPU <<< gridDim, blockDim, 0 >>> (XParam, XWaveParam, XModel.blocks, XModel.evolv, XModel.zb, XWave);
	CUDA_CHECK(cudaDeviceSynchronize());

	//============================================
	// Wave forces on the flow
	// TODO
}
template void WaveGPU<float>(Param XParam, WaveParam XWaveParam, float dt, float bndfac, Forcing<float> XForcing, Model<float> XModel, WaveModel<float> XWave);
template void WaveGPU<double>(Param XParam, WaveParam XWaveParam, double dt, double bndfac, Forcing<float> XForcing, Model<double> XModel, WaveModel<double> XWave);

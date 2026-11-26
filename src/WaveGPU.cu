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
 * @brief Explicit update of the action density N = max(N + dt*dN, 0) (GPU).
 */
template <class T> __global__ void UpdateWaveActionGPU(Param XParam, WaveParam XWaveParam, BlockP<T> XBlock, T dt, WaveAdvanceP<T> XAdv, WaveSpectrumP<T> XAction)
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
		size_t is = specloc(XWaveParam.specstride, ispec, i);
		XAction.N[is] = max(XAction.N[is] + dt * XAdv.dN[is], T(0.0));
	}
}
template __global__ void UpdateWaveActionGPU<float>(Param XParam, WaveParam XWaveParam, BlockP<float> XBlock, float dt, WaveAdvanceP<float> XAdv, WaveSpectrumP<float> XAction);
template __global__ void UpdateWaveActionGPU<double>(Param XParam, WaveParam XWaveParam, BlockP<double> XBlock, double dt, WaveAdvanceP<double> XAdv, WaveSpectrumP<double> XAction);


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
 * @brief Main GPU wave solver: advance the action density by one wave time step dt.
 *
 * This is the wave equivalent of FlowGPU. See WaveCPU for the description of the steps.
 *
 * @tparam T Data type (float or double)
 * @param XParam Flow model parameters (mesh information)
 * @param XWaveParam Wave model parameters
 * @param dt Wave time step (s)
 * @param XForcing Forcing (wind is used by the wave model)
 * @param XModel Flow model GPU structure (mesh, bathy, water depth and currents)
 * @param XWave Wave model GPU structure
 */
template <class T> void WaveGPU(Param XParam, WaveParam XWaveParam, T dt, Forcing<float> XForcing, Model<T> XModel, WaveModel<T> XWave)
{
	dim3 blockDim(XParam.blkwidth, XParam.blkwidth, 1);
	dim3 gridDim(XParam.nblk, 1, 1);

	//============================================
	// 1. Keep a copy of the action density at the previous step
	CUDA_CHECK(cudaMemcpy(XWave.action_o.N, XWave.action.N, size_t(XWaveParam.nspec) * size_t(XWaveParam.specstride) * sizeof(T), cudaMemcpyDeviceToDevice));

	//============================================
	// 2. Fill the halo for each spectral bin
	fillHaloWaveGPU(XParam, XWaveParam, XModel.blocks, XWave.action);

	//============================================
	// 3. Wave boundary conditions
	// TODO

	//============================================
	// Reset tendency
	ResetWaveAdvGPU <<< gridDim, blockDim, 0 >>> (XParam, XWaveParam, XModel.blocks, XWave.adv);
	CUDA_CHECK(cudaDeviceSynchronize());

	//============================================
	// 4. Geographic propagation
	// TODO

	//============================================
	// 5. Spectral propagation
	// TODO

	//============================================
	// 6. Source terms
	// TODO

	//============================================
	// 7. Update action density
	UpdateWaveActionGPU <<< gridDim, blockDim, 0 >>> (XParam, XWaveParam, XModel.blocks, dt, XWave.adv, XWave.action);
	CUDA_CHECK(cudaDeviceSynchronize());

	//============================================
	// 8. Bulk parameters and radiation stresses
	CalcWaveBulkGPU <<< gridDim, blockDim, 0 >>> (XParam, XWaveParam, XModel.blocks, XModel.evolv, XModel.zb, XWave);
	CUDA_CHECK(cudaDeviceSynchronize());

	//============================================
	// 9. Wave forces on the flow
	// TODO
}
template void WaveGPU<float>(Param XParam, WaveParam XWaveParam, float dt, Forcing<float> XForcing, Model<float> XModel, WaveModel<float> XWave);
template void WaveGPU<double>(Param XParam, WaveParam XWaveParam, double dt, Forcing<float> XForcing, Model<double> XModel, WaveModel<double> XWave);

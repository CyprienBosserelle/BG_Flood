


#include "Testing.h"




/*! \fn int main(int argc, char **argv)
* Main function 
* This function is the entry point to the software
*/
template <class T>
void TestingOutput(Param XParam, Model<T> XModel)
{
	std::string outvar;
	//log(std::to_string(XForcing.Bathy.val[50]));
	creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo);
	outvar = "h";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	outvar = "u";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	outvar = "v";
	copyID2var(XParam, XModel.blocks, XModel.OutputVarMap[outvar]);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	outvar = "zb";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	outvar = "zs";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	//outvar = "cf";
	//defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.cf);

	
}

template void TestingOutput<float>(Param XParam, Model<float> XModel);
template void TestingOutput<double>(Param XParam, Model<double> XModel);


template <class T> void copyID2var(Param XParam, BlockP<T> XBlock, T* z)
{
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int n = memloc(XParam, ix, iy, ib);
				z[n] = ib;
			}
		}
	}

}

template void copyID2var<float>(Param XParam, BlockP<float> XBlock, float* z);
template void copyID2var<double>(Param XParam, BlockP<double> XBlock, double* z);


template <class T> void CompareCPUvsGPU(Param XParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g)
{
	Loop<T> XLoop;
	// GPU stuff
	if (XParam.GPUDEVICE >= 0)
	{
		XLoop.blockDim = (16, 16, 1);
		XLoop.gridDim = (XParam.nblk, 1, 1);
	}

	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();


	T* gpureceive;
	T* diff;

	AllocateCPU(XParam.nblkmem, XParam.blksize, gpureceive);
	AllocateCPU(XParam.nblkmem, XParam.blksize, diff);


	//============================================
	// Compare gradients for evolving parameters
	cudaSetDevice(0);
	//GPU
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	//gradientGPU(XParam, XLoop, XModel_g.blocks, XModel_g.evolv, XModel_g.grad);

	gradientGPU(XParam, XLoop, XModel_g.blocks, XModel_g.evolv, XModel_g.grad);

	//============================================
	// Synchronise all ongoing streams
	CUDA_CHECK(cudaDeviceSynchronize());

	updateKurgXGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.grad, XModel_g.flux, XModel_g.time.dtmax);
	//updateKurgY << < XLoop.gridDim, XLoop.blockDim, 0, XLoop.streams[0] >> > (XParam, XLoop.epsilon, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax);
	updateKurgYGPU << < gridDim, blockDim, 0 >> > (XParam, XModel_g.blocks, XModel_g.evolv, XModel_g.grad, XModel_g.flux, XModel_g.time.dtmax);
	CUDA_CHECK(cudaDeviceSynchronize());

	fillHalo(XParam, XModel.blocks, XModel.evolv);

	//CPU
	gradientCPU(XParam, XLoop, XModel.blocks, XModel.evolv, XModel.grad);
	updateKurgXCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax);
	updateKurgYCPU(XParam, XModel.blocks, XModel.evolv, XModel.grad, XModel.flux, XModel.time.dtmax);
	
	// calculate difference
	//diffArray(XParam, XLoop, XModel.blocks, XModel.evolv.h, XModel_g.evolv.h, XModel.evolv_o.u);

	creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo);

	//Check evolving param
	diffArray(XParam, XLoop, XModel.blocks, "h", XModel.evolv.h, XModel_g.evolv.h, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "zs", XModel.evolv.zs, XModel_g.evolv.zs, gpureceive, diff);
	//check gradients
	diffArray(XParam, XLoop, XModel.blocks, "dhdx", XModel.grad.dhdx, XModel_g.grad.dhdx, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "dhdy", XModel.grad.dhdy, XModel_g.grad.dhdy, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "dzsdx", XModel.grad.dzsdx, XModel_g.grad.dzsdx, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "dzsdy", XModel.grad.dzsdy, XModel_g.grad.dzsdy, gpureceive, diff);

	//Check Kurganov
	diffArray(XParam, XLoop, XModel.blocks,"Fhu", XModel.flux.Fhu, XModel_g.flux.Fhu, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "Fqux", XModel.flux.Fqux, XModel_g.flux.Fqux, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "Su", XModel.flux.Su, XModel_g.flux.Su, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "Fqvx", XModel.flux.Fqvx, XModel_g.flux.Fqvx, gpureceive, diff);

	diffArray(XParam, XLoop, XModel.blocks, "Fhv", XModel.flux.Fhv, XModel_g.flux.Fhv, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "Fqvy", XModel.flux.Fqvy, XModel_g.flux.Fqvy, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "Sv", XModel.flux.Sv, XModel_g.flux.Sv, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "Fquy", XModel.flux.Fquy, XModel_g.flux.Fquy, gpureceive, diff);

	free(gpureceive);
	free(diff);
	
}
template void CompareCPUvsGPU<float>(Param XParam, Forcing<float> XForcing, Model<float> XModel, Model<float> XModel_g);
template void CompareCPUvsGPU<double>(Param XParam, Forcing<float> XForcing, Model<double> XModel, Model<double> XModel_g);




template <class T> void diffArray(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, std::string varname, T* cpu, T* gpu, T* dummy, T* out)
{
	T diff, maxdiff, rmsdiff;
	unsigned int nit = 0;
	//copy GPU back to the CPU (store in dummy)
	CopyGPUtoCPU(XParam.nblkmem, XParam.blksize, dummy, gpu);

	rmsdiff = T(0.0);
	maxdiff = XLoop.hugenegval;
	// calculate difference
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XBlock.active[ibl];


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int n = memloc(XParam, ix, iy, ib);
				diff = dummy[n] - cpu[n];
				maxdiff = utils::max(abs(diff), maxdiff);
				rmsdiff = rmsdiff + utils::sq(diff);
				nit++;
				out[n] = diff;
			}
		}
	}
	rmsdiff = rmsdiff / nit;

	

	if (maxdiff < (XLoop.epsilon * 2))
	{
		log(varname + " PASS");
	}
	else
	{
		log(varname + " FAIL: " + " Max difference: " + std::to_string(maxdiff) + " RMS difference: " + std::to_string(rmsdiff) + " Eps: " + std::to_string(XLoop.epsilon));
		defncvarBUQ(XParam, XBlock.active, XBlock.level, XBlock.xo, XBlock.yo, varname + "_CPU", 3, cpu);
		defncvarBUQ(XParam, XBlock.active, XBlock.level, XBlock.xo, XBlock.yo, varname + "_GPU", 3, dummy);
		defncvarBUQ(XParam, XBlock.active, XBlock.level, XBlock.xo, XBlock.yo, varname + "_diff", 3, out);
	}
	



}
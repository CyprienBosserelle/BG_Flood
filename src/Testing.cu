


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

	//============================================
	// Compare gradients for evolving parameters

	//GPU
	dim3 blockDim(16, 16, 1);
	dim3 gridDim(XParam.nblk, 1, 1);
	//gradientGPU(XParam, XLoop, XModel_g.blocks, XModel_g.evolv, XModel_g.grad);
	gradient << < gridDim, blockDim, 0 >> > (XParam.halowidth, XModel_g.blocks.active, XModel_g.blocks.level, (T)XParam.theta, (T)XParam.dx, XModel_g.evolv.h, XModel_g.grad.dhdx, XModel_g.grad.dhdy);
	CUDA_CHECK(cudaDeviceSynchronize());

	//CPU
	gradientCPU(XParam, XLoop, XModel.blocks, XModel.evolv, XModel.grad);

	
	// calculate difference
	diffArray(XParam, XLoop, XModel.blocks, XModel.grad.dhdx, XModel_g.grad.dhdx, XModel.evolv_o.h);

	creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo);
	//outvar = "h";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhdx_CPU", 3, XModel.grad.dhdx);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhdx_GPU", 3, XModel.evolv_o.h);

}
template void CompareCPUvsGPU<float>(Param XParam, Forcing<float> XForcing, Model<float> XModel, Model<float> XModel_g);
template void CompareCPUvsGPU<double>(Param XParam, Forcing<float> XForcing, Model<double> XModel, Model<double> XModel_g);




template <class T> void diffArray(Param XParam, Loop<T> XLoop, BlockP<T> XBlock, T* cpu, T* gpu, T* dummy)
{
	T diff, maxdiff, rmsdiff;
	unsigned int nit = 0;
	//copy GPU back to the CPU (store in dummy)
	CopyGPUtoCPU(XParam.nblk, XParam.blksize, dummy, gpu);

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
			}
		}
	}
	rmsdiff = rmsdiff / nit;
	log("Epsilon: " + std::to_string(XLoop.epsilon));
	log("RMS difference: " + std::to_string(rmsdiff));
	log("Max difference: " + std::to_string(maxdiff));

}



#include "Testing.h"




/*! \fn int main(int argc, char **argv)
* Main function 
* This function is the entry point to the software
*/
template <class T>
void TestingOutput(Param XParam, Model<T> XModel)
{
	std::string outvar;

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

	XLoop.totaltime = 0.0;

	XLoop.nextoutputtime = 0.2;

	//FlowCPU(XParam, XLoop, XModel);

	//log(std::to_string(XForcing.Bathy.val[50]));
	creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo);
	outvar = "h";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	outvar = "u";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	outvar = "v";
	//copyID2var(XParam, XModel.blocks, XModel.OutputVarMap[outvar]);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	outvar = "zb";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);
	outvar = "zs";
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.OutputVarMap[outvar]);


	FlowCPU(XParam, XLoop, XModel);


	//outvar = "cf";
	//defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, 3, XModel.cf);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhdx", 3, XModel.grad.dhdx);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhdy", 3, XModel.grad.dhdy);

	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fhv", 3, XModel.flux.Fhv);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fhu", 3, XModel.flux.Fhu);
	

	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqux", 3, XModel.flux.Fqux);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fquy", 3, XModel.flux.Fquy);

	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqvx", 3, XModel.flux.Fqvx);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqvy", 3, XModel.flux.Fqvy);


	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Su", 3, XModel.flux.Su);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Sv", 3, XModel.flux.Sv);


	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dh", 3, XModel.adv.dh);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhu", 3, XModel.adv.dhu);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhv", 3, XModel.adv.dhv);

	writenctimestep(XParam.outfile, XLoop.totaltime + XLoop.dt);
	

	outvar = "h";
	writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, XModel.OutputVarMap[outvar]);
	
	outvar = "zs";
	writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, XModel.OutputVarMap[outvar]);
	outvar = "u";
	writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, XModel.OutputVarMap[outvar]);
	outvar = "v";
	writencvarstepBUQ(XParam, 3, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, outvar, XModel.OutputVarMap[outvar]);
	
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


template <class T> void Gaussianhump(Param XParam, Forcing<float> XForcing, Model<T> XModel, Model<T> XModel_g)
{
	T x, y,delta;
	T cc = 100.0;
	T a = 0.2;

	T xorigin = XParam.xo + 0.5 * (XParam.xmax - XParam.xo);
	T yorigin = XParam.yo + 0.5 * (XParam.ymax - XParam.yo);
	Loop<T> XLoop;

	XLoop.hugenegval = std::numeric_limits<T>::min();

	XLoop.hugeposval = std::numeric_limits<T>::max();
	XLoop.epsilon = std::numeric_limits<T>::epsilon();

	XLoop.totaltime = 0.0;

	XLoop.nextoutputtime = 0.2;

	InitArrayBUQ(XParam, XModel.blocks, T(-1.0), XModel.zb);

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		int ib = XModel.blocks.active[ibl];
		delta = calcres(XParam.dx, XModel.blocks.level[ib]);


		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				//
				int n = memloc(XParam, ix, iy, ib);
				x = XModel.blocks.xo[ib] + ix * delta;
				y = XModel.blocks.yo[ib] + iy * delta;
				XModel.evolv.zs[n] = T(0.0) + a * exp(T(-1.0) * ((x - xorigin) * (x - xorigin) + (y - yorigin) * (y - yorigin)) / (2.0 * cc * cc));

				XModel.evolv.h[n] = XModel.evolv.zs[n] - XModel.zb[n];
			}
		}
	}


	for (int a = 0; a < 100; a++)
	{
		FlowCPU(XParam, XLoop, XModel);
	}
	

	creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo);


	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "h", 3, XModel.evolv.h);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "zs", 3, XModel.evolv.zs);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "u", 3, XModel.evolv.u);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "v", 3, XModel.evolv.v);


}
template void Gaussianhump<float>(Param XParam, Forcing<float> XForcing, Model<float> XModel, Model<float> XModel_g);
template void Gaussianhump<double>(Param XParam, Forcing<float> XForcing, Model<double> XModel, Model<double> XModel_g);



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

	XLoop.totaltime = 0.0;

	XLoop.nextoutputtime = 3600.0;


	T* gpureceive;
	T* diff;

	AllocateCPU(XParam.nblkmem, XParam.blksize, gpureceive);
	AllocateCPU(XParam.nblkmem, XParam.blksize, diff);


	//============================================
	// Compare gradients for evolving parameters
	
	// GPU
	FlowGPU(XParam, XLoop, XModel_g);
	T dtgpu = XLoop.dt;
	// CPU
	FlowCPU(XParam, XLoop, XModel);
	T dtcpu = XLoop.dt;
	// calculate difference
	//diffArray(XParam, XLoop, XModel.blocks, XModel.evolv.h, XModel_g.evolv.h, XModel.evolv_o.u);

	creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo);

	
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "h", 3, XModel.evolv_o.h);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "u", 3, XModel.evolv_o.u);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "v", 3, XModel.evolv_o.v);

	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqux", 3, XModel.flux.Fqux);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fquy", 3, XModel.flux.Fquy);

	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqvx", 3, XModel.flux.Fqvx);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Fqvy", 3, XModel.flux.Fqvy);


	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Su", 3, XModel.flux.Su);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "Sv", 3, XModel.flux.Sv);


	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dh", 3, XModel.adv.dh);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhu", 3, XModel.adv.dhu);
	defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "dhv", 3, XModel.adv.dhv);

	std::string varname = "dt";
	if (abs(dtgpu - dtcpu) < (XLoop.epsilon * 2))
	{
		log(varname + " PASS");
	}
	else
	{
		log(varname + " FAIL: " + " GPU(" + std::to_string(dtgpu) + ") - CPU("+std::to_string(dtcpu) +") =  difference: "+  std::to_string(abs(dtgpu - dtcpu)) + " Eps: " + std::to_string(XLoop.epsilon));
		
	}

	//Check evolving param
	diffArray(XParam, XLoop, XModel.blocks, "h", XModel.evolv_o.h, XModel_g.evolv_o.h, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "zs", XModel.evolv_o.zs, XModel_g.evolv_o.zs, gpureceive, diff);

	diffArray(XParam, XLoop, XModel.blocks, "u", XModel.evolv_o.u, XModel_g.evolv_o.u, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "v", XModel.evolv_o.v, XModel_g.evolv_o.v, gpureceive, diff);
	

	
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

	diffArray(XParam, XLoop, XModel.blocks, "dh", XModel.adv.dh, XModel_g.adv.dh, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "dhu", XModel.adv.dhu, XModel_g.adv.dhu, gpureceive, diff);
	diffArray(XParam, XLoop, XModel.blocks, "dhv", XModel.adv.dhv, XModel_g.adv.dhv, gpureceive, diff);



	
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

	

	if (maxdiff <= (XLoop.epsilon))
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
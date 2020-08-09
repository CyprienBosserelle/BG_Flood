


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
	XParam = creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo);
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
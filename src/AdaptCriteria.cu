


#include "AdaptCriteria.h"


template <class T> int AdaptCriteria(Param XParam, Forcing<float> XForcing, Model<T> XModel)
{
	int success = 0;
	if (XParam.AdatpCrit.compare("Threshold") == 0)
	{
		success = Thresholdcriteria(XParam, T(std::stod(XParam.Adapt_arg1)), XModel.OutputVarMap[XParam.Adapt_arg2], XModel.blocks, XModel.adapt.refine, XModel.adapt.coarsen);
	}
	if (XParam.AdatpCrit.compare("Inrange") == 0)
	{
		success = inrangecriteria(XParam, T(std::stod(XParam.Adapt_arg1)), T(std::stod(XParam.Adapt_arg2)), XModel.OutputVarMap[XParam.Adapt_arg3], XModel.blocks, XModel.adapt.refine, XModel.adapt.coarsen);
	}
	if (XParam.AdatpCrit.compare("Targetlevel") == 0)
	{
		for (int ig = 0; ig < XForcing.targetadapt.size(); ig++)
		{
			targetlevelcriteria(XParam, XForcing.targetadapt[ig], XModel.blocks, XModel.adapt.refine, XModel.adapt.coarsen);
		}
	}
	return success;
}
template int AdaptCriteria<float>(Param XParam, Forcing<float> XForcing, Model<float> XModel);
template int AdaptCriteria<double>(Param XParam, Forcing<float> XForcing, Model<double> XModel);


/*! \fn int Thresholdcriteria(Param XParam,T threshold, T* z, BlockP<T> XBlock,  bool*& refine, bool*& coarsen)
* Threshold criteria is a general form of wet dry criteria
* Simple wet/.dry refining criteria.
* if the block is wet -> refine is true
* if the block is dry -> coarsen is true
* beware the refinement sanity check is meant to be done after running this function
*/
template <class T> int Thresholdcriteria(Param XParam,T threshold, T* z, BlockP<T> XBlock, bool* refine, bool* coarsen)
{
	// Threshold criteria is a general form of wet dry criteria where esp is the threshold and h is the parameter tested
	// Below is written as a wet dry analogy where wet is vlaue above threshold and dry is below

	
	int success = 0;
	//int i;

	//Coarsen dry blocks and refine wet ones
	//CPU version


	// To start we assume all values are below the threshold
	bool iswet = false;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		refine[ib] = false; // only refine if all are wet
		coarsen[ib] = true; // always try to coarsen
		iswet = false;
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int i = memloc(XParam, ix, iy, ib);
				//(ix + XParam.halowidth) + (iy + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				
				if (z[i] > threshold)
				{
					iswet = true;
				}
			}
		}


		refine[ib] = iswet;
		coarsen[ib] = !iswet;

		//printf("ib=%d; refibe[ib]=%s\n", ib, iswet ? "true" : "false");
	}
	return success;
}
template  int Thresholdcriteria<float>(Param XParam, float threshold, float* z, BlockP<float> XBlock, bool* refine, bool* coarsen);
template  int Thresholdcriteria<double>(Param XParam, double threshold, double* z, BlockP<double> XBlock, bool* refine, bool* coarsen);

/*! \fn int inrangecriteria(Param XParam, T zmin, T zmax, T* z, BlockP<T> XBlock, bool*& refine, bool*& coarsen)
* Simple in-range refining criteria.
* if any value of z (could be any variable) is zmin <= z <= zmax the block will try to refine
* otherwise, the block will try to coarsen
* beware the refinement sanity check is meant to be done after running this function
*/
template<class T>
int inrangecriteria(Param XParam, T zmin, T zmax, T* z, BlockP<T> XBlock, bool* refine, bool* coarsen)
{
	// First use a simple refining criteria: zb>zmin && zb<zmax refine otherwise corasen
	int success = 0;
	//int i;


	// To start 
	bool isinrange = false;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		refine[ib] = false; // only refine if zb is in range
		coarsen[ib] = true; // always try to coarsen otherwise
		isinrange = false;
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = (ix + XParam.halowidth) + (iy + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				if (z[i] >= zmin && z[i] <= zmax)
				{
					isinrange = true;
				}
			}
		}


		refine[ib] = isinrange;
		coarsen[ib] = !isinrange;

		//printf("ib=%d; refibe[ib]=%s\n", ib, iswet ? "true" : "false");
	}
	return success;
}
template int inrangecriteria<float>(Param XParam, float zmin, float zmax, float* z, BlockP<float> XBlock, bool* refine, bool* coarsen);
template int inrangecriteria<double>(Param XParam, double zmin, double zmax, double* z, BlockP<double> XBlock, bool* refine, bool* coarsen);

/*! \fn 
*/
template<class T>
int targetlevelcriteria(Param XParam, StaticForcingP<int> targetlevelmap, BlockP<T> XBlock, bool* refine, bool* coarsen)
{
	int targetlevel;
	bool uplevel, downlevel;
	T delta, x, y;
	int success = 0;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];

		delta = calcres(XParam.dx, XBlock.level[ib]);

		uplevel = false;
		

		refine[ib] = false; // only refine if all are wet
		coarsen[ib] = true; // always try to coarsen
		
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				//
				int n = memloc(XParam, ix, iy, ib);
				x = XParam.xo + XBlock.xo[ib] + ix * delta;
				y = XParam.yo + XBlock.yo[ib] + iy * delta;

				targetlevel = int(round(interp2BUQ(x, y, targetlevelmap)));

				if (targetlevel >= XBlock.level[ib])
				{
					//printf("x=%f; y=%f; target=%d; level=%d", x, y, targetlevel, XBlock.level[ib]);
					uplevel = true;
					
				}


			}
		}

		if (uplevel)
		{
			refine[ib] = true; // only refine if all are wet
			coarsen[ib] = false;
		}
	}
	return success;
}
template int targetlevelcriteria<float>(Param XParam, StaticForcingP<int> targetlevelmap, BlockP<float> XBlock, bool* refine, bool* coarsen);
template int targetlevelcriteria<double>(Param XParam, StaticForcingP<int> targetlevelmap, BlockP<double> XBlock, bool* refine, bool* coarsen);

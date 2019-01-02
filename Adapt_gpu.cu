// This file contains functions for the model adaptivity.




int wetdryadapt(Param XParam)
{
	int sucess = 0;
	int i;
	//Coarsen dry blocks and refine wet ones
	//CPU version

	bool iswet = false;
	for (int ib = 0; ib < XParam.nblk; ib++)
	{
		newlevel[ib] = 0; // no resolution change by default
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * XParam.blksize;
				if (hh[i]>XParam.eps)
				{
					iswet = true;
				}
			}
		}
		if (iswet)
		{
			
				newlevel[ib] = 1;
			
		}
		


	}
	for (int ib = 0; ib < XParam.nblk; ib++)
	{
		//

	}

}


//int refineblk()
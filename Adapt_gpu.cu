// This file contains functions for the model adaptivity.




int wetdryadapt(Param XParam)
{
	int sucess = 0;
	int i;
	int tl, tr, lt, lb, bl, br, rb, rt;//boundary neighbour (max of 8)
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
		// if all the neighbour are not wet then coarsen if possible
		if (newlevel[topblk[ib]] == 0 && newlevel[botblk[ib]] == 0 && newlevel[leftblk[ib]] == 0 && newlevel[rightblk[ib]] == 0 && level[ib]<XParam.minlevel)
		{
			newlevel[ib] = -1;
		}
	}
	for (int ib = 0; ib < XParam.nblk; ib++)
	{
		//check whether neighbour need refinement
		if ((level[topblk[ib]] + newlevel[topblk[ib]] - newlevel[ib] - level[ib]) < -1);
		{
			newlevel[topblk[ib]] = newlevel[topblk[ib]] + 1;
			newlevel[rightblk[topblk[ib]]] = newlevel[rightblk[topblk[ib]]] + 1; // is this necessary?
		}
		if ((level[botblk[ib]] + newlevel[botblk[ib]] - newlevel[ib] - level[ib]) < -1);
		{
			newlevel[botblk[ib]] = newlevel[botblk[ib]] + 1;
			newlevel[rightblk[botblk[ib]]] = newlevel[rightblk[botblk[ib]]] + 1; // is this necessary?
		}
		if ((level[leftblk[ib]] + newlevel[leftblk[ib]] - newlevel[ib] - level[ib]) < -1);
		{
			newlevel[leftblk[ib]] = newlevel[leftblk[ib]]+1;
			newlevel[topblk[leftblk[ib]]] = newlevel[topblk[leftblk[ib]]]+1; // is this necessary?
		}
		if ((level[rightblk[ib]] + newlevel[rightblk[ib]] - newlevel[ib] - level[ib]) < -1);
		{
			newlevel[rightblk[ib]] = newlevel[rightblk[ib]]+1;
			newlevel[topblk[rightblk[ib]]] = newlevel[topblk[rightblk[ib]]]+1; // is this necessary?
		}

		

	}
	for (int ib = 0; ib < XParam.nblk; ib++)
	{
		if (newlevel[ib]>0)
		{
			//refine
		}
		else if (newlevel[ib] < 0)
		{
			//coarsen
		}
	}
	return 0;
}


//int refineblk()
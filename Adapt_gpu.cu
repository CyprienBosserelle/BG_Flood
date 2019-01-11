// This file contains functions for the model adaptivity.




int wetdryadapt(Param XParam)
{
	int success = 0;
	//int i;
	int tl, tr, lt, lb, bl, br, rb, rt;//boundary neighbour (max of 8)
	//Coarsen dry blocks and refine wet ones
	//CPU version

	bool iswet = false;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		newlevel[ib] = 0; // no resolution change by default
		iswet = false;
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				int i = ix + iy * 16 + ib * XParam.blksize;
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
	
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		// if all the neighbour are not wet then coarsen if possible
		if (newlevel[ib] == 0 && newlevel[topblk[ib]] == 0 && newlevel[rightblk[ib]] == 0 && newlevel[rightblk[topblk[ib]]] == 0 && level[ib]>XParam.minlevel)
		{
			newlevel[ib] = -1;
			newlevel[topblk[ib]] = -1;
			newlevel[rightblk[ib]] = -1;
			newlevel[rightblk[topblk[ib]]] = -1;
				
		}
	}

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		if (newlevel[ib] == 1 && level[ib] == XParam.maxlevel)
		{
			newlevel[ib] = 0;
		}
	}

	
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		//check whether neighbour need refinement
		
		if ((level[topblk[ib]] + newlevel[topblk[ib]] - newlevel[ib] - level[ib]) < -1)
		{
			printf("level diff=%d\n", level[topblk[ib]] + newlevel[topblk[ib]] - newlevel[ib] - level[ib]);
			newlevel[topblk[ib]] = min(newlevel[topblk[ib]] + 1,1);
			newlevel[rightblk[topblk[ib]]] = newlevel[rightblk[topblk[ib]]] + 1; // is this necessary?
		}
		if ((level[botblk[ib]] + newlevel[botblk[ib]] - newlevel[ib] - level[ib]) < -1)
		{
			newlevel[botblk[ib]] = newlevel[botblk[ib]] + 1;
			newlevel[rightblk[botblk[ib]]] = newlevel[rightblk[botblk[ib]]] + 1; // is this necessary?
		}
		if ((level[leftblk[ib]] + newlevel[leftblk[ib]] - newlevel[ib] - level[ib]) < -1)
		{
			newlevel[leftblk[ib]] = newlevel[leftblk[ib]]+1;
			newlevel[topblk[leftblk[ib]]] = newlevel[topblk[leftblk[ib]]]+1; // is this necessary?
		}
		if ((level[rightblk[ib]] + newlevel[rightblk[ib]] - newlevel[ib] - level[ib]) < -1)
		{
			newlevel[rightblk[ib]] = newlevel[rightblk[ib]]+1;
			newlevel[topblk[rightblk[ib]]] = newlevel[topblk[rightblk[ib]]]+1; // is this necessary?
		}

		

	}
	//Calc cumsum that will determine where the new cell will be located in the memory

	int csum = 0;
	int nrefineblk = 0;
	int ncoarsenlk = 0;
	int nnewblk = 0;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		//
		if (newlevel[ib]>0)
		{
			nrefineblk++;
			csum = csum + newlevel[ib] * 3;
		}
		if (newlevel[ib] < 0)
		{
			ncoarsenlk++;
		}

		csumblk[ib] = csum;

	}
	nnewblk = csum - ncoarsenlk/4*3;

	printf("%d blocks to be refiled, %d blocks to be coarsen; %d new blocks will be created\n", nrefineblk, ncoarsenlk, nnewblk);

	if (nnewblk>XParam.navailblk)
	{
		//reallocate memory to nmake more room
	}





	//coarsen


	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		int i, ii, ir , it , itr;
		if (newlevel[ib] < 0)
		{
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					i = ix + iy * 16 + ib * XParam.blksize;
					if (ix < 8 && iy < 8)
					{
						ii = ix * 2 + (iy * 2) * 16 + ib * XParam.blksize;
						ir = (ix * 2+1) + (iy * 2) * 16 + ib * XParam.blksize;
						it = (ix) * 2 + (iy * 2 + 1 ) * 16 + ib * XParam.blksize;
						itr = (ix * 2 +1) + (iy*2+1) * 16 + ib * XParam.blksize;
					}
					if (ix >= 8 && iy < 8)
					{
						ii = ((ix - 8) * 2) + (iy * 2) * 16 + rightblk[ib] * XParam.blksize;
						ir = ((ix - 8) * 2 + 1) + (iy * 2) * 16 + rightblk[ib] * XParam.blksize;
						it = ((ix - 8)) * 2 + (iy * 2 + 1) * 16 + rightblk[ib] * XParam.blksize;
						itr = ((ix - 8) * 2 + 1) + (iy * 2 + 1) * 16 + rightblk[ib] * XParam.blksize;
					}
					if (ix < 8 && iy >= 8)
					{
						ii = ix * 2 + ((iy-8) * 2) * 16 + topblk[ib] * XParam.blksize;
						ir = (ix * 2 + 1) + ((iy - 8) * 2) * 16 + topblk[ib] * XParam.blksize;
						it = (ix)* 2 + ((iy - 8) * 2 + 1) * 16 + topblk[ib] * XParam.blksize;
						itr = (ix * 2 + 1) + ((iy - 8) * 2 + 1) * 16 + topblk[ib] * XParam.blksize;
					}
					if (ix >= 8 && iy >= 8)
					{
						ii = (ix - 8) * 2 + ((iy - 8) * 2) * 16 + rightblk[topblk[ib]] * XParam.blksize;
						ir = ((ix - 8) * 2 + 1) + ((iy - 8) * 2) * 16 + rightblk[topblk[ib]] * XParam.blksize;
						it = (ix - 8) * 2 + ((iy - 8) * 2 + 1) * 16 + rightblk[topblk[ib]] * XParam.blksize;
						itr = ((ix - 8) * 2 + 1) + ((iy - 8) * 2 + 1) * 16 + rightblk[topblk[ib]] * XParam.blksize;
					}
					hh[i] = 0.25*(hho[ii] + hho[ir] + hho[it], hho[itr]);
					//zs, zb, uu,vv

					//update neighbour blk and neighbours' neighbours
					rightblk[ib] = rightblk[rightblk[ib]];
					topblk[ib] = topblk[topblk[ib]];


				}
			}
		}



	}



	return 0;
}


//int refineblk()
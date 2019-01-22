// This file contains functions for the model adaptivity.

bool isPow2(int x)
{
	//Greg Hewgill great explanation here:
	//https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
	//Note, this function will report true for 0, which is not a power of 2 but it is handy for us here

	return (x & (x - 1)) == 0;


}


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
		if (newlevel[ib] == 1 && level[ib] == XParam.maxlevel)
		{
			newlevel[ib] = 0;
		}
	}

	
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		// if all the neighbour are not wet then coarsen if possible
		double dxfac = XParam.dx/(1 << (level[ib] - 1));

		//only check for coarsening if the block analysed is a lower left corner block of the lower level
		
			if (isPow2((blockxo_d[ib] - XParam.xo + dxfac) / dxfac))
			{


				if (newlevel[topblk[ib]] == 0 && newlevel[rightblk[ib]] == 0 && newlevel[rightblk[topblk[ib]]] == 0 && level[ib] > XParam.minlevel)
				{
					newlevel[ib] = -1;
					newlevel[topblk[ib]] = -1;
					newlevel[rightblk[ib]] = -1;
					newlevel[rightblk[topblk[ib]]] = -1;

				}
				
			}
		
	}
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		//check whether neighbour need refinement

		if ((level[topblk[ib]] + newlevel[topblk[ib]] - newlevel[ib] - level[ib]) > 1)
		{
			//printf("level diff=%d\n", level[topblk[ib]] + newlevel[topblk[ib]] - newlevel[ib] - level[ib]);
			newlevel[ib] = min(newlevel[ib] + 1, 1);

		}
		if ((level[botblk[ib]] + newlevel[botblk[ib]] - newlevel[ib] - level[ib]) > 1)
		{
			newlevel[ib] = min(newlevel[ib] + 1, 1);

		}
		if ((level[leftblk[ib]] + newlevel[leftblk[ib]] - newlevel[ib] - level[ib]) > 1)
		{
			newlevel[ib] = min(newlevel[ib] + 1, 1);// is this necessary?
		}
		if ((level[rightblk[ib]] + newlevel[rightblk[ib]] - newlevel[ib] - level[ib]) > 1)
		{
			newlevel[ib] = min(newlevel[ib] + 1, 1); // is this necessary?
		}



	}

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		// if all the neighbour are not wet then coarsen if possible
		double dxfac = XParam.dx/(1 << (level[ib] - 1));

		//only check for coarsening if the block analysed is a lower left corner block of the lower level

		if (isPow2((blockxo_d[ib] - XParam.xo + dxfac) / dxfac))// Beware of round off error
		{
			if (newlevel[ib] < 0  && (newlevel[topblk[ib]] >= 0 || newlevel[rightblk[ib]] >= 0 || newlevel[rightblk[topblk[ib]]] >= 0))
			{
				newlevel[ib] = 0;
				newlevel[topblk[ib]] = 0;
				newlevel[rightblk[ib]] = 0;
				newlevel[rightblk[topblk[ib]]] = 0;

			}
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
			csum = csum + 3;
		}
		if (newlevel[ib] < 0)
		{
			ncoarsenlk++;
		}

		csumblk[ib] = csum;

	}
	nnewblk = 3*(nrefineblk - ncoarsenlk);

	printf("%d blocks to be refiled, %d blocks to be coarsen; %d new blocks will be created\n", nrefineblk, ncoarsenlk, nnewblk);

	if (nnewblk>XParam.navailblk)
	{
		//reallocate memory to make more room
	}





	//coarsen


	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		int i, ii, ir , it , itr;
		if (newlevel[ib] < 0)
		{
			double dxfac = 1.0/(1 << (level[ib] - 1))*XParam.dx;
			if (isPow2((blockxo_d[ib] - XParam.xo + dxfac) / dxfac))
			{
				for (int iy = 0; iy < 16; iy++)
				{
					for (int ix = 0; ix < 16; ix++)
					{
						i = ix + iy * 16 + ib * XParam.blksize;
						if (ix < 8 && iy < 8)
						{
							ii = ix * 2 + (iy * 2) * 16 + ib * XParam.blksize;
							ir = (ix * 2 + 1) + (iy * 2) * 16 + ib * XParam.blksize;
							it = (ix)* 2 + (iy * 2 + 1) * 16 + ib * XParam.blksize;
							itr = (ix * 2 + 1) + (iy * 2 + 1) * 16 + ib * XParam.blksize;
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
							ii = ix * 2 + ((iy - 8) * 2) * 16 + topblk[ib] * XParam.blksize;
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
						zs[i] = 0.25*(zso[ii] + zso[ir] + zso[it], zso[itr]);
						uu[i] = 0.25*(uuo[ii] + uuo[ir] + uuo[it], uuo[itr]);
						vv[i] = 0.25*(vvo[ii] + vvo[ir] + vvo[it], vvo[itr]);
						//zs, zb, uu,vv

						

						


					}
				}
				
				//Need more?
				
				
				availblk[XParam.navailblk] = rightblk[ib];
				availblk[XParam.navailblk+1] = topblk[ib];
				availblk[XParam.navailblk+2] = rightblk[topblk[ib]];

				XParam.navailblk = XParam.navailblk + 3;


				activeblk[rightblk[ib]] = -1;
				activeblk[topblk[ib]] = -1;
				activeblk[rightblk[topblk[ib]]] = -1;

				//check neighbour's
				rightblk[ib] = rightblk[rightblk[ib]];
				topblk[ib] = topblk[topblk[ib]];

				blockxo_d[ib] = blockxo_d[ib] + XParam.dx / (1 << (level[ib] + 1));
				blockyo_d[ib] = blockyo_d[ib] + XParam.dx / (1 << (level[ib] + 1));




			}
		}

	}


	//
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//
		int ib = activeblk[ibl];
		int o,ot,or,otr,i,ir,it,itr,ii,iit,iir,iitr,iii,iiir,iiit,iiitr;
		if (newlevel[ib] > 0)
		{

			double xoblk = blockxo_d[ib] - XParam.dx / (1 << (level[ib] + 1));
			double yoblk = blockyo_d[ib] - XParam.dx / (1 << (level[ib] + 1));
			

			//
			for (int iy = 0; iy < 16; iy++)
			{
				for (int ix = 0; ix < 16; ix++)
				{
					//
					o = ix + iy * 16 + ib * XParam.blksize;
					or = (ix+1) + iy * 16 + ib * XParam.blksize;
					ot = ix + (iy+1) * 16 + ib * XParam.blksize;
					otr = (ix+1) + (iy+1) * 16 + ib * XParam.blksize;
					
					i = ix + iy * 16 + availblk[csumblk[ibl]] * XParam.blksize;
					ii = ix + iy * 16 + availblk[csumblk[ibl]+1] * XParam.blksize;
					iii = ix + iy * 16 + availblk[csumblk[ibl]+2] * XParam.blksize;


					//hh[o] = hh[or] = hh[ot] = hh[tr] = hho[o];


		
				}
			}
		}

	}


	return 0;
}


//int refineblk()
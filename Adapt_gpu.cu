// This file contains functions for the model adaptivity.

bool isPow2(int x)
{
	//Greg Hewgill great explanation here:
	//https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
	//Note, this function will report true for 0, which is not a power of 2 but it is handy for us here

	return (x & (x - 1)) == 0;


}

template <class T> double calcres(T dx, int level)
{
	return level < 0 ? dx * (1 << abs(level)) : dx / (1 << level);
}

int wetdryadapt(Param XParam)
{


	// First use a simple refining criteria: wet or dry
	int success = 0;
	//int i;
	int tl, tr, lt, lb, bl, br, rb, rt;//boundary neighbour (max of 8)
	//Coarsen dry blocks and refine wet ones
	//CPU version


	// To start 
	bool iswet = false;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		refine[ib] = false; // only refine if all are wet
		coarsen[ib] = true; // always try to coarsen
		iswet = false;
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				int i = ix + iy * 16 + ib * XParam.blksize;
				if (hh[i] > XParam.eps)
				{
					iswet = true;
				}
			}
		}


		refine[ib] = iswet;
		coarsen[ib] = !iswet;
	}



	// Can't actually refine if the level is the max level (i.e. finest)

	// this may be over-ruled later on

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		if (refine[ib] == true && level[ib] == XParam.maxlevel)
		{
			refine[ib] = false;
		}
	}



	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		//check whether neighbour need refinement


		if (refine[ib] == false)
		{
			//topleft blk
			if (refine[topblk[ib]] == true && (level[topblk[ib]] - level[ib]) > 0)
			{
				refine[ib] = true;
				coarsen[ib] = false;
			}
			//top right if lev=lev+1
			if ((level[topblk[ib]] - level[ib]) > 0)
			{
				if (refine[rightblk[topblk[ib]]] == true && (level[rightblk[topblk[ib]]] - level[ib]) > 0)
				{
					refine[ib] = true;
					coarsen[ib] = false;
				}
			}
			//bot left

			if (refine[botblk[ib]] == true && (level[botblk[ib]] - level[ib]) > 0)
			{
				refine[ib] = true;
				coarsen[ib] = false;
			}
			//bot right
			if ((level[botblk[ib]] - level[ib]) > 0)
			{
				if (refine[rightblk[botblk[ib]]] == true && (level[rightblk[botblk[ib]]] - level[ib]) > 0)
				{
					refine[ib] = true;
					coarsen[ib] = false;
				}
			}
			//Left bottom
			if (refine[leftblk[ib]] == true && (level[leftblk[ib]] - level[ib]) > 0)
			{
				refine[ib] = true;
				coarsen[ib] = false;
			}
			if ((level[leftblk[ib]] - level[ib]) > 0)
			{
				//left top
				if (refine[topblk[leftblk[ib]]] == true && (level[topblk[leftblk[ib]]] - level[ib]) > 0)
				{
					refine[ib] = true;
					coarsen[ib] = false;
				}
			}
			if (refine[rightblk[ib]] == true && (level[rightblk[ib]] - level[ib]) > 0)
			{
				refine[ib] = true;
				coarsen[ib] = false;
			}
			if (level[rightblk[ib]] - level[ib] > 0)
			{
				//
				if (refine[topblk[rightblk[ib]]] == true && (level[topblk[rightblk[ib]]] - level[ib]) > 0)
				{
					refine[ib] = true;
					coarsen[ib] = false;
				}

			}

		}
	}




	// Can't actually coarsen if top, right and topright block are not all corsen
		
	
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];

		//printf("ib=%d\n", ib);
		// if all the neighbour are not wet then coarsen if possible
		double dxfac = calcres(XParam.dx, level[ib]);
		//printf("blockxo_d[ib]=%f, dxfac=%f, ((blx-xo)/dx)%2=%d\n", blockxo_d[ib], dxfac, (int((blockxo_d[ib] - XParam.xo) / dxfac / XParam.blkwidth) % 2));
		//only check for coarsening if the block analysed is a lower left corner block of the lower level
		//need to prevent coarsenning if the block is on the model edges...
		  //((int((blockxo_d[ib] - XParam.xo) / dxfac) % 2) == 0 && (int((blockyo_d[ib] - XParam.yo) / dxfac) % 2) == 0) && rightblk[ib] != ib && topblk[ib] != ib && rightblk[topblk[ib]] != topblk[ib]
		if (coarsen[ib] == true)
		{
			//if this block is a lower left corner block of teh potentialy coarser block
			if (((int((blockxo_d[ib] - XParam.xo) / dxfac / XParam.blkwidth) % 2) == 0 && (int((blockyo_d[ib] - XParam.yo) / dxfac / XParam.blkwidth) % 2) == 0 && rightblk[ib] != ib && topblk[ib] != ib && rightblk[topblk[ib]] != topblk[ib]))
			{
				//if all the neighbour blocks ar at the same level
				if (level[ib] == level[rightblk[ib]] && level[ib] == level[topblk[ib]] && level[ib] == level[rightblk[topblk[ib]]])
				{
					//if right, top and topright block teh same level and can coarsen
					if (coarsen[rightblk[ib]] == true && coarsen[topblk[ib]] == true && coarsen[rightblk[topblk[ib]]] == true)
					{
						//Yes 
						//coarsen[ib] = true;
					}
					else
					{
						coarsen[ib] = false;
					}
				}
				else
				{
					coarsen[ib] = false;
				}

			}
			else
			{
				coarsen[ib] = false;
			}
		}
		
	}
	/*for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		printf("level =%d, %d, %d, %d\n", ib, level[ib], refine[ib], coarsen[ib]);
	}*/

	


	




		
	//Calc cumsum that will determine where the new cell will be located in the memory

	int csum = 0;
	int nrefineblk = 0;
	int ncoarsenlk = 0;
	int nnewblk = 0;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		//
		if (refine[ib]==true)
		{
			nrefineblk++;
			csum = csum+3;
		}
		if (coarsen[ib] == true)
		{
			ncoarsenlk++;
		}

		csumblk[ib] = csum;

	}

	nnewblk = 3*nrefineblk - ncoarsenlk*3;

	printf("%d blocks to be refined, %d blocks to be coarsen; %d blocks to be freed (%d are already available) %d new blocks will be created\n", nrefineblk, ncoarsenlk, ncoarsenlk * 3 , XParam.navailblk, nnewblk);
	//printf("csunblk[end]=%d; navailblk=%d\n", csumblk[XParam.nblk - 1], XParam.navailblk);
	if (nnewblk>XParam.navailblk)
	{
		//reallocate memory to make more room
		int nblkmem;
		nblkmem = (int)ceil((XParam.nblk + nnewblk)* XParam.membuffer);
		XParam.nblkmem = nblkmem;

		// HH UU VV ZS ZB
		ReallocArray(nblkmem, XParam.blksize, hh);
		ReallocArray(nblkmem, XParam.blksize, zs);
		ReallocArray(nblkmem, XParam.blksize, zb);
		ReallocArray(nblkmem, XParam.blksize, uu);
		ReallocArray(nblkmem, XParam.blksize, vv);



		//also reallocate Blk info
		ReallocArray(nblkmem, 1, blockxo);
		ReallocArray(nblkmem, 1, blockyo);

		ReallocArray(nblkmem, 1, blockxo_d);
		ReallocArray(nblkmem, 1, blockyo_d);

		ReallocArray(nblkmem, 1, leftblk);
		ReallocArray(nblkmem, 1, rightblk);
		ReallocArray(nblkmem, 1, topblk);
		ReallocArray(nblkmem, 1, botblk);

		ReallocArray(nblkmem, 1, level);
		ReallocArray(nblkmem, 1, newlevel);
		ReallocArray(nblkmem, 1, activeblk);
		ReallocArray(nblkmem, 1, availblk);
		ReallocArray(nblkmem, 1, csumblk);
		
		ReallocArray(nblkmem, 1, coarsen);
		ReallocArray(nblkmem, 1, refine);
		
		
		
		


		// Reconstruct avail blk
		XParam.navailblk = 0;
		for (int ibl = 0; ibl < (XParam.nblkmem - XParam.nblk); ibl++)
		{

			availblk[ibl] = XParam.nblk + ibl;
			XParam.navailblk++;

		}
		
		printf("Reallocation complete: %d new blocks are available\n", XParam.navailblk);

	}





	//coarsen


	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		int i, ii, ir , it , itr;
		if (coarsen[ib] == true)
		{
			double dxfac = calcres(XParam.dx, level[ib]);
			int xnode = int((blockxo_d[ib] - XParam.xo) / dxfac / XParam.blkwidth);
			int ynode = int((blockyo_d[ib] - XParam.yo) / dxfac / XParam.blkwidth);
			//Problem in the if statement below...
			printf("ib=%d; xnode=%d; ynode=%d; right_id=%d, top_id=%d, topright_id=%d \n", ib, xnode, ynode, rightblk[ib], topblk[ib], rightblk[topblk[ib]]);
			
			//if (((xnode % 2) == 0 && (ynode % 2) == 0) && rightblk[ib] != ib && topblk[ib] != ib && rightblk[topblk[ib]] != topblk[ib])
			{

				//printf("blockxo_d[ib]=%f\tib=%d\trightblk[ib]=%d\ttopblk[ib]=%d\trightblk[topblk[ib]]=%d\n", blockxo_d[ib], ib, rightblk[ib], topblk[ib], rightblk[topblk[ib]]);
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
				
				// Make right, top and top-right block available for refine step
				availblk[XParam.navailblk] = rightblk[ib];
				availblk[XParam.navailblk+1] = topblk[ib];
				availblk[XParam.navailblk+2] = rightblk[topblk[ib]];
				//printf("availblk[XParam.navailblk]=%d; availblk[XParam.navailblk + 1]=%d; availblk[XParam.navailblk + 2]=%d; \n ", availblk[XParam.navailblk], availblk[XParam.navailblk + 1], availblk[XParam.navailblk + 2]);

				// increment available block count
				XParam.navailblk = XParam.navailblk + 3;

				// Make right, top and top-right block inactive
				activeblk[rightblk[ib]] = -1;
				activeblk[topblk[ib]] = -1;
				activeblk[rightblk[topblk[ib]]] = -1;

				//check neighbour's (Full neighbour happens below)
				rightblk[ib] = rightblk[rightblk[ib]];
				topblk[ib] = topblk[topblk[ib]];

				blockxo_d[ib] = blockxo_d[ib] + calcres(XParam.dx, level[ib] + 1);
				blockyo_d[ib] = blockyo_d[ib] + calcres(XParam.dx, level[ib] + 1);
				printf("ib=%d; blockxo_d[ib]=%f; blockyo_d[ib]=%f; right_id=%d, top_id=%d, topright_id=%d \n", ib, blockxo_d[ib], blockyo_d[ib], rightblk[ib], topblk[ib], rightblk[topblk[ib]]);



				




			}
		}

	}


	//refine
	int nblk = XParam.nblk;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//
		
		int ib = activeblk[ibl];
		int o, oo, ooo, oooo;
		int i, ii, iii, iiii;


		//printf("ib=%d\n", ib);


		if (ib >= 0) // ib can be -1 for newly inactive blocks
		{
			if (refine[ib] == true)
			{

				double delx = calcres(XParam.dx, level[ib] + 1);
				double xoblk = blockxo_d[ib] - 0.5*delx;
				double yoblk = blockyo_d[ib] - 0.5*delx;

				int oldtop, oldleft, oldright, oldbot;

				oldtop = topblk[ib];
				oldbot = botblk[ib];
				oldright = rightblk[ib];
				oldleft = leftblk[ib];

				//printf("ib=%d; availblk[csumblk[ibl]]=%d; availblk[csumblk[ibl] + 1]=%d; availblk[csumblk[ibl] + 2]=%d; \n ", ib, availblk[csumblk[ibl]], availblk[csumblk[ibl] + 1], availblk[csumblk[ibl] + 2]);


				//
				for (int iy = 0; iy < 16; iy++)
				{
					for (int ix = 0; ix < 16; ix++)
					{
						//

						o = ix + iy * 16 + ib * XParam.blksize;
						i = round(ix*0.5) + round(iy*0.5) * 16 + ib * XParam.blksize;
						oo = ix + iy * 16 + availblk[csumblk[ibl]] * XParam.blksize;
						ii = (round(ix*0.5) + 8) + round(iy*0.5) * 16 + ib * XParam.blksize;
						ooo = ix + iy * 16 + availblk[csumblk[ibl] + 1] * XParam.blksize;
						iii = round(ix*0.5) + (round(iy*0.5) + 8) * 16 + ib * XParam.blksize;
						oooo = ix + iy * 16 + availblk[csumblk[ibl] + 2] * XParam.blksize;
						iiii = (round(ix*0.5) + 8) + (round(iy*0.5) + 8) * 16 + ib * XParam.blksize;

						//printf("ib=%d; i=%d; o=%d; ii=%d; oo=%d; iii=%d; ooo=%d; iiii=%d; oooo=%d;\n", ib, i, o, ii, oo, iii, ooo, iiii, oooo);
						//printf("csumblk[ibl]=%d\tavailblk[csumblk[ibl]]=%d\tavailblk[csumblk[ibl]+1]=%d\tavailblk[csumblk[ibl]+2]=%d\n", csumblk[ibl], availblk[csumblk[ibl]], availblk[csumblk[ibl]+1], availblk[csumblk[ibl]+2]);


						//hh[o] = hh[or] = hh[ot] = hh[tr] = hho[o];
						//flat interpolation // need to replace with simplify bilinear
						//zs needs to be interpolated from texture
						hh[o] = hho[i];
						hh[oo] = hho[ii];
						hh[ooo] = hho[iii];
						hh[oooo] = hho[iiii];

					}
				}

				// sort out block info

				


				blockxo_d[ib] = xoblk;
				blockyo_d[ib] = yoblk;

				//bottom right blk
				blockxo_d[availblk[csumblk[ibl]]] = xoblk + 16 * delx;
				blockyo_d[availblk[csumblk[ibl]]] = yoblk;
				//top left blk
				blockxo_d[availblk[csumblk[ibl] + 1]] = xoblk;
				blockyo_d[availblk[csumblk[ibl] + 1]] = yoblk + 16 * delx;
				//top right blk
				blockxo_d[availblk[csumblk[ibl] + 2]] = xoblk + 16 * delx;
				blockyo_d[availblk[csumblk[ibl] + 2]] = yoblk + 16 * delx;


				//sort out blocks neighbour

				topblk[ib] = availblk[csumblk[ibl] + 1];
				topblk[availblk[csumblk[ibl] + 1]] = oldtop;
				topblk[availblk[csumblk[ibl]]] = availblk[csumblk[ibl] + 2];

				//printf("topblk[ib]=%d; oldtop=%d; topblk[availblk[csumblk[ibl] + 1]]=%d; topblk[availblk[csumblk[ibl]]]=%d; \n ", availblk[csumblk[ibl] + 1], oldtop, topblk[availblk[csumblk[ibl] + 1]], topblk[availblk[csumblk[ibl]]]);

				rightblk[ib] = availblk[csumblk[ibl]];
				rightblk[availblk[csumblk[ibl] + 1]] = availblk[csumblk[ibl] + 2];
				rightblk[availblk[csumblk[ibl]]] = oldright;

				botblk[availblk[csumblk[ibl] + 1]] = ib;
				botblk[availblk[csumblk[ibl] + 2]] = availblk[csumblk[ibl]];

				leftblk[availblk[csumblk[ibl]]] = ib;
				leftblk[availblk[csumblk[ibl] + 2]] = availblk[csumblk[ibl] + 1];

				
			}
		}

	}



	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//
		int oldtop, oldleft, oldright, oldbot;
		int ib = activeblk[ibl];
		int o, oo, ooo, oooo;
		int i, ii, iii, iiii;

		//printf("ib=%d\n", ib);
		if (ib >=0) // ib can be -1 for newly inactive blocks
		{
			if (refine[ib] == true)
			{
				////
				oldtop = topblk[topblk[ib]];
				oldright = rightblk[rightblk[ib]];
				oldleft = leftblk[ib];
				oldbot = botblk[ib];

				//printf("ib=%d; oldtop=%d; oldright=%d; oldleft=%d; oldbot=%d\n ", ib,oldtop,oldright,oldleft,oldbot);
				//printf("availblk[csumblk[ibl]]=%d, availblk[csumblk[ibl]+1] = %d, availblk[csumblk[ibl]+2]=%d\n", availblk[csumblk[ibl]], availblk[csumblk[ibl] + 1], availblk[csumblk[ibl] + 2]);
				if (level[oldtop] + newlevel[oldtop] < level[ib] + newlevel[ib])
				{
					topblk[availblk[csumblk[ibl] + 2]] = oldtop;
				}
				else
				{
					topblk[availblk[csumblk[ibl] + 2]] = rightblk[oldtop];
				}

				/////
				if (level[oldright] + newlevel[oldright] < level[ib] + newlevel[ib])
				{
					rightblk[availblk[csumblk[ibl] + 2]] = oldright;
				}
				else
				{
					rightblk[availblk[csumblk[ibl] + 2]] = topblk[oldright];
				}

				/////
				if (level[oldleft] + newlevel[oldleft] < level[ib] + newlevel[ib])
				{
					leftblk[availblk[csumblk[ibl] + 1]] = oldleft;
				}
				else
				{
					leftblk[availblk[csumblk[ibl] + 1]] = topblk[oldleft];
				}

				/////
				if (level[oldbot] + newlevel[oldbot] < level[ib] + newlevel[ib])
				{
					botblk[availblk[csumblk[ibl]]] = oldbot;
				}
				else
				{
					botblk[availblk[csumblk[ibl]]] = rightblk[oldbot];
				}

				//printf("level=%d\n", level[ib]);
				level[availblk[csumblk[ibl]]] = level[ib];
				level[availblk[csumblk[ibl]+1]] = level[ib];
				level[availblk[csumblk[ibl]+2]] = level[ib];

				newlevel[availblk[csumblk[ibl]]] = newlevel[ib];
				newlevel[availblk[csumblk[ibl] + 1]] = newlevel[ib];
				newlevel[availblk[csumblk[ibl] + 2]] = newlevel[ib];

				activeblk[nblk] = availblk[csumblk[ibl]];
				activeblk[nblk + 1] = availblk[csumblk[ibl] + 1];
				activeblk[nblk + 2] = availblk[csumblk[ibl] + 2];

				//printf("ib=%d; ib_right=%d; ib_top=%d; ib_TR=%d\n", ib, activeblk[nblk], activeblk[nblk + 1], activeblk[nblk + 2]);


				nblk = nblk + 3;
			}
		}
	}

	//update level
	for (int ibl = 0; ibl < nblk; ibl++)
	{
		//
		int oldlevel;
		int ib = activeblk[ibl];

		//printf("ib=%d; l=%d; xo=%f; yo=%f\n", ib, level[ib], blockxo_d[ib], blockyo_d[ib]);
		
		if (ib >= 0) // ib can be -1 for newly inactive blocks
		{

			oldlevel = level[ib];
			//printf("oldlevel=%d, newlevel=%d, level=%d\n", oldlevel, newlevel[ib], oldlevel + min(newlevel[ib], 1));
			if (coarsen[ib] == true)
			{
				level[ib] = oldlevel - 1;
			}
			if (refine[ib] == true)
			{
				level[ib] = oldlevel + 1;
			}


			{
				printf("ib=%d; oldlevel=%d; newlevel[ib]=%d; l=%d;  block_xo=%f; block_yo=%f\n", ib, oldlevel, newlevel[ib], level[ib], blockxo_d[ib], blockyo_d[ib]);
			}
		}
	}

	// Reorder activeblk
	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		//reuse newlevel as temporary storage for activeblk
		newlevel[ibl] = activeblk[ibl];
	
	//printf("ibl=%d; activeblk[ibl]=%d; newlevel[ibl]=%d;\n", ibl, activeblk[ibl], newlevel[ibl]);
	}

	int ib = 0;
	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		if (newlevel[ibl] != -1)//i.e. old activeblk
		{
			activeblk[ib] = newlevel[ibl];
			printf("ib=%d; l=%d; xo=%f; yo=%f\n", activeblk[ib], level[activeblk[ib]], blockxo_d[activeblk[ib]], blockyo_d[activeblk[ib]]);
			ib++;
		}
	}

	nblk = ib;

	for (int ibl = 0; ibl < nblk; ibl++)
	{
		//reset
		newlevel[ibl] = 0;
		ib = activeblk[ibl];

		//if (level[ib]>1)
		{
			//printf("ib=%d; l=%d; xo=%f; yo=%f\n", ib, level[ib], blockxo_d[ib], blockyo_d[ib]);
		}

		

	}

	//reset blockxo and blockyo
	
	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		blockxo[ibl] = blockxo_d[ibl];
		blockyo[ibl] = blockyo_d[ibl];
	}
	

	return nblk;
}


//int refineblk()
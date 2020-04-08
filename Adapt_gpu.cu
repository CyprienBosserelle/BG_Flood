// This file contains functions for the model adaptivity.

bool isPow2(int x)
{
	//Greg Hewgill great explanation here:
	//https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
	//Note, this function will report true for 0, which is not a power of 2 but it is handy for us here

	return (x & (x - 1)) == 0;


}

template <class T> 
__host__ __device__ double calcres(T dx, int level)
{
	return level < 0 ? dx * (1 << abs(level)) : dx / (1 << level);
}


int wetdrycriteria(Param XParam, bool*& refine, bool*& coarsen)
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
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = ix + iy * XParam.blkwidth + ib * XParam.blksize;
				if (hh[i] > XParam.eps)
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






Param adapt(Param XParam)
{


	



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
		if (refine[ib] == true )
		{
			coarsen[rightblk[ib]] = false;
			coarsen[leftblk[ib]] = false;
			coarsen[topblk[ib]] = false;
			coarsen[botblk[ib]] = false;
		}
	}

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = activeblk[ibl];
		if (coarsen[ib] == true)
		{
			int levi = level[ib];

			if (levi < level[leftblk[ib]] || levi < level[rightblk[ib]] || levi < level[topblk[ib]] || levi < level[botblk[ib]])
			{
				coarsen[ib] = false;
			}
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

	// Reconstruct avail blk
	XParam.navailblk = 0;
	for (int ibl = 0; ibl < (XParam.nblkmem - XParam.nblk); ibl++)
	{

		availblk[ibl] = XParam.nblk + ibl;
		XParam.navailblk++;

	}


	




		
	//Calc cumsum that will determine where the new blocks will be located in the memory

	int csum = -3;
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

	printf("%d blocks to be refined, %d blocks to be coarsen (with neighbour); %d blocks untouched; %d blocks to be freed (%d are already available) %d new blocks will be created\n", nrefineblk, ncoarsenlk, XParam.nblk- nrefineblk-4* ncoarsenlk,  ncoarsenlk * 3 , XParam.navailblk, nnewblk);
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

		// Also need to reallocate all the others!
		ReallocArray(XParam.nblkmem, XParam.blksize, hho);
		ReallocArray(XParam.nblkmem, XParam.blksize, zso);
		ReallocArray(XParam.nblkmem, XParam.blksize, uuo);
		ReallocArray(XParam.nblkmem, XParam.blksize, vvo);

		ReallocArray(XParam.nblkmem, XParam.blksize, dzsdx);
		ReallocArray(XParam.nblkmem, XParam.blksize, dhdx);
		ReallocArray(XParam.nblkmem, XParam.blksize, dudx);
		ReallocArray(XParam.nblkmem, XParam.blksize, dvdx);

		ReallocArray(XParam.nblkmem, XParam.blksize, dzsdy);
		ReallocArray(XParam.nblkmem, XParam.blksize, dhdy);
		ReallocArray(XParam.nblkmem, XParam.blksize, dudy);
		ReallocArray(XParam.nblkmem, XParam.blksize, dvdy);

		ReallocArray(XParam.nblkmem, XParam.blksize, Su);
		ReallocArray(XParam.nblkmem, XParam.blksize, Sv);
		ReallocArray(XParam.nblkmem, XParam.blksize, Fhu);
		ReallocArray(XParam.nblkmem, XParam.blksize, Fhv);

		ReallocArray(XParam.nblkmem, XParam.blksize, Fqux);
		ReallocArray(XParam.nblkmem, XParam.blksize, Fquy);
		ReallocArray(XParam.nblkmem, XParam.blksize, Fqvx);
		ReallocArray(XParam.nblkmem, XParam.blksize, Fqvy);

		ReallocArray(XParam.nblkmem, XParam.blksize, dh);
		ReallocArray(XParam.nblkmem, XParam.blksize, dhu);
		ReallocArray(XParam.nblkmem, XParam.blksize, dhv);
		ReallocArray(XParam.nblkmem, XParam.blksize, cf);

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
			activeblk[XParam.nblk + ibl] = -1;

			availblk[ibl] = XParam.nblk + ibl;
			XParam.navailblk++;

			printf("ibl=%d; availblk[ibl]=%d;\n",ibl, availblk[ibl]);

		}
		
		printf("Reallocation complete: %d new blocks are available\n", XParam.navailblk);

	}

	// Initialise newlevel (Do this every time because new level is reused later)

	for (int ib = 0; ib < XParam.nblkmem; ib++)
	{
		//
		
		//int ib = activeblk[ibl];

	
		newlevel[ib] = level[ib];

		
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


						// These are the only guys that need to be coarsen, other are recalculated on teh fly or interpolated from forcing
						hh[i] = 0.25*(hho[ii] + hho[ir] + hho[it] + hho[itr]);
						zs[i] = 0.25*(zso[ii] + zso[ir] + zso[it] + zso[itr]);
						uu[i] = 0.25*(uuo[ii] + uuo[ir] + uuo[it] + uuo[itr]);
						vv[i] = 0.25*(vvo[ii] + vvo[ir] + vvo[it] + vvo[itr]);
						//zb[i] = 0.25 * (zbo[ii] + zbo[ir] + zbo[it], zbo[itr]);
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
				// Bot and left blk should remain unchanged at this stage (they will change if the neighbour themselves change)

				blockxo_d[ib] = blockxo_d[ib] + calcres(XParam.dx, level[ib] + 1);
				blockyo_d[ib] = blockyo_d[ib] + calcres(XParam.dx, level[ib] + 1);


				newlevel[ib] = level[ib] - 1;
				//printf("ib=%d; blockxo_d[ib]=%f; blockyo_d[ib]=%f; right_id=%d, top_id=%d, topright_id=%d \n", ib, blockxo_d[ib], blockyo_d[ib], rightblk[ib], topblk[ib], rightblk[topblk[ib]]);



				




			
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

				//printf("ib=%d; nblk=%d; availblk[csumblk[ibl]]=%d; availblk[csumblk[ibl] + 1]=%d; availblk[csumblk[ibl] + 2]=%d; \n ", ib, nblk, availblk[csumblk[ibl]], availblk[csumblk[ibl] + 1], availblk[csumblk[ibl] + 2]);


				// Flat interolation
				/**
				for (int iy = 0; iy < 16; iy++)
				{
					for (int ix = 0; ix < 16; ix++)
					{
						//

						o = ix + iy * 16 + ib * XParam.blksize;
						i = floor(ix*0.5) + floor(iy*0.5) * 16 + ib * XParam.blksize;
						oo = ix + iy * 16 + availblk[csumblk[ibl]] * XParam.blksize;
						ii = (floor(ix*0.5) + 8) + floor(iy*0.5) * 16 + ib * XParam.blksize;
						ooo = ix + iy * 16 + availblk[csumblk[ibl] + 1] * XParam.blksize;
						iii = floor(ix*0.5) + (floor(iy*0.5) + 8) * 16 + ib * XParam.blksize;
						oooo = ix + iy * 16 + availblk[csumblk[ibl] + 2] * XParam.blksize;
						iiii = (floor(ix*0.5) + 8) + (floor(iy*0.5) + 8) * 16 + ib * XParam.blksize;

						//printf("ib=%d; i=%d; o=%d; ii=%d; oo=%d; iii=%d; ooo=%d; iiii=%d; oooo=%d;\n", ib, i, o, ii, oo, iii, ooo, iiii, oooo);
						//printf("csumblk[ibl]=%d\tavailblk[csumblk[ibl]]=%d\tavailblk[csumblk[ibl]+1]=%d\tavailblk[csumblk[ibl]+2]=%d\n", csumblk[ibl], availblk[csumblk[ibl]], availblk[csumblk[ibl]+1], availblk[csumblk[ibl]+2]);


						//hh[o] = hh[or] = hh[ot] = hh[tr] = hho[o];
						//flat interpolation // need to replace with simplify bilinear
						//zb needs to be interpolated from texture


						//printf("hho[%d]=%f; hho[%d]=%f; hho[%d]=%f; hho[%d]=%f\n",i,hho[i],ii,hho[ii],iii,hho[iii],iiii,hho[iiii]);
						hh[o] = hho[i];
						hh[oo] = hho[ii];
						hh[ooo] = hho[iii];
						hh[oooo] = hho[iiii];

					}
				}
				**/
				//printf("ibl=%d; ib=%d; csumblk[ib]=%d; availblk[csumblk[ib]]=%d; availblk[csumblk[ib]+1]=%d; availblk[csumblk[ib]+2]=%d;\n", ibl, ib, csumblk[ib], availblk[csumblk[ib]], availblk[csumblk[ib] + 1], availblk[csumblk[ib] + 2]);

				
				// Bilinear interpolation
				for (int iy = 0; iy < 16; iy++)
				{
					for (int ix = 0; ix < 16; ix++)
					{
						int kx[] = { 0, 8, 0, 8};
						int ky[] = { 0, 0, 8, 8};
						int kb[] = { ib, availblk[csumblk[ib]], availblk[csumblk[ib] + 1], availblk[csumblk[ib] + 2]};
						
						//double mx, my;

						for (int kk = 0; kk < 4; kk++)
						{

							int cx, fx, cy, fy;
							double h11, h12, h21, h22;
							double zs11, zs12, zs21, zs22;
							double u11, u12, u21, u22;
							double v11, v12, v21, v22;
							double lx, ly, rx, ry;

							lx = ix * 0.5 - 0.25;
							ly = iy * 0.5 - 0.25;


							fx = max((int) floor(lx) + kx[kk],0);
							cx = min((int) ceil(lx) + kx[kk],15);
							fy = max((int) floor(ly) + ky[kk],0);
							cy = min((int) ceil(ly) + ky[kk],15);

							rx= (lx) + (double)kx[kk];
							ry= (ly) + (double)ky[kk];


						


							o = ix + iy * 16 + kb[kk] * XParam.blksize;

							h11 = hho[fx + fy * 16 + ib * XParam.blksize];
							h21 = hho[cx + fy * 16 + ib * XParam.blksize];
							h12 = hho[fx + cy * 16 + ib * XParam.blksize];
							h22 = hho[cx + cy * 16 + ib * XParam.blksize];

							zs11 = zso[fx + fy * 16 + ib * XParam.blksize];
							zs21 = zso[cx + fy * 16 + ib * XParam.blksize];
							zs12 = zso[fx + cy * 16 + ib * XParam.blksize];
							zs22 = zso[cx + cy * 16 + ib * XParam.blksize];

							u11 = uuo[fx + fy * 16 + ib * XParam.blksize];
							u21 = uuo[cx + fy * 16 + ib * XParam.blksize];
							u12 = uuo[fx + cy * 16 + ib * XParam.blksize];
							u22 = uuo[cx + cy * 16 + ib * XParam.blksize];

							v11 = vvo[fx + fy * 16 + ib * XParam.blksize];
							v21 = vvo[cx + fy * 16 + ib * XParam.blksize];
							v12 = vvo[fx + cy * 16 + ib * XParam.blksize];
							v22 = vvo[cx + cy * 16 + ib * XParam.blksize];

							


							if (cy == 0)
							{
								h11 = BotAda(fx, cy, ib, botblk[ib], botblk[rightblk[ib]], botblk[leftblk[ib]], leftblk[botblk[ib]], rightblk[botblk[ib]], level, hho);
								h12 = BotAda(cx, cy, ib, botblk[ib], botblk[rightblk[ib]], botblk[leftblk[ib]], leftblk[botblk[ib]], rightblk[botblk[ib]], level, hho);
								
								zs11 = BotAda(fx, cy, ib, botblk[ib], botblk[rightblk[ib]], botblk[leftblk[ib]], leftblk[botblk[ib]], rightblk[botblk[ib]], level, zso);
								zs12 = BotAda(cx, cy, ib, botblk[ib], botblk[rightblk[ib]], botblk[leftblk[ib]], leftblk[botblk[ib]], rightblk[botblk[ib]], level, zso);

								u11 = BotAda(fx, cy, ib, botblk[ib], botblk[rightblk[ib]], botblk[leftblk[ib]], leftblk[botblk[ib]], rightblk[botblk[ib]], level, uuo);
								u12 = BotAda(cx, cy, ib, botblk[ib], botblk[rightblk[ib]], botblk[leftblk[ib]], leftblk[botblk[ib]], rightblk[botblk[ib]], level, uuo);

								v11 = BotAda(fx, cy, ib, botblk[ib], botblk[rightblk[ib]], botblk[leftblk[ib]], leftblk[botblk[ib]], rightblk[botblk[ib]], level, vvo);
								v12 = BotAda(cx, cy, ib, botblk[ib], botblk[rightblk[ib]], botblk[leftblk[ib]], leftblk[botblk[ib]], rightblk[botblk[ib]], level, vvo);
							
								
							}

							if (fy >= 15)
							{
								h12 = TopAda(fx, fy, ib, topblk[ib], topblk[rightblk[ib]], topblk[leftblk[ib]], leftblk[topblk[ib]], rightblk[topblk[ib]], level, hho);
								h22 = TopAda(cx, fy, ib, topblk[ib], topblk[rightblk[ib]], topblk[leftblk[ib]], leftblk[topblk[ib]], rightblk[topblk[ib]], level, hho);
								
								zs12 = TopAda(fx, fy, ib, topblk[ib], topblk[rightblk[ib]], topblk[leftblk[ib]], leftblk[topblk[ib]], rightblk[topblk[ib]], level, zso);
								zs22 = TopAda(cx, fy, ib, topblk[ib], topblk[rightblk[ib]], topblk[leftblk[ib]], leftblk[topblk[ib]], rightblk[topblk[ib]], level, zso);

								u12 = TopAda(fx, fy, ib, topblk[ib], topblk[rightblk[ib]], topblk[leftblk[ib]], leftblk[topblk[ib]], rightblk[topblk[ib]], level, uuo);
								u22 = TopAda(cx, fy, ib, topblk[ib], topblk[rightblk[ib]], topblk[leftblk[ib]], leftblk[topblk[ib]], rightblk[topblk[ib]], level, uuo);

								v12 = TopAda(fx, fy, ib, topblk[ib], topblk[rightblk[ib]], topblk[leftblk[ib]], leftblk[topblk[ib]], rightblk[topblk[ib]], level, vvo);
								v22 = TopAda(cx, fy, ib, topblk[ib], topblk[rightblk[ib]], topblk[leftblk[ib]], leftblk[topblk[ib]], rightblk[topblk[ib]], level, vvo);

								
								
							}

							if (cx == 0)
							{
								h12 = LeftAda(cx, cy, ib, leftblk[ib], leftblk[botblk[ib]], leftblk[topblk[ib]], botblk[leftblk[ib]], topblk[leftblk[ib]], level, hho);
								h11 = LeftAda(cx, fy, ib, leftblk[ib], leftblk[botblk[ib]], leftblk[topblk[ib]], botblk[leftblk[ib]], topblk[leftblk[ib]], level, hho);
								
								zs12 = LeftAda(cx, cy, ib, leftblk[ib], leftblk[botblk[ib]], leftblk[topblk[ib]], botblk[leftblk[ib]], topblk[leftblk[ib]], level, zso);
								zs11 = LeftAda(cx, fy, ib, leftblk[ib], leftblk[botblk[ib]], leftblk[topblk[ib]], botblk[leftblk[ib]], topblk[leftblk[ib]], level, zso);

								u12 = LeftAda(cx, cy, ib, leftblk[ib], leftblk[botblk[ib]], leftblk[topblk[ib]], botblk[leftblk[ib]], topblk[leftblk[ib]], level, uuo);
								u11 = LeftAda(cx, fy, ib, leftblk[ib], leftblk[botblk[ib]], leftblk[topblk[ib]], botblk[leftblk[ib]], topblk[leftblk[ib]], level, uuo);

								v12 = LeftAda(cx, cy, ib, leftblk[ib], leftblk[botblk[ib]], leftblk[topblk[ib]], botblk[leftblk[ib]], topblk[leftblk[ib]], level, vvo);
								v11 = LeftAda(cx, fy, ib, leftblk[ib], leftblk[botblk[ib]], leftblk[topblk[ib]], botblk[leftblk[ib]], topblk[leftblk[ib]], level, vvo);
							
								
							}

							if (fx >= 15)
							{
								h22 = RightAda(fx, cy, ib, rightblk[ib], rightblk[botblk[ib]], rightblk[topblk[ib]], botblk[rightblk[ib]], topblk[rightblk[ib]], level, hho);
								h21 = RightAda(fx, fy, ib, rightblk[ib], rightblk[botblk[ib]], rightblk[topblk[ib]], botblk[rightblk[ib]], topblk[rightblk[ib]], level, hho);
								
								zs22 = RightAda(fx, cy, ib, rightblk[ib], rightblk[botblk[ib]], rightblk[topblk[ib]], botblk[rightblk[ib]], topblk[rightblk[ib]], level, zso);
								zs21 = RightAda(fx, fy, ib, rightblk[ib], rightblk[botblk[ib]], rightblk[topblk[ib]], botblk[rightblk[ib]], topblk[rightblk[ib]], level, zso);

								u22 = RightAda(fx, cy, ib, rightblk[ib], rightblk[botblk[ib]], rightblk[topblk[ib]], botblk[rightblk[ib]], topblk[rightblk[ib]], level, uuo);
								u21 = RightAda(fx, fy, ib, rightblk[ib], rightblk[botblk[ib]], rightblk[topblk[ib]], botblk[rightblk[ib]], topblk[rightblk[ib]], level, uuo);

								v22 = RightAda(fx, cy, ib, rightblk[ib], rightblk[botblk[ib]], rightblk[topblk[ib]], botblk[rightblk[ib]], topblk[rightblk[ib]], level, vvo);
								v21 = RightAda(fx, fy, ib, rightblk[ib], rightblk[botblk[ib]], rightblk[topblk[ib]], botblk[rightblk[ib]], topblk[rightblk[ib]], level, vvo);

								
								
							}



							if (cy == 0)
							{
								fy = -1;
							}
							if (fy >= 15)
							{
								cy = 16;
							}
							if (cx == 0)
							{
								fx = -1;
							}
							if (fx >= 15)
							{
								cx = 16;
							}

							//printf("fx = %d; cx=%d; fy=%d; cy=%d; rx=%f; ry=%f\n", fx, cx, fy, cy,rx,ry);

							//printf("First blk %f\n",BilinearInterpolation(h11, h12, h21, h22, fx, cx, fy, cy, rx, ry));
							
							hh[o] = BilinearInterpolation(h11, h12, h21, h22, (double)fx, (double)cx, (double)fy, (double)cy, rx, ry);
							zs[o] = BilinearInterpolation(zs11, zs12, zs21, zs22, (double)fx, (double)cx, (double)fy, (double)cy, rx, ry);
							uu[o] = BilinearInterpolation(u11, u12, u21, u22, (double)fx, (double)cx, (double)fy, (double)cy, rx, ry);
							vv[o] = BilinearInterpolation(v11, v12, v21, v22, (double)fx, (double)cx, (double)fy, (double)cy, rx, ry);


						}

					}
				}
				


				// sort out block info

				newlevel[ib] = level[ib] + 1;
				newlevel[availblk[csumblk[ib]]] = level[ib] + 1;
				newlevel[availblk[csumblk[ib] + 1]] = level[ib] + 1;
				newlevel[availblk[csumblk[ib] + 2]] = level[ib] + 1;


				blockxo_d[ib] = xoblk;
				blockyo_d[ib] = yoblk;

				//bottom right blk
				blockxo_d[availblk[csumblk[ib]]] = xoblk + (XParam.blkwidth) * delx;
				blockyo_d[availblk[csumblk[ib]]] = yoblk;
				//top left blk
				blockxo_d[availblk[csumblk[ib] + 1]] = xoblk;
				blockyo_d[availblk[csumblk[ib] + 1]] = yoblk + (XParam.blkwidth) * delx;
				//top right blk
				blockxo_d[availblk[csumblk[ib] + 2]] = xoblk + (XParam.blkwidth) * delx;
				blockyo_d[availblk[csumblk[ib] + 2]] = yoblk + (XParam.blkwidth ) * delx;


				//sort out blocks neighbour

				topblk[ib] = availblk[csumblk[ib] + 1];
				topblk[availblk[csumblk[ib] + 1]] = oldtop;
				topblk[availblk[csumblk[ib]]] = availblk[csumblk[ib] + 2];

				//printf("topblk[ib]=%d; oldtop=%d; topblk[availblk[csumblk[ibl] + 1]]=%d; topblk[availblk[csumblk[ibl]]]=%d; \n ", availblk[csumblk[ibl] + 1], oldtop, topblk[availblk[csumblk[ibl] + 1]], topblk[availblk[csumblk[ibl]]]);

				rightblk[ib] = availblk[csumblk[ib]];
				rightblk[availblk[csumblk[ib] + 1]] = availblk[csumblk[ib] + 2];
				rightblk[availblk[csumblk[ib]]] = oldright;

				botblk[availblk[csumblk[ib] + 1]] = ib;
				botblk[availblk[csumblk[ib] + 2]] = availblk[csumblk[ib]];

				leftblk[availblk[csumblk[ib]]] = ib;
				leftblk[availblk[csumblk[ib] + 2]] = availblk[csumblk[ib] + 1];

				XParam.navailblk= XParam.navailblk-3;
			}
		}

	}

	// deal with the neighbour

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
				if (newlevel[oldtop] <  newlevel[ib])
				{
					topblk[availblk[csumblk[ib] + 2]] = oldtop;
					//botblk[oldtop] = availblk[csumblk[ib] + 2];
				}
				else
				{
					topblk[availblk[csumblk[ib] + 2]] = rightblk[oldtop];
					
				}


				/////
				if ( newlevel[oldright] <  newlevel[ib])
				{
					rightblk[availblk[csumblk[ib] + 2]] = oldright;
					
				}
				else
				{
					rightblk[availblk[csumblk[ib] + 2]] = topblk[oldright];
				}

				/////
				if (newlevel[oldleft] < newlevel[ib])
				{
					leftblk[availblk[csumblk[ib] + 1]] = oldleft;
				}
				else
				{
					leftblk[availblk[csumblk[ib] + 1]] = topblk[oldleft];
				}

				/////
				if ( newlevel[oldbot] <  newlevel[ib])
				{
					botblk[availblk[csumblk[ib]]] = oldbot;
				}
				else
				{
					botblk[availblk[csumblk[ib]]] = rightblk[oldbot];
				}

				//printf("level=%d\n", level[ib]);
				//newlevel[availblk[csumblk[ibl]]] = level[ib];
				//newlevel[availblk[csumblk[ibl]+1]] = level[ib];
				//newlevel[availblk[csumblk[ibl]+2]] = level[ib];


				//Below is only not valid if the block above/right adapts and is sequentially before the present block.

				botblk[oldtop] = availblk[csumblk[ib] + 1];
				leftblk[oldright] = availblk[csumblk[ib]];

				

				activeblk[nblk] = availblk[csumblk[ib]];
				activeblk[nblk + 1] = availblk[csumblk[ib] + 1];
				activeblk[nblk + 2] = availblk[csumblk[ib] + 2];

				//printf("ib=%d; ib_right=%d; ib_top=%d; ib_TR=%d\n", ib, activeblk[nblk], activeblk[nblk + 1], activeblk[nblk + 2]);


				nblk = nblk + 3;
			}
		}
	}

	//update level
	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		//
		int oldlevel;
		int ib = activeblk[ibl];

		//printf("ib=%d; l=%d; xo=%f; yo=%f\n", ib, level[ib], blockxo_d[ib], blockyo_d[ib]);
		
		if (ib >= 0) // ib can be -1 for newly inactive blocks
		{

			
			level[ib] = newlevel[ib];
			
			//printf("ib=%d; oldlevel=%d; newlevel[ib]=%d; l=%d;  block_xo=%f; block_yo=%f\n", ib, oldlevel, newlevel[ib], level[ib], blockxo_d[ib], blockyo_d[ib]);
			
		}
	}

	// Reorder activeblk
	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		//reuse newlevel as temporary storage for activeblk
		newlevel[ibl] = activeblk[ibl];
		activeblk[ibl] = -1;
	
	//printf("ibl=%d; activeblk[ibl]=%d; newlevel[ibl]=%d;\n", ibl, activeblk[ibl], newlevel[ibl]);
	}


	// cleanup and Reorder active block list
	int ib = 0;
	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		
		if (newlevel[ibl] != -1)//i.e. old activeblk
		{
			activeblk[ib] = newlevel[ibl];
			//printf("ib=%d; l=%d; xo=%f; yo=%f\n", activeblk[ib], level[activeblk[ib]], blockxo_d[activeblk[ib]], blockyo_d[activeblk[ib]]);
			ib++;
		}
	}

	//printf("ib=%d; nblk=%d; XParam.nblk=%d\n", ib, nblk, XParam.nblk);

	nblk = ib;



	//reset blockxo and blockyo
	
	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		blockxo[ibl] = blockxo_d[ibl];
		blockyo[ibl] = blockyo_d[ibl];
		newlevel[ibl] = 0;
		refine[ibl] = false;
		coarsen[ibl] = false;
	}

	

	

	//interp2BUQ(XParam.nblk, XParam.blksize, levdx, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, bathydata, zb);
	interp2BUQAda(nblk, XParam.blksize, XParam.dx, activeblk, level, blockxo_d, blockyo_d, XParam.Bathymetry.nx, XParam.Bathymetry.ny, XParam.Bathymetry.xo, XParam.Bathymetry.xmax, XParam.Bathymetry.yo, XParam.Bathymetry.ymax, XParam.Bathymetry.dx, bathydata, zb);

	// Because zb cannot be conserved through the refinement or coarsening
	// We have o decide whtether to conserve elevation (zs) or Volume (hh)
	// 

	for (int ibl = 0; ibl < nblk; ibl++)
	{
		int ib = activeblk[ibl];
		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				int i = ix + iy * 16 + ib * XParam.blksize;
				 
				hh[i] = max((float)XParam.eps, zs[i] - zb[i]);

			}
		}
	}

	// Copy basic info to hho zso uuo vvo for further iterations
	CopyArray(nblk, XParam.blksize, hh, hho);
	CopyArray(nblk, XParam.blksize, zs, zso);
	CopyArray(nblk, XParam.blksize, uu, uuo);
	CopyArray(nblk, XParam.blksize, vv, vvo);



	// Update nblk (nblk is the new number of block XParam.nblk is the previous number of blk)
	XParam.nblk = nblk;


	return XParam;
}









#include "Adaptation.h"



template <class T> void Adaptation(Param& XParam, Forcing<float> XForcing, Model<T>& XModel)
{
	int oldnblk = 0;

	int niteration = 0;

	int maxiteration = 20;
	//fillHalo(XParam, XModel.blocks, XModel.evolv_o);
	//fillCorners(XParam, XModel.blocks, XModel.evolv_o);
	if (XParam.maxlevel != XParam.minlevel)
	{
		while (oldnblk != XParam.nblk && niteration< maxiteration)
		{
			niteration++;
			log("\t Iteration " + std::to_string(niteration));
			// Fill halo and corners
			fillHalo(XParam, XModel.blocks, XModel.evolv_o);
			fillCorners(XParam, XModel.blocks, XModel.evolv_o);
			

			oldnblk = XParam.nblk;
			//wetdrycriteria(XParam, refine, coarsen);
			//inrangecriteria(XParam, (T)-5.2, (T)0.2, XModel.zb, XModel.blocks, XModel.adapt.refine, XModel.adapt.coarsen);
			AdaptCriteria(XParam, XForcing, XModel);
			refinesanitycheck(XParam, XModel.blocks, XModel.adapt.refine, XModel.adapt.coarsen);
			//XParam = adapt(XParam);
			Adapt(XParam, XForcing, XModel);


			if (!checkBUQsanity(XParam,XModel.blocks))
			{

				XParam.outfile = "Bad_mesh.nc";
				log("\tERROR!!!  Bad BUQ mesh layout! See file: "+ XParam.outfile);
				copyID2var(XParam, XModel.blocks, XModel.flux.Fhu);
				copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.LeftBot, XModel.grad.dhdx);
				copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.LeftTop, XModel.grad.dhdy);
				copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.TopLeft, XModel.grad.dzsdx);
				copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.TopRight, XModel.grad.dzsdy);
				copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.RightTop, XModel.grad.dudx);
				copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.RightBot, XModel.grad.dudy);
				copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.BotRight, XModel.grad.dvdx);
				copyBlockinfo2var(XParam, XModel.blocks, XModel.blocks.BotLeft, XModel.grad.dvdy);

				creatncfileBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo);
				defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "blockID", 3, XModel.flux.Fhu);
				
				defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "LeftBot", 3, XModel.grad.dhdx);
				defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "LeftTop", 3, XModel.grad.dhdy);

				defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "TopLeft", 3, XModel.grad.dzsdx);
				defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "TopRight", 3, XModel.grad.dzsdy);
				
				defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "RightTop", 3, XModel.grad.dudx);
				defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "RightBot", 3, XModel.grad.dudy);

				defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "BotLeft", 3, XModel.grad.dvdx);
				defncvarBUQ(XParam, XModel.blocks.active, XModel.blocks.level, XModel.blocks.xo, XModel.blocks.yo, "BotRight", 3, XModel.grad.dvdy);
				exit(2);
				break;
			}


		}
		//=====================================
		// Initialise Friction map

		if (!XForcing.cf.inputfile.empty())
		{
			interp2BUQ(XParam, XModel.blocks, XForcing.cf, XModel.cf);
		}
		else
		{
			InitArrayBUQ(XParam, XModel.blocks, (T)XParam.cf, XModel.cf);
		}
		// Set edges of friction map
		setedges(XParam, XModel.blocks, XModel.cf);

	}
}
template void Adaptation<float>(Param& XParam, Forcing<float> XForcing, Model<float>& XModel);
template void Adaptation<double>(Param& XParam, Forcing<float> XForcing, Model<double>& XModel);

//Initial adaptation also reruns initial conditions
template <class T> void InitialAdaptation(Param& XParam, Forcing<float> &XForcing, Model<T>& XModel)
{
	if (XParam.maxlevel != XParam.minlevel)
	{
		log("Adapting mesh");
		Adaptation(XParam, XForcing, XModel);

		//recalculate river positions (They may belong to a different block)
		InitRivers(XParam, XForcing, XModel);

		//rerun boundary block (there may be new bnd block and old ones that do not belong anymore)
		//Initbnds(XParam, XForcing, XModel);
		Calcbndblks(XParam, XForcing, XModel.blocks);
		Findbndblks(XParam, XModel, XForcing);

		//Recalculate the masks
		FindMaskblk(XParam, XModel.blocks);

		// Re run initial contions to avoid dry/wet issues
		initevolv(XParam, XModel.blocks, XForcing, XModel.evolv, XModel.zb);
	}
}
template void InitialAdaptation<float>(Param& XParam, Forcing<float> &XForcing, Model<float>& XModel);
template void InitialAdaptation<double>(Param& XParam, Forcing<float> &XForcing, Model<double>& XModel);


/*! \fn bool refinesanitycheck(Param XParam, bool*& refine, bool*& coarsen)
* check and correct the sanity of first order refining/corasening criteria.
*
*
*
*/
template <class T> bool refinesanitycheck(Param XParam, BlockP<T> XBlock,  bool*& refine, bool*& coarsen)
{
	// Can't actually refine if the level is the max level (i.e. finest)
	// this may be over-ruled later on
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		if (refine[ib] == true && XBlock.level[ib] == XParam.maxlevel)
		{
			refine[ib] = false;
			//printf("ib=%d; level[ib]=%d\n", ib, level[ib]);
		}
		if (coarsen[ib] == true && XBlock.level[ib] == XParam.minlevel)
		{
			coarsen[ib] = false;
		}
	}


	// Can't corasen if any of your direct neighbour refines
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		if (refine[ib] == true)
		{
			//Can probably get away with checking only the principal 4 ?
			coarsen[XBlock.RightBot[ib]] = false;
			coarsen[XBlock.RightTop[ib]] = false;
			coarsen[XBlock.LeftBot[ib]] = false;
			coarsen[XBlock.LeftTop[ib]] = false;
			coarsen[XBlock.TopLeft[ib]] = false;
			coarsen[XBlock.TopRight[ib]] = false;
			coarsen[XBlock.BotLeft[ib]] = false;
			coarsen[XBlock.BotRight[ib]] = false;
		}
	}

	// Can't coarsen if any neighbours have a higher level
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		if (coarsen[ib] == true)
		{
			int levi = XBlock.level[ib];
			//printf("ib=%d; leftblk[ib]=%d; rightblk[ib]=%d, topblk[ib]=%d, botblk[ib]=%d\n", ib, leftblk[ib], rightblk[ib], topblk[ib], botblk[ib]);
			if (levi < XBlock.level[XBlock.LeftBot[ib]] ||  levi < XBlock.level[XBlock.RightBot[ib]] || levi < XBlock.level[XBlock.TopLeft[ib]] || levi < XBlock.level[XBlock.BotLeft[ib]])
			{
				coarsen[ib] = false;
			}
		}
	}


	//check whether neighbour need refinement because they are too coarse to allow one to refine
	// This below could be cascading so need to iterate several time
	int iter = 1;

	while (iter > 0)
	{
		iter = 0;
		


		for (int ibl = 0; ibl < XParam.nblk; ibl++)
		{
			int ib = XBlock.active[ibl];
			

			if (refine[ib] == true)
			{
				iter += checkneighbourrefine(XBlock.TopLeft[ib], XBlock.level[ib], XBlock.level[XBlock.TopLeft[ib]], refine, coarsen);
				//iter += checkneighbourrefine(XBlock.TopRight[ib], XBlock.level[ib], XBlock.level[XBlock.TopRight[ib]], refine, coarsen);
				iter += checkneighbourrefine(XBlock.BotLeft[ib], XBlock.level[ib], XBlock.level[XBlock.BotLeft[ib]], refine, coarsen);
				//iter += checkneighbourrefine(XBlock.BotRight[ib], XBlock.level[ib], XBlock.level[XBlock.BotRight[ib]], refine, coarsen);
				iter += checkneighbourrefine(XBlock.LeftBot[ib], XBlock.level[ib], XBlock.level[XBlock.LeftBot[ib]], refine, coarsen);
				//iter += checkneighbourrefine(XBlock.LeftTop[ib], XBlock.level[ib], XBlock.level[XBlock.LeftTop[ib]], refine, coarsen);
				iter += checkneighbourrefine(XBlock.RightBot[ib], XBlock.level[ib], XBlock.level[XBlock.RightBot[ib]], refine, coarsen);
				//iter += checkneighbourrefine(XBlock.RightTop[ib], XBlock.level[ib], XBlock.level[XBlock.RightTop[ib]], refine, coarsen);

			}
			
		}
	}




	// Can't actually coarsen if top, right and topright block are not all corsen
	
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];

		//printf("ib=%d\n", ib);
		// if all the neighbour are not wet then coarsen if possible
		double dxfac = calcres(XParam.dx, XBlock.level[ib]);
		//printf("blockxo_d[ib]=%f, dxfac=%f, ((blx-xo)/dx)%2=%d\n", blockxo_d[ib], dxfac, (int((blockxo_d[ib] - XParam.xo) / dxfac / XParam.blkwidth) % 2));
		//only check for coarsening if the block analysed is a lower left corner block of the lower level
		//need to prevent coarsenning if the block is on the model edges...
		//((int((blockxo_d[ib] - XParam.xo) / dxfac) % 2) == 0 && (int((blockyo_d[ib] - XParam.yo) / dxfac) % 2) == 0) && rightblk[ib] != ib && topblk[ib] != ib && rightblk[topblk[ib]] != topblk[ib]
		if (coarsen[ib] == true)
		{
			//if this block is a lower left corner block of teh potentialy coarser block
			if (((int((XBlock.xo[ib]) / dxfac / XParam.blkwidth) % 2) == 0 && (int((XBlock.yo[ib]) / dxfac / XParam.blkwidth) % 2) == 0 && XBlock.RightBot[ib] != ib &&  XBlock.TopLeft[ib] != ib && XBlock.RightBot[XBlock.TopRight[ib]] != XBlock.TopRight[ib]))
			{
				//if all the neighbour blocks ar at the same level
				if (XBlock.level[ib] == XBlock.level[XBlock.RightBot[ib]] && XBlock.level[ib] == XBlock.level[XBlock.TopLeft[ib]] && XBlock.level[ib] == XBlock.level[XBlock.RightBot[XBlock.TopRight[ib]]])
				{
					//printf("Is it true?\t");
					//if right, top and topright block teh same level and can coarsen
					if (coarsen[XBlock.RightBot[ib]] == true && coarsen[XBlock.TopLeft[ib]] == true && coarsen[XBlock.RightBot[XBlock.TopRight[ib]]] == true)
					{
						//Yes
						//printf("Yes!\n");
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
	return true;
}


int checkneighbourrefine(int neighbourib,int levelib, int levelneighbour, bool*& refine, bool*& coarsen)
{
	int iter = 0;
	if (refine[neighbourib] == false && (levelneighbour < levelib))
	{
		refine[neighbourib] = true;
		coarsen[neighbourib] = false;
		iter++;
	}
	if (levelneighbour == levelib)
	{
		coarsen [neighbourib]= false;
	}
	return iter;
}

/*! \fn bool checkBUQsanity(Param XParam)
* Check the sanity of the BUQ mesh
* This function mostly checks the level of neighbouring blocks
*
*	Needs improvements
*/
template <class T> bool checkBUQsanity(Param XParam,BlockP<T> XBlock)
{
	bool check = true;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		int nib;
		// check that levels are consistent
		check = check && checklevel(ib, XBlock.level[ib], XBlock.LeftBot[ib], XBlock.level[XBlock.LeftBot[ib]]);
		check = check && checklevel(ib, XBlock.level[ib], XBlock.LeftTop[ib], XBlock.level[XBlock.LeftTop[ib]]);
		
		check = check && checklevel(ib, XBlock.level[ib], XBlock.TopLeft[ib], XBlock.level[XBlock.TopLeft[ib]]);
		check = check && checklevel(ib, XBlock.level[ib], XBlock.TopRight[ib], XBlock.level[XBlock.TopRight[ib]]);
		
		check = check && checklevel(ib, XBlock.level[ib], XBlock.RightTop[ib], XBlock.level[XBlock.RightTop[ib]]);
		check = check && checklevel(ib, XBlock.level[ib], XBlock.RightBot[ib], XBlock.level[XBlock.RightBot[ib]]);

		check = check && checklevel(ib, XBlock.level[ib], XBlock.BotRight[ib], XBlock.level[XBlock.BotRight[ib]]);
		check = check && checklevel(ib, XBlock.level[ib], XBlock.BotLeft[ib], XBlock.level[XBlock.BotLeft[ib]]);

		//check that neighbour distance makes sense with level
		nib = XBlock.LeftBot[ib];
		check = check && checkneighbourdistance(XParam.dx, ib, XBlock.level[ib], XBlock.xo[ib], nib, XBlock.level[nib], XBlock.xo[nib], false);
		nib = XBlock.LeftTop[ib];
		check = check && checkneighbourdistance(XParam.dx, ib, XBlock.level[ib], XBlock.xo[ib], nib, XBlock.level[nib], XBlock.xo[nib], false);

		nib = XBlock.RightTop[ib];
		check = check && checkneighbourdistance(XParam.dx, ib, XBlock.level[ib], XBlock.xo[ib], nib, XBlock.level[nib], XBlock.xo[nib], true);
		nib = XBlock.RightBot[ib];
		check = check && checkneighbourdistance(XParam.dx, ib, XBlock.level[ib], XBlock.xo[ib], nib, XBlock.level[nib], XBlock.xo[nib], true);

		nib = XBlock.TopRight[ib];
		check = check && checkneighbourdistance(XParam.dx, ib, XBlock.level[ib], XBlock.yo[ib], nib, XBlock.level[nib], XBlock.yo[nib], true);
		nib = XBlock.TopLeft[ib];
		check = check && checkneighbourdistance(XParam.dx, ib, XBlock.level[ib], XBlock.yo[ib], nib, XBlock.level[nib], XBlock.yo[nib], true);

		nib = XBlock.BotLeft[ib];
		check = check && checkneighbourdistance(XParam.dx, ib, XBlock.level[ib], XBlock.yo[ib], nib, XBlock.level[nib], XBlock.yo[nib], false);
		nib = XBlock.BotRight[ib];
		check = check && checkneighbourdistance(XParam.dx, ib, XBlock.level[ib], XBlock.yo[ib], nib, XBlock.level[nib], XBlock.yo[nib], false);
	}

	return check;

}
template bool checkBUQsanity<float>(Param XParam, BlockP<float> XBlock);
template bool checkBUQsanity<double>(Param XParam, BlockP<double> XBlock);

bool checklevel(int ib, int levelib, int neighbourib, int levelneighbour)
{
	bool check = true;
	if (abs(levelneighbour - (levelib)) > 1)
	{
		log("Warning! Bad Neighbour Level. ib="+std::to_string(ib)+"; level[ib]="+ std::to_string(levelib)+"; neighbour[ib]="+ std::to_string(neighbourib) +"; level[neighbour[ib]]="+ std::to_string(levelneighbour));
		check = false;
	}
	return check;
}

template <class T> bool checkneighbourdistance(double dx, int ib, int levelib, T blocko, int neighbourib, int levelneighbour, T neighbourblocko, bool rightortop )
{
	T expecteddistance= blocko;
	bool test;
	if (neighbourib != ib)
	{
		if (rightortop)
		{
			expecteddistance = blocko + calcres(T(dx), levelib) * 15.5 + 0.5 * calcres(T(dx), levelneighbour);
		}
		else
		{
			expecteddistance = blocko - calcres(T(dx), levelib) * 0.5 - 15.5 * calcres(T(dx), levelneighbour);
		}

	}

	test= abs(expecteddistance - neighbourblocko) < (calcres(T(dx), levelib) * 0.01);
	if (!test)
	{
		log("Warning! Bad Neighbour distance. ib=" + std::to_string(ib) + "; level[ib]=" + std::to_string(levelib) + "; neighbour[ib]=" + std::to_string(neighbourib) + "; level[neighbour[ib]]=" + std::to_string(levelneighbour));
	}

	return test;

}



template <class T> void Adapt(Param &XParam, Forcing<float> XForcing, Model<T>& XModel)
{
	int nnewblk = CalcAvailblk(XParam, XModel.blocks, XModel.adapt);

	// Check if there are enough available block to refin 
	if (nnewblk > XParam.navailblk)
	{
		//Reallocate
		int nblkmem=AddBlocks(nnewblk, XParam, XModel);

		log("\t\tReallocation complete: "+std::to_string(XParam.navailblk)+" new blocks are available ( "+std::to_string(nblkmem)+" blocks in memory) ");
	}
	//===========================================================
	//	Start coarsening and refinement
	// First Initialise newlevel (Do this every time because new level is reused later)

	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		// Set newlevel
		XModel.adapt.newlevel[ibl] = XModel.blocks.level[ibl];
	}

	//=========================================================
	//	COARSEN
	coarsen(XParam, XModel.blocks, XModel.adapt, XModel.evolv_o, XModel.evolv);

	//=========================================================
	//	REFINE
	refine(XParam, XModel.blocks, XModel.adapt, XModel.evolv_o, XModel.evolv);

	//=========================================================
	// CLEAN-UP
	Adaptationcleanup(XParam, XModel.blocks, XModel.adapt);

	//____________________________________________________
	//
	//	Reinterpolate zb. 
	//
	//	Isn't it better to do that only for newly refined blk?
	//  Not necessary if no coarsening/refinement occur
	interp2BUQ(XParam, XModel.blocks, XForcing.Bathy, XModel.zb);

	// Set edges
	setedges(XParam, XModel.blocks, XModel.zb);

	//____________________________________________________
	//
	//	Update hh and or zb
	//
	//	Recalculate hh from zs for fully wet cells and zs from zb for dry cells
	//

	// Because zb cannot be conserved through the refinement or coarsening
	// We have to decide whtether to conserve elevation (zs) or Volume (hh)
	// 
	
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		for (int iy = 0; iy < XParam.blkwidth; iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(XParam,ix,iy,ib);

				if (XModel.evolv.h[i] > XParam.eps)
				{
					XModel.evolv.h[i] = max((T)XParam.eps, XModel.evolv.zs[i] - XModel.zb[i]);
				}
				else
				{
					// when refining dry area zs should be zb!
					XModel.evolv.zs[i] = XModel.zb[i];
				}



			}
		}
	}
	
	//copy back hh and zs to hho and zso
	CopyArrayBUQ(XParam, XModel.blocks, XModel.evolv, XModel.evolv_o);
	
}

/*! \fn int CalcAvailblk(Param XParam, BlockP<T> XBlock, AdaptP& XAdapt)
* 
*
*
*
*/
template <class T> int CalcAvailblk(Param &XParam, BlockP<T> XBlock, AdaptP& XAdapt)
{
	//

	int csum = -3;
	int nrefineblk = 0;
	int ncoarsenlk = 0;
	int nnewblk = 0;

	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		XAdapt.invactive[ibl] = -1;


	}
	
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		XAdapt.invactive[ib] = ibl;

		// When refining we need csum
		if (XAdapt.refine[ib] == true)
		{
			nrefineblk++;
			csum = csum + 3;

		}
		if (XAdapt.coarsen[ib] == true)
		{
			ncoarsenlk++;


		}
		XAdapt.csumblk[ib] = csum;
	}
	
	//=========================================
	//	Reconstruct availblk
	XParam.navailblk = 0;
	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		if (XAdapt.invactive[ibl] == -1)
		{
			XAdapt.availblk[XParam.navailblk] = ibl;
			XParam.navailblk++;
		}

	}

	// How many new block are needed
	// This below would be ideal but I don't see how that could work.
	// One issue is to make the newly coarsen blocks directly available in the section above but that would make the code even more confusingalthough we haven't taken them into account in the 
	//nnewblk = 3*nrefineblk - ncoarsenlk*3;
	// Below is conservative and keeps the peice of code above a bit more simple
	nnewblk = 3 * nrefineblk;

	log("\t\tThere are "+ std::to_string(XParam.nblk) +" active blocks ("+ std::to_string(XParam.nblkmem) +" blocks allocated in memory), "+std::to_string(nrefineblk)+" blocks to be refined, "+std::to_string(ncoarsenlk)+" blocks to be coarsen (with neighbour); "+std::to_string(XParam.nblk - nrefineblk - 4 * ncoarsenlk)+" blocks untouched; "+std::to_string(ncoarsenlk * 3)+" blocks to be freed ("+ std::to_string(XParam.navailblk) +" are already available) "+std::to_string(nnewblk)+" new blocks will be created");

	return nnewblk;

}
template int CalcAvailblk<float>(Param &XParam, BlockP<float> XBlock, AdaptP& XAdapt);
template int CalcAvailblk<double>(Param &XParam, BlockP<double> XBlock, AdaptP& XAdapt);

template <class T> int AddBlocks(int nnewblk, Param& XParam, Model<T>& XModel)
{
	//
	int nblkmem, oldblkmem;
	oldblkmem = XParam.nblkmem;
	nblkmem = (int)ceil((XParam.nblk + nnewblk) * XParam.membuffer);
	XParam.nblkmem = nblkmem;
	ReallocArray(nblkmem, XParam.blksize, XParam, XModel);


	// Reconstruct blk info
	XParam.navailblk = 0;
	for (int ibl = 0; ibl < (XParam.nblkmem - XParam.nblk); ibl++)
	{
		XModel.blocks.active[XParam.nblk + ibl] = -1;
	}

	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		int ib = XModel.blocks.active[ibl];
		/*
		if (ib == -1)
		{
			XModel.adapt.coarsen[ib] = false;
			XModel.adapt.refine[ib] = false;
		}
		*/

		//printf("ibl=%d; availblk[ibl]=%d;\n",ibl, availblk[ibl]);

	}

	for (int ibl = 0; ibl < (XParam.nblkmem - oldblkmem); ibl++)
	{
		XModel.adapt.invactive[oldblkmem + ibl] = -1;


	}

	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		if (XModel.adapt.invactive[ibl] == -1)
		{
			XModel.adapt.availblk[XParam.navailblk] = ibl;
			XParam.navailblk++;
		}

	}

	//Because reallocation may be producing different pointers we need to update the output map array

	Initmaparray(XModel);
	return nblkmem;
}
template int AddBlocks<float>(int nnewblk, Param& XParam, Model<float>& XModel);
template int AddBlocks<double>(int nnewblk, Param& XParam, Model<double>& XModel);


template <class T> void coarsen(Param XParam, BlockP<T>& XBlock, AdaptP& XAdapt,EvolvingP<T> XEvo, EvolvingP<T>& XEv )
{
	//=========================================================
	//	COARSEN
	//=========================================================
	// This is a 2 step process
	// 1. First deal with the conserved variables (hh,uu,vv,zs,zb)
	// 2. Deactivate the block
	// 3. Fix neighbours

	//____________________________________________________
	//
	// Step 1 & 2: Average conserved variables and deactivate the blocks


	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		int i, ii, ir, it, itr;
		if (XAdapt.coarsen[ib] == true)
		{
			double dxfac = calcres(XParam.dx, XBlock.level[ib]);
			int xnode = int((XBlock.xo[ib]) / dxfac / XParam.blkwidth);
			int ynode = int((XBlock.yo[ib]) / dxfac / XParam.blkwidth);

			int ibr = XBlock.RightBot[ib];
			int ibtl = XBlock.TopLeft[ib];
			int ibtr = XBlock.TopLeft[XBlock.RightBot[ib]];


			int oldrightbot = XBlock.RightBot[ibr];
			int oldrighttop = XBlock.RightBot[ibtr];
			//int oldtopofright = topblk[oldright];
			int oldtopleft = XBlock.TopLeft[ibtl];
			int oldtopright = XBlock.TopLeft[ibtr];
			//int oldrightoftop = rightblk[oldtop];
			int oldleftbot = XBlock.LeftBot[ib];
			int oldlefttop = XBlock.LeftBot[ibtl];
			//int oldtopofleft = topblk[oldleft];
			int oldbotleft = XBlock.BotLeft[ib];
			int oldbotright = XBlock.BotLeft[ibr];
			//int oldrightofbot = rightblk[oldbot];

			


			for (int iy = 0; iy < XParam.blkwidth; iy++)
			{
				for (int ix = 0; ix < XParam.blkwidth; ix++)
				{
					i = memloc(XParam, ix, iy, ib);
					
					if (ix < (XParam.blkwidth / 2) && iy < (XParam.blkwidth /2))
					{
						ii = memloc(XParam, ix * 2, iy * 2, ib);// ix * 2 + (iy * 2) * 16 + ib * XParam.blksize;
						ir = memloc(XParam, (ix * 2 + 1), (iy * 2), ib); //(ix * 2 + 1) + (iy * 2) * 16 + ib * XParam.blksize;
						it = memloc(XParam, (ix * 2 ), (iy * 2 + 1), ib);// (ix) * 2 + (iy * 2 + 1) * 16 + ib * XParam.blksize;
						itr = memloc(XParam, (ix * 2 + 1), (iy * 2 + 1), ib); //(ix * 2 + 1) + (iy * 2 + 1) * 16 + ib * XParam.blksize;
					}
					if (ix >= (XParam.blkwidth / 2) && iy < (XParam.blkwidth / 2))
					{
						ii = memloc(XParam, (ix - XParam.blkwidth / 2) * 2, iy * 2, ibr);//((ix - 8) * 2) + (iy * 2) * 16 + rightblk[ib] * XParam.blksize;
						ir = memloc(XParam, (ix - XParam.blkwidth / 2) * 2 + 1, iy * 2, ibr);// ((ix - 8) * 2 + 1) + (iy * 2) * 16 + rightblk[ib] * XParam.blksize;
						it = memloc(XParam, (ix - XParam.blkwidth / 2) * 2, iy * 2 + 1, ibr);// ((ix - 8)) * 2 + (iy * 2 + 1) * 16 + rightblk[ib] * XParam.blksize;
						itr = memloc(XParam, (ix - XParam.blkwidth / 2) * 2 + 1, (iy * 2 + 1), ibr);// ((ix - 8) * 2 + 1) + (iy * 2 + 1) * 16 + rightblk[ib] * XParam.blksize;
					}
					if (ix < (XParam.blkwidth / 2) && iy >= (XParam.blkwidth / 2))
					{
						ii = memloc(XParam, ix * 2, (iy - XParam.blkwidth / 2) * 2, ibtl);// ix * 2 + ((iy - 8) * 2) * 16 + topblk[ib] * XParam.blksize;
						ir = memloc(XParam, ix * 2 + 1, (iy - XParam.blkwidth / 2) * 2, ibtl);//(ix * 2 + 1) + ((iy - 8) * 2) * 16 + topblk[ib] * XParam.blksize;
						it = memloc(XParam, ix * 2, (iy - XParam.blkwidth / 2) * 2 + 1, ibtl);//(ix) * 2 + ((iy - 8) * 2 + 1) * 16 + topblk[ib] * XParam.blksize;
						itr = memloc(XParam, ix * 2 + 1, (iy - XParam.blkwidth / 2) * 2 + 1, ibtl);//(ix * 2 + 1) + ((iy - 8) * 2 + 1) * 16 + topblk[ib] * XParam.blksize;
					}
					if (ix >= (XParam.blkwidth / 2) && iy >= (XParam.blkwidth / 2))
					{
						ii = memloc(XParam, (ix - XParam.blkwidth / 2) * 2, (iy - XParam.blkwidth / 2) * 2, ibtr);// (ix - 8) * 2 + ((iy - 8) * 2) * 16 + rightblk[topblk[ib]] * XParam.blksize;
						ir = memloc(XParam, (ix - XParam.blkwidth / 2) * 2 + 1, (iy - XParam.blkwidth / 2) * 2, ibtr);//((ix - 8) * 2 + 1) + ((iy - 8) * 2) * 16 + rightblk[topblk[ib]] * XParam.blksize;
						it = memloc(XParam, (ix - XParam.blkwidth / 2) * 2, (iy - XParam.blkwidth / 2) * 2 + 1, ibtr);//(ix - 8) * 2 + ((iy - 8) * 2 + 1) * 16 + rightblk[topblk[ib]] * XParam.blksize;
						itr = memloc(XParam, (ix - XParam.blkwidth / 2) * 2 + 1, (iy - XParam.blkwidth / 2) * 2 + 1, ibtr);//((ix - 8) * 2 + 1) + ((iy - 8) * 2 + 1) * 16 + rightblk[topblk[ib]] * XParam.blksize;
					}


					// These are the only guys that need to be coarsen, other are recalculated on the fly or interpolated from forcing
					XEv.h[i] = 0.25 * (XEvo.h[ii] + XEvo.h[ir] + XEvo.h[it] + XEvo.h[itr]);
					XEv.zs[i] = 0.25 * (XEvo.zs[ii] + XEvo.zs[ir] + XEvo.zs[it] + XEvo.zs[itr]);
					XEv.u[i] = 0.25 * (XEvo.u[ii] + XEvo.u[ir] + XEvo.u[it] + XEvo.u[itr]);
					XEv.v[i] =  0.25 * (XEvo.v[ii] + XEvo.v[ir] + XEvo.v[it] + XEvo.v[itr]);
					//zb will be interpolated from input grid later // I wonder is this makes the bilinear interpolation scheme crash at the refining step for zb?
					// No because zb is also interpolated later from the original mesh data
					//zb[i] = 0.25 * (zbo[ii] + zbo[ir] + zbo[it], zbo[itr]);


				}
			}

			//Need more?

			// Make right, top and top-right block available for refine step
			XAdapt.availblk[XParam.navailblk] = ibr;
			XAdapt.availblk[XParam.navailblk + 1] = ibtl;
			XAdapt.availblk[XParam.navailblk + 2] = ibtr;

			XAdapt.newlevel[ib] = XBlock.level[ib] - 1;

			//Do not comment! While this 3 line below seem irrelevant in a first order they are needed for the neighbours below (next step down) but then is not afterward
			XAdapt.newlevel[ibr] = XBlock.level[ib] - 1;
			XAdapt.newlevel[ibtl] = XBlock.level[ib] - 1;
			XAdapt.newlevel[ibtr] = XBlock.level[ib] - 1;



			// increment available block count
			XParam.navailblk = XParam.navailblk + 3;

			// Make right, top and top-right block inactive
			XBlock.active[XAdapt.invactive[ibr]] = -1;
			XBlock.active[XAdapt.invactive[ibtl]] = -1;
			XBlock.active[XAdapt.invactive[ibtr]] = -1;

			//check neighbour's (Full neighbour happens in the next big loop below)
			if (ibr == oldrightbot) // Surely that can never be true. if that was the case the coarsening would not have been allowed!
			{
				XBlock.RightBot[ib] = ib;
				//XBlock.RightTop[ib] = ib;
			}
			else
			{
				XBlock.RightBot[ib] = oldrightbot;
				//XBlock.RightTop[ib] = oldright;
			}
			if (ibtr == oldrighttop) // Surely that can never be true. if that was the case the coarsening would not have been allowed!
			{
				XBlock.RightTop[ib] = ib;
				//XBlock.RightTop[ib] = ib;
			}
			else
			{
				XBlock.RightTop[ib] = oldrighttop;
				//XBlock.RightTop[ib] = oldright;
			}



			if (ibtl == oldtopleft)//Ditto here
			{
				XBlock.TopLeft[ib] = ib;
			}
			else
			{
				XBlock.TopLeft[ib] = oldtopleft;
			}
			if (ibtr == oldtopright)//Ditto here
			{
				XBlock.TopRight[ib] = ib;
			}
			else
			{
				XBlock.TopRight[ib] = oldtopright;
			}



			XBlock.LeftBot[ib] = oldleftbot;// It is that already but it clearer to spell it out
			XBlock.LeftTop[ib] = oldlefttop;
			if (oldlefttop == ibtl)
			{
				XBlock.LeftTop[ib] = ib;
			}

			XBlock.BotLeft[ib] = oldbotleft;
			XBlock.BotRight[ib] = oldbotright;
			if (oldbotright == ibr)
			{
				XBlock.BotRight[ib] = ib;
			}
			
			//Also need to do lft and bottom!



			// Bot and left blk should remain unchanged at this stage(they will change if the neighbour themselves change)

			XBlock.xo[ib] = XBlock.xo[ib] + calcres(XParam.dx, XBlock.level[ib] + 1);
			XBlock.yo[ib] = XBlock.yo[ib] + calcres(XParam.dx, XBlock.level[ib] + 1);



		}

	}

	//____________________________________________________
	//
	// Step 3: deal with neighbour
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		int i, ii, ir, it, itr;
		if (ib >= 0) // ib can be -1 for newly inactive blocks
		{

			int oldrightbot = XBlock.RightBot[ib];
			
			int oldtopleft = XBlock.TopLeft[ib];
			
			int oldleftbot = XBlock.LeftBot[ib];
			
			int oldbotleft = XBlock.BotLeft[ib];
			




			if (XAdapt.newlevel[oldleftbot] < XBlock.level[oldleftbot])
			{
				//left blk has coarsen
				if (XAdapt.coarsen[XBlock.LeftBot[oldleftbot]])
				{
					XBlock.LeftBot[ib] = XBlock.LeftBot[oldleftbot];
					XBlock.LeftTop[ib] = XBlock.LeftBot[oldleftbot];
				}
				else
				{
					XBlock.LeftBot[ib] = XBlock.BotLeft[XBlock.LeftBot[oldleftbot]];
					XBlock.LeftTop[ib] = XBlock.BotLeft[XBlock.LeftBot[oldleftbot]];
				}
			}
			



			if (XAdapt.newlevel[oldbotleft] < XBlock.level[oldbotleft])
			{
				// botblk has coarsen
				if (XAdapt.coarsen[XBlock.BotLeft[oldbotleft]])
				{
					XBlock.BotLeft[ib] = XBlock.BotLeft[oldbotleft];
					XBlock.BotRight[ib] = XBlock.BotLeft[oldbotleft];
				}
				else
				{
					XBlock.BotLeft[ib] = XBlock.LeftBot[XBlock.BotLeft[oldbotleft]];
					XBlock.BotRight[ib] = XBlock.LeftBot[XBlock.BotLeft[oldbotleft]];
				}
			}
			


			if (XAdapt.newlevel[oldrightbot] < XBlock.level[oldrightbot])
			{
				// right block has coarsen
				if (!XAdapt.coarsen[oldrightbot])
				{
					XBlock.RightBot[ib] = XBlock.BotLeft[oldrightbot];
					XBlock.RightTop[ib] = XBlock.BotLeft[oldrightbot];

				}
				// else do nothing because the right block is the reference one
			}


			if (XAdapt.newlevel[oldtopleft] < XBlock.level[oldtopleft])
			{
				// top blk has coarsen
				if (!XAdapt.coarsen[oldtopleft])
				{
					XBlock.TopLeft[ib] = XBlock.LeftBot[oldtopleft];
					XBlock.TopRight[ib] = XBlock.LeftBot[oldtopleft];
				}


			}


		}
	}

	//____________________________________________________
	//
	// Step 4: deal with other neighbour pair
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		int i, ii, ir, it, itr;
		if (ib >= 0 && (XAdapt.newlevel[ib] < XBlock.level[ib])) // ib can be -1 for newly inactive blocks
		{
			if (XAdapt.newlevel[XBlock.LeftBot[ib]] <= XAdapt.newlevel[ib])
			{
				XBlock.LeftTop[ib] = XBlock.LeftBot[ib]; // this is fine even if this is a boundary edge
			}
			else //(XAdapt.newlevel[XBlock.LeftBot[ib]] > XAdapt.newlevel[ib])
			{
				XBlock.LeftTop[ib] = XBlock.TopLeft[XBlock.LeftBot[ib]];
			}

			if (XAdapt.newlevel[XBlock.RightBot[ib]] <= XAdapt.newlevel[ib])
			{
				XBlock.RightTop[ib] = XBlock.RightBot[ib]; // this is fine even if this is a boundary edge
			}
			else //(XAdapt.newlevel[XBlock.LeftBot[ib]] > XAdapt.newlevel[ib])
			{
				XBlock.RightTop[ib] = XBlock.TopLeft[XBlock.RightBot[ib]];
			}

			if (XAdapt.newlevel[XBlock.BotLeft[ib]] <= XAdapt.newlevel[ib])
			{
				XBlock.BotRight[ib] = XBlock.BotLeft[ib];
			}
			else //(XAdapt.newlevel[XBlock.LeftBot[ib]] > XAdapt.newlevel[ib])
			{
				XBlock.BotRight[ib] = XBlock.RightBot[XBlock.BotLeft[ib]];
			}


			if (XAdapt.newlevel[XBlock.TopLeft[ib]] <= XAdapt.newlevel[ib])
			{
				XBlock.TopRight[ib] = XBlock.TopLeft[ib];
			}
			else //(XAdapt.newlevel[XBlock.TopBot[ib]] > XAdapt.newlevel[ib])
			{
				XBlock.TopRight[ib] = XBlock.RightBot[XBlock.TopLeft[ib]];
			}


		}
	}
}

template void coarsen<float>(Param XParam, BlockP<float>& XBlock, AdaptP& XAdapt, EvolvingP<float> XEvo, EvolvingP<float>& XEv);
template void coarsen<double>(Param XParam, BlockP<double>& XBlock, AdaptP& XAdapt, EvolvingP<double> XEvo, EvolvingP<double>& XEv);

template <class T> void refine(Param XParam, BlockP<T>& XBlock, AdaptP& XAdapt, EvolvingP<T> XEvo, EvolvingP<T>& XEv)
{
	//==========================================================================
	//	REFINE
	//==========================================================================
	// This is also a multi step process:
	//	1. Interpolate conserved variables (although zb is done here it is overwritten later down the code)
	//	2. Set direct neighbours blockxo/yo and levels
	//	3. Set wider neighbourhood
	//	4. Activate new blocks 

	//____________________________________________________
	//
	// Step 1. Interpolate conserved variables

	int nblk = XParam.nblk;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//

		int ib = XBlock.active[ibl];
		int o, oo, ooo, oooo;
		int  ii, ir, it,itr;


		if (ib >= 0) // ib can be -1 for newly inactive blocks
		{
			if (XAdapt.refine[ib])
			{
				
				// Bilinear interpolation
				for (int iy = 0; iy < XParam.blkwidth; iy++)
				{
					for (int ix = 0; ix < XParam.blkwidth; ix++)
					{
						int kx[] = { 0, XParam.blkwidth/2, 0, XParam.blkwidth/2 };
						int ky[] = { 0, 0, XParam.blkwidth/2, XParam.blkwidth/2 };
						int kb[] = { ib, XAdapt.availblk[XAdapt.csumblk[ib]], XAdapt.availblk[XAdapt.csumblk[ib] + 1], XAdapt.availblk[XAdapt.csumblk[ib] + 2] };

						//double mx, my;

						for (int kk = 0; kk < 4; kk++)
						{

							int cx, fx, cy, fy;
							
							T lx, ly, rx, ry;

							lx = ix * 0.5 - 0.25;
							ly = iy * 0.5 - 0.25;


							fx = (int)floor(lx) + kx[kk];
							cx = (int)ceil(lx) + kx[kk];
							fy = (int)floor(ly) + ky[kk];
							cy = (int)ceil(ly) + ky[kk];

							rx = (lx)+(double)kx[kk];
							ry = (ly)+(double)ky[kk];

							o = memloc(XParam,ix,iy, kb[kk]);//ix + iy * 16 + kb[kk] * XParam.blksize;

							ii = memloc(XParam, fx, fy, ib);
							ir = memloc(XParam, cx, fy, ib);
							it = memloc(XParam, fx, cy, ib);
							itr = memloc(XParam, cx, cy, ib);


							//printf("fx = %d; cx=%d; fy=%d; cy=%d; rx=%f; ry=%f\n", fx, cx, fy, cy,rx,ry);

							//printf("First blk %f\n",BilinearInterpolation(h11, h12, h21, h22, fx, cx, fy, cy, rx, ry));

							XEv.h[o] = BilinearInterpolation(XEvo.h[ii], XEvo.h[it], XEvo.h[ir], XEvo.h[itr], (T)fx, (T)cx, (T)fy, (T)cy, rx, ry);
							XEv.zs[o] = BilinearInterpolation(XEvo.zs[ii], XEvo.zs[it], XEvo.zs[ir], XEvo.zs[itr], (T)fx, (T)cx, (T)fy, (T)cy, rx, ry);
							XEv.u[o] = BilinearInterpolation(XEvo.u[ii], XEvo.u[it], XEvo.u[ir], XEvo.u[itr], (T)fx, (T)cx, (T)fy, (T)cy, rx, ry);
							XEv.v[o] = BilinearInterpolation(XEvo.v[ii], XEvo.v[it], XEvo.v[ir], XEvo.v[itr], (T)fx, (T)cx, (T)fy, (T)cy, rx, ry);


						}

					}
				}
			}
		}
	}

	//____________________________________________________
	//	
	// Step 2. Set direct neighbours blockxo/yo and levels
	//
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		if (ib >= 0) // ib can be -1 for newly inactive blocks
		{
			if (XAdapt.refine[ib])
			{
				double delx = calcres(XParam.dx, XBlock.level[ib] + 1);
				double xoblk = XBlock.xo[ib] - 0.5 * delx;
				double yoblk = XBlock.yo[ib] - 0.5 * delx;

				int oldtopleft, oldleftbot, oldrightbot, oldbotleft;
				int oldtopright, oldlefttop, oldrighttop, oldbotright;


				oldtopleft = XBlock.TopLeft[ib];
				oldtopright = XBlock.TopRight[ib];

				oldbotleft = XBlock.BotLeft[ib];
				oldbotright = XBlock.BotRight[ib];

				oldrightbot = XBlock.RightBot[ib];
				oldrighttop = XBlock.RightTop[ib];

				oldleftbot = XBlock.LeftBot[ib];
				oldlefttop = XBlock.LeftTop[ib];

				// One block becomes 4 blocks:
				// ib is the starting blk and new bottom left blk
				// ibr is the new bottom right blk
				// ibtl is the new top left blk
				// ibtr is the new top right block

				int ibr, ibtl, ibtr;
				ibr = XAdapt.availblk[XAdapt.csumblk[ib]];
				ibtl = XAdapt.availblk[XAdapt.csumblk[ib] + 1];
				ibtr = XAdapt.availblk[XAdapt.csumblk[ib] + 2];

				// sort out block info
				XAdapt.newlevel[ib] = XBlock.level[ib] + 1;
				XAdapt.newlevel[ibr] = XBlock.level[ib] + 1;
				XAdapt.newlevel[ibtl] = XBlock.level[ib] + 1;
				XAdapt.newlevel[ibtr] = XBlock.level[ib] + 1;

				XBlock.xo[ib] = xoblk;
				XBlock.yo[ib] = yoblk;
				//bottom right blk
				XBlock.xo[ibr] = xoblk + (XParam.blkwidth) * delx;
				XBlock.yo[ibr] = yoblk;
				//top left blk
				XBlock.xo[ibtl] = xoblk;
				XBlock.yo[ibtl] = yoblk + (XParam.blkwidth) * delx;
				//top right blk
				XBlock.xo[ibtr] = xoblk + (XParam.blkwidth) * delx;
				XBlock.yo[ibtr] = yoblk + (XParam.blkwidth) * delx;


				//sort out internal blocks neighbour
				// external neighbours are dealt with in the following loop

				//top neighbours
				XBlock.TopLeft[ib] = ibtl;
				XBlock.TopRight[ib] = ibtl;

				XBlock.TopLeft[ibtl] = oldtopleft;
				XBlock.TopRight[ibtl] = oldtopleft;

				XBlock.TopLeft[ibr] = ibtr;
				XBlock.TopRight[ibr] = ibtr;

				XBlock.TopLeft[ibtr] = oldtopright;
				XBlock.TopRight[ibtr] = oldtopright;

				// Right neighbours
				XBlock.RightBot[ib] = ibr;
				XBlock.RightTop[ib] = ibr;

				XBlock.RightBot[ibr] = oldrightbot;
				XBlock.RightTop[ibr] = oldrightbot;

				XBlock.RightBot[ibtl] = ibtr;
				XBlock.RightTop[ibtl] = ibtr;

				XBlock.RightBot[ibtr] = oldrighttop;
				XBlock.RightTop[ibtr] = oldrighttop;
				
				//Bottom Neighbours
				XBlock.BotLeft[ibtl] = ib;
				XBlock.BotRight[ibtl] = ib;

				XBlock.BotLeft[ibtr] = ibr;
				XBlock.BotRight[ibtr] = ibr;

				XBlock.BotLeft[ibr] = oldbotright;
				XBlock.BotRight[ibr] = oldbotright;
				
				//Left neightbour
				XBlock.LeftBot[ibr] = ib;
				XBlock.LeftTop[ibr] = ib;

				XBlock.LeftBot[ibtr] = ibtl;
				XBlock.LeftTop[ibtr] = ibtl;

				XBlock.LeftBot[ibtl] = oldlefttop;
				XBlock.LeftTop[ibtl] = oldlefttop;
				
				
				
				XParam.navailblk = XParam.navailblk - 3;
			}
		}

	}


	//____________________________________________________
	//	
	//	Step 3. Set wider neighbourhood
	//
	// set the external neighbours
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];
		if (ib >= 0) // ib can be -1 for newly inactive blocks
		{
			if (XAdapt.refine[ib])
			{
				int oldtopleft, oldleftbot, oldrightbot, oldbotleft;
				int oldtopright, oldlefttop, oldrighttop, oldbotright;

				// One block becomes 4 blocks:
				// ib is the starting blk and new bottom left blk
				// ibr is the new bottom right blk
				// ibtl is the new top left blk
				// ibtr is the new top right block

				int ibr, ibtl, ibtr;
				ibr = XAdapt.availblk[XAdapt.csumblk[ib]];
				ibtl = XAdapt.availblk[XAdapt.csumblk[ib] + 1];
				ibtr = XAdapt.availblk[XAdapt.csumblk[ib] + 2];

				oldtopleft = XBlock.TopLeft[ibtl];
				oldtopright = XBlock.TopRight[ibtr];

				oldbotleft = XBlock.BotLeft[ib];
				oldbotright = XBlock.BotRight[ibr];

				oldrightbot = XBlock.RightBot[ibr];
				oldrighttop = XBlock.RightTop[ibtr];

				oldleftbot = XBlock.LeftBot[ib];
				oldlefttop = XBlock.LeftTop[ibtl];


				// Deal with neighbours 
				// This is F@*%!ng tedious!

				//_________________________________
				// Left Neighbours
				if (XAdapt.refine[oldleftbot])// is true and possibly the top left guy!!!
				{
					if (XAdapt.newlevel[oldleftbot] < XAdapt.newlevel[ib])
					{
						if (abs(XBlock.yo[ib] - XBlock.yo[oldleftbot]) < calcres(XParam.dx, XAdapt.newlevel[ib]))//(XBlock.RightBot[XBlock.RightBot[oldleftbot]] == ib) // bottom side
						{
							XBlock.LeftBot[ib] = XAdapt.availblk[XAdapt.csumblk[oldleftbot]];
							XBlock.LeftTop[ib] = XAdapt.availblk[XAdapt.csumblk[oldleftbot]];

							XBlock.LeftBot[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldleftbot]];
							XBlock.LeftTop[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldleftbot]];
						}
						else //Top side
						{
							XBlock.LeftBot[ib] = XAdapt.availblk[XAdapt.csumblk[oldleftbot] + 2];
							XBlock.LeftTop[ib] = XAdapt.availblk[XAdapt.csumblk[oldleftbot] + 2];

							XBlock.LeftBot[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldleftbot] + 2];
							XBlock.LeftTop[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldleftbot] + 2];
						}
					}
					else
					{
						if (oldleftbot == ib)
						{
							XBlock.LeftBot[ib] = ib;
							XBlock.LeftTop[ib] = ib;

							if (oldlefttop == ib)
							{
								XBlock.LeftBot[ibtl] = ibtl;
								XBlock.LeftTop[ibtl] = ibtl;
							}
							else if(XAdapt.refine[oldlefttop])
							{
								XBlock.LeftBot[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldlefttop]];
								XBlock.LeftTop[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldlefttop] + 2];
							}
							else
							{
								XBlock.LeftBot[ibtl] = oldlefttop;
								XBlock.LeftTop[ibtl] = oldlefttop;
								XBlock.RightBot[oldlefttop] = ibtl;
								XBlock.RightTop[oldlefttop] = ibtl;
							}
							
						}
						else if (XAdapt.newlevel[oldleftbot] == XAdapt.newlevel[ib])
						{
							XBlock.LeftBot[ib] = XAdapt.availblk[XAdapt.csumblk[oldleftbot]];
							XBlock.LeftTop[ib] = XAdapt.availblk[XAdapt.csumblk[oldleftbot]];

							XBlock.LeftBot[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldleftbot] + 2];
							XBlock.LeftTop[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldleftbot] + 2];
						}
						else
						{
							XBlock.LeftBot[ib] = XAdapt.availblk[XAdapt.csumblk[oldleftbot]];
							XBlock.LeftTop[ib] = XAdapt.availblk[XAdapt.csumblk[oldleftbot] + 2];
							if (oldlefttop == ib)
							{
								XBlock.LeftBot[ibtl] = ibtl;
								XBlock.LeftTop[ibtl] = ibtl;
							}
							else if (XAdapt.refine[oldlefttop])
							{
								XBlock.LeftBot[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldlefttop] ];
								XBlock.LeftTop[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldlefttop] + 2];
							}
							else
							{
								XBlock.LeftBot[ibtl] = oldlefttop;
								XBlock.LeftTop[ibtl] = oldlefttop;
								XBlock.RightBot[oldlefttop] = ibtl;
								XBlock.RightTop[oldlefttop] = ibtl;
							}
						}
					}
				}
				else // oldleftbot did not refine (couldn't have corasen either)
				{
					XBlock.LeftTop[ib] = oldleftbot;

					if (XAdapt.newlevel[oldleftbot] < XAdapt.newlevel[ib])
					{
						//Don't  need to do this part (i.e. it is already the case)
						//XBlock.LeftBot[ib] = oldleftbot;
						//XBlock.LeftTop[ib] = oldleftbot;

						//XBlock.LeftBot[ibtl] = oldleftbot;
						//XBlock.LeftTop[ibtl] = oldleftbot;
						XBlock.RightBot[oldleftbot] = ib;
						XBlock.RightTop[oldleftbot] = ibtl;
					}
					else
					{
						XBlock.RightBot[oldleftbot] = ib;
						XBlock.RightTop[oldleftbot] = ib;
						if (oldlefttop != ib)
						{
							
							if (!XAdapt.refine[oldlefttop])
							{
								XBlock.RightBot[oldlefttop] = ibtl;
								XBlock.RightTop[oldlefttop] = ibtl;
							}
							else
							{
								XBlock.LeftBot[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldlefttop]];
								XBlock.LeftTop[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldlefttop] + 2];
							}
						}
						else
						{
							XBlock.LeftBot[ibtl] = ibtl;
							XBlock.LeftTop[ibtl] = ibtl;
						}
						///
					}
				}
				
				//_________________________________
				// Right Neighbours
				if (XAdapt.refine[oldrightbot])// is true and possibly the top left guy!!!
				{
					if (XAdapt.newlevel[oldrightbot] < XAdapt.newlevel[ib])
					{
						if (abs(XBlock.yo[ib]-XBlock.yo[oldrightbot])<calcres(XParam.dx, XAdapt.newlevel[ib])) //XBlock.LeftBot[oldrightbot] == ib// bottom side
						{
							XBlock.RightBot[ibr] = oldrightbot;
							XBlock.RightTop[ibr] = oldrightbot;

							XBlock.RightBot[ibtr] = oldrightbot;
							XBlock.RightTop[ibtr] = oldrightbot;
						}
						else //Top side
						{
							XBlock.RightBot[ibr] = XAdapt.availblk[XAdapt.csumblk[oldrightbot] + 1];
							XBlock.RightTop[ibr] = XAdapt.availblk[XAdapt.csumblk[oldrightbot] + 1];

							XBlock.RightBot[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldrightbot] + 1];
							XBlock.RightTop[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldrightbot] + 1];
						}
					}
					else
					{
						if (oldrightbot == ib)
						{
							XBlock.RightBot[ibr] = ibr;
							XBlock.RightTop[ibr] = ibr;

							if (oldrighttop == ib)
							{
								XBlock.RightBot[ibtr] = ibtr;
								XBlock.RightTop[ibtr] = ibtr;
							}
							else if (XAdapt.refine[oldrighttop])
							{
								XBlock.RightBot[ibtr] = oldrighttop;
								XBlock.RightTop[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldrighttop] + 1];
							}
							else
							{
								XBlock.RightBot[ibtr] = oldrighttop;
								XBlock.RightTop[ibtr] = oldrighttop;
								XBlock.LeftBot[oldrighttop] = ibtr;
								XBlock.LeftTop[oldrighttop] = ibtr;
							}

						}
						else if (XAdapt.newlevel[oldrightbot] == XAdapt.newlevel[ib])
						{
							XBlock.RightBot[ibr] = oldrightbot;
							XBlock.RightTop[ibr] = oldrightbot;

							XBlock.RightBot[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldrightbot] + 1];
							XBlock.RightTop[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldrightbot] + 1];
						}
						else
						{
							XBlock.RightBot[ibr] = oldrightbot;
							XBlock.RightTop[ibr] = XAdapt.availblk[XAdapt.csumblk[oldrightbot] + 1];
							if (oldrighttop == ib)
							{
								XBlock.RightBot[ibtr] = ibtr;
								XBlock.RightTop[ibtr] = ibtr;
							}
							else if (XAdapt.refine[oldrighttop])
							{
								XBlock.RightBot[ibtr] = oldrighttop;
								XBlock.RightTop[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldrighttop] + 1];
							}
							else
							{
								XBlock.RightBot[ibtr] = oldrighttop;
								XBlock.RightTop[ibtr] = oldrighttop;
								XBlock.LeftBot[oldrighttop] = ibtr;
								XBlock.LeftTop[oldrighttop] = ibtr;
							}
						}
					}
				}
				else // oldrightbot did not refine (couldn't have corasen either)
				{
					//XBlock.RightTop[ib] = oldrightbot;
					if (XAdapt.newlevel[oldrightbot] < XAdapt.newlevel[ib])
					{
						//Don't  need to do this part (i.e. it is already the case)
						//XBlock.LeftBot[ib] = oldleftbot;
						//XBlock.LeftTop[ib] = oldleftbot;

						//XBlock.LeftBot[ibtl] = oldleftbot;
						//XBlock.LeftTop[ibtl] = oldleftbot;
						XBlock.LeftBot[oldrightbot] = ibr;
						XBlock.LeftTop[oldrightbot] = ibtr;
					}
					else
					{
						XBlock.LeftBot[oldrightbot] = ibr;
						XBlock.LeftTop[oldrightbot] = ibr;
						if (oldrighttop != ib)
						{
							
							if (!XAdapt.refine[oldrighttop])
							{
								XBlock.LeftBot[oldrighttop] = ibtr;
								XBlock.LeftTop[oldrighttop] = ibtr;
							}
							else
							{
								XBlock.RightBot[ibtr] = oldrighttop;
								XBlock.RightTop[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldrighttop] + 1];
							}
						}
						else
						{
							XBlock.RightBot[ibtr] = ibtr;
							XBlock.RightTop[ibtr] = ibtr;
						}
						///
					}
				}


				//_________________________________
				// Bottom Neighbours
				if (XAdapt.refine[oldbotleft])// is true and possibly the top left guy!!!
				{
					if (XAdapt.newlevel[oldbotleft] < XAdapt.newlevel[ib])
					{
						if (abs(XBlock.xo[ib] - XBlock.xo[oldbotleft]) < calcres(XParam.dx, XAdapt.newlevel[ib]))//(XBlock.TopLeft[XBlock.TopLeft[oldbotleft]] == ib) // left side
						{
							XBlock.BotLeft[ib] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 1];
							XBlock.BotRight[ib] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 1];

							XBlock.BotLeft[ibr] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 1];
							XBlock.BotRight[ibr] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 1];
						}
						else //Right side
						{
							XBlock.BotLeft[ib] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 2];
							XBlock.BotRight[ib] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 2];

							XBlock.BotLeft[ibr] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 2];
							XBlock.BotRight[ibr] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 2];
						}
					}
					else
					{
						if (oldbotleft == ib)
						{
							XBlock.BotLeft[ib] = ib;
							XBlock.BotRight[ib] = ib;

							if (oldbotright == ib)
							{
								XBlock.BotLeft[ibr] = ibr;
								XBlock.BotRight[ibr] = ibr;
							}
							else if (XAdapt.refine[oldbotright])
							{
								XBlock.BotLeft[ibr] = XAdapt.availblk[XAdapt.csumblk[oldbotright] + 1];
								XBlock.BotRight[ibr] = XAdapt.availblk[XAdapt.csumblk[oldbotright] + 2];
							}
							else
							{
								XBlock.BotLeft[ibr] = oldbotright;
								XBlock.BotRight[ibr] = oldbotright;
								XBlock.TopLeft[oldbotright] = ibr;
								XBlock.TopRight[oldbotright] = ibr;
							}

						}
						else if (XAdapt.newlevel[oldbotleft] == XAdapt.newlevel[ib])
						{
							XBlock.BotLeft[ib] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 1];
							XBlock.BotRight[ib] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 1];

							XBlock.BotLeft[ibr] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 2];
							XBlock.BotRight[ibr] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 2];
						}
						else
						{
							XBlock.BotLeft[ib] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 1];
							XBlock.BotRight[ib] = XAdapt.availblk[XAdapt.csumblk[oldbotleft] + 2];
							if (oldbotright == ib)
							{
								XBlock.BotLeft[ibr] = ibr;
								XBlock.BotRight[ibr] = ibr;
							}
							else if (XAdapt.refine[oldbotright])
							{
								XBlock.BotLeft[ibr] = XAdapt.availblk[XAdapt.csumblk[oldbotright] + 1];
								XBlock.BotRight[ibr] = XAdapt.availblk[XAdapt.csumblk[oldbotright] + 2];
							}
							else
							{
								XBlock.BotLeft[ibr] = oldbotright;
								XBlock.BotRight[ibr] = oldbotright;
								XBlock.TopLeft[oldbotright] = ibr;
								XBlock.TopRight[oldbotright] = ibr;
							}
						}
					}
				}
				else // oldbotleft did not refine (couldn't have corasen either)
				{
					/// How did we miss that before
					XBlock.BotRight[ib] = oldbotleft;

					if (XAdapt.newlevel[oldbotleft] < XAdapt.newlevel[ib])
					{
						//Don't  need to do this part (i.e. it is already the case)
						//XBlock.LeftBot[ib] = oldleftbot;
						//XBlock.LeftTop[ib] = oldleftbot;

						//XBlock.LeftBot[ibtl] = oldleftbot;
						//XBlock.LeftTop[ibtl] = oldleftbot;
						XBlock.TopLeft[oldbotleft] = ib;
						XBlock.TopRight[oldbotleft] = ibr;
					}
					else
					{
						XBlock.TopLeft[oldbotleft] = ib;
						XBlock.TopRight[oldbotleft] = ib;

						if (oldbotright != ib)
						{

							if (!XAdapt.refine[oldbotright])
							{
								XBlock.TopLeft[oldbotright] = ibr;
								XBlock.TopRight[oldbotright] = ibr;
							}
							else
							{
								XBlock.BotLeft[ibr] = XAdapt.availblk[XAdapt.csumblk[oldbotright] + 1];
								XBlock.BotRight[ibr] = XAdapt.availblk[XAdapt.csumblk[oldbotright] + 2];
							}
						}
						else
						{
							XBlock.BotLeft[ibr] = ibr;
							XBlock.BotRight[ibr] = ibr;
						}
						///
					}
				}

				//_________________________________
				// Top Neighbours
				if (XAdapt.refine[oldtopleft])// is true and possibly the top left guy!!!
				{
					if (XAdapt.newlevel[oldtopleft] < XAdapt.newlevel[ib])
					{
						if (abs(XBlock.xo[ib] - XBlock.xo[oldtopleft]) < calcres(XParam.dx, XAdapt.newlevel[ib]))//(XBlock.BotLeft[oldtopleft] == ib) // left side
						{
							XBlock.TopLeft[ibtl] = oldtopleft;
							XBlock.TopRight[ibtl] = oldtopleft;

							XBlock.TopLeft[ibtr] = oldtopleft;
							XBlock.TopRight[ibtr] = oldtopleft;
						}
						else //Right side
						{
							XBlock.TopLeft[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldtopleft]];
							XBlock.TopRight[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldtopleft]];

							XBlock.TopLeft[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldtopleft]];
							XBlock.TopRight[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldtopleft]];
						}
					}
					else
					{
						if (oldtopleft == ib)
						{
							XBlock.TopLeft[ibtl] = ibtl;
							XBlock.TopRight[ibtl] = ibtl;

							if (oldtopright == ib)
							{
								XBlock.TopLeft[ibtr] = ibtr;
								XBlock.TopRight[ibtr] = ibtr;
							}
							else if (XAdapt.refine[oldtopright])
							{
								XBlock.TopLeft[ibtr] = oldtopright;
								XBlock.TopRight[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldtopright]];
							}
							else
							{
								XBlock.TopLeft[ibtr] = oldbotright;
								XBlock.TopRight[ibtr] = oldbotright;
								XBlock.BotLeft[oldtopright] = ibtr;
								XBlock.BotRight[oldtopright] = ibtr;
							}

						}
						else if (XAdapt.newlevel[oldtopleft] == XAdapt.newlevel[ib])
						{
							XBlock.TopLeft[ibtl] = oldtopleft;
							XBlock.TopRight[ibtl] = oldtopleft;

							XBlock.TopLeft[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldtopleft] ];
							XBlock.TopRight[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldtopleft] ];
						}
						else
						{
							XBlock.TopLeft[ibtl] = oldtopleft;
							XBlock.TopRight[ibtl] = XAdapt.availblk[XAdapt.csumblk[oldtopleft]];
							if (oldtopright == ib)
							{
								XBlock.TopLeft[ibtr] = ibtr;
								XBlock.TopRight[ibtr] = ibtr;
							}
							else if (XAdapt.refine[oldtopright])
							{
								XBlock.TopLeft[ibtr] = oldtopright;
								XBlock.TopRight[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldtopright]];
							}
							else
							{
								XBlock.TopLeft[ibtr] = oldtopright;
								XBlock.TopRight[ibtr] = oldtopright;
								XBlock.BotLeft[oldtopright] = ibtr;
								XBlock.BotRight[oldtopright] = ibtr;
							}
						}
					}
				}
				else // oldleftbot did not refine (couldn't have corasen either)
				{
					//XBlock.TopRight[ib] = oldtopleft;
					if (XAdapt.newlevel[oldtopleft] < XAdapt.newlevel[ib])
					{
						//Don't  need to do this part (i.e. it is already the case)
						//XBlock.LeftBot[ib] = oldleftbot;
						//XBlock.LeftTop[ib] = oldleftbot;

						//XBlock.LeftBot[ibtl] = oldleftbot;
						//XBlock.LeftTop[ibtl] = oldleftbot;
						XBlock.BotLeft[oldtopleft] = ibtl;
						XBlock.BotRight[oldtopleft] = ibtr;
					}
					else
					{
						XBlock.BotLeft[oldtopleft] = ibtl;
						XBlock.BotRight[oldtopleft] = ibtl;
						if (oldtopright != ib)
						{

							if (!XAdapt.refine[oldtopright])
							{
								XBlock.BotLeft[oldtopright] = ibtr;
								XBlock.BotRight[oldtopright] = ibtr;
							}
							else
							{
								XBlock.TopLeft[ibtr] = oldtopright;
								XBlock.TopRight[ibtr] = XAdapt.availblk[XAdapt.csumblk[oldtopright]];
							}
						}
						else
						{
							XBlock.TopLeft[ibtr] = ibtr;
							XBlock.TopRight[ibtr] = ibtr;
						}
						///
					}
				}

			}
		}
	}
	//____________________________________________________
	//	
	//	Step 4. Activate new blocks 
	//

	nblk = XParam.nblk;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		//

		int ib = XBlock.active[ibl];

		if (ib >= 0) // ib can be -1 for newly inactive blocks
		{

			if (XAdapt.refine[ib] == true)
			{

				//After that we are done so activate the new blocks
				XBlock.active[nblk] = XAdapt.availblk[XAdapt.csumblk[ib]];
				XBlock.active[nblk + 1] = XAdapt.availblk[XAdapt.csumblk[ib] + 1];
				XBlock.active[nblk + 2] = XAdapt.availblk[XAdapt.csumblk[ib] + 2];



				nblk = nblk + 3;
			}
		}
	}

	// Now clean up the mess
}
template void refine<float>(Param XParam, BlockP<float>& XBlock, AdaptP& XAdapt, EvolvingP<float> XEvo, EvolvingP<float>& XEv);
template void refine<double>(Param XParam, BlockP<double>& XBlock, AdaptP& XAdapt, EvolvingP<double> XEvo, EvolvingP<double>& XEv);


template <class T> void Adaptationcleanup(Param &XParam, BlockP<T>& XBlock, AdaptP& XAdapt)
{
	//===========================================================
	// UPDATE all remaining variables and clean up

	//____________________________________________________
	//
	//	Update level
	//
	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		//
		int oldlevel;
		int ib = XBlock.active[ibl];


		if (ib >= 0) // ib can be -1 for newly inactive blocks
		{


			XBlock.level[ib] = XAdapt.newlevel[ib];


		}
	}

	//____________________________________________________
	//
	//	Reorder activeblk
	//
	// 
	int nblk = XParam.nblk;
	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		//reuse newlevel as temporary storage for activeblk
		XAdapt.newlevel[ibl] = XBlock.active[ibl];
		XBlock.active[ibl] = -1;


	}
	// cleanup and Reorder active block list
	int ib = 0;
	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{

		if (XAdapt.newlevel[ibl] != -1)//i.e. old activeblk
		{
			XBlock.active[ib] = XAdapt.newlevel[ibl];

			ib++;
		}
	}

	nblk = ib;

	//____________________________________________________
	//
	//	Reset adaptive info
	//
	// 
	for (int ibl = 0; ibl < XParam.nblkmem; ibl++)
	{
		
		XAdapt.newlevel[ibl] = 0;
		XAdapt.refine[ibl] = false;
		XAdapt.coarsen[ibl] = false;
	}
	XParam.nblk = nblk;
	
}

template void Adaptationcleanup<float>(Param& XParam, BlockP<float>& XBlock, AdaptP& XAdapt);
template void Adaptationcleanup<double>(Param& XParam, BlockP<double>& XBlock, AdaptP& XAdapt);

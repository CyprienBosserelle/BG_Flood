


#include "Adaptation.h"



template <class T> void Adaptation(Param& XParam, Model<T>& XModel)
{
	int oldnblk = 0;
	if (XParam.maxlevel != XParam.minlevel)
	{
		while (oldnblk != XParam.nblk)
			//for (int i=0; i<1;i++)
		{
			oldnblk = XParam.nblk;
			//wetdrycriteria(XParam, refine, coarsen);
			inrangecriteria(XParam, (T)-10.0, (T)-10.0, XModel.zb, XModel.blocks, XModel.adapt.refine, XModel.adapt.coarsen);
			refinesanitycheck(XParam, XModel.blocks, XModel.adapt.refine, XModel.adapt.coarsen);
			XParam = adapt(XParam);
			if (!checkBUQsanity(XParam))
			{
				log("Bad BUQ mesh layout\n");
				exit(2);
				//break;
			}


		}

	}
}



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
				iter += checkneighbourrefine(XBlock.BotLeft[ib], XBlock.level[ib], XBlock.level[XBlock.BotLeft[ib]], refine, coarsen);
				iter += checkneighbourrefine(XBlock.LeftBot[ib], XBlock.level[ib], XBlock.level[XBlock.LeftBot[ib]], refine, coarsen);
				iter += checkneighbourrefine(XBlock.RightBot[ib], XBlock.level[ib], XBlock.level[XBlock.RightBot[ib]], refine, coarsen);
				

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
			if (((int((XBlock.xo[ib] - XParam.xo) / dxfac / XParam.blkwidth) % 2) == 0 && (int((XBlock.yo[ib] - XParam.yo) / dxfac / XParam.blkwidth) % 2) == 0 && XBlock.RightBot[ib] != ib &&  XBlock.TopLeft[ib] != ib && XBlock.RightBot[XBlock.TopRight[ib]] != XBlock.TopRight[ib]))
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
template <class T>
bool checkBUQsanity(Param XParam,BlockP<T> XBlock)
{
	bool check = true;
	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		int ib = XBlock.active[ibl];

		check=check && checklevel(ib, XBlock.level[ib], XBlock.LeftBot[ib], XBlock.level[XBlock.LeftBot[ib]]);





		/*
		//check 1st order neighbours level
		if (abs(XBlock.level[XBlock.leftblk[ib]] - (XBlock.level[ib])) > 1)
		{
			printf("Warning! Bad Neighbour Level. ib=%d; level[ib]=%d; leftblk[ib]=%d; level[leftblk[ib]]=%d\n", ib, XBlock.level[ib], XBlock.LeftBot[ib], level[leftblk[ib]]);
			check = false;
		}
		if (abs(level[rightblk[ib]] - (level[ib])) > 1)
		{
			printf("Warning! Bad Neighbour Level. ib=%d; level[ib]=%d; rightblk[ib]=%d; level[rightblk[ib]]=%d\n", ib, level[ib], rightblk[ib], level[rightblk[ib]]);
			check = false;
		}
		if (abs(level[botblk[ib]] - (level[ib])) > 1)
		{
			printf("Warning! Bad Neighbour Level. ib=%d; level[ib]=%d; botblk[ib]=%d; level[botblk[ib]]=%d\n", ib, level[ib], botblk[ib], level[botblk[ib]]);
			check = false;
		}
		if (abs(level[topblk[ib]] - (level[ib])) > 1)
		{
			printf("Warning! Bad Neighbour Level. ib=%d; level[ib]=%d; topblk[ib]=%d; level[topblk[ib]]=%d\n", ib, level[ib], topblk[ib], level[topblk[ib]]);

			check = false;
		}

		if ((level[leftblk[ib]] - level[ib]) == 1)
		{
			if (abs(level[topblk[leftblk[ib]]] - level[ib]) > 1)
			{
				printf("Warning! Bad Neighbour Level. ib=%d; level[ib]=%d; topblk[leftblk[ib]]=%d; level[topblk[leftblk[ib]]]=%d\n", ib, level[ib], topblk[leftblk[ib]], level[topblk[leftblk[ib]]]);
				check = false;
			}
		}

		if ((level[rightblk[ib]] - level[ib]) == 1)
		{
			if (abs(level[topblk[rightblk[ib]]] - level[ib]) > 1)
			{
				printf("Warning! Bad Neighbour Level. ib=%d; level[ib]=%d; topblk[rightblk[ib]]=%d; level[topblk[rightblk[ib]]]=%d\n", ib, level[ib], topblk[rightblk[ib]], level[topblk[rightblk[ib]]]);
				check = false;
			}
		}
		if ((level[topblk[ib]] - level[ib]) == 1)
		{
			if (abs(level[rightblk[topblk[ib]]] - level[ib]) > 1)
			{
				printf("Warning! Bad Neighbour Level. ib=%d; level[ib]=%d; rightblk[topblk[ib]]=%d; level[rightblk[topblk[ib]]]=%d\n", ib, level[ib], rightblk[topblk[ib]], level[rightblk[topblk[ib]]]);
				check = false;
			}
		}

		if ((level[botblk[ib]] - level[ib]) == 1)
		{
			if (abs(level[rightblk[botblk[ib]]] - level[ib]) > 1)
			{
				printf("Warning! Bad Neighbour Level. ib=%d; level[ib]=%d; rightblk[botblk[ib]]=%d; level[rightblk[botblk[ib]]]=%d\n", ib, level[ib], rightblk[botblk[ib]], level[rightblk[botblk[ib]]]);
				check = false;
			}
		}
		*/

	}

	return check;

}

bool checklevel(int ib, int levelib, int neighbourib, int levelneighbour)
{
	bool check = true;
	if (abs(levelneighbour - (levelib)) > 1)
	{
		log("Warning! Bad Neighbour Level. ib="+std::to_string(ib)+"; level[ib]="+ std::to_string(levelib)+"; neighbour[ib]="+ std::to_string(neighbourib) +"; level[leftblk[ib]]="+ std::to_string(levelneighbour));
		check = false;
	}
	return check;
}
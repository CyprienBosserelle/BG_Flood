


#include "Adaptation.h"







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
			coarsen[Xblock.RightBot[ib]] = false;
			coarsen[Xblock.RightTop[ib]] = false;
			coarsen[Xblock.LeftBot[ib]] = false;
			coarsen[Xblock.LeftTop[ib]] = false;
			coarsen[Xblock.TopLeft[ib]] = false;
			coarsen[Xblock.TopRight[ib]] = false;
			coarsen[Xblock.BotLeft[ib]] = false;
			coarsen[Xblock.BotRight[ib]] = false;
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
			if (levi < XBlock.level[XBlock.LeftBot[ib]] ||  levi < XBlock.level[XBlock.RightBot[ib]] || levi < XBlock.level[XBlock.TopLeft[ib]] || levi < XBlock.level[Xblock.BotLeft[ib]])
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
					if (coarsen[XBlock..RightBot[ib]] == true && coarsen[XBlock.TopLeft[ib]] == true && coarsen[XBlock..RightBot[XBlock.TopRight[ib]]] == true)
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
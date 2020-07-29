#include "Halo.h"

template <class T> void fillHalo(Param XParam, int ib, BlockP<T> XBlock, T*& z)
{
	

	fillLeft(XParam, ib, XBlock, z);
	//fill right
	//fill bot
	//fill top
	

}
template void fillHalo<double>(Param XParam, int ib, BlockP<double> XBlock, double*& z);
template void fillHalo<float>(Param XParam, int ib, BlockP<float> XBlock, float*& z);

template <class T> void fillHalo(Param XParam, BlockP<T> XBlock, EvolvingP<T> Xev)
{
	int ib;

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];
		fillHalo(XParam, ib, XBlock, Xev.h);
		fillHalo(XParam, ib, XBlock, Xev.zs);
		fillHalo(XParam, ib, XBlock, Xev.u);
		fillHalo(XParam, ib, XBlock, Xev.v);
	}
}
template void fillHalo<float>(Param XParam, BlockP<float> XBlock, EvolvingP<float> Xev);
template void fillHalo<double>(Param XParam, BlockP<double> XBlock, EvolvingP<double> Xev);


template <class T> void fillLeft(Param XParam, int ib, BlockP<T> XBlock, T* &z)
{
	int jj,bb;
	int read, write;
	int ii, ir, it, itr;


	if (XBlock.LeftBot[ib] == ib)//The lower half is a boundary 
	{
		for (int j = 0; j < (XParam.blkwidth / 2); j++)
		{

			read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			write = memloc(XParam, -1, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
			z[write] = z[read];
		}

		if (XBlock.LeftTop[ib] == ib) // boundary on the top half too
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, -1, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else // boundary is only on the bottom half and implicitely level of lefttopib is levelib+1
		{

			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				write = memloc(XParam, -1, j, ib);
				jj = (j - 8) * 2;
				ii = memloc(XParam, (XParam.blkwidth - 1), jj, XBlock.LeftTop[ib]);
				ir = memloc(XParam, (XParam.blkwidth - 2), jj, XBlock.LeftTop[ib]);
				it = memloc(XParam, (XParam.blkwidth - 1), jj + 1, XBlock.LeftTop[ib]);
				itr = memloc(XParam, (XParam.blkwidth - 2), jj + 1, XBlock.LeftTop[ib]);

				z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);

			}
		}
	}
	else if (XBlock.level[ib] == XBlock.level[ XBlock.LeftBot[ib] ]) // LeftTop block does not exist
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			//

			write = memloc(XParam, -1, j, ib);
			read = memloc(XParam, (XParam.blkwidth - 1), j, XBlock.LeftBot[ib]);
			z[write] = z[read];
		}
	}
	else if (XBlock.level[XBlock.LeftBot[ib] ]> XBlock.level[ib])
	{

		for (int j = 0; j < XParam.blkwidth / 2; j++)
		{

			write = memloc(XParam, -1, j, ib);

			jj = j * 2;
			bb = XBlock.LeftBot[ib];

			ii = memloc(XParam, (XParam.blkwidth - 1), jj, bb);
			ir = memloc(XParam, (XParam.blkwidth - 2), jj, bb);
			it = memloc(XParam, (XParam.blkwidth - 1), jj + 1, bb);
			itr = memloc(XParam, (XParam.blkwidth - 2), jj + 1, bb);

			z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
		}
		//now find out aboy lefttop block
		if (XBlock.LeftTop[ib] == ib)
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//

				read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, -1, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				z[write] = z[read];
			}
		}
		else
		{
			for (int j = (XParam.blkwidth / 2); j < (XParam.blkwidth); j++)
			{
				//
				jj = (j - 8) * 2;
				bb = XBlock.LeftBot[ib];

				//read = memloc(XParam, 0, j, ib);// 1 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				write = memloc(XParam, -1, j, ib); //0 + (j + XParam.halowidth) * XParam.blkmemwidth + ib * XParam.blksize;
				//z[write] = z[read];
				ii = memloc(XParam, (XParam.blkwidth - 1), jj, bb);
				ir = memloc(XParam, (XParam.blkwidth - 2), jj, bb);
				it = memloc(XParam, (XParam.blkwidth - 1), jj + 1, bb);
				itr = memloc(XParam, (XParam.blkwidth - 2), jj + 1, bb);

				z[write] = T(0.25) * (z[ii] + z[ir] + z[it] + z[itr]);
			}
		}

	}
	else if (XBlock.level[XBlock.LeftBot[ib]] < XBlock.level[ib]) // Neighbour is coarser; using barycentric interpolation (weights are precalculated) for the Halo 
	{
		for (int j = 0; j < XParam.blkwidth; j++)
		{
			write = memloc(XParam, -1, j, ib);

			T w1, w2, w3;
			T zi, zn1, zn2;

			int jj = XBlock.RightBot[XBlock.LeftBot[ib]] == ib?ceil(j * (T)0.5): ceil(j * (T)0.5)+ XParam.blkwidth/2;
			w1 = 1.0 / 3.0;
			w2 = ceil(j * (T)0.5) * 2 > j ? T(1.0 / 6.0) : T(0.5);
			w3 = ceil(j * (T)0.5) * 2 > j ? T(0.5) : T(1.0 / 6.0);
						
			ii= memloc(XParam, 0, j, ib);
			ir= memloc(XParam, XParam.blkwidth-1, jj, XBlock.LeftBot[ib]);
			it = memloc(XParam, XParam.blkwidth-1, jj - 1, XBlock.LeftBot[ib]);
			//2 scenarios here ib is the rightbot neighbour of the leftbot block or ib is the righttop neighbour
			if (XBlock.RightBot[XBlock.LeftBot[ib]] == ib)
			{
				if (j == 0)
				{
					if (XBlock.BotRight[XBlock.LeftBot[ib]] == XBlock.LeftBot[ib]) // no botom of leftbot block
					{
						w3 = 0.5 * (1.0 - w1);
						w2 = w3;
						ir = it;

					}
					else if (XBlock.level[XBlock.BotRight[XBlock.LeftBot[ib]]] < XBlock.level[XBlock.LeftBot[ib]]) // exists but is coarser
					{
						w1 = 4.0 / 10.0;
						w2 = 5.0 / 10.0;
						w3 = 1.0 / 10.0;
						it = memloc(XParam, XParam.blkwidth-1, XParam.blkwidth - 1, XBlock.BotRight[XBlock.LeftBot[ib]]);
					}
					else if (XBlock.level[XBlock.BotRight[XBlock.LeftBot[ib]]] == XBlock.level[XBlock.LeftBot[ib]]) // exists with same level
					{
						it = memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth - 1, XBlock.BotRight[XBlock.LeftBot[ib]]);
					}
					else if (XBlock.level[XBlock.BotRight[XBlock.LeftBot[ib]]] > XBlock.level[XBlock.LeftBot[ib]]) // exists with higher level
					{
						w1 = 1.0 / 4.0;
						w2 = 1.0 / 2.0;
						w3 = 1.0 / 4.0;
						it = memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth - 1, XBlock.BotRight[XBlock.LeftBot[ib]]);
					}
					
					
				}
									
				
			}
			else//righttopleftif == ib
			{
				if (j == (XParam.blkwidth - 1))
				{
					if (XBlock.TopRight[XBlock.LeftTop[ib]] == XBlock.LeftTop[ib]) // no botom of leftbot block
					{
						w3 = 0.5*(1.0-w1);
						w2 = w3;
						ir = it;

					}
					else if (XBlock.level[XBlock.TopRight[XBlock.LeftTop[ib]]] < XBlock.level[XBlock.LeftTop[ib]]) // exists but is coarser
					{
						w1 = 4.0 / 10.0;
						w2 = 1.0 / 10.0;
						w3 = 5.0 / 10.0;
						ir = memloc(XParam, XParam.blkwidth - 1,0, XBlock.TopRight[XBlock.LeftTop[ib]]);
					}
					else if (XBlock.level[XBlock.TopRight[XBlock.LeftTop[ib]]] == XBlock.level[XBlock.LeftTop[ib]]) // exists with same level
					{
						ir = memloc(XParam, XParam.blkwidth - 1, 0, XBlock.TopRight[XBlock.LeftTop[ib]]);
					}
					else if (XBlock.level[XBlock.TopRight[XBlock.LeftTop[ib]]] > XBlock.level[XBlock.LeftTop[ib]]) // exists with higher level
					{
						w1 = 1.0 / 4.0;
						w2 = 1.0 / 2.0;
						w3 = 1.0 / 4.0;
						ir = memloc(XParam, XParam.blkwidth - 1, XParam.blkwidth - 1, XBlock.TopRight[XBlock.LeftTop[ib]]);
					}
				}
				//
			}


			z[write] = w1 * z[ii] + w2 * z[ir] + w3 * z[it];
		}
	}
	


}
//////////////////////////////////////////////////////////////////////////////////
//						                                                        //
//Copyright (C) 2018 Bosserelle                                                 //
// This code contains an adaptative verison of flow kernelfor BG-Flood   		//
// The original code was modified from
// http://basilisk.fr/src/saint-venant.h and									//
// S. Popinet. Quadtree-adaptive tsunami modelling. Ocean Dynamics,				//
// doi: 61(9) : 1261 - 1285, 2011												//
//                                                                              //
//This program is free software: you can redistribute it and/or modify          //
//it under the terms of the GNU General Public License as published by          //
//the Free Software Foundation.                                                 //
//                                                                              //
//This program is distributed in the hope that it will be useful,               //
//but WITHOUT ANY WARRANTY; without even the implied warranty of                //    
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                 //
//GNU General Public License for more details.                                  //
//                                                                              //
//You should have received a copy of the GNU General Public License             //
//along with this program.  If not, see <http://www.gnu.org/licenses/>.         //
//////////////////////////////////////////////////////////////////////////////////



template<class T>
__host__ __device__ T RightAda(int ix, int iy, int ibl, int rightibl, int rightofbotibl, int rightoftopibl, int botofrightibl, int topofrightibl, int * level, T* var)
{
	T rightvarval;
	int i;
	int n1, n2, jj, bb;
	int lev = level[ibl];
	int rightlev = level[rightibl];



	if (ix < 15)
	{
		i=  (ix + 1) + iy * 16 + ibl * (256);// replace with blockdim.x  ...etc
		rightvarval = var[i];
	}
	else
	{
		if (rightibl == ibl) // No neighbour
		{
			i = (15) + iy * 16 + ibl * (256);
			rightvarval = var[i];

		}
		else if (lev == rightlev)
		{
			i = (0) + iy * 16 + rightibl * (256);
			rightvarval = var[i];
		}
		else if (rightlev > lev)
		{
			int ii, ir, it, itr;
			

			if (iy < 8)
			{
				jj = iy * 2;
				bb = rightibl;

			}
			if (iy >= 8)
			{
				jj = (iy - 8) * 2;
				bb = topofrightibl;
				
			}
			ii = 0 + jj * 16 + bb * 256;
			ir = 1 + jj * 16 + bb * 256;
			it = 0 + (jj + 1) * 16 + bb * 256;
			itr = 1 + (jj + 1) * 16 + bb * 256;

			rightvarval = T(0.25) * (var[ii] + var[ir] + var[it]+ var[itr]);
		}
		else if (rightlev < lev)
		{
			T vari, varn1, varn2;
			
			i = (15) + iy * 16 + ibl * (256);
			
			vari = var[i];
			
			if (iy==0)
			{
				
				if (botofrightibl == rightibl) /// the bot right block does not exist
				{
					
					varn1 = var[0 + 0 * 16 + rightibl * 256];
					rightvarval = (vari + 2 * varn1) / 3.0;
				}
				else if (rightofbotibl == rightibl) /// we are at the top
				{
					jj = 8;
					varn1 = var[0 + jj * 16 + rightibl * 256];
					varn2 = var[0 + (jj - 1) * 16 + rightibl * 256];
					rightvarval = vari / 3.0 + 0.5 * varn1 + varn2 / 6.0;


				}
				else if (level[botofrightibl] == level[rightibl])
				{
					
					varn1 = var[0 + 0 * 16 + rightibl * 256];
					varn2 = var[0 + (15) * 16 + botofrightibl * 256];
					rightvarval = vari / 3.0 + 0.5 * varn1 + varn2 / 6.0;
				}
				else if (level[botofrightibl] > level[rightibl])
				{

					varn1 = var[0 + 0 * 16 + rightibl * 256];
					varn2 = var[0 + (15) * 16 + botofrightibl * 256];
					rightvarval = vari / 4.0 + 0.5 * varn1 + varn2 / 4.0;
				}
				else if (level[botofrightibl] < level[rightibl])
				{

					varn1 = var[0 + 0 * 16 + rightibl * 256];
					varn2 = var[0 + (15) * 16 + botofrightibl * 256];
					rightvarval = vari *0.4 + 0.5 * varn1 + varn2 *0.1;
				}
			}
			else if (iy == 15)
			{
				if (topofrightibl == rightibl) /// the top right block does not exist
				{

					varn1 = var[0 + iy * 16 + rightibl * 256];
					rightvarval = (vari + 2 * varn1) / 3.0;
				}
				else if (rightoftopibl == rightibl) /// we are at the bottom
				{
					
					varn1 = var[0 + 8 * 16 + rightibl * 256];
					varn2 = var[0 + 7 * 16 + rightibl * 256];
					rightvarval = vari / 3.0 + varn1 / 6.0 + varn2 / 2.0;


				}
				else if (level[topofrightibl] == level[rightibl])
				{

					varn1 = var[0 + 0 * 16 + topofrightibl * 256];
					varn2 = var[0 + (15) * 16 + rightibl * 256];
					rightvarval = vari / 3.0 +  varn1 / 6.0 + varn2 / 2.0;
				}
				else if (level[topofrightibl] > level[rightibl])
				{

					varn1 = var[0 + 15 * 16 + rightibl * 256];
					varn2 = var[0 + (0) * 16 + topofrightibl * 256];
					rightvarval = vari / 4.0 + 0.5 * varn1 + varn2 / 4.0;
				}
				else if (level[topofrightibl] < level[rightibl])//Bug? level[botofrightibl] < level[rightibl] or level[topofrightibl] < level[rightibl]
				{

					varn1 = var[0 + 15 * 16 + rightibl * 256];
					varn2 = var[0 + (0) * 16 + topofrightibl * 256];
					rightvarval = vari * 0.4 + 0.5 * varn1 + varn2 * 0.1;
				}

			}
			else
			{
				T w0, w1, w2;
				jj = ceil(iy * 0.5);
				w0 = 1.0 / 3.0;
				
				if (jj * 2 > iy)
				{
					w1 = 1.0 / 6.0;
					w2 = 0.5;


				}
				else
				{
					w1 = 0.5;
					w2 = 1.0 / 6.0;
				}

				if (rightofbotibl == rightibl)
				{
					jj = jj + 8;
				}

				
				varn1 = var[0 + (jj) * 16 + rightibl * 256];
				varn2 = var[0 + (jj-1) * 16 + rightibl * 256];
				rightvarval = vari * w0 + varn1 * w1 + varn2 * w2;
			}

			
				
		}
	}
	
	return rightvarval;
}

template<class T>
__host__ __device__ T LeftAda(int ix, int iy, int ibl, int leftibl, int leftofbotibl, int leftoftopibl, int botofleftibl, int topofleftibl, int* level, T* var)
{
	T varval;
	int i;
	int n1, n2, jj, bb;
	int lev = level[ibl];
	int leftlev = level[leftibl];



	if (ix > 0)
	{
		i = (ix - 1) + iy * 16 + ibl * (256);// replace with blockdim.x  ...etc
		varval = var[i];
	}
	else
	{
		if (leftibl == ibl) // No neighbour
		{
			i = (0) + iy * 16 + ibl * (256);
			varval = var[i];

		}
		else if (lev == leftlev) // neighbour blk is same resolution
		{
			i = (15) + iy * 16 + leftibl * (256);
			varval = var[i];
		}
		else if (leftlev > lev)
		{
			int ii, ir, it, itr;


			if (iy < 8)
			{
				jj = iy * 2;
				bb = leftibl;

			}
			if (iy >= 8)
			{
				jj = (iy - 8) * 2;
				bb = topofleftibl;

			}
			ii = 15 + jj * 16 + bb * 256;
			ir = 14 + jj * 16 + bb * 256;
			it = 15 + (jj + 1) * 16 + bb * 256;
			itr = 14 + (jj + 1) * 16 + bb * 256;

			varval = T(0.25) * (var[ii] + var[ir] + var[it] + var[itr]);
		}
		else if (leftlev < lev)
		{
			T vari, varn1, varn2;

			i = (0) + iy * 16 + ibl * (256);

			vari = var[i];

			if (iy == 0)
			{

				if (botofleftibl == leftibl) /// the bot left block does not exist
				{

					varn1 = var[15 + 0 * 16 + leftibl * 256];
					varval = (vari + 2 * varn1) / 3.0;
				}
				else if (leftofbotibl == leftibl) /// we are at the top
				{
					jj = 8;
					varn1 = var[15 + jj * 16 + leftibl * 256];
					varn2 = var[15 + (jj - 1) * 16 + leftibl * 256];
					varval = vari / 3.0 + 0.5 * varn1 + varn2 / 6.0;


				}
				else if (level[botofleftibl] == level[leftibl])
				{

					varn1 = var[15 + 0 * 16 + leftibl * 256];
					varn2 = var[15 + (15) * 16 + botofleftibl * 256];
					varval = vari / 3.0 + 0.5 * varn1 + varn2 / 6.0;
				}
				else if (level[botofleftibl] > level[leftibl])
				{

					varn1 = var[15 + 0 * 16 + leftibl * 256];
					varn2 = var[15 + (15) * 16 + botofleftibl * 256];
					varval = vari / 4.0 + 0.5 * varn1 + varn2 / 4.0;
				}
				else if (level[botofleftibl] < level[leftibl])
				{

					varn1 = var[15 + 0 * 16 + leftibl * 256];
					varn2 = var[15 + (15) * 16 + botofleftibl * 256];
					varval = vari * 0.4 + 0.5 * varn1 + varn2 * 0.1;
				}
			}
			else if (iy == 15)
			{
				if (topofleftibl == leftibl) /// the top right block does not exist
				{

					varn1 = var[15 + iy * 16 + leftibl * 256];
					varval = (vari + 2 * varn1) / 3.0;
				}
				else if (leftoftopibl == leftibl) /// we are at the bottom
				{

					varn1 = var[15 + 8 * 16 + leftibl * 256];
					varn2 = var[15 + 7 * 16 + leftibl * 256];
					varval = vari / 3.0 + varn1 / 6.0 + varn2 / 2.0;


				}
				else if (level[topofleftibl] == level[leftibl])
				{

					varn1 = var[15 + 0 * 16 + topofleftibl * 256];
					varn2 = var[15 + (15) * 16 + leftibl * 256];
					varval = vari / 3.0 + varn1 / 6.0 + varn2 / 2.0;
				}
				else if (level[topofleftibl] > level[leftibl])
				{

					varn1 = var[15 + 15 * 16 + leftibl * 256];
					varn2 = var[15 + (0) * 16 + topofleftibl * 256];
					varval = vari / 4.0 + 0.5 * varn1 + varn2 / 4.0;
				}
				else if (level[topofleftibl] < level[leftibl]) //bug? level[botofleftibl] < level[leftibl] or level[topofleftibl] < level[leftibl]
				{

					varn1 = var[15 + 15 * 16 + leftibl * 256];
					varn2 = var[15 + (0) * 16 + topofleftibl * 256];
					varval = vari * 0.4 + 0.5 * varn1 + varn2 * 0.1;
				}

			}
			else
			{
				T w0, w1, w2;
				jj = ceil(iy * 0.5);
				w0 = 1.0 / 3.0;

				if (jj * 2 > iy)
				{
					w1 = 1.0 / 6.0;
					w2 = 0.5;


				}
				else
				{
					w1 = 0.5;
					w2 = 1.0 / 6.0;
				}

				if (leftofbotibl == leftibl)
				{
					jj = jj + 8;
				}


				varn1 = var[15 + (jj) * 16 + leftibl * 256];
				varn2 = var[15 + (jj - 1) * 16 + leftibl * 256];
				varval = vari * w0 + varn1 * w1 + varn2 * w2;
			}



		}
	}

	return varval;
}



template<class T>
__host__ __device__ T TopAda(int ix, int iy, int ibl, int topibl, int topofrightibl, int topofleftibl, int leftoftopibl, int rightoftopibl, int* level, T* var)
{
	T varval;
	int i;
	int n1, n2, jj, bb;
	int lev = level[ibl];
	int toplev = level[topibl];



	if (iy < 15)
	{
		i =  ix + (iy + 1) * 16 + ibl * (256);// replace with blockdim.x  ...etc
		varval = var[i];
	}
	else
	{
		if (topibl == ibl) // No neighbour
		{
			i = ix + 15 * 16 + ibl * (256);
			varval = var[i];

		}
		else if (lev == toplev) // neighbour blk is same resolution
		{
			i = ix + (0) * 16 + topibl * (256);
			varval = var[i];
		}
		else if (toplev > lev)
		{
			int ii, ir, it, itr;


			if (ix < 8)
			{
				jj = ix * 2;
				bb = topibl;

			}
			if (ix >= 8)
			{
				jj = (ix - 8) * 2;
				bb = rightoftopibl;

			}
			ii = jj + 0 * 16 + bb * 256;
			it = jj + 1 * 16 + bb * 256;
			ir = (jj + 1) + 0 * 16 + bb * 256;
			itr = (jj + 1) + 1 * 16 + bb * 256;

			varval = T(0.25) * (var[ii] + var[ir] + var[it] + var[itr]);
		}
		else if (toplev < lev)
		{
			T vari, varn1, varn2;

			i = ix + iy * 16 + ibl * (256);

			vari = var[i];

			if (ix == 0)
			{

				if (leftoftopibl == topibl) // no neighbour
				{

					varn1 = var[0 + 0 * 16 + topibl * 256];
					varval = (vari + 2 * varn1) / 3.0;
				}
				else if (topofleftibl == topibl) 
				{
					jj = 8;
					varn1 = var[jj + 0 * 16 + topibl * 256];
					varn2 = var[(jj - 1) + 0 * 16 + topibl * 256];
					varval = vari / 3.0 + 0.5 * varn1 + varn2 / 6.0;


				}
				else if (level[topofleftibl] == level[topibl])
				{

					varn1 = var[0 + 0 * 16 + topibl * 256];
					varn2 = var[15 + (0) * 16 + topofleftibl * 256];
					varval = vari / 3.0 + 0.5 * varn1 + varn2 / 6.0;
				}
				else if (level[topofleftibl] > level[topibl])
				{

					varn1 = var[0 + 0 * 16 + topibl * 256];
					varn2 = var[15 + (0) * 16 + topofleftibl * 256];
					varval = vari / 4.0 + 0.5 * varn1 + varn2 / 4.0;
				}
				else if (level[topofleftibl] < level[topibl])
				{

					varn1 = var[0 + 0 * 16 + topibl * 256];
					varn2 = var[15 + (0) * 16 + topofleftibl * 256];
					varval = vari * 0.4 + 0.5 * varn1 + varn2 * 0.1;
				}
			}
			else if (ix == 15)
			{
				if (rightoftopibl == topibl) /// the top right block does not exist
				{

					varn1 = var[15 + 0 * 16 + topibl * 256];
					varval = (vari + 2 * varn1) / 3.0;
				}
				else if (topofrightibl == topibl) 
				{

					varn1 = var[8 + 0 * 16 + topibl * 256];
					varn2 = var[7 + 0 * 16 + topibl * 256];
					varval = vari / 3.0 + varn1 / 6.0 + varn2 / 2.0;


				}
				else if (level[topofrightibl] == level[topibl])
				{

					varn1 = var[0 + 0 * 16 + topofrightibl * 256];
					varn2 = var[15 + (0) * 16 + topibl * 256];
					varval = vari / 3.0 + varn1 / 6.0 + varn2 / 2.0;
				}
				else if (level[topofrightibl] > level[topibl])
				{

					varn1 = var[15 + 0 * 16 + topibl * 256];
					varn2 = var[0 + (0) * 16 + topofrightibl * 256];
					varval = vari / 4.0 + 0.5 * varn1 + varn2 / 4.0;
				}
				else if (level[topofrightibl] < level[topibl])
				{

					varn1 = var[0 + 0 * 16 + topibl * 256];
					varn2 = var[15 + (0) * 16 + topofrightibl * 256];
					varval = vari * 0.4 + 0.5 * varn1 + varn2 * 0.1;
				}

			}
			else
			{
				T w0, w1, w2;
				jj = ceil(ix * 0.5);
				w0 = 1.0 / 3.0;

				if (jj * 2 > ix)
				{
					w1 = 1.0 / 6.0;
					w2 = 0.5;


				}
				else
				{
					w1 = 0.5;
					w2 = 1.0 / 6.0;
				}

				if (topofleftibl == topibl)
				{
					jj = jj + 8;
				}


				varn1 = var[jj + (0) * 16 + topibl * 256];
				varn2 = var[(jj - 1) + 0 * 16 + topibl * 256];
				varval = vari * w0 + varn1 * w1 + varn2 * w2;
			}



		}
	}

	return varval;
}

template<class T>
__host__ __device__ T BotAda(int ix, int iy, int ibl, int botibl, int botofrightibl, int botofleftibl, int leftofbotibl, int rightofbotibl, int* level, T* var)
{
	T varval;
	int i;
	int n1, n2, jj, bb;
	int lev = level[ibl];
	int botlev = level[botibl];

	// replace with blockdim.x  ...etc for some values... This will need a CUA_ARCH if statement

	if (iy > 0)
	{
		i = ix + (iy - 1) * 16 + ibl * (256);// replace with blockdim.x  ...etc
		varval = var[i];
	}
	else
	{
		if (botibl == ibl) // No neighbour
		{
			i = ix + 0 * 16 + ibl * (256);
			varval = var[i];

		}
		else if (lev == botlev) // neighbour blk is same resolution
		{
			i = ix + (15) * 16 + botibl * (256);
			varval = var[i];
		}
		else if (botlev > lev)
		{
			int ii, ir, it, itr;


			if (ix < 8)
			{
				jj = ix * 2;
				bb = botibl;

			}
			if (ix >= 8)
			{
				jj = (ix - 8) * 2;
				bb = rightofbotibl;

			}
			ii = jj + 15 * 16 + bb * 256;
			it = jj + 14 * 16 + bb * 256;
			ir = (jj + 1) + 15 * 16 + bb * 256;
			itr = (jj + 1) + 14 * 16 + bb * 256;

			varval = T(0.25) * (var[ii] + var[ir] + var[it] + var[itr]);
		}
		else if (botlev < lev)
		{
			T vari, varn1, varn2;

			i = ix + iy * 16 + ibl * (256);

			vari = var[i];

			if (ix == 0)
			{

				if (leftofbotibl == botibl) // no neighbour
				{

					varn1 = var[0 + 0 * 16 + botibl * 256];
					varval = (vari + 2 * varn1) / 3.0;
				}
				else if (botofleftibl == botibl)
				{
					jj = 8;
					varn1 = var[jj + 15 * 16 + botibl * 256];
					varn2 = var[(jj - 1) + 15 * 16 + botibl * 256];
					varval = vari / 3.0 + 0.5 * varn1 + varn2 / 6.0;


				}
				else if (level[botofleftibl] == level[botibl])
				{

					varn1 = var[0 + 15 * 16 + botibl * 256];
					varn2 = var[15 + (15) * 16 + botofleftibl * 256];
					varval = vari / 3.0 + 0.5 * varn1 + varn2 / 6.0;
				}
				else if (level[botofleftibl] > level[botibl])
				{

					varn1 = var[0 + 15 * 16 + botibl * 256];
					varn2 = var[15 + (15) * 16 + botofleftibl * 256];
					varval = vari / 4.0 + 0.5 * varn1 + varn2 / 4.0;
				}
				else if (level[botofleftibl] < level[botibl])
				{

					varn1 = var[0 + 15 * 16 + botibl * 256];
					varn2 = var[15 + (15) * 16 + botofleftibl * 256];
					varval = vari * 0.4 + 0.5 * varn1 + varn2 * 0.1;
				}
			}
			else if (ix == 15)
			{
				if (rightofbotibl == botibl) /// the top right block does not exist
				{

					varn1 = var[15 + 15 * 16 + botibl * 256];
					varval = (vari + 2 * varn1) / 3.0;
				}
				else if (botofrightibl == botibl)
				{

					varn1 = var[8 + 15 * 16 + botibl * 256];
					varn2 = var[7 + 15 * 16 + botibl * 256];
					varval = vari / 3.0 + varn1 / 6.0 + varn2 / 2.0;


				}
				else if (level[botofrightibl] == level[botibl])
				{

					varn1 = var[0 + 15 * 16 + botofrightibl * 256];
					varn2 = var[15 + (15) * 16 + botibl * 256];
					varval = vari / 3.0 + varn1 / 6.0 + varn2 / 2.0;
				}
				else if (level[botofrightibl] > level[botibl])
				{

					varn1 = var[15 + 15 * 16 + botibl * 256];
					varn2 = var[0 + (15) * 16 + botofrightibl * 256];
					varval = vari / 4.0 + 0.5 * varn1 + varn2 / 4.0;
				}
				else if (level[botofrightibl] < level[botibl])
				{

					varn1 = var[0 + 15 * 16 + botibl * 256];
					varn2 = var[15 + (15) * 16 + botofrightibl * 256];
					varval = vari * 0.4 + 0.5 * varn1 + varn2 * 0.1;
				}

			}
			else
			{
				T w0, w1, w2;
				jj = ceil(ix * 0.5);
				w0 = 1.0 / 3.0;

				if (jj * 2 > ix)
				{
					w1 = 1.0 / 6.0;
					w2 = 0.5;


				}
				else
				{
					w1 = 0.5;
					w2 = 1.0 / 6.0;
				}

				if (botofleftibl == botibl)
				{
					jj = jj + 8;
				}


				varn1 = var[jj + (15) * 16 + botibl * 256];
				varn2 = var[(jj - 1) + 15 * 16 + botibl * 256];
				varval = vari * w0 + varn1 * w1 + varn2 * w2;
			}



		}
	}

	return varval;
}




template <class T> __global__ void gradientGPUXYBUQADASM(T theta, T dx, int * Activeblk, int * lev, int* leftblk, int* rightblk, int* topblk, int* botblk, T* a, T* dadx, T* dady)
{
	//int *leftblk,int *rightblk,int* topblk, int * botblk,

	//int ix = threadIdx.x+1;
	//int iy = threadIdx.y+1;
	int ix = threadIdx.x;
	int iy = threadIdx.y;
	int ib = Activeblk[blockIdx.x];

	// shared array index to make the code bit more readable
	int sx = ix + 1;
	int sy = iy + 1;

	T delta = calcres(dx, lev[ib]);

	int i = ix + iy * blockDim.x + ib * (blockDim.x * blockDim.y);




	int ileft, iright, itop, ibot;




	__shared__ T a_s[18][18];




	a_s[sx][sy] = a[i];
	//__syncthreads;
	//syncthread is needed here ?


	// read the halo around the tile
	if (threadIdx.x == blockDim.x - 1)
	{
		//iright = findrightGSM(ix, iy, rightblk[ibl], ibl, blockDim.x);
		a_s[sx + 1][sy] = RightAda(ix, iy, ib, rightblk[ib], rightblk[botblk[ib]], rightblk[topblk[ib]], botblk[rightblk[ib]], topblk[rightblk[ib]], lev, a);
	}


	if (threadIdx.x == 0)
	{
		//ileft = findleftGSM(ix, iy, leftblk[ibl], ibl, blockDim.x);
		a_s[sx - 1][sy] = LeftAda(ix, iy, ib, leftblk[ib], leftblk[botblk[ib]], leftblk[topblk[ib]], botblk[leftblk[ib]], topblk[leftblk[ib]], lev, a);

	}


	if (threadIdx.y == blockDim.y - 1)
	{
		//itop = findtopGSM(ix, iy, topblk[ibl], ibl, blockDim.x);
		a_s[sx][sy + 1] = TopAda(ix, iy, ib, topblk[ib], topblk[rightblk[ib]], topblk[leftblk[ib]], leftblk[topblk[ib]], rightblk[topblk[ib]], lev, a);
	}

	if (threadIdx.y == 0)
	{
		//ibot = findbotGSM(ix, iy, botblk[ibl], ibl, blockDim.x);
		a_s[sx][sy - 1] = BotAda(ix, iy, ib, botblk[ib], botblk[rightblk[ib]], botblk[leftblk[ib]], leftblk[botblk[ib]], rightblk[botblk[ib]], lev, a);
	}

	__syncthreads;
	/*
	a_i = a[i];
	a_r = a[iright];
	a_l = a[ileft];
	a_t = a[itop];
	a_b = a[ibot];
	*/

	dadx[i] = minmod2GPU(theta, a_s[sx - 1][sy], a_s[sx][sy], a_s[sx + 1][sy]) / delta;
	dady[i] = minmod2GPU(theta, a_s[sx][sy - 1], a_s[sx][sy], a_s[sx][sy + 1]) / delta;
	/*
	dadx[i] = minmod2GPU(theta, a_l, a_i, a_r) / delta;
	dady[i] = minmod2GPU(theta, a_b, a_i, a_t) / delta;
	*/

}

template <class T> void gradientADA(int nblk, int blksize, T theta, T dx, int * activeblk, int *level, int * leftblk, int * rightblk, int * topblk, int * botblk, T *a, T *&dadx, T * &dady)
{

	int i, ib, lev;
	T axplus, ayplus, axminus, ayminus;


	for (int ibl = 0; ibl < nblk; ibl++)
	{
		ib = activeblk[ibl];
		lev = level[ib];
		T delta = calcres(dx, lev);

		for (int iy = 0; iy < 16; iy++)
		{
			for (int ix = 0; ix < 16; ix++)
			{
				i = ix + iy * 16 + ib * blksize;
				//
				//
				axplus = RightAda(ix, iy, ib, rightblk[ib], rightblk[botblk[ib]], rightblk[topblk[ib]], botblk[rightblk[ib]], topblk[rightblk[ib]], level, a);

				axminus = LeftAda(ix, iy, ib, leftblk[ib], leftblk[botblk[ib]], leftblk[topblk[ib]], botblk[leftblk[ib]], topblk[leftblk[ib]], level, a);
				ayplus = TopAda(ix, iy, ib, topblk[ib], topblk[rightblk[ib]], topblk[leftblk[ib]], leftblk[topblk[ib]], rightblk[topblk[ib]], level, a);
				ayminus = BotAda(ix, iy, ib, botblk[ib], botblk[rightblk[ib]], botblk[leftblk[ib]], leftblk[botblk[ib]], rightblk[botblk[ib]], level, a);



				//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
				dadx[i] = minmod2GPU(theta, axminus, a[i], axplus) / delta;
				//dady[i] = (a[i] - a[ix + yminus*nx]) / delta;
				dady[i] = minmod2GPU(theta, ayminus, a[i], ayplus) / delta;
			}


		}
	}

}
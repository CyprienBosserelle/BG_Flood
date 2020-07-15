//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//Copyright (C) 2018 Bosserelle                                                 //
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


#include "Mesh.h"

int CalcInitnblk(Param XParam, Forcing<float> XForcing)
{

	////////////////////////////////////////////////
	// Rearrange the memory in uniform blocks
	////////////////////////////////////////////////


	//max nb of blocks is ceil(nx/16)*ceil(ny/16)
	int nblk = 0;
	int nmask = 0;
	int mloc = 0;

	double levdx = calcres(XParam.dx, XParam.initlevel);

	for (int nblky = 0; nblky < ceil(XForcing.Bathy.ny / XParam.blkwidth); nblky++)
	{
		for (int nblkx = 0; nblkx < ceil(XForcing.Bathy.nx / XParam.blkwidth); nblkx++)
		{
			nmask = 0;
			for (int i = 0; i < 16; i++)
			{
				for (int j = 0; j < 16; j++)
				{
					double x = XParam.xo + (i + XParam.blkwidth * nblkx) * levdx;
					double y = XParam.yo + (j + XParam.blkwidth * nblky) * levdx;

					if (x >= XForcing.Bathy.xo && x <= XForcing.Bathy.xmax && y >= XForcing.Bathy.yo && y <= XForcing.Bathy.ymax)
					{
						// cells that falls off this domain are assigned
						double x1, x2, y1, y2;
						double q11, q12, q21, q22, q;
						int cfi, cfip, cfj, cfjp;



						cfi = utils::min(utils::max((int)floor((x - XForcing.Bathy.xo) / XForcing.Bathy.dx), 0), XForcing.Bathy.nx - 2);
						cfip = cfi + 1;

						x1 = XForcing.Bathy.xo + XForcing.Bathy.dx * cfi;
						x2 = XForcing.Bathy.xo + XForcing.Bathy.dx * cfip;

						cfj = utils::min(utils::max((int)floor((y - XForcing.Bathy.yo) / XForcing.Bathy.dx), 0), XForcing.Bathy.ny - 2);
						cfjp = cfj + 1;

						y1 = XForcing.Bathy.yo + XForcing.Bathy.dx * cfj;
						y2 = XForcing.Bathy.yo + XForcing.Bathy.dx * cfjp;

						q11 = XForcing.Bathy.val[cfi + cfj * XForcing.Bathy.nx];
						q12 = XForcing.Bathy.val[cfi + cfjp * XForcing.Bathy.nx];
						q21 = XForcing.Bathy.val[cfip + cfj * XForcing.Bathy.nx];
						q22 = XForcing.Bathy.val[cfip + cfjp * XForcing.Bathy.nx];

						q = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
						//printf("q = %f\n", q);
						//printf("mloc: %i\n", mloc);
						if (q >= XParam.mask)
							nmask++;
					}
					else
					{
						//computational domnain is outside of the bathy domain
						nmask++;
					}

				}
			}
			if (nmask < XParam.blksize)
				nblk++;
		}
	}
}

void InitMesh(Param &XParam, Forcing<float> XForcing)
{
	int nblk;

	nblk= CalcInitnblk(XParam, XForcing);
	XParam.nblk = nblk;

	XParam.nblkmem = (int)ceil(nblk * XParam.membuffer); //5% buffer on the memory for adaptation 

	
	
}
	

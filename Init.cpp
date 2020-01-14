//////////////////////////////////////////////////////////////////////////////////
//						                                                        //
//Copyright (C) 2018 Bosserelle                                                 //
// This code contains an adaptation of the St Venant equation from Basilisk		//
// See																			//
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



unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

template <class T> void Allocate1CPU(int nx, int ny, T *&zb)
{
	zb = (T *)malloc(nx*ny * sizeof(T));
	if (!zb)
	{
		fprintf(stderr, "Memory allocation failure\n");

		exit(EXIT_FAILURE);
	}
}

template <class T> void Allocate4CPU(int nx, int ny, T *&zs, T *&hh, T *&uu, T *&vv)
{

	zs = (T *)malloc(nx*ny * sizeof(T));
	hh = (T *)malloc(nx*ny * sizeof(T));
	uu = (T *)malloc(nx*ny * sizeof(T));
	vv = (T *)malloc(nx*ny * sizeof(T));

	if (!zs || !hh || !uu || !vv)
	{
		fprintf(stderr, "Memory allocation failure\n");

		exit(EXIT_FAILURE);
	}
}


template <class T> void InitArraySV(int nblk, int blksize, T initval, T * & Arr)
{
	//inititiallise array with a single value
	for (int bl = 0; bl < nblk; bl++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int i = 0; i < 16; i++)
			{
				int n = i + j * 16 + bl * blksize;
				Arr[n] = initval;
			}
		}
	}
}

template <class T> void CopyArray(int nblk, int blksize, T* source, T * & dest)
{
	//
	for (int bl = 0; bl < nblk; bl++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int i = 0; i < 16; i++)
			{
				int n = i + j * 16 + bl * blksize;
				dest[n] = source[n];
			}
		}
	}
}

template <class T>  void setedges(int nblk, int * leftblk, int *rightblk, int * topblk, int* botblk, T *&zb)
{
	// template <class T> void setedges(int nblk, int nx, int ny, double xo, double yo, double dx, int * leftblk, int *rightblk, int * topblk, int* botblk, double *blockxo, double * blockyo, T *&zb)

	// here the bathy of the outter most cells of the domain are "set" to the same value as the second outter most.
	// this also applies to the blocks with no neighbour
	for (int bl = 0; bl < nblk; bl++)
	{

		if (bl == leftblk[bl])//i.e. if a block refers to as it's onwn neighbour then it doesn't have a neighbour/// This also applies to block that are on the edge of the grid so the above is commentted
		{
			int i = 0;
			for (int j = 0; j < 16; j++)
			{

				zb[i + j * 16 + bl * 256] = zb[i + 1 + j * 16 + bl * 256];
			}
		}
		if (bl == rightblk[bl])
		{
			int i = 15;
			for (int j = 0; j < 16; j++)
			{

				zb[i + j * 16 + bl * 256] = zb[i - 1 + j * 16 + bl * 256];
			}
		}
		if (bl == topblk[bl])
		{
			int j = 15;
			for (int i = 0; i < 16; i++)
			{

				zb[i + j * 16 + bl * 256] = zb[i + (j - 1) * 16 + bl * 256];
			}
		}
		if (bl == botblk[bl])
		{
			int j = 0;
			for (int i = 0; i < 16; i++)
			{

				zb[i + j * 16 + bl * 256] = zb[i + (j + 1) * 16 + bl * 256];
			}
		}

	}
}

template <class T> void carttoBUQ(int nblk, int nx, int ny, double xo, double yo, double dx, double* blockxo, double* blockyo, T * zb, T *&zb_buq)
{
	//
	int ix, iy;
	T x, y;
	for (int b = 0; b < nblk; b++)
	{

		for (int i = 0; i < 16; i++)
		{
			for (int j = 0; j < 16; j++)
			{
				x = blockxo[b] + i*dx;
				y = blockyo[b] + j*dx;
				ix = min(max((int)round((x - xo) / dx), 0), nx - 1); // min(max( part is overkill?
				iy = min(max((int)round((y - yo) / dx), 0), ny - 1);

				zb_buq[i + j * 16 + b * 256] = zb[ix + iy*nx];
				//printf("bid=%i\ti=%i\tj=%i\tix=%i\tiy=%i\tzb_buq[n]=%f\n", b,i,j,ix, iy, zb_buq[i + j * 16 + b * 256]);
			}
		}
	}
}

template <class T> void interp2BUQ(int nblk, double blksize, double blkdx, double* blockxo, double* blockyo, int nx, int ny, double xo, double xmax, double yo, double ymax, double dx, T * zb, T *&zb_buq)
{
	// This function interpolates the values in bathy maps or roughness map to cf using a bilinear interpolation

	double x, y;
	int n;

	for (int bl = 0; bl < nblk; bl++)
	{
		//printf("bl=%d\tblockxo[bl]=%f\tblockyo[bl]=%f\n", bl, blockxo[bl], blockyo[bl]);
		for (int j = 0; j < 16; j++)
		{
			for (int i = 0; i < 16; i++)
			{
				n = i + j * 16 + bl * blksize;
				x = blockxo[bl] + i*blkdx;
				y = blockyo[bl] + j*blkdx;

				//if (x >= xo && x <= xmax && y >= yo && y <= ymax)
				{
					//this is safer!
					x = max(min(x, xmax), xo);
					y = max(min(y, ymax), yo);
					// cells that falls off this domain are assigned 
					double x1, x2, y1, y2;
					double q11, q12, q21, q22;
					int cfi, cfip, cfj, cfjp;



					cfi = min(max((int)floor((x - xo) / dx), 0), nx - 2);
					cfip = cfi + 1;

					x1 = xo + dx*cfi;
					x2 = xo + dx*cfip;

					cfj = min(max((int)floor((y - yo) / dx), 0), ny - 2);
					cfjp = cfj + 1;

					y1 = yo + dx*cfj;
					y2 = yo + dx*cfjp;

					q11 = zb[cfi + cfj*nx];
					q12 = zb[cfi + cfjp*nx];
					q21 = zb[cfip + cfj*nx];
					q22 = zb[cfip + cfjp*nx];

					zb_buq[n] = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
					//printf("x=%f\ty=%f\tcfi=%d\tcfj=%d\tn=%d\tzb_buq[n] = %f\n", x,y,cfi,cfj,n,zb_buq[n]);
				}

			}
		}
	}
}

template <class T> void interp2cf(Param XParam, float * cfin, T* blockxo, T* blockyo, T * &cf)
{
	// This function interpolates the values in cfmapin to cf using a bilinear interpolation

	double x, y;
	int n;

	for (int bl = 0; bl < XParam.nblk; bl++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int i = 0; i < 16; i++)
			{
				n = i + j * 16 + bl * XParam.blksize;

				x = blockxo[bl] + i*XParam.dx;
				y = blockyo[bl] + j*XParam.dx;

				if (x >= XParam.roughnessmap.xo && x <= XParam.roughnessmap.xmax && y >= XParam.roughnessmap.yo && y <= XParam.roughnessmap.ymax)
				{
					// cells that falls off this domain are assigned 
					double x1, x2, y1, y2;
					double q11, q12, q21, q22;
					int cfi, cfip, cfj, cfjp;



					cfi = min(max((int)floor((x - XParam.roughnessmap.xo) / XParam.roughnessmap.dx), 0), XParam.roughnessmap.nx - 2);
					cfip = cfi + 1;

					x1 = XParam.roughnessmap.xo + XParam.roughnessmap.dx*cfi;
					x2 = XParam.roughnessmap.xo + XParam.roughnessmap.dx*cfip;

					cfj = min(max((int)floor((y - XParam.roughnessmap.yo) / XParam.roughnessmap.dx), 0), XParam.roughnessmap.ny - 2);
					cfjp = cfj + 1;

					y1 = XParam.roughnessmap.yo + XParam.roughnessmap.dx*cfj;
					y2 = XParam.roughnessmap.yo + XParam.roughnessmap.dx*cfjp;

					q11 = cfin[cfi + cfj*XParam.roughnessmap.nx];
					q12 = cfin[cfi + cfjp*XParam.roughnessmap.nx];
					q21 = cfin[cfip + cfj*XParam.roughnessmap.nx];
					q22 = cfin[cfip + cfjp*XParam.roughnessmap.nx];

					cf[n] = BilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
				}

			}
		}
	}
}

float maxdiff(int nxny, float * ref, float * pred)
{
	float maxd = 0.0f;
	for (int i = 0; i < nxny; i++)
	{
		maxd = max(abs(pred[i] - ref[i]), maxd);
	}
	return maxd;
}

float maxdiffID(int nx, int ny, int &im, int &jm, float * ref, float * pred)
{
	float maxd = 0.0f;

	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			if (abs(pred[i] - ref[i]) > maxd)
			{
				im = i;
				jm = j;
				maxd = abs(pred[i] - ref[i]);
			}
		}
	}
	return maxd;
}


int AllocMemCPU(Param XParam)
{
	//function to allocate the memory on the CPU
	// Pointers are Global !
	//Need to add a sucess check for each call to malloc

	int nblk = XParam.nblk;
	int blksize = XParam.blksize;


	if (XParam.doubleprecision == 1 || XParam.spherical == 1)
	{
		//allocate double *arrays
		Allocate1CPU(nblk, blksize, zb_d);
		Allocate4CPU(nblk, blksize, zs_d, hh_d, uu_d, vv_d);
		Allocate4CPU(nblk, blksize, zso_d, hho_d, uuo_d, vvo_d);
		Allocate4CPU(nblk, blksize, dzsdx_d, dhdx_d, dudx_d, dvdx_d);
		Allocate4CPU(nblk, blksize, dzsdy_d, dhdy_d, dudy_d, dvdy_d);

		Allocate4CPU(nblk, blksize, Su_d, Sv_d, Fhu_d, Fhv_d);
		Allocate4CPU(nblk, blksize, Fqux_d, Fquy_d, Fqvx_d, Fqvy_d);

		//Allocate4CPU(nblk, blksize, dh_d, dhu_d, dhv_d, dummy_d);
		Allocate1CPU(nblk, blksize, dh_d);
		Allocate1CPU(nblk, blksize, dhu_d);
		Allocate1CPU(nblk, blksize, dhv_d);

		Allocate1CPU(nblk, blksize, cf_d);


		//not allocating below may be usefull

		if (XParam.outhhmax == 1)
		{
			Allocate1CPU(nblk, blksize, hhmax_d);
		}
		if (XParam.outuumax == 1)
		{
			Allocate1CPU(nblk, blksize, uumax_d);
		}
		if (XParam.outvvmax == 1)
		{
			Allocate1CPU(nblk, blksize, vvmax_d);
		}
		if (XParam.outzsmax == 1)
		{
			Allocate1CPU(nblk, blksize, zsmax_d);
		}

		if (XParam.outhhmean == 1)
		{
			Allocate1CPU(nblk, blksize, hhmean_d);
		}
		if (XParam.outzsmean == 1)
		{
			Allocate1CPU(nblk, blksize, zsmean_d);
		}
		if (XParam.outuumean == 1)
		{
			Allocate1CPU(nblk, blksize, uumean_d);
		}
		if (XParam.outvvmean == 1)
		{
			Allocate1CPU(nblk, blksize, vvmean_d);
		}

		if (XParam.outvort == 1)
		{
			Allocate1CPU(nblk, blksize, vort);
		}

	}
	else
	{
		// allocate float *arrays (same template functions but different pointers)
		Allocate1CPU(nblk, blksize, zb);
		Allocate4CPU(nblk, blksize, zs, hh, uu, vv);
		Allocate4CPU(nblk, blksize, zso, hho, uuo, vvo);
		Allocate4CPU(nblk, blksize, dzsdx, dhdx, dudx, dvdx);
		Allocate4CPU(nblk, blksize, dzsdy, dhdy, dudy, dvdy);

		Allocate4CPU(nblk, blksize, Su, Sv, Fhu, Fhv);
		Allocate4CPU(nblk, blksize, Fqux, Fquy, Fqvx, Fqvy);

		//Allocate4CPU(nx, ny, dh, dhu, dhv, dummy);
		Allocate1CPU(nblk, blksize, dh);
		Allocate1CPU(nblk, blksize, dhu);
		Allocate1CPU(nblk, blksize, dhv);
		Allocate1CPU(nblk, blksize, cf);
		//not allocating below may be usefull

		if (XParam.outhhmax == 1)
		{
			Allocate1CPU(nblk, blksize, hhmax);
		}
		if (XParam.outuumax == 1)
		{
			Allocate1CPU(nblk, blksize, uumax);
		}
		if (XParam.outvvmax == 1)
		{
			Allocate1CPU(nblk, blksize, vvmax);
		}
		if (XParam.outzsmax == 1)
		{
			Allocate1CPU(nblk, blksize, zsmax);
		}

		if (XParam.outhhmean == 1)
		{
			Allocate1CPU(nblk, blksize, hhmean);
		}
		if (XParam.outzsmean == 1)
		{
			Allocate1CPU(nblk, blksize, zsmean);
		}
		if (XParam.outuumean == 1)
		{
			Allocate1CPU(nblk, blksize, uumean);
		}
		if (XParam.outvvmean == 1)
		{
			Allocate1CPU(nblk, blksize, vvmean);
		}

		if (XParam.outvort == 1)
		{
			Allocate1CPU(nblk, blksize, vort);
		}

	}
	return 1; //Need a real test here
}



template <class T>
int coldstart(Param XParam, T*zb, T *&uu, T*&vv, T*&zs, T*&hh)
{
	int coldstartsucess = 0;
	for (int bl = 0; bl < XParam.nblk; bl++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int i = 0; i < 16; i++)
			{
				int n = i + j * 16 + bl * XParam.blksize;

				uu[n] = T(0.0);
				vv[n] = T(0.0);
				//zb[n] = 0.0f;
				zs[n] = max(XParam.zsinit+XParam.zsoffset, zb[n]);
				//if (i >= 64 && i < 82)
				//{
				//	zs[n] = max(zsbnd+0.2f, zb[i + j*nx]);
				//}
				hh[n] = max(zs[n] - zb[n], XParam.eps);//0.0?

			}

		}
	}
	coldstartsucess = 1;
	return coldstartsucess = 1;
}

template <class T>
void warmstart(Param XParam, T*zb, T *&uu, T*&vv, T*&zs, T*&hh)
{
	double zsleft = 0.0;
	double zsright = 0.0;
	double zstop = 0.0;
	double zsbot = 0.0;
	T zsbnd = 0.0;

	double distleft, distright, disttop, distbot;

	double lefthere = 0.0;
	double righthere = 0.0;
	double tophere = 0.0;
	double bothere = 0.0;

	double xi, yi, jj, ii;

	for (int bl = 0; bl < XParam.nblk; bl++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int i = 0; i < 16; i++)
			{
				int n = i + j * 16 + bl * XParam.blksize;
				xi = blockxo_d[bl] + i*XParam.dx;
				yi = blockyo_d[bl] + j*XParam.dx;

				disttop = max((XParam.ymax - yi) / XParam.dx, 0.1);//max((double)(ny - 1) - j, 0.1);// WTF is that 0.1? // distleft cannot be 0 //theoretical minumun is 0.5?
				distbot = max((yi - XParam.yo) / XParam.dx, 0.1);
				distleft = max((xi - XParam.xo) / XParam.dx, 0.1);//max((double)i, 0.1);
				distright = max((XParam.xmax - xi) / XParam.dx, 0.1);//max((double)(nx - 1) - i, 0.1);

				jj = (yi - XParam.yo) / (XParam.ymax - XParam.yo);
				ii = (xi - XParam.xo) / (XParam.xmax - XParam.xo);

				if (XParam.leftbnd.on)
				{
					lefthere = 1.0;
					int SLstepinbnd = 1;



					// Do this for all the corners
					//Needs limiter in case WLbnd is empty
					double difft = XParam.leftbnd.data[SLstepinbnd].time - XParam.totaltime;

					while (difft < 0.0)
					{
						SLstepinbnd++;
						difft = XParam.leftbnd.data[SLstepinbnd].time - XParam.totaltime;
					}
					std::vector<double> zsbndvec;
					for (int n = 0; n < XParam.leftbnd.data[SLstepinbnd].wlevs.size(); n++)
					{
						zsbndvec.push_back(interptime(XParam.leftbnd.data[SLstepinbnd].wlevs[n], XParam.leftbnd.data[SLstepinbnd - 1].wlevs[n], XParam.leftbnd.data[SLstepinbnd].time - XParam.leftbnd.data[SLstepinbnd - 1].time, XParam.totaltime - XParam.leftbnd.data[SLstepinbnd - 1].time));

					}
					if (zsbndvec.size() == 1)
					{
						zsleft = zsbndvec[0];
					}
					else
					{
						int iprev = min(max((int)floor(jj * (zsbndvec.size() - 1)), 0), (int)zsbndvec.size() - 2);
						int inext = iprev + 1;
						// here interp time is used to interpolate to the right node rather than in time...
						zsleft = interptime(zsbndvec[inext], zsbndvec[iprev], 1.0, (double)(jj * (zsbndvec.size() - 1) - iprev));
					}

				}

				if (XParam.rightbnd.on)
				{
					int SLstepinbnd = 1;
					righthere = 1.0;


					// Do this for all the corners
					//Needs limiter in case WLbnd is empty
					double difft = XParam.rightbnd.data[SLstepinbnd].time - XParam.totaltime;

					while (difft < 0.0)
					{
						SLstepinbnd++;
						difft = XParam.rightbnd.data[SLstepinbnd].time - XParam.totaltime;
					}
					std::vector<double> zsbndvec;
					for (int n = 0; n < XParam.rightbnd.data[SLstepinbnd].wlevs.size(); n++)
					{
						zsbndvec.push_back(interptime(XParam.rightbnd.data[SLstepinbnd].wlevs[n], XParam.rightbnd.data[SLstepinbnd - 1].wlevs[n], XParam.rightbnd.data[SLstepinbnd].time - XParam.rightbnd.data[SLstepinbnd - 1].time, XParam.totaltime - XParam.rightbnd.data[SLstepinbnd - 1].time));

					}
					if (zsbndvec.size() == 1)
					{
						zsright = zsbndvec[0];
					}
					else
					{
						int iprev = min(max((int)floor(jj * (zsbndvec.size() - 1)), 0), (int)zsbndvec.size() - 2);
						int inext = iprev + 1;
						// here interp time is used to interpolate to the right node rather than in time...
						zsright = interptime(zsbndvec[inext], zsbndvec[iprev], 1.0, (double)(jj * (zsbndvec.size() - 1) - iprev));
					}


				}
				if (XParam.botbnd.on)
				{
					int SLstepinbnd = 1;
					bothere = 1.0;




					// Do this for all the corners
					//Needs limiter in case WLbnd is empty
					double difft = XParam.botbnd.data[SLstepinbnd].time - XParam.totaltime;

					while (difft < 0.0)
					{
						SLstepinbnd++;
						difft = XParam.botbnd.data[SLstepinbnd].time - XParam.totaltime;
					}
					std::vector<double> zsbndvec;
					for (int n = 0; n < XParam.botbnd.data[SLstepinbnd].wlevs.size(); n++)
					{
						zsbndvec.push_back(interptime(XParam.botbnd.data[SLstepinbnd].wlevs[n], XParam.botbnd.data[SLstepinbnd - 1].wlevs[n], XParam.botbnd.data[SLstepinbnd].time - XParam.botbnd.data[SLstepinbnd - 1].time, XParam.totaltime - XParam.botbnd.data[SLstepinbnd - 1].time));

					}
					if (zsbndvec.size() == 1)
					{
						zsbot = zsbndvec[0];
					}
					else
					{
						int iprev = min(max((int)floor(ii * (zsbndvec.size() - 1)), 0), (int)zsbndvec.size() - 2);
						int inext = iprev + 1;
						// here interp time is used to interpolate to the right node rather than in time...
						zsbot = interptime(zsbndvec[inext], zsbndvec[iprev], 1.0, (double)(ii * (zsbndvec.size() - 1) - iprev));
					}

				}
				if (XParam.topbnd.on)
				{
					int SLstepinbnd = 1;
					tophere = 1.0;




					// Do this for all the corners
					//Needs limiter in case WLbnd is empty
					double difft = XParam.topbnd.data[SLstepinbnd].time - XParam.totaltime;

					while (difft < 0.0)
					{
						SLstepinbnd++;
						difft = XParam.topbnd.data[SLstepinbnd].time - XParam.totaltime;
					}
					std::vector<double> zsbndvec;
					for (int n = 0; n < XParam.topbnd.data[SLstepinbnd].wlevs.size(); n++)
					{
						zsbndvec.push_back(interptime(XParam.topbnd.data[SLstepinbnd].wlevs[n], XParam.topbnd.data[SLstepinbnd - 1].wlevs[n], XParam.topbnd.data[SLstepinbnd].time - XParam.topbnd.data[SLstepinbnd - 1].time, XParam.totaltime - XParam.topbnd.data[SLstepinbnd - 1].time));

					}
					if (zsbndvec.size() == 1)
					{
						zstop = zsbndvec[0];
					}
					else
					{
						int iprev = min(max((int)floor(ii * (zsbndvec.size() - 1)), 0), (int)zsbndvec.size() - 2);
						int inext = iprev + 1;
						// here interp time is used to interpolate to the right node rather than in time...
						zstop = interptime(zsbndvec[inext], zsbndvec[iprev], 1.0, (double)(ii * (zsbndvec.size() - 1) - iprev));
					}

				}


				zsbnd = ((zsleft / distleft)*lefthere + (zsright  / distright)*righthere + (zstop / disttop)*tophere + (zsbot / distbot)*bothere) / ((1.0 / distleft)*lefthere + (1.0 / distright)*righthere + (1.0 / disttop)*tophere + (1.0 / distbot)*bothere);



				zs[n] = max(zsbnd, zb[n]);
				hh[n] = max(zs[n] - zb[n], T(XParam.eps));
				uu[n] = T(0.0);
				vv[n] = T(0.0);



			}
		}
	}
}

template <class T>
int AddZSoffset(Param XParam, T*&zb,  T*& zs, T*& hh)
{
	int success = 1;
	for (int bl = 0; bl < XParam.nblk; bl++)
	{
		for (int j = 0; j < 16; j++)
		{
			for (int i = 0; i < 16; i++)
			{
				int n = i + j * 16 + bl * XParam.blksize;

				if (hh[n] > XParam.eps)
				{

					zs[n] = max(zs[n] + T(XParam.zsoffset), zb[n]);

					hh[n] = zs[n] - zb[n];//0.0?
				}
			}

		}
	}
	
	return success;
}
//////////////////////////////////////////////////////////////////////////////////
//						                                                        //
//Copyright (C) 2018 Bosserelle                                                 //
// This code contains an adaptation of the St Venant equation from Basilisk		//
// See																			//
// http://basilisk.fr/src/green-naghdi.h										//
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


#include "GreenNaghdi.h"

template <class T> T dx(int ix, int iy,int ib, T* s)
{
	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
	int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
	return ((s[iright] - s[ileft]) / (2. * Delta));
}


/*
#define dx(s)  ((s[1,0] - s[-1,0])/(2.*Delta))
#define dy(s)  ((s[0,1] - s[0,-1])/(2.*Delta))
#define d2x(s) ((s[1,0] + s[-1,0] - 2.*s[])/sq(Delta))
#define d2y(s) ((s[0,1] + s[0,-1] - 2.*s[])/sq(Delta))
#define d2xy(s) ((s[1,1] - s[1,-1] - s[-1,1] + s[-1,-1])/sq(2.*Delta))

#define R1(h,zb,w) (-h[]*(h[]/3.*dx(w) + w[]*(dx(h) + dx(zb)/2.)))
#define R2(h,zb,w) (h[]/2.*dx(w) + w[]*(dx(zb) + dx(h)))

*/

template <class T> void residual_GN_X_CPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb, GNP<T> XGn)
{
	//
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	int lev;

	T alpha_d = T(1.153);

	T resmax = T(0.0);

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		lev = XBlock.level[ib];

		T delta = calcres(T(XParam.dx), lev);
		T itdelta = T(1.0) / (T(2.0) * delta);

		for (int iy = 0; iy < (XParam.blkwidth + XParam.halowidth); iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int ileft= memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
				int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);

				int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
				int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

				int itr = memloc(halowidth, blkmemwidth, ix + 1, iy + 1, ib);
				int itl = memloc(halowidth, blkmemwidth, ix - 1, iy + 1, ib);
				int ibr = memloc(halowidth, blkmemwidth, ix + 1, iy - 1, ib);
				int ibl = memloc(halowidth, blkmemwidth, ix - 1, iy - 1, ib);


				T hi= XEv.h[i];
				T hleft = XEv.h[ileft];
				T hright = XEv.h[iright];

				T dxh, dxzb, dyzb, dxzs, hl3, hr3;
				T dxeta;
				T d2xzb, d2xyzb;
				
				T dxDy, dyDy,d2xyDy;
				if (hleft >= XParam.eps && hi >= XParam.eps && hright >= XParam.eps)
				{
					//
					dxh = (hright - hleft) * itdelta;
					dxzb = (zb[iright] - zb[ileft]) * itdelta;
					dyzb = ((zb[itop] - zb[ibot]) * itdelta);
					dxDy = (XGn.Dy[iright] - XGn.Dy[ileft]) * itdelta;

					dyDy = ((XGn.Dy[itop] - XGn.Dy[ibot]) * itdelta);
					dxeta = dxzb + dxh;

					d2xzb = ((zb[iright] + zb[ileft] - 2. * zb[i]) / sq(delta));

					d2xyzb = ((zb[itr] - zb[itl] - zb[ibr] + zb[ibl]) / sq(2. * delta));
					d2xyDy = ((XGn.Dy[itr] - XGn.Dy[itl] - XGn.Dy[ibr] + XGn.Dy[ibl]) / sq(2. * delta));

					hl3 = (hi + hleft) / T(2.0);
					hl3 = hl3 * hl3 * hl3;

					hr3= (hi + hright) / T(2.0);
					hr3 = hr3 * hr3 * hr3;



					XGn.resx[i] = XGn.bx[i] -
						(-alpha_d / 3. * (hr3 * XGn.Dx[iright] + hl3 * XGn.Dx[ileft] -
						(hr3 + hl3) * XGn.Dx[i]) / sq(delta) +
							hi * (alpha_d * (dxeta * dxzb + hi / T(2.) * d2xzb) + T(1.)) * XGn.Dx[i] +
							alpha_d * hi * ((hi / 2. * d2xyzb + dxeta * dyzb)) * XGn.Dy[i] +
								hi / T(2.) * dyzb * dxDy - sq(hi) / T(3.) * d2xyDy
								- hi * dyDy * (dxh + dxzb) / T(2.));
					


				}
				else
				{
					XGn.resx[i] = 0.0;
				}


			}
		}
	}
}


template <class T> void residual_GN_Y_CPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb, GNP<T> XGn)
{
	//
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	int lev;

	T alpha_d = T(1.153);

	T resmax = T(0.0);

	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		lev = XBlock.level[ib];

		T delta = calcres(T(XParam.dx), lev);
		T itdelta = T(1.0) / (T(2.0) * delta);

		for (int iy = 0; iy < (XParam.blkwidth + XParam.halowidth); iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{
				int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
				int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
				int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);

				int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
				int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

				int itr = memloc(halowidth, blkmemwidth, ix + 1, iy + 1, ib);
				int itl = memloc(halowidth, blkmemwidth, ix - 1, iy + 1, ib);
				int ibr = memloc(halowidth, blkmemwidth, ix + 1, iy - 1, ib);
				int ibl = memloc(halowidth, blkmemwidth, ix - 1, iy - 1, ib);


				T hi = XEv.h[i];
				T hbot = XEv.h[ibot];
				T htop = XEv.h[itop];

				T dyh, dyzb, dxzb, dyzs, hb3, ht3;
				T dyeta;
				T d2yzb, d2xyzb;

				T dyDx, dxDx, d2xyDx;
				if (hbot >= XParam.eps && hi >= XParam.eps && htop >= XParam.eps)
				{
					//
					dyh = (htop - hbot) * itdelta;
					dxzb = (zb[iright] - zb[ileft]) * itdelta;
					dyzb = ((zb[itop] - zb[ibot]) * itdelta);
					dxDx = (XGn.Dx[iright] - XGn.Dx[ileft]) * itdelta;

					dyDx = ((XGn.Dx[itop] - XGn.Dx[ibot]) * itdelta);
					dyeta = dyzb + dyh;

					d2yzb = ((zb[itop] + zb[ibot] - 2. * zb[i]) / sq(delta));

					d2xyzb = ((zb[itr] - zb[itl] - zb[ibr] + zb[ibl]) / sq(2. * delta));
					d2xyDx = ((XGn.Dx[itr] - XGn.Dx[itl] - XGn.Dx[ibr] + XGn.Dx[ibl]) / sq(2. * delta));

					hb3 = (hi + hbot) / T(2.0);
					hb3 = hb3 * hb3 * hb3;

					ht3 = (hi + htop) / T(2.0);
					ht3 = ht3 * ht3 * ht3;



					XGn.resy[i] = XGn.by[i] -
						(-alpha_d / 3. * (ht3 * XGn.Dy[iright] + hb3 * XGn.Dy[ileft] -
						(ht3 + hb3) * XGn.Dy[i]) / sq(delta) +
							hi * (alpha_d * (dyeta * dyzb + hi / T(2.) * d2yzb) + T(1.)) * XGn.Dy[i] +
							alpha_d * hi * ((hi / 2. * d2xyzb + dyeta * dxzb)) * XGn.Dx[i] +
							hi / T(2.) * dxzb * dyDx - sq(hi) / T(3.) * d2xyDx
							- hi * dxDx * (dyh + dyzb) / T(2.));



				}
				else
				{
					XGn.resy[i] = T(0.0);
				}


			}
		}
	}
}




template <class T> void relax_GN_X_CPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb, GNP<T> XGn)
{
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	int lev;

	T alpha_d = T(1.153);


	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		lev = XBlock.level[ib];

		T delta = calcres(T(XParam.dx), lev);
		T itdelta = T(1.0) / (T(2.0) * delta);


		for (int iy = 0; iy < (XParam.blkwidth + XParam.halowidth); iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
				int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);

				int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
				int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

				int itr = memloc(halowidth, blkmemwidth, ix + 1, iy + 1, ib);
				int itl = memloc(halowidth, blkmemwidth, ix - 1, iy + 1, ib);
				int ibr = memloc(halowidth, blkmemwidth, ix + 1, iy - 1, ib);
				int ibl = memloc(halowidth, blkmemwidth, ix - 1, iy - 1, ib);

				T hi, hleft, hright;
				hi = XEv.h[i];
				hleft = XEv.h[ileft];
				hright = XEv.h[iright];

				bool weti, wetl, wetr;
				weti = hi > XParam.eps;
				wetl = hleft > XParam.eps;
				wetr = hright > XParam.eps;




				if (hi > XParam.eps && wetl && weti && wetr) 
				{
					T hc = hi;
					T dxh = (hright - hleft) * itdelta; 
					T dxzb = (zb[iright] - zb[ileft]) * itdelta;
					T dyzb = ((zb[itop] - zb[ibot]) * itdelta);
					T dxeta = dxzb + dxh;

					T dxDy = (XGn.Dy[iright] - XGn.Dy[ileft]) * itdelta;
					T dyDy = (XGn.Dy[itop] - XGn.Dy[ibot]) * itdelta;

					T d2xyDy = ((XGn.Dy[itr] - XGn.Dy[itl] - XGn.Dy[ibr] + XGn.Dy[ibl]) / sq(2. * delta));


					
					T d2xzb = ((zb[iright] + zb[ileft] - 2. * zb[i]) / sq(delta));
					

					T hl3, hr3;
					hl3 = (hi + hleft) / T(2.0);
					hl3 = hl3 * hl3 * hl3;

					hr3 = (hi + hright) / T(2.0);
					hr3 = hr3 * hr3 * hr3;

					T d2xyzb = ((zb[itr] - zb[itl] - zb[ibr] + zb[ibl]) / sq(2. * delta));


					XGn.Dx[i] = (XGn.bx[i] -
						(-alpha_d / 3. * (hr3 * XGn.Dx[iright] + hl3 * XGn.Dx[ileft]) / sq(Delta) +
							alpha_d * hc * ((hc / 2. * d2xyzb + dxeta * dyzb) * XGn.Dy[i] +
								hc / 2. * dyzb * dxDy - sq(hc) / 3. * d2xyDy
								- hc * dyDy * (dxh + dxzb / 2.)))) /
								(alpha_d * (hr3 + hl3) / (3. * sq(Delta)) +
									hc * (alpha_d * (dxeta * dxzb + hc / 2. * d2xzb) + 1.));
				}
				else
				{
					XGn.Dx[i] = T(0.0);
				}
					
			}
		}
	}
}

template <class T> void relax_GN_Y_CPU(Param XParam, BlockP<T> XBlock, EvolvingP<T> XEv, T* zb, GNP<T> XGn)
{
	int ib;
	int halowidth = XParam.halowidth;
	int blkmemwidth = XParam.blkmemwidth;

	int lev;

	T alpha_d = T(1.153);


	for (int ibl = 0; ibl < XParam.nblk; ibl++)
	{
		ib = XBlock.active[ibl];

		lev = XBlock.level[ib];

		T delta = calcres(T(XParam.dx), lev);
		T itdelta = T(1.0) / (T(2.0) * delta);


		for (int iy = 0; iy < (XParam.blkwidth + XParam.halowidth); iy++)
		{
			for (int ix = 0; ix < XParam.blkwidth; ix++)
			{

				int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
				int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);

				int ibot = memloc(halowidth, blkmemwidth, ix, iy - 1, ib);
				int itop = memloc(halowidth, blkmemwidth, ix, iy + 1, ib);

				int itr = memloc(halowidth, blkmemwidth, ix + 1, iy + 1, ib);
				int itl = memloc(halowidth, blkmemwidth, ix - 1, iy + 1, ib);
				int ibr = memloc(halowidth, blkmemwidth, ix + 1, iy - 1, ib);
				int ibl = memloc(halowidth, blkmemwidth, ix - 1, iy - 1, ib);

				T hi, hbot, htop;
				hi = XEv.h[i];
				hbot = XEv.h[ibot];
				htop = XEv.h[itop];

				bool weti, wetb, wett;
				weti = hi > XParam.eps;
				wetb = hbot > XParam.eps;
				wett = htop > XParam.eps;




				if (hi > XParam.eps && wetb && weti && wett)
				{
					T hc = hi;
					T dyh = (htop - hbot) * itdelta;
					T dxzb = (zb[iright] - zb[ileft]) * itdelta;
					T dyzb = ((zb[itop] - zb[ibot]) * itdelta);
					T dyeta = dyzb + dyh;

					T dxDy = (XGn.Dy[iright] - XGn.Dy[ileft]) * itdelta;
					T dyDy = (XGn.Dy[itop] - XGn.Dy[ibot]) * itdelta;

					T d2xyDy = ((XGn.Dy[itr] - XGn.Dy[itl] - XGn.Dy[ibr] + XGn.Dy[ibl]) / sq(2. * delta));



					T d2yzb = ((zb[itop] + zb[ibot] - 2. * zb[i]) / sq(delta));
					

					T hb3, ht3;
					hb3 = (hi + hbot) / T(2.0);
					hb3 = hb3 * hb3 * hb3;

					ht3 = (hi + htop) / T(2.0);
					ht3 = ht3 * ht3 * ht3;

					T d2xyzb = ((zb[itr] - zb[itl] - zb[ibr] + zb[ibl]) / sq(2. * delta));


					XGn.Dy[i] = (XGn.by[i] -
						(-alpha_d / 3. * (ht3 * XGn.Dy[itop] + hb3 * XGn.Dy[ibot]) / sq(Delta) +
							alpha_d * hc * ((hc / 2. * d2xyzb + dyeta * dxzb) * XGn.Dx[i] +
								hc / 2. * dxzb * dyDx - sq(hc) / 3. * d2xyDx
								- hc * dxDx * (dyh + dyzb / 2.)))) /
								(alpha_d * (ht3 + hb3) / (3. * sq(Delta)) +
									hc * (alpha_d * (dyeta * dyzb + hc / 2. * d2yzb) + 1.));
				}
				else
				{
					XGn.Dy[i] = T(0.0);
				}

			}
		}
	}
}
/*
static double residual_GN(scalar* a, scalar* r, scalar* resl, void* data)
{

	scalar* list = (scalar*)data;
	scalar h = list[0], zb = list[1], wet = list[2];
	vector D = vector(a[0]), b = vector(r[0]), res = vector(resl[0]);

	double maxres = 0.;
	foreach(reduction(max:maxres))
		foreach_dimension() {
		if (wet[-1] == 1 && wet[] == 1 && wet[1] == 1) {
			double hc = h[];
			dxh = dx(h);
			dxzb = dx(zb);
			dxeta = dxzb + dxh;
			double hl3 = (hc + h[-1]) / 2.;
			
			hl3 = cube(hl3);
			double hr3 = (hc + h[1]) / 2.;
			
			hr3 = cube(hr3);
			res.x[] = b.x[] -
				(-alpha_d / 3. * (hr3 * D.x[1] + hl3 * D.x[-1] -
				(hr3 + hl3) * D.x[]) / sq(Delta) +
					hc * (alpha_d * (dxeta * dxzb + hc / 2. * d2x(zb)) + 1.) * D.x[] +
					alpha_d * hc * ((hc / 2. * d2xy(zb) + dxeta * dy(zb)) * D.y[] +
						hc / 2. * dy(zb) * dx(D.y) - sq(hc) / 3. * d2xy(D.y)
						- hc * dy(D.y) * (dxh + dxzb / 2.)));
			if (fabs(res.x[]) > maxres)
				maxres = fabs(res.x[]);
		}
		else
			res.x[] = 0.;
	}
	return maxres / G;
}
*/


/*
static void relax_GN (scalar * a, scalar * r, int l, void * data)
{
  scalar * list = (scalar *) data;
  scalar h = list[0], zb = list[1], wet = list[2];
  vector D = vector(a[0]), b = vector(r[0]);
  foreach_level_or_leaf (l)
	foreach_dimension() {
	  if (h[] > dry && wet[-1] == 1 && wet[] == 1 && wet[1] == 1) {
	double hc = h[], dxh = dx(h), dxzb = dx(zb), dxeta = dxzb + dxh;
	double hl3 = (hc + h[-1])/2.; hl3 = cube(hl3);
	double hr3 = (hc + h[1])/2.;  hr3 = cube(hr3);
	D.x[] = (b.x[] -
		 (-alpha_d/3.*(hr3*D.x[1] + hl3*D.x[-1])/sq(Delta) +
		  alpha_d*hc*((hc/2.*d2xy(zb) + dxeta*dy(zb))*D.y[] +
				  hc/2.*dy(zb)*dx(D.y) - sq(hc)/3.*d2xy(D.y)
				  - hc*dy(D.y)*(dxh + dxzb/2.))))/
	  (alpha_d*(hr3 + hl3)/(3.*sq(Delta)) +
	   hc*(alpha_d*(dxeta*dxzb + hc/2.*d2x(zb)) + 1.));
	  }
	  else
	D.x[] = 0.;
	}
}
*/

/*
static double update_green_naghdi(scalar* current, scalar* updates,
	double dtmax)
{
	double dt = update_saint_venant(current, updates, dtmax);
	scalar h = current[0];
	vector u = vector(current[1]);
	vector b[];
	{
		scalar c[], d[];
		foreach() {
			double dxux = dx(u.x), dyuy = dy(u.y);
			c[] = -dxux * dyuy + dx(u.y) * dy(u.x) + sq(dxux + dyuy);
			d[] = sq(u.x[]) * d2x(zb) + sq(u.y[]) * d2y(zb) + 2. * u.x[] * u.y[] * d2xy(zb);
		}
		foreach()
			foreach_dimension()
			b.x[] = h[] * (G / alpha_d * dx(eta) - 2. * R1(h, zb, c) + R2(h, zb, d));
	}
	scalar wet[];
	foreach()
		wet[] = h[] > dry;
	scalar* list = { h, zb, wet };
	restriction(list);
	mgD = mg_solve((scalar*) { D }, (scalar*) { b },
		residual_GN, relax_GN, list, mgD.nrelax);
	vector dhu = vector(updates[1]);
	foreach()
		if (fabs(dx(eta)) < breaking && fabs(dy(eta)) < breaking)
			foreach_dimension()
			if (wet[-1] == 1 && wet[] == 1 && wet[1] == 1)
				dhu.x[] += h[] * (G / alpha_d * dx(eta) - D.x[]);

	return dt;
}
*/

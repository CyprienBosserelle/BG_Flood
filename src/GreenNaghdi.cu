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


#include "GreenNaghdi.h"

template <class T> T dx(int ix, int iy,int ib, T* s)
{
	int i = memloc(halowidth, blkmemwidth, ix, iy, ib);
	int ileft = memloc(halowidth, blkmemwidth, ix - 1, iy, ib);
	int iright = memloc(halowidth, blkmemwidth, ix + 1, iy, ib);
	return ((s[iright] - s[ileft]) / (2. * Delta));
}


/*
#define dy(s)  ((s[0,1] - s[0,-1])/(2.*Delta))
#define d2x(s) ((s[1,0] + s[-1,0] - 2.*s[])/sq(Delta))
#define d2y(s) ((s[0,1] + s[0,-1] - 2.*s[])/sq(Delta))
#define d2xy(s) ((s[1,1] - s[1,-1] - s[-1,1] + s[-1,-1])/sq(2.*Delta))

#define R1(h,zb,w) (-h[]*(h[]/3.*dx(w) + w[]*(dx(h) + dx(zb)/2.)))
#define R2(h,zb,w) (h[]/2.*dx(w) + w[]*(dx(zb) + dx(h)))

*/

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

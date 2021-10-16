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


#include "Poly.h"

// Copyright 2000 softSurfer, 2012 Dan Sunday
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.
// Code modified to fit the use in DisperGPU


// a Point is defined by its coordinates {int x, y;}
//===================================================================


// isLeft(): tests if a point is Left|On|Right of an infinite line.
//    Input:  three points P0, P1, and P2
//    Return: >0 for P2 left of the line through P0 and P1
//            =0 for P2  on the line
//            <0 for P2  right of the line
//    See: Algorithm 1 "Area of Triangles and Polygons"
template <class T> T isLeft(T P0x, T P0y, T P1x, T P1y, T P2x, T P2y)
{
	return ((P1x - P0x) * (P2y - P0y)
		- (P2x - P0x) * (P1y - P0y));
}
//===================================================================


// cn_PnPoly(): crossing number test for a point in a polygon
//      Input:   P = a point,
//               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
//      Return:  0 = outside, 1 = inside
// This code is patterned after [Franklin, 2000]
template <class T, class F> int cn_PnPoly(T Px, T Py, F* Vx, F* Vy, int n)
{
	int    cn = 0;    // the  crossing number counter

	// loop through all edges of the polygon
	for (int i = 0; i < n; i++) {    // edge from V[i]  to V[i+1]
		if (((Vy[i] <= Py) && (Vy[i + 1] > Py))     // an upward crossing
			|| ((Vy[i] > Py) && (Vy[i + 1] <= Py))) { // a downward crossing
			// compute  the actual edge-ray intersect x-coordinate
			T vt = (T)(Py - Vy[i]) / (Vy[i + 1] - Vy[i]);
			if (Px < Vx[i] + vt * (Vx[i + 1] - Vx[i])) // P.x < intersect
				++cn;   // a valid crossing of y=P.y right of P.x
		}
	}
	return (cn & 1);    // 0 if even (out), and 1 if  odd (in)

}
//===================================================================


// wn_PnPoly(): winding number test for a point in a polygon
//      Input:   P = a point,
//               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
//      Return:  wn = the winding number (=0 only when P is outside)
template <class T> int wn_PnPoly(T Px, T Py, T* Vx, T* Vy, unsigned int n)
{
	int    wn = 0;    // the  winding number counter

	// loop through all edges of the polygon
	for (int i = 0; i < n; i++) {   // edge from V[i] to  V[i+1]
		if (Vy[i] <= Py) {          // start y <= P.y
			if (Vy[i + 1] > Py)      // an upward crossing
				if (isLeft(Vx[i], Vy[i], Vx[i + 1], Vy[i + 1], Px, Py) > 0)  // P left of  edge
					++wn;            // have  a valid up intersect
		}
		else {                        // start y > P.y (no test needed)
			if (Vy[i + 1] <= Py)     // a downward crossing
				if (isLeft(Vx[i], Vy[i], Vx[i + 1], Vy[i + 1], Px, Py) < 0)  // P right of  edge
					--wn;            // have  a valid down intersect
		}
	}
	return wn;
}

template <class T> int wn_PnPoly(T Px, T Py, Polygon Poly)
{
	int    wn = 0;    // the  winding number counter

	// loop through all edges of the polygon
	for (int i = 0; i < (Poly.vertex.size() - 1); i++) {   // edge from V[i] to  V[i+1]
		if (Poly.vertex.y[i] <= Py) {          // start y <= P.y
			if (Poly.vertex.y[i + 1] > Py)      // an upward crossing
				if (isLeft(Poly.vertex.x[i], Poly.vertex.y[i], Poly.vertex.x[i + 1], Poly.vertex.y[i + 1], Px, Py) > 0)  // P left of  edge
					++wn;            // have  a valid up intersect
		}
		else {                        // start y > P.y (no test needed)
			if (Poly.vertex.y[i + 1] <= Py)     // a downward crossing
				if (isLeft(Poly.vertex.x[i], Poly.vertex.y[i], Poly.vertex.x[i + 1], Poly.vertex.y[i + 1], Px, Py) < 0)  // P right of  edge
					--wn;            // have  a valid down intersect
		}
	}
	return wn;
}
//===================================================================

template <class T> bool blockinpoly(T xo, T yo, T dx, int blkwidth, Polygon Poly)
{
	bool insidepoly = false;
	// First check if it isinmside the bounding box
	insidepoly = OBBdetect(xo, xo + dx * blkwidth, yo, yo + dx * blkwidth, Poly.xmin, Poly.xmax, Poly.ymin, Poly.ymax);

	if (insidepoly)
	{
		int wn;
		insidepoly = false;
		for (int i = 0; i < blkwidth; i++)
		{
			for (int j = 0; j < blkwidth; j++)
			{
				T Px = xo + i * dx;
				T Py = yo + j * dx;
				wn = wn_PnPoly(Px, Py, Poly);
				insidepoly = wn > 0;
			}
		}

	}

	return insidepoly;
}

//template <class T> Poly<T> ReadPoly();



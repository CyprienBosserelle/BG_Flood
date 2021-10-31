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

// wn_PnPoly(): winding number test for a point in a polygon
//      Input:   P = a point,
//               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
//      Return:  wn = the winding number (=0 only when P is outside)
template <class T> int wn_PnPoly(T Px, T Py, Polygon Poly)
{
	int    wn = 0;    // the  winding number counter

	// loop through all edges of the polygon
	for (int i = 0; i < (Poly.vertices.size() - 1); i++) {   // edge from V[i] to  V[i+1]
		if (Poly.vertices[i].y <= Py) {          // start y <= P.y
			if (Poly.vertices[i + 1].y > Py)      // an upward crossing
				if (isLeft(T(Poly.vertices[i].x), T(Poly.vertices[i].y), T(Poly.vertices[i + 1].x), T(Poly.vertices[i + 1].y), Px, Py) > 0)  // P left of  edge
					++wn;            // have  a valid up intersect
		}
		else {                        // start y > P.y (no test needed)
			if (Poly.vertices[i + 1].y <= Py)     // a downward crossing
				if (isLeft(T(Poly.vertices[i].x), T(Poly.vertices[i].y), T(Poly.vertices[i + 1].x), T(Poly.vertices[i + 1].y), Px, Py) < 0)  // P right of  edge
					--wn;            // have  a valid down intersect
		}
	}
	return wn;
}
template int wn_PnPoly<float>(float Px, float Py, Polygon Poly);
template int wn_PnPoly<double>(double Px, double Py, Polygon Poly);
//===================================================================

Vertex VertAdd(Vertex A, Vertex B)
{
	Vertex v;
	v.x = A.x + B.x; 
	v.y = A.y + B.y;

	return v;
}

Vertex VertSub(Vertex A, Vertex B)
{
	Vertex v;
	v.x = A.x - B.x;
	v.y = A.y - B.y;

	return v;
}

double dotprod(Vertex A, Vertex B)
{
	double a = 0.0;
	a = A.x * B.x + A.x + B.y + A.y * B.x + A.y * B.y;
	return a;
}

double xprod(Vertex A, Vertex B)
{
	double a = 0.0;
	a = A.x*B.y-A.y*B.x;
	return a;
}

//===================================================================
// Intersection between segments
bool SegmentIntersect(Polygon P, Polygon Q)
{
	//
	Vertex r, s, p, q, qmp;
	double rxs, qmpxr, eps, t, u;
	bool intersect = false;

	eps = 1e-9;

	p = P.vertices[0];
	q = Q.vertices[0];
	r = VertSub(P.vertices[1], P.vertices[0]);
	s = VertSub(Q.vertices[1], Q.vertices[0]);
	
	qmp= VertSub(q, p);

	rxs = xprod(r, s);

	qmpxr = xprod(qmp, r);

	


	if (abs(rxs) <= eps && abs(qmpxr) <= eps)
	{
		// colinear
		double t0, t1, rdr, sdr;
		sdr= dotprod(s, r);
		rdr = dotprod(r, r);

		t0 = dotprod(qmp, r) / rdr;
		t1 = t0 + dotprod(s, r) / rdr;

		if (sdr < 0.0)
		{
			intersect = (t0 >= 0.0 && t1 <= 1);
		}
		else
		{
			intersect = (t1 >= 0.0 && t0 <= 1);
		}


	}
	else if (abs(rxs) <= eps && abs(qmpxr) > eps)
	{
		// parallele lines and non intersecting
		intersect = false;
	}
	else if (abs(rxs) > eps)
	{
		t = xprod(qmp, s) / rxs;
		u = qmpxr / rxs;

		if (t >= 0.0 && t <= 1.0 && u <= 1.0 && u >= 0.0)
		{
			intersect = true;
		}

	}
	else
	{
		intersect = false;
	}

		
	return intersect;
}

bool PolygonIntersect(Polygon P, Polygon Q)
{
	bool intersect=false;
	for (int i = 0; i < (P.vertices.size() - 1); i++)
	{
		for (int j = 0; j < (Q.vertices.size() - 1); j++)
		{
			// build segments
			Polygon Pseg, Qseg;
			Pseg.vertices = { P.vertices[i], P.vertices[i + 1] };
			Qseg.vertices = { Q.vertices[i], Q.vertices[i + 1] };

			intersect = SegmentIntersect(Pseg, Qseg);

			if (intersect)
			{
				i = P.vertices.size();
				j = Q.vertices.size();
				break;
			}

		}
		
	}

	return intersect;

}


template <class T> bool blockinpoly(T xo, T yo, T dx, int blkwidth, Polygon Poly)
{
	bool insidepoly = false;
	// First check if it isinmside the bounding box
	insidepoly = OBBdetect(xo, xo + dx * blkwidth, yo, yo + dx * blkwidth, T(Poly.xmin), T(Poly.xmax), T(Poly.ymin), T(Poly.ymax));

	if (insidepoly)
	{
		// being in the bounding box doesn't say much

		// Is there any corner of the block inside the polygon?
		int wnBL,wnBR,wnTL,wnTR;
		insidepoly = false;
		
		wnBL = wn_PnPoly(xo, yo, Poly);
		wnBR = wn_PnPoly(xo + blkwidth*dx, yo, Poly);
		wnTL = wn_PnPoly(xo, yo + blkwidth * dx, Poly);
		wnTR = wn_PnPoly(xo + blkwidth * dx, yo + blkwidth * dx, Poly);

		insidepoly = (wnBL > 0 || wnBR > 0 || wnTL > 0 || wnTR > 0);

		if (!insidepoly)
		{
			// maybe a thin arn of the polygon intersect the block
			Polygon Polyblock;
			Vertex vxBL, vxBR, vxTL, vxTR;
			vxBL.x = xo; vxBL.y = yo;
			vxBR.x = xo + blkwidth * dx; vxBR.y = yo;
			vxTL.x = xo; vxTL.y = yo + blkwidth * dx;
			vxTR.x = xo + blkwidth * dx; vxTR.y = yo + blkwidth * dx;

			Polyblock.vertices.push_back(vxBL);
			Polyblock.vertices.push_back(vxBR);
			Polyblock.vertices.push_back(vxTR);
			Polyblock.vertices.push_back(vxTL);
			Polyblock.vertices.push_back(vxBL);

			insidepoly = PolygonIntersect(Polyblock, Poly);
		}

	}

	return insidepoly;
}
template bool blockinpoly<float>(float xo, float yo, float dx, int blkwidth, Polygon Poly);
template bool blockinpoly<double>(double xo, double yo, double dx, int blkwidth, Polygon Poly);
//template <class T> Poly<T> ReadPoly();



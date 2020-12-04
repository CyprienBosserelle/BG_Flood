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


#include "Util_CPU.h"


namespace utils {
	/*! \fn template <class T> T sq(T a)
	* Generic squaring function
	*/
	template <class T> __host__ __device__ T sq(T a) {
		return (a*a);
	}

	/*! \fn template <class T> const T& max(const T& a, const T& b)
	* Generic max function
	*/
	template <class T> __host__ __device__ const T& max(const T& a, const T& b) {
		return (a<b) ? b : a;     // or: return comp(a,b)?b:a; for version (2)
	}

	/*! \fn template <class T> const T& min(const T& a, const T& b)
	* Generic min function
	*/
	template <class T> __host__ __device__ const T& min(const T& a, const T& b) {
		return !(b<a) ? a : b;     // or: return comp(a,b)?b:a; for version (2)
	}


	template <class T> __host__ __device__ const T& nearest(const T& a, const T& b, const T& c) {
		return abs(b - c) > abs(a - c) ? a : b;     // Nearest element to c
	}

	template <class T> __host__ __device__ const T& nearest(const T& a, const T& b) {
		return abs(b) > abs(a) ? a : b;     // Nearest element to 0.0
	}

	template <class T> __host__ __device__ const T& floor(const T& a) {
		return abs(b) > abs(a) ? a : b;    
	}


	template __host__ __device__ const int& min<int>(const int& a, const int& b);
	template __host__ __device__ const float& min<float>(const float& a, const float& b);
	template __host__ __device__ const double& min<double>(const double& a, const double& b);

	template __host__ __device__ const int& max<int>(const int& a, const int& b);
	template __host__ __device__ const float& max<float>(const float& a, const float& b);
	template __host__ __device__ const double& max<double>(const double& a, const double& b);

	template int __host__ __device__ sq<int>(int a);
	template float __host__ __device__ sq<float>(float a);
	template double __host__ __device__ sq<double>(double a);

	template __host__ __device__ const int& nearest<int>(const int& a, const int& b, const int& c);
	template __host__ __device__ const float& nearest<float>(const float& a, const float& b, const float& c);
	template __host__ __device__ const double& nearest<double>(const double& a, const double& b, const double& c);

	template __host__ __device__ const int& nearest<int>(const int& a, const int& b);
	template __host__ __device__ const float& nearest<float>(const float& a, const float& b);
	template __host__ __device__ const double& nearest<double>(const double& a, const double& b);

}

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


double interptime(double next, double prev, double timenext, double time)
{
	return prev + (time) / (timenext)*(next - prev);
}



template <class T> T BilinearInterpolation(T q11, T q12, T q21, T q22, T x1, T x2, T y1, T y2, T x, T y)
{
	T x2x1, y2y1, x2x, y2y, yy1, xx1;
	x2x1 = x2 - x1;
	y2y1 = y2 - y1;
	x2x = x2 - x;
	y2y = y2 - y;
	yy1 = y - y1;
	xx1 = x - x1;
	return (T)1.0 / (x2x1 * y2y1) * (
		q11 * x2x * y2y +
		q21 * xx1 * y2y +
		q12 * x2x * yy1 +
		q22 * xx1 * yy1
		);
}

template float BilinearInterpolation<float>(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y);
template double BilinearInterpolation<double>(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y);


template <class T> T BarycentricInterpolation(T q1, T x1, T y1, T q2, T x2, T y2, T q3, T x3, T y3, T x, T y)
{
	T w1, w2, w3, D;

	D = (y2 - y3) * (x1 + x3) + (x3 - x2) * (y1 - y3);

	w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / D;
	w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / D;
	w3 = 1 - w1 - w2;

	return q1 * w1 + q2 * w2 + q3 * w3;
}

template float BarycentricInterpolation(float q1, float x1, float y1, float q2, float x2, float y2, float q3, float x3, float y3, float x, float y);
template double BarycentricInterpolation(double q1, double x1, double y1, double q2, double x2, double y2, double q3, double x3, double y3, double x, double y);


template <class T>
__host__ __device__ T calcres(T dx, int level)
{
	return level < 0 ? dx * (1 << abs(level)) : dx / (1 << level);
}

template __host__ __device__ double calcres<double>(double dx, int level);
template __host__ __device__ float calcres<float>(float dx, int level);

template <class T> __host__ __device__ T minmod2(T theta, T s0, T s1, T s2)
{
	//theta should be used as a global var 
	// can be used to tune the limiting (theta=1
	//gives minmod, the most dissipative limiter and theta = 2 gives
	//	superbee, the least dissipative).
	//float theta = 1.3f;
	if (s0 < s1 && s1 < s2) {
		T d1 = theta * (s1 - s0);
		T d2 = (s2 - s0) / T(2.0);
		T d3 = theta * (s2 - s1);
		if (d2 < d1) d1 = d2;
		return min(d1, d3);
	}
	if (s0 > s1 && s1 > s2) {
		T d1 = theta * (s1 - s0), d2 = (s2 - s0) / T(2.0), d3 = theta * (s2 - s1);
		if (d2 > d1) d1 = d2;
		return max(d1, d3);
	}
	return T(0.0);
}

template __host__ __device__ float minmod2(float theta, float s0, float s1, float s2);
template __host__ __device__ double minmod2(double theta, double s0, double s1, double s2);

#include "Spherical.h"




template <class T> __device__ __host__ T calcCM(Param XParam, T delta, T yo, int iy)
{
	T y = XBlock.yo[ib] + iy * delta / XParam.Radius * T(180.0 / pi);
	// THis should be the y of the face so fo the v face you need to remove 0.5*delta

	T phi = y * T(pi / 180.0);

	T dphi = delta / (T(2.0 * Radius));// dy*0.5f*pi/180.0f;

	T cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0 * dphi);

	return cm;
}
template __device__ __host__ double calcCM(Param XParam, double delta, double yo, double iy);
template __device__ __host__ float calcCM(Param XParam, float delta, float yo, float iy);


template <class T> __device__ __host__ T calcFM(Param XParam, T delta, T yo, int iy)
{
	T y = XBlock.yo[ib] + iy * delta / XParam.Radius * T(180.0 / pi);
	// THis should be the y of the face so fo the v face you need to remove 0.5*delta

	T phi = y * T(pi / 180.0);

	//T dphi = delta / (T(2.0 * Radius));// dy*0.5f*pi/180.0f;

	T fmu = cos(phi);

	return fmu;
}
template __device__ __host__ double calcFM(Param XParam, double delta, double yo, double iy);
template __device__ __host__ float calcFM(Param XParam, float delta, float yo, float iy);


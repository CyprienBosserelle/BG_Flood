#include "Spherical.h"




template <class T> 
__host__ __device__ T calcCM(T Radius, T delta, T yo, int iy)
{
	T y = yo + iy * delta / Radius * T(180.0 / pi);
	// THis should be the y of the face so fo the v face you need to remove 0.5*delta

	T phi = y * T(pi / 180.0);

	T dphi = delta / (T(2.0 * Radius));// dy*0.5f*pi/180.0f;

	T cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0 * dphi);

	return cm;
}
template __host__ __device__ double calcCM(double Radius, double delta, double yo, int iy);
template __host__ __device__ float calcCM(float Radius, float delta, float yo, int iy);



template <class T> 
__host__ __device__  T calcFM(T Radius, T delta, T yo, T iy)
{
	T dy = delta / Radius * T(180.0 / pi);
	T y = yo + iy * dy;
	// THis should be the y of the face so fo the v face you need to remove 0.5*delta

	T phi = y * T(pi / 180.0);

	//T dphi = delta / (T(2.0 * Radius));// dy*0.5f*pi/180.0f;

	T fmu = cos(phi);

	return fmu;
}
template __host__ __device__ double calcFM(double Radius, double delta, double yo, double iy);
template __host__ __device__ float calcFM(float Radius, float delta, float yo, float iy);




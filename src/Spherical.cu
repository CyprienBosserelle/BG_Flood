#include "Spherical.h"



/**
 * @brief Calculate the scale factor for the y face length in a spherical model.
 * This function computes the scale factor based on the sphere's radius, grid spacing, origin offset, and index in the y direction.
 * Scale factor for y face length (x face lengh scale is always 1 in spherical model assuming that lat long are entered)
 * @param Radius Radius of the sphere
 * @param delta Grid spacing
 * @param yo Origin offset in the y direction
 * @param iy Index in the y direction
 */
template <class T> 
__host__ __device__ T calcCM(T Radius, T delta, T yo, int iy)
{
	T y = yo + (iy+0.5) * delta / Radius * T(180.0 / pi);
	// THis should be the y of the face so fo the v face you need to remove 0.5*delta

	T phi = y * T(pi / 180.0);

	T dphi = delta / (T(2.0 * Radius));// dy*0.5f*pi/180.0f;

	T cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0 * dphi);

	return cm;
}
template __host__ __device__ double calcCM(double Radius, double delta, double yo, int iy);
template __host__ __device__ float calcCM(float Radius, float delta, float yo, int iy);


/**
 * @brief Calculate the scale factor for the y face length in a spherical model.
 * This function computes the scale factor based on the sphere's radius, grid spacing, origin offset
 * and index in the y direction.
 * Scale factor for y face length (x face lengh scale is always 1 in spherical model assuming that lat long are entered)
 * @param Radius Radius of the sphere
 * @param delta Grid spacing
 * @param yo Origin offset in the y direction
 * @param iy Index in the y direction
 */
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


/**
 * @brief Classic Haversine formula to calculate great-circle distance between two points on a sphere.
 * The function is too slow to use directly in BG_flood engine but is more usable (i.e. naive) for model setup
 * @param Radius Radius of the sphere
 * @param lon1 Longitude of the first point (in degrees)
 * @param lat1 Latitude of the first point (in degrees)
 * @param lon2 Longitude of the second point (in degrees)
 * @param lat2 Latitude of the second point (in degrees)
 * @return Great-circle distance between the two points
 */
template <class T>
__host__ __device__  T haversin(T Radius, T lon1, T lat1, T lon2, T lat2)
{
	T phi1, phi2, dphi, dlbda, a, c;
	dphi = (lat2 - lat1) * T(pi / 180.0);
	dlbda = (lon2 -lon1) * T(pi / 180.0);

	phi1 = lat1 * T(pi / 180.0);
	phi2 = lat2 * T(pi / 180.0);

	T sindphid2 = sin(dphi / T(2.0));
	T sindlbdad2 = sin(dlbda / T(2.0));
	
	a = sindphid2 * sindphid2 + cos(phi1) * cos(phi2) * sindlbdad2 * sindlbdad2;

	c = T(2.0) * atan2(sqrt(a), sqrt(T(1.0) - a));

	return Radius * c;

}

/**
 * @brief Calculate the surface area of a spherical cap.
 * @tparam T Data type (float or double)
 * @param Radius Radius of the sphere
 * @param lon Longitude of the center of the cap (in degrees)
 * @param lat Latitude of the center of the cap (in degrees)
 * @param dx Grid spacing (in degrees)
 * @return Surface area of the spherical cap
 */
template <class T>
__host__ __device__  T spharea(T Radius, T lon, T lat, T dx)
{
	T lon1, lon2, lat1, lat2;
	lon1 = lon - T(0.5) * dx;
	lon2 = lon + T(0.5) * dx;

	lat1 = lat - T(0.5) * dx;
	lat2 = lat + T(0.5) * dx;

	T a, b, c;

	a = haversin(Radius, lon1, lat1, lon2, lat1);
	c = haversin(Radius, lon1, lat2, lon2, lat2);
	b = haversin(Radius, lon1, lat1, lon1, lat2);

	T Area = T(0.5) * (a * b + c * b);

	return Area;

}
template __host__ __device__  double spharea(double Radius, double lon, double lat, double dx);
template __host__ __device__  float spharea(float Radius, float lon, float lat, float dx);

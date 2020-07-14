
#include "Utils_GPU.h"

template <class T>
__host__ __device__ double calcres(T dx, int level)
{
	return level < 0 ? dx * (1 << abs(level)) : dx / (1 << level);
}

template __host__ __device__ double calcres<double>(double dx, int level);
template __host__ __device__ double calcres<float>(float dx, int level);

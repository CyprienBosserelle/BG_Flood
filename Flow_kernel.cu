



__device__ float minmod2fGPU(float s0, float s1, float s2)
{
	//theta should be used as a global var 
	// can be used to tune the limiting (theta=1
	//gives minmod, the most dissipative limiter and theta = 2 gives
	//	superbee, the least dissipative).
	float theta = 1.3f;
	if (s0 < s1 && s1 < s2) {
		float d1 = theta*(s1 - s0);
		float d2 = (s2 - s0) / 2.0f;
		float d3 = theta*(s2 - s1);
		if (d2 < d1) d1 = d2;
		return min(d1, d3);
	}
	if (s0 > s1 && s1 > s2) {
		float d1 = theta*(s1 - s0), d2 = (s2 - s0) / 2.0f, d3 = theta*(s2 - s1);
		if (d2 > d1) d1 = d2;
		return max(d1, d3);
	}
	return 0.;
}

__global__ void gradientGPUX(unsigned int nx, unsigned int ny, float delta, float *a, float *&dadx)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	int xplus, yplus, xminus, yminus;

	
	//
	//
	xplus = min(ix + 1, nx - 1);
	xminus = max(ix - 1, (unsigned int) 0);
	yplus = min(iy + 1, ny - 1);
	yminus = max(iy - 1, (unsigned int) 0);
	i = ix + iy*nx;


	//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
	dadx[i] = minmod2fGPU(a[xminus + iy*nx], a[i], a[xplus + iy*nx]) / delta;
			




}

__global__ void gradientGPUY(unsigned int nx, unsigned int ny, float delta, float *a, float *&dady)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	int  xplus, yplus, xminus, yminus;


	//
	//
	xplus = min(ix + 1, nx - 1);
	xminus = max(ix - 1, (unsigned int)0);
	yplus = min(iy + 1, ny - 1);
	yminus = max(iy - 1, (unsigned int)0);
	i = ix + iy*nx;


	//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
	dady[i] = minmod2fGPU(a[ix + yminus*nx], a[i], a[ix + yplus*nx]) / delta;





}

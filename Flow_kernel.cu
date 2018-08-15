
// textures have to be declared here...
texture<float, 2, cudaReadModeElementType> texLBND;
texture<float, 2, cudaReadModeElementType> texRBND;
texture<float, 2, cudaReadModeElementType> texTBND;
texture<float, 2, cudaReadModeElementType> texBBND;

template<class T>
struct SharedMemory
{
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
	__device__ inline operator double *()
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}

	__device__ inline operator const double *() const
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}
};


template<class T>
__device__ T sq(T a)
{
	return a*a;
}


__device__ float minmod2fGPU(float theta,float s0, float s1, float s2)
{
	//theta should be used as a global var 
	// can be used to tune the limiting (theta=1
	//gives minmod, the most dissipative limiter and theta = 2 gives
	//	superbee, the least dissipative).
	//float theta = 1.3f;
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

template <class T> __device__ T minmod2GPU(T theta, T s0, T s1, T s2)
{
	//theta should be used as a global var 
	// can be used to tune the limiting (theta=1
	//gives minmod, the most dissipative limiter and theta = 2 gives
	//	superbee, the least dissipative).
	//float theta = 1.3f;
	if (s0 < s1 && s1 < s2) {
		T d1 = theta*(s1 - s0);
		T d2 = (s2 - s0) / T(2.0);
		T d3 = theta*(s2 - s1);
		if (d2 < d1) d1 = d2;
		return min(d1, d3);
	}
	if (s0 > s1 && s1 > s2) {
		T d1 = theta*(s1 - s0), d2 = (s2 - s0) / T(2.0), d3 = theta*(s2 - s1);
		if (d2 > d1) d1 = d2;
		return max(d1, d3);
	}
	return T(0.0);
}

template <class T> __global__ void gradientGPUXY(int nx, int ny, T theta, T delta, T *a, T *dadx, T *dady)
{
	//
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	//int tx = threadIdx.x;
	//int ty = threadIdx.y;
	int i = ix + iy*nx;
	int  xplus, yplus, xminus, yminus;

	T a_i,a_r, a_l, a_t, a_b;

	//__shared__ float a_s[18][18];
	if (ix < nx && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);

		/*
		a_s[tx + 1][ty + 1] = a[ix + iy*nx];
		__syncthreads;
		// read the halo around the tile
		if (threadIdx.x == blockDim.x - 1)
			a_s[tx + 2][ty + 1] = a[xplus + iy*nx];

		if (threadIdx.x == 0)
			a_s[tx][ty + 1] = a[xminus + iy*nx];

		if (threadIdx.y == blockDim.y - 1)
			a_s[tx + 1][ty + 2] = a[ix + yplus*nx];

		if (threadIdx.y == 0)
			a_s[tx + 1][ty] = a[ix + yminus*nx];

		__syncthreads;
		*/
		a_i = a[ix + iy*nx];
		a_r= a[xplus + iy*nx];
		a_l= a[xminus + iy*nx];
		a_t = a[ix + yplus*nx];
		a_b = a[ix + yminus*nx];


		//dadx[i] = minmod2fGPU(theta, a_s[tx][ty + 1], a_s[tx + 1][ty + 1], a_s[tx + 2][ty + 1]) / delta;
		//dady[i] = minmod2fGPU(theta, a_s[tx + 1][ty], a_s[tx + 1][ty + 1], a_s[tx + 1][ty + 2]) / delta;

		dadx[i] = minmod2GPU(theta, a_l, a_i, a_r) / delta;
		dady[i] = minmod2GPU(theta, a_b, a_i, a_t) / delta;
	}

}

__global__ void gradientGPUX(int nx, int ny,float theta, float delta, float *a, float *dadx)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i = ix + iy*nx;
	int xplus, xminus;

	__shared__ float a_s[16][16]; // Hard wired stuff Be carefull
	__shared__ float al_s[16][16];
	__shared__ float ar_s[16][16];
	//float dadxi;
	if (ix < nx && iy < ny)
	{
		//
		//
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);

		i = ix + iy*nx;
		//a_s[tx][ty] = a[ix + iy*nx];
		//al_s[tx][ty] = a[xminus + iy*nx];
		//ar_s[tx][ty] = a[xplus + iy*nx];
		
		a_s[tx][ty] = a[ix + iy*nx];

		// read the halo around the tile
		__syncthreads;

		al_s[tx][ty] = a[xminus + iy*nx];
		__syncthreads;
		ar_s[tx][ty] = a[xplus + iy*nx];
		
		// Need to wait for threadX 0 and threadX 16-1 to finish
		__syncthreads;
		




		//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
		//dadx[i] = minmod2fGPU(theta,a[xminus + iy*nx], a[i], a[xplus + iy*nx]) / delta;
		// These below are somewhat slower when using shared mem. I'm unsure why (bank conflict?)
		//dadx[i] = minmod2fGPU(theta, al_s[tx][ty], a_s[tx][ty], ar_s[tx][ty]) / delta;
		//dadx[i] = minmod2fGPU(theta, a_s[tx][ty], a_s[tx+1][ty], a_s[tx+2][ty]) / delta;
		//__device__ float minmod2fGPU(float theta,float s0, float s1, float s2)


		//float d1, d2, d3;
		//float s0, s1, s2;
		
		//dadxi = 0.0f;
		dadx[i] = minmod2fGPU(theta, al_s[tx][ty], a_s[tx][ty], ar_s[tx][ty]) / delta;
		/*
		s0 = al_s[tx][ty];// there will be bank conflict here
		s1 = a_s[tx][ty];// there will be bank conflict here
		s2 = ar_s[tx][ty];// there will be bank conflict here

		if (s0 < s1 && s1 < s2) {
			d1 = theta*(s1 - s0);
			d2 = (s2 - s0) / 2.0f;
			d3 = theta*(s2 - s1);
			if (d2 < d1) d1 = d2;
			dadxi = min(d1, d3);
		}
		if (s0 > s1 && s1 > s2) {
			d1 = theta*(s1 - s0);
			d2 = (s2 - s0) / 2.0f;
			d3 = theta*(s2 - s1);
			if (d2 > d1) d1 = d2;
			dadxi = max(d1, d3);
		}
		
		dadx[i] = dadxi / delta;
		*/
		
	}



}

__global__ void gradientGPUXOLD(int nx, int ny, float theta, float delta, float *a, float *dadx)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int  xplus, xminus;

	if (ix < nx && iy < ny)
	{
		//
		//

		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		i = ix + iy*nx;


		//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
		dadx[i] = minmod2fGPU(theta, a[xminus + iy*nx], a[i], a[xplus + iy*nx]) / delta;

	}



}
__global__ void gradientGPUYSM(int nx, int ny, float theta, float delta, float *a, float *dady)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i = ix + iy*nx;
	int  yplus, yminus;

	__shared__ float a_s[16][18];

	float dadyi = 0.0f;
	if (ix < nx && iy < ny)
	{
		//
		//

		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);
		i = ix + iy*nx;

		a_s[tx][ty] = a[ix + yminus*nx];

		// read the halo around the tile

		if (threadIdx.x == 15)//blockDim.x - 1
		{
			a_s[tx][ty + 1] = a[i];
			a_s[tx][ty + 2] = a[ix + yplus*nx];

		}

		// Need to wait for threadX 0 and threadX 16-1 to finish
		__syncthreads;

		//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
		//dady[i] = minmod2fGPU(theta, a[ix + yminus*nx], a[i], a[ix + yplus*nx]) / delta;
		//dady[i] = minmod2fGPU(theta, a_s[tx][ty], a_s[tx][ty + 1], a_s[tx][ty + 2]) / delta;
		float d1, d2, d3;
		if (a_s[tx][ty] < a_s[tx][ty + 1] && a_s[tx][ty + 1] < a_s[tx][ty + 2]) {
			d1 = theta*(a_s[tx][ty + 1] - a_s[tx][ty]);
			d2 = (a_s[tx][ty + 2] - a_s[tx][ty]) / 2.0f;
			d3 = theta*(a_s[tx][ty + 2] - a_s[tx][ty + 1]);
			if (d2 < d1) d1 = d2;
			dadyi = min(d1, d3);
		}
		if (a_s[tx][ty] > a_s[tx][ty + 1] && a_s[tx][ty + 1] > a_s[tx][ty + 2]) {
			d1 = theta*(a_s[tx][ty + 1] - a_s[tx][ty]);
			d2 = (a_s[tx][ty + 2] - a_s[tx][ty]) / 2.0f;
			d3 = theta*(a_s[tx][ty + 2] - a_s[tx][ty + 1]);
			if (d2 > d1) d1 = d2;
			dadyi = max(d1, d3);
		}
		dady[i] = dadyi / delta;
	}



}
__global__ void gradientGPUY(int nx, int ny,float theta, float delta, float *a, float *dady)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int  yplus, yminus;

	if (ix < nx && iy < ny)
	{
		//
		//
		
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);
		i = ix + iy*nx;


		//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
		dady[i] = minmod2fGPU(theta,a[ix + yminus*nx], a[i], a[ix + yplus*nx]) / delta;

	}



}


__global__ void updateKurgX( int nx, int ny, float delta, float g, float eps,float CFL, float * hh, float *zs, float *uu, float * vv, float *dzsdx, float *dhdx, float * dudx, float *dvdx, float *Fhu, float *Fqux, float *Fqvx, float *Su, float * dtmax)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	//int tx = threadIdx.x;
	//int ty = threadIdx.y;
	//int  xplus, yplus, xminus, yminus;
	int xminus;
	if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);


		



		float dhdxi= dhdx[i];
		float dhdxmin = dhdx[xminus + iy*nx];
		float cm = 1.0f;// 0.1;
		float fmu = 1.0f;
		//float fmv = 1.0;

		//__shared__ float hi[16][16];
		float hi = hh[i];

		float hn = hh[xminus + iy*nx];


		if (hi > eps || hn > eps)
		{
			float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm,sl,sr;

			// along X
			dx = delta*0.5f;
			zi = zs[i] - hi;

			//printf("%f\n", zi);


			//zl = zi - dx*(dzsdx[i] - dhdx[i]);
			zl = zi - dx*(dzsdx[i] - dhdxi);
			//printf("%f\n", zl);

			zn = zs[xminus + iy*nx] - hn;

			//printf("%f\n", zn);
			zr = zn + dx*(dzsdx[xminus + iy*nx] - dhdxmin);


			zlr = max(zl, zr);

			//hl = hi - dx*dhdx[i];
			hl = hi - dx*dhdxi;
			up = uu[i] - dx*dudx[i];
			hp = max(0.0f, hl + zl - zlr);

			hr = hn + dx*dhdxmin;
			um = uu[xminus + iy*nx] + dx*dudx[xminus + iy*nx];
			hm = max(0.0f, hr + zr - zlr);

			float fh, fu, fv;
			//float dtmaxf = 1 / 1e-30f;

			//We can now call one of the approximate Riemann solvers to get the fluxes.
			float cp, cmo, ap, am, qm, qp, a, dlt, ad, hm2, hp2, ga,apm;
			float epsi = 1e-30f;

			cp = sqrtf(g*hp);
			cmo = sqrtf(g*hm);

			ap = max(max(up + cp, um + cmo),0.0f);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo),0.0f);
			//am = min(am, 0.0f);
			ad = 1.0f / (ap - am);
			qm = hm*um;
			qp = hp*up;

			a = max(ap, -am);

			dlt = delta*cm / fmu;
			hm2 = sq(hm);
			hp2 = sq(hp);
			ga = g*0.5f;
			apm = ap*am;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2 ) - am*(qp*up + ga*hp2 ) + apm*(qp - qm)) *ad;
				//fu = (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm)) / (ap - am);
				float dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;


			}
			else
			{
				fh = 0.0f;
				fu = 0.0f;
				dtmax[i] = 1.0f / 1e-30f;
			}
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);

			/*
			void kurganovf(float hm, float hp, float um, float up, float Delta, float * fh, float * fq, float * dtmax)
			float eps = epsilon;
			float cp = sqrtf(g*hp), cm = sqrtf(g*hm);
			float ap = max(up + cp, um + cm); ap = max(ap, 0.0f);
			float am = min(up - cp, um - cm); am = min(am, 0.0f);
			float qm = hm*um, qp = hp*up;
			float a = max(ap, -am);
			if (a > eps) {
			*fh = (ap*qm - am*qp + ap*am*(hp - hm)) / (ap - am); // (4.5) of [1]
			*fq = (ap*(qm*um + g*sq(hm) / 2.) - am*(qp*up + g*sq(hp) / 2.) +
			ap*am*(qp - qm)) / (ap - am);
			float dt = CFL*Delta / a;
			if (dt < *dtmax)
			*dtmax = dt;
			}
			else
			*fh = *fq = 0.;*/

			if (fh > 0.0f)
			{
				fv = (vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx])*fh;
			}
			else 
			{
				fv = (vv[i] - dx*dvdx[i])*fh;
			}
			//fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
			//dtmax needs to be stored in an array and reduced at the end
			//dtmax = dtmaxf;
			//dtmaxtmp = min(dtmax, dtmaxtmp);
			/*if (ix == 11 && iy == 0)
			{
				printf("a=%f\t b=%f\t c=%f\t d=%f\n", ap*(qm*um + ga*hm2), -am*(qp*up + ga*hp2),( ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm) ) *ad/100.0f, ad);
			}
			*/
			/*
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhu[i] = fmu * fh;
			Fqux[i] = fmu * (fu - sl);
			Su[i] = fmu * (fu - sr);
			Fqvx[i] = fmu * fv;
		}
		else
		{
			dtmax[i] = 1.0f / 1e-30f;
			Fhu[i] = 0.0f;
			Fqux[i] = 0.0f;
			Su[i] = 0.0f;
			Fqvx[i] = 0.0f;
		}

	}


}


__global__ void updateKurgXD(int nx, int ny, double delta, double g, double eps, double CFL, double * hh, double *zs, double *uu, double * vv, double *dzsdx, double *dhdx, double * dudx, double *dvdx, double *Fhu, double *Fqux, double *Fqvx, double *Su, double * dtmax)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	//int tx = threadIdx.x;
	//int ty = threadIdx.y;
	//int  xplus, yplus, xminus, yminus;
	int xminus;
	if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);






		double dhdxi = dhdx[i];
		double dhdxmin = dhdx[xminus + iy*nx];
		double cm = 1.0;// 0.1;
		double fmu = 1.0;
		//float fmv = 1.0;

		//__shared__ float hi[16][16];
		double hi = hh[i];

		double hn = hh[xminus + iy*nx];


		if (hi > eps || hn > eps)
		{
			double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, sl, sr;

			// along X
			dx = delta*0.5;
			zi = zs[i] - hi;

			//printf("%f\n", zi);


			//zl = zi - dx*(dzsdx[i] - dhdx[i]);
			zl = zi - dx*(dzsdx[i] - dhdxi);
			//printf("%f\n", zl);

			zn = zs[xminus + iy*nx] - hn;

			//printf("%f\n", zn);
			zr = zn + dx*(dzsdx[xminus + iy*nx] - dhdxmin);


			zlr = max(zl, zr);

			//hl = hi - dx*dhdx[i];
			hl = hi - dx*dhdxi;
			up = uu[i] - dx*dudx[i];
			hp = max(0.0, hl + zl - zlr);

			hr = hn + dx*dhdxmin;
			um = uu[xminus + iy*nx] + dx*dudx[xminus + iy*nx];
			hm = max(0.0, hr + zr - zlr);

			double fh, fu, fv;
			//float dtmaxf = 1 / 1e-30f;

			//We can now call one of the approximate Riemann solvers to get the fluxes.
			double cp, cmo, ap, am, qm, qp, a, dlt, ad, hm2, hp2, ga, apm;
			double epsi = 1e-30;

			cp = sqrt(g*hp);
			cmo = sqrt(g*hm);

			ap = max(max(up + cp, um + cmo), 0.0);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo), 0.0);
			//am = min(am, 0.0f);
			ad = 1.0 / (ap - am);
			qm = hm*um;
			qp = hp*up;

			a = max(ap, -am);

			dlt = delta*cm / fmu;
			hm2 = sq(hm);
			hp2 = sq(hp);
			ga = g*0.5;
			apm = ap*am;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2) - am*(qp*up + ga*hp2) + apm*(qp - qm)) *ad;
				//fu = (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm)) / (ap - am);
				double dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;


			}
			else
			{
				fh = 0.0;
				fu = 0.0;
				dtmax[i] = 1.0 / 1e-30;
			}
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);

			/*
			void kurganovf(float hm, float hp, float um, float up, float Delta, float * fh, float * fq, float * dtmax)
			float eps = epsilon;
			float cp = sqrtf(g*hp), cm = sqrtf(g*hm);
			float ap = max(up + cp, um + cm); ap = max(ap, 0.0f);
			float am = min(up - cp, um - cm); am = min(am, 0.0f);
			float qm = hm*um, qp = hp*up;
			float a = max(ap, -am);
			if (a > eps) {
			*fh = (ap*qm - am*qp + ap*am*(hp - hm)) / (ap - am); // (4.5) of [1]
			*fq = (ap*(qm*um + g*sq(hm) / 2.) - am*(qp*up + g*sq(hp) / 2.) +
			ap*am*(qp - qm)) / (ap - am);
			float dt = CFL*Delta / a;
			if (dt < *dtmax)
			*dtmax = dt;
			}
			else
			*fh = *fq = 0.;*/

			if (fh > 0.0)
			{
				fv = (vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx])*fh;
			}
			else
			{
				fv = (vv[i] - dx*dvdx[i])*fh;
			}
			//fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
			//dtmax needs to be stored in an array and reduced at the end
			//dtmax = dtmaxf;
			//dtmaxtmp = min(dtmax, dtmaxtmp);
			/*if (ix == 11 && iy == 0)
			{
			printf("a=%f\t b=%f\t c=%f\t d=%f\n", ap*(qm*um + ga*hm2), -am*(qp*up + ga*hp2),( ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm) ) *ad/100.0f, ad);
			}
			*/
			/*
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhu[i] = fmu * fh;
			Fqux[i] = fmu * (fu - sl);
			Su[i] = fmu * (fu - sr);
			Fqvx[i] = fmu * fv;
		}
		else
		{
			dtmax[i] = 1.0 / 1e-30;
			Fhu[i] = 0.0;
			Fqux[i] = 0.0;
			Su[i] = 0.0;
			Fqvx[i] = 0.0;
		}

	}


}

__global__ void updateKurgXSPH(int nx, int ny, double delta, double g, double eps, double CFL, double yo, double Radius, double * hh, double *zs, double *uu, double * vv, double *dzsdx, double *dhdx, double * dudx, double *dvdx, double *Fhu, double *Fqux, double *Fqvx, double *Su, double * dtmax)
{
	//Same as updateKurgX but with Spherical coordinates
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	//int tx = threadIdx.x;
	//int ty = threadIdx.y;
	//int  xplus, yplus, xminus, yminus;
	int xminus;
	double cm, fmu,y,phi,dphi;

	if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);






		double dhdxi = dhdx[i];
		double dhdxmin = dhdx[xminus + iy*nx];

		y = yo + iy*delta / Radius*180.0 / pi;

		phi = y*pi / 180.0;

		dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

		cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

		fmu = 1.0;
		//fmv = cosf(phi);

		//float cm = 1.0f;// 0.1;
		//float fmu = 1.0f;
		//float fmv = 1.0;

		//__shared__ float hi[16][16];
		double hi = hh[i];

		double hn = hh[xminus + iy*nx];


		if (hi > eps || hn > eps)
		{
			double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm, sl, sr;

			// along X
			dx = delta*0.5;
			zi = zs[i] - hi;

			//printf("%f\n", zi);


			//zl = zi - dx*(dzsdx[i] - dhdx[i]);
			zl = zi - dx*(dzsdx[i] - dhdxi);
			//printf("%f\n", zl);

			zn = zs[xminus + iy*nx] - hn;

			//printf("%f\n", zn);
			zr = zn + dx*(dzsdx[xminus + iy*nx] - dhdxmin);


			zlr = max(zl, zr);

			//hl = hi - dx*dhdx[i];
			hl = hi - dx*dhdxi;
			up = uu[i] - dx*dudx[i];
			hp = max(0.0, hl + zl - zlr);

			hr = hn + dx*dhdxmin;
			um = uu[xminus + iy*nx] + dx*dudx[xminus + iy*nx];
			hm = max(0.0, hr + zr - zlr);

			double fh, fu, fv;
			//float dtmaxf = 1 / 1e-30f;

			//We can now call one of the approximate Riemann solvers to get the fluxes.
			double cp, cmo, qm, qp, a, dlt, hm2, hp2, ga, apm;
			double ap, am,ad;
			double epsi = 1e-30;

			cp = sqrt(g*hp);
			cmo = sqrt(g*hm);

			ap = max(max(up + cp, um + cmo), 0.0);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo), 0.0);
			//am = min(am, 0.0f);
			ad = 1.0 / (ap - am);
			qm = hm*um;
			qp = hp*up;

			a = max(ap, -am);

			dlt = delta*cm / fmu;
			hm2 = sq(hm);
			hp2 = sq(hp);
			ga = g*0.5;
			apm = ap*am;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2) - am*(qp*up + ga*hp2) + apm*(qp - qm)) *ad;
				//fu = (ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm)) / (ap - am);
				double dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;


			}
			else
			{
				fh = 0.0;
				fu = 0.0;
				dtmax[i] = 1.0 / 1e-30;
			}
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);

			/*
			void kurganovf(float hm, float hp, float um, float up, float Delta, float * fh, float * fq, float * dtmax)
			float eps = epsilon;
			float cp = sqrtf(g*hp), cm = sqrtf(g*hm);
			float ap = max(up + cp, um + cm); ap = max(ap, 0.0f);
			float am = min(up - cp, um - cm); am = min(am, 0.0f);
			float qm = hm*um, qp = hp*up;
			float a = max(ap, -am);
			if (a > eps) {
			*fh = (ap*qm - am*qp + ap*am*(hp - hm)) / (ap - am); // (4.5) of [1]
			*fq = (ap*(qm*um + g*sq(hm) / 2.) - am*(qp*up + g*sq(hp) / 2.) +
			ap*am*(qp - qm)) / (ap - am);
			float dt = CFL*Delta / a;
			if (dt < *dtmax)
			*dtmax = dt;
			}
			else
			*fh = *fq = 0.;*/

			if (fh > 0.0)
			{
				fv = (vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx])*fh;
			}
			else
			{
				fv = (vv[i] - dx*dvdx[i])*fh;
			}
			//fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
			//dtmax needs to be stored in an array and reduced at the end
			//dtmax = dtmaxf;
			//dtmaxtmp = min(dtmax, dtmaxtmp);
			/*if (ix == 11 && iy == 0)
			{
			printf("a=%f\t b=%f\t c=%f\t d=%f\n", ap*(qm*um + ga*hm2), -am*(qp*up + ga*hp2),( ap*(qm*um + g*sq(hm) / 2.0f) - am*(qp*up + g*sq(hp) / 2.0f) + ap*am*(qp - qm) ) *ad/100.0f, ad);
			}
			*/
			/*
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhu[i] = fmu * fh;
			Fqux[i] = fmu * (fu - sl);
			Su[i] = fmu * (fu - sr);
			Fqvx[i] = fmu * fv;
		}
		else
		{
			dtmax[i] = 1.0 / 1e-30;
			Fhu[i] = 0.0;
			Fqux[i] = 0.0;
			Su[i] = 0.0;
			Fqvx[i] = 0.0;
		}

	}


}

__global__ void updateKurgY(int nx, int ny, float delta, float g, float eps, float CFL, float * hh, float *zs, float *uu, float * vv, float *dzsdy, float *dhdy, float * dudy, float *dvdy, float *Fhv, float *Fqvy, float *Fquy, float *Sv, float * dtmax)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
//	int tx = threadIdx.x;
//	int ty = threadIdx.y;
	//int  xplus, yplus, xminus, yminus;
	int yminus;
	if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);


		float cm = 1.0f;// 0.1;
		//float fmu = 1.0;
		float fmv = 1.0f;

		//__shared__ float hi[16][16];
		float dhdyi = dhdy[i];
		float dhdymin = dhdy[ix + yminus*nx];
		float hi = hh[i];
		float hn = hh[ix + yminus*nx];
		float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;



		if (hi > eps || hn > eps)
		{
			hn = hh[ix + yminus*nx];
			dx = delta / 2.0f;
			zi = zs[i] - hi;
			zl = zi - dx*(dzsdy[i] - dhdyi);
			zn = zs[ix + yminus*nx] - hn;
			zr = zn + dx*(dzsdy[ix + yminus*nx] - dhdymin);
			zlr = max(zl, zr);

			hl = hi - dx*dhdyi;
			up = vv[i] - dx*dvdy[i];
			hp = max(0.f, hl + zl - zlr);

			hr = hn + dx*dhdymin;
			um = vv[ix + yminus*nx] + dx*dvdy[ix + yminus*nx];
			hm = max(0.f, hr + zr - zlr);

			//// Reimann solver
			float fh, fu, fv,sl,sr;
			//float dtmaxf = 1.0f / 1e-30f;
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
			//kurganovf(hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
			//We can now call one of the approximate Riemann solvers to get the fluxes.
			float cp, cmo, ap, am, qm, qp, a, dlt,ad,hm2,hp2,ga,apm;
			float epsi = 1e-30f;

			cp = sqrtf(g*hp);
			cmo = sqrtf(g*hm);

			ap = max(max(up + cp, um + cmo),0.0f);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo),0.0f);
			//am = min(am, 0.0f);
			ad = 1.0f / (ap - am);
			qm = hm*um;
			qp = hp*up;

			hm2 = sq(hm);
			hp2 = sq(hp);
			a = max(ap, -am);
			ga = g*0.5f;
			apm = ap*am;
			dlt = delta*cm / fmv;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2 ) - am*(qp*up + ga*hp2) +	apm*(qp - qm)) *ad;
				float dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;


			}
			else
			{
				fh = 0.0f;
				fu = 0.0f;
				dtmax[i] = 1.0f / 1e-30f;
			}

			if (fh > 0.0f)
			{
				fv = (uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx])*fh;
			}
			else
			{
				fv = (uu[i] - dx*dudy[i])*fh;
			}
			//fv = (fh > 0.f ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
			/**
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhv[i] = fmv * fh;
			Fqvy[i] = fmv * (fu - sl);
			Sv[i] = fmv * (fu - sr);
			Fquy[i] = fmv* fv;
		}
		else
		{
			dtmax[i] = 1.0f / 1e-30f;
			Fhv[i] = 0.0f;
			Fqvy[i] = 0.0f;
			Sv[i] = 0.0f;
			Fquy[i] = 0.0f;
		}
	}
}

__global__ void updateKurgYD(int nx, int ny, double delta, double g, double eps, double CFL, double * hh, double *zs, double *uu, double * vv, double *dzsdy, double *dhdy, double * dudy, double *dvdy, double *Fhv, double *Fqvy, double *Fquy, double *Sv, double * dtmax)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	//	int tx = threadIdx.x;
	//	int ty = threadIdx.y;
	//int  xplus, yplus, xminus, yminus;
	int yminus;
	if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);


		double cm = 1.0;// 0.1;
						//float fmu = 1.0;
		double fmv = 1.0;

		//__shared__ float hi[16][16];
		double dhdyi = dhdy[i];
		double dhdymin = dhdy[ix + yminus*nx];
		double hi = hh[i];
		double hn = hh[ix + yminus*nx];
		double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;



		if (hi > eps || hn > eps)
		{
			hn = hh[ix + yminus*nx];
			dx = delta / 2.0;
			zi = zs[i] - hi;
			zl = zi - dx*(dzsdy[i] - dhdyi);
			zn = zs[ix + yminus*nx] - hn;
			zr = zn + dx*(dzsdy[ix + yminus*nx] - dhdymin);
			zlr = max(zl, zr);

			hl = hi - dx*dhdyi;
			up = vv[i] - dx*dvdy[i];
			hp = max(0.0, hl + zl - zlr);

			hr = hn + dx*dhdymin;
			um = vv[ix + yminus*nx] + dx*dvdy[ix + yminus*nx];
			hm = max(0.0, hr + zr - zlr);

			//// Reimann solver
			double fh, fu, fv, sl, sr;
			//float dtmaxf = 1.0f / 1e-30f;
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
			//kurganovf(hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
			//We can now call one of the approximate Riemann solvers to get the fluxes.
			double cp, cmo, ap, am, qm, qp, a, dlt, ad, hm2, hp2, ga, apm;
			double epsi = 1e-30;

			cp = sqrt(g*hp);
			cmo = sqrt(g*hm);

			ap = max(max(up + cp, um + cmo), 0.0);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo), 0.0);
			//am = min(am, 0.0f);
			ad = 1.0 / (ap - am);
			qm = hm*um;
			qp = hp*up;

			hm2 = sq(hm);
			hp2 = sq(hp);
			a = max(ap, -am);
			ga = g*0.5;
			apm = ap*am;
			dlt = delta*cm / fmv;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2) - am*(qp*up + ga*hp2) + apm*(qp - qm)) *ad;
				double dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;


			}
			else
			{
				fh = 0.0;
				fu = 0.0;
				dtmax[i] = 1.0 / 1e-30;
			}

			if (fh > 0.0f)
			{
				fv = (uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx])*fh;
			}
			else
			{
				fv = (uu[i] - dx*dudy[i])*fh;
			}
			//fv = (fh > 0.f ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
			/**
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhv[i] = fmv * fh;
			Fqvy[i] = fmv * (fu - sl);
			Sv[i] = fmv * (fu - sr);
			Fquy[i] = fmv* fv;
		}
		else
		{
			dtmax[i] = 1.0 / 1e-30;
			Fhv[i] = 0.0;
			Fqvy[i] = 0.0;
			Sv[i] = 0.0;
			Fquy[i] = 0.0;
		}
	}
}
__global__ void updateKurgYSPH(int nx, int ny, double delta, double g, double eps, double CFL, double yo, double Radius, double * hh, double *zs, double *uu, double * vv, double *dzsdy, double *dhdy, double * dudy, double *dvdy, double *Fhv, double *Fqvy, double *Fquy, double *Sv, double * dtmax)
{
	// Same as updateKurgY but with Spherical coordinate corrections
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	//	int tx = threadIdx.x;
	//	int ty = threadIdx.y;
	//int  xplus, yplus, xminus, yminus;
	int yminus;
	double cm, fmv, phi, dphi, y;

	if (ix < nx && iy < ny)
	{
		//xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		//yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);

		y = yo + iy*delta / Radius*180.0 / pi;;

		phi = y*pi / 180.0;

		dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

		cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

		//fmu = 1.0f;
		fmv = cos(phi);

		//float cm = 1.0f;// 0.1;
						//float fmu = 1.0;
		//float fmv = 1.0f;

		//__shared__ float hi[16][16];
		double dhdyi = dhdy[i];
		double dhdymin = dhdy[ix + yminus*nx];
		double hi = hh[i];
		double hn = hh[ix + yminus*nx];
		double dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;



		if (hi > eps || hn > eps)
		{
			hn = hh[ix + yminus*nx];
			dx = delta / 2.;
			zi = zs[i] - hi;
			zl = zi - dx*(dzsdy[i] - dhdyi);
			zn = zs[ix + yminus*nx] - hn;
			zr = zn + dx*(dzsdy[ix + yminus*nx] - dhdymin);
			zlr = max(zl, zr);

			hl = hi - dx*dhdyi;
			up = vv[i] - dx*dvdy[i];
			hp = max(0.0, hl + zl - zlr);

			hr = hn + dx*dhdymin;
			um = vv[ix + yminus*nx] + dx*dvdy[ix + yminus*nx];
			hm = max(0.0, hr + zr - zlr);

			//// Reimann solver
			double fh, fu, fv, sl, sr;
			//float dtmaxf = 1.0f / 1e-30f;
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
			//kurganovf(hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
			//We can now call one of the approximate Riemann solvers to get the fluxes.
			double cp, cmo, qm, qp, a, dlt, hm2, hp2, ga, apm;
			double ap, am, ad;
			double epsi = 1e-30;

			cp = sqrt(g*hp);
			cmo = sqrt(g*hm);

			ap = max(max(up + cp, um + cmo), 0.0);
			//ap = max(ap, 0.0f);

			am = min(min(up - cp, um - cmo), 0.0);
			//am = min(am, 0.0f);
			ad = 1.0 / (ap - am);
			qm = hm*um;
			qp = hp*up;

			hm2 = sq(hm);
			hp2 = sq(hp);
			a = max(ap, -am);
			ga = g*0.5;
			apm = ap*am;
			dlt = delta*cm / fmv;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + apm*(hp - hm)) *ad;
				fu = (ap*(qm*um + ga*hm2) - am*(qp*up + ga*hp2) + apm*(qp - qm)) *ad;
				double dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;


			}
			else
			{
				fh = 0.0;
				fu = 0.0;
				dtmax[i] = 1.0 / 1e-30;
			}

			if (fh > 0.0)
			{
				fv = (uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx])*fh;
			}
			else
			{
				fv = (uu[i] - dx*dudy[i])*fh;
			}
			//fv = (fh > 0.f ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
			/**
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			sl = ga*(hp2 - sq(hl) + (hl + hi)*(zi - zl));
			sr = ga*(hm2 - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhv[i] = fmv * fh;
			Fqvy[i] = fmv * (fu - sl);
			Sv[i] = fmv * (fu - sr);
			Fquy[i] = fmv* fv;
		}
		else
		{
			dtmax[i] = 1.0 / 1e-30;
			Fhv[i] = 0.0;
			Fqvy[i] = 0.0;
			Sv[i] = 0.0;
			Fquy[i] = 0.0;
		}
	}
}

__global__ void updateEV(int nx, int ny, float delta, float g, float * hh, float *uu, float * vv, float * Fhu, float *Fhv, float * Su, float *Sv, float *Fqux, float *Fquy, float *Fqvx, float *Fqvy, float *dh, float *dhu, float *dhv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
//	int tx = threadIdx.x;
//	int ty = threadIdx.y;
	int  xplus, yplus;

	if (ix < nx && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

		float cm = 1.0f;// 0.1;
		float fmu = 1.0f;
		float fmv = 1.0f;

		float hi = hh[i];
		float uui = uu[i];
		float vvi = vv[i];


		float cmdinv, ga;

		cmdinv = 1.0f / (cm*delta);
		ga = 0.5f*g;
		////
		//vector dhu = vector(updates[1 + dimension*l]);
		//foreach() {
		//	double dhl =
		//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
		//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
		//	foreach_dimension()
		//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);
		//float cm = 1.0;

		dh[i] = -1.0f*(Fhu[xplus + iy*nx] - Fhu[i] + Fhv[ix + yplus*nx] - Fhv[i])*cmdinv;
		//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


		//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
		//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
		float dmdl = (fmu - fmu) / (cm*delta);// absurd if not spherical!
		float dmdt = (fmv - fmv) / (cm*delta);
		float fG = vvi * dmdl - uui * dmdt;
		dhu[i] = (Fqux[i] + Fquy[i] - Su[xplus + iy*nx] - Fquy[ix + yplus*nx]) *cmdinv;
		dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[ix + yplus*nx] - Fqvx[xplus + iy*nx]) *cmdinv;
		//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		dhu[i] += hi * (ga*hi *dmdl + fG*vvi);
		dhv[i] += hi * (ga*hi *dmdt - fG*uui);
	}
}
__global__ void updateEVD(int nx, int ny, double delta, double g, double * hh, double *uu, double * vv, double * Fhu, double *Fhv, double * Su, double *Sv, double *Fqux, double *Fquy, double *Fqvx, double *Fqvy, double *dh, double *dhu, double *dhv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	//	int tx = threadIdx.x;
	//	int ty = threadIdx.y;
	int  xplus, yplus;

	if (ix < nx && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

		double cm = 1.0;// 0.1;
		double fmu = 1.0;
		double fmv = 1.0;

		double hi = hh[i];
		double uui = uu[i];
		double vvi = vv[i];


		double cmdinv, ga;

		cmdinv = 1.0 / (cm*delta);
		ga = 0.5*g;
		////
		//vector dhu = vector(updates[1 + dimension*l]);
		//foreach() {
		//	double dhl =
		//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
		//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
		//	foreach_dimension()
		//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);
		//float cm = 1.0;

		dh[i] = -1.0*(Fhu[xplus + iy*nx] - Fhu[i] + Fhv[ix + yplus*nx] - Fhv[i])*cmdinv;
		//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


		//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
		//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
		double dmdl = (fmu - fmu) / (cm*delta);// absurd if not spherical!
		double dmdt = (fmv - fmv) / (cm*delta);
		double fG = vvi * dmdl - uui * dmdt;
		dhu[i] = (Fqux[i] + Fquy[i] - Su[xplus + iy*nx] - Fquy[ix + yplus*nx]) *cmdinv;
		dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[ix + yplus*nx] - Fqvx[xplus + iy*nx]) *cmdinv;
		//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		dhu[i] += hi * (ga*hi *dmdl + fG*vvi);
		dhv[i] += hi * (ga*hi *dmdt - fG*uui);
	}
}
__global__ void updateEVSPH(int nx, int ny, double delta, double g, double yo, double Radius, double * hh, double *uu, double * vv, double * Fhu, double *Fhv, double * Su, double *Sv, double *Fqux, double *Fquy, double *Fqvx, double *Fqvy, double *dh, double *dhu, double *dhv)
{
	// Same as updateEV but with Spherical coordinate corrections
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	//	int tx = threadIdx.x;
	//	int ty = threadIdx.y;
	int  xplus, yplus;// , xminus, yminus;

	double cm, fmu, fmv, y, phi, dphi,fmvp,fmup;


	if (ix < nx && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		//xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		//yminus = max(iy - 1, 0);

		y = yo + iy*delta / Radius*180.0 / pi;
		double yp= yo + yplus*delta / Radius*180.0 / pi;
		phi = y*pi / 180.0;

		dphi = delta / (2.0*Radius);// dy*0.5f*pi/180.0f;

		cm = (sin(phi + dphi) - sin(phi - dphi)) / (2.0*dphi);

		fmu = 1.0;
		fmup = 1.0;
		fmv = cos(phi);
		fmvp = cos(yp*pi/180.0);
		//float cm = 1.0f;// 0.1;
		//float fmu = 1.0f;
		//float fmv = 1.0f;

		double hi = hh[i];
		double uui = uu[i];
		double vvi = vv[i];


		float cmdinv, ga;

		cmdinv = 1.0 / (cm*delta);
		ga = 0.5*g;
		////
		//vector dhu = vector(updates[1 + dimension*l]);
		//foreach() {
		//	double dhl =
		//		layer[l] * (Fh.x[1, 0] - Fh.x[] + Fh.y[0, 1] - Fh.y[]) / (cm[] * Δ);
		//	dh[] = -dhl + (l > 0 ? dh[] : 0.);
		//	foreach_dimension()
		//		dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		//		dhu.y[] = (Fq.y.y[] + Fq.y.x[] - S.y[0,1] - Fq.y.x[1,0])/(cm[]*Delta);
		//float cm = 1.0;

		dh[i] = -1.0*(Fhu[xplus + iy*nx] - Fhu[i] + Fhv[ix + yplus*nx] - Fhv[i])*cmdinv;
		//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


		//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
		//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
		float dmdl = (fmup - fmu) / (cm*delta);// absurd even for spherical because fmu==1 always! What's up with that?
		float dmdt = (fmvp - fmv) / (cm*delta);
		float fG = vvi * dmdl - uui * dmdt;
		dhu[i] = (Fqux[i] + Fquy[i] - Su[xplus + iy*nx] - Fquy[ix + yplus*nx]) *cmdinv;
		dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[ix + yplus*nx] - Fqvx[xplus + iy*nx]) *cmdinv;
		//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		dhu[i] += hi * (ga*hi *dmdl + fG*vvi);
		dhv[i] += hi * (ga*hi *dmdt - fG*uui);
	}
}

template <class T> __global__ void Advkernel(int nx, int ny, T dt, T eps, T * hh, T *zb, T *uu, T * vv, T *dh, T *dhu, T * dhv, T *zso, T *hho, T *uuo, T *vvo )
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;

	if (ix < nx && iy < ny)
	{

		T hold = hh[i];
		T ho, uo, vo;
		ho = hold + dt*dh[i];


		if (ho > eps) {
			//for (int l = 0; l < nl; l++) {
			//vector uo = vector(output[1 + dimension*l]);
			//vector ui = vector(input[1 + dimension*l]),
			//dhu = vector(updates[1 + dimension*l]);
			//foreach_dimension()
			uo = (hold*uu[i] + dt*dhu[i]) / ho;
			vo = (hold*vv[i] + dt*dhv[i]) / ho;
			//}


			//In the case of [multiplelayers](multilayer.h#viscous-friction-between-layers) we add the
			//viscous friction between layers.


		}
		else
		{// dry
			//for (int l = 0; l < nl; l++) {
			//vector uo = vector(output[1 + dimension*l]);
			//foreach_dimension()
			uo = T(0.0);
			vo = T(0.0);
		}


		zso[i] = zb[i] + ho;
		hho[i] = ho;
		uuo[i] = uo;
		vvo[i] = vo;
	}

}



template <class T> __global__ void cleanupGPU(int nx, int ny, T * hhi, T *zsi, T *uui, T *vvi, T * hho, T *zso, T *uuo, T *vvo)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	if (ix < nx && iy < ny)
	{
		hho[i] = hhi[i];
		zso[i] = zsi[i];
		uuo[i] = uui[i];
		vvo[i] = vvi[i];
	}
}

template <class T> __global__ void initdtmax(int nx, int ny, T epsi,T *dtmax)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	
		dtmax[i] = T(1.0) / epsi;
	
}

__global__ void minmaxKernel(int ntot, float *max, float *min, float *a) {
	__shared__ double maxtile[32];
	__shared__ double mintile[32];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < ntot)
	{
		maxtile[tid] = a[i];
		mintile[tid] = a[i];
		__syncthreads();

		// strided index and non-divergent branch
		for (unsigned int s = 1; s < blockDim.x; s *= 2) {
			int index = 2 * s * tid;
			if (index < blockDim.x) {
				if (maxtile[tid + s] > maxtile[tid])
					maxtile[tid] = maxtile[tid + s];
				if (mintile[tid + s] < mintile[tid])
					mintile[tid] = mintile[tid + s];
			}
			__syncthreads();
		}

		if (tid == 0) {
			max[blockIdx.x] = maxtile[0];
			min[blockIdx.x] = mintile[0];
		}
	}
}

__global__ void finalminmaxKernel(float *max, float *min) {
	__shared__ double maxtile[32];
	__shared__ double mintile[32];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	maxtile[tid] = max[i];
	mintile[tid] = min[i];
	__syncthreads();

	// strided index and non-divergent branch
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
			if (mintile[tid + s] < mintile[tid])
				mintile[tid] = mintile[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		max[blockIdx.x] = maxtile[0];
		min[blockIdx.x] = mintile[0];
	}
}
template <class T>
__global__ void resetdtmax(int nx, int ny, T *dtmax)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	T initdt= T(1.0 / 1e-30);
	if (ix < nx && iy < ny)
	{
		dtmax[i] = initdt;
	}
}

__global__ void reduce3(float *g_idata, float *g_odata, unsigned int n)
{
	//T *sdata = SharedMemory<T>();
	extern __shared__ float sdata[];
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	float mySum = (i < n) ? g_idata[i] : 0;

	if (i + blockDim.x < n)
		mySum += g_idata[i + blockDim.x];

	sdata[tid] = mySum;
	__syncthreads();
	

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = mySum = mySum + sdata[tid + s];
		}

		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = mySum;
}

__global__ void reducemax3(float *g_idata, float *g_odata, unsigned int n)
{
	//T *sdata = SharedMemory<T>();
	extern __shared__ float sdata[];
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	float myMax = (i < n) ? g_idata[i] : -1e-30f;

	if (i + blockDim.x < n)
		myMax = max(myMax,g_idata[i + blockDim.x]);

	sdata[tid] = myMax;
	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = myMax = max(myMax, sdata[tid + s]);
		}

		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = myMax;
}

template <class T> __global__ void reducemin3(T *g_idata, T *g_odata, unsigned int n)
{
	//T *sdata = SharedMemory<T>();
	T *sdata = SharedMemory<T>();
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	T myMin = (i < n) ? g_idata[i] : T(1e30);

	if (i + blockDim.x < n)
		myMin = min(myMin, g_idata[i + blockDim.x]);

	sdata[tid] = myMin;
	__syncthreads();


	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = myMin = min(myMin, sdata[tid + s]);
		}

		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = myMin;
}


/*
template <unsigned int blockSize>
__global__ void reducemin6(int *g_idata, int *g_odata, unsigned int n)
{
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	sdata[tid] = 1e30f;
	while (i < n) { sdata[tid] = min(g_idata[i], min( g_idata[i + blockSize],sdata[tid])); i += gridSize; }
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid + 256], sdata[tid]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid + 128], sdata[tid]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] = min(sdata[tid + 64], sdata[tid]); } __syncthreads(); }
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] = min(sdata[tid + 32], sdata[tid]);
		if (blockSize >= 32) sdata[tid] = min(sdata[tid + 16], sdata[tid]);
		if (blockSize >= 16) sdata[tid] = min(sdata[tid + 8], sdata[tid]);
		if (blockSize >= 8) sdata[tid] = min(sdata[tid + 4], sdata[tid]);
		if (blockSize >= 4) sdata[tid] = min(sdata[tid + 2], sdata[tid]);
		if (blockSize >= 2) sdata[tid] = min(sdata[tid + 1], sdata[tid]);
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
*/

__global__ void leftdirichlet(int nx, int ny,int nybnd,float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv )
{
	int iy = blockIdx.x*blockDim.x + threadIdx.x;
	int ix = 0;
	//int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int xplus;
	//float hhi;
	float zsbnd;
	float itx = (iy*1.0f / ny*1.0f) / (1.0f / (1.0f*nybnd - 1.0f));//Bleark!
	zsbnd = tex2D(texLBND, itime+0.5f, itx+0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(?) 
	if (ix == 0 && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		hh[i] = zsbnd-zb[i];
		zs[i] = zsbnd;
		uu[i] = -2.0f*(sqrtf(g*max(hh[xplus + iy*nx], 0.0f)) - sqrtf(g*max(zsbnd - zb[xplus + iy*nx], 0.0f))) + uu[xplus + iy*nx];
		vv[i] = 0.0f;
		//if (iy == 0)
		//{
		//	printf("zsbnd=%f\t", zsbnd);
		//}
	}
}

__global__ void leftdirichletD(int nx, int ny, int nybnd, double g, double itime, double *zs, double *zb, double *hh, double *uu, double *vv)
{
	int iy = blockIdx.x*blockDim.x + threadIdx.x;
	int ix = 0;
	//int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int xplus;
	//float hhi;
	float zsbnd; //remains a float because this is how it is stored on the texture memory // I don't think it is a big deal
	float itx = (iy*1.0f / ny*1.0f) / (1.0f / (1.0f*nybnd - 1.0f));//Bleark!
	zsbnd = tex2D(texLBND, itime + 0.5f, itx + 0.5f); // textures use pixel registration so index of 0 is actually located at 0.5...(?) 
	if (ix == 0 && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		uu[i] = -2.0*(sqrt(g*max(hh[xplus + iy*nx], 0.0)) - sqrt(g*max(zsbnd - zb[xplus + iy*nx], 0.0))) + uu[xplus + iy*nx];
		vv[i] = 0.0;
		//if (iy == 0)
		//{
		//	printf("zsbnd=%f\t", zsbnd);
		//}
	}
}


__global__ void rightdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	int iy = blockIdx.x*blockDim.x + threadIdx.x;
	int ix = nx-1;
	//int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int xminus;
	//float hhi;
	float zsbnd;
	float itx = (iy*1.0f / ny*1.0f) / (1.0f / (1.0f*nybnd - 1.0f));//Bleark!
	zsbnd = tex2D(texRBND, itime+0.5f, itx+0.5f);
	if (ix == nx-1 && iy < ny)
	{
		xminus = max(ix - 1, 0);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		uu[i] = +2.0f*(sqrtf(g*max(hh[xminus + iy*nx], 0.0f)) - sqrtf(g*max(zsbnd - zb[xminus + iy*nx], 0.0f))) + uu[xminus + iy*nx];
		vv[i] = 0.0f;
	}
}

__global__ void rightdirichletD(int nx, int ny, int nybnd, double g, double itime, double *zs, double *zb, double *hh, double *uu, double *vv)
{
	int iy = blockIdx.x*blockDim.x + threadIdx.x;
	int ix = nx - 1;
	//int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int xminus;
	//float hhi;
	float zsbnd;
	float itx = (iy*1.0f / ny*1.0f) / (1.0f / (1.0f*nybnd - 1.0f));//Bleark!
	zsbnd = tex2D(texRBND, itime + 0.5f, itx + 0.5f);
	if (ix == nx - 1 && iy < ny)
	{
		xminus = max(ix - 1, 0);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		uu[i] = +2.0*(sqrt(g*max(hh[xminus + iy*nx], 0.0)) - sqrt(g*max(zsbnd - zb[xminus + iy*nx], 0.0))) + uu[xminus + iy*nx];
		vv[i] = 0.0f;
	}
}

__global__ void topdirichlet(int nx, int ny, int nxbnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = ny-1;
	//int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int yminus;
	//float hhi;
	float zsbnd;
	float itx = (ix*1.0f / nx*1.0f) / (1.0f / (1.0f*nxbnd - 1.0f));//Bleark!
	zsbnd = tex2D(texTBND, itime + 0.5f, itx + 0.5f);
	if (iy == ny-1 && ix < nx)
	{
		yminus = max(iy - 1, 0);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		vv[i] = +2.0f*(sqrtf(g*max(hh[ix + yminus*nx], 0.0f)) - sqrtf(g*max(zsbnd - zb[ix + yminus*nx], 0.0f))) + vv[ix + yminus*nx];
		uu[i] = 0.0f;
	}
}

__global__ void topdirichletD(int nx, int ny, int nxbnd, double g, double itime, double *zs, double *zb, double *hh, double *uu, double *vv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = ny - 1;
	//int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int yminus;
	//float hhi;
	float zsbnd;
	float itx = (ix*1.0f / nx*1.0f) / (1.0f / (1.0f*nxbnd - 1.0f));//Bleark!
	zsbnd = tex2D(texTBND, itime + 0.5f, itx + 0.5f);
	if (iy == ny - 1 && ix < nx)
	{
		yminus = max(iy - 1, 0);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		vv[i] = +2.0*(sqrt(g*max(hh[ix + yminus*nx], 0.0)) - sqrt(g*max(zsbnd - zb[ix + yminus*nx], 0.0))) + vv[ix + yminus*nx];
		uu[i] = 0.0;
	}
}
__global__ void botdirichlet(int nx, int ny, int nxbnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = 0;
	//int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int yplus;
	//float hhi;
	float zsbnd;
	float itx = (ix*1.0f / nx*1.0f) / (1.0f / (1.0f*nxbnd - 1.0f));//Bleark!
	zsbnd = tex2D(texBBND, itime + 0.5f, itx + 0.5f);
	if (iy == 0 && ix < nx)
	{
		yplus = min(iy + 1, ny-1);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		vv[i] = -2.0f*(sqrtf(g*max(hh[ix + yplus*nx], 0.0f)) - sqrtf(g*max(zsbnd - zb[ix + yplus*nx], 0.0f))) + vv[ix + yplus*nx];
		uu[i] = 0.0f;
	}
}

__global__ void botdirichletD(int nx, int ny, int nxbnd, double g, double itime, double*zs, double *zb, double *hh, double *uu, double *vv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = 0;
	//int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int yplus;
	//float hhi;
	float zsbnd;
	float itx = (ix*1.0f / nx*1.0f) / (1.0f / (1.0f*nxbnd - 1.0f));//Bleark!
	zsbnd = tex2D(texBBND, itime + 0.5f, itx + 0.5f);
	if (iy == 0 && ix < nx)
	{
		yplus = min(iy + 1, ny - 1);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		vv[i] = -2.0*(sqrt(g*max(hh[ix + yplus*nx], 0.0)) - sqrt(g*max(zsbnd - zb[ix + yplus*nx], 0.0))) + vv[ix + yplus*nx];
		uu[i] = 0.0;
	}
}

template <class T> __global__ void quadfriction(int nx, int ny, T dt,T eps, T cf, T *hh, T *uu, T *vv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	
	T normu,hhi,uui,vvi;
	
	if (ix < nx && iy < ny)
	{

		hhi = hh[i];
		uui = uu[i];
		vvi = vv[i];
		if (hhi > eps)
		{
			normu = uui * uui + vvi * vvi;
			T frc = (T(1.0) + dt*cf*(normu) / hhi);
				//u.x[] = h[]>dry ? u.x[] / (1 + dt*cf*norm(u) / h[]) : 0.;
			uu[i] = uui / frc;
			vv[i] = vvi / frc;
		}
		
	}

}


__global__ void noslipbndall(int nx, int ny, float dt, float eps, float *zb, float *zs, float *hh, float *uu, float *vv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int  xplus, yplus, xminus, yminus;
	//float normu, hhi;

	if (ix < nx && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);

		if (ix == 0 )
		{
			uu[i] = 0.0f;
			zs[i] = zs[xplus+iy*nx];
			hh[i] = max(zs[xplus + iy*nx]-zb[i],eps);
		}
		if ( ix == nx - 1)
		{
			uu[i] = 0.0f;
			zs[i] = zs[xminus + iy*nx];
			hh[i] = max(zs[xminus + iy*nx] - zb[i], eps);

		}

		if ( iy == 0 )
		{
			vv[i] = 0.0f;
			zs[i] = zs[ix + yplus*nx];
			hh[i] = max(zs[ix + yplus*nx] - zb[i], eps);
		}
		if ( iy == ny - 1)
		{
			vv[i] = 0.0f;
			zs[i] = zs[ix + yminus*nx];
			hh[i] = max(zs[ix + yminus*nx] - zb[i], eps);

		}

	}

}
template <class T> __global__ void noslipbndLeft(int nx, int ny, T eps, T *zb, T *zs, T *hh, T *uu, T *vv)
{
	int iy = blockIdx.x*blockDim.x + threadIdx.x;
	int ix = 0;
	int i = ix + iy*nx;
	int  xplus;
	T zsp;

	if (ix ==0 && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		

		zsp = zs[xplus + iy*nx];

		
		uu[i] = T(0.0);
		zs[i] = zsp;
		hh[i] = max(zsp - zb[i], eps);
		
		

	}

}
template <class T> __global__ void noslipbndBot(int nx, int ny, T eps, T *zb, T *zs, T *hh, T *uu, T *vv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = 0;
	int i = ix + iy*nx;
	int yplus;
	
	if (iy == 0 && ix < nx)
	{
		yplus = min(iy + 1, ny - 1);
		
		vv[i] = T(0.0);
		zs[i] = zs[ix + yplus*nx];
		hh[i] = max(zs[ix + yplus*nx] - zb[i], eps);



	}

}


template <class T> __global__ void noslipbndRight(int nx, int ny, T eps, T *zb, T *zs, T *hh, T *uu, T *vv)
{
	int iy = blockIdx.x*blockDim.x + threadIdx.x;
	int ix = nx-1;
	int i = ix + iy*nx;
	int xminus;
	

	if (ix == nx-1 && iy < ny)
	{
		xminus = max(ix - 1, 0);
		
		uu[i] = T(0.0);
		zs[i] = zs[xminus + iy*nx];
		hh[i] = max(zs[xminus + iy*nx] - zb[i], eps);



	}

}
template <class T> __global__ void noslipbndTop(int nx, int ny, T eps, T *zb, T *zs, T *hh, T *uu, T *vv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = ny - 1;
	int i = ix + iy*nx;
	int yminus;
	

	if (iy == ny - 1 && ix < nx)
	{
		yminus = max(iy - 1, 0);


		vv[i] = T(0.0);
		zs[i] = zs[ix + yminus*nx];
		hh[i] = max(zs[ix + yminus*nx] - zb[i], eps);



	}

}

template <class T>
__global__ void storeTSout(int nx, int ny,int noutnodes, int outnode, int istep, int inode,int jnode, T *zs, T *hh, T *uu, T *vv,T * store)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	

	if (ix == inode && iy == jnode )
	{
		store[0 + outnode * 4 + istep*noutnodes * 4] = hh[i];
		store[1 + outnode * 4 + istep*noutnodes * 4] = zs[i];
		store[2 + outnode * 4 + istep*noutnodes * 4] = uu[i];
		store[3 + outnode * 4 + istep*noutnodes * 4] = vv[i];
	}
}


template <class T>
__global__ void addavg_var(int nx, int ny, T * Varmean, T * Var)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	__shared__ T mvari[16][16];
	__shared__ T vari[16][16];

	if (ix < nx && iy < ny)
	{

		mvari[tx][ty] = Varmean[i];
		vari[tx][ty] = Var[i];

		Varmean[i] = mvari[tx][ty] + vari[tx][ty];
	}


}

template <class T>
__global__ void divavg_var(int nx, int ny, T ntdiv, T * Varmean)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	__shared__ T mvari[16][16];
	if (ix < nx && iy < ny)
	{
		mvari[tx][ty] = Varmean[i];
		Varmean[i] = mvari[tx][ty] / ntdiv;
	}


}

template <class T>
__global__ void resetavg_var(int nx, int ny, T * Varmean)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	if (ix < nx && iy < ny)
	{
		Varmean[i] = T(0.0);
	}
}

template <class T>
__global__ void resetmax_var(int nx, int ny, T * Varmax)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	if (ix < nx && iy < ny)
	{
		Varmax[i] = T(-1.0/epsilone);
	}
}

template <class T>
__global__ void max_var(int nx, int ny, T * Varmax, T * Var)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	__shared__ T mvari[16][16];
	__shared__ T vari[16][16];

	if (ix < nx && iy < ny)
	{

		mvari[tx][ty] = Varmax[i];
		vari[tx][ty] = Var[i];

		Varmax[i] = max(mvari[tx][ty], vari[tx][ty]);
	}


}

template <class T>
__global__ void CalcVorticity(int nx, int ny, T * Vort,T * dvdx, T * dudy)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	__shared__ T dvdxi[16][16];
	__shared__ T dudyi[16][16];

	if (ix < nx && iy < ny)
	{

		dvdxi[tx][ty] = dvdx[i];
		dudyi[tx][ty] = dudy[i];

		Vort[i] = dvdxi[tx][ty] - dudyi[tx][ty];
	}


}

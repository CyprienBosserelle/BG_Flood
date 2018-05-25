
__device__ float sq(float a)
{
	return a*a;
}


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

__global__ void gradientGPUX(int nx, int ny, float delta, float *a, float *dadx)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int xplus, yplus, xminus, yminus;

	if (ix < nx && iy < ny)
	{

		//
		//
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);
		i = ix + iy*nx;


		//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
		dadx[i] = minmod2fGPU(a[xminus + iy*nx], a[i], a[xplus + iy*nx]) / delta;

	}



}

__global__ void gradientGPUY(int nx, int ny, float delta, float *a, float *dady)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int  xplus, yplus, xminus, yminus;

	if (ix < nx && iy < ny)
	{
		//
		//
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);
		i = ix + iy*nx;


		//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
		dady[i] = minmod2fGPU(a[ix + yminus*nx], a[i], a[ix + yplus*nx]) / delta;

	}



}


__global__ void updateKurgX( int nx, int ny, float delta, float g, float eps,float CFL, float * hh, float *zs, float *uu, float * vv, float *dzsdx, float *dhdx, float * dudx, float *dvdx, float *Fhu, float *Fqux, float *Fqvx, float *Su, float * dtmax)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int  xplus, yplus, xminus, yminus;

	if (ix < nx && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);

		float cm = 1.0;// 0.1;
		float fmu = 1.0;
		float fmv = 1.0;

		//__shared__ float hi[16][16];
		float hi = hh[i];

		float hn = hh[xminus + iy*nx];


		if (hi > eps || hn > eps)
		{
			float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;

			// along X
			dx = delta / 2.;
			zi = zs[i] - hi;

			//printf("%f\n", zi);


			zl = zi - dx*(dzsdx[i] - dhdx[i]);
			//printf("%f\n", zl);

			zn = zs[xminus + iy*nx] - hn;

			//printf("%f\n", zn);
			zr = zn + dx*(dzsdx[xminus + iy*nx] - dhdx[xminus + iy*nx]);


			zlr = max(zl, zr);

			hl = hi - dx*dhdx[i];
			up = uu[i] - dx*dudx[i];
			hp = max(0.f, hl + zl - zlr);

			hr = hn + dx*dhdx[xminus + iy*nx];
			um = uu[xminus + iy*nx] + dx*dudx[xminus + iy*nx];
			hm = max(0.f, hr + zr - zlr);

			float fh, fu, fv;
			float dtmaxf = 1 / (float)epsilon;

			//We can now call one of the approximate Riemann solvers to get the fluxes.
			float cp, cm, ap, am, qm, qp, a, dlt;
			float epsi = epsilon;

			cp = sqrtf(g*hp);
			cm = sqrtf(g*hm);

			ap = max(up + cp, um + cm);
			ap = max(ap, 0.0f);

			am = min(up - cp, um - cm);
			am = min(am, 0.0f);

			qm = hm*um;
			qp = hp*up;

			a = max(ap, -am);

			dlt = delta*cm / fmu;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + ap*am*(hp - hm)) / (ap - am);
				fu = (ap*(qm*um + g*sq(hm) / 2.) - am*(qp*up + g*sq(hp) / 2.) +
					ap*am*(qp - qm)) / (ap - am);
				float dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;


			}
			else
			{
				fh = fu = 0.f;
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


			fv = (fh > 0.f ? vv[xminus + iy*nx] + dx*dvdx[xminus + iy*nx] : vv[i] - dx*dvdx[i])*fh;
			//dtmax needs to be stored in an array and reduced at the end
			//dtmax = dtmaxf;
			//dtmaxtmp = min(dtmax, dtmaxtmp);


			/*
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			float sl = g / 2.f*(sq(hp) - sq(hl) + (hl + hi)*(zi - zl));
			float sr = g / 2.f*(sq(hm) - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhu[i] = fmu * fh;
			Fqux[i] = fmu * (fu - sl);
			Su[i] = fmu * (fu - sr);
			Fqvx[i] = fmu * fv;
		}
		else
		{
			dtmax[i] = 1.0f / (float)epsilon;
			Fhu[i] = 0.0f;
			Fqux[i] = 0.0f;
			Su[i] = 0.0f;
			Fqvx[i] = 0.0f;
		}

	}


}

__global__ void updateKurgY(int nx, int ny, float delta, float g, float eps, float CFL, float * hh, float *zs, float *uu, float * vv, float *dzsdy, float *dhdy, float * dudy, float *dvdy, float *Fhv, float *Fqvy, float *Fquy, float *Sv, float * dtmax)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int  xplus, yplus, xminus, yminus;

	if (ix < nx && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);


		float cm = 1.0;// 0.1;
		float fmu = 1.0;
		float fmv = 1.0;

		//__shared__ float hi[16][16];
		float hi = hh[i];
		float hn = hh[ix + yminus*nx];
		float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;



		if (hi > eps || hn > eps)
		{
			hn = hh[ix + yminus*nx];
			dx = delta / 2.;
			zi = zs[i] - hi;
			zl = zi - dx*(dzsdy[i] - dhdy[i]);
			zn = zs[ix + yminus*nx] - hn;
			zr = zn + dx*(dzsdy[ix + yminus*nx] - dhdy[ix + yminus*nx]);
			zlr = max(zl, zr);

			hl = hi - dx*dhdy[i];
			up = vv[i] - dx*dvdy[i];
			hp = max(0.f, hl + zl - zlr);

			hr = hn + dx*dhdy[ix + yminus*nx];
			um = vv[ix + yminus*nx] + dx*dvdy[ix + yminus*nx];
			hm = max(0.f, hr + zr - zlr);

			//// Reimann solver
			float fh, fu, fv;
			float dtmaxf = 1 / (float)epsilon;
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
			//kurganovf(hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
			//We can now call one of the approximate Riemann solvers to get the fluxes.
			float cp, cm, ap, am, qm, qp, a, dlt;
			float epsi = epsilon;

			cp = sqrtf(g*hp);
			cm = sqrtf(g*hm);

			ap = max(up + cp, um + cm);
			ap = max(ap, 0.0f);

			am = min(up - cp, um - cm);
			am = min(am, 0.0f);

			qm = hm*um;
			qp = hp*up;

			a = max(ap, -am);

			dlt = delta*cm / fmv;

			if (a > epsi)
			{
				fh = (ap*qm - am*qp + ap*am*(hp - hm)) / (ap - am);
				fu = (ap*(qm*um + g*sq(hm) / 2.) - am*(qp*up + g*sq(hp) / 2.) +
					ap*am*(qp - qm)) / (ap - am);
				float dt = CFL*dlt / a;
				if (dt < dtmax[i])
				{
					dtmax[i] = dt;
				}
				//	*dtmax = dt;


			}
			else
			{
				fh = fu = 0.f;
			}
			fv = (fh > 0. ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
			/**
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			float sl = g / 2.*(sq(hp) - sq(hl) + (hl + hi)*(zi - zl));
			float sr = g / 2.*(sq(hm) - sq(hr) + (hr + hn)*(zn - zr));

			////Flux update

			Fhv[i] = fmv * fh;
			Fqvy[i] = fmv * (fu - sl);
			Sv[i] = fmv * (fu - sr);
			Fquy[i] = fmv* fv;
		}
		else
		{
			dtmax[i] = 1.0f / (float)epsilon;
			Fhv[i] = 0.0f;
			Fqvy[i] = 0.0f;
			Sv[i] = 0.0f;
			Fquy[i] = 0.0f;
		}
	}
}

__global__ void updateEV(int nx, int ny, float delta, float g, float * hh, float *uu, float * vv, float * Fhu, float *Fhv, float * Su, float *Sv, float *Fqux, float *Fquy, float *Fqvx, float *Fqvy, float *dh, float *dhu, float *dhv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int  xplus, yplus, xminus, yminus;

	if (ix < nx && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);

		float cm = 1.0;// 0.1;
		float fmu = 1.0;
		float fmv = 1.0;

		float hi = hh[i];
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

		dh[i] = -1.0*(Fhu[xplus + iy*nx] - Fhu[i] + Fhv[ix + yplus*nx] - Fhv[i]) / (cm * delta);
		//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


		//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
		//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
		float dmdl = (fmu - fmu) / (cm * delta);// absurd!
		float dmdt = (fmv - fmv) / (cm  * delta);// absurd!
		float fG = vv[i] * dmdl - uu[i] * dmdt;
		dhu[i] = (Fqux[i] + Fquy[i] - Su[xplus + iy*nx] - Fquy[ix + yplus*nx]) / (cm*delta);
		dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[ix + yplus*nx] - Fqvx[xplus + iy*nx]) / (cm*delta);
		//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		dhu[i] += hi * (g*hi / 2.*dmdl + fG*vv[i]);
		dhv[i] += hi * (g*hi / 2.*dmdt - fG*uu[i]);
	}
}

__global__ void Advkernel(int nx, int ny, float dt, float eps, float * hh, float *zb, float *uu, float * vv, float *dh, float *dhu, float * dhv, float *zso, float *hho, float *uuo, float *vvo )
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;

	if (ix < nx && iy < ny)
	{

		float hold = hh[i];
		float ho, uo, vo;
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
			uo = 0.f;
			vo = 0.f;
		}


		zso[i] = zb[i] + ho;
		hho[i] = ho;
		uuo[i] = uo;
		vvo[i] = vo;
	}

}



__global__ void cleanupGPU(int nx, int ny, float * hhi, float *zsi, float *uui, float *vvi, float * hho, float *zso, float *uuo, float *vvo)
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

__global__ void initdtmax(int nx, int ny, float epsi,float *dtmax)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	
		dtmax[i] = 1.0f / epsi;
	
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

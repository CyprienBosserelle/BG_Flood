
// textures have to be declared here...
texture<float, 2, cudaReadModeElementType> texLBND;
texture<float, 2, cudaReadModeElementType> texRBND;
texture<float, 2, cudaReadModeElementType> texTBND;
texture<float, 2, cudaReadModeElementType> texBBND;


__device__ float sq(float a)
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

__global__ void gradientGPUX(int nx, int ny,float theta, float delta, float *a, float *dadx)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int i = ix + iy*nx;
	int xplus, xminus;

	__shared__ float a_s[18][16]; // Hard wired stuff Be carefull
	//__shared__ float al_s[16][16];
	//__shared__ float ar_s[16][16];
	float dadxi=0.0f;
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
		
		a_s[tx][ty] = a[xminus + iy*nx];

		// read the halo around the tile
		
		if (threadIdx.x == 15)//blockDim.x - 1
		{
			a_s[tx + 1][ty] = a[i];
			a_s[tx + 2][ty] = a[xplus + iy*nx];
			
		}

		// Need to wait for threadX 0 and threadX 16-1 to finish
		__syncthreads;
		

		//dadx[i] = (a[i] - a[xminus + iy*nx]) / delta;//minmod2(a[xminus+iy*nx], a[i], a[xplus+iy*nx]);
		//dadx[i] = minmod2fGPU(theta,a[xminus + iy*nx], a[i], a[xplus + iy*nx]) / delta;
		// These below are somewhat slower when using shared mem. I'm unsure why (bank conflict?)
		//dadx[i] = minmod2fGPU(theta, al_s[tx][ty], a_s[tx][ty], ar_s[tx][ty]) / delta;
		//dadx[i] = minmod2fGPU(theta, a_s[tx][ty], a_s[tx+1][ty], a_s[tx+2][ty]) / delta;
		//__device__ float minmod2fGPU(float theta,float s0, float s1, float s2)
		float d1, d2, d3;
		if (a_s[tx][ty] < a_s[tx + 1][ty] && a_s[tx + 1][ty] < a_s[tx + 2][ty]) {
			d1 = theta*(a_s[tx + 1][ty] - a_s[tx][ty]);
			d2 = (a_s[tx + 2][ty] - a_s[tx][ty]) / 2.0f;
			d3 = theta*(a_s[tx + 2][ty] - a_s[tx + 1][ty]);
			if (d2 < d1) d1 = d2;
			dadxi=min(d1, d3);
		}
		if (a_s[tx][ty] > a_s[tx + 1][ty] && a_s[tx + 1][ty] > a_s[tx + 2][ty]) {
			d1 = theta*(a_s[tx + 1][ty] - a_s[tx][ty]);
			d2 = (a_s[tx + 2][ty] - a_s[tx][ty]) / 2.0f;
			d3 = theta*(a_s[tx + 2][ty] - a_s[tx + 1][ty]);
			if (d2 > d1) d1 = d2;
			dadxi=max(d1, d3);
		}
		dadx[i] = dadxi;

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
__global__ void gradientGPUY(int nx, int ny, float theta, float delta, float *a, float *dady)
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
		dady[i] = dadyi;
	}



}
__global__ void gradientGPUYOLD(int nx, int ny,float theta, float delta, float *a, float *dady)
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
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int  xplus, yplus, xminus, yminus;

	if (ix < nx && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);


		



		float dhdxi= dhdx[i];
		float cm = 1.0f;// 0.1;
		float fmu = 1.0f;
		//float fmv = 1.0;

		//__shared__ float hi[16][16];
		float hi = hh[i];

		float hn = hh[xminus + iy*nx];


		if (hi > eps || hn > eps)
		{
			float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;

			// along X
			dx = delta / 2.0f;
			zi = zs[i] - hi;

			//printf("%f\n", zi);


			//zl = zi - dx*(dzsdx[i] - dhdx[i]);
			zl = zi - dx*(dzsdx[i] - dhdxi);
			//printf("%f\n", zl);

			zn = zs[xminus + iy*nx] - hn;

			//printf("%f\n", zn);
			zr = zn + dx*(dzsdx[xminus + iy*nx] - dhdx[xminus + iy*nx]);


			zlr = max(zl, zr);

			//hl = hi - dx*dhdx[i];
			hl = hi - dx*dhdxi;
			up = uu[i] - dx*dudx[i];
			hp = max(0.0f, hl + zl - zlr);

			hr = hn + dx*dhdx[xminus + iy*nx];
			um = uu[xminus + iy*nx] + dx*dudx[xminus + iy*nx];
			hm = max(0.0f, hr + zr - zlr);

			float fh, fu, fv;
			//float dtmaxf = 1 / 1e-30f;

			//We can now call one of the approximate Riemann solvers to get the fluxes.
			float cp, cmo, ap, am, qm, qp, a, dlt;
			float epsi = 1e-30f;

			cp = sqrtf(g*hp);
			cmo = sqrtf(g*hm);

			ap = max(up + cp, um + cmo);
			ap = max(ap, 0.0f);

			am = min(up - cp, um - cmo);
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
			dtmax[i] = 1.0f / 1e-30f;
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
//	int tx = threadIdx.x;
//	int ty = threadIdx.y;
	int  xplus, yplus, xminus, yminus;

	if (ix < nx && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);


		float cm = 1.0f;// 0.1;
		//float fmu = 1.0;
		float fmv = 1.0f;

		//__shared__ float hi[16][16];
		float dhdyi = dhdy[i];
		float hi = hh[i];
		float hn = hh[ix + yminus*nx];
		float dx, zi, zl, zn, zr, zlr, hl, up, hp, hr, um, hm;



		if (hi > eps || hn > eps)
		{
			hn = hh[ix + yminus*nx];
			dx = delta / 2.;
			zi = zs[i] - hi;
			zl = zi - dx*(dzsdy[i] - dhdyi);
			zn = zs[ix + yminus*nx] - hn;
			zr = zn + dx*(dzsdy[ix + yminus*nx] - dhdy[ix + yminus*nx]);
			zlr = max(zl, zr);

			hl = hi - dx*dhdyi;
			up = vv[i] - dx*dvdy[i];
			hp = max(0.f, hl + zl - zlr);

			hr = hn + dx*dhdy[ix + yminus*nx];
			um = vv[ix + yminus*nx] + dx*dvdy[ix + yminus*nx];
			hm = max(0.f, hr + zr - zlr);

			//// Reimann solver
			float fh, fu, fv;
			//float dtmaxf = 1.0f / 1e-30f;
			//kurganovf(hm, hp, um, up, delta*cm / fmu, &fh, &fu, &dtmaxf);
			//kurganovf(hm, hp, um, up, delta*cm / fmv, &fh, &fu, &dtmaxf);
			//We can now call one of the approximate Riemann solvers to get the fluxes.
			float cp, cmo, ap, am, qm, qp, a, dlt;
			float epsi = 1e-30f;

			cp = sqrtf(g*hp);
			cmo = sqrtf(g*hm);

			ap = max(up + cp, um + cmo);
			ap = max(ap, 0.0f);

			am = min(up - cp, um - cmo);
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
			fv = (fh > 0.f ? uu[ix + yminus*nx] + dx*dudy[ix + yminus*nx] : uu[i] - dx*dudy[i])*fh;
			/**
			#### Topographic source term

			In the case of adaptive refinement, care must be taken to ensure
			well-balancing at coarse/fine faces (see [notes/balanced.tm]()). */
			float sl = g / 2.0f*(sq(hp) - sq(hl) + (hl + hi)*(zi - zl));
			float sr = g / 2.0f*(sq(hm) - sq(hr) + (hr + hn)*(zn - zr));

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

__global__ void updateEV(int nx, int ny, float delta, float g, float * hh, float *uu, float * vv, float * Fhu, float *Fhv, float * Su, float *Sv, float *Fqux, float *Fquy, float *Fqvx, float *Fqvy, float *dh, float *dhu, float *dhv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
//	int tx = threadIdx.x;
//	int ty = threadIdx.y;
	int  xplus, yplus, xminus, yminus;

	if (ix < nx && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);

		float cm = 1.0f;// 0.1;
		float fmu = 1.0f;
		float fmv = 1.0f;

		float hi = hh[i];
		float uui = uu[i];
		float vvi = vv[i];
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

		dh[i] = -1.0f*(Fhu[xplus + iy*nx] - Fhu[i] + Fhv[ix + yplus*nx] - Fhv[i]) / (cm * delta);
		//printf("%f\t%f\t%f\n", x[i], y[i], dh[i]);


		//double dmdl = (fmu[xplus + iy*nx] - fmu[i]) / (cm * delta);
		//double dmdt = (fmv[ix + yplus*nx] - fmv[i]) / (cm  * delta);
		float dmdl = (fmu - fmu) / (cm * delta);// absurd!
		float dmdt = (fmv - fmv) / (cm  * delta);// absurd!
		float fG = vvi * dmdl - uui * dmdt;
		dhu[i] = (Fqux[i] + Fquy[i] - Su[xplus + iy*nx] - Fquy[ix + yplus*nx]) / (cm*delta);
		dhv[i] = (Fqvy[i] + Fqvx[i] - Sv[ix + yplus*nx] - Fqvx[xplus + iy*nx]) / (cm*delta);
		//dhu.x[] = (Fq.x.x[] + Fq.x.y[] - S.x[1, 0] - Fq.x.y[0, 1]) / (cm[] * Δ);
		dhu[i] += hi * (g*hi / 2.f*dmdl + fG*vvi);
		dhv[i] += hi * (g*hi / 2.f*dmdt - fG*uui);
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
__global__ void resetdtmax(int nx, int ny, float *dtmax)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	if (ix < nx && iy < ny)
	{
		dtmax[i] = 1.0f / 1e-30f;
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

__global__ void reducemin3(float *g_idata, float *g_odata, unsigned int n)
{
	//T *sdata = SharedMemory<T>();
	extern __shared__ float sdata[];
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	float myMin = (i < n) ? g_idata[i] : 1e30f;

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
	float hhi;
	float zsbnd;
	float itx = (iy*1.0f / ny*1.0f) / (1.0f / (1.0f*nybnd - 1.0f));//Bleark!
	zsbnd = tex2D(texLBND, itime, iy*1.0f / ny*1.0f);
	if (ix == 0 && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		hh[i] = zsbnd-zb[i];
		zs[i] = zsbnd;
		uu[i] = -2.0f*(sqrtf(g*max(hh[xplus + iy*nx], 0.0f)) - sqrtf(g*max(zsbnd - zb[xplus + iy*nx], 0.0f))) + uu[xplus + iy*nx];
		vv[i] = 0.0f;
	}
}


__global__ void rightdirichlet(int nx, int ny, int nybnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	int iy = blockIdx.x*blockDim.x + threadIdx.x;
	int ix = nx-1;
	//int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int xminus;
	float hhi;
	float zsbnd;
	float itx = (iy*1.0f / ny*1.0f) / (1.0f / (1.0f*nybnd - 1.0f));//Bleark!
	zsbnd = tex2D(texRBND, itime, iy*1.0f / ny*1.0f);
	if (ix == nx-1 && iy < ny)
	{
		xminus = max(ix - 1, 0);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		uu[i] = +2.0f*(sqrtf(g*max(hh[xminus + iy*nx], 0.0f)) - sqrtf(g*max(zsbnd - zb[xminus + iy*nx], 0.0f))) + uu[xminus + iy*nx];
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
	float hhi;
	float zsbnd;
	float itx = (ix*1.0f / nx*1.0f) / (1.0f / (1.0f*nxbnd - 1.0f));//Bleark!
	zsbnd = tex2D(texTBND, itime, ix*1.0f / nx*1.0f);
	if (iy == ny-1 && ix < nx)
	{
		yminus = max(iy - 1, 0);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		vv[i] = +2.0f*(sqrtf(g*max(hh[ix + yminus*nx], 0.0f)) - sqrtf(g*max(zsbnd - zb[ix + yminus*nx], 0.0f))) + vv[ix + yminus*nx];
		uu[i] = 0.0f;
	}
}
__global__ void botdirichlet(int nx, int ny, int nxbnd, float g, float itime, float *zs, float *zb, float *hh, float *uu, float *vv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = 0;
	//int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	int yplus;
	float hhi;
	float zsbnd;
	float itx = (ix*1.0f / nx*1.0f) / (1.0f / (1.0f*nxbnd - 1.0f));//Bleark!
	zsbnd = tex2D(texBBND, itime, ix*1.0f / nx*1.0f);
	if (iy == 0 && ix < nx)
	{
		yplus = min(iy + 1, ny-1);
		hh[i] = zsbnd - zb[i];
		zs[i] = zsbnd;
		vv[i] = -2.0f*(sqrtf(g*max(hh[ix + yplus*nx], 0.0f)) - sqrtf(g*max(zsbnd - zb[ix + yplus*nx], 0.0f))) + vv[ix + yplus*nx];
		uu[i] = 0.0f;
	}
}

__global__ void quadfriction(int nx, int ny, float dt,float eps, float cf, float *hh, float *uu, float *vv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = blockIdx.y*blockDim.y + threadIdx.y;
	int i = ix + iy*nx;
	
	float normu,hhi,uui,vvi;
	
	if (ix < nx && iy < ny)
	{

		hhi = hh[i];
		uui = uu[i];
		vvi = vv[i];
		if (hhi > eps)
		{
			normu = uui * uui + vvi * vvi;
			float frc = (1.0f + dt*cf*normu / hhi);
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
	float normu, hhi;

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
__global__ void noslipbndLeft(int nx, int ny, float eps, float *zb, float *zs, float *hh, float *uu, float *vv)
{
	int iy = blockIdx.x*blockDim.x + threadIdx.x;
	int ix = 0;
	int i = ix + iy*nx;
	int  xplus, yplus, xminus, yminus;
	float normu, hhi;

	if (ix ==0 && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);

		
		uu[i] = 0.0f;
		zs[i] = zs[xplus + iy*nx];
		hh[i] = max(zs[xplus + iy*nx] - zb[i], eps);
		
		

	}

}
__global__ void noslipbndBot(int nx, int ny, float eps, float *zb, float *zs, float *hh, float *uu, float *vv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = 0;
	int i = ix + iy*nx;
	int  xplus, yplus, xminus, yminus;
	float normu, hhi;

	if (iy == 0 && ix < nx)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);


		vv[i] = 0.0f;
		zs[i] = zs[ix + yplus*nx];
		hh[i] = max(zs[ix + yplus*nx] - zb[i], eps);



	}

}


__global__ void noslipbndRight(int nx, int ny, float eps, float *zb, float *zs, float *hh, float *uu, float *vv)
{
	int iy = blockIdx.x*blockDim.x + threadIdx.x;
	int ix = nx-1;
	int i = ix + iy*nx;
	int  xplus, yplus, xminus, yminus;
	float normu, hhi;

	if (ix == nx-1 && iy < ny)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);


		uu[i] = 0.0f;
		zs[i] = zs[xminus + iy*nx];
		hh[i] = max(zs[xminus + iy*nx] - zb[i], eps);



	}

}
__global__ void noslipbndTop(int nx, int ny, float eps, float *zb, float *zs, float *hh, float *uu, float *vv)
{
	int ix = blockIdx.x*blockDim.x + threadIdx.x;
	int iy = ny - 1;
	int i = ix + iy*nx;
	int  xplus, yplus, xminus, yminus;
	float normu, hhi;

	if (iy == ny - 1 && ix < nx)
	{
		xplus = min(ix + 1, nx - 1);
		xminus = max(ix - 1, 0);
		yplus = min(iy + 1, ny - 1);
		yminus = max(iy - 1, 0);


		vv[i] = 0.0f;
		zs[i] = zs[ix + yminus*nx];
		hh[i] = max(zs[ix + yminus*nx] - zb[i], eps);



	}

}

__global__ void storeTSout(int nx, int ny,int noutnodes, int outnode, int istep, int inode,int jnode, float *zs, float *hh, float *uu, float *vv,float * store)
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



__global__ void addavg_var(int nx, int ny, float * Varmean, float * Var)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	__shared__ float mvari[16][16];
	__shared__ float vari[16][16];

	if (ix < nx && iy < ny)
	{

		mvari[tx][ty] = Varmean[i];
		vari[tx][ty] = Var[i];

		Varmean[i] = mvari[tx][ty] + vari[tx][ty];
	}


}


__global__ void divavg_var(int nx, int ny, float ntdiv, float * Varmean)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	__shared__ float mvari[16][16];
	if (ix < nx && iy < ny)
	{
		mvari[tx][ty] = Varmean[i];
		Varmean[i] = mvari[tx][ty] / ntdiv;
	}


}

__global__ void resetavg_var(int nx, int ny, float * Varmean)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	if (ix < nx && iy < ny)
	{
		Varmean[i] = 0.0f;
	}
}

__global__ void max_var(int nx, int ny, float * Varmax, float * Var)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	__shared__ float mvari[16][16];
	__shared__ float vari[16][16];

	if (ix < nx && iy < ny)
	{

		mvari[tx][ty] = Varmax[i];
		vari[tx][ty] = Var[i];

		Varmax[i] = max(mvari[tx][ty], vari[tx][ty]);
	}


}

__global__ void CalcVorticity(int nx, int ny, float * Vort,float * dvdx, float * dudy)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = ix + iy*nx;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	__shared__ float dvdxi[16][16];
	__shared__ float dudyi[16][16];

	if (ix < nx && iy < ny)
	{

		dvdxi[tx][ty] = dvdx[i];
		dudyi[tx][ty] = dudy[i];

		Vort[i] = dvdxi[tx][ty] - dudyi[tx][ty];
	}


}
